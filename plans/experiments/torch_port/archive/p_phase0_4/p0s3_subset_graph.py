"""Phase 0, spike 3: a VCD-subset-shaped update loop, eager vs CUDA graph.

Question (port_plan.md section 6): can the per-subset update run with
near-zero host cost under CUDA graph capture?  The jax attribution showed VCD
at interactive sizes to be ~95% host dispatch, and the subset update has the
two properties graph capture requires: fixed shapes per partition granularity
and no host syncs in the line search.  This spike measures the graph's win
directly on a MOCK subset update.

The mock reproduces the update's structure and op mix, not the full engine:
per subset it computes scatter centers for all views, back-projects the
weighted error onto the subset (per-tap gather), forms the preconditioned
delta, forward-projects the delta (per-tap scatter), runs the alpha line
search entirely on device, and applies the in-place recon and error-sinogram
updates.  The qGGMRF prior term is replaced by a scalar; the projector work
dominates the op count.  Subset indices CHANGE per subset: they are staged in
a device-side pool and copied into a static input buffer between graph
replays, which is the same mechanism a real port would use.

CUDA only.  Without CUDA the script runs one eager smoke pass on CPU to
validate shapes (SMOKE_ON_CPU) and exits, so it can be sanity-checked on the
Mac before a cluster run.

Run (no CLI arguments; edit the CONFIG block):
    <mbirtorch-env python> p0s3_subset_graph.py
"""

import json
import os
import platform
import time

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
# (cell, subset sizes): the 200-class and 512-class parallel cells, each at a
# coarse and a fine granularity (P_full/16 and P_full/64, rounded).
CASES = [
    dict(tag="200c", V=200, R=208, C=160, S=208, NR=160, NC=160,
         subset_sizes=[1240, 310]),
    dict(tag="512c", V=512, R=448, C=384, S=448, NR=384, NC=384,
         subset_sizes=[7240, 1810]),
]
K_SUBSETS = 32                     # subsets per timed pass
WARMUP_PASSES = 1
TRIALS = 3
PSF_RADIUS = 1
SMOKE_ON_CPU = True                # no CUDA -> one CPU eager pass, then exit
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def build_case(torch, case, num_pixels, dev):
    """Static tensors + the subset-step closure for one (cell, subset size)."""
    f32 = torch.float32
    V, R, C, S = case["V"], case["R"], case["C"], case["S"]
    NR, NC = case["NR"], case["NC"]
    g = torch.Generator(device="cpu").manual_seed(SEED)

    angles = torch.linspace(0, float(np.pi), V + 1, device=dev)[:-1]
    flat_recon = torch.rand((NR * NC, S), generator=g).to(dev)
    fm_hessian = torch.rand((NR * NC, S), generator=g).to(dev) + 0.5
    error_sino = torch.rand((V, R, C), generator=g).to(dev)
    error_sino_T = error_sino.permute(0, 2, 1).contiguous()      # (V, C, R)
    prior_hess = torch.tensor(2.0, device=dev)
    fm_c = torch.tensor(1.7, device=dev)

    # The K subsets' indices, staged in a device pool; idx_buf is the static
    # graph input the pool rows are copied into.
    pool_np = np.stack([np.random.RandomState(SEED + k).choice(
        NR * NC, size=num_pixels, replace=False) for k in range(K_SUBSETS)])
    idx_pool = torch.tensor(pool_np, dtype=torch.int64, device=dev)
    idx_buf = idx_pool[0].clone()

    det_center = (C - 1) / 2.0

    def subset_step():
        """One subset update; reads idx_buf, mutates flat_recon/error_sino_T."""
        rows = (idx_buf // NC).to(f32) - (NR - 1) / 2.0
        cols = (idx_buf % NC).to(f32) - (NC - 1) / 2.0
        cos = torch.cos(angles)[:, None]
        sin = torch.sin(angles)[:, None]
        n_p = cos * cols[None, :] - sin * rows[None, :] + det_center
        footprint = torch.maximum(cos.abs(), sin.abs())
        W, ws = footprint, 1.0 / footprint
        L_max = torch.clamp(W, max=1.0)
        centers = torch.round(n_p).to(torch.int64)
        v_idx = torch.arange(V, device=dev)[:, None]

        # Back-project the (weighted) error onto the subset: per-tap gather.
        grad = torch.zeros((num_pixels, R), dtype=f32, device=dev)
        for off in range(-PSF_RADIUS, PSF_RADIUS + 1):
            n = centers + off
            A = torch.clamp((W + 1.0) / 2.0 - (n_p - n.to(f32)).abs(), min=0.0)
            A = torch.minimum(A, L_max) * ws
            A = A * ((n >= 0) & (n < C)).to(f32)
            grad += torch.einsum("vp,vpr->pr", A, error_sino_T[v_idx, n.clamp(0, C - 1)])
        grad = -fm_c * grad

        # Preconditioned direction from the gathered Hessian.
        hess = fm_c * fm_hessian[idx_buf]
        delta = -grad / (hess + prior_hess)

        # Forward-project the direction: per-tap scatter into (V*C, S).
        acc = torch.zeros((V * C, S), dtype=f32, device=dev)
        row_base = v_idx * C
        for off in range(-PSF_RADIUS, PSF_RADIUS + 1):
            n = centers + off
            A = torch.clamp((W + 1.0) / 2.0 - (n_p - n.to(f32)).abs(), min=0.0)
            A = torch.minimum(A, L_max) * ws
            A = A * ((n >= 0) & (n < C)).to(f32)
            idx = (row_base + n.clamp(0, C - 1)).reshape(-1)
            acc.index_add_(0, idx, (A.unsqueeze(-1) * delta).reshape(-1, S))
        delta_sino_T = acc.view(V, C, S)                       # channel-major

        # Alpha line search, entirely on device (no .item()).
        fl = (error_sino_T * delta_sino_T).sum()
        fq = fm_c * (delta_sino_T * delta_sino_T).sum()
        pq = (prior_hess * delta * delta).sum()
        alpha = (fl / (fq + pq + 1e-8)).clamp(0.0, 2.0)

        # In-place state updates (the donation-free torch idiom).
        flat_recon.index_add_(0, idx_buf, alpha * delta)
        error_sino_T.sub_(alpha * delta_sino_T)

    return subset_step, idx_buf, idx_pool


def run_case(torch, case, num_pixels, dev, mode):
    subset_step, idx_buf, idx_pool = build_case(torch, case, num_pixels, dev)

    if mode == "graph":
        # Standard capture recipe: warm up on a side stream, then capture one
        # step; replay with fresh indices copied into the static buffer.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                subset_step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            subset_step()

        def one_pass():
            for k in range(K_SUBSETS):
                idx_buf.copy_(idx_pool[k])
                graph.replay()
    else:
        def one_pass():
            for k in range(K_SUBSETS):
                idx_buf.copy_(idx_pool[k])
                subset_step()

    for _ in range(WARMUP_PASSES):
        one_pass()
    torch.cuda.synchronize()
    times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        one_pass()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def main():
    import torch

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not torch.cuda.is_available():
        if not SMOKE_ON_CPU:
            print("CUDA unavailable and SMOKE_ON_CPU off; nothing to do.")
            return
        print("CUDA unavailable: running one CPU eager smoke pass (shapes only).",
              flush=True)
        case = CASES[0]
        step, _, _ = build_case(torch, case, case["subset_sizes"][1],
                                torch.device("cpu"))
        with torch.inference_mode():
            step()
        print("smoke pass OK", flush=True)
        return

    dev = torch.device("cuda")
    results = dict(run_label=RUN_LABEL, host=platform.node(),
                   torch_version=torch.__version__,
                   device_name=torch.cuda.get_device_name(0),
                   k_subsets=K_SUBSETS, trials=TRIALS, runs=[])
    with torch.inference_mode():
        for case in CASES:
            for num_pixels in case["subset_sizes"]:
                for mode in ("eager", "graph"):
                    times = run_case(torch, case, num_pixels, dev, mode)
                    per_subset_ms = 1000 * min(times) / K_SUBSETS
                    results["runs"].append(dict(
                        tag=case["tag"], num_pixels=num_pixels, mode=mode,
                        times=times, per_subset_ms=per_subset_ms))
                    print(f"{case['tag']} P={num_pixels} {mode}: "
                          f"{per_subset_ms:.2f} ms/subset", flush=True)

    out = os.path.join(RESULTS_DIR, f"p0s3_summary_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
