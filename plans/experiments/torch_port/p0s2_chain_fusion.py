"""Phase 0, spike 2: eager vs torch.compile on the port's two hot chains.

Question (port_plan.md section 6): what does compilation buy, in time AND peak
memory, on (a) the fan weight chain + scatter (the forward projector's body)
and (b) a qGGMRF-shaped neighbor chain at volume scale?  Chain (b) is the
port's named memory attention point (port_plan.md section 3): eager holds
two-to-three live volume buffers per step where a fused chain holds one or
two.

The qGGMRF chain here is a SURROGATE: it reproduces the op mix (shift,
subtract, abs, divide, power, clip, multiply-accumulate over 6 neighbors),
not the exact b_tilde formulas -- fusion behavior depends on the op mix, not
the constants.  The fan chain is the spike-1 formulation at a single
view-batch call.

Structure: a torch-free orchestrator launches one subprocess per (device,
chain, size, mode).  Each subprocess owns its process, so CPU peak RSS
(ru_maxrss) is an honest per-config ruler alongside CUDA
max_memory_allocated / MPS current_allocated_memory.

Run (no CLI arguments; edit the CONFIG block):
    <mbirtorch-env python> p0s2_chain_fusion.py
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
# P0_* environment overrides: for the checked-in cluster sbatch script only
# (see p0s1_fan_kernels.py); local runs use the defaults.
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")
DEVICES = (os.environ.get("P0_TORCH_DEVICES").split(",")
           if os.environ.get("P0_TORCH_DEVICES") else ["cpu", "mps"])
MODES = ["eager", "compile"]

# Fan-chain shapes: one view-batch call of the spike-1 forward at the
# 200-class and 512-class cells (P ~ the ROR pixel counts, rounded).
FAN_SHAPES = [
    dict(tag="200c", vb=64, P=20000, S=208, C=160),
    dict(tag="512c", vb=8, P=116000, S=448, C=384),
]
# qGGMRF-surrogate volumes: the 200-class and 512-class recon shapes.
QGGMRF_SHAPES = [
    dict(tag="200c", shape=(160, 160, 208)),
    dict(tag="512c", shape=(384, 384, 448)),
]

WARMUP = 2                          # 2: the first compiled call pays compilation
TRIALS = 3
PSF_RADIUS = 1
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def torch_worker(cfg):
    import numpy as np
    import torch

    device = cfg["device"]
    dev = torch.device(device)
    f32 = torch.float32
    torch.manual_seed(SEED)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    # ── the two chains ────────────────────────────────────────────────────────
    if cfg["chain"] == "fan":
        p = cfg["fan"]
        vb, P, S, C = p["vb"], p["P"], p["S"], p["C"]
        angles = torch.linspace(0.1, 3.0, vb, device=dev)
        x_t = torch.rand(P, device=dev) * C - C / 2
        y_t = torch.rand(P, device=dev) * C - C / 2
        values = torch.rand(P, S, device=dev)

        def chain():
            # The spike-1 forward body: geometry + trapezoid taps + scatter.
            cos = torch.cos(angles)[:, None]
            sin = torch.sin(angles)[:, None]
            n_p = cos * x_t[None, :] - sin * y_t[None, :] + (C - 1) / 2.0
            footprint = torch.maximum(cos.abs(), sin.abs())
            W = footprint
            ws = 1.0 / footprint
            L_max = torch.clamp(W, max=1.0)
            centers = torch.round(n_p).to(torch.int64)
            acc = torch.zeros((vb * C, S), dtype=f32, device=dev)
            row_base = torch.arange(vb, device=dev)[:, None] * C
            for off in range(-PSF_RADIUS, PSF_RADIUS + 1):
                n = centers + off
                A = torch.clamp((W + 1.0) / 2.0 - (n_p - n.to(f32)).abs(), min=0.0)
                A = torch.minimum(A, L_max) * ws
                A = A * ((n >= 0) & (n < C)).to(f32)
                idx = (row_base + n.clamp(0, C - 1)).reshape(-1)
                src = (A.unsqueeze(-1) * values).reshape(-1, S)
                acc.index_add_(0, idx, src)
            return acc

    else:
        shape = tuple(cfg["qggmrf"]["shape"])
        x = torch.rand(shape, device=dev)
        sigma = 0.3
        p_pow, q_pow, T = 2.0, 1.2, 1.0
        shifts = [(1, 0), (-1, 0), (1, 1), (-1, 1), (1, 2), (-1, 2)]

        def chain():
            # Surrogate qGGMRF gradient+Hessian accumulation: per neighbor,
            # a difference and a rational influence chain, accumulated.
            grad = torch.zeros_like(x)
            hess = torch.zeros_like(x)
            for amount, dim in shifts:
                delta = x - torch.roll(x, amount, dims=dim)
                ad = delta.abs() / sigma
                num = ad.clamp(min=1e-6) ** (q_pow - p_pow)
                ratio = num / (1.0 + num)
                btil = (ratio * (q_pow / p_pow) + (1.0 - ratio)).clamp(min=1e-6)
                grad = grad + btil * delta / (sigma * sigma)
                hess = hess + btil / (sigma * sigma)
            return grad, hess

    # ── mode: eager or compiled ───────────────────────────────────────────────
    compile_error = None
    fn = chain
    if cfg["mode"] == "compile":
        try:
            fn = torch.compile(chain)
            fn()                                   # trigger compilation now
            sync()
        except Exception as e:                     # noqa: BLE001
            compile_error = f"{type(e).__name__}: {e}"[:500]

    result = dict(framework="torch", torch_version=torch.__version__,
                  device=device, num_threads=torch.get_num_threads(),
                  chain=cfg["chain"], tag=cfg.get("tag"), mode=cfg["mode"],
                  compile_error=compile_error)
    if compile_error is not None:
        return result

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(WARMUP + TRIALS):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        if i >= WARMUP:
            times.append(time.perf_counter() - t0)
    del out

    peak = dict(ru_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if device == "cuda":
        peak["cuda_max_alloc"] = int(torch.cuda.max_memory_allocated())
    if device == "mps":
        peak["mps_current_alloc"] = int(
            getattr(torch.mps, "current_allocated_memory", lambda: 0)())
    result.update(times=times, peak=peak)
    return result


def run_worker(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p0s2.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p0s2.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    proc = subprocess.run([TORCH_PYTHON, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path])
    if proc.returncode != 0:
        return dict(error=f"worker exited {proc.returncode}", cfg=cfg)
    with open(out_path) as f:
        return json.load(f)


def main():
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       warmup=WARMUP, trials=TRIALS, runs=[])
    jobs = []
    for device in DEVICES:
        for fan in FAN_SHAPES:
            for mode in MODES:
                jobs.append(dict(device=device, chain="fan", tag=fan["tag"],
                                 fan=fan, mode=mode))
        for q in QGGMRF_SHAPES:
            for mode in MODES:
                jobs.append(dict(device=device, chain="qggmrf", tag=q["tag"],
                                 qggmrf=q, mode=mode))
    for cfg in jobs:
        label = f"{cfg['device']}/{cfg['chain']}/{cfg['tag']}/{cfg['mode']}"
        print(f"{label} ...", flush=True)
        r = run_worker(cfg)
        all_results["runs"].append(r)
        if "error" in r:
            print(f"  FAILED: {r['error']}", flush=True)
        elif r.get("compile_error"):
            print(f"  compile unavailable: {r['compile_error'][:120]}", flush=True)
        else:
            rss = r["peak"]["ru_maxrss_bytes"] / 2**30
            print(f"  min {min(r['times']):.3f}s  peak_rss {rss:.2f} GiB "
                  + (f" cuda_alloc {r['peak']['cuda_max_alloc']/2**30:.2f} GiB"
                     if "cuda_max_alloc" in r["peak"] else "")
                  + (f" mps_alloc {r['peak']['mps_current_alloc']/2**30:.2f} GiB"
                     if "mps_current_alloc" in r["peak"] else ""), flush=True)

    out = os.path.join(RESULTS_DIR, f"p0s2_summary_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        with open(sys.argv[2]) as f:
            cfg = json.load(f)
        result = torch_worker(cfg)
        with open(sys.argv[3], "w") as f:
            json.dump(result, f)
    else:
        main()
