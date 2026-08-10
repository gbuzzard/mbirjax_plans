"""Phase 0, spike 1: the parallel-beam fan kernels, jax vs eager torch.

Question (port_plan.md section 6): what do the eager torch fan kernels cost,
relative to jax, at production shapes -- on CPU, CUDA, and MPS?  The jax side
is the REAL mbirjax public path (``sparse_forward_project`` /
``sparse_back_project``), so the baseline includes the wrappers users get.
The torch side is the view-batched eager formulation a Phase 1 port would
write (per-tap scatter-add forward, per-tap gather back; the portable forms,
no sorted reduce, no stacked gather).

The spike also answers two correctness questions for free:
  * cross-framework value agreement (rel-max diff of forward and back outputs
    -- the first calibration data for the port's golden tolerances), and
  * the rounding-tie diagnostic (how many scatter centers round differently
    across frameworks; port_plan.md section 7, item 2).

Structure: a framework-free orchestrator (numpy + stdlib only) launches one
subprocess per (framework, device, cell) so each measurement owns its process
(honest memory, no allocator crosstalk).  The jax worker runs first per cell:
it times the baseline, and saves the shared inputs + its outputs + scatter
centers to an .npz that the torch workers then load and compare against.

Run (no CLI arguments; edit the CONFIG block below):
    <mbirtorch-env python> p0s1_fan_kernels.py
Results: results/p0s1_<label>.json plus a printed table.  Copy numbers that
drive decisions into plans/torch_port/phases/phase0_findings.md (results/ is
scratch).
"""

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Interpreter per framework (separate conda envs; each worker imports only its
# own framework).  The P0_* environment overrides exist so the CHECKED-IN
# sbatch script (p0_gautschi.sbatch) can point at the cluster envs without
# forking this file; local runs never need them.
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# Sinogram cells (n_views, n_det_rows, n_det_channels) -- the mbirjax_metrics
# sizes (port_plan.md section 4).  The torch device list is per platform: on
# the Mac run cpu+mps; on a cluster GPU node set TORCH_DEVICES = ["cuda"].
CELLS = [(128, 112, 96), (200, 208, 160), (512, 448, 384)]
# Torch devices per cell: CPU runs only the CPU-gated cells (the harness's CPU
# sizes top out at 200; an eager-CPU 512 sweep is long and ungated), while MPS
# runs everything (the informational Mac platform).  On a cluster GPU node set
# this to {"cuda": CELLS}.
TORCH_DEVICE_CELLS = {
    "cpu": [(128, 112, 96), (200, 208, 160)],
    "mps": [(128, 112, 96), (200, 208, 160), (512, 448, 384)],
}
if os.environ.get("P0_TORCH_DEVICES"):        # e.g. "cuda" on a cluster node
    TORCH_DEVICE_CELLS = {d: CELLS for d in os.environ["P0_TORCH_DEVICES"].split(",")}
RUN_JAX = True                     # False to reuse existing per-cell .npz files

# Timing: warmup calls (compile/first-touch) then timed trials per op.
WARMUP = 1
TRIALS = 3

# Torch view-batch sweep per cell (the eager transient is ~(Vb, P, S) floats;
# keep the 512 cell's batches small so MPS fits).
VIEW_BATCHES = {
    (128, 112, 96): [32, 64, 128],
    (200, 208, 160): [32, 64, 128],
    (512, 448, 384): [4, 8, 16],
}

# Save per-view scatter centers for the tie diagnostic only where the npz
# stays small.
SAVE_CENTERS_CELLS = [(128, 112, 96), (200, 208, 160)]

# MPS op-coverage policy: leave the CPU-fallback OFF so an unsupported op
# fails loudly and is recorded as a coverage gap (that is data, not a nuisance).
MPS_FALLBACK = False

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
SEED = 0
# ──────────────────────────────────────────────────────────────────────────────


def cell_tag(cell):
    return "x".join(str(c) for c in cell)


def npz_path(cell):
    return os.path.join(RESULTS_DIR, f"p0s1_ref_{cell_tag(cell)}.npz")


# ══════════════════════════════════════════════════════════════════════════════
# jax worker: the real mbirjax baseline + the shared reference data
# ══════════════════════════════════════════════════════════════════════════════
def jax_worker(cfg):
    import jax
    import mbirjax

    cell = tuple(cfg["cell"])
    n_views, n_rows, n_channels = cell
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(cell, angles)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    indices = np.asarray(
        mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params("use_ror_mask"))
    )
    num_pixels = int(indices.shape[0])
    num_slices = recon_shape[2]

    rng = np.random.RandomState(SEED)
    voxel_values = rng.rand(num_pixels, num_slices).astype(np.float32)

    # Place inputs on device BEFORE timing so trials measure the op, not a
    # host transfer (lessons.md section 5).
    voxel_dev = jax.device_put(voxel_values)
    idx_dev = jax.device_put(indices)

    def timed(fn, *args):
        out = fn(*args)                      # warmup: trace + compile + run
        jax.block_until_ready(out)
        for _ in range(max(0, WARMUP - 1)):
            jax.block_until_ready(fn(*args))
        times = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            out = fn(*args)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return out, times

    sino, fwd_times = timed(model.sparse_forward_project, voxel_dev, idx_dev)
    sino_np = np.asarray(sino).astype(np.float32)
    back, back_times = timed(model.sparse_back_project, sino, idx_dev)
    back_np = np.asarray(back).astype(np.float32)

    # Scatter centers for the tie diagnostic: the same public float chain the
    # kernels consume (compute_channel_coordinate), rounded exactly as the
    # wrappers round.  Optional: a failure here (internal attribute drift)
    # must not discard the timing data above.
    centers = None
    centers_error = None
    if cfg["save_centers"]:
        try:
            import jax.numpy as jnp
            pp = model.projector_functions.projector_params
            one = lambda a: jnp.round(
                mbirjax.ParallelBeamModel.compute_channel_coordinate(idx_dev, a, pp)
            ).astype(jnp.int32)
            centers = np.asarray(jax.vmap(one)(model.get_params("angles")))
        except Exception as e:                              # noqa: BLE001
            centers_error = f"{type(e).__name__}: {e}"

    save = dict(
        angles=angles, indices=indices, voxel_values=voxel_values,
        sinogram=sino_np, back=back_np,
        recon_shape=np.array(recon_shape),
        psf_radius=np.array(model.get_psf_radius()),
    )
    if centers is not None:
        save["centers"] = centers
    np.savez_compressed(cfg["npz"], **save)

    return dict(
        framework="jax", jax_version=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        cell=list(cell), recon_shape=list(recon_shape), num_pixels=num_pixels,
        psf_radius=int(model.get_psf_radius()),
        fwd_times=fwd_times, back_times=back_times,
        centers_error=centers_error,
    )


# ══════════════════════════════════════════════════════════════════════════════
# torch worker: the view-batched eager fan kernels
# ══════════════════════════════════════════════════════════════════════════════
def torch_worker(cfg):
    import torch

    device = cfg["device"]
    ref = np.load(cfg["npz"])
    recon_shape = tuple(int(x) for x in ref["recon_shape"])
    psf_radius = int(ref["psf_radius"])
    n_views, n_rows, n_channels = tuple(cfg["cell"])
    num_slices = recon_shape[2]

    dev = torch.device(device)
    f32 = torch.float32
    angles = torch.tensor(np.asarray(ref["angles"]), dtype=f32, device=dev)
    indices = torch.tensor(np.asarray(ref["indices"]), dtype=torch.int64, device=dev)
    values = torch.tensor(np.asarray(ref["voxel_values"]), dtype=f32, device=dev)
    sino_in = torch.tensor(np.asarray(ref["sinogram"]), dtype=f32, device=dev)
    num_pixels = indices.shape[0]

    # Geometry constants: the mbirjax defaults (delta_det_channel = delta_voxel
    # = 1, aspect 1, offset 0), so values are comparable across frameworks.
    nr, nc = recon_shape[0], recon_shape[1]
    det_center = (n_channels - 1) / 2.0

    def geometry(angle_batch):
        """Vectorized compute_proj_data for a batch of views.

        Returns n_p (Vb, P) float, centers (Vb, P) int64, and the per-view
        scalars W (Vb, 1), weight_scale (Vb, 1), L_max (Vb, 1).  Computed in
        f32 to mirror the jax chain.
        """
        rows = (indices // nc).to(f32)
        cols = (indices % nc).to(f32)
        y_t = rows - (nr - 1) / 2.0
        x_t = cols - (nc - 1) / 2.0
        cos = torch.cos(angle_batch)[:, None]
        sin = torch.sin(angle_batch)[:, None]
        x = cos * x_t[None, :] - sin * y_t[None, :]
        n_p = x + det_center
        footprint = torch.maximum(cos.abs(), sin.abs())
        W = footprint            # / delta_det_channel == 1
        weight_scale = 1.0 / footprint
        L_max = torch.clamp(W, max=1.0)
        centers = torch.round(n_p).to(torch.int64)
        return n_p, centers, W, weight_scale, L_max

    def tap_weights(n_p, n, W, weight_scale, L_max):
        """The shared trapezoid weight for tap n: zero outside the detector,
        indices clipped into range (torch index ops assert on out-of-bounds,
        unlike jax's drop/clamp semantics -- port_plan.md section 7, item 1)."""
        A = torch.clamp((W + 1.0) / 2.0 - (n_p - n.to(f32)).abs(), min=0.0)
        A = torch.minimum(A, L_max) * weight_scale
        A = A * ((n >= 0) & (n < n_channels)).to(f32)
        return A, n.clamp(0, n_channels - 1)

    def forward(view_batch):
        """Eager forward: per-tap scatter-add into (Vb*C, S), views batched."""
        out = torch.empty((n_views, n_rows, n_channels), dtype=f32, device=dev)
        for v0 in range(0, n_views, view_batch):
            ab = angles[v0:v0 + view_batch]
            vb = ab.shape[0]
            n_p, centers, W, ws, L_max = geometry(ab)
            acc = torch.zeros((vb * n_channels, num_slices), dtype=f32, device=dev)
            row_base = (torch.arange(vb, device=dev)[:, None] * n_channels)
            for off in range(-psf_radius, psf_radius + 1):
                A, n = tap_weights(n_p, centers + off, W, ws, L_max)
                idx = (row_base + n).reshape(-1)
                src = (A.unsqueeze(-1) * values).reshape(-1, num_slices)
                acc.index_add_(0, idx, src)
            # (Vb, C, S) channel-major -> the sinogram's (Vb, R, C) layout.
            out[v0:v0 + vb] = acc.view(vb, n_channels, num_slices).permute(0, 2, 1)
        return out

    def back(view_batch):
        """Eager back: per-tap gather from channel-major views, summed over
        views (the adjoint of forward, matching horizontal_fan_back)."""
        out = torch.zeros((num_pixels, n_rows), dtype=f32, device=dev)
        for v0 in range(0, n_views, view_batch):
            ab = angles[v0:v0 + view_batch]
            vb = ab.shape[0]
            n_p, centers, W, ws, L_max = geometry(ab)
            sT = sino_in[v0:v0 + vb].permute(0, 2, 1).contiguous()   # (Vb, C, R)
            v_idx = torch.arange(vb, device=dev)[:, None]
            for off in range(-psf_radius, psf_radius + 1):
                A, n = tap_weights(n_p, centers + off, W, ws, L_max)
                g = sT[v_idx, n]                                     # (Vb, P, R)
                out += torch.einsum("vp,vpr->pr", A, g)
        return out

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    def peak_mem_reset():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_mem_read():
        if device == "cuda":
            return int(torch.cuda.max_memory_allocated())
        if device == "mps":
            return int(getattr(torch.mps, "current_allocated_memory", lambda: 0)())
        return None

    results = dict(
        framework="torch", torch_version=torch.__version__, device=device,
        device_name=(torch.cuda.get_device_name(0) if device == "cuda"
                     else platform.processor() or platform.machine()),
        num_threads=torch.get_num_threads(),
        cell=list(cfg["cell"]), num_pixels=int(num_pixels),
        sweeps=[],
    )

    with torch.inference_mode():
        for op_name, op in (("fwd", forward), ("back", back)):
            for vbatch in cfg["view_batches"]:
                fn = lambda: op(vbatch)
                out = fn(); sync()                       # warmup / first-touch
                for _ in range(max(0, WARMUP - 1)):
                    fn(); sync()
                peak_mem_reset()
                times = []
                for _ in range(TRIALS):
                    sync(); t0 = time.perf_counter()
                    out = fn(); sync()
                    times.append(time.perf_counter() - t0)
                entry = dict(op=op_name, view_batch=vbatch, times=times,
                             peak_bytes=peak_mem_read())
                results["sweeps"].append(entry)
            # Value check once per op (independent of view batch).
            out_np = out.cpu().numpy()
            ref_out = np.asarray(ref["sinogram"] if op_name == "fwd" else ref["back"])
            rel_max = float(np.max(np.abs(out_np - ref_out)) / np.max(np.abs(ref_out)))
            results[f"{op_name}_rel_max_vs_jax"] = rel_max

        # Rounding-tie diagnostic: count centers that differ from jax's.
        if "centers" in ref.files:
            n_p, centers, _, _, _ = geometry(angles)
            jax_centers = torch.tensor(np.asarray(ref["centers"]).astype(np.int64),
                                       device=dev)
            mism = int((centers != jax_centers).sum().item())
            results["center_mismatch_count"] = mism
            results["center_total"] = int(centers.numel())

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════════════
def run_worker(python, role, cfg):
    """Launch one measurement in its own process; return its result dict."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_{role}_{cfg.get('device','')}.json")
    out_path = cfg_path.replace("_cfg_", "_out_")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    env = dict(os.environ)
    if role == "torch" and cfg.get("device") == "mps" and MPS_FALLBACK:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker", role,
                           cfg_path, out_path], env=env)
    if proc.returncode != 0:
        return dict(error=f"worker exited {proc.returncode}", role=role, cfg=cfg)
    with open(out_path) as f:
        return json.load(f)


def main():
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       warmup=WARMUP, trials=TRIALS, cells=[])
    for cell in CELLS:
        print(f"\n=== cell {cell_tag(cell)} ===", flush=True)
        cell_res = dict(cell=list(cell), workers=[])
        cfg_common = dict(cell=list(cell), npz=npz_path(cell),
                          save_centers=(tuple(cell) in map(tuple, SAVE_CENTERS_CELLS)))
        if RUN_JAX:
            print("jax worker ...", flush=True)
            r = run_worker(JAX_PYTHON, "jax", dict(cfg_common))
            cell_res["workers"].append(r)
            if "error" in r:
                print("  jax worker FAILED; skipping cell", r, flush=True)
                all_results["cells"].append(cell_res)
                continue
            print(f"  fwd {min(r['fwd_times']):.3f}s back {min(r['back_times']):.3f}s "
                  f"(min of {TRIALS}); recon_shape={r['recon_shape']} "
                  f"pixels={r['num_pixels']}", flush=True)
        for device, dev_cells in TORCH_DEVICE_CELLS.items():
            if tuple(cell) not in map(tuple, dev_cells):
                continue
            print(f"torch worker ({device}) ...", flush=True)
            cfg = dict(cfg_common, device=device,
                       view_batches=VIEW_BATCHES[tuple(cell)])
            r = run_worker(TORCH_PYTHON, "torch", cfg)
            cell_res["workers"].append(r)
            if "error" not in r:
                for op in ("fwd", "back"):
                    rows = [s for s in r["sweeps"] if s["op"] == op]
                    best = min(rows, key=lambda s: min(s["times"]))
                    print(f"  {op}: best {min(best['times']):.3f}s "
                          f"@ view_batch {best['view_batch']}; "
                          f"rel_max_vs_jax {r.get(op + '_rel_max_vs_jax'):.2e}",
                          flush=True)
                if "center_mismatch_count" in r:
                    print(f"  center mismatches: {r['center_mismatch_count']} "
                          f"/ {r['center_total']}", flush=True)
        all_results["cells"].append(cell_res)

    out = os.path.join(RESULTS_DIR, f"p0s1_summary_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        role, cfg_path, out_path = sys.argv[2], sys.argv[3], sys.argv[4]
        with open(cfg_path) as f:
            cfg = json.load(f)
        result = jax_worker(cfg) if role == "jax" else torch_worker(cfg)
        with open(out_path, "w") as f:
            json.dump(result, f)
    else:
        main()
