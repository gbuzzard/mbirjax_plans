"""Phase 2: the first gate-cell readout at n=1 (port_plan.md sections 4-5).

Measures the harness op set -- direct_filter, forward, back, and a
3-iteration nonconstant-weights VCD -- on the mbirjax_metrics cells, for BOTH
frameworks with ONE ruler: each (framework, device, cell, op) runs in its own
subprocess, warm-timed (min of trials after a warmup call that also pays any
compile cost), with honest per-process peak memory (ru_maxrss everywhere;
plus cuda max_memory_allocated / jax peak_bytes_in_use on CUDA).

Inputs are deterministic and identical across frameworks: the Shepp-Logan
phantom at the cell's recon shape (the generators match exactly; Phase 1
findings), its forward projection as the sinogram, transmission-root weights,
and np.random.seed(13) for the VCD partitions.

The printed table reports torch/jax ratios against the gates (<= 2x time,
~1.5x memory).  Run (no CLI args; edit CONFIG):
    <mbirtorch-env python> p2_gate_readout.py
On a cluster GPU node set P0_TORCH_DEVICES=cuda and the P0_*_PYTHON paths
(the same overrides as the Phase 0 spikes).
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# (cell, ops) per device.  CPU runs the harness CPU cells; MPS adds the 512
# cell (informational platform); CUDA (via P0_TORCH_DEVICES=cuda) runs the GPU
# cells including the 1024 capacity probe at trials=1.
CPU_CELLS = [(128, 112, 96), (129, 113, 97), (200, 208, 160)]
GPU_CELLS = [(200, 208, 160), (512, 448, 384), (513, 449, 385), (1024, 1008, 992)]
OPS = ["direct_filter", "forward", "back", "vcd"]

if os.environ.get("P0_TORCH_DEVICES") == "cuda":
    PLAN = [("jax", "cuda", c) for c in GPU_CELLS] + \
           [("torch", "cuda", c) for c in GPU_CELLS]
else:
    PLAN = ([("jax", "cpu", c) for c in CPU_CELLS]
            + [("torch", "cpu", c) for c in CPU_CELLS]
            + [("torch", "mps", c) for c in CPU_CELLS + [(512, 448, 384)]])

WARMUP = 1
TRIALS = 3
SINGLE_TRIAL_CELLS = [(1024, 1008, 992)]   # capacity probes: one trial, all ops
VCD_ITERATIONS = 3
VCD_SEED = 13
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def worker(cfg):
    framework, device, cell = cfg["framework"], cfg["device"], tuple(cfg["cell"])
    import numpy as np

    n_views = cell[0]
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    trials = 1 if cell in map(tuple, SINGLE_TRIAL_CELLS) else TRIALS

    if framework == "jax":
        import jax
        import mbirjax
        model = mbirjax.ParallelBeamModel(cell, angles)
        recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = model.forward_project(phantom)
        weights = np.exp(-np.asarray(sinogram) / (2 * np.max(sinogram)))
        sino_dev = jax.device_put(np.asarray(sinogram))
        w_dev = jax.device_put(weights)
        model.set_params(no_warning=True, verbose=0)

        def sync(x):
            jax.block_until_ready(x)

        ops = {
            "direct_filter": lambda: model.direct_filter(sino_dev, output_sharded=True),
            "forward": lambda: model.forward_project(jax.device_put(phantom),
                                                     output_sharded=True),
            "back": lambda: model.back_project(sino_dev, output_sharded=True),
        }

        def vcd():
            np.random.seed(VCD_SEED)
            recon, _ = model.recon(np.asarray(sinogram), weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
            return recon

        def gpu_peak():
            if device == "cuda":
                return int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', 0))
            return None
        version = f"jax {jax.__version__}"
    else:
        import torch
        import mbirtorch
        model = mbirtorch.ParallelBeamModel(cell, angles)
        model.configure_devices(devices=[device])
        recon_shape = tuple(model.get_params('recon_shape'))
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = model.forward_project(phantom)
        weights = np.exp(-np.asarray(sinogram) / (2 * np.max(sinogram)))
        sino_dev = torch.as_tensor(sinogram, device=model.torch_device)
        phantom_dev = torch.as_tensor(phantom, device=model.torch_device)
        model.set_params(no_warning=True, verbose=0)

        def sync(x):
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()

        ops = {
            "direct_filter": lambda: model.direct_filter(sino_dev, output_sharded=True),
            "forward": lambda: model.forward_project(phantom_dev, output_sharded=True),
            "back": lambda: model.back_project(sino_dev, output_sharded=True),
        }

        def vcd():
            np.random.seed(VCD_SEED)
            recon, _ = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
            return recon

        def gpu_peak():
            if device == "cuda":
                return int(torch.cuda.max_memory_allocated())
            return None
        version = f"torch {torch.__version__}"

    result = dict(framework=framework, device=device, cell=list(cell),
                  recon_shape=list(recon_shape), version=version, op_times={},
                  trials=trials)
    op_name = cfg["op"]
    if op_name == "vcd":
        # The first run pays each framework's per-process compile/trace cost
        # (inductor for torch -- amortized across processes by the persistent
        # cache once warm -- and any uncached jax compiles); the SECOND run is
        # the op cost.  Both are recorded for BOTH frameworks at every cell
        # (the warm-vcd protocol; the earlier readout's missing warm entries
        # at >=512 made those rows cold+compile).
        t0 = time.perf_counter()
        out = vcd()
        result["op_times"]["vcd"] = [time.perf_counter() - t0]
        t0 = time.perf_counter()
        out = vcd()
        result["op_times"]["vcd_warm"] = [time.perf_counter() - t0]
    else:
        fn = ops[op_name]
        out = fn()
        sync(out)
        for _ in range(max(0, WARMUP - 1)):
            sync(fn())
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            out = fn()
            sync(out)
            times.append(time.perf_counter() - t0)
        result["op_times"][op_name] = times
    del out

    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["gpu_peak_bytes"] = gpu_peak()
    return result


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p2.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p2.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path])
    if proc.returncode != 0:
        return dict(error=f"worker exited {proc.returncode}", **cfg)
    with open(out_path) as f:
        return json.load(f)


def main():
    all_results = dict(run_label=RUN_LABEL, host=platform.node(), rows=[])
    for framework, device, cell in PLAN:
        python = JAX_PYTHON if framework == "jax" else TORCH_PYTHON
        for op in OPS:
            if op == "vcd" and device == "mps" and cell[0] >= 512:
                continue          # informational platform; keep the run short
            label = f"{framework}/{device}/{'x'.join(map(str, cell))}/{op}"
            print(f"{label} ...", flush=True)
            r = run_one(python, dict(framework=framework, device=device,
                                     cell=list(cell), op=op))
            all_results["rows"].append(r)
            if "error" in r:
                print(f"  FAILED: {r['error']}", flush=True)
            else:
                times = list(r["op_times"].values())[0]
                extra = (f"  warm {min(r['op_times']['vcd_warm']):.3f}s"
                         if "vcd_warm" in r["op_times"] else "")
                mem = r["peak_rss_bytes"] / 2**30
                gm = (f" gpu {r['gpu_peak_bytes']/2**30:.2f}G"
                      if r.get("gpu_peak_bytes") else "")
                print(f"  {min(times):.3f}s{extra}  rss {mem:.2f}G{gm}", flush=True)

    out = os.path.join(RESULTS_DIR, f"p2_gate_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        with open(sys.argv[2]) as f:
            cfg = json.load(f)
        result = worker(cfg)
        with open(sys.argv[3], "w") as f:
            json.dump(result, f)
    else:
        main()
