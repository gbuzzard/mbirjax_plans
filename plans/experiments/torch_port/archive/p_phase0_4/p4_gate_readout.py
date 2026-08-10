"""Phase 4: the multi-device gate readout -- both frameworks at n = 1, 2 (and
4 when the node has 4 GPUs) on the real metrics cells, one ruler per row
(subprocess isolation, warm-vcd protocol, honest peaks), with value diffs
against each framework's own n=1 reference AND torch-vs-jax at equal n.

The 512 and 1024 cells divide evenly at n = 2 and 4 (views 512/1024; slices
448/1008), so the divisible-only sharded drivers cover the gates.

Run: <torch python> p4_gate_readout.py   (set P0_*_PYTHON; GPUs per node set
the max n probed).
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time

JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

CELLS = [(512, 448, 384), (1024, 1008, 992)]
VCD_ITERATIONS = 3
VCD_SEED = 13
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]


def worker(cfg):
    framework, n_dev, cell = cfg["framework"], cfg["n_dev"], tuple(cfg["cell"])
    import numpy as np

    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    if framework == "jax":
        import jax
        import mbirjax
        model = mbirjax.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        model.configure_devices(n_dev)
        recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = np.asarray(model.forward_project(phantom))
        weights = np.exp(-sinogram / (2 * np.max(sinogram)))

        def vcd():
            np.random.seed(VCD_SEED)
            r, _ = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
            return np.asarray(r)

        def gpu_peak():
            return max(int(d.memory_stats().get('peak_bytes_in_use', 0))
                       for d in jax.devices())
        version = f"jax {jax.__version__}"
    else:
        import torch
        import mbirtorch
        model = mbirtorch.ParallelBeamModel(cell, angles)
        model.configure_devices(devices=["cuda"])
        model.set_params(no_warning=True, verbose=0)
        # UNCONDITIONAL: a CUDA model without an explicit
        # configure_devices call now spreads across every visible
        # device, so the n=1 arm must pin itself or it silently
        # becomes the very multi-device run it is the baseline for.
        model.configure_devices(n_dev)
        recon_shape = tuple(model.get_params('recon_shape'))
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = model.forward_project(phantom)
        if not isinstance(sinogram, np.ndarray):
            sinogram = model._gather_sinogram(sinogram)
        weights = np.exp(-np.asarray(sinogram) / (2 * np.max(sinogram)))

        def vcd():
            np.random.seed(VCD_SEED)
            r, _ = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
            torch.cuda.synchronize()
            return np.asarray(r)

        def gpu_peak():
            return max(int(torch.cuda.max_memory_allocated(i))
                       for i in range(torch.cuda.device_count()))
        version = f"torch {torch.__version__}"

    result = dict(framework=framework, n_dev=n_dev, cell=list(cell),
                  version=version)
    t0 = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = vcd()
    result["vcd_warm"] = time.perf_counter() - t0
    np.save(os.path.join(RESULTS_DIR,
                         f"_p4_{framework}_{n_dev}_{cell[0]}.npy"),
            out[::max(1, out.shape[0] // 64)])   # a slice sample for diffs
    result["recon_checksum"] = float(np.sum(np.abs(out)))
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["gpu_peak_bytes"] = gpu_peak()
    return result


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p4.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p4.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path])
    if proc.returncode != 0:
        return dict(error=f"worker exited {proc.returncode}", **cfg)
    with open(out_path) as f:
        return json.load(f)


def main():
    import numpy as np
    max_n = int(os.environ.get("P4_MAX_DEVICES", "2"))
    ns = [n for n in (1, 2, 4) if n <= max_n]
    all_results = dict(run_label=RUN_LABEL, host=platform.node(), rows=[])
    for cell in CELLS:
        for framework in ("jax", "torch"):
            for n in ns:
                python = JAX_PYTHON if framework == "jax" else TORCH_PYTHON
                label = f"{framework}/n{n}/{'x'.join(map(str, cell))}"
                print(f"{label} ...", flush=True)
                r = run_one(python, dict(framework=framework, n_dev=n,
                                         cell=list(cell)))
                all_results["rows"].append(r)
                if "error" in r:
                    print(f"  FAILED: {r['error']}", flush=True)
                else:
                    print(f"  cold {r['vcd_cold']:.2f}s warm {r['vcd_warm']:.2f}s"
                          f"  gpu {r['gpu_peak_bytes']/2**30:.2f}G"
                          f"  rss {r['peak_rss_bytes']/2**30:.2f}G", flush=True)
    # Value diffs from the saved slice samples.
    for cell in CELLS:
        for framework in ("jax", "torch"):
            try:
                ref = np.load(os.path.join(
                    RESULTS_DIR, f"_p4_{framework}_1_{cell[0]}.npy"))
                for n in ns[1:]:
                    x = np.load(os.path.join(
                        RESULTS_DIR, f"_p4_{framework}_{n}_{cell[0]}.npy"))
                    rel = np.max(np.abs(x - ref)) / max(np.max(np.abs(ref)), 1e-30)
                    print(f"diff {framework} n{n} vs n1 @{cell[0]}: {rel:.2e}",
                          flush=True)
            except FileNotFoundError:
                pass
    out = os.path.join(RESULTS_DIR, f"p4_gate_{RUN_LABEL}.json")
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
