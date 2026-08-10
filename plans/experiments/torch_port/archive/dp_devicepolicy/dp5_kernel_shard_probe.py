"""Single-variable probe: do the Triton kernels hold their values under
sharding?

The flip gate read n=2 and n=4 against n=1 at order-1 relative difference,
where the established multi-device float floor is about 1e-3.  A CPU
bisection cleared the device-policy changes: the same comparison on CPU
reads 9.6e-4 both before and after them, so the engine is unchanged.  What
CPU does not exercise is the Triton kernels, which are selected by their own
availability probe and are NOT turned off by ``compile_mode='off'``.

This probe varies exactly one thing.  Each arm runs the same seeded
reconstruction at the same cell and the same device count, once with the
kernels enabled and once with ``MBIRTORCH_DISABLE_TRITON=1`` forcing the
torch bodies.  If the divergence follows the kernels, the kernels and the
banded multi-device drivers disagree; if it follows the device count in both
settings, the engine is at fault and the kernels are exonerated.

The question matters because the n=2/n=4 gates predate the Triton kernels,
so the kernel-times-sharding combination may never have been measured.

Protocol note, now standing for every gate: `compile_mode='off'` does NOT
disable the kernels.  An arm that intends the plain torch engine sets
`MBIRTORCH_DISABLE_TRITON=1`, which this probe does per arm in the arm's own
environment, because the availability probe caches its verdict per process.

Environment:
    DP5_RESULTS=<dir>   where the jsonl goes
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

RESULTS_DIR = os.environ.get("DP5_RESULTS", ".")
CELL = (256, 64, 64)
VCD_ITERATIONS = 3
VCD_SEED = 4321
# The established multi-device float floor for cells of this class.
FLOOR = 5e-3


def _build(geometry, cell):
    import mbirtorch

    if geometry == "parallel":
        angles = np.linspace(0, np.pi, cell[0], endpoint=False)
        _model = mbirtorch.ParallelBeamModel(cell, angles, 
                                           compile_mode="off")
        _model.configure_devices(devices=["cuda"])
        return _model
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    sdd = 4 * cell[2]
    _model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=sdd,
                                   source_iso_dist=sdd, 
                                   compile_mode="off")
    _model.configure_devices(devices=["cuda"])
    return _model


def _sino_path(geometry):
    return os.path.join(RESULTS_DIR, f"_dp4_sino_{geometry}_{CELL[0]}.npy")


def _out_path(geometry, arm, n):
    return os.path.join(RESULTS_DIR, f"_dp5_{geometry}_{arm}{n}.npy")


def _force_bodies(model, geometry, arm):
    """Bind a MIXED body pair, so one direction can be blamed on its own.

    ``arm`` is 'triton' (both kernels), 'torch' (neither), 'fwd' (the kernel
    forward against the torch back), or 'back' (the mirror).  The composed
    gates used the same mixed-selection pattern.
    """
    if arm in ("triton", "torch"):
        return
    if geometry == "parallel":
        from mbirtorch.parallel_beam import (_parallel_forward_view_batch,
                                             _parallel_back_view_batch)
        torch_fwd, torch_back = (_parallel_forward_view_batch,
                                 _parallel_back_view_batch)
    else:
        from mbirtorch.cone_beam import (_cone_forward_view_batch,
                                         _cone_back_view_batch)
        torch_fwd, torch_back = _cone_forward_view_batch, _cone_back_view_batch
    original = model._view_batch_bodies

    def mixed():
        fwd, back = original()
        return (fwd, torch_back) if arm == "fwd" else (torch_fwd, back)
    model._view_batch_bodies = mixed
    model.create_projectors()


def run_arm(cfg):
    import torch

    geometry, kernels, n = cfg["geometry"], cfg["kernels"], cfg["n"]
    arm = cfg.get("arm", "triton" if kernels else "torch")
    model = _build(geometry, CELL)
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(n)
    _force_bodies(model, geometry, arm)

    fwd, back = model._view_batch_bodies()
    result = dict(cfg, forward_body=fwd.__name__, back_body=back.__name__,
                  visible=torch.cuda.device_count())

    sinogram = np.load(_sino_path(geometry))
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
    np.random.seed(VCD_SEED)
    start = time.time()
    recon, _info = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
    result["seconds"] = time.time() - start
    np.save(_out_path(geometry, arm, n), np.asarray(recon, np.float32))
    return result


def _spawn(cfg):
    env = dict(os.environ)
    # The kill switch is read INSIDE the availability probe, which caches its
    # verdict per process, so it must be set before the interpreter starts.
    env["MBIRTORCH_DISABLE_TRITON"] = "0" if cfg["kernels"] else "1"
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker",
         json.dumps(cfg)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-2500:])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return dict(cfg, error="no result line")


def main():
    import torch

    counts = [n for n in (1, 2, 4) if n <= torch.cuda.device_count()]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"dp5_kernel_shard_{stamp}.jsonl")
    rows = []
    for geometry in ("parallel", "cone"):
        if not os.path.exists(_sino_path(geometry)):
            print(f"missing sinogram for {geometry}; run dp4 first", flush=True)
            continue
        for arm in ("triton", "torch", "fwd", "back"):
            for n in counts:
                print(f"  {geometry} {arm} n={n}", flush=True)
                row = _spawn(dict(geometry=geometry, arm=arm,
                                  kernels=arm != "torch", n=n))
                rows.append(row)
                with open(out_path, "a") as handle:
                    handle.write(json.dumps(row) + "\n")

    print(f"\n===== kernels x sharding ({out_path}) =====")
    header = (f'{"geometry":>10}{"arm":>8}{"n":>4}{"rel vs n=1":>13}'
              f'{"verdict":>9}  forward / back bodies')
    print(header)
    print("-" * len(header))
    for geometry in ("parallel", "cone"):
        for arm in ("torch", "triton", "fwd", "back"):
            base = _out_path(geometry, arm, 1)
            if not os.path.exists(base):
                continue
            ref = np.load(base)
            for n in counts[1:]:
                path = _out_path(geometry, arm, n)
                if not os.path.exists(path):
                    continue
                rel = float(np.max(np.abs(ref - np.load(path)))
                            / max(np.max(np.abs(ref)), 1e-30))
                row = next((r for r in rows if r.get("geometry") == geometry
                            and r.get("arm") == arm and r.get("n") == n), {})
                bodies = (f'{row.get("forward_body", "?")[-18:]} / '
                          f'{row.get("back_body", "?")[-18:]}')
                print(f'{geometry:>10}{arm:>8}{n:>4}{rel:>13.3e}'
                      f'{"ok" if rel < FLOOR else "HIGH":>9}  {bodies}')
    print("\narm key: torch = neither kernel, triton = both, "
          "fwd = kernel forward only, back = kernel back only.")
    print("Whichever single-direction arm is HIGH names the broken kernel.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        print("__RESULT__" + json.dumps(run_arm(json.loads(sys.argv[2]))))
    else:
        main()
