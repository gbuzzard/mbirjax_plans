"""Job D, the ruler repair: the composed gate's VALUE comparison re-run with a
SHARED sinogram.

The parallel gate's one red cell was value-vs-jax at 1024 (rel 0.375, identical
in the pure-body arm).  The chain behind it: each arm generated its OWN phantom,
the two frameworks' phantom generators differ at f32-vs-f64 ellipsoid
boundaries (documented at the top of mbirtorch/utilities.py), those flipped
voxels entered each framework's sinogram, and the value comparison inherited
them.  p5k7 localized it to the init, p5k8 refuted the competing channel-tie
explanation on its own prediction, p5k9 links the divergent slices to the
phantom-diff slices.

So the gate was measuring its own ruler.  This job repairs it: ONE phantom, ONE
forward projection, ONE sinogram written to disk, and every arm reconstructs
THAT array.  Nothing else changes -- same cell, same seed, same iteration
count, same metric, same arms, same envelope.

    generator     one framework builds phantom -> sinogram -> .npy
    torch_kernel  both parallel kernels, reconstructing the shared sinogram
    torch_body    MBIRTORCH_DISABLE_TRITON=1, same input
    jax           mbirjax, same input

VALUE ONLY.  Wall times are recorded because they are free, but they are NOT
the gate's timing numbers: this job runs a different input path and its arms
are not the timed arms.  Job A's table stands for timing.

Weights are recomputed per arm from the shared sinogram with one formula and
one dtype (float32 in both, which the original gate did only on the torch
side) -- a ruler-repair job may not leave a second asymmetry in place.

PROTOCOL RULE this establishes, for the findings: a cross-framework VALUE gate
hands one sinogram to both frameworks.  In-framework phantom generation stays
fine for timing arms, where each framework should exercise its own chain.

Run:
    <torch python> p5ka_shared_gate.py     on a CUDA node (see p5ka_gautschi.sbatch)
    python p5ka_shared_gate.py --dry-run   anywhere: print the plan
    python p5ka_shared_gate.py --help

Environment (export from the SUBMITTING SHELL, never in an --export list):
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the arm subprocesses
    P5KA_CELLS=1024                   cells to run (default 1024, the red cell)
    P5KA_ITERATIONS=3                 VCD iterations (single depth)
    P5KA_ITER_LIST='3,6,10'           convergence trace: run each arm at
                                      each depth and report both metrics
                                      at each (unset = single depth)
    P5KA_GENERATOR=torch|jax          which framework builds the shared input
    P5KA_KEEP_SINO=1                  keep the shared sinogram .npy afterwards
    P5KA_SMOKE=1 / P5KA_DEVICE=cpu    local smoke
"""

import json
import os
import platform
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The red cell only by default: 512 already passed at 5.5e-4 with each
# framework on its own phantom, which is the number this run should reproduce.
ALL_CELLS = [(512, 448, 384), (1024, 1008, 992)]
CELLS = [(1024, 1008, 992)]
ARMS = ("torch_kernel", "torch_body", "jax")
EXPECTED_BODIES = {"torch_kernel": (True, True), "torch_body": (False, False)}

SMOKE = os.environ.get("P5KA_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5KA_DEVICE", "cpu" if SMOKE else "cuda")
GENERATOR = os.environ.get("P5KA_GENERATOR", "torch")

VCD_ITERATIONS = int(os.environ.get("P5KA_ITERATIONS", "3"))
# Optional CONVERGENCE TRACE: run each arm at several depths and report both
# metrics at each.  Every depth runs FROM SCRATCH, because recon(k) with the
# seed reset is exactly the first k iterations of a longer run -- the sequence
# is truncated rather than resampled and early stopping is off, the same prefix
# property the p5k7 attribution trace relied on.  Unset, this reproduces the
# single-depth run exactly.
ITER_LIST = [int(x) for x in
             os.environ.get("P5KA_ITER_LIST", "").replace(";", ",").split(",")
             if x.strip()] or [VCD_ITERATIONS]
VCD_SEED = 13
SAMPLE_ROWS = 16          # the p5k6 artifact convention, so the numbers compare
VALUE_REL_TOL = 5e-3

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    wanted = os.environ.get("P5KA_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in ALL_CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _sino_path(cell):
    return os.path.join(RESULTS_DIR, f"_p5ka_shared_sino_{cell[0]}.npy")


def _sample_path(arm, cell, iterations):
    return os.path.join(RESULTS_DIR,
                        f"_p5ka_{arm}_{cell[0]}_k{iterations}.npy")


def arm_env(arm):
    """The environment overrides that DEFINE an arm (as in p5k6_pgate.py):
    every switch set explicitly, none inherited."""
    if arm == "torch_kernel":
        env = {"MBIRTORCH_ENABLE_TRITON_PBACK": "1",
               "MBIRTORCH_ENABLE_TRITON_PFWD": "1",
               "MBIRTORCH_DISABLE_TRITON": "0"}
    elif arm == "torch_body":
        env = {"MBIRTORCH_ENABLE_TRITON_PBACK": "0",
               "MBIRTORCH_ENABLE_TRITON_PFWD": "0",
               "MBIRTORCH_DISABLE_TRITON": "1"}
    else:
        env = {}
    return env


def _weights(sinogram):
    """The gate's weighting, in ONE dtype for every arm.  The original gate
    cast to float32 on the torch side only; a ruler-repair job may not leave
    that asymmetry standing."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def generator_worker(cfg):
    """Build the shared input once: phantom -> sinogram -> .npy on scratch."""
    import numpy as np

    cell = tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if GENERATOR == "jax":
        import jax
        import mbirjax as mj

        model = mj.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mj.generate_3d_shepp_logan_low_dynamic_range
        version = f"jax {jax.__version__}"
    else:
        import torch

        import mbirtorch as mt

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mt.generate_3d_shepp_logan_low_dynamic_range
        version = f"torch {torch.__version__}"

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    phantom = phantom_fn(recon_shape)
    sinogram = np.ascontiguousarray(np.asarray(model.forward_project(phantom),
                                               dtype=np.float32))
    np.save(_sino_path(cell), sinogram)
    return dict(role="generator", framework=GENERATOR, cell=list(cell),
                version=version, recon_shape=list(recon_shape),
                sinogram_shape=list(sinogram.shape),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)),
                sinogram_bytes=int(sinogram.nbytes),
                path=_sino_path(cell))


def arm_worker(cfg):
    """One arm's recon of the SHARED sinogram."""
    import numpy as np

    arm, cell = cfg["arm"], tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    sinogram = np.load(_sino_path(cell))
    weights = _weights(sinogram)

    if arm == "jax":
        import jax
        import mbirjax as mj

        model = mj.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        result = dict(arm=arm, framework="jax",
                      version=f"jax {jax.__version__}")

        def peak():
            return max(int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
                       for d in jax.devices())

        def sync():
            pass
    else:
        import torch

        import mbirtorch as mt
        import mbirtorch.triton_parallel as tp

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        fwd_hook, back_hook = model._view_batch_bodies()
        fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
        back_name = getattr(back_hook, "__name__", str(back_hook))
        bound = ("triton" in fwd_name, "triton" in back_name)
        expected = EXPECTED_BODIES[arm]
        result = dict(arm=arm, framework="torch",
                      version=f"torch {torch.__version__}",
                      fwd_body=fwd_name, back_body=back_name,
                      expected_bodies=list(expected),
                      arm_ok=(bound == expected),
                      constants=[tp.PARALLEL_BACK_BLOCK_P,
                                 tp.PARALLEL_BACK_BLOCK_R,
                                 tp.PARALLEL_BACK_NUM_WARPS,
                                 tp.PARALLEL_BACK_NUM_STAGES],
                      fwd_constants=[tp.PARALLEL_FWD_BLOCK_P,
                                     tp.PARALLEL_FWD_BLOCK_R,
                                     tp.PARALLEL_FWD_NUM_WARPS,
                                     tp.PARALLEL_FWD_NUM_STAGES])

        def peak():
            if DEVICE == "cuda":
                value = int(torch.cuda.max_memory_allocated())
            else:
                value = 0
            return value

        def sync():
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    result.update(cell=list(cell), vcd_iterations=VCD_ITERATIONS,
                  iter_list=list(ITER_LIST),
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)),
                  weights_checksum=float(np.sum(weights, dtype=np.float64)))

    walls, checksums = {}, {}
    for iterations in ITER_LIST:
        np.random.seed(VCD_SEED)
        t0 = time.perf_counter()
        recon, _ = model.recon(sinogram, weights=weights,
                               max_iterations=iterations,
                               stop_threshold_change_pct=0.0)
        sync()
        # Incidental, NOT the gate's timing (different input path, cold
        # process, and each depth restarts from scratch).
        walls[str(iterations)] = time.perf_counter() - t0
        recon = np.asarray(recon)
        step = max(1, recon.shape[0] // SAMPLE_ROWS)
        np.save(_sample_path(arm, cell, iterations), recon[::step])
        checksums[str(iterations)] = float(np.sum(np.abs(recon),
                                                  dtype=np.float64))
        del recon
    result.update(sample_step=step, gpu_peak_bytes=peak(),
                  wall_incidental_s=walls, recon_checksums=checksums)
    return result


def run_one(python, cfg, env_extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5ka.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5ka.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ, **(env_extra or {}))
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=env)
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(error=f"worker exited {proc.returncode}", **cfg)
    else:
        with open(out_path) as f:
            row = json.load(f)
    return row


def _rel_max(path_a, path_b):
    import numpy as np

    if os.path.exists(path_a) and os.path.exists(path_b):
        a, b = np.load(path_a), np.load(path_b)
        if a.shape == b.shape:
            value = float(np.max(np.abs(a - b))
                          / max(float(np.max(np.abs(b))), 1e-30))
        else:
            value = None
    else:
        value = None
    return value


def _norm_rel(path_a, path_b):
    import numpy as np

    if os.path.exists(path_a) and os.path.exists(path_b):
        a, b = np.load(path_a), np.load(path_b)
        if a.shape == b.shape:
            value = float(np.linalg.norm(a - b)
                          / max(float(np.linalg.norm(b)), 1e-30))
        else:
            value = None
    else:
        value = None
    return value


def analyze(cell, rows):
    by_arm = {r.get("arm"): r for r in rows if "error" not in r}
    label = "x".join(map(str, cell))
    print(f"\n=== shared-sinogram value gate, cell {label} ===", flush=True)
    summary = dict(cell=list(cell), generator=GENERATOR,
                   missing=[a for a in ARMS if a not in by_arm],
                   arms_failed=[a for a, r in by_arm.items()
                                if r.get("arm_ok") is False])
    if summary["missing"]:
        print(f"  MISSING ARMS: {summary['missing']}", flush=True)
    if summary["arms_failed"]:
        print(f"  ARM CHECK FAILED: {summary['arms_failed']} -- the value "
              f"numbers below do not describe the configuration they claim",
              flush=True)

    # Every arm must have consumed the SAME input: that is the whole point.
    # The generator's own checksum joins the comparison -- it is the value the
    # arms are supposed to have read back off the disk.
    checksums = {(r.get("arm") or r.get("role")): r.get("sinogram_checksum")
                 for r in rows if "error" not in r}
    summary["sinogram_checksums"] = checksums
    identical = len(set(f"{v:.10g}" for v in checksums.values() if v)) <= 1
    summary["shared_input_verified"] = identical
    print(f"  shared input verified: {identical}  (sinogram checksums "
          f"{ {a: f'{v:.8g}' for a, v in checksums.items()} })", flush=True)

    pairs = (("torch_kernel", "jax"), ("torch_body", "jax"),
             ("torch_kernel", "torch_body"))
    print(f"  {'comparison':<28}{'iters':>6}{'max rel':>12}{'norm rel':>12}"
          f"   vs {VALUE_REL_TOL:.0e}", flush=True)
    for left, right in pairs:
        if left in by_arm and right in by_arm:
            trace = []
            for iterations in ITER_LIST:
                rel = _rel_max(_sample_path(left, cell, iterations),
                               _sample_path(right, cell, iterations))
                norm = _norm_rel(_sample_path(left, cell, iterations),
                                 _sample_path(right, cell, iterations))
                trace.append(dict(iterations=iterations, max_rel=rel,
                                  norm_rel=norm,
                                  passes=(rel is not None
                                          and rel <= VALUE_REL_TOL)))
                if rel is not None:
                    print(f"  {left + ' vs ' + right:<28}{iterations:>6}"
                          f"{rel:>12.3e}{norm:>12.3e}   "
                          f"{'PASS' if rel <= VALUE_REL_TOL else 'FAIL'}",
                          flush=True)
            summary[f"trace_{left}_vs_{right}"] = trace
            # The single-depth keys stay, at the DEEPEST depth measured, so a
            # trace run and a plain run report the same field names.
            if trace:
                summary[f"value_rel_{left}_vs_{right}"] = trace[-1]["max_rel"]
                summary[f"norm_rel_{left}_vs_{right}"] = trace[-1]["norm_rel"]
    # Does the max-rel actually fall as the trajectories converge, or plateau?
    key_trace = summary.get("trace_torch_kernel_vs_jax") or []
    live = [t for t in key_trace if t["max_rel"] is not None]
    if len(live) >= 2:
        first, last = live[0], live[-1]
        ratio = last["max_rel"] / max(first["max_rel"], 1e-30)
        summary["max_rel_trend_ratio"] = ratio
        if last["passes"]:
            trend = (f"CONVERGES UNDER THE ENVELOPE by iteration "
                     f"{last['iterations']}")
        elif ratio < 0.7:
            trend = (f"falling ({ratio:.2f}x over "
                     f"{first['iterations']}->{last['iterations']}) but still "
                     f"above {VALUE_REL_TOL:.0e}")
        else:
            trend = (f"PLATEAUS above the envelope ({ratio:.2f}x over "
                     f"{first['iterations']}->{last['iterations']}) -- this is "
                     f"the metric question, not a convergence question")
        summary["max_rel_trend"] = trend
        print(f"  trend (torch_kernel vs jax, max rel): {trend}", flush=True)
    for arm, row in by_arm.items():
        summary[f"{arm}_gpu_peak_bytes"] = row.get("gpu_peak_bytes")
        summary[f"{arm}_wall_incidental_s"] = row.get("wall_incidental_s")
    return summary


def print_plan():
    print(f"cells:      {[('x'.join(map(str, c))) for c in selected_cells()]}")
    print(f"generator:  {GENERATOR} builds one phantom -> one sinogram -> .npy")
    print(f"arms:       {list(ARMS)} (all reconstruct THAT array)")
    print(f"iterations: {VCD_ITERATIONS} (seed {VCD_SEED}); VALUE only, "
          f"timing is incidental")
    print(f"envelope:   {VALUE_REL_TOL:.0e}")
    for arm in ARMS:
        print(f"  {arm:<13} env {arm_env(arm)}  expect (fwd, back) triton = "
              f"{EXPECTED_BODIES.get(arm)}")
    print(f"results:    {RESULTS_DIR}/p5ka_shared_gate_{RUN_LABEL}.json")


def main():
    cells = selected_cells()
    print(f"p5ka shared-sinogram value gate on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, generator {GENERATOR}, "
          f"{VCD_ITERATIONS} iterations, VALUE only", flush=True)
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", generator=GENERATOR,
                       vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                       value_rel_tol=VALUE_REL_TOL, rows=[], summaries=[])
    for cell in cells:
        label = "x".join(map(str, cell))
        print(f"\n{label}/generator ({GENERATOR}) ...", flush=True)
        gen_python = JAX_PYTHON if GENERATOR == "jax" else TORCH_PYTHON
        gen_row = run_one(gen_python, dict(role="generator", cell=list(cell)))
        rows = [gen_row]
        if "error" in gen_row:
            print(f"  FAILED: {gen_row['error'][:200]}", flush=True)
            all_results["rows"].extend(rows)
            continue
        print(f"  wrote {gen_row['sinogram_bytes'] / 2**30:.2f} GiB, checksum "
              f"{gen_row['sinogram_checksum']:.8g}", flush=True)
        for arm in ARMS:
            python = JAX_PYTHON if arm == "jax" else TORCH_PYTHON
            print(f"{label}/{arm} ...", flush=True)
            row = run_one(python, dict(role="arm", arm=arm, cell=list(cell)),
                          arm_env(arm))
            rows.append(row)
            if "error" in row:
                print(f"  FAILED: {row['error'][:200]}", flush=True)
            else:
                mark = "" if row.get("arm_ok", True) else "  ARM MISMATCH"
                sums = row.get("recon_checksums") or {}
                text = "  ".join(f"k{k} {v:.8g}" for k, v in sums.items())
                print(f"  recon checksums {text}  "
                      f"gpu {row['gpu_peak_bytes'] / 2**30:.2f}G{mark}",
                      flush=True)
        summary = analyze(cell, rows)
        all_results["rows"].extend(rows)
        all_results["summaries"].append(summary)
        if os.environ.get("P5KA_KEEP_SINO", "0") != "1":
            # A 4 GiB scratch artifact per cell, deleted unless asked for.
            try:
                os.remove(_sino_path(cell))
            except OSError:
                pass

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5ka_shared_gate_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


def _run_worker(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        if cfg["role"] == "generator":
            result = generator_worker(cfg)
        else:
            result = arm_worker(cfg)
    except Exception as e:                                        # noqa: BLE001
        traceback.print_exc()
        result = dict(error=f"{type(e).__name__}: {e}"[:400], **cfg)
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _run_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--dry-run":
        print_plan()
    elif len(sys.argv) >= 2:
        print(f"unknown argument {sys.argv[1]!r}; try --help")
        sys.exit(2)
    else:
        main()
