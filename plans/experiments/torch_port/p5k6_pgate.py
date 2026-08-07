"""The COMPOSED gate for the Triton PARALLEL pair -- the seeded 3-iteration
warm parallel-beam VCD recon at both gate cells, one variant per subprocess.
The cone counterpart is p5k2_gate.py; this is its parallel twin, same structure
and the same rigor, asking the same question of a different geometry: do the
parallel back and forward kernels earn their default-on?

Composition is measured, not extrapolated (the pallas E4 lesson): the isolated
body bench p5k6_psweep.py ranks constants, and THIS script decides.  Five
variants per cell, each its own process and its own ruler:

    torch_kernel        PBACK=1 and PFWD=1 -- both parallel kernels
    torch_back_only     PBACK=1 only -- the back kernel's own share, and the
                        baseline the forward's marginal win is measured against
    torch_body          MBIRTORCH_DISABLE_TRITON=1 -- neither kernel, the pure
                        torch baseline (and the honest CURRENT baseline: the
                        pre-kernel numbers quoted in the plan predate the
                        projector restructure)
    torch_body_repeat   the body arm again: the NONDETERMINISM FLOOR
    jax                 mbirjax, the pallas reference the replacement rule
                        names

`torch_back_only` is what separates the two kernels' contributions inside one
run.  Without it the only comparison is both-kernels vs no-kernels, which
cannot show a forward regression hiding behind a back win, and splitting the
comparison across two runs is the cross-run extrapolation E4 warns about.  Drop
it with P5K6_SKIP_BACK_ONLY=1 only if the queue forces it.

The floor arm exists because the value diffs are not measured against zero: two
identical torch-body processes at the smoke cell already differ by rel 7.5e-7
(measured locally, CPU, 2026-08-06) because reductions reorder between
processes.  The parallel FORWARD kernel adds atomics on top of that, whose own
floor p5k6_psweep.py measures per configuration at body level.  Drop the floor
arm with P5K6_SKIP_FLOOR=1 when the job is tight.

Reported per variant: cold and warm recon wall, the process GPU peak and the
warm-only allocator peak, the full-recon checksum, and which bodies were
ACTUALLY bound -- both of them, against a per-variant expectation.  A gate that
silently measured torch against torch would report a flawless 1.00x, so every
arm is verified from the model rather than from the env var meant to produce
it, and the shared launch-key set (whose keys lead with 'pback'/'pfwd') gives
the positive witness that each kernel really launched inside the recon.

Derived per cell: the forward kernel's marginal speedup over back-only, the
back kernel's own speedup over the torch body, the combined speedup, the
torch-vs-jax time and memory ratios of the replacement rule (2x time, ~1.5x
memory -- the pre-kernel parallel baselines were 1.86x at 512 and 3.61x at
1024, so 1024 is the cell that has to move), and the value diffs at the
compile-latitude envelope of 5e-3: kernel vs the torch body (the isolating
comparison) and kernel vs jax (the cross-framework one the plan names), each
printed beside the repeat-run floor that says what "different" is worth.

Run:
    <torch python> p5k6_pgate.py          on a CUDA node (see p5k6_pgate_gautschi.sbatch)
    python p5k6_pgate.py --dry-run        anywhere: print the variant plan
    python p5k6_pgate.py --help

Environment.  Pass these from the SUBMITTING SHELL, never in an sbatch
`--export=ALL,VAR=a,b,c` list: slurm splits that list on commas, so a
four-field tuple arrives mangled (measured 2026-08-06).  A semicolon form,
'16;64;4;1', is accepted for exactly that reason and survives an --export list.
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the variant subprocesses
    P5K6_PBACK_CONSTANTS='P,R,warps,stages'   parallel BACK kernel constants
    P5K6_PFWD_CONSTANTS='P,R,warps,stages'    parallel FORWARD kernel constants
                                      (both default to the values pinned in
                                      mbirtorch; feed them the sweep winners)
    P5K6_CELLS=512,1024               run a subset of the cells (by view count)
    P5K6_ITERATIONS=3                 VCD iterations per recon
    P5K6_SKIP_JAX=1                   run the torch arms only
    P5K6_SKIP_FLOOR=1                 drop the repeat (floor) arm
    P5K6_SKIP_BACK_ONLY=1             drop the back-only arm
    P5K6_SMOKE=1                      tiny cell on P5K6_DEVICE (default cpu)
    P5K6_DEVICE=cuda|cpu|mps          the torch device for the torch arms
"""

import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

CELLS = [(512, 448, 384), (1024, 1008, 992)]
VARIANTS = ("torch_kernel", "torch_back_only", "torch_body",
            "torch_body_repeat", "jax")

# What each torch arm must turn out to be, as (forward is triton, back is
# triton).  The arm check compares this against the bodies the MODEL bound, so
# an opt-in that stopped working is caught as a failed arm rather than reported
# as a ratio.
EXPECTED_BODIES = {"torch_kernel": (True, True),
                   "torch_back_only": (False, True),
                   "torch_body": (False, False),
                   "torch_body_repeat": (False, False)}

SMOKE = os.environ.get("P5K6_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K6_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("P5K6_ITERATIONS", "3"))
VCD_SEED = 13
SAMPLE_ROWS = 16          # recon rows kept per variant for the value diffs

# The gates.  Value: the compile-latitude envelope, not the body-level 1e-5 --
# the recon is a nonlinear iteration and every kernel rounding difference
# enters every subset update.
VALUE_REL_TOL = 5e-3
# The replacement rule (port_plan): torch within 2x of jax on time, ~1.5x on
# memory.  Pre-kernel parallel warm vcd was 1.86x at 512 (passes) and 3.61x at
# 1024 (fails), with the back projector 7.6x/4.4x behind and the forward
# 1.78x/1.90x -- those are the gaps these kernels exist to close.
TIME_RULE_MAX = 2.0
MEMORY_RULE_MAX = 1.5

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    """The cells to run: the smoke cell, all of CELLS, or the subset named by
    P5K6_CELLS as a comma-separated list of view counts."""
    wanted = os.environ.get("P5K6_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def parse_constants(text, name):
    """('P,R,warps,stages') -> [P, R, warps, stages], or None when unset.
    Raises on a malformed value: a silently ignored tuning request would make
    the gate measure the WRONG configuration and never say so.  Semicolons are
    accepted as separators because a comma list does not survive slurm's
    --export."""
    text = (text or "").strip()
    if not text:
        constants = None
    else:
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        parts = [p for p in parts if p]
        if len(parts) != 4:
            raise ValueError(f"{name} must be 'P,R,warps,stages' (or the same "
                             f"four fields separated by semicolons); got "
                             f"{text!r}")
        constants = [int(p) for p in parts]
    return constants


def parse_all_constants():
    """(back, forward) parallel constant lists from the environment."""
    back = parse_constants(os.environ.get("P5K6_PBACK_CONSTANTS"),
                           "P5K6_PBACK_CONSTANTS")
    forward = parse_constants(os.environ.get("P5K6_PFWD_CONSTANTS"),
                              "P5K6_PFWD_CONSTANTS")
    return back, forward


def variant_env(variant):
    """The environment overrides that DEFINE a variant.  Every switch is set
    explicitly in every torch arm, never left inherited: an inherited
    MBIRTORCH_DISABLE_TRITON=1 from the submitting shell would turn the ON arm
    into a second copy of the body arm -- a run that looks like a perfect
    1.00x -- and an inherited opt-in would blur the back-only arm.  The arm
    check, not this dict, is what proves the configuration."""
    if variant == "torch_kernel":
        env = {"MBIRTORCH_ENABLE_TRITON_PBACK": "1",
               "MBIRTORCH_ENABLE_TRITON_PFWD": "1",
               "MBIRTORCH_DISABLE_TRITON": "0"}
    elif variant == "torch_back_only":
        env = {"MBIRTORCH_ENABLE_TRITON_PBACK": "1",
               "MBIRTORCH_ENABLE_TRITON_PFWD": "0",
               "MBIRTORCH_DISABLE_TRITON": "0"}
    elif variant.startswith("torch_body"):
        env = {"MBIRTORCH_ENABLE_TRITON_PBACK": "0",
               "MBIRTORCH_ENABLE_TRITON_PFWD": "0",
               "MBIRTORCH_DISABLE_TRITON": "1"}
    else:
        env = {}
    return env


# ── the variant workers ───────────────────────────────────────────────────────
def _launch_key_counts(triton_parallel):
    """(pback, pfwd) launch-key counts.  All four cone and parallel kernels
    share one key set and every key leads with the kernel's name, so the set
    doubles as a per-kernel launch witness -- the positive evidence behind the
    arm check."""
    back, forward = 0, 0
    for key in triton_parallel._COMPILED_LAUNCH_KEYS:
        name = key[0] if isinstance(key, tuple) and key else None
        if name == "pback":
            back += 1
        elif name == "pfwd":
            forward += 1
    return back, forward


def torch_worker(cfg):
    """One torch arm: build the parallel model, forward-project the phantom,
    then the seeded VCD recon cold and warm."""
    import numpy as np
    import torch

    import mbirtorch
    import mbirtorch.triton_parallel as triton_parallel

    # Module attributes read at LAUNCH time by the wrappers, so setting them
    # here -- before the model exists -- governs every kernel call below.
    constants = cfg.get("constants")
    if constants:
        (triton_parallel.PARALLEL_BACK_BLOCK_P,
         triton_parallel.PARALLEL_BACK_BLOCK_R,
         triton_parallel.PARALLEL_BACK_NUM_WARPS,
         triton_parallel.PARALLEL_BACK_NUM_STAGES) = [int(c) for c in constants]
    fwd_constants = cfg.get("fwd_constants")
    if fwd_constants:
        (triton_parallel.PARALLEL_FWD_BLOCK_P,
         triton_parallel.PARALLEL_FWD_BLOCK_R,
         triton_parallel.PARALLEL_FWD_NUM_WARPS,
         triton_parallel.PARALLEL_FWD_NUM_STAGES) = [int(c)
                                                     for c in fwd_constants]

    cell = tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles, device=DEVICE)
    model.set_params(no_warning=True, verbose=0)

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elif DEVICE == "mps":
            torch.mps.synchronize()

    def peak_bytes():
        if DEVICE == "cuda":
            value = int(torch.cuda.max_memory_allocated())
        else:
            value = 0
        return value

    # WHICH BODIES ARE BOUND -- the guard that keeps this from being a
    # torch-vs-torch measurement wearing a kernel's name.  The hook answers
    # what the geometry selected; the projector answers what the driver will
    # actually call (maybe_compile passes a hand-written kernel through
    # unchanged and wraps a torch body as compiled_<name>).
    fwd_hook, back_hook = model._view_batch_bodies()
    bound_fwd = model.projector_functions._fwd_body_per_dev[0]
    bound_back = model.projector_functions._back_body_per_dev[0]
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    fwd_is_kernel = "triton" in fwd_name
    back_is_kernel = "triton" in back_name
    expected = EXPECTED_BODIES.get(cfg["variant"])
    result = dict(variant=cfg["variant"], framework="torch", cell=list(cell),
                  version=f"torch {torch.__version__}", device=DEVICE,
                  device_name=(torch.cuda.get_device_name(0)
                               if DEVICE == "cuda" else DEVICE),
                  fwd_body=fwd_name, back_body=back_name,
                  bound_fwd_body=getattr(bound_fwd, "__name__",
                                         str(bound_fwd)),
                  bound_back_body=getattr(bound_back, "__name__",
                                          str(bound_back)),
                  fwd_kernel_selected=fwd_is_kernel,
                  back_kernel_selected=back_is_kernel,
                  expected_bodies=list(expected),
                  arm_ok=((fwd_is_kernel, back_is_kernel) == expected),
                  constants=[triton_parallel.PARALLEL_BACK_BLOCK_P,
                             triton_parallel.PARALLEL_BACK_BLOCK_R,
                             triton_parallel.PARALLEL_BACK_NUM_WARPS,
                             triton_parallel.PARALLEL_BACK_NUM_STAGES],
                  fwd_constants=[triton_parallel.PARALLEL_FWD_BLOCK_P,
                                 triton_parallel.PARALLEL_FWD_BLOCK_R,
                                 triton_parallel.PARALLEL_FWD_NUM_WARPS,
                                 triton_parallel.PARALLEL_FWD_NUM_STAGES],
                  disable_env=os.environ.get("MBIRTORCH_DISABLE_TRITON", ""),
                  pback_env=os.environ.get("MBIRTORCH_ENABLE_TRITON_PBACK", ""),
                  pfwd_env=os.environ.get("MBIRTORCH_ENABLE_TRITON_PFWD", ""),
                  vcd_iterations=VCD_ITERATIONS)
    # Ask for the reasons only where a kernel was WANTED: a self-check compiles
    # a kernel, and the pure-torch arms' numbers must not carry that cost.
    from mbirtorch.kernel_availability import (parallel_back_kernel_usable,
                                               parallel_forward_kernel_usable)
    if expected[1]:
        result["back_usable"] = list(parallel_back_kernel_usable(model))
    if expected[0]:
        result["fwd_usable"] = list(parallel_forward_kernel_usable(model))

    # A kernel's first-use self-check launches it once, so the launch-key
    # counts are already nonzero before any projection: the witness that the
    # RECON used a kernel is the delta across the runs, not the total.
    keys_before = _launch_key_counts(triton_parallel)
    result["launch_keys_before"] = list(keys_before)

    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram)))

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _ = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
        sync()
        return np.asarray(recon)

    t0 = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - t0
    # The process peak (what p2/p3/p4 recorded, and what the jax arm can
    # report) and the warm-only allocator peak, which is the honest
    # steady-state number: the cold run's peak carries compile allocations.
    peak_after_cold = peak_bytes()
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = vcd()
    result["vcd_warm"] = time.perf_counter() - t0
    result["gpu_peak_warm_bytes"] = peak_bytes()
    result["gpu_peak_bytes"] = max(peak_after_cold,
                                   result["gpu_peak_warm_bytes"])
    keys_after = _launch_key_counts(triton_parallel)
    result["launch_keys"] = list(keys_after)
    result["pback_launch_keys_delta"] = keys_after[0] - keys_before[0]
    result["pfwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
    _finish(result, out)
    return result


def jax_worker(cfg):
    """The jax arm: the same cell and iteration count through mbirjax."""
    import numpy as np

    import jax
    import mbirjax

    cell = tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(cell, angles)
    model.set_params(no_warning=True, verbose=0)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    weights = np.exp(-sinogram / (2 * np.max(sinogram)))
    result = dict(variant=cfg["variant"], framework="jax", cell=list(cell),
                  version=f"jax {jax.__version__}",
                  recon_shape=list(recon_shape),
                  sinogram_checksum=float(np.sum(np.abs(sinogram))),
                  vcd_iterations=VCD_ITERATIONS)

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _ = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
        return np.asarray(recon)

    t0 = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = vcd()
    result["vcd_warm"] = time.perf_counter() - t0
    # jax exposes no peak reset, so this is the process peak -- the same
    # quantity p2/p3/p4 recorded, and the one the torch process peak matches.
    # memory_stats() is None on the CPU backend (the local smoke path).
    result["gpu_peak_bytes"] = max(
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.devices())
    _finish(result, out)
    return result


def _finish(result, out):
    """The common tail: the full-recon checksum, a strided row sample saved for
    the cross-variant value diffs, and the host peak."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result["recon_checksum"] = float(np.sum(np.abs(out), dtype=np.float64))
    step = max(1, out.shape[0] // SAMPLE_ROWS)
    sample_path = os.path.join(
        RESULTS_DIR, f"_p5k6_{result['variant']}_{result['cell'][0]}.npy")
    np.save(sample_path, out[::step])
    result["sample_path"] = sample_path
    result["sample_step"] = step
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def run_one(python, cfg):
    """One variant in its own process, with the variant's env overrides."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k6_gate.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k6_gate.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ, **variant_env(cfg["variant"]))
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=env)
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(error=f"worker exited {proc.returncode}", **cfg)
    else:
        with open(out_path) as f:
            row = json.load(f)
    return row


def _rel_max(path_a, path_b):
    """max |a - b| / max |b| over the saved row samples, or None if either is
    missing."""
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


def summarize_cell(cell, rows, requested):
    """The gate verdict for one cell: the ratios and the value diffs, each with
    the rule it is measured against.  ``requested`` is the variant list this
    run asked for, so a deliberately skipped arm is not reported missing while
    a CRASHED one still is."""
    by_variant = {r.get("variant"): r for r in rows if "error" not in r}
    kernel = by_variant.get("torch_kernel")
    back_only = by_variant.get("torch_back_only")
    body = by_variant.get("torch_body")
    repeat = by_variant.get("torch_body_repeat")
    jax_row = by_variant.get("jax")
    summary = dict(cell=list(cell), requested=list(requested),
                   missing=[v for v in requested if v not in by_variant],
                   arms_failed=[v for v, r in by_variant.items()
                                if r.get("arm_ok") is False])
    if kernel is not None:
        summary["fwd_kernel_selected"] = kernel.get("fwd_kernel_selected")
        summary["back_kernel_selected"] = kernel.get("back_kernel_selected")
        summary["arm_ok"] = kernel.get("arm_ok")
        summary["pfwd_launch_keys_delta"] = kernel.get("pfwd_launch_keys_delta")
        summary["pback_launch_keys_delta"] = kernel.get(
            "pback_launch_keys_delta")
        summary["pback_constants"] = kernel.get("constants")
        summary["pfwd_constants"] = kernel.get("fwd_constants")
        summary["kernel_warm_s"] = kernel["vcd_warm"]
        summary["kernel_gpu_peak_bytes"] = kernel["gpu_peak_bytes"]
        summary["kernel_gpu_peak_warm_bytes"] = kernel.get("gpu_peak_warm_bytes")
    if back_only is not None:
        summary["back_only_arm_ok"] = back_only.get("arm_ok")
        summary["back_only_warm_s"] = back_only["vcd_warm"]
        summary["back_only_gpu_peak_bytes"] = back_only["gpu_peak_bytes"]
    if body is not None:
        summary["body_arm_ok"] = body.get("arm_ok")
        summary["body_warm_s"] = body["vcd_warm"]
        summary["body_gpu_peak_bytes"] = body["gpu_peak_bytes"]
        summary["body_gpu_peak_warm_bytes"] = body.get("gpu_peak_warm_bytes")
    if jax_row is not None:
        summary["jax_warm_s"] = jax_row["vcd_warm"]
        summary["jax_gpu_peak_bytes"] = jax_row["gpu_peak_bytes"]
    if kernel is not None and back_only is not None:
        summary["fwd_speedup_vs_back_only"] = (back_only["vcd_warm"]
                                               / kernel["vcd_warm"])
        summary["fwd_vs_back_only_memory"] = (
            kernel["gpu_peak_bytes"] / max(1, back_only["gpu_peak_bytes"]))
    if back_only is not None and body is not None:
        summary["back_speedup_vs_body"] = (body["vcd_warm"]
                                           / back_only["vcd_warm"])
    if kernel is not None and body is not None:
        summary["kernel_speedup_vs_body"] = (body["vcd_warm"]
                                             / kernel["vcd_warm"])
        summary["kernel_vs_body_memory"] = (kernel["gpu_peak_bytes"]
                                            / max(1, body["gpu_peak_bytes"]))
        rel = _rel_max(kernel["sample_path"], body["sample_path"])
        summary["value_rel_kernel_vs_body"] = rel
        summary["value_pass_vs_body"] = (rel is not None
                                         and rel <= VALUE_REL_TOL)
    if repeat is not None and body is not None:
        # The ruler for the lines above: two identical body processes.
        summary["value_rel_floor"] = _rel_max(repeat["sample_path"],
                                              body["sample_path"])
        summary["body_warm_repeat_s"] = repeat["vcd_warm"]
    if kernel is not None and jax_row is not None:
        summary["torch_kernel_vs_jax_time"] = (kernel["vcd_warm"]
                                               / jax_row["vcd_warm"])
        summary["torch_kernel_vs_jax_memory"] = (
            kernel["gpu_peak_bytes"] / max(1, jax_row["gpu_peak_bytes"]))
        summary["time_rule_pass"] = (summary["torch_kernel_vs_jax_time"]
                                     <= TIME_RULE_MAX)
        summary["memory_rule_pass"] = (summary["torch_kernel_vs_jax_memory"]
                                       <= MEMORY_RULE_MAX)
        # The cross-framework value envelope the plan names.  The two
        # frameworks build their own sinograms and iterate in their own float
        # order, so the body-vs-jax number below is the baseline this is read
        # against, exactly as the repeat floor serves the body comparison.
        rel_jax = _rel_max(kernel["sample_path"], jax_row["sample_path"])
        summary["value_rel_kernel_vs_jax"] = rel_jax
        summary["value_pass_vs_jax"] = (rel_jax is not None
                                        and rel_jax <= VALUE_REL_TOL)
    if body is not None and jax_row is not None:
        summary["torch_body_vs_jax_time"] = (body["vcd_warm"]
                                             / jax_row["vcd_warm"])
        summary["torch_body_vs_jax_memory"] = (
            body["gpu_peak_bytes"] / max(1, jax_row["gpu_peak_bytes"]))
        summary["value_rel_body_vs_jax"] = _rel_max(body["sample_path"],
                                                    jax_row["sample_path"])
    return summary


def print_cell_summary(summary):
    cell = "x".join(map(str, summary["cell"]))
    print(f"\n=== gate summary, cell {cell} ===", flush=True)
    if summary.get("missing"):
        print(f"  MISSING VARIANTS: {summary['missing']}", flush=True)
    bad_arms = summary.get("arms_failed") or []
    fwd_launches = summary.get("pfwd_launch_keys_delta")
    back_launches = summary.get("pback_launch_keys_delta")
    if bad_arms or not fwd_launches or not back_launches:
        print(f"  ARM CHECK FAILED: arms not in their claimed configuration "
              f"{bad_arms}, recon launch keys pfwd={fwd_launches} "
              f"pback={back_launches} -- every ratio below is meaningless",
              flush=True)
    else:
        print(f"  arm check ok: both triton bodies bound in the ON arm, "
              f"{fwd_launches} forward and {back_launches} back recon launch "
              f"key(s); pback constants {summary.get('pback_constants')}, "
              f"pfwd {summary.get('pfwd_constants')}", flush=True)
    for key, fmt in (("kernel_warm_s", "{:.2f} s"),
                     ("back_only_warm_s", "{:.2f} s"),
                     ("body_warm_s", "{:.2f} s"),
                     ("body_warm_repeat_s", "{:.2f} s"),
                     ("jax_warm_s", "{:.2f} s")):
        if summary.get(key) is not None:
            print(f"  {key:<28} {fmt.format(summary[key])}", flush=True)
    for key in ("kernel_gpu_peak_bytes", "back_only_gpu_peak_bytes",
                "body_gpu_peak_bytes", "jax_gpu_peak_bytes"):
        if summary.get(key) is not None:
            print(f"  {key:<28} {summary[key] / 2**30:.2f} GiB", flush=True)
    if summary.get("back_speedup_vs_body") is not None:
        print(f"  back kernel vs torch body "
              f"{summary['back_speedup_vs_body']:.2f}x", flush=True)
    if summary.get("fwd_speedup_vs_back_only") is not None:
        print(f"  forward kernel on top of back-only "
              f"{summary['fwd_speedup_vs_back_only']:.2f}x"
              f"  (memory {summary['fwd_vs_back_only_memory']:.2f}x)",
              flush=True)
    if summary.get("kernel_speedup_vs_body") is not None:
        print(f"  BOTH kernels vs torch body "
              f"{summary['kernel_speedup_vs_body']:.2f}x"
              f"  (memory {summary['kernel_vs_body_memory']:.2f}x)", flush=True)
    floor = summary.get("value_rel_floor")
    floor_text = "" if floor is None else f"  (repeat-run floor {floor:.2e})"
    if summary.get("value_rel_kernel_vs_body") is not None:
        verdict = "PASS" if summary["value_pass_vs_body"] else "FAIL"
        print(f"  value kernel vs body rel "
              f"{summary['value_rel_kernel_vs_body']:.2e}  <= "
              f"{VALUE_REL_TOL:.0e} ? {verdict}{floor_text}", flush=True)
    if summary.get("value_rel_kernel_vs_jax") is not None:
        verdict = "PASS" if summary["value_pass_vs_jax"] else "FAIL"
        baseline = summary.get("value_rel_body_vs_jax")
        base_text = ("" if baseline is None
                     else f"  (torch body vs jax {baseline:.2e})")
        print(f"  value kernel vs jax  rel "
              f"{summary['value_rel_kernel_vs_jax']:.2e}  <= "
              f"{VALUE_REL_TOL:.0e} ? {verdict}{base_text}", flush=True)
    if summary.get("torch_kernel_vs_jax_time") is not None:
        print(f"  torch(kernels) vs jax time "
              f"{summary['torch_kernel_vs_jax_time']:.2f}x <= {TIME_RULE_MAX} ? "
              f"{'PASS' if summary['time_rule_pass'] else 'FAIL'}", flush=True)
        print(f"  torch(kernels) vs jax memory "
              f"{summary['torch_kernel_vs_jax_memory']:.2f}x <= "
              f"{MEMORY_RULE_MAX} ? "
              f"{'PASS' if summary['memory_rule_pass'] else 'FAIL'}", flush=True)
    if summary.get("torch_body_vs_jax_time") is not None:
        print(f"  (torch body vs jax time "
              f"{summary['torch_body_vs_jax_time']:.2f}x, memory "
              f"{summary['torch_body_vs_jax_memory']:.2f}x -- the gap the "
              f"kernels are closing)", flush=True)


def main():
    cells = selected_cells()
    constants, fwd_constants = parse_all_constants()
    skip_jax = (os.environ.get("P5K6_SKIP_JAX", "0") == "1"
                or not os.path.exists(JAX_PYTHON))
    skip_floor = os.environ.get("P5K6_SKIP_FLOOR", "0") == "1"
    skip_back_only = os.environ.get("P5K6_SKIP_BACK_ONLY", "0") == "1"
    variants = [v for v in VARIANTS
                if not (v == "jax" and skip_jax)
                and not (v == "torch_body_repeat" and skip_floor)
                and not (v == "torch_back_only" and skip_back_only)]
    print(f"p5k6 parallel composed gate on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, variants {variants}, "
          f"{VCD_ITERATIONS} iterations, pback constants "
          f"{constants or 'pinned'}, pfwd constants "
          f"{fwd_constants or 'pinned'}", flush=True)

    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", vcd_iterations=VCD_ITERATIONS,
                       vcd_seed=VCD_SEED, pback_constants=constants,
                       pfwd_constants=fwd_constants,
                       gates=dict(value_rel_tol=VALUE_REL_TOL,
                                  time_rule_max=TIME_RULE_MAX,
                                  memory_rule_max=MEMORY_RULE_MAX),
                       rows=[], summaries=[])
    for cell in cells:
        cell_rows = []
        label = "x".join(map(str, cell))
        for variant in variants:
            python = JAX_PYTHON if variant == "jax" else TORCH_PYTHON
            cfg = dict(variant=variant, cell=list(cell))
            if variant != "jax":
                cfg["constants"] = constants
                cfg["fwd_constants"] = fwd_constants
            print(f"\n{label}/{variant} ...", flush=True)
            row = run_one(python, cfg)
            cell_rows.append(row)
            if "error" in row:
                print(f"  FAILED: {row['error'][:200]}", flush=True)
            else:
                arm = "" if row.get("arm_ok", True) else "  ARM MISMATCH"
                print(f"  cold {row['vcd_cold']:.2f}s warm {row['vcd_warm']:.2f}s"
                      f"  gpu {row['gpu_peak_bytes'] / 2**30:.2f}G"
                      f"  fwd {row.get('bound_fwd_body', '-')}"
                      f"  back {row.get('bound_back_body', '-')}{arm}",
                      flush=True)
        summary = summarize_cell(cell, cell_rows, variants)
        print_cell_summary(summary)
        all_results["rows"].extend(cell_rows)
        all_results["summaries"].append(summary)

    out = os.path.join(RESULTS_DIR, f"p5k6_pgate_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


def print_plan():
    constants, fwd_constants = parse_all_constants()
    print(f"cells:       {[('x'.join(map(str, c))) for c in selected_cells()]}")
    print(f"iterations:  {VCD_ITERATIONS} (seed {VCD_SEED})")
    print(f"pback const: {constants or 'pinned (mbirtorch defaults)'}")
    print(f"pfwd const:  {fwd_constants or 'pinned (mbirtorch defaults)'}")
    print(f"gates:       value {VALUE_REL_TOL:.0e} (vs body and vs jax), "
          f"time {TIME_RULE_MAX}x, memory {MEMORY_RULE_MAX}x")
    for variant in VARIANTS:
        python = JAX_PYTHON if variant == "jax" else TORCH_PYTHON
        print(f"  {variant:<18} {python}")
        print(f"  {'':<18} env {variant_env(variant)}  expect "
              f"(fwd, back) triton = {EXPECTED_BODIES.get(variant)}")
    print(f"results:     {RESULTS_DIR}/p5k6_pgate_{RUN_LABEL}.json")


def _run_worker(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        if cfg["variant"] == "jax":
            result = jax_worker(cfg)
        else:
            result = torch_worker(cfg)
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
