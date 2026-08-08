"""The COMPOSED five-arm gate for kernel-aware view batching -- the seeded
3-iteration warm VCD recon for BOTH geometries at both gate cells, one arm per
subprocess, re-measuring the phase5 gate tables after the batching change.

Why this re-runs at all: changing view batches changes float summation order
and measured peaks (the calibration warning on ``_transient_cols``), so the
composed gates and value envelopes re-run at both cells of both geometries,
with memory re-measured, never inferred.  The baselines this table reads
against are the phase5 composed results: parallel 1.21x / 1.90x of jax time,
cone 1.00x / 1.18x, memory 0.56-0.63x everywhere.

The five TIMED arms per (geometry, cell), each its own process and ruler
(the p5k6_pgate structure):

    torch_kernel        both kernels, default selection
    torch_back_only     back kernel only -- the per-direction decomposition
    torch_body          MBIRTORCH_DISABLE_TRITON=1: the pure torch baseline,
                        which nothing in this change touched, so it doubles
                        as the zero-behavior-change CONTROL against the
                        phase5 body numbers
    torch_body_repeat   the body arm again: the nondeterminism floor
    jax                 mbirjax, the replacement-rule reference

The opt-in era is over (all four kernels default on), so the back-only arm
can no longer be an environment switch: the worker instead seeds the
direction's self-check cache with a decline before the model is built, which
exercises the real fallback path the mixed-selection design names.  The arm
check still verifies what was BOUND, never what was requested.

ARM CHECKS, extended for the batching change.  Each torch arm verifies (1)
the bodies bound per direction against the arm's expectation, with the
launch-key delta as the positive witness that kernels really launched inside
the recon; and (2) the realized view batch per direction, read from the
driver's own ``_effective_view_batch`` at the full-pixel-set inputs, against
the bound body's OWN formula -- the kernel cost model for a kernel body, the
legacy 64-capped charge for a torch body.  A silently-inactive cost
attribute realizes the legacy batch and fails the check rather than shipping
a null result (the phase5 arm-check lesson).

VALUE protocol (the p5ka lesson, now the rule): the cross-framework value
comparison hands ONE shared sinogram artifact to both frameworks, because
the phantom generators differ at ellipsoid-boundary ties across frameworks
AND platforms.  Per (geometry, cell) a generator arm builds phantom ->
sinogram -> .npy once (torch builds it; the choice is arbitrary because
every arm reconstructs the same array), and shared-value arms
(torch_kernel, torch_body, jax) each reconstruct THAT array once.  Weights
are recomputed per arm from the shared sinogram with one formula and one
dtype (float32 everywhere -- both the timed and the shared arms, closing
the asymmetry the original gate carried).  The in-framework kernel-vs-body
value diff comes from the timed arms' samples, read beside the repeat
floor; the timed arms' wall clocks are the gate's timing numbers, the
shared arms' are not.

Run:
    <torch python> kb3_gate.py           on a CUDA node (see kb3_gautschi.sbatch)
    python kb3_gate.py --dry-run         anywhere: print the arm plan
    python kb3_gate.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the arm subprocesses
    KB3_GEOMS=parallel,cone           subset of the geometries
    KB3_CELLS=512,1024                subset of the cells (by view count)
    KB3_ITERATIONS=3                  VCD iterations per recon
    KB3_SKIP_JAX=1                    torch arms only (no ratios, no shared jax)
    KB3_SKIP_FLOOR=1                  drop the repeat (floor) arm
    KB3_SKIP_BACK_ONLY=1              drop the back-only arm
    KB3_SKIP_SHARED=1                 drop the shared-sinogram value arms
    KB3_SMOKE=1 / KB3_DEVICE=cpu      local smoke
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
GEOMETRIES = ("parallel", "cone")
TIMED_ARMS = ("torch_kernel", "torch_back_only", "torch_body",
              "torch_body_repeat", "jax")
SHARED_ARMS = ("shared_generator", "shared_torch_kernel", "shared_torch_body",
               "shared_jax")

# What each torch arm must bind, as (forward is triton, back is triton).
EXPECTED_BODIES = {"torch_kernel": (True, True),
                   "torch_back_only": (False, True),
                   "torch_body": (False, False),
                   "torch_body_repeat": (False, False),
                   "shared_torch_kernel": (True, True),
                   "shared_torch_body": (False, False),
                   "shared_generator": (True, True)}

# The phase5 composed baselines this run's table reads against.
BASELINE_TORCH_OVER_JAX = {("parallel", 512): 1.21, ("parallel", 1024): 1.90,
                           ("cone", 512): 1.00, ("cone", 1024): 1.18}

SMOKE = os.environ.get("KB3_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("KB3_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("KB3_ITERATIONS", "3"))
VCD_SEED = 13
SAMPLE_ROWS = 16          # the p5k6 sample convention, so the numbers compare

VALUE_REL_TOL = 5e-3      # the compile-latitude envelope of the composed gates
TIME_RULE_MAX = 2.0       # the replacement rule: torch within 2x of jax time
MEMORY_RULE_MAX = 1.5     # ... at ~1.5x of jax memory

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """Refuse garbage: every token must name a member of ``allowed`` (the
    slurm --export comma-split lesson)."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = cast(token)
        except ValueError:
            raise ValueError(f"{env_name}: unparsable token {token!r}")
        if value not in allowed:
            raise ValueError(f"{env_name}: {value!r} is not one of "
                             f"{sorted(allowed)}")
        chosen.append(value)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return chosen


def selected_plan():
    if SMOKE:
        cells = [SMOKE_CELL]
    else:
        keep = _strict_subset("KB3_CELLS", {c[0] for c in CELLS}, int)
        cells = [c for c in CELLS if c[0] in keep]
    geometries = _strict_subset("KB3_GEOMS", set(GEOMETRIES))
    arms = list(TIMED_ARMS)
    if os.environ.get("KB3_SKIP_BACK_ONLY", "0") == "1":
        arms.remove("torch_back_only")
    if os.environ.get("KB3_SKIP_FLOOR", "0") == "1":
        arms.remove("torch_body_repeat")
    if os.environ.get("KB3_SKIP_JAX", "0") == "1":
        arms.remove("jax")
    shared = []
    if os.environ.get("KB3_SKIP_SHARED", "0") != "1":
        shared = [a for a in SHARED_ARMS
                  if not (a == "shared_jax"
                          and os.environ.get("KB3_SKIP_JAX", "0") == "1")]
    return geometries, cells, arms, shared


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_kb3_sino_{geometry}_{cell[0]}.npy")


def _sample_path(geometry, cell, arm):
    return os.path.join(RESULTS_DIR, f"_kb3_{geometry}_{cell[0]}_{arm}.npy")


def arm_env(arm):
    """The env that DEFINES a torch arm, set explicitly so nothing inherits
    (the p5k6 rule).  Only the kill switch remains a switch; the back-only
    arm is made in-worker by seeding the self-check cache."""
    if arm.startswith("torch_body") or arm == "shared_torch_body":
        env = {"MBIRTORCH_DISABLE_TRITON": "1"}
    elif arm == "jax" or arm == "shared_jax":
        env = {}
    else:
        env = {"MBIRTORCH_DISABLE_TRITON": "0"}
    return env


def _weights(sinogram):
    """One weighting formula, one dtype, every arm and both frameworks."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


# ── the torch side ────────────────────────────────────────────────────────────
def _build_torch_model(geometry, cell):
    import numpy as np

    import mbirtorch

    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
        model.configure_devices(devices=[DEVICE])
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels)
        model.configure_devices(devices=[DEVICE])
    model.set_params(no_warning=True, verbose=0)
    return model


def _decline_forward_kernel(geometry):
    """The back-only arm's mechanism: seed the forward self-check cache with a
    decline BEFORE the model is built, so create_projectors takes the real
    fallback path (the mixed selection the batching design must serve)."""
    from mbirtorch import kernel_availability as ka

    cache = (ka._PARALLEL_FWD_RESULTS if geometry == "parallel"
             else ka._CONE_FWD_RESULTS)
    for key in ("cuda", "cuda:0", "cpu", "mps"):
        cache[key] = (False, "declined by the gate's back-only arm")


def _launch_key_counts(geometry):
    """Per-kernel launch-key counts (the positive witness): all four kernels
    share one key set and every key leads with its kernel's name."""
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    names = (("pback", "pfwd") if geometry == "parallel" else ("back", "fwd"))
    back = sum(1 for k in _COMPILED_LAUNCH_KEYS
               if isinstance(k, tuple) and k and k[0] == names[0])
    fwd = sum(1 for k in _COMPILED_LAUNCH_KEYS
              if isinstance(k, tuple) and k and k[0] == names[1])
    return back, fwd


def _view_batch_check(model, expected_bodies):
    """The realized view batches at the full-pixel-set inputs, each direction
    checked against the formula of the body EXPECTED to be bound: the kernel
    cost model where a kernel is expected, the legacy 64-capped torch charge
    where the torch body is.  Returns (record, ok)."""
    import mbirtorch

    pf = model.projector_functions
    args = model._view_batch_args()
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    budget = pf._transient_budget_bytes()
    cols = dict(fwd=int(recon_shape[2]), back=int(sinogram_shape[1]))
    bound = dict(fwd=pf._fwd_body_per_dev[0], back=pf._back_body_per_dev[0])
    expected_is_kernel = dict(fwd=expected_bodies[0], back=expected_bodies[1])

    record, ok = {}, True
    for direction in ("fwd", "back"):
        realized = int(pf._effective_view_batch(bound[direction], num_pixels,
                                                cols[direction], args))
        legacy = max(1, min(64, budget
                            // max(1, num_pixels
                                   * model._transient_cols(cols[direction])
                                   * 4)))
        cost = getattr(bound[direction], "_view_batch_cost", None)
        if expected_is_kernel[direction]:
            if cost is None:
                expected = None          # the body check already failed
            else:
                bytes_pv, chunk = cost(num_pixels, cols[direction], args)
                expected = max(1, min(int(chunk), budget // max(1, bytes_pv)))
        else:
            expected = legacy
        record[f"{direction}_view_batch"] = realized
        record[f"{direction}_view_batch_expected"] = expected
        record[f"{direction}_view_batch_legacy"] = int(legacy)
        ok = ok and (expected is not None) and (realized == expected)
    record["num_pixels_full"] = num_pixels
    record["budget_bytes"] = int(budget)
    return record, ok


def torch_worker(cfg):
    """One torch arm: timed (cold + warm vcd) or shared-value (one recon of
    the shared sinogram)."""
    import numpy as np
    import torch

    import mbirtorch

    arm, geometry = cfg["arm"], cfg["geometry"]
    cell = tuple(cfg["cell"])
    if arm == "torch_back_only":
        _decline_forward_kernel(geometry)
    model = _build_torch_model(geometry, cell)

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elif DEVICE == "mps":
            torch.mps.synchronize()

    def peak_bytes():
        return int(torch.cuda.max_memory_allocated()) if DEVICE == "cuda" else 0

    # ── arm check 1: the bodies bound, per direction ─────────────────────────
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    fwd_is_kernel = "triton" in fwd_name
    back_is_kernel = "triton" in back_name
    expected = EXPECTED_BODIES[arm]
    bound_fwd = model.projector_functions._fwd_body_per_dev[0]
    bound_back = model.projector_functions._back_body_per_dev[0]
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE,
                  device_name=(torch.cuda.get_device_name(0)
                               if DEVICE == "cuda" else DEVICE),
                  fwd_body=fwd_name, back_body=back_name,
                  bound_fwd_body=getattr(bound_fwd, "__name__", str(bound_fwd)),
                  bound_back_body=getattr(bound_back, "__name__",
                                          str(bound_back)),
                  fwd_kernel_selected=fwd_is_kernel,
                  back_kernel_selected=back_is_kernel,
                  expected_bodies=list(expected),
                  arm_ok=((fwd_is_kernel, back_is_kernel) == expected),
                  disable_env=os.environ.get("MBIRTORCH_DISABLE_TRITON", ""),
                  vcd_iterations=VCD_ITERATIONS)

    # ── arm check 2: the realized view batches follow the bound bodies ──────
    vb_record, vb_ok = _view_batch_check(model, expected)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    # The chunk constants the kernel modules carry (the gated configuration).
    from mbirtorch import triton_cone, triton_parallel
    result["view_chunks"] = dict(
        pback=triton_parallel.PARALLEL_BACK_VIEW_CHUNK,
        pfwd=triton_parallel.PARALLEL_FWD_VIEW_CHUNK,
        cback=triton_cone.CONE_BACK_VIEW_CHUNK,
        cfwd=triton_cone.CONE_FWD_VIEW_CHUNK)

    keys_before = _launch_key_counts(geometry)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)

    if arm.startswith("shared"):
        sinogram = np.load(_sino_path(geometry, cell))
    else:
        phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(
            recon_shape)
        sinogram = np.asarray(model.forward_project(phantom),
                              dtype=np.float32)
    weights = _weights(sinogram)
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram),
                                               dtype=np.float64))

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
    if not arm.startswith("shared"):
        peak_after_cold = peak_bytes()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = vcd()
        result["vcd_warm"] = time.perf_counter() - t0
        result["gpu_peak_warm_bytes"] = peak_bytes()
        result["gpu_peak_bytes"] = max(peak_after_cold,
                                       result["gpu_peak_warm_bytes"])
    else:
        result["gpu_peak_bytes"] = peak_bytes()
    keys_after = _launch_key_counts(geometry)
    result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
    result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
    _finish(result, out, cfg)
    return result


def jax_worker(cfg):
    """The jax arms: timed (own phantom) or shared-value (the shared .npy)."""
    import numpy as np

    import jax
    import mbirjax

    arm, geometry = cfg["arm"], cfg["geometry"]
    cell = tuple(cfg["cell"])
    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirjax.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirjax.ConeBeamModel(cell, angles,
                                      source_detector_dist=4.0 * num_channels,
                                      source_iso_dist=2.0 * num_channels)
    model.set_params(no_warning=True, verbose=0)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    if arm.startswith("shared"):
        sinogram = np.load(_sino_path(geometry, cell))
    else:
        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = np.asarray(model.forward_project(phantom), dtype=np.float32)
    weights = _weights(sinogram)
    result = dict(cfg, framework="jax", version=f"jax {jax.__version__}",
                  recon_shape=list(recon_shape),
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)),
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
    if not arm.startswith("shared"):
        t0 = time.perf_counter()
        out = vcd()
        result["vcd_warm"] = time.perf_counter() - t0
    result["gpu_peak_bytes"] = max(
        int((d.memory_stats() or {}).get("peak_bytes_in_use", 0))
        for d in jax.devices())
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build the shared input once: phantom -> sinogram -> .npy (torch builds
    it; every arm reconstructs the same array, so the builder is arbitrary)."""
    import numpy as np

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    import mbirtorch

    model = _build_torch_model(geometry, cell)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.ascontiguousarray(np.asarray(model.forward_project(phantom),
                                               dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(_sino_path(geometry, cell), sinogram)
    return dict(cfg, framework="torch", role="generator",
                sinogram_shape=list(sinogram.shape),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)),
                path=_sino_path(geometry, cell))


def _finish(result, out, cfg):
    """The common tail: checksum, the strided row sample for value diffs, and
    the host peak."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result["recon_checksum"] = float(np.sum(np.abs(out), dtype=np.float64))
    step = max(1, out.shape[0] // SAMPLE_ROWS)
    sample_path = _sample_path(cfg["geometry"], cfg["cell"], cfg["arm"])
    np.save(sample_path, out[::step])
    result["sample_path"] = sample_path
    result["sample_step"] = step
    result["peak_rss_bytes"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss


# ── the runner ────────────────────────────────────────────────────────────────
def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_kb3.json")
    out_path = os.path.join(RESULTS_DIR, "_out_kb3.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    python = JAX_PYTHON if "jax" in cfg["arm"] else TORCH_PYTHON
    env = dict(os.environ, **arm_env(cfg["arm"]))
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


def summarize_cell(geometry, cell, rows, requested):
    """The gate verdict for one (geometry, cell)."""
    by_arm = {r.get("arm"): r for r in rows if "error" not in r}
    kernel = by_arm.get("torch_kernel")
    back_only = by_arm.get("torch_back_only")
    body = by_arm.get("torch_body")
    repeat = by_arm.get("torch_body_repeat")
    jax_row = by_arm.get("jax")
    summary = dict(geometry=geometry, cell=list(cell),
                   requested=list(requested),
                   missing=[a for a in requested if a not in by_arm],
                   arms_failed=[a for a, r in by_arm.items()
                                if r.get("arm_ok") is False],
                   vb_failed=[a for a, r in by_arm.items()
                              if r.get("vb_ok") is False])
    print(f"\n===== {geometry} {cell} =====")
    print(f"{'arm':>20}{'cold_s':>9}{'warm_s':>9}{'peak_GB':>9}{'arm':>5}"
          f"{'vb':>4}{'fwd_vb':>8}{'back_vb':>9}")
    for row in rows:
        if row.get("error"):
            print(f"{row.get('arm', '?'):>20}  ERROR: {row['error'][:120]}")
            continue
        peak = row.get("gpu_peak_bytes", 0) / 2 ** 30
        warm = row.get("vcd_warm")
        print(f"{row['arm']:>20}{row.get('vcd_cold', 0):>9.1f}"
              f"{(f'{warm:.1f}' if warm is not None else '-'):>9}"
              f"{peak:>9.2f}"
              f"{('ok' if row.get('arm_ok') else ('-' if row.get('arm_ok') is None else 'FAIL')):>5}"
              f"{('ok' if row.get('vb_ok') else ('-' if row.get('vb_ok') is None else 'FAIL')):>4}"
              f"{str(row.get('fwd_view_batch', '-')):>8}"
              f"{str(row.get('back_view_batch', '-')):>9}")

    def warm(row):
        return row.get("vcd_warm") if row else None

    if kernel and body and warm(kernel) and warm(body):
        summary["kernel_speedup_over_body"] = warm(body) / warm(kernel)
    if kernel and back_only and warm(back_only):
        summary["fwd_marginal_speedup"] = warm(back_only) / warm(kernel)
    if body and back_only and warm(back_only):
        summary["back_speedup_over_body"] = warm(body) / warm(back_only)
    if kernel and jax_row and warm(jax_row):
        summary["torch_over_jax_time"] = warm(kernel) / warm(jax_row)
        jax_peak = jax_row.get("gpu_peak_bytes", 0)
        if jax_peak:
            summary["torch_over_jax_memory"] = (
                kernel.get("gpu_peak_bytes", 0) / jax_peak)
        summary["time_rule_pass"] = (
            summary["torch_over_jax_time"] <= TIME_RULE_MAX)
        if "torch_over_jax_memory" in summary:
            summary["memory_rule_pass"] = (
                summary["torch_over_jax_memory"] <= MEMORY_RULE_MAX)
        baseline = BASELINE_TORCH_OVER_JAX.get((geometry, cell[0]))
        summary["baseline_torch_over_jax"] = baseline

    # Value diffs: in-framework from the timed samples, cross-framework from
    # the shared-sinogram arms, each beside its floor.
    def sample(arm):
        return _sample_path(geometry, cell, arm)

    if kernel and body:
        summary["value_kernel_vs_body"] = _rel_max(sample("torch_kernel"),
                                                   sample("torch_body"))
    if body and repeat:
        summary["value_floor_body_repeat"] = _rel_max(
            sample("torch_body_repeat"), sample("torch_body"))
    if "shared_torch_kernel" in by_arm and "shared_jax" in by_arm:
        summary["shared_value_kernel_vs_jax"] = _rel_max(
            sample("shared_torch_kernel"), sample("shared_jax"))
    if "shared_torch_body" in by_arm and "shared_jax" in by_arm:
        summary["shared_value_body_vs_jax"] = _rel_max(
            sample("shared_torch_body"), sample("shared_jax"))
    if "shared_torch_kernel" in by_arm and "shared_torch_body" in by_arm:
        summary["shared_value_kernel_vs_body"] = _rel_max(
            sample("shared_torch_kernel"), sample("shared_torch_body"))

    for key in ("kernel_speedup_over_body", "fwd_marginal_speedup",
                "back_speedup_over_body", "torch_over_jax_time",
                "torch_over_jax_memory", "baseline_torch_over_jax"):
        if key in summary and summary[key] is not None:
            print(f"  {key}: {summary[key]:.2f}")
    for key in ("time_rule_pass", "memory_rule_pass"):
        if key in summary:
            print(f"  {key}: {summary[key]}")
    for key in ("value_kernel_vs_body", "value_floor_body_repeat",
                "shared_value_kernel_vs_jax", "shared_value_body_vs_jax",
                "shared_value_kernel_vs_body"):
        if summary.get(key) is not None:
            print(f"  {key}: {summary[key]:.2e}"
                  + ("  (over the 5e-3 envelope)"
                     if key != "value_floor_body_repeat"
                     and summary[key] > VALUE_REL_TOL else ""))
    if summary["arms_failed"] or summary["vb_failed"]:
        print(f"  ARM CHECK FAILURES: bodies={summary['arms_failed']} "
              f"view_batches={summary['vb_failed']}")
    return summary


def main():
    geometries, cells, arms, shared = selected_plan()
    if "--dry-run" in sys.argv:
        for geometry in geometries:
            for cell in cells:
                print(geometry, cell, "timed:", arms, "shared:", shared)
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"kb3_gate_{RUN_LABEL}_{stamp}.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"kb3 composed gate on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, cells {[c[0] for c in cells]} -> {out_path}")
    summaries = []
    with open(out_path, "w") as sink:
        for geometry in geometries:
            for cell in cells:
                rows = []
                plan = [dict(arm=a, geometry=geometry, cell=list(cell))
                        for a in arms]
                if shared:
                    plan += [dict(arm=a, geometry=geometry, cell=list(cell))
                             for a in shared]
                for cfg in plan:
                    row = run_one(cfg)
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                summary = summarize_cell(geometry, cell, rows,
                                         arms + list(shared))
                summaries.append(summary)
                sink.write(json.dumps(dict(summary=summary)) + "\n")
                sink.flush()
                sino = _sino_path(geometry, cell)
                if os.path.exists(sino):
                    os.remove(sino)
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        if cfg["arm"] == "shared_generator":
            row = generator_worker(cfg)
        elif "jax" in cfg["arm"]:
            row = jax_worker(cfg)
        else:
            row = torch_worker(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(error=traceback.format_exc()[-2000:], **cfg)
    with open(out_path, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        main()
