"""mg3 -- THE VALUE COMPARISON: the shared-sinogram cross-framework and
cross-configuration residuals at n = 1, 2, 4, which separate the residual terms
BY CONSTRUCTION rather than by tolerance.

The charter is `multigpu_plan.md` §4 (mg3) under the eleven protocols of §3.
Goal 3, plus the coverage role the goldens ruling assigned this instrument: the
golden archives are opt-in everywhere and skip in the nightly's fresh clones
(`nightly_plan.md` §10.4), so mg3's cross-framework columns at n = 1, 2, 4 are
the campaign's half of the port-fidelity question.

THE THREE ARMS AT EVERY (geometry, cell, count), all reconstructing ONE shared
sinogram artifact:

    jax             mbirjax production                       -- the reference
    torch_eager     EAGER-PLAIN: kernels OFF and compile OFF -- the engine floor
    torch_prod      torch production: kernels on, compile on -- what ships

EAGER-PLAIN IS TWO MECHANISMS, NOT ONE, and neither implies the other:

    kernels off   MBIRTORCH_DISABLE_TRITON=1 in the arm's environment.
                  `compile_mode` does NOT disable the kernels: selection is
                  availability-driven (protocol 2, and dp4_flip_gate.py's
                  docstring).
    compile off   `compile_mode='off'` passed to the model CONSTRUCTOR, which
                  sets `TomographyModel.compile_enabled` False
                  (tomography_model.py:182), which `Projectors.__init__` reads
                  as `use_compile` (projectors.py:259) and hands to
                  `maybe_compile`, which then returns each body unwrapped
                  (projectors.py:95).  There is no environment variable for it.

The arm check verifies what was BOUND, never what was requested: a compiled
body is renamed `compiled_<fn>` by `maybe_compile` (projectors.py:127), so the
bound body's own name is the witness in both directions, and
`projectors._COMPILE_ERRORS` is read as well, because a compile that silently
fell back to eager would make a "production" arm secretly eager.

ONE STRUCTURAL FACT, recorded rather than asserted away.  The Triton bodies
carry `_mbirtorch_no_compile = True` (triton_parallel.py:292 and :475,
triton_cone.py:431 and :682), so `maybe_compile` returns them UNWRAPPED: when a
kernel is bound, the projector body is never torch.compiled.  torch.compile
still covers the qGGMRF subset denoiser (denoising.py:250), the diagonal update
direction (tomography_model.py:2015) and the cone damped update
(cone_beam.py:469).  The compile-latitude term this instrument reads therefore
comes from those chains plus the kernels, and §4's claim that the bundle is
compile-dominated rests on kb4's e-6 bound for the kernels.  The optional tier
below is the ordering check on exactly that claim.

THE FIVE VALUE COLUMNS (§4), each reported as BOTH max-rel and norm-rel on the
gathered volumes (kb3's comparison mechanics, extended to the L2 ratio because
a max-rel is a single voxel and a norm-rel is the whole field):

    torch_eager  vs jax          at equal n   the PARTITION-ORDER term plus both
                                              frameworks' remaining latitude
    torch_prod   vs torch_eager  at equal n   the COMPILE-LATITUDE term
    torch_prod   vs jax          at equal n   what a user actually sees
    jax_n        vs jax n=1                   partition order, jax side (a BOUND:
                                              jax is always jitted, so this
                                              carries XLA's per-shape latitude)
    torch_eager  vs its own n=1               partition order, clean -- this is
                                              the column that should land on the
                                              recorded engine floors

TWO EXPECTATIONS, REGISTERED IN ADVANCE (§4), stated beside the columns they
bind in `EXPECTED_PARTITION_FLOORS` and `EXPECTED_COMPILE_LATITUDE` below.

DEPTH SETTLES WHAT TOLERANCE CANNOT.  All arms run at three iterations.
Parallel 1024 adds a TEN-iteration tier, because that cell carries the
documented 6.1e-3 three-iteration residual and its sevenfold decay to 8.8e-4 by
ten.  The decay at n>1 is a HYPOTHESIS this instrument tests, not a recorded
fact; a term that grows with n is a finding.

THE PRICING PROBE (§4, §5 increment 2, §7).  mg3's allocation is committed only
after a probe, because 15 of its arms are eager-plain and no eager-plain
composed time has ever been measured at the 1024 cells.  `MG3_PROBE=1` (or
`PROBE_ONLY = True` below) runs EXACTLY TWO MEASURED ARMS -- torch_eager, n=1,
3 iterations, at each 1024 cell -- and re-derives the full mg3 wall from them by
pricing the harness's OWN plan, arm by arm.  The two untimed generator arms the
probe needs to stage its artifacts are staging, not measurement.

ARM COUNT (§7: "probe, then 45 arms + optional 3"):

    base tier      2 geometries x 2 cells x 3 counts x 3 classes  = 36
    depth-10 tier  parallel 1024 only,      3 counts x 3 classes  =  9
                                                             total  45
    optional tier  parallel 1024, compile ON + kernels OFF, 3 counts =  3
    generators     one shared artifact per (geometry, cell), untimed =  4

ARM ORDER.  Protocol 11 puts every n=1 arm FIRST, so a truncated job still
yields the n=1 validity column; protocol 9 asks for counts blocked-and-reversed
at each cell.  With THREE arm classes an exact count palindrome is impossible
(a palindrome needs even multiplicities, and each count appears three times per
cell), so this harness runs each class's {2, 4} pair REVERSED relative to its
neighbour -- 2,4 | 4,2 | 2,4 -- and records `block_index` / `block_size` on
every row so the analysis can regress a linear drift out of what the reversal
cannot cancel.  See the report's deviation list.

THE GATHER CONTRACT (nt2_local_shard_check.py).  `Shards.gather()` ALREADY
returns numpy (_sharding.py:181).  Re-detaching its result is the recorded
failure that cost the nightly's first 4-GPU trial all 32 of its n>1 rows.
Every host exit here goes through `_to_numpy`, which never re-detaches a gather.

Run:
    <torch python> mg3_value.py              on a 4-GPU node (mg3_gautschi.sbatch)
    MG3_PROBE=1 <torch python> mg3_value.py  the pricing probe (1 GPU is enough)
    python mg3_value.py --dry-run            anywhere: print the arm plan
    python mg3_value.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the arm subprocesses
    MG3_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG3_PROBE=1                       the pricing probe only (two measured arms)
    MG3_GEOMS=parallel,cone           subset of the geometries
    MG3_CELLS=512,1024                subset of the cells (by view count)
    MG3_COUNTS=1,2,4                  subset of the device counts
    MG3_ITERATIONS=3                  base-tier VCD iterations
    MG3_DEEP_ITERATIONS=10            depth tier's VCD iterations
    MG3_WARM_REPEATS=3                warm repeats after the discarded cold pass
    MG3_SKIP_JAX=1                    torch arms only
    MG3_SKIP_DEEP=1                   drop the depth-10 tier
    MG3_ONLY_DEEP=1                   run ONLY the depth-10 tier (plus its
                                      generator); exclusive with SKIP_DEEP
    MG3_DEEP_CLASSES=jax,torch_prod   restrict the depth tier's arm classes
                                      (probe ruling 2026-08-09: the depth-10
                                      EAGER arms are deferred on measured cost
                                      -- 240.5 s per 3-iteration eager recon at
                                      parallel 1024, so ~2.5-4 h for the three
                                      deep eager arms alone)
    MG3_SKIP_COMPILED_NOKERNEL=1      drop the optional ordering tier
    MG3_KEEP_ARTIFACTS=1              keep the multi-GB .npy files after the run
    MG3_SMOKE=1                       the local smoke (tiny cell, few iters)
    MG3_DEVICE=cpu                    smoke device
"""

import hashlib
import json
import os
import platform
import resource
import statistics
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
COUNTS = (1, 2, 4)

# The three arm classes of §4.  `torch_eager` is EAGER-PLAIN: kernels off AND
# compile off (see the module docstring -- two mechanisms, neither implying the
# other).
ARM_CLASSES = ("torch_eager", "torch_prod", "jax")

# The depth tier: §4 scopes it to parallel 1024, the cell that carries the
# documented 6.1e-3 three-iteration residual and its decay to 8.8e-4 by ten.
# §8 records the recommendation to keep it scoped and add cone depth only if
# cone's depth-3 readings surprise.
DEEP_TIER = ("parallel", 1024)

# THE OPTIONAL ORDERING TIER, ON by default.  compile ON, kernels OFF, at
# parallel 1024.  Its purpose is one ordering check and nothing else: §4 claims
# the torch_prod-vs-torch_eager bundle reads the COMPILER to leading order,
# because kb4 bounded the kernels' own contribution at the e-6 parity class.
# This tier isolates the compiler alone at n>1.  If
# (compiled_nokernel vs torch_eager) is close to (torch_prod vs torch_eager),
# the bundle is compile-dominated and §4's reading stands; if it is far below
# it, the kernels carry the term at n>1 and the attribution changes.
RUN_COMPILED_NOKERNEL = True

SMOKE = os.environ.get("MG3_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG3_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG3_ITERATIONS", "1" if SMOKE else "3"))
DEEP_ITERATIONS = int(os.environ.get("MG3_DEEP_ITERATIONS",
                                     "2" if SMOKE else "10"))
VCD_SEED = 13             # kb3's seed, so these rows compare to its baselines
SAMPLE_ROWS = 16          # kb3's / p5k6's / mg1's sample convention
WARM_REPEATS = int(os.environ.get("MG3_WARM_REPEATS", "1" if SMOKE else "3"))

# EXPECTATION 1 (§4, registered in advance).  The partition-order columns --
# `jax_n_vs_n1` and `eager_n_vs_n1` -- should TRACK the engine floors, in the
# 5e-4 class, growing mildly with n.  These are the dp4 flip-gate readings
# (job 14973466) at the (256, 64, 64) probe cell, which the ks1 isolation
# matrix reproduces to three significant figures.  They are a CLASS, not a
# gate: per-cell tolerances calibrate against the same cell's dividing-case
# parity floor (protocol 8), so no number here is copied onto another cell.
EXPECTED_PARTITION_FLOORS = {("parallel", 2): 4.47e-04,
                             ("parallel", 4): 9.49e-04,
                             ("cone", 2): 4.57e-04,
                             ("cone", 4): 4.95e-04}
# EXPECTATION 2 (§4, registered in advance).  The compile-latitude column --
# `prod_vs_eager` -- should sit where the composed kernel-versus-body columns
# sit at n=1: low e-3 in parallel, e-4 in cone, decaying with depth.  THE DECAY
# AT n>1 IS THE HYPOTHESIS THIS INSTRUMENT TESTS, NOT A RECORDED FACT.  A term
# that instead GROWS with n is a finding, and it would redirect the tuning.
EXPECTED_COMPILE_LATITUDE = {"parallel": 5e-3, "cone": 5e-4}
# The documented cross-framework residual at parallel 1024 from the shared
# sinogram (`phase5_findings.md`), and its decay by ten iterations.  The depth
# tier exists to re-read exactly this pair at n>1.
DOCUMENTED_CROSS_FRAMEWORK = {3: 6.1e-3, 10: 8.8e-4}

# ── the pricing probe's model (§4, §7) ────────────────────────────────────────
# The probe measures ONE thing: eager-plain composed time at n=1 at each 1024
# cell.  Everything else in the projection is a NAMED scaling assumption, so a
# reader can see exactly which numbers are measured and which are assumed.
PROBE_ONLY = os.environ.get("MG3_PROBE", "0") == "1"
PROBE_CELL_VIEWS = 1024
PROBE_COUNT = 1
# Work scales with sinogram elements between cells of one geometry.
# COUNT_SCALE is the §2 PRE-KERNEL torch matrix, used as the PESSIMISTIC bound
# §7 prescribes; it is cell-dependent because the prior's own shape is (at 512
# n=4 collapsed to 2.7x, at 1024 it did not).  Cone has no timed n>1 prior at
# all, so it borrows parallel's -- recorded here rather than hidden.
COUNT_SCALE = {512: {1: 1.00, 2: 0.82, 4: 2.73},
               1024: {1: 1.00, 2: 0.74, 4: 0.66}}
# Non-eager classes (torch_prod, jax) priced at this fraction of the measured
# eager-plain time.  1.0 is the strict UPPER bound: eager-plain is the slowest
# class and is the whole reason the probe exists, so any real speedup only
# shortens the job.  The probe also prints the sensitivity at the fractions
# below.
NON_EAGER_FRACTION = 1.0
NON_EAGER_SENSITIVITY = (1.0, 0.5, 0.25)
# The one cost the probe CANNOT see: eager-plain neither JITs nor compiles a
# Triton kernel, so its cold pass carries no compile.  Every other class does,
# and protocol 7 records that Triton compiles PER DEVICE, so an n-device cold
# pass pays n compiles per shape.  This allowance is added to each compiling
# arm's cold pass, per device.  It is an ASSUMPTION, from §7's 2-5 minute fixed
# cost; the probe reports it separately so it can be argued with.
COLD_COMPILE_ALLOWANCE_S = 60.0
# Walltime the sbatch requests; the probe says whether the projection fits.
SBATCH_WALLTIME_H = 8.0

# The throttle rule (protocol 11 / nightly_plan.md §10.5): sw_power_cap at a
# normal temperature is the boost governor -- recorded and KEPT.  A row is
# marked for re-run only when it is hot AND its clock is depressed.
HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

RESULTS_DIR = os.environ.get(
    "MG3_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """Refuse garbage: every token must name a member of ``allowed`` (kb3's
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
        geometries = _strict_subset("MG3_GEOMS", set(GEOMETRIES)) \
            if os.environ.get("MG3_GEOMS", "").strip() else ["parallel"]
    else:
        keep = _strict_subset("MG3_CELLS", {c[0] for c in CELLS}, int)
        cells = [c for c in CELLS if c[0] in keep]
        geometries = [g for g in GEOMETRIES
                      if g in _strict_subset("MG3_GEOMS", set(GEOMETRIES))]
    if SMOKE and not os.environ.get("MG3_COUNTS", "").strip():
        # The env pin is a CUDA-only mechanism (the policy short-circuits at
        # `visible < 2`, tomography_model.py:894), so a pinned n>1 torch arm on
        # CPU would silently measure n=1.  The smoke therefore runs n=1 only.
        counts = [1]
    else:
        chosen = _strict_subset("MG3_COUNTS", set(COUNTS), int)
        counts = [n for n in COUNTS if n in chosen]
    return geometries, cells, counts


def _deep_cell(geometries, cells):
    """The (geometry, cell) the depth tier runs at, or None.

    The geometry must be among the SELECTED ones, not just named in DEEP_TIER:
    the generators are built per selected (geometry, cell), so a depth tier
    scheduled outside the selection would have no artifact to read and every one
    of its arms would fail on a missing file.  Off the cluster the smoke has
    only its own tiny cell, so the tier follows it there rather than vanishing
    -- the point of the smoke is that this code path RUNS."""
    if SMOKE:
        return (geometries[0], cells[0]) if (geometries and cells) else None
    geometry, views = DEEP_TIER
    if geometry not in geometries:
        return None
    for cell in cells:
        if cell[0] == views:
            return geometry, cell
    return None


# ── staged-artifact mechanics (protocol 5) ────────────────────────────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg3_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id):
    return os.path.join(RESULTS_DIR, f"_mg3_sample_{arm_id}.npy")


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, chunked: at the 1024 cells the artifact is a
    ~4 GB array and a corrupt Lustre read is a recorded failure mode
    (protocol 5)."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(x):
    """The ONE host exit.  ``Shards.gather()`` already returns numpy
    (_sharding.py:181) -- the nt2 shard check's recorded failure class is
    re-detaching that result -- so a gather is never followed by ``.detach()``."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    gather = getattr(x, "gather", None)
    if callable(gather) and hasattr(x, "placement"):
        return gather()                       # ALREADY numpy: do not re-detach
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weights(sinogram):
    """One weighting formula, one dtype, every arm and both frameworks (kb3's
    rule -- the asymmetry the original gate carried is closed)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


# ── the GPU health sample (protocol 11) ───────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks (SM + memory), temps (core + HBM), and active throttle
    reasons, via nvidia-smi.  ``[]`` when nvidia-smi is unavailable.  The fields
    mirror the nightly's own sample so the two are comparable."""
    for fields in (_GPU_FIELDS_FULL, _GPU_FIELDS_MIN):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + fields,
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
        except Exception:                                         # noqa: BLE001
            return []
        if proc.returncode != 0:
            continue
        full = fields is _GPU_FIELDS_FULL
        out = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            entry = {"index": _gi(parts[0]), "sm_mhz": _gi(parts[1])}
            if full and len(parts) >= 9:
                entry["mem_mhz"] = _gi(parts[2])
                entry["temp_c"] = _gi(parts[3])
                entry["mem_temp_c"] = _gi(parts[4])
                entry["throttle"] = [name for name, value
                                     in zip(_THROTTLE_NAMES, parts[5:9])
                                     if value.lower() == "active"]
            else:
                entry["temp_c"] = _gi(parts[2])
            out.append(entry)
        if out:
            return out
    return []


def worst_health(samples):
    """Per-GPU worst case across samples: MIN clocks, MAX temps, the union of
    throttle reasons.  A single post-run snapshot misses the dips."""
    agg = {}
    for snapshot in samples:
        for gpu in snapshot:
            index = gpu.get("index")
            slot = agg.setdefault(index, {"index": index, "sm_mhz": None,
                                          "mem_mhz": None, "temp_c": None,
                                          "mem_temp_c": None, "throttle": []})
            for key in ("sm_mhz", "mem_mhz"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else min(slot[key],
                                                                    value)
            for key in ("temp_c", "mem_temp_c"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else max(slot[key],
                                                                    value)
            for reason in gpu.get("throttle", []):
                if reason not in slot["throttle"]:
                    slot["throttle"].append(reason)
    return [agg[k] for k in sorted(agg, key=lambda i: (i is None, i))]


def row_is_hot(health):
    """Hot by TEMPERATURE alone (the reliable signal).  Not a re-run verdict on
    its own: protocol 11's re-run rule needs a depressed clock too, and
    sw_power_cap at a normal temperature is the boost governor -- recorded and
    KEPT."""
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── the torch side ────────────────────────────────────────────────────────────
def _compile_mode(arm_class):
    """`torch_eager` is the ONLY class that turns the compiler off, and it does
    so through the model CONSTRUCTOR: there is no environment variable for
    compile (see the module docstring)."""
    return "off" if arm_class == "torch_eager" else "auto"


def _build_torch_model(geometry, cell, arm_class, pin_devices=None):
    """The model.  ``pin_devices`` is a device LIST for the smoke's CPU paths
    only; on CUDA nothing is configured here, because protocol 1 pins through
    MBIRTORCH_NUM_DEVICES and an explicit configure_devices call would take the
    explicit branch and skip the preflight."""
    import numpy as np

    import mbirtorch

    num_views, _, num_channels = cell
    compile_mode = _compile_mode(arm_class)
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles,
                                            compile_mode=compile_mode)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels,
                                        compile_mode=compile_mode)
    if pin_devices is not None:
        model.configure_devices(devices=pin_devices)
    model.set_params(no_warning=True, verbose=0)
    return model


def _launch_key_counts(geometry):
    """Per-kernel launch-key counts (kb3's positive witness): all four kernels
    share one key set and every key leads with its kernel's name."""
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    names = (("pback", "pfwd") if geometry == "parallel" else ("back", "fwd"))
    back = sum(1 for k in _COMPILED_LAUNCH_KEYS
               if isinstance(k, tuple) and k and k[0] == names[0])
    fwd = sum(1 for k in _COMPILED_LAUNCH_KEYS
              if isinstance(k, tuple) and k and k[0] == names[1])
    return back, fwd


def _view_batch_static(model, expect_kernels):
    """kb3's realized-view-batch check, PER DEVICE (protocol 3: the batch budget
    divides by the count while the kernel cost models do not all follow, so the
    realized batch is not invariant in n).  Computed at the full-pixel-set
    inputs against the formula of the body EXPECTED to be bound: the kernel cost
    model where a kernel is expected, the legacy 64-capped torch charge where a
    torch body is."""
    import mbirtorch

    pf = model.projector_functions
    args = model._view_batch_args()
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    budget = pf._transient_budget_bytes()
    n_dev = model.sino_placement.n_devices
    cols = dict(fwd=int(-(-recon_shape[2] // n_dev)),
                back=int(sinogram_shape[1]))
    bodies = dict(fwd=pf._fwd_body_per_dev, back=pf._back_body_per_dev)
    expected_is_kernel = dict(fwd=expect_kernels[0], back=expect_kernels[1])

    record, ok = {}, True
    for direction in ("fwd", "back"):
        realized, expected_all = [], []
        for body in bodies[direction]:
            value = int(pf._effective_view_batch(body, num_pixels,
                                                 cols[direction], args))
            legacy = max(1, min(64, budget
                                // max(1, num_pixels
                                       * model._transient_cols(cols[direction])
                                       * 4)))
            cost = getattr(body, "_view_batch_cost", None)
            if expected_is_kernel[direction]:
                if cost is None:
                    expected = None          # the body check already failed
                else:
                    bytes_pv, chunk = cost(num_pixels, cols[direction], args)
                    expected = max(1, min(int(chunk),
                                          budget // max(1, bytes_pv)))
            else:
                expected = int(legacy)
            realized.append(value)
            expected_all.append(expected)
            ok = ok and (expected is not None) and (value == expected)
        record[f"{direction}_view_batch_per_device"] = realized
        record[f"{direction}_view_batch_expected_per_device"] = expected_all
    record["num_pixels_full"] = num_pixels
    record["budget_bytes"] = int(budget)
    return record, ok


def _compile_witness(model, arm_class, expect_compile):
    """ARM CHECK: what the COMPILER actually bound, per direction per device.

    `maybe_compile` renames its wrapper `compiled_<fn>` (projectors.py:127), so
    the bound body's own name says whether torch.compile wrapped it.  Two facts
    make the expectation direction-dependent rather than uniform:

      * the Triton bodies carry `_mbirtorch_no_compile = True`, so a bound
        kernel is NEVER torch.compiled even when compile is on;
      * `_COMPILE_ERRORS` records a compile that fell back to eager -- a
        production arm with a populated table is secretly eager, which is
        exactly the "verify what was BOUND" failure this check exists to catch.
    """
    from mbirtorch import projectors

    pf = model.projector_functions
    names = {"fwd": [getattr(b, "__name__", str(b))
                     for b in pf._fwd_body_per_dev],
             "back": [getattr(b, "__name__", str(b))
                      for b in pf._back_body_per_dev]}
    wrapped = {d: [n.startswith("compiled_") for n in names[d]]
               for d in names}
    is_kernel = {d: ["triton" in n for n in names[d]] for d in names}
    record = {"bound_body_names": names,
              "bound_body_is_compiled": wrapped,
              "bound_body_is_kernel": is_kernel,
              "compile_mode": model.compile_mode,
              "compile_enabled": bool(model.compile_enabled),
              "compile_errors": dict(projectors._COMPILE_ERRORS),
              "compile_cache_entries": len(projectors._COMPILE_CACHE)}
    # A projector body is torch.compiled exactly when compile is on AND the
    # bound body is not a kernel.  That predicate is the expectation.
    ok = (bool(model.compile_enabled) == expect_compile)
    for direction in names:
        for kernel, comp in zip(is_kernel[direction], wrapped[direction]):
            ok = ok and (comp == (expect_compile and not kernel))
    record["compile_ok"] = bool(ok and not projectors._COMPILE_ERRORS)
    return record


def torch_worker(cfg):
    """One torch arm: one DISCARDED cold pass, then WARM_REPEATS warm repeats
    (protocols 7 and 9)."""
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    arm_class, n_dev, depth = cfg["arm_class"], cfg["n_dev"], cfg["iterations"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = None if cuda else [DEVICE]
    model = _build_torch_model(geometry, cell, arm_class,
                               pin_devices=pin_devices)

    # What this arm must BIND.  Kernels are a CUDA-only availability, so the
    # expectation follows the backend, never the request.
    expect_kernels = (cuda and arm_class == "torch_prod",) * 2
    expect_compile = (arm_class != "torch_eager")

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=depth, warm_repeats=WARM_REPEATS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"))
    # Protocol 6: the calibration mode owns max_memory_allocated, and mg2 is the
    # only job that sets it.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))

    # ── arm check: the bodies bound, per direction (kb3's witnesses) ──────────
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_kernel_selected="triton" in fwd_name,
                  back_kernel_selected="triton" in back_name,
                  expected_bodies=list(expect_kernels))
    result["bodies_ok"] = ((result["fwd_kernel_selected"],
                            result["back_kernel_selected"])
                           == expect_kernels)
    # ── arm check: what the COMPILER bound (the eager-plain half of §4) ──────
    result.update(_compile_witness(model, arm_class, expect_compile))
    result["expected_compile"] = expect_compile

    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)

    # ── the shared sinogram artifact (protocol 5), md5-verified ──────────────
    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    result["sino_md5"] = actual_md5
    result["sino_md5_ok"] = (actual_md5 == expected_md5)
    if not result["sino_md5_ok"]:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5} (the Lustre "
                           f"corrupt-read failure mode, protocol 5)")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram),
                                               dtype=np.float64))

    def peaks():
        if not cuda:
            return []
        return [int(torch.cuda.max_memory_allocated(d))
                for d in model.sino_placement.devices]

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=depth,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(recon)

    keys_before = _launch_key_counts(geometry) if cuda else (0, 0)
    health = [sample_gpu_health()]

    # ── the cold pass, DISCARDED from the warm statistics (protocol 7) ───────
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the warm repeats (protocol 9) ────────────────────────────────────────
    if cuda:
        for device in model.sino_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        warm.append(time.perf_counter() - start)
        health.append(sample_gpu_health())
    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_min"] = min(warm)
    result["vcd_warm_max"] = max(warm)
    result["vcd_warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    # ── arm check: the REALIZED device list after the timed call (protocol 1) ─
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev) if cuda else \
        (len(realized) == len(pin_devices or [DEVICE]))

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        launched = (result["back_launch_keys_delta"] > 0
                    and result["fwd_launch_keys_delta"] > 0)
        # The witness cuts BOTH ways: a kernel arm must show launches, and a
        # kernels-off arm must show none.
        result["kernels_launched_ok"] = (launched == expect_kernels[0])

    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    # ── arm check, POST-RUN: did a compile fall back to eager DURING the run? ─
    # `maybe_compile`'s guard is lazy -- it compiles at the FIRST call and, on
    # any exception there, retries eagerly and rebinds permanently
    # (projectors.py:111-123).  So a fallback only becomes visible after the
    # cold pass, and the pre-run witness above cannot see it.  A production arm
    # that fell back is secretly eager, which would silently move this
    # instrument's whole compile-latitude column.
    from mbirtorch import projectors as _projectors
    result["compile_errors_after_run"] = dict(_projectors._COMPILE_ERRORS)
    result["compile_fallback_free_ok"] = not _projectors._COMPILE_ERRORS
    result["compile_ok"] = bool(result["compile_ok"]
                                and result["compile_fallback_free_ok"])

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


# ── the jax side ──────────────────────────────────────────────────────────────
def jax_worker(cfg):
    """One jax arm, same protocol, same job, same node, same shared sinogram, so
    every column is same-run."""
    import numpy as np

    # mbirjax BEFORE jax: on a CPU-only host mbirjax's _setup_devices sets the
    # host-platform device count, and importing jax first freezes it at 1 (the
    # warning the local smoke printed).  Irrelevant on GPU, free to get right.
    import mbirjax
    import jax

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev, depth = cfg["n_dev"], cfg["iterations"]
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
    # mbirjax has no environment pin: configure_devices(int n) IS its pin (the
    # p4 pattern), and the realized count is asserted below exactly as the torch
    # arms assert theirs.
    model.configure_devices(n_dev)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))

    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5} (protocol 5)")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)

    # The REALIZED list, read from mbirjax's own derived view of the layout
    # (`_set_device_layout`: "mesh / shard_devices are derived views of them").
    # No silent fallback to `jax.devices()[:n]`: that would make the assertion
    # trivially true if the attribute ever moved, which is the arm-check failure
    # class this campaign keeps re-learning.
    shard_devices = getattr(model, "shard_devices", None)
    if shard_devices is None:
        raise RuntimeError("mbirjax model exposes no `shard_devices`; the "
                           "realized device count cannot be asserted, so this "
                           "arm is not a valid pinned arm (protocol 1).")
    realized = [str(d) for d in shard_devices]
    result = dict(cfg, framework="jax", version=f"jax {jax.__version__}",
                  recon_shape=list(recon_shape), vcd_iterations=depth,
                  warm_repeats=WARM_REPEATS,
                  pin_mechanism="mbirjax configure_devices(n)",
                  visible_devices=len(jax.devices()),
                  realized_devices=realized, realized_n_devices=len(realized),
                  devices_ok=(len(realized) == n_dev),
                  sino_md5=actual_md5, sino_md5_ok=True,
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)))

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=depth,
                                   stop_threshold_change_pct=0.0)
        return np.asarray(recon)

    health = [sample_gpu_health()]
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    health.append(sample_gpu_health())
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        warm.append(time.perf_counter() - start)
        health.append(sample_gpu_health())
    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_min"] = min(warm)
    result["vcd_warm_max"] = max(warm)
    result["vcd_warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    per_device = []
    for device in jax.devices():
        stats = device.memory_stats() or {}
        per_device.append(int(stats.get("peak_bytes_in_use", 0)))
    result["gpu_peak_per_device"] = per_device
    result["gpu_peak_bytes"] = max(per_device, default=0)
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build ONE shared sinogram per (geometry, cell): phantom -> sinogram ->
    .npy, plus its md5 sidecar (protocol 5).  Torch builds it; the choice is
    arbitrary because every arm reconstructs the same array.  Pinned to one
    device so the generator cannot itself become a multi-device run, and built
    in the PRODUCTION configuration so the artifact matches kb3's."""
    import numpy as np

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    model = _build_torch_model(geometry, cell, "torch_prod",
                               pin_devices=[DEVICE])
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _sino_path(geometry, cell)
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(geometry, cell), "w") as handle:
        handle.write(digest + "\n")
    # The round trip is verified HERE too, not only by the consumers: a write
    # that did not land is cheaper to find at the generator than 12 arms later.
    reread = np.load(path, mmap_mode="r")
    return dict(cfg, framework="torch", role="generator", path=path,
                sino_md5=digest, sino_md5_ok=(_md5(path) == digest),
                sinogram_shape=list(sinogram.shape),
                reread_shape=list(reread.shape),
                recon_shape=list(recon_shape),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


def _finish(result, out, cfg):
    """The common tail: checksum, the strided row sample the value columns are
    computed from, and the host peak."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result["recon_checksum"] = float(np.sum(np.abs(out), dtype=np.float64))
    step = max(1, out.shape[0] // SAMPLE_ROWS)
    path = _sample_path(cfg["arm_id"])
    np.save(path, out[::step])
    result["sample_path"] = path
    result["sample_step"] = step
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ── the runner ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits (kb3's
    rule).

    Protocol 1: a pinned arm pins ONLY through MBIRTORCH_NUM_DEVICES, which
    keeps the model on the automatic branch where the preflight still runs; an
    explicit configure_devices call would take the explicit branch and get no
    preflight, so the two are not interchangeable.

    Protocol 2: an arm that intends the plain torch engine sets
    MBIRTORCH_DISABLE_TRITON=1.  `compile_mode` does not disable the kernels.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # protocol 6
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    if cfg["framework"] == "torch":
        kernels_off = cfg["arm_class"] in ("torch_eager",
                                           "torch_compiled_nokernel")
        env["MBIRTORCH_DISABLE_TRITON"] = "1" if kernels_off else "0"
        if cfg.get("n_dev") and DEVICE == "cuda":
            env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg3_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg3_{cfg['arm_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    python = JAX_PYTHON if cfg["framework"] == "jax" else TORCH_PYTHON
    start = time.perf_counter()
    proc = subprocess.run([python, "-u", os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=arm_env(cfg))
    subprocess_wall = time.perf_counter() - start
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(cfg, error=f"worker exited {proc.returncode}")
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    # Protocol 7: the total subprocess wall is recorded even though the warm
    # protocol discards it, because §6's cadence decision and §7's estimates
    # need exactly the costs the warm protocol throws away.
    row["subprocess_wall_s"] = subprocess_wall
    return row


def _arm(framework, arm_class, geometry, cell, n_dev, iterations, tier,
         block_tier=None):
    """One arm.  ``tier`` is the SCHEDULING tier (it names the arm, and the
    sbatch's split guidance skips by it); ``block_tier`` is the COMPARISON
    block the value columns are computed within.  They differ for exactly one
    class: the optional compiled-kernels-off arms are scheduled as their own
    tier but must be compared against the base tier's eager-plain arms at the
    same cell, depth and count -- otherwise their column has no partner and can
    never be computed (caught by the local smoke)."""
    tag = f"{tier}_{arm_class}_n{n_dev}" if n_dev else f"{tier}_{arm_class}"
    return dict(framework=framework, arm_class=arm_class, tier=tier,
                block_tier=block_tier or tier,
                geometry=geometry, cell=list(cell), n_dev=n_dev,
                iterations=iterations,
                arm_id=f"{geometry}_{cell[0]}_{tag}")


def _classes(skip_jax):
    return [c for c in ARM_CLASSES if not (c == "jax" and skip_jax)]


def _reversed_block(classes, counts):
    """Protocol 9's blocked-and-reversed order, as far as three arm classes
    allow.  Each class's {2, 4} pair runs REVERSED relative to its neighbour
    (2,4 | 4,2 | 2,4).  An exact count palindrome is impossible here: a
    palindrome needs even multiplicities and each count appears three times per
    cell, once per class.  Every row carries `block_index` / `block_size` so the
    analysis can regress out what the reversal cannot cancel."""
    order = []
    for index, arm_class in enumerate(classes):
        pair = list(counts) if index % 2 == 0 else list(reversed(counts))
        for n in pair:
            order.append((arm_class, n))
    return order


def build_plan(geometries, cells, counts):
    """The arm plan, in JOB ORDER.  PHASE 0 is every generator and every n=1
    arm (protocol 11: a truncated job still yields the n=1 validity column);
    PHASE 1 is the reversed n>1 blocks, base tier first, then the depth tier,
    then the optional ordering tier -- so a truncation loses the least
    load-bearing tier first."""
    skip_jax = os.environ.get("MG3_SKIP_JAX", "0") == "1"
    skip_deep = os.environ.get("MG3_SKIP_DEEP", "0") == "1"
    only_deep = os.environ.get("MG3_ONLY_DEEP", "0") == "1"
    if skip_deep and only_deep:
        raise ValueError("MG3_SKIP_DEEP and MG3_ONLY_DEEP are exclusive")
    skip_optional = (not RUN_COMPILED_NOKERNEL
                     or os.environ.get("MG3_SKIP_COMPILED_NOKERNEL",
                                       "0") == "1"
                     or only_deep)
    classes = _classes(skip_jax)
    # The depth tier's classes are separately selectable (probe ruling,
    # 2026-08-09): the measured eager-plain cost at the 1024 cells prices the
    # three depth-10 eager arms at 2.5-4 h alone, and the documented decay
    # target -- the cross-framework residual -- needs jax and production at
    # depth, not eager.  The deferred arms stay one env token away.
    deep_classes = [c for c in _strict_subset("MG3_DEEP_CLASSES", set(classes))
                    if c in classes] or classes
    deep_classes = [c for c in classes if c in deep_classes]
    higher = [n for n in counts if n != 1]
    deep = None if skip_deep else _deep_cell(geometries, cells)
    phase0, phase1 = [], []

    def framework_of(arm_class):
        return "jax" if arm_class == "jax" else "torch"

    # -- generators: one shared artifact per (geometry, cell) ------------------
    # Under MG3_ONLY_DEEP the only artifact any arm reads is the deep cell's.
    gen_coords = ([deep] if only_deep and deep else
                  [(g, c) for g in geometries for c in cells])
    for geometry, cell in gen_coords:
        phase0.append(dict(framework="torch", arm_class="generator",
                           tier="generator", geometry=geometry,
                           cell=list(cell), n_dev=None, iterations=0,
                           arm_id=f"{geometry}_{cell[0]}_generator"))

    # -- PHASE 0: every n=1 arm, base tier then depth tier then optional -------
    if 1 in counts:
        if not only_deep:
            for geometry in geometries:
                for cell in cells:
                    for arm_class in classes:
                        phase0.append(_arm(framework_of(arm_class), arm_class,
                                           geometry, cell, 1, VCD_ITERATIONS,
                                           "base"))
        if deep:
            for arm_class in deep_classes:
                phase0.append(_arm(framework_of(arm_class), arm_class, deep[0],
                                   deep[1], 1, DEEP_ITERATIONS, "deep"))
        if not skip_optional and deep:
            phase0.append(_arm("torch", "torch_compiled_nokernel", deep[0],
                               deep[1], 1, VCD_ITERATIONS, "ordering",
                               block_tier="base"))

    # -- PHASE 1: the reversed n>1 blocks --------------------------------------
    if not only_deep:
        for geometry in geometries:
            for cell in cells:
                for arm_class, n in _reversed_block(classes, higher):
                    phase1.append(_arm(framework_of(arm_class), arm_class,
                                       geometry, cell, n, VCD_ITERATIONS,
                                       "base"))
    if deep:
        for arm_class, n in _reversed_block(deep_classes, higher):
            phase1.append(_arm(framework_of(arm_class), arm_class, deep[0],
                               deep[1], n, DEEP_ITERATIONS, "deep"))
    if not skip_optional and deep:
        for n in higher:
            phase1.append(_arm("torch", "torch_compiled_nokernel", deep[0],
                               deep[1], n, VCD_ITERATIONS, "ordering",
                               block_tier="base"))
    return phase0, phase1


def probe_plan(geometries, cells):
    """THE PRICING PROBE (§4): EXACTLY TWO MEASURED ARMS -- torch_eager, n=1,
    base depth, at each 1024 cell -- plus the two untimed generator arms that
    stage their artifacts.  About ten minutes on one GPU, and the mg3 wall is
    re-derived from it before the allocation is committed."""
    probe_cells = [c for c in cells if c[0] == PROBE_CELL_VIEWS] or cells[-1:]
    staging, measured = [], []
    for geometry in geometries:
        for cell in probe_cells:
            staging.append(dict(framework="torch", arm_class="generator",
                                tier="generator", geometry=geometry,
                                cell=list(cell), n_dev=None, iterations=0,
                                arm_id=f"{geometry}_{cell[0]}_generator"))
            measured.append(_arm("torch", "torch_eager", geometry, cell,
                                 PROBE_COUNT, VCD_ITERATIONS, "probe"))
    return staging, measured


# ── the pricing projection ────────────────────────────────────────────────────
def _sino_elements(cell):
    return int(cell[0]) * int(cell[1]) * int(cell[2])


def _count_scale(cell, n):
    table = COUNT_SCALE.get(int(cell[0])) or COUNT_SCALE[1024]
    return table.get(int(n), max(table.values()))


def price_arm(cfg, probe_by_geometry, fixed_s, non_eager_fraction):
    """Price ONE planned arm from the probe's measurement plus the named
    assumptions at the top of this file.  Returns (seconds, detail)."""
    if cfg["arm_class"] == "generator":
        # A generator is one forward projection: cheap next to a 3-iteration
        # composed recon, and priced as one pass at the eager rate.
        base = probe_by_geometry.get(cfg["geometry"])
        if base is None:
            return 0.0, {}
        passes = 1.0
        class_fraction, compiles = 1.0, False
    else:
        base = probe_by_geometry.get(cfg["geometry"]) \
            or next(iter(probe_by_geometry.values()), None)
        if base is None:
            return 0.0, {}
        passes = 1.0 + WARM_REPEATS
        class_fraction = (1.0 if cfg["arm_class"] == "torch_eager"
                          else non_eager_fraction)
        compiles = cfg["arm_class"] != "torch_eager"
    cell = tuple(cfg["cell"])
    cell_factor = _sino_elements(cell) / max(1, base["sino_elements"])
    count_factor = _count_scale(cell, cfg.get("n_dev") or 1)
    depth_factor = ((cfg["iterations"] / VCD_ITERATIONS)
                    if cfg.get("iterations") else 1.0)
    per_pass = (base["warm"] * cell_factor * count_factor * depth_factor
                * class_fraction)
    cold_extra = (base["cold"] - base["warm"]) * cell_factor * count_factor \
        * depth_factor * class_fraction
    compile_allowance = (COLD_COMPILE_ALLOWANCE_S * (cfg.get("n_dev") or 1)
                         if compiles else 0.0)
    seconds = fixed_s + passes * per_pass + max(0.0, cold_extra) \
        + compile_allowance
    return seconds, dict(cell_factor=cell_factor, count_factor=count_factor,
                         depth_factor=depth_factor,
                         class_fraction=class_fraction, per_pass_s=per_pass,
                         compile_allowance_s=compile_allowance)


def project_wall(probe_rows, geometries, cells, counts):
    """Re-derive the full mg3 wall from the probe, by pricing the harness's OWN
    plan arm by arm -- so the projection cannot drift from what will run."""
    probe_by_geometry, fixed_candidates = {}, []
    for row in probe_rows:
        if row.get("error") or row.get("arm_class") != "torch_eager":
            continue
        cell = tuple(row["cell"])
        timed = row.get("vcd_cold", 0.0) + sum(row.get("vcd_warm_all") or [])
        fixed_candidates.append(max(0.0, row.get("subprocess_wall_s", 0.0)
                                    - timed))
        probe_by_geometry[row["geometry"]] = {
            "warm": row["vcd_warm"], "cold": row["vcd_cold"],
            "sino_elements": _sino_elements(cell), "cell": list(cell)}
    if not probe_by_geometry:
        return None
    fixed_s = max(fixed_candidates) if fixed_candidates else 0.0

    phase0, phase1 = build_plan(geometries, cells, counts)
    plan = phase0 + phase1
    report = {"fixed_subprocess_s": fixed_s,
              "cold_compile_allowance_s": COLD_COMPILE_ALLOWANCE_S,
              "probe": probe_by_geometry, "n_planned_arms": len(plan),
              "by_fraction": {}}
    for fraction in NON_EAGER_SENSITIVITY:
        total, by_tier = 0.0, {}
        for cfg in plan:
            seconds, _detail = price_arm(cfg, probe_by_geometry, fixed_s,
                                         fraction)
            total += seconds
            by_tier[cfg["tier"]] = by_tier.get(cfg["tier"], 0.0) + seconds
        report["by_fraction"][f"{fraction:g}"] = {
            "total_hours": total / 3600.0,
            "by_tier_hours": {k: v / 3600.0 for k, v in by_tier.items()}}
    return report


def print_projection(report):
    if report is None:
        print("\nPRICING PROBE: no usable eager-plain arm; nothing to project.")
        return
    print("\n===== mg3 PRICING PROBE -> projected wall =====")
    for geometry, base in report["probe"].items():
        print(f"  measured eager-plain {geometry} {base['cell']} n=1 "
              f"{VCD_ITERATIONS} iters: cold {base['cold']:.1f}s  "
              f"warm {base['warm']:.1f}s")
    print(f"  measured fixed subprocess cost: {report['fixed_subprocess_s']:.1f}s"
          f"   (import + CUDA init + model build + input load)")
    print(f"  ASSUMED per-device cold compile allowance for every non-eager "
          f"arm: {report['cold_compile_allowance_s']:.0f}s")
    print(f"  planned arms (generators included): {report['n_planned_arms']}")
    print(f"\n  {'non-eager fraction':>20}{'total h':>10}   by tier (h)")
    for fraction, entry in report["by_fraction"].items():
        tiers = "  ".join(f"{k}={v:.2f}"
                          for k, v in sorted(entry["by_tier_hours"].items()))
        print(f"  {fraction:>20}{entry['total_hours']:>10.2f}   {tiers}")
    committed = report["by_fraction"][f"{NON_EAGER_FRACTION:g}"]
    total = committed["total_hours"]
    deep = committed["by_tier_hours"].get("deep", 0.0)
    print(f"\n  COMMITTED (upper bound, non-eager fraction "
          f"{NON_EAGER_FRACTION:g}): {total:.2f} h against the "
          f"{SBATCH_WALLTIME_H:.0f} h sbatch walltime.")
    if total > SBATCH_WALLTIME_H:
        print(f"  VERDICT: the single job does NOT fit.  Split it: run the "
              f"depth-3 tiers ({total - deep:.2f} h) and the depth-10 tier "
              f"({deep:.2f} h) as two jobs, chained with --dependency, per "
              f"mg3_gautschi.sbatch's header.")
    else:
        print("  VERDICT: the single job fits the requested walltime.")
    print("  Caveats: COUNT_SCALE is the §2 PRE-KERNEL prior used as the "
          "pessimistic bound, and cone borrows parallel's because cone has no "
          "timed n>1 prior at all.")


# ── the value columns ─────────────────────────────────────────────────────────
def _rel_pair(path_a, path_b):
    """kb3's comparison mechanics on the gathered volumes, reported BOTH ways:
    ``max_rel`` is the worst single voxel and ``norm_rel`` is the whole field.
    A max-rel alone can be one boundary tie; a norm-rel alone can hide one."""
    import numpy as np

    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return {"max_rel": None, "norm_rel": None}
    a, b = np.load(path_a), np.load(path_b)
    if a.shape != b.shape:
        return {"max_rel": None, "norm_rel": None, "shape_mismatch": True}
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    ref = np.asarray(b, dtype=np.float64)
    max_rel = float(np.max(np.abs(diff)) / max(float(np.max(np.abs(ref))),
                                               1e-30))
    norm_rel = float(np.linalg.norm(diff.ravel())
                     / max(float(np.linalg.norm(ref.ravel())), 1e-30))
    return {"max_rel": max_rel, "norm_rel": norm_rel}


def value_rows(rows, geometry, cell, tier, iterations, counts, classes):
    """One value row per (geometry, cell, tier, count): the five §4 columns,
    each as max-rel and norm-rel, with the artifact md5 every consumer read.

    The columns and the expectation each one carries:

      eager_vs_jax     PARTITION ORDER + both frameworks' remaining latitude
      prod_vs_eager    COMPILE LATITUDE -- expected at EXPECTED_COMPILE_LATITUDE
                       for this geometry, DECAYING with depth.  Growth with n is
                       a finding (§4).
      prod_vs_jax      what a user sees; reported against the documented
                       DOCUMENTED_CROSS_FRAMEWORK pair at parallel 1024
      jax_n_vs_n1      partition order, jax side -- a BOUND, because jax is
                       always jitted and carries XLA's per-shape latitude
      eager_n_vs_n1    partition order, clean -- expected to land on
                       EXPECTED_PARTITION_FLOORS for this (geometry, n)
    """
    by_key = {(r.get("arm_class"), r.get("n_dev")): r for r in rows
              if not r.get("error")}

    def sample(arm_class, n):
        row = by_key.get((arm_class, n))
        return row.get("sample_path") if row else None

    def pair(a_class, a_n, b_class, b_n):
        a, b = sample(a_class, a_n), sample(b_class, b_n)
        if not (a and b):
            return {"max_rel": None, "norm_rel": None}
        return _rel_pair(a, b)

    md5 = next((r.get("sino_md5") for r in rows if r.get("sino_md5")), None)
    out = []
    have_jax = "jax" in classes
    for n in counts:
        entry = dict(kind="values", geometry=geometry, cell=list(cell),
                     tier=tier, iterations=iterations, n_dev=n, sino_md5=md5)
        entry["prod_vs_eager"] = pair("torch_prod", n, "torch_eager", n)
        entry["eager_n_vs_n1"] = pair("torch_eager", n, "torch_eager", 1)
        if have_jax:
            entry["eager_vs_jax"] = pair("torch_eager", n, "jax", n)
            entry["prod_vs_jax"] = pair("torch_prod", n, "jax", n)
            entry["jax_n_vs_n1"] = pair("jax", n, "jax", 1)
        if ("torch_compiled_nokernel", n) in by_key:
            # The ordering check: the compiler ALONE, against eager-plain.
            entry["compiled_nokernel_vs_eager"] = pair(
                "torch_compiled_nokernel", n, "torch_eager", n)
        entry["expected_partition_floor"] = EXPECTED_PARTITION_FLOORS.get(
            (geometry, n))
        entry["expected_compile_latitude"] = EXPECTED_COMPILE_LATITUDE.get(
            geometry)
        entry["documented_cross_framework"] = (
            DOCUMENTED_CROSS_FRAMEWORK.get(iterations)
            if (geometry, cell[0]) == DEEP_TIER else None)
        out.append(entry)
    return out


def print_block(geometry, cell, tier, iterations, rows, values):
    print(f"\n===== {geometry} {tuple(cell)}  tier={tier}  "
          f"iterations={iterations} =====")
    print(f"{'arm':>28}{'n':>3}{'cold_s':>9}{'warm_s':>9}{'spread':>8}"
          f"{'peakGB':>8}{'dev':>5}{'body':>6}{'cmp':>5}{'vb':>4}{'hot':>5}")
    for row in rows:
        if row.get("error"):
            print(f"{row.get('arm_class', '?'):>28}"
                  f"{str(row.get('n_dev', '-')):>3}  ERROR: "
                  f"{str(row['error'])[:100]}")
            continue

        def flag(key):
            value = row.get(key)
            return "-" if value is None else ("ok" if value else "FAIL")
        peak = row.get("gpu_peak_bytes", 0) / 2 ** 30
        spread = row.get("vcd_warm_spread")
        print(f"{row['arm_class']:>28}{str(row.get('n_dev', '-')):>3}"
              f"{row.get('vcd_cold', 0):>9.2f}{row.get('vcd_warm', 0):>9.2f}"
              f"{(f'{spread:.1%}' if spread is not None else '-'):>8}"
              f"{peak:>8.2f}{flag('devices_ok'):>5}{flag('bodies_ok'):>6}"
              f"{flag('compile_ok'):>5}{flag('vb_ok'):>4}"
              f"{('HOT' if row.get('gpu_hot') else '-'):>5}")
    for entry in values:
        print(f"  n={entry['n_dev']}")
        for key in ("eager_vs_jax", "prod_vs_eager", "prod_vs_jax",
                    "jax_n_vs_n1", "eager_n_vs_n1",
                    "compiled_nokernel_vs_eager"):
            column = entry.get(key)
            if not column or column.get("max_rel") is None:
                continue
            note = ""
            if key == "eager_n_vs_n1" and entry.get("expected_partition_floor"):
                note = (f"   (engine floor "
                        f"{entry['expected_partition_floor']:.2e})")
            if key == "prod_vs_eager" and \
                    entry.get("expected_compile_latitude"):
                note = (f"   (expected <= "
                        f"{entry['expected_compile_latitude']:.0e})")
            if key == "prod_vs_jax" and entry.get("documented_cross_framework"):
                note = (f"   (documented "
                        f"{entry['documented_cross_framework']:.1e})")
            print(f"    {key:>28}  max {column['max_rel']:.2e}  "
                  f"norm {column['norm_rel']:.2e}{note}")


# ── main ──────────────────────────────────────────────────────────────────────
def _cleanup(paths):
    if os.environ.get("MG3_KEEP_ARTIFACTS", "0") == "1":
        return
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    geometries, cells, counts = selected_plan()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = "probe" if PROBE_ONLY else "value"
    out_path = os.path.join(RESULTS_DIR,
                            f"mg3_{mode}_{RUN_LABEL}_{stamp}.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if PROBE_ONLY:
        staging, measured = probe_plan(geometries, cells)
        print(f"mg3 PRICING PROBE on {RUN_LABEL} ({DEVICE}): "
              f"{len(measured)} measured arm(s), {len(staging)} generator(s) "
              f"-> {out_path}")
        rows = []
        with open(out_path, "w") as sink:
            for cfg in staging + measured:
                print(f"  {cfg['arm_id']} ...", flush=True)
                row = run_one(cfg)
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            report = project_wall([r for r in rows
                                   if r.get("tier") == "probe"],
                                  geometries, cells, counts)
            sink.write(json.dumps(dict(kind="projection",
                                       projection=report)) + "\n")
        print_projection(report)
        _cleanup([_sino_path(c["geometry"], tuple(c["cell"]))
                  for c in staging]
                 + [_sample_path(c["arm_id"]) for c in measured])
        print(f"\nwrote {out_path}")
        return

    phase0, phase1 = build_plan(geometries, cells, counts)
    plan = phase0 + phase1
    measured = [c for c in plan if c["arm_class"] != "generator"]
    print(f"mg3 value comparison on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, cells {[c[0] for c in cells]}, counts {counts}")
    print(f"  {len(measured)} measured arms + "
          f"{len(plan) - len(measured)} generators -> {out_path}")
    print("  (§7 expects 45 measured arms + optional 3, and 4 generators, at "
          "the full plan)")

    rows_by_block, all_rows = {}, []
    with open(out_path, "w") as sink:
        for index, cfg in enumerate(plan):
            cfg = dict(cfg, block_index=index, block_size=len(plan))
            print(f"  [{index + 1}/{len(plan)}] {cfg['arm_id']} ...",
                  flush=True)
            row = run_one(cfg)
            all_rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()                       # protocol 11: incremental jsonl
            if cfg["arm_class"] != "generator":
                # Grouped by the COMPARISON block, not the scheduling tier:
                # the optional ordering arms compare against the base tier's
                # eager-plain arms (see `_arm`).
                key = (cfg["geometry"], tuple(cfg["cell"]),
                       cfg.get("block_tier", cfg["tier"]), cfg["iterations"])
                rows_by_block.setdefault(key, []).append(row)

        # The value columns, once each block's arms are all in.
        skip_jax = os.environ.get("MG3_SKIP_JAX", "0") == "1"
        classes = _classes(skip_jax)
        for (geometry, cell, tier, iterations), rows in rows_by_block.items():
            values = value_rows(rows, geometry, cell, tier, iterations, counts,
                                classes)
            for entry in values:
                sink.write(json.dumps(entry) + "\n")
            sink.flush()
            print_block(geometry, cell, tier, iterations, rows, values)

        summary = dict(kind="summary", run_label=RUN_LABEL, host=platform.node(),
                       measured_arms=len(measured),
                       errors=[r.get("arm_id") for r in all_rows
                               if r.get("error")],
                       arm_check_failures=[
                           r.get("arm_id") for r in all_rows
                           if False in (r.get("devices_ok"),
                                        r.get("bodies_ok"),
                                        r.get("compile_ok"),
                                        r.get("compile_fallback_free_ok"),
                                        r.get("vb_ok"),
                                        r.get("sino_md5_ok"),
                                        r.get("kernels_launched_ok"),
                                        r.get("calibration_absent_ok"))],
                       hot_rows=[r.get("arm_id") for r in all_rows
                                 if r.get("gpu_hot")])
        sink.write(json.dumps(summary) + "\n")

    print(f"\nerrors: {summary['errors'] or 'none'}")
    print(f"arm-check failures: {summary['arm_check_failures'] or 'none'}")
    print(f"hot rows (protocol 11: re-run only if the clock is ALSO "
          f"depressed): {summary['hot_rows'] or 'none'}")
    _cleanup([_sino_path(g, c) for g in geometries for c in cells]
             + [_sample_path(c["arm_id"]) for c in measured])
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        if cfg["arm_class"] == "generator":
            row = generator_worker(cfg)
        elif cfg["framework"] == "jax":
            row = jax_worker(cfg)
        else:
            row = torch_worker(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)


def _dry_run():
    geometries, cells, counts = selected_plan()
    if PROBE_ONLY:
        staging, measured = probe_plan(geometries, cells)
        print(f"PROBE: {len(measured)} measured arms, {len(staging)} generators")
        for cfg in staging + measured:
            print("  ", cfg["arm_id"])
        return
    phase0, phase1 = build_plan(geometries, cells, counts)
    for label, block in (("PHASE 0 (generators + every n=1 arm)", phase0),
                         ("PHASE 1 (reversed n>1 blocks)", phase1)):
        print(f"\n{label}: {len(block)} arms")
        for cfg in block:
            print(f"   {cfg['arm_id']:>52}  n={cfg['n_dev']}  "
                  f"iters={cfg['iterations']}  tier={cfg['tier']}")
    measured = [c for c in phase0 + phase1 if c["arm_class"] != "generator"]
    print(f"\ntotal {len(phase0) + len(phase1)} arms "
          f"({len(measured)} measured, "
          f"{len(phase0) + len(phase1) - len(measured)} generators)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    elif "--dry-run" in sys.argv:
        _dry_run()
    else:
        main()
