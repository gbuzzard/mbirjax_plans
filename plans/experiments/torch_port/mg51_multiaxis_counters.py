"""mg51 -- WHAT LIMITS THE MULTIAXIS GEOMETRY'S COMPILED PROJECTION BODIES.

WHAT THIS RUN IS.  MultiAxisParallelModel has no hand-written kernels.  Both
of its projection directions run bodies that torch.compile generated from the
python in mbirtorch/multiaxis_parallel.py (``_multiaxis_forward_view_batch``
and ``_multiaxis_back_view_batch``), and the model says so itself: its
``_view_batch_bodies`` returns those two functions with no kernel in sight.
Every other geometry in this tree that anyone has tried to speed up got a
hand-written kernel first and a counter reading afterwards.  This run takes
the counter reading FIRST, so that whether multiaxis is worth a kernel at all
is a question with measurements under it.

WHAT IT MEASURES.  Five things, per direction and at two sizes:

  1. HOW MANY DISTINCT GENERATED KERNELS one projection call runs.  A compiled
     body is not one kernel; inductor splits it into as many as its fusion
     decides on.  A body that runs three kernels and a body that runs forty
     are different engineering problems.
  2. HOW MANY LAUNCHES one call issues in total.  Distinct kernels times the
     view-batch iteration count, roughly -- but only roughly, and the profiler
     counts rather than infers it.
  3. WHERE THE DEVICE TIME CONCENTRATES.  The top ten kernels by device time,
     with their call counts and their mean duration.  A single dominant kernel
     is a kernel-writing target; twenty kernels at five percent each is not.
  4. HOW HOST ENQUEUE TIME COMPARES TO DEVICE BUSY TIME.  If the host spends
     more time issuing launches than the device spends running them, the body
     is launch-bound and no amount of kernel tuning moves it; the remedy would
     be fewer, larger launches.  This is read two ways -- from the profiler's
     own host and device totals, and from a profiler-free pair of clocks in
     the timing leg -- so neither reading has to be taken on faith.
  5. WHAT THE HARDWARE COUNTERS SAY about the kernels that matter.  Nsight
     Compute on the top three generated kernels in each direction: achieved
     occupancy, SM throughput, memory throughput, the L1 and L2 hit rates and
     the DRAM bytes moved.  Memory-bound, latency-bound and compute-bound are
     three different verdicts with three different remedies, and these are the
     columns that separate them.

THIS RUN DECIDES NOTHING AND IMPLEMENTS NOTHING.  It writes no kernel and
changes no library file.  The verdict is read by a person from the tables in
this log and the rows in the jsonl.  The exit code reports instrument health
only: whether the measurement was taken correctly, never what it found.

## THE TWO CELLS

Cell A, multiaxis sinogram (512, 448, 384).  Cell B, (1024, 1008, 992).  Both
are cells the multiaxis widening floors were measured at, so a wall here sits
beside a recorded wall without adjustment.  One H100, one device, placed
explicitly.

## THE PROTOCOL IS THE FLOORS REFRESH'S

Same model construction (azimuths evenly spaced over half a turn, elevations
swept across +/- 0.5 radians), same shepp-logan low-dynamic-range phantom,
same seed 13 reset before every call.  The floors refresh times a whole
reconstruction; this run times ONE projection call in each direction, through
the public funnel the reconstruction itself calls
(``TomographyModel.sparse_forward_project`` and ``sparse_back_project``).
The phantom is forward projected once to make the sinogram the back
projection is given, which is the same array the floors refresh stages.

## THE THREE LEGS

THE TIMING LEG runs first and always.  One cold call per direction, then warm
calls on a plain host clock with a device synchronize.  It exists for two
reasons: the warm walls anchor against the recorded component numbers, and
the same walls compared against the profiled calls show what the profiler
itself costs.  Each warm call also records two extra clocks -- the host time
to RETURN from the call without synchronizing (the enqueue cost) and the
device span between a pair of CUDA events around it -- which is the
profiler-free form of question 4.

THE PROFILER LEG runs torch.profiler with CPU and CUDA activities around warm
calls: three at cell A, one at cell B, because a 1024-class call is expensive
and one profile is enough to bound it.  Everything in questions 1 through 4
comes out of that trace.

THE COUNTER LEG runs at cell A only, and it never changes the exit code.  It
discovers the top generated kernel names at runtime in the process ncu will
actually profile, then runs ncu once per kernel against a small runner mode of
this same script, with a name filter, a launch skip past the cold call and a
launch count of one.  Many clusters keep GPU performance counters closed to
unprivileged users, so the leg probes for that first and records a refusal as
a finding.

## MEMORY AT CELL B

The 1024-class cell holds a 3.5 GB voxel array, a 4.1 GB sinogram and a
similar output per call on one device.  That is far inside an 80 GB H100 on
paper, but paper is not a measurement, so every cell-B leg is wrapped: an
out-of-memory is caught, recorded as that leg's result, and the run continues.

## THE LOCAL SMOKE

MG51_SMOKE=1 runs the whole flow at a tiny cell on the CPU.  There are no CUDA
kernels there, so the profiler leg records zero of them, the device-event
clocks are skipped, and the counter leg is skipped -- each recorded rather
than silently absent.  What the smoke exercises is the plumbing: the plan, the
model construction, the phantom, the production-route calls, the profiler
parse, the rows and the tables.  It is not a measurement.

Run:
    <torch python> mg51_multiaxis_counters.py        on a 1-GPU node
    MG51_DRY=1 <python> mg51_multiaxis_counters.py   print the plan and stop
    MG51_SMOKE=1 <python> mg51_multiaxis_counters.py the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized cell or direction is an error, not a
silent skip.
    MG51_RESULTS=<dir>       where the jsonl and the ncu logs go
    MG51_CELLS=a,b           subset of the cells
    MG51_DIRECTIONS=forward  subset of the directions
    MG51_DRY=1               print the plan and exit; imports no torch
    MG51_SMOKE=1             the local CPU smoke
    MG51_NCU=0               skip the counter leg entirely
    MG51_WARM=3              warm timed calls per direction and cell
    MG51_NCU_RUNNER=<json>   internal: run just the calls ncu profiles
"""

import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────


def _flag(name, default="0"):
    """An environment flag that must read exactly "0" or "1".

    Accepting "true" or "yes" silently as false has cost this work a repeat
    before: the run prints the plan it was asked for and measures another one.
    """
    raw = os.environ.get(name, default).strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name}: {raw!r} is not 0 or 1")
    return raw == "1"


def _positive_int(name, default):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"{name}: {value} is not at least 1")
    return value


SMOKE = _flag("MG51_SMOKE")
DRY = _flag("MG51_DRY")
#: The internal runner mode.  Non-empty means this process exists only to make
#: the calls the counter leg profiles; it takes a json config and prints one
#: __RESULT__ line.  The sbatch unsets it, so a stray value in the submitting
#: shell cannot turn the real run into a runner.
NCU_RUNNER_RAW = os.environ.get("MG51_NCU_RUNNER", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: (views, detector rows, channels).  Both are cells the multiaxis widening
#: floors were measured at, so a wall here is comparable to a recorded one.
CELL_SPECS = (
    dict(cell_id="a", cell=(512, 448, 384),
         note="the 512-class multiaxis cell"),
    dict(cell_id="b", cell=(1024, 1008, 992),
         note="the 1024-class multiaxis cell; near one device's memory edge, "
              "so its legs are wrapped against an out-of-memory"),
)
SMOKE_CELL_SPECS = (
    dict(cell_id="a", cell=(12, 16, 16),
         note="the smoke's tiny cell: plumbing only, not a measurement"),
    dict(cell_id="b", cell=(16, 20, 20),
         note="a second tiny cell, so the smoke exercises the per-cell loop "
              "the real run makes"),
)

DIRECTIONS = ("forward", "back")

#: The floors refresh's seed, reset before every call it times.  This run
#: resets it before every call it times too, for the same reason: the protocol
#: is the one the recorded multiaxis walls were measured under.
SEED = 13
#: The elevation half-range the floors refresh sweeps.  Mirrored here so the
#: dry plan can print the recon shape without importing numpy; the real run
#: reads the model's own recon_shape and records whether the two agreed.
ELEVATION_HALF_RANGE = 0.5
#: The geometry's clamp on the smallest |cos(elevation)| (multiaxis_parallel's
#: auto_set_recon_geometry).  Mirrored for the same reason.
MIN_COS_ELEVATION = 0.1

WARM_REPEATS = _positive_int("MG51_WARM", 3)
#: Profiled warm calls per direction.  Three at cell A; one at cell B, because
#: a 1024-class call is expensive and one profile bounds the wall.  The LAST
#: repeat is the one the tables report, and every repeat is on the row.
PROFILE_REPEATS = {"a": 3, "b": 1}
SMOKE_PROFILE_REPEATS = {"a": 2, "b": 1}
#: How many kernels the profiler leg reports per direction and cell.
TOP_KERNELS = 10

# ── the counter leg ───────────────────────────────────────────────────────────
NCU_ENABLED = _flag("MG51_NCU", "1")
#: The counter leg profiles cell A only.  A 1024-class replay has never been
#: timed by anyone and the intensities the counters report are per-launch, so
#: the smaller cell answers the same question for a bounded cost.
NCU_CELL_ID = "a"
#: Top generated kernels per direction that ncu is pointed at.
NCU_TOP = 3
#: Calls the runner makes.  The first is cold (it pays the compile and any
#: first-touch allocation); the profiled launch is taken from the second, and
#: the third is margin in case a kernel's per-call launch count is not exactly
#: what the discovery run counted.
NCU_RUNNER_CALLS = 3
#: How many of those calls the launch skip steps past.
NCU_SKIP_CALLS = 1
#: A launch skip is capped so that a kernel with a very large per-call launch
#: count cannot ask ncu to walk past more launches than the runner makes.
NCU_MAX_SKIP = 4096
#: Two bounds, because ncu replays each profiled kernel several times to
#: collect its counters and nobody has timed a replay of THESE kernels.  One
#: attempt cannot run longer than the first, and the whole leg cannot run
#: longer than the second; whatever is left unprofiled is recorded as
#: unprofiled.  The timing and profiler legs have already finished by then, so
#: a leg that runs out of budget costs the run nothing it needed.
NCU_TIMEOUT_S = 480
NCU_LEG_BUDGET_S = 3000
NCU_PROBE_TIMEOUT_S = 180
NCU_DISCOVERY_TIMEOUT_S = 900

#: Everything this run wants, copied from the counter runs that came before
#: (mg25 and mg28) so the columns line up with theirs without adjustment.
METRICS_FULL = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
#: The set that must not fail: the names the earlier counter runs collected on
#: this cluster's ncu.  Nothing unproven goes in it.
METRICS_CORE = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sectors_op_read.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
#: The attempts, in order: (metric set name, metric set, launch skip policy).
#: "warm" means the discovered per-call launch count times NCU_SKIP_CALLS;
#: "first" means zero, which profiles the first launch of the cold call and is
#: worth having if the aimed skip matched nothing.
NCU_ATTEMPTS = (
    ("full", METRICS_FULL, "warm"),
    ("core", METRICS_CORE, "warm"),
    ("core", METRICS_CORE, "first"),
)
NCU_PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                          "insufficient permission")

#: What an inductor-generated kernel is called.  The counter leg aims at these
#: and records what it aimed at; a run in which no generated kernel appears is
#: itself a finding, and the leg then falls back to the top device kernels of
#: any kind rather than profiling nothing.
GENERATED_PREFIX = "triton_"

# ── GPU health ────────────────────────────────────────────────────────────────
# A thermally throttled or power-capped device produces valid counters and an
# invalid wall, and this run reports walls beside counters.
HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,"
                    "temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

RESULTS_DIR = os.environ.get(
    "MG51_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
NAME_COL = 46
# ──────────────────────────────────────────────────────────────────────────────


def cell_specs():
    return SMOKE_CELL_SPECS if SMOKE else CELL_SPECS


def profile_repeats(cell_id):
    table = SMOKE_PROFILE_REPEATS if SMOKE else PROFILE_REPEATS
    return table.get(cell_id, 1)


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured less than it printed has cost this work a repeat
    before.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"{env_name}: {token!r} is not one of "
                             f"{list(allowed)}")
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return [name for name in allowed if name in chosen]


def mirrored_recon_shape(cell):
    """The recon shape this cell should produce, derived from the geometry's
    own rule without importing anything.

    The rule is multiaxis_parallel.auto_set_recon_geometry: the in-plane
    extent comes from the channel coverage, the slice extent from the row
    coverage divided by the smallest |cos(elevation)|, clamped at 0.1.  With
    the detector pitches and voxel aspects at their defaults of 1.0 the whole
    rule is arithmetic, which is what lets the dry plan print it.  This is a
    MIRROR: the real run reads the model's own recon_shape and records whether
    the two agreed, so the mirror is checked rather than trusted.
    """
    _views, num_rows, num_channels = cell
    max_u = num_channels / 2.0
    max_v = num_rows / 2.0
    min_cos = max(math.cos(ELEVATION_HALF_RANGE), MIN_COS_ELEVATION)
    return (int(math.floor(2 * max_u)), int(math.floor(2 * max_u)),
            int(math.floor(2 * (max_v / min_cos))))


def build_plan():
    """One entry per selected cell, in table order, each carrying the
    directions selected for it."""
    allowed_cells = [spec["cell_id"] for spec in cell_specs()]
    keep_cells = _strict_subset("MG51_CELLS", allowed_cells)
    keep_dirs = _strict_subset("MG51_DIRECTIONS", DIRECTIONS)
    plan = []
    for spec in cell_specs():
        if spec["cell_id"] not in keep_cells:
            continue
        plan.append(dict(cell_id=spec["cell_id"], cell=list(spec["cell"]),
                         note=spec["note"], directions=list(keep_dirs),
                         warm=WARM_REPEATS,
                         profile_repeats=profile_repeats(spec["cell_id"]),
                         mirrored_recon_shape=list(
                             mirrored_recon_shape(spec["cell"]))))
    if not plan:
        raise ValueError("MG51_CELLS selects no cell")
    return plan


# ── GPU health ────────────────────────────────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Clocks, temperatures and throttle flags, sampled before and after the
    measured legs."""
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


def health_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C) or gpu.get("throttle"):
            return True
    return False


# ── the tree under test ───────────────────────────────────────────────────────
def tree_witnesses():
    """What tree produced these numbers, measured rather than asserted.

    None of these three change what this run measures directly -- the
    multiaxis bodies run no hand-written kernel and no padded wrapper -- but
    they are what lets a reader of the jsonl tell which tree a row came from.
    A lookup that fails is recorded as failed; nothing here raises.
    """
    record = dict(available=True)
    try:
        import inspect

        from mbirtorch import projectors
        from mbirtorch._utils import padded_kernel_width

        record["padded_kernel_width_504"] = int(padded_kernel_width(504))
        record["recompile_limit_floor"] = int(
            getattr(projectors, "_RECOMPILE_LIMIT_FLOOR", -1))
        source = inspect.getsource(projectors.maybe_compile)
        record["raise_on_compiling_thread"] = bool(
            "_raise_recompile_budget()"
            in source.split("_GLOBAL_COMPILE_LOCK:")[-1])
        record["ok"] = bool(record["padded_kernel_width_504"] == 512
                            and record["recompile_limit_floor"] >= 64
                            and record["raise_on_compiling_thread"])
    except Exception as exc:                                      # noqa: BLE001
        record.update(available=False, ok=False,
                      reason=f"{type(exc).__name__}: {exc}")
    return record


# ── the model and its inputs, built the floors refresh's way ──────────────────
def build_model(cell):
    """The multiaxis model at ``cell``, constructed exactly as the widening
    floors refresh constructs it.

    Two angles per view: azimuths evenly spaced over half a turn, elevations
    swept across +/- 0.5 radians.  Those are the geometry's own test defaults
    and the values every recorded multiaxis wall was measured with; the
    elevation range also sets the slice count, so changing it would change the
    problem rather than the measurement.

    The device is named EXPLICITLY.  On the one-GPU allocation this job asks
    for, an explicit ['cuda:0'] and the floors refresh's count pin realize the
    same single device; naming it means the realized device is a fact on the
    row instead of a policy outcome.

    ``skip_memory_preflight`` is set before the device is configured: the
    preflight prices a whole reconstruction, and this run projects once in
    each direction and never reconstructs.
    """
    import numpy as np

    import mbirtorch

    num_views = int(cell[0])
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.linspace(-ELEVATION_HALF_RANGE, ELEVATION_HALF_RANGE,
                            num_views)
    model = mbirtorch.MultiAxisParallelModel(
        tuple(int(x) for x in cell), np.stack([azimuth, elevation], axis=1))
    model.skip_memory_preflight = True
    model.configure_devices(
        devices=[DEVICE + (":0" if DEVICE == "cuda" else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def build_phantom(model):
    """The floors refresh's phantom, as a host float32 array.

    The shepp-logan builder places its ellipsoids as fractions of the volume,
    and on a volume only a few voxels deep every one of them can miss, leaving
    the phantom all zeros.  An all-zero phantom forward projects to an
    all-zero sinogram, so the back projection would then be given nothing to
    do and the measurement would describe an empty problem.  The floors
    refresh falls back to a seeded uniform volume in that case and records
    that it did; so does this.
    """
    import numpy as np

    import mbirtorch

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    fallback = None
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
        fallback = "seeded uniform (shepp-logan returned all zeros)"
    return np.asarray(phantom, dtype=np.float32), recon_shape, fallback


def _is_oom(exc):
    """Whether an exception is a device out-of-memory.

    The class name is checked as well as the message because torch raises its
    own OutOfMemoryError on some paths and a plain RuntimeError on others, and
    a leg that mistook one for a real failure would kill a job that was only
    too big for one card.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(exc).lower()


def _free_device_memory(torch_module):
    if DEVICE == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def _sync(torch_module, device):
    if DEVICE == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.synchronize(device)


class CellContext:
    """One cell's model and its two projection inputs, built once and used by
    every leg at that cell.

    Building the inputs is the expensive part at cell B -- a phantom the size
    of the volume, a fancy-index down to the mask, and one forward projection
    -- so the legs are ordered by cell rather than by leg.  The FIRST forward
    call is the timing leg's cold call, and its output is the sinogram the
    back projection is given, so nothing is projected twice for bookkeeping.
    """

    def __init__(self, spec):
        self.spec = spec
        self.cell = tuple(int(x) for x in spec["cell"])
        self.cell_id = spec["cell_id"]
        self.model = None
        self.indices = None
        self.voxel_values = None
        self.sinogram = None
        self.record = dict(kind="cell_setup", cell_id=self.cell_id,
                           cell=list(self.cell), device=DEVICE)

    def build(self):
        """Model, mask, voxel values.  The sinogram is made later, by the
        forward direction's cold call."""
        import numpy as np
        import torch

        start = time.perf_counter()
        self.model = build_model(self.cell)
        model = self.model
        self.record["device_realized"] = str(model.torch_device)
        self.record["device_expected"] = (
            "cuda:0" if DEVICE == "cuda" else DEVICE)
        self.record["device_ok"] = (self.record["device_realized"]
                                    == self.record["device_expected"])

        from mbirtorch import _memory_ledger
        directions = list(_memory_ledger.torch_body_directions(model))
        self.record["torch_body_directions"] = directions
        # The premise of this whole run: BOTH multiaxis directions are compiled
        # torch bodies, with no hand-written kernel anywhere.  Measured, not
        # assumed -- if a kernel ever appears for this geometry, every number
        # below would describe something else.
        self.record["bodies_ok"] = (directions == ["forward", "back"])
        fwd_body, back_body = model._view_batch_bodies()
        self.record["forward_body"] = fwd_body.__name__
        self.record["back_body"] = back_body.__name__
        self.record["compile_enabled"] = bool(model.compile_enabled)
        self.record["compile_mode"] = str(model.compile_mode)

        phantom, recon_shape, fallback = build_phantom(model)
        self.record["recon_shape"] = list(recon_shape)
        self.record["mirrored_recon_shape"] = list(
            mirrored_recon_shape(self.cell))
        self.record["recon_shape_mirror_agrees"] = (
            list(recon_shape) == list(mirrored_recon_shape(self.cell)))
        self.record["phantom_fallback"] = fallback

        self.indices = model.full_indices_device()
        num_pixels = int(self.indices.shape[0])
        self.record["num_pixels"] = num_pixels
        # The mask index is taken on the HOST.  Indexing the whole volume on
        # the device would hold the volume and its gathered subset at once,
        # which is the largest single allocation this run would ever make and
        # is avoidable: the gathered subset is what the projector wants.
        idx_host = self.indices.detach().cpu().numpy()
        flat = phantom.reshape(-1, recon_shape[2])
        values_host = np.ascontiguousarray(flat[idx_host])
        phantom = None
        flat = None
        self.voxel_values = torch.as_tensor(values_host, dtype=torch.float32,
                                            device=model.torch_device)
        values_host = None
        self.record["voxel_values_shape"] = [
            int(x) for x in self.voxel_values.shape]
        self.record["voxel_values_bytes"] = int(
            self.voxel_values.numel()) * 4
        self.record["sinogram_bytes"] = int(
            self.cell[0] * self.cell[1] * self.cell[2] * 4)

        args = model._view_batch_args()
        self.record["psf_radius"] = int(args["psf_radius"])
        pf = model.projector_functions
        # How many views one body call takes, per direction, by the driver's
        # own rule.  This is what turns a launch count into launches per view
        # batch, so it is recorded rather than inferred from the trace.
        self.record["view_batch"] = dict(
            forward=int(pf._effective_view_batch(
                pf._fwd_body_per_dev[0], num_pixels, int(recon_shape[2]),
                args)),
            back=int(pf._effective_view_batch(
                pf._back_body_per_dev[0], num_pixels, int(self.cell[1]),
                args)))
        self.record["view_batch_iterations"] = {
            name: int(math.ceil(self.cell[0] / max(1, batch)))
            for name, batch in self.record["view_batch"].items()}
        self.record["setup_s"] = time.perf_counter() - start
        return self.record

    def compile_state(self):
        """Whether the compiled bodies really compiled, read after the calls
        that would have compiled them.

        THE ASSUMPTION THIS WHOLE RUN RESTS ON is that the bodies being
        measured are COMPILED bodies.  ``maybe_compile`` guards the first call
        of each body: if the inductor backend raises, the call is retried
        eagerly, the error is filed in ``projectors._COMPILE_ERRORS``, and the
        body is permanently rebound to eager.  That is the right behavior for
        a library and the wrong thing to be unaware of here -- the run would
        measure eager python and report it as a compiled body.  So the error
        table is read and put on the row rather than assumed empty.
        """
        try:
            from mbirtorch import projectors
            errors = dict(getattr(projectors, "_COMPILE_ERRORS", {}) or {})
        except Exception as exc:                                  # noqa: BLE001
            return dict(available=False,
                        reason=f"{type(exc).__name__}: {exc}")
        multiaxis = {name: text for name, text in errors.items()
                     if "multiaxis" in name.lower()}
        return dict(available=True, compile_errors=errors,
                    multiaxis_compile_errors=multiaxis,
                    bodies_really_compiled=not multiaxis)

    def call(self, direction):
        """One production-route projection call in ``direction``.

        This is the funnel the reconstruction itself calls: every sparse
        projection in the library, from the recon engine and from outside it,
        routes through these two methods.
        """
        model = self.model
        if direction == "forward":
            return model.sparse_forward_project(self.voxel_values,
                                                self.indices)
        return model.sparse_back_project(self.sinogram, self.indices)

    def ensure_sinogram(self):
        """Make the sinogram the back projection needs, without timing it.

        In a full run the forward direction's cold call makes it, so nothing
        is projected twice.  This exists for a run restricted to the back
        direction -- a re-run of one leg after a job ran short -- which would
        otherwise have no input at all.
        """
        import torch

        if self.sinogram is not None:
            return False
        _seed_before_call()
        self.sinogram = self.call("forward")
        _sync(torch, self.model.torch_device)
        return True

    def ready(self, direction):
        """Whether ``direction`` has the input it needs.  The back projection
        needs the sinogram the forward direction's cold call produced."""
        if direction == "back":
            return self.sinogram is not None
        return self.voxel_values is not None

    def release(self):
        import torch

        self.voxel_values = None
        self.sinogram = None
        self.indices = None
        self.model = None
        _free_device_memory(torch)


# ── the timing leg ────────────────────────────────────────────────────────────
def _seed_before_call():
    """The floors refresh resets seed 13 before every call it times; so does
    this, so the protocol is the recorded one."""
    import numpy as np

    np.random.seed(SEED)


def timing_leg(context, direction):
    """One cold call, then warm calls on three clocks.

    THE THREE CLOCKS, and why there are three.  The plain wall (host clock
    around the call plus a device synchronize) is the number that anchors
    against the recorded component walls, and comparing it with the profiled
    call's wall is what shows the profiler's own overhead.  The enqueue clock
    is the host time to RETURN from the call WITHOUT synchronizing: for an
    asynchronous device workload that is what the host spent issuing the work.
    The device clock is the span between a pair of CUDA events recorded on the
    default stream around the call, resolved after the synchronize.  Enqueue
    close to the wall means the host is the limit; enqueue far below the
    device span means the device is saturated and the host is waiting.

    Off CUDA there are no events and the enqueue clock has no meaning (every
    CPU call is synchronous), so both are skipped and recorded as skipped.
    """
    import torch

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row = dict(kind="timing", cell_id=context.cell_id,
               cell=list(context.cell), direction=direction, device=DEVICE,
               warm_repeats=WARM_REPEATS, device_events=bool(cuda),
               seed=SEED)

    _seed_before_call()
    start = time.perf_counter()
    out = context.call(direction)
    if direction == "forward":
        # The cold forward call's OUTPUT is the sinogram the back projection
        # is given, which is the array the floors refresh stages.  Keeping it
        # here means the run forward projects the phantom once, not twice.
        context.sinogram = out
    _sync(torch, device)
    row["cold_s"] = time.perf_counter() - start
    row["output_shape"] = [int(x) for x in out.shape]
    if direction != "forward":
        out = None
    # Read AFTER the cold call, which is the call that would have compiled
    # this direction's body and the call whose failure would have rebound it
    # to eager.
    row["compile_state"] = context.compile_state()

    walls, enqueues, spans = [], [], []
    for _ in range(WARM_REPEATS):
        _seed_before_call()
        events = None
        if cuda:
            events = (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
            events[0].record(torch.cuda.default_stream(device))
        start = time.perf_counter()
        warm_out = context.call(direction)
        if cuda:
            events[1].record(torch.cuda.default_stream(device))
        enqueue = time.perf_counter() - start
        _sync(torch, device)
        walls.append(time.perf_counter() - start)
        enqueues.append(enqueue)
        if cuda:
            try:
                spans.append(float(events[0].elapsed_time(events[1])) / 1000.0)
            except Exception:                                     # noqa: BLE001
                # An unreadable event is recorded as unreadable rather than as
                # zero, which would silently claim the device did nothing.
                spans.append(None)
        warm_out = None

    row["warm_all_s"] = walls
    row["warm_s"] = statistics.median(walls)
    row["enqueue_all_s"] = enqueues
    row["enqueue_s"] = statistics.median(enqueues)
    readable = [value for value in spans if isinstance(value, float)]
    row["device_span_all_s"] = spans
    row["device_span_s"] = statistics.median(readable) if readable else None
    row["spread"] = ((max(walls) - min(walls)) / row["warm_s"]
                     if row["warm_s"] else None)
    if row["warm_s"]:
        row["enqueue_frac_of_wall"] = row["enqueue_s"] / row["warm_s"]
    if row["device_span_s"]:
        row["enqueue_over_device_span"] = (row["enqueue_s"]
                                           / row["device_span_s"])
    if cuda:
        row["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return row


# ── the profiler leg ──────────────────────────────────────────────────────────
def _is_device_row(row):
    """Whether a key_averages row describes work that ran ON the device.

    torch.profiler puts device kernels in the same table as host operators and
    tells them apart by device_type.  The enum's spelling has moved between
    releases, so the comparison is on the string, and an unrecognized value is
    treated as a host row -- undercounting device work is visible in the
    totals, whereas miscounting a host row as a kernel would inflate both the
    kernel count and the device time with no way to see it.
    """
    kind = getattr(row, "device_type", None)
    if kind is None:
        return False
    text = str(kind).upper()
    return text.endswith("CUDA") or text.endswith("HIP") \
        or text.endswith("PRIVATEUSE1")


def _is_transfer(name):
    """Memcpy and memset rows are device work but not kernels; they are
    counted separately so 'how many kernels' means kernels."""
    lowered = str(name).lower()
    return lowered.startswith("memcpy") or lowered.startswith("memset") \
        or "memcpy" in lowered or "memset" in lowered


def _is_sync(name):
    lowered = str(name).lower()
    return "synchronize" in lowered or lowered.endswith("eventsynchronize")


def _row_device_us(row):
    value = getattr(row, "self_device_time_total", None)
    if value is None:
        value = getattr(row, "self_cuda_time_total", 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _row_host_us(row):
    try:
        return float(getattr(row, "self_cpu_time_total", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_profile(prof, top=TOP_KERNELS):
    """Everything questions 1 through 4 want, out of one trace.

    Host time is reported twice: with and without the synchronize.  The call
    is profiled with a synchronize inside the trace so the device work is
    attributed to this call and not the next one, and the host time spent
    WAITING in that synchronize is not host work -- counting it as host work
    would make a device-bound call look host-bound.  The two numbers are both
    on the row, and the one that answers 'is the host the limit' is the one
    without the wait.
    """
    rows = list(prof.key_averages())
    kernels, transfers = [], []
    host_us = host_us_no_sync = sync_us = 0.0
    launch_calls = launch_us = 0.0
    for row in rows:
        name = str(getattr(row, "key", ""))
        if _is_device_row(row):
            entry = dict(name=name, calls=int(getattr(row, "count", 0) or 0),
                         device_us=_row_device_us(row))
            entry["mean_device_us"] = (entry["device_us"] / entry["calls"]
                                       if entry["calls"] else None)
            (transfers if _is_transfer(name) else kernels).append(entry)
            continue
        this_host = _row_host_us(row)
        host_us += this_host
        if _is_sync(name):
            sync_us += this_host
        else:
            host_us_no_sync += this_host
        # The runtime's own launch row is the most direct statement of how
        # many launches the HOST issued and what issuing them cost.  It is a
        # cross-check on the device-side launch count, and the two disagreeing
        # is worth seeing rather than hiding.
        if name in ("cudaLaunchKernel", "cudaLaunchKernelExC"):
            launch_calls += int(getattr(row, "count", 0) or 0)
            launch_us += this_host

    kernels.sort(key=lambda entry: -entry["device_us"])
    transfers.sort(key=lambda entry: -entry["device_us"])
    kernel_us = sum(entry["device_us"] for entry in kernels)
    transfer_us = sum(entry["device_us"] for entry in transfers)
    generated = [entry for entry in kernels
                 if entry["name"].startswith(GENERATED_PREFIX)]
    summary = dict(
        distinct_kernels=len(kernels),
        distinct_generated_kernels=len(generated),
        kernel_launches=int(sum(entry["calls"] for entry in kernels)),
        generated_launches=int(sum(entry["calls"] for entry in generated)),
        transfer_ops=len(transfers),
        transfer_launches=int(sum(entry["calls"] for entry in transfers)),
        device_us=kernel_us + transfer_us,
        kernel_device_us=kernel_us,
        transfer_device_us=transfer_us,
        generated_device_us=sum(entry["device_us"] for entry in generated),
        host_us=host_us, host_us_no_sync=host_us_no_sync, sync_us=sync_us,
        runtime_launch_calls=int(launch_calls), runtime_launch_us=launch_us,
        top_kernels=kernels[:top], top_transfers=transfers[:4],
        profiler_rows=len(rows))
    if kernel_us:
        summary["host_no_sync_over_device"] = host_us_no_sync / kernel_us
    return summary


def _warm_up_profiler(torch_module, device):
    """Pay kineto's start-up cost once, on a trivial op, so it does not land
    inside a measured profile.

    Without this, the FIRST profiled call in the process carries the
    profiler's own initialization.  Cell A runs three repeats and would absorb
    it, but cell B runs one, and a run restricted to cell B would have nothing
    to absorb it at all.
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:                                             # noqa: BLE001
        return dict(ok=False, reason="torch.profiler is unavailable")
    cuda = DEVICE == "cuda" and torch_module.cuda.is_available()
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
    start = time.perf_counter()
    try:
        with profile(activities=activities) as prof:
            scratch = torch_module.ones(1024, device=device)
            scratch = (scratch * 2.0).sum()
            if cuda:
                torch_module.cuda.synchronize(device)
        _rows = len(list(prof.key_averages()))
    except Exception as exc:                                      # noqa: BLE001
        return dict(ok=False, reason=f"{type(exc).__name__}: {exc}")
    return dict(ok=True, wall_s=time.perf_counter() - start, rows=_rows)


def profiler_leg(context, direction, repeats):
    """torch.profiler with CPU and CUDA activities around warm calls.

    One profile context per call, rather than one context around several, so
    every count on the row is a PER-CALL count.  The counter leg needs the
    per-call launch count of a kernel to know how many launches to skip past,
    and dividing an aggregate by the repeat count would be an inference where
    a measurement is available.
    """
    import torch

    from torch.profiler import ProfilerActivity, profile

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
    row = dict(kind="profile", cell_id=context.cell_id,
               cell=list(context.cell), direction=direction, device=DEVICE,
               repeats=repeats, activities=[str(a) for a in activities],
               cuda=cuda, repeats_detail=[])
    for index in range(repeats):
        _seed_before_call()
        with profile(activities=activities) as prof:
            start = time.perf_counter()
            out = context.call(direction)
            _sync(torch, device)
            wall = time.perf_counter() - start
        out = None
        summary = summarize_profile(prof)
        summary["profiled_wall_s"] = wall
        summary["repeat"] = index
        row["repeats_detail"].append(summary)
    if row["repeats_detail"]:
        # The LAST repeat is the representative one: it is the most warm, and
        # at cell A the earlier repeats are what let a reader see whether the
        # trace was stable across them.
        row.update({key: value
                    for key, value in row["repeats_detail"][-1].items()
                    if key not in ("repeat",)})
    if cuda:
        row["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return row


# ── the runner the counter leg drives ─────────────────────────────────────────
def ncu_runner(cfg):
    """Just the calls the counter leg profiles, in a fresh process.

    Three modes.  "trivial" launches one tiny kernel, which is the permission
    probe's payload.  "calls" builds the counter cell and makes
    ``calls`` production-route projection calls in one direction, which is
    what ncu attaches to.  With ``profile_names`` set, the LAST call is also
    traced by torch.profiler and the kernel names and per-call launch counts
    it saw are returned -- that is the discovery run, and it happens in this
    process rather than the parent's so the names the filter is built from are
    names that exist where ncu will look for them.
    """
    import torch

    mode = str(cfg.get("mode", "calls"))
    result = dict(cfg, mode=mode, device=DEVICE)
    if mode == "trivial":
        if not torch.cuda.is_available():
            return dict(result, cuda=False)
        scratch = torch.ones(1 << 16, device="cuda")
        total = float((scratch * 2.0).sum())
        torch.cuda.synchronize()
        return dict(result, cuda=True, checksum=total)

    direction = str(cfg.get("direction", "forward"))
    calls = int(cfg.get("calls", NCU_RUNNER_CALLS))
    spec = next((entry for entry in cell_specs()
                 if entry["cell_id"] == str(cfg.get("cell_id", NCU_CELL_ID))),
                cell_specs()[0])
    context = CellContext(spec)
    result["setup"] = context.build()
    result["direction"] = direction
    result["calls"] = calls
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result["cuda"] = cuda
    device = context.model.torch_device

    if direction == "back":
        # The back projection needs a sinogram, and the sinogram is the
        # forward projection of the phantom.  Making it here costs one forward
        # call BEFORE the calls ncu is aimed at.
        #
        # WHY THAT CALL IS COUNTED.  A launch skip is arithmetic on the number
        # of launches that come before the one to profile, and inductor names
        # its kernels per graph -- two different graphs can each produce a
        # kernel called triton_poi_fused_0.  If the forward graph happens to
        # have generated a kernel with the same name as the back kernel being
        # aimed at, the forward call's launches of it come first and the skip
        # would land inside that call instead of inside a warm back call.  So
        # the discovery run profiles this call as well and reports what it
        # launched by name; the counter leg adds those counts to the skip.
        # Guessing that the names never collide would be free and wrong.
        _seed_before_call()
        if cfg.get("profile_names") and cuda:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as pre_prof:
                context.sinogram = context.call("forward")
                _sync(torch, device)
            # Every kernel, not a top-N slice: this map exists so no name can
            # be missed, and a name missed here would make a launch skip land
            # in the wrong call without saying so.
            pre = summarize_profile(pre_prof, top=1 << 20)
            result["pre_call_launches"] = {
                str(entry["name"]): int(entry["calls"] or 0)
                for entry in pre["top_kernels"]}
            result["pre_call_profile"] = dict(
                distinct_kernels=pre["distinct_kernels"],
                kernel_launches=pre["kernel_launches"])
        else:
            context.sinogram = context.call("forward")
            _sync(torch, device)

    names = []
    for index in range(calls):
        last = index == calls - 1
        _seed_before_call()
        if last and cfg.get("profile_names"):
            from torch.profiler import ProfilerActivity, profile
            activities = [ProfilerActivity.CPU]
            if cuda:
                activities.append(ProfilerActivity.CUDA)
            with profile(activities=activities) as prof:
                start = time.perf_counter()
                out = context.call(direction)
                _sync(torch, device)
                wall = time.perf_counter() - start
            summary = summarize_profile(prof)
            summary["profiled_wall_s"] = wall
            result["profile"] = summary
            names = summary["top_kernels"]
        else:
            out = context.call(direction)
            _sync(torch, device)
        out = None
    result["kernel_names"] = names
    return result


def _worker_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return None


# ── the counter leg ───────────────────────────────────────────────────────────
def _run(cmd, timeout, env=None):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return dict(returncode=proc.returncode, stdout=proc.stdout,
                    stderr=proc.stderr, wall_s=time.perf_counter() - start,
                    timed_out=False)
    except subprocess.TimeoutExpired:
        return dict(returncode=None, stdout="", stderr="",
                    wall_s=time.perf_counter() - start, timed_out=True)
    except FileNotFoundError as exc:
        return dict(returncode=None, stdout="", stderr=str(exc),
                    wall_s=time.perf_counter() - start, timed_out=False,
                    missing=True)


def runner_env(cfg):
    """The environment a runner subprocess gets, set explicitly so nothing is
    inherited.

    The device pin and the calibration mode would each change what the runner
    builds, and this run pins neither: it uses the single device the job
    allocates.  The smoke and dry flags are forced off, because a runner is
    only ever spawned by a real CUDA run and an inherited flag would make it
    measure something else.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    env["MG51_SMOKE"] = "0"
    env["MG51_DRY"] = "0"
    env["MG51_NCU_RUNNER"] = json.dumps(cfg)
    return env


def parse_ncu_csv(text):
    """One dict of metric name to value per profiled kernel.

    Two layouts are accepted because the two ncu pages disagree and the format
    has moved between releases.  The wide layout puts one metric per column
    and one kernel per row.  The long layout puts one metric per ROW, with the
    metric name and its value in their own columns.  A unit row directly under
    the header is kept when it appears, because it is the only place the wide
    layout says what its numbers are in.  Numbers may carry thousands
    separators, which are stripped.
    """
    import csv as csv_module

    rows = list(csv_module.reader(text.splitlines()))

    def number(text_value):
        cleaned = str(text_value).replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return cleaned

    def parse_from(header_index):
        header = [cell_text.strip() for cell_text in rows[header_index]]
        body = rows[header_index + 1:]
        kernel_at = header.index("Kernel Name")
        if "Metric Name" in header and "Metric Value" in header:
            name_at = header.index("Metric Name")
            value_at = header.index("Metric Value")
            unit_at = (header.index("Metric Unit") if "Metric Unit" in header
                       else None)
            by_kernel = {}
            for row in body:
                if len(row) <= max(name_at, value_at, kernel_at):
                    continue
                key = row[kernel_at].strip()
                entry = by_kernel.setdefault(key, dict(kernel=key, metrics={},
                                                       units={}))
                entry["metrics"][row[name_at].strip()] = number(row[value_at])
                if unit_at is not None and len(row) > unit_at:
                    entry["units"][row[name_at].strip()] = row[unit_at].strip()
            return list(by_kernel.values())
        out, units = [], {}
        for row in body:
            if len(row) < len(header):
                continue
            values = [number(cell_text) for cell_text in row]
            if all(isinstance(value, str) for value in values):
                if not units:
                    units = {name: str(value).strip()
                             for name, value in zip(header, values)
                             if name and str(value).strip()}
                continue
            entry = dict(kernel=row[kernel_at].strip(), metrics={},
                         units=dict(units))
            for name, value in zip(header, values):
                if name and name != "Kernel Name":
                    entry["metrics"][name] = value
            out.append(entry)
        return out

    # Several rows can carry "Kernel Name": ncu prints a header per page, and
    # some releases repeat it.  Every candidate is parsed and the parse that
    # recovered the most kernels, and then the most numbers, wins.  The choice
    # of header row is therefore not a guess about the release.
    best, best_score = [], (0, 0)
    for index, row in enumerate(rows):
        if not any(cell_text.strip() == "Kernel Name" for cell_text in row):
            continue
        try:
            parsed = parse_from(index)
        except (ValueError, IndexError):
            continue
        scored = [entry for entry in parsed
                  if any(isinstance(value, (int, float))
                         for value in entry["metrics"].values())]
        numbers = sum(1 for entry in scored
                      for value in entry["metrics"].values()
                      if isinstance(value, (int, float)))
        if (len(scored), numbers) > best_score:
            best, best_score = scored, (len(scored), numbers)
    return best


def ncu_pattern(name):
    """The --kernel-name expression for a kernel, and whether it is exact.

    An inductor-generated kernel has a plain C identifier for a name, so the
    escaped name matches exactly.  A library kernel's name is a demangled C++
    signature with template arguments and spaces, which will not match ncu's
    name matching as written; for those the longest identifier run in the name
    is used instead and the row records that the filter was DERIVED rather
    than exact, because a derived filter can match more than one kernel.
    """
    text = str(name).strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return "regex:" + re.escape(text), True
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if not tokens:
        return None, False
    longest = max(tokens, key=len)
    return "regex:" + re.escape(longest), False


def _ncu_targets(summary, pre_launches=None, top=NCU_TOP):
    """The kernels ncu is pointed at, from one direction's discovery profile.

    Generated kernels first, because they are what a compiled body is made of
    and what a hand-written kernel would replace.  If a direction produced
    fewer than ``top`` generated kernels, the list is filled out with the next
    device kernels of any kind and every entry records which it is -- a
    compiled body that ends up running library kernels is a finding, not a
    reason to profile nothing.

    ``pre_launches`` maps kernel name to the number of launches the runner
    makes BEFORE the calls ncu is aimed at (the back direction's sinogram
    call).  It lands on the target as ``launches_before`` and the leg adds it
    to the launch skip.
    """
    kernels = list((summary or {}).get("top_kernels") or [])
    pre_launches = dict(pre_launches or {})
    generated_at = [index for index, entry in enumerate(kernels)
                    if str(entry.get("name", "")).startswith(GENERATED_PREFIX)]
    other_at = [index for index in range(len(kernels))
                if index not in set(generated_at)]
    chosen_at = generated_at[:top]
    if len(chosen_at) < top:
        chosen_at = chosen_at + other_at[:top - len(chosen_at)]
    out = []
    for rank, index in enumerate(chosen_at):
        entry = kernels[index]
        name = str(entry.get("name", ""))
        pattern, exact = ncu_pattern(name)
        out.append(dict(
            rank=rank, name=entry.get("name"),
            generated=name.startswith(GENERATED_PREFIX),
            launches_per_call=int(entry.get("calls") or 0),
            launches_before=int(pre_launches.get(name, 0)),
            device_us_per_call=entry.get("device_us"),
            mean_device_us=entry.get("mean_device_us"),
            filter=pattern, filter_is_exact=exact))
    return out


def ncu_leg(plan, torch_python, results_dir, sink, profiles):
    """Probe, discover, profile.  Every step records what it saw, and none of
    them changes the exit code.

    ``profiles`` is the profiler leg's own per-direction summary at the
    counter cell, carried in so the leg can report whether the names it
    discovered in the runner process agree with the names the main process
    saw.  They should: same script, same cell, same compile order.  A
    disagreement means the compiled kernel names are not stable across
    processes, which would change how any later run has to aim a filter.
    """
    leg = dict(kind="ncu_leg", attempted=True, enabled=NCU_ENABLED,
               cell_id=NCU_CELL_ID, top=NCU_TOP,
               runner_calls=NCU_RUNNER_CALLS, skip_calls=NCU_SKIP_CALLS,
               metrics_full=list(METRICS_FULL),
               metrics_core=list(METRICS_CORE), directions={})
    counter_cell = next((entry for entry in plan
                         if entry["cell_id"] == NCU_CELL_ID), None)
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG51_NCU=0")
        return leg
    if SMOKE or DEVICE != "cuda":
        leg.update(attempted=False,
                   reason="this run is on the CPU, where there are no GPU "
                          "performance counters to read")
        return leg
    if counter_cell is None:
        leg.update(attempted=False,
                   reason=f"cell {NCU_CELL_ID} is not in this run's plan, and "
                          "the counter leg profiles that cell only")
        return leg
    ncu = shutil.which("ncu")
    leg["ncu_path"] = ncu
    if ncu is None:
        leg.update(attempted=False,
                   reason="ncu is not on PATH; the sbatch loads the cuda "
                          "module, which is where Nsight Compute lives")
        return leg
    version = _run([ncu, "--version"], 60)
    leg["ncu_version"] = (version["stdout"].strip().splitlines()[:2]
                          if version["stdout"] else version["stderr"][:200])

    # ── step 1: the permission probe ────────────────────────────────────────
    probe = _run([ncu, "--launch-count", "1", "--metrics",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  torch_python, "-u", os.path.abspath(__file__)],
                 NCU_PROBE_TIMEOUT_S, env=runner_env(dict(mode="trivial")))
    blob = (probe["stdout"] or "") + (probe["stderr"] or "")
    refused = any(marker.lower() in blob.lower()
                  for marker in NCU_PERMISSION_MARKERS)
    # The probe's own worker prints a result line, so a probe that produced no
    # metric can be told apart from one whose python never ran at all.  The
    # two have different remedies and reporting them as one would send a
    # reader to the wrong place.
    worker_ran = "__RESULT__" in blob
    empty = "sm__warps_active" not in blob
    leg["permission_probe"] = dict(
        returncode=probe["returncode"], timed_out=probe["timed_out"],
        refused=refused, profile_empty=empty, worker_ran=worker_ran,
        message=blob.strip()[-800:])
    leg["profiler_permitted"] = not (refused or empty or probe["timed_out"])
    if not leg["profiler_permitted"]:
        if refused:
            leg["reason"] = ("the driver refused performance counters to this "
                             "user")
        elif probe["timed_out"]:
            leg["reason"] = (f"the permission probe did not finish within "
                             f"{NCU_PROBE_TIMEOUT_S} s")
        elif not worker_ran:
            leg["reason"] = ("the probe's own python never ran, so this says "
                             "nothing about counter permission; read the "
                             "message on the row")
        else:
            leg["reason"] = ("the probe ran and produced no metric, so the "
                             "counters are unavailable")
        return leg

    leg_start = time.perf_counter()
    leg["budget_s"] = NCU_LEG_BUDGET_S
    for direction in counter_cell["directions"]:
        record = dict(direction=direction, kernels={})
        leg["directions"][direction] = record

        # ── step 2: the kernel names, discovered in the runner process ──────
        # Two calls, and the SECOND is the one profiled: a cold call can carry
        # launches a warm one does not, and the per-call launch count read
        # here is the arithmetic the launch skip is built from.
        cfg = dict(mode="calls", direction=direction, cell_id=NCU_CELL_ID,
                   calls=2, profile_names=True)
        discovery = _run([torch_python, "-u", os.path.abspath(__file__)],
                         NCU_DISCOVERY_TIMEOUT_S, env=runner_env(cfg))
        row = _worker_result(discovery["stdout"] or "")
        record["discovery"] = dict(
            returncode=discovery["returncode"],
            timed_out=discovery["timed_out"], wall_s=discovery["wall_s"],
            error=(None if row else (discovery["stderr"] or "")[-1500:]))
        if not row or not row.get("profile"):
            record["reason"] = ("the discovery run produced no profile, so "
                                "there are no runtime kernel names to aim a "
                                "filter at; a filter is not guessed")
            continue
        targets = _ncu_targets(row.get("profile"),
                               row.get("pre_call_launches"))
        record["targets"] = targets
        record["pre_call_launches"] = row.get("pre_call_launches")
        record["discovery_top"] = (row.get("profile") or {}).get("top_kernels")
        # Raw top names against raw top names, from the two processes.  Both
        # sides are the unfiltered device-time ranking, because comparing the
        # ranking against the TARGET list would compare two different things:
        # the targets prefer generated kernels, and the ranking does not.
        main_top = [entry.get("name") for entry
                    in ((profiles.get(direction) or {}).get("top_kernels")
                        or [])][:NCU_TOP]
        runner_top = [entry.get("name")
                      for entry in (record["discovery_top"] or [])][:NCU_TOP]
        record["main_process_top"] = main_top
        record["runner_process_top"] = runner_top
        record["names_agree_with_main_process"] = (
            runner_top == main_top if main_top else None)
        if not targets:
            record["reason"] = "the discovery profile listed no device kernel"
            continue

        # ── step 3: one profiled launch per kernel ──────────────────────────
        for target in targets:
            key = f"{direction}_k{target['rank']}"
            spent = time.perf_counter() - leg_start
            if spent > NCU_LEG_BUDGET_S:
                record["kernels"][key] = dict(
                    target, reason=f"the counter leg's {NCU_LEG_BUDGET_S} s "
                                   f"budget was already spent ({spent:.0f} s);"
                                   " this kernel was not profiled")
                continue
            if not target["filter"]:
                record["kernels"][key] = dict(
                    target, reason="this kernel's name holds no identifier to "
                                   "build a filter from")
                continue
            entry = dict(target, attempts=[])
            print(f"    {direction} rank {target['rank']}: "
                  f"{str(target['name'])[:70]}", flush=True)
            cfg = dict(mode="calls", direction=direction,
                       cell_id=NCU_CELL_ID, calls=NCU_RUNNER_CALLS)
            for set_name, metrics, skip_policy in NCU_ATTEMPTS:
                if skip_policy == "warm":
                    # Past everything the runner launches before the calls ncu
                    # is aimed at, then past NCU_SKIP_CALLS whole calls of this
                    # kernel.  The launch that gets profiled is therefore the
                    # first one of a warm call.
                    skip = min(NCU_MAX_SKIP,
                               target.get("launches_before", 0)
                               + target["launches_per_call"] * NCU_SKIP_CALLS)
                else:
                    skip = 0
                cmd = [ncu, "--csv", "--page", "raw", "--target-processes",
                       "all", "--kernel-name", target["filter"],
                       "--launch-skip", str(skip), "--launch-count", "1",
                       "--metrics", ",".join(metrics),
                       torch_python, "-u", os.path.abspath(__file__)]
                got = _run(cmd, NCU_TIMEOUT_S, env=runner_env(cfg))
                log_path = os.path.join(
                    results_dir,
                    f"mg51_ncu_{key}_{set_name}_skip{skip}.log")
                with open(log_path, "w") as log_sink:
                    log_sink.write(" ".join(cmd) + "\n")
                    log_sink.write(f"MG51_NCU_RUNNER={json.dumps(cfg)}\n\n")
                    log_sink.write(got["stdout"] or "")
                    log_sink.write("\n----- stderr -----\n")
                    log_sink.write(got["stderr"] or "")
                parsed = parse_ncu_csv(got["stdout"] or "")
                entry["attempts"].append(dict(
                    metric_set=set_name, launch_skip=skip,
                    returncode=got["returncode"], timed_out=got["timed_out"],
                    wall_s=got["wall_s"], kernels=len(parsed), log=log_path))
                if parsed:
                    entry.update(metric_set=set_name, launch_skip=skip,
                                 kernels=parsed, wall_s=got["wall_s"],
                                 log=log_path)
                    break
                if got["timed_out"]:
                    # Retrying a timeout with a smaller metric set would spend
                    # the same minutes again for the same reason.
                    entry["reason"] = (
                        f"the profile did not finish within {NCU_TIMEOUT_S} s")
                    break
            if "kernels" not in entry and "reason" not in entry:
                entry["reason"] = ("no kernel matched the filter, or the "
                                   "profile was empty; the raw output is in "
                                   "the logs above")
            record["kernels"][key] = entry
            sink.write(json.dumps(dict(kind="ncu_kernel", cell_id=NCU_CELL_ID,
                                       direction=direction, key=key,
                                       **counter_row(entry))) + "\n")
            sink.flush()
    return leg


# ── reading the counters ──────────────────────────────────────────────────────
def _metric(kernel, name, default=None):
    metrics = (kernel or {}).get("metrics") or {}
    if name in metrics:
        return metrics[name]
    for key, value in metrics.items():
        if key.split(" ")[0] == name:
            return value
    return default


def _number(value):
    return value if isinstance(value, (int, float)) else None


def _duration_ms(kernel):
    """gpu__time_duration.sum in milliseconds, using the unit ncu reported.

    ncu emits this metric in nanoseconds, microseconds or milliseconds
    depending on the release and the page, so the unit is read rather than
    assumed.  An unrecognized unit returns None and the raw value stays on the
    row.
    """
    value = _number(_metric(kernel, "gpu__time_duration.sum"))
    if value is None:
        return None
    units = (kernel or {}).get("units") or {}
    unit = ""
    for key, text in units.items():
        if key.split(" ")[0] == "gpu__time_duration.sum":
            unit = str(text).strip().lower()
            break
    if unit in ("nsecond", "ns", "nanosecond"):
        return value / 1e6
    if unit in ("usecond", "us", "microsecond"):
        return value / 1e3
    if unit in ("msecond", "ms", "millisecond"):
        return value
    if unit in ("second", "s"):
        return value * 1e3
    return None


def counter_row(entry):
    """The counter columns this run reads, pulled out of one profiled
    kernel's parsed metrics.

    A metric that the winning set did not collect is None on the row rather
    than zero: zero would read as a measurement of nothing, and the whole
    point of the fallback set is that a reader can tell which columns were
    actually collected.
    """
    kernels = entry.get("kernels") or []
    first = kernels[0] if kernels else None
    row = dict(name=entry.get("name"), rank=entry.get("rank"),
               generated=entry.get("generated"),
               filter=entry.get("filter"),
               filter_is_exact=entry.get("filter_is_exact"),
               launches_per_call=entry.get("launches_per_call"),
               launches_before=entry.get("launches_before"),
               metric_set=entry.get("metric_set"),
               launch_skip=entry.get("launch_skip"),
               log=entry.get("log"), reason=entry.get("reason"),
               profiled=bool(kernels))
    if first is None:
        return row
    dram_read = _number(_metric(first, "dram__bytes_read.sum"))
    dram_write = _number(_metric(first, "dram__bytes_write.sum"))
    row.update(
        ncu_kernel=first.get("kernel"),
        duration_ms=_duration_ms(first),
        achieved_occupancy_pct=_number(_metric(
            first, "sm__warps_active.avg.pct_of_peak_sustained_active")),
        sm_throughput_pct=_number(_metric(
            first, "sm__throughput.avg.pct_of_peak_sustained_elapsed")),
        memory_throughput_pct=_number(_metric(
            first,
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")),
        l2_hit_pct=_number(_metric(first, "lts__t_sector_hit_rate.pct")),
        l1_hit_pct=_number(_metric(
            first, "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct")),
        sectors_per_request=_number(_metric(
            first,
            "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")),
        long_scoreboard_ratio=_number(_metric(
            first,
            "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio")),
        registers_per_thread=_number(_metric(
            first, "launch__registers_per_thread")),
        grid_size=_number(_metric(first, "launch__grid_size")),
        block_size=_number(_metric(first, "launch__block_size")),
        dram_bytes_read=dram_read, dram_bytes_write=dram_write,
        dram_bytes=(None if dram_read is None and dram_write is None
                    else (dram_read or 0.0) + (dram_write or 0.0)),
        atomic_sectors=_number(_metric(first, "lts__t_sectors_op_atom.sum")),
        reduction_sectors=_number(_metric(first, "lts__t_sectors_op_red.sum")))
    return row


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=3):
    if value is None:
        return f'{"-":>{width}}'
    if isinstance(value, str):
        return f"{value:>{width}}"
    if kind == "d":
        # An integer format takes no precision, and a float that reached an
        # integer column is rounded rather than refused: a count that arrived
        # as a float is still a count.
        return f"{int(round(float(value))):>{width}d}"
    return f"{value:>{width}.{prec}{kind}}"


def _short(name, width=NAME_COL):
    text = str(name)
    return text if len(text) <= width else text[:width - 3] + "..."


def timing_table(rows):
    print("\n── TIMING: the production route on plain clocks ────────────────")
    print(f'  {"cell":<6}{"direction":<10}{"cold s":>10}{"warm s":>10}'
          f'{"enqueue s":>11}{"dev span s":>12}{"enq/wall":>10}{"spread":>9}')
    for row in rows:
        if row.get("skipped") or row.get("error"):
            print(f'  {row.get("cell_id", "?"):<6}'
                  f'{row.get("direction", "?"):<10}'
                  f'  {str(row.get("reason") or row.get("error"))[:70]}')
            continue
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(row.get("cold_s"))}{_fmt(row.get("warm_s"))}'
              f'{_fmt(row.get("enqueue_s"), 11, prec=4)}'
              f'{_fmt(row.get("device_span_s"), 12)}'
              f'{_fmt(row.get("enqueue_frac_of_wall"), 10)}'
              f'{_fmt(row.get("spread"), 9)}')
    print("  enqueue s is the host time to RETURN from the call without a")
    print("  synchronize; dev span s is the CUDA-event span around the same")
    print("  call.  enq/wall near 1 means the host is the limit.")


def profile_table(rows):
    print("\n── PROFILE: what one call runs ─────────────────────────────────")
    print(f'  {"cell":<6}{"direction":<10}{"kernels":>9}{"gen":>5}'
          f'{"launches":>10}{"dev ms":>10}{"host ms":>10}{"wait ms":>10}'
          f'{"host/dev":>10}{"wall s":>9}')
    for row in rows:
        if row.get("skipped") or row.get("error"):
            print(f'  {row.get("cell_id", "?"):<6}'
                  f'{row.get("direction", "?"):<10}'
                  f'  {str(row.get("reason") or row.get("error"))[:70]}')
            continue
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(row.get("distinct_kernels"), 9, "d", 0)}'
              f'{_fmt(row.get("distinct_generated_kernels"), 5, "d", 0)}'
              f'{_fmt(row.get("kernel_launches"), 10, "d", 0)}'
              f'{_fmt((row.get("kernel_device_us") or 0) / 1000.0, 10, prec=2)}'
              f'{_fmt((row.get("host_us_no_sync") or 0) / 1000.0, 10, prec=2)}'
              f'{_fmt((row.get("sync_us") or 0) / 1000.0, 10, prec=2)}'
              f'{_fmt(row.get("host_no_sync_over_device"), 10)}'
              f'{_fmt(row.get("profiled_wall_s"), 9)}')
    print("  host ms excludes the synchronize; wait ms IS the synchronize.")
    print("  host/dev above 1 means the host spent longer issuing the work")
    print("  than the device spent running it.")


def kernel_tables(rows):
    for row in rows:
        if row.get("skipped") or row.get("error") or not row.get("top_kernels"):
            continue
        total = row.get("kernel_device_us") or 0.0
        print(f'\n── TOP KERNELS: cell {row["cell_id"]} {row["direction"]} '
              f"─────────────────────────────")
        print(f'  {"kernel":<{NAME_COL}}{"calls":>7}{"dev ms":>10}'
              f'{"mean us":>10}{"share":>8}')
        for entry in row["top_kernels"]:
            share = (entry["device_us"] / total) if total else None
            print(f'  {_short(entry["name"]):<{NAME_COL}}'
                  f'{_fmt(entry.get("calls"), 7, "d", 0)}'
                  f'{_fmt(entry["device_us"] / 1000.0, 10, prec=2)}'
                  f'{_fmt(entry.get("mean_device_us"), 10, prec=1)}'
                  f'{_fmt(share, 8, prec=3)}')
        if row.get("top_transfers"):
            for entry in row["top_transfers"]:
                print(f'  {_short(entry["name"]):<{NAME_COL}}'
                      f'{_fmt(entry.get("calls"), 7, "d", 0)}'
                      f'{_fmt(entry["device_us"] / 1000.0, 10, prec=2)}'
                      f'{_fmt(entry.get("mean_device_us"), 10, prec=1)}'
                      f'{"  (copy)":>8}')


def counter_table(leg):
    directions = (leg.get("directions") or {})
    if not directions:
        return
    print("\n── COUNTERS: Nsight Compute on the top generated kernels ───────")
    print(f'  {"kernel":<{NAME_COL}}{"dir":<9}{"set":>6}{"dur ms":>9}'
          f'{"occ %":>8}{"SM %":>8}{"mem %":>8}{"L2 %":>8}{"L1 %":>8}'
          f'{"DRAM GB":>9}')
    for direction, record in directions.items():
        for entry in (record.get("kernels") or {}).values():
            row = counter_row(entry)
            if not row.get("profiled"):
                print(f'  {_short(row.get("name")):<{NAME_COL}}{direction:<9}'
                      f'  {str(row.get("reason"))[:60]}')
                continue
            dram = row.get("dram_bytes")
            print(f'  {_short(row.get("name")):<{NAME_COL}}{direction:<9}'
                  f'{str(row.get("metric_set")):>6}'
                  f'{_fmt(row.get("duration_ms"), 9, prec=3)}'
                  f'{_fmt(row.get("achieved_occupancy_pct"), 8, prec=1)}'
                  f'{_fmt(row.get("sm_throughput_pct"), 8, prec=1)}'
                  f'{_fmt(row.get("memory_throughput_pct"), 8, prec=1)}'
                  f'{_fmt(row.get("l2_hit_pct"), 8, prec=1)}'
                  f'{_fmt(row.get("l1_hit_pct"), 8, prec=1)}'
                  f'{_fmt(None if dram is None else dram / 2 ** 30, 9, prec=3)}')
    print("  Durations here are ncu's, not wall times: ncu serializes kernels")
    print("  and replays each one to collect its counters.  The timing leg")
    print("  owns time.  A blank column is one the winning metric set did not")
    print("  collect, not a zero.")


def reading_guide():
    print("\n── HOW TO READ THIS ────────────────────────────────────────────")
    print("  Memory-bound: memory throughput high and well above SM")
    print("  throughput, with a low L1 or L2 hit rate and DRAM bytes near")
    print("  what the algorithm has to move.  A kernel would help only by")
    print("  moving fewer bytes.")
    print("  Latency-bound: both throughputs low, occupancy low, and the")
    print("  long-scoreboard stall ratio high.  More concurrency or fewer")
    print("  dependent loads is the remedy.")
    print("  Compute-bound: SM throughput high with memory throughput well")
    print("  below it.  This is the case a hand-written kernel helps least.")
    print("  Launch-bound: read the PROFILE table instead.  Many launches,")
    print("  short mean kernel durations, and host time above device time")
    print("  mean the body's shape is the problem, not any one kernel.")
    print("  This run decides nothing; the verdict is read by a person from")
    print("  the tables above and the rows in the jsonl.")


def summarize(header, setups, timings, profiles, leg, out_path):
    print()
    timing_table(timings)
    profile_table(profiles)
    kernel_tables(profiles)
    counter_table(leg)
    reading_guide()

    checks = []
    for setup in setups:
        if setup.get("error") and not setup.get("oom"):
            checks.append(f'cell {setup.get("cell_id")} setup failed: '
                          f'{str(setup.get("error"))[:200]}')
        if setup.get("device_ok") is False:
            checks.append(f'cell {setup.get("cell_id")} realized '
                          f'{setup.get("device_realized")} and not '
                          f'{setup.get("device_expected")}')
        if setup.get("bodies_ok") is False:
            checks.append(
                f'cell {setup.get("cell_id")} did not bind a torch body in '
                f'both directions: {setup.get("torch_body_directions")}.  '
                "This run measures compiled bodies, so that is a different "
                "subject, not a different number.")
    for row in timings + profiles:
        if row.get("error") and not row.get("oom"):
            checks.append(f'{row.get("kind")} {row.get("cell_id")}/'
                          f'{row.get("direction")}: '
                          f'{str(row.get("error"))[:200]}')
        # A body that fell back to eager is not the body this run is named
        # for.  The library is right to fall back -- an eager retry is how it
        # survives a broken backend -- but a run that measured eager python
        # and reported it as a compiled body would answer a different
        # question with the same table.
        state = row.get("compile_state") or {}
        if state.get("multiaxis_compile_errors"):
            checks.append(
                f'{row.get("cell_id")}/{row.get("direction")}: a multiaxis '
                f'body did NOT compile and was rebound to eager: '
                f'{state["multiaxis_compile_errors"]}')
    if not header.get("tree_witnesses", {}).get("ok"):
        checks.append(f'tree witnesses: {header.get("tree_witnesses")}')

    oom = [f'{row.get("kind")} {row.get("cell_id")}/{row.get("direction")}'
           for row in setups + timings + profiles if row.get("oom")]
    if oom:
        print(f"\n{len(oom)} leg(s) ran out of device memory and are recorded "
              f"as that: {oom}.  An out-of-memory is this run's answer for "
              "that leg, not a failure of the run.")
    if header.get("gpu_hot_or_throttled"):
        print("\nNOTE: the device sampled hot or throttled.  The walls are a "
              "rate reading, so read them with that in mind.")
    unprofiled = []
    for direction, record in (leg.get("directions") or {}).items():
        for key, entry in (record.get("kernels") or {}).items():
            if not entry.get("kernels"):
                unprofiled.append(key)
    if unprofiled:
        print(f"\n{len(unprofiled)} kernel(s) produced no counter row: "
              f"{unprofiled}.  The counter leg never changes the exit code.")
    if leg.get("reason"):
        print(f'\ncounter leg: {leg["reason"]}')

    healthy = not checks
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"}.  It covers five things: '
          "every planned leg produced a row or a recorded out-of-memory, the "
          "realized device is the one asked for, both directions bound a "
          "torch body, both bodies really compiled rather than falling back "
          "to eager, and the tree witnesses hold.")
    for line in checks:
        print(f"  FAIL: {line}")
    print("The counter leg's absence never changes it, and neither does what "
          "any leg found.")
    return dict(kind="summary", healthy=healthy, checks=checks, oom=oom,
                unprofiled=unprofiled,
                profiler_permitted=leg.get("profiler_permitted"),
                profiler_reason=leg.get("reason"),
                timings=len(timings), profiles=len(profiles),
                out_path=out_path)


# ── the runner ────────────────────────────────────────────────────────────────
def _dry_run(plan):
    print(f"mg51 multiaxis compiled-body counters: {len(plan)} cell(s), "
          f"device {DEVICE}")
    print("  what limits the compiled projection bodies of "
          "MultiAxisParallelModel: how many generated kernels a call runs, "
          "how many launches, where the device time goes, host enqueue "
          "against device busy, and the hardware counters on the top "
          "kernels.  It decides nothing.")
    print(f"  results, and one ncu log per attempt -> {RESULTS_DIR}")
    print(f'\n  {"cell":<6}{"sinogram":>20}{"recon (mirrored)":>24}'
          f'{"warm":>6}{"prof":>6}  directions')
    for entry in plan:
        print(f'  {entry["cell_id"]:<6}'
              f'{str(tuple(entry["cell"])):>20}'
              f'{str(tuple(entry["mirrored_recon_shape"])):>24}'
              f'{entry["warm"]:>6}{entry["profile_repeats"]:>6}  '
              f'{",".join(entry["directions"])}')
    print("  the recon shapes above come from this file's mirror of the "
          "geometry's own rule; the real run reads the model's recon_shape "
          "and records whether the two agreed.")
    for entry in plan:
        print(f'    {entry["cell_id"]}: {entry["note"]}')
    print(f"\n  protocol: the widening floors refresh's -- azimuths over half "
          f"a turn, elevations across +/- {ELEVATION_HALF_RANGE}, the "
          f"shepp-logan low-dynamic-range phantom, seed {SEED} reset before "
          "every call, one device named explicitly")
    print("  route: TomographyModel.sparse_forward_project and "
          "sparse_back_project, the funnel the reconstruction itself calls")
    print(f"\n  timing leg: 1 cold + {WARM_REPEATS} warm call(s) per direction "
          "and cell, on a host clock with a device synchronize, plus the "
          "enqueue clock and a CUDA-event device span"
          + (" (both SKIPPED in the smoke)" if SMOKE else ""))
    print("  profiler leg: torch.profiler with CPU and CUDA activities "
          "around one warm call per repeat; the row carries the distinct "
          f"kernel count, the launch count, device and host totals, and the "
          f"top {TOP_KERNELS} kernels by device time")
    print(f"\n  counter leg: {'on' if NCU_ENABLED else 'off (MG51_NCU=0)'}"
          + (", cell " + NCU_CELL_ID if NCU_ENABLED else ""))
    if NCU_ENABLED and not SMOKE:
        print(f"    top {NCU_TOP} generated kernels per direction, names "
              "discovered at runtime in the runner process")
        print(f"    runner: {NCU_RUNNER_CALLS} production calls; the launch "
              f"skip steps past whatever the runner launches before them and "
              f"then past {NCU_SKIP_CALLS} whole call(s) of that kernel, "
              f"capped at {NCU_MAX_SKIP}")
        print("    attempts per kernel, in order:")
        for set_name, metrics, policy in NCU_ATTEMPTS:
            print(f"      metric set {set_name} ({len(metrics)} metrics), "
                  f"page raw, launch count 1, skip {policy}")
        print(f"    bounds: {NCU_TIMEOUT_S} s per attempt, "
              f"{NCU_LEG_BUDGET_S} s for the whole leg, "
              f"{NCU_PROBE_TIMEOUT_S} s for the permission probe, "
              f"{NCU_DISCOVERY_TIMEOUT_S} s per discovery run")
        print("    the counter leg never changes the exit code")
    elif SMOKE:
        print("    SKIPPED in the smoke and recorded as skipped: there are no "
              "GPU performance counters on the CPU")
    print("\n  cell b's legs are wrapped: an out-of-memory is caught and "
          "recorded as that leg's result, and the run continues")
    print("  exit code = instrument health: every planned leg produced a row "
          "or a recorded out-of-memory, the realized device is the one asked "
          "for, both directions bound a torch body, both bodies really "
          "compiled rather than falling back to eager, and the tree "
          "witnesses hold")
    print("  no library file is touched: every call goes through the public "
          "projection funnel")


def _leg_error(kind, context, direction, exc):
    """One failed leg, recorded as that leg's result.

    An out-of-memory is marked as such and is not a health failure: the 1024
    cell is near one device's edge, and 'it did not fit' is an answer to the
    question this run asks.
    """
    return dict(kind=kind, cell_id=context.cell_id, cell=list(context.cell),
                direction=direction, device=DEVICE, error=str(exc)[:1200],
                oom=_is_oom(exc), traceback=traceback.format_exc()[-2000:])


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    import torch

    if not SMOKE and not torch.cuda.is_available():
        print("this run needs CUDA; use MG51_SMOKE=1 for the CPU plumbing "
              "pass")
        return 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        RESULTS_DIR, f"mg51_multiaxis_counters_{RUN_LABEL}_{stamp}.jsonl")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    print(f"mg51 multiaxis compiled-body counters on {RUN_LABEL} ({DEVICE}); "
          f"{len(plan)} cell(s) -> {out_path}", flush=True)

    header = dict(kind="run", smoke=SMOKE, device=DEVICE,
                  plan=plan, seed=SEED, warm_repeats=WARM_REPEATS,
                  top_kernels=TOP_KERNELS, ncu_enabled=NCU_ENABLED,
                  torch=torch.__version__, python=platform.python_version(),
                  node=platform.node(), run_label=RUN_LABEL, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  device_count=(torch.cuda.device_count() if cuda else 0),
                  tree_witnesses=tree_witnesses(),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                  health_before=sample_gpu_health())
    try:
        import mbirtorch
        header["mbirtorch_file"] = mbirtorch.__file__
    except Exception as exc:                                      # noqa: BLE001
        header["mbirtorch_file"] = f"{type(exc).__name__}: {exc}"

    setups, timings, profiles = [], [], []
    counter_cell_profiles = {}
    with open(out_path, "w") as sink:
        header["profiler_warm_up"] = _warm_up_profiler(
            torch, torch.device(DEVICE + (":0" if cuda else "")))
        sink.write(json.dumps(header) + "\n")
        sink.flush()

        for spec in plan:
            print(f'\n  cell {spec["cell_id"]} {tuple(spec["cell"])}',
                  flush=True)
            context = CellContext(spec)
            try:
                setup = context.build()
            except Exception as exc:                              # noqa: BLE001
                setup = dict(kind="cell_setup", cell_id=spec["cell_id"],
                             cell=list(spec["cell"]), device=DEVICE,
                             error=str(exc)[:1200], oom=_is_oom(exc),
                             traceback=traceback.format_exc()[-2000:])
                setups.append(setup)
                sink.write(json.dumps(setup) + "\n")
                sink.flush()
                print(f'    setup failed: {str(exc)[:200]}', flush=True)
                context.release()
                continue
            setups.append(setup)
            sink.write(json.dumps(setup) + "\n")
            sink.flush()
            print(f'    recon {tuple(setup["recon_shape"])}, '
                  f'{setup["num_pixels"]} pixels, view batch '
                  f'{setup["view_batch"]}', flush=True)

            if cuda:
                torch.cuda.reset_peak_memory_stats(context.model.torch_device)
            # The forward direction runs first whatever the selection order:
            # its cold call makes the sinogram the back direction needs.
            ordered = [name for name in DIRECTIONS
                       if name in spec["directions"]]
            if "back" in ordered and "forward" not in ordered:
                # A back-only selection (a re-run of one leg) has no forward
                # cold call to make its input, so one untimed forward call is
                # made here and recorded as having been made.
                try:
                    setup["untimed_forward_for_sinogram"] = (
                        context.ensure_sinogram())
                except Exception as exc:                          # noqa: BLE001
                    setup["untimed_forward_for_sinogram"] = False
                    setup["sinogram_error"] = str(exc)[:600]
                    setup["oom"] = _is_oom(exc)
                    _free_device_memory(torch)
                sink.write(json.dumps(dict(setup,
                                           kind="cell_setup_update")) + "\n")
                sink.flush()
            for direction in ordered:
                print(f"    timing {direction}", flush=True)
                try:
                    if not context.ready(direction):
                        row = dict(kind="timing", cell_id=context.cell_id,
                                   cell=list(context.cell),
                                   direction=direction, device=DEVICE,
                                   skipped=True,
                                   reason="the back projection has no "
                                          "sinogram: the forward call that "
                                          "makes it did not run or did not "
                                          "finish; see the cell setup row")
                    else:
                        row = timing_leg(context, direction)
                except Exception as exc:                          # noqa: BLE001
                    row = _leg_error("timing", context, direction, exc)
                    _free_device_memory(torch)
                timings.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()

            for direction in ordered:
                print(f"    profiling {direction}", flush=True)
                try:
                    if not context.ready(direction):
                        row = dict(kind="profile", cell_id=context.cell_id,
                                   cell=list(context.cell),
                                   direction=direction, device=DEVICE,
                                   skipped=True,
                                   reason="no input for this direction; see "
                                          "the timing row")
                    else:
                        row = profiler_leg(context, direction,
                                           spec["profile_repeats"])
                except Exception as exc:                          # noqa: BLE001
                    row = _leg_error("profile", context, direction, exc)
                    _free_device_memory(torch)
                profiles.append(row)
                if spec["cell_id"] == NCU_CELL_ID and not row.get("error"):
                    counter_cell_profiles[direction] = row
                sink.write(json.dumps(row) + "\n")
                sink.flush()
            context.release()

        print("\n  counter leg", flush=True)
        leg = ncu_leg(plan, sys.executable, RESULTS_DIR, sink,
                      counter_cell_profiles)
        sink.write(json.dumps(leg) + "\n")
        sink.flush()

        health_after = sample_gpu_health()
        header["health_after"] = health_after
        header["gpu_hot_or_throttled"] = bool(
            health_is_hot(header.get("health_before") or [])
            or health_is_hot(health_after))
        sink.write(json.dumps(dict(kind="run_close",
                                   health_after=health_after,
                                   gpu_hot_or_throttled=header[
                                       "gpu_hot_or_throttled"])) + "\n")
        sink.flush()
        summary = summarize(header, setups, timings, profiles, leg, out_path)
        sink.write(json.dumps(summary) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    if NCU_RUNNER_RAW:
        try:
            runner_cfg = json.loads(NCU_RUNNER_RAW)
            if not isinstance(runner_cfg, dict):
                runner_cfg = dict(mode="calls")
        except ValueError:
            # MG51_NCU_RUNNER=1 is accepted as "run the default calls", so a
            # person can drive the runner by hand without writing json.
            runner_cfg = dict(mode="calls")
        try:
            runner_out = ncu_runner(runner_cfg)
        except Exception:                                         # noqa: BLE001
            runner_out = dict(runner_cfg,
                              error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(runner_out))
    else:
        sys.exit(main())
