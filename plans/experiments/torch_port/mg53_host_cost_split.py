"""mg53 -- WHERE THE HOST TIME OF ONE COMPILED PROJECTION CALL GOES.

WHAT THIS RUN IS.  A separate measurement (mg51) timed one compiled multiaxis
projection call on two clocks at once: the host time to return from the public
call without synchronizing, and the profiler's account of what the host spent
that time on.  At the (1024, 1008, 992) multiaxis cell the two do not add up.
About 35 ms of host time PER VIEW BATCH is neither the kernel-launch API nor
the final synchronize.  At the (512, 448, 384) cell the same unaccounted host
time is about 0.35 ms per view batch.  A hundredfold difference in per-batch
host cost is a large enough gap that guessing at it is not worth doing, and
nothing in the launch counts explains it.  This run attributes it.

THE FOUR CANDIDATE MECHANISMS, and the leg that separates each from the rest:

  1. WORK INSIDE THE COMPILED CALLABLE.  A compiled body is not only its
     kernels: calling it costs a dynamo cache lookup, a guard evaluation over
     the arguments, and the inductor wrapper's own python before any launch is
     issued.  Whatever that costs is charged INSIDE the body call, so the
     body_split leg puts it on one side of its split and the host_ops leg
     names it (the profiler's TorchDynamo, Torch-Compiled and CompiledFunction
     rows).
  2. ALLOCATOR CALLS THAT BLOCK.  cudaMalloc and cudaFree are synchronizing
     driver calls, and an allocation that misses the caching allocator's
     free-block pools triggers a retry that frees cached blocks first.  At the
     1024-class cell the per-batch transients are large enough that this is
     plausible where it would not be at 512.  The body_split leg reads the
     allocator's own counters across the same call, host_ops names cudaMalloc
     and cudaFree directly, and the ablation arm re-times the call under a
     different allocator setting.
  3. A HIDDEN PER-BATCH SYNCHRONIZATION.  A device-to-host copy, an .item(),
     or anything else that reads a device value on the host makes the host
     wait for the device once per view batch.  That would show up as host time
     with no host work under it.  The sync_detector leg turns on torch's own
     synchronization warnings and records what the call trips, by name.
  4. HOST WORK IN THE DRIVER LOOP.  The view-batch loop around the compiled
     body is ordinary python: it slices the view parameters, allocates or
     indexes the output block, and assigns or adds each batch's result.  That
     work is OUTSIDE the body call, so the body_split leg puts it on the other
     side of its split.

THIS RUN DECIDES NOTHING AND IMPLEMENTS NOTHING.  It writes no kernel and
changes no library file; every call goes through the public projection funnel.
The verdict is read by a person from the tables in this log and the rows in
the jsonl.  The exit code reports instrument health only: whether the
measurement was taken correctly, never what it found.

## THE TWO CELLS

Cell A, multiaxis sinogram (512, 448, 384).  Cell B, (1024, 1008, 992).  These
are the two cells mg51 measured, so the split here lands on the same numbers
without adjustment.  One H100, one device, placed explicitly.

## THE PROTOCOL IS MG51'S

Same model construction (azimuths evenly spaced over half a turn, elevations
swept across +/- 0.5 radians), same shepp-logan low-dynamic-range phantom,
same seed 13 reset before every call, same public route
(``TomographyModel.sparse_forward_project`` and ``sparse_back_project``).  The
phantom is forward projected once to make the sinogram the back projection is
given, so nothing is projected twice for bookkeeping.

## THE FOUR LEGS, per cell and direction

Every leg runs AFTER one untimed warm-up call in that direction, so no leg
pays the compile.

THE TIMING LEG reproduces mg51's timing semantics so this run's rows can be
laid beside mg51's.  The warm-up call is the cold call; a second timed cold
call would measure nothing new.  It times MG53_WARM warm calls, each on two
clocks: the plain host wall with a final synchronize, and the enqueue time,
which is the host time for the public call to RETURN before that synchronize.
Medians are recorded.

THE BODY_SPLIT LEG is the central measurement.  The model exposes its
projection driver as ``model.projector_functions``, and that object holds the
per-device bound bodies in ``_fwd_body_per_dev`` and ``_back_body_per_dev``.
The single-device driver reads index 0 of the relevant list at every public
call and invokes it once per view batch.  This leg replaces index 0 with a
closure that reads a host clock either side of the original body call, records
the duration and returns the result unchanged, makes ONE public call, and puts
the original back.  The split is then arithmetic: the enqueue time of the call
minus the sum of the in-body times is the host time spent in the driver loop
OUTSIDE the compiled callable.  Candidate 1 and candidate 4 fall on opposite
sides of it.  The same call also carries the allocator counters, which is
candidate 2 read where it would happen.

  Two constraints on the wrapper.  It must NOT carry a ``_view_batch_cost``
  attribute: the driver reads that attribute off the body it is about to call
  with a default of None, and its presence would select a different view batch
  size and so a different number of body calls.  And it is installed
  immediately before the measured call and removed immediately after, in a
  finally, because a reconfiguration rebuilds the projector object and a
  wrapper left behind would be timing something no leg asked for.

THE HOST_OPS LEG runs torch.profiler around ONE warm call and reports the top
thirty events by self CPU time, the total self CPU time, and, by name, the
runtime calls that would settle candidates 1, 2 and 3: cudaLaunchKernel,
cudaMalloc, cudaFree, cudaMemcpyAsync, cudaStreamSynchronize,
cudaDeviceSynchronize, and every compiled-dispatch row the profiler emits.

THE SYNC_DETECTOR LEG sets torch's synchronization debug mode to "warn" and
makes ONE warm call with warnings captured.  A hidden per-batch synchronize
appears here as a warning naming the operation that caused it, which is
candidate 3 answered by name rather than by inference.  The mode is restored
in a finally.  Off CUDA the API has nothing to report and the row says so.

## THE ABLATION ARM

The caching allocator's segment policy is read from the environment when CUDA
initializes and cannot be changed afterwards, so re-timing under a different
policy needs a fresh process.  After the main legs are written, this file
spawns ONE subprocess of ITSELF with expandable segments turned on.  The child
runs the timing leg only, at the same cells and directions, writes its rows to
its own jsonl, and exits; the parent appends those rows to the main file under
the leg name "timing_expandable_segments".  If the unaccounted host time is
allocator segment churn, the two timing legs differ.  A child that fails is
recorded as failed and changes nothing else: the main legs are already on
disk by then.

## MEMORY AT CELL B

The 1024-class cell holds a 3.5 GB voxel array, a 4.1 GB sinogram and a
similar output per call on one device.  That is far inside an 80 GB H100 on
paper, but paper is not a measurement, so every leg is wrapped: an
out-of-memory is caught, recorded as that leg's result, and the run continues.

## INSTRUMENT HEALTH

The body_split leg's own enqueue time is compared against the timing leg's
median enqueue at the same cell and direction.  They should agree: the wrapper
adds two clock reads per view batch and nothing else.  A disagreement beyond a
factor of 1.5 means the wrapper changed the call it was measuring, so both
numbers and the ratio go on the row and the run says so in the report.

## THE LOCAL SMOKE

MG53_SMOKE=1 runs the whole flow at a tiny cell on the CPU.  There are no CUDA
allocator counters and no synchronization debug mode there, so those parts are
recorded as skipped rather than silently absent, and the ablation arm does not
run.  What the smoke exercises is the plumbing: the plan, the model
construction, the phantom, the public-route calls, the body wrapper and its
restore, the profiler parse, the rows and the tables.  It is not a
measurement.

Run:
    <torch python> mg53_host_cost_split.py        on a 1-GPU node
    MG53_DRY=1 <python> mg53_host_cost_split.py   print the plan and stop
    MG53_SMOKE=1 <python> mg53_host_cost_split.py the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized cell or direction is an error, not a
silent skip.
    MG53_RESULTS=<dir>       where the jsonl goes
    MG53_CELLS=a,b           subset of the cells
    MG53_DIRECTIONS=forward  subset of the directions
    MG53_DRY=1               print the plan and exit; imports no torch
    MG53_SMOKE=1             the local CPU smoke
    MG53_WARM=3              warm timed calls per direction and cell
    MG53_CHILD=1             internal: the ablation arm's subprocess
    MG53_CHILD_OUT=<path>    internal: where that subprocess writes its rows
"""

import json
import math
import os
import platform
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


SMOKE = _flag("MG53_SMOKE")
DRY = _flag("MG53_DRY")
#: The ablation arm's subprocess mode.  Set, this process runs the timing leg
#: only, under the allocator setting its parent put in its environment, and
#: writes to MG53_CHILD_OUT.  The sbatch unsets it, so a stray value in the
#: submitting shell cannot turn the real run into a child.
CHILD = _flag("MG53_CHILD")
CHILD_OUT = os.environ.get("MG53_CHILD_OUT", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: (views, detector rows, channels).  The two cells mg51 measured, so the
#: split here lands on numbers that already exist.
CELL_SPECS = (
    dict(cell_id="a", cell=(512, 448, 384),
         note="the 512-class multiaxis cell; mg51 measured about 0.35 ms of "
              "unaccounted host time per view batch here"),
    dict(cell_id="b", cell=(1024, 1008, 992),
         note="the 1024-class multiaxis cell; mg51 measured about 35 ms of "
              "unaccounted host time per view batch here, and it is near one "
              "device's memory edge, so its legs are wrapped against an "
              "out-of-memory"),
)
SMOKE_CELL_SPECS = (
    # The view count is a whole number of view batches at the default batch of
    # 64, so the smoke's driver loop runs more than once and the body wrapper
    # is exercised per batch rather than once per call.  The detector is tiny,
    # so the extra views cost almost nothing.
    dict(cell_id="a", cell=(128, 16, 16),
         note="the smoke's tiny cell: plumbing only, not a measurement"),
)

DIRECTIONS = ("forward", "back")
#: The legs, in the order they run at each cell and direction.  The timing leg
#: runs first because the body_split leg's health check compares against it.
LEGS = ("timing", "body_split", "host_ops", "sync_detector")

#: mg51's seed, reset before every call it times.  This run resets it before
#: every call it times too, so the protocol is the one the recorded multiaxis
#: numbers were measured under.
SEED = 13
#: The elevation half-range mg51 sweeps.  Mirrored here so the dry plan can
#: print the recon shape without importing numpy; the real run reads the
#: model's own recon_shape and records whether the two agreed.
ELEVATION_HALF_RANGE = 0.5
#: The geometry's clamp on the smallest |cos(elevation)| (multiaxis_parallel's
#: auto_set_recon_geometry).  Mirrored for the same reason.
MIN_COS_ELEVATION = 0.1

WARM_REPEATS = _positive_int("MG53_WARM", 3)
#: How many profiler events the host_ops leg puts on its row, ranked by self
#: CPU time.  Thirty is enough to reach past the launch API into whatever else
#: the host was doing.
TOP_EVENTS = 30
#: The runtime calls the host_ops leg reports by name whether or not they made
#: the top list.  Each one is a specific candidate: the launch API is the cost
#: mg51 already accounted for, malloc and free are the blocking allocator
#: calls, and the copy and the two synchronizes are the ways a call can make
#: the host wait for the device.
NAMED_EVENTS = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cudaMalloc",
                "cudaFree", "cudaMemcpyAsync", "cudaStreamSynchronize",
                "cudaDeviceSynchronize")
#: Substrings that identify the profiler's compiled-dispatch rows.  Their
#: spelling has moved between torch releases, so all three are matched and
#: whatever is present is reported.
COMPILED_MARKERS = ("TorchDynamo", "Torch-Compiled", "CompiledFunction")
#: The allocator counters read across the body_split leg's call.
ALLOC_STATS = ("num_device_alloc", "num_device_free", "num_alloc_retries",
               "segment.all.current")

#: How far the body_split leg's own enqueue may sit from the timing leg's
#: median enqueue before the run says the wrapper changed the call.  The
#: wrapper adds two clock reads per view batch and nothing else, so a factor
#: of 1.5 is loose; it is a check on the instrument, not on the finding.
ENQUEUE_AGREEMENT_FACTOR = 1.5

# ── the ablation arm ──────────────────────────────────────────────────────────
#: The allocator setting the child runs under.  It must be in the environment
#: before CUDA initializes, which is why the child is a separate process.
ALLOC_CONF = "expandable_segments:True"
#: The child repeats the timing leg at every cell in the plan, paying its own
#: model builds and its own compiles.  The bound is generous because the main
#: legs are already written when it starts, so a child that runs out of time
#: costs the run nothing it needed.
CHILD_TIMEOUT_S = 2400

# ── GPU health ────────────────────────────────────────────────────────────────
# A thermally throttled or power-capped device produces a valid split and an
# invalid wall, and this run reports walls beside the split.
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
    "MG53_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
NAME_COL = 44

#: Stamped onto every row written by this run, so one row read on its own says
#: which host, which run, which torch and which device it came from.  Filled in
#: once torch is imported; the dry plan imports no torch and writes no rows.
IDENTITY = dict(run_label=RUN_LABEL, timestamp=RUN_STAMP, device=DEVICE,
                smoke=SMOKE, child=CHILD)
# ──────────────────────────────────────────────────────────────────────────────


def cell_specs():
    return SMOKE_CELL_SPECS if SMOKE else CELL_SPECS


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
    keep_cells = _strict_subset("MG53_CELLS", allowed_cells)
    keep_dirs = _strict_subset("MG53_DIRECTIONS", DIRECTIONS)
    plan = []
    for spec in cell_specs():
        if spec["cell_id"] not in keep_cells:
            continue
        plan.append(dict(cell_id=spec["cell_id"], cell=list(spec["cell"]),
                         note=spec["note"], directions=list(keep_dirs),
                         warm=WARM_REPEATS,
                         mirrored_recon_shape=list(
                             mirrored_recon_shape(spec["cell"]))))
    if not plan:
        raise ValueError("MG53_CELLS selects no cell")
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


# ── the model and its inputs, built mg51's way ────────────────────────────────
def build_model(cell):
    """The multiaxis model at ``cell``, constructed exactly as mg51
    constructs it.

    Two angles per view: azimuths evenly spaced over half a turn, elevations
    swept across +/- 0.5 radians.  Those are the geometry's own test defaults
    and the values every recorded multiaxis number was measured with; the
    elevation range also sets the slice count, so changing it would change the
    problem rather than the measurement.

    The device is named EXPLICITLY.  On the one-GPU allocation this job asks
    for, an explicit ['cuda:0'] realizes the single device; naming it means the
    realized device is a fact on the row instead of a policy outcome.

    ``skip_memory_preflight`` is set before the device is configured: the
    preflight prices a whole reconstruction, and this run projects a handful
    of times in each direction and never reconstructs.
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
    """mg51's phantom, as a host float32 array.

    The shepp-logan builder places its ellipsoids as fractions of the volume,
    and on a volume only a few voxels deep every one of them can miss, leaving
    the phantom all zeros.  An all-zero phantom forward projects to an
    all-zero sinogram, so the back projection would then be given nothing to
    do and the measurement would describe an empty problem.  mg51 falls back
    to a seeded uniform volume in that case and records that it did; so does
    this.
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


def _seed_before_call():
    """mg51 resets seed 13 before every call it times; so does this, so the
    protocol is the recorded one."""
    import numpy as np

    np.random.seed(SEED)


class CellContext:
    """One cell's model and its two projection inputs, built once and used by
    every leg at that cell.

    Building the inputs is the expensive part at cell B -- a phantom the size
    of the volume, a fancy-index down to the mask, and one forward projection
    -- so the legs are ordered by cell rather than by leg.  The forward
    direction's WARM-UP call produces the sinogram the back projection is
    given, so nothing is projected twice for bookkeeping.
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
                           cell=list(self.cell))

    def build(self):
        """Model, mask, voxel values.  The sinogram is made later, by the
        forward direction's warm-up call."""
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
        # assumed -- a hand-written kernel body carries its own view-batch cost
        # attribute and would be batched by a different rule, so the split
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
        # own rule.  The body_split leg counts body calls, and this is what
        # turns that count into calls per view batch without inferring it.
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
        a library and the wrong thing to be unaware of here -- an eager body
        has no dynamo cache lookup, no guard evaluation and no inductor
        wrapper, so the first of the four candidates would be absent by
        construction and the split would say so without meaning it.
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
        """One public-route projection call in ``direction``.

        This is the funnel the reconstruction itself calls: every sparse
        projection in the library, from the recon engine and from outside it,
        routes through these two methods.
        """
        model = self.model
        if direction == "forward":
            return model.sparse_forward_project(self.voxel_values,
                                                self.indices)
        return model.sparse_back_project(self.sinogram, self.indices)

    def warm_up(self, direction):
        """The untimed call every leg at this cell and direction runs after.

        It pays the compile, the first-touch allocations and the inductor
        cache warm-up, none of which any leg is measuring.  In the forward
        direction its output is kept: it is the sinogram the back projection
        is given.
        """
        import torch

        _seed_before_call()
        start = time.perf_counter()
        out = self.call(direction)
        _sync(torch, self.model.torch_device)
        record = dict(warm_up_s=time.perf_counter() - start,
                      output_shape=[int(x) for x in out.shape])
        if direction == "forward":
            self.sinogram = out
        return record

    def ready(self, direction):
        """Whether ``direction`` has the input it needs.  The back projection
        needs the sinogram the forward direction's warm-up call produced."""
        if direction == "back":
            return self.sinogram is not None
        return self.voxel_values is not None

    def ensure_sinogram(self):
        """Make the sinogram the back projection needs, without timing it.

        In a full run the forward direction's warm-up call makes it.  This
        exists for a run restricted to the back direction -- a re-run of one
        leg after a job ran short -- which would otherwise have no input at
        all.
        """
        import torch

        if self.sinogram is not None:
            return False
        _seed_before_call()
        self.sinogram = self.call("forward")
        _sync(torch, self.model.torch_device)
        return True

    def release(self):
        import torch

        self.voxel_values = None
        self.sinogram = None
        self.indices = None
        self.model = None
        _free_device_memory(torch)


def _leg_row(leg, context, direction):
    return dict(kind="leg", leg=leg, cell_id=context.cell_id,
                cell=list(context.cell), direction=direction)


# ── the timing leg ────────────────────────────────────────────────────────────
def timing_leg(context, direction):
    """MG53_WARM warm calls on two clocks, which are mg51's timing semantics.

    The plain wall is a host clock around the call plus a device synchronize,
    and it is the number that lays beside mg51's walls.  The enqueue clock is
    the host time to RETURN from the call WITHOUT synchronizing: for an
    asynchronous device workload that is what the host spent issuing the work,
    and it is the quantity the body_split leg divides in two.

    There is no timed cold call.  The warm-up call before this leg already was
    the cold call, and repeating it would only pay a second compile-free first
    call that no question here asks about.  Off CUDA the enqueue clock has no
    meaning -- every CPU call is synchronous -- and is recorded all the same,
    where it equals the wall.
    """
    import torch

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row = _leg_row("timing", context, direction)
    row.update(warm_repeats=WARM_REPEATS, seed=SEED,
               alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    # Read after the warm-up call, which is the call that would have compiled
    # this direction's body and the call whose failure would have rebound it
    # to eager.
    row["compile_state"] = context.compile_state()

    walls, enqueues = [], []
    for _ in range(WARM_REPEATS):
        _seed_before_call()
        start = time.perf_counter()
        warm_out = context.call(direction)
        enqueue = time.perf_counter() - start
        _sync(torch, device)
        walls.append(time.perf_counter() - start)
        enqueues.append(enqueue)
        warm_out = None

    row["warm_all_s"] = walls
    row["warm_s"] = statistics.median(walls)
    row["enqueue_all_s"] = enqueues
    row["enqueue_s"] = statistics.median(enqueues)
    row["spread"] = ((max(walls) - min(walls)) / row["warm_s"]
                     if row["warm_s"] else None)
    if row["warm_s"]:
        row["enqueue_frac_of_wall"] = row["enqueue_s"] / row["warm_s"]
    if cuda:
        row["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return row


# ── the body_split leg ────────────────────────────────────────────────────────
BODY_LIST_ATTR = dict(forward="_fwd_body_per_dev", back="_back_body_per_dev")


def _install_body_timer(projector_functions, direction, durations):
    """Put a host-clock wrapper over the bound body at index 0 and return the
    body it replaced.

    The driver reads its body out of these lists at every public call, so
    replacing index 0 is enough and nothing else has to be touched.  What the
    wrapper measures is the host time for the body call to RETURN, which for
    an asynchronous device workload is that body call's share of the enqueue
    cost -- the same quantity the leg then subtracts from.

    TWO THINGS THE WRAPPER MUST NOT DO.  It must not carry a
    ``_view_batch_cost`` attribute: the driver reads that attribute off the
    body it is about to call, with a default of None, and takes the torch-body
    batching branch only when it is absent.  A wrapper carrying it would
    select a different view batch size, which would change the very count this
    leg reports.  That is why this is a bare closure and not a functools.wraps
    wrapper, which copies the wrapped object's attributes onto the wrapper.
    And it must not swallow or convert the body's return value, because the
    driver assigns and accumulates it.
    """
    bodies = getattr(projector_functions, BODY_LIST_ATTR[direction])
    original = bodies[0]

    def timed_body(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        durations.append(time.perf_counter() - start)
        return result

    bodies[0] = timed_body
    return original


def _restore_body(projector_functions, direction, original):
    getattr(projector_functions, BODY_LIST_ATTR[direction])[0] = original


def _stat_delta(before, after, key):
    """One allocator counter's change across the call, or None if this torch
    does not keep that counter.

    None rather than zero: zero would read as "the call made no such
    allocation", and a counter that does not exist says nothing at all.
    """
    start, end = before.get(key), after.get(key)
    if start is None or end is None:
        return None
    return int(end) - int(start)


def body_split_leg(context, direction, timing_row):
    """The split: how much of one call's host time is spent INSIDE the
    compiled body and how much is spent in the driver loop around it.

    ONE call, not several.  The wrapper is installed immediately before it and
    removed immediately after, in a finally, because the projector object is
    rebuilt on reconfiguration and a wrapper left behind would time calls no
    leg asked for.

    The same call carries the allocator counters, because the allocator
    candidate and the driver-loop candidate have to be read on the SAME call
    to be compared: a call that ran during a different allocator state would
    put the two on different footings.  ``reset_peak_memory_stats`` runs first
    so the peaks on the row belong to this call.

    ``timing_row`` is the timing leg's row at the same cell and direction.
    Its median enqueue is compared against this call's enqueue, which is how
    the run checks that the wrapper did not change the call it measured.
    """
    import torch

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    projector_functions = model.projector_functions
    durations = []
    row = _leg_row("body_split", context, direction)
    row.update(body_list=BODY_LIST_ATTR[direction],
               alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
               allocator_counters=cuda)

    before = {}
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
        before = dict(torch.cuda.memory_stats(device))

    original = _install_body_timer(projector_functions, direction, durations)
    try:
        _seed_before_call()
        start = time.perf_counter()
        out = context.call(direction)
        enqueue = time.perf_counter() - start
        _sync(torch, device)
        wall = time.perf_counter() - start
    finally:
        _restore_body(projector_functions, direction, original)
    out = None
    row["wrapper_restored"] = (
        getattr(projector_functions, BODY_LIST_ATTR[direction])[0] is original)

    if cuda:
        after = dict(torch.cuda.memory_stats(device))
        row["allocator"] = {name: _stat_delta(before, after, name)
                            for name in ALLOC_STATS}
        row["allocator"]["segment_all_current_after"] = after.get(
            "segment.all.current")
        row["allocator"]["peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device))
        row["allocator"]["peak_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(device))

    row["enqueue_s"] = enqueue
    row["wall_s"] = wall
    row["body_calls"] = len(durations)
    row["in_body_sum_s"] = float(sum(durations))
    row["first_body_durations_s"] = [float(x) for x in durations[:5]]
    if durations:
        row["in_body_min_s"] = float(min(durations))
        row["in_body_median_s"] = float(statistics.median(durations))
        row["in_body_max_s"] = float(max(durations))
    else:
        row["in_body_min_s"] = None
        row["in_body_median_s"] = None
        row["in_body_max_s"] = None
    row["driver_host_s"] = enqueue - row["in_body_sum_s"]
    if enqueue:
        row["in_body_frac_of_enqueue"] = row["in_body_sum_s"] / enqueue
        row["driver_frac_of_enqueue"] = row["driver_host_s"] / enqueue
    if durations:
        row["driver_host_per_body_call_s"] = (row["driver_host_s"]
                                              / len(durations))

    # The wrapper recorded nothing means the driver never called the body this
    # leg replaced -- a different device index, a different route, or a
    # rebuilt projector object.  The split on such a row is not a measurement,
    # so it is marked as an error here and the exit code reports it.
    row["recorded_any"] = bool(durations)
    if not durations:
        row["error"] = ("the wrapped body was never called, so this row's "
                        "split describes no call; the driver did not route "
                        "through index 0 of " + BODY_LIST_ATTR[direction])

    # Instrument health: the wrapper adds two clock reads per view batch and
    # nothing else, so this call's enqueue should sit on top of the timing
    # leg's median enqueue.  Both numbers and their ratio go on the row.
    reference = (timing_row or {}).get("enqueue_s")
    row["timing_enqueue_s"] = reference
    if reference and enqueue:
        ratio = enqueue / reference
        row["enqueue_ratio"] = ratio
        row["enqueue_agrees"] = bool(
            1.0 / ENQUEUE_AGREEMENT_FACTOR <= ratio
            <= ENQUEUE_AGREEMENT_FACTOR)
        if not row["enqueue_agrees"]:
            row["warning"] = (
                f"this call's enqueue is {ratio:.2f}x the timing leg's "
                f"median enqueue, past the {ENQUEUE_AGREEMENT_FACTOR}x this "
                "run treats as agreement; the wrapper may have changed the "
                "call it measured")
    else:
        row["enqueue_ratio"] = None
        row["enqueue_agrees"] = None
    return row


# ── the host_ops leg ──────────────────────────────────────────────────────────
def _event_us(event, names):
    """One timing field off a key_averages row, by the first name this torch
    has.

    The device-time fields were renamed from cuda to device between releases
    and both spellings are still in circulation, so the names are tried in
    order and the one that answered is recorded on the row.
    """
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            try:
                return float(value), name
            except (TypeError, ValueError):
                return 0.0, name
    return 0.0, None


DEVICE_US_NAMES = ("device_time_total", "cuda_time_total")
SELF_DEVICE_US_NAMES = ("self_device_time_total", "self_cuda_time_total")


def host_ops_leg(context, direction):
    """torch.profiler around ONE warm call, read for host operators.

    Stacks and shapes are off.  Both are expensive to collect and neither
    answers any of the four candidates; a profiler that costs more than the
    thing it profiles would move the number this run is trying to attribute.

    The row carries three things.  The top thirty events by SELF CPU time is
    where unaccounted host time would show up under whatever name it has.  The
    total self CPU time is what those thirty are a share of.  And the named
    events are the specific candidates, reported whether or not they made the
    top list, so an absent cudaMalloc is on the row as absent rather than as
    silence.
    """
    import torch

    from torch.profiler import ProfilerActivity, profile

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    activities = [ProfilerActivity.CPU]
    if cuda:
        activities.append(ProfilerActivity.CUDA)
    row = _leg_row("host_ops", context, direction)
    row.update(activities=[str(a) for a in activities], cuda=cuda,
               top_events_requested=TOP_EVENTS)

    _seed_before_call()
    with profile(activities=activities, record_shapes=False,
                 with_stack=False) as prof:
        start = time.perf_counter()
        out = context.call(direction)
        _sync(torch, device)
        row["profiled_wall_s"] = time.perf_counter() - start
    out = None

    events = list(prof.key_averages())
    row["profiler_rows"] = len(events)
    entries, total_self_cpu_us = [], 0.0
    device_attr = self_device_attr = None
    for event in events:
        name = str(getattr(event, "key", ""))
        self_cpu_us = float(getattr(event, "self_cpu_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(event, "cpu_time_total", 0.0) or 0.0)
        device_us, device_attr = _event_us(event, DEVICE_US_NAMES)
        _self_device_us, self_device_attr = _event_us(event,
                                                      SELF_DEVICE_US_NAMES)
        total_self_cpu_us += self_cpu_us
        entries.append(dict(name=name,
                            count=int(getattr(event, "count", 0) or 0),
                            self_cpu_us=self_cpu_us, cpu_us=cpu_us,
                            device_us=device_us))
    entries.sort(key=lambda entry: -entry["self_cpu_us"])
    row["total_self_cpu_us"] = total_self_cpu_us
    row["top_events"] = entries[:TOP_EVENTS]
    row["device_time_attr"] = device_attr
    row["self_device_time_attr"] = self_device_attr

    by_name = {entry["name"]: entry for entry in entries}
    named = []
    for name in NAMED_EVENTS:
        entry = by_name.get(name)
        if entry is not None:
            named.append(dict(name=name, count=entry["count"],
                              self_cpu_us=entry["self_cpu_us"]))
    # The compiled-dispatch rows are matched by substring rather than by exact
    # name: their spelling carries a graph identifier that differs per body and
    # per release, so an exact list would go stale without saying so.
    for entry in entries:
        if any(marker in entry["name"] for marker in COMPILED_MARKERS):
            named.append(dict(name=entry["name"], count=entry["count"],
                              self_cpu_us=entry["self_cpu_us"]))
    row["named_events"] = named
    row["named_events_absent"] = [name for name in NAMED_EVENTS
                                  if name not in by_name]
    if cuda:
        row["peak_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return row


# ── the sync_detector leg ─────────────────────────────────────────────────────
def sync_detector_leg(context, direction):
    """torch's own synchronization warnings, around ONE warm call.

    In "warn" mode the runtime raises a python warning at every operation that
    makes the host wait for the device -- a device-to-host copy, an .item(), a
    boolean mask index, and the rest.  A per-batch synchronize hidden inside a
    compiled body therefore arrives here NAMED, which is what separates the
    third candidate from the other three instead of leaving it as an
    inference.

    The mode is restored in a finally: it is process-wide state, and a run
    that left it on would make every later leg print warnings that belong to
    this one.

    The deliberate synchronize after the call is taken OUTSIDE the capture, so
    a warning this run asked for cannot be counted as one the call caused.

    The count is of EVERY warning the call raised, not only synchronization
    ones: the filter is set to "always" so nothing is suppressed, and a
    library warning that happens to fire during the call is counted too.  The
    distinct messages on the row are what say which is which.
    """
    import warnings

    import torch

    row = _leg_row("sync_detector", context, direction)
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    setter = getattr(torch.cuda, "set_sync_debug_mode", None)
    if not cuda or setter is None:
        row.update(available=False,
                   reason=("this run is on the CPU, where there is no device "
                           "to wait for" if not cuda else
                           "torch.cuda.set_sync_debug_mode is missing on this "
                           "torch"))
        return row
    row["available"] = True

    device = context.model.torch_device
    captured = []
    try:
        setter("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _seed_before_call()
            start = time.perf_counter()
            out = context.call(direction)
            row["enqueue_s"] = time.perf_counter() - start
            captured = list(caught)
    finally:
        try:
            setter("default")
        except Exception as exc:                                  # noqa: BLE001
            row["restore_error"] = f"{type(exc).__name__}: {exc}"
    _sync(torch, device)
    row["wall_s"] = time.perf_counter() - start
    out = None
    row["sync_debug_mode_restored"] = (
        int(torch.cuda.get_sync_debug_mode())
        if hasattr(torch.cuda, "get_sync_debug_mode") else None)

    messages = []
    for item in captured:
        text = str(getattr(item, "message", item))[:200]
        if text not in messages:
            messages.append(text)
    row["warning_count"] = len(captured)
    row["distinct_message_count"] = len(messages)
    row["distinct_messages"] = messages[:10]
    return row


# ── the ablation arm ──────────────────────────────────────────────────────────
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


def child_env(child_out):
    """The environment the ablation child gets.

    Every knob this run reads is forwarded, so the child measures the same
    plan the parent measured; the one difference is the allocator setting,
    which is the variable under test.  The dry and smoke flags are forced off
    because a child is only ever spawned by a real CUDA run, and an inherited
    flag would make it print a plan or build a tiny cell instead.
    """
    env = dict(os.environ)
    env["PYTORCH_CUDA_ALLOC_CONF"] = ALLOC_CONF
    env["MG53_CHILD"] = "1"
    env["MG53_CHILD_OUT"] = child_out
    env["MG53_DRY"] = "0"
    env["MG53_SMOKE"] = "0"
    return env


def child_rows_path(out_path):
    stem = out_path[:-len(".jsonl")] if out_path.endswith(".jsonl") \
        else out_path
    return stem + "_child.jsonl"


def ablation_arm(out_path, torch_python):
    """One subprocess of this file under a different allocator setting, then
    its timing rows folded into this run's own file.

    The setting is read by the caching allocator when CUDA initializes and
    cannot be changed afterwards, which is the whole reason this is a
    subprocess and not another leg.  The child runs the timing leg only: the
    question the arm asks is whether the wall and the enqueue move, and the
    split and the profiler are the parent's job.

    A child that fails is recorded as failed and nothing else happens.  The
    main legs are already on disk when this starts, so the arm can cost the
    run nothing it needed.
    """
    record = dict(kind="ablation", attempted=True, alloc_conf=ALLOC_CONF,
                  timeout_s=CHILD_TIMEOUT_S)
    child_out = child_rows_path(out_path)
    record["child_rows"] = child_out
    cmd = [torch_python, "-u", os.path.abspath(__file__)]
    record["cmd"] = cmd
    got = _run(cmd, CHILD_TIMEOUT_S, env=child_env(child_out))
    record.update(returncode=got["returncode"], timed_out=got["timed_out"],
                  wall_s=got["wall_s"])

    rows = []
    if not os.path.exists(child_out):
        record.update(ok=False,
                      reason=("the child wrote no rows file; its output is on "
                              "this row"),
                      stderr=(got["stderr"] or "")[-2000:],
                      stdout=(got["stdout"] or "")[-1000:])
        return record, rows
    with open(child_out) as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("kind") == "leg" and entry.get("leg") == "timing":
                entry["leg"] = "timing_expandable_segments"
                entry["alloc_conf"] = ALLOC_CONF
                entry["from_child"] = True
                rows.append(entry)
    record["rows"] = len(rows)
    record["ok"] = bool(rows) and got["returncode"] == 0
    if not record["ok"]:
        record["reason"] = (
            f'the child exited {got["returncode"]} and left {len(rows)} '
            "timing row(s); whatever it did write is folded in above")
        record["stderr"] = (got["stderr"] or "")[-2000:]
    return record, rows


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


def _of_leg(rows, leg):
    return [row for row in rows if row.get("leg") == leg]


def _failed(row):
    return bool(row.get("error") or row.get("skipped"))


def _fail_line(row):
    return (f'  {row.get("cell_id", "?"):<6}{row.get("direction", "?"):<10}'
            f'  {str(row.get("reason") or row.get("error"))[:70]}')


def timing_table(rows):
    print("\n── TIMING: the public route on plain clocks ────────────────────")
    print(f'  {"cell":<6}{"direction":<10}{"warm s":>10}{"enqueue s":>12}'
          f'{"enq/wall":>10}{"spread":>9}  allocator')
    for row in rows:
        if _failed(row):
            print(_fail_line(row))
            continue
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(row.get("warm_s"))}'
              f'{_fmt(row.get("enqueue_s"), 12, prec=4)}'
              f'{_fmt(row.get("enqueue_frac_of_wall"), 10)}'
              f'{_fmt(row.get("spread"), 9)}  '
              f'{row.get("alloc_conf") or "(default)"}')
    print("  enqueue s is the host time to RETURN from the call without a")
    print("  synchronize.  It is the quantity the split below divides.")


def split_table(rows, setups):
    print("\n── SPLIT: host time inside the compiled body against outside ───")
    batches = {(row.get("cell_id"), name): count
               for row in setups
               for name, count in (row.get("view_batch_iterations")
                                   or {}).items()}
    print(f'  {"cell":<6}{"direction":<10}{"calls":>7}{"enqueue s":>12}'
          f'{"in body s":>12}{"driver s":>12}{"driver/enq":>12}'
          f'{"driver/call ms":>15}')
    for row in rows:
        if _failed(row):
            print(_fail_line(row))
            continue
        per_call = row.get("driver_host_per_body_call_s")
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(row.get("body_calls"), 7, "d", 0)}'
              f'{_fmt(row.get("enqueue_s"), 12, prec=4)}'
              f'{_fmt(row.get("in_body_sum_s"), 12, prec=4)}'
              f'{_fmt(row.get("driver_host_s"), 12, prec=4)}'
              f'{_fmt(row.get("driver_frac_of_enqueue"), 12)}'
              f'{_fmt(None if per_call is None else per_call * 1000.0, 15, prec=3)}')
        expected = batches.get((row.get("cell_id"), row.get("direction")))
        if expected is not None and expected != row.get("body_calls"):
            print(f'         body calls {row.get("body_calls")} against '
                  f'{expected} view batches from the driver\'s own rule; the '
                  "public route may take more than one pass")
    print("  in body s is the host time spent INSIDE the compiled callable,")
    print("  summed over the body calls of one public call: the compiled")
    print("  dispatch plus everything the body itself issues.  driver s is")
    print("  the rest of the enqueue time, which is the view-batch loop.")


def allocator_table(rows):
    printed = False
    for row in rows:
        if _failed(row) or not row.get("allocator"):
            continue
        if not printed:
            print("\n── ALLOCATOR: what one call asked the allocator for ────"
                  "────────")
            print(f'  {"cell":<6}{"direction":<10}{"dev alloc":>11}'
                  f'{"dev free":>10}{"retries":>9}{"seg delta":>11}'
                  f'{"peak GB":>9}{"resvd GB":>10}')
            printed = True
        alloc = row["allocator"]
        peak = alloc.get("peak_allocated_bytes")
        reserved = alloc.get("peak_reserved_bytes")
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(alloc.get("num_device_alloc"), 11, "d", 0)}'
              f'{_fmt(alloc.get("num_device_free"), 10, "d", 0)}'
              f'{_fmt(alloc.get("num_alloc_retries"), 9, "d", 0)}'
              f'{_fmt(alloc.get("segment.all.current"), 11, "d", 0)}'
              f'{_fmt(None if peak is None else peak / 2 ** 30, 9, prec=2)}'
              f'{_fmt(None if reserved is None else reserved / 2 ** 30, 10, prec=2)}')
    if printed:
        print("  dev alloc and dev free count calls into cudaMalloc and")
        print("  cudaFree, both of which block the host.  retries counts")
        print("  allocations that missed the cache and freed blocks to")
        print("  proceed.  All three are deltas across the split leg's call.")


def host_ops_table(rows):
    for row in rows:
        if _failed(row):
            print(_fail_line(row))
            continue
        print(f'\n── HOST OPS: cell {row["cell_id"]} {row["direction"]} '
              f'─────────────────────────────')
        total = row.get("total_self_cpu_us") or 0.0
        print(f'  total self CPU {total / 1000.0:.1f} ms over '
              f'{row.get("profiler_rows")} event(s), wall '
              f'{_fmt(row.get("profiled_wall_s"), 1).strip()} s')
        print(f'  {"event":<{NAME_COL}}{"count":>8}{"self ms":>10}'
              f'{"share":>8}{"dev ms":>10}')
        for entry in (row.get("top_events") or [])[:12]:
            share = (entry["self_cpu_us"] / total) if total else None
            print(f'  {_short(entry["name"]):<{NAME_COL}}'
                  f'{_fmt(entry.get("count"), 8, "d", 0)}'
                  f'{_fmt(entry["self_cpu_us"] / 1000.0, 10, prec=2)}'
                  f'{_fmt(share, 8, prec=3)}'
                  f'{_fmt((entry.get("device_us") or 0.0) / 1000.0, 10, prec=2)}')
        named = row.get("named_events") or []
        if named:
            print("  named:")
            for entry in named:
                print(f'    {_short(entry["name"], NAME_COL - 2):<{NAME_COL - 2}}'
                      f'{_fmt(entry.get("count"), 8, "d", 0)}'
                      f'{_fmt(entry["self_cpu_us"] / 1000.0, 10, prec=2)}')
        if row.get("named_events_absent"):
            print(f'  absent: {", ".join(row["named_events_absent"])}')


def sync_table(rows):
    print("\n── SYNCHRONIZATION: what the call made the host wait for ───────")
    for row in rows:
        if _failed(row):
            print(_fail_line(row))
            continue
        if not row.get("available"):
            print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
                  f'  {str(row.get("reason"))[:70]}')
            continue
        print(f'  {row["cell_id"]:<6}{row["direction"]:<10}'
              f'{_fmt(row.get("warning_count"), 8, "d", 0)} warning(s), '
              f'{_fmt(row.get("distinct_message_count"), 4, "d", 0)} distinct')
        for text in (row.get("distinct_messages") or [])[:4]:
            print(f"      {text[:100]}")
    print("  A warning here names the operation that made the host wait.  No")
    print("  warnings means the call issued no host-visible synchronization.")


def ablation_table(rows, child_rows):
    if not child_rows:
        return
    print("\n── ALLOCATOR ABLATION: the same timing leg under expandable "
          "segments ──")
    baseline = {(row.get("cell_id"), row.get("direction")): row
                for row in rows}
    print(f'  {"cell":<6}{"direction":<10}{"warm s":>10}{"was":>10}'
          f'{"ratio":>9}{"enqueue s":>12}{"was":>12}{"ratio":>9}')
    for row in child_rows:
        key = (row.get("cell_id"), row.get("direction"))
        was = baseline.get(key) or {}
        warm, warm_was = row.get("warm_s"), was.get("warm_s")
        enq, enq_was = row.get("enqueue_s"), was.get("enqueue_s")
        print(f'  {str(key[0]):<6}{str(key[1]):<10}'
              f'{_fmt(warm)}{_fmt(warm_was)}'
              f'{_fmt(None if not (warm and warm_was) else warm / warm_was, 9)}'
              f'{_fmt(enq, 12, prec=4)}{_fmt(enq_was, 12, prec=4)}'
              f'{_fmt(None if not (enq and enq_was) else enq / enq_was, 9)}')
    print("  A ratio near 1 means the allocator's segment policy is not what")
    print("  the host time is spent on.")


def reading_guide():
    print("\n── HOW TO READ THIS ────────────────────────────────────────────")
    print("  Work inside the compiled callable: in body s is most of the")
    print("  enqueue, and the host ops table shows compiled-dispatch rows")
    print("  with large self CPU time.  The remedy would be fewer, larger")
    print("  body calls.")
    print("  Allocator calls that block: the allocator table shows device")
    print("  allocs, frees or retries scaling with the body call count, the")
    print("  host ops table shows cudaMalloc or cudaFree self CPU time, and")
    print("  the ablation arm moves the wall.")
    print("  A hidden synchronization: the synchronization table names it,")
    print("  and the host ops table shows cudaMemcpyAsync or a stream")
    print("  synchronize once per body call.")
    print("  Host work in the driver loop: driver s is most of the enqueue")
    print("  and driver/call is the per-batch cost, with nothing in the host")
    print("  ops table large enough to account for it.")
    print("  This run decides nothing; the verdict is read by a person from")
    print("  the tables above and the rows in the jsonl.")


def summarize(header, setups, rows, ablation, child_rows, out_path):
    timings = _of_leg(rows, "timing")
    splits = _of_leg(rows, "body_split")
    print()
    timing_table(timings)
    split_table(splits, setups)
    allocator_table(splits)
    host_ops_table(_of_leg(rows, "host_ops"))
    sync_table(_of_leg(rows, "sync_detector"))
    ablation_table(timings, child_rows)
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
                "This run splits the host time of a compiled body, so that "
                "is a different subject, not a different number.")
    for row in rows:
        if row.get("error") and not row.get("oom"):
            checks.append(f'{row.get("leg")} {row.get("cell_id")}/'
                          f'{row.get("direction")}: '
                          f'{str(row.get("error"))[:200]}')
        # A body that fell back to eager is not the body this run is named
        # for.  The library is right to fall back -- an eager retry is how it
        # survives a broken backend -- but an eager body has no compiled
        # dispatch to charge, so the split would answer a different question
        # under the same headings.
        state = row.get("compile_state") or {}
        if state.get("multiaxis_compile_errors"):
            checks.append(
                f'{row.get("cell_id")}/{row.get("direction")}: a multiaxis '
                f'body did NOT compile and was rebound to eager: '
                f'{state["multiaxis_compile_errors"]}')
        if row.get("wrapper_restored") is False:
            checks.append(f'{row.get("cell_id")}/{row.get("direction")}: the '
                          "body wrapper was not restored after its leg")
    if not header.get("tree_witnesses", {}).get("ok"):
        checks.append(f'tree witnesses: {header.get("tree_witnesses")}')

    oom = [f'{row.get("leg") or row.get("kind")} {row.get("cell_id")}/'
           f'{row.get("direction")}'
           for row in setups + rows if row.get("oom")]
    if oom:
        print(f"\n{len(oom)} leg(s) ran out of device memory and are recorded "
              f"as that: {oom}.  An out-of-memory is this run's answer for "
              "that leg, not a failure of the run.")

    disagreed = [f'{row.get("cell_id")}/{row.get("direction")} '
                 f'{row.get("enqueue_ratio"):.2f}x'
                 for row in splits if row.get("enqueue_agrees") is False]
    if disagreed:
        print(f"\nNOTE: the split leg's enqueue disagreed with the timing "
              f"leg's median enqueue at {disagreed}.  Both numbers are on the "
              "rows.  This is a warning about the instrument and does not "
              "change the exit code.")
    if ablation and ablation.get("attempted") and not ablation.get("ok"):
        print(f'\nNOTE: the allocator ablation child did not finish: '
              f'{str(ablation.get("reason"))[:200]}.  The main legs were '
              "already written; the arm never changes the exit code.")
    if header.get("gpu_hot_or_throttled"):
        print("\nNOTE: the device sampled hot or throttled.  The walls are a "
              "rate reading, so read them with that in mind.")

    healthy = not checks
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"}.  It covers six things: '
          "every planned leg produced a row or a recorded out-of-memory, the "
          "realized device is the one asked for, both directions bound a "
          "torch body, both bodies really compiled rather than falling back "
          "to eager, the wrapped body was called and put back, and the tree "
          "witnesses hold.")
    for line in checks:
        print(f"  FAIL: {line}")
    print("The ablation arm's absence never changes it, and neither does "
          "what any leg found.")
    return dict(kind="summary", healthy=healthy, checks=checks, oom=oom,
                enqueue_disagreements=disagreed,
                ablation_ok=(ablation or {}).get("ok"),
                rows=len(rows), child_rows=len(child_rows),
                out_path=out_path)


# ── the runner ────────────────────────────────────────────────────────────────
def _dry_run(plan):
    print(f"mg53 host cost split: {len(plan)} cell(s), device {DEVICE}")
    print("  where the host time of one compiled multiaxis projection call "
          "goes.  A separate measurement found about 35 ms per view batch of "
          "host time at the 1024-class cell that is neither the kernel-launch "
          "API nor the final synchronize, against about 0.35 ms per batch at "
          "the 512-class cell.  This run attributes it.  It decides nothing.")
    print(f"  results -> {RESULTS_DIR}")
    print(f'\n  {"cell":<6}{"sinogram":>20}{"recon (mirrored)":>24}'
          f'{"warm":>6}  directions')
    for entry in plan:
        print(f'  {entry["cell_id"]:<6}'
              f'{str(tuple(entry["cell"])):>20}'
              f'{str(tuple(entry["mirrored_recon_shape"])):>24}'
              f'{entry["warm"]:>6}  '
              f'{",".join(entry["directions"])}')
    print("  the recon shapes above come from this file's mirror of the "
          "geometry's own rule; the real run reads the model's recon_shape "
          "and records whether the two agreed.")
    for entry in plan:
        print(f'    {entry["cell_id"]}: {entry["note"]}')

    print(f"\n  protocol: mg51's -- azimuths over half a turn, elevations "
          f"across +/- {ELEVATION_HALF_RANGE}, the shepp-logan "
          f"low-dynamic-range phantom, seed {SEED} reset before every call, "
          "one device named explicitly")
    print("  route: TomographyModel.sparse_forward_project and "
          "sparse_back_project, the funnel the reconstruction itself calls")
    print("  every leg runs after ONE untimed warm-up call in its direction, "
          "which pays the compile")

    print(f'\n  legs, in order at each cell and direction: '
          f'{", ".join(LEGS)}')
    print(f"    timing: {WARM_REPEATS} warm call(s) on a host clock with a "
          "device synchronize, plus the enqueue clock; medians recorded.  "
          "These are mg51's timing semantics, so the rows lay beside its "
          "numbers.")
    print("    body_split: index 0 of the projector's bound-body list is "
          "wrapped in a host clock, ONE call is made, and the original is put "
          "back in a finally.  The enqueue time minus the summed in-body time "
          "is the host time spent in the view-batch driver loop.  The same "
          "call carries the allocator counters"
          + (", SKIPPED in the smoke" if SMOKE else "")
          + f": {', '.join(ALLOC_STATS)}, plus the call's peak allocated and "
            "reserved bytes.")
    print(f"    host_ops: torch.profiler around ONE warm call, no stacks and "
          f"no shapes; the top {TOP_EVENTS} events by self CPU time, the "
          f"total self CPU time, and these by name whether or not they made "
          f"the list: {', '.join(NAMED_EVENTS)}, plus every row whose name "
          f"contains {' or '.join(COMPILED_MARKERS)}")
    print("    sync_detector: the synchronization debug mode set to warn "
          "around ONE warm call, with warnings captured and the mode restored "
          "in a finally"
          + ("; SKIPPED in the smoke, where there is no device to wait for"
             if SMOKE else ""))

    if SMOKE:
        print("\n  ablation arm: SKIPPED in the smoke and recorded as skipped")
    else:
        print(f"\n  ablation arm: one subprocess of this file with "
              f"PYTORCH_CUDA_ALLOC_CONF={ALLOC_CONF}, running the timing leg "
              f"only at the same cells and directions, bounded at "
              f"{CHILD_TIMEOUT_S} s.  Its rows are folded into this run's "
              "file as leg timing_expandable_segments.  The setting is read "
              "when CUDA initializes, which is why it needs its own process.")
        print("  a child that fails is recorded as failed and changes nothing "
              "else: the main legs are already written when it starts")

    print("\n  every leg is wrapped: an out-of-memory is caught and recorded "
          "as that leg's result, and the run continues.  That is what the "
          "1024-class cell needs; it is near one device's edge.")
    print(f"  instrument health: the split leg's own enqueue must sit within "
          f"{ENQUEUE_AGREEMENT_FACTOR}x of the timing leg's median enqueue; "
          "a disagreement is recorded and reported as a warning")
    print("  exit code = instrument health: every planned leg produced a row "
          "or a recorded out-of-memory, the realized device is the one asked "
          "for, both directions bound a torch body, both bodies really "
          "compiled rather than falling back to eager, the wrapped body was "
          "called and put back, and the tree witnesses hold")
    print("  no library file is touched: every call goes through the public "
          "projection funnel, and the body wrapper is removed in a finally")


def _leg_error(leg, context, direction, exc):
    """One failed leg, recorded as that leg's result.

    An out-of-memory is marked as such and is not a health failure: the 1024
    cell is near one device's edge, and 'it did not fit' is an answer to the
    question this run asks.
    """
    return dict(kind="leg", leg=leg, cell_id=context.cell_id,
                cell=list(context.cell), direction=direction,
                error=str(exc)[:1200], oom=_is_oom(exc),
                traceback=traceback.format_exc()[-2000:])


def write_row(sink, row):
    """One jsonl row, stamped with the run's identity and flushed.

    Flushed per row because a job that is killed mid-run should leave every
    row it had already finished, and stamped because a row read on its own has
    to say which host, run, torch and device it came from.
    """
    merged = dict(IDENTITY)
    merged.update(row)
    sink.write(json.dumps(merged) + "\n")
    sink.flush()
    return merged


def run_plan(plan, sink, timing_only):
    """Every cell in the plan: build it, then run the legs at each direction.

    ``timing_only`` is the ablation child's mode.  The child exists to re-time
    the public route under a different allocator setting, and the split, the
    profiler and the synchronization warnings are the parent's job, so the
    child runs the timing leg alone.
    """
    import torch

    setups, rows = [], []
    for spec in plan:
        print(f'\n  cell {spec["cell_id"]} {tuple(spec["cell"])}', flush=True)
        context = CellContext(spec)
        try:
            setup = context.build()
        except Exception as exc:                                  # noqa: BLE001
            setup = dict(kind="cell_setup", cell_id=spec["cell_id"],
                         cell=list(spec["cell"]), error=str(exc)[:1200],
                         oom=_is_oom(exc),
                         traceback=traceback.format_exc()[-2000:])
            setups.append(write_row(sink, setup))
            print(f'    setup failed: {str(exc)[:200]}', flush=True)
            context.release()
            continue
        setups.append(write_row(sink, setup))
        print(f'    recon {tuple(setup["recon_shape"])}, '
              f'{setup["num_pixels"]} pixels, view batch '
              f'{setup["view_batch"]}, '
              f'{setup["view_batch_iterations"]} view batches', flush=True)

        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(context.model.torch_device)
        # The forward direction runs first whatever the selection order: its
        # warm-up call makes the sinogram the back direction needs.
        ordered = [name for name in DIRECTIONS if name in spec["directions"]]
        if "back" in ordered and "forward" not in ordered:
            # A back-only selection (a re-run of one leg) has no forward
            # warm-up call to make its input, so one untimed forward call is
            # made here and recorded as having been made.
            try:
                setup["untimed_forward_for_sinogram"] = (
                    context.ensure_sinogram())
            except Exception as exc:                              # noqa: BLE001
                setup["untimed_forward_for_sinogram"] = False
                setup["sinogram_error"] = str(exc)[:600]
                setup["oom"] = _is_oom(exc)
                _free_device_memory(torch)
            write_row(sink, dict(setup, kind="cell_setup_update"))

        for direction in ordered:
            if not context.ready(direction):
                # One skipped row per planned leg, so every table shows the
                # gap rather than only the tables whose leg happens to be
                # named on a single row.
                for leg in (("timing",) if timing_only else LEGS):
                    rows.append(write_row(sink, dict(
                        _leg_row(leg, context, direction), skipped=True,
                        reason="the back projection has no sinogram: the "
                               "forward call that makes it did not run or did "
                               "not finish; see the cell setup row")))
                continue
            print(f"    warm-up {direction}", flush=True)
            try:
                warm_up = context.warm_up(direction)
            except Exception as exc:                              # noqa: BLE001
                rows.append(write_row(
                    sink, _leg_error("warm_up", context, direction, exc)))
                _free_device_memory(torch)
                print(f'      warm-up failed: {str(exc)[:200]}', flush=True)
                continue
            write_row(sink, dict(_leg_row("warm_up", context, direction),
                                 **warm_up))

            timing_row = None
            for leg in (("timing",) if timing_only else LEGS):
                print(f"    {leg} {direction}", flush=True)
                try:
                    if leg == "timing":
                        row = timing_leg(context, direction)
                        timing_row = row
                    elif leg == "body_split":
                        row = body_split_leg(context, direction, timing_row)
                    elif leg == "host_ops":
                        row = host_ops_leg(context, direction)
                    else:
                        row = sync_detector_leg(context, direction)
                except Exception as exc:                          # noqa: BLE001
                    row = _leg_error(leg, context, direction, exc)
                    _free_device_memory(torch)
                    print(f'      {leg} failed: {str(exc)[:200]}', flush=True)
                rows.append(write_row(sink, row))
        context.release()
    return setups, rows


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    import torch

    if not SMOKE and not torch.cuda.is_available():
        print("this run needs CUDA; use MG53_SMOKE=1 for the CPU plumbing "
              "pass")
        return 2
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    IDENTITY.update(torch=torch.__version__,
                    device_name=(torch.cuda.get_device_name(0) if cuda
                                 else DEVICE))

    if CHILD:
        # The ablation child writes where its parent told it to and prints no
        # tables: its rows are read back by the parent and folded into the
        # parent's file.
        out_path = CHILD_OUT or os.path.join(
            RESULTS_DIR, f"mg53_host_cost_split_{RUN_LABEL}_{RUN_STAMP}"
                         "_child.jsonl")
    else:
        out_path = os.path.join(
            RESULTS_DIR,
            f"mg53_host_cost_split_{RUN_LABEL}_{RUN_STAMP}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"mg53 host cost split on {RUN_LABEL} ({DEVICE}); "
          f'{len(plan)} cell(s){" [ablation child]" if CHILD else ""} -> '
          f"{out_path}", flush=True)

    header = dict(kind="run", plan=plan, seed=SEED, legs=list(LEGS),
                  warm_repeats=WARM_REPEATS, top_events=TOP_EVENTS,
                  python=platform.python_version(), node=platform.node(),
                  cuda=cuda,
                  device_count=(torch.cuda.device_count() if cuda else 0),
                  tree_witnesses=tree_witnesses(),
                  alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  inductor_cache=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
                  health_before=sample_gpu_health())
    try:
        import mbirtorch
        header["mbirtorch_file"] = mbirtorch.__file__
    except Exception as exc:                                      # noqa: BLE001
        header["mbirtorch_file"] = f"{type(exc).__name__}: {exc}"

    with open(out_path, "w") as sink:
        write_row(sink, header)
        setups, rows = run_plan(plan, sink, timing_only=CHILD)
        if CHILD:
            print(f"\nwrote {out_path}")
            return 0

        ablation, child_rows = None, []
        if SMOKE:
            ablation = dict(kind="ablation", attempted=False,
                            reason="the smoke runs on the CPU, where the CUDA "
                                   "allocator setting has nothing to change")
            write_row(sink, ablation)
        else:
            print("\n  allocator ablation child", flush=True)
            ablation, child_rows = ablation_arm(out_path, sys.executable)
            write_row(sink, ablation)
            for row in child_rows:
                write_row(sink, row)

        health_after = sample_gpu_health()
        header["gpu_hot_or_throttled"] = bool(
            health_is_hot(header.get("health_before") or [])
            or health_is_hot(health_after))
        write_row(sink, dict(kind="run_close", health_after=health_after,
                             gpu_hot_or_throttled=header[
                                 "gpu_hot_or_throttled"]))
        summary = summarize(header, setups, rows, ablation, child_rows,
                            out_path)
        write_row(sink, summary)
    print(f"\nwrote {out_path}")
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    sys.exit(main())
