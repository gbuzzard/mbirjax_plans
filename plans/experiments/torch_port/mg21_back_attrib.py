"""mg21 -- WHERE THE CONE BACK PROJECTION SPENDS ITS TIME AT THE 2048 CELL,
and why it spends more of it on four devices than on three.

WHY THIS RUN EXISTS.

mg19 measured the first 2048-class reconstructions on this hardware (2026-08-17).
Its per-call brackets say the cone back projection's busy time ROSE with the
device count: 137 s on three devices and 228 s on four, for the same three
iterations of the same problem.  Adding a device made the back projection cost
more, which is the opposite of what sharding is for.

The recorded hypothesis is that per-band-call costs which do not shrink with the
band are what grow.  Each device count runs the same total voxel work, but a
larger device count cuts that work into more, smaller pieces, so any cost that
is charged once per piece is paid more often.  That is a hypothesis and nothing
in mg19's rows tests it.

This run tests it by measuring inside the calls.  It instruments the sharded
cone back projection at six places, splits each call's time into named parts,
and reports how each part's total changes from three devices to four.  A remedy
design can then rest on measured shares rather than on the hypothesis.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It changes no library file
and flips no default.  Every instrument is a monkey patch applied inside the
arm's own subprocess, before the model is built, and every patch calls the
original function with its arguments unchanged.

TERMS USED BELOW, defined once here:
    arm            one measured configuration -- one device count -- run in its
                   own fresh process.
    cell           the sinogram shape, (views, detector rows, detector
                   channels).
    settle         the call that makes the model choose and hold its device
                   layout, ``model._apply_device_policy()``.
    funnel         the library's single public entry point for one projection
                   direction.  Every caller of the sparse back projection goes
                   through ``TomographyModel.sparse_back_project``.
    pass           one call of the sharded back projection funnel.  A pass
                   covers every band and every device.
    band           one contiguous block of reconstruction slices, owned by one
                   device.  By default a band is a device's whole slice shard.
    view-range     one device's back projection of its own views onto one
      call         band.  Every device makes one of these per band, and their
                   results are summed onto the band's owner.
    body call      one call of the projection body inside a view-range call.
                   The view-range loop walks its views in batches and calls the
                   body once per batch.
    reduce         the step that sums the per-device partials of one band onto
                   that band's owner (``_sharding.sum_band_to_owner``).
    busy time      device time inside a bracket, measured with CUDA event
                   pairs.

THE CELL, mg19's cone cell so the two runs describe the same problem:

    sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
    recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask

The model is built exactly as mg19 builds it: a full turn of views, the
source-to-detector distance at four detector widths, the source-to-isocenter
distance at two.

THE SIX INSTRUMENTS, and what each one answers.

  1  ``TomographyModel._sparse_back_project_sharded``  -- the pass.  Gives the
     host wall of a whole call and the list of bands it will walk.
  2  ``Projectors.sparse_back_project_view_range``     -- one call per
     (device, band).  Gives the device time one device spent on one band.
  3  ``triton_cone._cone_horizontal_data``             -- the first eager
     builder inside the body.
  4  ``triton_cone._cone_vertical_affine``             -- the second builder.
  5  ``triton_cone._cone_back_view_batch_triton``      -- the body itself.
     Gives the total each body call took and the view batch it realized.
  6  ``triton_cone._cone_back_kernel``                 -- the Triton launch,
     replaced by a proxy that brackets the launcher and records the grid.

Plus ``_sharding.sum_band_to_owner``, which is the cross-device sum at the end
of each band.

FROM THOSE, PER BODY CALL, three named parts and a remainder:

    builders  = instrument 3 + instrument 4
    kernel    = instrument 6
    residual  = instrument 5 minus builders minus kernel

The residual is dominated by one copy.  Before the launch the body writes
``sino_batch.permute(0, 2, 1).contiguous()``, a channel-major copy of the whole
detector-row block for that view batch, and that copy does not shrink when the
BAND shrinks -- it is sized by the detector, not by the slices being written.
The rest of the residual is the output allocation and small glue.  One
microbench in the four-device arm prices that copy on its own, so the residual
can be read against a direct measurement of its main constituent.

HOW THE DEVICE TIMES ARE TAKEN, and one thing they include.  Every bracket is a
CUDA event pair created and recorded inside ``with torch.cuda.device(dev)``, so
the markers are recorded on the stream that carries that device's work.  NOTHING IS
SYNCHRONIZED INSIDE A PASS: an event record places a marker and no work, so the
devices still overlap exactly as they do without the instrument, and a
synchronize per call would serialize the overlap being measured.  Every elapsed
time is read after the pass ends and every placement device has been
synchronized.

An elapsed time between two events on one stream is wall-clock time on that
device, so it includes any period in which the host failed to keep the stream
fed.  That is deliberate.  Host-side serialization is part of what makes a
per-call cost a per-call cost, and it is exactly what would grow with the number
of calls.  Host ``perf_counter`` walls are recorded at every seam as well, so
host time can be read separately from device time.

THE VARIANTS, per arm, in this order.  Each runs one labeled warmup repeat and
then two timed repeats.  The warmup is recorded and excluded from the analysis,
because the first call at a new shape pays a Triton compilation the later ones
do not.

    full_p1     the full pixel set, coeff_power 1.  The reference call.
    full_p2     the full pixel set, coeff_power 2.  This is the Hessian call a
                reconstruction makes once.
    sub4        a subset of size P/4.
    sub16       a subset of size P/16.
    sub64       a subset of size P/64.
    band_half   the full pixel set, coeff_power 1, with the slice band set to
                half the largest slice shard.

WHY band_half IS THE DISCRIMINATOR.  Halving the band doubles the number of
band calls and halves the voxel work in each one.  A cost that scales with the
band is then unchanged in total.  A cost charged once per band call is doubled.
So if per-band-call costs drive the growth, band_half costs roughly twice as
much as full_p1; if the kernel's own work dominates, band_half barely moves.
That is one variable changed and everything else held, and it is the reading
this run turns on.

WHY THE SUBSET SIZES ARE 4, 16 AND 64.  The shipped three-iteration schedule
uses partition sequence entries 2, 4 and 6 against the granularity list
[1, 2, 4, 8, 16, 32, 64, ...], so its three iterations run at 4, 16 and 64
subsets.  An iteration at granularity g makes g back projection calls, each on
about P/g pixels.  Timing one call at each of those three sizes therefore lets
this run add up an estimate of a whole reconstruction's back projection time and
compare it with mg19's measured 137 s and 228 s.

ONE THING THOSE SUBSETS ARE NOT.  They are drawn from a single seeded
permutation of the pixel indices, first block first, so they have the right SIZE
and no spatial structure.  A real VCD subset is spatially structured, and
structure changes how a gather hits memory.  The subset timings are therefore a
size sweep, and the reconstruction estimate built from them is an estimate.  The
summary says so where it prints the number.

THE INPUT IS SHARDED ONCE PER ARM.  The sinogram is placed on the devices
through the model's own ``_shard_sinogram``, which is the same call the public
funnel makes, and the resulting shard container is handed to every timed call.
The funnel recognizes an already-placed input and returns it unchanged, so no
host-to-device movement happens inside any bracket.

THE STAGED INPUT.  mg19 left its cone phantom and sinogram in the results
directory, and this run reads that sinogram rather than making a new one:

    mg19_2k_cone_sinogram.npy    read, checksum verified on every arm
    mg19_2k_cone_phantom.npy     read ONLY if the sinogram is missing

``MG21_MG19_DIR`` says where to look and defaults to ``MG21_RESULTS``.  If the
sinogram is not there, the arm stages a phantom with mg19's seed and geometry
and forward-projects it at the arm's own device count.  That is recorded on the
row as a fresh sinogram and it does not fail the run: the back projection's time
does not depend on which sinogram it is given, only on its shape.

THE VALUE WITNESS, taken after every timed call.  Each output shard is reduced
in float64 on its own device to a sum of absolute values and a sum of squares,
and the shards' two scalars are combined into two numbers on the row.  Comparing
three devices against four is a cross-count check on the values, and it is
REPORT ONLY: it has no gate here, and the expected agreement is the 1e-6
relative class.

WHAT IS RECORDED ON EVERY ROW, beyond the timings:
  * the realized device list and whether its length is the arm's pin;
  * which projection directions run as general torch code, from
    ``_memory_ledger.torch_body_directions``.  On an H100 cone binds
    hand-written kernels in both directions, so this must be EMPTY;
  * the name of the back body the model bound, and whether it is this run's
    wrapper carrying both required attributes;
  * the per-device block lengths on the view and slice axes;
  * the forward pixel batch the model reports.  The shipped default moved to
    32768; this is recorded and not gated, because this run measures the back
    projection and never calls the forward one in a bracket;
  * the widening-floor staleness note, as informational text;
  * a GPU health sample, so a thermally throttled node is visible;
  * the environment the arm ran under.

TWO ATTRIBUTES THE BODY WRAPPER MUST CARRY, and why the run stops without them.
The bound body is read for two things.  ``_view_batch_cost`` tells the driver
how many views one body call takes; without it the driver would fall back to the
general torch-body cost model, choose a different view batch, and this run would
measure a workload production never runs.  ``_mbirtorch_no_compile`` tells
``maybe_compile`` to leave the body eager; without it torch.compile would trace
the wrapper and the launch inside it.  The wrapper copies both from the original
and asserts that it has them.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when both arms ran and
were recorded, the staged sinogram's checksum verified, every arm ran on its own
device count and bound the kernel body through the wrapper, every variant
produced the number of records its band structure predicts, and no arm ran out
of event pairs.  What the parts add up to never touches it.

THE LOCAL SMOKE.  MG21_SMOKE=1 runs a (32, 24, 20) cone cell at one and two
virtual CPU devices, tiny subsets, one repeat per variant, and no microbench.
On a CPU the Triton module never binds, so the torch body runs and instruments 3
through 6 record nothing; the pass, view-range and reduce instruments still work
and the row plumbing is exercised end to end.  One device takes the funnel's
single-device path, which runs no pass and no reduce at all, and the smoke
checks that case separately rather than pretending it has bands.  Two virtual
CPU devices are both named ``cpu``, so the smoke's per-device table adds them
together into one line; the record counts, which the smoke's checks use, come
from the driver's device index and are not affected.

Run:
    <torch python> mg21_back_attrib.py         on a 4-GPU node
    MG21_DRY=1 <python> mg21_back_attrib.py    print the plan and stop
    MG21_SMOKE=1 <python> mg21_back_attrib.py  the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized token
is an error, not a silent skip.
    MG21_RESULTS=<dir>          where the jsonl goes
    MG21_MG19_DIR=<dir>         where mg19's staged cone files are; defaults to
                                MG21_RESULTS
    MG21_ARMS=n3,n4             subset of the arms, by arm name
    MG21_VARIANTS=full_p1,...   subset of the variants, by variant name
    MG21_DRY=1                  print the plan and exit
    MG21_SMOKE=1                the local CPU smoke
"""

import functools
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG21_SMOKE", "0") == "1"
DRY = os.environ.get("MG21_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

GEOMETRY = "cone"

# mg19's cone cell, so the two runs describe the same problem.  The recon shape
# and pixel count are what the geometry produces from this cell today; they are
# registered so a moved default is visible on the row rather than silent.
CELL = (2048, 2016, 1984)
RECON_SHAPE = (1984, 1984, 2016)
NUM_PIXELS = 3088364
SMOKE_CELL = (32, 24, 20)
SMOKE_RECON_SHAPE = (20, 20, 24)
SMOKE_NUM_PIXELS = 276
CELL_LABEL = "smoke" if SMOKE else "2k"

# The device counts.  Three is where mg19 read 137 s of back busy time and four
# is where it read 228 s, so those are the two counts this run compares.
ARM_DEVICE_COUNTS = (3, 4)
# The smoke has no CUDA devices, so its arms run at one and two virtual CPU
# devices.  The arm names keep their production meaning and every row records
# both numbers.
SMOKE_PINS = {3: 1, 4: 2}

# The subset granularities, and where they come from.  The shipped defaults are
# partition_sequence [2, 4, 6, ...] against granularity
# [1, 2, 4, 8, 16, 32, 64, 128, ...], so a three-iteration reconstruction visits
# granularities 4, 16 and 64.  An iteration at granularity g makes g back
# projection calls of about P/g pixels each.
SUBSET_GRANULARITIES = (4, 16, 64)
SUBSET_SEED = 12345          # the seed every other run in this series uses

PHANTOM_SEED = 20260817      # mg19's phantom seed, used only by the fallback

# One warmup repeat, then two timed repeats.  The warmup is recorded and
# excluded from the analysis: the first call at a new shape pays a Triton
# compilation the later ones do not.
WARMUP_REPEATS = 1
TIMED_REPEATS = 1 if SMOKE else 2

# mg19's measured cone back busy seconds per composed reconstruction, quoted
# here as cross-run CONTEXT for the reconstruction estimate this run prints.
# They are not a gate and nothing compares against them automatically.
MG19_BACK_BUSY_S = {3: 137.0, 4: 228.0}
MG19_BACK_BUSY_SOURCE = ("mg19, 2026-08-17, cone composed arms, busiest "
                         "device, three iterations plus the Hessian")

# The microbench: the channel-major copy the residual is mostly made of.
MICROBENCH_REPEATS = 5
MICROBENCH_ARM = 4           # run it in the four-device arm only

# The event-pair budget for ONE timed repeat.  Events are read and released
# after every repeat, so this bounds how many are live at once rather than how
# many the arm makes in total.  The largest repeat here is band_half at four
# devices, which makes about 5,200 pairs.
MAX_EVENT_PAIRS = 40000

# Bytes of float64 promotion the on-device value witness holds at once.
WITNESS_CHUNK_BYTES = 256 << 20
# Rows of the fallback phantom drawn per call.  Eight rows of the production
# volume is a 256 MiB float64 draw; the whole volume in one call would build a
# 59 GiB float64 array before casting it.
PHANTOM_SLAB_ROWS = 8

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
    "MG21_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
MG19_DIR = os.environ.get("MG21_MG19_DIR", RESULTS_DIR)
RUN_LABEL = platform.node().split(".")[0]
VAR_COL = 12
# ──────────────────────────────────────────────────────────────────────────────


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


def cell():
    return SMOKE_CELL if SMOKE else CELL


def recon_shape():
    return SMOKE_RECON_SHAPE if SMOKE else RECON_SHAPE


def num_pixels():
    return SMOKE_NUM_PIXELS if SMOKE else NUM_PIXELS


def pin_for(n_dev):
    """The device count an arm actually runs at.

    The production pin is the arm's own count.  The smoke has no CUDA devices,
    so its arms run at one or two virtual CPU devices.
    """
    return SMOKE_PINS[n_dev] if SMOKE else n_dev


def variant_names():
    return ["full_p1", "full_p2"] + [f"sub{g}" for g in SUBSET_GRANULARITIES] \
        + ["band_half"]


def arm_specs():
    """Every arm, in the order the run takes them."""
    return [dict(arm=f"n{n}", n_dev=n) for n in ARM_DEVICE_COUNTS]


# ── artifact paths and checksums ──────────────────────────────────────────────
# mg19's file names, reproduced here so this run reads the files that run wrote.
# The cell label is in every name, so a smoke run and a production run can share
# a results directory without either reading the other's bytes.
def _mg19_phantom_path():
    return os.path.join(MG19_DIR, f"mg19_{CELL_LABEL}_{GEOMETRY}_phantom.npy")


def _mg19_sinogram_path():
    return os.path.join(MG19_DIR, f"mg19_{CELL_LABEL}_{GEOMETRY}_sinogram.npy")


def _own_phantom_path():
    """Where the fallback stages its own phantom.

    It is named for mg21 so it can never be mistaken for mg19's file and can
    never be overwritten on top of one.
    """
    return os.path.join(RESULTS_DIR, f"mg21_{CELL_LABEL}_{GEOMETRY}_phantom.npy")


def _md5_path(path):
    return path + ".md5"


def _md5(path, chunk=8 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _write_md5(path):
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    return digest


def _staged(path):
    """Whether an artifact and its checksum are both on disk."""
    return os.path.exists(path) and os.path.exists(_md5_path(path))


def _recorded_md5(path):
    """The checksum written beside an artifact, without re-reading the artifact.

    Used for a file this run does NOT open.  A checksum that was never checked
    against the bytes is not a verification, and every caller of this function
    records it as a recorded value rather than a verified one.
    """
    try:
        with open(_md5_path(path)) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _verified_load(path):
    """Memory-map a staged artifact after checking its checksum.

    Every read of a large artifact goes through here.  A truncated file on a
    shared parallel filesystem is a recorded failure mode of this work, and a
    comparison against a file that changed underneath the run would be a quietly
    wrong answer rather than a loud one.  Nothing here loads the array into host
    memory: the sinogram is 30.5 GiB.
    """
    import numpy as np

    expected = _recorded_md5(path)
    actual = _md5(path)
    if actual != expected:
        raise RuntimeError(f"staged artifact checksum mismatch at {path}: "
                           f"{actual} != {expected}")
    return np.load(path, mmap_mode="r"), actual


def _stage_memmap(path, shape, fill, row_step):
    """Write one large artifact through a memory map, then checksum it.

    ``fill(out, start, stop)`` writes rows [start, stop) of the map.  The file
    is built under a temporary name and renamed at the end, so a killed job
    cannot leave a half-written array where a later run would read from.
    """
    import numpy as np
    from numpy.lib.format import open_memmap

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".partial"
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=tuple(shape))
    try:
        for start in range(0, int(shape[0]), row_step):
            fill(out, start, min(start + row_step, int(shape[0])))
        out.flush()
    finally:
        del out
    os.replace(tmp, path)
    return _write_md5(path)


# ── the GPU health sample ─────────────────────────────────────────────────────
# A throttled GPU produces a valid record structure and an invalid timing, and
# every number this run makes is a timing.
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
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


def row_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── model construction ────────────────────────────────────────────────────────
def _build_model(pin_devices=None):
    """Build the cone model at this run's cell.

    This is mg19's construction, unchanged, so a row here and a row there
    describe the same model.  On CUDA nothing is configured: the device count is
    pinned through MBIRTORCH_NUM_DEVICES, which leaves the model on the
    automatic branch the production path takes.  ``pin_devices`` is the CPU
    smoke's path only, because the automatic search short-circuits when fewer
    than two CUDA devices are visible.
    """
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    num_views, channels = shape[0], shape[2]
    # A full turn of views, and the source distances written as multiples of the
    # detector width so the same expression builds the smoke model: the
    # source-to-detector distance at four widths and the source-to-isocenter
    # distance at two, which puts the object halfway between the source and the
    # detector at a magnification of two.
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    model = mbirtorch.ConeBeamModel(
        shape, angles, source_detector_dist=4.0 * channels,
        source_iso_dist=2.0 * channels)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _shape_check(model):
    """Did the geometry defaults produce the recon shape this file registered?

    Recorded rather than raised: a moved default is worth knowing about, but it
    does not make a timing wrong.
    """
    realized = tuple(int(s) for s in model.get_params("recon_shape"))
    pixels = int(model.full_index_count())
    return dict(recon_shape=list(realized), num_pixels_full=pixels,
                recon_shape_expected=list(recon_shape()),
                recon_shape_ok=(realized == tuple(recon_shape())),
                num_pixels_expected=int(num_pixels()),
                num_pixels_ok=(pixels == int(num_pixels())))


def _block_lengths(model):
    """The block each device owns on the two sharded axes, and whether the
    device count divides either axis exactly.

    At the production cell three devices split 2048 views 683/683/682 and 2016
    slices 672/672/672, so the slice axis divides and the view axis does not.
    Four devices divide both.  Recorded on every row so the reader does not have
    to re-derive it.
    """
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    shape = tuple(model.get_params("recon_shape"))
    n = model.sino_placement.n_devices
    views = [end - start for _device, (start, end)
             in model.sino_placement.shard_ranges(sinogram_shape[0])]
    slices = [end - start for _device, (start, end)
              in model.recon_placement.shard_ranges(shape[2])]
    return dict(view_blocks=views, slice_blocks=slices,
                views_divide=(int(sinogram_shape[0]) % max(1, n) == 0),
                slices_divide=(int(shape[2]) % max(1, n) == 0))


# ── device-side helpers ───────────────────────────────────────────────────────
def _sync(model, cuda):
    """Wait for every placement device to finish.

    A wall taken without this measures how long it took to QUEUE the work, not
    how long the work took.
    """
    if not cuda:
        return
    import torch

    for device in model.sino_placement.devices:
        torch.cuda.synchronize(device)


def _release(cuda):
    if not cuda:
        return
    import torch

    torch.cuda.empty_cache()


def _first_tensor_device(args):
    """The device of the first tensor in an argument list.

    The two builder functions take a tensor first, so this finds the device
    their work will run on without the wrapper having to know either signature.
    """
    import torch

    for value in args:
        if isinstance(value, torch.Tensor):
            return value.device
    return None


# ── the instrument ────────────────────────────────────────────────────────────
class BackProbe:
    """Six brackets inside the sharded cone back projection, plus the reduce.

    HOW RECORDS ARE CORRELATED.  The drivers run one worker thread per device,
    so a record made deep inside a body call has to find the view-range call it
    belongs to without being passed a handle.  A ``threading.local`` carries the
    innermost two contexts -- the current view-range record and the current body
    record -- and a body call, its two builders and its kernel launch all run on
    one thread, so the builders and the launch find their body record there.

    The pass id and the band index are NOT thread-local.  The pass driver runs
    on the calling thread and the workers run on other threads, so those two are
    plain attributes on this object.  That is correct here because this harness
    never has two passes in flight at once, and because the driver fully joins
    each band's workers before it starts the next band.  The band index is
    exactly the number of reduces already done in this pass.

    HOW THE DEVICE TIMES ARE TAKEN.  Every bracket creates and records its two
    events inside ``with torch.cuda.device(dev)``, which is what puts the
    markers on the stream carrying that device's work.  Nothing is synchronized
    inside a pass.  :meth:`finish` synchronizes every placement device, reads
    every elapsed time into its record, and releases the events, so the number
    of live events is bounded by one timed repeat rather than by the whole arm.

    On the CPU smoke there are no CUDA events and the host wall of each call
    stands in for its device time.  Every record says which backend produced its
    numbers.
    """

    def __init__(self, torch_module, cuda):
        self.torch = torch_module
        self.cuda = cuda
        self.local = threading.local()
        self.lock = threading.Lock()
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"

        # Records, appended when a call STARTS so the inner calls can point at
        # them.  A call that raises leaves a record without timings, which is
        # visible rather than missing.
        self.passes = []
        self.view_calls = []
        self.body_calls = []
        self.reduces = []

        # Correlation state.
        self.current_pass = None
        self.reduce_count = 0          # bands finished in the current pass
        self.variant = None
        self.repeat = None

        # Event bookkeeping for the repeat in progress.
        self._events = []
        self.pair_count = 0
        self.pair_high_water = 0
        self.cap_hit = False

        # Calls that arrive without the context they should have, counted
        # rather than raised so an unexpected call appears as a number on the
        # row instead of stopping the arm.
        self.orphan_builder_calls = 0
        self.orphan_kernel_calls = 0
        self.orphan_body_calls = 0

        # Which patches actually went on.
        self.installed = {}

    # -- event helpers ----------------------------------------------------
    def _start(self, device):
        """Open one bracket.  Returns a start event, the string ``'host'`` on
        the CPU smoke, or None when the event budget is exhausted."""
        if not self.cuda or device is None:
            return "host"
        with self.lock:
            if self.pair_count >= MAX_EVENT_PAIRS:
                self.cap_hit = True
                return None
        with self.torch.cuda.device(device):
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
        return event

    def _stop(self, record, key, device, start, host_s):
        """Close one bracket.  The elapsed time is NOT read here; it is read in
        :meth:`finish`, after every device has been synchronized."""
        if start is None:
            return
        if start == "host":
            record[key] = host_s
            return
        with self.torch.cuda.device(device):
            end = self.torch.cuda.Event(enable_timing=True)
            end.record()
        with self.lock:
            self._events.append((record, key, start, end))
            self.pair_count += 1
            self.pair_high_water = max(self.pair_high_water, self.pair_count)

    def begin_repeat(self, variant, repeat):
        self.variant = variant
        self.repeat = repeat
        self.pair_count = 0
        self.pair_high_water = 0

    def finish(self, devices):
        """Per-device synchronize, THEN read every span.  Never inside a pass.

        The events are released here, so the next repeat starts with an empty
        budget and the arm never holds more events than one repeat makes.
        """
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
            for record, key, start, end in self._events:
                record[key] = start.elapsed_time(end) / 1e3
        self._events = []
        self.pair_count = 0

    # -- record helpers ---------------------------------------------------
    def _new(self, bucket, record):
        with self.lock:
            record["index"] = len(bucket)
            bucket.append(record)
        return record

    # -- 1: the pass ------------------------------------------------------
    def wrap_pass(self, original):
        """Bracket one whole sharded back projection call.

        The band list is rebuilt here from the model's own two helpers, so the
        record says exactly which bands the driver is about to walk without this
        wrapper having to guess.
        """
        probe = self

        @functools.wraps(original)
        def wrapped(model, sino_shards, pixel_indices, coeff_power=1):
            count = int(getattr(pixel_indices, "shape", [0])[0])
            record = probe._new(probe.passes, dict(
                kind="pass", variant=probe.variant, repeat=probe.repeat,
                num_pixels=count, coeff_power=int(coeff_power),
                bands=probe._band_plan(model, count)))
            probe.current_pass = record["index"]
            probe.reduce_count = 0
            host0 = time.perf_counter()
            try:
                return original(model, sino_shards, pixel_indices,
                                coeff_power=coeff_power)
            finally:
                record["host_s"] = time.perf_counter() - host0
                record["bands_walked"] = probe.reduce_count
                probe.current_pass = None
        return wrapped

    @staticmethod
    def _band_plan(model, count):
        """The bands the driver will walk, computed with the driver's own two
        helpers so the plan cannot drift from the loop."""
        fixed = getattr(model, "back_project_slice_band", None)
        n_dev = model.sino_placement.n_devices
        bands = []
        for owner, (s0, s1) in model.recon_placement.shard_ranges():
            band_len = model._slice_band_length(s1 - s0, n_dev, count, fixed)
            for (l0, l1) in model._balanced_slice_bounds(s1 - s0, band_len):
                bands.append(dict(owner=str(owner), slice_start=int(s0 + l0),
                                  band_slices=int(l1 - l0)))
        return bands

    # -- 2: one device's work on one band ---------------------------------
    def wrap_view_range(self, original):
        probe = self

        @functools.wraps(original)
        def wrapped(projectors, local_sino, pixel_indices, view_range,
                    coeff_power=1, slice_start=0, band_slices=None,
                    dev_index=0, plan=None):
            device = local_sino.device
            record = probe._new(probe.view_calls, dict(
                kind="view_range", variant=probe.variant, repeat=probe.repeat,
                pass_id=probe.current_pass, band_index=probe.reduce_count,
                device=str(device), dev_index=int(dev_index),
                view_range=[int(view_range[0]), int(view_range[1])],
                slice_start=int(slice_start),
                band_slices=(None if band_slices is None else int(band_slices)),
                num_pixels=int(pixel_indices.shape[0]),
                coeff_power=int(coeff_power), body_calls=0))
            previous = getattr(probe.local, "view", None)
            probe.local.view = record
            start = probe._start(device)
            host0 = time.perf_counter()
            try:
                return original(projectors, local_sino, pixel_indices,
                                view_range, coeff_power=coeff_power,
                                slice_start=slice_start,
                                band_slices=band_slices, dev_index=dev_index,
                                plan=plan)
            finally:
                host = time.perf_counter() - host0
                record["host_s"] = host
                probe._stop(record, "dev_s", device, start, host)
                probe.local.view = previous
        return wrapped

    # -- 3 and 4: the two eager builders inside the body ------------------
    def wrap_builder(self, name, original):
        """Bracket one of the body's two precomputes.

        The device comes from the first tensor argument, which is where both
        builders take their input.  A call that arrives with no body context
        cannot be attributed and is counted instead; that would mean the torch
        body ran, which happens only at a coeff_power this run never uses.
        """
        probe = self

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            body = getattr(probe.local, "body", None)
            if body is None:
                probe.orphan_builder_calls += 1
                return original(*args, **kwargs)
            device = _first_tensor_device(args)
            start = probe._start(device)
            host0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                body[name + "_host_s"] = host
                probe._stop(body, name + "_s", device, start, host)
        return wrapped

    # -- 5: the body ------------------------------------------------------
    def wrap_body(self, original):
        """Bracket one body call and record the view batch it realized.

        THE WRAPPER MUST CARRY TWO ATTRIBUTES.  ``_view_batch_cost`` is what the
        driver reads to choose the view batch, and ``_mbirtorch_no_compile`` is
        what keeps ``maybe_compile`` from compiling the wrapper.  ``functools.
        wraps`` copies the original's ``__dict__``, which brings both across;
        they are then set again by name and asserted by the installer, because
        this run measures the wrong workload without the first and traces its
        own wrapper without the second.
        """
        probe = self

        @functools.wraps(original)
        def wrapped(sino_batch, pixel_indices, view_params_batch, *args,
                    **kwargs):
            view = getattr(probe.local, "view", None)
            if view is None:
                probe.orphan_body_calls += 1
            device = sino_batch.device
            record = probe._new(probe.body_calls, dict(
                kind="body", variant=probe.variant, repeat=probe.repeat,
                pass_id=probe.current_pass,
                band_index=(view["band_index"] if view else None),
                view_call=(view["index"] if view else None),
                device=str(device),
                view_batch=int(view_params_batch.shape[0]),
                num_pixels=int(pixel_indices.shape[0]),
                slice_start=int(kwargs.get("slice_start", 0)),
                band_slices=(None if kwargs.get("band_slices") is None
                             else int(kwargs["band_slices"])),
                coeff_power=int(kwargs.get("coeff_power", 1))))
            if view is not None:
                view["body_calls"] += 1
            previous = getattr(probe.local, "body", None)
            probe.local.body = record
            start = probe._start(device)
            host0 = time.perf_counter()
            try:
                return original(sino_batch, pixel_indices, view_params_batch,
                                *args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                record["host_s"] = host
                probe._stop(record, "dev_s", device, start, host)
                probe.local.body = previous
        return wrapped

    # -- 6: the Triton launch ---------------------------------------------
    def kernel_proxy(self, kernel):
        """Stand in for the compiled kernel object so the launch can be timed.

        The body launches as ``_cone_back_kernel[grid](...)``.  This proxy
        answers that subscript, records the grid, and returns a callable that
        brackets the real launcher and forwards every argument unchanged.  The
        events are created on the CURRENT device, which is the right one: the
        body performs its launch inside ``with torch.cuda.device(...)`` on the
        sinogram's device, so the current device inside the launch is the
        device the kernel runs on.  Everything else on the kernel object is
        forwarded, so a caller that reads another attribute still gets the real
        one; only the subscript is used by the body.
        """
        probe = self

        class _KernelProxy:
            def __init__(self, wrapped_kernel):
                self._kernel = wrapped_kernel

            def __getitem__(self, grid):
                launcher = self._kernel[grid]

                def run(*args, **kwargs):
                    body = getattr(probe.local, "body", None)
                    if body is None:
                        probe.orphan_kernel_calls += 1
                        return launcher(*args, **kwargs)
                    device = (probe.torch.cuda.current_device()
                              if probe.cuda else None)
                    body["grid"] = [int(g) for g in grid]
                    start = probe._start(device)
                    host0 = time.perf_counter()
                    try:
                        return launcher(*args, **kwargs)
                    finally:
                        host = time.perf_counter() - host0
                        body["kernel_host_s"] = host
                        probe._stop(body, "kernel_s", device, start, host)
                return run

            def __getattr__(self, name):
                return getattr(self._kernel, name)

        return _KernelProxy(kernel)

    # -- the cross-device sum at the end of each band ----------------------
    def wrap_reduce(self, original):
        """Bracket the sum of one band's partials onto its owner.

        The event pair is recorded on the OWNER's stream, so it measures the owner's
        own work: waiting for each partial to arrive and adding it in.  A copy
        that a source device issues on its own stream shows up here only through
        the owner's wait for it, which is the number that matters, because the
        owner is what the next band waits for.
        """
        probe = self

        @functools.wraps(original)
        def wrapped(partials, owner, dev2dev_safe=True):
            shape = None
            if partials:
                shape = [int(s) for s in partials[0].shape]
            record = probe._new(probe.reduces, dict(
                kind="reduce", variant=probe.variant, repeat=probe.repeat,
                pass_id=probe.current_pass, band_index=probe.reduce_count,
                owner=str(owner), n_partials=len(partials),
                partial_shape=shape,
                partial_devices=[str(p.device) for p in partials]))
            start = probe._start(owner)
            host0 = time.perf_counter()
            try:
                return original(partials, owner, dev2dev_safe=dev2dev_safe)
            finally:
                host = time.perf_counter() - host0
                record["host_s"] = host
                probe._stop(record, "dev_s", owner, start, host)
                probe.reduce_count += 1
        return wrapped


def install_probe(torch_module, cuda):
    """Put every instrument in place, BEFORE any model is built.

    Nothing in the mbirtorch package is edited.  Two patches replace methods on
    classes and the rest replace names in module namespaces, and every one of
    them calls the original.

    WHY THE MODULE-NAMESPACE PATCHES WORK.  ``ConeBeamModel._view_batch_bodies``
    performs ``from .triton_cone import _cone_back_view_batch_triton`` when it
    is called, not when the module is imported, so it picks up whatever that
    name refers to at the time the projectors are built.  The body reads
    ``_cone_horizontal_data``, ``_cone_vertical_affine`` and
    ``_cone_back_kernel`` as globals of its own module for the same reason.  So
    the patches must be in place before the model settles its layout, and this
    function is called before the model is even constructed.

    Returns the probe and a record of what went on.
    """
    from mbirtorch import _sharding
    from mbirtorch.projectors import Projectors
    from mbirtorch.tomography_model import TomographyModel

    probe = BackProbe(torch_module, cuda)
    installed = probe.installed

    TomographyModel._sparse_back_project_sharded = probe.wrap_pass(
        TomographyModel._sparse_back_project_sharded)
    installed["pass"] = True

    Projectors.sparse_back_project_view_range = probe.wrap_view_range(
        Projectors.sparse_back_project_view_range)
    installed["view_range"] = True

    _sharding.sum_band_to_owner = probe.wrap_reduce(_sharding.sum_band_to_owner)
    installed["reduce"] = True

    # The Triton module imports without triton, so this import succeeds on a
    # CPU machine as well.  What differs there is that nothing calls these
    # names, because the model binds the torch body instead.
    try:
        from mbirtorch import triton_cone
    except Exception as exc:                                      # noqa: BLE001
        installed["triton_module"] = f"not importable: {exc}"
        return probe, dict(installed)
    installed["triton_module"] = True

    triton_cone._cone_horizontal_data = probe.wrap_builder(
        "horizontal", triton_cone._cone_horizontal_data)
    triton_cone._cone_vertical_affine = probe.wrap_builder(
        "vertical", triton_cone._cone_vertical_affine)
    installed["builders"] = True

    original_body = triton_cone._cone_back_view_batch_triton
    body = probe.wrap_body(original_body)
    # Set again by name even though functools.wraps copied them, and then
    # checked.  Losing either one silently changes what this run measures.
    body._view_batch_cost = original_body._view_batch_cost
    body._mbirtorch_no_compile = original_body._mbirtorch_no_compile
    if getattr(body, "_view_batch_cost", None) is not original_body._view_batch_cost:
        raise RuntimeError("the body wrapper does not carry _view_batch_cost; "
                           "the driver would choose a different view batch and "
                           "this run would measure a workload production never "
                           "runs")
    if not getattr(body, "_mbirtorch_no_compile", False):
        raise RuntimeError("the body wrapper does not carry "
                           "_mbirtorch_no_compile; maybe_compile would compile "
                           "the wrapper and trace the launch inside it")
    triton_cone._cone_back_view_batch_triton = body
    installed["body"] = True
    installed["body_attributes"] = ["_view_batch_cost", "_mbirtorch_no_compile"]

    triton_cone._cone_back_kernel = probe.kernel_proxy(
        triton_cone._cone_back_kernel)
    installed["kernel_proxy"] = True

    probe.probe_body = body
    return probe, dict(installed)


# ── per-repeat aggregation ────────────────────────────────────────────────────
def _dev_seconds(record, key):
    value = record.get(key)
    return 0.0 if value is None else float(value)


def _aggregate(probe, variant, repeat, warmup, wall_s, devices, coeff_power,
               count, bands_expected):
    """Turn one repeat's records into the per-device totals the summary prints.

    The per-device totals are the deliverable.  The raw records stay in the
    worker: at the production cell one full-pixel repeat makes about 640 body
    records, and writing them all would make the jsonl large without adding a
    reading.
    """
    def mine(records):
        return [r for r in records
                if r.get("variant") == variant and r.get("repeat") == repeat]

    passes = mine(probe.passes)
    views = mine(probe.view_calls)
    bodies = mine(probe.body_calls)
    reduces = mine(probe.reduces)

    names = [str(d) for d in devices]
    per_device = {}
    for name in names:
        per_device[name] = dict(
            view_range_dev_s=0.0, view_range_host_s=0.0, view_calls=0,
            body_dev_s=0.0, body_host_s=0.0, body_calls=0,
            builders_dev_s=0.0, kernel_dev_s=0.0, residual_dev_s=0.0,
            accum_dev_s=0.0, reduce_dev_s=0.0, reduce_host_s=0.0,
            reduce_calls=0)

    for record in views:
        entry = per_device.setdefault(record["device"], dict())
        entry["view_range_dev_s"] = entry.get("view_range_dev_s", 0.0) + \
            _dev_seconds(record, "dev_s")
        entry["view_range_host_s"] = entry.get("view_range_host_s", 0.0) + \
            _dev_seconds(record, "host_s")
        entry["view_calls"] = entry.get("view_calls", 0) + 1

    view_batch_census, grid_census = {}, {}
    for record in bodies:
        entry = per_device.setdefault(record["device"], dict())
        total = _dev_seconds(record, "dev_s")
        builders = _dev_seconds(record, "horizontal_s") + \
            _dev_seconds(record, "vertical_s")
        kernel = _dev_seconds(record, "kernel_s")
        entry["body_dev_s"] = entry.get("body_dev_s", 0.0) + total
        entry["body_host_s"] = entry.get("body_host_s", 0.0) + \
            _dev_seconds(record, "host_s")
        entry["builders_dev_s"] = entry.get("builders_dev_s", 0.0) + builders
        entry["kernel_dev_s"] = entry.get("kernel_dev_s", 0.0) + kernel
        entry["residual_dev_s"] = entry.get("residual_dev_s", 0.0) + \
            (total - builders - kernel)
        entry["body_calls"] = entry.get("body_calls", 0) + 1
        key = str(record.get("view_batch"))
        view_batch_census[key] = view_batch_census.get(key, 0) + 1
        if record.get("grid"):
            gkey = "x".join(str(g) for g in record["grid"])
            grid_census[gkey] = grid_census.get(gkey, 0) + 1

    for record in reduces:
        entry = per_device.setdefault(record["owner"], dict())
        entry["reduce_dev_s"] = entry.get("reduce_dev_s", 0.0) + \
            _dev_seconds(record, "dev_s")
        entry["reduce_host_s"] = entry.get("reduce_host_s", 0.0) + \
            _dev_seconds(record, "host_s")
        entry["reduce_calls"] = entry.get("reduce_calls", 0) + 1

    for entry in per_device.values():
        # What the view-range loop spent OUTSIDE its body calls.  The loop adds
        # each body's block into an accumulator, and that addition is a full
        # pass over a band-sized array per body call after the first.
        entry["accum_dev_s"] = entry.get("view_range_dev_s", 0.0) - \
            entry.get("body_dev_s", 0.0)
        entry["device_total_s"] = entry.get("view_range_dev_s", 0.0) + \
            entry.get("reduce_dev_s", 0.0)

    # The per-band table: for each band, which device owns it, how long the
    # slowest worker took on it, and what its reduce cost.  There is no table
    # when the funnel took its single-device path, because that path has no
    # bands at all.
    band_rows = []
    for index in range(0 if not passes else
                       max(bands_expected, probe_band_count(reduces, views))):
        band_views = [r for r in views if r.get("band_index") == index]
        band_reduce = [r for r in reduces if r.get("band_index") == index]
        if not band_views and not band_reduce:
            continue
        band_rows.append(dict(
            band_index=index,
            owner=(band_reduce[0]["owner"] if band_reduce else None),
            slice_start=(band_views[0]["slice_start"] if band_views else None),
            band_slices=(band_views[0]["band_slices"] if band_views else None),
            workers=len(band_views),
            max_worker_view_s=(max((_dev_seconds(r, "dev_s")
                                    for r in band_views), default=None)),
            sum_worker_view_s=sum(_dev_seconds(r, "dev_s")
                                  for r in band_views),
            reduce_dev_s=(_dev_seconds(band_reduce[0], "dev_s")
                          if band_reduce else None),
            reduce_host_s=(_dev_seconds(band_reduce[0], "host_s")
                           if band_reduce else None),
            n_partials=(band_reduce[0]["n_partials"] if band_reduce else None)))

    pass_host_s = sum(_dev_seconds(p, "host_s") for p in passes)
    # The worker count comes from the driver's device INDEX, not from the device
    # name.  Two virtual CPU devices are both called 'cpu', so counting names
    # would say one worker where there are two.
    workers = len({r["dev_index"] for r in views}) or 1
    expected_views = bands_expected * workers
    accounting_ok = True
    accounting_note = ""
    if passes:
        if len(passes) != 1:
            accounting_ok = False
            accounting_note = f"{len(passes)} pass records for one call"
        elif len(views) != expected_views or len(reduces) != bands_expected:
            accounting_ok = False
            accounting_note = (
                f"{len(views)} view-range records and {len(reduces)} reduce "
                f"records for {bands_expected} band(s) and {workers} worker(s); "
                f"expected {expected_views} and {bands_expected}")
    else:
        # No pass record means the funnel took its single-device path, which
        # runs no bands and no reduce at all.  One view-range call is the whole
        # of that path.
        if len(views) != 1 or reduces:
            accounting_ok = False
            accounting_note = (
                f"the single-device path made {len(views)} view-range record(s) "
                f"and {len(reduces)} reduce record(s); expected 1 and 0")
        else:
            accounting_note = ("the single-device path: no sharded pass, so no "
                               "bands and no reduce")

    return dict(
        variant=variant, repeat=repeat, warmup=bool(warmup),
        coeff_power=int(coeff_power), num_pixels=int(count),
        wall_s=float(wall_s), pass_host_s=pass_host_s,
        sharded=bool(passes), bands_planned=int(bands_expected),
        bands_expected=(int(bands_expected) if passes else 0),
        bands_walked=(passes[0].get("bands_walked") if passes else 0),
        n_view_records=len(views), n_body_records=len(bodies),
        n_reduce_records=len(reduces),
        accounting_ok=accounting_ok, accounting_note=accounting_note,
        per_device=per_device, view_batch_census=view_batch_census,
        grid_census=grid_census, bands=band_rows,
        event_backend=probe.backend, event_pairs=probe.pair_high_water,
        event_cap_hit=probe.cap_hit)


def probe_band_count(reduces, views):
    """How many distinct bands the records mention.

    Used only to size the per-band table when the driver walked more bands than
    the plan predicted, which would itself be a finding.
    """
    indices = {r.get("band_index") for r in reduces}
    indices |= {r.get("band_index") for r in views}
    indices.discard(None)
    return (max(indices) + 1) if indices else 0


# ── the value witness ─────────────────────────────────────────────────────────
def _shard_witness(tensor, torch_module):
    """Sum of absolute values and sum of squares, in float64, on the device.

    Chunked over rows.  A whole float64 promotion of one production output shard
    is 12 GiB, and a float32 reduction over 1.5 billion values loses digits, so
    neither the promotion nor the single-call reduction is used.
    """
    rows = int(tensor.shape[0])
    if rows == 0 or tensor.numel() == 0:
        return 0.0, 0.0
    per_row = max(1, tensor.numel() // rows)
    step = max(1, WITNESS_CHUNK_BYTES // (per_row * 8))
    abs_sum = torch_module.zeros((), dtype=torch_module.float64,
                                 device=tensor.device)
    sq_sum = torch_module.zeros((), dtype=torch_module.float64,
                                device=tensor.device)
    for start in range(0, rows, step):
        block = tensor[start:start + step].to(torch_module.float64)
        abs_sum += block.abs().sum()
        sq_sum += (block * block).sum()
    return float(abs_sum), float(sq_sum)


def _output_witness(output, torch_module):
    """The two scalars for a whole output, whether it is shards or one tensor."""
    tensors = getattr(output, "tensors", None)
    if tensors is None:
        tensors = [output]
    abs_total, sq_total, shards = 0.0, 0.0, []
    for tensor in tensors:
        abs_sum, sq_sum = _shard_witness(tensor, torch_module)
        shards.append(dict(device=str(tensor.device),
                           shape=[int(s) for s in tensor.shape],
                           abs_sum=abs_sum, sq_sum=sq_sum))
        abs_total += abs_sum
        sq_total += sq_sum
    return dict(abs_sum=abs_total, sq_sum=sq_total, shards=shards)


# ── the input ─────────────────────────────────────────────────────────────────
def _seeded_phantom_into(out, start, stop, cols, slices):
    """Fill rows [start, stop) of the fallback phantom map.

    Called in order from a stream seeded once before the first call, so the map
    ends up holding what a single whole-volume draw would have produced.  numpy
    fills any request in C order from one stream.  The single call is what this
    avoids: at the production shape it would build a 59 GiB float64 array before
    casting it.
    """
    import numpy as np

    out[start:stop] = np.random.rand(stop - start, cols,
                                     slices).astype(np.float32)


def _obtain_sinogram(model, result, cuda):
    """The sharded sinogram every timed call is given.

    mg19's staged cone sinogram is the input when it is on disk, because reusing
    it costs one checksum and making a new one costs a full forward projection.
    The file is checksum-verified and then placed on the devices through the
    model's own ``_shard_sinogram``, which is the same call the public funnel
    makes.  The funnel recognizes an already-placed input, so no host-to-device
    movement happens inside any bracket.

    mg19's phantom is NOT read when the sinogram is there.  This run never opens
    its bytes, so its recorded checksum is reported as recorded and not as
    verified.

    Without the sinogram the arm makes its own: mg19's phantom if that is
    staged, otherwise a phantom drawn with mg19's seed, forward-projected at
    this arm's own device count and kept in its sharded form.  That path is
    recorded and does not fail the run.  A back projection's time depends on the
    shape of the sinogram it is given, not on its values.
    """
    sino_path = _mg19_sinogram_path()
    phantom_path = _mg19_phantom_path()
    result["mg19_dir"] = MG19_DIR
    result["mg19_sinogram_path"] = sino_path
    result["mg19_phantom_path"] = phantom_path
    result["mg19_phantom_staged"] = _staged(phantom_path)
    result["mg19_phantom_md5_recorded"] = _recorded_md5(phantom_path)
    result["mg19_phantom_note"] = (
        "recorded, not verified: this run reads the sinogram and never opens "
        "the phantom's bytes")

    if _staged(sino_path):
        start = time.perf_counter()
        sinogram, digest = _verified_load(sino_path)
        result["sinogram_md5_verified"] = digest
        result["sinogram_verify_s"] = time.perf_counter() - start
        result["sinogram_fresh"] = False
        result["sinogram_source"] = "mg19's staged cone sinogram"
        start = time.perf_counter()
        shards = model._shard_sinogram(sinogram)
        _sync(model, cuda)
        result["sinogram_place_s"] = time.perf_counter() - start
        del sinogram
        return shards

    # The fallback.
    result["sinogram_fresh"] = True
    result["sinogram_md5_verified"] = None
    shape = tuple(recon_shape())
    if _staged(phantom_path):
        phantom, digest = _verified_load(phantom_path)
        result["phantom_md5_verified"] = digest
        result["phantom_source"] = "mg19's staged phantom"
    else:
        import numpy as np

        own = _own_phantom_path()
        if not _staged(own):
            rows, cols, slices = (int(s) for s in shape)
            np.random.seed(PHANTOM_SEED)
            _stage_memmap(own, (rows, cols, slices),
                          lambda out, a, b: _seeded_phantom_into(out, a, b,
                                                                 cols, slices),
                          PHANTOM_SLAB_ROWS)
        phantom, digest = _verified_load(own)
        result["phantom_md5_verified"] = digest
        result["phantom_source"] = f"drawn here with mg19's seed into {own}"
    result["sinogram_source"] = (
        "forward-projected here at this arm's own device count; mg19's staged "
        "sinogram was not present")
    start = time.perf_counter()
    shards = model.forward_project(phantom, output_sharded=True)
    _sync(model, cuda)
    result["sinogram_forward_s"] = time.perf_counter() - start
    del phantom
    _release(cuda)
    return shards


def _pixel_sets(model):
    """The full pixel set and one subset at each schedule granularity.

    The subsets come from ONE seeded permutation, first block first, so they
    have the right size and no spatial structure.  A real VCD subset is
    spatially structured, and structure changes how a gather hits memory.  These
    are therefore a size sweep, and the reconstruction estimate built from them
    is an estimate.
    """
    import numpy as np

    full = np.asarray(model._full_indices()).reshape(-1).astype(np.int64)
    count = int(full.shape[0])
    order = np.random.RandomState(SUBSET_SEED).permutation(count)
    sets = {"full": full}
    for g in SUBSET_GRANULARITIES:
        size = max(1, count // int(g))
        sets[f"sub{g}"] = full[order[:size]].copy()
    return sets, count


def _variant_plan(pixel_sets, count, largest_slice_shard):
    """Every variant, in the order the arm takes them."""
    half = max(1, -(-int(largest_slice_shard) // 2))     # ceil division
    plan = [
        dict(name="full_p1", indices="full", coeff_power=1, band=None,
             what="the reference call: every pixel, the gradient path"),
        dict(name="full_p2", indices="full", coeff_power=2, band=None,
             what="the Hessian call a reconstruction makes once"),
    ]
    for g in SUBSET_GRANULARITIES:
        plan.append(dict(name=f"sub{g}", indices=f"sub{g}", coeff_power=1,
                         band=None,
                         what=f"one subset of about P/{g} pixels"))
    plan.append(dict(
        name="band_half", indices="full", coeff_power=1, band=half,
        what="every pixel with the slice band halved: twice the band calls, "
             "half the voxel work in each"))
    for entry in plan:
        entry["num_pixels"] = int(pixel_sets[entry["indices"]].shape[0])
    return plan


# ── the worker: one arm, one process ──────────────────────────────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the arm does."""
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda, geometry=GEOMETRY,
                  cell=list(cell()), cell_label=CELL_LABEL,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  realized_pin=pin_for(cfg["n_dev"]),
                  subset_seed=SUBSET_SEED,
                  subset_granularities=list(SUBSET_GRANULARITIES),
                  warmup_repeats=WARMUP_REPEATS, timed_repeats=TIMED_REPEATS,
                  max_event_pairs=MAX_EVENT_PAIRS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    # This run reads no memory ledger and must not own the per-device peak
    # counters, so the calibration mode must be off on every arm.
    present = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1"
    result["calibration_env_present"] = present
    result["calibration_env_ok"] = not present
    result["invalid_reasons"] = []
    if present:
        result["invalid_reasons"].append(
            "MBIRTORCH_MEMORY_CALIBRATION is set; this run makes no memory "
            "reading and must not reset the peak counters")
    return result, cuda


def _settle_and_witness(model, cfg, result, cuda, probe):
    """Settle the layout and record everything that says what this arm is.

    THE SETTLE MUST HAPPEN BEFORE ANYTHING IS MEASURED.  On CUDA an arm pins
    its device count through
    MBIRTORCH_NUM_DEVICES, that pin acts only through the model's device policy,
    and a freshly built automatic model still holds the trivial single-device
    placement.  A projection taken before the layout has settled would run on
    one device at every device count, so this run would compare two identical
    arms and call the difference a device-count effect.
    """
    from mbirtorch import _memory_ledger, _widening_floors

    n_dev = pin_for(cfg["n_dev"])
    model._apply_device_policy()

    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))

    # WHICH BODIES ARE BOUND.  On an H100 cone binds hand-written Triton kernels
    # in both directions, so the list of directions running as general torch
    # code must be EMPTY.  On the CPU smoke the kernels do not exist and both
    # directions fall back, so it must be both.  The row says which environment
    # it expected, because the two answers are opposites.
    directions = tuple(_memory_ledger.torch_body_directions(model))
    expected = () if cuda else ("forward", "back")
    result["torch_body_directions"] = list(directions)
    result["torch_body_expected"] = list(expected)
    result["torch_body_expectation_basis"] = (
        "H100: cone binds hand-written Triton kernels in both directions, so "
        "no direction runs as general torch code" if cuda else
        "CPU smoke: the Triton kernels are unavailable, so both directions "
        "fall back to general torch code")
    result["torch_bodies_ok"] = (directions == expected)

    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__
    # The wrapper reports the original's name, so identity is what says whether
    # the instrument is really in the call path.
    wrapper = getattr(probe, "probe_body", None)
    result["back_body_is_probe_wrapper"] = (
        wrapper is not None and back_body is wrapper)
    result["back_body_has_view_batch_cost"] = (
        getattr(back_body, "_view_batch_cost", None) is not None)
    result["back_body_no_compile"] = bool(
        getattr(back_body, "_mbirtorch_no_compile", False))
    result["probe_installed"] = dict(probe.installed)

    result["forward_pixel_batch"] = int(model._forward_pixel_batch())
    result["forward_pixel_batch_note"] = (
        "recorded, not gated: this run measures the back projection and never "
        "brackets a forward call")
    result["back_project_slice_band_attr"] = getattr(
        model, "back_project_slice_band", None)
    result["reduce_slab_bytes"] = int(_reduce_slab_bytes())
    result["widening_floors_stale_note"] = _widening_floors.stale_note()

    result.update(_shape_check(model))
    result["blocks"] = _block_lengths(model)

    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"realized {realized} for a pin of {n_dev}")
    if not result["torch_bodies_ok"]:
        result["invalid_reasons"].append(
            f"torch_body_directions is {list(directions)}, not "
            f"{list(expected)}; this arm is not running the bodies it claims")
    if cuda and not result["back_body_is_probe_wrapper"]:
        result["invalid_reasons"].append(
            "the model did not bind this run's body wrapper, so instruments 3 "
            "through 6 record nothing and the arm measures no parts")


def _reduce_slab_bytes():
    from mbirtorch import _sharding

    return _sharding.REDUCE_SLAB_BYTES


def _microbench_channel_major(torch_module, view_batch, result):
    """Price the channel-major copy the residual is mostly made of.

    The body writes ``sino_batch.permute(0, 2, 1).contiguous()`` before every
    launch.  That copy reads and writes a block sized by the DETECTOR, so it
    does not shrink when the slice band shrinks.  This measures it directly on a
    fresh tensor of the shape the arm actually used, so the residual can be read
    against a measurement of its main constituent rather than against a guess.
    """
    rows, channels = int(cell()[1]), int(cell()[2])
    device = torch_module.device("cuda:0")
    source = torch_module.empty((int(view_batch), rows, channels),
                                dtype=torch_module.float32, device=device)
    source.normal_()
    torch_module.cuda.synchronize(device)
    times = []
    for _ in range(MICROBENCH_REPEATS + 1):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        copied = source.permute(0, 2, 1).contiguous()
        end.record()
        torch_module.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / 1e3)
        del copied
    # The first repeat is dropped: it pays the allocator's first request for the
    # destination block, which the later ones reuse.
    times = sorted(times[1:])
    median = times[len(times) // 2]
    moved = 2 * int(view_batch) * rows * channels * 4
    result["microbench"] = dict(
        shape=[int(view_batch), rows, channels],
        repeats=MICROBENCH_REPEATS, seconds=times, median_s=median,
        bytes_moved=moved,
        effective_gb_per_s=(moved / median / 1e9) if median > 0 else None,
        what="x.permute(0, 2, 1).contiguous() on a fresh float32 tensor of the "
             "shape the body copies before every launch; bytes_moved counts "
             "the read and the write")
    del source
    torch_module.cuda.empty_cache()


def run_arm(cfg):
    """One arm, in its own process.

    A fresh process per arm is not tidiness.  The patches replace names in
    library modules and on library classes, compiled and Triton bodies are
    cached at module level for the life of a process, and the pin acts through
    the environment.  All three would leak from one arm into the next if they
    shared an interpreter.
    """
    result, cuda = _base_result(cfg)
    health = [sample_gpu_health()]
    try:
        import torch

        # Every instrument goes on BEFORE the model exists, because the body
        # selection and the projector binding both read these names once, when
        # the layout settles.
        probe, installed = install_probe(torch, cuda)
        result["probe_installed"] = installed

        model = _build_model(pin_devices=(None if cuda else
                                          [DEVICE] * pin_for(cfg["n_dev"])))
        _settle_and_witness(model, cfg, result, cuda, probe)
        if result["invalid_reasons"]:
            result["timing_skipped_reason"] = "; ".join(result["invalid_reasons"])
            return result

        sino_shards = _obtain_sinogram(model, result, cuda)
        pixel_sets, count = _pixel_sets(model)
        result["pixel_set_sizes"] = {name: int(idx.shape[0])
                                     for name, idx in pixel_sets.items()}
        largest = max(end - start for _d, (start, end)
                      in model.recon_placement.shard_ranges())
        plan = _variant_plan(pixel_sets, count, largest)
        wanted = _strict_subset("MG21_VARIANTS", variant_names())
        plan = [entry for entry in plan if entry["name"] in wanted]
        result["largest_slice_shard"] = int(largest)
        result["variant_plan"] = [dict(entry) for entry in plan]

        devices = list(model.recon_placement.devices)
        repeats = []
        full_batches = {}
        for entry in plan:
            indices = pixel_sets[entry["indices"]]
            if entry["band"] is not None:
                # Set on this model INSTANCE, between calls, and removed again
                # below.  The driver reads it with getattr each time it runs, so
                # no re-settle is needed, and the layout the settle priced still
                # bounds this: a shorter band makes every per-band array
                # SMALLER, so nothing here can exceed what the settle allowed.
                model.back_project_slice_band = int(entry["band"])
            try:
                bands = len(BackProbe._band_plan(model, int(indices.shape[0])))
                for repeat in range(WARMUP_REPEATS + TIMED_REPEATS):
                    warmup = repeat < WARMUP_REPEATS
                    probe.begin_repeat(entry["name"], repeat)
                    _sync(model, cuda)
                    start = time.perf_counter()
                    output = model.sparse_back_project(
                        sino_shards, indices,
                        coeff_power=entry["coeff_power"])
                    _sync(model, cuda)
                    wall = time.perf_counter() - start
                    probe.finish(devices)
                    record = _aggregate(probe, entry["name"], repeat, warmup,
                                        wall, devices, entry["coeff_power"],
                                        indices.shape[0], bands)
                    record["slice_band"] = entry["band"]
                    record["witness"] = _output_witness(output, torch)
                    repeats.append(record)
                    if entry["name"] == "full_p1" and not warmup:
                        for key, hits in record["view_batch_census"].items():
                            full_batches[key] = full_batches.get(key, 0) + hits
                    del output
                    _release(cuda)
            finally:
                if entry["band"] is not None:
                    # Removed rather than set back to None: the library reads
                    # this with getattr and a present None is a different state
                    # from an absent attribute only in principle, but leaving
                    # the instance as it was found is what keeps the next
                    # variant honest.
                    del model.back_project_slice_band
        result["repeats"] = repeats
        result["orphan_calls"] = dict(builder=probe.orphan_builder_calls,
                                      kernel=probe.orphan_kernel_calls,
                                      body=probe.orphan_body_calls)

        if cuda and pin_for(cfg["n_dev"]) == MICROBENCH_ARM and full_batches:
            # The realized view batch of the full-pixel calls, which is what the
            # body actually copied.  The most common value is used; the last
            # batch of each view span is a shorter remainder.
            common = max(full_batches.items(), key=lambda kv: kv[1])[0]
            _microbench_channel_major(torch, int(common), result)

        del sino_shards
        _release(cuda)
    finally:
        health.append(sample_gpu_health())
        result["gpu_health"] = [g for snap in health for g in snap]
        result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited.

    The device pin and the calibration mode are popped first and then set, so a
    value exported by the submitting shell cannot reach an arm that did not ask
    for it.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(pin_for(cfg["n_dev"]))
    return env


def _spawn(cfg):
    """Run one arm in a FRESH interpreter."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        # An arm that runs out of device memory lands here.  That is recorded as
        # a row and the run continues with the next arm.
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def build_plan():
    names = [spec["arm"] for spec in arm_specs()]
    keep = _strict_subset("MG21_ARMS", names)
    # The variant list is parsed here as well as in the worker, so a typo stops
    # the run before any subprocess starts rather than after the first arm has
    # imported torch and built a model at this cell.
    _strict_subset("MG21_VARIANTS", variant_names())
    plan = [dict(spec, arm_id=spec["arm"]) for spec in arm_specs()
            if spec["arm"] in keep]
    if not plan:
        raise ValueError("MG21_ARMS selects no arm")
    return plan


def _dry_run(plan):
    shape = tuple(cell())
    volume = tuple(recon_shape())
    sino_gib = shape[0] * shape[1] * shape[2] * 4 / 2 ** 30
    variants = _strict_subset("MG21_VARIANTS", variant_names())
    print(f"mg21 where the cone back projection spends its time: "
          f"{len(plan)} arm(s), device {DEVICE}")
    print(f"  results -> {RESULTS_DIR}")
    print(f"  mg19's staged cone files read from {MG19_DIR}")
    print(f"  sinogram {shape} {sino_gib:.2f} GiB -> recon {volume}, "
          f"{num_pixels()} pixels inside the mask")
    print(f'  {"arm":<8}{"pin":>5}  what it measures')
    for cfg in plan:
        print(f'  {cfg["arm_id"]:<8}{pin_for(cfg["n_dev"]):>5}  '
              "every variant, instrumented at six places inside the back "
              "projection")
    print(f"  variants, in order: {', '.join(variants)}")
    print(f"  {WARMUP_REPEATS} warmup repeat and {TIMED_REPEATS} timed "
          "repeat(s) per variant; the warmup is recorded and excluded")
    print("no library file is edited: every instrument is a patch applied "
          "inside the arm's own subprocess, before the model is built")


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg21_back_attrib_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg21 cone back attribution on {RUN_LABEL} ({DEVICE}); "
          f"{len(plan)} arm(s) -> {out_path}", flush=True)
    rows = []
    # Rows are written as they finish, so a job that runs out of wall time still
    # yields every arm it completed.
    with open(out_path, "w") as sink:
        for cfg in plan:
            print(f'  {cfg["arm_id"]}', flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        summary = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    return 0 if summary["healthy"] else 2


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=3):
    if value is None:
        return f'{"-":>{width}}'
    return f"{value:>{width}.{prec}{kind}}"


def _timed(row, variant):
    """The timed repeats of one variant, warmup excluded."""
    return [r for r in (row.get("repeats") or [])
            if r.get("variant") == variant and not r.get("warmup")]


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def _variant_mean(row, variant, getter):
    """The mean of one number over a variant's timed repeats."""
    return _mean([getter(r) for r in _timed(row, variant)])


def _busiest(record):
    """The device with the largest total inside this run's brackets."""
    per_device = record.get("per_device") or {}
    if not per_device:
        return None, {}
    name = max(per_device, key=lambda d: per_device[d].get("device_total_s", 0.0))
    return name, per_device[name]


def _parts_on_busiest(row, variant):
    """The named parts on the busiest device, averaged over the timed repeats.

    The parts are additive by construction.  A body call's device time splits
    into builders, kernel and residual.  A view-range call's device time is its
    body calls plus the accumulation between them.  A device's total is its
    view-range time plus the reduces it owns.  The gap is what the pass's host
    wall holds beyond that total.
    """
    records = _timed(row, variant)
    if not records:
        return None
    totals = dict(kernel=[], builders=[], residual=[], accum=[], reduce=[],
                  device_total=[], gap=[], wall=[])
    for record in records:
        name, entry = _busiest(record)
        if name is None:
            continue
        totals["kernel"].append(entry.get("kernel_dev_s"))
        totals["builders"].append(entry.get("builders_dev_s"))
        totals["residual"].append(entry.get("residual_dev_s"))
        totals["accum"].append(entry.get("accum_dev_s"))
        totals["reduce"].append(entry.get("reduce_dev_s"))
        totals["device_total"].append(entry.get("device_total_s"))
        wall = record.get("pass_host_s") or record.get("wall_s")
        totals["wall"].append(wall)
        totals["gap"].append((wall or 0.0) - (entry.get("device_total_s") or 0.0))
    out = {key: _mean(values) for key, values in totals.items()}
    out["device"] = _busiest(records[0])[0]
    return out


def _ratio(new, old):
    if new is None or old is None or old == 0:
        return None
    return new / old


def summarize(rows, out_path):
    """The tables a person reads the attribution from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and this function keeps them apart.  What the
    parts add up to is a FINDING and never touches the exit code.  An arm that
    ran on the wrong device count, did not bind this run's wrapper, produced the
    wrong number of records for its band structure, or ran out of event pairs is
    an instrument failure, because it did not measure what the plan said.
    """
    print(f"\n===== mg21 cone back projection attribution ({out_path}) =====")
    broken, findings = [], []
    by_arm = {}

    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'\n{arm_id}: ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:100]}')
            broken.append(f"{arm_id}|error")
            continue
        by_arm[arm_id] = row
        pin = pin_for(row["n_dev"])
        through = (" (through this run's wrapper)"
                   if row.get("back_body_is_probe_wrapper") else "")
        print(f'\n--- {arm_id}: {row.get("realized_n_devices")} device(s), '
              f'pin {pin} ---')
        print(f'  devices {row.get("realized_devices")}, back body '
              f'{row.get("back_body")}{through}')
        print(f'  view blocks {(row.get("blocks") or {}).get("view_blocks")}, '
              f'slice blocks {(row.get("blocks") or {}).get("slice_blocks")}, '
              f'forward pixel batch {row.get("forward_pixel_batch")}, '
              f'reduce slab {row.get("reduce_slab_bytes", 0) // 2 ** 20} MiB')
        print(f'  sinogram: {row.get("sinogram_source")}')
        if row.get("widening_floors_stale_note"):
            print(f'  widening floors note (informational): '
                  f'{row["widening_floors_stale_note"]}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    ARM CHECK FAIL: {reason}")
            broken.append(f"{arm_id}|check")
        orphans = row.get("orphan_calls") or {}
        if any(orphans.values()):
            print(f"    NOTE: calls that could not be attributed: {orphans}")

        header = (f'  {"variant":<{VAR_COL}}{"pixels":>10}{"wall s":>9}'
                  f'{"pass s":>9}{"bands":>7}{"bodies":>8}{"view s":>9}'
                  f'{"body s":>9}{"build s":>9}{"kernel s":>9}{"resid s":>9}'
                  f'{"accum s":>9}{"reduce s":>9}')
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name in variant_names():
            records = _timed(row, name)
            if not records:
                continue
            parts = _parts_on_busiest(row, name) or {}
            first = records[0]
            view_s = _mean([_busiest(r)[1].get("view_range_dev_s")
                            for r in records])
            body_s = _mean([_busiest(r)[1].get("body_dev_s") for r in records])
            print(f'  {name:<{VAR_COL}}{first.get("num_pixels", 0):>10}'
                  f'{_fmt(_variant_mean(row, name, lambda r: r.get("wall_s")), 9, "f", 2)}'
                  f'{_fmt(_variant_mean(row, name, lambda r: r.get("pass_host_s")), 9, "f", 2)}'
                  f'{first.get("bands_expected", 0):>7}'
                  f'{first.get("n_body_records", 0):>8}'
                  f'{_fmt(view_s, 9, "f", 2)}'
                  f'{_fmt(body_s, 9, "f", 2)}'
                  f'{_fmt(parts.get("builders"), 9, "f", 2)}'
                  f'{_fmt(parts.get("kernel"), 9, "f", 2)}'
                  f'{_fmt(parts.get("residual"), 9, "f", 2)}'
                  f'{_fmt(parts.get("accum"), 9, "f", 2)}'
                  f'{_fmt(parts.get("reduce"), 9, "f", 2)}')
        print("  " + "-" * (len(header) - 2))
        print("  every seconds column above is the BUSIEST device's total for "
              "one call, averaged over the timed repeats.  view s is the whole "
              "of that device's view-range time; body s is its body calls; "
              "build s, kernel s and resid s split body s; accum s is view s "
              "minus body s, which is the loop adding each body's block into "
              "its accumulator")

        # The per-device detail, the censuses and the bands, for full_p1.
        reference = _timed(row, "full_p1")
        if reference:
            record = reference[0]
            print("\n  full_p1, first timed repeat, per device:")
            for device in sorted(record["per_device"]):
                entry = record["per_device"][device]
                print(f'    {device:<10} view {entry.get("view_range_dev_s", 0.0):7.2f}s  '
                      f'body {entry.get("body_dev_s", 0.0):7.2f}s  '
                      f'build {entry.get("builders_dev_s", 0.0):6.2f}s  '
                      f'kernel {entry.get("kernel_dev_s", 0.0):7.2f}s  '
                      f'resid {entry.get("residual_dev_s", 0.0):7.2f}s  '
                      f'accum {entry.get("accum_dev_s", 0.0):6.2f}s  '
                      f'reduce {entry.get("reduce_dev_s", 0.0):6.2f}s  '
                      f'({entry.get("view_calls", 0)} view calls, '
                      f'{entry.get("body_calls", 0)} body calls, '
                      f'{entry.get("reduce_calls", 0)} reduces)')
            census = record.get("view_batch_census") or {}
            print(f'  realized view batches: '
                  f'{ {k: census[k] for k in sorted(census, key=int)} }')
            grids = record.get("grid_census") or {}
            print(f'  launch grids: {grids if grids else "none recorded"}')
            print(f'  {"band":>5}{"owner":>10}{"start":>8}{"slices":>8}'
                  f'{"max worker s":>14}{"reduce s":>10}{"partials":>10}')
            for band in record.get("bands") or []:
                print(f'  {band["band_index"]:>5}{str(band.get("owner")):>10}'
                      f'{str(band.get("slice_start")):>8}'
                      f'{str(band.get("band_slices")):>8}'
                      f'{_fmt(band.get("max_worker_view_s"), 14, "f", 3)}'
                      f'{_fmt(band.get("reduce_dev_s"), 10, "f", 3)}'
                      f'{str(band.get("n_partials")):>10}')

        if row.get("microbench"):
            bench = row["microbench"]
            print(f'\n  microbench, the channel-major copy on device 0: shape '
                  f'{bench["shape"]}, median {bench["median_s"] * 1e3:.3f} ms, '
                  f'{bench.get("effective_gb_per_s") or 0:.1f} GB/s counting '
                  "the read and the write.  That is one body call's copy; "
                  "multiply by the body calls in a column above to compare it "
                  "with resid s")

        # Instrument health for this arm.
        for record in row.get("repeats") or []:
            if not record.get("accounting_ok"):
                print(f'    RECORD CHECK FAIL on {record["variant"]} repeat '
                      f'{record["repeat"]}: {record.get("accounting_note")}')
                broken.append(f'{arm_id}|{record["variant"]}|accounting')
            if record.get("event_cap_hit"):
                print(f'    EVENT BUDGET EXHAUSTED on {record["variant"]} '
                      f'repeat {record["repeat"]}: some brackets recorded no '
                      "device time")
                broken.append(f'{arm_id}|{record["variant"]}|events')
        if row.get("recon_shape_ok") is False:
            print(f'    NOTE: recon shape {row.get("recon_shape")} is not the '
                  f'registered {row.get("recon_shape_expected")}; a geometry '
                  "default has moved.  Recorded, not failed")
        if row.get("sinogram_fresh"):
            print("    NOTE: this arm made its own sinogram; mg19's staged one "
                  "was not present.  The back projection's time depends on the "
                  "sinogram's shape, not its values")

    # ── the attribution, three devices against four ──────────────────────────
    low = by_arm.get(f"n{ARM_DEVICE_COUNTS[0]}")
    high = by_arm.get(f"n{ARM_DEVICE_COUNTS[1]}")
    if low and high:
        n_low, n_high = ARM_DEVICE_COUNTS
        print(f"\n===== attribution: {n_low} devices against {n_high} =====")
        print("Each number is the busiest device's total for one call, averaged "
              "over the timed repeats.  The ratio is the four-device number "
              "over the three-device one, so a ratio above 1 is a part that "
              "grew when a device was added.")
        for variant in variant_names():
            a = _parts_on_busiest(low, variant)
            b = _parts_on_busiest(high, variant)
            if not a or not b:
                continue
            print(f"\n  {variant}")
            print(f'    {"part":<12}{f"n={n_low} s":>12}{f"n={n_high} s":>12}'
                  f'{"ratio":>9}')
            for key, label in (("kernel", "kernel"), ("builders", "builders"),
                               ("residual", "residual"), ("accum", "accum"),
                               ("reduce", "reduce"),
                               ("device_total", "device total"),
                               ("gap", "gap"), ("wall", "pass wall")):
                print(f'    {label:<12}{_fmt(a[key], 12, "f", 2)}'
                      f'{_fmt(b[key], 12, "f", 2)}'
                      f'{_fmt(_ratio(b[key], a[key]), 9, "f", 2)}')
        print("\n  residual is one body call's time minus its builders and its "
              "kernel.  It is mostly the channel-major copy the body writes "
              "before every launch, which is sized by the detector and does "
              "not shrink with the band.  gap is the pass's host wall minus "
              "the busiest device's total, so it holds whatever the pass spent "
              "outside these brackets")

        # The band-halving discriminator.
        for row, count in ((low, ARM_DEVICE_COUNTS[0]),
                           (high, ARM_DEVICE_COUNTS[1])):
            base = _parts_on_busiest(row, "full_p1")
            half = _parts_on_busiest(row, "band_half")
            if not base or not half:
                continue
            ratio = _ratio(half["device_total"], base["device_total"])
            if ratio is None:
                continue
            if ratio >= 1.6:
                reading = ("halving the band nearly doubled the cost, so "
                           "per-band-call costs dominate")
            elif ratio <= 1.2:
                reading = ("halving the band barely moved the cost, so the "
                           "work inside a band dominates")
            else:
                reading = ("halving the band moved the cost part way, so both "
                           "kinds of cost matter here")
            print(f"\n  band halving at {count} devices: band_half over "
                  f"full_p1 is {ratio:.2f} on the busiest device.  {reading}")

        # The reconstruction estimate, beside mg19's measurement.
        print("\n===== a reconstruction's back projection, estimated from "
              "these calls =====")
        print("  T(full_p2) + sum over g in "
              f"{list(SUBSET_GRANULARITIES)} of g * T(sub_g).  A "
              "three-iteration reconstruction makes one Hessian call on every "
              "pixel and then g calls of about P/g pixels at each of those "
              "three granularities.")
        for row, count in ((low, ARM_DEVICE_COUNTS[0]),
                           (high, ARM_DEVICE_COUNTS[1])):
            parts = _parts_on_busiest(row, "full_p2")
            if not parts or parts["device_total"] is None:
                continue
            total = parts["device_total"]
            pieces = [f'hessian {total:.1f}']
            ok = True
            for g in SUBSET_GRANULARITIES:
                sub = _parts_on_busiest(row, f"sub{g}")
                if not sub or sub["device_total"] is None:
                    ok = False
                    break
                total += g * sub["device_total"]
                pieces.append(f"{g}x{sub['device_total']:.2f}")
            if not ok:
                continue
            # mg19's numbers exist for three and four real devices only.  The
            # smoke runs the same arm names at one and two virtual CPU devices,
            # so there is nothing to put beside its estimate and the line says
            # so instead of quoting a number that does not apply.
            measured = (MG19_BACK_BUSY_S.get(count)
                        if pin_for(count) == count else None)
            beside = (f"beside mg19's measured {measured:.0f} s"
                      if measured is not None else
                      "with no mg19 measurement at this device count")
            print(f"  {count} devices: {' + '.join(pieces)} = {total:.0f} s, "
                  f"{beside}")
        print(f"  mg19's numbers come from {MG19_BACK_BUSY_SOURCE}.  They are "
              "CROSS-RUN CONTEXT, not a gate: they were measured in a different "
              "job, on a different day, inside a real reconstruction, and the "
              "subsets here have the right size and no spatial structure.  "
              "Nothing in this run compares against them automatically")

        # The cross-count value witness.
        print("\n===== value witness, three devices against four =====")
        print("  Report only, with no gate.  The expectation is agreement in "
              "the 1e-6 relative class.")
        for variant in variant_names():
            a = _timed(low, variant)
            b = _timed(high, variant)
            if not a or not b:
                continue
            wa = (a[0].get("witness") or {}).get("abs_sum")
            wb = (b[0].get("witness") or {}).get("abs_sum")
            qa = (a[0].get("witness") or {}).get("sq_sum")
            qb = (b[0].get("witness") or {}).get("sq_sum")
            rel_abs = (abs(wb - wa) / abs(wa)) if (wa not in (None, 0)
                                                   and wb is not None) else None
            rel_sq = (abs(qb - qa) / abs(qa)) if (qa not in (None, 0)
                                                  and qb is not None) else None
            print(f'  {variant:<{VAR_COL}} sum|x| rel {_fmt(rel_abs, 11, "e", 3)}'
                  f'   sum x^2 rel {_fmt(rel_sq, 11, "e", 3)}')
    else:
        print("\nboth arms are needed for the attribution table, and at least "
              "one did not produce a row")

    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}.  A "
              "throttled device makes every timing in this run invalid")
        findings.append("gpu-hot")
    healthy = not broken
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"} '
          f"({len(broken)} check(s) failed).  It is NOT the verdict.  What the "
          "parts add up to, whether halving the band doubled the cost, and how "
          "the estimate compares with mg19 are all read by a person from the "
          "tables above")
    return dict(healthy=healthy, broken=broken, findings=findings, hot=hot,
                arms=len(rows))


# ── the smoke's own checks ────────────────────────────────────────────────────
def smoke_checks(rows):
    """What the local smoke asserts, beyond the health accounting.

    The smoke exercises the harness, not the physics.  Three things are checked
    here because they are cheap to check at a tiny size and expensive to
    discover on a cluster.
    """
    problems = []
    if not rows:
        problems.append("no rows were written")
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            problems.append(f"{arm_id}: the arm failed: "
                            f'{str(row["error"]).splitlines()[-1][:120]}')
            continue
        records = row.get("repeats") or []
        if not records:
            problems.append(f"{arm_id}: the arm wrote no repeat records")
        for record in records:
            if not record.get("accounting_ok"):
                problems.append(f'{arm_id}: {record["variant"]} repeat '
                                f'{record["repeat"]}: '
                                f'{record.get("accounting_note")}')
        installed = row.get("probe_installed") or {}
        if installed.get("triton_module") is True:
            if not installed.get("body"):
                problems.append(f"{arm_id}: the triton module imported but the "
                                "body wrapper was not installed")
            elif installed.get("body_attributes") != ["_view_batch_cost",
                                                      "_mbirtorch_no_compile"]:
                problems.append(f"{arm_id}: the body wrapper does not carry "
                                "both required attributes")
        else:
            print(f'  smoke note: {arm_id} could not import the triton module '
                  f'({installed.get("triton_module")}), so the body-wrapper '
                  "check is skipped")
    if problems:
        print("\nSMOKE CHECKS FAILED:")
        for problem in problems:
            print(f"  {problem}")
    else:
        print("\nsmoke checks passed: rows written, every pass accounted for, "
              "and the body wrapper carries both required attributes")
    return not problems


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_cfg = json.loads(sys.argv[2])
        try:
            out = run_arm(worker_cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif SMOKE and not DRY:
        smoke_plan = build_plan()
        smoke_rows = [_spawn(cfg) for cfg in smoke_plan]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        smoke_path = os.path.join(
            RESULTS_DIR, f"mg21_back_attrib_smoke_{RUN_LABEL}_{stamp}.jsonl")
        with open(smoke_path, "w") as smoke_sink:
            for smoke_row in smoke_rows:
                smoke_sink.write(json.dumps(smoke_row) + "\n")
        smoke_summary = summarize(smoke_rows, smoke_path)
        smoke_ok = smoke_checks(smoke_rows)
        print(f"\nwrote {smoke_path}")
        sys.exit(0 if (smoke_summary["healthy"] and smoke_ok) else 2)
    else:
        sys.exit(main())
