"""mg19 -- THE FIRST 2048-CLASS RECONSTRUCTIONS, cone and parallel, on four
H100s.

WHY THIS RUN EXISTS.

A 2048-class reconstruction is the low end of production, and nothing has ever
been measured at that size.  What exists is a closed-form capacity table
(plans/torch_port/active/two_k_design.md), built by evaluating the library's own
memory model in the production configuration.  It says a 2048-class run does not
fit on one or two H100s, fits on three with about 1.9 GiB to spare, and fits
comfortably on four.  A model calibrated at 512 and 1024 is being asked to
predict a problem four times larger in every dimension, so the table's own
section says the first composed 2048-class runs are what validate it.

This run is those runs.  It carries four readings at once, because the job is
expensive and the arms share their staged inputs:

  * THE MEMORY VALIDATION.  Three-iteration reconstructions at three devices
    (the model's edge) and at four (the operating point), taken in the ledger's
    calibration mode, which reports the modeled peak beside the measured one
    per device.  A two-device arm asserts the table's refusal for real.
  * THE BATCH SWEEP.  The forward projection gathers pixel columns in batches of
    8192 by default.  The 1024-class sweep never bracketed an optimum, so this
    one runs 8192, 16384, 32768 and 65536 at production scale.
  * THE COMBINING SLAB.  The back projection's cross-device sum streams in
    64 MiB slabs.  Two cone arms move that to 16 MiB and 256 MiB, which is the
    slab-size measurement the combining-step ruling attached to this job.
  * THE BACK PROJECTION'S SHARE OF TIME.  Every composed arm brackets each
    forward and back funnel call with CUDA events, per device, so the split
    between the two directions is read at this scale for the first time.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It changes no library file
and flips no default.  The two arms that change the slab size do it by setting a
module attribute inside their own subprocess, before anything is built, and they
record the value they set.

TERMS USED BELOW, defined once here:
    arm            one measured configuration -- one geometry, one device count,
                   one pixel batch, one slab size -- run in its own fresh
                   process.
    cell           the sinogram shape, (views, detector rows, detector
                   channels).
    recon shape    the reconstruction volume, (rows, columns, slices).
    settle         the call that makes the model choose and hold its device
                   layout, ``model._apply_device_policy()``.
    the preflight  the memory check inside the settle.  It refuses a layout
                   whose modeled demand does not fit the device's free memory.
    calibration    the library mode, switched on by an environment variable,
                   in which a reconstruction reports its modeled peak against
                   its measured peak for each device.
    composed arm   an arm that runs a three-iteration reconstruction.
    busy time      the time a device spent inside forward or back projection
                   calls, measured with CUDA event pairs around each call.

THE CELL.  One per geometry, the reference problem the capacity table was
computed for:

    sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
    recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask

Cone uses a full turn of views with the source-to-detector distance at four
detector widths and the source-to-isocenter distance at two.  Parallel uses half
a turn, which is a full parallel-beam scan.  The realized recon shape and pixel
count are recorded on every row and compared with the values above; a mismatch
is recorded, not failed, because a moved geometry default is worth knowing about
but does not make a memory reading wrong.

THE ARTIFACTS ARE MEMORY MAPPED, and that is not a detail.  The phantom is
29.6 GiB and the sinogram 30.5 GiB.  Both are written with
``numpy.lib.format.open_memmap`` and read back with ``mmap_mode='r'``, so
neither is ever held whole in host memory as a second copy.  The phantom is
drawn straight into its map in slabs of rows: one call to numpy's legacy
generator for the whole volume would build a float64 array of 59 GiB before
casting it, and that array has no reason to exist.  numpy fills any request in C
order from a single stream, so a sequence of slabs holds exactly the values one
call would have produced; the smoke checks that at a size where checking is
free.

THE ARMS, per geometry, in the order the job takes them.  Cone runs its whole
block first, then parallel.

     1  gen              n=4.  Stages the phantom, projects it, stages the
                         sinogram.  Every later arm of this geometry reads both.
                         The calibration mode is OFF here.
     2  refuse_n2        n=2.  Settles and EXPECTS the preflight to refuse.
                         Nothing else runs.
     3  n3_b8192         n=3, calibration on.  The model's edge.
     4  n4_b8192         n=4, calibration on.  The operating point.
     5  n4_b8192_repeat  an exact repeat of arm 4.
     6  n4_b16384        n=4, pixel batch 16384.
     7  n4_b32768        n=4, pixel batch 32768.
     8  n4_b65536        n=4, pixel batch 65536.
     9  n4_slab16        n=4, combining slab 16 MiB.   CONE ONLY.
    10  n4_slab256       n=4, combining slab 256 MiB.  CONE ONLY.

Ten cone arms and eight parallel arms, eighteen in all.

WHY ARM 5 EXISTS.  Arms 6 through 10 are read as differences against arm 4, and
a difference is meaningless without a floor.  Arms 4 and 5 are the same
configuration run twice, so the spread between them is how much two identical
arms differ on this machine at this size.  A batch difference or a slab
difference smaller than that spread is the machine, not the knob.

ONE THING THE WALL CLOCK CARRIES AND THE BUSY TIME DOES NOT.  Generated kernels
are cached on disk, and the cache survives between arms, so the first arm to
reach a given call shape pays a compilation the later ones do not.  Arm 4 is
usually that first arm and arm 5 is not, which inflates the pair's spread.  The
summary therefore prints the forward busy seconds beside every batch wall: the
pixel batch is a forward-projection knob, busy time is measured on the device,
and host-side compilation barely touches it.  Read the two lines together.

WHAT ARM 2 MEANS.  The capacity table says two devices cannot hold this problem,
by 19 GiB against a 78.7 GiB budget.  The preflight is the code that acts on
that arithmetic, so the arm that tests it is the arm that tries to settle at two
devices and expects to be told no.  A refusal with a message is the arm's PASS,
and the message tail is recorded.  A settle that goes through instead is the
interesting outcome: the ledger and the budget are recorded and the summary says
so loudly, because it would mean the table is wrong in the direction that
matters.

WHAT ARM 3 MEANS, AND WHAT HAPPENS IF IT REFUSES.  Three devices is where the
table's slack is thinnest -- 1.9 GiB out of 78.7 -- so a refusal there is a real
result about the model's edge and not a harness fault.  It is recorded as such.
Only then does the arm rebuild the same model with ``skip_memory_preflight``
set and run once more, because the measured peak is the number the whole memory
validation wants and a refusal would otherwise leave it unmeasured.  The skip is
recorded on the row, so no reader can mistake a forced run for a clean one.

THE VALUES GATE, TAKEN BEFORE EACH RECONSTRUCTION.  A memory reading from an
arm that computes the wrong answer is not a memory reading.  Every calibration
arm therefore forward-projects the staged phantom on its own device layout and
compares against the staged sinogram, which is that phantom projected at four
devices by arm 1:

    rel = max|out - ref| / max|ref|      computed in float64, walked in slabs

The gate is 1e-4 and the expectation is the e-6 class or better.  Both
geometries bind hand-written Triton kernels for the projection bodies on an
H100, and a Triton body is not recompiled per shape the way a compiled torch
body is, so the reduction-order latitude that puts torch bodies in the 6e-4
class does not apply here -- which matters because the three-device split of
2048 views is uneven (683, 683, 682).  An arm whose gate fails records the
failure and SKIPS its reconstruction.

HOW THE MEMORY READING IS TAKEN.  The calibration mode is switched on in the
arm's own subprocess environment.  In that mode the reconstruction resets the
per-device peak counters as it begins and, when it ends, reports one row per
device: the modeled peak, the measured peak, and their ratio.  Those rows are
this run's deliverable.  They are read straight off the model rather than
scraped from a log.

The values pass above happens BEFORE the reconstruction and its output is
released before the reconstruction starts, so nothing it allocated is inside the
measured peak.

HOW THE TIME IS SPLIT BETWEEN THE DIRECTIONS.  Every composed arm wraps the two
projection funnels on its own model instance.  Each call records one CUDA event
pair per placement device, created and recorded inside
``with torch.cuda.device(dev)`` so the events sit on that device's stream.  The
elapsed times are read only after every device has been synchronized at the end
of the reconstruction.  Nothing is synchronized inside the loop: a synchronize
per call would serialize the very overlap this is measuring.  The per-device
forward and back busy seconds and the call counts are report-only fields.

FINGERPRINTS INSTEAD OF STAGED VOLUMES.  A reconstruction here is 29.6 GiB and
there are up to eight of them per geometry, which is more disk than the readings
are worth.  Each composed arm instead records the volume's L2 norm and the
values at 65,536 seeded sample positions, drawn once from the flattened volume
and identical across arms.  The samples are staged as a small file with a
checksum, and each arm compares its samples against every earlier arm's.  Those
comparisons are report-only and have no gate: three iterations of a nonlinear
optimizer amplify float-level differences, and no threshold measured at smaller
sizes transfers here.

WHAT IS RECORDED ON EVERY ROW, beyond the numbers above:
  * the realized device list and whether its length is the arm's pin;
  * which projection directions run as general torch code, read from
    ``_memory_ledger.torch_body_directions``.  On an H100 both geometries bind
    hand-written kernels in both directions, so this must be EMPTY.  On the CPU
    smoke the kernels do not exist and it must be ('forward', 'back').  Each row
    says which environment it expected, so the two cannot be confused;
  * the per-device block lengths on the view and slice axes, and whether the
    device count divides either axis exactly;
  * the pixel batch the model reports it will use;
  * the memory ledger for the settled layout;
  * the calibration environment variable, asserted PRESENT on the calibration
    arms and ABSENT on the generator;
  * a GPU health sample, so a thermally throttled node is visible;
  * the environment the arm ran under.

ARTIFACTS, under MG19_RESULTS, each with a checksum written beside it and
VERIFIED ON EVERY READ.  A truncated file on a shared parallel filesystem is a
recorded failure mode of this work, and a comparison against a file that changed
underneath the run would be a quietly wrong answer rather than a loud one.

    mg19_<label>_<geometry>_phantom.npy    the seeded phantom
    mg19_<label>_<geometry>_sinogram.npy   that phantom projected at n=4
    mg19_<label>_<geometry>_fp_<arm>.npy   one arm's 65,536 sample values

The two large files are about 120 GiB per full run and are NOT deleted: a
re-run of a subset through MG19_ARMS depends on them.  Remove them by hand.

HOW THE READINGS ARE JUDGED.  Calibration ratios are read against the 1.00 to
1.30 band the library declares.  Below 1.00 is printed as UNDER, which is the
direction the ledger may not err in: an under-prediction lets a doomed run
start, and preventing that is what the ledger is for.  Above 1.30 is printed as
over, which wastes devices but breaks nothing.  Both are findings.  Batch and
slab differences are read against the arm 4 to arm 5 spread; a difference inside
that spread means the knob did not matter at this scale, and the summary says it
that way rather than reporting a number as though it were a result.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
ran and was recorded, every artifact was staged, every checksum verified, and no
arm ran on the wrong device count, bound the wrong kind of projection body, ran
under the wrong calibration setting, or used a pixel batch other than its own.
It is NOT the verdict.  A values gate failure, a preflight refusal, a missing
refusal, and a calibration ratio outside the band are all printed in full and
all leave the exit code alone.  A person reads the verdicts from the table this
job prints.

THE LOCAL SMOKE.  MG19_SMOKE=1 runs the same arm plan on a (32, 24, 20) cell at
one and two virtual CPU devices, pinned by explicit device list, with one
iteration, the shipped batch only, and no slab arms.  It exercises the harness,
not the physics.  Two things genuinely cannot happen there and are recorded
rather than failed: a CPU layout has no memory budget, so the two-device arm
cannot be refused and says ``refusal_not_applicable``; and the measured side of
a calibration row is a CUDA-only counter, so the calibration table is empty and
says why.

Run:
    <torch python> mg19_two_k_baselines.py        on a 4-GPU node
    MG19_DRY=1 <python> mg19_two_k_baselines.py   print the arm plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized token
is an error, not a silent skip.
    MG19_RESULTS=<dir>              where the jsonl and the artifacts go
    MG19_GEOMS=cone,parallel        subset of the geometries
    MG19_ARMS=n4_b8192,n4_b16384    subset of the arms, by arm name
    MG19_DRY=1                      print the arm plan and exit
    MG19_SMOKE=1                    the local CPU smoke
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG19_SMOKE", "0") == "1"
DRY = os.environ.get("MG19_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

GEOMETRIES = ("cone", "parallel")

# The reference problem of the capacity table.  The recon shape and pixel count
# are what the geometries produce from this cell today (checked 2026-08-17 in
# this checkout); they are registered so a moved default is visible on the row.
CELL = (2048, 2016, 1984)
RECON_SHAPE = (1984, 1984, 2016)
NUM_PIXELS = 3088364
# The smoke's cell.  Both geometries give (20, 20, 24) and 276 pixels here.
SMOKE_CELL = (32, 24, 20)
SMOKE_RECON_SHAPE = (20, 20, 24)
SMOKE_NUM_PIXELS = 276
CELL_LABEL = "smoke" if SMOKE else "2k"

# The shipped pixel batch (tomography_model.FORWARD_PIXEL_BATCH).  Arms at this
# value do NOT set the attribute -- they run the shipped configuration, and the
# row asserts that the model really reports 8192.  The sweep arms set it.
SHIPPED_BATCH = 8192
BATCH_SWEEP = () if SMOKE else (16384, 32768, 65536)
# The combining slab, in MiB, for the cone-only rider.  The shipped value is 64.
SLAB_SWEEP_MIB = () if SMOKE else (16, 256)
SHIPPED_SLAB_MIB = 64

# Device pins.  The production pins are the table's counts; the smoke has no
# CUDA devices, so it runs the same arm plan at one and two virtual CPU devices
# and each row records which mechanism pinned it.
SMOKE_PINS = {2: 2, 3: 1, 4: 2}

VCD_ITERATIONS = 1 if SMOKE else 3
VCD_SEED = 12345                 # the seed every other run in this series uses
PHANTOM_SEED = 20260817          # this run's phantom, fixed here and nowhere else

# The values gate, and the class of number to expect beside it.  Both geometries
# bind hand-written Triton kernels on an H100.  A Triton body is not regenerated
# per call shape, so the reduction-order latitude that puts a COMPILED TORCH
# body in the 6e-4 class at an uneven split does not reach these arms, and the
# uneven three-device view split (683, 683, 682) is not a reason to loosen this.
VALUES_GATE_REL = 1e-4
VALUES_EXPECTATION = (
    "near machine zero on CPU" if SMOKE else
    "1e-6 class or better: Triton bodies, not compiled torch bodies")

# The band the library declares for a hand-written-kernel reconstruction
# (_memory_ledger.CALIBRATION_BAND).  Repeated here so the summary can print a
# verdict without importing torch in the parent process.
CAL_BAND = (1.00, 1.30)

# The reconstruction fingerprint: an L2 norm plus values at seeded positions.
FINGERPRINT_SAMPLES = 65536
FINGERPRINT_SEED = 7

# Rows of the phantom drawn per call.  Eight rows of the production volume is a
# 256 MiB float64 draw; the whole volume in one call would be 59 GiB of float64.
PHANTOM_SLAB_ROWS = 8
# Bytes of float64 promotion the chunked comparison and the norm are allowed to
# hold at once.
COMPARE_BUDGET_BYTES = 64 << 20

# The event-pair ceiling for the per-call bracket.  A three-iteration
# reconstruction makes on the order of a hundred funnel calls, so this is not a
# limit in practice; it exists so a partition sequence that grew could not fill
# memory with events, and the row says whether it was hit.
MAX_EVENT_PAIRS = 20000

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
    "MG19_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 26                  # wide enough for the longest arm id printed
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a repeat
    before.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return set(allowed)
    chosen = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"{env_name}: {token!r} is not one of "
                             f"{sorted(allowed)}")
        chosen.add(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return chosen


def cell():
    return SMOKE_CELL if SMOKE else CELL


def recon_shape():
    return SMOKE_RECON_SHAPE if SMOKE else RECON_SHAPE


def num_pixels():
    return SMOKE_NUM_PIXELS if SMOKE else NUM_PIXELS


def pin_for(n_dev):
    """The device count an arm actually runs at.

    The production pin is the arm's own count.  The smoke has no CUDA devices,
    so its arms run at one or two virtual CPU devices; the arm names keep their
    production meaning and the row records both numbers.
    """
    return SMOKE_PINS[n_dev] if SMOKE else n_dev


def arm_specs(geometry):
    """Every arm of one geometry, in the order the run takes them.

    ``pixel_batch`` None means the arm does not touch the batch attribute and so
    runs the shipped configuration; ``batch`` is the value the arm's name claims
    either way, and every row checks the model against it.
    """
    specs = [
        dict(arm="gen", kind="gen", n_dev=4, pixel_batch=None,
             batch=SHIPPED_BATCH, calibration=False, slab_mib=None),
        dict(arm="refuse_n2", kind="refuse", n_dev=2, pixel_batch=None,
             batch=SHIPPED_BATCH, calibration=False, slab_mib=None),
        dict(arm="n3_b8192", kind="composed", n_dev=3, pixel_batch=None,
             batch=SHIPPED_BATCH, calibration=True, slab_mib=None),
        dict(arm="n4_b8192", kind="composed", n_dev=4, pixel_batch=None,
             batch=SHIPPED_BATCH, calibration=True, slab_mib=None),
        dict(arm="n4_b8192_repeat", kind="composed", n_dev=4, pixel_batch=None,
             batch=SHIPPED_BATCH, calibration=True, slab_mib=None),
    ]
    for batch in BATCH_SWEEP:
        specs.append(dict(arm=f"n4_b{batch}", kind="composed", n_dev=4,
                          pixel_batch=batch, batch=batch, calibration=True,
                          slab_mib=None))
    if geometry == "cone":
        # The slab rider is cone only: one geometry answers the question, and a
        # second copy of it would cost twenty minutes of GPU time for nothing.
        for slab in SLAB_SWEEP_MIB:
            specs.append(dict(arm=f"n4_slab{slab}", kind="composed", n_dev=4,
                              pixel_batch=None, batch=SHIPPED_BATCH,
                              calibration=True, slab_mib=slab))
    return specs


def all_arm_names():
    names = []
    for geometry in GEOMETRIES:
        for spec in arm_specs(geometry):
            if spec["arm"] not in names:
                names.append(spec["arm"])
    return names


# ── artifact paths and checksums ──────────────────────────────────────────────
# The cell label is in every file name, so a smoke run and a production run can
# share a results directory without either reading the other's bytes.
def _phantom_path(geometry):
    return os.path.join(RESULTS_DIR,
                        f"mg19_{CELL_LABEL}_{geometry}_phantom.npy")


def _sinogram_path(geometry):
    return os.path.join(RESULTS_DIR,
                        f"mg19_{CELL_LABEL}_{geometry}_sinogram.npy")


def _fingerprint_path(geometry, arm):
    return os.path.join(RESULTS_DIR,
                        f"mg19_{CELL_LABEL}_{geometry}_fp_{arm}.npy")


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


def _verified_load(path):
    """Memory-map a staged artifact after checking its checksum.

    Every read goes through here, and nothing here loads a large array into host
    memory: the two big artifacts are 30 GiB each and the arms that use them are
    already holding a reconstruction.
    """
    import numpy as np

    with open(_md5_path(path)) as handle:
        expected = handle.read().strip()
    actual = _md5(path)
    if actual != expected:
        raise RuntimeError(f"staged artifact checksum mismatch at {path}: "
                           f"{actual} != {expected}")
    return np.load(path, mmap_mode="r"), actual


def _stage_memmap(path, shape, fill, row_step):
    """Write one large artifact through a memory map, then checksum it.

    ``fill(out, start, stop)`` writes rows [start, stop) of the map.  The file is
    built under a temporary name and renamed at the end, so a killed job cannot
    leave a half-written array in the place a later run would read from.
    """
    import numpy as np
    from numpy.lib.format import open_memmap

    os.makedirs(RESULTS_DIR, exist_ok=True)
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


def _stage_small(path, array):
    """Write one small artifact (a fingerprint sample vector) and checksum it."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(path, np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
    return _write_md5(path)


def _to_numpy(x):
    """The ONE host exit.

    A sharded array's ``gather()`` ALREADY returns a numpy array; calling
    ``.detach()`` on that result is a recorded failure that once cost a whole
    multi-device run its rows.
    """
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


def compare_arrays(out, ref, gate=None, budget_bytes=COMPARE_BUDGET_BYTES):
    """max|out - ref| / max|ref| in float64, with an L2-relative distance beside
    it.

    Walked in slabs along the first axis, so neither the float32 arrays nor
    their float64 promotions are ever held whole: one float64 copy of a
    2048-class sinogram is 61 GiB.  The maximum is accumulated slab by slab,
    which is exact -- a maximum of maxima is the maximum, with none of the
    summation error a norm carries.  Either array may be memory-mapped.

    ``gate`` None means the comparison is reported and not judged.
    """
    import numpy as np

    if tuple(out.shape) != tuple(ref.shape):
        return dict(ok=False, rel=None, gate=gate,
                    reason=f"shape {list(out.shape)} is not the reference's "
                           f"{list(ref.shape)}")
    row_bytes = max(1, int(np.prod(ref.shape[1:])) * 8)
    step = max(1, int(budget_bytes // row_bytes))
    max_diff, max_ref, sq_diff, sq_ref = 0.0, 0.0, 0.0, 0.0
    for start in range(0, ref.shape[0], step):
        a = np.asarray(ref[start:start + step], dtype=np.float64)
        b = np.asarray(out[start:start + step], dtype=np.float64)
        diff = b - a
        max_ref = max(max_ref, float(np.max(np.abs(a))))
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
        sq_diff += float(np.sum(diff * diff))
        sq_ref += float(np.sum(a * a))
    if max_ref <= 0.0:
        return dict(ok=False, rel=None, gate=gate,
                    reason="the reference is all zeros, so a relative "
                           "comparison has no denominator")
    rel = max_diff / max_ref
    return dict(ok=(None if gate is None else rel <= gate), rel=rel, gate=gate,
                max_abs_diff=max_diff, max_abs_ref=max_ref,
                shape=list(ref.shape),
                l2_rel=((sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else None))


# ── the GPU health sample ─────────────────────────────────────────────────────
# A GPU that is thermally throttled produces a valid memory reading but an
# invalid timing one, and this run reads both.  A hot node usually also means a
# neighbour job is sharing the hardware, which the memory readings care about
# even more than the timings do.
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
def _build_model(geometry, pin_devices=None):
    """Build the model for one geometry at this run's cell.

    This is mg15's construction, unchanged, so a row here and a row there
    describe the same model.  On CUDA nothing is configured: the device count is
    pinned through the environment, which leaves the model on the automatic
    branch where the memory ledger is actually consumed and where the preflight
    can refuse.  ``pin_devices`` is the CPU smoke's path only, because the
    automatic search short-circuits when fewer than two CUDA devices are
    visible.
    """
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    num_views, channels = shape[0], shape[2]
    if geometry == "cone":
        # A full turn of views, and the source distances written as multiples of
        # the detector width so the same expression builds the smoke model:
        # source-to-detector at four widths, source-to-isocenter at two, which
        # puts the object halfway between the source and the detector at a
        # magnification of two.
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            shape, angles, source_detector_dist=4.0 * channels,
            source_iso_dist=2.0 * channels)
    else:
        # Half a turn is a full parallel-beam scan: the second half repeats the
        # first, reflected.
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(shape, angles)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _patch_slab(slab_mib):
    """Move the combining slab for this subprocess only.

    THE PATCH POINT WAS VERIFIED by reading both users in this checkout
    (2026-08-17).  ``_sharding.sum_band_to_owner``, which is the code that
    streams the cross-device sum, and ``_memory_ledger.band_reduce``, which is
    the term that prices it, both reach the slab size only by calling
    ``_sharding.reduce_slab_rows``; that function reads the module global
    ``_sharding.REDUCE_SLAB_BYTES`` at call time.  So this single assignment
    moves what the code does AND what the ledger charges for it, which is what
    makes the arm's calibration row describe the arm.
    """
    from mbirtorch import _sharding

    before = int(_sharding.REDUCE_SLAB_BYTES)
    _sharding.REDUCE_SLAB_BYTES = int(slab_mib) * 2 ** 20
    return dict(slab_bytes_before=before,
                slab_bytes_after=int(_sharding.REDUCE_SLAB_BYTES),
                slab_mib=int(slab_mib))


def _shape_check(model):
    """Did the geometry defaults produce the recon shape this file registered?

    Recorded rather than raised: a moved default is worth knowing about, but it
    does not make a memory reading wrong.
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

    At the production cell three devices divide neither axis: 2048 views split
    683/683/682 and 2016 slices split 672/672/672 -- the slice axis does divide.
    Recorded on every row so the reader does not have to re-derive it.
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


def _ledger_record(model, devices):
    """The modeled peak per device for the settled layout.

    ``last_memory_ledger`` holds the ledger the device policy actually used.
    Under the calibration mode the policy always leaves one, and every
    calibration arm here is on that path; a model that somehow left none has one
    rebuilt from the same two library functions the policy calls, so the numbers
    are the library's own either way.
    """
    from mbirtorch import _memory_ledger

    ledger = getattr(model, "last_memory_ledger", None)
    source = "model.last_memory_ledger (built by the device policy)"
    if ledger is None:
        ledger = _memory_ledger.estimate_peak_device_bytes(
            _memory_ledger.plan_from_model(model, devices, workload="recon"))
        source = "rebuilt here via plan_from_model + estimate_peak_device_bytes"
    return dict(source=source, devices=[str(d) for d in ledger.devices],
                modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
                dominant_phase=[ledger.dominant_phase(i).name
                                for i in range(len(ledger.devices))],
                num_pixels_full=int(ledger.num_pixels_full))


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


def _seeded_phantom_into(out, start, stop, cols, slices):
    """Fill rows [start, stop) of the phantom map.

    Called in order from a stream seeded once before the first call, so the map
    ends up holding exactly what a single whole-volume draw would have produced
    -- numpy's legacy generator fills any request in C order from one stream.
    The single call is what this avoids: at the production shape it would build
    a 59 GiB float64 array before casting it to float32.
    """
    import numpy as np

    out[start:stop] = np.random.rand(stop - start, cols,
                                     slices).astype(np.float32)


def _ensure_phantom(geometry, shape):
    """Stage this geometry's phantom if it is not already staged."""
    import numpy as np

    path = _phantom_path(geometry)
    if _staged(path):
        _verified_load(path)                    # verify, then drop the map
        return path, _md5(path), False
    rows, cols, slices = (int(s) for s in shape)
    np.random.seed(PHANTOM_SEED)
    digest = _stage_memmap(
        path, (rows, cols, slices),
        lambda out, a, b: _seeded_phantom_into(out, a, b, cols, slices),
        PHANTOM_SLAB_ROWS)
    if SMOKE:
        # The slab draw against the single call, at a size where checking both
        # forms is free.  If this ever fails, the stream-order assumption above
        # is wrong and every phantom in this series is a different array than
        # its seed says.
        np.random.seed(PHANTOM_SEED)
        direct = np.random.rand(rows, cols, slices).astype(np.float32)
        staged, _digest = _verified_load(path)
        if not np.array_equal(direct, np.asarray(staged)):
            raise RuntimeError("the slab-wise phantom draw does not reproduce "
                               "the single-call draw; the stream-order "
                               "assumption in _seeded_phantom_into is wrong")
        del direct, staged
    return path, digest, True


def _volume_fingerprint(volume):
    """The reconstruction's fingerprint: an L2 norm and seeded sample values.

    The norm is accumulated in float64 over chunks rather than taken in one
    call, because a single-precision reduction over eight billion values loses
    digits and a whole float64 promotion of the volume is 63 GiB.  The sample
    positions come from one seeded draw over the flattened volume, so every arm
    of this run samples the same positions and two arms' vectors can be compared
    element for element.  They are sorted, which changes nothing about which
    positions were drawn and makes the gather sequential.
    """
    import numpy as np

    flat = volume.reshape(-1)
    total = int(flat.shape[0])
    step = max(1, COMPARE_BUDGET_BYTES // 8)
    squares = 0.0
    for start in range(0, total, step):
        chunk = np.asarray(flat[start:start + step], dtype=np.float64)
        squares += float(np.sum(chunk * chunk))
    count = min(FINGERPRINT_SAMPLES, total)
    idx = np.sort(np.random.default_rng(FINGERPRINT_SEED).choice(
        total, size=count, replace=False))
    samples = np.asarray(flat[idx], dtype=np.float32)
    return dict(l2_norm=squares ** 0.5, elements=total,
                sample_count=int(count),
                sample_seed=FINGERPRINT_SEED), samples


# ── the per-call bracket on the two projection funnels ─────────────────────────
class FunnelInstrument:
    """CUDA event pairs around every forward and back projection call.

    This is mg9's region instrument, reduced to the two funnels this run reads.
    One pair per placement device is created and recorded INSIDE
    ``with torch.cuda.device(dev)``, which is what puts the markers on the
    stream that carries that device's kernels; the funnel itself is called
    outside that context, so it runs exactly as it runs without the instrument.

    NOTHING IS SYNCHRONIZED INSIDE THE LOOP.  An event record places a marker
    and no work, so the calls still overlap across devices the way they do
    untouched.  A synchronize per call would serialize precisely the overlap
    being measured, and the elapsed times would then describe the instrument.
    Every elapsed time is read in :meth:`finish`, after a per-device
    synchronize, once the reconstruction is over.

    On the CPU smoke there are no CUDA events, so the host wall of each call
    stands in and the row says which backend produced the numbers.
    """

    REGIONS = ("forward", "back")

    def __init__(self, torch_module, cuda):
        self.torch = torch_module
        self.cuda = cuda
        self.calls = {region: 0 for region in self.REGIONS}
        self.host_s = {region: 0.0 for region in self.REGIONS}
        self._pairs = {region: {} for region in self.REGIONS}
        self._cpu_ms = {region: {} for region in self.REGIONS}
        self.devices_seen = {region: [] for region in self.REGIONS}
        self.pair_count = 0
        self.cap_hit = False
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"

    def _start(self, region, devices):
        for device in devices:
            name = str(device)
            if name not in self.devices_seen[region]:
                self.devices_seen[region].append(name)
        if not self.cuda:
            return None
        if self.pair_count + len(devices) > MAX_EVENT_PAIRS:
            self.cap_hit = True
            return None
        events = []
        for device in devices:
            with self.torch.cuda.device(device):
                start = self.torch.cuda.Event(enable_timing=True)
                start.record()
            events.append((device, start))
        return events

    def _stop(self, region, events):
        if events is None:
            return
        for device, start in events:
            with self.torch.cuda.device(device):
                end = self.torch.cuda.Event(enable_timing=True)
                end.record()
            self._pairs[region].setdefault(str(device), []).append((start, end))
            self.pair_count += 1

    def wrap(self, region, resolve_devices, func):
        """Return ``func`` bracketed.  ``resolve_devices`` is called per
        invocation so a mid-run change of layout cannot leave the instrument on
        a stale device list.  The wrapper returns the funnel's own value
        unchanged, so the reconstruction engine cannot tell it is there."""
        def wrapped(*args, **kwargs):
            devices = resolve_devices()
            events = self._start(region, devices)
            host0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                self._stop(region, events)
                self.calls[region] += 1
                self.host_s[region] += host
                if not self.cuda:
                    for device in devices:
                        name = str(device)
                        self._cpu_ms[region][name] = (
                            self._cpu_ms[region].get(name, 0.0) + host * 1e3)
        return wrapped

    def finish(self, devices):
        """Per-device synchronize, THEN read the spans.  Never inside the loop.

        Returns the jsonl-ready record: per-device busy seconds and call counts
        for each direction.
        """
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        record = {}
        for region in self.REGIONS:
            per_device = {}
            if self.cuda:
                for name, pairs in self._pairs[region].items():
                    per_device[name] = sum(s.elapsed_time(e)
                                           for s, e in pairs) / 1e3
            else:
                for name, total_ms in self._cpu_ms[region].items():
                    per_device[name] = float(total_ms) / 1e3
            ordered = [str(d) for d in devices]
            busy = [float(per_device.get(name, 0.0)) for name in ordered]
            record[region] = dict(
                busy_s_per_device=busy,
                busy_s_max=(max(busy) if busy else 0.0),
                busy_s_sum=float(sum(busy)),
                calls=self.calls[region],
                host_wall_s=self.host_s[region],
                devices=ordered)
        return dict(regions=record, event_backend=self.backend,
                    event_pairs=self.pair_count, event_cap_hit=self.cap_hit)


def attach_funnel_instrument(model, torch_module, cuda):
    """Bracket the two projection funnels on THIS model instance.

    Nothing in the mbirtorch package is edited: the funnels are shadowed as
    instance attributes, and the reconstruction engine looks both up on the
    instance at call time.  The forward funnel's output side is the sinogram
    placement and the back funnel's is the reconstruction placement, so each
    direction's events are created on the devices that direction writes to.

    Those two placements hold the SAME device list -- the model builds both from
    one list and shards them on different axes -- so the per-device readings of
    the two directions line up index for index, and one synchronize over the
    sinogram placement covers both.
    """
    instrument = FunnelInstrument(torch_module, cuda)
    model.sparse_forward_project = instrument.wrap(
        "forward", lambda: list(model.sino_placement.devices),
        model.sparse_forward_project)
    model.sparse_back_project = instrument.wrap(
        "back", lambda: list(model.recon_placement.devices),
        model.sparse_back_project)
    return instrument


# ── the worker: one arm, one process ──────────────────────────────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the arm does."""
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  cell=list(cell()), cell_label=CELL_LABEL,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  realized_pin=pin_for(cfg["n_dev"]),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  phantom_seed=PHANTOM_SEED,
                  values_gate=VALUES_GATE_REL,
                  values_expectation=VALUES_EXPECTATION,
                  shipped_slab_mib=SHIPPED_SLAB_MIB,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_column_gather=os.environ.get(
                      "MBIRTORCH_FORWARD_COLUMN_GATHER"))
    # The calibration mode owns the per-device peak counters.  It must be on for
    # every arm whose deliverable is a calibration row, and off for the
    # generator, which has no memory reading to make and should not reset
    # anything.
    present = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1"
    result["calibration_env_present"] = present
    result["calibration_env_ok"] = (present == bool(cfg.get("calibration")))
    result["invalid_reasons"] = []
    # Checked here rather than after the settle, because it is knowable before
    # anything runs and an arm whose settle refuses would otherwise never have
    # it checked at all.
    if not result["calibration_env_ok"]:
        result["invalid_reasons"].append(
            "MBIRTORCH_MEMORY_CALIBRATION is "
            f'{"set" if present else "absent"} on an arm that wanted it '
            f'{"set" if cfg.get("calibration") else "absent"}')
    return result, cuda


def _settle_and_witness(model, cfg, result, cuda):
    """Settle the layout and record everything that says what this arm is.

    THE SETTLE IS LOAD-BEARING, and this is mg15's lesson.  On CUDA an arm pins
    its device count through MBIRTORCH_NUM_DEVICES, that pin acts only through
    the model's device policy, and ``forward_project`` never calls the policy.
    A freshly built automatic model still holds the trivial single-device
    placement, so a projection taken before the layout has settled would run on
    ONE device at every device count and compare its result against a sinogram
    made the same way -- passing while gating nothing.  The pixel batch and the
    slab size are both in place before this call as well, because the memory
    ledger prices the path the arm is about to run.

    Returns True when the layout settled, False when the preflight refused; the
    refusal's message is on the row either way.
    """
    from mbirtorch import _memory_ledger

    n_dev = pin_for(cfg["n_dev"])
    try:
        model._apply_device_policy()
    except _memory_ledger.MemoryPreflightError as exc:
        result["refused_by_preflight"] = True
        result["preflight_message"] = str(exc)[-3000:]
        result["realized_devices"] = [str(d)
                                      for d in model.sino_placement.devices]
        return False
    result["refused_by_preflight"] = False

    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))
    result["preflight_skipped"] = bool(
        getattr(model, "skip_memory_preflight", False))

    # WHICH BODIES ARE BOUND.  On an H100 both cone and parallel bind
    # hand-written Triton kernels in both directions, so the list of directions
    # running as general torch code must be EMPTY.  On the CPU smoke the kernels
    # do not exist and both directions fall back, so it must be both.  The row
    # says which environment it expected, because the two answers are opposites
    # and a row without that note could be read either way.
    directions = tuple(_memory_ledger.torch_body_directions(model))
    expected = () if cuda else ("forward", "back")
    result["torch_body_directions"] = list(directions)
    result["torch_body_expected"] = list(expected)
    result["torch_body_expectation_basis"] = (
        "H100: hand-written Triton kernels bind in both directions, so no "
        "direction runs as general torch code" if cuda else
        "CPU smoke: the Triton kernels are unavailable, so both directions "
        "fall back to general torch code")
    result["torch_bodies_ok"] = (directions == expected)

    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__

    result["column_gather_forward"] = bool(model._column_gather_forward())
    realized_batch = int(model._forward_pixel_batch())
    result["forward_pixel_batch"] = realized_batch
    result["forward_pixel_batch_attr"] = getattr(
        model, "forward_project_pixel_batch", None)
    result["batch_ok"] = (realized_batch == int(cfg["batch"]))

    result.update(_shape_check(model))
    result["blocks"] = _block_lengths(model)
    result["ledger"] = _ledger_record(model, model.sino_placement.devices)

    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"realized {realized} for a pin of {n_dev}")
    if not result["torch_bodies_ok"]:
        result["invalid_reasons"].append(
            f"torch_body_directions is {list(directions)}, not "
            f"{list(expected)}; this arm is not running the bodies it claims")
    if not result["batch_ok"]:
        result["invalid_reasons"].append(
            f'the model reports pixel batch {realized_batch}, but this arm is '
            f'named for {cfg["batch"]}')
    return True


def run_gen(cfg, result, cuda):
    """Arm 1: stage the phantom, project it at four devices, stage the sinogram.

    Every later arm of this geometry reads both files, so this arm is the one
    whose failure costs the geometry.  It runs with the calibration mode OFF: it
    makes no memory reading and has no business resetting the peak counters.
    """
    import numpy as np

    geometry = cfg["geometry"]
    shape = tuple(recon_shape())
    path, digest, written = _ensure_phantom(geometry, shape)
    result.update(phantom_path=path, phantom_md5=digest,
                  phantom_written=written)

    model = _build_model(geometry, pin_devices=(None if cuda else
                                                [DEVICE] * pin_for(cfg["n_dev"])))
    if not _settle_and_witness(model, cfg, result, cuda):
        return
    phantom, phantom_md5 = _verified_load(path)
    result["phantom_md5_verified"] = phantom_md5

    sino_path = _sinogram_path(geometry)
    if _staged(sino_path):
        # Already there from an earlier run of this job.  Verify it and say the
        # forward projection was not repeated, rather than spend ten minutes
        # rewriting the same bytes.
        _verified_load(sino_path)
        result.update(sinogram_path=sino_path, sinogram_md5=_md5(sino_path),
                      sinogram_written=False,
                      sinogram_note="already staged and verified; this arm did "
                                    "not repeat the forward projection")
        return

    start = time.perf_counter()
    projected = _to_numpy(model.forward_project(phantom))
    _sync(model, cuda)
    result["forward_s"] = time.perf_counter() - start
    result["projection_devices"] = [str(d)
                                    for d in model.sino_placement.devices]
    result["projection_n_devices"] = len(result["projection_devices"])
    result["projection_devices_ok"] = (
        result["projection_n_devices"] == pin_for(cfg["n_dev"]))
    if not result["projection_devices_ok"]:
        result["invalid_reasons"].append(
            f'the staging projection ran on {result["projection_devices"]}, '
            f'not on the arm\'s {pin_for(cfg["n_dev"])} devices')

    # Written through a map in row slabs, so the 30.5 GiB result is never
    # duplicated in host memory just to reach the disk.
    def copy_views(out, start, stop):
        out[start:stop] = np.asarray(projected[start:stop], dtype=np.float32)

    row_bytes = max(1, int(np.prod(projected.shape[1:])) * 4)
    result["sinogram_md5"] = _stage_memmap(
        sino_path, projected.shape, copy_views,
        max(1, (256 << 20) // row_bytes))
    result["sinogram_path"] = sino_path
    result["sinogram_written"] = True
    result["sinogram_shape"] = list(projected.shape)
    del projected
    _release(cuda)


def run_refuse(cfg, result, cuda):
    """Arm 2: the two-device verdict, tested for real.

    The capacity table says two devices cannot hold this problem.  The preflight
    is the code that acts on that arithmetic, so this arm settles at two devices
    and expects to be told no.  A refusal WITH A MESSAGE is the pass.  A settle
    that goes through is the finding: the ledger and the per-device budget are
    recorded so a reader can see by how much the table was wrong.

    No projection and no reconstruction run here under either outcome.
    """
    from mbirtorch import _memory_ledger

    model = _build_model(cfg["geometry"],
                         pin_devices=(None if cuda else
                                      [DEVICE] * pin_for(cfg["n_dev"])))
    if not cuda:
        # An explicit CPU device list is not the automatic branch, and a CPU
        # device has no memory budget to check against, so the preflight cannot
        # run at all here.  That is a property of the smoke, not a result.
        result["refusal_not_applicable"] = True
        result["refusal_note"] = (
            "the CPU smoke pins by explicit device list and a CPU device "
            "reports no memory budget, so no preflight runs and no refusal is "
            "possible; this arm checks only that the harness reaches it")
        _settle_and_witness(model, cfg, result, cuda)
        result["refusal_ok"] = True
        return

    result["refusal_not_applicable"] = False
    settled = _settle_and_witness(model, cfg, result, cuda)
    if not settled:
        result["refusal_ok"] = True
        result["refusal_verdict"] = (
            "the preflight refused, which is what the capacity table predicts "
            "for two devices at this size")
        return
    # It settled.  Record what it settled on and what the device could hold, so
    # the size of the surprise is on the row and not only in a printed line.
    result["refusal_ok"] = False
    result["refusal_verdict"] = (
        "the preflight did NOT refuse two devices; the capacity table says it "
        "should have, by about 19 GiB")
    budgets = []
    for device in model.sino_placement.devices:
        budgets.append(_memory_ledger.device_budget_bytes(device))
    result["device_budget_bytes"] = [None if b is None else int(b)
                                     for b in budgets]


def run_composed(cfg, result, cuda):
    """Arms 3 through 10: one calibrated three-iteration reconstruction.

    The order inside this function is the measurement's design, so it is stated
    once here and followed exactly below:

      1. patch the combining slab, if this arm moves it, before anything exists
         that could have read the old value;
      2. build the model and set the pixel batch, both before the settle, so the
         memory ledger prices the path this arm will run;
      3. settle, and on a refusal record the model-edge finding and repeat once
         with the preflight skipped;
      4. project the staged phantom and gate it against the staged sinogram at
         1e-4, then release the result, so no allocation of the values pass is
         inside the measured peak;
      5. seed, arm the per-call bracket, and reconstruct;
      6. read the calibration rows, the wall, the busy times, and the
         fingerprint.

    The wall covers the reconstruction call exactly as the earlier runs in this
    series measured it, which includes gathering the finished volume to the host
    at the end.  That is a few seconds of several minutes at this size, and it
    is the same few seconds in every arm, so differences between arms are not
    made of it -- and keeping the definition means these walls can be read
    beside the 1024-class walls the job's estimate came from.
    """
    import numpy as np
    import torch

    geometry = cfg["geometry"]
    if cfg.get("slab_mib"):
        result["slab"] = _patch_slab(cfg["slab_mib"])

    def build():
        model = _build_model(geometry,
                             pin_devices=(None if cuda else
                                          [DEVICE] * pin_for(cfg["n_dev"])))
        if cfg.get("pixel_batch"):
            # Set on this model INSTANCE and nowhere else, and before the
            # settle: the ledger reads this while pricing the layout, so a batch
            # applied afterwards would be measured against a plan for a
            # different batch.
            model.forward_project_pixel_batch = int(cfg["pixel_batch"])
        return model

    model = build()
    settled = _settle_and_witness(model, cfg, result, cuda)
    if not settled:
        # A refusal here is a reading about the model's edge, not a fault, and
        # it is recorded as one BEFORE anything is forced.  Then the same arm
        # runs once more with the preflight skipped, because the measured peak
        # is the number the memory validation exists to obtain and a refusal
        # would otherwise leave it unmeasured.
        result["preflight_refused_at_settle"] = True
        result["preflight_refusal_message"] = result.pop("preflight_message",
                                                         None)
        result["model_edge_finding"] = (
            f'the preflight refused {pin_for(cfg["n_dev"])} devices at this '
            "size; the capacity table calls this count feasible, so the "
            "refusal is a finding about the model's edge.  The arm was then "
            "repeated once with skip_memory_preflight set, to obtain the "
            "measured peak")
        model = build()
        model.skip_memory_preflight = True
        if not _settle_and_witness(model, cfg, result, cuda):
            result["preflight_refused_twice"] = True
            result["timing_skipped_reason"] = (
                "the preflight refused even with skip_memory_preflight set; "
                "nothing was measured")
            return
    result.setdefault("preflight_refused_at_settle", False)

    if result["invalid_reasons"]:
        result["timing_skipped_reason"] = "; ".join(result["invalid_reasons"])
        return

    # ── the values leg, before any timing ────────────────────────────────────
    phantom, result["phantom_md5"] = _verified_load(_phantom_path(geometry))
    start = time.perf_counter()
    projected = _to_numpy(model.forward_project(phantom))
    _sync(model, cuda)
    result["values_forward_s"] = time.perf_counter() - start
    result["projection_devices"] = [str(d)
                                    for d in model.sino_placement.devices]
    sinogram, result["sinogram_md5"] = _verified_load(_sinogram_path(geometry))
    result["values"] = compare_arrays(projected, sinogram, VALUES_GATE_REL)
    del projected, phantom
    _release(cuda)
    if not result["values"].get("ok"):
        result["values_failed"] = True
        result["timing_skipped_reason"] = (
            f'the values gate failed at rel {result["values"].get("rel")} '
            f"against a gate of {VALUES_GATE_REL}; a memory reading from an arm "
            "that computes a different sinogram would measure nothing")
        return

    # ── the reconstruction ───────────────────────────────────────────────────
    instrument = attach_funnel_instrument(model, torch, cuda)
    # The seed goes here, immediately before the reconstruction and inside this
    # process, so every arm draws the same pixel partitions and the comparisons
    # between arms are between device counts, batches and slabs -- not between
    # partition sequences.
    np.random.seed(VCD_SEED)
    _sync(model, cuda)
    start = time.perf_counter()
    volume, _info = model.recon(sinogram, max_iterations=VCD_ITERATIONS,
                                stop_threshold_change_pct=0.0,
                                logfile_path=None, print_logs=False)
    _sync(model, cuda)
    result["recon_s"] = time.perf_counter() - start
    result["instrument"] = instrument.finish(model.sino_placement.devices)
    del sinogram

    # THE DELIVERABLE.  Read off the model rather than scraped from a log.
    result["calibration"] = [
        dict(device=str(device), modeled_bytes=int(modeled),
             measured_bytes=int(measured), ratio=float(ratio))
        for device, modeled, measured, ratio
        in (model.last_memory_calibration or [])]
    result["calibration_band"] = list(CAL_BAND)
    if not result["calibration"]:
        result["calibration_skipped_reason"] = (
            "the measured side is torch.cuda.max_memory_allocated, which is "
            f"CUDA-only, and this arm ran on {DEVICE}; the modeled column in "
            "the ledger field is still real, the measured column does not "
            "exist here")
        if cuda:
            result["invalid_reasons"].append(
                "a CUDA calibration arm produced no calibration rows, so the "
                "reading this arm exists for is missing")

    volume = np.ascontiguousarray(np.asarray(_to_numpy(volume),
                                             dtype=np.float32))
    result["recon_volume_shape"] = list(volume.shape)
    fingerprint, samples = _volume_fingerprint(volume)
    result["fingerprint"] = fingerprint
    del volume
    _release(cuda)

    fp_path = _fingerprint_path(geometry, cfg["arm"])
    result["fingerprint_path"] = fp_path
    result["fingerprint_md5"] = _stage_small(fp_path, samples)

    # Every composed arm of this geometry that has already staged samples.
    # Report-only: three iterations of a nonlinear optimizer amplify
    # float-level differences, and no threshold measured at a smaller size
    # transfers to this one.
    cross = []
    for other in arm_specs(geometry):
        if other["kind"] != "composed" or other["arm"] == cfg["arm"]:
            continue
        other_path = _fingerprint_path(geometry, other["arm"])
        if not _staged(other_path):
            continue
        other_samples, other_md5 = _verified_load(other_path)
        entry = compare_arrays(samples, other_samples)
        entry.update(other_arm=other["arm"], other_n_dev=other["n_dev"],
                     other_batch=other["batch"],
                     other_slab_mib=other["slab_mib"], other_md5=other_md5)
        cross.append(entry)
        del other_samples
    result["cross_arm"] = cross


def run_arm(cfg):
    """One arm, in its own process.

    A fresh process per arm is not tidiness.  Compiled and Triton bodies are
    cached at module level for the life of a process, the peak memory counters
    are per process, and the combining-slab arms patch a module attribute -- all
    three would leak from one arm into the next if they shared an interpreter.
    """
    result, cuda = _base_result(cfg)
    health = [sample_gpu_health()]
    try:
        if cfg["kind"] == "gen":
            run_gen(cfg, result, cuda)
        elif cfg["kind"] == "refuse":
            run_refuse(cfg, result, cuda)
        else:
            run_composed(cfg, result, cuda)
    finally:
        health.append(sample_gpu_health())
        result["gpu_health"] = [g for snap in health for g in snap]
        result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited.

    Three variables are popped first and then set: the device pin, the
    calibration mode, and the process-wide column-gather override.  A value
    exported by the submitting shell therefore cannot reach an arm that did not
    ask for it -- in particular the generator, which must not own the peak
    counters, and the sweep arms, which must not have their driver overruled.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(pin_for(cfg["n_dev"]))
    if cfg.get("calibration"):
        env["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        # An arm that runs out of device memory lands here.  That is a reading,
        # not a harness fault, so it is recorded as a row and the run continues
        # with the next arm.
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def build_plan():
    """The plan in job order: cone's whole block, then parallel's, so a job that
    runs out of wall time still holds whole geometries."""
    geometries = _strict_subset("MG19_GEOMS", set(GEOMETRIES))
    keep_arms = _strict_subset("MG19_ARMS", set(all_arm_names()))
    plan = []
    for geometry in GEOMETRIES:
        if geometry not in geometries:
            continue
        arms = [a for a in arm_specs(geometry) if a["arm"] in keep_arms]
        if not arms:
            continue
        if not any(a["arm"] == "gen" for a in arms) and not (
                _staged(_phantom_path(geometry))
                and _staged(_sinogram_path(geometry))):
            # The arms selected do not include the one that stages this
            # geometry's inputs, and the inputs are not on disk.  Say so here
            # rather than fail every arm on a missing file.
            raise ValueError(
                f"MG19_ARMS excludes 'gen' for {geometry}, but its phantom and "
                f"sinogram are not staged under {RESULTS_DIR}")
        for aspec in arms:
            plan.append(dict(geometry=geometry,
                             arm_id=f'{geometry}_{aspec["arm"]}', **aspec))
    if not plan:
        raise ValueError("MG19_GEOMS and MG19_ARMS together select no arm")
    return plan


def _dry_run(plan):
    shape = tuple(cell())
    volume = tuple(recon_shape())
    sino_gib = shape[0] * shape[1] * shape[2] * 4 / 2 ** 30
    recon_gib = volume[0] * volume[1] * volume[2] * 4 / 2 ** 30
    print(f"mg19 the first 2048-class reconstructions: {len(plan)} arms, "
          f"device {DEVICE}, {VCD_ITERATIONS} iteration(s) per composed arm")
    print(f"  results and artifacts -> {RESULTS_DIR}")
    print(f"  sinogram {shape} {sino_gib:.2f} GiB -> recon {volume} "
          f"{recon_gib:.2f} GiB, {num_pixels()} pixels inside the mask")
    print(f"  values gate {VALUES_GATE_REL:.0e}, expected {VALUES_EXPECTATION}")
    print(f"  calibration ratios read against {CAL_BAND[0]:.2f}-"
          f"{CAL_BAND[1]:.2f}; below the floor is UNDER, the direction the "
          "ledger may not err in")
    print(f'  {"arm":<{ARM_COL}}{"pin":>5}{"batch":>8}{"slab MiB":>10}'
          f'{"calib":>7}  what it does')
    what = dict(gen="stages the phantom and the sinogram every later arm reads",
                refuse="settles and expects the preflight to refuse",
                composed="one calibrated reconstruction, bracketed per call")
    for cfg in plan:
        print(f'  {cfg["arm_id"]:<{ARM_COL}}{pin_for(cfg["n_dev"]):>5}'
              f'{cfg["batch"]:>8}{(cfg["slab_mib"] or "-"):>10}'
              f'{("on" if cfg["calibration"] else "off"):>7}  '
              f'{what[cfg["kind"]]}')
    print("no library file is edited: the pixel batch is set on the model "
          "instance, and the two slab arms patch a module attribute inside "
          "their own subprocess")


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg19_baselines_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg19 2048-class baselines on {RUN_LABEL} ({DEVICE}); "
          f"{len(plan)} arms -> {out_path}", flush=True)
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
    """One table cell, with a missing value padded to the same width as a
    present one, so the columns line up whether an arm produced a number or
    not."""
    if value is None:
        return f'{"-":>{width}}'
    return f"{value:>{width}.{prec}{kind}}"


def _busy(row, region, field="busy_s_max"):
    regions = ((row.get("instrument") or {}).get("regions") or {})
    entry = regions.get(region) or {}
    return entry.get(field)


def _cal_ratios(row):
    return [c["ratio"] for c in (row.get("calibration") or [])]


def summarize(rows, out_path):
    """The table a person reads the verdicts from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and this function keeps them apart.  A values
    gate failure, a preflight refusal, a missing refusal and a calibration ratio
    outside the band are FINDINGS: they are printed in full and none of them
    touches the exit code.  An arm that ran on the wrong device count, bound the
    wrong kind of body, used a pixel batch other than its own, ran under the
    wrong calibration setting, or produced no row at all is an instrument
    failure, because it did not measure what the plan said it would.
    """
    print(f"\n===== mg19 the first 2048-class reconstructions ({out_path}) "
          "=====")
    broken, findings = [], []
    by_arm = {}

    header = (f'{"arm":<{ARM_COL}}{"pin":>4}{"batch":>7}{"dev":>5}'
              f'{"values rel":>12}{"recon s":>10}{"cal min":>9}{"cal max":>9}'
              f'{"fwd busy":>10}{"back busy":>10}{"back share":>11}')
    print(header)
    print("-" * len(header))
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'{arm_id:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            broken.append(f"{arm_id}|error")
            continue
        by_arm[arm_id] = row
        ratios = _cal_ratios(row)
        fwd, back = _busy(row, "forward"), _busy(row, "back")
        share = (back / (fwd + back)) if (fwd is not None and back is not None
                                          and (fwd + back) > 0) else None
        values = row.get("values") or {}
        print(f'{arm_id:<{ARM_COL}}{pin_for(row["n_dev"]):>4}'
              f'{row.get("forward_pixel_batch", row["batch"]):>7}'
              f'{row.get("realized_n_devices", "-"):>5}'
              f'{_fmt(values.get("rel"), 12, "e", 3)}'
              f'{_fmt(row.get("recon_s"), 10, "f", 1)}'
              f'{_fmt(min(ratios) if ratios else None, 9, "f", 3)}'
              f'{_fmt(max(ratios) if ratios else None, 9, "f", 3)}'
              f'{_fmt(fwd, 10, "f", 2)}{_fmt(back, 10, "f", 2)}'
              f'{_fmt(share, 11, "f", 3)}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    ARM CHECK FAIL: {reason}")
            broken.append(f"{arm_id}|check")
        if row.get("kind") == "refuse":
            if row.get("refusal_not_applicable"):
                print(f'    refusal not applicable: {row.get("refusal_note")}')
            elif row.get("refusal_ok"):
                message = (row.get("preflight_message") or "").strip()
                tail = message.splitlines()[-1][:100] if message else ""
                print(f'    REFUSED, which is this arm\'s PASS.  '
                      f"message tail: {tail}")
            else:
                print(f'    *** FINDING: {row.get("refusal_verdict")}.  '
                      f'ledger '
                      f'{[round(b / 2 ** 30, 1) for b in (row.get("ledger") or {}).get("modeled_peak_bytes", [])]}'
                      f' GiB against budgets '
                      f'{[None if b is None else round(b / 2 ** 30, 1) for b in row.get("device_budget_bytes") or []]}'
                      " GiB.  Read this before anything else in this table")
                findings.append(f"{arm_id}|no-refusal")
        if row.get("kind") == "gen" and row.get("refused_by_preflight"):
            # The generator stages what every later arm of its geometry reads,
            # so a refusal here costs the whole geometry and must not be a
            # quiet line in a jsonl.
            message = (row.get("preflight_message") or "").strip()
            print("    *** THE GENERATOR WAS REFUSED BY THE PREFLIGHT, so this "
                  "geometry has no phantom and no sinogram and every later arm "
                  "of it will fail on a missing file.  message tail: "
                  f'{message.splitlines()[-1][:100] if message else ""}')
            findings.append(f"{arm_id}|gen-refusal")
        if row.get("preflight_refused_at_settle"):
            print(f'    *** MODEL-EDGE FINDING: '
                  f'{row.get("model_edge_finding")}')
            findings.append(f"{arm_id}|edge-refusal")
        if row.get("preflight_skipped"):
            print("    preflight SKIPPED on this arm: the peak below is a "
                  "forced run, not a clean one")
        if row.get("values_failed"):
            print(f'    *** VALUES GATE FAILED at rel '
                  f'{values.get("rel"):.3e} against {VALUES_GATE_REL:.0e}; the '
                  "reconstruction was skipped.  This is a FINDING, not an "
                  "instrument fault")
            findings.append(f"{arm_id}|values")
        for entry in row.get("calibration") or []:
            if entry["ratio"] < CAL_BAND[0]:
                print(f'    *** UNDER on {entry["device"]}: modeled '
                      f'{entry["modeled_bytes"] / 2 ** 30:.2f} GiB against '
                      f'measured {entry["measured_bytes"] / 2 ** 30:.2f} GiB, '
                      f'ratio {entry["ratio"]:.3f}.  This is the direction the '
                      "ledger may not err in")
                findings.append(f'{arm_id}|under|{entry["device"]}')
            elif entry["ratio"] > CAL_BAND[1]:
                print(f'    over on {entry["device"]}: ratio '
                      f'{entry["ratio"]:.3f} above {CAL_BAND[1]:.2f}; the '
                      "ledger over-charges here, which wastes devices and "
                      "breaks nothing")
                findings.append(f'{arm_id}|over|{entry["device"]}')
        if row.get("calibration_skipped_reason") and not row.get("calibration"):
            print(f'    no calibration table: '
                  f'{row["calibration_skipped_reason"]}')
        if row.get("recon_shape_ok") is False:
            print(f'    NOTE: recon shape {row.get("recon_shape")} is not the '
                  f'registered {row.get("recon_shape_expected")}; a geometry '
                  "default has moved.  Recorded, not failed")
        if row.get("slab"):
            print(f'    combining slab moved from '
                  f'{row["slab"]["slab_bytes_before"] / 2 ** 20:.0f} MiB to '
                  f'{row["slab"]["slab_bytes_after"] / 2 ** 20:.0f} MiB for '
                  "this subprocess")
        for entry in row.get("cross_arm") or []:
            print(f'    vs {entry.get("other_arm"):<18} samples: max-rel '
                  f'{_fmt(entry.get("rel"), 0, "e", 3)}, L2-rel '
                  f'{_fmt(entry.get("l2_rel"), 0, "e", 3)}')
    print("-" * len(header))
    print("fwd busy / back busy are the BUSIEST DEVICE's seconds inside the "
          "per-call brackets for each direction; the per-device lists and the "
          "sums are on the rows.  back share = back / (fwd + back) from those "
          "same two numbers")

    # ── the readings, per geometry ───────────────────────────────────────────
    for geometry in GEOMETRIES:
        arms = {a["arm"]: by_arm.get(f'{geometry}_{a["arm"]}')
                for a in arm_specs(geometry)}
        if not any(arms.values()):
            continue
        print(f"\n--- {geometry} ---")

        base = arms.get("n4_b8192") or {}
        repeat = arms.get("n4_b8192_repeat") or {}
        w1, w2 = base.get("recon_s"), repeat.get("recon_s")
        spread = abs(w2 - w1) if (w1 is not None and w2 is not None) else None

        # The batch sweep, in one line, against the repeat pair's spread.  The
        # forward busy time is printed beside each wall because the pixel batch
        # is a forward-projection knob: it is the number the batch should move,
        # and it is measured on the device, so it carries far less of the
        # host-side compilation than the wall does.
        walls, busies = [], []
        for spec in arm_specs(geometry):
            if spec["kind"] != "composed" or spec["slab_mib"] or \
                    spec["n_dev"] != 4:
                continue
            row = arms.get(spec["arm"]) or {}
            wall = row.get("recon_s")
            if wall is None:
                continue
            walls.append(f'{spec["arm"]}:{wall:.1f}s')
            fwd = _busy(row, "forward")
            if fwd is not None:
                busies.append(f'{spec["arm"]}:{fwd:.2f}s')
        if walls:
            print(f'  batch sweep, recon walls   {"  ".join(walls)}')
            if busies:
                print(f'  batch sweep, forward busy  {"  ".join(busies)}')
            if spread is not None:
                print(f"  the ruler: the two shipped-batch arms ran {w1:.1f}s "
                      f"and {w2:.1f}s, {spread:.1f}s apart.  A batch "
                      "difference smaller than that is the machine, not the "
                      "batch")
                print("  read that ruler knowing the compiled and generated "
                      "kernels are cached on disk across arms, so the FIRST "
                      "arm at a new call shape pays a compilation the later "
                      "ones do not.  The forward busy line above is measured "
                      "on the device and carries much less of that cost")
            else:
                print("  the repeat pair did not complete, so the batch walls "
                      "have no spread to be read against")

        # Three devices against four: the memory validation's own comparison.
        for name in ("n3_b8192", "n4_b8192"):
            row = arms.get(name) or {}
            ratios = _cal_ratios(row)
            if not ratios:
                print(f"  {name:<16} no calibration rows")
                continue
            worst = min(ratios)
            verdict = ("UNDER the floor" if worst < CAL_BAND[0] else
                       "inside the band" if max(ratios) <= CAL_BAND[1] else
                       "over the top of the band")
            modeled = [c["modeled_bytes"] / 2 ** 30 for c in row["calibration"]]
            measured = [c["measured_bytes"] / 2 ** 30
                        for c in row["calibration"]]
            print(f"  {name:<16} ratios "
                  f'{[round(r, 3) for r in ratios]}  {verdict}; modeled '
                  f'{[round(m, 1) for m in modeled]} GiB against measured '
                  f'{[round(m, 1) for m in measured]} GiB')
            if row.get("preflight_skipped"):
                print("                   (that peak came from a run with the "
                      "preflight skipped)")

        # The slab rider, cone only.
        slab_arms = [s for s in arm_specs(geometry) if s["slab_mib"]]
        if slab_arms:
            backs = [_busy(arms.get(n) or {}, "back")
                     for n in ("n4_b8192", "n4_b8192_repeat")]
            backs = [b for b in backs if b is not None]
            if len(backs) == 2:
                ruler = abs(backs[1] - backs[0])
                middle = sum(backs) / 2
                print(f"  combining slab rider, back-projection busy seconds "
                      f"on the busiest device.  The shipped "
                      f"{SHIPPED_SLAB_MIB} MiB arms read {backs[0]:.2f}s and "
                      f"{backs[1]:.2f}s, so the repeat spread is "
                      f"{ruler:.2f}s")
                for spec in slab_arms:
                    value = _busy(arms.get(spec["arm"]) or {}, "back")
                    if value is None:
                        print(f'    {spec["arm"]:<12} did not complete')
                        continue
                    delta = value - middle
                    if abs(delta) <= ruler:
                        print(f'    {spec["arm"]:<12} {value:.2f}s, '
                              f'{delta:+.2f}s from the shipped pair, which is '
                              "inside the repeat spread.  THAT IS THE READING: "
                              "the slab size does not matter at this scale")
                    else:
                        print(f'    {spec["arm"]:<12} {value:.2f}s, '
                              f'{delta:+.2f}s from the shipped pair, outside '
                              "the repeat spread, so the slab size does move "
                              "the back projection here")
            else:
                print("  combining slab rider: the shipped pair did not "
                      "complete, so the slab arms have no ruler to be read "
                      "against")

    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}.  A "
              "throttled device gives a valid memory reading and an invalid "
              "timing one")
    if findings:
        print(f"\n{len(findings)} finding(s): {findings}.  None of these "
              "changes the exit code.  They are what the run is for")
    healthy = not broken
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"} '
          f"({len(broken)} arm(s) did not measure what the plan said).  The "
          "verdicts -- whether the capacity table holds at 2048, which pixel "
          "batch to ship, and whether the combining slab matters -- are read "
          "by a person from the table above and the rows in the jsonl")
    return dict(healthy=healthy, broken=broken, findings=findings, hot=hot,
                arms=len(rows))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_cfg = json.loads(sys.argv[2])
        try:
            out = run_arm(worker_cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    else:
        sys.exit(main())
