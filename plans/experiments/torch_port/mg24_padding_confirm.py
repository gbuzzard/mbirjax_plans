"""mg24 -- THE CONFIRMATION RUNS FOR THE BAND-PADDING REMEDY, cone at the
2048 class, on the padded tree.

WHY THIS RUN EXISTS.

The cone back projection cost more on four devices than on three.  mg19
measured the composed back busy time at the 2048-class cell: 136.8 s at three
devices and 227.9 s at four.  mg21 and mg21b found the cause.  Triton compiles
a separate, faster kernel for each integer argument it can prove is a multiple
of 16.  The back kernel's band argument is the slice block each device owns:
672 slices at three devices, which is a multiple of 16, and 504 at four, which
is not.  The remedy is to round that argument up to the next multiple of 16
inside the kernel wrapper, compute the padded columns, and discard them.  The
design note is plans/torch_port/active/back_remedy_design.md; this job is its
increment 3.

This job is the acceptance evidence for that remedy.  It re-runs mg19's cone
composed arms at three and four devices on the tree that carries the padding,
and reads the back busy times against the design note's section 3 projections:

    three devices    136.8 s, and it should NOT move.  The band is 672,
                     which is already a multiple of 16, so the padding
                     changes nothing there.
    four devices     227.9 s, and it should FALL TO NEAR 100 s.  The band
                     goes from 504 to a padded 512, so the kernel returns to
                     the rate the divisible bands run at.

Those are projections from measured rates, not measurements.  Turning them
into recorded numbers is what this job does.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It changes no library file
and flips no default.  Everything it reads is the shipped configuration.

TERMS USED BELOW, defined once here:
    arm            one measured configuration -- one device count, run in its
                   own new process.
    cell           the sinogram shape, (views, detector rows, detector
                   channels).
    recon shape    the reconstruction volume, (rows, columns, slices).
    band           the slice block one device owns in the banded back
                   projection, which is the kernel's band argument.
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

THE CELL.  mg19's cone cell, unchanged, so the two runs measure the same
problem:

    sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
    recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask

The slice axis is 2016.  Three devices own 672 slices each and four own 504
each.  672 is a multiple of 16 and 504 is not, so the three-device arm should
not move and the four-device arm should.  Every row records the realized band
and its padded value, so a reader does not have to derive them.

THE ARMS, four, cone only, in the order the job takes them:

     1  gen                n=4.  VERIFIES the staged phantom and sinogram
                           against their checksums.  mg19 staged both on
                           2026-08-17 and this job reuses those files; the arm
                           regenerates only what is missing.  The calibration
                           mode is OFF here.
     2  n3_shipped         n=3, calibration on.  The divisible band, 672.
     3  n4_shipped         n=4, calibration on.  The padded band, 504 to 512.
     4  n4_shipped_repeat  an exact repeat of arm 3.

WHY ARM 4 EXISTS.  The four-device reading is the one that has to move, and a
move can only be read against how much this machine varies on its own.  Arms 3
and 4 are the same configuration run twice, so the spread between them is that
variation, measured at this size.  mg19's own pair differed by 0.31 s of back
busy time, against a projected fall of about 128 s.

THE PIXEL BATCH, AND WHY THE BACK COMPARISON SURVIVES IT.  The arms run the
shipped pixel batch, which is 32768 today.  mg19's comparison arms ran 8192.
The pixel batch is a forward-projection knob: it sets how many pixel columns
the forward gather takes at a time, and the back projection never reads it.
The back busy comparison against mg19's 136.8 s and 227.9 s is therefore a
comparison of the same work.  The forward busy time WILL differ, and the
summary prints why, with mg19's own batch sweep beside it: mg19 measured cone
forward busy at 127.90 s at four devices with a batch of 32768, against
151.58 s at 8192.  The arm names say "shipped" rather than a number, and every
row records the batch the model reports.

ONE MORE SHIPPED VALUE MOVED SINCE mg19.  The back projection's cross-device
sum streams in slabs, and that slab was 64 MiB when mg19 ran and is 256 MiB
now.  mg19 measured the slab's effect directly: its 256 MiB cone arm read
226.12 s of back busy time against 227.85 s and 228.16 s for the shipped pair.
That is a difference of about 2 s against a projected fall of about 128 s, so
the slab is far too small to explain the fall.  The summary prints it anyway,
so nobody has to wonder.  Every row records the realized slab size.

THE TREE THIS JOB REQUIRES.  Three witnesses say the checkout is the merged
tree, and a fourth says the padding is in it.  Every row carries all four, and
an arm that fails any of them is an instrument failure, because it measured a
different library than the one this job is about:

  * ``TomographyModel`` has ``_sparse_forward_project_cylinders``;
  * ``_sharding`` has ``transfer_cylinder_batch``;
  * ``_sharding`` does NOT have ``broadcast_band_to_views``;
  * ``from mbirtorch._utils import padded_kernel_width`` succeeds and
    ``padded_kernel_width(504) == 512``.

The first three are the 2026-08-17 merge that removed the banded multi-device
forward.  The fourth is this job's own witness and is new: 504 is the
four-device band, 512 is what the remedy must turn it into, and an unpadded
tree either has no such function or returns 504.

THE STALENESS NOTE IS RECORDED, NOT GATED.  The library keeps a note that says
the widening speed floors were measured against projection-cost code that has
since changed.  The padding edits the two Triton files, so that note is
non-None on this tree and that is EXPECTED.  Every row records the note's text
and the summary prints it once.  It gates nothing: the floors still govern,
and re-measuring them is a separate piece of work the design note names as the
padding's fourth increment.

THE VALUES GATE, TAKEN BEFORE EACH RECONSTRUCTION.  A timing reading from an
arm that computes the wrong answer is not a timing reading.  Every calibration
arm forward-projects the staged phantom on its own device layout and compares
against the staged sinogram:

    rel = max|out - ref| / max|ref|      computed in float64, walked in slabs

The gate is 1e-4 and the expectation is the e-6 class or better.  Cone binds a
hand-written Triton kernel for both projection bodies on an H100, and a Triton
body is not recompiled per shape the way a compiled torch body is, so the
reduction-order latitude that puts torch bodies in the 6e-4 class does not
apply here.  mg21b measured the padding's own value question directly: it
compared a 512-band launch against a 672-band launch on the slices they share,
and every comparison read exactly zero.  An arm whose gate fails records the
failure and SKIPS its reconstruction.

HOW THE MEMORY READING IS TAKEN, AND WHY THIS JOB STILL TAKES ONE.  The
calibration mode is switched on in the arm's own subprocess environment.  In
that mode the reconstruction resets the per-device peak counters as it begins
and, when it ends, reports one row per device: the modeled peak, the measured
peak, and their ratio.  The padding changed the ledger as well as the code:
the ledger now charges the PADDED band length, read from the same helper the
kernel wrapper uses.  These calibration rows are therefore how the padded
charge is checked at production scale, which is the only scale where the
charge matters.

The values pass happens BEFORE the reconstruction and its output is released
before the reconstruction starts, so nothing it allocated is inside the
measured peak.

WHAT A THREE-DEVICE REFUSAL WOULD MEAN.  Three devices is where the capacity
table's slack is thinnest, 1.9 GiB out of 78.7.  The design note's section 4
says the three-device band takes no pad at all, so the padding should not move
that verdict.  If the preflight refuses anyway, that is a real result about
the model's edge and it is recorded as one.  Only then does the arm rebuild
the same model with ``skip_memory_preflight`` set and run once more, because
the timing is the number this job exists for.  The skip is recorded on the
row, so no reader can mistake a forced run for a clean one.

HOW THE TIME IS SPLIT BETWEEN THE DIRECTIONS.  Every composed arm wraps the
two projection funnels on its own model instance.  Each call records one CUDA
event pair per placement device, created and recorded inside
``with torch.cuda.device(dev)`` so the events sit on that device's stream.
The elapsed times are read only after every device has been synchronized at
the end of the reconstruction.  Nothing is synchronized inside the loop: a
synchronize per call would serialize the very overlap this is measuring.  The
summary prints the per-device forward and back busy seconds and the call
counts for every arm.

FINGERPRINTS INSTEAD OF STAGED VOLUMES.  A reconstruction here is 29.6 GiB and
there are three of them, which is more disk than the readings are worth.  Each
composed arm records the volume's L2 norm and the values at 65,536 seeded
sample positions, drawn once from the flattened volume and identical across
arms.  Each arm compares its samples against every earlier arm's.  Those
comparisons are report-only and have no gate: three iterations of a nonlinear
optimizer amplify float-level differences, and no threshold measured at a
smaller size transfers here.  The comparison is WITHIN this run only.  mg19's
fingerprints are not compared against, because mg19's arms ran a different
pixel batch and a difference between the two runs would mix the padding with
the batch.

WHAT IS RECORDED ON EVERY ROW, beyond the numbers above:
  * the four tree witnesses and the staleness note;
  * the realized device list and whether its length is the arm's pin;
  * the realized band, its padded value, and whether the two differ;
  * which projection directions run as general torch code, read from
    ``_memory_ledger.torch_body_directions``.  On an H100 cone binds
    hand-written kernels in both directions, so this must be EMPTY.  On the
    CPU smoke the kernels do not exist and it must be ('forward', 'back').
    Each row says which environment it expected, so the two cannot be
    confused;
  * the per-device block lengths on the view and slice axes;
  * the pixel batch and the combining slab size the model will use;
  * the memory ledger for the settled layout;
  * the calibration environment variable, asserted PRESENT on the composed
    arms and ABSENT on the generator;
  * a GPU health sample, so a thermally throttled node is visible;
  * the environment the arm ran under.

ARTIFACTS, under MG24_RESULTS, each with a checksum written beside it and
VERIFIED ON EVERY READ.  A truncated file on a shared parallel filesystem is a
recorded failure mode of this work, and a comparison against a file that
changed underneath the run would be a quietly wrong answer rather than a loud
one.

    mg19_<label>_cone_phantom.npy    the seeded phantom, STAGED BY mg19
    mg19_<label>_cone_sinogram.npy   that phantom projected at n=4, BY mg19
    mg24_<label>_cone_fp_<arm>.npy   one arm's 65,536 sample values

The two large files keep mg19's names because they are mg19's files and this
job reads them rather than replacing them.  If either is missing the generator
arm rebuilds it, and the row and the summary both say so loudly: a rebuilt
sinogram comes from THIS tree, not from the tree mg19 measured, and that is
worth knowing before anyone reads the values gate as a cross-tree comparison.
The large files are about 60 GiB and are NOT deleted.  Remove them by hand.

HOW THE READINGS ARE JUDGED.  The back busy times are read against mg19's
recorded numbers and the design note's projections, and the summary prints all
three side by side.  Cross-run numbers are context, not gates: they were taken
on another day, on another node, at another pixel batch.  Calibration ratios
are read against the 1.00 to 1.30 band the library declares.  Below 1.00 is
printed as UNDER, which is the direction the ledger may not err in: an
under-prediction lets a doomed run start, and preventing that is what the
ledger is for.  Above 1.30 is printed as over, which wastes devices but breaks
nothing.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
ran and was recorded, every artifact was verified, and no arm ran on the wrong
device count, on the wrong tree, bound the wrong kind of projection body, or
ran under the wrong calibration setting.  It is NOT the verdict.  A values
gate failure, a preflight refusal, a calibration ratio outside the band, and a
back busy time that misses the projection are all printed in full and all
leave the exit code alone.  A person reads the verdicts from the table this
job prints.

THE LOCAL SMOKE.  MG24_SMOKE=1 runs the same arm plan on a (32, 24, 20) cell
at one and two virtual CPU devices, pinned by explicit device list, with one
iteration.  It exercises the harness, not the physics.  Two things genuinely
cannot happen there and are recorded rather than failed: the measured side of
a calibration row is a CUDA-only counter, so the calibration table is empty
and says why; and the CPU bodies are general torch code, so the band this job
is about is not a Triton launch argument at all.  The band and its padded
value are still recorded, because the harness that records them is what the
smoke checks.

Run:
    <torch python> mg24_padding_confirm.py        on a 4-GPU node
    MG24_DRY=1 <python> mg24_padding_confirm.py   print the arm plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized
token is an error, not a silent skip.
    MG24_RESULTS=<dir>              where the jsonl goes and the artifacts live
    MG24_ARMS=n3_shipped,n4_shipped subset of the arms, by arm name
    MG24_DRY=1                      print the arm plan and exit
    MG24_SMOKE=1                    the local CPU smoke
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
SMOKE = os.environ.get("MG24_SMOKE", "0") == "1"
DRY = os.environ.get("MG24_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

# Cone only.  Parallel's back projection rose 1.13x from three devices to four
# where cone's rose 1.67x, so the effect this job confirms is much larger on
# cone.  The parallel kernels take the same wrapper-level pad, and the gate run
# of the design note's increment 2 is what measures them.
GEOMETRY = "cone"

# mg19's cone cell, unchanged, so the two runs measure the same problem.  The
# recon shape and pixel count are what the geometry produces from this cell
# (checked 2026-08-17 in this checkout); they are registered so a moved default
# is visible on the row.
CELL = (2048, 2016, 1984)
RECON_SHAPE = (1984, 1984, 2016)
NUM_PIXELS = 3088364
# The smoke's cell.  Cone gives (20, 20, 24) and 276 pixels here.
SMOKE_CELL = (32, 24, 20)
SMOKE_RECON_SHAPE = (20, 20, 24)
SMOKE_NUM_PIXELS = 276
CELL_LABEL = "smoke" if SMOKE else "2k"

# The shipped pixel batch (tomography_model.FORWARD_PIXEL_BATCH).  No arm sets
# the attribute -- every arm runs the shipped configuration -- and every row
# records what the model reports.  A model reporting something other than this
# value is a FINDING, printed in the summary, because the forward comparison
# against mg19's batch sweep is written for 32768.  It is not an instrument
# failure: the arm still ran the shipped configuration its name claims.
SHIPPED_BATCH = 32768

# Device pins.  The production pins are the arms' own counts; the smoke has no
# CUDA devices, so it runs the same arm plan at one and two virtual CPU devices
# and each row records which mechanism pinned it.
SMOKE_PINS = {3: 1, 4: 2}

VCD_ITERATIONS = 1 if SMOKE else 3
VCD_SEED = 12345                 # the seed every other run in this series uses
PHANTOM_SEED = 20260817          # mg19's phantom seed; the files are mg19's

# THE PADDING WITNESS.  504 is the four-device band at this cell and 512 is
# what the remedy must turn it into.  An unpadded tree either has no
# padded_kernel_width at all or returns 504 from it.  The multiple is the
# library's _utils.KERNEL_WIDTH_MULTIPLE, repeated here so the summary can name
# it without importing the library in the parent process.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512
KERNEL_WIDTH_MULTIPLE = 16

# The values gate, and the class of number to expect beside it.  Cone binds
# hand-written Triton kernels on an H100.  A Triton body is not regenerated per
# call shape, so the reduction-order latitude that puts a COMPILED TORCH body in
# the 6e-4 class at an uneven split does not reach these arms, and the uneven
# three-device view split (683, 683, 682) is not a reason to loosen this.
VALUES_GATE_REL = 1e-4
VALUES_EXPECTATION = (
    "near machine zero on CPU" if SMOKE else
    "1e-6 class or better: Triton bodies, not compiled torch bodies, and "
    "mg21b read exactly zero between the padded and unpadded compilations")

# The band the library declares for a hand-written-kernel reconstruction
# (_memory_ledger.CALIBRATION_BAND).  Repeated here so the summary can print a
# verdict without importing torch in the parent process.
CAL_BAND = (1.00, 1.30)

# ── mg19's recorded cone numbers, and the design note's projections ───────────
# READ IN SESSION on 2026-08-18 from
# plans/experiments/torch_port/rows/mg19_baselines_h003_20260817_082830.jsonl,
# which mg19 wrote on 2026-08-17.  Every value below is the busiest device's
# seconds inside that run's own per-call brackets, or that run's recon wall.
# Nothing here is a gate.  It is printed beside this run's numbers so a reader
# sees both without leaving the log.
MG19 = dict(
    source="rows/mg19_baselines_h003_20260817_082830.jsonl (mg19, 2026-08-17)",
    batch=8192,                  # mg19's comparison arms
    slab_mib=64,                 # mg19's shipped combining slab
    back_n3=136.77,              # cone_n3_b8192
    back_n4=227.85,              # cone_n4_b8192
    back_n4_repeat=228.16,       # cone_n4_b8192_repeat
    forward_n3=203.47,           # cone_n3_b8192, at batch 8192
    forward_n4=151.58,           # cone_n4_b8192, at batch 8192
    forward_n4_b32768=127.90,    # cone_n4_b32768, the batch this job runs
    back_n4_slab256=226.12,      # cone_n4_slab256, at the slab this job runs
    recon_n3=459.4, recon_n4=485.1, recon_n4_repeat=419.7,
)
# The design note's section 3 projections, in its own words.
PROJECTED = {
    3: (None, "unchanged: the band is 672, already a multiple of 16"),
    4: (100.0, "near 100 s: the band goes from 504 to a padded 512"),
}

# The reconstruction fingerprint: an L2 norm plus values at seeded positions.
FINGERPRINT_SAMPLES = 65536
FINGERPRINT_SEED = 7

# Rows of the phantom drawn per call, if the phantom has to be rebuilt.  Eight
# rows of the production volume is a 256 MiB float64 draw; the whole volume in
# one call would be 59 GiB of float64.
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
    "MG24_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 24                  # wide enough for the longest arm id printed
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


def arm_specs():
    """Every arm, in the order the run takes them."""
    return [
        dict(arm="gen", kind="gen", n_dev=4, calibration=False),
        dict(arm="n3_shipped", kind="composed", n_dev=3, calibration=True),
        dict(arm="n4_shipped", kind="composed", n_dev=4, calibration=True),
        dict(arm="n4_shipped_repeat", kind="composed", n_dev=4,
             calibration=True),
    ]


def all_arm_names():
    return [spec["arm"] for spec in arm_specs()]


# ── artifact paths and checksums ──────────────────────────────────────────────
# The cell label is in every file name, so a smoke run and a production run can
# share a results directory without either reading the other's bytes.  The two
# large files keep mg19's prefix because they ARE mg19's files.
def _phantom_path():
    return os.path.join(RESULTS_DIR,
                        f"mg19_{CELL_LABEL}_{GEOMETRY}_phantom.npy")


def _sinogram_path():
    return os.path.join(RESULTS_DIR,
                        f"mg19_{CELL_LABEL}_{GEOMETRY}_sinogram.npy")


def _fingerprint_path(arm):
    return os.path.join(RESULTS_DIR,
                        f"mg24_{CELL_LABEL}_{GEOMETRY}_fp_{arm}.npy")


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
# invalid timing one, and this run's deliverable is a timing one.  A hot node
# usually also means a neighbour job is sharing the hardware.
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


# ── the tree witnesses ────────────────────────────────────────────────────────
def tree_witness():
    """Is this the merged tree, and does it carry the padding?

    Four checks, taken in the arm's own subprocess before anything is built.
    Three of them say the checkout is the 2026-08-17 merge that removed the
    banded multi-device forward.  The fourth is this job's own and is new: it
    imports the padding helper and asks it for the four-device band.  An
    unpadded tree either has no such helper or returns 504 unchanged, and an
    arm that ran there measured a different library than this job is about.

    Returns the record and a list of reasons, which is empty when the tree is
    the right one.
    """
    record = {}
    reasons = []

    try:
        from mbirtorch.tomography_model import TomographyModel
        cylinders = hasattr(TomographyModel,
                            "_sparse_forward_project_cylinders")
    except Exception as exc:                                      # noqa: BLE001
        cylinders = None
        record["cylinder_forward_error"] = f"{type(exc).__name__}: {exc}"
    record["has_cylinder_forward"] = cylinders
    if cylinders is not True:
        reasons.append("TomographyModel has no "
                       "_sparse_forward_project_cylinders; this is not the "
                       "merged tree")

    try:
        from mbirtorch import _sharding
        transfer = hasattr(_sharding, "transfer_cylinder_batch")
        broadcast = hasattr(_sharding, "broadcast_band_to_views")
    except Exception as exc:                                      # noqa: BLE001
        transfer, broadcast = None, None
        record["sharding_error"] = f"{type(exc).__name__}: {exc}"
    record["has_transfer_cylinder_batch"] = transfer
    record["has_broadcast_band_to_views"] = broadcast
    if transfer is not True:
        reasons.append("_sharding has no transfer_cylinder_batch; this is not "
                       "the merged tree")
    if broadcast is not False:
        reasons.append("_sharding still has broadcast_band_to_views; the "
                       "banded multi-device forward is still in this tree")

    # THE PADDING WITNESS.  This is the one that says the remedy is present.
    try:
        from mbirtorch._utils import padded_kernel_width
        probe = int(padded_kernel_width(PAD_PROBE_WIDTH))
    except Exception as exc:                                      # noqa: BLE001
        probe = None
        record["padding_error"] = f"{type(exc).__name__}: {exc}"
    record["padded_kernel_width_probe"] = probe
    record["padded_kernel_width_probe_input"] = PAD_PROBE_WIDTH
    record["padded_kernel_width_probe_expected"] = PAD_PROBE_EXPECTED
    record["padding_present"] = (probe == PAD_PROBE_EXPECTED)
    if probe is None:
        reasons.append("mbirtorch._utils has no padded_kernel_width, so this "
                       "tree does not carry the band padding this job "
                       "confirms")
    elif probe != PAD_PROBE_EXPECTED:
        reasons.append(f"padded_kernel_width({PAD_PROBE_WIDTH}) is {probe}, "
                       f"not {PAD_PROBE_EXPECTED}; the four-device band is not "
                       "being padded")

    record["ok"] = not reasons
    return record, reasons


def staleness_note():
    """The library's own notice that the widening speed floors owe a
    re-measure, RECORDED AND NOT GATED.

    The padding edits the two Triton files, and those files are projection-cost
    inputs the floors were measured against, so this note is non-None on the
    padded tree.  That is expected and it is what the design note's fourth
    increment exists to clear.  Recording it here means the log carries the
    library's own words rather than this file's summary of them.
    """
    try:
        from mbirtorch import _widening_floors
        note = _widening_floors.stale_note()
    except Exception as exc:                                      # noqa: BLE001
        return dict(note=None, read_ok=False,
                    error=f"{type(exc).__name__}: {exc}")
    return dict(note=(None if note is None else str(note)[:1500]),
                read_ok=True, is_none=(note is None),
                expectation="non-None on the padded tree: the padding edits "
                            "triton_cone.py and triton_parallel.py, which are "
                            "projection-cost inputs the floors were measured "
                            "against.  Recorded, never gated")


# ── model construction ────────────────────────────────────────────────────────
def _build_model(pin_devices=None):
    """Build the cone model at this run's cell.

    This is mg19's construction, unchanged, so a row here and a row there
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
    # A full turn of views, and the source distances written as multiples of the
    # detector width so the same expression builds the smoke model:
    # source-to-detector at four widths, source-to-isocenter at two, which puts
    # the object halfway between the source and the detector at a magnification
    # of two.
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
    does not make a timing reading wrong.
    """
    realized = tuple(int(s) for s in model.get_params("recon_shape"))
    pixels = int(model.full_index_count())
    return dict(recon_shape=list(realized), num_pixels_full=pixels,
                recon_shape_expected=list(recon_shape()),
                recon_shape_ok=(realized == tuple(recon_shape())),
                num_pixels_expected=int(num_pixels()),
                num_pixels_ok=(pixels == int(num_pixels())))


def _bands(model):
    """The band each device owns, and what the padding does to it.

    THE BAND IS THE SLICE BLOCK.  ``TomographyModel._slice_band_length``
    returns the slice-owner's whole shard unless ``back_project_slice_band`` is
    set on the model, and no arm here sets it.  So the band is the recon
    placement's slice block, which is 672 at three devices and 504 at four for
    a slice axis of 2016.  Those two numbers are the whole subject of this job:
    672 is a multiple of 16 and 504 is not.
    """
    from mbirtorch._utils import padded_kernel_width

    shape = tuple(model.get_params("recon_shape"))
    blocks = [end - start for _device, (start, end)
              in model.recon_placement.shard_ranges(shape[2])]
    padded = [int(padded_kernel_width(b)) for b in blocks]
    fixed = getattr(model, "back_project_slice_band", None)
    return dict(band_lengths=blocks, padded_band_lengths=padded,
                pad_slices=[p - b for b, p in zip(blocks, padded)],
                any_padded=any(p != b for b, p in zip(blocks, padded)),
                all_divisible=all(b % KERNEL_WIDTH_MULTIPLE == 0
                                  for b in blocks),
                fixed_band_attr=fixed,
                basis="the band is the recon placement's slice block, because "
                      "_slice_band_length returns the whole shard when "
                      "back_project_slice_band is unset")


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
    Under the calibration mode the policy always leaves one, and every composed
    arm here is on that path; a model that somehow left none has one rebuilt
    from the same two library functions the policy calls, so the numbers are the
    library's own either way.
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


def _ensure_phantom(shape):
    """Verify the staged phantom, or build it if it is not there.

    mg19 staged this file and this job reads it.  The rebuild path exists so a
    results directory that lost the file can still run, and the row records
    which path was taken.
    """
    import numpy as np

    path = _phantom_path()
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

    This is mg19's instrument, unchanged.  One pair per placement device is
    created and recorded INSIDE ``with torch.cuda.device(dev)``, which is what
    puts the markers on the stream that carries that device's kernels; the
    funnel itself is called outside that context, so it runs exactly as it runs
    without the instrument.

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
                  device=DEVICE, cuda=cuda, geometry=GEOMETRY,
                  cell=list(cell()), cell_label=CELL_LABEL,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  realized_pin=pin_for(cfg["n_dev"]),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  phantom_seed=PHANTOM_SEED,
                  values_gate=VALUES_GATE_REL,
                  values_expectation=VALUES_EXPECTATION,
                  shipped_batch_registered=SHIPPED_BATCH,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    result["invalid_reasons"] = []

    # THE TREE, checked before anything is built.  An arm that ran on the wrong
    # tree measured a different library, so this is an instrument failure and
    # not a finding.
    tree, tree_reasons = tree_witness()
    result["tree"] = tree
    result["invalid_reasons"].extend(tree_reasons)

    # The staleness note, RECORDED and never gated.
    result["staleness"] = staleness_note()

    # The combining slab the back projection will stream in.  Recorded, not
    # set: mg19 ran a 64 MiB slab and the shipped value is 256 MiB now, so the
    # summary needs the realized number to say which of mg19's arms is the
    # nearer comparison.
    try:
        from mbirtorch import _sharding
        result["reduce_slab_bytes"] = int(_sharding.REDUCE_SLAB_BYTES)
    except Exception as exc:                                      # noqa: BLE001
        result["reduce_slab_bytes"] = None
        result["reduce_slab_error"] = f"{type(exc).__name__}: {exc}"

    # The calibration mode owns the per-device peak counters.  It must be on for
    # every arm whose deliverable is a calibration row, and off for the
    # generator, which has no memory reading to make and should not reset
    # anything.
    present = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1"
    result["calibration_env_present"] = present
    result["calibration_env_ok"] = (present == bool(cfg.get("calibration")))
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
    A newly built automatic model still holds the trivial single-device
    placement, so a projection taken before the layout has settled would run on
    ONE device at every device count and compare its result against a sinogram
    made the same way -- passing while gating nothing.

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

    # WHICH BODIES ARE BOUND.  On an H100 cone binds hand-written Triton
    # kernels in both directions, so the list of directions running as general
    # torch code must be EMPTY.  On the CPU smoke the kernels do not exist and
    # both directions fall back, so it must be both.  The row says which
    # environment it expected, because the two answers are opposites and a row
    # without that note could be read either way.
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

    realized_batch = int(model._forward_pixel_batch())
    result["forward_pixel_batch"] = realized_batch
    result["forward_pixel_batch_attr"] = getattr(
        model, "forward_project_pixel_batch", None)
    # Recorded as a FINDING, not an instrument failure.  The arm runs the
    # shipped configuration whatever the shipped value is, which is what its
    # name claims; what a moved value breaks is the forward comparison against
    # mg19's batch sweep, and the summary says so there.
    result["batch_is_registered_shipped"] = (realized_batch == SHIPPED_BATCH)

    result.update(_shape_check(model))
    result["blocks"] = _block_lengths(model)
    result["bands"] = _bands(model)
    result["ledger"] = _ledger_record(model, model.sino_placement.devices)

    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"realized {realized} for a pin of {n_dev}")
    if not result["torch_bodies_ok"]:
        result["invalid_reasons"].append(
            f"torch_body_directions is {list(directions)}, not "
            f"{list(expected)}; this arm is not running the bodies it claims")
    return True


def run_gen(cfg, result, cuda):
    """Arm 1: verify the staged phantom and sinogram, and rebuild what is
    missing.

    mg19 staged both files on 2026-08-17 and this job reads them, so on the
    cluster this arm should verify two checksums and project nothing.  The
    rebuild path exists so a results directory that lost a file can still run.
    A rebuilt sinogram is recorded loudly, because it would come from THIS tree
    rather than from the tree mg19 measured.

    The arm runs with the calibration mode OFF: it makes no memory reading and
    has no business resetting the peak counters.
    """
    import numpy as np

    shape = tuple(recon_shape())
    path, digest, written = _ensure_phantom(shape)
    result.update(phantom_path=path, phantom_md5=digest,
                  phantom_written=written)

    model = _build_model(pin_devices=(None if cuda else
                                      [DEVICE] * pin_for(cfg["n_dev"])))
    if not _settle_and_witness(model, cfg, result, cuda):
        return
    phantom, phantom_md5 = _verified_load(path)
    result["phantom_md5_verified"] = phantom_md5

    sino_path = _sinogram_path()
    if _staged(sino_path):
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
    result["sinogram_note"] = (
        "REBUILT by this job: the staged file was missing, so this sinogram "
        "comes from the padded tree rather than from the tree mg19 measured")
    del projected
    _release(cuda)


def run_composed(cfg, result, cuda):
    """Arms 2 through 4: one calibrated three-iteration reconstruction.

    The order inside this function is the measurement's design, so it is stated
    once here and followed exactly below:

      1. build the model and settle, so the memory ledger prices the path this
         arm will run;
      2. on a refusal record the model-edge finding and repeat once with the
         preflight skipped;
      3. project the staged phantom and gate it against the staged sinogram at
         1e-4, then release the result, so no allocation of the values pass is
         inside the measured peak;
      4. seed, arm the per-call bracket, and reconstruct;
      5. read the calibration rows, the wall, the busy times, and the
         fingerprint.

    The wall covers the reconstruction call exactly as mg19 measured it, which
    includes gathering the finished volume to the host at the end.  That is a
    few seconds of several minutes at this size, and it is the same few seconds
    in every arm, so differences between arms are not made of it -- and keeping
    the definition means these walls can be read beside mg19's.
    """
    import numpy as np
    import torch

    def build():
        return _build_model(pin_devices=(None if cuda else
                                         [DEVICE] * pin_for(cfg["n_dev"])))

    model = build()
    settled = _settle_and_witness(model, cfg, result, cuda)
    if not settled:
        # A refusal here is a reading about the model's edge, not a fault, and
        # it is recorded as one BEFORE anything is forced.  Then the same arm
        # runs once more with the preflight skipped, because the timing is the
        # number this job exists to obtain and a refusal would otherwise leave
        # it unmeasured.
        result["preflight_refused_at_settle"] = True
        result["preflight_refusal_message"] = result.pop("preflight_message",
                                                         None)
        result["model_edge_finding"] = (
            f'the preflight refused {pin_for(cfg["n_dev"])} devices at this '
            "size.  The capacity table calls this count feasible and the "
            "design note says the padding does not move that verdict, so the "
            "refusal is a finding about the model's edge.  The arm was then "
            "repeated once with skip_memory_preflight set, to obtain the "
            "timing")
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
    phantom, result["phantom_md5"] = _verified_load(_phantom_path())
    start = time.perf_counter()
    projected = _to_numpy(model.forward_project(phantom))
    _sync(model, cuda)
    result["values_forward_s"] = time.perf_counter() - start
    result["projection_devices"] = [str(d)
                                    for d in model.sino_placement.devices]
    sinogram, result["sinogram_md5"] = _verified_load(_sinogram_path())
    result["values"] = compare_arrays(projected, sinogram, VALUES_GATE_REL)
    del projected, phantom
    _release(cuda)
    if not result["values"].get("ok"):
        result["values_failed"] = True
        result["timing_skipped_reason"] = (
            f'the values gate failed at rel {result["values"].get("rel")} '
            f"against a gate of {VALUES_GATE_REL}; a timing reading from an arm "
            "that computes a different sinogram would measure nothing")
        return

    # ── the reconstruction ───────────────────────────────────────────────────
    instrument = attach_funnel_instrument(model, torch, cuda)
    # The seed goes here, immediately before the reconstruction and inside this
    # process, so every arm draws the same pixel partitions and the comparison
    # between arms is between device counts, not between partition sequences.
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

    # THE SECOND DELIVERABLE.  The ledger now charges the PADDED band length,
    # so these ratios are how the padded charge is read at production scale.
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
                "a CUDA calibration arm produced no calibration rows, so one "
                "of the two readings this arm exists for is missing")

    volume = np.ascontiguousarray(np.asarray(_to_numpy(volume),
                                             dtype=np.float32))
    result["recon_volume_shape"] = list(volume.shape)
    fingerprint, samples = _volume_fingerprint(volume)
    result["fingerprint"] = fingerprint
    del volume
    _release(cuda)

    fp_path = _fingerprint_path(cfg["arm"])
    result["fingerprint_path"] = fp_path
    result["fingerprint_md5"] = _stage_small(fp_path, samples)

    # Every composed arm of THIS run that has already staged samples.
    # Report-only: three iterations of a nonlinear optimizer amplify
    # float-level differences, and no threshold measured at a smaller size
    # transfers to this one.  mg19's fingerprints are deliberately not compared
    # against: mg19 ran a different pixel batch, so a difference between the
    # runs would mix the padding with the batch.
    cross = []
    for other in arm_specs():
        if other["kind"] != "composed" or other["arm"] == cfg["arm"]:
            continue
        other_path = _fingerprint_path(other["arm"])
        if not _staged(other_path):
            continue
        other_samples, other_md5 = _verified_load(other_path)
        entry = compare_arrays(samples, other_samples)
        entry.update(other_arm=other["arm"], other_n_dev=other["n_dev"],
                     other_md5=other_md5)
        cross.append(entry)
        del other_samples
    result["cross_arm"] = cross


def run_arm(cfg):
    """One arm, in its own process.

    A new process per arm is not tidiness.  Compiled and Triton bodies are
    cached at module level for the life of a process and the peak memory
    counters are per process, so both would leak from one arm into the next if
    they shared an interpreter.
    """
    result, cuda = _base_result(cfg)
    health = [sample_gpu_health()]
    try:
        if cfg["kind"] == "gen":
            run_gen(cfg, result, cuda)
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

    Two variables are popped first and then set: the device pin and the
    calibration mode.  A value exported by the submitting shell therefore
    cannot reach an arm that did not ask for it -- in particular the generator,
    which must not own the peak counters.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(pin_for(cfg["n_dev"]))
    if cfg.get("calibration"):
        env["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
    return env


def _spawn(cfg):
    """Run one configuration in a NEW interpreter."""
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
    keep_arms = _strict_subset("MG24_ARMS", set(all_arm_names()))
    arms = [a for a in arm_specs() if a["arm"] in keep_arms]
    if not arms:
        raise ValueError("MG24_ARMS selects no arm")
    if not any(a["arm"] == "gen" for a in arms) and not (
            _staged(_phantom_path()) and _staged(_sinogram_path())):
        # The arms selected do not include the one that verifies the inputs,
        # and the inputs are not on disk.  Say so here rather than fail every
        # arm on a missing file.
        raise ValueError(
            f"MG24_ARMS excludes 'gen', but the phantom and sinogram are not "
            f"staged under {RESULTS_DIR}")
    return [dict(arm_id=f'{GEOMETRY}_{a["arm"]}', **a) for a in arms]


def _dry_run(plan):
    shape = tuple(cell())
    volume = tuple(recon_shape())
    sino_gib = shape[0] * shape[1] * shape[2] * 4 / 2 ** 30
    recon_gib = volume[0] * volume[1] * volume[2] * 4 / 2 ** 30
    print(f"mg24 the band-padding confirmation runs: {len(plan)} arms, "
          f"cone only, device {DEVICE}, {VCD_ITERATIONS} iteration(s) per "
          "composed arm")
    print(f"  results and artifacts -> {RESULTS_DIR}")
    print(f"  sinogram {shape} {sino_gib:.2f} GiB -> recon {volume} "
          f"{recon_gib:.2f} GiB, {num_pixels()} pixels inside the mask")
    print(f"  values gate {VALUES_GATE_REL:.0e}, expected {VALUES_EXPECTATION}")
    print(f"  calibration ratios read against {CAL_BAND[0]:.2f}-"
          f"{CAL_BAND[1]:.2f}; below the floor is UNDER, the direction the "
          "ledger may not err in")
    print(f"  the tree must carry the padding: padded_kernel_width"
          f"({PAD_PROBE_WIDTH}) == {PAD_PROBE_EXPECTED}, checked on every arm")
    print(f'  {"arm":<{ARM_COL}}{"pin":>5}{"calib":>7}  what it does')
    what = dict(gen="verifies the staged phantom and sinogram, rebuilding only "
                    "what is missing",
                composed="one calibrated reconstruction, bracketed per call")
    for cfg in plan:
        print(f'  {cfg["arm_id"]:<{ARM_COL}}{pin_for(cfg["n_dev"]):>5}'
              f'{("on" if cfg["calibration"] else "off"):>7}  '
              f'{what[cfg["kind"]]}')
    print("no library file is edited and no default is flipped: every arm runs "
          "the shipped configuration")
    print(f'  mg19\'s cone readings for comparison, from {MG19["source"]}: '
          f'back busy {MG19["back_n3"]:.1f}s at n=3 and {MG19["back_n4"]:.1f}s '
          f'at n=4.  The design note\'s section 3 projects n=3 unchanged and '
          f'n=4 near {PROJECTED[4][0]:.0f}s')


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg24_padding_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg24 band-padding confirmation on {RUN_LABEL} ({DEVICE}); "
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


def _region(row, region):
    regions = ((row.get("instrument") or {}).get("regions") or {})
    return regions.get(region) or {}


def _busy(row, region, field="busy_s_max"):
    return _region(row, region).get(field)


def _cal_ratios(row):
    return [c["ratio"] for c in (row.get("calibration") or [])]


def _band_text(row):
    bands = row.get("bands") or {}
    lengths = bands.get("band_lengths") or []
    padded = bands.get("padded_band_lengths") or []
    if not lengths:
        return "-"
    if lengths == padded:
        return f'{lengths[0]} (no pad)'
    return f'{lengths[0]} -> {padded[0]}'


def summarize(rows, out_path):
    """The table a person reads the verdicts from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and this function keeps them apart.  A values
    gate failure, a preflight refusal, a calibration ratio outside the band and
    a back busy time that misses the projection are FINDINGS: they are printed
    in full and none of them touches the exit code.  An arm that ran on the
    wrong tree, on the wrong device count, bound the wrong kind of body, ran
    under the wrong calibration setting, or produced no row at all is an
    instrument failure, because it did not measure what the plan said it would.
    """
    print(f"\n===== mg24 the band-padding confirmation runs ({out_path}) "
          "=====")
    broken, findings = [], []
    by_arm = {}

    header = (f'{"arm":<{ARM_COL}}{"pin":>4}{"batch":>7}{"dev":>5}'
              f'{"band":>14}{"values rel":>12}{"recon s":>10}{"cal min":>9}'
              f'{"cal max":>9}{"fwd busy":>10}{"back busy":>10}')
    print(header)
    print("-" * len(header))
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'{arm_id:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            broken.append(f"{arm_id}|error")
            continue
        by_arm[row.get("arm")] = row
        ratios = _cal_ratios(row)
        values = row.get("values") or {}
        print(f'{arm_id:<{ARM_COL}}{pin_for(row["n_dev"]):>4}'
              f'{row.get("forward_pixel_batch", "-"):>7}'
              f'{row.get("realized_n_devices", "-"):>5}'
              f'{_band_text(row):>14}'
              f'{_fmt(values.get("rel"), 12, "e", 3)}'
              f'{_fmt(row.get("recon_s"), 10, "f", 1)}'
              f'{_fmt(min(ratios) if ratios else None, 9, "f", 3)}'
              f'{_fmt(max(ratios) if ratios else None, 9, "f", 3)}'
              f'{_fmt(_busy(row, "forward"), 10, "f", 2)}'
              f'{_fmt(_busy(row, "back"), 10, "f", 2)}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    ARM CHECK FAIL: {reason}")
            broken.append(f"{arm_id}|check")
        if row.get("kind") == "gen":
            if row.get("phantom_written") and not SMOKE:
                print("    the phantom was REBUILT by this job; mg19's file "
                      "was not in the results directory")
            if row.get("sinogram_written") and SMOKE:
                print("    the smoke staged its own phantom and sinogram, "
                      "which is what a smoke does")
            elif row.get("sinogram_written"):
                print("    *** THE SINOGRAM WAS REBUILT BY THIS JOB, so it "
                      "comes from the padded tree and not from the tree mg19 "
                      "measured.  The values gate below is then a "
                      "self-consistency check within this run")
                findings.append(f"{arm_id}|sinogram-rebuilt")
            elif row.get("sinogram_md5"):
                print("    the staged phantom and sinogram verified against "
                      "their checksums; no projection was repeated")
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
        if row.get("batch_is_registered_shipped") is False:
            print(f'    *** the model reports pixel batch '
                  f'{row.get("forward_pixel_batch")}, not the '
                  f'{SHIPPED_BATCH} this file registered as the shipped '
                  "value.  The back comparison still holds, because the back "
                  "projection does not read the batch; the forward comparison "
                  "against mg19 does not")
            findings.append(f"{arm_id}|batch-moved")
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
        for entry in row.get("cross_arm") or []:
            print(f'    vs {entry.get("other_arm"):<18} samples: max-rel '
                  f'{_fmt(entry.get("rel"), 0, "e", 3)}, L2-rel '
                  f'{_fmt(entry.get("l2_rel"), 0, "e", 3)}')
    print("-" * len(header))
    print("fwd busy / back busy are the BUSIEST DEVICE's seconds inside the "
          "per-call brackets for each direction.  band is the slice block one "
          "device owns and, after the arrow, what the kernel wrapper pads it "
          "to")

    _print_per_device(by_arm)
    _print_back_comparison(by_arm, findings)
    _print_forward_note(by_arm)
    _print_calibration(by_arm)
    _print_pair_spread(by_arm)
    _print_tree_and_staleness(rows)

    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}.  A "
              "throttled device gives a valid memory reading and an invalid "
              "timing one, and the timing is what this job is for")
    if findings:
        print(f"\n{len(findings)} finding(s): {findings}.  None of these "
              "changes the exit code.  They are what the run is for")
    healthy = not broken
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"} '
          f"({len(broken)} arm(s) did not measure what the plan said).  The "
          "verdict -- whether the padding restored monotone back scaling at "
          "the 2048 class -- is read by a person from the comparison table "
          "above")
    return dict(healthy=healthy, broken=broken, findings=findings, hot=hot,
                arms=len(rows))


def _print_per_device(by_arm):
    """Every arm's forward and back busy seconds, per device, with the call
    counts.  The table above prints only the busiest device."""
    print("\n--- busy seconds per device, and the call counts ---")
    for spec in arm_specs():
        row = by_arm.get(spec["arm"])
        if not row or not row.get("instrument"):
            continue
        for region in ("forward", "back"):
            entry = _region(row, region)
            busy = entry.get("busy_s_per_device") or []
            print(f'  {spec["arm"]:<{ARM_COL}}{region:<9}'
                  f'{"  ".join(f"{b:.2f}" for b in busy)}  s over '
                  f'{entry.get("calls")} calls, sum '
                  f'{entry.get("busy_s_sum", 0.0):.2f} s')
        note = (row.get("instrument") or {})
        if note.get("event_cap_hit"):
            print(f'  {spec["arm"]:<{ARM_COL}}the event-pair cap was hit, so '
                  "some calls are missing from the sums above")


def _print_back_comparison(by_arm, findings):
    """THE DELIVERABLE: back busy time, this run against mg19 and against the
    design note's projections.

    THE mg19 COLUMN IS PRINTED ONLY AT THE PRODUCTION CELL.  mg19's numbers
    were measured on the 2048-class cone cell, and the smoke runs a (32, 24,
    20) cell on CPU.  Printing them beside each other would put a difference of
    two hundred seconds on the page that is entirely the cell, and a reader who
    saw that line out of context would read it as the remedy working.
    """
    print("\n--- back projection busy time: this run, mg19, and the design "
          "note's section 3 projections ---")
    if SMOKE:
        for spec in arm_specs():
            if spec["kind"] != "composed":
                continue
            row = by_arm.get(spec["arm"]) or {}
            print(f'  {spec["arm"]:<{ARM_COL}}{_band_text(row):>14}'
                  f'{_fmt(_busy(row, "back"), 11, "f", 2)} s')
        print(f"  no comparison is printed: this is the smoke, on a "
              f"{tuple(cell())} cell on {DEVICE}, and mg19 measured a "
              f"{CELL} cell on four H100s.  The two numbers describe "
              "different problems")
        return
    print(f'  {"arm":<{ARM_COL}}{"band":>14}{"this run":>11}{"mg19":>9}'
          "  the design note's section 3 projection")
    reference = {"n3_shipped": MG19["back_n3"],
                 "n4_shipped": MG19["back_n4"],
                 "n4_shipped_repeat": MG19["back_n4_repeat"]}
    for spec in arm_specs():
        if spec["kind"] != "composed":
            continue
        row = by_arm.get(spec["arm"]) or {}
        here = _busy(row, "back")
        target, words = PROJECTED[spec["n_dev"]]
        print(f'  {spec["arm"]:<{ARM_COL}}{_band_text(row):>14}'
              f'{_fmt(here, 11, "f", 1)}'
              f'{reference[spec["arm"]]:>9.1f}  {words}')
        if here is None:
            continue
        if spec["n_dev"] == 4 and target is not None:
            fell = reference[spec["arm"]] - here
            print(f'  {"":<{ARM_COL}}fell {fell:.1f} s from mg19, which is '
                  f'{100.0 * fell / reference[spec["arm"]]:.0f} percent.  The '
                  f'projection was a fall to near {target:.0f} s')
            if here > 1.5 * target:
                print(f'  {"":<{ARM_COL}}*** THIS MISSES THE PROJECTION by '
                      "more than half again.  Read the per-device lines and "
                      "the band column above before anything else")
                findings.append(f'{spec["arm"]}|missed-projection')
        if spec["n_dev"] == 3:
            moved = here - reference["n3_shipped"]
            print(f'  {"":<{ARM_COL}}moved {moved:+.1f} s from mg19.  The '
                  "projection was no change, because 672 is already a "
                  "multiple of 16")
    print("  CROSS-RUN NUMBERS ARE CONTEXT, NOT GATES.  mg19 ran on another "
          "day, on another node, at a pixel batch of "
          f'{MG19["batch"]} and a combining slab of {MG19["slab_mib"]} MiB.  '
          f'Its source is {MG19["source"]}')
    print(f'  the slab moved since mg19: mg19\'s own {MG19["slab_mib"]} MiB '
          f'pair read {MG19["back_n4"]:.2f} s and {MG19["back_n4_repeat"]:.2f} '
          f's, and its 256 MiB arm read {MG19["back_n4_slab256"]:.2f} s, so '
          "the slab is worth about 2 s here against a projected fall of about "
          f'{MG19["back_n4"] - PROJECTED[4][0]:.0f} s')


def _print_forward_note(by_arm):
    """The forward busy times, and why they differ from mg19's."""
    print("\n--- forward projection busy time, and why it is not comparable "
          "the same way ---")
    for spec in arm_specs():
        if spec["kind"] != "composed":
            continue
        row = by_arm.get(spec["arm"]) or {}
        here = _busy(row, "forward")
        batch = row.get("forward_pixel_batch")
        if SMOKE:
            against = "the smoke cell is not mg19's cell, so nothing is "\
                      "compared here"
        elif spec["n_dev"] == 4:
            against = (f'mg19 measured {MG19["forward_n4_b32768"]:.1f} s at '
                       f'this device count with a batch of '
                       f'{SHIPPED_BATCH}, and {MG19["forward_n4"]:.1f} s with '
                       f'a batch of {MG19["batch"]}')
        else:
            against = (f'mg19 measured {MG19["forward_n3"]:.1f} s at this '
                       f'device count with a batch of {MG19["batch"]}; it '
                       f'never ran three devices at {SHIPPED_BATCH}')
        print(f'  {spec["arm"]:<{ARM_COL}}{_fmt(here, 9, "f", 1)} s at batch '
              f'{batch}.  {against}')
    print("  THE PIXEL BATCH IS A FORWARD-PROJECTION KNOB.  It sets how many "
          "pixel columns the forward gather takes at a time, and the back "
          "projection never reads it, so the back comparison above compares "
          "the same work and this one does not")


def _print_calibration(by_arm):
    """The calibration ratios, which are also how the padded band charge is
    read at production scale."""
    print("\n--- calibration: the modeled peak against the measured peak ---")
    for spec in arm_specs():
        if spec["kind"] != "composed":
            continue
        row = by_arm.get(spec["arm"]) or {}
        ratios = _cal_ratios(row)
        if not ratios:
            print(f'  {spec["arm"]:<{ARM_COL}}no calibration rows')
            continue
        verdict = ("UNDER the floor" if min(ratios) < CAL_BAND[0] else
                   "inside the band" if max(ratios) <= CAL_BAND[1] else
                   "over the top of the band")
        modeled = [c["modeled_bytes"] / 2 ** 30 for c in row["calibration"]]
        measured = [c["measured_bytes"] / 2 ** 30 for c in row["calibration"]]
        print(f'  {spec["arm"]:<{ARM_COL}}ratios '
              f'{[round(r, 3) for r in ratios]}  {verdict}; modeled '
              f'{[round(m, 1) for m in modeled]} GiB against measured '
              f'{[round(m, 1) for m in measured]} GiB')
        if row.get("preflight_skipped"):
            print(f'  {"":<{ARM_COL}}(that peak came from a run with the '
                  "preflight skipped)")
    print(f"  read against {CAL_BAND[0]:.2f} to {CAL_BAND[1]:.2f}.  These "
          "ratios also read the ledger's PADDED band charge at production "
          "scale: the ledger charges the padded band length, from the same "
          "helper the kernel wrapper uses, so a ratio inside the band says "
          "the padded charge describes what the code allocates")


def _print_pair_spread(by_arm):
    """The four-device pair: how much two identical arms differ here."""
    print("\n--- the four-device pair: how much two identical arms differ on "
          "this machine ---")
    first = by_arm.get("n4_shipped") or {}
    second = by_arm.get("n4_shipped_repeat") or {}
    readings = (("back busy", lambda r: _busy(r, "back")),
                ("forward busy", lambda r: _busy(r, "forward")),
                ("recon wall", lambda r: r.get("recon_s")))
    any_line = False
    for label, read in readings:
        a, b = read(first), read(second)
        if a is None or b is None:
            continue
        any_line = True
        print(f'  {label:<14}{a:.2f} s and {b:.2f} s, {abs(b - a):.2f} s '
              "apart")
    if not any_line:
        print("  the pair did not complete, so this run measured no spread")
        return
    print("  a difference smaller than that spread is the machine, not the "
          "change.  The recon wall carries a first-compile cost the busy "
          "times do not")
    if not SMOKE:
        print("  mg19's own four-device pair differed by "
              f'{abs(MG19["back_n4_repeat"] - MG19["back_n4"]):.2f} s of back '
              "busy time and "
              f'{abs(MG19["recon_n4_repeat"] - MG19["recon_n4"]):.1f} s of '
              "wall")


def _print_tree_and_staleness(rows):
    """The tree witnesses and the library's staleness note, once."""
    trees = [r.get("tree") for r in rows if r.get("tree")]
    if trees:
        tree = trees[0]
        print("\n--- the tree these arms ran on ---")
        print(f'  _sparse_forward_project_cylinders present: '
              f'{tree.get("has_cylinder_forward")}; transfer_cylinder_batch '
              f'present: {tree.get("has_transfer_cylinder_batch")}; '
              f'broadcast_band_to_views present: '
              f'{tree.get("has_broadcast_band_to_views")}')
        print(f'  padded_kernel_width({tree.get("padded_kernel_width_probe_input")}'
              f') = {tree.get("padded_kernel_width_probe")}, expected '
              f'{tree.get("padded_kernel_width_probe_expected")}; the padding '
              f'is {"present" if tree.get("padding_present") else "ABSENT"}')
    notes = [r.get("staleness") for r in rows if r.get("staleness")]
    if notes:
        note = notes[0]
        print("\n--- the widening floors staleness note, RECORDED and NOT "
              "GATED ---")
        if not note.get("read_ok"):
            print(f'  the note could not be read: {note.get("error")}')
        elif note.get("note") is None:
            print("  the library reports no staleness note.  That is a "
                  "SURPRISE on this tree: the padding edits triton_cone.py "
                  "and triton_parallel.py, which are projection-cost inputs "
                  "the floors were measured against")
        else:
            print(f'  {note["note"]}')
            print("  this is EXPECTED on the padded tree and it gates "
                  "nothing.  The floors still govern; re-measuring them is "
                  "the design note's fourth increment")


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
