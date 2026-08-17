"""mg18 -- THE BANDED WALK AGAINST THE COLUMN GATHER, on the two geometries
that still run the banded walk: translation and multiaxis parallel.

WHY THIS RUN EXISTS.

mbirtorch has two drivers for a multi-device forward projection.  The older
one walks the reconstruction's slice axis in bands: it visits each slice-owner
in turn, copies that owner's band to every view-owner, and makes one projector
call per band (TomographyModel._sparse_forward_project_sharded).  The newer one
walks the pixel axis in batches: each view-owner gathers a batch of full-height
pixel columns from every slice-owner and makes ONE projector call per batch
over the whole slice range (TomographyModel._sparse_forward_project_columns).
Both compute the same operator, and the back projection is untouched by the
choice.

Cone beam and parallel beam have measured the two against each other on real
GPUs.  The gather won by 1.2x to 1.6x on the composed reconstruction, and both
classes now declare ``column_gather_geometry = True``.  TranslationModel and
MultiAxisParallelModel still run the banded walk, and the recorded rule is that
a geometry switches only on its OWN measurement.  Nobody has measured these
two.  This run is that measurement.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It flips no default and
changes no library file.  Each arm forces the driver it wants by setting two
attributes on its own model INSTANCE -- ``column_gather_geometry`` and
``forward_project_pixel_batch`` -- and every arm then records which driver the
model reports it will use, so a forcing that silently failed is visible rather
than assumed.  Nothing outside the arm's own process is touched.

THE QUESTION, stated so a reader does not have to reconstruct it: at each
geometry's production-representative cell, and at two and four devices, is the
column gather faster than the banded walk on a single forward projection, and
is it faster on a three-iteration reconstruction?  A secondary question comes
with it: does the pixel batch size matter, measured at the shipped 8192 and at
32768?

TERMS USED BELOW, defined once here:
    arm            one measured configuration -- one geometry, one device
                   count, one driver, one pixel batch -- run in its own fresh
                   process.
    cell           the sinogram shape, (views, detector rows, detector
                   channels).
    recon shape    the reconstruction volume, (rows, columns, slices).
    banded         the slice-band driver, the current default for these two
                   geometries.
    gather         the pixel-batched column-gather driver.
    forward arm    an arm that times one sparse forward projection.
    composed arm   an arm that times a three-iteration reconstruction.
    anchor         the one-device arm.  It stages the phantom and the
                   reference sinogram every other arm is judged against, and
                   it gives the one-device time the multi-device times are
                   read against.
    P              the number of pixels the region-of-reconstruction mask
                   keeps: the full pixel set one forward projection covers.

THE CELLS.  One per geometry, built exactly as mg8_geom_calib.py builds them,
so these rows sit beside mg8's rows for the same models.

    multiaxis   ma1024   (1024, 1008, 992)  -> recon (992, 992, 1148), 771,240
                         pixels.  Azimuths evenly spaced over half a turn,
                         elevations swept across +/- 0.5 radians.
    translation tct2k    (256, 1900, 3000)  -> recon (118, 360, 240), 42,480
                         pixels.  A 16 x 16 grid of translations at 24.0 and
                         16.0 spacing, source-to-isocenter and
                         source-to-detector both half the smaller detector
                         dimension.  This is the production translation-CT
                         shape the prerelease review priced.

Both realized recon shapes and pixel counts are recorded per arm and compared
against the expected values above.  A mismatch is RECORDED and does not fail
the arm, exactly as mg8's shape check does: a moved geometry default is worth
knowing about, but it does not make a timing wrong.

THE ARMS, per geometry, in this order.  The order is blocked so that the
drift witness comes last.

     1  gen_anchor              n=1, stages the phantom and the reference
     2  n2_banded
     3  n2_gather8192
     4  n2_gather32768
     5  n4_banded
     6  n4_gather8192
     7  n4_gather32768
     8  n2_composed_banded
     9  n2_composed_gather8192
    10  n4_composed_banded
    11  n4_composed_gather8192
    12  n2_banded_repeat        an exact repeat of arm 2

Arm 12 is the cross-time witness.  It runs the same configuration as arm 2 at
the end of the geometry's block, so the run says how much of any banded-gather
difference is simply the machine drifting over the hour between them.  A
banded-gather gap smaller than the arm 2 to arm 12 gap means nothing.

THE VALUES GATE, TAKEN BEFORE ANY TIMING.  A faster driver that computes a
different sinogram is not a faster driver.  Every forward arm therefore starts
with one cold pass through the public ``model.forward_project(phantom)``, which
takes a host array and returns a host array, and compares it against the staged
reference by

    rel = max|out - ref| / max|ref|      computed in float64

The gate is 1e-3.  The expectation to read beside it is the e-5 to e-6 class:
the same comparison on virtual CPU devices read 1.7e-6 on multiaxis and 1.4e-7
on translation at two and three devices (2026-08-16, this repository).  1e-3 is
not slack.  It is the documented envelope for a compiled torch body: a body
called at two distinct shapes is compiled dynamically, and dynamically
generated code contracts a reduction in a different order, which puts a single
application of the operator in the 6e-4 class on multiaxis (mg15c, mg15d).  A
gate below 1e-3 would fire on that known and benign difference.  A real driver
fault -- a dropped view, a duplicated slice, a misplaced batch boundary --
moves whole planes of the sinogram and lands near one, three orders above the
gate.  An arm whose gate fails is recorded as values-failed and its timing is
SKIPPED, because timing a driver that computes the wrong answer measures
nothing.

THE SETTLE, AND WHY IT IS LOAD-BEARING.  This is mg15's lesson, and it is the
one place where the obvious code is wrong.  On CUDA an arm pins its device
count through the environment variable MBIRTORCH_NUM_DEVICES.  That pin acts
only through the model's device policy, and ``forward_project`` never calls the
policy.  A freshly built automatic model still holds the trivial single-device
placement, so a projection taken before the layout has settled runs on ONE
device at every device count.  Every arm here therefore calls
``model._apply_device_policy()`` immediately after the model is built -- and
after the two gather attributes are set, so the memory ledger prices the path
the arm is about to run -- and only then asserts the device list it realized.
The realized device list is recorded on every row rather than assumed.

If the settle refuses the layout, it raises MemoryPreflightError.  That is
caught: the arm is recorded as refused-by-preflight with the message, and the
run continues with the next arm.  Every arm planned here models under 60 GB of
demand on an 80 GB device, so a refusal is information about the ledger rather
than an expected path.

HOW A FORWARD ARM IS TIMED.  The values pass above goes through the public
entry point, which converts host arrays at both ends, so it is not a timing.
The timing is taken on the device-resident surface instead:

  * the input is pre-sharded ONCE, before any timing, by flattening the
    phantom to (P, slices) at the mask's pixel indices and handing that to
    ``model._shard_recon``;
  * one warm-up pass runs and is discarded, so no compilation is inside a
    timed pass;
  * three passes of ``model.sparse_forward_project(shards, indices)`` are
    timed, each wall-clocked with ``time.perf_counter`` after every placement
    device has been synchronized, and each pass's output is released before
    the next one starts;
  * the three walls, their median and their spread are all recorded, because
    a median with no spread beside it cannot be judged.

Per-device peak memory is read as well.  The peak counters are reset after the
warm-up and before the first timed pass, so the peak describes a steady-state
pass rather than the compilation that preceded it.  The environment variable
MBIRTORCH_MEMORY_CALIBRATION must be ABSENT for that to mean anything -- the
calibration mode owns and resets those counters -- so its absence is asserted
and recorded.

HOW A COMPOSED ARM IS TIMED.  The composed arms read the staged reference
sinogram and run a three-iteration reconstruction, seeded immediately before
the call so every arm draws the same pixel partitions.  The wall is taken
around the reconstruction with every placement device synchronized before the
clock is read.  The final volume is gathered to the host, staged, and compared
against the OTHER driver's volume at the same device count, whichever of the
pair runs second doing the comparison.  Both an L2-relative and a
max-relative distance are recorded.  Neither is gated.  Compiled bodies called
at different shapes carry a documented reduction-order latitude in the 6e-4
class for a single application, and three iterations of a nonlinear optimizer
amplify that, so there is no threshold here that would separate a real fault
from the known one.

WHAT IS RECORDED ON EVERY ROW, beyond the numbers above:
  * the realized device list and whether its length is the arm's count;
  * whether both projection directions run as general torch code, read from
    ``_memory_ledger.torch_body_directions``.  These two geometries have no
    hand-written kernels, so this must be ('forward', 'back'); an arm where it
    is not is measuring something else and is marked invalid;
  * the driver the model reports through ``_column_gather_forward()``, checked
    against the driver the arm claims to be running;
  * whether the device count divides the view axis and the slice axis evenly,
    and the per-device block lengths on both;
  * the per-view charge the driver's own cost model returns for this arm's
    call shape, read through ``projector_functions.view_batch_charge``;
  * the memory ledger for the settled layout, or a ledger rebuilt from
    ``plan_from_model`` and ``estimate_peak_device_bytes`` when the settle left
    none;
  * a GPU health sample, so a thermally throttled node is visible;
  * the environment the arm ran under.

ARTIFACTS.  Written under MG18_RESULTS, regenerated only when missing, with an
md5 recorded beside each and verified on EVERY read.  An interrupted transfer
on a shared parallel filesystem is a recorded failure mode of this work, and a
comparison against a file that changed underneath the run would be a quietly
wrong answer rather than a loud one.

    mg18_<cell>_phantom.npy      the seeded phantom, one per cell
    mg18_<cell>_reference.npy    that phantom projected on ONE device
    mg18_<cell>_fwd_<arm>.npy    each forward arm's cold-pass output
    mg18_<cell>_recon_<arm>.npy  each composed arm's final volume

Every forward arm's output is staged so that later arms can measure their
distance from it.  That is how the same-count banded-to-gather distances and
the arm 2 to arm 12 repeat distance are obtained.  All of those are
report-only.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
ran and was recorded, every artifact was staged, and every md5 verified; and
when no arm ran on the wrong device count, ran the wrong driver, or turned out
to bind something other than a torch body.  It is NOT the verdict.  A values
gate failure and a preflight refusal are both printed in full and both leave
the exit code alone: they are findings.  The verdict -- whether either geometry
should switch its default -- is read by a person from the rows and the table
this job prints.

THE LOCAL SMOKE.  MG18_SMOKE=1 runs the whole arm plan on mg8's smoke shapes,
at device counts 1 and 2 only, on virtual CPU devices pinned by an explicit
device list, with composed arms at one iteration.  The gate stays at 1e-3; only
the expectation printed beside it changes, because CPU runs are close to
deterministic.  At these sizes the whole pixel set fits in one batch, so the
two gather arms of a count run the same call shape and are expected to agree
exactly; the smoke tests that the harness is sound, not that batching matters.

Run:
    <torch python> mg18_column_gather_ab.py        on a 4-GPU node
    MG18_DRY=1 <python> mg18_column_gather_ab.py   print the arm plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized
token is an error, not a silent skip.
    MG18_RESULTS=<dir>                   where the jsonl and the artifacts go
    MG18_GEOMS=multiaxis,translation     subset of the geometries
    MG18_ARMS=n2_banded,n2_gather8192    subset of the arms, by arm name
    MG18_DRY=1                           print the arm plan and exit
    MG18_SMOKE=1                         the local CPU smoke
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG18_SMOKE", "0") == "1"
DRY = os.environ.get("MG18_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

# One cell per geometry, carrying everything needed to rebuild the model plus
# the recon shape and pixel count mg8 recorded for it.  The model construction
# below is mg8's, unchanged, so a row here and a row there describe the same
# model.
CELLS = (
    dict(geometry="multiaxis", name="ma1024", cell=(1024, 1008, 992),
         recon_shape=(992, 992, 1148), num_pixels=771240),
    dict(geometry="translation", name="tct2k", cell=(256, 1900, 3000),
         translations=(16, 16), spacing=(24.0, 16.0),
         recon_shape=(118, 360, 240), num_pixels=42480),
)
# mg8's smoke shapes: the two geometries' own test shapes, so the smoke runs a
# configuration already known to reconstruct.  Counts stop at two, because at
# these sizes a four-way split of the translation reconstruction would leave a
# device owning no real data, which the layout validation rejects -- correctly,
# and for reasons unrelated to anything measured here.
SMOKE_CELLS = (
    dict(geometry="multiaxis", name="ma_tiny", cell=(16, 24, 20),
         recon_shape=(20, 20, 27), num_pixels=276),
    dict(geometry="translation", name="tct_tiny", cell=(16, 40, 32),
         translations=(4, 4), spacing=(3.0, 2.0),
         recon_shape=(2, 9, 6), num_pixels=18),
)
GEOMETRIES = ("multiaxis", "translation")

COUNTS = (2, 4)
SMOKE_COUNTS = (2,)
# The shipped pixel batch is 8192 (tomography_model.FORWARD_PIXEL_BATCH).  The
# second value asks whether a four times larger batch buys anything: a larger
# batch means fewer projector calls and fewer transfers, against a larger
# resident cylinder.
GATHER_BATCHES = (8192, 32768)
# The composed arms measure one batch only.  Two composed arms per count
# already cost about as much wall as the whole forward set.
COMPOSED_BATCH = 8192

VCD_ITERATIONS = 1 if SMOKE else 3
VCD_SEED = 12345                 # the seed every other run in this series uses
PHANTOM_SEED = 20260816          # this run's phantom, fixed here and nowhere else
TIMED_PASSES = 3

# The compiled-torch-body envelope, not a slack allowance.  See the module
# docstring for the measurement it comes from.
VALUES_GATE_REL = 1e-3
VALUES_EXPECTATION = (
    "1e-5 to 1e-6 class on GPU; near machine zero on CPU" if SMOKE else
    "1e-5 to 1e-6 class (virtual-CPU readings 1.7e-6 multiaxis, "
    "1.4e-7 translation)")

# Reading a whole phantom as float64 before casting would double the largest
# host array this run holds, so the draw is taken in slabs.  numpy's legacy
# generator fills a request in C order from one stream, so slab by slab is the
# same array the single call would have produced; the smoke asserts exactly
# that, at a size where both forms are cheap.
PHANTOM_SLAB_ROWS = 32

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
    "MG18_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 32                  # wide enough for the longest arm name printed
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
    repeat before.
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


def all_cells():
    return SMOKE_CELLS if SMOKE else CELLS


def counts():
    return SMOKE_COUNTS if SMOKE else COUNTS


def _cell_for(geometry):
    for spec in all_cells():
        if spec["geometry"] == geometry:
            return spec
    raise KeyError(f"no cell for geometry {geometry!r}")


def arm_specs():
    """Every arm of one geometry, in the order the run takes them.

    The forward arms come first, grouped by device count; the composed arms
    follow; the repeat of the first multi-device banded arm comes last, so the
    time between it and its original spans the whole geometry's block.
    """
    specs = [dict(arm="gen_anchor", kind="anchor", n_dev=1, form="banded",
                  pixel_batch=None, repeat_of=None)]
    for n in counts():
        specs.append(dict(arm=f"n{n}_banded", kind="forward", n_dev=n,
                          form="banded", pixel_batch=None, repeat_of=None))
        for batch in GATHER_BATCHES:
            specs.append(dict(arm=f"n{n}_gather{batch}", kind="forward",
                              n_dev=n, form="gather", pixel_batch=batch,
                              repeat_of=None))
    for n in counts():
        specs.append(dict(arm=f"n{n}_composed_banded", kind="composed",
                          n_dev=n, form="banded", pixel_batch=None,
                          repeat_of=None))
        specs.append(dict(arm=f"n{n}_composed_gather{COMPOSED_BATCH}",
                          kind="composed", n_dev=n, form="gather",
                          pixel_batch=COMPOSED_BATCH, repeat_of=None))
    first = counts()[0]
    specs.append(dict(arm=f"n{first}_banded_repeat", kind="forward",
                      n_dev=first, form="banded", pixel_batch=None,
                      repeat_of=f"n{first}_banded"))
    return specs


# ── artifact paths and checksums ──────────────────────────────────────────────
# The cell name rather than the geometry name is in every file name, so a smoke
# run and a production run can share a results directory without either reading
# the other's bytes.
def _phantom_path(cell_name):
    return os.path.join(RESULTS_DIR, f"mg18_{cell_name}_phantom.npy")


def _reference_path(cell_name):
    return os.path.join(RESULTS_DIR, f"mg18_{cell_name}_reference.npy")


def _forward_path(cell_name, arm):
    return os.path.join(RESULTS_DIR, f"mg18_{cell_name}_fwd_{arm}.npy")


def _recon_path(cell_name, arm):
    return os.path.join(RESULTS_DIR, f"mg18_{cell_name}_recon_{arm}.npy")


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


def _staged(path):
    """Whether an artifact and its checksum are both on disk."""
    return os.path.exists(path) and os.path.exists(_md5_path(path))


def _stage_array(path, array):
    """Write one artifact and its checksum, and return the checksum."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.save(path, np.ascontiguousarray(np.asarray(array, dtype=np.float32)))
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    return digest


def _verified_load(path, mmap=True):
    """Load a staged artifact after checking its checksum.

    Every read goes through here.  A truncated or half-written file on a shared
    parallel filesystem is a recorded failure mode of this work, and it has to
    stop the arm rather than quietly change a comparison.
    """
    import numpy as np

    with open(_md5_path(path)) as handle:
        expected = handle.read().strip()
    actual = _md5(path)
    if actual != expected:
        raise RuntimeError(f"staged artifact checksum mismatch at {path}: "
                           f"{actual} != {expected}")
    return np.load(path, mmap_mode="r" if mmap else None), actual


def _to_numpy(x):
    """The ONE host exit.

    A sharded array's ``gather()`` ALREADY returns a numpy array; calling
    ``.detach()`` on that result is a recorded failure that once cost a whole
    multi-device run its rows.  Nothing else in this file leaves the device.
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


def compare_arrays(out, ref, gate=None, budget_bytes=64 << 20):
    """max|out - ref| / max|ref| in float64, with an L2-relative distance
    beside it.

    Walked in slabs along the first axis, so neither the float32 arrays nor
    their float64 promotions are ever held whole: at the production shapes a
    single float64 copy of a sinogram is over 8 GiB, on a host that has just
    finished a multi-GPU run.  The maximum is accumulated slab by slab, which
    is exact -- a maximum of maxima is the maximum, with none of the summation
    error a norm carries.  Either array may be memory-mapped.

    ``gate`` None means the comparison is reported and not judged, which is
    every comparison here except the one against the reference.
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
# invalid timing one, and this run is a timing run, so the sample matters more
# here than it did where it was written.  A hot node usually also means a
# neighbour job is sharing the hardware.
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
def _build_model(spec, pin_devices=None):
    """Build the model for one cell.  This is mg8's construction, unchanged.

    On CUDA nothing is configured here: the device count is pinned through the
    environment, which leaves the model on the automatic branch where the
    ledger is actually consumed.  ``pin_devices`` is the CPU smoke's path only,
    where the environment pin does not apply -- the automatic search
    short-circuits when fewer than two CUDA devices are visible -- so the smoke
    pins by explicit device list and every row says which mechanism it used.
    """
    import numpy as np

    import mbirtorch

    cell = tuple(spec["cell"])
    if spec["geometry"] == "multiaxis":
        # Two angles per view: azimuth around the object, elevation (tilt) out
        # of the plane.  These are the geometry's own test defaults -- azimuths
        # evenly spaced over half a turn, elevations swept across +/- 0.5
        # radians.  The elevation range matters for the recon shape: the
        # automatic geometry divides the detector height by the smallest
        # |cos(elevation)|, and clamps that divisor at 0.1, so a range wide
        # enough to reach the clamp would inflate the slice count roughly
        # tenfold.  0.5 radians is far from the clamp.
        num_views = cell[0]
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            cell, np.stack([azimuth, elevation], axis=1))
    else:
        # A translation scan moves the object across a fixed source/detector
        # pair on a grid, so the "views" are grid positions rather than angles.
        num_x, num_z = spec["translations"]
        x_spacing, z_spacing = spec["spacing"]
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != cell[0]:
            raise RuntimeError(
                f'{spec["name"]}: {num_x}x{num_z} translations give '
                f"{vectors.shape[0]} views, but the sinogram has {cell[0]}")
        source_iso_dist = min(cell[1], cell[2]) / 2
        model = mbirtorch.TranslationModel(
            cell, vectors, source_detector_dist=source_iso_dist,
            source_iso_dist=source_iso_dist)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _force_form(model, form, pixel_batch):
    """Force one driver on this model INSTANCE, before the layout settles.

    Setting ``column_gather_geometry`` on the instance shadows the class
    attribute for this model and nothing else, so no default moves and no other
    process is affected.  The pixel batch is set the same way.  Both have to be
    in place before ``_apply_device_policy`` runs, because the memory ledger
    reads the driver choice while pricing the layout, and a ledger that priced
    the banded walk would not describe a gather arm.
    """
    if form == "gather":
        model.column_gather_geometry = True
        model.forward_project_pixel_batch = int(pixel_batch)
    # A banded arm sets nothing: False is the class attribute these two
    # geometries already carry, and leaving it alone is what makes the banded
    # arm the shipped configuration rather than a second forced one.


def _shape_check(model, spec):
    """Did the geometry defaults produce the recon shape this file registered?

    Recorded rather than raised: a moved default is worth knowing about, but it
    does not make a timing wrong.  This is mg8's check.
    """
    realized = tuple(int(s) for s in model.get_params("recon_shape"))
    pixels = int(model.full_index_count())
    expected = tuple(spec["recon_shape"])
    return dict(recon_shape=list(realized), num_pixels_full=pixels,
                recon_shape_expected=list(expected),
                recon_shape_ok=(realized == expected),
                num_pixels_expected=int(spec["num_pixels"]),
                num_pixels_ok=(pixels == int(spec["num_pixels"])))


def _block_lengths(model):
    """The block each device owns on the two sharded axes, and whether the
    device count divides either axis exactly.

    A count that divides both axes gives every device the same block length,
    which is the case where a compiled body stays specialized to one shape.  An
    uneven split is the case where it does not.  Recorded on every row so the
    reader does not have to re-derive it from the shapes.
    """
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    recon_shape = tuple(model.get_params("recon_shape"))
    n = model.sino_placement.n_devices
    views = [end - start for _device, (start, end)
             in model.sino_placement.shard_ranges(sinogram_shape[0])]
    slices = [end - start for _device, (start, end)
              in model.recon_placement.shard_ranges(recon_shape[2])]
    return dict(view_blocks=views, slice_blocks=slices,
                views_divide=(int(sinogram_shape[0]) % max(1, n) == 0),
                slices_divide=(int(recon_shape[2]) % max(1, n) == 0))


def _forward_view_charge(model, form, pixel_batch, slice_blocks):
    """What the per-view cost model charges for THIS arm's forward call shape.

    Two numbers come back from ``view_batch_charge``: how many views one body
    call takes, and how many bytes one of those views holds.  The call shape
    differs by driver.  The banded walk calls with the whole pixel set and one
    device's slice band; the gather calls with one pixel batch and the whole
    slice axis.  Both are recorded here in the arm's own terms.

    Note for the reader: these two geometries derive the transient's width from
    their parameters rather than from the band they are handed
    (``_transient_cols`` on each class), so the column count passed here does
    not move the charge.  It is recorded anyway, because that is a property of
    the current code and not of the measurement.
    """
    projector_functions = model.projector_functions
    fwd_body, _back_body = model._view_batch_bodies()
    args = model._view_batch_args()
    num_pixels = int(model.full_index_count())
    num_slices = int(model.get_params("recon_shape")[2])
    if form == "gather":
        p_call = min(int(pixel_batch), num_pixels)
        col_set = [num_slices]
        basis = ("one pixel batch over the whole slice axis "
                 "(the gather's call shape)")
    else:
        p_call = num_pixels
        col_set = sorted({int(c) for c in slice_blocks}, reverse=True)
        basis = ("the whole pixel set over one device's slice band "
                 "(the banded walk's call shape)")
    entries = []
    for cols in col_set:
        batch, per_view = projector_functions.view_batch_charge(
            fwd_body, p_call, int(cols), args)
        entries.append(dict(p_call=int(p_call), band_cols=int(cols),
                            view_batch=int(batch),
                            bytes_per_view=int(per_view)))
    return dict(basis=basis, entries=entries)


def _ledger_record(model, realized_devices):
    """The modeled peak per device for the settled layout.

    ``last_memory_ledger`` holds the ledger the device policy actually used
    whenever the policy built one, which is every automatic CUDA settle.  A
    model pinned by an explicit device list -- the CPU smoke -- skips that
    branch and leaves the attribute None, so the ledger is rebuilt here from
    the same two library functions the policy calls.  Either way the numbers
    are the library's own; nothing is re-derived by this harness.
    """
    from mbirtorch import _memory_ledger

    ledger = getattr(model, "last_memory_ledger", None)
    source = "model.last_memory_ledger (built by the device policy)"
    if ledger is None:
        ledger = _memory_ledger.estimate_peak_device_bytes(
            _memory_ledger.plan_from_model(model, realized_devices,
                                           workload="recon"))
        source = "rebuilt here via plan_from_model + estimate_peak_device_bytes"
    dominant = []
    for i in range(len(ledger.devices)):
        dominant.append(ledger.dominant_phase(i).name)
    return dict(source=source, devices=[str(d) for d in ledger.devices],
                modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
                dominant_phase=dominant,
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


def _reset_peaks(model, cuda):
    if not cuda:
        return
    import torch

    for device in model.sino_placement.devices:
        torch.cuda.reset_peak_memory_stats(device)


def _read_peaks(model, cuda):
    if not cuda:
        return None
    import torch

    return [int(torch.cuda.max_memory_allocated(device))
            for device in model.sino_placement.devices]


def _shard_tensors(x):
    """The per-device tensors of a projection result, whichever form it took.

    A one-device model returns a plain tensor; a multi-device model returns a
    Shards container.
    """
    tensors = getattr(x, "tensors", None)
    return list(tensors) if tensors is not None else [x]


def _device_rel_max(a, b):
    """max|a - b| / max|a| computed on the devices the arrays already live on.

    Used for the anchor's warm-repeat floor, where both arrays are sinogram
    sized and moving them to the host would cost more than the measurement is
    worth.
    """
    import torch

    max_diff, max_ref = 0.0, 0.0
    for x, y in zip(_shard_tensors(a), _shard_tensors(b)):
        if x.numel() == 0:
            continue
        max_ref = max(max_ref, float(torch.max(torch.abs(x))))
        max_diff = max(max_diff, float(torch.max(torch.abs(x - y))))
    return (max_diff / max_ref) if max_ref > 0 else None


def _seeded_phantom(recon_shape, seed):
    """The run's phantom: uniform noise on the reconstruction grid.

    Drawn in slabs of rows rather than in one call.  numpy's legacy generator
    fills any request in C order from a single stream, so a sequence of slabs
    holds exactly the values the single call would have produced, and the
    single call would have held the whole volume as float64 first -- twice the
    bytes of the float32 result.  The smoke checks the two forms against each
    other, at a size where checking is free.
    """
    import numpy as np

    np.random.seed(seed)
    rows, cols, slices = (int(s) for s in recon_shape)
    out = np.empty((rows, cols, slices), dtype=np.float32)
    for start in range(0, rows, PHANTOM_SLAB_ROWS):
        stop = min(start + PHANTOM_SLAB_ROWS, rows)
        out[start:stop] = np.random.rand(stop - start, cols,
                                         slices).astype(np.float32)
    if SMOKE:
        np.random.seed(seed)
        direct = np.random.rand(rows, cols, slices).astype(np.float32)
        if not np.array_equal(direct, out):
            raise RuntimeError("the slab-wise phantom draw does not reproduce "
                               "the single-call draw; the stream-order "
                               "assumption in _seeded_phantom is wrong")
    return out


# ── the worker: one arm, one process ──────────────────────────────────────────
def _arm_preamble(cfg, spec):
    """Everything an arm does before it measures anything: build the model,
    force the driver, settle the layout, and record the witnesses.

    Returns ``(model, phantom, result)``, or ``(None, None, result)`` when the
    settle refused the layout.  ``result`` always carries enough to explain
    itself.
    """
    import torch

    from mbirtorch import _memory_ledger

    n_dev, form = cfg["n_dev"], cfg["form"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = None if cuda else [DEVICE] * n_dev

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  phantom_seed=PHANTOM_SEED,
                  values_gate=VALUES_GATE_REL,
                  values_expectation=VALUES_EXPECTATION,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_column_gather=os.environ.get(
                      "MBIRTORCH_FORWARD_COLUMN_GATHER"))
    # The peak counters this arm reads are owned by the calibration mode when
    # that mode is on, and it resets them at points this harness does not
    # control.  Its absence is the condition under which the peaks below mean
    # what they say.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") is None)
    result["invalid_reasons"] = []

    model = _build_model(spec, pin_devices=pin_devices)
    _force_form(model, form, cfg.get("pixel_batch"))
    result.update(_shape_check(model, spec))

    if cfg["kind"] == "anchor":
        # The anchor's real work is one forward projection, but the preflight
        # prices a full reconstruction, and at the multiaxis cell that price
        # sits within half a GiB of an idle H100's budget.  A refusal here
        # would cost the whole geometry, because every later arm reads the
        # reference this arm stages.  mg8 ran the full three-iteration
        # reconstruction at this cell on one H100, so the workload the anchor
        # actually runs is known to fit.  The skip is recorded on the row.
        model.skip_memory_preflight = True
    result["preflight_skipped"] = bool(
        getattr(model, "skip_memory_preflight", False))

    # THE SETTLE.  See the module docstring: the environment pin acts only
    # through the device policy, and forward_project never calls the policy, so
    # without this line every multi-device arm would run on one device.
    try:
        model._apply_device_policy()
    except _memory_ledger.MemoryPreflightError as exc:
        result["refused_by_preflight"] = True
        result["preflight_message"] = str(exc)[-3000:]
        result["realized_devices"] = [str(d)
                                      for d in model.sino_placement.devices]
        result["gpu_health"] = sample_gpu_health()
        result["gpu_hot"] = row_is_hot(result["gpu_health"])
        return None, None, result
    result["refused_by_preflight"] = False

    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))

    # Neither of these geometries has a hand-written kernel, so both directions
    # must run as general torch code.  If that ever stops being true the arm is
    # timing a different thing than it claims to.
    directions = tuple(_memory_ledger.torch_body_directions(model))
    result["torch_body_directions"] = list(directions)
    result["torch_bodies_ok"] = (directions == ("forward", "back"))

    # The driver the model says it will use, against the one this arm claims.
    gathering = bool(model._column_gather_forward())
    result["column_gather_forward"] = gathering
    result["form_ok"] = (gathering == (form == "gather"))
    result["forward_pixel_batch"] = int(model._forward_pixel_batch())
    result["driver"] = (
        "single-device projector (neither multi-device driver runs)"
        if len(realized) == 1 else
        "_sparse_forward_project_columns" if gathering else
        "_sparse_forward_project_sharded")

    result["blocks"] = _block_lengths(model)
    result["forward_view_charge"] = _forward_view_charge(
        model, form, cfg.get("pixel_batch"), result["blocks"]["slice_blocks"])
    result["ledger"] = _ledger_record(model, model.sino_placement.devices)

    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"realized {realized} for n={n_dev}")
    if not result["torch_bodies_ok"]:
        result["invalid_reasons"].append(
            f"torch_body_directions is {list(directions)}, not "
            "['forward', 'back']; a hand-written kernel is now bound")
    if not result["form_ok"]:
        result["invalid_reasons"].append(
            f"_column_gather_forward() is {gathering} on a {form} arm; the "
            "instance attributes did not force the driver")
    if not result["calibration_absent_ok"]:
        result["invalid_reasons"].append(
            "MBIRTORCH_MEMORY_CALIBRATION is set, so this arm does not own "
            "the peak counters it reads")

    if cfg["kind"] == "composed":
        # A composed arm reads the reference sinogram, not the phantom, and at
        # the production recon shape the phantom is gigabytes.  Reading it here
        # would cost a full pass over the filesystem for nothing.
        return model, None, result
    phantom, phantom_md5 = _verified_load(_phantom_path(spec["name"]),
                                          mmap=False)
    result["phantom_md5"] = phantom_md5
    return model, phantom, result


def _ensure_phantom(spec):
    """Stage the phantom for one cell if it is not already staged."""
    path = _phantom_path(spec["name"])
    if _staged(path):
        _verified_load(path)                    # verify, then drop the map
        return path, _md5(path), False
    phantom = _seeded_phantom(spec["recon_shape"], PHANTOM_SEED)
    digest = _stage_array(path, phantom)
    del phantom
    return path, digest, True


def _ensure_reference(spec, model, phantom, result):
    """Stage the reference sinogram, or compare this pass against the staged
    one.

    The reference is the phantom projected on ONE device through the public
    entry point.  When it is already on disk this pass is compared against it
    instead, which is what makes a re-run of the anchor a real reading rather
    than a repeat of a write.
    """
    path = _reference_path(spec["name"])
    start = time.perf_counter()
    projected = _to_numpy(model.forward_project(phantom))
    result["cold_forward_s"] = time.perf_counter() - start
    result["projection_devices"] = [str(d)
                                    for d in model.sino_placement.devices]
    if _staged(path):
        reference, digest = _verified_load(path)
        result["reference_written"] = False
        result["values"] = compare_arrays(projected, reference,
                                          VALUES_GATE_REL)
        del reference
    else:
        digest = _stage_array(path, projected)
        result["reference_written"] = True
        # This pass IS the reference, so the comparison is with itself and says
        # nothing.  It is recorded as a pass so the row has the same shape as
        # every other row, and the flag above says which case it was.
        result["values"] = dict(ok=True, rel=0.0, gate=VALUES_GATE_REL,
                                reason="this pass wrote the reference")
    result["reference_md5"] = digest
    result["reference_path"] = path
    del projected
    return path


def _stage_forward_output(spec, cfg, model, phantom, result):
    """The cold values pass of a forward arm: project, gate, compare against
    every earlier arm of the same device count, and stage the output.

    The gate comes before any timing.  A driver that computes a different
    sinogram is not a faster driver, and timing it would produce a number
    somebody could quote.
    """
    reference_path = _reference_path(spec["name"])
    start = time.perf_counter()
    projected = _to_numpy(model.forward_project(phantom))
    result["cold_forward_s"] = time.perf_counter() - start
    result["projection_devices"] = [str(d)
                                    for d in model.sino_placement.devices]

    reference, result["reference_md5"] = _verified_load(reference_path)
    result["values"] = compare_arrays(projected, reference, VALUES_GATE_REL)
    del reference

    # Every forward arm of this cell that has already run and staged an output
    # at the SAME device count.  A different driver gives the banded-to-gather
    # distance; the same driver gives the repeat-to-original distance, which is
    # the cross-time witness.  All report-only.
    cross = []
    for other in arm_specs():
        if other["kind"] != "forward" or other["arm"] == cfg["arm"]:
            continue
        if other["n_dev"] != cfg["n_dev"]:
            continue
        other_path = _forward_path(spec["name"], other["arm"])
        if not _staged(other_path):
            continue
        other_array, other_md5 = _verified_load(other_path)
        entry = compare_arrays(projected, other_array)
        entry.update(other_arm=other["arm"], other_form=other["form"],
                     other_pixel_batch=other["pixel_batch"],
                     other_md5=other_md5,
                     same_form=(other["form"] == cfg["form"]))
        cross.append(entry)
        del other_array
    result["cross_form"] = cross

    out_path = _forward_path(spec["name"], cfg["arm"])
    result["forward_output_path"] = out_path
    result["forward_output_md5"] = _stage_array(out_path, projected)
    del projected


def _time_forward(model, phantom, cfg, result, cuda):
    """The device-resident timing of one forward projection driver.

    The public entry point converts host arrays at both ends, so it is not what
    gets timed.  The input is pre-sharded once, a warm-up pass pays whatever
    compilation the driver needs, and only then are three passes wall-clocked.
    """
    import numpy as np
    import torch

    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    idx = np.asarray(model._full_indices(), dtype=np.int64)
    values2d = phantom.reshape(recon_shape[0] * recon_shape[1],
                               recon_shape[2])[idx]
    shards = model._shard_recon(torch.as_tensor(values2d))
    # On the model device, which is where a production call's indices already
    # sit: forward_project passes model.full_indices_device().  Handing over a
    # host tensor instead would put a host-to-device copy of the index array
    # inside every timed pass.
    idx_tensor = torch.as_tensor(idx, dtype=torch.int64,
                                 device=model.torch_device)
    del values2d
    result["timed_pixels"] = int(idx.shape[0])

    out = model.sparse_forward_project(shards, idx_tensor)   # warm-up
    _sync(model, cuda)
    out = None

    # After the settle and after the warm-up, so the peaks describe a
    # steady-state pass rather than the compilation that preceded it.
    _reset_peaks(model, cuda)

    walls = []
    for _ in range(TIMED_PASSES):
        _sync(model, cuda)
        start = time.perf_counter()
        out = model.sparse_forward_project(shards, idx_tensor)
        _sync(model, cuda)
        walls.append(time.perf_counter() - start)
        # Released before the next pass, so no pass is timed with the previous
        # pass's output still resident.
        out = None

    peaks = _read_peaks(model, cuda)
    result["forward_walls_s"] = walls
    result["forward_median_s"] = statistics.median(walls)
    result["forward_spread_s"] = max(walls) - min(walls)
    result["peak_bytes"] = peaks
    result["peak_max_bytes"] = (max(peaks) if peaks else None)
    return shards, idx_tensor


def _warm_repeat_floor(model, shards, idx_tensor, result, cuda):
    """How far two warm passes of the same driver sit from each other.

    This is the floor every other distance in the run has to be read against.
    Two passes that already differ by some amount say that a difference of that
    size between two drivers means nothing.
    """
    first = model.sparse_forward_project(shards, idx_tensor)
    _sync(model, cuda)
    second = model.sparse_forward_project(shards, idx_tensor)
    _sync(model, cuda)
    result["warm_repeat_rel"] = _device_rel_max(first, second)
    first, second = None, None


def _run_composed(spec, cfg, model, result, cuda):
    """One three-iteration reconstruction, timed, staged, and compared against
    the other driver at the same device count."""
    import numpy as np

    reference, result["reference_md5"] = _verified_load(
        _reference_path(spec["name"]), mmap=False)

    _reset_peaks(model, cuda)
    # The seed goes here, immediately before the reconstruction and inside this
    # process, so both drivers at a device count draw the same pixel partitions
    # and the comparison is between drivers and nothing else.
    np.random.seed(VCD_SEED)
    _sync(model, cuda)
    start = time.perf_counter()
    recon, _info = model.recon(reference, max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0,
                               logfile_path=None, print_logs=False)
    _sync(model, cuda)
    result["recon_s"] = time.perf_counter() - start
    peaks = _read_peaks(model, cuda)
    result["peak_bytes"] = peaks
    result["peak_max_bytes"] = (max(peaks) if peaks else None)
    del reference

    volume = np.ascontiguousarray(np.asarray(_to_numpy(recon),
                                             dtype=np.float32))
    del recon
    out_path = _recon_path(spec["name"], cfg["arm"])
    result["recon_output_path"] = out_path
    result["recon_output_md5"] = _stage_array(out_path, volume)
    result["recon_volume_shape"] = list(volume.shape)

    # The other driver at this device count.  Whichever of the pair runs second
    # does the comparison; the first finds nothing staged and records none.
    cross = []
    for other in arm_specs():
        if other["kind"] != "composed" or other["arm"] == cfg["arm"]:
            continue
        if other["n_dev"] != cfg["n_dev"] or other["form"] == cfg["form"]:
            continue
        other_path = _recon_path(spec["name"], other["arm"])
        if not _staged(other_path):
            continue
        other_volume, other_md5 = _verified_load(other_path)
        entry = compare_arrays(volume, other_volume)
        entry.update(other_arm=other["arm"], other_form=other["form"],
                     other_pixel_batch=other["pixel_batch"],
                     other_md5=other_md5)
        cross.append(entry)
        del other_volume
    result["cross_form"] = cross
    del volume


def run_arm(cfg):
    """One arm, in its own process.

    A fresh process per arm is not tidiness.  Compiled bodies are cached at
    module level for the life of a process, and the peak memory counters are
    per process, so two arms in one process would share both.
    """
    import torch

    spec = _cell_for(cfg["geometry"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    health = [sample_gpu_health()]

    if cfg["kind"] == "anchor":
        _ensure_phantom(spec)
    model, phantom, result = _arm_preamble(cfg, spec)
    if model is None:                     # refused by the memory preflight
        return result

    if cfg["kind"] == "anchor":
        _ensure_reference(spec, model, phantom, result)
    elif cfg["kind"] == "forward":
        _stage_forward_output(spec, cfg, model, phantom, result)

    if result["invalid_reasons"]:
        # The arm is not measuring what it claims to.  Nothing is timed, and
        # the reasons are on the row.
        result["timing_skipped_reason"] = "; ".join(result["invalid_reasons"])
    elif cfg["kind"] == "composed":
        _run_composed(spec, cfg, model, result, cuda)
    elif not result["values"].get("ok"):
        result["values_failed"] = True
        result["timing_skipped_reason"] = (
            f'the values gate failed at rel {result["values"].get("rel")} '
            f"against a gate of {VALUES_GATE_REL}; timing a driver that "
            "computes a different sinogram would measure nothing")
    else:
        shards, idx_tensor = _time_forward(model, phantom, cfg, result, cuda)
        if cfg["kind"] == "anchor":
            _warm_repeat_floor(model, shards, idx_tensor, result, cuda)
        shards, idx_tensor = None, None

    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def stage_inputs(cfg):
    """Stage one cell's phantom and reference without measuring anything.

    Runs only when a run selects arms of a geometry but not that geometry's
    anchor, and the artifacts are not already on disk.  Pinned to one device,
    because the reference is by definition the one-device projection.
    """
    import torch

    spec = _cell_for(cfg["geometry"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"))
    path, digest, written = _ensure_phantom(spec)
    result.update(phantom_path=path, phantom_md5=digest,
                  phantom_written=written)
    model = _build_model(spec, pin_devices=(None if cuda else [DEVICE]))
    # One device, staging only: the same preflight skip the anchor arm takes,
    # for the same reason (the reconstruction-sized price sits near the budget
    # at the multiaxis cell; the staging projection itself is known to fit).
    model.skip_memory_preflight = True
    result["preflight_skipped"] = True
    model._apply_device_policy()
    result["realized_devices"] = [str(d) for d in model.sino_placement.devices]
    phantom, _md5_value = _verified_load(_phantom_path(spec["name"]),
                                         mmap=False)
    _ensure_reference(spec, model, phantom, result)
    return result


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited.

    The device pin is MBIRTORCH_NUM_DEVICES and nothing else.  Three variables
    are popped first and then set: the pin, the calibration mode (which would
    take ownership of the peak counters this run reads), and the process-wide
    column-gather override (which would overrule the per-instance forcing every
    arm depends on).  A value exported by the submitting shell therefore cannot
    reach an arm that did not ask for it.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
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
    """The plan in job order: each geometry's whole block, one geometry after
    the other, so a truncated job still holds whole geometries."""
    geometries = _strict_subset("MG18_GEOMS", set(GEOMETRIES))
    every_arm = {a["arm"] for a in arm_specs()}
    keep_arms = _strict_subset("MG18_ARMS", every_arm)
    plan = []
    for geometry in GEOMETRIES:
        if geometry not in geometries:
            continue
        spec = _cell_for(geometry)
        arms = [a for a in arm_specs() if a["arm"] in keep_arms]
        if not arms:
            continue
        ready = (_staged(_phantom_path(spec["name"]))
                 and _staged(_reference_path(spec["name"])))
        if not ready and not any(a["arm"] == "gen_anchor" for a in arms):
            # The arms selected do not include the one that stages this cell's
            # inputs, and the inputs are not on disk.  Stage them first, and
            # say so, rather than fail every arm on a missing file.
            entry = dict(mode="stage", geometry=geometry,
                         cell_name=spec["name"], cell=list(spec["cell"]),
                         arm="stage_inputs", n_dev=1,
                         arm_id=f'{spec["name"]}_stage_inputs')
            plan.append(entry)
        for aspec in arms:
            entry = dict(mode="arm", geometry=geometry,
                         cell_name=spec["name"], cell=list(spec["cell"]),
                         arm_id=f'{spec["name"]}_{aspec["arm"]}', **aspec)
            plan.append(entry)
    if not plan:
        raise ValueError("MG18_GEOMS and MG18_ARMS together select no arm")
    return plan


def _dry_run(plan):
    measured = [c for c in plan if c["mode"] == "arm"]
    print(f"mg18 banded walk against column gather: {len(measured)} arms "
          f"({len(plan) - len(measured)} input-staging jobs), device {DEVICE}, "
          f"{VCD_ITERATIONS} iteration(s) per composed arm")
    print(f"  results and artifacts -> {RESULTS_DIR}")
    print(f"  values gate {VALUES_GATE_REL:.0e}, expected {VALUES_EXPECTATION}")
    for spec in all_cells():
        gib = (spec["cell"][0] * spec["cell"][1] * spec["cell"][2] * 4
               / 2 ** 30)
        recon_gib = (spec["recon_shape"][0] * spec["recon_shape"][1]
                     * spec["recon_shape"][2] * 4 / 2 ** 30)
        print(f'  {spec["name"]:<9} {spec["geometry"]:>11} '
              f'sinogram {tuple(spec["cell"])!s:>20} {gib:6.2f} GiB '
              f'-> recon {tuple(spec["recon_shape"])!s:<18} {recon_gib:6.2f} GiB')
    print(f'  {"arm":<{ARM_COL}}{"n":>3}{"form":>9}{"batch":>8}  kind')
    for cfg in plan:
        if cfg["mode"] != "arm":
            print(f'  {cfg["arm_id"]:<{ARM_COL}}{cfg["n_dev"]:>3}{"-":>9}'
                  f'{"-":>8}  stages this cell\'s phantom and reference')
            continue
        print(f'  {cfg["arm_id"]:<{ARM_COL}}{cfg["n_dev"]:>3}{cfg["form"]:>9}'
              f'{(cfg["pixel_batch"] or "-"):>8}  {cfg["kind"]}')
    print("no default is flipped: each arm forces its driver on its own model "
          "instance, and records what the model reports it will run")


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg18_ab_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg18 banded walk against column gather on {RUN_LABEL} ({DEVICE}); "
          f'{len([c for c in plan if c["mode"] == "arm"])} arms -> {out_path}',
          flush=True)
    rows = []
    # Rows are written as they finish, so a job that runs out of wall time
    # still yields every arm it completed.
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


def summarize(rows, out_path):
    """The table a person reads the verdict from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and the function keeps them apart.  A values
    gate failure and a preflight refusal are findings: they are printed in full
    and they do not touch the exit code.  An arm that ran on the wrong device
    count, ran the wrong driver, bound something other than a torch body, or
    did not produce a row at all is an instrument failure, because it did not
    measure what the plan said it would.
    """
    print(f"\n===== mg18 banded walk against column gather ({out_path}) =====")
    broken, refused, values_failed = [], [], []
    by_arm = {}

    header = (f'{"arm":<{ARM_COL}}{"n":>3}{"form":>9}{"batch":>8}'
              f'{"devices":>9}{"values":>11}{"median s":>11}{"spread s":>10}'
              f'{"peak GiB":>10}  driver')
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("mode") == "stage":
            if row.get("error"):
                print(f'{row.get("arm_id"):<{ARM_COL}}  ERROR staging inputs: '
                      f'{str(row["error"]).splitlines()[-1][:70]}')
                broken.append(f'{row.get("arm_id")}|stage-error')
            continue
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'{arm_id:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            broken.append(f"{arm_id}|error")
            continue
        by_arm[arm_id] = row
        if row.get("refused_by_preflight"):
            print(f'{arm_id:<{ARM_COL}}{row["n_dev"]:>3}{row["form"]:>9}'
                  f'{(row.get("pixel_batch") or "-"):>8}'
                  f'{"-":>9}{"-":>11}{"REFUSED":>11}{"-":>10}{"-":>10}  '
                  "the memory preflight refused every layout")
            refused.append(arm_id)
            continue
        values = row.get("values") or {}
        peak_bytes = row.get("peak_max_bytes")
        median = row.get("forward_median_s")
        if row.get("kind") == "composed":
            median, spread = row.get("recon_s"), None
        else:
            spread = row.get("forward_spread_s")
        print(f'{arm_id:<{ARM_COL}}{row["n_dev"]:>3}{row["form"]:>9}'
              f'{(row.get("pixel_batch") or "-"):>8}'
              f'{row.get("realized_n_devices", "-"):>9}'
              f'{_fmt(values.get("rel"), 11, "e", 3)}'
              f'{_fmt(median, 11, "f", 2)}{_fmt(spread, 10, "f", 3)}'
              f'{_fmt((peak_bytes / 2 ** 30) if peak_bytes else None, 10, "f", 2)}  '
              f'{row.get("driver", "")}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    ARM CHECK FAIL: {reason}")
            broken.append(f"{arm_id}|check")
        if row.get("values_failed"):
            print(f'    VALUES GATE FAILED at rel {values.get("rel"):.3e} '
                  f'against {VALUES_GATE_REL:.0e}; timing skipped.  This is a '
                  "FINDING, not an instrument fault: read it, do not ignore it")
            values_failed.append(arm_id)
        if row.get("recon_shape_ok") is False:
            print(f'    NOTE: recon shape {row.get("recon_shape")} is not the '
                  f'registered {row.get("recon_shape_expected")}; a geometry '
                  "default has moved.  Recorded, not failed")
        if row.get("warm_repeat_rel") is not None:
            print(f'    warm-repeat floor: two warm passes of the same driver '
                  f'differ by {row["warm_repeat_rel"]:.3e} relative')
        for entry in row.get("cross_form") or []:
            kind = "same driver, repeat" if entry.get("same_form") else \
                "other driver"
            print(f'    vs {entry.get("other_arm"):<26} ({kind}): max-rel '
                  f'{_fmt(entry.get("rel"), 0, "e", 3)}, L2-rel '
                  f'{_fmt(entry.get("l2_rel"), 0, "e", 3)}')
    print("-" * len(header))

    # ── the comparison the run exists to make ────────────────────────────────
    print("\nbanded against gather, per geometry and device count "
          "(banded median / gather median; above 1.00 means the gather is "
          "faster):")
    for geometry in GEOMETRIES:
        spec = None
        for candidate in all_cells():
            if candidate["geometry"] == geometry:
                spec = candidate
        if spec is None:
            continue
        if not any(arm_id.startswith(spec["name"] + "_") for arm_id in by_arm):
            continue                    # this run selected no arm here
        anchor = by_arm.get(f'{spec["name"]}_gen_anchor') or {}
        anchor_median = anchor.get("forward_median_s")
        printed = False
        for n in counts():
            banded = by_arm.get(f'{spec["name"]}_n{n}_banded') or {}
            base = banded.get("forward_median_s")
            for batch in GATHER_BATCHES:
                gather = by_arm.get(f'{spec["name"]}_n{n}_gather{batch}') or {}
                other = gather.get("forward_median_s")
                if base is None or other is None:
                    continue
                printed = True
                print(f'  {geometry:<12} forward  n={n} batch {batch:<6} '
                      f'banded {base:7.3f}s  gather {other:7.3f}s  '
                      f'ratio {base / other:5.2f}x')
            composed_b = by_arm.get(
                f'{spec["name"]}_n{n}_composed_banded') or {}
            composed_g = by_arm.get(
                f'{spec["name"]}_n{n}_composed_gather{COMPOSED_BATCH}') or {}
            wb, wg = composed_b.get("recon_s"), composed_g.get("recon_s")
            if wb is not None and wg is not None:
                printed = True
                print(f'  {geometry:<12} composed n={n} batch '
                      f'{COMPOSED_BATCH:<6} banded {wb:7.1f}s  gather '
                      f'{wg:7.1f}s  ratio {wb / wg:5.2f}x')
        if anchor_median is not None:
            print(f'  {geometry:<12} one device, forward: '
                  f'{anchor_median:.3f}s')
        # The drift witness.  A banded-to-gather gap smaller than this gap
        # measures the machine, not the drivers.
        first = counts()[0]
        original = by_arm.get(f'{spec["name"]}_n{first}_banded') or {}
        repeat = by_arm.get(f'{spec["name"]}_n{first}_banded_repeat') or {}
        a, b = original.get("forward_median_s"), repeat.get("forward_median_s")
        if a is not None and b is not None:
            print(f'  {geometry:<12} drift witness: n={first} banded ran '
                  f'{a:.3f}s early in the block and {b:.3f}s at the end '
                  f'({abs(b - a) / a * 100:.1f}% apart).  Read every ratio '
                  "above against this")
        if not printed:
            print(f"  {geometry:<12} no pair completed")

    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}")
    if refused:
        print(f"\n{len(refused)} arm(s) refused by the memory preflight: "
              f"{refused}.  The multi-device arms model under 60 GiB of "
              "demand and the one-device anchors skip the preflight, so a "
              "refusal is information about the ledger, not an expected path.  "
              "The message is on the row.  This does not change the exit code")
    if values_failed:
        print(f"\n{len(values_failed)} arm(s) failed the values gate: "
              f"{values_failed}.  Their timings were skipped.  This does not "
              "change the exit code either -- it is the run's most important "
              "finding if it happened")
    healthy = not broken
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"} '
          f"({len(broken)} arm(s) did not measure what the plan said).  "
          "The verdict -- whether either geometry should switch its default -- "
          "is read by a person from the table above and the rows in the jsonl")
    return dict(healthy=healthy, broken=broken, refused=refused,
                values_failed=values_failed, hot=hot,
                arms=len([r for r in rows if r.get("mode") == "arm"]))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_cfg = json.loads(sys.argv[2])
        try:
            out = (stage_inputs(worker_cfg) if worker_cfg["mode"] == "stage"
                   else run_arm(worker_cfg))
        except Exception:                                         # noqa: BLE001
            out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    else:
        sys.exit(main())
