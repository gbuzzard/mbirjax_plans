"""mg54 -- THE FIRST COMPOSED SPEED READING OF THE TWO MULTIAXIS TRITON
KERNELS: THE KERNEL ROUTE AGAINST THE TORCH-BODY ROUTE, SAME STAGED INPUTS.

WHY THIS RUN EXISTS.  The multiaxis geometry now selects two hand-written
Triton kernels wherever their availability checks pass.  Those checks say one
thing only -- that each kernel REPRODUCES the torch body it replaces on the
device that will run it -- and the geometry's own ``_view_batch_bodies`` says
so at the selection: the tile constants were adopted from the cone kernels
rather than swept, and nothing has yet timed a multiaxis reconstruction with
the kernels bound.  So there is no composed speed measurement for them at all.
This run makes the first one.  Seeded 3-iteration reconstructions at the three
recorded multiaxis cells on one H100, the kernel route against the torch-body
route, the same staged sinograms and weights on both sides, warm medians.

THIS RUN DECIDES NOTHING.  It edits no library file and writes no kernel.  The
one thing it varies is MBIRTORCH_DISABLE_TRITON, per arm, in a fresh process,
which is the library's own kill switch and not a knob invented here.  The exit
code reports whether the instrument worked; the readings are read by a person.

THE SECOND READING: WHAT THE VIEW BATCH DID.  A separate measurement (mg53)
found that at the (1024, 1008, 992) cell the torch-body driver chose a view
batch of ONE view -- 1024 body calls per projection -- against 9 views (57 body
calls) at the (512, 448, 384) cell, and that most of one call's host time was
spent inside those 1024 body calls.  The batch is chosen by the cost model of
the BODY actually bound: a torch body is charged the gather transient the
driver prices itself, and a kernel body carries its own ``_view_batch_cost``
attribute stating its real charged residency and its own nominal chunk.  The
kernel bodies declare a much smaller per-view cost than that gather transient,
so the same driver should choose far fewer, far larger view batches with them
bound.  This run re-reads ONE projection call per direction on the kernel route
with the body-wrapper split, at the 1024-class cell, and records the body-call
count beside the driver's chosen batch.  That is the number that says whether
the one-view batches became large batches under the kernels' cost model.

TERMS, defined once here.
    cell        one problem to measure: a multiaxis geometry and a sinogram
                shape.
    staging     the untimed step that builds one cell's sinogram and weights
                once and writes them to one npz with an md5 beside it.
    route       which projection bodies the arm binds.  "kernel" is the
                default environment, where the availability checks decide;
                "torch" sets MBIRTORCH_DISABLE_TRITON=1 in the arm's
                environment, which is the library's kill switch.
    arm         one route reconstructing one cell, in its own process.
    cold pass   the first reconstruction in a process.  It pays whatever
                compiles the route needs, so it is timed and reported but
                never used for the ratio.
    warm pass   a reconstruction after the cold one.  MG54_WARM of them are
                timed and the row carries their median and their spread.
    fingerprint two float64 reductions of a reconstruction -- the sum of
                absolute values and the sum of squares -- accumulated on the
                DEVICE in float64, in fixed-size chunks, and read back.

THE CELLS, in run order, cheapest first, so a harness defect shows up in
minutes rather than after an hour.  All three are multiaxis:

    (512, 448, 384)
    (768, 672, 576)
    (1024, 1008, 992)

THE MODEL CONSTRUCTION IS MG52'S, unchanged, because the recorded one-device
walls this run's torch arms should reproduce were measured under it: two angles
per view, azimuths evenly spaced over half a turn,
np.linspace(0, pi, num_views, endpoint=False), and elevations swept across
+/- 0.5 radians, np.linspace(-0.5, 0.5, num_views), stacked into a
(num_views, 2) array.  The elevation range is part of what a multiaxis cell
measures: the geometry divides the detector height by the smallest
|cos(elevation)|, so a wider sweep would inflate the slice count.  The phantom
is generate_3d_shepp_logan_low_dynamic_range at the model's own recon shape,
with mg52's fallback to a seeded uniform volume when the phantom comes back all
zeros; the sinogram is that phantom forward projected, float32; the weights are
exp(-sinogram / (2 * max(sinogram))), float32.  Seed 13.

THE STAGED INPUTS ARE MG52'S, REUSED WHEN THEY ARE THERE.  mg52 staged exactly
these three multiaxis cells, to files named mg52_stage_multiaxis_<V>x<R>x<C>.npz
with an md5 sidecar, and its run record says they remain on scratch under
results/mg52_framework_anchor for repeat runs.  Its npz carries everything this
run needs and more -- sinogram, weights, view_params, sinogram_shape,
recon_shape, distances, geometry, name, delta_voxel, the two voxel aspects,
psf_radius and the phantom-fallback note -- so this run uses THE SAME FILENAMES
AND THE SAME KEYS and reads them with the same loader.  It searches its own
results directory first and then mg52's, verifies the md5 of whatever it finds,
and reuses it; only when no verified copy exists anywhere does it stage fresh,
under this run's own results directory, to the same filename.  Every staging
row says which of the two happened and where the bytes came from.

A FRESH STAGE RUNS WITH THE KERNELS OFF, deliberately.  Staging forward
projects a phantom to make the sinogram both arms then reconstruct.  mg52's
files were staged before the multiaxis kernels were routed, so their sinograms
came from the torch bodies; a fresh stage under this run's tree would come from
the kernels unless it is told otherwise, and the two would then not be the same
input across runs.  Both arms of a cell read one file either way, so this does
not touch the comparison -- it keeps a re-stage comparable with what mg52 left
behind.  The staging row records the setting it ran under.

THE PROTOCOL PER ARM, in a fresh process with a hard time cap: assert the
premise, then np.random.seed(13) immediately before every reconstruction, a
3-iteration reconstruction with the stopping threshold disabled as the cold
pass, then MG54_WARM more.  The peak counters are reset before the cold pass
and again before the warm passes, so the cold peak and the warm peak are two
separate readings on the row, both allocated and reserved.  The fingerprint is
taken from the LAST warm reconstruction, after the peaks are read, so the
reading's own device memory cannot land in the number it is reported beside.

THE PREMISE IS ASSERTED, NOT ASSUMED.  A kernel arm checks that
``model._view_batch_bodies()`` returns the two Triton wrappers -- names ending
_view_batch_triton -- and records what the two availability checks answered and
why.  A torch arm checks that both bodies are the torch ones.  An arm whose
premise fails records that and times NOTHING: an arm that measured the other
route under this route's name is the one way this comparison could be quietly
wrong.

FAILURE IS A READING.  An arm that runs out of device memory or hits the
per-arm cap is recorded as that arm's result and the run continues.  The exit
code separates those readings from instrument faults.

THE FINGERPRINTS ARE RECORDED, NOT GATED.  The kernels' own availability checks
already hold them to a relative tolerance against the torch bodies at a tiny
shape; what a composed fingerprint difference over three VCD iterations shows is
float summation order accumulated through a reconstruction, and a threshold
there would report arithmetic rather than correctness.  A relative difference
above 1e-3 puts a note on the comparison row and changes no exit code.

THE SINGLE-CALL LEG runs LAST, in the parent process, on the kernel route only,
at the (1024, 1008, 992) cell, and is skippable with MG54_SINGLE_CALL=0.  It
builds the model and its inputs as mg53 did -- explicit device, the shepp-logan
phantom, the host-side mask index, the sinogram made by the forward direction's
own warm-up call -- makes one untimed warm-up call per direction, and then one
measured call per direction with a host-clock wrapper over index 0 of the
projector's bound-body list.  It records the body-call count, the enqueue
seconds, the summed in-body host seconds, the wall, and the driver's chosen view
batch from the cell-setup bookkeeping.

  ONE DELIBERATE DIFFERENCE FROM MG53'S WRAPPER, and it is the whole reason
  this leg needs its own code.  mg53's wrapper must NOT carry a
  ``_view_batch_cost`` attribute, because the bodies it wrapped were torch
  bodies, which have none, and a wrapper that carried one would have selected a
  different batch.  Here the wrapped bodies are KERNEL bodies, which DO carry
  one, and the driver reads that attribute off the body it is about to call.  A
  bare closure would therefore drop the kernel's cost model, push the driver
  back onto the torch-body gather charge, and change the very batch this leg
  reports.  So the wrapper COPIES the attribute when the wrapped body has one
  and carries nothing when it does not, and the leg computes the effective view
  batch both with the wrapper installed and without it and puts both on the
  row.  The wrapper is still installed immediately before the measured call and
  removed immediately after, in a finally.

OUTPUT.  One jsonl under MG54_RESULTS, named
mg54_multiaxis_kernel_ab_<node>_<stamp>.jsonl: a header row carrying the torch
and mbirtorch identity, the GPU, the tree witnesses and the staged files' md5
sums; one row per staged cell; one row per arm; one comparison row per cell;
the single-call leg's setup row and its two body-split rows; and a summary row.
Rows are flushed as they are written, so a job that runs out of wall time still
yields everything it finished, and MG54_CELLS and MG54_ROUTES re-run the rest.

Run:
    <torch python> mg54_multiaxis_kernel_ab.py        on a one-GPU node
    MG54_DRY=1 <any python> mg54_multiaxis_kernel_ab.py    the plan, then stop
    MG54_SMOKE=1 <python> mg54_multiaxis_kernel_ab.py      the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized cell or route is an error, not a
silent skip.
    MG54_RESULTS=<dir>         where the jsonl and any fresh staging go
    MG54_CELLS=a,b             subset of the cells, by name
    MG54_ROUTES=kernel,torch   subset of the routes
    MG54_WARM=3                warm reconstructions after the cold pass
    MG54_ARM_TIMEOUT_MIN=45    the per-arm hard time cap, in minutes
    MG54_SINGLE_CALL=1         set 0 to skip the single-call leg
    MG54_DRY=1                 print the plan and exit; imports no torch
    MG54_SMOKE=1               the local CPU smoke
    MG54_CHILD=<path>          internal: an arm's job description.  Its
                               presence puts this process in child mode.
    MG54_CHILD_OUT=<path>      internal: where that child writes its row

THE LOCAL SMOKE runs the whole flow at one tiny cell on the CPU: the plan, the
staging and its md5, ONE subprocess arm so the child protocol is exercised, the
comparison, the single-call leg and the tables.  There is no triton on a CPU
install at all, so the kernel/torch route distinction DEGRADES there -- both
routes bind the torch bodies -- and the arm's row says so rather than pretending
otherwise.  The smoke is plumbing, not a measurement.
"""

import hashlib
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
        raise ValueError("{}: {!r} is not 0 or 1".format(name, raw))
    return raw == "1"


def _positive_int(name, default):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("{}: {!r} is not an integer".format(name, raw)) from None
    if value < 1:
        raise ValueError("{}: {} is not at least 1".format(name, value))
    return value


SMOKE = _flag("MG54_SMOKE")
DRY = _flag("MG54_DRY")
#: The arm's subprocess mode: the path to the job description this process is
#: to run.  Non-empty means child mode.  The sbatch unsets it, so a stray value
#: in the submitting shell cannot turn the real run into an arm.
CHILD = os.environ.get("MG54_CHILD", "").strip()
CHILD_OUT = os.environ.get("MG54_CHILD_OUT", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: The two routes, in run order.  "kernel" is the shipped default environment,
#: where the availability checks decide; "torch" sets the library's own kill
#: switch in the arm's environment.  That switch is the ONLY thing this run
#: varies, and it is varied in a fresh process because the availability probe
#: reads it once and caches the answer for the life of the process.
ROUTES = ("kernel", "torch")
ROUTE_DISABLE_TRITON = dict(kernel="0", torch="1")
#: The smoke plans ONE arm, so that the child protocol runs without the smoke
#: paying for a matrix it cannot tell apart: on a CPU install both routes bind
#: the torch bodies.
SMOKE_DEFAULT_ROUTES = ("kernel",)

#: (views, detector rows, channels).  The three multiaxis cells mg52 staged and
#: measured, so this run's torch arms lay beside numbers that already exist.
PRODUCTION_CELLS = (
    dict(name="multiaxis_512", cell=(512, 448, 384)),
    dict(name="multiaxis_768", cell=(768, 672, 576)),
    dict(name="multiaxis_1024", cell=(1024, 1008, 992)),
)
#: The smoke's stand-in.  128 views is two whole view batches at the torch
#: bodies' default nominal of 64, so the single-call leg's wrapper is exercised
#: per batch rather than once per call.  The detector is tiny, so the extra
#: views cost almost nothing.
SMOKE_CELLS = (
    dict(name="multiaxis_smoke", cell=(128, 16, 16)),
)

#: The cell the single-call leg runs at, by name.  In production that is the
#: 1024-class cell, where the torch-body driver was measured choosing a view
#: batch of one; in the smoke it is the only cell there is.  The leg is skipped
#: with a recorded reason when this cell is not in the plan.
SINGLE_CALL_CELL_NAME = "multiaxis_smoke" if SMOKE else "multiaxis_1024"
SINGLE_CALL = _flag("MG54_SINGLE_CALL", "1")
DIRECTIONS = ("forward", "back")

# ── the reconstruction protocol, taken from mg52 ──────────────────────────────
#: A seeded 3-iteration reconstruction with the stopping threshold disabled, so
#: both routes do exactly the same amount of work.  Not a knob: the recorded
#: walls this run's torch arms should reproduce were measured at three.
VCD_ITERATIONS = 3
#: The seed, reset immediately before every reconstruction on both routes.  The
#: library draws its pixel partitions from numpy's global generator, so this is
#: the same mechanism either way.
VCD_SEED = 13
WARM_REPEATS = _positive_int("MG54_WARM", 3)
#: The multiaxis elevation sweep, in radians, and the geometry's clamp on the
#: smallest |cos(elevation)|.  Both are mirrored here so the dry plan can print
#: the recon shape without importing anything; the real run reads the model's
#: own recon_shape and records whether the two agreed.
ELEVATION_HALF_RANGE = 0.5
MIN_COS_ELEVATION = 0.1

#: The per-arm hard time cap.  An arm that exceeds it is killed and the timeout
#: is recorded as that arm's result.
ARM_TIMEOUT_S = 60.0 * float(os.environ.get("MG54_ARM_TIMEOUT_MIN", "45"))
#: The identity probe imports torch and mbirtorch and reads the tree witnesses.
#: It does no real work, so ten minutes is generous even with a cold module
#: cache on a shared filesystem.
PROBE_TIMEOUT_S = 600.0
#: The staging job builds one cell's sinogram from a phantom and writes about
#: eight gigabytes at the largest cell, or verifies an existing file's md5.
#: Given its own cap so a slow filesystem does not eat an arm's budget.
STAGE_TIMEOUT_S = 3600.0

# ── how a body is recognized ──────────────────────────────────────────────────
#: Every hand-written multiaxis body's name ends in this.  The kernel wrappers
#: are _multiaxis_forward_view_batch_triton and _multiaxis_back_view_batch_triton
#: in mbirtorch/triton_multiaxis.py; the torch bodies they replace are
#: _multiaxis_forward_view_batch and _multiaxis_back_view_batch.
KERNEL_BODY_SUFFIX = "_view_batch_triton"
#: Where the projector keeps its per-device bound bodies, per direction.  The
#: single-device driver reads index 0 of the relevant list at every public call
#: and invokes it once per view batch.
BODY_LIST_ATTR = dict(forward="_fwd_body_per_dev", back="_back_body_per_dev")

# ── recorded context, not gates ───────────────────────────────────────────────
#: The one-device warm walls recorded for these three cells, in seconds, read in
#: session from mg52's run record (which quotes them as the recorded values its
#: own torch arms reproduced: 11.41 against 11.4, 56.30 against 56.3 and 310.06
#: against 309.9).  Printed beside this run's torch arms so a reader can see at
#: a glance whether the torch route landed where it has always landed.  Nothing
#: is gated on them.
RECORDED_TORCH_WALL_S = {
    "multiaxis_512": 11.4,
    "multiaxis_768": 56.3,
    "multiaxis_1024": 309.9,
}
#: The md5 sums mg52's run record lists for the three multiaxis staged files.
#: When a staged file this run reuses hashes to one of these, the two runs
#: reconstructed the same bytes, and the row says so.  A staged file that does
#: not match is not wrong -- a regenerated sinogram is never bit-identical --
#: so this is recorded and never gated.
MG52_RECORDED_STAGE_MD5 = {
    "multiaxis_512": "d148b0890904e138bbc5d7e5b06d3af8",
    "multiaxis_768": "bca9706523734478917d14cedee7c810",
    "multiaxis_1024": "798c72e1cf5bb7803b9f2b02294753c6",
}
#: What the torch-body driver was measured choosing at two of these cells, read
#: in session from mg53's run record: 9 views (57 body calls per projection) at
#: the 512-class cell and 1 view (1024 body calls) at the 1024-class.  The
#: single-call leg prints its own reading beside this one.  Recorded context,
#: never a threshold.
MG53_TORCH_VIEW_BATCH = {
    "multiaxis_512": dict(view_batch=9, body_calls=57),
    "multiaxis_1024": dict(view_batch=1, body_calls=1024),
}
#: Fingerprint differences above this print a NOTE.  Nothing fails on it.
FINGERPRINT_NOTE_LEVEL = 1e-3
#: Elements promoted to float64 at a time when a fingerprint is taken.  Eight
#: million float64 is 64 MiB, which bounds the reading's own device memory at
#: any volume size.
FINGERPRINT_CHUNK_ELEMS = 1 << 23

#: Substrings that mark an arm's failure as a CAPACITY reading rather than a
#: harness fault.  A cell that does not fit one device is an outcome, not a
#: broken instrument.  Matched case-insensitively against the traceback.
CAPACITY_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory",
                    "failed to allocate", "memoryerror", "cannot allocate")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("MG54_RESULTS",
                             os.path.join(SCRIPT_DIR, "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 26                  # wide enough for the longest arm id printed
NAME_COL = 40
# ──────────────────────────────────────────────────────────────────────────────


def stage_search_dirs():
    """Where a staged npz may already be, in search order.

    This run's own results directory first, then mg52's beside it, then the
    two local defaults a laptop smoke would have used.  mg52 wrote its staged
    files into its results directory, which on the cluster is a sibling of this
    run's, so the sibling lookup is what makes the reuse work without anybody
    naming a path.  Duplicates are dropped and order is kept.
    """
    candidates = [
        RESULTS_DIR,
        os.path.join(os.path.dirname(os.path.abspath(RESULTS_DIR)),
                     "mg52_framework_anchor"),
        os.path.join(SCRIPT_DIR, "results", "mg52_framework_anchor"),
        os.path.join(SCRIPT_DIR, "results"),
    ]
    seen, out = set(), []
    for path in candidates:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def cells():
    return SMOKE_CELLS if SMOKE else PRODUCTION_CELLS


def default_routes():
    return SMOKE_DEFAULT_ROUTES if SMOKE else ROUTES


def arm_id(spec, route):
    return "{}_{}".format(spec["name"], route)


def _strict_subset(env_name, allowed, default=None):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
    repeat before.  The error names the full valid list, because the caller who
    mistyped one id needs to see the others.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed if default is None else default)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError("{}: {!r} is not one of this run's: {}".format(
                env_name, token, ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError("{}: no valid tokens in {!r}.  The valid ones are: {}"
                         .format(env_name, raw, ", ".join(allowed)))
    # Normalized to the DECLARED order: the run order is load-bearing (cheapest
    # cell first, kernel before torch), so it must not depend on the order
    # somebody typed the tokens in.
    return [name for name in allowed if name in chosen]


def mirrored_recon_shape(cell):
    """The recon shape this cell should produce, derived from the geometry's
    own rule without importing anything.

    The rule is multiaxis_parallel's auto_set_recon_geometry: the in-plane
    extent comes from the channel coverage, the slice extent from the row
    coverage divided by the smallest |cos(elevation)|, clamped at 0.1.  With the
    detector pitches and voxel aspects at their defaults of 1.0 the whole rule
    is arithmetic, which is what lets the dry plan print it.  This is a MIRROR:
    every staged file carries the recon shape the model really realized, and the
    staging row records whether the two agreed.
    """
    _views, num_rows, num_channels = cell
    max_u = num_channels / 2.0
    max_v = num_rows / 2.0
    min_cos = max(math.cos(ELEVATION_HALF_RANGE), MIN_COS_ELEVATION)
    return (int(math.floor(2 * max_u)), int(math.floor(2 * max_u)),
            int(math.floor(2 * (max_v / min_cos))))


# ── the staged file: mg52's names, mg52's keys, mg52's loader ─────────────────
def stage_name(spec):
    """mg52's filename for one multiaxis cell, reproduced exactly so a file it
    left on scratch is found by name and a file this run writes is found by the
    next run of either."""
    return "mg52_stage_multiaxis_{}x{}x{}.npz".format(*spec["cell"])


def md5_path(path):
    return path + ".md5"


def file_md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: the largest of these npz files is
    about eight gigabytes and reading it whole to hash it would be wasteful."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def staged_present(path):
    return os.path.exists(path) and os.path.exists(md5_path(path))


def recorded_md5(path):
    with open(md5_path(path)) as handle:
        return handle.read().strip()


def find_staged(spec):
    """The first staged copy of this cell that exists with its md5 sidecar, or
    None.  Existence only -- the md5 is VERIFIED by whoever reads it."""
    name = stage_name(spec)
    for directory in stage_search_dirs():
        path = os.path.join(directory, name)
        if staged_present(path):
            return path
    return None


def stage_write_path(spec):
    """Where a fresh stage goes: this run's own results directory, under mg52's
    filename."""
    return os.path.join(RESULTS_DIR, stage_name(spec))


def read_staged(path, with_arrays=True):
    """The npz read itself, WITHOUT the md5 check.  Callers that have just
    hashed the file use this; everything else goes through ``load_staged``, so a
    file is never read twice to verify it once.

    ``with_arrays=False`` reads everything EXCEPT the sinogram and the weights.
    An npz member is only read when it is asked for, and at the largest cell
    those two are about eight gigabytes.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as handle:
        meta = dict(
            view_params=handle["view_params"],
            sinogram_shape=[int(v) for v in handle["sinogram_shape"]],
            recon_shape=[int(v) for v in handle["recon_shape"]],
            geometry=str(handle["geometry"].item()),
            name=str(handle["name"].item()),
            delta_voxel=float(handle["delta_voxel"]),
            voxel_row_aspect=float(handle["voxel_row_aspect"]),
            voxel_slice_aspect=float(handle["voxel_slice_aspect"]),
            psf_radius=int(handle["psf_radius"]),
            phantom_fallback=str(handle["phantom_fallback"].item()))
        if with_arrays:
            meta["sinogram"] = handle["sinogram"]
            meta["weights"] = handle["weights"]
    return meta


def load_staged(path, with_arrays=True):
    """Read one staged npz into a plain dict, after verifying its md5.

    Raises on a mismatch.  An arm that reconstructed different bytes than its
    sibling did not measure what the plan said, and a truncated read on a shared
    parallel filesystem is a recorded failure mode of this work.
    """
    expected = recorded_md5(path)
    actual = file_md5(path)
    if actual != expected:
        raise ValueError("the staged file at {} hashes to {}, not the recorded "
                         "{}".format(path, actual, expected))
    meta = read_staged(path, with_arrays=with_arrays)
    meta["md5"] = actual
    return meta


# ── the model, built mg52's way ───────────────────────────────────────────────
def view_params_for(cell):
    """One cell's per-view parameters, built the way mg52 builds them: two
    angles per view, azimuths over half a turn and elevations across
    +/- 0.5 radians.  Used only by staging; the arms read the staged copy, so
    both routes are handed the same numbers."""
    import numpy as np

    num_views = int(cell[0])
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.linspace(-ELEVATION_HALF_RANGE, ELEVATION_HALF_RANGE,
                            num_views)
    return np.stack([azimuth, elevation], axis=1)


def build_arm_model(sinogram_shape, view_params):
    """The multiaxis model an arm reconstructs with, built exactly as mg52
    builds it.

    On CUDA nothing is configured here: the count comes from
    MBIRTORCH_NUM_DEVICES, which this file sets to 1 in every job's environment
    and which keeps the model on the automatic branch where the memory preflight
    still runs.  The pin is a CUDA mechanism -- the policy short-circuits below
    two visible CUDA devices -- so the CPU smoke places its one device by hand,
    exactly as mg52 does.
    """
    import mbirtorch

    model = mbirtorch.MultiAxisParallelModel(
        tuple(int(v) for v in sinogram_shape), view_params)
    if DEVICE != "cuda":
        model.configure_devices(devices=[DEVICE])
    model.set_params(no_warning=True, verbose=0)
    return model


def build_single_call_model(cell):
    """The model the single-call leg uses, built as mg53 built it.

    The device is named EXPLICITLY here rather than left to the count: on the
    one-GPU allocation this job asks for, an explicit ['cuda:0'] realizes the
    single device, and naming it means the realized device is a fact on the row
    instead of a policy outcome.  ``skip_memory_preflight`` is set before the
    device is configured, because the preflight prices a whole reconstruction
    and this leg projects twice in each direction and never reconstructs.
    """
    import mbirtorch

    model = mbirtorch.MultiAxisParallelModel(
        tuple(int(v) for v in cell), view_params_for(cell))
    model.skip_memory_preflight = True
    model.configure_devices(
        devices=[DEVICE + (":0" if DEVICE == "cuda" else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def build_phantom(recon_shape):
    """mg52's phantom, as a host float32 array, with mg52's fallback.

    The shepp-logan builder places its ellipsoids as fractions of the volume,
    and on a volume only a few voxels deep every one of them can miss, leaving
    the phantom all zeros.  An all-zero phantom forward projects to an all-zero
    sinogram, so both arms would time a reconstruction of nothing.  The fallback
    is a seeded uniform volume, and the row records that it was used.
    """
    import numpy as np

    import mbirtorch

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    fallback = ""
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(VCD_SEED).rand(*recon_shape),
                             dtype=np.float32)
        fallback = "seeded uniform (shepp-logan returned all zeros)"
    return np.asarray(phantom, dtype=np.float32), fallback


def to_numpy(x):
    """The one host exit, used by staging.  A gathered container ALREADY
    returns numpy, so a gather is never followed by ``.detach()`` -- re-detaching
    one is a recorded way to lose rows."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, "gather", None)) and hasattr(x, "placement"):
        return x.gather()
    return (x.detach().cpu().numpy()
            if callable(getattr(x, "detach", None)) else np.asarray(x))


# ── the value fingerprint ─────────────────────────────────────────────────────
def _flat_tensor(volume, torch_module):
    """One flat torch view of a reconstruction, whatever ``recon`` handed back.

    ``recon`` returns a host numpy array in this library, and a gathered
    container returns numpy too, so this is normally a zero-copy wrap of host
    memory.  A device tensor is accepted and left where it is.
    """
    import numpy as np

    if isinstance(volume, torch_module.Tensor):
        return volume.detach().reshape(-1)
    if callable(getattr(volume, "gather", None)) and hasattr(volume, "placement"):
        volume = volume.gather()
    return torch_module.as_tensor(
        np.ascontiguousarray(np.asarray(volume))).reshape(-1)


def fingerprint(volume, torch_module, device):
    """Two float64 reductions of a reconstruction -- the sum of absolute values
    and the sum of squares -- accumulated ON THE DEVICE in float64 and read
    back.

    Two numbers rather than one, because a sum of absolute values alone cannot
    see a rearrangement that preserves magnitudes.  Accumulated in fixed-size
    chunks with plain ``torch.sum``: a float32 sum over a billion-element volume
    loses the digits this comparison needs, promoting a whole volume at once
    would double what the reading costs, and this library's own notes record
    that the one-line norm reductions are both inaccurate at scale and slow on
    some backends, so the chunked sum is the reduction to use.  Each chunk's
    partial is read back to the host immediately, so nothing but one chunk is
    ever resident.
    """
    flat = _flat_tensor(volume, torch_module)
    abs_sum = 0.0
    sq_sum = 0.0
    total = int(flat.numel())
    for start in range(0, total, FINGERPRINT_CHUNK_ELEMS):
        block = flat[start:start + FINGERPRINT_CHUNK_ELEMS].to(
            device=device, dtype=torch_module.float64)
        abs_sum += float(torch_module.sum(torch_module.abs(block)))
        sq_sum += float(torch_module.sum(block * block))
        block = None
    return abs_sum, sq_sum, total


def relative_gap(value, reference):
    """|value - reference| / |reference|, with a zero reference reported as an
    absolute gap rather than as infinity."""
    if value is None or reference is None:
        return None
    scale = abs(reference)
    return abs(value - reference) / (scale if scale > 0.0 else 1.0)


def _is_oom(exc):
    """Whether an exception is a device out-of-memory.

    The class name is checked as well as the message because torch raises its
    own OutOfMemoryError on some paths and a plain RuntimeError on others, and
    a leg that mistook one for a real failure would kill a job that was only too
    big for one card.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(exc).lower()


def git_identity(path):
    """The commit a checkout sits at, and whether it is dirty.  ``None`` when
    the directory is not a git checkout or git is unavailable, which is a
    recorded state rather than an error: an rsynced export has no commit."""
    out = dict(commit=None, dirty=None, root=path)
    try:
        proc = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            out["commit"] = proc.stdout.strip()
    except Exception:                                             # noqa: BLE001
        return out
    try:
        proc = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            out["dirty"] = bool(proc.stdout.strip())
    except Exception:                                             # noqa: BLE001
        pass
    return out


# ── the tree under test ───────────────────────────────────────────────────────
def tree_witnesses():
    """What tree produced these numbers, measured rather than asserted.

    The first three are mg53's block, unchanged: they say this is the padded,
    recompile-remedied tree the recorded one-device walls were measured on, and
    the third matters directly here because a tree without it would hand the
    torch arms eager python rather than compiled bodies.

    The last two are this run's own, and they are what makes the kernel route a
    real route rather than a name.  The kernel module must import -- it is
    written to import without triton, so this works on a laptop too -- and the
    geometry's ``_view_batch_bodies`` must consult the two availability
    functions and reach the two Triton wrappers.  Both are read by SOURCE
    INSPECTION and by attribute lookup: no model is built, no device is touched,
    and no CUDA is initialized, so the witness block costs nothing and can run
    anywhere.

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

        from mbirtorch import triton_multiaxis
        from mbirtorch.multiaxis_parallel import MultiAxisParallelModel

        record["triton_multiaxis_file"] = triton_multiaxis.__file__
        # The two kernel wrappers exist, are named the way the premise check
        # recognizes them, and carry the per-view cost attribute the driver
        # reads to choose a view batch.  That attribute is the whole reason the
        # single-call leg expects a different batch on this route.
        kernels = {}
        for direction, attr in (("forward",
                                 "_multiaxis_forward_view_batch_triton"),
                                ("back",
                                 "_multiaxis_back_view_batch_triton")):
            body = getattr(triton_multiaxis, attr, None)
            kernels[direction] = dict(
                present=body is not None,
                name=getattr(body, "__name__", None),
                named_like_a_kernel=bool(
                    getattr(body, "__name__", "").endswith(
                        KERNEL_BODY_SUFFIX)),
                has_view_batch_cost=getattr(body, "_view_batch_cost",
                                            None) is not None)
        record["multiaxis_kernels"] = kernels
        selection = inspect.getsource(
            MultiAxisParallelModel._view_batch_bodies)
        record["selection_consults_availability"] = bool(
            "multiaxis_forward_kernel_usable" in selection
            and "multiaxis_back_kernel_usable" in selection)
        record["selection_reaches_kernels"] = bool(
            "_multiaxis_forward_view_batch_triton" in selection
            and "_multiaxis_back_view_batch_triton" in selection)

        record["ok"] = bool(
            record["padded_kernel_width_504"] == 512
            and record["recompile_limit_floor"] >= 64
            and record["raise_on_compiling_thread"]
            and all(entry["present"] and entry["named_like_a_kernel"]
                    and entry["has_view_batch_cost"]
                    for entry in kernels.values())
            and record["selection_consults_availability"]
            and record["selection_reaches_kernels"])
    except Exception as exc:                                      # noqa: BLE001
        record.update(available=False, ok=False,
                      reason="{}: {}".format(type(exc).__name__, exc))
    return record


# ── the premise: which bodies did this model really bind ──────────────────────
def route_expectation(route):
    """What an arm on ``route`` must have bound before it may time anything,
    and the reason when that is not what the route's name says.

    On a CPU install there is no triton at all, so the kernel route binds the
    torch bodies and the distinction between the two routes DEGRADES to
    nothing.  The smoke says that on the row rather than pretending otherwise;
    what the smoke exercises is the child protocol, not the kernels.
    """
    if DEVICE != "cuda":
        return "torch", ("this run is on {}, where there is no triton at all, "
                         "so both routes bind the torch bodies and the route "
                         "distinction is degraded".format(DEVICE))
    return route, ""


def bound_body_report(model):
    """Which bodies this model bound, what the availability checks answered,
    and why.

    ``_view_batch_bodies`` is the geometry's selection hook and returns the two
    plain functions, before compilation and before any per-device binding, so
    their names are readable whatever torch did with them afterwards.  The two
    availability functions are asked separately, because the geometry asks them
    separately and a machine may bind one kernel and keep the other direction's
    torch body; their reasons are the record of WHY a node is not using a
    kernel, which is the first question anybody reading these walls will ask.
    """
    from mbirtorch import _memory_ledger
    from mbirtorch.kernel_availability import (multiaxis_back_kernel_usable,
                                               multiaxis_forward_kernel_usable)

    fwd_body, back_body = model._view_batch_bodies()
    fwd_ok, fwd_reason = multiaxis_forward_kernel_usable(model)
    back_ok, back_reason = multiaxis_back_kernel_usable(model)
    names = dict(forward=fwd_body.__name__, back=back_body.__name__)
    report = dict(
        forward_body=names["forward"], back_body=names["back"],
        forward_kernel_available=bool(fwd_ok),
        back_kernel_available=bool(back_ok),
        forward_availability_reason=str(fwd_reason),
        back_availability_reason=str(back_reason),
        # A kernel body carries a per-view cost attribute and general torch
        # code carries none, so this is the same question asked a second way,
        # through the library's own reader rather than through a name.
        torch_body_directions=list(
            _memory_ledger.torch_body_directions(model)),
        env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
        compile_enabled=bool(model.compile_enabled),
        compile_mode=str(model.compile_mode))
    report["kernels_bound"] = all(
        name.endswith(KERNEL_BODY_SUFFIX) for name in names.values())
    report["torch_bodies_bound"] = (
        report["torch_body_directions"] == list(DIRECTIONS))
    return report


def check_premise(report, route):
    """Whether an arm on ``route`` bound what its name says, and the sentence
    that goes on the row when it did not.

    An arm whose premise fails times NOTHING.  An arm that measured the other
    route under this route's name is the one way this comparison could be
    quietly wrong, and a wrong number that looks right is worse here than no
    number at all.
    """
    expected, degraded_reason = route_expectation(route)
    if expected == "kernel":
        ok = report["kernels_bound"]
        reason = ("" if ok else
                  "the kernel route bound {} and {}, and a kernel body's name "
                  "ends in {}.  The availability checks said: forward {}, "
                  "back {}".format(report["forward_body"], report["back_body"],
                                   KERNEL_BODY_SUFFIX,
                                   report["forward_availability_reason"],
                                   report["back_availability_reason"]))
    else:
        ok = report["torch_bodies_bound"] and not report["kernels_bound"]
        reason = ("" if ok else
                  "this arm was to bind the torch bodies in both directions "
                  "and bound {} and {}; the library reports torch bodies in {}"
                  .format(report["forward_body"], report["back_body"],
                          report["torch_body_directions"] or "neither "
                          "direction"))
    return ok, reason, expected, degraded_reason


# ── the workers: the identity probe, one staging job, or one arm ──────────────
def run_identity(cfg):
    """Which library this run is about to measure, and on what.

    Run before anything else and in its own process, so the header row can name
    the torch version, the device, the tree under test and its witnesses without
    the driver importing torch at all until the single-call leg.
    """
    import torch

    import mbirtorch

    row = dict(cfg)
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row.update(torch_version=torch.__version__,
               library_file=mbirtorch.__file__,
               device_count=(torch.cuda.device_count() if cuda else 1),
               device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
               cuda=cuda,
               triton_version=None,
               python=platform.python_version())
    try:
        import triton
        row["triton_version"] = triton.__version__
    except Exception as exc:                                      # noqa: BLE001
        row["triton_version"] = "{}: {}".format(type(exc).__name__, exc)
    package_root = os.path.dirname(os.path.dirname(
        os.path.abspath(mbirtorch.__file__)))
    row["git"] = git_identity(package_root)
    row["tree_witnesses"] = tree_witnesses()
    return row


def run_stage(cfg):
    """Build ONE cell's sinogram and weights, once, and write them to one npz
    with an md5 sidecar -- or verify and reuse the copy already on disk.

    Both arms of the cell then load that same file, so the two routes
    reconstruct identical input and the comparison between them is controlled
    rather than incidental.  A file already on disk is VERIFIED rather than
    rebuilt: a regenerated sinogram is not bit-identical, so a rebuild would
    silently change what both arms of this cell reconstruct and would break the
    tie to what mg52 measured on the same bytes.
    """
    import numpy as np

    spec = cfg["spec"]
    row = dict(cfg)
    cell = tuple(int(v) for v in spec["cell"])
    row["mirrored_recon_shape"] = list(mirrored_recon_shape(cell))
    row["mg52_recorded_md5"] = MG52_RECORDED_STAGE_MD5.get(spec["name"])

    found = find_staged(spec)
    if found is not None:
        expected = recorded_md5(found)
        actual = file_md5(found)
        row.update(stage_path=found, reused=True, md5=actual,
                   md5_ok=(actual == expected), recorded_md5=expected,
                   bytes_on_disk=os.path.getsize(found),
                   reused_from=os.path.dirname(os.path.abspath(found)),
                   same_bytes_as_mg52=(actual == row["mg52_recorded_md5"]))
        if actual == expected:
            # Hashed once, just above; read the metadata only -- the sinogram
            # and the weights are read by the arms, not here.
            meta = read_staged(found, with_arrays=False)
            row.update(recon_shape=meta["recon_shape"],
                       sinogram_shape=meta["sinogram_shape"],
                       geometry=meta["geometry"],
                       delta_voxel=meta["delta_voxel"],
                       psf_radius=meta["psf_radius"],
                       phantom_fallback=meta["phantom_fallback"])
            row["recon_shape_mirror_agrees"] = (
                meta["recon_shape"] == row["mirrored_recon_shape"])
            row["shape_ok"] = (meta["sinogram_shape"] == list(cell)
                               and meta["geometry"] == "multiaxis")
            if not row["shape_ok"]:
                row["invalid_reasons"] = [
                    "the staged file at {} holds a {} sinogram of shape {}, "
                    "not a multiaxis {}".format(found, meta["geometry"],
                                                meta["sinogram_shape"],
                                                list(cell))]
        else:
            row["invalid_reasons"] = [
                "the staged file at {} hashes to {}, not the recorded {}"
                .format(found, actual, expected)]
        return row

    # Nothing verified anywhere, so build it here, under this run's own results
    # directory and under mg52's filename.
    path = stage_write_path(spec)
    view_params = np.asarray(view_params_for(cell), dtype=np.float32)
    model = build_arm_model(cell, view_params)
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    phantom, phantom_fallback = build_phantom(recon_shape)

    sinogram = np.ascontiguousarray(
        np.asarray(to_numpy(model.forward_project(phantom)), dtype=np.float32))
    # mg52's weighting formula.  copy=False only skips a repeat of an array
    # that is already float32; the values are the same ones that expression has
    # always produced, and at the largest cell each avoided copy is four
    # gigabytes of host memory.
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32,
                                                                copy=False)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(path, sinogram=sinogram, weights=weights,
             view_params=view_params,
             sinogram_shape=np.asarray(cell, dtype=np.int64),
             recon_shape=np.asarray(recon_shape, dtype=np.int64),
             distances=np.asarray((), dtype=np.float64),
             geometry=np.asarray("multiaxis"), name=np.asarray(spec["name"]),
             delta_voxel=np.asarray(float(model.get_params("delta_voxel"))),
             voxel_row_aspect=np.asarray(
                 float(model.get_params("voxel_row_aspect"))),
             voxel_slice_aspect=np.asarray(
                 float(model.get_params("voxel_slice_aspect"))),
             psf_radius=np.asarray(int(model.get_psf_radius())),
             phantom_fallback=np.asarray(phantom_fallback))
    digest = file_md5(path)
    with open(md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    body = bound_body_report(model)
    row.update(stage_path=path, reused=False, md5=digest, md5_ok=True,
               shape_ok=True, recon_shape=list(recon_shape),
               sinogram_shape=list(cell), geometry="multiaxis",
               recon_shape_mirror_agrees=(
                   list(recon_shape) == row["mirrored_recon_shape"]),
               delta_voxel=float(model.get_params("delta_voxel")),
               psf_radius=int(model.get_psf_radius()),
               phantom_fallback=phantom_fallback,
               phantom_max=float(np.max(phantom)),
               bytes_on_disk=os.path.getsize(path),
               same_bytes_as_mg52=(digest == row["mg52_recorded_md5"]),
               # Which bodies made these bytes.  A fresh stage runs with the
               # kernels off, so this should name the torch bodies; the row
               # carries the answer rather than the intention.
               stage_bodies=body)
    return row


def run_arm(cfg):
    """One route reconstructing one cell: assert the premise, then a cold pass
    and MG54_WARM timed warm passes."""
    import numpy as np
    import torch

    import mbirtorch

    route = cfg["route"]
    path = cfg["stage_path"]
    meta = load_staged(path)              # raises on an md5 mismatch
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row = dict(cfg, stage_path=path, staged_md5=meta["md5"],
               staged_recon_shape=meta["recon_shape"],
               phantom_fallback=meta["phantom_fallback"],
               vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
               warm_repeats=WARM_REPEATS, invalid_reasons=[],
               torch_version=torch.__version__, library_file=mbirtorch.__file__,
               device=DEVICE, cuda=cuda,
               device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
               visible_devices=(torch.cuda.device_count() if cuda else 1),
               env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
               env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
               recorded_torch_wall_s=RECORDED_TORCH_WALL_S.get(cfg["name"]))

    model = build_arm_model(meta["sinogram_shape"], meta["view_params"])
    realized = [int(s) for s in model.get_params("recon_shape")]
    row["recon_shape"] = realized
    row["recon_shape_ok"] = (realized == list(meta["recon_shape"]))
    if not row["recon_shape_ok"]:
        row["invalid_reasons"].append(
            "this arm's model realized recon shape {}, but the staged cell was "
            "built at {}".format(realized, meta["recon_shape"]))
        return row

    # ── the premise, before any clock starts ─────────────────────────────────
    report = bound_body_report(model)
    row.update(report)
    ok, reason, expected, degraded_reason = check_premise(report, route)
    row.update(premise_ok=ok, route_expected_bodies=expected,
               route_degraded=bool(degraded_reason),
               route_degraded_reason=degraded_reason)
    if not ok:
        row["invalid_reasons"].append(reason)
        # Nothing is timed.  A row that carried a wall taken on the other
        # route's bodies would be worse than this row, which carries none.
        return row

    # The view batch the driver will choose, per direction, by its own rule and
    # off the bodies this arm actually bound.  It is recorded rather than
    # derived, because it is the quantity the kernels' own cost model changes.
    try:
        args = model._view_batch_args()
        indices = model.full_indices_device()
        num_pixels = int(indices.shape[0])
        pf = model.projector_functions
        row["num_pixels"] = num_pixels
        row["view_batch"] = dict(
            forward=int(pf._effective_view_batch(
                pf._fwd_body_per_dev[0], num_pixels, int(realized[2]), args)),
            back=int(pf._effective_view_batch(
                pf._back_body_per_dev[0], num_pixels,
                int(meta["sinogram_shape"][1]), args)))
        row["view_batch_iterations"] = {
            name: int(math.ceil(meta["sinogram_shape"][0] / max(1, batch)))
            for name, batch in row["view_batch"].items()}
        indices = None
    except Exception as exc:                                      # noqa: BLE001
        row["view_batch"] = None
        row["view_batch_error"] = "{}: {}".format(type(exc).__name__, exc)

    sinogram, weights = meta["sinogram"], meta["weights"]

    def one_recon():
        """One reconstruction, exactly as mg52 calls it."""
        np.random.seed(VCD_SEED)
        out, _info = model.recon(sinogram, weights=weights,
                                 max_iterations=VCD_ITERATIONS,
                                 stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return out

    def peaks(devices):
        """The two peak counters, both recorded: allocated is what the library
        asked for and reserved is what the caching allocator kept from the
        driver.  A route that fragments differently moves the second without
        moving the first, so neither stands in for the other."""
        if not cuda:
            return None, None
        return (max(int(torch.cuda.max_memory_allocated(d)) for d in devices),
                max(int(torch.cuda.max_memory_reserved(d)) for d in devices))

    if cuda:
        for device in model.sino_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out = one_recon()
    row["cold_s"] = time.perf_counter() - start

    # The layout has settled, so this describes the layout the timed passes
    # actually run on.
    devices_now = list(model.recon_placement.devices)
    row["realized_devices"] = [str(d) for d in devices_now]
    row["realized_n_devices"] = len(devices_now)
    fingerprint_device = devices_now[0] if devices_now else DEVICE

    # The cold peaks are read BEFORE the counters are reset, so the row carries
    # the cold reading and the warm reading as two separate numbers.
    row["cold_peak_bytes"], row["cold_reserved_bytes"] = peaks(devices_now)
    if cuda:
        for device in devices_now:
            torch.cuda.reset_peak_memory_stats(device)

    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = one_recon()
        warm.append(time.perf_counter() - start)
    row["warm_all"] = warm
    row["warm_s"] = statistics.median(warm)
    row["warm_min"] = min(warm)
    row["warm_max"] = max(warm)
    row["warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    row["warm_peak_bytes"], row["warm_reserved_bytes"] = peaks(devices_now)
    if cuda:
        # The comparison uses the PROCESS-LIFETIME peak, which is the larger of
        # the two readings: a route whose cold pass costs more than its warm
        # passes still needed that much device memory to run at all.
        row["peak_bytes"] = max(row["warm_peak_bytes"], row["cold_peak_bytes"])
        row["reserved_bytes"] = max(row["warm_reserved_bytes"],
                                    row["cold_reserved_bytes"])
    else:
        row["peak_bytes"] = None
        row["reserved_bytes"] = None
    row["peak_kind"] = ("torch.cuda.max_memory_allocated and "
                        "max_memory_reserved, reset before the cold pass and "
                        "again before the warm passes")

    # AFTER the peaks are read: the fingerprint promotes chunks to float64 on
    # the device, and a reading that landed inside the number it is reported
    # beside would be a measurement of itself.
    abs_sum, sq_sum, elements = fingerprint(out, torch, fingerprint_device)
    row["fingerprint_abs_sum"] = abs_sum
    row["fingerprint_sq_sum"] = sq_sum
    row["fingerprint_elements"] = elements
    row["fingerprint_where"] = "float64 chunked sums on {}".format(
        fingerprint_device)
    return row


def run_job(cfg):
    started = time.time()
    if cfg["kind"] == "identity":
        row = run_identity(cfg)
    elif cfg["kind"] == "stage":
        row = run_stage(cfg)
    else:
        row = run_arm(cfg)
    row["worker_wall_s"] = time.time() - started
    return row


# ── the driver ────────────────────────────────────────────────────────────────
def job_env(cfg):
    """The environment that DEFINES one job, set explicitly so nothing leaks in
    from the submitting shell.

    MBIRTORCH_DISABLE_TRITON is the one variable under test and is set here,
    per job, never inherited: a value left in the shell would make both arms of
    a cell the same arm under two names.  The staging job gets it set to 1 --
    the sinogram both arms then reconstruct is a torch-body forward projection,
    which is what mg52's staged files hold and what makes a re-stage comparable
    with them.

    PYTHONPATH IS INHERITED, and this is where this file differs from mg52's
    job environment, which pops it.  mg52 selected two libraries by two
    interpreters, so a stray PYTHONPATH could have crossed them.  Here both
    routes are the SAME library in the same interpreter, and on the cluster that
    library is a candidate tree reached only through PYTHONPATH; popping it
    would silently run the installed tree instead, which does not route these
    kernels at all.  Every job records the mbirtorch file it actually imported,
    so which tree ran is a fact on the row rather than an assumption.
    """
    env = dict(os.environ)
    env.pop("MG54_DRY", None)           # a worker never prints a plan
    env.pop("MG54_CELLS", None)         # a worker runs its cfg, not a plan
    env.pop("MG54_ROUTES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)  # it owns the peak counters
    env.pop("MBIRTORCH_WIDENING_GUARD", None)      # the pin bypasses it
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    if DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = "1"
    if cfg["kind"] == "arm":
        env["MBIRTORCH_DISABLE_TRITON"] = ROUTE_DISABLE_TRITON[cfg["route"]]
    elif cfg["kind"] == "stage":
        env["MBIRTORCH_DISABLE_TRITON"] = "1"
    else:
        env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def spawn(cfg, timeout_s):
    """Run one job in a NEW interpreter, with a hard time cap.

    A new process per job is not tidiness.  The availability probe and the
    per-device value self-checks are cached for the life of a process, and the
    kill switch that separates the two routes is read once when the probe first
    runs; compiled bodies and allocator state are cached the same way.  The row
    travels through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR,
                            "_mg54_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR,
                            "_mg54_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = job_env(cfg)
    env["MG54_CHILD"] = cfg_path
    env["MG54_CHILD_OUT"] = out_path
    start = time.perf_counter()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                              env=env, timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # The cap is a READING, not a fault: an arm that cannot finish inside it
        # has told us something about the route at this size.
        timed_out = True
    wall = time.perf_counter() - start
    if timed_out:
        row = dict(cfg, error="timed out after {:.1f} min".format(
            timeout_s / 60.0), timed_out=True)
    elif not os.path.exists(out_path):
        # A job killed by the out-of-memory killer lands here, having written
        # nothing.  That is a reading too, and the classifier below decides.
        row = dict(cfg, error="worker exited {} and wrote no row"
                   .format(returncode), worker_returncode=returncode)
    else:
        with open(out_path) as handle:
            row = json.load(handle)
        row["worker_returncode"] = returncode
    row["subprocess_wall_s"] = wall
    return row


def is_capacity_reading(row):
    """Whether an arm's failure is a capacity or timeout READING rather than a
    harness fault.  A cell that does not fit one device is an outcome, so it
    must not be reported as a broken instrument."""
    if row.get("timed_out"):
        return True
    text = str(row.get("error", "")).lower()
    if not text:
        return False
    if any(marker in text for marker in CAPACITY_MARKERS):
        return True
    # A worker killed outright by the operating system's memory manager writes
    # no traceback at all.  Signal 9 with no row is the shape that leaves.
    return "wrote no row" in text and str(row.get("worker_returncode")) == "-9"


# ── the single-call leg ───────────────────────────────────────────────────────
def _install_body_timer(projector_functions, direction, durations):
    """Put a host-clock wrapper over the bound body at index 0 and return the
    body it replaced.

    The driver reads its body out of these lists at every public call, so
    replacing index 0 is enough and nothing else has to be touched.  What the
    wrapper measures is the host time for the body call to RETURN, which for an
    asynchronous device workload is that body call's share of the enqueue cost.

    THE ONE THING THIS WRAPPER DOES THAT MG53'S MUST NOT.  The driver reads a
    ``_view_batch_cost`` attribute off the body it is about to call, with a
    default of None, and takes the torch-body gather-transient branch only when
    it is absent.  mg53 wrapped TORCH bodies, which have no such attribute, so
    its wrapper had to carry none either.  The bodies here may be KERNEL bodies,
    which do carry one, and a bare closure would drop it, push the driver onto
    the torch charge, and change the very view batch this leg reports.  So the
    attribute is copied when the wrapped body has one, and nothing is copied
    when it does not.  Only that one attribute: a functools.wraps wrapper would
    copy everything, including the no-compile marker, which is read somewhere
    else entirely.

    The wrapper must also not swallow or convert the body's return value,
    because the driver assigns and accumulates it.
    """
    bodies = getattr(projector_functions, BODY_LIST_ATTR[direction])
    original = bodies[0]

    def timed_body(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        durations.append(time.perf_counter() - start)
        return result

    cost = getattr(original, "_view_batch_cost", None)
    if cost is not None:
        timed_body._view_batch_cost = cost
    bodies[0] = timed_body
    return original, cost is not None


def _restore_body(projector_functions, direction, original):
    getattr(projector_functions, BODY_LIST_ATTR[direction])[0] = original


class SingleCallContext:
    """The model and the two projection inputs the single-call leg uses, built
    as mg53 built them.

    Building the inputs is the expensive part at the 1024-class cell -- a
    phantom the size of the volume, a fancy-index down to the mask, and one
    forward projection -- so they are built once and both directions use them.
    The forward direction's WARM-UP call produces the sinogram the back
    projection is given, so nothing is projected twice for bookkeeping.
    """

    def __init__(self, spec):
        self.spec = spec
        self.cell = tuple(int(x) for x in spec["cell"])
        self.name = spec["name"]
        self.model = None
        self.indices = None
        self.voxel_values = None
        self.sinogram = None
        self.record = dict(row="single_call_setup", name=self.name,
                           cell=list(self.cell), route="kernel")

    def build(self):
        """Model, premise, mask, voxel values, and the driver's own view-batch
        bookkeeping.  The sinogram is made later, by the forward direction's
        warm-up call."""
        import numpy as np
        import torch

        start = time.perf_counter()
        self.model = build_single_call_model(self.cell)
        model = self.model
        self.record["device_realized"] = str(model.torch_device)
        self.record["device_expected"] = (
            "cuda:0" if DEVICE == "cuda" else DEVICE)
        self.record["device_ok"] = (self.record["device_realized"]
                                    == self.record["device_expected"])

        report = bound_body_report(model)
        self.record.update(report)
        ok, reason, expected, degraded = check_premise(report, "kernel")
        self.record.update(premise_ok=ok, route_expected_bodies=expected,
                           route_degraded=bool(degraded),
                           route_degraded_reason=degraded,
                           premise_reason=reason)

        recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
        phantom, fallback = build_phantom(recon_shape)
        self.record["recon_shape"] = list(recon_shape)
        self.record["mirrored_recon_shape"] = list(
            mirrored_recon_shape(self.cell))
        self.record["recon_shape_mirror_agrees"] = (
            list(recon_shape) == list(mirrored_recon_shape(self.cell)))
        self.record["phantom_fallback"] = fallback

        self.indices = model.full_indices_device()
        num_pixels = int(self.indices.shape[0])
        self.record["num_pixels"] = num_pixels
        # The mask index is taken on the HOST.  Indexing the whole volume on the
        # device would hold the volume and its gathered subset at once, which is
        # the largest single allocation this leg would ever make and is
        # avoidable: the gathered subset is what the projector wants.
        idx_host = self.indices.detach().cpu().numpy()
        flat = phantom.reshape(-1, recon_shape[2])
        values_host = np.ascontiguousarray(flat[idx_host])
        phantom = None
        flat = None
        self.voxel_values = torch.as_tensor(values_host, dtype=torch.float32,
                                            device=model.torch_device)
        values_host = None
        self.record["voxel_values_bytes"] = int(
            self.voxel_values.numel()) * 4
        self.record["sinogram_bytes"] = int(
            self.cell[0] * self.cell[1] * self.cell[2] * 4)

        args = model._view_batch_args()
        pf = model.projector_functions
        # How many views one body call takes, per direction, by the driver's own
        # rule and off the bodies actually bound.  The measured leg counts body
        # calls, and this is what turns that count into calls per view batch
        # without inferring it.
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
        self.record["torch_body_view_batch_recorded"] = (
            MG53_TORCH_VIEW_BATCH.get(self.name))
        self.record["setup_s"] = time.perf_counter() - start
        return self.record

    def call(self, direction):
        """One public-route projection call in ``direction``.  This is the
        funnel the reconstruction itself calls: every sparse projection in the
        library routes through these two methods."""
        model = self.model
        if direction == "forward":
            return model.sparse_forward_project(self.voxel_values,
                                                self.indices)
        return model.sparse_back_project(self.sinogram, self.indices)

    def warm_up(self, direction):
        """The untimed call the measured call runs after.

        It pays whatever first-call cost this route has -- a compile on the
        torch bodies, a first kernel launch and its autotune on the kernels --
        and the first-touch allocations, none of which the measured call is
        about.  In the forward direction its output is KEPT: it is the sinogram
        the back projection is given.
        """
        import numpy as np
        import torch

        np.random.seed(VCD_SEED)
        start = time.perf_counter()
        out = self.call(direction)
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.model.torch_device)
        record = dict(warm_up_s=time.perf_counter() - start,
                      output_shape=[int(x) for x in out.shape])
        if direction == "forward":
            self.sinogram = out
        return record

    def release(self):
        import torch

        self.voxel_values = None
        self.sinogram = None
        self.indices = None
        self.model = None
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def body_split_call(context, direction):
    """ONE measured projection call with the body wrapper installed.

    The wrapper goes on immediately before the call and comes off immediately
    after, in a finally, because the projector object is rebuilt on
    reconfiguration and a wrapper left behind would time calls nothing asked
    for.

    The row carries the body-call count beside the driver's chosen view batch,
    which together are the reading this leg exists for: whether the one-view
    batches the torch bodies were forced into at this cell became large batches
    under the kernels' own cost model.  The effective view batch is computed
    BOTH ways -- with the wrapper installed and with the original body -- so the
    row can say for itself that the wrapper did not change what it measured.
    """
    import numpy as np
    import torch

    model = context.model
    device = model.torch_device
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pf = model.projector_functions
    durations = []
    row = dict(row="body_split", name=context.name, cell=list(context.cell),
               direction=direction, route="kernel",
               body_list=BODY_LIST_ATTR[direction],
               alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    args = model._view_batch_args()
    num_pixels = int(context.indices.shape[0])
    band_cols = (int(model.get_params("recon_shape")[2]) if direction == "forward"
                 else int(context.cell[1]))
    row["setup_view_batch"] = (context.record.get("view_batch") or {}).get(
        direction)

    if cuda:
        torch.cuda.reset_peak_memory_stats(device)

    original, cost_copied = _install_body_timer(pf, direction, durations)
    row["wrapper_carries_view_batch_cost"] = cost_copied
    try:
        # Read with the wrapper in place, which is the batch the measured call
        # is actually about to use.
        row["wrapped_view_batch"] = int(pf._effective_view_batch(
            getattr(pf, BODY_LIST_ATTR[direction])[0], num_pixels, band_cols,
            args))
        row["unwrapped_view_batch"] = int(pf._effective_view_batch(
            original, num_pixels, band_cols, args))
        np.random.seed(VCD_SEED)
        start = time.perf_counter()
        out = context.call(direction)
        enqueue = time.perf_counter() - start
        if cuda:
            torch.cuda.synchronize(device)
        wall = time.perf_counter() - start
    finally:
        _restore_body(pf, direction, original)
    out = None
    row["wrapper_restored"] = (
        getattr(pf, BODY_LIST_ATTR[direction])[0] is original)
    row["view_batch_unchanged"] = (
        row["wrapped_view_batch"] == row["unwrapped_view_batch"])

    row["enqueue_s"] = enqueue
    row["wall_s"] = wall
    row["body_calls"] = len(durations)
    row["in_body_sum_s"] = float(sum(durations))
    row["first_body_durations_s"] = [float(x) for x in durations[:5]]
    if durations:
        row["in_body_median_s"] = float(statistics.median(durations))
        row["in_body_max_s"] = float(max(durations))
    else:
        row["in_body_median_s"] = None
        row["in_body_max_s"] = None
    row["driver_host_s"] = enqueue - row["in_body_sum_s"]
    if enqueue:
        row["in_body_frac_of_enqueue"] = row["in_body_sum_s"] / enqueue
    if durations:
        row["driver_host_per_body_call_s"] = (row["driver_host_s"]
                                              / len(durations))
    row["expected_body_calls"] = (
        (context.record.get("view_batch_iterations") or {}).get(direction))
    row["torch_body_view_batch_recorded"] = MG53_TORCH_VIEW_BATCH.get(
        context.name)
    if cuda:
        row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        row["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))

    # The wrapper recorded nothing means the driver never called the body this
    # leg replaced -- a different device index, a different route, or a rebuilt
    # projector object.  The split on such a row is not a measurement.
    row["recorded_any"] = bool(durations)
    if not durations:
        row["error"] = ("the wrapped body was never called, so this row "
                        "describes no call; the driver did not route through "
                        "index 0 of " + BODY_LIST_ATTR[direction])
    return row


def single_call_leg(spec, sink, findings):
    """The whole leg: build, one untimed warm-up per direction, one measured
    call per direction.  Every row is written as it is made, so a leg cut short
    still leaves what it finished."""
    import torch

    rows = []
    setup = None
    context = SingleCallContext(spec)
    try:
        setup = context.build()
    except Exception as exc:                                      # noqa: BLE001
        setup = dict(context.record, error=str(exc)[:1200], oom=_is_oom(exc),
                     traceback=traceback.format_exc()[-2000:])
        write_row(sink, setup)
        print("    setup failed: {}".format(str(exc)[:200]), flush=True)
        context.release()
        return setup, rows
    write_row(sink, setup)
    print("    recon {}, {} pixels, view batch {}, {} view batches".format(
        tuple(setup["recon_shape"]), setup["num_pixels"],
        setup["view_batch"], setup["view_batch_iterations"]), flush=True)
    if not setup.get("premise_ok"):
        findings.append("the single-call leg's model did not bind the kernel "
                        "route: {}".format(setup.get("premise_reason")))

    for direction in DIRECTIONS:
        print("    warm-up {}".format(direction), flush=True)
        try:
            warm_up = context.warm_up(direction)
        except Exception as exc:                                  # noqa: BLE001
            row = dict(row="body_split", name=context.name,
                       cell=list(context.cell), direction=direction,
                       route="kernel", error=str(exc)[:1200], oom=_is_oom(exc),
                       stage="warm_up",
                       traceback=traceback.format_exc()[-2000:])
            rows.append(write_row(sink, row))
            print("      warm-up failed: {}".format(str(exc)[:200]), flush=True)
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        print("    measured call {}".format(direction), flush=True)
        try:
            row = body_split_call(context, direction)
            row.update(warm_up)
        except Exception as exc:                                  # noqa: BLE001
            row = dict(row="body_split", name=context.name,
                       cell=list(context.cell), direction=direction,
                       route="kernel", error=str(exc)[:1200], oom=_is_oom(exc),
                       stage="measured",
                       traceback=traceback.format_exc()[-2000:])
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows.append(write_row(sink, row))
    context.release()
    return setup, rows


# ── the plan ──────────────────────────────────────────────────────────────────
def build_plan():
    """Every job, in run order: the identity probe, then every cell's staging,
    then each cell's arms, cheapest cell first and kernel before torch.

    Staging runs BEFORE the header row is written, because the header records
    the staged files' md5 sums.  It is cheap to repeat: a staged file whose md5
    matches is reused, so a re-run stages nothing.
    """
    keep_cells = _strict_subset("MG54_CELLS", [s["name"] for s in cells()])
    keep_routes = _strict_subset("MG54_ROUTES", ROUTES, default=default_routes())
    probe = dict(kind="identity", job_id="identity")
    stages, arms = [], []
    for spec in cells():
        if spec["name"] not in keep_cells:
            continue
        stages.append(dict(kind="stage", spec=spec, name=spec["name"],
                           job_id="stage_" + spec["name"]))
        for route in keep_routes:
            arms.append(dict(kind="arm", spec=spec, name=spec["name"],
                             route=route, arm=arm_id(spec, route),
                             job_id=arm_id(spec, route)))
    if not arms:
        raise ValueError("MG54_CELLS and MG54_ROUTES together select no arm")
    single = None
    if SINGLE_CALL:
        for spec in cells():
            if spec["name"] == SINGLE_CALL_CELL_NAME \
                    and spec["name"] in keep_cells:
                single = spec
    return probe, stages, arms, single


def staged_gib(spec):
    """What one cell's npz costs on disk: the sinogram and the weights, both
    float32, at the sinogram shape."""
    num_views, num_rows, num_channels = spec["cell"]
    return 2.0 * num_views * num_rows * num_channels * 4 / 2 ** 30


def print_plan(probe, stages, arms, single):
    print("mg54 the multiaxis kernel route against the torch-body route: "
          "{} arm(s) over {} cell(s), device {}, {} VCD iteration(s), {} warm "
          "reconstruction(s) after a cold pass"
          .format(len(arms), len(stages), DEVICE, VCD_ITERATIONS,
                  WARM_REPEATS))
    print("  the multiaxis geometry now selects two hand-written kernels "
          "wherever their availability checks pass, and no composed speed "
          "measurement exists for them.  This is the first one.  It decides "
          "nothing.")
    print("  rows -> {}".format(RESULTS_DIR))
    print("  interpreter: {}".format(sys.executable))
    print("  PYTHONPATH:  {}".format(os.environ.get("PYTHONPATH") or "(none)"))
    print("  staged inputs are searched for, in this order, and reused when "
          "their md5 verifies:")
    for directory in stage_search_dirs():
        print("    {}{}".format(directory,
                                "  (fresh staging is written here)"
                                if os.path.abspath(directory)
                                == os.path.abspath(RESULTS_DIR) else ""))
    print("  a fresh stage runs with MBIRTORCH_DISABLE_TRITON=1, so its "
          "sinogram is a torch-body forward projection, which is what the "
          "staged files already on disk hold")
    print("  per-arm hard time cap {:.0f} min; an arm that exceeds it, or runs "
          "out of device memory, is RECORDED and the run continues"
          .format(ARM_TIMEOUT_S / 60.0))
    print("  fingerprints are recorded, not gated; a gap above {:.0e} relative "
          "prints a note".format(FINGERPRINT_NOTE_LEVEL))

    print("\n  {:<{w}}{:>22}{:>22}{:>12}  what it does".format(
        "job", "sinogram", "recon (mirrored)", "staged GiB", w=ARM_COL))
    print("  {:<{w}}{:>22}{:>22}{:>12}  {}".format(
        probe["job_id"], "-", "-", "-",
        "names torch, mbirtorch, the device and the tree witnesses",
        w=ARM_COL))
    for cfg in stages:
        spec = cfg["spec"]
        print("  {:<{w}}{:>22}{:>22}{:>12.1f}  {}".format(
            cfg["job_id"], str(tuple(spec["cell"])),
            str(tuple(mirrored_recon_shape(spec["cell"]))),
            staged_gib(spec),
            "reuses this cell's staged npz, or builds it once", w=ARM_COL))
    for cfg in arms:
        spec = cfg["spec"]
        print("  {:<{w}}{:>22}{:>22}{:>12}  {}".format(
            cfg["job_id"], str(tuple(spec["cell"])),
            str(tuple(mirrored_recon_shape(spec["cell"]))), "-",
            "MBIRTORCH_DISABLE_TRITON={}, premise asserted, cold pass then "
            "{} warm".format(ROUTE_DISABLE_TRITON[cfg["route"]], WARM_REPEATS),
            w=ARM_COL))
    print("  the recon shapes above come from this file's mirror of the "
          "geometry's own rule; every staged file carries the shape the model "
          "really realized and the run records whether the two agreed")

    print("\n  routes:")
    print("    kernel  the default environment.  The arm asserts that "
          "_view_batch_bodies returned two names ending {} and records what "
          "both availability checks answered and why.".format(
              KERNEL_BODY_SUFFIX))
    print("    torch   MBIRTORCH_DISABLE_TRITON=1 in the arm's environment, "
          "which is the library's own kill switch.  The arm asserts both "
          "bodies are the torch ones.")
    print("    an arm whose premise fails times NOTHING: a wall taken on the "
          "other route's bodies would be worse than no wall")
    if DEVICE != "cuda":
        print("    ON {} THE DISTINCTION DEGRADES: there is no triton at all, "
              "so both routes bind the torch bodies and every arm row says "
              "so.".format(DEVICE.upper()))

    print("\n  per arm, in a fresh process: seed {} before every "
          "reconstruction, one {}-iteration reconstruction with the stopping "
          "threshold disabled as the cold pass, then {} more.  Recorded: the "
          "walls, the warm median and spread, the peak allocated and reserved "
          "bytes reset before the cold pass and again before the warm passes, "
          "the bound body names, and a float64 value fingerprint of the final "
          "reconstruction taken on the device."
          .format(VCD_SEED, VCD_ITERATIONS, WARM_REPEATS))
    print("  per cell the parent computes the kernel/torch warm ratio, the "
          "peak-memory ratio and the relative fingerprint differences")

    print("\n  single-call leg: {}".format(
        "at the {} cell, kernel route, in the parent process".format(
            SINGLE_CALL_CELL_NAME) if single else
        "SKIPPED ({})".format("MG54_SINGLE_CALL=0" if not SINGLE_CALL
                              else "the {} cell is not in this plan".format(
                                  SINGLE_CALL_CELL_NAME))))
    if single:
        print("    the model and inputs are built as mg53 built them, one "
              "untimed warm-up call per direction, then one measured call per "
              "direction with a host clock over index 0 of the projector's "
              "bound-body list.  Recorded: body-call count, enqueue seconds, "
              "in-body host seconds, wall, and the driver's chosen view batch.")
        recorded = MG53_TORCH_VIEW_BATCH.get(SINGLE_CALL_CELL_NAME)
        if recorded:
            print("    the torch bodies were measured choosing a view batch of "
                  "{} at this cell, which is {} body calls per projection.  "
                  "This leg reads what the kernels' own cost model chooses "
                  "instead.".format(recorded["view_batch"],
                                    recorded["body_calls"]))
        print("    the wrapper COPIES the wrapped body's _view_batch_cost "
              "attribute when it has one, unlike mg53's, because dropping it "
              "would change the very batch this leg reports; the row carries "
              "the effective batch computed both with and without the wrapper")

    print("\n  exit code = INSTRUMENT HEALTH ONLY: every planned arm produced "
          "a row or a recorded out-of-memory or timeout, every kernel arm "
          "really bound both kernels and every torch arm really bound both "
          "torch bodies, the staged inputs md5-verified, and the tree "
          "witnesses hold.  What any arm MEASURED never changes it.")
    print("  no library file is edited: the one variable is "
          "MBIRTORCH_DISABLE_TRITON, which is the library's own kill switch, "
          "and the body wrapper is removed in a finally")


# ── rows ──────────────────────────────────────────────────────────────────────
def write_row(sink, row):
    """One jsonl row, flushed.

    Flushed per row because a job that is killed mid-run should leave every row
    it had already finished.
    """
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return row


# ── the comparison and the report ─────────────────────────────────────────────
def compare_cell(spec, arm_rows):
    """One cell's two routes, side by side: the warm-median ratio, the peaks and
    the relative fingerprint differences.

    The TORCH route is the reference in every ratio, because it is the route the
    recorded walls were measured on.  A warm ratio below 1 means the kernel
    route was faster.
    """
    out = dict(name=spec["name"], cell=list(spec["cell"]),
               recorded_torch_wall_s=RECORDED_TORCH_WALL_S.get(spec["name"]))
    for route in ROUTES:
        row = arm_rows.get(arm_id(spec, route)) or {}
        if not row:
            status = "not planned"
        elif row.get("error") and row.get("timed_out"):
            status = "timeout"
        elif row.get("error") and is_capacity_reading(row):
            status = "capacity"
        elif row.get("error"):
            status = "error"
        elif row.get("invalid_reasons"):
            status = "premise"
        elif row.get("warm_s") is None:
            status = "no wall"
        else:
            status = "ok"
        out[route + "_status"] = status
        out[route + "_cold_s"] = row.get("cold_s")
        out[route + "_warm_s"] = row.get("warm_s")
        out[route + "_warm_spread"] = row.get("warm_spread")
        out[route + "_cold_peak_bytes"] = row.get("cold_peak_bytes")
        out[route + "_warm_peak_bytes"] = row.get("warm_peak_bytes")
        out[route + "_peak_bytes"] = row.get("peak_bytes")
        out[route + "_reserved_bytes"] = row.get("reserved_bytes")
        out[route + "_forward_body"] = row.get("forward_body")
        out[route + "_back_body"] = row.get("back_body")
        out[route + "_view_batch"] = row.get("view_batch")
        out[route + "_abs_sum"] = row.get("fingerprint_abs_sum")
        out[route + "_sq_sum"] = row.get("fingerprint_sq_sum")
        out[route + "_degraded"] = row.get("route_degraded")
        if row.get("error"):
            out[route + "_error"] = str(
                row["error"]).strip().splitlines()[-1][:300]

    kw, tw = out["kernel_warm_s"], out["torch_warm_s"]
    out["warm_ratio_kernel_over_torch"] = (kw / tw if kw and tw else None)
    out["kernel_speedup_over_torch"] = (tw / kw if kw and tw else None)
    kp, tp = out["kernel_peak_bytes"], out["torch_peak_bytes"]
    out["peak_ratio_kernel_over_torch"] = (kp / tp if kp and tp else None)
    kr, tr = out["kernel_reserved_bytes"], out["torch_reserved_bytes"]
    out["reserved_ratio_kernel_over_torch"] = (kr / tr if kr and tr else None)
    out["abs_sum_rel_gap"] = relative_gap(out["kernel_abs_sum"],
                                          out["torch_abs_sum"])
    out["sq_sum_rel_gap"] = relative_gap(out["kernel_sq_sum"],
                                         out["torch_sq_sum"])
    gaps = [g for g in (out["abs_sum_rel_gap"], out["sq_sum_rel_gap"])
            if g is not None]
    out["fingerprint_note"] = (
        "" if not gaps or max(gaps) <= FINGERPRINT_NOTE_LEVEL else
        "the two routes' volumes differ by {:.2e} relative, above the {:.0e} "
        "this run treats as worth a second look.  The availability checks hold "
        "each kernel to a tolerance against its torch body at a tiny shape; "
        "what accumulates over {} VCD iterations is summation order.  Recorded, "
        "not a failure.".format(max(gaps), FINGERPRINT_NOTE_LEVEL,
                                VCD_ITERATIONS))
    # The recorded wall is context, never a threshold: it says whether the torch
    # route landed where it has always landed on this problem.
    recorded = out["recorded_torch_wall_s"]
    out["torch_wall_vs_recorded"] = (tw / recorded
                                     if tw and recorded else None)
    return out


def _fmt(value, width=10, kind="f", prec=2):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    if isinstance(value, str):
        return "{:>{w}}".format(value, w=width)
    if kind == "d":
        return "{:>{w}d}".format(int(round(float(value))), w=width)
    return "{:>{w}.{p}{k}}".format(value, w=width, p=prec, k=kind)


def _batches(value):
    """One route's per-direction view batch, in a width a table column can
    hold: the two numbers, forward first."""
    if not value:
        return "-"
    return "{} / {}".format(value.get("forward", "-"), value.get("back", "-"))


def print_comparison(item):
    print("    comparison {}: kernel {} / torch {}".format(
        item["name"], item["kernel_status"], item["torch_status"]))
    ratio = item["warm_ratio_kernel_over_torch"]
    print("      warm medians  kernel {}  torch {}   kernel/torch {}".format(
        _fmt(item["kernel_warm_s"], 8), _fmt(item["torch_warm_s"], 8),
        _fmt(ratio, 6) if ratio else "     -"))
    print("      peaks         kernel {}  torch {}   kernel/torch {}".format(
        _fmt(None if item["kernel_peak_bytes"] is None
             else item["kernel_peak_bytes"] / 2 ** 30, 8),
        _fmt(None if item["torch_peak_bytes"] is None
             else item["torch_peak_bytes"] / 2 ** 30, 8),
        _fmt(item["peak_ratio_kernel_over_torch"], 6)))
    print("      fingerprints  abs gap {}  sq gap {}".format(
        _fmt(item["abs_sum_rel_gap"], 10, "e", 2),
        _fmt(item["sq_sum_rel_gap"], 10, "e", 2)))
    if item["fingerprint_note"]:
        print("      NOTE {}".format(item["fingerprint_note"]))


def print_table(comparisons):
    """The table a person reads: one row per cell, both routes side by side.
    Times are warm medians of seeded 3-iteration reconstructions on one device;
    peaks are each process's own peak device allocation."""
    print("\n### the multiaxis kernel route against the torch-body route, one "
          "device, warm median of {} seeded {}-iteration reconstruction(s)"
          .format(WARM_REPEATS, VCD_ITERATIONS))
    print("| cell | kernel | torch | kernel/torch | kernel peak | torch peak "
          "| peak ratio | abs gap | sq gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item in comparisons:
        def seconds(route):
            value = item[route + "_warm_s"]
            return ("{:.2f} s".format(value) if value is not None
                    else item[route + "_status"])

        def peak(route):
            value = item[route + "_peak_bytes"]
            return ("{:.2f} GB".format(value / 2 ** 30)
                    if value is not None else "-")

        ratio = item["warm_ratio_kernel_over_torch"]
        peak_ratio = item["peak_ratio_kernel_over_torch"]
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            tuple(item["cell"]), seconds("kernel"), seconds("torch"),
            "{:.2f}x".format(ratio) if ratio else "-",
            peak("kernel"), peak("torch"),
            "{:.2f}x".format(peak_ratio) if peak_ratio else "-",
            "{:.2e}".format(item["abs_sum_rel_gap"])
            if item["abs_sum_rel_gap"] is not None else "-",
            "{:.2e}".format(item["sq_sum_rel_gap"])
            if item["sq_sum_rel_gap"] is not None else "-"))
    print("A kernel/torch ratio below 1 means the kernel route was faster.  "
          "Peaks are each process's own peak device allocation, the larger of "
          "its cold and warm readings; the reserved-byte peaks are on the rows.")

    print("\n### the torch route against its recorded wall, and what view "
          "batch each route chose (forward / back)")
    print("| cell | torch warm | recorded | ratio | batch, kernel | batch, "
          "torch |")
    print("|---|---|---|---|---|---|")
    for item in comparisons:
        recorded = item["recorded_torch_wall_s"]
        ratio = item["torch_wall_vs_recorded"]
        print("| {} | {} | {} | {} | {} | {} |".format(
            tuple(item["cell"]),
            "{:.2f} s".format(item["torch_warm_s"])
            if item["torch_warm_s"] is not None else item["torch_status"],
            "{:.1f} s".format(recorded) if recorded else "-",
            "{:.3f}x".format(ratio) if ratio else "-",
            _batches(item["kernel_view_batch"]),
            _batches(item["torch_view_batch"])))
    print("The recorded walls are the one-device warm medians this problem has "
          "always produced.  They are context: they say whether the torch route "
          "landed where it has always landed, and nothing is gated on them.  "
          "The view batch is chosen by the cost model of the body actually "
          "bound, so the two batch columns are what the two routes' cost "
          "models decided on the same problem.")


def print_split(setup, rows):
    if setup is None:
        return
    print("\n── ONE CALL ON THE KERNEL ROUTE, INSIDE THE BODY AGAINST OUTSIDE ──")
    if setup.get("error"):
        print("  the leg's setup failed: {}".format(
            str(setup["error"])[:200]))
        return
    print("  cell {} recon {}, {} pixels, bodies {} / {}".format(
        tuple(setup["cell"]), tuple(setup["recon_shape"]),
        setup["num_pixels"], setup.get("forward_body"),
        setup.get("back_body")))
    if setup.get("route_degraded"):
        print("  ROUTE DEGRADED: {}".format(setup.get("route_degraded_reason")))
    print("  {:<10}{:>8}{:>10}{:>12}{:>12}{:>12}{:>14}".format(
        "direction", "batch", "calls", "enqueue s", "in body s", "wall s",
        "was, torch"))
    for row in rows:
        if row.get("error"):
            print("  {:<10}  {}".format(row.get("direction", "?"),
                                        str(row["error"])[:70]))
            continue
        recorded = row.get("torch_body_view_batch_recorded") or {}
        print("  {:<10}{}{}{}{}{}{:>14}".format(
            row["direction"],
            _fmt(row.get("wrapped_view_batch"), 8, "d", 0),
            _fmt(row.get("body_calls"), 10, "d", 0),
            _fmt(row.get("enqueue_s"), 12, prec=4),
            _fmt(row.get("in_body_sum_s"), 12, prec=4),
            _fmt(row.get("wall_s"), 12, prec=3),
            "{} / {}".format(recorded.get("view_batch", "-"),
                             recorded.get("body_calls", "-"))))
        if row.get("view_batch_unchanged") is False:
            print("      the wrapper changed the view batch, {} against {}: "
                  "this row's counts describe a call the driver would not "
                  "otherwise have made".format(row.get("wrapped_view_batch"),
                                               row.get("unwrapped_view_batch")))
        if row.get("expected_body_calls") not in (None, row.get("body_calls")):
            print("      body calls {} against {} view batches from the "
                  "driver's own rule; the public route may take more than one "
                  "pass".format(row.get("body_calls"),
                                row.get("expected_body_calls")))
    print("  batch is the views one body call takes, read off the body the "
          "driver was about to call.  \"was, torch\" is what the torch bodies "
          "were measured choosing at this cell, as batch / body calls.")


def summarize(identity, stage_rows, arm_rows, comparisons, arms, split_setup,
              split_rows, findings, out_path):
    """The table a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A slow
    arm, a wide spread, an arm that ran out of memory and an arm that hit the
    time cap are FINDINGS: they are printed and none of them touches the exit
    code.  A missing row, an md5 that did not verify, an arm that bound the
    other route's bodies, and an error that is not a capacity or timeout reading
    are instrument failures, because they mean the run did not measure what the
    plan said it would.
    """
    print("\n===== mg54 the multiaxis kernel route against the torch-body "
          "route ({}) =====".format(out_path))
    broken = []
    findings = list(findings)

    header = ("{:<{w}}{:>9}{:>10}{:>9}{:>10}{:>11}{:>12}"
              .format("arm", "cold s", "warm s", "spread", "peak GB",
                      "resvd GB", "state", w=ARM_COL))
    print(header)
    print("-" * len(header))
    for cfg in arms:
        name = cfg["arm"]
        row = arm_rows.get(name)
        if row is None:
            print("{:<{w}}  no row".format(name, w=ARM_COL))
            broken.append("{}|no row".format(name))
            continue
        if row.get("error"):
            capacity = is_capacity_reading(row)
            state = "timeout" if row.get("timed_out") else (
                "capacity" if capacity else "ERROR")
            print("{:<{w}}  {}: {}".format(
                name, state,
                str(row["error"]).strip().splitlines()[-1][:90], w=ARM_COL))
            if capacity:
                findings.append("{}: {}".format(name, state))
            else:
                broken.append("{}|{}".format(
                    name, str(row["error"]).strip().splitlines()[-1][:200]))
            continue
        print("{:<{w}}{}{}{:>9}{}{}{:>12}".format(
            name, _fmt(row.get("cold_s"), 9), _fmt(row.get("warm_s"), 10),
            "-" if row.get("warm_spread") is None
            else "{:.1%}".format(row["warm_spread"]),
            _fmt(None if row.get("peak_bytes") is None
                 else row["peak_bytes"] / 2 ** 30, 10),
            _fmt(None if row.get("reserved_bytes") is None
                 else row["reserved_bytes"] / 2 ** 30, 11),
            "ok" if not row.get("invalid_reasons") else "PREMISE",
            w=ARM_COL))
        print("{:<{w}}  bodies {} / {}{}".format(
            "", row.get("forward_body"), row.get("back_body"),
            "  [route degraded]" if row.get("route_degraded") else "",
            w=ARM_COL))
        for reason in row.get("invalid_reasons") or []:
            print("    ARM CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(name, reason))
        if row.get("realized_n_devices") not in (None, 1):
            findings.append("{}: realized {} devices, not one".format(
                name, row.get("realized_n_devices")))
        if row.get("route_degraded"):
            findings.append("{}: {}".format(name,
                                            row.get("route_degraded_reason")))

    print("\n-- the staged inputs --")
    for row in stage_rows:
        print("  {} {}: md5 {}{}{}".format(
            row.get("name"), tuple(row.get("sinogram_shape") or ()),
            row.get("md5", "-"),
            "  (reused from {})".format(row.get("reused_from"))
            if row.get("reused") else "  (built here)",
            "  same bytes mg52 measured" if row.get("same_bytes_as_mg52")
            else ""))
        if row.get("phantom_fallback"):
            print("    phantom: {}".format(row["phantom_fallback"]))
        if row.get("error"):
            broken.append("{}|{}".format(row.get("job_id"),
                                         str(row["error"])[:200]))
        for reason in row.get("invalid_reasons") or []:
            print("    STAGE CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(row.get("job_id"), reason))
        if row.get("recon_shape_mirror_agrees") is False:
            findings.append("{}: the staged recon shape {} is not this file's "
                            "mirror of the geometry's rule, {}".format(
                                row.get("name"), row.get("recon_shape"),
                                row.get("mirrored_recon_shape")))

    print_table(comparisons)
    print_split(split_setup, split_rows)

    print("\n-- what ran --")
    row = identity or {}
    commit = (row.get("git") or {}).get("commit")
    print("  torch {} | triton {} | {} | {} device(s)".format(
        row.get("torch_version", "?"), row.get("triton_version", "?"),
        row.get("device_name", "?"), row.get("device_count", "?")))
    print("  mbirtorch {} | commit {}{}".format(
        row.get("library_file", "?"), commit or "unknown",
        " (dirty)" if (row.get("git") or {}).get("dirty") else ""))
    if row.get("error"):
        print("    PROBE FAILED: {}".format(str(row["error"])[-300:]))
        broken.append("identity|{}".format(
            str(row["error"]).strip().splitlines()[-1][:200]))
    witnesses = (row.get("tree_witnesses") or {})
    if witnesses.get("ok"):
        print("  tree witnesses ok: the padded, recompile-remedied tree, the "
              "kernel module imports, and the geometry's selection consults "
              "both availability checks and reaches both kernels")
    else:
        print("  TREE WITNESSES: {}".format(witnesses))
        broken.append("tree witnesses|{}".format(witnesses))

    for item in comparisons:
        if item["fingerprint_note"]:
            findings.append("{}: {}".format(item["name"],
                                            item["fingerprint_note"]))

    print("\n-- instrument health --")
    print("  the exit code covers four things: every planned arm produced a "
          "row or a recorded out-of-memory or timeout, every kernel arm really "
          "bound both kernels and every torch arm really bound both torch "
          "bodies, the staged inputs md5-verified, and the tree witnesses "
          "hold.  What any arm MEASURED never changes it.")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm produced a result, every arm bound the "
              "bodies its route names, every staged file verified its md5, and "
              "the tree witnesses hold")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings outside the tables")

    return dict(row="summary", healthy=not broken, broken=broken,
                findings=findings, comparisons=comparisons,
                single_call_rows=len(split_rows),
                arms={name: dict(warm_s=row.get("warm_s"),
                                 cold_s=row.get("cold_s"),
                                 peak_bytes=row.get("peak_bytes"),
                                 forward_body=row.get("forward_body"),
                                 back_body=row.get("back_body"),
                                 error=row.get("error"))
                      for name, row in arm_rows.items()},
                out_path=out_path)


# ── the child entry point ─────────────────────────────────────────────────────
def _child_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = run_job(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)
    return 0


def main():
    probe, stages, arms, single = build_plan()
    if DRY:
        print_plan(probe, stages, arms, single)
        return 0
    print_plan(probe, stages, arms, single)
    findings = []

    # ── the identity probe and the staging, both before the header row ───────
    print("\n-- identity probe --", flush=True)
    identity = spawn(probe, PROBE_TIMEOUT_S)
    print("  {}".format(identity.get("error") or
                        "torch {} | {} | {} device(s) | {}".format(
                            identity.get("torch_version"),
                            identity.get("device_name"),
                            identity.get("device_count"),
                            identity.get("library_file"))), flush=True)

    print("\n-- staged inputs (reused when the md5 verifies) --", flush=True)
    stage_rows = []
    for index, cfg in enumerate(stages):
        print("  [{}/{}] {}".format(index + 1, len(stages), cfg["job_id"]),
              flush=True)
        row = spawn(cfg, STAGE_TIMEOUT_S)
        stage_rows.append(row)
        if row.get("error"):
            print("    ERROR: {}".format(str(row["error"])[:400]), flush=True)
        else:
            print("    md5 {} {} recon {}".format(
                row.get("md5"),
                "(reused from {})".format(row.get("reused_from"))
                if row.get("reused") else "(built here)",
                row.get("recon_shape")), flush=True)
    staged_by_name = {row.get("name"): row for row in stage_rows}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        "mg54_multiaxis_kernel_ab_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    arm_rows = {}
    comparisons = []
    split_setup, split_rows = None, []
    with open(out_path, "w") as sink:
        write_row(sink, dict(
            row="run_header", script="mg54_multiaxis_kernel_ab.py",
            node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
            driver_python=sys.executable,
            pythonpath=os.environ.get("PYTHONPATH"),
            results_dir=RESULTS_DIR, stage_search_dirs=stage_search_dirs(),
            identity=identity, tree_witnesses=identity.get("tree_witnesses"),
            staged_md5={row.get("name"): row.get("md5")
                        for row in stage_rows},
            mg52_recorded_stage_md5=MG52_RECORDED_STAGE_MD5,
            recorded_torch_wall_s=RECORDED_TORCH_WALL_S,
            torch_body_view_batch_recorded=MG53_TORCH_VIEW_BATCH,
            cells=[dict(spec) for spec in cells()], routes=list(ROUTES),
            route_disable_triton=ROUTE_DISABLE_TRITON,
            vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
            warm_repeats=WARM_REPEATS, arm_timeout_s=ARM_TIMEOUT_S,
            fingerprint_note_level=FINGERPRINT_NOTE_LEVEL,
            single_call_cell=(single or {}).get("name"),
            plan=[dict(kind=cfg["kind"], job_id=cfg["job_id"],
                       name=cfg.get("name"), route=cfg.get("route"))
                  for cfg in [probe] + list(stages) + list(arms)]))
        for row in stage_rows:
            write_row(sink, dict(row="stage", **row))

        for index, cfg in enumerate(arms):
            print("\n  [{}/{}] {}".format(index + 1, len(arms), cfg["job_id"]),
                  flush=True)
            stage_row = staged_by_name.get(cfg["name"]) or {}
            if not stage_row.get("md5_ok") or not stage_row.get("stage_path"):
                # No verified input, so there is nothing honest to time.  The
                # staging row already carries the reason; this arm records that
                # it never ran rather than running on unverified bytes.
                row = dict(cfg, error="this cell has no md5-verified staged "
                                      "input; see its staging row")
            else:
                row = spawn(dict(cfg, stage_path=stage_row["stage_path"],
                                 stage_md5=stage_row["md5"]), ARM_TIMEOUT_S)
            arm_rows[cfg["arm"]] = row
            write_row(sink, dict(row="arm", **row))
            if row.get("error"):
                print("    {}: {}".format(
                    "READING" if is_capacity_reading(row) else "ERROR",
                    str(row["error"]).strip().splitlines()[-1][:200]),
                    flush=True)
            elif row.get("invalid_reasons"):
                print("    PREMISE: {}".format(
                    str(row["invalid_reasons"][0])[:250]), flush=True)
            else:
                print("    bodies {} / {}".format(row.get("forward_body"),
                                                  row.get("back_body")),
                      flush=True)
                print("    cold {:.2f}s  warm {:.2f}s  spread {:.1%}  "
                      "peak {}  view batch {}".format(
                          row.get("cold_s", 0.0), row.get("warm_s", 0.0),
                          row.get("warm_spread", 0.0),
                          "-" if row.get("peak_bytes") is None
                          else "{:.2f} GB".format(row["peak_bytes"] / 2 ** 30),
                          row.get("view_batch")), flush=True)
            # The comparison row goes out as soon as every PLANNED arm of a
            # cell is in, so a job cut short still carries every cell it
            # finished, and a run restricted to one route still gets its row --
            # with the other route recorded as not planned rather than missing.
            spec = cfg["spec"]
            planned = [other["arm"] for other in arms
                       if other["name"] == spec["name"]]
            if all(name in arm_rows for name in planned):
                comparison = compare_cell(spec, arm_rows)
                comparisons.append(comparison)
                write_row(sink, dict(row="comparison", **comparison))
                print_comparison(comparison)

        # ── the single-call leg, last and in this process ────────────────────
        if single is None:
            reason = ("MG54_SINGLE_CALL=0" if not SINGLE_CALL else
                      "the {} cell is not in this plan".format(
                          SINGLE_CALL_CELL_NAME))
            print("\n  single-call leg SKIPPED: {}".format(reason), flush=True)
            write_row(sink, dict(row="single_call_setup", skipped=True,
                                 reason=reason))
        elif os.environ.get("MBIRTORCH_DISABLE_TRITON") == "1":
            # The parent runs the kernel route, so the kill switch must not be
            # set in this process.  The sbatch deliberately does not export it.
            reason = ("MBIRTORCH_DISABLE_TRITON=1 in this process, so the "
                      "parent cannot run the kernel route this leg is about")
            print("\n  single-call leg SKIPPED: {}".format(reason), flush=True)
            write_row(sink, dict(row="single_call_setup", skipped=True,
                                 reason=reason))
            findings.append("single-call leg skipped: " + reason)
        else:
            print("\n  single-call leg, kernel route, cell {}".format(
                single["name"]), flush=True)
            split_setup, split_rows = single_call_leg(single, sink, findings)

        summary = summarize(identity, stage_rows, arm_rows, comparisons, arms,
                            split_setup, split_rows, findings, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        write_row(sink, summary)
    print("\nwrote {}".format(out_path))
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    if CHILD:
        sys.exit(_child_main(CHILD, CHILD_OUT))
    sys.exit(main())
