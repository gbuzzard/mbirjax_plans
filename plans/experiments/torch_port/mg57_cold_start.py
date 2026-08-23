"""mg57 -- WHAT DOES A USER WAIT THROUGH ON A FIRST RECONSTRUCTION, AND WHICH
PARTS OF THAT WAIT ARE PAID AGAIN?

WHY THIS RUN EXISTS.  Every recorded measurement in this series reports a WARM
median and throws the cold pass away.  That protocol is right for comparing two
routes -- a first pass carries compilation and one-time setup that say nothing
about which route is faster -- but it means the cold pass has never been
attributed.  A user does not get to discard it.  The first reconstruction of a
session is the one they sit and watch, and nobody here can currently say what it
is made of.

This run takes that wait apart, once, at the cell the recorded production class
uses: a PARALLEL BEAM reconstruction of a (1024, 1008, 992) sinogram.  It splits
one cold reconstruction into its phases and it separates three costs that behave
differently from each other:

    WORK A COMPILE CACHE REMOVES.  torch's inductor cache and Triton's JIT cache
    both survive a process.  Work that lands in them is paid once per machine
    (per cache directory, really) and never again, so a user's second session
    does not repeat it.

    WORK EVERY PROCESS PAYS ONCE.  Importing the library, creating a CUDA
    context on each device, building the model, settling the device layout,
    loading a compiled artifact back out of the cache.  A cache does not remove
    any of this; a new process pays it all again.

    THE RECONSTRUCTION ARITHMETIC.  What is left when both of the above are
    gone: the warm, in-process wall that the warm medians in this series report.

THIS RUN DECIDES NOTHING.  It edits no library file, changes no default, and
proposes no remedy.  It prints numbers.  Whether any of the cost it attributes
is worth reducing, and how, is a person's judgement made after reading them --
which is why nothing below draws a conclusion, and why the exit code reports
only whether the instrument worked.

THE INPUT IS STAGED BY A SEPARATE PROCESS, AND THAT IS THE WHOLE POINT.
A measured arm LOADS its sinogram from an npz and verifies the md5 beside it.
It never forward projects one, because a forward projection compiles the
projection bodies and fills the very caches the arm exists to find empty -- it
would destroy the cold measurement in the act of preparing for it.  So the
staging runs first, in its own process, with its own cache directories that no
arm ever reads, and every arm afterwards reads bytes from disk.  A staged file
that verifies is reused; only when none exists is one built.

THE CELL.  Parallel beam, sinogram (1024, 1008, 992), built the way the sibling
harnesses in this directory build a parallel cell: angles evenly spaced over
half a turn, a seeded shepp-logan low dynamic range phantom (seed 13) forward
projected through the model.  The weights are NOT staged.  They are built on the
host inside each arm, from the loaded sinogram, before the timed reconstruction
and timed separately -- a user pays that host arithmetic on every run, but it is
not part of the reconstruction and it should not be inside its wall.

THE TWO CACHE DIRECTORIES ARE BOTH SET, PER ARM, AND THAT IS LOAD-BEARING.
torch's inductor cache is TORCHINDUCTOR_CACHE_DIR and Triton's JIT cache is
TRITON_CACHE_DIR.  The sbatch files in this series export the first and leave
the second alone, which sends Triton's cache to a directory under the user's
home; a "cold" arm run that way would quietly reuse compiled Triton artifacts
from every earlier job on this cluster and report a cold start that was nothing
of the kind.  So this harness owns both directories, sets both explicitly in
each arm's environment, and records for every arm both paths and the file count
and total bytes in each, sampled before and after the arm's work.

THE ARMS.  Four, each a fresh subprocess under a 30-minute cap.

    first_run_n1   Both cache directories fresh and empty.  One device, named
                   explicitly.  ONE reconstruction, fully instrumented.  This is
                   a user's first-ever run on a machine.
    cached_n1      Both cache directories pointed at the ones first_run_n1
                   filled.  One device.  TWO reconstructions in the one process,
                   instrumented and recorded separately: the first is a cold
                   PROCESS with warm caches, the second is the in-process warm
                   baseline that the recorded 21.301 s wall should reproduce.
    first_run_n4   As first_run_n1 but four devices, with fresh cache
                   directories of its own.  Four devices is what the automatic
                   policy chooses at this cell, so this is the other cold start
                   a user actually meets.
    cached_n4      As cached_n1 but four devices, pointed at first_run_n4's
                   directories.

Every arm: seed 13 reset immediately before each reconstruction, three VCD
iterations, the stopping threshold disabled, and the sinogram loaded from the
staged npz with its md5 verified before it is used.  The device list is named
one by one on every arm, so a two-way comparison is between the layouts this
plan asked for rather than between whatever a policy preferred.

THE PHASE SPLIT, WHICH IS THE CENTRAL INSTRUMENT.  A reconstruction is timed
from OUTSIDE the package: a host clock is wrapped around each of the seams
below, the wrappers are installed immediately before the measured reconstruction
and removed in a finally, and no library file is touched.

    initialize_recon            the whole call
    gen_set_of_pixel_partitions inside it (a module-level function, wrapped on
                                mbirtorch.vcd_utils)
    _apply_device_policy        where the layout is settled
    recon_direct                the initial reconstruction, run because
                                init_recon is None
    _initial_error_state
    compute_hessian_diagonal
    create_vcd_subset_updater
    _vcd_recon                  the whole call, so the iteration total is what
                                is left after its own setup phases

Every timer synchronizes the model's devices before it starts and again before
it stops, so device work is charged to the phase that issued it rather than to
whichever later phase happened to wait for it.  That synchronization removes
some overlap between phases and so lengthens the reconstruction a little: it is
the price of attribution, it is paid deliberately, and the row carries the plain
wall of an UN-INSTRUMENTED reconstruction of the same input beside it so a
reader can see exactly what the instrument cost.  A seam that does not exist in
the tree under test is recorded as absent on the row; nothing raises.

COMPILE ATTRIBUTION COMES FROM THE TOOLS' OWN ACCOUNTING, not from inference.

    THE TORCH SIDE.  After each reconstruction:
    torch._dynamo.utils.compile_times(repr='csv', aggregate=True), recorded
    exactly as it comes back, and torch._dynamo.utils.counters reduced to a
    plain dict.  The counters are cleared before each reconstruction, so each
    row's counters describe that reconstruction; the cumulative per-phase
    nanosecond totals dynamo also keeps are recorded as a before/after delta,
    which needs no reset at all.

    THE TRITON SIDE.  The four hand-written kernel wrappers live in
    mbirtorch.triton_parallel and mbirtorch.triton_cone and share one launch-key
    set, mbirtorch.triton_cone._COMPILED_LAUNCH_KEYS.  Its size is recorded
    before and after each reconstruction.  The two parallel wrappers are also
    wrapped with a timer, on the bodies the driver really calls, which records
    per call the duration and whether that call GREW the launch-key set -- a
    call that grew it is a call that compiled.  The row carries the number of
    compiling calls, their summed duration, and the summed duration of all
    calls.

    THE WRAPPER'S FORM IS NOT FREE CHOICE.  It copies the wrapped body's
    _view_batch_cost attribute when there is one, because the driver reads that
    attribute off the body it is about to call to choose a view batch, and a
    bare closure would drop it and change the very batching the arm reports.
    It copies nothing else -- in particular not _mbirtorch_no_compile, which is
    read in a different place at a different time.  This is a recorded trap from
    an earlier harness in this directory.

    AND THE PROJECTOR LAYER'S OWN COUNT: len(mbirtorch.projectors._COMPILE_CACHE)
    before and after, which is how many compiled callables that layer holds.

ALSO RECORDED, PER ARM: the wall from process start to the end of
``import mbirtorch`` (measured around the import, with the torch import timed
separately inside it), the wall to build the model, the wall to configure the
devices, the cost of creating a CUDA context on each device (timed as a first
trivial allocation, taken before anything else touches a device so that the
model build and the device configuration do not absorb it), the staged load and
md5 verification, the host weights build, and the peak allocated bytes per
device for each reconstruction.

OUTPUT.  One jsonl under MG57_RESULTS, named
mg57_cold_start_<node>_<stamp>.jsonl: a header row carrying the torch and
mbirtorch identity, the GPUs, the tree witnesses and the staged file's md5; the
staging row; one row per arm; a comparison row per device count; and a summary
row.  Rows are flushed as they are written, so a job that runs out of wall time
yields everything it finished and MG57_ARMS re-runs the rest.  The parent
process never imports torch: every device touch happens in an arm.

Run:
    <torch python> mg57_cold_start.py            on a four-GPU node
    MG57_DRY=1 <any python> mg57_cold_start.py        the plan, then stop
    MG57_SMOKE=1 <python> mg57_cold_start.py          the local CPU smoke

Configuration is by environment variable only; there is no command line.  Export
from the SUBMITTING SHELL, never through an sbatch --export list, which slurm
splits on commas.  An unrecognized arm name is an error, not a silent skip.
    MG57_RESULTS=<dir>         where the jsonl, the staged npz and the cache
                               directories this run owns all go
    MG57_ARMS=a,b              subset of the arms, by arm name
    MG57_ITERATIONS=3          VCD iterations per reconstruction
    MG57_ARM_TIMEOUT_MIN=30    the per-arm hard cap
    MG57_PLAIN_PASS=1          set 0 to skip the un-instrumented pass that
                               prices the instrument itself
    MG57_DRY=1                 print the plan and exit; imports no torch
    MG57_SMOKE=1               the local CPU smoke
    MG57_CHILD=<path>          internal: a job description.  Its presence puts
                               this process in child mode.
    MG57_CHILD_OUT=<path>      internal: where that child writes its row

THE LOCAL SMOKE runs the whole flow at a tiny CPU cell, and it degrades in
places rather than pretending otherwise.  There is ONE subprocess arm, not four:
a laptop has one device, so neither four-device arm can run, and a second
process pointed at a cache the first filled cannot be exercised without a second
arm.  That single arm does two reconstructions, so the cold-then-warm
bookkeeping, the phase split, the un-instrumented pass and the child protocol
all run.  There is no triton on a CPU install, so TRITON_CACHE_DIR governs
nothing, the launch-key set never grows and the wrapped bodies are the torch
ones rather than the Triton wrappers.  The row and the plan say all of this.
The smoke is plumbing, not a measurement.
"""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback

#: When this interpreter started, as close to the top of the file as a python
#: statement can be.  An arm's "what did the user wait through" figures are
#: measured against this, so it has to be taken before anything heavy runs.
PROCESS_START = time.time()

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


SMOKE = _flag("MG57_SMOKE")
DRY = _flag("MG57_DRY")
#: The subprocess mode: the path to the job description this process is to run.
#: Non-empty means child mode.  The sbatch unsets it, so a stray value in the
#: submitting shell cannot turn the real run into a single job.
CHILD = os.environ.get("MG57_CHILD", "").strip()
CHILD_OUT = os.environ.get("MG57_CHILD_OUT", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: The cell.  Parallel beam at the recorded production class: this is the shape
#: the one-device warm wall quoted below was measured at, so the second
#: reconstruction of a cached arm has a number to land beside.
CELL = (1024, 1008, 992)
#: The smoke's stand-in.  Small enough that the whole flow -- staging, one arm,
#: two reconstructions, an un-instrumented pass and every table -- runs in about
#: a minute on a laptop CPU.
SMOKE_CELL = (24, 20, 16)

# ── the reconstruction protocol ───────────────────────────────────────────────
#: Three iterations with the stopping threshold disabled, so every
#: reconstruction in this run does exactly the same amount of work and the
#: recorded warm wall below is a wall for the same protocol.
VCD_ITERATIONS = _positive_int("MG57_ITERATIONS", 3)
#: Reset immediately before every reconstruction.  The library draws its pixel
#: partitions from numpy's global generator, so this is what makes two
#: reconstructions in one process the same reconstruction twice.
VCD_SEED = 13
#: Whether each arm ends with one UN-INSTRUMENTED reconstruction.  On by
#: default: without it nothing in this run prices the instrument's own cost, and
#: the phase totals below would have no plain wall to be read against.
PLAIN_PASS = _flag("MG57_PLAIN_PASS", "1")

#: The per-arm hard cap.  An arm that exceeds it is killed and the timeout is
#: recorded as that arm's result -- a cold start that does not finish inside
#: half an hour is a reading, not a broken instrument.
ARM_TIMEOUT_S = 60.0 * float(os.environ.get("MG57_ARM_TIMEOUT_MIN", "30"))
#: The identity probe imports torch and mbirtorch and reads the tree witnesses
#: and nothing else, so ten minutes is generous even with a cold module cache on
#: a shared filesystem.
PROBE_TIMEOUT_S = 600.0
#: The staging job verifies an existing file's md5 or builds and writes one.  It
#: gets its own cap so a slow filesystem cannot eat an arm's budget.
STAGE_TIMEOUT_S = 3600.0

# ── recorded context, not thresholds ──────────────────────────────────────────
#: What this exact cell measured on one device, read in session from the job log
#: of the widening-floors refresh (job 15435735, its parallel (1024, 1008, 992)
#: rows): 21.301 s warm median at one device with a 0.1 percent spread, and
#: 9.527 s at four devices with a 5.2 percent spread, both on the same seeded
#: three-iteration protocol this run uses.  Printed beside the in-process warm
#: reconstruction of the cached arms so a reader can see at a glance whether it
#: landed where it landed before.  NOTHING is gated on it.
RECORDED_WARM_S = {1: 21.301, 4: 9.527}
RECORDED_WARM_SPREAD = {1: 0.001, 4: 0.052}
RECORDED_WARM_SOURCE = ("the widening-floors refresh job log 15435735, its "
                        "parallel (1024, 1008, 992) rows")

#: Substrings that mark an arm's failure as a capacity READING rather than a
#: harness fault.  A run that does not fit the devices it was given is an
#: outcome.  Matched case-insensitively.
CAPACITY_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory",
                    "failed to allocate", "memoryerror", "cannot allocate",
                    "memorypreflighterror")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("MG57_RESULTS",
                             os.path.join(SCRIPT_DIR, "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 16                  # wide enough for the longest arm id printed
PHASE_COL = 32                # wide enough for the longest phase name printed
DEV_COL = 13                  # wide enough for "cuda:0,1,2,3"
# ──────────────────────────────────────────────────────────────────────────────


# ── the arms ──────────────────────────────────────────────────────────────────
#: The four arms, in run order.  A "fresh" arm's cache directories are emptied
#: by the parent before it starts; a "reuse" arm is pointed at the pair the
#: named arm filled.  ``recons`` is how many instrumented reconstructions the
#: arm runs in its one process.
REAL_ARMS = (
    dict(name="first_run_n1", count=1, cache_owner="n1", caches="fresh",
         recons=1, filled_by=None,
         what="a first-ever run: both caches empty, one device"),
    dict(name="cached_n1", count=1, cache_owner="n1", caches="reuse",
         recons=2, filled_by="first_run_n1",
         what="a new process on filled caches, one device: the first "
              "reconstruction is a cold process, the second the in-process "
              "warm baseline"),
    dict(name="first_run_n4", count=4, cache_owner="n4", caches="fresh",
         recons=1, filled_by=None,
         what="a first-ever run at the device count the automatic policy "
              "chooses here"),
    dict(name="cached_n4", count=4, cache_owner="n4", caches="reuse",
         recons=2, filled_by="first_run_n4",
         what="a new process on filled caches, four devices"),
)
#: The smoke's single arm.  One subprocess, one device, two reconstructions.
SMOKE_ARMS = (
    dict(name="smoke_n1", count=1, cache_owner="smoke", caches="fresh",
         recons=2, filled_by=None,
         what="the whole flow at a tiny cell on one CPU device"),
)


def arms_declared():
    return SMOKE_ARMS if SMOKE else REAL_ARMS


def cell():
    return SMOKE_CELL if SMOKE else CELL


def device_list(count):
    """The devices an arm is given, named one by one.

    On CUDA that is cuda:0 .. cuda:<count-1>.  The smoke has one CPU device and
    one arm, so the list is that device.
    """
    if DEVICE != "cuda":
        return [DEVICE]
    return ["cuda:{}".format(i) for i in range(count)]


def device_label(devices):
    """The device list in a width a table column can hold: cuda:0,1,2,3."""
    if not devices:
        return "-"
    if all(str(d).startswith("cuda:") for d in devices):
        return "cuda:" + ",".join(str(d).split(":", 1)[1] for d in devices)
    return ",".join(str(d) for d in devices)


def counts_measured():
    """The device counts that have arms in this run, ascending."""
    return sorted({int(spec["count"]) for spec in arms_declared()})


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a repeat
    before.  The error names the full valid list, because the caller who
    mistyped one id needs to see the others.
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
            raise ValueError("{}: {!r} is not one of this run's: {}".format(
                env_name, token, ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError("{}: no valid tokens in {!r}.  The valid ones are: {}"
                         .format(env_name, raw, ", ".join(allowed)))
    # Normalized to the DECLARED order: the run order is load-bearing, because a
    # reuse arm reads the directories a fresh arm filled.
    return [name for name in allowed if name in chosen]


# ── the two compile caches this harness owns ──────────────────────────────────
def cache_root():
    return os.path.join(RESULTS_DIR, "caches")


def cache_dirs(owner):
    """The inductor and Triton cache directories one owner uses.

    BOTH are named here and both are set in the child's environment.  Setting
    only the first -- which is what the sbatch files in this series do -- leaves
    Triton's JIT cache in the user's home directory, where a "cold" arm would
    silently reuse compiled Triton artifacts from every earlier job on the
    machine.
    """
    base = os.path.join(cache_root(), owner)
    return dict(inductor=os.path.join(base, "inductor"),
                triton=os.path.join(base, "triton"))


def cache_state(path):
    """One cache directory's file count and total bytes.

    Walked rather than listed: both caches nest their artifacts several levels
    deep, so a top-level listing would report a handful of directories and no
    files at all.  A file that vanishes mid-walk (another process evicting an
    entry) is skipped rather than raising.
    """
    if not os.path.isdir(path):
        return dict(path=path, exists=False, files=0, bytes=0)
    files, total = 0, 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            files += 1
    return dict(path=path, exists=True, files=files, bytes=total)


def cache_states(dirs):
    return {kind: cache_state(path) for kind, path in dirs.items()}


def empty_caches(dirs):
    """Remove and recreate both directories, so a fresh arm is really fresh.

    Done in the PARENT, before the arm is spawned, because the child cannot
    empty a directory torch has already read.
    """
    for path in dirs.values():
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


# ── the staged input ──────────────────────────────────────────────────────────
def stage_name():
    """The staged filename.  Reproduced from the cell so a file this run writes
    is found by the next run of the same cell."""
    return "mg57_stage_parallel_{}x{}x{}.npz".format(*tuple(cell()))


def md5_path(path):
    return path + ".md5"


def file_md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: the real cell's npz is about
    3.8 GiB and reading it whole to hash it would be wasteful."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stage_search_dirs():
    """Where a staged copy may already be, in search order: this run's own
    results directory first, then the script's local default.  Duplicates are
    dropped and order is kept."""
    candidates = [RESULTS_DIR, os.path.join(SCRIPT_DIR, "results")]
    seen, out = set(), []
    for path in candidates:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def staged_present(path):
    return os.path.exists(path) and os.path.exists(md5_path(path))


def recorded_md5(path):
    with open(md5_path(path)) as handle:
        return handle.read().strip()


def find_staged():
    """The first staged copy that exists with its md5 sidecar, or None.
    Existence only -- the md5 is VERIFIED by whoever reads it."""
    name = stage_name()
    for directory in stage_search_dirs():
        path = os.path.join(directory, name)
        if staged_present(path):
            return path
    return None


def stage_write_path():
    return os.path.join(RESULTS_DIR, stage_name())


def read_staged(path, with_sinogram=True):
    """The npz read itself, WITHOUT the md5 check.  Callers that have just
    hashed the file use this; everything else goes through ``load_staged``, so a
    file is never read twice to verify it once.

    ``with_sinogram=False`` reads everything except the sinogram.  An npz member
    is only read when it is asked for, and that one is the whole file.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as handle:
        meta = dict(
            angles=handle["angles"],
            sinogram_shape=[int(v) for v in handle["sinogram_shape"]],
            recon_shape=[int(v) for v in handle["recon_shape"]],
            geometry=str(handle["geometry"].item()),
            delta_voxel=float(handle["delta_voxel"]),
            phantom_fallback=str(handle["phantom_fallback"].item()))
        if with_sinogram:
            meta["sinogram"] = handle["sinogram"]
    return meta


def load_staged(path, with_sinogram=True):
    """Read the staged npz into a plain dict, after verifying its md5.

    Raises on a mismatch.  An arm that reconstructed different bytes than its
    siblings did not measure what the plan said, and a truncated read on a
    shared parallel filesystem is a recorded failure mode of this work.
    """
    expected = recorded_md5(path)
    actual = file_md5(path)
    if actual != expected:
        raise ValueError("the staged file at {} hashes to {}, not the recorded "
                         "{}".format(path, actual, expected))
    meta = read_staged(path, with_sinogram=with_sinogram)
    meta["md5"] = actual
    return meta


# ── the model and its inputs ──────────────────────────────────────────────────
def angles_for(shape):
    """The per-view angles: evenly spaced over half a turn, which is how every
    parallel cell in this series is built."""
    import numpy as np

    return np.linspace(0, np.pi, int(shape[0]), endpoint=False)


def build_model(sinogram_shape, angles, devices):
    """The parallel model an arm reconstructs with, on the devices it names.

    The device list is EXPLICIT, which turns the automatic choice off.  That is
    deliberate: the two device counts here are the two a user meets, and naming
    them makes each arm's layout a fact on its row rather than a policy outcome
    that could differ between the cold arm and the cached one.
    """
    import mbirtorch

    model = mbirtorch.ParallelBeamModel(
        tuple(int(v) for v in sinogram_shape), angles)
    model.configure_devices(devices=list(devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def build_phantom(recon_shape):
    """The phantom, as a host float32 array, with the staging's own fallback.

    The shepp-logan builder places its ellipsoids as fractions of the volume,
    and on a volume only a few voxels deep every one of them can miss, leaving
    the phantom all zeros -- which the smoke's tiny cell can reach.  An all-zero
    phantom forward projects to an all-zero sinogram, so an arm would time a
    reconstruction of nothing.  The fallback is a seeded uniform volume, and the
    row records that it was used.
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
    """The one host exit.  A gathered container ALREADY returns numpy, so a
    gather is never followed by ``.detach()`` -- re-detaching one is a recorded
    way to lose rows."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, "gather", None)) and hasattr(x, "placement"):
        return x.gather()
    return (x.detach().cpu().numpy()
            if callable(getattr(x, "detach", None)) else np.asarray(x))


def build_weights(sinogram):
    """The weights formula every harness in this series uses, evaluated on the
    host.

    Timed and reported separately from the reconstruction.  It is real cost a
    user pays on every run -- it is not free and it is not cached -- but it is
    not reconstruction work, and folding it into the wall would misattribute a
    host array pass as device time.
    """
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32,
                                                             copy=False)


# ── the tree under test ───────────────────────────────────────────────────────
def tree_witnesses():
    """What tree produced these numbers, measured rather than asserted.

    The first three are the block every job in this series carries: they say
    this is the padded, recompile-remedied tree the recorded walls were measured
    on, and the third matters directly here because a tree without it would hand
    a torch body eager python rather than a compiled one -- and compilation is
    the subject.

    The multiaxis pair is carried verbatim from the sibling that established
    this block, because it identifies the committed tip this run is meant to be
    testing.  The parallel pair is added, because those are the wrappers this
    run times: both must exist, both must carry the per-view cost attribute the
    driver reads to choose a view batch, and the parallel geometry's selection
    hook must consult both availability functions and reach both wrappers.

    Everything here is read by SOURCE INSPECTION and attribute lookup: no model
    is built, no device is touched and no CUDA is initialized, so the witness
    block costs nothing and can run anywhere.  A lookup that fails is recorded
    as failed; nothing here raises.
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

        from mbirtorch import triton_multiaxis, triton_parallel
        from mbirtorch.multiaxis_parallel import MultiAxisParallelModel
        from mbirtorch.parallel_beam import ParallelBeamModel

        record["triton_multiaxis_file"] = triton_multiaxis.__file__
        record["triton_parallel_file"] = triton_parallel.__file__

        def kernel_pair(module, names):
            out = {}
            for direction, attr in names:
                body = getattr(module, attr, None)
                out[direction] = dict(
                    present=body is not None,
                    name=getattr(body, "__name__", None),
                    named_like_a_kernel=bool(
                        getattr(body, "__name__", "").endswith(
                            "_view_batch_triton")),
                    has_view_batch_cost=getattr(body, "_view_batch_cost",
                                                None) is not None)
            return out

        multiaxis = kernel_pair(
            triton_multiaxis,
            (("forward", "_multiaxis_forward_view_batch_triton"),
             ("back", "_multiaxis_back_view_batch_triton")))
        parallel = kernel_pair(
            triton_parallel,
            (("forward", "_parallel_forward_view_batch_triton"),
             ("back", "_parallel_back_view_batch_triton")))
        record["multiaxis_kernels"] = multiaxis
        record["parallel_kernels"] = parallel

        multiaxis_selection = inspect.getsource(
            MultiAxisParallelModel._view_batch_bodies)
        record["multiaxis_selection_ok"] = bool(
            "multiaxis_forward_kernel_usable" in multiaxis_selection
            and "multiaxis_back_kernel_usable" in multiaxis_selection
            and "_multiaxis_forward_view_batch_triton" in multiaxis_selection
            and "_multiaxis_back_view_batch_triton" in multiaxis_selection)
        parallel_selection = inspect.getsource(
            ParallelBeamModel._view_batch_bodies)
        record["parallel_selection_consults_availability"] = bool(
            "parallel_forward_kernel_usable" in parallel_selection
            and "parallel_back_kernel_usable" in parallel_selection)
        record["parallel_selection_reaches_kernels"] = bool(
            "_parallel_forward_view_batch_triton" in parallel_selection
            and "_parallel_back_view_batch_triton" in parallel_selection)

        record["ok"] = bool(
            record["padded_kernel_width_504"] == 512
            and record["recompile_limit_floor"] >= 64
            and record["raise_on_compiling_thread"]
            and all(entry["present"] and entry["named_like_a_kernel"]
                    and entry["has_view_batch_cost"]
                    for entry in list(multiaxis.values())
                    + list(parallel.values()))
            and record["multiaxis_selection_ok"]
            and record["parallel_selection_consults_availability"]
            and record["parallel_selection_reaches_kernels"])
    except Exception as exc:                                      # noqa: BLE001
        record.update(available=False, ok=False,
                      reason="{}: {}".format(type(exc).__name__, exc))
    return record


# ── the phase split ───────────────────────────────────────────────────────────
#: The seams a reconstruction is split at.  ``owner`` says where the wrapper is
#: installed: "model" is an instance attribute, which shadows the class method
#: for this model alone and is removed again by name; "vcd_utils" is a
#: module-level function.  ``within`` names the phase a seam runs inside, so the
#: table can show which rows are already counted in another row.
PHASE_SEAMS = (
    dict(name="initialize_recon", owner="model", within=None),
    dict(name="gen_set_of_pixel_partitions", owner="vcd_utils",
         within="initialize_recon"),
    dict(name="_apply_device_policy", owner="model", within="several"),
    dict(name="recon_direct", owner="model", within="_vcd_recon"),
    dict(name="_initial_error_state", owner="model", within="_vcd_recon"),
    dict(name="compute_hessian_diagonal", owner="model", within="_vcd_recon"),
    dict(name="create_vcd_subset_updater", owner="model", within="_vcd_recon"),
    dict(name="_vcd_recon", owner="model", within=None),
)
#: The two seams that between them cover the whole public call.  Their sum plus
#: the derived remainder is the reconstruction's wall.
TOP_LEVEL_PHASES = ("initialize_recon", "_vcd_recon")
#: The setup phases inside _vcd_recon.  The iteration loop is what is left of
#: _vcd_recon after these, which is why the iteration total is derived rather
#: than wrapped: there is no single call to put a clock around.
VCD_SETUP_PHASES = ("recon_direct", "_initial_error_state",
                    "compute_hessian_diagonal", "create_vcd_subset_updater")
#: The derived rows, computed from the wrapped ones.
DERIVED_ITERATIONS = "vcd iterations (remainder)"
DERIVED_OTHER = "recon, outside both (remainder)"


class PhaseTimers:
    """Host clocks around the seams of one reconstruction, installed from
    outside the package and removed in a finally.

    THE SYNCHRONIZATION IS DELIBERATE AND IT IS NOT FREE.  Each timer
    synchronizes every device the model is using before it starts and again
    before it stops, so that work a phase issued is charged to that phase rather
    than to whichever later phase first waited on it.  That removes some of the
    overlap between phases that a normal run enjoys, so an instrumented
    reconstruction is a little longer than a plain one.  That is the price of
    attribution and it is paid on purpose; the arm also runs one plain,
    un-instrumented reconstruction so a reader can see exactly what the price
    was.

    A seam that does not exist in the tree under test is recorded in ``missing``
    and the rest continue.  Nothing here raises into the reconstruction.
    """

    def __init__(self, model, devices, cuda, torch_module, vcd_utils_module):
        self.model = model
        self.torch = torch_module
        self.vcd_utils = vcd_utils_module
        self.cuda = cuda
        # Fixed at install time from the arm's own device list.  The layout is
        # explicit on every arm, so this is the set the model uses throughout,
        # and reading it once keeps the sync itself off the critical path.
        self.devices = list(devices) if cuda else []
        self.totals = {}
        self.calls = {}
        self.missing = []
        self.sync_errors = []
        self._installed = []          # (owner, name, had_own, previous, timed)

    # -- the sync ------------------------------------------------------------
    def _sync(self):
        if not self.cuda:
            return
        for device in self.devices:
            try:
                self.torch.cuda.synchronize(device)
            except Exception as exc:                              # noqa: BLE001
                if len(self.sync_errors) < 4:
                    self.sync_errors.append("{}: {}".format(
                        type(exc).__name__, exc))

    def _make_timer(self, name, original):
        def timed(*args, **kwargs):
            self._sync()
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self._sync()
                self.totals[name] = self.totals.get(name, 0.0) + (
                    time.perf_counter() - start)
                self.calls[name] = self.calls.get(name, 0) + 1
        return timed

    # -- install and restore -------------------------------------------------
    def install(self):
        for seam in PHASE_SEAMS:
            name = seam["name"]
            self.totals.setdefault(name, 0.0)
            self.calls.setdefault(name, 0)
            if seam["owner"] == "model":
                # getattr on the instance returns a BOUND method, so the
                # wrapper calls it without passing self.  The instance
                # attribute shadows the class method for this model only, which
                # is what keeps the patch off every other model in the process.
                original = getattr(self.model, name, None)
                if original is None:
                    self.missing.append(name)
                    continue
                had_own = name in vars(self.model)
                previous = vars(self.model).get(name)
                setattr(self.model, name, self._make_timer(name, original))
                self._installed.append(("model", name, had_own, previous,
                                        getattr(self.model, name)))
            else:
                original = getattr(self.vcd_utils, name, None)
                if original is None:
                    self.missing.append(name)
                    continue
                timed = self._make_timer(name, original)
                setattr(self.vcd_utils, name, timed)
                self._installed.append(("vcd_utils", name, True, original,
                                        timed))
        return self

    def restore(self):
        """Put every seam back and report whether it took.

        Reported rather than assumed: a wrapper left behind would time calls no
        arm asked for, and in this run it would also keep synchronizing devices
        inside the un-instrumented pass that exists to have no synchronization
        in it.
        """
        ok = True
        for owner, name, had_own, previous, timed in reversed(self._installed):
            try:
                if owner == "model":
                    if had_own:
                        setattr(self.model, name, previous)
                    else:
                        delattr(self.model, name)
                    ok = ok and getattr(self.model, name, None) is not timed
                else:
                    setattr(self.vcd_utils, name, previous)
                    ok = ok and getattr(self.vcd_utils, name,
                                        None) is not timed
            except Exception:                                     # noqa: BLE001
                ok = False
        self._installed = []
        return ok

    # -- the reading ---------------------------------------------------------
    def report(self, wall_s):
        """The phase rows for one reconstruction, wrapped and derived.

        The two derived rows are remainders rather than measurements, and they
        are labelled that way: the iteration loop is _vcd_recon minus its own
        setup phases, and the last row is the public call minus both top-level
        phases -- the input checks, the gather back to the host and the
        bookkeeping around them.
        """
        rows = []
        for seam in PHASE_SEAMS:
            name = seam["name"]
            present = name not in self.missing
            total = self.totals.get(name, 0.0) if present else None
            rows.append(dict(phase=name, seconds=total,
                             calls=self.calls.get(name, 0) if present else None,
                             within=seam["within"], present=present,
                             kind="wrapped"))
        vcd = self.totals.get("_vcd_recon")
        if vcd is not None and "_vcd_recon" not in self.missing:
            setup = sum(self.totals.get(name, 0.0) for name in VCD_SETUP_PHASES
                        if name not in self.missing)
            rows.append(dict(phase=DERIVED_ITERATIONS, seconds=vcd - setup,
                             calls=None, within="_vcd_recon", present=True,
                             kind="derived"))
        top = sum(self.totals.get(name, 0.0) for name in TOP_LEVEL_PHASES
                  if name not in self.missing)
        rows.append(dict(phase=DERIVED_OTHER, seconds=wall_s - top, calls=None,
                         within=None, present=True, kind="derived"))
        for row in rows:
            row["share_of_wall"] = (row["seconds"] / wall_s
                                    if row["seconds"] is not None and wall_s
                                    else None)
        return rows


# ── the Triton launch timers ──────────────────────────────────────────────────
def _launch_record():
    return {direction: dict(calls=0, all_s=0.0, compiling_calls=0,
                            compiling_s=0.0, compiling_durations_s=[])
            for direction in ("forward", "back")}


def _timed_launch(original, direction, record, launch_keys):
    """One host clock around a bound projection body, recording per call whether
    that call GREW the shared launch-key set.

    A Triton wrapper adds a key the first time it launches a given
    configuration, so a call that grew the set is a call that compiled.  That is
    the launch-key set's own accounting rather than an inference from a
    duration.

    THE WRAPPER COPIES ``_view_batch_cost`` AND NOTHING ELSE.  The driver reads
    that attribute off the body it is about to call to choose a view batch, and
    a bare closure would drop it, push the driver onto the charge it applies to
    a general torch body, and change the very batching this arm reports.  It
    does NOT copy ``_mbirtorch_no_compile``: that marker is read by the compile
    step, which has already run by the time these bound entries exist, and
    copying everything with functools.wraps would carry it into a place it does
    not belong.

    The wrapper must not swallow or convert the body's return value, because the
    driver assigns and accumulates it.
    """
    entry = record[direction]

    def timed(*args, **kwargs):
        before = len(launch_keys)
        start = time.perf_counter()
        result = original(*args, **kwargs)
        elapsed = time.perf_counter() - start
        entry["calls"] += 1
        entry["all_s"] += elapsed
        if len(launch_keys) > before:
            entry["compiling_calls"] += 1
            entry["compiling_s"] += elapsed
            if len(entry["compiling_durations_s"]) < 32:
                entry["compiling_durations_s"].append(elapsed)
        return result

    cost = getattr(original, "_view_batch_cost", None)
    if cost is not None:
        timed._view_batch_cost = cost
    return timed


class LaunchTimers:
    """Timers on the bodies the driver really calls, one per device.

    Installed on ``projector_functions._fwd_body_per_dev`` and its back twin,
    which is where every driver -- plain and sharded -- reads its body from.
    EVERY index is wrapped, not just the first: a four-device arm launches from
    four bound entries and wrapping one of them would report a quarter of the
    calls.

    The bound entry for a hand-written kernel IS the Triton wrapper, because the
    compile step hands a no-compile body back unchanged; the row records whether
    that identity held, so a run where the torch bodies were bound instead says
    so rather than reporting an empty launch record as if it meant something.
    """

    BODY_LISTS = (("forward", "_fwd_body_per_dev"),
                  ("back", "_back_body_per_dev"))

    def __init__(self, model, launch_keys, triton_parallel_module):
        self.model = model
        self.launch_keys = launch_keys
        self.triton_parallel = triton_parallel_module
        self.record = _launch_record()
        self.bound = {}
        self.cost_copied = {}
        self._installed = []          # (list object, index, original)
        self.error = None

    def install(self):
        try:
            functions = self.model.projector_functions
        except Exception as exc:                                  # noqa: BLE001
            self.error = "{}: {}".format(type(exc).__name__, exc)
            return self
        expected = {
            "forward": getattr(self.triton_parallel,
                               "_parallel_forward_view_batch_triton", None),
            "back": getattr(self.triton_parallel,
                            "_parallel_back_view_batch_triton", None)}
        for direction, attr in self.BODY_LISTS:
            bodies = getattr(functions, attr, None)
            if bodies is None:
                self.error = "the projector functions hold no {}".format(attr)
                continue
            self.bound[direction] = [getattr(body, "__name__", str(body))
                                     for body in bodies]
            self.cost_copied[direction] = [
                getattr(body, "_view_batch_cost", None) is not None
                for body in bodies]
            self.bound[direction + "_is_triton_wrapper"] = [
                body is expected[direction] for body in bodies]
            for index, original in enumerate(bodies):
                bodies[index] = _timed_launch(original, direction, self.record,
                                              self.launch_keys)
                self._installed.append((bodies, index, original))
        return self

    def restore(self):
        ok = True
        for bodies, index, original in reversed(self._installed):
            try:
                bodies[index] = original
                ok = ok and bodies[index] is original
            except Exception:                                     # noqa: BLE001
                ok = False
        self._installed = []
        return ok

    def report(self):
        out = dict(error=self.error, bound_bodies=self.bound,
                   wrapper_carries_view_batch_cost=self.cost_copied,
                   wrapper_copies_no_compile_marker=False)
        for direction in ("forward", "back"):
            out[direction] = dict(self.record[direction])
        out["calls"] = sum(self.record[d]["calls"] for d in ("forward", "back"))
        out["compiling_calls"] = sum(self.record[d]["compiling_calls"]
                                     for d in ("forward", "back"))
        out["compiling_s"] = sum(self.record[d]["compiling_s"]
                                 for d in ("forward", "back"))
        out["all_calls_s"] = sum(self.record[d]["all_s"]
                                 for d in ("forward", "back"))
        return out


# ── what the compilers themselves report ──────────────────────────────────────
def _jsonable(value):
    """A value a jsonl row can hold, whatever shape the tool handed back.

    ``compile_times`` has returned a formatted string in some torch versions and
    a (headers, rows) pair in others.  Both are recorded as they came rather
    than reshaped into one, because the point of that field is to be the tool's
    own words.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def dynamo_before():
    """The dynamo state to measure a reconstruction against.

    The counters are CLEARED, so the row that follows describes this
    reconstruction alone.  The cumulative per-phase nanosecond totals are
    snapshotted instead of cleared: a delta needs no reset and cannot be
    disturbed by anything else in the process.
    """
    out = dict(counters_reset=False, cumulative_ns_before=None,
               error=None)
    try:
        from torch._dynamo import utils as dynamo_utils

        counters = getattr(dynamo_utils, "counters", None)
        if counters is not None and hasattr(counters, "clear"):
            counters.clear()
            out["counters_reset"] = True
        cumulative = getattr(dynamo_utils, "cumulative_time_spent_ns", None)
        if cumulative is not None:
            out["cumulative_ns_before"] = {str(k): float(v)
                                           for k, v in cumulative.items()}
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def dynamo_after(before):
    """What dynamo says it did during the reconstruction just finished.

    ``compile_times(repr='csv', aggregate=True)`` is recorded exactly as it
    comes back.  The counters are the ones cleared before the call when the API
    allowed it, and the row says which -- a cumulative counter read as a
    per-reconstruction one would overstate every reconstruction after the first.
    """
    out = dict(before)
    out["compile_times_csv"] = None
    out["counters"] = None
    out["counters_are_cumulative"] = not before.get("counters_reset")
    out["cumulative_ns_delta"] = None
    out["compile_phase_seconds"] = None
    out["compile_outer_phase"] = None
    out["compile_seconds_outer"] = None
    out["compile_seconds_sum_of_phases"] = None
    try:
        from torch._dynamo import utils as dynamo_utils

        out["compile_times_csv"] = _jsonable(
            dynamo_utils.compile_times(repr="csv", aggregate=True))
        counters = getattr(dynamo_utils, "counters", None)
        if counters is not None:
            out["counters"] = {str(key): _jsonable(dict(value))
                               if hasattr(value, "items") else _jsonable(value)
                               for key, value in counters.items()}
        cumulative = getattr(dynamo_utils, "cumulative_time_spent_ns", None)
        if cumulative is not None:
            after = {str(k): float(v) for k, v in cumulative.items()}
            start = before.get("cumulative_ns_before") or {}
            delta = {key: after[key] - start.get(key, 0.0) for key in after
                     if after[key] - start.get(key, 0.0) > 0.0}
            out["cumulative_ns_delta"] = delta
            seconds = {key: value / 1e9 for key, value in delta.items()}
            out["compile_phase_seconds"] = seconds
            # THESE PHASES NEST.  entire_frame_compile contains
            # backend_compile, which contains inductor_compile, which contains
            # code_gen, and so on, so their sum double counts and can exceed
            # the wall of the reconstruction it was measured over.  The
            # outermost phase is the honest single number, and it is simply the
            # largest, since an inner phase cannot outlast the one that
            # contains it.  The sum is kept beside it, named for what it is.
            if seconds:
                outer = max(seconds.items(), key=lambda item: item[1])
                out["compile_outer_phase"] = outer[0]
                out["compile_seconds_outer"] = outer[1]
                out["compile_seconds_sum_of_phases"] = sum(seconds.values())
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


# ── the shared measurement ────────────────────────────────────────────────────
def peak_readings(torch_module, devices, cuda):
    """The peak allocated bytes on every device in the list.

    Per device rather than per process: a four-device arm that loaded one device
    and left three idle is a different reading from one that divided the work,
    and a single maximum cannot tell them apart.
    """
    if not cuda:
        return None
    return [int(torch_module.cuda.max_memory_allocated(d)) for d in devices]


def reset_peaks(torch_module, devices, cuda):
    if not cuda:
        return
    for device in devices:
        torch_module.cuda.reset_peak_memory_stats(device)


def convergence_record(recon_dict):
    """The forward-model error the reconstruction reports, and whether it fell.

    RECORDED, NEVER GATED.  A reconstruction whose error did not fall is a
    reading about the problem, and this run is not the instrument that should
    judge it.
    """
    out = dict(fm_rmse=None, fm_rmse_decreased=None, num_iterations=None)
    try:
        params = (recon_dict or {}).get("recon_params") or {}
        rmse = [float(v) for v in (params.get("fm_rmse") or [])]
        out["fm_rmse"] = rmse
        out["num_iterations"] = params.get("num_iterations")
        if len(rmse) >= 2:
            out["fm_rmse_decreased"] = bool(rmse[-1] < rmse[0])
    except Exception as exc:                                      # noqa: BLE001
        out["convergence_error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def _sync_devices(torch_module, devices, cuda):
    if not cuda:
        return
    for device in devices:
        torch_module.cuda.synchronize(device)


def one_reconstruction(model, sinogram, weights, devices, cuda, index,
                       instrumented, modules):
    """One reconstruction, with the instruments installed around it or not.

    The wrappers go on immediately before the call and come off in a finally,
    which matters twice here: a wrapper left behind would keep timing calls no
    arm asked for, and it would keep synchronizing inside the plain pass whose
    whole purpose is to have no synchronization in it.

    The seed is reset immediately before the call, so the second reconstruction
    of a cached arm is the same reconstruction as the first rather than a
    different draw of pixel partitions.
    """
    import numpy as np

    torch_module = modules["torch"]
    projectors = modules["projectors"]
    triton_cone = modules["triton_cone"]
    triton_parallel = modules["triton_parallel"]
    vcd_utils = modules["vcd_utils"]
    launch_keys = getattr(triton_cone, "_COMPILED_LAUNCH_KEYS", set())
    compile_cache = getattr(projectors, "_COMPILE_CACHE", {})

    row = dict(index=index, instrumented=bool(instrumented),
               iterations=VCD_ITERATIONS, seed=VCD_SEED,
               launch_keys_before=len(launch_keys),
               compile_cache_before=len(compile_cache))

    reset_peaks(torch_module, devices, cuda)
    dynamo_state = dynamo_before()

    timers = None
    launches = None
    if instrumented:
        timers = PhaseTimers(model, devices, cuda, torch_module,
                             vcd_utils).install()
        launches = LaunchTimers(model, launch_keys, triton_parallel).install()

    volume = None
    info = None
    try:
        np.random.seed(VCD_SEED)
        start = time.perf_counter()
        volume, info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        _sync_devices(torch_module, devices, cuda)
        wall = time.perf_counter() - start
    finally:
        if launches is not None:
            row["launch_timers_restored"] = launches.restore()
        if timers is not None:
            row["phases_restored"] = timers.restore()

    row["wall_s"] = wall
    row["peak_bytes_per_device"] = peak_readings(torch_module, devices, cuda)
    if row["peak_bytes_per_device"]:
        row["busiest_peak_bytes"] = max(row["peak_bytes_per_device"])
    row["launch_keys_after"] = len(launch_keys)
    row["launch_keys_added"] = (row["launch_keys_after"]
                                - row["launch_keys_before"])
    row["compile_cache_after"] = len(compile_cache)
    row["compile_cache_added"] = (row["compile_cache_after"]
                                  - row["compile_cache_before"])
    row["dynamo"] = dynamo_after(dynamo_state)
    row.update(convergence_record(info))
    if timers is not None:
        row["phases"] = timers.report(wall)
        row["phase_seams_missing"] = list(timers.missing)
        row["phase_sync_errors"] = list(timers.sync_errors)
        row["phase_sync_devices"] = [str(d) for d in timers.devices]
    if launches is not None:
        row["triton_launches"] = launches.report()
    volume = None
    info = None
    return row


# ── the workers ───────────────────────────────────────────────────────────────
def run_identity(cfg):
    """Which library this run is about to measure, and on what.

    Run first and in its own process, so the header row can name the torch
    version, the devices and the tree under test without the parent importing
    torch at all -- the parent never does, because a parent holding a CUDA
    context would sit inside every arm's peak reading.
    """
    import torch

    import mbirtorch

    row = dict(cfg)
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda else 1
    row.update(torch_version=torch.__version__,
               library_file=mbirtorch.__file__,
               device_count=count,
               device_names=[torch.cuda.get_device_name(i)
                             for i in range(count)] if cuda else [DEVICE],
               cuda=cuda,
               python=platform.python_version(),
               inductor_cache_dir=os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
               triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"))
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


def run_stage(cfg):
    """Build the sinogram once and write it to an npz with an md5 sidecar -- or
    verify and reuse the copy already on disk.

    THIS IS A SEPARATE PROCESS AND IT HAS ITS OWN CACHE DIRECTORIES.  Building a
    sinogram means forward projecting a phantom, which compiles the projection
    bodies and fills both compile caches.  If it filled an arm's directories,
    the arm that was supposed to start cold would start warm and this run would
    measure nothing it set out to measure.  So the staging writes into cache
    directories no arm ever reads, and it runs before any arm.

    The weights are NOT staged.  They are host arithmetic over the sinogram, and
    each arm builds and times them itself.
    """
    import numpy as np

    row = dict(cfg)
    shape = tuple(int(v) for v in cfg["cell"])

    found = find_staged()
    if found is not None:
        expected = recorded_md5(found)
        actual = file_md5(found)
        row.update(stage_path=found, reused=True, md5=actual,
                   md5_ok=(actual == expected), recorded_md5=expected,
                   bytes_on_disk=os.path.getsize(found),
                   reused_from=os.path.dirname(os.path.abspath(found)))
        if actual == expected:
            meta = read_staged(found, with_sinogram=False)
            row.update(recon_shape=meta["recon_shape"],
                       sinogram_shape=meta["sinogram_shape"],
                       geometry=meta["geometry"],
                       delta_voxel=meta["delta_voxel"],
                       phantom_fallback=meta["phantom_fallback"])
            row["shape_ok"] = (meta["sinogram_shape"] == list(shape)
                               and meta["geometry"] == "parallel")
            if not row["shape_ok"]:
                row["invalid_reasons"] = [
                    "the staged file at {} holds a {} sinogram of shape {}, "
                    "not a parallel {}".format(found, meta["geometry"],
                                               meta["sinogram_shape"],
                                               list(shape))]
        else:
            row["invalid_reasons"] = [
                "the staged file at {} hashes to {}, not the recorded {}"
                .format(found, actual, expected)]
        return row

    path = stage_write_path()
    angles = np.asarray(angles_for(shape), dtype=np.float32)
    started = time.perf_counter()
    model = build_model(shape, angles, device_list(1))
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    phantom, phantom_fallback = build_phantom(recon_shape)
    sinogram = np.ascontiguousarray(
        np.asarray(to_numpy(model.forward_project(phantom)), dtype=np.float32))
    phantom = None
    row["build_s"] = time.perf_counter() - started

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(path, sinogram=sinogram, angles=angles,
             sinogram_shape=np.asarray(shape, dtype=np.int64),
             recon_shape=np.asarray(recon_shape, dtype=np.int64),
             geometry=np.asarray("parallel"),
             delta_voxel=np.asarray(float(model.get_params("delta_voxel"))),
             phantom_fallback=np.asarray(phantom_fallback))
    sinogram = None
    digest = file_md5(path)
    with open(md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    row.update(stage_path=path, reused=False, md5=digest, md5_ok=True,
               shape_ok=True, recon_shape=list(recon_shape),
               sinogram_shape=list(shape), geometry="parallel",
               delta_voxel=float(model.get_params("delta_voxel")),
               phantom_fallback=phantom_fallback,
               bytes_on_disk=os.path.getsize(path),
               forward_body=str(getattr(model._view_batch_bodies()[0],
                                        "__name__", "?")),
               back_body=str(getattr(model._view_batch_bodies()[1],
                                     "__name__", "?")))
    return row


def run_arm(cfg):
    """One arm: a fresh process that imports the library, touches its devices,
    builds a model, loads the staged sinogram, builds the weights, and then
    reconstructs once or twice with everything timed.

    The order below is the order a user's script would run in, and each step is
    clocked where it happens.  The CUDA context is created FIRST, on purpose: a
    trivial allocation on each device pays the context cost where it can be seen
    rather than leaving it buried inside whichever of the model build or the
    device configuration happened to touch a device first.
    """
    import numpy as np

    torch_import_start = time.perf_counter()
    import torch
    torch_import_s = time.perf_counter() - torch_import_start

    mbirtorch_import_start = time.perf_counter()
    import mbirtorch
    mbirtorch_import_s = time.perf_counter() - mbirtorch_import_start
    import_end_from_process_start_s = time.time() - PROCESS_START

    from mbirtorch import projectors, triton_cone, triton_parallel, vcd_utils

    devices = list(cfg["devices"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row = dict(cfg, invalid_reasons=[], iterations=VCD_ITERATIONS,
               seed=VCD_SEED, torch_version=torch.__version__,
               library_file=mbirtorch.__file__, device=DEVICE, cuda=cuda,
               requested_devices=[str(d) for d in devices],
               requested_n_devices=len(devices),
               visible_devices=(torch.cuda.device_count() if cuda else 1),
               torch_import_s=torch_import_s,
               mbirtorch_import_s=mbirtorch_import_s,
               process_start_to_import_end_s=import_end_from_process_start_s,
               env_inductor_cache_dir=os.environ.get(
                   "TORCHINDUCTOR_CACHE_DIR"),
               env_triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"),
               env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
               env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    row["cache_dirs"] = dict(cfg["cache_dirs"])
    row["cache_dirs_set_in_env"] = (
        row["env_inductor_cache_dir"] == cfg["cache_dirs"]["inductor"]
        and row["env_triton_cache_dir"] == cfg["cache_dirs"]["triton"])
    row["cache_before"] = cache_states(cfg["cache_dirs"])

    # ── the CUDA context, first and alone ───────────────────────────────────
    context = []
    context_start = time.perf_counter()
    if cuda:
        for device in devices:
            one = time.perf_counter()
            probe = torch.zeros(1, device=device)
            torch.cuda.synchronize(device)
            context.append(dict(device=str(device),
                                seconds=time.perf_counter() - one))
            probe = None
    row["cuda_context_per_device"] = context
    row["cuda_context_s"] = time.perf_counter() - context_start

    # ── the staged input ────────────────────────────────────────────────────
    load_start = time.perf_counter()
    meta = load_staged(cfg["stage_path"])      # raises on an md5 mismatch
    row["staged_load_s"] = time.perf_counter() - load_start
    row.update(stage_path=cfg["stage_path"], staged_md5=meta["md5"],
               staged_md5_verified=True,
               staged_recon_shape=meta["recon_shape"],
               phantom_fallback=meta["phantom_fallback"],
               sinogram_shape=meta["sinogram_shape"])

    # ── the model ───────────────────────────────────────────────────────────
    model_start = time.perf_counter()
    model = mbirtorch.ParallelBeamModel(
        tuple(int(v) for v in meta["sinogram_shape"]), meta["angles"])
    row["model_build_s"] = time.perf_counter() - model_start

    configure_start = time.perf_counter()
    model.configure_devices(devices=list(devices))
    model.set_params(no_warning=True, verbose=0)
    row["configure_devices_s"] = time.perf_counter() - configure_start

    realized = [int(s) for s in model.get_params("recon_shape")]
    row["recon_shape"] = realized
    row["recon_shape_ok"] = (realized == list(meta["recon_shape"]))
    if not row["recon_shape_ok"]:
        row["invalid_reasons"].append(
            "this arm's model realized recon shape {}, but the staged cell was "
            "built at {}".format(realized, meta["recon_shape"]))
        return row

    try:
        fwd_body, back_body = model._view_batch_bodies()
        row["forward_body"] = getattr(fwd_body, "__name__", "?")
        row["back_body"] = getattr(back_body, "__name__", "?")
    except Exception as exc:                                      # noqa: BLE001
        row["body_error"] = "{}: {}".format(type(exc).__name__, exc)
    row["compile_enabled"] = bool(getattr(model, "compile_enabled", None))

    # ── the weights, on the host, before anything is timed as recon ─────────
    weights_start = time.perf_counter()
    weights = build_weights(meta["sinogram"])
    row["weights_build_s"] = time.perf_counter() - weights_start
    row["weights_kind"] = ("exp(-sinogram / (2 * max(sinogram))), built on the "
                           "host in this arm and timed apart from the "
                           "reconstruction")

    modules = dict(torch=torch, projectors=projectors,
                   triton_cone=triton_cone, triton_parallel=triton_parallel,
                   vcd_utils=vcd_utils)

    # ── the reconstructions ─────────────────────────────────────────────────
    # The plan's count is kept under its own name, because the row's "recons"
    # becomes the list of reconstruction records below.
    row["recons_planned"] = int(cfg["recons"])
    recons = []
    for index in range(int(cfg["recons"])):
        recons.append(one_reconstruction(
            model, meta["sinogram"], weights, devices, cuda, index,
            instrumented=True, modules=modules))
        if index == 0:
            # What a user waits through before the first volume exists: the
            # process start, the imports, the context, the load, the weights
            # and the reconstruction itself.
            row["first_volume_ready_s"] = time.time() - PROCESS_START
        # The layout is settled once the first reconstruction has run, so this
        # describes the layout every timed pass ran on.
        try:
            realized_devices = [str(d) for d in model.recon_placement.devices]
        except Exception:                                         # noqa: BLE001
            realized_devices = []
        row["realized_devices"] = realized_devices
        row["realized_n_devices"] = len(realized_devices)
        row["devices_as_asked"] = (realized_devices
                                   == [str(d) for d in devices])
    row["recons"] = recons

    # ── one plain pass, so the instrument's own cost is on the row ──────────
    if PLAIN_PASS:
        plain = one_reconstruction(model, meta["sinogram"], weights, devices,
                                   cuda, len(recons), instrumented=False,
                                   modules=modules)
        row["plain"] = plain
        row["plain_wall_s"] = plain["wall_s"]
        row["plain_note"] = (
            "an extra reconstruction of the same input with every wrapper "
            "removed and no added synchronization.  It is a WARM pass: in a "
            "two-reconstruction arm it is directly comparable to the second, "
            "instrumented one and their difference is what the instrument "
            "cost; in a one-reconstruction arm the cold pass beside it is not "
            "comparable to it.")
        if len(recons) >= 2 and recons[1].get("wall_s"):
            row["instrument_cost_s"] = recons[1]["wall_s"] - plain["wall_s"]
            row["instrument_cost_ratio"] = (recons[1]["wall_s"]
                                            / plain["wall_s"])

    row["cache_after"] = cache_states(cfg["cache_dirs"])
    row["cache_growth"] = {
        kind: dict(files=row["cache_after"][kind]["files"]
                   - row["cache_before"][kind]["files"],
                   bytes=row["cache_after"][kind]["bytes"]
                   - row["cache_before"][kind]["bytes"])
        for kind in row["cache_after"]}
    row["arm_total_s"] = time.time() - PROCESS_START
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

    THE TWO CACHE DIRECTORIES ARE THE VARIABLE UNDER TEST, so both are set here
    and neither is ever inherited.  A job that took TORCHINDUCTOR_CACHE_DIR from
    the shell and left TRITON_CACHE_DIR alone -- which is what the sbatch files
    in this series do -- would run its "cold" arm against a Triton cache in the
    user's home directory full of artifacts from earlier jobs.

    MBIRTORCH_NUM_DEVICES IS REMOVED rather than set.  Every arm names its
    devices one by one, so a process-wide count pin is not the mechanism here
    and setting one would only cap something else at a count this run did not
    ask for.

    PYTHONPATH IS INHERITED.  A candidate tree reached only through PYTHONPATH
    would otherwise be swapped for the installed one without a word.  Every job
    records the mbirtorch file it actually imported, so which tree ran is a fact
    on the row.
    """
    env = dict(os.environ)
    env.pop("MG57_DRY", None)           # a worker never prints a plan
    env.pop("MG57_ARMS", None)          # a worker runs its cfg, not a plan
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)  # it owns the peak counters
    env.pop("MBIRTORCH_WIDENING_GUARD", None)      # explicit layouts bypass it
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    dirs = cfg.get("cache_dirs")
    if dirs:
        env["TORCHINDUCTOR_CACHE_DIR"] = dirs["inductor"]
        env["TRITON_CACHE_DIR"] = dirs["triton"]
    else:
        env.pop("TORCHINDUCTOR_CACHE_DIR", None)
        env.pop("TRITON_CACHE_DIR", None)
    return env


def spawn(cfg, timeout_s):
    """Run one job in a NEW interpreter, with a hard time cap.

    A new process per job is the whole instrument, not tidiness: compiled
    callables, the kernel availability probes, the launch-key set and the CUDA
    context all live for the life of a process, and this run is measuring
    exactly what a process has to pay to acquire them.  The row travels through
    a file rather than through stdout, so the worker's own output streams into
    the job log while it runs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR,
                            "_mg57_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR,
                            "_mg57_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = job_env(cfg)
    env["MG57_CHILD"] = cfg_path
    env["MG57_CHILD_OUT"] = out_path
    start = time.perf_counter()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                              env=env, timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # The cap is a READING, not a fault: a cold start that cannot finish
        # inside it has told us something about this arm.
        timed_out = True
    wall = time.perf_counter() - start
    if timed_out:
        row = dict(cfg, error="timed out after {:.1f} min".format(
            timeout_s / 60.0), timed_out=True)
    elif not os.path.exists(out_path):
        # A job killed by the operating system's memory manager lands here,
        # having written nothing.  That is a reading too; the classifier decides.
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
    harness fault.  A run that does not fit the devices it was given is an
    outcome, so it must not be reported as a broken instrument."""
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


# ── the plan ──────────────────────────────────────────────────────────────────
def build_plan():
    """Every job, in run order: the identity probe, the staging, then the arms.

    The arm order is load-bearing.  A reuse arm reads the cache directories a
    fresh arm filled, so the fresh arm of a count must run before the reuse arm
    of that count; the declared order does that and MG57_ARMS is normalized back
    into it.
    """
    allowed = [spec["name"] for spec in arms_declared()]
    keep = _strict_subset("MG57_ARMS", allowed)

    probe = dict(kind="identity", job_id="identity",
                 cache_dirs=cache_dirs("probe"))
    stage = dict(kind="stage", job_id="stage", cell=list(cell()),
                 cache_dirs=cache_dirs("stage"))
    arms = []
    for spec in arms_declared():
        if spec["name"] not in keep:
            continue
        arms.append(dict(kind="arm", arm=spec["name"], job_id=spec["name"],
                         count=int(spec["count"]),
                         devices=device_list(int(spec["count"])),
                         cache_owner=spec["cache_owner"],
                         caches=spec["caches"], recons=int(spec["recons"]),
                         filled_by=spec["filled_by"], what=spec["what"],
                         cell=list(cell()),
                         cache_dirs=cache_dirs(spec["cache_owner"])))
    if not arms:
        raise ValueError("MG57_ARMS selects no arm")
    return probe, stage, arms


def sinogram_gib(shape):
    return (int(shape[0]) * int(shape[1]) * int(shape[2]) * 4) / 2 ** 30


def print_plan(probe, stage, arms):
    print("mg57 what a first reconstruction costs, and which parts of it are "
          "paid again: {} arm(s), device {}, {} VCD iteration(s)".format(
              len(arms), DEVICE, VCD_ITERATIONS))
    print("  every recorded measurement in this series reports a warm median "
          "and discards the cold pass, so what a user waits through on a first "
          "reconstruction has never been attributed.  This run attributes it, "
          "at the parallel {} cell, and separates work a compile cache "
          "removes, work every process pays once, and the reconstruction "
          "arithmetic.".format(tuple(cell())))
    print("  rows -> {}".format(RESULTS_DIR))
    print("  interpreter: {}".format(sys.executable))
    print("  PYTHONPATH:  {}".format(os.environ.get("PYTHONPATH") or "(none)"))
    print("  the parent process imports no torch: every device touch happens "
          "inside an arm")

    print("\n  {:<{w}}{:>{d}}{:>9}{:>8}  {}".format(
        "job", "devices", "cap min", "recons", "what it does", w=ARM_COL,
        d=DEV_COL))
    print("  {:<{w}}{:>{d}}{:>9}{:>8}  {}".format(
        probe["job_id"], "-", int(PROBE_TIMEOUT_S / 60), "-",
        "names torch, mbirtorch, the devices and the tree witnesses",
        w=ARM_COL, d=DEV_COL))
    print("  {:<{w}}{:>{d}}{:>9}{:>8}  {}".format(
        stage["job_id"], device_label(device_list(1)),
        int(STAGE_TIMEOUT_S / 60), "-",
        "reuses the staged sinogram ({:.1f} GiB) or builds it once, in its "
        "OWN cache directories".format(sinogram_gib(cell())), w=ARM_COL,
        d=DEV_COL))
    for cfg in arms:
        print("  {:<{w}}{:>{d}}{:>9}{:>8}  {}".format(
            cfg["job_id"], device_label(cfg["devices"]),
            int(ARM_TIMEOUT_S / 60), cfg["recons"], cfg["what"], w=ARM_COL,
            d=DEV_COL))

    print("\n  the two compile caches this run owns, per arm:")
    for cfg in arms:
        dirs = cfg["cache_dirs"]
        print("    {:<{w}} {:<7} inductor {}".format(
            cfg["job_id"],
            "fresh" if cfg["caches"] == "fresh" else "reuse", dirs["inductor"],
            w=ARM_COL))
        print("    {:<{w}} {:<7} triton   {}{}".format(
            "", "", dirs["triton"],
            "  (filled by {})".format(cfg["filled_by"]) if cfg["filled_by"]
            else "", w=ARM_COL))
    print("    a fresh arm's directories are emptied by this process before it "
          "starts; both variables are set in every arm's environment, because "
          "an unset TRITON_CACHE_DIR would send Triton's cache to the user's "
          "home directory and a cold arm would reuse compiles from earlier "
          "jobs")

    print("\n  the phase split, timed from outside the package and restored in "
          "a finally:")
    for seam in PHASE_SEAMS:
        print("    {:<{w}}  {}".format(
            seam["name"],
            "the whole call" if seam["within"] is None
            else "inside {}".format(seam["within"]), w=PHASE_COL))
    print("    {:<{w}}  {}".format(DERIVED_ITERATIONS,
                                   "derived: _vcd_recon minus its setup phases",
                                   w=PHASE_COL))
    print("    {:<{w}}  {}".format(DERIVED_OTHER,
                                   "derived: the wall minus both top-level "
                                   "phases", w=PHASE_COL))
    print("    every timer synchronizes the arm's devices before it starts and "
          "before it stops, which removes some overlap and lengthens the "
          "reconstruction slightly; that is the price of attribution, and each "
          "arm also runs one un-instrumented pass so the price is on the row")

    print("\n  compile attribution, from the tools' own accounting: dynamo's "
          "compile_times(repr='csv', aggregate=True) and its counters after "
          "each reconstruction, the size of the shared Triton launch-key set "
          "before and after, per-call timers on the two parallel wrappers "
          "recording which calls grew that set, and the number of compiled "
          "callables the projector layer holds.")

    print("\n  also per arm: the wall to import mbirtorch, to build the model, "
          "to configure the devices, the CUDA context cost as a first trivial "
          "allocation on each device, the staged load with its md5 check, the "
          "host weights build, and the peak allocated bytes per device.")

    if RECORDED_WARM_S and not SMOKE:
        print("\n  recorded context, never a threshold: this cell's warm "
              "median reads {} at one device and {} at four, from {}."
              .format("{:.3f} s".format(RECORDED_WARM_S[1]),
                      "{:.3f} s".format(RECORDED_WARM_S[4]),
                      RECORDED_WARM_SOURCE))

    if DEVICE != "cuda":
        print("\n  ON {} THE RUN DEGRADES.  There is one device, so neither "
              "four-device arm can run and a second process pointed at a cache "
              "the first filled cannot be exercised: this is ONE arm standing "
              "in for four, and it runs two reconstructions so the "
              "cold-then-warm bookkeeping, the phase split, the plain pass and "
              "the child protocol are all exercised.  There is no triton, so "
              "TRITON_CACHE_DIR governs nothing, the launch-key set never "
              "grows and the wrapped bodies are the torch ones.  The rows say "
              "all of this.".format(DEVICE.upper()))

    print("\n  exit code = INSTRUMENT HEALTH ONLY: every planned arm produced "
          "a row or a recorded out-of-memory or timeout, every arm loaded the "
          "staged input with its md5 verified, every arm realized the device "
          "count it asked for, the wrapped phases were restored, and the tree "
          "witnesses hold.  What any arm MEASURED never changes it, including "
          "a cold start that turns out to be slow.")
    print("  no library file is edited: the seams are wrapped from outside and "
          "put back in a finally")


# ── rows ──────────────────────────────────────────────────────────────────────
def write_row(sink, row):
    """One jsonl row, flushed.

    Flushed per row because a job that is killed mid-run should leave every row
    it had already finished.
    """
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return row


# ── the tables ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=2):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    if isinstance(value, str):
        return "{:>{w}}".format(value, w=width)
    if kind == "d":
        return "{:>{w}d}".format(int(round(float(value))), w=width)
    return "{:>{w}.{p}{k}}".format(value, w=width, p=prec, k=kind)


def _gb_list(values):
    if not values:
        return "-"
    return " / ".join("{:.2f}".format(v / 2 ** 30) for v in values)


def _bytes_gib(value):
    return "-" if value is None else "{:.2f}".format(value / 2 ** 30)


def arm_status(row):
    if not row:
        return "not planned"
    if row.get("error") and row.get("timed_out"):
        return "timeout"
    if row.get("error") and is_capacity_reading(row):
        return "capacity"
    if row.get("error"):
        return "error"
    if row.get("invalid_reasons"):
        return "invalid"
    if not row.get("recons"):
        return "no recon"
    return "ok"


def print_phase_table(arm, recon):
    """One reconstruction's phases, sorted by cost, with each phase's share of
    that reconstruction's wall."""
    wall = recon.get("wall_s")
    print("\n### {} reconstruction {} phases, wall {:.2f} s".format(
        arm, recon.get("index"), wall or 0.0))
    rows = [row for row in (recon.get("phases") or [])
            if row.get("seconds") is not None]
    rows.sort(key=lambda row: row["seconds"], reverse=True)
    print("| phase | seconds | share of wall | calls | within | kind |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| {} | {:.3f} | {} | {} | {} | {} |".format(
            row["phase"], row["seconds"],
            "{:.1%}".format(row["share_of_wall"])
            if row.get("share_of_wall") is not None else "-",
            row.get("calls") if row.get("calls") is not None else "-",
            row.get("within") or "-", row.get("kind")))
    absent = [row["phase"] for row in (recon.get("phases") or [])
              if not row.get("present")]
    if absent:
        print("Absent from this tree, so never timed: {}".format(
            ", ".join(absent)))
    print("A row with a phase named in 'within' is already counted inside that "
          "phase; the rows with '-' there sum to the wall.")


def print_arm_costs(arm, row):
    """The once-per-process costs this arm paid, outside any reconstruction."""
    print("\n### {} once-per-process costs".format(arm))
    print("| step | seconds |")
    print("|---|---|")
    for label, key in (("import torch", "torch_import_s"),
                       ("import mbirtorch", "mbirtorch_import_s"),
                       ("process start to end of import",
                        "process_start_to_import_end_s"),
                       ("CUDA context, all devices", "cuda_context_s"),
                       ("staged load and md5", "staged_load_s"),
                       ("build the model", "model_build_s"),
                       ("configure the devices", "configure_devices_s"),
                       ("build the weights on the host", "weights_build_s")):
        value = row.get(key)
        print("| {} | {} |".format(
            label, "-" if value is None else "{:.3f}".format(value)))
    for entry in row.get("cuda_context_per_device") or []:
        print("| CUDA context on {} | {:.3f} |".format(entry["device"],
                                                       entry["seconds"]))
    if row.get("first_volume_ready_s") is not None:
        print("| process start to the first volume | {:.3f} |".format(
            row["first_volume_ready_s"]))
    if row.get("arm_total_s") is not None:
        print("| process start to the end of the arm | {:.3f} |".format(
            row["arm_total_s"]))


def print_cache_table(arm, row):
    print("\n### {} cache directories, before and after".format(arm))
    print("| cache | directory | files before | GiB before | files after | "
          "GiB after |")
    print("|---|---|---|---|---|---|")
    before = row.get("cache_before") or {}
    after = row.get("cache_after") or {}
    for kind in sorted(set(list(before) + list(after))):
        start = before.get(kind) or {}
        end = after.get(kind) or {}
        print("| {} | {} | {} | {} | {} | {} |".format(
            kind, (start.get("path") or end.get("path") or "-"),
            start.get("files", "-"), _bytes_gib(start.get("bytes")),
            end.get("files", "-"), _bytes_gib(end.get("bytes"))))
    print("Both variables were set in this arm's environment: {}".format(
        row.get("cache_dirs_set_in_env")))


def print_compile_table(arm_rows, arms):
    print("\n### compile attribution, from the tools' own accounting")
    print("| arm | recon | wall s | dynamo outer phase | dynamo outer s | "
          "dynamo unique graphs | launch keys before -> after | compiled "
          "callables before -> after | bound bodies are the Triton wrappers | "
          "body calls | compiling calls | compiling s | all calls s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in arms:
        row = arm_rows.get(cfg["arm"]) or {}
        for recon in row.get("recons") or []:
            dynamo = recon.get("dynamo") or {}
            counters = dynamo.get("counters") or {}
            stats = counters.get("stats") or {}
            launches = recon.get("triton_launches") or {}
            bound = launches.get("bound_bodies") or {}
            wrappers = [flag for key in ("forward_is_triton_wrapper",
                                         "back_is_triton_wrapper")
                        for flag in (bound.get(key) or [])]
            print("| {} | {} | {} | {} | {} | {} | {} -> {} | {} -> {} | {} | "
                  "{} | {} | {} | {} |".format(
                      cfg["arm"], recon.get("index"),
                      "-" if recon.get("wall_s") is None
                      else "{:.2f}".format(recon["wall_s"]),
                      dynamo.get("compile_outer_phase") or "-",
                      "-" if dynamo.get("compile_seconds_outer") is None
                      else "{:.2f}".format(dynamo["compile_seconds_outer"]),
                      stats.get("unique_graphs", "-"),
                      recon.get("launch_keys_before"),
                      recon.get("launch_keys_after"),
                      recon.get("compile_cache_before"),
                      recon.get("compile_cache_after"),
                      "-" if not wrappers else
                      ("all {}".format(len(wrappers)) if all(wrappers)
                       else "{} of {}".format(sum(1 for f in wrappers if f),
                                              len(wrappers))),
                      launches.get("calls", "-"),
                      launches.get("compiling_calls", "-"),
                      "-" if launches.get("compiling_s") is None
                      else "{:.2f}".format(launches["compiling_s"]),
                      "-" if launches.get("all_calls_s") is None
                      else "{:.2f}".format(launches["all_calls_s"])))
    print("The body timers wrap the entries the driver really calls, one per "
          "device; the wrapper column says how many of those entries were the "
          "hand-written Triton wrappers, because on a build without triton "
          "they are the torch bodies instead and no call can grow the "
          "launch-key set.  A call that GREW that set is a call that "
          "compiled.  The dynamo columns are that tool's own per-phase "
          "cumulative totals over the reconstruction; its phases NEST, so the "
          "outermost one is shown rather than a sum -- every phase's seconds "
          "are on the row in the jsonl, along with compile_times(repr='csv') "
          "verbatim and the counters.")


def compare_counts(count, arm_rows):
    """One device count: the first run, the cached process, and the in-process
    warm pass, with the differences between them named.

    The three walls are the same reconstruction three ways: with both caches
    empty, with both caches full but the process new, and with the process
    already warm.  Their differences are what the caches removed and what a
    fresh process still pays.
    """
    # Which arm is which is read off the arm's own cache plan, not off its
    # name: MG57_ARMS can select a subset, and the smoke runs one arm that is
    # not named for a count at all.
    at_count = [row for row in arm_rows.values()
                if int(row.get("count") or 0) == int(count)]
    first = next((row for row in at_count if row.get("caches") == "fresh"), {})
    cached = next((row for row in at_count if row.get("caches") == "reuse"), {})

    def wall(row, index):
        recons = row.get("recons") or []
        if len(recons) > index:
            return recons[index].get("wall_s")
        return None

    # The in-process warm pass is the second reconstruction of whichever arm
    # ran two.  Normally that is the cached arm; when only a fresh arm ran two
    # (which is what the smoke does) the row says where the number came from.
    warm_source = cached if len(cached.get("recons") or []) > 1 else (
        first if len(first.get("recons") or []) > 1 else {})
    out = dict(row="comparison", count=int(count),
               first_run_arm=first.get("arm"), cached_arm=cached.get("arm"),
               warm_from_arm=warm_source.get("arm"),
               first_run_cold_s=wall(first, 0),
               cached_cold_s=wall(cached, 0),
               in_process_warm_s=wall(warm_source, 1),
               plain_warm_s=(cached.get("plain_wall_s")
                             or first.get("plain_wall_s")),
               # The recorded wall describes the real cell only, so the smoke
               # is not given a number its tiny cell could never land beside.
               recorded_warm_s=(None if SMOKE
                                else RECORDED_WARM_S.get(int(count))),
               recorded_warm_spread=(None if SMOKE
                                     else RECORDED_WARM_SPREAD.get(int(count))),
               first_run_process_s=first.get("first_volume_ready_s"),
               cached_process_s=cached.get("first_volume_ready_s"))
    a, b, c = (out["first_run_cold_s"], out["cached_cold_s"],
               out["in_process_warm_s"])
    out["cache_removed_s"] = (a - b) if (a is not None
                                         and b is not None) else None
    out["fresh_process_pays_s"] = (b - c) if (b is not None
                                              and c is not None) else None
    out["arithmetic_s"] = c
    out["first_over_warm"] = (a / c) if (a and c) else None
    out["cached_over_warm"] = (b / c) if (b and c) else None
    out["warm_over_recorded"] = (
        c / out["recorded_warm_s"]
        if c and out.get("recorded_warm_s") else None)
    # The once-per-process steps beside the walls, from the cached arm where
    # there is one: those are the numbers a cache cannot touch.
    source = cached or first
    out["per_process_from_arm"] = source.get("arm")
    out["per_process"] = {
        key: source.get(key)
        for key in ("torch_import_s", "mbirtorch_import_s",
                    "process_start_to_import_end_s", "cuda_context_s",
                    "staged_load_s", "model_build_s", "configure_devices_s",
                    "weights_build_s")}
    return out


def print_comparison(item):
    count = item["count"]
    print("\n### {} device(s): the same reconstruction three ways".format(
        count))
    print("| what | seconds | note |")
    print("|---|---|---|")
    rows = (
        ("first run, both caches empty", item.get("first_run_cold_s"),
         item.get("first_run_arm") or "no arm ran it"),
        ("new process, caches full", item.get("cached_cold_s"),
         item.get("cached_arm") or "no arm ran it"),
        ("in-process warm", item.get("in_process_warm_s"),
         "the second reconstruction in {}".format(item.get("warm_from_arm"))
         if item.get("warm_from_arm") else "no arm ran two"),
        ("in-process warm, un-instrumented", item.get("plain_warm_s"),
         "the plain pass"),
        ("recorded warm median", item.get("recorded_warm_s"),
         RECORDED_WARM_SOURCE if item.get("recorded_warm_s")
         else "no recorded wall for this cell"),
    )
    for label, value, note in rows:
        print("| {} | {} | {} |".format(
            label, "-" if value is None else "{:.2f}".format(value), note))
    print("\n| difference | seconds |")
    print("|---|---|")
    print("| what the caches removed (first run - new process) | {} |".format(
        "-" if item.get("cache_removed_s") is None
        else "{:.2f}".format(item["cache_removed_s"])))
    print("| what a fresh process still pays (new process - in-process warm) "
          "| {} |".format(
              "-" if item.get("fresh_process_pays_s") is None
              else "{:.2f}".format(item["fresh_process_pays_s"])))
    print("| the reconstruction arithmetic (in-process warm) | {} |".format(
        "-" if item.get("arithmetic_s") is None
        else "{:.2f}".format(item["arithmetic_s"])))
    print("\n| once per process, outside any reconstruction, in {} | seconds |"
          .format(item.get("per_process_from_arm") or "-"))
    print("|---|---|")
    for key, label in (("process_start_to_import_end_s",
                        "process start to end of import mbirtorch"),
                       ("torch_import_s", "of which, import torch"),
                       ("cuda_context_s", "CUDA context, all devices"),
                       ("staged_load_s", "staged load and md5"),
                       ("model_build_s", "build the model"),
                       ("configure_devices_s", "configure the devices"),
                       ("weights_build_s", "build the weights on the host")):
        value = (item.get("per_process") or {}).get(key)
        print("| {} | {} |".format(
            label, "-" if value is None else "{:.3f}".format(value)))
    print("| process start to the first volume, first run | {} |".format(
        "-" if item.get("first_run_process_s") is None
        else "{:.2f}".format(item["first_run_process_s"])))
    print("| process start to the first volume, cached | {} |".format(
        "-" if item.get("cached_process_s") is None
        else "{:.2f}".format(item["cached_process_s"])))


def summarize(identity, stage_row, arm_rows, comparisons, arms, findings,
              out_path):
    """The tables a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A slow
    cold start, a large cache, a compile that took a minute and an arm that ran
    out of memory are FINDINGS: they are printed and none of them touches the
    exit code.  A missing row, an md5 that did not verify, an arm that realized
    a different device count than it asked for, a wrapper that was not restored
    and a failed tree witness are instrument failures, because they mean the run
    did not measure what the plan said it would.
    """
    print("\n===== mg57 what a first reconstruction costs ({}) =====".format(
        out_path))
    broken = []
    findings = list(findings)

    header = ("{:<{w}}{:>{d}}{:>11}{:>11}{:>12}{:>10}"
              .format("arm", "devices", "recon 0 s", "recon 1 s", "busiest GB",
                      "state", w=ARM_COL, d=DEV_COL))
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
        recons = row.get("recons") or []
        busiest = max([r.get("busiest_peak_bytes") or 0 for r in recons] or [0])
        print("{:<{w}}{:>{d}}{}{}{}{:>10}".format(
            name, device_label(row.get("realized_devices")
                               or row.get("requested_devices") or []),
            _fmt(recons[0].get("wall_s") if len(recons) > 0 else None, 11),
            _fmt(recons[1].get("wall_s") if len(recons) > 1 else None, 11),
            _fmt(busiest / 2 ** 30 if busiest else None, 12),
            arm_status(row), w=ARM_COL, d=DEV_COL))
        print("{:<{w}}  bodies {} / {} | caches {} -> {} files".format(
            "", row.get("forward_body"), row.get("back_body"),
            sum((row.get("cache_before") or {}).get(k, {}).get("files", 0)
                for k in ("inductor", "triton")),
            sum((row.get("cache_after") or {}).get(k, {}).get("files", 0)
                for k in ("inductor", "triton")), w=ARM_COL))
        if row.get("plain_wall_s") is not None:
            print("{:<{w}}  plain (un-instrumented) pass {:.2f} s{}".format(
                "", row["plain_wall_s"],
                "  instrument cost {:.2f} s ({:.2f}x)".format(
                    row["instrument_cost_s"], row["instrument_cost_ratio"])
                if row.get("instrument_cost_s") is not None else "",
                w=ARM_COL))

        # -- instrument health, per arm ------------------------------------
        for reason in row.get("invalid_reasons") or []:
            print("    ARM CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(name, reason))
        if not row.get("staged_md5_verified"):
            broken.append("{}|the staged input was not loaded with its md5 "
                          "verified".format(name))
        if row.get("devices_as_asked") is False:
            broken.append("{}|asked for {} and realized {}".format(
                name, row.get("requested_devices"),
                row.get("realized_devices")))
        for recon in recons:
            if recon.get("instrumented") and recon.get("phases_restored") \
                    is not True:
                broken.append("{}|reconstruction {}: the wrapped phases were "
                              "not restored".format(name, recon.get("index")))
            if recon.get("instrumented") and recon.get(
                    "launch_timers_restored") is not True:
                broken.append("{}|reconstruction {}: the launch timers were "
                              "not restored".format(name, recon.get("index")))
            for reason in recon.get("phase_sync_errors") or []:
                findings.append("{}: a phase timer could not synchronize: "
                                "{}".format(name, reason))
            missing = recon.get("phase_seams_missing") or []
            if missing:
                findings.append("{}: these seams do not exist in this tree and "
                                "were never timed: {}".format(
                                    name, ", ".join(missing)))
            if recon.get("fm_rmse_decreased") is False:
                findings.append("{}: reconstruction {}'s forward-model error "
                                "did not fall ({})".format(
                                    name, recon.get("index"),
                                    recon.get("fm_rmse")))
        if row.get("cache_dirs_set_in_env") is False:
            broken.append("{}|the two cache directories were not the ones this "
                          "arm's environment named".format(name))
        if cfg.get("caches") == "reuse":
            before = row.get("cache_before") or {}
            files = sum(entry.get("files", 0) for entry in before.values())
            if files == 0:
                findings.append(
                    "{}: its cache directories were empty at the start, so it "
                    "is not the cached reading it was planned to be; {} either "
                    "did not run or filled nothing".format(
                        name, cfg.get("filled_by")))
        launches = [r.get("triton_launches") for r in recons
                    if r.get("triton_launches")]
        for entry in launches:
            if entry.get("error"):
                findings.append("{}: the launch timers reported {}".format(
                    name, entry["error"]))

    # -- the staged input -----------------------------------------------------
    print("\n-- the staged input --")
    row = stage_row or {}
    print("  {} {}: md5 {}{}".format(
        row.get("geometry", "?"), tuple(row.get("sinogram_shape") or ()),
        row.get("md5", "-"),
        "  (reused from {})".format(row.get("reused_from"))
        if row.get("reused") else "  (built here)"))
    if row.get("phantom_fallback"):
        print("    phantom: {}".format(row["phantom_fallback"]))
    if row.get("build_s") is not None:
        print("    built in {:.1f} s, in cache directories no arm reads".format(
            row["build_s"]))
    if row.get("error"):
        broken.append("stage|{}".format(str(row["error"])[:200]))
    for reason in row.get("invalid_reasons") or []:
        print("    STAGE CHECK FAIL: {}".format(reason))
        broken.append("stage|{}".format(reason))

    # -- the tables -----------------------------------------------------------
    for cfg in arms:
        arm = arm_rows.get(cfg["arm"]) or {}
        if arm.get("error") or not arm.get("recons"):
            continue
        print_arm_costs(cfg["arm"], arm)
        print_cache_table(cfg["arm"], arm)
        for recon in arm["recons"]:
            if recon.get("phases"):
                print_phase_table(cfg["arm"], recon)
    for item in comparisons:
        print_comparison(item)
    print_compile_table(arm_rows, arms)

    # -- what ran -------------------------------------------------------------
    print("\n-- what ran --")
    row = identity or {}
    commit = (row.get("git") or {}).get("commit")
    print("  torch {} | triton {} | {} | {} device(s)".format(
        row.get("torch_version", "?"), row.get("triton_version", "?"),
        ", ".join(row.get("device_names") or ["?"]),
        row.get("device_count", "?")))
    print("  mbirtorch {} | commit {}{}".format(
        row.get("library_file", "?"), commit or "unknown",
        " (dirty)" if (row.get("git") or {}).get("dirty") else ""))
    if row.get("error"):
        print("    PROBE FAILED: {}".format(str(row["error"])[-300:]))
        broken.append("identity|{}".format(
            str(row["error"]).strip().splitlines()[-1][:200]))
    witnesses = (row.get("tree_witnesses") or {})
    if witnesses.get("ok"):
        print("  tree witnesses ok: the padded, recompile-remedied tree, both "
              "kernel modules import with their per-view cost attributes, and "
              "both geometries' selection hooks consult their availability "
              "checks and reach their kernels")
    else:
        print("  TREE WITNESSES: {}".format(witnesses))
        broken.append("tree witnesses|{}".format(witnesses))

    # -- instrument health ----------------------------------------------------
    print("\n-- instrument health --")
    print("  the exit code covers five things: every planned arm produced a "
          "row or a recorded out-of-memory or timeout, every arm loaded the "
          "staged input with its md5 verified, every arm realized the device "
          "count it asked for, the wrapped phases and launch timers were "
          "restored, and the tree witnesses hold.  What any arm MEASURED never "
          "changes it, including a cold start that turns out to be slow.")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm produced a result, every arm verified its "
              "staged input and realized the device count it asked for, every "
              "wrapper was restored, and the tree witnesses hold")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings outside the tables")

    return dict(row="summary", healthy=not broken, broken=broken,
                findings=findings, comparisons=comparisons,
                arms={name: dict(
                    status=arm_status(row),
                    walls=[r.get("wall_s") for r in (row.get("recons") or [])],
                    plain_wall_s=row.get("plain_wall_s"),
                    first_volume_ready_s=row.get("first_volume_ready_s"),
                    cache_growth=row.get("cache_growth"),
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
    probe, stage, arms = build_plan()
    print_plan(probe, stage, arms)
    if DRY:
        return 0
    findings = []

    # ── the identity probe ───────────────────────────────────────────────────
    print("\n-- identity probe --", flush=True)
    identity = spawn(probe, PROBE_TIMEOUT_S)
    print("  {}".format(identity.get("error") or
                        "torch {} | {} | {} device(s) | {}".format(
                            identity.get("torch_version"),
                            ", ".join(identity.get("device_names") or []),
                            identity.get("device_count"),
                            identity.get("library_file"))), flush=True)

    # ── the staging, in its own process and its own cache directories ────────
    print("\n-- the staged sinogram (reused when its md5 verifies) --",
          flush=True)
    empty_caches(stage["cache_dirs"])
    stage_row = spawn(stage, STAGE_TIMEOUT_S)
    if stage_row.get("error"):
        print("    ERROR: {}".format(str(stage_row["error"])[:400]), flush=True)
    else:
        print("    md5 {} {} recon {}".format(
            stage_row.get("md5"),
            "(reused from {})".format(stage_row.get("reused_from"))
            if stage_row.get("reused") else "(built here)",
            stage_row.get("recon_shape")), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, "mg57_cold_start_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    arm_rows = {}
    comparisons = []
    with open(out_path, "w") as sink:
        write_row(sink, dict(
            row="run_header", script="mg57_cold_start.py", node=RUN_LABEL,
            stamp=stamp, device=DEVICE, smoke=SMOKE,
            driver_python=sys.executable,
            pythonpath=os.environ.get("PYTHONPATH"),
            results_dir=RESULTS_DIR, cache_root=cache_root(),
            stage_search_dirs=stage_search_dirs(),
            identity=identity, tree_witnesses=identity.get("tree_witnesses"),
            staged_md5=stage_row.get("md5"), cell=list(cell()),
            vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
            plain_pass=PLAIN_PASS, arm_timeout_s=ARM_TIMEOUT_S,
            recorded_warm_s=RECORDED_WARM_S,
            recorded_warm_spread=RECORDED_WARM_SPREAD,
            recorded_warm_source=RECORDED_WARM_SOURCE,
            phase_seams=[dict(seam) for seam in PHASE_SEAMS],
            plan=[dict(kind=cfg["kind"], job_id=cfg["job_id"],
                       devices=cfg.get("devices"),
                       caches=cfg.get("caches"),
                       cache_dirs=cfg.get("cache_dirs"),
                       recons=cfg.get("recons"))
                  for cfg in [probe, stage] + list(arms)]))
        write_row(sink, dict(row="stage", **stage_row))

        for index, cfg in enumerate(arms):
            print("\n  [{}/{}] {} on {}, caches {}".format(
                index + 1, len(arms), cfg["job_id"],
                device_label(cfg["devices"]), cfg["caches"]), flush=True)
            if not stage_row.get("md5_ok") or not stage_row.get("stage_path"):
                # No verified input, so there is nothing honest to time.  The
                # staging row already carries the reason; this arm records that
                # it never ran rather than running on unverified bytes.
                row = dict(cfg, error="there is no md5-verified staged input; "
                                      "see the staging row")
            else:
                if cfg["caches"] == "fresh":
                    empty_caches(cfg["cache_dirs"])
                else:
                    for path in cfg["cache_dirs"].values():
                        os.makedirs(path, exist_ok=True)
                prepared = cache_states(cfg["cache_dirs"])
                print("    caches at start: {}".format(", ".join(
                    "{} {} files".format(kind, entry["files"])
                    for kind, entry in sorted(prepared.items()))), flush=True)
                row = spawn(dict(cfg, stage_path=stage_row["stage_path"],
                                 stage_md5=stage_row["md5"],
                                 cache_prepared=prepared), ARM_TIMEOUT_S)
            arm_rows[cfg["arm"]] = row
            write_row(sink, dict(row="arm", **row))
            if row.get("error"):
                print("    {}: {}".format(
                    "READING" if is_capacity_reading(row) else "ERROR",
                    str(row["error"]).strip().splitlines()[-1][:200]),
                    flush=True)
            elif row.get("invalid_reasons"):
                print("    INVALID: {}".format(
                    str(row["invalid_reasons"][0])[:250]), flush=True)
            else:
                for recon in row.get("recons") or []:
                    print("    reconstruction {} wall {:.2f} s  launch keys "
                          "{} -> {}  compiled callables {} -> {}".format(
                              recon.get("index"), recon.get("wall_s", 0.0),
                              recon.get("launch_keys_before"),
                              recon.get("launch_keys_after"),
                              recon.get("compile_cache_before"),
                              recon.get("compile_cache_after")), flush=True)
                if row.get("plain_wall_s") is not None:
                    print("    plain pass {:.2f} s".format(row["plain_wall_s"]),
                          flush=True)

        # ── the comparisons, once every arm of a count is in ─────────────────
        for count in counts_measured():
            item = compare_counts(count, arm_rows)
            comparisons.append(item)
            write_row(sink, item)

        summary = summarize(identity, stage_row, arm_rows, comparisons, arms,
                            findings, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        write_row(sink, summary)
    print("\nwrote {}".format(out_path))
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    if CHILD:
        sys.exit(_child_main(CHILD, CHILD_OUT))
    sys.exit(main())
