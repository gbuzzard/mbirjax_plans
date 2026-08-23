"""mg59 -- WHICH FRAMES DOES DYNAMO TRACE ON EVERY RECONSTRUCTION, AND WOULD
MARKING THE PIXEL-COUNT DIMENSION DYNAMIC REMOVE MOST OF THAT TRACING?

WHY THIS RUN EXISTS.  mg57 measured, at the parallel (1024, 1008, 992) cell,
that a reconstruction in a process with FULL compile caches still spends 4.42 s
at one device and 8.00 s at four inside torch's dynamo, by dynamo's own
`entire_frame_compile` phase, and that the unique graph count is 9 at one device
and 36 at four -- exactly four times, which follows the per-device compiled
instance design in mbirtorch/projectors.py (`maybe_compile(..., instance_key=i)`).
At four devices that 8.00 s precedes a 9.60 s reconstruction, so it is the
largest remaining per-process cost, and a compile cache cannot remove it: the
cache skips inductor code generation, not dynamo tracing.

What has never been said is WHICH frames those graphs belong to and WHY each
variant was made.  This run says it, and then tests one candidate change without
editing the library.

PART 1, ATTRIBUTION: NAME THE GRAPHS.  Every measured arm runs one
three-iteration parallel-beam reconstruction at that cell with full compile
caches and records, from the tools' own accounting:

    * `torch._dynamo.utils.counters` -- cleared before each reconstruction, so
      each row's frame and unique-graph totals describe that reconstruction.
    * `torch._dynamo.eval_frame._debug_get_cache_entry_list(code)` for every
      function the library compiled, which is the authoritative PER-FRAME
      variant count: one cache entry is one compiled variant of that frame.
      The functions are enumerated by reading (never writing) the keys of
      `mbirtorch.projectors._COMPILE_CACHE`, which is where `maybe_compile`
      records everything it compiled.
    * The `TORCH_LOGS=recompiles` output, written to a file by
      `TORCH_LOGS_OUT` and read back and summarized by the arm itself.  That
      log names the function, its file and line, and the GUARD FAILURE that
      forced each recompilation, which is the "why" the counters cannot give.
    * `torch._dynamo.utils.compile_times(repr='csv')` and the cumulative
      per-phase nanosecond totals, recorded as a before/after delta.

    A NOTE ON THE COUNTERS.  On the torch this was written against, the
    counters carry per-RECONSTRUCTION totals (frames, unique graphs, cache hits
    and misses) but no per-frame breakdown, so the per-frame variant counts come
    from the cache-entry list above rather than from the counters.  Both are
    recorded; a disagreement between the two is printed as a finding.

PART 2, THE CANDIDATE CHANGE, MEASURED WITHOUT EDITING THE LIBRARY.  The
hypothesis is that the compiled bodies specialize on the PIXEL-COUNT dimension,
which varies across a reconstruction because each VCD iteration projects a
different subset size: the default granularity list is
[1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128] and a three-iteration run visits
levels 4, 16 and 64, so a body called once per subset size compiles once per
size.  Marking that dimension dynamic (`torch._dynamo.mark_dynamic(tensor, dim)`)
would let one graph serve every size.

    THE MARKING RULE IS ONE SENTENCE, AND IT NEEDS NO PER-FUNCTION KNOWLEDGE.
    Before delegating to a wrapped body, the wrapper marks dimension 0 dynamic
    on every tensor argument whose dimension-0 size equals the PIXEL COUNT OF
    THE SUBSET THE DRIVER IS CURRENTLY UPDATING, and on nothing else.  That
    count is read where the driver hands it over, from the pixel index vector
    passed to the subset updater.  Arguments whose leading dimension is a fixed
    size -- a whole sinogram, the whole flat volume -- are therefore left alone,
    which matters: marking a dimension that never varies would buy a more
    general graph for nothing and would confound the walls this run compares.
    Every mark is counted, and so is every mark that raised.

    WHERE THE WRAPPERS GO.  Three seams, all reached from outside the package,
    all restored in a finally:

      1. `model.projector_functions._fwd_body_per_dev` and
         `_back_body_per_dev`, the per-device lists the driver reads its
         projection body from -- the seam mg54 and mg57 use.  FOR PARALLEL BEAM
         ON A BUILD WITH TRITON THESE ARE HAND-WRITTEN KERNELS, NOT COMPILED, so
         marking their inputs does nothing; the row records which bound bodies
         were the Triton wrappers so a reader can see that.  On a build without
         Triton they are compiled torch bodies and the marking does reach them.
      2. `model._get_update_direction`, a model method, wrapped as an instance
         attribute.  It receives the four per-subset arrays and passes them
         straight to the compiled `_diagonal_update_direction`.
      3. The per-device compiled bodies the subset updater holds.  These are
         built inside `create_vcd_subset_updater` and are reachable only through
         the closure of the updater it returns, so this harness wraps
         `model.create_vcd_subset_updater` as an instance attribute and, on the
         updater that comes back, replaces the entries of the per-device lists
         its closure cells hold.  Those lists are per-call objects created by
         that one call on this one model; NOTHING at module level in the
         mbirtorch package is written.  The lists are named on the row, with the
         name of every body found in each, so which callables were wrapped is a
         fact rather than an assumption.

    THE WRAPPER'S FORM IS NOT FREE CHOICE.  It copies the wrapped body's
    `_view_batch_cost` attribute when there is one, because the driver reads
    that attribute off the body it is about to call to choose a view batch, and
    a bare closure would drop it and change the very batching the arm reports.
    It copies nothing else -- in particular NOT `_mbirtorch_no_compile`, which
    is read in a different place at a different time.  This is a recorded trap
    from an earlier harness in this directory.

THE ARMS.  Eight, each a fresh subprocess, all with FULL compile caches.

    fill_base_n1   baseline, one device, its own EMPTY cache pair.  Not
                   measured: it exists to fill that pair.
    base_n1        baseline, one device, the pair fill_base_n1 filled.
    fill_cand_n1   candidate, one device, its own empty cache pair.
    cand_n1        candidate, one device, the pair fill_cand_n1 filled.
    ... and the same four at four devices.

    EACH MODE FILLS ITS OWN CACHE PAIR, AND THAT IS LOAD-BEARING.  The
    candidate's graphs are DIFFERENT graphs -- one dynamic graph where the
    baseline has several static ones -- so a candidate arm pointed at a cache
    the baseline filled would miss on every one of them and pay inductor code
    generation the baseline arm did not.  The wall comparison would then be
    about cache contents rather than about tracing.  Four cache owners, one per
    (mode, device count), each filled by its own throwaway arm.

    Each measured arm runs TWO reconstructions in its one process.  The first
    carries the tracing cost this run is about, and its wall is the wall to
    compare; the second is the in-process warm pass, which shows whether the
    candidate's one general graph is slower at steady state than the several
    specialized graphs it replaces.  Both are fingerprinted.

    Every arm: seed 13 reset immediately before each reconstruction, three VCD
    iterations, the stopping threshold disabled, the device list named one by
    one, and the sinogram loaded from a staged npz with its md5 verified.

THE VALUE CHECK.  Each reconstruction is fingerprinted in float64 -- the sum of
absolute values and the sum of squares over the returned volume, accumulated in
slabs so no full-size temporary is made -- and the comparison row carries the
RELATIVE difference between the candidate's fingerprint and the baseline's at
the same device count.  A candidate that changes the values is a failed
candidate and the row says so.  The tolerance is DECLARED, not derived: these
are float32 sums over about a billion elements and a differently scheduled
kernel reassociates them, so a relative difference below 1e-6 is called
unchanged.  The baseline's own one-device-to-four-device difference is printed
beside it as a yardstick measured in this same run.

THE INPUT IS STAGED BY A SEPARATE PROCESS, AND THAT IS THE WHOLE POINT.  A
measured arm LOADS its sinogram from an npz and verifies the md5 beside it.  It
never forward projects one, because a forward projection compiles the projection
bodies and traces the frames the arm exists to count.  So the staging runs
first, in its own process, with its own cache directories that no arm ever
reads.  A staged file that verifies is reused, including one an earlier
harness in this directory wrote for the same cell.

THE TWO CACHE DIRECTORIES ARE BOTH SET, PER ARM.  torch's inductor cache is
TORCHINDUCTOR_CACHE_DIR and Triton's JIT cache is TRITON_CACHE_DIR.  The sbatch
files in this series export the first and leave the second alone, which sends
Triton's cache to a directory under the user's home where artifacts from every
earlier job on the machine are sitting.  This harness owns both, sets both in
each arm's environment, and records both paths with the file count and total
bytes in each before and after the arm's work.

THIS RUN DECIDES NOTHING.  It edits no library file, changes no default and
proposes no remedy.  It prints numbers and tables.  The exit code reports
whether the INSTRUMENT worked -- never what it found: a candidate that helps
nothing, or that changes the values, is a recorded result and not a failure.

OUTPUT.  One jsonl under MG59_RESULTS, named
mg59_dynamo_tracing_<node>_<stamp>.jsonl: a header row carrying the torch and
mbirtorch identity, the GPUs, the tree witnesses and the staged file's md5; the
staging row; one row per arm; a comparison row per device count; and a summary
row.  Rows are flushed as they are written, so a job that runs out of wall time
yields everything it finished and MG59_ARMS re-runs the rest.  The parent
process never imports torch: every device touch happens in an arm.

Run:
    <torch python> mg59_dynamo_tracing.py             on a four-GPU node
    MG59_DRY=1 <any python> mg59_dynamo_tracing.py    the plan, then stop
    MG59_SMOKE=1 <python> mg59_dynamo_tracing.py      the local CPU smoke

Configuration is by environment variable only; there is no command line.  Export
from the SUBMITTING SHELL, never through an sbatch --export list, which slurm
splits on commas.  An unrecognized arm name is an error, not a silent skip.
    MG59_RESULTS=<dir>         where the jsonl, the staged npz, the recompile
                               logs and the cache directories this run owns go
    MG59_ARMS=a,b              subset of the arms, by arm name
    MG59_ITERATIONS=3          VCD iterations per reconstruction
    MG59_ARM_TIMEOUT_MIN=15    the per-arm hard cap
    MG59_STAGE_DIRS=a,b        extra directories to look for a staged npz in
    MG59_DRY=1                 print the plan and exit; imports no torch
    MG59_SMOKE=1               the local CPU smoke
    MG59_CHILD=<path>          internal: a job description.  Its presence puts
                               this process in child mode.
    MG59_CHILD_OUT=<path>      internal: where that child writes its row

THE LOCAL SMOKE runs the whole flow at a tiny CPU cell, and it degrades in
places rather than pretending otherwise.  There is one device, so the
four-device arms cannot run and the smoke runs the four one-device arms only.
There is no triton on a CPU install, so TRITON_CACHE_DIR governs nothing and the
two projection bodies ARE compiled torch bodies rather than hand-written
kernels -- which means the smoke exercises a seam the real run finds inert, and
the real run exercises frames the smoke reaches by the same route.  The rows and
the plan say all of this.  The smoke is plumbing, not a measurement.
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
#: statement can be.
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


SMOKE = _flag("MG59_SMOKE")
DRY = _flag("MG59_DRY")
#: The subprocess mode: the path to the job description this process is to run.
#: Non-empty means child mode.  The sbatch unsets it, so a stray value in the
#: submitting shell cannot turn the real run into a single job.
CHILD = os.environ.get("MG59_CHILD", "").strip()
CHILD_OUT = os.environ.get("MG59_CHILD_OUT", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: The cell.  Parallel beam at the recorded production class -- the cell whose
#: dynamo seconds and graph counts this run exists to attribute.
CELL = (1024, 1008, 992)
#: The smoke's stand-in.  Small enough that the whole flow -- staging, four
#: arms, two reconstructions each and every table -- runs in a few minutes on a
#: laptop CPU.
SMOKE_CELL = (24, 20, 16)

# ── the reconstruction protocol ───────────────────────────────────────────────
#: Three iterations with the stopping threshold disabled, so every
#: reconstruction in this run does exactly the same amount of work.  Three is
#: also what makes the subject visible: the partition sequence's first three
#: entries visit granularity levels 4, 16 and 64, so a body that specializes on
#: the pixel count sees three sizes.
VCD_ITERATIONS = _positive_int("MG59_ITERATIONS", 3)
#: Reset immediately before every reconstruction.  The library draws its pixel
#: partitions from numpy's global generator, so this is what makes two
#: reconstructions in one process, and two arms in two processes, the same
#: reconstruction.
VCD_SEED = 13

#: The per-arm hard cap.  An arm that exceeds it is killed and the timeout is
#: recorded as that arm's result rather than failing the job.
ARM_TIMEOUT_S = 60.0 * float(os.environ.get("MG59_ARM_TIMEOUT_MIN", "15"))
#: The identity probe imports torch and mbirtorch and reads the tree witnesses
#: and nothing else.
PROBE_TIMEOUT_S = 600.0
#: The staging job verifies an existing file's md5 or builds and writes one.
STAGE_TIMEOUT_S = 3600.0

# ── the value check ───────────────────────────────────────────────────────────
#: The relative fingerprint difference below which the candidate is called
#: unchanged.  DECLARED, not derived: the fingerprints are float32 sums over
#: about a billion elements, and a differently scheduled kernel reassociates
#: them, so some difference is expected from arithmetic order alone.  1e-6 is
#: far above that reassociation and far below anything that would mean the
#: candidate computed a different reconstruction.  The comparison row prints the
#: measured difference itself, so a reader is never left with only a verdict.
VALUE_TOLERANCE = 1e-6

#: Substrings that mark an arm's failure as a capacity READING rather than a
#: harness fault.  Matched case-insensitively.
CAPACITY_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory",
                    "failed to allocate", "memoryerror", "cannot allocate",
                    "memorypreflighterror")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("MG59_RESULTS",
                             os.path.join(SCRIPT_DIR, "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 16                  # wide enough for the longest arm id printed
DEV_COL = 13                  # wide enough for "cuda:0,1,2,3"
FRAME_COL = 40                # wide enough for the longest frame name printed
# ──────────────────────────────────────────────────────────────────────────────


# ── the arms ──────────────────────────────────────────────────────────────────
#: The arms, in run order.  A "fresh" arm's cache directories are emptied by the
#: parent before it starts; a "reuse" arm is pointed at the pair the named arm
#: filled.  ``mode`` is baseline or candidate.  ``measured`` says whether the
#: comparison tables read this arm: the fill arms exist only to put this mode's
#: compiled artifacts in this mode's cache pair.
REAL_ARMS = (
    dict(name="fill_base_n1", count=1, mode="baseline", cache_owner="base_n1",
         caches="fresh", recons=1, measured=False, filled_by=None,
         what="fills the baseline one-device cache pair; not measured"),
    dict(name="base_n1", count=1, mode="baseline", cache_owner="base_n1",
         caches="reuse", recons=2, measured=True, filled_by="fill_base_n1",
         what="baseline, one device, full caches"),
    dict(name="fill_cand_n1", count=1, mode="candidate",
         cache_owner="cand_n1", caches="fresh", recons=1, measured=False,
         filled_by=None,
         what="fills the candidate one-device cache pair; not measured"),
    dict(name="cand_n1", count=1, mode="candidate", cache_owner="cand_n1",
         caches="reuse", recons=2, measured=True, filled_by="fill_cand_n1",
         what="candidate, one device, full caches"),
    dict(name="fill_base_n4", count=4, mode="baseline", cache_owner="base_n4",
         caches="fresh", recons=1, measured=False, filled_by=None,
         what="fills the baseline four-device cache pair; not measured"),
    dict(name="base_n4", count=4, mode="baseline", cache_owner="base_n4",
         caches="reuse", recons=2, measured=True, filled_by="fill_base_n4",
         what="baseline, four devices, full caches"),
    dict(name="fill_cand_n4", count=4, mode="candidate",
         cache_owner="cand_n4", caches="fresh", recons=1, measured=False,
         filled_by=None,
         what="fills the candidate four-device cache pair; not measured"),
    dict(name="cand_n4", count=4, mode="candidate", cache_owner="cand_n4",
         caches="reuse", recons=2, measured=True, filled_by="fill_cand_n4",
         what="candidate, four devices, full caches"),
)
#: The smoke's arms: the four one-device arms, on CPU.  A laptop has one device,
#: so the four-device pair cannot run at all.
SMOKE_ARMS = tuple(spec for spec in REAL_ARMS if int(spec["count"]) == 1)


def arms_declared():
    return SMOKE_ARMS if SMOKE else REAL_ARMS


def cell():
    return SMOKE_CELL if SMOKE else CELL


def device_list(count):
    """The devices an arm is given, named one by one."""
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
    """The device counts that have measured arms in this run, ascending."""
    return sorted({int(spec["count"]) for spec in arms_declared()
                   if spec["measured"]})


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a repeat
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
            raise ValueError("{}: {!r} is not one of this run's: {}".format(
                env_name, token, ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError("{}: no valid tokens in {!r}.  The valid ones are: {}"
                         .format(env_name, raw, ", ".join(allowed)))
    # Normalized to the DECLARED order: the run order is load-bearing, because a
    # reuse arm reads the directories a fill arm filled.
    return [name for name in allowed if name in chosen]


# ── the two compile caches this harness owns ──────────────────────────────────
def cache_root():
    return os.path.join(RESULTS_DIR, "caches")


def cache_dirs(owner):
    """The inductor and Triton cache directories one owner uses.

    BOTH are named here and both are set in the child's environment.  Setting
    only the first -- which is what the sbatch files in this series do -- leaves
    Triton's JIT cache in the user's home directory, where an arm would reuse
    compiled Triton artifacts from every earlier job on the machine.
    """
    base = os.path.join(cache_root(), owner)
    return dict(inductor=os.path.join(base, "inductor"),
                triton=os.path.join(base, "triton"))


def cache_state(path):
    """One cache directory's file count and total bytes.

    Walked rather than listed: both caches nest their artifacts several levels
    deep, so a top-level listing would report a handful of directories and no
    files at all.
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
    """Remove and recreate both directories, so a fill arm really starts empty.

    Done in the PARENT, before the arm is spawned, because the child cannot
    empty a directory torch has already read.
    """
    for path in dirs.values():
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)


# ── the staged input ──────────────────────────────────────────────────────────
def stage_name():
    """The staged filename this run writes."""
    return "mg59_stage_parallel_{}x{}x{}.npz".format(*tuple(cell()))


def stage_names_accepted():
    """Every filename a staged copy of this cell may already carry.

    The sibling cold-start harness stages the same parallel cell in the same
    format with the same md5 sidecar, and a verified 3.8 GiB file is worth
    reusing rather than rebuilding.  Whatever is found is md5-verified and its
    shape and geometry are checked before any arm reads it.
    """
    shape = tuple(cell())
    return (stage_name(),
            "mg57_stage_parallel_{}x{}x{}.npz".format(*shape))


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
    results directory, the script's local default, the sibling cold-start
    harness's results directory beside this one, and anything MG59_STAGE_DIRS
    names.  Duplicates are dropped and order is kept."""
    candidates = [RESULTS_DIR, os.path.join(SCRIPT_DIR, "results"),
                  os.path.join(os.path.dirname(os.path.abspath(RESULTS_DIR)),
                               "mg57_cold_start"),
                  os.path.join(SCRIPT_DIR, "results", "mg57_cold_start")]
    extra = os.environ.get("MG59_STAGE_DIRS", "").strip()
    if extra:
        candidates.extend(part.strip() for part in extra.split(",")
                          if part.strip())
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
    for directory in stage_search_dirs():
        for name in stage_names_accepted():
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

    The device list is EXPLICIT, which turns the automatic choice off, so each
    arm's layout is a fact on its row rather than a policy outcome.
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
    host.  Timed and reported apart from the reconstruction."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32,
                                                             copy=False)


def fingerprint(volume):
    """A float64 fingerprint of one reconstruction: the sum of absolute values
    and the sum of squares, with the element count and dtype beside them.

    Accumulated in SLABS along the leading axis.  The real cell's volume is
    about 3.8 GiB of float32 and ``np.sum(np.abs(v))`` would make a temporary
    that size; the slabs bound it, and the float64 accumulator makes the sums
    reproducible enough to compare between arms.

    This is the whole value check: two arms that reconstruct the same input with
    the same seed must land on the same pair of numbers, up to the reassociation
    a different kernel schedule causes.
    """
    import numpy as np

    array = np.asarray(to_numpy(volume))
    out = dict(shape=[int(s) for s in array.shape], dtype=str(array.dtype),
               elements=int(array.size), abs_sum=None, sq_sum=None,
               finite=None, error=None)
    try:
        if array.ndim == 0 or array.size == 0:
            values = array.reshape(1, -1) if array.size else array.reshape(0, 1)
        else:
            values = array.reshape(int(array.shape[0]), -1)
        abs_sum = 0.0
        sq_sum = 0.0
        finite = True
        rows = int(values.shape[0])
        step = max(1, min(rows, 32)) if rows else 1
        for start in range(0, rows, step):
            slab = np.asarray(values[start:start + step], dtype=np.float64)
            abs_sum += float(np.sum(np.abs(slab)))
            sq_sum += float(np.sum(slab * slab))
            if finite and not bool(np.all(np.isfinite(slab))):
                finite = False
        out.update(abs_sum=abs_sum, sq_sum=sq_sum, finite=finite)
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def relative_difference(candidate, base):
    """|candidate - base| / |base|, or None when either is missing.  Zero base
    falls back to the absolute difference, which is the honest reading there."""
    if candidate is None or base is None:
        return None
    scale = abs(base)
    if scale == 0.0:
        return abs(candidate - base)
    return abs(candidate - base) / scale


# ── the tree under test ───────────────────────────────────────────────────────
def tree_witnesses():
    """What tree produced these numbers, measured rather than asserted.

    The first three are the block every job in this series carries: they say
    this is the padded, recompile-remedied tree the recorded walls were measured
    on, and the third matters directly here because a tree without it would hand
    a torch body eager python rather than a compiled one -- and compilation is
    the subject.

    The kernel pairs identify the committed tip and, for parallel, the wrappers
    whose presence is why the two projection bodies do NOT compile on this
    build.  The last block names the seams PART 2 wraps: the two model methods
    and the per-device lists inside the subset updater, checked by source
    inspection so the candidate cannot be silently wrapping nothing.

    Everything here is read by SOURCE INSPECTION and attribute lookup: no model
    is built, no device is touched and no CUDA is initialized.  A lookup that
    fails is recorded as failed; nothing here raises.
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
        record["maybe_compile_takes_instance_key"] = bool(
            "instance_key" in source)

        from mbirtorch import triton_multiaxis, triton_parallel
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

        parallel = kernel_pair(
            triton_parallel,
            (("forward", "_parallel_forward_view_batch_triton"),
             ("back", "_parallel_back_view_batch_triton")))
        record["parallel_kernels"] = parallel
        parallel_selection = inspect.getsource(
            ParallelBeamModel._view_batch_bodies)
        record["parallel_selection_consults_availability"] = bool(
            "parallel_forward_kernel_usable" in parallel_selection
            and "parallel_back_kernel_usable" in parallel_selection)
        record["parallel_selection_reaches_kernels"] = bool(
            "_parallel_forward_view_batch_triton" in parallel_selection
            and "_parallel_back_view_batch_triton" in parallel_selection)

        # -- the seams part 2 wraps -----------------------------------------
        from mbirtorch.projectors import Projectors
        from mbirtorch.tomography_model import TomographyModel

        record["projector_body_lists"] = bool(
            "_fwd_body_per_dev" in inspect.getsource(Projectors.__init__)
            and "_back_body_per_dev" in inspect.getsource(Projectors.__init__))
        record["has_get_update_direction"] = hasattr(
            TomographyModel, "_get_update_direction")
        record["has_create_vcd_subset_updater"] = hasattr(
            TomographyModel, "create_vcd_subset_updater")
        updater_source = inspect.getsource(
            TomographyModel.create_vcd_subset_updater)
        record["updater_body_lists"] = sorted(
            name for name in UPDATER_BODY_LISTS if name in updater_source)
        record["updater_binds_per_device"] = bool(
            "instance_key=i" in updater_source)

        # -- the introspection this run reads its numbers from --------------
        import torch
        from torch._dynamo import eval_frame as dynamo_eval_frame
        from torch._dynamo import utils as dynamo_utils

        record["has_mark_dynamic"] = callable(
            getattr(torch._dynamo, "mark_dynamic", None))
        record["has_cache_entry_list"] = callable(
            getattr(dynamo_eval_frame, "_debug_get_cache_entry_list", None))
        record["has_counters"] = getattr(dynamo_utils, "counters",
                                         None) is not None
        record["has_compile_times"] = callable(
            getattr(dynamo_utils, "compile_times", None))

        record["ok"] = bool(
            record["padded_kernel_width_504"] == 512
            and record["recompile_limit_floor"] >= 64
            and record["raise_on_compiling_thread"]
            and record["maybe_compile_takes_instance_key"]
            and all(entry["present"] and entry["named_like_a_kernel"]
                    and entry["has_view_batch_cost"]
                    for entry in parallel.values())
            and record["parallel_selection_consults_availability"]
            and record["parallel_selection_reaches_kernels"]
            and record["projector_body_lists"]
            and record["has_get_update_direction"]
            and record["has_create_vcd_subset_updater"]
            and len(record["updater_body_lists"]) == len(UPDATER_BODY_LISTS)
            and record["updater_binds_per_device"]
            and record["has_mark_dynamic"]
            and record["has_cache_entry_list"]
            and record["has_counters"]
            and record["has_compile_times"])
    except Exception as exc:                                      # noqa: BLE001
        record.update(available=False, ok=False,
                      reason="{}: {}".format(type(exc).__name__, exc))
    return record


# ── the candidate ─────────────────────────────────────────────────────────────
#: The names of the per-device lists of compiled bodies the subset updater
#: holds in its closure.  These are the callables `create_vcd_subset_updater`
#: binds once for all subsets, one instance per device thread, and they are the
#: frames a three-iteration run calls at three different subset sizes.
UPDATER_BODY_LISTS = ("qggmrf_grad_hess", "prior_line_terms", "lin_quad_const",
                      "lin_quad_weighted", "apply_update")


class Candidate:
    """The candidate change, installed from OUTSIDE the package and removed in
    a finally.

    WHAT IT DOES.  Before a wrapped body is called, every tensor argument whose
    leading dimension equals the pixel count of the subset the driver is
    currently updating has that dimension marked dynamic.  Nothing else is
    marked: an argument whose leading dimension never varies -- a whole
    sinogram, the whole flat volume -- would only buy a more general graph for
    nothing, and would confound the wall this arm reports.

    WHERE THE PIXEL COUNT COMES FROM.  The driver hands the subset's pixel
    index vector to the updater on every subset, so the wrapper around the
    updater records its length before delegating and clears it afterwards.  A
    body called while no subset is in flight -- the initial reconstruction and
    the Hessian diagonal, which use the whole pixel set and so have one shape --
    marks nothing.  The recorded count is written by one thread and read by the
    per-device worker threads of the same subset, which all see the same value.

    WHAT IT DOES NOT DO.  It writes nothing at module level in the mbirtorch
    package.  The two model methods are shadowed by instance attributes, which
    affect this model alone and are deleted again by name; the projector body
    lists belong to this model's projector functions; and the updater's
    per-device lists are objects one call on this model just created.
    """

    #: The two per-device body lists the projection driver reads from.  On a
    #: build with Triton these hold hand-written kernels, which do not compile,
    #: so marking their inputs does nothing -- the row records which entries
    #: were the Triton wrappers so that is visible rather than assumed.
    PROJECTOR_BODY_LISTS = (("forward", "_fwd_body_per_dev"),
                            ("back", "_back_body_per_dev"))

    def __init__(self, model, torch_module, triton_parallel_module):
        self.model = model
        self.torch = torch_module
        self.triton_parallel = triton_parallel_module
        #: The pixel count of the subset in flight, or None between subsets.
        self.pixels = dict(count=None)
        self.marks = {}
        self.wrapped = {}
        self.bound_is_triton_wrapper = {}
        self.cost_copied = {}
        self.notes = []
        self.mark_seconds = 0.0
        self.subsets_seen = 0
        self.subset_sizes = []
        self._undo = []
        self.error = None

    # -- the marking ---------------------------------------------------------
    def _mark(self, label, args, kwargs):
        """Mark dimension 0 dynamic on every tensor argument whose leading
        dimension is this subset's pixel count."""
        count = self.pixels["count"]
        entry = self.marks.setdefault(
            label, dict(calls=0, calls_with_a_subset=0, marked=0, errors=0,
                        first_error=None))
        entry["calls"] += 1
        if count is None:
            return
        entry["calls_with_a_subset"] += 1
        started = time.perf_counter()
        values = list(args) + [kwargs[key] for key in sorted(kwargs)]
        flat = []
        for value in values:
            # One level of nesting: the geometry's view-batch arguments arrive
            # as a tuple of tensors.
            if isinstance(value, (tuple, list)):
                flat.extend(value)
            else:
                flat.append(value)
        for value in flat:
            if not self.torch.is_tensor(value) or value.dim() < 1:
                continue
            if int(value.shape[0]) != int(count):
                continue
            try:
                self.torch._dynamo.mark_dynamic(value, 0)
                entry["marked"] += 1
            except Exception as exc:                              # noqa: BLE001
                entry["errors"] += 1
                if entry["first_error"] is None:
                    entry["first_error"] = "{}: {}".format(
                        type(exc).__name__, exc)[:200]
        self.mark_seconds += time.perf_counter() - started

    def _wrap(self, original, label):
        """One marking wrapper around one body.

        IT COPIES ``_view_batch_cost`` AND NOTHING ELSE.  The driver reads that
        attribute off the body it is about to call to choose a view batch, and a
        bare closure would drop it, push the driver onto the charge it applies
        to a general torch body, and change the very batching this arm reports.
        It does NOT copy ``_mbirtorch_no_compile``: that marker is read by the
        compile step, which has already run by the time these bound entries
        exist, and copying everything would carry it into a place it does not
        belong.

        The wrapper must not swallow or convert the body's return value, because
        every call site here assigns or unpacks it.
        """
        def marked(*args, **kwargs):
            self._mark(label, args, kwargs)
            return original(*args, **kwargs)

        cost = getattr(original, "_view_batch_cost", None)
        if cost is not None:
            marked._view_batch_cost = cost
        marked.__name__ = "marked_" + getattr(original, "__name__", "body")
        return marked

    # -- install and restore -------------------------------------------------
    def _install_projector_bodies(self):
        functions = getattr(self.model, "projector_functions", None)
        if functions is None:
            self.notes.append("the model holds no projector_functions")
            return
        expected = {
            "forward": getattr(self.triton_parallel,
                               "_parallel_forward_view_batch_triton", None),
            "back": getattr(self.triton_parallel,
                            "_parallel_back_view_batch_triton", None)}
        for direction, attr in self.PROJECTOR_BODY_LISTS:
            bodies = getattr(functions, attr, None)
            if bodies is None:
                self.notes.append("the projector functions hold no "
                                  "{}".format(attr))
                continue
            self.wrapped[attr] = [getattr(body, "__name__", str(body))
                                  for body in bodies]
            self.bound_is_triton_wrapper[attr] = [
                body is expected[direction] for body in bodies]
            self.cost_copied[attr] = [
                getattr(body, "_view_batch_cost", None) is not None
                for body in bodies]
            for index, original in enumerate(bodies):
                bodies[index] = self._wrap(original,
                                           "{}[{}]".format(attr, index))
                self._undo.append(("list", bodies, index, original))

    def _install_model_method(self, name):
        original = getattr(self.model, name, None)
        if original is None:
            self.notes.append("the model has no {}".format(name))
            return None
        had_own = name in vars(self.model)
        previous = vars(self.model).get(name)
        setattr(self.model, name, self._wrap(original, name))
        self._undo.append(("model", name, had_own, previous))
        return original

    def _install_updater(self):
        """Wrap ``create_vcd_subset_updater`` so the updater it returns has its
        per-device compiled bodies wrapped and its subset size recorded.

        The bodies are built inside that method and are reachable only through
        the closure of the updater it hands back, so the replacement happens
        there, on lists this one call just created.  ``stage_halos`` is copied
        onto the returned callable because the partition iterator looks for it
        by name and stages the qGGMRF boundary halos through it once per pass;
        dropping it would silently change the prior at every shard boundary.
        """
        name = "create_vcd_subset_updater"
        original_create = getattr(self.model, name, None)
        if original_create is None:
            self.notes.append("the model has no {}".format(name))
            return
        had_own = name in vars(self.model)
        previous = vars(self.model).get(name)

        def create(*args, **kwargs):
            updater = original_create(*args, **kwargs)
            try:
                cells = dict(zip(updater.__code__.co_freevars,
                                 updater.__closure__ or ()))
            except Exception as exc:                              # noqa: BLE001
                self.notes.append("the updater's closure could not be read: "
                                  "{}: {}".format(type(exc).__name__, exc))
                return updater
            for list_name in UPDATER_BODY_LISTS:
                cell = cells.get(list_name)
                if cell is None:
                    self.notes.append("the updater's closure holds no "
                                      "{}".format(list_name))
                    continue
                try:
                    bodies = cell.cell_contents
                except ValueError:
                    self.notes.append("the closure cell {} is empty".format(
                        list_name))
                    continue
                if not isinstance(bodies, list):
                    self.notes.append("the closure cell {} is a {}, not a "
                                      "list".format(list_name,
                                                    type(bodies).__name__))
                    continue
                self.wrapped[list_name] = [getattr(body, "__name__",
                                                   str(body))
                                           for body in bodies]
                for index, body in enumerate(bodies):
                    bodies[index] = self._wrap(
                        body, "{}[{}]".format(list_name, index))

            def counting(flat_recon, error_sinogram, pixel_indices):
                try:
                    self.pixels["count"] = int(len(pixel_indices))
                    self.subsets_seen += 1
                    if len(self.subset_sizes) < 512:
                        self.subset_sizes.append(int(len(pixel_indices)))
                except Exception:                                 # noqa: BLE001
                    self.pixels["count"] = None
                try:
                    return updater(flat_recon, error_sinogram, pixel_indices)
                finally:
                    self.pixels["count"] = None

            halos = getattr(updater, "stage_halos", None)
            if halos is not None:
                counting.stage_halos = halos
            else:
                self.notes.append("the updater carried no stage_halos")
            counting.__name__ = "counting_" + getattr(updater, "__name__",
                                                      "vcd_subset_updater")
            return counting

        setattr(self.model, name, create)
        self._undo.append(("model", name, had_own, previous))

    def install(self):
        try:
            self._install_projector_bodies()
            self._install_model_method("_get_update_direction")
            self._install_updater()
        except Exception as exc:                                  # noqa: BLE001
            self.error = "{}: {}".format(type(exc).__name__, exc)
        return self

    def restore(self):
        """Put every seam back and report whether it took.

        Reported rather than assumed: a wrapper left behind would keep marking
        inside a reconstruction that was supposed to run without it.
        """
        ok = True
        for item in reversed(self._undo):
            try:
                if item[0] == "list":
                    _, bodies, index, original = item
                    bodies[index] = original
                    ok = ok and bodies[index] is original
                else:
                    _, name, had_own, previous = item
                    if had_own:
                        setattr(self.model, name, previous)
                    else:
                        delattr(self.model, name)
            except Exception:                                     # noqa: BLE001
                ok = False
        self._undo = []
        return ok

    def report(self):
        return dict(
            error=self.error, notes=list(self.notes),
            wrapped_bodies=dict(self.wrapped),
            bound_bodies_are_triton_wrappers=dict(
                self.bound_is_triton_wrapper),
            wrapper_carries_view_batch_cost=dict(self.cost_copied),
            wrapper_copies_no_compile_marker=False,
            marks={key: dict(value) for key, value in self.marks.items()},
            marks_total=sum(entry["marked"] for entry in self.marks.values()),
            mark_errors_total=sum(entry["errors"]
                                  for entry in self.marks.values()),
            mark_seconds=self.mark_seconds,
            subsets_seen=self.subsets_seen,
            subset_sizes_seen=sorted(set(self.subset_sizes)))


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


def compiled_frames(projectors_module):
    """Every frame the library compiled, with its per-frame VARIANT COUNT.

    The functions come from the keys of ``projectors._COMPILE_CACHE``, which is
    where ``maybe_compile`` records everything it compiled; a key is either the
    function or a (function, instance key) pair, and several per-device
    instances of one function share one code object.  The variant count for each
    is the length of dynamo's own cache-entry list for that code object: one
    entry is one compiled variant, with its own guards.  That list is the
    authoritative per-frame count -- the counters carry no per-frame breakdown.

    READ-ONLY.  The keys are copied out before anything is looked up, and
    nothing here writes to the cache or to any package module.
    """
    out = dict(available=False, frames=[], error=None,
               compile_cache_entries=None)
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
        return out
    try:
        cache = getattr(projectors_module, "_COMPILE_CACHE", {})
        keys = list(cache.keys())
        out["compile_cache_entries"] = len(keys)
        by_code = {}
        for key in keys:
            fn = key[0] if isinstance(key, tuple) else key
            instance = key[1] if isinstance(key, tuple) else None
            code = getattr(fn, "__code__", None)
            if code is None:
                continue
            record = by_code.setdefault(id(code), dict(
                function=getattr(fn, "__name__", "?"),
                module=getattr(fn, "__module__", "?"),
                file=getattr(code, "co_filename", "?"),
                line=int(getattr(code, "co_firstlineno", 0)),
                instances=[], code=code))
            record["instances"].append(instance)
        frames = []
        for record in by_code.values():
            code = record.pop("code")
            try:
                entries = _debug_get_cache_entry_list(code)
                record["variants"] = len(entries)
                record["compile_ids"] = [str(getattr(entry, "compile_id", "?"))
                                         for entry in entries]
            except Exception as exc:                              # noqa: BLE001
                record["variants"] = None
                record["compile_ids"] = []
                record["variants_error"] = "{}: {}".format(
                    type(exc).__name__, exc)
            record["instances"] = sorted(
                str(item) for item in record["instances"])
            record["instance_count"] = len(record["instances"])
            frames.append(record)
        frames.sort(key=lambda item: (item["module"], item["function"]))
        out.update(available=True, frames=frames,
                   variants_total=sum(item["variants"] or 0
                                      for item in frames))
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def frame_variant_delta(before, after):
    """The per-frame variants one reconstruction ADDED, by frame name."""
    start = {(item["module"], item["function"]): (item["variants"] or 0)
             for item in (before or {}).get("frames") or []}
    rows = []
    for item in (after or {}).get("frames") or []:
        key = (item["module"], item["function"])
        rows.append(dict(module=item["module"], function=item["function"],
                         file=item["file"], line=item["line"],
                         instances=item.get("instance_count"),
                         variants_before=start.get(key, 0),
                         variants_after=item.get("variants"),
                         variants_added=((item.get("variants") or 0)
                                         - start.get(key, 0)),
                         compile_ids=item.get("compile_ids")))
    rows.sort(key=lambda row: (-(row["variants_after"] or 0), row["module"],
                               row["function"]))
    return rows


# ── the recompile log ─────────────────────────────────────────────────────────
#: The marker torch's recompiles artifact puts on every line it writes.
RECOMPILE_MARKER = "[__recompiles] "


def recompile_log_path():
    """Where this arm's TORCH_LOGS output is written, as its environment says.
    Empty when the arm was not asked to keep one."""
    return os.environ.get("TORCH_LOGS_OUT", "").strip()


def log_size(path):
    try:
        return int(os.path.getsize(path)) if path and os.path.exists(path) \
            else 0
    except OSError:
        return 0


def read_recompiles(path, start=0, end=None):
    """The recompilations torch reported between two byte offsets of its log.

    The log's shape is one "Recompiling function NAME in FILE:LINE" line
    followed by the guard failures that forced it, each on its own line.  This
    returns one record per recompilation with its reasons, and a tally by
    (function, reason) so the table can say how often each guard failed.

    The continuation lines of a user stack trace are skipped: they repeat the
    call site of the reason above them and would otherwise be counted as
    reasons of their own.
    """
    out = dict(path=path, available=False, events=[], by_reason=[],
               lines_read=0, error=None)
    if not path or not os.path.exists(path):
        out["error"] = "no recompile log at {!r}".format(path)
        return out
    try:
        with open(path, "r", errors="replace") as handle:
            handle.seek(int(start))
            text = handle.read() if end is None else handle.read(
                max(0, int(end) - int(start)))
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
        return out

    events = []
    current = None
    lines = text.splitlines()
    out["lines_read"] = len(lines)
    for raw in lines:
        if RECOMPILE_MARKER not in raw:
            continue
        body = raw.split(RECOMPILE_MARKER, 1)[1].rstrip()
        stripped = body.strip()
        if stripped.startswith("Recompiling function "):
            rest = stripped[len("Recompiling function "):]
            name, _, where = rest.partition(" in ")
            current = dict(function=name.strip(), at=where.strip(), reasons=[])
            events.append(current)
            continue
        if current is None:
            continue
        if stripped.startswith("- ") and not stripped.startswith("-   ") \
                and stripped != "- User stack trace:":
            reason = stripped[2:].strip()
            # "<compile id>: <reason>" -- the id names the variant whose guard
            # failed, and the reason is what failed on it.
            head, sep, tail = reason.partition(": ")
            if sep and "/" in head and len(head) <= 12:
                current["reasons"].append(dict(variant=head.strip(),
                                               reason=tail.strip()[:600]))
            else:
                current["reasons"].append(dict(variant=None,
                                               reason=reason[:600]))
    tally = {}
    for event in events:
        for item in event["reasons"]:
            # The reason text carries the offending sizes, which differ between
            # subsets; the head of it is the guard that failed.
            key = (event["function"], item["reason"][:160])
            tally[key] = tally.get(key, 0) + 1
    out.update(available=True, events=events,
               by_reason=[dict(function=function, reason=reason, count=count)
                          for (function, reason), count
                          in sorted(tally.items(), key=lambda kv: -kv[1])])
    return out


# ── the shared measurement ────────────────────────────────────────────────────
def dynamo_before():
    """The dynamo state to measure a reconstruction against.

    The counters are CLEARED, so the row that follows describes this
    reconstruction alone.  The cumulative per-phase nanosecond totals are
    snapshotted instead of cleared: a delta needs no reset and cannot be
    disturbed by anything else in the process.
    """
    out = dict(counters_reset=False, cumulative_ns_before=None, error=None)
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
    """What dynamo says it did during the reconstruction just finished."""
    out = dict(before)
    out["compile_times_csv"] = None
    out["counters"] = None
    out["counters_are_cumulative"] = not before.get("counters_reset")
    out["cumulative_ns_delta"] = None
    out["compile_phase_seconds"] = None
    out["compile_outer_phase"] = None
    out["compile_seconds_outer"] = None
    out["compile_seconds_sum_of_phases"] = None
    out["unique_graphs"] = None
    out["frames_total"] = None
    try:
        from torch._dynamo import utils as dynamo_utils

        out["compile_times_csv"] = _jsonable(
            dynamo_utils.compile_times(repr="csv", aggregate=True))
        counters = getattr(dynamo_utils, "counters", None)
        if counters is not None:
            out["counters"] = {str(key): _jsonable(dict(value))
                               if hasattr(value, "items") else _jsonable(value)
                               for key, value in counters.items()}
            stats = (out["counters"].get("stats") or {})
            frames = (out["counters"].get("frames") or {})
            # A reconstruction that compiled nothing leaves those keys absent
            # rather than zero, and an absent key here MEANS zero -- but only
            # when the counters really were cleared first.  Read as None it
            # would print as an unknown, which is a different and wrong claim.
            default = 0 if before.get("counters_reset") else None
            out["unique_graphs"] = stats.get("unique_graphs", default)
            out["frames_total"] = frames.get("total", default)
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
            # backend_compile, which contains inductor_compile, and so on, so
            # their sum double counts and can exceed the wall of the
            # reconstruction it was measured over.  The outermost phase is the
            # honest single number, and it is simply the largest, since an
            # inner phase cannot outlast the one that contains it.  The sum is
            # kept beside it, named for what it is.
            if seconds:
                outer = max(seconds.items(), key=lambda item: item[1])
                out["compile_outer_phase"] = outer[0]
                out["compile_seconds_outer"] = outer[1]
                out["compile_seconds_sum_of_phases"] = sum(seconds.values())
    except Exception as exc:                                      # noqa: BLE001
        out["error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def peak_readings(torch_module, devices, cuda):
    """The peak allocated bytes on every device in the list."""
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


def one_reconstruction(model, sinogram, weights, devices, cuda, index, mode,
                       modules):
    """One reconstruction, with the candidate installed around it or not.

    The candidate's wrappers go on immediately before the call and come off in a
    finally: a wrapper left behind would keep marking inside a later pass that
    was supposed to run without it.

    The seed is reset immediately before the call, so the second reconstruction
    of an arm is the same reconstruction as the first rather than a different
    draw of pixel partitions, and so two arms in two processes reconstruct the
    same thing.
    """
    import numpy as np

    torch_module = modules["torch"]
    projectors = modules["projectors"]
    triton_parallel = modules["triton_parallel"]

    log_path = recompile_log_path()
    row = dict(index=index, mode=mode, iterations=VCD_ITERATIONS,
               seed=VCD_SEED, recompile_log=log_path)
    row["frames_before"] = compiled_frames(projectors)
    row["log_bytes_before"] = log_size(log_path)

    reset_peaks(torch_module, devices, cuda)
    dynamo_state = dynamo_before()

    candidate = None
    if mode == "candidate":
        candidate = Candidate(model, torch_module, triton_parallel).install()

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
        if candidate is not None:
            row["candidate_restored"] = candidate.restore()

    row["wall_s"] = wall
    row["peak_bytes_per_device"] = peak_readings(torch_module, devices, cuda)
    if row["peak_bytes_per_device"]:
        row["busiest_peak_bytes"] = max(row["peak_bytes_per_device"])
    row["dynamo"] = dynamo_after(dynamo_state)
    row["frames_after"] = compiled_frames(projectors)
    row["frame_variants"] = frame_variant_delta(row["frames_before"],
                                                row["frames_after"])
    row["log_bytes_after"] = log_size(log_path)
    row["recompiles"] = read_recompiles(log_path, row["log_bytes_before"],
                                        row["log_bytes_after"])
    if candidate is not None:
        row["candidate"] = candidate.report()
    row.update(convergence_record(info))
    # The fingerprint is taken AFTER the wall is stopped, so the host arithmetic
    # that computes it is never inside the number this run compares.
    started = time.perf_counter()
    row["fingerprint"] = fingerprint(volume)
    row["fingerprint_s"] = time.perf_counter() - started
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
    bodies and traces the very frames the arms exist to count.  So the staging
    writes into cache directories no arm ever reads, and it runs before any arm.

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
               bytes_on_disk=os.path.getsize(path))
    return row


def run_arm(cfg):
    """One arm: a fresh process that imports the library, touches its devices,
    builds a model, loads the staged sinogram, builds the weights, and then
    reconstructs once or twice with everything recorded."""
    import numpy as np

    torch_import_start = time.perf_counter()
    import torch
    torch_import_s = time.perf_counter() - torch_import_start

    mbirtorch_import_start = time.perf_counter()
    import mbirtorch
    mbirtorch_import_s = time.perf_counter() - mbirtorch_import_start

    from mbirtorch import projectors, triton_parallel

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
               process_start_to_import_end_s=time.time() - PROCESS_START,
               env_inductor_cache_dir=os.environ.get(
                   "TORCHINDUCTOR_CACHE_DIR"),
               env_triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"),
               env_torch_logs=os.environ.get("TORCH_LOGS"),
               env_torch_logs_out=os.environ.get("TORCH_LOGS_OUT"),
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
    row["granularity"] = [int(v) for v in model.get_params("granularity")]
    row["partition_sequence_head"] = [
        int(v) for v in list(model.get_params("partition_sequence"))
        [:VCD_ITERATIONS]]

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

    modules = dict(torch=torch, projectors=projectors,
                   triton_parallel=triton_parallel)

    # ── the reconstructions ─────────────────────────────────────────────────
    row["recons_planned"] = int(cfg["recons"])
    recons = []
    for index in range(int(cfg["recons"])):
        recons.append(one_reconstruction(
            model, meta["sinogram"], weights, devices, cuda, index,
            cfg["mode"], modules))
        try:
            realized_devices = [str(d) for d in model.recon_placement.devices]
        except Exception:                                         # noqa: BLE001
            realized_devices = []
        row["realized_devices"] = realized_devices
        row["realized_n_devices"] = len(realized_devices)
        row["devices_as_asked"] = (realized_devices
                                   == [str(d) for d in devices])
    row["recons"] = recons
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

    THE TWO CACHE DIRECTORIES ARE SET HERE AND NEVER INHERITED.  A job that took
    TORCHINDUCTOR_CACHE_DIR from the shell and left TRITON_CACHE_DIR alone --
    which is what the sbatch files in this series do -- would run against a
    Triton cache in the user's home directory full of artifacts from earlier
    jobs, and this run's cache state would not be the one its plan describes.

    TORCH_LOGS IS SET ON EVERY ARM, BASELINE AND CANDIDATE ALIKE.  The
    recompiles artifact is where the guard reasons come from, and enabling it on
    one mode only would make the two modes' walls incomparable.  TORCH_LOGS_OUT
    sends a copy to a file this arm reads back and summarizes.

    MBIRTORCH_NUM_DEVICES IS REMOVED rather than set.  Every arm names its
    devices one by one, so a process-wide count pin is not the mechanism here.

    PYTHONPATH IS INHERITED.  A candidate tree reached only through PYTHONPATH
    would otherwise be swapped for the installed one without a word.  Every job
    records the mbirtorch file it actually imported.
    """
    env = dict(os.environ)
    env.pop("MG59_DRY", None)           # a worker never prints a plan
    env.pop("MG59_ARMS", None)          # a worker runs its cfg, not a plan
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
    if cfg.get("kind") == "arm" and cfg.get("recompile_log"):
        env["TORCH_LOGS"] = "recompiles"
        env["TORCH_LOGS_OUT"] = cfg["recompile_log"]
    else:
        env.pop("TORCH_LOGS", None)
        env.pop("TORCH_LOGS_OUT", None)
    return env


def spawn(cfg, timeout_s):
    """Run one job in a NEW interpreter, with a hard time cap.

    A new process per job is the whole instrument, not tidiness: dynamo's traced
    frames, its cache entries, the compiled callables and the CUDA context all
    live for the life of a process, and this run is measuring exactly what a
    process has to trace before it can reconstruct.  The row travels through a
    file rather than through stdout, so the worker's own output streams into the
    job log while it runs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR,
                            "_mg59_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR,
                            "_mg59_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    # A stale log from an earlier run of the same arm would be read back as this
    # arm's recompilations, so it is removed before the arm starts.
    if cfg.get("recompile_log") and os.path.exists(cfg["recompile_log"]):
        os.remove(cfg["recompile_log"])
    env = job_env(cfg)
    env["MG59_CHILD"] = cfg_path
    env["MG59_CHILD_OUT"] = out_path
    start = time.perf_counter()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                              env=env, timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # The cap is a READING, not a fault.
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
    harness fault."""
    if row.get("timed_out"):
        return True
    text = str(row.get("error", "")).lower()
    if not text:
        return False
    if any(marker in text for marker in CAPACITY_MARKERS):
        return True
    return "wrote no row" in text and str(row.get("worker_returncode")) == "-9"


# ── the plan ──────────────────────────────────────────────────────────────────
def build_plan():
    """Every job, in run order: the identity probe, the staging, then the arms.

    The arm order is load-bearing.  A reuse arm reads the cache directories its
    fill arm filled, so the fill arm of a (mode, device count) must run before
    the measured arm of that pair; the declared order does that and MG59_ARMS is
    normalized back into it.
    """
    allowed = [spec["name"] for spec in arms_declared()]
    keep = _strict_subset("MG59_ARMS", allowed)

    probe = dict(kind="identity", job_id="identity",
                 cache_dirs=cache_dirs("probe"))
    stage = dict(kind="stage", job_id="stage", cell=list(cell()),
                 cache_dirs=cache_dirs("stage"))
    arms = []
    for spec in arms_declared():
        if spec["name"] not in keep:
            continue
        arms.append(dict(kind="arm", arm=spec["name"], job_id=spec["name"],
                         count=int(spec["count"]), mode=spec["mode"],
                         devices=device_list(int(spec["count"])),
                         cache_owner=spec["cache_owner"],
                         caches=spec["caches"], recons=int(spec["recons"]),
                         measured=bool(spec["measured"]),
                         filled_by=spec["filled_by"], what=spec["what"],
                         cell=list(cell()),
                         cache_dirs=cache_dirs(spec["cache_owner"]),
                         recompile_log=os.path.join(
                             RESULTS_DIR,
                             "mg59_recompiles_{}.log".format(spec["name"]))))
    if not arms:
        raise ValueError("MG59_ARMS selects no arm")
    return probe, stage, arms


def sinogram_gib(shape):
    return (int(shape[0]) * int(shape[1]) * int(shape[2]) * 4) / 2 ** 30


def print_plan(probe, stage, arms):
    print("mg59 which frames dynamo traces on every reconstruction, and whether "
          "marking the pixel-count dimension dynamic removes most of that "
          "tracing: {} arm(s), device {}, {} VCD iteration(s)".format(
              len(arms), DEVICE, VCD_ITERATIONS))
    print("  part 1 names every frame the library compiled, how many variants "
          "each produced and the guard failure recorded for each "
          "recompilation.  part 2 marks the pixel-count dimension dynamic from "
          "OUTSIDE the package and measures what that does to the graph count, "
          "the dynamo seconds, the wall and the reconstructed values.")
    print("  cell: parallel {}, {:.1f} GiB of sinogram".format(
        tuple(cell()), sinogram_gib(cell())))
    print("  rows -> {}".format(RESULTS_DIR))
    print("  interpreter: {}".format(sys.executable))
    print("  PYTHONPATH:  {}".format(os.environ.get("PYTHONPATH") or "(none)"))
    print("  the parent process imports no torch: every device touch happens "
          "inside an arm")

    print("\n  {:<{w}}{:>{d}}{:>11}{:>9}{:>8}  {}".format(
        "job", "devices", "mode", "cap min", "recons", "what it does",
        w=ARM_COL, d=DEV_COL))
    print("  {:<{w}}{:>{d}}{:>11}{:>9}{:>8}  {}".format(
        probe["job_id"], "-", "-", int(PROBE_TIMEOUT_S / 60), "-",
        "names torch, mbirtorch, the devices and the tree witnesses",
        w=ARM_COL, d=DEV_COL))
    print("  {:<{w}}{:>{d}}{:>11}{:>9}{:>8}  {}".format(
        stage["job_id"], device_label(device_list(1)), "-",
        int(STAGE_TIMEOUT_S / 60), "-",
        "reuses a verified staged sinogram or builds one, in its OWN cache "
        "directories", w=ARM_COL, d=DEV_COL))
    for cfg in arms:
        print("  {:<{w}}{:>{d}}{:>11}{:>9}{:>8}  {}".format(
            cfg["job_id"], device_label(cfg["devices"]), cfg["mode"],
            int(ARM_TIMEOUT_S / 60), cfg["recons"], cfg["what"], w=ARM_COL,
            d=DEV_COL))

    print("\n  the staged input is searched for, in this order:")
    for directory in stage_search_dirs():
        print("    {}".format(directory))
    print("    accepting any of: {}".format(", ".join(stage_names_accepted())))
    print("    whatever is found is md5-verified and its shape and geometry "
          "checked before any arm reads it; a measured arm never forward "
          "projects its own input, because that would trace the frames it "
          "exists to count")

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
    print("    each MODE fills its OWN pair: the candidate's graphs are "
          "different graphs, so a candidate arm pointed at a cache the "
          "baseline filled would miss on every one of them and pay code "
          "generation the baseline did not, and the wall comparison would be "
          "about cache contents rather than about tracing")

    print("\n  where the candidate's wrappers go, all from outside the package "
          "and all restored in a finally:")
    print("    projector_functions._fwd_body_per_dev / _back_body_per_dev   "
          "the per-device projection bodies (hand-written Triton kernels on a "
          "build with triton, which do not compile, so marking there is inert "
          "-- the row records which entries were the wrappers)")
    print("    model._get_update_direction                                  "
          "an instance attribute over the model method")
    print("    the subset updater's per-device lists {}"
          .format(", ".join(UPDATER_BODY_LISTS)))
    print("      reached through the closure of the updater "
          "model.create_vcd_subset_updater returns, which is where the driver "
          "holds them; those lists are objects that one call just created, and "
          "nothing at module level in the package is written")
    print("    the rule: mark dimension 0 dynamic on every tensor argument "
          "whose leading dimension equals the pixel count of the subset the "
          "driver is updating, and on nothing else.  the count is read from "
          "the pixel index vector the driver hands the updater.")
    print("    the wrapper copies _view_batch_cost when the body carries one, "
          "because the driver reads it to choose a view batch; it copies "
          "nothing else, and in particular not _mbirtorch_no_compile")

    print("\n  part 1 reads its numbers from the tools' own accounting: "
          "dynamo's counters (cleared before each reconstruction), the length "
          "of dynamo's cache-entry list for every function the library "
          "compiled (one entry is one compiled variant), the TORCH_LOGS="
          "recompiles output written to a file and read back, and "
          "compile_times(repr='csv') with the cumulative per-phase totals as a "
          "before/after delta.")

    print("\n  each reconstruction is fingerprinted in float64 -- the sum of "
          "absolute values and the sum of squares over the volume, in slabs -- "
          "and the comparison prints the candidate's relative difference from "
          "the baseline at the same device count.  a relative difference above "
          "{:g} is reported as a candidate that changed the values.".format(
              VALUE_TOLERANCE))

    if DEVICE != "cuda":
        print("\n  ON {} THE RUN DEGRADES.  There is one device, so the "
              "four-device arms cannot run and only the four one-device arms "
              "are planned.  There is no triton, so TRITON_CACHE_DIR governs "
              "nothing and the two projection bodies ARE compiled torch bodies "
              "rather than hand-written kernels -- which means the smoke "
              "exercises a seam the real run finds inert.  The rows say all of "
              "this.  The smoke is plumbing, not a measurement.".format(
                  DEVICE.upper()))

    print("\n  exit code = INSTRUMENT HEALTH ONLY: every planned arm produced "
          "a row or a recorded out-of-memory or timeout, every arm loaded the "
          "staged input with its md5 verified, every arm realized the device "
          "count it asked for, every candidate arm installed its wrappers and "
          "restored them, every arm produced a readable recompile log, and the "
          "tree witnesses hold.  What any arm MEASURED never changes it -- "
          "including a candidate that helps nothing, or one that changes the "
          "reconstructed values.")
    print("  no library file is edited: every wrapper goes on an object the "
          "model instance holds and comes off in a finally")


# ── rows ──────────────────────────────────────────────────────────────────────
def write_row(sink, row):
    """One jsonl row, flushed, so a job that is killed mid-run leaves every row
    it had already finished."""
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


def _seconds(value, prec=2):
    return "-" if value is None else "{:.{p}f}".format(value, p=prec)


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


def print_frame_table(arm, row, recon):
    """PART 1: every frame the library compiled in this arm, with the number of
    variants each produced."""
    print("\n### {} reconstruction {}: the frames dynamo compiled".format(
        arm, recon.get("index")))
    rows = recon.get("frame_variants") or []
    if not rows:
        print("No frame table: {}".format(
            (recon.get("frames_after") or {}).get("error")
            or "the compiled-function cache named nothing"))
        return
    print("| frame | module | file:line | compiled instances | variants at end "
          "| variants this reconstruction added | compile ids |")
    print("|---|---|---|---|---|---|---|")
    for item in rows:
        print("| {} | {} | {}:{} | {} | {} | {} | {} |".format(
            item["function"], item["module"],
            os.path.basename(str(item.get("file"))), item.get("line"),
            item.get("instances"),
            item.get("variants_after") if item.get("variants_after")
            is not None else "-",
            item.get("variants_added"),
            ", ".join(item.get("compile_ids") or []) or "-"))
    dynamo = recon.get("dynamo") or {}
    total = sum(item.get("variants_added") or 0 for item in rows)
    print("A VARIANT is one entry in dynamo's cache for that frame's code "
          "object: one compiled graph with its own guards.  Several per-device "
          "compiled instances of one function share one code object, so a "
          "frame's variants count every device's.  This reconstruction added "
          "{} variants across these frames; dynamo's own counters report {} "
          "unique graphs and {} frames for it.".format(
              total, dynamo.get("unique_graphs"), dynamo.get("frames_total")))


def print_recompile_table(arm, recon):
    """PART 1: why each variant after the first was created."""
    record = recon.get("recompiles") or {}
    print("\n### {} reconstruction {}: why each recompilation happened"
          .format(arm, recon.get("index")))
    if not record.get("available"):
        print("No recompile log: {}".format(record.get("error")))
        return
    rows = record.get("by_reason") or []
    if not rows:
        print("The log records no recompilation in this reconstruction, which "
              "is what a process that traced every frame exactly once looks "
              "like.")
        return
    print("| frame | guard failure that forced the recompilation | times |")
    print("|---|---|---|")
    for item in rows:
        print("| {} | {} | {} |".format(item["function"],
                                        item["reason"].replace("|", "/"),
                                        item["count"]))
    print("From TORCH_LOGS=recompiles, which torch writes when a compiled "
          "frame's guards fail and it must trace again.  The FIRST compile of "
          "a frame is not a recompilation and so is not in this table; the "
          "frame table above counts it.  {} recompilation event(s) in "
          "{} log line(s).".format(len(record.get("events") or []),
                                   record.get("lines_read")))


def print_candidate_table(arm, recon):
    """PART 2: what the candidate's wrappers did in this reconstruction."""
    record = recon.get("candidate")
    if not record:
        return
    print("\n### {} reconstruction {}: what the candidate marked".format(
        arm, recon.get("index")))
    print("| wrapped body | calls | calls with a subset in flight | tensors "
          "marked | mark errors |")
    print("|---|---|---|---|---|")
    for label in sorted(record.get("marks") or {}):
        entry = record["marks"][label]
        print("| {} | {} | {} | {} | {} |".format(
            label, entry.get("calls"), entry.get("calls_with_a_subset"),
            entry.get("marked"), entry.get("errors")))
    print("\n| what | value |")
    print("|---|---|")
    print("| subsets the driver updated | {} |".format(
        record.get("subsets_seen")))
    print("| distinct subset sizes seen | {} |".format(
        record.get("subset_sizes_seen")))
    print("| seconds spent marking | {} |".format(
        _seconds(record.get("mark_seconds"), 3)))
    print("| wrappers restored | {} |".format(recon.get("candidate_restored")))
    print("| wrapper carries _view_batch_cost | {} |".format(
        record.get("wrapper_carries_view_batch_cost")))
    print("| wrapper copies _mbirtorch_no_compile | {} |".format(
        record.get("wrapper_copies_no_compile_marker")))
    for attr, flags in (record.get("bound_bodies_are_triton_wrappers")
                        or {}).items():
        print("| {} entries that were the Triton wrappers | {} of {} |".format(
            attr, sum(1 for flag in flags if flag), len(flags)))
    for name, bodies in sorted((record.get("wrapped_bodies") or {}).items()):
        print("| bodies wrapped in {} | {} |".format(name, ", ".join(bodies)))
    for note in record.get("notes") or []:
        print("| note | {} |".format(note))


def print_arm_costs(arm, row):
    """The once-per-process costs this arm paid, outside any reconstruction."""
    print("\n### {} once-per-process costs".format(arm))
    print("| step | seconds |")
    print("|---|---|")
    for label, key in (("import torch", "torch_import_s"),
                       ("import mbirtorch", "mbirtorch_import_s"),
                       ("CUDA context, all devices", "cuda_context_s"),
                       ("staged load and md5", "staged_load_s"),
                       ("build the model", "model_build_s"),
                       ("configure the devices", "configure_devices_s"),
                       ("build the weights on the host", "weights_build_s")):
        print("| {} | {} |".format(label, _seconds(row.get(key), 3)))
    print("| cache files, inductor + triton, before -> after | {} -> {} |"
          .format(sum((row.get("cache_before") or {}).get(k, {}).get("files", 0)
                      for k in ("inductor", "triton")),
                  sum((row.get("cache_after") or {}).get(k, {}).get("files", 0)
                      for k in ("inductor", "triton"))))


def print_compile_summary(arm_rows, arms):
    """One line per reconstruction: the tracing cost and what it produced."""
    print("\n### the tracing cost of every reconstruction in this run")
    print("| arm | mode | devices | recon | wall s | dynamo outer phase | "
          "dynamo outer s | unique graphs | frames | variants added | "
          "recompile events | abs sum | sq sum |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg in arms:
        row = arm_rows.get(cfg["arm"]) or {}
        for recon in row.get("recons") or []:
            dynamo = recon.get("dynamo") or {}
            marks = recon.get("fingerprint") or {}
            variants = sum(item.get("variants_added") or 0
                           for item in (recon.get("frame_variants") or []))
            events = len(((recon.get("recompiles") or {}).get("events")) or [])
            print("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | "
                  "{} | {} |".format(
                      cfg["arm"], cfg["mode"],
                      device_label(row.get("realized_devices")
                                   or row.get("requested_devices") or []),
                      recon.get("index"), _seconds(recon.get("wall_s")),
                      dynamo.get("compile_outer_phase") or "-",
                      _seconds(dynamo.get("compile_seconds_outer")),
                      dynamo.get("unique_graphs")
                      if dynamo.get("unique_graphs") is not None else "-",
                      dynamo.get("frames_total")
                      if dynamo.get("frames_total") is not None else "-",
                      variants, events,
                      "-" if marks.get("abs_sum") is None
                      else "{:.9e}".format(marks["abs_sum"]),
                      "-" if marks.get("sq_sum") is None
                      else "{:.9e}".format(marks["sq_sum"])))
    print("Dynamo's phases NEST, so the outermost one is shown rather than a "
          "sum; every phase's seconds are on the row in the jsonl, along with "
          "compile_times(repr='csv') verbatim and the counters.  The fill arms "
          "are here for completeness; the comparison below reads the measured "
          "arms only.")


def compare_modes(count, arm_rows, arms_by_name):
    """PART 2 at one device count: the baseline against the candidate."""
    measured = [(name, row) for name, row in arm_rows.items()
                if (arms_by_name.get(name) or {}).get("measured")
                and int((arms_by_name.get(name) or {}).get("count") or 0)
                == int(count)]
    base = next((row for name, row in measured
                 if (arms_by_name.get(name) or {}).get("mode") == "baseline"),
                {})
    cand = next((row for name, row in measured
                 if (arms_by_name.get(name) or {}).get("mode") == "candidate"),
                {})

    def recon(row, index):
        recons = row.get("recons") or []
        return recons[index] if len(recons) > index else {}

    def read(row, index, path, default=None):
        item = recon(row, index)
        for key in path:
            item = (item or {}).get(key) if isinstance(item, dict) else None
        return default if item is None else item

    def variants(row, index):
        return sum(item.get("variants_added") or 0
                   for item in (recon(row, index).get("frame_variants") or []))

    out = dict(row="comparison", count=int(count),
               baseline_arm=base.get("arm"), candidate_arm=cand.get("arm"),
               tolerance=VALUE_TOLERANCE)
    for label, row in (("baseline", base), ("candidate", cand)):
        out[label] = dict(
            arm=row.get("arm"),
            wall_first_s=read(row, 0, ["wall_s"]),
            wall_second_s=read(row, 1, ["wall_s"]),
            dynamo_outer_s=read(row, 0, ["dynamo", "compile_seconds_outer"]),
            dynamo_outer_phase=read(row, 0,
                                    ["dynamo", "compile_outer_phase"]),
            unique_graphs=read(row, 0, ["dynamo", "unique_graphs"]),
            frames_total=read(row, 0, ["dynamo", "frames_total"]),
            variants_added=variants(row, 0) if row.get("recons") else None,
            recompile_events=len(read(row, 0, ["recompiles", "events"]) or []),
            abs_sum=read(row, 0, ["fingerprint", "abs_sum"]),
            sq_sum=read(row, 0, ["fingerprint", "sq_sum"]),
            abs_sum_second=read(row, 1, ["fingerprint", "abs_sum"]),
            sq_sum_second=read(row, 1, ["fingerprint", "sq_sum"]),
            finite=read(row, 0, ["fingerprint", "finite"]),
            mark_seconds=read(row, 0, ["candidate", "mark_seconds"]),
            marks_total=read(row, 0, ["candidate", "marks_total"]),
            mark_errors=read(row, 0, ["candidate", "mark_errors_total"]))

    base_side, cand_side = out["baseline"], out["candidate"]
    out["graphs_removed"] = (
        None if base_side["unique_graphs"] is None
        or cand_side["unique_graphs"] is None
        else base_side["unique_graphs"] - cand_side["unique_graphs"])
    out["dynamo_seconds_removed"] = (
        None if base_side["dynamo_outer_s"] is None
        or cand_side["dynamo_outer_s"] is None
        else base_side["dynamo_outer_s"] - cand_side["dynamo_outer_s"])
    out["wall_first_removed_s"] = (
        None if base_side["wall_first_s"] is None
        or cand_side["wall_first_s"] is None
        else base_side["wall_first_s"] - cand_side["wall_first_s"])
    out["wall_second_removed_s"] = (
        None if base_side["wall_second_s"] is None
        or cand_side["wall_second_s"] is None
        else base_side["wall_second_s"] - cand_side["wall_second_s"])
    out["abs_sum_relative_difference"] = relative_difference(
        cand_side["abs_sum"], base_side["abs_sum"])
    out["sq_sum_relative_difference"] = relative_difference(
        cand_side["sq_sum"], base_side["sq_sum"])
    differences = [value for value in (out["abs_sum_relative_difference"],
                                       out["sq_sum_relative_difference"])
                   if value is not None]
    out["values_unchanged"] = (
        None if not differences
        else bool(max(differences) <= VALUE_TOLERANCE))
    return out


def print_comparison(item):
    count = item["count"]
    print("\n### {} device(s): the baseline against the candidate".format(
        count))
    base, cand = item.get("baseline") or {}, item.get("candidate") or {}
    print("| what | baseline ({}) | candidate ({}) | baseline minus candidate, "
          "so a positive number is what the candidate removed |".format(
              base.get("arm") or "no arm ran",
              cand.get("arm") or "no arm ran"))
    print("|---|---|---|---|")
    rows = (
        ("unique graphs, first reconstruction", "unique_graphs", "d",
         item.get("graphs_removed")),
        ("frames dynamo compiled, first reconstruction", "frames_total", "d",
         None),
        ("variants added, first reconstruction", "variants_added", "d", None),
        ("recompile events, first reconstruction", "recompile_events", "d",
         None),
        ("dynamo outer phase seconds", "dynamo_outer_s", "f",
         item.get("dynamo_seconds_removed")),
        ("wall, first reconstruction", "wall_first_s", "f",
         item.get("wall_first_removed_s")),
        ("wall, second reconstruction in the same process", "wall_second_s",
         "f", item.get("wall_second_removed_s")),
    )
    for label, key, kind, difference in rows:
        def show(side):
            value = side.get(key)
            if value is None:
                return "-"
            return str(value) if kind == "d" else "{:.2f}".format(value)
        print("| {} | {} | {} | {} |".format(
            label, show(base), show(cand),
            "-" if difference is None else
            (str(difference) if kind == "d" else "{:+.2f}".format(difference))))
    print("| the candidate's outermost dynamo phase | {} | {} | - |".format(
        base.get("dynamo_outer_phase") or "-",
        cand.get("dynamo_outer_phase") or "-"))
    print("| seconds the candidate spent marking tensors | - | {} | - |".format(
        _seconds(cand.get("mark_seconds"), 3)))
    print("| tensors marked / marks that raised | - | {} / {} | - |".format(
        cand.get("marks_total"), cand.get("mark_errors")))

    print("\n| the reconstructed values | baseline | candidate | relative "
          "difference |")
    print("|---|---|---|---|")
    for label, key, diff_key in (
            ("sum of absolute values", "abs_sum",
             "abs_sum_relative_difference"),
            ("sum of squares", "sq_sum", "sq_sum_relative_difference")):
        difference = item.get(diff_key)
        print("| {} | {} | {} | {} |".format(
            label,
            "-" if base.get(key) is None else "{:.12e}".format(base[key]),
            "-" if cand.get(key) is None else "{:.12e}".format(cand[key]),
            "-" if difference is None else "{:.3e}".format(difference)))
    verdict = item.get("values_unchanged")
    print("Declared tolerance {:g} on the relative difference.  {}".format(
        VALUE_TOLERANCE,
        "No pair to compare." if verdict is None else
        ("The candidate reconstructed the same values." if verdict else
         "THE CANDIDATE CHANGED THE RECONSTRUCTED VALUES, so it is a failed "
         "candidate.")))


def print_value_yardstick(comparisons):
    """The baseline's own difference between device counts, printed beside the
    candidate's, so the tolerance has a measurement from this run to sit next
    to rather than only a declaration."""
    baselines = [(item["count"], (item.get("baseline") or {}).get("abs_sum"),
                  (item.get("baseline") or {}).get("sq_sum"))
                 for item in comparisons]
    baselines = [entry for entry in baselines if entry[1] is not None]
    if len(baselines) < 2:
        return
    print("\n### a yardstick for the value check, measured in this run")
    first, second = baselines[0], baselines[-1]
    print("The same baseline reconstruction at {} device(s) and at {} "
          "device(s) differs by {:.3e} on the sum of absolute values and "
          "{:.3e} on the sum of squares.  That difference is arithmetic order, "
          "not a different answer, and it is the scale the candidate's "
          "difference should be read against.".format(
              first[0], second[0],
              relative_difference(second[1], first[1]),
              relative_difference(second[2], first[2])))


def summarize(identity, stage_row, arm_rows, comparisons, arms, findings,
              out_path):
    """The tables a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A
    candidate that removes no graphs, a candidate that changes the values, a
    tracing cost that turns out to be larger than mg57 recorded -- all FINDINGS:
    printed, and none of them touches the exit code.  A missing row, an md5 that
    did not verify, an arm that realized a different device count than it asked
    for, a candidate wrapper that was not restored, a missing recompile log and
    a failed tree witness are instrument failures, because they mean the run did
    not measure what the plan said it would.
    """
    print("\n===== mg59 which frames dynamo traces, and whether marking the "
          "pixel-count dimension dynamic removes them ({}) =====".format(
              out_path))
    broken = []
    findings = list(findings)
    arms_by_name = {cfg["arm"]: cfg for cfg in arms}

    header = ("{:<{w}}{:>11}{:>{d}}{:>11}{:>11}{:>12}{:>10}"
              .format("arm", "mode", "devices", "recon 0 s", "recon 1 s",
                      "busiest GB", "state", w=ARM_COL, d=DEV_COL))
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
        print("{:<{w}}{:>11}{:>{d}}{}{}{}{:>10}".format(
            name, cfg["mode"],
            device_label(row.get("realized_devices")
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
        if row.get("cache_dirs_set_in_env") is False:
            broken.append("{}|the two cache directories were not the ones this "
                          "arm's environment named".format(name))
        if cfg.get("caches") == "reuse":
            before = row.get("cache_before") or {}
            files = sum(entry.get("files", 0) for entry in before.values())
            if files == 0:
                broken.append(
                    "{}|its cache directories were empty at the start, so it "
                    "is not the full-cache reading it was planned to be; {} "
                    "either did not run or filled nothing".format(
                        name, cfg.get("filled_by")))
        for recon in recons:
            if cfg["mode"] == "candidate":
                if recon.get("candidate_restored") is not True:
                    broken.append("{}|reconstruction {}: the candidate's "
                                  "wrappers were not restored".format(
                                      name, recon.get("index")))
                candidate = recon.get("candidate") or {}
                if candidate.get("error"):
                    broken.append("{}|reconstruction {}: the candidate failed "
                                  "to install: {}".format(
                                      name, recon.get("index"),
                                      candidate["error"]))
                if not candidate.get("wrapped_bodies"):
                    broken.append("{}|reconstruction {}: the candidate wrapped "
                                  "nothing".format(name, recon.get("index")))
                missing = [n for n in UPDATER_BODY_LISTS
                           if n not in (candidate.get("wrapped_bodies") or {})]
                if missing:
                    broken.append("{}|reconstruction {}: the candidate did not "
                                  "reach {}".format(name, recon.get("index"),
                                                    ", ".join(missing)))
                if candidate.get("mark_errors_total"):
                    findings.append(
                        "{}: reconstruction {}: {} mark call(s) raised".format(
                            name, recon.get("index"),
                            candidate["mark_errors_total"]))
                for note in candidate.get("notes") or []:
                    findings.append("{}: the candidate noted: {}".format(
                        name, note))
            log = recon.get("recompiles") or {}
            if not log.get("available"):
                broken.append("{}|reconstruction {}: no readable recompile "
                              "log ({})".format(name, recon.get("index"),
                                                log.get("error")))
            frames = recon.get("frames_after") or {}
            if not frames.get("available"):
                broken.append("{}|reconstruction {}: the per-frame variant "
                              "counts could not be read ({})".format(
                                  name, recon.get("index"),
                                  frames.get("error")))
            dynamo = recon.get("dynamo") or {}
            if dynamo.get("error"):
                broken.append("{}|reconstruction {}: dynamo's own accounting "
                              "could not be read: {}".format(
                                  name, recon.get("index"), dynamo["error"]))
            if dynamo.get("counters_are_cumulative"):
                findings.append("{}: reconstruction {}: the counters could not "
                                "be cleared, so they are cumulative".format(
                                    name, recon.get("index")))
            added = sum(item.get("variants_added") or 0
                        for item in (recon.get("frame_variants") or []))
            graphs = dynamo.get("unique_graphs")
            if graphs is not None and added != graphs:
                findings.append(
                    "{}: reconstruction {}: the per-frame variant counts add "
                    "to {} but dynamo counted {} unique graphs, so {} graph(s) "
                    "belong to frames the compiled-function cache does not "
                    "name".format(name, recon.get("index"), added, graphs,
                                  graphs - added))
            if recon.get("fm_rmse_decreased") is False:
                findings.append("{}: reconstruction {}'s forward-model error "
                                "did not fall ({})".format(
                                    name, recon.get("index"),
                                    recon.get("fm_rmse")))
            marks = recon.get("fingerprint") or {}
            if marks.get("finite") is False:
                findings.append("{}: reconstruction {} produced a volume that "
                                "is not everywhere finite".format(
                                    name, recon.get("index")))
            if marks.get("error"):
                broken.append("{}|reconstruction {}: the fingerprint could not "
                              "be taken: {}".format(name, recon.get("index"),
                                                    marks["error"]))

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
    print_compile_summary(arm_rows, arms)
    for cfg in arms:
        arm = arm_rows.get(cfg["arm"]) or {}
        if arm.get("error") or not arm.get("recons"):
            continue
        if not cfg.get("measured"):
            continue
        print_arm_costs(cfg["arm"], arm)
        for recon in arm["recons"]:
            print_frame_table(cfg["arm"], arm, recon)
            print_recompile_table(cfg["arm"], recon)
            print_candidate_table(cfg["arm"], recon)
    for item in comparisons:
        print_comparison(item)
        if item.get("values_unchanged") is False:
            findings.append(
                "{} device(s): the candidate changed the reconstructed values "
                "(relative difference {:.3e} on the sum of absolute values, "
                "{:.3e} on the sum of squares), so it is a failed "
                "candidate".format(
                    item["count"],
                    item.get("abs_sum_relative_difference") or float("nan"),
                    item.get("sq_sum_relative_difference") or float("nan")))
    print_value_yardstick(comparisons)

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
        print("  tree witnesses ok: the padded, recompile-remedied tree with "
              "per-device compiled instances, both parallel kernel wrappers "
              "present with their per-view cost attributes, the three seams "
              "part 2 wraps all present, and the dynamo introspection this run "
              "reads its numbers from all available")
    else:
        print("  TREE WITNESSES: {}".format(witnesses))
        broken.append("tree witnesses|{}".format(witnesses))

    # -- instrument health ----------------------------------------------------
    print("\n-- instrument health --")
    print("  the exit code covers: every planned arm produced a row or a "
          "recorded out-of-memory or timeout, every arm loaded the staged "
          "input with its md5 verified, every arm realized the device count it "
          "asked for, every reuse arm found its caches already filled, every "
          "candidate arm installed its wrappers and restored them, every arm "
          "produced a readable recompile log and a readable per-frame variant "
          "count, and the tree witnesses hold.  What any arm MEASURED never "
          "changes it -- including a candidate that removes no graphs or one "
          "that changes the reconstructed values.")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm produced a result, every arm verified its "
              "staged input and realized the device count it asked for, every "
              "candidate wrapper was installed and restored, every log and "
              "variant count was readable, and the tree witnesses hold")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings outside the tables")

    return dict(row="summary", healthy=not broken, broken=broken,
                findings=findings, comparisons=comparisons,
                tolerance=VALUE_TOLERANCE,
                arms={name: dict(
                    status=arm_status(row),
                    mode=(arms_by_name.get(name) or {}).get("mode"),
                    measured=(arms_by_name.get(name) or {}).get("measured"),
                    walls=[r.get("wall_s") for r in (row.get("recons") or [])],
                    dynamo_outer_s=[(r.get("dynamo") or {}).get(
                        "compile_seconds_outer")
                        for r in (row.get("recons") or [])],
                    unique_graphs=[(r.get("dynamo") or {}).get("unique_graphs")
                                   for r in (row.get("recons") or [])],
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
        RESULTS_DIR, "mg59_dynamo_tracing_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    arm_rows = {}
    comparisons = []
    arms_by_name = {cfg["arm"]: cfg for cfg in arms}
    with open(out_path, "w") as sink:
        write_row(sink, dict(
            row="run_header", script="mg59_dynamo_tracing.py", node=RUN_LABEL,
            stamp=stamp, device=DEVICE, smoke=SMOKE,
            driver_python=sys.executable,
            pythonpath=os.environ.get("PYTHONPATH"),
            results_dir=RESULTS_DIR, cache_root=cache_root(),
            stage_search_dirs=stage_search_dirs(),
            stage_names_accepted=list(stage_names_accepted()),
            identity=identity, tree_witnesses=identity.get("tree_witnesses"),
            staged_md5=stage_row.get("md5"), cell=list(cell()),
            vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
            arm_timeout_s=ARM_TIMEOUT_S, value_tolerance=VALUE_TOLERANCE,
            updater_body_lists=list(UPDATER_BODY_LISTS),
            plan=[dict(kind=cfg["kind"], job_id=cfg["job_id"],
                       devices=cfg.get("devices"), mode=cfg.get("mode"),
                       caches=cfg.get("caches"),
                       cache_dirs=cfg.get("cache_dirs"),
                       measured=cfg.get("measured"),
                       recompile_log=cfg.get("recompile_log"),
                       recons=cfg.get("recons"))
                  for cfg in [probe, stage] + list(arms)]))
        write_row(sink, dict(row="stage", **stage_row))

        for index, cfg in enumerate(arms):
            print("\n  [{}/{}] {} ({}) on {}, caches {}".format(
                index + 1, len(arms), cfg["job_id"], cfg["mode"],
                device_label(cfg["devices"]), cfg["caches"]), flush=True)
            if not stage_row.get("md5_ok") or not stage_row.get("stage_path"):
                # No verified input, so there is nothing honest to measure.
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
                    dynamo = recon.get("dynamo") or {}
                    print("    reconstruction {} wall {:.2f} s  dynamo {} "
                          "{}  unique graphs {}".format(
                              recon.get("index"), recon.get("wall_s", 0.0),
                              dynamo.get("compile_outer_phase") or "-",
                              _seconds(dynamo.get("compile_seconds_outer")),
                              dynamo.get("unique_graphs")), flush=True)

        # ── the comparisons, once every arm of a count is in ─────────────────
        for count in counts_measured():
            item = compare_modes(count, arm_rows, arms_by_name)
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
