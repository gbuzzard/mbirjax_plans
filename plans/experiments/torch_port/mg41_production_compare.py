"""mg41 -- WHAT ONE PRODUCTION RECONSTRUCTION COSTS IN EACH FRAMEWORK, ON ONE
GPU.

WHY THIS RUN EXISTS.  Every cross-framework number the campaign has recorded
comes from a 3-iteration reconstruction.  Three iterations is a good ruler: it
is short enough to run a whole matrix of device counts, and it isolates the
per-iteration cost from the one-time setup.  It is not what a person runs.  A
real reconstruction runs to convergence, which at these problem sizes is on
the order of fifteen iterations, and the question the docs and the list of
remaining work still ask is the plain one: at a production size, on one GPU,
how long does each library take and how much device memory does it hold.

This run answers that question and nothing else.  One problem size, one
device, both geometries, both libraries, fifteen iterations.  It does NOT
replace the 3-iteration tables; those stay as they are, and they remain the
right instrument for comparing device counts.  This run is a separate reading
at a separate iteration count, and it says so on every row it writes.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It edits no library file,
flips no default, and sets no knob under test.  Every variant runs the shipped
configuration of its library.

TERMS USED BELOW, defined once here.
    variant     one measured configuration: one geometry, one library, run in
                its own new process.
    cell        the problem size, named by its sinogram shape.  This run has
                one: sinogram (1024, 1008, 992), which the campaign calls the
                1024 class.
    stage       the untimed step that builds one geometry's sinogram and
                writes it to disk.  Normally it finds the file already there
                and only verifies it.
    cold pass   the first reconstruction in a process.  It pays the kernel
                compiles, so it is timed, recorded, and then discarded.
    warm pass   a reconstruction after the cold one.  Three of them are timed
                and the row carries their median.
    fingerprint two float64 reductions of a reconstruction, used to report how
                far apart the two libraries' answers are.

THE VARIANTS, four of them: geometry in (parallel, cone) x library in
(mbirtorch, mbirjax), all at one device.

THE PROTOCOL.  Everything except the iteration count is the campaign's, so
these rows sit next to the recorded ones without translation.

    model       parallel: angles evenly spaced over half a turn,
                np.linspace(0, pi, num_views, endpoint=False).
                cone: angles evenly spaced over a full turn,
                np.linspace(0, 2*pi, num_views, endpoint=False), with
                source_detector_dist 4.0 x num_channels and source_iso_dist
                2.0 x num_channels, where num_channels is the cell's third
                entry.  Both libraries build their model from these same
                formulas, and both then take set_params(no_warning=True,
                verbose=0).
    phantom     generate_3d_shepp_logan_low_dynamic_range(recon_shape), where
                recon_shape is whatever the model's own defaults produced.
                Used only when a sinogram has to be staged.
    sinogram    that phantom forward projected, cast to float32, read from
                disk by both libraries.
    weights     exp(-sinogram / (2 * max(sinogram))), float32.  These are the
                transmission-shaped weights the campaign uses; they are not
                uniform, so the weighted branch of both libraries is what runs.
    recon       np.random.seed(13) immediately before every call, then
                model.recon(sinogram, weights=weights, max_iterations=15,
                stop_threshold_change_pct=0.0).  The threshold is disabled so
                that every variant does exactly fifteen iterations and none of
                them stops early on its own convergence test.
    timing      one cold pass DISCARDED, then three warm passes timed with
                perf_counter around the whole call.  The row carries the
                median, the minimum, the maximum, and the spread, which is
                (max - min) / median.
    memory      the largest per-device peak.  How each library's peak is read
                is NOT the same on both sides; see the next paragraph.

THE MEMORY CAVEAT, which matters when reading the two peak columns against
each other.  On the torch side the per-device peak counters are reset after
the cold pass, so the torch peak covers the warm passes alone.  On the jax
side there is no equivalent: peak_bytes_in_use counts from the start of the
process and cannot be reset, so the jax peak covers the whole process,
including whatever the cold pass allocated while compiling.  The jax column is
therefore the larger of the two for that reason alone, over and above any real
difference in what the libraries hold.  This is the same pair of instruments
the recorded cross-framework tables were read with, so these columns compare
with those; it is stated here, on every jax row, and under the printed table so
nobody has to rediscover it.

THE PIN.  One device on both sides, by each library's own mechanism.  On the
torch side each variant runs in a fresh subprocess with MBIRTORCH_NUM_DEVICES
set to 1, which fixes the count while leaving the model on the automatic
branch where the memory preflight still runs.  On the jax side
configure_devices(1) is the mechanism, and the realized device list is read
back off the model and asserted.  Both sides record what they actually got.

STAGING.  One sinogram per geometry, shared by both libraries, so the two are
reconstructing byte-identical input and the comparison is controlled rather
than incidental.  The files are the ones the reference-timing runs already
staged: same directory, same names, same md5 sidecars.  A missing file is
built here by a torch staging job, by the same recipe.  The md5 is verified on
every load and recorded on every row, because a truncated read on a shared
parallel filesystem is a recorded failure mode of this work.  The 1024-class
files are about 4.1 GB each and are NOT deleted.

THE CROSS-FRAMEWORK VALUE CHECK IS A REPORT, NOT A GATE.  Within each geometry
the run prints how far the two libraries' fingerprints are apart, on both
components.  It does not fail on that number and it deliberately sets no
tolerance.  The two libraries are separate implementations: they partition the
VCD work differently and they sum in a different order, so their answers are
not expected to agree to the last digits, and how closely they do agree was
already measured by the run that produced the recorded comparison tables.  The
gap is printed so that a difference far outside that class would be visible to
a reader, and for nothing else.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned
variant produced a row, realized exactly one device, and read an md5-verified
sinogram, and when every torch variant bound hand-written kernels in both
projection directions on CUDA.  It is NOT a verdict on the timings.  A slow
variant, a wide spread, a hot GPU, a peak that moved, and any distance between
the two libraries' answers are all printed in full and none of them touches
the exit code.  A person reads the table.

THE ORDER: parallel torch, parallel jax, cone torch, cone jax.  Parallel runs
first because it is the cheaper geometry, so a defect in the harness surfaces
in minutes rather than after the cone passes.  Each geometry's staging runs
immediately before its own two variants.

WALL ESTIMATE.  A warm pass at fifteen iterations is close to five times the
recorded 3-iteration time, which puts torch parallel near 106 s, torch cone
near 308 s, jax parallel near 129 s, and jax cone near 314 s.  Four passes of
each, plus compiles and the staging check, come to roughly 75 minutes.  The
sbatch asks for three hours.

OUTPUT.  One jsonl under MG41_RESULTS, named
mg41_production_<node>_<stamp>.jsonl: a run-header row, one row per stage and
per variant, and a summary row.  Rows are flushed as they finish, so a job cut
short by wall time still yields everything it completed.  The run then prints
one markdown table with a row per (geometry, library) and columns for the cold
time, the warm median, the spread and the peak, followed by the
instrument-health block.

Run:
    <torch python> mg41_production_compare.py       on a one-GPU node
    MG41_DRY=1 <python> mg41_production_compare.py  print the plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized
token is an error, not a silent skip.
    MG41_RESULTS=<dir>      where the jsonl goes, and where a newly staged
                            sinogram goes when MG41_SINO_DIR is left unset,
                            which is the smoke's case
    MG41_SINO_DIR=<dir>     where the staged sinograms are read from, and
                            written to if one is missing.  Defaults to the
                            reference-timing run's directory on scratch, so
                            this run reuses those files rather than spending
                            an hour rebuilding them
    MG41_TORCH_PYTHON=<exe> the interpreter the torch variants run in
    MG41_JAX_PYTHON=<exe>   the interpreter the jax variants run in
    MG41_SMOKE=1            the local smoke: a tiny cell on the CPU, one
                            iteration, both libraries
    MG41_DRY=1              print the plan and exit, importing neither library
    MG41_REPEATS=3          warm repeats after the discarded cold pass
    MG41_VARIANTS=a,b       a subset, by variant id, e.g. parallel_torch
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
SMOKE = os.environ.get("MG41_SMOKE", "0") == "1"
DRY = os.environ.get("MG41_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

GEOMETRIES = ("parallel", "cone")
#: The two libraries, in RUN order within a geometry.  Torch first only so that
#: the geometry's first variant is the one whose harness is exercised most
#: often elsewhere in the campaign.
FRAMEWORKS = ("torch", "jax")
#: What each library is called in the printed table and in prose.
FRAMEWORK_LABEL = {"torch": "mbirtorch", "jax": "mbirjax"}

#: The one production cell: sinogram (views, rows, channels).
CELL = (1024, 1008, 992)
#: The smoke's stand-in.  A parallel model at (8, 24, 20) reconstructs a
#: (20, 20, 24) volume, which is large enough to exercise every code path in
#: this harness and small enough to run on a laptop CPU in seconds.
SMOKE_CELL = (8, 24, 20)

#: One device.  This run is about what a single GPU costs, so there is no
#: device sweep here; the recorded 3-iteration tables carry that.
N_DEVICES = 1

# ── the reconstruction protocol ───────────────────────────────────────────────
#: The production iteration count, and the one thing this run changes relative
#: to the campaign's 3-iteration ruler.  Fifteen is what a reconstruction at
#: these sizes actually runs, which is the whole reason the run exists.  The
#: smoke drops to one only to keep the local round trip short, and every row
#: records what it ran.
VCD_ITERATIONS = 1 if SMOKE else 15
#: The campaign's seed, reset immediately before every reconstruction so that
#: the two libraries start from the same random state.
VCD_SEED = 13
#: The cone construction: the source-detector and source-isocenter distances
#: are these multiples of the detector channel count.
CONE_SDD_PER_CHANNEL = 4.0
CONE_SID_PER_CHANNEL = 2.0

#: Warm repeats after the discarded cold pass.  Three is the campaign ruler.
WARM_REPEATS = max(1, int(os.environ.get("MG41_REPEATS", "3")))

# ── recorded context, not gates ───────────────────────────────────────────────
#: The padding witness.  504 is a four-device slice band at the 2048 cell and
#: 512 is what the rounding must turn it into; a tree without the padding
#: either has no such function or returns 504.  Recorded on every torch row so
#: a reader can tell which tree produced these numbers without leaving the
#: jsonl.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512
#: The shipped forward pixel batch (tomography_model.FORWARD_PIXEL_BATCH).  No
#: variant sets it; every torch row records what the model reported.
SHIPPED_PIXEL_BATCH = 32768

#: How each library's peak is read, carried on every row so the two numbers are
#: never compared without their difference in front of the reader.
TORCH_PEAK_BASIS = ("torch.cuda.max_memory_allocated per device, with the "
                    "counters reset after the cold pass, so this peak covers "
                    "the warm passes alone")
JAX_PEAK_BASIS = ("jax device.memory_stats()['peak_bytes_in_use'] per device.  "
                  "This counter runs from the start of the process and cannot "
                  "be reset, so this peak covers the whole process including "
                  "what the cold pass allocated while compiling.  It is "
                  "therefore larger than the torch peak beside it for that "
                  "reason alone, on top of any real difference.  The recorded "
                  "cross-framework tables were read with this same pair of "
                  "instruments, so these columns compare with those.")

# ── the GPU health sample ─────────────────────────────────────────────────────
HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

#: Elements of the reconstruction promoted to float64 at a time when the
#: fingerprint is taken.  Eight million float64 is 64 MiB, which bounds the
#: reading's own memory at any volume size.
FINGERPRINT_CHUNK_ELEMS = 1 << 23

RESULTS_DIR = os.environ.get(
    "MG41_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
#: Where the shared sinograms live.  The default is the reference-timing run's
#: directory, whose files this run reuses: rebuilding a 4 GB sinogram would
#: cost an hour and, because the forward kernel's atomics are not bit-exact
#: across runs, would also change what is being reconstructed.  The smoke has
#: no such directory, so it stages into its own results directory.
SINO_DIR = os.environ.get(
    "MG41_SINO_DIR",
    RESULTS_DIR if SMOKE
    else "/scratch/gautschi/buzzard/torch_p3/results/mg27_reference")

#: One interpreter per library, because the two do not live in the same
#: environment on the cluster.  Both default to the interpreter running this
#: driver, which is what makes a single-environment local smoke work.
TORCH_PYTHON = os.environ.get("MG41_TORCH_PYTHON", sys.executable)
JAX_PYTHON = os.environ.get("MG41_JAX_PYTHON", sys.executable)

RUN_LABEL = platform.node().split(".")[0]
VARIANT_COL = 22               # wide enough for the longest job id printed
# ──────────────────────────────────────────────────────────────────────────────


def cell_shape():
    return SMOKE_CELL if SMOKE else CELL


def variant_id(geometry, framework):
    return f"{geometry}_{framework}"


def all_variant_ids():
    """Every variant, in RUN order: parallel before cone, and within a geometry
    torch before jax."""
    return [variant_id(g, f) for g in GEOMETRIES for f in FRAMEWORKS]


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer variants than it printed has cost this work a
    repeat before.  The error names the full valid list, because the caller
    who mistyped one id needs to see the others.
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
            raise ValueError(f"{env_name}: {token!r} is not a variant of this "
                             f"run.  The valid ids are: "
                             f"{', '.join(allowed)}")
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}.  The valid "
                         f"ids are: {', '.join(allowed)}")
    # Normalized to the DECLARED order: the run order is load-bearing (the
    # cheaper geometry first), so it must not depend on the order someone typed
    # the tokens in.
    return [name for name in allowed if name in chosen]


# ── the staged sinogram ───────────────────────────────────────────────────────
def _sino_path(geometry, cell):
    """One file per geometry, under the shared sinogram directory.

    The name is the reference-timing run's, deliberately: this run reads that
    run's files, and a different name here would mean rebuilding four
    gigabytes per geometry to reconstruct the same thing.  The cell's view
    count is in the name, so a smoke run and a production run can share a
    directory without either reading the other's bytes.
    """
    return os.path.join(SINO_DIR, f"mg27_sino_{geometry}_{cell[0]}.npy")


def _md5_path(path):
    return path + ".md5"


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: at the production cell the file is
    over four gigabytes and reading it whole to hash it would be wasteful."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _staged(path):
    return os.path.exists(path) and os.path.exists(_md5_path(path))


def _to_numpy(x):
    """The one host exit, and it serves both libraries.

    ``Shards.gather()`` ALREADY returns numpy, so a gather is never followed by
    ``.detach()`` -- re-detaching one is a recorded way to lose every
    multi-device row in a run.  A jax array has neither method and falls
    through to ``np.asarray``.
    """
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, "gather", None)) and hasattr(x, "placement"):
        return x.gather()                      # ALREADY numpy: do not re-detach
    return (x.detach().cpu().numpy()
            if callable(getattr(x, "detach", None)) else np.asarray(x))


def _weights(sinogram):
    """The campaign's weighting formula, one dtype, every variant.

    These weights are not uniform, so both libraries take their weighted path,
    which is what a real reconstruction runs.
    """
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


# ── the GPU health sample ─────────────────────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks (SM and memory), temperatures (core and HBM), and active
    throttle reasons, via nvidia-smi.  ``[]`` when nvidia-smi is unavailable,
    which is the case on the local smoke."""
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
    """Hot by TEMPERATURE alone.  Recorded, never gated: a boost governor at a
    normal temperature is the machine working as designed."""
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


def throttle_reasons(health):
    seen = []
    for gpu in health:
        for reason in gpu.get("throttle", []):
            if reason not in seen:
                seen.append(reason)
    return seen


# ── the models, one construction per library from the same formulas ───────────
def build_model(geometry, cell, cpu_devices=None):
    """The torch model.

    ``cpu_devices`` is for the smoke only.  On CUDA nothing is configured here:
    the count comes from MBIRTORCH_NUM_DEVICES, which keeps the model on the
    automatic branch where the memory preflight still runs, and an explicit
    configure_devices call would take the explicit branch instead.  The pin is
    a CUDA mechanism -- the policy short-circuits when fewer than two CUDA
    devices are visible -- so the smoke places its CPU device by hand and every
    row records which mechanism actually pinned it.
    """
    import numpy as np

    import mbirtorch

    num_views, _num_rows, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(tuple(cell), angles)
    elif geometry == "cone":
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            tuple(cell), angles,
            source_detector_dist=CONE_SDD_PER_CHANNEL * num_channels,
            source_iso_dist=CONE_SID_PER_CHANNEL * num_channels)
    else:
        # Falling through to parallel beam would time parallel beam and record
        # the result under another geometry's name, which is the one way a
        # comparison can be wrong without anything looking wrong.
        raise ValueError(f"mg41 has no torch model construction for geometry "
                         f"{geometry!r}")
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def build_jax_model(geometry, cell):
    """The jax model, from the SAME angle and distance formulas as the torch
    one.  If these two constructions ever drift apart the run stops comparing
    libraries and starts comparing geometries, so they are written to be read
    side by side."""
    import numpy as np

    import mbirjax

    num_views, _num_rows, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirjax.ParallelBeamModel(tuple(cell), angles)
    elif geometry == "cone":
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirjax.ConeBeamModel(
            tuple(cell), angles,
            source_detector_dist=CONE_SDD_PER_CHANNEL * num_channels,
            source_iso_dist=CONE_SID_PER_CHANNEL * num_channels)
    else:
        raise ValueError(f"mg41 has no jax model construction for geometry "
                         f"{geometry!r}")
    model.set_params(no_warning=True, verbose=0)
    return model


def pin_devices_for(n_dev):
    """The explicit torch device list a variant needs, or None.

    None on CUDA, where MBIRTORCH_NUM_DEVICES does the pinning.  A list of
    virtual CPU devices on the smoke, where the environment pin cannot.
    """
    return None if DEVICE == "cuda" else ["cpu"] * n_dev


def expected_torch_bodies():
    """Which projection directions are expected to run as general torch code.

    None on CUDA: both geometries bind hand-written Triton kernels in both
    directions there, so a non-empty reading means this run measured a
    different implementation than the recorded tables describe.  Both
    directions on CPU, where those kernels do not exist.
    """
    return [] if DEVICE == "cuda" else ["forward", "back"]


def mbirjax_revision(module_file):
    """The git revision of the mbirjax checkout, or 'unknown'.

    Which mbirjax produced these numbers is part of the reading, and a reader
    a month from now cannot recover it from the version string alone.  A copy
    installed from a wheel has no git tree, so a failure here reports
    'unknown' rather than stopping the variant.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(module_file)))
    try:
        proc = subprocess.run(["git", "-C", root, "log", "-1", "--format=%h"],
                              capture_output=True, text=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:                                             # noqa: BLE001
        pass
    return "unknown"


# ── the value fingerprint ─────────────────────────────────────────────────────
def fingerprint(volume):
    """Two float64 reductions of a reconstruction: the sum of absolute values
    and the sum of squares.

    Two numbers rather than one, because a sum of absolute values alone cannot
    see a rearrangement that preserves magnitudes.  Both accumulate in float64
    in fixed-size chunks: a float32 sum over a billion-element volume loses the
    digits this comparison needs, and promoting the whole volume at once would
    double what the reading costs in memory.

    This one reduces on the host in numpy, where the recorded within-library
    checks reduce on the device in torch, and the reason is the comparison it
    serves.  Both libraries' volumes have to be reduced by the same arithmetic
    in the same order, or the number printed would be partly a property of the
    two rulers rather than of the two reconstructions.  numpy is also the only
    thing both environments are guaranteed to have.
    """
    import numpy as np

    flat = np.ascontiguousarray(volume).reshape(-1)
    abs_sum = 0.0
    sq_sum = 0.0
    for start in range(0, flat.shape[0], FINGERPRINT_CHUNK_ELEMS):
        block = flat[start:start + FINGERPRINT_CHUNK_ELEMS].astype(np.float64)
        abs_sum += float(np.sum(np.abs(block)))
        sq_sum += float(np.sum(block * block))
    return abs_sum, sq_sum


def relative_gap(value, reference):
    """|value - reference| / |reference|, with a zero reference reported as an
    absolute gap rather than as infinity."""
    if value is None or reference is None:
        return None
    scale = abs(reference)
    return abs(value - reference) / (scale if scale > 0.0 else 1.0)


# ── the workers: one stage or one variant, each in its own process ────────────
def _smoke_import_skip(framework):
    """In the SMOKE only, a library that does not import is a skip.

    The smoke's job is to exercise this harness on a laptop, and a laptop may
    well have only one of the two libraries installed.  Skipping with a
    recorded row keeps that case honest: the row says which library was
    missing, and the run does not pretend to have measured it.  On the cluster
    a missing library is a real failure and this returns nothing, so the import
    raises where it stands.
    """
    if not SMOKE:
        return None
    module = "mbirtorch" if framework == "torch" else "mbirjax"
    try:
        __import__(module)
    except Exception as exc:                                      # noqa: BLE001
        return (f"{module} does not import in {sys.executable}: "
                f"{type(exc).__name__}: {exc}")
    return None


def _read_staged(result, geometry, cell):
    """The shared sinogram, md5-verified, for either library.

    Returns the array, or None when the row already records why this variant
    cannot run.  A variant that reconstructed a different array than its
    sibling did not measure what the plan said, so an md5 mismatch stops the
    variant rather than producing a row that looks comparable and is not.
    """
    import numpy as np

    path = _sino_path(geometry, cell)
    result["sino_path"] = path
    if not _staged(path):
        if SMOKE:
            result.update(skipped=True,
                          skip_reason=f"no staged sinogram at {path}; the "
                                      f"smoke's staging did not run")
        else:
            result["invalid_reasons"].append(
                f"no staged sinogram at {path}")
        return None
    with open(_md5_path(path)) as handle:
        expected = handle.read().strip()
    actual = _md5(path)
    result.update(sino_md5=actual, sino_md5_ok=(actual == expected))
    if not result["sino_md5_ok"]:
        result["invalid_reasons"].append(
            f"the staged sinogram at {path} hashes to {actual}, not the "
            f"recorded {expected}")
        return None
    return np.load(path)


def _base_result(cfg):
    """The fields every torch row carries, whatever the job is."""
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, framework="torch", library="mbirtorch",
                  version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  warm_repeats=WARM_REPEATS,
                  peak_basis=TORCH_PEAK_BASIS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "MBIRTORCH_NUM_DEVICES is set as on CUDA, and "
                                 "the count is realized by "
                                 "configure_devices(devices=['cpu', ...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"))
    result["invalid_reasons"] = []

    # The calibration mode owns the per-device peak counters, and this run
    # reads those counters itself, so the mode must be absent everywhere.
    calibration = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION")
    result["calibration_absent_ok"] = calibration in (None, "", "0")
    if not result["calibration_absent_ok"]:
        result["invalid_reasons"].append(
            f"MBIRTORCH_MEMORY_CALIBRATION is {calibration!r}; it owns the "
            f"peak counters this run reads itself")

    # The padding witness, recorded so a reader can tell which tree produced
    # these numbers from the row alone.  Recorded, not gated: the sbatch
    # asserts it before any variant runs.
    try:
        from mbirtorch._utils import padded_kernel_width
        result["padded_kernel_width_probe"] = int(
            padded_kernel_width(PAD_PROBE_WIDTH))
    except Exception as exc:                                      # noqa: BLE001
        result["padded_kernel_width_probe"] = None
        result["padded_kernel_width_error"] = f"{type(exc).__name__}: {exc}"
    result["padding_present"] = (
        result["padded_kernel_width_probe"] == PAD_PROBE_EXPECTED)
    return result, cuda


def run_stage(cfg):
    """Make sure ONE geometry's sinogram is on disk, and verify it.

    Normally the file is already there from the reference-timing runs and this
    only re-hashes it, which is the point: both libraries then reconstruct the
    same bytes those runs did, so this run's rows and the recorded ones are
    about the same input.  When the file is absent it is built here by the same
    recipe -- phantom, forward projection, float32 -- and kept, so the next run
    reuses it.

    The staging process is pinned to one device, and the projection is taken on
    a freshly built model, so nothing here is a multi-device run.
    """
    import numpy as np

    import mbirtorch

    result, cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    path = _sino_path(geometry, cell)
    result["sino_path"] = path
    result["sino_dir"] = SINO_DIR

    if _staged(path):
        # Already on disk.  Verify it rather than rebuild it: the forward
        # kernel's atomics make a regenerated sinogram non-identical at the e-7
        # class, so a rebuild would silently change what both libraries
        # reconstruct and would make this run's rows incomparable to the
        # recorded ones.
        with open(_md5_path(path)) as handle:
            expected = handle.read().strip()
        actual = _md5(path)
        result.update(reused=True, sino_md5=actual,
                      sino_md5_ok=(actual == expected))
        if not result["sino_md5_ok"]:
            result["invalid_reasons"].append(
                f"the staged sinogram at {path} hashes to {actual}, not the "
                f"recorded {expected}")
        array = np.load(path, mmap_mode="r")
        result["sinogram_shape"] = list(array.shape)
        return result

    model = build_model(geometry, cell, cpu_devices=pin_devices_for(1))
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    result["phantom_max"] = float(np.max(phantom))
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(SINO_DIR, exist_ok=True)
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    result.update(reused=False, sino_md5=digest, sino_md5_ok=True,
                  sinogram_shape=list(sinogram.shape),
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)),
                  stage_devices=[str(d)
                                 for d in model.sino_placement.devices])
    return result


def run_torch_variant(cfg):
    """One mbirtorch variant: a cold pass discarded, then WARM_REPEATS timed
    warm passes.

    ORDERING NOTE, load-bearing.  Every check that reads the projectors runs
    AFTER the cold pass.  The automatic branch settles the device layout inside
    the first ``recon`` call, and a settle that changes the count rebuilds
    ``model.projector_functions``.  A body reading taken before that would
    describe a one-device projector set under whatever label happened to apply.
    """
    import numpy as np
    import torch

    from mbirtorch import _memory_ledger

    result, cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])

    model = build_model(geometry, cell, cpu_devices=pin_devices_for(n_dev))
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    sinogram = _read_staged(result, geometry, cell)
    if sinogram is None:
        return result
    weights = _weights(sinogram)

    devices_now = [None]

    def vcd():
        """One reconstruction, the campaign's call with the production
        iteration count."""
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(recon)

    # ── the cold pass, DISCARDED (it pays the kernel compiles) ───────────────
    start = time.perf_counter()
    out = vcd()
    result["cold_s"] = time.perf_counter() - start

    # ── the layout has settled, so the projector-dependent checks can run ────
    devices_now[0] = list(model.sino_placement.devices)
    realized = [str(d) for d in devices_now[0]]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev)
    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"pinned to {n_dev} device(s) and realized {len(realized)}: "
            f"{realized}")
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))

    fwd_body, back_body = model._view_batch_bodies()
    result["fwd_body"] = getattr(fwd_body, "__name__", str(fwd_body))
    result["back_body"] = getattr(back_body, "__name__", str(back_body))
    bodies = list(_memory_ledger.torch_body_directions(model))
    result["torch_body_directions"] = bodies
    result["torch_body_directions_expected"] = expected_torch_bodies()
    result["torch_bodies_ok"] = (bodies == expected_torch_bodies())
    if not result["torch_bodies_ok"]:
        result["invalid_reasons"].append(
            f"projection directions running as general torch code are "
            f"{bodies}, expected {expected_torch_bodies()} on {DEVICE}")

    # Recorded, not set.  Every row says which pixel batch it ran.
    try:
        result["forward_pixel_batch"] = int(model._forward_pixel_batch())
    except Exception as exc:                                      # noqa: BLE001
        result["forward_pixel_batch"] = None
        result["forward_pixel_batch_error"] = f"{type(exc).__name__}: {exc}"
    result["forward_pixel_batch_shipped"] = SHIPPED_PIXEL_BATCH

    # ── the warm repeats ─────────────────────────────────────────────────────
    # The peak counters are reset AFTER the cold pass, so the torch memory
    # reading covers the warm passes and not the compiles.  The jax side has no
    # equivalent; see the module docstring.
    if cuda:
        for device in devices_now[0]:
            torch.cuda.reset_peak_memory_stats(device)
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        warm.append(time.perf_counter() - start)
    result["warm_all"] = warm
    result["warm_s"] = statistics.median(warm)
    result["warm_min"] = min(warm)
    result["warm_max"] = max(warm)
    result["warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    if cuda:
        peaks = [int(torch.cuda.max_memory_allocated(d))
                 for d in devices_now[0]]
    else:
        peaks = []
    result["peak_per_device_bytes"] = peaks
    result["peak_bytes"] = max(peaks, default=0)

    # ── the value fingerprint of the LAST warm reconstruction ────────────────
    abs_sum, sq_sum = fingerprint(out)
    result["fingerprint_abs_sum"] = abs_sum
    result["fingerprint_sq_sum"] = sq_sum
    result["fingerprint_elements"] = int(np.asarray(out).size)
    return result


def run_jax_variant(cfg):
    """One mbirjax variant: the same protocol, the same staged sinogram, in a
    process of its own.

    The two differences from the torch variant are both properties of the
    library and both are recorded on the row.  mbirjax has no environment pin,
    so ``configure_devices(n)`` is the mechanism and the realized list is read
    back off the model.  And its peak counter cannot be reset, so the peak this
    row carries covers the whole process rather than the warm passes alone.
    """
    import numpy as np

    import jax
    import mbirjax

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])

    result = dict(cfg, framework="jax", library="mbirjax",
                  version=f"jax {jax.__version__}",
                  jax_version=jax.__version__,
                  mbirjax_file=mbirjax.__file__,
                  mbirjax_revision=mbirjax_revision(mbirjax.__file__),
                  device=DEVICE,
                  jax_platform=jax.devices()[0].platform,
                  visible_devices=len(jax.devices()),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  warm_repeats=WARM_REPEATS,
                  peak_basis=JAX_PEAK_BASIS,
                  pin_mechanism="mbirjax configure_devices(n)")
    result["invalid_reasons"] = []

    model = build_jax_model(geometry, cell)
    model.configure_devices(n_dev)
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    # No silent fallback.  Falling back to jax.devices()[:n] here would make
    # the realized-device check vacuously true if mbirjax ever renamed this
    # property, which is exactly the kind of vacuity the check exists to catch.
    shard_devices = getattr(model, "shard_devices", None)
    if shard_devices is None:
        raise RuntimeError("mbirjax model has no shard_devices property; the "
                           "realized-device assertion cannot run")
    realized = [str(d) for d in shard_devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"configure_devices({n_dev}) realized {len(realized)} device(s): "
            f"{realized}")

    sinogram = _read_staged(result, geometry, cell)
    if sinogram is None:
        return result
    weights = _weights(sinogram)

    def vcd():
        """The same call the torch variant makes, with the same seed reset
        immediately before it."""
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        return _to_numpy(recon)

    start = time.perf_counter()
    out = vcd()
    result["cold_s"] = time.perf_counter() - start

    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        warm.append(time.perf_counter() - start)
    result["warm_all"] = warm
    result["warm_s"] = statistics.median(warm)
    result["warm_min"] = min(warm)
    result["warm_max"] = max(warm)
    result["warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    # Read over every visible device, which is what the recorded jax column
    # read.  On the CPU smoke memory_stats returns nothing and the peak is 0.
    peaks = []
    for device in jax.devices():
        stats = device.memory_stats() or {}
        peaks.append(int(stats.get("peak_bytes_in_use", 0)))
    result["peak_per_device_bytes"] = peaks
    result["peak_bytes"] = max(peaks, default=0)
    result["peak_covers_cold_pass"] = True

    abs_sum, sq_sum = fingerprint(out)
    result["fingerprint_abs_sum"] = abs_sum
    result["fingerprint_sq_sum"] = sq_sum
    result["fingerprint_elements"] = int(np.asarray(out).size)
    return result


def run_job(cfg):
    """One stage or one variant, in its own process, with a health sample on
    either side of it.

    A new process per job is not tidiness.  Compiled and hand-written kernel
    bodies are cached at module level for the life of a process, and the peak
    memory counters are per process, so both would leak from one variant into
    the next if they shared an interpreter.  The two libraries also do not
    share an environment on the cluster, so they could not share one anyway.
    """
    skip = _smoke_import_skip(cfg["framework"])
    if skip:
        return dict(cfg, skipped=True, skip_reason=skip,
                    vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS)

    before = sample_gpu_health()
    started = time.time()
    try:
        if cfg["kind"] == "stage":
            result = run_stage(cfg)
        elif cfg["framework"] == "jax":
            result = run_jax_variant(cfg)
        else:
            result = run_torch_variant(cfg)
    finally:
        after = sample_gpu_health()
    result["gpu_health_before"] = before
    result["gpu_health_after"] = after
    result["gpu_hot"] = row_is_hot(before) or row_is_hot(after)
    result["gpu_throttle"] = throttle_reasons(before + after)
    result["worker_wall_s"] = time.time() - started
    return result


# ── the driver ────────────────────────────────────────────────────────────────
def job_env(cfg):
    """The environment that DEFINES a job, set explicitly so nothing is
    inherited from the submitting shell.

    On the torch side MBIRTORCH_NUM_DEVICES is popped and then set, so a value
    exported by the shell cannot reach a job that asked for a different count.
    It is set on the smoke too, exactly as on CUDA, so the subprocess protocol
    under test is the same one the real run uses; the smoke's count is then
    realized by an explicit CPU device list, because the pin acts only through
    the device policy and that policy short-circuits below two visible CUDA
    devices.  A jax job gets neither variable: its pin is a call, not an
    environment value, and setting a torch variable in a jax process would only
    invite a reader to think otherwise.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # it owns the peak counters
    if cfg["framework"] == "torch":
        env["MBIRTORCH_DISABLE_TRITON"] = "0"         # the shipped configuration
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def interpreter_for(cfg):
    """The interpreter a job runs in, chosen by library: the two do not live in
    the same environment on the cluster, and both default to this driver's own
    interpreter so a single-environment local smoke still works."""
    return JAX_PYTHON if cfg["framework"] == "jax" else TORCH_PYTHON


def spawn(cfg):
    """Run one configuration in a NEW interpreter.

    The row goes through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.  On an hour-long job that is
    the difference between watching progress and waiting in the dark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_mg41_cfg_{cfg['job_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_mg41_out_{cfg['job_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    start = time.perf_counter()
    proc = subprocess.run([interpreter_for(cfg), "-u", os.path.abspath(__file__),
                           "--worker", cfg_path, out_path], env=job_env(cfg))
    wall = time.perf_counter() - start
    if not os.path.exists(out_path):
        # A job that ran out of device memory lands here.  That is a reading,
        # not a harness fault, so it is recorded as a row and the run goes on.
        row = dict(cfg, error=f"worker exited {proc.returncode} and wrote no "
                              f"row")
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    row["subprocess_wall_s"] = wall
    return row


def build_plan():
    """Every job, in run order: each geometry's staging immediately before that
    geometry's own two variants, parallel first."""
    keep = _strict_subset("MG41_VARIANTS", all_variant_ids())
    cell = cell_shape()
    plan = []
    for geometry in GEOMETRIES:
        wanted = [f for f in FRAMEWORKS if variant_id(geometry, f) in keep]
        if not wanted:
            continue
        plan.append(dict(kind="stage", framework="torch", geometry=geometry,
                         cell=list(cell), n_dev=N_DEVICES,
                         job_id=f"stage_{geometry}_{cell[0]}"))
        for framework in wanted:
            name = variant_id(geometry, framework)
            plan.append(dict(kind="variant", framework=framework,
                             geometry=geometry, cell=list(cell),
                             n_dev=N_DEVICES, variant=name, job_id=name))
    if not plan:
        raise ValueError("MG41_VARIANTS selects no variant")
    return plan


def print_plan(plan):
    variants = [c for c in plan if c["kind"] == "variant"]
    stages = [c for c in plan if c["kind"] == "stage"]
    cell = cell_shape()
    print(f"mg41 the production-shaped cross-framework comparison: "
          f"{len(variants)} variant(s) and {len(stages)} staged sinogram(s), "
          f"device {DEVICE}, cell sinogram {cell}, {N_DEVICES} device, "
          f"{VCD_ITERATIONS} VCD iteration(s), "
          f"{WARM_REPEATS} warm repeat(s) after a discarded cold pass")
    print(f"  jsonl -> {RESULTS_DIR}")
    print(f"  staged sinograms read from (and written to, if missing) "
          f"-> {SINO_DIR}")
    print(f"  mbirtorch interpreter -> {TORCH_PYTHON}")
    print(f"  mbirjax   interpreter -> {JAX_PYTHON}")
    print(f"  this is a {VCD_ITERATIONS}-iteration reading and does not "
          f"replace the recorded 3-iteration tables; it answers what one real "
          f"reconstruction costs on one GPU")
    print(f"  projection directions expected to run as general torch code on "
          f"{DEVICE}: {expected_torch_bodies() or 'none'}")
    print("  the two peak columns come from different instruments: the torch "
          "counters are reset after the cold pass, and the jax counter runs "
          "from process start and cannot be reset, so the jax peak also "
          "includes the cold pass's compile allocations")
    print("  the distance between the two libraries' answers is REPORTED and "
          "not gated: they are separate implementations whose partitions and "
          "summation orders differ")
    total_gib = cell[0] * cell[1] * cell[2] * 4 / 2 ** 30 * len(stages)
    print(f"  the staged sinograms total about {total_gib:.2f} GiB and are "
          f"kept")
    header = (f'  {"job":<{VARIANT_COL}}{"library":>10}{"pin":>5}{"cell":>20}'
              f'{"geometry":>10}  what it does')
    print(header)
    what = dict(stage="verifies this geometry's sinogram, building it if absent",
                variant="cold pass discarded, then the timed warm passes")
    for cfg in plan:
        print(f'  {cfg["job_id"]:<{VARIANT_COL}}'
              f'{FRAMEWORK_LABEL[cfg["framework"]]:>10}{cfg["n_dev"]:>5}'
              f'{str(tuple(cfg["cell"])):>20}{cfg["geometry"]:>10}  '
              f'{what[cfg["kind"]]}')
    print("  no library file is edited and no default is flipped: every "
          "variant runs the shipped configuration")


def main():
    plan = build_plan()
    if DRY:
        print_plan(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg41_production_{RUN_LABEL}_{stamp}.jsonl")
    print_plan(plan)
    print(f"\nrunning -> {out_path}", flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        header = dict(row="run_header", script="mg41_production_compare.py",
                      node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
                      python=sys.executable, torch_python=TORCH_PYTHON,
                      jax_python=JAX_PYTHON, results_dir=RESULTS_DIR,
                      sino_dir=SINO_DIR,
                      geometries=list(GEOMETRIES),
                      frameworks=list(FRAMEWORKS),
                      cell=list(cell_shape()), n_devices=N_DEVICES,
                      vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                      warm_repeats=WARM_REPEATS,
                      torch_peak_basis=TORCH_PEAK_BASIS,
                      jax_peak_basis=JAX_PEAK_BASIS,
                      plan=[dict(c) for c in plan])
        sink.write(json.dumps(header) + "\n")
        sink.flush()
        for index, cfg in enumerate(plan):
            print(f'\n  [{index + 1}/{len(plan)}] {cfg["job_id"]}', flush=True)
            row = spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if row.get("error"):
                print(f'    ERROR: {str(row["error"])[:400]}', flush=True)
            elif row.get("skipped"):
                print(f'    SKIPPED: {row.get("skip_reason", "")}', flush=True)
            elif cfg["kind"] == "variant":
                print(f'    warm {row.get("warm_s", 0):.3f}s  '
                      f'spread {row.get("warm_spread", 0):.1%}  '
                      f'peak {row.get("peak_bytes", 0) / 2 ** 30:.2f} GB  '
                      f'{row.get("realized_n_devices", "-")} device(s)',
                      flush=True)
        summary = summarize(rows, plan, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        sink.write(json.dumps(dict(row="summary", **summary)) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    print(f"elapsed {summary['elapsed_min']:.1f} min")
    return 0 if summary["healthy"] else 2


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=3):
    """One table cell, with a missing value padded to the width of a present
    one, so the columns line up whether a variant produced a number or not."""
    if value is None:
        return f'{"-":>{width}}'
    return f"{value:>{width}.{prec}{kind}}"


def report_cross_framework(by_variant):
    """How far apart the two libraries' answers are, per geometry.

    REPORTED, NOT GATED, and the run's exit code never looks at it.  The two
    libraries are separate implementations: they split the VCD work
    differently and they sum in a different order, so agreement to the last
    digits was never on offer, and how close they do come was already measured
    by the run that produced the recorded comparison tables.  This is printed
    so a reader can see at a glance whether these reconstructions sit in that
    same class.
    """
    print("\n-- how far apart the two libraries' answers are (reported, not "
          "gated) --")
    print("Each gap is the mbirjax fingerprint against the mbirtorch one, "
          "relative.")
    header = f'{"geometry":<14}{"abs gap":>12}{"sq gap":>12}'
    print(header)
    print("-" * len(header))
    records = []
    for geometry in GEOMETRIES:
        torch_row = by_variant.get(variant_id(geometry, "torch"))
        jax_row = by_variant.get(variant_id(geometry, "jax"))
        if torch_row is None or jax_row is None:
            why = ("neither library produced a row"
                   if torch_row is None and jax_row is None
                   else "only one library produced a row")
            print(f'{geometry:<14}{"-":>12}{"-":>12}   {why}')
            records.append(dict(geometry=geometry, compared=False))
            continue
        abs_gap = relative_gap(jax_row.get("fingerprint_abs_sum"),
                               torch_row.get("fingerprint_abs_sum"))
        sq_gap = relative_gap(jax_row.get("fingerprint_sq_sum"),
                              torch_row.get("fingerprint_sq_sum"))
        print(f'{geometry:<14}{_fmt(abs_gap, 12, "e", 2)}'
              f'{_fmt(sq_gap, 12, "e", 2)}')
        records.append(dict(geometry=geometry, compared=True,
                            abs_gap=abs_gap, sq_gap=sq_gap))
    return records


def print_comparison_table(by_variant):
    """The one table this run exists to produce: a row per geometry and
    library, with the cold time, the warm median, the spread and the peak."""
    cell = cell_shape()
    print(f"\n### the production comparison, sinogram {tuple(cell)}, "
          f"{N_DEVICES} device, {VCD_ITERATIONS} VCD iterations")
    print("| geometry | library | cold s | warm median s | spread | peak GB |")
    print("|---|---|---|---|---|---|")
    for geometry in GEOMETRIES:
        for framework in FRAMEWORKS:
            row = by_variant.get(variant_id(geometry, framework))
            label = FRAMEWORK_LABEL[framework]
            if row is None:
                print(f"| {geometry} | {label} | - | - | - | - |")
                continue
            spread = row.get("warm_spread")
            spread_text = "-" if spread is None else f"{spread:.1%}"
            peak_gb = row.get("peak_bytes", 0) / 2 ** 30
            print(f'| {geometry} | {label} | {row.get("cold_s", 0):.2f} '
                  f'| {row.get("warm_s", 0):.2f} | {spread_text} '
                  f'| {peak_gb:.2f} |')
    where = "one H100" if DEVICE == "cuda" else "one CPU device (the smoke)"
    print(f"Warm median of {WARM_REPEATS} seeded {VCD_ITERATIONS}-iteration "
          f"reconstructions with transmission-shaped weights, on {where}, "
          f"after one cold pass that is discarded because it pays the "
          f"compiles; the cold column reports that discarded pass.  Both "
          f"libraries reconstruct the same staged sinogram, verified by md5.")
    print("The two peak columns come from different instruments and the "
          "mbirjax one is larger partly for that reason: the mbirtorch "
          "counters are reset after the cold pass, so its peak covers the warm "
          "passes alone, while mbirjax's peak_bytes_in_use runs from the start "
          "of the process and cannot be reset, so its peak also includes what "
          "the cold pass allocated while compiling.  The recorded "
          "cross-framework tables were read with the same pair of instruments.")


def summarize(rows, plan, out_path):
    """The table a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A slow
    variant, a wide spread, a hot GPU, a peak that moved, and any distance
    between the two libraries' answers are FINDINGS: they are printed and none
    of them touches the exit code.  A variant that produced no row, ran on the
    wrong device count, bound the wrong kind of projection body, or read an
    unverified sinogram is an instrument failure, because it did not measure
    what the plan said it would.
    """
    print(f"\n===== mg41 the production comparison ({out_path}) =====")
    broken, findings = [], []
    by_variant, stages, skipped = {}, [], []

    header = (f'{"variant":<{VARIANT_COL}}{"library":>10}{"dev":>5}'
              f'{"cold s":>9}{"warm s":>9}{"spread":>8}{"peak GB":>9}'
              f'{"checks":>22}')
    print(header)
    print("-" * len(header))
    for row in rows:
        job_id = row.get("job_id", "?")
        if row.get("error"):
            print(f'{job_id:<{VARIANT_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            broken.append(f"{job_id}|error")
            continue
        if row.get("skipped"):
            # Smoke only: the library is not installed in this interpreter, or
            # its staging did not run.  Recorded, and not a failure.
            print(f'{job_id:<{VARIANT_COL}}  SKIPPED: '
                  f'{str(row.get("skip_reason", ""))[:80]}')
            skipped.append(row)
            continue
        if row.get("kind") == "stage":
            stages.append(row)
            broken.extend(f"{job_id}|{reason}"
                          for reason in row.get("invalid_reasons") or [])
            continue
        by_variant[row["variant"]] = row
        marks = []
        for name, flag in (("dev", row.get("devices_ok")),
                           ("bod", row.get("torch_bodies_ok")),
                           ("md5", row.get("sino_md5_ok")),
                           ("cal", row.get("calibration_absent_ok"))):
            if flag is False:
                marks.append(f"{name}:FAIL")
        spread = row.get("warm_spread")
        spread_text = "-" if spread is None else f"{spread:.1%}"
        print(f'{row["variant"]:<{VARIANT_COL}}'
              f'{FRAMEWORK_LABEL[row["framework"]]:>10}'
              f'{row.get("realized_n_devices", "-"):>5}'
              f'{_fmt(row.get("cold_s"), 9, "f", 2)}'
              f'{_fmt(row.get("warm_s"), 9, "f", 2)}'
              f'{spread_text:>8}'
              f'{_fmt(row.get("peak_bytes", 0) / 2 ** 30, 9, "f", 2)}'
              f'{(",".join(marks) if marks else "ok"):>22}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    VARIANT CHECK FAIL: {reason}")
            broken.append(f'{row["variant"]}|{reason}')
        if row.get("gpu_hot"):
            findings.append(f'{row["variant"]}: GPU hot during this variant')
        if row.get("gpu_throttle"):
            findings.append(f'{row["variant"]}: throttle reasons '
                            f'{row["gpu_throttle"]}')

    # Every PLANNED variant produced a row.  Read off the plan rather than off
    # the rows: a variant whose subprocess died before writing anything leaves
    # no row to notice its absence in.  A variant already reported above as an
    # error, or recorded as a smoke skip, is not reported again.
    reported = {item.split("|", 1)[0] for item in broken}
    reported.update(row.get("job_id") for row in skipped)
    for cfg in plan:
        name = cfg.get("variant")
        if name and name not in by_variant and name not in reported:
            broken.append(f"{name}|no row")

    for row in stages:
        print(f'\nstaged {row["geometry"]} {tuple(row["cell"])}: '
              f'md5 {row.get("sino_md5", "-")}'
              f'{"  (reused from disk)" if row.get("reused") else "  (built by this run)"}')

    for row in by_variant.values():
        if row["framework"] == "jax":
            print(f'\nmbirjax under test: {row.get("mbirjax_file", "-")}  '
                  f'revision {row.get("mbirjax_revision", "-")}  '
                  f'{row.get("version", "-")}')
            break

    value_records = report_cross_framework(by_variant)
    print_comparison_table(by_variant)

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print(f"  BROKEN {item}")
    elif skipped:
        print("  every variant that was not skipped ran, realized one device, "
              "read a verified sinogram, and bound the expected projection "
              "bodies")
    else:
        print("  every planned variant ran, realized one device, read a "
              "verified sinogram, and bound the expected projection bodies")
    for row in skipped:
        print(f'  skipped (smoke only) {row.get("job_id")}: '
              f'{row.get("skip_reason", "")}')
    for item in findings:
        print(f"  finding (not gated) {item}")
    if not findings:
        print("  no thermal or throttle findings")

    return dict(healthy=not broken, broken=broken, findings=findings,
                skipped=[row.get("job_id") for row in skipped],
                variants={name: dict(cold_s=row.get("cold_s"),
                                     warm_s=row.get("warm_s"),
                                     warm_spread=row.get("warm_spread"),
                                     peak_bytes=row.get("peak_bytes"),
                                     peak_basis=row.get("peak_basis"),
                                     realized_n_devices=row.get(
                                         "realized_n_devices"))
                          for name, row in by_variant.items()},
                cross_framework=value_records)


# ── the worker entry point ────────────────────────────────────────────────────
def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = run_job(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
