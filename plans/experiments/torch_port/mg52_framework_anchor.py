"""mg52 -- MBIRTORCH AGAINST MBIRJAX ON THE TWO GEOMETRIES THAT HAVE NO
HAND-WRITTEN KERNELS: MULTIAXIS AND TRANSLATION.

WHY THIS RUN EXISTS.  Every cross-framework timing this work has recorded is
for parallel beam or cone beam, and both of those bind hand-written kernels in
both projection directions.  Multiaxis and translation bind none: their
``_view_batch_bodies`` return general torch code, which the sbatch's witness
block asserts before any measurement runs.  So the case for writing kernels
for them rests on indirect evidence -- kernel gains measured on OTHER
geometries, and mbirtorch-only timings that have nothing to divide by.  This
run measures the missing number directly: the same staged sinograms
reconstructed by both libraries, at matched sizes, on ONE device, with the
walls, the peak device memory and the value fingerprints printed side by side.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It edits no library file,
flips no default and sets no knob under test.  Both frameworks run their
shipped configuration.  Neither framework's numbers are a verdict on the
other's: the exit code reports whether the instrument worked, and a person
reads the table.

TERMS, defined once here.
    cell        one problem to measure: a geometry and a sinogram shape.
    staging     the untimed step that builds one cell's sinogram and weights
                once, with mbirtorch, and writes them to one npz.
    arm         one framework reconstructing one cell, in its own process.
    cold pass   the first reconstruction in a process.  It pays the compiles,
                so it is timed and reported but never used for the ratio.
    warm pass   a reconstruction after the cold one.  Three are timed and the
                row carries their median.
    fingerprint two float64 reductions of a reconstruction -- the sum of
                absolute values and the sum of squares -- taken on the host in
                numpy for BOTH frameworks, so the fingerprint compares the
                volumes rather than the two libraries' reduction orders.

THE CELLS, in run order, cheap first, so a harness defect shows up in minutes
rather than after an hour:

    translation  (256, 1900, 3000)   the production translation scan
    multiaxis    (512, 448, 384)
    multiaxis    (768, 672, 576)
    multiaxis    (1024, 1008, 992)

THE MODEL CONSTRUCTION is the widening-floors refresh's, per geometry, and
every constant below was read out of ``dev_scripts/refresh_widening_floors.py``
in session rather than remembered.  It is that protocol and not another one
because the recorded mbirtorch walls this run sits beside were measured under
it.

    multiaxis    two angles per view: azimuths evenly spaced over half a turn,
                 np.linspace(0, pi, num_views, endpoint=False), and elevations
                 swept across +/- 0.5 radians, np.linspace(-0.5, 0.5,
                 num_views), stacked into a (num_views, 2) array.  The
                 elevation range is part of what a multiaxis cell measures:
                 the automatic geometry divides the detector height by the
                 smallest |cos(elevation)|, so a wider sweep would inflate the
                 slice count.
    translation  a 16 x 16 grid of object translations -- 256 views -- with
                 24.0 and 16.0 ALU spacing in x and z, from
                 gen_translation_vectors, and both source distances equal to
                 half the smaller detector extent.  A sinogram shape alone
                 does not determine a translation model the way an angle list
                 does, so the grid and the spacing are written down here.
    both         phantom: generate_3d_shepp_logan_low_dynamic_range at the
                 model's own recon shape, with the refresh's fallback to a
                 seeded uniform volume when the phantom comes back all zeros
                 (which happens on a volume only a few voxels deep, as in the
                 smoke's translation cell).  Sinogram: that phantom forward
                 projected, float32.  Weights: exp(-sinogram / (2 *
                 max(sinogram))), float32.

THE MEASUREMENT, identical for both frameworks: np.random.seed(13)
immediately before every reconstruction, then a 3-iteration recon with the
stopping threshold disabled, one cold pass, then three timed warm passes on a
host clock.  Both libraries draw their pixel partitions from numpy's global
generator, so the seed is the same mechanism in both.  Both return a host
numpy array from ``recon``, so the torch call is already synchronous when the
clock stops; the jax result is passed through ``jax.block_until_ready``
anyway, so nothing depends on that staying true.

MATCHED MODELS, CHECKED RATHER THAN ASSUMED.  The staged npz carries every
constructor parameter the jax model needs -- the per-view parameters, the
source distances, the sinogram shape -- plus the recon shape mbirtorch
realized.  Each arm rebuilds its framework's model from those parameters and
CHECKS its own recon shape against the staged one.  A mismatch stops that arm:
two libraries reconstructing different volumes are not comparable, and the one
way this run could be quietly wrong is to compare them anyway.  The derived
voxel geometry (delta_voxel and the two aspect ratios) is recorded on both
sides and compared in the report; a difference there is printed, not gated.

ONE DEVICE, BOTH FRAMEWORKS.  The torch arms pin with MBIRTORCH_NUM_DEVICES=1,
which is the refresh's mechanism and keeps the model on the automatic branch.
The jax arms call configure_devices(1), which is mbirjax's documented way to
force single-device operation.  Each arm records how many devices it realized.

FAILURE IS A READING.  An arm that runs out of device memory or hits the
per-arm time cap is recorded as that arm's result and the run continues.  At
the largest multiaxis cell either framework may genuinely not fit one device,
and that outcome is itself the datum this run went looking for.  The exit code
separates those readings from instrument faults: a capacity or timeout result
is healthy, an unrecognized error is not.

THE FINGERPRINTS ARE RECORDED, NOT GATED.  Cross-framework differences for
these geometries at three iterations were measured near 1e-3 relative during
the port, so a larger gap prints a note and changes nothing.  Two different
libraries running different code in a different order at float32 are not
expected to agree more tightly than that, and a gate at this altitude would
report arithmetic, not correctness.

MEMORY, AND WHY THE TWO NUMBERS ARE NOT QUITE THE SAME NUMBER.  The torch arm
reads torch.cuda.max_memory_allocated, which it resets after the cold pass, so
it can report the cold peak and the warm peak separately.  jax exposes a
device peak through memory_stats but no way to reset it, so the jax peak
covers the whole process.  The comparison therefore uses each arm's
PROCESS-LIFETIME peak -- for torch, the larger of its cold and warm readings.
Both are in the row.  The jax arms also record the allocator's bytes_limit,
because jax preallocates a fraction of the device by default and a jax arm
that runs out of memory ran out of that fraction, not of the card.

OUTPUT.  One jsonl under MG52_RESULTS, named
mg52_framework_anchor_<node>_<stamp>.jsonl: a header row carrying both
frameworks' versions, the GPU, both library commit identities and the staged
files' md5 sums; one row per staged cell; one row per arm; one comparison row
per cell; and a summary row.  Rows are flushed as they are written, so a job
that runs out of wall time still yields everything it finished, and MG52_ARMS
re-runs the rest.

Run:
    <torch python> mg52_framework_anchor.py       on a one-GPU node
    MG52_DRY=1 <any python> mg52_framework_anchor.py     the plan, then stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  MG52_ARMS is parsed strictly: an unrecognized token
is an error, not a silent skip.
    MG52_RESULTS=<dir>        where the jsonl goes
    MG52_STAGE_DIR=<dir>      where the staged npz files go (default: results)
    MG52_TORCH_PYTHON=<path>  the interpreter the torch arms and staging run in
    MG52_JAX_PYTHON=<path>    the interpreter the jax arms run in (required)
    MG52_TORCH_PYTHONPATH=<p> PYTHONPATH for torch jobs only (the local smoke)
    MG52_JAX_PYTHONPATH=<p>   PYTHONPATH for jax jobs only
    MG52_DRY=1                print the plan and exit, importing no framework
    MG52_SMOKE=1              tiny cells on CPU, both frameworks, whole flow
    MG52_ARMS=a,b             run only these arms, e.g. multiaxis_512_jax
    MG52_ARM_TIMEOUT_MIN=45   the per-arm hard time cap, in minutes
    MG52_REPEATS=3            warm repeats after the cold pass
    MG52_ITERATIONS=3         VCD iterations per reconstruction
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
SMOKE = os.environ.get("MG52_SMOKE", "0") == "1"
DRY = os.environ.get("MG52_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

FRAMEWORKS = ("torch", "jax")

#: The cells, in RUN order: cheapest first.  Each carries everything the model
#: construction needs that the sinogram shape does not say.  ``grid`` and
#: ``spacing`` are the translation scan's own parameters -- the number of x and
#: z translations, and their ALU spacing -- and are unused by multiaxis.
PRODUCTION_CELLS = (
    dict(name="translation_256", geometry="translation",
         cell=(256, 1900, 3000), grid=(16, 16), spacing=(24.0, 16.0)),
    dict(name="multiaxis_512", geometry="multiaxis", cell=(512, 448, 384)),
    dict(name="multiaxis_768", geometry="multiaxis", cell=(768, 672, 576)),
    dict(name="multiaxis_1024", geometry="multiaxis", cell=(1024, 1008, 992)),
)

#: The smoke's stand-ins: one cell of each geometry, small enough that the
#: whole flow -- staging, both arms, the comparison -- runs on a laptop CPU in
#: minutes.  The translation cell's 4 x 2 grid gives its 8 views; its
#: reconstruction is one row deep, which is the shape at which the shepp-logan
#: builder returns all zeros and the seeded-uniform fallback takes over.  That
#: makes the smoke exercise the fallback too.
SMOKE_CELLS = (
    dict(name="multiaxis_smoke", geometry="multiaxis", cell=(12, 16, 16)),
    dict(name="translation_smoke", geometry="translation", cell=(8, 24, 32),
         grid=(4, 2), spacing=(6.0, 4.0)),
)

# ── the reconstruction protocol, taken from the floors refresh ────────────────
#: The campaign ruler: a seeded 3-iteration reconstruction with the stopping
#: threshold disabled, so both frameworks do exactly the same amount of work.
VCD_ITERATIONS = max(1, int(os.environ.get("MG52_ITERATIONS", "3")))
#: The seed, reset immediately before every reconstruction, in both
#: frameworks.  Both libraries draw their pixel partitions from numpy's global
#: generator, so this is the same mechanism on both sides.
VCD_SEED = 13
#: Warm repeats after the cold pass.  Three is the campaign ruler.
WARM_REPEATS = max(1, int(os.environ.get("MG52_REPEATS", "3")))
#: The multiaxis elevation sweep, in radians, and the translation source
#: distance rule.  Both are the floors refresh's and are named here so a reader
#: can see what a cell means without opening that file.
MULTIAXIS_ELEVATION_HALF_RANGE = 0.5

#: The per-arm hard time cap.  An arm that exceeds it is killed and the timeout
#: is recorded as that arm's result.
ARM_TIMEOUT_S = 60.0 * float(os.environ.get("MG52_ARM_TIMEOUT_MIN", "45"))
#: The identity probes import a framework and print its versions.  They do no
#: real work, so a minute is generous even with a cold module cache.
PROBE_TIMEOUT_S = 600.0

# ── recorded context, not gates ───────────────────────────────────────────────
#: The mbirjax commit these arms are written against.  The jax model
#: constructors and the recon signature were read at this commit; the sbatch
#: asserts the cluster clone sits here before any jax arm runs, and the header
#: row records what the probe actually found.
PINNED_MBIRJAX_COMMIT = "7bb20093d635802ba8505c9366ff109ff6b35b76"
#: Cross-framework fingerprint differences above this print a NOTE.  Nothing
#: fails on it.  3-iteration differences for these geometries were measured
#: near 1e-3 relative during the port, so this is where "worth a second look"
#: starts, not where "wrong" starts.
FINGERPRINT_NOTE_LEVEL = 1e-3
#: Elements promoted to float64 at a time when a fingerprint is taken.  Eight
#: million float64 is 64 MiB, which bounds the reading's own memory at any
#: volume size.
FINGERPRINT_CHUNK_ELEMS = 1 << 23

#: Substrings that mark an arm's failure as a CAPACITY reading rather than a
#: harness fault.  A cell that does not fit one device is the outcome this run
#: went looking for at the largest multiaxis size, so it must not be reported
#: as a broken instrument.  Matched case-insensitively against the traceback.
CAPACITY_MARKERS = ("out of memory", "outofmemory", "resource_exhausted",
                    "resourceexhausted", "cuda error: out of memory",
                    "failed to allocate", "memoryerror", "cannot allocate")

RESULTS_DIR = os.environ.get(
    "MG52_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
STAGE_DIR = os.environ.get("MG52_STAGE_DIR", RESULTS_DIR)
TORCH_PYTHON = os.environ.get("MG52_TORCH_PYTHON", sys.executable)
JAX_PYTHON = os.environ.get("MG52_JAX_PYTHON", "")
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 24                  # wide enough for the longest arm id printed
# ──────────────────────────────────────────────────────────────────────────────


def cells():
    return SMOKE_CELLS if SMOKE else PRODUCTION_CELLS


def arm_id(spec, framework):
    return "{}_{}".format(spec["name"], framework)


def all_arm_ids():
    return [arm_id(spec, fw) for spec in cells() for fw in FRAMEWORKS]


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
    repeat before.  The error names the full valid list, because the caller who
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
            raise ValueError("{}: {!r} is not an arm of this run.  The valid "
                             "ids are: {}".format(env_name, token,
                                                  ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError("{}: no valid tokens in {!r}.  The valid ids are: {}"
                         .format(env_name, raw, ", ".join(allowed)))
    # Normalized to the DECLARED order: the run order is load-bearing (cheapest
    # cell first, torch before jax), so it must not depend on the order someone
    # typed the tokens in.
    return [name for name in allowed if name in chosen]


# ── the staged file ───────────────────────────────────────────────────────────
def stage_path(spec):
    """One npz per cell.  The geometry and the sinogram shape are both in the
    name, so a smoke run and a production run can share a directory without
    either reading the other's bytes."""
    return os.path.join(STAGE_DIR, "mg52_stage_{}_{}x{}x{}.npz".format(
        spec["geometry"], *spec["cell"]))


def md5_path(path):
    return path + ".md5"


def file_md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: the translation cell's npz is
    about eleven gigabytes and reading it whole to hash it would be wasteful."""
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


def load_staged(path, with_arrays=True):
    """Read one staged npz into a plain dict, after verifying its md5.

    Raises on a mismatch.  An arm that reconstructed different bytes than its
    sibling did not measure what the plan said, and a truncated read on a
    shared parallel filesystem is a recorded failure mode of this work.

    ``with_arrays=False`` reads everything EXCEPT the sinogram and the weights.
    An npz member is only read when it is asked for, and at the translation
    cell those two are about eleven gigabytes, so the staging step's
    already-on-disk path uses this to read the metadata without pulling them
    into memory for nothing.
    """
    expected = recorded_md5(path)
    actual = file_md5(path)
    if actual != expected:
        raise ValueError("the staged file at {} hashes to {}, not the "
                         "recorded {}".format(path, actual, expected))
    meta = read_staged(path, with_arrays=with_arrays)
    meta["md5"] = actual
    return meta


def read_staged(path, with_arrays=True):
    """The npz read itself, WITHOUT the md5 check.  Callers that have just
    hashed the file use this; everything else goes through ``load_staged``, so
    a file is never read twice to verify it once."""
    import numpy as np

    with np.load(path, allow_pickle=False) as handle:
        meta = dict(
            view_params=handle["view_params"],
            sinogram_shape=[int(v) for v in handle["sinogram_shape"]],
            recon_shape=[int(v) for v in handle["recon_shape"]],
            distances=[float(v) for v in handle["distances"]],
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


# ── the model, built the floors refresh's way ─────────────────────────────────
def translation_distances(cell):
    """Both source distances of a translation cell: half the smaller detector
    extent, which is what the floors refresh uses."""
    return float(min(cell[1], cell[2]) / 2.0), float(min(cell[1], cell[2]) / 2.0)


def view_params_for(spec):
    """One cell's per-view parameters, built the way the floors refresh builds
    them.  Used only by staging; the arms read the staged copy instead, so both
    frameworks are handed the same numbers."""
    import numpy as np

    import mbirtorch

    geometry, cell = spec["geometry"], tuple(spec["cell"])
    num_views = cell[0]
    if geometry == "multiaxis":
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-MULTIAXIS_ELEVATION_HALF_RANGE,
                                MULTIAXIS_ELEVATION_HALF_RANGE, num_views)
        return np.stack([azimuth, elevation], axis=1)
    if geometry == "translation":
        num_x, num_z = spec["grid"]
        x_spacing, z_spacing = spec["spacing"]
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != num_views:
            raise ValueError(
                "translation cell {} has {} views, but its {}x{} grid gives {} "
                "translations".format(cell, num_views, num_x, num_z,
                                      vectors.shape[0]))
        return vectors
    # Falling through to another geometry would time that geometry and record
    # the result under this one's name, which is the one way a comparison can
    # be wrong without anything looking wrong.
    raise ValueError("mg52 has no view parameters for geometry {!r}"
                     .format(geometry))


def build_torch_model(geometry, sinogram_shape, view_params, distances):
    """The mbirtorch model, from the staged constructor parameters.

    On CUDA nothing is configured here: the count comes from
    MBIRTORCH_NUM_DEVICES, which keeps the model on the automatic branch where
    the memory preflight still runs.  The pin is a CUDA mechanism -- the policy
    short-circuits below two visible CUDA devices -- so the CPU smoke places
    its one device by hand, exactly as the floors refresh does.
    """
    import mbirtorch

    cell = tuple(int(v) for v in sinogram_shape)
    if geometry == "multiaxis":
        model = mbirtorch.MultiAxisParallelModel(cell, view_params)
    elif geometry == "translation":
        model = mbirtorch.TranslationModel(
            cell, view_params, source_detector_dist=distances[0],
            source_iso_dist=distances[1])
    else:
        raise ValueError("mg52 has no mbirtorch model for geometry {!r}"
                         .format(geometry))
    if DEVICE != "cuda":
        model.configure_devices(devices=[DEVICE])
    model.set_params(no_warning=True, verbose=0)
    return model


def build_jax_model(geometry, sinogram_shape, view_params, distances):
    """The mbirjax model, from the SAME staged constructor parameters.

    ``configure_devices(1)`` is mbirjax's documented way to force
    single-device operation; without it the automatic policy shards across
    every visible device, including the virtual CPU devices the CPU smoke
    would otherwise see.  set_params re-runs the device selection but keeps an
    explicit pin, so the order below is safe.
    """
    import mbirjax as mj

    cell = tuple(int(v) for v in sinogram_shape)
    if geometry == "multiaxis":
        model = mj.MultiAxisParallelModel(cell, view_params)
    elif geometry == "translation":
        model = mj.TranslationModel(
            cell, view_params, source_detector_dist=distances[0],
            source_iso_dist=distances[1])
    else:
        raise ValueError("mg52 has no mbirjax model for geometry {!r}"
                         .format(geometry))
    model.configure_devices(1)
    model.set_params(no_warning=True, verbose=0)
    return model


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


def fingerprint(volume):
    """Two float64 reductions of a reconstruction: the sum of absolute values
    and the sum of squares.

    Two numbers rather than one, because a sum of absolute values alone cannot
    see a rearrangement that preserves magnitudes.  Taken on the HOST in numpy
    for both frameworks -- both ``recon`` calls return host arrays -- so the
    reading compares the volumes and not the two libraries' reduction orders.
    Accumulated in fixed-size chunks: a float32 sum over a billion-element
    volume loses the digits this comparison needs, and promoting the whole
    volume at once would double what the reading costs in memory.
    """
    import numpy as np

    flat = np.ascontiguousarray(volume).reshape(-1)
    abs_sum = 0.0
    sq_sum = 0.0
    for start in range(0, flat.shape[0], FINGERPRINT_CHUNK_ELEMS):
        block = np.asarray(flat[start:start + FINGERPRINT_CHUNK_ELEMS],
                           dtype=np.float64)
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


def git_identity(path):
    """The commit a checkout sits at, and whether it is dirty.  ``None`` when
    the directory is not a git checkout or git is unavailable, which is a
    recorded state rather than an error: an installed wheel has no commit."""
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


# ── the workers: one identity probe, one staging job, or one arm ──────────────
def run_identity(cfg):
    """Which libraries this run is about to measure, and on what.

    Run before anything else and in its own process, so the header row can name
    both frameworks' versions, the device, and both checkouts' commits without
    the driver ever importing either framework.
    """
    row = dict(cfg)
    if cfg["framework"] == "torch":
        import torch

        import mbirtorch

        cuda = DEVICE == "cuda" and torch.cuda.is_available()
        row.update(version="torch {}".format(torch.__version__),
                   library_file=mbirtorch.__file__,
                   device_count=(torch.cuda.device_count() if cuda else 1),
                   device_name=(torch.cuda.get_device_name(0) if cuda
                                else DEVICE),
                   cuda=cuda)
        package_root = os.path.dirname(os.path.dirname(
            os.path.abspath(mbirtorch.__file__)))
        row["git"] = git_identity(package_root)
        # The premise of the whole run, asserted rather than assumed: neither
        # geometry binds a hand-written kernel, so both projection directions
        # run as general torch code.  Read off tiny models, which allocate
        # nothing and need no device.
        try:
            from mbirtorch import _memory_ledger
            probe = {}
            for spec in SMOKE_CELLS:
                geometry = spec["geometry"]
                model = build_torch_model(
                    geometry, spec["cell"], view_params_for(spec),
                    translation_distances(tuple(spec["cell"])))
                probe[geometry] = list(
                    _memory_ledger.torch_body_directions(model))
            row["torch_body_directions"] = probe
            row["no_kernels_confirmed"] = all(
                set(v) == {"forward", "back"} for v in probe.values())
        except Exception as exc:                                  # noqa: BLE001
            row["torch_body_directions"] = None
            row["no_kernels_confirmed"] = None
            row["body_probe_error"] = "{}: {}".format(type(exc).__name__, exc)
    else:
        # mbirjax before jax, deliberately: mbirjax's device setup runs at
        # import and warns that it can no longer choose the device set when
        # jax has already initialised.
        import mbirjax as mj

        import jax

        devices = jax.devices()
        row.update(version="jax {}".format(jax.__version__),
                   library_file=mj.__file__,
                   device_count=len(devices),
                   device_name=str(devices[0]) if devices else "none",
                   device_platform=(devices[0].platform if devices else None),
                   preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
        package_root = os.path.dirname(os.path.dirname(
            os.path.abspath(mj.__file__)))
        row["git"] = git_identity(package_root)
        row["pinned_commit"] = PINNED_MBIRJAX_COMMIT
        row["at_pinned_commit"] = (row["git"].get("commit")
                                   == PINNED_MBIRJAX_COMMIT)
    return row


def run_stage(cfg):
    """Build ONE cell's sinogram and weights, once, with mbirtorch, at one
    device, and write them to one npz with an md5 sidecar.

    Both arms of the cell then load that same file, so the two frameworks
    reconstruct identical input and the comparison between them is controlled
    rather than incidental.  The npz also carries every constructor parameter
    the matched jax model needs and the recon shape mbirtorch realized, so an
    arm never has to re-derive a model parameter from the cell shape.
    """
    import numpy as np

    import mbirtorch

    spec = cfg["spec"]
    row = dict(cfg)
    path = stage_path(spec)
    row["stage_path"] = path

    if staged_present(path):
        # Already on disk.  VERIFY it rather than rebuild it: a regenerated
        # sinogram is not bit-identical, so a rebuild would silently change
        # what both arms of this cell reconstruct.
        expected = recorded_md5(path)
        actual = file_md5(path)
        row.update(reused=True, md5=actual, md5_ok=(actual == expected),
                   recorded_md5=expected)
        if actual == expected:
            # Hashed once, just above; read the metadata only -- the sinogram
            # and the weights are read by the arms, not here.
            meta = read_staged(path, with_arrays=False)
            row.update(recon_shape=meta["recon_shape"],
                       sinogram_shape=meta["sinogram_shape"],
                       delta_voxel=meta["delta_voxel"],
                       voxel_row_aspect=meta["voxel_row_aspect"],
                       voxel_slice_aspect=meta["voxel_slice_aspect"],
                       psf_radius=meta["psf_radius"],
                       phantom_fallback=meta["phantom_fallback"],
                       bytes_on_disk=os.path.getsize(path))
        else:
            row["invalid_reasons"] = [
                "the staged file at {} hashes to {}, not the recorded {}"
                .format(path, actual, expected)]
        return row

    geometry, cell = spec["geometry"], tuple(spec["cell"])
    view_params = np.asarray(view_params_for(spec), dtype=np.float32)
    distances = translation_distances(cell)
    model = build_torch_model(geometry, cell, view_params, distances)
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    # The shepp-logan builder places its ellipsoids as fractions of the volume,
    # and on a volume only a few voxels deep every one of them can miss,
    # leaving the phantom all zeros.  An all-zero phantom forward projects to
    # an all-zero sinogram, so both arms would time a reconstruction of
    # nothing.  This is the floors refresh's fallback, and the row records that
    # it was used.
    phantom_fallback = ""
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(VCD_SEED).rand(*recon_shape),
                             dtype=np.float32)
        phantom_fallback = "seeded uniform (shepp-logan returned all zeros)"

    sinogram = np.ascontiguousarray(
        np.asarray(to_numpy(model.forward_project(phantom)), dtype=np.float32))
    # The floors refresh's weighting formula.  copy=False only skips a repeat
    # of an array that is already float32; the values are the same ones that
    # expression has always produced, and at the translation cell each avoided
    # copy is five gigabytes of host memory.
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32,
                                                                copy=False)

    os.makedirs(STAGE_DIR, exist_ok=True)
    np.savez(path, sinogram=sinogram, weights=weights,
             view_params=view_params,
             sinogram_shape=np.asarray(cell, dtype=np.int64),
             recon_shape=np.asarray(recon_shape, dtype=np.int64),
             distances=np.asarray(
                 distances if geometry == "translation" else (),
                 dtype=np.float64),
             geometry=np.asarray(geometry), name=np.asarray(spec["name"]),
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
    row.update(reused=False, md5=digest, md5_ok=True,
               recon_shape=list(recon_shape), sinogram_shape=list(cell),
               delta_voxel=float(model.get_params("delta_voxel")),
               voxel_row_aspect=float(model.get_params("voxel_row_aspect")),
               voxel_slice_aspect=float(model.get_params("voxel_slice_aspect")),
               psf_radius=int(model.get_psf_radius()),
               phantom_fallback=phantom_fallback,
               phantom_max=float(np.max(phantom)),
               # Chunked, like every other reading of a whole volume here: the
               # one-expression form would hold a second copy of a five
               # gigabyte sinogram just to sum it.
               sinogram_abs_sum=fingerprint(sinogram)[0],
               bytes_on_disk=os.path.getsize(path),
               stage_devices=[str(d) for d in model.sino_placement.devices])
    return row


def _arm_common(cfg):
    """The staged input, the model, and the recon-shape check every arm makes,
    whichever framework it is."""
    spec = cfg["spec"]
    path = stage_path(spec)
    meta = load_staged(path)              # raises on an md5 mismatch
    row = dict(cfg, stage_path=path, staged_md5=meta["md5"],
               staged_recon_shape=meta["recon_shape"],
               staged_delta_voxel=meta["delta_voxel"],
               staged_voxel_row_aspect=meta["voxel_row_aspect"],
               staged_voxel_slice_aspect=meta["voxel_slice_aspect"],
               staged_psf_radius=meta["psf_radius"],
               phantom_fallback=meta["phantom_fallback"],
               vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
               warm_repeats=WARM_REPEATS, invalid_reasons=[])
    return meta, row


def _check_recon_shape(model, meta, row):
    """The matched-model check.  Two libraries reconstructing different volumes
    are not comparable, so a mismatch stops the arm rather than producing a row
    that looks comparable and is not."""
    realized = [int(s) for s in model.get_params("recon_shape")]
    row["recon_shape"] = realized
    row["recon_shape_ok"] = (realized == list(meta["recon_shape"]))
    if not row["recon_shape_ok"]:
        row["invalid_reasons"].append(
            "this framework's model realized recon shape {}, but the staged "
            "cell was built at {}".format(realized, meta["recon_shape"]))
    # The derived voxel geometry: recorded on both sides and compared in the
    # report.  A difference here is printed, never gated -- the two libraries
    # compute the same formula in different float widths.
    for name in ("delta_voxel", "voxel_row_aspect", "voxel_slice_aspect"):
        row[name] = float(model.get_params(name))
    row["psf_radius"] = int(model.get_psf_radius())
    return row["recon_shape_ok"]


def run_torch_arm(cfg):
    """One mbirtorch arm: a cold pass, then WARM_REPEATS timed warm passes."""
    import numpy as np
    import torch

    meta, row = _arm_common(cfg)
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    row.update(version="torch {}".format(torch.__version__), cuda=cuda,
               device=DEVICE,
               device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
               visible_devices=(torch.cuda.device_count() if cuda else 1),
               pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda
                              else "configure_devices(devices=['cpu'])"),
               env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
               env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))

    model = build_torch_model(meta["geometry"], meta["sinogram_shape"],
                              meta["view_params"], meta["distances"])
    if not _check_recon_shape(model, meta, row):
        return row

    sinogram, weights = meta["sinogram"], meta["weights"]

    def one_recon():
        """One reconstruction, exactly as the floors refresh calls it."""
        np.random.seed(VCD_SEED)
        out, _info = model.recon(sinogram, weights=weights,
                                 max_iterations=VCD_ITERATIONS,
                                 stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return to_numpy(out)

    if cuda:
        for device in model.recon_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    out = one_recon()
    row["cold_s"] = time.perf_counter() - start

    # The layout has settled, so the projector reading describes the layout the
    # timed passes actually run on.
    devices_now = list(model.recon_placement.devices)
    row["realized_devices"] = [str(d) for d in devices_now]
    row["realized_n_devices"] = len(devices_now)
    try:
        from mbirtorch import _memory_ledger
        row["torch_body_directions"] = list(
            _memory_ledger.torch_body_directions(model))
    except Exception as exc:                                      # noqa: BLE001
        row["torch_body_directions"] = None
        row["body_read_error"] = "{}: {}".format(type(exc).__name__, exc)

    # The cold peak is read BEFORE the counters are reset, so the row can carry
    # both the cold and the warm reading and the comparison can use the larger
    # of the two -- the process peak, which is what the jax reading is.
    if cuda:
        row["cold_peak_bytes"] = max(int(torch.cuda.max_memory_allocated(d))
                                     for d in devices_now)
        for device in devices_now:
            torch.cuda.reset_peak_memory_stats(device)
    else:
        row["cold_peak_bytes"] = None

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

    if cuda:
        row["warm_peak_bytes"] = max(int(torch.cuda.max_memory_allocated(d))
                                     for d in devices_now)
        row["peak_bytes"] = max(row["warm_peak_bytes"], row["cold_peak_bytes"])
    else:
        row["warm_peak_bytes"] = None
        row["peak_bytes"] = None
    row["peak_kind"] = "torch.cuda.max_memory_allocated, process lifetime"

    abs_sum, sq_sum = fingerprint(out)
    row["fingerprint_abs_sum"] = abs_sum
    row["fingerprint_sq_sum"] = sq_sum
    row["fingerprint_elements"] = int(np.asarray(out).size)
    return row


def _jax_peak_bytes(jax_module):
    """The device's peak allocator use, and its limit.

    jax exposes a peak but no way to reset it, so this covers the whole
    process.  The CPU backend does not implement memory_stats at all, which is
    a recorded state rather than an error.
    """
    try:
        device = jax_module.devices()[0]
        stats = device.memory_stats() or {}
    except Exception:                                             # noqa: BLE001
        return None, None
    peak = stats.get("peak_bytes_in_use")
    limit = stats.get("bytes_limit")
    return (int(peak) if peak is not None else None,
            int(limit) if limit is not None else None)


def run_jax_arm(cfg):
    """One mbirjax arm, on the same staged bytes and the same protocol."""
    # mbirjax before jax: its device setup runs at import and warns that it
    # can no longer choose the device set when jax has already initialised.
    import mbirjax                                                # noqa: F401

    import jax
    import numpy as np

    meta, row = _arm_common(cfg)
    devices = jax.devices()
    row.update(version="jax {}".format(jax.__version__), device=DEVICE,
               device_name=str(devices[0]) if devices else "none",
               device_platform=(devices[0].platform if devices else None),
               visible_devices=len(devices),
               pin_mechanism="configure_devices(1)",
               preallocate=os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    if DEVICE == "cuda" and (not devices or devices[0].platform != "gpu"):
        row["invalid_reasons"].append(
            "jax sees no GPU device: platform is {!r}".format(
                devices[0].platform if devices else None))
        return row

    model = build_jax_model(meta["geometry"], meta["sinogram_shape"],
                            meta["view_params"], meta["distances"])
    if not _check_recon_shape(model, meta, row):
        return row
    placement_devices = getattr(getattr(model, "recon_placement", None),
                                "devices", None)
    row["realized_devices"] = ([str(d) for d in placement_devices]
                               if placement_devices else None)
    row["realized_n_devices"] = (len(placement_devices)
                                 if placement_devices else None)

    sinogram, weights = meta["sinogram"], meta["weights"]

    def one_recon():
        """The same call the torch arm makes, with the same seed, the same
        iteration count and the same disabled stopping rule."""
        np.random.seed(VCD_SEED)
        out, _info = model.recon(sinogram, weights=weights,
                                 max_iterations=VCD_ITERATIONS,
                                 stop_threshold_change_pct=0.0)
        # recon gathers to a host array, so this is already finished; the
        # block is here so the clock does not depend on that staying true.
        return np.asarray(jax.block_until_ready(out))

    start = time.perf_counter()
    out = one_recon()
    row["cold_s"] = time.perf_counter() - start
    cold_peak, _limit = _jax_peak_bytes(jax)
    row["cold_peak_bytes"] = cold_peak

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

    peak, limit = _jax_peak_bytes(jax)
    row["warm_peak_bytes"] = peak
    row["peak_bytes"] = peak
    row["device_bytes_limit"] = limit
    row["peak_kind"] = ("jax device memory_stats peak_bytes_in_use, process "
                        "lifetime (jax exposes no reset)")

    abs_sum, sq_sum = fingerprint(out)
    row["fingerprint_abs_sum"] = abs_sum
    row["fingerprint_sq_sum"] = sq_sum
    row["fingerprint_elements"] = int(np.asarray(out).size)
    return row


def run_job(cfg):
    started = time.time()
    if cfg["kind"] == "identity":
        row = run_identity(cfg)
    elif cfg["kind"] == "stage":
        row = run_stage(cfg)
    elif cfg["framework"] == "torch":
        row = run_torch_arm(cfg)
    else:
        row = run_jax_arm(cfg)
    row["worker_wall_s"] = time.time() - started
    return row


# ── the driver ────────────────────────────────────────────────────────────────
def job_python(cfg):
    """The interpreter one job runs in.  The two frameworks live in different
    environments, which is the whole reason a job is a subprocess here."""
    return TORCH_PYTHON if cfg.get("framework", "torch") == "torch" \
        else JAX_PYTHON


def job_env(cfg):
    """The environment that DEFINES a job, set explicitly so nothing leaks in
    from the submitting shell or across frameworks.

    PYTHONPATH is popped and then set from the framework's own variable.  A
    PYTHONPATH that reached the wrong job would select the wrong library, in
    exactly the silent direction that would make this run compare a library
    against itself.
    """
    env = dict(os.environ)
    env.pop("MG52_DRY", None)          # a worker never prints a plan
    env.pop("PYTHONPATH", None)
    framework = cfg.get("framework", "torch")
    extra = os.environ.get("MG52_TORCH_PYTHONPATH" if framework == "torch"
                           else "MG52_JAX_PYTHONPATH", "")
    if extra:
        env["PYTHONPATH"] = extra
    if framework == "torch":
        env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)  # it owns the peak counter
        env.pop("MBIRTORCH_WIDENING_GUARD", None)      # the pin bypasses it
        env["MBIRTORCH_DISABLE_TRITON"] = "0"          # the shipped setting
        env.pop("MBIRTORCH_NUM_DEVICES", None)
        if DEVICE == "cuda":
            env["MBIRTORCH_NUM_DEVICES"] = "1"
    return env


def spawn(cfg, timeout_s):
    """Run one job in a NEW interpreter, with a hard time cap.

    A new process per job is not tidiness.  Compiled bodies and allocator state
    are cached for the life of a process, and the two frameworks do not live in
    the same environment at all.  The row travels through a file rather than
    through stdout, so the worker's own output streams into the job log while
    it runs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_mg52_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR, "_mg52_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    python = job_python(cfg)
    if not python:
        return dict(cfg, error="no interpreter configured for framework {!r}: "
                              "set MG52_JAX_PYTHON".format(cfg.get("framework")))
    start = time.perf_counter()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run([python, "-u", os.path.abspath(__file__),
                               "--worker", cfg_path, out_path],
                              env=job_env(cfg), timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # The cap is a READING, not a fault: an arm that cannot finish inside
        # it has told us something about the framework at this size.
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
    harness fault.  A cell that does not fit one device is what this run went
    looking for at the largest size, so it must not be reported as a broken
    instrument."""
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


def build_plan():
    """Every job, in run order: both identity probes, then every cell's staging,
    then each cell's two arms.

    Staging runs BEFORE the header row is written, because the header records
    the staged files' md5 sums.  It is cheap to repeat: a staged file whose md5
    matches is reused, so a re-run stages nothing.
    """
    keep = _strict_subset("MG52_ARMS", all_arm_ids())
    probes = [dict(kind="identity", framework=fw, job_id="identity_" + fw)
              for fw in FRAMEWORKS]
    stages, arms = [], []
    for spec in cells():
        wanted = [fw for fw in FRAMEWORKS if arm_id(spec, fw) in keep]
        if not wanted:
            continue
        stages.append(dict(kind="stage", framework="torch", spec=spec,
                           name=spec["name"],
                           job_id="stage_" + spec["name"]))
        for fw in wanted:
            arms.append(dict(kind="arm", framework=fw, spec=spec,
                             name=spec["name"], arm=arm_id(spec, fw),
                             job_id=arm_id(spec, fw)))
    if not arms:
        raise ValueError("MG52_ARMS selects no arm")
    return probes, stages, arms


def staged_gib(spec):
    """What one cell's npz costs on disk: the sinogram and the weights, both
    float32, at the sinogram shape."""
    num_views, num_rows, num_channels = spec["cell"]
    return 2.0 * num_views * num_rows * num_channels * 4 / 2 ** 30


def print_plan(probes, stages, arms):
    print("mg52 mbirtorch against mbirjax on multiaxis and translation: "
          "{} arm(s) over {} cell(s), device {}, {} VCD iteration(s), "
          "{} warm repeat(s) after a cold pass"
          .format(len(arms), len(stages), DEVICE, VCD_ITERATIONS,
                  WARM_REPEATS))
    print("  rows          -> {}".format(RESULTS_DIR))
    print("  staged inputs -> {}".format(STAGE_DIR))
    print("  torch interpreter: {}".format(TORCH_PYTHON))
    print("  jax interpreter:   {}".format(JAX_PYTHON or
                                           "NOT SET (MG52_JAX_PYTHON)"))
    print("  torch PYTHONPATH:  {}".format(
        os.environ.get("MG52_TORCH_PYTHONPATH") or "(none)"))
    print("  jax PYTHONPATH:    {}".format(
        os.environ.get("MG52_JAX_PYTHONPATH") or "(none)"))
    print("  mbirjax is expected at commit {}".format(PINNED_MBIRJAX_COMMIT))
    print("  per-arm hard time cap {:.0f} min; an arm that exceeds it, or runs "
          "out of device memory, is RECORDED and the run continues"
          .format(ARM_TIMEOUT_S / 60.0))
    print("  fingerprints are recorded, not gated; a gap above {:.0e} relative "
          "prints a note".format(FINGERPRINT_NOTE_LEVEL))
    total = sum(staged_gib(cfg["spec"]) for cfg in stages)
    print("  staged npz files total about {:.1f} GiB and are KEPT for re-runs"
          .format(total))
    header = ("  {:<{w}}{:>10}{:>22}{:>13}  what it does"
              .format("job", "framework", "cell", "staged GiB", w=ARM_COL))
    print(header)
    what = dict(identity="names the library, the version and the device",
                stage="builds this cell's sinogram and weights once, "
                      "with mbirtorch",
                arm="cold pass, then the timed warm passes")
    for cfg in list(probes) + list(stages) + list(arms):
        spec = cfg.get("spec")
        cell = str(tuple(spec["cell"])) if spec else "-"
        gib = ("{:.1f}".format(staged_gib(spec))
               if spec and cfg["kind"] == "stage" else "-")
        print("  {:<{w}}{:>10}{:>22}{:>13}  {}".format(
            cfg["job_id"], cfg["framework"], cell, gib, what[cfg["kind"]],
            w=ARM_COL))
    print("  no library file is edited and no default is flipped: both "
          "frameworks run their shipped configuration")


def main():
    probes, stages, arms = build_plan()
    if DRY:
        print_plan(probes, stages, arms)
        return 0
    if not JAX_PYTHON:
        raise SystemExit(
            "MG52_JAX_PYTHON is not set, so the jax arms have no interpreter "
            "to run in.  Point it at the overlay venv's python (the sbatch "
            "builds one) and re-run.  MG52_DRY=1 prints the plan without it.")
    print_plan(probes, stages, arms)

    # ── the identity probes and the staging, both before the header row ──────
    print("\n-- identity probes --", flush=True)
    identity = {}
    for cfg in probes:
        row = spawn(cfg, PROBE_TIMEOUT_S)
        identity[cfg["framework"]] = row
        print("  {}: {}".format(cfg["framework"],
                                row.get("error") or
                                "{} | {} | {} device(s) | {}".format(
                                    row.get("version"), row.get("device_name"),
                                    row.get("device_count"),
                                    row.get("library_file"))), flush=True)

    print("\n-- staging (mbirtorch; reused when the md5 matches) --", flush=True)
    stage_rows = []
    for index, cfg in enumerate(stages):
        print("  [{}/{}] {}".format(index + 1, len(stages), cfg["job_id"]),
              flush=True)
        row = spawn(cfg, ARM_TIMEOUT_S)
        stage_rows.append(row)
        if row.get("error"):
            print("    ERROR: {}".format(str(row["error"])[:400]), flush=True)
        else:
            print("    md5 {} {} recon {}".format(
                row.get("md5"), "(reused)" if row.get("reused") else "(built)",
                row.get("recon_shape")), flush=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        "mg52_framework_anchor_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    arm_rows = {}
    comparisons = []
    with open(out_path, "w") as sink:
        header = dict(
            row="run_header", script="mg52_framework_anchor.py",
            node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
            driver_python=sys.executable, torch_python=TORCH_PYTHON,
            jax_python=JAX_PYTHON, results_dir=RESULTS_DIR,
            stage_dir=STAGE_DIR,
            torch_identity=identity.get("torch"),
            jax_identity=identity.get("jax"),
            pinned_mbirjax_commit=PINNED_MBIRJAX_COMMIT,
            staged_md5={row.get("name"): row.get("md5") for row in stage_rows},
            cells=[dict(spec) for spec in cells()],
            vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
            warm_repeats=WARM_REPEATS,
            arm_timeout_s=ARM_TIMEOUT_S,
            fingerprint_note_level=FINGERPRINT_NOTE_LEVEL,
            plan=[dict(cfg) for cfg in list(probes) + list(stages) + list(arms)])
        sink.write(json.dumps(header) + "\n")
        sink.flush()
        for row in stage_rows:
            sink.write(json.dumps(dict(row="stage", **row)) + "\n")
        sink.flush()

        for index, cfg in enumerate(arms):
            print("\n  [{}/{}] {}".format(index + 1, len(arms), cfg["job_id"]),
                  flush=True)
            row = spawn(cfg, ARM_TIMEOUT_S)
            arm_rows[cfg["arm"]] = row
            sink.write(json.dumps(dict(row="arm", **row)) + "\n")
            sink.flush()
            if row.get("error"):
                print("    {}: {}".format(
                    "READING" if is_capacity_reading(row) else "ERROR",
                    str(row["error"]).strip().splitlines()[-1][:200]),
                    flush=True)
            else:
                print("    cold {:.2f}s  warm {:.2f}s  spread {:.1%}  "
                      "peak {}".format(
                          row.get("cold_s", 0.0), row.get("warm_s", 0.0),
                          row.get("warm_spread", 0.0),
                          "-" if row.get("peak_bytes") is None
                          else "{:.2f} GB".format(
                              row["peak_bytes"] / 2 ** 30)), flush=True)
            # The comparison row goes out as soon as BOTH arms of a cell are
            # in, so a job cut short still carries every cell it finished.
            spec = cfg["spec"]
            if all(arm_id(spec, fw) in arm_rows for fw in FRAMEWORKS):
                comparison = compare_cell(spec, arm_rows)
                comparisons.append(comparison)
                sink.write(json.dumps(dict(row="comparison", **comparison))
                           + "\n")
                sink.flush()
                print_comparison(comparison)

        summary = summarize(identity, stage_rows, arm_rows, comparisons, arms,
                            out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        sink.write(json.dumps(dict(row="summary", **summary)) + "\n")
        sink.flush()
    print("\nwrote {}".format(out_path))
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


# ── the comparison and the report ─────────────────────────────────────────────
def compare_cell(spec, arm_rows):
    """One cell's two arms, side by side: the warm-median ratio, the peaks, and
    the relative fingerprint differences.

    mbirtorch is the reference in every ratio, because mbirtorch staged the
    input both arms read.  A ratio above 1 means mbirjax took longer.
    """
    torch_row = arm_rows.get(arm_id(spec, "torch"), {})
    jax_row = arm_rows.get(arm_id(spec, "jax"), {})
    out = dict(name=spec["name"], geometry=spec["geometry"],
               cell=list(spec["cell"]))
    for label, row in (("torch", torch_row), ("jax", jax_row)):
        status = ("ok" if not row.get("error") and row.get("warm_s") is not None
                  else ("timeout" if row.get("timed_out")
                        else ("capacity" if is_capacity_reading(row)
                              else "error")))
        out[label + "_status"] = status
        out[label + "_cold_s"] = row.get("cold_s")
        out[label + "_warm_s"] = row.get("warm_s")
        out[label + "_warm_spread"] = row.get("warm_spread")
        out[label + "_peak_bytes"] = row.get("peak_bytes")
        out[label + "_recon_shape"] = row.get("recon_shape")
        out[label + "_delta_voxel"] = row.get("delta_voxel")
        out[label + "_voxel_row_aspect"] = row.get("voxel_row_aspect")
        out[label + "_abs_sum"] = row.get("fingerprint_abs_sum")
        out[label + "_sq_sum"] = row.get("fingerprint_sq_sum")
        if row.get("error"):
            out[label + "_error"] = str(row["error"]).strip().splitlines()[-1][:300]

    tw, jw = out["torch_warm_s"], out["jax_warm_s"]
    out["warm_ratio_jax_over_torch"] = (jw / tw if tw and jw else None)
    out["torch_speedup_over_jax"] = (tw / jw if tw and jw else None)
    tp, jp = out["torch_peak_bytes"], out["jax_peak_bytes"]
    out["peak_ratio_jax_over_torch"] = (jp / tp if tp and jp else None)
    out["abs_sum_rel_gap"] = relative_gap(out["jax_abs_sum"],
                                          out["torch_abs_sum"])
    out["sq_sum_rel_gap"] = relative_gap(out["jax_sq_sum"], out["torch_sq_sum"])
    gaps = [g for g in (out["abs_sum_rel_gap"], out["sq_sum_rel_gap"])
            if g is not None]
    out["fingerprint_note"] = (
        "" if not gaps or max(gaps) <= FINGERPRINT_NOTE_LEVEL else
        "the two frameworks' volumes differ by {:.2e} relative, above the "
        "{:.0e} that 3-iteration cross-framework differences were measured at "
        "during the port -- recorded, not a failure".format(
            max(gaps), FINGERPRINT_NOTE_LEVEL))
    out["voxel_geometry_note"] = ""
    for field in ("delta_voxel", "voxel_row_aspect"):
        gap = relative_gap(out["jax_" + field], out["torch_" + field])
        if gap is not None and gap > 1e-5:
            out["voxel_geometry_note"] += (
                "{} differs between the frameworks by {:.2e} relative "
                "({} against {}).  ".format(field, gap, out["jax_" + field],
                                            out["torch_" + field]))
    return out


def _fmt(value, width=10, kind="f", prec=2):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    return "{:>{w}.{p}{k}}".format(value, w=width, p=prec, k=kind)


def print_comparison(comparison):
    print("    comparison {}: torch {} / jax {}".format(
        comparison["name"], comparison["torch_status"],
        comparison["jax_status"]))
    ratio = comparison["warm_ratio_jax_over_torch"]
    print("      warm medians  torch {}  jax {}   jax/torch {}".format(
        _fmt(comparison["torch_warm_s"], 8), _fmt(comparison["jax_warm_s"], 8),
        _fmt(ratio, 6) if ratio else "     -"))
    print("      fingerprints  abs gap {}  sq gap {}".format(
        _fmt(comparison["abs_sum_rel_gap"], 10, "e", 2),
        _fmt(comparison["sq_sum_rel_gap"], 10, "e", 2)))
    if comparison["fingerprint_note"]:
        print("      NOTE {}".format(comparison["fingerprint_note"]))
    if comparison["voxel_geometry_note"]:
        print("      NOTE {}".format(comparison["voxel_geometry_note"]))


def print_table(comparisons):
    """The table a person reads: one row per cell, both frameworks side by
    side.  Times are warm medians of three seeded 3-iteration reconstructions
    on one device; peaks are each process's own peak device allocation."""
    print("\n### mbirtorch against mbirjax, one device, warm median of {} "
          "seeded {}-iteration reconstructions".format(WARM_REPEATS,
                                                       VCD_ITERATIONS))
    print("| cell | geometry | mbirtorch | mbirjax | jax/torch | torch peak "
          "| jax peak | abs gap | sq gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for item in comparisons:
        def seconds(label):
            value = item[label + "_warm_s"]
            return ("{:.2f} s".format(value) if value is not None
                    else item[label + "_status"])

        def peak(label):
            value = item[label + "_peak_bytes"]
            return ("{:.2f} GB".format(value / 2 ** 30)
                    if value is not None else "-")

        ratio = item["warm_ratio_jax_over_torch"]
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            tuple(item["cell"]), item["geometry"], seconds("torch"),
            seconds("jax"),
            "{:.2f}x".format(ratio) if ratio else "-",
            peak("torch"), peak("jax"),
            "{:.2e}".format(item["abs_sum_rel_gap"])
            if item["abs_sum_rel_gap"] is not None else "-",
            "{:.2e}".format(item["sq_sum_rel_gap"])
            if item["sq_sum_rel_gap"] is not None else "-"))
    print("A ratio above 1 means mbirjax took longer.  Peaks are each "
          "framework's own process-lifetime peak device allocation, which the "
          "two libraries report through different counters; the jax arms also "
          "record the allocator's byte limit, because jax preallocates a "
          "fraction of the device.")


def summarize(identity, stage_rows, arm_rows, comparisons, arms, out_path):
    """The table a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A slow
    arm, a wide spread, an arm that ran out of memory and an arm that hit the
    time cap are FINDINGS: they are printed and none of them touches the exit
    code.  A missing row, an md5 that did not verify, a recon shape that did
    not match the staged one, and an error that is not a capacity or timeout
    reading are instrument failures, because they mean the run did not measure
    what the plan said it would.
    """
    print("\n===== mg52 mbirtorch against mbirjax ({}) =====".format(out_path))
    broken, findings = [], []

    header = ("{:<{w}}{:>9}{:>10}{:>9}{:>10}{:>10}{:>12}"
              .format("arm", "cold s", "warm s", "spread", "peak GB",
                      "devices", "state", w=ARM_COL))
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
        print("{:<{w}}{}{}{:>9}{}{:>10}{:>12}".format(
            name, _fmt(row.get("cold_s"), 9), _fmt(row.get("warm_s"), 10),
            "-" if row.get("warm_spread") is None
            else "{:.1%}".format(row["warm_spread"]),
            _fmt(None if row.get("peak_bytes") is None
                 else row["peak_bytes"] / 2 ** 30, 10),
            str(row.get("realized_n_devices", "-")),
            "ok" if not row.get("invalid_reasons") else "CHECK FAIL",
            w=ARM_COL))
        for reason in row.get("invalid_reasons") or []:
            print("    ARM CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(name, reason))
        if row.get("realized_n_devices") not in (None, 1):
            findings.append("{}: realized {} devices, not one".format(
                name, row.get("realized_n_devices")))

    for row in stage_rows:
        print("\nstaged {} {}: md5 {}{}".format(
            row.get("name"), tuple(row.get("sinogram_shape") or ()),
            row.get("md5", "-"),
            "  (reused from disk)" if row.get("reused") else ""))
        if row.get("phantom_fallback"):
            print("  phantom: {}".format(row["phantom_fallback"]))
        if row.get("error"):
            broken.append("{}|{}".format(row.get("job_id"),
                                         str(row["error"])[:200]))
        for reason in row.get("invalid_reasons") or []:
            broken.append("{}|{}".format(row.get("job_id"), reason))

    print_table(comparisons)

    print("\n-- what each library was --")
    for framework in FRAMEWORKS:
        row = identity.get(framework) or {}
        commit = (row.get("git") or {}).get("commit")
        print("  {:<6} {} | {} | commit {}{}".format(
            framework, row.get("version", "?"),
            row.get("library_file", "?"), commit or "unknown",
            " (dirty)" if (row.get("git") or {}).get("dirty") else ""))
        if row.get("error"):
            print("    PROBE FAILED: {}".format(str(row["error"])[-300:]))
            broken.append("identity_{}|{}".format(
                framework,
                str(row["error"]).strip().splitlines()[-1][:200]))
    torch_probe = identity.get("torch") or {}
    if torch_probe.get("no_kernels_confirmed") is True:
        print("  premise confirmed: both geometries run BOTH projection "
              "directions as general torch code (no hand-written kernels)")
    elif torch_probe.get("no_kernels_confirmed") is False:
        findings.append("at least one geometry under test now binds a "
                        "hand-written kernel: {}".format(
                            torch_probe.get("torch_body_directions")))
    jax_probe = identity.get("jax") or {}
    if jax_probe.get("at_pinned_commit") is False:
        findings.append("mbirjax is at {} and the arms were written against "
                        "{}".format((jax_probe.get("git") or {}).get("commit"),
                                    PINNED_MBIRJAX_COMMIT))

    for item in comparisons:
        if item["fingerprint_note"]:
            findings.append("{}: {}".format(item["name"],
                                            item["fingerprint_note"]))
        if item["voxel_geometry_note"]:
            findings.append("{}: {}".format(item["name"],
                                            item["voxel_geometry_note"]))

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm produced a result, every staged file "
              "verified its md5, and every model that ran matched the staged "
              "recon shape")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings outside the table")

    return dict(healthy=not broken, broken=broken, findings=findings,
                comparisons=comparisons,
                arms={name: dict(warm_s=row.get("warm_s"),
                                 cold_s=row.get("cold_s"),
                                 peak_bytes=row.get("peak_bytes"),
                                 error=row.get("error"))
                      for name, row in arm_rows.items()})


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
