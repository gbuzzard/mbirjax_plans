"""mg15c -- ADJUDICATING THE ADJOINT: is the back-projection cross-count
difference a SPLIT DEFECT, or a float32 GROUPING property that predates the
padding removal?

WHERE THIS SITS.  mg15 gated the removal of the sharding pad and passed every
leg but one: the multiaxis four-device reconstruction backstop read 2.969e-02
against a 1e-2 gate.  mg15b (slurm 15304687) adjudicated that reading and
answered it -- the reconstruction difference is trajectory-shaped: it decayed
from 2.969e-02 at three iterations to 1.682e-02 at ten, its boundary peak
decayed with the interior, and the boundary slices ranked #271, #272 and #120
of 510 rather than at the top.  But mg15b's own new instrument, the ADJOINT
leg it added because mg15 lacked one, tripped its gate:

    back projection, multiaxis, n=4 against n=1     6.289e-04   (nrmse 5.580e-06)
    forward projection, multiaxis, n=4 (mg15)       1.585e-05
    same-count reconstruction floors (mg15b)        3.4e-06 and 2.0e-06

A single back projection is one application of a linear operator, so that
6.289e-04 cannot be trajectory divergence -- there is no trajectory in one
operator.  It needs its own adjudication, and this probe is it.

THE LEADING HYPOTHESIS, stated before the run so the result cannot be
rationalized afterwards.  The forward and back directions are not symmetric
under a VIEW-sharded split, and the asymmetry is exactly the shape of the
readings above:

  * FORWARD.  One output element of a forward projection is one detector
    element of one view, and one view lives entirely on one device.  Its sum
    never crosses the split.  Whatever device count is used, each output
    element is accumulated by the same additions in the same grouping, so the
    forward leg reads scheduling noise only -- which is what 1.585e-05 is.

  * BACK.  One output element of a back projection is one voxel, and every
    view contributes to it.  Its sum genuinely splits.  At one device, ~512
    same-sign contributions are accumulated in one long float32 chain; at four
    devices, four shorter per-device chains are formed and then combined.  Same
    magnitude, same signs, different association order -- and float32 addition
    is not associative.  Sequential-against-blocked accumulation of same-sign
    terms differs at precisely this scale: the repo's own lessons file measured
    2.1e-3 at 384^3 for sequential-against-pairwise on a different operator.

If that is what 6.289e-04 is, then it is a GROUPING property of splitting a
reduction, not a property of where the block boundaries fall.  And the crucial
consequence: it would have read the same on the PADDED code, because padding
only appended zeros to the end of the axis -- zeros change no sum -- and the
number of devices, and therefore the number of partial chains, was the same.
It would then be a pre-existing property of multi-device back projection that
the first cross-count VALUE instrument at this problem size has simply
surfaced for the first time, and not a regression introduced by the pad
removal.

THREE INSTRUMENTS, on three independent axes.  Every arm here is a SINGLE back
projection: no reconstruction, no iteration, no memory ledger, no calibration
mode.  Every arm pins with an explicit configure_devices device list installed
at construction, so the projectors run on the real n-device layout from the
first call and mg15's settle trap cannot arise.

  1. THE COUNT MATRIX (current tree, multiaxis).  The staged sinogram is
     back-projected at one, two, three and four devices, and the driver prints
     the full pairwise table of max|a - b| / max|b| -- 1v2, 1v3, 1v4, 2v3, 2v4,
     3v4 -- plus a SAME-COUNT repeat at n=1 and at n=4, two back projections in
     one process on one model, which is the adjoint instrument's own atomics
     floor and is a number mg15b never measured.

     THE DISCRIMINATING PATTERN.  A grouping property depends on how many
     partial chains there are and hardly at all on where they are cut, so it
     puts n=2, n=3 and n=4 tightly TOGETHER and leaves all three sitting at a
     similar ~6e-4 distance from n=1: the step is one-against-many, not
     between the many.  A split-length defect depends on where the cuts fall,
     so it would single out the NON-DIVIDING count -- 512 views over three
     devices leaves 2, where two and four divide 512 exactly -- and n=3 would
     stand apart from n=2 and n=4 rather than beside them.

  2. THE PRE-REMOVAL ABLATION (the decisive one).  The same one-against-four
     adjoint comparison, run on the PADDED code: greg_dev commit 672edbd
     ("Implement padding-removal P2"), the last commit before P3 changed the
     split.  That tree is staged separately on the cluster and its arms run
     under their own overlay interpreter; they read the SAME staged sinogram
     bytes, because the staging format did not change between the two commits.

     THE VERDICT RULE, registered here.  If the padded tree reads within a
     factor of about two of the current one, the adjoint difference PREDATES
     the pad removal and is the grouping property, not a regression.  If the
     padded tree reads near 1e-7 while the current tree reads 6e-4, the removal
     changed the adjoint and the split is implicated.

  3. FAMILY CONTEXT (current tree).  The same adjoint comparison at n=3 against
     n=1 for cone and for parallel.  Their back projections run the
     hand-written Triton kernels and, for parallel, the row-aligned driver,
     where multiaxis runs a general torch body.  Report-only: it says whether
     the adjoint cross-count reading is specific to torch bodies or universal
     to splitting a reduction, which is a fact the findings need either way.

WHAT THE TWO TREES DISAGREE ABOUT IN THEIR OWN API, which the p2 arms have to
work around.  672edbd predates commit 7032535, which renamed Placement's third
argument from ``real_size`` to ``axis_len``, so no keyword spelling works on
both trees and every call here is POSITIONAL.  The difference goes further than
the rename, and the check had to be written from the p2 source rather than
assumed: on the padded tree ``shard_ranges`` takes a REQUIRED size and REFUSES
a length its device count does not divide -- that refusal IS the padding
contract -- so 512 over three devices is reached there through
``padded_shard_ranges()``, or equivalently ``shard_ranges(padded_size)`` with
padded_size 513, and yields three EQUAL blocks of 171.  The current tree has no
padded_size at all and splits 512 into 171, 171, 170 directly.  Each arm
records the split signature of the tree it is actually running, so the two
trees identify themselves in the rows rather than being taken on trust, and
each p2 arm additionally asserts that its imported mbirtorch really came from
the p2 checkout.

NO GATES ANYWHERE.  This is an adjudication probe.  Its exit code reports
INSTRUMENT HEALTH only -- 0 when every arm ran and every comparison computed, 2
otherwise -- and the verdict is read by a human from the printed matrix.  A
threshold on any of these numbers would be exactly the kind of borrowed
constant that produced the question in the first place.

Run:
    <torch python> mg15c_adjoint_matrix.py     on a 4-GPU node
    python mg15c_adjoint_matrix.py --dry-run   anywhere: print the arm plan
    python mg15c_adjoint_matrix.py --help

Environment:
    MG15C_RESULTS=<dir>            where the jsonl and the artifacts go
    MG15C_KEEP_ARTIFACTS=1         keep the staged volumes after the run
    MG15C_P2_PYTHON=<path>         the padded tree's interpreter
    MG15C_SMOKE=1 / MG15C_DEVICE=cpu   the local CPU smoke, which skips the
                                   padded-tree arms: that checkout and its
                                   overlay venv exist only on the cluster
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
# The three families, on mg15's and mg15b's cells so every number here is
# commensurable with the readings being adjudicated.  multiaxis is the family
# in question and carries the count matrix; cone and parallel appear only as
# context at one non-dividing count.
FAMILIES = {
    "multiaxis": dict(cell=(512, 448, 384)),
    "cone": dict(cell=(512, 448, 384)),
    "parallel": dict(cell=(512, 448, 384)),
}
MATRIX_COUNTS = (1, 2, 3, 4)     # instrument 1, multiaxis, current tree
REPEAT_COUNTS = (1, 4)           # which matrix arms project twice in-process
P2_COUNTS = (1, 4)               # instrument 2, multiaxis, padded tree
CONTEXT_COUNTS = (1, 3)          # instrument 3, cone and parallel

SMOKE = os.environ.get("MG15C_SMOKE", "0") == "1"
SMOKE_FAMILIES = {
    "multiaxis": dict(cell=(16, 24, 20)),
    "cone": dict(cell=(16, 14, 12)),
    "parallel": dict(cell=(16, 14, 12)),
}

DEVICE = os.environ.get("MG15C_DEVICE", "cpu" if SMOKE else "cuda")
FAMILY_SPECS = SMOKE_FAMILIES if SMOKE else FAMILIES
PHANTOM_SEED = 12345             # mg15b's seed, used for the dots draw too

# The padded tree: the last commit before the split changed.  Staged on the
# cluster before the job runs, with its own overlay venv so its arms import it
# instead of the current checkout.  The smoke skips these arms entirely.
P2_COMMIT = "672edbd"
P2_ROOT = os.environ.get("MG15C_P2_ROOT",
                         "/scratch/gautschi/buzzard/torch_p3/mbirtorch_p2")
P2_PYTHON = os.environ.get(
    "MG15C_P2_PYTHON", "/scratch/gautschi/buzzard/torch_p3/venv_p2/bin/python")

# The readings this probe is placed beside, so the summary can print them next
# to the new ones instead of asking a reader to hold them in their head.
REFERENCE_READINGS = (
    ("mg15b back projection ma n=4 vs n=1", 6.289e-04, 5.580e-06,
     "THE READING IN QUESTION"),
    ("mg15  forward projection ma n=4", 1.585e-05, None,
     "the direction whose sums do not split"),
    ("mg15b same-count recon floor n=4", 3.4e-06, None, "reconstruction, not adjoint"),
    ("mg15b same-count recon floor n=1", 2.0e-06, None, "reconstruction, not adjoint"),
    ("mg15b recon n=4 vs n=1, 3 iter", 2.969e-02, None, "decayed to 1.682e-02 at 10"),
)
# The prior measurement the grouping hypothesis rests on, quoted so instrument 1
# has a documented scale to be read against rather than an invented one.
LESSONS_NOTE = ("the repo's lessons file measured 2.1e-3 at 384^3 for "
                "sequential-against-pairwise float32 accumulation on a "
                "different operator")

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
    "MG15C_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _sino_path(family):
    return os.path.join(RESULTS_DIR, f"_mg15c_sino_{family}.npy")


def _phantom_path(family):
    return os.path.join(RESULTS_DIR, f"_mg15c_phantom_{family}.npy")


def _bp_path(family, tree, n_dev):
    return os.path.join(RESULTS_DIR, f"_mg15c_bp_{family}_{tree}_n{n_dev}.npy")


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


def compare_arrays(out, ref, budget_bytes=64 << 20):
    """max|out - ref| / max|ref| in float64, with a normalized RMS beside it.

    mg15b's comparator, unchanged, so a number printed here means the same
    thing as a number printed there.  Walked in slabs along the first axis so
    neither the float32 arrays nor their float64 promotions are held whole; the
    maximum is accumulated slab by slab, which is exact.  No gate: this probe
    reports readings and does not judge them.
    """
    import numpy as np

    if tuple(out.shape) != tuple(ref.shape):
        return dict(rel=None,
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
        return dict(rel=None,
                    reason="the reference is all zeros, so a relative "
                           "comparison has no denominator")
    return dict(rel=max_diff / max_ref, max_abs_diff=max_diff,
                max_abs_ref=max_ref, shape=list(ref.shape),
                nrmse=((sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else None))


# ── which tree am I? ──────────────────────────────────────────────────────────
def split_signature(n_devices=3, axis_len=512):
    """What the mbirtorch in THIS interpreter does when it splits ``axis_len``
    across ``n_devices``.  Written to run on both trees.

    Every call is POSITIONAL.  672edbd names Placement's third argument
    ``real_size`` and the current tree names it ``axis_len`` (renamed in
    7032535), so no keyword spelling works on both.  The difference is not only
    the name: on the PADDED tree ``shard_ranges`` takes a REQUIRED size and
    raises on a length its device count does not divide -- that refusal is the
    padding contract itself -- so the padded tree is asked for its blocks
    through ``padded_shard_ranges``, while the current tree splits the real
    length directly.  Which branch was taken is recorded, because it is the
    cleanest evidence of which tree actually got imported.
    """
    import mbirtorch
    from mbirtorch._sharding import Placement

    placement = Placement(["cpu"] * n_devices, 0, axis_len)
    pads = hasattr(placement, "padded_shard_ranges")
    padded_size = getattr(placement, "padded_size", None)
    if pads:
        ranges = placement.padded_shard_ranges()
        blocks = [end - start for _d, (start, end), *_rest in ranges]
    else:
        ranges = placement.shard_ranges(axis_len)
        blocks = [end - start for _d, (start, end) in ranges]
    return dict(axis_len=axis_len, n_devices=n_devices, blocks=blocks,
                equal_blocks=(len(set(blocks)) == 1), pads_api=bool(pads),
                padded_size=(int(padded_size) if padded_size else None),
                is_padded=bool(getattr(placement, "is_padded", False)),
                mbirtorch_file=str(getattr(mbirtorch, "__file__", "")))


def _assert_tree(result, tree):
    """Confirm the arm imported the tree it was supposed to.

    The padded arms are the whole point of instrument 2, and an overlay venv
    that silently resolved to the current checkout would produce two identical
    columns and a confident wrong answer.  So the check is on the imported
    FILE PATH, not on a flag, and the split signature is recorded beside it as
    independent corroboration.
    """
    signature = split_signature()
    result["split_signature"] = signature
    path = signature["mbirtorch_file"]
    result["mbirtorch_file"] = path
    in_p2 = "mbirtorch_p2" in path
    result["tree"] = tree
    if tree == "p2":
        result["tree_ok"] = bool(in_p2 and signature["equal_blocks"]
                                 and signature["pads_api"])
        if not in_p2:
            raise RuntimeError(
                f"a p2 arm imported mbirtorch from {path!r}, which is not the "
                f"padded checkout at {P2_ROOT}: the overlay interpreter did "
                f"not take effect, and this arm would have measured the "
                f"current tree twice")
        if not (signature["pads_api"] and signature["equal_blocks"]):
            raise RuntimeError(
                f"the tree at {path!r} does not pad: 512 over 3 devices gives "
                f'{signature["blocks"]}, and a padded tree must give three '
                f"equal blocks")
    else:
        result["tree_ok"] = not in_p2
        if in_p2:
            raise RuntimeError(
                f"a current-tree arm imported mbirtorch from {path!r}, which "
                f"is the padded checkout: the interpreters are crossed")


# ── the GPU health sample ─────────────────────────────────────────────────────
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
def device_list(n_dev):
    """The explicit device list an arm pins to.

    Installed at construction, so back_project runs on the real n-device layout
    from the first call.  mg15 had to pin through MBIRTORCH_NUM_DEVICES because
    its memory leg needed the automatic branch, and that pin acts only through
    the device policy, which the projectors never call; here there is no
    automatic branch and therefore no settle to wait for.
    """
    if DEVICE == "cuda":
        return [f"cuda:{i}" for i in range(n_dev)]
    return [DEVICE] * n_dev


def build_model(family, n_dev):
    """mg15's and mg15b's model builders, unchanged, pinned to an explicit
    device list."""
    import numpy as np

    import mbirtorch

    cell = tuple(FAMILY_SPECS[family]["cell"])
    num_views, channels = cell[0], cell[2]
    if family == "cone":
        # Source distances written as multiples of the detector width so the
        # same expression builds the smoke model: source-to-detector at four
        # widths, source-to-isocenter at two.
        model = mbirtorch.ConeBeamModel(
            cell, np.linspace(0, 2 * np.pi, num_views, endpoint=False),
            source_detector_dist=4.0 * channels,
            source_iso_dist=2.0 * channels)
    elif family == "parallel":
        # Half a turn is a full parallel-beam scan.  This is the row-aligned
        # geometry, so its detector rows shard with its recon slices.
        model = mbirtorch.ParallelBeamModel(
            cell, np.linspace(0, np.pi, num_views, endpoint=False))
    else:
        # mg8's, mg15's and mg15b's ma512 angles exactly: azimuths evenly
        # spaced over half a turn, elevations swept across +/- 0.5 radians.
        model = mbirtorch.MultiAxisParallelModel(
            cell, np.stack([np.linspace(0, np.pi, num_views, endpoint=False),
                            np.linspace(-0.5, 0.5, num_views)], axis=1))
    model.configure_devices(devices=device_list(n_dev))
    model.set_params(no_warning=True, verbose=0)
    return model


def _blocks(model):
    """The block each device owns on the two sharded axes, as this model
    realized them."""
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    recon_shape = tuple(model.get_params("recon_shape"))
    try:
        views = [end - start for _d, (start, end)
                 in model.sino_placement.shard_ranges(sinogram_shape[0])]
        slices = [end - start for _d, (start, end)
                  in model.recon_placement.shard_ranges(recon_shape[2])]
    except Exception as exc:                                      # noqa: BLE001
        # The padded tree refuses a length its device count does not divide.
        # That refusal is itself informative, so it is recorded rather than
        # allowed to fail the arm.
        return dict(view_blocks=None, slice_blocks=None,
                    unavailable=f"{type(exc).__name__}: {exc}")
    return dict(view_blocks=views, slice_blocks=slices,
                views_divide=(sinogram_shape[0] % max(1, len(views)) == 0),
                slices_divide=(recon_shape[2] % max(1, len(slices)) == 0))


def _verify_staged(result, family):
    """Re-check the staged phantom and sinogram before using them.

    Ten arms across TWO interpreters read these bytes.  The padded-tree arms in
    particular must read exactly what the current-tree arms read, or instrument
    2 compares two different problems and says nothing.
    """
    for label, path in (("sino", _sino_path(family)),
                        ("phantom", _phantom_path(family))):
        with open(_md5_path(path)) as handle:
            expected = handle.read().strip()
        actual = _md5(path)
        result[f"{label}_md5"] = actual
        result[f"{label}_md5_ok"] = (actual == expected)
        if actual != expected:
            raise RuntimeError(f"staged {label} checksum mismatch at {path}: "
                               f"{actual} != {expected}")


def _save(path, volume):
    """Stage a volume as float32 with a checksum sidecar, mg15b's discipline."""
    import numpy as np

    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    np.save(path, volume)
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    return dict(path=path, md5=digest, shape=list(volume.shape),
                abs_sum=float(np.sum(np.abs(volume, dtype=np.float64))))


# ── the workers ───────────────────────────────────────────────────────────────
def run_backproject(cfg):
    """One back projection of the staged sinogram on this arm's layout.

    That is the entire arm.  There is no reconstruction here and nothing
    iterative: a single application of a linear operator, which is what makes
    every reading in this probe attributable to the operator rather than to a
    trajectory.

    ``repeat`` runs the projection a second time in the SAME process on the
    SAME model.  On CUDA that measures the adjoint's own run-to-run floor,
    which comes from the scatter-adds accumulating in a nondeterministic order;
    on CPU it is expected to be exactly zero, which is itself worth recording
    because it says the CPU path has no such nondeterminism to confound the
    smoke.
    """
    import numpy as np
    import torch

    family, n_dev = cfg["family"], cfg["n_dev"]
    result = dict(cfg, device=DEVICE)
    _assert_tree(result, cfg.get("tree", "current"))
    model = build_model(family, n_dev)
    realized = [str(d) for d in model.sino_placement.devices]
    result.update(requested_devices=device_list(n_dev),
                  realized_devices=realized, realized_n_devices=len(realized),
                  devices_ok=(len(realized) == n_dev),
                  pin_mechanism="configure_devices(devices=[...])",
                  recon_shape=list(model.get_params("recon_shape")),
                  blocks=_blocks(model),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  interpreter=sys.executable)
    _verify_staged(result, family)
    sinogram = np.load(_sino_path(family))
    health = [sample_gpu_health()]

    def one_projection():
        start = time.perf_counter()
        volume = _to_numpy(model.back_project(sinogram))
        if DEVICE == "cuda" and torch.cuda.is_available():
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return volume, time.perf_counter() - start

    first, first_s = one_projection()
    walls = [first_s]
    result["staged"] = _save(_bp_path(family, result["tree"], n_dev), first)
    if cfg.get("repeat"):
        second, second_s = one_projection()
        walls.append(second_s)
        # No gate: this reading IS the scale the cross-count numbers are read
        # against, so judging it would be circular.
        result["same_count"] = compare_arrays(second, first)
        del second
    result["back_project_s"] = walls
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def generate(cfg):
    """The staged phantom and its one-device sinogram for one family.

    Pinned to a single device so the generator cannot itself become a
    multi-device run.  The dots draw is seeded so a re-run of this job stages
    the same phantom; the checksums make that a fact rather than a hope, and
    they are what lets the padded-tree arms in another interpreter read
    provably identical bytes.
    """
    import numpy as np
    import torch

    import mbirtorch

    family = cfg["family"]
    result = dict(cfg, device=DEVICE)
    _assert_tree(result, "current")
    model = build_model(family, 1)
    recon_shape = tuple(model.get_params("recon_shape"))
    np.random.seed(PHANTOM_SEED)
    phantom = np.ascontiguousarray(np.asarray(
        mbirtorch.gen_translation_phantom(recon_shape, "dots", None,
                                          fill_rate=0.05), dtype=np.float32))
    sinogram = _to_numpy(model.forward_project(phantom))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result.update(recon_shape=list(recon_shape),
                  num_pixels_full=int(model.full_index_count()))
    result["phantom"] = _save(_phantom_path(family), phantom)
    result["sino"] = _save(_sino_path(family), sinogram)
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


WORKERS = dict(generate=generate, backproject=run_backproject)


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing leaks in.

    Both mbirtorch knobs are POPPED and neither is set: this probe reads no
    memory, so it must not own a peak counter, and every arm pins by explicit
    device list, so an inherited count pin would describe a mechanism that is
    not in use.

    PYTHONPATH is dropped from the padded-tree arms.  Their whole purpose is to
    import a DIFFERENT checkout through their own interpreter, and an inherited
    PYTHONPATH pointing at the current tree would shadow it -- silently, and in
    the one direction that makes instrument 2 report two identical columns.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg.get("tree") == "p2":
        env.pop("PYTHONPATH", None)
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter -- and for the padded arms,
    in a DIFFERENT one, which is how two trees are measured in one job."""
    payload = json.dumps(cfg)
    interpreter = cfg.get("python") or sys.executable
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [interpreter, "-u", os.path.abspath(__file__), "--worker", payload],
            capture_output=True, text=True, env=arm_env(cfg))
    except (OSError, ValueError) as exc:
        # A missing overlay interpreter lands here.  It is a setup failure, not
        # a measurement, and it must be a legible row rather than a traceback
        # that costs the job its other nine arms.
        return dict(cfg, error=f"cannot run {interpreter!r}: {exc}",
                    subprocess_wall_s=time.perf_counter() - start)
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def build_plan():
    """The plan in job order: the generators, then the count matrix, then the
    padded ablation, then the family context.

    Instrument 1 runs before instrument 2 on purpose: if the matrix already
    shows n=3 standing apart from n=2 and n=4, the ablation is being read
    against a different question than if it does not.
    """
    families = ["multiaxis", "cone", "parallel"]
    plan = [dict(mode="generate", arm_id=f"generate_{f}", family=f, n_dev=1,
                 tree="current") for f in families]
    for n in MATRIX_COUNTS:
        plan.append(dict(mode="backproject", arm_id=f"ma_current_n{n}",
                         family="multiaxis", n_dev=n, tree="current",
                         repeat=(n in REPEAT_COUNTS)))
    if not SMOKE:
        for n in P2_COUNTS:
            plan.append(dict(mode="backproject", arm_id=f"ma_p2_n{n}",
                             family="multiaxis", n_dev=n, tree="p2",
                             repeat=False, python=P2_PYTHON))
    for family in ("cone", "parallel"):
        for n in CONTEXT_COUNTS:
            plan.append(dict(mode="backproject",
                             arm_id=f"{family}_current_n{n}", family=family,
                             n_dev=n, tree="current", repeat=False))
    return plan


def _dry_run(plan):
    arms = [c for c in plan if c["mode"] == "backproject"]
    print(f"mg15c adjoint matrix: {len(arms)} back-projection arms "
          f"({len(plan) - len(arms)} generators), device {DEVICE}")
    print(f"  adjudicating mg15b's adjoint reading "
          f"{REFERENCE_READINGS[0][1]:.3e}: grouping property or split defect?")
    for cfg in plan:
        if cfg["mode"] == "generate":
            print(f'  {cfg["arm_id"]:<20} stage phantom + one-device sinogram, '
                  f'cell {tuple(FAMILY_SPECS[cfg["family"]]["cell"])}')
            continue
        views = FAMILY_SPECS[cfg["family"]]["cell"][0]
        n = cfg["n_dev"]
        divides = "divides" if views % n == 0 else f"leaves {views % n}"
        print(f'  {cfg["arm_id"]:<20} n={n} tree={cfg["tree"]:<7} '
              f'{"repeat x2  " if cfg.get("repeat") else "           "}'
              f'{views} views over {n}: {divides}')
    if SMOKE:
        print(f"  SMOKE: the {len(P2_COUNTS)} padded-tree arms are SKIPPED -- "
              f"the {P2_COMMIT} checkout and its overlay venv exist only on "
              f"the cluster")
    else:
        print(f"  padded-tree arms run under {P2_PYTHON}")
        print(f"    over the {P2_COMMIT} checkout at {P2_ROOT}")
    print("no gates anywhere: the exit code reports INSTRUMENT HEALTH only, "
          "and the verdict is read from the matrix.")


def main():
    plan = build_plan()
    if "--dry-run" in sys.argv:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, f"mg15c_adjoint_matrix_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg15c adjoint matrix on {RUN_LABEL} ({DEVICE}) -> {out_path}",
          flush=True)
    rows = []
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
    if os.environ.get("MG15C_KEEP_ARTIFACTS", "0") != "1":
        paths = []
        for family in FAMILY_SPECS:
            paths.extend([_sino_path(family), _phantom_path(family)])
            for tree in ("current", "p2"):
                for n in set(MATRIX_COUNTS) | set(P2_COUNTS) | set(CONTEXT_COUNTS):
                    paths.append(_bp_path(family, tree, n))
        for path in list(paths):
            paths.append(_md5_path(path))
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
    else:
        print(f"MG15C_KEEP_ARTIFACTS=1: the staged volumes are left in "
              f"{RESULTS_DIR}")
    print(f"\nwrote {out_path}")
    return 0 if summary["instruments_healthy"] else 2


# ── the summary ───────────────────────────────────────────────────────────────
def _staged(by_id, arm_id):
    entry = (by_id.get(arm_id) or {}).get("staged") or {}
    path = entry.get("path")
    return path if path and os.path.exists(path) else None


def _pair(by_id, arm_a, arm_b, problems, label):
    """max|a - b| / max|b| between two staged back projections."""
    import numpy as np

    a, b = _staged(by_id, arm_a), _staged(by_id, arm_b)
    if not a or not b:
        problems.append(f"{label}|missing volume")
        return None
    return compare_arrays(np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r"))


def _fmt(check, key="rel"):
    if not check or check.get(key) is None:
        return "n/a"
    return f'{check[key]:.3e}'


def summarize(rows, out_path):
    print(f"\n===== mg15c: is the adjoint cross-count difference a GROUPING "
          f"property or a SPLIT defect? ({out_path}) =====")
    by_id = {r.get("arm_id"): r for r in rows}
    problems = []

    # ── the arms, and which tree each actually imported ──────────────────────
    print("\nARMS")
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'  {arm_id:<20} ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:78]}')
            problems.append(f"{arm_id}|error")
            continue
        if row.get("mode") == "generate":
            print(f'  {arm_id:<20} staged recon shape {row.get("recon_shape")}')
            continue
        sig = row.get("split_signature") or {}
        blocks = row.get("blocks") or {}
        walls = "".join(f"{w:.1f}s " for w in (row.get("back_project_s") or []))
        print(f'  {arm_id:<20} tree={row.get("tree"):<7} '
              f'on {row.get("realized_devices")}  '
              f'512/3 -> {sig.get("blocks")}'
              f'{" (padded)" if sig.get("pads_api") else ""}  '
              f'views {blocks.get("view_blocks")}  {walls}')
        if row.get("devices_ok") is False:
            print(f'    ARM CHECK FAIL: realized {row.get("realized_devices")} '
                  f'for n={row.get("n_dev")}')
            problems.append(f"{arm_id}|devices")
        if row.get("tree_ok") is False:
            print(f'    ARM CHECK FAIL: imported {row.get("mbirtorch_file")}, '
                  f'which is not the {row.get("tree")} tree')
            problems.append(f"{arm_id}|tree")

    # ── instrument 1: the count matrix ───────────────────────────────────────
    print(f"\nINSTRUMENT 1 -- THE COUNT MATRIX (multiaxis, current tree)")
    print(f"  a GROUPING property puts n=2/3/4 together, all a similar "
          f"distance from n=1 (the step is one-against-many).")
    print(f"  a SPLIT defect singles out the NON-DIVIDING count: 512 views "
          f"over 3 leaves 2, where 2 and 4 divide 512 exactly.")
    matrix = {}
    counts = list(MATRIX_COUNTS)
    for i, a in enumerate(counts):
        for b in counts[i + 1:]:
            matrix[f"{a}v{b}"] = _pair(by_id, f"ma_current_n{b}",
                                       f"ma_current_n{a}", problems,
                                       f"matrix|{a}v{b}")
    header = "        " + "".join(f"{'n=' + str(b):>12}" for b in counts[1:])
    print(f"\n  max|a - b| / max|b|, current tree")
    print(header)
    for i, a in enumerate(counts[:-1]):
        cells = "".join(
            f'{_fmt(matrix.get(f"{a}v{b}")):>12}' if b > a else f'{"":>12}'
            for b in counts[1:])
        print(f"  n={a:<4}{cells}")
    floors = {}
    for n in REPEAT_COUNTS:
        floors[n] = (by_id.get(f"ma_current_n{n}") or {}).get("same_count")
        if floors[n] is None:
            problems.append(f"instrument1|same-count floor n{n} missing")
    print("\n  same-count repeats (the adjoint's own run-to-run floor, "
          "never measured before):")
    for n in REPEAT_COUNTS:
        print(f'    n={n} vs itself      {_fmt(floors.get(n)):>12}   '
              f'nrmse {_fmt(floors.get(n), "nrmse")}')

    # The reading that decides instrument 1: how tightly the multi-device
    # counts agree with EACH OTHER, against how far they all sit from n=1.
    one_vs = [matrix.get(f"1v{b}") for b in counts[1:]]
    among = [matrix.get(f"{a}v{b}") for i, a in enumerate(counts[1:])
             for b in counts[1:][i + 1:]]
    one_vals = [c["rel"] for c in one_vs if c and c.get("rel") is not None]
    among_vals = [c["rel"] for c in among if c and c.get("rel") is not None]
    if one_vals and among_vals:
        spread = max(one_vals) / min(one_vals) if min(one_vals) > 0 else float("inf")
        print(f"\n  one-against-many: {', '.join(f'{v:.3e}' for v in one_vals)}"
              f"  (spread {spread:.2f}x)")
        print(f"  among the many:   {', '.join(f'{v:.3e}' for v in among_vals)}"
              f"  (largest {max(among_vals):.3e})")
        # BEFORE reading the pattern, ask whether there is a pattern to read.
        # If every entry sits at the same scale the matrix is flat, and a flat
        # matrix supports no conclusion in either direction -- least of all the
        # alarming one.  This is the expected state on CPU, where the back
        # projection is deterministic (the same-count floor is exactly 0) and
        # every entry is at the float32 representation granularity.
        all_vals = one_vals + among_vals
        flat_ratio = (max(all_vals) / min(all_vals)
                      if min(all_vals) > 0 else float("inf"))
        floor_vals = [f["rel"] for f in floors.values()
                      if f and f.get("rel") is not None]
        if flat_ratio <= 2.0:
            print(f"  -> THE MATRIX IS FLAT ({flat_ratio:.2f}x between its "
                  f"largest and smallest entry).  Every device count agrees "
                  f"with every other to the same scale, so there is no "
                  f"pattern here to read in EITHER direction.")
            if floor_vals and max(floor_vals) <= 0.0:
                print(f"     The same-count floor is exactly 0, so this "
                      f"back projection is deterministic and the entries "
                      f"above are float32 representation granularity, not a "
                      f"measurement of grouping.  Expected on CPU; only a run "
                      f"whose readings rise above this floor can discriminate.")
        elif max(among_vals) < min(one_vals):
            print(f"  -> the multi-device counts agree with each other more "
                  f"closely than any of them agrees with n=1, and the "
                  f"non-dividing n=3 does not stand apart.  That is the "
                  f"GROUPING pattern: the step is one-against-many.")
        else:
            print(f"  -> at least one pair among the multi-device counts "
                  f"differs by as much as the step from n=1.  Read which pair "
                  f"from the matrix: if it involves n=3, the non-dividing "
                  f"count is implicated.")
        if floor_vals and max(floor_vals) > 0:
            print(f"     for scale, the largest matrix entry is "
                  f"{max(all_vals) / max(floor_vals):.1f}x the adjoint's own "
                  f"same-count floor ({max(floor_vals):.3e})")
    print(f"  for scale, {LESSONS_NOTE}")

    # ── instrument 2: the pre-removal ablation ───────────────────────────────
    print(f"\nINSTRUMENT 2 -- THE PRE-REMOVAL ABLATION (the decisive one)")
    current_1v4 = matrix.get(f"1v{MATRIX_COUNTS[-1]}")
    p2_1v4 = None
    if SMOKE:
        print(f"  SKIPPED on the smoke: the padded {P2_COMMIT} checkout and "
              f"its overlay venv exist only on the cluster.  On the cluster "
              f"this is the reading that settles the question.")
    else:
        p2_1v4 = _pair(by_id, f"ma_p2_n{P2_COUNTS[-1]}",
                       f"ma_p2_n{P2_COUNTS[0]}", problems, "instrument2")
        print(f'  padded tree ({P2_COMMIT})   n=4 vs n=1   '
              f'{_fmt(p2_1v4):>12}   nrmse {_fmt(p2_1v4, "nrmse")}')
        print(f'  current tree             n=4 vs n=1   '
              f'{_fmt(current_1v4):>12}   nrmse {_fmt(current_1v4, "nrmse")}')
        if p2_1v4 and current_1v4 and p2_1v4.get("rel") and current_1v4.get("rel"):
            ratio = current_1v4["rel"] / p2_1v4["rel"]
            print(f"  the current tree reads {ratio:.2f}x the padded tree.")
            if ratio <= 2.0 and ratio >= 0.5:
                print(f"  -> WITHIN A FACTOR OF TWO: the adjoint difference "
                      f"PREDATES the pad removal.  It is a property of "
                      f"splitting a float32 reduction across devices, not a "
                      f"regression, and mg15b's 6.289e-04 surfaced it rather "
                      f"than caused it.")
            elif p2_1v4["rel"] < 1e-6 <= current_1v4["rel"]:
                print(f"  -> the padded tree reproduces to ~1e-7 while the "
                      f"current tree does not.  The removal CHANGED the "
                      f"adjoint and the split is implicated.")
            else:
                print(f"  -> neither clean case: the two trees differ by "
                      f"{ratio:.2f}x.  Read this together with instrument 1 "
                      f"before concluding.")

    # ── instrument 3: family context ─────────────────────────────────────────
    print(f"\nINSTRUMENT 3 -- FAMILY CONTEXT (report only)")
    print(f"  cone and parallel back-project through the hand-written Triton "
          f"kernels; multiaxis runs a general torch body.  This says whether "
          f"the\n  adjoint reading is torch-body-specific or universal to "
          f"splitting a reduction.")
    context = {}
    for family in ("cone", "parallel"):
        hi, lo = CONTEXT_COUNTS[-1], CONTEXT_COUNTS[0]
        context[family] = _pair(by_id, f"{family}_current_n{hi}",
                                f"{family}_current_n{lo}", problems,
                                f"instrument3|{family}")
        print(f'  {family:<10} n={hi} vs n={lo}   {_fmt(context[family]):>12}   '
              f'nrmse {_fmt(context[family], "nrmse")}')
    print(f'  {"multiaxis":<10} n=4 vs n=1   {_fmt(current_1v4):>12}   '
          f'nrmse {_fmt(current_1v4, "nrmse")}  (this run)')

    # ── everything on one page ───────────────────────────────────────────────
    print(f"\n{'READING':<40}{'rel':>12}{'nrmse':>12}  note")
    print("  " + "-" * 74)
    print("  -- earlier jobs --")
    for label, rel, nrmse, note in REFERENCE_READINGS:
        print(f'  {label:<38} {rel:>11.3e} '
              f'{("n/a" if nrmse is None else f"{nrmse:.3e}"):>11}  {note}')
    print("  -- mg15c, this run --")
    for key in sorted(matrix):
        print(f'  {"multiaxis current " + key:<38} '
              f'{_fmt(matrix[key]):>11} {_fmt(matrix[key], "nrmse"):>11}  '
              f'count matrix')
    for n in REPEAT_COUNTS:
        print(f'  {"multiaxis same-count n=" + str(n):<38} '
              f'{_fmt(floors.get(n)):>11} {_fmt(floors.get(n), "nrmse"):>11}  '
              f'adjoint run-to-run floor')
    if not SMOKE:
        print(f'  {"multiaxis PADDED tree 1v4":<38} {_fmt(p2_1v4):>11} '
              f'{_fmt(p2_1v4, "nrmse"):>11}  the ablation')
    for family in ("cone", "parallel"):
        print(f'  {family + " current 1v3":<38} '
              f'{_fmt(context[family]):>11} '
              f'{_fmt(context[family], "nrmse"):>11}  family context')
    print("  " + "-" * 74)

    healthy = not problems
    print(f"\n===== INSTRUMENT HEALTH: {'OK' if healthy else 'DEGRADED'} =====")
    if problems:
        print(f"  {len(problems)} problem(s): {', '.join(problems)}")
        print("  Readings above may be incomplete.")
    else:
        print("  every arm ran and every comparison computed; the readings "
              "above are the adjudication")
    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"  GPU health: {len(hot)} row(s) sampled hot: {hot}")
    print(f"  exit code   {0 if healthy else 2}")
    return dict(instruments_healthy=healthy, problems=problems,
                matrix=matrix, same_count_floors={str(k): v
                                                  for k, v in floors.items()},
                padded_tree=p2_1v4, current_1v4=current_1v4,
                family_context=context, smoke=SMOKE,
                reference=[list(r) for r in REFERENCE_READINGS], hot=hot)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        try:
            out = WORKERS[cfg["mode"]](cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        sys.exit(main())
