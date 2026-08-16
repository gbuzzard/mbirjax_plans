"""mg15b -- ADJUDICATING THE ONE FAILING READING FROM mg15: is the multiaxis
four-device reconstruction difference a SPLIT DEFECT or OPTIMIZER-TRAJECTORY
DIVERGENCE?

WHAT HAPPENED.  mg15 (slurm 15304566, h016, 2026-08-16) gated the removal of
the sharding pad on four H100s and passed every leg but one:

    projector leg (forward)   cone_n3       5.696e-07   PASS at 1e-4
                              parallel_n3   5.420e-07   PASS at 1e-4
                              multiaxis_n4  1.585e-05   PASS at 1e-4
    ledger floor              every device, lowest ratio 1.016   PASS
    ma512_n4 ledger row       measured
    reconstruction backstop   cone_n3       6.861e-03   PASS at 1e-2
                              parallel_n3   1.817e-03   PASS at 1e-2
                              multiaxis_n4  2.969e-02   FAIL at 1e-2
                                            (nrmse 7.624e-04)

mg15's own docstring predicted that the max-norm statistic would grow with
problem size and warned that no threshold measured at small sizes transfers
exactly to the production cell, and multiaxis was already the noisiest family
in the smoke.  That is a reason to suspect trajectory divergence.  It is not
evidence, and the readings above cannot settle it: mg15's fine instrument is
the FORWARD projector, and a forward projection cannot see the adjoint.  A
defect that lives only in the sharded BACK projection -- the direction that
scatters into the sharded slice axis, and the direction whose blocks the pad
removal changed -- would leave the forward leg clean at 1.6e-5 and show up only
after the reconstruction had folded it in.  So the question is open, and this
probe exists to close it with instruments rather than with judgement.

WHAT WOULD DISTINGUISH THE TWO.  A systematic split defect and a divergence of
trajectories look different in four ways, and this probe measures all four:

    a split defect                        trajectory divergence
    ----------------------------------    ----------------------------------
    shows in a SINGLE back projection     needs the iterated loop to appear
    exceeds the same-count noise floor    sits at the same-count noise floor
    persists or grows with iterations     DECAYS with iterations
    boundary peak holds as interior       boundary peak decays with the
    falls away                            whole difference

The fourth row is deliberately not "clusters at the boundaries against spreads
through the interior".  Both start at the boundaries -- see instrument 4 -- so
only the change over iterations separates them.

THE FOUR INSTRUMENTS, in that order.

  1. BACK-PROJECTION LEG -- the instrument mg15 was missing.  The staged
     sinogram is back-projected at one device and at four, and the two are
     compared at a 1e-4 gate.  This is a single application of a linear
     operator: no iteration, no line search, no partition draw, so its only
     spread is float32 accumulation order, and mg15's forward analogue of it
     read 1.585e-05.  A misplaced block boundary moves whole slices and lands
     near one.  If this leg fails, the answer is a split defect and nothing
     further is needed; if it passes at the forward leg's scale, the adjoint
     direction is exonerated as a single operator.

  2. SAME-COUNT NOISE FLOOR -- the ruler for reading 2.969e-02.  At four
     devices, two seeded three-iteration reconstructions are run IN THE SAME
     PROCESS on the SAME model, with np.random.seed(12345) immediately before
     each, and the second is compared against the first.  Same device count,
     same partitions, same everything: whatever this reads is pure run-to-run
     nondeterminism, which on GPU comes from the scatter-adds in the torch
     bodies accumulating in a nondeterministic order.  The same measurement is
     taken at one device.  Neither is gated -- they ARE the scale that the
     cross-count number has to be read against.  If the four-device same-count
     spread is within an order of magnitude of 2.969e-02, then mg15's reading
     is noise-floor-dominated and says nothing about the split.

  3. ITERATION DECAY -- the discriminator that needs no reference at all.  The
     library documents that a cross-count difference DECAYS as iterations
     proceed: "measured to fall from 6.1e-3 at 3 iterations to 8.8e-4 at 10"
     (docs/source/usr_multi_gpu.rst).  That is the signature of two trajectories
     converging to the same optimum from slightly different starts.  A
     systematic split error does not decay -- it is re-injected every iteration
     and persists or grows.  So the one-against-four comparison is taken at
     three iterations and again at ten, and the direction of travel between
     them is the reading.

  4. BOUNDARY CONCENTRATION -- report-only, and WEAKER EVIDENCE THAN IT LOOKS.
     For the cross-count difference volume, the maximum absolute difference is
     taken per slice along the SHARDED slice axis, and the profile is printed
     two ways: the ten slices carrying the largest difference, and the six
     slices straddling each block boundary.  Under the new split the 510 slices
     divide into blocks of [128, 128, 127, 127], so the interior boundaries
     fall after slices 127, 255 and 382.

     THE TRAP, and it is worth stating plainly because the obvious reading of
     this instrument is wrong.  Boundary concentration on its own does NOT
     indicate a defect.  Everything that differs between a one-device and an
     n-device run is SEEDED at the seams -- the prior's halo exchange and the
     per-shard reductions live there and nowhere else -- so a difference of any
     origin starts at the boundaries and spreads inward as iterations carry it.
     At three iterations it has barely spread.  This probe's own CPU smoke
     demonstrates it: on a tiny volume whose adjoint leg reads 1.2e-07 and
     whose cross-count difference decays by 0.45x from three iterations to ten
     -- a split that is correct by every other instrument -- the two boundary
     slices and their immediate neighbours are still the top four slices in the
     profile, ranked #0 and #3 of 27.  A reader who took clustering as proof of
     a defect would have convicted a working split.

     So this instrument is read for CHANGE, not for shape.  It is printed at
     three iterations and again at ten: a boundary peak that decays along with
     the global difference is a seam-seeded trajectory settling down, while one
     that holds or grows while the interior falls away is systematic.  The
     ONE-device same-count repeat is printed beside them as the null reference
     -- it contains no boundary at all, so it shows what the interior noise
     looks like with nothing seeded anywhere.  The decisive instrument remains
     the adjoint, which has no iteration in it for a seam to seed.

WHAT THIS PROBE DELIBERATELY DOES NOT DO, and why that makes it simpler and
safer than mg15.  It measures VALUES ONLY.  There is no memory ledger leg, no
calibration mode, and no automatic device branch: MBIRTORCH_MEMORY_CALIBRATION
is popped from every arm's environment rather than set, and every arm pins its
devices with an explicit configure_devices(devices=[...]) list.

That last point matters for a specific reason.  mg15 had to pin through
MBIRTORCH_NUM_DEVICES, because the memory ledger is consumed by the automatic
device search and the measurement had to be taken on the branch that uses it.
But that pin acts only through the model's device policy, and neither
forward_project nor back_project calls the policy -- they project on whatever
placement the model currently holds, and a freshly built automatic model holds
the trivial single-device one until a reconstruction settles the layout.  mg15
handles that by taking its projector leg after the reconstruction.  Here the
problem does not arise at all: an explicit device list is installed at
construction, so back_project and forward_project run on the real n-device
layout from the first call, with no settle to wait for and no ordering
constraint to get right.  Every arm still records the device list it actually
realized, and a mismatch invalidates the arm.

THE ARMS.  6 arms plus 1 untimed generator, all on mg15's exact multiaxis
configuration -- cell (512, 448, 384), azimuths evenly spaced over half a turn,
elevations swept across +/- 0.5 radians, the dots phantom at fill_rate 0.05,
weights exp(-sino / (2 max)), seed 12345 -- so that every number here is
commensurable with the one being adjudicated.

    backproject   n=1    the adjoint reference
    backproject   n=4    instrument 1
    repeat        n=1    two 3-iteration passes: instrument 2 at one device,
                         and the null profile for instrument 4.  Its first
                         pass is also the 3-iteration one-device reference.
    repeat        n=4    two 3-iteration passes: instrument 2 at four devices.
                         Its first pass is also the 3-iteration four-device
                         volume, so instrument 3's short reading and
                         instrument 4's profile both come from it.
    recon_long    n=1    one 10-iteration pass
    recon_long    n=4    one 10-iteration pass: instrument 3's long reading

Running the repeat arm's first pass as the canonical three-iteration volume is
what keeps this at six arms instead of eight, and it costs nothing: that pass
is a seeded three-iteration reconstruction on the pinned layout, which is
exactly what a separate arm would have produced.

EXIT CODE: INSTRUMENT HEALTH, NOT THE VERDICT.  This job exits 0 when every arm
ran and every comparison computed, and 2 when an instrument failed to produce a
reading.  It deliberately does NOT encode the adjudication in its exit status,
because the adjudication is a judgement across four readings that point at each
other, and a single automatic threshold on it would be exactly the kind of
number this probe was written to avoid trusting.  A human reads the summary
block.  The back-projection leg's 1e-4 verdict is printed as a headline anyway,
because if that one fails the answer is already known.

Run:
    <torch python> mg15b_multiaxis_probe.py     on a 4-GPU node
    python mg15b_multiaxis_probe.py --dry-run   anywhere: print the arm plan
    python mg15b_multiaxis_probe.py --help

Environment (export from the SUBMITTING SHELL, never through an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas):
    MG15B_RESULTS=<dir>            where the jsonl and the artifacts go
    MG15B_KEEP_ARTIFACTS=1         keep the staged volumes after the run
    MG15B_SMOKE=1 / MG15B_DEVICE=cpu   the local CPU smoke
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
# mg15's multiaxis family, reproduced exactly.  These are also mg8's ma512
# numbers, which is what makes the readings here comparable to both.
CELL = (512, 448, 384)
RECON_SHAPE = (384, 384, 510)
NUM_PIXELS = 115164
COUNTS = (1, 4)

# The smoke runs the same instrument set on virtual CPU devices at a tiny size.
# Its slice axis is 27 over three devices -- blocks [9, 9, 9], boundaries after
# slices 8 and 17 -- so the boundary diagnostic has real boundaries to report on
# and its printing path is exercised before any GPU time is spent.  No recon
# shape is registered: a tiny one would pin down a geometry default no
# production run uses.
SMOKE = os.environ.get("MG15B_SMOKE", "0") == "1"
SMOKE_CELL = (16, 24, 20)
SMOKE_COUNTS = (1, 3)

DEVICE = os.environ.get("MG15B_DEVICE", "cpu" if SMOKE else "cuda")
CELL_IN_USE = SMOKE_CELL if SMOKE else CELL
COUNTS_IN_USE = SMOKE_COUNTS if SMOKE else COUNTS
ITERS_SHORT = 3          # mg15's iteration count: the reading being adjudicated
ITERS_LONG = 10          # the library's documented decay point
VCD_SEED = 12345         # mg15's seed, and mg8's before it

# The back-projection leg is the one gated comparison here, and it is gated at
# mg15's projector threshold so the two directions are judged alike.  mg15's
# forward reading on this same arm was 1.585e-05.
BACKPROJECT_GATE_REL = 1e-4

# The mg15 readings this probe exists to adjudicate, carried here so the summary
# can print them beside the new ones instead of asking a reader to hold them in
# their head.  Source: slurm 15304566 on h016, 2026-08-16.
MG15_JOB = "slurm 15304566, h016, 2026-08-16"
MG15_READINGS = (
    ("multiaxis n=4 recon, 3 iter", 2.969e-02, 7.624e-04, "THE READING IN QUESTION"),
    ("multiaxis n=4 forward proj", 1.585e-05, None, "passed at 1e-4"),
    ("cone n=3 recon, 3 iter", 6.861e-03, None, "passed at 1e-2"),
    ("parallel n=3 recon, 3 iter", 1.817e-03, None, "passed at 1e-2"),
)
# The library's own statement of what trajectory divergence does over
# iterations, quoted so instrument 3 has a documented expectation to be read
# against rather than an invented one.
DOC_DECAY = ("docs/source/usr_multi_gpu.rst: cross-count differences are "
             "'measured to fall from 6.1e-3 at 3 iterations to 8.8e-4 at 10'")

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
    "MG15B_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _sino_path():
    return os.path.join(RESULTS_DIR, "_mg15b_sino.npy")


def _phantom_path():
    return os.path.join(RESULTS_DIR, "_mg15b_phantom.npy")


def _backproj_path(n_dev):
    return os.path.join(RESULTS_DIR, f"_mg15b_backproj_n{n_dev}.npy")


def _recon_path(n_dev, iters):
    return os.path.join(RESULTS_DIR, f"_mg15b_recon_n{n_dev}_i{iters}.npy")


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


def _weights(sinogram):
    """mg15's weighting formula, unchanged, so these reconstructions are
    weighted exactly as the one being adjudicated was."""
    import numpy as np

    scale = float(np.max(sinogram))
    if scale <= 0:
        raise RuntimeError("sinogram is all zeros; the phantom did not project")
    return np.exp(-sinogram / (2 * scale)).astype(np.float32)


def compare_arrays(out, ref, gate=None, budget_bytes=64 << 20):
    """max|out - ref| / max|ref| in float64, with a normalized RMS beside it.

    mg15's comparator, unchanged, so a number printed here means the same thing
    as a number printed there.  Walked in slabs along the first axis so neither
    the float32 arrays nor their float64 promotions are held whole; the maximum
    is accumulated slab by slab, which is exact.  ``gate`` may be None, which
    records the reading without a verdict -- the same-count floors are measured
    scales, not tests.
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
                nrmse=((sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else None))


def slice_profile(out, ref, budget_bytes=64 << 20):
    """max|out - ref| per slice along the SHARDED slice axis (the last one).

    This is instrument 4's raw material.  Walked in slabs along the first axis
    with a running per-slice maximum, which is exact and never holds a float64
    copy of a volume.  Returns one float per slice, in slice order.
    """
    import numpy as np

    if tuple(out.shape) != tuple(ref.shape):
        return None
    row_bytes = max(1, int(np.prod(ref.shape[1:])) * 8)
    step = max(1, int(budget_bytes // row_bytes))
    profile = np.zeros(int(ref.shape[-1]), dtype=np.float64)
    for start in range(0, ref.shape[0], step):
        a = np.asarray(ref[start:start + step], dtype=np.float64)
        b = np.asarray(out[start:start + step], dtype=np.float64)
        # Reduce every axis but the slice axis, then fold into the running max.
        chunk = np.abs(b - a)
        profile = np.maximum(profile, chunk.reshape(-1, chunk.shape[-1]).max(axis=0))
    return [float(v) for v in profile]


# ── the GPU health sample ─────────────────────────────────────────────────────
# A throttled GPU still produces valid values, but a hot node usually means a
# neighbour job is sharing the hardware, which is worth knowing when a reading
# is being adjudicated rather than merely recorded.
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

    Explicit on purpose, and on CUDA it names the devices individually.  An
    explicit list is installed at construction, so the projectors run on the
    real n-device layout from the first call -- there is no automatic branch to
    settle and therefore no ordering trap of the kind mg15 had to work around.
    """
    if DEVICE == "cuda":
        return [f"cuda:{i}" for i in range(n_dev)]
    return [DEVICE] * n_dev


def build_model(n_dev):
    """mg15's multiaxis model, pinned to an explicit device list.

    Two angles per view: azimuth around the object, elevation (tilt) out of the
    plane.  These are mg15's and mg8's ma512 angles exactly -- azimuths evenly
    spaced over half a turn, elevations swept across +/- 0.5 radians -- because
    every number here has to be commensurable with the reading it adjudicates.
    """
    import numpy as np

    import mbirtorch

    cell = tuple(CELL_IN_USE)
    num_views = cell[0]
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.linspace(-0.5, 0.5, num_views)
    model = mbirtorch.MultiAxisParallelModel(
        cell, np.stack([azimuth, elevation], axis=1))
    model.configure_devices(devices=device_list(n_dev))
    model.set_params(no_warning=True, verbose=0)
    return model


def _blocks(model):
    """The block each device owns on the two sharded axes.

    Instrument 4 needs the slice blocks to know where the boundaries are, and
    printing them on every row means the boundary positions are read from the
    run rather than assumed from the docstring.
    """
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    recon_shape = tuple(model.get_params("recon_shape"))
    views = [end - start for _device, (start, end)
             in model.sino_placement.shard_ranges(sinogram_shape[0])]
    slices = [end - start for _device, (start, end)
              in model.recon_placement.shard_ranges(recon_shape[2])]
    return dict(view_blocks=views, slice_blocks=slices,
                views_divide=(sinogram_shape[0] % max(1, len(views)) == 0),
                slices_divide=(recon_shape[2] % max(1, len(slices)) == 0))


def _common(model, cfg, n_dev):
    """The fields every arm records: what it ran on, and whether that is what
    it was asked to run on."""
    realized = [str(d) for d in model.sino_placement.devices]
    return dict(cfg, device=DEVICE, vcd_seed=VCD_SEED,
                requested_devices=device_list(n_dev),
                realized_devices=realized, realized_n_devices=len(realized),
                devices_ok=(len(realized) == n_dev),
                pin_mechanism="configure_devices(devices=[...])",
                layout_is_automatic=bool(
                    getattr(model, "device_layout_is_automatic", False)),
                env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                recon_shape=list(model.get_params("recon_shape")),
                num_pixels_full=int(model.full_index_count()),
                blocks=_blocks(model))


def _verify_staged(result):
    """Re-check the staged phantom and sinogram before using them.

    Six arms read these bytes.  A comparison against a file that changed
    underneath the run would be a silently wrong answer rather than a loud one,
    which in an adjudication is the worst possible failure mode.
    """
    for label, path in (("sino", _sino_path()), ("phantom", _phantom_path())):
        with open(_md5_path(path)) as handle:
            expected = handle.read().strip()
        actual = _md5(path)
        result[f"{label}_md5"] = actual
        result[f"{label}_md5_ok"] = (actual == expected)
        if actual != expected:
            raise RuntimeError(f"staged {label} checksum mismatch at {path}: "
                               f"{actual} != {expected}")


def _save(path, volume):
    """Stage a volume as float32 with a checksum sidecar, mg15's discipline."""
    import numpy as np

    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    np.save(path, volume)
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    return dict(path=path, md5=digest, shape=list(volume.shape),
                abs_sum=float(np.sum(np.abs(volume, dtype=np.float64))))


# ── the workers: one arm, one process ─────────────────────────────────────────
def run_backproject(cfg):
    """INSTRUMENT 1.  One back projection of the staged sinogram on this arm's
    layout, staged for the driver to compare across device counts.

    This is the direction mg15 could not see.  It is a single application of a
    linear operator, so a correct split reproduces the one-device answer to
    float32 accumulation order and a misplaced block lands near one.
    """
    import numpy as np
    import torch

    n_dev = cfg["n_dev"]
    model = build_model(n_dev)
    result = _common(model, cfg, n_dev)
    _verify_staged(result)
    sinogram = np.load(_sino_path())
    health = [sample_gpu_health()]
    start = time.perf_counter()
    volume = _to_numpy(model.back_project(sinogram))
    if DEVICE == "cuda" and torch.cuda.is_available():
        for device in model.sino_placement.devices:
            torch.cuda.synchronize(device)
    result["back_project_s"] = time.perf_counter() - start
    result["staged"] = _save(_backproj_path(n_dev), volume)
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def run_repeat(cfg):
    """INSTRUMENT 2, and the source volume for instruments 3 and 4.

    Two seeded three-iteration reconstructions on ONE model in ONE process, the
    seed re-set immediately before each so both draw identical pixel
    partitions.  Everything that could differ between them has been held fixed
    -- same device count, same layout, same partitions, same weights -- so the
    difference between them is run-to-run nondeterminism and nothing else.

    The first pass is also staged as this arm's canonical three-iteration
    volume: it is a seeded three-iteration reconstruction on the pinned layout,
    which is exactly what a separate arm would have produced, so running one
    here instead of twice saves an arm and changes no number.
    """
    import numpy as np
    import torch

    n_dev = cfg["n_dev"]
    model = build_model(n_dev)
    result = _common(model, cfg, n_dev)
    _verify_staged(result)
    sinogram = np.load(_sino_path())
    weights = _weights(sinogram)
    health = [sample_gpu_health()]

    def one_pass():
        # The seed goes immediately before each call, inside this process, so
        # the two passes draw the same partitions from the same global stream.
        np.random.seed(VCD_SEED)
        start = time.perf_counter()
        volume, _info = model.recon(sinogram, weights=weights,
                                    max_iterations=ITERS_SHORT,
                                    stop_threshold_change_pct=0.0,
                                    logfile_path=None, print_logs=False)
        if DEVICE == "cuda" and torch.cuda.is_available():
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(volume), time.perf_counter() - start

    first, first_s = one_pass()
    second, second_s = one_pass()
    result["recon_s"] = [first_s, second_s]
    result["iterations"] = ITERS_SHORT
    result["staged"] = _save(_recon_path(n_dev, ITERS_SHORT), first)
    # No gate: this reading IS the scale that the cross-count number is read
    # against, so a verdict on it would be circular.
    result["same_count"] = compare_arrays(second, first, gate=None)
    result["same_count_slice_profile"] = slice_profile(second, first)
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def run_recon_long(cfg):
    """INSTRUMENT 3's long arm: one seeded ten-iteration reconstruction.

    One pass only.  The comparison this feeds is against the other device
    count's ten-iteration pass, and the question asked of it is whether the
    cross-count difference has DECAYED relative to three iterations.
    """
    import numpy as np
    import torch

    n_dev = cfg["n_dev"]
    model = build_model(n_dev)
    result = _common(model, cfg, n_dev)
    _verify_staged(result)
    sinogram = np.load(_sino_path())
    weights = _weights(sinogram)
    health = [sample_gpu_health()]
    np.random.seed(VCD_SEED)
    start = time.perf_counter()
    volume, _info = model.recon(sinogram, weights=weights,
                                max_iterations=ITERS_LONG,
                                stop_threshold_change_pct=0.0,
                                logfile_path=None, print_logs=False)
    if DEVICE == "cuda" and torch.cuda.is_available():
        for device in model.sino_placement.devices:
            torch.cuda.synchronize(device)
    result["recon_s"] = [time.perf_counter() - start]
    result["iterations"] = ITERS_LONG
    result["staged"] = _save(_recon_path(n_dev, ITERS_LONG), _to_numpy(volume))
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def generate(cfg):
    """The staged phantom and its one-device sinogram, checksummed.

    Pinned to a single device so the generator cannot itself become a
    multi-device run.  Every arm reads these same bytes, which is what makes
    the arms comparable at all.
    """
    import numpy as np
    import torch

    import mbirtorch

    model = build_model(1)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = np.ascontiguousarray(np.asarray(
        mbirtorch.gen_translation_phantom(recon_shape, "dots", None,
                                          fill_rate=0.05), dtype=np.float32))
    sinogram = _to_numpy(model.forward_project(phantom))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = dict(cfg, recon_shape=list(recon_shape),
               num_pixels_full=int(model.full_index_count()))
    out["phantom"] = _save(_phantom_path(), phantom)
    out["sino"] = _save(_sino_path(), sinogram)
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing leaks in.

    Both mbirtorch environment knobs are POPPED and neither is set.  The
    calibration mode is popped because this probe measures no memory and must
    not take ownership of a peak counter it never reads; the device-count pin is
    popped because every arm pins by explicit list, and an inherited value would
    describe a mechanism this job does not use.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter, so no state carries from
    one arm to the next."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
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
    """The plan in job order: the generator, then the adjoint instrument, then
    the reconstruction arms.  Instrument 1 runs early on purpose -- if the back
    projection fails its gate the answer is already known, and the rest of the
    job is then confirmation rather than adjudication."""
    plan = [dict(mode="generate", arm_id="generate", n_dev=1)]
    for n in COUNTS_IN_USE:
        plan.append(dict(mode="backproject", arm_id=f"backproject_n{n}", n_dev=n))
    for n in COUNTS_IN_USE:
        plan.append(dict(mode="repeat", arm_id=f"repeat_n{n}", n_dev=n))
    for n in COUNTS_IN_USE:
        plan.append(dict(mode="recon_long", arm_id=f"recon_long_n{n}", n_dev=n))
    return plan


WORKERS = dict(generate=generate, backproject=run_backproject,
               repeat=run_repeat, recon_long=run_recon_long)


def _dry_run(plan):
    arms = [c for c in plan if c["mode"] != "generate"]
    print(f"mg15b multiaxis probe: {len(arms)} arms "
          f"({len(plan) - len(arms)} generator), device {DEVICE}, cell "
          f"{tuple(CELL_IN_USE)}, counts {list(COUNTS_IN_USE)}")
    print(f"  adjudicating mg15's multiaxis n=4 reading "
          f"{MG15_READINGS[0][1]:.3e} against the 1e-2 backstop ({MG15_JOB})")
    for cfg in plan:
        note = {"generate": "stage the phantom and its one-device sinogram",
                "backproject": f"instrument 1: one back projection, gate "
                               f"{BACKPROJECT_GATE_REL:g}",
                "repeat": f"instrument 2: two {ITERS_SHORT}-iteration passes "
                          f"in one process (also stages the "
                          f"{ITERS_SHORT}-iteration volume)",
                "recon_long": f"instrument 3: one {ITERS_LONG}-iteration pass"}
        print(f'  {cfg["arm_id"]:<18} n={cfg["n_dev"]}  {note[cfg["mode"]]}')
    print(f"  instrument 4 (boundary concentration) is computed by the driver "
          f"from the staged {ITERS_SHORT}-iteration volumes; no arm of its own")
    print(f"exit code is INSTRUMENT HEALTH only -- 0 if every arm ran and every "
          f"comparison computed.  The adjudication is read from the table.")


def main():
    plan = build_plan()
    if "--dry-run" in sys.argv:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, f"mg15b_multiaxis_probe_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg15b multiaxis probe on {RUN_LABEL} ({DEVICE}); cell "
          f"{tuple(CELL_IN_USE)}, counts {list(COUNTS_IN_USE)} -> {out_path}",
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
    if os.environ.get("MG15B_KEEP_ARTIFACTS", "0") != "1":
        paths = [_sino_path(), _phantom_path()]
        for n in COUNTS_IN_USE:
            paths.append(_backproj_path(n))
            for iters in (ITERS_SHORT, ITERS_LONG):
                paths.append(_recon_path(n, iters))
        for path in list(paths):
            paths.append(_md5_path(path))
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
    else:
        print(f"MG15B_KEEP_ARTIFACTS=1: the staged volumes are left in "
              f"{RESULTS_DIR}")
    print(f"\nwrote {out_path}")
    return 0 if summary["instruments_healthy"] else 2


# ── instrument 4: where in the volume the difference lives ────────────────────
def boundary_report(profile, slice_blocks, label, top=10, span=3):
    """Print where a difference volume's error sits along the sharded axis.

    Two views of the same profile.  The largest slices say whether the error
    has a few hot spots or is spread everywhere; the slices straddling each
    block boundary say whether those hot spots are AT the seams.  A split
    defect can only put them at the seams -- that is the one place a block
    boundary exists -- so error at the seams is decisive and error spread
    through the interior is decisive the other way.
    """
    if not profile:
        print(f"  {label}: no profile (the two volumes did not match in shape)")
        return None
    order = sorted(range(len(profile)), key=lambda i: profile[i], reverse=True)
    peak = profile[order[0]] if profile else 0.0
    print(f"  {label}: {len(profile)} slices, blocks {slice_blocks}, peak "
          f"{peak:.6g}")
    print(f"    top {top} slices by max|diff|: " +
          ", ".join(f"s{i}={profile[i]:.3e}" for i in order[:top]))
    # The interior boundaries: the last slice index of every block but the last.
    bounds, running = [], 0
    for length in (slice_blocks or [])[:-1]:
        running += int(length)
        bounds.append(running - 1)
    if not bounds:
        print("    no interior block boundary at this device count")
        return dict(peak=peak, top_slices=[[i, profile[i]] for i in order[:top]],
                    boundaries=[], boundary_windows=[], boundary_rank=None)
    windows = []
    for b in bounds:
        lo, hi = max(0, b - span + 1), min(len(profile), b + span + 1)
        window = [[i, profile[i]] for i in range(lo, hi)]
        windows.append(dict(boundary_slice=b, window=window))
        cells = ", ".join(f"{'>' if i == b else ''}s{i}={profile[i]:.3e}"
                          for i, _v in window)
        print(f"    around boundary after slice {b}: {cells}")
    # The plainest single number: how high the boundary slices rank among all
    # slices.  If the seams were the problem they would be at the very top.
    rank = {b: order.index(b) for b in bounds}
    print(f"    rank of each boundary slice among all {len(profile)} slices "
          f"(0 = largest): " + ", ".join(f"s{b}->#{r}" for b, r in rank.items()))
    return dict(peak=peak, top_slices=[[i, profile[i]] for i in order[:top]],
                boundaries=bounds,
                boundary_windows=windows,
                boundary_rank={str(b): r for b, r in rank.items()})


def _reading(label, check, note=""):
    """One line of the summary table, tolerant of a comparison that failed to
    compute."""
    if not check or check.get("rel") is None:
        return (f'  {label:<34} {"n/a":>11} {"n/a":>11}  '
                f'{(check or {}).get("reason", "not computed")}')
    verdict = ""
    if check.get("gate") is not None:
        verdict = "PASS" if check.get("ok") else "FAIL"
        verdict = f'{verdict} at {check["gate"]:g}'
    nrmse = check.get("nrmse")
    return (f'  {label:<34} {check["rel"]:>11.3e} '
            f'{("n/a" if nrmse is None else f"{nrmse:.3e}"):>11}  '
            f'{verdict}{" " if verdict and note else ""}{note}')


def summarize(rows, out_path):
    import numpy as np

    print(f"\n===== mg15b: adjudicating mg15's multiaxis n={COUNTS_IN_USE[-1]} "
          f"reconstruction reading ({out_path}) =====")
    by_id = {r.get("arm_id"): r for r in rows}
    n_ref, n_test = COUNTS_IN_USE[0], COUNTS_IN_USE[-1]
    problems = []

    # ── instrument health first: which arms can be believed ──────────────────
    print("\nARMS")
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'  {arm_id:<18} ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            problems.append(f"{arm_id}|error")
            continue
        if row.get("mode") == "generate":
            # The generator pins to one device and stages bytes; it has no
            # split to report and no comparison to contribute.
            sino = row.get("sino") or {}
            print(f'  {arm_id:<18} staged recon shape '
                  f'{row.get("recon_shape")}, sinogram md5 '
                  f'{str(sino.get("md5"))[:12]}')
            continue
        blocks = row.get("blocks") or {}
        walls = row.get("recon_s") or ([row["back_project_s"]]
                                       if row.get("back_project_s") else [])
        print(f'  {arm_id:<18} on {row.get("realized_devices")}  '
              f'slices {blocks.get("slice_blocks")}'
              f'{"" if blocks.get("slices_divide", True) else " (uneven)"}  '
              f'{"".join(f"{w:.1f}s " for w in walls)}')
        if row.get("devices_ok") is False:
            print(f'    ARM CHECK FAIL: realized {row.get("realized_devices")} '
                  f'for n={row.get("n_dev")}')
            problems.append(f"{arm_id}|devices")
        for label in ("sino", "phantom"):
            if row.get(f"{label}_md5_ok") is False:
                print(f"    ARM CHECK FAIL: staged {label} checksum mismatch")
                problems.append(f"{arm_id}|{label}_md5")

    def staged(arm_id):
        row = by_id.get(arm_id) or {}
        entry = row.get("staged") or {}
        path = entry.get("path")
        return path if path and os.path.exists(path) else None

    # ── instrument 1: the adjoint ────────────────────────────────────────────
    back = None
    ref_path, test_path = staged(f"backproject_n{n_ref}"), staged(f"backproject_n{n_test}")
    if ref_path and test_path:
        back = compare_arrays(np.load(test_path, mmap_mode="r"),
                              np.load(ref_path, mmap_mode="r"),
                              gate=BACKPROJECT_GATE_REL)
    else:
        problems.append("instrument1|missing volume")

    # ── instrument 2: the same-count floors ──────────────────────────────────
    floors = {}
    for n in COUNTS_IN_USE:
        row = by_id.get(f"repeat_n{n}") or {}
        floors[n] = row.get("same_count")
        if not floors[n] or floors[n].get("rel") is None:
            problems.append(f"instrument2|n{n} not computed")

    # ── instrument 3: three iterations against ten ───────────────────────────
    cross = {}
    for iters in (ITERS_SHORT, ITERS_LONG):
        a, b = staged_pair = (staged_recon(by_id, n_ref, iters),
                              staged_recon(by_id, n_test, iters))
        if a and b:
            cross[iters] = compare_arrays(np.load(b, mmap_mode="r"),
                                          np.load(a, mmap_mode="r"), gate=None)
        else:
            cross[iters] = None
            problems.append(f"instrument3|{iters} iterations missing a volume")
        del staged_pair

    # ── the summary table ────────────────────────────────────────────────────
    print(f"\n{'READING':<36}{'rel':>11}{'nrmse':>12}  verdict / note")
    print("  " + "-" * 74)
    print(f"  -- mg15, {MG15_JOB} --")
    for label, rel, nrmse, note in MG15_READINGS:
        print(f'  {label:<34} {rel:>11.3e} '
              f'{("n/a" if nrmse is None else f"{nrmse:.3e}"):>11}  {note}')
    print(f"  -- mg15b, this run --")
    print(_reading(f"1. back projection n{n_test} vs n{n_ref}", back,
                   "THE ADJOINT INSTRUMENT mg15 LACKED"))
    for n in COUNTS_IN_USE:
        print(_reading(f"2. same-count floor n={n} ({ITERS_SHORT} iter)",
                       floors.get(n), "run-to-run only, not gated"))
    print(_reading(f"3. n{n_test} vs n{n_ref}, {ITERS_SHORT} iterations",
                   cross.get(ITERS_SHORT), "compare with mg15's 2.969e-02"))
    print(_reading(f"3. n{n_test} vs n{n_ref}, {ITERS_LONG} iterations",
                   cross.get(ITERS_LONG), "decay -> trajectory; growth -> defect"))
    print("  " + "-" * 74)

    # ── the three things a reader should weigh, stated as ratios ─────────────
    print("\nHOW TO READ THE TABLE")
    if back and back.get("rel") is not None:
        print(f'  1. the adjoint {"PASSES" if back.get("ok") else "FAILS"} at '
              f'{BACKPROJECT_GATE_REL:g} ({back["rel"]:.3e}).  '
              + ("A single back projection reproduces the one-device answer, "
                 "so the sharded adjoint places its blocks correctly; a "
                 "misplaced block would land near one."
                 if back.get("ok") else
                 "A SINGLE back projection already disagrees.  That cannot be "
                 "trajectory divergence -- there is no trajectory in one "
                 "linear operator -- so this is a split defect and the "
                 "remaining instruments are confirmation."))
    floor_test = (floors.get(n_test) or {}).get("rel")
    mg15_rel = MG15_READINGS[0][1]
    if floor_test:
        ratio = mg15_rel / floor_test if floor_test > 0 else float("inf")
        print(f"  2. mg15's {mg15_rel:.3e} is {ratio:.1f}x the same-count "
              f"floor at n={n_test} ({floor_test:.3e}).  "
              + ("Within an order of magnitude: the reading is "
                 "noise-floor-dominated and constrains nothing about the split."
                 if ratio <= 10 else
                 "More than an order of magnitude above the floor, so the "
                 "cross-count difference is not merely run-to-run noise -- "
                 "read it together with instruments 1 and 3."))
    short, long = cross.get(ITERS_SHORT), cross.get(ITERS_LONG)
    if short and long and short.get("rel") and long.get("rel"):
        factor = long["rel"] / short["rel"]
        print(f'  3. from {ITERS_SHORT} to {ITERS_LONG} iterations the '
              f'cross-count difference moves {short["rel"]:.3e} -> '
              f'{long["rel"]:.3e} ({factor:.2f}x).  '
              + ("It DECAYS, which is the documented signature of two "
                 "trajectories converging: a systematic split error is "
                 "re-injected every iteration and cannot decay."
                 if factor < 1.0 else
                 "It does NOT decay.  Trajectory divergence is documented to "
                 "fall over iterations, so a flat or growing difference points "
                 "at something systematic."))
        print(f"     for reference, {DOC_DECAY}")

    # ── instrument 4: where the difference lives ─────────────────────────────
    print(f"\nINSTRUMENT 4 -- BOUNDARY CONCENTRATION (report only)")
    print(f"  READ THIS FOR CHANGE, NOT FOR SHAPE.  Every cross-count "
          f"difference is seeded at the seams -- the halo exchange and the "
          f"per-shard\n  reductions live there -- so ANY difference clusters "
          f"at the boundaries before iterations spread it inward.  This "
          f"probe's own\n  CPU smoke shows the boundary slices ranked #0 and "
          f"#3 of 27 on a split that is correct by every other instrument.  "
          f"What\n  separates the two cases is whether the boundary peak "
          f"DECAYS with the whole difference or holds while the interior "
          f"falls away.")
    blocks_test = ((by_id.get(f"repeat_n{n_test}") or {}).get("blocks")
                   or {}).get("slice_blocks")
    profiles, null_entry = {}, None
    for iters in (ITERS_SHORT, ITERS_LONG):
        a, b = staged_recon(by_id, n_ref, iters), staged_recon(by_id, n_test, iters)
        if a and b:
            profiles[iters] = boundary_report(
                slice_profile(np.load(b, mmap_mode="r"),
                              np.load(a, mmap_mode="r")),
                blocks_test, f"n{n_test} vs n{n_ref} at {iters} iterations")
        else:
            profiles[iters] = None
            print(f"  the {iters}-iteration difference volume is not available")
            problems.append(f"instrument4|missing volume at {iters} iterations")
    profile_entry = profiles.get(ITERS_SHORT)
    # The reading that actually discriminates: how the boundary peak moved
    # relative to how the whole difference moved.  A ratio near or below the
    # global one means the seam-seeded difference is settling with everything
    # else; a ratio well above it means the seams are holding on to error the
    # interior has shed.
    short_p, long_p = profiles.get(ITERS_SHORT), profiles.get(ITERS_LONG)
    if short_p and long_p and short_p.get("peak") and short and long \
            and short.get("rel") and long.get("rel"):
        peak_factor = long_p["peak"] / short_p["peak"]
        global_factor = long["rel"] / short["rel"]
        print(f"\n  boundary peak {short_p['peak']:.3e} -> "
              f"{long_p['peak']:.3e} ({peak_factor:.2f}x) while the whole "
              f"difference moved {global_factor:.2f}x.")
        # Three cases, kept separate because the middle one is the interesting
        # one and folding it into either neighbour would hide it.
        if peak_factor >= 1.0:
            verdict = ("the peak HELD OR GREW while iterations proceeded.  A "
                       "seam-seeded trajectory settles; an error re-injected "
                       "at the seam every iteration does not.  Systematic "
                       "signature -- weigh this against instrument 1.")
        elif global_factor > 0 and peak_factor > 2.0 * global_factor:
            verdict = (f"the peak decayed, but {peak_factor / global_factor:.1f}x "
                       f"more slowly than the difference as a whole: the "
                       f"interior is settling faster than the seams.  Not the "
                       f"clean trajectory signature; worth a closer look.")
        else:
            verdict = ("the peak is decaying with the rest: a seam-seeded "
                       "trajectory settling as the two runs converge.")
        print(f"    {verdict}")
    null_row = by_id.get(f"repeat_n{n_ref}") or {}
    null_profile = null_row.get("same_count_slice_profile")
    if null_profile:
        # The null reference: a same-count repeat at ONE device has no block
        # boundary anywhere, so whatever shape its interior noise takes is the
        # shape that means "nothing to see here".
        null_entry = boundary_report(
            null_profile, (null_row.get("blocks") or {}).get("slice_blocks"),
            f"NULL REFERENCE: n{n_ref} same-count repeat")
    else:
        problems.append("instrument4|null profile missing")

    healthy = not problems
    print(f"\n===== INSTRUMENT HEALTH: {'OK' if healthy else 'DEGRADED'} =====")
    if problems:
        print(f"  {len(problems)} problem(s): {', '.join(problems)}")
        print("  Readings above may be incomplete; the adjudication needs "
              "every instrument.")
    else:
        print("  every arm ran and every comparison computed; the readings "
              "above are the adjudication")
    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"  GPU health: {len(hot)} row(s) sampled hot: {hot}")
    print(f"  exit code   {0 if healthy else 2}")
    return dict(instruments_healthy=healthy, problems=problems,
                back_projection=back,
                same_count_floors={str(k): v for k, v in floors.items()},
                cross_count={str(k): v for k, v in cross.items()},
                boundary=profile_entry,
                boundary_long=profiles.get(ITERS_LONG),
                boundary_null=null_entry,
                mg15_readings=[list(r) for r in MG15_READINGS],
                mg15_job=MG15_JOB, hot=hot)


def staged_recon(by_id, n_dev, iters):
    """The staged reconstruction volume for one device count and iteration
    count, from whichever arm produced it."""
    arm_id = (f"repeat_n{n_dev}" if iters == ITERS_SHORT
              else f"recon_long_n{n_dev}")
    entry = (by_id.get(arm_id) or {}).get("staged") or {}
    path = entry.get("path")
    return path if path and os.path.exists(path) else None


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
