"""mg15 -- THE CLUSTER GATE FOR THE SHARDING-PAD REMOVAL: values, memory and
one re-measured ledger row, on real GPUs at device counts that do not divide
the sharded axes.

WHY THIS RUN EXISTS.

mbirtorch used to shard by padding.  An axis that was split across devices was
first rounded UP to a multiple of the device count, every device got an equal
block, and the padding was carried through the arithmetic and thrown away at
the end.  That is gone.  A sharded axis is now split into contiguous blocks
whose lengths differ by at most one, with the longer blocks first --
numpy.array_split's convention -- so no device holds padding and the blocks are
no longer all the same length.

The local test suite already covers the split itself and the values it
produces: it runs the geometries on virtual CPU devices at counts that do not
divide the axes and checks the reconstruction against the single-device answer.
What a laptop cannot show is the thing the change was made for.  This job adds
exactly that, and nothing else:

  (a) VALUES.  Real multi-GPU reconstructions at counts that do not divide the
      sharded axes, compared against the single-device reference on the same
      seed.  The uneven blocks now reach the hand-written (Triton) kernels, the
      cross-device transfers and the row-aligned parallel-beam path, none of
      which the CPU tests exercise.

  (b) MEMORY.  Per-device peak memory against the library's own memory ledger.
      The ledger prices a device's share of an axis, and the share it prices is
      now the block that device actually owns rather than a padded block that
      is the same on every device.  The ledger's one hard rule is that it may
      over-estimate but must never UNDER-estimate, because an under-estimate
      lets a doomed run start and then die inside the allocator.  This job
      checks that floor on the new split.

  (c) ONE LEDGER ROW.  The measured arm 'ma512_n4' in tests/test_memory_ledger.py
      MEASURED_ARMS was recorded on the PADDED split, and its split changed:
      510 recon slices over four devices used to be padded to 512 and cut into
      four blocks of 128, and are now cut into 128, 128, 127, 127.  The row is
      therefore stale as a record of what that configuration peaks at.  This
      job re-measures it on the new split and prints the replacement line ready
      to paste.

THE NON-DIVIDING ARITHMETIC, stated explicitly so the point of each arm is not
left to be re-derived.  The sinogram shards by VIEW and the reconstruction
shards by SLICE, so both axes have to be checked:

    cone       512 views % 3 = 2   and  448 recon slices % 3 = 1
    parallel   512 views % 3 = 2   and  448 recon slices % 3 = 1
    multiaxis  512 views % 4 = 0   and  510 recon slices % 4 = 2

Under the old padding every one of those axes was rounded up and split evenly.
Under the new split the cone and parallel arms hand two devices a 171-view
block and one a 170-view block, and one device a 149-slice block against two
with 150; the multiaxis arm divides evenly on views and unevenly on slices,
which is the axis its stale ledger row was priced on.  Parallel beam is the
row-aligned geometry (rows_track_slices is True: detector row r is recon slice
r), so at n=3 its DETECTOR-ROW axis shards unevenly right along with the
slices -- that is the path that used to carry the detector-row pad, and it has
no counterpart in cone or multiaxis.

TERMS USED BELOW, defined once here:
    arm            one measured configuration: one geometry family, one device
                   count, run in its own fresh process.
    family         cone, parallel or multiaxis: one geometry, one shape, and
                   the device counts it is measured at.
    sinogram       the projection data, shaped
                   (views, detector rows, detector channels).
    recon shape    the reconstruction volume, (rows, columns, slices).
    modeled peak   the ledger's estimate of peak bytes on one device.
    measured peak  torch.cuda.max_memory_allocated on that device, which is
                   CUDA-only -- there is no such counter on CPU.
    ratio          modeled / measured on one device.  At or above 1.00 the
                   ledger covered the run; below 1.00 it did not.
    dominant phase the phase whose per-device bytes set the modeled peak.
    reference      the n=1 reconstruction of a family, which every multi-device
                   reconstruction of that family is compared against.

THE VALUE GATE, AND WHY IT HAS TWO PARTS.  Every arm of a family works from the
same staged phantom and the same staged sinogram, with np.random.seed(12345)
called immediately before recon, so every arm draws the SAME pixel partitions
and the comparison is between device layouts and nothing else.  Both parts use
the same statistic, computed in float64:

    rel = max|out - ref| / max|ref|

    (a) PROJECTOR, gate 1e-4.  The arm forward-projects the staged phantom on
        its own device layout and compares against the staged sinogram, which
        is that same phantom projected on ONE device.  This is a single
        application of a linear operator: no iteration, no line search, no
        partition draw, so the only spread is float32 accumulation order.
        This is the part that resolves the split.  It is taken AFTER the
        reconstruction, for a reason worth stating where it cannot be missed:
        on CUDA the device count is pinned through MBIRTORCH_NUM_DEVICES, that
        pin acts only through the model's device policy, and forward_project
        never calls the policy -- so before a reconstruction has settled the
        layout, a freshly built automatic model still holds the trivial
        single-device placement.  Projecting first would have projected on one
        device at every device count and compared the result against a
        one-device sinogram, passing at ~1e-7 while gating nothing.  Every arm
        therefore records the device list its projection actually ran on, and
        a projection that did not run on the arm's own device count makes the
        arm invalid.

    (b) RECONSTRUCTION, gate 1e-2.  The arm's 3-iteration reconstruction
        against the n=1 arm's.  This is a gross-error backstop, and the loose
        threshold is not slack -- it is where the noise floor of this
        particular measurement actually sits.

WHY THE TWO THRESHOLDS DIFFER BY TWO ORDERS OF MAGNITUDE, measured rather than
assumed (ablations on virtual CPU devices, 2026-08-16, parallel/cone/multiaxis,
cells from (16, 14, 12) to (128, 112, 96), device counts 1 to 4):

  * The FORWARD PROJECTION reproduces the single-device answer to 1.0e-7 to
    1.9e-7 at every device count tested, and the reading does not move when the
    blocks stop being equal: at cell (64, 56, 48) parallel reads 1.100e-7 at
    n=2 (both axes even), 1.100e-7 at n=3 (both axes uneven) and 1.100e-7 at
    n=4.  A 1e-4 gate therefore sits three orders of magnitude above this
    noise floor, and a misplaced block boundary -- a dropped view, a duplicated
    slice, an off-by-one seam -- moves whole planes of the sinogram and lands
    at order one.  That is a real gate with real resolving power.

  * The 3-ITERATION RECONSTRUCTION does not behave that way, and a gate at 1e-4
    on it would be below its own noise floor.  A rerun at the SAME device count
    reproduces to about 1e-7, but any change of device count moves the answer
    by 1e-4 to 2e-3, and the size of that move is set by the DEVICE COUNT and
    the problem size, not by whether the blocks are equal.  Two decisive
    readings, both of which put the EVEN split on the worse side:

        cell (48, 54, 48)  n=3 divides both axes exactly    9.8e-4
        cell (49, 55, 48)  n=3 divides neither axis         7.8e-4

        cell (128, 112, 96)  n=2 even  1.97e-3   n=4 even  1.97e-3
                             n=3 UNEVEN on both axes        1.19e-3

    VCD is a nonlinear iterative optimizer with a line search whose sums are
    combined across shards on the host in a different order for each device
    count; a different alpha at the first iteration puts the run on a slightly
    different trajectory, and three iterations amplify a 1e-7 difference to
    1e-3.  It is not a split error, and no reachable threshold below 1e-3 can
    tell the two apart.

WHY THE RECONSTRUCTION GATE IS 1e-2 AND NOT TIGHTER.  The divergence above also
GROWS with problem size, because a maximum over more voxels of a diverging
trajectory is a larger maximum: 1.7e-4 at the (16, 14, 12) cell, 4.1e-4 at
(64, 56, 48), 2.0e-3 at (128, 112, 96).  The production cell here is larger
again, so the max-norm statistic this gate uses is NOT scale-stable and no
threshold measured at small sizes transfers to it exactly.  1e-2 is set five
times above the worst reading measured and roughly an order of magnitude below
what any real split fault produces -- a misplaced block boundary moves whole
slices and lands near one, not near a hundredth.  Treat a reconstruction
reading between 2e-3 and 1e-2 as expected at production scale, not as a
near-miss.  Every comparison also records a normalized RMS beside the maximum,
which is far less sensitive to problem size; it is reported and not gated,
because there is no measured threshold for it yet, and it is exactly the number
a later run would use to set a tighter, scale-stable gate.

So the fine-grained gate is on the projector, where it discriminates, and the
reconstruction is a gross-error backstop.  Reading the two together is what
makes the leg meaningful: a correct split passes both, a misplaced block fails
the projector immediately, and a reconstruction that went wrong while the
projector stayed right fails the backstop.

A NOTE ON WHAT THIS CHANGES ABOUT THE JOB'S CLAIM.  This job can state that the
sharded projectors reproduce the single-device projector on non-dividing splits
to within 1e-4, and that the reconstructions agree to the scale multi-device
reconstructions agree to anyway.  It cannot state that the reconstruction is
bit-reproducible across device counts, because it is not, and it was not before
the padding was removed either.

WHAT WOULD MAKE THIS JOB FAIL, and it is a gate, so failing means something:
either half of the value gate above its threshold, any device of any arm whose
modeled peak sits below its measured peak, or any arm that cannot be believed
at all (wrong realized device count, missing calibration environment, staged
sinogram or phantom checksum mismatch, unexpected recon shape, or a CUDA arm
that did not take the automatic device layout).  The BAND TOP -- an over-estimate larger than the ledger's declared
band allows -- is reported as a WARNING and does not fail the job: over-charging
spreads a run wider than it needed, which is worth knowing but is not the
failure the ledger exists to prevent.

The exit code IS the verdict: 0 when all three legs pass, 2 otherwise.  The job
script chains a follow-on measurement job with --dependency=afterok, so a
failure here stops that job rather than letting it measure on top of a split
that does not reconstruct correctly.

THE ARMS.  6 measured arms plus 3 untimed generators.

    cone       (512, 448, 384) -> recon (384, 384, 448)   at 1 and 3 devices
    parallel   (512, 448, 384) -> recon (384, 384, 448)   at 1 and 3 devices
    multiaxis  (512, 448, 384) -> recon (384, 384, 510)   at 1 and 4 devices

The n=1 arm of each family is the reference the multi-device arm is judged
against, so it is a measured arm in its own right and not a warm-up.  The
multiaxis arm is EXACTLY mg8's ma512 arm -- same cell, same angles, same dots
phantom, same seed, weighted, three iterations, one pass is the reading -- so
that the row it produces is a replacement for the recorded one and not a
different measurement wearing the same name.  Its count is 4 and not 3 for the
same reason: the recorded row is a four-device row.

Every arm supplies weights, computed from its own sinogram, because the
recorded ma512_n4 row was measured weighted and an unweighted run holds a
different resident array.

HOW THE MEASUREMENT IS TAKEN.  One arm per subprocess, so no allocator state
carries from one arm to the next.  On CUDA the device count is pinned by
setting MBIRTORCH_NUM_DEVICES in that subprocess's environment (popped first,
then set, so nothing is inherited), and the arm asserts the device list the
model actually realized.  Pinning by environment rather than by an explicit
configure_devices call matters: an explicit device list turns off the automatic
device-count search, and the ledger is what that search consumes, so the
measurement has to be taken on the branch that uses it.  The CPU smoke cannot
use that mechanism -- the automatic search short-circuits below two visible
CUDA devices -- so it pins by explicit device list instead, and every row
records which mechanism it used.

MBIRTORCH_MEMORY_CALIBRATION=1 is set for the measurement arms.  That mode
computes the ledger at any device count including one, resets
torch.cuda.max_memory_allocated at the start of the reconstruction, and reads
it at the end.  Because it RESETS that counter it owns it, which is why it is
off by default and why the three generator subprocesses do not get it.

ONE reconstruction per arm, and that reconstruction is the reading -- memory,
wall and values all come from the same pass, compilation included, because a
production run pays that compilation exactly once and the ledger is supposed to
cover the peak a production run actually reaches.  This matches mg8, so the
ma512_n4 row this job prints is measured the same way the row it replaces was.

That match is exact, and the ordering above is what makes it exact: mg8 did no
projection before its reconstruction either, so from process start to the
moment the calibration rows are read off the model, this arm does what mg8's
ma512_n4 arm did, allocation for allocation.  The projector leg runs only after
those rows are in hand, where nothing it allocates can reach them.

WHAT IS NOT MEASURED HERE.  This job does not time the split against the padded
one: the padding is already gone, so there is no padded arm to time against,
and speed was never the reason for the change.  It does not sweep device counts
either -- each family is measured at one non-dividing count and at one, which
is what a value comparison needs.  And it does not re-measure mg8's other
eleven rows: only ma512_n4's split changed shape in a way that moves its
memory, and re-measuring the rest would cost hours of GPU time to reproduce
numbers that did not move.

ARTIFACTS.  Per family, one phantom and one sinogram, written once, checksummed
and re-verified by both arms that read them (0.33 GiB of sinogram and 0.26-0.29
GiB of phantom each, about 1.8 GiB in total), plus one reconstruction volume
per arm for the backstop (0.26 GiB per cone/parallel arm, 0.29 GiB per
multiaxis arm, about 1.6 GiB in total).  About 3.4 GiB at once.  All of it is
deleted at the end unless MG15_KEEP_ARTIFACTS=1 is exported.  The jsonl is
always kept.

The phantom is staged rather than regenerated inside each arm because the
projector leg needs BOTH arms to project the identical volume, and the dot
phantom is a random draw; staging it makes that identity a checksum rather than
an assumption about reproducible random state across processes.

Run:
    <torch python> mg15_p5_gate.py       on a 4-GPU node (mg15_gautschi.sbatch)
    python mg15_p5_gate.py --dry-run     anywhere: print the arm plan
    python mg15_p5_gate.py --help

Environment (export from the SUBMITTING SHELL, never through an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed strictly: an unrecognized token is an error, not a silent skip.
    MG15_RESULTS=<dir>                    where the jsonl and the artifacts go
    MG15_FAMILIES=cone,parallel,multiaxis subset of the families
    MG15_ITERATIONS=3                     reconstruction iterations per arm
    MG15_KEEP_ARTIFACTS=1                 keep the sinograms and the volumes
    MG15_SMOKE=1 / MG15_DEVICE=cpu        the local CPU smoke

The device counts are NOT an environment knob.  Every leg of this gate compares
a multi-device arm against the n=1 arm of the same family in the same job, so
the count axis may not be narrowed or split across jobs.  MG15_FAMILIES exists
to re-run one family, which keeps both of its arms together.
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
# Each family carries everything needed to rebuild its model, plus the recon
# shape and pixel count it produced when this file was written (mbirtorch on
# greg_dev at dba9652, 2026-08-16).  The arm re-derives both and flags a
# mismatch: a change in either would move every number in the table, and it
# would also change which axis lengths are being split, which is the whole
# subject of this job.
#
# All three families share the (512, 448, 384) cell.  Cone and parallel
# reconstruct into (384, 384, 448) -- 448 slices, one per detector row -- while
# multiaxis reconstructs into (384, 384, 510), because its swept elevation
# needs more slices to cover the same detector height.  The three grids are the
# same 384 x 384 disc, so all three carry 115164 masked pixels.
#
# 'band' names which of the library's two acceptance bands a family is judged
# against.  Cone and parallel have hand-written kernels that declare their own
# per-view cost, so they are priced by a real measurement and judged against
# the tight CALIBRATION_BAND.  Multiaxis has no kernels: it runs on general
# torch code, priced by the measured slab count, and the library judges that
# against the wider TORCH_BODY_CALIBRATION_BAND.
FAMILIES = (
    dict(name="cone", cell=(512, 448, 384), recon_shape=(384, 384, 448),
         num_pixels=115164, counts=(1, 3), band="CALIBRATION_BAND"),
    dict(name="parallel", cell=(512, 448, 384), recon_shape=(384, 384, 448),
         num_pixels=115164, counts=(1, 3), band="CALIBRATION_BAND"),
    dict(name="multiaxis", cell=(512, 448, 384), recon_shape=(384, 384, 510),
         num_pixels=115164, counts=(1, 4), band="TORCH_BODY_CALIBRATION_BAND"),
)

# The CPU smoke runs the same three families on virtual devices at a tiny size,
# so the harness itself -- the subprocess plumbing, the checksums, the value
# gate, the summary and the exit code -- is exercised end to end before any GPU
# time is spent.  It must exercise a NON-DIVIDING count, because a count that
# divided every axis would leave the one thing this job is about untested: at
# these shapes n=3 splits 16 views into 6, 5, 5 and 14 recon slices into 5, 5,
# 4 for cone and parallel, and 16 views into 6, 5, 5 for multiaxis.
#
# The smoke registers no recon shape.  Registering a tiny one would pin down a
# geometry default that no production run uses, and a change in it would fail
# the smoke for a reason the gate does not care about; the arm records the
# shape it realized and skips the comparison.  The registered-shape check
# applies to the real shapes, which are what the ledger rows are about.
SMOKE = os.environ.get("MG15_SMOKE", "0") == "1"
SMOKE_FAMILIES = (
    dict(name="cone", cell=(16, 14, 12), recon_shape=None, num_pixels=None,
         counts=(1, 3), band="CALIBRATION_BAND"),
    dict(name="parallel", cell=(16, 14, 12), recon_shape=None, num_pixels=None,
         counts=(1, 3), band="CALIBRATION_BAND"),
    dict(name="multiaxis", cell=(16, 24, 20), recon_shape=None,
         num_pixels=None, counts=(1, 3), band="TORCH_BODY_CALIBRATION_BAND"),
)

DEVICE = os.environ.get("MG15_DEVICE", "cpu" if SMOKE else "cuda")
VCD_ITERATIONS = int(os.environ.get("MG15_ITERATIONS", "3"))
VCD_SEED = 12345               # the seed every calibration run in this family uses

# The two halves of the value gate, both on max|out - ref| / max|ref| in
# float64.  The module docstring records the ablation these come from; the
# short version is that the projector reproduces the single-device answer to
# ~1e-7 at every device count, so 1e-4 there is a real gate, while a
# 3-iteration reconstruction moves by up to 2e-3 with the device count alone,
# so 1e-4 there would be below its own noise floor.
PROJECTOR_GATE_REL = 1e-4
RECON_GATE_REL = 1e-2

# The bands the library itself declares (mbirtorch/_memory_ledger.py,
# CALIBRATION_BAND and TORCH_BODY_CALIBRATION_BAND, read 2026-08-16).  Repeated
# here rather than imported so the harness can print the band it judged against
# even if the library's value moves underneath it; each arm records the
# library's value too, and a mismatch is reported on the row.
BANDS = {"CALIBRATION_BAND": (1.00, 1.30),
         "TORCH_BODY_CALIBRATION_BAND": (1.00, 5.80)}

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
    "MG15_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
    repeat before.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = cast(token)
        except ValueError:
            raise ValueError(f"{env_name}: unparsable token {token!r}")
        if value not in allowed:
            raise ValueError(f"{env_name}: {value!r} is not one of "
                             f"{sorted(allowed)}")
        chosen.append(value)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return chosen


def all_families():
    return SMOKE_FAMILIES if SMOKE else FAMILIES


def selected_families():
    """The families this run will measure, in declared order so the job order
    is reproducible."""
    keep = _strict_subset("MG15_FAMILIES", {f["name"] for f in all_families()})
    families = [f for f in all_families() if f["name"] in keep]
    if not families:
        raise ValueError("MG15_FAMILIES selects no family")
    return families


def _family_by_name(name):
    for spec in all_families():
        if spec["name"] == name:
            return spec
    raise KeyError(f"no family named {name!r}")


def _sino_path(name):
    return os.path.join(RESULTS_DIR, f"_mg15_sino_{name}.npy")


def _phantom_path(name):
    # Staged beside the sinogram because the projector leg needs the SAME
    # phantom the sinogram was made from.  Regenerating it in the arm would
    # work only as long as the dot phantom's random draw stayed reproducible
    # across processes, which is a dependency this measurement does not need.
    return os.path.join(RESULTS_DIR, f"_mg15_phantom_{name}.npy")


def _recon_path(name, n_dev):
    return os.path.join(RESULTS_DIR, f"_mg15_recon_{name}_n{n_dev}.npy")


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


def compare_arrays(out, ref, gate, budget_bytes=64 << 20):
    """max|out - ref| / max|ref| in float64, with a normalized RMS beside it.

    Walked in slabs along the first axis, so neither the float32 arrays nor
    their float64 promotions are ever held whole: at the production shapes a
    single float64 copy of a sinogram is 0.7 GiB, on a host that has just
    finished a multi-GPU run.  The maximum is accumulated slab by slab, which
    is exact -- a maximum of maxima is the maximum, with none of the summation
    error a norm carries.  ``out`` and ``ref`` may be memory-mapped.

    The normalized RMS is recorded but never gated: unlike the maximum it
    barely moves with problem size, so it is the statistic a later run would
    set a scale-stable threshold from, and this run's job is to produce it.
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
    return dict(ok=(rel <= gate), rel=rel, gate=gate, max_abs_diff=max_diff,
                max_abs_ref=max_ref, shape=list(ref.shape),
                nrmse=((sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else None))


def _weights(sinogram):
    """The weighting formula the other calibration runs use, so these arms are
    weighted the same way theirs are -- and in particular so the multiaxis arm
    reproduces the conditions the recorded ma512_n4 row was measured under."""
    import numpy as np

    scale = float(np.max(sinogram))
    if scale <= 0:
        raise RuntimeError("sinogram is all zeros; the phantom did not project")
    return np.exp(-sinogram / (2 * scale)).astype(np.float32)


# ── the GPU health sample ─────────────────────────────────────────────────────
# A GPU that is thermally throttled produces a valid memory reading but an
# invalid timing one, and a hot node is worth knowing about even here, because
# it usually means a neighbour job is sharing the hardware -- which can also
# move the free-memory reading the ledger's preflight takes.
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
    """Build the model for one family.

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
    num_views, channels = cell[0], cell[2]
    if spec["name"] == "cone":
        # A full turn of views, and the source distances written as multiples
        # of the detector width so the same expression builds the smoke model:
        # source-to-detector at four widths, source-to-isocenter at two, which
        # puts the object halfway between the source and the detector at a
        # magnification of two.
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            cell, angles, source_detector_dist=4.0 * channels,
            source_iso_dist=2.0 * channels)
    elif spec["name"] == "parallel":
        # Half a turn is a full parallel-beam scan: the second half repeats the
        # first, reflected.  This is the row-aligned geometry, so its detector
        # rows shard with its recon slices.
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    else:
        # Two angles per view: azimuth around the object, elevation (tilt) out
        # of the plane.  These are mg8's ma512 angles exactly -- azimuths evenly
        # spaced over half a turn, elevations swept across +/- 0.5 radians --
        # because the row this arm re-measures was recorded from them.  The
        # elevation range matters for the recon shape: the automatic geometry
        # divides the detector height by the smallest |cos(elevation)| and
        # clamps that divisor at 0.1, so a range wide enough to reach the clamp
        # would inflate the slice count roughly tenfold.  0.5 radians is far
        # from the clamp.
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            cell, np.stack([azimuth, elevation], axis=1))
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _shape_check(model, spec):
    """Did the geometry defaults produce the recon shape this file registered?

    Recorded rather than raised on the real shapes: a moved default is worth
    knowing about, but it does not make the memory reading itself wrong.  The
    smoke registers no shape, so its rows carry None and the summary skips the
    comparison instead of failing it.
    """
    realized = tuple(int(s) for s in model.get_params("recon_shape"))
    pixels = int(model.full_index_count())
    expected = spec.get("recon_shape")
    expected_pixels = spec.get("num_pixels")
    return dict(
        recon_shape=list(realized), num_pixels_full=pixels,
        recon_shape_expected=(list(expected) if expected else None),
        recon_shape_ok=(None if expected is None
                        else realized == tuple(expected)),
        num_pixels_expected=(int(expected_pixels) if expected_pixels else None),
        num_pixels_ok=(None if expected_pixels is None
                       else pixels == int(expected_pixels)))


def _block_lengths(model):
    """The block each device actually owns on the two sharded axes.

    This is the change itself, written down: under the old padding every block
    on an axis had the same length, and under the new split they differ by at
    most one.  Recorded on every arm so the jsonl says what was split rather
    than leaving it to be re-derived from the shapes.
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


def _ledger_record(model):
    """The modeled side, per device, read off the model's own ledger.

    Nothing here re-derives a number: the table reports what the production
    path computed, so a disagreement between this harness and production is
    impossible by construction.  The dominant phase and its largest terms are
    kept because they are what an under-floor reading has to be explained
    from -- a floor violation is a failure here, and a failure has to say which
    phase and which term fell short.
    """
    ledger = model.last_memory_ledger
    if ledger is None:
        return None
    dominant, top_terms = [], []
    for i in range(len(ledger.devices)):
        phase = ledger.dominant_phase(i)
        dominant.append(phase.name)
        top_terms.append([[name, int(value)]
                          for name, value in phase.dominant_terms(i, count=4)])
    return dict(devices=[str(d) for d in ledger.devices],
                modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
                dominant_phase=dominant, dominant_terms=top_terms,
                num_pixels_full=int(ledger.num_pixels_full),
                phases=[dict(name=phase.name,
                             per_device_bytes=[int(b) for b in phase.per_device])
                        for phase in ledger.phases])


# ── the worker: one arm, one process ──────────────────────────────────────────
def run_arm(cfg):
    """One (family, device count) measurement, in its own process.

    MBIRTORCH_MEMORY_CALIBRATION is set in this process's environment by the
    runner, so the mode owns the peak counter from the moment the
    reconstruction starts.  The single reconstruction is the reading for all
    three legs at once: the memory counters, the wall clock and the volume the
    value gate compares.
    """
    import numpy as np
    import torch

    from mbirtorch import _memory_ledger

    spec = _family_by_name(cfg["family"])
    n_dev = cfg["n_dev"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = cfg.get("cpu_devices") if not cuda else None
    if not cuda and pin_devices is None:
        pin_devices = [DEVICE]

    model = _build_model(spec, pin_devices=pin_devices)
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    # This is the one job class where the calibration mode is expected to be on.
    result["calibration_env_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1")
    # The band this family is judged against, beside the library's own value,
    # so a band that moved underneath the harness is visible on the row.
    band_name = spec["band"]
    library_band = tuple(float(x) for x in getattr(_memory_ledger, band_name))
    result["band_name"] = band_name
    result["band"] = list(BANDS[band_name])
    result["library_band"] = list(library_band)
    result["band_matches_library"] = (library_band == BANDS[band_name])

    result.update(_shape_check(model, spec))

    # Both staged artifacts are re-verified here.  Two arms and one generator
    # read these bytes, and a comparison against a file that changed underneath
    # the run would be a silently wrong answer rather than a loud one.
    sino_path, phantom_path = _sino_path(spec["name"]), _phantom_path(spec["name"])
    for label, path in (("sino", sino_path), ("phantom", phantom_path)):
        with open(_md5_path(path)) as handle:
            expected_md5 = handle.read().strip()
        actual_md5 = _md5(path)
        result[f"{label}_md5"] = actual_md5
        result[f"{label}_md5_ok"] = (actual_md5 == expected_md5)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"staged {label} checksum mismatch at {path}: "
                               f"{actual_md5} != {expected_md5}")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)

    health = [sample_gpu_health()]

    # The seed goes here, immediately before recon and inside this process, so
    # every arm of a family draws the same pixel partitions from the same
    # global generator state.  Without it the value gate would be comparing two
    # different partition sequences and would mean nothing.
    np.random.seed(VCD_SEED)
    start = time.perf_counter()
    # logfile_path=None keeps the results directory free of log files, and
    # print_logs=False keeps the subprocess's output to the one result line.
    # Nothing is lost: the calibration rows the library would print are read
    # straight off the model below.
    recon, _info = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0,
                               logfile_path=None, print_logs=False)
    if cuda:
        # The peak counters and the wall clock are both read after every
        # device has finished, not after the last one was queued.
        for device in model.sino_placement.devices:
            torch.cuda.synchronize(device)
    result["recon_s"] = time.perf_counter() - start
    health.append(sample_gpu_health())

    result["calibration"] = [
        dict(device=str(device), modeled_bytes=int(modeled),
             measured_bytes=int(measured), ratio=float(ratio))
        for device, modeled, measured, ratio
        in (model.last_memory_calibration or [])]
    if not result["calibration"]:
        # The measured side is torch.cuda.max_memory_allocated, which exists
        # only on CUDA.  Say so, so an empty column is never read as a pass.
        result["calibration_skipped_reason"] = (
            "torch.cuda.max_memory_allocated is CUDA-only and this arm ran on "
            f"{DEVICE}; the modeled column is still real, the measured column "
            "does not exist here")

    # ── the projector leg, taken AFTER the reconstruction ────────────────────
    # One forward projection of the staged phantom on this arm's device layout,
    # against the staged sinogram, which is that phantom projected on ONE
    # device.
    #
    # THE ORDER HERE IS LOAD-BEARING, and the obvious order is the wrong one.
    # On CUDA this arm pins its device count through MBIRTORCH_NUM_DEVICES, and
    # that pin acts only through the model's device policy -- which
    # forward_project never calls.  It projects on whatever placement the model
    # currently holds, and a freshly built automatic model holds the trivial
    # single-device one until a reconstruction settles the layout.  Projecting
    # before the reconstruction would therefore have run on ONE device at every
    # device count and compared the result against a one-device sinogram: a
    # self-comparison that passes at ~1e-7 while appearing to gate the split.
    # Taken here, after recon has settled the pinned layout, the projection
    # inherits the real n-device placement.  The CPU smoke cannot expose that
    # trap, because it pins by explicit device list and so has its placement
    # from the moment the model is built -- which is exactly why the realized
    # device list of the projection is recorded below and checked, rather than
    # assumed.
    #
    # The memory reading is safe at this point for a stronger reason than the
    # old order gave: the calibration mode reset the peak counter at the start
    # of the reconstruction and it has already been read off the model above,
    # so nothing this projection allocates can reach that reading.
    phantom = np.load(phantom_path)
    projected = _to_numpy(model.forward_project(phantom))
    # The placement the projection actually ran on, read from the model right
    # after it and recorded as its own field rather than folded into the
    # reconstruction's.  If this ever falls back to one device the arm is
    # invalid and says so, instead of reporting a comparison of a volume
    # against itself.
    projection_devices = [str(d) for d in model.sino_placement.devices]
    result["projection_devices"] = projection_devices
    result["projection_n_devices"] = len(projection_devices)
    result["projection_devices_ok"] = (len(projection_devices) == n_dev)
    result["projector"] = compare_arrays(projected, sinogram,
                                         PROJECTOR_GATE_REL)
    del projected, phantom
    if cuda:
        torch.cuda.empty_cache()

    # ── arm check: the device list the model actually realized ───────────────
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))
    result["blocks"] = _block_lengths(model)
    result["ledger"] = _ledger_record(model)
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])

    # ── the value leg: the volume itself, staged for the driver to compare ────
    # recon() with the default output_sharded=False has already gathered to the
    # host, so _to_numpy passes it straight through.  It is written as float32
    # -- the dtype the reconstruction is computed in -- so nothing is invented
    # by the file format; the comparison is promoted to float64 when it is
    # read.  The abs-sum is a cheap independent handle on the volume: two rows
    # whose md5 differ but whose abs-sum agrees to many digits are the same
    # reconstruction with rounding differences, which is exactly the case the
    # value gate has to distinguish from a real one.
    volume = np.ascontiguousarray(np.asarray(_to_numpy(recon), dtype=np.float32))
    out_path = _recon_path(spec["name"], n_dev)
    np.save(out_path, volume)
    digest = _md5(out_path)
    with open(_md5_path(out_path), "w") as handle:
        handle.write(digest + "\n")
    result["recon_path"] = out_path
    result["recon_md5"] = digest
    result["recon_abs_sum"] = float(np.sum(np.abs(volume, dtype=np.float64)))
    result["recon_volume_shape"] = list(volume.shape)
    return result


def generate(cfg):
    """One sinogram per family, checksummed, for both arms of that family to
    read.

    Pinned to a single device so the generator cannot itself become a
    multi-device run, and run WITHOUT the calibration mode so it never resets
    the peak counter an arm is about to read.  Both arms of a family read the
    same bytes, which is what makes their reconstructions comparable at all.

    The phantom is the sparse-dot volume these geometries' own tests use, and
    it is the leanest option at these sizes: it builds one float32 array, where
    the Shepp-Logan builder holds six volume-shaped grids at once.
    """
    import numpy as np
    import torch

    import mbirtorch

    spec = _family_by_name(cfg["family"])
    model = _build_model(spec, pin_devices=(cfg.get("cpu_devices") or [DEVICE]))
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.gen_translation_phantom(recon_shape, "dots", None,
                                                fill_rate=0.05)
    phantom = np.ascontiguousarray(np.asarray(phantom, dtype=np.float32))
    # This forward projection is the projector leg's REFERENCE: the same
    # phantom through the same operator on ONE device.  Every arm re-projects
    # the staged phantom on its own layout and compares against these bytes.
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = dict(cfg, recon_shape=list(recon_shape))
    for label, path, array in (("sino", _sino_path(spec["name"]), sinogram),
                               ("phantom", _phantom_path(spec["name"]), phantom)):
        np.save(path, array)
        digest = _md5(path)
        with open(_md5_path(path), "w") as handle:
            handle.write(digest + "\n")
        out[f"{label}_path"] = path
        out[f"{label}_md5"] = digest
        out[f"{label}_bytes"] = int(array.nbytes)
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited.

    The device pin is MBIRTORCH_NUM_DEVICES and nothing else.  Both it and the
    calibration flag are popped first and then set, so a value exported by the
    submitting shell or the job script cannot leak into an arm that did not ask
    for it -- in particular the generators, which must not own the peak
    counter.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg["mode"] == "arm":
        env["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
        if cfg.get("n_dev") and DEVICE == "cuda":
            env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter, so memory is re-measured
    per arm and never inferred from a process that has already run one."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        # An arm that runs out of device memory lands here.  That is a real
        # reading and it is a gate failure -- it says the ledger let a run
        # start that could not finish -- so it is recorded as a row, counted
        # against the job, and the run continues to the next arm.
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def _arm_cfg(spec, n):
    entry = dict(mode="arm", family=spec["name"], cell=list(spec["cell"]),
                 n_dev=n, arm_id=f'{spec["name"]}_n{n}')
    if DEVICE != "cuda":
        # SMOKE ONLY.  The environment pin is a CUDA-only mechanism -- the
        # automatic search short-circuits below two visible CUDA devices -- so
        # the CPU path pins by device LIST and the row says so.
        entry["cpu_devices"] = [DEVICE] * n
    return entry


def build_plan(families):
    """The plan in job order: every generator first, then the arms grouped by
    family, with the n=1 reference arm before the multi-device arm it is the
    reference for, so a truncated job still holds whole families."""
    plan = []
    for spec in families:
        entry = dict(mode="generate", family=spec["name"],
                     cell=list(spec["cell"]), n_dev=None)
        if DEVICE != "cuda":
            entry["cpu_devices"] = [DEVICE]
        plan.append(entry)
    for spec in families:
        for n in spec["counts"]:
            plan.append(_arm_cfg(spec, n))
    return plan


def _dry_run(families, plan):
    measured = [c for c in plan if c["mode"] == "arm"]
    print(f"mg15 plan: {len(measured)} measured arms "
          f"({len(plan) - len(measured)} generators), "
          f"{VCD_ITERATIONS} iterations each, device {DEVICE}")
    total = 0.0
    for spec in families:
        cell = spec["cell"]
        gib = cell[0] * cell[1] * cell[2] * 4 / 2 ** 30
        total += gib
        shape = spec.get("recon_shape")
        print(f'  sinogram {spec["name"]:<10} {tuple(cell)!s:>20} -> recon '
              f'{(tuple(shape) if shape else "(derived at runtime)")!s:<20} '
              f'{gib:7.2f} GiB')
    kept = os.environ.get("MG15_KEEP_ARTIFACTS") == "1"
    print(f"  {total:.2f} GiB of sinogram written to {RESULTS_DIR}"
          f'{"" if kept else ", deleted at the end"}, plus one phantom per '
          f"family and one recon volume per arm for the value gate")
    for cfg in plan:
        if cfg["mode"] != "arm":
            continue
        views, _rows, _channels = cfg["cell"]
        n = cfg["n_dev"]
        # The point of the arm in one line: whether this count divides the view
        # axis.  The slice axis is only known once the model is built, so it is
        # reported on the row rather than guessed here.
        divides = "divides" if views % n == 0 else f"leaves {views % n}"
        print(f'  {cfg["arm_id"]:<16} {tuple(cfg["cell"])!s:>20} n={n}  '
              f"{views} views over {n}: {divides}")
    print(f"gates: forward projection <= {PROJECTOR_GATE_REL:g} against the "
          f"single-device sinogram; reconstruction <= {RECON_GATE_REL:g} "
          f"against the n=1 arm of the same family; modeled peak >= measured "
          f"peak on every device; every arm believable.  Exit code is the "
          f"verdict.")


def main():
    families = selected_families()
    plan = build_plan(families)
    if "--dry-run" in sys.argv:
        _dry_run(families, plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg15_p5_gate_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg15 sharding-pad-removal gate on {RUN_LABEL} ({DEVICE}); families "
          f'{[f["name"] for f in families]} -> {out_path}', flush=True)
    rows = []
    # Rows are written as they finish, so a job that runs out of wall time
    # still yields every arm it completed.
    with open(out_path, "w") as sink:
        for cfg in plan:
            # A generator is skipped only when BOTH of its artifacts are
            # already staged and checksummed.  A half-written pair from an
            # interrupted run has to be regenerated, not read.
            if cfg["mode"] == "generate" and all(
                    os.path.exists(_md5_path(path(cfg["family"])))
                    for path in (_sino_path, _phantom_path)):
                continue
            label = cfg.get("arm_id", f'generate {cfg["family"]}')
            print(f"  {label}", flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        # The value gate reads the staged volumes, so it has to run before the
        # artifacts are removed below.
        summary = summarize(rows, families, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    if os.environ.get("MG15_KEEP_ARTIFACTS", "0") != "1":
        for spec in families:
            paths = [_sino_path(spec["name"]), _phantom_path(spec["name"])]
            for n in spec["counts"]:
                paths.append(_recon_path(spec["name"], n))
            for path in list(paths):
                paths.append(_md5_path(path))
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)
    else:
        print("MG15_KEEP_ARTIFACTS=1: the sinograms, the recon volumes and "
              f"their checksums are left in {RESULTS_DIR}")
    print(f"\nwrote {out_path}")
    return 0 if summary["passed"] else 2


# ── the value gate ────────────────────────────────────────────────────────────
def compare_volumes(ref_path, out_path):
    """The reconstruction backstop, between two staged volumes on disk.

    Memory-mapped so the driver never holds either volume, which matters
    because this runs in the parent process right after six multi-GPU arms.
    """
    import numpy as np

    return compare_arrays(np.load(out_path, mmap_mode="r"),
                          np.load(ref_path, mmap_mode="r"), RECON_GATE_REL)


def _fmt(value, spec="{:.3e}"):
    """Format a number that may be absent, so one missing diagnostic never
    turns a summary line into a traceback."""
    return "n/a" if value is None else spec.format(value)


def _arm_invalidity(row):
    """The reasons a row cannot be believed, as (tag, message) pairs.

    Separate from the gates themselves: a row that cannot be believed is not a
    reading that failed, it is a reading that was never taken, and the two are
    reported differently even though both fail the job.
    """
    problems = []
    if row.get("error"):
        problems.append(("error", "the arm's subprocess did not finish: "
                                  f'{str(row["error"]).splitlines()[-1][:200]}'))
        return problems
    if row.get("devices_ok") is False:
        problems.append(("devices",
                         f'realized {row.get("realized_devices")} for '
                         f'n={row.get("n_dev")}'))
    if row.get("projection_devices_ok") is False:
        # The projector leg is the half of the value gate with resolving power,
        # and it only has that power if it ran on the arm's real layout.  A
        # projection that fell back to fewer devices compares a volume against
        # something close to itself and passes while measuring nothing, so it
        # invalidates the arm rather than quietly reporting ~1e-7.
        problems.append(("projection_devices",
                         f'the projector leg ran on '
                         f'{row.get("projection_devices")} '
                         f'({row.get("projection_n_devices")} device(s)), not '
                         f'on the arm\'s {row.get("n_dev")}; its comparison '
                         f'does not gate the split'))
    if row.get("calibration_env_ok") is False:
        problems.append(("calibration_env",
                         "MBIRTORCH_MEMORY_CALIBRATION was not set in the "
                         "arm's own environment"))
    if row.get("sino_md5_ok") is False:
        problems.append(("sino_md5", "the staged sinogram's checksum does not "
                                     "match the one the generator wrote"))
    if row.get("phantom_md5_ok") is False:
        problems.append(("phantom_md5", "the staged phantom's checksum does "
                                        "not match the one the generator "
                                        "wrote"))
    if row.get("recon_shape_ok") is False:
        problems.append(("recon_shape",
                         f'recon shape {row.get("recon_shape")} is not the '
                         f'registered {row.get("recon_shape_expected")}; a '
                         f'geometry default has moved'))
    if row.get("num_pixels_ok") is False:
        problems.append(("num_pixels",
                         f'{row.get("num_pixels_full")} masked pixels, not the '
                         f'registered {row.get("num_pixels_expected")}'))
    if row.get("cuda") and row.get("layout_is_automatic") is False:
        problems.append(("layout", "this CUDA arm did not take the automatic "
                                   "device layout, so it did not consume the "
                                   "ledger the memory leg is about"))
    return problems


def summarize(rows, families, out_path):
    print(f"\n===== mg15 sharding-pad-removal gate ({out_path}) =====")
    arms = [r for r in rows if r.get("mode") == "arm"]
    by_id = {r.get("arm_id"): r for r in arms}

    # ── the arms as measured ─────────────────────────────────────────────────
    header = (f'{"arm":>16}{"n":>3}{"device":>9}{"modeled":>11}{"measured":>11}'
              f'{"ratio":>8}{"floor":>7}{"band":>8}  dominant phase')
    print(header)
    print("-" * len(header))
    invalid, floor_fails, band_warnings, arm_summaries = [], [], [], []
    for row in arms:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'{arm_id:>16}{row.get("n_dev", 0):>3}   ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:74]}')
        ledger = row.get("ledger") or {}
        dominant = ledger.get("dominant_phase") or []
        low, high = row.get("band") or BANDS["CALIBRATION_BAND"]
        entry = dict(arm_id=arm_id, family=row.get("family"),
                     n_dev=row.get("n_dev"), cell=row.get("cell"),
                     recon_shape=row.get("recon_shape"),
                     blocks=row.get("blocks"), recon_s=row.get("recon_s"),
                     band=row.get("band"), per_device=[])
        if not row.get("error") and not row.get("calibration"):
            modeled = ledger.get("modeled_peak_bytes") or []
            print(f'{arm_id:>16}{row["n_dev"]:>3}{"-":>9}'
                  f'{(max(modeled) / 2 ** 30 if modeled else 0):>10.2f}G'
                  f'{"n/a":>11}{"n/a":>8}{"n/a":>7}{"no-cuda":>8}  '
                  f'{dominant[0] if dominant else ""}')
            entry["calibration_skipped_reason"] = row.get(
                "calibration_skipped_reason")
        for i, cal in enumerate(row.get("calibration", [])):
            ratio = cal["ratio"]
            # Two separate judgements on the same number.  The FLOOR is the
            # rule the ledger exists to keep and a violation fails the job; the
            # BAND TOP is an efficiency statement and a violation is a warning.
            floor_ok = ratio >= low
            band_ok = ratio <= high
            if not floor_ok:
                floor_fails.append((row, i, cal))
            elif not band_ok:
                band_warnings.append((row, i, cal))
            entry["per_device"].append(
                dict(device=cal["device"], modeled_bytes=cal["modeled_bytes"],
                     measured_bytes=cal["measured_bytes"], ratio=ratio,
                     floor_ok=floor_ok, band_ok=band_ok,
                     dominant_phase=(dominant[i] if i < len(dominant) else None)))
            print(f'{arm_id:>16}{row["n_dev"]:>3}{cal["device"]:>9}'
                  f'{cal["modeled_bytes"] / 2 ** 30:>10.2f}G'
                  f'{cal["measured_bytes"] / 2 ** 30:>10.2f}G'
                  f'{ratio:>8.3f}{("ok" if floor_ok else "UNDER"):>7}'
                  f'{("ok" if band_ok else "over"):>8}  '
                  f'{dominant[i] if i < len(dominant) else ""}')
        blocks = row.get("blocks") or {}
        if blocks:
            # The split itself, printed once per arm.  This is the change under
            # test, so it is worth reading off the log rather than inferring
            # from the shapes.
            print(f'{"":>16}   blocks: views {blocks.get("view_blocks")}'
                  f'{"" if blocks.get("views_divide") else " (uneven)"}, '
                  f'slices {blocks.get("slice_blocks")}'
                  f'{"" if blocks.get("slices_divide") else " (uneven)"}')
        for tag, message in _arm_invalidity(row):
            print(f"    ARM CHECK FAIL: {message}")
            invalid.append(f"{arm_id}|{tag}")
        if row.get("band_matches_library") is False:
            print(f'    NOTE: the library now declares '
                  f'{row.get("band_name")} = {row.get("library_band")}, not '
                  f'{row.get("band")}; the verdicts above used '
                  f'{row.get("band")}')
        arm_summaries.append(entry)
    print("-" * len(header))

    # ── leg one: the values, in two parts ────────────────────────────────────
    # Part (a) is the gate with resolving power: one forward projection of the
    # staged phantom on this arm's layout against the same phantom projected on
    # one device.  Part (b) is the gross-error backstop on the reconstruction.
    # The module docstring records the ablation that set the two thresholds.
    print(f"\nVALUE GATE (a) PROJECTOR -- the staged phantom forward-projected "
          f"on this arm's layout against the single-device sinogram, gate "
          f"{PROJECTOR_GATE_REL:g}:")
    value_fail = False
    projector_results = []
    for row in arms:
        arm_id, check = row.get("arm_id"), row.get("projector")
        if row.get("error"):
            continue                    # already counted as an unbelievable row
        entry = dict(arm_id=arm_id, family=row.get("family"),
                     n_dev=row.get("n_dev"), part="projector",
                     projection_devices=row.get("projection_devices"),
                     projection_n_devices=row.get("projection_n_devices"),
                     projection_devices_ok=row.get("projection_devices_ok"))
        if not check:
            entry.update(ok=False, rel=None,
                         reason="the arm recorded no projector comparison")
            print(f"  {arm_id:<16} FAIL -- {entry['reason']}")
            value_fail = True
        else:
            entry.update(check)
            if check.get("rel") is None:
                print(f'  {arm_id:<16} FAIL -- {check.get("reason")}')
                value_fail = True
            else:
                # The device count the projection ran on is printed on every
                # line, not only when it is wrong: this leg is worthless if it
                # silently ran on one device, so the log has to show that it
                # did not.
                print(f'  {arm_id:<16} rel {check["rel"]:.3e}  '
                      f'{"PASS" if check["ok"] else "FAIL"}  '
                      f'(projected on {row.get("projection_n_devices")} '
                      f'device(s), nrmse {_fmt(check.get("nrmse"))}, max|ref| '
                      f'{check["max_abs_ref"]:.6g}, shape {check["shape"]})')
                if not check["ok"]:
                    value_fail = True
        projector_results.append(entry)

    print(f"\nVALUE GATE (b) RECONSTRUCTION -- {VCD_ITERATIONS} iterations "
          f"against the n=1 arm of the same family, gate {RECON_GATE_REL:g} "
          f"(a backstop, not a fine gate: see the module docstring):")
    recon_results = []
    for spec in families:
        name = spec["name"]
        ref_row = by_id.get(f"{name}_n1")
        for n in spec["counts"]:
            if n == 1:
                continue
            out_row = by_id.get(f"{name}_n{n}")
            entry = dict(family=name, n_dev=n, ref_arm=f"{name}_n1",
                         arm_id=f"{name}_n{n}", part="recon")
            missing = None
            if ref_row is None or not ref_row.get("recon_path"):
                missing = f"the reference arm {name}_n1 produced no volume"
            elif out_row is None or not out_row.get("recon_path"):
                missing = f"the arm {name}_n{n} produced no volume"
            elif not os.path.exists(ref_row["recon_path"]) or \
                    not os.path.exists(out_row["recon_path"]):
                missing = "a staged volume is no longer on disk"
            if missing:
                entry.update(ok=False, rel=None, reason=missing)
                print(f"  {name:<12} n={n}  FAIL -- {missing}")
                value_fail = True
                recon_results.append(entry)
                continue
            entry.update(compare_volumes(ref_row["recon_path"],
                                         out_row["recon_path"]))
            recon_results.append(entry)
            if entry.get("rel") is None:
                print(f'  {name:<12} n={n}  FAIL -- {entry.get("reason")}')
                value_fail = True
                continue
            print(f'  {name:<12} n={n}  rel {entry["rel"]:.3e}  '
                  f'{"PASS" if entry["ok"] else "FAIL"}  '
                  f'(nrmse {_fmt(entry.get("nrmse"))}, max|ref| '
                  f'{entry["max_abs_ref"]:.6g}, shape {entry["shape"]})')
            if not entry["ok"]:
                value_fail = True
    if not recon_results:
        print("  no multi-device arm ran, so no family was compared")
    value_results = projector_results + recon_results

    # ── leg two: the memory floor ────────────────────────────────────────────
    measured_any = any(r.get("calibration") for r in arms)
    print(f"\nLEDGER FLOOR -- modeled peak >= measured peak on every device of "
          f"every arm:")
    if not measured_any:
        # Vacuous rather than silent.  An empty measured column is not a pass,
        # and a reader who is told nothing would take it for one.
        print("  NO MEASURED READING: torch.cuda.max_memory_allocated is "
              f"CUDA-only and this run was on {DEVICE}, so the floor passes "
              "vacuously.  The modeled column above is real; the measured one "
              "does not exist here.  Only a CUDA run gates this leg.")
    elif not floor_fails:
        worst = min((d["ratio"] for e in arm_summaries for d in e["per_device"]),
                    default=None)
        print(f"  PASS on every device; lowest ratio {worst:.3f}")
    for row, i, cal in floor_fails:
        ledger = row.get("ledger") or {}
        terms = ((ledger.get("dominant_terms") or [[]])[i]
                 if i < len(ledger.get("dominant_terms") or []) else [])
        print(f'\n  UNDER THE FLOOR {row.get("arm_id")} on {cal["device"]}: '
              f'modeled {cal["modeled_bytes"] / 2 ** 30:.2f}G < measured '
              f'{cal["measured_bytes"] / 2 ** 30:.2f}G (ratio '
              f'{cal["ratio"]:.3f}); short by '
              f'{(cal["measured_bytes"] - cal["modeled_bytes"]) / 2 ** 30:.2f}G')
        print(f'    dominant phase: '
              f'{(ledger.get("dominant_phase") or [None])[i]}')
        for name, value in terms:
            print(f"      {name:<40}{value / 2 ** 30:>8.2f}G")
        blocks = row.get("blocks") or {}
        print(f'    this device owns view block '
              f'{(blocks.get("view_blocks") or [None])[i] if i < len(blocks.get("view_blocks") or []) else None} '
              f'and slice block '
              f'{(blocks.get("slice_blocks") or [None])[i] if i < len(blocks.get("slice_blocks") or []) else None}')

    # ── the band top, reported and not gated ─────────────────────────────────
    if band_warnings:
        print(f"\nBAND TOP (warning only, not a failure): "
              f"{len(band_warnings)} reading(s) over-charged beyond the band:")
        for row, _i, cal in band_warnings:
            low, high = row.get("band")
            print(f'  {row.get("arm_id")} on {cal["device"]}: ratio '
                  f'{cal["ratio"]:.3f} above {high:.2f} '
                  f'({row.get("band_name")}).  An over-estimate spreads a run '
                  f'wider than it needs; it does not risk the allocator.')

    # ── the paste row for the stale ledger-calibration arm ───────────────────
    paste = _paste_row(by_id.get("multiaxis_n4"))

    # ── the verdict ──────────────────────────────────────────────────────────
    passed = (not value_fail) and (not floor_fails) and (not invalid)
    print(f"\n===== VERDICT: {'PASS' if passed else 'FAIL'} =====")
    print(f"  values      {'pass' if not value_fail else 'FAIL'} "
          f"({len(projector_results)} projector comparison(s) at "
          f"{PROJECTOR_GATE_REL:g}, {len(recon_results)} reconstruction "
          f"comparison(s) at {RECON_GATE_REL:g})")
    print(f"  ledger floor{' pass' if not floor_fails else ' FAIL'} "
          f"({len(floor_fails)} device(s) below the floor"
          f'{"; vacuous, no CUDA reading" if not measured_any else ""})')
    print(f"  arm checks  {'pass' if not invalid else 'FAIL'} "
          f"({len(invalid)} row(s) that cannot be believed"
          f'{": " + ", ".join(invalid) if invalid else ""})')
    print(f"  band top    {len(band_warnings)} warning(s), not gated")
    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"  GPU health  {len(hot)} row(s) sampled hot: {hot}")
    print(f"  exit code   {0 if passed else 2}")
    return dict(passed=passed, arms=arm_summaries, values=value_results,
                invalid=invalid,
                floor_fails=[f'{r.get("arm_id")}|{c["device"]}'
                             for r, _i, c in floor_fails],
                band_warnings=[f'{r.get("arm_id")}|{c["device"]}'
                               for r, _i, c in band_warnings],
                hot=hot, paste_row=paste,
                projector_gate_rel=PROJECTOR_GATE_REL,
                recon_gate_rel=RECON_GATE_REL,
                bands={k: list(v) for k, v in BANDS.items()})


def _paste_row(row):
    """The replacement line for MEASURED_ARMS['ma512_n4'].

    The recorded row was measured on the padded split, where 510 recon slices
    were rounded to 512 and cut into four equal blocks.  On the new split the
    blocks are 128, 128, 127, 127, so the peaks the row records are peaks of a
    configuration that no longer exists.  This prints the replacement in the
    table's own format, with the measured per-device peaks from the arm above,
    and names where it came from so the table's provenance stays traceable.
    """
    print("\n===== ma512_n4: the replacement row for "
          "tests/test_memory_ledger.py MEASURED_ARMS =====")
    if row is None or row.get("error") or not row.get("calibration"):
        why = ("that arm did not run" if row is None else
               "that arm failed" if row.get("error") else
               "that arm produced no measured peaks (CUDA-only counter)")
        print(f"  NOT AVAILABLE: {why}, so the recorded row stands unchanged.")
        return None
    peaks = [int(cal["measured_bytes"]) for cal in row["calibration"]]
    blocks = row.get("blocks") or {}
    job = os.environ.get("SLURM_JOB_ID", "no slurm job id")
    line = (f"    'ma512_n4': ((512, 448, 384), (384, 384, 510), 115164,\n"
            f'                 [{", ".join(str(p) for p in peaks)}]),')
    print(line)
    print(f"    # re-measured on the unpadded split by mg15_p5_gate (slurm "
          f"{job}) on {RUN_LABEL}, {time.strftime('%Y-%m-%d')}; slice blocks "
          f'{blocks.get("slice_blocks")}, view blocks '
          f'{blocks.get("view_blocks")}')
    return dict(peaks=peaks, line=line, host=RUN_LABEL, slurm_job=job,
                date=time.strftime("%Y-%m-%d"),
                slice_blocks=blocks.get("slice_blocks"),
                view_blocks=blocks.get("view_blocks"))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        try:
            out = generate(cfg) if cfg["mode"] == "generate" else run_arm(cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        # The exit code IS the verdict: the job script chains a follow-on run
        # with --dependency=afterok, so a failing gate has to stop it.
        sys.exit(main())
