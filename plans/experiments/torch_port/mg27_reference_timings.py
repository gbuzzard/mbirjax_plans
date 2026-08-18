"""mg27 -- THE SINGLE-DEVICE REFERENCE TIMINGS AND THE 512-CLASS COLUMN,
RE-MEASURED ON THE CURRENT TIP.

WHY THIS RUN EXISTS.  The comparison table in
`plans/torch_port/active/execution_overview.md` carries one mbirtorch column
per problem size, and every speedup ratio in it divides by that column's
single-device row.  Those single-device rows and the whole 512-class column
were measured before three changes that moved how long a reconstruction takes:
the column-gather forward projection became the default on 2026-08-11, the
forward pixel batch default became 32768 on 2026-08-17, and the kernel width
padding landed on 2026-08-18 (mbirtorch commit 64dedb8).  The 512-class column
was measured on 2026-08-09 and has not been touched since.  This run
re-measures the full mbirtorch column at the current tip: both geometries,
both standard cells, three device counts.  That is open item C1.

The mbirjax column is NOT re-measured.  It comes from a jax version and branch
that has not moved during the campaign, so re-running it would spend an hour
reproducing numbers that are already current.  This run therefore writes only
the mbirtorch half of each table and leaves the mbirjax half alone.

The same numbers feed the user-facing timing table in
`mbirtorch/docs/source/usr_multi_gpu.rst`, which is open item F1.  That table
still shows 94.0 s for the 1024-class single-device reconstruction, which
predates the hand-written kernels entirely.  This run prints a third table in
that page's own shape so the page can be refreshed from one place.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It edits no library file,
flips no default, and sets no knob under test.  Every variant runs the shipped
configuration.

TERMS USED BELOW, defined once here.
    variant     one measured configuration: one geometry, one cell, one device
                count, run in its own new process.
    cell        a problem-size class, named by its sinogram shape.  This run
                has two: the 1024 class, sinogram (1024, 1008, 992), and the
                512 class, sinogram (512, 448, 384).
    count       the number of devices a variant reconstructs on.
    stage       the untimed step that builds one cell's sinogram once and
                writes it to disk.
    cold pass   the first reconstruction in a process.  It pays the per-device
                kernel compiles, so it is discarded.
    warm pass   a reconstruction after the cold one.  Three of them are timed
                and the row carries their median.
    fingerprint two float64 reductions of a reconstruction, used to check that
                every device count computed the same answer.

THE VARIANTS, twelve of them: geometry in (parallel, cone) x cell in (512
class, 1024 class) x count in (1, 2, 4).

THE PROTOCOL, which is mg1's and is unchanged so the new rows are directly
comparable to the recorded ones.  Every value below was read out of
`mg1_readout.py` in session rather than remembered.

    model       parallel: angles evenly spaced over half a turn,
                np.linspace(0, pi, num_views, endpoint=False), passed to
                ParallelBeamModel with the cell.
                cone: angles evenly spaced over a full turn,
                np.linspace(0, 2*pi, num_views, endpoint=False), passed to
                ConeBeamModel with source_detector_dist 4.0 x num_channels and
                source_iso_dist 2.0 x num_channels, where num_channels is the
                cell's third entry.
                Both then take set_params(no_warning=True, verbose=0).
    phantom     generate_3d_shepp_logan_low_dynamic_range(recon_shape), where
                recon_shape is whatever the model's own defaults produced.
    sinogram    that phantom forward projected, cast to float32.
    weights     exp(-sinogram / (2 * max(sinogram))), float32.
    recon       np.random.seed(13) immediately before every call, then
                model.recon(sinogram, weights=weights, max_iterations=3,
                stop_threshold_change_pct=0.0).  The seed is mg1's, and mg1
                took it from kb3 so its rows would compare to kb3's.
    timing      one cold pass DISCARDED, then three warm passes timed with
                perf_counter around the whole call.  The row carries the
                median, the minimum, the maximum, and the spread, which is
                (max - min) / median.
    memory      torch.cuda.max_memory_allocated per device, over the sinogram
                placement's devices, with the counters reset after the cold
                pass so the reading covers the warm passes alone.  The row
                carries every device's peak and the largest of them, which is
                what the comparison table's memory column means.

THE PIN.  Each variant runs in a fresh subprocess with MBIRTORCH_NUM_DEVICES
set to its count.  That is the refresh script's mechanism and it is chosen for
two reasons.  The pin fixes the count while leaving the model on the automatic
branch, where the memory preflight still runs; an explicit configure_devices
call would take the explicit branch instead.  And a count changed inside one
process would inherit the previous variant's compiled kernels and allocator
state, so the process boundary is what makes the counts comparable.

STAGING.  One sinogram per (geometry, cell), built once in its own
single-device subprocess and saved as a float32 .npy with an md5 sidecar.
Every count variant of that pair loads that same file, so the three counts
reconstruct the same array and the comparison between them is controlled.  The
md5 is verified on every load and recorded on every row: a truncated read on a
shared parallel filesystem is a recorded failure mode of this work.  The
1024-class files are about 4.1 GB each and the 512-class files about 0.34 GB,
so a full run stages about 8.9 GB.  They are NOT deleted.

THE CROSS-COUNT VALUE CHECK.  Within one (geometry, cell), each count's
fingerprint must agree with the count-1 fingerprint to 1e-3 relative on both
components.  The gate is 1e-3 and not tighter because a compiled torch body is
regenerated per call shape, and at an uneven shard split the different shapes
sum their reductions in a different order; multigpu_findings.md section 1.16
records that difference at about 6e-4.  It is deterministic and benign, and a
1e-4 gate would flag it on every uneven split.

THE OTHER INSTRUMENT-HEALTH CHECKS.  Every planned variant produced a row.
Every variant realized the count it was pinned to.  No projection direction
ran as general torch code: on CUDA both geometries bind hand-written kernels
in both directions, so `_memory_ledger.torch_body_directions(model)` must be
empty, and every row records what it actually was.  On the CPU smoke the
kernels do not exist and the same reading must be ('forward', 'back'), so each
row also says which of the two it expected.  A GPU health sample is taken
before and after every variant and the hot and throttle readings are recorded.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned
variant ran, realized its pinned count, bound the expected kind of projection
body, read a verified sinogram, and agreed with its count-1 sibling.  It is
NOT a verdict on the timings.  A slow variant, a wide spread, a hot GPU and a
peak that moved are all printed in full and none of them touches the exit
code.  A person reads the three tables.

THE ORDER.  The 512-class variants run first, because they cost minutes and a
harness defect shows up there rather than after an hour of 1024-class work.
Within a cell the counts run ascending, so the count-1 fingerprint every other
count is checked against exists before those counts run.  Each cell's staging
runs immediately before its own variants, so a run that is cut short has not
paid for a sinogram it never used.  The four-device variants need a four-GPU
allocation; the pin selects a subset of those GPUs for the one- and
two-device variants.

WALL ESTIMATE.  The 1024-class cone single-device reconstruction is the
slowest variant at about 62 s, and it runs four passes, so it is about four
minutes on its own.  Twelve variants with their cold passes, their staging and
their model builds come to roughly 45 minutes.  The sbatch asks for two hours.

OUTPUT.  One jsonl under MG27_RESULTS, named
mg27_reference_h<node>_<stamp>.jsonl: a run-header row, one row per stage and
per variant, and a summary row.  Rows are flushed as they finish, so a job
that runs out of wall time still yields everything it completed.  The run then
prints three tables.  The first two are markdown in the shape of
execution_overview.md sections 5.1 and 5.2, carrying the mbirtorch columns
only.  The third is a reStructuredText list-table in the shape of the one in
usr_multi_gpu.rst, times only.  Every table says the numbers are warm medians
of three on an H100.

Run:
    <torch python> mg27_reference_timings.py         on a 4-GPU node
    MG27_DRY=1 <python> mg27_reference_timings.py    print the plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized
token is an error, not a silent skip.
    MG27_RESULTS=<dir>      where the jsonl and the staged sinograms go
    MG27_SMOKE=1            the local smoke: tiny CPU cells at counts 1 and 2
    MG27_DRY=1              print the variant plan and exit, importing no torch
    MG27_REPEATS=3          warm repeats after the discarded cold pass
    MG27_VARIANTS=a,b       a subset, by variant id, e.g. parallel_1024_n2
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
SMOKE = os.environ.get("MG27_SMOKE", "0") == "1"
DRY = os.environ.get("MG27_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

GEOMETRIES = ("parallel", "cone")
#: The two standard cells, in RUN order: the cheap one first, so a harness
#: defect surfaces in minutes.  The REPORT prints them the other way round,
#: because execution_overview.md section 5.1 is the 1024 class and 5.2 is the
#: 512 class.
CELLS = ((512, 448, 384), (1024, 1008, 992))
COUNTS = (1, 2, 4)

#: The smoke's stand-ins.  Two tiny cells so both table rows exist, and two
#: counts so the cross-count check has something to compare.  A parallel model
#: at (8, 24, 20) reconstructs a (20, 20, 24) volume, which two virtual CPU
#: devices can still split on both axes.
SMOKE_CELLS = ((8, 24, 20), (12, 24, 20))
SMOKE_COUNTS = (1, 2)

# ── mg1's reconstruction protocol, reused verbatim ────────────────────────────
# Every constant in this block was read out of mg1_readout.py rather than
# chosen here.  Changing one would make the new rows incomparable to the
# recorded ones, which is the whole point of the run.
#: mg1's VCD iteration count.  The campaign ruler is a 3-iteration seeded
#: reconstruction; the smoke drops to 1 only to keep the local round trip
#: short, and every row records what it ran.
VCD_ITERATIONS = 1 if SMOKE else 3
#: mg1's seed, reset immediately before every reconstruction.  mg1 took it
#: from kb3 so its one-device rows would compare against kb3's baselines.
VCD_SEED = 13
#: mg1's cone construction: the source-detector and source-isocenter distances
#: are these multiples of the detector channel count.
CONE_SDD_PER_CHANNEL = 4.0
CONE_SID_PER_CHANNEL = 2.0

#: Warm repeats after the discarded cold pass.  Three is the campaign ruler.
WARM_REPEATS = max(1, int(os.environ.get("MG27_REPEATS", "3")))

# ── the cross-count value check ───────────────────────────────────────────────
#: Each count's fingerprint is compared against the count-1 fingerprint at this
#: relative tolerance, on both components.
#:
#: WHY 1e-3 AND NOT TIGHTER.  A compiled torch body is regenerated for each
#: call shape it sees.  When a device count does not divide a sharded axis the
#: shards have different shapes, so those bodies sum their reductions in a
#: different order, and the result differs at about 6e-4 between counts.
#: multigpu_findings.md section 1.16 records that reading.  It is
#: deterministic and benign, so cross-count comparisons gate at 1e-3; a 1e-4
#: gate would fail on every uneven split and say nothing.
CROSS_COUNT_REL_GATE = 1e-3
#: The count every other count is compared against.  One device is the
#: reference the whole run exists to re-anchor, so it is also the value
#: reference; it is the first count in both the production and the smoke list.
BASE_COUNT = 1

# ── recorded context, not gates ───────────────────────────────────────────────
#: The three changes these references postdate, printed with the plan so the
#: log opens with the reason the run exists.
CHANGES_SINCE_REFERENCES = (
    "column-gather forward default, 2026-08-11",
    "forward pixel batch default 32768, 2026-08-17",
    "kernel width padding, mbirtorch 64dedb8, 2026-08-18",
)
#: The padding witness.  504 is a four-device slice band at the 2048 cell and
#: 512 is what the rounding must turn it into; an unpadded tree either has no
#: such function or returns 504.  Recorded on every row so a reader can tell
#: which tree produced these numbers without leaving the jsonl.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512
#: The shipped forward pixel batch (tomography_model.FORWARD_PIXEL_BATCH).  No
#: variant sets it; every row records what the model reported.
SHIPPED_PIXEL_BATCH = 32768

# ── the GPU health sample (mg20 / mg21b's) ────────────────────────────────────
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
    "MG27_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
VARIANT_COL = 22               # wide enough for the longest variant id printed
# ──────────────────────────────────────────────────────────────────────────────


def cells():
    return SMOKE_CELLS if SMOKE else CELLS


def counts():
    return SMOKE_COUNTS if SMOKE else COUNTS


def report_cells():
    """The cells in REPORT order.  execution_overview.md prints the 1024 class
    first and the 512 class second, so the two markdown tables do too, even
    though the run measures them the other way round."""
    return tuple(reversed(cells()))


def variant_id(geometry, cell, n_dev):
    return f"{geometry}_{cell[0]}_n{n_dev}"


def all_variant_ids():
    return [variant_id(g, c, n)
            for c in cells() for g in GEOMETRIES for n in counts()]


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
    # cheap cell first, counts ascending), so it must not depend on the order
    # someone typed the tokens in.
    return [name for name in allowed if name in chosen]


# ── the staged sinogram ───────────────────────────────────────────────────────
def _sino_path(geometry, cell):
    """One file per (geometry, cell).  The cell's view count is in the name, so
    a smoke run and a production run can share a results directory without
    either reading the other's bytes."""
    return os.path.join(RESULTS_DIR, f"mg27_sino_{geometry}_{cell[0]}.npy")


def _md5_path(path):
    return path + ".md5"


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: at the 1024 cell the file is over
    four gigabytes and reading it whole to hash it would be wasteful."""
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
    """The one host exit.  ``Shards.gather()`` ALREADY returns numpy, so a
    gather is never followed by ``.detach()`` -- re-detaching one is a recorded
    way to lose every multi-device row in a run."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, "gather", None)) and hasattr(x, "placement"):
        return x.gather()                      # ALREADY numpy: do not re-detach
    return (x.detach().cpu().numpy()
            if callable(getattr(x, "detach", None)) else np.asarray(x))


def _weights(sinogram):
    """mg1's weighting formula, one dtype, every variant."""
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


# ── the model, built mg1's way ────────────────────────────────────────────────
def build_model(geometry, cell, cpu_devices=None):
    """mg1's model construction, unchanged.

    ``cpu_devices`` is for the smoke only.  On CUDA nothing is configured here:
    the count comes from MBIRTORCH_NUM_DEVICES, which keeps the model on the
    automatic branch where the memory preflight still runs, and an explicit
    configure_devices call would take the explicit branch instead.  The pin is
    a CUDA mechanism -- the policy short-circuits when fewer than two CUDA
    devices are visible -- so the smoke places its virtual CPU devices by hand
    and every row records which mechanism actually pinned it.
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
        # reference can be wrong without anything looking wrong.
        raise ValueError(f"mg27 has no model construction for geometry "
                         f"{geometry!r}")
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def pin_devices_for(n_dev):
    """The explicit device list a variant needs, or None.

    None on CUDA, where MBIRTORCH_NUM_DEVICES does the pinning.  A list of
    virtual CPU devices on the smoke, where the environment pin cannot.
    """
    return None if DEVICE == "cuda" else ["cpu"] * n_dev


def expected_torch_bodies():
    """Which projection directions are expected to run as general torch code.

    None on CUDA: both geometries bind hand-written Triton kernels in both
    directions there, so a non-empty reading means this run measured a
    different implementation than the references describe.  Both directions on
    CPU, where those kernels do not exist.
    """
    return [] if DEVICE == "cuda" else ["forward", "back"]


# ── the value fingerprint ─────────────────────────────────────────────────────
def fingerprint(volume, torch_module, device):
    """Two float64 reductions of a reconstruction, accumulated ON THE DEVICE
    and returned as python floats: the sum of absolute values and the sum of
    squares.

    Two numbers rather than one, because a sum of absolute values alone cannot
    see a rearrangement that preserves magnitudes.  Both accumulate in float64
    in fixed-size chunks: a float32 sum over a billion-element volume loses the
    digits this comparison needs, and promoting the whole volume at once would
    double what the reading costs in memory.
    """
    import numpy as np

    flat = np.ascontiguousarray(volume).reshape(-1)
    abs_sum = torch_module.zeros((), dtype=torch_module.float64, device=device)
    sq_sum = torch_module.zeros((), dtype=torch_module.float64, device=device)
    for start in range(0, flat.shape[0], FINGERPRINT_CHUNK_ELEMS):
        block = torch_module.from_numpy(
            flat[start:start + FINGERPRINT_CHUNK_ELEMS]).to(
                device=device, dtype=torch_module.float64)
        abs_sum += block.abs().sum()
        sq_sum += (block * block).sum()
    return float(abs_sum.item()), float(sq_sum.item())


def relative_gap(value, reference):
    """|value - reference| / |reference|, with a zero reference reported as an
    absolute gap rather than as infinity."""
    if value is None or reference is None:
        return None
    scale = abs(reference)
    return abs(value - reference) / (scale if scale > 0.0 else 1.0)


# ── the worker: one stage or one variant, in its own process ──────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the job is."""
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                  warm_repeats=WARM_REPEATS,
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
    """Build ONE cell's sinogram, once, at one device, and write it with an md5
    sidecar.

    Every count variant of this (geometry, cell) then loads that same array, so
    the three counts reconstruct identical input and the comparison between
    them is controlled rather than incidental.  The generation is mg1's:
    phantom, then forward projection, then float32.

    The staging process is pinned to one device, and the projection is taken on
    a freshly built model, which holds the trivial single-device placement
    until a reconstruction settles a layout.  So nothing here is a
    multi-device run.
    """
    import numpy as np

    import mbirtorch

    result, cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    path = _sino_path(geometry, cell)
    result["sino_path"] = path

    if _staged(path):
        # Already on disk from an earlier job.  Verify it rather than rebuild
        # it: the forward kernel's atomics make a regenerated sinogram
        # non-identical at the e-7 class, so a rebuild would silently change
        # what every variant of this pair reconstructs.
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
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


def run_variant(cfg):
    """One variant: a cold pass discarded, then WARM_REPEATS timed warm passes.

    ORDERING NOTE, load-bearing and taken from mg1.  Every check that reads the
    projectors runs AFTER the cold pass.  The automatic branch settles the
    device layout inside the first ``recon`` call, and a settle that changes
    the count rebuilds ``model.projector_functions``.  A body reading taken
    before that would describe a one-device projector set under an n-device
    label.
    """
    import numpy as np
    import torch

    from mbirtorch import _memory_ledger

    result, cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])

    model = build_model(geometry, cell, cpu_devices=pin_devices_for(n_dev))
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    # ── the staged sinogram, md5-verified on every load ──────────────────────
    path = _sino_path(geometry, cell)
    with open(_md5_path(path)) as handle:
        expected = handle.read().strip()
    actual = _md5(path)
    result.update(sino_path=path, sino_md5=actual,
                  sino_md5_ok=(actual == expected))
    if not result["sino_md5_ok"]:
        # A variant that reconstructed a different array than its siblings did
        # not measure what the plan said, so this stops the variant rather than
        # producing a row that looks comparable and is not.
        result["invalid_reasons"].append(
            f"the staged sinogram at {path} hashes to {actual}, not the "
            f"recorded {expected}")
        return result
    sinogram = np.load(path)
    weights = _weights(sinogram)

    devices_now = [None]

    def vcd():
        """One reconstruction, exactly as mg1 called it."""
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(recon)

    # ── the cold pass, DISCARDED (it pays the per-device kernel compiles) ────
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

    # Recorded, not set.  The batch default moved on 2026-08-17 and is one of
    # the three reasons this run exists, so every row says what it ran.
    try:
        result["forward_pixel_batch"] = int(model._forward_pixel_batch())
    except Exception as exc:                                      # noqa: BLE001
        result["forward_pixel_batch"] = None
        result["forward_pixel_batch_error"] = f"{type(exc).__name__}: {exc}"
    result["forward_pixel_batch_shipped"] = SHIPPED_PIXEL_BATCH

    sinogram_shape = tuple(int(s) for s in model.get_params("sinogram_shape"))
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    result["view_blocks"] = [end - start for _d, (start, end)
                             in model.sino_placement.shard_ranges(
                                 sinogram_shape[0])]
    result["slice_blocks"] = [end - start for _d, (start, end)
                              in model.recon_placement.shard_ranges(
                                  recon_shape[2])]

    # ── the warm repeats ─────────────────────────────────────────────────────
    # The peak counters are reset AFTER the cold pass, so the memory reading
    # covers the warm passes and not the compiles.
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
    fp_device = devices_now[0][0] if devices_now[0] else torch.device("cpu")
    abs_sum, sq_sum = fingerprint(out, torch, fp_device)
    result["fingerprint_abs_sum"] = abs_sum
    result["fingerprint_sq_sum"] = sq_sum
    result["fingerprint_device"] = str(fp_device)
    result["fingerprint_elements"] = int(np.asarray(out).size)
    return result


def run_job(cfg):
    """One stage or one variant, in its own process, with a health sample on
    either side of it.

    A new process per job is not tidiness.  Compiled and Triton bodies are
    cached at module level for the life of a process, and the peak memory
    counters are per process, so both would leak from one variant into the
    next if they shared an interpreter.
    """
    before = sample_gpu_health()
    started = time.time()
    try:
        result = (run_stage(cfg) if cfg["kind"] == "stage"
                  else run_variant(cfg))
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

    MBIRTORCH_NUM_DEVICES is popped and then set, so a value exported by the
    shell cannot reach a job that asked for a different count.  It is set on
    the smoke too, exactly as on CUDA, so the subprocess protocol under test is
    the same one the real run uses; the smoke's count is then realized by an
    explicit CPU device list, because the pin acts only through the device
    policy and that policy short-circuits below two visible CUDA devices.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # it owns the peak counters
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # the shipped configuration
    env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def spawn(cfg):
    """Run one configuration in a NEW interpreter.

    The row goes through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.  On a 45-minute job that is
    the difference between watching progress and waiting in the dark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_mg27_cfg_{cfg['job_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_mg27_out_{cfg['job_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
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
    """Every job, in run order: each cell's staging immediately before that
    cell's own variants, cheap cell first, counts ascending."""
    keep = _strict_subset("MG27_VARIANTS", all_variant_ids())
    plan = []
    for cell in cells():
        for geometry in GEOMETRIES:
            wanted = [n for n in counts()
                      if variant_id(geometry, cell, n) in keep]
            if not wanted:
                continue
            plan.append(dict(kind="stage", geometry=geometry, cell=list(cell),
                             n_dev=1,
                             job_id=f"stage_{geometry}_{cell[0]}"))
            for n_dev in wanted:
                name = variant_id(geometry, cell, n_dev)
                plan.append(dict(kind="variant", geometry=geometry,
                                 cell=list(cell), n_dev=n_dev,
                                 variant=name, job_id=name))
    if not plan:
        raise ValueError("MG27_VARIANTS selects no variant")
    return plan


def print_plan(plan):
    variants = [c for c in plan if c["kind"] == "variant"]
    stages = [c for c in plan if c["kind"] == "stage"]
    print(f"mg27 the reference-timing re-anchor: {len(variants)} variant(s) "
          f"and {len(stages)} staged sinogram(s), device {DEVICE}, "
          f"{VCD_ITERATIONS} VCD iteration(s), "
          f"{WARM_REPEATS} warm repeat(s) after a discarded cold pass")
    print(f"  results and staged sinograms -> {RESULTS_DIR}")
    print("  the references these variants replace predate: "
          + "; ".join(CHANGES_SINCE_REFERENCES))
    print(f"  cross-count fingerprints gate at {CROSS_COUNT_REL_GATE:.0e} "
          f"relative, because compiled torch bodies differ by about 6e-4 "
          f"across counts at an uneven split (multigpu_findings.md 1.16)")
    print(f"  projection directions expected to run as general torch code on "
          f"{DEVICE}: {expected_torch_bodies() or 'none'}")
    total_gib = 0.0
    for cfg in stages:
        cell = tuple(cfg["cell"])
        total_gib += cell[0] * cell[1] * cell[2] * 4 / 2 ** 30
    print(f"  staged sinograms total about {total_gib:.2f} GiB and are kept")
    header = (f'  {"job":<{VARIANT_COL}}{"pin":>5}{"cell":>20}'
              f'{"geometry":>10}  what it does')
    print(header)
    what = dict(stage="builds this cell's sinogram once, at one device",
                variant="cold pass discarded, then the timed warm passes")
    for cfg in plan:
        print(f'  {cfg["job_id"]:<{VARIANT_COL}}{cfg["n_dev"]:>5}'
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
                            f"mg27_reference_{RUN_LABEL}_{stamp}.jsonl")
    print_plan(plan)
    print(f"\nrunning -> {out_path}", flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        header = dict(row="run_header", script="mg27_reference_timings.py",
                      node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
                      python=sys.executable, results_dir=RESULTS_DIR,
                      geometries=list(GEOMETRIES),
                      cells=[list(c) for c in cells()],
                      counts=list(counts()),
                      vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                      warm_repeats=WARM_REPEATS,
                      cross_count_gate=CROSS_COUNT_REL_GATE,
                      changes_since_references=list(CHANGES_SINCE_REFERENCES),
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


def check_cross_counts(by_variant, broken):
    """Within one (geometry, cell), every count's fingerprint against the
    count-1 fingerprint.

    A count that computed a different answer did not measure the same problem,
    so this is instrument health rather than a finding.  A run whose selection
    left out count 1 has no reference to compare against; that is reported as
    unchecked and does not fail, because it is what the caller asked for.
    """
    print("\n-- cross-count value check "
          f"(gate {CROSS_COUNT_REL_GATE:.0e} relative, both components) --")
    header = (f'{"geometry / cell":<24}{"n":>3}{"abs gap":>12}{"sq gap":>12}'
              f'{"verdict":>12}')
    print(header)
    print("-" * len(header))
    records = []
    for cell in report_cells():
        for geometry in GEOMETRIES:
            base = by_variant.get(variant_id(geometry, cell, BASE_COUNT))
            label = f"{geometry} {cell[0]}"
            for n_dev in counts():
                row = by_variant.get(variant_id(geometry, cell, n_dev))
                if row is None:
                    continue
                if base is None:
                    print(f'{label:<24}{n_dev:>3}{"-":>12}{"-":>12}'
                          f'{f"no n={BASE_COUNT}":>12}')
                    records.append(dict(geometry=geometry, cell=list(cell),
                                        n_dev=n_dev, checked=False))
                    continue
                abs_gap = relative_gap(row.get("fingerprint_abs_sum"),
                                       base.get("fingerprint_abs_sum"))
                sq_gap = relative_gap(row.get("fingerprint_sq_sum"),
                                      base.get("fingerprint_sq_sum"))
                ok = (abs_gap is not None and sq_gap is not None
                      and abs_gap <= CROSS_COUNT_REL_GATE
                      and sq_gap <= CROSS_COUNT_REL_GATE)
                print(f'{label:<24}{n_dev:>3}{_fmt(abs_gap, 12, "e", 2)}'
                      f'{_fmt(sq_gap, 12, "e", 2)}'
                      f'{("ok" if ok else "FAIL"):>12}')
                records.append(dict(geometry=geometry, cell=list(cell),
                                    n_dev=n_dev, checked=True,
                                    abs_gap=abs_gap, sq_gap=sq_gap, ok=ok))
                if not ok:
                    broken.append(f"{variant_id(geometry, cell, n_dev)}|"
                                  f"fingerprint disagrees with n="
                                  f"{BASE_COUNT}")
    return records


def print_overview_table(by_variant, cell, section):
    """One markdown table in the shape of execution_overview.md's, carrying the
    mbirtorch columns only.  The mbirjax columns of that document are not
    re-measured by this run and are left for the editor to keep."""
    print(f"\n### paste into execution_overview.md {section} "
          f"-- the mbirtorch columns, cell sinogram {tuple(cell)}")
    print("| geometry | devices | mbirtorch time | mbirtorch peak |")
    print("|---|---|---|---|")
    for geometry in GEOMETRIES:
        for n_dev in counts():
            row = by_variant.get(variant_id(geometry, cell, n_dev))
            if row is None:
                print(f"| {geometry} | {n_dev} | - | - |")
                continue
            peak_gb = row.get("peak_bytes", 0) / 2 ** 30
            print(f'| {geometry} | {n_dev} | {row.get("warm_s", 0):.2f} s '
                  f'| {peak_gb:.2f} GB |')
    print("Warm median of three seeded 3-iteration reconstructions on one "
          "H100 node.  Peak is the largest per-device peak, which is the "
          "busiest device.")


def print_docs_table(by_variant):
    """The third table, in the shape of the one in usr_multi_gpu.rst.  That
    page is reStructuredText, so this prints a list-table rather than
    markdown, and times only: the page's table has no memory column."""
    print("\n### paste into mbirtorch/docs/source/usr_multi_gpu.rst "
          "-- times only")
    print("The page's table has one row per volume and one column per device "
          "count.  It names no geometry, and the numbers below are PARALLEL "
          "BEAM.  Whoever edits the page should either say so in the caption "
          "or add a geometry column; the cone times are in the two tables "
          "above.")
    print()
    print(".. list-table::")
    print("   :header-rows: 1")
    print("   :widths: 34 22 22 22")
    print()
    print("   * - Volume")
    for n_dev in counts():
        print(f'     - {n_dev} device{"" if n_dev == 1 else "s"}')
    for cell in cells():
        print(f"   * - {cell[0]} x {cell[1]} x {cell[2]}")
        for n_dev in counts():
            row = by_variant.get(variant_id("parallel", cell, n_dev))
            print(f'     - {row["warm_s"]:.2f} s' if row else "     - -")
    print()
    print("Warm median of three seeded 3-iteration reconstructions on one "
          "H100 node.")


def summarize(rows, plan, out_path):
    """The three tables a person reads, and the instrument-health accounting
    the exit code comes from.

    These are two different things and this function keeps them apart.  A slow
    variant, a wide spread, a hot GPU and a peak that moved are FINDINGS: they
    are printed and none of them touches the exit code.  A variant that
    produced no row, ran on the wrong device count, bound the wrong kind of
    projection body, read an unverified sinogram, or disagreed with its
    count-1 sibling is an instrument failure, because it did not measure what
    the plan said it would.
    """
    print(f"\n===== mg27 the reference-timing re-anchor ({out_path}) =====")
    broken, findings = [], []
    by_variant, stages = {}, []

    header = (f'{"variant":<{VARIANT_COL}}{"pin":>4}{"dev":>5}{"cold s":>9}'
              f'{"warm s":>9}{"spread":>8}{"peak GB":>9}{"batch":>7}'
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
        print(f'{row["variant"]:<{VARIANT_COL}}{row["n_dev"]:>4}'
              f'{row.get("realized_n_devices", "-"):>5}'
              f'{_fmt(row.get("cold_s"), 9, "f", 2)}'
              f'{_fmt(row.get("warm_s"), 9, "f", 2)}'
              f'{spread_text:>8}'
              f'{_fmt(row.get("peak_bytes", 0) / 2 ** 30, 9, "f", 2)}'
              f'{str(row.get("forward_pixel_batch", "-")):>7}'
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
    # error is not reported twice.
    reported = {item.split("|", 1)[0] for item in broken}
    for cfg in plan:
        name = cfg.get("variant")
        if name and name not in by_variant and name not in reported:
            broken.append(f"{name}|no row")

    for row in stages:
        print(f'\nstaged {row["geometry"]} {tuple(row["cell"])}: '
              f'md5 {row.get("sino_md5", "-")}'
              f'{"  (reused from disk)" if row.get("reused") else ""}')

    value_records = check_cross_counts(by_variant, broken)

    for cell, section in zip(report_cells(),
                             ("section 5.1", "section 5.2")):
        print_overview_table(by_variant, cell, section)
    print_docs_table(by_variant)

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print(f"  BROKEN {item}")
    else:
        print("  every planned variant ran, realized its pinned count, bound "
              "the expected projection bodies, read a verified sinogram, and "
              "agreed with its count-1 sibling")
    for item in findings:
        print(f"  finding (not gated) {item}")
    if not findings:
        print("  no thermal or throttle findings")

    return dict(healthy=not broken, broken=broken, findings=findings,
                variants={name: dict(warm_s=row.get("warm_s"),
                                     warm_spread=row.get("warm_spread"),
                                     peak_bytes=row.get("peak_bytes"),
                                     realized_n_devices=row.get(
                                         "realized_n_devices"))
                          for name, row in by_variant.items()},
                cross_count=value_records)


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
