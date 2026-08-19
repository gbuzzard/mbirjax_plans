"""mg42a -- WHERE A RECONSTRUCTION'S REAL PEAK SITS, AND WHICH REGION CARRIES
IT.

WHY THIS RUN EXISTS.  The library carries a closed-form memory ledger
(`mbirtorch/_memory_ledger.py`) that predicts each device's peak bytes before a
reconstruction allocates anything, and the automatic device count is chosen
from that prediction.  The prediction has been checked against measured peaks
as a single number per device: modeled over measured, in or out of a band.  A
single number says the model is high somewhere; it does not say WHERE.  This
run measures where.  It runs reconstructions with the library's calibration
mode on, and around named regions of the library it samples the per-device
memory counters, so each arm ends with both readings side by side: the
whole-run peak the mode reports, and a per-region attribution of what carried
it.

The three questions it answers are the measurement questions in the design
note in the plans repository (`plans/torch_port/active/ledger_calibration_
design.md`, section 2).  They are named A, B and C here and every arm id
starts with its question's letter.

    QUESTION A, the back-projection attribution.  At the 1024-class on two
    devices, what does the back-projection worker region actually hold?  The
    ledger charges that region a `back batch` term beside a `back output`
    term, and the reading wanted is the worker region's own transient against
    that pair.

    QUESTION B, the three-device over-read.  At the 512-class on one, two and
    three devices, which ledger term over-charges as the count rises?  The
    calibration ratios at three devices read as high as 1.417 against a band
    whose top is 1.30, and the per-term breakdown this run records beside each
    ratio is what names the term.

    QUESTION C, the lead-device transient.  At the 1024-class on four devices,
    in a fresh process, does device 0 transiently hold more than the
    one-device arm's own peak?  A nightly reading said it exceeded it by about
    3.1 GB, out of a harness whose counter accumulates across a whole process.
    The first thing to establish is whether that reproduces in a process that
    does one reconstruction and nothing else.  If it does, the region that
    carries it names the term the ledger is missing.

THE INSTRUMENT: RESET-FREE WATERMARK SAMPLING.  `torch.cuda.max_memory_
allocated(dev)` is a monotone high-water mark.  This probe NEVER resets it.
Around every instrumented region it records, for every visible device, four
numbers: the watermark before, the watermark after, the live bytes before, and
the live bytes after.

    Why reset-free.  The obvious instrument -- reset the counter at the start
    of each region and read it at the end -- makes a defect possible that
    cannot be noticed from its output.  Once a region has reset the counter,
    every later reading is relative to that reset, so the run's own whole-run
    peak is no longer the maximum of what the regions reported: it is whatever
    the LAST region happened to accumulate.  A table built that way can be
    internally consistent, add up, and describe nothing.  A phase probe in the
    device-policy work did exactly this and printed an impossible table before
    anyone noticed.  With no resets the arithmetic cannot go wrong: every
    sample reads one counter that only ever rises, so the maximum over all
    sampled watermarks IS the whole-run peak, by construction rather than by
    agreement.

    What a region's numbers mean.  A region whose after-watermark exceeds its
    before-watermark is a region in which the process peak ROSE, and the rise
    is the direct evidence that this region carried the peak.  The region's
    transient on a device reads as (watermark after - live bytes before): what
    the region added on top of what was already resident when it was entered.

    No synchronize calls anywhere in the sampling.  These are host-side
    allocator counters, and a synchronize would change what overlaps, so the
    probe would be measuring its own instrument.

WHY THE LIBRARY'S CALIBRATION MODE COEXISTS IN THE SAME PROCESS.  Every arm
runs with MBIRTORCH_MEMORY_CALIBRATION=1.  That mode resets the peak counters
exactly ONCE, at the settle -- immediately after the device layout is chosen
and before the sinogram, the weights or anything else is placed -- and at the
end of the reconstruction it compares its modeled peak against the counter it
reset.  Its one reset therefore lands before every wrapped region that holds
anything, so the mode and this probe read the same monotone counter over the
same scope, and the two readings are directly comparable.  Each arm records
both: the mode's own modeled-versus-measured row fields, and the wrapper
samples that attribute the same peak to regions.

    The one region whose samples straddle that reset is the settle itself,
    whose first sample is taken before the mode has reset anything.  Every
    sample carries a flag saying which side of the reset it was taken on.  In
    a fresh process the pre-reset side holds almost nothing -- the model is
    built and the policy is closed form -- so it cannot inflate anything, but
    it is labeled rather than assumed.

THE REGIONS, and the library seam each one wraps.  All seven are resolved by
import path in the worker and wrapped BEFORE the model is built.  A seam that
cannot be resolved FAILS its arm with an error naming the seam: a silently
unwrapped region is the vacuity this whole instrument exists to avoid.

    settle                      TomographyModel._apply_device_policy
    place sinogram-like         TomographyModel._shard_sinogram
    place recon-like            TomographyModel._shard_recon
    back projection (sharded)   TomographyModel._sparse_back_project_sharded
    back worker call            Projectors.sparse_back_project_view_range
    band reduce                 _sharding.sum_band_to_owner
    forward projection (sharded)
                                TomographyModel._sparse_forward_project_cylinders

The sinogram placement fires twice per reconstruction, once for the sinogram
and once for the weights; the sample's sequence number tells them apart.  The
back worker call and the band reduce fire many times per back projection, so
the worker keeps a RUNNING AGGREGATE per region and device -- the largest
transient, the largest watermark rise, the largest watermark, the smallest
live-bytes-before, and the call count -- plus the first forty raw samples of
each region for inspection.  Three regions exist only on a multi-device
placement (the two back-worker regions and the cylinder forward), and a
single-device arm records them as attached with no calls.

THE ARMS, ten of them, each in a fresh subprocess with the count pinned by
MBIRTORCH_NUM_DEVICES, the calibration mode on, the campaign's seed reset
immediately before the call, and the campaign's transmission-shaped weights:

    b512_<geometry>_n1,n2,n3    3-iteration reconstruction, 512-class
    a1024_<geometry>_n2         1-iteration reconstruction, 1024-class
    c1024_<geometry>_n4         3-iteration reconstruction, 1024-class

for both geometries.  Question A wants one reconstruction's worth of big back
projections and no more, so its arms run a single iteration: the direct
reconstruction and the hessian both back-project at full size inside it.

WHAT EACH ARM RECORDS BESIDES THE SAMPLES.  The ledger's own prediction, read
after the reconstruction: the per-device modeled peak, and for each device the
dominant phase with its largest terms by name and bytes -- that per-term
breakdown is what question B needs beside the ratio.  The measured whole-run
peak per device, read from the counter the mode reset at the settle, before
anything else in the worker allocates.  The modeled-over-measured ratio per
device and its verdict against the library's own calibration band.  And the
usual health fields: the realized device count, the staged sinogram's md5, and
the padding witness.  The whole ledger read is guarded, so a failure there
records its traceback instead of losing the arm's samples.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
produced a row, realized the device count it was pinned to, read an
md5-verified sinogram, attached every wrapper seam, and exercised every region
that its device count can exercise.  It is NOT a verdict on what was measured.
Ratios, band verdicts, transients, the size of the lead device's peak and the
comparison against the nightly's reading are all FINDINGS: they are printed in
full and none of them touches the exit code.  A person reads the tables.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It edits no library file
and flips no default.  The only thing it changes at runtime is that it wraps
seven library functions in the worker process to sample counters around them;
each wrapper calls through to the original and returns its result unchanged.

STAGING.  The sinograms are the ones the reference-timing runs already staged:
same directory, same names, same md5 sidecars, verified on every load and
recorded on every row.  A missing file is built here by the same recipe --
phantom, forward projection, float32 -- and kept.  The 1024-class files are
about 4.1 GB each and the 512-class files about 0.34 GB.

ORDER AND WALL ESTIMATE.  The 512-class arms run first, so a harness defect
surfaces in minutes rather than after an hour.  Within a cell, each geometry's
staging runs immediately before that geometry's arms, and the arms run in
question order with counts ascending.  Six 3-iteration 512-class arms, two
1-iteration 1024-class arms and two 3-iteration 1024-class arms, each paying
its own compiles in its own process, come to roughly an hour.  The sbatch asks
for two.

OUTPUT.  One jsonl under MG42_RESULTS, named mg42a_ledger_<node>_<stamp>.jsonl:
a run-header row, one row per stage and per arm, and a summary row, flushed as
they finish so a job cut short still yields everything it completed.  The run
then prints one block per arm (the per-device modeled/measured table, the
dominant phase and its terms, and the regions ranked by watermark growth), the
question-C block, and a closing paste-ready summary.

Run:
    <torch python> mg42a_ledger_probe.py         on a 4-GPU node
    MG42_DRY=1 <python> mg42a_ledger_probe.py    print the plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized
token is an error, not a silent skip.
    MG42_RESULTS=<dir>      where the jsonl goes, and where a newly staged
                            sinogram goes when MG42_SINO_DIR is left unset,
                            which is the smoke's case
    MG42_SINO_DIR=<dir>     where the staged sinograms are read from, and
                            written to if one is missing.  Defaults to the
                            reference-timing run's directory on scratch, so
                            this run reuses those files rather than spending an
                            hour rebuilding them
    MG42_SMOKE=1            the local smoke: one tiny cell on the CPU, one
                            iteration, and no arm above two devices.  The
                            counters do not exist there, so every snapshot
                            reads zeros and the tables print with zeros; what
                            the smoke exercises is the wrapping, the seam
                            names, the plan, the row fields and the report.
                            The calibration mode is still set, so the mode's
                            own path runs too
    MG42_DRY=1              print the arm plan and exit, importing no torch
    MG42_ARMS=a,b           a subset, by arm id, e.g. b512_parallel_n3
"""

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG42_SMOKE", "0") == "1"
DRY = os.environ.get("MG42_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

GEOMETRIES = ("parallel", "cone")

#: The two cells, in RUN order: the cheap one first, so a harness defect
#: surfaces in minutes rather than after an hour of 1024-class work.
CELL_512 = (512, 448, 384)
CELL_1024 = (1024, 1008, 992)
CELLS = (CELL_512, CELL_1024)

#: The smoke's stand-in.  A parallel model at (8, 24, 20) reconstructs a
#: (20, 20, 24) volume, which two virtual CPU devices can still split on both
#: axes, and which runs on a laptop in seconds.  All three questions share it.
SMOKE_CELL = (8, 24, 20)
#: The most devices a smoke arm runs on.  Two is enough to reach every region:
#: the sharded back workers, the band reduce and the cylinder forward all
#: exist only above one device.  A question that asks for more devices than
#: this runs at this count in the smoke, and its arm id says the count it ran.
SMOKE_MAX_DEVICES = 2

#: The three questions, in the order the arms are declared.  Each names its
#: cell, the device counts it asks for, the iteration count its arms run, and
#: what it is asking, which is printed with the plan.
QUESTIONS = (
    dict(letter="a", cell=CELL_1024, counts=(2,), iterations=1,
         asks="what the back-projection worker region holds, against the "
              "ledger's back batch and back output charges"),
    dict(letter="b", cell=CELL_512, counts=(1, 2, 3), iterations=3,
         asks="which ledger term over-charges as the device count rises"),
    dict(letter="c", cell=CELL_1024, counts=(4,), iterations=3,
         asks="whether the lead device transiently holds more than the "
              "one-device arm's own peak, in a fresh process"),
)

# ── the reconstruction protocol, the campaign's ───────────────────────────────
#: The seed, reset immediately before every reconstruction.  It is the
#: campaign's, so these arms reconstruct what the recorded rows reconstructed.
VCD_SEED = 13
#: The cone construction: the source-detector and source-isocenter distances
#: are these multiples of the detector channel count.
CONE_SDD_PER_CHANNEL = 4.0
CONE_SID_PER_CHANNEL = 2.0

# ── the instrument ────────────────────────────────────────────────────────────
#: The library seams wrapped in every arm's worker, as
#: (module path, class name or None, attribute, region name).  A None class
#: means the attribute lives on the module itself.  The order is the order the
#: wrappers go on and the order regions are reported in.
SEAMS = (
    ("mbirtorch.tomography_model", "TomographyModel", "_apply_device_policy",
     "settle"),
    ("mbirtorch.tomography_model", "TomographyModel", "_shard_sinogram",
     "place sinogram-like"),
    ("mbirtorch.tomography_model", "TomographyModel", "_shard_recon",
     "place recon-like"),
    ("mbirtorch.tomography_model", "TomographyModel",
     "_sparse_back_project_sharded", "back projection (sharded)"),
    ("mbirtorch.projectors", "Projectors", "sparse_back_project_view_range",
     "back worker call"),
    ("mbirtorch._sharding", None, "sum_band_to_owner", "band reduce"),
    ("mbirtorch.tomography_model", "TomographyModel",
     "_sparse_forward_project_cylinders", "forward projection (sharded)"),
)

#: The calibration mode's one reset, wrapped as an EPOCH MARKER rather than as
#: a region: it holds nothing and takes no time, and what it is needed for is
#: to stamp every later sample as taken after the reset.  Without the stamp,
#: the settle's first sample -- the only one taken on the other side of it --
#: would be indistinguishable from the rest.
RESET_SEAM = ("mbirtorch._memory_ledger", None, "calibration_start")

#: Regions that must fire in EVERY arm.  A region that attached and never ran
#: measured nothing, and a table of regions that never ran is exactly the
#: vacuity this instrument exists to avoid, so this is instrument health.
#:
#: The back worker call is here rather than below because the single-device
#: back driver routes through the same function the banded workers call, so it
#: runs at every device count.
REGIONS_ALWAYS = ("settle", "place sinogram-like", "place recon-like",
                  "back projection (sharded)", "back worker call")
#: Regions that exist only above one device: a single-device back projection
#: has no partials to reduce, and a single-device forward never transfers
#: cylinders.  Required on a multi-device arm only.
REGIONS_MULTI_DEVICE = ("band reduce", "forward projection (sharded)")

#: How many raw samples of each region the row keeps, beside the running
#: aggregate.  The back worker call fires thousands of times in a 1024-class
#: arm; keeping them all would make the row unreadable and the file large,
#: while the first few dozen are what a sequence trail is read from.
RAW_SAMPLE_KEEP = 40
#: How many of an arm's kept raw samples the question-C trail prints, in
#: sequence order.  The rest stay in the jsonl.
TRAIL_PRINT_LIMIT = 80
#: How many of a dominant phase's terms the row records and the report prints.
LEDGER_TERM_COUNT = 6
#: How many rows the per-arm region ranking prints.
TOP_REGION_ROWS = 8

# ── recorded context, not gates ───────────────────────────────────────────────
#: The one-device peak the nightly recorded at the 1024 class, which question C
#: compares device 0 against.  It is READ FROM A DIFFERENT HARNESS AND A
#: DIFFERENT PROCESS than anything this run measures, and the comparison is
#: printed with that caveat attached every time it is printed.
NIGHTLY_ONE_DEVICE_PEAK_GIB = 23.4
NIGHTLY_PEAK_CAVEAT = (
    "read from a DIFFERENT harness and a DIFFERENT process than this run: its "
    "counter accumulates over a whole process rather than over one "
    "reconstruction, so the two numbers are not the same measurement and the "
    "difference between them is not by itself a finding")

#: The padding witness.  504 is a four-device slice band at the 2048 cell and
#: 512 is what the rounding must turn it into; a tree without the padding
#: either has no such function or returns 504.  Recorded on every row so a
#: reader can tell which tree produced these numbers without leaving the jsonl.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512

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

RESULTS_DIR = os.environ.get(
    "MG42_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
#: Where the staged sinograms live.  The default is the reference-timing run's
#: directory, whose files this run reuses: rebuilding a 4 GB sinogram would
#: cost an hour and, because the forward kernel's atomics are not bit-exact
#: across runs, would also change what is being reconstructed.  The smoke has
#: no such directory, so it stages into its own results directory.
SINO_DIR = os.environ.get(
    "MG42_SINO_DIR",
    RESULTS_DIR if SMOKE
    else "/scratch/gautschi/buzzard/torch_p3/results/mg27_reference")

RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 22                        # wide enough for the longest arm id printed
# ──────────────────────────────────────────────────────────────────────────────


# ── the arms ──────────────────────────────────────────────────────────────────
def question_cell(question):
    """The cell one question's arms run at.  The smoke collapses all three onto
    one tiny cell, so a laptop exercises every arm's code path in seconds."""
    return SMOKE_CELL if SMOKE else question["cell"]


def question_counts(question):
    """The device counts one question's arms run at.

    The smoke cannot reach four devices usefully, so a count above the smoke's
    ceiling runs at the ceiling instead, and duplicates that creates collapse.
    The arm id is built from the count that RAN, so a smoke arm never claims a
    count it did not have.
    """
    if not SMOKE:
        return tuple(question["counts"])
    kept = []
    for count in question["counts"]:
        count = min(count, SMOKE_MAX_DEVICES)
        if count not in kept:
            kept.append(count)
    return tuple(kept)


def question_iterations(question):
    return 1 if SMOKE else question["iterations"]


def run_cells():
    """The cells in RUN order, cheap first, with any cell no question asks for
    dropped."""
    wanted = [question_cell(q) for q in QUESTIONS]
    ordered = (SMOKE_CELL,) if SMOKE else CELLS
    return tuple(cell for cell in ordered if cell in wanted)


def arm_id(question, cell, geometry, n_dev):
    return f'{question["letter"]}{cell[0]}_{geometry}_n{n_dev}'


def all_arms():
    """Every arm, in RUN order: the cheap cell first, then within a cell each
    geometry's arms together, in question order with counts ascending."""
    arms = []
    for cell in run_cells():
        for geometry in GEOMETRIES:
            for question in QUESTIONS:
                if question_cell(question) != cell:
                    continue
                for n_dev in question_counts(question):
                    arms.append(dict(
                        kind="arm", question=question["letter"],
                        asks=question["asks"], geometry=geometry,
                        cell=list(cell), n_dev=n_dev,
                        iterations=question_iterations(question),
                        arm=arm_id(question, cell, geometry, n_dev),
                        job_id=arm_id(question, cell, geometry, n_dev)))
    return arms


def all_arm_ids():
    return [cfg["arm"] for cfg in all_arms()]


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
            raise ValueError(f"{env_name}: {token!r} is not an arm of this "
                             f"run.  The valid ids are: {', '.join(allowed)}")
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
    """One file per (geometry, cell), under the shared sinogram directory.

    The name is the reference-timing run's, deliberately: this run reads that
    run's files, and a different name here would mean rebuilding gigabytes to
    reconstruct the same thing.  The cell's view count is in the name, so a
    smoke run and a production run can share a directory without either reading
    the other's bytes.
    """
    return os.path.join(SINO_DIR, f"mg27_sino_{geometry}_{cell[0]}.npy")


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
    """The campaign's weighting formula, one dtype, every arm.

    These weights are not uniform, so the weighted branch is what runs, which
    is both what a real reconstruction does and what the ledger's charges are
    written against.
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


# ── the model, built the campaign's way ───────────────────────────────────────
def build_model(geometry, cell, cpu_devices=None):
    """The model construction the campaign's rows were all measured from.

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
        # Falling through to parallel beam would measure parallel beam and
        # record the result under another geometry's name, which is the one way
        # a reading can be wrong without anything looking wrong.
        raise ValueError(f"this run has no model construction for geometry "
                         f"{geometry!r}")
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def pin_devices_for(n_dev):
    """The explicit device list an arm needs, or None.

    None on CUDA, where MBIRTORCH_NUM_DEVICES does the pinning.  A list of
    virtual CPU devices on the smoke, where the environment pin cannot.
    """
    return None if DEVICE == "cuda" else ["cpu"] * n_dev


# ── the instrument: reset-free watermark sampling ─────────────────────────────
#: The device indices every snapshot reads, set once per worker before any
#: region is wrapped.  On CUDA it is every visible device, because question C
#: is about one device holding more than its share and a reading restricted to
#: the placement could not see a device that was not supposed to hold anything.
#: On the smoke there are no counters, so it is the arm's own count and every
#: reading is zero.
_SNAPSHOT_DEVICES = []

#: Whether the calibration mode's one reset has happened yet, and at which
#: sample.  Set by the epoch marker; read by every sample.
_RESET_MARK = dict(done=False, at_seq=None)


def set_snapshot_devices(n_dev):
    """Fix which devices every snapshot reads, once, before wrapping."""
    global _SNAPSHOT_DEVICES

    if DEVICE == "cuda":
        import torch

        _SNAPSHOT_DEVICES = list(range(torch.cuda.device_count()))
    else:
        _SNAPSHOT_DEVICES = list(range(n_dev))
    return list(_SNAPSHOT_DEVICES)


def snapshot():
    """The per-device watermark and live bytes, right now.

    Two host-side allocator counters per device and NOTHING else.  No
    synchronize: these counters are updated on the host at allocation time, and
    a synchronize here would change what overlaps, so the probe would be
    measuring its own instrument rather than the run.

    The watermark is never reset by this probe, so it only rises; the live
    reading is the resident bytes the region was entered with.  On the smoke
    there are no CUDA counters, so both read zero and every caller still runs.
    """
    if DEVICE != "cuda":
        zeros = [0] * len(_SNAPSHOT_DEVICES)
        return dict(watermark=list(zeros), live=list(zeros))

    import torch

    return dict(
        watermark=[int(torch.cuda.max_memory_allocated(d))
                   for d in _SNAPSHOT_DEVICES],
        live=[int(torch.cuda.memory_allocated(d)) for d in _SNAPSHOT_DEVICES])


class RegionSamples:
    """Every region's running aggregate, plus the first few raw samples.

    The back worker call and the band reduce fire thousands of times in a
    1024-class arm, so a list of full samples would be both enormous and
    unreadable.  What a reader needs from those is the extremes, which fold
    into an aggregate as the samples arrive, and a short prefix in sequence
    order, which is what a trail is read from.

    The back workers run on a thread pool, so appends arrive from several
    threads at once and the aggregate is updated under a lock.  The sequence
    number the caller computed is advisory for that reason; the authoritative
    one is assigned here, under the same lock, so the trail's order is the
    order the samples actually arrived in.
    """

    def __init__(self, device_count):
        self.device_count = int(device_count)
        self._lock = threading.Lock()
        self._count = 0
        self._agg = {}                 # region -> aggregate dict
        self._raw = []                 # kept samples, all regions, in order

    def __len__(self):
        return self._count

    def _new_aggregate(self, region):
        return dict(
            region=region, calls=0, calls_before_reset=0,
            per_device=[dict(device_index=d, max_transient_bytes=0,
                             max_growth_bytes=0, max_watermark_after_bytes=0,
                             min_live_before_bytes=None)
                        for d in range(self.device_count)])

    def append(self, sample):
        with self._lock:
            seq = self._count
            self._count += 1
            sample["seq"] = seq
            sample["after_reset"] = bool(_RESET_MARK["done"])
            region = sample["region"]
            before, after = sample["before"], sample["after"]
            agg = self._agg.get(region)
            if agg is None:
                agg = self._agg[region] = self._new_aggregate(region)
            agg["calls"] += 1
            if not sample["after_reset"]:
                agg["calls_before_reset"] += 1
            for index, entry in enumerate(agg["per_device"]):
                if index >= len(after["watermark"]):
                    continue
                watermark_after = after["watermark"][index]
                watermark_before = before["watermark"][index]
                live_before = before["live"][index]
                # The transient is what the region added on top of what was
                # already resident when it was entered.  The growth is how far
                # the process high-water mark moved inside it, which is the
                # direct evidence that this region carried the peak.
                entry["max_transient_bytes"] = max(
                    entry["max_transient_bytes"], watermark_after - live_before)
                entry["max_growth_bytes"] = max(
                    entry["max_growth_bytes"], watermark_after - watermark_before)
                entry["max_watermark_after_bytes"] = max(
                    entry["max_watermark_after_bytes"], watermark_after)
                if entry["min_live_before_bytes"] is None:
                    entry["min_live_before_bytes"] = live_before
                else:
                    entry["min_live_before_bytes"] = min(
                        entry["min_live_before_bytes"], live_before)
            kept = sum(1 for s in self._raw if s["region"] == region)
            if kept < RAW_SAMPLE_KEEP:
                self._raw.append(sample)

    def summary(self, region_order):
        """The aggregates, in the order the regions were wrapped, with a region
        that never fired still present and carrying a zero call count.  A
        region missing from the table would be indistinguishable from a region
        that was never wrapped."""
        return [self._agg.get(region) or self._new_aggregate(region)
                for region in region_order]

    def raw_samples(self):
        """The kept samples, in arrival order."""
        return list(self._raw)


def wrap(holder, attr, region_name, samples):
    """Sample the counters around one library function, and call through.

    The wrapper adds two counter reads per device on each side and nothing
    else: it does not synchronize, does not copy, and does not touch the
    result.  The sample is recorded in a finally block, so a region that raised
    is still recorded rather than silently missing.
    """
    original = getattr(holder, attr)

    def wrapped(*args, **kwargs):
        before = snapshot()             # per-device watermark + live bytes
        try:
            return original(*args, **kwargs)
        finally:
            samples.append(dict(region=region_name, before=before,
                                after=snapshot(),
                                seq=len(samples)))

    setattr(holder, attr, wrapped)
    return original


def _resolve_seam(module_path, class_name, attr):
    """The object the seam's attribute lives on, or a reason it cannot be
    wrapped.  Returns (holder, None) or (None, reason)."""
    import importlib

    try:
        module = importlib.import_module(module_path)
    except Exception as exc:                                      # noqa: BLE001
        return None, f"{module_path} does not import: {type(exc).__name__}: {exc}"
    holder = module
    if class_name is not None:
        holder = getattr(module, class_name, None)
        if holder is None:
            return None, f"{module_path} has no {class_name}"
    if not hasattr(holder, attr):
        where = module_path if class_name is None \
            else f"{module_path}.{class_name}"
        return None, f"{where} has no {attr}"
    return holder, None


def install_wrappers(samples):
    """Wrap every seam, before the model is built, or fail naming the seam.

    Every seam is resolved FIRST and the missing ones are reported together, so
    a tree that moved two functions says so once instead of once per run.  A
    silently unwrapped region would leave a table that looks complete and
    attributes the peak to the wrong place, which is the failure this
    instrument exists to prevent, so an unresolvable seam stops the arm.
    """
    resolved, missing = [], []
    for module_path, class_name, attr, region in SEAMS:
        holder, reason = _resolve_seam(module_path, class_name, attr)
        if holder is None:
            missing.append(f"{region}: {reason}")
        else:
            resolved.append((holder, attr, region))
    marker_holder, marker_reason = _resolve_seam(*RESET_SEAM)
    if marker_holder is None:
        missing.append(f"the calibration reset marker: {marker_reason}")
    if missing:
        raise RuntimeError(
            "these instrumented seams could not be resolved in the library "
            "under test, so this arm would attribute the peak to an incomplete "
            "set of regions: " + "; ".join(missing))

    for holder, attr, region in resolved:
        wrap(holder, attr, region, samples)

    # The epoch marker.  It records nothing about memory: it stamps every later
    # sample as taken after the calibration mode's one reset, so the settle's
    # first sample -- the only one on the other side of it -- is labeled rather
    # than assumed.
    original_reset = getattr(marker_holder, RESET_SEAM[2])

    def marked_reset(*args, **kwargs):
        try:
            return original_reset(*args, **kwargs)
        finally:
            _RESET_MARK["done"] = True
            _RESET_MARK["at_seq"] = len(samples)

    setattr(marker_holder, RESET_SEAM[2], marked_reset)
    return [region for _holder, _attr, region in resolved]


def band_verdict(ratio, band):
    """The library's own wording for a ratio against its calibration band."""
    if ratio is None or band is None:
        return "-"
    low, high = band
    if ratio < low:
        return "UNDER"
    if ratio > high:
        return "over"
    return "ok"


def _finite(value):
    """A float that json can write, or None.  A measured peak of zero makes the
    ratio infinite, which is a real reading and not a number."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# ── the worker: one stage or one arm, in its own process ──────────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the job is.

    The calibration check here is the REVERSE of the campaign's timing runs.
    Those runs read the peak counters themselves, so the mode had to be absent.
    This run reads the counters the mode reset, so the mode is REQUIRED on every
    arm -- and equally required OFF on a staging job, which reads no counter and
    must not own one.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    calibration = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION")
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_seed=VCD_SEED,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "MBIRTORCH_NUM_DEVICES is set as on CUDA, and "
                                 "the count is realized by "
                                 "configure_devices(devices=['cpu', ...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=calibration)
    result["invalid_reasons"] = []

    if cfg.get("kind") == "arm":
        result["calibration_on_ok"] = (calibration == "1")
        if not result["calibration_on_ok"]:
            result["invalid_reasons"].append(
                f"MBIRTORCH_MEMORY_CALIBRATION is {calibration!r} and every arm "
                f"REQUIRES '1': the mode's one reset at the settle is the "
                f"origin every wrapper sample and the whole-run peak are read "
                f"against")
    else:
        result["calibration_off_ok"] = calibration in (None, "", "0")
        if not result["calibration_off_ok"]:
            result["invalid_reasons"].append(
                f"MBIRTORCH_MEMORY_CALIBRATION is {calibration!r} on a staging "
                f"job; staging reads no counter and must not own one")

    # The padding witness, recorded so a reader can tell which tree produced
    # these numbers from the row alone.  Recorded, not gated: the sbatch asserts
    # it before any arm runs.
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


def _read_staged(result, geometry, cell):
    """The staged sinogram, md5-verified.

    Returns the array, or None when the row already records why this arm cannot
    run.  An arm that reconstructed a different array than its siblings did not
    measure what the plan said, so a mismatch stops the arm rather than
    producing a row that looks comparable and is not.
    """
    import numpy as np

    path = _sino_path(geometry, cell)
    result["sino_path"] = path
    result["sino_dir"] = SINO_DIR
    if not _staged(path):
        result["invalid_reasons"].append(f"no staged sinogram at {path}")
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


def run_stage(cfg):
    """Make sure ONE (geometry, cell) sinogram is on disk, and verify it.

    Normally the file is already there from the reference-timing runs and this
    only re-hashes it, which is the point: these arms then reconstruct the same
    bytes those runs did, so a peak measured here and a time recorded there are
    about the same input.  When the file is absent it is built here by the same
    recipe -- phantom, forward projection, float32 -- and kept.

    The staging process is pinned to one device, runs with the calibration mode
    off, and takes its projection on a freshly built model, so nothing here is
    a multi-device run and nothing here touches a peak counter.
    """
    import numpy as np

    import mbirtorch

    result, _cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    path = _sino_path(geometry, cell)
    result["sino_path"] = path
    result["sino_dir"] = SINO_DIR

    if _staged(path):
        # Already on disk.  Verify it rather than rebuild it: the forward
        # kernel's atomics make a regenerated sinogram non-identical at the e-7
        # class, so a rebuild would silently change what every arm of this pair
        # reconstructs.
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


def read_ledger(result, model, weights):
    """The ledger's own prediction for this arm, per device, with the dominant
    phase and its largest terms.

    Read AFTER the reconstruction and guarded whole.  The ledger is closed form
    and costs nothing to rebuild, and the arm's samples are the expensive part
    of the row, so a failure here records its traceback and keeps everything
    else rather than losing the arm.
    """
    try:
        ledger = model._build_memory_ledger(workload="recon", weights=weights)
        result["ledger_devices"] = [str(d) for d in ledger.devices]
        result["modeled_peak_bytes"] = [int(b)
                                        for b in ledger.per_device_peaks()]
        dominant = []
        for index in range(len(ledger.devices)):
            phase = ledger.dominant_phase(index)
            dominant.append(dict(
                device=str(ledger.devices[index]),
                phase=str(phase.name),
                phase_bytes=int(phase.per_device[index]),
                terms=[[str(name), int(value)] for name, value
                       in phase.dominant_terms(index,
                                               count=LEDGER_TERM_COUNT)]))
        result["ledger_dominant"] = dominant
        result["ledger_phases"] = [
            dict(phase=str(p.name),
                 per_device=[int(b) for b in p.per_device])
            for p in ledger.phases]
        result["ledger_ok"] = True
    except Exception:                                             # noqa: BLE001
        result["ledger_ok"] = False
        result["ledger_error"] = traceback.format_exc()[-3000:]


def run_arm(cfg):
    """One arm: one reconstruction, in a fresh process, with the calibration
    mode on and every region wrapped.

    ORDER, and all of it is load-bearing.  The wrappers go on BEFORE the model
    is built, because the settle happens inside the first reconstruction call
    and a wrapper installed after it would miss the region question C is most
    interested in.  The measured peak is read immediately after the
    reconstruction returns and before anything else in this function allocates,
    so it covers the same scope the mode's own table reports.  The ledger is
    read after that, because rebuilding it allocates nothing but reading it
    before the reconstruction would price a layout that had not settled.

    There is no cold pass and no repeat.  A repeat would measure a process that
    already holds a reconstruction's cached allocations, and the questions are
    all about the first one.
    """
    import numpy as np
    import torch

    from mbirtorch import _memory_ledger

    result, cuda = _base_result(cfg)
    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])
    iterations = int(cfg["iterations"])

    # ── the instrument goes on first ─────────────────────────────────────────
    devices_sampled = set_snapshot_devices(n_dev)
    samples = RegionSamples(len(devices_sampled))
    result["snapshot_device_indices"] = list(devices_sampled)
    result["regions_wrapped"] = install_wrappers(samples)
    result["seams_ok"] = True
    result["calibration_band"] = list(_memory_ledger.CALIBRATION_BAND)

    model = build_model(geometry, cell, cpu_devices=pin_devices_for(n_dev))
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    sinogram = _read_staged(result, geometry, cell)
    if sinogram is None:
        return result
    weights = _weights(sinogram)

    np.random.seed(VCD_SEED)
    started = time.perf_counter()
    _recon, _stats = model.recon(sinogram, weights=weights,
                                 max_iterations=iterations,
                                 stop_threshold_change_pct=0.0)
    result["recon_s"] = time.perf_counter() - started

    # ── the measured whole-run peak, before this function allocates ──────────
    # The counter was reset by the calibration mode at the settle, so this
    # covers settle to here, which is the scope the mode's own table reports.
    devices_now = list(model.sino_placement.devices)
    realized = [str(d) for d in devices_now]
    if cuda:
        measured = [int(torch.cuda.max_memory_allocated(d)) for d in devices_now]
        measured_visible = [int(torch.cuda.max_memory_allocated(d))
                            for d in devices_sampled]
    else:
        measured = [0] * len(devices_now)
        measured_visible = [0] * len(devices_sampled)
    result["measured_peak_bytes"] = measured
    result["measured_peak_all_visible_bytes"] = measured_visible

    # ── the mode's own table, from its row fields rather than its log ────────
    mode_rows = getattr(model, "last_memory_calibration", None) or []
    result["mode_rows"] = [dict(device=str(device), modeled=int(modeled),
                                measured=int(measured_bytes),
                                ratio=_finite(ratio))
                           for device, modeled, measured_bytes, ratio
                           in mode_rows]

    # ── the realized layout ──────────────────────────────────────────────────
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
    # The axis lengths are passed explicitly, because the TRIVIAL single-device
    # placement is built without one and shard_ranges() raises on it.  The CPU
    # smoke cannot catch this: its explicit device list builds a placement that
    # carries the length, so only a CUDA one-device arm reaches the bare form.
    sinogram_shape = [int(s) for s in model.get_params("sinogram_shape")]
    result["view_blocks"] = [end - start for _d, (start, end)
                             in model.sino_placement.shard_ranges(
                                 sinogram_shape[0])]
    result["slice_blocks"] = [end - start for _d, (start, end)
                              in model.recon_placement.shard_ranges(
                                  int(result["recon_shape"][2]))]

    # ── the ledger's prediction, and the ratio against it ────────────────────
    read_ledger(result, model, weights)
    modeled = result.get("modeled_peak_bytes") or []
    ratios, verdicts = [], []
    for index, measured_bytes in enumerate(measured):
        if index >= len(modeled):
            ratios.append(None)
            verdicts.append("-")
            continue
        ratio = (modeled[index] / measured_bytes) if measured_bytes else None
        ratios.append(_finite(ratio))
        verdicts.append(band_verdict(ratios[-1],
                                     result["calibration_band"]))
    result["modeled_over_measured"] = ratios
    result["band_verdicts"] = verdicts

    # ── the regions ──────────────────────────────────────────────────────────
    result["regions"] = samples.summary(result["regions_wrapped"])
    result["region_raw"] = samples.raw_samples()
    result["total_samples"] = len(samples)
    result["reset_at_seq"] = _RESET_MARK["at_seq"]
    result["reset_seen"] = bool(_RESET_MARK["done"])
    if result.get("calibration_on_ok") and not result["reset_seen"]:
        # The mode was on and its reset never happened, so the measured peak
        # covers the whole process rather than settle to end.  That is a
        # different measurement than every other row's, and it would be
        # invisible in the number itself.
        result["invalid_reasons"].append(
            "the calibration mode was on and never reset the peak counters, "
            "so this arm's measured peak covers the whole process rather than "
            "the reconstruction the other arms' peaks cover")

    fired = {entry["region"] for entry in result["regions"] if entry["calls"]}
    required = list(REGIONS_ALWAYS)
    if len(realized) > 1:
        required += list(REGIONS_MULTI_DEVICE)
    silent = [region for region in required if region not in fired]
    result["regions_required"] = required
    result["regions_fired"] = sorted(fired)
    result["regions_ok"] = not silent
    if silent:
        result["invalid_reasons"].append(
            f"these regions attached and never ran on a {len(realized)}-device "
            f"arm, so nothing was attributed to them: {', '.join(silent)}")
    return result


def run_job(cfg):
    """One stage or one arm, in its own process, with a health sample on either
    side of it.

    A new process per job is not tidiness.  The peak counters are per process
    and this run reads them as a monotone high-water mark, so a second arm in
    the same interpreter would start from the first arm's watermark and every
    region reading would be a tail of it.  Compiled and hand-written kernel
    bodies are cached at module level for the life of a process as well, and
    the wrappers themselves are installed on library classes, so they would
    stack.
    """
    before = sample_gpu_health()
    started = time.time()
    try:
        result = (run_stage(cfg) if cfg["kind"] == "stage" else run_arm(cfg))
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

    Both variables are popped and then set, so a value exported by the shell
    cannot reach a job that asked for something else.  The count is set on the
    smoke too, exactly as on CUDA, so the subprocess protocol under test is the
    same one the real run uses; the smoke's count is then realized by an
    explicit CPU device list, because the pin acts only through the device
    policy and that policy short-circuits below two visible CUDA devices.

    The calibration mode is ON for an arm and OFF for a staging job.  An arm
    needs it because its reset is the origin every reading is taken from; a
    staging job reads no counter and must not own one.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # the shipped configuration
    env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    env["MBIRTORCH_MEMORY_CALIBRATION"] = "1" if cfg["kind"] == "arm" else ""
    return env


def spawn(cfg):
    """Run one configuration in a NEW interpreter.

    The row goes through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.  On an hour-long job that is
    the difference between watching progress and waiting in the dark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_mg42a_cfg_{cfg['job_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_mg42a_out_{cfg['job_id']}.json")
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
    """Every job, in run order: each (geometry, cell) staging immediately before
    that pair's own arms, cheap cell first."""
    keep = _strict_subset("MG42_ARMS", all_arm_ids())
    arms = [cfg for cfg in all_arms() if cfg["arm"] in keep]
    plan, staged = [], set()
    for cfg in arms:
        cell = tuple(cfg["cell"])
        key = (cfg["geometry"], cell)
        if key not in staged:
            staged.add(key)
            plan.append(dict(kind="stage", geometry=cfg["geometry"],
                             cell=list(cell), n_dev=1,
                             job_id=f'stage_{cfg["geometry"]}_{cell[0]}'))
        plan.append(cfg)
    if not plan:
        raise ValueError("MG42_ARMS selects no arm")
    return plan


def print_plan(plan):
    arms = [c for c in plan if c["kind"] == "arm"]
    stages = [c for c in plan if c["kind"] == "stage"]
    print(f"mg42a the memory-ledger calibration probe: {len(arms)} arm(s) and "
          f"{len(stages)} staged sinogram(s), device {DEVICE}")
    print(f"  jsonl -> {RESULTS_DIR}")
    print(f"  staged sinograms read from (and written to, if missing) "
          f"-> {SINO_DIR}")
    print("  every arm runs in a fresh subprocess with the library's "
          "calibration mode on, which resets the peak counters once at the "
          "settle; this probe never resets them again, so the maximum over "
          "its samples is the whole-run peak by construction")
    for question in QUESTIONS:
        print(f'  question {question["letter"].upper()}: '
              f'{question["asks"]}')
    print("  the regions sampled around, in wrapping order: "
          + ", ".join(region for _m, _c, _a, region in SEAMS))
    print(f"  a region that attaches and never runs fails the arm; the three "
          f"multi-device regions ({', '.join(REGIONS_MULTI_DEVICE)}) are "
          f"required only above one device")
    if SMOKE:
        print(f"  SMOKE: one tiny cell {SMOKE_CELL}, one iteration, no arm "
              f"above {SMOKE_MAX_DEVICES} device(s), and no CUDA counters -- "
              f"every snapshot reads zeros and the tables print with zeros")
    total_gib = 0.0
    for cfg in stages:
        cell = tuple(cfg["cell"])
        total_gib += cell[0] * cell[1] * cell[2] * 4 / 2 ** 30
    print(f"  staged sinograms total about {total_gib:.2f} GiB and are kept")
    header = (f'  {"job":<{ARM_COL}}{"pin":>5}{"iters":>7}{"cell":>20}'
              f'{"geometry":>10}  what it does')
    print(header)
    for cfg in plan:
        if cfg["kind"] == "stage":
            what = "verifies this pair's sinogram, building it if absent"
            iters = "-"
        else:
            what = (f'one reconstruction, every region sampled '
                    f'(question {cfg["question"].upper()})')
            iters = str(cfg["iterations"])
        print(f'  {cfg["job_id"]:<{ARM_COL}}{cfg["n_dev"]:>5}{iters:>7}'
              f'{str(tuple(cfg["cell"])):>20}{cfg["geometry"]:>10}  {what}')
    print("  no library file is edited and no default is flipped: the only "
          "runtime change is the seven wrappers, each of which calls through "
          "and returns its original's result unchanged")


def main():
    plan = build_plan()
    if DRY:
        print_plan(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg42a_ledger_{RUN_LABEL}_{stamp}.jsonl")
    print_plan(plan)
    print(f"\nrunning -> {out_path}", flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        header = dict(row="run_header", script="mg42a_ledger_probe.py",
                      node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
                      python=sys.executable, results_dir=RESULTS_DIR,
                      sino_dir=SINO_DIR, geometries=list(GEOMETRIES),
                      vcd_seed=VCD_SEED,
                      questions=[dict(letter=q["letter"],
                                      cell=list(question_cell(q)),
                                      counts=list(question_counts(q)),
                                      iterations=question_iterations(q),
                                      asks=q["asks"]) for q in QUESTIONS],
                      regions=[region for _m, _c, _a, region in SEAMS],
                      regions_always=list(REGIONS_ALWAYS),
                      regions_multi_device=list(REGIONS_MULTI_DEVICE),
                      raw_sample_keep=RAW_SAMPLE_KEEP,
                      nightly_one_device_peak_gib=NIGHTLY_ONE_DEVICE_PEAK_GIB,
                      nightly_peak_caveat=NIGHTLY_PEAK_CAVEAT,
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
            elif cfg["kind"] == "arm":
                ratios = [r for r in row.get("modeled_over_measured") or []
                          if r is not None]
                print(f'    recon {row.get("recon_s", 0):.1f}s  '
                      f'peak {max(row.get("measured_peak_bytes") or [0]) / 2 ** 30:.2f} GB  '
                      f'worst ratio '
                      f'{max(ratios) if ratios else float("nan"):.3f}  '
                      f'{row.get("realized_n_devices", "-")} device(s)  '
                      f'{row.get("total_samples", 0)} sample(s)', flush=True)
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
    one, so the columns line up whether an arm produced a number or not."""
    if value is None:
        return f'{"-":>{width}}'
    return f"{value:>{width}.{prec}{kind}}"


def _gb(num_bytes):
    """Bytes as GB, in the ledger's own units so the two tables compare."""
    if num_bytes is None:
        return "-"
    return f"{num_bytes / 2 ** 30:.2f}"


def worst_device(row):
    """The device whose ratio is the reading to quote for this arm.

    An UNDER is the failure the ledger exists to prevent, so it wins if one
    exists; otherwise the largest ratio is the over-read this pass is chasing.
    When no ratio exists at all -- the smoke, where there are no counters --
    the device with the largest measured peak is named anyway, with no ratio,
    so the rest of the line still says something.

    Returns (device index, ratio or None, verdict).
    """
    ratios = row.get("modeled_over_measured") or []
    verdicts = row.get("band_verdicts") or []
    pairs = [(i, r) for i, r in enumerate(ratios) if r is not None]
    if not pairs:
        measured = row.get("measured_peak_bytes") or []
        if not measured:
            return None, None, "-"
        return max(range(len(measured)), key=lambda i: measured[i]), None, "-"
    unders = [(i, r) for i, r in pairs
              if i < len(verdicts) and verdicts[i] == "UNDER"]
    index, ratio = min(unders, key=lambda p: p[1]) if unders \
        else max(pairs, key=lambda p: p[1])
    verdict = verdicts[index] if index < len(verdicts) else "-"
    return index, ratio, verdict


def ranked_regions(row):
    """Every (region, device) pair, ranked by how far the process high-water
    mark rose inside that region.

    Growth is the ranking rather than the transient because growth is what
    answers "which region carried the peak": a region can hold a lot and still
    move the watermark not at all, if an earlier region already held more.
    """
    ranked = []
    for entry in row.get("regions") or []:
        for device in entry.get("per_device") or []:
            ranked.append(dict(region=entry["region"], calls=entry["calls"],
                               device_index=device["device_index"],
                               growth=device["max_growth_bytes"],
                               transient=device["max_transient_bytes"],
                               watermark=device["max_watermark_after_bytes"],
                               live_before=device["min_live_before_bytes"]))
    ranked.sort(key=lambda item: (-item["growth"], -item["transient"],
                                  item["region"], item["device_index"]))
    return ranked


def top_region_name(row):
    ranked = ranked_regions(row)
    if not ranked:
        return "-"
    top = ranked[0]
    return f'{top["region"]} (dev {top["device_index"]})'


def print_arm_block(row):
    """One arm's three tables: the per-device modeled against measured, the
    ledger's dominant phase with its largest terms, and the regions ranked by
    how far each one moved the high-water mark."""
    band = row.get("calibration_band")
    band_text = "-" if not band else f"{band[0]:.2f} to {band[1]:.2f}"
    print(f'\n===== {row.get("arm", row.get("job_id", "?"))}  '
          f'(question {str(row.get("question", "?")).upper()}: '
          f'{row.get("asks", "")}) =====')
    print(f'  cell {tuple(row.get("cell", ()))}, '
          f'{row.get("realized_n_devices", "-")} device(s), '
          f'{row.get("iterations", "-")} iteration(s), '
          f'recon {_fmt(row.get("recon_s"), 1, "f", 1).strip()} s, '
          f'{row.get("total_samples", 0)} sample(s), band {band_text}')
    if row.get("ledger_ok") is False:
        print("  THE LEDGER READ FAILED for this arm; the samples below still "
              "stand.  The traceback is in the jsonl.")

    modeled = row.get("modeled_peak_bytes") or []
    measured = row.get("measured_peak_bytes") or []
    ratios = row.get("modeled_over_measured") or []
    verdicts = row.get("band_verdicts") or []
    devices = row.get("realized_devices") or []
    header = (f'  {"dev":>4}  {"device":<10}{"modeled GB":>13}'
              f'{"measured GB":>13}{"ratio":>9}{"verdict":>10}')
    print(header)
    print("  " + "-" * (len(header) - 2))
    for index in range(len(measured)):
        name = devices[index] if index < len(devices) else str(index)
        print(f'  {index:>4}  {name:<10}'
              f'{(_gb(modeled[index]) if index < len(modeled) else "-"):>13}'
              f'{_gb(measured[index]):>13}'
              f'{_fmt(ratios[index] if index < len(ratios) else None, 9, "f", 3)}'
              f'{(verdicts[index] if index < len(verdicts) else "-"):>10}')
    mode_rows = row.get("mode_rows") or []
    if mode_rows:
        print("  the mode's own table, from its recorded rows: "
              + "; ".join(f'{r["device"]} {_gb(r["modeled"])}/'
                          f'{_gb(r["measured"])} GB '
                          f'= {r["ratio"]:.3f}' if r["ratio"] is not None
                          else f'{r["device"]} ratio -'
                          for r in mode_rows))

    for index, entry in enumerate(row.get("ledger_dominant") or []):
        terms = ", ".join(f'{name} {_gb(value)} GB'
                          for name, value in entry.get("terms") or [])
        print(f'  dominant phase on dev {index} ({entry["device"]}): '
              f'{entry["phase"]}  {_gb(entry["phase_bytes"])} GB')
        print(f'      {terms if terms else "no terms recorded"}')

    ranked = ranked_regions(row)
    print(f'  the regions that moved the high-water mark furthest '
          f'({min(TOP_REGION_ROWS, len(ranked))} of {len(ranked)} '
          f'region-device pairs)')
    header = (f'  {"region":<30}{"dev":>5}{"calls":>8}{"growth GB":>12}'
              f'{"transient GB":>14}{"watermark GB":>14}')
    print(header)
    print("  " + "-" * (len(header) - 2))
    for item in ranked[:TOP_REGION_ROWS]:
        print(f'  {item["region"]:<30}{item["device_index"]:>5}'
              f'{item["calls"]:>8}{_gb(item["growth"]):>12}'
              f'{_gb(item["transient"]):>14}{_gb(item["watermark"]):>14}')
    silent = [r for r in row.get("regions") or [] if not r["calls"]]
    if silent:
        print("  regions that attached and never ran: "
              + ", ".join(r["region"] for r in silent))


def print_question_c(by_arm):
    """Question C's own block: device 0's trail through the regions, in the
    order the samples were taken, and the one comparison the question is
    about.

    The comparison is printed with its caveat every time, because the number it
    is against was not read by this run, this harness or this process.
    """
    print("\n===== question C: does the lead device hold more than the "
          "one-device arm's peak, in a fresh process? =====")
    arms = [row for row in by_arm.values() if row.get("question") == "c"]
    if not arms:
        print("  no question-C arm ran in this job, so there is nothing to "
              "read here.  The arms it looks for are the ones whose id starts "
              "with 'c'.")
        return
    for row in sorted(arms, key=lambda r: r.get("arm", "")):
        measured = row.get("measured_peak_bytes") or []
        lead_gib = (measured[0] / 2 ** 30) if measured else None
        print(f'\n-- {row.get("arm")} --')
        if lead_gib is None:
            print("  no measured peak on device 0")
        else:
            gap = lead_gib - NIGHTLY_ONE_DEVICE_PEAK_GIB
            print(f'  device 0 measured {lead_gib:.2f} GiB in this fresh '
                  f'process, against the nightly one-device reading of '
                  f'{NIGHTLY_ONE_DEVICE_PEAK_GIB:.2f} GiB: '
                  f'{gap:+.2f} GiB.')
            print(f'  CAVEAT: the {NIGHTLY_ONE_DEVICE_PEAK_GIB:.1f} GiB is '
                  f'{NIGHTLY_PEAK_CAVEAT}.')
        raw = sorted(row.get("region_raw") or [], key=lambda s: s["seq"])
        reset_seq = row.get("reset_at_seq")
        print(f'  device 0 through the regions, in sample order '
              f'({min(len(raw), TRAIL_PRINT_LIMIT)} of {len(raw)} kept '
              f'samples; the calibration mode reset the counters at sample '
              f'{reset_seq if reset_seq is not None else "-"})')
        header = (f'  {"seq":>6}  {"region":<30}{"live before GB":>16}'
                  f'{"watermark after GB":>20}{"epoch":>14}')
        print(header)
        print("  " + "-" * (len(header) - 2))
        for sample in raw[:TRAIL_PRINT_LIMIT]:
            before = sample.get("before") or {}
            after = sample.get("after") or {}
            live = (before.get("live") or [None])[0]
            watermark = (after.get("watermark") or [None])[0]
            epoch = "after reset" if sample.get("after_reset") else "PRE-RESET"
            print(f'  {sample["seq"]:>6}  {sample["region"]:<30}'
                  f'{_gb(live):>16}{_gb(watermark):>20}{epoch:>14}')
        if len(raw) > TRAIL_PRINT_LIMIT:
            print(f'  ... {len(raw) - TRAIL_PRINT_LIMIT} further kept sample(s) '
                  f'are in the jsonl')


def print_observations(by_arm, plan):
    """The closing block: one line per arm, in plan order, short enough to
    paste into a note."""
    print("\n===== paste-ready observations =====")
    print("One line per arm: the worst device's modeled-over-measured ratio "
          "and its verdict, the phase the ledger says dominates that device, "
          "and the region that moved the high-water mark furthest.  Every "
          "number here is a FINDING and none of them touches the exit code.")
    for cfg in plan:
        if cfg["kind"] != "arm":
            continue
        row = by_arm.get(cfg["arm"])
        if row is None:
            print(f'  {cfg["arm"]:<{ARM_COL}}  no row')
            continue
        index, ratio, verdict = worst_device(row)
        # The ledger's per-device entries are in the placement's own order, so
        # the worst device's entry is read by index rather than matched by
        # name: two devices can carry the same name (the CPU smoke does) and a
        # name match would then quote the wrong one.
        dominant = row.get("ledger_dominant") or []
        phase = (dominant[index]["phase"]
                 if index is not None and index < len(dominant) else "-")
        ratio_text = "-" if ratio is None else f"{ratio:.3f}"
        device_text = "-" if index is None else f"dev {index}"
        print(f'  {cfg["arm"]:<{ARM_COL}}  worst {device_text} ratio '
              f'{ratio_text} ({verdict})  dominant phase {phase}  '
              f'top region {top_region_name(row)}')


def summarize(rows, plan, out_path):
    """The blocks a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A ratio
    outside the band, a large transient, a lead device holding more than its
    share, a hot GPU: all FINDINGS, all printed, none of them gated.  An arm
    that produced no row, ran on the wrong device count, read an unverified
    sinogram, failed to attach a seam, or left a region that its device count
    should have exercised with no calls did not measure what the plan said it
    would, and that is an instrument failure.
    """
    print(f"\n===== mg42a the memory-ledger calibration probe ({out_path}) "
          f"=====")
    broken, findings = [], []
    by_arm, stages = {}, []

    header = (f'{"arm":<{ARM_COL}}{"pin":>4}{"dev":>5}{"recon s":>9}'
              f'{"peak GB":>9}{"ratio":>8}{"verdict":>9}{"samples":>9}'
              f'{"checks":>20}')
    print(header)
    print("-" * len(header))
    for row in rows:
        job_id = row.get("job_id", "?")
        if row.get("error"):
            print(f'{job_id:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            broken.append(f"{job_id}|error")
            continue
        if row.get("kind") == "stage":
            stages.append(row)
            broken.extend(f"{job_id}|{reason}"
                          for reason in row.get("invalid_reasons") or [])
            continue
        by_arm[row["arm"]] = row
        marks = []
        for name, flag in (("dev", row.get("devices_ok")),
                           ("md5", row.get("sino_md5_ok")),
                           ("cal", row.get("calibration_on_ok")),
                           ("seam", row.get("seams_ok")),
                           ("region", row.get("regions_ok"))):
            if flag is False:
                marks.append(f"{name}:FAIL")
        _index, ratio, verdict = worst_device(row)
        peaks = row.get("measured_peak_bytes") or [0]
        print(f'{row["arm"]:<{ARM_COL}}{row["n_dev"]:>4}'
              f'{row.get("realized_n_devices", "-"):>5}'
              f'{_fmt(row.get("recon_s"), 9, "f", 1)}'
              f'{_fmt(max(peaks) / 2 ** 30, 9, "f", 2)}'
              f'{_fmt(ratio, 8, "f", 3)}{verdict:>9}'
              f'{row.get("total_samples", 0):>9}'
              f'{(",".join(marks) if marks else "ok"):>20}')
        for reason in row.get("invalid_reasons") or []:
            print(f"    ARM CHECK FAIL: {reason}")
            broken.append(f'{row["arm"]}|{reason}')
        if verdict == "over":
            findings.append(f'{row["arm"]}: the ledger reads {ratio:.3f} of '
                            f'the measured peak, above the band')
        if verdict == "UNDER":
            findings.append(f'{row["arm"]}: the ledger reads {ratio:.3f} of '
                            f'the measured peak, BELOW the band')
        if row.get("gpu_hot"):
            findings.append(f'{row["arm"]}: GPU hot during this arm')
        if row.get("gpu_throttle"):
            findings.append(f'{row["arm"]}: throttle reasons '
                            f'{row["gpu_throttle"]}')

    # Every PLANNED arm produced a row.  Read off the plan rather than off the
    # rows: an arm whose subprocess died before writing anything leaves no row
    # to notice its absence in.  An arm already reported above as an error is
    # not reported twice.
    reported = {item.split("|", 1)[0] for item in broken}
    for cfg in plan:
        name = cfg.get("arm")
        if name and name not in by_arm and name not in reported:
            broken.append(f"{name}|no row")

    for row in stages:
        print(f'\nstaged {row["geometry"]} {tuple(row["cell"])}: '
              f'md5 {row.get("sino_md5", "-")}'
              f'{"  (reused from disk)" if row.get("reused") else "  (built by this run)"}')

    for cfg in plan:
        if cfg["kind"] == "arm" and cfg["arm"] in by_arm:
            print_arm_block(by_arm[cfg["arm"]])

    print_question_c(by_arm)
    print_observations(by_arm, plan)

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print(f"  BROKEN {item}")
    else:
        print("  every planned arm ran, realized its pinned count, read a "
              "verified sinogram, attached every wrapper seam, and exercised "
              "every region its device count can exercise")
    for item in findings:
        print(f"  finding (not gated) {item}")
    if not findings:
        print("  no findings outside the band and no thermal or throttle "
              "findings")

    return dict(healthy=not broken, broken=broken, findings=findings,
                arms={name: dict(
                    question=row.get("question"),
                    realized_n_devices=row.get("realized_n_devices"),
                    recon_s=row.get("recon_s"),
                    modeled_peak_bytes=row.get("modeled_peak_bytes"),
                    measured_peak_bytes=row.get("measured_peak_bytes"),
                    modeled_over_measured=row.get("modeled_over_measured"),
                    band_verdicts=row.get("band_verdicts"),
                    top_region=top_region_name(row),
                    total_samples=row.get("total_samples"))
                      for name, row in by_arm.items()})


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
