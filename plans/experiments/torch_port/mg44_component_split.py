"""mg44 -- WHICH COMPONENT OF A RECONSTRUCTION CARRIES THE TWO-DEVICE LOSS,
AND HOW THAT MOVES WITH PROBLEM SIZE.

WHY THIS RUN EXISTS.  The multiaxis geometry reconstructs SLOWER on two
devices at some sizes and faster at others, and nothing recorded so far says
which part of the reconstruction the loss lives in.  These are the warm walls
this run is measured against, read on 2026-08-18 from
/scratch/gautschi/buzzard/torch_p3/results/mg26_floors/_out_arm_*.json on the
gautschi cluster:

    multiaxis   (512, 448, 384)     n1  11.4 s   n2  32.6 s    0.35x
    multiaxis   (768, 672, 576)     n1  56.3 s   n2  38.5 s    1.46x
    multiaxis  (1024, 1008, 992)    n1 309.9 s   n2 388.8 s    0.80x
    translation (256, 1900, 3000)   n1  12.6 s   n2  14.2 s    0.89x

The ratio is the one the floors tool prints: the one-device warm median over
the two-device warm median, so above 1.0 means two devices are faster.  A
single wall per arm says two devices lost; it does not say what lost.  This
run times each component CALL of the reconstruction at one and two devices and
reports which component's warm time carries the difference, and how that moves
from the 512 class to the 1024 class.

THE INSTRUMENT: TWO VIEWS OF EVERY REGION, AND NO SYNCHRONIZATION.  Around
every wrapped region the probe records both of these, per call:

    HOST    time.perf_counter() immediately before and after the call-through.
    DEVICE  one pair of torch.cuda.Event(enable_timing=True) per relevant
            device, recorded on that device's default stream at entry and at
            exit.

Why BOTH, and why events rather than a host timer alone.  The VCD subset loop
has no host synchronization at any device count -- the line search stays on
device and the update is applied on device (see the docstring of
``create_vcd_subset_updater``; the loop's one host sync per iteration is the
statistics readout in ``vcd_recon``).  A host timer around a region therefore
measures how long it took to ENQUEUE that region's work, which can be a small
fraction of how long the device spends on it, or -- when dispatch, a thread
pool handoff or a compile guard is the cost -- almost all of it.  A region
whose host time is large beside its device time is itself a finding, so both
numbers are kept and both are printed.

Why default-stream event pairs partition a device's timeline exactly.  All
library compute lands on each device's default stream: the per-device worker
threads never change streams, and the side copy streams used by the cylinder
transfer are opened, waited on by their consumers and closed again, so nothing
that this probe times runs anywhere else.  A pair of events recorded on the
default stream therefore brackets exactly the work enqueued between them, and
the elapsed time between them is that device's busy plus stall time inside the
region.  Nothing is synchronized to read them: the probe records events during
the reconstruction and resolves them only AFTER the reconstruction call has
returned, on the synchronize the timing protocol already does.  A synchronize
inside the run would change what overlaps, which is the very thing being
measured.

What a device number means, and what it does not.  A region's device time is
wall time on that device's stream between two markers.  It INCLUDES time the
stream sat idle waiting for something else, so per-region device times inside
one iteration add up to the iteration's own device span (up to the glue between
them, which the report prints as a residual line).  It is not a busy-only
measure and is not read as one anywhere in the report.

THE REGIONS, and the library seam each one wraps.  Every seam is resolved
BEFORE the model is built.  A seam that cannot be resolved FAILS its arm with
an error naming the seam: a silently unwrapped region would leave a table that
looks complete and attributes the loss to the wrong place.

    phases (events on every placement device)
        settle                  TomographyModel._apply_device_policy
        init direct recon       <model class>.direct_recon
        initial error state     TomographyModel._initial_error_state
        hessian diagonal        TomographyModel.compute_hessian_diagonal
        iteration               TomographyModel.vcd_partition_iterator
        iteration stats         TomographyModel._iteration_stats
    the two projection funnels (events on every placement device)
        forward projection      TomographyModel.sparse_forward_project
        back projection         TomographyModel.sparse_back_project
    cross-device primitives
        halo exchange           _sharding.exchange_qggmrf_halos
        band reduce             _sharding.sum_band_to_owner  (owner device
                                only: that is where the reduce runs)
    the per-call projector bodies (events on the one device the call is on)
        forward body call       Projectors.sparse_forward_project_view_range
        back body call          Projectors.sparse_back_project_view_range
    the fan-out seam
        fan-out <name>          _sharding.run_per_device

The class-method seams are patched on whichever class in the MODEL FAMILY'S
method resolution order actually holds the attribute, not on TomographyModel
unconditionally.  This matters: MultiAxisParallelModel and TranslationModel
both OVERRIDE ``direct_recon``, so patching the base class alone would wrap a
method that never runs and record an empty region.

THE FAN-OUT SEAM is what splits one iteration into its components without any
closure access.  Every per-device fan-out in the library goes through
``_sharding.run_per_device``, and the worker it is handed names the component:
the label is the last part of the worker's qualified name, so one seam
separates prior_worker, direction_worker, lin_quad_worker, positivity_worker,
apply_worker, sino_worker (the statistics), dots_worker (the initial error),
the cylinder forward's worker, and the anonymous workers, which are labeled by
the function they are written inside plus the word lambda.

MOVE_SHARD is tallied rather than sampled.  ``_sharding.move_shard`` fires many
times per subset for scalar-sized moves, so a per-call sample of it would be
mostly bookkeeping.  The row records a count and a summed host time per
reconstruction call instead, which is the reading that matters: whether the
scalar traffic between components is a visible cost.

NESTING.  Body calls sit inside the funnels, the funnels sit inside an
iteration, and fan-outs occur at every depth.  Nothing in the report ever sums
across those depths.  Regions are grouped into three levels -- phases,
in-iteration components, inner detail -- and each table stays inside one level.
Every sample also records whether it was taken inside an iteration, because
both funnels also fire outside one (in the direct reconstruction, the initial
error state and the Hessian), and mixing those into the in-iteration table
would break the per-iteration residual.

THE ARMS, each in a fresh subprocess, cheap first so a harness defect surfaces
in minutes rather than after an hour:

    ma512_n1, ma512_n2                  multiaxis (512, 448, 384), wrapped
    ma512_n1_control, ma512_n2_control  the same, with NO wrappers installed
    tr_n1, tr_n2                        translation (256, 1900, 3000), wrapped
    ma768_n1, ma768_n2                  multiaxis (768, 672, 576), wrapped
    ma1024_n1, ma1024_n2                multiaxis (1024, 1008, 992), wrapped

The two control arms are the instrument-overhead check: identical protocol,
identical pins, no wrappers.  Their warm medians beside the wrapped arms' say
what the instrument itself costs.  A ratio within a few percent of 1.0 means
the wrapped numbers describe the library; a larger one is a finding about this
probe rather than about the library.

THE PROTOCOL IS THE FLOORS TOOL'S, DELIBERATELY.  Model construction, the
seed, the weights, the reconstruction call, the timing envelope and the
cold-then-warm structure are all copied from ``measure()`` in
``dev_scripts/refresh_widening_floors.py``, because the anomaly this run
explains was measured by that tool and a different protocol would measure a
different thing.  In particular: the multiaxis model takes two angles per view
(azimuth over half a turn, elevation swept across +/- 0.5 radians); the
translation model takes the production 16x16 grid with 24.0 and 16.0 spacing
and both source distances at half the smaller detector extent; numpy's seed is
reset to 13 immediately before EVERY reconstruction call; the weights are
exp(-sino / (2 * max(sino))) in float32; the call is
recon(sino, weights=weights, max_iterations=3, stop_threshold_change_pct=0.0);
and the timed envelope includes the synchronize on every placement device and
the gather of the output to numpy, exactly as the floors tool times it.  One
cold pass, then three warm repeats, with the warm median and spread recorded.

WHAT THIS RUN DOES AND DOES NOT DO.  It measures.  It edits no library file and
flips no default.  The only runtime change is that named library functions are
wrapped in the worker process; each wrapper reads two clocks, records two
events, calls through, and returns its original's result untouched.

STAGING.  The sinograms are the ones the floors refresh already staged: same
directory, same names.  There are no md5 sidecars there, so each arm hashes the
file it loaded and records the digest on its row, and the driver checks at the
end that every arm of a cell read the same bytes.  A missing file is built here
by the floors tool's own recipe -- phantom, forward projection, float32 -- in a
one-device staging job that runs before that cell's arms.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
produced a row, realized the device count it was pinned to, resolved every
seam, read the same sinogram as its siblings, ran without the calibration mode,
and exercised the regions its device count can exercise.  Which component
carries the loss, how far a wall drifted from the recorded one, whether a
region is host-bound, whether a card was hot: all FINDINGS.  They are printed
in full and none of them touches the exit code.  A person reads the tables.

OUTPUT.  One jsonl under MG44_RESULTS, named
mg44_component_<node>_<stamp>.jsonl: a run-header row, one row per staging job
and per arm, and a summary row, flushed as they finish so a job cut short still
yields everything it completed.  The run then prints the wall table, the
instrument-overhead check, the component split per cell, the translation block,
the host-bound list and a paste-ready line per arm.

Run:
    <torch python> mg44_component_split.py        on a 2-GPU node
    MG44_DRY=1 <python> mg44_component_split.py   print the plan and stop

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized token
is an error, not a silent skip.
    MG44_RESULTS=<dir>      where the jsonl goes, and where a newly staged
                            sinogram goes when MG44_SINO_DIR is left unset,
                            which is the smoke's case
    MG44_SINO_DIR=<dir>     where the staged sinograms are read from, and
                            written to if one is missing.  Defaults to the
                            floors refresh's directory on scratch, so this run
                            reconstructs the same bytes the recorded walls were
                            measured on
    MG44_SMOKE=1            the local smoke: tiny cells on virtual CPU devices,
                            one iteration, one warm repeat.  There are no CUDA
                            events there, so every device column reads zero and
                            only the host columns carry numbers; what the smoke
                            proves is that every seam resolves and every region
                            fires, and it ASSERTS both
    MG44_DRY=1              print the arm plan and exit, importing no torch
    MG44_ARMS=a,b           a subset, by arm id, e.g. ma512_n1,ma512_n2
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG44_SMOKE", "0") == "1"
DRY = os.environ.get("MG44_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

#: The reconstruction protocol, the floors refresh's.  Changing either number
#: would measure a different amount of work than the recorded walls describe.
SEED = 13
ITERATIONS = 1 if SMOKE else 3
WARM_REPEATS = 1 if SMOKE else 3

#: The cells.  The three multiaxis cells are the shared floors ladder's top
#: three; the translation cell is the production one the floors tool measures.
MA_512 = (512, 448, 384)
MA_768 = (768, 672, 576)
MA_1024 = (1024, 1008, 992)
TR_CELL = (256, 1900, 3000)

#: The smoke's stand-ins.  Both are the floors tool's own smoke cells, so the
#: translation grid below already covers the small one.
SMOKE_MA_CELL = (8, 24, 20)
SMOKE_TR_CELL = (16, 40, 32)

#: The translation grid and spacing behind each translation cell, copied from
#: the floors refresh: ``cell -> (num_x, num_z, x_spacing, z_spacing)``.  A
#: translation scan's views are grid positions rather than angles, so the same
#: sinogram shape could come from many grids and the grid has to be written
#: down rather than derived.
TRANSLATION_SPECS = {
    (256, 1900, 3000): (16, 16, 24.0, 16.0),
    (16, 40, 32): (4, 4, 3.0, 2.0),
}

# ── the walls this run is measured against ────────────────────────────────────
#: Warm walls measured 2026-08-18 by the floors refresh (mg26, job 15342578),
#: read from that run's own arm rows on 2026-08-19.  They are QUOTED here so
#: the report can print this run's ratio to them beside every arm; nothing
#: gates on them.
RECORDED_SOURCE = ("/scratch/gautschi/buzzard/torch_p3/results/mg26_floors/"
                   "_out_arm_*.json (mg26, job 15342578; measured 2026-08-18, "
                   "read 2026-08-19)")
RECORDED_WALLS = {
    ("multiaxis", MA_512): dict(n1_cold=20.4, n1_warm=11.4,
                                n2_cold=48.2, n2_warm=32.6, ratio=0.35),
    ("multiaxis", MA_768): dict(n1_cold=64.4, n1_warm=56.3,
                                n2_cold=53.9, n2_warm=38.5, ratio=1.46),
    ("multiaxis", MA_1024): dict(n1_cold=323.8, n1_warm=309.9,
                                 n2_cold=402.8, n2_warm=388.8, ratio=0.80),
    ("translation", TR_CELL): dict(n1_cold=24.5, n1_warm=12.6,
                                   n2_cold=31.1, n2_warm=14.2, ratio=0.89),
}
#: How far this run's warm wall may sit from the recorded one before the report
#: prints a WARNING line.  The multiaxis path in this tree is byte-identical to
#: the one those walls were measured on, so a large drift means the ruler or the
#: node changed rather than the library.  A WARNING is a finding: it does not
#: touch the exit code.
WALL_DRIFT_WARN = 0.10

# ── the instrument ────────────────────────────────────────────────────────────
#: The class-method seams, as (attribute, region name).  Each is resolved by
#: walking the MODEL FAMILY'S method resolution order and patching the first
#: class whose own ``__dict__`` holds the attribute -- the base class for most
#: of them, the geometry subclass for ``direct_recon``, which both families
#: override.
MODEL_SEAMS = (
    ("_apply_device_policy", "settle"),
    ("direct_recon", "init direct recon"),
    ("_initial_error_state", "initial error state"),
    ("compute_hessian_diagonal", "hessian diagonal"),
    ("vcd_partition_iterator", "iteration"),
    ("_iteration_stats", "iteration stats"),
    ("sparse_forward_project", "forward projection"),
    ("sparse_back_project", "back projection"),
)
#: The per-call projector bodies, patched on ``Projectors``.  Both fire at every
#: device count: the single-device drivers and the multi-device drivers funnel
#: through the same two methods.
PROJECTOR_SEAMS = (
    ("sparse_forward_project_view_range", "forward body call"),
    ("sparse_back_project_view_range", "back body call"),
)
#: The module-level seams in ``mbirtorch._sharding``.  Every library caller
#: reaches these through the module object (``_sharding.sum_band_to_owner``),
#: never through a from-import, so patching the module attribute intercepts all
#: of them.
SHARDING_SEAMS = (
    ("exchange_qggmrf_halos", "halo exchange"),
    ("sum_band_to_owner", "band reduce"),
)
#: The fan-out seam and the host-only tally seam, both in ``_sharding``.
FANOUT_SEAM = "run_per_device"
MOVE_SEAM = "move_shard"

#: The region name every fan-out label is prefixed with, so a fan-out is never
#: confused with a named seam.
FANOUT_PREFIX = "fan-out "

#: Which reporting level each region belongs to.  The levels are chosen so that
#: the regions inside one level do not nest inside each other, which is what
#: makes a table over a level meaningful.  A region not named here lands in
#: ``detail``, which is the safe default: detail is never summed against
#: anything.
REGION_LEVEL = {
    "settle": "phase",
    "init direct recon": "phase",
    "initial error state": "phase",
    "hessian diagonal": "phase",
    "iteration": "phase",
    "iteration stats": "phase",
    "forward projection": "component",
    "back projection": "component",
    "halo exchange": "component",
    "band reduce": "detail",
    "forward body call": "detail",
    "back body call": "detail",
}
#: The fan-outs that are DIRECT children of one subset update or of the
#: partition pass around it.  Together with the two funnels and the halo
#: exchange these tile an iteration, so their per-device times can be subtracted
#: from the iteration's to leave a residual.  Every other fan-out sits inside
#: one of them and is reported as detail.
COMPONENT_FANOUTS = ("prior_worker", "vcd_subset_updater lambda",
                     "direction_worker", "lin_quad_worker",
                     "positivity_worker", "apply_worker")

#: Regions every wrapped arm must exercise at any device count.  A region that
#: attached and never ran measured nothing, and a table of regions that never
#: ran is the vacuity the resolve-or-fail rule exists to avoid.
REGIONS_ALWAYS = ("settle", "init direct recon", "initial error state",
                  "hessian diagonal", "iteration", "iteration stats",
                  "forward projection", "back projection", "halo exchange",
                  "forward body call", "back body call",
                  FANOUT_PREFIX + "prior_worker",
                  FANOUT_PREFIX + "direction_worker",
                  FANOUT_PREFIX + "lin_quad_worker",
                  FANOUT_PREFIX + "apply_worker")
#: Regions that exist only above one device: a single-device back projection has
#: no partials to reduce, and the single-device statistics take the fused
#: single-tensor path rather than fanning out.
REGIONS_MULTI_DEVICE = ("band reduce", FANOUT_PREFIX + "sino_worker")

#: How many raw samples of each region a row keeps beside the aggregate.  The
#: body-call regions fire hundreds of times per reconstruction; keeping them all
#: would make the file large and the row unreadable, while the first few dozen
#: in sequence order are what a call trail is read from.
RAW_SAMPLE_KEEP = 40
#: How many rows each region table prints.  The rest stay in the jsonl.
TABLE_ROWS = 12

# ── recorded context, not gates ───────────────────────────────────────────────
#: The padding witness.  504 is a four-device slice band at the 2048 cell and
#: 512 is what the rounding must turn it into; a tree without the padding either
#: has no such function or returns 504.  Recorded on every row so a reader can
#: tell which tree produced these numbers without leaving the jsonl.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512

# ── the GPU health sample ─────────────────────────────────────────────────────
HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,"
                    "temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

RESULTS_DIR = os.environ.get(
    "MG44_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
#: Where the staged sinograms live.  The default is the floors refresh's own
#: results directory, whose files this run reuses: the recorded walls were
#: measured on those bytes, and the forward kernel's atomics are not bit-exact
#: across runs, so rebuilding would quietly change what is reconstructed.  The
#: smoke has no such directory and stages into its own results directory.
SINO_DIR = os.environ.get(
    "MG44_SINO_DIR",
    RESULTS_DIR if SMOKE
    else "/scratch/gautschi/buzzard/torch_p3/results/mg26_floors")

RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 18                        # wide enough for the longest arm id printed
#: Wide enough for the longest region name, which is a fan-out label carrying
#: the name of the function its anonymous worker is written inside.
REGION_COL = 44
# ──────────────────────────────────────────────────────────────────────────────


# ── the arms ──────────────────────────────────────────────────────────────────
def _cell_for(family, cell):
    """The cell an arm actually runs at.  The smoke collapses every cell of a
    family onto that family's tiny stand-in, so a laptop exercises every arm's
    code path in seconds; the arm ids do not change, so MG44_ARMS selects the
    same names either way."""
    if not SMOKE:
        return cell
    return SMOKE_MA_CELL if family == "multiaxis" else SMOKE_TR_CELL


#: Every arm, in RUN order, as (arm id, family, cell, device count, wrapped).
#: Cheap first: a harness defect surfaces in minutes rather than after an hour
#: of 1024-class work.  The two control arms run right after the cheap wrapped
#: pair, so the instrument-overhead reading is in hand before anything large is
#: spent on it.
ARM_SPECS = (
    ("ma512_n1", "multiaxis", MA_512, 1, True),
    ("ma512_n2", "multiaxis", MA_512, 2, True),
    ("ma512_n1_control", "multiaxis", MA_512, 1, False),
    ("ma512_n2_control", "multiaxis", MA_512, 2, False),
    ("tr_n1", "translation", TR_CELL, 1, True),
    ("tr_n2", "translation", TR_CELL, 2, True),
    ("ma768_n1", "multiaxis", MA_768, 1, True),
    ("ma768_n2", "multiaxis", MA_768, 2, True),
    ("ma1024_n1", "multiaxis", MA_1024, 1, True),
    ("ma1024_n2", "multiaxis", MA_1024, 2, True),
)


def all_arms():
    """Every arm's configuration dict, in run order."""
    arms = []
    for arm, family, cell, n_dev, wrapped in ARM_SPECS:
        arms.append(dict(kind="arm", arm=arm, job_id=arm, family=family,
                         cell=list(_cell_for(family, cell)),
                         declared_cell=list(cell), n_dev=n_dev,
                         wrapped=wrapped, iterations=ITERATIONS,
                         warm_repeats=WARM_REPEATS))
    return arms


def all_arm_ids():
    return [spec[0] for spec in ARM_SPECS]


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
            raise ValueError(f"{env_name}: {token!r} is not an arm of this run."
                             f"  The valid ids are: {', '.join(allowed)}")
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}.  The valid "
                         f"ids are: {', '.join(allowed)}")
    # Normalized to the DECLARED order: the run order is load-bearing (cheap
    # first, controls next), so it must not depend on the order someone typed
    # the tokens in.
    return [name for name in allowed if name in chosen]


# ── the staged sinogram ───────────────────────────────────────────────────────
def _sino_path(family, cell):
    """One file per (family, cell), under the shared sinogram directory.

    The name is the floors refresh's, deliberately: this run reads that tool's
    files, and a different name here would mean rebuilding gigabytes to
    reconstruct the same thing.
    """
    return os.path.join(SINO_DIR,
                        "_sino_{}_{}x{}x{}.npy".format(family, *cell))


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


def _check_sino_dir_writable():
    """Fail early, naming the path, when a sinogram has to be built and the
    directory cannot take it.  A staging job that discovers this after building
    a multi-gigabyte array has already spent the time."""
    try:
        os.makedirs(SINO_DIR, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"the staged-sinogram directory {SINO_DIR} does not exist and "
            f"cannot be created ({exc}).  Set MG44_SINO_DIR to a writable "
            f"directory, or stage the sinograms there first.")
    if not os.access(SINO_DIR, os.W_OK):
        raise RuntimeError(
            f"the staged-sinogram directory {SINO_DIR} is not writable, and a "
            f"sinogram is missing from it.  Set MG44_SINO_DIR to a writable "
            f"directory, or stage the sinograms there first.")


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
    """The floors refresh's weighting formula, one dtype, every arm.

    These weights are not uniform, so the weighted branch is what runs, which is
    both what a real reconstruction does and what the recorded walls were
    measured on.
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
    normal temperature is the machine working as designed.  A twenty-minute arm
    on a card that heat-soaks would read slow for a reason that has nothing to
    do with the device count, so the flag has to be visible beside the wall."""
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


# ── the model, built the floors refresh's way ─────────────────────────────────
def build_model(family, cell, cpu_devices=None):
    """The model construction the recorded walls were measured from.

    This is ``_build_model`` in ``dev_scripts/refresh_widening_floors.py``,
    copied rather than imported: this script lives outside the library tree and
    must keep working when that tool moves.  Any drift between the two would
    make this run's walls incomparable with the recorded ones, which is the one
    way this measurement can be wrong without anything looking wrong.

    ``cpu_devices`` is for the smoke only.  On CUDA nothing is configured here:
    the count comes from MBIRTORCH_NUM_DEVICES, which keeps the model on the
    automatic branch, and an explicit configure_devices call would take the
    explicit branch instead.  The pin is a CUDA mechanism -- the policy
    short-circuits when fewer than two CUDA devices are visible -- so the smoke
    places its virtual CPU devices by hand, and every row records which
    mechanism actually pinned it.
    """
    import numpy as np

    import mbirtorch

    num_views, _rows, num_channels = cell
    if family == "multiaxis":
        # Two angles per view: azimuth around the object, elevation out of the
        # plane.  The elevation range is part of what the cell measures rather
        # than a free choice: the automatic geometry divides the detector
        # height by the smallest |cos(elevation)|, so a wider range would
        # inflate the slice count and change the problem.
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            tuple(cell), np.stack([azimuth, elevation], axis=1))
    elif family == "translation":
        # A translation scan moves the object across a fixed source and
        # detector on a grid, so its views are grid positions rather than
        # angles and the cell alone does not say what the grid is.
        spec = TRANSLATION_SPECS.get(tuple(cell))
        if spec is None:
            raise ValueError(
                f"this run has no translation grid recorded for cell "
                f"{tuple(cell)}.  Add it to TRANSLATION_SPECS before measuring "
                f"that cell.")
        num_x, num_z, x_spacing, z_spacing = spec
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != num_views:
            raise ValueError(
                f"translation cell {tuple(cell)} has {num_views} views, but its "
                f"{num_x}x{num_z} grid gives {vectors.shape[0]} translations.")
        # Both distances are half the smaller detector extent, as recorded.
        source_dist = min(cell[1], cell[2]) / 2
        model = mbirtorch.TranslationModel(
            tuple(cell), vectors, source_detector_dist=source_dist,
            source_iso_dist=source_dist)
    else:
        # Falling through to another geometry would time that geometry and
        # record the result under this family's name.
        raise ValueError(f"this run has no model construction for family "
                         f"{family!r}")
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def model_class_for(family):
    """The class whose method resolution order the seams are looked up in.

    Named per family rather than taken from a built model, because the wrappers
    go on BEFORE the model is built: the settle happens inside the first
    reconstruction call, and a wrapper installed after the model exists would
    still catch it, but a wrapper installed after the model is built cannot
    catch anything the constructor does.
    """
    import mbirtorch

    if family == "multiaxis":
        return mbirtorch.MultiAxisParallelModel
    if family == "translation":
        return mbirtorch.TranslationModel
    raise ValueError(f"this run has no model class for family {family!r}")


def pin_devices_for(n_dev):
    """The explicit device list an arm needs, or None.

    None on CUDA, where MBIRTORCH_NUM_DEVICES does the pinning.  A list of
    virtual CPU devices on the smoke, where the environment pin cannot: the
    device policy short-circuits below two visible CUDA devices, so the pin has
    nothing to act through there.
    """
    return None if DEVICE == "cuda" else ["cpu"] * n_dev


# ── the instrument: paired host and device timing around named regions ────────
#: The torch module, bound once in the worker so the wrappers do not repeat an
#: import lookup on every call.  None until a worker binds it.
_TORCH = None

#: The CUDA device indices this arm records events on: the devices the arm was
#: pinned to.  The automatic policy always takes cuda:0 .. cuda:(n-1), and every
#: row records the realized device list so a layout that did not take is visible
#: rather than assumed.  Empty off CUDA, where there are no events at all.
_EVENT_DEVICES = []

#: How deep the current thread of control is inside an iteration.  Set by the
#: iteration wrapper and read by every sample, because both projection funnels
#: also fire OUTSIDE an iteration -- in the direct reconstruction, the initial
#: error state and the Hessian -- and a table that mixed the two could not be
#: subtracted from the iteration's own span.  A plain counter is enough: the
#: reconstruction drives its phases from one host thread, and the worker threads
#: only run while that thread is inside the region that started them.
_ITERATION_DEPTH = [0]


def set_event_devices(n_dev):
    """Fix which devices events are recorded on, once, before wrapping."""
    global _EVENT_DEVICES, _TORCH

    import torch

    _TORCH = torch
    if DEVICE == "cuda" and torch.cuda.is_available():
        _EVENT_DEVICES = list(range(min(int(n_dev), torch.cuda.device_count())))
    else:
        _EVENT_DEVICES = []
    return list(_EVENT_DEVICES)


def _cuda_index(device):
    """The CUDA device index of ``device``, or None when it is not a CUDA
    device.  An unindexed 'cuda' means the calling thread's current device,
    which is what torch itself resolves it to."""
    if _TORCH is None:
        return None
    try:
        dev = _TORCH.device(device)
    except Exception:                                             # noqa: BLE001
        return None
    if dev.type != "cuda":
        return None
    if dev.index is not None:
        return int(dev.index)
    try:
        return int(_TORCH.cuda.current_device())
    except Exception:                                             # noqa: BLE001
        return None


def _record_events(indices):
    """One timing event per device in ``indices``, recorded on that device's
    DEFAULT stream, right now.

    The default stream is where every library kernel this run times is issued,
    so a pair of these brackets exactly the work enqueued in between.  Recording
    an event does not synchronize anything: it puts a marker in the stream and
    returns.  Off CUDA there are no events and this returns nothing, which every
    caller handles.

    One consequence worth knowing.  The settle is the FIRST region, and its
    entry markers go on every device this arm was pinned to, so on a two-device
    arm a CUDA context exists on the second device from the settle onward rather
    than from the first tensor placed there.  That device is used by the arm
    either way, so what moves is when its context is created, not whether.  A
    one-device arm records on device 0 only and never touches another card.
    """
    if not indices or _TORCH is None:
        return []
    marks = []
    for index in indices:
        event = _TORCH.cuda.Event(enable_timing=True)
        event.record(_TORCH.cuda.default_stream(_TORCH.device("cuda", index)))
        marks.append((index, event))
    return marks


class RegionSamples:
    """Every region's per-call samples, folded into aggregates once the events
    can be read.

    Two things make this more than a list.  The per-device workers run on a
    thread pool, so samples arrive from several threads at once and the
    bookkeeping is done under a lock.  And a timing event cannot be READ until
    the device has reached it, so the events are held unresolved through the
    whole reconstruction call and converted to milliseconds afterwards, on the
    synchronize the timing protocol already does.  Resolving earlier would mean
    synchronizing inside the run, which would change the overlap being measured.

    The aggregate is keyed by (region, inside-an-iteration, reconstruction call
    index) and, within that, by device index.  A short prefix of raw samples per
    region is kept as well, in arrival order, because a call trail is what tells
    a reader how a region is reached.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._seq = 0
        self._pending = []
        self._agg = {}
        self._raw = []
        self._raw_per_region = {}
        self._labels = set()
        #: Which reconstruction call the samples arriving now belong to:
        #: 0 is the cold pass, 1 upward are the warm repeats.
        self.call_index = 0

    def __len__(self):
        return self._seq

    def append(self, region, in_iteration, t0, t1, starts, ends):
        with self._lock:
            seq = self._seq
            self._seq += 1
            self._labels.add(region)
            self._pending.append(dict(region=region, seq=seq,
                                      call=self.call_index,
                                      in_iteration=bool(in_iteration),
                                      host_ms=(t1 - t0) * 1000.0,
                                      starts=starts, ends=ends))

    def labels(self):
        with self._lock:
            return sorted(self._labels)

    def resolve(self):
        """Turn every held event pair into milliseconds and fold the samples
        into the aggregate.  Called after a reconstruction has returned and its
        devices have been synchronized, so every event has been reached."""
        with self._lock:
            pending, self._pending = self._pending, []
        for sample in pending:
            per_device = []
            for (index, start), (_index2, end) in zip(sample["starts"],
                                                      sample["ends"]):
                try:
                    per_device.append([index, float(start.elapsed_time(end))])
                except Exception:                                 # noqa: BLE001
                    # An event that cannot be read is recorded as unreadable
                    # rather than as zero: zero would silently subtract from the
                    # region it belongs to.
                    per_device.append([index, None])
            self._fold(sample, per_device)

    def _fold(self, sample, per_device):
        key = (sample["region"], sample["in_iteration"], sample["call"])
        entry = self._agg.get(key)
        if entry is None:
            entry = self._agg[key] = dict(
                region=sample["region"], in_iteration=sample["in_iteration"],
                call=sample["call"], level=region_level(sample["region"]),
                calls=0, host_ms=0.0, host_ms_max=0.0, per_device={})
        entry["calls"] += 1
        entry["host_ms"] += sample["host_ms"]
        entry["host_ms_max"] = max(entry["host_ms_max"], sample["host_ms"])
        for index, ms in per_device:
            slot = entry["per_device"].get(index)
            if slot is None:
                slot = entry["per_device"][index] = dict(
                    device_index=index, calls=0, total_ms=0.0, max_ms=0.0,
                    unreadable=0)
            slot["calls"] += 1
            if ms is None:
                slot["unreadable"] += 1
                continue
            slot["total_ms"] += ms
            slot["max_ms"] = max(slot["max_ms"], ms)
        kept = self._raw_per_region.get(sample["region"], 0)
        if kept < RAW_SAMPLE_KEEP:
            self._raw_per_region[sample["region"]] = kept + 1
            self._raw.append(dict(region=sample["region"], seq=sample["seq"],
                                  call=sample["call"],
                                  in_iteration=sample["in_iteration"],
                                  host_ms=sample["host_ms"],
                                  device_ms=per_device))

    def aggregates(self):
        """The folded samples as a flat, json-safe list, in region order then
        call order."""
        rows = []
        for entry in self._agg.values():
            rows.append(dict(
                region=entry["region"], in_iteration=entry["in_iteration"],
                call=entry["call"], level=entry["level"], calls=entry["calls"],
                host_ms=entry["host_ms"], host_ms_max=entry["host_ms_max"],
                per_device=[entry["per_device"][index]
                            for index in sorted(entry["per_device"])]))
        rows.sort(key=lambda r: (r["region"], not r["in_iteration"], r["call"]))
        return rows

    def raw_samples(self):
        return list(self._raw)


class MoveTally:
    """The host-only count and time for ``move_shard``.

    It fires many times per subset for scalar-sized moves -- one per device for
    each combined line-search partial -- so a per-call sample would be mostly
    bookkeeping.  A count and a summed host time per reconstruction call is the
    reading that matters: whether the scalar traffic between components is
    visible at all.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._by_call = {}
        self.call_index = 0

    def add(self, seconds):
        with self._lock:
            slot = self._by_call.get(self.call_index)
            if slot is None:
                slot = self._by_call[self.call_index] = dict(
                    call=self.call_index, calls=0, host_s=0.0)
            slot["calls"] += 1
            slot["host_s"] += seconds

    def rows(self):
        return [self._by_call[call] for call in sorted(self._by_call)]


def region_level(region):
    """Which reporting level a region belongs to.  An unrecognized fan-out
    label lands in ``detail``, which is the safe default: detail rows are
    printed and never subtracted from anything."""
    if region in REGION_LEVEL:
        return REGION_LEVEL[region]
    if region.startswith(FANOUT_PREFIX):
        name = region[len(FANOUT_PREFIX):]
        return "component" if name in COMPONENT_FANOUTS else "detail"
    return "detail"


def fan_out_label(worker_fn):
    """The component name a fan-out's worker function stands for.

    A nested function's qualified name ends with its own name after the last
    ``<locals>``, which is exactly the component: prior_worker, apply_worker,
    sino_worker and so on.  An anonymous worker ends in ``<lambda>``, which
    names nothing, so it is labeled by the function it is written inside plus
    the word lambda -- the back driver's two anonymous workers both become
    ``_sparse_back_project_sharded lambda``, which is what they are.
    """
    qualified = getattr(worker_fn, "__qualname__", None)
    if not qualified:
        qualified = getattr(worker_fn, "__name__", None) or "worker"
    parts = qualified.split(".")
    if parts[-1] != "<lambda>":
        return parts[-1]
    enclosing = parts[-3] if len(parts) >= 3 else parts[0]
    return enclosing + " lambda"


# ── the wrappers ──────────────────────────────────────────────────────────────
def _devices_all(_args, _kwargs):
    """Every device this arm was pinned to.  Used by the phase seams and the two
    projection funnels, whose work lands on all of them."""
    return list(_EVENT_DEVICES)


def _devices_owner(args, kwargs):
    """The one device a band reduce runs on: the band's slice-owner, which is
    ``sum_band_to_owner``'s second argument."""
    owner = kwargs.get("owner")
    if owner is None and len(args) >= 2:
        owner = args[1]
    index = _cuda_index(owner)
    return [index] if index in _EVENT_DEVICES else []


def _devices_first_tensor(args, kwargs):
    """The one device a projector body call runs on: the device of its first
    tensor argument.  These are patched on the class, so ``args[0]`` is the
    Projectors instance and ``args[1]`` is the voxel block or the local
    sinogram."""
    tensor = args[1] if len(args) >= 2 else None
    if tensor is None:
        for value in kwargs.values():
            if hasattr(value, "device"):
                tensor = value
                break
    index = _cuda_index(getattr(tensor, "device", None))
    return [index] if index in _EVENT_DEVICES else []


def _devices_of_list(args, kwargs):
    """The devices a fan-out was handed, keeping only the CUDA ones this arm
    records on."""
    devices = kwargs.get("devices")
    if devices is None and args:
        devices = args[0]
    out = []
    for device in list(devices or []):
        index = _cuda_index(device)
        if index in _EVENT_DEVICES and index not in out:
            out.append(index)
    return out


def make_wrapper(original, region, devices_of, samples, marks_iteration=False):
    """Time one library function on both clocks and call through.

    The wrapper does four things and nothing else: read the host clock, record
    one entry event per relevant device, call the original, and record the exit
    events and the host clock again.  It does not synchronize, does not
    allocate a tensor, and does not touch the result.  The sample is recorded in
    a finally block, so a region that raised is still recorded rather than
    silently missing.

    The entry events are recorded BEFORE the host clock starts and the exit
    events AFTER it stops, so the host reading is the library's own enqueue cost
    with the instrument's cost outside it.
    """
    def wrapped(*args, **kwargs):
        indices = devices_of(args, kwargs)
        in_iteration = _ITERATION_DEPTH[0] > 0
        starts = _record_events(indices)
        t0 = time.perf_counter()
        if marks_iteration:
            _ITERATION_DEPTH[0] += 1
        try:
            return original(*args, **kwargs)
        finally:
            if marks_iteration:
                _ITERATION_DEPTH[0] -= 1
            t1 = time.perf_counter()
            samples.append(region, in_iteration, t0, t1, starts,
                           _record_events(indices))
    return wrapped


def make_fanout_wrapper(original, samples):
    """The one seam that splits an iteration into its components.

    Every per-device fan-out in the library goes through ``run_per_device``, and
    the worker function it is handed names the component.  Labeling by the
    worker's own name needs no access to any closure and survives a component
    moving between functions, which a hard-coded list of call sites would not.
    """
    def wrapped(*args, **kwargs):
        worker_fn = kwargs.get("worker_fn")
        if worker_fn is None and len(args) >= 2:
            worker_fn = args[1]
        region = FANOUT_PREFIX + fan_out_label(worker_fn)
        indices = _devices_of_list(args, kwargs)
        in_iteration = _ITERATION_DEPTH[0] > 0
        starts = _record_events(indices)
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            samples.append(region, in_iteration, t0, t1, starts,
                           _record_events(indices))
    return wrapped


def make_move_wrapper(original, tally):
    """``move_shard``, counted and timed on the host only.  No events: it fires
    many times per subset for scalar-sized moves, and a marker pair per call
    would cost more than the call it measured."""
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            tally.add(time.perf_counter() - t0)
    return wrapped


def _mro_holder(cls, attr):
    """The class in ``cls``'s method resolution order whose OWN ``__dict__``
    holds ``attr``.

    Patching the base class is not enough: both model families override
    ``direct_recon``, so a wrapper on TomographyModel.direct_recon would attach
    to a method that never runs and the region would record nothing while
    looking attached.
    """
    for klass in cls.__mro__:
        if attr in klass.__dict__:
            return klass
    return None


def install_wrappers(family, samples, tally):
    """Wrap every seam, before the model is built, or fail naming the seams.

    Every seam is resolved FIRST and the missing ones are reported together, so
    a tree that moved two functions says so once instead of once per run.  A
    silently unwrapped region would leave a table that looks complete and
    attributes the loss to the wrong component, which is the failure this
    instrument exists to prevent, so an unresolvable seam stops the arm.

    Returns the region names that were attached, in wrapping order.
    """
    import mbirtorch                                                # noqa: F401
    from mbirtorch import _sharding
    from mbirtorch.projectors import Projectors

    model_cls = model_class_for(family)
    planned, missing = [], []

    for attr, region in MODEL_SEAMS:
        holder = _mro_holder(model_cls, attr)
        if holder is None:
            missing.append(f"{region}: no class in {model_cls.__name__}'s "
                           f"method resolution order defines {attr}")
        else:
            planned.append((holder, attr, region, _devices_all,
                            region == "iteration"))

    for attr, region in PROJECTOR_SEAMS:
        if not hasattr(Projectors, attr):
            missing.append(f"{region}: mbirtorch.projectors.Projectors has no "
                           f"{attr}")
        else:
            planned.append((Projectors, attr, region, _devices_first_tensor,
                            False))

    for attr, region in SHARDING_SEAMS:
        if not hasattr(_sharding, attr):
            missing.append(f"{region}: mbirtorch._sharding has no {attr}")
        else:
            devices_of = (_devices_owner if attr == "sum_band_to_owner"
                          else _devices_all)
            planned.append((_sharding, attr, region, devices_of, False))

    for attr in (FANOUT_SEAM, MOVE_SEAM):
        if not hasattr(_sharding, attr):
            missing.append(f"the {attr} seam: mbirtorch._sharding has no "
                           f"{attr}")

    if missing:
        raise RuntimeError(
            "these instrumented seams could not be resolved in the library "
            "under test, so this arm would split the reconstruction over an "
            "incomplete set of components: " + "; ".join(missing))

    attached = []
    for holder, attr, region, devices_of, marks in planned:
        setattr(holder, attr, make_wrapper(getattr(holder, attr), region,
                                           devices_of, samples,
                                           marks_iteration=marks))
        attached.append(region)
    setattr(_sharding, FANOUT_SEAM,
            make_fanout_wrapper(getattr(_sharding, FANOUT_SEAM), samples))
    attached.append(FANOUT_PREFIX + "<by worker name>")
    setattr(_sharding, MOVE_SEAM,
            make_move_wrapper(getattr(_sharding, MOVE_SEAM), tally))
    attached.append("move_shard tally")
    return attached


# ── compile visibility ────────────────────────────────────────────────────────
def dynamo_snapshot():
    """What torch's compiler has done so far, guarded whole.

    Two readings, either of which may be absent on a given torch: the dynamo
    statistics counters, and how many functions have compile times recorded.  An
    absent api records None rather than failing the arm.  The point is narrow: a
    WARM reconstruction that still compiles is a finding, and without this the
    only evidence would be a warm repeat that ran long for no visible reason.
    """
    stats, entries = None, None
    try:
        from torch._dynamo.utils import counters
        stats = {str(k): int(v) for k, v in dict(counters.get("stats", {})).items()}
    except Exception:                                             # noqa: BLE001
        stats = None
    try:
        from torch._dynamo.utils import compile_times
        times = compile_times(repr="csv", aggregate=True)
        entries = len(times[0]) if isinstance(times, tuple) else None
    except Exception:                                             # noqa: BLE001
        entries = None
    return dict(stats=stats, compiled_functions=entries)


def dynamo_delta(before, after):
    """What changed between two snapshots, as a small dict."""
    delta = {}
    if before.get("stats") is not None and after.get("stats") is not None:
        for key, value in after["stats"].items():
            moved = value - before["stats"].get(key, 0)
            if moved:
                delta[key] = moved
    compiled = None
    if (before.get("compiled_functions") is not None
            and after.get("compiled_functions") is not None):
        compiled = after["compiled_functions"] - before["compiled_functions"]
    return dict(stats_delta=delta or None,
                compiled_functions_delta=compiled,
                compiled_functions_after=after.get("compiled_functions"))


# ── the worker: one staging job or one arm, in its own process ────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the job is.

    The calibration check here is the opposite of the memory work's.  This is a
    TIMING probe, and the library's calibration mode does extra work at the
    settle and at the end of the reconstruction, so an arm that inherited it
    would time something other than a reconstruction.  Every job requires it
    OFF, and the row records what it actually saw.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    calibration = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION")
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  seed=SEED,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "MBIRTORCH_NUM_DEVICES is set as on CUDA, and "
                                 "the count is realized by "
                                 "configure_devices(devices=['cpu', ...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_widening_guard=os.environ.get("MBIRTORCH_WIDENING_GUARD"),
                  env_calibration=calibration)
    result["invalid_reasons"] = []
    result["calibration_off_ok"] = calibration in (None, "", "0")
    if not result["calibration_off_ok"]:
        result["invalid_reasons"].append(
            f"MBIRTORCH_MEMORY_CALIBRATION is {calibration!r}; this is a timing "
            f"probe and the calibration mode does extra work at the settle and "
            f"at the end of every reconstruction, so an arm that ran with it "
            f"did not time what the other arms timed")

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


def run_stage(cfg):
    """Make sure ONE (family, cell) sinogram is on disk, and hash it.

    Normally the file is already there from the floors refresh and this only
    hashes it, which is the point: these arms then reconstruct the same bytes
    the recorded walls were measured on.  When the file is absent it is built
    here by that tool's own recipe -- phantom, forward projection, float32 --
    and kept.  The staging process is pinned to one device and takes its
    projection on a freshly built model, so nothing here is a multi-device run.
    """
    import numpy as np

    import mbirtorch

    result, _cuda = _base_result(cfg)
    family, cell = cfg["family"], tuple(cfg["cell"])
    path = _sino_path(family, cell)
    result["sino_path"] = path
    result["sino_dir"] = SINO_DIR

    if os.path.exists(path):
        # Already on disk.  Hash it rather than rebuild it: the forward kernel's
        # atomics make a regenerated sinogram non-identical at the e-7 class, so
        # a rebuild would silently change what every arm of this cell
        # reconstructs.
        array = np.load(path, mmap_mode="r")
        result.update(reused=True, sino_md5=_md5(path),
                      sinogram_shape=list(array.shape))
        return result

    _check_sino_dir_writable()
    model = build_model(family, cell, cpu_devices=pin_devices_for(1))
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    # The shepp-logan builder places its ellipsoids as fractions of the volume,
    # and on a volume only a few voxels deep every one of them can miss, leaving
    # the phantom all zeros -- which forward projects to an all-zero sinogram,
    # so every arm of that cell would time a reconstruction of nothing.  The
    # smoke's translation cell is that shallow.  A seeded uniform volume has the
    # same shape and a comparable dynamic range, and the row records that it was
    # used.  This is the floors refresh's own fallback.
    fallback = None
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
        fallback = "seeded uniform (shepp-logan returned all zeros)"
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    np.save(path, sinogram)
    result.update(reused=False, phantom_fallback=fallback,
                  sino_md5=_md5(path), sinogram_shape=list(sinogram.shape),
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)),
                  stage_devices=[str(d)
                                 for d in model.sino_placement.devices])
    return result


def run_arm(cfg):
    """One arm: a cold reconstruction, then the warm repeats, with every region
    timed on both clocks.

    ORDER, and all of it is load-bearing.  The wrappers go on BEFORE the model
    is built, because the settle happens inside the first reconstruction call.
    The events of one reconstruction are resolved immediately after that
    reconstruction returns and before the next one starts, because an event can
    only be read once the device has reached it and the timing protocol's own
    synchronize is the first moment that is true.  The wall is stopped BEFORE
    the events are resolved, so the instrument's own bookkeeping is outside
    every number this run compares.

    A control arm (``wrapped`` false) skips the wrapping entirely and runs the
    same protocol, so its walls say what the instrument costs.
    """
    import numpy as np
    import torch

    result, cuda = _base_result(cfg)
    family, cell = cfg["family"], tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])
    wrapped = bool(cfg["wrapped"])

    # ── the instrument goes on first ─────────────────────────────────────────
    devices_sampled = set_event_devices(n_dev)
    samples = RegionSamples()
    tally = MoveTally()
    result["event_device_indices"] = list(devices_sampled)
    if wrapped:
        result["regions_wrapped"] = install_wrappers(family, samples, tally)
        result["seams_ok"] = True
    else:
        result["regions_wrapped"] = []
        result["seams_ok"] = None

    model = build_model(family, cell, cpu_devices=pin_devices_for(n_dev))
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    path = _sino_path(family, cell)
    result["sino_path"] = path
    result["sino_dir"] = SINO_DIR
    if not os.path.exists(path):
        result["invalid_reasons"].append(f"no staged sinogram at {path}")
        return result
    result["sino_md5"] = _md5(path)
    staged = np.load(path)
    result["sinogram_shape"] = list(staged.shape)
    weights = _weights(staged)

    # ── the timed protocol, the floors refresh's ─────────────────────────────
    def one():
        np.random.seed(SEED)
        out, _info = model.recon(staged, weights=weights,
                                 max_iterations=ITERATIONS,
                                 stop_threshold_change_pct=0.0)
        if DEVICE == "cuda":
            # Both placements name the same device list; the recon one is named
            # because it is the one the output is gathered from.
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(out)

    dynamo_rows = []
    before = dynamo_snapshot()
    samples.call_index = tally.call_index = 0
    start = time.perf_counter()
    out = one()
    cold = time.perf_counter() - start
    samples.resolve()
    after = dynamo_snapshot()
    dynamo_rows.append(dict(call=0, **dynamo_delta(before, after)))

    warm = []
    for repeat in range(WARM_REPEATS):
        samples.call_index = tally.call_index = repeat + 1
        before = after
        start = time.perf_counter()
        out = one()
        warm.append(time.perf_counter() - start)
        samples.resolve()
        after = dynamo_snapshot()
        dynamo_rows.append(dict(call=repeat + 1, **dynamo_delta(before, after)))

    median = statistics.median(warm)
    result.update(cold_s=cold, warm_all=warm, warm_s=median,
                  spread=(max(warm) - min(warm)) / median if median else None,
                  recon_checksum=float(np.sum(np.abs(out), dtype=np.float64)),
                  dynamo=dynamo_rows)

    # ── the realized layout ──────────────────────────────────────────────────
    realized = [str(d) for d in model.recon_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["sino_devices"] = [str(d) for d in model.sino_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev)
    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            f"pinned to {n_dev} device(s) and realized {len(realized)}: "
            f"{realized}")
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))
    # The axis lengths are passed explicitly, because the TRIVIAL single-device
    # placement is built without one and shard_ranges() raises on it.
    sinogram_shape = [int(s) for s in model.get_params("sinogram_shape")]
    result["view_blocks"] = [end - start for _d, (start, end)
                             in model.sino_placement.shard_ranges(
                                 sinogram_shape[0])]
    result["slice_blocks"] = [end - start for _d, (start, end)
                              in model.recon_placement.shard_ranges(
                                  int(result["recon_shape"][2]))]

    # ── the regions ──────────────────────────────────────────────────────────
    result["regions"] = samples.aggregates()
    result["region_raw"] = samples.raw_samples()
    result["region_labels"] = samples.labels()
    result["total_samples"] = len(samples)
    result["move_shard"] = tally.rows()

    if wrapped:
        fired = set(result["region_labels"])
        required = list(REGIONS_ALWAYS)
        if len(realized) > 1:
            required += list(REGIONS_MULTI_DEVICE)
        silent = [region for region in required if region not in fired]
        result["regions_required"] = required
        result["regions_ok"] = not silent
        if silent:
            result["invalid_reasons"].append(
                f"these regions attached and never ran on a {len(realized)}-"
                f"device arm, so nothing was attributed to them: "
                f"{', '.join(silent)}")
    else:
        result["regions_required"] = []
        result["regions_ok"] = None
    return result


def run_job(cfg):
    """One staging job or one arm, in its own process, with a health sample on
    either side of it.

    A new process per job is not tidiness.  Compiled and hand-written kernel
    bodies are cached at module level for the life of a process, the allocator
    keeps its pools, and the wrappers themselves are installed on library
    classes, so a second arm in the same interpreter would inherit the first
    arm's compiles and stack a second set of wrappers on top of the first.
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

    Every variable is popped and then set (or deliberately not set), so a value
    exported by the shell cannot reach a job that asked for something else.

    MBIRTORCH_MEMORY_CALIBRATION is popped and NEVER set: this is a timing
    probe, the mode does extra work, and an arm that ran with it would not have
    timed what the other arms timed.  MBIRTORCH_WIDENING_GUARD is popped because
    the count pin bypasses the guard anyway, which is what the floors refresh
    does.  The count is set on the smoke too, exactly as on CUDA, so the
    subprocess protocol under test is the same one the real run uses; the
    smoke's count is then realized by an explicit CPU device list, because the
    pin acts only through the device policy and that policy short-circuits below
    two visible CUDA devices.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_WIDENING_GUARD", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # the shipped configuration
    env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def spawn(cfg):
    """Run one configuration in a NEW interpreter.

    The row goes through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.  On an hour-long job that is
    the difference between watching progress and waiting in the dark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_mg44_cfg_{cfg['job_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_mg44_out_{cfg['job_id']}.json")
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
    cell's own arms."""
    keep = _strict_subset("MG44_ARMS", all_arm_ids())
    arms = [cfg for cfg in all_arms() if cfg["arm"] in keep]
    plan, staged = [], set()
    for cfg in arms:
        key = (cfg["family"], tuple(cfg["cell"]))
        if key not in staged:
            staged.add(key)
            plan.append(dict(kind="stage", family=cfg["family"],
                             cell=list(cfg["cell"]), n_dev=1,
                             job_id="stage_{}_{}x{}x{}".format(
                                 cfg["family"], *cfg["cell"])))
        plan.append(cfg)
    if not plan:
        raise ValueError("MG44_ARMS selects no arm")
    return plan


def print_plan(plan):
    arms = [c for c in plan if c["kind"] == "arm"]
    stages = [c for c in plan if c["kind"] == "stage"]
    print(f"mg44 the component split of a reconstruction: {len(arms)} arm(s) "
          f"and {len(stages)} staged sinogram(s), device {DEVICE}")
    print(f"  jsonl -> {RESULTS_DIR}")
    print(f"  staged sinograms read from (and written to, if missing) "
          f"-> {SINO_DIR}")
    print(f"  protocol: seed {SEED} reset before every call, "
          f"{ITERATIONS} iteration(s), one cold pass then {WARM_REPEATS} warm "
          f"repeat(s), the timed call including the per-device synchronize and "
          f"the gather -- the floors refresh's protocol, so these walls compare "
          f"with its recorded ones")
    print("  every wrapped region is timed on BOTH clocks: the host clock "
          "around the call, and a pair of CUDA events on each relevant "
          "device's default stream.  Nothing is synchronized during a "
          "reconstruction; the events are read afterwards")
    print("  the regions, in wrapping order: "
          + ", ".join([region for _a, region in MODEL_SEAMS]
                      + [region for _a, region in PROJECTOR_SEAMS]
                      + [region for _a, region in SHARDING_SEAMS]
                      + ["fan-out <worker name>", "move_shard (tally only)"]))
    print("  the recorded walls this run is compared against, from "
          + RECORDED_SOURCE + ":")
    for (family, cell), rec in sorted(RECORDED_WALLS.items()):
        print(f'    {family:<12}{str(cell):>20}  n1 {rec["n1_warm"]:>6.1f} s   '
              f'n2 {rec["n2_warm"]:>6.1f} s   {rec["ratio"]:.2f}x')
    if SMOKE:
        print(f"  SMOKE: tiny cells {SMOKE_MA_CELL} and {SMOKE_TR_CELL} on "
              f"virtual CPU devices, {ITERATIONS} iteration(s), "
              f"{WARM_REPEATS} warm repeat(s).  There are no CUDA counters "
              f"there, so every device column reads zero and only the host "
              f"columns carry numbers; what the smoke proves is that every "
              f"seam resolves and every region fires, and it ASSERTS both")
    # Wider than ARM_COL: a staging job's id carries its cell, which is the
    # longest name printed anywhere in this run.
    job_col = max([len(cfg["job_id"]) for cfg in plan] + [len("job")]) + 2
    header = (f'  {"job":<{job_col}}{"pin":>5}{"wrapped":>9}{"cell":>20}'
              f'{"family":>13}  what it does')
    print(header)
    for cfg in plan:
        if cfg["kind"] == "stage":
            what = "hashes this cell's sinogram, building it if absent"
            wrapped = "-"
        else:
            what = ("one cold pass and the warm repeats, every region timed"
                    if cfg["wrapped"] else
                    "the same protocol with NO wrappers: the overhead control")
            wrapped = "yes" if cfg["wrapped"] else "no"
        print(f'  {cfg["job_id"]:<{job_col}}{cfg["n_dev"]:>5}{wrapped:>9}'
              f'{str(tuple(cfg["cell"])):>20}{cfg["family"]:>13}  {what}')
    print("  no library file is edited and no default is flipped: the only "
          "runtime change is the wrappers, each of which reads two clocks, "
          "calls through, and returns its original's result unchanged")


def main():
    plan = build_plan()
    if DRY:
        print_plan(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg44_component_{RUN_LABEL}_{stamp}.jsonl")
    print_plan(plan)
    print(f"\nrunning -> {out_path}", flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        header = dict(row="run_header", script="mg44_component_split.py",
                      node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
                      python=sys.executable, results_dir=RESULTS_DIR,
                      sino_dir=SINO_DIR, seed=SEED, iterations=ITERATIONS,
                      warm_repeats=WARM_REPEATS,
                      recorded_source=RECORDED_SOURCE,
                      recorded_walls=[
                          dict(family=family, cell=list(cell), **rec)
                          for (family, cell), rec in RECORDED_WALLS.items()],
                      regions_always=list(REGIONS_ALWAYS),
                      regions_multi_device=list(REGIONS_MULTI_DEVICE),
                      raw_sample_keep=RAW_SAMPLE_KEEP,
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
                print(f'    cold {row.get("cold_s", 0):.1f}s  '
                      f'warm {row.get("warm_s", 0):.1f}s  '
                      f'spread {(row.get("spread") or 0):.1%}  '
                      f'{row.get("realized_n_devices", "-")} device(s)  '
                      f'{row.get("total_samples", 0)} sample(s)', flush=True)
        summary = summarize(rows, plan, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        sink.write(json.dumps(dict(row="summary", **summary)) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    print(f"elapsed {summary['elapsed_min']:.1f} min")
    return 0 if summary["healthy"] else 2


# ── reading the rows ──────────────────────────────────────────────────────────
def _fmt(value, width=10, prec=2):
    """One table cell, with a missing value padded to the width of a present
    one, so the columns line up whether an arm produced a number or not."""
    if value is None:
        return f'{"-":>{width}}'
    return f"{value:>{width}.{prec}f}"


def per_recon_regions(row):
    """One arm's regions, reduced to PER WARM RECONSTRUCTION numbers.

    Returns ``{(region, in_iteration): entry}`` where each entry carries the
    level, the call count, the host milliseconds, and the device milliseconds
    per device -- each divided by the number of warm reconstructions, so an arm
    with three warm repeats and an arm with one are directly comparable.

    The cold pass is left out on purpose: it pays this process's compiles, and
    the walls this run explains are warm walls.
    """
    warm_calls = len(row.get("warm_all") or []) or 1
    out = {}
    for entry in row.get("regions") or []:
        if entry["call"] == 0:
            continue
        key = (entry["region"], bool(entry["in_iteration"]))
        slot = out.get(key)
        if slot is None:
            slot = out[key] = dict(region=entry["region"],
                                   in_iteration=bool(entry["in_iteration"]),
                                   level=entry.get("level")
                                   or region_level(entry["region"]),
                                   calls=0.0, host_ms=0.0, device_ms={})
        slot["calls"] += entry["calls"] / warm_calls
        slot["host_ms"] += entry["host_ms"] / warm_calls
        for device in entry.get("per_device") or []:
            index = device["device_index"]
            slot["device_ms"][index] = (slot["device_ms"].get(index, 0.0)
                                        + device["total_ms"] / warm_calls)
    for slot in out.values():
        values = list(slot["device_ms"].values())
        # The WALL is set by the slowest device, so the maximum over devices is
        # the number that competes with it; the sum is kept beside it because a
        # region whose work is spread evenly and a region that runs on one
        # device look the same in the maximum alone.
        slot["device_ms_max"] = max(values) if values else None
        slot["device_ms_sum"] = sum(values) if values else None
    return out


def cell_arms(by_arm, family, cell, wrapped=True):
    """The (n1 row, n2 row) pair for one cell, either of which may be None.

    ``cell`` is the DECLARED cell, not the one the arm ran at.  The smoke
    collapses all three multiaxis cells onto one tiny stand-in, so matching on
    the cell that ran would make three different arms answer to the same key and
    the last one declared would silently stand in for the other two.
    """
    found = {}
    for arm, family_name, arm_cell, n_dev, is_wrapped in ARM_SPECS:
        if (family_name != family or tuple(arm_cell) != tuple(cell)
                or is_wrapped != wrapped):
            continue
        row = by_arm.get(arm)
        if row is not None:
            found[n_dev] = row
    return found.get(1), found.get(2)


def timed_cells():
    """The (family, declared cell) pairs the wrapped arms cover, in run
    order."""
    seen = []
    for _arm, family, cell, _n, wrapped in ARM_SPECS:
        key = (family, cell)
        if wrapped and key not in seen:
            seen.append(key)
    return seen


# ── the report ────────────────────────────────────────────────────────────────
def print_wall_table(by_arm):
    """Every arm's walls, beside the recorded ones, and each cell's n2-over-n1
    ratio beside the recorded anomaly ratio."""
    print("\n===== the walls =====")
    print("Cold and warm are this run's; recorded is the floors refresh's warm "
          "median from " + RECORDED_SOURCE + ".  The ratio column is warm(n1) "
          "over warm(n2), so above 1.00 means two devices are faster.  Every "
          "number here is a FINDING; none of them touches the exit code.")
    header = (f'{"arm":<{ARM_COL}}{"pin":>4}{"dev":>4}{"cold s":>9}'
              f'{"warm s":>9}{"spread":>9}{"recorded":>10}{"this/rec":>10}')
    print(header)
    print("-" * len(header))
    for arm, family, cell, n_dev, wrapped in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None:
            print(f'{arm:<{ARM_COL}}{n_dev:>4}  no row')
            continue
        # A control arm is compared with the same recorded wall as its wrapped
        # twin: that is the whole point of it, and only the wrapped arm's drift
        # gets a WARNING line, because a control that reads slow says something
        # about the node rather than about the instrument.
        rec = RECORDED_WALLS.get((family, cell)) or {}
        recorded = rec.get(f"n{n_dev}_warm")
        warm = row.get("warm_s")
        drift = (warm / recorded) if (warm and recorded) else None
        print(f'{arm:<{ARM_COL}}{n_dev:>4}'
              f'{row.get("realized_n_devices", "-"):>4}'
              f'{_fmt(row.get("cold_s"), 9, 2)}{_fmt(warm, 9, 2)}'
              f'{_fmt((row.get("spread") or 0) * 100, 8, 1)}%'
              f'{_fmt(recorded, 10, 1)}{_fmt(drift, 10, 2)}')
        # The smoke reconstructs a tiny stand-in cell, so it has nothing to
        # compare with a production wall and the drift line is left off there.
        if (drift is not None and not SMOKE and wrapped
                and abs(drift - 1.0) > WALL_DRIFT_WARN):
            print(f'    WARNING {arm}: this run\'s warm wall is '
                  f'{(drift - 1.0) * 100:+.0f} percent from the recorded '
                  f'{recorded:.1f} s.  The library path is the one those walls '
                  f'were measured on, so a drift this large points at the '
                  f'ruler or at the node, not at the code.')

    print("\nper cell, this run's n1-over-n2 warm ratio beside the recorded "
          "one")
    header = (f'{"cell":<32}{"n1 warm":>10}{"n2 warm":>10}{"ratio":>9}'
              f'{"recorded":>10}')
    print(header)
    print("-" * len(header))
    for family, cell in timed_cells():
        one, two = cell_arms(by_arm, family, cell)
        rec = RECORDED_WALLS.get((family, cell)) or {}
        ratio = None
        if one and two and two.get("warm_s"):
            ratio = one["warm_s"] / two["warm_s"]
        label = f"{family} {tuple(cell)}"
        print(f'{label:<32}{_fmt(one.get("warm_s") if one else None, 10, 2)}'
              f'{_fmt(two.get("warm_s") if two else None, 10, 2)}'
              f'{_fmt(ratio, 9, 2)}{_fmt(rec.get("ratio"), 10, 2)}')
    if SMOKE:
        print("  SMOKE: these arms ran tiny stand-in cells, so the recorded "
              "columns above name a problem none of them measured.")


def print_overhead(by_arm):
    """What the instrument itself costs: the 512 cell wrapped against the same
    cell with no wrappers, at both counts."""
    print("\n===== instrument overhead =====")
    print("The control arms run the identical protocol with NO wrappers "
          "installed.  A ratio within a few percent of 1.00 means the wrapped "
          "numbers describe the library; a larger one is a finding about this "
          "probe.")
    header = (f'{"count":>7}{"wrapped s":>12}{"control s":>12}'
              f'{"wrapped/control":>18}')
    print(header)
    print("-" * len(header))
    for n_dev in (1, 2):
        wrapped = by_arm.get(f"ma512_n{n_dev}")
        control = by_arm.get(f"ma512_n{n_dev}_control")
        a = wrapped.get("warm_s") if wrapped else None
        b = control.get("warm_s") if control else None
        ratio = (a / b) if (a and b) else None
        print(f'{n_dev:>7}{_fmt(a, 12, 2)}{_fmt(b, 12, 2)}'
              f'{_fmt(ratio, 18, 3)}')


def _level_rows(one, two, level, in_iteration):
    """The rows of one level's table: every region either arm recorded at that
    level and that nesting, with the two arms' numbers side by side and sorted
    by how much MORE device time the two-device arm spent in it."""
    left = per_recon_regions(one) if one else {}
    right = per_recon_regions(two) if two else {}
    keys = set()
    for table in (left, right):
        for key, entry in table.items():
            if entry["level"] == level and entry["in_iteration"] == in_iteration:
                keys.add(key)
    rows = []
    for key in keys:
        a, b = left.get(key), right.get(key)
        a_ms = a["device_ms_max"] if a else None
        b_ms = b["device_ms_max"] if b else None
        excess = (b_ms - a_ms) if (a_ms is not None and b_ms is not None) else None
        rows.append(dict(region=key[0], n1=a, n2=b, n1_ms=a_ms, n2_ms=b_ms,
                         excess=excess))
    rows.sort(key=lambda r: (r["excess"] is None,
                             -(r["excess"] if r["excess"] is not None else 0.0),
                             r["region"]))
    return rows


def _print_level_table(title, rows, one, two):
    """One level's table.  The device column is the per-reconstruction device
    milliseconds on the BUSIEST device, because that is the one competing with
    the wall; the share is that number over the arm's own warm wall."""
    print(f"\n  {title}")
    if not rows:
        print("    no region fired at this level")
        return
    wall1 = (one.get("warm_s") or 0) * 1000.0 if one else 0.0
    wall2 = (two.get("warm_s") or 0) * 1000.0 if two else 0.0
    header = (f'    {"region":<{REGION_COL}}{"n1 calls":>9}{"n1 ms":>10}'
              f'{"n1 %":>7}{"n2 calls":>9}{"n2 ms":>10}{"n2 %":>7}'
              f'{"n2-n1 ms":>11}{"n1 host":>9}{"n2 host":>9}')
    print(header)
    print("    " + "-" * (len(header) - 4))
    for entry in rows[:TABLE_ROWS]:
        a, b = entry["n1"], entry["n2"]
        share1 = (entry["n1_ms"] / wall1 * 100) if (entry["n1_ms"] and wall1) else None
        share2 = (entry["n2_ms"] / wall2 * 100) if (entry["n2_ms"] and wall2) else None
        print(f'    {entry["region"]:<{REGION_COL}}'
              f'{_fmt(a["calls"] if a else None, 9, 0)}'
              f'{_fmt(entry["n1_ms"], 10, 1)}{_fmt(share1, 7, 1)}'
              f'{_fmt(b["calls"] if b else None, 9, 0)}'
              f'{_fmt(entry["n2_ms"], 10, 1)}{_fmt(share2, 7, 1)}'
              f'{_fmt(entry["excess"], 11, 1)}'
              f'{_fmt(a["host_ms"] if a else None, 9, 1)}'
              f'{_fmt(b["host_ms"] if b else None, 9, 1)}')
    if len(rows) > TABLE_ROWS:
        print(f'    ... {len(rows) - TABLE_ROWS} further region(s) are in the '
              f'jsonl')


def print_residual(row, label):
    """Per device, the iteration's own span minus the spans of the components
    inside it.  What is left is the glue between components: scalar moves, host
    gaps, and anything else that happens between one component finishing and the
    next starting."""
    if row is None:
        return
    table = per_recon_regions(row)
    iteration = table.get(("iteration", False))
    if iteration is None:
        return
    components = [entry for key, entry in table.items()
                  if entry["level"] == "component" and entry["in_iteration"]]
    parts = []
    for index in sorted(iteration["device_ms"]):
        span = iteration["device_ms"][index]
        inside = sum(entry["device_ms"].get(index, 0.0) for entry in components)
        parts.append(f"dev {index}: {span - inside:.1f} ms of {span:.1f} ms")
    if parts:
        print(f'    residual inside the iteration ({label}): '
              + "; ".join(parts))
    else:
        print(f'    residual inside the iteration ({label}): no device timing '
              f'(the smoke has no CUDA events)')


def print_cell_block(by_arm, family, cell):
    """One cell's component split: the deliverable."""
    one, two = cell_arms(by_arm, family, cell)
    print(f"\n===== the component split: {family} {tuple(cell)} =====")
    if one is None or two is None:
        print("  this cell needs both its arms to compare; "
              f'n1 {"ran" if one else "did not run"}, '
              f'n2 {"ran" if two else "did not run"}.')
        if one is None and two is None:
            return
    print("  Device milliseconds are per WARM reconstruction, on the busiest "
          "device of that arm.  The n2-n1 column is what two devices spend "
          "MORE of in that region, so the component carrying the loss is at the "
          "top.  Host milliseconds are per warm reconstruction too, summed over "
          "the calls.")
    _print_level_table("phases (the reconstruction's own stages; the settle "
                       "also fires again, cheaply, inside the direct "
                       "reconstruction and the Hessian)",
                       _level_rows(one, two, "phase", False), one, two)
    _print_level_table("in-iteration components (these tile one iteration)",
                       _level_rows(one, two, "component", True), one, two)
    print_residual(one, "n1")
    print_residual(two, "n2")
    _print_level_table("inner detail, inside the iterations",
                       _level_rows(one, two, "detail", True), one, two)
    _print_level_table("everything else outside the iterations",
                       _level_rows(one, two, "component", False)
                       + _level_rows(one, two, "detail", False), one, two)


def top_excess_region(by_arm, family, cell):
    """The in-iteration component the two-device arm spends most extra device
    time in, or None when the cell is missing an arm."""
    one, two = cell_arms(by_arm, family, cell)
    if one is None or two is None:
        return None
    rows = _level_rows(one, two, "component", True)
    for entry in rows:
        if entry["excess"] is not None:
            return entry
    return None


def print_translation_block(by_arm):
    """Translation's own tables, and the one line the shared-mechanism question
    turns on."""
    print_cell_block(by_arm, "translation", TR_CELL)
    print("\n===== translation against multiaxis =====")
    translation_top = top_excess_region(by_arm, "translation", TR_CELL)
    print("  The question is whether one mechanism explains both families.  "
          "This is the data, not a verdict.")
    if translation_top is None:
        print("  translation: no pair of arms to compare")
    else:
        print(f'  translation {tuple(TR_CELL)}: the largest n2 excess is '
              f'{translation_top["region"]} at '
              f'{translation_top["excess"]:+.1f} ms per reconstruction')
    for family, cell in timed_cells():
        if family != "multiaxis":
            continue
        entry = top_excess_region(by_arm, family, cell)
        if entry is None:
            print(f'  multiaxis {tuple(cell)}: no pair of arms to compare')
        else:
            print(f'  multiaxis {tuple(cell)}: the largest n2 excess is '
                  f'{entry["region"]} at {entry["excess"]:+.1f} ms per '
                  f'reconstruction')


def print_host_bound(by_arm):
    """Regions whose host time is more than half their device time.

    A region like that is not waiting on the device: the cost is in dispatch, in
    a thread pool handoff, or in a compile guard.  That is a different problem
    from a slow kernel and wants a different fix, so it is listed on its own.
    """
    print("\n===== host-bound regions =====")
    print("Listed where the warm host milliseconds exceed HALF the warm device "
          "milliseconds on the busiest device.  A region here is paying for "
          "dispatch, a thread handoff or a compile guard rather than waiting on "
          "the device.")
    found = False
    for arm, _family, _cell, _n, wrapped in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None or not wrapped:
            continue
        for key, entry in sorted(per_recon_regions(row).items()):
            device_ms = entry.get("device_ms_max")
            if not device_ms:
                continue
            if entry["host_ms"] > 0.5 * device_ms:
                found = True
                where = "in an iteration" if key[1] else "outside iterations"
                print(f'  {arm:<{ARM_COL}}{entry["region"]:<{REGION_COL}}'
                      f'host {entry["host_ms"]:.1f} ms vs device '
                      f'{device_ms:.1f} ms  ({where})')
    if not found:
        print("  none (on the CPU smoke there are no device numbers to "
              "compare against, so this list is empty there by construction)")


def print_observations(by_arm):
    """The closing block: one line per timed arm, short enough to paste into a
    note."""
    print("\n===== paste-ready observations =====")
    print("One line per timed arm: its warm wall, how that compares with the "
          "recorded one, and the component its cell's two-device arm spends "
          "most extra device time in.  Every number here is a FINDING and none "
          "of them touches the exit code.")
    for arm, family, cell, n_dev, wrapped in ARM_SPECS:
        if not wrapped:
            continue
        row = by_arm.get(arm)
        if row is None:
            print(f"  {arm:<{ARM_COL}}  no row")
            continue
        rec = (RECORDED_WALLS.get((family, cell)) or {}).get(f"n{n_dev}_warm")
        warm = row.get("warm_s")
        if warm is None:
            print(f"  {arm:<{ARM_COL}}  no wall: "
                  f'{"; ".join(row.get("invalid_reasons") or ["reason not recorded"])}')
            continue
        drift = (f"{warm / rec:.2f}x recorded" if rec
                 else "no recorded wall")
        entry = top_excess_region(by_arm, family, cell)
        top = ("no pair to compare" if entry is None else
               f'top n2 excess {entry["region"]} {entry["excess"]:+.1f} ms')
        print(f"  {arm:<{ARM_COL}}  warm {warm:.2f} s ({drift})  {top}")


def smoke_checks(by_arm):
    """The smoke's assertions, which DO gate the exit code.

    The smoke measures nothing worth reading -- tiny cells on virtual CPU
    devices with no timing events at all.  What it proves is that the instrument
    is wired to the library: every seam resolved, and every region that a
    two-device reconstruction must reach actually fired.  A probe that silently
    stopped attaching to a renamed function would print a complete-looking table
    of nothing, and this is what catches that before a cluster job pays for it.
    """
    problems = []
    required_fanouts = ("prior_worker", "direction_worker", "lin_quad_worker",
                        "apply_worker", "sino_worker")
    required_regions = ("forward projection", "back projection",
                        "halo exchange", "band reduce",
                        "forward body call", "back body call")
    for arm, _family, _cell, n_dev, wrapped in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None:
            continue
        if wrapped and row.get("seams_ok") is not True:
            problems.append(f"{arm}|not every seam resolved")
        if not wrapped or n_dev < 2:
            continue
        fired = set(row.get("region_labels") or [])
        for name in required_fanouts:
            if FANOUT_PREFIX + name not in fired:
                problems.append(f"{arm}|the fan-out {name} never fired")
        for name in required_regions:
            if name not in fired:
                problems.append(f"{arm}|the region {name!r} never fired")
    return problems


def summarize(rows, plan, out_path):
    """The blocks a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  Which
    component carries the loss, how far a wall drifted, whether a region is
    host-bound, whether a card was hot: all FINDINGS, all printed, none of them
    gated.  An arm that produced no row, ran on the wrong device count, failed
    to resolve a seam, ran with the calibration mode on, read different bytes
    from its siblings, or left a region its device count should have exercised
    with no calls did not measure what the plan said it would, and that is an
    instrument failure.
    """
    print(f"\n===== mg44 the component split ({out_path}) =====")
    broken, findings = [], []
    by_arm, stages = {}, []

    for row in rows:
        job_id = row.get("job_id", "?")
        if row.get("error"):
            print(f'{job_id:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:120]}')
            broken.append(f"{job_id}|error")
            continue
        if row.get("kind") == "stage":
            stages.append(row)
            broken.extend(f"{job_id}|{reason}"
                          for reason in row.get("invalid_reasons") or [])
            continue
        by_arm[row["arm"]] = row
        for reason in row.get("invalid_reasons") or []:
            broken.append(f'{row["arm"]}|{reason}')
        if row.get("gpu_hot"):
            findings.append(f'{row["arm"]}: GPU hot during this arm, so its '
                            f'wall may be a thermal reading rather than a '
                            f'device-count one')
        if row.get("gpu_throttle"):
            findings.append(f'{row["arm"]}: throttle reasons '
                            f'{row["gpu_throttle"]}')
        for entry in row.get("dynamo") or []:
            if entry["call"] and entry.get("compiled_functions_delta"):
                findings.append(
                    f'{row["arm"]}: warm call {entry["call"]} compiled '
                    f'{entry["compiled_functions_delta"]} further function(s), '
                    f'so that repeat paid compile time the others did not')

    # Every PLANNED arm produced a row.  Read off the plan rather than off the
    # rows: an arm whose subprocess died before writing anything leaves no row
    # to notice its absence in.
    reported = {item.split("|", 1)[0] for item in broken}
    for cfg in plan:
        name = cfg.get("arm")
        if name and name not in by_arm and name not in reported:
            broken.append(f"{name}|no row")

    # Every arm of a cell must have read the SAME bytes.  There are no md5
    # sidecars in the staged directory, so each arm hashed what it loaded and
    # this is where the digests are compared: an arm that reconstructed a
    # different array than its siblings is not comparable with them, and nothing
    # in its own row would say so.
    digests = {}
    for row in list(by_arm.values()) + stages:
        key = (row.get("family"), tuple(row.get("cell") or ()))
        digest = row.get("sino_md5")
        if digest:
            digests.setdefault(key, {}).setdefault(digest, []).append(
                row.get("job_id"))
    for key, seen in sorted(digests.items()):
        if len(seen) > 1:
            broken.append(
                "{}|the jobs of this cell read different sinograms: {}".format(
                    key[0],
                    "; ".join(f"{digest[:12]} <- {', '.join(jobs)}"
                              for digest, jobs in sorted(seen.items()))))

    for row in stages:
        print(f'staged {row["family"]} {tuple(row["cell"])}: '
              f'md5 {str(row.get("sino_md5", "-"))[:12]}'
              f'{"  (reused from disk)" if row.get("reused") else "  (built by this run)"}')

    print_wall_table(by_arm)
    print_overhead(by_arm)
    for family, cell in timed_cells():
        if family == "multiaxis":
            print_cell_block(by_arm, family, cell)
    print_translation_block(by_arm)
    print_host_bound(by_arm)
    print_observations(by_arm)

    if SMOKE:
        problems = smoke_checks(by_arm)
        print("\n-- smoke assertions --")
        if problems:
            for item in problems:
                print(f"  FAIL {item}")
        else:
            print("  every seam resolved, and at every two-device arm the "
                  "fan-outs prior_worker, direction_worker, lin_quad_worker, "
                  "apply_worker and sino_worker fired, along with both "
                  "projection funnels, the halo exchange, the band reduce and "
                  "both body-call regions")
        broken.extend(problems)

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print(f"  BROKEN {item}")
    else:
        print("  every planned arm ran, realized its pinned count, resolved "
              "every seam, ran with the calibration mode off, read the same "
              "sinogram as its siblings, and exercised every region its device "
              "count can exercise")
    for item in findings:
        print(f"  finding (not gated) {item}")
    if not findings:
        print("  no thermal, throttle or warm-compile findings")

    return dict(healthy=not broken, broken=broken, findings=findings,
                arms={name: dict(
                    family=row.get("family"), cell=row.get("cell"),
                    wrapped=row.get("wrapped"),
                    realized_n_devices=row.get("realized_n_devices"),
                    cold_s=row.get("cold_s"), warm_s=row.get("warm_s"),
                    spread=row.get("spread"),
                    recon_checksum=row.get("recon_checksum"),
                    total_samples=row.get("total_samples"),
                    region_labels=row.get("region_labels"))
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
