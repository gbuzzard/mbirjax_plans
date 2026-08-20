"""mg49 -- DOES A DENOISE EVER GET FASTER ON MORE DEVICES, AND WHERE DOES THE
SHARDED SWEEP'S TIME GO?

WHY THIS RUN EXISTS.  The QGGMRFDenoiser is held to one device by two sentinel
rows in the shipped widening-floors table.  A sentinel row records a device
count with NO admission size: splitting lost at every size ever probed, so the
automatic path never widens a denoise for speed and only the memory preflight
-- capacity, not speed -- ever widens one.  This run asks two questions.

QUESTION A, THE LADDER.  The measured loss SHRINKS as the problem grows, so an
admission size may simply sit above the largest cell anyone has probed.  These
are the warm walls the sentinels rest on, measured 2026-08-20 by the full
floors refresh (mg48, job 15399595) and read from that job's log.  The ratio is
the one the floors tool prints -- the one-device warm median over the wider
count's warm median, so above 1.00 means the wider count is faster -- and both
denoiser floor rows are taken against ONE device, because no smaller denoiser
count is admitted:

    (512, 448, 384)      n1 0.233 s   n2 0.394 s   n4 0.506 s    0.592x  0.461x
    (768, 672, 576)      n1 0.664 s   n2 1.018 s   n4 1.253 s    0.652x  0.530x
    (1024, 1008, 992)    n1 2.191 s   n2 3.269 s   n4 3.574 s    0.670x  0.613x

This run extends that ladder to the 1280-, 1536- and 1664-class and re-anchors
at the 1024-class, looking for the size where two or four devices first win.

WHY THE LADDER STOPS AT THE 1664-CLASS, AND WHY THAT MAKES THE ANSWER FINAL.
The speed sentinel only governs where one device can still HOLD the problem.
Above that size the memory preflight widens a denoise whatever the sentinel
says, because no single device can take it.  The denoiser's own memory ledger,
priced at one device (a closed form; it queries no device and allocates
nothing), puts the boundary just above the 1664-class:

    1024-class  1.02e9 voxels  13.7 GB
    1280-class  2.02e9 voxels  26.9 GB
    1536-class  3.51e9 voxels  46.7 GB
    1664-class  4.48e9 voxels  59.5 GB  -> 68.4 GB with the preflight's margin,
                                           which an 80 GB H100 still holds
    1792-class  5.60e9 voxels  74.4 GB  -> 85.6 GB with the margin, which it
                                           does NOT hold

So the 1664-class is the last size at which the speed question has any
practical effect.  With it in the ladder, a run that finds no admission
anywhere has closed the question across the sentinel's whole domain rather
than leaving "maybe it wins at some larger size" open; without it, a run can
only push the open edge upward.  The figures above are QUOTED here for the
reader; the report prints the same table read from the tree under test at run
time (``_build_memory_ledger(workload='denoise')``), so nothing in the output
depends on this docstring being current.

QUESTION B, THE ATTRIBUTION.  Where does the sharded sweep's time actually go?
A structural fact from the code makes this sharper than a plain split: the
single-device and the multi-device denoisers are DIFFERENT IMPLEMENTATIONS.

    At one device (``QGGMRFDenoiser.denoise``, the else branch) the whole
    per-subset update is ONE compiled call to ``vcd_subset_denoiser``.

    Above one device (``QGGMRFDenoiser._denoise_sharded``) that same update
    becomes two per-device fan-outs, four 0-d scalar reductions combined on
    the lead device, a step-size broadcast back to every device, and a halo
    exchange once per pass.

So the comparison is not "the same work, split".  It is one fused compiled
kernel per subset against a distributed sequence per subset.  The deliverable
is how the two-device sweep's time divides among the prior-and-direction
worker, the update-apply worker, the halo exchange, the scalar moves, and the
residual gaps between them -- on a host clock and on a device clock.

THE INSTRUMENT, AND WHY IT IS IMPORTED RATHER THAN COPIED.  The paired-clock
probe this run needs already exists: mg44 built it, ran it on the cluster, and
its bookkeeping around CUDA events is subtle enough that a second copy would
be a second thing to get wrong.  mg49 therefore IMPORTS the instrument from
``mg44_component_split`` (a sibling module in this directory) and supplies only
what is specific to a denoiser.  Reused verbatim: the event recording, the
per-region sample store, the move tally, the three wrapper factories, the
fan-out label rule, the MRO seam resolver, the dynamo snapshot, the GPU health
sample, the md5 and the table formatter.

Importing a module that configures itself from the environment has three
couplings, and all three are made explicit in the code below because a silent
mismatch in any of them would produce a complete-looking table of zeros:

    1.  mg44 reads its own mode from ``MG44_*`` variables AT IMPORT TIME, and
        its module-level ``DEVICE`` decides whether CUDA events are recorded at
        all.  mg49 therefore sets ``MG44_SMOKE`` to match its own smoke mode
        BEFORE importing mg44, and asserts ``mg44.DEVICE`` equals this run's
        device immediately after.  Without that assert, a mode disagreement
        would leave every device column reading zero with nothing to say so.
    2.  mg44's sample store folds each sample's reporting LEVEL through mg44's
        module-level ``REGION_LEVEL`` and ``COMPONENT_FANOUTS``, which name
        reconstruction regions.  mg49 rebinds both to its own denoiser taxonomy
        right after the import and before any sampling.
    3.  Anything mg44 does not export, mg49 defines here.

THE SEAMS.  Class-method seams are resolved by walking QGGMRFDenoiser's method
resolution order and patching the class whose own ``__dict__`` holds the
attribute, so a method the denoiser overrides is wrapped where it is defined.
Every seam is resolved BEFORE the model is built, and a seam that cannot be
resolved FAILS the arm naming it: a silently unwrapped region would leave a
table that looks complete and attribute the time to the wrong place.

    denoise call          QGGMRFDenoiser.denoise                  (total)
    settle                TomographyModel._apply_device_policy    (setup)
    place image           TomographyModel._shard_recon            (setup)
    noise estimate        QGGMRFDenoiser.estimate_image_noise_std (setup)
    auto regularization   TomographyModel.auto_set_regularization_params
                                                                  (setup)
    sweep (sharded)       QGGMRFDenoiser._denoise_sharded         (sweep)
    halo exchange         mbirtorch._sharding.exchange_qggmrf_halos
                                                                  (component)
    per-pass ell1         mbirtorch.denoising.image_ell1          (component)
    fan-out <worker>      mbirtorch._sharding.run_per_device      (component)
    move_shard            mbirtorch._sharding.move_shard   (host tally only)

The sweep seam passes ``marks_iteration=True``, which flags every sample taken
inside it.  That flag is what separates the sweep's components from the setup
regions, and it is the only thing that does: setup and sweep components are
otherwise ordinary calls at the same depth inside ``denoise``.

TWO SEAMS ARE DELIBERATELY NOT TAKEN, and the report must not be read as if
they were.  ``denoising.vcd_subset_denoiser`` and
``qggmrf.qggmrf_gradient_and_hessian_at_indices`` are both handed to
``maybe_compile``, so a wrapper on either would be COMPILED: dynamo would trace
the timing calls -- ``time.perf_counter``, CUDA event construction and record
-- and either graph-break or fail outright, and in doing so would change the
very thing being measured.  This is why the ONE-DEVICE arm has no internal
seams at all: its whole per-subset update IS that one compiled call.  The
one-device sweep is therefore reported as a RESIDUAL (the total minus the
wrapped setup regions), labeled as a residual everywhere it appears, and never
as a measurement.

ONE MORE THING THE ARM SHAPE MAKES TRUE.  The fan-outs, the halo exchange and
the sweep region itself exist only inside ``_denoise_sharded``, which a
one-device denoise never enters -- ``_shard_recon`` returns a plain tensor on a
single-device placement, not a shard set.  A one-device arm therefore records
those regions as attached with zero calls, which is CORRECT and must not fail
the arm.  ``per-pass ell1`` is the exception: it fires on both paths (once per
pass at one device, once per shard per pass above one), which makes it the one
component the two implementations share.

A THIRD SEAM RESOLVES AND NEVER FIRES, BY DESIGN.  ``noise estimate`` wraps
``estimate_image_noise_std``, which ``denoise`` calls only when the caller
supplies no ``sigma_noise``.  The floors protocol always supplies it, precisely
so that no arm pays a host-side estimate that does no device work and would add
a size-dependent constant to every reading.  The seam is taken anyway, so that
a future protocol change shows up as a region with calls rather than as
silence, and the region is excluded from the must-fire list.

THE ARMS, each in a fresh subprocess, cheap first.

    Part B, the attribution, at the 1024-class (1024, 1008, 992):
        s1024_n1, s1024_n2, s1024_n4          wrapped
        s1024_n1_control, s1024_n2_control    NOT wrapped
    Part A, the ladder, at four cells, each at one, two and four devices, with
    NOTHING wrapped -- a plain wall measurement, so it is the floors protocol
    and nothing else:
        d1024_n1/n2/n4    (1024, 1008, 992)   re-anchors the recorded ratios
        d1280_n1/n2/n4    (1280, 1264, 1248)
        d1536_n1/n2/n4    (1536, 1520, 1504)
        d1664_n1/n2/n4    (1664, 1648, 1632)  the capacity boundary, last

The two control arms are the instrument-overhead check: identical protocol,
identical placement, no wrappers.  Their warm medians beside the wrapped arms'
say what the probe itself costs.

THE PROTOCOL IS THE FLOORS TOOL'S, DELIBERATELY.  Model construction, the
placement mechanism, the seed, the input recipe, the call and the timing
envelope are all the ones in ``dev_scripts/refresh_widening_floors.py``,
because the sentinels this run tests were measured by that tool and a different
protocol would measure a different thing.  In particular: the model is
``QGGMRFDenoiser(tuple(cell))`` where the cell IS the image shape (the denoiser
sets its sinogram shape equal to its image shape, which is why its floors are
read in image voxels); the placement is EXPLICIT, through
``configure_devices(devices=['cuda:0', ...])``, not through the
``MBIRTORCH_NUM_DEVICES`` pin, and every row records the realized device list
so a count that did not take is visible; the input is a Shepp-Logan phantom
plus seeded gaussian noise at sigma 0.1; numpy's seed is reset to 13
immediately before EVERY call; the call is ``denoise(staged, sigma_noise=0.1,
max_iterations=3, stop_threshold_change_pct=0.0)``; and the timed envelope
includes the synchronize on every placement device and the gather to numpy.
One cold pass, then three warm repeats, with the warm median and spread
recorded.

TWO THINGS EVERY ARM RECORDS BEYOND THE USUAL.  ``recon_checksum``, the sum of
absolute values of the last warm output, and whether that output is finite.
The 1536-class cell holds 3.51e9 voxels and the 1664-class 4.48e9, both across
int32's 2**31 boundary -- element counts and flat indices at that scale are a
known trap class in this project.  A non-finite or wildly-off output at either
cell is a FINDING ABOUT THE LIBRARY, printed loudly, not an instrument failure.
Every arm also records its dynamo deltas, so a warm call that still compiles is
visible rather than showing up as an unexplained slow repeat.

AN ARM WHOSE WORKER DIES is recorded as a row and the run continues.  Running
out of device memory at a large cell on one device is the plausible case, and
at the 1664-class one device holds roughly three full 18 GB buffers: n=1 is
expected to fit an 80 GB H100 but is not guaranteed.  If it does not fit, that
reading IS the capacity boundary and the report says so plainly rather than
treating it as a defect.

STAGING.  One input file per cell, reused across that cell's arms.
``MG49_SINO_DIR`` defaults to the mg48 floors results directory, whose name for
a denoiser cell is ``_sino_denoiser_<v>x<r>x<c>.npy`` -- so the 1024-class file
is already there and reusing it makes this run's 1024 row directly comparable
with the recorded one.  Each arm hashes the file it loaded and records the md5;
the driver asserts at summary time that every arm of a cell read the same
digest.  A missing file is built by a separate one-device staging job that runs
before that cell's arms.

THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY.  It is 0 when every planned arm
produced a row, realized its configured device count, read the same input as
its siblings, resolved every seam, ran without the calibration mode, and
exercised every region its device count can exercise.  Ratios, verdicts,
finiteness, thermals: all FINDINGS.  They are printed in full and none of them
touches the exit code.  A person reads the tables.

OUTPUT.  One jsonl under MG49_RESULTS, named
mg49_denoiser_<node>_<stamp>.jsonl: a run-header row, one row per staging job
and per arm, and a summary row, flushed as they finish so a job cut short still
yields everything it completed.  The run then prints the ladder table, the
memory-ledger line beneath it, the attribution block at the 1024-class, the
one-device residual note, the instrument-overhead check and a paste-ready line
per arm.

Run:
    <torch python> mg49_denoiser_split.py        on a 4-GPU node
    MG49_DRY=1 <python> mg49_denoiser_split.py   print the plan and stop

Configuration is by environment variable only; there is no command line apart
from mg44's ``--worker`` entry, which this script reuses for its subprocesses.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  List values are parsed strictly: an unrecognized token
is an error, not a silent skip.
    MG49_RESULTS=<dir>    where the jsonl goes, and where a newly staged input
                          goes when MG49_SINO_DIR is left unset (the smoke's
                          case)
    MG49_SINO_DIR=<dir>   where the staged inputs are read from, and written to
                          if one is missing.  Defaults to the mg48 floors
                          results directory, so the 1024-class arms denoise the
                          same bytes the recorded walls were measured on
    MG49_SMOKE=1          the local smoke: one tiny cell on virtual CPU
                          devices, one iteration, one warm repeat, no arm above
                          two devices.  There are no CUDA events there, so
                          every device column reads zero and only the host
                          columns carry numbers; what the smoke proves is that
                          every seam resolves and every region fires, and it
                          ASSERTS both
    MG49_DRY=1            print the arm plan and exit, importing no torch
    MG49_ARMS=a,b         a subset, by arm id, e.g. s1024_n1,s1024_n2
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG49_SMOKE", "0") == "1"
DRY = os.environ.get("MG49_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

# ── COUPLING 1: mg44 configures itself AT IMPORT TIME ─────────────────────────
# mg44 reads MG44_SMOKE when it is imported and sets its module-level DEVICE
# from it, and that DEVICE is what decides whether CUDA events are recorded at
# all.  So the variable has to be set BEFORE the import, not after it, and the
# assert below is not decoration: if mg49 ran on CUDA while mg44 believed it
# was on CPU, every wrapper would still fire, every host column would still
# fill, and every device column would silently read zero.
os.environ["MG44_SMOKE"] = "1" if SMOKE else "0"
# mg44 is a sibling file in this directory.  sys.path already carries the
# script's own directory when a script is run by path, but PYTHONPATH is
# deliberately unset by this run's sbatch, so the directory is added explicitly
# rather than assumed.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    import mg44_component_split as mg44
except ImportError as _exc:                                       # noqa: BLE001
    raise ImportError(
        "mg49 imports its paired-clock instrument from mg44_component_split, "
        "which must sit beside this script.  Looked in {}.  Original error: "
        "{}".format(_HERE, _exc))
assert mg44.DEVICE == DEVICE, (
    "mg44 was imported in {!r} mode but this run is {!r}.  mg44 reads "
    "MG44_SMOKE at import time and its module-level DEVICE decides whether "
    "CUDA events are recorded, so a mismatch would leave every device column "
    "reading zero with nothing to say so.".format(mg44.DEVICE, DEVICE))

#: The protocol, the floors refresh's.  Changing any of these would measure a
#: different amount of work than the recorded walls describe.
SEED = 13
DENOISE_SIGMA = 0.1
ITERATIONS = 1 if SMOKE else 3
WARM_REPEATS = 1 if SMOKE else 3

#: The ladder's cells.  For a denoiser the cell IS the image shape: the class
#: sets its sinogram shape equal to its image shape, which is why its floors
#: are read in image voxels where every other family's are read in sinogram
#: elements.
CELL_1024 = (1024, 1008, 992)
CELL_1280 = (1280, 1264, 1248)
CELL_1536 = (1536, 1520, 1504)
CELL_1664 = (1664, 1648, 1632)

#: One class beyond the ladder, priced but never run.  It is in the capacity
#: probe below so the report can BRACKET where the memory preflight takes the
#: decision away from the speed sentinel, rather than only saying that the
#: largest measured cell still fits.
CELL_1792 = (1792, 1776, 1760)

#: Every cell the report prices from the tree under test, in ascending order.
CAPACITY_PROBE_CELLS = (CELL_1024, CELL_1280, CELL_1536, CELL_1664, CELL_1792)

#: The smoke's stand-in.  Every cell collapses onto it, so a laptop exercises
#: every arm's code path in seconds.  The arm ids do not change, so MG49_ARMS
#: selects the same names either way.  Its slice count (the last axis) must
#: divide into at least one slice per device at two devices, and its pixel
#: count (the first two axes) must fill the denoiser's 16 fixed subsets.
SMOKE_CELL = (8, 24, 20)

# ── the walls this run is measured against ────────────────────────────────────
#: Warm walls measured 2026-08-20 by the full floors refresh (mg48, job
#: 15399595), read from that job's log.  They are QUOTED here so the report can
#: print this run's re-anchored 1024-class ratios beside them; nothing gates on
#: them, and a large disagreement is a WARNING about the ruler rather than a
#: finding about size.
RECORDED_SOURCE = ("mg48, job 15399595 (the full widening-floors refresh on "
                   "the recompile-remedied tree; measured 2026-08-20, read "
                   "from the job log)")
#: The same provenance, short enough to sit at the end of a table row.
RECORDED_LABEL = "mg48, job 15399595"
#: Per cell: the warm median in seconds at one, two and four devices, and the
#: ratios against ONE device, which is what both denoiser floor rows use.
RECORDED_WALLS = {
    (512, 448, 384): dict(n1=0.233, n2=0.394, n4=0.506,
                          ratio_n2=0.592, ratio_n4=0.461),
    (768, 672, 576): dict(n1=0.664, n2=1.018, n4=1.253,
                          ratio_n2=0.652, ratio_n4=0.530),
    CELL_1024: dict(n1=2.191, n2=3.269, n4=3.574,
                    ratio_n2=0.670, ratio_n4=0.613),
}
#: How far this run's 1024-class ratio may sit from the recorded one before the
#: report prints a WARNING line.  The denoiser path in this tree is the one
#: those walls were measured on, so a large drift points at the ruler or at the
#: node rather than at the code.  A WARNING is a finding: it does not gate.
ANCHOR_DRIFT_WARN = 0.15

# ── COUPLING 2: mg44's level taxonomy names RECONSTRUCTION regions ────────────
#: Which reporting level each region belongs to.  The levels are chosen so that
#: regions inside one level do not nest inside each other, which is what makes
#: a table over a level meaningful.  ``total`` is the whole denoise call;
#: ``setup`` is everything the call does before the sweep starts; ``sweep`` is
#: the sharded sweep itself; ``component`` is what tiles that sweep.
REGION_LEVEL = {
    "denoise call": "total",
    "settle": "setup",
    "place image": "setup",
    "noise estimate": "setup",
    "auto regularization": "setup",
    "sweep (sharded)": "sweep",
    "halo exchange": "component",
    "per-pass ell1": "component",
}
#: The fan-outs that are direct children of one subset update.  Together with
#: the halo exchange and the per-pass ell1 these tile the sharded sweep, so
#: their per-device times can be subtracted from the sweep's to leave a
#: residual.  mg44's sample store folds each sample's level through this tuple
#: and through REGION_LEVEL above, so both are rebound onto mg44 immediately
#: below -- before any sample is taken, since the level is decided at fold
#: time and a sample folded under mg44's own reconstruction taxonomy would land
#: in the wrong table with nothing to say so.
COMPONENT_FANOUTS = ("terms_worker", "apply_worker")
mg44.REGION_LEVEL = REGION_LEVEL
mg44.COMPONENT_FANOUTS = COMPONENT_FANOUTS

#: mg44's prefix on every fan-out label, reused so a fan-out is never confused
#: with a named seam.
FANOUT_PREFIX = mg44.FANOUT_PREFIX

# ── the seams ─────────────────────────────────────────────────────────────────
#: The class-method seams, as (attribute, region, marks_iteration).  Each is
#: resolved by walking QGGMRFDenoiser's method resolution order and patching
#: the first class whose own __dict__ holds the attribute: ``denoise``,
#: ``estimate_image_noise_std`` and ``_denoise_sharded`` are the denoiser's own,
#: the other three are TomographyModel's.  Patching one class unconditionally
#: would wrap a method that never runs and record an empty region.
MODEL_SEAMS = (
    ("denoise", "denoise call", False),
    ("_apply_device_policy", "settle", False),
    ("_shard_recon", "place image", False),
    ("estimate_image_noise_std", "noise estimate", False),
    ("auto_set_regularization_params", "auto regularization", False),
    # marks_iteration: every sample taken inside the sharded sweep is flagged,
    # which is what separates the sweep's components from the setup regions.
    ("_denoise_sharded", "sweep (sharded)", True),
)
#: The module-level seam in ``mbirtorch._sharding``.  Every library caller
#: reaches it through the module object, never through a from-import (verified
#: by grep across the package), so patching the module attribute intercepts all
#: of them.
SHARDING_SEAMS = (("exchange_qggmrf_halos", "halo exchange"),)
#: The fan-out seam and the host-only tally seam, both in ``_sharding``.
FANOUT_SEAM = "run_per_device"
MOVE_SEAM = "move_shard"

#: The ell-1 seam, and it MUST be taken at ``mbirtorch.denoising.image_ell1``
#: rather than at ``mbirtorch._memory_ledger.image_ell1``, where the function
#: is defined.  denoising.py does ``from ._memory_ledger import image_ell1`` at
#: import time, so the name the denoiser's calls resolve through is the
#: denoising module's own binding.  Patching the definition site would attach a
#: wrapper that nothing in the denoise path calls, and the region would report
#: as silently empty while looking attached.  (Patching the module binding also
#: keeps this probe out of the reconstruction's own uses of the same function,
#: which reach it through ``_memory_ledger.image_ell1`` and are not part of a
#: denoise.)
ELL1_SEAM_MODULE = "mbirtorch.denoising"
ELL1_SEAM_ATTR = "image_ell1"
ELL1_REGION = "per-pass ell1"

#: Regions every wrapped arm must exercise at ANY device count.  A region that
#: attached and never ran measured nothing, and a table of regions that never
#: ran is the vacuity the resolve-or-fail rule exists to prevent.
REGIONS_ALWAYS = ("denoise call", "settle", "place image",
                  "auto regularization", ELL1_REGION)
#: Regions that exist only ABOVE one device.  ``_shard_recon`` returns a plain
#: tensor on a single-device placement, so a one-device denoise never enters
#: ``_denoise_sharded`` and never reaches the fan-outs or the halo exchange.  A
#: one-device arm records these attached with zero calls, which is correct and
#: must not fail the arm.
REGIONS_MULTI_DEVICE = ("sweep (sharded)", "halo exchange",
                        FANOUT_PREFIX + "terms_worker",
                        FANOUT_PREFIX + "apply_worker")
#: Wrapped, and expected NEVER to fire under this protocol: ``denoise`` calls
#: the noise estimate only when the caller supplies no sigma_noise, and the
#: floors protocol always supplies it.  Named here so the report can say the
#: silence is by design rather than leave a reader wondering.
REGIONS_EXPECTED_SILENT = ("noise estimate",)

#: How many rows each region table prints.  The rest stay in the jsonl.
TABLE_ROWS = 14

# ── recorded context, not gates ───────────────────────────────────────────────
#: The padding witness, recorded on every row so a reader can tell which tree
#: produced these numbers without leaving the jsonl.  504 is a four-device
#: slice band at the 2048 cell and 512 is what the rounding must turn it into.
#: The denoiser runs no projector, so the padding does not touch what this run
#: measures; it is a tree fingerprint, nothing more.
PAD_PROBE_WIDTH = 504
PAD_PROBE_EXPECTED = 512
#: The recompile-remedy witness, likewise recorded.  The remedy raised torch's
#: per-function recompile budget on the compiling thread; it is what separates
#: this tree from the one the pre-remedy walls were measured on.
RECOMPILE_FLOOR_EXPECTED = 64

RESULTS_DIR = os.environ.get(
    "MG49_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
#: Where the staged inputs live.  The default is the mg48 floors refresh's own
#: results directory: the 1024-class denoiser input is already there, and
#: reusing those bytes is what makes this run's 1024-class row directly
#: comparable with the recorded one.  The smoke has no such directory and
#: stages into its own results directory.
SINO_DIR = os.environ.get(
    "MG49_SINO_DIR",
    RESULTS_DIR if SMOKE
    else "/scratch/gautschi/buzzard/torch_p3/results/mg48_floors")

RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 20                        # wide enough for the longest arm id
REGION_COL = 26                     # wide enough for the longest region name

#: How many slices of the first axis one staging chunk builds at a time.  See
#: ``_staged_input`` for why the noisy phantom is built in chunks and why that
#: is value-identical to the floors tool's whole-array expression.
STAGE_CHUNK_VOXELS = 64 * 1024 * 1024
# ──────────────────────────────────────────────────────────────────────────────


# ── the arms ──────────────────────────────────────────────────────────────────
def _cell_for(cell):
    """The cell an arm actually runs at.  The smoke collapses every cell onto
    one tiny stand-in, so the arm ids stay the same and MG49_ARMS selects the
    same names either way."""
    return SMOKE_CELL if SMOKE else cell


#: Which question an arm belongs to.  The attribution arms and the ladder arms
#: share the 1024-class cell and the ladder arms and the control arms are both
#: unwrapped, so a report that selected arms by cell and wrapping alone would
#: mix the three sets.  Naming the part keeps that intent in the code instead
#: of in the order the arms happen to be declared.
PART_ATTRIBUTION = "attribution"
PART_CONTROL = "control"
PART_LADDER = "ladder"

#: Every arm, in RUN order, as (arm id, cell, device count, wrapped, part).
#: Cheap first: a harness defect surfaces in minutes rather than after an hour
#: of 1664-class work.  Part B (the attribution, wrapped, at the 1024-class)
#: leads, with its two controls right behind so the instrument-overhead reading
#: is in hand before anything large is spent on it.  Part A (the ladder,
#: nothing wrapped) follows in ascending cell size, the 1664-class last because
#: it is both the largest and the one that may not fit one device.
ARM_SPECS = (
    ("s1024_n1", CELL_1024, 1, True, PART_ATTRIBUTION),
    ("s1024_n2", CELL_1024, 2, True, PART_ATTRIBUTION),
    ("s1024_n4", CELL_1024, 4, True, PART_ATTRIBUTION),
    ("s1024_n1_control", CELL_1024, 1, False, PART_CONTROL),
    ("s1024_n2_control", CELL_1024, 2, False, PART_CONTROL),
    ("d1024_n1", CELL_1024, 1, False, PART_LADDER),
    ("d1024_n2", CELL_1024, 2, False, PART_LADDER),
    ("d1024_n4", CELL_1024, 4, False, PART_LADDER),
    ("d1280_n1", CELL_1280, 1, False, PART_LADDER),
    ("d1280_n2", CELL_1280, 2, False, PART_LADDER),
    ("d1280_n4", CELL_1280, 4, False, PART_LADDER),
    ("d1536_n1", CELL_1536, 1, False, PART_LADDER),
    ("d1536_n2", CELL_1536, 2, False, PART_LADDER),
    ("d1536_n4", CELL_1536, 4, False, PART_LADDER),
    ("d1664_n1", CELL_1664, 1, False, PART_LADDER),
    ("d1664_n2", CELL_1664, 2, False, PART_LADDER),
    ("d1664_n4", CELL_1664, 4, False, PART_LADDER),
)

#: The ladder's cells and the counts the report reads a verdict for, in order.
LADDER_CELLS = (CELL_1024, CELL_1280, CELL_1536, CELL_1664)
LADDER_COUNTS = (2, 4)
#: The cell Part B attributes, and the counts it compares there.
ATTRIBUTION_CELL = CELL_1024
ATTRIBUTION_COUNTS = (1, 2, 4)

#: In the smoke every cell collapses onto one tiny stand-in and no arm runs
#: above two devices, so the four-device arms are left out of the plan rather
#: than run at a count the smoke does not exercise.  Left out, not silently
#: failed: an arm that is not planned is not reported missing.
SMOKE_MAX_DEVICES = 2


def all_arms():
    """Every arm's configuration dict, in run order."""
    arms = []
    for arm, cell, n_dev, wrapped, part in ARM_SPECS:
        if SMOKE and n_dev > SMOKE_MAX_DEVICES:
            continue
        arms.append(dict(kind="arm", arm=arm, job_id=arm,
                         family="denoiser", cell=list(_cell_for(cell)),
                         declared_cell=list(cell), n_dev=n_dev,
                         wrapped=wrapped, part=part, iterations=ITERATIONS,
                         warm_repeats=WARM_REPEATS))
    return arms


def all_arm_ids():
    return [cfg["arm"] for cfg in all_arms()]


def arms_for(cell, part):
    """The declared (arm id, device count) pairs of one cell in one part, in
    declared order.

    Matching is on the DECLARED cell, never on the cell an arm ran at: the
    smoke collapses every cell onto one stand-in, so matching on what ran would
    make four different cells answer to the same key.
    """
    return [(arm, n_dev)
            for arm, arm_cell, n_dev, _wrapped, arm_part in ARM_SPECS
            if tuple(arm_cell) == tuple(cell) and arm_part == part]


# ── the staged input ──────────────────────────────────────────────────────────
def _input_path(cell):
    """One file per cell, under the shared input directory.

    The name is the floors refresh's, deliberately: this run reads that tool's
    files, and a different name here would mean rebuilding gigabytes to denoise
    the same thing.
    """
    return os.path.join(SINO_DIR,
                        "_sino_denoiser_{}x{}x{}.npy".format(*cell))


def _check_input_dir_writable():
    """Fail early, naming the path, when an input has to be built and the
    directory cannot take it.  A staging job that discovers this after building
    a seventeen-gigabyte array has already spent the time."""
    try:
        os.makedirs(SINO_DIR, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            "the staged-input directory {} does not exist and cannot be "
            "created ({}).  Set MG49_SINO_DIR to a writable directory, or "
            "stage the inputs there first.".format(SINO_DIR, exc))
    if not os.access(SINO_DIR, os.W_OK):
        raise RuntimeError(
            "the staged-input directory {} is not writable, and an input is "
            "missing from it.  Set MG49_SINO_DIR to a writable directory, or "
            "stage the inputs there first.".format(SINO_DIR))


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


def _staged_input(recon_shape):
    """The floors tool's own input recipe: a Shepp-Logan phantom plus seeded
    gaussian noise at sigma 0.1, float32 and contiguous.

    Returns ``(array, phantom_fallback, chunks)``.

    THE CHUNKING, and why it changes no value.  The floors tool writes this as
    one expression over the whole volume:

        noise = np.random.RandomState(13).randn(*recon_shape)
        staged = np.asarray(phantom + 0.1 * noise, dtype=np.float32)

    ``randn`` returns FLOAT64, so at the 1664-class that one line holds a 36 GB
    noise array, a 36 GB scaled copy, a 36 GB sum and the 18 GB result at once,
    on top of the 18 GB phantom -- roughly 145 GB of host memory for a volume
    whose answer is 18 GB.  This builds the same array a slab of rows at a
    time, into a float32 output allocated once.  The values are identical, bit
    for bit, for two reasons that both have to hold: numpy's legacy gaussian
    generator fills its output sequentially in C order, so drawing the volume
    in consecutive leading-axis slabs from ONE RandomState consumes exactly the
    stream one whole-volume draw would; and every operation here is elementwise
    in float64 with no reduction and no reordering, so slicing the expression
    cannot move a rounding.  A future floors refresh that rebuilds one of these
    files whole will therefore get the same digest.
    """
    import numpy as np

    import mbirtorch

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    # The Shepp-Logan builder places its ellipsoids as fractions of the volume,
    # and on a volume only a few voxels deep every one of them can miss,
    # leaving the phantom all zeros -- which would make every arm of that cell
    # denoise nothing.  A seeded uniform volume has the same shape and a
    # comparable dynamic range, and the row records that it was used.  This is
    # the floors refresh's own fallback.
    fallback = None
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
        fallback = "seeded uniform (shepp-logan returned all zeros)"

    rows = int(recon_shape[0])
    per_row = int(recon_shape[1]) * int(recon_shape[2])
    step = max(1, min(rows, STAGE_CHUNK_VOXELS // max(1, per_row)))
    generator = np.random.RandomState(SEED)
    staged = np.empty(tuple(int(s) for s in recon_shape), dtype=np.float32)
    chunks = 0
    for start in range(0, rows, step):
        stop = min(rows, start + step)
        noise = generator.randn(stop - start, int(recon_shape[1]),
                                int(recon_shape[2]))
        staged[start:stop] = np.asarray(
            phantom[start:stop] + DENOISE_SIGMA * noise, dtype=np.float32)
        chunks += 1
    # np.empty above is already C-contiguous, so this is a no-op that documents
    # the requirement rather than a copy.
    return np.ascontiguousarray(staged), fallback, chunks


# ── the model, built the floors refresh's way ─────────────────────────────────
def build_model(cell, n_dev):
    """The model one arm times, built the way the recorded denoiser rows were.

    This is the denoiser branch of ``_build_model`` in
    ``dev_scripts/refresh_widening_floors.py``, copied rather than imported:
    this script lives outside the library tree and must keep working when that
    tool moves.  Any drift between the two would make this run's walls
    incomparable with the recorded ones, which is the one way this measurement
    can be wrong without anything looking wrong.

    THE PLACEMENT IS EXPLICIT at every device count, on CUDA and on the smoke
    alike.  That is the protocol the recorded denoiser rows were measured
    under, and it is not interchangeable with the ``MBIRTORCH_NUM_DEVICES``
    pin: the pin acts through the automatic device policy, and an explicit
    layout takes a different branch of that policy (it is never second-guessed,
    and it skips the speed floors this run is about).  The pin would also do
    nothing at all on the smoke, where the policy short-circuits below two
    visible CUDA devices.  Every row records the realized device list, so a
    count that did not take is visible rather than assumed.
    """
    import mbirtorch

    model = mbirtorch.QGGMRFDenoiser(tuple(cell))
    if DEVICE == "cuda":
        model.configure_devices(
            devices=["cuda:{}".format(i) for i in range(n_dev)])
    else:
        # Virtual CPU devices: the smoke's way of realizing a device count on a
        # machine with no CUDA at all.
        model.configure_devices(devices=["cpu"] * n_dev)
    model.set_params(no_warning=True, verbose=0)
    return model


def model_class():
    """The class whose method resolution order the seams are looked up in.

    Named rather than taken from a built model, because the wrappers go on
    BEFORE the model is built.
    """
    import mbirtorch

    return mbirtorch.QGGMRFDenoiser


# ── the instrument: mg44's wrappers, wired to this run's seams ────────────────
def _devices_of_tensor_arg(args, kwargs):
    """The one device a call's first tensor argument lives on.

    Used for ``image_ell1``, which takes one tensor and is called once per
    SHARD on the sharded path.  mg44's ``_devices_all`` would bracket every
    placement device around every one of those calls, so the per-device totals
    would count each call's whole span on devices that were only waiting; the
    per-shard resolver gives each device exactly its own shard's reduction.
    """
    tensor = args[0] if args else None
    if tensor is None:
        for value in kwargs.values():
            if hasattr(value, "device"):
                tensor = value
                break
    index = mg44._cuda_index(getattr(tensor, "device", None))
    # _EVENT_DEVICES is rebound by mg44.set_event_devices, so it is read off
    # the module here rather than imported by name.
    return [index] if index in mg44._EVENT_DEVICES else []


def install_wrappers(samples, tally):
    """Wrap every seam, before the model is built, or fail naming the seams.

    Every seam is resolved FIRST and the missing ones are reported together, so
    a tree that moved two functions says so once instead of once per run.  A
    silently unwrapped region would leave a table that looks complete and
    attribute the sweep's time to the wrong place, which is the failure this
    instrument exists to prevent, so an unresolvable seam stops the arm.

    Returns the region names that were attached, in wrapping order.
    """
    import importlib

    from mbirtorch import _sharding

    cls = model_class()
    denoising = importlib.import_module(ELL1_SEAM_MODULE)
    planned, missing = [], []

    for attr, region, marks in MODEL_SEAMS:
        holder = mg44._mro_holder(cls, attr)
        if holder is None:
            missing.append(
                "{}: no class in {}'s method resolution order defines "
                "{}".format(region, cls.__name__, attr))
        else:
            planned.append((holder, attr, region, mg44._devices_all, marks))

    for attr, region in SHARDING_SEAMS:
        if not hasattr(_sharding, attr):
            missing.append("{}: mbirtorch._sharding has no {}".format(region,
                                                                      attr))
        else:
            planned.append((_sharding, attr, region, mg44._devices_all, False))

    # The ell-1 seam, at the CALLER'S binding rather than at the definition
    # site.  See ELL1_SEAM_ATTR above for why that distinction is load-bearing.
    if not hasattr(denoising, ELL1_SEAM_ATTR):
        missing.append(
            "{}: {} has no {}.  It is imported there as `from ._memory_ledger "
            "import image_ell1`, and that binding is what the denoiser's calls "
            "resolve through; patching the definition site instead would "
            "attach a wrapper nothing calls".format(
                ELL1_REGION, ELL1_SEAM_MODULE, ELL1_SEAM_ATTR))
    else:
        planned.append((denoising, ELL1_SEAM_ATTR, ELL1_REGION,
                        _devices_of_tensor_arg, False))

    for attr in (FANOUT_SEAM, MOVE_SEAM):
        if not hasattr(_sharding, attr):
            missing.append(
                "the {} seam: mbirtorch._sharding has no {}".format(attr, attr))

    if missing:
        raise RuntimeError(
            "these instrumented seams could not be resolved in the library "
            "under test, so this arm would split the denoise over an "
            "incomplete set of regions: " + "; ".join(missing))

    attached = []
    for holder, attr, region, devices_of, marks in planned:
        setattr(holder, attr,
                mg44.make_wrapper(getattr(holder, attr), region, devices_of,
                                  samples, marks_iteration=marks))
        attached.append(region)
    setattr(_sharding, FANOUT_SEAM,
            mg44.make_fanout_wrapper(getattr(_sharding, FANOUT_SEAM), samples))
    attached.append(FANOUT_PREFIX + "<by worker name>")
    setattr(_sharding, MOVE_SEAM,
            mg44.make_move_wrapper(getattr(_sharding, MOVE_SEAM), tally))
    attached.append("move_shard tally")
    return attached


# ── the worker: one staging job or one arm, in its own process ────────────────
def _base_result(cfg):
    """The fields every row carries, whatever the job is.

    The calibration check is the same one mg44 makes and for the same reason.
    This is a TIMING probe, and the library's calibration mode does extra work
    at the settle and at the end of the call, so an arm that inherited it would
    time something other than a denoise.  Every job requires it OFF, and the
    row records what it actually saw.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    calibration = os.environ.get("MBIRTORCH_MEMORY_CALIBRATION")
    result = dict(cfg, framework="torch", version="torch " + torch.__version__,
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  seed=SEED, denoise_sigma=DENOISE_SIGMA,
                  placement_mechanism=(
                      "configure_devices(devices=[...]) -- explicit at every "
                      "count, which is the protocol the recorded denoiser rows "
                      "were measured under; MBIRTORCH_NUM_DEVICES is NOT set"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_widening_guard=os.environ.get("MBIRTORCH_WIDENING_GUARD"),
                  env_recompile_limit=os.environ.get(
                      "MBIRTORCH_RECOMPILE_LIMIT"),
                  env_calibration=calibration)
    result["invalid_reasons"] = []
    result["calibration_off_ok"] = calibration in (None, "", "0")
    if not result["calibration_off_ok"]:
        result["invalid_reasons"].append(
            "MBIRTORCH_MEMORY_CALIBRATION is {!r}; this is a timing probe and "
            "the calibration mode does extra work at the settle and at the end "
            "of every call, so an arm that ran with it did not time what the "
            "other arms timed".format(calibration))
    if cuda:
        result["device_total_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory)
    else:
        result["device_total_bytes"] = None

    # The tree witnesses, recorded so a reader can tell which tree produced
    # these numbers from the row alone.  Recorded, not gated: the sbatch
    # asserts both before any arm runs.
    try:
        from mbirtorch._utils import padded_kernel_width
        result["padded_kernel_width_probe"] = int(
            padded_kernel_width(PAD_PROBE_WIDTH))
    except Exception as exc:                                      # noqa: BLE001
        result["padded_kernel_width_probe"] = None
        result["padded_kernel_width_error"] = "{}: {}".format(
            type(exc).__name__, exc)
    result["padding_present"] = (
        result["padded_kernel_width_probe"] == PAD_PROBE_EXPECTED)
    try:
        from mbirtorch import projectors
        result["recompile_limit_floor"] = int(
            projectors._RECOMPILE_LIMIT_FLOOR)
    except Exception as exc:                                      # noqa: BLE001
        result["recompile_limit_floor"] = None
        result["recompile_floor_error"] = "{}: {}".format(type(exc).__name__,
                                                          exc)
    result["recompile_remedy_present"] = (
        (result["recompile_limit_floor"] or 0) >= RECOMPILE_FLOOR_EXPECTED)
    return result, cuda


def _preflight_margin():
    """The margin the memory preflight adds to a modeled peak before it asks
    whether a device can hold it, read from the tree under test rather than
    written down here.  A hand-copied constant is exactly the kind of figure
    that goes stale without anything looking wrong."""
    import inspect

    from mbirtorch import _memory_ledger

    try:
        default = inspect.signature(
            _memory_ledger.layout_fits).parameters["margin"].default
        return float(default), "read from _memory_ledger.layout_fits"
    except Exception as exc:                                      # noqa: BLE001
        return None, "could not be read ({}: {})".format(type(exc).__name__,
                                                         exc)


def one_device_ledger(cell):
    """The denoiser's modeled ONE-DEVICE peak for one cell, in bytes.

    Pure arithmetic: the ledger queries no device and allocates nothing, which
    is what lets the widening policy price a device count the model is not
    configured for.  Returned with the dominant phase's name, because a peak
    without the phase that sets it says nothing about what would have to change
    to lower it.
    """
    import numpy as np

    model = build_model(cell, 1)
    ledger = model._build_memory_ledger(devices=[model.torch_device],
                                        workload="denoise")
    peak = int(ledger.per_device_peaks()[0])
    return dict(cell=list(cell), voxels=int(np.prod(np.asarray(cell,
                                                              dtype=np.int64))),
                one_device_peak_bytes=peak,
                dominant_phase=ledger.dominant_phase(0).name)


def capacity_probe():
    """The modeled one-device peak of every probe cell, with the preflight's
    margin applied, so the report can say where CAPACITY takes the widening
    decision away from the speed sentinel.

    Built in a staging job rather than in every arm: it is the same closed form
    at every device count and costs nothing to run once.
    """
    margin, margin_note = _preflight_margin()
    rows = []
    for cell in CAPACITY_PROBE_CELLS:
        try:
            row = one_device_ledger(cell)
        except Exception as exc:                                  # noqa: BLE001
            rows.append(dict(cell=list(cell),
                             error="{}: {}".format(type(exc).__name__, exc)))
            continue
        if margin is not None:
            row["demand_bytes"] = int((1.0 + margin)
                                      * row["one_device_peak_bytes"])
        rows.append(row)
    return dict(margin=margin, margin_note=margin_note, rows=rows)


def run_stage(cfg):
    """Make sure ONE cell's input array is on disk, and hash it.

    Normally the 1024-class file is already there from the mg48 floors refresh
    and this only hashes it, which is the point: those arms then denoise the
    same bytes the recorded walls were measured on.  When a file is absent it
    is built here by that tool's own recipe, in a job pinned to one device.

    The staging job also carries this run's capacity probe, because it is the
    one job per cell that is guaranteed to run and the probe is pure arithmetic
    that no arm should repeat.
    """
    import numpy as np

    result, _cuda = _base_result(cfg)
    cell = tuple(cfg["cell"])
    path = _input_path(cell)
    result["input_path"] = path
    result["input_dir"] = SINO_DIR
    try:
        result["capacity_probe"] = capacity_probe()
    except Exception as exc:                                      # noqa: BLE001
        result["capacity_probe"] = dict(
            error="{}: {}".format(type(exc).__name__, exc))

    if os.path.exists(path):
        array = np.load(path, mmap_mode="r")
        result.update(reused=True, sino_md5=mg44._md5(path),
                      input_shape=list(array.shape))
        return result

    _check_input_dir_writable()
    model = build_model(cell, 1)
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    staged, fallback, chunks = _staged_input(recon_shape)
    np.save(path, staged)
    result.update(reused=False, phantom_fallback=fallback,
                  stage_chunks=chunks, sino_md5=mg44._md5(path),
                  input_shape=list(staged.shape),
                  input_checksum=float(np.sum(np.abs(staged),
                                              dtype=np.float64)),
                  stage_devices=[str(d)
                                 for d in model.recon_placement.devices])
    return result


def run_arm(cfg):
    """One arm: a cold denoise, then the warm repeats, with every region timed
    on both clocks when the arm is wrapped.

    ORDER, and all of it is load-bearing.  The wrappers go on BEFORE the model
    is built.  The events of one call are resolved immediately after that call
    returns and before the next one starts, because a CUDA event can only be
    read once the device has reached it and the protocol's own synchronize is
    the first moment that is true.  The wall is stopped BEFORE the events are
    resolved, so the instrument's own bookkeeping sits outside every number
    this run compares.

    A control arm (``wrapped`` false) skips the wrapping entirely and runs the
    identical protocol, so its walls say what the instrument costs.
    """
    import numpy as np
    import torch

    result, cuda = _base_result(cfg)
    cell = tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])
    wrapped = bool(cfg["wrapped"])

    # ── the instrument goes on first ─────────────────────────────────────────
    devices_sampled = mg44.set_event_devices(n_dev)
    samples = mg44.RegionSamples()
    tally = mg44.MoveTally()
    result["event_device_indices"] = list(devices_sampled)
    if wrapped:
        result["regions_wrapped"] = install_wrappers(samples, tally)
        result["seams_ok"] = True
    else:
        result["regions_wrapped"] = []
        result["seams_ok"] = None

    model = build_model(cell, n_dev)
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]
    try:
        result["ledger"] = one_device_ledger(cell)
    except Exception as exc:                                      # noqa: BLE001
        result["ledger"] = dict(error="{}: {}".format(type(exc).__name__, exc))

    path = _input_path(cell)
    result["input_path"] = path
    result["input_dir"] = SINO_DIR
    if not os.path.exists(path):
        result["invalid_reasons"].append("no staged input at " + path)
        return result
    result["sino_md5"] = mg44._md5(path)
    staged = np.load(path)
    result["input_shape"] = list(staged.shape)

    # ── the timed protocol, the floors refresh's ─────────────────────────────
    def one():
        np.random.seed(SEED)
        out, _info = model.denoise(staged, sigma_noise=DENOISE_SIGMA,
                                   max_iterations=ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if DEVICE == "cuda":
            # Both placements name the same device list; the recon one is
            # named because it is the one the denoiser divides its image on.
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(out)

    dynamo_rows = []
    before = mg44.dynamo_snapshot()
    samples.call_index = tally.call_index = 0
    start = time.perf_counter()
    out = one()
    cold = time.perf_counter() - start
    samples.resolve()
    after = mg44.dynamo_snapshot()
    dynamo_rows.append(dict(call=0, **mg44.dynamo_delta(before, after)))

    warm = []
    for repeat in range(WARM_REPEATS):
        samples.call_index = tally.call_index = repeat + 1
        before = after
        start = time.perf_counter()
        out = one()
        warm.append(time.perf_counter() - start)
        samples.resolve()
        after = mg44.dynamo_snapshot()
        dynamo_rows.append(dict(call=repeat + 1,
                                **mg44.dynamo_delta(before, after)))

    median = statistics.median(warm)
    # The checksum and the finiteness flag are the int32-boundary watch.  At the
    # 1536- and 1664-class the volume crosses 2**31 elements, and a flat index
    # or an element count that overflowed there would show up here first.
    result.update(cold_s=cold, warm_all=warm, warm_s=median,
                  spread=(max(warm) - min(warm)) / median if median else None,
                  recon_checksum=float(np.sum(np.abs(out), dtype=np.float64)),
                  recon_finite=bool(np.all(np.isfinite(out))),
                  output_shape=list(np.shape(out)),
                  dynamo=dynamo_rows)

    # ── the realized layout ──────────────────────────────────────────────────
    realized = [str(d) for d in model.recon_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            "configured for {} device(s) and realized {}: {}".format(
                n_dev, len(realized), realized))
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))
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
                "these regions attached and never ran on a {}-device arm, so "
                "nothing was attributed to them: {}".format(len(realized),
                                                            ", ".join(silent)))
    else:
        result["regions_required"] = []
        result["regions_ok"] = None
    return result


def run_job(cfg):
    """One staging job or one arm, in its own process, with a health sample on
    either side of it.

    A new process per job is not tidiness.  Compiled kernel bodies are cached
    at module level for the life of a process, the allocator keeps its pools,
    and the wrappers themselves are installed on library classes, so a second
    arm in the same interpreter would inherit the first arm's compiles and
    stack a second set of wrappers on top of the first.
    """
    before = mg44.sample_gpu_health()
    started = time.time()
    try:
        result = (run_stage(cfg) if cfg["kind"] == "stage" else run_arm(cfg))
    finally:
        after = mg44.sample_gpu_health()
    result["gpu_health_before"] = before
    result["gpu_health_after"] = after
    result["gpu_hot"] = mg44.row_is_hot(before) or mg44.row_is_hot(after)
    result["gpu_throttle"] = mg44.throttle_reasons(before + after)
    result["worker_wall_s"] = time.time() - started
    return result


# ── the driver ────────────────────────────────────────────────────────────────
def job_env(cfg):
    """The environment that DEFINES a job, set explicitly so nothing is
    inherited from the submitting shell.

    MBIRTORCH_NUM_DEVICES is popped and NEVER set.  That is the one place this
    run's environment differs from mg44's, and it is deliberate: every arm here
    places its model with an explicit ``configure_devices`` call, which is the
    protocol the recorded denoiser rows were measured under, and a pin left in
    the environment beside it would be a second mechanism claiming the same
    decision.

    MBIRTORCH_MEMORY_CALIBRATION is popped and never set: this is a timing
    probe and the mode does extra work.  MBIRTORCH_WIDENING_GUARD is popped
    because an explicit layout bypasses the speed floors anyway.
    MG44_SMOKE is set to match this run's mode, because the worker re-imports
    mg44 and that import is where mg44 decides whether to record events.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_WIDENING_GUARD", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # the shipped configuration
    env["MG44_SMOKE"] = "1" if SMOKE else "0"
    env["MG49_SMOKE"] = "1" if SMOKE else "0"
    return env


def spawn(cfg):
    """Run one configuration in a NEW interpreter.

    The row goes through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.  On an hour-long job that is
    the difference between watching progress and waiting in the dark.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR,
                            "_mg49_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR,
                            "_mg49_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                           "--worker", cfg_path, out_path], env=job_env(cfg))
    wall = time.perf_counter() - start
    if not os.path.exists(out_path):
        # A job that ran out of device memory lands here.  At the 1664-class
        # that is a READING -- the capacity boundary -- not a harness fault, so
        # it is recorded as a row and the run goes on.
        row = dict(cfg, error="worker exited {} and wrote no row".format(
            proc.returncode))
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    row["subprocess_wall_s"] = wall
    return row


def build_plan():
    """Every job, in run order: each cell's staging immediately before that
    cell's own arms."""
    keep = mg44._strict_subset("MG49_ARMS", all_arm_ids())
    arms = [cfg for cfg in all_arms() if cfg["arm"] in keep]
    plan, staged = [], set()
    for cfg in arms:
        key = tuple(cfg["cell"])
        if key not in staged:
            staged.add(key)
            plan.append(dict(kind="stage", family="denoiser",
                             cell=list(cfg["cell"]),
                             declared_cell=list(cfg["declared_cell"]),
                             n_dev=1, wrapped=False,
                             job_id="stage_{}x{}x{}".format(*cfg["cell"])))
        plan.append(cfg)
    if not plan:
        raise ValueError("MG49_ARMS selects no arm")
    return plan


def print_plan(plan):
    arms = [c for c in plan if c["kind"] == "arm"]
    stages = [c for c in plan if c["kind"] == "stage"]
    print("mg49 the denoiser ladder and the sharded sweep's split: {} arm(s) "
          "and {} staged input(s), device {}".format(len(arms), len(stages),
                                                     DEVICE))
    print("  jsonl -> " + RESULTS_DIR)
    print("  staged inputs read from (and written to, if missing) -> "
          + SINO_DIR)
    print("  protocol: seed {} reset before every call, sigma_noise {}, "
          "{} iteration(s), one cold pass then {} warm repeat(s), the timed "
          "call including the per-device synchronize and the gather -- the "
          "floors refresh's protocol, so these walls compare with its recorded "
          "ones".format(SEED, DENOISE_SIGMA, ITERATIONS, WARM_REPEATS))
    print("  placement: configure_devices(devices=[...]) at EVERY count; "
          "MBIRTORCH_NUM_DEVICES is popped from every job environment and "
          "never set, because an explicit layout is the protocol the recorded "
          "denoiser rows were measured under")
    print("  the instrument is imported from mg44_component_split (mg44.DEVICE "
          "is {!r}, matching this run); every wrapped region is timed on BOTH "
          "clocks: the host clock around the call, and a pair of CUDA events "
          "on each relevant device's default stream.  Nothing is synchronized "
          "during a call; the events are read afterwards".format(mg44.DEVICE))
    print("  the regions, in wrapping order: "
          + ", ".join([region for _a, region, _m in MODEL_SEAMS]
                      + [region for _a, region in SHARDING_SEAMS]
                      + [ELL1_REGION, "fan-out <worker name>",
                         "move_shard (tally only)"]))
    print("  NOT wrapped, deliberately: denoising.vcd_subset_denoiser and "
          "qggmrf.qggmrf_gradient_and_hessian_at_indices.  Both are handed to "
          "maybe_compile, so a wrapper on either would be COMPILED and dynamo "
          "would trace the timing calls.  That is why the one-device sweep is "
          "reported as a residual rather than measured")
    print("  the walls this run is anchored against, from " + RECORDED_SOURCE
          + ":")
    for cell in sorted(RECORDED_WALLS):
        rec = RECORDED_WALLS[cell]
        print("    {:>20}  n1 {:>6.3f} s  n2 {:>6.3f} s  n4 {:>6.3f} s   "
              "{:.3f}x (n2)  {:.3f}x (n4)".format(
                  str(cell), rec["n1"], rec["n2"], rec["n4"],
                  rec["ratio_n2"], rec["ratio_n4"]))
    if SMOKE:
        print("  SMOKE: one tiny cell {} on virtual CPU devices, {} "
              "iteration(s), {} warm repeat(s), no arm above {} devices.  "
              "There are no CUDA events there, so every device column reads "
              "zero and only the host columns carry numbers; what the smoke "
              "proves is that every seam resolves and every region fires, and "
              "it ASSERTS both".format(SMOKE_CELL, ITERATIONS, WARM_REPEATS,
                                       SMOKE_MAX_DEVICES))
    job_col = max([len(cfg["job_id"]) for cfg in plan] + [len("job")]) + 2
    header = ("  {:<{w}}{:>6}{:>9}{:>22}  what it does".format(
        "job", "dev", "wrapped", "cell", w=job_col))
    print(header)
    for cfg in plan:
        if cfg["kind"] == "stage":
            what = "hashes this cell's input, building it if absent"
            wrapped = "-"
        else:
            what = ("one cold pass and the warm repeats, every region timed"
                    if cfg["wrapped"] else
                    "the same protocol with NO wrappers: a plain wall")
            wrapped = "yes" if cfg["wrapped"] else "no"
        print("  {:<{w}}{:>6}{:>9}{:>22}  {}".format(
            cfg["job_id"], cfg["n_dev"], wrapped, str(tuple(cfg["cell"])),
            what, w=job_col))
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
                            "mg49_denoiser_{}_{}.jsonl".format(RUN_LABEL,
                                                               stamp))
    print_plan(plan)
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        header = dict(row="run_header", script="mg49_denoiser_split.py",
                      node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
                      python=sys.executable, results_dir=RESULTS_DIR,
                      sino_dir=SINO_DIR, seed=SEED,
                      denoise_sigma=DENOISE_SIGMA, iterations=ITERATIONS,
                      warm_repeats=WARM_REPEATS,
                      instrument="mg44_component_split",
                      mg44_device=mg44.DEVICE,
                      recorded_source=RECORDED_SOURCE,
                      recorded_walls=[dict(cell=list(cell), **rec)
                                      for cell, rec in RECORDED_WALLS.items()],
                      regions_always=list(REGIONS_ALWAYS),
                      regions_multi_device=list(REGIONS_MULTI_DEVICE),
                      regions_expected_silent=list(REGIONS_EXPECTED_SILENT),
                      region_level=dict(REGION_LEVEL),
                      component_fanouts=list(COMPONENT_FANOUTS),
                      plan=[dict(c) for c in plan])
        sink.write(json.dumps(header) + "\n")
        sink.flush()
        for index, cfg in enumerate(plan):
            print("\n  [{}/{}] {}".format(index + 1, len(plan),
                                          cfg["job_id"]), flush=True)
            row = spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if row.get("error"):
                print("    ERROR: {}".format(str(row["error"])[:400]),
                      flush=True)
            elif cfg["kind"] == "arm":
                print("    cold {:.2f}s  warm {:.2f}s  spread {:.1%}  "
                      "{} device(s)  {} sample(s)  finite {}".format(
                          row.get("cold_s", 0), row.get("warm_s", 0),
                          row.get("spread") or 0,
                          row.get("realized_n_devices", "-"),
                          row.get("total_samples", 0),
                          row.get("recon_finite")), flush=True)
        summary = summarize(rows, plan, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        sink.write(json.dumps(dict(row="summary", **summary)) + "\n")
        sink.flush()
    print("\nwrote " + out_path)
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


# ── reading the rows ──────────────────────────────────────────────────────────
def _gb(num_bytes):
    return num_bytes / float(2 ** 30)


def per_call_regions(row):
    """One arm's regions, reduced to PER WARM CALL numbers.

    Returns ``{region: entry}`` with the level, the call count, the host
    milliseconds and the device milliseconds per device, each divided by the
    number of warm calls, so arms with different repeat counts are directly
    comparable.

    The cold pass is left out on purpose: it pays this process's compiles, and
    the walls this run explains are warm walls.

    The in-iteration flag is FOLDED AWAY here rather than kept as part of the
    key, unlike mg44's reconstruction report.  In a denoise each region fires
    at exactly one nesting per device count -- the setup regions always outside
    the sweep, the fan-outs and the halo exchange always inside it -- with one
    exception that makes folding the right choice rather than a shortcut: the
    per-pass ell-1 runs INSIDE the sharded sweep above one device and OUTSIDE
    any sweep region at one device, because a one-device denoise never enters
    ``_denoise_sharded``.  Keying on the flag would file the same component
    under two different keys and make the one column that compares the two
    implementations disappear.
    """
    warm_calls = len(row.get("warm_all") or []) or 1
    out = {}
    for entry in row.get("regions") or []:
        if entry["call"] == 0:
            continue
        region = entry["region"]
        slot = out.get(region)
        if slot is None:
            slot = out[region] = dict(
                region=region,
                level=entry.get("level") or mg44.region_level(region),
                calls=0.0, host_ms=0.0, device_ms={}, in_iteration=False)
        slot["calls"] += entry["calls"] / warm_calls
        slot["host_ms"] += entry["host_ms"] / warm_calls
        slot["in_iteration"] = slot["in_iteration"] or bool(
            entry["in_iteration"])
        for device in entry.get("per_device") or []:
            index = device["device_index"]
            slot["device_ms"][index] = (slot["device_ms"].get(index, 0.0)
                                        + device["total_ms"] / warm_calls)
    for slot in out.values():
        values = list(slot["device_ms"].values())
        # The WALL is set by the slowest device, so the maximum over devices is
        # the number that competes with it; the sum is kept beside it because a
        # region spread evenly and a region that runs on one device look the
        # same in the maximum alone.
        slot["device_ms_max"] = max(values) if values else None
        slot["device_ms_sum"] = sum(values) if values else None
    return out


def _regions_at_level(table, level):
    return sorted([entry for entry in table.values()
                   if entry["level"] == level],
                  key=lambda e: -(e["device_ms_max"] or e["host_ms"] or 0.0))


def one_device_sweep_residual(row):
    """The one-device sweep, DERIVED rather than measured.

    A one-device denoise runs its whole per-subset update inside one compiled
    call, and that call cannot be wrapped without being compiled along with the
    body (see the module docstring).  So the one-device sweep is the total
    minus the setup regions, on each clock.  Everything unaccounted for --
    including the per-pass ell-1, which does fire at one device -- lands inside
    this residual.  It is labeled a residual everywhere it is printed.

    Returns ``(device_ms_by_index, host_ms)``, either of which may be empty or
    None when the arm recorded no regions.
    """
    table = per_call_regions(row)
    total = table.get("denoise call")
    if total is None:
        return {}, None
    setup = [entry for entry in table.values() if entry["level"] == "setup"]
    device_ms = {}
    for index, span in total["device_ms"].items():
        device_ms[index] = span - sum(entry["device_ms"].get(index, 0.0)
                                      for entry in setup)
    host_ms = total["host_ms"] - sum(entry["host_ms"] for entry in setup)
    return device_ms, host_ms


# ── the report ────────────────────────────────────────────────────────────────
def print_ladder(by_arm):
    """QUESTION A: the walls at one, two and four devices, the ratios against
    one device, and a verdict per (cell, count) under the shipped coarse
    rule."""
    from mbirtorch._widening_floors import ADMISSION_MARGIN

    print("\n===== QUESTION A: the ladder =====")
    print("Warm medians in seconds, and the ratio warm(n1) over warm(n), so "
          "above 1.00 means the wider count is faster.  Both denoiser floor "
          "rows are taken against ONE device, so that is what these ratios "
          "compare.  The admission rule is the shipped coarse one: a win must "
          "clear 1.00 by more than the cell's own warm spread AND reach the "
          "{:.2f}x admission margin, which is read from "
          "mbirtorch._widening_floors.ADMISSION_MARGIN rather than written "
          "down here.".format(ADMISSION_MARGIN))
    print("  The provenance of the recorded line beneath each measured cell is "
          + RECORDED_SOURCE + ".")
    header = ("{:<20}{:>10}{:>10}{:>10}{:>9}{:>9}{:>9}   {}".format(
        "cell", "n1 warm", "n2 warm", "n4 warm", "spread", "n2/n1", "n4/n1",
        "verdict"))
    print(header)
    print("-" * len(header))
    admitted = []
    for cell in LADDER_CELLS:
        walls, spreads = {}, []
        for arm, n_dev in arms_for(cell, PART_LADDER):
            row = by_arm.get(arm)
            if row is None:
                continue
            walls[n_dev] = row.get("warm_s")
            if row.get("spread") is not None:
                spreads.append(float(row["spread"]))
        spread = max(spreads) if spreads else None
        ratios, verdicts = {}, []
        for count in LADDER_COUNTS:
            base, wide = walls.get(1), walls.get(count)
            ratio = (base / wide) if (base and wide) else None
            ratios[count] = ratio
            if ratio is None:
                verdicts.append("n{}:-".format(count))
                continue
            clears_noise = spread is None or ratio > 1.0 + spread
            clears_margin = ratio >= ADMISSION_MARGIN
            if clears_noise and clears_margin:
                verdicts.append("n{}:WIN".format(count))
                admitted.append((cell, count, ratio))
            else:
                verdicts.append("n{}:no".format(count))
        print("{:<20}{}{}{}{}{}{}   {}".format(
            str(tuple(cell)),
            mg44._fmt(walls.get(1), 10, 3), mg44._fmt(walls.get(2), 10, 3),
            mg44._fmt(walls.get(4), 10, 3),
            "{:>9}".format("-") if spread is None
            else "{:>8.1f}%".format(spread * 100),
            mg44._fmt(ratios.get(2), 9, 3), mg44._fmt(ratios.get(4), 9, 3),
            " ".join(verdicts)))
        rec = RECORDED_WALLS.get(tuple(cell))
        if rec is not None:
            print("    recorded ({}): n1 {:.3f} s  n2 {:.3f} s  n4 {:.3f} s   "
                  "{:.3f}x (n2)  {:.3f}x (n4)".format(
                      RECORDED_LABEL, rec["n1"], rec["n2"], rec["n4"],
                      rec["ratio_n2"], rec["ratio_n4"]))
            if SMOKE:
                print("    SMOKE: this arm ran a tiny stand-in cell, so the "
                      "recorded line above names a problem it did not "
                      "measure.")
            else:
                for count in LADDER_COUNTS:
                    ratio = ratios.get(count)
                    recorded = rec.get("ratio_n{}".format(count))
                    if ratio is None or not recorded:
                        continue
                    drift = ratio / recorded - 1.0
                    if abs(drift) > ANCHOR_DRIFT_WARN:
                        print("    WARNING the anchor moved: this run's n{} "
                              "ratio {:.3f}x is {:+.0f} percent from the "
                              "recorded {:.3f}x.  The denoiser path in this "
                              "tree is the one those walls were measured on, "
                              "so a gap this large points at the RULER or the "
                              "node, not at problem size.".format(
                                  count, ratio, drift * 100, recorded))
    print("\n  " + ("the smallest admission found: cell {} at n={} "
                    "({:.3f}x)".format(tuple(admitted[0][0]), admitted[0][1],
                                       admitted[0][2])
                    if admitted else
                    "NO cell and count in this ladder clears the admission "
                    "rule, so on this evidence the two denoiser sentinel rows "
                    "stand."))


def print_capacity_line(rows):
    """The one line the ladder's answer needs to be final: what one device is
    modeled to hold, and where capacity takes the widening decision away from
    the speed sentinel."""
    print("\n----- the modeled one-device peak, and where capacity takes over "
          "-----")
    probe = None
    for row in rows:
        candidate = row.get("capacity_probe")
        if candidate and candidate.get("rows"):
            probe = candidate
            break
    if probe is None:
        print("  no staging job recorded a capacity probe, so this line has "
              "nothing to read.")
        return
    budgets = [row.get("device_total_bytes") for row in rows
               if row.get("device_total_bytes")]
    budget = min(budgets) if budgets else None
    margin = probe.get("margin")
    print("  The peak is the denoiser's own memory ledger at ONE device, read "
          "from the tree under test ({}); the demand adds the memory "
          "preflight's margin, {}.".format(
              "_build_memory_ledger(workload='denoise')",
              "{:.0%}, {}".format(margin, probe.get("margin_note"))
              if margin is not None else probe.get("margin_note")))
    header = "  {:<20}{:>14}{:>12}{:>12}{:>10}".format(
        "cell", "voxels", "peak GB", "demand GB", "fits?")
    print(header)
    print("  " + "-" * (len(header) - 2))
    crossover = None
    for entry in probe["rows"]:
        if entry.get("error"):
            print("  {:<20}  could not be priced: {}".format(
                str(tuple(entry["cell"])), entry["error"]))
            continue
        demand = entry.get("demand_bytes")
        fits = None if (budget is None or demand is None) else demand <= budget
        if fits is False and crossover is None:
            crossover = entry
        print("  {:<20}{:>14.3e}{}{}{:>10}".format(
            str(tuple(entry["cell"])), float(entry["voxels"]),
            mg44._fmt(_gb(entry["one_device_peak_bytes"]), 12, 2),
            mg44._fmt(None if demand is None else _gb(demand), 12, 2),
            "-" if fits is None else ("yes" if fits else "NO")))
    if budget is None:
        print("  no CUDA device budget was recorded (the smoke has none), so "
              "the fits column is empty and no crossover can be named.")
    elif crossover is None:
        print("  every priced cell still fits one {:.0f} GB device, so the "
              "speed sentinel governs across this whole probe.".format(
                  _gb(budget)))
    else:
        print("  capacity takes the decision away from the speed sentinel at "
              "the {} cell: its {:.1f} GB demand exceeds the {:.0f} GB device. "
              " Above that size the preflight widens a denoise whatever the "
              "sentinel says, so the largest cell BELOW it is the last size at "
              "which the speed question has any practical effect.".format(
                  str(tuple(crossover["cell"])),
                  _gb(crossover["demand_bytes"]), _gb(budget)))
    print("  The largest cell this run MEASURED is {}.".format(
        str(tuple(LADDER_CELLS[-1]))))


def print_finiteness(by_arm):
    """The int32-boundary watch.  At the 1536- and 1664-class the volume
    crosses 2**31 elements, and an element count or a flat index that
    overflowed there is a finding about the LIBRARY, not about this probe."""
    print("\n----- the output check at and above 2**31 voxels -----")
    print("A non-finite output, or a checksum that disagrees across device "
          "counts by far more than float32 reordering explains, is a FINDING "
          "ABOUT THE LIBRARY.  It is printed here and gates nothing.")
    problems = []
    for cell in LADDER_CELLS:
        checks = []
        for arm, n_dev in arms_for(cell, PART_LADDER):
            row = by_arm.get(arm)
            if row is None or row.get("recon_checksum") is None:
                continue
            checks.append((arm, n_dev, row["recon_checksum"],
                           row.get("recon_finite")))
        if not checks:
            continue
        base = checks[0][2]
        for arm, n_dev, checksum, finite in checks:
            spread = abs(checksum - base) / abs(base) if base else None
            flag = ""
            if finite is False:
                flag = "  <-- NOT FINITE"
                problems.append("{}: the output is not finite".format(arm))
            elif spread is not None and spread > 1e-3:
                flag = "  <-- {:.2%} from the n=1 checksum".format(spread)
                problems.append(
                    "{}: its checksum is {:.2%} from the one-device "
                    "value".format(arm, spread))
            print("  {:<20}n={}  checksum {:.6e}  finite {}{}".format(
                arm, n_dev, checksum, finite, flag))
    if problems:
        print("  FINDING (about the library, not the instrument):")
        for item in problems:
            print("    " + item)
    else:
        print("  every ladder arm returned a finite output, and the checksums "
              "agree across device counts to within float32 reordering.")


def _print_level_table(title, arms, level, walls):
    """One reporting level's table, with a column group per device count.

    The device column is the per-warm-call device milliseconds on the BUSIEST
    device, because that is the one competing with the wall.  The host column
    is the per-warm-call host milliseconds summed over the calls, which is what
    the region cost to ENQUEUE.
    """
    print("\n  " + title)
    tables = [(n_dev, per_call_regions(row)) for n_dev, row in arms]
    regions = []
    for _n, table in tables:
        for entry in _regions_at_level(table, level):
            if entry["region"] not in regions:
                regions.append(entry["region"])
    if not regions:
        print("    no region fired at this level")
        return
    header = "    {:<{w}}".format("region", w=REGION_COL)
    for n_dev, _t in tables:
        header += "{:>10}{:>11}{:>10}".format("n{} calls".format(n_dev),
                                              "n{} dev ms".format(n_dev),
                                              "n{} host".format(n_dev))
    print(header)
    print("    " + "-" * (len(header) - 4))
    for region in regions[:TABLE_ROWS]:
        line = "    {:<{w}}".format(region, w=REGION_COL)
        for n_dev, table in tables:
            entry = table.get(region)
            line += (mg44._fmt(entry["calls"] if entry else None, 10, 0)
                     + mg44._fmt(entry["device_ms_max"] if entry else None,
                                 11, 1)
                     + mg44._fmt(entry["host_ms"] if entry else None, 10, 1))
        print(line)
    if len(regions) > TABLE_ROWS:
        print("    ... {} further region(s) are in the jsonl".format(
            len(regions) - TABLE_ROWS))
    if level == "total":
        line = "    {:<{w}}".format("(the warm wall, ms)", w=REGION_COL)
        for n_dev, _t in tables:
            line += "{:>10}".format("-") + mg44._fmt(walls.get(n_dev), 11, 1) \
                + "{:>10}".format("-")
        print(line)


def print_attribution(by_arm):
    """QUESTION B: where the sharded sweep's time goes, at the 1024-class."""
    print("\n===== QUESTION B: the attribution at {} =====".format(
        str(tuple(ATTRIBUTION_CELL))))
    arms, walls = [], {}
    planned = set(all_arm_ids())
    for arm, n_dev in arms_for(ATTRIBUTION_CELL, PART_ATTRIBUTION):
        row = by_arm.get(arm)
        if row is None:
            print("  {} {}".format(
                arm, "is not planned in this mode, so it has no column"
                if arm not in planned else "produced no row"))
            continue
        arms.append((n_dev, row))
        walls[n_dev] = (row.get("warm_s") or 0.0) * 1000.0
    if not arms:
        print("  no wrapped arm of this cell produced a row.")
        return
    print("  Milliseconds are PER WARM CALL.  The device column is that "
          "region's span on the busiest device; the host column is what the "
          "region cost to enqueue.  A region whose host time is large beside "
          "its device time is paying for dispatch, a thread handoff or a "
          "compile guard rather than waiting on a device.")
    print("  The one-device and multi-device denoisers are DIFFERENT "
          "implementations, so this is not one workload split three ways: at "
          "n=1 the whole per-subset update is one compiled call, and above n=1 "
          "it is two fan-outs, four scalar reductions on the lead device, a "
          "step-size broadcast back and a halo exchange per pass.")
    _print_level_table("the whole call", arms, "total", walls)
    _print_level_table("setup, before the sweep starts", arms, "setup", walls)
    _print_level_table("the sweep itself (measured above one device only)",
                       arms, "sweep", walls)
    _print_level_table("the components that tile the sharded sweep", arms,
                       "component", walls)

    print("\n  the sweep's own span, and what is NOT inside a named component")
    print("    The residual is the sweep's device span on one device minus the "
          "spans of the components on that same device.  It is the gap between "
          "components: dispatch, the thread-pool handoff, and the cross-device "
          "lock-step where one device waits for another's scalar.")
    for n_dev, row in arms:
        table = per_call_regions(row)
        sweep = table.get("sweep (sharded)")
        if sweep is None:
            device_ms, host_ms = one_device_sweep_residual(row)
            parts = ["dev {}: {:.1f} ms".format(index, device_ms[index])
                     for index in sorted(device_ms)]
            print("    n={}  sweep DERIVED as a residual (the total minus the "
                  "setup regions), not measured: {}{}".format(
                      n_dev, "; ".join(parts) or "no device timing",
                      "" if host_ms is None
                      else "; host {:.1f} ms".format(host_ms)))
            continue
        components = [entry for entry in table.values()
                      if entry["level"] == "component"]
        parts = []
        for index in sorted(sweep["device_ms"]):
            span = sweep["device_ms"][index]
            inside = sum(entry["device_ms"].get(index, 0.0)
                         for entry in components)
            parts.append("dev {}: {:.1f} ms of {:.1f} ms ({:.0f}%)".format(
                index, span - inside, span,
                100.0 * (span - inside) / span if span else 0.0))
        print("    n={}  residual inside the sweep: {}".format(
            n_dev, "; ".join(parts) or "no device timing (the smoke has no "
                                       "CUDA events)"))

    print("\n  the scalar traffic: move_shard, counted and timed on the host "
          "only")
    print("    It fires many times per subset for scalar-sized moves -- one "
          "per device for each combined line-search partial, plus the "
          "step-size broadcast -- so it is tallied rather than sampled.  The "
          "question it answers is whether that traffic is visible at all.")
    for n_dev, row in arms:
        warm_calls = len(row.get("warm_all") or []) or 1
        moves = [entry for entry in row.get("move_shard") or []
                 if entry["call"] > 0]
        calls = sum(entry["calls"] for entry in moves) / warm_calls
        host_s = sum(entry["host_s"] for entry in moves) / warm_calls
        print("    n={}  {:.0f} move(s) per warm call, {:.1f} ms of host time "
              "in all".format(n_dev, calls, host_s * 1000.0))


def print_one_device_note(by_arm):
    """Why the one-device sweep is a residual, said once, plainly."""
    print("\n===== the one-device note =====")
    print("  The one-device sweep in every table above is a RESIDUAL, not a "
          "measurement.  Its whole per-subset update is a single call to "
          "denoising.vcd_subset_denoiser, which is handed to maybe_compile.  A "
          "timing wrapper on it would be compiled along with the body: dynamo "
          "would trace time.perf_counter and the CUDA event calls, and would "
          "either graph-break or fail -- and either way it would change the "
          "thing being measured.  The same is true of "
          "qggmrf.qggmrf_gradient_and_hessian_at_indices, which the sharded "
          "path also compiles per device.  So the one-device arm has no "
          "internal seams at all, and its sweep is derived as the whole "
          "denoise call minus the wrapped setup regions.  Everything the "
          "one-device sweep does lands inside that residual, including the "
          "per-pass ell-1, which is the ONE component both implementations "
          "share and the only line in the component table that compares them "
          "directly.")
    silent = []
    for arm, _cell, _n, wrapped, _part in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None or not wrapped:
            continue
        fired = set(row.get("region_labels") or [])
        for region in REGIONS_EXPECTED_SILENT:
            if region not in fired:
                silent.append((arm, region))
    if silent:
        print("  Expected silence, confirmed: {} recorded no calls, because "
              "denoise reaches the noise estimate only when the caller "
              "supplies no sigma_noise and this protocol always supplies "
              "it.".format(", ".join(sorted(set(
                  "{} on {}".format(region, arm)
                  for arm, region in silent)))))


def print_overhead(by_arm):
    """What the instrument itself costs: the 1024-class wrapped against the
    same cell with no wrappers, at one and two devices."""
    print("\n===== instrument overhead =====")
    print("The control arms run the identical protocol with NO wrappers "
          "installed.  A ratio within a few percent of 1.000 means the wrapped "
          "numbers describe the library; a larger one is a finding about this "
          "probe rather than about the denoiser.")
    header = "{:>7}{:>13}{:>13}{:>18}".format("count", "wrapped s",
                                              "control s", "wrapped/control")
    print(header)
    print("-" * len(header))
    for n_dev in (1, 2):
        wrapped = by_arm.get("s1024_n{}".format(n_dev))
        control = by_arm.get("s1024_n{}_control".format(n_dev))
        a = wrapped.get("warm_s") if wrapped else None
        b = control.get("warm_s") if control else None
        ratio = (a / b) if (a and b) else None
        print("{:>7}{}{}{}".format(n_dev, mg44._fmt(a, 13, 3),
                                   mg44._fmt(b, 13, 3),
                                   mg44._fmt(ratio, 18, 3)))


def print_observations(by_arm):
    """One line per arm, short enough to paste into a note."""
    print("\n===== paste-ready observations =====")
    print("One line per arm.  Every number here is a FINDING and none of them "
          "touches the exit code.")
    planned = set(all_arm_ids())
    for arm, cell, n_dev, _wrapped, part in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None:
            print("  {:<{w}}  {}".format(
                arm, "not planned in this mode" if arm not in planned
                else "no row", w=ARM_COL))
            continue
        warm = row.get("warm_s")
        if warm is None:
            reasons = row.get("invalid_reasons") or [
                str(row.get("error", "reason not recorded"))[:160]]
            print("  {:<{w}}  no wall: {}".format(arm, "; ".join(reasons),
                                                  w=ARM_COL))
            continue
        base = None
        for other, other_n in arms_for(cell, part):
            if other_n == 1:
                base_row = by_arm.get(other)
                base = base_row.get("warm_s") if base_row else None
        ratio = ("{:.3f}x vs n1".format(base / warm)
                 if (base and n_dev != 1) else "the n1 anchor"
                 if n_dev == 1 else "no n1 arm")
        print("  {:<{w}}  {} n={}  warm {:.3f} s  spread {:.1%}  {}  "
              "finite {}  {} sample(s)".format(
                  arm, str(tuple(cell)), n_dev, warm, row.get("spread") or 0,
                  ratio, row.get("recon_finite"), row.get("total_samples", 0),
                  w=ARM_COL))


def smoke_checks(by_arm):
    """The smoke's assertions, which DO gate its exit code.

    The smoke measures nothing worth reading -- one tiny cell on virtual CPU
    devices with no timing events at all.  What it proves is that the
    instrument is wired to the library: that mg44 was imported in the matching
    mode, that every seam resolved, and that every region a two-device denoise
    must reach actually fired.  A probe that silently stopped attaching to a
    renamed function would print a complete-looking table of nothing, and this
    is what catches that before a cluster job pays for it.
    """
    problems = []
    if mg44.DEVICE != DEVICE:
        problems.append("mg44 was imported in {!r} mode and this run is {!r}"
                        .format(mg44.DEVICE, DEVICE))
    required_two_device = ("sweep (sharded)", "halo exchange", ELL1_REGION,
                           FANOUT_PREFIX + "terms_worker",
                           FANOUT_PREFIX + "apply_worker")
    for arm, _cell, n_dev, wrapped, _part in ARM_SPECS:
        row = by_arm.get(arm)
        if row is None:
            continue
        if wrapped and row.get("seams_ok") is not True:
            problems.append("{}|not every seam resolved".format(arm))
        if not wrapped or n_dev < 2:
            continue
        fired = set(row.get("region_labels") or [])
        for name in required_two_device:
            if name not in fired:
                problems.append("{}|the region {!r} never fired".format(arm,
                                                                        name))
    return problems


def summarize(rows, plan, out_path):
    """The blocks a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  Whether
    a device count wins, how far a ratio drifted from the recorded one, whether
    an output was finite, whether a card was hot: all FINDINGS, all printed,
    none of them gated.  An arm that produced no row, realized the wrong device
    count, failed to resolve a seam, ran with the calibration mode on, read
    different bytes from its siblings, or left a region its device count should
    have exercised with no calls did not measure what the plan said it would,
    and that is an instrument failure.
    """
    print("\n===== mg49 the denoiser ladder and sweep split ({}) =====".format(
        out_path))
    broken, findings = [], []
    by_arm, stages = {}, []

    for row in rows:
        job_id = row.get("job_id", "?")
        if row.get("error"):
            print("{:<{w}}  ERROR: {}".format(
                job_id, str(row["error"]).splitlines()[-1][:160], w=ARM_COL))
            broken.append("{}|error".format(job_id))
            continue
        if row.get("kind") == "stage":
            stages.append(row)
            broken.extend("{}|{}".format(job_id, reason)
                          for reason in row.get("invalid_reasons") or [])
            continue
        by_arm[row["arm"]] = row
        for reason in row.get("invalid_reasons") or []:
            broken.append("{}|{}".format(row["arm"], reason))
        if row.get("gpu_hot"):
            findings.append("{}: GPU hot during this arm, so its wall may be a "
                            "thermal reading rather than a device-count "
                            "one".format(row["arm"]))
        if row.get("gpu_throttle"):
            findings.append("{}: throttle reasons {}".format(
                row["arm"], row["gpu_throttle"]))
        if row.get("recon_finite") is False:
            findings.append("{}: the denoised output is NOT finite -- a "
                            "finding about the library at this size, not about "
                            "this probe".format(row["arm"]))
        for entry in row.get("dynamo") or []:
            if entry["call"] and entry.get("compiled_functions_delta"):
                findings.append(
                    "{}: warm call {} compiled {} further function(s), so that "
                    "repeat paid compile time the others did not".format(
                        row["arm"], entry["call"],
                        entry["compiled_functions_delta"]))
        if not row.get("padding_present"):
            findings.append("{}: the padding witness did not read {} -- this "
                            "is not the tree the recorded walls describe"
                            .format(row["arm"], PAD_PROBE_EXPECTED))
        if not row.get("recompile_remedy_present"):
            findings.append("{}: the recompile-remedy witness is absent (floor "
                            "{}), so this tree is not the remedied one"
                            .format(row["arm"], row.get("recompile_limit_floor")))

    # Every PLANNED arm produced a row.  Read off the plan rather than off the
    # rows: an arm whose subprocess died before writing anything leaves no row
    # to notice its absence in.
    reported = set(item.split("|", 1)[0] for item in broken)
    for cfg in plan:
        name = cfg.get("arm")
        if name and name not in by_arm and name not in reported:
            broken.append("{}|no row".format(name))

    # Every arm of a cell must have read the SAME bytes.  There are no md5
    # sidecars in the staged directory, so each arm hashed what it loaded and
    # this is where the digests are compared: an arm that denoised a different
    # array than its siblings is not comparable with them, and nothing in its
    # own row would say so.
    digests = {}
    for row in list(by_arm.values()) + stages:
        key = tuple(row.get("cell") or ())
        digest = row.get("sino_md5")
        if digest:
            digests.setdefault(key, {}).setdefault(digest, []).append(
                row.get("job_id"))
    for key, seen in sorted(digests.items()):
        if len(seen) > 1:
            broken.append(
                "{}|the jobs of this cell read different inputs: {}".format(
                    key, "; ".join("{} <- {}".format(digest[:12],
                                                     ", ".join(jobs))
                                   for digest, jobs in sorted(seen.items()))))

    for row in stages:
        print("staged {}: md5 {}{}".format(
            str(tuple(row["cell"])), str(row.get("sino_md5", "-"))[:12],
            "  (reused from disk)" if row.get("reused")
            else "  (built by this run in {} chunk(s))".format(
                row.get("stage_chunks", "?"))))

    print_ladder(by_arm)
    print_capacity_line(stages + list(by_arm.values()))
    print_finiteness(by_arm)
    print_attribution(by_arm)
    print_one_device_note(by_arm)
    print_overhead(by_arm)
    print_observations(by_arm)

    if SMOKE:
        problems = smoke_checks(by_arm)
        print("\n-- smoke assertions --")
        if problems:
            for item in problems:
                print("  FAIL " + item)
        else:
            print("  mg44 was imported in the matching mode, every seam "
                  "resolved on every wrapped arm, and at the two-device "
                  "wrapped arm the regions 'sweep (sharded)', 'halo exchange', "
                  "'per-pass ell1', 'fan-out terms_worker' and 'fan-out "
                  "apply_worker' all fired")
        broken.extend(problems)

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print("  BROKEN " + item)
    else:
        print("  every planned arm ran, realized its configured device count, "
              "resolved every seam, ran with the calibration mode off, read "
              "the same input as its siblings, and exercised every region its "
              "device count can exercise")
    for item in findings:
        print("  finding (not gated) " + item)
    if not findings:
        print("  no thermal, throttle, finiteness, warm-compile or tree-"
              "witness findings")

    return dict(healthy=not broken, broken=broken, findings=findings,
                arms=dict((name, dict(
                    cell=row.get("cell"), declared_cell=row.get("declared_cell"),
                    wrapped=row.get("wrapped"),
                    realized_n_devices=row.get("realized_n_devices"),
                    cold_s=row.get("cold_s"), warm_s=row.get("warm_s"),
                    spread=row.get("spread"),
                    recon_checksum=row.get("recon_checksum"),
                    recon_finite=row.get("recon_finite"),
                    total_samples=row.get("total_samples"),
                    region_labels=row.get("region_labels")))
                    for name, row in by_arm.items()))


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
