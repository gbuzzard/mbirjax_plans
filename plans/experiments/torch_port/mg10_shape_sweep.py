"""mg10 -- the two proposed multi-GPU forward remedies, measured on OUR kernels
before either one is written into the library.

WHAT THIS FILE IS FOR.  mg9 settled the mechanism: the forward projection's
per-device time is kernel-busy time, 97 to 98 percent of the bracket at every
device count, so copying and waiting are not what keeps the forward from
getting faster when GPUs are added.  A source reading of mbirjax then supplied
the SHAPE of the fix, and it is a different shape in each geometry.  Neither
shape carries a number that is ours.  mg10 measures both on our own Triton
kernels, changing nothing in mbirtorch: everything here is installed on the
live model object at run time.

    Experiment 1, the parallel knee.  Today each slice-owner's whole shard is
    walked as ONE band -- 504 slices wide at two devices, 252 at four.  mbirjax
    walks fixed 256-slice bands instead and its source says why: "the measured
    knee; whole-shard bands are WORSE at scale".  mg9 measured two points on
    our kernel, 41.5 ms per launch at a 504-slice band and 21.4 ms at 252, so
    our knee is somewhere in or below that interval.  This experiment sweeps
    the sub-band length and reads the per-launch time and the per-device span
    against it.

    Experiment 2, the cone column gather.  Today cone walks slice bands like
    parallel and adds up one partial sinogram per band.  mbirjax does the
    opposite: for each batch of pixel columns it gathers that batch's
    FULL-HEIGHT cylinder from every slice-owner and makes ONE projector call
    spanning every slice.  mbirjax measured that form at about 2x faster than
    the banded one, for about 10 percent more transient memory.  This
    experiment builds that shape as a harness prototype -- the same kernel
    bodies, the same transfer primitive -- and reads time, memory and VALUE
    against the banded path we ship.

    Experiment 3, THE ONE-DEVICE WIDTH DISCRIMINATOR (added after the first
    mg10 run, and OFF unless MG10_ARMS asks for it by name).  Experiment 1's
    sweep produced one result nobody can attribute yet.  At parallel 1024 with
    two devices the per-launch time is very nearly LINEAR in the sub-band
    width -- 4.70, 10.99, 17.01, 21.21 and 41.51 ms at widths 63, 126, 168, 252
    and 504 -- so the cost per slice is about 0.084 ms at EVERY width.  But the
    single-device arm runs the same kernel at 0.0412 ms per slice at width
    1008.  The cost per slice therefore DOUBLES between one device at width
    1008 and two-or-four devices at any width, and two different stories
    explain that equally well:
        (a) a MULTI-DEVICE effect -- the doubling follows from n > 1 itself
            (peer-visible memory, cold reads of a copied tensor, whatever), or
        (b) a KERNEL WIDTH effect -- the kernel is simply half as efficient per
            slice at widths at or below 504 as it is at 1008, at ANY device
            count.
    The cell that separates them was never run: ONE device with a NARROWED
    values block.  This experiment runs it, at widths 1008 (the shipped
    monolithic call), 504, 252 and 63.  If the narrowed one-device arms land
    near 0.041 ms per slice, the width is innocent and the doubling belongs to
    the device count; if they land near 0.084, the device count is innocent and
    the doubling belongs to the width.  Width 63 carries a second question of
    its own: it was the only sub-band that BEAT the control at two devices
    (by 9.5 percent), and it is the only one whose per-launch atomic write
    target (128 views x 63 rows x 1024 channels x 4 B = 33 MB) fits inside an
    H100's 50 MB L2 (126 rows is 66 MB, just over).  Whether that win survives
    at one device says whether it is an L2 effect or a sharding effect.

    Experiment 4, THE SAME SLAB FROM THE OTHER AXIS (also opt-in, and added
    with experiment 3).  One forward launch's atomic write target is the zeroed
    plane the kernel accumulates into: (view batch) x (channels) x (values
    columns), float32.  Experiment 3 varies the values columns of that plane;
    this varies the VIEW BATCH, at a values width nobody touches.  The two axes
    reach the same byte counts from different directions -- 8 views at the full
    1008-slice width writes what 128 views at 63 slices writes -- so if fitting
    the plane in cache is what the width-63 anomaly was about, the two have to
    agree.  Four arms: forward view chunk 8 and 16 at ONE device (the shipped
    1008-slice call, slabs of about 32 and 64 MB) and the same two at TWO
    devices (the shipped 504-slice bands, slabs of about 16 and 32 MB).  The
    two-device pair is a second discriminator on its own: if the two-device
    cost per slice falls toward 0.041 when the launch writes less, then how
    much one launch writes explains the doubling and the device count is a
    bystander.

WHAT THE DECISION READS OFF THIS.  Three numbers, one per open question in the
remedy memo's section 8.6, plus the two slab discriminations.
    * The parallel band knee: per-launch ms against sub-band length, at two and
      four devices.  A remedy exists only where (launches x per-launch time)
      falls below the control's.  Sub-banding multiplies the launch count by
      the number of sub-bands, so a band half as wide has to cost LESS THAN
      half as much per launch to pay for itself.  mg9's two points sit almost
      exactly on the break-even line, so the answer is not predictable and has
      to be measured.
    * The cone column-batch size: per-device span and per-device peak memory
      against the pixel batch, against the banded control at the same count.
    * The cone value gate: the column gather changes the order the vertical
      contributions are summed in, so this file records three distances and the
      run-to-run floor to read them against.
    * Experiment 3's discrimination: ms_per_slice at one device against the
      values-block width, printed beside the two- and four-device readings of
      the same column.  The summary prints the RULE (0.041 says multi-device,
      0.084 says width) and nothing else; the attribution is analysis.
    * Experiment 4's: the same column against the forward VIEW chunk, at one
      and two devices, with the write slab in megabytes beside every row so the
      crossing of the H100's 50 MB L2 is legible without arithmetic.

THE NORMALIZED COST COLUMNS, and why a null result must be legible.  A narrower
launch always has a smaller per-launch time, which on its own says nothing at
all.  Every table here therefore carries `ms_per_slice` -- the per-launch time
over the mean column count of the values blocks the kernel was actually handed
-- and its cone twin `ms_per_kpix`, over their mean pixel count in thousands.
mg9's own readings in that column are 0.0412 ms per slice at a 1008-slice band
against 0.0823 at 504 and 0.0849 at 252: the WIDEST band was the most efficient
per slice.  A sweep in which no length beats the shipped whole-shard control is
therefore a real possible outcome of this job, not a sign that something broke,
and the tables are built so that outcome reads at a glance instead of having to
be derived.

THE DESIGN NOTE'S SLOTS.  The note that consumes these rows has a fixed slot
list, and a slot filled by hand out of a table is a slot that can be filled
wrong.  The summary's last block prints each slot name verbatim with this run's
values beside it, and prints NOT MEASURED IN THIS RUN for any slot whose arms
did not run, so a narrowed or truncated job cannot leave a slot looking
answered.  No slot carries a verdict: where a slot names a "chosen" value the
harness prints the best point IN THIS SWEEP by the normalized column, with the
whole sweep beside it, and choosing stays the note's job.

TERMS OF ART, each defined once, here.
    arm          one subprocess run at fixed parameters -- one geometry, one
                 device count, and one shape knob.  Eighteen arms by default;
                 experiment 3 adds four more that run ONLY when MG10_ARMS names
                 them, so a plain re-run is the same eighteen it always was.
    sub-band     a fixed-length slice range within ONE slice-owner's shard.
                 Today there is one sub-band per owner and it is the whole
                 shard; experiment 1 makes them shorter.
    piece        experiment 3's unit: a width-W slice range of the WHOLE volume
                 at one device.  A piece is what a sub-band would be if a
                 sub-band existed at n = 1, and the two are deliberately tiled
                 by the same rule so their widths are comparable.
    column batch a batch of pixel columns whose FULL-HEIGHT cylinder (every
                 slice, gathered from every slice-owner) is assembled on one
                 view-owner and projected in a single call.  Experiment 2's
                 unit of work.
    control      an arm with no patch installed: the shipped path.  Every
                 control arm PROVES the patch was inactive, and every patched
                 arm PROVES the patch ran; see "the two witnesses" below.
    bracket      mg5's and mg9's per-device CUDA event pair around the whole
                 forward projection region.  Unchanged, so mg10's spans are
                 directly comparable with mg9's.
    busy time    the sum of per-call CUDA event pairs around each individual
                 forward projection BODY call on one device, plus the call
                 count.  Busy divided by calls is the per-launch time, which is
                 the number both experiments turn on.
    the gap      bracket minus busy: time inside the bracket that no body call
                 covers.

THE TWO WITNESSES, and why an arm dies without them.  A patch that silently
fails to engage produces a plausible row that means nothing, and a whole sweep
can be read off such rows without anyone noticing.  So each experiment carries
a counter that is checked against arithmetic derived from the arm's own
parameters, and the check runs after the FIRST timed reconstruction and aborts
the arm loudly on any disagreement.
    Experiment 1's witness is the band broadcast.  `broadcast_band_to_views` is
    called from exactly one place in mbirtorch -- the forward driver, once per
    sub-band -- so wrapping it counts sub-bands and measures each one's width
    without touching the back projection.  The expected count is
    (forward funnel calls) x (slice owners) x (sub-bands per owner) and the
    expected widths come from re-deriving the driver's balanced tiling here,
    independently of the library.  A control arm must show exactly one width,
    equal to the whole shard.
    Experiment 2's witness is the prototype's own counters: column batches
    gathered per view-owner, pieces moved per batch, and the COLUMN HEIGHT of
    every assembled cylinder.  The height histogram is the load-bearing one: a
    single key equal to the full slice count is what proves the gather is
    full-height and not a band by another name.  The prototype's arm also
    requires the broadcast counter to read exactly ZERO, which is what proves
    the banded branch did not run.
    Experiment 4's witness is the REALIZED FORWARD VIEW BATCH, which the
    harness already records per device by position.  It is asserted on every
    parallel arm, not only on the arms that narrow it: an arm whose batch
    quietly moved would report a write slab it never wrote.  A view-chunk arm
    must read its own chunk and nothing else; every other parallel arm must
    read the shipped chunk.  The arm also proves the seam is FORWARD-ONLY --
    the back chunk constant is re-read and must be untouched -- and, on the
    kernel path, that the module constant did all the work: the clamp that
    exists for the CPU smoke's torch bodies must never have fired.
    Experiment 3's witness is DOUBLE, and the two halves are recorded at
    different places by different code.  The splitter counts, per forward
    funnel call, how many pieces it cut and how wide each one was; the busy
    probe independently records the column count of every values block the
    KERNEL was handed.  Both are checked against this file's own re-derivation
    of the tiling, and they are checked against each other, so a splitter that
    counted pieces it did not actually project could not pass.  The one-device
    CONTROL arm proves the opposite: exactly one piece, of the full slice
    count, which is the shipped monolithic call.  Every experiment-3 arm also
    proves it ran on ONE device -- zero band fan-outs, zero column gathers,
    zero cross-device bytes, and a device list of length exactly one.

WHY THE PATCHES SIT WHERE THEY SIT.

    Experiment 1 sets `model.forward_project_slice_band`, the model attribute
    the forward driver already reads (tomography_model.py line 461) and turns
    into band bounds three lines later.  This is not a shortcut around a
    monkeypatch, it is the more faithful instrument: the proposed remedy IS a
    change to the default of `_slice_band_length`, so a fixed value through the
    existing knob produces the identical walk a library implementation would
    produce -- the same balanced tiling, the same one broadcast per sub-band,
    the same projector calls, the same assembly.  It is also the only seam that
    is forward-only.  `_slice_band_length` and `_balanced_slice_bounds` are
    static methods shared with the BACK driver, so shadowing either of them
    would re-band the back projection too and move a number this measurement
    does not want moved -- the timed reconstruction contains both.

    Experiment 2 replaces `model._sparse_forward_project_sharded` with a bound
    prototype, on the INSTANCE.  The public funnel resolves that name on `self`
    at call time (tomography_model.py line 341), so the replacement is picked
    up without editing the package, and it sits INSIDE mg9's forward bracket so
    the span stays comparable.  The prototype delegates to the original for a
    trivial (one-device) placement and for any row-aligned geometry, so it can
    only ever change the cone multi-device branch.  It returns exactly what the
    original returns: a `Shards` of one (views, rows, channels) tensor per
    view-owner, in device order, with any padded view tail zero-filled.

    Experiment 3 CANNOT use experiment 1's knob, and the driver's own source is
    why.  `_sparse_forward_project_sharded` returns on its first statement when
    the placement is trivial (tomography_model.py lines 448 to 452), thirteen
    lines BEFORE the line that reads `forward_project_slice_band` (line 461).
    At one device the knob is never consulted, so an experiment-1 arm at n = 1
    would set an attribute nothing reads and report the control under a
    narrowed arm's name -- the exact silent-no-op failure this file exists to
    prevent.  The batch file re-checks that ordering against the tree's own
    source before any experiment-3 arm runs.  So experiment 3 patches
    `Projectors._sparse_forward_project_single_device` instead, ON THE CLASS.
    That function is the single choke point for one device: its only two
    callers are the funnel's non-`Shards` branch (line 342) and the trivial
    early return above (line 450), and mg10's timed reconstruction reaches it
    through BOTH -- the recon entry projects a plain tensor, and the VCD loop
    projects a one-shard `Shards`.  Patching the class rather than the
    projector INSTANCE is what makes it survive the device-count settle that
    rebuilds the projector object inside the cold pass; the patch checks
    `self.model is` this arm's model and delegates otherwise, so it cannot
    reach anything else in the process.
    The patched body does exactly what the shipped one does, once per piece:
    the same `Projectors.sparse_forward_project_view_range`, the same
    `(0, num_views)` view range, the same `dev_index=0`, the same argument
    coercion -- handed `values[:, l0:l1]` instead of `values`.  Rows track
    slices one to one in parallel beam, so a piece's sinogram IS the matching
    row band, and the pieces concatenate along the row axis in slice order.
    Nothing is summed and nothing is reordered, which is the same assembly the
    two-device driver performs for a row-aligned geometry (line 540).
    ONE FIDELITY NOTE, measured rather than argued.  `values[:, l0:l1]` is a
    STRIDED view, and the Triton parallel forward body calls `.contiguous()` on
    its values argument (triton_parallel.py line 437), inside the per-call
    bracket.  So a narrowed piece pays a pack of its own block where the width
    1008 control pays none.  That is not a harness artifact: it is exactly what
    the two-device driver already pays on every band it projects from its OWN
    shard (`move_shard` to a tensor's own device returns the tensor itself, so
    half of the two-device fan-out's bands arrive strided and half arrive as
    fresh contiguous copies).  The pack is width-proportional, so it cannot
    bend the shape of the width curve, and every arm measures its own pack cost
    directly (`values_pack_ms`) so a reader can subtract it.

WHEN THE PATCHES GO IN, and why it is not where mg9 puts its probes.  mg9
installs its probes AFTER the discarded cold pass, because they live inside the
projector object and a device-count settle rebuilds that object.  The mg10
patches must go in BEFORE the cold pass, for a different and stronger reason:
they change the SHAPES the Triton kernels are launched at, and the launch key
in triton_cone.py includes the pixel count, the band length and the view count.
A patch installed after the cold pass would put every one of its first launches
-- and the compile lock they take -- inside the first timed reconstruction.
Both patches survive a projector rebuild (one is a plain attribute, the other
an instance method), and the arm re-verifies both before and after every timed
reconstruction anyway.

FOUR CORRECTNESS TRAPS carried over from mg9, all still live here.
    1. Object identity.  Every entry of `Projectors._fwd_body_per_dev` is the
       SAME object, because `maybe_compile` returns a body carrying
       `_mbirtorch_no_compile` unchanged.  Nothing here is keyed by `id()`:
       bodies are replaced BY POSITION with closures carrying their device
       index explicitly.
    2. Event mechanics.  A worker thread's current CUDA device is 0, so every
       event is created and recorded inside `with torch.cuda.device(dev)`, in
       the thread that issues the work.  Nothing synchronizes in a hot path;
       `elapsed_time` is read only after a full synchronize.
    3. Attribute pass-through.  The driver picks the view batch by reading
       `_view_batch_cost` OFF THE BODY, so the wrappers use `functools.wraps`
       and then assert the attribute survived.
    4. A silent no-op instrument.  Every arm asserts its own witnesses, and
       aborts rather than emit an empty row.

ONE THREADING DIFFERENCE mg9 did not have.  The cone prototype issues its
cross-device copies from the per-view-owner WORKER threads, which is where
mbirjax issues them and therefore what a library implementation would do.  The
banded path issues them from the loop thread.  So the copy recorder here takes
a lock, uncontended on the banded path and lightly contended on the prototype,
and the prototype's per-owner counters are per-index buckets with a single
writer each, needing none.  The consequence for the reading is recorded on
every prototype row: the prototype also moves copy ISSUE onto worker threads,
which is the memo's declined option A2, whose whole measured target is under
one second of a thirty second span.

VALUE, and what "exactly" can and cannot mean here.  Both forward kernels
accumulate with float atomics, which are commutative but not associative, so
neither kernel is bit-reproducible even between two runs of the SAME
configuration (triton_cone.py lines 66 to 72 state this and the kernel tests
measure the spread).  A patched arm therefore cannot be expected to match its
control bitwise, and an exact-equality test would fail for a reason that has
nothing to do with either remedy.  So every arm records a checksum for EACH
timed reconstruction and saves two strided samples of the reconstruction, and
the summary reports every cross-arm distance beside the same arm's own
pass-to-pass distance.  A cross-arm distance at the level of the run-to-run
floor is the strongest statement this instrument can make, and it is the
statement the design note's value gate should be priced on.

ARM CONSTRUCTION is mg9's, unchanged, so the spans stay comparable: the same
cell (1024, 1008, 992), the same phantom and shared md5-verified sinogram per
geometry, the same weights formula, the same seed, three VCD iterations, one
discarded cold pass, three timed reconstructions, and the SHIPPED forward view
chunk.  mg10 moves no chunk constant; it reads both and asserts the forward one
is still 128.

ARM ORDER is by decreasing value, because the rows are written incrementally
and a truncated job should lose the least.  The parallel two-device sweep is
first (the core of experiment 1), then the cone one-device value anchor (which
every cone value row needs), then the cone two-device batches (the core of
experiment 2), then cone at four, then the parallel four-device arms, which the
brief names as the first thing to trim.  Experiment 3's four arms are declared
LAST and are OPT-IN: they are not in the default run at all, so a plain re-run
of this file is the same eighteen arms it has always been, and the follow-up job
asks for them by name with
MG10_ARMS=p1_shipped,p1_band0504,p1_band0252,p1_band0063.

Run:
    <torch python> mg10_shape_sweep.py        on a 4-GPU node (mg10_gautschi.sbatch)
    python mg10_shape_sweep.py --dry-run      anywhere: print the arm plan
    MG10_SMOKE=1 python mg10_shape_sweep.py   the local CPU smoke
    python mg10_shape_sweep.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas):
    P0_TORCH_PYTHON=<python>       interpreter for the arm subprocesses
    MG10_RESULTS=<dir>             where the jsonl and the artifacts go
    MG10_ARMS=p2_shard,p2_128,...  subset of the arms, by token
    MG10_PARALLEL_BANDS=64,128,192,256,384    the swept sub-band lengths
    MG10_CONE_BATCHES=2048,4096,8192          the swept column-batch sizes
    MG10_PARALLEL_N1_WIDTHS=504,252,63        experiment 3's one-device
                                   values-block widths.  The tokens follow the
                                   widths (504 -> p1_band0504), so overriding
                                   this renames the arms MG10_ARMS must ask for.
    MG10_PARALLEL_VIEW_CHUNKS=8,16            experiment 4's forward view
                                   chunks.  The tokens follow these too
                                   (8 -> p1_vc8 and p2_vc8).
    MG10_ITERATIONS=3              VCD iterations per reconstruction
    MG10_WARM_REPEATS=3            timed reconstructions after the cold pass
    MG10_MAX_EVENT_PAIRS=400000    per-reconstruction event budget
    MG10_KEEP_ARTIFACTS=1          keep the sinograms and value samples
    MG10_SMOKE=1                   the local CPU smoke (tiny cell, few iters)
    MG10_DEVICE=cpu                smoke device
"""

import functools
import json
import hashlib
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# mg9's cell, and nothing else.  cell = (num_views, num_det_rows,
# num_det_channels); at this cell both geometries give recon (992, 992, 1008),
# so the slice count is 1008 and it divides 1, 2 and 4 exactly -- no padding.
CELL = (1024, 1008, 992)

SMOKE = os.environ.get("MG10_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG10_DEVICE", "cpu" if SMOKE else "cuda")


def _int_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    out = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out or list(default)


# Experiment 1's swept sub-band lengths, and experiment 2's swept column-batch
# sizes.  All three are env-overridable so a follow-up can extend the sweep
# without editing this file (the sweep's own result may well say the
# interesting region is outside these).  The four-device parallel list is
# separate and shorter because the shard is only 252 slices there, so most of
# the two-device lengths would be capped back onto the control.
PARALLEL_BANDS = _int_list("MG10_PARALLEL_BANDS",
                           (4,) if SMOKE else (64, 128, 192, 256, 384))
PARALLEL_BANDS_N4 = _int_list("MG10_PARALLEL_BANDS_N4",
                              (4,) if SMOKE else (128, 256))
CONE_BATCHES = _int_list("MG10_CONE_BATCHES",
                         (64,) if SMOKE else (2048, 4096, 8192))

# Experiment 3's ONE-DEVICE values-block widths, and their smoke stand-ins.
# The production widths are the two-device sweep's widths that are also
# divisors of the whole 1008-slice volume (504 and 252) plus the width that won
# there (63), so every one of them lines up with a measured n > 1 point.  The
# smoke volume is 24 slices, so the smoke substitutes widths that fit it; the
# TOKENS stay named after the production widths either way, because MG10_ARMS
# has to name the same arms on the cluster and on this Mac.
PARALLEL_N1_WIDTHS = _int_list("MG10_PARALLEL_N1_WIDTHS", (504, 252, 63))
PARALLEL_N1_SMOKE_WIDTHS = (12, 6, 3)

# Experiment 4's forward view chunks: the OTHER axis of one launch's atomic
# write slab.  The slab is (view batch x channels x values columns) x 4 B, so
# 8 views at the full 1008-slice width covers the same bytes as 128 views at
# 63 slices -- the width that won at two devices.  These run at both device
# counts, at the SHIPPED band width, so nothing but the view axis moves.
PARALLEL_VIEW_CHUNKS = _int_list("MG10_PARALLEL_VIEW_CHUNKS", (8, 16))

VCD_ITERATIONS = int(os.environ.get("MG10_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # mg1's / mg5's / mg9's seed, so the arms stay comparable
WARM_REPEATS = max(1, int(os.environ.get("MG10_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# The shipped forward view chunk, read and asserted, never set.
SHIPPED_CHUNK = 128

# mg9's measured anchors (findings 1.7, job 15152345 on h018, tree f985a6e).
# The control arms must reproduce these; a large disagreement means the node or
# the tree moved and the whole sweep is suspect.  Keys are (geometry, n_dev).
MG9_ANCHOR = {
    ("parallel", 2): dict(bracket_s=28.77, busy_s=28.24, calls=680,
                          per_launch_ms=41.5, composed_s=39.28, peak_gb=12.48),
    ("parallel", 4): dict(bracket_s=14.88, busy_s=14.55, calls=680,
                          per_launch_ms=21.4, composed_s=23.48, peak_gb=7.31),
    ("cone", 2): dict(bracket_s=30.65, busy_s=29.70, calls=None,
                      per_launch_ms=None, composed_s=67.30, peak_gb=14.31),
}
# EXPERIMENT 3's REFERENCE COLUMN: the parallel per-launch and per-slice
# readings this job has ALREADY taken, so the discriminator table can print the
# n > 1 side beside the new n = 1 side even when only the n = 1 arms are run.
# Keyed by (device count, the values-block width the kernel was handed).  The
# n = 2 and n = 4 entries are mg10's own first run (h004, 2026-08-10, rows
# mg10_shape_sweep_h004_20260810_174925.jsonl), read off its summary series.
# The n = 1 entry at width 1008 is mg9's (finding 1.7, job 15152345 on h018),
# and it is the one number in this table that the new p1_shipped control
# re-measures rather than merely quotes -- if the control disagrees with it, the
# node or the tree moved and the whole discrimination is suspect.
PARALLEL_PER_SLICE_REFERENCE = {
    (1, 1008): dict(per_launch_ms=None, ms_per_slice=0.0412, source="mg9 1.7"),
    (2, 504): dict(per_launch_ms=41.515, ms_per_slice=0.08237, source="mg10 p2_shard"),
    (2, 252): dict(per_launch_ms=21.213, ms_per_slice=0.08418, source="mg10 p2_256"),
    (2, 168): dict(per_launch_ms=17.009, ms_per_slice=0.10124, source="mg10 p2_192"),
    (2, 126): dict(per_launch_ms=10.990, ms_per_slice=0.08722, source="mg10 p2_128"),
    (2, 63): dict(per_launch_ms=4.697, ms_per_slice=0.07456, source="mg10 p2_64"),
    (4, 252): dict(per_launch_ms=21.399, ms_per_slice=0.08492, source="mg10 p4_shard"),
    (4, 126): dict(per_launch_ms=11.223, ms_per_slice=0.08907, source="mg10 p4_128"),
}
# The two readings the discrimination rule is stated against, and nothing else:
# the one-device width-1008 cost per slice, and the cost per slice the n > 1
# sweep held at every width it tried.
N1_WIDE_MS_PER_SLICE = 0.0412
NGT1_MS_PER_SLICE = 0.084
# Width 63 was the only two-device sub-band that beat its control, by this much.
N2_WIDTH63_WIN_FRAC = 0.095

# From mg5 / findings 1.5, for the arms mg9 did not run.
MEMO_FORWARD_SPAN_S = {("cone", 1): 32.18, ("cone", 2): 30.61, ("cone", 4): 30.48,
                       ("parallel", 1): 28.87, ("parallel", 2): 28.75}
MEMO_COMPOSED_S = {("cone", 1): 61.57, ("cone", 2): 67.23,
                   ("parallel", 1): 40.00, ("parallel", 2): 39.40}

# Reconciliation tolerance, mg1's constant.
RECONCILE_SLACK = 0.02

RESULTS_DIR = os.environ.get(
    "MG10_RESULTS", os.path.dirname(os.path.abspath(__file__)))
RUN_LABEL = platform.node().split(".")[0]

SPEC = {
    "parallel": dict(kernel_module="mbirtorch.triton_parallel",
                     fwd_chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                     back_chunk_const="PARALLEL_BACK_VIEW_CHUNK"),
    "cone": dict(kernel_module="mbirtorch.triton_cone",
                 fwd_chunk_const="CONE_FWD_VIEW_CHUNK",
                 back_chunk_const="CONE_BACK_VIEW_CHUNK"),
}

# The value sample: one reconstruction voxel out of every VALUE_SAMPLE_TARGET
# per axis, so the sample is a few hundred thousand values whatever the cell.
VALUE_SAMPLE_TARGET = 64

# ── the region definitions and GPU-health machinery, COPIED from mg9 ──────────
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
NESTED_REGIONS = ("band_reduce",)
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
MAX_EVENT_PAIRS = int(os.environ.get("MG10_MAX_EVENT_PAIRS", "400000"))

HOT_CORE_C = 85
HOT_HBM_C = 95
CLOCK_DEPRESSED_FRAC = 0.90
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")
# ──────────────────────────────────────────────────────────────────────────────


# ── the driver's band tiling, re-derived here ─────────────────────────────────
# Deliberately a SECOND implementation of TomographyModel._balanced_slice_bounds
# and _slice_band_length rather than an import of them.  The witness these feed
# has to be able to disagree with the library; a check written by calling the
# thing it checks cannot fail.
def balanced_slice_bounds(extent, band_len):
    """Tile [0, extent) into the fewest bands no longer than ``band_len``, with
    lengths as equal as possible.  The driver's rule (tomography_model.py lines
    391 to 404), re-derived."""
    num_bands = -(-extent // band_len)
    base, rem = divmod(extent, num_bands)
    bounds, start = [], 0
    for k in range(num_bands):
        length = base + (1 if k < rem else 0)
        bounds.append((start, start + length))
        start += length
    return bounds


def realized_band_lengths(shard_len, requested):
    """The sub-band lengths one slice-owner's shard is actually walked in.

    ``requested`` is what an arm asked for, or None for the shipped default
    (the whole shard).  The driver caps the request at the shard length
    (`_slice_band_length` returns min(b, slices_per_dev)), so a request LARGER
    than the shard is the control by another name, and two different requests
    can land on the identical walk -- the balanced tiling snaps to a divisor.
    Both facts are surfaced in the plan and on the row rather than hidden."""
    band = min(int(requested), shard_len) if requested else shard_len
    return [b - a for a, b in balanced_slice_bounds(shard_len, band)]


# ── the arm plan ──────────────────────────────────────────────────────────────
def build_arms():
    """Every arm, in job order (least valuable last -- rows write
    incrementally).  Each entry is a dict describing one subprocess run."""
    arms = []

    def add(token, geometry, n_dev, band=None, batch=None, width=None,
            chunk=None, role="control"):
        arms.append(dict(token=token, geometry=geometry, n_dev=n_dev,
                         band_len=band, pixel_batch=batch, values_width=width,
                         view_chunk=chunk, role=role))

    # Experiment 1's core: parallel at two devices.  Control first, so its
    # reproduction of mg9's anchor is known before any patched arm is read.
    add("p2_shard", "parallel", 2)
    for length in PARALLEL_BANDS:
        add(f"p2_{length}", "parallel", 2, band=length, role="subband")
    # Experiment 2's value anchor.  One device, unpatched: the whole cylinder
    # is already local and the shipped single-device path makes exactly the
    # full-height calls the prototype is trying to reproduce at n > 1.
    add("c1", "cone", 1, role="anchor")
    # Experiment 2's core: cone at two devices.
    add("c2_banded", "cone", 2)
    for batch in CONE_BATCHES:
        add(f"c2_{batch}", "cone", 2, batch=batch, role="columns")
    add("c4_banded", "cone", 4)
    for batch in CONE_BATCHES:
        add(f"c4_{batch}", "cone", 4, batch=batch, role="columns")
    # The first thing to trim, per the brief.
    add("p4_shard", "parallel", 4)
    for length in PARALLEL_BANDS_N4:
        add(f"p4_{length}", "parallel", 4, band=length, role="subband")
    # EXPERIMENT 3, declared last and NOT in the default run: `selected_arms`
    # drops every one of these unless MG10_ARMS names it, so a plain re-run of
    # this file is the eighteen arms it has always been.  Control first, so its
    # reproduction of mg9's one-device anchor is known before any narrowed arm
    # is read -- the same ordering rule experiment 1 follows.
    add("p1_shipped", "parallel", 1, role="anchor")
    widths = list(PARALLEL_N1_WIDTHS)
    if SMOKE:
        # The smoke volume is 24 slices, so the production widths would all cap
        # back onto the whole volume and measure the control four times.  The
        # tokens keep the production names (see PARALLEL_N1_SMOKE_WIDTHS).
        widths = list(PARALLEL_N1_SMOKE_WIDTHS)[:len(PARALLEL_N1_WIDTHS)]
    for nominal, width in zip(PARALLEL_N1_WIDTHS, widths):
        add(f"p1_band{nominal:04d}", "parallel", 1, width=width,
            role="values_split")
    # EXPERIMENT 4, also opt-in: the same write slab from the VIEW axis, at
    # both device counts and at the shipped band width.  The one-device arms
    # read against p1_shipped and the two-device arms against p2_shard, both of
    # which must be in the same job for the comparison to be same-run.
    for chunk in PARALLEL_VIEW_CHUNKS:
        add(f"p1_vc{chunk}", "parallel", 1, chunk=chunk, role="view_chunk")
    for chunk in PARALLEL_VIEW_CHUNKS:
        add(f"p2_vc{chunk}", "parallel", 2, chunk=chunk, role="view_chunk")
    return arms


# The tokens experiments 3 and 4 add, which are OPT-IN: `selected_arms` keeps
# them out of an unnarrowed run.  Declared here rather than derived at the use
# site so the exclusion and the arm list cannot drift apart.  p2_shard is NOT
# in this set -- it is one of the original eighteen, and the two-device view
# chunk arms read against it, so the narrowed job names it too.
OPT_IN_TOKENS = frozenset(
    ["p1_shipped"]
    + [f"p1_band{w:04d}" for w in PARALLEL_N1_WIDTHS]
    + [f"p1_vc{c}" for c in PARALLEL_VIEW_CHUNKS]
    + [f"p2_vc{c}" for c in PARALLEL_VIEW_CHUNKS])


ARMS = build_arms()


def selected_arms():
    """The arms to run, narrowed by MG10_ARMS (comma-separated tokens).

    Experiment 3's arms are OPT-IN.  An unnarrowed run drops them, so this
    file's default job is exactly the eighteen arms mg10 has always run and a
    plain resubmit of the batch file cannot quietly grow four arms and a
    wall.  Naming one in MG10_ARMS is the only way to run it."""
    by_token = {a["token"]: a for a in ARMS}
    raw = os.environ.get("MG10_ARMS", "").strip()
    if not raw:
        chosen = [a for a in ARMS if a["token"] not in OPT_IN_TOKENS]
    else:
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in by_token:
                raise ValueError(f"MG10_ARMS: {token!r} is not one of "
                                 f"{sorted(by_token)}")
            wanted.add(token)
        if not wanted:
            raise ValueError(f"MG10_ARMS: no valid tokens in {raw!r}")
        chosen = [a for a in ARMS if a["token"] in wanted]   # declared order
    if SMOKE and not os.environ.get("MG10_ARMS", "").strip():
        # The device-count pin is a CUDA-only mechanism, so on CPU an n > 1 arm
        # pins by device LIST instead and says so on its row.  The smoke keeps
        # one arm of every KIND -- control, sub-band, anchor, prototype,
        # one-device control and one-device split -- so every patch seam,
        # witness and value column is exercised, and drops the four-device
        # arms, which add nothing the two-device ones do not.  The experiment-3
        # arms are opt-in in a real run but kept here on purpose: the smoke is
        # the only place their seam and their two witnesses get exercised
        # before the cluster sees them.
        keep = {"p2_shard", f"p2_{PARALLEL_BANDS[0]}", "c1", "c2_banded",
                f"c2_{CONE_BATCHES[0]}", "p1_shipped",
                f"p1_band{PARALLEL_N1_WIDTHS[0]:04d}",
                f"p1_vc{PARALLEL_VIEW_CHUNKS[0]}",
                f"p2_vc{PARALLEL_VIEW_CHUNKS[0]}"}
        chosen = [a for a in ARMS if a["token"] in keep]
    return chosen


def cell_for(_geometry):
    return SMOKE_CELL if SMOKE else CELL


def num_slices_for(cell):
    """The reconstruction's slice count at this cell.  Both geometries derive
    it from the detector row count (measured: cell (1024, 1008, 992) gives
    recon (992, 992, 1008) for parallel AND for cone), and the worker asserts
    this against the model's own recon_shape before it is used for anything."""
    return int(cell[1])


# ── staged-artifact mechanics (mg5's / mg9's md5 discipline) ──────────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg10_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id, index):
    return os.path.join(RESULTS_DIR, f"_mg10_sample_{arm_id}_p{index}.npy")


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
    """The ONE host exit.  ``Shards.gather()`` already returns numpy -- the
    recorded failure class is re-detaching that result."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    gather = getattr(x, "gather", None)
    if callable(gather) and hasattr(x, "placement"):
        return gather()
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weights(sinogram):
    """One weighting formula, one dtype, every arm (mg1's rule)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks, temperatures and active throttle reasons."""
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


def worst_health(samples):
    """Per-GPU worst case across samples: MIN clocks, MAX temps, union of
    throttle reasons."""
    agg = {}
    for snapshot in samples:
        for gpu in snapshot:
            index = gpu.get("index")
            slot = agg.setdefault(index, {"index": index, "sm_mhz": None,
                                          "mem_mhz": None, "temp_c": None,
                                          "mem_temp_c": None, "throttle": []})
            for key in ("sm_mhz", "mem_mhz"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else min(slot[key],
                                                                    value)
            for key in ("temp_c", "mem_temp_c"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else max(slot[key],
                                                                    value)
            for reason in gpu.get("throttle", []):
                if reason not in slot["throttle"]:
                    slot["throttle"].append(reason)
    return [agg[k] for k in sorted(agg, key=lambda i: (i is None, i))]


def row_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── THE EXISTING BRACKET: mg1's region instrument, copied through mg9 ─────────
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    reconstruction loop's calling thread.  Copied from mg9_fwd_instrument.py,
    which copied it from mg1_readout.py without change, so mg10's forward
    bracket IS mg5's and mg9's forward bracket.

    CUDA path: for each device in the region's placement a start and an end
    event are CREATED AND RECORDED inside ``with torch.cuda.device(dev)``, on
    that device's current stream.  The end event is recorded AFTER the call
    returns, so it queues behind everything the call enqueued.  Elapsed times
    are read only in :meth:`finish`, after a per-device synchronize.

    CPU path (the local smoke ONLY): perf_counter walls stand in behind the
    same interface.  Two smoke artifacts follow from virtual cpu devices
    sharing one name: the per-device map collapses to a single ``'cpu'`` key,
    and its span sum is the host wall times the device count.
    """

    def __init__(self, torch_module, cuda):
        self.torch = torch_module
        self.cuda = cuda
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.devices_seen = {region: [] for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"

    def reset(self):
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False

    def _start(self, region, devices):
        for device in devices:
            name = str(device)
            if name not in self.devices_seen[region]:
                self.devices_seen[region].append(name)
        if not self.cuda:
            return None
        if self.pair_count + len(devices) > MAX_EVENT_PAIRS:
            self.cap_hit = True
            return None
        events = []
        for device in devices:
            with self.torch.cuda.device(device):
                start = self.torch.cuda.Event(enable_timing=True)
                start.record()
            events.append((device, start))
        return events

    def _stop(self, region, events):
        if events is None:
            return
        for device, start in events:
            with self.torch.cuda.device(device):
                end = self.torch.cuda.Event(enable_timing=True)
                end.record()
            self._pairs[region].setdefault(str(device), []).append((start, end))
            self.pair_count += 1

    def wrap(self, region, resolve_devices, func):
        def wrapped(*args, **kwargs):
            devices = resolve_devices(*args, **kwargs)
            events = self._start(region, devices)
            host0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                self._stop(region, events)
                self.calls[region] += 1
                self.host_wall[region] += host
                if not self.cuda:
                    for device in devices:
                        self._cpu_spans[region].setdefault(
                            str(device), []).append(host * 1e3)
        return wrapped

    def finish(self, devices):
        """Per-device synchronize, THEN read the spans (never inside the loop)."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        record = {}
        for region in REGIONS:
            per_device = {}
            if self.cuda:
                for name, pairs in self._pairs[region].items():
                    per_device[name] = sum(s.elapsed_time(e) for s, e in pairs)
            else:
                for name, spans in self._cpu_spans[region].items():
                    per_device[name] = float(sum(spans))
            record[region] = dict(
                calls=self.calls[region],
                host_wall_s=self.host_wall[region],
                device_span_ms=per_device,
                device_span_max_ms=(max(per_device.values())
                                    if per_device else 0.0),
                device_span_sum_ms=float(sum(per_device.values())),
                devices=self.devices_seen[region])
        return dict(regions=record, event_backend=self.backend,
                    event_pairs=self.pair_count, event_cap_hit=self.cap_hit)


def attach_region_instrument(model, torch_module, cuda):
    """Wrap the five regions on THIS model instance (mg1's attach_instrument).
    Nothing in the mbirtorch package is edited: the funnels are shadowed as
    instance attributes and the sharding seams as module attributes, all of
    which the engine looks up at call time."""
    from mbirtorch import _sharding

    instrument = RegionInstrument(torch_module, cuda)

    model.sparse_forward_project = instrument.wrap(
        "forward_funnel",
        lambda *a, **k: list(model.sino_placement.devices),
        model.sparse_forward_project)
    model.sparse_back_project = instrument.wrap(
        "back_funnel",
        lambda *a, **k: list(model.recon_placement.devices),
        model.sparse_back_project)

    original_halos = _sharding.exchange_qggmrf_halos
    _sharding.exchange_qggmrf_halos = instrument.wrap(
        "halo",
        lambda shards, *a, **k: list(shards.placement.devices),
        original_halos)

    original_reduce = _sharding.sum_band_to_owner
    _sharding.sum_band_to_owner = instrument.wrap(
        "band_reduce",
        lambda partials, owner, *a, **k: [owner],
        original_reduce)

    original_run = _sharding.run_per_device
    wrapped_prior = instrument.wrap(
        "prior",
        lambda devices, *a, **k: list(devices),
        original_run)

    def run_per_device(devices, worker_fn, executor=None):
        if getattr(worker_fn, "__name__", "") == "prior_worker":
            return wrapped_prior(devices, worker_fn, executor=executor)
        return original_run(devices, worker_fn, executor=executor)

    _sharding.run_per_device = run_per_device

    def detach():
        _sharding.exchange_qggmrf_halos = original_halos
        _sharding.sum_band_to_owner = original_reduce
        _sharding.run_per_device = original_run

    return instrument, detach


def _concrete_device(torch_module, device, cuda):
    """A device carrying a concrete index, for recording events on and for
    comparing against a tensor's device.  A model that resolved its device
    automatically holds ``torch.device('cuda')`` with NO index, while a
    tensor's ``.device`` always carries one."""
    if not cuda or getattr(device, "type", None) != "cuda":
        return device
    if device.index is None:
        return torch_module.device("cuda", torch_module.cuda.current_device())
    return device


# ── INSTRUMENT 1: busy time, per device, per body call (mg9's, unchanged) ─────
class BusyProbe:
    """Times each individual forward projection BODY call, in buckets keyed by
    DEVICE INDEX -- never by object identity, because every entry of
    ``_fwd_body_per_dev`` is the same object.

    Busy divided by the call count is the PER-LAUNCH time, which is the number
    both of mg10's experiments turn on.

    Threading.  Body calls arrive on the per-device worker threads of
    ``run_per_device``, which runs at most one thread per device index at a
    time and waits for all of them before the next fan-out, so each bucket has
    a single writer and needs no lock.  That holds for the cone prototype too:
    it fans out once per view-owner and each worker touches only its own index.

    Events are created and recorded inside ``with torch.cuda.device(dev)``; the
    body itself is called OUTSIDE that context, so it runs in exactly the
    device context it would without the instrument.
    """

    def __init__(self, torch_module, cuda, n_dev, pairs_per_device):
        self.torch = torch_module
        self.cuda = cuda
        self.n_dev = n_dev
        self.pairs_per_device = pairs_per_device
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"
        self._pairs = [[] for _ in range(n_dev)]
        self._cpu_ms = [0.0] * n_dev
        self.calls = [0] * n_dev
        self.host_s = [0.0] * n_dev
        self.cap_hit = [False] * n_dev
        # Positive witness for the positional key: a forward body's first
        # tensor argument is the values block, which lives on the device that
        # position is supposed to name.
        self.device_mismatch = [0] * n_dev
        # The COLUMN COUNT and the PIXEL COUNT of every values block a body was
        # handed, per device.  For the parallel sweep the column count is the
        # realized sub-band width seen from inside the kernel call, which is an
        # independent second reading of the band the driver built; for the cone
        # prototype it is the gathered cylinder's height and must equal the full
        # slice count.  The pixel count is the other axis of the same block, and
        # it is what the cone batch sweep normalizes by.  Both feed the
        # per-slice and per-pixel cost columns, which are what make a NULL
        # result -- no shape beating the shipped one -- legible at a glance.
        self.cols_hist = [{} for _ in range(n_dev)]
        self.pixels_hist = [{} for _ in range(n_dev)]

    def wrap(self, body, dev_index, device):
        """Return ``body`` bracketed, carrying its device index EXPLICITLY.

        ``functools.wraps`` copies the wrapped function's ``__dict__``, which is
        where a kernel body keeps ``_view_batch_cost`` and
        ``_mbirtorch_no_compile``; the caller asserts both survived."""
        torch_module, cuda = self.torch, self.cuda
        device = _concrete_device(torch_module, device, cuda)

        @functools.wraps(body)
        def wrapped(*args, **kwargs):
            if args and torch_module.is_tensor(args[0]):
                if args[0].device != device:
                    self.device_mismatch[dev_index] += 1
                cols = int(args[0].shape[-1])
                bucket = self.cols_hist[dev_index]
                bucket[cols] = bucket.get(cols, 0) + 1
                pixels = int(args[0].shape[0])
                bucket = self.pixels_hist[dev_index]
                bucket[pixels] = bucket.get(pixels, 0) + 1
            if not cuda:
                host0 = time.perf_counter()
                try:
                    return body(*args, **kwargs)
                finally:
                    host = time.perf_counter() - host0
                    self.calls[dev_index] += 1
                    self.host_s[dev_index] += host
                    self._cpu_ms[dev_index] += host * 1e3
            budget = len(self._pairs[dev_index]) < self.pairs_per_device
            start = None
            if budget:
                with torch_module.cuda.device(device):
                    start = torch_module.cuda.Event(enable_timing=True)
                    start.record()
            else:
                self.cap_hit[dev_index] = True
            host0 = time.perf_counter()
            try:
                return body(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                if start is not None:
                    with torch_module.cuda.device(device):
                        end = torch_module.cuda.Event(enable_timing=True)
                        end.record()
                    self._pairs[dev_index].append((start, end))
                self.calls[dev_index] += 1
                self.host_s[dev_index] += host

        wrapped._mg10_dev_index = dev_index
        wrapped._mg10_wrapped_body = body
        return wrapped

    def drain(self, devices):
        """Read the elapsed times and reset, ONCE PER TIMED RECONSTRUCTION.
        Every device is synchronized first, so no ``elapsed_time`` is read on
        an event the stream has not reached."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        busy_ms, calls, host_s, mismatch = [], [], [], []
        cols, pixels = [], []
        for i in range(self.n_dev):
            if self.cuda:
                busy_ms.append(float(sum(s.elapsed_time(e)
                                         for s, e in self._pairs[i])))
            else:
                busy_ms.append(float(self._cpu_ms[i]))
            calls.append(int(self.calls[i]))
            host_s.append(float(self.host_s[i]))
            mismatch.append(int(self.device_mismatch[i]))
            cols.append({str(k): v for k, v in sorted(self.cols_hist[i].items())})
            pixels.append({str(k): v for k, v
                           in sorted(self.pixels_hist[i].items())})
        record = dict(busy_ms_per_device=busy_ms,
                      busy_calls_per_device=calls,
                      busy_host_s_per_device=host_s,
                      busy_device_mismatch_per_device=mismatch,
                      busy_value_cols_per_device=cols,
                      busy_value_pixels_per_device=pixels,
                      busy_cap_hit=any(self.cap_hit))
        self._pairs = [[] for _ in range(self.n_dev)]
        self._cpu_ms = [0.0] * self.n_dev
        self.calls = [0] * self.n_dev
        self.host_s = [0.0] * self.n_dev
        self.device_mismatch = [0] * self.n_dev
        self.cols_hist = [{} for _ in range(self.n_dev)]
        self.pixels_hist = [{} for _ in range(self.n_dev)]
        self.cap_hit = [False] * self.n_dev
        return record


# ── INSTRUMENT 2: the cross-device copies, both shapes ────────────────────────
class TransferProbe:
    """Times the cross-device copies the forward makes, in both shapes.

    Shape one, the SHIPPED band broadcast: host wall around each
    ``_sharding.broadcast_band_to_views`` call, plus the width of the band it
    fanned out (the sub-band witness for experiment 1), plus a device event
    pair around each individual copy.
    Shape two, the PROTOTYPE's column gather: host wall around each column
    batch's set of moves, plus the same per-copy event pairs.

    The copies are timed with event pairs recorded on the SOURCE device's
    current stream, which is the stream torch enqueues a cross-device copy on:
    ``move_shard``'s direct path is ``x.to(target)``, and torch runs that on
    the source's stream and makes the destination's stream wait on a barrier.
    Events on the destination's stream would have measured the barrier wait and
    read as ~0, which is exactly what a wrong-stream bracket looks like, so the
    summary flags a ~0 reading rather than reporting it as a fast copy.
    The determination holds for the DIRECT path; when ``dev2dev_safe`` is False
    the primitive bounces through host memory and the source-stream bracket
    covers only the device-to-host half, which ``copy_measurement`` says.

    The original primitives are CALLED, never reimplemented: the per-copy
    bracket is installed on ``move_shard``, which both the shipped fan-out and
    the prototype resolve as a module global at call time.  A thread-local flag
    keeps the bracket confined to copies issued from inside one of the two
    shapes, so the other users of ``move_shard`` -- the halo exchange, the band
    reduce, the view parameter placement -- run untouched and untimed.

    THE LOCK.  mg9 needed none: its broadcast ran on the loop thread alone.
    The prototype issues its gathers from the per-view-owner worker threads
    (which is where mbirjax issues them), so the shared counters take a lock.
    It is uncontended on the banded path.
    """

    def __init__(self, torch_module, cuda, max_pairs):
        self.torch = torch_module
        self.cuda = cuda
        self.max_pairs = max_pairs
        self.lock = threading.Lock()
        # the shipped band fan-out
        self.broadcast_calls = 0
        self.broadcast_host_s = 0.0
        self.band_cols = {}           # band width -> how many fan-outs
        # the prototype's column gather
        self.gather_calls = 0
        self.gather_host_s = 0.0
        # both
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        self._pairs = []              # (src name, dst name, start, end)
        self.copy_measurement = None

    def drain(self, devices):
        """Read the copy spans and reset, once per timed reconstruction."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        by_src, by_dst, total = {}, {}, 0.0
        for src, dst, start, end in self._pairs:
            span = float(start.elapsed_time(end))
            by_src[src] = by_src.get(src, 0.0) + span
            by_dst[dst] = by_dst.get(dst, 0.0) + span
            total += span
        record = dict(broadcast_calls=int(self.broadcast_calls),
                      broadcast_host_wall_s=float(self.broadcast_host_s),
                      band_cols_hist={str(k): v for k, v
                                      in sorted(self.band_cols.items())},
                      gather_calls=int(self.gather_calls),
                      gather_host_wall_s=float(self.gather_host_s),
                      copy_count=int(self.copy_count),
                      copy_noop_count=int(self.copy_noop_count),
                      copy_bytes=int(self.copy_bytes),
                      copy_device_ms_total=total,
                      copy_device_ms_by_src=by_src,
                      copy_device_ms_by_dst=by_dst,
                      copy_cap_hit=bool(self.cap_hit),
                      copy_measurement=self.copy_measurement)
        self.broadcast_calls = 0
        self.broadcast_host_s = 0.0
        self.band_cols = {}
        self.gather_calls = 0
        self.gather_host_s = 0.0
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        self._pairs = []
        return record


def attach_forward_probes(model, torch_module, cuda, max_pairs,
                          prototype_active=False):
    """Install the busy and transfer instruments; return
    ``(busy, transfer, verify, detach, observed)``.

    Call this AFTER the discarded cold pass.  The body wrappers live inside the
    projector object, and a device-count settle during the first reconstruction
    rebuilds that object (``_install_device_layout`` -> ``create_projectors``),
    which would throw them away.  ``verify()`` re-checks, before and after every
    timed reconstruction, that the projector object and the wrapped list are
    still the ones the driver will call."""
    from mbirtorch import _sharding

    pf = model.projector_functions
    devices = list(model.sino_placement.devices)
    n_dev = len(devices)
    bodies = pf._fwd_body_per_dev
    if len(bodies) != n_dev:
        raise RuntimeError(
            f"the projector holds {len(bodies)} forward bodies for {n_dev} "
            f"devices; the positional device key would be wrong")

    busy = BusyProbe(torch_module, cuda, n_dev,
                     max(1, max_pairs // max(1, 2 * n_dev)))
    originals = list(bodies)
    wrappers = []
    for index, body in enumerate(originals):
        wrapper = busy.wrap(body, index, devices[index])
        # Trap 3: the driver chooses the view batch by reading this attribute
        # OFF THE BODY, so a wrapper that lost it would silently move the
        # kernel onto the torch batching rule.
        if getattr(body, "_view_batch_cost", None) is not \
                getattr(wrapper, "_view_batch_cost", None):
            raise RuntimeError(
                "the body wrapper did not carry _view_batch_cost through; the "
                "realized view batch would change and this arm would not be "
                "comparable with mg9's")
        wrappers.append(wrapper)
    # Mutated IN PLACE rather than rebound, so any other reference to the list
    # sees the wrappers too.
    for index, wrapper in enumerate(wrappers):
        bodies[index] = wrapper

    # The realized view batch, observed PER DEVICE by the positional key.  This
    # matters more in mg10 than it did in mg9: the cone prototype hands the
    # driver a far smaller pixel count per call, which RAISES the batch the
    # transient budget allows, and that is a real difference between the two
    # cone shapes rather than an instrument artifact.
    observed = {}
    observed_lock = threading.Lock()
    original_effective = pf._effective_view_batch

    def effective_view_batch(body, num_pixels, band_cols, args):
        value = original_effective(body, num_pixels, band_cols, args)
        index = getattr(body, "_mg10_dev_index", None)
        key = (f"fwd_dev{index}" if index is not None
               else "back_body_device_not_recoverable")
        with observed_lock:
            bucket = observed.setdefault(key, {})
            bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = effective_view_batch

    # -- the copies -----------------------------------------------------------
    transfer = TransferProbe(torch_module, cuda, max_pairs)
    if not cuda:
        transfer.copy_measurement = (
            "host_wall_only: the CPU smoke has no CUDA events, and its virtual "
            "devices make every copy a no-op")
    elif getattr(model, "dev2dev_safe", True):
        transfer.copy_measurement = (
            "device_events_on_source_stream: move_shard's direct path is "
            "x.to(dst), and torch runs a cross-device copy on the SOURCE "
            "device's current stream (the destination's stream only waits on a "
            "barrier), so each copy is bracketed on the source device")
        if prototype_active:
            # The banded fan-out issues its copies from one thread, so its
            # per-copy brackets never overlap.  The prototype issues them from
            # every view-owner's worker thread, and two workers pulling from
            # the SAME slice-owner enqueue onto that one source stream at the
            # same time, so their brackets can cover each other's copies.  The
            # per-copy device total is therefore an upper bound on a prototype
            # arm rather than a sum of disjoint intervals.  The byte total and
            # the gather host wall carry no such caveat, and the fix -- holding
            # the lock across the copy -- is refused because it would serialize
            # the very issue pattern the prototype exists to measure.
            transfer.copy_measurement += (
                "; UPPER BOUND on this arm: several view-owners copy from one "
                "slice-owner concurrently, so per-copy brackets on that "
                "source's stream can overlap.  Read copy_bytes and the gather "
                "host wall as exact and copy_device_s as a bound")
    else:
        transfer.copy_measurement = (
            "host_wall_only: dev2dev_safe is False, so move_shard routes "
            "through host memory and a source-stream bracket would cover only "
            "the device-to-host half of each copy; the device numbers below "
            "are partial and must not be read as copy time")

    state = threading.local()
    original_move = _sharding.move_shard
    original_broadcast = _sharding.broadcast_band_to_views

    def move_shard(x, target, dev2dev_safe=True):
        if not getattr(state, "timing", False):
            return original_move(x, target, dev2dev_safe=dev2dev_safe)
        with transfer.lock:
            timed = cuda and len(transfer._pairs) < transfer.max_pairs
            if cuda and not timed:
                transfer.cap_hit = True
        start = end = None
        source = x.device
        if timed:
            with torch_module.cuda.device(source):
                start = torch_module.cuda.Event(enable_timing=True)
                start.record()
        out = original_move(x, target, dev2dev_safe=dev2dev_safe)
        if timed:
            with torch_module.cuda.device(source):
                end = torch_module.cuda.Event(enable_timing=True)
                end.record()
        # A copy to the tensor's own device returns the tensor itself, so one
        # move per fan-out (and one per column batch) is free.  Identity is the
        # exact test.  A free copy is counted but kept OUT of the spans and the
        # bytes, so the copy columns mean what actually landed on a device.
        noop = out is x
        with transfer.lock:
            transfer.copy_count += 1
            if noop:
                transfer.copy_noop_count += 1
            else:
                transfer.copy_bytes += int(x.numel()) * int(x.element_size())
                if start is not None:
                    transfer._pairs.append((str(source), str(target), start, end))
        return out

    def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
        state.timing = True
        host0 = time.perf_counter()
        try:
            return original_broadcast(band, view_owners,
                                      dev2dev_safe=dev2dev_safe)
        finally:
            state.timing = False
            cols = int(band.shape[-1])
            with transfer.lock:
                transfer.broadcast_host_s += time.perf_counter() - host0
                transfer.broadcast_calls += 1
                transfer.band_cols[cols] = transfer.band_cols.get(cols, 0) + 1

    class _GatherTiming:
        """Context manager the cone prototype wraps one column batch's moves
        in, so those copies are timed exactly as the fan-out's are."""

        def __enter__(self):
            state.timing = True
            self.host0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            state.timing = False
            with transfer.lock:
                transfer.gather_host_s += time.perf_counter() - self.host0
                transfer.gather_calls += 1
            return False

    _sharding.move_shard = move_shard
    _sharding.broadcast_band_to_views = broadcast_band_to_views
    model._mg10_gather_timing = _GatherTiming

    def verify():
        """Is the instrument still on the path the driver takes?"""
        live = model.projector_functions
        return dict(
            projector_object_same=(live is pf),
            body_list_same=(live._fwd_body_per_dev is bodies),
            wrappers_in_place=all(
                bodies[i] is wrappers[i] for i in range(len(wrappers))),
            broadcast_wrapped=(
                _sharding.broadcast_band_to_views is broadcast_band_to_views),
            move_shard_wrapped=(_sharding.move_shard is move_shard))

    def detach():
        for index, body in enumerate(originals):
            bodies[index] = body
        pf._effective_view_batch = original_effective
        _sharding.move_shard = original_move
        _sharding.broadcast_band_to_views = original_broadcast
        model.__dict__.pop("_mg10_gather_timing", None)

    return busy, transfer, verify, detach, observed


# ── PATCH 1: the parallel sub-band walk ───────────────────────────────────────
def install_parallel_subband(model, band_len):
    """Walk each slice-owner's shard in sub-bands of ``band_len`` slices.

    THE SEAM.  `model.forward_project_slice_band` is read by the forward driver
    at tomography_model.py line 461 and handed to `_slice_band_length` at 471,
    whose result `_balanced_slice_bounds` tiles at 473.  Setting it is not a
    monkeypatch and that is the point: the proposed remedy is a change to
    `_slice_band_length`'s DEFAULT, so a fixed value through this knob produces
    exactly the walk a library implementation would produce -- the same
    balanced tiling, one broadcast per sub-band, the same projector calls, the
    same assembly, and no change to any accumulation.  It is also the only
    forward-only seam: `_slice_band_length` and `_balanced_slice_bounds` are
    static methods the BACK driver shares, so shadowing either would re-band
    the back projection, which is inside the same timed reconstruction.

    Returns ``(verify, detach)``.  The witness is not here -- it is the
    broadcast counter, checked against arithmetic in the arm."""
    if band_len is None:
        raise ValueError("install_parallel_subband needs a band length")
    previous = getattr(model, "forward_project_slice_band", None)
    if previous is not None:
        raise RuntimeError(
            f"the model already carries forward_project_slice_band={previous}; "
            "this arm would not be measuring what it says it measures")
    model.forward_project_slice_band = int(band_len)

    def verify():
        return dict(
            forward_band_set=(getattr(model, "forward_project_slice_band", None)
                              == int(band_len)),
            back_band_untouched=(getattr(model, "back_project_slice_band", None)
                                 is None))

    def detach():
        try:
            del model.forward_project_slice_band
        except AttributeError:
            model.forward_project_slice_band = None

    return verify, detach


# ── PATCH 2: the cone column-gather prototype ─────────────────────────────────
def install_cone_column_gather(model, pixel_batch):
    """Replace the cone multi-device forward with the column-gather shape.

    WHAT IT DOES, and how it differs from what ships.  The shipped cone branch
    walks slice-owner bands exactly as parallel does: broadcast a band to every
    view-owner, have each view-owner project its OWN views from that band into
    a FULL-detector partial sinogram, and add the partials up on the host, one
    add per band (tomography_model.py lines 508 to 538).  This prototype does
    what mbirjax does instead.  Each view-owner walks the pixel axis in batches
    of ``pixel_batch`` columns; for each batch it gathers that batch's piece
    from EVERY slice-owner and concatenates them along the slice axis into one
    FULL-HEIGHT cylinder on its own device, then makes ONE projector call on
    that cylinder over its own views, and accumulates one call's output per
    column batch.

    WHY IT IS FAITHFUL, item by item, because the measurement is worth only
    what the faithfulness is worth.
      * The kernel bodies are the shipped ones.  The call goes through
        `Projectors.sparse_forward_project_view_range`, the same entry the
        driver uses, with the same `dev_index`, so the same per-device body
        instance runs and the driver's own view batching applies.
      * The transfer primitive is the shipped one.  `_sharding.move_shard` is
        resolved as a module global at call time, so it is whatever the driver
        would have called, instrument included.
      * `slice_start=0` with a cylinder spanning every slice is exactly the
        unbanded projection.  The cone kernel's own docstring
        (triton_cone.py lines 613 to 617) states the contract the other way
        round: a band at `slice_start` sums, over a tiling of the slice axis,
        to the unbanded projection.  A full-height call at 0 is that tiling
        with one tile.
      * The return is the driver's return: `_sharding.Shards` of one
        (views, rows, channels) tensor per view-owner, in device order, with a
        padded view tail zero-filled -- the same lines the driver runs.
      * The fan-out is `_sharding.run_per_device` over the sino placement with
        the reconstruction loop's own pool, so the thread structure is the
        driver's.
      * A trivial placement and any row-aligned geometry DELEGATE to the
        original, so this can only ever change the cone multi-device branch.

    WHERE IT IS NOT FAITHFUL, recorded because a reader must be able to
    discount it.
      * It issues its copies from the per-view-owner worker threads, where
        mbirjax issues them, while the shipped band fan-out issues them from
        the loop thread.  That is also the memo's declined option A2, whose
        measured target is under one second of a thirty second span, so the
        prototype's time carries at most that much of a second effect.
      * It accumulates one full sinogram shard per COLUMN BATCH where the
        shipped path accumulates one per BAND, which at this cell is a couple
        of hundred host-side adds of a two-gigabyte tensor instead of one.
        mbirjax's loop has the same form, so this is faithful to the reference,
        but a tuned implementation would accumulate in place inside the
        projector call, and the pixel-batch sweep prices exactly this term.

    Returns ``(counters, verify, detach)``."""
    import torch
    from mbirtorch import _sharding

    if pixel_batch is None or int(pixel_batch) < 1:
        raise ValueError("install_cone_column_gather needs a positive batch")
    pixel_batch = int(pixel_batch)
    original = type(model)._sparse_forward_project_sharded
    n_dev_guess = model.sino_placement.n_devices

    counters = dict(
        pixel_batch=pixel_batch,
        proto_calls=0,             # cone multi-device forwards the prototype ran
        delegated_trivial=0,       # one-device placements handed back
        delegated_aligned=0,       # row-aligned geometries handed back
        expected_batches=0,        # column batches ONE view-owner should walk
        pixel_counts={},           # per-call pixel-set size -> how many calls
        batches=[0] * n_dev_guess,     # column batches each view-owner walked
        moves=[0] * n_dev_guess,       # pieces each view-owner moved
        cyl_height={},             # assembled cylinder height -> count (per arm)
        cyl_width={},              # assembled cylinder pixel width -> count
    )
    counters_lock = threading.Lock()

    def prototype(self, voxel_shards, pixel_indices):
        # The two delegations, first: this prototype exists only for the cone
        # multi-device branch.
        if voxel_shards.placement.is_trivial:
            counters["delegated_trivial"] += 1
            return original(self, voxel_shards, pixel_indices)
        if self.rows_track_slices:
            counters["delegated_aligned"] += 1
            return original(self, voxel_shards, pixel_indices)

        sp, rp, view_spans, band_ranges, idx_per = self._banded_setup(
            pixel_indices)
        pf = self.projector_functions
        num_rows = int(self.get_params('sinogram_shape')[1])
        num_channels = int(self.get_params('sinogram_shape')[2])
        num_pixels = int(idx_per[0].shape[0])
        shards = voxel_shards.tensors      # in slice-owner (global slice) order
        if rp.is_padded:
            # Not reachable at mg10's cell (1008 slices divide 1, 2 and 4), and
            # a gathered cylinder that carries a padded tail is a path nothing
            # has tested.  Refuse loudly rather than measure something unknown.
            raise RuntimeError(
                "the cone column-gather prototype refuses a padded recon "
                "placement: the assembled cylinder would carry a zero tail "
                "whose projection nothing here has verified.  mg10's cell has "
                "1008 slices, which divides every device count it runs at.")
        n_batches = -(-num_pixels // pixel_batch)
        with counters_lock:
            counters["proto_calls"] += 1
            counters["expected_batches"] += n_batches
            counters["pixel_counts"][num_pixels] = \
                counters["pixel_counts"].get(num_pixels, 0) + 1
            # Sized here rather than at install time: the device count settles
            # inside the first reconstruction, so a list sized against the
            # placement this object had when it was built could be too short.
            for key in ("batches", "moves"):
                if len(counters[key]) < sp.n_devices:
                    counters[key] = counters[key] + \
                        [0] * (sp.n_devices - len(counters[key]))
        timing = getattr(self, "_mg10_gather_timing", None)

        def worker(i, dev):
            v0, v1, _block = view_spans[i]
            if v1 <= v0:
                # A view-owner with no real views produces an empty block, the
                # same shape the shipped branch produces for it.
                return torch.zeros((0, num_rows, num_channels),
                                   dtype=voxel_shards.dtype, device=dev)
            local_idx = idx_per[i]
            owned = None
            for p0 in range(0, num_pixels, pixel_batch):
                p1 = min(p0 + pixel_batch, num_pixels)
                # THE GATHER: this column batch's piece from every slice-owner,
                # moved to this view-owner and concatenated along the SLICE
                # axis in global slice order.
                if timing is not None:
                    with timing():
                        pieces = [_sharding.move_shard(t[p0:p1], dev,
                                                       self.dev2dev_safe)
                                  for t in shards]
                else:
                    pieces = [_sharding.move_shard(t[p0:p1], dev,
                                                   self.dev2dev_safe)
                              for t in shards]
                full_cyl = (pieces[0] if len(pieces) == 1
                            else torch.cat(pieces, dim=1))
                counters["batches"][i] += 1
                counters["moves"][i] += len(pieces)
                pieces = None
                if i == 0:
                    # One writer, one device: the shape witness is recorded on
                    # view-owner 0 only, so it needs no lock.
                    height = int(full_cyl.shape[1])
                    width = int(full_cyl.shape[0])
                    counters["cyl_height"][height] = \
                        counters["cyl_height"].get(height, 0) + 1
                    counters["cyl_width"][width] = \
                        counters["cyl_width"].get(width, 0) + 1
                part = pf.sparse_forward_project_view_range(
                    full_cyl, local_idx[p0:p1], (v0, v1), slice_start=0,
                    dev_index=i)
                full_cyl = None
                if owned is None:
                    owned = part
                else:
                    owned.add_(part)
                # Released AFTER the accumulation, so the summation order is
                # untouched and only the residency changes -- the same release
                # the shipped branch makes on its per-band partials.
                part = None
            return owned

        with self._band_pool(sp.n_devices) as pool:
            tensors = _sharding.run_per_device(sp.devices, worker,
                                               executor=pool)
        if sp.is_padded:
            # The driver's own tail fill (tomography_model.py lines 544 to 551).
            tensors = [
                t if t.shape[0] == block else torch.cat(
                    [t, torch.zeros((block - t.shape[0],) + tuple(t.shape[1:]),
                                    dtype=t.dtype, device=t.device)])
                for t, (_v0, _v1, block) in zip(tensors, view_spans)]
        return _sharding.Shards(tensors, sp)

    prototype.__name__ = "mg10_cone_column_gather"
    bound = prototype.__get__(model, type(model))
    model._sparse_forward_project_sharded = bound

    def verify():
        return dict(
            prototype_bound=(model.__dict__.get(
                "_sparse_forward_project_sharded") is bound),
            class_method_untouched=(
                type(model)._sparse_forward_project_sharded is original))

    def detach():
        model.__dict__.pop("_sparse_forward_project_sharded", None)

    return counters, verify, detach


# ── PATCH 3: the one-device narrowed values block ─────────────────────────────
def install_parallel_n1_values_split(model, width):
    """Hand the ONE-device forward its values block in width-``width`` pieces.

    WHY THIS IS NOT EXPERIMENT 1'S KNOB.  `forward_project_slice_band` is read
    inside `_sparse_forward_project_sharded`, thirteen lines AFTER that driver
    has already returned for a trivial placement (tomography_model.py 448 to
    452, then 461).  At one device the knob is inert.  Setting it here would
    produce a row that claims a narrowed walk and measures the control, which
    is the one failure mode this file is built to make impossible.

    THE SEAM, and why it is this one.
      * `Projectors._sparse_forward_project_single_device` is the SINGLE choke
        point for one device.  Its only two callers are the funnel's
        non-`Shards` branch (line 342) and the trivial early return (line 450),
        and a timed reconstruction reaches it through both: `recon` projects
        the initial volume as a plain tensor, and the VCD loop projects each
        subset's delta as a one-shard `Shards`.  A patch on either caller alone
        would leave part of every reconstruction running the shipped
        monolithic call, and the mixture would read as a narrowed arm.
      * It is patched ON THE CLASS.  A device-count settle inside the cold pass
        rebuilds the projector OBJECT (`_install_device_layout` ->
        `create_projectors`), which would throw away an instance attribute; the
        class survives that, and the new projector instance still holds the
        same model.  The patch checks `self.model is` this arm's model and
        delegates otherwise, so nothing else in the process can reach it.

    WHAT ONE PIECE'S CALL LOOKS LIKE, against the shipped call.  The shipped
    body coerces the values and the indices, then makes ONE call:
        sparse_forward_project_view_range(values, idx, (0, num_views),
                                          dev_index=0)
    with `values` the whole (P, num_slices) block.  This body makes the same
    coercion and then the same call once per piece, with `values[:, l0:l1]` in
    place of `values` -- same entry, same view range, same device index, same
    `slice_start` default, same per-device compiled body, same view batching
    (the transient budget's cap is far above the nominal 128 at every width
    here, so the view batch is 128 for the pieces exactly as for the whole
    block).  Rows track slices one to one in parallel beam, so piece k's
    sinogram IS detector rows [l0, l1), and `torch.cat(..., dim=1)` in slice
    order rebuilds the whole sinogram: no summation, no reordering, and the
    same concatenation the two-device driver performs for a row-aligned
    geometry (tomography_model.py line 540).

    THE TILING is this file's own `balanced_slice_bounds` -- the driver's rule,
    re-derived here -- applied to the WHOLE volume rather than to a shard, so a
    piece width at one device means the same thing a sub-band width means at
    two.  A request wider than the volume caps to the volume, exactly as
    `_slice_band_length` caps a band request at the shard.

    ONE COST THAT IS INSIDE THE TIMED BODY, named so it can be subtracted.
    `values[:, l0:l1]` is a strided view and the Triton parallel forward body
    calls `.contiguous()` on its values argument (triton_parallel.py line 437),
    so a narrowed piece packs its own block inside the per-call bracket where
    the width-1008 control packs nothing.  The two-device driver already pays
    exactly this on every band it projects from its own shard, because a
    `move_shard` to a tensor's own device returns the tensor itself -- half of
    that fan-out's bands arrive strided.  The arm measures the pack directly
    (`values_pack_ms`) so a reader can subtract it; it is proportional to the
    width, so it cannot bend the shape of the width curve.

    Returns ``(counters, verify, detach)``."""
    import torch

    from mbirtorch.projectors import Projectors

    if width is None or int(width) < 1:
        raise ValueError("install_parallel_n1_values_split needs a width")
    width = int(width)
    if not getattr(model, "rows_track_slices", False):
        raise RuntimeError(
            "the one-device values split is a ROW-ALIGNED geometry's "
            "measurement: it assembles the sinogram by concatenating each "
            "piece's detector rows, which is only the right answer when rows "
            "track slices one to one.  This model says they do not.")
    original = Projectors._sparse_forward_project_single_device
    if getattr(original, "_mg10_n1_split", False):
        raise RuntimeError(
            "Projectors._sparse_forward_project_single_device is already "
            "patched; this arm would measure a split of a split")

    counters = dict(
        width=width,
        split_calls=0,          # one-device forwards this body served
        delegated_other_model=0,  # calls from some other model, handed back
        pieces_per_call={},     # pieces cut per call -> how many calls
        piece_widths={},        # piece width -> how many pieces
        piece_contiguous={},    # was the piece contiguous -> how many pieces
        cat_calls=0,            # calls whose pieces had to be concatenated
    )
    lock = threading.Lock()

    def patched(self, voxel_values, pixel_indices):
        if self.model is not model:
            with lock:
                counters["delegated_other_model"] += 1
            return original(self, voxel_values, pixel_indices)
        m = self.model
        num_views = int(m.get_params('sinogram_shape')[0])
        # The shipped body's own coercion, verbatim: the seam is the split, not
        # the placement.
        values = torch.as_tensor(voxel_values, dtype=torch.float32,
                                 device=m.torch_device)
        indices = torch.as_tensor(pixel_indices, dtype=torch.int64,
                                  device=m.torch_device)
        extent = int(values.shape[-1])
        bounds = balanced_slice_bounds(extent, min(width, extent))
        pieces = []
        for (l0, l1) in bounds:
            block = values[:, l0:l1]
            with lock:
                counters["piece_widths"][l1 - l0] = \
                    counters["piece_widths"].get(l1 - l0, 0) + 1
                key = bool(block.is_contiguous())
                counters["piece_contiguous"][key] = \
                    counters["piece_contiguous"].get(key, 0) + 1
            pieces.append(self.sparse_forward_project_view_range(
                block, indices, (0, num_views), dev_index=0))
        with lock:
            counters["split_calls"] += 1
            counters["pieces_per_call"][len(bounds)] = \
                counters["pieces_per_call"].get(len(bounds), 0) + 1
            if len(pieces) > 1:
                counters["cat_calls"] += 1
        # Slice order, along the ROW axis -- the driver's own assembly for a
        # row-aligned geometry.  A single piece is returned as is, so a width
        # that caps back onto the whole volume is bit-for-bit the shipped call.
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

    patched._mg10_n1_split = True
    patched.__name__ = "mg10_n1_values_split"
    Projectors._sparse_forward_project_single_device = patched

    def verify():
        return dict(
            split_installed=(
                Projectors._sparse_forward_project_single_device is patched),
            class_method_replaced=(original is not patched),
            band_knob_untouched=(
                getattr(model, "forward_project_slice_band", None) is None
                and getattr(model, "back_project_slice_band", None) is None))

    def detach():
        Projectors._sparse_forward_project_single_device = original

    return counters, verify, detach


# ── PATCH 4: the narrowed forward view chunk ──────────────────────────────────
def install_parallel_view_chunk(model, chunk, kernel_module, fwd_const,
                                back_const, expect_kernel):
    """Narrow the FORWARD view chunk to ``chunk`` views per launch.

    WHAT THIS MOVES, and why it is the other axis of the same question.  One
    forward launch's atomic write target is the zeroed output plane the kernel
    accumulates into: ``(view batch, channels, values columns)`` float32
    (triton_parallel.py line 444).  Experiment 3 varies the VALUES COLUMNS of
    that slab; this varies the VIEW BATCH, at a values width nobody touches.
    Both land on the same byte count from different directions, so if the slab
    fitting in cache is what the two-device width-63 anomaly was about, the two
    axes have to agree.

    THE SEAM.  ``Projectors.view_batch_charge`` takes its nominal from
    ``model.view_batch_size`` when the user set one, and otherwise from the
    BODY's own ``_view_batch_cost``, which for the Triton parallel forward
    returns the module constant ``PARALLEL_FWD_VIEW_CHUNK`` -- read as a module
    global at call time (triton_parallel.py line 484), so rebinding the
    constant is exactly the change a library edit would make and nothing else
    moves.  ``model.view_batch_size`` was NOT used: it is the nominal for every
    body, forward and back alike, and the back projection is inside the same
    timed reconstruction.  The forward and back chunks are separate constants
    (lines 115 and 130), so this seam is forward-only by construction, and the
    arm re-reads the back constant afterwards to prove it.

    THE SMOKE'S SECOND HALF.  On CPU the parallel bodies are torch bodies with
    no ``_view_batch_cost``, so the constant is never consulted and the nominal
    is ``Projectors.VIEW_BATCH_BODY_DEFAULT``.  A constant-only patch would
    make the local smoke exercise nothing.  So this also clamps
    ``Projectors._effective_view_batch`` -- on the CLASS, so a projector
    rebuild cannot drop it -- for FORWARD bodies only, identified by identity
    against ``_fwd_body_per_dev`` (which holds the busy probe's wrappers once
    those are installed, and the raw bodies before).  On the kernel path the
    clamp must never fire, because the constant already produced the same
    number, and the arm ASSERTS that: a firing clamp on CUDA means the constant
    seam moved and the row would be measuring the clamp instead.

    Returns ``(counters, verify, detach)``."""
    from mbirtorch.projectors import Projectors

    if chunk is None or int(chunk) < 1:
        raise ValueError("install_parallel_view_chunk needs a positive chunk")
    chunk = int(chunk)
    original_fwd_chunk = int(getattr(kernel_module, fwd_const))
    original_back_chunk = int(getattr(kernel_module, back_const))
    if chunk >= original_fwd_chunk:
        raise RuntimeError(
            f"the view chunk arm asked for {chunk} against a shipped forward "
            f"chunk of {original_fwd_chunk}; a chunk at or above the shipped "
            f"one is the control under another name")
    original_effective = Projectors._effective_view_batch
    if getattr(original_effective, "_mg10_view_chunk", False):
        raise RuntimeError(
            "Projectors._effective_view_batch is already clamped; this arm "
            "would be measuring a clamp of a clamp")

    counters = dict(chunk=chunk, forward_calls=0, back_calls=0,
                    clamped_calls=0, forward_values={}, back_values={})
    lock = threading.Lock()

    setattr(kernel_module, fwd_const, chunk)

    def clamped(self, body, num_pixels, band_cols, args):
        value = int(original_effective(self, body, num_pixels, band_cols, args))
        forward = any(body is other for other in self._fwd_body_per_dev)
        out = min(value, chunk) if forward else value
        with lock:
            if forward:
                counters["forward_calls"] += 1
                counters["forward_values"][out] = \
                    counters["forward_values"].get(out, 0) + 1
                if out != value:
                    counters["clamped_calls"] += 1
            else:
                counters["back_calls"] += 1
                counters["back_values"][value] = \
                    counters["back_values"].get(value, 0) + 1
        return out

    clamped._mg10_view_chunk = True
    Projectors._effective_view_batch = clamped

    def verify():
        out = dict(
            fwd_chunk_forced=(int(getattr(kernel_module, fwd_const)) == chunk),
            back_chunk_untouched=(
                int(getattr(kernel_module, back_const)) == original_back_chunk),
            clamp_installed=(Projectors._effective_view_batch is clamped))
        if expect_kernel:
            # A kernel body takes its nominal from the constant, so on this
            # path the clamp must never have anything to do.  A clamp that
            # fired means the constant seam moved and the row would be
            # measuring the clamp instead of the change a library edit makes.
            # On the CPU smoke's torch bodies the clamp IS the mechanism, so
            # the key is absent there rather than false.
            out["clamp_never_fired_on_the_kernel_path"] = (
                counters["clamped_calls"] == 0)
        return out

    def detach():
        setattr(kernel_module, fwd_const, original_fwd_chunk)
        Projectors._effective_view_batch = original_effective
        pf = getattr(model, "projector_functions", None)
        # attach_forward_probes restores its observer by ASSIGNING the bound
        # method it captured, which lands as an instance attribute that would
        # shadow the class method restored above.  There was none before this
        # arm, so removing it is the restoration.
        if pf is not None and "_effective_view_batch" in pf.__dict__:
            del pf.__dict__["_effective_view_batch"]

    return counters, verify, detach


def measure_values_pack(model, torch_module, cuda, width, repeats=5):
    """Time the strided-to-contiguous pack a narrowed piece pays INSIDE the
    timed body, so it can be subtracted from that arm's per-launch time.

    The Triton parallel forward body calls ``.contiguous()`` on its values
    argument, and a width-W piece of the volume is a strided view, so the pack
    is real work charged to the launch.  This measures it directly on a block
    of the arm's own shape rather than modelling it: allocate one
    (num_pixels, num_slices) float32 block, take ``[:, 0:W]``, and time
    ``.contiguous()`` on it.  Returns the MEDIAN of ``repeats`` timings in
    milliseconds, or None when the shape does not fit or the arm is not
    narrowed.

    Run OUTSIDE every timed reconstruction and after the probes are detached,
    so its own allocation never lands in a measured peak or a measured span.
    """
    import mbirtorch

    if not width:
        return None
    recon_shape = tuple(model.get_params("recon_shape"))
    num_slices = int(recon_shape[2])
    if width >= num_slices:
        return 0.0
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    device = model.sino_placement.devices[0]
    try:
        block = torch_module.zeros((num_pixels, num_slices),
                                   dtype=torch_module.float32, device=device)
    except Exception:                                             # noqa: BLE001
        return None
    try:
        view = block[:, 0:width]
        timings = []
        for index in range(repeats + 1):
            if cuda:
                with torch_module.cuda.device(device):
                    start = torch_module.cuda.Event(enable_timing=True)
                    end = torch_module.cuda.Event(enable_timing=True)
                    start.record()
                    packed = view.contiguous()
                    end.record()
                torch_module.cuda.synchronize(device)
                span = float(start.elapsed_time(end))
            else:
                host0 = time.perf_counter()
                packed = view.contiguous()
                span = (time.perf_counter() - host0) * 1e3
            packed = None
            if index:                       # the first pass is the warm-up
                timings.append(span)
        return float(statistics.median(timings)) if timings else None
    finally:
        block = None
        if cuda:
            torch_module.cuda.empty_cache()


# ── the torch side (mg5's / mg9's model builder and checks) ───────────────────
def _build_torch_model(geometry, cell, pin_devices=None):
    """The model.  ``pin_devices`` is a device LIST for the smoke's CPU paths
    only; on CUDA nothing is configured here, because an arm pins through
    MBIRTORCH_NUM_DEVICES and an explicit configure_devices call would take the
    explicit branch and skip the preflight."""
    import numpy as np

    import mbirtorch

    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels)
    if pin_devices is not None:
        model.configure_devices(devices=pin_devices)
    model.set_params(no_warning=True, verbose=0)
    return model


def _launch_key_counts(geometry):
    """Per-kernel launch-key counts (the positive witness that the kernels
    ran): all four kernels share one key set and every key leads with its
    kernel's name."""
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    names = (("pback", "pfwd") if geometry == "parallel" else ("back", "fwd"))
    back = sum(1 for k in _COMPILED_LAUNCH_KEYS
               if isinstance(k, tuple) and k and k[0] == names[0])
    fwd = sum(1 for k in _COMPILED_LAUNCH_KEYS
              if isinstance(k, tuple) and k and k[0] == names[1])
    return back, fwd


def _view_batch_static(model, expect_kernels, arm_fwd_cols,
                       forced_fwd_chunk=None):
    """The realized view batch per direction per device, against the formula of
    the body EXPECTED to be bound (mg1's static probe).  Run BEFORE the probes
    are installed, so the probe's own observer cannot see this traffic.

    ``arm_fwd_cols`` is this arm's own forward column count -- the sub-band
    width, or the full slice count for the cone prototype -- so the row carries
    the batch the arm will actually run at as well as the shipped default's.

    ``forced_fwd_chunk`` is experiment 4's narrowed view chunk.  On the kernel
    path the constant already produced it and this changes nothing; on the CPU
    smoke's torch bodies the arm forces it through the clamp instead, and the
    expectation has to know that or a smoke arm would report vb:FAIL for doing
    exactly what it was asked to do."""
    import mbirtorch

    pf = model.projector_functions
    args = model._view_batch_args()
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    budget = pf._transient_budget_bytes()
    n_dev = model.sino_placement.n_devices
    cols = dict(fwd=int(-(-recon_shape[2] // n_dev)),
                back=int(sinogram_shape[1]))
    bodies = dict(fwd=pf._fwd_body_per_dev, back=pf._back_body_per_dev)
    expected_is_kernel = dict(fwd=expect_kernels[0], back=expect_kernels[1])

    record, ok = {}, True
    for direction in ("fwd", "back"):
        realized, expected_all = [], []
        for body in bodies[direction]:
            value = int(pf._effective_view_batch(body, num_pixels,
                                                 cols[direction], args))
            legacy = max(1, min(64, budget
                                // max(1, num_pixels
                                       * model._transient_cols(cols[direction])
                                       * 4)))
            cost = getattr(body, "_view_batch_cost", None)
            if expected_is_kernel[direction]:
                if cost is None:
                    expected = None
                else:
                    bytes_pv, chunk = cost(num_pixels, cols[direction], args)
                    expected = max(1, min(int(chunk),
                                          budget // max(1, bytes_pv)))
            else:
                expected = int(legacy)
            if direction == "fwd" and forced_fwd_chunk and expected is not None:
                expected = min(int(expected), int(forced_fwd_chunk))
            realized.append(value)
            expected_all.append(expected)
            ok = ok and (expected is not None) and (value == expected)
        record[f"{direction}_view_batch_per_device"] = realized
        record[f"{direction}_view_batch_expected_per_device"] = expected_all
    record["num_pixels_full"] = num_pixels
    record["budget_bytes"] = int(budget)
    record["view_batch_cols"] = cols
    # This arm's own point on the same formula.  The cone prototype hands the
    # driver `pixel_batch` pixels instead of the whole set, which lowers the
    # per-view charge and therefore RAISES the batch the budget allows; that is
    # a real property of the proposed shape and belongs on the row.
    arm_pixels = int(arm_fwd_cols.get("pixels") or num_pixels)
    arm_cols = int(arm_fwd_cols.get("cols") or cols["fwd"])
    record["arm_fwd_pixels"] = arm_pixels
    record["arm_fwd_cols"] = arm_cols
    record["arm_fwd_view_batch"] = int(
        pf._effective_view_batch(bodies["fwd"][0], arm_pixels, arm_cols, args))
    return record, ok


def _per_device_body_names(model):
    pf = model.projector_functions
    return ([getattr(b, "__name__", str(b)) for b in pf._fwd_body_per_dev],
            [getattr(b, "__name__", str(b)) for b in pf._back_body_per_dev])


def _merge_regions(total, part):
    for region in REGIONS:
        block = part[region]
        slot = total.setdefault(region, dict(calls=0, host_wall_s=0.0,
                                             device_span_ms={}, devices=[]))
        slot["calls"] += block["calls"]
        slot["host_wall_s"] += block["host_wall_s"]
        for name, span in block["device_span_ms"].items():
            slot["device_span_ms"][name] = \
                slot["device_span_ms"].get(name, 0.0) + span
        for name in block["devices"]:
            if name not in slot["devices"]:
                slot["devices"].append(name)
    return total


def _sample_steps(shape):
    """A stride per axis giving roughly VALUE_SAMPLE_TARGET samples along it."""
    return tuple(max(1, int(n) // VALUE_SAMPLE_TARGET) for n in shape)


def torch_worker(cfg):
    """One arm: the patch (if any), one discarded cold pass, then WARM_REPEATS
    timed reconstructions with every instrument live.

    TWO ORDERINGS, both load-bearing.
      The PATCH goes in BEFORE the cold pass, because it changes the shapes the
    Triton kernels launch at and the launch key includes the pixel count, the
    band length and the view count; installed later, every first launch and the
    compile lock it takes would land inside the first timed reconstruction.
      The PROBES go in AFTER the cold pass, because the body wrappers live
    inside the projector object and a device-count settle during the first
    reconstruction rebuilds it.  Both patch and probes are re-verified around
    every timed pass."""
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg.get("n_dev")
    band_len = cfg.get("band_len")
    pixel_batch = cfg.get("pixel_batch")
    values_width = cfg.get("values_width")
    view_chunk = cfg.get("view_chunk")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    import importlib
    spec = SPEC[geometry]
    kernel_module = importlib.import_module(spec["kernel_module"])
    shipped_fwd_chunk = int(getattr(kernel_module, spec["fwd_chunk_const"]))
    shipped_back_chunk = int(getattr(kernel_module, spec["back_chunk_const"]))

    pin_devices = None
    if not cuda and smoke_cpu_devices:
        pin_devices = list(smoke_cpu_devices)
    elif not cuda:
        pin_devices = [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=pin_devices)

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  compile_enabled=bool(getattr(model, "compile_enabled", False)),
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  shipped_fwd_chunk=shipped_fwd_chunk,
                  shipped_back_chunk=shipped_back_chunk)

    # ── the env arm checks ───────────────────────────────────────────────────
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))
    if cuda:
        result["pin_env_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") == str(n_dev))
    result["shipped_chunk_is_the_anchor_ok"] = (
        shipped_fwd_chunk == SHIPPED_CHUNK)

    expect_kernels = (cuda, cuda)
    result["expected_bodies_kernel"] = list(expect_kernels)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    num_slices = int(recon_shape[2])
    result["num_slices"] = num_slices
    result["num_slices_planned_ok"] = (num_slices == num_slices_for(cell))

    # ── the arm's own shape arithmetic, derived here and checked later ───────
    # Everything the witnesses are compared against comes from these numbers,
    # and every one of them is derived from the arm's parameters rather than
    # read back out of the library.
    n_owners = n_dev if cuda else len(pin_devices or [DEVICE])
    slices_per_dev = num_slices // max(1, n_owners)
    # ONE width request per arm, whichever seam produces it, tiled by the same
    # rule: experiment 1's sub-band inside a shard, experiment 3's piece inside
    # the whole one-device volume (which at n = 1 IS the shard).  Keeping one
    # set of realized-width fields is what lets the n = 1 and n > 1 rows be read
    # in one table; `width_seam` says which mechanism produced them.
    width_request = band_len if band_len is not None else values_width
    lengths = realized_band_lengths(slices_per_dev, width_request)
    result["slices_per_dev"] = slices_per_dev
    result["requested_band_len"] = band_len
    result["requested_values_width"] = values_width
    result["width_seam"] = (
        "model.forward_project_slice_band (the driver's own knob)" if band_len
        else ("Projectors._sparse_forward_project_single_device (the harness "
              "splits the one-device values block; the knob is inert at n=1)"
              if values_width else None))
    result["realized_band_lengths"] = lengths
    result["realized_band_len"] = lengths[0] if lengths else None
    result["sub_bands_per_owner"] = len(lengths)
    result["band_is_whole_shard"] = (len(lengths) == 1)
    result["requested_view_chunk"] = view_chunk
    names = []
    if pixel_batch:
        names.append("cone_column_gather")
    if band_len:
        names.append("parallel_subband")
    if values_width:
        names.append("parallel_n1_values_split")
    if view_chunk:
        names.append("parallel_view_chunk")
    result["patch"] = "+".join(names) if names else "none"

    # ── the shared sinogram artifact, md5-verified ───────────────────────────
    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    result["sino_md5"] = actual_md5
    result["sino_md5_ok"] = (actual_md5 == expected_md5)
    if not result["sino_md5_ok"]:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5} (the Lustre "
                           f"corrupt-read failure mode)")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram),
                                               dtype=np.float64))

    # ── THE PATCHES, before the cold pass (see the docstring) ────────────────
    # A list, because the view chunk is an INDEPENDENT seam that can ride along
    # with any of the others: it moves one axis of the write slab and the shape
    # patches move a different one.  Every arm here uses at most one shape
    # patch plus at most one chunk patch, and the composite verify is the flat
    # merge of the parts (no two parts share a key).
    patch_parts, detach_parts = [], []
    cone_counters = split_counters = chunk_counters = None
    if band_len is not None and values_width is not None:
        raise RuntimeError(
            "an arm asked for BOTH a sub-band and a one-device values split; "
            "they are two different seams answering two different questions "
            "and an arm carrying both would be neither measurement")
    if band_len is not None:
        if geometry != "parallel":
            raise RuntimeError("a sub-band arm must be a parallel arm")
        if (n_owners or 1) < 2:
            raise RuntimeError(
                "a sub-band arm needs more than one device: the driver reads "
                "forward_project_slice_band only AFTER its trivial-placement "
                "return, so at one device the knob is inert and this arm would "
                "report the shipped control under a narrowed arm's name.  The "
                "one-device widths are experiment 3's arms (p1_band*).")
        parts = install_parallel_subband(model, band_len)
        patch_parts.append(parts[0])
        detach_parts.append(parts[1])
    elif values_width is not None:
        if geometry != "parallel":
            raise RuntimeError(
                "a one-device values-split arm must be a parallel arm: the "
                "assembly concatenates each piece's detector rows, which is "
                "only correct where rows track slices one to one")
        if (n_owners or 1) != 1:
            raise RuntimeError(
                "a one-device values-split arm must run at exactly one device; "
                "at more than one the shipped banded driver owns the walk and "
                "experiment 1's knob is the seam")
        split_counters, verify_fn, detach_fn = \
            install_parallel_n1_values_split(model, values_width)
        patch_parts.append(verify_fn)
        detach_parts.append(detach_fn)
    elif pixel_batch is not None:
        if geometry != "cone":
            raise RuntimeError("a column-gather arm must be a cone arm")
        if (n_owners or 1) < 2:
            raise RuntimeError(
                "a column-gather arm needs more than one device: at one device "
                "the placement is trivial and the shipped single-device path "
                "already makes the full-height calls this prototype imitates")
        cone_counters, verify_fn, detach_fn = \
            install_cone_column_gather(model, pixel_batch)
        patch_parts.append(verify_fn)
        detach_parts.append(detach_fn)
    if view_chunk is not None:
        if geometry != "parallel":
            raise RuntimeError(
                "the view-chunk arms are parallel arms: the constant they "
                "rebind is the parallel forward kernel's own, and cone's "
                "realized batch is capped by the transient budget rather than "
                "by the constant at this cell, so the seam would not bite")
        chunk_counters, verify_fn, detach_fn = install_parallel_view_chunk(
            model, view_chunk, kernel_module, spec["fwd_chunk_const"],
            spec["back_chunk_const"], expect_kernels[0])
        patch_parts.append(verify_fn)
        detach_parts.append(detach_fn)
    patch_verify = None
    patch_detach = None
    if patch_parts:
        def patch_verify():                                       # noqa: F811
            merged = {}
            for part in patch_parts:
                merged.update(part())
            return merged

        def patch_detach():                                       # noqa: F811
            for part in reversed(detach_parts):
                part()
    result["patch_verify_at_install"] = patch_verify() if patch_verify else None
    if patch_verify and not all(result["patch_verify_at_install"].values()):
        raise RuntimeError(
            f"the {result['patch']} patch did not install: "
            f"{result['patch_verify_at_install']}")

    # The region instrument goes in BEFORE the cold pass and is drained after
    # every timed pass.
    instrument, detach_regions = attach_region_instrument(model, torch, cuda)

    def peaks():
        if not cuda:
            return []
        return [int(torch.cuda.max_memory_allocated(d))
                for d in model.sino_placement.devices]

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(recon)

    keys_before = _launch_key_counts(geometry) if cuda else (0, 0)
    health = [sample_gpu_health()]

    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the projector-dependent checks, now that the layout has SETTLED ──────
    result["patch_verify_after_cold"] = patch_verify() if patch_verify else None
    if patch_verify and not all(result["patch_verify_after_cold"].values()):
        raise RuntimeError(
            f"the {result['patch']} patch did not survive the cold pass: "
            f"{result['patch_verify_after_cold']}")
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    fwd_per_dev, back_per_dev = _per_device_body_names(model)
    n_realized = model.sino_placement.n_devices
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_body_per_device=fwd_per_dev,
                  back_body_per_device=back_per_dev)
    want_fwd_kernel, want_back_kernel = expect_kernels
    result["bodies_per_device_ok"] = (
        len(fwd_per_dev) == n_realized and len(back_per_dev) == n_realized
        and all(("triton" in name) == want_fwd_kernel for name in fwd_per_dev)
        and all(("triton" in name) == want_back_kernel for name in back_per_dev))
    result["bodies_ok"] = (
        result["bodies_per_device_ok"]
        and ("triton" in fwd_name) == want_fwd_kernel
        and ("triton" in back_name) == want_back_kernel)
    pf = model.projector_functions
    result["fwd_bodies_distinct_objects"] = (
        len({id(b) for b in pf._fwd_body_per_dev}) == len(pf._fwd_body_per_dev))

    # A wrong device count invalidates every shape number in this arm, because
    # the shard length -- and therefore the sub-band arithmetic and the column
    # height -- is derived from it.  So this one is fatal here rather than a
    # check on the summary line.
    realized_devices = [str(d) for d in model.sino_placement.devices]
    expected_count = n_dev if cuda else len(pin_devices or [DEVICE])
    if len(realized_devices) != expected_count:
        raise RuntimeError(
            f"this arm asked for {expected_count} device(s) and settled on "
            f"{len(realized_devices)} ({realized_devices}).  Every shape "
            f"number in the arm is derived from the shard length, so the "
            f"whole row would be mislabeled.")
    if not result["num_slices_planned_ok"]:
        raise RuntimeError(
            f"the plan assumed {num_slices_for(cell)} slices at this cell and "
            f"the model built {num_slices}; the sub-band and column-height "
            f"arithmetic in the plan print is wrong for this tree.")

    arm_cols = dict(
        pixels=(pixel_batch if pixel_batch else None),
        cols=(num_slices if pixel_batch else result["realized_band_len"]))
    vb_record, vb_ok = _view_batch_static(model, expect_kernels, arm_cols,
                                          forced_fwd_chunk=view_chunk)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    result["dev2dev_safe"] = bool(getattr(model, "dev2dev_safe", True))
    result["fwd_chunk_after"] = int(getattr(kernel_module,
                                            spec["fwd_chunk_const"]))
    result["back_chunk_after"] = int(getattr(kernel_module,
                                             spec["back_chunk_const"]))
    # Two different questions, kept apart.  "unchanged" is the literal reading
    # and stays on the row for every arm; "as intended" is what the summary
    # flags, because a view-chunk arm is SUPPOSED to have moved the forward
    # constant and is not supposed to have moved the back one.
    result["chunks_unchanged_ok"] = (
        result["fwd_chunk_after"] == shipped_fwd_chunk
        and result["back_chunk_after"] == shipped_back_chunk)
    result["chunk_as_intended_ok"] = (
        result["fwd_chunk_after"] == (view_chunk or shipped_fwd_chunk)
        and result["back_chunk_after"] == shipped_back_chunk)

    # ── the instruments, installed on the settled projector ──────────────────
    busy, transfer, verify, detach_probes, observed = attach_forward_probes(
        model, torch, cuda, MAX_EVENT_PAIRS,
        prototype_active=(pixel_batch is not None))
    result["probe_verify_before"] = verify()
    if not all(result["probe_verify_before"].values()):
        raise RuntimeError(
            f"the forward probes are not on the driver's path before the timed "
            f"passes: {result['probe_verify_before']}")

    # ── the timed reconstructions ────────────────────────────────────────────
    instrument.reset()
    if cuda:
        for device in model.sino_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    devices = list(model.sino_placement.devices)
    device_names = [str(d) for d in devices]
    warm, per_recon, region_totals = [], [], {}
    pair_total, cap_hit = 0, False
    checksums, sample_paths = [], []
    steps = None
    for repeat in range(WARM_REPEATS):
        if cone_counters is not None:
            # Per-reconstruction, so the witness arithmetic compares like with
            # like.  Reset AFTER the previous pass has been recorded.
            for key in ("proto_calls", "expected_batches"):
                cone_counters[key] = 0
            cone_counters["pixel_counts"] = {}
            cone_counters["cyl_height"] = {}
            cone_counters["cyl_width"] = {}
            cone_counters["batches"] = [0] * len(devices)
            cone_counters["moves"] = [0] * len(devices)
        if split_counters is not None:
            # Per-reconstruction, for the same reason the cone counters are:
            # the witness compares piece counts against THIS pass's forward
            # funnel call count.  Reset AFTER the previous pass was recorded.
            split_counters["split_calls"] = 0
            split_counters["cat_calls"] = 0
            split_counters["pieces_per_call"] = {}
            split_counters["piece_widths"] = {}
            split_counters["piece_contiguous"] = {}
        if chunk_counters is not None:
            chunk_counters["forward_calls"] = 0
            chunk_counters["back_calls"] = 0
            chunk_counters["clamped_calls"] = 0
            chunk_counters["forward_values"] = {}
            chunk_counters["back_values"] = {}
        start = time.perf_counter()
        out = vcd()
        wall = time.perf_counter() - start
        warm.append(wall)
        # Drained AFTER the wall is recorded, so the read-out's synchronize and
        # elapsed_time calls never land inside a timed number.
        readout = instrument.finish(devices)
        _merge_regions(region_totals, readout["regions"])
        pair_total += readout["event_pairs"]
        cap_hit = cap_hit or readout["event_cap_hit"]
        instrument.reset()
        record = dict(recon_index=repeat, wall_s=wall)
        record.update(busy.drain(devices))
        record.update(transfer.drain(devices))
        forward = readout["regions"]["forward_funnel"]
        record["bracket_ms_per_device"] = [
            float(forward["device_span_ms"].get(name, 0.0))
            for name in device_names]
        record["bracket_host_wall_s"] = float(forward["host_wall_s"])
        record["bracket_calls"] = int(forward["calls"])
        back = readout["regions"]["back_funnel"]
        record["back_bracket_ms_per_device"] = [
            float(back["device_span_ms"].get(name, 0.0))
            for name in device_names]
        record["gap_ms_per_device"] = [
            b - u for b, u in zip(record["bracket_ms_per_device"],
                                  record["busy_ms_per_device"])]
        record["probe_verify"] = verify()
        record["patch_verify"] = patch_verify() if patch_verify else None
        # The realized forward view batch, per device, keyed BY POSITION -- the
        # observer accumulates from the moment the probes went in, so at the
        # first timed pass this snapshot IS that pass and the witness can read
        # it.  Later passes read it cumulatively, which is what the summary
        # wants.
        record["view_batch_observed"] = {
            key: sorted(bucket.items()) for key, bucket in observed.items()}
        # THE MEMORY COLUMN: the per-device peak as it stands after this timed
        # reconstruction.  No reset between passes, so the series is a running
        # maximum and the first pass carries most of it; the point of reading
        # it per pass is to see whether a later pass pushes it higher.
        record["peak_bytes_per_device"] = peaks()
        if cone_counters is not None:
            record["cone_proto"] = dict(
                proto_calls=cone_counters["proto_calls"],
                delegated_trivial=cone_counters["delegated_trivial"],
                delegated_aligned=cone_counters["delegated_aligned"],
                expected_batches_per_owner=cone_counters["expected_batches"],
                batches_per_owner=list(cone_counters["batches"]),
                moves_per_owner=list(cone_counters["moves"]),
                cyl_height_hist={str(k): v for k, v
                                 in sorted(cone_counters["cyl_height"].items())},
                cyl_width_hist={str(k): v for k, v
                                in sorted(cone_counters["cyl_width"].items())},
                pixel_counts={str(k): v for k, v
                              in sorted(cone_counters["pixel_counts"].items())})
        if split_counters is not None:
            record["n1_split"] = dict(
                width=split_counters["width"],
                split_calls=split_counters["split_calls"],
                delegated_other_model=split_counters["delegated_other_model"],
                cat_calls=split_counters["cat_calls"],
                pieces_per_call={str(k): v for k, v in
                                 sorted(split_counters["pieces_per_call"].items())},
                piece_widths={str(k): v for k, v in
                              sorted(split_counters["piece_widths"].items())},
                piece_contiguous={str(k): v for k, v in
                                  sorted(split_counters["piece_contiguous"].items())})
        if chunk_counters is not None:
            record["view_chunk_patch"] = dict(
                chunk=chunk_counters["chunk"],
                forward_calls=chunk_counters["forward_calls"],
                back_calls=chunk_counters["back_calls"],
                clamped_calls=chunk_counters["clamped_calls"],
                forward_values={str(k): v for k, v in
                                sorted(chunk_counters["forward_values"].items())},
                back_values={str(k): v for k, v in
                             sorted(chunk_counters["back_values"].items())})
        per_recon.append(record)
        health.append(sample_gpu_health())

        # ── the value column ─────────────────────────────────────────────────
        checksums.append(float(np.sum(np.abs(out), dtype=np.float64)))
        if steps is None:
            steps = _sample_steps(out.shape)
        if repeat < 2:
            # Two samples: one for the cross-arm distances and a second so the
            # summary can state this arm's OWN pass-to-pass distance, which is
            # the floor every cross-arm number has to be read against (both
            # forward kernels accumulate with float atomics and are not
            # bit-reproducible).
            path = _sample_path(cfg["arm_id"], repeat)
            np.save(path, np.ascontiguousarray(
                out[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32))
            sample_paths.append(path)

        # ── the witnesses (trap 4), on the first timed reconstruction ────────
        if repeat == 0:
            _check_witnesses(result, record, device_names, n_owners,
                             lengths, num_slices, pixel_batch, band_len,
                             values_width, view_chunk,
                             expected_fwd_view_batch=(
                                 view_chunk or result.get("arm_fwd_view_batch")),
                             expect_fwd_kernel=expect_kernels[0])

    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_min"] = min(warm)
    result["vcd_warm_max"] = max(warm)
    result["vcd_warm_spread"] = ((max(warm) - min(warm))
                                 / statistics.median(warm)) if warm else None
    result["per_recon"] = per_recon
    result["device_names"] = device_names
    result["view_batch_observed_per_device"] = {
        k: sorted(v.items()) for k, v in observed.items()}
    result["busy_backend"] = busy.backend
    result["probe_verify_after"] = verify()
    result["patch_verify_after"] = patch_verify() if patch_verify else None
    detach_probes()
    detach_regions()
    if patch_detach:
        patch_detach()

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    # The pack a narrowed one-device piece pays INSIDE the timed body, measured
    # directly so it can be subtracted (see install_parallel_n1_values_split).
    # Run LAST: after every probe is detached and after the peak columns have
    # been read, because it allocates a volume-sized block of its own and the
    # peak counter is a running maximum that no reset may be spent on here.
    result["values_pack_ms"] = (
        measure_values_pack(model, torch, cuda, values_width)
        if values_width else None)

    result["realized_devices"] = realized_devices
    result["realized_n_devices"] = len(realized_devices)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = True          # fatal above if it were not

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (
            result["back_launch_keys_delta"] > 0
            and result["fwd_launch_keys_delta"] > 0)

    # ── the region read-out, in mg5's field names so the rows diff ───────────
    regions = {}
    for region in REGIONS:
        slot = region_totals.get(region, dict(calls=0, host_wall_s=0.0,
                                              device_span_ms={}, devices=[]))
        spans = slot["device_span_ms"]
        regions[region] = dict(
            calls=slot["calls"], host_wall_s=slot["host_wall_s"],
            device_span_ms=spans,
            device_span_max_ms=(max(spans.values()) if spans else 0.0),
            device_span_sum_ms=float(sum(spans.values())),
            devices=slot["devices"])
    result["regions"] = regions
    result["event_backend"] = instrument.backend
    result["event_pairs"] = pair_total
    result["event_cap_hit"] = cap_hit

    count = len(realized_devices)
    missing = []
    for region in REGIONS:
        if region in REGIONS_ABSENT_AT_N1 and count == 1:
            continue
        if regions[region]["host_wall_s"] <= 0.0:
            missing.append(f"{region}:host")
            continue
        if region in REGIONS_HOST_ONLY_AT_N1 and count == 1:
            continue
        if regions[region]["device_span_sum_ms"] <= 0.0:
            missing.append(f"{region}:device")
    result["region_nonzero_ok"] = not missing
    result["region_missing"] = missing

    passes = max(1, WARM_REPEATS)
    composed = result["vcd_warm"]
    for region in REGIONS:
        block = regions[region]
        result[f"{region}_wall_per_pass_s"] = block["host_wall_s"] / passes
        result[f"{region}_dev_span_max_per_pass_s"] = (
            block["device_span_max_ms"] / 1e3 / passes)
        result[f"{region}_calls"] = block["calls"]
        result[f"{region}_gap_per_pass_s"] = (
            result[f"{region}_wall_per_pass_s"]
            - result[f"{region}_dev_span_max_per_pass_s"])
    top = sum(regions[r]["host_wall_s"] for r in REGIONS
              if r not in NESTED_REGIONS)
    per_pass = top / passes
    result["region_wall_total_s"] = top
    result["region_wall_per_pass_s"] = per_pass
    result["region_remainder_per_pass_s"] = composed - per_pass
    result["region_remainder_frac"] = ((composed - per_pass) / composed
                                       if composed else None)
    result["reconcile_ok"] = (per_pass <= composed * (1.0 + RECONCILE_SLACK))
    result["forward_share_of_composed"] = (
        result["forward_funnel_wall_per_pass_s"] / composed if composed else None)

    result["recon_checksums"] = checksums
    result["recon_checksum"] = checksums[-1] if checksums else None
    result["recon_checksum_spread"] = (
        (max(checksums) - min(checksums)) / statistics.median(checksums)
        if len(checksums) > 1 and statistics.median(checksums) else None)
    result["value_sample_paths"] = sample_paths
    result["value_sample_steps"] = list(steps or ())
    result["recon_shape_out"] = list(out.shape)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result.update(summarize_arm(result))
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def _check_witnesses(result, record, device_names, n_owners, lengths,
                     num_slices, pixel_batch, band_len, values_width=None,
                     view_chunk=None, expected_fwd_view_batch=None,
                     expect_fwd_kernel=False):
    """The proof that this arm measured what its name says, run on the first
    timed reconstruction and fatal on any disagreement.

    Every expected number here comes from the arm's own parameters and from
    this file's re-derivation of the driver's balanced tiling -- never from
    calling the thing being checked."""
    funnel_calls = int(record["bracket_calls"])
    n_dev = len(device_names)
    checks = {}

    # -- shared: the instrument itself is on the path ------------------------
    missing = [name for name, calls
               in zip(device_names, record["busy_calls_per_device"])
               if calls <= 0]
    if missing:
        raise RuntimeError(
            f"the busy instrument recorded NO forward body calls on {missing} "
            f"in the first timed reconstruction.  The wrappers are not on the "
            f"path the driver takes, and every per-device number in this arm "
            f"would be empty.  verify={record['probe_verify']}")
    if not all(record["probe_verify"].values()):
        raise RuntimeError(
            f"the forward probes left the driver's path during the first timed "
            f"reconstruction: {record['probe_verify']}")
    if record.get("patch_verify") and not all(record["patch_verify"].values()):
        raise RuntimeError(
            f"the {result['patch']} patch left the driver's path during the "
            f"first timed reconstruction: {record['patch_verify']}")

    # -- shared, PARALLEL only: the realized forward view batch ---------------
    # The view batch is the other axis of one launch's atomic write slab, so it
    # has to be pinned on EVERY parallel arm and not only on the arms that
    # narrow it: an arm whose batch quietly moved would report a slab size it
    # never wrote.  Cone is excluded because its realized batch is capped by
    # the transient budget rather than by the constant at this cell (mg10's own
    # cone rows carry two values, 52 and 128), so a single-value check would
    # abort a healthy arm.
    if result.get("geometry") == "parallel" and expected_fwd_view_batch:
        observed = record.get("view_batch_observed") or {}
        fwd_seen = set()
        for key, pairs in observed.items():
            if key.startswith("fwd_dev"):
                fwd_seen.update(int(value) for value, _count in pairs)
        if not fwd_seen:
            raise RuntimeError(
                "no forward view batch was observed by position; the batch "
                "chooser's observer is not on the driver's path and the write "
                "slab this arm claims cannot be derived.")
        if fwd_seen != {int(expected_fwd_view_batch)}:
            raise RuntimeError(
                f"the realized forward view batch was {sorted(fwd_seen)} where "
                f"this arm expects {int(expected_fwd_view_batch)}"
                + (f" (the arm forces chunk {view_chunk})" if view_chunk else
                   " (the shipped chunk, which no arm here moves)")
                + ".  One launch's write slab is view batch x channels x "
                  "values columns, so a batch that is not the expected one "
                  "mislabels every slab number in this row.")
        checks["fwd_view_batch"] = int(expected_fwd_view_batch)
    if view_chunk is not None:
        patch = record.get("view_chunk_patch") or {}
        if not patch:
            raise RuntimeError(
                "the view-chunk arm recorded nothing from its batch chooser; "
                "Projectors._effective_view_batch is not the function the "
                "driver called.")
        if patch["forward_calls"] <= 0:
            raise RuntimeError(
                "the view-chunk arm's batch chooser saw ZERO forward calls; "
                "the forward bodies it identifies by position are not the "
                "ones the driver passed it.")
        fwd_values = {int(k) for k in patch["forward_values"]}
        if fwd_values != {int(view_chunk)}:
            raise RuntimeError(
                f"the batch chooser returned {sorted(fwd_values)} for forward "
                f"calls where this arm forces {view_chunk}.")
        if expect_fwd_kernel and patch["clamped_calls"]:
            raise RuntimeError(
                f"the clamp fired {patch['clamped_calls']} times on the KERNEL "
                f"path.  On that path the module constant supplies the nominal "
                f"and the clamp must have nothing to do; a firing clamp means "
                f"the constant seam moved and this row measures the clamp "
                f"instead of the change a library edit would make.")
        if result.get("back_chunk_after") != result.get("shipped_back_chunk"):
            raise RuntimeError(
                f"the back view chunk moved from "
                f"{result.get('shipped_back_chunk')} to "
                f"{result.get('back_chunk_after')}; this seam is forward-only "
                f"and the back projection is inside the same timed "
                f"reconstruction.")
        checks["view_chunk_forward_calls"] = patch["forward_calls"]
        checks["view_chunk_clamped_calls"] = patch["clamped_calls"]
        checks["view_chunk_back_values"] = patch["back_values"]

    if values_width is not None or (n_dev == 1 and result.get("geometry") ==
                                    "parallel" and band_len is None
                                    and pixel_batch is None):
        # ── EXPERIMENT 3's witnesses: the piece structure, twice over ────────
        # This branch serves BOTH the narrowed one-device arms and the
        # one-device control, because the control's claim ("the shipped
        # monolithic call ran") is the same claim with a piece count of one and
        # has to be proved just as hard.  There are two independent readings of
        # the same structure: the splitter's own count of pieces cut, and the
        # busy probe's count of the COLUMN COUNT of every values block the
        # kernel was handed.  The second is recorded by different code at a
        # different place, so a splitter that counted pieces it never projected
        # cannot pass.  The control has no splitter, so only the second exists
        # for it -- which is exactly what "no patch is installed" means.
        widths = [b for b in lengths]
        pieces = len(widths)
        want_widths = {}
        for value in widths:
            want_widths[value] = want_widths.get(value, 0) + funnel_calls
        # -- half one: single device, nothing fanned out or gathered ----------
        if n_dev != 1:
            raise RuntimeError(
                f"an experiment-3 arm settled on {n_dev} devices "
                f"({device_names}); the whole point of these arms is that the "
                f"device count is ONE while the width varies.")
        for name, count in (("band fan-outs", int(record["broadcast_calls"])),
                            ("column gathers", int(record["gather_calls"])),
                            ("cross-device copies", int(record["copy_count"])),
                            ("cross-device bytes", int(record["copy_bytes"]))):
            if count:
                raise RuntimeError(
                    f"a one-device arm recorded {count} {name}.  At a trivial "
                    f"placement the forward moves nothing, so this row is not "
                    f"the single-device measurement it claims to be.")
        # -- half two: the kernel really saw pieces of this width -------------
        seen = {}
        for hist in record["busy_value_cols_per_device"]:
            for key, count in hist.items():
                seen[int(key)] = seen.get(int(key), 0) + int(count)
        if set(seen) != set(want_widths):
            raise RuntimeError(
                f"the kernel was handed values blocks of widths "
                f"{sorted(seen)} where this arm's own tiling of "
                f"{num_slices} slices at width {values_width or num_slices} "
                f"says {sorted(want_widths)}.  Either the split did not "
                f"engage, or part of the forward took another path.")
        launches = sum(seen.values())
        if launches != int(record["busy_calls_per_device"][0]):
            raise RuntimeError(
                f"the width histogram covers {launches} launches against "
                f"{record['busy_calls_per_device'][0]} timed body calls; the "
                f"two are recorded by the same probe and must agree.")
        # Every piece is projected in the same number of view batches, so the
        # launch count has to divide evenly into (funnel calls x pieces).
        if launches % max(1, funnel_calls * pieces):
            raise RuntimeError(
                f"{launches} launches do not divide into {funnel_calls} "
                f"forward funnel calls x {pieces} pieces.  Either the walk is "
                f"not the one this row claims, or the view batch varied "
                f"between calls -- at mg10's cell the transient budget's cap "
                f"sits far above the nominal 128 at every width here, so it "
                f"does not, but a larger cell could change that "
                f"(view_batch_observed_per_device on the row says).")
        per_piece = launches // max(1, funnel_calls * pieces)
        for value, count in want_widths.items():
            if seen.get(value) != count * per_piece:
                raise RuntimeError(
                    f"width {value} was handed to the kernel "
                    f"{seen.get(value)} times where this arm's tiling says "
                    f"{count * per_piece} ({count} pieces x {per_piece} view "
                    f"batches each).")
        # -- half three: the splitter's own count, for a narrowed arm ---------
        split = record.get("n1_split")
        if values_width is None:
            if split is not None:
                raise RuntimeError(
                    "the one-device CONTROL arm recorded a values split; it "
                    "must run the shipped monolithic call")
            if view_chunk is None and record.get("patch_verify") is not None:
                raise RuntimeError("a control arm must carry no patch")
            if pieces != 1 or widths[0] != num_slices:
                raise RuntimeError(
                    f"the one-device control's own tiling says {pieces} x "
                    f"{widths[0]}; the shipped call is one piece of "
                    f"{num_slices} slices.")
        else:
            if not split:
                raise RuntimeError(
                    "the one-device values splitter recorded nothing: "
                    "Projectors._sparse_forward_project_single_device is not "
                    "the function the driver called.")
            if split["split_calls"] != funnel_calls:
                raise RuntimeError(
                    f"the splitter served {split['split_calls']} calls against "
                    f"{funnel_calls} forward funnel calls.  Both of the "
                    f"one-device entries -- the recon's plain-tensor forward "
                    f"and the VCD loop's one-shard forward -- must land on it, "
                    f"or part of this row is the shipped monolithic call.")
            if split["delegated_other_model"]:
                raise RuntimeError(
                    f"the splitter handed back {split['delegated_other_model']} "
                    "calls from another model; this arm builds one model and "
                    "that number must be zero.")
            got_pieces = {int(k): v for k, v in split["pieces_per_call"].items()}
            if got_pieces != {pieces: funnel_calls}:
                raise RuntimeError(
                    f"the splitter cut {got_pieces} where this arm's tiling of "
                    f"{num_slices} slices at width {values_width} says "
                    f"{{{pieces}: {funnel_calls}}}.")
            got_widths = {int(k): v for k, v in split["piece_widths"].items()}
            if got_widths != want_widths:
                raise RuntimeError(
                    f"the splitter cut widths {got_widths} where this arm's "
                    f"tiling says {want_widths}.")
            if pieces == 1:
                raise RuntimeError(
                    f"width {values_width} caps back onto the whole "
                    f"{num_slices}-slice volume, so this arm IS the control "
                    f"under another name; run it as p1_shipped or narrow it.")
        checks["one_device_ok"] = True
        checks["pieces_per_forward_call"] = pieces
        checks["piece_widths"] = {str(k): v for k, v in sorted(want_widths.items())}
        checks["launches_per_piece"] = per_piece
        checks["kernel_saw_widths"] = {str(k): v for k, v in sorted(seen.items())}
        if values_width is not None:
            checks["splitter_calls"] = record["n1_split"]["split_calls"]
            checks["piece_contiguous"] = record["n1_split"]["piece_contiguous"]
    elif pixel_batch is not None:
        # ── EXPERIMENT 2's witness: the column gather ────────────────────────
        proto = record.get("cone_proto") or {}
        if not proto:
            raise RuntimeError(
                "the cone column-gather prototype recorded nothing: the "
                "instance method was not the one the funnel called.")
        if proto["proto_calls"] <= 0:
            raise RuntimeError(
                "the cone column-gather prototype ran ZERO times while the "
                f"forward funnel was entered {funnel_calls} times.  Something "
                "else is serving _sparse_forward_project_sharded.")
        if proto["proto_calls"] != funnel_calls:
            raise RuntimeError(
                f"the prototype ran {proto['proto_calls']} times against "
                f"{funnel_calls} forward funnel calls; they must agree, or "
                "part of the forward took another path.")
        if proto["delegated_trivial"] or proto["delegated_aligned"]:
            raise RuntimeError(
                f"the prototype DELEGATED on this arm "
                f"(trivial={proto['delegated_trivial']}, "
                f"aligned={proto['delegated_aligned']}): the shipped banded "
                "branch ran and this row is not a column-gather measurement.")
        expected = int(proto["expected_batches_per_owner"])
        if expected <= 0:
            raise RuntimeError("the prototype expected zero column batches")
        for name, got in zip(device_names, proto["batches_per_owner"]):
            if got != expected:
                raise RuntimeError(
                    f"view-owner {name} walked {got} column batches where the "
                    f"arm's own arithmetic says {expected} "
                    f"(pixel batch {pixel_batch}).")
        for name, got in zip(device_names, proto["moves_per_owner"]):
            if got != expected * n_dev:
                raise RuntimeError(
                    f"view-owner {name} moved {got} pieces where a full-height "
                    f"gather needs {expected * n_dev} "
                    f"({expected} batches x {n_dev} slice-owners).")
        heights = {int(k) for k in proto["cyl_height_hist"]}
        if heights != {num_slices}:
            raise RuntimeError(
                f"the assembled cylinders were {sorted(heights)} slices tall; "
                f"a full-height gather must be exactly {num_slices}.  This is "
                "the check that separates a column gather from a band by "
                "another name.")
        widths = {int(k) for k in proto["cyl_width_hist"]}
        if max(widths) > pixel_batch:
            raise RuntimeError(
                f"a column batch was {max(widths)} pixels wide, more than the "
                f"arm's batch of {pixel_batch}.")
        if int(record["broadcast_calls"]) != 0:
            raise RuntimeError(
                f"the shipped band fan-out fired {record['broadcast_calls']} "
                "times inside a column-gather arm.  Part of this arm ran the "
                "banded branch and its time is a mixture of the two shapes.")
        if int(record["gather_calls"]) != expected * n_dev:
            raise RuntimeError(
                f"the gather timer saw {record['gather_calls']} column batches "
                f"against {expected * n_dev} walked over all view-owners.")
        checks["column_batches_per_owner"] = expected
        checks["cylinder_height"] = num_slices
    else:
        # ── EXPERIMENT 1's witness (and every control's) : the band walk ─────
        # One fan-out per (forward funnel call, slice owner, sub-band), each
        # carrying a band whose width comes from this file's own tiling.  At
        # ONE device there is no band walk at all -- the placement is trivial
        # and the driver hands the whole cylinder to the single-device path --
        # so the expectation there is zero of everything, which is what proves
        # the value anchor really took the path it is the anchor for.
        expected_calls = (funnel_calls * n_owners * len(lengths)
                          if n_dev > 1 else 0)
        got_calls = int(record["broadcast_calls"])
        if n_dev == 1 and got_calls != 0:
            raise RuntimeError(
                f"the one-device anchor fanned out {got_calls} bands; at a "
                "trivial placement the driver must take the single-device "
                "path and move nothing.")
        if n_dev > 1 and got_calls != expected_calls:
            raise RuntimeError(
                f"the band fan-out fired {got_calls} times where this arm's "
                f"arithmetic says {expected_calls} "
                f"({funnel_calls} funnel calls x {n_owners} slice-owners x "
                f"{len(lengths)} sub-bands).  Either the sub-band patch did "
                f"not engage or the driver's walk is not what this harness "
                f"models.")
        if n_dev > 1:
            want = {}
            for length in lengths:
                want[length] = want.get(length, 0) + funnel_calls * n_owners
            got = {int(k): v for k, v in record["band_cols_hist"].items()}
            if got != want:
                raise RuntimeError(
                    f"the fanned-out band widths were {got} where this arm's "
                    f"tiling says {want}.  The walk is not the one this row "
                    f"claims to measure.")
        if band_len is None and len(lengths) != 1:
            raise RuntimeError(
                "a control arm must walk ONE band per slice-owner; this file's "
                f"tiling says {len(lengths)}, so the plan is inconsistent.")
        if band_len is None and view_chunk is None \
                and record.get("patch_verify") is not None:
            # A view-chunk arm walks the shipped band and carries a patch on
            # the OTHER axis, so it is not a control and this does not apply.
            raise RuntimeError("a control arm must carry no patch")
        if int(record["gather_calls"]) != 0:
            raise RuntimeError(
                f"the column-gather timer fired {record['gather_calls']} times "
                "on a banded arm; the prototype is installed where it should "
                "not be.")
        checks["fan_outs_expected"] = expected_calls
        checks["fan_outs_seen"] = got_calls
        checks["band_widths"] = record["band_cols_hist"]
    result["witnesses"] = checks


def parallel_control_token(n_dev):
    """The token of the unpatched parallel arm at this device count.

    One place, because three different readers look it up -- the value table,
    the experiment-1 table and the design note's slots -- and experiment 3's
    one-device control does not follow the `p<n>_shard` pattern (it is named
    for what it is, the shipped call, not for a shard it does not have)."""
    return "p1_shipped" if n_dev == 1 else f"p{n_dev}_shard"


def _hist_mean(hist):
    """The count-weighted mean of a {value: how many} histogram, with string
    keys as the jsonl carries them.  ``None`` for an empty histogram."""
    total = weight = 0
    for key, count in (hist or {}).items():
        total += int(key) * int(count)
        weight += int(count)
    return (total / weight) if weight else None


def summarize_arm(result):
    """Per device: the bracket, the busy sum, the call count, the per-launch
    time, the gap, and the two NORMALIZED cost columns -- each the median over
    the timed reconstructions -- plus the transfer totals.  Medians, because a
    single reconstruction can carry a scheduling artifact.

    THE NORMALIZED COLUMNS are what the decision is read off, and they exist
    because a raw per-launch time falls whenever the launch is given less to
    do, which says nothing.  ms_per_slice divides the per-launch time by the
    mean column count of the values blocks the kernel was actually handed --
    the sub-band width for parallel -- and ms_per_kpix divides it by their mean
    pixel count in thousands, which is what the cone batch sweep moves.  A
    shape is worth taking only where its normalized cost BEATS the shipped
    one's; mg9's own readings (0.041 ms per slice at a 1008-slice band against
    0.082 and 0.085 at 504 and 252) say the widest band was the most efficient
    per slice, so a null result is a real possible outcome here and these
    columns are what make it legible."""
    names = result["device_names"]
    passes = result["per_recon"]
    if not passes:
        return dict(per_device=[], transfer=None)

    def median(values):
        return float(statistics.median(values)) if values else None

    per_device = []
    for index, name in enumerate(names):
        bracket = [p["bracket_ms_per_device"][index] / 1e3 for p in passes]
        busy = [p["busy_ms_per_device"][index] / 1e3 for p in passes]
        calls = [p["busy_calls_per_device"][index] for p in passes]
        host = [p["busy_host_s_per_device"][index] for p in passes]
        back = [p["back_bracket_ms_per_device"][index] / 1e3 for p in passes]
        copy_in = [p["copy_device_ms_by_dst"].get(name, 0.0) / 1e3
                   for p in passes]
        copy_out = [p["copy_device_ms_by_src"].get(name, 0.0) / 1e3
                    for p in passes]
        peak = [(p.get("peak_bytes_per_device") or [0] * len(names))[index]
                for p in passes]
        bracket_med, busy_med = median(bracket), median(busy)
        calls_med = median(calls)
        per_launch = ((busy_med * 1e3 / calls_med)
                      if busy_med and calls_med else None)
        cols_hist = passes[0]["busy_value_cols_per_device"][index]
        pix_hist = passes[0].get("busy_value_pixels_per_device",
                                 [{}] * len(names))[index]
        mean_cols, mean_pixels = _hist_mean(cols_hist), _hist_mean(pix_hist)
        per_device.append(dict(
            device=name,
            bracket_span_s=bracket_med,
            busy_sum_s=busy_med,
            busy_calls=calls_med,
            per_launch_ms=per_launch,
            mean_cols_per_launch=mean_cols,
            mean_pixels_per_launch=mean_pixels,
            ms_per_slice=(per_launch / mean_cols
                          if per_launch and mean_cols else None),
            ms_per_kpix=(per_launch / (mean_pixels / 1000.0)
                         if per_launch and mean_pixels else None),
            busy_host_s=median(host),
            gap_s=(bracket_med - busy_med),
            busy_frac_of_bracket=(busy_med / bracket_med if bracket_med else None),
            back_bracket_span_s=median(back),
            copy_device_in_s=median(copy_in),
            copy_device_out_s=median(copy_out),
            peak_bytes=median(peak),
            value_cols=cols_hist,
            value_pixels=pix_hist,
            device_mismatch=sum(p["busy_device_mismatch_per_device"][index]
                                for p in passes)))

    bytes_med = median([p["copy_bytes"] for p in passes])
    dev_ms_med = median([p["copy_device_ms_total"] for p in passes])
    transfer = dict(
        broadcast_calls_per_recon=median([p["broadcast_calls"] for p in passes]),
        broadcast_host_s_per_recon=median([p["broadcast_host_wall_s"]
                                           for p in passes]),
        gather_calls_per_recon=median([p["gather_calls"] for p in passes]),
        gather_host_s_per_recon=median([p["gather_host_wall_s"]
                                        for p in passes]),
        copy_count_per_recon=median([p["copy_count"] for p in passes]),
        copy_noop_count_per_recon=median([p["copy_noop_count"] for p in passes]),
        copy_bytes_per_recon=bytes_med,
        copy_device_s_per_recon=(dev_ms_med / 1e3 if dev_ms_med is not None
                                 else None),
        copy_gb_per_s=((bytes_med / 1e9) / (dev_ms_med / 1e3)
                       if bytes_med and dev_ms_med else None),
        copy_measurement=passes[0].get("copy_measurement"),
        dev2dev_safe=result.get("dev2dev_safe"))
    # A device-side copy time of ~0 at two or more devices is what a bracket on
    # the WRONG STREAM looks like, so it is flagged rather than reported as a
    # fast copy.
    transfer["copy_device_plausible"] = (
        None if len(names) < 2 or not result.get("cuda")
        else bool(dev_ms_med and dev_ms_med > 0.0))
    return dict(per_device=per_device, transfer=transfer)


def generator_worker(cfg):
    """Build ONE shared sinogram per geometry: phantom -> sinogram -> .npy, plus
    its md5 sidecar.  Every arm at that geometry reconstructs THAT array, so no
    arm's timing or value carries an input difference.  Pinned to one device so
    the generator cannot itself become a multi-device run.  (mg9 deleted its
    own copies at the end of its job, so these are regenerated here.)"""
    import numpy as np

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    devices = cfg.get("cpu_devices") or [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=devices)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _sino_path(geometry, cell)
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(geometry, cell), "w") as handle:
        handle.write(digest + "\n")
    return dict(cfg, framework="torch", role="generator", path=path,
                sino_md5=digest, sinogram_shape=list(sinogram.shape),
                recon_shape=list(recon_shape),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


# ── the value read-out ────────────────────────────────────────────────────────
def _rel_distance(path_a, path_b):
    """Relative L2 distance between two strided reconstruction samples, and the
    largest single-voxel relative difference.  ``None`` when either sample is
    missing (a truncated job, or an arm that failed)."""
    import numpy as np

    if not path_a or not path_b:
        return None
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
    a = np.load(path_a).astype(np.float64)
    b = np.load(path_b).astype(np.float64)
    if a.shape != b.shape:
        return None
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return None
    diff = a - b
    scale = float(np.max(np.abs(b)))
    return dict(rel_l2=float(np.linalg.norm(diff) / denom),
                max_abs=float(np.max(np.abs(diff))),
                max_rel_of_peak=(float(np.max(np.abs(diff)) / scale)
                                 if scale else None))


def _checksum_stats(row):
    """The three per-reconstruction checksums of one arm, reduced: the median,
    and the REPEAT-TO-REPEAT spread as a fraction of it.  That spread is the
    run-to-run noise floor every cross-arm checksum distance is read against;
    both forward kernels accumulate with float atomics, so it is not zero on
    the GPU even for two runs of the identical configuration."""
    values = row.get("recon_checksums") or []
    if not values:
        return None, None
    mid = statistics.median(values)
    spread = (max(values) - min(values)) / mid if mid else None
    return mid, spread


def _checksum_distance(row, other):
    """Relative distance between two arms' median checksums."""
    a, _ = _checksum_stats(row)
    b, _ = _checksum_stats(other or {})
    if a is None or b is None or not b:
        return None
    return abs(a - b) / abs(b)


def value_table(rows):
    """Every distance the design note's value gate is priced on, computed two
    ways so neither has to be trusted alone.

    The SAMPLE distances are relative L2 between strided samples of the
    reconstructions -- sensitive to a local change that a global sum could
    cancel.  The CHECKSUM distances are relative differences of the whole
    volume's sum of absolute values, computed per timed reconstruction, so they
    can be recomputed from the rows alone without the samples.

    For each arm: its OWN repeat-to-repeat distance (the floor -- both forward
    kernels accumulate with float atomics and are not bit-reproducible), its
    distance to the control at the same device count, and, for cone, its
    distance to the one-device anchor.  A cross-arm distance at the level of an
    arm's own floor is the strongest statement this instrument can make."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by_token = {r.get("token"): r for r in live}
    anchor = by_token.get("c1")
    out = []
    for row in live:
        token = row.get("token")
        geometry, n_dev = row.get("geometry"), row.get("n_dev")
        samples = row.get("value_sample_paths") or []
        own = _rel_distance(samples[1], samples[0]) if len(samples) > 1 else None
        control_token = (parallel_control_token(n_dev) if geometry == "parallel"
                         else "c%d_banded" % n_dev)
        control = by_token.get(control_token)
        vs_control = None
        if control is not None and control is not row:
            vs_control = _rel_distance(
                (samples or [None])[0],
                ((control.get("value_sample_paths") or [None]) + [None])[0])
        vs_anchor = None
        if geometry == "cone" and anchor is not None and anchor is not row:
            vs_anchor = _rel_distance(
                (samples or [None])[0],
                ((anchor.get("value_sample_paths") or [None]) + [None])[0])
        median_checksum, repeat_spread = _checksum_stats(row)
        entry = dict(token=token, geometry=geometry, n=n_dev,
                     patch=row.get("patch"),
                     band=(row.get("requested_band_len")
                           if row.get("requested_band_len") is not None
                           else row.get("requested_values_width")),
                     # What this arm changed, in words: the value slots print
                     # one line per arm and "band None" would name nothing.
                     label=(
                         f"sub-band {row.get('requested_band_len')}"
                         if row.get("requested_band_len") else
                         (f"values width {row.get('requested_values_width')}"
                          if row.get("requested_values_width") else
                          (f"view chunk {row.get('requested_view_chunk')}"
                           if row.get("requested_view_chunk") else
                           "the shipped walk"))),
                     batch=row.get("pixel_batch"),
                     checksums=row.get("recon_checksums"),
                     checksum_median=median_checksum,
                     # repeat-to-repeat: the run-to-run floor, from this arm's
                     # own three timed reconstructions.
                     checksum_repeat_spread=repeat_spread,
                     checksum_spread=row.get("recon_checksum_spread"),
                     # band-to-band / shape-to-shape, from the same checksums.
                     checksum_vs_control=(
                         _checksum_distance(row, control)
                         if control is not None and control is not row
                         else None),
                     checksum_vs_anchor=(
                         _checksum_distance(row, anchor)
                         if geometry == "cone" and anchor is not None
                         and anchor is not row else None),
                     own_pass_to_pass=own,
                     vs_control=vs_control, vs_control_token=control_token,
                     vs_anchor=vs_anchor)
        # The brief's parallel expectation, recorded rather than asserted: with
        # one producing sub-band per detector row the walk is order-preserving
        # at the DRIVER level, so a patched parallel checksum "should" match its
        # control -- but the kernel's own atomics are not bit-reproducible, so
        # the honest test is against this arm's pass-to-pass floor, not zero.
        # The one-device values split carries the same expectation for the same
        # reason: each piece produces its own detector rows and the pieces are
        # concatenated, so nothing is summed and nothing is reordered.  So does
        # the view chunk: view batches write disjoint view slices of the same
        # output, so which pixels accumulate into a given element is unchanged
        # and only the launch grouping moves.
        if geometry == "parallel" and set(
                (row.get("patch") or "none").split("+")) & {
                    "parallel_subband", "parallel_n1_values_split",
                    "parallel_view_chunk"}:
            cs = row.get("recon_checksums") or []
            ctrl_cs = (control or {}).get("recon_checksums") or []
            entry["checksum_equals_control_exactly"] = bool(
                cs and ctrl_cs and cs[0] == ctrl_cs[0])
        out.append(entry)
    return out


# ── the runner (mg5's / mg9's subprocess pattern) ─────────────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.

    An arm pins ONLY through MBIRTORCH_NUM_DEVICES, which keeps the model on the
    automatic branch where the preflight still runs; an explicit
    configure_devices call would take the explicit branch and get no preflight,
    so the two are not interchangeable and mixing them would make the arms
    incomparable with mg5's and mg9's."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg10_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg10_{cfg['arm_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    start = time.perf_counter()
    proc = subprocess.run([TORCH_PYTHON, "-u", os.path.abspath(__file__),
                           "_worker", cfg_path, out_path], env=arm_env(cfg))
    subprocess_wall = time.perf_counter() - start
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(cfg, error=f"worker exited {proc.returncode}")
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    row["subprocess_wall_s"] = subprocess_wall
    return row


def build_plan(arms):
    """The generator arms, then the measured arms in declared order."""
    plan = []
    for geometry in sorted({a["geometry"] for a in arms}):
        cell = cell_for(geometry)
        gen = dict(framework="torch", arm_class="generator", geometry=geometry,
                   cell=list(cell), n_dev=None, token=f"gen_{geometry}",
                   band_len=None, pixel_batch=None, values_width=None,
                   view_chunk=None,
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if SMOKE and DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        plan.append(gen)
    measured = []
    for arm in arms:
        cell = cell_for(arm["geometry"])
        if arm["band_len"] is not None:
            tag = f"band{arm['band_len']:04d}"
        elif arm.get("values_width") is not None:
            # Named for the width ACTUALLY used, which is the smoke's
            # substituted width on the smoke and the asked width otherwise --
            # the arm_id should say what the arm did, while the token stays
            # stable across cells so MG10_ARMS names the same arms everywhere.
            tag = f"band{arm['values_width']:04d}"
        elif arm.get("view_chunk") is not None:
            tag = f"vc{arm['view_chunk']:03d}"
        elif arm["pixel_batch"] is not None:
            tag = f"cols{arm['pixel_batch']:05d}"
        elif arm["role"] == "anchor" and arm["geometry"] == "cone":
            tag = "anchor"
        else:
            tag = "shipped"
        cfg = dict(framework="torch", arm_class="instrument",
                   geometry=arm["geometry"], cell=list(cell),
                   n_dev=arm["n_dev"], token=arm["token"], role=arm["role"],
                   band_len=arm["band_len"], pixel_batch=arm["pixel_batch"],
                   values_width=arm.get("values_width"),
                   view_chunk=arm.get("view_chunk"),
                   arm_id=f"{arm['geometry']}_{cell[0]}_n{arm['n_dev']}_{tag}")
        if SMOKE and DEVICE != "cuda":
            # SMOKE ONLY: virtual cpu devices, so the n>1 wiring -- the band
            # fan-out, the prototype's gather, the per-device workers, the
            # witnesses -- is exercised without CUDA.  The env pin is CUDA-only,
            # so this pins by device LIST and says so on the row.
            cfg["cpu_devices"] = [DEVICE] * arm["n_dev"]
        measured.append(cfg)
    return plan, measured


# ── the summary ───────────────────────────────────────────────────────────────
def _fmt(value, spec, dash="-"):
    """``format(value, spec)``, with a missing value rendered as a dash padded
    to the SAME width -- an unpadded dash shifts every column to its right."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    width = ""
    for char in spec:
        if char.isdigit():
            width += char
        else:
            break
    return f"{dash:>{int(width) if width else 1}}"


DISCRIMINATION_RULE = (
    "  THE RULE, stated before the numbers so it cannot be fitted to them.\n"
    "    The cost per slice at ONE device with a narrowed values block is the\n"
    "    reading.  Two anchors bracket it: 0.041 ms per slice is what one "
    "device costs at the\n"
    "    full 1008-slice width, and 0.084 is what two and four devices cost at "
    "EVERY width they\n"
    "    were tried at (63 through 504).\n"
    "      * near 0.041 at every narrowed width -> the width is innocent, and "
    "the doubling\n"
    "        belongs to the DEVICE COUNT (hypothesis a).\n"
    "      * near 0.084 at every narrowed width -> the device count is "
    "innocent, and the\n"
    "        doubling belongs to the KERNEL WIDTH (hypothesis b).\n"
    "      * anything else -- a reading that climbs or falls across the widths, "
    "or sits\n"
    "        between the two anchors -- means neither story is the whole one "
    "and the shape of\n"
    "        the curve is the finding.\n"
    "    THE SECOND QUESTION, width 63.  At two devices width 63 was the only "
    "sub-band that\n"
    "    beat its control, by 9.5 percent, and it is the only width whose "
    "per-launch atomic\n"
    "    write target fits inside an H100's 50 MB L2 (128 views x 63 rows x "
    "1024 channels x\n"
    "    4 B = 33 MB; 126 rows is 66 MB, just over).  If width 63 also wins at "
    "one device, the\n"
    "    win is about the cache and not about sharding; if it does not, it is "
    "about sharding.\n"
    "    Both readings are printed above.  No verdict is printed: the "
    "attribution is analysis.")


READING_TEXT = (
    "  HOW TO READ THE PARALLEL SWEEP.  Sub-banding multiplies the launch "
    "count by the number of\n"
    "    sub-bands, so a band half as wide must cost LESS than half as much "
    "per launch to pay for\n"
    "    itself.  The break-even column below is exactly that test: it is the "
    "arm's per-device busy\n"
    "    time over the control's, and a remedy exists only where it is below "
    "1.00.\n"
    "  HOW TO READ THE CONE SWEEP.  Compare each prototype arm's per-device "
    "bracket and peak with\n"
    "    the banded control at the SAME device count.  mbirjax measured its "
    "own column-gather at\n"
    "    about 2x faster for about 10 percent more transient memory; neither "
    "number is ours.\n"
    "  HOW TO READ THE VALUE COLUMNS.  Both forward kernels accumulate with "
    "float atomics and are\n"
    "    not bit-reproducible, so every cross-arm distance is reported beside "
    "the same arm's own\n"
    "    pass-to-pass distance.  A cross-arm distance at that floor is the "
    "strongest statement\n"
    "    available; a distance far above it is a real change in the computed "
    "value.\n"
    "  No verdict is printed here: the attribution is analysis, not a harness "
    "output.")


def print_arm_table(row):
    """The per-device table for one arm, in plain English."""
    geometry, n_dev = row.get("geometry"), row.get("n_dev")
    parts = set((row.get("patch") or "none").split("+"))
    if "parallel_subband" in parts:
        shape = (f"sub-band {row.get('requested_band_len')} -> walked as "
                 f"{row.get('sub_bands_per_owner')} x "
                 f"{row.get('realized_band_len')}")
    elif "parallel_n1_values_split" in parts:
        shape = (f"values block cut into {row.get('sub_bands_per_owner')} x "
                 f"{row.get('realized_band_len')} slices "
                 f"(asked {row.get('requested_values_width')})")
    elif "cone_column_gather" in parts:
        shape = (f"column gather, {row.get('pixel_batch')} pixels x "
                 f"{row.get('num_slices')} slices")
    elif n_dev == 1:
        shape = (f"shipped call, the whole {row.get('num_slices')}-slice "
                 f"values block in one piece")
    else:
        shape = f"shipped walk, band = the whole {row.get('slices_per_dev')}-slice shard"
    if "parallel_view_chunk" in parts:
        shape += (f", forward view chunk {row.get('requested_view_chunk')} "
                  f"(shipped {row.get('shipped_fwd_chunk')})")
    print(f"\n--- {row.get('arm_id')}: {geometry} {row['cell'][0]}, {n_dev} "
          f"device(s), {shape} (median of "
          f"{len(row.get('per_recon') or [])} timed reconstructions) ---")
    print(f"   {'device':>10}{'bracket_s':>11}{'busy_s':>9}{'per_launch_ms':>15}"
          f"{'ms_per_slice':>14}{'ms_per_kpix':>13}{'calls':>8}{'busy/brk':>10}"
          f"{'gap_s':>8}{'peak_GB':>9}{'copy_in_s':>11}{'copy_out_s':>12}")
    for entry in row.get("per_device") or []:
        print(f"   {entry['device']:>10}"
              f"{_fmt(entry['bracket_span_s'], '11.2f')}"
              f"{_fmt(entry['busy_sum_s'], '9.2f')}"
              f"{_fmt(entry['per_launch_ms'], '15.2f')}"
              f"{_fmt(entry['ms_per_slice'], '14.4f')}"
              f"{_fmt(entry['ms_per_kpix'], '13.4f')}"
              f"{_fmt(entry['busy_calls'], '8.0f')}"
              f"{_fmt(entry['busy_frac_of_bracket'], '10.2f')}"
              f"{_fmt(entry['gap_s'], '8.2f')}"
              f"{_fmt((entry['peak_bytes'] or 0) / 2 ** 30, '9.2f')}"
              f"{_fmt(entry['copy_device_in_s'], '11.2f')}"
              f"{_fmt(entry['copy_device_out_s'], '12.2f')}")
    print("   bracket_s     = the whole forward projection region, this "
          "device's event span.")
    print("   busy_s        = the same device's time inside the per-call "
          "brackets around the")
    print("                   projection body: what it spent computing.")
    print("   per_launch_ms = busy_s / calls.")
    print("   ms_per_slice  = per_launch_ms / the mean column count of the "
          "values blocks the kernel")
    print("                   was handed (the sub-band width).  ms_per_kpix "
          "is the same over their")
    print("                   mean pixel count in thousands.  These are the "
          "columns a shape is")
    print("                   judged on: a narrower launch always has a "
          "smaller per_launch_ms.")
    print("   peak_GB       = torch.cuda.max_memory_allocated on this device "
          "after the last")
    print("                   timed reconstruction (no calibration mode; the "
          "counter is untouched).")
    cols = {}
    for entry in row.get("per_device") or []:
        for key, count in (entry.get("value_cols") or {}).items():
            cols[key] = cols.get(key, 0) + count
    if cols:
        print(f"   the column count of every values block the kernel was "
              f"handed: {cols}")
    if row.get("witnesses"):
        print(f"   witnesses: {row['witnesses']}")
    if row.get("width_seam"):
        print(f"   the width seam: {row['width_seam']}")
    if row.get("values_pack_ms") is not None:
        print(f"   the strided-to-contiguous pack this width costs INSIDE the "
              f"timed body: {row['values_pack_ms']:.3f} ms per launch")
        print("      (the kernel body calls .contiguous() on its values "
              "argument, and a narrowed piece")
        print("      of the volume is a strided view.  The two-device driver "
              "pays the same on every band")
        print("      it projects from its own shard.  Subtract it from "
              "per_launch_ms for the pure kernel.)")
    xfer = row.get("transfer") or {}
    if xfer:
        rate = xfer.get("copy_gb_per_s")
        rate_text = f" = {rate:.2f} GB/s" if rate else ""
        print(f"   cross-device transfer per reconstruction: "
              f"{_fmt(xfer.get('broadcast_calls_per_recon'), '.0f')} band "
              f"fan-outs (host wall "
              f"{_fmt(xfer.get('broadcast_host_s_per_recon'), '.2f')} s), "
              f"{_fmt(xfer.get('gather_calls_per_recon'), '.0f')} column "
              f"gathers (host wall "
              f"{_fmt(xfer.get('gather_host_s_per_recon'), '.2f')} s),")
        print(f"      {_fmt(xfer.get('copy_count_per_recon'), '.0f')} copies "
              f"({_fmt(xfer.get('copy_noop_count_per_recon'), '.0f')} to the "
              f"piece's own device, which are free), device time on the source "
              f"stream {_fmt(xfer.get('copy_device_s_per_recon'), '.2f')} s "
              f"for {_fmt((xfer.get('copy_bytes_per_recon') or 0) / 1e9, '.2f')}"
              f" GB{rate_text}")
        if xfer.get("copy_device_plausible") is False:
            print("      WARNING: the device-side copy time reads ~0 at more "
                  "than one device.  That is")
            print("      what a bracket on a stream the copies do not use "
                  "looks like; read the host wall")
            print("      instead and re-check the stream determination before "
                  "ruling on these numbers.")
    anchor = MG9_ANCHOR.get((geometry, n_dev))
    if anchor and row.get("patch") == "none":
        print(f"   mg9's anchor at this point: bracket {anchor['bracket_s']} s, "
              f"busy {anchor['busy_s']} s, "
              f"{anchor['calls'] if anchor['calls'] else '-'} calls, composed "
              f"{anchor['composed_s']} s, peak {anchor['peak_gb']} GB.")
    if row.get("view_batch_observed_per_device"):
        print(f"   realized forward view batch, per device, keyed BY POSITION: "
              f"{row['view_batch_observed_per_device']}")


def _observed_fwd_view_batch(row):
    """The realized forward view batch this arm ran at, read off the observer's
    per-position record.  A single value at every parallel arm (the witness
    asserts it); the largest is taken if a future cell ever produces more."""
    seen = set()
    for key, pairs in (row.get("view_batch_observed_per_device") or {}).items():
        if key.startswith("fwd_dev"):
            for value, _count in pairs:
                seen.add(int(value))
    return max(seen) if seen else None


def _write_slab_bytes(view_batch, band_cols, num_channels):
    """One forward launch's atomic write target: the zeroed output plane the
    parallel kernel accumulates into is (view batch, channels, values columns)
    float32 (triton_parallel.py line 444).  This is the number the L2 reading
    turns on -- an H100's L2 is 50 MB."""
    if not (view_batch and band_cols and num_channels):
        return None
    return int(view_batch) * int(num_channels) * int(round(band_cols)) * 4


def _arm_series_entry(row):
    """The one-line reduction of an arm: the LARGEST reading over its devices,
    because the reconstruction waits for its slowest device."""
    entries = row.get("per_device") or []
    if not entries:
        return None
    bracket = max((e["bracket_span_s"] or 0.0) for e in entries)
    busy = max((e["busy_sum_s"] or 0.0) for e in entries)
    calls = max((e["busy_calls"] or 0) for e in entries)
    peak = max((e["peak_bytes"] or 0) for e in entries)
    per_launch = (busy * 1e3 / calls) if calls else None
    mean_cols = _hist_mean(entries[0].get("value_cols"))
    mean_pixels = _hist_mean(entries[0].get("value_pixels"))
    xfer = row.get("transfer") or {}
    fwd_vb = _observed_fwd_view_batch(row)
    channels = (row.get("cell") or [None, None, None])[2]
    slab = _write_slab_bytes(fwd_vb, mean_cols, channels)
    return dict(token=row.get("token"), arm_id=row.get("arm_id"),
                arm_role=row.get("role"),
                geometry=row.get("geometry"), n=row.get("n_dev"),
                patch=row.get("patch"),
                band=row.get("requested_band_len"),
                values_width=row.get("requested_values_width"),
                view_chunk=row.get("requested_view_chunk"),
                fwd_view_batch=fwd_vb,
                num_channels=channels,
                write_slab_bytes=slab,
                width_seam=row.get("width_seam"),
                values_pack_ms=row.get("values_pack_ms"),
                realized_band=row.get("realized_band_len"),
                sub_bands=row.get("sub_bands_per_owner"),
                batch=row.get("pixel_batch"),
                bracket_s=bracket, busy_s=busy, gap_s=bracket - busy,
                calls=calls,
                per_launch_ms=per_launch,
                mean_cols_per_launch=mean_cols,
                mean_pixels_per_launch=mean_pixels,
                ms_per_slice=(per_launch / mean_cols
                              if per_launch and mean_cols else None),
                ms_per_kpix=(per_launch / (mean_pixels / 1000.0)
                             if per_launch and mean_pixels else None),
                peak_gb=peak / 2 ** 30,
                composed_s=row.get("vcd_warm"),
                copy_gb=(xfer.get("copy_bytes_per_recon") or 0) / 1e9,
                copy_s=xfer.get("copy_device_s_per_recon"),
                cylinder_transient_mb=(
                    (row.get("pixel_batch") or 0) * (row.get("num_slices") or 0)
                    * 4 / 2 ** 20 if row.get("pixel_batch") else None))


def n1_rows(series):
    """Experiment 3's arms, widest values block first: the one-device parallel
    arms that do NOT move the view chunk (those are experiment 4's)."""
    out = [e for e in series
           if e["geometry"] == "parallel" and e["n"] == 1
           and not e["view_chunk"]]
    return sorted(out, key=lambda e: -(e["realized_band"] or 0))


def view_chunk_rows(series):
    """Experiment 4's arms and the same-run controls they are read against:
    every parallel arm that walks the SHIPPED band, at any device count, with
    or without a narrowed view chunk.  Sorted by device count then by
    decreasing chunk, so each control leads its own pair."""
    out = [e for e in series
           if e["geometry"] == "parallel"
           and (e["view_chunk"]
                or (e["patch"] == "none" and e["sub_bands"] == 1))]
    if not any(e["view_chunk"] for e in out):
        return []
    return sorted(out, key=lambda e: (e["n"], -(e["view_chunk"] or 10 ** 6)))


def print_n1_discriminator(series):
    """Experiment 3's table: the cost per slice at ONE device against the
    values-block width, with the two- and four-device readings of the SAME
    column beside it, then the rule that reads them.

    The n > 1 side is printed whether or not those arms ran in this job, from
    PARALLEL_PER_SLICE_REFERENCE -- the follow-up job runs the four n = 1 arms
    alone, and a discrimination table missing the side it discriminates
    against would have to be assembled by hand from two logs."""
    rows = n1_rows(series)
    print("\n===== experiment 3: one device, narrowed values block =====")
    if not rows:
        print("   NOT MEASURED IN THIS RUN -- no one-device parallel arm ran.")
        print("   These arms are opt-in: "
              "MG10_ARMS=p1_shipped,p1_band0504,p1_band0252,p1_band0063")
        return
    print(f"{'arm':>12}{'n':>3}{'width':>8}{'pieces':>8}{'vb':>5}"
          f"{'slab_MB':>9}{'launches':>10}"
          f"{'per_launch_ms':>15}{'pack_ms':>9}{'ms_per_slice':>14}"
          f"{'busy_s':>9}{'bracket_s':>11}{'composed_s':>12}{'peak_GB':>9}"
          f"{'vs_control':>12}")
    control = None
    for entry in rows:
        if entry["patch"] == "none":
            control = entry
    for entry in rows:
        ratio = (entry["busy_s"] / control["busy_s"]
                 if control and control["busy_s"] else None)
        slab = entry["write_slab_bytes"]
        print(f"{entry['token']:>12}{entry['n']:>3}"
              f"{_fmt(entry['realized_band'], '8.0f')}"
              f"{_fmt(entry['sub_bands'], '8.0f')}"
              f"{_fmt(entry['fwd_view_batch'], '5.0f')}"
              f"{_fmt((slab / 2 ** 20) if slab else None, '9.1f')}"
              f"{_fmt(entry['calls'], '10.0f')}"
              f"{_fmt(entry['per_launch_ms'], '15.2f')}"
              f"{_fmt(entry['values_pack_ms'], '9.3f')}"
              f"{_fmt(entry['ms_per_slice'], '14.4f')}"
              f"{_fmt(entry['busy_s'], '9.2f')}"
              f"{_fmt(entry['bracket_s'], '11.2f')}"
              f"{_fmt(entry['composed_s'], '12.2f')}"
              f"{_fmt(entry['peak_gb'], '9.2f')}"
              f"{_fmt(ratio, '12.3f')}")
    print("   width   = the column count of the values blocks the kernel was "
          "handed, measured")
    print("             inside the call, not asked for.  pieces = how many of "
          "them one forward")
    print("             call cut the volume into.  vb = the realized forward "
          "view batch.")
    print("   slab_MB = ONE launch's atomic write target, vb x channels x "
          "width x 4 B: the tensor")
    print("             the kernel zeroes and accumulates into.  An H100's L2 "
          "is 50 MB, so this")
    print("             column is where a cache reading crosses over.")
    print("   pack_ms = the strided-to-contiguous copy each narrowed piece "
          "pays INSIDE the timed")
    print("             body, measured separately so it can be subtracted; the "
          "full-width control")
    print("             pays none.")
    print("   vs_control = this arm's busy time over the one-device shipped "
          "call's.")

    print("\n   the same column, at every device count measured so far:")
    print(f"{'n':>5}{'width':>8}{'vb':>5}{'slab_MB':>9}{'per_launch_ms':>15}"
          f"{'ms_per_slice':>14}   source")
    measured = {}
    for entry in rows:
        width = entry["realized_band"]
        if width:
            measured[(1, int(width))] = dict(
                per_launch_ms=entry["per_launch_ms"],
                ms_per_slice=entry["ms_per_slice"],
                view_batch=entry["fwd_view_batch"],
                slab=entry["write_slab_bytes"],
                source=f"THIS RUN, {entry['token']}")
    table = dict(PARALLEL_PER_SLICE_REFERENCE)
    table.update(measured)              # this run's readings win over the quote
    for (n_dev, width) in sorted(table, key=lambda k: (k[0], -k[1])):
        block = table[(n_dev, width)]
        view_batch = block.get("view_batch", SHIPPED_CHUNK)
        slab = block.get("slab")
        if slab is None:
            slab = _write_slab_bytes(view_batch, width, CELL[2])
        print(f"{n_dev:>5}{width:>8}{_fmt(view_batch, '5.0f')}"
              f"{_fmt((slab / 2 ** 20) if slab else None, '9.1f')}"
              f"{_fmt(block['per_launch_ms'], '15.2f')}"
              f"{_fmt(block['ms_per_slice'], '14.4f')}   {block['source']}")
    print("   the n=2 and n=4 lines are mg10's first run (h004); the n=1 "
          "width-1008 line is mg9's")
    print("   finding 1.7 unless this run re-measured it, in which case this "
          "run's number replaces")
    print("   it and the two should agree -- a control that does not reproduce "
          "its anchor means the")
    print("   node or the tree moved and nothing below can be read.  The "
          "quoted lines all ran at")
    print(f"   the shipped view chunk of {SHIPPED_CHUNK} and at "
          f"{CELL[2]} channels, which is what their slab is computed from.")
    print()
    print(DISCRIMINATION_RULE)


def print_view_chunk_table(series):
    """Experiment 4's table: the same write slab reached from the VIEW axis.

    Every arm here walks the SHIPPED band width, so the only thing that moves
    between an arm and its control is how many views one launch covers -- and
    therefore how many bytes one launch writes."""
    rows = view_chunk_rows(series)
    print("\n===== experiment 4: the forward view chunk =====")
    if not rows:
        print("   NOT MEASURED IN THIS RUN -- no view-chunk arm ran.")
        print("   These arms are opt-in: MG10_ARMS=...,"
              + ",".join([f"p1_vc{c}" for c in PARALLEL_VIEW_CHUNKS]
                         + ["p2_shard"]
                         + [f"p2_vc{c}" for c in PARALLEL_VIEW_CHUNKS]))
        return
    print(f"{'arm':>12}{'n':>3}{'chunk':>7}{'width':>8}{'vb':>5}{'slab_MB':>9}"
          f"{'launches':>10}{'per_launch_ms':>15}{'ms_per_slice':>14}"
          f"{'busy_s':>9}{'bracket_s':>11}{'composed_s':>12}{'peak_GB':>9}"
          f"{'vs_control':>12}")
    controls = {e["n"]: e for e in rows if not e["view_chunk"]}
    for entry in rows:
        control = controls.get(entry["n"])
        ratio = (entry["busy_s"] / control["busy_s"]
                 if control and control["busy_s"] else None)
        slab = entry["write_slab_bytes"]
        print(f"{entry['token']:>12}{entry['n']:>3}"
              f"{_fmt(entry['view_chunk'], '7.0f')}"
              f"{_fmt(entry['realized_band'], '8.0f')}"
              f"{_fmt(entry['fwd_view_batch'], '5.0f')}"
              f"{_fmt((slab / 2 ** 20) if slab else None, '9.1f')}"
              f"{_fmt(entry['calls'], '10.0f')}"
              f"{_fmt(entry['per_launch_ms'], '15.2f')}"
              f"{_fmt(entry['ms_per_slice'], '14.4f')}"
              f"{_fmt(entry['busy_s'], '9.2f')}"
              f"{_fmt(entry['bracket_s'], '11.2f')}"
              f"{_fmt(entry['composed_s'], '12.2f')}"
              f"{_fmt(entry['peak_gb'], '9.2f')}"
              f"{_fmt(ratio, '12.3f')}")
    for n_dev in sorted(controls):
        if not any(e["view_chunk"] and e["n"] == n_dev for e in rows):
            print(f"   NOTE: no view-chunk arm ran at n={n_dev}; that "
                  f"control's line is here only as the reference.")
    for n_dev in sorted({e["n"] for e in rows if e["view_chunk"]}):
        if n_dev not in controls:
            print(f"   WARNING: no same-run control at n={n_dev}.  The "
                  f"vs_control column is empty for those arms and the "
                  f"comparison would have to be made against another job's "
                  f"row, which is not what this instrument is for.")
    print("   chunk   = the forward view chunk this arm forced; a blank is the "
          "shipped 128.")
    print("   width   = the values-block width, which no arm here moves: every "
          "line walks the")
    print("             shipped band, so the ONLY thing that changes between "
          "an arm and its")
    print("             control is how many views one launch covers.")
    print("   slab_MB = the same column as experiment 3's, reached from the "
          "other axis.  8 views")
    print("             at the full width covers the same bytes as 128 views "
          "at a narrow one.")
    print("   ms_per_slice is per_launch_ms over the width, which is constant "
          "down a device-count")
    print("             block here, so it falls exactly as the chunk falls "
          "unless the kernel is")
    print("             getting more efficient per byte -- which is the whole "
          "question.")
    print("   READ IT AS: at two devices the shipped walk costs about 0.084 ms "
          "per slice at every")
    print("     width tried.  If the two-device chunk arms bring that toward "
          "0.041 -- the cost per")
    print("     slice one device pays at the full width -- then how much one "
          "launch writes is what")
    print("     the doubling was about, and the device count is a bystander.  "
          "If they do not move")
    print("     it, the slab is not the mechanism.  No verdict is printed "
          "here.")


def print_slots(series, by_token, values):
    """Print this run's numbers against the design note's slot names, verbatim.

    The design note that consumes these rows has a fixed slot list, and a slot
    filled by hand from a table is a slot that can be filled wrong.  So the
    harness prints each slot name exactly as the note carries it, with the
    values this run produced beside it.  A slot whose arms did not run prints
    NOT MEASURED IN THIS RUN rather than nothing, so a truncated or narrowed
    job cannot leave a slot looking answered.

    No slot is a verdict.  "the chosen knee" and "the chosen pixel batch" are
    the best point IN THIS SWEEP by the normalized cost column, printed with
    the whole sweep beside them; choosing is the note's job, not the
    harness's."""
    print("\n===== the design note's slots =====")

    def by_geometry(geometry, patched, drop_n1=False):
        # drop_n1 keeps experiments 3 and 4's arms out of experiment 1's knee
        # slot: the knee is a question about how a SHARD is walked, a
        # one-device row has no shard to walk, and a view-chunk row walks the
        # shipped band.  Both have their own slots below.
        out = [e for e in series if e["geometry"] == geometry
               and ((e["patch"] != "none") == patched)
               and not (drop_n1 and (e["n"] == 1 or e["view_chunk"]))]
        return sorted(out, key=lambda e: (e["n"], e["band"] or e["batch"] or 0))

    def missing(name, why):
        print(f"  [SLOT: {name}]")
        print(f"      NOT MEASURED IN THIS RUN -- {why}")

    # ── shape P, the parallel band knee ──────────────────────────────────────
    swept = by_geometry("parallel", True, drop_n1=True)
    if not swept:
        missing("mg10 parallel band knee", "no sub-band arm ran")
        missing("mg10 shape P band-copy values at the chosen knee",
                "no sub-band arm ran")
    else:
        print("  [SLOT: mg10 parallel band knee]")
        best = None
        for entry in swept:
            control = by_token.get(parallel_control_token(entry["n"]))
            ratio = (entry["busy_s"] / control["busy_s"]
                     if control and control["busy_s"] else None)
            print(f"      n={entry['n']} asked {entry['band']:>3} -> walked "
                  f"{entry['sub_bands']} x {entry['realized_band']}: "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(entry['ms_per_slice'], '.4f')} ms/slice, "
                  f"busy {_fmt(entry['busy_s'], '.2f')} s, "
                  f"{_fmt(ratio, '.3f')} x the control")
            if ratio is not None and (best is None or ratio < best[1]):
                best = (entry, ratio)
        for entry in by_geometry("parallel", False, drop_n1=True):
            print(f"      n={entry['n']} CONTROL (whole "
                  f"{entry['realized_band']}-slice shard): "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(entry['ms_per_slice'], '.4f')} ms/slice, "
                  f"busy {_fmt(entry['busy_s'], '.2f')} s")
        if best is None:
            print("      no control ran beside the swept arms, so no ratio "
                  "can be formed")
        elif best[1] >= 1.0:
            print(f"      BEST SWEPT LENGTH IS STILL WORSE THAN THE SHIPPED "
                  f"WALK: the best is {best[0]['band']} at "
                  f"{best[1]:.3f} x the control.  On these numbers there is no "
                  f"knee below the shard in the swept range.")
        else:
            print(f"      best in this sweep: asked {best[0]['band']}, walked "
                  f"{best[0]['sub_bands']} x {best[0]['realized_band']}, at "
                  f"{best[1]:.3f} x the control's busy time")
        print("  [SLOT: mg10 shape P band-copy values at the chosen knee]")
        for entry in swept + by_geometry("parallel", False, drop_n1=True):
            label = (f"asked {entry['band']}" if entry["band"]
                     else "control, whole shard")
            print(f"      n={entry['n']} {label:>22}: band copies "
                  f"{_fmt(entry['copy_gb'], '.2f')} GB per reconstruction, "
                  f"per-device peak {_fmt(entry['peak_gb'], '.2f')} GB "
                  f"(band {entry['realized_band']} slices)")

    # ── the one-device width discrimination ──────────────────────────────────
    n1 = n1_rows(series)
    if not n1:
        missing("mg10 one-device width discrimination",
                "no one-device parallel arm ran (they are opt-in: "
                "MG10_ARMS=p1_shipped,p1_band0504,p1_band0252,p1_band0063)")
    else:
        print("  [SLOT: mg10 one-device width discrimination]")
        for entry in n1:
            print(f"      n=1 width {str(entry['realized_band']):>4} "
                  f"({entry['sub_bands']} piece(s)): "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(entry['ms_per_slice'], '.4f')} ms/slice, "
                  f"busy {_fmt(entry['busy_s'], '.2f')} s, pack "
                  f"{_fmt(entry['values_pack_ms'], '.3f')} ms")
        for n_dev in (2, 4):
            for (ref_n, width) in sorted(PARALLEL_PER_SLICE_REFERENCE,
                                         key=lambda k: -k[1]):
                if ref_n != n_dev:
                    continue
                block = PARALLEL_PER_SLICE_REFERENCE[(ref_n, width)]
                print(f"      n={ref_n} width {width:>4} (already measured): "
                      f"{_fmt(block['per_launch_ms'], '.2f')} ms/launch, "
                      f"{_fmt(block['ms_per_slice'], '.4f')} ms/slice "
                      f"[{block['source']}]")
        narrowed = [e for e in n1 if e["patch"] != "none"]
        if narrowed:
            per_slice = [e["ms_per_slice"] for e in narrowed
                         if e["ms_per_slice"]]
            if per_slice:
                print(f"      the narrowed one-device readings span "
                      f"{min(per_slice):.4f} to {max(per_slice):.4f} ms per "
                      f"slice, against the two anchors "
                      f"{N1_WIDE_MS_PER_SLICE:.4f} (one device, width 1008) "
                      f"and {NGT1_MS_PER_SLICE:.4f} (two and four devices, "
                      f"every width).")
        control = next((e for e in n1 if e["patch"] == "none"), None)
        w63 = next((e for e in narrowed if e["realized_band"] == 63), None)
        if control and w63 and control["busy_s"]:
            ratio = w63["busy_s"] / control["busy_s"]
            print(f"      width 63 at one device: {ratio:.3f} x the "
                  f"one-device control's busy time.  At two devices the same "
                  f"width ran at "
                  f"{1.0 - N2_WIDTH63_WIN_FRAC:.3f} x its control.")
        print("      the rule these are read by is printed in full with "
              "experiment 3's table above.")

    # ── the write slab from the view axis ────────────────────────────────────
    vc = [e for e in view_chunk_rows(series) if e["view_chunk"]]
    if not vc:
        missing("mg10 forward view-chunk discrimination",
                "no view-chunk arm ran (they are opt-in: p1_vc*, p2_vc*, and "
                "p2_shard has to be in the same job as the two-device pair)")
    else:
        print("  [SLOT: mg10 forward view-chunk discrimination]")
        controls = {e["n"]: e for e in view_chunk_rows(series)
                    if not e["view_chunk"]}
        for entry in vc:
            control = controls.get(entry["n"])
            ratio = (entry["busy_s"] / control["busy_s"]
                     if control and control["busy_s"] else None)
            slab = entry["write_slab_bytes"]
            print(f"      n={entry['n']} chunk {entry['view_chunk']:>4} at "
                  f"width {entry['realized_band']:>4} (slab "
                  f"{_fmt((slab / 2 ** 20) if slab else None, '.1f')} MB): "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(entry['ms_per_slice'], '.4f')} ms/slice, "
                  f"{_fmt(ratio, '.3f')} x its same-run control")
        for n_dev, control in sorted(controls.items()):
            slab = control["write_slab_bytes"]
            print(f"      n={n_dev} CONTROL chunk {SHIPPED_CHUNK} at width "
                  f"{control['realized_band']:>4} (slab "
                  f"{_fmt((slab / 2 ** 20) if slab else None, '.1f')} MB): "
                  f"{_fmt(control['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(control['ms_per_slice'], '.4f')} ms/slice")
        print(f"      the two anchors these are read against are the same "
              f"two: {N1_WIDE_MS_PER_SLICE:.4f} ms per slice at one device "
              f"and the full width, and {NGT1_MS_PER_SLICE:.4f} at two and "
              f"four devices at every width tried.")

    # ── shape C, the cone pixel batch ────────────────────────────────────────
    gathered = by_geometry("cone", True)
    if not gathered:
        missing("mg10 cone pixel batch", "no column-gather arm ran")
        missing("mg10 shape C column-transient values at the chosen pixel batch",
                "no column-gather arm ran")
    else:
        print("  [SLOT: mg10 cone pixel batch]")
        best = None
        for entry in gathered:
            control = by_token.get(f"c{entry['n']}_banded")
            ratio = (entry["busy_s"] / control["busy_s"]
                     if control and control["busy_s"] else None)
            print(f"      n={entry['n']} batch {entry['batch']:>5}: "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"{_fmt(entry['ms_per_kpix'], '.4f')} ms/kpix, "
                  f"busy {_fmt(entry['busy_s'], '.2f')} s, "
                  f"{_fmt(ratio, '.3f')} x the banded control")
            if ratio is not None and (best is None or ratio < best[1]):
                best = (entry, ratio)
        for entry in by_geometry("cone", False):
            shape = "single device" if entry["n"] == 1 else "banded control"
            print(f"      n={entry['n']} {shape:>14}: "
                  f"{_fmt(entry['per_launch_ms'], '.2f')} ms/launch, "
                  f"busy {_fmt(entry['busy_s'], '.2f')} s")
        if best is None:
            print("      no banded control ran beside the gather arms, so no "
                  "ratio can be formed")
        elif best[1] >= 1.0:
            print(f"      NO COLUMN BATCH BEAT THE BANDED WALK: the best is "
                  f"{best[0]['batch']} at {best[1]:.3f} x the control.")
        else:
            print(f"      best in this sweep: batch {best[0]['batch']} at "
                  f"{best[1]:.3f} x the banded control's busy time")
        print("  [SLOT: mg10 shape C column-transient values at the chosen "
              "pixel batch]")
        for entry in gathered:
            print(f"      n={entry['n']} batch {entry['batch']:>5}: gathered "
                  f"cylinder {_fmt(entry['cylinder_transient_mb'], '.1f')} MB "
                  f"(batch x slices x 4 B), cross-device "
                  f"{_fmt(entry['copy_gb'], '.2f')} GB per reconstruction, "
                  f"per-device peak {_fmt(entry['peak_gb'], '.2f')} GB")
        for entry in by_geometry("cone", False):
            if entry["n"] == 1:
                continue
            print(f"      n={entry['n']} banded control: broadcast band "
                  f"{_fmt((entry['mean_pixels_per_launch'] or 0) * (entry['realized_band'] or 0) * 4 / 2 ** 20, '.1f')}"
                  f" MB (pixels x band x 4 B), cross-device "
                  f"{_fmt(entry['copy_gb'], '.2f')} GB per reconstruction, "
                  f"per-device peak {_fmt(entry['peak_gb'], '.2f')} GB")

    # ── the value slots ──────────────────────────────────────────────────────
    def rel(block):
        return None if not block else block.get("rel_l2")

    par = [v for v in values if v["geometry"] == "parallel"]
    if not par:
        missing("mg10 parallel repeat-to-repeat distance", "no parallel arm ran")
        missing("mg10 parallel band-to-band distances", "no parallel arm ran")
    else:
        print("  [SLOT: mg10 parallel repeat-to-repeat distance]")
        for entry in par:
            print(f"      {entry['token']:>10}: checksum repeat spread "
                  f"{_fmt(entry['checksum_repeat_spread'], '.2e')}, sample "
                  f"pass-to-pass {_fmt(rel(entry['own_pass_to_pass']), '.2e')}")
        print("  [SLOT: mg10 parallel band-to-band distances]")
        for entry in par:
            if entry["patch"] == "none":
                continue
            print(f"      {entry['label']:>18} vs the shipped "
                  f"control at n={entry['n']}: checksum "
                  f"{_fmt(entry['checksum_vs_control'], '.2e')}, sample "
                  f"{_fmt(rel(entry['vs_control']), '.2e')} "
                  f"(this arm's own repeat floor "
                  f"{_fmt(entry['checksum_repeat_spread'], '.2e')})")

    cone = [v for v in values if v["geometry"] == "cone"]
    anchor_ran = any(v["token"] == "c1" for v in cone)
    banded = [v for v in cone if v["patch"] == "none" and v["n"] > 1]
    gather = [v for v in cone if v["patch"] == "cone_column_gather"]
    if not (anchor_ran and banded):
        missing("mg10 cone distance, banded to n=1",
                "the one-device anchor or the banded control did not run")
    else:
        print("  [SLOT: mg10 cone distance, banded to n=1]")
        for entry in banded:
            print(f"      n={entry['n']} banded vs n=1: checksum "
                  f"{_fmt(entry['checksum_vs_anchor'], '.2e')}, sample "
                  f"{_fmt(rel(entry['vs_anchor']), '.2e')} (repeat floor "
                  f"{_fmt(entry['checksum_repeat_spread'], '.2e')})")
    if not (anchor_ran and gather):
        missing("mg10 cone distance, column-gather to n=1",
                "the one-device anchor or the column-gather arms did not run")
    else:
        print("  [SLOT: mg10 cone distance, column-gather to n=1]")
        for entry in gather:
            print(f"      n={entry['n']} batch {entry['batch']:>5} vs n=1: "
                  f"checksum {_fmt(entry['checksum_vs_anchor'], '.2e')}, "
                  f"sample {_fmt(rel(entry['vs_anchor']), '.2e')} (repeat "
                  f"floor {_fmt(entry['checksum_repeat_spread'], '.2e')})")
    if not gather:
        missing("mg10 cone distance, banded to column-gather",
                "no column-gather arm ran")
    else:
        print("  [SLOT: mg10 cone distance, banded to column-gather]")
        for entry in gather:
            print(f"      n={entry['n']} batch {entry['batch']:>5} vs the "
                  f"banded control: checksum "
                  f"{_fmt(entry['checksum_vs_control'], '.2e')}, sample "
                  f"{_fmt(rel(entry['vs_control']), '.2e')} (repeat floor "
                  f"{_fmt(entry['checksum_repeat_spread'], '.2e')})")


def summarize(rows, out_path):
    """The per-arm tables, then the two sweep tables, the value table, and the
    design note's slots."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    print(f"\n===== mg10 shape sweep ({out_path}) =====")

    for row in rows:
        if row.get("arm_class") == "generator":
            continue
        if row.get("error"):
            print(f"\n--- {row.get('arm_id', '?')}: ERROR "
                  f"{str(row['error'])[-600:]}")
            continue
        checks = []
        for name, flag in (("pin", row.get("pin_env_ok")),
                           ("bod", row.get("bodies_ok")),
                           ("bpd", row.get("bodies_per_device_ok")),
                           ("vb", row.get("vb_ok")),
                           ("chunk", row.get("shipped_chunk_is_the_anchor_ok")),
                           ("chunk_same", row.get("chunk_as_intended_ok")),
                           ("kern", row.get("kernels_launched_ok")),
                           ("kill", row.get("kill_switch_off_ok")),
                           ("cal", row.get("calibration_absent_ok")),
                           ("md5", row.get("sino_md5_ok")),
                           ("slices", row.get("num_slices_planned_ok")),
                           ("rgn", row.get("region_nonzero_ok")),
                           ("rec", row.get("reconcile_ok"))):
            if flag is False:
                checks.append(f"{name}:FAIL")
        print_arm_table(row)
        mismatch = sum(e.get("device_mismatch", 0)
                       for e in row.get("per_device") or [])
        if mismatch:
            checks.append(f"device_key:{mismatch} MISATTRIBUTED CALLS")
        if any(p.get("busy_cap_hit") for p in row.get("per_recon") or []):
            checks.append("busy_events:CAPPED (raise MG10_MAX_EVENT_PAIRS)")
        if any(p.get("copy_cap_hit") for p in row.get("per_recon") or []):
            checks.append("copy_events:CAPPED (raise MG10_MAX_EVENT_PAIRS)")
        memo_composed = MEMO_COMPOSED_S.get((row.get("geometry"),
                                             row.get("n_dev")))
        print(f"   composed reconstruction "
              f"{_fmt(row.get('vcd_warm'), '.2f')} s"
              f"{'' if memo_composed is None or row.get('patch') != 'none' else f' (mg5 {memo_composed:.2f} s)'}"
              f" (spread "
              f"{_fmt((row.get('vcd_warm_spread') or 0) * 100, '.1f')}%), peak "
              f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:.2f} GB, checks: "
              f"{','.join(checks) if checks else 'ok'}")

    series = [e for e in (_arm_series_entry(r) for r in live) if e]
    by_token = {e["token"]: e for e in series}

    # ── EXPERIMENT 1 ─────────────────────────────────────────────────────────
    print("\n===== experiment 1: the parallel band knee =====")
    print(f"{'arm':>12}{'n':>3}{'asked':>7}{'walked':>16}{'launches':>10}"
          f"{'per_launch_ms':>15}{'ms_per_slice':>14}{'busy_s':>9}"
          f"{'bracket_s':>11}{'composed_s':>12}{'peak_GB':>9}{'vs_control':>12}")
    for entry in series:
        # The one-device arms are experiment 3's and the view-chunk arms are
        # experiment 4's; both have their own table below.  This one is the
        # knee, which is a question about how a SHARD is walked.
        if entry["geometry"] != "parallel" or entry["n"] == 1 \
                or entry["view_chunk"]:
            continue
        control = by_token.get(parallel_control_token(entry["n"]))
        ratio = (entry["busy_s"] / control["busy_s"]
                 if control and control["busy_s"] else None)
        walked = (f"{entry['sub_bands']} x {entry['realized_band']}"
                  if entry["sub_bands"] else "-")
        print(f"{entry['token']:>12}{entry['n']:>3}"
              f"{_fmt(entry['band'], '7.0f')}{walked:>16}"
              f"{_fmt(entry['calls'], '10.0f')}"
              f"{_fmt(entry['per_launch_ms'], '15.2f')}"
              f"{_fmt(entry['ms_per_slice'], '14.4f')}"
              f"{_fmt(entry['busy_s'], '9.2f')}"
              f"{_fmt(entry['bracket_s'], '11.2f')}"
              f"{_fmt(entry['composed_s'], '12.2f')}"
              f"{_fmt(entry['peak_gb'], '9.2f')}"
              f"{_fmt(ratio, '12.3f')}")
    print("   asked  = the sub-band length the arm requested; a blank is the "
          "shipped control.")
    print("   walked = what the driver's balanced tiling actually produced.  "
          "The request is capped")
    print("            at the shard and the tiling snaps to a divisor, so two "
          "requests can land on")
    print("            the same walk -- those pairs are a free repeat and give "
          "the arm-to-arm floor.")
    print("   ms_per_slice = per_launch_ms divided by the band's slice count: "
          "what one launch costs")
    print("            per slice it covers.  This is the column the knee is "
          "read off, because a")
    print("            narrower band always has a smaller per_launch_ms and "
          "that alone says nothing.")
    print("            mg9's readings for reference: 0.0412 at a 1008-slice "
          "band, 0.0823 at 504,")
    print("            0.0849 at 252 -- the WIDEST band was the most efficient "
          "per slice, so a sweep")
    print("            in which no length beats the shipped control is a real "
          "possible outcome.")
    print("   vs_control = this arm's per-device busy time over the control's "
          "at the same count.")
    print("            Below 1.00 is a win; at or above it, sub-banding costs "
          "more launches than")
    print("            it saves per launch.  It equals ms_per_slice over the "
          "control's ms_per_slice")
    print("            whenever both walk the same total slice work, which "
          "every arm here does.")

    # ── EXPERIMENT 2 ─────────────────────────────────────────────────────────
    print("\n===== experiment 2: the cone column gather =====")
    print(f"{'arm':>12}{'n':>3}{'shape':>16}{'batch':>8}{'launches':>10}"
          f"{'per_launch_ms':>15}{'ms_per_kpix':>13}{'busy_s':>9}"
          f"{'bracket_s':>11}{'composed_s':>12}{'peak_GB':>9}{'copy_GB':>9}"
          f"{'vs_control':>12}")
    for entry in series:
        if entry["geometry"] != "cone":
            continue
        control = by_token.get(f"c{entry['n']}_banded")
        ratio = (entry["busy_s"] / control["busy_s"]
                 if control and control["busy_s"] and control is not entry
                 else None)
        shape = ("column gather" if entry["patch"] == "cone_column_gather"
                 else ("single device" if entry["n"] == 1 else "banded walk"))
        print(f"{entry['token']:>12}{entry['n']:>3}{shape:>16}"
              f"{_fmt(entry['batch'], '8.0f')}"
              f"{_fmt(entry['calls'], '10.0f')}"
              f"{_fmt(entry['per_launch_ms'], '15.2f')}"
              f"{_fmt(entry['ms_per_kpix'], '13.4f')}"
              f"{_fmt(entry['busy_s'], '9.2f')}"
              f"{_fmt(entry['bracket_s'], '11.2f')}"
              f"{_fmt(entry['composed_s'], '12.2f')}"
              f"{_fmt(entry['peak_gb'], '9.2f')}"
              f"{_fmt(entry['copy_gb'], '9.2f')}"
              f"{_fmt(ratio, '12.3f')}")
    print("   batch   = the column batch: how many pixel columns are gathered "
          "at full height per call.")
    print("   ms_per_kpix = per_launch_ms divided by the launch's pixel count "
          "in thousands: what one")
    print("             launch costs per thousand pixel columns.  This is the "
          "cone twin of the")
    print("             parallel sweep's ms_per_slice, and for the same reason "
          "-- a smaller batch")
    print("             always has a smaller per_launch_ms, which on its own "
          "says nothing.  The")
    print("             banded control's launches carry the whole pixel set at "
          "a partial slice band,")
    print("             so its two normalized columns are the mirror image of "
          "a gather arm's.")
    print("   copy_GB = cross-device bytes per reconstruction.  The banded "
          "walk moves whole slice")
    print("             bands; the gather moves pixel-column pieces, and the "
          "two totals are the")
    print("             ledger input for the design note's residency section.")

    # ── EXPERIMENTS 3 AND 4 ──────────────────────────────────────────────────
    print_n1_discriminator(series)
    print_view_chunk_table(series)

    # ── THE VALUE COLUMNS ────────────────────────────────────────────────────
    print("\n===== the value columns =====")
    print("   sample distances (relative L2 between strided reconstruction "
          "samples):")
    print(f"{'arm':>12}{'own_floor':>12}{'vs_control':>12}{'vs_n1_anchor':>14}"
          f"   | checksum distances (from the three per-recon checksums)")
    print(f"{'':>12}{'':>12}{'':>12}{'':>14}"
          f"   | {'repeat':>10}{'vs_control':>12}{'vs_anchor':>12}")
    values = value_table(rows)
    for entry in values:
        def rel(block):
            return None if not block else block.get("rel_l2")
        print(f"{str(entry['token']):>12}"
              f"{_fmt(rel(entry['own_pass_to_pass']), '12.2e')}"
              f"{_fmt(rel(entry['vs_control']), '12.2e')}"
              f"{_fmt(rel(entry['vs_anchor']), '14.2e')}"
              f"   | {_fmt(entry['checksum_repeat_spread'], '10.2e')}"
              f"{_fmt(entry['checksum_vs_control'], '12.2e')}"
              f"{_fmt(entry['checksum_vs_anchor'], '12.2e')}")
    print("   own_floor / repeat = this arm's REPEAT-TO-REPEAT distance: two "
          "timed reconstructions")
    print("                  of the identical configuration.  Not zero on the "
          "GPU -- both forward")
    print("                  kernels accumulate with float atomics -- and it "
          "is the bar every")
    print("                  cross-arm number below is read against.")
    print("   vs_control   = BAND-TO-BAND (parallel) or shape-to-shape (cone) "
          "against the shipped")
    print("                  walk at the same device count.")
    print("   vs_n1_anchor = against the one-device cone arm (cone rows only): "
          "the banded")
    print("                  control's own distance to it is the baseline the "
          "prototype's distance")
    print("                  has to be read against, since both include the "
          "sharding difference.")
    print("   The two families are computed independently -- the sample "
          "distance can see a local")
    print("   change a whole-volume sum would cancel, and the checksum "
          "distance can be recomputed")
    print("   from the rows with no artifacts kept.  They should agree in "
          "order of magnitude.")
    exact = [e for e in values if e.get("checksum_equals_control_exactly")
             is not None]
    if exact:
        print("   the parallel exact-match expectation (recorded, not "
              "asserted -- the kernel's atomics")
        print("   are not bit-reproducible, so an inequality here is expected "
              "rather than a failure):")
        for entry in exact:
            print(f"      {entry['token']:>12}: checksum equals its control "
                  f"exactly = {entry['checksum_equals_control_exactly']}")

    print()
    print(READING_TEXT)
    print_slots(series, by_token, values)

    backends = {r.get("event_backend") for r in live if r.get("event_backend")}
    if any(b != "cuda_events" for b in backends):
        print(f"\nNOTE: event backend {sorted(backends)}.  On the CPU path the "
              f"per-device span map collapses to a single 'cpu' key whose span "
              f"IS the host wall, every cross-device copy is a no-op, and the "
              f"peak columns are empty.  The busy, gap, copy and memory "
              f"columns price nothing there; the witnesses and the value "
              f"columns still do.")

    best_clock = {}
    for row in rows:
        for gpu in row.get("gpu_health") or []:
            if gpu.get("sm_mhz"):
                best_clock[gpu["index"]] = max(best_clock.get(gpu["index"], 0),
                                               gpu["sm_mhz"])
    rerun = []
    for row in rows:
        if row.get("error"):
            continue
        depressed = any(
            gpu.get("sm_mhz") and best_clock.get(gpu["index"])
            and gpu["sm_mhz"] < CLOCK_DEPRESSED_FRAC * best_clock[gpu["index"]]
            for gpu in row.get("gpu_health") or [])
        if row.get("gpu_hot") and depressed:
            rerun.append(row.get("arm_id"))
    print(f"\nthrottle rule: sw_power_cap at normal temperature is recorded and "
          f"KEPT; {len(rerun)} row(s) hot AND clock-depressed -> re-run: "
          f"{rerun}")
    return series, values, rerun


# ── the wall estimate, printed by --dry-run ───────────────────────────────────
# mg9's MEASURED subprocess walls on h018 at this cell, per arm, cold pass and
# torch import included: parallel n=1 203 s, n=2 200 s, n=4 145 s, cone n=2
# 308 s, each generator 65 s.  Those are the base rates below.  A patched arm's
# wall is unknown by construction -- that is what the job measures -- so the
# estimate carries a range: the low end assumes a patched arm costs what its
# control costs, the high end assumes half again as much.
BASE_ARM_S = {("parallel", 1): 203, ("parallel", 2): 200, ("parallel", 4): 145,
              ("cone", 1): 310, ("cone", 2): 308, ("cone", 4): 300}
GENERATOR_S = 65


def wall_estimate(generators, measured):
    low = GENERATOR_S * len(generators)
    high = low
    for cfg in measured:
        base = BASE_ARM_S.get((cfg["geometry"], cfg["n_dev"]), 300)
        patched = (cfg["band_len"] is not None
                   or cfg["pixel_batch"] is not None
                   or cfg.get("values_width") is not None
                   or cfg.get("view_chunk") is not None)
        low += base
        high += int(base * (1.5 if patched else 1.05))
    return low, high


def main():
    arms = selected_arms()
    generators, measured = build_plan(arms)
    if "--dry-run" in sys.argv:
        low, high = wall_estimate(generators, measured)
        print(f"mg10 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  cell {cell_for('parallel')}, slices {num_slices_for(cell_for('parallel'))}, "
              f"warm repeats {WARM_REPEATS}, iterations {VCD_ITERATIONS}, "
              f"device {DEVICE}, results {RESULTS_DIR}")
        print(f"  sub-band lengths swept {PARALLEL_BANDS}; column batches "
              f"swept {CONE_BATCHES}")
        for cfg in generators:
            print(f"  {cfg['arm_id']:<42} {'(generator)':>28}")
        num_slices = num_slices_for(cell_for("parallel"))
        for cfg in measured:
            shard = num_slices // max(1, cfg["n_dev"] or 1)
            if cfg["band_len"] is not None:
                lengths = realized_band_lengths(shard, cfg["band_len"])
                note = (f"asked {cfg['band_len']:>3} -> walks "
                        f"{len(lengths)} x {lengths[0]}"
                        + ("  [same walk as the control]"
                           if len(lengths) == 1 else ""))
            elif cfg.get("values_width") is not None:
                lengths = realized_band_lengths(shard, cfg["values_width"])
                note = (f"one device, values block asked "
                        f"{cfg['values_width']:>4} -> cut into "
                        f"{len(lengths)} x {lengths[0]}"
                        + ("  [same call as the control]"
                           if len(lengths) == 1 else ""))
            elif cfg.get("view_chunk") is not None:
                note = (f"forward view chunk {cfg['view_chunk']:>3} at the "
                        f"shipped {shard}-slice band (shipped chunk "
                        f"{SHIPPED_CHUNK})")
            elif cfg["pixel_batch"] is not None:
                note = (f"column gather, {cfg['pixel_batch']} pixels x "
                        f"{num_slices} slices")
            elif cfg["n_dev"] == 1 and cfg["geometry"] == "parallel":
                note = ("one-device control: the shipped call, the whole "
                        f"{num_slices}-slice values block in one piece")
            elif cfg["n_dev"] == 1:
                note = "value anchor: the shipped single-device path"
            else:
                note = f"control: one {shard}-slice band per slice-owner"
            print(f"  {cfg['arm_id']:<42} n={cfg['n_dev']} "
                  f"[{cfg['token']}] {note}")
        # Requests that land on the same walk are named out loud: the balanced
        # tiling snaps to a divisor of the shard and the request is capped at
        # the shard, so two arms can be the same measurement twice.  They are
        # kept because a repeated configuration is the only arm-to-arm noise
        # floor this job has, but a reader must not read them as two points.
        seen = {}
        for cfg in measured:
            request = (cfg["band_len"] if cfg["band_len"] is not None
                       else cfg.get("values_width"))
            if request is None:
                continue
            shard = num_slices // max(1, cfg["n_dev"] or 1)
            key = (cfg["n_dev"], tuple(realized_band_lengths(shard, request)))
            seen.setdefault(key, []).append(cfg["token"])
        for (n_dev, lengths), tokens in sorted(seen.items()):
            shard = num_slices // max(1, n_dev or 1)
            control_same = (len(lengths) == 1)
            if len(tokens) > 1 or control_same:
                same = tokens + ([parallel_control_token(n_dev)]
                                 if control_same else [])
                print(f"  NOTE at n={n_dev}: {', '.join(same)} all walk "
                      f"{len(lengths)} x {lengths[0]} -- the same measurement, "
                      f"kept as the arm-to-arm noise floor.")
        print(f"  wall estimate {low / 60:.0f} to {high / 60:.0f} minutes "
              f"(mg9's measured per-arm walls as the base; a patched arm's cost "
              f"is what this job measures, so the high end assumes half again "
              f"as much).")
        print("  if that must be cut: MG10_ARMS drops arms by token.  Trim the "
              "parallel four-device")
        print("  arms first (p4_*), then the largest column batch "
              f"(c2_{CONE_BATCHES[-1]}, c4_{CONE_BATCHES[-1]}).  The parallel "
              "two-device")
        print("  sweep (p2_*) and the two-device column batches (c2_*) are the "
              "core and the cone")
        print("  one-device anchor (c1) is what every cone value row is "
              "measured against.")
        print("  experiments 3 and 4 are OPT-IN and are not in the list above "
              "unless MG10_ARMS")
        print("  named them.  They run as one short job:")
        print("    MG10_ARMS=" + ",".join(
            ["p1_shipped"]
            + [f"p1_band{w:04d}" for w in PARALLEL_N1_WIDTHS]
            + [f"p1_vc{c}" for c in PARALLEL_VIEW_CHUNKS]
            + ["p2_shard"]
            + [f"p2_vc{c}" for c in PARALLEL_VIEW_CHUNKS]))
        print("  Both controls have to be in that list: p1_shipped is what "
              "every one-device arm is")
        print("  read against and p2_shard is what the two-device chunk arms "
              "are read against, and")
        print("  the comparisons are same-run by design.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg10_shape_sweep_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg10 shape sweep on {RUN_LABEL} ({DEVICE}); "
          f"{len(measured)} arms -> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY: a truncated job still yields the arms it
    # finished, which is why the arm order runs the core first.
    with open(out_path, "w") as sink:
        for cfg in generators + measured:
            print(f"  [{cfg['arm_id']}]", flush=True)
            row = run_one(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        series, values, rerun = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=dict(series=series,
                                                values=values))) + "\n")
        sink.write(json.dumps(dict(thermal_rerun=rerun)) + "\n")
        sink.flush()
    if os.environ.get("MG10_KEEP_ARTIFACTS", "0") != "1":
        # The sinograms and the value samples are internal to this job -- the
        # distances are computed above, before anything is removed.
        for geometry in sorted({a["geometry"] for a in arms}):
            for path in (_sino_path(geometry, cell_for(geometry)),
                         _md5_path(geometry, cell_for(geometry))):
                if os.path.exists(path):
                    os.remove(path)
        for row in rows:
            for path in row.get("value_sample_paths") or []:
                if os.path.exists(path):
                    os.remove(path)
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        if cfg["arm_class"] == "generator":
            row = generator_worker(cfg)
        else:
            row = torch_worker(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        main()
