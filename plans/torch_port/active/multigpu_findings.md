# Multi-GPU performance: the readout and its findings

**Status:** INCREMENT-1 CHECKPOINT (2026-08-09), with both tuning
charters' probes now in hand.  The gate readout, the ledger
calibration, the crossover ladder, and the shape probes are complete
and validated.  The shape probes decided the guard's metric in §3.5.
Charter A's forward attribution is in §1.5, and it corrects one
reading in §1.3.  Charter B's back-loop probe is in §2, and it
confirms that charter's memo.  Every section is closed: §4's
value rows are read and goal 3 is ruled met, so the campaign's
remaining work is implementation, mapped in the plan's §0.  The plan is
`multigpu_plan.md`; terms and protocols keep its meanings.  The
checkpoint was RULED AND ENDORSED on 2026-08-09 with one amendment,
floor_4, folded into §3.1 and §6.1 below.

The campaign's instruments worked.  Every arm of every completed job
passed every arm check, the n=1 rows reproduce the kb3 baselines to
the printed digit, and the two anomalies the readout surfaced arrived
with the attribution data needed to act on them.

**Jobs and rows.**  mg1 is job 15011662 (1:22 wall), mg2 is 15011663
(0:15), the mg3 pricing probe is 15026960 (timeout by design's
benefit, one arm complete), mg4 is 15026979 (0:36).  Rows are
`mg1_readout_h014_20260809_050121.jsonl`,
`mg2_ledger_calib_h011_20260809_062252.jsonl`,
`mg3_probe_h004_20260809_063824.jsonl`, and
`mg4_ladder_h001_20260809_073857.jsonl`, all in
`/scratch/gautschi/buzzard/torch_p3/results/`.  The later jobs: mg3a
is 15029600 and mg3b is 15029601 (rows
`mg3_value_h001_20260809_081542.jsonl` and
`mg3_value_h001_20260809_161639.jsonl`), mg4b is 15030714
(`mg4_ladder_h001_20260809_165602.jsonl`), mg5 is 15034136
(`mg5_fwd_attrib_h005_20260809_195819.jsonl`), mg6 is 15034661
(`mg6_backloop_h005_20260809_223535.jsonl`), and mg7 is 15078246
(`mg7_conebatch_h004_20260810_041607.jsonl`).  Copies of every row
file are archived in `plans/experiments/torch_port/rows/`.

---

## 1. The gate readout (mg1)

This section reports the campaign's core measurement: the full
n = 1, 2, 4 matrix, both frameworks, both geometries, both gate cells,
at the shipped production configuration.  Its job is threefold: prove
the instrument valid against the recorded baselines, state the
multi-device performance picture, and deliver the forward's share of
composed time, which is item 13's entry gate.

### 1.1 Validity

The readout is valid.  All 40 arms pass every arm check.  The n=1
torch-over-jax ratios read 1.15, 1.55, 0.89, and 0.98 against kb3's
1.13, 1.55, 0.87, and 0.99.  The n=1 torch peaks read 1.93, 23.22,
2.15, and 23.68 GB, each equal to its recorded baseline to the printed
digit.  The forward bracket sits where it should: the parallel-1024
n=1 forward region reads 15.9 s against the 14.4 s composed remainder
kb3 attributed to the forward.  No row was hot with a depressed clock.

Two instrument notes are recorded.  The value-ruler verdict tripped
once at ratio 3.19 with both divergences in the e-5 class; the
criterion needed an absolute-floor clause, folded into the harness on
2026-08-09 per the checkpoint ruling (`VALUE_RULER_FLOOR` raised to
the documented benign class), and the numbers themselves are
unremarkable.  The twin-within-spread criterion proved stricter
than the plan's 2 percent bound, because within-arm spreads came in at
0.0 to 0.5 percent; eleven of twelve twins sit within 2 percent of
their siblings, and the twelfth reads 3.6 percent at a 2.7-second
cell.

### 1.2 The gate table

Warm seeded 3-iteration vcd, medians of three repeats, one H100 node
(h014), same-run jax rulers.

| cell | n | torch | jax | t/j | torch scale | jax scale | torch peak |
|---|---|---|---|---|---|---|---|
| parallel 512 | 1 | 1.91 s | 1.66 s | 1.15 | 1.00x | 1.00x | 1.93 GB |
| parallel 512 | 2 | 1.57 s | 1.98 s | **0.79** | 1.22x | 0.84x | 1.11 GB |
| parallel 512 | 4 | 2.52 s | 3.12 s | **0.81** | 0.76x | 0.53x | 0.65 GB |
| parallel 1024 | 1 | 40.00 s | 25.80 s | 1.55 | 1.00x | 1.00x | 23.22 GB |
| parallel 1024 | 2 | 39.40 s | 14.33 s | 2.75 | **1.02x** | 1.80x | 14.04 GB |
| parallel 1024 | 4 | 23.36 s | 11.52 s | 2.03 | 1.71x | 2.24x | 7.31 GB |
| cone 512 | 1 | 2.74 s | 3.07 s | 0.89 | 1.00x | 1.00x | 2.15 GB |
| cone 512 | 2 | 2.78 s | 2.93 s | 0.95 | 0.98x | 1.05x | 1.70 GB |
| cone 512 | 4 | 4.07 s | 3.76 s | 1.08 | 0.67x | 0.82x | 1.07 GB |
| cone 1024 | 1 | 61.57 s | 62.75 s | 0.98 | 1.00x | 1.00x | 23.68 GB |
| cone 1024 | 2 | 67.23 s | 43.37 s | 1.55 | **0.92x** | 1.45x | 14.31 GB |
| cone 1024 | 4 | 53.10 s | 25.78 s | 2.06 | **1.16x** | 2.43x | 9.08 GB |

Three readings carry the table.  At parallel 512 torch now beats jax
at both n>1 counts, and n=2 beats n=1; the phase-4 shape survives with
its collapse softened from about 3x to 1.3x.  At the 1024 cells torch
scales far under jax, and the two anomalies of §1.3 say why.  Memory
shards cleanly everywhere: the torch n=4 peaks are half of jax's, and
at n=1 torch holds 23 GB where jax holds 50.

The value columns are pristine.  Torch's own-count divergences match
jax's nearly digit for digit at the 512 cells (8.72e-5 against
8.70e-5; 3.03e-4 against 3.03e-4), and the cross-framework columns sit
at their documented classes, 6.1e-3 at parallel 1024 and the e-4 class
elsewhere.

### 1.3 The attribution, and the two anomalies

The three-region instrument reports per-pass host walls at every
(cell, count).  Two anomalies stand out, and each is an mg5 trigger
firing with its evidence attached.

**The cone forward is flat in device count.**  Its region reads 31.0,
29.9, and 28.6 s at n = 1, 2, 4 on cone 1024, while per-device views
drop fourfold.  The cone forward is 40 to 54 percent of that cell's
composed wall, so its flatness is why cone n=2 is a net regression
(0.92x) and cone n=4 gains only 1.16x.  The mechanism is not yet
attributed.

**The parallel forward rises at n=2.**  Its region reads 15.9, 18.0,
and 7.3 s at n = 1, 2, 4 on parallel 1024.  The rise at n=2 is why
that cell's n=2 scaling is 1.02x against jax's 1.80x.  A forward that
costs more with half the views per device points at a per-device
regime change rather than at communication, and the realized-batch
columns are in the rows to test that.

The back region behaves by contrast: 3.5, 2.2, 3.0 s at parallel 1024,
and 25.3, 30.8, 17.5 s at cone 1024, whose n=2 rise is the back half
of the cone regression.  The band reduce is negligible in time
everywhere, at or under 0.04 s per pass.  The remainder runs 8 to 65
percent of composed wall, largest at the parallel cells; it holds the
filter, the statistics, the partition machinery, and the applies, and
its composition is unmeasured by design in this readout.

**The forward share, item 13's entry gate.**  The parallel forward's
share of composed wall reads 22 to 40 percent at n=1, 16 to 46 percent
at n=2, and 27 to 31 percent at n=4.  These numbers say the forward is
the single largest attributed term at the 1024 cells at every count.

Both anomalies were carried to the attribution sweep, and §1.5 reports
what it found.  The cone reading below survives that sweep.  The
parallel reading does not.

### 1.4 Costs, for the cadence decision

The full 44-arm readout cost 1:22 on four GPUs, about 5.5 GPU-hours.
Per-arm subprocess walls run 1 to 2.5 minutes at the 512 cells and 2
to 5 minutes at the 1024 cells, cold pass included.  These are the
per-cell costs the nightly's n>1 rows would add; the nightly's own
trial remains the authoritative per-night number.

### 1.5 The forward attribution, and one corrected reading

This section resolves §1.3's two anomalies.  A sweep varied the
per-device view chunk over 32, 64, 128, and 256 at both 1024 cells,
and it ran one arm per cell with the torch forward body bound in place
of the kernel.  All 27 arms passed every arm check.  The drift arms
measured the job's ordering bias at 1.000x, so every scaling number
below is free of that bias.

The chunk constant is not a remedy.  Total reconstruction time barely
moved across the whole ladder.  Cone at one device ran 62.12, 61.78,
61.53, and 61.31 s over the four chunks, and parallel ran 40.17,
40.05, 40.00, and 39.94 s.  The best chunk beats the shipped chunk by
under one percent at every cell and count.  Device-count scaling did
not move either, holding at 1.16x for cone at four devices under every
chunk.

The realized batch does not vary with device count.  Cone's batch is
capped by the transient budget at 52 views for every chunk at or above
64.  Parallel's batch equals its chunk.  These results refute the
hypothesis that the count-divided budget shifts the batch.

The forward's device-side time is flat in device count.  Per-device
event spans at cone read 32.18, 30.61, and 30.48 s at one, two, and
four devices.  Parallel reads 28.87 and 28.75 s at one and two.  Each
device spends the same GPU time on the forward even though its share
of the views falls fourfold.

The driver offered a candidate mechanism, and it did not survive
verification (correction, 2026-08-10).  The banded forward walks the
reconstruction one slice band at a time and broadcasts each band to
every projecting device, so every device receives the entire
reconstruction whatever the count.  Two checks refute the bytes moved
as the cost.  The implied bandwidth is about 0.44 GB per second at
parallel 1024 with two devices, far below any transfer path on the
node.  And the torch-body arm runs the identical driver and identical
broadcast, yet its forward span falls 1.59x from one device to two
(39.69 to 24.96 s) while the kernel arm stays flat; an invariant
broadcast cost would appear in both.  The flatness is solidly
measured; its mechanism is UNATTRIBUTED, sits inside the kernel arm
specifically, and needs one cheap instrument arm that separates the
broadcast and per-device busy time inside the forward region.

One reading from §1.3 does not survive.  The parallel forward does not
rise at two devices.  Its host-side bracket reads 20.83, 12.44, 15.93,
and 1.99 s across the chunk ladder while total time stays at 40 s.
That bracket therefore records where synchronization landed, not what
the forward cost.  The device span is the trustworthy measure, and it
holds at 28.8 s throughout.  Cone's brackets agree with its device
spans to within three seconds, so §1.3's cone reading stands
unchanged.

The corrected measure strengthens the entry gate rather than weakening
it.  At parallel 1024 the forward's device span is 28.9 s inside a
40.0 s reconstruction.  The forward therefore occupies about seventy
percent of the run's GPU time at that cell.

The torch forward body is not an alternative to the kernels.  Its
device span at cone runs 87.80, 113.04, and 575.65 s as devices are
added.  The kernels are load-bearing at every count.

### 1.6 The batch at small cells (the cone batch probe)

This section closes the one question §1.5 left open: whether the
count-scaled transient budget moves the realized batch at cells where
the budget is not clamped at its ceiling.  A nine-arm probe measured
the realized forward batch per device at the cone 384, cone 512, and
parallel 512 cells at every count, against predictions computed from
the live cost functions and registered before the run.

The predictions held to the digit.  Cone's batch reads 128, 128, and
85 at the 384 cell and 128, 128, and 113 at the 512 cell across one,
two, and four devices.  Parallel's holds 128 at every count.  These
results confirm the mechanism: the budget falls with the count,
cone's per-view cost does not, and the batch drops where the falling
cap crosses the chunk.  The shipped chunk of 128 floors the effect.
The drop therefore appears at four devices only, and only on the
full-pixel forward, whose calls sit as a second batch population
beside the subset forwards' 128.

The confirmed effect is marginal in cost.  Cone's summed launches at
four devices read 1376 against parallel's 1360, a 1.2 percent
difference.  These results close the budget-proportionality question
at these cells.  The proportionality is real, its measured cost is
percent-scale, and no knob change is warranted now.  The question
returns only if a tiling change moves the per-view cost.

The probe also read the call counts, and a correction applies
(2026-08-10): the recorded series are TOTALS over all devices, not
per-device counts.  The observer keys calls by the projection body
object, and the kernel bodies are one shared object across devices,
so all devices collapse into one key; the rows show exactly one
forward key per arm, carrying the last device's label.  Corrected,
per-device view-range calls read 85, 170, and 340 at one, two, and
four devices — the funnel's 85 entries times the band count — and
per-device kernel launches are FLAT in the count.  What grows is
per-device CALLS, linearly, from the banded walk visiting every
owner's band.  The instrument defect is recorded for the harness
lineage, and the corrected numbers are charter A's input.

### 1.7 The validating instrument (mg9): the flat span is kernel-busy time

Measured 2026-08-10, job 15152345 on h018, on the merged tip f985a6e.
The rows are
`plans/experiments/torch_port/rows/mg9_fwd_instrument_h018_20260810_152514.jsonl`.

The forward-remedy memo asked for one measurement before either remedy
was chosen: record copy time separately from kernel time inside the
forward call, and record each device's busy time separately from its
bracket, so waiting is distinguished from computing.  The instrument
added two readings to §1.5's per-device bracket.  The first is a CUDA
event pair around every individual projection body call, summed per
device (the busy time), with a call counter.  The second is the band
broadcast's host wall plus a device-side event pair around every copy,
bracketed on the source device's stream because that is the stream
torch runs a cross-device copy on.  Four arms ran: parallel 1024 at
one, two, and four devices, and cone 1024 at two.  The four-device
parallel arm closes the gap §1.5 recorded as never measured.

The reading is valid on five checks.  Every arm's built-in checks
passed, and no arm tripped the copy-stream warning.  The brackets
reproduce mg5's anchors to within 0.15 percent (28.90 against 28.87,
28.77 against 28.75, 30.65 against 30.61).  The reconstruction walls
match §1.2's within noise.  The view batches read as shipped, 128 at
parallel and 52 at cone.  The per-device wrappers carry the device
index by position, and the rows witness the §1.6 collapsed-key
situation directly (`fwd_bodies_distinct_objects` reads false at more
than one device), so the defect that corrupted mg7's counts cannot
recur here.

The table.  Each row is the largest reading over that arm's devices,
because the reconstruction waits for its slowest device.

| geometry | devices | bracket s | busy s | busy/bracket | gap s | broadcast device s | broadcast GB/s |
|---|---|---|---|---|---|---|---|
| parallel | 1 | 28.90 | 28.19 | 0.98 | 0.72 | 0.00 | — |
| parallel | 2 | 28.77 | 28.24 | 0.98 | 0.53 | 0.05 | 257 |
| parallel | 4 | 14.88 | 14.55 | 0.98 | 0.33 | 0.19 | 197 |
| cone | 2 | 30.65 | 29.70 | 0.97 | 0.95 | 0.06 | 197 |

The first conclusion is that copying and waiting are refuted as the
flat term.  The broadcast moves 12.4 GB per reconstruction at two
devices in 48 ms, which is 257 GB/s.  §1.5's data-movement inference
implied 0.44 GB/s, so the measured copy path is about 580 times
faster than the inference required.  The bracket-minus-busy gap is
0.3 to 1.0 s and it shrinks as devices are added.  The remedy memo's
serialization options, A2 and A3, could recover at most about one
second of a 29 s span, and they are declined.

The second conclusion is that the cost sits inside the kernel
launches.  Busy time is 97 to 98 percent of the bracket at every
count.  Per-device launches hold at 680 in every parallel arm while
each launch's band narrows with the count.  The memo's rule for this
outcome selects the per-launch remedy, option A4, gated on its 2K
residency pricing.  For cone the code-visible per-launch term is the
kernel grid's full-detector-rows axis, so a grid sized to the band's
detector-row span is the kernel-level variant to price beside A4.  That
cone variant was superseded the same evening, when a reading of
mbirjax's cone path led the remedy memo's §8 to replace it with a
pixel-batched full-height cylinder gather.  That same reading also
dissolved A4's 2K residency gate, because the gathered form never
assembles a whole cylinder (memo §8.4).

The third conclusion is new: parallel is flat only from one device to
two.  At four devices the span halves, with busy time 28.19, 28.24,
and 14.55 s at one, two, and four devices over the same 680 launches.
The per-launch time is 41.5 ms at the full and the half band and
21.4 ms at the quarter band.  The composed reconstruction at parallel
1024 scales 1.70x at four devices (39.96 s to 23.48 s), against the
1.02x that §1.2 measured at two.  These results indicate that
parallel's pathology is confined to the one-to-two leg, while cone
stays flat through four (§1.5).

One reproduction is worth its own sentence.  Cone at two devices
shows a back-projection device span of 30.33 s beside the forward's
30.65 s, which confirms §6.2's recorded rise and keeps the cone back
remedy a separate decision that no forward option addresses.

### 1.8 The shape sweep (mg10): one remedy declined, one adopted

Measured 2026-08-10, job 15158216 on h004, on the merged tip f985a6e,
in 1:07:01 of wall.  The rows are
`plans/experiments/torch_port/rows/mg10_shape_sweep_h004_20260810_174925.jsonl`.

The forward-remedy memo's §8 named two shapes and left three things
to be established on our own kernels: the parallel band knee, the cone
column-batch size, and the value gate for cone's order change.  mg10
measured all three.  It swept the parallel slice band at a fixed device
count, which is the single-variable arm §1.7 could not supply, and it
ran a column-gather prototype for cone over three pixel batches.  The
design note that consumes this reading is
`active/forward_remedy_design.md`.

The reading is valid on four checks.  All eighteen arms passed their
built-in checks, and no arm needed a thermal re-run.  All four anchors
from mg9 and mg5 reproduced within 0.5 percent.  Two arms are
noise-floor pairs by construction, and both are labelled as such.  The
asked-384 parallel arm collapses onto the asked-256 walk, because a
504-slice shard tiles the same way for both.  The asked-256 arm at four
devices is the control exactly, because a 252-slice shard caps the
request.  The value comparisons carry their own repeat arms, so every
distance is read against a measured floor rather than against an assumed
one.

The parallel sweep, at the 1024 cell with two devices, against a control
of 28.23 s of per-device busy time.

| asked band | realized walk | busy s | against control | per launch ms | per slice ms |
|---|---|---|---|---|---|
| control | 1 x 504 | 28.23 | 1.000x | 41.51 | 0.082 |
| 64 | 8 x 63 | 25.55 | 0.905x | 4.70 | 0.075 |
| 128 | 4 x 126 | 29.89 | 1.059x | 10.99 | 0.087 |
| 192 | 3 x 168 | 34.70 | 1.229x | 17.01 | 0.101 |
| 256 | 2 x 252 | 28.85 | 1.022x | 21.21 | 0.084 |

The cone batch sweep, at the 1024 cell, against the banded control's
per-device busy time.

| pixel batch | n=2 busy s | against control | n=4 busy s | against control |
|---|---|---|---|---|
| banded control | 29.69 | 1.000x | 29.32 | 1.000x |
| 2048 | 24.57 | 0.828x | 20.61 | 0.703x |
| 4096 | 20.65 | 0.696x | 15.60 | 0.532x |
| 8192 | 19.40 | 0.654x | 15.27 | 0.521x |

The first conclusion is that the parallel fixed-band shape is declined
as a time remedy.  Per-launch time is linear in the band within a device
count, so a narrower band raises the launch count by the factor it
lowers the per-launch cost.  Three of the four swept bands are slower
than the control, by 2, 6, and 23 percent.  The 9.5 percent win at the
63-slice walk is real and clean, at a value distance of 7.80e-10 against
its own repeat floor of 7.22e-10, and it is non-monotonic and
unexplained.  It is recorded and not built on.  The shape survives as
the memory knob mbirjax's own record uses it for.  Per-device peaks fall
from 12.48 GB to 11.84, 11.88, 11.91, and 11.97 GB across the four
walks, and the copied bytes are unchanged at 12.44 GB per
reconstruction.  The knob already exists as
`forward_project_slice_band`, so no library default change is proposed.

The band knob moves no value.  Every swept band sits at or within 8
percent of its own repeat floor on both distance metrics.  The checksum
distances run 9.24e-11 to 7.80e-10 against floors of 3.26e-10 to
7.44e-10.  The sample distances run 1.72e-07 to 2.38e-07 against floors
of 1.74e-07 to 2.31e-07.

The second conclusion is that the cone column gather breaks cone's
flatness and is adopted; the checkpoint ruled the same evening, and the
design note's status header carries the ruling with its two verified
conditions.  The banded control
moves from 29.69 to 29.32 s between two and four devices, and the column
gather at batch 8192 moves from 19.40 to 15.27 s over the same step.
Cone's forward falls when devices are added for the first time in this
campaign.  Three further readings support adoption.  Busy time was still
falling at the largest batch measured, so the knee sits at or above 8192
and the library increment must sweep higher.  Per-device peaks read
12.76 to 12.81 GB against the banded control's 14.31 GB, so the gathered
form is cheaper in peak as well as faster.  The composed walls fall with
the busy times, to 58.45 s at two devices and 39.13 s at four, against
banded controls of 67.16 s and 52.96 s and a one-device anchor of
61.54 s.  Cone's two-device leg therefore stops being a regression.

The value gate is the one place the prototype does not clear the bar the
memo set, and the design note carries the ruling request.  The rule was
that the column gather's distance to the one-device anchor may not
exceed the banded walk's.  On the checksum metric at two devices the
column gather reads 2.47e-10, 2.58e-11, and 7.09e-11 against that
anchor, below the banded walk's 1.01e-09.  At four devices it reads
4.13e-11, 1.59e-10, and 9.25e-11, where the banded walk reads 7.20e-11.
All four four-device readings sit below their own repeat floors of
6.59e-10 to 1.13e-09, so that metric cannot resolve the ordering there.
The sample metric is the more sensitive of the two.  On it the column
gather reads 1.47e-06 to 1.54e-06 from the anchor, at every batch and
both counts.  The banded walk reads 2.83e-07 at two devices and
5.41e-07 at four, at its own pass-to-pass floor.  The excess is expected rather than
anomalous, because the column gather changes the summation structure
twice where the banded walk changes it not at all.  Its size is small
against the bar the library ships: 1.5e-06 sits three orders of
magnitude inside the standing parity suite's 5e-3 relative floor.
Which bar governs is `active/forward_remedy_design.md` §12's sixth
question.

One qualification belongs with the time numbers.  The prototype issues
its gathers serially, so its brackets carry unoverlapped stalls of 1.4
to 9.4 s.  Busy time is therefore the target a real implementation aims
at, while the bracket is a conservative floor.

The third conclusion is that the parallel mystery has moved rather than
closed.  Per-slice time at two devices runs 0.075 to 0.101 ms over bands
from 63 slices to 504, with no trend in the width, and it is 0.041 ms at
one device at width 1008 (§1.7).  The factor of two therefore sits
between one device and more than one device, and not between one band
width and another, which is the opposite of what §1.7's three points
suggested.  One device is also the only place a 1008-slice band was ever
measured, so the two readings stay confounded.  The discriminating arm
has not been run and is cheap: parallel at one device with the band knob
set to 504.  A per-slice reading near 0.041 ms would make the factor of
two a multi-device effect.  A reading near 0.082 ms would make it a
kernel width effect spanning 504 to 1008, because 0.082 ms is what two
devices read at that same width.  Neither adopted decision depends on
the answer, and the answer decides whether the column-gather shape is
also the right remedy for parallel.

### 1.9 The discrimination: the parallel flatness is a kernel width effect

Measured 2026-08-10, job 15159551 on h008, on the same frozen tree.
The rows are
`plans/experiments/torch_port/rows/mg10_shape_sweep_h008_20260810_201612.jsonl`.
Nine arms ran: the one-device control, three one-device arms with the
values block cut into fixed-width pieces, view-chunk arms at 8 and 16
at one and two devices, and the two-device control.

The reading is valid on four checks.  All nine arms passed their
checks, including the piece-structure witnesses on both sides of every
patched arm.  The one-device control reproduced §1.7's anchor at
0.0411 against 0.0412 ms per slice.  Every cross-arm value distance
sits at its own repeat floor (checksum spreads 1.3e-12 to 7.8e-10
against floors of 2.2e-10 to 1.0e-09), so every arm computes the same
reconstruction.  The strided-piece packing cost was measured per arm
at 0.2 to 1.3 ms per launch set against 21 to 41 ms launches, so
netting it out changes no conclusion.

The discrimination table, one device throughout:

| values-block width | pieces | per launch | per slice | busy |
|---|---|---|---|---|
| 1008 (the shipped call) | 1 | 41.38 ms | 0.0411 ms | 28.14 s |
| 504 | 2 | 41.46 ms | 0.0823 ms | 56.39 s |
| 252 | 4 | 21.18 ms | 0.0840 ms | 57.60 s |
| 63 | 16 | 4.69 ms | 0.0744 ms | 51.02 s |

The first conclusion is that the doubling is a property of the kernel
and the width alone.  Cutting the one-device call into two 504-wide
pieces doubles its cost, and a launch at width 504 takes the same
41.4 ms a launch at width 1008 takes.  Per-slice cost at every width
at or under 504 matches what two and four devices read at their
matched widths, so the device count contributes nothing.  The
multi-device hypothesis is refuted.

The second conclusion is that the write slab's L2 residency is the
sweep's small anomaly and not its doubling.  The view-chunk arms put
the accumulation slab well inside L2 at both counts, and busy time
did not move: 28.90 against 28.14 s at one device, 28.98 against
28.17 s at two.  The width-63 bonus reproduced at one device at the
same ten percent.  These results indicate that L2 residency buys the
ten percent and the width regime costs the factor of two.

The third conclusion is the remedy.  At more than one device the
banded walk hands the kernel 504-wide or narrower blocks, which is
the inefficient regime; the column gather hands it full-width blocks,
which is the efficient one, at a bounded transient the 2K point
survives.  The column-gather shape adopted for cone is therefore also
parallel's remedy, for a different measured reason, and for parallel
it is order-preserving because each detector row keeps a single
producing piece.  From the measured rates, parallel 1024 at two
devices should fall from 28.2 to about 14.1 s of forward busy and
from 39.2 to about 25 s composed.  The design note's addendum carries
the extension proposal.  Why the kernel runs twice as efficiently at
width 1008 as at 504 remains unexplained; the leading candidate is
grid occupancy, and the question now belongs to the kernel campaigns,
because no adopted remedy depends on the answer.

---

## 2. The ledger at n>1 (mg2)

This section answers whether the memory model the widening rule
trusts is accurate at the counts where it actually operates.  The
ledger was calibrated at one device only, and the preflight consumes
it at n>1, so mg2 measured modeled against measured per-device peaks
at n = 2 and 4.

The ledger is safe at n>1 and mis-shaped at n=4.  No per-device
reading sits below 1.00, so the preflight cannot start a doomed run.
Sixteen of thirty readings sit above the 1.30 ceiling, all in the
over-charge direction, worst at n=4: 1.31 to 1.59 at parallel 512,
1.42 to 1.43 at parallel 1024.  An over-charging ledger falls back to
fewer devices earlier than it needs to near capacity.

The mis-shape is localized.  The modeled dominant phase at every
weighted cell is the direct-recon back loop, and the band-reduce term
is charged at 26 to 60 percent of the measured peak.  The over-charge
concentrates exactly where those two n>1-specific charges are largest.

The over-charge admitted two readings, and a sub-phase probe decided
between them.  The answer is a third reading.  Both sub-steps of the
back loop are real, and the ledger charges them together although they
never run at the same time.  The probe measured the workers' peak and
the reduce's peak separately, on every device.  At parallel 1024 with
four devices the charge exceeds the larger of the two by 4.2
cylinders, which is about 3.1 GB per device.  At cone 1024 with four
devices it exceeds it by 4.1 cylinders.  These results identify the
phantom sum as the dominant error, and its remedy is a charge
correction.

The probe also confirmed one real cost that the ledger does not charge
at all.  Every band pass entered with two extra cylinders live on the
first device and one on the others, at every arm.  That signature is
the previous pass's per-device results staying alive through the next
pass.  The cost is therefore real and avoidable, and its remedy is the
one-line release described in `backloop_attribution.md`.

One term stays undecided here.  The back batch charge is real in
kind and conservative in magnitude, and the probe's worker-transient
readout that prices it sits in the mg6 rows.  The charge
re-derivation consumes that readout, and the margin ruling waits on
the same re-derivation.

One measurement refines the code reading behind that memo.  The live
block count in the view loop measured 2.49, 0.99, and 1.97 cylinders
where the reading predicted 3, 2, and 3.  The loop holds about one
cylinder less than predicted.  The charge correction should therefore
follow these measurements rather than the reading.

The `hess_weights` question closes.  Unweighted and weighted peaks at
parallel 1024 differ by -0.06 GB at n=2 and -0.03 GB at n=4, so the
declined release stays declined with n>1 evidence behind it.

**The release is verified, and this section closes (2026-08-10).**  The
verification ran on the code of the merged tip and returned four green
signatures.  The standing 2-GPU kernel gate passed 12 of 12.  The
mg2-style re-calibration measured post-release peaks at all ten cells,
and no per-device reading fell under 1.00.  The minimum was 1.103, with
the pre-release pad still in place.  Each release saving matched its
registration.  Cone 1024 at four devices fell 10.5 percent, exactly as
registered.  Parallel 1024 at four devices did not move, also exactly
as predicted, because its peak is a reduce and no release touches a
reduce.  Parallel 1024 at two devices fell 11 to 16 percent, down to
its next phase floor, and the 512 cells were essentially unmoved.  The
mg6-style re-probe then read the registered residency signature
exactly.  Band-pass entry steps are +1 cylinder on device 0 and 0 on
the other devices, at every arm.  The stale partial is therefore
eliminated, and the surviving +1 cylinder is the finished own band,
which is the pass's output.  Live block counts sit at the rider's
min(2, nb) ceiling, and cone 1024 at four devices dropped a full
cylinder to 0.97.

The pre-release pad is retired, and the ledger now carries no fitted
constant.  The `forward margin (pre-release)` placeholder is replaced
by two closed-form terms.  The forward block spans the detector rows on
a two-fan geometry, which is the whole cone correction.  The view-range
loop's second live block is charged as min(2, batches) - 1.  Against
the post-release rows, zero of thirty readings sit under 1.00 and the
minimum is 1.004.  The count above the 1.30 ceiling falls from 12 to 6.
The 1.004 minimum is a back phase, the hessian's band reduce, and that
phase is bit-identical across the release.  That reading is therefore
the genuine peak instant at its cell, not a term the release masked.
The ledger question this section opened is CLOSED.

### 1.10 The flip gates passed, and the column gather is now the default (mg11)

Measured 2026-08-11, job 15163071 on h001, on the tree of commit
a33c7e8.  The rows are
`plans/experiments/torch_port/rows/mg11_flip_gates_h001_20260811_041522.jsonl`,
and the job's log sits beside them.

mg11 is the combined gate campaign the checkpoint ruled for the design
note's increment 7 and its parallel extension.  It ran 22 arms at the
1024-class cell: a one-device anchor per geometry, a banded control at
two and at four devices, and column-gather arms over a pixel-batch
sweep, with three batches on cone and four on parallel.  Each geometry
was read against three gates: speed, value, and memory.  All six
readings passed, and the harness printed the flip authorization for
both geometries.

The speed gate asks the best gather arm to beat the banded control by
more than the control's own pass-to-pass spread.  Cone composed wall
fell from 67.12 s to 55.05 s at two devices and from 53.00 s to
37.28 s at four, ratios of 1.219 and 1.422.  Parallel composed wall
fell from 39.21 s to 27.88 s at two devices and from 23.46 s to
18.53 s at four, ratios of 1.406 and 1.266.  The smallest margin was
4.93 s, against a control spread of 0.04 s.

The value gate is the shipped parity floor, which the checkpoint ruled
the governing bar.  The largest distance any gather arm showed from
the one-device anchor was 4.887e-06 relative, against the 5e-3 bar.
The checksum comparison against the banded control stayed in the
1e-09 class on every arm.  The logged
expectation held as well: the relative L2 distance to the one-device
anchor read 1.474e-06 on cone and 1.410e-06 on parallel, against the
1.50e-06 class mg10 registered, and no reading approached the 1e-05
marker.

The memory gate is the ledger floor.  The smallest modeled-to-measured
ratio over the gather arms was 1.003, and every arm of the campaign,
controls included, sat inside the 1.00 to 1.30 calibration band, with
1.158 the largest reading.  The library's own `last_memory_ledger`
agreed with the harness's independently built ledger on every arm.
These readings also serve as increment 8's re-calibration for the
gather path: the closed-form column terms landed with the
implementation, and every cell they price now has an in-band measured
reading.

The parallel extension's caveat check printed beside the parallel
verdict.  Section 13 predicted a 2.00x forward-busy improvement at two
devices with the pixel count held full; the measured improvement at
the swept batches was 1.76x.  The prediction named that direction: the
gather cuts each call's pixel count to the batch, and a narrower call
runs the kernel less efficiently.  The gate did not depend on reaching
the prediction.

The batch sweep moved the best batch above the shipped default, and
the default deliberately stays.  Composed wall kept improving through
16384 and 32768 depending on the cell, by 4 to 15 percent over 8192,
so the knee is still not bracketed.  These readings come from the
1024-class harness.  Production runs at the 2K class and above, where
the batch's cross-device transient grows with the slice axis, and no
sweep has run there.  `FORWARD_PIXEL_BATCH` therefore remains 8192,
its comment records this sweep, and the 2K sweep belongs to the
production-scale charter.

The default flip landed in the library the same morning.  Unset
`forward_column_gather` now selects the gather on cone and parallel,
an explicit False selects the banded walk, and the environment
variable still overrides in both directions.  The full suite passes in
all three environment states: variable unset, gather forced, and
banded forced.  The banded-forced state was never green before,
because the gather-specific tests did not pin their shape against the
environment; they now clear the variable the way the banded-specific
tests always did.  The widening-floors staleness note is tripped by
design and passes stale.  The floors re-measure runs on the committed
flip, because the refresh script stamps the commit it measured.

### 1.11 The first GPU nightly on greg_dev found a compiler defect at
### single-pixel calls (2026-08-11)

The nightly regression of 2026-08-11 ran greg_dev for the first time,
on a four-GPU node, against commit a33c7e8.  main passed clean.
greg_dev reported two test failures, and this section records what
each one is.

The first failure is a test whose discrimination premise is
input-lucky, not a library defect.  In
`test_masked_single_device_plastic_floor_keeps_the_unsharded_arithmetic`,
the assertion that the library keeps the float32 masked-sum floor
expression passed.  What failed is the test's second demand: that the
float64-divide variant produce a visibly different output on its
seeded input.  On the GPU node the two floor values differ at the
eighth decimal and zero output elements move, so the inequality
assert fails with the library fully correct.  The fix is to make the
discrimination robust to the platform, and it is being made in the
test only.

The second failure is a real wrong answer with a diagnosed mechanism:
torch 2.13.0's CPU compiler on linux miscompiles the parallel-beam
projector bodies when their first compilation is for a single-pixel
call.  The failing test leg drives the column gather at a pixel batch
of one, which makes every call carry one pixel, and it read a 6.56e-02
relative error against a comparison class of 1e-07.  The diagnosis
ran as a chain of single-variable measurements on the cluster, each
eliminating one layer.  The error reproduces in a fresh process, so
it does not depend on test order.  Pixel batches of two, three, and
four are correct, and a width-one remainder call after a wider batch
is also correct, so the defect is specific to width one.  A
single-device model shows the same error when one pixel is projected
alone through the plain projectors, so the sharding drivers are not
involved.  The geometry chain and the per-tap weights are bit-identical
between the solo and the batched evaluation, so the library's
arithmetic is not involved.  With compilation genuinely disabled
(TORCHDYNAMO_DISABLE=1) the solo call is correct at 1.05e-07, and with
it enabled the same call misplaces the pixel's footprint one full
detector channel to the right.  These results identify the compiled
width-one specialization as the defect: a body first compiled at a
larger width handles width one correctly through its dynamic-shape
recompile, and only a body whose first compile is at width one emits
the wrong kernel.  The back projector body carries the same defect,
measured at 5.04e-02 on the same probe.  The cone bodies do not: their
width-one test legs passed on the same node the same night.  macOS
compiles the same width-one bodies correctly, which is why the local
suite never showed it.

The defect predates the column gather and is reachable without it.
Any caller of `sparse_forward_project` or `sparse_back_project` with a
single pixel, on a compiled parallel-beam model on linux, gets the
wrong answer on the committed tree.  The gather merely added a second
route to the same width through its pixel-batch knob.  The mg11 gates
are unaffected: their batches were 4096 and above with remainders of
256, every call was wide, and the value readings stand.

The remedy is to keep the compiled bodies away from width one
entirely.  A thin uncompiled shim pads a single-pixel call to two
pixels -- a zero-valued column on the forward, whose contribution is
exactly zero, and a duplicated index with an output slice on the back
-- so every compiled call has width two or more, which is the measured
correct region.  The shim is gated to parallel beam, changes nothing
for wider calls, and lands with a regression test asserting the
solo-pixel identity on both directions.  The fix is its own increment
beside the flip, because the flip neither introduced nor enlarges the
defect.

### 1.12 The floors refresh on the new default, and the kernel probe
### (2026-08-11, job 15172987)

The widening floors were re-measured on commit 4a222c7, the first
refresh under the column-gather forward.  The refresh ran on a
four-GPU node in 33 minutes, and its block is pasted into
`_widening_floors.py` with the staleness note now clean.

Three floors confirmed and one moved.  Parallel and cone at two
devices stay at the 512-class shape, both now winning by 1.21x where
cone's 2026-08-10 admission had been marginal at 1.02x.  Cone at four
devices stays at the 1024-class shape, re-derived against two devices
as the crossover rule requires, winning by 1.45x.  Parallel at four
devices MOVED DOWN: the floor fell from the 1024-class shape to the
768-class, a 3.4x smaller admission size.  These results indicate the
new forward did not only speed up the admitted shapes; it widened
where four devices pay for themselves at all.  Translation and
multiaxis still declare no floor family, so the parallel floors
continue to govern their automatic device count, and their own
measurement remains open.

The one-pixel kernel probe rode the same session and closed the last
open corner of the single-pixel question.  The hand-written CUDA
kernel bodies, which the padding fix deliberately does not wrap,
match the full pass at one-pixel calls in both directions: 3.15e-07
relative on the forward over forty solo calls, 3.51e-07 on the worst
back row.  The compiler defect of §1.11 is therefore confined to the
compiled bodies, and the padding fix covers every affected path.

### 1.13 The copy streams land: the transfer stall closes, and the busy
### reading is corrected (mg12, 2026-08-11)

Measured 2026-08-11, job 15175187 on h012, 37 minutes.  The rows are
`plans/experiments/torch_port/rows/mg12_stream_gate_h012_20260811_113813.jsonl`,
and the harness and job files sit beside the other mg entries.

mg12 measured three trees in one session, five configurations each.
The control was the committed tip.  The first overlay added only the
prefetch: the driver issues each batch's gather one batch ahead of the
projection that reads it, with every copy still on the device's
default stream.  The second overlay added dedicated per-device copy
streams with per-batch event ordering on top of the prefetch.  Every
arm witnessed its transfer route, and all fifteen ran direct
device-to-device.

The copy streams won everywhere, by far more than the visible stall.
The forward wall fell from 16.57 to 9.32 s at the widest-stall
configuration (parallel, four devices, batch 4096) and by comparable
ratios at the other four, and the composed reconstruction followed:
1.41x, 1.31x, 1.20x, 1.27x, and 1.07x across the five configurations,
with pass-to-pass spreads two orders below the margins.  Values sat at
the control's own repeat floor on every arm, and no memory reading
fell below the model's floor: the smallest modeled-to-measured ratio
was 1.004, and the measured allocated peaks were unchanged, so the
concern that a second allocator pool would raise the peak did not
materialize.

The size of the win exposed a measurement error this page must
correct.  GPU-busy time fell along with the wall -- 12.70 to 7.00 s at
the widest configuration -- and busy was defined as the event-bracketed
time of the projection calls alone, which a transfer change should not
touch.  The explanation is that the old busy reading was never pure:
with one stream per device, the copies serving OTHER devices' gathers
interleave between a device's own kernel launches, inside the
bracketed windows.  The visible stall (wall minus busy) was therefore
only the part of the transfer cost that fell between brackets, and the
rest hid inside the busy column.  Moving the copies to their own
streams removed the contention and purified the metric in one step.
The correction's reach: every busy reading taken at more than one
device on the gather path (mg10, mg11, and the §1.9 table's
multi-device columns) carries the same contamination, and the
conclusions built on one-device readings -- the kernel width effect
above all -- are untouched, because a single device serves no peer
copies.

The prefetch-only prediction was refuted, five of five.  Issuing the
copies early bought 0.8 to 2.4 s of wall on its own, against a
prediction of no change.  The refuted argument assumed a copy and a
projection on one stream cannot overlap at all; the measured gain says
the issue order still matters within a stream's schedule.  The
prediction check cost one arm per configuration and converted a wrong
belief into a measured one, which is what it was for.

Both overlays are one staged increment: the prefetch and the copy
streams, with the resident-count charge at three, its tests, and the
ordering witness.  The design note's increment 5 closes with this
measurement.  Increment 6, the per-batch accumulation, remains open,
and the corrected busy metric is the right instrument to measure it
with.

### 1.14 The sorted-channel probe: the win exists, the compiler undoes
### it, and the production kernel outruns both (mg13, 2026-08-11)

Measured 2026-08-11, job 15183553 on h009.  The rows are
`plans/experiments/torch_port/rows/mg13_sorted_probe_h009_20260811_153801.jsonl`.
The job crashed after its four one-device arms: the conda
environments were rebuilt at that moment (deliberate maintenance on a
corrupted unrelated environment), which removed the base interpreter
the job's per-arm subprocesses resolve through a symlink, so the
four-device composed arms never ran.  The one-device arms are the
question, by the harness's own design, and they completed.

The probe measured the light per-call sorted form: sort each call's
detector-channel writes, reduce the runs with `torch.segment_reduce`,
and write once per channel, replacing the atomic scatter.  Three
bodies ran on the same inputs at the 1024-class cell: the hand-written
triton kernel (the production path), the torch scatter (the baseline
the sorted form modifies), and the sorted candidate.

Three readings, in eager mode, at full width.  The sorted form beats
the torch scatter by 1.17x at collision ratio 12, 1.22x at ratio 25,
and 1.32x on the full pass at ratio 2332.  The win grows with the
collision ratio, which is the physics mbirjax's record describes.
Values sat in the 1e-08 class against the scatter everywhere, and the
memory-bounding chunked sort was bit-identical to the unchunked form.

The compiled reading inverts the eager one.  Under torch.compile the
sorted body takes three to four graph breaks where the scatter takes
none, and the per-launch time reads 12.0 ms against the scatter's
6.4 ms — the composed reconstruction pays 174 s against 140.  The
reduction is faster; the broken compiled region is slower.

The production kernel outruns both torch forms by about 3.5x.  The
triton body ran the full pass at 6.8 ms against the sorted form's
17.9 and the scatter's 23.6, and the composed reconstruction at 41 s
against their 140 and 174.  These results indicate that the sorted
idea has no role as a torch-level replacement in mbirtorch: even
perfectly compiled, its ceiling sits far above the kernel the parallel
path actually runs.  What the probe validates is the collision-ratio
physics itself, and the one place that physics can still pay here is
inside a kernel: sorted or segmented accumulation on-chip, which is
the cache direction the checkpoint already assigned to the kernel
campaign.  The item-13 stop therefore stands, with its rationale
sharpened from "not worth it at the measured ratios" to "the win is
real but lives below the level torch code can reach."

One operational note: the job printed its completion sentinel after
the crash, because the sentinel line was unconditional.  Future
harnesses print it only on a completed arm set.

### 1.15 The fused accumulation lands, and a verdict line asks the
### wrong question (mg14, 2026-08-11)

Measured 2026-08-11, job 15183914 on h009, 13 minutes.  The rows are
`plans/experiments/torch_port/rows/mg14_accum_gate_h009_20260811_163857.jsonl`.
An earlier submission of the same job died at its first interpreter
check when the conda environments were rebuilt (the same maintenance
event that ended mg13); the environments were restored within the
hour and the job ran unchanged.

The change under measurement is the design note's increment 6.  Each
pixel batch's projection previously allocated a fresh full-size
output block, filled it, and handed it back for the driver to add
into the running total — one allocation and two shard-sized passes
per batch, at a hundred or more batches per pass.  The fused form
adds every batch after the first directly into the running total,
inside the projector's own view loop, through one optional argument
on a plain python method.  The same summands are added in the same
order, so the values are bit-identical by construction.

The measurement confirmed it everywhere it should show and nowhere it
should not.  The forward wall fell 9.34 to 8.74 s, 8.37 to 8.07, and
9.19 to 8.87 across the three configurations — 3.5 to 6.4 percent —
against pass-to-pass spreads of a millisecond or less, and the
composed reconstructions moved with it.  GPU-busy time was flat to
0.2 percent, as it must be: the kernels are untouched, and the
removed work lived between the kernel windows.  The measured
per-device peak fell by 0.92 to 0.95 GB — the vanished per-batch
block — while the model deliberately still charges the two-block
shape it shares with the banded path, so the floor readings rose from
1.007 to 1.160.  Values sat at the control's own repeat floor in
every comparison.

The verdict lines printed NO CHANGE, and the reason is a lesson worth
recording: the job was briefed to draw its verdict on busy time, on
the mistaken premise that the removed work sat inside the busy
windows.  Busy sums only the kernel-body event pairs; driver-side
work between them shows in the bracket, which is where the design
note's own guidance for the sibling increment points.  The harness
printed the flawed premise verbatim and concluded conservatively; the
readings underneath were unambiguous, and the checkpoint's standing
rule — a mechanical verdict contradicted by the printed readings goes
to a person — is what carried the result.  Increment 6 closes with
this measurement, and with it every implementation increment of the
forward remedy is complete.

### 1.16 The padding-removal gate on real GPUs, and a compiled-reduction
### finding (mg15, mg15b, mg15c, mg15d, 2026-08-16)

Measured 2026-08-16 in four jobs: 15304566 (h016, the gate), 15304687
(h014, the reconstruction adjudication), 15304817 (h007, the adjoint
matrix), and 15304868 (h014, the compile ablation).  The rows are the
four `mg15*` files under `plans/experiments/torch_port/rows/`.  This
is the padding-removal design note's P5: the local virtual-CPU tests
had already verified the unpadded split's values, and these jobs add
what a laptop cannot show — real multi-GPU runs at device counts that
do not divide the sharded axes.

The gate ran three legs at the 512-class cell.  The forward-projector
leg passed on every arm: cone and parallel read 5.7e-7 and 5.4e-7 at
three devices, where both axes split unevenly, and multiaxis read
1.6e-5 at four, all against a 1e-4 gate.  The memory leg passed on
every device of every arm, with the lowest modeled-over-measured
ratio at 1.016.  The same job re-measured the one ledger-calibration
arm whose split the removal changed, and the replacement `ma512_n4`
row now sits in `tests/test_memory_ledger.py` with its provenance.
Five readings sat above the 1.30 band top, all on the cone and
parallel three-device arms, the largest at 1.417.  An over-charge is
a warning rather than a failure, and it is recorded here as
calibration context for the next ledger pass.

One reading failed: the multiaxis four-device reconstruction differed
from its single-device reference by 2.97e-2 against a 1e-2 backstop.
Three probes adjudicated it.  mg15b added the adjoint instrument the
gate lacked and read 6.29e-4 on a single back projection, with
same-count reconstruction repeats at 2e-6 to 3.4e-6, decay to
1.68e-2 at ten iterations, and no error concentration at the block
boundaries.  mg15c then measured the full count matrix.  One and two
devices agree at 7.0e-7.  Three devices, where the views split
unevenly, and four devices, where the slices do, each sit near
6.29e-4 from every other count, and the same-count repeats read
exactly zero.  The same job ran the padded
672edbd tree, which read 5.98e-7 at four devices, and the cone and
parallel adjoints, which read 7.1e-7 and 6.9e-7 at their uneven
splits.  These results localize the effect precisely: it is
deterministic, it appears only on the torch-body geometry, only at
uneven splits, and only on the unpadded tree.

mg15d isolated the mechanism with one variable.  With
`compile_mode='off'` the same uneven four-device adjoint reads
5.98e-7; with the default compilation it reads 6.29e-4; and an
eager-against-compiled comparison at a fixed count reads 6.41e-4.
These results identify torch.compile as the mechanism.  They also
give the scale its meaning: a compiled body's reductions contract in
a different order than eager's, and that reordering moves this body's
output by about 6e-4 max-relative at this size.  An even split compiles one static shape
per device instance, and all such instances agree to 1e-7.  An uneven
split hands each instance a second block length, the second
compilation orders its reductions differently, and cross-count
comparisons then read the compiled-against-compiled analogue of the
eager-against-compiled difference.  The design note's own review
predicted this compile path, as a speed cost; the value consequence
is what these jobs measured.  Two of the probes' verdict lines misread their own data before a
person read the numbers.  mg15b's boundary diagnostic would have
convicted a correct split, because differences of any origin are
seeded at the shard seams and spread inward with iterations.  mg15d's
positive control chose the eager-against-compiled difference as its
noise floor, and that difference has the same mechanism and magnitude
as the effect under test.  Both corrections are written into the
probes.

The verdict on P5: the split is correct.  Data placement is verified
by the matrix, the boundary profile, the padded-tree ablation, and an
eager adjoint at 6e-7 on the very split in question.  What changed
with the removal is numerical identity, not placement: torch-body
geometries at uneven splits now carry a deterministic
compiled-reduction difference of about 6e-4 max-relative (5.6e-6
normalized RMS) between device counts, which amplifies to the 1e-2
scale over three VCD iterations and decays with further iterations.
The Triton geometries are unaffected.  Three remedies are recorded
for a later decision: accept the property and gate cross-count
torch-body comparisons at the 1e-3 full-pipeline level; mark the band
dimension dynamic unconditionally so every count runs the same
compiled code; or pad compile shapes alone.  The first is in effect
implicitly, and no landed gate is threatened by it.

Two harness lessons from these jobs are worth keeping.  A projection
taken before a reconstruction runs on the model's construction
placement, because the environment pin acts only through the device
policy, so a value instrument must run after the settle or it
compares a volume against itself.  And an adjudication probe needs a
positive control plus a null reference measured in the same run,
because three of this campaign's four diagnostic instruments would
otherwise have printed confident wrong verdicts.

### 1.17 The floors refresh seeds the denoiser family with sentinels
### (mg16, 2026-08-16)

Measured 2026-08-16, job 15304592 on h016, 31.7 minutes, on the same
verified dba9652 tree as mg15.  The rows are
`plans/experiments/torch_port/rows/mg16_floors_h016_20260816.jsonl`.
This is the entry-point plan's increment 5 measurement: the full
widening-floors refresh, extended with the denoiser family's first
probes.  Running it after mg15's gate meant the split it measured on
had just been verified on the same hardware.

The four existing rows did not move.  Cone admits two devices at the
512-class (1.29x) and four at the 1024-class (1.71x); parallel admits
two at the 512-class (1.26x) and four at the 768-class (1.02x).
These margins match the 2026-08-13 refresh within the run spreads.
This stability was expected: every measured floor cell divides evenly
at its measured counts, so the pad removal left those splits
unchanged.  The refresh also re-blessed the three cost-input hashes
the prerelease merge had drifted, with a real measurement behind the
re-bless.

The denoiser family produced no admission size at all.  Two devices
lost at every probe cell, reading 0.58x at the 512-class and 0.65x at
the 1024-class; four devices read 0.44x to 0.59x.  The spreads sat
between 0.6 and 2.3 percent, and every arm realized its pinned count,
so the readings are believed.  The mechanism is visible in the
absolute times: a three-iteration denoise of a billion voxels takes
2.2 seconds on one H100, so the sweep is dominated by its per-subset
host synchronizations and a split has nothing to amortize.  Both rows
therefore entered the table as sentinels, the first shipped sentinel
rows since the mechanism was built.  The consequence is the design
outcome of entry-point increment 5.  An automatic denoiser stays on
one device at every measured size, and the run log names the sentinel
when devices sit idle.  Only capacity widens the layout, when an
image can no longer fit on one device.  The losing ratio rises with size, so a
ladder above a billion voxels may yet find an admission point;
`largest_tested` records where this refresh stopped.

### 1.18 The column gather wins on translation and multiaxis
### (mg18, 2026-08-17)

Measured 2026-08-17, job 15307729 on h014, 1 hour 34 minutes, on the
synced 78b4f78 tree.  The rows are
`plans/experiments/torch_port/rows/mg18_ab_h014_20260816_231137.jsonl`,
and the run detail is in `mg18_column_gather_ab.md` beside the
script.

This is the measurement the forward-remedy record reserved for the
two geometries without hand-written kernels.  Translation and
multiaxis still run the banded multi-device forward, and the recorded
rule is that a geometry switches to the column gather only on its own
measurement.  mg18 compared the two drivers at each geometry's
production cell, mg8's ma1024 and tct2k, at two and four devices,
with values gated before timing.  No default was flipped: each arm
forced its driver on its own model instance and recorded the driver
the model reported.

The reading is valid on the arm witnesses.  All 24 arms ran and none
was refused.  Every arm realized its pinned count, bound torch bodies
in both directions, and ran the driver it claimed.  Both cells split
both sharded axes evenly at both counts.  The drift witnesses read
0.0 percent on both geometries, and the warm-repeat floors read
1.8e-6 on multiaxis and 3.2e-7 on translation.

The values gate passed on every arm.  Every distance to the
one-device reference sat between 9e-7 and 2.5e-5 against the 1e-3
gate, inside the registered e-5-to-e-6 expectation.  The
banded-to-gather distances at a fixed count read the same class, and
the composed reconstructions agreed across drivers at 2.8e-6 to
2.6e-5.

The gather is faster everywhere measured.  The table gives the ratio
of the banded median to the gather median at the shipped batch of
8192; above 1.00 means the gather is faster.  The 32768-batch arms
are in the rows.

| geometry | reading | n=2 | n=4 |
|---|---|---|---|
| multiaxis | forward | 1.27x | 1.86x |
| multiaxis | composed | 1.13x | 1.20x |
| translation | forward | 1.86x | 25.4x |
| translation | composed | 1.37x | 1.94x |

Three further readings give the ratios their meaning.  First, the
gather repairs the translation four-device forward outright.  The
banded walk took 3.69 s where one device takes 0.36 s, and the
gather takes 0.10 to 0.15 s, so under the gather this forward gains
from added devices for the first time.  Second, the multiaxis
forward now scales: 17.6 s at two devices and 9.0 s at four under
the gather, against the banded walk's 22.3 and 16.6.  Third, the
multiaxis composed anti-scaling is not the forward's.  The composed
wall still rises from 394 s at two devices to 951 s at four with the
forward fixed, so the rise lives in the back projection or the
phases around it, which this change does not touch.

Memory moved with the speed at the shipped batch.  The gather's
per-device peaks sat below the banded walk's on every batch-8192
arm, with the largest saving on the translation four-device forward,
26.5 GiB to 10.6.  The 32768 batch raised the multiaxis four-device
peak to 23.2 GiB against the banded walk's 11.8 and bought no speed
over 8192 there, so the shipped 8192 default keeps its reasons.

The recommendation is to switch both geometries' defaults to the
column gather.  Translation wins at every count and its four-device
pathology disappears; multiaxis wins at every count.  The flip is a
reviewed change: it sets `column_gather_geometry = True` on the two
classes, brings the parity suites along, and re-raises the question
of these geometries' own widening floors, which today reuse
parallel's (item C2) and were measured here to scale differently.
The decision is Greg's.

RULED (Greg, 2026-08-17): use the gather.  The flip landed the same
day: both classes now declare `column_gather_geometry = True` with
the measurement in their comments, the suite passes in all three
environment states, the pinned banded-path calibration rows keep the
form they were measured on, and the floors staleness machinery
confirms no cost input moved.  The staged change awaits Greg's
commit.  The C2 question stays open.

### 1.19 The width mechanism is the compiler's divisibility
### specialization (mg20, 2026-08-17)

Measured 2026-08-17 in two jobs on one H100 each: 15316533 (h007, the
timing leg) and 15316589 (the counter leg, after a path fix put
Nsight Compute on the batch environment's PATH).  The rows are
`plans/experiments/torch_port/rows/mg20_width_h007_20260817_092645.jsonl`
and the second job's sibling; the run detail is in
`mg20_width_mechanism.md` beside the script.

This is the mechanism probe behind the kernel-width puzzle: why a
parallel forward launch at width 504 costs twice per slice what a
launch at width 1008 costs.  The probe varied one thing per arm.
Alongside the four widths of the original discrimination, it ran a
two-piece arm at widths 512 and 496, both divisible by 16 and
together within 1.6 percent of 504's work; an arm at width 504 inside
a 1008-wide allocation; and two strided arms doing 512 columns of
work over row strides of 1008 and 2016, using the shipped kernel with
a truncated launch grid.

The reading is valid on the anchors.  The four original widths
reproduced the earlier per-slice ratios to within 0.01 (2.02, 2.05,
and 1.79 against 2.00, 2.04, and 1.81), every arm's values sat at
5e-7 to 7e-7 against the full-width reference, and the timing spreads
were below one percent.

The mechanism is the compiler's divisibility-by-16 specialization of
the kernel's width argument.  The evidence, each varied alone:

| arm | width argument | divisible by 16 | per-slice vs width 1008 |
|---|---|---|---|
| w1008 | 1008 | yes | 1.00 |
| w504 | 504 | no | 2.02 |
| w512_496 | 512 and 496 | yes | 0.91 and 1.00 |
| w504_alloc1008 | 504 | no | 2.02 |
| w512_stride1008 | 1008 | yes | 1.00 |
| w512_stride2016 | 2016 | yes | 1.01 |

A width the compiler can prove divisible by 16 runs at full
efficiency at any width tested, and a width it cannot runs at about
half, wherever the data lives and whatever the allocation.  The
contiguous arms launch the same total blocks across their pieces, so
machine filling by block count cannot explain the step.

The counters name the path the cost takes.  The unspecialized
compilation uses more registers, which caps occupancy at 5 blocks per
multiprocessor against the specialized 8, reading 60 percent against
90.  The L2 hit rate is HIGHER at the narrow widths (90.7 against
80.2 percent), so cache residency is innocent, and the earlier
candidates are settled: not launch overhead, not occupancy as a
block-count question, not memory layout, but register pressure and
access width from the missing specialization.

One correction to the earlier record rides along.  The
"per launch" figures in §1.9 are means over a reconstruction's 680
calls at four pixel counts, not costs of one full-pixel launch; the
probe reproduced that mixture from its own pixel ladder.  Every
per-slice ratio and every decision built on them is unchanged.

The remedy this implies is recorded, not implemented: a call site
that hands the kernel a width not divisible by 16 pays about two
times, and padding the width argument to the next multiple of 16 is
near free.  The production gather calls at the standard cells are
already divisible (1008 and 2016), so this is a robustness increment
for arbitrary user shapes, not a production speedup at today's cells.
Whether the back kernel carries the same property is unmeasured.

For the sorted-channel question the counters cut both ways, and they
do not decide it.  The atomic write path generates 192 times the
ideal L2 sector traffic at every width, which is the inefficiency
sorting would attack; but the DRAM write traffic is already near one
times the output slab, and the fast widths carry the same 192 without
penalty, so the counters do not show the write path binding time at
this kernel's operating point.  The question stays open, gated on the
2048-class attribution and on the cache-effects reading Greg
registered.

CORRECTION (2026-08-18, mg28, §1.26): the "192 times the ideal" was
a pricing error.  The formula omitted the launch's view axis, so the
ideal was 128 times too small; the corrected ratio, measured
directly, is 1.50.  The write path was never wasteful, and the
paragraph's conclusion that it does not bind survives strengthened.
The sorted-channel question is settled by §1.26: the kernel is
load-bound on the per-view values reload, not write-bound.

### 1.20 The first 2048-class reconstructions: the ledger holds, the
### batch knee brackets, and the cone back projection becomes the
### target (mg19, 2026-08-17)

Measured 2026-08-17, job 15314401 on h003, 3.3 hours on four H100s,
on the flipped 7cd32ed tree.  The rows are
`plans/experiments/torch_port/rows/mg19_baselines_h003_20260817_082830.jsonl`;
the run detail is in `mg19_two_k_baselines.md` beside the script.
These are the first composed reconstructions ever run at the
2048-class cell, cone and parallel, eighteen arms.

The reading is valid on the arm witnesses.  Every arm ran, every
values leg passed (4.0e-6 to 9.6e-6 against a 1e-4 gate), the Triton
kernels bound in both directions on every arm, and the repeat pairs
put the forward-busy spread at 0.13 percent.  Both two-device arms
were REFUSED by the memory preflight, which is the capacity table's
two-device verdict confirmed on hardware.

**The memory model validates at the 2048 class.**  Every
modeled-over-measured ratio sits inside the (1.00, 1.30) band and
none sits under the floor:

| arm | ratios per device | modeled GiB | measured GiB |
|---|---|---|---|
| cone n=3 | 1.182 to 1.183 | 66.7 | 56.5 |
| cone n=4 | 1.103 to 1.180 | 51.1 | 43.1 to 46.3 |
| parallel n=3 | 1.182 to 1.183 | 66.7 | 56.4 |
| parallel n=4 | 1.103 to 1.180 | 51.1 | 43.1 to 46.3 |

The capacity table's verdicts therefore hold where they matter: two
devices refuse, three devices fit and ran, four devices fit with
room.

**The component split, from the per-call device brackets.**  The
walls below are the warm repeats where one exists; a starred wall is
the first arm at its call shapes and carries compilation in its
"other" column.

| geometry | n | wall s | forward busy s | back busy s | other s | back share of wall |
|---|---|---|---|---|---|---|
| cone | 3 | 459* | 203 | 137 | 119* | 0.30 |
| cone | 4 | 420 | 151 | 228 | 40 | 0.54 |
| parallel | 3 | 299* | 193 | 32 | 75* | 0.11 |
| parallel | 4 | 216 | 142 | 36 | 38 | 0.17 |

Two attributions follow.  Parallel at production scale is
forward-dominated, at about two thirds of the wall.  Cone at four
devices is BACK-dominated, at more than half the wall, and the back
projection's busy time RISES from 137 s at three devices to 228 s at
four.  The two-device cone back anomaly of §6.2 is therefore not a
two-device curiosity: the cone back projection anti-scales with the
device count at the 2048 class, and it is now the largest single
component of a production cone reconstruction.  A mechanism
consistent with the numbers, recorded as a hypothesis: the banded
back walks one band per slice-owner, each band call pays the cone
kernel's full-detector-row grid, so total back grid work grows with
the count — the same cost structure the forward had before the
cylinder transfer replaced it.  The kernel campaign's first target is
the cone back projection, and the sorted-accumulation question (B3)
lands there, where the scatter is largest.

**The pixel-batch sweep brackets its knee.**  Read on forward busy
seconds, which carry almost no compilation:

| batch | cone fwd busy s | parallel fwd busy s |
|---|---|---|
| 8192 (shipped) | 151.6 / 151.4 | 142.1 / 142.1 |
| 16384 | 135.2 | 127.1 |
| 32768 | 127.9 | 120.5 |
| 65536 | 125.8 | 116.8 |

Doubling from 8192 buys 11 percent, the next doubling 5 percent, and
the last 2 to 3 percent, so the knee sits at or just above 32768.
The transferred-cylinder residents stay under 1.5 GiB at 65536
against 43-plus GiB peaks, so memory does not constrain the choice at
this scale.  A recommendation is recorded: move the default to 32768,
as a reviewed change that re-anchors the affected rows.  The 1024
class measured the same direction (mg11), so one default serves both.

**The combining-slab rider.**  Back-projection busy on the busiest
device: 227.9 and 228.2 s at the shipped 64 MiB (spread 0.31 s),
227.2 s at 16 MiB, 226.1 s at 256 MiB.  Both moves sit outside the
repeat spread, so the slab size is measurable, and its whole range is
0.8 percent of the back projection.  The open slab item can close on
this reading: 256 MiB is marginally best, the default is defensible,
and no structural work is warranted.

One value note for the record.  The parallel three-against-four
composed fingerprints read 2.9e-4 max-relative, while every
same-count repeat reads under 9e-7 and cone's cross-count pairs read
2.4e-6.  The three-device view split is uneven, and the compiled
torch code around the projectors carries the documented
reduction-order latitude there (§1.16), so the reading sits inside
the 1e-3 envelope that property set.

### 1.21 The cone back anomaly is the width mechanism on the band
### argument (mg21, mg21b, 2026-08-17)

Measured 2026-08-17 in three jobs: 15327847 (mg21, four H100s on
h017, 16 minutes), 15327968 (mg21b, one H100 on h007, 2 minutes),
and 15328160 (the mg21b addendum, one H100 on h006, 3 minutes), on
the merged 6d90601 tree.  The rows are
`plans/experiments/torch_port/rows/mg21_back_attrib_h017_20260817_192528.jsonl`
and `rows/mg21b_band_gpu_h007_20260817_194843.jsonl`; the run detail
is in `mg21_back_attrib.md` beside the scripts.

This is B1's attribution probe.  mg21 instrumented the sharded cone
back projection at six places at the 2048-class cell and split each
call's time into named parts: the two eager builders, the
channel-major copy, the Triton kernel, the accumulation between body
calls, the cross-device reduce, and whatever is left.  Both counts
ran on mg19's staged sinogram, with values witnessed between counts
at 1e-12 relative.

**The kernel itself is what grows, and nothing else moves.**  On the
busiest device, one full-pixel back projection reads:

| part | n=3 s | n=4 s | ratio |
|---|---|---|---|
| kernel | 24.3 | 45.8 | 1.88 |
| builders | 0.76 | 0.76 | 1.00 |
| copy and other residual | 0.07 | 0.08 | 1.01 |
| accumulation | 1.28 | 0.97 | 0.76 |
| reduce | 0.06 | 0.07 | 1.13 |
| whole pass | 26.5 | 47.7 | 1.80 |

The subset variants read the same shape at every granularity the
shipped schedule visits.  The recorded hypothesis is refuted twice
over: the launch grid is band-sized rather than detector-sized
(193023x11 at n=3, 193023x8 at n=4), and halving the band -- which
doubles every per-band-call cost -- moved the total by 12 percent at
n=3 and 2 percent at n=4.  Per unit of voxel work the n=4 kernel runs
2.5 times slower than the n=3 kernel.

**The mechanism is the compiler's divisibility specialization on the
band argument, the same mechanism §1.19 measured on the parallel
forward's width.**  At n=3 each band is 672 slices and 672 is
divisible by 16; at n=4 each band is 504, which is not.  mg21b varied
the band alone -- one device, one sinogram, one pixel set, eight
bands interleaved -- with the prediction recorded first.  The five
divisible bands share one rate (18,600 to 21,500 ns per view-slice)
and the three non-divisible bands another (46,200 to 50,700), ratio
2.44, with 496 and 512 fast on either side of 504.  The two runs
close arithmetically: mg21's work falls to 0.75x while its time
rises 1.88x, implying 2.51.  The addendum re-ran the sweep on a
second node, where every rate reproduced within 0.03 percent, and
moved the band START across the divisibility boundary at a fixed
band of 512: rates within 0.4 percent, so the start does not matter
and the band argument alone governs.

Three consequences follow.  First, the cone back projection's
anti-scaling is not structural: the banded walk, the combining step,
and the per-call costs are all healthy, and the whole anomaly is
which band lengths the device count happens to produce.  The
1024-class record now reads as the same effect: one and three
devices land on divisible bands (1008, 336) and two and four on
non-divisible ones (504, 252), which is exactly where §1.7 and §6.2
measured the back slow.  Second, the remedy is B2's padding
increment, not a restructure: pad the kernel's band argument to the
next multiple of 16 and discard the padded columns, since no equal
division of 504 is divisible by 16 and the existing band knob can
therefore never reach a fast band at the slow counts.  The projected effect at the
2048 class is the n=4 kernel falling from 45.8 s to about 19 s per
full-pixel pass, which restores monotone scaling (137 s of back busy
at n=3 against about 100 at n=4, in place of today's 228).  Third,
the back kernel's width sensitivity, which §1.19 left unmeasured, is
now measured, and it is the dominant effect at production scale.

The remedy design note carries the arithmetic, the gates, and the
decision: `plans/torch_port/active/back_remedy_design.md`.

### 1.22 The floors refresh on the merged tree: multiaxis measured,
### translation seeded, and one new anomaly (mg22, 2026-08-17)

Measured 2026-08-17, job 15327855 on h017, 3 hours 9 minutes, 50
timed arms and 18 generators on the merged 6d90601 tree.  The arm
records are under `results/mg22_floors/` on scratch (torch_p3); the
run detail is in `mg22_floors.md` beside the sbatch file.  The paste
landed the same evening: every row in
`mbirtorch/_widening_floors.py` now carries this run's provenance,
and the re-recorded hashes cleared the staleness note the 2026-08-17
merge had left tripping.

**The four existing projection floors did not move.**  Parallel and
cone re-measured at their 2026-08-16 floors with nearly the same
brackets, so neither the banded-forward removal, the 32768 pixel
batch, nor the 256 MiB slab moved a crossover.  The denoiser rows
stay sentinels (0.45x to 0.67x across the probes).

**Multiaxis measured its own thresholds, and the two-device reading
is not monotone.**  Two devices win at the 384-class (1.25x), lose
at the 512-class (0.35x; 11.4 s at one device against 32.6 s at two,
repeats tight, cross-count values at 3e-8), and win again at the
768-class (1.46x).  A single threshold cannot admit the 384-class
win without admitting the measured three-times regression above it,
so the pasted floor sits at the 768-class.  Four devices lose at
every probe against two (0.87x, 0.23x, 0.40x), so the n=4 row stays
a sentinel, now from the full warm-median protocol.  The 512-class
two-device slowdown and the 768-class four-device 0.23x are one
unexplained anomaly shape, recorded here as an open question: it is
not the cone back kernel's divisibility effect (multiaxis runs
compiled torch bodies, and the fast and slow cells do not sort by
band divisibility), and no probe has looked inside it yet.

**Translation's first rows are both sentinels, and they fix a live
mis-widening.**  On production-anchored cells up to the
(256, 1900, 3000) production scan, two devices lose at every size
(0.66x to 0.88x, rising with size) and four lose worse (0.15x to
0.64x).  Until this run TranslationModel took the parallel floors,
which admit two and four devices at those sizes -- an automatic
production-scale translation reconstruction widened into a measured
12 percent (n=2) or 56 percent (n=4) slowdown.  The class now
declares `_floor_family = 'translation'`, and the sentinels hold it
to one device until a refresh finds an admission size; the rising
two-device trend says one may sit just above the largest size
tested.



Measured 2026-08-10, job 15149052 on h014, on the verified 1a2deb0
tree.  The rows are
`plans/experiments/torch_port/rows/mg8_geom_calib_h014_20260810_142004.jsonl`,
and the correction they produced is staged in mbirtorch the same day.

The two geometries with no hand-written kernels, TranslationModel and
MultiAxisParallelModel, run their projections as general torch code,
which the ledger calls a torch body.  The ledger priced a torch body's
view batch at one nominal slab.  A single-body measurement during the
port review put the real transient at about ten times that slab, so
mg8 measured the composed reconstructions: two shapes per geometry at
one, two, and four devices, twelve arms, with the calibration mode
owning the peak counter.  Twenty-seven of twenty-eight device-readings
sat below the 1.00 floor, and zero rows were invalid.  The worst
readings were 0.264 on multiaxis 512 and 0.136 on the half-scale
translation shape.

The corrected charge is closed-form, derived jointly from the rows and
the bodies' source.  Every array in a torch body's interpolation loop
has the shape (view batch, pixels, width), where the width is the
detector rows or the call's slice band, whichever axis that body
sweeps.  The per-view charge is therefore a slab count times pixels
times the wider of the detector rows and the call's slice band, times
four bytes.  The slab count is 14, a measured multiplier set eight
percent above the tightest constraining reading; the rows cannot
resolve it into named arrays, and the ledger comment says so plainly.
A second premise fell with the first: a torch body's output plane is
not covered by any declared per-view cost, so torch bodies now pay
both live blocks where kernel bodies pay one.

Under the corrected charge all twenty-eight device-readings sit at or
above the floor.  The floor reading is 1.057, and the worst over-charge
is 5.74x.  Most of that ceiling is irreducible from shapes: two
two-device runs peaked about 2.1x apart on identical shards, and a
model that sees only shapes must cover the higher device.  The
torch-body band is recorded as (1.00, 5.80) beside the kernel path's
(1.00, 1.30), and the kernel path is numerically untouched.  The suite
verifies both statements, 488 passing with 19 new tests pinning the
measured arms.

Two consequences deserve their own sentences.  Sharding barely shrinks
a torch-body geometry's per-device peak, and the measurement agrees
with the corrected model: the half-scale translation shape's measured
peak GREW from 8.2 to 27.2 GB between one and four devices.  The
automatic count search will therefore widen less often for these
geometries, and the preflight will refuse outright where a doomed run
previously started and died inside the allocator; multiaxis 1024 now
models at 68 GB on one device, which sits at an H100's edge once the
margin is applied.  One cheap follow-up is recorded: the translation 1K
two-device arm needs 3.9 slabs where its four-device arm needs 12.6
from identical shapes, a run-to-run spread the rows do not explain, and
one repeat of the two-device arms is the cheapest way to shrink the
5.74x ceiling.

---

### 1.23 The band padding lands: the cliff closes and 2048-class back
### scaling turns monotone (mg23, mg24, 2026-08-18)

Measured 2026-08-18 in two jobs: 15336959 (mg23, the gate run, two
H100s on h004, 11 minutes) and 15337015 (mg24, the composed
confirmation, four H100s on h017, 36 minutes), on the merged tree
plus the padding implementation Greg committed the same morning as
mbirtorch 64dedb8.  The rows are
`rows/mg21b_band_gpu_h004_20260818_064628.jsonl`,
`rows/mg23_pband_gpu_h004_20260818_064809.jsonl`, and
`rows/mg24_padding_h017_20260818_065201.jsonl`; the run detail is in
`mg24_padding_confirm.md` beside the scripts.

This is §1.21's remedy, implemented and confirmed.  The kernel
wrappers round their width-class arguments up to the next multiple of
16 before the launch, allocate the output at the padded width, and
return the real-width slice; the memory ledger charges the padded
band through the same helper.  Cone and parallel take the pad in both
directions; translation and multiaxis have no hand-written kernel and
are unchanged.

**The gate run passed whole.**  The suite read 674 passed on GPUs,
including new value tests at 1e-5 against the torch bodies and
windowed references.  The cone band sweep that measured the 2.44x
divisibility cliff on the unpadded tree read 1.06x on the padded one:
bands 504, 344, and 252 fell from about 47,000 to about 20,400 ns per
view-slice.  The parallel back sweep read 1.07x, flat across the
boundary.  The residual over the divisible rate is the discarded
padded lanes, 1.6 to 3 percent by arithmetic.

**The 2048-class composed confirmation hit the projections.**  On
mg19's staged cell, three-iteration cone reconstructions read, on the
busiest device:

| arm | band | back busy s | mg19's reading | the §3 projection |
|---|---|---|---|---|
| n=3 | 672, no pad | 137.1 | 136.8 | unchanged |
| n=4 | 504 padded to 512 | 106.4 | 227.8 | near 100 |
| n=4 repeat | 504 padded to 512 | 104.2 | 228.2 | near 100 |

Back busy time is monotone in the device count again, and 104 over
137 is 0.76 against the ideal work ratio of 0.75.  The forward at
n=4 reproduced mg19's batch-32768 arm at 127.9 s, and every
calibration ratio sat inside the (1.00, 1.30) band with the padded
ledger charges, which validates the design note's ledger rule at
scale.

**The nightly on the padded tip closed B1 the same morning** (job
15338176, tip 64dedb8, suite 651 passed, results file
`regression_gpu-torch_20260818T120327Z_64dedb87.yaml`).  The per-call
back rows moved by the mechanism ratio everywhere a band was
non-divisible: the cone 1024-class two-device back fell from 5.78 s
to 2.37 s per call, which is 2.44x -- the measured cliff exactly --
and its speedup against one device went from 0.78x to 1.92x, so the
two-device anomaly that opened B1 is gone.  Cone at four devices
fell 2.33x (speedup 1.52x to 3.55x).  The parallel back moved the
same way: its 1024-class two-device call went from break-even
(0.997x) to 1.88x, which retroactively measures the parallel
sensitivity the design note could only bound.  The odd 513x449x385
single-device rows fell 2.4x, which is the robustness case measured.
The gate tripped on one hard row that is the padding's priced cost
appearing where designed: the parallel forward at the odd cell now
makes a zero-padded input copy (+182 MB, +21 percent at one device),
which exists only at non-divisible shapes and is charged by the
ledger; production cells take no copy.  Two warn-level forward time
rows at the 512 cell sit in the node-noise class the 2026-08-17
forced repeat established.

### 1.24 The counter run on the cone back kernel: the gathers are
### cache-absorbed, and registers cap occupancy (mg25, 2026-08-18)

Measured 2026-08-18, job 15342576 on h000, one H100, under three
minutes of wall, on the padded 64dedb8 tree.  The rows are
`plans/experiments/torch_port/rows/mg25_back_counters_h000_20260818_121603.jsonl`;
the run detail is in `mg25_back_counters.md` beside the script.

This is B3's named precondition (back_remedy_design.md §6): a Nsight
Compute counter run on the cone back kernel itself, at the divisible
rate the padding now guarantees, asking whether the kernel's gathers
leave single-call headroom worth a kernel-campaign increment.  mg20's
counters measured the parallel forward's atomic write path, and this
kernel has no atomics, so this run is the back kernel's own reading.

The reading is valid on its anchors.  Five bands were timed through
the production route on one device, the production bands 1008, 672,
512, 336, and 256, and every rate sat in the divisible class
(18,390 to 21,517 ns per view-slice against mg23's 20,400 anchor).
Every cross-band witness read exactly zero, and all six profiled
variants collected the full 21-metric set on the first attempt.

**The counters, one warm production-shaped launch per band.**  The
readings are nearly constant across the bands; the table shows the
band-672 variant beside its full-pixel-mask control:

| counter | 672, quarter mask | 672, full mask |
|---|---|---|
| achieved occupancy | 24.9 percent | 24.9 percent |
| occupancy limiter | registers, 4 blocks per SM | same |
| registers per thread | 121 | 121 |
| SM throughput, percent of peak | 71.6 | 71.6 |
| memory throughput, percent of peak | 63.6 | 62.5 |
| L2 hit rate | 81.2 percent | 91.6 percent |
| L1 load hit rate | 90.8 percent | 96.3 percent |
| L1 load sectors over the coalesced ideal | 4.16 | 4.09 |
| long-scoreboard stall per issue-active cycle | 0.48 | 0.44 |
| DRAM write over the output partial | 1.00 | 1.00 |
| atomic plus reduction sectors | 0 | 0 |

Three answers follow, one per question the run was chartered to ask.
First, the gathers are NOT transaction-bound at the divisible rate.
SM throughput sits above memory throughput at every variant, the
long-scoreboard stall is about half a cycle per issue-active cycle,
and the DRAM rate during the profiled launch is a few percent of the
device's bandwidth (54.4 GB over a 211.6 ms launch at the largest
band).  The gather's sector amplification is real, about 4x the
coalesced ideal, and the caches absorb it: nine of ten load sectors
hit in L1.  Second, the sector efficiency is therefore a cache story
rather than a bandwidth story.  The kernel re-reads its sinogram
batch tens of times from DRAM across a launch (20x to 91x by the
counter), and that traffic still costs almost nothing against the
device's bandwidth.  Third, the occupancy limiter is registers: 121
registers per thread cap residency at 4 blocks per multiprocessor,
which is the 24.9 percent achieved occupancy, and the kernel keeps
SM throughput at 71.6 percent anyway through per-thread parallelism.
The store path is already minimal, one write per output element
(DRAM write exactly 1.00x the partial), and the zero-witness measured
the no-atomics claim rather than assuming it.

**What this says for B3.**  Sorted or segmented accumulation attacks
scattered writes and wasted read sectors.  This kernel has neither:
its writes are one store per output element, and its read waste is
absorbed by the caches while DRAM sits nearly idle.  The counters
therefore show little for a sorting or reordering increment to win.
What headroom remains is bounded and lives elsewhere: SM throughput
at 71.6 percent of peak bounds any scheduling or occupancy remedy at
about 1.4x, and the named limiter is register pressure at the
current tile, which is a tile-and-warp retuning question rather than
a memory-ordering one.  The disposition of B3 on these readings was
Greg's ruling, and he ruled the same day: B3 is closed, with the
register-pressure retune recorded as the only measured headroom and
not scheduled, and with the standing note that a performance item
closed this way can rise again after further refactoring.

### 1.25 The floors refresh on the padded tree: one cone floor drops
### a class, and the multiaxis window appears (mg26, 2026-08-18)

Measured 2026-08-18, job 15342578 on h009, 2 hours 57 minutes on four
H100s, on the padded 64dedb8 tree.  The run detail is in
`mg26_floors.md` beside the sbatch; the refreshed rows are pasted
into `mbirtorch/_widening_floors.py` with current hashes and
checksum, and the floors test passes on them.

The movements match the band arithmetic stated before the run.  The
padding changes a projection's cost only where a count produces a
band not divisible by 16, so only specific cone and parallel cells
could move, and the sentinel families could not.  That is what the
run read.  Cone n=4 is the one floor that moved: its 768-class cell
ran a non-divisible band 168 before the padding and read 0.87x, and
padded it reads 1.14x, so the floor drops from the 1024-class to the
768-class.  Cone n=2's losing cell rose from 0.76x to 0.87x without
flipping, parallel's rows reproduced with a slightly widened n=4
margin (1.06x, still the table's thinnest admission), and the
multiaxis, translation, and denoiser sentinels reproduced within run
noise.

One reading is new rather than moved, and it worsens B6.  The
refresh measured multiaxis n=2 at the 1024-class for the first time,
and it LOSES at 0.80x -- above the 768-class cell that wins at
1.46x.  The two-device multiaxis win is therefore a WINDOW in
problem size, not a threshold, and a floor cannot express that: the
pasted floor admits two devices at the 1024-class into a measured 20
percent slowdown.  The row's note carries the warning, and the
coarsening proposal asks whether the row should hold the family to
one device until B6 finds the mechanism.

The refresh also evidenced the family-scoped mode the coarsening
proposal drafts.  `--bless` named the cost inputs that moved since
mg22 as `triton_cone.py` and `triton_parallel.py` alone, which is
the padding's exact footprint: a scoped refresh of cone and parallel
would have measured everything this run's rows moved, at about a
third of its 12 GPU-hours.

### 1.26 The parallel forward counter run: load-bound on the
### per-view values reload, and a correction to §1.19's write-path
### pricing (mg28, 2026-08-18)

Measured 2026-08-18, job 15344936 on h001, one H100, 78 seconds of
wall, on the padded 64dedb8 tree.  The rows are
`plans/experiments/torch_port/rows/mg28_pfwd_counters_h001_20260818_154504.jsonl`;
the run detail is in `mg28_pfwd_counters.md` beside the script.
This is B7's instrument, ordered by Greg the same day: mg25's
counter harness pointed at the parallel forward kernel at the
production width 1008, at the full pixel mask and at the 12,051-pixel
subset the production schedule calls most often.

The reading is valid on its anchor.  The timing leg reproduced
mg20's full-pixel launch at 1.01x (864.5 ms against the recorded
859.1), and both profiled variants collected the full 21-metric set
on the first attempt.  The two variants read the same intensities,
so the story holds at production granularity.

**The kernel is load-bound on the per-view reload of the values
block.**  Memory throughput reads 52 percent of speed of light
against SM's 16, with 38 warps stalled on a long scoreboard per
issue-active cycle -- against the cone back kernel's 0.5.  The load
path's traffic names the mechanism: DRAM read is 130 times the
values block's bytes per launch, which is the block re-fetched once
per view of the 128-view batch (the 3.1 GB block cannot stay
resident in a 50 MB L2 across concurrent programs), and the L1 load
hit rate is 16 percent against the back kernel's 91.  The kernel's
own docstring names this trade and the remedy shape: the values tile
does not depend on the view, so moving the view axis from the grid
into an in-program loop reads each tile once instead of once per
view.  The counters point at exactly that specialization.

**The atomic write path is nearly coalesced, and §1.19's "192x" was
a pricing error.**  The atomic-path sectors read exactly 1.50 times
the one-sector-per-eight-adds ideal at both variants.  mg20's
write-path table priced the ideal from taps x pixels x width,
omitting the launch's view axis; the issued count is view_batch
times larger, and 192 divided by the 128-view batch is the 1.50
measured here.  §1.19's conclusion that the write path was not
binding survives the correction and is strengthened by it: the
scatter the sorted-channel idea would reorder is already within 50
percent of coalesced, DRAM write is 1.0x to 1.05x the output slab,
and sectors per request read 3.95 against the coalesced 4.  What
does not survive is the "192 times the ideal" characterization; the
write path was never wasteful, only priced against the wrong
denominator.

Two consequences follow.  First, mbirjax's segmented-accumulation
design difference is NOT where the parallel gap lives on the torch
side: the write path has almost nothing to give.  The gap's measured
mechanism is the values reload on the load path, and the candidate
remedy is the view-loop respecialization of the kernel's grid, a
kernel-campaign increment with a measured target (memory throughput
at 52 percent driven by a 130x re-read that the loop removes by
construction).  Second, the occupancy story is the reverse of the
back kernel's: 89.5 percent achieved with 26 registers per thread,
so there is no register headroom to chase here, and the stall
profile says latency, not residency, is the cost.  The disposition
-- whether to open the increment -- is Greg's; nothing is
implemented or scheduled by this run.

### Further plain-language explanation:

### What this kernel does

A forward projection takes the volume's voxel values and adds each voxel's contribution into the sinogram, view by view. For the 1024-class problem, the input is a big table of voxel values — 771,240 pixel columns × 1008 slices, about 3.1 GB — and one "launch" of the kernel processes 128 views at once. The GPU does this by creating tens of thousands of small independent work units. Today, each work unit is assigned one small tile of the input **and one specific view**: it reads its tile of voxel values, computes the geometry weights for its view, and adds the results into that view's sinogram.

#### What the measurements say, in plain terms

**The arithmetic units are idle; the memory system is the busy resource.** A GPU has two separate capacities: how fast it can do arithmetic, and how fast it can move data. During this kernel, the arithmetic units run at 16% of what they could do, while the data-moving machinery runs at about half of its capacity. Compare the cone back kernel we profiled this morning: 72% arithmetic. This kernel is not limited by math.

**The threads spend their time waiting for data.** GPUs tolerate slow memory by keeping many thread groups ready, so while one group waits for its data, another computes. The profiler counts how many groups are parked waiting at any moment. In the back kernel that count was about 0.5 — essentially nobody waiting. Here it is 38. Nearly every thread group is standing in line for data almost all the time.

**The data they wait for is data the kernel already read — 129 times before.** This is the decisive number. One launch reads 377 GB from the GPU's main memory, but the input table is only 3.1 GB. The same table is being fetched about 130 times per launch — essentially **once per view**. The reason is cache size: the GPU has about 50 MB of fast on-chip cache, which cannot hold a 3.1 GB table. So by the time the work units for view 2 want the table, view 1's traffic has already pushed it out, and everything comes from main memory again. Each processor's small private cache confirms it: only 16% of requests find their data already nearby (the back kernel: 91%).

**The writes are fine, and the old record saying otherwise was an arithmetic mistake.** When many threads add into the same sinogram, they use a collision-safe add. Memory moves in fixed 32-byte packets, and a perfectly organized writer puts 8 useful numbers in every packet it touches. This kernel's writes use packets at two-thirds efficiency — 1.5 packets where a perfect writer would use 1. That is close to ideal. The August 17 record said the writes were "192× wasteful"; that number came from dividing the measured packet count by a baseline that forgot the launch does its work 128 times, once per view. 192 ÷ 128 = 1.5 — the writes were always nearly optimal. This matters because mbirjax's clever sorted-write design attacks exactly this write path, and our write path has almost nothing left to give. The gap is not there.

So the one-sentence diagnosis: **the kernel re-reads its 3-GB input from slow memory once per view, 128 times per launch, and its threads spend most of their time waiting on those repeated reads.**

#### The proposed remedy

The fix exploits a symmetry specific to parallel-beam geometry: **a voxel's value does not depend on the view.** (Its detector position changes per view; the value itself doesn't.) The kernel's author left a note in the code anticipating exactly this: whether to exploit it "depends on whether the kernel is atomic-bound or load-bound, which is a measurement, not an argument." mg28 is that measurement, and it came back load-bound.

The change: instead of creating a work unit per (tile, view) — which is what forces every view to re-read the table — create a work unit per tile only, and have each unit **loop over the 128 views internally**. The unit reads its tile of voxel values once, then for each view computes that view's weights and adds into that view's sinogram. The input is then read once per launch instead of once per view: the 377 GB of main-memory traffic becomes roughly 3 GB, plus the unchanged writes.

What this does *not* change: the arithmetic is identical, the write pattern and count are identical, and the results should be identical to float precision. What it risks: each work unit now does 128× more work, so there are 128× fewer of them — but the arithmetic says ~48,000 units remain, and the GPU only needs about a thousand to be fully occupied, so parallelism survives comfortably. The genuine design work is deciding where the tile lives during the view loop (in the unit's registers if the tile is made small enough, otherwise in the near cache, which is still an on-chip re-read rather than a main-memory one) — that's ordinary kernel tuning, guided by the same counters.

**What it's plausibly worth:** I won't promise a number — this is exactly the "measure, then specialize" the docstring ordered — but the arithmetic of the prize is simple. The forward is 29 s of the 40 s one-device wall, its dominant cost is the 130× redundant read traffic, and the remedy removes that redundancy by construction. If the forward merely halved, the wall would land near 26 s — which is mbirjax parity.

One closing observation that makes the picture cohere: **cone beam cannot use this trick, and that's consistent with cone already matching mbirjax.** In cone geometry the slice a detector row sees *does* change with the view, so the cone forward's per-view re-reading is inherent to the physics, and both libraries pay it. Parallel beam has an exploitable invariance the current kernel leaves on the table — and mg28 measured that table-stakes to be nearly the whole gap.

All of this is recorded in findings §1.26 with the plain-language mechanism, and B7 holds the increment awaiting your go/no-go.

### 1.27 The two view-loop spikes: the traffic moves, the atomic
### path stays, and 1.17x is this variant family's ceiling (mg29,
### mg30, 2026-08-18)

Measured 2026-08-18 in two jobs on one H100 each: 15345411 (mg29,
h002) and 15345519 (mg30, h001), each under two minutes, on the
padded 64dedb8 tree.  The rows are
`rows/mg29_pfwd_viewloop_h002_20260818_162605.jsonl` and
`rows/mg30_pfwd_viewchunk_h001_20260818_163404.jsonl`; the run detail
is in `mg29_pfwd_viewloop.md` beside the scripts.  This is B7's spike
step, ruled by Greg the same day.  Both spikes were valid on their
anchors: the shipped baseline reproduced mg20's 859 ms launch, and
all sixteen swept configurations passed values against the shipped
wrapper at the 6e-7 class, so every variant computes the projection.

mg29 measured the remedy §1.26 pointed at, and it bought far less
than the mechanism promised.  The whole-batch view loop collapsed the
input re-read as designed, from 130x the values block to 12x, and the
launch got only 1.10x faster on the production mixture.  The counters
name the reason: the traffic moved instead of vanishing.  With every
program walking all 128 views on its own schedule, the concurrently
active output planes stopped fitting L2, and the atomic adds thrashed
their planes to DRAM -- write traffic rose from the shipped 0.5 GB to
262 GB per launch, nearly the read traffic saved.

mg30 swept the middle the two ends bracket, a view CHUNK per program,
and found the interior better but bounded.  The speedup rises
monotonically with the chunk at both tiles and tops out at 1.17x on
the mixture (1.19x at the full mask) at chunk 32 on the shipped 8x128
tile.  The traffic account, full-mask launches:

| design | DRAM read GB | DRAM write GB | launch ms |
|---|---|---|---|
| shipped, one view per program | 377 | 0.5 | 864 |
| mg29, all 128 views per program | 38 | 262 | 757 |
| mg30 winner, chunk 32 | 19 | 49 | 728 |

One number is identical in all three rows and it is the finding: the
atomic-path sector count, 5.597e10 per launch, 1.79 TB of 32-byte
sectors through L2's atomic path in about three quarters of a second.
Cutting the DRAM traffic 5.5x moved the launch only 1.19x, the
memory-side throughput stayed near 60 percent of peak, and the stall
reading ROSE to 50.7 warps per issue-active cycle.  These readings
indicate the kernel's floor is the atomic path's own throughput, which
no loop reordering touches, because every design issues the same
taps x pixels x width x views atomic adds -- the volume is the
algorithm's, not the schedule's.

What the floor implies is a different increment, and its arithmetic
is recorded here for the design note to start from.  The way past an
atomic-volume floor is fewer atomics, not better-ordered traffic.
Within one (pixels, columns) tile, neighboring pixels project to
neighboring channels, so a P-pixel tile's taps land in a span of
about P plus two channels.  Accumulating the tile's span in shared
memory and issuing one atomic per (channel, column) of the span cuts
the adds by about 3P over P plus 2: 2.4x at the shipped 8-pixel tile,
rising toward 2.8x at 32.  That reduction matches the remaining gap
to mbirjax's segmented forward, and it is the in-kernel form of the
segmented accumulation whose torch-level form mg13 rejected.  It is a
new kernel interior rather than a knob, so it needs a design note and
a ruling before any code.

Nothing ships from the spikes.  The view-chunk variant at 1.17x is
values-safe and could take the library step through the composed
re-gate; whether to ship it, fold its loop into a segmented design,
or hold, is Greg's ruling on B7's next step.

### 1.28 The cone forward counters: a mixed profile, and the shared
### scatter's 1.50x constant confirmed a third time (mg31, 2026-08-18)

Measured 2026-08-18, job 15345826 on h007, one H100, 81 seconds, on
the padded 64dedb8 tree.  The rows are
`rows/mg31_pfwd_counters_h007_20260818_165713.jsonl`; the run detail
is in `mg31_cfwd_counters.md` beside the script.  Greg ordered the
run as the segmented design note's cone input, so the note commits
to cone on a measurement rather than an analogy.

The cone forward reads mixed, between its two siblings.  SM
throughput is 50.4 percent of peak against memory's 63.6, with the
stall at 9.0 warps per issue-active cycle -- against the parallel
forward's 15.6 over 52.3 with stall 38, and the cone back's 71.6
over 63.6 with stall 0.5.  The vertical tap loop gives this kernel
real arithmetic beside its memory traffic, and neither side is
cleanly binding.

Two components split the memory side.  The scatter is the shared
horizontal-fan block, and it reads exactly 1.50x the coalesced ideal
at the same per-view atomic volume as parallel's -- the third kernel
to measure that constant.  The gather is the larger stream: the
vertical fan re-reads the values block 60x per 52-view launch from
DRAM (175 GB), with the L1 load hit at 42 percent.  Segmented
accumulation therefore applies to cone unchanged, but it relieves
the smaller of the two streams; the design note bounds the cone
payoff near 1.3x and recommends cone ride the spike
opportunistically while parallel remains the primary target
(`pfwd_segmented_design.md` §4).  The gather-side re-read is
recorded as an observation for a possible later increment, outside
that note's scope.

### 1.29 The segmented spike: the contraction wins 1.68x at the full
### mask, and the subset calls belong to a tap path (mg32, 2026-08-18)

Measured 2026-08-18 in two submissions of one spike on h002 (jobs
15346037 and 15346085, 74 and 106 seconds), on the padded 64dedb8
tree.  The rows are
`rows/mg32_pfwd_segmented_h002_20260818_172015.jsonl`; the run detail
including the first submission's compiler lesson is in
`mg32_pfwd_segmented.md` beside the script.  This is the segmented
design note's increment 1, approved the same evening.

The mechanism worked exactly as designed, where its assumption
holds.  At the full pixel mask, the winning gated configuration (a
16-pixel tile with a 32-channel window, 128 columns, 32-view chunks)
reads 1.68x over the shipped kernel at 3.7e-6 values, and the
counters close the account: the atomic-path sectors fell 2.64x
(against the predicted 2.4x to 2.8x), the memory-wait stall fell
from 38.2 to 3.4 warps per issue-active cycle, and SM throughput
rose from 15.6 to 38.7 percent of peak.  The TF32 twin of the same
shape, run ungated to answer Greg's precision question, reads 1.84x
at 6.69e-4 worst values -- the tensor-core default buys about ten
percent more speed at three decades of accuracy, so the
full-precision mode is the right default and TF32 remains a
contract question, not a knob.

The other half of the verdict is the subset calls, and it reshapes
the library step.  Every subset ladder point read 0.87x to 0.88x,
so the plain production mixture breaks even (0.99x).  The mechanism
is channel locality: a subset call's pixels are spread across the
mask (the harness strides them, production draws them randomly), so
a 16-pixel tile spans far more channels than any window and every
subset tile takes the fallback.  The contraction's domain is
full-mask and locality-preserving calls; subset calls belong to a
tap path.  These results indicate the library form is a per-call
selection: the segmented kernel where the call's pixels are
channel-local, the existing kernels elsewhere -- the same
per-kernel selection discipline the library already applies by
platform.  A size-gated selection measured from this run's points
reads 1.11x on the mixture; pairing the segmented full-mask path
with mg30's chunk path on subsets projects higher and is the
library step's A/B to measure, not to assume.

Two items ride the record.  The 32-pixel tile is disqualified
(0.49x to 0.65x: the 64-channel window outgrows the win).  And the
fallback-rate counter under-counts by about the column-tile factor
(a uniform 12.5 percent reading at points whose spans guarantee
near-total fallback); the subset values passing at 3.7e-6 proves
the routing is sound, and the counter accounting is to be fixed
before the library step leans on rates.

### 1.30 The sorted contraction wins everywhere: 3.97x at the full
### mask, 2.8x to 3.9x at the subsets, and the atomic volume falls
### 31.6x (mg33, 2026-08-18)

Measured 2026-08-18, job 15346414 on h002, one H100, 74 seconds, on
the padded 64dedb8 tree.  The rows are
`rows/mg33_pfwd_sorted_h002_20260818_175404.jsonl`; the run detail is
in `mg33_pfwd_sorted.md` beside the script.  This is the design
note's increment 1b, the answer to Greg's channel-sorting question,
approved and run the same evening.

The sorted contraction removed the segmented kernel's one measured
weakness and then improved on its strength.  Per view, the call's
pixels are sorted by channel center outside the kernel, and the
values tile is gathered through the per-view permutation inside it.
Every configuration won at every ladder point at 3.3e-6 worst
values: the winner (a 32-pixel tile with a 16-channel window) reads
3.97x at the full mask and 2.77x to 3.85x across the subsets, with
the fallback rate at zero everywhere -- sorted tiles fit even the
16-channel window.  The combined selection across paths reads 3.45x
on the production call mixture, and the sorted path is the selected
path at every point.

The counters explain the size of the win.  Sorted 32-pixel tiles
collapse into a 2-to-3-channel span, so the contraction performs a
deep segmented reduction: the atomic-path sectors fell from the
shipped kernel's 5.597e10 to 1.773e9, which is 31.6x, an order
beyond the unsorted window's 2.64x.  The memory-wait stall fell
from 38.2 to 2.6 warps per issue-active cycle, and the memory
throughput at 81.5 percent of peak is now spent on useful traffic.

One pre-registered expectation was wrong, and the correction carries
the design insight.  The full-mask point was expected to lose,
because the per-view values gather looked like mg29's reload; it won
3.97x instead.  The sorted order at each view walks the dense mask
in stripes, so consecutive sorted pixels are geometric neighbors,
the row gather stays coherent, and DRAM read held at 11.2x the
values block -- the chunk path's class -- while the atomic collapse
dominated.  The locality partition the note drew between a sorted
subset path and an unsorted full-mask path may therefore be
unnecessary: one always-sorted kernel is the simpler candidate, and
the composed A/B decides between them.

The sort's own cost is small even unamortized: 14.6 ms at the full
mask against a 648 ms saving, under 4 ms at every subset, and the
shipped route pays it per call.  Amortization is available but not
taken: a reconstruction draws its partitions once at setup and only
reshuffles the subset visit order per pass, so each subset's
orderings are fixed for the run.  A plan-slot memoization must key
on the partition ENTRY, though, not the granularity: the default
granularity list carries four independent 128-subset partitions and
the default sequence cycles all four, so holding the fine level
costs four times any one-partition estimate (the earlier
few-hundred-megabyte figure rested on one partition per level and
is withdrawn), and a 3-iteration run repeats nothing at all -- its
sequence entries are distinct.  The reuse that pays is the long
tail: twenty-five visits per 128-entry at default length, doubled
per visit when positivity re-projects the clipped update.  A single
shared 128-partition per granularity (Greg's proposed restriction,
2026-08-18) would cut the cache four-fold -- free as an opt-in,
owing convergence and parity evidence as a default.  The
memoization stays the recorded follow-up, with its ledger charge.

The arithmetic this leaves for the library step, stated as
arithmetic: the parallel forward was 28.9 s of the 40 s one-device
1024-class wall, and 3.45x on the forward puts the wall near 20 s
against mbirjax's 25.8.  The composed re-gate turns that into a
measurement or refutes it; the 2048-class gather behavior and the
multi-device paths are its open cells.

### 1.31 The sorted forward passes its composed gate: the suite
### whole, and 1.43x to 1.89x at the 1024 class (mg34, 2026-08-18)

Measured 2026-08-18, job 15346797 on h006, 22 minutes on four H100s,
on the staged tree: 64dedb8 plus the sorted-contraction forward
change (triton_parallel.py and two test files, synced per file by
md5).  The rows are
`rows/mg34_sorted_ab_h006_20260818_183711.jsonl`; the harness is
mg27's protocol with the route switch as the A/B axis, reusing
mg27's staged sinograms.  This is the segmented design note's
increment 2, first half.

The gate passed whole.  The full GPU suite ran first and read rc 0,
with the sorted route's own value gates in it: the two kernels
against each other, the sparse-set fallback, and the view-chunk
tail.  All twelve composed arms then ran, every arm realized its
pinned count, and both value gates held at every (cell, count): each
arm against its own route's one-device arm, and the two routes
against each other, all inside 1e-3.

The composed A/B, warm medians, busiest-device peaks:

| cell | devices | per-tap s | sorted s | speedup | peak moved |
|---|---|---|---|---|---|
| 1024-class | 1 | 39.96 | 21.19 | 1.89x | none |
| 1024-class | 2 | 23.73 | 14.25 | 1.66x | none |
| 1024-class | 4 | 15.42 | 10.75 | 1.43x | +0.07 GB |
| 512-class | 1 | 1.84 | 1.31 | 1.41x | +0.17 GB |
| 512-class | 2 | 1.48 | 1.26 | 1.17x | +0.08 GB |
| 512-class | 4 | 2.12 | 2.05 | 1.03x | +0.08 GB |

Against the mbirjax column these numbers close the parallel gap B7
opened.  At the 1024 class the sorted route reads 21.19, 14.25, and
10.75 s against mbirjax's 25.80, 14.33, and 11.52: faster at one
device by 1.22x, at parity at two, faster at four.  Parallel now
matches or beats mbirjax at every measured count, as cone has since
the padding landed.  The composed one-device gain (1.89x) sits
where the spike arithmetic put it: 3.45x on a forward that is
seventy percent of the wall.

The 2048-class confirmation completed the gate the same evening
(mg35, job 15347172, after a first submission whose staging failure
is C5's fourth input): three devices read 239.10 s per-tap against
134.84 sorted (1.77x), four devices 188.98 against 113.25 (1.67x),
with the mode-pair value gate holding at both counts and the
busiest-device peaks identical between routes.  The 24 GB values
block was the sorted gather's one unmeasured territory, and it
carries the same win class as the standard cells.  Both halves of
the composed gate have now passed whole, so the staged change meets
the design note's shipping condition; what follows the commit is the
floors staleness note on triton_parallel.py and a re-anchor of the
comparison tables, whose mbirtorch columns measured the per-tap
route.  The run detail for both halves is in mg34_sorted_ab.md.

### 1.32 The cone grouped-forward spike loses: the windows fit, the
### reads never amortize (mg38, 2026-08-19)

Measured 2026-08-19, job 15361079 on h003, one H100, about 50
minutes, exit 0 with every instrument healthy.  The spike implements
design note §9 -- the pixel-batched cone forward with two-axis
grouping -- authored by an Opus agent from the section and the mg33
template, reviewed and submitted after an independent smoke.  Rows:
`rows/mg38_cone_grouped_h003_20260819_085812.jsonl`.

Every configuration loses at every ladder point.  The best arm (16x8
window, 8-view chunk, 8-slice tile) reads 1968.37 ms against the
shipped wrapper's 477.13 ms at the full-mask 64-view launch --
0.24x -- with the wide-window arms at 0.07x to 0.13x.  Subset
points fall back entirely and read 0.03x, which is §9's
disengagement rider measured rather than a defect.

The instructive part is what validated and what did not.  The
geometry all validated: the two-axis grouping put 99.8 percent of
full-mask tiles on the window path (mg37's prediction, realized),
the flush cut atomic sectors about 5x, the sort and gather cost
single-digit milliseconds, values held at 1e-5 everywhere through
the inverted arithmetic (after the author's one real catch: the
trapezoid must be inverted AT THE ROW, as the shipped kernel does,
or float32 cancellation costs 5e-5 and fails the gate with no bug
behind it), and the helical z-offset term rode intact.  What failed
is the single prize the design exists for: the counter pass on the
winner read DRAM at 73.7x the recon block per launch -- the shipped
kernel's once-per-view signature, not the once-per-chunk the design
predicted -- so the values tile never stayed resident across the
view chunk, and the window path's arithmetic tax (about 5x
per-voxel operations) was paid on top of unamortized reads.  The
compile-record introspection returned None for registers and
spills, so whether the tile spilled to local memory or was
rematerialized per view is undistinguished; that diagnostic is the
one bounded follow-up left open.  Disposition: recorded as a
measured negative (the run record is mg38_cone_grouped.md); §9
carries the status, and reopening is Greg's call.

### 1.33 The sorted order into the unchanged cone kernel also
### loses, and the rework line's arithmetic closes (mg39, 2026-08-19)

Measured 2026-08-19, job 15362636 on h003, one H100, exit 0, every
values gate inside 1e-5 (worst 3.1e-6).  Rework candidate A from §9
-- Greg's pick after the mg38 loss -- fed the two-axis sorted pixel
order to the UNCHANGED shipped cone forward: no new kernel, the
compound sort and permutes outside.  Rows:
`rows/mg39_cone_sorted_h003_20260819_094504.jsonl`.

Every arm loses at every ladder point.  The whole-batch compound
sort is the best of them at 0.87x on the full mask (546.96 ms
against 474.50, the sort itself 3.3 ms); per-chunk sorts read 0.83x
to 0.85x, the channel-only ablation 0.61x, and the subsets 0.29x to
0.90x.  The locality hypothesis is refuted at the timing level: the
kernel's raster order is already spatially local for its
output-anchored (pixel tile, row tile, view) grid, and mg31's own
counters said as much in advance -- L2 hit 87.2 percent and DRAM
read at about 452 GB/s, roughly 13 percent of an H100's bandwidth.
That last figure also evaluates candidate B without running it:
cache-blocking by slice band exists to cut DRAM re-reads, and DRAM
is measured at 13 percent utilization with the caches already
absorbing 87 percent of L2 traffic -- the hardware is already doing
what B would arrange.  The cone forward's cost is the vertical
gather's sector and latency machinery at the algorithm's own tap
count: the same kind of floor the parallel forward's atomic volume
was, but without a sorting-shaped exit, because the expensive
operation here is a read that must happen, not an add that can be
merged.  The recommendation recorded for Greg: decline B, close the
§9 rework line on three measured verdicts (mg38, mg39, and B's
arithmetic), and note that the cone forward already matches or
beats mbirjax at every measured cell, so nothing is owed
competitively.  Run detail: mg39_cone_sorted_order.md, including
two harness defects (a contaminated sort/kernel split in the
chunked arms' printout, and an ncu launch-skip that profiled the
availability self-check's tiny launch instead of the full-mask
call); neither gated anything.

### 1.34 The first scoped floors refresh: the coarse table lands,
### and the parallel n=2 floor rises a class on the sorted kernel
### (mg40, 2026-08-19)

Measured 2026-08-19, job 15369689 on h018, four H100s, 7.5 minutes
of measurement, exit 0.  The run detail is in `mg40_floors_scoped.md`
beside the sbatch; the refreshed parallel rows are pasted into
`mbirtorch/_widening_floors.py` with current hashes and checksum, and
the floors and device-policy tests pass on the pasted table (staged).

The run also carried E3's implementation, which rode this refresh as
ruled.  The coarse admission rule is now code: a floor needs a 1.15x
win (`ADMISSION_MARGIN`), a thinner win rounds up one class, and the
multiaxis n=2 row is a sentinel that holds the family to one device
until B6's mechanism is known.  The cost inputs became per-family
(`FAMILY_COST_INPUTS`), with the three gaps from the proposal closed:
`_utils.py` joined the shared set, each geometry's body file joined
its family, and the denoiser gained its own set.  The refresh tool
gained the `--families` mode with the carry-refusal rule.

The scope came from the hashes rather than from a person.  The
channel-sorted forward kernel (c761b24) changed `triton_parallel.py`
alone, that file prices only the parallel family, and the bare
`--families` flag resolved to exactly that family: 10 timed arms and
4 generators against mg26's 66 and 17.  mg26 ran 2 hours 57 minutes;
this run's measurement took 7.5 minutes, and the carried families'
rows passed through verbatim.

The parallel n=2 floor moved up one class, from the 512-class to the
768-class.  Against one device, two devices read 0.62x at the
384-class, 0.97x at the 512-class, and 1.38x at the 768-class.  The
512-class win was 1.20x on the per-tap kernel (mg26), so the sorted
kernel did not merely thin that win below the margin -- it removed
it.  The direction is the expected one: splitting's overheads did
not change, and the work they amortize shrank.

The parallel n=4 floor held at the 1024-class, now by measurement
rather than by rounding.  Against two devices, four read 0.80x at
the 768-class and 1.32x at the 1024-class.  mg26 read a thin 1.06x
win at the 768-class and the margin rule rounded the floor up; the
sorted kernel turned that same cell into an outright loss, so the
rounding anticipated exactly this kernel change one day before it
was measured.

One consequence for the table's shape: the n=2 rows no longer share.
Cone n=2 keeps its 512-class floor and parallel n=2 now sits at the
768-class, so the shared-row convention applies only where the
measured floors agree, which today is the n=4 pair.  Both rows' notes
name the split.

### 1.35 The ledger probe: the three-device over-read attributed,
### the lead-device transient refuted, and one under-read found
### (mg42a, 2026-08-19)

Measured 2026-08-19, jobs 15376256 and 15377054 on h004, four H100s,
about ten minutes of measurement across the two.  This is the
calibration design's probe (ledger_calibration_design.md §2), run
before any term changes.  Ten arms ran with the calibration mode on
and seven library seams wrapped by a reset-free watermark sampler, so
every arm carries modeled-versus-measured per device plus a per-region
attribution of where the peak rose.  The run detail, including the
probe's own one-device defect and its re-run, is in
mg42a_ledger_probe.md; the rows are the two mg42a jsonl files.

The three-device over-read reproduced and split by geometry.  Cone
reads 1.427 at three devices, and its dominant modeled phase is the
direct-recon back workers, where the back batch charge alone is
0.74 GB of a 1.33 GB phase.  That is the same charge the
back-attribution arms measured, so the first two calibration inputs
are one defect: the batch charge counts a live set the launch instant
does not hold.  Parallel reads 1.436 at three devices with a different
mechanism: its dominant phase is the initial forward projection, whose
top terms are the deliberate covers -- the doubled forward output, the
forward batch, and three resident cylinder batches -- which loom large
when shards are small.

The lead-device transient does not exist in a single reconstruction.
In a fresh process the four-device 1024-class arms peak device 0 at
6.84 GiB, against the nightly's 26.6 GiB reading, and the placement
trail shows shard-sized steps only.  So no ledger term is owed: the
ledger prices one reconstruction, and one reconstruction never holds
what the nightly read.  What produced the 26.6 remains UNEXPLAINED,
and the two easy mechanisms are now refuted.  A reference cycle
holding each call's end-state until garbage collection was the first
candidate; mg42b checked it locally on two virtual CPU devices with
automatic gc disabled, and across three back-to-back reconstructions
garbage collection freed zero tensor bytes -- the end-state frees at
refcount, so no library cycle fix is owed.  Trial-to-trial
accumulation in the nightly's own loop was the second candidate; its
time_op was read and is disciplined (the previous result is dropped
and gc runs before each timed call).  The accuracy remedy for the
nightly does not need the mechanism: reset the peak counters before
EACH trial and report the warm trial's peak, recording the per-trial
peaks so the column diagnoses itself -- if every trial reads high the
cost is real and visible, and if only the warmup does, it was setup.
Until that lands, the nightly's multi-device memory columns are read
as unreliable; its timings and value gates are unaffected.

One reading is new, and it leads the next increment: parallel at ONE
device reads 0.935, UNDER the band, reproduced in both runs (modeled
1.96 GB, measured 2.10 at the 512-class).  The model's peak phase is
the initial dot products; the measured watermark accumulates inside
the back worker region.  Cone at one device is in band at 1.104.  An
under-read is the direction the ledger exists to prevent, so the
term-change increment starts here, and its first discriminating step
is the same arm on the per-tap forward route, which says whether the
sorted kernel brought the gap or the model always had it.

### 1.36 The component split: the multiaxis and translation
### two-device losses are the back projection running uncompiled
### (mg44, 2026-08-19)

Measured 2026-08-19, job 15391547 on h004, two H100s, 66 minutes,
exit 0.  This is B6's first probe.  Ten arms ran in fresh processes:
multiaxis at the 512-, 768-, and 1024-class cells and translation at
the production cell, each at one and two devices, plus two unwrapped
control arms at the 512-class.  The protocol is the floors refresh's,
copied exactly.  The rows are
`rows/mg44_component_h004_20260819_210951.jsonl`; the run detail is
in `mg44_component_split.md` beside the script.

**The instrument read cleanly, and the anomaly reproduced exactly.**
Every arm's warm wall matched its recorded mg26 wall at 1.00x (worst
0.99x), so the run measured the recorded anomaly rather than an
approximation of it.  The wrapped and control arms agree within 0.4
percent, so the wrappers cost nothing visible.  Each region was timed
on two clocks: the host clock around the call, and CUDA event pairs
on each device's default stream, resolved only after each
reconstruction returned.  The two clocks separate enqueue cost from
device time, which is what names the mechanism below.

**The back projection carries the loss at every losing cell.**  The
table gives per-warm-reconstruction device milliseconds on the
busiest device, one device against two:

| cell | back n1 | back n2 | forward n1 | forward n2 | warm wall n1 -> n2 |
|---|---|---|---|---|---|
| multiaxis 512 | 3,014 | 16,523 | 4,084 | 2,051 | 11.4 -> 32.6 s |
| multiaxis 768 | 15,456 | 13,403 | 21,032 | 10,555 | 56.3 -> 38.5 s |
| multiaxis 1024 | 92,146 | 250,053 | 126,160 | 63,315 | 309.9 -> 388.3 s |
| translation prod | 879 | 4,908 | 2,545 | 1,346 | 12.5 -> 14.2 s |

The forward projection HALVES at two devices at every cell, which is
the scaling the split is supposed to buy.  The prior, the line-search
reductions, the update apply, the halo exchange, and the band reduce
all sit at or under a third of a second per reconstruction at every
cell.  At the losing cells the back projection alone explains the
wall gap.  At the 512-class it adds 13.5 s inside the iterations and
another 10.3 s in the setup phases; the Hessian diagonal and the
direct-recon initializer also back-project, so both setup phases pay
the same path.  The whole warm gap there is 21.2 s once the
forward's 2.0 s gain is netted out.

**The mechanism is torch.compile's per-frame recompile budget,
which the per-device compiled instances share.**  `maybe_compile`
binds one `torch.compile` wrapper per device (`instance_key=i`), a
split introduced to isolate triton launcher state.  Dynamo's variant
cache and its recompile limit (`config.recompile_limit`, default 8)
attach to the function's CODE OBJECT, and both wrappers share it.
The compiled variants guard on the input tensors' device index, so a
two-device run must hold separate variants for each device, and the
shape specializations multiply them.  When a frame's budget fills,
dynamo stops compiling that frame, and calls that match no existing
variant run eagerly from then on.  The job log carries torch's own
warning (`torch._dynamo hit config.recompile_limit (8)`, last reason
a device-index guard) for `_multiaxis_back_view_batch` at exactly the
three losing multiaxis arms and for `_translation_back_view_batch` at
the translation two-device arm.  No warning fires at any one-device
arm, and none fires at the 768-class, the one two-device cell that
wins.  The unwrapped control arm carries the same warning, so the
mechanism is the library's, not the instrument's.

**The warm loss is the eager end-state, not compile time.**  The
dynamo counters read zero compiles on every warm call of every arm;
all compile activity is in the discarded cold pass.  The host clock
confirms what eager execution looks like.  At the 512-class
two-device arm the back body's host time is 34.5 s per warm
reconstruction against 16.5 s of device time; the one-device
compiled back body spends 1.2 s of host time on the same work.
Host time above device time is the op-by-op dispatch signature, so
these readings say the two-device back runs uncompiled.  At the
1024-class the eager back's host and device times are both about
250 s, so the device work dominates there.  This size dependence is
why the loss is 0.80x at the 1024-class against 0.35x at the
512-class: eager kernels on large tensors do real work, and the
dispatch overhead amortizes.

**Which callers go eager depends on when the budget fills.**  The
budget is consumed in call order, so the trip point in one cell's
chronology decides which later callers run eager.  At the 512-class
the Hessian's back projection is also eager (1.4 s to 12.1 s); at the
1024-class the same phase improves (31.4 s to 26.4 s) because its
variants compiled before the budget filled.  Both two-device arms at
the 512- and 768-class hold about 36 unique graphs, so the aggregate
graph count does not separate the winning cell from the losing one;
the per-frame chronology does.  Enumerating the variants that fill
the budget takes one cheap run with `TORCH_LOGS=recompiles` at the
512- and 768-class two-device cells, recorded here as the
discriminator if a remedy needs it.

**Translation shares the mechanism, which closes B6's rider.**  Its
two-device back projection reads 4.9 s against 0.9 s at one device
while its forward halves, and its back body's frame carries the same
recompile-limit warning.

**The mg26 refresh's own log corroborates, unread until now.**  That
log holds 33 recompile-limit warnings.  They name the multiaxis and
translation back and forward bodies, the qGGMRF prior body, and the
update-apply body, and they cluster in that run's multiaxis and
translation arm blocks.  The deepest recorded loss (multiaxis n=4 at
the 768-class, 0.23x) sits in a block where four frames tripped.
These warnings were present at the original anomaly measurements, so
the mechanism is not new to today's tree.

Three consequences are recorded, none scheduled.  First, remedy
candidates: raise the per-frame recompile limit for the compiled
bodies, give each device instance a frame of its own, or pre-mark the
varying dimensions dynamic; any of them gates cheaply at the
512-class two-device cell, whose warm loss predicts about a 3x
recovery.  Second, the multiaxis floors sentinel's stated condition
("until B6's mechanism is known") is now met; the row should stay a
sentinel until a remedy lands and a family-scoped refresh re-measures
the window.  Third, the n=4 readings (0.23x to 0.87x) were measured
under the same mechanism with four device indices competing for the
same budgets, so they are due a re-measure after any remedy, not a
separate investigation.

### 1.37 The recompile-budget remedy lands in two forms, and both
### losing gate cells flip to wins (mg45, mg46, 2026-08-19)

Measured 2026-08-19 in three short jobs on two H100s: the first gate
(15394465, seven minutes), the assignment tracer (mg46, 15394591, 53
seconds), and the passing gate (15394667, six minutes).  The remedy
raises torch's per-function recompile budget from 8 to 64 in
`mbirtorch/projectors.py` (`_RECOMPILE_LIMIT_FLOOR`, with the
arithmetic in its comment; `MBIRTORCH_RECOMPILE_LIMIT` overrides the
floor verbatim).  Each gate re-ran the component split's cheap arms
on its tree: multiaxis at the 512-class and translation at the
production cell, one and two devices each, the floors protocol.  The
local suite passes on the change (599 passed).

**The first form failed its gate, and the failure measured a torch
behavior worth recording on its own.**  The first form raised the
budget once, where the compiled wrapper is created, on the creating
thread.  Its gate read every wall unchanged, and torch's warning
still printed the default limit.  The reason is that dynamo consults
a PER-THREAD view of this config: an assignment made on one thread
does not reach another.  This was measured three ways.  Locally, a
limit assigned on the main thread capped nothing when the compiled
function was called from a worker thread, and the same assignment
made on the worker thread capped it.  On the cluster, a tracer on the
config module's setter (mg46) showed the library's raises firing with
no assignment ever writing the default back, while the conversion
warning still read the default: the converting pool thread never saw
the main thread's writes.  The same behavior explains why one device
never tripped the budget: at one device the per-device fan-out
short-circuits onto the main thread, where the raise was visible.

**The second form raises the budget on the compiling thread, and the
gate passes.**  `maybe_compile`'s wrapper now calls the raise on each
first sight of an input shape, on the calling thread, under the
compile lock, before any call that can compile.  The passing gate's
log carries no recompile-limit warning, and both cells flip:

| cell | n1 warm | n2 warm before | n2 warm after | two-device ratio |
|---|---|---|---|---|
| multiaxis 512 | 11.40 s | 32.6 s | 7.47 s | 0.35x to 1.53x |
| translation prod | 12.57 s | 14.2 s | 10.04 s | 0.89x to 1.25x |

The component split says the mechanism is gone rather than thinner.
The 512-class back projection reads 2.34 s at two devices against
3.01 s at one, where the eager form read 16.5 s.  The Hessian phase
reads 1.14 s against its eager 12.1 s.  The one-device walls are
unchanged, which is the mechanism's own prediction.  The gate rows
are `rows/mg44_component_h012_20260819_230041.jsonl` (the failed
first form) and `rows/mg44_component_h007_20260819_232038.jsonl`
(the passing second form).

**What this opens, pending re-measures.**  Both cells that held
their families to one device now win at two, so the shipped sentinel
rows are stale in the conservative direction: automatic multiaxis and
translation reconstructions stay on one device and forgo measured
wins.  Three re-measures come before any floors change: the
1024-class pair (about 50 minutes on two GPUs), a 768-class check
that the raised budget cost that winning cell nothing, and the
family-scoped floors refresh for multiaxis and translation, which
also re-measures the n=4 sentinels (four pool threads paid the same
per-thread mechanism).  The follow-on paths beyond this remedy are
in `multigpu_plan_part_2.md`.

**The large-cell confirmation ran the same night, and the whole
window is now wins (mg47, job 15398646, 2026-08-20).**  The
1024-class two-device warm wall reads 203.6 s against the recorded
388.8, with one device unchanged at 308.6 s, so the recorded 0.80x
loss is a 1.52x win.  Its back projection reads 65.7 s at two
devices against 91.6 s at one, where the eager form read 250 s.  The
768-class reproduces at 1.45x against the recorded 1.46x, so the
raised budget cost the winning cell nothing.  With the gate cells,
the multiaxis two-device column across the measured ladder is now
1.53x, 1.45x, 1.52x, and translation reads 1.25x at production.  No
recompile-limit warning appears in the run's log.  The rows are
`rows/mg44_component_h001_20260820_072619.jsonl`.  One scope note:
the remedy changed a shared floors cost input, so the refresh that
follows is the FULL refresh rather than the family-scoped one the
previous entry anticipated -- the tool itself refuses to carry any
family whose cost inputs moved, and the cone and parallel closures
are compiled bodies the remedy also touches.

### 1.38 The full floors refresh on the remedied tree: all four
### torch-body sentinels clear, and the kernel families reproduce
### (mg48, 2026-08-20)

Measured 2026-08-20, job 15399595 on h007, four H100s, 1 hour 34
minutes, exit 0 -- against 2 hours 57 minutes for the same full plan
before the remedy.  The run detail and the verbatim paste block are
in `mg48_floors.md` beside the sbatch; the arm records are under
`results/mg48_floors/` on scratch.  Nothing is pasted into the
library by this run; the rulings are Greg's.

**The remedy holds everywhere.**  The log carries ZERO
recompile-limit warnings across every family, cell, and device
count, where the pre-remedy refresh's log carried 33.

**All four torch-body sentinels clear.**  Multiaxis n=2 wins at
every probed cell (1.525x, 1.461x, 1.515x), so its proposed floor is
the 512-class.  Multiaxis n=4, measured against one device, wins
everywhere by more than two (2.027x, 2.185x, 2.167x); its proposed
floor is also the 512-class, and the family is monotone in device
count at every measured size (the 1024-class walls read 308 s, 204 s,
142 s across one, two, and four devices).  Translation n=2 clears at
its middle production-anchored cell (1.192x, and 1.264x at
production); translation n=4 clears at the production cell (1.433x).
The old n=4 readings for these families, 0.23x to 0.87x, were the
per-thread recompile mechanism at four pool threads; with the remedy
they are the table's largest wins.

**The kernel families and the denoiser reproduce.**  Cone n=2 reads
0.872x, 1.305x, 1.606x (floor unchanged at the 512-class); cone n=4
reads 1.121x at the 768-class, still under the 1.15x margin, so that
floor stays at the 1024-class.  Parallel n=2 and n=4 reproduce their
floors at the 768- and 1024-class.  The denoiser stays a sentinel at
both counts.  These reproductions say the remedy moved nothing it
was not supposed to move.

**Two independent harnesses agree.**  The refresh's ratios match the
component harness's readings within noise: 1.525 against 1.53 at the
multiaxis 512-class n=2, 1.515 against 1.52 at the 1024-class, 1.264
against 1.25 at translation's production cell.

One coverage note rides the proposed multiaxis floors: the sentinel
probes are the ladder's top three cells, so the 384-class was not
probed, and the pre-anomaly record (mg22) read a 1.25x two-device
win there.  The proposed 512-class floor forgoes that win, bounded
by the cell's 3.9 s one-device wall; a ladder extension can revisit.

**The ruling and the landing (2026-08-20).**  Greg accepted the full
proposed table the same day.  The paste is in
mbirtorch/_widening_floors.py with hand-written notes, the blessed
hashes and checksum from the tool, and the module docstring's
sentinel narrative updated.  Two behavior tests pinned the old
sentinel rulings and were re-pinned to the new floors: the multiaxis
admission test now asserts admission at the 512-class and refusal
below it at every count, and the translation direct-recon test now
expects the finite-floor refusal wording for its small sinogram.
The floors, device-policy, and full suites pass (599).  This closes
the item the component split opened; what the remedy did not change
-- the kernel families' floors and the denoiser sentinels -- is the
table's own evidence that the change was scoped as intended.

### 1.39 The denoiser: no admission size exists, and the multi-device
### cost is the output gather rather than the computation
### (mg49, mg50, 2026-08-20)

Measured 2026-08-20 in two jobs on four H100s: mg49 (job 15402884 on
h005, 24 minutes) and mg50 (job 15403256, 7 minutes).  Both ran the
floors protocol exactly, so their walls compare with the recorded
denoiser rows.  The run detail is in `mg49_denoiser_split.md` and
`mg50_denoiser_gather.md`; the rows are the two jsonl files under
`rows/`.

**No admission size exists, and the sentinel rows are correct across
their whole domain.**  The denoiser ladder ran at four cells, each at
one, two and four devices.  Against one device, two devices read
0.640x, 0.654x, 0.643x and 0.657x at the 1024-, 1280-, 1536- and
1664-class; four devices read 0.594x to 0.620x.  The ratio is FLAT
across a 4.4 times range in volume.  The earlier rise across the
512-, 768- and 1024-class cells was a small-size effect that
plateaus, not a trend approaching admission.  The denoiser's own
memory ledger, read from the tree under test, prices one device at
13.7 GB at the 1024-class rising to 59.5 GB at the 1664-class, and
at the 1792-class the demand with the preflight's margin is 85.6 GB
against a 79 GB device.  Capacity therefore takes the widening
decision at the 1792-class, so the ladder covers every size at which
the speed sentinel has any effect.  The question is closed rather
than moved upward.

**The sharded sweep is a twentieth of the call.**  The component
split wrapped the denoise call, its setup phases, the sharded sweep
and the components inside that sweep.  At the 1024-class on two
devices the sweep reads 175 ms of a 3,401 ms call, and at four
devices 190 ms.  Setup reads 1,165 ms, of which the host-side
automatic regularization is about 650 ms and the image placement
about 500 ms.  That accounting left 2,060 ms, sixty percent of the
call, in no wrapped region.  The seam list had no entry for the
output gather, which is the defect this run's own numbers exposed.

**The gather is the whole multi-device penalty, and it is the
largest single cost of a denoise.**  mg50 varied one thing, whether
`denoise` was asked for the host form or the device form, and
measured both:

| cell | devices | host form | device form | the gather |
|---|---|---|---|---|
| 1024 | 1 | 2.164 s | 1.223 s | 0.941 s |
| 1024 | 2 | 3.307 s | 1.299 s | 2.009 s |
| 1024 | 4 | 3.633 s | 1.344 s | 2.289 s |
| 1664 | 1 | 8.513 s | 4.283 s | 4.230 s |
| 1664 | 2 | 12.997 s | 4.648 s | 8.348 s |
| 1664 | 4 | 13.863 s | 4.247 s | 9.617 s |

Two readings follow, and they are the finding.  The gather is 43 to
69 percent of a denoise call, at every count including one device.
And the gather roughly doubles when the volume is sharded, from
0.941 s to 2.009 s at the 1024-class and from 4.230 s to 8.348 s at
the 1664-class, while the computation barely moves.  At the
1024-class the whole one-to-two-device penalty is 1.143 s, of which
the gather accounts for 1.068 s, or 93 percent.  At the 1664-class on
four devices the computation is slightly FASTER than on one (4.247 s
against 4.283 s) and the entire measured penalty is the gather.

**In the device form the count penalty nearly disappears.**  Against
one device, the device-form ratios read 0.942x and 0.910x at the
1024-class and 0.921x and 1.009x at the 1664-class.  So a denoise
that is left where it was computed costs about the same on four
devices as on one, and at the largest measured cell it is at parity.
The checksums agree across every arm to 1e-7, so all six arms of a
cell reconstructed the same volume.

The mechanism is in `Shards.gather`, which brings every shard to the
host and then concatenates them on the sharded axis.  For a
recon-like array that axis is the LAST one, so the concatenate reads
and writes with the least favorable memory locality and allocates a
second full-size host array.  A single-device gather is one
contiguous copy with no concatenate, which is why sharding doubles
the cost.

Three consequences are recorded, none scheduled.  First, the two
sentinel rows are right for what they govern: the automatic choice
applies to the default call, and that call gathers.  They are
MISLEADING for a caller that passes `output_sharded=True`, whose
measured penalty is a few percent rather than the rows' 0.35x to
0.67x; a plug-and-play or ADMM loop is exactly that caller, and an
explicit `configure_devices` is how it should widen the denoiser.
Second, the gather is a real optimization target, and the candidate
is closed-form: allocate the host array once and copy each shard
into its own slice, rather than concatenating per-shard arrays.
Third, the same gather ends every multi-device reconstruction, where
the same two seconds sits against a wall of minutes and is
negligible; the cost is per volume, so it dominates cheap operations
and disappears into expensive ones.

### 1.40 The first two nights on the fixed writer: the inflated
### memory readings were the recompile budget, not the instrument
### (2026-08-21, read from the committed run files)

Read 2026-08-21 from the two nightly run files committed since the
per-trial memory fix landed (mbirjax_metrics be56fd0, pushed
2026-08-19).  The night of 2026-08-20 measured mbirtorch 8959e324
and the night of 2026-08-21 measured b6e5699, both on four H100s
with one warmup and one warm trial at the 1024-class.  The files are
`regression_gpu-torch_20260819T212248Z_8959e324.yaml` and
`regression_gpu-torch_20260820T191915Z_b6e56997.yaml` under
`results/gpu-torch/greg_dev/`.  Both nights gated PASS with no hard
or soft trips.

**The old inflation was never a warmup effect.**  On both nights,
every row's warmup peak equals its warm-trial peak, including the
inflated rows.  The first night reproduced the old readings exactly:
the 1024-class VCD arm read 26,612 MB at one device and 13,370 MB at
two, in both geometries, in the warm trial as much as in the warmup.
So the readings were real memory in every iteration.  G4 asked
whether the inflation was warmup-only or recurring, and the answer
is recurring.

**The second night removed the inflation, and the lineage names the
mechanism.**  On the night of 2026-08-21 the parallel rows dropped
to 23,418 MB at one device and 12,055 MB at two, and the cone rows
to 23,503 and 12,767.  The four-device rows did not move.  These
dropped values match the 2026-08-18 two-GPU interactive run
(c761b244) within 2 MB at all four cells.  The only relevant change
between the two measured tips is the recompile-budget remedy:
8959e324 predates it, and b6e5699 contains it (mbirtorch commits
c012379 and 214ed15; findings §1.37).  These readings indicate the
same mechanism B6 found for speed: the nightly runs all its arms in
one process, the
compiled-variant cap filled as the arms accumulated, and the
late-running 1024-class one- and two-device arms fell back to eager
execution.  Eager execution materializes every intermediate, which
is the extra 0.6 to 3.2 GB.  The four-device arms run before the cap
fills, so they never inflated.  The two-GPU interactive run had no
four-device arms ahead of the 1024-class, which is why it measured
the clean values on the same pre-remedy code.

**Two recorded framings are corrected.**  The plan entry said a
four-device arm recorded the 26.6 GiB watermark.  The run files show
the reading was always the ONE-device 1024-class row on a four-GPU
night, and no four-device row ever read it.  The ledger
calibration input said the four-device arm transiently pushes the
lead device 3.1 GB above the one-device arm's own peak.  No such
cross-arm transient exists.  The one-device arm itself ran eager,
and its own peak was 3.2 GB higher than its compiled form.  The
ledger conclusions of §1.35 stand: they were drawn from fresh
compiled processes, which the probe measured correctly.

**The record book reacted as G4 predicted.**  The second night set
new minimum memory records at the affected rows, so the baselines
ratcheted down.  A future eager fallback at these cells now trips
the HARD memory gate instead of hiding.  The per-trial lists did
their diagnostic job on their first exercise, and G4 closes with
this read.

One live flag rides the second night.  Its automatic-choice check
passed, with the settle choosing two devices of four as the floors
admit, but the check carries a floors staleness note.  The note
fired because b8b2d61 touched `_sharding.py` and b6e5699 touched
`denoising.py`, both of which price floor rows.  The plan already
records the reasoning for ruling
those changes cost-neutral: neither alters what a host-array timing
run executes (multigpu_plan_part_2.md, the note for the next floors
refresh).  The note persists on every future night until someone
either re-measures with `dev_scripts/refresh_widening_floors.py` or
blesses the hashes on that reasoning.

### 1.41 The divided-form sweep of the public API: one entry gained
### real support, fourteen gained refusals (2026-08-21, verified on
### the tree committed as 42574f8)

The original survey for divided-form support grepped docstrings for
"shard", which misses entries whose handling is implicit in the
code.  This sweep enumerated every public method and module function
that takes an array-like data input and handed each one a real
two-shard CPU `Shards`.  Each failure was traced to its deciding
line rather than judged from the entry alone.

**Most of the surface already handles the form.**  Over thirty
entries carry the divided form end to end.  These include the
projectors, `recon`, `prox_map`, the direct reconstructions in all
four geometries, the denoiser, and the HDF5 writers.  Three entries
refused the form cleanly before the sweep: `recon_split_sino`,
`recon_plastic_metal`, and `stitch_arrays`.  The preprocessing
entries already refused it through their shared check.

**Fifteen entries broke below the entry with misleading errors.**
The failures were a missing attribute, an unsupported operand, or a
numpy error raised long after `np.asarray` had silently built a 0-d
object array.  The worst diagnosis came from `recon` and `prox_map`,
which told the caller their divided initializer had shape `()`,
because `np.shape` of a `Shards` is `()`.  The fifteen were
`get_voxels_at_indices`, `auto_set_sigma_y`,
`get_forward_model_loss`, `get_forward_lin_quad`, `reshape_recon`,
`recon` and `prox_map` on `init_recon`, `recon_simple_parallel`,
`recon_simple_cone`, `gen_weights_mar`, `median_filter3d`,
`qggmrf_loss`, `qggmrf_gradient_and_hessian_at_indices`, and the two
differentiable projector wrappers.

**One entry gained real support.**  `get_voxels_at_indices` now
indexes each shard on its own device and returns a container over
the same placement.  The flattening touches only the row and column
axes, and the selection touches only rows, so the shard axis is
never crossed.  The gathered result therefore equals the whole-array
result exactly, and the gate asserts bitwise equality on an uneven
4+3 slice split.

**The other fourteen gained the entry refusal.**  The shared check
moved from the preprocessing pipeline into
`_sharding.reject_shards`, and the pipeline aliases it, so one copy
owns the wording.  Every refusal names the function, the offending
argument, and `shards.gather()` as the fix.  One refusal sits in the
innermost compiled kernel, `qggmrf_gradient_and_hessian_at_indices`,
so its cost was checked with `torch._dynamo.explain`: one graph and
zero graph breaks.  The `isinstance` folds at trace time.

**Review added one refusal the sweep missed.**  Cone's
`recon_split_sino` refused a divided sinogram and weights but let a
divided `init_recon` through to host slicing.  The refusal now
covers all three arguments.

**Candidates for real support later, each blocked by a design
question.**  `recon(init_recon=...)` needs the whole-volume shape
check reconstructed across shards, and the optimal scaling of the
initializer raises a lifetime question for a caller-owned divided
array.  `reshape_recon` targets the whole volume's slice count,
which no shard has.  `get_forward_model_loss` and
`get_forward_lin_quad` reduce to scalars, which needs a cross-device
reduction and a choice of where the scalar lands; the VCD loop
already solves this per shard, with partials combined by the caller.
`median_filter3d` needs a halo exchange, the same machinery the
qGGMRF kernel takes as explicit halo arguments.  The one-call
reconstruction functions read the geometry off the sinogram's shape
and would also need to reconcile the incoming placement with the new
model's own device choice.  `auto_set_sigma_y` points its refusal at
`subsample_views`, which already provides the reduction a caller
needs.

### 1.42 What the compiled multiaxis bodies are bound on: the
### device at the 512-class, per-batch host work at the 1024-class,
### and gigabytes of intermediate traffic at both (2026-08-21, mg51,
### job 15424602)

The kernel decision for multiaxis needed its first direct
measurement: what one compiled projection call actually runs, and
what limits it.  One H100 ran the production projection funnels at
the two floors cells under the floors protocol, with a plain-clock
timing leg, a torch-profiler leg, and Nsight Compute on the top
generated kernels.  The instrument was healthy throughout, and the
run detail is in mg51_multiaxis_counters.md.

**The limit changes with size.**  At the 512-class the host issues a
forward call in 10 ms and the device runs it for 1.16 s, so the
device is the limit.  At the 1024-class the host takes 34.95 s to
issue the same call against a 38.05 s device span, and the back
direction reads 19.35 s against 23.64 s.  The host is within 8 to 18
percent of pacing both directions at the production-class size.
This sharpens §1.36's reconstruction-level reading to the single
call.

**The host cost is not the launch API.**  A forward call at the
1024-class is 12,288 kernel launches, but the runtime recorded one
launch call per view batch, 6.4 ms in total.  The remaining host
time is per-batch work outside launching and outside the
synchronize: about 35 ms per view batch at the 1024-class against
0.35 ms at the 512-class, a hundredfold growth where per-batch
device time grew 1.8x.  These results indicate the host cost scales
with the cell, and naming its mechanism needs the profiler's
host-op table, which this run did not export.  That is the one
follow-up this run leaves.

**A call is a dozen kernels per view batch, none dominant.**  The
forward runs 12 distinct kernels per batch and the back 4 to 6.
The top three kernels carry 58 to 59 percent of forward device time
and 85 to 98 percent of back.  There is no single-kernel tuning
target; a change would replace the body.

**The counters price the intermediates.**  The whole problem's
arrays at the 512-class total about 0.65 GB, yet single launches of
the top kernels move 7 to 9.7 GB of DRAM traffic.  The back's top
kernels sit at the memory ceiling (88 and 94 percent of memory
throughput, one with an 11 percent L2 hit rate).  The forward's top
kernels run near the arithmetic ceiling instead (83 to 85 percent
SM throughput at 96 percent occupancy) while still moving those
gigabytes.  Peak device allocation during one projection call was
11.4 GB at the 512-class and 25.1 GB at the 1024-class.  These
readings are the slab traffic the memory ledger charges the torch
bodies, measured at the kernel level.

**What this says for a hand-written kernel, stated as evidence.**  A
kernel body would collapse each batch's dozen launches and its
per-batch host work into one launch, which is the only remedy the
1024-class host pacing has.  It would not materialize the slabs,
which is what the memory-ceiling back kernels and the
gigabytes-per-launch forward traffic both price.  The counter-case
is the 512-class forward: near the SM ceiling with the host quiet,
so at sizes where the host keeps up, the compiled forward leaves no
large single-kernel factor visible beyond the byte traffic.  The
cross-framework anchor (mg52) supplies the remaining decision
column.

### 1.43 The cross-framework anchor: mbirtorch meets or beats
### mbirjax at every torch-body cell, and the lead grows with size
### (2026-08-22, mg52, job 15428371)

No timing comparison against mbirjax existed for the two geometries
without hand-written kernels, so the kernel question leaned on
indirect evidence.  This run reconstructed identical staged
sinograms with both libraries on one H100 at the four recorded
cells, under the floors protocol, with md5-verified inputs and
matched model construction asserted in every arm.  The run detail
is in mg52_framework_anchor.md.

**The result, warm medians of seeded 3-iteration reconstructions:**

| cell | geometry | mbirtorch | mbirjax | jax/torch |
|---|---|---|---|---|
| (256, 1900, 3000) | translation | 12.59 s | 16.26 s | 1.29x |
| (512, 448, 384) | multiaxis | 11.41 s | 11.06 s | 0.97x |
| (768, 672, 576) | multiaxis | 56.30 s | 60.06 s | 1.07x |
| (1024, 1008, 992) | multiaxis | 310.06 s | 431.07 s | 1.39x |

mbirtorch runs at parity at the smallest multiaxis cell and leads
everywhere else, by 1.39x at the 1024-class and 1.29x at the
translation production cell.  The mbirtorch walls also reproduce
their recorded floors values to the first decimal, which ties the
two harnesses together.  The volume fingerprints agree across
frameworks at 5.7e-6 to 3.9e-5 relative on every cell.  These
results indicate the two libraries compute the same reconstruction
and the compiled torch bodies are not behind the reference
implementation anywhere in this family.

**Memory reads mixed, with mbirtorch leaner at the production
sizes.**  The jax arm peaked at 58.16 GB against mbirtorch's 34.75
at the multiaxis 1024-class, and at 41.46 against 27.22 at the
translation cell; at the multiaxis 512-class the order reverses
(6.04 against 11.38).

**One note rides the 1024-class peak.**  The memory ledger models
that cell at 68 GB on one device, at an H100's edge once the margin
applies (§1.22's corrected pricing).  The measured process peak
here is 34.75 GB for the whole 3-iteration reconstruction.  The
ledger prices the worst recorded slab concurrency, so the
conservative direction is expected; the factor of two at this
decision-relevant cell is recorded for the next time the capacity
question is asked, since a preflight priced 2x above the measured
peak refuses runs that fit.

**What this does to the kernel question.**  The anchor removes the
catching-up motive: the other framework demonstrates no speed these
bodies fail to reach.  What remains in favor of kernels is §1.42's
absolute evidence, which binds both frameworks alike: per-batch
host work near pacing at the 1024-class, back kernels at the memory
ceiling, and single launches moving gigabytes of intermediates.
The parallel kernel history (a 3.45x forward) shows what exploiting
the access pattern can return where the compiled realization sits
at those limits.  The capacity argument stands on the 2048-class
target, where sharding is known not to divide torch-body peaks.
The decision rule stays the recorded one: need above the
1024-class, not elegance.

### 1.44 The 1024-class "host pacing" is device back-pressure
### through a full launch queue, not host work (2026-08-22, mg53,
### job 15429313)

mg51 left one follow-up: about 35 ms of host time per view batch
at the multiaxis 1024-class that was neither the launch API nor
the synchronize.  mg53 split that time on one H100 with four
instruments and one ablation: a timing leg tied to mg51's
semantics, a wrapper on the bound body separating host time
inside the compiled callable from the driver loop around it, a
profiler host-operator table, torch's own synchronization
warnings, and a re-timing under a different allocator setting in
a fresh process.  The job ran 12m40s and exited healthy; the run
detail is in mg53_host_cost_split.md.  The warm walls and
enqueues reproduce mg51's numbers (38.053/34.956 s forward and
23.647/19.347 s back at the 1024-class, against mg51's
38.05/34.95 and 23.64/19.35), which ties the two instruments.

**The host time is the device's own rate reflected back through a
full launch queue.**  The profiler names the mechanism: an event
called "Command Buffer Full" carries 35.27 s of the forward
call's profiled host time (91 percent of self CPU) and 19.34 s of
the back's (81 percent).  The runtime queues kernel launches, and
when the outstanding launches exceed the queue's depth, the next
launch call blocks until the device drains one; that blocked
interval is recorded under this name, separately from the launch
rows.  This separation is why mg51's runtime launch rows read
only 6.4 ms and the time seemed unaccounted.  The wrapped-body
split agrees with the profiler: 84 percent of the forward
enqueue lands inside body calls (29.43 of 34.96 s), and the rest
lands in the driver's own enqueued assembly operations, which
block on the same queue.

**Every other candidate measured zero or trivial.**  The compiled
dispatch costs 0.16 ms per batch: the compiled-graph call, the
compiled-region entry, and the dynamo cache lookup sum to 167 ms
over 1024 batches.  The allocator recorded zero device
allocations, zero frees, and zero retries across the measured
calls at both cells, and the expandable-segments ablation moved
no wall by more than 0.5 percent.  The synchronization detector
captured zero warnings at every cell and direction.

**What this corrects.**  §1.42 read the 1024-class as the host
within 8 to 18 percent of pacing both directions.  The causality
is the reverse.  The device is the limit at both cells; at the
1024-class each call issues 12,288 launches, far past the queue
depth, so the host clock reads the device's rate minus the
buffered tail.  At the 512-class a whole call's launches fit in
the queue, which is why its enqueue reads 10 ms.  There is no
per-batch host work worth remedying.

**What this does to the kernel case.**  The speed argument loses
its host-pacing part and keeps its device-side parts: the back's
top kernels at the memory ceiling, and every batch moving
gigabytes of slab intermediates that a fused kernel does not
materialize.  The capacity case is untouched and remains the
strongest.  One instrument lesson is recorded: the runtime's
launch rows exclude queue-blocked time, which the profiler
reports under its own event name, so an enqueue clock near the
wall does not by itself say the host is doing work.

### 1.45 The multiaxis kernels' first composed measurement: 4.0x
### to 4.6x over the compiled bodies at every floors cell, the
### same values, and the slab class gone from the peaks
### (2026-08-22, mg54, job 15432699)

The multiaxis geometry gained hand-written forward and back
kernels this day and began selecting them wherever the per-device
value self-check passes, with no composed speed measurement
behind that selection.  mg54 made the first one, on one H100.
Each cell ran the kernel route and the torch-body route in fresh
processes on the same staged sinogram, the exact bytes mg52
measured, under the floors protocol: seed 13, a cold pass, then
the warm median of three 3-iteration reconstructions.  The run
detail is in mg54_multiaxis_kernel_ab.md.

**The kernel route is 4.0x to 4.6x faster, and the lead grows
with size.**

| cell | kernel | torch | kernel/torch | mbirjax (mg52) |
|---|---|---|---|---|
| (512, 448, 384) | 2.90 s | 11.41 s | 0.25 | 11.06 s |
| (768, 672, 576) | 12.71 s | 55.99 s | 0.23 | 60.06 s |
| (1024, 1008, 992) | 67.64 s | 309.92 s | 0.22 | 431.07 s |

The torch arms reproduce the recorded floors walls at 0.995 to
1.001, which ties this instrument to every earlier one.  Read
against the mg52 anchor, the kernel route at the 1024-class runs
6.4x faster than mbirjax on the same staged input.

**The values are the same.**  The float64 fingerprints of the
final reconstructions agree between the two routes at 1.6e-7
relative or better at every cell.

**The peaks lose the slab class.**  Peak device allocation reads
1.96 GB against 11.38 at the 512-class, 6.54 against 15.12 at the
768-class, and 24.11 against 34.75 at the 1024-class.  The
remaining kernel-route peak is the problem's own arrays and the
engine's state.  Those are the terms sharding divides, which is
what the multi-device and 2048-class measurements test next.

**The batch structure moved exactly as the kernels' cost model
predicts.**  The kernel bodies declare per-view costs with no
gather slab, so the driver chose 128-view batches at every cell,
where the torch bodies were forced to 9, 2, and 1 view.  One
projection call at the 1024-class is 8 body calls instead of
1024.  The measured single call: the forward runs 8.97 s of wall
with 7.82 s of enqueue, and the back runs 4.80 s of wall with
5.6 ms of enqueue, against the torch bodies' 38.05 and 23.65 s
(§1.42).  The back's launch-queue pressure is gone entirely; the
forward keeps some, across its 8 large launches.

**This closes item D7's multiaxis half by measurement.**  The
shipped forward kernel gathers on the vertical axis and scatters
on the horizontal, which is the organization Charlie
hypothesized.  The composed route built on that organization runs
4.0x to 4.6x faster than the torch route, whose forward scatters
on the vertical axis.  The scatter-mirror kernel form stays
unbuilt: it was the fallback for a disappointment that did not
occur.  D7's translation half stays open with translation.

**What this does to the campaign.**  The kernels now hold their
default-on selection under the first campaign's own standard,
a composed win at every measured cell.  The tile constants are
still the cone kernels' adopted values, so a tuning sweep is a
recorded follow-up that can only add to these numbers.  The
capacity question moves to the multi-device and 2048-class
measurements.

### 1.46 Sharding divides the kernel-route peaks, four devices
### run the 1024-class 2.97x faster, and the 2048-class multiaxis
### reconstruction completes on the standard node (2026-08-22,
### mg55, job 15434826)

Two questions remained that only several devices could answer:
whether sharding divides the per-device peaks now that the torch
bodies' temporaries are gone, and whether the 2048-class runs at
all.  mg55 answered both on one four-H100 node, kernel route
throughout, every arm asserting both kernels bound, the 1024
input reused from the earlier runs by md5.  The run detail is in
mg55_multiaxis_scale.md.

**The 1024-class now scales, in speed and in memory.**

| devices | warm | speedup | busiest peak | peak ratio |
|---|---|---|---|---|
| 1 | 67.63 s | 1.00x | 24.11 GB | 1.00x |
| 2 | 37.43 s | 1.81x | 12.95 GB | 0.54x |
| 4 | 22.75 s | 2.97x | 7.49 GB | 0.31x |

The one-device arm reproduces mg54.  The fingerprints agree
across counts at 5.4e-9 relative or better, and every count
follows the identical forward-model error trajectory.  The
comparison with history: the torch bodies LOST at two devices at
this cell before the recompile remedy (0.80x, §1.36), and their
peaks barely shrank under sharding.  The kernel route's busiest
peak now halves with each doubling.  Four devices run this cell
in 22.75 s, which is 13.6x the torch route's one-device wall and
19x the recorded mbirjax wall on the same staged input.

**The 2048-class demonstration completed.**  Cell (2048, 2016,
1984), recon (1984, 1984, 2297), 9.0e9 voxels.  The four-device
arm built its own input in 90 s (one full forward projection of
the volume through the kernels took 51 s), then reconstructed:
cold 355.95 s, warm 298.81 s at 0.1 percent spread, per-device
peaks 50.59 / 50.27 / 50.27 / 48.61 GB, forward-model error
falling across the iterations.  The two-device arm ran out of
memory and is recorded as that result: this cell fits the node at
four devices and not at two.  No torch-body route can run this
cell at any device count, which was the capacity case the
campaign was ruled on.

**The memory ledger is nearly exact on the kernel route.**  Its
modeled busiest-device peaks against the measured ones: 1.03x,
1.07x, and 1.03x at the 1024-class counts, and 1.07x at the
2048-class on four devices.  It models the 2048-class at 194 GB
on one device and 103 GB per device on two, so it correctly
refuses the count that measured out of memory and correctly
admits the one that completed.  These readings resolve the
recorded 2x-conservatism follow-up (§1.43): that factor described
the TORCH-body pricing at the multiaxis 1024-class, and the
kernel route the geometry now selects is priced to within 7
percent.  The torch-body 2x stands only for the fallback path.

**What remains of the campaign's measurement phase.**  The
widening floors for multiaxis are still the sentinel values of
the torch-body era, and the kernel route's measured knees (1.81x
at two devices, 2.97x at four at the 1024-class) say real floors
now exist to record.  The closing floors refresh re-measures the
family and blesses the cost hashes it froze.

### 1.47 The full floors refresh on the kernel tree: the
### multiaxis four-device floor rises to the 1024-class, cone's
### falls a class, every other floor reproduces, and the
### staleness note clears (2026-08-22, mg56, job 15435735)

Eight cost inputs had drifted since mg48: the multiaxis kernel
work and six files from earlier commits.  mg56 therefore ran the
full refresh, every family at every planned row, on four H100s
against the committed kernel tree (c024ec9), in 38m51s against
mg48's 1h34m; the kernels cut the multiaxis arms about four-fold.
44 arms ran, warm spreads mostly under 3 percent.  The verdicts,
the pasted floors, and the run detail are in mg56_floors.md.

**The multiaxis knees moved the way a four-fold one-device
speedup predicts.**  The two-device floor holds at the 512-class
(0.86x at the 384-class, 1.38x at the 512-class, 1.72x at the
768-class): the one-device and two-device walls fell together, so
the knee stayed.  The four-device floor RISES from the 512-class
to the 1024-class.  mg56 read 0.70x at the 512-class and a thin
1.09x at the 768-class, under the margin, with nothing larger on
its ladder; the winning reading is mg55's 1.64x at the
1024-class under the same protocol, and the coarse rule's
round-up lands on the same cell.  The row is placed by hand on
those two instruments, and its note says so.  The reading makes
physical sense: when one device is this fast, small cells no
longer amortize a four-way fan-out.

**Cone's four-device floor falls one class.**  The 768-class read
1.15x against two devices, clearing the margin it had missed at
every measurement since the width padding landed, with 1.64x at
the 1024-class.  Parallel and translation reproduce mg48 within
their spreads and every floor holds.  The denoiser stays a
sentinel at both counts, but the top-cell ratios rose from
0.67x/0.61x to 0.93x/0.90x after the denoiser performance
commit, so a larger ladder may yet find admission.

**The bookkeeping closes clean.**  The floors, the fifteen cost
hashes (triton_multiaxis.py now among them), and the table
checksum were re-recorded together; the staleness note reads
None, the floors test battery passes at 30, and the suite holds
696.  One test moved with the floors: the multiaxis admission
test now states the split behavior (two devices from the
512-class, four from the 1024-class) instead of the torch-body
era's shared 512-class floor.

### 1.48 What a first reconstruction costs: the compile caches
### carry most of it, dynamo tracing is what a cache cannot
### remove, and two host-side setup steps are pure overhead
### (2026-08-23, mg57 job 15449106 and mg58 job 15449170)

Every measurement in this series reports a warm median and
discards the cold pass, so what a user waits through on a first
reconstruction had never been attributed.  mg57 measured it at
the parallel (1024, 1008, 992) cell on H100s, with the sinogram
staged by a separate process so no measured arm compiled anything
to build its own input, and with both compile caches (torch's
inductor cache and Triton's JIT cache) owned by the harness so a
cold arm could not silently reuse an earlier job's compiles.  The
run detail is in mg57_cold_start.md.

**The same reconstruction, three ways:**

| | one device | four devices |
|---|---|---|
| first run, both caches empty | 49.48 s | 41.11 s |
| new process, caches full | 26.02 s | 19.51 s |
| in-process warm | 21.17 s | 9.60 s |

The warm readings reproduce the recorded floors-refresh medians
(21.30 s and 9.53 s), which ties this instrument to the earlier
ones.  Read the columns as three costs: the caches remove 23.46 s
at one device and 21.60 s at four; a fresh process still pays
4.84 s and 9.90 s beyond warm; the rest is the reconstruction.
**At four devices the fresh-process cost exceeds the
reconstruction it precedes.**

**Dynamo tracing is what a compile cache cannot remove, and it
multiplies by device count.**  From dynamo's own accounting: the
outermost compile phase reads 16.07 s at one device on the first
run and 4.42 s in every later process; at four devices, 25.20 s
and 8.00 s.  The unique graph count is 9 at one device and 36 at
four, exactly four times, which is the per-device compiled
instance design.  The library already pins the inductor cache and
enables the FX-graph cache, and its comment states the residue
correctly; what the measurement adds is the size of that residue
at four devices, where 8.00 s of tracing precedes a 9.60 s
reconstruction.  The Triton side is small by comparison: 10
compiling launches costing 1.19 s at one device, 57 costing
4.82 s at four.

**Two host-side setup steps are overhead rather than
arithmetic.**  The setup phase `initialize_recon` costs 2.6 to
2.7 s warm and does not change with device count, which is the
signature of host work.  mg58 attributed it directly at the same
cell.  The input validation runs `np.isfinite` over the whole
sinogram and the whole weights array on the host, 0.502 s and
0.504 s for 3.81 GiB each; the same check on the device, where
the sinogram is sent anyway, measures 0.0076 s, or 66 times
cheaper.  The pixel partitions cost 0.414 s and build all eleven
granularity levels, while a three-iteration run visits three of
them (4, 16 and 64); building only the visited levels measures
0.125 s.  The remaining setup steps are the regularization
parameters at 0.320 s and the volume's return to the host at
about 0.95 s.

**One caveat rides the partition reading.**  The generator's
random-call sequence is deliberate and its docstring forbids
reordering, because seeded runs must reproduce exactly.  Skipping
unvisited levels would change which draws happen, so that saving
is not free: it costs reproducibility against every recorded
seeded result unless the draws are preserved and only the sort
and the transfer are skipped.

**One hygiene gap the run exposed.**  The library pins the
inductor cache to `~/.mbirtorch/torch_cache` but leaves Triton's
JIT cache at its default, so the two halves of the compile cache
live in different places and `mbirtorch.clear_cache()`, which
removes the `~/.mbirtorch` tree, clears only one of them.  The
cluster's own cache directory held 145 MB in 939 entries at the
time of this run, none of it under the library's control.

### 1.49 The input checks now read a minimum and a maximum, and
### the setup phase falls from 2.61 s to 0.855 s (2026-08-23,
### mg60, job 15449436)

§1.48 found the reconstruction's setup phase spending most of its
time on host-side input validation.  The old formulation asked
each question with its own full pass: `np.isfinite` over the
sinogram, and over the weights an `isfinite`, a negative test and
an all-zero test, each allocating a boolean array as long as the
input.  Every one of those questions is answered by the array's
minimum and maximum together, because a NaN propagates into both,
an infinity appears in the extreme on its side, and the sign and
all-zero questions are the pair's to answer directly.  The
library now reads that pair through torch on the array's own
device, so a host array is read across the process's threads
rather than on one, and an array already on a device is not
pulled back to be checked.

**Measured at the 1024-class parallel cell, 3.81 GiB per array,
medians of three:** the sinogram check falls from 0.505 s to
0.061 s and the weights checks from 1.309 s to 0.063 s, so the
validation falls from 1.814 s to 0.124 s.  The whole
`initialize_recon` phase falls from the 2.61 s of §1.48 to
0.855 s, and that remainder now attributes exactly: the pixel
partitions at 0.414 s, the regularization parameters at 0.320 s,
and the checks at 0.124 s.  The saving is about 1.76 s per
reconstruction, which is 8 percent of the one-device warm wall
and 18 percent of the four-device one.

The old and new formulations were compared on the same arrays and
agree on every case tested: clean, NaN, positive and negative
infinity, negative values, all-zero, and partly-zero.  The seven
error paths raise the same exceptions with the same messages, and
the suite is unchanged at 696 passed.

**The weights reading corrects §1.48 in passing.**  That section
attributed 1.0 s to two host scans and left about 0.9 s of the
phase unexplained.  The weights path was making three passes, not
one, at 1.309 s in total; the unexplained residue was its other
two.

**One hygiene change rides with it.**  The library now pins
Triton's JIT cache beside the inductor cache it already pinned,
at `~/.mbirtorch/triton_cache`, so both halves of the compile
cache live in one place and `clear_cache()` clears both.  Before
this the Triton half sat at its own default, outside anything the
library named.

### 1.50 What torch compiles during a parallel reconstruction,
### why making its shapes dynamic does not help, and the trade
### the compile itself is (2026-08-23, mg59 job 15449621 and
### mg61 job 15449694)

§1.48 left dynamo tracing as the largest per-process cost that a
compile cache cannot remove: 4.42 s at one device and 8.00 s at
four, with the graph count multiplying by the device count.  Two
experiments followed.

**Six frames compile, and none of them is a projection.**  mg59
enumerated them from torch's own cache entries: the qGGMRF
gradient and Hessian, the update apply, the update direction, the
prior line terms, and two line-search terms.  All six are reached
through the VCD subset updater, and all are small array
arithmetic.  The projection bodies never appear, because parallel
beam runs hand-written kernels; the run confirmed all eight bound
entries were the Triton wrappers.

**The extra variants are pixel-count guards, and removing them
does not remove seconds.**  At one device each of four frames
compiled twice, the second time because a tensor's leading
dimension changed with the subset size (192,810 against 48,203).
Marking that dimension dynamic cut the graphs from 9 to 7 at one
device and from 36 to 22 at four, with the reconstruction values
unchanged (differences of 1e-10 against the run's own 1e-7
device-count yardstick).  The tracing time did not follow: 5.52 s
for 7 graphs against 4.15 s for 9 at one device, and 7.49 s
against 7.74 s at four.  A dynamic graph costs more to trace than
a static one, so the count was never the driver.  **The candidate
is therefore declined.**

**What the four-fold at four devices really is.**  Two mechanisms
together, both named in torch's own guard messages: the
per-device compiled instances the projector layer creates, and
dynamo's own device-index guard.  The second acts alone as well.
The update-direction frame has ONE compiled instance and still
produced eight variants at four devices, guarded on the tensor's
device index.  A remedy aimed only at the per-device instances
would therefore not remove the multiplication.  A third guard
appears at four devices only: the qGGMRF halo is None at a true
volume edge and a tensor at an interior shard boundary.

**The compile is a trade, and mg61 priced it at the DEFAULT
workload.**  Compiling those six frames is worth measuring
against not compiling them at all, because on this geometry
nothing else goes through the compiler.  At the library's default
of 15 iterations, warm medians and first-in-process walls:

| devices | setting | first | warm |
|---|---|---|---|
| 1 | compiled | 131.62 s | 115.43 s |
| 1 | off | 149.83 s | 146.35 s |
| 4 | compiled | 63.40 s | 50.37 s |
| 4 | off | 54.70 s | 52.25 s |

The values are identical to 2e-13 through 1e-10.  **At one device
compiling wins outright**, by 30.92 s on a warm reconstruction and
by 18.21 s on the very first one in the process, so it repays its
own cost within a single reconstruction.  At four devices it buys
1.88 s per warm reconstruction against a first-reconstruction
penalty of 8.70 s, so it repays after about five reconstructions
in one process.

**The first reading of this trade was taken at the wrong
workload, and it inverted the one-device answer.**  mg61 first
ran at 3 iterations, the mg series' measuring protocol, where
compiling bought 2.04 s at one device and 0.39 s at four against
the same one-time cost.  That charged the full per-process cost
against a small fraction of the recurring benefit.  The unit that
matters is not the iteration but the SUBSET UPDATE, which is what
calls these frames: the default partition sequence spends its
first three iterations on the 4-, 16- and 64-subset partitions
and every later one on the 128-subset partition, so 3 iterations
make 84 subset updates and 15 make 1,620, a factor of 19 for a
factor of 5 in iterations.  Per subset update the two runs agree
(about 24 ms and 19 ms at one device), which is what makes the
short run's conclusion an artifact of its length rather than a
disagreement.

**The recommendation is therefore to leave the default alone.**
Compiling earns its cost at the workload users actually run.  The
one case where turning it off helps is a single reconstruction on
four devices, worth 8.70 s of a 63.40 s wall; a second
reconstruction in the same process erases most of that and a
fifth erases all of it.  The four-device penalty also
cross-checks §1.48: 63.40 minus 50.37 is 13.03 s of
first-reconstruction cost with compilation against 2.45 s
without, a difference of 10.6 s beside that section's
independently measured 9.90 s.

**What this does NOT license.**  The setting is global, and on
the geometries whose projection bodies are torch code rather than
kernels the compile is worth far more than this; the recorded
chain-level wins there run to a factor of 22.  Nothing here
argues for changing what those geometries do.

## 3. The crossover ladder (mg4)

This section locates where each device count starts paying, which is
decision 1's data.  The ladder sweeps six parallel-family sizes below
the gate cells, spot-checks cone, and re-measures the 1024 anchor to
tie itself to mg1.  Its deliverables are the per-count admission
points the guard's floors will encode, and the measured cost of
today's capacity-only automatic choice.

### 3.1 The parallel family

Speedup over n=1, warm medians, spreads 0.0 to 5.4 percent (the 5.4
is the 384-class n=1 arm; the next largest is 4.3):

| cell (sino elements) | n=2 | n=4 |
|---|---|---|
| 128 (1.4M) | 0.27x | 0.08x |
| 192 (4.6M) | 0.44x | 0.12x |
| 256 (11M) | 0.47x | 0.14x |
| 384 (37M) | 0.63x | 0.33x |
| 512 (88M) | **1.22x** | 0.75x |
| 768 (297M) | 1.60x | **1.18x** |
| 1024 anchor (1024M) | 1.02x | 1.72x |

The anchor ties the ladder to mg1: its 1.02x and 1.72x reproduce the
readout's numbers on a different node and job.  The knees are clean
under the §4 admission rule, which defines each count's crossover
against the best smaller ADMITTED count, not against n=1.  Parallel
n=2 admits at the 512 cell and wins at every larger size.  Parallel
n=4 first beats n=1 at the 768 cell, but its §4 crossover is against
n=2, and at 768 n=2 still wins, 1.60x to 1.18x; the n=4 floor
therefore sits at the conservative end of the 768-to-1024 bracket
(the floor_4 amendment, checkpoint ruling 2026-08-09).  The expected
256-class knee from the vcd size-floor prior was wrong by two ladder
steps; the measured knee is the 512 cell.

The largest-first loop with the amended floors chooses correctly at
every measured size: n=1 below 512, n=2 from 512 through 768, and n=4
past the bracket, where its 1.72x anchor beats the n=2 flatline.  The
pre-amendment floors would have picked a run about 36 percent longer
than the best choice at the 768 cell — exactly a mid-size the guard
exists to protect.

### 3.2 Cone

Cone's spot check reads n=2 at 0.55x, 0.66x, and 1.02x across the 256,
384, and 512 cells, with mg1 adding 0.92x at 1024.  Cone n=2 therefore
never clears the admission rule: its one nominal win is 1.02x within a
1.1 percent spread, and the larger cell is a regression.  Cone n=4
reads 0.15x, 0.35x, and 0.68x at the spots and 1.16x at 1024, so its
floor sits between the 512 and 1024 cells.  The floor family
expresses cone by setting its n=2 floor to infinity, which loses
nothing measurable.  A bracketing cone cell between 512 and 1024
would tighten floor_4; it is DECLINED under §3.3's staleness
reasoning, and the coarse floor takes the conservative end of the
bracket instead.

### 3.3 Which knees are durable, and the staleness rule

The ladder's readings split by what limits them, and the split decides
how much floor precision is worth buying (Greg's concern, 2026-08-09).
Below the 384 cell the speedups are 0.08x to 0.63x and the limiter is
the fan-out and glue, which no forward change touches; those knees are
durable.  The forward-sensitive readings are cone at every n>1 count
and parallel 1024 at n=2, exactly where the §1.3 anomalies bind, so
the pending forward work — the attribution charter now, item 13 later —
can move those numbers substantially.

The floors survive that movement in the safe direction.  A faster
forward at n>1 moves knees down, so today's floors would hold a count
back where it newly pays; the cost is percent-scale on mid-size runs,
never the 3x-to-13x harm the guard prevents.  The rule that follows:
the floors ship coarse and conservative from this table, no further
ladder investment is made now, and a knee refresh — the three or four
cells nearest each floor, about fifteen minutes through the harness's
family and cell knobs — re-measures the floors after any forward-path
change lands.

### 3.4 The auto arms, and the harm the guard prevents

The automatic policy chose all four devices at every one of the
eleven measured shapes.  The cost of that choice is now a measured
curve: 13x slower than n=1 at the 128 cell, 7x at 256, 3x at 384, and
1.3x at parallel 512.  The choice is right only from the 768 cell up.

### 3.5 The shape probes, and the metric verdict

The pencil probe (1152, 336, 96) measured n=1 best, at n=2 0.48x.
Both candidate metrics agree with that outcome once the measured knee
is used: its 37M sinogram elements and its 3.1M recon voxels both sit
below their floors.  The probe therefore records a consistency point
rather than a discrimination.

The discrimination came from mg4b's sparse-view probe, and it settles
the metric.  That probe is (64, 448, 384), a shape where the two
candidates disagree.  Its 11M sinogram elements sit below the measured
512-cell floor, so the sinogram-element metric holds n=1.  Its 66M
recon voxels sit in the 512 cell's class, so the recon-voxel metric
admits n=2.  The measurement is unambiguous: n=1 runs 0.45 s, n=2 runs
0.84 s, and n=4 runs 2.32 s.  Widening to two devices is therefore a
1.87x regression at this shape, and widening to four costs 5.1x.
These results refute the recon-voxel metric and confirm the
sinogram-element metric.  **The guard indexes its floors on sinogram
elements.**

The thin-volume probe confirms that the shared conservatism is
correct.  That probe is (1024, 32, 768), where the view-dominated
physics argued that widening should pay even though both metrics hold
n=1.  It measured 1.61 s at n=1, 1.87 s at n=2, and 3.75 s at n=4.
Widening does not pay there either, so neither metric is too
conservative at this shape.

Both probes add auto-policy harm at shapes the ladder did not cover.
The automatic policy took all four devices at both, which cost 2.3x at
the thin-volume probe and 5.1x at the sparse-view probe.  The eight
measured arms passed every arm check, no row ran hot, and the warm
spreads sit at or under 2.7 percent everywhere except the two smallest
arms, whose spreads are 6.6 and 17.6 percent on sub-second cells.
Those spreads are far smaller than the differences they separate.

---

## 4. The value comparison (mg3)

This section carries goal 3: separating the value residual into its
partition-order and compile-latitude terms by construction, from one
shared sinogram at every count.  It is also the campaign's half of
the port-fidelity coverage, now that the goldens are opt-in
everywhere and inert in the nightly.

The pricing probe measured the unknown it existed for: eager-plain at
parallel 1024 costs 240.5 s per 3-iteration recon, cold equal to warm,
and the cone arm's partial progress implies about 550 s.  The scope
ruling followed: the three depth-10 eager arms are deferred at their
measured 2.5-to-4-hour price, and mg3 runs split as mg3a (job
15029600, base and ordering tiers) and mg3b (15029601, depth-10 jax
and production).  The deferred arms stay one env token away
(`MG3_DEEP_CLASSES`).

mg3b completed and mg3a did not.  mg3b ran its depth-10 tier to
completion in 39 minutes.  mg3a reached 31 of its 36 measured arms
within its eight-hour walltime, and slurm killed it during arm 36.
The five arms it missed are the cone-1024 tail at n>1: one eager arm,
two production arms, and two jax arms.  The cone-1024 eager arms are
what consumed the walltime, at 2.6 to 3.4 hours each.

A re-run of that tail would cost far more than five arms suggest.
Every arm of a cell reads one shared sinogram, and mg3b's cleanup
deleted the sinograms when it finished.  Only the md5 sidecars
survive.  A re-run must therefore regenerate the cone-1024 sinogram,
and its arms could not be compared against mg3a's completed cone-1024
rows, which read the deleted one.  The comparable scope is the whole
cone-1024 base tier, which carries three eager arms and costs about
eleven GPU-hours.  That is the same price the earlier scope ruling
already declined for the depth-10 eager arms, so the gap stays.

A cheaper partial re-run exists if the gap later matters.  Dropping
the eager class from that tier leaves the production and jax arms,
which cost about 2.5 GPU-hours and still measure the partition-order
and cross-framework terms at cone 1024.  The harness has no knob for
that today, because its class filter covers only the depth tier.

### 4.1 Validity, and how the columns were obtained

The readout is valid.  All 37 completed arms pass every arm check.  No
row ran hot, and no throttle reason was active on any GPU in any
sample.  Warm spreads sit at or under 3 percent on 35 of the 37 arms.
The two exceptions read 6.0 percent at the parallel 1024 jax n=4 arm
and 5.1 percent at the cone 512 eager-plain n=4 arm.

One provenance note is recorded.  The harness computes its value
columns only after every planned arm finishes, so the killed mg3a job
wrote none.  Its 31 per-arm sample volumes survived the kill, because
the cleanup step never ran.  The base-tier columns below were
recomputed from those samples through the harness's own comparison
function.  The depth-10 columns are the harness's own, written by the
completed mg3b job.

One rule was applied at analysis time.  mg3 carries no tolerance of its
own, and its expectation constants are annotations on each value row
rather than gates.  Ratios were therefore judged against the amended
benign-tiny rule, which holds that a ratio of two divergences both
below 1e-4 is meaningless.  That rule's floor is mg1's
`VALUE_RULER_FLOOR`.  The increment-1 checkpoint raised that floor to
1e-4.  The rule decides the device-count trends at parallel 1024 and
every trend in the compile-latitude column.

Base tier, three iterations, max-relative on the gathered volumes, one
shared sinogram per cell.  The partition-order entry is the worst of
the three same-framework columns at that coordinate.

| cell | n | partition order | its floor | compile latitude | cross framework |
|---|---|---|---|---|---|
| parallel 512 | 1 | — | — | 5.15e-6 | 5.40e-4 |
| parallel 512 | 2 | 8.75e-5 | 4.47e-4 | 2.84e-6 | 5.47e-4 |
| parallel 512 | 4 | 3.03e-4 | 9.49e-4 | 2.23e-6 | 5.43e-4 |
| parallel 1024 | 1 | — | — | 1.22e-5 | 6.11e-3 |
| parallel 1024 | 2 | 1.05e-5 | 4.47e-4 | 7.75e-6 | 6.11e-3 |
| parallel 1024 | 4 | 2.22e-5 | 9.49e-4 | 5.83e-6 | 6.08e-3 |
| cone 512 | 1 | — | — | 5.68e-6 | 8.43e-5 |
| cone 512 | 2 | 2.40e-4 | 4.57e-4 | 6.08e-6 | 8.45e-5 |
| cone 512 | 4 | 2.40e-4 | 4.95e-4 | 6.13e-6 | 8.44e-5 |
| cone 1024 | 1 | — | — | 9.25e-6 | 2.96e-4 |
| cone 1024 | 2 | 1.62e-5 | 4.57e-4 | absent | absent |
| cone 1024 | 4 | absent | 4.95e-4 | absent | absent |

### 4.2 The partition-order term

The partition-order term sits under its engine floor at every
coordinate measured.  The term is the divergence between one
framework's n-device reconstruction and its own single-device
reconstruction, at fixed values and one shared sinogram.  Its largest
reading is 3.03e-4, at parallel 512 with four devices, against a
registered floor of 9.49e-4 there.  Its closest approach to a floor is
at cone 512 with two devices, where 2.40e-4 sits at 0.53 of the 4.57e-4
floor.  These results indicate that partitioning costs less value than
the recorded engine floors allow.

The term belongs to the partition and not to the engine.  Torch
eager-plain, jax and torch production read the same number at every
cell.  At parallel 512 with four devices the three read 3.034e-4,
3.028e-4 and 3.034e-4.  At cone 512 with two devices they read
2.397e-4, 2.397e-4 and 2.399e-4.  These results indicate that neither
the framework nor the kernel-and-compiler choice moves the term, which
is what the by-construction separation predicted.

The term grows mildly with device count.  At parallel 512 it grows 3.5x
from two devices to four devices.  The floor class itself grows 2.1x
over that same step.  At cone 512 the term is flat, at 1.00x.  At
parallel 1024 both endpoints sit below 1e-4, so their ratio is not
interpretable and only the absolute floor applies there.  These results
indicate mild growth, because the steepest slope still leaves the term
a factor of three under its floor at four devices.

One slope is recorded rather than dismissed.  The parallel 512 slope
runs 1.6x steeper than the floor class's own slope over the same step.
The registration named strong growth with n a finding, and this is the only
cell where an interpretable slope exceeds the floor's.

This column also cross-checks against mg1.  mg1's gate table reports
own-count divergences at parallel 512 of 8.72e-5 and 8.70e-5 at two
devices, and 3.03e-4 twice at four devices.  mg3 reads 8.723e-5 and
8.692e-5, then 3.034e-4 and 3.028e-4, on a different node in a
different job from a different sinogram.  These results indicate that
the term is reproducible across instruments to three significant
figures.

### 4.3 The compile-latitude term

The compile-latitude term came in two to three decades below its
registered class.  The term is the divergence between the production
engine and eager-plain at equal coordinates.  Its largest reading is
1.22e-5, at parallel 1024 with one device, against a registered
parallel class of 5e-3.  Its largest cone reading is 9.25e-6, at cone
1024 with one device, against a registered cone class of 5e-4.  Every
reading of the column sits between 2.2e-6 and 1.2e-5.  These results
indicate that the production engine and the plain torch engine reach
the same answer at every cell and every count.

The registered class needs re-registering.  The registration imported it from the
composed kernel-versus-body columns at n=1, which sit in the low e-3
class for parallel and the e-4 class for cone.  Those columns do not
predict this one.  The ceiling this instrument measured is e-5.

The term does not grow with device count.  Every endpoint of every
ratio in this column sits below 1e-4, so every trend in it is
benign-tiny.  The raw ratios fall with device count at both parallel
cells and rise 8 percent at cone 512.  The registration named growth with n a finding, and no growth
appears even before the rule is applied.

The registered decay with depth is untested.  The depth tier ran jax
and production only, under the probe ruling that deferred its three
eager arms.  The production-versus-eager column therefore has no
partner at ten iterations, and the harness wrote it null at all three
counts.  Testing that decay needs the deferred arms, at their recorded
price of 2.5 to 4 GPU-hours.

### 4.4 The cross-framework column

The cross-framework column reproduced both documented classes exactly.
The column is production torch against jax, from one shared sinogram,
at equal coordinates.  At parallel 1024 and three iterations it reads
6.112e-3, 6.111e-3 and 6.084e-3 at one, two and four devices.  The
documented value there is 6.1e-3.  At ten iterations the same
coordinates read 8.768e-4, 8.770e-4 and 8.733e-4, against a documented
8.8e-4.

The other cells sit in the documented e-4 class.  Parallel 512 reads
5.4e-4 at all three counts, and cone 1024 reads 2.96e-4 at one device.
Cone 512 reads 8.4e-5, one notch below the e-4 class the expectation
named.  That departure runs downward.

The documented sevenfold decay holds at every device count.

| n | 3 iterations | 10 iterations | decay |
|---|---|---|---|
| 1 | 6.112e-3 | 8.768e-4 | 6.97x |
| 2 | 6.111e-3 | 8.770e-4 | 6.97x |
| 4 | 6.084e-3 | 8.733e-4 | 6.97x |

The documented pair implies 6.93x.  These results indicate that the
decay is a property of the iteration and not of the partition, which is
the question the depth tier was built to answer.

The whole-field measure decays further than the worst voxel.  The
norm-relative column falls from 8.03e-4 to 4.11e-5 over the same depth
step, a 19.5x decay, at every count.

The eager arms make the production column interpretable.  Eager-plain
against jax and production against jax agree to within 0.71 percent at
every coordinate, and to within 0.02 percent at parallel 1024.  These
results indicate that a user's cross-framework residual is the
framework difference, and not the torch engine's own kernel and
compiler choices.

Cone's depth-3 readings do not surprise, so the question of cone
depth-10 arms does not reopen.  Cone's cross-framework column reads
8.4e-5 at 512 and 2.96e-4 at 1024, its compile-latitude column reads
9.25e-6 at worst, and its partition-order column reads 2.40e-4 against
a 4.57e-4 floor.

### 4.5 What exceeded expectations, and what is missing

Nothing exceeded its registered expectation.  Every measured value sits
at or below the class registered for it, in every column and at every
coordinate.  Two departures run downward and are recorded above.  Those
two are the compile-latitude class of §4.3 and cone 512's
cross-framework reading of §4.4.

One unregistered observation is recorded without interpretation.  The
partition-order term is larger at the 512 cells than at the 1024 cells
in both geometries.  Parallel reads 8.75e-5 at 512 against 1.05e-5 at
1024 at two devices, and cone reads 2.40e-4 against 1.62e-5 at the same
count.  Nothing in the campaign registered a cell dependence for this
term.

The optional ordering tier never ran, and its absence is a harness
finding.  Its arms are gated on the depth cell in both scheduling
phases.  mg3a set `MG3_SKIP_DEEP`, which leaves no depth cell.  mg3b
set `MG3_ONLY_DEEP`, which sets the tier's own skip flag.  The split
therefore lost the tier by two different branches, and no warning was
printed.  The plan's claim that the production-versus-eager bundle reads
the compiler to leading order consequently has no check.  That claim also
matters less than it did, because the bundle itself measures 1.2e-5 at
worst.

Two smaller gaps are recorded.  Registered expectation 1 is stated over
torch production against its own single-device run.  The harness
computes that column for eager-plain and jax only.  That column was
recomputed here, and it tracks the eager column to within 1 percent.

The two tiers also read different artifacts at parallel 1024.  mg3b
regenerated the sinogram rather than reusing mg3a's, and the two
checksums differ by 2.2e-9 in relative terms.  That difference sits five
decades below the residual the depth comparison resolves, so the
comparison stands.

**Goal 3 is met (ruled 2026-08-10).**  The partition-order and
compile-latitude terms are separated by construction at every cell
that completed, and the cross-framework column reproduces both
documented classes at every device count.  Three readings stay
uncovered: cone 1024 above one device, the compile-latitude decay
with depth, and the compiler-versus-kernel ordering check.  The read
makes none of them look load-bearing, so the provisional acceptance
of the cone 1024 tail is confirmed and no partial re-run is
scheduled.

---

## 5. The mg5 triggers, evaluated (the checkpoint's ruling input)

This section converts the measurements into the checkpoint's decision
input.  The plan made every tuning lever contingent on a data
trigger, and the table below states each trigger's state with the
evidence that fired it or kept it quiet.  The ruling that follows
from this table is §6's first item.

| lever | trigger state | evidence |
|---|---|---|
| per-device view chunk | FIRED, then CLOSED by the sweep | the sweep left the shipped chunk within one percent of the best over chunks 32 to 256 at both 1024 cells, and left device-count scaling unchanged (§1.5) |
| seam: band-reduce restructure | NOT FIRED for time; FIRED for the ledger's model of it | band reduce is at or under 0.04 s per pass everywhere; mg2 charges it at up to 60 percent of the measured peak |
| streaming: `back_project_slice_band` | NOT FIRED at 1K; fires by arithmetic at 2K | every measured n>1 peak sits far below capacity (worst 14.3 GB of 80); at 2K the flat band-reduce term is 37 GB on the owner and becomes the wall (§6, the memory-scaling charter) |
| widening margin and ledger terms | FIRED | sixteen over-ceiling readings at n>1, localized to the back-loop and band-reduce charges; the back-loop item addresses the loop's real residency first, not only its charge (§2, §6) |
| orchestration (mega-region, CUDA graphs) | recorded, separate charter | the small-cell collapse persists with kernels (0.08x to 0.47x below the knee), dominated by fan-out and glue the regions do not cover |

The ruling on which arms proceed is §6, endorsed at the increment-1
checkpoint.

---

## 6. Recommendations (ruled 2026-08-09)

This section is the checkpoint's output: what proceeds, what waits,
and what does not happen, each with its reason.  The increment-1
ruling endorsed it with one amendment, floor_4, folded into 6.1; the
tuning charters and the guard implementation still land as their own
reviewed changes.  Each recommendation opens with a two-line summary,
and §6.7 gives the implementation order.

### 6.1 The guard: ship it now, with coarse floors

The guard ships with the measured, conservative floors, and the
ladder gets no further investment.  Forward work can only move the
knees down, so coarse floors stay safe while precision bought now
would be erased.

The floors are per geometry and per count, consulted only on the
unpinned automatic branch, and each count's floor comes from its
crossover against the best smaller ADMITTED count (the floor_4
amendment).  Parallel admits n=2 at the 512 cell's work size and n=4
at the conservative end of the 768-to-1024 bracket, because at 768
n=2 still wins, 1.60x to 1.18x.  Cone admits n=4 at the conservative
end of its own 512-to-1024 bracket, and never auto-admits n=2.
Cone's comparison count is n=1, because cone n=2 is never admitted.
Monotone floors express the mid-size reversal under this crossover
definition, which closes the plan's §8 preference-rule question.
Capacity always wins, exactly as planned.  The floors carry §3.3's
staleness rule: a fifteen-minute knee refresh re-measures them after
any forward-path change.

The metric that indexes the floors is now decided.  mg4b's
sparse-view probe refuted recon voxels and confirmed sinogram
elements, as §3.5 reports.  The guard therefore has everything it
needs, and it implements next.  The metric choice also survives
forward changes, even though the knee values do not.

**Refresh addendum (2026-08-10).**  The refresh script had its first
real run, 31 minutes on four GPUs, and it re-measured the floors on the
post-release code.  The cone n=2 sentinel tripped, exactly as its
registered condition specified.  Cone n=2 now wins at the 512-class
cell by 1.02x against a 0.52 percent spread, which is the same nominal
win that sat inside its own noise before the releases.  That entry is
therefore the table's first finite cone n=2 floor, and its note marks
it marginal.  The other floors kept their values, and cone n=4's
bracket tightened from 512-to-1024 down to 768-to-1024.  The
stale-hash debt the back-loop pair stamped is cleared, so `STALE_SINCE`
is None again.  One numerical coincidence is recorded without
interpretation.  The cone and parallel floor tables are now numerically
identical.  Their brackets, spreads, and comparison counts still
differ.  The refresh's dry planning output already pairs cone n=4
against n=2, which is what the crossover rule prescribes for the next
refresh now that cone n=2 is admitted.

### 6.2 Charter A: attribute the forward at n>1

One probe sweep attributes the campaign's two performance anomalies
before any constant moves.  The flat cone forward and the parallel
n=2 rise are the terms costing torch its n>1 scaling, and both point
at a per-device regime change the sweep can isolate.

The sweep has run, and §1.5 reports it.  The attribution rules out a
chunk constant, because the whole ladder moves total time by under one
percent.  It rules out the realized batch, because that batch does not
vary with device count.  It attributes the flatness to data movement
that is invariant in device count, namely the broadcast of every
reconstruction band to every device and, for cone, the accumulation of
full-row partial sinograms.  (Superseded 2026-08-10: the mg9 instrument
measured that attribution directly and refuted it, placing the flat
term inside the kernel launches; §1.7 carries the measurement and the
remedy memo's §8 carries the revised remedy.)

The remedy is therefore a driver change or the sorted-stream work, and
this charter's next step is to choose between them.  Two follow-ups
belong to that step.  The cone back projection also rises at two
devices, from 25.3 to 30.8 s, and this sweep varied only the forward
chunk, so the back needs its own variant.  The parallel remainder
cannot be read from host-side brackets, for the reason §1.5 gives, so
any remainder study must use device spans.

### 6.3 Charter B: the direct-recon back loop — attribute, then fix

The back loop itself gets addressed, not just its ledger charge.
Attribution comes first, and the remedy menu has a shallow and a
structural option.

Step one is complete, and §2 reports it.  The probe confirmed the real
residency and the phantom sum, so the shallow remedy proceeds.  That
remedy is the one-line release of the previous pass's partials, and it
lands paired with the ledger's sub-phase split in one change.  The
split must use the probe's measured block counts rather than the code
reading, which ran about one cylinder high.  The optional rider stays
optional, and the measured block counts suggest its gain is smaller
than the memo estimated.

The structural remedy remains the second option, and nothing about it
has changed.
The structural remedy is pixel batching inside the view batch, which
mbirjax's own back projection already implements and the port
simplified away; it bounds the full-index transient at a chunk-sized
block and also serves the translation-scale need.  Its costs are
known in kind and need one probe in degree: sinogram traffic
amplifies by roughly the chunk count, and the chunk shape must be
static with a padded tail, or per-chunk recompiles eat the win.  Only
the gap remaining after the chosen remedy becomes a charge
correction, band reduce included.

### 6.4 Charter C: memory scaling by 2D tiling, at the 2K design point

Production recons are 2K-class and larger, and at that scale the
memory story changes from tuning to feasibility.  This charter
unifies the scattered memory levers under one design: full
mbirjax-parity 2D TILING of both projection directions, plus the
slice-band and reduce legs.

The 1K cells this campaign gates on are the coming low end of
production, and the scale arithmetic flips two of §5's verdicts at
2K.  At (2048, 2016, 1984) the recon is 31.7 GB, the sinogram
32.8 GB, and one full-pixel cylinder 24.9 GB, so single-device VCD is
impossible and multi-device becomes the only path rather than the
faster one.  The direct recon's three-cylinder moment is about 75 GB,
which exceeds a device even sharded, so pixel chunking is mandatory
there.  The band-reduce term is flat in the count at about 1.5
cylinders, 37 GB on the owner at 2K, so it becomes the capacity wall
that more devices cannot shrink.  The charter's legs: the 2D tiling
(view axis shipped, pixel axis restored per charter B), the
slice-band leg through the existing streaming hooks, and the seam
restructure as the reduce leg, with item 6's translation-scale need
served by the same structure.  Its first step costs no cluster time.
The ledger is closed-form, so the full 2K capacity table per count is
a design-instrument computation, and it says which legs bind before
any code moves.

### 6.5 Declined at the measured cells

Three levers are declined at 1K, each with its reason and its return
condition.  Streaming has no capacity pressure at the measured cells;
the seam restructure costs nothing in time (0.04 s per pass); the
orchestration levers are real but sit where the guard already holds
n=1.  The first two return as charter C's legs at 2K.

**Addendum (2026-08-10): the seam's memory premise is now met at 1K.**
The seam restructure was declined here because it cost nothing in time,
and it returned at 2K as charter C's reduce leg.  The memory side of
that return condition is now met at 1K by direct measurement.  With the
releases in, the binding sub-step is the band reduce at three of the
four re-probe arms, namely both parallel 1024 counts and cone 1024 at
four devices.  This measurement updates the ruling's premise.  It does
not re-rule the decision.  The seam is therefore a candidate to pull
forward from charter C, and the decision is Greg's.  §6.7 carries that
pending decision in the sequencing.

### 6.6 The cadence numbers

The per-cell costs for the nightly's n>1 cadence are in §1.4, and
the call is made (Greg, 2026-08-09): the n>1 rows keep full nightly
cadence through the tuning window, cheap insurance at about 2
GPU-hours per night while the guard, charter B's fix, and any chunk
changes move the n>1 path.  No nightly wiring changes.  Revisit at
campaign close; the authoritative per-night figure remains the
nightly's own trial.

### 6.7 Sequencing

Three tracks run in parallel, because they touch disjoint code.  The
guard (6.1) implements as soon as mg4b lands its metric answer, and
waits on nothing else.  Charter A's probe sweep and charter B's
attribution both start immediately; they read different regions and
can share a cluster window.

Within and after the tracks, order is forced by dependency.
Charter B's remedy lands before the ledger charges are re-derived,
because the charges must describe the fixed loop, not the old one.
The refined charges then feed charter C's 2K table, because a table
built on the over-charges mg2 flagged would mis-rank the legs.  The
table picks charter C's first leg.  Last, the knee refresh of §3.3
runs after charter A's remedy (or item 13) changes the forward path,
and the floors update from it.  At close-out the `usr_multi_gpu.rst`
timing table refreshes from mg1's gate table, and the item-13 entry
gate is formally recorded as satisfied from §1.5's device-span share,
which supersedes §1.3's parallel host-bracket shares.  The cadence
call (6.6) is decided and recorded there.

The seam sequencing question is closed.  Greg ruled on 2026-08-10 to
hold the reduce leg to 2K, where charter C's design work owns it;
§6.5's addendum carries the measured premise that informed the call.

---

## 7. Also recorded

This section holds results that belong to the campaign's record but
to no single instrument: cross-cutting confirmations, economics, and
bookkeeping a future reader will want beside the tables.

The funnel refactor's neutrality is measured, not asserted: the
2026-08-09 nightly gated the funnel commit against the `ae9bb6f9` n>1
seed and read no changes at any cell or count.

One harness defect is recorded so that it is not mistaken for a
library defect.  The back-loop probe's single-device ruler arm failed
with a `padded_shard_ranges` error.  The failing frame is in the probe
itself, and the reconstruction it was measuring completed normally.
Single-device reconstruction is exercised by the whole test suite and
by the n=1 arms of four other campaign jobs, all of which pass.  The
ruler arm returns whenever the probe's n=1 path is fixed.

The probe economics held the campaign to budget.  Pre-tuning spend
through mg4 is about 7 GPU-hours against the plan's 20-to-34
projection for the same jobs, mostly because warm repeats came in far
under the pessimistic per-arm estimates.

Every mg script, both sbatch chains, and this page are staged in
`mbirjax_plans`; raw rows stay on scratch per convention.
