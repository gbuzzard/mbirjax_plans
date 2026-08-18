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
