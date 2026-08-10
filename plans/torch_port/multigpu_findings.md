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

---

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
full-row partial sinograms.

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
