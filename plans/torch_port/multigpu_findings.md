# Multi-GPU performance: the readout and its findings

**Status:** INCREMENT-1 CHECKPOINT (2026-08-09), with the mg4 ladder
already in hand.  mg1, mg2, and mg4 are complete and validated.  mg3a,
mg3b, and the mg4b probe addendum are running; their sections below are
marked OPEN and this page closes when they land.  The plan is
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
`/scratch/gautschi/buzzard/torch_p3/results/`.

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

### 1.4 Costs, for the cadence decision

The full 44-arm readout cost 1:22 on four GPUs, about 5.5 GPU-hours.
Per-arm subprocess walls run 1 to 2.5 minutes at the 512 cells and 2
to 5 minutes at the 1024 cells, cold pass included.  These are the
per-cell costs the nightly's n>1 rows would add; the nightly's own
trial remains the authoritative per-night number.

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

The over-charge admits two readings, and the distinction decides the
remedy (Greg's direction, 2026-08-09).  The ledger may be carrying
the n=1-era three-cylinder back-loop charge into the banded n>1 path,
whose reduce structure differs; that reading is a phantom charge, and
its remedy is a charge correction.  Or the loop may genuinely hold an
avoidable cylinder at n>1, the same residency species as the
`weighted_fwd` release; that reading is a real cost, and its remedy is
a code fix that the ledger predicts before any cluster time is spent.
The charter in §6 attributes first and prefers the code fix wherever
the residency is real, so the loop itself is addressed rather than
only its prediction.

The `hess_weights` question closes.  Unweighted and weighted peaks at
parallel 1024 differ by -0.06 GB at n=2 and -0.03 GB at n=4, so the
declined release stays declined with n>1 evidence behind it.

---

## 3. The crossover ladder (mg4)

This section locates where each device count starts paying, which is
decision 1's data.  The ladder sweeps six parallel-family sizes below
the gate cells, spot-checks cone, and re-measures the 1024 anchor to
tie itself to mg1.  Its deliverables are the per-count admission
points the guard's floors will encode, and the measured cost of
today's capacity-only automatic choice.

### 3.1 The parallel family

Speedup over n=1, warm medians, spreads 0.0 to 2.5 percent:

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

### 3.5 The pencil probe, and where discrimination now lives

The pencil probe (1152, 336, 96) measured n=1 best, at n=2 0.48x.
Both candidate metrics agree with that outcome once the measured knee
is used: its 37M sinogram elements and its 3.1M recon voxels both sit
below their floors.  The probe therefore records a consistency point
rather than a discrimination.  The discrimination now rests on mg4b's
sparse-view probe, where the metrics genuinely disagree: 11M sinogram
elements say hold n=1, while its 512-class recon says admit n=2.  The
thin-volume probe tests both metrics' conservatism at once, because
the view-dominated physics argues widening should pay there while
both metrics say hold.

**OPEN: mg4b (job 15030714) runs after mg3b.**

---

## 4. The value comparison (mg3) — OPEN

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

**OPEN: results land this afternoon.**

---

## 5. The mg5 triggers, evaluated (the checkpoint's ruling input)

This section converts the measurements into the checkpoint's decision
input.  The plan made every tuning lever contingent on a data
trigger, and the table below states each trigger's state with the
evidence that fired it or kept it quiet.  The ruling that follows
from this table is §6's first item.

| lever | trigger state | evidence |
|---|---|---|
| per-device view chunk | FIRED, via the forward anomalies | the cone forward flat in count; the parallel forward rising at n=2; realized-batch columns recorded per device for the attribution |
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
end of its own 512-to-1024 bracket — its comparison count is n=1,
since cone n=2 is never admitted — and never auto-admits n=2.
Monotone floors express the mid-size reversal under this crossover
definition, which closes the plan's §8 preference-rule question.
Capacity always wins, exactly as planned.  The floors carry §3.3's
staleness rule: a fifteen-minute knee refresh re-measures them after
any forward-path change.  The metric that indexes the floors
waits one job.  mg4b's sparse-view probe is the point where the
candidate metrics disagree, and the metric choice survives forward
changes even though the knee values do not.

### 6.2 Charter A: attribute the forward at n>1

One probe sweep attributes the campaign's two performance anomalies
before any constant moves.  The flat cone forward and the parallel
n=2 rise are the terms costing torch its n>1 scaling, and both point
at a per-device regime change the sweep can isolate.

The sweep runs at the two anomalous coordinates, cone 1024 at all
counts and parallel 1024 at n=2.  It varies the per-device view chunk
and reads the realized batches, which mg1 already records per device.
Its output is an attribution, and the attribution decides whether a
chunk constant, a driver change, or item 13's sorted stream is the
remedy.

### 6.3 Charter B: the direct-recon back loop — attribute, then fix

The back loop itself gets addressed, not just its ledger charge.
Attribution comes first, and the remedy menu has a shallow and a
structural option.

Step one attributes the n>1 residency: does the banded back path
actually hold the three cylinders the ledger charges, read from the
code and one phase-probe run.  Step two fixes what is real before
re-deriving what is not.  The shallow remedy is the `weighted_fwd`
treatment for an avoidable co-live cylinder: a small code change
whose effect the ledger predicts and one calibration run confirms.
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
gate is formally recorded as satisfied from §1.3's forward shares.
The cadence call (6.6) is decided and recorded there.

---

## 7. Also recorded

This section holds results that belong to the campaign's record but
to no single instrument: cross-cutting confirmations, economics, and
bookkeeping a future reader will want beside the tables.

The funnel refactor's neutrality is measured, not asserted: the
2026-08-09 nightly gated the funnel commit against the `ae9bb6f9` n>1
seed and read no changes at any cell or count.

The probe economics held the campaign to budget.  Pre-tuning spend
through mg4 is about 7 GPU-hours against the plan's 20-to-34
projection for the same jobs, mostly because warm repeats came in far
under the pessimistic per-arm estimates.

Every mg script, both sbatch chains, and this page are staged in
`mbirjax_plans`; raw rows stay on scratch per convention.
