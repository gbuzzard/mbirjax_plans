# Multi-GPU performance: the n=1/2/4 readout and tuning — plan

**Status:** EXECUTED through increment 1 and RULED.  The plan passed
its review STOP (three-reviewer panel 2026-08-08, Fable revision),
mg1–mg4 ran under it, and the increment-1 checkpoint was endorsed on
2026-08-09 with the floor_4 amendment.  mg3a/b, mg4b, mg5, and mg6
are in the queue.  `multigpu_findings.md` is the live record; this
plan remains the contract for terms, protocols, and triggers.

The charter is `current_plans.md` item 3, and it sets four goals: the
full n=1/2/4 gate readout with the repaired kernels; the attribution
and tuning of any gap; the deferred cross-framework value comparison;
and the forward-attribution number that gates item 13.  The campaign
also owes two decisions.  The first is a work-size guard for the
automatic widening path.  The second is the cadence of the nightly's
n>1 rows.

Five records are prior art.  The n=1 composed baselines and the
arm-check discipline are in `kernel_batching_findings.md`.  The
widening rule, the ledger, and the closing rulings are in
`device_policy_design.md`, with the measurements in
`device_policy_findings.md`.  The forward-kernel repair and the
standing kernel-times-sharding gate are in
`kernel_sharding_findings.md`.  The shared-sinogram value protocol is
in `phase5_findings.md`.  The pre-kernel multi-device measurements are
in `phase4_findings.md`.

**Terms.**  A CELL is one (geometry, sinogram size) coordinate.  The
GATE CELLS are parallel and cone at (512, 448, 384) and
(1024, 1008, 992).  A COUNT is a device count n.  An ARM is one
subprocess-isolated measured run at one (cell, count, configuration).
The READOUT is the mg1 arm matrix.  The LADDER is the mg4 size sweep.
The GUARD is the proposed rule that bounds the automatic device count
by problem size.  A FLOOR is the work size below which the guard does
not admit a given count; the guard carries one floor per count.  The
LEDGER and the PREFLIGHT keep their `device_policy_design.md`
meanings.  A PARITY FLOOR is the measured float-agreement floor of a
comparison, below which two runs cannot be told apart.  The ENGINE
FLOOR is a cell's eager n>1-versus-n=1 parity floor.  The
COMPILE-LATITUDE TERM is the divergence a production torch run carries
over an eager run of the same arm.  The PARTITION-ORDER TERM is the
divergence both frameworks share from partition arithmetic.  Script
prefixes name their campaigns: kb is kernel batching, dp is device
policy, ks is kernel sharding, p4 is phase 4, and mg is this campaign.

---

## 1. What this campaign measures, and why now

Every existing n>1 performance number predates the machine it would
now describe.  Since the phase-4 matrix was recorded, four changes
landed on the multi-device path:

- the Triton kernels became default in both directions at every
  placement;
- the launch-context repair made the forward kernels correct under
  sharding;
- kernel-aware view batching replaced the old per-view rule;
- two residency fixes cut the composed peaks 4.9 to 14.6 percent
  across the four gate cells, 11 to 15 at three of them.

The all-device default then shipped, so an unpinned multi-GPU run now
widens automatically.  Multi-GPU is therefore the out-of-box
experience, and its performance has never been measured in this
configuration.

Three recorded priors are stale, and the campaign re-measures rather
than assumes them.  One prior says four devices run the 512 cell about
2.7x slower than one device.  That reading is the phase-4 matrix,
which predates the kernels and the batching repair.  A second prior
says the n>1 back limiter is the band transpose.  That reading is a
July mbirjax profiling result, taken on a different code base.  A
third prior is the +16 percent one-band residual at n=4 on parallel
1024, which was flagged to the kernel re-baseline.  The kernels and
their batching have both changed since that flag, so it needs a fresh
reading.

This is a measurement campaign.  The readout comes first.  Attribution
comes from its arms, and tuning happens only where the data indicates
a specific lever.  The campaign ends with two recommendations for
review rather than with unilateral changes.

---

## 2. What is already measured

**The n=1 baselines are current and reproduce.**  The composed
five-arm gate re-ran clean after the kernel repair (kb3, job
14975410).  These numbers are the campaign's n=1 column, and the
readout must reproduce them within the measured spread before its n>1
columns mean anything.

| cell | torch/jax warm time | torch peak |
|---|---|---|
| parallel 512 | 1.13 | 1.93 GB |
| parallel 1024 | 1.55 | 23.22 GB |
| cone 512 | 0.87 | 2.15 GB |
| cone 1024 | 0.99 | 23.68 GB |

**The n>1 priors are pre-kernel, and parallel-only.**  The last timed
torch matrix is phase 4a (warm 3-iteration vcd; stage 2, with stage
3's one-band re-runs at n=4).  The jax columns are the original
phase-4 matrix; the stage-2 re-run reproduced them at 0.93 to 1.00x.

| parallel cell | frame | n=1 | n=2 | n=4 | n=4, stage 3 |
|---|---|---|---|---|---|
| 512x448x384 | torch | 3.15 s | 2.58 s | 8.59 s | ≈8.8 s |
| 512x448x384 | jax | 1.76 s | 2.09 s | 3.13 s | — |
| 1024x1008x992 | torch | 94.0 s | 69.9 s | 61.9 s | 71.7 s |
| 1024x1008x992 | jax | 26.0 s | 14.5 s | 11.7 s | — |

That matrix came from its own harness, so its absolute times are not
comparable to the kb3 numbers.  Its usable prior is the scaling shape
across counts.  Three readings of that shape matter here.  At the 512
cell n=2 beat n=1 by 18 percent while n=4 collapsed, so the best count
is size-dependent and is not always 1 or all.  The n=4 collapse was
attributed to the per-subset band fan-out, a driver term the 1-device
diagnostic prices at about 2x; the kernels changed the per-call costs
around that term, not the fan-out count, so the new balance is
unknown.  Cone has never had a timed n>1 row at all, and its
multi-device engine is correctness-gated only.  The nightly's reduced
n>1 rows begin seeding concurrently under item 4; they protect this
campaign's changes but do not replace the readout.

**The engine floors are current.**  The flip gate (dp4, job 14973466)
read the engine floors at n=2 and n=4 against n=1: 4.47e-04 and
9.49e-04 in parallel, 4.57e-04 and 4.95e-04 in cone.  The probe cell
is the (256, 64, 64) cell the ks1 isolation matrix also used, and the
two agree to three significant figures.  The shared-sinogram
cross-framework residual at parallel 1024 is 6.1e-3 at three
iterations, decaying to 8.8e-4 by ten.  The dup2 control showed the
banded parity floor is a property of the partition arithmetic, not of
the devices; that floor is this plan's partition-order term.

**The memory model is calibrated at n=1 only.**  The ledger envelops
at 1.001 to 1.104 across the five calibration cells.  Every one of
those cells ran at one device.  The preflight consumes the ledger at
n>1, where its accuracy has never been measured.  The ledger also
states the term the seam hooks exist for.  The band reduce holds n
partials plus the running total on the owner device.  That term comes
to about 1.5 x cyl(P, S_pad) for n = 2 to 4, flat in the count, so
more devices do not shrink it.  `back_project_slice_band` is the one
lever that reduces it.  The lever's pre-kernel time cost was +8 to
+66 percent depending on the cell, and the cheapest cell (+8 percent,
n=4 at 512) was also the largest peak saving (6.6 to 2.6 GiB).

**The expected knee region is recorded.**  `lessons.md` §6 records
that vcd has a problem-size floor near 256-class cells: per-subset
work is num_subsets times smaller than a bare projection, and the
per-subset host scalar syncs are the GPU-specific limiter.  Three of
the six ladder cells sit at or below that size, which is where the
ladder should find the knee.

**The orchestration levers are recorded.**  The 1-device sharded
diagnostic priced the fan-out and eager glue at about 2x on a small
cell.  Two levers are recorded: the per-device compiled mega-region,
and CUDA graphs.  The mega-region is deferred for a cone-seam parity
risk, and CUDA graphs are the escalation.  The on-device scalar
combine and pool reuse have already shipped.

---

## 3. The protocols that bind every arm

Each rule below is load-bearing, and each was earned by a recorded
failure or ruling.

1. **Every pinned arm pins through `MBIRTORCH_NUM_DEVICES` and
   asserts the realized list.**  The env pin and an explicit
   `configure_devices` call are not equivalent: the env pin keeps the
   model on the automatic branch, where the preflight still runs,
   while an explicit call takes the explicit branch and gets no
   preflight.  One mechanism therefore serves every pinned arm, so
   arms are comparable, and the campaign uses the env pin.  After the
   timed call the arm reads `model.sino_placement.devices` and raises
   unless the length equals the intended n.
2. **An arm that intends the plain torch engine sets
   `MBIRTORCH_DISABLE_TRITON=1`.**  `compile_mode` does not disable
   the kernels.  Selection is availability-driven.
3. **Arm checks verify what was BOUND.**  Kernel arms assert the
   Triton bodies bound in both directions, through the selection
   witnesses the kb3 gate already carries.  Torch arms assert the
   kernels absent.  Every arm records the realized view batch per
   direction per device, because the batch budget divides by the
   count while the kernel cost models do not all follow, so the
   realized batch is not invariant in n.
4. **n>1 value gates run eager to eager.**  Compiled n>1 comparisons
   measure the compiler's latitude, per the compile-latitude policy.
   Production-configuration values are read against the documented
   amplified envelope, not against the eager floor.
5. **Cross-framework value arms hand ONE shared sinogram artifact to
   both frameworks.**  Per-framework phantoms differ at boundary
   ties, and the jax phantom additionally differs across platforms.
   The artifact is generated once by the harness, staged beside the
   scripts, and md5-verified like any other staged file; at the 1024
   cells it is a multi-GB array, and a corrupt read is a recorded
   Lustre failure mode.
6. **Memory is re-measured per arm in a fresh subprocess, never
   inferred.**  The peak is `torch.cuda.max_memory_allocated(d)` read
   per device, reported as the full per-device list and its maximum.
   `MBIRTORCH_MEMORY_CALIBRATION` is asserted absent except in mg2.
7. **Warm and cold are labeled, and cold is discarded from warm
   statistics.**  Warm compiled runs are bitwise reproducible, and a
   cold-versus-warm diff measures the compile.  Triton compiles per
   device, so an n-device cold pass pays n compiles per shape.  Each
   arm records its cold-pass wall and its total subprocess wall
   anyway, because the §6 cadence decision and the §7 estimates need
   exactly the costs the warm protocol discards.
8. **Per-cell tolerances calibrate against the same cell's
   dividing-case parity floor.**  The parity floor grows with
   in-plane coupling and shrinking data constraint, so no tolerance
   is copied across cells.
9. **Every timing arm runs repeats, and every decision rule reads
   against the measured spread.**  Each arm runs one discarded cold
   pass and at least three warm repeats, reporting the median and the
   spread.  The composed within-job run-to-run spread has never been
   measured, and three of this plan's rules are stated in terms of
   noise, so the spread is itself a deliverable.  Counts run
   blocked-and-reversed at each cell (1, 2, 4, 4, 2, 1 across arms)
   so thermal drift shows up as a within-pair spread rather than a
   bias on the scaling numbers.
10. **Instrument events record from the loop thread, in the device
    context.**  The engine calls the projection funnels from the recon
    loop's thread, so the instrument wraps the model methods there and
    never records an event from a worker thread — a worker thread's
    current device is 0, and an event recorded there measures nothing
    (the ks1 launch-context mechanism arriving through the
    instrument).  For each device in the placement, the start and end
    events are created and recorded inside
    `with torch.cuda.device(dev)` on that device's default stream,
    which is where the repaired drivers place all work.  Elapsed times
    are read only after a per-device synchronize at run end, never
    inside the loop, so the instrument cannot serialize the overlap
    it measures.
11. **Cluster discipline governs staging and scheduling.**  The
    environment passes through the submission shell, never
    `--export`.  Every staged file is scp'd per file and
    md5-verified.  No sync happens into a tree a running job imports
    from, and jobs chain with `--dependency` when they share the env
    or the tree.  Each row records the GPU health sample, and the throttle
    rule follows the nightly's finding (`nightly_plan.md` §10.5): the
    H100 flags heavy torch kernels at `sw_power_cap` and normal
    temperature routinely, which is the boost governor, so that flag
    is recorded and kept; a row flagged with high temperature and a
    depressed clock is thermal and is re-run.  Any comparison that crosses jobs
    pins the node or carries a shared anchor cell, per the h010
    precedent.  Rows write incrementally to jsonl, and each job runs
    its n=1 reproduction arms first, so a truncated job still yields
    its validity check.  Walltimes are requested at twice the upper
    wall estimate, and submissions are timed (or `--begin`-delayed)
    to clear the 02:00 jax nightly and the 03:00 torch nightly, each
    a four-GPU block and the torch one up to four hours on a
    changed-tip night.

---

## 4. The instruments

### mg1 — the gate readout (goals 1 and 4)

The readout is one sbatch job on four H100s that measures the full
matrix: both frameworks, both geometries, both gate cells, n = 1, 2,
4, warm seeded 3-iteration vcd, one subprocess per arm.  The torch
arms run the production configuration, which is kernels on in both
directions.  The jax arms run in the same job, so every ratio is
same-run and same-node.  The n=1 arms double as the reproduction
check against the recorded baselines.

Four auto arms ride along, one per (geometry, cell).  An auto arm
sets no pin, records the policy's chosen count, and records the
per-candidate rejection reasons the selection loop already logs at
verbose 2.  The expectation is the pre-guard policy's choice, which
is all visible devices at every gate cell; a different choice is
captured as a finding, with the rejection log attached, rather than
crashing the arm.  Two arm checks keep the observation honest:
`MBIRTORCH_NUM_DEVICES` is absent from the arm's environment, and
`configure_devices` is never called, because under the one-bit rule
any call is explicit and would silently disable the automatic path
being observed.  The auto arms also serve protocol 9: each is a
repeat of the pinned all-device arm at its cell, so the pair prices
the within-job spread.

The attribution instrument answers goal 4 and arms goal 2.
Instrumented twin arms run at both geometries and both gate cells for
each count, twelve in all.  Each twin brackets the engine's own calls
to the projection FUNNELS — `sparse_forward_project` and
`sparse_back_project` on the model — plus the prior-and-halo region,
under protocol 10.  The funnel is the right site because of the
call-structure change landed 2026-08-08: every engine projection now
routes through the public pair, which is also exactly what the
nightly's `forward` and `back` ops measure, so the share and the
dashboard rows describe one surface.  The site is also
structure-proof: whatever a future driver does below the funnel stays
inside the bracket, where a deeper site could be silently bypassed by
the next refactor.  One supplementary bracket sits below the funnel,
around `sum_band_to_owner`, because that reduce is itself the named
mg5 seam lever and the back funnel's span would otherwise hide it.

Each region reports two readings.  The host wall of the call gives
the region's span in the serial subset loop.  The per-device event
spans give the device-side window: the end event is recorded after
the call returns and queues behind everything the call enqueued, so
async spillover past the return is covered.  The wall-minus-span gap
prices orchestration, which is what the mg5 orchestration trigger
reads.  Item 13's entry gate reads the forward region's share of
composed wall at each count, in the composed regime, which
`phase5_findings.md` showed cannot be extrapolated from isolated
benches.  Three checks bound the instrument itself.  Each twin runs
adjacent to its uninstrumented sibling and must match its wall within
the protocol-9 spread.  The region sums plus the unbracketed
remainder must reconcile with the composed wall, which also guards
against any future call path that skips the funnel.  And the n=1
forward reading at parallel 1024 must be commensurate with the kb3
close-out, which attributed the 14.4 s composed remainder over jax to
the forward; a forward region far from that class says the brackets
sit in the wrong place.

The readout's value columns are two, and they are gated differently.
The within-framework column diffs each production arm against its own
framework's n=1 at the same cell in the same job; torch's divergence
is gated against jax's at the same coordinates, the ruler the phase-4
matrix validated when torch matched jax to three digits at 512.  The
cross-framework production column uses the shared sinogram artifact
per protocol 5 and is reported, not gated, until mg3 supplies the
eager floors that make it interpretable.

The readout's deliverable is the gate table: warm medians and
spreads, per-device peaks, ratios against jax at equal n, scaling
against own n=1, both value columns, the realized view batches, the
auto-arm findings, and the three-region attribution at every (cell,
count).

### mg2 — the ledger at n>1

The widening rule trusts the ledger at exactly the counts where the
ledger is unmeasured.  mg2 closes that gap with the dp2 calibration
harness extended to counts: modeled versus measured per-device peaks
at both gate cells, both geometries, n = 2 and 4.  Eight weighted
arms carry the gate cells, and two unweighted arms at parallel 1024
carry the configuration whose dominant phase differs and whose
`hess_weights` sharding has never been measured.  The acceptance band
is the standing 1.00 to 1.30 per device.  A cell below the band's
lower bound is fixed by adding the missing term, never by a factor,
per the checkpoint-2 rule.  The dp3 phase-resolved probe is staged
alongside, because the n=1 calibration needed it to attribute every
sub-band reading, and a sub-band reading at n>1 without it would end
the increment with a gap instead of a term.  mg2 chains after mg1 on
the same allocation class and the same staged tree.

### mg3 — the value comparison (goal 3)

The comparison separates the residual terms by construction.  At each
(geometry, cell, count), three arms reconstruct from one shared
sinogram artifact: jax production, torch eager-plain, and torch
production.  Eager-plain means kernels off and compile off.  Three
columns then carry the reading.  Torch-eager versus jax-production
reads the partition-order term plus both frameworks' remaining
latitude.  Torch-production versus torch-eager reads the
compile-latitude term; kb4 bounded the kernels' own contribution at
the e-6 parity class against eager bodies, so this bundle reads the
compiler to leading order, and one optional compiled-kernels-off arm
tier at parallel 1024 verifies that ordering at n>1.  The
within-framework columns cost nothing, because n=1 is already in the
count set: jax-n versus its own n=1, and torch-eager-n versus its own
n=1, are the clean partition-order readings, and the second should
land on the recorded engine floors.  One caveat is recorded rather
than solved: jax is always jitted, so its within-framework diff
carries XLA's own per-shape latitude and reads as a bound.

Depth settles what tolerance cannot.  All arms run at three
iterations.  Parallel 1024 adds a ten-iteration tier, because that
cell carries the documented 6.1e-3 three-iteration residual and its
sevenfold decay.  Two expectations are registered in advance.  The
partition-order columns should track the engine floors, in the 5e-4
class, growing mildly with n.  The compile-latitude column should sit
where the composed kernel-versus-body columns sit at n=1 — low e-3 in
parallel, e-4 in cone — and decay with depth; the decay at n>1 is a
hypothesis this instrument tests, not a recorded fact.  A term that
instead grows with n is a finding, and it would redirect the tuning.

mg3's allocation is committed only after a pricing probe, because 15
of its arms are eager-plain and no eager-plain composed time has ever
been measured at the 1024 cells.  The probe runs one eager-plain arm
at each 1024 cell at n=1, about ten minutes, and the mg3 wall is
re-derived from it.

mg3 also carries a coverage role the goldens ruling assigned it.  The
golden archives are now opt-in everywhere and skip in the nightly's
fresh clones (`nightly_plan.md` §10.4), so the port-fidelity question
is asked at development time, at releases, and by exactly this
instrument.  mg3's cross-framework columns at n = 1, 2, 4 are the
campaign's half of that coverage.

### mg4 — the crossover ladder (decision 1's data)

The ladder measures where each count stops paying.  Parallel-family
cells scale the 512 gate cell's proportions exactly: (128, 112, 96),
(192, 168, 144), (256, 224, 192), (384, 336, 288), (512, 448, 384),
and (768, 672, 576).  Each cell runs n = 1, 2, 4 pinned plus one auto
arm, and the auto arm doubles as the protocol-9 repeat.  Views and
slices divide by 4 at every ladder cell, so no padding effect
confounds the knee.  The (1024, 1008, 992) gate cell is re-measured
inside mg4 as an off-family size anchor, tying the ladder to mg1's
matrix without a cross-job comparison.  Cone runs a three-size spot
check bracketing the parallel knee.  All arms are warm seeded
3-iteration vcd, the same protocol as the readout, because the
guard's subject is the out-of-box `recon()`.

The ladder's deliverable is the measured crossover per count: the
smallest cell at which n=2 beats n=1, the smallest at which n=4 beats
n=2, and the speedup curves around both.  The knee rule is stated
against noise in advance: a count is admitted at a size only if it
wins by more than the protocol-9 spread there and at every larger
ladder size, so one noisy cell cannot move a floor.  The auto arms
record what the pre-guard policy chooses at every size, which is
where today's harm lands.

One thin-volume probe point rides along, sized to discriminate: its
shape is chosen so the candidate work-size metrics disagree about the
decision (large sinogram, small recon), and the measurement records
which metric the outcome vindicated.  §8 carries the metric question
this probe informs.

Two off-family probe classes were added by Greg's direction
(2026-08-09), covering the shapes where the guard could be wrong
rather than merely conservative.  The thin-volume class proper — few
slices, many views, (1024, 32, 768) — is the regime where a
sinogram-elements floor risks holding n=1 where widening pays.  The
sparse-view class — the 512-cell volume at 64 views — is the opposite
stress: a small sinogram over a large recon.  Each runs the standard
four arms as an addendum job (mg4b), and every other shape class the
campaign does not measure is recorded as out of scope in the findings
page, with the ledger carrying memory prediction there and the
guard's conservatism as the stated time policy.

### mg5 — tuning arms, only where the data points

Tuning is contingent on attribution, so mg5 is a menu with triggers.
The mg1 three-region attribution and the realized-batch column exist
precisely so every trigger below is decidable at the checkpoint.

| lever | trigger in the data | arm design |
|---|---|---|
| per-device view chunk | realized batches shift at n>1, or a projector region is off-model | kb2-style sweep over chunk {64, 128, 256} at the affected cell and counts; the constants were pinned by an n=1 sweep only |
| seam: band-reduce restructure | the reduce region visible in the n>1 attribution, or mg2 shows the band-reduce term binding | redesign `sum_band_to_owner` accumulation order; any change gates on the standing 2-GPU kernel gate |
| streaming: `back_project_slice_band` | an n>1 cell near per-device capacity, or a memory escape needed below a floor | re-price the band-streaming sweep with kernels on; the +8 to +66 percent costs are pre-kernel |
| widening margin (0.15) | mg2 ratios far from 1 at n>1 | recompute the margin from the measured n>1 envelope, as a reviewed knob change |
| orchestration (mega-region, CUDA graphs) | small-cell n>1 collapse persists with kernels on | separate charter; the cone-seam parity risk stands, so this is a recommendation, not an mg5 arm |

Each triggered lever gets its own single-variable arm set and its own
gate against the readout's numbers, and it lands only through review.

---

## 5. Increments and their gates

**Increment 0 — staging.**  Write mg1 and mg2, and sync them per file
with md5 verification.  The conventions are standing ones: scripts in
`plans/experiments/torch_port/` under the mg prefix, runs from the
`torch_p3` scratch checkout under `TORCHPY`, rows to the scratch
`results/` area.  Submissions wait until the torch nightly's trial
and first night are done, because those jobs share the queue and the
scratch conventions.  The scripts follow the pinned-arm pattern.  The
phase-4-era scripts are prior art, but they were audited for exactly
the unpinned-arm hazard, so the mg scripts assert their counts rather
than inherit the pattern on trust.  Before any mg harness is written,
the item-4 session's `nt2_local_shard_check.py` runs locally, about a
minute on CPU virtual devices.  That check reproduces the sharded
gather seam whose bug cost the nightly's first 4-GPU trial all 32 of
its n>1 rows: `Shards.gather()` already returns numpy, and
re-detaching its result is the recorded failure.  The mg harnesses
honor the same contract.

**Increment 1 — mg1 + mg2, then a CHECKPOINT.**  The readout and the
ledger check run as chained jobs.  The checkpoint delivers four
things: the gate table, the three-region attribution, the
ledger-at-n>1 verdict, and the mg1 validity record (arm checks, n=1
reproduction, spreads).  The mg5 triggers are then evaluated against
that data, and Fable rules on which tuning arms proceed.  This is the
fork in the campaign, because tuning before attribution would be
guessing.

**Increment 2 — mg3 and mg4.**  Both are measurement, not tuning, so
they do not wait on the checkpoint's ruling.  They do wait on mg1's
validity record, because this project's failure history is harness
flaws that invalidate whole matrices, and sequencing behind the
validity read costs nothing when the jobs chain anyway.  The mg3
pricing probe runs first and re-derives the mg3 wall before its
allocation is committed.  The thin-volume probe rides with mg4.

**Increment 3 — the ruled tuning arms, if any.**  Each lands with its
own gate re-run: the affected composed cells, and the standing 2-GPU
kernel gate when a kernel or driver file changes.

**Increment 4 — the two recommendations and the close-out.**  The
findings page (`multigpu_findings.md`) carries the tables.  The
recommendations go to review, and `current_plans.md` item 3 updates
at the close.  Any library change this campaign produces ships as an
ordinary reviewed commit, which the live nightly then attributes to a
moved tip.  The nightly is regression protection for this campaign,
not its instrument.  The campaign's numbers come from its own gated
harnesses.

---

## 6. The two decisions this campaign owes

### Decision 1: the widening speed guard

The automatic path is capacity-only today, so it widens small
problems onto counts that run slower.  The phase-4 prior says the
harm is real and also says its shape: at the 512 cell, n=4 ran about
2.7x slower than n=1 while n=2 beat n=1 by 18 percent.  A single
stay-at-n=1 threshold therefore gets that cell wrong on either side,
and the guard must be a per-count rule.  The proposed shape is one
reviewed knob family: a floor per count, below which the automatic
path does not admit that count.  The selection loop already walks
counts from largest to smallest, so the guard simply skips a count
whose floor is unmet, and the loop's existing fallback does the rest.
At a 512-class size the floors would exclude n=4, admit n=2, and the
loop would choose 2, which is what the prior measured as optimal.
Capacity always wins: a problem that fits no admitted count widens to
whatever count fits, exactly as today.

The guard's scope is the automatic branch alone.  It is consulted
only when no pin is present, per the design's "an explicit pin means
explicit" ruling.  Both pin mechanisms bypass it: an explicit
`configure_devices` call, and the `MBIRTORCH_NUM_DEVICES` pin —
which matters, because every pinned arm in this campaign and every
nightly n>1 row uses the env pin, and a guard that consulted floors
on the pinned branch would silently override them all.  The knob
family ships with measured defaults and an environment override, and
the verbose log states when a floor excluded a count.

The data picks the floors.  Each count's floor lands at the ladder's
admission point for that count under the §4 knee rule, with a margin
toward fewer devices.  The margin direction follows from the measured
asymmetry.  Widening a below-knee cell cost about 2.7x in the prior.
Holding a just-above-knee cell at a smaller count costs a few percent
of a seconds-scale run.  The floors' metric should be a quantity the
decision site already knows.  Sinogram elements is the candidate, the
thin-volume probe tests it, and §8 carries the open questions.  The
recommendation, with the curves behind it, goes to Fable before any
implementation.

### Decision 2: the nightly n>1 cadence

The nightly plan priced its n>1 increment at about 2.7 GPU-hours per
changed-branch night in its §3(c) estimate.  The nightly's own trial
then measured a full changed-branch pass at 0.26 GPU-hours, halving
the cost basis to about 1.4 GPU-hours per night, dominated by the
four-GPU allocation.  This campaign refines that basis but does not
own the final number.  mg1's per-arm cold and total subprocess walls
(protocol 7) price the vcd cells the n>1 rows would add, and the
close-out reports the refreshed per-night cost at both cadences.  The
authoritative number remains the n>1 increment's own trial run, which
the nightly plan already prescribes.  Greg decides; this campaign
prices the options.

---

## 7. Costs

The estimates below use the measured n=1 times, the pre-kernel n>1
ratios as the pessimistic bound, and a fixed subprocess cost of 2 to
5 minutes per arm.  The fixed cost covers import, CUDA init, model
build, input generation, and compile, and protocol 7's per-device
Triton compile multiplies the compile share at n>1.  Protocol 9's
repeats add warm time only, which the fixed cost dominates at every
cell below 1024.

| job | arms | wall estimate | allocation | GPU-hours |
|---|---|---|---|---|
| mg1 readout | 24 matrix + 4 auto + 12 instrumented | 2.5–4 h | 4 GPUs | 10–16 |
| mg2 ledger | 10 calibrated arms | 1–1.5 h | 4 GPUs | 4–6 |
| mg3 value | probe, then 45 arms + optional 3 | 3–6 h | 4 GPUs | 12–24 |
| mg4 ladder | 28 parallel + 12 cone + 1 probe | 1.5–3 h | 4 GPUs | 6–12 |
| mg5 tuning | per ruled lever | priced when chartered | 2–4 GPUs | — |

The pre-tuning total is about 32 to 58 GPU-hours across four chained
jobs, at walltimes requested at twice the upper estimates.  The mg3
row is the widest spread and the least trusted, because its
eager-plain arm class has never been priced at the 1024 cells; its
pricing probe exists to replace that spread with a measurement before
the allocation is committed.  The scheduling rule of protocol 11
keeps every job clear of the 02:00 and 03:00 nightlies.

---

## 8. Open questions for the review

**The guard's work-size metric.**  Sinogram elements is the simple
candidate, known at the decision site.  The ladder scales all three
axes together, so it cannot distinguish sinogram elements from recon
voxels or from their product; the thin-volume probe is designed to,
because its shape makes the candidates disagree.  A second candidate
axis is recorded for the review rather than swept: the crossover is
plausibly governed by orchestration count (iterations times subsets
times bands) against per-call work, and the partition granularity
sequence is not a function of size alone, so two cells with equal
sinogram elements can sit on opposite sides of a knee.  The
recommendation is to accept a simple metric with these caveats
recorded, and let the probe say which simple metric.

**Whether the guard needs a count-preference rule above the floors.**
The current loop prefers the largest fitting count.  The ladder will
say whether that preference is right in the mid sizes, where n=2 can
beat both neighbors.  The per-count floors encode that reversal
naturally if the floors are monotone in n; the open question is
whether the data shows a regime the floor family cannot express, and
the answer goes to Fable with the curves.  CLOSED at the increment-1
ruling (2026-08-09): the ladder's mid-size reversal — n=2 over n=4 at
the 768 cell, 1.60x to 1.18x — is expressed by monotone floors once
each count's floor is defined by its crossover against the best
smaller ADMITTED count (the floor_4 amendment).  No preference rule
is needed.

**Whether mg3 needs cone depth-10 arms.**  The documented decay cell
is parallel 1024, and §4 scopes the depth-10 tier to it.  Cone's
three-iteration residuals are already inside the envelope at n=1.
The recommendation is to keep the tier as scoped and add cone depth
only if its depth-3 readings surprise.

**Whether the readout should carry back-only arms.**  The kb3
five-arm structure prices the forward and back kernels separately at
n=1.  At n>1 the same differencing would cost 12 more arms.  The
recommendation is no: the three-region instrument answers goals 2 and
4 directly, and the arm differencing joins mg5 only if a specific gap
needs it.

**The +16 percent n=4 residual.**  The stage-2 code state no longer
exists, so the residual cannot be reproduced as a diff.  Its tell
can: the flag closes if mg1 shows n=4 at parallel 1024 scaling
consistently with n=2 at the same cell, and becomes an mg5 trigger if
the anomaly pattern (n=4 off while n=2 is clean) reappears in the
current tree.

**Coordination with the nightly's first nights.**  The campaign's
jobs and the torch nightly share the account, the queue, and the
scratch conventions.  The plan sequences increment 0 after the first
real nightly run lands, keeps campaign syncs out of `mbirtorch_src`
while any nightly or trial job is live, and schedules per protocol
11.  One sequencing fact is recorded so the early rows read
correctly: Greg directed the nightly's n>1 increment to proceed
without its three-night n=1 soak, precisely so a multi-device
baseline is on the board for this campaign, so n>1 rows appear after
one night by direction rather than by process slip.
