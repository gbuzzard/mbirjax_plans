# Multi-GPU performance: the n=1/2/4 readout and tuning — plan

**Status:** DRAFT, awaiting Fable review at the plan STOP.  No cluster
job has been submitted for this campaign.

The charter is `current_plans.md` item 3.  Its four goals are the full
n=1/2/4 gate readout with the repaired kernels, the attribution and
tuning of any gaps, the deferred cross-framework value comparison, and
the forward-attribution arm that gates item 13.  The campaign also owes
two decisions: a work-size floor for the automatic widening path, and
the cadence of the nightly's n>1 rows.

Five records are prior art.  The n=1 composed baselines and the arm-check
discipline are in `kernel_batching_findings.md`.  The widening rule, the
ledger, and the closing rulings are in `device_policy_design.md`, with the
measurements in `device_policy_findings.md`.  The forward-kernel repair
and the standing kernel-times-sharding gate are in
`kernel_sharding_findings.md`.  The shared-sinogram value protocol is in
`phase5_findings.md`.  The pre-kernel multi-device measurements are in
`phase4_findings.md`.

**Terms.**  A CELL is one (geometry, sinogram size) coordinate; the GATE
CELLS are parallel and cone at (512, 448, 384) and (1024, 1008, 992).  A
COUNT is a device count n.  An ARM is one subprocess-isolated measured
run at one (cell, count, configuration).  The READOUT is the mg1 arm
matrix.  The LADDER is the mg4 size sweep.  The GUARD is the proposed
work-size floor on automatic widening, and the FLOOR is its threshold
value.  The LEDGER and the PREFLIGHT keep their `device_policy_design.md`
meanings.  The ENGINE FLOOR is the eager n>1-versus-n=1 float divergence
of a cell.  The COMPILE-LATITUDE TERM is the extra divergence a compiled
torch run carries.  The PARTITION-ORDER TERM is the divergence both
frameworks share from partition arithmetic.

---

## 1. What this campaign measures, and why now

Every existing n>1 performance number predates the machine it would now
describe.  Since the phase-4 matrix was recorded, four changes landed on
the multi-device path: the Triton kernels became default in both
directions at every placement, the launch-context repair made the
forward kernels correct under sharding, kernel-aware view batching
replaced the old per-view rule, and two residency fixes cut the composed
peaks 11 to 15 percent.  The all-device default then shipped, so an
unpinned multi-GPU run now widens automatically.  Multi-GPU is therefore
the out-of-box experience, and its performance has never been measured
in this configuration.

Three recorded priors are stale, and the campaign re-measures rather
than assumes them.  The old reading that four devices run the 512 cell
about 3x slower than one device predates the kernels and the batching
repair.  The reading that the n>1 back limiter is the band transpose is
a June mbirjax profiling result, taken on a different code base.  The
open +16 percent residual at n=4 on parallel 1024 was flagged to the
kernel re-baseline.  The kernels and their batching have both changed
since that flag, so it needs a fresh reading.

This is a measurement campaign.  The readout comes first, attribution
comes from its arms, and tuning happens only where the data indicates a
specific lever.  The campaign ends with two recommendations for review
rather than with unilateral changes.

---

## 2. What is already measured

**The n=1 baselines are current and reproduce.**  The composed five-arm
gate re-ran clean after the kernel repair (job 14975410).  The
torch-over-jax warm time ratios are 1.13 (parallel 512), 1.55 (parallel
1024), 0.87 (cone 512), and 0.99 (cone 1024).  The kernel-arm peaks are
1.93, 23.22, 2.15, and 23.68 GB.  These numbers are the campaign's n=1
column, and the readout must reproduce them within noise before its n>1
columns mean anything.

**The n>1 priors are pre-kernel, and parallel-only.**  The last timed
matrix is phase 4a stage 2/3 (warm 3-iteration vcd, torch): at the 512
cell 3.15, 2.58, and 8.59 s for n = 1, 2, 4; at the 1024 cell 94.0,
69.9, and 61.9 s.  The same runs put jax at 1.76, 2.09, 3.13 s and 26.0,
14.5, 11.7 s.  That matrix came from its own harness, so its absolute
times are not comparable to the kb3 numbers; its usable prior is the
scaling shape across counts.  Three facts stand out.  The kernels and
their batching have since bought 2.6x to 3.9x composed over the torch
bodies at the four gate cells, so the n>1 balance is unknown.  The n=4
collapse at the 512 cell was the per-subset band loop growing as bands
times devices, which is exactly the term the kernels changed.  Cone has
never had a timed n>1 row at all.  Its multi-device engine is
correctness-gated only.

**The value floors are current.**  The flip gate (job 14973466) read the
engine floors at n=2 and n=4 against n=1: 4.47e-04 and 9.49e-04 in
parallel, 4.57e-04 and 4.95e-04 in cone, at the dp5 cell.  The
shared-sinogram cross-framework residual at parallel 1024 is 6.1e-3 at
three iterations, decaying to 8.8e-4 by ten.  The dup2 control showed
the banded float floor is a property of the partition arithmetic, not of
the devices.

**The memory model is calibrated at n=1 only.**  The ledger envelops at
1.001 to 1.104 across the five calibration cells, all at one device.
The preflight consumes the ledger at n>1, where its accuracy has never
been measured.  The ledger also states the term the seam hooks exist
for.  The band reduce holds n partials plus the running total on the
owner device, which comes to about 1.5 x cyl(P, S_pad) for n = 2 to 4.
That term is flat in the count, so more devices do not shrink it.
`back_project_slice_band` is the one lever that reduces it, and the
lever's pre-kernel time cost was +24 to +66 percent.

**The orchestration levers are recorded.**  The 1-device sharded
diagnostic priced the fan-out and eager glue at about 2x on a small
cell.  The recorded levers are the per-device compiled mega-region
(deferred for a cone-seam parity risk) and CUDA graphs (the
escalation).  The on-device scalar combine and pool reuse have already
shipped.

---

## 3. The protocols that bind every arm

Each rule below is load-bearing, and each was earned by a recorded
failure or ruling.

1. **Every arm pins its count and asserts the realized list.**  The arm
   sets `MBIRTORCH_NUM_DEVICES=<n>` in its subprocess environment or
   calls `configure_devices(num_devices=n)` before any model use.  After
   the timed call it reads `model.sino_placement.devices` and raises
   unless the length equals the intended n.  The automatic default is
   live, so an unpinned arm measures whatever the policy chooses.
2. **An arm that intends the plain torch engine sets
   `MBIRTORCH_DISABLE_TRITON=1`.**  `compile_mode` does not disable the
   kernels; selection is availability-driven.
3. **Arm checks verify what was BOUND.**  Kernel arms assert the Triton
   bodies bound in both directions, through the selection witnesses the
   kb3 gate already carries.  Torch arms assert the kernels absent.
4. **n>1 value gates run eager to eager.**  Compiled n>1 comparisons
   measure the compiler's latitude, per the compile-latitude policy.
   Production-configuration values are read against the documented
   amplified envelope, not against the eager floor.
5. **Cross-framework value arms hand ONE shared sinogram artifact to
   both frameworks.**  Per-framework phantoms differ at boundary ties,
   and the jax phantom additionally differs across platforms.
6. **Memory is re-measured per arm in a fresh subprocess, never
   inferred.**  The peak is `torch.cuda.max_memory_allocated(d)` read
   per device, reported as the full per-device list and its maximum.
   `MBIRTORCH_MEMORY_CALIBRATION` is asserted absent except in mg2.
7. **Warm and cold are labeled, and cold is discarded.**  Warm compiled
   runs are bitwise reproducible, and a cold-versus-warm diff measures
   the compile.  Triton compiles per device, so an n-device cold pass
   pays n compiles per shape.  The warm protocol keeps that cost out of
   the measurements, and the §7 wall estimates put it back in.
8. **Per-cell tolerances calibrate against the same cell's dividing-case
   floor.**  The parity floor grows with in-plane coupling and shrinking
   data constraint, so no tolerance is copied across cells.
9. **Cluster discipline.**  Environment passes through the submission
   shell, never `--export`.  Every staged file is scp'd per file and
   md5-verified.  No sync into a tree a running job imports from; jobs
   chain with `--dependency` when they share the env or the tree.  Each
   row records the GPU health sample, and throttled rows are flagged.

---

## 4. The instruments

### mg1 — the gate readout (goals 1 and 4)

The readout is one sbatch job on four H100s that measures the full
matrix: both frameworks, both geometries, both gate cells, n = 1, 2, 4,
warm seeded 3-iteration vcd, one subprocess per arm.  The torch arms run
the production configuration, which is kernels on in both directions.
The jax arms run in the same job, so every ratio is same-run and
same-node.  The n=1 arms double as the reproduction check against the
recorded baselines.

Four AUTO arms ride along, one per (geometry, cell).  An auto arm sets
no pin, observes the policy's choice, and asserts it equals the visible
device count.  Its warm time must match the pinned arm at the same
count within noise.  This is the dp4 auto-versus-explicit check carried
onto the timing side, and it is what "the readout runs at the live
default" means operationally.

The forward-attribution instrument answers goal 4.  Instrumented twin
arms run at parallel 512 and 1024 for each count, with CUDA-event pairs
accumulated around every `sparse_forward_project` call on each shard
device's stream.  The instrument reports per-device forward sums and
the share of composed wall time, per device and as the device maximum.
Two checks bound the instrument: the twin's wall time must sit within 2
percent of its uninstrumented sibling, and the n=1 share must be
consistent with the kb3 arm differencing at the same cell.  The share
at n = 1, 2, 4 is item 13's entry gate number, and it also prices what
the guard can ignore.

The readout's deliverable is the gate table: warm times, per-device
peaks, ratios against jax at equal n, scaling against own n=1, value
columns at the production configuration, and the forward shares.

### mg2 — the ledger at n>1

The widening rule trusts the ledger at exactly the counts where the
ledger is unmeasured.  mg2 closes that gap with the dp2 calibration
harness extended to counts: modeled versus measured per-device peaks at
both gate cells, both geometries, n = 2 and 4, weighted arms.  The
acceptance band is the standing 1.00 to 1.30 per device.  A cell below
the floor is fixed by adding the missing term, never by a factor, per
the checkpoint-2 rule.  mg2 chains after mg1 in the same allocation
class and shares its staged tree.

### mg3 — the value comparison (goal 3)

The comparison separates the two residual terms by construction.  At
each (geometry, cell, count), three arms reconstruct from ONE shared
sinogram artifact: jax production, torch eager-plain (kernels off,
compile off), and torch production.  Torch-eager versus jax reads the
partition-order term plus the framework float floor.  Torch-production
versus torch-eager reads the compile-latitude term alone.
Torch-production versus jax reads their combination, which is the
number the composed gates already report.

Depth settles what tolerance cannot.  All arms run at three iterations;
parallel 1024 adds a ten-iteration tier, because that cell carries the
documented 6.1e-3 three-iteration residual and its sevenfold decay.
The expectation to test: the partition-order term tracks the engine
floors (about 5e-4 class, growing mildly with n), and the
compile-latitude term stays in its 5e-3 class at depth 3 and decays
with depth, at every count.  A term that instead GROWS with n is a
finding, and it would redirect the tuning.

### mg4 — the crossover ladder (decision 1's data)

The ladder measures where multi-device stops paying.  Parallel-family
cells scale the 512 gate cell's proportions: (128, 112, 96),
(192, 168, 144), (256, 224, 192), (384, 336, 288), (512, 448, 384), and
(768, 672, 576), each at n = 1, 2, 4 pinned plus one auto arm.  The
1024 point comes from mg1.  Views and slices divide by 4 at every
ladder cell, so no padding effect confounds the knee.  Cone runs a
three-size spot check bracketing the parallel knee.  All arms are warm
seeded 3-iteration vcd, the same protocol as the readout, because the
guard's subject is the out-of-box `recon()`.

The ladder's deliverable is the measured crossover: for each count, the
smallest cell at which n devices beat one device, and the speedup curve
around it.  The auto arms record what the policy would have chosen at
every size.  A widened choice at a size where n=1 wins is the harm the
guard exists to prevent, and the auto arms show exactly where that harm
lands today.

### mg5 — tuning arms, only where the data points

Tuning is contingent on attribution, so mg5 is a menu with triggers
rather than a schedule.

| lever | trigger in the data | arm design |
|---|---|---|
| per-device view chunk | projector phase off-model at n>1, or forward share anomalous vs n=1 | kb2-style sweep over chunk {64, 128, 256} at the affected cell and counts; the constants were pinned by an n=1 sweep only |
| seam: band-reduce restructure | back/assembly term visible in n>1 time attribution, or mg2 shows the band-reduce term binding | redesign `sum_band_to_owner` accumulation order (its findings would gate any change on the standing 2-GPU gate) |
| streaming: `back_project_slice_band` | an n>1 cell near per-device capacity, or the guard analysis needs a memory escape below the floor | re-price the band ladder with kernels on; the +24 to +66 percent cost numbers are pre-kernel |
| widening margin (0.15) | mg2 ratios far from 1 at n>1 | recompute the margin from measured n>1 envelope, as a reviewed knob change |
| orchestration (mega-region, CUDA graphs) | small-cell n>1 collapse persists with kernels on (ladder shape) | separate charter; the cone-seam parity risk stands, so this is a recommendation, not an mg5 arm |

Each triggered lever gets its own single-variable arm set, its own gate
against the readout's numbers, and lands only through review.

---

## 5. Increments and their gates

**Increment 0 — staging.**  Write mg1 and mg2, and sync them per file
with md5 verification.  The scripts live in
`plans/experiments/torch_port/` under the `mg` prefix, run from the
`torch_p3` scratch checkout under `TORCHPY`, and write rows to the
scratch `results/` area, all per the standing kb and dp conventions.
Submit nothing while the torch nightly's trial and first night are
live, because those jobs share the queue and the scratch conventions.
The scripts follow the pinned-arm pattern.  The phase-4-era scripts are
prior art, but they were audited for exactly the unpinned-arm hazard,
so the mg scripts assert their counts rather than inherit the pattern
on trust.

**Increment 1 — mg1 + mg2, then a CHECKPOINT.**  The readout and the
ledger check run as chained jobs.  The checkpoint delivers the gate
table, the forward shares, the ledger-at-n>1 verdict, and the
attribution of any gap, with the mg5 triggers evaluated against the
data.  Fable rules on which tuning arms proceed.  This is the fork in
the campaign, because tuning before attribution would be guessing.

**Increment 2 — mg3 and mg4, in either order or together.**  Both
depend only on the staged tree, not on increment 1's outcome.  Their
results feed the close-out and decision 1 respectively.

**Increment 3 — the ruled tuning arms, if any.**  Each lands with its
own gate re-run: the affected composed cells, and the standing 2-GPU
kernel gate when a kernel or driver file changes.

**Increment 4 — the two recommendations and the close-out.**  The
findings page (`multigpu_findings.md`) carries the tables.  The
recommendations go to review, and `current_plans.md` item 3 updates at
the close.  Any library change this campaign produces ships as an
ordinary reviewed commit, which the live nightly then attributes to a
moved tip.  The nightly is regression protection for this campaign,
not its instrument; the campaign's numbers come from its own gated
harnesses.

---

## 6. The two decisions this campaign owes

### Decision 1: the widening speed guard

The automatic path is capacity-only today, so it widens small problems
that run faster on one device.  The phase-4 prior says the harm was
real: the 512 cell ran about 3x slower at n=4 than at n=1.  The kernels
have since moved the crossover an unknown amount.  The proposed remedy
is one robust knob: a work-size FLOOR below which the automatic path
prefers n=1, provided n=1 fits the ledger.  Capacity always wins, so a
problem that does not fit one device widens regardless of the floor.

The mechanics are small by design.  The floor is consulted at the top
of the existing selection loop in the automatic path.  That loop
already runs at `vcd_recon` entry with the sinogram and recon shapes in
hand.  An explicit `configure_devices` call bypasses the floor, exactly
as it bypasses the rest of the automatic policy.  The knob ships with a
measured default and an environment override, and the verbose log
states when the floor held a run at n=1.

The data picks the knee.  The floor lands at the largest ladder cell
where n=1 still wins, with a margin toward staying at n=1.  The margin
direction follows from the measured asymmetry.  Widening below the knee
cost about 3x in the prior, while holding n=1 slightly above the knee
costs a few percent of a seconds-scale run.  The floor's metric should
be a quantity the decision site already knows.  Sinogram elements is
the candidate, and §8 raises the metric choice and the thin-volume
caveat for review.  The recommendation, with the curve behind it, goes
to Fable before any implementation.

### Decision 2: the nightly n>1 cadence

The approved nightly plan prices its n>1 increment at about 2.7
GPU-hours per changed-branch night, dominated by the four-GPU
allocation, and defers the nightly-versus-weekly call to measured
numbers.  The readout supplies them: per-arm warm and wall costs at n =
2 and 4 for exactly the cells the nightly would add.  The close-out
reports the refreshed per-night cost at both cadences.  Greg decides;
this campaign only prices the options.

---

## 7. Costs

The estimates below use the measured n=1 times, the pre-kernel n>1
ratios as the pessimistic bound, and 2 to 5 minutes per arm of fixed
subprocess cost (import, CUDA init, model build, inputs, compile).

| job | arms | wall estimate | allocation | GPU-hours |
|---|---|---|---|---|
| mg1 readout | 24 matrix + 4 auto + 6 instrumented | 2.0–2.5 h | 4 GPUs | 8–10 |
| mg2 ledger | 8 calibrated arms | 1 h | 4 GPUs | 4 |
| mg3 value | 36 depth-3 + 9 depth-10 arms | 2.5–4 h | 4 GPUs | 10–16 |
| mg4 ladder | 24 parallel + 12 cone arms | 1.5 h | 4 GPUs | 6 |
| mg5 tuning | per ruled lever | priced when chartered | 2–4 GPUs | — |

The pre-tuning total is about 28 to 36 GPU-hours across four jobs.  The
mg3 spread is the widest because its eager-plain arms have no compiled
speed and no kernel speed, and eager cost at the 1024 cells has never
been measured.  The jobs chain rather than overlap, per protocol 9, and
they yield to the 02:00 jax nightly and the new torch nightly's first
nights.

---

## 8. Open questions for the review

**The guard's work-size metric.**  Sinogram elements is the simple
candidate, known at the decision site.  The ladder scales all three
axes together, so it cannot distinguish sinogram elements from recon
voxels or from their product.  Off-family shapes can distinguish them.
A thin volume has many views and few slices, so its sinogram is large
while its recon is small, and the empty-shard extensions exist for
exactly those workloads.  The options are to accept the simple metric
with the caveat recorded, or to add off-family ladder points that test
it.  The recommendation is the middle: the caveat recorded, plus one
thin-volume probe point priced at a few minutes.

**Whether mg3 needs cone depth-10 arms.**  The documented decay cell is
parallel 1024, and §4 scopes the depth-10 tier to it.  Cone's
three-iteration residuals are already inside the envelope at n=1.  The
recommendation is to keep the tier as scoped and add cone depth only if
its depth-3 readings surprise.

**Whether the readout should carry back-only arms.**  The kb3 five-arm
structure prices the forward and back kernels separately at n=1.  At
n>1 the same differencing would cost 12 more arms.  The recommendation
is no: the forward-share instrument answers goal 4 directly, and the
arm differencing joins mg5 only if a specific gap needs it.

**The +16 percent n=4 residual.**  If the readout reproduces it with
kernels on, its attribution becomes an mg5 trigger of its own.  If it
does not reproduce, the flag closes as overtaken by the kernel
re-baseline.

**Coordination with the nightly's first nights.**  The campaign's jobs
and the torch nightly share the account, the queue, and the scratch
area's env conventions.  The plan sequences increment 0 after the first
real nightly run lands, and keeps campaign syncs out of
`mbirtorch_src` while any nightly or trial job is live.
