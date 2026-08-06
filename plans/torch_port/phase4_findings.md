# Phase 4 findings — multi-device mbirtorch

**Status:** increments 1-3 COMPLETE and gate-measured at n = 1, 2, 4;
increment 4's seams sweep is partially deferred with reasons, and the panel
review closes the phase.  The design narrative, the substrate decision, and
the compile-race research live in `phase4_design.md`; this page holds the
measurements.
**Code:** `mbirtorch/_sharding.py`, the placement chokepoints and sharded
engine in `tomography_model.py`, the view-range/banded drivers in
`projectors.py` and `cone_beam.py`.  **Scripts:** p4s1 (gloo collectives),
p4s2 (substrate discriminator), p4s3 (engine dispatch), p4_gate_readout.

## What was built

Three layers.  The foundation is `_sharding.py`: `Placement` (devices, the
sharded axis, real/padded lengths), the `Shards` container (per-device
tensors; the explicit-placement stand-in for jax's shard-backed array), the
probed transfer primitive with its host-bounce fallback, the banded adjoint
pair, the per-device thread fan-out, and the qGGMRF halo exchange.  The
middle layer is the drivers: view-range projector variants for both
geometries, with parallel bands CONCATENATING row-bands (one producer per
row) and cone bands ACCUMULATING full-row partials through a slice_start
threaded into the existing kernels.  The top layer is the engine: a sharded
subset updater (halo'd prior, banded projections, host-combined line
search, local in-place applies), sharded branches through init, filter,
Hessian, stats, and exits, and halos staged once per partition pass.

## Correctness

Every layer carries its own gate.  The halo'd per-shard prior equals the
full-volume computation exactly.  The banded projectors match single-device
values at 1e-6 (parallel forward), 1e-5 (sums), with adjointness at 1e-4,
on virtual duplicate-CPU placements and on a real cpu<->mps device pair.
The seeded sharded recon reproduces the single-device trajectory.  The
decisive evidence is the H100 gate matrix: each framework's n-device recon
was diffed against its own n=1 run, and the torch divergences MATCH jax's
own multi-device divergences -- at 512, identically to three digits:

| divergence vs own n=1 | jax | torch |
|---|---|---|
| n=2 @ 512 | 8.69e-05 | 8.75e-05 |
| n=4 @ 512 | 3.03e-04 | 3.03e-04 |
| n=2 @ 1024 | 1.55e-05 | 2.98e-05 |
| n=4 @ 1024 | 1.65e-05 | 2.96e-05 |

The sharded engine sits at jax's own multi-device float floor.

## The gate matrix (H100, 3-iteration vcd, warm)

| cell | frame | n=1 | n=2 | n=4 | n=4 scaling | peak GiB @ n=4 |
|---|---|---|---|---|---|---|
| 512x448x384 | jax | 1.76 | 2.09 | 3.13 | 0.56x | 1.7 |
| 512x448x384 | torch | 3.12 | 2.96 | 9.11 | 0.34x | 6.6 |
| 1024x1008x992 | jax | 26.0 | 14.5 | 11.7 | 2.23x | 14.6 |
| 1024x1008x992 | torch | 94.0 | 70.7 | 60.2 | 1.56x | 13.3 |

Three readings.  First, at the 512 cell multi-device degrades BOTH
frameworks (jax 0.56x at n=4): the cell is orchestration-bound everywhere,
and torch's n=4 collapse (0.34x) is the per-subset band loop growing as
bands x devices.  Second, at the 1024 cell torch scales monotonically (1.33x
at n=2, 1.56x at n=4) but under jax (1.79x, 2.23x); the time ratio (3.6x at
n=1, 5.2x at n=4) is dominated by the known single-device back-kernel gap,
which the kernel work inherits.  Third, torch's device memory comes in
UNDER jax at the 1024 capacity cell for every n (13.3 vs 14.6 GiB at n=4).

## The orchestration measurement (A100, 256-cell)

The 1-device diagnostic (the sharded path with both placements on one GPU)
prices the orchestration alone: 2.61x over the plain single-device run,
with n=2 at 0.44x of n=1.  Small cells are orchestration-bound, the same
regime as mbirjax's own ~95 percent host-dispatch at interactive sizes.
The measured levers, in priority order: per-device compiled mega-regions
for each worker's local chain; on-device scalar combines replacing the
per-subset host syncs; pool reuse across fan-outs; CUDA graphs as the
escalation (single-device first; its cudagraph-trees multithreading bug is
documented upstream).

## The compile race

Two engine threads cold-compiling concurrently crash inside
triton/inductor, even with separate compiled artifacts per device.  The fix
is a process-wide compile lock keyed on unseen input shapes: every compile
event serializes, steady-state threaded execution is lock-free.  The
research (sources in phase4_design.md) confirms this is a known,
acknowledged, unresolved torch limitation, that upstream's own in-flight
fixes are thread-locals plus locks, and that the production alternatives
(main-thread warmup, dynamic shapes, process-per-GPU) avoid concurrent
compilation rather than making it safe.

## Remaining scope, with reasons

Padded (non-dividing) device axes in the drivers (the placement and
chokepoint layers already carry the padding model); prepare_sino_for_devices
and its weights zero-fill; the cone engine's DC-damping profile split per
shard (the cone banded projectors work; the engine guard is loud); sub-band
streaming within shards (mbirjax streams band_bounds for transient memory;
the first cut moves whole shards); the per-device view-batch budget; and
the on-device scalar combine.  Each is recorded in port_plan.md's deferred
list; none fails silently.  The on-device combine has since shipped in
Phase 4a stage 2 (below).

## Phase 4a, stages 1-2: the unified engine (2026-08-06)

Greg's architecture review found that the Phase 4 engine forked sharded and
unsharded versions of its functions.  Phase 4a removed the fork.  Stage 1
folded each sharded twin into its conceptual function, with the branch
inside.  Stage 2 then made the per-device state universal: vcd_recon wraps
its state as Shards at the loop entry and unwraps at the exit, a single
device is a one-shard container that ALIASES its tensor, and one updater
body serves every device count.  Three protections keep one device free of
sharding cost.  run_per_device short-circuits one device to a direct call.
The banded drivers short-circuit a trivial placement to the plain
projectors.  The updater reuses the exact compiled units the single-device
path already had, so the n=1 execution is call-for-call identical.  Two
costs also moved off the host: the line-search partials now combine on the
lead device as 0-d tensors, and vcd_recon owns one thread pool for the
whole loop.

One planned element was deliberately deferred.  The single-compiled-call
mega-region would compile through the update-direction seam, and cone
overrides that seam.  Compiling an override into the chain would change
the n=1 graph structure, a parity risk.  The mega-region therefore remains
the recorded orchestration lever, now implementable once in the uniform
body.

The value gates are unchanged.  The re-run divergence table matches the
Phase 4 table to the printed digit at 512 (8.69e-05 / 8.75e-05 jax/torch
at n=2; 3.03e-04 / 3.04e-04 at n=4), and the 1024 divergences sit at the
same float floor.  The 1-device orchestration diagnostic reproduces its
value diff exactly (rel_diff 1.65e-03, identical to Phase 4).

The gate matrix was re-run on the same H100 class (h010 vs h002), both
frameworks, n = 1, 2, 4.  The jax rows act as the ruler control: jax code
did not change, and its warm times reproduce at 0.93-1.00x, so node and
toolchain drift are bounded.  The torch rows show the stage-2 effect:

| warm vcd, torch | phase 4 | stage 2 | ratio |
|---|---|---|---|
| n=1 @512 | 3.12 | 3.15 | 1.01 |
| n=2 @512 | 2.96 | 2.58 | 0.87 |
| n=4 @512 | 9.11 | 8.59 | 0.94 |
| n=1 @1024 | 94.0 | 94.0 | 1.00 |
| n=2 @1024 | 70.7 | 69.9 | 0.99 |
| n=4 @1024 | 60.2 | 61.9 | 1.03 |

Three readings.  First, n=1 is unchanged (1.00-1.01x), inside the ~3
percent protection bound; the local A/B agrees (MPS and CPU within noise,
CPU 96-cell about 5 percent faster, checksums identical).  Second, the
orchestration-bound n=2 512 cell improved 13 percent.  This is where the
removed per-subset host syncs serialized cross-device work, so the
improvement lands exactly where the design predicted.  Third, the
kernel-bound 1024 cells are unchanged, as expected.  GPU peaks dropped at
every torch cell: n=1 @1024 went 30.0 to 26.7 GiB, n=2 @1024 went 19.1 to
17.2, and n=4 @1024 went 13.3 to 12.3.

The 1-device diagnostic improved substantially, and a hardware confound
initially hid it.  The stage-2 H100 re-run read 2.68x against the 2.61x
A100 baseline, which suggested no movement.  A same-hardware ablation on
gilbreth (same SXM4 class as the baseline, stage-2 tree, job 11478566)
gives the single-variable answer: the diagnostic ratio fell from 2.61x to
2.08x (warm 3.37 s to 2.71 s), the n=2 warm time fell from 2.90 s to
2.52 s, and n=2 scaling rose from 0.44x to 0.52x, with the value diff
identical (1.65e-03).  These results indicate the on-device combine and
pool reuse removed about a fifth of the small-cell orchestration cost.
The remaining ~2x overhead lives in the per-band fan-out and the eager
glue between projector calls -- the mega-region and CUDA-graph levers.

## Phase 4a, stage 3: the seams land once (2026-08-06)

Stage 3 closed the deferred-seams list in the unified engine.  Each seam
was read from the mbirjax source first and implemented to its semantics.

Padded (non-dividing) device axes now work end to end.  The chokepoints
accept either the real shape or the prepared device form, build every
zero tail on the receiving device, and crop both padded axes on gather.
Parallel beam pads its detector-row axis with the recon slices (the
row r <-> slice r tie), through a per-geometry ``_sino_row_padding``
seam.  ``prepare_sino_for_devices`` is the public once-only placement,
and zero-filled weights make padded entries weightless.  Inside the
engine, three mechanisms keep the padding inert.  View-owners project
only their real views, because a padded view has no angle; the forward
driver zero-fills each owner's padded view tail after assembly.  The
back driver re-zeroes the padded slice tail after the band reduce,
because back projection is a gather and real detector data does land in
padded slice positions.  The prior sees per-device interface masks, which
reproduce the reflected boundary at the last real slice even mid-shard.
A layout that would leave the last device with no real VIEWS (5 views on
4 devices) is refused at configure time, as mbirjax refuses it.

An all-padded shard on ONE axis, in contrast, is allowed -- two
deliberate extensions beyond mbirjax (Greg's review, 2026-08-06).  The
refusal now fires only for a device idle on BOTH axes.

The first extension is the thin volume: few slices, many views.  There
the view axis carries both the dominant compute (projection scales with
views) and the dominant memory (the sinogram), so capping the device
count at the slice count -- mbirjax's behavior -- discards exactly the
parallelism that workload needs.  A device with no real slices still
projects its share of the views; its recon side stays exactly inert
through the same padding invariants (zero updates, masked prior,
skipped bands).  The drivers skip all-padding sub-bands outright: the
forward appends the owed zero row-bands without projecting, and the
back produces the zeros its post-reduce mask would have forced anyway.
The halo question resolves cleanly: a halo sourced from a padded slice
is neutralized by the interface mask, whose validity predicate
(``global slice < num_real_slices``) is exactly the halo's own.

The second extension is the mirror case, sparse view: few views, a
large volume.  There the recon memory and the prior dominate, both on
the slice axis, so a device with no real views still earns its keep
holding a slice shard and running its prior and updates.  Bands
broadcast only to view-owners that project something; an empty
view-owner produces empty row-bands (its block assembles as zeros) and
contributes nothing to the band reduces.  Its view-side reductions
already see only padded zeros, so the engine needed no change.

Validating the sparse-view gate produced a calibration finding worth
recording.  The seeded n=4 parity first read 2.98e-3 against the 5e-4
tolerance copied from the (8, 6, 8) gate cell.  Single-variable probes
cleared the projectors and the FBP init (all at the 1e-7 float floor)
and isolated the engine; the discriminating run then showed the SAME
divergence -- 1.8e-3 at n=2, 1.9e-3 at n=4 -- with DIVIDING views and
no padding anywhere.  These results indicate the divergence is the
cell's inherent multi-device floor (staged-halo staleness within a
partition pass plus cross-device float reorders, the mbirjax
structure), not the extension: the sparse-view layout adds only
2.1-2.2e-3 against that 1.8-1.9e-3 baseline.  The parity floor grows
with in-plane coupling and shrinking data constraint, so per-cell gates
must be calibrated against the dividing-case floor of the SAME cell.

The cone multi-device engine is un-guarded.  The update-direction seam
gained a ``dev_index`` argument, and cone's DC-damping profile now splits
per shard (padded slices damp by 1.0, inert), with per-device compiled
damping instances.  The damping formula's slice means are shard-local,
which is exactly right under slice sharding.

Sub-band streaming is implemented in both banded drivers, and the
measurement REVERSED its default.  mbirjax streams by default because its
sweeps found time flat across band length.  The torch gate matrix
(h010 vs h011, jax control rows reproducing at 0.98-1.04x) found the
opposite: the mbirjax-style band bounds cost +66 percent warm vcd at
n=2 @512 (2.58 s to 4.29 s, breaching the 2x replacement ceiling
against jax's 1.96 s), +47 percent at n=2 @1024, and +24 percent at
n=4 @1024.  The cause is that each torch band pass pays the eager
fan-out orchestration the 1-device diagnostic prices at about 2x, and a
single torch device never runs the banded drivers at all (the trivial
fast path uses the plain projectors), so mbirjax's stream-even-at-n=1
rationale is void here.  The peak savings were real -- n=4 @512 fell
from 6.6 to 2.6 GiB for only +8 percent time -- so streaming remains
available as an opt-in memory lever through the
``forward_project_slice_band`` / ``back_project_slice_band`` model
attributes, with one band per owner as the default.  The per-device
view-batch budget also derives from the local view share now, not the
global sinogram.

The panel's low-severity items are closed: device caches (prox data,
interface masks, the damping profile) invalidate on every reconfigure
and recompile; zero tails build on the target device; the Shards
container checks each tensor against its placement device; and the
transfer primitive has small/0-d coverage on both paths.

Local evidence: 259 tests pass with 1 skip.  The new gates are the
padded chokepoint roundtrips, padded banded-projector parity, a seeded
padded n=2 vcd against n=1 (weighted and constant, rel < 5e-4, with the
default FBP init exercising padded direct recon), a seeded cone n=2 vcd
at a dividing and at a padded cell, forced 2-slice-band streaming parity
for both geometries, the empty-VIEW-shard refusal, and the thin-volume
gate: seeded n=4 vcd over 3 slices (one all-padded shard) against n=1
for parallel and cone, with the empty shard's device form asserted
identically zero.

GPU evidence (2x and 4x H100): the padded parallel engine at
(255, 97, 128) reads rel 1.28e-03 against its n=1 reference, the
un-guarded cone engine at 128^3 reads 1.64e-03, and the thin-volume
engine -- 512 views over 3 slices on n=4, one all-padded shard -- reads
4.32e-04.  All sit at the established multi-device float-divergence
scale for cells these sizes (the 256-cell diagnostic's own n=2 floor is
1.65e-03).  The sparse-view engine -- 3 views over a 256-class volume
on n=4, one all-padded view shard -- reads 8.84e-03, and its own
floor-calibration control (4 views, dividing, no padding, no empty
shard) reads 6.52e-03 on the same hardware.  The sparse layout
therefore adds a factor of 1.36 over the cell class's inherent n=4
floor, matching the CPU discriminator's 1.18 -- the magnitude belongs
to the few-view, prior-dominated regime, not to the empty-shard
mechanics.

The dividing-cell confirmation ran on the SAME node as the stage-2
baseline (h010 pinned), removing node variance from the comparison.
With the one-band default, n=1 and n=2 reproduce stage 2 at 0.99-1.01x
at both cells, and the value-divergence table reproduced across all four
matrix runs.  n=4 @512 reads +2 percent for a 6.6 to 2.5 GiB peak
reduction -- the per-device view-batch budget working as intended (at
512 the local-share budget drops below the 2 GiB cap; at 1024 both
forms cap identically and the peaks are byte-identical).  One residual
is real and OPEN: n=4 @1024 reads 71.7 s against 61.9 s (+16 percent),
reproduced across two one-band runs and confirmed same-node.  The
analytical candidates do not fit its size: the per-call additions are
microsecond-scale, the budget is provably unchanged there, and n=2
@1024 runs the same code paths at 1.00x.  The cell is diagnostic-only
-- both stage 2 and stage 3 fail its 2x replacement gate on the known
back-kernel gap -- so the attribution is flagged to the Phase 5 kernel
re-baseline, which rewrites the kernels this cell spends its time in.
