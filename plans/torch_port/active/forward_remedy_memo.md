# Summary as of 08-10-2026, 4:08PM, revised the same evening after the mbirjax source reading:
Here's the full picture as it stands tonight.

## The bottleneck

Adding GPUs doesn't make the forward projection faster, and the forward is the largest term in the reconstruction. Its per-device time is flat in the device count: cone 1024 holds at 32.2 / 30.6 / 30.5 s across one, two, four devices; parallel 1024 holds at 28.9 / 28.8 s from one to two. On the same node, jax scales these cells 1.45–2.43x. This is why torch's composed reconstruction scaled only 1.02x at parallel-1024 two-device while jax scaled 1.80x. Yesterday's mg9 added one big qualification: parallel *does* halve at four devices (14.9 s, composed scaling 1.70x), so parallel's pathology is specifically the one-to-two leg, while cone is flat at every count.

## What was proposed and ruled out

**Broadcast bytes** (the original §1.5 attribution — every device receives every reconstruction band, and that count-invariant data movement was blamed for the flat span). Refuted in two stages: first by arithmetic (the bytes could only explain the span at an implied 0.44 GB/s, absurdly slow for an H100 node) and by an ablation (the torch-body arm runs the identical driver and broadcast yet scales 1.59x); then definitively by mg9's direct measurement — the broadcast moves at 197–257 GB/s and costs 48–189 *milliseconds* per reconstruction against a 15–31 *second* span.

**Waiting/serialization** (devices idle while copies are issued serially from the loop thread). Refuted by mg9's central measurement: busy time — event pairs around every individual kernel call — is 97–98% of the bracket at every count, and the bracket-minus-busy gap not only stays under a second, it *shrinks* as devices are added. The devices compute essentially the whole time.

Those two refutations killed remedies **A2** and **A3** (move copy issue onto worker threads; overlap next band's broadcast with current compute) — their entire target is under one second. **A1** (broadcast only to devices that need it) was ruled out structurally from the start: forward projection of a view needs every voxel, so every view-owner needs every band. **A5** (project locally, exchange sinogram rows) was excluded because it breaks the documented adjoint pairing between the broadcast and the reduce, and because for cone it needs the cross-device sinogram reduce you held to 2K.

## What's confirmed, and the proposed solution

The cost lives **inside the kernel launches**: per-device launches hold at exactly 680 in every parallel arm while each launch's slice band narrows with the count — you pay full-launch prices for fractional-launch work. For cone this is visible in the code: the launch grid's middle axis is the full detector row count regardless of how narrow a band it was handed.

That mechanism ruling stands. What changed the same evening is the **shape** of the remedy. A source reading of mbirjax, the reference implementation, showed how it solved this same problem, and it solved it differently in each geometry with measured evidence for both. Two shapes now replace the single A4 proposal.

**Parallel: keep the banded walk, and pin the band at a fixed measured knee.** mbirjax bands parallel as we do, and it sizes the band by a constant rather than by the shard. The constant is at `parallel_beam.py:116`: `_FWD_SLICE_BAND_GPU = 256  # the measured knee; whole-shard bands are WORSE at scale`. Per-device dispatches are then about slices/256, which is constant in the device count. Adding devices shrinks each call along the view axis instead. Our port sizes the band to the shard, 504 slices wide at two devices, and mg9 measured that width inside the flat regime. mg9's per-launch times put our own knee in the same region. They are flat at 41.5 ms for the 1008-slice and the 504-slice bands and fall to 21.4 ms at 252, so our knee sits between 252 and 504. The remedy is to keep the walk and to fix the band at a knee swept on **our** Triton kernel.

**Cone: don't band at all, and gather pixel-batched full-height cylinders.** mbirjax's cone path is the opposite structure, and its docstring gives the reason (`tomography_model.py:1654-1660`): "A slice can project to a RANGE of detector rows (cone), so every view-owner needs ALL slices to produce its own views' rows -- it cannot stream slice-bands." Each view-owner gathers a narrow column of pixels at full height and projects it in one call, with `fwd_pixel_batch` at 4096 for 768 slices and above. mbirjax measured the two forms against each other: the full-cylinder form "increased transient memory for the projector by about 10% but decreased time by about 2x" over the per-band form (`dev_sharding_overview.rst:119-124`).

**That dissolves A4's 2K objection, and it retires the grid variant.** The cone shape is A4's coalescing done with pixel tiling, so nothing ever assembles a whole cylinder. The transient is 4096 pixel columns by the slice count by 4 bytes, which is about **34 MB at 2048³** and independent of the device count, rather than 24.9 GB. The same shape moots mg9's cone grid-narrowing variant, because a full-height column genuinely needs the full detector-row range, so that grid axis is no longer wasted work. The **cone value gate survives**. Cone today accumulates per-band partials, which is n host-side adds, and projecting each full-height column in one call changes the vertical accumulation order. Parallel's fixed-knee band needs no value gate, because it is the same walk with a different band size and it moves no accumulation. One point of port history belongs here: the banded cone walk was the *port's* unification of cone onto the parallel shape, and mbirjax's comment plus its measured comparison are direct evidence of what that unification cost. Writing that design note and taking it to a checkpoint ruling is still the next step, and the plan's step 5 carries the charge.

## Sorted stream: yes, still on the table

Item 13 is scheduled, not declined — its entry gate is satisfied and recorded (the forward is ~72% of GPU time at parallel 1024). It's sequenced *after* the driver remedy for a priced reason: the two attack **different failures**. The driver remedy addresses *why the cost doesn't fall when you add devices* (the scaling failure). Sorted stream addresses *how large the cost is at any one count* (the level failure: parallel runs 40.0 s at one device against jax's 25.8, with 14.4 s attributed to the forward kernel and flat across every batch size — atomic throughput, which batching can't touch). They're complements, and ordering A first matters twice over: A moves the baseline B's feasibility probe would price against, and if A restores scaling, B's kernel win is realized at every device count instead of mostly at one.

**Why parallel-only — measured need, and that alone.** This paragraph used to give two reasons, one measured and one structural, and the mbirjax reading showed the structural one was wrong. mbirjax sorts cone too. The sorted form lives in the shared projector helpers behind a flag (`projectors.py:83-114`), and mbirjax enables that flag for cone as well as for parallel (`parallel_beam.py:143`, `cone_beam.py:449`). mbirjax's own campaign record adds the policy behind that flag. All four geometries carry the sorted branch, and three of them enable it by policy: parallel, cone, and multiaxis (the mbirjax campaign record, `plans/projector_kernels/fwd_back_findings.md`). The selection rule is numeric rather than geometric. The guards are MIN_COLS=48, MAX_COLS=1280, and MIN_COLLISION_RATIO=4 (`projectors.py:66-81`), and the variable they gate is the mean channel-collision count, `psf_width * num_pixels / num_det_channels`. Every win that record carries sits at a collision ratio of 6 or more. The fourth geometry, translation, declines the form deliberately, and it declines on a measured result: "at its real detector shapes the sorted form measured 4.5--6.5x slower". Translation's real shapes average about 2 collisions per channel, so sorting cannot recover its own cost there. So the scope is measured need alone. Cone has no level gap to close, at 0.98x of jax at one device, while parallel runs 14.4 s above jax. Cone's entire deficit is scaling, which sorted stream does nothing for, because it changes what happens inside one projector call and not how calls scale. One note applies to the re-gate. mbirjax's shipped win is a *per-call* in-kernel sort (`lax.sort_key_val` + `segment_sum`) rather than the cached pre-sorted streams of our K5 design. It caches nothing because "its per-call setup is ~free", so the light per-call form deserves the first probe.

One more piece deserves its place in the picture: **cone's back projection is the other half of cone's problem** — it *rises* at two devices (23.6 → 30.3 s, reproduced exactly in mg9) and now co-dominates with cone's forward. No forward option touches it; it's flagged for its own remedy choice. So even a perfect A4 leaves cone-1024-n2 around 53 s against jax's 43.4 (findings §1.2), and closing that last gap runs through the back loop, not the forward.

# Decision memo: the remedy for the multi-GPU forward

**Status.** Draft for the session lead's review, then the repo owner's ruling.
Nothing here is implemented.  The decision is dashboard step 5 of item 3,
"the forward-remedy choice (driver change versus item 13's sorted stream)".

**Sources.** Every number below was read from its named source and checked
against the raw rows.  The rows are
`plans/experiments/torch_port/rows/mg5_fwd_attrib_h005_20260809_195819.jsonl`
and `plans/experiments/torch_port/rows/mg7_conebatch_h004_20260810_041607.jsonl`.
The driver is `TomographyModel._sparse_forward_project_sharded` in
`mbirtorch/mbirtorch/tomography_model.py`, lines 434 to 552.

---

## 1. The problem

The forward projection's per-device GPU time does not fall when devices are
added, and the forward is the largest term in the reconstruction, so the
reconstruction does not scale either.  At parallel 1024 the forward's
per-device span holds at 28.87 s and 28.75 s at one and two devices, inside
reconstructions of 40.00 s and 39.30 s.  At cone 1024 the same span holds at
32.18 s, 30.61 s, and 30.48 s at one, two, and four devices.  On the same node
jax scales that cell 1.45x at two devices and 2.43x at four.

---

## 2. What the measurements establish, and where they stop

This section separates the measured facts from the mechanism inferred from
them.  The separation matters because both remedies below are priced against
the mechanism, and the mechanism has an unmeasured step.

### 2.1 The measured facts

Five facts are direct readings, and all five reproduce in the raw rows.

| fact | numbers | source |
|---|---|---|
| the forward's per-device span is flat in the device count | cone 1024: 32.18, 30.61, 30.48 s at n = 1, 2, 4; parallel 1024: 28.87, 28.75 s at n = 1, 2 | §1.5, mg5 rows |
| no view-chunk constant helps | cone 1024 total 62.12, 61.78, 61.53, 61.31 s over chunks 32 to 256; parallel 1024 total 40.17, 40.05, 40.00, 39.94 s | §1.5, mg5 rows |
| the realized view batch does not vary with the count | parallel holds 128 at every count; cone's full-pixel call drops to 85 at the 384 cell and 113 at the 512 cell, at four devices only | §1.6, mg7 rows |
| view-range calls, summed over devices, grow fourfold per count doubling | 85, 340, 1360 per reconstruction at n = 1, 2, 4; see the label correction in §2.2 | §1.6, mg7 rows |
| the forward is the largest attributed term | 28.87 s of device span inside a 40.00 s reconstruction at parallel 1024, about 72 percent | §1.5 |

These figures are the measured base for everything below.

### 2.2 One correction to §1.6's call counts

The 85, 340, 1360 series is the total over all devices, not the per-device
count.  §1.6 labels it per device.  The label is wrong, and the reason is a
collapsed instrument key.

The observer identifies a call's device from the identity of the projection
body it was given.  `observe_view_batches` in `mg1_readout.py` builds
`fwd_ids = {id(b): i for i, b in enumerate(pf._fwd_body_per_dev)}`.  The
per-device bodies are built by `maybe_compile(fwd_body, use_compile,
instance_key=i)` in `mbirtorch/mbirtorch/projectors.py` line 261.
`maybe_compile` returns the function unchanged when it carries
`_mbirtorch_no_compile`, and both Triton forward bodies set that marker.  Every
entry of `_fwd_body_per_dev` is therefore the same object.  The identity map
collapses to a single entry holding the last device index.  Every device's
calls are then recorded under that one key.

The rows confirm the collapse.  At one, two, and four devices the forward keys
present are exactly one each, and they are `fwd_dev0`, `fwd_dev1`, and
`fwd_dev3`, always the highest device index and never the others.

The corrected counts follow from the driver's structure, and they match the
rows exactly.  The forward funnel is entered 85 times per reconstruction.  Each
entry walks n slice-owner bands and fans each band out to n devices, so the
total is 85n² calls: 85, 340, 1360.  Per device the count is 85n: 85, 170, 340.
Per-device kernel launches are flat rather than growing, at 340 per
reconstruction at the 512 cells and 680 at the 1024 cells.  They are flat
because the view batch stays at 128 while each device's view count falls.

Two consequences follow.  The growth a coalescing remedy could remove is 85n
Python-level calls per device, not a fourfold launch growth.  The batch
findings of §1.6 are unaffected, because a batch value is a property of the
call rather than of the device that recorded it.

### 2.3 Where the evidence stops

The mechanism in §1.5 is an inference, and no instrument measured it.  §1.5
attributes the flatness to data movement that is invariant in the device count,
namely the broadcast of every reconstruction band to every device.  The
three-region instrument brackets the whole `sparse_forward_project` call with
CUDA events per device.  It does not separate copy time from kernel time inside
that bracket, and the campaign ran no arm that does.

Two checks in hand make the byte volume an unlikely explanation on its own.

The first check is arithmetic on the bytes.  At the 1024 cell the full-pixel
cylinder is 771,240 pixels by 1008 slices by 4 bytes, which is 3.11 GB.  One
reconstruction runs 85 forward calls, of which one is full-pixel and 84 are
subset calls.  The subset sizes sum to the full pixel count once per iteration
over three iterations, so the traffic is about four full cylinders per
reconstruction.  Each device receives the (n-1)/n of that which it does not
own, which is 6.2 GB at two devices and 9.3 GB at four.  The time to be
explained at parallel 1024 at two devices is about 14 s per reconstruction,
and 6.2 GB in 14 s is 0.44 GB/s.  That rate is far below any device-to-device
or host-staged path on an H100 node, so the copied bytes alone cannot fill the
flat span.  The count of 84 subset calls rests on one inference, and mg7
supports it.  Cone at four devices shows 16 launches in excess of its event
count.  Sixteen is one full-pixel funnel call fanned out to n² view-range
calls.

The second check is an ablation already in the mg5 rows.  The torch-body arm
runs the identical driver with the identical broadcast and differs only in the
projection body.  At parallel 1024 its forward span reads 39.69 s at one device
and 24.96 s at two, a fall of 1.59x.  The kernel arm at the same two counts
reads 28.87 s and 28.75 s, a fall of 1.00x.  If the driver's data movement were
the invariant term, it would appear in both arms.  It appears in one.  Fitting
a constant plus a term falling as one over the count gives an invariant part of
about 5 s for the torch-body arm.  The same fit gives 28.6 s for the kernel
arm.  These results indicate that most of the invariant term sits inside the
kernel path, or in how the driver serializes against it, rather than in the
broadcast's bytes.

Two candidate mechanisms survive both checks, and the evidence does not choose
between them.  The first candidate is serialization.  The band walk issues each
band's copies from the loop thread, and it never overlaps one band's transfer
with the previous band's compute.  A device's event bracket can therefore cover
time the device spends waiting rather than computing.  The second candidate is
a per-launch cost that does not shrink when the band narrows.  For cone that
cost is visible in the code.  `_cone_forward_view_batch_triton` in
`mbirtorch/mbirtorch/triton_cone.py` sizes its grid as
`(ceil(num_pixels/block_p), ceil(num_rows_r/block_r), num_views)`.  The middle
term is the full detector row count and not the band length, so the launch
spans the whole detector whatever band it was handed.  For parallel the grid
does shrink with the band, so the second candidate is weaker there.

Two further gaps are worth recording.  mg5 ran no parallel arm at four devices,
so the parallel forward's device span at four devices is unmeasured.  mg5
varied only the forward chunk, so cone's back-projection rise at two devices,
from 23.59 s to 30.30 s of device span, is unattributed and is untouched by
either remedy below.

---

## 3. Option A: change the driver

Option A changes `_sparse_forward_project_sharded` and the transfer primitive
it calls, leaving the projection kernels alone.  Five candidates were developed
from the code.  Two of them are not available, and the reasons are as useful as
the candidates that are.

### A1. Broadcast each band only to the devices that need it

This candidate is not available for dense placements, and the reason is
structural.  Forward projection of a view needs every voxel, so every
view-owner needs every band.  The only devices that can be skipped are those
owning no real views, and the driver already skips them.  Line 465 builds
`proj_devs` from the owners whose view span is non-empty.  No further reduction
exists without changing which device computes which output.

**Size:** zero.  **Recommendation:** drop it.

### A2. Move the copies onto the destination devices' threads

`broadcast_band_to_views` in `mbirtorch/mbirtorch/_sharding.py` line 252 builds
its copies in a serial dictionary comprehension on the calling thread.  The
calling thread is the reconstruction loop's thread.  Each destination's copy is
therefore issued after the previous destination's copy, from one thread and one
device context.  Issuing each copy from the worker thread that will consume it
puts the n copies on n threads and n device contexts.

**Mechanism attacked:** serialization of the fan-out, the first surviving
candidate in §2.3.  **Predicted effect:** unquantified.  This is a prediction
with no measurement behind it.  If serialization is the binding term, the
parallel 1024 two-device span should move from 28.75 s toward the 14.5 s that
halved arithmetic implies.  If the binding term is a per-launch kernel cost,
this candidate moves nothing.  **Memory interaction:** none.  The same copies
exist, at the same times, so the ledger's `forward_band_copy` term is
unchanged.  **Risk class:** value-neutral.  A copy is exact, the projector
calls are unchanged, and no summation order moves.  **Size:** about 10 to 20
lines across `_sharding.py` and the one call site in `tomography_model.py`.

### A3. Overlap a band's broadcast with the previous band's compute

The band loop at lines 468 to 538 runs strictly in sequence: broadcast band k,
project band k, broadcast band k+1, project band k+1.  Issuing band k+1's
copies before waiting on band k's projections lets the transfer and the
compute occupy the device at the same time.  `run_per_device` performs no
synchronization by design, and its docstring names this overlap as the reason.

**Mechanism attacked:** the same serialization as A2, one level up.
**Predicted effect:** unquantified, and a prediction.  The ceiling is the same
14.5 s target as A2, and A2 and A3 overlap in what they would recover, so
their gains do not add.  **Memory interaction:** real, and it must be charged.
Prefetching one band ahead keeps two broadcast copies resident per device
instead of one.  The ledger's `forward_band_copy` in
`mbirtorch/mbirtorch/_memory_ledger.py` line 371 charges one cylinder shard,
which is `num_pixels` by the device's slice block by 4 bytes.  At the 1024 cell
that is 1.55 GB at two devices and 0.78 GB at four, so the term doubles by
those amounts.  The measured peaks are 14.04 GB and 7.31 GB against 80 GB of
capacity, so the increase is affordable at the 1K cells.  At charter C's 2K
design point the reconstruction is 31.7 GB, so the added residency is about
7.9 GB per device at four devices.  The 2K capacity table must carry that
addition.
**Risk class:** value-neutral.  The projector calls and their inputs are
identical, and only the time at which a copy is made changes.  **Size:** about
30 to 60 lines inside one function in `tomography_model.py`, plus about 10
lines in `_memory_ledger.py`, plus the ledger's calibration check.

### A4. Coalesce the per-owner walk into one call per device

Each device currently makes n view-range calls per forward call, one per
slice-owner band, each over a band of `slices/n` columns.  Assembling the n
received bands into one cylinder and making a single call restores the
one-device call shape: full slice range, one launch sequence, and each device's
own views.  Per-device view-range calls fall from 85n to 85, and per-device
launches fall as one over the count instead of staying flat.

**Mechanism attacked:** the per-launch cost that does not shrink with the band,
the second surviving candidate in §2.3.  It also removes n-1 of the n
copy-then-compute stages per forward call, so it attacks serialization as well.
**Predicted effect:** unquantified, and a prediction.  If the flat term is a
band-width-independent per-launch cost, this candidate removes it, because
every launch returns to the shape it has at one device.  **Memory interaction:**
the largest of any candidate.  The assembled cylinder is `num_pixels` by all
slices, which is n times the current broadcast-band charge, or 3.11 GB per
device at the 1024 cell.  That fits at 1K, where the measured peak is 7.31 GB
at four devices.  It does not fit at 2K, where the full cylinder is 24.9 GB per
device and single-device residency of it is already ruled impossible in §6.4.
A4 is therefore a 1K remedy that charter C would have to undo.  **Risk class:**
mixed.  For parallel it is order-preserving, because each detector row has a
single producer either way.  For cone it is order-changing, because n
host-side partial accumulations become one in-kernel atomic accumulation.
**Size:** about 40 to 80 lines in `tomography_model.py`, plus the ledger terms,
plus a value gate for cone.

### A5. Project locally and exchange sinogram rows

For a row-aligned geometry each slice band produces exactly the detector rows
that match it.  A device could project its own band into all views and then
send each view-owner the rows it owns.  The traffic becomes (n-1)/n of one
sinogram instead of n-1 cylinders.

This candidate is excluded, on two grounds.  The first is the adjoint
contract.  `broadcast_band_to_views` is documented as the transpose of
`sum_band_to_owner`, and that relationship is what keeps the sharded forward
and back projections adjoint.  Changing one side alone breaks it, so A5 is a
restructure of both drivers rather than of one.  The second is the seam ruling.
For cone the same scheme requires a new cross-device sinogram reduce, and the
repo owner ruled on 2026-08-10 that the reduce leg is held to 2K under charter
C.  A remedy chosen now may not depend on that leg.

**Size:** 100 to 200 lines across two drivers and a new exchange primitive,
plus adjoint and value gates.  **Recommendation:** do not take it now.

### What Option A does not touch

The forward driver never calls `sum_band_to_owner`.  Its cone branch
accumulates locally, with `partial_shards[i].add_(partials[i])` at line 527.
The band-reduce calls counted in the mg5 rows belong to the back driver, where
the region is nested.  Candidates A2, A3, and A4 are therefore clear of the
held seam.

---

## 4. Option B: item 13, the sorted-stream parallel forward

Item 13 replaces the parallel forward kernel's per-tap atomic scatter with a
two-phase sorted-stream walk.  The streams are built from the existing
horizontal-fan contract, and they are cached through the `plan` slot that
`Projectors.sparse_forward_project_view_range` already accepts and today
ignores.  The design is recorded in `plans/torch_port/phases/phase5_kernel_design.md`
as K5, and the decision that declined it, with its revisit triggers, is in
`plans/torch_port/closed/kernel_batching_findings.md`.

**Which mechanism it attacks.** It attacks the forward kernel's atomic
throughput at a fixed device count.  The target is the measured 14.4 s by which
torch's parallel 1024 reconstruction exceeds jax's.  That residual was flat
across every batch size in the batching sweep, so batching cannot address it.

**Whether it reduces per-device data movement.** It does not.  Item 13 changes
what happens inside one projector call.  The driver calls
`broadcast_band_to_views` before any projector call, so the band broadcast
survives item 13 unchanged, in count, in size, and in timing.

**Whether it reorders work so movement overlaps.** It does not.  The band walk,
its serial copy issue, and its per-band fan-out are untouched.

**What it does not fix.** It does not fix the device-count scaling.  A faster
kernel lowers only the part of the forward's span that falls with the count.
§2.3's fit puts that part at about 0.24 s of the 28.87 s measured at parallel
1024.  On that fit the whole of item 13's win would land at one device and
almost none of it at more.  That fit is an inference from two points and should
be treated as such.  The safe statement is narrower.  Item 13 addresses the
level of the forward's cost and not its flatness, and nothing in its design
touches the term that makes the forward flat.

**Which geometry it serves.** Parallel only.  The sorted-stream design was
declined for cone on measured need, and cone already runs at 0.98x of jax at
one device.  Cone's whole problem is scaling, at 0.92x at two devices and 1.16x
at four, so Option B offers cone nothing.

**Scope and risk.** The charter commits to the full kernel protocol: an
emulator, a battery, a sweep, composed gates, and the default flip at a Fable
checkpoint.  The risk class is a new kernel with a new value class, which is
the campaign's most expensive class of change and the one with the most
established protocol.

**What the charter already commits.** Step 1 is a feasibility probe on item 3's
attributed numbers, with the view-loop forward variant as a cheap comparison
arm.  The charter's STOP condition is a projected composed win below a few
seconds.  The entry gate is satisfied and recorded.  By device span the forward
is about 72 percent of GPU time at parallel 1024.

---

## 5. Are A and B substitutes or complements?

They are complements, and they address different failures.  Option A addresses
why the forward's cost does not fall when devices are added.  Option B
addresses how large that cost is at any one count.  Neither substitutes for the
other, and the measured numbers separate the two failures cleanly.  The level
failure is 40.00 s against jax's 25.80 s at one device.  The scaling failure is
1.02x against jax's 1.80x at two devices.

A cheap value-neutral A candidate changes B's economics in two ways, and both
argue for ordering A first.  The first way is the baseline.  B's step-1 probe
prices itself on item 3's attributed numbers, and any A candidate moves those
numbers, so a probe run before A would price the wrong baseline.  The second
way is the multiplier.  If A restores the fall with the count, B's kernel win
is realized at every count rather than at one, which raises B's value.  If A
does not land, B's win at more than one device is diluted by the flat term that
remains.

One asymmetry decides the order if only one thing is done.  Option A serves
both geometries and Option B serves one.  Cone's scaling is the worse of the
two, and only A can move it.

---

## 6. Recommendation

**Buy one measurement before committing to either option.**  The mechanism
behind the flat span is not established.  §1.5's attribution to the broadcast's
bytes does not survive either check in §2.3.  Both options are priced against
that mechanism, so committing now risks spending the more expensive option on a
term it does not touch.

The measurement is small and it discriminates.  Add a broadcast sub-region to
the existing three-region instrument, so copy time inside the forward call is
recorded separately from kernel time.  Record each device's busy time
separately from its event bracket, so waiting is distinguished from computing.
Then re-run three arms: parallel 1024 at one and two devices, and cone 1024 at
two devices.  The instrument lives in the plans repository and mbirtorch is not
edited, so the change is a harness change.  The cluster cost is one arm triple
at the 1024 cells, which §1.4 prices at 2 to 5 minutes of subprocess wall each.

**Then take Option A, in the order the measurement selects.**  If copy time or
waiting dominates, take A2 and then A3, both value-neutral and both small.  If a
per-launch kernel cost dominates, take A4, and price it against charter C's 2K
design point before it lands, because A4's residency does not survive at 2K.
Do not take A1 or A5.

**Keep item 13 scheduled, and re-gate it after A lands.**  Its entry gate is
satisfied and its design is ready.  Its step-1 STOP threshold should be
evaluated on post-A numbers rather than on today's, for the reason §5 gives.

**Record two follow-ups that neither option covers.**  Cone's back projection
rises from 23.59 s to 30.30 s of device span at two devices, and charter A
already names it as needing its own variant.  The parallel forward's device
span at four devices was never measured, so the four-device arm should be added
to whichever sweep runs next.

### The validating measurement

The widening floors carry §3.3's staleness rule, so a knee refresh follows any
forward-path change.  The refresh is automated in
`mbirtorch/mbirtorch/_widening_floors.py`, and its first real run took 31
minutes on four GPUs.  §3.3 estimates about fifteen minutes for the three or
four cells nearest each floor.

The refresh should move the forward-sensitive knees down and leave the others
alone.  The forward-sensitive readings are cone at every count above one and
parallel 1024 at two devices.  The readings below the 384 cell are limited by
the fan-out and glue, which no forward change touches, so those knees should
hold.

Four numbers in the findings should move, and their predicted directions are
these.  §1.5's forward device spans should acquire a fall with the device
count, from cone's 32.18, 30.61, 30.48 s and parallel's 28.87, 28.75 s.
§1.2's torch scale column at parallel 1024 at two devices should rise from
1.02x toward jax's 1.80x.  §1.2's torch scale column at cone 1024 at two
devices should rise from 0.92x toward jax's 1.45x.  §1.2's torch-to-jax ratio
at parallel 1024 at two devices should fall from 2.75.

Two size predictions follow if Option A removes the flat term entirely, and
both are predictions rather than measurements.  Parallel 1024 at two devices
would fall from 39.40 s to about 25 s, which is a scale of about 1.60x against
jax's 1.80x.  Cone 1024 at two devices would fall from 67.23 s to about 53 s,
which is a scale of about 1.17x against jax's 1.45x.  Cone stays short of jax
in that prediction, and the reason is the cone back projection's own rise at
two devices, which Option A does not address.

If Option B lands instead, a different number should move.  Parallel 1024 at
one device should fall from 40.00 s toward jax's 25.80 s, by up to the 14.4 s
the batching findings attribute to the forward.  The scaling columns should
not move.  Cone should not move at all.

### One record correction to make

§1.6's call-count sentence should be corrected whatever is ruled.  The 85, 340,
1360 series is the total over all devices and not the per-device count, for the
reason §2.2 gives.  The per-device series is 85, 170, 340.  The mg7 arm check
`fwd_batch_uniform_across_devices` should also be marked as not established,
because the collapsed key leaves only one device's readings in the row.

---

## 7. The measurement's answer (2026-08-10)

The §6 instrument ran the same day as mg9, job 15152345 on h018, on
the merged tip f985a6e.  Findings §1.7 carries the full table and the
validity checks.  The answer in one sentence: the flat span is
kernel-busy time, not copying and not waiting.

Three numbers make the ruling.  Busy time is 97 to 98 percent of the
bracket at every count.  The broadcast costs 48 to 189 ms per
reconstruction and moves at 197 to 257 GB/s, against the 0.44 GB/s
that §1.5's byte inference required.  The bracket-minus-busy gap is
0.3 to 1.0 s and shrinks as devices are added.

The order §6 asked the measurement to select is therefore this.  A2
and A3 are declined, because their entire target is under one second.
A4 goes forward, gated on the 2K residency pricing §3 requires.  For
cone, one variant must be priced beside A4 before either is
implemented: sizing the cone kernel's launch grid to the band's
detector-row span instead of the full detector, the grid term §2.3
named.  The variant changes only the launch shape, keeps the per-band
accumulation order, and adds no residency; whether it recovers what
A4 recovers is not yet measured.

One §6 gap is now filled, and it amends a prediction.  mg5 never
measured parallel at four devices; mg9 did, and the span is 14.88 s,
half of the one- and two-device 28.8 s.  Parallel's flatness is
confined to the one-to-two leg, and the composed reconstruction at
parallel 1024 already scales 1.70x at four devices.  The A4 sizing in
§3 should therefore be priced primarily against the one-to-two leg
for parallel and against every leg for cone.  Item 13's re-gate after
a remedy lands is unchanged, and the cone back rise reproduced at
30.33 s of device span, so its separate variant stands.

---

## 8. The remedy, revised against mbirjax's source (2026-08-10, evening)

§7's mechanism ruling stands, and §7's choice of remedy shape does not.
On the evening of the mg9 read the repo owner asked three questions
about mbirjax, the reference implementation this port follows.  The
questions were how mbirjax dispatches a projection to several devices,
what each device holds resident during a sharded forward, and what
bounds per-device memory at the 2K design point.  A source reading
answered all three from the mbirjax checkout at
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax`, and its
file-and-line record is the basis of everything below, and that record
is archived as `reviews/mbirjax_source_reading_2026-08-10.md`.  The
answers matter because mbirjax met the same problem this memo is about.  It
solved that problem differently in each geometry, and it recorded
measured evidence for each of the two solutions.

### 8.1 Parallel: the band is pinned to a measured knee

mbirjax bands the parallel forward as this port does, and it sizes the
band by a constant rather than by the shard.  The constant is in
`parallel_beam.py` line 116: `_FWD_SLICE_BAND_GPU = 256  # the measured
knee; whole-shard bands are WORSE at scale`.  The band length used is
`min(256, slices_per_dev)`, at `parallel_beam.py` lines 186 to 192.

The dispatch arithmetic follows from that constant.  Per-device calls
are the device count times `ceil(slices_per_dev / 256)`, which is about
`num_slices / 256` and therefore constant in the device count.  At 1024
slices the per-device count is four at one device, four at two, and
four at four.  Adding devices shrinks each call along the view axis,
because each call carries only the calling device's view range.  This
arithmetic was derived by the source reader from the code rather than
read from a comment, and it is flagged as an inference in the reading.

Our port sizes the band to the shard instead.  Its per-device
view-range calls therefore grow with the device count, at 85n per
reconstruction by §2.2, where mbirjax's hold constant.  At two devices
our band is 504 slices, and mg9 measured that width inside the flat
regime.  mg9's per-launch times agree with mbirjax's constant to within
the granularity of the measurement.  They are 41.5 ms at the full
1008-slice band, 41.5 ms at the 504-slice band, and 21.4 ms at the
252-slice band (findings §1.7).  A knee between 252 and 504 is
consistent with both readings.

The parallel remedy is therefore to keep the band walk and to fix the
band length at a knee swept on our own kernel.  mbirjax's 256 is the
knee of mbirjax's kernel, and it is not automatically the knee of our
Triton kernel.

This shape is not A4's coalescing.  A4 assembles the bands a device
receives into one call spanning every slice.  The comment at
`parallel_beam.py` line 116 records the opposite measured result, that
whole-shard bands are worse at scale, and A4's assembled band is wider
than a whole shard.

### 8.2 Cone: no banding at all, and full-height column batches

mbirjax does not band the cone forward, and its own docstring gives the
reason.  `tomography_model.py` lines 1654 to 1660 state it: "A slice can
project to a RANGE of detector rows (cone), so every view-owner needs
ALL slices to produce its own views' rows -- it cannot stream
slice-bands."

Each cone view-owner instead gathers full-height cylinders one pixel
batch at a time.  The loop is at `tomography_model.py` lines 1695 to
1709.  For each batch of pixel columns the view-owner concatenates that
column range from every slice-owner into one full-height cylinder, then
makes a single projector call on it.  The batch size is
`fwd_pixel_batch`, which is 4096 at 768 slices and above, at
`cone_beam.py` lines 415 to 416 and 436 to 437.  The cross-device
transient is therefore 4096 pixel columns by the slice count by 4
bytes, which is about 34 MB at 2048³.  That size does not grow with the
device count.

mbirjax measured the two forms against each other, and the record is in
`docs/source/dev_sharding_overview.rst` lines 119 to 124.  The
whole-cylinder form "increased transient memory for the projector by
about 10% but decreased time by about 2x" over the per-band form.
These results indicate that a banded cone walk costs about a factor of
two in the reference implementation, at a transient-memory saving of
about ten percent.

One point of port history belongs in this record.  The banded cone walk
in this port was the port's own unification of cone onto the parallel
shape.  mbirjax's cone path is the opposite structure, so the comment
quoted above and the measured comparison beside it are direct evidence
of what that unification cost.

Both constants the two shapes rest on also appear in a record this
repository already carries.  The mbirjax campaign record at
`plans/projector_kernels/fwd_back_findings.md` states the measured
tiling as "slice band 256, pixel batch 8192; cone gets pixel batch 4096
above 768 slices", which corroborates the source reading from an
independent document.

### 8.3 Three further observations from the reading

Three points support the two shapes above without deciding anything.

mbirjax keeps every loop that moves no data between devices inside one
compiled program.  Its pixel batches run as `jax.lax.scan` and its view
batches as `jax.lax.map`, both inside a single `jax.jit`, at
`projectors.py` lines 710, 816, and 896 to 898.  Only the loop that
actually copies between devices is written in Python.

mbirjax carries an explicit floor on the work in one dispatch.
`_BACK_PROJECT_MIN_BAND_WORK = 4_000_000` elements exists because bands
of about 0.8M elements "added dispatch overhead with no memory benefit"
while bands of about 50M "scaled fine and even sped up", at
`tomography_model.py` lines 2218 to 2225.  A band knee therefore has a
floor beneath it as well as a ceiling above it.

mbirjax probes device-to-device copies for correctness before trusting
them.  `transfer.py` lines 36 to 58 record that a direct `device_put` on
L40S "silently produces zeros on the destination — no error is raised",
and the code falls back to a host bounce when the probe fails.  This
port's `dev2dev_safe` flag covers the same hazard.

### 8.4 What this supersedes in §6 and §7

Two items in the earlier record are superseded, and each for a
different reason.

A4's open 2K-residency question is dissolved rather than answered.  §3's
A4 entry charges a whole assembled cylinder, which is 3.11 GB per device
at the 1024 cell and 24.9 GB at 2K.  Findings §6.4 already rules
whole-cylinder residency impossible at 2K.  mbirjax's pixel tiling never
assembles a whole cylinder.  It assembles one column batch at a time, so
the residency question A4 raised does not arise in the tiled form.  The
number that replaces 24.9 GB is about 34 MB at 2048³.

§7's cone grid variant is withdrawn as moot.  That variant proposed
sizing the cone launch grid to the band's detector-row span instead of
to the full detector.  It was worth pricing only while cone was banded
by slices, because a narrow slice band reaches only part of the
detector.  A full-height column of voxels reaches the whole detector-row
range, so under the column-gather shape the grid's full-detector axis is
not wasted work.  §2.3's reading of that grid term stands as a reading
of the code, and the remedy built on it does not.

### 8.5 What survives, and one argument that does not

Four earlier rulings are unaffected by this revision.

The mechanism ruling stands.  The flat span is kernel-busy time, at 97
to 98 percent of the per-device bracket at every count (§7, findings
§1.7).

A2 and A3 stay declined.  Their target is the bracket-minus-busy gap,
which measures 0.3 to 1.0 s against a 29 s span.

The cone value gate stands.  Cone today accumulates n host-side partial
sums, one per band, at `partial_shards[i].add_(partials[i])` on line 527
of the driver.  The column-gather form projects each full-height column
in one call, so the order of the vertical accumulation changes.
Parallel's fixed-knee band needs no such gate, because it is the same
walk with a different band length and it moves no summation.

Item 13 keeps its position, with one addition.  It stays scheduled after
the driver remedy, for §5's reasons, and it stays scoped to parallel.
The addition concerns its form.  mbirjax's shipped sorted-channel reduce
is a per-call sort inside the compiled kernel, `jax.lax.sort_key_val`
followed by `jax.ops.segment_sum`, at `projectors.py` lines 125 to 143.
It caches nothing, because "its per-call setup is ~free"
(`parallel_beam.py` lines 136 to 137).  Our K5 design instead builds
cached pre-sorted streams.  The light per-call form should therefore be
the first probe when item 13 re-gates.

One argument for the parallel-only scope does not survive the reading.
That scope has been argued on two grounds, one measured and one
structural.  The structural ground was that cone offers no static
structure against which a sort could be cached.  mbirjax selects the
sorted form through a flag in the shared projector helpers
(`projectors.py` lines 83 to 114), and it enables that flag for cone as
well as for parallel (`parallel_beam.py` line 143, `cone_beam.py` line
449).  mbirjax's own campaign record for these kernels states the
policy behind that flag.  All four geometries carry the sorted branch,
and three of them enable it by policy: parallel, cone, and multiaxis
(the mbirjax campaign record,
`plans/projector_kernels/fwd_back_findings.md`).

The selection rule is numeric rather than geometric.  The guards are
`SORTED_CHANNEL_REDUCE_MIN_COLS = 48`, `MAX_COLS = 1280`, and
`MIN_COLLISION_RATIO = 4`, at `projectors.py` lines 66 to 81.  The
variable those guards gate is the mean channel-collision count,
`psf_width * num_pixels / num_det_channels`, and every win the campaign
record carries sits at a collision ratio of 6 or more.  Translation is
the fourth geometry, and it declines the form deliberately on a
measured result: "at its real detector shapes the sorted form measured
4.5--6.5x slower" (`dev_projector_kernels.rst` lines 46 to 48).
Translation's real detector shapes average about 2 collisions per
channel, so sorting cannot recover its own cost there (the mbirjax
campaign record).  A cone sorted form is therefore not structurally
barred.  It stays unscheduled on measured need alone, because cone runs
at 0.98x of jax at one device while parallel runs 14.4 s above jax.
The summary block at the top of this memo is corrected accordingly.

### 8.6 What the design note must still establish

The reading transfers a shape, not a number.  Three things must be
established on our own kernels, and the design note owes all three.

The first is the parallel band knee.  Sweep the band length on our
Triton parallel forward at a fixed device count, and read the per-launch
time against the band length.  mg9 supplies two points already, 41.5 ms
at 504 slices and 21.4 ms at 252, so the sweep needs the interval
between them and the region below 252.  §8.3's dispatch floor is the
reason to sweep downward as well as upward.

The second is the cone column-batch size.  Sweep the analogue of
`fwd_pixel_batch` on our cone kernel, and read both the time and the
transient against it.  mbirjax's 4096 is a knee on its own kernel and
its own tiling, and our kernel batches pixels differently.

The third is the value gate protocol for cone's order change.  The
column-gather form changes the vertical accumulation order, so the note
must name the gate it will run and the tolerance it will hold before any
code moves.
