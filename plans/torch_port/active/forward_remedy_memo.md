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
