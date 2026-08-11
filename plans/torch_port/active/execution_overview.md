# How mbirjax and mbirtorch execute a reconstruction

**Status:** REFERENCE, written 2026-08-11.  This page describes how each
library chooses its devices, what it moves between them, and what it runs
inside one.  It covers parallel beam and cone beam, the two geometries with
hand-written kernels and measured device-count floors.  Translation and
multiaxis are named only where their treatment differs.

The page reads the code as of mbirtorch commit 2d2b99a and the mbirjax
prerelease branch.  Measured numbers carry their job and their date, and §5
names which job each number came from.  Where a number is an argument rather
than a measurement, the text says so.

## Contents

1. [The shape of each library](#1-the-shape-of-each-library)
2. [Choosing devices](#2-choosing-devices) — including
   [what re-evaluation does](#23-what-re-evaluation-does-and-what-it-does-not-do)
   and [which entries reach the policy](#24-which-entry-points-reach-the-policy)
3. [What crosses between devices](#3-what-crosses-between-devices)
4. [What runs inside one device](#4-what-runs-inside-one-device)
5. [Measured vcd_recon on gautschi](#5-measured-vcd_recon-on-gautschi)
6. [The tile policy](#6-the-tile-policy)
7. [Open questions this page surfaced](#7-open-questions-this-page-surfaced)

---

## 1. The shape of each library

The two libraries share their parallel structure and differ in how they
express it.  Both run one process that drives every device.  Both shard
recon-like arrays by SLICE and sinogram-like arrays by VIEW.  Both issue
per-device work from one python thread per device.  Both zero-pad a sharded
axis that does not divide the device count, and both keep that padding
exactly inert.

The libraries differ in what a sharded array IS.  In mbirjax a sharded array
is one jax array with a `NamedSharding`, assembled from per-device pieces by
`make_array_from_single_device_arrays`.  In mbirtorch there is no such
object.  A sharded array is a `Shards` container holding one plain tensor per
device plus the `Placement` that says which global block each tensor covers.
The port avoided DTensor deliberately, judging it immature for these
index-heavy kernels.

That difference explains most of what follows.  mbirjax can hand a sharded
array to a jitted function and let the SPMD compiler place the work.
mbirtorch must name every device and every transfer in python.  Each library
then wrote its drivers in the style its array type allows.

One consequence is worth stating in advance.  At a single device mbirtorch
takes a separate path.  A trivial placement short-circuits to the plain
projectors, and the sharded drivers never run.  mbirjax keeps the placement
path at every device count, with a single device as the trivial one-shard
case, and short-circuits only inside the back projector on GPU.

## 2. Choosing devices

This is where the two libraries diverge most.  mbirjax uses every available
device.  mbirtorch searches for a device count, and the search consults both
speed and memory.

### 2.1 mbirjax: all available devices

mbirjax shards across every device of the default platform.
`_auto_device_pool` returns all GPUs when a GPU backend is present, and all
CPU devices otherwise.  `_auto_device_count` then trims that pool for exactly
one reason.  It skips any count whose LAST shard would hold zero real slices.
Nothing else reduces the count.

The rule was decided deliberately, and its rationale is recorded in the
method.  Sharding's value is capacity plus near-linear speedup at the sizes
that matter, and over-sharding a small problem was judged a mild overhead.

The nightly measurements say that judgment is now wrong at small sizes.  On
the 200-class cell, parallel `vcd_nonconst` reads 556 ms at one device and
2,627 ms at four, which is 0.21x.  At the 512-class cell the same series reads
1,375 ms and 2,909 ms, which is 0.47x.  These are the default choices a user
gets on a four-GPU node.  §7 carries this as an open question rather than a
finding, because whether mbirjax gets further development is a separate
decision.

The virtual CPU device count is set at import, before jax initializes.
`_device_setup` caps it at `DEFAULT_MAX_CPU_DEVICES = 2`, sized from
performance cores on macOS and from the process CPU affinity set on Linux.
`MBIRJAX_NUM_CPU_DEVICES` overrides it.

`configure_devices` is the explicit door.  It accepts None for automatic,
`'cpu'` or `'gpu'` for a platform, an integer count, a list of indices, or a
list of devices.  A call to it PINS the layout, so a later `set_params` does
not re-select.

### 2.2 mbirtorch: floors, then a memory ledger

mbirtorch resolves a lead device lazily, preferring cuda, then mps, then cpu.
On CUDA with two or more visible devices it then chooses a device count, and
that choice happens at `vcd_recon` entry rather than at construction.
`_apply_device_policy` is the one site, and §2.4 covers the entry points that
allocate across devices without reaching it.

Two reasons put the choice there.  The memory ledger needs a free-memory
reading, and that is only knowable when the reconstruction is about to start.
A developer calling a projector directly has not asked for a layout change.

The search runs in two stages.  First the candidate counts are ORDERED by the
widening speed floors.  Then each candidate in turn is priced by the memory
ledger against actual free memory, and the first that fits is taken.

The widening speed floors live in `mbirtorch/_widening_floors.py`.  A floor is
the problem size, measured in sinogram elements, at or above which a device
count is worth using.  Each count's floor is the size at which it overtakes
the best smaller ADMITTED count, not the size at which it overtakes one
device.  The floors as refreshed on 2026-08-11 against commit 4a222c7:

| family | count | floor (sinogram elements) | floor cell | measured against | margin |
|---|---|---|---|---|---|
| parallel | 2 | 88,080,384 | (512, 448, 384) | n=1 | 1.21x |
| parallel | 4 | 297,271,296 | (768, 672, 576) | n=2 | 1.10x |
| cone | 2 | 88,080,384 | (512, 448, 384) | n=1 | 1.21x |
| cone | 4 | 1,023,934,464 | (1024, 1008, 992) | n=2 | 1.45x |

The floors REORDER the candidates and never remove any.  Admitted counts come
first, largest first, then held counts, largest first.  Two consequences
follow.  Capacity always wins, because a held count is reached only after
every admitted count has been refused.  And `skip_memory_preflight` does not
disable the guard, because that flag makes the loop settle on its first
candidate, which the ordering has already made the first ADMITTED count.

Sinogram elements is the size metric, and measurement chose it.  On a
sparse-view problem whose sinogram elements and recon voxels point at
different sizes, widening to two devices was a 1.87x regression, which is what
the sinogram count predicts.

A geometry declares which floor family governs it through `_floor_family`.
Parallel and cone declare their own.  Translation and multiaxis declare none,
so the parallel floors govern them, and the run log says so at verbosity 2.
Their own floors have never been measured.

The floors are measurements, so they age.  The first guard consultation in a
process hashes the projection-cost inputs against a recorded list of blessed
hashes, and `stale_note` names whatever moved.  The note warns and the run
continues, because out-of-date numbers are a reason to re-measure rather than
a reason to stop a reconstruction.  `dev_scripts/refresh_widening_floors.py`
is the sole writer of the table.

The memory ledger is a closed-form per-device peak, phase by phase, for a
CANDIDATE device list.  Pricing a list the model is not currently in is what
lets the search evaluate a layout before installing it.  The ledger is
calibrated against measured peaks, and its calibration band is 1.00 to 1.30.
The mg11 gates read a smallest ratio of 1.003 and a largest of 1.158.

Two things pin the count and bypass the floors.  `configure_devices` takes
the layout out of the library's hands entirely, and `num_devices=1` is the
reproducibility pin.  The `MBIRTORCH_NUM_DEVICES` environment variable pins
the count process-wide, which is how every nightly row runs.  A pin bypasses
the floors by construction, so no nightly row exercises them.

### 2.3 What re-evaluation does, and what it does not do

The policy runs on every `vcd_recon` entry while the layout is still
automatic, and `recon` and `prox_map` both reach it.  Running is not the same
as changing anything.  `_settle` reinstalls the placements ONLY when the
chosen count differs from the current one.  An unchanged count costs one
closed-form ledger pass and one free-memory query per device, and it touches
nothing else.

A count that DOES change invalidates sharded data rather than moving it.
There is no re-shard path in the library.  `_install_device_layout` builds new
`Placement` objects and recreates the projectors, and `Shards` are bound to
their placement by OBJECT IDENTITY.  `_shard_sinogram` raises on a mismatch,
with the message that the shards belong to a different device configuration
and must be re-placed.  A caller holding sharded data therefore has to place
it again from the host.

Within one reconstruction the ordering is safe.  The policy settles before the
first large allocation, and `_shard_sinogram` and `_shard_recon` then place
onto the settled layout.  Nothing moves between devices mid-reconstruction on
account of the policy.

The exposure is a long-lived model across CALLS.  A Plug-and-Play loop through
`prox_map` re-enters the policy on every pass.  If free memory moves between
passes, because another process arrived on the GPU, the count can change
mid-loop and invalidate arrays the caller is holding.

### 2.4 Which entry points reach the policy

Choosing a device count and allocating across devices are different things,
and most entry points do only the second.  `_apply_device_policy` has exactly
one call site on the current tree, inside `vcd_recon`.  Everything else falls
into one of three groups.

Most entries INHERIT the model's current placement.  `direct_recon`,
`fbp_recon`, `fdk_recon`, `forward_project`, `back_project`,
`compute_hessian_diagonal`, `prepare_sino_for_devices`, and
`QGGMRFDenoiser.denoise` all shard through `_shard_sinogram` or
`_shard_recon`, and none of them chooses a count.  On a fresh model that
placement is trivial single-device.  So `model.fbp_recon(sinogram)` on a fresh
four-GPU model runs on ONE device, consulting neither the floors, nor the
ledger, nor the environment pin.

One entry chooses by its OWN rule.  `preprocess.scan_to_sino` takes all
visible CUDA devices, capped by `MBIRTORCH_NUM_DEVICES`, with no memory
preflight and no speed floors.  It is a second device-choosing site in the
package today, and its own comment records that the broader rule for
preprocessing is an open question.

One entry PINS by definition, which is `configure_devices`.

The systematic table is `active/entry_point_survey.md`, written 2026-08-10 for
Greg's uniformity ruling.  Two things have moved since it was written.  The
prerelease merge landed the `scan_to_sino` device choice the survey predicted.
And the environment-pin cap the survey flagged as missing has since been
added, so that entry now honors a pin even though it still ignores the floors
and the ledger.

### 2.5 The two policies side by side

| question | mbirjax | mbirtorch |
|---|---|---|
| when is the count chosen | at `set_devices`, on every recompile | at `vcd_recon` entry only, re-evaluated while automatic |
| what other entries do | all inherit the current layout | all inherit, except `scan_to_sino`, which chooses |
| default count | every available device | the widest count that is both admitted and fits |
| speed considered | no | yes, the widening floors |
| memory considered | no | yes, the closed-form ledger against free memory |
| what a non-dividing axis does | zero-padded, inert | zero-padded, inert |
| explicit door | `configure_devices(...)` | `configure_devices(...)` |
| process-wide pin | `MBIRJAX_NUM_CPU_DEVICES` (CPU only) | `MBIRTORCH_NUM_DEVICES` (any) |
| behavior when nothing fits | not checked; the allocator fails | `MemoryPreflightError` naming the dominant phase |

## 3. What crosses between devices

Under view and slice sharding the only data that crosses the recon-sinogram
boundary is voxel cylinders.  The sinogram is written locally on its
view-shard and never moves.  Both libraries build every crossing from one
transfer primitive, `move_shard`, which places a tensor on a target device
and routes through host memory if a hardware probe found direct copies
unsafe.

Three transfer SHAPES exist, and they differ in which axis of the cylinder is
cut.

**The slice band, broadcast.**  `broadcast_band_to_views` copies a band of
slices from its slice-owner to every view-owner.  Each view-owner then
forward-projects its OWN views from that band.  For a row-aligned geometry
the band produces exactly detector rows `[g0:g1)`, so each detector row has a
single producer and no reduce is needed.

**The slice band, reduced.**  `sum_band_to_owner` sums each view-owner's band
partial onto the band's slice-owner.  This is the back projection's
reduce-scatter, and it is the adjoint of the broadcast.  Both libraries use it
for all back projection at more than one device.

**The pixel column batch.**  A batch of pixel columns is gathered from every
slice-owner at EVERY slice onto one view-owner, which assembles those
columns' whole cylinder there.  What one gather costs is set by the width of
the column batch and not by the device count.  The gather changes which
device assembles which voxels, never which device produces which sinogram
rows, so it has no adjoint of its own and the back projection is untouched.

### 3.1 Which driver uses which shape

The two libraries assign these shapes to geometries differently, and on the
parallel forward they have traded places.

| direction | geometry | mbirjax | mbirtorch |
|---|---|---|---|
| forward | parallel | slice band, broadcast | **pixel column batch** |
| forward | cone | **pixel column batch** | **pixel column batch** |
| back | both | slice band, reduced | slice band, reduced |

In mbirjax the pixel column batch is the GEOMETRY-NEUTRAL default.
`_forward_project_to_view_shards` gathers one pixel batch's slices from every
slice-owner, runs the monolithic forward on that batch, and sums over
batches.  A geometry whose slices spread over a range of detector rows needs
every slice before it can produce any of its own rows, so it cannot stream
slice bands.  Cone therefore takes this default.  `ParallelBeamModel`
OVERRIDES the method with a banded forward that never gathers a full
cylinder, exploiting its detector-row-follows-slice identity.

In mbirtorch the pixel column batch is now the default for BOTH geometries,
and the two take it for different measured reasons.  Cone needs it.  Its
banded forward was measured flat in device count at 32.2, 30.6 and 30.5 s
over one, two and four devices, because a band-sized call still writes every
detector row.  Parallel merely wants it.  Its forward kernel runs about twice
as efficiently per slice on a full-width block of values as on the
shard-width blocks the banded walk hands it.  The measurement is 0.0411 ms
per slice on a 1008-wide block against 0.0823 ms on a 504-wide one, at one
device throughout, so the device count contributes nothing to the effect
(findings §1.9).

The banded walk remains in mbirtorch as the rollback.  Setting
`forward_column_gather = False` selects it, and the environment variable
`MBIRTORCH_FORWARD_COLUMN_GATHER` overrides the switch in either direction.
A geometry opts into the gather by declaring `column_gather_geometry`, which
parallel and cone do and translation and multiaxis do not.

### 3.2 What mbirtorch added around the gather

The column gather's copies overlap the projections that consume them, and two
mechanisms make that happen.  Each batch's gather is issued ONE BATCH AHEAD of
the projection that reads it.  On CUDA the copies run on a dedicated per-device
copy stream rather than the stream the device projects on.

Both mechanisms were needed, and the measurement says so.  Issuing early on
its own bought 0.8 to 2.4 s of wall across five configurations, against a
registered prediction of no change.  Adding the copy streams on top cut the
forward wall from 16.57 to 9.32 s at the widest-stall configuration, and the
composed reconstruction improved by 1.07x to 1.41x across the five (mg12,
findings §1.13).

The measurement also corrected an instrument.  GPU-busy time had been defined
as the event-bracketed time of the projection calls alone, and it fell along
with the wall, which a transfer change should not touch.  With one stream per
device, the copies serving OTHER devices' gathers interleave between a
device's own kernel launches, inside the bracketed windows.  Every busy
reading taken at more than one device on the gather path therefore carries
that contamination.  Conclusions built on one-device readings are untouched,
because a single device serves no peer copies.

Four orderings keep the overlap correct, and each covers a different way the
copies and the projections could interfere.  Both ends of every copy sit on a
copy stream, because torch issues a copy on the source's current stream and
orders the destination's current stream behind it.  Each copy stream waits
once for its device's compute stream before any copy starts, so a copy cannot
read a shard before the kernel that wrote it has finished.  Each batch carries
its own event, and the compute stream waits for that one event immediately
before the projection that reads that batch.  After the pass each compute
stream waits for its copy stream, so a later update cannot overwrite a shard a
copy is still reading.

### 3.3 The band reduce

`sum_band_to_owner` streams its sum in row slabs, and that is what bounds the
owner's peak.  Moving every partial across first and then summing them held n
whole bands on the owner at once.  Because one band is the whole shard by
default in mbirtorch, that transient did not shrink as devices were added.
Streaming leaves the owner holding its running total plus one bounded slab per
source.  The slab is 64 MiB, a reasoned default rather than a measured knee.

The summation order is unchanged by the streaming.  Streaming partitions the
elements, and no element's own sequence of additions is touched, so the result
is bit for bit what the unstreamed reduce produced.

mbirjax does not stream its reduce this way.  It bounds the same transient
through the band LENGTH instead, which §4.3 covers.

## 4. What runs inside one device

Two decisions govern per-device work: which implementation of a projection
runs, and how much work one call of it does.

### 4.1 Which implementation runs

mbirjax compiles everything through XLA, and adds Pallas kernels for parallel
and cone.  The Pallas paths are selected by four flags in the tile policy,
and the flags split by device count.  At one device `fwd_pallas` and
`back_pallas` serve the single-device drivers.  At more than one device
`fwd_pallas_band` and `back_pallas_band` serve the banded per-owner calls.
The split keeps each flag true only where its driver actually runs.  A
single-device back driver is never reached above one device, so an ungated
flag would report a kernel that did not run.

`_pallas_kernels.is_available()` gates all four.  It requires a GPU, an
allowlisted architecture, and a probe compile that actually runs.  An
unsupported toolchain silently keeps the XLA path.  Cone adds one more gate on
the forward.  Its kernel statically unrolls the row taps, so a geometry with
`bp_psf_radius` above 2 keeps XLA.

mbirtorch has two mechanisms rather than one, and they are independent.

`torch.compile` covers the torch bodies, and it is on by default
(`compile_mode='auto'`).  The compiled callables are cached per function and
per DEVICE INDEX.  The per-device split exists because compiled artifacts
carry Triton-launcher state that must not be shared across concurrently
executing threads.  A process-wide lock serializes the COMPILE events, because
inductor compilation is not thread-safe.  That lock is taken only for input
shapes a wrapper has not completed before, so steady-state threaded execution
stays lock-free.  A compile failure retries the call eagerly, records the
error, and rebinds to eager permanently.

Hand-written Triton kernels cover parallel and cone, forward and back, and
all four are on by default.  A kernel body carries `_mbirtorch_no_compile`, so
`maybe_compile` returns it unwrapped whatever `compile_mode` says.  The marker
is needed because `torch.compile` unwraps a `torch.compiler.disable`
decorator and traces the original function, launch and all.

Two gates admit a Triton kernel, and they ask different questions.  A
once-per-process probe compiles and runs a trivial kernel end to end, which
tests the toolchain rather than a version number.  Then a per-kernel,
per-DEVICE self-check runs one kernel-against-torch-body comparison at a tiny
shape on the device that will run it, and falls back on a tolerance breach at
1e-4 relative.  The second gate is the one that catches a miscompiling
toolchain on an architecture that passed the first.
`MBIRTORCH_DISABLE_TRITON=1` is the kill switch for all four.

Kernel selection is layout-independent in both libraries, but they arrived
there differently.  mbirjax splits its flags by device count because the
DRIVERS differ.  mbirtorch once withheld its forward kernel from sharded
layouts, and that interim was retired.  The defect was the LAUNCH rather than
the kernel: a Triton launch targets the launching thread's current CUDA
device, and the per-device workers launch from threads sitting on device 0.
The wrappers now bracket their launches with `torch.cuda.device(...)`.

Translation and multiaxis have no hand-written kernels in either library.
They run compiled torch bodies in mbirtorch and XLA in mbirjax.

### 4.2 How many views one call takes

Both libraries batch a projection over views, and both cap the batch so one
call's transient stays bounded.  They differ in where the cap comes from.

mbirjax sets the view batch in the tile policy, from the per-device view
shard and a constant cap.  Forward and back want OPPOSITE policies, and the
policy records why.  The forward's transient scales with the batch width per
device, and view sharding does not shrink it, so the forward keeps a flat
OOM-safe width of 128 clipped to the shard.  The back single-vmaps its whole
per-device view shard, because a smaller batch drops into an accumulating
scan whose live carry inflates the peak.  The back cap is 128 unsharded and
512 sharded.

mbirtorch computes the view batch PER CALL, from a transient budget and a
per-body cost model.  The budget is 2 GiB on CPU.  On a device backend it
scales with the problem: eight times the per-device sinogram bytes, capped at
2 GiB and floored at 256 MiB.  Deriving it from the per-device shard is what
keeps a flat budget from letting a small problem hold a transient many times
the size of its own sinogram.

The cost model is per body, which is the point of the design.  A torch body
charges `num_pixels * cols * 4` bytes per view, the width of its gather slab.
A hand-written kernel body carries a `_view_batch_cost` attribute returning
its own charged bytes per view and its own swept nominal chunk.  All four
Triton kernels declare a nominal chunk of 128 views.  The parallel kernels
charge 16 bytes per (view, pixel) plus one channel-major plane.  The cone
kernels charge 48 bytes per (view, pixel) plus a plane spanning the FULL
detector rows, because a cone call writes every row whatever slice band it
was handed.

One function serves two consumers, deliberately.  `view_batch_charge` returns
both the batch and the bytes per view.  The driver consumes the batch, and the
memory ledger consumes both, so the size the code moves and the size the model
charges cannot drift apart.

The realized batches at the 1024-class cell are 128 on parallel and 52 on
cone.  Cone's is capped by the transient budget rather than by the nominal
chunk.  A probe confirmed that the budget falls with the device count while
cone's per-view cost does not, so cone's batch drops at four devices where the
falling cap crosses the chunk.  The measured cost of that effect is 1.2
percent of summed launches, so no knob change was warranted (findings §1.6).

### 4.3 How the slice axis is cut

The two libraries take OPPOSITE defaults on the slice band, and both defaults
are measured.

mbirjax streams by default.  `_slice_band_length` returns the smaller of two
upper bounds, floored below.  The reduce-gather bound is `slices_per_dev /
n_dev`, which pins the n-way gather transient to about one output slice
shard, so the per-device peak tracks the shrinking shard instead of
plateauing.  The compute bound is `100,000,000 / num_pixels`, which is what
makes a SINGLE device stream and took a 1024-class single-GPU back projection
from about 28 GB to about 10 GB at no time cost.  The floor is `4,000,000 /
num_pixels`, which keeps small recons from splitting into many tiny calls.
Sweeps found time essentially flat across the band on GPU, so a smaller band
is close to a free memory win.

mbirtorch uses one band per slice-owner by default, which is the whole shard.
The rationale is recorded in its `_slice_band_length` and rests on two facts.
The torch banded pass pays a fixed orchestration cost per band, measured at 2
to 23 percent more busy time when the shard was split, depending on the walk.
And time buys nothing back at one device, because a single torch device never
runs the banded drivers at all.

The band survives in mbirtorch as a MEMORY lever, not a time lever.  The mg10
sweep read per-device peaks of 11.84 to 11.97 GB across sub-band walks against
12.48 GB at the default, with total copied bytes unchanged.  That sweep
predates the streamed reduce of §3.3, so the current gap should be narrower.
`forward_project_slice_band` and `back_project_slice_band` opt in with a fixed
band when a run is memory-constrained.

Under the column gather the forward bands nothing at all.  What bounds the
transfer there is the pixel batch, and the memory ledger stops charging the
band copy to match.

### 4.4 The pixel batch

Both libraries batch the gathered forward over pixel columns, and the two
libraries scope the constant differently.

mbirjax's `fwd_pixel_batch` serves two roles at once.  It sets the pixel scan
inside the plain per-call forward, and it sets the cross-device batch of the
gather forward.  Its base value is `_PIXEL_BATCH_DEFAULT = 2048`.  Parallel
raises it to 8192 on GPU, and cone raises it to 4096 on GPU above 768 slices.
Parallel's value never reaches a cross-device gather, because parallel
overrides the gather forward away.

mbirtorch's `FORWARD_PIXEL_BATCH = 8192` has the second role only.  It is the
column gather's cross-device batch, and nothing else reads it.  So the
cross-device batch actually in use is 4096 in mbirjax cone and 8192 in
mbirtorch cone.

mbirtorch's 8192 is deliberately below the best measured value.  Composed wall
kept improving through 16384 and 32768 at the 1024-class cell, by 4 to 15
percent over 8192, so the knee is not bracketed.  Production runs at the 2K
class and above, where the batch's cross-device transient grows with the slice
axis, and no sweep has run there.  The default therefore stays at 8192 and the
2K sweep belongs to the production-scale work.

## 5. Measured vcd_recon on gautschi

Two rulers measured these numbers, and they disagree.  This section reports
both and never blends them.

The CAMPAIGN ruler is warm seeded 3-iteration vcd, medians of three repeats,
one H100 node, with same-run jax rulers where both frameworks appear.  The
NIGHTLY ruler is the `vcd_nonconst` regression row.  On the same cell the two
rulers differ by up to 15 percent.  jax parallel 1024 at one device reads
25.80 s under the campaign ruler and 22.45 s under the nightly ruler.

### 5.1 The 1024-class cell, current best

The cell is sinogram (1024, 1008, 992).  Memory is the largest per-device
peak.  The mbirtorch column is the current tip and comes from three jobs, so
§5.3 names the provenance of every entry.

| geometry | devices | mbirtorch time | mbirtorch peak | mbirjax time | mbirjax peak |
|---|---|---|---|---|---|
| parallel | 1 | 39.90 s | 22.87 GB | 25.80 s | 49.81 GB |
| parallel | 2 | 27.83 s | 12.82 GB | 14.33 s | 19.71 GB |
| parallel | 4 | 17.55 s | 7.31 GB | 11.52 s | 14.73 GB |
| cone | 1 | 61.66 s | 22.95 GB | 62.75 s | 48.45 GB |
| cone | 2 | 54.86 s | 12.84 GB | 43.37 s | 21.52 GB |
| cone | 4 | 32.37 s | 7.31 GB | 25.78 s | 12.23 GB |

Three readings carry the table.  mbirtorch is slower than mbirjax at every
multi-device cell here: 1.94x and 1.52x on parallel at two and four devices,
and 1.26x and 1.26x on cone.  mbirtorch holds less memory at every cell, at
0.46x to 0.47x of mbirjax's peak at one device and 0.50x to 0.65x at the
multi-device cells.  Both libraries shard memory cleanly, and mbirtorch's
four-device peak is 0.32x of its own one-device peak.

The mbirtorch scaling improved substantially this week and the mbirjax column
did not move.  mbirtorch parallel at four devices ran 23.36 s on 2026-08-09
and runs 17.55 s now, and cone at four ran 53.10 s and runs 32.37 s.  Cone at
two devices was a REGRESSION against one device on 2026-08-09, at 67.23 s
against 61.57 s, and it is now a 1.12x gain.

### 5.2 The 512-class cell

The cell is sinogram (512, 448, 384).  These rows are the campaign ruler of
2026-08-09 and predate the column gather and the copy streams, so the
mbirtorch entries are stale in the improving direction.  No post-flip
measurement of this cell exists.

| geometry | devices | mbirtorch time | mbirtorch peak | mbirjax time | mbirjax peak |
|---|---|---|---|---|---|
| parallel | 1 | 1.91 s | 1.93 GB | 1.66 s | 3.89 GB |
| parallel | 2 | 1.57 s | 1.11 GB | 1.98 s | 2.14 GB |
| parallel | 4 | 2.52 s | 0.65 GB | 3.12 s | 1.66 GB |
| cone | 1 | 2.74 s | 2.15 GB | 3.07 s | 3.96 GB |
| cone | 2 | 2.78 s | 1.70 GB | 2.93 s | 2.15 GB |
| cone | 4 | 4.07 s | 1.07 GB | 3.76 s | 1.33 GB |

One reading carries this table.  Neither library gains from four devices at
this size, and mbirjax loses more than mbirtorch does.  This is the size range
the widening floors govern, and both floor families admit two devices exactly
here.

### 5.3 Provenance

Every mbirtorch number in §5.1 comes from one of three jobs, all on H100
nodes.

The one-device rows are mg11, job 15163071, 2026-08-11, on commit a33c7e8.
The two- and four-device rows are mg12, job 15175187, 2026-08-11, on the
current tip with the copy streams in place, at pixel batch 8192.  The
512-class rows and every mbirjax row are mg1, job 15011662, 2026-08-09.

One caveat applies to the one-device rows.  They were measured before the
column-gather default flip, and a single device should not be affected,
because a trivial placement short-circuits to the plain projectors and never
enters the gather.  That is an argument from the code and not a measurement.
Since every scaling ratio in §5.1 divides by the one-device row, §7 carries
re-measuring it as an open item.

The mbirjax rows come from a jax version and branch that has not moved during
the campaign, and mg1's n=1 ratios reproduced the earlier kb3 baselines to the
printed digit.  For reference, the mbirjax nightly of 2026-08-08 on the
prerelease branch reads parallel 1024 at 22.45, 10.44 and 8.32 s and cone 1024
at 61.75, 43.90 and 27.22 s, under the other ruler.

## 6. The tile policy

mbirjax has a tile policy and mbirtorch does not.  This section states what
the object is, what mbirtorch uses instead, and what should change.

### 6.1 What mbirjax's TilePolicy is

`TilePolicy` is an immutable namedtuple holding every batching knob and every
kernel-algorithm flag the projectors consume.  It has twelve fields:
`fwd_view_batch`, `back_view_batch`, `fwd_pixel_batch`, `back_pixel_batch`,
`fwd_slice_band`, `back_slice_band`, the four Pallas flags, `sort_by_channel`,
and `back_stacked_gather`.

Three properties make it useful, and each addresses a specific failure.

It has ONE decision site.  `_select_tile_policy` is the only place a value is
chosen, and it re-runs on every device re-layout.  A geometry class overrides
the method to change only what it has measured, and the base policy holds the
long-standing defaults.

It binds LATE.  Consumers read `self.tiles` at call time, so a
`configure_devices` re-layout takes effect on the next projection without
recreating the projectors.  One field is the exception.  `sort_by_channel` is
baked into the static projector params, and a stale value there can cost
speed but never correctness, because the two reductions are value-equal.

It is IMMUTABLE.  An experiment overrides a field with
`model.tiles = model.tiles._replace(...)` rather than mutating scattered
attributes, and the old per-attribute setters raise with a message pointing at
the replacement.

### 6.2 What mbirtorch has instead

mbirtorch's `projectors.py` states the omission deliberately: no sorted
channel reduction, no stacked gather, no tile policy, described there as the
jax performance layer.  The knobs that do exist sit in five places.

The five sites are these: `FORWARD_PIXEL_BATCH`, a module constant in
`tomography_model.py`; the `VIEW_BATCH_*` class attributes on the projector
class; `_slice_band_length`, a static method; `_column_gather_forward`, a
method reading an environment variable and an instance attribute; and
`_view_batch_bodies`, a per-geometry method returning the two bodies.

The knobs also differ in KIND from mbirjax's, and this is the substantive
point.  mbirjax's view batch is derived from the device layout, so a value
chosen once per layout is exactly right.  mbirtorch's view batch is derived
per CALL, from the actual pixel count, the actual band width, and the body's
own cost function.  A namedtuple selected once per layout cannot express that.
A straight port of `TilePolicy` would therefore be wrong.

mbirtorch already applies the anti-drift discipline that motivated
`TilePolicy`, through a different mechanism.  The memory ledger calls
`view_batch_charge` and `_forward_pixel_batch` rather than re-deriving either
number, so a changed default cannot leave the charge behind.  That is the same
guarantee, enforced by a shared function instead of by a shared record.

### 6.3 Recommendation

Adopt the DECISION-SITE discipline for the layout-derived knobs, and leave the
per-call view batch where it is.

Four settings in mbirtorch do not depend on the shape of an individual call,
and it is those four that a decision site could hold: which bodies run,
whether the column gather is on, the forward pixel batch, and the slice band.
Each of the four is fixed once the geometry and the device layout are known.
Collecting them into one method that re-runs on `_install_device_layout` would
buy three things.  A reader would have one place to look.  The run log and
`last_memory_ledger` could report the selected values as a unit.  And an
experiment could override one setting without knowing which of the five sites
holds it.

Two things argue against doing it now.  The forward path is still moving:
increment 6 of the forward remedy is open, the pixel-batch knee is not
bracketed above 8192, and the 2K sweep has not run.  Freezing a record around
knobs that are about to change would cost a second refactor.  And the current
scatter has not yet caused a measured defect, so the case rests on
maintainability rather than on a failure.

The recommendation is therefore to defer, and to attach the collection to the
close-out of the multi-GPU campaign rather than to open a separate item.  If a
knob moves before then, that is the moment to reconsider, because the cost of
the scatter is paid on each move.

## 7. Open questions this page surfaced

These are questions the reading raised, each with what would answer it.  None
is a finding.

**The mbirtorch one-device anchor has not been measured since the flip.**
Every scaling ratio in §5.1 divides by it.  The code argument says a trivial
placement never enters the gather, and that argument has not been checked
against the current tip.  One arm per geometry answers it.

**mbirjax's automatic device count has no speed rule.**  The nightly reads
0.21x at the 200-class cell and 0.47x at the 512-class cell for four devices
on parallel.  Those are the default choices on a four-GPU node.  Whether to
act depends on whether mbirjax gets further development, which is a separate
decision; `current_plans.md` item 11 already carries a narrower version of
this as the `_auto_device_count` basis question.

**No post-flip measurement of the 512-class cell exists.**  §5.2 is entirely
pre-flip on the mbirtorch side.  That cell is where both floor families place
their two-device admission, so the floors were refreshed against it while the
comparison table was not.

**Translation and multiaxis have no floors of their own.**  They inherit the
parallel floors, which are the more permissive measured set, and neither has
ever been timed multi-device.  This is already recorded as an obligation in
`multigpu_plan.md` step 1.

**The band reduce slab is a reasoned default, not a measured knee.**  64 MiB
was chosen by argument about launch overhead against transfer cost.  A sweep
has not run.

**`usr_multi_gpu.rst` does not say which entry points spread and which do
not.**  Two passages read as though `fbp_recon` and `fdk_recon` participate in
the automatic spread, and §2.4 shows they inherit instead.  The sharper of the
two is the tip that recommends `prepare_sino_for_devices` for repeated
reconstructions.  That method inherits, so on a fresh model it prepares onto
one device, and the first `recon` can then widen the count and invalidate the
prepared array.  The method's own docstring states that consequence plainly.
The user page's tip does not, and the tip is where the workflow is read.  The
page also does not say that preprocessing chooses its own count by a different
rule.  This belongs with the `usr_multi_gpu.rst` pass already scheduled in
`multigpu_plan.md` step 7.
