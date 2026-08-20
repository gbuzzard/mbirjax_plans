# Multi-device speed, part 2: the torch-body geometries

Written 2026-08-19.  The first multi-device campaign closed with the
parallel and cone work recorded in `multigpu_plan.md` and
`multigpu_findings.md`.  This document holds what comes next for the two
geometries that run compiled torch bodies instead of hand-written
kernels: multiaxis and translation.  The component split (findings
§1.36) found the mechanism behind their two-device losses, and the first
remedy landed the same day.  What remains here is recorded, not
scheduled: two structural alternatives to the landed remedy, and the
hand-written-kernel escalation path with its entry gates.

## What landed: the raised recompile budget

torch caps the compiled variants of one function at 8, the cap attaches
to the function's code object, and the per-device compiled instances
share that code object.  The compiled variants guard on the input
tensors' device index, so a run on n devices needs about n times the
one-device variant count.  Where a body's budget filled, its remaining
calls ran eagerly, and that eager back projection was the whole measured
two-device loss of both geometries.

The remedy raises the per-function budget to 64
(`mbirtorch/projectors.py`, `_RECOMPILE_LIMIT_FLOOR`, with the
arithmetic in its comment).  `MBIRTORCH_RECOMPILE_LIMIT` overrides the
floor verbatim, downward included, as the debugging escape.  The
accumulated cross-function limit (torch default 256) is left alone as
the backstop against unbounded variant growth.

The remedy needed two forms, and the difference is a torch behavior
worth recording on its own.  The first form raised the budget once,
where the compiled wrapper is created, on the thread that creates it.
Its gate (mg45, job 15394465) read every wall unchanged and torch's
warning still reported the default limit.  The reason is that dynamo
consults a PER-THREAD view of this config: an assignment made on one
thread does not reach another.  This was measured three ways.  A limit
assigned on the main thread capped nothing when the compiled function
was called from a worker thread; the same assignment made on the worker
thread capped it; and a tracer on the config module's setter, run on
the cluster (mg46, job 15394591), showed the raises firing with no
assignment ever setting the default back, while the warning read the
default -- the converting pool thread simply never saw the writes.  The
per-device fan-outs run the compiled bodies on pool threads, which is
also why one device never tripped: at one device the fan-out
short-circuits onto the main thread.

The second form raises the budget inside `maybe_compile`'s wrapper, on
each first sight of an input shape -- which is on the calling thread,
under the compile lock, before any call that can compile.

The rerun gate passed (job 15394667, six minutes).  Its log carries no
recompile-limit warning, and both cells that lost at two devices now
win there:

| cell | n1 warm | n2 warm, before | n2 warm, after | two-device ratio |
| --- | --- | --- | --- | --- |
| multiaxis (512, 448, 384) | 11.40 s | 32.6 s | 7.47 s | 0.35x -> 1.53x |
| translation (256, 1900, 3000) | 12.57 s | 14.2 s | 10.04 s | 0.89x -> 1.25x |

The component split confirms the mechanism is gone rather than merely
thinner: the 512-class back projection reads 2.34 s at two devices
against 3.01 s at one, where the eager form read 16.5 s, and the
Hessian phase reads 1.14 s against its eager 12.1 s.  The one-device
walls are unchanged, which is the mechanism's own prediction.  The
finding is multigpu_findings.md §1.37; the gate rows are
`rows/mg44_component_h012_20260819_230041.jsonl` (the failed first
form) and `rows/mg44_component_h007_20260819_232038.jsonl` (the
passing second form).

The large-cell confirmation followed the same night (mg47, job
15398646, 2026-08-20).  The 1024-class flips as well: its two-device
warm wall drops from the recorded 388.8 s to 203.6 s against an
unchanged 308.6 s at one device, a 0.80x loss become a 1.52x win, with
the back projection at 65.7 s against 91.6 s where the eager form read
250 s.  The 768-class reproduces at 1.45x (recorded 1.46x), so the
raised budget cost the one previously winning cell nothing.  Every
cell of the old two-device window is now a win: 1.53x, 1.45x, 1.52x
at the 512-, 768-, and 1024-class, and translation 1.25x at
production.  The full-refresh verdicts and the sentinel rulings are
what remain.

The follow-ups all completed on 2026-08-20.  The 1024-class pair
flipped (0.80x to 1.52x) and the 768-class reproduced (mg47).  The
floors refresh ran as the FULL refresh -- the remedy changed a shared
cost input, so the tool refused any narrower scope -- and cleared all
four torch-body sentinels while the kernel families and the denoiser
reproduced (mg48, findings §1.38).  Greg accepted the refreshed table
the same day; the paste, the blessed constants, and the re-pinned
behavior tests are in the library.  Multiaxis widens from the
512-class and translation from its measured cells; the old n=4
readings (0.23x to 0.87x) are now the table's largest wins (above
2x).

## Approach 2, kept in reserve: a budget per device instance

Give each per-device compiled instance its own copy of the body
function, so torch's per-function budget applies per device.  A closure
or a `functools.partial` does not do this: both share the underlying
code object, and the budget follows the code object.  The instance
would need a genuine clone (a copied function object with a copied code
object), built inside `maybe_compile` where `instance_key` already
names the instance.

This is the structurally cleanest form: device variants can never
collide, each device needs only the one-device variant count, and the
scaling to any device count is by construction.  It changes no global
state, which is its advantage over the landed remedy.  Its cost is the
cloning machinery, which is unusual enough to need a careful comment
and a test.  Adopt it if the raised global budget proves wrong for
someone: a workload where 64 is too small, or an application that
needs torch's default semantics untouched.

## Approach 3, kept in reserve: pre-marked dynamic dimensions

Tell torch at compile time which dimensions vary across calls (the
pixel count, the view batch, the band length), so it compiles
shape-flexible variants up front instead of discovering flexibility by
recompiling.  Today each discovery burns budget; pre-marking skips
those recompiles and shortens the cold pass.

Two limits keep this in reserve rather than first.  The device-index
guard remains, so the device multiplier on the variant count remains,
and four devices would still not fit in a budget of 8 by this route
alone.  And shape-flexible compiled code can run slower than
specialized code, so each body would need its own measured gate.  Worth
riding along if cold-pass compile time ever becomes a problem, or as
hygiene during kernel work.

## The denoiser, and the plug-and-play composition

The denoiser is the third model with no hand-written kernels, and its two
floor rows are sentinels: no admission size has ever been measured, so the
automatic path never widens a denoise for speed.  Two runs on 2026-08-20
settled why, and the answer moved the question somewhere else entirely
(findings §1.39).

The ladder question is closed.  A four-cell ladder from the 1024-class to
the 1664-class measured a FLAT ratio of about 0.65 at two devices and 0.60
at four, across a 4.4 times range in volume.  Capacity takes the widening
decision at the 1792-class, so that ladder covers every size at which the
speed sentinel has any effect.  There is no admission size to find, and
the sentinel rows are correct for what they govern.

What they govern is the narrower point.  The measured penalty is almost
entirely the OUTPUT GATHER, not the computation: at the 1024-class the
whole one-to-two-device penalty is 1.143 s, of which the gather is
1.068 s.  Left on the devices, a denoise costs about the same on four
devices as on one, and at the 1664-class on four devices it is at parity.
So the sentinel is right about the default call, which gathers, and
misleading for a caller that passes `output_sharded=True`.

That caller is the plug-and-play or ADMM loop, which alternates
`prox_map` with `denoise` on a volume it wants to keep resident.  Three
things blocked that composition until 2026-08-20, all verified on virtual
CPU devices: `_shard_recon` and `_shard_sinogram` compared placements by
object identity, so two models configured alike could not exchange
shards; `vcd_recon` reached `prox_input` through `.shape` and `.reshape`,
which a `Shards` container does not have; and there was no way to ask one
model for another's device configuration.  The change that unblocks it
landed the same day (staged): `Placement` gained value equality, the
`prox_input` path gained a `Shards` branch, and `configure_devices`
gained `like=other_model`, which copies the other model's device list
after checking that the two recon shapes agree.  A three-iteration
alternation now runs on two virtual CPU devices with every hop staying
in the device form.

The denoiser's own input gather was the next transfer, and it went the
same day.  `denoise` used to bring a sharded input to the host whole,
for the noise and regularization statistics, even when the caller
supplied `sigma_noise`.  Neither statistic needs the volume: the
regularization parameters reduce immediately to `image[::step]`, which
is 21 rows of 1024, and the noise estimate strides to at most five
million points.  Both subsamples are now assembled from the shards
directly.  Rows and columns are not sharded, so every shard strides
identically there; on the sharded axis, shard k's sampled positions
begin at local offset `(-start_k) % stride`, so concatenating the
per-shard blocks reproduces the strided volume exactly.  That
exactness is a data-movement identity, and the tests gate it with
exact equality rather than a tolerance.  The statistics path now moves
about 2.5 percent of the volume at the 1024-class and 1.4 percent at
the 1664-class, and no full gather runs at all.

One transfer remains, and it is the sinogram side, which has the same
blocker the volume side lost: `vcd_recon` reads `sinogram.shape` and
`initialize_recon` calls `np.asarray` on it, so a `Shards` sinogram
cannot be passed to `recon` or `prox_map` even though
`prepare_sino_for_devices` returns one.  A loop therefore re-places its
sinogram from the host on every `prox_map` call.  That is open item D9.

A note for the next floors refresh.  These changes touched
`_sharding.py`, which prices every family, and `denoising.py`, which
prices the denoiser, so the shipped table now reports every family
stale.  The staleness note is a true statement about the file hashes
and a false alarm about cost: the placement change adds comparison
methods that no timed path calls, and the denoiser change alters only
what a SHARDED input does, while the floors protocol passes a host
array and therefore runs the same work it always did.  The hashes were
deliberately NOT re-recorded, because recording them without
re-measuring is what hides a real change from the next reader.  Whoever
runs the next refresh can either measure and paste as usual, or rule
the change cost-neutral and bless it; the reasoning above is what that
ruling would rest on.

The guidance that follows is worth stating plainly, because it runs
against the shipped table.  In a plug-and-play loop, configure the
denoiser to match the reconstruction model and keep the volume on the
devices.  The floors row says one device; it is answering a question
about a call that gathers, and the loop never does.

One optimization candidate is recorded, and it is not the sweep.
`Shards.gather` brings every shard to the host and concatenates them on
the sharded axis, which for a recon-like array is the LAST axis: the
least favorable memory locality, plus a second full-size host
allocation.  A single-device gather is one contiguous copy with no
concatenate, which is why sharding roughly doubles the cost.  The
candidate is to allocate the host array once and copy each shard into
its own slice.  The gather is 43 to 69 percent of a denoise call at
every device count, so this is the largest single line in a denoise; the
same gather ends every multi-device reconstruction, where two seconds
against a wall of minutes does not matter.

## The escalation path: hand-written kernels for multiaxis and translation

The projectors module has anticipated this path since the layer design
("their torch analogs belong with the future Triton kernel work").
Three things argue for it, each with recorded evidence.

First, structural immunity.  A hand-written kernel body carries the
no-compile marker, so torch's compiler is never involved: no guards, no
budgets, no eager fallback, no launch-shape retraces.  The whole
mechanism class of findings §1.36 becomes impossible rather than tuned
around.

Second, memory, which is the strongest argument.  The torch bodies
build (view batch, pixels, width) gather intermediates; the memory
ledger charges them 14 slabs and both live blocks where a kernel body
pays one (findings §1.22's correction).  The measured consequences:
sharding barely shrinks a torch-body geometry's per-device peak (the
half-scale translation peak grew from 8.2 to 27.2 GB between one and
four devices), and multiaxis 1024 models at 68 GB on one device, at an
H100's edge.  Kernels cut the transients, let sharding actually divide
the peak, and open both capacity and the widening decision.

Third, speed headroom.  At one device the compiled multiaxis forward is
enqueue-paced: at the 1024-class its host time (139 s) exceeds its
device time (126 s), so the device idles between the many small
compiled-graph launches (mg44's tables).  A kernel is one launch per
view batch.  The parallel kernel history says the ceiling can be high
when the access pattern is exploited (the channel-sorted forward read
3.45x).  The caveat gets its own sentence: no cross-framework anchor
exists for these geometries, so the multiplier is unmeasured.

The cost yardstick is the cone and parallel kernel campaign: forward
and back kernels per geometry, adjoint agreement, goldens, the width
rule, and floors-refresh scope, with the recorded traps (divisibility
specialization, data-dependent launch shapes, width padding).
Multiaxis is geometrically richer than parallel: two angles per view,
with the elevation coupling slices to detector rows.  Translation is a
different geometry again.  Call it two campaigns, smaller than the
first only because the infrastructure now exists.

Two entry gates come before any kernel design, and both are cheap:

1. A counter run on the multiaxis bodies (the mg25 harness pointed at
   them), naming what the compiled bodies are bound on.
2. A cross-framework anchor: multiaxis and translation timed against
   mbirjax at matched cells, which turns "kernels would probably help"
   into a number.

The decision rule is need, not elegance: Charlie's release scope and
actual users.  If large multiaxis reconstruction (above the 1024-class)
is ever required, kernels stop being a speed preference and become a
capacity necessity, because of the 68 GB single-device wall.
