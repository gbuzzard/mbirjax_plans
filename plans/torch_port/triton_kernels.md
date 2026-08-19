# The hand-written GPU kernels: how they work and what they cost

**Written 2026-08-18.**  This document describes mbirtorch's four
hand-written GPU routines -- the forward and back projections for
parallel beam and cone beam -- for a reader who knows the mathematics
but not GPU programming.  Every number is cited to the finding and job
that measured it, and the tables mark what has never been measured.
The sorted parallel forward described in §3.1 passed its full
composed gate on 2026-08-18 -- the GPU suite, the A/B at the 512- and
1024-class cells, and the 2048-class confirmation (findings §1.31) --
and awaits its commit.

A "kernel" here is one GPU routine, written in Triton, that the
library binds in place of its general torch implementation of the same
computation.  The two implementations take the same inputs and return
the same shapes, so everything above them -- the drivers, the
multi-device machinery, the memory ledger -- does not know which one
ran.  The kernels exist because they are faster: each earned its default
at a composed performance gate against the compiled torch body it
replaces, at the 1.9x class and above, and the selection comments in
the code carry each gate's figures.

## 1. The vocabulary, in plain words

A GPU runs a kernel as tens of thousands of small independent **work
units** (the hardware calls them programs or thread blocks).  Each
work unit is assigned a **tile** of the problem -- for these kernels,
some pixels by some columns -- named by its position on a **grid**.
Issuing the grid is a **launch**.  One launch here typically covers a
batch of views at once; the **view batch** is how many, and a **view
chunk** is a smaller group of views that one work unit walks in a
loop.

Memory comes in a ladder.  Each work unit has a small amount of
private, immediately-usable storage (**registers**); each processor
has a small fast cache (**L1**, a few hundred kilobytes); the chip has
one shared cache (**L2**, about 50 MB on the H100 this project
measures on); and everything else lives in **main memory** (80 GB on
an H100), which is large but far slower to reach.  Data moves in
fixed 32-byte packets, and an access pattern is **coalesced** when it
fills the packets it touches -- eight float32 numbers per packet is
perfect.

When many work units add into the same output location, they use a
**collision-safe add** (an atomic add): the hardware serializes the
additions so none is lost.  These adds are correct but they all pass
through shared machinery, and this document will show that machinery
becoming the limiting resource and then being relieved.

Finally, the counter runs classify each kernel by which resource is
busy.  A **compute-side** kernel keeps the arithmetic units busy and
its memory system quiet; a **memory-side** kernel is the reverse, and
its work units spend their time waiting for data.  The profiler
reports both as a percentage of the hardware's peak, alongside a
count of how many warps (groups of 32 threads) are parked waiting at
any moment.

## 2. The data and its shapes

Four arrays flow through every projection call.

* **The values block** is the reconstruction's content, arranged one
  row per **pixel** -- and a pixel here is a whole voxel *cylinder*:
  one (row, column) position of the volume with all of its slices.
  The block is (pixels, slices) float32.  Under parallel beam a
  cylinder's slice k lands on detector row k at every view, which is
  the invariance much of §3 exploits; under cone beam the landing row
  depends on the view.
* **The geometry contract** is a small set of per-(view, pixel)
  arrays the horizontal fan produces: each pixel's detector **channel
  center** at that view, and the trapezoid weights around it.  Cone
  adds the vertical fan's terms.  The contract is rebuilt per call by
  an eager builder outside the kernel.
* **The sinogram** is (views, rows, channels) float32.  The kernels
  work on a **channel-major** copy or output, (views, channels,
  rows), so that the row axis -- the axis a tile sweeps -- is
  contiguous in memory and the accesses coalesce.
* **The pixel list** says which cylinders a call covers: the full
  region-of-reconstruction mask, or one of VCD's random subsets.
  The distinction matters because consecutive entries of the full
  mask are geometric neighbors and consecutive entries of a random
  subset are not -- the fact that drove the sorting design in §3.1.

The running example is the 1024-class cell:

| array | shape | size |
|---|---|---|
| sinogram | (1024, 1008, 992) | 3.8 GiB |
| reconstruction | (992, 992, 1008) | 3.7 GiB |
| values block, full mask | (771,240, 1008) | 2.9 GiB |
| contract, one 128-view batch | 2 to 4 arrays of (128, 771,240) | 0.4 GiB each |
| pixel list, one VCD subset | (12,051,) | small |

## 3. The four kernels

Each subsection follows one template: what the kernel computes, how
the work is subdivided, where the memory traffic goes, and what
limits it.  Code pointers are collected in §8.

### 3.1 The parallel forward: sort, contract, add once

**What it computes.**  For each view, each pixel spreads its cylinder
into the sinogram: every slice of the cylinder adds into the matching
detector row, at the two or three channels the trapezoid weights
name, scaled by those weights.

**How the work is subdivided.**  The launch grid is (pixel tiles,
column tiles, view chunks): each work unit owns 32 pixels by up to
128 columns and walks a 16-view chunk in a loop.  Before the kernel
runs, the wrapper sorts each view's pixels by their channel center
and gathers the contract into that order; the kernel reaches the
values rows through the per-view permutation.  The sort is per view
because a pixel's channel center depends on the view angle, so one
ordering cannot serve a whole batch.

**Where the sorting and the matrix multiplication come in.**  Sorted,
a tile's 32 pixels land in a span of only two or three channels on
the full mask, and about eighteen at the widest.  The tile's
trapezoid weights then form a narrow banded matrix W, one row per
pixel, one column per channel of a 16-channel window.  The tile's
whole contribution to the window is one small dense product,
transpose(W) times the tile's values -- a matrix multiply the GPU's
tensor cores execute -- and where many pixels share a channel the
multiply sums them, which is a segmented reduction.  The window then
lands in the sinogram with ONE collision-safe add per (channel,
column), in place of one per (pixel, tap, column).  The multiply
runs in the full-precision input mode, because the tensor-core
default rounds its inputs to a 10-bit mantissa and would fail the
1e-5 value gates.

Computing the product densely, zeros included, is a deliberate trade.
The counter runs showed this kernel's arithmetic units 84 percent
idle while the collision-safe add machinery was the bottleneck;
the multiply spends the idle resource to remove load from the
congested one.  Measured, the adds fell 31.6x and the kernel ran
3.97x faster at the full mask (findings §1.30).

**The guard.**  A tile whose sorted span still exceeds the window --
a sparse pixel list is the ordinary cause -- takes the original
per-tap path for that tile, so correctness never rests on the span.
Only speed does.

**What limits it now.**  After the sort, the kernel's memory system
spends its budget on useful traffic: 82 percent of peak memory
throughput, 58 percent arithmetic, and 2.6 warps waiting where the
per-tap kernel had 38 (mg33's counter reading).  The sort itself
costs 14.6 ms at the full mask and under 4 ms at subsets, against
savings tens to hundreds of milliseconds per call, and the shipped
code simply pays it per call.  The orderings are REUSABLE but not yet
reused: a reconstruction draws its partitions once at setup and
their memberships never change afterward -- only the subset visit
order reshuffles per pass.  One caution for any future cache: the
default granularity list carries four INDEPENDENT 128-subset
partitions, not one, and the default sequence cycles through all
four for the tail of a long run, so a cache must key on the
partition entry, and holding the fine level whole costs four times
any one-partition estimate.  The reuse is also back-loaded: a pass
visits each entry once, and a 3-iteration run's sequence entries
are all distinct, so nothing repeats until runs get long -- a
default-length run visits each 128-entry twenty-five times, and a
positivity-constrained update projects its subset twice back to
back.  Restricting the schedule to a single 128-partition would cut
the cache four-fold, free as an opt-in; as a default change it is
an algorithm decision owing convergence and parity evidence, since
the four-way cycling is inherited mbirjax behavior.  All of which
is why the memoization stays a recorded follow-up with a ledger
charge rather than a free win.

### 3.2 The parallel back: gather and store once

**What it computes.**  The adjoint: each pixel's cylinder collects,
for every view, the weighted sinogram values its channels and rows
touch, and the results accumulate over views into a (pixels,
columns) partial.

**How the work is subdivided.**  One work unit per (pixel tile,
column tile); views accumulate inside the driver's view-batch loop.
The output is written once per element -- a plain store, no
collision-safe adds -- because each output element belongs to exactly
one work unit.  The kernel's width argument is rounded up to the
next multiple of 16 before the launch (the padding rule of findings
§1.23), because the compiler produces a slower kernel for widths it
cannot prove divisible by 16; the padded columns are computed and
discarded.

**Where the traffic goes.**  The kernel gathers from a channel-major
sinogram copy; along a tile's column axis the gathered addresses are
nearly consecutive, so the reads coalesce well.

**What limits it.**  Honesty note: this is the one kernel that has
never been counter-profiled -- its rows are timing-only.  Its
measured behavior tracks its cone sibling's (the padding moved its
1024-class two-device call from break-even to 1.88x, findings
§1.23), and §8 records the missing profile as an open edge.

### 3.3 The cone forward: the vertical fan, then the same scatter

**What it computes.**  As §3.1, with one more sum: under cone beam a
cylinder's slice lands on a view-dependent detector row, so for each
(pixel, detector row, view) the kernel first builds the contribution
by walking the slice taps of the vertical fan, and then spreads it
across the channels exactly as the parallel forward does.  The
scatter block in the code is labeled "the plain per-tap atomic
form", and it is the same three-tap collision-safe add the parallel
kernel had before the sorting.

**How the work is subdivided.**  One work unit per (pixel tile,
detector-row tile, view).  The values cannot be hoisted across views
-- which slices feed which rows changes with the view -- so each
view's work re-reads them; that re-read is inherent to the geometry,
not a defect, and it is why cone never had the reload pathology
described in §5's history column for parallel.

**What limits it.**  Mixed, between its two siblings: 50 percent
arithmetic against 64 percent memory, nine warps waiting, occupancy
69 percent at 40 registers (mg31).  Its memory side is dominated by
the vertical fan's gather (the values block re-read about 60x per
52-view launch), not by the scatter; the scatter is the same shared
block at exactly 1.50x the coalesced ideal.

**The design sketch: the sorted scatter applied to cone.**  The
sorted contraction of §3.1 targets exactly the scatter cone shares,
and the transformation transfers unchanged: sort each view's pixels
by channel center, build the banded weight matrix per tile, contract
against the tile's vertical-fan output (in place of the raw values),
and add the window once.  Two things temper the payoff.  Cone's
adds are the smaller of its two memory streams, so removing most of
them is bounded near 1.3x even if the atomic share vanished
entirely; and the per-view sort adds gathers to a kernel already
gather-heavy.  Against that, the same counters show cone's scatter
volume per view identical to parallel's, and mbirjax's cone runs
2.4x its own parallel -- so a sorted cone forward that lands even
its bounded win would pull AHEAD of mbirjax at the cone cells,
where parity is today's standing.  The increment is unopened; it
would follow the same spike-then-composed-gate path §3.1 took.

### 3.4 The cone back: the band walk

**What it computes.**  The cone adjoint over a slice **band**: the
multi-device driver hands each call a range of slices, and for each
(pixel, band slice) the kernel gathers the weighted sinogram values
from both fans' taps across all views of the batch, storing each
output element once.

**How the work is subdivided.**  One work unit per (pixel tile,
band-slice tile), with the views walked in a loop inside the kernel.
The band argument is padded to a multiple of 16 exactly as in §3.2;
this padding is what closed the campaign's oldest anomaly, where the
band lengths that two and four devices happen to produce ran the
kernel at half rate (findings §1.21, §1.23; the 2.44x cliff).

**What limits it.**  Arithmetic, mildly: 72 percent of peak against
64 memory, half a warp waiting, and the gathers 90 percent absorbed
by the near cache (mg25).  Its one recorded bound: 121 registers per
thread cap occupancy at 25 percent, so a tile-and-warp retune is
worth at most about 1.4x and is recorded, not scheduled (the B3
closure).  Nothing here has atomics to sort -- the zero-witness
measured exactly zero -- which is why the sorting campaign went to
the forwards.

## 4. Where the time goes

Three tables, all measured on H100, all cited.  A dash marks an
unmeasured cell.

**Table 1 -- the parallel forward per launch, 1024-class, 128
views** (per-tap and chunk: mg32's recorded medians; sorted: mg33's
winner; the divisors are VCD's call sizes):

| pixel set | per-tap ms | 32-view chunk ms | sorted ms | sorted speedup |
|---|---|---|---|---|
| full mask (771,240) | 864.7 | 727.7 | 217.2 | 3.97x |
| 1/4 subset | 216.4 | 244.8 | 56.2 | 3.85x |
| 1/16 subset | 55.1 | 62.9 | 15.5 | 3.55x |
| 1/64 subset | 14.8 | 17.1 | 5.3 | 2.77x |

**Table 2 -- the four kernels' counter profiles**, one warm
production-shaped launch each (mg28, mg33, mg31, mg25; the parallel
back has no profile):

| reading | par fwd, per-tap | par fwd, sorted | cone fwd | cone back | par back |
|---|---|---|---|---|---|
| arithmetic, % of peak | 15.6 | 57.8 | 50.4 | 71.6 | - |
| memory, % of peak | 52.3 | 81.5 | 63.6 | 63.6 | - |
| warps waiting on memory | 38.2 | 2.6 | 9.0 | 0.5 | - |
| occupancy (limiter) | 89.5% (registers) | - | 68.9% (registers) | 24.9% (registers) | - |
| collision-safe add packets per launch | 5.6e10 | 1.8e9 | 2.3e10 | 0 | - |
| verdict | waits on its own adds | spends memory on useful traffic | mixed, gather-heavy | compute-side | unprofiled |

**Table 3 -- what the kernels are of a whole reconstruction** (warm
3-iteration walls; §1.5 and §1.20 for the shares, mg34 for the
sorted-era walls):

| cell, devices | wall, per-tap era | forward share | back share | wall, sorted route |
|---|---|---|---|---|
| parallel 1024, n=1 | 39.96 s | ~72% (28.9 s) | small | 21.19 s |
| parallel 2048, n=4 | 188.98 s | 120.5 s busy | 36 s busy | 113.25 s |
| cone 1024, n=1 | 61.56 s | ~52% (32.2 s) | - | unchanged |
| cone 2048, n=4 | ~420 s | 127.9 s busy | 104 s busy (padded) | unchanged |

For context, the recorded mbirjax walls at parallel 1024 are 25.80,
14.33, and 11.52 s at one, two, and four devices; the sorted route
reads 21.19, 14.25, and 10.75 (findings §1.31).

## 5. Where the memory goes

The arrays one forward or back call holds resident, with the
1024-class numbers.  These are the same terms the memory ledger
charges, read from the same helpers, so the code and the charge
cannot disagree.

| resident | formula | parallel fwd (sorted) | cone back |
|---|---|---|---|
| values block | pixels x slices x 4 | 2.9 GiB (call-fixed) | - |
| output | views x channels x padded cols x 4 per batch | 0.5 GiB | pixels x padded band x 4 |
| contract, per view in batch | 16 B x pixels | 12 MB/view | 48 B x pixels/view |
| sort artifacts, per view | 20 B x pixels | 15 MB/view | none |
| channel-major sinogram copy | batch x channels x rows x 4 | none (output is channel-major) | one per batch |

The sorted route's 20 bytes per (view, pixel) are the argsort's
order, the permutation, and the gathered contract copies; the charge
switches off with the route (`sorted_forward_enabled`), so flipping
the switch flips the charge.  The peak of a whole reconstruction is
the ledger's business, not this table's; the calibration bands are
(1.00, 1.30) for kernel-path geometries (findings §1.20).

## 6. The safety nets

**Value gates.**  Every kernel gates against the torch body it
replaces at 1e-5 relative single-shot and 1e-4 iterated, and
full-pipeline comparisons gate at 1e-3.  The classes exist because
collision-safe adds and reordered contractions sum the same numbers
in a different order; the gates catch a miscompile, not a ULP.  The
sorted route adds two gates of its own: the two forward kernels
against each other, and the sparse-set fallback against the body.

**The span guard.**  §3.1's window path runs only when a tile's
measured span fits; the fallback is the shipped per-tap block,
verbatim.  Correctness never rests on the sorting.

**Padding contracts.**  Widths and bands round up to multiples of 16;
padded lanes either carry zeros that add nothing (forward) or are
computed and sliced off (back), and the tests assert both behaviors
and the returned strides.

**Availability and the switches.**  Each kernel binds only after a
per-device value self-check on the actual hardware.
`MBIRTORCH_DISABLE_TRITON=1` disables all four;
`MBIRTORCH_SORTED_FORWARD=0` restores the per-tap parallel forward.
Both are read per call.

## 7. Decisions and their one-line reasons

| decision | reason |
|---|---|
| channel-major layouts | the axis a tile sweeps becomes contiguous, so packets fill |
| pad widths and bands to multiples of 16 | the compiler emits a half-rate kernel for widths it cannot prove divisible (findings §1.21) |
| sort per view, not once | a pixel's channel center depends on the view angle |
| contract densely, zeros included | idle arithmetic is spent to relieve the congested add machinery |
| full-precision multiply mode | the tensor-core default rounds to a 10-bit mantissa and fails the 1e-5 gates; its measured worth was 10 percent (findings §1.29) |
| window is a compile-time 16 | the dot's hard minimum, and sorted spans measured 2 to 3 |
| per-tap fallback kept in-kernel | sparse pixel lists are legal inputs, and speed may depend on locality but values may not |
| sort recomputed per call, uncached | milliseconds against hundreds saved; the `plan` slot is the recorded memoization follow-up |
| view chunk of 16 on the sorted route | grid granularity only; the per-view gather already bounds reuse |

## 8. Provenance, code pointers, and open edges

**Code.**  `mbirtorch/triton_parallel.py`: `_parallel_forward_kernel`
(per-tap), `_parallel_forward_sorted_kernel`,
`_parallel_forward_view_batch_triton` (the wrapper and the sort),
`sorted_forward_enabled`, `_parallel_back_kernel`,
`_parallel_back_view_batch_triton`.  `mbirtorch/triton_cone.py`:
`_cone_forward_kernel`, `_cone_back_kernel`, and their wrappers.
`mbirtorch/_utils.py`: `padded_kernel_width`, the one definition of
the rounding rule.

**Provenance.**  The width mechanism and padding: findings §1.19,
§1.21, §1.23 (mg20, mg21, mg21b, mg23, mg24).  The counter profiles:
§1.24 (mg25), §1.26 (mg28), §1.28 (mg31), §1.30 (mg33).  The loop
spikes and their ceiling: §1.27 (mg29, mg30).  The segmented spikes:
§1.29, §1.30 (mg32, mg33).  The composed gate: §1.31 (mg34).  The
design note: `active/pfwd_segmented_design.md`.

**Open edges.**  The parallel back kernel has never been
counter-profiled.  The cone forward's 60x gather re-read is recorded
but unattacked, and its sorted scatter (the §3.3 sketch) is an
unopened increment.  The cone back's register-pressure retune is
recorded and unscheduled (B3's closure).  Once the sorted route
commits, the floors and the comparison tables owe the re-measures
§1.31 names.

## Appendix: how to re-measure each table

All harnesses live in `plans/experiments/torch_port/`, each with its
sbatch and run record beside it.

* Table 1: `mg32_pfwd_segmented.py` (per-tap and chunk columns) and
  `mg33_pfwd_sorted.py` (sorted column), one GPU, about two minutes
  each.
* Table 2: `mg28_pfwd_counters.py`, `mg33_pfwd_sorted.py` (its
  counter attempt), `mg31_cfwd_counters.py`, `mg25_back_counters.py`;
  one GPU each, minutes each.
* Table 3: `mg34_sorted_ab.py` for the composed walls (four GPUs,
  about 20 minutes, reusing mg27's staged sinograms);
  `mg19_two_k_baselines.py` for the 2048-class shares.
* The band-divisibility cliff: `mg21b_band_divisibility.py`
  (unpadded-era record) against `mg23_parallel_band.py` and the mg23
  job's cone sweep on the padded tree.
