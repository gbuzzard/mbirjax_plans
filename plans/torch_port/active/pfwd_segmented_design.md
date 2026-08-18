# Design note: segmented accumulation for the forward kernels

**Status: SPIKES COMPLETE, 2026-08-18 (mg32 and mg33, findings
§1.29 and §1.30); the library step (§6.2) awaits Greg's go.**  Opened 2026-08-18,
evening; the cone counter reading (mg31) landed the same evening and
§4 carries it.  Greg approved the spike the same evening and asked
for the plain-language explanation now following §2; the spike is
mg32.  The library step stays conditional on the spike and the
composed re-gate.

**What this note is.**  The B7 investigation measured the parallel
forward kernel's floor and found it is the atomic write path's own
volume: every design that keeps one atomic add per (pixel, channel
tap, column, view) carries the same 1.79 TB of atomic sectors per
launch, and reordering the loops around that volume ceilinged at
1.17x (findings §1.26, §1.27).  The way past the floor is fewer
atomic adds.  This note designs that reduction: in-tile accumulation
for the horizontal-fan scatter the two forward kernels share.  It is
the in-kernel form of the sorted-channel idea whose torch-level form
was rejected on 2026-08-11 (findings §1.14); the counters now give
the in-kernel form a measured target the torch-level form never had.

**Sources.**  The measured base is findings §1.26 and §1.27 with rows
mg28 through mg30, and mg31 pending for cone.  The code read for this
note is the parallel forward kernel and wrapper
(triton_parallel.py lines 370 to 560) and the cone forward kernel's
scatter (triton_cone.py, the block its own comment labels "the plain
per-tap atomic form").  The precedent is mbirjax's July kernel
campaign, which replaced its scatter-fusion forward with segmented
kernels and gated 2.13x on the production fine-tail case
(projector_kernels/gpu_headroom_findings.md).

---

## 1. The finding that shapes this note

Three loop structures were measured on the parallel forward at the
1024-class cell, full mask, 128-view launches:

| design | DRAM read GB | DRAM write GB | atomic sectors | launch ms |
|---|---|---|---|---|
| shipped, one view per program | 377 | 0.5 | 5.597e10 | 864 |
| whole-batch view loop (mg29) | 38 | 262 | 5.597e10 | 757 |
| 32-view chunks (mg30 winner) | 19 | 49 | 5.597e10 | 728 |

Cutting the DRAM traffic 5.5x moved the launch 1.19x, and the column
that never moved names the floor: the atomic-path sector count is the
algorithm's, not the schedule's, because every design issues
taps x pixels x width x views adds.  The stall reading rose as the
other traffic fell.  A remedy must reduce the count itself.

## 2. The proposal: accumulate per tile, then add once per channel

Within one (BLOCK_P pixels, BLOCK_R columns) tile at one view,
neighboring pixels project to neighboring channels -- the property
both kernels' grid comments already rely on for cache locality.  A
tile's taps therefore land in a channel WINDOW of about
BLOCK_P + 2 x PSF_RADIUS channels.  The proposal: accumulate the
tile's contributions into that window privately, then issue ONE
global atomic add per (window channel, column), in place of one per
(pixel, tap, column).  The adds fall by about
3 x BLOCK_P over (BLOCK_P + 2): 2.4x at the shipped 8-pixel tile,
2.8x at 32.

Five design points, decided here so the spike is mechanical:

* **The accumulation is a dense contraction.**  For one view, the
  tile's horizontal-fan weights form a (BLOCK_P, WINDOW) matrix W,
  banded three wide, computable directly from each pixel's channel
  center against each window channel -- no tap loop.  The window's
  accumulation is then out_window = transpose(W) @ vals, one
  `tl.dot` per view, which turns the scatter into tensor-core work.
  Two constraints discovered while preparing the spike are recorded
  here.  First, the shared-memory atomic form this point originally
  named as the fallback design is not expressible in Triton, which
  manages shared memory implicitly and exposes no user-level atomics
  on it; the contraction is the implementable form, and a
  scratch-atomic kernel would need a CUDA- or pallas-level rewrite
  outside this campaign's tooling.  Second, `tl.dot` on float32
  inputs uses the reduced-precision tensor-core mode by default,
  which would fail the 1e-5 values gate; the spike runs the
  contraction with the full-precision input mode, and any
  reduced-precision arm is a separate question with its own gate
  discussion, not taken here.
* **The window is a compile-time constant with a per-tile guard.**
  WINDOW is a constexpr sized BLOCK_P + 2 x PSF_RADIUS rounded up to
  a tensor-core-friendly multiple.  Each tile computes its channel
  span at run time; a tile whose span exceeds WINDOW takes the
  shipped per-tap atomic path for that tile.  The guard keeps
  correctness for any pixel ordering or mask shape; the raster mask
  makes wide-span tiles rare, and the spike reports the measured
  fallback rate.
* **The view chunk composes with the segmentation.**  mg30's chunk
  (reads amortized, output planes cache-resident) rides unchanged as
  the outer loop for parallel; the chunk knob re-sweeps in the spike
  because the segmented kernel's traffic balance differs.  Cone has
  no view-invariant values, so its kernel keeps its per-view grid;
  the segmentation applies to its scatter as is.
* **The subset path sorts per view, and the sort is what makes the
  contraction universal (added 2026-08-18, after mg32; Greg's
  question).**  mg32 measured subset calls losing (0.88x) because a
  tile of subset pixels has no channel locality for any window.
  Sorting restores it, with one crucial form: the sort is per VIEW,
  because a pixel's channel center depends on the view angle and one
  ordering cannot serve a 128-view batch.  Per view, the call's
  contract arrays are gathered into channel order outside the
  kernel, and the kernel's values-tile load gathers its rows through
  the per-view permutation.  The density arithmetic then favors
  subsets: 12,051 pixels over 992 channels put sorted 16-pixel tiles
  in a 2-to-3-channel span, so the adds fall up to 16x there, and
  the contraction becomes a true segmented reduction (many W rows
  landing on one channel, summed by the dot).  This is the
  sorted-stream idea placed where the tooling can express it; the
  torch-level form mg13 rejected materialized its streams, and this
  form applies a permutation at load time.  Two costs land softly at
  exactly the scale that needs the sort.  A subset's values block is
  L2-resident (about 48 MB), so per-view gathered rows are on-chip
  reads and losing the chunk's hoist there costs little.  And VCD's
  subsets are fixed per reconstruction, so the per-(subset, view)
  orderings compute once and are reused every iteration; the cached
  permutations are about 6 MB per subset per view batch and take a
  ledger charge, or recompute per call at the few-millisecond class.
  The full mask keeps the unsorted chunk path: raster order is
  already channel-local, and a per-view gather of a 3.1 GB block
  would resurrect the reload mg29 measured.
* **Shared memory stays small.**  The window accumulator is
  WINDOW x BLOCK_R floats: 6 KB at an 8-pixel tile with a 12-channel
  window and 128 columns.  Occupancy arithmetic does not move.
* **The values contract is unchanged above the kernel.**  The
  wrapper, the drivers, the batching rule, the ledger terms, and the
  padding rule all keep their current form; the change is the kernel
  interior and its launch constants.

### Plain-language explanation (added at Greg's request)

The proposal above packs three decisions into technical shorthand;
this section says each one plainly.

**The tile's scatter is a small matrix multiply.**  Take one work
unit's tile: 8 or 16 pixels by 64 or 128 columns, at one view.
Today each pixel computes its three weighted contributions and fires
each one individually into the sinogram, and every one of those
additions goes through the GPU's shared collision-safe add
machinery, which the counters showed is the congested resource.  But
all of the tile's pixels land in a window of a dozen adjacent
channels, and the weights form a little matrix W: one row per pixel,
one column per window channel, each row holding just the three
trapezoid weights near the diagonal -- a narrow banded matrix.  The
tile's entire contribution to the window is one product,
transpose(W) times the tile's block of voxel values.  One small
matrix multiply replaces all the scattered additions, and the tile
then writes its finished window into the sinogram with one
collision-safe add per (window channel, column) -- the 2.4x to 2.8x
reduction in traffic through the congested machinery.

**Why compute it densely when the matrix is mostly zeros?**  Because
of what the hardware offers.  GPUs carry dedicated units that
perform small dense matrix multiplies at enormous rates, and this
kernel's profile showed its arithmetic units 84 percent idle while
the addition machinery was the bottleneck.  Multiplying by the zeros
in W wastes arithmetic on paper, but it is arithmetic we have in
surplus, spent to remove operations from the resource we are starved
for.  This trade is also the shape of what mbirjax's kernels do.

**The span guard: correctness never rests on the geometry.**  The
matrix form assumes a tile's pixels all project into one small
channel window.  For the raster-ordered mask that is almost always
true, because consecutive pixels are geometric neighbors.  But the
kernel must be correct for any pixel list: subset schedules,
arbitrary masks, and the wrap at the end of a mask row where two
consecutive pixels sit on opposite sides of the volume.  So each
tile first measures its actual span, the lowest to highest channel
its pixels touch.  A span that fits the window takes the fast path;
a span that does not sends that one tile through the shipped
per-tap code, unchanged.  Only speed depends on the geometric
assumption, never values, and the spike reports how often the
fallback fires so the fast path's coverage is a measurement.

## 3. The arithmetic

The atomic sectors fall by the window ratio, 2.4x to 2.8x by tile
size.  What that buys depends on how much of the floor the atomic
path carries, which the counters bound but do not settle: the memory
side runs near 60 percent of speed of light with the atomic path the
dominant unchanged component, so the honest projection is a launch
speedup in the 1.5x to 2.5x range over the shipped kernel, on top of
nothing else changing.  The comparison arithmetic: the parallel
forward is 28.9 s of the 40 s one-device 1024-class wall, so 1.5x on
the forward puts the wall near 30 s and 2.5x puts it near 23 s,
against mbirjax's 25.8.  These are projections from measured shares,
not measurements; the spike turns them into numbers.

## 4. Scope

* **Parallel forward: in scope on a measured basis** (findings §1.26,
  §1.27).  The primary target: memory-side at 52 percent against SM's
  16, stall 38, and the atomic path the dominant unchanged component.
* **Cone forward: in scope opportunistically, on mg31's mixed
  reading** (job 15345826, rows
  `rows/mg31_pfwd_counters_h007_20260818_165713.jsonl`).  The cone
  forward is neither parallel's extreme nor the cone back's: SM 50.4
  percent against memory 63.6, stall 9.0, occupancy 68.9 percent at
  40 registers.  Its scatter is the same shared block at exactly
  1.50x the coalesced ideal and the same per-view atomic volume as
  parallel's, so the segmentation applies unchanged -- but its
  memory side is gather-dominated (the vertical fan re-reads the
  values block 60x per 52-view launch from DRAM, L1 load hit 42
  percent), which the segmentation does not touch.  The projected
  cone payoff is therefore materially smaller than parallel's,
  bounded near 1.3x if the atomic share vanished entirely.  The
  recommendation: cone rides the spike (the kernel change is shared
  and one sweep column is nearly free), and its library adoption
  gates on its own composed numbers.  The gather-side re-read is
  recorded as a separate observation, not this note's scope.
* **The back kernels: out of scope.**  They gather and store once per
  output element, with zero atomic sectors measured (§1.24).

## 5. The value gates

The segmented sum reaches each (channel, column) through a different
addition order: a dense contraction over the tile's pixels, then one
atomic combine across tiles.  The sums are the same commutative
arithmetic, so the standing calibration applies: 1e-5 relative
single-shot against the shipped wrapper, 1e-4 iterated, and the
goldens unchanged (lessons §2).  Two arms are specific to this
design: a forced-fallback arm (WINDOW set below the span so every
tile takes the per-tap path, which must reproduce the shipped values
exactly at the gate) and a mixed arm (a mask engineered so some tiles
fall back, gating the seam between the two paths).

## 6. Implementation increments

1. **The spike (mg32).**  The segmented parallel kernel written
   inside the harness: sweep BLOCK_P with its window (16 pixels with
   a 32-channel window, 32 with 64 -- `tl.dot` and `tl.arange` need
   powers of two), the column tile, and the view chunk; the
   contraction in the full-precision input mode; values gates at
   every point; a forced-fallback configuration (window 16, below
   the 18-channel span) that must reproduce the shipped values
   through the per-tap path; the measured fallback rate on every
   configuration; one counter reading on the winner, whose
   confirmation is the atomic sector count falling by the window
   ratio.  Parallel first: cone follows in its own spike on
   parallel's verdict, per §4's opportunistic scope.
1b. **The subset spike (mg33).**  The always-sorted variant of the
   mg32 kernel: per-view argsort and contract gather outside the
   kernel, the values rows gathered through the permutation inside
   it, the same window machinery and values gates.  Its domain is
   the subset ladder points where mg32 lost; its full-mask point is
   expected to lose (the per-view gather at 3.1 GB) and is kept as
   the demonstration that the two paths partition by call locality.
   The reading that matters is the combined projection: best of the
   chunk path, the sorted path, and the shipped kernel per ladder
   point, which is the mixture the library selection would realize.
2. **The library step, only if the spike gates -- now shaped by the
   spike's verdict (findings §1.29).**  The spike measured 1.68x at
   the full mask with the mechanism confirmed, and 0.88x at subset
   calls, whose randomly drawn pixels have no channel locality for
   any window.  The library form is therefore a PER-CALL SELECTION,
   not a replacement: the segmented kernel where the call's pixel
   set is channel-local (the full mask, and any caller that
   preserves raster locality), the existing tap kernels elsewhere.
   The selection predicate, the subset path's own choice (the
   sorted contraction if mg33 gates it, else the shipped kernel or
   mg30's chunk form), and the fallback-rate counter fix are the
   step's design work; the composed A/B at the
   512-, 1024-, and 2048-class cells is its gate, and a spike win
   that does not survive composition does not ship (lessons §5).
   AMENDED after mg33 (findings §1.30): the sorted contraction won at
   EVERY ladder point including the full mask, where the sorted
   order's geometric coherence kept the values gather in the chunk
   path's traffic class while the atomic volume fell 31.6x.  The
   step's selection question therefore simplifies to one A/B:
   the always-sorted kernel against the locality partition, with the
   2048-class gather behavior and the multi-device paths as the open
   cells the composed gate must cover.
3. **The floors consequence.**  The kernel files are cost inputs, so
   the staleness note fires and the cone and parallel rows owe a
   refresh -- the family-scoped mode of the coarsening proposal, if
   approved, makes that a 3 to 5 GPU-hour run.
4. **The records.**  The comparison table and the user-docs timing
   table re-anchor if the library step ships (the C1 and F1
   instruments, re-run).

## 7. What this note does not change

The drivers, the batching rules, the memory ledger's charges, the
padding rule, the multi-device mechanics, and every back-projection
path are untouched.  mg30's view-chunk variant ships nowhere on its
own under this note: its loop rides inside the segmented kernel, and
its standalone 1.17x is recorded in §1.27 as the loop family's
ceiling if this design is declined.

## 8. The ruling

* **(a) Approve the spike (increment 1)** on the parallel forward,
  with cone included if mg31 reads atomic-bound.  This is the
  recommendation: the design attacks the one component the counters
  say is binding, the arithmetic matches the remaining gap to
  mbirjax, and the spike is a two-minute-class harness run away from
  a measured answer.
* **(b) Approve the full increment (1 and 2)** in one ruling, with
  the library step still conditional on the spike and the composed
  re-gate.
* **(c) Decline, and take the recorded alternative:** ship mg30's
  view-chunk variant at 1.17x through the composed re-gate, or hold
  the forward as it is.
