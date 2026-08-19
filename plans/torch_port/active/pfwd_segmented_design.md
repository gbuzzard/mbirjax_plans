# Design note: segmented accumulation for the forward kernels

**Status: the parallel library step SHIPPED as mbirtorch c761b24
(2026-08-18; composed gate, findings §1.31), and the cone rework
line (§9) CLOSED by Greg's ruling 2026-08-19 (findings §1.32,
§1.33).**  Opened 2026-08-18,
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
  subset membership is fixed per reconstruction (the partitions draw
  once at setup; only the visit order reshuffles per pass), so the
  per-(subset, view) orderings are reusable once per pass at their
  granularity; the shipped form recomputes them per call at the
  measured few-millisecond class, and the plan-slot memoization is
  the recorded follow-up, at about 6 MB per subset per view batch
  under a ledger charge.  (Sizing caution, 2026-08-18: the default
  granularity list carries four independent 128-subset partitions
  and the default sequence cycles all four, so the cache keys on
  the partition entry and the fine level costs four times any
  one-partition estimate.  Greg's proposed lever: one shared
  128-partition per granularity -- free as an opt-in, an algorithm
  change owing convergence and parity evidence as a default.)
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
  gates on its own composed numbers.  The gather-side re-read was
  recorded here as out of scope; §9 (added 2026-08-19) now carries
  the design that attacks it.
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
## 9. The cone pixel-batched form: two-axis grouping (2026-08-19)

Added after the sorted parallel forward shipped, from Greg's proposal
in discussion; this section carries the design that §4 previously
declared out of scope.

**What it attacks.**  mg31's gather reading: 175.4 GB of DRAM read
per full-mask 52-view launch, 60.6x the flat recon.  Divided by the
views that is 1.17x per view -- each view reads the volume
essentially once, nearly perfectly -- so nothing inside a view is
worth reordering.  The whole prize is CROSS-VIEW amortization: hold
a piece of the flat recon and serve several views from it.

**The inverted decomposition.**  A work unit holds a (pixel tile x
slice tile) block of the flat recon in registers and serves a chunk
of 8 to 16 consecutive views; reads fall by roughly the chunk
length.  The hazard is the traffic-conservation lesson (mg29):
anchor the loop on the input and the OUTPUT becomes the moving
target.

**The two-axis grouping (the proposal).**  Ungrouped, a tile's
slices scatter over hundreds of detector rows, because magnification
varies across the mask and row = m0 + W_p_r x slice (cone_beam.py's
sanctioned affine bridge).  Group pixels so each tile is compact in
BOTH detector axes -- channel center for the horizontal, W_p_r for
the vertical -- and the tile's footprint collapses to a (channel
window x row window) patch: accumulate it in registers, flush it
once per view.  The per-view flush decouples the two concerns: chunk
drift moves the window's POSITION between views (recomputed, free),
while the window's SIZE is set by the within-view spread of the
group plus a pad for membership going stale across the chunk's
angles.  The ordering is ONE compound sort per chunk, not two full
sorts: coarse W_p_r buckets as the primary key, channel center as
the secondary; consecutive tiles inherit bucket-width compactness in
magnification and sort-order compactness in channel.  Against the
parallel route's per-view sorts this is a factor-of-chunk fewer
sorts, each slightly heavier for its fused key.  If the paper check
finds lexicographic tiles too loose in magnification at viable
bucket widths, the fallback organization is true two-axis bucketing
-- still one pass, different constants.

**Required fallbacks (Greg's rider, 2026-08-19).**  Three inputs
must disengage the route rather than degrade it.  Sparse pixel
SUBSETS thin the groups; the per-tile per-tap fallback carries over
from the parallel design unchanged.  SMALL reconstructions fall
below the tile and window minimums; the whole call takes the per-tap
route behind a switch, the pattern the parallel forward ships.
Sparse-VIEW problems break the chunk's premise of nearby angles, so
the availability rule must key on ANGULAR SPACING, not view count:
the chunk shrinks as spacing grows and the route disengages when a
chunk of one is all that remains.

**Helical scope (Greg's note, 2026-08-19).**  Nonzero helical
z-shifts fit this design rather than fighting it.  The shift enters
the affine as a per-view term (the z_offset of
`_cone_vertical_affine`), so within one view it moves every pixel's
rows together -- common-mode, which the per-view flush absorbs as
window position rather than size -- and consecutive views of a
chunk carry similar shifts because the helix is continuous in
angle, so the membership staleness padding stays small.  The
spike's kernel therefore keeps the shipped z-offset arithmetic even
though its test cell is non-helical, and a helical arm of the mg37
check (one extra cell construction; the harness already reads the z
column of the view parameters) is the cheap confirmation when this
scope is exercised.

**Ceiling and ladder.**  The kernel runs at 50.4 percent arithmetic
(mg31), so a perfect memory fix bounds near 2x on the kernel --
about 16 s of the 59 s cone 1024 wall.  The ladder: mg37, the paper
check, on the real builders on CPU (bucket count against the row
window budget against tile fill; chunk staleness padding; the
accumulator's register bill); a spike in mg33's template only if
mg37 closes; the composed gate only on a spike win.

**The paper check closed (mg37, run 2026-08-19, same day).**
Computed from the shipped builders on CPU at the 1024-class cone
cell, four reference angles, the compound sort exactly as above.
With 32 W_p_r buckets, an 8- or 16-view chunk, and slice tiles of 4
or 8, at least 99.8 percent of tiles fit a 16-row x 8-channel
window (per-view flush sizing: row p99 15.7, channel p99 7.3), and
every swept combination fits 32 x 16 at 99.7 percent or better; the
leftovers are the per-tile fallback's job, and bucket-boundary
tiles number at most 31 of 24,101.  The accumulator patch at
16 x 8 is 512 bytes, far under mg32's 6 KB precedent.  The check
also measured WHY the flush must be per view: the union footprint
across a chunk runs 27 channels wide at 8 views and 54 at 16 (the
common-mode rotation drift), so a per-chunk flush is infeasible
while the per-view window stays single-digit channels.  Chunk 16
prices nearly the same windows as chunk 8, so the read amortization
can take the longer chunk essentially free.  Rows:
`rows/mg37_cone_window_Gregs-MacBook-Pro-2_20260819_075351.jsonl`.
The spike is worth building; it awaits Greg's go.

**The spike ran and lost (mg38, 2026-08-19; findings §1.32).**
0.24x at the best arm on the full mask, every values gate held.
The grouping and the window mechanics validated exactly as the
paper check predicted (99.8 percent of tiles on the window path,
atomics down 5x), but the read amortization -- this design's one
prize -- never materialized: the counter read DRAM at the shipped
kernel's once-per-view signature (73.7x the recon block per
launch), so the values tile did not stay resident across the view
chunk, and the window path's arithmetic tax was paid on top.  The
compile introspection could not read registers or spills (None
throughout), so spill-versus-rematerialization is undistinguished.
Status: a measured negative.  The one bounded follow-up left open
is the spill diagnostic; reopening on it is Greg's call, with the
honest arithmetic that even amortized reads faced a thin margin
under the 2x ceiling and the tax.

**mg38 follow-up:**

**What mg38 actually isolated.** The loss decomposes into two 
independent problems, and the promising parts survive both: 
(1) the *residency failure* — the values tile never stayed 
in registers across the view chunk, so reads stayed at 
once-per-view; and (2) the *arithmetic tax* — the window 
path evaluates all 16 window rows per (pixel, slice) where 
only 2–3 carry weight, because Triton can't scatter into a 
register tile by computed index, so it must evaluate 
densely. Fixing only the residency leaves the tax: my 
arithmetic says reads dropping to the predicted 8x would 
take 1968 ms to roughly 1200 — still 0.4x. Any winning 
rework must beat *both*, or sidestep the register-residency 
idea entirely. Three candidates, ranked:

**A. The sorted order fed to the *unchanged* shipped 
kernel — my top pick.** Everything mg38 built for grouping 
runs *outside* the kernel: the compound sort, the gathered 
contract, the permuted values. Feed exactly that sorted 
order to the shipped per-tap kernel — no new kernel, zero 
arithmetic tax, values identical to within summation order. 
The shipped kernel's measured pathology at mg31 wasn't DRAM 
bandwidth, it was *locality*: L1 hit 41.7%, nine warps 
parked. Sorted tiles make the 32 pixels of each program 
gather overlapping slice ranges and adjacent channels, 
which is precisely what raises L1/L2 hits. If locality 
alone recovers 1.2–1.5x, it ships through a composed gate 
as a wrapper-level data-order change — the same shape of 
change as the parallel sort, minus the new kernel risk. 
Cost: one harness arm reusing mg38's machinery, 
~30 GPU-minutes.

**B. Cache-blocking by slice band through the existing 
kernel.** The wrapper already accepts `slice_start` with 
banded values. Loop (slice band × view chunk): a ~16-slice 
band of the flat recon is ~49 MB — it *fits in L2* — so 
serving a whole view chunk from one band gets the cross-view 
amortization mg38 wanted, but through the cache instead of 
through registers, with the shipped kernel's arithmetic 
untouched. Two mechanical needs: narrow the launched 
row-tile range per band (the mg37 affine machinery computes 
each band's row span directly — typically a few dozen rows, 
so the grid shrinks instead of wasting work), and accumulate 
across bands (either a no-zero variant of the output or 
summing band outputs). This is the design that actually 
attacks the 60x with the hardware doing what it's good at.

**C. Repair mg38's kernel itself** — fix the compile 
introspection to read spills, split fitting/non-fitting 
tiles into two launches so the divergent branch leaves 
the view loop, shrink the live set (accumulate the 8×16 
patch instead of the 32×16 pixel block), and cut the tax 
with slice tiles of 1–2 and 8-row windows (tax falls to 
~2.7x). Most effort, and the ceiling math is thinnest — 
it needs *everything* to land to reach maybe 1.3–1.6x.

A and B compose: sorted order helps B's within-band 
locality too. My recommendation is a single mg39 that 
runs A as the headline arm and B as the second arm 
(both reuse mg38's harness bones), with C held unless 
the other two disappoint. Rough cost: a half-day of 
authoring plus under an hour of GPU. Greg's decision:
start with A, then evaluate whether to proceed with B; 
C is not considered a good candidate.

**A ran and lost (mg39, 2026-08-19; findings §1.33).**  Every arm
below the raster baseline, 0.87x at best (the whole-batch compound
sort, its sort cost only 3.3 ms), all values gates inside 1e-5.
The raster order is already spatially local for the shipped
kernel's output-anchored grid; sorting bought nothing the caches
were not delivering.  **The evaluation of B, from mg31's own
counters rather than a run:** B exists to cut DRAM re-reads, and
DRAM is measured at about 13 percent of bandwidth with L2 already
absorbing 87 percent -- the hardware is already doing what B would
arrange.  Recommendation: decline B and close this rework line on
the three measured verdicts (mg38, mg39, B's arithmetic).  The
cone forward's cost is the vertical gather's sector and latency
machinery at the algorithm's own tap count -- a floor of the same
kind the parallel atomic volume was, without a sorting-shaped
exit -- and cone already matches or beats mbirjax at every
measured cell.  Greg so ruled 2026-08-19: A and B are closed, and
the §9 rework line with them.