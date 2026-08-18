# mg31 run record

One run.  The finding is in
`plans/torch_port/active/multigpu_findings.md` §1.28, and the reading
fills §4 of `pfwd_segmented_design.md`; this file holds the run
detail.

## mg31, the counter run on the cone forward kernel (job 15345826)

* Node h007, one H100, 1 minute 21 seconds, on the padded 64dedb8
  tree.  Exit 0.  Ordered by Greg 2026-08-18 as the segmented design
  note's cone input.
* The cell is the 1024-class cone of the comparison table (mg1's
  construction), recon (992, 992, 1008), full mask 771,240 pixels.
  The view batch came from the driver's own rule rather than a pin:
  52 views at the full mask, 128 at the 12,051-pixel subset.  The
  full-pixel launch timed 388.25 ms, descriptive only -- no recorded
  single-launch cone anchor exists.
* Both variants collected the full metric set on the first attempt.
  The readings: SM throughput 50.4 percent of peak against memory's
  63.6; long-scoreboard stall 9.0 warps per issue-active cycle;
  occupancy 68.9 percent at 40 registers per thread, limiter
  registers at 6 blocks per multiprocessor; L2 hit 87.2 percent; L1
  load hit 41.7 percent; DRAM read 175.4 GB per full-mask launch
  (60.6x the values block) against DRAM write at 1.07x the output
  slab.
* The atomic path read exactly 1.50x the coalesced ideal -- the third
  kernel confirming that constant for the shared horizontal scatter
  -- at the same per-view atomic volume as the parallel forward.
* The verdict the design note carries: mixed.  The cone forward is
  memory-side by a modest margin with real compute beside it, its
  scatter is identical in kind to parallel's, and its memory traffic
  is dominated by the vertical fan's gather re-reads rather than by
  the atomics.  Segmentation applies; the projected payoff is
  materially smaller than parallel's.
* One cosmetic note: the jsonl kept the mg28-derived filename prefix
  (`mg31_pfwd_counters_...`); the content is the cone run.
* Output rows:
  `rows/mg31_pfwd_counters_h007_20260818_165713.jsonl` (md5
  24aad1f05caa68a9f78bd85d5aa11f21, verified after copy).
