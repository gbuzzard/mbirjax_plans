# mg28 run record

One run.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.26, with the
correction it makes to §1.19 noted there; this file holds the run
detail.

## mg28, the counter run on the parallel forward kernel (job 15344936)

* Node h001, one H100 80GB, 78 seconds of wall, on the padded 64dedb8
  tree, submitted after mg26 released the shared environment.
* Exit 0.  The timing anchor held: the full-pixel launch read 864.46
  ms against mg20's recorded 859.13 (1.01x), spread under 0.1
  percent.  The model bound the Triton forward, and both profiled
  variants collected the full 21-metric set on the first attempt at
  the aimed warm launch.
* The cell is mg20's: parallel sinogram (1024, 1008, 992), width
  1008, view batch 128, psf_radius 1.  Two variants through the
  shipped wrapper: the full 771,240-pixel mask, and the mask
  subsampled 64x to 12,051 pixels, the most common call size in the
  production schedule's mixture.
* The counter table, both variants agreeing to the printed digit:
  achieved occupancy 89.5 percent; limiter registers at 8 blocks per
  multiprocessor; 26 registers per thread; SM throughput 15.6 percent
  of speed of light; memory throughput 52.3 percent; L2 hit 80.2
  percent; L1 load hit 15.9 percent; 3.95 sectors per load request;
  long-scoreboard stall 38.2 warps per issue-active cycle.
* The two memory paths priced, view factor included: atomic-path
  sectors at exactly 1.50x the one-sector-per-eight-adds ideal; load
  sectors at 1.25x their ideal; DRAM write 1.00x to 1.05x the output
  slab; DRAM read 130x (full mask) and 138x (subset) the values
  block's bytes -- the block re-fetched from memory about once per
  view of the 128-view batch.
* The pricing formulas are stated in the script and carry the view
  axis mg20's write-path table omitted; 192 divided by the 128-view
  batch is the 1.50 measured, which is the §1.19 correction.
* GPU health: no throttle flags, no hot readings.  Timing-leg peak
  device memory 4.4 GiB.
* Output rows:
  `rows/mg28_pfwd_counters_h001_20260818_154504.jsonl` (md5
  8e5e09badbf7223573e9eb03928f5529, verified after copy), with one
  ncu log per attempt beside it on scratch
  (`results/mg28_ncu_*_full_skip4.log`).
