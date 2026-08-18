# mg25 run record

One run.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.24; this file holds
the run detail.

## mg25, the counter run on the cone back kernel (job 15342576)

* Node h000, one H100 80GB, 2 minutes 44 seconds of wall by sacct, on
  the padded 64dedb8 tree (verified per-file by md5 before submission).
* Exit 0.  Every witness held: all five band variants timed, every
  cross-band witness read exactly zero, every rate sat in the
  divisible class, no projection direction ran as general torch code,
  and the realized device was cuda:0.
* The cell is mg21b's: cone sinogram (256, 2016, 1984), recon
  (1984, 1984, 2016), full pixel mask 3,088,364, psf_radius 1 (3 taps
  per axis).  The sinogram is a seeded uniform draw; a back
  projection's time does not depend on its values.
* Timing leg, through the production route
  (`sparse_back_project_view_range`, driver view batch 13), one warm
  and three timed calls per band:

  | band | median s | ns per view-slice |
  |---|---|---|
  | 1008 | 4.746 | 18,390 |
  | 672 | 3.298 | 19,174 |
  | 512 | 2.444 | 18,645 |
  | 336 | 1.851 | 21,517 |
  | 256 | 1.277 | 19,487 |

  The padded tree's anchor is 20,400 (mg23), and the gate was 30,000.
  Peak device memory for the leg: 32.2 GiB.
* Counter leg: Nsight Compute 2024.3.1 at the cuda module's bin, with
  counter permission granted.  The kernel filter was discovered at
  runtime as `_cone_back_kernel`.  All six profiled variants (the five
  bands at pixel divisor 4, which is 772,091 pixels and view batch 40,
  plus the band-672 control at the full mask and view batch 13)
  collected the FULL 21-metric set on the first attempt, launch skip 4
  of 5, so no fallback set was needed.
* The counter table, constant across the five divisor-4 bands to the
  printed digit except where noted: achieved occupancy 24.9 percent;
  occupancy limiter registers, at 4 blocks per multiprocessor; 121
  registers per thread; SM throughput 71.5 percent of speed of light;
  memory throughput 60.7 to 65.5 percent; L2 hit 80.6 to 81.7 percent
  (91.6 at the full-mask control); L1 load hit 90.8 percent (96.3 at
  the control); 11.7 to 13.7 sectors per load request;
  long-scoreboard stall 0.44 to 0.49 per issue-active cycle.
* The gather pricing: L1 load sectors at 3.87x to 4.21x the
  one-sector-per-eight-taps ideal; DRAM write exactly 1.00x the
  output partial at every variant; DRAM read 20x to 91x the sinogram
  batch bytes, but at 54.4 GB over a 211.6 ms launch the DRAM rate is
  about 257 GB/s, a few percent of an H100's bandwidth.
* The zero-witness read exactly zero atomic and reduction sectors at
  every variant, which measures the design note's no-atomics claim.
* GPU health sampled before and after the timing leg: no throttle
  flags, no hot readings.
* Output rows:
  `rows/mg25_back_counters_h000_20260818_121603.jsonl` (md5
  7a6f5aeff10dce5d32366af4992ced63, verified after copy), with one ncu
  log per attempt beside it on scratch
  (`results/mg25_ncu_*_full_skip4.log`).
