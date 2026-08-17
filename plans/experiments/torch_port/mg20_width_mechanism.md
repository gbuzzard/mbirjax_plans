# mg20 run record

Two runs.  The finding and its tables are in
`plans/torch_port/active/multigpu_findings.md` §1.19; this file holds
the run detail.

## Run of 2026-08-17, timing leg (job 15316533)

* Node h007, one H100, about 40 minutes of wall.
* All nine timing arms ran; the two strided arms ran through the
  harness's copied launch, whose full-width output first matched the
  library wrapper's.
* Values: 4.6e-7 to 6.8e-7 against the full-width reference on every
  arm.  Anchors: the four original widths reproduced the recorded
  per-slice ratios within 0.01.
* The profiler leg did not run: `ncu` was not on PATH in the batch
  environment, although the cuda module carries it.
* Output rows:
  `plans/experiments/torch_port/rows/mg20_width_h007_20260817_092645.jsonl`.

## Run of 2026-08-17, counter leg (job 15316589)

* Same script, after the sbatch gained a PATH fallback to the cuda
  module's own bin directory.  `ncu` was found and the counters were
  collected for widths 1008, 504, 252, 63, and the strided 512-over-
  1008 arm.
* Counter highlights, one warm launch each: occupancy 89.5 / 60.3 /
  60.4 / 95.2 / 89.5 percent, with the register limiter at 8 blocks
  per multiprocessor on the divisible widths and 5 on the
  non-divisible ones; L2 hit 80.2 / 90.7 / 90.7 / 89.1 / 80.1
  percent; atomic sector traffic 192 times ideal on the wide arms,
  rising to 235 at width 63; DRAM write traffic 1.05 to 1.16 times
  the output slab.

Notes a later reader may want:

* The "per launch" figures in the earlier §1.9 record are per-call
  means over a reconstruction's pixel ladder (512 calls at 12,051
  pixels, 128 at 48,203, 32 at 192,810, 8 at 771,240).  A single
  full-pixel launch at width 1008 costs about 859 ms.  The probe
  rebuilt the ladder mean (40.4 against 41.4 ms) to reconcile the two
  records.
* The per-launch fixed costs measured inside every arm, for anyone
  subtracting them: the horizontal-fan contract build is about
  2.3 to 2.6 ms and the zeroed output slab 0.04 to 0.31 ms.
* Nsight Compute durations are replay durations, not wall times; the
  timing leg owns time.
* No artifacts are staged; each arm computes its reference in its own
  process.  Only the jsonl rows and the ncu logs are written.
