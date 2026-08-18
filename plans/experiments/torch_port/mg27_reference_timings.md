# mg27 run record

One run.  The refreshed comparison tables are in
`plans/torch_port/active/execution_overview.md` §5.1 and §5.2, with
provenance in §5.3; the user-docs table is in
`mbirtorch/docs/source/usr_multi_gpu.rst`.  This file holds the run
detail.

## mg27, the reference-timing re-anchor (job 15342577)

* Node h012, four H100s, 19 minutes 49 seconds of wall, on the padded
  64dedb8 tree (verified per-file by md5 before submission), chained
  behind mg25 on the shared environment.
* Exit 0.  All twelve variants ran (parallel and cone, the 512-class
  and 1024-class cells, one, two, and four devices), each realized its
  pinned count, bound Triton kernels in both directions, and read an
  md5-verified staged sinogram.  No thermal or throttle findings.
* The protocol is mg1's: shepp-logan low-dynamic-range phantom,
  exponential weights, seed 13 before every call, 3-iteration recon
  with the stopping threshold disabled, one discarded cold pass, warm
  median of three.  Each variant ran in a fresh subprocess pinned
  through `MBIRTORCH_NUM_DEVICES`, all counts of a cell reconstructing
  one staged sinogram.
* Cross-count value fingerprints (float64 sum of absolute values and
  sum of squares) agreed with the one-device row at 1.6e-10 to 4.0e-8
  relative, far under the 1e-3 gate; every split at these cells is
  even, so the uneven-split reduction-order term never entered.
* The warm medians, seconds, with the busiest device's peak:

  | cell | geometry | n=1 | n=2 | n=4 |
  |---|---|---|---|---|
  | 1024-class | parallel | 39.98 (22.87 GB) | 23.84 (11.77 GB) | 15.50 (6.23 GB) |
  | 1024-class | cone | 61.56 (22.95 GB) | 35.13 (12.47 GB) | 22.34 (6.84 GB) |
  | 512-class | parallel | 1.85 (1.93 GB) | 1.49 (1.06 GB) | 2.15 (0.62 GB) |
  | 512-class | cone | 2.74 (2.15 GB) | 2.13 (1.33 GB) | 2.80 (0.87 GB) |

* Readings against the stale references: every one-device time
  reproduced its recorded value (39.98 against 39.90, 61.56 against
  61.66, 2.74 against 2.74), which turns the earlier trivial-placement
  code argument into a measurement.  The cone multi-device rows took
  the padding's gain (54.86 to 35.13 s at two devices, 32.37 to 22.34
  at four), and cone is now faster than the recorded mbirjax column at
  both multi-device counts.
* Output rows:
  `rows/mg27_reference_h012_20260818_121856.jsonl` (md5
  7dcfbc0ad5d0911aac4c2b58c052d36a, verified after copy).  The staged
  sinograms remain on scratch under `results/mg27_reference/` for
  repeat runs.
