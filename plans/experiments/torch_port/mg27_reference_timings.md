# mg27 run record

Two runs of one harness.  The refreshed comparison tables are in
`plans/torch_port/active/execution_overview.md` §5.1 and §5.2, with
provenance in §5.3; the user-docs table is in
`mbirtorch/docs/source/usr_multi_gpu.rst`.  This file holds the run
detail; the newest run is first.

## mg27 re-run, the sorted-kernel re-anchor (job 15369703)

* 2026-08-19, node h018, four H100s, 16.6 minutes, chained behind
  mg40 on the shared environment.  Exit 0; every variant healthy, no
  thermal or throttle findings.
* The tree carries the channel-sorted parallel forward (c761b24 on
  the f9fde0a tip, synced per file by md5).  The job asserted the
  padding witness, the sorted-forward witness, and the absence of the
  route override before any variant ran.  This run absorbs the fourth
  recorded change since the original references; the header names all
  four.
* The staged sinograms were REUSED from the 2026-08-18 run, verified
  by md5 on every load, so both runs reconstruct identical arrays and
  their columns compare directly.
* Cross-count value fingerprints agreed with the one-device row at
  4.1e-8 relative or better on every cell.
* The warm medians, seconds, with the busiest device's peak:

  | cell | geometry | n=1 | n=2 | n=4 |
  |---|---|---|---|---|
  | 1024-class | parallel | 21.26 (22.87 GB) | 14.24 (11.77 GB) | 10.84 (6.30 GB) |
  | 1024-class | cone | 61.65 (22.95 GB) | 35.06 (12.47 GB) | 22.33 (6.84 GB) |
  | 512-class | parallel | 1.31 (2.10 GB) | 1.29 (1.14 GB) | 2.10 (0.70 GB) |
  | 512-class | cone | 2.75 (2.15 GB) | 2.14 (1.33 GB) | 2.80 (0.87 GB) |

* Readings against the 2026-08-18 column: parallel took the sorted
  kernel's whole gain (1.88x, 1.67x, and 1.43x at one, two, and four
  devices at the 1024-class; 1.41x and 1.16x at the 512-class's one
  and two), and cone reproduced within run noise at every cell, which
  is what a parallel-only kernel change predicts.  The 1024-class
  parallel values also reproduce mg34's composed A/B and mg40's
  floors arms within noise -- three harnesses, one reading.
* Output rows: `rows/mg27_reference_h018_20260819_130958.jsonl` (md5
  5bbcab9ca644adb150849d1d8d7fc779, verified after copy).

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
