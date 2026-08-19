# mg42a run record

Two runs of one harness.  The finding is in
`plans/torch_port/active/multigpu_findings.md` §1.35; the design this
probe serves is `plans/torch_port/active/ledger_calibration_design.md`.
This file holds the run detail.

## mg42a, the ledger probe (job 15376256), plus the n=1 re-run
## (job 15377054)

* 2026-08-19, node h004, four H100s.  The main run measured 8 of its
  10 arms in 5.7 minutes and exited 2: both one-device arms crashed in
  the harness's post-reconstruction bookkeeping.  The defect was the
  probe's own -- `shard_ranges()` called without an axis length, which
  only the TRIVIAL single-device CUDA placement lacks.  The CPU smoke
  could not catch it, because the smoke's explicit device list builds a
  placement that carries the length.  The fix passes the lengths
  explicitly, as the reference-timing harness always did, and the
  re-run recovered both arms cleanly (rc 0, 4.4 minutes).  Both
  one-device arms' calibration tables had also printed in the main run
  before the crash, and the re-run reproduced them (0.937 against
  0.935, 1.106 against 1.104).
* The instrument: reset-free watermark sampling around seven wrapped
  library seams, with the calibration mode on in every arm and its one
  reset stamped as an epoch marker.  Every seam attached in every arm,
  every region a count can exercise fired, and the whole-run peak is
  the max over samples by construction.
* The staged sinograms were mg27's, reused by md5.

## The readings, worst device per arm

| arm | modeled/measured | verdict | dominant modeled phase |
|---|---|---|---|
| b512_parallel_n1 | 0.935 | UNDER | initial dot products |
| b512_parallel_n2 | 1.298 | ok | subset delta forward projection |
| b512_parallel_n3 | 1.436 | over | initial forward projection |
| b512_cone_n1 | 1.104 | ok | direct recon (back loop) |
| b512_cone_n2 | 1.328 | over | direct recon (back loop) [back workers] |
| b512_cone_n3 | 1.427 | over | direct recon (back loop) [back workers] |
| a1024_parallel_n2 | 1.160 | ok | subset delta forward projection |
| a1024_cone_n2 | 1.104 | ok | subset delta forward projection |
| c1024_parallel_n4 | 1.188 | ok | subset delta forward projection |
| c1024_cone_n4 | 1.130 | ok | subset delta forward projection |

Band 1.00 to 1.30.  Full per-device tables, per-phase ledgers, term
breakdowns, and region aggregates are in the two jsonl files:
`rows/mg42a_ledger_h004_20260819_160056.jsonl` (the main run) and
`rows/mg42a_ledger_h004_20260819_161008.jsonl` (the n=1 re-run).

## What each question answered

* **Question A (the back pair).**  The attribution data is captured.
  At the 1024-class two-device arms the back-projection region's
  largest transient reads 5.72 GB on both devices and both geometries,
  and the region that moved the watermark furthest in nearly every arm
  is the sharded back projection.  The term arithmetic against the
  ledger's back charges is the term-change increment's work; the
  kernel cost functions import on the Mac, so it needs no further GPU
  time.
* **Question B (the three-device over-read).**  Reproduced at 1.436
  (parallel) and 1.427 (cone), and attributed differently by geometry.
  Cone's dominant phase is the direct-recon back workers, and its top
  term is the back batch at 0.74 GB of a 1.33 GB phase -- the same
  charge question A measures, so the first two inputs close together.
  Parallel's dominant phase is the initial forward projection, whose
  top terms are the deliberate covers: the doubled forward output
  (0.22 GB), the forward batch (0.22 GB), and three resident cylinder
  batches (0.16 GB), which loom large at small shards.
* **Question C (the lead-device transient).**  Does not reproduce.  In
  a fresh process the four-device 1024-class arms peak device 0 at
  6.84 GiB (cone) against the nightly's 26.6 GiB reading and the
  one-device arm's own 23.4 GiB.  The placement trail shows shard-sized
  steps only.  The nightly reading was its harness's cumulative
  watermark accumulating across arms in one process; the input closes
  with the note landing on the nightly's comparison method, and no
  ledger term is owed.
* **A new finding: parallel n=1 reads UNDER, 0.935.**  Modeled 1.96 GB
  against a measured 2.10 GB at the 512-class, reproduced across both
  runs.  The model's peak phase is the initial dot products; the
  measured watermark accumulates inside the back worker region (its
  growth is the largest, 0.55 GB).  Cone n=1 is in band at 1.104.  An
  under-read is the one direction the ledger must not err in, so this
  finding leads the term-change increment.  Whether it predates the
  channel-sorted kernel is not in tonight's evidence; the increment
  should read the same arm on the per-tap route
  (MBIRTORCH_SORTED_FORWARD=0) to split kernel from model.
