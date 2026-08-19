# mg40 run record

One run.  The finding is in
`plans/torch_port/active/multigpu_findings.md` §1.34; the refreshed
parallel rows are pasted into `mbirtorch/_widening_floors.py`
(staged).  This file holds the run detail.

## mg40, the first family-scoped floors refresh (job 15369689)

* Node h018, four H100s, 7.5 minutes of measurement (about 9 minutes
  of wall with the install and witnesses).  Exit 0.  mg26, the last
  full refresh, ran 2 hours 57 minutes; this run measured the one
  family whose cost inputs moved.
* The tree is f9fde0a plus the staged per-family floors machinery,
  synced per file and verified by md5 before submission.  The
  projection code is c761b24's: the channel-sorted parallel forward,
  with every other cost input unchanged since mg26's 64dedb8.  The
  job asserted three witnesses before any GPU work: the padded width
  rule, the sorted forward's default-on switch, and
  `stale_families() == ['parallel']`.
* The scope resolved from the hashes, not from a hand-typed list:
  the job ran `--families` bare, and the tool measured the parallel
  family and carried cone, multiaxis, translation, and the denoiser
  verbatim.
* The plan was 5 (family, count, cell) rows, 10 timed arms after
  dedup, and 4 generators, from the coarse table's brackets.
* The crossover verdicts, against the coarse 1.15x admission margin:
  - parallel n=2: 0.618x / 0.969x / 1.379x at the 384-, 512-, and
    768-class cells; THE FLOOR MOVED UP from the 512-class to the
    768-class.  The 512-class two-device win (1.20x on the per-tap
    kernel in mg26) is gone outright, not merely thinned.
  - parallel n=4: 0.803x / 1.324x at the 768- and 1024-class cells
    against n=2; the floor holds at the 1024-class.  mg26 read a
    thin 1.06x win at the 768-class and the margin rule rounded it
    up; the sorted kernel turned that cell into a 0.80x loss, so the
    rounding anticipated the kernel change.
* Per-cell warm medians for the parallel arms (seconds, median of
  three after a discarded cold pass): 384-class 0.60 / 0.96 at n=1 /
  n=2; 512-class 1.31 / 1.35; 768-class 4.80 / 3.48 / 4.34 at n=1 /
  n=2 / n=4; 1024-class 21.23 / 14.27 / 10.78 at n=1 / n=2 / n=4.
  The 1024-class values reproduce mg34's composed A/B (21.19 /
  14.25 / 10.75) within run noise.
* After the paste, `--bless` recorded the current hashes;
  `triton_parallel.py` is the one input whose hash moved since mg26,
  which is the scoped mode's premise read off the tool itself.
* `tests/test_widening_floors.py` (30 tests) and
  `tests/test_device_policy.py` pass on the pasted table; the two
  tests that named the old 512-class parallel floor now name the
  768-class one, and the fallback-family test widens at the
  768-class cell.
* Arm records: `results/mg40_floors/` on scratch (torch_p3); the
  verdicts and the paste are in the job log (`mg40_15369689.log`).
