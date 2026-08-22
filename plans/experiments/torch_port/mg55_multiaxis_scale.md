# mg55 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.46; this file
holds the run detail.

## mg55, device counts and the 2048-class (job 15434826)

* 2026-08-22, node h013, four H100 80GB, 32 minutes 38 seconds by
  sacct, exit 0, instrument health "healthy".  The one recorded
  finding is the two-device 2048-class arm's out-of-memory, which
  is that arm's result and not a failure.
* The library under test is the candidate tree at
  /scratch/gautschi/buzzard/torch_p3/mbirtorch_dev through
  PYTHONPATH, asserted before any work; torch 2.13.0+cu130,
  triton 3.7.1.  Every arm asserted both multiaxis Triton kernels
  bound.  Tree witnesses held, including the selection witness.
* Part A reused the staged 1024-class input
  (md5 798c72e1cf5bb7803b9f2b02294753c6, the same bytes mg52 and
  mg54 measured).  Arms at explicit device lists, fresh
  subprocesses, seed 13, cold 3-iteration reconstruction plus
  warm median of three:

  | devices | cold s | warm s | spread | peaks per device GB |
  |---|---|---|---|---|
  | cuda:0 | 70.34 | 67.63 | 0.1% | 24.11 |
  | cuda:0,1 | 47.55 | 37.43 | 0.1% | 12.95 / 12.44 |
  | cuda:0,1,2,3 | 39.31 | 22.75 | 0.0% | 7.49 / 7.39 / 7.39 / 6.64 |

  Fingerprint gaps across counts: 3.42e-09 (two devices) and
  5.44e-09 (four) relative to the one-device arm.  All three
  counts read the same forward-model error trajectory (1.255,
  0.5631, 0.2752).  Reserved peaks: 36.99, 22.09, 12.77 GB.
* Part B built its own 2048-class input in-process, per arm, with
  the kernels (phantom 13.3 s, forward projection 51.1 s, weights
  in blocks; staging 89.9 s total).  The memory preflight was off,
  with the reason on the row.
  - Four devices: cold 355.95 s, warm 298.81 s at 0.1 percent
    spread, per-device peaks 50.59 / 50.27 / 50.27 / 48.61 GB,
    reserved 76.19 GB busiest, forward-model error 1.237, 0.515,
    0.2393 (fell).  The job log carries caching-allocator retry
    warnings during this arm; the retries succeeded and the arm
    completed cleanly, so the arm ran near the node's edge and
    inside it.
  - Two devices: out of memory (a 13.25 GiB allocation against
    522 MiB free with 78.66 GiB already held), recorded as the
    arm's result.
* Part C, the ledger read-only pricing beside the measurements:

  | cell | devices | modeled busiest GB | measured GB | ratio |
  |---|---|---|---|---|
  | 1024-class | 1 | 24.88 | 24.11 | 1.03x |
  | 1024-class | 2 | 13.80 | 12.95 | 1.07x |
  | 1024-class | 4 | 7.68 | 7.49 | 1.03x |
  | 2048-class | 1 | 194.35 | not run | - |
  | 2048-class | 2 | 103.08 | out of memory | - |
  | 2048-class | 4 | 54.08 | 50.59 | 1.07x |

  The modeled 103 GB per device at two devices exceeds the card,
  so the ledger agrees with the measured out-of-memory; the
  modeled 54 GB at four devices admits the run that completed.
* Output rows:
  `rows/mg55_multiaxis_scale_h013_20260822_111112.jsonl` (md5
  a7d04ee4d7c7a112453dd044f46793d7, verified after copy).
