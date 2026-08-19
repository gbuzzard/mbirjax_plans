# mg34 and mg35 run record

The composed gate on the sorted-contraction forward, both halves.
The finding is in `plans/torch_port/active/multigpu_findings.md`
§1.31; this file holds the run detail.  The tree under test in both
jobs is 64dedb8 plus the staged sorted-forward change
(triton_parallel.py and two test files), synced per file by md5.

## mg34, the suite and the standard-cell A/B (job 15346797)

* Node h006, four H100s, 22 minutes.  Exit 0.
* The full GPU suite ran first and read rc 0, with the sorted route's
  own gates in it (the kernel pair, the sparse-set fallback, the
  view-chunk tail, the switch contract).  The CPU suite on the same
  staged tree read 584 passed, including the four new sorted-route
  gates.
* Twelve composed arms at mg27's protocol, reusing mg27's staged
  sinograms by md5: parallel, the 512- and 1024-class cells, one,
  two, and four devices, sorted against per-tap.  Every arm realized
  its pinned count; both value gates held everywhere (each arm
  against its own route's one-device arm, and the two routes against
  each other, all inside 1e-3).
* The A/B: 1024-class 1.89x, 1.66x, 1.43x at one, two, four devices
  (21.19, 14.25, 10.75 s sorted); 512-class 1.41x, 1.17x, 1.03x.
  Peaks essentially unmoved.
* Rows: `rows/mg34_sorted_ab_h006_20260818_183711.jsonl` (md5
  fff405243d5543cfc51cab9add2adf8c, verified after copy).

## mg35, the 2048-class confirmation (jobs 15347106 and 15347172)

* The first submission failed in 67 seconds at staging, and the
  failure is a recorded finding: the 2048-class direct forward
  projection, under a four-device pin on the automatic branch,
  settled on ONE device and ran out of memory assembling the 30.5 GiB
  sinogram beside the 29.6 GiB phantom -- it neither widened nor was
  refused.  The observation is filed under C5 (the direct-entry
  preflight under-prices this call); the fix is an explicit
  four-device layout in the staging, which shards all three arrays.
* The second submission: node h005, four H100s, 56 minutes, exit 0.
  Staging generated and checksummed the 30.5 GiB sinogram; all eight
  arms ran (three and four devices, the counts the capacity table
  admits at this class); the mode-pair value gate held at both
  counts; no thermal findings.
* The A/B at the 2048 class: three devices 239.10 s per-tap against
  134.84 sorted (1.77x); four devices 188.98 against 113.25 (1.67x).
  The busiest-device peaks are identical between routes at both
  counts, so the sorted route's residents are invisible at this
  scale.  The 24 GB values block was the sorted gather's one
  unmeasured territory, and it holds the same win class as the
  standard cells.
* Rows: `rows/mg35_sorted_2k_h005_20260818_190130.jsonl` (md5
  d21822fb2c5e4e2d1cec68ce9de35fde, verified after copy); the first
  submission's error rows remain on scratch beside them.

## The gate's verdict

Both halves passed whole.  The staged change meets the design note's
shipping condition (increment 2), and what follows the commit is
recorded there: the floors staleness note fires on
triton_parallel.py, the cone and parallel rows owe a refresh, and
the comparison tables (execution_overview §5, the user-docs table)
owe a re-anchor because their mbirtorch columns measured the per-tap
route.
