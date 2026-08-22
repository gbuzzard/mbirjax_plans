# mg56 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.47; this file
holds the run detail.

## mg56, the full widening-floors refresh (job 15435735)

* 2026-08-22, node with four H100 80GB, 38 minutes 51 seconds by
  sacct, exit 0.  The refresh script measured 44 arms; the
  verdicts and the paste-ready block are in the job log
  (mg56_15435735.log), and the per-arm files remain under
  /scratch/gautschi/buzzard/torch_p3/results/mg56_floors.
* The tree: the committed tip c024ec9 (the multiaxis kernel
  work), rsynced to mbirtorch_src with md5 verification and
  editable-installed by the job.  Stale families at submission:
  cone, denoiser, multiaxis, parallel, translation -- every
  family, because eight cost inputs had moved since mg48.  The
  witnesses held, including the multiaxis kernel-selection
  witness.
* The verdicts, crossover against the next admitted smaller
  count, margin 1.15x over the cell's warm spread:

  | family, count | readings across the probes | floor |
  |---|---|---|
  | cone, 2 | 0.88x / 1.31x / 1.65x | holds at the 512-class |
  | cone, 4 | 1.15x / 1.64x | MOVES DOWN to the 768-class |
  | denoiser, 2 | 0.73x / 0.87x / 0.93x | stays a sentinel |
  | denoiser, 4 | 0.55x / 0.75x / 0.90x | stays a sentinel |
  | multiaxis, 2 | 0.86x / 1.38x / 1.72x | holds at the 512-class |
  | multiaxis, 4 | 0.51x / 0.70x / 1.09x | RISES to the 1024-class |
  | parallel, 2 | 1.09x (in 11.2% spread) / 1.46x / 1.59x | holds at the 768-class |
  | parallel, 4 | 0.82x / 1.40x | holds at the 1024-class |
  | translation, 2 | 0.64x / 1.19x / 1.25x | holds at the middle scan |
  | translation, 4 | 0.79x / 1.15x | holds at the production scan |

* The multiaxis four-device row could not be placed by this
  instrument alone: its thin 1.09x at the 768-class sits under
  the margin and the ladder tops out there.  The row was placed
  by hand at the 1024-class on mg55's reading (job 15434826, the
  same seeded warm-median protocol): four devices beat two at
  1.64x with 0.1 percent spread.  The coarse rule's round-up of
  the thin 768-class win lands on the same cell.  The row's note
  in _widening_floors.py carries both instruments.
* The paste: the FLOORS dict, the fifteen BLESSED_COST_HASHES
  (triton_multiaxis.py newly recorded), STALE_SINCE None, and the
  TABLE_CHECKSUM, all re-recorded together with commit c024ec9
  stamped on every row and the notes written by hand.  After the
  paste: stale_note() reads None, stale_families() is empty,
  tests/test_widening_floors.py passes at 30, and the full suite
  reads 696 passed.
* One test moved with the floors.
  test_multiaxis_floors_admit_at_the_512_class asserted the
  torch-body era's shared 512-class floor; it is now
  test_multiaxis_floors_split_by_count and asserts the measured
  split (two devices from the 512-class, four from the
  1024-class, both refused below).
