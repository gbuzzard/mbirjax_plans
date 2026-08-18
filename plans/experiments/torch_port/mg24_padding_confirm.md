# mg23 and mg24 run record

Two jobs, one gate campaign: the band-padding remedy's gate run and its
composed confirmation (back_remedy_design.md §7, increments 2 and 3).
The finding and its tables are in
`plans/torch_port/active/multigpu_findings.md` §1.23; this file holds
the run detail.  Both jobs ran the merged tree plus the padding
implementation, shipped file by file with md5 verification before Greg
committed it as 64dedb8.

## mg23, the gate run (job 15336959)

* Node h004, two H100s, 11 minutes, exit 0.  Three legs, serial.
* Leg 1, the suite on GPUs: 674 passed, 4 skipped -- every CUDA-gated
  test ran, including the new padded value tests (1e-5 relative
  against the torch bodies and windowed references) and the stride
  tests that pin the no-op property at divisible bands.  One
  known-benign warning: h004's device pair failed the direct
  device-to-device round-trip probe, so `test_sharding` exercised the
  host-bounce path; that is the library's own fallback working.
* Leg 2, the cone band sweep (mg21b re-run): the three non-divisible
  bands read 19,800 to 20,900 ns per view-slice against the divisible
  bands' 18,650 to 21,500 -- medians 20,359 against 19,252, a ratio
  of 1.06 where the unpadded tree measured 2.44.  The residual is the
  discarded padded lanes, 1.6 to 3 percent by arithmetic.  The
  slice-start arms stayed flat.  Witnesses read exactly zero on the
  shared slices, across compilations.
* Leg 3, the parallel back sweep (mg23_parallel_band.py): medians
  3,657 against 3,419 ns per view-row, a ratio of 1.07, flat across
  the divisibility boundary as predicted.  The device sampled
  hot-or-throttled during this leg and the row says so; a flat
  reading is unaffected by a level shift.
* Rows: `rows/mg21b_band_gpu_h004_20260818_064628.jsonl` and
  `rows/mg23_pband_gpu_h004_20260818_064809.jsonl`.

## mg24, the composed confirmation (job 15337015)

* Node h017, four H100s, 35 minutes 38 seconds, exit 0, chained after
  mg23.  Four arms: the generator (verified mg19's staged cone
  artifacts by checksum; no projection repeated), then composed
  three-iteration reconstructions at n=3 and twice at n=4,
  calibration on, values gated at 1e-4 before each (readings at
  2.2e-7).
* **The back busy times, against mg19 and the note's projections:**

  | arm | band | this run s | mg19 s | projection |
  |---|---|---|---|---|
  | n3_shipped | 672 (no pad) | 137.1 | 136.8 | unchanged |
  | n4_shipped | 504 -> 512 | 106.4 | 227.8 | near 100 |
  | n4_shipped_repeat | 504 -> 512 | 104.2 | 228.2 | near 100 |

  The four-device fall is 53 to 54 percent, and 104/137 = 0.76 sits
  on the ideal work ratio of 0.75, so back scaling is monotone again.
* The forward at n=4 read 127.9 s at the shipped batch of 32768,
  reproducing mg19's own 32768 arm to the tenth of a second.  The n=3
  forward (172.7 s) has no mg19 comparator at this batch, and the
  record says so rather than inventing one.
* Calibration: every ratio inside the (1.00, 1.30) band -- 1.185 to
  1.187 at n=3, 1.114 to 1.184 at n=4 -- with the ledger's PADDED
  band charges, which is the at-scale validation of the §4 ledger
  rule.
* The slab context: mg19 ran a 64 MiB combining slab, this tree ships
  256 MiB; mg19's own 256 MiB arm read 226.1 s against its 64 MiB
  pair's 227.9/228.2, so the slab accounts for about 2 s of the
  124 s fall.
* Rows: `rows/mg24_padding_h017_20260818_065201.jsonl`.

Notes a later reader may want:

* mg24 is mg19's harness trimmed to four cone arms and repaired for
  the merged tree: the removed `_column_gather_forward` witness and
  the removed environment variable are gone, the batch witness is
  recorded rather than gated (the arms run the shipped batch by
  design), and the tree witness checks the cylinder-transfer names
  plus `padded_kernel_width(504) == 512`.
* The floors staleness note fires on this tree (the two triton files
  are cost inputs) and is recorded on every row; the re-measure is
  the next session's second item.
* The 1024-class two-device back reading -- the last line of B1's
  closing condition -- came from the first nightly on the padded tip
  the same morning (job 15338176): the per-call row fell from 5.78 s
  to 2.37 s and its speedup against one device went from 0.78x to
  1.92x.  Findings §1.23 carries the reading and the one priced
  memory alert that rode the same run.
