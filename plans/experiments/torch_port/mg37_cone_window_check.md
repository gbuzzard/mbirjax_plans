# mg37 run record

The paper check for the cone two-axis grouping (design note
`pfwd_segmented_design.md` §9): does sorting pixels by (W_p_r
bucket, channel center) give tiles whose detector footprint fits a
small (channel x row) window?  No GPU and no kernel -- the windows
are computed exactly from the shipped geometry builders
(`_cone_horizontal_data`, `_cone_vertical_affine`) on the CPU.

* Run locally on Greg's Mac (miniforge env), 2026-08-19, about two
  minutes.  The 1024-class cone cell by mg31's construction; full
  mask, 771,240 pixels, 24,101 tiles of 32.  Reference angles
  0/45/90/135 degrees; chunks 8 and 16; W_p_r buckets 16/32/64;
  slice tiles 4/8/16; probe slices 0 to 991 (the affine is linear
  in the slice index, so endpoint evaluation is exact).
* The verdict: the windows close.  At 32 buckets, chunk 8 or 16,
  slice tile 4 or 8, at least 99.8 percent of tiles fit a 16-row x
  8-channel window (row p99 15.7, channel p99 7.3, per-view flush
  sizing); every swept combination fits 32 x 16 at 99.7 percent or
  better.  Bucket-boundary tiles: at most 31 of 24,101.  The
  16 x 8 accumulator patch is 512 bytes.
* The check's own lesson: the first pass sized windows for a
  per-CHUNK flush (the union footprint across the chunk's views)
  and read 27 channels at 8 views, 54 at 16 -- the common-mode
  rotation drift.  The design's per-view flush excludes common-mode
  drift by repositioning the window each view, and the within-view
  spread stays single-digit channels.  Both sizings are in the
  rows (`chan_window`/`row_window` per-view, `chan_union`/
  `row_union`); the first pass's union-only file is superseded and
  not staged.
* Rows:
  `rows/mg37_cone_window_Gregs-MacBook-Pro-2_20260819_075351.jsonl`
  (md5 dbe497863e9613618f2bf4fa0662d003).
