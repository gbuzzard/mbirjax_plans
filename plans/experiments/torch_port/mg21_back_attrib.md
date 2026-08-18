# mg21 and mg21b run record

Two runs.  The finding and its tables are in
`plans/torch_port/active/multigpu_findings.md` §1.21; this file holds
the run detail.

## mg21, the attribution probe (job 15327847)

* Node h017, four H100s, 15 minutes 52 seconds of wall, on the merged
  6d90601 tree synced to scratch and verified by per-file md5.
* Both arms ran (n=3 and n=4), every variant timed, exit 0.  The
  witnesses all held: each arm realized its pinned count, the back
  body was this run's wrapper around the Triton body (checked by
  identity, since the wrapper reports the original's name), no
  direction ran as general torch code, and the calibration variable
  was absent.
* The input was mg19's staged cone sinogram, checksum verified, placed
  once per arm through the model's own sharding call.  The phantom's
  bytes were never opened; its checksum is recorded, not verified.
* Instrument accounting: every pass produced exactly n_bands times
  n_workers view-range records and n_bands reduce records.  Twelve
  calls arrived without an attribution context (8 builder calls and
  4 body calls per arm), all from the forward projection the sinogram
  fallback path would have used; they carry no timings and touch no
  table.
* The value witness between counts (sum of absolute values and sum of
  squares per variant, float64 on device) agreed at 5e-14 to 9e-12
  relative, against a 1e-6 expectation.
* The channel-major copy microbench on device 0 read 0.440 ms for a
  (13, 2016, 1984) block, 945 GB/s counting the read and the write.
  One device makes 160 body calls per full-pixel pass at n=4, so the
  copy is 0.070 s per pass, which is the measured residual almost
  exactly.
* GPU health sampled before and after each arm: no throttle flags, no
  hot readings.
* Output rows:
  `results/mg21_back_attrib_h017_20260817_192528.jsonl` on scratch
  (torch_p3).

Numbers a later reader may want, busiest device, timed-repeat means:

| variant | n=3 wall s | n=4 wall s | n=3 kernel s | n=4 kernel s |
|---|---|---|---|---|
| full_p1 | 26.7 | 47.9 | 24.3 | 45.8 |
| full_p2 | 27.2 | 48.9 | 24.9 | 46.9 |
| sub4 | 6.7 | 12.6 | 6.3 | 12.1 |
| sub16 | 1.7 | 3.3 | 1.6 | 3.1 |
| sub64 | 0.49 | 0.87 | 0.39 | 0.76 |
| band_half | 29.8 | 48.7 | 26.6 | 45.8 |

* Builders 0.76 s, residual 0.07 s, accumulation 1.3 s falling to
  1.0 s, reduce 0.06 s, gap under 0.02 s -- at both counts, on the
  full-pixel variants.  The kernel is the only part that grows.
* Realized view batches: 13 on the full-pixel calls at both counts
  (the 2 GiB transient budget over 164 MB per view), with short
  remainder batches of 5 to 7.  Launch grids: 193023x11 at n=3,
  193023x8 at n=4 -- the grid is band-sized, not detector-sized.
* The reconstruction estimate from the bare calls (hessian plus g
  calls at P/g for g in 4, 16, 64) reads 112 s at n=3 and 207 s at
  n=4, beside mg19's composed 137 s and 228 s.  The gap is the
  spatially structured subsets and the two extra calls a real
  reconstruction makes; the shares are what this run is for.

## mg21b, the divisibility discriminator (job 15327968)

* Node h007, one H100, 2 minutes 6 seconds of wall.  Same tree.
* One process, eight arms: band lengths 672, 504, 512, 496, 336, 344,
  256, 252, interleaved divisible and not, at a fixed cell
  (256, 2016, 1984), the full 3,088,364-pixel mask, and view range
  (0, 256), called directly through
  `Projectors.sparse_back_project_view_range`.
* The prediction was recorded in the script before the run: divisible
  bands share one rate, non-divisible bands run 2.0 to 2.5 times
  slower, and the boundary is divisibility, not size.
* Measured, in nanoseconds per (view, slice) of work: 18,643 to
  21,514 on the five divisible bands; 46,194 to 50,665 on the three
  non-divisible ones; medians 19,245 against 46,990, ratio 2.44.
  The three near-equal-work bands split exactly at the boundary:
  496 and 512 fast, 504 slow.
* Timing spreads 0.0 to 0.1 percent over three timed repeats per arm.
  The value witness (each arm against the 672 arm on shared slices)
  read exactly zero on every arm.
* No throttle flags before or after.
* Output rows:
  `results/mg21b_band_gpu_h007_20260817_194843.jsonl` on scratch.

## mg21b addendum, the band-start arms (job 15328160)

* Node h006, one H100, 3 minutes.  The design review asked whether
  the band start shares the band length's effect: the start is also
  a specialized integer, and two of the four production band starts
  at four devices (504 and 1512) are not divisible by 16.
* The extended script re-ran the eight band arms (every rate
  reproduced the first run within 0.03 percent, on a different node)
  and added two arms at a fixed band of 512 with the start moved:
  start 0 read 18,645, start 504 read 18,652, and start 1008 read
  18,711 ns per view-slice.  The start does not matter; the band
  argument alone governs.  The start arms skip the values witness,
  because a shifted band overlaps the reference on a shifted range;
  the question was the rate.
* Output rows:
  `results/mg21b_band_gpu_h006_20260817_201343.jsonl` on scratch.

Notes a later reader may want:

* The two runs close arithmetically.  mg21's per-device kernel work
  falls to 0.75x from n=3 to n=4 (512 views over 683, same total
  slices), while its kernel time rises 1.88x; the implied efficiency
  ratio is 2.51, and mg21b measured 2.44 with everything but the band
  held.
* The 1024-class history reads the same way: bands 1008 and 336 at
  one and three devices are divisible by 16, bands 504 and 252 at two
  and four are not, which is where the cone back projection was
  measured slow.
* mg21's band_half variant is consistent too: at n=3 it moved the
  bands from 672 to 336, both divisible, and cost 12 percent (tile
  padding); at n=4 it moved 504 to 252, both non-divisible, and cost
  2 percent.
* No equal division of 504 is divisible by 16, so the existing
  `back_project_slice_band` knob cannot reach a fast band at the
  counts that are slow; only padding the kernel's band argument can.
