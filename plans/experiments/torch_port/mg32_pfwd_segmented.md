# mg32 run record

Two submissions of one spike (the design note's increment 1, approved
2026-08-18).  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.29; this file holds
the run detail.

## First submission (job 15346037): a compiler constraint, and one
## accidental finding

* Failed by design visibility, 74 seconds: Triton's structured
  branching requires one shape per variable name across an if/else,
  and the contraction branch's `out_ptrs` was (WINDOW, BLOCK_R)
  against the fallback's (BLOCK_P, BLOCK_R), so every configuration
  with WINDOW different from BLOCK_P refused to compile.  The one
  coinciding configuration (window 16, tile 16) ran and passed values
  at 1.18e-6 through both paths and their seam.
* The accidental finding: the channel span is ANGLE-DEPENDENT.  At
  oblique views, consecutive raster pixels project onto nearly the
  same channel, so even a 16-channel window took the contraction on
  87.5 percent of (tile, view) pairs.  The "forced fallback"
  configuration was therefore really a mixed arm, and it is relabeled
  so (mx_).

## Second submission (job 15346085): the sweep

* Node h002, one H100, 1 minute 46 seconds.  Exit 0.  The shipped
  baseline reproduced the anchor (864.7 ms), and every gated
  configuration passed values at 3.6e-6 to 3.7e-6 -- the contraction
  path live, against the first submission's fallback-only reading.
  Fixes carried: the branch-name rename, and a span mask on the
  window's atomic adds so the power-of-two window does not inflate
  the add count.
* The full-mask launch, the headline: the best gated configuration
  (16-pixel tile, 32-channel window, 128 columns, chunk 32) reads
  515.7 ms against the shipped 864.7 -- 1.68x -- and the TF32 twin of
  the same shape reads 470.4 ms -- 1.84x -- at 6.69e-4 worst
  relative values, the tensor-core default's 10-bit-mantissa class
  measured (Greg's question; reported, not gated).
* The counter confirmation on the winner: atomic-path sectors fell
  from the shipped 5.597e10 to 2.120e10, which is 2.64x against the
  predicted 2.4x to 2.8x window; the memory-wait stall fell from
  38.2 to 3.4 warps per issue-active cycle; SM throughput rose from
  15.6 to 38.7 percent; DRAM read held at 6.2x the values block (the
  chunk's amortization).  The mechanism worked exactly as designed.
* The subset points are the other half of the verdict: every subset
  ladder point read 0.87x to 0.88x, so the production-call mixture
  breaks even (0.99x) and a size-gated selection (segmented at the
  full mask, shipped below) reads 1.11x.  The mechanism: the
  harness's subsets are strided, so a 16-pixel tile's pixels sit 4 to
  64 raster columns apart and span far past any window -- the
  contraction cannot serve channel-nonlocal tiles, and production
  VCD subsets are random draws with the same nonlocality.  Subset
  calls fundamentally belong to a tap path; the contraction's
  domain is full-mask and locality-preserving calls.
* The 32-pixel tile is disqualified on its timings (0.49x to 0.65x
  mixture): the 64-channel window's dot and adds outgrow the win.
* One harness defect stands, and the values carry the truth past it:
  the fallback-rate counter read exactly 12.5 percent everywhere,
  including subset points whose spans guarantee near-total fallback.
  The subset values passing at 3.7e-6 proves the fallback path
  executed there (a contraction on a span past the window would miss
  most taps and fail by orders of magnitude), so the counter
  under-counts by about the column-tile factor and the RATE numbers
  are unusable; the path routing is sound.  The counter accounting
  is to be fixed before any library step leans on rates.
* Rows: `rows/mg32_pfwd_segmented_h002_20260818_172015.jsonl` (md5
  5ead04ec88bdef3a2ce372bd85da7f13, verified after copy); the first
  submission's partial rows remain on scratch
  (`mg32_pfwd_segmented_h002_20260818_171505.jsonl`).
