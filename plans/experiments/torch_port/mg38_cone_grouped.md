# mg38 run record

The spike on the cone pixel-batched forward with two-axis grouping
(design note `pfwd_segmented_design.md` §9).  The verdict is a loss;
the finding is in `multigpu_findings.md` §1.32.  The harness was
authored by an Opus agent from §9 and the mg33 template, reviewed
and submitted after an independent local smoke.

## The run (job 15361079)

* Node h003, one H100, about 50 minutes.  Exit 0 -- the exit code is
  instrument health, and every instrument held: the arithmetic
  preflight (a torch twin of both kernel branches, 4.9e-7 against
  the torch body with both branches exercised), the 1e-5 values gate
  at every (configuration, ladder point) against the shipped wrapper
  (worst 9.0e-6), no configuration raising, and the body-selection
  preflight.
* The cell is the 1024-class cone (mg31's construction), 64-view
  batches; ladder full mask, /64, /128 strided subsets; eight
  grouped configurations (windows 16x8 and 32x16, chunks 8 and 16,
  slice tiles 4 and 8) plus a forced-fallback arm; medians of three.
* Baseline: the shipped per-tap wrapper at 477.13 ms per full-mask
  64-view launch (consistent with mg31's 388.25 ms at 52 views).

## The verdict

* Every configuration loses at every point.  Best at the full mask:
  16x8 window, chunk 8, slice tile 8, at 1968.37 ms -- 0.24x.  The
  wide-window arms read 0.07x to 0.13x; the forced fallback 0.03x.
  Subset points fall back entirely (fallback fraction 1.000) and
  read 0.03x to 0.04x -- the disengagement evidence §9's rider asks
  for, measured.
* What VALIDATED despite the loss: the two-axis grouping fit exactly
  as mg37 predicted (window-path tiles 99.8 to 99.9 percent at the
  full mask -- fallback fraction 0.001 to 0.002); the window flush
  cut the atomic sectors about 5x (4.542e9 against the shipped
  2.3e10); the sort and gather cost 1.2 to 3.4 ms; values held at
  1e-5 through the inverted arithmetic; the helical z-offset term
  rode intact.
* What FAILED is the design's one prize: the read amortization.  The
  counter pass on the winner read DRAM at 73.7x the recon block per
  64-view launch -- the shipped kernel's once-per-view signature
  (64 x 1.15), not the once-per-chunk the design predicted (about
  8x).  The values tile did not stay resident across the view
  chunk.  The compile-record introspection returned None for
  registers, spills, and shared memory (a harness gap), so whether
  the tile spilled to local memory or was rematerialized per view
  is not distinguished by this run.  With reads unamortized, the
  arithmetic tax of the window path (about 5x per-voxel operations
  at the cheap arm) was paid on top of the shipped kernel's own
  traffic, and the counter shows the result balanced at SM 50.2
  against memory 51.4 percent of peak.
* Rows: `rows/mg38_cone_grouped_h003_20260819_085812.jsonl` (md5
  67db785b42f50a852332b0a4a3f2d510, verified after copy).

## Disposition

Recorded as a measured negative; §9's status carries it.  The one
bounded diagnostic left open, if Greg wants it: fix the compile
introspection and read whether the values tile spills, since a
register-pressure fix is the only path on which the design's
premise could still be tested.  Absent that, the increment closes:
even with reads fixed, the arithmetic tax against the 2x ceiling
left a thin margin, and the measurement says the hardware never
delivered the premise.
