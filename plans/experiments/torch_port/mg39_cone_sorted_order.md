# mg39 run record

Rework candidate A of design note §9: the two-axis sorted pixel
order fed to the UNCHANGED shipped cone forward kernel.  The finding
is in `multigpu_findings.md` §1.33; Greg's candidate ranking and
decision are §9's closing paragraphs.

* Job 15362636, node h003, one H100, minutes of measurement.  Exit 0
  with every instrument healthy: all four arms within 1e-5 of the
  raster baseline at every ladder point (worst 3.1e-6), the full
  mask realized, the Triton wrapper bound.
* The verdict: every arm loses at every point.  Best at the full
  mask: one compound sort for the whole 64-view batch, 546.96 ms
  against the raster baseline's 474.50 -- 0.87x, with the sort
  itself only 3.3 ms.  The chunked arms read 0.83x to 0.85x and the
  channel-only ablation 0.61x.  Subset points: 0.80x to 0.90x for
  the batch sort, worse chunked.  Sorting the pixels does not help
  this kernel: its raster order is already spatially local for an
  output-anchored (pixel tile, row tile, view) grid, and the
  two-axis order buys nothing the caches were not already
  delivering (mg31: L2 hit 87.2 percent in raster order).
* Two harness notes for the next reader.  The chunked arms' printed
  sort_ms is contaminated: the per-chunk timing block's synchronize
  absorbs the previous chunk's asynchronous kernel work, so only the
  arm TOTALS are trustworthy there (the batch arm's 3.3 ms split is
  clean).  And the bounded ncu pass misfired: its launch-skip
  landed on a small early launch of the same kernel name (the
  availability self-check runs one on first use), so the profiled
  numbers describe a tiny launch, not the full-mask call; the
  mechanism reading was not taken.  Neither gated anything.
* Rows: `rows/mg39_cone_sorted_h003_20260819_094504.jsonl` (md5
  044ade34264ae37a55799a375632f96b, verified after copy).
