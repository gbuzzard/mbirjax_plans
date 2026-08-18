# mg33 run record

One run.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.30; this file holds
the run detail.

## mg33, the sorted-subset spike (job 15346414)

* Node h002, one H100, 1 minute 14 seconds, on the padded 64dedb8
  tree.  Exit 0.  The shipped baseline reproduced the anchor, every
  configuration passed values at 3.2e-6 to 3.3e-6 worst, and the
  fallback counter read zero everywhere: sorted tiles fit even the
  16-channel window at every ladder point.
* The design measured: per-view argsort of the channel centers and a
  gather of the contract arrays outside the kernel; inside, mg32's
  window contraction with the values tile gathered through the
  per-view permutation inside the view loop.
* All eight configurations won at every ladder point.  The winner by
  the subset mixture (32-pixel tile, 16-channel window, 128 columns,
  8 warps): 3.97x at the full mask (217.2 ms against 864.7), 3.85x,
  3.55x, and 2.77x at the /4, /16, /64 subsets, 3.30x on the subset
  mixture.  Wider windows lost ground to the narrow ones at every
  tile, as the sorted spans predict.
* The pre-registered expectation that the full-mask point would lose
  to the per-view gather was WRONG, and the reason is recorded: the
  sorted order at each view walks the dense mask in stripes, so
  consecutive sorted pixels are geometric neighbors, the values-row
  gather stays coherent, and DRAM read held at 11.2x the values
  block -- the same class as the chunk path -- while the atomic
  collapse dominated everything.
* The counter reading on the winner: atomic-path sectors fell from
  the shipped 5.597e10 to 1.773e9, which is 31.6x -- sorted 32-pixel
  tiles collapse into a 2-to-3-channel span, so the contraction acts
  as a deep segmented reduction.  The memory-wait stall read 2.6
  warps per issue-active cycle (shipped: 38.2), SM throughput 57.8
  percent, memory throughput 81.5 percent: the kernel now spends its
  memory budget on useful traffic.
* The sort's own cost, timed separately and UNAMORTIZED: 14.6 ms at
  the full mask, 3.5, 1.1, and 0.6 ms at the subsets -- small against
  the savings even paid per call, and production pays them once per
  (pixel set, view batch) per reconstruction, since VCD's subsets and
  the mask are fixed across iterations.
* The combined projection across paths reads 3.45x on the production
  mixture, and the sorted path is the selected path at EVERY point,
  which collapses the library step's selection design to one
  candidate question: always-sorted against the locality partition,
  decided by the composed A/B.
* Rows: `rows/mg33_pfwd_sorted_h002_20260818_175404.jsonl` (md5
  ab99cafad33b47eae09f67a65c51b3b0, verified after copy).
* Standing caveats: one device, the 1024-class cell, three-iteration
  warm launches; the 2048-class values block is 24 GB and its sorted
  gather behavior is unmeasured; the fallback-rate counter's
  under-count (found in mg32) is moot here only because the rate read
  zero; the composed re-gate owns the shipping decision.
