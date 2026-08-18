# mg29 and mg30 run record

Two runs, one increment: B7's spike step, ruled by Greg 2026-08-18.
The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.27; this file holds
the run detail.

## mg29, the view-loop spike (job 15345411)

* Node h002, one H100, 1 minute 51 seconds, on the padded 64dedb8
  tree.  Exit 0.  The shipped baseline reproduced mg20's anchor
  (757 to 864 ms across the sweep's baseline reads, anchor 859.13),
  and every configuration passed values at 5.6e-7 to 6.9e-7 against
  the shipped wrapper at every ladder point.
* Eight configurations of the whole-batch view loop (the values tile
  loaded once, all 128 views looped in-program).  Best mixture
  speedup 1.10x (tile 16x64, 4 warps); best full-mask 1.14x (tile
  8x128, 8 warps); the largest tile (64x128) lost at 0.72x.
* The counter reading on the winner: DRAM read fell to 12.2x the
  values block (from the shipped 130x), but DRAM WRITE rose from the
  shipped 0.50 GB to 262 GB per launch -- the 128 concurrently
  walked output planes stopped fitting L2 (hit 98.6 percent on
  reads, but the atomic adds spilled their planes), and the traffic
  moved from the read side to the write side almost one for one.
  Registers 44 per thread, occupancy 62.4 percent, no spills.
* Rows: `rows/mg29_pfwd_viewloop_h002_20260818_162605.jsonl` (md5
  10831b0ca88a24ecaf44a93f88f27eb2, verified after copy).

## mg30, the view-chunk spike (job 15345519)

* Node h001, one H100, 1 minute 46 seconds, same tree.  Exit 0; every
  configuration passed values at the same 6e-7 class.
* Eight configurations walking the interior between the shipped
  kernel (chunk 1) and mg29's whole batch (chunk 128): chunks 4, 8,
  16, 32 at the two best mg29 tiles.  The speedup rises monotonically
  with the chunk at both tiles; the winner is chunk 32 on the 8x128
  tile at 8 warps: 1.17x mixture, 1.19x full mask.
* The counter reading on the winner closed the traffic account.  DRAM
  read 19.3 GB (6.2x the values block; the amortization held), DRAM
  write 48.8 GB (down from mg29's 262, above the shipped 0.5),
  occupancy 98.8 percent, L2 hit 99.0 percent, registers 32, and the
  atomic-path sectors IDENTICAL to both other designs at 5.597e10.
  Total DRAM traffic fell 5.5x against the shipped kernel (377.5 to
  68 GB) while the time moved only 1.19x, and the stall reading rose
  to 50.7 warps per issue-active cycle: the kernel's floor is not
  DRAM traffic but the atomic path itself, 1.79 TB of 32-byte
  sectors through L2 per launch at about 2.5 TB per second, carried
  unchanged by every design because every design issues the same
  taps x pixels x width x views atomic adds.
* Rows: `rows/mg30_pfwd_viewchunk_h001_20260818_163404.jsonl` (md5
  839dec4e026ca66b375e7b88e7a04b2b, verified after copy), with the
  ncu log beside it on scratch.

## Notes

* Neither spike touched a library file; both variants live inside
  their harnesses, launched through a copy of the shipped wrapper's
  launch section with the grid and constants changed.
* One harness blemish, cosmetic: the in-process reader of Triton's
  compile cache printed empty register fields in both runs (the cache
  entry fields moved on this Triton version); the register counts in
  the record above come from ncu's launch attributes instead.
* GPU health: no throttle flags, no hot readings, both runs.
