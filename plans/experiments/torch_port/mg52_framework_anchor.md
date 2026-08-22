# mg52 run record

One run, after one 22-second false start.  The finding and its
reading are in `plans/torch_port/active/multigpu_findings.md` §1.43;
this file holds the run detail.

## mg52, the cross-framework anchor (job 15428371)

* 2026-08-22, node h002, one H100 80GB, 1 hour 9 minutes 50 seconds
  by sacct, exit 0.  The first submission (job 15424635) failed in
  22 seconds before staging anything: the batch shell has no
  `module` command and therefore no `conda` function, so the
  `conda run` that read the jax environment's site-packages path
  returned nothing.  Every earlier job in this series used absolute
  paths, which is why the missing `module` had never surfaced.  The
  sbatch now names the environment's python by its absolute path
  (`~/.conda/envs/mbirjax_regression/bin/python`) and needs neither.
* The libraries: mbirtorch at the committed tip 4bb3be5 (torch
  2.13.0+cu130; the tree is the rsynced export, so the identity row
  reads "commit unknown" and the md5 sync record stands in), and
  mbirjax pinned at 7bb20093d635802ba8505c9366ff109ff6b35b76
  (v0.7.1, jax 0.10.1), asserted on the node before any jax arm.
  The jax side ran in a bare overlay venv over the nightly
  regression's persistent environment; the validation printed jax
  with a GPU device and mbirjax resolving from the pinned checkout.
* The premise witness held: both geometries run both projection
  directions as general torch code in this tree.
* Staging built all four cells fresh with mbirtorch under the floors
  protocol and recorded an md5 beside each npz (about 21 GB total):
  translation_256 a9f38d55f8b0602fe329a36a1f1678d4, multiaxis_512
  d148b0890904e138bbc5d7e5b06d3af8, multiaxis_768
  bca9706523734478917d14cedee7c810, multiaxis_1024
  798c72e1cf5bb7803b9f2b02294753c6.  Every arm re-verified its md5
  on load, and every model matched the staged recon shape.
* The protocol per arm, in a fresh subprocess with a 45-minute cap
  (none was hit): seed 13, a 3-iteration reconstruction with the
  stopping threshold disabled, one cold pass, then the warm median
  of three.  Warm spreads read 0.0 to 0.6 percent.
* The measured table, one device, warm medians:

  | cell | geometry | mbirtorch | mbirjax | jax/torch | torch peak | jax peak |
  |---|---|---|---|---|---|---|
  | (256, 1900, 3000) | translation | 12.59 s | 16.26 s | 1.29x | 27.22 GB | 41.46 GB |
  | (512, 448, 384) | multiaxis | 11.41 s | 11.06 s | 0.97x | 11.38 GB | 6.04 GB |
  | (768, 672, 576) | multiaxis | 56.30 s | 60.06 s | 1.07x | 15.12 GB | 17.55 GB |
  | (1024, 1008, 992) | multiaxis | 310.06 s | 431.07 s | 1.39x | 34.75 GB | 58.16 GB |

  A ratio above 1 means mbirjax took longer.  Peaks are each
  framework's own process-lifetime peak device allocation.  Cold
  passes ran 23 to 322 s for torch and 31 to 452 s for jax.
* The mbirtorch one-device warm walls reproduce their recorded
  values (11.41 against the recorded 11.4, 56.30 against 56.3,
  310.06 against 309.9, 12.59 against 12.6), which ties this
  harness to the floors measurements.
* Value fingerprints (float64 sum of absolute values and of
  squares) agree across frameworks at 5.7e-6 to 3.9e-5 relative on
  every cell, far under the 1e-3 note level.
* Instrument health: every planned arm produced a result, no
  out-of-memory, no timeout, no findings outside the table.
* Output rows:
  `rows/mg52_framework_anchor_h002_20260822_050011.jsonl` (md5
  960d65e86af69e04c0eeda037aef739b, verified after copy).  The
  staged npz files remain on scratch under
  `results/mg52_framework_anchor/` for repeat runs.
