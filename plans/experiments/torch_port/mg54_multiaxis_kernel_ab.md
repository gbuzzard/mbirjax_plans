# mg54 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.45; this file
holds the run detail.

## mg54, the kernel route against the torch route (job 15432699)

* 2026-08-22, node h001, one H100 80GB, 34 minutes 19 seconds by
  sacct, exit 0, instrument health "healthy", no findings outside
  the tables.
* The library under test is the CANDIDATE tree at
  /scratch/gautschi/buzzard/torch_p3/mbirtorch_dev, selected by
  PYTHONPATH with no editable install, asserted before any work.
  The tree is the committed tip 4bb3be5 plus the staged multiaxis
  kernel work (the kernel module, its routing, and their tests).
  torch 2.13.0+cu130, triton 3.7.1.  The tree witnesses held,
  including the new one: the geometry's selection consults both
  availability checks and reaches both kernels.
* The staged inputs are mg52's own npz files, reused from
  results/mg52_framework_anchor with their md5s verified:
  d148b0890904e138bbc5d7e5b06d3af8 (512-class),
  bca9706523734478917d14cedee7c810 (768-class),
  798c72e1cf5bb7803b9f2b02294753c6 (1024-class).  Both routes and
  the mg52 anchor therefore measured the same bytes.
* Six arms, each a fresh subprocess with a 45-minute cap (none
  was hit): per cell, the kernel route (default selection) and
  the torch route (MBIRTORCH_DISABLE_TRITON=1).  Every arm's
  bound bodies were asserted and recorded; every kernel arm bound
  the two Triton wrappers and every torch arm the two torch
  bodies.  Protocol per arm: seed 13, a 3-iteration
  reconstruction with stopping disabled as the cold pass, then
  the warm median of three.  Warm spreads 0.0 to 0.3 percent.
* The measured table, one device:

  | cell | kernel | torch | ratio | kernel peak | torch peak |
  |---|---|---|---|---|---|
  | (512, 448, 384) | 2.90 s | 11.41 s | 0.25 | 1.96 GB | 11.38 GB |
  | (768, 672, 576) | 12.71 s | 55.99 s | 0.23 | 6.54 GB | 15.12 GB |
  | (1024, 1008, 992) | 67.64 s | 309.92 s | 0.22 | 24.11 GB | 34.75 GB |

  Reserved-byte peaks: kernel 2.87 / 9.02 / 36.99 GB, torch
  12.83 / 30.58 / 47.07 GB.  Cold passes: kernel 10.54 / 17.41 /
  69.53 s, torch 17.67 / 61.45 / 316.10 s.
* The torch arms reproduce the recorded one-device walls (11.41
  against 11.4, 55.99 against 56.3, 309.92 against 309.9), which
  ties this harness to the floors measurements and to mg52.
* Value fingerprints (float64 sum of absolute values and of
  squares of the final reconstruction) agree between routes at
  1.33e-07 / 1.54e-07 (512), 1.64e-07 / 1.92e-08 (768), and
  7.43e-09 / 8.73e-09 (1024) relative.
* The view batch, chosen by the cost model of the body actually
  bound: 128 views in both directions at every cell on the kernel
  route, against the torch bodies' 9, 2, and 1.
* The single-call leg at the 1024-class, kernel route, after one
  untimed warm-up per direction, with the body wrapper forwarding
  the kernel's _view_batch_cost attribute so the driver's batch
  choice was untouched:

  | direction | batch | calls | enqueue s | in body s | wall s |
  |---|---|---|---|---|---|
  | forward | 128 | 8 | 7.821 | 7.820 | 8.972 |
  | back | 128 | 8 | 0.0056 | 0.0053 | 4.804 |

  The torch bodies at the same cell measured 34.96 s of enqueue
  against a 38.05 s wall forward and 19.35 against 23.65 back
  (mg51/mg53), at 1024 one-view body calls.
* Output rows:
  `rows/mg54_multiaxis_kernel_ab_h001_20260822_094414.jsonl`
  (md5 92b656b6396e03e1857d1ff8c66ec810, verified after copy).
