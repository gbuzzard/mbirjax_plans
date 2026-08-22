# mg51 run record

One run.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.42; this file holds
the run detail.

## mg51, the counter run on the compiled multiaxis bodies (job 15424602)

* 2026-08-21, node h008, one H100 80GB, 8 minutes 29 seconds by
  sacct, exit 0.  The tree is the committed tip 4bb3be5 (version
  0.0.2), rsynced to `mbirtorch_src` and spot-verified by md5 on
  three files before submission.
* Instrument health: every check held.  Both directions bound the
  torch bodies (`_multiaxis_forward_view_batch`,
  `_multiaxis_back_view_batch`), compilation was on, the realized
  device was the one asked for, no leg hit an out-of-memory, no
  kernel went unprofiled, and the GPU showed no throttle or hot
  flags.
* The cells are the floors cells: multiaxis sinogram (512, 448, 384)
  with recon (384, 384, 510), and (1024, 1008, 992) with recon
  (992, 992, 1148).  The protocol is the floors refresh's: the same
  model construction, the shepp-logan low-dynamic-range phantom,
  seed 13 before every call, one device placed explicitly.  The
  route is the production funnel pair,
  `TomographyModel.sparse_forward_project` and
  `sparse_back_project`, one cold call then three warm.
* The timing leg, plain clocks.  `enqueue s` is the host time to
  return from the call with no synchronize:

  | cell | direction | cold s | warm s | enqueue s | enq/wall |
  |---|---|---|---|---|---|
  | 512-class | forward | 5.097 | 1.160 | 0.010 | 0.009 |
  | 512-class | back | 1.443 | 1.049 | 0.008 | 0.008 |
  | 1024-class | forward | 46.066 | 38.046 | 34.953 | 0.919 |
  | 1024-class | back | 27.579 | 23.642 | 19.354 | 0.819 |

* The profiler leg, one trace carrying both sides.  Launch counts
  are per one projection call:

  | cell | direction | distinct kernels | launches | device s | host s (no sync) | host/device |
  |---|---|---|---|---|---|---|
  | 512-class | forward | 10 (9 generated) | 684 | 1.179 | 0.020 | 0.017 |
  | 512-class | back | 6 (5 generated) | 284 | 1.066 | 0.013 | 0.012 |
  | 1024-class | forward | 11 (10 generated) | 12,288 | 38.783 | 35.594 | 0.918 |
  | 1024-class | back | 4 (3 generated) | 5,118 | 23.913 | 19.524 | 0.816 |

  The launch counts are the batch structure exactly: 57 view
  batches at the 512-class and 1024 at the 1024-class, times 12
  kernels per batch forward and 5 back (one of each is an eager
  elementwise, the rest generated).  The runtime cross-check
  recorded one launch-API call per view batch, costing 264 us total
  at the 512-class forward and 6.4 ms total at the 1024-class
  forward.  So the host time is not the launch API.  It is
  per-batch host work: about 0.35 ms per view batch at the
  512-class and about 35 ms per view batch at the 1024-class,
  against per-batch device times of 20 ms and 37 ms.
* Concentration: no direction has a single dominant kernel.  The
  top three kernels carry 59 percent of forward device time at the
  512-class and 58 percent at the 1024-class; the back
  concentrates harder, 98 percent at the 512-class and, at the
  1024-class, two generated kernels at 46 and 39 percent plus an
  eager elementwise at 15.
* The counter leg, Nsight Compute at the 512-class, top three
  generated kernels per direction.  All six collected the full
  21-metric set on the first attempt, with exact name filters and
  launch skips of 56 or 57 aimed past the runner's earlier calls:

  | direction | rank | dur ms | occ % | SM % | mem % | L2 % | L1 % | DRAM GB |
  |---|---|---|---|---|---|---|---|---|
  | forward | 0 | 5.57 | 97.1 | 83.2 | 52.3 | 73.4 | 37.4 | 9.10 |
  | forward | 1 | 4.22 | 96.1 | 84.5 | 54.1 | 46.0 | 20.0 | 7.13 |
  | forward | 2 | 4.21 | 96.1 | 83.8 | 54.3 | 46.1 | 20.0 | 7.13 |
  | back | 0 | 8.90 | 60.5 | 77.8 | 94.0 | 11.2 | 71.4 | 9.71 |
  | back | 1 | 8.40 | 72.2 | 50.0 | 88.3 | 99.7 | 69.8 | 1.75 |
  | back | 2 | 5.18 | 68.6 | 85.2 | 48.6 | 99.9 | 96.9 | 7.86 |

  For scale, the whole problem's arrays at this cell total about
  0.65 GB (sinogram 0.35, volume 0.30), so a single launch moving
  7 to 9.7 GB is moving intermediates.
* Peak device allocation during one projection call: 11.4 GB at
  the 512-class and 25.1 GB at the 1024-class.
* The ncu permission probe passed on the first try; the binary came
  from the cuda module's bin as on the earlier counter runs.
* Output rows:
  `rows/mg51_multiaxis_counters_h008_20260821_232543.jsonl` (md5
  6c3b3685d4d66f86d76d024aa05f24fe, verified after copy), with one
  ncu log per kernel beside it on scratch
  (`results/mg51_ncu_*_full_skip5?.log`).
