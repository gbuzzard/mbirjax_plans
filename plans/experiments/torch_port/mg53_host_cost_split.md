# mg53 run record

One run, first submission.  The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.44; this file
holds the run detail.

## mg53, the host-cost attribution (job 15429313)

* 2026-08-22, node h002, one H100 80GB, 12 minutes 40 seconds by
  sacct, exit 0, instrument health "healthy" with no checks.
* The tree: the committed tip 4bb3be5 (torch 2.13.0+cu130) through
  the editable install on scratch, witnesses asserted in the sbatch
  and recorded on the header row.  The premise witness held: both
  multiaxis directions bound compiled torch bodies, and the
  compile-error record shows no eager fallback.
* The protocol is mg51's: the same two cells, the same model
  construction, seed 13 before every call, the public
  `sparse_forward_project` / `sparse_back_project` route, one
  untimed warm-up call per direction paying the compile.  The view
  batch the driver chose: 9 views at the 512-class (57 body calls
  per projection) and 1 view at the 1024-class (1024 body calls).
* The timing leg reproduces mg51 (warm medians, spread 0.000):

  | cell | direction | warm s | enqueue s | enqueue/wall |
  |---|---|---|---|---|
  | (512, 448, 384) | forward | 1.161 | 0.0097 | 0.008 |
  | (512, 448, 384) | back | 1.050 | 0.0086 | 0.008 |
  | (1024, 1008, 992) | forward | 38.053 | 34.956 | 0.919 |
  | (1024, 1008, 992) | back | 23.647 | 19.347 | 0.818 |

* The body split, one wrapped call per row; driver s is the
  enqueue minus the summed in-body time:

  | cell | direction | body calls | enqueue s | in body s | driver s |
  |---|---|---|---|---|---|
  | 512-class | forward | 57 | 0.0096 | 0.0084 | 0.0012 |
  | 512-class | back | 57 | 0.0089 | 0.0077 | 0.0012 |
  | 1024-class | forward | 1024 | 34.956 | 29.434 | 5.522 |
  | 1024-class | back | 1024 | 19.345 | 17.826 | 1.520 |

  The split leg's enqueue agreed with the timing leg's median at
  every row, so the wrapper did not change the call it measured.
* The host-operator table names the mechanism.  At the 1024-class
  an event named "Command Buffer Full" carries 35,274 ms of the
  forward call's self CPU (91 percent; 8474 events) and 19,341 ms
  of the back's (81 percent; 3765 events).  The launch rows
  themselves are milliseconds (cudaLaunchKernel 5.3 ms,
  cuLaunchKernel 22.7 ms, cuLaunchKernelEx 19.2 ms on the
  forward), and the compiled dispatch is 0.16 ms per batch
  (compiled-graph call 122.7 ms, compiled-region entry 32.0 ms,
  dynamo cache lookup 12.4 ms, over 1024 batches).  At the
  512-class no queue event appears and the largest host row is
  the deliberate final synchronize.
* The allocator counters read zero across the measured calls at
  both cells and directions: zero device allocations, zero frees,
  zero retries, zero segment change.  Call peaks at the
  1024-class: 23.34 GB allocated on the forward, 16.62 GB on the
  back, 27.83 GB reserved.
* The synchronization detector captured zero warnings at every
  cell and direction.
* The ablation child (fresh process, expandable segments on)
  moved no wall: warm ratios 1.000 to 1.007, enqueue ratios 1.001
  and 1.000 at the 1024-class.
* Output rows:
  `rows/mg53_host_cost_split_h002_20260822_070418.jsonl` (md5
  f6dc40f491f877edf3d8ee100f266827, verified after copy) with the
  child's timing rows folded in as leg
  `timing_expandable_segments`; the child's own file
  `rows/mg53_host_cost_split_h002_20260822_070418_child.jsonl`
  (md5 81193696ee97038297a571c3f725b223).
