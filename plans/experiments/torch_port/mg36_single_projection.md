# mg36 run record

Single-projection anchors and the 1/128 launch row for the kernels
document (`plans/torch_port/triton_kernels.md` §3 and Table 1).
Ordered by Greg 2026-08-18 as part of the document's revision; no
record held these numbers.

* Job 15352605, node h001, one H100, about 7 minutes end to end.
  Exit 0.  The tree under test is c761b24 (the committed sorted
  forward), already installed on scratch by the mg34/mg35 syncs.
* The cell is mg20's 1024-class construction for both geometries
  (mg33's parallel builder and mg31's cone builder, verbatim); full
  mask 771,240 pixels, realized and gated.  Medians of three timed
  repeats after one discarded warm-up, device-synchronized.
* Whole-call projections through the model API (`sparse_forward_
  project` / `sparse_back_project`), full mask, all 1024 views:
  parallel forward 7187.1 ms per-tap against 2526.9 ms sorted
  (2.84x at the call level); parallel back 791.7 ms; cone forward
  7992.0 ms; cone back 4544.5 ms.
* The 1/128-subset launch, mg33's exact quantity (one 128-view
  wrapper call, strided subset, 6,025 pixels): 6.65 ms per-tap
  against 3.24 ms sorted, 2.05x -- the sorted win's thinning trend
  (3.97, 3.85, 3.55, 2.77, now 2.05) continued at the size the
  default schedule's long tail runs.
* Value gates: the two routes agree at 5.1e-6 relative on the
  whole-call sinogram and 2.9e-7 at the subset launch, both inside
  the 1e-5 gate.  The per-tap arms flip `MBIRTORCH_SORTED_FORWARD=0`
  around the call (the switch is read per call).
* Rows: `rows/mg36_single_projection_h001_20260818_231605.jsonl`
  (md5 cdba721e4acfb947eeeb4d6d12c19ad3, verified after copy).
