# plans/ — internal design docs, plans, findings, and their supporting code

This directory collects the project's internal (developer-facing) documentation — architecture
decisions, program plans and status, and the findings/design documents produced by
experiments — together with the scripts that produced the numbers.  The layout rule:

> **documentation at `plans/<area>/`; that area's supporting scripts at
> `plans/experiments/<area>/`.**

(User-facing documentation lives in `docs/` (readthedocs); `experiments/` at the repo top
level holds research areas unrelated to these plans.)

Docs here are records: they capture the reasoning and measurements behind decisions at the
time they were made, and are not rewritten as the code evolves — code comments carry the
current state; these carry the why and the numbers.

## Architecture and decision records (in `sharding/`)

- `sharding/back_projection_overview.md` — data layouts and the structure of forward/back
  projection across geometries.
- `sharding/sinogram_sharding.md` — why sinograms shard by VIEW and recons by SLICE, with
  the parked detector-row-sharding alternative and its halo analysis.

## Program plans and status

- **`current_plans.md` — the EVOLVING forward plan (start here):** the running list of open
  work, updated as programs complete (originally the post-sharding plan).
- `sharding/` — the multi-device sharding program (COMPLETE, shipped 2026-07).
  `sharding/sharding_status.md` is the end-state summary; `sharding_implementation_plan_v3.md`
  is the final plan of record (v1/v2 are its history); the remaining files are per-workstream
  designs (MAR refactor, preprocessing pipeline, correctness gating, performance tracking,
  increment designs).  `sharding/_file_index.md` has one line per file.
  `sharding/parallel_performance/` holds the parallelization option studies (fbp filter
  strategies, forward-vs-back discussion).
- `partition_sequence/partition_sequence_plan.md` — the VCD partition-sequence convergence
  study (ACTIVE as of 2026-07; experiment code in the sibling repo,
  `mbirjax_metrics/experiments/partition_sequence/`).
- `flash_remediation/flash_remediation_plan.md` — the FoV-truncation "flash" remediation
  program (ACTIVE as of 2026-07): sinogram weight edge tapering vs recon-support padding
  (`scale_recon_shape`); synthetic characterization in
  `experiments/flash_remediation/`.
- `torch_port/` — the PyTorch port program (ACTIVE 2026-08-04).
  `port_plan.md` is the assessment and phase plan: motivation and the replacement
  decision rule (within 2x time / ~1.5x memory of jax on the metrics cells), parity
  gates, and the six-phase incremental plan.  `phase0_findings.md` records the
  de-risking spikes (local + H100; spike scripts in `experiments/torch_port/`);
  `phase1_findings.md` records the parallel-beam vertical slice's gate results
  (goldens, convergence parity, backend smoke); `phase2_findings.md` the compile
  integration and gate-cell readouts; `phase3_findings.md` the cone-beam port;
  `panel_review.md` the 30-agent review and fix status.  The port code lives in
  the separate `mbirtorch` repo.
- `viewer/slice_viewer_eval.md` — slice-viewer evaluation and refactor
  recommendations (2026-08-05): stay with matplotlib but
  restructured (model/view split, headless-silent import, easygui removed,
  blitting for the remote-X slider lag); pyqtgraph as an optional later
  frontend; an eight-step refactor sequence with the cluster- and
  mbirtorch-port-load-bearing steps marked.  `viewer/mbirtorch_viewer_build.md`
  is the follow-on build spec (decision 2026-08-05): mbirjax's viewer stays
  as-is; the restructured viewer is built greenfield into mbirtorch, in four
  checkpointed stages, with an mbirjax retrofit after field use.
  `viewer/mbirtorch_viewer_findings.md` is the as-built record (2026-08-05,
  built and field-tested; ThinLinc-verified 2026-08-07): feature parity with
  the mbirjax viewer, the latent-bug non-reproduction checklist, the
  deviations that field testing drove (native-first dialogs, the restored
  right-click menu), and three matplotlib 3.11 macosx problems.  Decision
  2026-08-07: mbirjax keeps its current viewer; the retrofit is recorded but
  not planned.
- `preprocessing/` — the scanner-reader API refactor (readers return a ready model via
  `get_sino_and_model` / `build_model`; landed on `prerelease` 2026-07, PR #219):
  `preprocessing_pipeline_refactor_plan.md` is its plan and as-built design record;
  `mbirjax_applications_migration_plan.md` is the companion migration of the
  `mbirjax_applications` repo to the new API.  Same FILENAME, different document:
  `sharding/preprocessing_pipeline_refactor_plan.md` is the earlier (2026-06) fused /
  view-sharded pipeline plan from the sharding program.

## Findings from experiments

- `projector_kernels/fwd_back_findings.md` — THE record of the 2026-07 projector-kernel
  campaign: forward/back attribution, the sorted channel reduction and its guard constants,
  the TilePolicy, per-geometry rollouts (including translation's measured collision-cliff
  non-adoption), the DRY fan helpers, and the concrete-scatter-centers rounding-bug fix with
  its verification chain.
- `projector_batching/` — the earlier projector-batching characterization and the retired v2
  batching refactor (a worked example of driver-level wins failing to compose end-to-end).
- `bugs_and_artifacts/jax rounding bug/` — the XLA round-in-jit miscompilation:
  `jax_rounding_bug.md` (the bug record; the bug still exists in JAX) and `phase_d_design.md`
  (the concrete-input fix design + as-built notes; the horizontal fans are fixed, the
  vertical fans' per-slice rounds are documented accepted risk).
- `bugs_and_artifacts/center slice noise/` — the center-slice noise investigation and
  preconditioner notes.

## Supporting code (`plans/experiments/`)

- `experiments/projector_kernels/` — the kernel-campaign benches (A/B microbenches, tile and
  crossover sweeps).
- `experiments/projector_batching/` — the batching characterization probes.
- `experiments/bugs_and_artifacts/jax rounding bug/` — the rounding-bug repros
  (`lax_map_scatter_bug/` has the minimized T1–T15j sweep that anchors the fix);
  `center slice noise/` scripts alongside.
- `experiments/sharding/` — the sharding program's scripts, feature probes
  (`features/`), and the scaling-test collateral.

## Related working documents (not in this directory)

- `.claude/claude_prompt.md` — collaboration style and code-change workflow.
- `.claude/lessons.md` — the engineering playbook (operative rules distilled from these
  programs; the narrative history behind each rule lives in the documents above).
