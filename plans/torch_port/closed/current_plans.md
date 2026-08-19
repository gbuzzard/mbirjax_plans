# Current forward plan — goals for the next release

(The EVOLVING running list of open work, at `plans/current_plans.md`.  Rewritten
2026-08-07 around the agreed mbirtorch priorities; compressed 2026-08-09 —
completed items are reduced to outcome bullets, and the detail lives in the
findings docs each item cites.  This file should be cleaned periodically to
avoid build-up of historical detail.)

## Contents

Items 1–8 are the agreed top priorities (2026-08-07), in rough order.

1. [Kernel-aware view batching — COMPLETE](#1-kernel-aware-view-batching--complete-2026-08-07)
2. [Device policy — COMPLETE](#2-device-policy--complete-2026-08-08)
3. [Multi-GPU performance investigation — IN PROGRESS](#3-multi-gpu-performance-investigation--in-progress)
4. [mbirtorch in the nightly — COMPLETE](#4-mbirtorch-in-the-nightly--complete-2026-08-08)
5. [Remaining Sphinx documentation pages — COMPLETE](#5-remaining-sphinx-documentation-pages--largely-complete)
6. [Additional geometries: translation and multiaxis — COMPLETE](#6-additional-geometries-translation-and-multiaxis)
7. [Release workflow — COMPLETE](#7-release-workflow)
8. [Remaining utility API surface — COMPLETE](#8-remaining-utility-api-surface--largely-complete)
9. [MAR: cache H](#9-mar-cache-h)
10. [LEAP/SVMBIR interfaces (back-burnered)](#10-leapsvmbir-interfaces-back-burnered)
11. [Miscellaneous / cleanup](#11-miscellaneous--cleanup)
12. [Possible future direction: multi-resolution reconstruction](#12-possible-future-direction-multi-resolution-reconstruction-post-next-main)
13. [Sorted-stream parallel forward — SCHEDULED](#13-sorted-stream-parallel-forward--scheduled-after-item-3)
14. [Forward-kernel repair under sharding — COMPLETE](#14-forward-kernel-repair-under-sharding--complete-2026-08-08)
15. [Sharded phantom generation — COMPLETE](#15-sharded-phantom-generation--on-deck-after-item-3)

---

## 1. Kernel-aware view batching — COMPLETE (2026-08-07)

Record: `plans/torch_port/closed/kernel_batching_findings.md` (Fable-reviewed).

- The driver batches each projector body by the body's own cost model.
- Composed gates improved at every cell: parallel 1.13x/1.56x of jax, cone
  0.88x/1.00x, memory 0.57–0.63x.  These are item 3's n=1 baselines.
- The sorted-stream forward was declined on these numbers; the revisit
  triggers are in the findings (→ item 13).

## 2. Device policy — COMPLETE (2026-08-08)

Records: `plans/torch_port/closed/device_policy_design.md` (rulings) and
`device_policy_findings.md` (measurements); three Fable-reviewed checkpoints.

- The memory-ledger preflight is live: calibrated 1.001–1.10 against measured
  peaks, and the two residency fixes it predicted cut composed peaks 11–15%.
- The all-device default shipped: automatic capacity-aware widening on
  multi-GPU CUDA, with `configure_devices` as the single explicit door and
  the reproducibility pin.  The constructor `device` kwarg was removed.
- The flip gate exposed the forward-kernel sharding defect; item 14 repaired
  it and retired the interim.

## 3. Multi-GPU performance investigation — MEASUREMENT COMPLETE; implementation in progress

**State (2026-08-10):** every instrument has run and every result is read;
`plans/torch_port/active/multigpu_findings.md` is the results record, and
`multigpu_plan.md` §0 is the live dashboard of what remains.  Landed in
mbirtorch: the widening speed guard with measured per-geometry floors and
a refresh script (`mbirtorch/_widening_floors.py`); corrected memory
charges (the back-loop over-charge removed, two forward under-charges
found and fixed, no fitted constants); two stale-residency releases worth
8 to 16 percent of peak memory at the large sizes; and automatic
staleness detection, so the library and the test suite both keep working
when the floors age.

**The four goals:** 1. DONE — the full gate readout is complete and
valid; the `usr_multi_gpu.rst` timing-table refresh is a close-out item.
2. DONE in substance — the guard shipped; the cadence decision is made
(full n>1 cadence through the tuning window); the forward's poor scaling
is attributed to data movement that does not fall with the device count,
and choosing its remedy is the next design step.  3. DONE — the value
comparison is read and goal 3 is ruled met.  4. DONE — by device-span
measurement the forward is about 70 percent of GPU time at the large
parallel size, so item 13's entry gate is SATISFIED.

**Remaining (`multigpu_plan.md` steps 5 through 9):** the forward-remedy choice
(driver change versus item 13's sorted stream); the seam decision, whose
memory premise is now met at the 1K sizes and which is Greg's call; the
2K capacity table on the corrected charges; close-out; moving the floors
refresh into the nightly; and a simplification of the floors for
robustness.  One ruling spans the public surface: every entry point that
starts sharding takes the same device policy, with a systematic entry
list and recorded design questions ahead of implementation.

## 4. mbirtorch in the nightly — COMPLETE (2026-08-08)

Record: `nightly_plan.md` §10–§12 (plan, trial gates, findings).

- Both series are live and on the dashboard: `gpu-torch` at 03:00 on
  gautschi (n∈{1,2,4} rows at the two large cells, n=1 elsewhere) beside
  the unchanged 02:00 jax block, and `cpu-torch` at 10:00 on the Mac, with
  the shared cell activating the cross-platform reference.
- Every knob was measured, not assumed: the memory window is 1 at every
  device count (0.000% spread), and a night costs ~0.26 GPU-hours at n=1,
  ~2 with the multi-device rows.
- Installation bycatch: the jax macOS agent had failed 51 straight nights
  (launchd cannot read `~/Documents` under TCC) while every watching
  surface reported health; both agents now run from entry clones outside
  the protected tree.
- Goldens are opt-in per the §10.4 Resolution (marker + `addopts`), so the
  nightly needed no suite wiring and its skip set is hardware gates alone.
- Watch item: the three-night soak was waived to give item 3 its baseline,
  so the first unattended nights deserve a `status_torch_nightly.sh` look.

## 5. Remaining Sphinx documentation pages — COMPLETE

- DONE (2026-08-08): the docs.md work list — the section-8 restores with
  the `__all__` rule, the multi-GPU rewrite, the demos and data-generation
  halves, and the kernel and geometry developer pages — plus the panel
  review's doc fixes, applied in the post-merge fix pass.
- DONE (2026-08-10, Charlie's session): the four Sphinx warnings Greg
  listed here.  install.rst is written (mbirtorch b27be6a), which
  resolves both InstallationDocs label warnings; the two references to
  the held dashboard page are repaired with PENDING markers.  The two
  geometry pages landed with the item-6 ports.  dev_maintenance.rst is
  written (mbirtorch 921e5a5) around the release routine, reviewed and
  tightened by Charlie.  The build now has ZERO warnings, and the CI
  docs job runs with -W, so a new broken reference fails its pull
  request.
- DONE: the dashboard page .

## 6. Additional geometries: translation and multiaxis — COMPLETE (2026-08-10).

**State (2026-08-10, evening):** both ports landed and were reviewed
(verdicts LAND NOW; reviews archived in `plans/torch_port/reviews/`), and all
three queued follow-ups are done: `copy_ct_model`/`get_ct_model` are widened,
the twelve parity goldens pass, and the memory calibration ran and its
correction is implemented.  The calibration (mg8) found the torch-body
per-view proxy short on every projection-dominated arm, and the ledger now
charges a measured closed form; findings §2.1 in
`plans/torch_port/active/multigpu_findings.md` carries the read, and the
corrected ledger with its tests is staged in mbirtorch.  One consequence for
users: the preflight now refuses torch-body reconstructions that previously
started doomed, and multiaxis 1024 sits at an H100's edge on one device.
Original charter: section B of
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md`.

- Port TranslationModel and the multiaxis parallel geometry; the jax
  regression harness already carries both, so gate rows have baselines from
  day one.
- Translation includes the companions the application scripts call:
  `gen_translation_vectors`, `display_translation_vectors`,
  `calc_tct_recon_params`, `gen_translation_phantom`, and the
  `delta_recon_row` parameter.
- Two recorded lessons become first-class here: pixel-axis chunking
  (detectors heading to 6K x 10K views) and the collision-cliff class
  (probe production shapes before enabling any tuned path).
- Kernel treatment decided by measurement; the Phase 5 protocol applies if
  the torch bodies do not suffice.

## 7. Release workflow — COMPLETE (2026-08-10).

**State:**   The full proposal as implemented is at
`plans/torch_port/active/release_workflow.md`.

- GitHub-Release-driven publishing: main/prerelease branch model, CI on
  pull requests, PyPI via Trusted Publishing, Read the Docs on
  `stable`/`latest`.

## 8. Remaining utility API surface — COMPLETE

Work list and protocols: `plans/torch_port/active/preprocessing.md` and the census
in `plans/torch_port/active/docs.md` §5.

- DONE: the preprocess package, MAR, the coupled functions, the download
  utilities, the HDF5 save/load family, and the hsnt and vcls modules
  (Charlie's session; reviewed and merged).
- DONE: `split_sino_recon` with `stitch_arrays` and `copy_ct_model` — the
  full mbirjax logic, with the half-model device pin unconditional.
- DONE (2026-08-08, panel-reviewed): `get_ct_model` and the demo-data
  generators (phantom builds byte-exact against mbirjax);
  `device_summary` confirmed replaced by `get_memory_stats`; the
  beyond-charter run-logging port accepted, with the review fixes landed.
- DONE: the Lilly NSI end-to-end run (needs Charlie and real data).

## 9. MAR: cache H

Reframed 2026-07-10 (from `mar_refactor_plan.md` Phase 3, which is SPEED-only —
the fit's memory was solved in Phase 2):

- **The direction is caching**: compute each `H` column once instead of the
  O(num_cols²) recompute — a cheap, self-contained win with no statistical
  questions.
- **Subsampling is deprioritized**: uniform view/stride subsampling is wrong
  for this fit (identifiable only from pixels spanning diverse metal path
  length, which are sparse in a mostly-plastic object).  If revisited: A/B
  the estimation in isolation and gate on the corrected recon within a
  documented tolerance — not byte-identical by design.

## 10. LEAP/SVMBIR interfaces (back-burnered)

**State:** Back-burnered per the port decision (`port_plan.md` §1): a jax-side
wrapper would be throwaway work, and LEAP presents as a PyTorch front end, so
the wrapper is thinner on mbirtorch.

- Goal: lower the barrier for LEAP (https://github.com/LLNL/leap) and SVMBIR
  (https://github.com/cabouman/svmbir) users to transition.
- Steps when picked up: run their examples, scope a translation, then
  design and validate interfaces.

## 11. Miscellaneous / cleanup

Delivered:

- **Memory preflight** — DONE in mbirtorch via item 2 (the closed-form
  per-device ledger; design in `device_policy_design.md`).  The jax-side
  version stays parked and is likely moot as mbirjax winds down.  (Residual
  UX notes from the motivating incident: `python -u` for chronological
  sweep logs; `TF_CPP_MIN_LOG_LEVEL=2` quiets a BFC warning wall.)
- **Device-count policy** — the torch widening rule is live (item 2), and
  its speed-guard knob is an item-3 deliverable.  The jax
  `_auto_device_count` slice-vs-view basis fix stays open only if jax
  development continues.
- **`configure_devices`/`use_gpu` unification** — resolved in mbirtorch:
  `configure_devices` is the single door, no constructor kwarg.
- **`propagate = False`** on the run logger — landed in the 2026-08-08 fix
  pass.

Open:

- **Multi-device completion checklist (Charlie-side):** preprocess
  sharding, export acceptance, and the geometry ports proceed in parallel
  with item 3.  Sequencing, riders, and per-item status live in
  `plans/torch_port/active/preprocess_sharding_translation_multiaxis.md`.
- **Denoiser scope — DECIDED 2026-08-09, full sharding parity.**  The work
  is chartered as item A4 of the checklist above.  A4 includes both
  companion gaps: the `.clone()`-on-`Shards` failure and the log arguments
  the other entry points gained.
- **Run-logging inherited defects** (faithful ports of mbirjax behavior;
  carry as upstream notes): the per-class logger lets two live models of
  one class stomp each other's logs; the `FileHandler` is never closed, so
  after `split_sino_recon` merges and deletes the half logs, later lines
  land in a deleted inode; `verbose=0` still creates an empty log file.
  Candidate repairs: per-instance logger keying and a `close()` in
  `setup_logger`.
- **Test-quality tail:** the demo-data gates sit 300–640x above their
  measured floors; `test_logging` wants a two-instance case and
  module-scoped cone fixtures (72 s serial today).
- **Suite efficiency:** simplify tests and reduce time on tests.
- **Forward pixel-batch default** (minor API open).
- **Residual rounding-bug risk (monitor only):** the six vertical-fan
  per-slice round sites keep the in-jit precondition
  (`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md` §7).
- **Archive plans:** move older plan docs and scripts to archived storage.

## 12. Possible future direction: multi-resolution reconstruction (post-next-main)

Coarse-to-fine MBIR: reconstruct at binned resolution(s), upsample as the init for the
next-finer level.  Added 2026-07-10 (Greg); investigation-first, not for the next main.

**Rationale.**  VCD is coordinate descent, so low-frequency corrections propagate slowly
at fine resolution; a coarse level handles them at ~1/8 the voxels (and ~1/8 the sino if
rows/channels bin).  The partition-sequence study's finding that the coarse-GRANULARITY
ramp added ~nothing supports this framing: granularity coarsening changes the update
grouping but still pays full-resolution cost per iteration — GRID coarsening is the
principled low-frequency accelerator that ramp was groping at.

**Where it pays (cost model):** large problems and cap-bound hard objects.  At small
sizes VCD is ~95% host-dispatch-bound (§3), so coarse levels cost fixed per-iteration
host overhead + a compile per shape, not 1/16 flops — don't expect interactive-size wins.

**Null hypothesis to kill first:** coarse-MBIR init must beat FDK/FBP init (one cheap
full-resolution call) on wall-clock-to-matched-quality.  Expected to win only under
heavy noise / sparse views / truncation-corrupted FBP, or where the 0.2%-stop drags.

**The matching problems (the real work) and what softens them:**
- *Volumes:* offsets are in ALU, hence scale-invariant across levels (verified on Lilly:
  −1.98 mm = −3.9 rows at 4× = −1.95 rows at 8×); the upsample must map voxel centers
  PHYSICALLY (the `recon_ijk_to_xyz` chain) since `auto_set_recon_geometry`'s ceils
  break exact shape nesting.  Init-only use makes residual sub-voxel phase error cheap
  (a few iterations, not an artifact) — but do it right; see the 2c misalignment lesson.
- *Parameters:* sharpness/snr_db are scale-free; `auto_set_sigma_y` already carries a
  pixel-pitch^0.5 consistency factor — per-level auto-regularization may be most of the
  answer.  Open question: qGGMRF edge-threshold scale consistency (test: coarse solution
  ≈ downsampled fine solution?).
- *Data per level:* bin the LOG sinogram linearly — provably consistent (it is exactly
  the projection of an axially/laterally smoothed object; the flash-remediation round-3
  result).  No per-level re-preprocessing.

**Pilot (before any library code):** the Lilly 8× workhorse + one large synthetic; A/B
wall-clock-to-matched-quality across {zero init, FDK init, 2-level, 3-level}, compiles
counted honestly, metrics flash-cropped (§2 caveat).  The flash-remediation Lilly
infrastructure (ds4/ds8 pipelines, converged references, seam/region metrics) is
directly reusable.  If the pilot wins: implementation is a `split_sino_recon`-shaped
driver (~100 lines — `copy_ct_model` per level, physical-coordinate upsample, per-level
auto-regularization, loose stopping on coarse levels).

## 13. Sorted-stream parallel forward — SCHEDULED (after item 3)

**State:** Chartered 2026-08-07 (Greg).  The entry gate is item 3's
forward-attribution arm; the decision record that declined this work, with
its revisit triggers, is in `plans/torch_port/closed/kernel_batching_findings.md`.

- The target: the parallel forward's remaining measured gap (14.4 s over
  jax at the 1024 cell, atomic-bound and flat across batch sizes), where
  jax runs its sorted-stream pallas design.  A torch version would build
  streams from the existing hfan contract and cache them through the
  `plan` slot every body already accepts.  The value likely grows by
  waiting: translation (item 6) shares this forward structure.
- Step 1 is a feasibility probe on item 3's attributed numbers, with the
  view-loop forward variant as a cheap comparison arm; STOP if the
  projected composed win falls below a few seconds.
- If the probe holds: the full kernel protocol (emulator, battery, sweep,
  composed gates) with the default flip at a Fable checkpoint.

## 14. Forward-kernel repair under sharding — COMPLETE (2026-08-08)

Record: `plans/torch_port/closed/kernel_sharding_findings.md`.

- The defect was the LAUNCH, not the kernels: a Triton launch targets the
  launching thread's current CUDA device, and the banded drivers launch
  from worker threads sitting on device 0.  A seven-arm single-variable
  matrix isolated it; the back kernels' four-digit survival was an
  accident of reduce topology.
- The repair: all four wrapper launches run under
  `with torch.cuda.device(...)`, the device leads the compile-lock key,
  and the interim retired — kernel selection is layout-independent again.
- Gates: kernel-times-sharding 12/12, flip gate 18/18 (n=4 auto-vs-explicit
  4.58e-01 → 3.4e-07), composed n=1 baselines reproduced, H100 suite green.

## 15. Sharded phantom generation — COMPLETE

**State:** Chartered 2026-08-08 (Greg); next up once the multi-GPU campaign
settles.

- Port mbirjax's distributed `generate_3d_shepp_logan_low_dynamic_range`
  builder (`devices=`, the row-blocked single-device path under
  `max_block_gb`, `target_max_attenuation`), keeping the host gather as
  the default contract.
- Add `output_sharded=True`: return the phantom as engine `Shards` so a
  large phantom feeds `forward_project`/`recon` with no host round-trip.
  The build is per-voxel independent (slice bands, no communication;
  padded slices zero and inert).  One design constraint: `_shard_recon`
  accepts a matching `Shards` but checks placement by object identity, so
  the sharded form must be built over the consuming model's own placement
  — the API takes the model.
- Gates: the gathered sharded build equals the host build exactly at
  n=1/2/4 (byte equality is correct here, unlike recon), and the host
  build keeps its parity with the mbirjax reference.

---

**Recently completed (records live elsewhere):** the **mbirtorch device
policy** (item 2 above; preflight, all-device default, constructor
simplification); the **sharpness/snr_db streak
study, CLOSED 2026-08-07** — the two-start experiment showed the streaks are the
converged minimizer's own content under lateral truncation (a ground-truth start
converges to the same streaked solution, which fits the truncated data 2.3×
better than the truth), so the per-iteration schedule premise is refuted and the
remedy question moves to truncation handling in the model
(→ `plans/sharpness_schedule/converged_streaks.html`, with the Phase A/B record
in `findings.html` and `phase_b_results.html`); the **mbirtorch Phase 5 Triton
kernel campaign** — all four projector kernels default-on, the replacement rule
passing at every gate cell of both geometries
(→ `plans/torch_port/phases/phase5_findings.md`); the **mbirtorch slice viewer port**
— the package-independent greenfield viewer plus the tensor-aware wrapper,
field-tested through four review rounds; the **Pallas projector-kernel campaign**
— the full custom-kernel path for both projectors and geometries, shipped and
soak-validated (design → `docs/source/dev_projector_kernels.rst`; measured record
→ `plans/projector_kernels/gpu_headroom_findings.md`); the **default partition-sequence
change to `[2, 4, 6, 7]`** (→ `plans/partition_sequence/`); the **flash-remediation
padding remedies** (→ `plans/flash_remediation/`); the earlier projector-kernel /
profiling campaign (→ `plans/projector_kernels/fwd_back_findings.md`); the multiaxis
vertical-fan path-length factor (shipped); the sparse-projector batching investigation
(closed with code unchanged → `plans/projector_batching/batching_refactor_design.md`).
