# Current forward plan — goals for the next release

(The EVOLVING running list of open work, at `plans/current_plans.md`.  Rewritten
2026-08-07 around the agreed mbirtorch priorities; renumbered the same day — the
kernel follow-ups formerly §5 are now §1, and the device policy formerly §6 is
now §2.  Completed campaigns live in their own findings docs, cited where their
results guide the work below.  This file should be cleaned periodically to avoid
build-up of historical detail.)

## Contents

Items 1–8 are the agreed top priorities (2026-08-07), in rough order.

1. [Finish the kernel-aware view batching](#1-finish-the-kernel-aware-view-batching)
2. [Device policy: all-device default behind a memory preflight](#2-device-policy-all-device-default-behind-a-memory-preflight)
3. [Multi-GPU performance investigation and tuning](#3-multi-gpu-performance-investigation-and-tuning)
4. [mbirtorch in the nightly, including multi-GPU](#4-mbirtorch-in-the-nightly-including-multi-gpu)
5. [Remaining Sphinx documentation pages](#5-remaining-sphinx-documentation-pages)
6. [Additional geometries: translation and multiaxis](#6-additional-geometries-translation-and-multiaxis)
7. [Release workflow](#7-release-workflow)
8. [Remaining utility API surface](#8-remaining-utility-api-surface)
9. [MAR: cache H](#9-mar-cache-h)
10. [LEAP/SVMBIR interfaces (back-burnered)](#10-leapsvmbir-interfaces-back-burnered)
11. [Miscellaneous / cleanup](#11-miscellaneous--cleanup)
12. [Possible future direction: multi-resolution reconstruction](#12-possible-future-direction-multi-resolution-reconstruction-post-next-main)
13. [Sorted-stream parallel forward (chartered, after items 2–3)](#13-sorted-stream-parallel-forward-chartered-after-items-2-3)

---

## 1. Kernel-aware view batching — COMPLETE (2026-08-07)

**State:** COMPLETE; both goals resolved.  The record is
`plans/torch_port/kernel_batching_findings.md` (Fable-reviewed at both
checkpoints).

**Outcome:** The driver now batches each body by the body's own cost model,
and the composed gates improved at every cell: parallel 1.13x/1.56x of jax
(was 1.21x/1.90x), cone 0.88x/1.00x (was 1.00x/1.18x) — cone now at or
below jax outright — with memory 0.57–0.63x and the pure-torch control arms
reproducing the Phase 5 numbers within 0.4 percent.  The sorted-stream
forward was NOT taken (Fable, on the 1.56x number): no gate demands it, and
the revisit triggers are recorded in the findings.  The gate baselines for
item 3 are this table's.

## 2. Device policy: all-device default behind a memory preflight

**State:** Decided 2026-08-07 (Greg), not started.  Supersedes the
opt-in-persists decision; the revision is recorded in
`plans/torch_port/docs.md` §4.

**Overview:** mbirtorch inherits the mbirjax policy: a reconstruction spreads
across the available GPUs by default.  A user with four GPUs who silently gets
one, or who gets a late out-of-memory failure, is worse off than a user who
pays a device-count-dependent float difference (measured to decay from 6.1e-3
at 3 iterations to 8.8e-4 at 10) or a modest time penalty on small problems.
The memory preflight moves up in priority because it is what makes the
widening safe, and it doubles as the widening criterion (spread only onto
devices that can hold their share).  `configure_devices(num_devices=1)`
remains the reproducibility pin.

**Goals:**
1. The mbirtorch preflight ledger, checked at the top of `recon()` (the parked
   mbirjax design in §11 transfers; the torch version is simpler because the
   engine is eager and the batching work already counted the residencies).
2. The default flip: capacity-aware widening at layout-build time, validated
   by the existing empty-shard rules with graceful fallback.
3. The docs step: `usr_multi_gpu.rst` written against the new behavior.

## 3. Multi-GPU performance investigation and tuning

**State:** Not started; follows items 1–2.

**Overview:** The n=2/4 gates predate the Triton kernels, and the kernels plus
the batching change shift the multi-device balance (the measured n>1 back
limiter was the band transpose).  With the all-device default, multi-GPU
becomes the out-of-box experience, so its performance and its knobs need the
same measured treatment the single-device path got.

**Goals:**
1. A full n=1/2/4 gate readout with kernels on, at the new default, both
   geometries — the composed-gate discipline of `phase5_findings.md` (arm
   checks, shared-sinogram value protocol, memory re-measured).
2. Attribute any gaps and tune what the data indicates: per-device view
   chunks, the seam/streaming hooks, and the widening rule's parameters.
3. The deferred value comparison rides along: jax vs torch across device
   counts, separating the shared partition-order term from torch's
   compile-latitude term.
4. A forward-attribution arm: the parallel forward's measured share of the
   composed time at n=1/2/4.  This number is item 13's entry gate.

## 4. mbirtorch in the nightly, including multi-GPU

**State:** Not started.  The torch harness writer exists, and the first real
gpu-torch H100 run landed at `results/gpu-torch/` in mbirjax_metrics.

**Overview:** Wire the torch writer into the nightly so mbirtorch gets the
same regression protection as mbirjax: correctness gating against goldens,
time and memory gates, and the dashboard rows already built for it.  Multi-GPU
rows start EARLY per the approved plan (amended 2026-08-07, superseding the
original wait-for-item-3 sequencing): history-based gates self-seed, and the
risky interval the rows should protect is exactly the items-2-3 multi-device
work.  Their preconditions are the device-policy flip landing, the measured
rolling-min window, and a three-night n=1 soak — `nightly_plan.md` §3(c).

**Goals:**
1. The scheduling decision and the writer wiring (the open Phase 2 tail item).
2. n>1 nightly rows with the established memory-gate discipline (the
   rolling-min lesson and the platform-mismatch guard both apply).

## 5. Remaining Sphinx documentation pages

**State:** The docs port session finished the build scaffold, the background
pages, the user-API pages, and the two sharding developer pages; six pages
remain (two blocked on the kernel work, four held at Greg's request), and the
package-side instruction list in `plans/torch_port/docs.md` awaits execution.

**Overview:** The remaining build warnings are references to the unwritten
developer pages, so finishing them also gets the build to warning-free (and
lets CI adopt `-W`).  `usr_multi_gpu.rst` stays deferred until item 2 lands,
per the revised decision in docs.md §4.

**Goals:**
1. The five developer pages.
2. The docs.md package-change instructions (narrowed `__all__`, the four
   docstrings, the cross-reference demotion, the print_params guard).
3. `usr_multi_gpu.rst` after the device-policy flip.

## 6. Additional geometries: translation and multiaxis

**State:** Not started (~3.4k lines in mbirjax; the beyond-gates tail of the
port plan).

**Overview:** Port TranslationModel and the multiaxis parallel geometry.  The
jax regression harness already carries both, so gate rows have baselines from
day one.  Translation is where two recorded lessons become first-class: the
pixel-axis chunking need (detectors heading to 6K x 10K views) and the
collision-cliff class (probe production shapes before enabling any tuned
path).

**Goals:**
1. Port both geometries with goldens and convergence parity at the
   established floors, including the translation companions the application
   scripts call: `gen_translation_vectors`, `display_translation_vectors`,
   `calc_tct_recon_params`, `gen_translation_phantom`, and the
   `delta_recon_row` parameter (a parameter-handler extension).
2. Gate rows on both platforms; kernel treatment decided by measurement (the
   torch bodies may suffice — the Phase 5 protocol applies if not).

## 7. Release workflow

**State:** A complete proposal exists at
`plans/torch_port/release_workflow.md`; nothing implemented.

**Overview:** GitHub-Release-driven publishing for mbirtorch: the
main/prerelease branch model (renaming `master`), CI on pull requests, PyPI
via Trusted Publishing with a tag-vs-version check, and Read the Docs wired to
`stable`/`latest`.  The proposal ends with five open decisions — the GPU
coverage gap in CI, the TestPyPI cadence, the Python version matrix,
post-publish wheel testing, and the release home repository.

**Goals:**
1. Decide the five open questions and refine the proposal accordingly.
2. Implement: the eight manual setup steps, the three workflow files, and the
   `dev_maintenance.rst` rewrite around the new routine.

## 8. Remaining utility API surface

**State:** Confirmed 2026-08-07 (Greg).  The work list is the absent-API
census in `plans/torch_port/docs.md` §5.  The preprocessing piece is
delegated to Charlie's session under its own plan,
`plans/torch_port/preprocessing.md`.

**Overview:** Twenty-one documented mbirjax names do not exist in mbirtorch,
and replacement needs most of them: the HDF5 save/load family (recon and data),
the model factories (`get_ct_model`, `copy_ct_model`, `build_model`), the demo
utilities, and the preprocessing/MAR entry points (`gen_weights_mar`,
`median_filter3d`).  The docs pages carry PENDING markers that restore as each
lands.

**Goals:**
1. Charlie's session, per `preprocessing.md`: the preprocessing package plus
   MAR, `gen_weights_mar`, `median_filter3d`, the download utilities, the
   HDF5 save/load family (with `get_recon_dict`/`get_all_params`), and the
   `hsnt` and `vcls` modules, gated end to end on the Lilly NSI script.
2. The factories, the phantom/demo-data generators, and `device_summary`.
3. `split_sino_recon`, ported with the full mbirjax logic (overlapping
   detector-row halves reconstructed separately and stitched, with
   `stitch_arrays`).  This is a capacity feature, not a compatibility shim:
   it nearly doubles the feasible cone recon size at a fixed GPU count, and
   it composes with the item-2 preflight (which budgets it per half).
   Reversed from the docs replaced bucket 2026-08-07 — the nsi
   split-sinogram demo calls it.

## 9. MAR: cache H

Reframed 2026-07-10 (from `mar_refactor_plan.md` Phase 3, which is SPEED-only — the
fit's memory was solved in Phase 2):

- **The direction is caching**: compute each `H` column once instead of the
  O(num_cols²) recompute — a cheap, self-contained win with no statistical questions.
- **Subsampling is deprioritized**: uniform view/stride subsampling is wrong for this
  fit (the model is identifiable only from pixels spanning diverse metal path length,
  which are sparse in a mostly-plastic object), and metal-thresholded stratification is
  exactly the finicky-threshold pattern the streak-remedy work deliberately avoided.  If it
  is ever revisited: A/B the estimation in isolation first (fitted `theta` + corrected
  recon, full vs subsample) and gate on the corrected recon within a documented
  tolerance — not byte-identical by design.

## 10. LEAP/SVMBIR interfaces (back-burnered)

**State:** Back-burnered per the port decision (`port_plan.md` §1): a jax-side
wrapper would be throwaway work if replacement proceeds, and LEAP itself
presents as a PyTorch front end, so the wrapper is thinner on mbirtorch.

**Overview:** Some of our collaborators are using LEAP (https://github.com/LLNL/leap)
or SVMBIR (https://github.com/cabouman/svmbir - parallel beam only).  We'd like to
lower the barriers to entry for them to transition to using MBIRJAX.  So,
we'd like to develop an easy way to replace LEAP/SVMBIR with MBIRJAX.

**Goals:**
1. Download and run some examples using each of these packages.
2. Determine the scope of a possible translation from these to MBIRJAX.
3. Design, implement, and validate interfaces.

## 11. Miscellaneous / cleanup

- **Compile-free memory preflight in `recon()`** (parked 2026-07-10; priority
  raised 2026-08-07 — the mbirtorch charter is §2, and this design transfers to
  it; design agreed with
  Greg after a student's 2-GPU full recon at 1600×1617×1422 spent 32 min in XLA's BFC
  retry loop before surfacing RESOURCE_EXHAUSTED — ~1,900 warning lines; the allocator's
  retry policy is not user-tunable, so fail BEFORE the first doomed allocation).
  Two parts:
  1. *The gate*: a closed-form per-device peak ledger,
     `_estimate_peak_device_bytes(sino_shape, recon_shape, partition_sequence,
     weights_present, placement)`, checked ONCE at the top of `recon()` before any big
     allocation (pool ≈ fully free via `device.memory_stats()`), so failure lands in
     seconds with one readable error naming the dominant phase + the existing remedy
     hint; `skip_memory_preflight` override, ~10–15% margin.  The ledger enumerates,
     per phase, persistent set + largest co-live transient lineup: persistent = sino +
     weights + error_sino (3× sino-shaped) + flat_recon + fm_hessian (2× recon-shaped);
     subset update per granularity = the granularity-INDEPENDENT sino-shaped pair
     (`weighted_error_sinogram` + `delta_sinogram` — what killed the student's run;
     skip-1 sequences don't touch these) + 4–5 subset-shaped arrays (freed at the
     mid-updater `del`) + the projector transient as a measured geometry-dependent
     multiplier (constants already in code comments / the dashboard 12× aggregate).
     Max over phases and over the granularities actually in the sequence.  A MODEL, not
     a compile query, because the updater is eager Python — no single
     `memory_analysis()` sees the cross-call lineup — and because it must run before
     the compiles it would otherwise wait for.  Covers split_sino_recon (per half),
     prox, and the denoiser for free via the `vcd_recon` entry path.
  2. *Calibration, not user-facing*: at compile sites, a CI/debug-mode assertion
     compares each program's actual `compiled.memory_analysis().temp_size_in_bytes`
     against the modeled term and warns on excess; with the nightly `peak_bytes_in_use`
     gates, model drift is caught by the dashboard, not by user crashes.
  Implementation details settled: ledger terms overridable per geometry (cone vs
  parallel differ mainly in the projector multiplier); print the ledger at verbose≥2 on
  successful runs (free memory-budget printout, keeps the model inspectable).  Also
  catches most of the removed multi-GPU-OOM-hang family (deterministic OOMs die before
  any device enters a collective).  Related UX notes from the same incident: stdout
  block-buffering makes sweep logs non-chronological (`python -u`); `TF_CPP_MIN_LOG_LEVEL=2`
  can silence a residual BFC warning wall (document in the OOM hint, don't default).
- **Device-count policy:** a simple, robust rule — even if suboptimal — over a tuned
 choose-N-vs-communication model; this area is potentially finicky for a modest payoff.
  1. **Concrete first step:** fix the auto-device-count basis.  `_auto_device_count` trims
  on the recon-SLICE axis, but projection compute lives on the VIEW-owners, so the slice
  axis is the wrong proxy for "does this device do real work" — switch the basis to
  views (or both).  Small, clearly right, and independent of any cost model.
  2. The full choose-N policy (when does adding a device pay vs its comms cost, incl. the
  CPU-cluster topology rule) stays deferred unless a real workload demands it
  (`sharding_implementation_plan_v3.md` §5/§6) — with the mbirtorch widening rule
  (§2 above) as the live consumer of whatever simple rule emerges.
- **mbirtorch denoiser sharding gap (found 2026-08-07 during the
  device-policy design):** `QGGMRFDenoiser.denoise` raises under any
  non-trivial placement — it calls `.clone()` on a `_shard_recon` result and
  `Shards` has no `clone`.  The device-policy work deliberately leaves the
  denoiser single-device and outside the widening; fix or formally scope the
  denoiser to one device (error message at configure time) as a small
  follow-up.
- **Suite efficiency:** simplify tests and reduce time on tests.
- **Minor API opens:** `configure_devices`/`use_gpu` unification; the forward
  pixel-batch default.
- **Residual rounding-bug risk (monitor only)**: the six vertical-fan per-slice round
  sites keep the in-jit precondition — accepted + monitored
  (`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md` §7).
- **Archive plans:** Many plan docs and scripts could be moved out of the repo and into
 archived storage - e.g., another repo or data depot.

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

## 13. Sorted-stream parallel forward (chartered, after items 2–3)

**State:** Chartered 2026-08-07 (Greg); sequenced explicitly after items 2–3.
The entry gate is item 3's forward-attribution arm, and the decision record
that declined this work at the batching checkpoint — with its revisit
triggers — is in `plans/torch_port/kernel_batching_findings.md`.

**Overview:** The parallel forward is the one remaining measured gap: 14.4 s
over jax at the 1024 cell, atomic-bound and flat across batch sizes, where
jax runs its sorted-stream pallas design (2.1x over plain scatter on subset
batches, 16–26x in the fine tail with plan caching).  A torch sorted-stream
forward would build its streams from the existing hfan contract and cache
them through the `plan` slot every body already accepts.  Memory headroom is
ample (0.57–0.63x of jax) and the kernel protocol machinery (emulator,
battery, sweep, five-arm gate) is amortized; the cost is the plan builder,
the cache lifecycle, and the two-phase kernel itself.  The value likely
grows by waiting: translation (item 6) shares this forward structure, and
atomic contention rises with detector scale.

**Goals:**
1. A feasibility probe on item 3's attributed numbers: plan build cost and
   stream sizes at the gate cells, with the view-loop forward variant (the
   note recorded in the kernel's docstring) as a cheap comparison arm in the
   same harness.  Stop here if the projected composed win falls below a few
   seconds.
2. If the probe holds: the sorted-stream kernel through the full protocol —
   emulator validation, CUDA battery, constant sweep, composed five-arm
   gates against the then-current baselines — with the default flip at a
   Fable checkpoint, as for the four shipped kernels.

---

**Recently completed (records live elsewhere):** the **sharpness/snr_db streak
study, CLOSED 2026-08-07** — the two-start experiment showed the streaks are the
converged minimizer's own content under lateral truncation (a ground-truth start
converges to the same streaked solution, which fits the truncated data 2.3×
better than the truth), so the per-iteration schedule premise is refuted and the
remedy question moves to truncation handling in the model
(→ `plans/sharpness_schedule/converged_streaks.html`, with the Phase A/B record
in `findings.html` and `phase_b_results.html`); the **mbirtorch Phase 5 Triton
kernel campaign** — all four projector kernels default-on, the replacement rule
passing at every gate cell of both geometries
(→ `plans/torch_port/phase5_findings.md`); the **mbirtorch slice viewer port**
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
