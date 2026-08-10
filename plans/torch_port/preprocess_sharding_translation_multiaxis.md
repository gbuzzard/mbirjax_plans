# Instructions re the checklist below


## Overall

Much of this work can proceed in parallel with the multi-gpu investigation, but the right split is by what each item **touches**, not by section. The stable surface is the Shards/placement API and the public projection funnels; the moving parts are the banded drivers' internals, the memory ledger, the automatic device policy, and the chunk constants. Items that only *consume* the stable surface parallelize cleanly; items that *modify or depend on* the moving parts should wait or ride the campaign. That test sorts section A into three bins rather than one.

In the mbirjax_plans repo, please read /plans/current_plans.md, plans/torch_port/multigpu_plan.md, and .claude/lessons.md for context.  
This file is available at https://github.com/gbuzzard/mbirjax_plans/blob/main/plans/torch_port/preprocess_sharding_translation_multiaxis.md.
Please keep it up-to-date and pushed. Mark an item as "STARTED" when you start one of these tasks and "COMPLETED" when you finish.
Please leave a status report for each item you complete or for blocked/not fully resolved items.  
If an item is marked as "STARTED" when you're ready to begin a new task, then choose a different task.
mbirtorch work for these items lands on the `prerelease` branch.

**Writing rule (Charlie, 2026-08-10).** Write plan entries and status
reports in plain English.  Technical terms are fine when they are
broadly understood by the community or defined where they are first
used.  Do not use invented or undefined jargon (for example, "merge
hygiene" — say instead "do not edit these files while Greg's changes
are landing").  The same rule applies to code comments.

## Section A, per item

**Go: A4 is in flight; section B is the open front.  (A1, A5, A3, A7
COMPLETED 2026-08-09.)**
- **A1 `segment_plastic_metal`** — COMPLETED; status report at the checklist entry.
- **A5 `export_recon_hdf5`** — COMPLETED; status report at the checklist entry.
- **A3 pipeline view-sharding** — COMPLETED; status report at the checklist entry.  One piece is still owed per its own caveat: the concurrency win is unmeasured until a Gautschi run.
- **A7 `correct_sino_plastic_metal`** — COMPLETED; status report at the checklist entry, design gates met (f64 host partials, exact maxima, host-side OSQP).  Its ENGINE NOTE is dispositioned (Fable verified it at source, 2026-08-09): the tensor-only init_recon validation at `tomography_model.py:2469` is real, and it is the only barrier — the engine's own default init is already sharded (`direct_recon(..., output_sharded=True)` at :2465 feeds the same `_shard_recon` passthrough).  The fix rides charter B's `tomography_model.py` commit in the multi-gpu campaign; Fable relays it at the next campaign gate.  The one-gather-per-pass workaround is correct meanwhile: the gather is host-side and the re-shard moves per-slice, so no full volume ever lands on one device, and the Gautschi rerun need not wait.
- **A4 `QGGMRFDenoiser`** — the sharded denoiser rides the prior/halo path, not the projector internals that Greg's multi-gpu campaign is tuning.  Placing A4 in this list resolves the `current_plans.md` §11 denoiser scope decision in the full-parity direction (Greg, 2026-08-09).  That §11 item bundles two companion gaps, and A4 includes both: the `.clone()`-on-`Shards` failure, and the log arguments the other entry points gained.

**Defer: A2 (`direct_recon` device policy).** Not because it's big — Charlie's "small fix" label is mechanically right — but because it's the same code the guard patch is about to modify, and the semantics question is real: the guard's floors are being calibrated on 3-iteration vcd (the plan says explicitly the guard's subject is out-of-box `recon()`), while a standalone direct recon is one filter plus one back projection with a completely different crossover profile. "Just call `_apply_device_policy`" would consult the wrong ruler. The right semantics — capacity-only, own floors, or guard-exempt — is a guard-design question. This will be done as part of the larger multi-gpu campaign.

**Skip: A6 (sharded phantom).** That's item 15 — on deck after item 3, with the design already recorded (sharded output over the *consuming model's own* placement, identity check, byte-equality gate across counts). Charlie's checklist should mark it covered elsewhere so two sessions don't build it twice.

## Section B: yes, go in parallel

Both mbirjax modules are fully multi-device (Fable verified, 2026-08-09), so per the checklist's own rule the ports owe the device case too — but that's fine, because new geometries *implement* the settled engine contract (the per-view-batch bodies and view-range seams parallel and cone already define) rather than modifying the shared drivers the campaign is tuning. Collision risk is low, and charter A/B improvements to the shared drivers land underneath the new geometries for free. Four riders:

1. Build to the existing body contract as-is, including the `plan` slot — no speculative accommodation for the sorted stream; item 13 was designed to arrive compatibly through that slot later.
2. Translation's back projection at production translation scale wants the pixel batching that charter B/C may restore. Record it as a known scale limit at port time; don't build a workaround the structural remedy will obsolete.
3. New-geometry parity against mbirjax uses opt-in goldens, per the recorded ruling that porting charters opt in explicitly.  Mark the new tests with the `goldens` pytest marker, following the existing pattern in `tests/`.  Generate any new golden archives from the mbirjax env, and announce each generation or regeneration to Greg.
4. Merge hygiene: keep the work off `tomography_model.py`, `projectors.py`, `_memory_ledger.py`, and the policy code while charters A/B land. New-module work naturally does.

**Net state (2026-08-09, later):** A1, A5, A3, A7 complete; A4 in flight; section B is the open front.  A2 rides the guard; A6 stays item 15.  Section A's close-out gate is the Gautschi rerun of job 15001292's configuration, and it can run now (the init-gather workaround is correct; see A7).

**Update (2026-08-10): the Gautschi rerun ran and failed, blocked on A2.**
Job 15047383 (full-resolution Lilly MAR, 4 H100s, mbirtorch prerelease
944aec2) ran out of GPU memory 2m40s in, at the segmentation step of the
first beam-hardening pass.  Cause: the initial FDK reconstruction never
makes the automatic choice to use all the GPUs (that choice is made only
inside the iterative solver), so FDK built the full 18 GB volume on GPU 0
alone.  Segmentation then received an ordinary tensor on GPU 0, took its
single-device path, and could not allocate the 18 GB it needed for a
class mask.  All of the sharded code from A1/A7 is in place and was never
reached.  So the final real-hardware test of section A waits on the A2
fix (deferred to the multi-gpu campaign, see above).  Decision (Charlie,
2026-08-10): do not work around it by setting the device list in the
application script; rerun the job after A2 lands.

# mbirtorch port — remaining work (updated 2026-08-09)

Read and update this during mbirtorch work.  Rule: a function is not
fully ported until its multi-device case works.  Every function that
supports multiple devices in mbirjax must support them in mbirtorch.

## A. Multi-device checklist

Go down this list.  Each entry names the mbirtorch function and the
mbirjax reference implementation.

- [x] COMPLETED 2026-08-09 (Charlie's session, mbirtorch a5b04ce)
      `preprocess/segmentation.py: segment_plastic_metal`.
      Status: per-shard histogram in bounded chunks, int64 host sum, so
      thresholds equal the unsharded ones exactly; edge masking, class
      masks, and scale factors run shard by shard; masks return sharded on
      the volume's devices.  Inputs follow the array-forms rule (numpy /
      tensor / 1-shard Shards / N-shard Shards in, same form out; rule
      recorded in Charlie's notes, emailed to Greg).  Three tensor-only
      seams in _est_plastic_metal_sinos_from_recon fixed with it.  Gate:
      sharded vs unsharded masks identical (tests/test_sharded_segmentation.py);
      suite 368 + goldens 62 pass; single-device behavior unchanged.
      NOT covered (new item A7 below): the downstream beam-hardening fit.
- [x] COMPLETED 2026-08-09 (Charlie's session, mbirtorch 944aec2)
      **A7** `preprocess/mar.py: correct_sino_plastic_metal`.
      Status: the whole BH chain runs on sinograms in either form via
      form-agnostic helpers at the top of mar.py -- elementwise per piece
      on its own device; reductions combine per-piece partials on the host
      in f64 (the design-gate option); maxima exact; OSQP host-side
      unchanged; the constraint argmin combines with per-piece view
      offsets, preserving tie-breaking.  Gate: seeded 2-CPU-shard vs
      1-device full MAR pipeline rel_max 1.7e-4 (< the 1e-3 full-pipeline
      gate); corrected sinogram alone ~1e-6.  Suite 372 + goldens 62 pass.
      ENGINE NOTE for the campaign: recon()'s user-supplied init_recon
      validation is tensor-only (np.shape at tomography_model.py:2469), so
      recon_plastic_metal gathers a sharded init before each pass (one
      gather per BH pass).  A one-line shards-aware validation there
      removes the gather; deferred per merge hygiene.
      With A1 + A5 + A7 the full-res MAR chain is sharded end to end on
      paper; the Gautschi rerun of job 15001292's configuration is the
      remaining real-hardware gate.
- [ ] `tomography_model.py: direct_recon / fdk_recon` — never make the
      use-N-GPUs decision (`_apply_device_policy` runs only in recon).
      A direct FDK call runs on 1 GPU.  Small fix.
      DEFERRED — rides the multi-gpu campaign's guard change (see A2 above).
      Now the one blocker for the full-res MAR hardware test: job 15047383
      (2026-08-10) failed on exactly this (see the Update note above).
- [x] COMPLETED 2026-08-09 (Charlie's session, mbirtorch 712c523)
      `preprocess/pipeline.py` (scan preprocessing).
      Status: map_view_batches spreads the views over a device list --
      contiguous in-order blocks, one worker thread per device, disjoint
      writes into the one pre-allocated host output (the mbirjax driver's
      design; host footprint stays input + output).  scan_to_sino defaults
      to all visible CUDA devices; per the design note the recon policy is
      NOT consulted anywhere in preprocess.  Gate: 1-device vs 3-device
      results byte identical (tests/test_sharded_pipeline.py); suite 370 +
      goldens 62 pass.  Caveat: correctness gated on CPU pseudo-devices;
      the concurrency win itself is unmeasured until a Gautschi run.
- [x] COMPLETED 2026-08-09 (Charlie's session, mbirtorch 2b1d02c)
      `denoising.py: QGGMRFDenoiser`.
      Status: two paths as in mbirjax -- one device keeps the compiled
      in-place sweep (golden vs mbirjax unchanged); several devices
      slice-shard the image, stage the qGGMRF halos once per pass, and
      combine the four line-search sums on the host into one alpha (the
      single-device formula).  Both companion gaps closed: the
      .clone()-on-Shards crash, and denoise now takes logfile_path /
      print_logs and returns via get_recon_dict (recon_log + notes ride
      along).  Automatic widening still does not apply to the denoiser --
      multi-device is explicit via configure_devices, matching the
      guard-scope decision.  Gate: 1-device vs 2-padded-CPU-shards rel_max
      < 1e-4 (tests/test_denoiser.py); suite 371 + goldens 62 pass.
      Caveat: CPU-shard correctness only; GPU concurrency unmeasured.
- [x] COMPLETED 2026-08-09 (Charlie's session, mbirtorch c5b5438)
      `utilities.py: export_recon_hdf5`.
      Status: _to_host (shared by export_recon_hdf5 and save_data_hdf5)
      gathers a Shards container on the host and crops the sharded axis's
      zero-padding, so the file equals the single-device export byte for
      byte.  The recorded gather()-already-returns-numpy gotcha is heeded.
      Gate: whole vs 2-padded-shards export files load back equal
      (tests/test_sharded_segmentation.py); suite 369 passed.
- [ ] `utilities.py: generate_3d_shepp_logan_low_dynamic_range` — mbirjax
      can build the phantom slice-sharded across devices; mbirtorch builds
      whole on the host (documented divergence at utilities.py:136).
      SKIP — assigned as current_plans item 15 (see A6 above).

Found by grepping mbirjax docstrings/comments for "shard" (2026-08-09).
That grep has a blind spot, and A7 is its proof: jax's global arrays
make sharding transparent, so a sharded-capable mbirjax function may
never say "shard" anywhere.  Re-sweep after the geometries land by the
method that caught A1's seams: from each entry point that now accepts
Shards, trace the downstream callees for tensor-only assumptions.

## B. Unported modules

- [ ] `translation_model.py` (+ gen_translation_phantom, the demo-data
      translation branch, staged doc page)
- [ ] `multiaxis_parallel.py` (+ staged doc page)
- `vcd_utils.py` blue-noise partition functions: NOT porting (Charlie,
  2026-08-09).

## C. Chores / Charlie-side

- [ ] mbirjax comparison run on Lilly (speed + memory); script exists on
      Gautschi at ~/GitHub/mbirjax_applications/nsi/Lilly_recon.py
- [x] Gautschi cache cleanup (2026-08-09): ~/.mbirtorch removed, ~/.bashrc
      points both compile caches at /scratch/gautschi/bouman/torch_cache.
- [ ] Charlie: plans-fork PR to Greg (cf4882c)
- [ ] Charlie: tell Greg about split-sino device handling
      (split-sino-device-handling.md)
