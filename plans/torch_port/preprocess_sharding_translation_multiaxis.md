# Instructions re the checklist below


## Overall

Much of this work can proceed in parallel with the multi-gpu investigation, but the right split is by what each item **touches**, not by section. The stable surface is the Shards/placement API and the public projection funnels; the moving parts are the banded drivers' internals, the memory ledger, the automatic device policy, and the chunk constants. Items that only *consume* the stable surface parallelize cleanly; items that *modify or depend on* the moving parts should wait or ride the campaign. That test sorts section A into three bins rather than one.

In the mbirjax_plans repo, please read /plans/current_plans.md, plans/torch_port/multigpu_plan.md, and .claude/lessons.md for context.  
This file is available at https://github.com/gbuzzard/mbirjax_plans/blob/main/plans/torch_port/preprocess_sharding_translation_multiaxis.md.
Please keep it up-to-date and pushed. Mark an item as "STARTED" when you start one of these tasks and "COMPLETED" when you finish.
Please leave a status report for each item you complete or for blocked/not fully resolved items.  
If an item is marked as "STARTED" when you're ready to begin a new task, then choose a different task.
mbirtorch work for these items lands on the `prerelease` branch.

## Section A, per item

**Go now: A1, A5, A3, A4.**
- **A1 `segment_plastic_metal`** first — it's a real observed blocker (full-res MAR OOM, job 15001292), it's preprocess-side with zero campaign overlap, and the per-shard-Otsu design consumes only the stable Shards surface. It also plausibly gates the Lilly comparison in section C if that run wants full-res MAR, which is another reason to front-load it.
- **A5 `export_recon_hdf5`** is a trivial additive rider. One recorded gotcha for whoever writes it: `Shards.gather()` already returns numpy — re-detaching it is the exact bug that cost the nightly's first 4-GPU trial its n>1 rows.
- **A3 pipeline view-sharding** is additive preprocess work on the same stable surface. One design note: its device choice should follow the visible-device list / model placement, not the recon policy — the widening guard now landing is recon-path-only, and preprocess shouldn't consult floors calibrated on vcd.
- **A4 `QGGMRFDenoiser`** — the sharded denoiser rides the prior/halo path, not the projector internals that Greg's multi-gpu campaign is tuning.  Placing A4 in this list resolves the `current_plans.md` §11 denoiser scope decision in the full-parity direction (Greg, 2026-08-09).  That §11 item bundles two companion gaps, and A4 includes both: the `.clone()`-on-`Shards` failure, and the log arguments the other entry points gained.

**Defer: A2 (`direct_recon` device policy).** Not because it's big — Charlie's "small fix" label is mechanically right — but because it's the same code the guard patch is about to modify, and the semantics question is real: the guard's floors are being calibrated on 3-iteration vcd (the plan says explicitly the guard's subject is out-of-box `recon()`), while a standalone direct recon is one filter plus one back projection with a completely different crossover profile. "Just call `_apply_device_policy`" would consult the wrong ruler. The right semantics — capacity-only, own floors, or guard-exempt — is a guard-design question. This will be done as part of the larger multi-gpu campaign.

**Skip: A6 (sharded phantom).** That's item 15 — on deck after item 3, with the design already recorded (sharded output over the *consuming model's own* placement, identity check, byte-equality gate across counts). Charlie's checklist should mark it covered elsewhere so two sessions don't build it twice.

## Section B: yes, go in parallel

Both mbirjax modules are fully multi-device (Fable verified, 2026-08-09), so per the checklist's own rule the ports owe the device case too — but that's fine, because new geometries *implement* the settled engine contract (the per-view-batch bodies and view-range seams parallel and cone already define) rather than modifying the shared drivers the campaign is tuning. Collision risk is low, and charter A/B improvements to the shared drivers land underneath the new geometries for free. Four riders:

1. Build to the existing body contract as-is, including the `plan` slot — no speculative accommodation for the sorted stream; item 13 was designed to arrive compatibly through that slot later.
2. Translation's back projection at production translation scale wants the pixel batching that charter B/C may restore. Record it as a known scale limit at port time; don't build a workaround the structural remedy will obsolete.
3. New-geometry parity against mbirjax uses opt-in goldens, per the recorded ruling that porting charters opt in explicitly.  Mark the new tests with the `goldens` pytest marker, following the existing pattern in `tests/`.  Generate any new golden archives from the mbirjax env, and announce each generation or regeneration to Greg.
4. Merge hygiene: keep the work off `tomography_model.py`, `projectors.py`, `_memory_ledger.py`, and the policy code while charters A/B land. New-module work naturally does.

**Net recommendation:** Charlie proceeds now with A1 → A5 → A3 → A4 and section B in parallel; A2 rides the guard; A6 stays item 15. 

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
- [ ] **A7 (found during A1)** `preprocess/mar.py: correct_sino_plastic_metal`
      — everything downstream of segmentation (sinogram maxima and
      normalization, the polynomial H fit, the constraint updates) operates
      on plain tensors and has never run multi-device; mbirjax's version
      works on sharded sinograms via jax's global-array semantics, so its
      docstrings never say "sharded" and the checklist grep missed it.
      Full-resolution MAR needs this after A1.  Needs Greg's triage: it
      touches sinogram-side statistics, not the projector internals.
- [ ] `tomography_model.py: direct_recon / fdk_recon` — never make the
      use-N-GPUs decision (`_apply_device_policy` runs only in recon).
      A direct FDK call runs on 1 GPU.  Small fix.
      DEFERRED — rides the multi-gpu campaign's guard change (see A2 above).
- [ ] `preprocess/pipeline.py` (scan preprocessing) — mbirjax runs it
      view-sharded, one share per device; mbirtorch has no multi-device
      mode at all.
- [ ] `denoising.py: QGGMRFDenoiser` — single-device only (its docstring
      says so).  mbirjax slice-shards the image like a reconstruction.
      Known gap in Greg's plan (denoiser `.clone()` on Shards fails).
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
Worth re-sweeping after the geometries land.

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
