(For Charlie's session.  One session covers both items deliberately: the
demo-data generator it ports in item 8 is exactly what unblocks the
commented-out documentation it restores in item 5.)

We're continuing the mbirtorch port (the `mbirtorch` repo; mbirjax is the
READ-ONLY reference).  This session's tasks: finish `current_plans.md`
item 8 (the remaining utility API surface) and execute item 5 (the
remaining documentation), in that order where they couple.

**IMPORTANT — workflow reminders:** work in PR-sized increments as before
(Greg reviews).  No plan notation (item numbers, increment names) in code,
tests, or docs.  Announce golden regeneration before touching
`tests/goldens/` — a silent regeneration once invalidated another session's
suite baseline.  Durable notes in the plans repo follow
`.claude/writing_style.md`.

Read for orientation:
1. `plans/torch_port/active/preprocessing.md` — your original plan; its porting
   rules, golden protocol (SHARED INPUTS, measured tolerances,
   batch-invariance tests), and coordination rules all still apply.
2. `plans/torch_port/active/docs.md` — THE work list for item 5: sections 1–10
   are instructions, several already partially applied; section 8 (restore
   landed PENDING blocks plus the `__all__` rule), section 9 (the census
   update), and section 10 (the kill switch, for the kernel developer
   page) are the open ones.
3. `plans/current_plans.md` items 5 and 8 — current status.
4. `.claude/lessons.md` §2 and §5 — the tolerance and measurement rules
   your tests follow.

## Item 8 — what remains

1. **`get_ct_model`** and the demo-data generators (`generate_demo_data`,
   `generate_3d_shepp_logan_reference`): faithful ports per the
   preprocessing.md rules, goldens per the established mechanism.
2. **`device_summary`: CONFIRM REPLACED — do not port.**  The application
   census found no script calling it, and `get_memory_stats` covers the
   need; the docs keep its REPLACED marker.  This closes the last
   replaced-name question.
3. **The Lilly NSI end-to-end run** (the increment-4 gate of
   preprocessing.md): the translated script against real data, MAR recon
   compared to the mbirjax run at the documented tolerance.

## Item 5 — the documentation

Work from `docs.md`; the highlights, in a sensible order:

1. The remaining section-8 PENDING restores with their `__all__` additions
   (the invariant: documented if and only if declared), and the section-9
   census updates — several of the names are your own recent landings
   (hsnt, vcls, `split_sino_recon`, `stitch_arrays`, `copy_ct_model`).
2. `usr_multi_gpu.rst` REWRITE — now unblocked: the all-device default is
   live.  Content per docs.md §4: the auto-spread default, the memory
   preflight that guards it (user knobs `skip_memory_preflight` and
   `memory_preflight_margin`), `configure_devices(num_devices=1)` as the
   reproducibility pin, and one sentence that results can differ slightly
   with device count and the difference decays with iterations.  The
   constructor no longer takes a device argument — `configure_devices` is
   the single door; the related paragraphs in `overview.rst`,
   `advanced_features.rst`, `usr_tomography_model.rst`, and the FAQ revert
   toward mbirjax's wording per the docs.md TBD notes.
3. The Demos and Data Generation halves of `demos_and_faqs.rst` restore
   once `generate_demo_data` lands (the item-8 coupling).
4. `dev_projector_kernels.rst` — now unblocked (the kernel work is
   stable); document the kill switch per docs.md §10 (escape hatch, read
   at first probe and cached per process so it is set before the first
   model, never presented as a user knob).  `dev_api.rst` and
   `new_model_template.py` ride with it.
5. STILL HELD (do not write): `install.rst`, `dev_maintenance.rst` (waits
   on the release workflow), `dev_performance_dashboard.rst`.
6. Rebuild after each restore; the standing checks are the warning count
   and the page-size comparisons against mbirjax's build.

## Coordination

Other sessions are active in mbirtorch (nightly wiring in mbirjax_metrics;
a performance campaign later).  Your shared-file surface is
`mbirtorch/__init__.py` (the `__all__` list ONLY — a lazy-exports block
sits below it; leave that untouched, and note new hsnt/vcls public
functions each need a `_LAZY_NAMES` line) and `pyproject.toml` (isolated
commits, as before).  Everything under `docs/`, `mbirtorch/preprocess/`,
and your new modules is yours.  Questions and upstream surprises go to
Greg rather than being worked around silently.
