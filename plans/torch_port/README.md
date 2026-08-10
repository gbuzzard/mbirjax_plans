# torch_port — what lives where

This folder holds the planning and findings documents for the PyTorch port of
mbirjax.  The documents are filed by lifecycle, which means by whether the work
they describe is still open.  The four subfolders are `active/`, `closed/`,
`phases/`, and `reviews/`.  One document, `port_plan.md`, stays at the top
level.

The experiment scripts for this program are not here.  They are in
`plans/experiments/torch_port/`, which has its own README.

## The four subfolders

`active/` holds the documents you need to read today.  A document belongs here
while the `plans/current_plans.md` item it serves is still open.

`closed/` holds the records of campaigns that finished.  These are still cited
by open work, so they stay in the tree rather than being deleted.

`phases/` holds the record of the original six-phase port, phase 0 through
phase 5.  A phase document goes here even when later work cites it heavily.
The folder name means "part of the original six-phase port", and phase 5 is one
of the six.

`reviews/` holds review archives.  These were written once, are read for
reference, and are never updated.

## Why `port_plan.md` is not in a subfolder

`port_plan.md` is the plan of record for the whole program, and it indexes all
six phases.  It is also the one path in this folder that is quoted from the
other repository: `mbirtorch/README.md` line 6 reads
`mbirjax_plans/plans/torch_port/port_plan.md`.  Keeping the file at the top
level costs one entry in the listing and keeps that cross-repository reference
correct without an edit.

## The rule for keeping this organized

A document lives in `active/` while its `current_plans.md` item is open.  It
moves to `closed/` in the same commit that marks the item COMPLETE.  Applying
the rule at close time is what stops `active/` from filling up with finished
work.  Closing item 3, for example, is one `git mv` of five known files:
`multigpu_plan.md`, `multigpu_findings.md`, `forward_remedy_memo.md`,
`entry_point_survey.md`, and `backloop_attribution.md`.

Two standing exceptions are worth stating, because a reader will otherwise
question them.  `array_forms_rule.md` is a standing rule rather than a campaign
record, so it stays in `active/` and does not close.  `backloop_attribution.md`
stays in `active/` even though its finding was verified and endorsed, because
item 3 is still open and `multigpu_findings.md` section 6.3 sends readers to it.

## Why this index exists

Documents in this tree refer to each other by bare filename, with no path, in
about 72 places.  None of those references is a hyperlink, so nothing breaks in
a renderer.  What changed is that a bare filename and a sibling file used to be
the same thing, and now they are not.  The table below turns any bare filename
into a folder in one lookup.

## Index — every document and its folder

| document | folder | what it is |
|---|---|---|
| `port_plan.md` | *(top level)* | The plan of record: motivation, the replacement decision rule, the parity gates, and the six-phase plan. |
| `array_forms_rule.md` | `active/` | Standing rule: the four input forms a user-facing transform accepts and returns.  Adopted 2026-08-09. |
| `backloop_attribution.md` | `active/` | Item 3, charter B step one: attribution of the direct-recon back loop at more than one device. |
| `docs.md` | `active/` | Item 5: the Sphinx documentation port.  Section 5 is the work list. |
| `entry_point_survey.md` | `active/` | Survey of every entry point that allocates a device.  Design input for extending `_apply_device_policy`. |
| `forward_remedy_memo.md` | `active/` | Item 3, open decision: driver change against item 13's sorted stream.  Nothing in it is implemented yet. |
| `multigpu_findings.md` | `active/` | Item 3's results record.  Section 6 carries the two tuning charters. |
| `multigpu_plan.md` | `active/` | Item 3's contract: terms, protocols, triggers, and the mg1 through mg8 instrument charters. |
| `preprocess_sharding_translation_multiaxis.md` | `active/` | Item 6 charter: what parallelizes now, and what must wait for the item 3 campaign. |
| `preprocessing.md` | `active/` | Item 8: the `mbirjax.preprocess` package and the main-package pieces coupled to it. |
| `release_workflow.md` | `active/` | Item 7: the eight manual release steps, and which of them are done. |
| `device_policy_design.md` | `closed/` | Item 2 design: the memory preflight and the all-device default. |
| `device_policy_findings.md` | `closed/` | Item 2 findings: the memory ledger implemented and measured. |
| `kernel_batching_design.md` | `closed/` | Item 1 design: kernel-aware view batching. |
| `kernel_batching_findings.md` | `closed/` | Item 1 findings: the four chunk constants, and the back-path reading attributed to reassociation in the compiled reference. |
| `kernel_sharding_findings.md` | `closed/` | Item 14: diagnosis and repair of the Triton forward projector under the banded drivers. |
| `nightly_plan.md` | `closed/` | Item 4: mbirtorch in the nightly run, including the multi-device cells.  All seven increments shipped. |
| `projector_layer_design.md` | `closed/` | The projector-layer division of labor.  Implemented 2026-08-06. |
| `torch_port_reorg_proposal.md` | `closed/` | The proposal for this folder layout, approved and executed 2026-08-10.  Its inventory describes the layout before the move. |
| `phase0_findings.md` | `phases/` | De-risking spikes.  No blocker found; sized the eager-mode gap at 10x to 15x. |
| `phase1_findings.md` | `phases/` | Parallel-beam vertical slice.  Every gate passes with margin. |
| `phase2_findings.md` | `phases/` | Compile integration and the first gate-cell readout. |
| `phase3_findings.md` | `phases/` | Cone-beam port, validated for parity including helical and curved detector. |
| `phase4_design.md` | `phases/` | Multi-device design spike, and the decision to use gloo on two H100s. |
| `phase4_findings.md` | `phases/` | Multi-device measurements for increments 1 through 3, at one, two, and four devices. |
| `phase5_findings.md` | `phases/` | Triton kernels complete.  Four kernels default-on, and the replacement rule passes everywhere. |
| `phase5_kernel_design.md` | `phases/` | The design of the four Triton kernel bodies.  Sorted streams were deliberately not taken. |
| `dashboard_mockup.html` | `phases/` | Static display mockup for the torch backend on the metrics dashboard. |
| `panel_review.md` | `reviews/` | The 30-agent, five-lens panel review after phase 2.  24 findings confirmed, 1 refuted. |
| `prerelease_review_2026-08-10.md` | `reviews/` | Five incoming `origin/prerelease` commits reviewed against `greg_dev`. |
| `prerelease_review_convergent_2026-08-10.md` | `reviews/` | Four convergent fixes, with the exact resolution and the named cherry-picks. |
| `prerelease_review_multiaxis_2026-08-10.md` | `reviews/` | The multiaxis port and the segmentation overlap commit. |
| `prerelease_review_translation_2026-08-10.md` | `reviews/` | The TranslationModel port.  Verdict: issues found, land now. |
