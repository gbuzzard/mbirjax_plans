# Open items — v5

**Reorganized 2026-08-21.**  This file supersedes `plans/open_items_v4.md`, absorbing its still-live items;
the migration audit and previously closed items are in the last section of this file.  Each open item
appears here once, written to say what is true now rather than how
it got there, and keeping the source citations.  Each closed item
keeps one headline sentence and a pointer to its record.  Items that
closed before v3 was compiled are in v2's audit list (its section J)
and in the retired current_plans.md, which moves to the archive with
H5's sweep.  The port moves daily, so an item here can close between
compilations.

One naming convention, used throughout: the "device-count thresholds"
are the measured problem sizes below which the automatic device choice
keeps a reconstruction on fewer GPUs (the code calls them widening
speed floors).

## Contents

- [Start here: the next session](#start-here-the-next-session)
- [A. CLOSED: Large-scale reconstruction (2048-class and larger)](#a-large-scale-reconstruction-2048-class-and-larger)
- [B. CLOSED: Speed investigations at current sizes](#b-speed-investigations-at-current-sizes)
- [C. Measurement and calibration gaps](#c-measurement-and-calibration-gaps)
- [D. Correctness and API-contract gaps](#d-correctness-and-api-contract-gaps)
- [E. CLOSED: Decisions waiting on a person](#e-decisions-waiting-on-a-person)
- [F. CLOSED: Documentation and demos](#f-documentation-and-demos)
- [G. Release, nightly, and automation](#g-release-nightly-and-automation)
- [H. Scheduled and back-burner work](#h-scheduled-and-back-burner-work)
- [I. CLOSED: Accepted gaps and monitor-only notes](#i-accepted-gaps-and-monitor-only-notes)
- [J. New features and improvements](#j-new-features-and-improvements)

---

## Start here: the next session

The 2026-08-21 session closed G4 (findings §1.40), implemented the
queue Greg ruled on (J3, D6, D5, D1, D2, D9, D8 issues 3, 6, and 7,
and J2), closed D4 (the divided-form sweep; findings §1.41), and
closed J1: parallel `recon_split_sino` reconstructs in N parts, with
the part count taken from `slices_per_part` or estimated from the
memory ledger.  Greg committed all of it in 42574f8; the tip is
4bb3be5 at version 0.0.2.  The suite passes at 679 and the docs
build with no warnings.

What remains open:

- **H8's multiaxis kernel campaign is COMPLETE** (every increment
  landed 2026-08-22; active/multiaxis_kernel_plan.md).  Translation
  kernels wait on the triggers in
  active/translation_kernel_memo.md.
- **D7's multiaxis half is closed by measurement** (findings
  §1.45): the gather-vertical forward organization won.  The
  translation half holds.
- **Follow-ups recorded in place:** the ledger term for the unmasked
  hessian back projection (D8 issue 6's residue), the GPU-scale
  gather confirmation (J3), external material that teaches the old
  method names (D8 issue 7), and the divided-form support candidates
  (findings §1.41).

## D. Correctness and API-contract gaps

**D7. The projection loop organization is unexamined across
geometries.**  The multi-axis port carried over the original's choice of
which loops scatter and which gather.  Charlie's hypothesis (gather on
the vertical axis, scatter on the horizontal) is testable one axis at a
time against the stored reference results.
2026-08-22: the multiaxis half is CLOSED by measurement.  The
multiaxis forward kernel ships the organization Charlie
hypothesized, gather on the vertical axis and scatter on the
horizontal, and the composed route built on it runs 4.0x to 4.6x
faster than the torch bodies' scatter-organized route at every
floors cell (findings §1.45; active/multiaxis_kernel_plan.md).
The translation half holds with translation.
*(open_items D4;
closed/preprocess_sharding_translation_multiaxis.md:273.)*

## G. Release, nightly, and automation

**G1. The release-workflow remainder.**  The Read the Docs stable
default and its token matter only at the first tagged release;
`release.yml` and the optional documentation preview are unwritten; the
wheel-check developer script is unwritten.  The version policy is E7.
*(open_items I1; closed/release_workflow.md:26.)*

**G2. Read the dependency watch's first quiet week.**  The watch is live
and armed; its first week of watchdog lines on the gautschi nightly is
to be read once, by Greg.
*(open_items I2; closed/python_matrix_nightly_check.md:352.)*

**G3. Migrate to mbirtorch_metrics and retire the mbirjax dashboard.** 
A new mbirtorch_metrics repo and dashboard should become independent and record mbirtorch only, 
in a format similar to the mbirjax dashboard prior to the addition of mbirtorch. 
The mbirjax nightly runs will eventually be retired.  

## H. Scheduled and back-burner work

**H1. MAR: cache the fitting matrix.**  Compute each column of the
metal-artifact fit's matrix once instead of recomputing it
quadratically often.  It is a cheap, self-contained speed win with no
statistical questions.  Subsampling stays deprioritized, with the
reasoning recorded.
*(current_plans.md item 9.)*

**H3. Multi-resolution reconstruction, back-burnered.**  A possible future direction (may be superseded by NN/INR approaches): 
reconstruct at binned resolution and upsample as the
initializer for the next-finer level.  The pilot design, the null
hypothesis to beat (a direct-reconstruction initializer), and the
matching problems are in the migrated design notes.
*(torch_port/active/multires_direction.md; current_plans.md item 12,
migrated 2026-08-19.)*

**H5. Archive old plans, back-burnered.**  Move older plan documents and scripts to
archived storage; the plans tree accumulates.  The current repo has multiple directories to organize
by topic.  Could consider a move to a new repo `mbirtorch_plans`. *(current_plans.md item 11.)*

**H6. Test-suite quality and cost.**  The demo-data gates sit 300 to
640x above their measured noise floors; `test_logging` wants a
two-instance case; the cone fixtures could be module-scoped (72 s
serial today); and the suite overall is worth a simplification and
speed pass.
*(current_plans.md item 11, two bullets.)*

**H8. Multi-device speed, part 2: the torch-body geometries.**
Greg ruled 2026-08-22: build hand-written kernels for the multiaxis
geometry now; translation waits and is priced by the multiaxis
work.  The case in brief: the compiled bodies are limited by the
GPU itself and move gigabytes of temporary data per launch,
mbirjax demonstrates no speed these bodies lack, and above the
1024-class the temporaries' memory cost blocks reconstruction
outright.  The campaign ran to completion on 2026-08-22; the
plan and every increment's record are in
active/multiaxis_kernel_plan.md, and the translation price is in
active/translation_kernel_memo.md.  The decision evidence is
findings §1.42 to §1.44, with run detail in
mg51_multiaxis_counters.md, mg52_framework_anchor.md, and
mg53_host_cost_split.md beside the experiment scripts.  The
recompile-budget remedy's two structural alternatives stay in
reserve in active/multigpu_plan_part_2.md.
*(torch_port/active/multiaxis_kernel_plan.md;
multigpu_findings.md §1.36, §1.42-§1.44.)*

# Closed items (2026-08-21)


## A. Large-scale reconstruction (2048-class and larger)

**A1. CLOSED 2026-08-16.  Determine the memory needs and bottlenecks
for 2048-class reconstructions on multiple GPUs.**  The capacity table
opens `torch_port/active/two_k_design.md`.

**A2. CLOSED 2026-08-17, by ruling.  Restructure how partial
back-projections are combined.**  The restructure that landed on
2026-08-11 is accepted; the record is `two_k_design.md` §3.

**A3. CLOSED 2026-08-17.  Validate the memory estimate at the 2048
class.**  The verdicts are in `two_k_design.md` §6 and
multigpu_findings.md §1.20.

**A4. CLOSED 2026-08-17, by ruling.  Sweep the forward pixel-batch size
at large scale.**  The default moved to 32768; the sweep is in
multigpu_findings.md §1.20 and in the constant's comment.

**A5. CLOSED 2026-08-19, with B3.  Two-dimensional tiling and the
cache directions.**  The capacity case was refuted at both measured
scales, and the speed case closed with the kernel questions.
→ multigpu_findings.md §1.24, §1.33; greg_notes.md the closing addendum.

## B. Speed investigations at current sizes

**B1. CLOSED 2026-08-18.  The cone back projection costs more on more
devices.**  The mechanism was the compiler's divisibility
specialization on the back kernel's band argument; the band-padding
remedy landed as mbirtorch 64dedb8 and confirmed everywhere it was
read.  The record is multigpu_findings.md §1.21 (the mechanism) and §1.23 (the
remedy landing), with the design in
`torch_port/active/back_remedy_design.md`.

**B2. CLOSED 2026-08-18.  Pad the kernel width arguments to the next
multiple of 16.**  Landed and confirmed with B1's remedy; arbitrary
user shapes no longer pay the two-times penalty.  → multigpu_findings.md §1.23.

**B3. CLOSED 2026-08-18, by ruling.  In-kernel sorted or segmented
accumulation.**  The profile run (mg25, multigpu_findings.md §1.24) found nothing
for sorting to fix in the cone back kernel: no atomics (measured
zero), one write per output element, gather waste absorbed by the
caches with DRAM nearly idle, and the kernel mildly compute-side.
Greg closed the item on those readings.  Two notes ride the closure.
The only measured headroom is a register-pressure retune bounded near
1.4x on a kernel that is about a tenth of the four-device 2048-class
wall; it is recorded, not scheduled.  And Greg's closing note applies
to this whole family of items: a performance investigation closed
here can rise again after further refactoring.
*(open_items E4; greg_notes.md item 6 and the addendum;
current_plans.md item 13; multigpu_findings.md §1.14, §1.24.)*

**B4. CLOSED 2026-08-17, by ruling.  Time translation and multi-axis on
the newer forward path.**  The column gather won everywhere measured
and shipped as mbirtorch commit 7cd32ed; the record is
multigpu_findings.md §1.18.

**B5. CLOSED 2026-08-17.  Remove the banded forward path, and rename
the survivor to cylinder transfer.**  →
`torch_port/closed/banded_forward_removal.md`.

**B6. CLOSED 2026-08-20, ruled and shipped.  The multi-axis geometry
ran slower on two GPUs at some problem sizes and faster at others.**
The full arc took two days: the component split found the mechanism,
the remedy landed in two forms, the confirmations flipped every
losing cell to a win, the full floors refresh cleared all four
torch-body sentinels, and Greg accepted the refreshed table the same
day.  The paste is in mbirtorch/_widening_floors.py with hand-written
notes, blessed hashes, and the two behavior tests re-pinned (staged);
automatic multiaxis reconstructions widen from the 512-class (1.5x at
two devices, above 2x at four) and translation from its measured
cells.  One recorded caveat: the 384-class was never probed for
admission, so a ladder extension could lower the multiaxis floors.
The detail below is the pre-closure record.  The
component split (mg44) attributed every losing cell to the back
projection running uncompiled: torch caps one function's compiled
variants at 8, the per-device compiled instances share that cap, and
the variants guard on the device index, so two devices filled it and
the remaining calls ran eagerly.  Translation shared the mechanism.
The remedy raises the cap to 64 in projectors.py, and it took two
forms because dynamo consults a per-thread view of the setting: a
raise made where the wrapper is created never reached the per-device
pool threads, so the shipped form raises it on the compiling thread
inside maybe_compile.  The gate and the large-cell confirmation flipped every losing cell
to a two-device WIN: multiaxis 512 from 0.35x to 1.53x, multiaxis
1024 from 0.80x to 1.52x, translation production from 0.89x to
1.25x, with the 768-class win reproduced at 1.45x.  The remedy
changed a shared floors cost input, so the FULL floors refresh runs
(mg48) rather than a family-scoped one; the sentinel rulings follow
its verdicts and are Greg's.  The paths beyond this remedy,
including the multiaxis and translation kernel question, are H8's
document.
*(multigpu_findings.md §1.36, §1.37; mg44_component_split.md;
multigpu_plan_part_2.md; the floor rows' notes in
mbirtorch/_widening_floors.py.)*

**B7. CLOSED 2026-08-19.  The parallel-beam forward projection was
much slower than mbirjax's, and a faster kernel now ships
(c761b24).**  The kernel sorts each view's pixels by detector
channel and read 3.45x on the production mixture of calls.  It
shipped as the default route.  The kernel owed two re-measures, and
both ran as one chained submission.  The first family-scoped
threshold refresh (mg40) re-measured the parallel thresholds on the
new kernel in 7.5 minutes, against about 3 hours for a full refresh.
The two-device threshold moved up one class, and E3's coarse table
landed in the same pass.  The mg27 re-run then re-anchored the
comparison tables and the user-docs timing table.  mbirtorch now
matches or beats mbirjax at every measured cell of both geometries.
The cone extensions of the sorted idea lost, and that line is closed
(multigpu_findings.md §1.32, §1.33; pfwd_segmented_design.md §9).
→ multigpu_findings.md §1.26 through §1.31 (the mechanism and the
kernel), §1.34 (the scoped refresh); execution_overview.md §5;
mg40_floors_scoped.md and mg27_reference_timings.md (the run
records).


## C. Measurement and calibration gaps

**C1. CLOSED 2026-08-18.  Re-anchor the cross-framework comparison.**
mg27 re-measured the full mbirtorch column on the padded tip; cone
took the padding's gain at two and four devices.  → execution_overview.md
§5.1 to §5.3; `experiments/torch_port/mg27_reference_timings.md`.

**C2. CLOSED 2026-08-17.  Translation and multi-axis have no
device-count thresholds of their own.**  Both families now have
measured rows; the record is multigpu_findings.md §1.22.

**C3. PARKED 2026-08-21. The scan preprocessing pipeline's concurrency is unmeasured on
GPUs.**  The multi-device path is correctness-gated only; whether it
actually runs faster has never been measured on a GPU node.  (The
denoiser half of the original item was measured on 2026-08-16 and is
closed: splitting never paid.)
*(open_items F5;
closed/preprocess_sharding_translation_multiaxis.md:171.)*

**C4. CLOSED 2026-08-17, measured.  The combining-step slab size was a
reasoned default, not a measured one.**  The constant moved to 256 MiB;
the sweep is in multigpu_findings.md §1.20 and in the constant's
comment.

**C5. CLOSED 2026-08-19: the probe ran and the remainder is ruled.
The memory ledger's charges were owed a calibration pass.**  The
probe (mg42a, findings §1.35) attributed the over-reads (cone's back
batch charge; parallel's deliberate forward covers at small shards),
refuted the lead-device transient in a single reconstruction, and
found one under-read (parallel at one device, 0.935).  Greg ruled the
same day: the term changes are TABLED as not urgent, because the
preflight's 15 percent margin absorbs the under-read and the
over-reads cost only headroom; and the projection-entry preflight is
CLOSED WITHOUT ACTION, since no additional workloads join the ledger
-- an explicit device list remains the way to run a bare projection
at a scale one device cannot hold.  The revisit triggers: an
under-read past the margin, or a widening decision measured wrong.
→ ledger_calibration_design.md (the design and the ruling);
ledger_calibration_inputs.md (the six inputs and their verdicts);
multigpu_findings.md §1.35.  The one live residue is the nightly's
memory column, which is G4.

**C6. CLOSED 2026-08-19.  No nightly row exercised the automatic device
choice.**  Every measured row pins its device count, and a pin
bypasses the automatic path, so the choice a multi-GPU user gets by
default never ran on a schedule.  The torch writer now runs one
UNPINNED check per multi-GPU night: it settles a 512-class cone model
and asserts the realized count equals the widest count the shipped
floors admit, computed at run time from the table so a floors refresh
moves the expectation with it.  The check records its verdict, the
per-count reasons, any leaked pin, and the floors staleness note
under its own key in the run file (no collision with the pinned
rows), and a mismatch gates HARD, cold start included, because the
expectation needs no prior run.  Verified on four H100s (mg43, job
15388660): the settle chose two devices of four visible, as the
floors say.  Single-device and cpu nights record a skip.
*(open_items F6; multigpu_plan.md §0a item 8;
mg43_autochoice_gautschi.sbatch.)*


## D. Correctness and API-contract gaps

**D3. CLOSED 2026-08-19, by verification -- the defect no longer
exists.  A leftover-memory pattern in the forward driver.**  The
record described a pre-fix snapshot: the same commit that fixed the
back loop (mbirtorch a880d9c, 2026-08-10) added the release to the
banded forward too, and the banded forward was later removed whole
(bfbd286).  The current forward driver, the cylinder transfer,
releases each cylinder batch after its projection is issued
(``full_cyl = None`` in ``_sparse_forward_project_cylinders``), and
its only multi-batch residency is the deliberate transfer-ahead
overlap, which the memory ledger charges.  No code change was needed.
*(open_items K1; closed/backloop_attribution.md §5 item 3.)*

**D1. CLOSED 2026-08-21, staged.  `stitch_arrays` mishandles mixed
inputs.**  It now refuses what it silently mishandled: tensors on
more than one device raise an error naming the devices, and a
divided-form input raises an error directing the caller to gather to
the host first.  A new test file pins the working forms' values and
covers both refusals.  *(API_specification.md Issue 2, marked fixed;
tests/test_utilities.py.)*

**D5. CLOSED 2026-08-21, staged.  Run-logging defects inherited from
the original package.**  All four defects are fixed.  Each model
instance now has its own logger, named by the class plus a
never-reused counter, and the constructor shares it with the runs.
The log file handler closes when a run finishes and reopens in append
mode when a later call continues the run, so the half-log merges
delete closed files.  The file handler is created with delay=True, so
a verbose=0 run creates no empty file while a warning would still
land.  The denoiser's matching close landed in the same pass.
test_logging grew the two-instance, closed-handle, and no-empty-file
cases and passes 18.  One recorded residue: per-instance loggers
persist in Python's logger registry for the life of the process, a
few KB per model.

**D6. CLOSED 2026-08-21, staged.  Translation demo data generates an
all-zero phantom.**  `generate_demo_data(model_type='translation')`
now builds its phantom with `gen_translation_phantom` (the dot
pattern), so the demo projects real content where it projected zeros,
and the test asserts nonzero content like its cube sibling.  One
note: the dot builder seeds numpy's global random generator, which is
pre-existing behavior of that builder.

**D2. CLOSED 2026-08-21, staged.  The array-forms rule is not applied
everywhere.**  The six streaming preprocessing entries now refuse the
divided form at entry, with a message naming the function, the
offending argument, and the fix, and their docstrings state the
accepted forms.  The check lives beside the batching driver it
protects (`pipeline.reject_shards`).  The `denoise` docstring gained
the divided form the code already accepted, which resolves the
classification conflict toward the code.  One adjacent defect was
found and fixed in the same pass: a tensor blank or dark scan crashed
the transmission computation through numpy's mean, and the three
entries that take reference scans now bring them to host numpy at
entry.  The entry-contract tests run in the default suite
(tests/test_preprocess_entry_forms.py, 15 passing).
*(open_items K3; closed/entry_point_survey.md rows E6, D8.)*

**D9. CLOSED 2026-08-21, staged.  A sharded sinogram cannot be passed
to `recon` or `prox_map`.**  The reconstruction entries now accept
the device form that `prepare_sino_for_devices` returns, for the
sinogram and the weights both, so a Plug-and-Play loop pays the
host-to-device transfer once instead of on every call.  The repair
follows the denoiser's template: `subsample_views` assembles the
statistics subsample from the shards exactly, `initialize_recon`
validates per shard on the shard's own device, and `vcd_recon`'s
entry checks the per-shard shapes the way the prox input's branch
does.  The two paths compute identical regularization parameters,
and the reconstruction difference sits at the run-to-run floor,
forty times under the file's gate.  Twenty-two tests landed with it
(test_device_policy passes 108).  The docstring and multi-GPU-page
promises this item flagged are now true.
*(multigpu_plan_part_2.md, the denoiser and plug-and-play section.)*

**D4. CLOSED 2026-08-21, staged.  Re-sweep for missed multi-GPU
support.**  The sweep enumerated every public entry that takes array
data and handed each one a real two-shard CPU `Shards`.  Over thirty
entries already carried the form end to end.  Fifteen broke below
the entry with misleading errors.  One of those,
`get_voxels_at_indices`, gained a real per-shard branch with a
bitwise gate.  The other fourteen gained the shared entry refusal,
which now lives in `_sharding.reject_shards` with the preprocessing
pipeline aliasing it.  Review added the divided `init_recon` case to
cone's `recon_split_sino` refusal.  The candidates for real support
later are recorded with their design questions.  The suite passes at
669 with the new gates.
*(open_items J5;
closed/preprocess_sharding_translation_multiaxis.md:209;
multigpu_findings.md §1.41.)*

**D8. Other items.**

  * **Issue 3: FIXED 2026-08-21, staged.  The saved-file format tag
names the old package.**  New files carry
`'mbirtorch_preprocessing_v1'`; the loader accepts both tags and
files with no tag, and rejects an unknown tag by name.  The stored
golden keeps its old tag and loads through the accept-both path, so
no regeneration was needed.

  * **Issue 4: CLOSED.  the FBP theory derivation has no pointer.**
The docstring linked to the derivation on the old package's
documentation site; the link was removed with the other old-package
references.  The link is now at the top of `theory.rst`.

  * **Issue 6: FIXED 2026-08-21, staged.  Two helpers can refuse a
problem they could handle.**  `prepare_sino_for_devices` and
`compute_hessian_diagonal` now pass the direct workload to the device
policy, as Greg ruled.  The layout is still sized for a full
reconstruction when one fits.  On a problem too large for any full
reconstruction, the check falls back to the helper's own footprint,
and a later full `recon` on such a layout re-runs the memory check
rather than reusing it; both behaviors are pinned by tests.  One
recorded residue: a standalone `compute_hessian_diagonal` with
default indices back-projects the unmasked grid, which the direct
plan under-prices on that fallback path.  Measured against the direct
plan's per-device peak, the dense default form needs 0.85x to 1.29x
across the 128- to 1024-class at one to four devices, while the
masked form is covered at 0.93x to 1.00x.  The exposure is narrow
(a direct user call, default indices, at beyond-full-recon scale),
and the candidate repair is a ledger term for the unmasked back
projection.

  * **Issue 7: FIXED 2026-08-21, staged.  Reconstruction method
names.**  The methods are renamed for consistency with `recon`:
`recon_direct`, `recon_split_sino`, `recon_fbp`, `recon_fdk`, and the
iteration engine is private as `_vcd_recon`.  Clean renames, no
aliases, since the package has not had a release: 160 replacements
across 33 files, zero old names remaining, dispatch-by-identity sites
and log labels included.  External material that teaches the old
names (mbirjax_applications scripts) needs its own pass.

## E. Decisions waiting on a person

**E1. CLOSED 2026-08-18, by ruling.  The remedy for compiled
cross-count value differences.**  Greg accepted the status quo: the
6e-4 differences at uneven splits stay, and comparisons across counts
gate at 1e-3.  The record is multigpu_findings.md §1.16.

**E2. PARKED 2026-08-19, with a revisit trigger.  The floors-refresh
automation questions.**  The plan for moving threshold re-measurement
into the nightly is drafted with seven open questions.  With E3 ruled
coarser and family-scoped, the automation case weakened -- a coarse
table drifts less -- so Greg parked the item.  The revisit trigger: a
family-scoped refresh measures drift in the coarse table.
*(open_items G2 and I3; active/floors_refresh_automation.md §7;
multigpu_plan.md §0a item 8.)*

**E3. CLOSED 2026-08-19, ruled and implemented.  Whether and when to
simplify the device-count thresholds.**  Coarser table; family-scoped
refreshes; cone n=4 joins the shared row; multiaxis n=2 is a
sentinel until B6's mechanism is known.  The implementation rode
B7's refresh and landed the same day (mg40, findings §1.34).
→ floors_coarsening_proposal.md §5 (the ruling and the landing).

**E4. CLOSED 2026-08-19, confirmed.  The nightly cadence for
multi-GPU rows.**  Fire-on-change at the measured price: about 1.6
GPU-hours on a night a tracked branch moved, seconds on a quiet one.
→ multigpu_plan.md item 7 and §0a.

**E5. CLOSED 2026-08-19, by ruling.  Torch version-advance handling.**
Merge floor advances as they arrive; each pull request is small,
mechanical, and CI-gated.  → closed/python_matrix_nightly_check.md §6.

**E6. CLOSED 2026-08-18, by ruling.  Where demos live in the docs.**
The demos-and-FAQs page keeps its structure; Greg confirmed it reads
as before and closed the question.

**E7. CLOSED 2026-08-18.  A written Python-version increment policy.**
The policy is written and accepted: mbirtorch tests the Python
versions torch supports and advances its torch floor deliberately
(`closed/python_matrix_nightly_check.md` §2, pointed to by
`closed/release_workflow.md` decision 3).  What remains of version
handling is E5's cadence question, which is Charlie's.


## F. Documentation and demos

**F1. CLOSED 2026-08-18, as a staged edit.  The multi-GPU page's
timing table is stale.**  The table took mg27's numbers and lost the
old 94 s row; B7's close refreshed it again on 2026-08-19 with the
sorted-kernel re-anchor (21.3, 14.2, 10.8 s at the 1024 class), and
the page's floors paragraph now describes the coarse table.  Both
edits are committed (mbirtorch 88abb4c).
*(open_items H2; multigpu_plan.md item 7; closed/docs.md:12.)*

**F2. CLOSED 2026-08-19.  Two demo-side documentation pieces.**
Charlie finished both.  → closed/demo_consolidation.md:229, :251.

## G. Release, nightly, and automation

**G4. CLOSED 2026-08-21.  The nightly's inflated memory readings
were the recompile budget, not the instrument.**  The first two
nights on the fixed writer answered the open question.  The inflated
readings recur in warm trials, so they were never a warmup effect,
and they vanish exactly when the recompile-budget remedy enters the
measured tip.
The late-running 1024-class one- and two-device arms had been
falling back to eager execution after the compiled-variant cap
filled, and eager execution's materialized intermediates were the
extra 0.6 to 3.2 GB.  The record book ratcheted down, both nights
gated PASS, and a future eager fallback now trips the HARD memory
gate.  One correction rides the closure: the 26.6 GiB reading was
always the ONE-device 1024-class row on a four-GPU night, not a
four-device arm's row.  → multigpu_findings.md §1.40.

## H. Scheduled and back-burner work

**H2. MOVED TO D11 2026-08-21: A functional recon interface.**  Provide a functional, not
object-oriented, interface for basic parallel-beam and cone-beam
reconstruction.
*(functional_interface_proposal.md item 10.)*

**H4. CLOSED 2026-08-19: Charlie's queue.**  Three recorded items: the Lilly comparison run
of the original package for speed and memory (run informally - happy with results); 
the plans-fork pull request to Greg (stale); and telling Greg about
`split_sino_recon` device handling (superseded by D8).  The multi-device completion
checklist those items came from is otherwise closed.
*(open_items J4;
closed/preprocess_sharding_translation_multiaxis.md section C.)*

**H7. CLOSED IN FAVOR OF H2 2026-08-19: LEAP and SVMBIR interfaces.**  Lower the
transition barrier for users of LEAP and SVMBIR.  LEAP presents as a
PyTorch front end, so the wrapper is thinner on mbirtorch than it
would have been on jax.  When picked up: run their examples, scope a
translation, then design and validate the interfaces.
*(current_plans.md item 10; port_plan.md §1.)*

## I. Accepted gaps and monitor-only notes

* **Cone 1024 above one device is uncovered in the cross-framework
  value comparison.**  Accepted 2026-08-10; a full re-run prices at
  eleven GPU-hours.  *(open_items F4; multigpu_findings.md §4.)*
* **The mg13 job's four-device runs never happened**, and the
  conclusion stood on the one-device runs by design.  *(open_items J2;
  multigpu_findings.md §1.14.)*
* **The truncation warning fires even after its own fix is applied**,
  because it reads the sinogram alone.  Noted for Greg, unscheduled.
  *(open_items K2; closed/demo_consolidation.md.)*
* **The jax rounding-bug precondition** stays monitor-only at the six
  per-slice rounding sites.  *(current_plans.md item 11.)*
* **Two documentation defects found upstream** in the original package;
  the copies here are corrected, and upstream is winding down.
  *(open_items H4.)*

## J. New features and improvements

**J2. CLOSED 2026-08-21, staged.  A functional recon interface.**
`recon_simple_parallel` and `recon_simple_cone` reconstruct in one
call, with weights, sharpness, and max_iterations as the only knobs;
they build the model internally and return `recon`'s (recon, dict)
pair.  Greg named the functions and their homes (parallel_beam.py and
cone_beam.py).  The proposal's filtered-back-projection claim was
verified before it was documented: max_iterations=0 returns the
direct reconstruction scaled to fit the data, bitwise equal to
recon_fbp or recon_fdk up to that scale, and the scale measurably
lowers the sinogram error.  Both functions are exported, documented
on their model pages ahead of the class, and covered by twelve
tests.  The wider docs placement Greg approved is applied: the
quick-start page opens with the one-call example and shows the model
form second for control, the API overview gains a One-Call
Reconstruction section ahead of Geometry Models, and the README
carries the one-line example.  The docs build with no warnings.
*(functional_interface_proposal.md.)*

**J3. CLOSED 2026-08-21, staged.  Refactor the gather.**
`Shards.gather` now allocates the host array once and copies each
shard into its own slice of it.  Measured on CPU, the new gather is
3.3 to 3.6 times faster on last-axis sharding and 2.0 times on the
view axis.  The speed comes from torch's multithreaded copy; the
removed second full-size allocation is the memory half of the win.
Equivalence was checked byte-exactly over 832 CPU and 252 MPS
configurations, and two new tests pin the round trip and the
container-consistency errors.  Two findings ride the closure.  Metal
refuses copies into destinations off a 4-byte boundary, which only 1-
and 2-byte dtypes can hit; a per-shard host staging fallback covers
it.  And a mixed-dtype shard list, formerly promoted by concatenate,
would now cast to the first dtype; every construction site builds
uniform shards today, and a dtype check in the constructor is the
recorded candidate.  The denoise-scale improvement on GPU awaits a
cluster confirmation.
*(multigpu_plan_part_2.md, end of the denoiser section;
multigpu_findings.md §1.39; _sharding.py.)*

**J1. CLOSED 2026-08-21, staged.  Add a ParallelBeam version of
recon_split_sino.**  The parallel method splits the detector rows
into N overlapping parts rather than two halves, per Greg's
direction of the same day.  Row r is slice r in this geometry, so
the parts decouple exactly in the forward model and the overlap only
feeds the prior across each seam.  A `slices_per_part` argument sets
the kept slices per part, bounded below by twice `half_overlap` so
the seam overlaps of a part cannot collide.  The default asks the
device-policy memory ledger for the fewest parts whose largest part
fits.  That question is answered by a new read-only predicate,
`TomographyModel._fits_available_devices`, which prices a candidate
model the way `_apply_device_policy` would while settling nothing on
it.  One part means a plain `recon`.  The split matches `recon` at
NRMSE 0.032 to 0.038 on the gate phantom, gated at 0.1, with the
measured values kept in the test comments.  Ten tests landed, the
docs FAQ now teaches the method for both geometries, and each
geometry page renders its own override's docstring.  The preflight
memory-remedy text names the method for parallel as well as cone.
One test note: a bitwise repeat of a CPU reconstruction needs one
torch thread, because float32 reductions differ run to run at the
default thread count.
*(demos_and_faqs.rst; usr_parallel_beam_model.rst;
usr_cone_beam_model.rst; tests/test_split_sino.py.)*

## The current_plans.md migration (2026-08-19)

Greg retired `plans/current_plans.md` in favor of this file.  What
moved and where:

* Items 1, 2, 4 through 8, 14, and 15 were complete with their
  records named in place; the retired file is their audit copy and
  moves to the archive with H5's sweep.  (Item 15's body carried a
  stale charter paragraph; the sharded phantom builder is in the
  library with its devices policy.)
* Item 3, the multi-GPU campaign, closes with this compilation: its
  remaining steps all landed -- the forward remedy as B7's committed
  kernel, the capacity table as A1, the floors simplification as
  E3's ruling, the cadence as E4's confirmation -- and its residue
  is B7's two re-measures.
* Item 9 is H1.  Item 11's open bullets are D5, H5, H6, and the
  monitor notes, with its forward-pixel-batch bullet subsumed by
  A4's measured default; its jax-side notes retire with mbirjax's
  wind-down.  Item 12's design notes moved verbatim to
  `torch_port/active/multires_direction.md`, which H3 cites.  Item
  10 is the new H7.
* Item 13, the sorted-stream parallel forward, is superseded: B7's
  committed kernel is the sorted design in its in-kernel form, and
  the stream-caching half survives as the design note's recorded
  memoization follow-up (pfwd_segmented_design.md §2).

`plans/torch_port/active/functional_interface_proposal.md` is its
own open proposal, awaiting review; H2 is its entry in this list.
The `closed/` records carry decisions with revisit triggers rather
than open items.  Items that closed before 2026-08-16 are audited in
`plans/open_items_v2.md` section J.
