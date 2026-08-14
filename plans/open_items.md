# Open items — collected from the torch-port active records

**Compiled 2026-08-11** by reading every file in
`plans/torch_port/active/`.  Each item names where it is recorded.  Paths
are relative to this repository unless they name `mbirtorch`.

This file collects open items.  It does not rank the project's work, and it
does not replace `current_plans.md`, which carries the agreed priorities and
the completed history.  Where an item already appears there, the entry says
so.

One caution about currency.  A concurrent session landed commits `5df9c6f`,
`fbb7c9f`, and `2465830` during the compilation, adding the mg13 and mg14
records, `floors_refresh_automation.md`, and the increment-6 closure.  Those
are read in.  The port is moving daily, so an item here can close between
compilations.

## Start here: the eight items with the most at stake

1. **The widening speed floors are stale on `greg_dev` right now.**  This one
   is measured, not inferred, and it is cheap to clear.  See [A1](#a1).
2. **Five device-policy design questions are waiting on a ruling**, and they
   block the largest coherent piece of remaining work.  See
   [B1](#b1).
3. **The FDK device-policy fix is on `prerelease` and not on `greg_dev`**,
   and it consults a rule its own commit message says may be wrong.  See
   [B2](#b2).
4. **The 2K production-scale work has not started.**  Production runs at 2K
   and above, and every measurement so far is at 1K.  See [C1](#c1).
5. **Translation and multiaxis have never been timed on more than one
   device**, and they inherit parallel's floors and parallel's forward
   shape.  See [D1](#d1).
6. **The cone back projection rises at two devices, and nobody owns it.**
   See [E1](#e1).
7. **The mbirtorch one-device anchors have not been re-measured since the
   column-gather flip**, and every scaling ratio divides by them.  See
   [F1](#f1).
8. **Gautschi's `/scratch` was returning write errors on 2026-08-11**, which
   affects every cluster item above.  See [J1](#j1).

## Contents

- [A. Measurement debt that is live now](#a-measurement-debt-that-is-live-now)
- [B. Device policy across entry points](#b-device-policy-across-entry-points)
- [C. Production scale, 2K and beyond](#c-production-scale-2k-and-beyond)
- [D. Geometry coverage](#d-geometry-coverage)
- [E. Performance questions with no owner](#e-performance-questions-with-no-owner)
- [F. Measurement gaps](#f-measurement-gaps)
- [G. Decisions waiting on a person](#g-decisions-waiting-on-a-person)
- [H. Documentation](#h-documentation)
- [I. Release, nightly, and automation](#i-release-nightly-and-automation)
- [J. Operational and housekeeping](#j-operational-and-housekeeping)
- [K. Defects and roughness carried deliberately](#k-defects-and-roughness-carried-deliberately)

---

## A. Measurement debt that is live now

### A1

**The widening speed floors are stale.**  `stale_note()` on the current
`greg_dev` tree names three drifted cost inputs:
`TomographyModel._sparse_forward_project_columns`, `_sharding.py`, and
`projectors.py`.  The floors were measured against commit `4a222c7`, and the
copy-stream and transient-memory commits have moved that code since.  The
guard is therefore governing the automatic device count with numbers measured
against code that has changed.

The consequence is bounded and the remedy is known.  A stale table still runs,
by design, and the note warns on every automatic selection.  Clearing it is one
run of `mbirtorch/dev_scripts/refresh_widening_floors.py` on a four-GPU node,
about 33 minutes, followed by pasting its block.

Recorded at `mbirtorch/mbirtorch/_widening_floors.py` (the MAINTENANCE
paragraph of the module docstring) and
`plans/torch_port/active/multigpu_findings.md` §1.12.

### A2

**Increment 9 of the forward remedy reads half complete, and its refresh has
run.**  `forward_remedy_design.md` §9 says the floors refresh "waits for the
flip's commit," and `multigpu_findings.md` §1.12 reports that refresh
completed on `4a222c7`.  The design note's status header carries the same
stale sentence.  Increment 6 and increment 11 were both marked complete on
2026-08-11, so increment 9 is now the only one that reads incomplete.  Only
the record is out of date, not the work.  A one-line correction closes it.

Recorded at `plans/torch_port/active/forward_remedy_design.md:8` and
`:770`, against `plans/torch_port/active/multigpu_findings.md` §1.12.

## B. Device policy across entry points

This section is one body of work.  Greg ruled on 2026-08-10 that every
public-facing entry that starts sharding takes the same device policy.  The
ruling is at `plans/torch_port/active/multigpu_plan.md:172`.  Implementation
waits on a systematic list, and that list is
`plans/torch_port/active/entry_point_survey.md`, which tables 57 entry points.

### B1

**Five design questions need a ruling before any code moves.**  They are
stated in full at
`plans/torch_port/active/entry_point_survey.md:158` (§4).

- **D1.** Does an entry that runs before any model exists consult a policy at
  all, and if not, must it still honor the pins?
- **D2.** When an entry inherits a layout from a previous reconstruction on
  the same model, may it re-choose?
- **D3.** Should `QGGMRFDenoiser.denoise` take the policy call now that its
  loop can run sharded?
- **D4.** Do the recon-calibrated floors and the full-recon ledger apply to
  workloads that are not VCD reconstructions?
- **D5.** Do `generate_demo_data`'s `devices=` argument and the vcls sibling's
  pin keep their pins with a recorded reason, or become policy consumers?

D4 is the one that governs the others.  The floors are measured on
3-iteration VCD reconstructions, and the ledger prices a full VCD plan.
Applying both to a filtered back projection over-prices its memory and decides
its device count on a crossover measured for a different curve.

### B2

**The FDK fix is unmerged, and it uses the rule D4 is about.**  mbirtorch
commit `72208bb` on `prerelease` makes `fdk_recon` call
`_apply_device_policy` before its first placement.  It is one commit ahead of
`greg_dev` and is not in the tree today.  Its own note says it consults the
VCD-calibrated floors and invites a different rule.

The fix is load-bearing.  Without it, full-resolution MAR fails on GPU 0 down
every entry path, because the initial FDK builds an 18 GB volume on one
device.  With it, job 15185020 completed in 49 minutes at 37 GB per device.

The other geometries' direct reconstructions carry the same gap, untouched.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:81`
and `:103`.

### B3

**`scan_to_sino` chooses a device count by its own rule.**  It takes all
visible CUDA devices, capped by `MBIRTORCH_NUM_DEVICES`, with no memory
preflight and no speed floors.  It is the second device-choosing site in the
package.  Its unfused siblings, `compute_sino_transmission`,
`correct_det_rotation`, and `downsample_view_data`, stay single-device, so the
fused path and the unfused path now differ with no stated reason.

Recorded at `plans/torch_port/active/entry_point_survey.md` rows D2 and D3,
and in the code at `mbirtorch/mbirtorch/preprocess/utilities.py:436`.

### B4

**A count change invalidates sharded data that a caller still holds.**  There
is no re-shard path.  `Shards` are bound to their placement by object
identity, and the entry placement raises on a mismatch.  Four holders are
exposed: `prepare_sino_for_devices`, a precomputed `fm_hessian`, an
`output_sharded=True` reconstruction, and a beam-hardening pass's
`init_recon`.

The sharpest case is `recon_plastic_metal`, where the policy re-runs on every
beam-hardening pass while the previous pass's reconstruction is fed back as
`init_recon`.  This is design question D2's subject.

Recorded at `plans/torch_port/active/entry_point_survey.md` rows A9, D7, and
§4 D2, and described at
`plans/torch_port/active/execution_overview.md` §2.3.

### B5

**Two entries place full-size arrays before any layout is chosen.**
`gen_weights` places a full sinogram and a full weights array on the lead
device, and `gen_weights_mar` runs a full forward projection, both typically
before the first reconstruction.  At production scale each doubles the lead
device's peak.  `estimate_sino_view_offset` and `align_sino_views` do the same
through `direct_recon` and `forward_project`.

Recorded at `plans/torch_port/active/entry_point_survey.md` rows F1, F2, D9.

### B6

**Two utilities do not recognize the sharded form.**  `stitch_arrays` selects
its device from the first tensor in the list and silently migrates the others
onto it, and it does not recognize `Shards`.  `apply_cylindrical_mask` and
`interpolate_defective_pixels` fail an `isinstance(recon, torch.Tensor)` check
on a `Shards` volume.  Neither is a policy question.  Both are gaps in the
array-forms rule that `array_forms_rule.md` states.

Recorded at `plans/torch_port/active/entry_point_survey.md` rows E6 and D8.

## C. Production scale, 2K and beyond

### C1

**The 2K design work has not started, and it waits on nothing.**  This is step
6 of the campaign plan.  Charter C's capacity table computes from the
corrected ledger charges, costs no cluster time, and picks the first tiling
leg.

The scale arithmetic is what makes this the feasibility question rather than a
tuning question.  At (2048, 2016, 1984) the recon is 31.7 GB, the sinogram
32.8 GB, and one full-pixel cylinder 24.9 GB.  Single-device VCD is impossible
there.

Recorded at `plans/torch_port/active/multigpu_plan.md:137` (step 6) and
`plans/torch_port/active/multigpu_findings.md:1477` (§6.4).

### C2

**The band-reduce term is the structural memory ceiling, and it is flat in
device count.**  Each slice-owner holds about 1.5 cylinders, which is 37 GB at
2K, and adding devices does not shrink it.  Restructuring that reduce is the
one identified change that would let per-device memory scale down with the
device count.

Greg's own note ranks this as one of two picks, beside the transfer overlap
that has since landed.  The seam restructure is the same subject from the
campaign's side, and its memory premise was measured-met at 1K on 2026-08-10.
Greg ruled on that date to hold it to 2K.

Recorded at `plans/torch_port/active/greg_notes.md:21` and
`plans/torch_port/active/multigpu_findings.md:1503` (§6.5 addendum).

### C3

**The forward pixel batch has never been swept at production scale.**
`FORWARD_PIXEL_BATCH` stays at 8192 deliberately.  Composed wall kept
improving through 16384 and 32768 at the 1024-class cell, by 4 to 15 percent,
so the knee is not bracketed.  The batch's cross-device transient grows with
the slice axis, so the 2K sweep is the one that should set the default.

Recorded at `plans/torch_port/active/multigpu_findings.md` §1.10 and
`plans/torch_port/active/greg_notes.md:11`.

### C4

**The memory ledger has never been validated at 2K.**  Every calibration ran
at the 1024 class.  The first composed 2K runs both check the ledger where it
matters and anchor the tiling work.

Recorded at `plans/torch_port/active/greg_notes.md:22`.

## D. Geometry coverage

### D1

**Translation and multiaxis have no floors of their own.**  Neither declares
a `_floor_family`, so the parallel floors govern their automatic device count.
The parallel floors are the more permissive measured set.  Neither geometry
has been timed at more than one device.

Recorded at `plans/torch_port/active/multigpu_plan.md:71` (step 1's second
obligation) and `plans/torch_port/active/multigpu_findings.md` §1.12.

### D2

**Translation and multiaxis still run the banded forward.**  They share cone's
banded branch and its band-independent per-call cost, so the column gather
should help them.  Neither has been timed on it, and the recorded rule is that
a geometry switches only on its own measurement.  One two-arm job per geometry
would settle it.

Recorded at `plans/torch_port/active/greg_notes.md:10` and in the code at
`mbirtorch/mbirtorch/tomography_model.py` (`_column_gather_forward`).

### D3

**`generate_demo_data(model_type='translation')` returns an all-zero
phantom**, in mbirtorch and in mbirjax both.  Translation's two-row recon
volume breaks the generic phantom builders.  The port reproduces mbirjax
faithfully.  Charlie wants it fixed in mbirtorch before release.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:240`
and `plans/torch_port/active/demo_consolidation.md:225`.

### D4

**The scatter-versus-gather loop organization is unexamined across
geometries.**  The multiaxis port carried over mbirjax's forward vertical fan
as a scatter.  Charlie's hypothesis is gather vertical, scatter horizontal,
testable one axis at a time against the goldens.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:273`.

## E. Performance questions with no owner

### E1

**The cone back projection rises at two devices, and nothing addresses it.**
Its device span read 30.33 s at two devices beside the forward's 30.65 s.  No
forward remedy touches it, and the campaign records it as a separate decision
that was never made.

Recorded at `plans/torch_port/active/multigpu_findings.md` §1.7 and §6.2, and
ranked at `plans/torch_port/active/greg_notes.md:17`.

### E2

**The parallel kernel's factor of two is unexplained, and its discriminating
arm is cheap and unrun.**  A launch at width 504 costs the same 41.4 ms as a
launch at width 1008, so a narrow call wastes half its cost.  The mechanism is
unknown, and occupancy is the leading candidate.

The arm is parallel at one device with `forward_project_slice_band` set to
504.  A per-slice reading near 0.041 ms would make the effect a multi-device
one, and a reading near 0.082 ms would make it a kernel width effect.  No
adopted decision depends on the answer.  Understanding it is the entry point
for making the kernels faster, including the single-device case every
configuration pays.

Recorded at `plans/torch_port/active/forward_remedy_design.md:890` (§12,
question 1) and ranked at `plans/torch_port/active/greg_notes.md:15`.

### E3

**Two-dimensional tiling and the cache directions are proposed and
unmeasured.**  Blocking both axes of a call's working set so it stays resident
in L2 is the next claim after the column gather's one-dimensional tiling.  The
1K measurements mostly refuted L2-residency effects at that size.  The honest
sequence is the 2K baseline runs first, and tiling is the main lever if those
runs show cache-boundedness or a memory squeeze.  Phase-blocked accumulation is
the entry-level version of the same idea.

Recorded at `plans/torch_port/active/greg_notes.md:34`.

### E4

**The sorted-stream question is closed at the torch level and open inside the
kernel.**  The mg13 probe measured the light per-call sorted form winning
1.17x to 1.32x in eager mode over the torch scatter, losing under
`torch.compile` to graph breaks, and losing to the production Triton kernel by
about 3.5x either way.  The item-13 stop stands, with its rationale sharpened.
What remains is sorted or segmented accumulation on-chip, which belongs to the
kernel campaign.

Recorded at `plans/torch_port/active/multigpu_findings.md` §1.14.

## F. Measurement gaps

### F1

**The mbirtorch one-device anchors predate the column-gather flip.**  Every
scaling ratio in the current comparison divides by them.  The code argument
says a trivial placement never enters the gather, and that argument has not
been checked against the current tip.  One arm per geometry answers it.

Recorded at `plans/torch_port/active/execution_overview.md` §5.3 and §7.

### F2

**The 512-class cell has no post-flip measurement.**  Its whole mbirtorch
column is the campaign ruler of 2026-08-09.  That cell is where both floor
families place their two-device admission, so the floors were refreshed
against it while the comparison table was not.

Recorded at `plans/torch_port/active/execution_overview.md` §5.2.

### F3

**The band-reduce slab is a reasoned default, not a measured knee.**  64 MiB
was chosen by an argument about launch overhead against transfer cost.  No
sweep has run.

Recorded at `mbirtorch/mbirtorch/_sharding.py` (`REDUCE_SLAB_BYTES`) and
`plans/torch_port/active/execution_overview.md` §7.

### F4

**Cone 1024 above one device is uncovered in the cross-framework value
comparison.**  The gap is provisionally accepted, re-examined on 2026-08-10,
and confirmed.  A full comparable re-run prices at eleven GPU-hours and a
partial one at about 2.5.

Recorded at `plans/torch_port/active/multigpu_plan.md:39` and
`plans/torch_port/active/multigpu_findings.md` §4.

### F5

**Three ported multi-device paths are correctness-gated only.**  The scan
preprocessing pipeline's concurrency win is unmeasured until a Gautschi run.
The sharded denoiser's GPU concurrency is likewise unmeasured, its gate having
run on CPU shards.  Both carry the caveat at their own checklist entries.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:171`
and `:182`.

### F6

**No nightly row exercises the widening guard.**  Every row pins its count
through `MBIRTORCH_NUM_DEVICES`, and a pin bypasses the guard by construction.
One unpinned auto-assert row was proposed to Greg and is unanswered.  The
chosen-count tests in `tests/test_device_policy.py` are the only place the
ordering rule runs end to end.

Recorded at `plans/torch_port/active/multigpu_plan.md:71` and in the code at
`mbirtorch/mbirtorch/_widening_floors.py` (the module docstring).

### F7

**The back-batch memory charge is about 0.8 GB conservative at parallel 1024
with two devices.**  The cost model counts four slabs where only three are live
at the launch instant.  This is the residual behind the model's 5.4 percent
over-read at that cell.

Recorded at `plans/torch_port/active/backloop_attribution.md:204` (§5, item 2).

## G. Decisions waiting on a person

### G1

**The five device-policy questions.**  See [B1](#b1).  These are the ones with
the most downstream work behind them.

### G2

**Seven questions on automating the floors refresh.**  The plan is a draft of
2026-08-11 and asks the checkpoint to rule on: which branch to watch; whether
the refresh script gains a `--write` mode; whether one open proposal at a time
is rate limit enough; whether the artifact travelling through
`mbirjax_metrics` widens the trust boundary; who owns the mail when the
measuring job fails; whether a refresh is forced on a schedule independent of
drift; and how this interacts with simplifying the floors.

Recorded at `plans/torch_port/active/floors_refresh_automation.md:518` (§7).

### G3

**Whether to simplify the floors, and when.**  Step 9 of the campaign plan.
The floors are measured at specific shapes, one GPU model, and one run
configuration, and Greg recorded that precision as fragile.  Fewer and coarser
thresholds that survive shape and hardware variation are preferred to exact
crossovers that do not.  This interacts with G2: a coarser table would drift
less often and would need the automation less.

Recorded at `plans/torch_port/active/multigpu_plan.md:164`.

### G4

**Whether torch floor advances merge on arrival or batch yearly.**  For
Charlie.  The dependency watch behaves identically either way.  The answer
only sets how long an eager torch pull request may sit closed.

Recorded at `plans/torch_port/active/python_matrix_nightly_check.md:356`
(§6).

### G5

**Where demos live in the docs.**  The one demo question the critiques did not
settle: the current demos-and-FAQs page structure, or something simpler.

Recorded at `plans/torch_port/active/demo_consolidation.md:239` (§7,
question 5).

### G6

**A written Python-version increment policy.**  A proposed policy is on the
page for Greg's and Charlie's review.

Recorded at `plans/torch_port/active/release_workflow.md:331` (decision 3).

## H. Documentation

### H1

**`usr_multi_gpu.rst` does not say which entry points spread and which do
not.**  Two passages read as though `fbp_recon` and `fdk_recon` participate in
the automatic spread, and they inherit instead.  The sharper of the two
recommends `prepare_sino_for_devices` for repeated reconstructions without the
ordering caveat the method's own docstring carries.  The page also does not
say that preprocessing chooses its own count by a different rule.

Recorded at `plans/torch_port/active/execution_overview.md` §7.

### H2

**The `usr_multi_gpu.rst` timing table is stale.**  It comes from the phase-4
gate matrix, and the mbirtorch numbers have moved substantially since.  It is
the only place a user sees those numbers.  This is a close-out item of the
campaign and is already carried as a rider on `current_plans.md` item 5.

Recorded at `plans/torch_port/active/multigpu_plan.md:141` (step 7) and
`plans/torch_port/active/docs.md:12`.

### H3

**The performance dashboard page stays held at Greg's request.**  The
`PENDING` markers in `index.rst` say where it reconnects.  Carried on
`current_plans.md` item 5.

Recorded at `plans/torch_port/active/docs.md:49`.

### H4

**Two upstream mbirjax documentation defects were found by the port.**
`usr_translation_model.rst` still says translation has no direct
reconstruction, which is outdated since `fdk_recon` landed.  The mbirtorch
copy is corrected.  The second is recorded in the same entry.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:246`.

### H5

**Two demo-side documentation pieces are owed.**  An FAQ paragraph for the
reversed rotation direction, carrying the symptom, the cause, and the fix.
And the demo set itself, which is designed and not yet implemented, one demo at
a time with review as each lands.

Recorded at `plans/torch_port/active/demo_consolidation.md:229` and `:251`.

## I. Release, nightly, and automation

### I1

**Four release-workflow pieces remain.**  Steps 7 and 8 are the Read the Docs
stable default and `RTD_TOKEN`, and both matter only at the first tagged
release.  `release.yml` and the optional documentation preview are not
written.  The wheel-check developer script is not written.  The
Python-version increment policy is [G6](#g6).

Recorded at `plans/torch_port/active/release_workflow.md:26`.

### I2

**The dependency watch's watchdog line on the gautschi nightly is still
Greg's.**  The watch itself is live and armed.  The first quiet week's
watchdog lines are to be read once.

Recorded at `plans/torch_port/active/python_matrix_nightly_check.md:352`.

### I3

**The floors-refresh automation is a draft plan, not built.**  It is step 8 of
the campaign, and its seven open questions are [G2](#g2).

Recorded at `plans/torch_port/active/floors_refresh_automation.md` and
`plans/torch_port/active/multigpu_plan.md:154`.

### I4

**Campaign close-out has not started.**  Its parts: finalize the item-13 entry
gate record from the device-span share; the `usr_multi_gpu.rst` pass; update
`current_plans.md` item 3; run the refresh script once end to end as proof of
mechanism; and confirm or revisit the nightly's n>1 cadence from the refreshed
per-night cost.

Recorded at `plans/torch_port/active/multigpu_plan.md:141` (step 7).

## J. Operational and housekeeping

### J1

**Gautschi storage was failing on 2026-08-11.**  `/scratch` returned I/O
errors on writes, and `/home` was at its 25 GB quota.  The job environment,
working directory, and torch compile caches were moved temporarily to
`/depot/bouman/data/claude_runtime/`.  They move back when RCAC repairs
`/scratch`.  Anything on gautschi that writes to `/scratch` or `/home`,
including the nightly, may fail meanwhile.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:116`.

### J2

**The mg13 job's four-device arms never ran.**  A conda environment rebuild
during the job removed the interpreter its per-arm subprocesses resolve
through.  The one-device arms are the question by the harness's own design and
they completed, so the conclusion stands.  The four-device composed arms are
unmeasured.

Recorded at `plans/torch_port/active/multigpu_findings.md` §1.14.

### J3

**Two harness defects are recorded for the lineage.**  A job printed its
completion sentinel after a crash, because the sentinel line was
unconditional.  And an mg14 verdict line drew its conclusion from busy time
when the change it measured lived between the busy windows, so it printed NO
CHANGE against unambiguous readings.

Recorded at `plans/torch_port/active/multigpu_findings.md` §1.14 and §1.15.

### J4

**Three items are Charlie's.**  An mbirjax comparison run on Lilly for speed
and memory, with the script already on gautschi.  A plans-fork pull request to
Greg.  And telling Greg about split-sino device handling.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:281`
(section C).

### J5

**A re-sweep for sharding gaps is owed.**  The original sweep grepped mbirjax
docstrings for "shard", and that grep has a blind spot: jax's global arrays
make sharding transparent, so a sharded-capable function may never say
"shard".  The recorded method for the re-sweep is to trace downstream callees
from each entry point that now accepts `Shards`, looking for tensor-only
assumptions.

Recorded at
`plans/torch_port/active/preprocess_sharding_translation_multiaxis.md:209`.

## K. Defects and roughness carried deliberately

### K1

**The same evaluate-before-rebind pattern exists in the forward driver.**  The
back loop's version was fixed.  The forward's two-fan branch has the same
species of transient retention, and it was out of that charter's scope.

Recorded at `plans/torch_port/active/backloop_attribution.md:204` (§5,
item 3).

### K2

**The lateral field-of-view truncation warning fires on the sinogram alone.**
It therefore still appears after the user applies the fix it recommends, which
reads as "the fix did not work".  The warning could stay quiet when the
reconstruction region already exceeds the field of view.  Noted for Greg, not
scheduled.

Recorded at `plans/torch_port/active/demo_consolidation.md:229`.

### K3

**The array-forms rule is not universally applied.**  Amendment 1 requires a
function that does not implement the sharded case to say so in its docstring
AND raise at entry on multi-shard input.  The two utilities in [B6](#b6) do
neither.

Recorded at `plans/torch_port/active/array_forms_rule.md:27` and
`plans/torch_port/active/entry_point_survey.md` rows E6 and D8.

### K4

**Preprocessing docstrings went stale at the prerelease merge.**
`correct_zinger_pixels` and `BH_correction` become genuinely multi-device when
a caller passes a device list, and their docstrings still say single device.

Recorded at `plans/torch_port/active/entry_point_survey.md` row D4.

---

## What this file does not cover

Three classes of open work live elsewhere and are not repeated here.

`current_plans.md` carries the agreed priorities and several open items of its
own, including the MAR `H` caching direction, the LEAP and SVMBIR interfaces,
the run-logging inherited defects, the test-quality tail, and multi-resolution
reconstruction.

`plans/torch_port/closed/` carries decisions with revisit triggers rather than
open items.  The kernel-batching findings hold the triggers that would reopen
the sorted-stream question, and [E4](#e4) reports what the mg13 probe did to
them.

The `preprocessing.md` and `docs.md` work lists are substantially complete.
Their residue appears above as individual items rather than as whole sections.
