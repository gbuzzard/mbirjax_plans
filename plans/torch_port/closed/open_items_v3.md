# Open items — v3

**Compiled 2026-08-18 as v3; updated 2026-08-19, with B6, B7,
and C5 rewritten to the house style.**  This file supersedes
`plans/open_items_v2.md`, which was
compiled 2026-08-16 and carried status updates through 2026-08-18.  Each open item appears here once, rewritten to say what is
true now rather than how it got there, and keeping the source citations
from v2.  Each closed item keeps one headline sentence and a pointer to
its record.  Items that closed before v2 was compiled are not repeated
here; they are in v2's audit list, its section J.  The port moves daily,
so an item here can close between compilations.

One naming convention, used throughout: the "device-count thresholds"
are the measured problem sizes below which the automatic device choice
keeps a reconstruction on fewer GPUs (the code calls them widening
speed floors).

## Contents

- [Start here: the next session](#start-here-the-next-session)
- [A. Large-scale reconstruction (2048-class and larger)](#a-large-scale-reconstruction-2048-class-and-larger)
- [B. Speed investigations at current sizes](#b-speed-investigations-at-current-sizes)
- [C. Measurement and calibration gaps](#c-measurement-and-calibration-gaps)
- [D. Correctness and API-contract gaps](#d-correctness-and-api-contract-gaps)
- [E. Decisions waiting on a person](#e-decisions-waiting-on-a-person)
- [F. Documentation and demos](#f-documentation-and-demos)
- [G. Release, nightly, and automation](#g-release-nightly-and-automation)
- [H. Scheduled and back-burner work](#h-scheduled-and-back-burner-work)
- [I. Accepted gaps and monitor-only notes](#i-accepted-gaps-and-monitor-only-notes)

---

## Start here: the next session

Ordered 2026-08-18 under Greg's priority: parallel and cone
performance first.  The padding remedy is committed (64dedb8) and
confirmed everywhere it was read, and B1 and B2 are closed: the
padded-tip nightly's 1024-class two-device back fell 2.44x and now
beats one device (findings §1.23).

1. **DONE 2026-08-18.  The profile run on the cone back kernel.**
   mg25 ran and the profiles answered B3's precondition: the gathers
   are cache-absorbed, not transaction-bound, and registers cap
   occupancy (findings §1.24).  B3's disposition on those readings is
   a ruling request to Greg.  See
   [B3](#b-speed-investigations-at-current-sizes).
2. **DONE 2026-08-18, ruling pending.  Re-measure the device-count
   thresholds, and land the coarsening proposal.**  mg26 ran on the
   padded tree and its rows are pasted into
   `mbirtorch/_widening_floors.py` (staged; the floors test passes).
   Cone n=4's floor dropped to the 768-class, the rest reproduced,
   and multiaxis n=2 gained a measured loss above its floor
   (findings §1.25).  The coarsening proposal is complete with its
   §2 rows filled: `torch_port/active/floors_coarsening_proposal.md`,
   §5 is the ruling request.  See
   [E3](#e-decisions-waiting-on-a-person).
3. **DONE 2026-08-18.  Re-anchor the comparisons and the docs.**  mg27
   re-measured the full mbirtorch column (C1, closed), and the
   user-docs timing table carries the new numbers as a staged edit
   (F1).  See [C1](#c-measurement-and-calibration-gaps).
4. **DONE 2026-08-18.  The parallel forward profile variant (B7).**
   mg28 ran at Greg's direction and the verdict is unambiguous: the
   kernel is load-bound on the per-view reload of the values block,
   the atomic write path is within 1.5x of coalesced, and §1.19's
   "192x" was a pricing error now corrected (findings §1.26).  The
   candidate remedy is the kernel docstring's own view-loop
   respecialization; opening that increment is Greg's call.  See
   [B7](#b-speed-investigations-at-current-sizes).
5. **Riders as time allows.**  The ledger calibration pass (C5), the
   nightly's automatic-device row (C6), and B6's multiaxis probe,
   which Greg's priority ruling places behind the cone and parallel
   work.  See [C5](#c-measurement-and-calibration-gaps).

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
scales, and the speed case routed through the kernel questions that
are now measured and closed: the cone back has nothing for blocking
to fix (B3), and the cone forward's cache-blocking candidate was
declined on the profiles' arithmetic (DRAM at 13 percent of
bandwidth, findings §1.33).
*(open_items E3; greg_notes.md the closing addendum; findings §1.24,
§1.33.)*

## B. Speed investigations at current sizes

**B1. CLOSED 2026-08-18.  The cone back projection costs more on more
devices.**  The mechanism was the compiler's divisibility
specialization on the back kernel's band argument; the band-padding
remedy landed as mbirtorch 64dedb8 and confirmed everywhere it was
read.  The record is findings §1.21 (the mechanism) and §1.23 (the
remedy landing), with the design in
`torch_port/active/back_remedy_design.md`.

**B2. CLOSED 2026-08-18.  Pad the kernel width arguments to the next
multiple of 16.**  Landed and confirmed with B1's remedy; arbitrary
user shapes no longer pay the two-times penalty.  → findings §1.23.

**B3. CLOSED 2026-08-18, by ruling.  In-kernel sorted or segmented
accumulation.**  The profile run (mg25, findings §1.24) found nothing
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

**B6. OPEN, grown 2026-08-18.  The multi-axis geometry runs slower on
two GPUs at some problem sizes and faster at others, and no mechanism
explains it.**  Two threshold refreshes measured the same shape: two
devices win at the 384-class, lose badly at the 512-class, win at the
768-class, and lose again at the 1024-class.  The sizes where two
devices help are therefore a window, not a threshold, which the
device-count thresholds cannot express.  The current threshold admits
two devices from the 768-class up, and therefore into the 1024-class
slowdown.  The coarsening proposal (E3) recommends holding this
geometry to one device until the mechanism is known.  The proposed
first step is to break the slow 512-class two-device run into its
component calls, with the 1024-class loss as a second case.
Translation loses at every device count and may share the mechanism.
*(multigpu_findings.md §1.22, §1.25; the floor rows' notes in
mbirtorch/_widening_floors.py.)*

**B7. MOSTLY COMPLETE; two re-measures finish it, an early task for
the next session.  The parallel-beam forward projection was much
slower than mbirjax's, and a faster kernel now ships (c761b24).**  At the 1024 class the
forward took 28.9 s of a 40 s one-device reconstruction, more than
mbirjax spends on a whole reconstruction.  Profile runs put the cost
in the kernel's memory traffic.  The design that measured best sorts
each view's pixels by detector channel, accumulates a tile, and adds
once per channel.  That form reads 3.45x on the production mixture of
calls.  It is committed as the default route in `triton_parallel.py`,
and it passed the test suites and composed reconstructions at the
1024 and 2048 classes.  Parallel now matches or beats mbirjax at
every measured device count.  What remains is a threshold refresh
and a re-anchor of the comparison tables, which measured the old
kernel.  A cone extension of the same idea was tried and lost
(findings §1.32), as did the sorted-order rework that followed it
(findings §1.33); Greg closed that line 2026-08-19, and Section 9 of
`plans/torch_port/active/pfwd_segmented_design.md` carries the
measured verdicts.
*(mg27 through mg31 rows; findings §1.5, §1.19 corrected, §1.26
through §1.31; pfwd_segmented_design.md;
projector_kernels/gpu_headroom_findings.md; greg_notes.md item 6.)*

## C. Measurement and calibration gaps

**C1. CLOSED 2026-08-18.  Re-anchor the cross-framework comparison.**
mg27 re-measured the full mbirtorch column on the padded tip: both
geometries, both standard cells, all three counts, under mg1's
protocol.  Every one-device time reproduced its recorded value, and
the cone multi-device rows took the padding's gain, so cone is now
faster than the recorded mbirjax column at two and four devices.  The
record is execution_overview.md §5.1 to §5.3, with the run detail in
`experiments/torch_port/mg27_reference_timings.md`.
*(open_items F1 and F2; active/execution_overview.md §5.2, §5.3.)*

**C2. CLOSED 2026-08-17.  Translation and multi-axis have no
device-count thresholds of their own.**  Both families now have
measured rows; the record is multigpu_findings.md §1.22.

**C3. The scan preprocessing pipeline's concurrency is unmeasured on
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

**C5. OPEN, inputs accumulating for the next calibration pass.  The
memory ledger's back-projection charge is conservative, and four
further inputs are waiting for the same pass.**  The charge counts
four memory blocks where only three are live at once, about 0.8 GB
high at parallel 1024 on two devices.  Two later runs measured
further over-charges.  Two other runs found costs the ledger does not
price at all: one projection ran out of memory instead of widening or
being refused, and a multi-device run briefly peaks on its lead
device above what any ledger term names.  The five inputs are
collected, with their citations, in
`torch_port/active/ledger_calibration_inputs.md`.  The proposed next
step is one calibration pass that answers all five.
*(open_items F7; active/ledger_calibration_inputs.md.)*

**C6. No nightly row exercises the automatic device choice.**  Every
nightly row pins its device count, which bypasses the automatic path
entirely; one unpinned row asserting the automatic choice is proposed
and unanswered.
*(open_items F6; multigpu_plan.md §0a item 8.)*

## D. Correctness and API-contract gaps

**D1. `stitch_arrays` mishandles mixed inputs.**  It silently moves
tensors from different devices onto the first tensor's device, and it
fails without a clear message on the divided multi-GPU form.  The
proposal is to raise a named error in both cases, since its one
internal caller always passes host arrays.
*(open_items B6; API_specification.md Issue 2.)*

**D2. The array-forms rule is not applied everywhere.**  The rule
requires a function that does not implement the divided multi-GPU form
to say so in its docstring and to raise at entry.  The streaming
preprocessing entries (`scan_to_sino` and its siblings) do neither: a
divided input fails deep inside the pipeline, and a tensor input
returns numpy instead of mirroring its input form.  The rule document
also classifies the denoiser one way while the code implements the
other; one of the two should be amended.
*(open_items K3; array_forms_rule.md;
closed/entry_point_survey.md rows E6, D8.)*

**D3. A leftover-memory pattern in the forward driver.**  The forward
projection's two-fan branch keeps a large temporary alive across a
rebind, the same pattern that was fixed in the back-projection loop; it
was out of that fix's scope.
*(open_items K1; closed/backloop_attribution.md §5 item 3.)*

**D4. Re-sweep for missed multi-GPU support.**  The original survey
grepped docstrings for "shard", which misses functions whose multi-GPU
support is implicit.  The recorded method is to trace downstream
callees from each entry point that accepts the divided form, looking
for tensor-only assumptions.
*(open_items J5;
closed/preprocess_sharding_translation_multiaxis.md:209.)*

**D5. Run-logging defects inherited from the original package.**  Two
live models of one class write into each other's logs; the log file
handle is never closed, so lines written after `split_sino_recon`
merges and deletes its half-logs land in a deleted file; and
`verbose=0` still creates an empty log file.  Candidate repairs are
recorded: per-instance logger keys and a close call in the setup.
*(current_plans.md item 11.)*

**D6. Translation demo data generates an all-zero phantom.**  The
translation geometry's flat, two-row volume breaks the generic phantom
builders, in this package and the original both.  Charlie wants it
fixed here before release.
*(open_items D3;
closed/preprocess_sharding_translation_multiaxis.md:240.)*

**D7. The projection loop organization is unexamined across
geometries.**  The multi-axis port carried over the original's choice of
which loops scatter and which gather.  Charlie's hypothesis (gather on
the vertical axis, scatter on the horizontal) is testable one axis at a
time against the stored reference results.
*(open_items D4;
closed/preprocess_sharding_translation_multiaxis.md:273.)*

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

**E3. CLOSED 2026-08-19, fully ruled.  Whether and when to simplify
the device-count thresholds.**  Greg ruled for the coarser table
(2026-08-18), for family-scoped refreshes that measure only the
families whose cost inputs moved, and for the two judgment calls
(2026-08-19): cone at four devices joins the shared row despite its
1.145 reading against the 1.15 margin, and multiaxis at two devices
becomes a sentinel rather than a threshold until B6's mechanism is
known.  Implementing the coarse table rides B7's owed family-scoped
refresh.  The ruling is recorded in the proposal's §5.
*(open_items G3; multigpu_plan.md item 9 and §0a;
active/floors_coarsening_proposal.md §5.)*
*(open_items G3; multigpu_plan.md item 9 and §0a;
active/floors_coarsening_proposal.md.)*

**E4. CLOSED 2026-08-19, confirmed.  The nightly cadence for
multi-GPU rows.**  Greg confirmed fire-on-change at the measured
price: about 1.6 GPU-hours on a night a tracked branch moved,
seconds on a quiet night.  The campaign entry that would have been
updated lives in `current_plans.md`, which this file's successor
replaces (the migration is recorded in v4's header).
*(open_items I4; multigpu_plan.md item 7 and §0a.)*

**E5. CLOSED 2026-08-19, by ruling.  Torch version-advance handling
for the dependency watch.**  Greg decided: merge floor advances as
they arrive.  Each pull request is small, mechanical, and CI-gated,
and torch's roughly quarterly minors would leave a yearly batch
several behind.
*(open_items G4; closed/python_matrix_nightly_check.md §6.)*

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
timing table is stale.**  The table now carries mg27's parallel-beam
numbers (40.0, 23.8, 15.5 s at the 1024 class), names the geometry,
and the surrounding ratios are recomputed; the old 94 s row is gone.
The edit is staged in mbirtorch (`docs/source/usr_multi_gpu.rst`) for
Greg's commit.
*(open_items H2; multigpu_plan.md item 7; closed/docs.md:12.)*

**F2. CLOSED 2026-08-19.  Two demo-side documentation pieces.**
Charlie finished both (Greg's report closing the item).
*(open_items H5; closed/demo_consolidation.md:229, :251.)*

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

## H. Scheduled and back-burner work

**H1. MAR: cache the fitting matrix.**  Compute each column of the
metal-artifact fit's matrix once instead of recomputing it
quadratically often.  It is a cheap, self-contained speed win with no
statistical questions.  Subsampling stays deprioritized, with the
reasoning recorded.
*(current_plans.md item 9.)*

**H2. A functional recon interface.**  Provide a functional, not
object-oriented, interface for basic parallel-beam and cone-beam
reconstruction.
*(functional_interface_proposal.md item 10.)*

**H3. Multi-resolution reconstruction.**  A future direction, not for
the next release: reconstruct at binned resolution and upsample as the
initializer for the next-finer level.  The pilot design, the null
hypothesis to beat (a direct-reconstruction initializer), and the
matching problems are recorded in the item.
*(current_plans.md item 12.)*

**H4. Charlie's queue.**  Three recorded items: the Lilly comparison run
of the original package for speed and memory (script already on the
cluster); the plans-fork pull request to Greg; and telling Greg about
`split_sino_recon` device handling.  The multi-device completion
checklist those items came from is otherwise closed.
*(open_items J4;
closed/preprocess_sharding_translation_multiaxis.md section C.)*

**H5. Archive old plans.**  Move older plan documents and scripts to
archived storage; the plans tree accumulates.
*(current_plans.md item 11.)*

**H6. Test-suite quality and cost.**  The demo-data gates sit 300 to
640x above their measured noise floors; `test_logging` wants a
two-instance case; the cone fixtures could be module-scoped (72 s
serial today); and the suite overall is worth a simplification and
speed pass.
*(current_plans.md item 11, two bullets.)*

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

## What this file does not cover

`plans/torch_port/active/functional_interface_proposal.md` is its own
open proposal, awaiting review; H2 is its entry in this list.  The
`closed/` records carry decisions with revisit triggers rather than
open items.  Items that closed before 2026-08-16 are not listed here;
`plans/open_items_v2.md` section J is their audit record.
