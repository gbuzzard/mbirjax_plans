# Open items — v3

**Compiled 2026-08-18.**  This file supersedes `plans/open_items_v2.md`,
which was compiled 2026-08-16 and carried status updates through
2026-08-18.  Each open item appears here once, rewritten to say what is
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

1. **DONE 2026-08-18.  The counter run on the cone back kernel.**
   mg25 ran and the counters answered B3's precondition: the gathers
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
4. **DONE 2026-08-18.  The parallel forward counter variant (B7).**
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

**A5. OPEN, back burner.  Two-dimensional tiling and the cache
directions.**  Blocking both axes of a call's working set so it stays
resident in cache is the next idea after the one-dimensional pixel
blocking that already landed.  Phase-blocked accumulation is the
entry-level version of the same idea.  The 1024-class measurements
mostly refuted cache-residency effects at that size, and the 2048-class
baselines found no memory squeeze at three or more devices, so tiling
has no capacity case.  With B1 closed, the speed case routes through
the kernel campaign's remaining question (B3), and this item stays
behind it.
*(open_items E3; greg_notes.md the closing addendum.)*

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
accumulation.**  The counter run (mg25, findings §1.24) found nothing
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

**B6. OPEN, grown 2026-08-18.  The multiaxis torch-body slowdowns do
not have a mechanism, and the two-device win is a window.**  The mg22
floors refresh measured two readings no current mechanism explains.
Multiaxis at two devices wins at the 384-class (1.25x), loses hard at
the 512-class (0.35x; 11.4 s at one device against 32.6 s at two,
repeats tight, values agreeing), and wins again at the 768-class
(1.46x); and four devices lose 0.23x at the 768-class against two.
mg26 added a third reading on the padded tree: two devices LOSE again
at the 1024-class (0.80x), above the 768-class win, so the admission
region is a window in problem size and the floor structure cannot
express it (findings §1.25).  The pasted floor still admits two
devices from the 768-class, into that measured 20 percent slowdown at
the 1024-class; the coarsening proposal recommends holding the family
to one device until this item finds the mechanism.  The readings do
not sort by band divisibility, and multiaxis runs compiled torch
bodies, so this is not the cone back kernel's specialization effect.
The natural first probe is an mg21-style decomposition of the
512-class two-device composed call, now with the 1024-class loss as a
second cell to discriminate against.  Translation's uniform losses at
every count may share the mechanism and would ride the same
attribution.
*(multigpu_findings.md §1.22, §1.25; the floor rows' notes in
mbirtorch/_widening_floors.py.)*

**B7. OPEN, measured 2026-08-18; the increment is Greg's call.  The
parallel forward kernel against mbirjax's.**  vcd parallel trails
mbirjax at every count (mg27: 1.55x at one device) while cone matches
or beats it, and the gap is one object: the parallel forward's device
span is 28.9 s of the 40 s one-device wall (findings §1.5), so
torch's forward alone exceeds mbirjax's whole 25.8 s reconstruction.
The counter run Greg ordered (mg28, findings §1.26) named the
mechanism.  The kernel is LOAD-BOUND on the per-view reload of the
values block: memory throughput 52 percent of peak against SM's 16,
38 warps stalled on memory per issue-active cycle, and DRAM
re-fetching the 3.1 GB values block 130 times per 128-view launch.
The atomic write path is NOT the problem: it measures 1.50x from
coalesced, and §1.19's "192x" characterization was a pricing error,
corrected in §1.26.  mbirjax's segmented-accumulation design
difference is therefore not where the torch-side gap lives.  The
candidate remedy is the one the kernel's own docstring reserved for
exactly this measurement: move the view axis from the launch grid
into an in-program loop, so each values tile is read once instead of
once per view.  Greg ruled 2026-08-18: proceed, and the spike step ran twice the
same evening (mg29 and mg30, findings §1.27).  The spikes settled
the loop-structure family: reordering which loop owns the view moves
the DRAM traffic but not the floor, because all three designs issue
the same atomic-add volume -- 1.79 TB of atomic sectors per launch
through L2, which the counters now name as the kernel's binding
resource.  The best interior point, a 32-view chunk per program,
reads 1.17x on the production mixture, values-safe.  The way past
the floor is fewer atomics: in-tile accumulation, one add per
(channel, column) of a tile's span, an arithmetic 2.4x to 2.8x
reduction that matches the remaining gap to mbirjax's segmented
forward.  The design note is drafted at Greg's direction
(`torch_port/active/pfwd_segmented_design.md`), with the cone
forward's counter reading taken the same evening (mg31, findings
§1.28: mixed profile, same shared scatter, smaller projected payoff
-- cone rides the spike opportunistically).  Two spikes ran the same
evening.  mg32 (findings §1.29): the contraction wins 1.68x at the
full mask, subset calls lose for lack of channel locality, and the
TF32 twin answers the precision question (1.84x at 1e-3-class
values).  mg33, at Greg's channel-sorting question (findings
§1.30): sorting per view makes the contraction win EVERYWHERE --
3.97x at the full mask, 2.8x to 3.9x at the subsets, the atomic
volume down 31.6x, values at 3e-6, sort costs in the milliseconds
and amortizable.  The combined selection reads 3.45x on the
production mixture, which by §1.5's shares puts the one-device
1024-class wall near 20 s against mbirjax's 25.8.  Greg gave the
library-step go the same evening, and the step is staged and half
gated: the sorted route landed in triton_parallel.py as the
wrapper's default (MBIRTORCH_SORTED_FORWARD=0 restores the per-tap
kernel), the CPU suite reads 584 passed with four new sorted-route
gates, and mg34 passed the composed gate's first half whole -- the
full GPU suite, then 1.43x to 1.89x at the 1024 class and 1.03x to
1.41x at the 512 class with every value gate inside 1e-3 (findings
§1.31).  Parallel now matches or beats the recorded mbirjax column
at every measured count.  The 2048-class confirmation is running as
mg35 (job 15347106); the staged change is committed only after it.
*(mg27 through mg31 rows; findings §1.5, §1.19 corrected, §1.26
through §1.28; pfwd_segmented_design.md;
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
back-projection batch memory charge is conservative.**  The estimate
counts four slabs where only three are live at once, about 0.8 GB high
at parallel 1024 on two devices.  The 2026-08-16 gate run added related
readings: cone and parallel at three devices over-read the declared
band's top (up to 1.417 against 1.30).  The 2048-class runs added a
third input: every ratio there sits between 1.10 and 1.19
(multigpu_findings.md §1.20).
*(open_items F7; closed/backloop_attribution.md §5;
multigpu_findings.md §1.16.)*

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

**E2. The floors-refresh automation questions.**  The plan for moving
threshold re-measurement into the nightly is drafted, and it asks seven
questions before implementation: which branch to watch, whether the
refresh script gains a write mode, the proposal rate limit, the trust
boundary of the artifact's path, who owns failure mail, whether to
force a periodic refresh, and the interaction with E3.  Needs
coordination with the nightly's owner.
*(open_items G2 and I3; active/floors_refresh_automation.md §7;
multigpu_plan.md §0a item 8.)*

**E3. RULED 2026-08-18: coarsen.  Whether and when to simplify the
device-count thresholds.**  The thresholds are measured at specific
shapes, one GPU model, and one run configuration, and that precision
is fragile.  Greg ruled for fewer, coarser thresholds that survive
shape and hardware variation.  The proposal is drafted:
`torch_port/active/floors_coarsening_proposal.md` carries the
coarsening rules, and, at Greg's 2026-08-18 direction, a
family-scoped mode for the refresh tool so future refreshes measure
only the families whose cost inputs moved.  The proposal's rows fill
from the mg26 padded-tree refresh, and its §5 is the ruling request.
A coarser table also drifts less and needs the E2 automation less.
*(open_items G3; multigpu_plan.md item 9 and §0a;
active/floors_coarsening_proposal.md.)*

**E4. The nightly cadence for multi-GPU rows, and campaign close-out.**
The cost in question is GPU-hours on the ai partition against the
group's allocation, plus the nightly's wall-clock window; no money
changes hands.  The figure is now measured: one moved-branch pass with
the full device sweep took 24 to 25 minutes on four H100s, which is
about 1.6 GPU-hours, and a night where both tracked branches moved
took 48 minutes (the 2026-08-17 forced repeat).  Fire-on-change makes
an unmoved night cost seconds.  Remains: confirm the full cadence at
that price, then update the campaign's entry in `current_plans.md`.
*(open_items I4; multigpu_plan.md item 7 and §0a.)*

**E5. Torch version-advance handling for the dependency watch.**  For
Charlie: merge floor advances as they arrive, or batch them yearly.
The watch behaves identically either way; the answer only sets how long
an eager pull request may sit.  Input recorded 2026-08-18: torch minor
releases arrive roughly quarterly, so a yearly batch would leave the
floor several minors behind; the session's recommendation is merge as
they arrive, since each pull request is small, mechanical, and
CI-gated, and a standing "no" nags the watchdog line nightly.
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

**F2. Two demo-side documentation pieces.**  An FAQ paragraph for the
reversed rotation direction, with the symptom, the cause, and the fix;
and the demo set itself, which is designed but not built, to land one
demo at a time with review.
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
