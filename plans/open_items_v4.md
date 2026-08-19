# Open items — v4

**Compiled 2026-08-19.**  This file supersedes `plans/open_items_v3.md`
and retires `plans/current_plans.md`, absorbing its still-live items;
the migration audit is the last section of this file.  Each open item
appears here once, rewritten to say what is true now rather than how
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

Ordered 2026-08-19, after Greg's E-section rulings and the close of
the cone forward rework line (multigpu_findings.md §1.32, §1.33).

1. **Finish B7, one chained submission.**  The family-scoped
   threshold refresh on the committed sorted forward, implementing
   E3's ruled coarse table in the same pass
   (`floors_coarsening_proposal.md` §5), chained with a re-run of
   mg27 to re-anchor the comparison tables; then paste the rows,
   refresh the user-docs timing table, and close B7.
2. **D3, the forward driver's leftover-memory fix.**  The last
   code-level item on the parallel and cone drivers.
3. **MBIRJAX/MBIRTorch comparison.** Time and peak memory for a 15 iteration vcd recon with nontrivial weights at 1024**3 on 1 GPU, each of parallel and cone, each of MBIRJAX and MBIRTorch.  
3. **C5, the ledger calibration pass.**  One pass answering the five
   collected inputs (`torch_port/active/ledger_calibration_inputs.md`).
4. **Riders as time allows.**  C6's automatic-device nightly row, and
   B6's multiaxis probe: break the 512-class two-device loss into its
   component calls, with the 1024-class loss as the second case.

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

**B6. OPEN, grown 2026-08-18.  The multi-axis geometry runs slower on
two GPUs at some problem sizes and faster at others, and no mechanism
explains it.**  Two threshold refreshes measured the same shape: two
devices win at the 384-class, lose badly at the 512-class, win at the
768-class, and lose again at the 1024-class.  The sizes where two
devices help are therefore a window, not a threshold, which the
device-count thresholds cannot express.  The current threshold admits
two devices from the 768-class up, and therefore into the 1024-class
slowdown.  E3's ruling holds this geometry to
one device (a sentinel, not a threshold) until the mechanism is
known.  The proposed
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
(multigpu_findings.md §1.32), as did the sorted-order rework that followed it
(multigpu_findings.md §1.33); Greg closed that line 2026-08-19, and Section 9 of
`plans/torch_port/active/pfwd_segmented_design.md` carries the
measured verdicts.
*(mg27 through mg31 rows; multigpu_findings.md §1.5, §1.19 corrected, §1.26
through §1.31; pfwd_segmented_design.md;
projector_kernels/gpu_headroom_findings.md; greg_notes.md item 6.)*

## C. Measurement and calibration gaps

**C1. CLOSED 2026-08-18.  Re-anchor the cross-framework comparison.**
mg27 re-measured the full mbirtorch column on the padded tip; cone
took the padding's gain at two and four devices.  → execution_overview.md
§5.1 to §5.3; `experiments/torch_port/mg27_reference_timings.md`.

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

**D8. Other open issues from `API_specification.md`.**

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
the device-count thresholds.**  Coarser table; family-scoped
refreshes; cone n=4 joins the shared row; multiaxis n=2 is a
sentinel until B6's mechanism is known.  Implementation rides B7's
owed refresh.  → floors_coarsening_proposal.md §5 (the ruling).

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
timing table is stale.**  The table now carries mg27's parallel-beam
numbers (40.0, 23.8, 15.5 s at the 1024 class), names the geometry,
and the surrounding ratios are recomputed; the old 94 s row is gone.
The edit is staged in mbirtorch (`docs/source/usr_multi_gpu.rst`) for
Greg's commit.
*(open_items H2; multigpu_plan.md item 7; closed/docs.md:12.)*

**F2. CLOSED 2026-08-19.  Two demo-side documentation pieces.**
Charlie finished both.  → closed/demo_consolidation.md:229, :251.

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

**H2. A functional recon interface.**  Provide a functional, not
object-oriented, interface for basic parallel-beam and cone-beam
reconstruction.
*(functional_interface_proposal.md item 10.)*

**H3. Multi-resolution reconstruction, back-burnered.**  A possible future direction (may be superseded by NN/INR approaches): 
reconstruct at binned resolution and upsample as the
initializer for the next-finer level.  The pilot design, the null
hypothesis to beat (a direct-reconstruction initializer), and the
matching problems are in the migrated design notes.
*(torch_port/active/multires_direction.md; current_plans.md item 12,
migrated 2026-08-19.)*

**H4. CLOSED 2026-08-19: Charlie's queue.**  Three recorded items: the Lilly comparison run
of the original package for speed and memory (run informally - happy with results); 
the plans-fork pull request to Greg (stale); and telling Greg about
`split_sino_recon` device handling (superseded by D8).  The multi-device completion
checklist those items came from is otherwise closed.
*(open_items J4;
closed/preprocess_sharding_translation_multiaxis.md section C.)*

**H5. Archive old plans, back-burnered.**  Move older plan documents and scripts to
archived storage; the plans tree accumulates.  The current repo has multiple directories to organize
by topic.  Could consider a move to a new repo `mbirtorch_plans`. *(current_plans.md item 11.)*

**H6. Test-suite quality and cost.**  The demo-data gates sit 300 to
640x above their measured noise floors; `test_logging` wants a
two-instance case; the cone fixtures could be module-scoped (72 s
serial today); and the suite overall is worth a simplification and
speed pass.
*(current_plans.md item 11, two bullets.)*

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
