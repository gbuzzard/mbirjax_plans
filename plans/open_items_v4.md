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

Reordered 2026-08-19, through the day.  Closed or landed today: B7
(the chained mg40 and mg27 re-run, findings §1.34), D3 (by
verification), the production comparison (mg41; the table is
execution_overview.md §5.4 -- mbirtorch faster in both geometries at
0.47x the memory), C5 (probe run and remainder ruled, findings
§1.35), G4's nightly memory fix (pushed), C6's automatic-choice
check (verified on four GPUs, staged for the push), and B6's first
probe (mg44, findings §1.36: the mechanism is found).  What remains
is below, in order.

1. **B6's floors refresh and the sentinel rulings.**  The remedy's
   confirmations are all in: every cell of the old two-device window
   now wins (1.53x / 1.45x / 1.52x across the multiaxis ladder,
   translation 1.25x at production; findings §1.37).  The remedy
   changed a shared floors cost input, so the FULL refresh is
   running (mg48, job 15399595) rather than the family-scoped one.
   Its verdicts land in the job log; nothing is pasted by the job.
   The rulings on the multiaxis, translation, and n=4 sentinel rows
   follow the verdicts and are Greg's.
2. **Read the first fixed night's memory rows (G4's follow-up).**
   The per-trial peaks land with the next nightly; one read says
   whether the old 26.6 GiB inflation was warmup-only or recurs in
   warm trials, which localizes the open mechanism.

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

**B6. OPEN; the mechanism was found and remedied 2026-08-19, and the
re-measures and sentinel rulings remain.  The multi-axis geometry ran
slower on two GPUs at some problem sizes and faster at others.**  The
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

**G4. FIX LANDED 2026-08-19 (pushed); the mechanism behind the old
readings stays open.  The nightly's multi-device memory columns were
unreliable.**  A four-device 1024-class arm recorded a 26.6 GiB
lead-device watermark where a fresh single reconstruction peaks at
6.84 GiB.  The two easy mechanisms are refuted: reconstruction
end-states free at refcount (mg42b, run locally), and the harness's
trial loop already drops results and collects before each timed call.
The fix is in
`mbirjax_metrics/tooling/scaling_tests/torch_backend_writer.py`,
pushed 2026-08-19, so the next nightly run carries it: the peak
counters reset before every iteration, the row's `mem_mb` is the
largest single-call peak among the warm trials, and the warmup and
per-trial peaks are recorded beside it, so an inflated call is visible
by itself.  Three consequences to expect: multi-device memory series
may step down (an improvement, not a regression; the record book
keeps the min, so baselines ratchet down); a recurrence of the old
inflation then trips the HARD memory gate instead of hiding; and the
per-trial lists localize the open mechanism on the next night it
appears.  Reading the first fixed night's rows is the follow-up.
*(multigpu_findings.md §1.35; mg42b_cycle_check.py.)*

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

**H8. Multi-device speed, part 2: the torch-body geometries.**  The
recompile-budget remedy's two structural alternatives (a budget per
device instance, and pre-marked dynamic dimensions), and the
hand-written-kernel path for multiaxis and translation with its two
entry gates: a counter run on the compiled bodies, and a
cross-framework anchor against mbirjax.  The kernel path's strongest
case is capacity rather than speed: the torch bodies' transients keep
sharding from shrinking per-device peaks, and multiaxis 1024 models
at 68 GB on one device.
*(torch_port/active/multigpu_plan_part_2.md; multigpu_findings.md
§1.36.)*

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
