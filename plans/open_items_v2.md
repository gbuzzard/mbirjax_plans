# Open items — v2

**Compiled 2026-08-16, statuses updated 2026-08-17** from three
sources: `plans/open_items.md` (the
2026-08-11 compilation), `plans/current_plans.md`, and
`plans/torch_port/active/greg_notes.md`.  Every item was cross-checked
against the work that has landed since its source was written, so items
that closed in the meantime appear only in the audit list at the end.
Where the same item appears in more than one source, it appears here
once and cites all of them.  Several cited files have moved from
`plans/torch_port/active/` to `plans/torch_port/closed/`; references
here use the current locations.  The port moves daily, so an item here
can close between compilations.

One naming convention, used throughout: the "device-count thresholds"
are the measured problem sizes below which the automatic device choice
keeps a reconstruction on fewer GPUs (the code calls them widening
speed floors).

## Contents

- [Suggested execution order for A and B](#suggested-execution-order-for-a-and-b)
- [A. Large-scale reconstruction (2048-class and larger)](#a-large-scale-reconstruction-2048-class-and-larger)
- [B. Speed investigations at current sizes](#b-speed-investigations-at-current-sizes)
- [C. Measurement and calibration gaps](#c-measurement-and-calibration-gaps)
- [D. Correctness and API-contract gaps](#d-correctness-and-api-contract-gaps)
- [E. Decisions waiting on a person](#e-decisions-waiting-on-a-person)
- [F. Documentation and demos](#f-documentation-and-demos)
- [G. Release, nightly, and automation](#g-release-nightly-and-automation)
- [H. Scheduled and back-burner work](#h-scheduled-and-back-burner-work)
- [I. Accepted gaps and monitor-only notes](#i-accepted-gaps-and-monitor-only-notes)
- [J. Closed since the sources were written](#j-closed-since-the-sources-were-written)

---

## Suggested execution order for A and B

The order below came out of a discussion with Greg on 2026-08-16.  The
principle is cheap diagnostics and desk work first, expensive remedies
last, and it cuts across the two categories.

1. **A1, the capacity table.**  Desk work on charges already in the
   memory ledger; nothing in B can invalidate it, and it decides
   whether A2 is a feasibility prerequisite or an optimization.
2. **B2's discriminating run, and B4.**  Nearly free on any cluster
   session, and B2's answer compounds later.  Neither gates A.
3. **A2, decided — and landed if the table confirms it binds — before
   any baselines.**  The baselines should measure the memory structure
   that will ship.
4. **A3 and A4 together**: the first composed 2048-class runs, with
   the batch sweep riding the same jobs, instrumented to read the back
   projection's share of time.  That reading diagnoses B1 at the scale
   that matters.
5. **Only then B3, A5, and any B1 remedy**, gated on what the
   baselines show.

**Progress note (2026-08-17, evening):** steps 1 through 4 are done.
Step 5 is scoped by the baseline readings: the kernel campaign starts
at the cone back projection (B1), the width mechanism is closed (B2),
and sorted accumulation (B3) points at the back.

Three reasons the expensive B work waits.  The 1024-class evidence
already refuted cache-residency effects at that size, so the
kernel-interior bets need 2048-class attribution to justify them.
B1's anomaly is specific to two devices, and memory forces 2048-class
runs to four or more devices, so it may not bind at scale.  And the
invalidation asymmetry favors baselines first: re-running baselines
after a kernel change costs GPU-days, while a mistargeted kernel
campaign costs weeks.

## A. Large-scale reconstruction (2048-class and larger)

**A1. CLOSED 2026-08-16.  Determine the memory needs and bottlenecks
for 2048-class reconstructions on multiple GPUs.**  At (2048, 2016, 1984) no single
GPU can hold a reconstruction, so the first question is which device
counts can, and which memory term limits each count.  The deliverable
is a capacity table computed from the library's memory ledger.
*(open_items C1; multigpu_plan.md item 6; greg_notes.md the "2K+"
group.)*
**Update (2026-08-16, evening).  This completes the item:**
* The capacity table is computed and opens
  `torch_port/active/two_k_design.md`.
* Three GPUs fit with little margin, and four or more fit
  comfortably.
* The bottleneck at every workable count is a set of six per-device
  arrays whose sizes all fall with the device count.  No tiling is
  needed for capacity.
* The table is a model.  Validating it at the 2048 class is item A3.

**A2. CLOSED 2026-08-17, by ruling.  Restructure how partial
back-projections are combined.**  When
several GPUs back-project, one GPU collects and adds the partial
results, holding about 37 GB at the 2048 class.  That cost does not
fall as GPUs are added, which makes it the per-GPU memory ceiling at
that scale; a tree-shaped or in-place combination is the one
identified change that would let memory scale down with the device
count.  Greg's top memory pick.
*(open_items C2; greg_notes.md item 8; multigpu_findings.md §6.5.)*
**Update (2026-08-16, evening).  The item is answered; one ruling
remains:**
* The combining step was restructured on 2026-08-11, after the notes
  above were written.  Its cost now falls with the device count.
* The capacity table shows the combining step is no longer the memory
  ceiling at any count up to eight, so no further restructuring has a
  capacity case.
* Ruled (Greg, 2026-08-17): the landed restructure is accepted, and
  this item is closed.  The slab-size sweep (C4) rides the 2048-class
  baseline runs.

**A3. CLOSED 2026-08-17.  Validate the memory estimate at the 2048
class.**  Every
calibration so far ran at the 1024 class.  The first composed
2048-class runs both check the estimate where it matters and anchor
the tiling work.
*(open_items C4; greg_notes.md item 9.)*
**Update (2026-08-17, afternoon).  This completes the item:**
* The first 2048-class reconstructions ran, cone and parallel, at
  three and four devices.  Every calibration ratio sits between 1.10
  and 1.19, inside the band and never under the floor.
* Both two-device arms were refused by the preflight, which is the
  capacity table's two-device verdict confirmed on hardware.
* The verdicts are recorded in `two_k_design.md` §6 and
  multigpu_findings.md §1.20.

**A4. OPEN, one ruling left: the default.  Sweep the forward
pixel-batch size at large scale.**  The
default stays at 8192 deliberately: the sweep that favored 16384 to
32768 (by 4 to 15 percent) ran at the 1024 class and never bracketed
the optimum, and the batch's cross-device transient grows with the
slice count, so the large-scale sweep is the one that should set the
default.
*(open_items C3; greg_notes.md item 4; current_plans.md item 11.)*
**Update (2026-08-17, afternoon).  Measured; the default ruling
remains:**
* The sweep ran at production scale.  Forward busy time falls 17 to
  18 percent from batch 8192 to 65536, and the last doubling is worth
  2 to 3 percent, so the knee is bracketed at or just above 32768.
* Memory does not constrain the choice: the transferred cylinders
  stay under 1.5 GiB at the largest batch.
* Remains: Greg's ruling on moving the default to 32768, a reviewed
  change that re-anchors the affected nightly rows.

**A5. OPEN, back burner.  Two-dimensional tiling and the cache
directions.**  Blocking
both axes of a call's working set so it stays resident in cache is the
next idea after the one-dimensional pixel blocking that already
landed.  The 1024-class measurements mostly refuted cache-residency
effects at that size, so the honest sequence is the 2048 baselines
first; tiling is the main lever if those runs show cache-boundedness
or a memory squeeze.  Phase-blocked accumulation is the entry-level
version of the same idea.
**Update (2026-08-17):** the baselines ran and found no memory
squeeze at three or more devices, so tiling has no capacity case.
The speed case now routes through the cone back projection (B1); this
item stays behind the kernel campaign.
*(open_items E3; greg_notes.md the closing addendum.)*

## B. Speed investigations at current sizes

**B1. OPEN, the kernel campaign's first item.  The cone back
projection takes longer on two devices than on one.**  Its GPU time reads 30.3 s at two devices, and splitting only
pays at three or more.  Nobody owns finding out why, and no landed
change addresses it.
*(open_items E1; greg_notes.md item 7; multigpu_findings.md §6.2.)*
**Update (2026-08-17, afternoon).  The anomaly generalizes, and this
is now the top speed target:**
* At the 2048 class the cone back projection's busy time RISES from
  137 s at three devices to 228 s at four, and it is more than half
  of a four-device cone wall.
* A mechanism hypothesis is recorded: each band call pays the cone
  back kernel's full-detector-row grid, one band per slice-owner, so
  total back work grows with the device count — the cost structure
  the forward had before the cylinder transfer replaced it.
* Remains: design the back-projection counterpart of the cylinder
  transfer, or a kernel-level fix, as the kernel campaign's first
  item.  → multigpu_findings.md §1.20.

**B2. OPEN, one small increment left: pad widths to multiples of 16.
The kernel-width puzzle.**  A parallel projection-kernel launch
at half width costs the same as one at full width, so a narrow call
wastes half its cost.
**Known (mechanism found 2026-08-17):**
* The cost step comes from the kernel width, not from the device
  count.  The column-gather remedy followed from that answer and has
  shipped on all four geometries.
* The mechanism is the compiler's divisibility-by-16 specialization
  of the kernel's width argument.  Widths divisible by 16 run at full
  efficiency wherever the data lives; widths that are not run at
  about half, through higher register use that caps occupancy.
  Cache residency, allocation size, and layout were each varied alone
  and are innocent.
* Production gather calls at the standard cells already use
  divisible widths (1008 and 2016), so nothing ships slow today.
**Remains:**
* A small robustness increment: pad the width argument to the next
  multiple of 16 at kernel entry, so arbitrary user shapes cannot pay
  the two-times penalty.  Whether the back kernel carries the same
  property is unmeasured.
* For B3: the counters show the atomic write path generating 192
  times the ideal L2 sector traffic at every width, which is what
  sorting would attack, but the fast widths carry it without penalty
  and DRAM writes are already near ideal.  B3 stays open, and the
  2048-class attribution now points it at the cone BACK projection,
  which dominates a four-device cone wall (B1's update).
*(multigpu_findings.md §1.9, §1.19, and §1.20.)*

**B3. OPEN, now pointed at the cone back kernel.  In-kernel sorted or
segmented accumulation.**  Sorting each
call's detector writes so the scattered additions become ordered,
collision-free segments.  The cheap per-call form was re-tested on
2026-08-11 and stands rejected: it wins over the plain torch scatter
in eager mode but loses to the production Triton kernel by about
3.5x.  What remains is the deeper form inside the kernel itself, which
belongs to a kernel campaign and is coupled to B2's answer.
*(open_items E4; greg_notes.md item 6 and the addendum;
current_plans.md item 13; multigpu_findings.md §1.14.)*

**B4. CLOSED 2026-08-17.  Time translation and multi-axis on the
newer forward path.**
Both geometries still run the older slice-band forward, whose cost
structure is what made cone slow, and the recorded rule is that a
geometry switches only on its own measurement.  One two-run job per
geometry would settle whether they get the same 1.2 to 1.6x the
switch gave cone.
*(open_items D2; greg_notes.md item 3.)*
**Update (2026-08-17, early morning).  Measured; the switch awaits
review:**
* The comparison ran at both production cells, at two and four
  devices.  Values passed everywhere, at the e-5 class against a
  1e-3 gate.
* The gather is faster everywhere measured.  Multiaxis: 1.3x to 1.9x
  on the forward and 1.1x to 1.2x composed.  Translation: 1.9x to
  38x on the forward and 1.4x to 1.9x composed.  At four devices the
  banded translation forward ran ten times slower than one device,
  and the gather removes that pathology.
* Memory fell too at the shipped batch, most at the translation
  four-device forward (26.5 GiB down to 10.6).
* Ruled (Greg, 2026-08-17): use the gather.  The flip landed as
  mbirtorch commit 7cd32ed, with the suite green and the parity tests
  extended.  Today's manual nightly and the 2048-class baselines are
  the confirmation runs.  The floors question (C2) stays open, and
  the banded forward path's removal is B5.  →
  multigpu_findings.md §1.18.

**B5. OPEN, implementation done, merge left.  Remove the banded
forward path, and rename the survivor to cylinder transfer.**  With the gather the measured
winner on all four geometries, the banded multi-device forward is
unused code behind a switch that CI must keep green and production
never runs.  Greg ruled on 2026-08-17 to remove it.
* Done: the removal and the cylinder-transfer rename are implemented,
  gated (suite green, byte-identical defaults, grep-clean), and
  committed on the worktree branch.  The confirmation runs passed:
  the manual nightly on the flipped tree and the 2048-class
  baselines.
* Remains, for Greg: merge the branch, then run the floors re-bless
  (one command; the staleness note names the inputs).  A pre-existing
  device-policy test fragility is tracked as its own task chip.
* Details: `torch_port/active/banded_forward_removal.md`, including
  the old-to-new name mapping.

## C. Measurement and calibration gaps

**C1. Re-anchor the cross-framework comparison.**  The single-device
reference timings that every speedup ratio divides by, and the whole
512-class column of the comparison table, predate the 2026-08-11
change that made the column-gather forward projection the default.
One run per geometry re-anchors them.
*(open_items F1 and F2; active/execution_overview.md §5.2, §5.3.)*

**C2. OPEN, a decision plus one measurement.  Translation and
multi-axis have no device-count thresholds of their own.**  Both reuse parallel-beam's thresholds, the most
permissive measured set.  This is also an input to the
threshold-simplification decision (E3).
*(open_items D1; multigpu_plan.md §0a item 9.)*
**Update (2026-08-17).  Now measured, and the reuse is harmful in one
case:**
* The 2026-08-17 comparison timed both geometries at two and four
  devices.  Multiaxis composed time still rises from two devices to
  four (394 s to 951 s at its production cell, on the gather), while
  parallel's thresholds admit four devices at that size.  An
  automatic multiaxis reconstruction there widens into a slowdown.
* Translation reads flat from two to four devices, so the reuse is
  harmless for it.
* Remains: measure these geometries' own thresholds, or hold
  multiaxis to two devices until then.  The decision is Greg's.

**C3. The scan preprocessing pipeline's concurrency is unmeasured on
GPUs.**  The multi-device path is correctness-gated only; whether it
actually runs faster has never been measured on a GPU node.  (The
denoiser half of the original item was measured on 2026-08-16 and is
closed: splitting never paid.)
*(open_items F5;
closed/preprocess_sharding_translation_multiaxis.md:171.)*

**C4. CLOSED 2026-08-17, measured.  The combining-step slab size was
a reasoned default, not a measured one.**  The 64 MiB slab used when moving partial results
between devices was chosen by an argument about launch overhead
against transfer cost; no sweep has run.
*(open_items F3; mbirtorch `_sharding.py`, REDUCE_SLAB_BYTES.)*
**Update (2026-08-17, afternoon).  Measured; this can close:**
* The whole 16-to-256 MiB range moves the back projection by 0.8
  percent at the 2048 class, outside the repeat spread but small.
* 256 MiB is marginally best; the 64 MiB default is defensible; no
  structural work is warranted.  The one open question is whether to
  bump the constant, and it is cosmetic.

**C5. OPEN, inputs accumulating for the next calibration pass.  The
back-projection batch memory charge is conservative.**  The
estimate counts four slabs where only three are live at once, about
0.8 GB high at parallel 1024 on two devices.  The 2026-08-16 gate run
added related readings: cone and parallel at three devices over-read
the declared band's top (up to 1.417 against 1.30).  Both are inputs
to the next calibration pass.  The 2048-class runs added a third
input: every ratio there sits between 1.10 and 1.19
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
rebind, the same pattern that was fixed in the back-projection loop;
it was out of that fix's scope.
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
geometries.**  The multi-axis port carried over the original's choice
of which loops scatter and which gather.  Charlie's hypothesis (gather
on the vertical axis, scatter on the horizontal) is testable one axis
at a time against the stored reference results.
*(open_items D4;
closed/preprocess_sharding_translation_multiaxis.md:273.)*

## E. Decisions waiting on a person

**E1. The remedy for compiled cross-count value differences.**  New
2026-08-16, recorded outside the three source files and included here
for completeness.  Geometries without hand-written kernels produce
deterministic differences of about 6e-4 between device counts whenever
the split is uneven, because compiled code for a second block length
orders its reductions differently.  Three options are recorded: accept
it and gate such comparisons at 1e-3 (the implicit status quo), compile
every count the same way, or pad compile shapes only.
*(multigpu_findings.md §1.16; closed/remove_padding.md P5.)*

**E2. The floors-refresh automation questions.**  The plan for moving
threshold re-measurement into the nightly is drafted, and it asks
seven questions before implementation: which branch to watch, whether
the refresh script gains a write mode, the proposal rate limit, the
trust boundary of the artifact's path, who owns failure mail, whether
to force a periodic refresh, and the interaction with E3.  Needs
coordination with the nightly's owner.
*(open_items G2 and I3; active/floors_refresh_automation.md §7;
multigpu_plan.md §0a item 8.)*

**E3. Whether and when to simplify the device-count thresholds.**  The
thresholds are measured at specific shapes, one GPU model, and one run
configuration, and that precision is fragile.  Fewer, coarser
thresholds that survive shape and hardware variation are preferred to
exact crossovers that do not.  Interacts with E2: a coarser table
drifts less and needs the automation less.
*(open_items G3; multigpu_plan.md item 9 and §0a.)*

**E4. The nightly cadence for multi-GPU rows, and campaign close-out.**
Report the per-night cost of the multi-GPU nightly rows at both
candidate cadences; the nightly's own trial run is the authoritative
figure.  Confirm or revisit the full-cadence choice, then update the
campaign's entry in `current_plans.md`.
*(open_items I4; multigpu_plan.md item 7 and §0a.)*

**E5. Torch version-advance handling for the dependency watch.**  For
Charlie: merge floor advances as they arrive, or batch them yearly.
The watch behaves identically either way; the answer only sets how
long an eager pull request may sit.
*(open_items G4; closed/python_matrix_nightly_check.md §6.)*

**E6. Where demos live in the docs.**  The one demo question the
critiques did not settle: keep the current demos-and-FAQs page
structure, or something simpler.
*(open_items G5; closed/demo_consolidation.md §7.)*

**E7. A written Python-version increment policy.**  A proposed policy
is on the page for Greg's and Charlie's review.
*(open_items G6; closed/release_workflow.md decision 3.)*

## F. Documentation and demos

**F1. The multi-GPU page's timing table is stale.**  The table in
`usr_multi_gpu.rst` still shows 94 s for the 1024-class single-device
reconstruction, which predates the kernel-era path (the same
reconstruction now runs near 40 s).  It is the only place a user sees
these numbers, and refreshing it is part of campaign close-out (E4).
*(open_items H2; multigpu_plan.md item 7; closed/docs.md:12.)*

**F2. Two demo-side documentation pieces.**  An FAQ paragraph for the
reversed rotation direction, with the symptom, the cause, and the fix;
and the demo set itself, which is designed but not built, to land one
demo at a time with review.
*(open_items H5; closed/demo_consolidation.md:229, :251.)*

## G. Release, nightly, and automation

**G1. The release-workflow remainder.**  The Read the Docs stable
default and its token matter only at the first tagged release;
`release.yml` and the optional documentation preview are unwritten;
the wheel-check developer script is unwritten.  The version policy is
E7.
*(open_items I1; closed/release_workflow.md:26.)*

**G2. Read the dependency watch's first quiet week.**  The watch is
live and armed; its first week of watchdog lines on the gautschi
nightly is to be read once, by Greg.
*(open_items I2; closed/python_matrix_nightly_check.md:352.)*

## H. Scheduled and back-burner work

**H1. MAR: cache the fitting matrix.**  Compute each column of the
metal-artifact fit's matrix once instead of recomputing it
quadratically often — a cheap, self-contained speed win with no
statistical questions.  Subsampling stays deprioritized, with the
reasoning recorded.
*(current_plans.md item 9.)*

**H2. Functional recon interface**  Include a functional (non-object oriented) 
interface for basic parallel beam and cone beam recon 
*(functional_interface_proposal.md item 10.)*

**H3. Multi-resolution reconstruction.**  A future direction, not for
the next release: reconstruct at binned resolution and upsample as the
initializer for the next-finer level.  The pilot design, the null
hypothesis to beat (a direct-reconstruction initializer), and the
matching problems are recorded in the item.
*(current_plans.md item 12.)*

**H4. Charlie's queue.**  Three recorded items: the Lilly comparison
run of the original package for speed and memory (script already on
the cluster); the plans-fork pull request to Greg; and telling Greg
about `split_sino_recon` device handling.  The multi-device completion
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
  value comparison.**  Accepted 2026-08-10 after re-examination; a
  full re-run prices at eleven GPU-hours, a partial at 2.5.
  *(open_items F4; multigpu_findings.md §4.)*
* **The mg13 job's four-device runs never happened** (a conda rebuild
  removed the interpreter mid-job); the conclusion stood on the
  one-device runs by design.  *(open_items J2; multigpu_findings.md
  §1.14.)*
* **The truncation warning fires even after its own fix is applied**,
  because it reads the sinogram alone; it could stay quiet when the
  reconstruction region already exceeds the field of view.  Noted for
  Greg, unscheduled.  *(open_items K2; closed/demo_consolidation.md.)*
* **The jax rounding-bug precondition** stays monitor-only at the six
  per-slice rounding sites.  *(current_plans.md item 11.)*
* **Two documentation defects found upstream** in the original
  package; the copies here are corrected, and upstream is winding
  down.  *(open_items H4.)*

## J. Closed since the sources were written

Each of these appeared open in a source file and has since closed;
the pointer is to the closing record.

* **Entry-point device policy, all of it** (open_items B4, B5; the
  section-B remainder): the plan's nine increments landed and the plan
  closed 2026-08-16.  → `closed/entry_point_plan.md`.
* **`interpolate_defective_pixels` input forms** (half of open_items
  B6): fixed 2026-08-16.  → API_specification.md Issue 1.
* **The transfer-stall overlap** (greg_notes item 1): the copy streams
  landed 2026-08-11, closing most of the stall.  →
  multigpu_findings.md §1.13.
* **The per-batch accumulation cost** (greg_notes item 2): the fused
  accumulation landed 2026-08-11.  → multigpu_findings.md §1.15.
* **The cheap sorted-form re-test** (the probe half of greg_notes
  item 6): ran 2026-08-11; the rejection stands.  →
  multigpu_findings.md §1.14.  The in-kernel form remains as B3.
* **The two housekeeping items** (greg_notes closing note): the
  threshold re-measurement ran twice since (2026-08-13 and
  2026-08-16), and the single-pixel kernel check passed on the
  cluster.  → multigpu_findings.md §1.12, §1.17.
* **The sharded denoiser's GPU concurrency** (half of open_items F5):
  measured 2026-08-16; splitting never paid, and the thresholds now
  hold an automatic denoiser to one device.  → multigpu_findings.md
  §1.17.
* **Which entry points spread** (open_items H1): the page was
  rewritten in the entry-point plan's documentation increment, and
  every entry point now takes the same policy.
* **The dashboard documentation page** (open_items H3): landed; no
  held markers remain in the docs tree.
* **Stale preprocessing docstrings** (open_items K4): corrected in the
  entry-point plan's documentation increment.
* **The gautschi storage outage of 2026-08-11** (open_items J1):
  resolved; the 2026-08-16 cluster campaign ran entirely on
  `/scratch` without incident.
* **The denoiser sharding-parity charter** (current_plans.md item 11,
  A4): the sharded denoiser landed with the prerelease merge, with the
  log arguments the other entry points carry.
* **The divided-form phantom return** (current_plans.md item 15's
  second bullet): superseded by the ruling that data crosses from
  generation to reconstruction through host memory.  →
  `closed/entry_point_plan.md` §8, increment 7.
* **B2's discriminating arm**: found on 2026-08-16 to have already
  run on 2026-08-10.  The B2 entry above now records the answer, and
  only the mechanism question remains.  → multigpu_findings.md §1.9.
* **A2's memory premise**: the combining restructure landed on
  2026-08-11, between the sources and this compilation.  The A2
  entry above records the update.  →
  `torch_port/active/two_k_design.md` §3.

## What this file does not cover

`plans/torch_port/active/functional_interface_proposal.md` was added
on 2026-08-16 and is its own open proposal, awaiting review.  The
`closed/` records carry decisions with revisit triggers rather than
open items.  
