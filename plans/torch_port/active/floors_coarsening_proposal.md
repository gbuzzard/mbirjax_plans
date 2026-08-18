# Proposal: coarser device-count thresholds, and a family-scoped refresh

**Status: COMPLETE, awaiting Greg's ruling (§5).**  Opened 2026-08-18;
§2's rows filled the same day from mg26's verdicts.  The fine-grained
refresh itself is pasted and staged in `mbirtorch/_widening_floors.py`
(the floors test passes on it); nothing from §2 or §3 is implemented.

**What this note is.**  E3 is ruled: the device-count thresholds (the
widening speed floors in `mbirtorch/_widening_floors.py`) should become
fewer and coarser, so they survive shape and hardware variation.  The
ruling notes the proposal should follow the padded-tree re-measure, and
that re-measure ran 2026-08-18 as mg26 (job 15342578).  This note
proposes the coarser table from its rows (§2).  It also proposes a
family-scoped mode for the refresh tool (§3), because the full refresh
costs about 12 GPU-hours and most of that re-measures rows whose costs
did not change.  Greg asked for the mode to be drafted into this
proposal during the mg26 run.

**Sources.**  The current table and its machinery are
`mbirtorch/_widening_floors.py` (FLOORS, COST_INPUT_FILES,
COST_INPUT_METHODS, BLESSED_COST_HASHES, TABLE_CHECKSUM, stale_note)
and `dev_scripts/refresh_widening_floors.py` (the plan builder, the
paste-block printer, `--bless`).  The measured base is mg22 (findings
§1.22, the unpadded tree) and mg26 (the padded tree, rows under
`results/mg26_floors/` on scratch).

---

## 1. What the padding was expected to move, and why the refresh ran whole

The padding changes a projection's cost only where a hand-written
kernel takes a width-class launch argument that is not a multiple
of 16.  At the ladder cells the forward always sees divisible widths,
so only the banded back projection at specific (cell, count) pairs
moves: the 384-class at two and four devices, the 768-class at four,
and the 1024-class at two and four.  The rows that could move are
therefore cone's and parallel's; multiaxis, translation, and the
denoiser run no hand-written kernels and their rows were expected to
reproduce.

The refresh still ran every family, because the tool has no smaller
unit.  It is the sole writer of three things that move together (the
floors, the recorded cost-input hashes, and the checksum binding
them), and its plan covers every family the table names.  About 8 of
the run's 12 GPU-hours re-measured rows expected unchanged.  That
cost is what §3 removes.

## 2. The coarser table

*The specific rows are filled from mg26's verdicts; this section's
rules are fixed first so the numbers cannot steer them.*

Three rules produce the coarser table from a refresh's verdicts:

1. **Floors sit on the ladder classes, and a family pair shares one
   row where their measured floors agree.**  Cone and parallel are
   both kernel-path projection families; where mg26 puts their
   crossovers on the same class, the table carries one shared row for
   the pair rather than two rows that agree by coincidence.
2. **A floor must win by a robust margin, not by a spread test.**  The
   current rule admits a count when it clears 1.0x by more than the
   cell's warm spread; parallel n=4 is admitted today at 1.02x.  The
   coarse rule: a count's floor is the smallest class where it wins by
   at least 1.15x, and a thinner win rounds the floor up one class.
   The margin is what survives a different GPU, a different shape in
   the same class, and run-to-run drift.
3. **Sentinels stay sentinels.**  Multiaxis n=4, translation, and the
   denoiser hold their families to safe counts; coarsening changes
   nothing about them.  Multiaxis n=2 keeps its own measured row (its
   768-class floor fences the unexplained 512-class anomaly, findings
   §1.22, and no shared row may absorb that).

**The proposed rows, from mg26's verdicts (job 15342578, the padded
64dedb8 tree; findings §1.25).**  Applying the three rules to the
measured crossovers:

| family | count | proposed floor | measured basis | rule applied |
|---|---|---|---|---|
| parallel and cone, shared | 2 | 88,080,384 (the 512-class) | parallel wins 1.20x, cone 1.34x, both at or above 1.15x | rules 1 and 2 |
| parallel and cone, shared | 4 | 1,023,934,464 (the 1024-class) | at the 768-class parallel wins 1.055x and cone 1.145x, both under 1.15x, so both round up; at the 1024-class they win 1.54x and 1.60x | rules 1 and 2 |
| multiaxis | 2 | SENTINEL (recommended; see below) | wins only in a window: 0.35x at the 512-class, 1.46x at the 768-class, 0.80x at the 1024-class | rule 3, extended |
| multiaxis | 4 | sentinel | loses at every probe | rule 3 |
| translation | 2 and 4 | sentinels | lose at every probe | rule 3 |
| denoiser | 2 and 4 | sentinels | lose at every probe | rule 3 |

Two judgment calls inside that table are named for the ruling rather
than buried.  First, cone n=4 misses the 1.15x margin by 0.005 (it
wins 1.145x at the 768-class), so the rule rounds it up to the
1024-class beside parallel; a 1.10x threshold would instead admit
cone at the 768-class and either split the families or carry an
asymmetric shared row.  The strict rule is recommended, because a
margin chosen after seeing the numbers is not a rule.  Second, the
multiaxis n=2 row: mg26 measured a LOSS above the current floor
(0.80x at the 1024-class over the 1.46x win at the 768-class), so
the two-device win is a window a floor cannot express.  The
recommended row is a sentinel until B6 finds the mechanism, which is
the same principle the family's own 2026-08-17 note applied below
the floor -- no admission that also admits a measured loss.  The
cost is real and named: the sentinel gives up a measured 1.46x win
at the 768-class.  Keeping the 768-class floor with a warning note
is the alternative.

Against the pasted fine table, the coarse table changes two
admissions: four devices wait one class longer for cone and
parallel (giving up measured wins of 1.145x and 1.055x at the
768-class), and multiaxis two-device widening stops (giving up the
768-class window win, avoiding the 1024-class regression).  The
distinct measured threshold values drop from four to two.

What the notes carry under the coarse table: the class, the bracket
readings the floor was read from, the margin rule applied, and one
provenance line (run, job, GPU, config).  The per-row histories move
to the findings pages, where they already live.

## 3. The family-scoped refresh mode

**The change.**  `refresh_widening_floors.py` gains a `--families`
flag naming the families to measure, and the cost-input hashes become
per-family sets.  A scoped run measures the named families, carries
every other family's rows forward verbatim, and still prints one
whole-table paste, so the sole-writer property and the checksum
binding survive unchanged.

Five design points, decided here so the increment is mechanical:

* **The cost inputs partition by family.**  The shared set, priced by
  every projection family: `projectors.py`, `_sharding.py`, the three
  driver methods, and `_utils.py` (see the gaps below).  On top of the
  shared set: parallel adds `triton_parallel.py` and
  `triton_cone.py`; cone adds `triton_cone.py`.  `triton_cone.py`
  appears in both kernel families because it hosts the shared kernel
  helpers (`_tile_size`, the compile-guard keys) that
  `triton_parallel.py` imports.  Multiaxis and translation carry the
  shared set alone.  `stale_note()` then names the FAMILIES whose
  inputs moved, not only the files, and the note's reader knows the
  scope of the refresh owed.
* **`--families` with no value defaults to the stale families.**  The
  tool computes which families' inputs moved and measures exactly
  those.  Naming families explicitly remains possible for a
  deliberate re-measure (new hardware, a schedule change).
* **The tool refuses to carry a stale family.**  Carrying a family
  forward asserts that its costs did not change.  With per-family
  input sets that assertion is checkable, so a scoped run errors
  before any GPU work if a family it would carry has drifted inputs.
  A change to a shared input therefore forces the full refresh, and a
  change confined to a kernel file allows the scoped one.
  Re-blessing the hashes after a scoped run is then honest.  Every
  family's rows are either measured under the current code or carried
  with their inputs provably unmoved.
* **Carried rows keep their provenance.**  A carried row's
  `measured`, `commit`, and note fields pass through untouched, so
  the table never claims a measurement that did not happen.  The
  paste header names which families this run measured and which it
  carried.
* **The checksum machinery does not change.**  TABLE_CHECKSUM still
  binds the floors, the hash table, and STALE_SINCE as one unit;
  `--bless` still writes them together after a run.  The only change
  inside the bound constants is the hash table's per-family shape.

**The payoff.**  Under this mode, the padding refresh would have been
`--families cone,parallel`: about a third of the full plan's arms,
estimated at 3 to 5 GPU-hours against the full run's 12.  mg26
evidenced the premise from the tool itself: after the run, `--bless`
named the cost inputs that moved since mg22 as `triton_cone.py` and
`triton_parallel.py` alone, exactly the two-family scope.  The
follow-ups in view are scoped the same way: a B7 forward-kernel
change touches parallel (and cone, sharing `triton_cone.py`'s
helpers), and a B6 multiaxis remedy touches multiaxis only.  The
mode also shrinks what the E2 automation would submit on a nightly
staleness hit.

**Three gaps in today's hash set, found while designing the
partition.**  These are observations for the ruling, not decisions:

1. `_utils.py` is not hashed, and `padded_kernel_width` lives there.
   The padding tripped the note only because the kernel wrappers
   changed in the same commit.  A later edit to the rounding rule
   alone would not trip it.  The shared set above adds `_utils.py`.
2. The geometry body files (`parallel_beam.py`, `cone_beam.py`,
   `multiaxis_parallel.py`, `translation_model.py`) are not hashed,
   so a change to a torch body's cost never trips the note.  The
   partition makes adding them cheap (each file goes to its own
   family); whether to add them is part of the ruling.
3. The denoiser family has no cost inputs at all in today's set: no
   file hashed prices a denoise.  Its rows are sentinels, so the
   exposure is low, but the partition should either give it a set
   (`denoising.py`, `qggmrf.py`) or record why it has none.

## 4. What this proposal does not change

The floors' meaning, the guard's consultation path, the capacity
override, and the pin mechanisms are untouched.  The refresh
protocol per arm (one cold pass, warm median of three, fresh
subprocess, staged sinogram) is untouched.  E2's automation questions
stay open; this proposal only shrinks what an automated refresh would
run.

## 5. The ruling

* **(a) Approve the coarse table (§2)** as pasted rows, with the
  margin rule recorded in the module docstring.
* **(b) Approve the family-scoped refresh mode (§3)**: the
  `--families` flag, the per-family cost-input sets, the
  carry-refusal rule, and the three gap closures (or name which gaps
  stay open).
* **(c) Or direct otherwise** -- the current fine-grained table and
  whole-table refresh keep working as they are; nothing here is
  urgent except the GPU-hours the next refresh costs.
