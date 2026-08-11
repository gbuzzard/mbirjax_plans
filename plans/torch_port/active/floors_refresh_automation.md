# Automating the widening-floors refresh — plan

**Status:** DRAFT 2026-08-11, for the checkpoint to rule.  This is
`multigpu_plan.md` step 8, "Move the floors refresh into the nightly",
which Greg directed on 2026-08-10 to be written as a short plan of its
own, coordinated with the nightly's owner.  The design it settles is
where the automation runs and how its output reaches the floors file.
The mechanism it imitates is the dependency watch of
`python_matrix_nightly_check.md`, which went live on 2026-08-10 and runs
nightly today.

## 1. Terms

- **The floors**: the measured table in `mbirtorch/_widening_floors.py`.
  A floor is the sinogram size, in `prod(sinogram_shape)` elements, at
  or above which a device count is worth using.  The floors govern the
  automatic device-count choice only.
- **The bound constants**: `FLOORS`, `BLESSED_COST_HASHES` and
  `STALE_SINCE`, together with the `TABLE_CHECKSUM` that binds them.
  These four move as one unit or not at all.
- **The cost inputs**: the four files and three `TomographyModel`
  methods named in `COST_INPUT_FILES` and `COST_INPUT_METHODS`.  Their
  sha256 digests are what `BLESSED_COST_HASHES` records.
- **A drift**: a cost input whose current digest differs from its
  recorded one.  A drift means the floors were measured against code
  that has since changed.
- **The staleness note**: the sentence `_widening_floors.stale_note()`
  returns on a drift.  It is printed in every automatic device
  selection until the drift is cleared, and
  `tests/test_widening_floors.py` prints it, warns, and passes.
- **The refresh**: one run of `dev_scripts/refresh_widening_floors.py`
  on a four-GPU node.  It re-measures the bracket arms and prints a
  paste-ready block.
- **The block**: the refresh's output, which is a complete replacement
  for the four bound constants.  The script prints it today, and §3.4
  proposes a mode that writes it into the file instead.
- **The measuring job**: NEW, a scheduled slurm job on gautschi that
  detects drift and runs the refresh.
- **The proposing workflow**: NEW, a scheduled GitHub Actions workflow
  in `cabouman/mbirtorch` that turns a measured block into a pull
  request.

## 2. What this plan does, and why now

The refresh is deliberate work that nothing schedules.  Someone must
notice a drift, hold a four-GPU node for half an hour to an hour, run
the script, and paste its block.  This plan schedules the noticing and
the measuring, and leaves the merging to a human.

The cost inputs are now changing more than once a week, which is what
makes the manual cadence expensive.  Two refreshes ran this week.  The
2026-08-11 refresh re-measured every row under the column-gather
forward and moved the parallel four-device floor down by a factor of
3.4 in sinogram elements (`multigpu_findings.md` §1.12).  The
copy-stream commit landed days later, changed `_sharding.py` and the
sharded drivers, and re-fired the staleness note
(`multigpu_findings.md` §1.13).  Each such change re-fires the note for
every user of the branch it lands on, and the note stands until someone
runs the refresh by hand.

The automation's job is to shorten the stale window, not to guard
correctness.  Correctness is already guarded, and it is guarded by the
note.  A stale floor still admits or holds a device count, and the
worst it costs is a few percent on a mid-size run.  The reason is
recorded in `multigpu_findings.md` §3.3.  A faster projection moves
knees down, so today's floors hold a count back where it newly pays.
The harm the guard prevents is the 3x-to-13x direction, and stale
floors are on the safe side of it.

That framing sizes everything below.  The automation may fail, skip
nights, and defer to other work, and no reconstruction is wrong as a
result.  So this plan buys reliability only where reliability is cheap,
and it declines the proof-of-life machinery the dependency watch needed.

## 3. Design

### 3.1 Where the measurement runs: gautschi, and nowhere else

The measurement has exactly one possible host.  A refresh needs four
H100s for 30 to 60 minutes, so it cannot run on GitHub's runners and it
cannot run on the Macs.  Greg ruled scheduled work off the Macs on
2026-08-10, and `python_matrix_nightly_check.md` §3.4 and §3.6 both
rest on that ruling.  gautschi is where the four-GPU allocation exists,
so the measuring job runs there, as its own scrontab block.

The proposal cannot run there.  Opening a pull request needs a GitHub
credential, and the dependency watch stores none.  It runs inside
Actions with the built-in token, which GitHub mints for each run and
expires when the run ends (`python_matrix_nightly_check.md` §3.5).  A
cluster job that pushed to `cabouman/mbirtorch` would need a stored
token that does not exist today.

So the automation is two actors, split along the line the constraints
draw.  The cluster does the thing only the cluster can do, which is
hold four GPUs and measure.  GitHub does the thing only GitHub can do,
which is open a pull request and email both maintainers.  The measured
block travels between them through `mbirjax_metrics`, a repository the
cluster can already push to and GitHub can read without a credential.

This is a refinement of the candidate shape rather than a fourth
candidate.  The candidate was a nightly job that checks the hash
cheaply and, on a drift, measures and proposes.  Every part of that
survives, and only the actor performing the last step changes.

### 3.2 Which branch the automation watches

The automation watches `prerelease`, and proposes against `prerelease`.
Three reasons, in order of weight.

`prerelease` is where the floors that ship are decided.  Every change
reaches `prerelease` before `main`, so a drift is visible there at the
earliest moment it is real.  Watching `main` would re-measure only
after a release merge, which is the latest possible moment.

Watching a working branch would measure code that is about to change.
Both 2026-08-11 drifts arose on `greg_dev`, and a refresh measured
there would be superseded by the next merge.  The dependency watch
targets `prerelease` for the same reason, and one convention is easier
to reason about than two.

One consequence must be stated.  A developer working on `greg_dev` sees
the staleness note until the change merges, and this automation does
not shorten that.  That is the note working as designed, because the
floors on that branch really were measured against different code.

### 3.3 The measuring job

The measuring job is one scrontab block on gautschi, at 05:00, with
four GPUs and a two-hour walltime.  It is a sibling of the two nightly
blocks and shares no mutable state with either.  Its steps are these.

1. **Read the kill switch.**  An `ENABLED` flag in the job's own
   configuration file, matching the nightlies' convention.  A disabled
   job prints one line and exits.
2. **Check for competing work.**  The job reads `squeue -u buzzard` and
   exits without measuring if any job of ours is queued or running.
   §3.7 gives the rule and its limits.
3. **Clone `prerelease` from origin.**  A fresh clone, as the nightly
   does, so the commit the refresh stamps is a real origin commit.  The
   staged campaign tree at `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`
   is not a git checkout, and `head_commit()` would return `unknown`
   there, which the provenance test rejects.
4. **Check for drift.**  Import `_widening_floors` from the clone and
   call `stale_cost_inputs()`.  This hashes four files and three method
   sources and takes milliseconds.  An empty result ends the run.
5. **Preflight the hardware.**  Assert that exactly four CUDA devices
   are visible and that their name matches `MEASURED_GPU`.  §3.6 says
   why this assertion is load-bearing rather than decorative.
6. **Run the refresh.**  `dev_scripts/refresh_widening_floors.py`,
   unmodified in what it measures, with the `--write` mode of §3.4.
7. **Publish the result.**  Commit one artifact file into
   `mbirjax_metrics` and push it, using the credential the nightly
   already holds.
8. **Print one status line** into the job's log, whatever happened.

The job uses its own scratch directory and its own environment.  The
environment is a venv layered over `mbirtorch_regression` with
`--system-site-packages`, into which the clone is installed with
`pip install -e <clone> --no-deps`, which is the recipe in
`cluster_use.md`.  Installing into `mbirtorch_regression` directly
would risk the recorded failure where two jobs `pip install -e` into
one environment and one reads a torn `.pth` file.

The walltime request must not be sized from the current table alone,
because the arm list is read off that table and moves when a floor
moves.  The 2026-08-11 refresh measured 33 minutes, and the plan it ran
reported 24 timed arms and 8 generators.  Running `--plan` against
today's table reports 27 timed arms and the same 8 generators, because
the parallel four-device floor moved down and its bracket gained a cell
below it.  A second effect will grow the list again.  `--plan` reports
that `MultiAxisParallelModel` and `TranslationModel` declare no floor
family, and measuring floors for either one adds rows.

One reporting detail matters for the same estimate.  `print_plan` sums
the counts over plan rows, while `run_plan` measures the distinct
`(family, cell, count)` arms, and the two differ because bracket cells
overlap between rows.  Today's plan reports 27 and would execute 21.
The walltime should be requested against the executed count and against
the largest plan the table can produce, not against the printed figure.

The cost is about 2.7 GPU-hours per refresh, from four GPUs held for
roughly 40 minutes.  The nightly costs about 2 GPU-hours on a
changed-branch night.  A night with no drift costs the queue time plus
seconds.

### 3.4 The one script change: the refresh writes the file

The refresh script must gain a mode that writes `_widening_floors.py`
rather than printing a block for a human to paste.  Automation cannot
paste.  The alternative is worse than the script change.  A workflow
that parsed the block and edited the file would make the automation a
second writer of the floors, which is what the sole-writer rule
forbids.  A `--write` mode keeps the script the sole writer literally,
and the automation only runs it.

The mode has one hazard, and naming it is the point of this section.
`bless_lines()` computes `TABLE_CHECKSUM` over the `FLOORS` it has
imported, not over the floors the run just measured.  The documented
manual workflow depends on that ordering.  A human pastes the new
`FLOORS` first, then runs `--bless` in the edited tree, and only then
does the checksum describe the new table.  A `--write` mode that
composed the two printers in the wrong order would emit a checksum over
the old floors, which would fail the checksum test on arrival.  The
mode must rewrite `FLOORS` and then compute the checksum over the new
rows in one pass.

The mode changes nothing about the measurement.  The ladder, the
crossover rule, the fresh-subprocess-per-arm discipline, and the
one-sinogram-per-cell protocol are untouched.

### 3.5 The proposing workflow, and how a block becomes a pull request

The proposing workflow is a scheduled GitHub Actions workflow in
`cabouman/mbirtorch`, modelled directly on `dependency_watch.yml` and
running a few hours after the cluster window.  It performs six actions.

1. It reads the newest artifact published by the measuring job.  The
   artifact names the commit it measured, carries the rewritten file,
   and carries the verdict table, the slurm job id, and the wall time.
2. It cuts a branch from the measured commit, named
   `nightly/widening-floors-<sha>`.  Cutting from the measured commit
   is what makes the recorded hashes describe the code they were
   measured against.  The dependency watch cuts from the commit it read
   for the same reason.
3. It applies the rewritten file to that branch.
4. It verifies before proposing anything.  It runs
   `tests/test_widening_floors.py` on the branch, which is where the
   hard checksum assertion and the provenance assertions live.  A
   failure opens no pull request and writes the reason into the run
   summary.  This converts trust in the artifact into a check on it.
5. It opens ONE pull request against `prerelease`, authored by
   `github-actions[bot]`, with review requested from `cabouman` and
   `gbuzzard`.  The review requests are what make GitHub send the
   email.  The body names the measured commit, the drifted cost inputs,
   the verdict for each row, and the retained human step of §3.6.
6. It dispatches `ci.yml` on the branch, because events caused by the
   built-in token start no workflow runs and the pull request would
   otherwise show no checks.

The duplicate rule is the dependency watch's, keyed on the pull
request.  A pull request in any state for a branch is a standing
answer.  Closing one unmerged is a "no" that the automation does not
reopen.

One case is benign and should be expected rather than guarded against.
If `prerelease` moves past the measured commit before the pull request
merges, the merged result's floors may not describe the merged code.
The staleness note then fires again, correctly, and the next night's
run proposes a new measurement.  The automation's own failure mode is
therefore the condition it exists to detect, which is the cheapest
possible arrangement.

### 3.6 The retained human step: the notes

Every proposed row arrives with the literal placeholder `note='...'`,
because `_ROW_TEMPLATE` prints it.  The script's closing line says why.
A note is the one field a machine cannot fill in.  The note is where a
reader learns whether a floor moved, what it moved against, how much
margin the win had, and what the bracket does or does not establish.
The 2026-08-11 rows are the standard to meet.

So the human step is writing the notes, and it happens at merge time on
the pull request.  This mirrors the dependency watch, which also leaves
the merge to a maintainer.  The reason is the same in both cases.  A
machine can compose the change and prove it consistent.  It cannot
judge it.

The step should be enforced rather than assumed.
`test_every_floor_carries_the_provenance_to_re_measure_it` asserts
`floor.note` is non-empty, and the placeholder passes that assertion.
One added line rejecting the literal placeholder would turn the human
step into a red check on any pull request that skipped it.

Two smaller fields also need a human eye at merge time.  `MEASURED_GPU`
and `MEASURED_CONFIG` are constants the refresh does not rewrite, and
the script prints a reminder to confirm they still describe the run.
The §3.3 hardware preflight makes that confirmation cheap, because a
run that did not get four H100s never publishes an artifact.

That preflight matters more than it looks, because a refresh on a
two-GPU allocation does not fail loudly.  Its four-device arms error,
the verdict records no measurement, and `print_table` then proposes
`elements=None`.  Such a proposal would convert two working floors into
sentinels and exclude four devices everywhere.

### 3.7 Deferring while campaign work is in flight

The stated worry is that a refresh mid-campaign changes the floors
under a running comparison.  The premise needs one correction before
the guard is designed.  Almost no campaign arm consults the floors.
Every mg arm and every nightly row pins the count through
`MBIRTORCH_NUM_DEVICES`, and a pin bypasses the guard by construction.
`_widening_floors.py`'s own docstring states that.  The refresh pins
every arm too.  So a floors change does not move what a pinned
measurement measures.

Three real interactions remain, and they justify the guard on their own
terms.  A campaign arm that deliberately exercises the automatic path
does consult the floors, and `multigpu_findings.md` §3.4 is the readout
of exactly such arms.  A merged floors change moves the tip of the
branch the campaign measures against, and the nightly then attributes
the move to a changed tip.  And a four-GPU refresh competes for the
same account, queue, and node class as the campaign's own four-GPU
jobs, which is the contention `cluster_use.md` says to coordinate
around.

The guard is therefore a deferral, not a lock.  The measuring job reads
`squeue -u buzzard` and exits without measuring when it sees any of
four job names: one beginning with `torch_mg`, `mbirjax-nightly`,
`mbirtorch-nightly`, or an interactive job.  The `torch_mg` prefix is
how every campaign harness names itself, and all thirteen `mg` batch
files set it.  Deferring costs one night of stale window and nothing
else.

The name check is a cheap guard and not a complete one.  Campaign work
also happens inside Greg's interactive sessions and continues after a
job finishes, and neither state is visible in a job name.  The complete
guard is the `ENABLED` kill switch of §3.3, which a campaign session can
clear for the duration of a comparison.  Stating both is more honest
than claiming the automatic check is sufficient.

### 3.8 Failure isolation

The automation must never break or delay the nightly it runs beside,
and separation is what makes that structural rather than careful.  The
measuring job is its own scrontab block, with its own work directory,
its own clone, its own venv, its own log, and its own mail identity.
`nightly_plan.md` §3(a) made the same argument when it declined to put
the torch measurements inside the jax nightly's job.  Its evidence was
the 2026-07-10 cascade, in which crashed workers left the GPUs unusable
for the engine that ran after them.  Two jobs cannot do that to each
other.

The 05:00 slot sits after both nightly blocks, and the §3.7 squeue
check enforces the ordering rather than trusting the clock.  The torch
nightly's walltime ceiling is four hours against a measured pass of
about 30 minutes, so a clock-only rule would either start too late
every night or occasionally overlap.

The proposing workflow isolates its own failures the same way the
dependency watch does.  A failed run writes a failed run and costs
nothing lasting, because the artifact is still there tomorrow and the
drift is still there tomorrow.

### 3.9 Confirmation delay: not needed, and what replaces it

The dependency watch acts only on a divergence seen on two consecutive
nights.  That rule exists because its signal is an external package
index that can be read mid-rollout, so a partial upload can look like a
divergence without being one.

The floors signal has no such failure.  It is a sha256 of files and
method sources in the repository at a named commit.  The same commit
always produces the same answer, and there is no external state being
sampled.  A second night would filter nothing.

The costs also run the other way.  The dependency watch's check takes
15 seconds, so waiting a night is nearly free.  A refresh is 30 to 60
minutes on four GPUs, and a confirmation delay would spend a day of the
stale window that the automation exists to shorten.

So the recommendation is no confirmation delay.  What the checksum
binding does and does not buy should be stated precisely, because it is
easy to overstate.  The checksum makes a block that does not match its
commit fail a hard test on the pull request, so a mismatched, stale, or
hand-tampered block cannot land green.  It says nothing about whether
the measurement was right.  Measurement quality rests on the crossover
rule instead, which counts a win only when it clears 1.0x by more than
that cell's warm spread.

A different control is needed, and it is a rate limit rather than a
delay.  Cost inputs are changing several times a week, so an unlimited
automation would open a proposal on many of those nights.  Each
proposal costs 2.7 GPU-hours and a maintainer's attention.  The rule is
one open proposal at a time.  While a floors pull request is open, the
measuring job checks for drift, reports what it found, and measures
nothing.  Whether a minimum interval between measurements is also
wanted is an open question in §7.

### 3.10 Silent death, and why this needs no watchdog line

An automation that stopped running looks like an automation that found
nothing, and the dependency watch needed a whole watchdog line to tell
the two apart.  This one needs none, and the reason is worth stating
rather than assumed.

A dead floors automation degrades to today's manual state, and today's
manual state is visible.  The staleness note prints in every automatic
device selection and in every run of `tests/test_widening_floors.py`.
Nothing in this plan touches that surface, so silence in the automation
never reads as freshness in the floors.

The cheap half is still worth having.  The measuring job prints one
status line whatever it found, so its log answers three questions in
one place: whether it ran, what the drift check said, and what it did.
A second, independent watchdog on top of that is declined as
disproportionate.

## 4. The two shapes declined, with reasons

### 4.1 Riding the nightly wrapper as a conditional step

**Declined.**  This shape would add a refresh step to
`run_torch_regression.sh`, which already has a four-GPU allocation, a
fresh clone, and an installed environment on a changed-branch night.
That saving is real, and it is a few minutes.  Four costs outweigh it.

It reverses a decision the nightly plan made deliberately.
`nightly_plan.md` §3(a) chose a separate job specifically so that
failure isolation would be structural, and a refresh inside the
nightly's job puts the refresh's failure modes inside the nightly's
job.

It makes the nightly's duration unpredictable.  The measured torch pass
is about 30 minutes against a four-hour ceiling, and a refresh would
double or triple it on the nights it ran.  The nightly's value is
regression protection that arrives every morning.

It watches the wrong branch.  The nightly tracks `main` alone, and §3.2
gives the reasons the floors should be watched on `prerelease`.
Changing the nightly's tracked branches to serve this plan would change
what the regression series measures, which is a much larger decision.

It still cannot open the pull request.  The credential argument of §3.1
does not depend on which cluster job does the measuring.  So this shape
removes one scrontab block and keeps every other moving part of the
design.

Four of the wrapper's conventions are worth borrowing regardless: the
bootstrap re-exec, the fresh clone, the fire-on-change check, and the
alert mail.  All four are proven on real nights, so the measuring job
should follow them rather than invent its own.

### 4.2 Staying manual, with a printed reminder in the nightly log

**Declined.**  This shape adds a line to the nightly report saying the
floors are stale and leaves the refresh to a human.

Charlie's finding decides it.  `python_matrix_nightly_check.md` §2
records that a line in a report will never be noticed, and that the
alert must arrive by email and state exactly what to do.  That finding
is the reason the dependency watch opens a pull request rather than
printing a divergence.

The staleness note is already this shape, and it is already stronger.
It prints in every automatic device selection and in the test suite,
which is more places than a nightly log reaches.  Both refreshes this
week still waited on a person deciding to run one, so printing has
never been the missing part.  The missing part is someone holding a
four-GPU node for half an hour.

This shape is the honest fallback.  If the automation proves more
trouble than it saves, this is where the work returns, and the cost is
a longer stale window and nothing else.

## 5. File inventory

### mbirtorch

| file | change |
|---|---|
| `dev_scripts/refresh_widening_floors.py` | a `--write` mode that rewrites the four bound constants in one pass, per §3.4; no change to what is measured |
| `tests/test_widening_floors.py` | reject the literal placeholder note, per §3.6 |
| `.github/workflows/floors_refresh.yml` | NEW — the proposing workflow: the schedule, the three permission grants, and the calls into the composer |
| `ci/floors_refresh.py` | NEW — read the artifact, cut the branch, apply the file, compose the pull request title and body, dry-run mode; pure logic, unit-testable |
| `ci/test_floors_refresh.py` | NEW — unit tests on a saved artifact, including a mismatched commit, a placeholder note, a failed verification, and an already-open proposal |

### mbirjax_metrics

| file | change |
|---|---|
| `tooling/regression/run_floors_refresh.sh` | NEW — the measuring job's wrapper: kill switch, squeue check, clone, drift check, hardware preflight, refresh, publish, status line |
| `tooling/regression/floors_refresh.env` | NEW — `ENABLED`, the work directory, the watched branch, the walltime, the mail address |
| `tooling/regression/enable_floors_refresh.sh`, `disable_floors_refresh.sh`, `status_floors_refresh.sh` | NEW — the scrontab block, matching the two existing nightlies' scripts |
| `results/widening_floors/` | NEW — the published artifacts, one per measured commit |

### mbirjax_plans

This file.  `multigpu_plan.md` step 8 gains a pointer to it, and a
status line when the automation is live.

## 6. Implementation increments

Each increment ends in a state that can be checked, and no schedule is
installed before the increment before it passes.

1. **The `--write` mode and its test.**  Add the mode to the refresh
   script.  Verify on the Mac with `--smoke`, which needs no GPU: run
   the smoke, write the file, and confirm that
   `tests/test_widening_floors.py` passes against the rewritten file,
   including the checksum assertion.  The smoke's floors are not real
   floors, so this proves the writer and not the measurement.  Revert
   the file afterward.
2. **The placeholder-note test.**  One line, red against a written
   file whose notes are untouched, green after they are written.
3. **The measuring job, without publishing.**  Write the wrapper and
   its configuration.  Run it once by hand on gautschi with publishing
   disabled.  It must clone `prerelease`, report the drift check, pass
   the hardware preflight on four H100s, complete the refresh, write
   the artifact locally, and print its status line.  This run also
   replaces the estimate in §3.3 with a measured wall time.
4. **The proposing workflow, dry-run.**  The workflow lands running the
   composer with `--dry-run`: it prints the branch name, the applied
   diff, and the pull-request title and body into the run summary, and
   writes nothing.  Its unit tests run in CI.
5. **Enable.**  Install the scrontab block, turn on publishing, and
   switch the workflow to acting mode.  The first real proposal is the
   trial.  The dependency watch's increment 4 records Charlie's ruling
   that a separate staged trial is not worth its ceremony when the
   worst outcome is a harmless pull request.  That reasoning applies
   here unchanged.
6. **Read the first two weeks.**  Confirm three behaviors on real
   nights: that a night with no drift costs seconds, that the squeue
   check defers rather than measuring beside campaign work, and that
   the one-open-proposal rule holds when cost inputs change twice in a
   week.

## 7. Open questions for the checkpoint

**Is `prerelease` the right branch to watch?**  §3.2 recommends it, and
the alternative is `main`.  The trade is between refreshing at the
earliest moment a change is real and refreshing only what has shipped.
The recommendation follows the dependency watch, which also targets
`prerelease`.

**Should the refresh script gain a `--write` mode at all?**  It changes
the script's contract from printing to writing, and the sole-writer
rule is what makes the checksum meaningful.  §3.4 argues that the
script writing its own output preserves that rule and that the
alternative breaks it.  This is the one change to a file the campaign
treats as settled, so it deserves an explicit ruling.

**Is one open proposal at a time enough of a rate limit, or is a
minimum interval also wanted?**  §3.9 recommends the first and leaves
the second open.  A minimum interval of, say, four days would bound the
GPU cost during a week like this one, at the price of a longer stale
window.  The answer depends on how much of the tuning work remains.

**Does the artifact travelling through `mbirjax_metrics` widen the
trust boundary in a way that matters?**  The dependency watch declined
a design where mbirtorch's watch took its CODE from `mbirjax_metrics`,
because that added the second repository's writers to the boundary.
Here only DATA crosses, the two repositories have the same writers
today, the proposal is verified by mbirtorch's own tests before it is
opened, and no change lands without a human merge.  The judgment is
that the boundary does not widen; a maintainer should confirm it.

**Who owns the mail when the measuring job fails?**  The nightlies use
`--mail-type=FAIL` with a job name that identifies the backend.  A
failing floors job is not urgent, because the note is the guard.  Three
options follow: a mail per failure, a mail after several consecutive
failures, or no mail and the status line alone.

**Should a refresh be forced on a schedule, independent of drift?**  The
floors go stale for reasons no hash can see, and `STALE_SINCE` exists
for exactly those reasons: new hardware, a changed subset schedule, a
different iteration count.  Nothing in this plan detects them.  A
quarterly forced refresh would, at four refreshes a year and about 11
GPU-hours.

**How does this interact with step 9 of `multigpu_plan.md`?**  Step 9
proposes simplifying the floors for robustness, with fewer and coarser
thresholds that survive shape and hardware variation.  A coarser table
would drift less often and would need this automation less.  Building
the automation first is still defensible, because the automation's cost
is bounded and the simplification has no date, but the sequencing is a
ruling rather than an assumption.
