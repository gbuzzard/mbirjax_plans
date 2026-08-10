# Python and torch version watch — plan

**Status:** REVISED 2026-08-10, after Greg reviewed the first reviewed
draft.  Greg ruled three ways: no machine account; the watch does not
run on the Macs; torch joins the scope now rather than in a follow-on
plan.  This revision moves the watch into `cabouman/mbirtorch` as a
scheduled GitHub Actions workflow, keeps the nightly as an independent
witness, and folds the torch floor into the same watch.  The detection
logic, the pull-request shape, and the veto rule survive from the
reviewed draft.  The panel record at the bottom is kept, with a note on
the findings this revision resolves differently.

## 1. Terms

- **The watch**: a scheduled GitHub Actions workflow in
  `cabouman/mbirtorch`, running nightly on GitHub's own runners.
- **The version file**: `.github/python-versions.json`, NEW, the single
  place the tested Python versions are written down, including the docs
  job's Python.  `ci.yml` reads its test matrix from this file at run
  time.  The file exists so that automated edits never touch a workflow
  file; §3.5 gives the reason.
- **The matrix**: the version list in the version file.  These are the
  versions CI tests.
- **The torch list**: the CPython versions for which torch publishes
  Linux x86_64 wheels on its CPU index, in its newest stable release.
  The CPU index is `https://download.pytorch.org/whl/cpu/`, and it is
  the source the watch reads because it is the index CI installs torch
  from.  (The panel found that PyPI, the first draft's source, is a
  different distribution channel whose Python coverage differs from the
  CPU index's today.)
- **The floor**: the `requires-python` lower bound in mbirtorch's
  `pyproject.toml` (today `>=3.11`).
- **The torch floor**: the torch lower bound in `pyproject.toml`
  (today `torch>=2.13`).
- **A divergence**: either the matrix differs from the torch list,
  restricted to versions at or above the floor, or the newest stable
  torch minor on the CPU index is above the torch floor.  An addition
  is due when the torch list has a Python version the matrix lacks.  A
  removal is due when the matrix has a version the torch list lacks.
  Python versions below the floor are reported as an informational line
  and never acted on.

## 2. What the watch does, and why

The accepted policy is that mbirtorch tests the Python versions torch
supports, and advances its torch floor deliberately.  The maintenance
problem is that someone must notice when torch's side changes.  Charlie
has said a line in a report will never be noticed.  The alert must
arrive by email and state exactly what to do.

The watch runs every night on GitHub's scheduler.  It compares the
matrix and the torch floor against the CPU index.  On a divergence, it
opens ONE pull request in `cabouman/mbirtorch` containing the complete
edit for that divergence.  The pull request's review requests make
GitHub email both maintainers.  CI check results appear on the pull
request; §3.2 says how.  A maintainer reads the diff and clicks Merge.

The safety property, stated precisely: nothing reaches `prerelease` or
`main` without a human merge.  The watch does push branches named
`nightly/*` automatically; those branches carry proposals, not accepted
code.

## 3. Design

### 3.1 Detection

The watch reads the torch list from the CPU index, which is a standard
package index in the PEP 503 HTML format.  The steps are these.  Find
the newest stable torch release; pre-release and yanked files are
ignored.  Keep only that release's Linux x86_64 wheel files, because
that is the platform CI runs on.  Take each wheel filename's Python
tag, which is the third dash-separated field; for example `cp312`
means CPython 3.12.  Variant tags such as `cp314t` (the free-threaded
build) are not versions and are skipped, as are non-wheel files.  Every
parsed version must match the pattern `3.<digits>` before it is used
anywhere; a non-match is a logged failure, not an action.

The matrix, the floor, and the torch floor are read from the version
file and `pyproject.toml` on the `prerelease` branch of
`cabouman/mbirtorch`.  The watch records the commit identifier it read,
and every later step works from that same commit.

To keep a mid-rollout wheel upload from triggering a false alarm, the
watch acts only on a divergence it has observed on two consecutive
nights.  The state this rule needs is one small file naming the
divergence seen last night, and the workflow keeps that file in the
Actions cache.  A lost cache entry resets the count.  The cost of a
reset is one extra night of delay, never a false alarm.

### 3.2 The pull request

On a confirmed divergence, the watch performs five actions.

1. It creates a branch from the recorded commit, named for the whole
   divergence: for example `nightly/python-matrix-add-3.13-3.14`, or
   `nightly/torch-floor-2.14`.  One divergence produces one branch and
   one pull request, however many versions it involves.
2. It edits every file that records a version, and every such file is a
   data file.  A Python addition edits the version file.  A Python
   removal edits the version file and the floor in `pyproject.toml`.  A
   torch floor advance edits the torch floor in `pyproject.toml`.  No
   edit touches `ci.yml` or any other workflow file, by the §3.5
   design.  The commits and the pull request are authored by
   `github-actions[bot]`, so automated changes are recognizable at a
   glance.
3. It opens a pull request with base `prerelease`.  The title names the
   change ("Add Python 3.13 and 3.14 to the CI test matrix").  The body
   names the torch release that triggered it, the commit the branch was
   cut from, and the policy.  A removal's body additionally says that
   CI cannot prove a removal correct, so the reviewer must judge it.  A
   torch floor advance's body additionally says that the green checks
   cover the CPU suite, and that the Triton and CUDA paths are proven
   by the cluster nightly after the merge.  That division is the same
   one every ordinary pull request lives with today.
4. It dispatches `ci.yml` on the new branch.  This step exists because
   of a GitHub rule: events caused by the built-in token start no
   workflow runs, so the pull request alone would show no checks.
   GitHub documents two exceptions to that rule, and explicit dispatch
   is one of them.  The dispatched run's check results attach to the
   branch's head commit, so they appear on the pull request and satisfy
   any required-checks protection.  `ci.yml` needs the
   `workflow_dispatch` trigger for this; §5's one-time change adds it.
5. It requests review from `cabouman` and `gbuzzard`.  The pull
   request's author is `github-actions[bot]`, not either maintainer, so
   both can be requested reviewers and both receive GitHub's email.
   The reviewed draft's machine account existed to make exactly this
   true; the bot author makes it true with no account.

The pull request targets `prerelease` because every change is merged
into `prerelease` first.

One consequence deserves a sentence: this is a same-repository pull
request, so a dispatched CI run executes the branch's own copy of
`ci.yml` with repository secrets in scope.  The reviewer must read the
diff, not only the check result.  Today the repository holds no
secrets; when `RTD_TOKEN` is added for releases, it should be scoped to
a GitHub environment restricted to `main`, not stored as a plain
repository secret.

### 3.3 One pull request per divergence, and a visible veto

The branch name is determined entirely by the divergence, so the same
divergence always produces the same branch name.  The watch's duplicate
rule keys on the pull request, not the branch alone.  If a pull request
for the branch exists and is open or merged, the watch does nothing.
If the branch exists but has no pull request — the leftover of a partly
failed night — the watch repairs the state by opening the pull request.
If the pull request was closed without merging, that is the
maintainers' standing "no", and the watch does not reopen it; deleting
the branch withdraws the "no".

This veto is also how a conservative torch policy stays cheap.  If the
maintainers prefer to batch torch floor advances yearly, they close the
eager pull request, and the watchdog line then reports the standing
"no" nightly until the batch lands.

A suppressed divergence must stay visible.  That duty belongs to the
watchdog line, which §3.4 defines.

### 3.4 Silent death, and the watchdog

A watch that stopped running is indistinguishable from a watch that
found nothing, so the ways this watch can go silent are named here.
GitHub disables a scheduled workflow after sixty days without
repository activity, with a warning email first.  Scheduled runs are
delivered best-effort and can be delayed.  A failing scheduled run
emails only the account that last touched the workflow file.  None of
these surfaces is one the maintainers watch daily.

The answer is one report line in the gautschi nightly, called the
watchdog line.  The gautschi nightly re-runs the pure checker (§5,
increment 1) and reads the repository's open and closed pull requests.
Both reads are unauthenticated, because the repository is public and
the watchdog writes nothing.  The line states three facts separately:
the divergence the checker found, the pull-request state it found, and
the verdict.  Three verdicts exist.  "No divergence" is the quiet
nightly case.  "Divergence, and its pull request is open or merged" is
the watch working.  "Divergence, and no open or merged pull request" is
the alarm: the watch is broken, disabled, or vetoed, and the line uses
the closed-pull-request record to tell a veto from a failure.  The
watchdog runs on gautschi and not on the Macs.  Greg ruled the slurm
jobs the more reliable host, and the watchdog is the reliability
backstop.

The watch still isolates its own failures.  A checker bug on one night
writes a failed run and costs nothing lasting, because a real
divergence is still there the next night.

### 3.5 Credentials: none stored

The reviewed draft's machine account is declined, and nothing replaces
it.  The workflow runs with the repository's built-in token, which
GitHub mints for each run and expires when the run ends.  The
`permissions` key grants it exactly three abilities: `contents: write`
to push `nightly/*` branches, `pull-requests: write` to open the pull
request and request reviewers, and `actions: write` to dispatch
`ci.yml`.  There is no token to store, leak, rotate, or count down to
expiry, so the reviewed draft's storage rules and credential-expiry
countdown are retired with the account.

The three GitHub facts the panel established still hold, and the bot
author satisfies them.  A pull request's author cannot be one of its
requested reviewers, and GitHub does not email an account about its own
actions; the author here is `github-actions[bot]`, so both maintainers
are reviewers and both are emailed.  The fine-grained-token fact is
moot, because no token is issued.

One further GitHub rule shapes the design.  The built-in token cannot
create or update files under `.github/workflows/`, so the watch could
never push an edit to `ci.yml`.  The version file exists for this
reason.  A one-time human change (§5, increment 2) converts `ci.yml` to
read its matrix from the version file, and every automated edit
thereafter touches data files only.

Blast radius, stated plainly: a compromise of the workflow's run can
push to unprotected branches for the life of that run, and nothing
after it.  The standing controls are unchanged: `main` accepts changes
only by pull request with passing CI, and publishing requires Charlie's
manual approval of the `pypi` environment.  The repository's default
workflow permissions should stay read-only, with the three write grants
made in this workflow's file alone.

One trust statement replaces the reviewed draft's.  The watch's code
lives in `cabouman/mbirtorch` itself, so the people who can change what
the watch does are exactly the people who can already write to the
repository the watch proposes changes to.  No second repository joins
the trust boundary.  The reviewed draft's watch updated its code from
`mbirjax_metrics`, which added that repository's writers to the
boundary.

### 3.6 Where the watch runs

The watch runs on GitHub's scheduler, in the repository it serves.
Three reasons replace the reviewed draft's three for the Mac.  The
check needs nothing local, because it reads a public index and the
repository's own files.  The actor needs GitHub credentials, and only
GitHub's own runner has an ephemeral one.  The Macs sleep and move, and
gautschi has maintenance windows and the filesystem-client failures the
campaign fights weekly, so GitHub's runners are the most reliable of
the three hosts for a nightly network check.  The sixty-day disable
rule and the best-effort scheduling are the price, and the §3.4
watchdog line is sized to catch both.

### 3.7 Torch scope

Torch joins the watch now, by Greg's ruling, rather than in a follow-on
plan.  One workflow computes both checks from the same CPU index read:
the Python matrix against the newest stable torch release's wheels, and
the torch floor against the newest stable torch minor.  One workflow
exists so the two checks stay coherent.  A new torch release can change
both answers at once, and one watch proposes one consistent pull
request instead of two conflicting ones.

A torch floor advance is proposed eagerly and vetoed cheaply.  The pull
request appears when the newest stable minor exceeds the floor, with
the CPU-only caveat in its body (§3.2).  Whether the maintainers merge
on arrival or batch advances yearly is a release-policy choice that
stays with them; the §3.3 veto records a "not yet" without silencing
the watch.  The `pyproject.toml` torch line already carries the intent
in its comment, "Re-test on each torch bump"; the watch is that intent
made mechanical.

## 4. File inventory

### mbirtorch

| file | change |
|---|---|
| `.github/python-versions.json` | NEW — the version file: the tested Python list and the docs job's Python |
| `.github/workflows/ci.yml` | one-time human edit: read the matrix and the docs Python from the version file; add the `workflow_dispatch` trigger |
| `.github/workflows/dependency_watch.yml` | NEW — the schedule, the three permission grants, and the calls into the checker |
| `ci/dependency_watch.py` | NEW — detection, the pull-request actions, duplicate suppression, dry-run mode; pure logic, unit-testable |
| `ci/test_dependency_watch.py` | NEW — unit tests on saved index and version-file fixtures, including a `cp314t` tag, a non-wheel file, a multi-version divergence, a below-floor version, and a torch floor advance |

### mbirjax_metrics

| file | change |
|---|---|
| the gautschi nightly wrapper | the watchdog line: re-run the pure checker, read the pull-request list, print the three facts and the verdict |
| the status script | print the watchdog line's last verdict |

### mbirjax_plans

This file.  `release_workflow.md`'s delivery-mechanism record gains the
same-day revision paragraph, and gains a status line when the watch is
live.

## 5. Implementation increments

Each increment is small and independently testable.

1. **The pure checker.**  Detection only: read the CPU index, the
   version file, and `pyproject.toml`; print the divergence or "none".
   Unit tests run on saved copies of the inputs.  No credential and no
   writes.  This is also the code the watchdog line runs on gautschi.
2. **The one-time `ci.yml` conversion.**  A human pull request adds the
   version file, converts the matrix and the docs job to read from it,
   and adds the `workflow_dispatch` trigger.  This pull request is
   human-authored, so CI fires on it normally and proves the conversion
   against the unchanged version list.
3. **The workflow, dry-run.**  The scheduled workflow lands running the
   checker with `--dry-run`: it prints the branch name, each edited
   file, and the pull-request title and body into the run summary, and
   writes nothing.
4. **The live trial.**  A forced divergence is made by removing 3.12
   from the version file on `prerelease`.  The workflow is dispatched
   by hand.  The trial must confirm six behaviors: the pull request
   appears, authored by `github-actions[bot]`; both maintainers receive
   the email; the dispatched CI run's checks appear on the pull
   request; closing it unmerged stops the watch from proposing the same
   change again; the watchdog line reports the standing "no" the next
   night; and deleting the branch withdraws it.  The trial pull request
   is then closed, its branch deleted, and the version file reverted.
5. **Enable.**  The `--dry-run` flag comes off the scheduled run, and
   the watchdog line is enabled in the gautschi nightly.  The first
   quiet week's watchdog lines are read once.

## 6. Greg's rulings, and what stays open

Three rulings on 2026-08-10 replace the reviewed draft's open
questions.  No machine account: resolved by §3.5, which stores no
credential at all.  Not on the Macs: resolved by §3.6 and §3.4, which
put the actor on GitHub's scheduler and the watchdog on gautschi.
Torch in scope now: resolved by §3.7.

One question stays open, for Charlie: merge torch floor advances on
arrival, or batch them yearly behind the veto?  The watch behaves
identically either way.  The answer only sets the expectation for how
long an eager torch pull request may sit closed.

## Panel review record (2026-08-10)

Four Opus reviewers — technical accuracy, failure modes, security, and
style — returned 49 findings on the first draft.  All are incorporated.
The findings that changed the design, rather than the wording:

- **Wrong index.**  CI installs torch from the CPU index at
  download.pytorch.org, not from PyPI, and the two channels' Python
  coverage differs today.  The watch now reads the CPU index and
  filters to the Linux x86_64 wheels CI actually uses.
- **Impossible credential mechanics.**  A fine-grained token works
  only from Charlie's account; an author cannot be a requested
  reviewer; GitHub does not email an account about its own actions.
  The first draft would have left Charlie — the person the email is
  for — unnotified.  Resolved by the machine account, whose token also
  needs the `workflow` scope the first draft omitted (the edited file
  is a workflow file).
- **Day-one multi-version divergence.**  The matrix is [3.11, 3.12]
  and torch's newest release covers 3.10 through 3.14, so the first
  run faces three candidate additions, one of them below the floor.
  The divergence definition now excludes below-floor versions, and one
  divergence produces one pull request covering all its versions.
- **Partial failure wedged the alert channel.**  Branch-existence as
  the only duplicate key turned any failure after the branch push into
  permanent silence.  The duplicate rule now keys on the pull request,
  repairs a branch that lacks one, and reports every suppressed
  divergence nightly.
- **The watch could die silently.**  Placed after the measurement
  work, it would not have run at all on unchanged nights, and no
  surface distinguished "ran, no divergence" from "did not run".  It
  now runs on the wrapper's unconditional path and carries the §3.4
  proof-of-life mechanisms, including the nightly credential-expiry
  countdown.
- **Unstated blast radius.**  The token can push to the unprotected
  `prerelease` and create releases; §3.5 now states this and names the
  two standing controls that bound it.
- **Incomplete edit.**  Removals must also update the docs job's
  hardcoded Python version in ci.yml, and their pull requests must say
  that CI cannot prove a removal correct.
- **A broken trial.**  The first draft's live trial used a scratch
  base branch, on which CI does not trigger; the trial now forces the
  divergence on `prerelease` itself.

**Revision note (2026-08-10, after Greg's rulings).**  Two of the
findings above are resolved differently by the revision than by the
reviewed draft.  The impossible-credential-mechanics finding forced the
machine account only while the watch lived outside GitHub; with the
watch inside Actions, the `github-actions[bot]` author restores both
maintainers as emailed reviewers with no account and no stored token.
The same finding's `workflow`-scope requirement is retired by the
version file, because no automated edit touches a workflow file any
longer; that rule is also enforced by the built-in token itself (§3.5).
The silent-death finding's proof-of-life machinery (the state file and
the credential countdown) is replaced by the §3.4 watchdog line, which
alarms from a second, independent host.  The remaining findings stand
and are carried in the revised text: the CPU index and the
two-consecutive-nights rule (§3.1), the below-floor exclusion and the
one-pull-request-per-divergence rule (§1, §3.1, §3.2), the
pull-request-keyed duplicate rule with repair (§3.3), the visible
suppression (§3.3, §3.4), the removal caveats with the docs-job Python
now carried in the version file (§3.2, §4), and the live trial on the
real base branch (§5).
