# Python-version watch in the nightly — plan

**Status:** REVIEWED DRAFT, for Greg.  Greg requested this plan on
2026-08-10, after the follow-torch Python-version policy was accepted
(`release_workflow.md`, the "Which Python versions?" question).  A
four-reviewer Opus panel reviewed the first draft; every finding is
incorporated, and the section "Panel review record" at the bottom lists
the findings that changed the design.

## 1. Terms

- **The watch**: the new nightly step this plan adds.
- **The matrix**: the `python-version` list in mbirtorch's
  `.github/workflows/ci.yml`.  These are the versions CI tests.
- **The torch list**: the CPython versions for which torch publishes
  Linux x86_64 wheels on its CPU index, in its newest stable release.
  The CPU index is `https://download.pytorch.org/whl/cpu/`, and it is
  the source the watch reads because it is the index CI installs torch
  from.  (The panel found that PyPI, the first draft's source, is a
  different distribution channel whose Python coverage differs from the
  CPU index's today.)
- **The floor**: the `requires-python` lower bound in mbirtorch's
  `pyproject.toml` (today `>=3.11`).
- **A divergence**: the matrix differs from the torch list, restricted
  to versions at or above the floor.  An addition is due when the torch
  list has a version the matrix lacks.  A removal is due when the
  matrix has a version the torch list lacks.  Versions below the floor
  are reported as an informational line and never acted on.

## 2. What the watch does, and why

The accepted policy is that mbirtorch tests the Python versions torch
supports.  The maintenance problem is that someone must notice when
torch's list changes.  Charlie has said a line in a report will never
be noticed.  The alert must arrive by email and state exactly what to
do.

The watch runs every night.  It compares the matrix against the torch
list.  On a divergence, it opens ONE pull request in
`cabouman/mbirtorch` containing the complete edit for that divergence.
GitHub emails both maintainers when the pull request opens.  CI runs on
the pull request and reports whether the proposed versions pass the
test suite.  A maintainer reads the diff and clicks Merge.

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

The matrix and the floor are read from `ci.yml` and `pyproject.toml`
on the `prerelease` branch of `cabouman/mbirtorch`.  The watch records
the commit identifier it read, and every later step works from that
same commit.

To keep a mid-rollout wheel upload from triggering a false alarm, the
watch acts only on a divergence it has observed on two consecutive
nights.

### 3.2 The pull request

On a confirmed divergence, the watch performs four actions.

1. It creates a branch from the recorded commit, named for the whole
   divergence: for example `nightly/python-matrix-add-3.13-3.14`.  One
   divergence produces one branch and one pull request, however many
   versions it involves.
2. It edits every file that names a Python version.  An addition edits
   the matrix.  A removal edits the matrix, the floor in
   `pyproject.toml`, and the `docs` job's separate hardcoded
   `python-version` in `ci.yml` when that job pins the removed version.
   The commit's author is the watch's own name, for example
   `mbirjax-nightly-watch <buzzard@purdue.edu>`, so automated commits
   are recognizable at a glance.
3. It opens a pull request with base `prerelease`.  The title names the
   change ("Add Python 3.13 and 3.14 to the CI test matrix").  The body
   names the torch release that triggered it, the commit the branch was
   cut from, and the policy.  A removal's body additionally says that
   CI cannot prove a removal correct, so the reviewer must judge it.
4. It requests review from `cabouman` and `gbuzzard`, which makes
   GitHub email them both.

The pull request targets `prerelease` for two reasons.  Every change is
merged into `prerelease` first.  CI runs on pull requests to
`prerelease`, so the pass or fail result appears on the pull request
itself.

One consequence deserves a sentence: this is a same-repository pull
request, so CI runs the branch's own copy of `ci.yml` with repository
secrets in scope.  The reviewer must read the diff, not only the check
result.  Today the repository holds no secrets; when `RTD_TOKEN` is
added for releases, it should be scoped to a GitHub environment
restricted to `main`, not stored as a plain repository secret.

### 3.3 One pull request per divergence, and a visible veto

The branch name is determined entirely by the divergence, so the same
divergence always produces the same branch name.  The watch's
duplicate rule keys on the pull request, not the branch alone.  If a
pull request for the branch exists and is open or merged, the watch
does nothing.  If the branch exists but has no pull request — the
leftover of a partly failed night — the watch repairs the state by
opening the pull request.  If the pull request was closed without
merging, that is the maintainers' standing "no", and the watch does not
reopen it; deleting the branch withdraws the "no".

A suppressed divergence must stay visible.  On every night that the
watch stands down because of a closed pull request, it writes one
report line naming the branch and the closing date.  The repository
being knowingly out of policy is then a nightly statement, not a
one-day event.

### 3.4 Failure isolation, and proof of life

The watch must never break the nightly.  It runs on the wrapper's
unconditional path (see §3.6), catches every failure inside itself,
writes the failure to the nightly log and the report, and leaves the
nightly's exit status unchanged.  Losing one night's check has no
lasting cost, because a real divergence is still there the next night.

A silent death must be visible, because a watch that stopped running
is indistinguishable from a watch that found nothing.  Three
mechanisms provide proof of life.  First, the watch writes a
last-successful-check timestamp to a state file, and the nightly
report and the status script both print it.  Second, the report line
states three facts separately — the files it read, the matrix it
found, and the verdict — so "could not read the matrix" never reads as
"no divergence".  Third, the watch makes one authenticated request to
GitHub every night regardless of divergence, and reports the
credential's validity and days to expiry; an expiring token becomes a
nightly countdown rather than a surprise on the night the alert was
needed.

### 3.5 Credentials

The panel established three GitHub facts that the first draft got
wrong, and they force the design.  A fine-grained token for
`cabouman/mbirtorch` can only be issued by `cabouman`, the repository
owner.  A pull request's author cannot be one of its requested
reviewers.  GitHub does not email an account about its own actions.
Together: if either maintainer's token opens the pull request, that
maintainer — the person the email exists for — never gets the email.

The proposal is therefore a dedicated machine account.  Its properties:

- a separate GitHub account (for example `mbirjax-bot`), added as a
  collaborator on `cabouman/mbirtorch` with write permission;
- a classic personal access token of that account with the `repo` and
  `workflow` scopes — `workflow` is required because the watch edits a
  file under `.github/workflows/`, which GitHub refuses without it;
- the token stored in a file under `~/.mbirtorch/` on the machine that
  runs the watch, with file mode 0600, a location outside every git
  working tree and outside the folders macOS restricts for background
  jobs; the watch refuses a credential path that resolves inside a git
  working tree;
- the token is passed to the GitHub client through the environment or
  a file, never as a command-line argument, and the error path filters
  any token-shaped string (`ghp_*`, `github_pat_*`) before writing to
  the log or the report.

What this token can do if stolen, stated plainly: push to any
unprotected branch (including `prerelease`), and create tags and
releases.  Two standing controls bound the damage: `main` accepts
changes only by pull request with passing CI, and publishing to PyPI
requires Charlie's manual approval of the `pypi` environment.  Those
two controls must stay in place for as long as the watch runs.
Revocation is one step — delete the token in the machine account's
settings — and its only consequence is that the watch stops.

### 3.6 Where the watch runs

One wrapper serves all the nightly jobs (`run_torch_regression.sh`);
the Mac and Gautschi differ only in configuration.  The watch is
enabled by one knob in `torch_regression.env` (`PYTHON_WATCH_ENABLED`),
set on the Mac and unset on Gautschi.  Three reasons favor the Mac:
its network access is unrestricted; the credential file is stored on a
machine the maintainers control physically; and the Mac job runs
nightly at near-zero cost.

Placement inside the wrapper matters.  The Mac torch nightly skips
almost all of its work on nights when mbirtorch has not changed, and
most nights are such nights.  The watch call therefore goes on the
wrapper's unconditional path, before the unchanged-night skip, so it
runs on every wake.

One trust relationship must be stated: the nightly wrapper updates its
own code from the `mbirjax_metrics` repository and then runs the
updated copy.  Whoever can push to `mbirjax_metrics` can therefore
change what the watch does with its credential.  Today that is the two
maintainers, which is acceptable; it stops being acceptable if
`mbirjax_metrics` ever gains other writers.

## 4. File inventory

### mbirjax_metrics

| file | change |
|---|---|
| `tooling/dependency_watch/python_matrix_watch.py` | NEW — detection, the pull-request actions, duplicate suppression, proof of life, dry-run mode |
| `tooling/dependency_watch/watch_config.env` | NEW — repository name, base branch, credential path, the enable knob's documentation |
| `tooling/dependency_watch/test_python_matrix_watch.py` | NEW — unit tests on saved index and ci.yml fixtures, including a `cp314t` tag, a non-wheel file, a multi-version divergence, and a below-floor version |
| `tooling/regression/run_torch_regression.sh` | one call on the unconditional path, enabled by the knob |
| `tooling/regression/torch_regression.env` | the `PYTHON_WATCH_ENABLED` knob |
| `tooling/regression/status_torch_nightly.sh` | print the last-successful-check timestamp |
| `tooling/regression/README.md` | a dependency-watch section |

### mbirtorch

No changes to any tracked branch.  The watch pushes `nightly/*`
branches, which carry proposals until a maintainer merges or closes
them.

### mbirjax_plans

This file.  `release_workflow.md` gains a status line when the watch is
live.

## 5. Implementation increments

Each increment is small and independently testable.

1. **The pure checker.**  Detection only: read the CPU index and
   ci.yml, print the divergence or "none".  Unit tests run on saved
   copies of both inputs.  No credential, and no writes anywhere.
2. **Dry-run pull requests.**  The full path up to but not including
   the GitHub write.  `--dry-run` prints the branch name, each edited
   file, and the pull-request title and body.
3. **The machine account and one live trial.**  The account is
   created, granted access, and its token issued and stored.  The
   watch then runs once by hand against a forced divergence, made by
   temporarily removing 3.12 from the matrix on `prerelease`, so the
   trial exercises the real base branch and the real CI trigger.  The
   trial must confirm five behaviors: the pull request appears; both
   maintainers receive the email; CI runs on the pull request; closing
   it unmerged stops the watch from proposing the same change again;
   and the suppressed divergence appears as a report line the next
   night.  The trial pull request is then closed and its branch
   deleted, and the matrix edit on `prerelease` is reverted.
4. **Enable in the Mac nightly.**  The knob is set, and the log is
   read after the first night the watch runs.

## 6. Open questions for Greg

1. **The machine account.**  Does a dedicated account (`mbirjax-bot`
   or similar) fit how you want the repositories administered?  The
   alternative — a maintainer's own token — was ruled out by the
   review-request and self-notification facts in §3.5, so declining
   the machine account means accepting that one maintainer gets no
   email.
2. **Mac dependence.**  §3.6 puts the watch on the Mac nightly.  If
   the Mac nightly is ever retired, the watch must move with it.  Is
   that dependency acceptable?
3. **Scope.**  This watch covers the Python matrix only.  Your comment
   placed it "along with all the other dependency updates".  The torch
   floor advance is the other deliberate yearly update.  Should it get
   the same pull-request treatment in a follow-on plan once this watch
   has run for a season?

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
