# Automated release workflow — proposal

**Setup progress (2026-08-10, Charlie's session).**  Of the eight manual
setup steps in this plan: steps 1 and 2 (main rename, prerelease branch)
were done earlier during the docs work, and step 5 is now DONE — Trusted
Publishing "pending publishers" for project `mbirtorch` are registered on
BOTH registries (repository `cabouman/mbirtorch`, workflow `release.yml`,
environment `testpypi` on TestPyPI and `pypi` on PyPI).  This also
resolved a years-old blocker: Charlie's TestPyPI web login was broken
(stale password + the TestPyPI authenticator secret living in Duo under
the same "PyPI" label as the real-PyPI secret in MS Authenticator);
fixed, with recovery codes generated.  Remaining setup: branch
protection (3), version single-sourcing (4), the GitHub `pypi`/`testpypi`
environments with the approval rule (6), Read the Docs default-version
setting (7, partially done), and the RTD token (8).  The five open
decisions below are still open.

**Status:** PROPOSAL, workflow files not yet implemented.  This page proposes a release
automation for mbirtorch and lists the manual setup it needs.  Neither mbirtorch
nor mbirjax has a `.github/` directory today, so there is no existing workflow to
extend and no house style to match.  Every decision below is therefore open.

**Scope.**  The proposal covers the branch model, the continuous-integration
checks, publication to PyPI, and the Read the Docs build.  It does not cover GPU
testing, which GitHub-hosted runners cannot do.

## The shape of the proposal

The release is driven by a GitHub Release, not by a branch merge.  Merging to
`main` prepares a release; publishing a GitHub Release performs it.  That split
matters because it puts a human gate on the irreversible step.  A PyPI version
number can never be reused, so the action that consumes one should be deliberate.

Three workflow files do the whole job.  `ci.yml` runs tests and builds the docs on
every pull request.  `release.yml` builds the distribution, publishes it, and
triggers a documentation build.  `docs-preview.yml` is optional and builds the
docs on pull requests that touch `docs/`.

The route to PyPI is Trusted Publishing rather than an API token.  PyPI mints a
short-lived credential from GitHub's OIDC identity at publish time.  No long-lived
secret is stored in the repository, and a leaked repository secret cannot be
replayed.  This is the current PyPI recommendation, and it removes the token
rotation problem entirely.

## Branch model

The three-tier model matches the existing practice.  Feature branches merge to
`prerelease` by pull request.  `prerelease` merges to `main` by pull request when
a release is ready.  `main` is the released state, and every commit on it should
correspond to a published version or a documentation fix.

The primary branch renames from `master` to `main` first.  mbirtorch has only
`master` today, both locally and on the remote, and no `prerelease` branch exists
yet.  mbirjax already uses `main` and `prerelease`, so the rename also aligns the
two repositories.

### Renaming master to main

GitHub renames a branch in place and redirects most references.  The steps are::

    git branch -m master main
    git push -u origin main
    # then, in the GitHub UI: Settings -> Branches -> rename / set default to main
    git push origin --delete master     # only after the default branch has moved

Three things do not follow the rename automatically.  Open pull requests
retarget themselves, but local clones do not, so every collaborator runs
`git branch -m master main` and `git fetch --prune` once.  Read the Docs stores
the branch name in its project configuration, so its default version needs
updating.  Any local script or CI reference to `origin/master` needs a grep.

The `prerelease` branch is then created from `main`::

    git checkout -b prerelease main
    git push -u origin prerelease

Branch protection is worth setting on both.  The suggested rules are: require a
pull request, require the `ci.yml` checks to pass, and disallow force pushes.

## Versioning

The version is currently declared twice.  `pyproject.toml` sets
`version = "0.0.1"` and `mbirtorch/__init__.py` sets `__version__ = "0.0.1"`.
Two declarations drift, and a release workflow makes the drift expensive, because
the wheel and the runtime would disagree about what was published.

The proposal single-sources the version from the package.  In `pyproject.toml`::

    [project]
    dynamic = ["version"]

    [tool.setuptools.dynamic]
    version = {attr = "mbirtorch.__version__"}

Setuptools reads that attribute by parsing the source, not by importing it, so the
environment side effects at the top of `__init__.py` do not run at build time.
The assignment is a plain string literal, which is the form the static parser
handles.  `mbirtorch.__version__` keeps working at runtime, unchanged.

The release workflow then checks the tag against the version.  A release tagged
`v0.1.0` must build a distribution whose version is `0.1.0`, and the job fails
otherwise.  This catches the most common release mistake, which is tagging without
bumping.

## Workflow 1: continuous integration

`ci.yml` runs on pull requests to `prerelease` and `main`, and on pushes to those
branches.  It installs the CPU-only torch wheel, since the CUDA wheel is several
gigabytes and no runner has a GPU.

```yaml
name: CI
on:
  pull_request:
    branches: [prerelease, main]
  push:
    branches: [prerelease, main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install CPU torch first
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install package and test extra
        run: pip install -e ".[test]"
      - name: Run tests
        run: python -m pytest -n 4 tests

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install CPU torch first
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install package and docs extra
        run: pip install -e ".[docs]"
      - name: Build docs
        run: python -m sphinx -b html docs/source docs/build/html
```

The GPU tests skip on these runners rather than fail.  The local run reports 279
passed and 52 skipped, and the skip count will be higher in CI, because the Triton
kernel tests and the CUDA sharding tests both need hardware.  That is a real
coverage gap, and the proposal does not close it.  Closing it needs a self-hosted
runner with a GPU, which is a separate decision.

The docs job currently cannot use `-W` to fail on warnings.  The build reports 15
warnings, all of them references to pages the port has not written yet.  Once the
port completes and that count reaches zero, `-W` should be added, so a broken
cross-reference fails the pull request instead of accumulating.

Adding a `pytest.ini` is worth doing at the same time.  mbirjax has one and
mbirtorch does not, so CI and local runs can disagree about which paths are
collected.  mbirjax's file sets `testpaths` and `norecursedirs`.

## Workflow 2: release

`release.yml` runs when a GitHub Release is published.  The release being marked
as a pre-release routes the distribution to TestPyPI; otherwise it goes to PyPI.
One workflow therefore covers both destinations, and the choice is visible in the
GitHub Release UI rather than buried in branch logic.

```yaml
name: Release
on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install --upgrade build twine
      - run: python -m build
      - run: python -m twine check dist/*
      - name: Verify tag matches package version
        run: |
          TAG="${GITHUB_REF_NAME#v}"
          WHL=$(ls dist/*.whl)
          echo "$WHL" | grep -q "mbirtorch-${TAG}-" || {
            echo "Tag $TAG does not match built version in $WHL"; exit 1; }
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish-testpypi:
    needs: build
    if: github.event.release.prerelease
    runs-on: ubuntu-latest
    environment: testpypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/

  publish-pypi:
    needs: build
    if: ${{ !github.event.release.prerelease }}
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

The `permissions: id-token: write` line is what enables Trusted Publishing.  The
`environment:` line is what lets a GitHub environment protection rule require a
manual approval before the publish step runs.  Setting that rule on the `pypi`
environment adds a second human gate, which is worth having on the irreversible
action.

Building once and publishing the same artifact is deliberate.  Building separately
per destination could publish two distributions that differ, and the TestPyPI run
would then not be a rehearsal of the real one.

## Read the Docs

Read the Docs builds automatically once its GitHub integration is connected.  The
integration installs a webhook, and a push to a tracked branch or a new tag starts
a build.  Connecting it is a manual step in the Read the Docs project settings, and
it is the primary mechanism this proposal relies on.

Two Read the Docs settings need to match the branch model.  The default version
should be `stable`, which Read the Docs maps to the highest semantic-version tag.
The `main` branch should be tracked as `latest`.  The result is that the published
default documentation corresponds to the released version on PyPI, and `latest`
shows what is on `main`.

An explicit trigger is added as well, so a release does not silently ship without
documentation.  The step below runs at the end of the release workflow.

```yaml
  trigger-docs:
    needs: [build]
    if: ${{ !github.event.release.prerelease }}
    runs-on: ubuntu-latest
    steps:
      - name: Trigger Read the Docs build of stable
        run: |
          curl -fsSL -X POST \
            -H "Authorization: Token ${{ secrets.RTD_TOKEN }}" \
            https://readthedocs.org/api/v3/projects/mbirtorch/versions/stable/builds/
```

This duplicates what the webhook already does, and that redundancy is the point.
The webhook is the normal path, and the explicit call makes the release fail
loudly if the documentation build was not started.  It needs one repository
secret, `RTD_TOKEN`, which is the only long-lived secret in the proposal.

`.readthedocs.yaml` already exists in the repository and needs no change.  It
installs the CPU torch wheel before `pip install .[docs]`, for the same
build-size reason as CI.

## Manual setup, in order

Eight steps are needed before the first automated release, and six of them are
one-time.

1. Rename `master` to `main` and set it as the default branch.
2. Create `prerelease` from `main`.
3. Set branch protection on `main` and `prerelease`.
4. Single-source the version in `pyproject.toml`.
5. Register the project on PyPI and on TestPyPI, and configure Trusted
   Publishing on each for the `cabouman/mbirtorch` repository and the
   `release.yml` workflow.
6. Create the `pypi` and `testpypi` GitHub environments, with an approval rule on
   `pypi`.
7. Connect the Read the Docs project to the repository, and set the default
   version to `stable`.
8. Add the `RTD_TOKEN` repository secret.

Step 5 has an ordering constraint worth noting.  Trusted Publishing for a project
that does not exist on PyPI yet uses a "pending publisher", which is configured
before the first upload rather than after.

## Releasing, once set up

The routine has five steps.  Bump `__version__` on a branch and merge it to
`prerelease`.  Publish a GitHub Release marked as a pre-release, tagged
`v0.1.0rc1`, targeting `prerelease`, which uploads to TestPyPI.  Install from
TestPyPI and run the tests against the installed wheel.  Open a pull request from
`prerelease` to `main` and merge it.  Publish a GitHub Release tagged `v0.1.0`
targeting `main`, which uploads to PyPI and triggers the documentation build.

`dev_maintenance.rst` should then be rewritten around this routine.  That page is
currently on hold, and it presently describes mbirjax's manual `twine upload`
procedure.  Once this proposal is implemented, the manual procedure becomes the
fallback rather than the method, and the page should say so.

## Open questions

Five decisions are needed before implementation.

**Does the GPU coverage gap need closing now?**  CI on GitHub-hosted runners skips
every kernel and CUDA sharding test.  A release could pass CI with a broken Triton
path.  A self-hosted GPU runner would close it, at the cost of maintaining one.

Greg:  No, we'll rely on the nightly runs on Gautschi.  

**Should `prerelease` publish to TestPyPI on every merge?**  The proposal says no,
and requires an explicit pre-release tag.  Publishing on every merge would consume
version numbers quickly, since TestPyPI also refuses reuse.

Greg:  No, this should be a trigger that mimics the trigger used for PyPI 
release. The true PyPI release trigger should have an interactive "Are you sure?" verification.

**Which Python versions?**  The proposal tests 3.11 and 3.12.  `requires-python`
is `>=3.11`.  Adding 3.13 depends on torch wheel availability at the pinned floor
of 2.13.

Greg:  We'll need some way of adding increments every year or 
so.  We'll need advice about how to manage that in a reliable way.

**Should the wheel be tested after publication?**  mbirjax's maintenance page
includes a step that installs from PyPI and runs the tests.  Automating it means a
second workflow triggered after publication, which is straightforward but adds a
moving part.

Greg:  I think it should be tested locally.  Charlie typically
does that by hand (and enjoys the immediate feedback), but perhaps
there could be a single dev_scripts script to do that with a single run.

**Is `cabouman/mbirtorch` the release home?**  `pyproject.toml` currently lists
`source = "https://github.com/cabouman/mbirjax"` under `[project.urls]`, which
points at the wrong repository.  That needs fixing regardless, and Trusted
Publishing needs the owning repository named exactly.

Greg: Yes, `cabouman/mbirtorch`.
