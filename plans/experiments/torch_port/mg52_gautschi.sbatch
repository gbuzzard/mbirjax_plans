#!/bin/bash
# mg52: how long does the same reconstruction take in mbirtorch and in
# mbirjax, for the two geometries that have no hand-written kernels?
#
# WHY THIS JOB EXISTS.  Multiaxis and translation are the only two projection
# geometries in mbirtorch whose forward and back projections both run as
# general torch code: neither binds a hand-written kernel in either direction,
# and the witness block below asserts exactly that before any measurement
# runs.  Every cross-framework timing this work has recorded is for parallel
# beam or cone beam, both of which DO bind kernels in both directions.  So the
# question of whether these two geometries need kernels of their own has never
# been asked against the other library -- the case for writing them rests on
# gains measured elsewhere and on mbirtorch-only walls with nothing to divide
# by.  This job measures the missing number: the same staged sinograms
# reconstructed by both libraries, at matched sizes, on ONE GPU, with the
# walls, the peak device memory and the value fingerprints printed side by
# side.
#
# WHAT IT MEASURES, four cells, cheapest first so a defect shows up in minutes:
#
#   translation  (256, 1900, 3000)   the production translation scan
#   multiaxis    (512, 448, 384)
#   multiaxis    (768, 672, 576)
#   multiaxis    (1024, 1008, 992)
#
# Each cell is staged once with mbirtorch -- model, phantom, forward
# projection, weights -- and written to one npz with an md5 beside it.  Both
# arms of that cell then load the same bytes, verify the md5, rebuild their own
# framework's model from the staged constructor parameters, check that their
# recon shape equals the staged one, and run the same seeded three-iteration
# reconstruction with the stopping rule disabled: one cold pass, then the warm
# median of three.  The full statement of the protocol is in
# mg52_framework_anchor.py; it is the widening-floors refresh's model
# construction, unchanged, because that is the protocol the recorded mbirtorch
# walls this job sits beside were measured under.
#
# NEITHER FRAMEWORK IS UNDER TEST AGAINST A THRESHOLD.  There are no gates on
# any measured number.  The exit code reports whether the instrument worked --
# every arm produced a result, every staged file verified, every model matched
# the staged recon shape -- and a person reads the table in this log.
# Fingerprints are recorded and never gated: three-iteration differences
# between the two libraries for these geometries were measured near 1e-3
# relative during the port, and a gap above that prints a note.
#
# AN ARM THAT DOES NOT FIT, OR DOES NOT FINISH, IS A RESULT.  Each arm runs in
# its own process under a 45-minute cap.  An arm that runs out of device memory
# or hits the cap is recorded as that arm's result and the run continues.  At
# the 1024-class multiaxis cell either framework may genuinely not fit one
# device, and that outcome is itself the datum this job went looking for.  Note
# for whoever reads a jax memory failure: jax preallocates a fraction of the
# device by default, so a jax arm that runs out of memory ran out of that
# fraction rather than of the card, and every jax row records the allocator's
# byte limit alongside its peak.
#
# THE JAX SIDE IS BUILT, NOT BORROWED.  The persistent mbirjax_regression conda
# environment holds a CUDA-capable jax, but its own mbirtorch/mbirjax install
# points at worktrees that are deleted between nightly runs, so importing
# mbirjax from it is not dependable.  This job therefore builds a BARE venv
# under the results directory with that environment's python, installs the
# pinned mbirjax checkout into it editable, and appends the conda
# environment's site-packages as a plain path line in a .pth file -- a path
# line adds the directory to sys.path without processing that directory's own
# .pth files, so jax, numpy and the CUDA wheels resolve from the shared
# environment while no other editable finder registers and mbirjax can only
# come from the pinned checkout.  The venv is VALIDATED rather than trusted: a
# build that cannot import jax with a GPU visible, or whose mbirjax does not
# live under mbirjax_src, is torn down and rebuilt, and the job stops if the
# rebuild fails too.
#
# IF THE VENV VALIDATES ITS IMPORT BUT SEES NO GPU: the environment's jax
# resolves its CUDA runtime from pip wheels in its own site-packages
# (jax_cuda13_pjrt and jax_cuda13_plugin were verified there 2026-08-22), so
# GPU visibility should not depend on any activation.  If it fails anyway,
# check that those wheel directories still exist in the site-packages the log
# prints above; a nightly dependency change is the likely mover.  Note the
# batch shell has no `conda` command at all, so any manual poking at this
# environment goes through its absolute python path.
#
# THE MBIRJAX CLONE MUST BE STAGED BEFORE THIS JOB RUNS, at the commit the jax
# arms were written against.  It is not created here:
#     git clone <mbirjax> /scratch/gautschi/buzzard/torch_p3/mbirjax_src
#     git -C /scratch/gautschi/buzzard/torch_p3/mbirjax_src checkout 7bb2009
# The preflight below asserts that hash before any arm runs.  A different
# commit is not a smaller version of this measurement: the model constructors
# and the recon signature the jax arms call were read at that commit.
#
# WALL ESTIMATE, and its arithmetic.  The mbirtorch side is known.  The
# recorded one-device warm walls are about 12.6 s for the translation cell and
# 11.4 s, 56.3 s and 310 s for the multiaxis 512-, 768- and 1024-class cells,
# and each arm runs a cold pass plus three warm ones, so the four torch arms
# come to roughly 40 minutes of compute -- the 1024-class arm alone is about 25
# minutes, which is not far under its own 45-minute cap.  The mbirjax side is
# unknown, which is the entire reason for the job; each of its four arms is
# capped at 45 minutes.  Staging writes about 21 GB across the four cells (10.9
# for translation, 0.7, 2.2 and 7.6 for the three multiaxis cells) and costs
# maybe fifteen minutes of phantom building, forward projection and disk.  So
# the expectation is around two and a half hours, and 04:00:00 is requested to
# cover staging plus the torch side plus two jax arms running all the way into
# their caps.  A job cut short still yields everything it finished -- the jsonl
# is flushed row by row -- and MG52_ARMS=<comma-separated arm ids> re-runs only
# the rest, reusing every staged file whose md5 still matches.
#
# DISK.  About 21 GB of staged npz files in $MG52_RESULTS, KEPT: a re-run
# stages nothing, and a later run of this comparison reconstructs the same
# bytes rather than a regenerated sinogram that is not bit-identical.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment, so
# it must not run at the same time as another torch_p3 job that installs into
# that environment; submit it with --dependency=afterany:<job id> behind any
# such job already queued.  It also READS the mbirjax_regression conda
# environment through the path line above, so a nightly that rebuilds that
# environment while this job runs could pull jax out from under a running arm.
# At four hours this job will very likely overlap a nightly window, so chain it
# behind the nightly if the two can be ordered.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg52_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J torch_mg52
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg52_%j.log
# One GPU and 14 cores: the one-GPU shape every single-device job in this
# campaign has used, which is a quarter of the node.
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
MBIRJAX_SRC=/scratch/gautschi/buzzard/torch_p3/mbirjax_src
MBIRJAX_COMMIT=7bb20093d635802ba8505c9366ff109ff6b35b76
export MG52_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg52_framework_anchor
JAX_VENV="$MG52_RESULTS/venv_jax"
JAXPY="$JAX_VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg
mkdir -p "$MG52_RESULTS"

nvidia-smi -L

# ── the mbirtorch preflight ──────────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this one
# and leaves a torn .pth.  Chain instead, with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# The tree witnesses.  The first three say which mbirtorch tree produced these
# walls -- the padded, recompile-remedied tree the recorded one-device walls
# were measured on.  The fourth is this job's PREMISE: neither geometry under
# test binds a hand-written kernel, so both projection directions run as
# general torch code.  If that ever stops being true, these walls describe a
# different library than the question does, and the job should stop rather than
# quietly measure the new one.
"$TORCHPY" -c "
import inspect
import numpy as np
import mbirtorch
from mbirtorch import _memory_ledger, projectors
from mbirtorch._utils import padded_kernel_width
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
az = np.linspace(0, np.pi, 12, endpoint=False)
el = np.linspace(-0.5, 0.5, 12)
models = (('multiaxis', mbirtorch.MultiAxisParallelModel((12, 16, 16),
                                                         np.stack([az, el], axis=1))),
          ('translation', mbirtorch.TranslationModel(
              (8, 24, 32),
              mbirtorch.gen_translation_vectors(4, 2, x_spacing=6.0, z_spacing=4.0),
              source_detector_dist=12.0, source_iso_dist=12.0)))
for name, model in models:
    directions = tuple(_memory_ledger.torch_body_directions(model))
    assert directions == ('forward', 'back'), (name, directions)
print('tree witnesses ok; multiaxis and translation run general torch code in both directions')
" || exit 1

# ── the pinned mbirjax checkout ──────────────────────────────────────────────
# Staged before the job, not created here.  Checked BEFORE the venv is built,
# so a wrong or missing checkout costs a second rather than an hour.
test -d "$MBIRJAX_SRC" || { echo "the mbirjax checkout is missing at $MBIRJAX_SRC: clone it and check out $MBIRJAX_COMMIT before submitting"; exit 1; }
FOUND=$(git -C "$MBIRJAX_SRC" rev-parse HEAD) || exit 1
if [ "$FOUND" != "$MBIRJAX_COMMIT" ]; then
    echo "$MBIRJAX_SRC is at $FOUND, and the jax arms were written against $MBIRJAX_COMMIT: check that commit out before submitting"
    exit 1
fi
echo "mbirjax checkout at the pinned commit $FOUND"

# ── the jax overlay venv ─────────────────────────────────────────────────────
# Built BARE from the mbirjax_regression environment's python, with the pinned
# checkout installed editable and that environment's site-packages appended as
# a path line.  --system-site-packages is not used: a venv built over another
# environment inherits that environment's own .pth processing, which is how an
# unrelated editable install would come back in.  Idempotent: a venv that
# already validates is reused.
#
# The environment's python is named by its ABSOLUTE path.  The batch shell has
# no `module` command and therefore no `conda` function (the module-not-found
# lines from load_conda_cuda.sh above are this, and every earlier job in this
# series ran absolute paths so nothing noticed), so `conda run` produces
# nothing here.  The absolute path needs neither.
JAXBASEPY=/home/buzzard/.conda/envs/mbirjax_regression/bin/python
test -x "$JAXBASEPY" || { echo "no python at $JAXBASEPY; is the mbirjax_regression environment present?"; exit 1; }
CONDA_SITE=$("$JAXBASEPY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])" | tail -1)
test -n "$CONDA_SITE" || { echo "could not read the mbirjax_regression site-packages path from $JAXBASEPY"; exit 1; }
echo "shared environment site-packages: $CONDA_SITE"
if ! "$JAXPY" -c "import jax, mbirjax; assert 'mbirjax_src' in mbirjax.__file__" 2>/dev/null; then
    rm -rf "$JAX_VENV"
    "$JAXBASEPY" -m venv "$JAX_VENV" || exit 1
    "$JAXPY" -m pip install --no-cache-dir -q --no-deps -e "$MBIRJAX_SRC" || exit 1
    JAX_SITE=$("$JAXPY" -c "import site; print(site.getsitepackages()[0])") || exit 1
    echo "$CONDA_SITE" > "$JAX_SITE/zz_jax_env.pth" || exit 1
fi
# Validate rather than trust.  Three things, each of which has silently gone
# wrong before in a layered environment: jax imports and sees a GPU, mbirjax
# comes from the pinned checkout and not from the conda environment's own
# install, and both versions are on the record in this log.
"$JAXPY" -c "
import importlib.metadata as md
import mbirjax
import jax
devices = jax.devices()
assert 'mbirjax_src' in mbirjax.__file__, mbirjax.__file__
platforms = sorted({d.platform for d in devices})
assert any(p == 'gpu' for p in platforms), ('no GPU device visible to jax', platforms, devices)
try:
    version = md.version('mbirjax')
except Exception:
    version = 'unknown'
print('overlay venv ok')
print('  jax', jax.__version__, 'devices', devices)
print('  mbirjax', version, 'from', mbirjax.__file__)
" || { echo "the jax overlay venv at $JAX_VENV did not validate: see the note in this file's header about the conda environment's libraries and GPU visibility"; exit 1; }

# ── the shipped configuration, set explicitly ────────────────────────────────
# The kernels stay on.  Neither geometry under test has any, so this changes
# nothing they do; it keeps the environment the one every other run in this
# campaign used.
export MBIRTORCH_DISABLE_TRITON=0
# Every torch arm is pinned to one device by MBIRTORCH_NUM_DEVICES, which the
# script sets per job.  An inherited value would describe a different count
# than the plan.
unset MBIRTORCH_NUM_DEVICES
# This is a TIMING probe, and it reads the peak memory counters itself.  The
# calibration mode owns those counters and does extra work at the settle and at
# the end of every call, so an arm that inherited it would neither time nor
# measure what the other arms did.
unset MBIRTORCH_MEMORY_CALIBRATION
# A one-device pin bypasses the speed floors anyway; unset so no inherited
# value can change what an arm does.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted.
unset MBIRTORCH_RECOMPILE_LIMIT
unset MBIRTORCH_SORTED_FORWARD
# PYTHONPATH must not point at either checkout.  Each framework's arms select
# their library through their INTERPRETER, and an inherited PYTHONPATH would
# shadow that silently, in the one direction that would make this job compare a
# library against itself.  The script pops it for every job and sets only the
# framework-specific variables below, both of which are empty on the cluster.
unset PYTHONPATH
unset MG52_TORCH_PYTHONPATH
unset MG52_JAX_PYTHONPATH
unset MG52_SMOKE
unset MG52_DRY
unset MG52_ARMS
unset MG52_ARM_TIMEOUT_MIN
unset MG52_REPEATS
unset MG52_ITERATIONS
unset MG52_STAGE_DIR

export MG52_TORCH_PYTHON="$TORCHPY"
export MG52_JAX_PYTHON="$JAXPY"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
test -f mg52_framework_anchor.py || { echo "mg52_framework_anchor.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with the fourteen jobs in run order and a
# mis-set variable is caught before any GPU work.  The dry plan imports no
# framework.
export MG52_DRY=1
"$TORCHPY" -u mg52_framework_anchor.py || exit 1
unset MG52_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what it
# measured: the side-by-side table, the ratios, the peaks and the fingerprint
# gaps are all in this log, and an arm that ran out of memory or hit its cap is
# a recorded result rather than a failure.
"$TORCHPY" -u mg52_framework_anchor.py
rc=$?
echo "MG52 DONE rc=$rc"
exit $rc
