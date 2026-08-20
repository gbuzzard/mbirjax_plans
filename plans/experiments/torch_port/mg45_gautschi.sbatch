#!/bin/bash
# mg45: the recompile-budget remedy's gate.
#
# WHY THIS JOB EXISTS.  The component split (job 15391547, the mg44 log)
# attributed the multiaxis and translation two-device losses to the back
# projection running uncompiled: torch caps the compiled variants of one
# function at 8, the per-device compiled instances share that budget, and
# the variants guard on the device index, so two devices fill the budget
# and the remaining calls run eagerly.  The remedy raises the budget to 64
# before the library compiles anything (projectors.py,
# _RECOMPILE_LIMIT_FLOOR).  This job re-runs the component split's cheap
# arms on the remedied tree: multiaxis at the 512-class and translation at
# the production cell, each at one and two devices, wrapped.
#
# WHAT SUCCESS LOOKS LIKE.  The report compares every wall with the
# recorded pre-remedy walls, so RECOVERY PRINTS AS WARNING LINES (a warm
# wall far below the recorded one).  The specific expectations:
#   - no torch recompile-limit warning anywhere in this log;
#   - multiaxis 512 two-device warm well below the recorded 32.6 s, with
#     the in-iteration back projection near its compiled one-device cost
#     rather than the eager 16.5 s;
#   - translation two-device warm at or below the recorded 14.2 s;
#   - the one-device walls unchanged at 11.4 s and 12.6 s, because one
#     device never filled the budget.
# The exit code stays instrument health only.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that
# installs into the same environment.  Chain with
# --dependency=afterany:<job id> behind any such job already queued.
#
# WALL ESTIMATE.  Two staging jobs that only re-hash files on disk and
# four timed arms, the largest about three minutes.  Roughly fifteen
# minutes in all; forty-five are requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=2
#SBATCH -t 00:45:00
#SBATCH -J torch_mg45
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg45_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L

# ── the mbirtorch preflight ──────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this
# one and leaves a torn .pth.  Chain instead, with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 2, torch.cuda.device_count()" || exit 1

# The tree must carry the remedy this job gates.  A tree without it would
# re-measure the defect and report the remedy failed.
"$TORCHPY" -c "
from mbirtorch import projectors
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, \
    'the tree is missing the recompile-budget remedy this job gates'
" || exit 1
# The width padding witness, as on every job since it landed.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
assert padded_kernel_width(504) == 512, \
    'the tree is missing the width padding these readings describe'
" || exit 1

export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_WIDENING_GUARD
unset MBIRTORCH_RECOMPILE_LIMIT
unset PYTHONPATH
unset MG44_SMOKE
unset MG44_DRY

export MG44_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg44_component
export MG44_SINO_DIR=/scratch/gautschi/buzzard/torch_p3/results/mg26_floors
# The cheap arms only: the 512-class pair carried the deepest loss and the
# translation pair carries the shared-mechanism reading.  The 1024-class
# confirmation is a separate submission, priced at about 50 minutes.
export MG44_ARMS=ma512_n1,ma512_n2,tr_n1,tr_n2
mkdir -p "$MG44_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG44_DRY=1
"$TORCHPY" -u mg44_component_split.py || exit 1
unset MG44_DRY

"$TORCHPY" -u mg44_component_split.py
rc=$?
echo "MG45 DONE rc=$rc"
exit $rc
