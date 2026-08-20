#!/bin/bash
# mg47: the remedy's large-cell confirmation.
#
# WHY THIS JOB EXISTS.  The recompile-budget remedy (projectors.py,
# raised on the compiling thread) passed its cheap gate: the multiaxis
# 512-class and translation production cells flipped from two-device
# losses to wins (job 15394667, findings section 1.37).  Two readings
# remain before the floors refresh.  The 1024-class pair says whether the
# deepest remaining loss (0.80x recorded) flips too, with the component
# split attributing whatever remains.  The 768-class pair says the raised
# budget cost the one previously WINNING cell nothing: its recorded 1.46x
# ran with the budget surviving, so its walls should reproduce.
#
# WHAT SUCCESS LOOKS LIKE.  No torch recompile-limit warning in this log;
# the 768-class walls at 1.00x of recorded; the 1024-class two-device
# warm well below the recorded 388.8 s, with the in-iteration back
# projection near its one-device 92 s rather than the eager 250 s; the
# one-device walls unchanged (recovery prints as WARNING lines, since the
# report compares against the pre-remedy recorded walls).
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  Chain with
# --dependency=afterany:<job id> behind any such job already queued.
#
# WALL ESTIMATE.  Two staging jobs that only re-hash files on disk and
# four timed arms.  The 1024-class one-device arm is about 21 minutes and
# its two-device arm at most that if the remedy holds.  Roughly an hour;
# ninety minutes are requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=2
#SBATCH -t 01:30:00
#SBATCH -J torch_mg47
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg47_%j.log
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

# The tree must carry the remedy, in its shipped (compiling-thread) form:
# the raise must sit inside the wrapper's first-sight path, not only at
# wrapper creation, or the pool threads compile at torch's default.
"$TORCHPY" -c "
import inspect
from mbirtorch import projectors
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, \
    'the tree is missing the recompile-budget remedy'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path this job confirms'
" || exit 1
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
# The large-cell pairs: the 768-class no-regression check first (cheap),
# then the 1024-class confirmation.
export MG44_ARMS=ma768_n1,ma768_n2,ma1024_n1,ma1024_n2
mkdir -p "$MG44_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG44_DRY=1
"$TORCHPY" -u mg44_component_split.py || exit 1
unset MG44_DRY

"$TORCHPY" -u mg44_component_split.py
rc=$?
echo "MG47 DONE rc=$rc"
exit $rc
