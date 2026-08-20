#!/bin/bash
# mg48: the full widening-floors refresh on the recompile-remedied tree.
#
# WHY THIS JOB EXISTS.  The recompile-budget remedy (projectors.py,
# raised on the compiling thread; findings section 1.37) changed a file
# in the floors' SHARED cost-input set, so every family's rows are stale
# and the refresh tool refuses any narrower scope: carrying a family
# verbatim asserts its costs did not move, and that assertion is no
# longer safe for anyone -- the cone and parallel reconstructions also
# run compiled torch closures (the prior, the update apply) whose
# recompile budgets the remedy raised, and the earlier refresh's own log
# shows those frames hitting the old limit in four-device arms.  This is
# therefore the full refresh, every family at every planned row, on the
# remedied tree.
#
# WHAT CAN MOVE.  The multiaxis and translation two-device cells already
# flipped from losses to wins at the gate (0.35x to 1.53x at the
# multiaxis 512-class; 0.89x to 1.25x at the translation production
# cell), so their sentinel rows should print proposed floors for the
# first time.  The four-device sentinels of both families were measured
# under the same per-thread mechanism and can move anywhere.  Cone and
# parallel should reproduce at one and two devices and may improve at
# four.  The verdicts and the paste-ready rows land in this log; NOTHING
# is pasted into the library by this job, because the sentinel rulings
# are Greg's.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  It is submitted with
# --dependency=afterany behind the mg47 confirmation for that reason.
#
# WALL ESTIMATE.  The pre-remedy full refresh (mg26) ran 2 hours 57
# minutes, dominated by the multiaxis 1024-class arms whose two-device
# walls the remedy has since cut sharply.  Staged inputs regenerate into
# this run's own results directory.  Three and a half hours are
# requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:30:00
#SBATCH -J torch_mg48
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg48_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The tree witnesses.  The floors this run prints price the padded,
# channel-sorted kernels AND the recompile remedy in its shipped
# (compiling-thread) form; all three must be present.  The stale set is
# printed rather than asserted: the shared-input change makes the
# projection families stale by construction, and the full scope
# re-measures everything either way.
"$TORCHPY" -c "
import inspect
from mbirtorch._utils import padded_kernel_width
from mbirtorch.triton_parallel import sorted_forward_enabled
from mbirtorch import projectors
from mbirtorch import _widening_floors as wf
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert sorted_forward_enabled(), 'the sorted forward is not the default here'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
print('stale families at submission:', wf.stale_families())
" || exit 1

# The kernels are the production configuration the floors describe.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# The script pins each arm through MBIRTORCH_NUM_DEVICES in its own
# subprocess; a value inherited from this shell would collapse every arm
# to one count.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_RECOMPILE_LIMIT

# The staged inputs and per-arm rows go to scratch where they can be
# inspected after the run, rather than into the checkout.
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg48_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The plan first, so the log opens with the cells and counts this run
# will measure and a wrong scope is caught before any GPU work.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py
rc=$?
echo "MG48 DONE rc=$rc"
exit $rc
