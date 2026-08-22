#!/bin/bash
# mg56: the full widening-floors refresh on the tree with the multiaxis
# Triton kernels.
#
# WHY THIS JOB EXISTS.  The multiaxis geometry now selects hand-written
# forward and back kernels wherever their per-device value self-check
# passes (commit c024ec9), which changed what a multiaxis projection
# costs, and triton_multiaxis.py joined the multiaxis family's cost
# inputs with no recorded hash.  Six other cost inputs had already moved
# in earlier commits (the denoiser performance work and the sharding
# changes among them), so every family's rows are stale and carrying any
# family verbatim would assert costs that did move.  This is therefore
# the full refresh, every family at every planned row, exactly as mg48
# was on the recompile-remedied tree.
#
# WHAT CAN MOVE.  The multiaxis floors were measured on the torch
# bodies.  On the kernels the one-device walls fell about four-fold
# while the multi-device speedups at the 1024-class read 1.81x at two
# devices and 2.97x at four, so the multiaxis knees can move in either
# direction and their rows are the ones this run exists to re-measure.
# Parallel, cone, and translation re-measure under the drifted shared
# inputs and should broadly reproduce their recorded rows.  The
# denoiser rows re-test their sentinels after the denoiser performance
# commit.  The verdicts and the paste-ready rows land in this log;
# NOTHING is pasted into the library by this job, because the paste,
# the commit stamp, and the row notes are the human step.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  The queue holds none at
# submission.
#
# WALL ESTIMATE.  mg48, the same full scope on the remedied tree, ran
# 1 hour 34 minutes, dominated by the multiaxis large-cell arms, which
# the kernels have since cut about four-fold.  Three hours are
# requested for margin, because a moved floor can add bracket cells to
# the plan.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:00:00
#SBATCH -J torch_mg56
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg56_%j.log
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
# channel-sorted kernels, the recompile remedy, AND the multiaxis
# kernel selection; all four must be present.  The stale set is printed
# rather than asserted: the full scope re-measures everything either
# way.
"$TORCHPY" -c "
import inspect
from mbirtorch._utils import padded_kernel_width
from mbirtorch.triton_parallel import sorted_forward_enabled
from mbirtorch import projectors
from mbirtorch import _widening_floors as wf
import mbirtorch.triton_multiaxis
from mbirtorch.multiaxis_parallel import MultiAxisParallelModel
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert sorted_forward_enabled(), 'the sorted forward is not the default here'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
selection = inspect.getsource(MultiAxisParallelModel._view_batch_bodies)
assert 'multiaxis_back_kernel_usable' in selection \
    and 'multiaxis_forward_kernel_usable' in selection, \
    'the multiaxis selection does not consult the kernel availability checks'
print('stale families at submission:', wf.stale_families())
" || exit 1

# The kernels are the production configuration the floors describe,
# now including the multiaxis pair.
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
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg56_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The plan first, so the log opens with the cells and counts this run
# will measure and a wrong scope is caught before any GPU work.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py
rc=$?
echo "MG56 DONE rc=$rc"
exit $rc
