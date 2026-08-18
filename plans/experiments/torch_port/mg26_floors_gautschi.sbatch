#!/bin/bash
# mg26: the widening-floors refresh on the padded tree.
#
# WHY THIS JOB EXISTS.  The kernel width padding landed as mbirtorch
# 64dedb8 on 2026-08-18: the hand-written cone and parallel kernels now
# round their width-class launch arguments up to the next multiple of 16,
# and the non-divisible bands that ran at half rate run at the divisible
# rate (multigpu_findings.md section 1.23).  That changes what a cone or
# parallel projection costs exactly at the device counts whose bands were
# not divisible, so the measured device-count thresholds -- the widening
# speed floors in mbirtorch/_widening_floors.py -- are owed a re-measure.
# The current table's rows were measured 2026-08-17 on the unpadded
# 6d90601 tree (mg22, job 15327855), and the staleness machinery names
# the moved cost inputs on every automatic device selection until a new
# paste clears it.
#
# The same rows feed the threshold-coarsening proposal: Greg ruled on
# 2026-08-18 for fewer, coarser thresholds (open item E3), and the ruling
# notes the proposal should follow this re-measure rather than precede
# it.  One run feeds both.
#
# WHAT MOVES AND WHAT DOES NOT.  The padding has a site only where a
# hand-written kernel takes a width-class launch argument, which is cone
# and parallel in both directions.  The cone rows at counts with
# non-divisible bands are the rows expected to move.  Multiaxis and
# translation run compiled torch bodies with no such kernel, so their
# costs did not change; the refresh script still re-measures whatever its
# plan derives from the current table, and the fresh rows land with
# current provenance either way.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  Submit it with
# --dependency=afterany:<job id> behind any such job already queued.
#
# WALL ESTIMATE.  mg22, the same refresh on the unpadded tree, ran 3
# hours 9 minutes on four H100s, with the multiaxis sentinel probes at
# the 1024-class cell the most expensive part.  The padding makes the
# cone arms faster and leaves multiaxis unchanged, so this run should
# come in at or under mg22's wall.  Six hours is requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 06:00:00
#SBATCH -J torch_mg26
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg26_%j.log
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
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The checkout must be the padded tip, because the padded kernels are the
# code these floors price.  padded_kernel_width is the one definition of
# the rounding rule, and 504 -> 512 is the production band the padding
# exists for.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
msg = 'the tree is not the padded 64dedb8 tip this refresh prices'
assert padded_kernel_width(504) == 512, msg
assert padded_kernel_width(512) == 512, msg
" || exit 1

# The kernels are the production configuration the floors describe.
export MBIRTORCH_DISABLE_TRITON=0
# The script pins each projection variant through MBIRTORCH_NUM_DEVICES
# in its own subprocess and places each denoiser variant with
# configure_devices; a value inherited from this shell would collapse
# every variant to one count.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION

# The staged inputs and per-variant rows go to scratch where they can be
# inspected after the run, rather than into the checkout.
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg26_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The plan first, so the log opens with the cells and counts this run
# will measure and a mis-set variable is caught before any GPU work.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py
rc=$?
echo "MG26 DONE rc=$rc"
exit $rc
