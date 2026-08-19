#!/bin/bash
# mg40: the first family-scoped widening-floors refresh, on the sorted
# parallel forward.
#
# WHY THIS JOB EXISTS.  The channel-sorted parallel forward kernel shipped
# as mbirtorch c761b24 on 2026-08-18 and changed triton_parallel.py alone
# (multigpu_findings.md sections 1.30 and 1.31).  That file prices only
# the parallel family's floors, so the parallel rows in
# mbirtorch/_widening_floors.py are owed a re-measure and every other
# family's rows are not.  The refresh tool's new --families mode measures
# exactly the stale families and carries the rest verbatim; this job is
# that mode's first production run, and its scope should resolve to the
# parallel family alone.  The coarse admission rule (a floor needs a
# 1.15x win; ruled 2026-08-19) is already in the tool, so the rows this
# run prints are coarse rows.
#
# WHAT CAN MOVE.  The sorted kernel makes the forward faster at every
# device count, so the crossovers can move in either direction; the
# current coarse table has parallel n=2 at the 512-class and n=4 at the
# 1024-class.  The plan brackets those floors: the 384- through
# 768-class cells at counts 1 and 2, and the 768- and 1024-class cells
# at counts 1, 2 and 4.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  Submit it first and
# chain the mg27 re-run behind it with --dependency=afterany:<this id>.
#
# WALL ESTIMATE.  The scoped plan is 4 staged sinograms and 10 timed
# arms, all parallel beam.  The most expensive arms are the 1024-class
# cells at about 21 s per warm reconstruction plus per-arm compiles.
# mg26's full refresh ran just under 3 hours for a 29-row plan dominated
# by multiaxis; this plan is about a sixth of that work.  Two hours is
# requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg40
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg40_%j.log
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
# channel-sorted kernels, so both changes must be present, and the only
# stale family must be parallel -- anything else means the sync missed a
# file and the scope would be wrong.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
from mbirtorch.triton_parallel import sorted_forward_enabled
from mbirtorch import _widening_floors as wf
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert sorted_forward_enabled(), 'the sorted forward is not the default here'
assert wf.stale_families() == ['parallel'], wf.stale_families()
" || exit 1

# The kernels are the production configuration the floors describe, and
# the sorted route must take its shipped default.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# The script pins each arm through MBIRTORCH_NUM_DEVICES in its own
# subprocess; a value inherited from this shell would collapse every arm
# to one count.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION

# The staged inputs and per-arm rows go to scratch where they can be
# inspected after the run, rather than into the checkout.
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg40_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The scoped plan first, so the log opens with the cells and counts this
# run will measure and a wrong scope is caught before any GPU work.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan --families || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --families
rc=$?
echo "MG40 DONE rc=$rc"
exit $rc
