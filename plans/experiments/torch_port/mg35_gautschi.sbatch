#!/bin/bash
# mg35: THE 2048-CLASS CONFIRMATION OF THE SORTED FORWARD.  Four H100s.
# The full statement is in mg35_sorted_2k.py.
#
# WHAT THIS JOB IS.  mg34 passed the gate's first half (the full GPU suite,
# then the composed A/B at the 512- and 1024-class cells, 1.03x to 1.89x
# with every value gate holding).  This job is the second half, the mg24
# pattern: the same A/B at the 2048-class parallel cell, three and four
# devices (one and two refuse by the capacity table), where the 24 GB
# values block is the sorted route's unmeasured territory.  The value gate
# is the mode pair at each count, 1e-3.
#
# THE TREE UNDER TEST is 64dedb8 plus the staged sorted-forward change,
# already synced and suite-passed by mg34.  The suite leg is not repeated.
#
# WALL ESTIMATE.  Staging the 33 GB sinogram at four devices runs about
# ten minutes.  Eight timed arms: the per-tap references measured 299 s
# (n=3) and 216 s (n=4) a pass at this cell, the sorted arms should run
# under them, and each arm is a cold pass plus three warm -- about 75
# minutes of arms.  03:30:00 is requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:30:00
#SBATCH -J torch_mg35
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg35_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once; chain with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The tree must carry the staged sorted-forward change.
"$TORCHPY" -c "
from mbirtorch.triton_parallel import sorted_forward_enabled
from mbirtorch._utils import padded_kernel_width
assert sorted_forward_enabled(), 'the sorted route must default ON'
assert padded_kernel_width(504) == 512
" || exit 1

export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
unset PYTHONPATH
unset MG35_SMOKE
unset MG35_DRY
unset MG35_VARIANTS
unset MG35_REPEATS

# ── the composed A/B; the 2048 sinogram stages fresh here ──
export MG35_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg35_2k
mkdir -p "$MG35_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
export MG35_DRY=1
"$TORCHPY" -u mg35_sorted_2k.py || exit 1
unset MG35_DRY
"$TORCHPY" -u mg35_sorted_2k.py
rc=$?
echo "MG35 DONE rc=$rc"
exit $rc
