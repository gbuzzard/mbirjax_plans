#!/bin/bash
# mg34: THE COMPOSED A/B GATE ON THE SORTED-CONTRACTION FORWARD.  Four H100s.
# The full statement is in mg34_sorted_ab.py.
#
# WHAT THIS JOB IS.  The sorted-contraction forward route is STAGED in the
# library (triton_parallel.py: the wrapper's default route, with
# MBIRTORCH_SORTED_FORWARD=0 restoring the per-tap kernel).  The spikes
# measured 2.8x to 4.0x per launch (findings 1.29 and 1.30 in the plans
# repository), and a kernel spike's win is not the driver's, so nothing
# ships without a composed gate.  This job is that gate's first half: the
# FULL GPU TEST SUITE first -- including the sorted route's own value
# gates, its sparse-set fallback, and its view-chunk tail -- and then the
# composed A/B at the 512- and 1024-class parallel cells, one, two, and
# four devices, sorted against per-tap, reusing mg27's staged sinograms.
# The 2048-class confirmation follows separately, the mg24 pattern.
#
# THE TREE UNDER TEST is 64dedb8 plus the staged sorted-forward change
# (triton_parallel.py and two test files), synced per file with md5
# verification before submission.
#
# WALL ESTIMATE.  The GPU suite runs about 10 minutes.  Twelve composed
# arms at mg27's protocol: the slowest (1024-class per-tap n=1) is about
# 40 s a pass, and the sorted arms are faster, so the arms total roughly
# 35 minutes with staging reused.  02:30:00 is requested for margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:30:00
#SBATCH -J torch_mg34
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg34_%j.log
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
unset MG34_SMOKE
unset MG34_DRY
unset MG34_VARIANTS
unset MG34_REPEATS

# ── the suite leg: every gate in tests/, on the GPUs ──
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest tests/ -q 2>&1 | tail -5
suite_rc=${PIPESTATUS[0]}
echo "SUITE rc=$suite_rc"
if [ "$suite_rc" -ne 0 ]; then
    echo "the GPU suite failed; the composed A/B would gate a broken tree, so the job stops here"
    exit "$suite_rc"
fi

# ── the composed A/B, reusing mg27's staged sinograms ──
export MG34_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg27_reference
mkdir -p "$MG34_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
export MG34_DRY=1
"$TORCHPY" -u mg34_sorted_ab.py || exit 1
unset MG34_DRY
"$TORCHPY" -u mg34_sorted_ab.py
rc=$?
echo "MG34 DONE rc=$rc"
exit $rc
