#!/bin/bash
# mg27: re-anchor the reference timings of the cross-framework comparison.
#
# WHY THIS JOB EXISTS.  Every speedup ratio in the comparison tables of
# execution_overview.md section 5 divides by a single-device reference
# timing, and the whole 512-class column was measured 2026-08-09.  Three
# changes since then moved what a reconstruction costs: the column-gather
# forward default (2026-08-11), the forward pixel-batch default of 32768
# (2026-08-17), and the kernel width padding (mbirtorch 64dedb8,
# 2026-08-18).  This job re-measures the full mbirtorch column at the
# current tip: parallel and cone, the 512-class and 1024-class cells,
# one, two and four devices, under mg1's exact reconstruction protocol
# (open item C1).  The same numbers refresh the user-docs timing table in
# usr_multi_gpu.rst, which still shows the pre-kernel-era 94 s for the
# 1024-class single-device reconstruction (open item F1).  The mbirjax
# column is not re-measured; it has not moved during the campaign.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  Submit it with
# --dependency=afterany:<job id> behind any such job already queued.
#
# WALL ESTIMATE.  The slowest variant is the 1024-class cone
# single-device reconstruction at about 62 s; its cold pass and three
# warm passes are about 4 minutes.  Twelve variants with their staging,
# model builds and per-variant compiles total roughly 45 minutes.  Two
# hours is requested for margin.
#
# DISK.  The four staged sinograms total about 8.3 GiB under
# $MG27_RESULTS and are kept, so a repeat run reconstructs identical
# arrays; plus one jsonl of rows.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg27
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg27_%j.log
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
# NEVER run two such jobs at once: a second concurrent install races this
# one and leaves a torn .pth.  Chain instead, with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The checkout must be the padded tip, because the padding is one of the
# three changes these references exist to absorb.  padded_kernel_width is
# the one definition of the rounding rule, and 504 -> 512 is the
# production band the padding exists for.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
msg = 'the tree is not the padded 64dedb8 tip these references describe'
assert padded_kernel_width(504) == 512, msg
assert padded_kernel_width(512) == 512, msg
" || exit 1

# The kernels are the production configuration the references describe.
export MBIRTORCH_DISABLE_TRITON=0
# The script pins each variant through MBIRTORCH_NUM_DEVICES in its own
# subprocess; a value inherited from this shell would collapse every
# variant to one count.  The calibration mode owns the peak counters this
# run reads itself.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG27_SMOKE
unset MG27_DRY
unset MG27_VARIANTS
unset MG27_REPEATS

export MG27_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg27_reference
mkdir -p "$MG27_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The plan first, so the log opens with the sixteen jobs in run order and
# a mis-set variable is caught before any GPU work.
export MG27_DRY=1
"$TORCHPY" -u mg27_reference_timings.py || exit 1
unset MG27_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what
# they measured; the three paste tables are in this log.
"$TORCHPY" -u mg27_reference_timings.py
rc=$?
echo "MG27 DONE rc=$rc"
exit $rc
