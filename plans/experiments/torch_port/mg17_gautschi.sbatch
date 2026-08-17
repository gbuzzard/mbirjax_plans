#!/bin/bash
# mg17: THE 2048-CLASS CAPACITY TABLE, on one H100.  The full statement is in
# mg17_capacity_table.py; the essentials:
#
# This job runs NO reconstruction.  It evaluates the library's own closed-form
# memory ledger (plan_from_model / estimate_peak_device_bytes) for cone and
# parallel at the 2048-class reference shape (2048, 2016, 1984), at device
# counts one through eight, four combining-step variants per count, plus
# 1024-class anchor rows and the two torch-body geometries' production shapes
# as an appendix.  The GPU is needed only so the kernel availability probe
# binds the production Triton bodies (a CPU evaluation would price cone and
# parallel as torch bodies) and so the device budget is read from a real idle
# H100.
#
# WALL ESTIMATE.  Model construction and plan evaluation are seconds per row;
# the ~90 rows plus one torch import land the whole job under five minutes.
# The walltime below is 00:20:00, comfortably over twice that.
#
# DISK.  One small jsonl in $MG17_RESULTS.  Nothing else is written.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg17_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:20:00
#SBATCH -J torch_mg17
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg17_%j.log
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
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 1, torch.cuda.device_count(); print('library under test:', mbirtorch.__file__)" || exit 1

# The kernels must be selectable: this table prices the production
# configuration, and the in-script witness raises if a cone or parallel row
# binds a torch body.
export MBIRTORCH_DISABLE_TRITON=0
# The ledger evaluation reads no peak counter and pins no count; both are
# cleared so the evaluation runs exactly as a user's process would.
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset MG17_SMOKE

export MG17_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG17_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg17_capacity_table.py
rc=$?
echo "MG17 DONE rc=$rc"
exit $rc
