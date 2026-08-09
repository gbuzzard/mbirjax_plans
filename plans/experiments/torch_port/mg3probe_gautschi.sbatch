#!/bin/bash
# mg3 PRICING PROBE on 1 H100 -- exactly two measured arms (torch eager-plain,
# n=1, 3 iterations, at each 1024 cell) plus their two untimed generators, then
# a projection of the full mg3 job priced from mg3_value.py's OWN plan.
#
# This is the probe block mg3_gautschi.sbatch's header prescribes, materialized
# as its own file so the submitter can chain it:
#     sbatch --dependency=afterany:<mg2 jobid> mg3probe_gautschi.sbatch
# The full mg3 job is submitted ONLY after this probe's projection is read
# (multigpu_plan.md §4 mg3, §5 increment 2, §7), with the walltime -- or the
# split the printout recommends -- taken from the projection.
#
# The pip install below rewrites TORCHPY's editable .pth, so this job must
# never run concurrently with another job importing mbirtorch from that env
# (the recorded torn-.pth race): chain it, never co-schedule it.
#
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 01:00:00
#SBATCH -J torch_mg3probe
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg3probe_%j.log
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
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# The probe's two arms are eager-plain, and the script sets the per-arm env
# itself (MBIRTORCH_DISABLE_TRITON=1 on those arms); nothing is exported here.
unset MBIRTORCH_DISABLE_TRITON
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION

export MG3_PROBE=1
export P0_TORCH_PYTHON=$TORCHPY
export MG3_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG3_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg3_value.py
echo "MG3PROBE DONE"
