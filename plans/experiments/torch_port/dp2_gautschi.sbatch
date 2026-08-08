#!/bin/bash
# dp2: memory-ledger calibration on 1 H100 -- modeled against measured
# per-device peaks at the four gate cells, n=1, kernels on, warm seeded
# 3-iteration vcd with weights supplied (the composed-gate protocol), plus
# one unweighted arm at parallel-1024 to size the hess_weights residency
# that a weighted run cannot show.  See dp2_ledger_calib.py.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch dp2_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J torch_dp2
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/dp2_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

# The kernels are the shipped default and this reading is the kernels-on
# cell, so the kill switch is explicitly OFF rather than merely unset.
# Protocol note: compile_mode='off' does NOT disable the Triton kernels.
# Selection is availability-driven, not compile-driven, so an arm that intends
# the plain torch engine must set MBIRTORCH_DISABLE_TRITON=1 or bind the torch
# bodies itself.  This cost a full investigation once; it is stated here so no
# future arm silently measures kernels while believing it measured eager.
export MBIRTORCH_DISABLE_TRITON=0
export DP2_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$DP2_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u dp2_ledger_calib.py
echo "DP2 DONE"
