#!/bin/bash
# dp3: phase-resolved attribution of the memory-ledger under-charge, on 1
# H100.  The dp2 calibration table reported the four gate cells 0.80-1.03 of
# their measured peaks; one number per cell cannot say which phase is short.
# This runs the probe on both geometries at the 1024 cell, plus the
# unweighted parallel arm, reusing the sinograms dp2 already cached.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch dp3_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J torch_dp3
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/dp3_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

export MBIRTORCH_DISABLE_TRITON=0
export MBIRTORCH_MEMORY_CALIBRATION=1
export DP3_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

DP3_GEOMETRY=parallel DP3_VIEWS=1024 DP3_WEIGHTED=0 "$TORCHPY" -u dp3_phase_probe.py
DP3_GEOMETRY=parallel DP3_VIEWS=1024 DP3_WEIGHTED=1 "$TORCHPY" -u dp3_phase_probe.py
echo "DP3 DONE"
