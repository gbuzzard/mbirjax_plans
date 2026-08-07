#!/bin/bash
# Full mbirtorch test suite on a gautschi GPU node (post projector
# restructure): the same suite that gates locally, on CUDA hardware.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p4b_suite_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J torch_suite
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/suite_%j.log

set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest tests/ -q
echo "SUITE EXIT $?"
