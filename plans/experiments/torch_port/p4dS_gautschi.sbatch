#!/bin/bash
# n=4 CUDA value-anomaly discriminator (projector restructure).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p4a3b_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 00:45:00
#SBATCH -J torch_p4dS
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p4dS_%j.log

set -x
export CUDA_LAUNCH_BLOCKING=1
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4" || exit 1
"$TORCHPY" -u ../p4d_n4_bisect.py
echo "P4D DONE"
