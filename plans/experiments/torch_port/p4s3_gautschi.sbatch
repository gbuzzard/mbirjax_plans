#!/bin/bash
# Phase 4 spike 3: the sharded engine dispatch measurement on 2 H100s.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p4s3_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 00:45:00
#SBATCH -J torch_p4s3
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p4s3_%j.log

set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2; assert hasattr(mbirtorch, 'ConeBeamModel')" || exit 1
"$TORCHPY" -u ../p4s3_engine_dispatch.py
echo "P4S3 DONE"
