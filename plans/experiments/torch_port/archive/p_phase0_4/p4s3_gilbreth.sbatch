#!/bin/bash
# Phase 4 spike 3 on gilbreth: the sharded engine dispatch measurement on two
# A100-SXM4 (--constraint=N pins the NVLink class; A100 walls are NOT
# comparable to the H100 gates -- this job answers the dispatch/scaling
# question, not absolute time).
# Submit: cd /scratch/gilbreth/buzzard/torch_p4 && sbatch p4s3_gilbreth.sbatch
#SBATCH -A bouman
#SBATCH -p a100-40gb
#SBATCH --constraint=N
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH -n 28
#SBATCH --mem=80G
#SBATCH -t 00:45:00
#SBATCH -J torch_p4s3g
#SBATCH -o /scratch/gilbreth/buzzard/torch_p4/p4s3_%j.log

set -x
TORCHPY=/scratch/gilbreth/buzzard/torch_p4/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gilbreth/buzzard/torch_p4/torch_cache

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gilbreth/buzzard/torch_p4/mbirtorch_src || exit 1
mkdir -p /scratch/gilbreth/buzzard/torch_p4/run
cd /scratch/gilbreth/buzzard/torch_p4/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2; assert hasattr(mbirtorch, 'ConeBeamModel')" || exit 1
"$TORCHPY" -u ../p4s3_engine_dispatch.py
echo "P4S3G DONE"
