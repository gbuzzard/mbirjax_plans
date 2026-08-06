#!/bin/bash
# Phase 4 spike 2 on gautschi: the substrate discriminator on 2 H100s --
# threaded dual-GPU scaling (the GIL question), p2p band-copy bandwidth, and
# NCCL broadcast/reduce at band sizes.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p4_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 00:30:00
#SBATCH -J torch_p4s2
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p4s2_%j.log

set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache

mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import torch; assert torch.cuda.device_count() >= 2" || exit 1

"$TORCHPY" -u ../p4s2_dual_gpu.py
"$TORCHPY" -m torch.distributed.run --nproc_per_node=2 ../p4s2_dual_gpu.py nccl
echo "P4S2 DONE"
