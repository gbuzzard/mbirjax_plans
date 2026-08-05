#!/bin/bash
# Phase 3 on gautschi: the cone-beam gate-cell readout (both frameworks, one
# ruler, warm-vcd protocol) on one H100 -- the p2 protocol with the cone
# geometry (full-circle angles, sdd = 4 * channels, magnification 2).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p3_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J torch_p3
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p3_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

# Keep the inductor cache on scratch, not in the 25 GB home quota.
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache

"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q -e /scratch/gautschi/buzzard/torch_p3/mbirtorch
"$TORCHPY" -c "import torch, mbirtorch; assert torch.cuda.is_available()" || exit 1

cd /scratch/gautschi/buzzard/torch_p3 || exit 1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P0_TORCH_DEVICES=cuda
"$TORCHPY" -u p3_cone_readout.py
echo "P3 DONE"
