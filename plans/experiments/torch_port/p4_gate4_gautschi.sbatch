#!/bin/bash
# Phase 4 gate readout on gautschi: both frameworks, n=1/2 on 2 H100s
# (a separate 4-GPU submission probes n=4 when account headroom allows).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p4_gate_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 03:00:00
#SBATCH -J torch_p4g4
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p4gate_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache

"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2" || exit 1

export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P4_MAX_DEVICES=4
"$TORCHPY" -u ../p4_gate_readout.py
echo "P4GATE DONE"
