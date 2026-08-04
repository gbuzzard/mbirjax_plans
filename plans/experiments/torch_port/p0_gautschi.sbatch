#!/bin/bash
# Phase 0 CUDA cells on gautschi (port_plan.md section 6): spikes 1-3 on one
# H100.  Staging layout (created by the submitter):
#   /scratch/gautschi/buzzard/torch_p0/scripts/   this file + the p0s*.py spikes
#   /scratch/gautschi/buzzard/torch_p0/env/       torch venv (created below on
#                                                 first run -- inside the job,
#                                                 so no login-node compute)
# Submit from the login node:
#   cd /scratch/gautschi/buzzard/torch_p0/scripts && sbatch p0_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p0
#SBATCH -o /scratch/gautschi/buzzard/torch_p0/p0_%j.log

set -x
STAGE=/scratch/gautschi/buzzard/torch_p0
source ~/load_conda_cuda.sh                     # conda on PATH, cuda module, proxy

# The jax baseline runs in the existing interactive env (GPU jax + mbirjax).
JAXPY=$(conda run -n mbirjax which python)

# Torch venv on scratch (home quota is 25 GB and fails silently): seeded from
# the mbirjax env's python, torch cu wheels bundle their CUDA libs.
if [ ! -x "$STAGE/env/bin/python" ]; then
    "$JAXPY" -m venv "$STAGE/env"
    "$STAGE/env/bin/pip" install --no-cache-dir torch numpy
fi
TORCHPY=$STAGE/env/bin/python

# Preflight (cluster_use.md): catch a silent CPU fallback on either framework
# and record which library states are running.
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu', \
f'NOT ON GPU: {jax.devices()}'; print('jax', jax.__version__, 'mbirjax at', mbirjax.__file__)" || exit 1
"$TORCHPY" -c "import torch; assert torch.cuda.is_available(), 'torch: no CUDA'; \
print('torch', torch.__version__, torch.cuda.get_device_name(0))" || exit 1

cd "$STAGE/scripts" || exit 1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P0_TORCH_DEVICES=cuda

"$TORCHPY" -u p0s1_fan_kernels.py
"$TORCHPY" -u p0s2_chain_fusion.py
"$TORCHPY" -u p0s3_subset_graph.py
echo "ALL SPIKES DONE"
