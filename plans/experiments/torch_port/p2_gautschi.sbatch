#!/bin/bash
# Phase 2 CUDA gate readout on gautschi (port_plan.md section 4): both
# frameworks, one ruler, the GPU metrics cells, one H100.  Reuses the Phase 0
# torch venv; the mbirtorch repo is rsynced to torch_p1 by the submitter and
# installed into the venv here (editable, so reruns pick up the rsync).
# Submit: cd /scratch/gautschi/buzzard/torch_p2 && sbatch p2_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p2
#SBATCH -o /scratch/gautschi/buzzard/torch_p2/p2_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu', \
f'NOT ON GPU: {jax.devices()}'; print('jax', jax.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q -e /scratch/gautschi/buzzard/torch_p1/mbirtorch
"$TORCHPY" -c "import torch, mbirtorch; assert torch.cuda.is_available(); \
print('torch', torch.__version__, 'mbirtorch at', mbirtorch.__file__)" || exit 1

cd /scratch/gautschi/buzzard/torch_p2 || exit 1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P0_TORCH_DEVICES=cuda
"$TORCHPY" -u p2_gate_readout.py
echo "P2 READOUT DONE"
