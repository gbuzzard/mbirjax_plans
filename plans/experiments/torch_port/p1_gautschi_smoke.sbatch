#!/bin/bash
# Phase 1 CUDA smoke (port_plan.md section 5, Phase 1 gates): run the mbirtorch
# test suite and the demo on one H100.  Reuses the Phase 0 torch venv on
# scratch; the repo + goldens are rsynced by the submitter to
#   /scratch/gautschi/buzzard/torch_p1/mbirtorch/
# Submit from the login node:
#   cd /scratch/gautschi/buzzard/torch_p1 && sbatch p1_gautschi_smoke.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:30:00
#SBATCH -J torch_p1
#SBATCH -o /scratch/gautschi/buzzard/torch_p1/p1_%j.log

set -x
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
REPO=/scratch/gautschi/buzzard/torch_p1/mbirtorch

"$TORCHPY" -c "import torch; assert torch.cuda.is_available(), 'torch: no CUDA'; \
print('torch', torch.__version__, torch.cuda.get_device_name(0))" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q pytest

cd "$REPO" || exit 1
PYTHONPATH=$REPO "$TORCHPY" -m pytest tests -q -s
PYTHONPATH=$REPO "$TORCHPY" demo/demo_1_shepp_logan.py
echo "P1 SMOKE DONE"
