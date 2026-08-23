#!/bin/bash
# mg58: attribute the recon setup's host-side validation cost, and time the
# same check on a device.  Follows mg57, which measured initialize_recon at
# 2.6 to 2.7 seconds warm and unchanged between one device and four.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:20:00
#SBATCH -J torch_mg58
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg58_%j.log
set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
export MG58_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg58_probe.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg58_probe.py
echo "MG58 DONE rc=$?"
