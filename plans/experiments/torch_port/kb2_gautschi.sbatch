#!/bin/bash
# kb2: the view-chunk sweep for kernel-aware view batching on 1 H100 --
# driver-level rows over both geometries, directions, gate cells, and pixel
# classes; picks the four *_VIEW_CHUNK constants.  See kb2_vbsweep.py.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch kb2_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.  This
# job needs only P0_TORCH_PYTHON, set below.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:30:00
#SBATCH -J torch_kb2
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/kb2_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg
export P0_TORCH_PYTHON="$TORCHPY"

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u kb2_vbsweep.py
echo "KB2 DONE"
