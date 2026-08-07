#!/bin/bash
# Job D, the ruler repair: the composed gate's VALUE comparison re-run with a
# SHARED sinogram -- one phantom, one forward projection, one .npy, and every
# arm (both-kernels / pure-torch-body / jax) reconstructs THAT array at the
# 1024 cell (see p5ka_shared_gate.py).  VALUE ONLY: Job A's table stands for
# timing, and the wall times here are incidental.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5ka_gautschi.sbatch
#
# Optional overrides -- export in the SUBMITTING SHELL, never in an
# `--export=ALL,VAR=a,b` list (slurm splits it on commas):
#   P5KA_CELLS=512,1024     add the 512 control
#   P5KA_GENERATOR=jax      let jax build the shared input instead
#   P5KA_KEEP_SINO=1        keep the ~4 GiB shared sinogram on scratch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p5ka
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5ka_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1

# MBIRTORCH_ENABLE_TRITON_PBACK / _PFWD and MBIRTORCH_DISABLE_TRITON are set
# PER ARM by the script and verified against the bodies the model bound.
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5ka_shared_gate.py
echo "P5KA SHARED GATE DONE"
