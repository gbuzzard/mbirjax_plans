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
# --no-deps keeps the install hermetic on compute nodes (non-torch deps such
# as psutil are installed once from the login node, which has network).
# The checkout is staged as mbirtorch_SRC: a directory named mbirtorch
# anywhere on sys.path shadows the installed package as an empty namespace
# package.  python -c puts the CWD first; python script.py puts the SCRIPT'S
# directory first -- so both the run directory and the script's directory
# must be free of a dir named mbirtorch (runs 1-2's
# "no attribute ConeBeamModel" torch failures).
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1

mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available(); assert hasattr(mbirtorch, 'ConeBeamModel')" || exit 1

export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P0_TORCH_DEVICES=cuda
"$TORCHPY" -u ../p3_cone_readout.py
echo "P3 DONE"
