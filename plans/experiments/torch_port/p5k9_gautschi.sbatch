#!/bin/bash
# The linkage check: are the direct-recon divergent SLICES a subset of the
# PHANTOM-diff slices?  Direct recon is per-slice separable under parallel
# beam, so the phantom explanation predicts they must be -- and a divergent
# slice with no phantom difference would mean a SECOND cause (see
# p5k9_phantom_link.py).  Reads the p5k8 artifacts; run it AFTER that job.
# NO TIMING.  Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k9_gautschi.sbatch
#
# Optional overrides -- export in the SUBMITTING SHELL, never in an
# `--export=ALL,VAR=a,b` list (slurm splits it on commas):
#   P5K9_CELLS=1024        the divergent cell only
#   P5K9_THRESHOLD=0.01    divergence threshold, fraction of max
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_p5k9
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k9_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg
export MBIRTORCH_DISABLE_TRITON=1

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5k9_phantom_link.py
echo "P5K9 LINKAGE DONE"
