#!/bin/bash
# The decisive element-wise probe: do the two frameworks' gate-protocol
# PHANTOMS and SINOGRAMS differ at the six recon-divergent slices/rows, and is
# either chain z-flipped?  Two reductions disagreed about the phantom, so this
# stops reducing and compares element-wise -- torch writes phantom+sinogram,
# jax loads them beside its own and computes every difference itself (see
# p5kb_sino_rows.py).  NO TIMING.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5kb_gautschi.sbatch
#
# Optional overrides -- export in the SUBMITTING SHELL, never in an
# `--export=ALL,VAR=a,b` list (slurm splits it on commas):
#   P5KB_CELLS=512,1024   add the 512 control
#   P5KB_KEEP=1           keep the ~8 GiB scratch arrays for follow-up
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
# No --mem line: gautschi allocates memory proportionally to the GPU/CPU
# request (1 GPU, 14 CPUs -> ~129 GB) and REJECTS an explicit --mem.  The
# comparer's peak is ~16 GB (two volumes, then two sinograms), well inside.
#SBATCH -t 01:00:00
#SBATCH -J torch_p5kb
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5kb_%j.log

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
"$TORCHPY" -u ../p5kb_sino_rows.py
echo "P5KB PROBE DONE"
