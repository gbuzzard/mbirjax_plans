#!/bin/bash
# Value-divergence attribution for the parallel gate on 1 H100 -- why
# torch-vs-jax recon samples agree to 5.5e-4 at the 512 cell and diverge by rel
# 0.375 at 1024, in the PURE-BODY arm as much as in the kernel arm (see
# p5k7_value_attrib.py).  NO TIMING is measured here; every number is a value
# and every comparison varies exactly one thing.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k7_gautschi.sbatch
#
# Optional overrides -- export them in the SUBMITTING SHELL, e.g.
#   P5K7_CELLS=1024 sbatch p5k7_gautschi.sbatch
# and NEVER as `sbatch --export=ALL,P5K7_CELLS=512,1024 ...`: slurm splits that
# list on commas, so a multi-field value arrives mangled (measured 2026-08-06).
#   P5K7_CELLS=1024           the divergent cell only
#   P5K7_ITERATIONS=3         iteration depth of the trace
#   P5K7_SKIP_ABLATION=1      measurements 1-3 only
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p5k7
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k7_%j.log

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
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

# The Triton kernels are irrelevant to this question -- the divergence shows in
# the pure-body arm -- so the kill switch is set: the trace measures the torch
# BODIES against jax, with one less moving part.
export MBIRTORCH_DISABLE_TRITON=1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5k7_value_attrib.py
echo "P5K7 ATTRIB DONE"
