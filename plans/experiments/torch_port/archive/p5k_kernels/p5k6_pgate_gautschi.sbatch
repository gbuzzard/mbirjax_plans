#!/bin/bash
# The COMPOSED gate for the Triton PARALLEL pair on 1 H100 -- the seeded
# 3-iteration warm parallel VCD recon at (512,448,384) and (1024,1008,992),
# five variants per cell, each in its own process (see p5k6_pgate.py):
#   torch_kernel (PBACK=1 + PFWD=1), torch_back_only (PBACK=1), torch_body
#   (MBIRTORCH_DISABLE_TRITON=1 -- the honest current baseline),
#   torch_body_repeat (the value floor), jax.
# This is the pair's gate; p5k6_psweep_gautschi.sbatch only ranks the constants
# that feed it.  Budget: 10 recons per cell (cold + warm per variant); split by
# cell with P5K6_CELLS, or drop an arm, if the queue is tight.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k6_pgate_gautschi.sbatch
#
# Re-run with the swept winners -- export in the SUBMITTING SHELL:
#   P5K6_PBACK_CONSTANTS='16,256,8,1' P5K6_PFWD_CONSTANTS='8,128,8,1' \
#     sbatch p5k6_pgate_gautschi.sbatch
# NEVER `sbatch --export=ALL,P5K6_PBACK_CONSTANTS='16,256,8,1' ...`: slurm
# splits that list on commas and the tuple arrives mangled (measured
# 2026-08-06).  sbatch exports the submitting environment by default, so the
# shell form needs no --export; if an --export list is unavoidable, the script
# also accepts the semicolon form '16;256;8;1', which survives the split.
# Other optional overrides (same rule -- P5K6_CELLS is a comma list too):
#   P5K6_CELLS=512, P5K6_ITERATIONS=3, P5K6_SKIP_JAX=1,
#   P5K6_SKIP_FLOOR=1 (drops the repeat arm),
#   P5K6_SKIP_BACK_ONLY=1 (drops the arm that separates the two kernels)
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:00:00
#SBATCH -J torch_p5k6g
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k6g_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

# MBIRTORCH_ENABLE_TRITON_PBACK / _PFWD and MBIRTORCH_DISABLE_TRITON are set
# PER VARIANT by the script, explicitly on every torch arm, so none is exported
# here -- an inherited value would blur the arms.  The script verifies each arm
# against the bodies the MODEL bound rather than trusting these variables.
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5k6_pgate.py
echo "P5K6 PGATE DONE"
