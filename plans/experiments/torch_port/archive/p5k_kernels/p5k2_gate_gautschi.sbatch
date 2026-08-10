#!/bin/bash
# The COMPOSED gate for the Triton cone kernels on 1 H100 -- the seeded
# 3-iteration warm cone VCD recon at (512,448,384) and (1024,1008,992), five
# variants per cell, each in its own process (see p5k2_gate.py):
#   torch_kernel (forward opt-in ON + back default-on), torch_back_only
#   (today's shipped configuration -- the K4 baseline), torch_body (neither
#   kernel), torch_body_repeat (the value floor), jax.
# It gated the back kernel at K2 and now gates the FORWARD kernel at K4; the
# sweeps (p5k2_gautschi.sbatch, p5k4_fwd_gautschi.sbatch) only rank the
# constants that feed it.  Budget: 10 recons per cell (cold + warm per
# variant); split by cell with P5K2_CELLS, or drop an arm, if the queue is
# tight.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k2_gate_gautschi.sbatch
#
# Re-run with the swept winners -- export in the SUBMITTING SHELL:
#   P5K2_CONSTANTS='16,64,4,1' P5K4_FWD_CONSTANTS='16,64,4,1' \
#     sbatch p5k2_gate_gautschi.sbatch
# NEVER `sbatch --export=ALL,P5K2_CONSTANTS='16,64,4,1' ...`: slurm splits that
# list on commas and the tuple arrives mangled (measured 2026-08-06).  sbatch
# exports the submitting environment by default, so the shell form above needs
# no --export at all; if an --export list is unavoidable, the script also
# accepts the semicolon form '16;64;4;1', which survives the split.
# Other optional overrides (same rule -- P5K2_CELLS is a comma list too):
#   P5K2_CELLS=512, P5K2_ITERATIONS=3, P5K2_SKIP_JAX=1,
#   P5K2_SKIP_FLOOR=1 (drops the repeat arm),
#   P5K4_SKIP_BACK_ONLY=1 (drops the shipped-configuration arm -- but that is
#   the arm the forward kernel's marginal win is measured against)
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p5k2g
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k2gate_%j.log

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

# MBIRTORCH_ENABLE_TRITON_FWD and MBIRTORCH_DISABLE_TRITON are set PER VARIANT
# by the script, explicitly on every torch arm, so neither is exported here --
# an inherited value would blur the arms, and with the back kernel now
# default-on an arm that merely fails to opt in is no longer a torch baseline.
# The script verifies each arm against the bodies the MODEL bound rather than
# trusting these variables.
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5k2_gate.py
echo "P5K2 GATE DONE"
