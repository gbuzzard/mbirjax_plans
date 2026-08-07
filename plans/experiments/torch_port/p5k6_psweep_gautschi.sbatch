#!/bin/bash
# The Triton PARALLEL pair's CONSTANT SWEEP on 1 H100 -- 30 pruned
# configurations per (cell, direction) at (512,448,384) and (1024,1008,992),
# both directions, each row its own process (see p5k6_psweep.py).  Isolated
# body bench: it ranks constants, it does not decide the gate (that is
# p5k6_pgate_gautschi.sbatch).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k6_psweep_gautschi.sbatch
#
# Optional overrides -- export them in the SUBMITTING SHELL, e.g.
#   P5K6_CELLS=512 P5K6_DIRECTIONS=back sbatch p5k6_psweep_gautschi.sbatch
# and NEVER as `sbatch --export=ALL,P5K6_CELLS=512,1024 ...`: slurm splits that
# list on commas, so a multi-field value arrives mangled (measured 2026-08-06).
# sbatch exports the submitting environment by default, so no --export is
# needed at all.
#   P5K6_CELLS=512            one cell only (faster re-run)
#   P5K6_DIRECTIONS=back      one direction only
#   P5K6_VIEWS=8              override the driver's view batch for the rows
#   P5K6_SKIP_JAX=1           skip the optional mbirjax reference rows
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:00:00
#SBATCH -J torch_p5k6s
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k6s_%j.log

set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
mkdir -p /scratch/gautschi/buzzard/torch_p3/run
cd /scratch/gautschi/buzzard/torch_p3/run || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

# The parallel kernels are OPT-IN: each row sets the switch for the direction
# it measures (and only that one, so the other direction's availability
# self-check never runs), then arm-checks that production would bind the triton
# body on this device.  Nothing is exported here.
export P0_TORCH_PYTHON=$TORCHPY
JAXPY=$(conda run -n mbirjax which python 2>/dev/null)
if [ -n "$JAXPY" ] && [ -x "$JAXPY" ]; then
  export P0_JAX_PYTHON=$JAXPY
else
  export P5K6_SKIP_JAX=1
fi
"$TORCHPY" -u ../p5k6_psweep.py
echo "P5K6 PSWEEP DONE"
