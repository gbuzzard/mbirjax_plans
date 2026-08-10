#!/bin/bash
# K4 part 1: the Triton cone FORWARD kernel's CONSTANT SWEEP on 1 H100 -- 30
# pruned configurations per cell at (512,448,384) and (1024,1008,992), each
# row its own process (see p5k4_fwd_sweep.py).  Isolated body bench: it ranks
# constants, it does not decide the gate (that is p5k2_gate_gautschi.sbatch).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k4_fwd_gautschi.sbatch
#
# Optional overrides -- export them in the SUBMITTING SHELL, e.g.
#   P5K4_CELLS=512 sbatch p5k4_fwd_gautschi.sbatch
# and NEVER as `sbatch --export=ALL,P5K4_CELLS=512,1024 ...`: slurm splits that
# list on commas, so a multi-field value arrives mangled (measured 2026-08-06
# on P5K2_CONSTANTS).  sbatch exports the submitting environment by default,
# which is why the shell form needs no --export at all.
#   P5K4_CELLS=512        one cell only (faster re-run)
#   P5K4_VIEWS=8          override the driver's view batch for the rows
#   P5K4_SKIP_JAX=1       skip the optional mbirjax forward reference
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:00:00
#SBATCH -J torch_p5k4
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k4_%j.log

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

# MBIRTORCH_ENABLE_TRITON_FWD is set PER ROW by the script (the forward is
# opt-in until this gate passes, and each row also arm-checks that production
# would bind the triton forward on this device).  The back kernel rides its
# own default-on, as production does.
export P0_TORCH_PYTHON=$TORCHPY
JAXPY=$(conda run -n mbirjax which python 2>/dev/null)
if [ -n "$JAXPY" ] && [ -x "$JAXPY" ]; then
  export P0_JAX_PYTHON=$JAXPY
else
  export P5K4_SKIP_JAX=1
fi
"$TORCHPY" -u ../p5k4_fwd_sweep.py
echo "P5K4 FWD SWEEP DONE"
