#!/bin/bash
# kb3: the composed five-arm gate re-run for kernel-aware view batching on 1
# H100 -- BOTH geometries at (512,448,384) and (1024,1008,992), the seeded
# 3-iteration warm VCD per arm, plus the shared-sinogram value arms (the
# p5ka protocol).  Arm checks pin each arm's bodies AND its realized view
# batches.  See kb3_gate.py; run AFTER the kb2 sweep's winners are pinned in
# the mbirtorch source, because this gates the shipped configuration.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch kb3_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 04:00:00
#SBATCH -J torch_kb3
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/kb3_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.is_available()" || exit 1

# MBIRTORCH_DISABLE_TRITON is set PER ARM by the script, explicitly on every
# torch arm; nothing is exported here (an inherited value would blur arms).
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u kb3_gate.py
echo "KB3 DONE"
