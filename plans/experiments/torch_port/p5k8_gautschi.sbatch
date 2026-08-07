#!/bin/bash
# Job C: is the torch-vs-jax direct_recon divergence a ROUNDING-TIE class?
# One H100, NO TIMING -- direct_recon only at (1024,1008,992) with
# (512,448,384) as the control, then the divergence census, the half-integer
# tie fingerprint against a control set, a parameter-parity check, and the
# cylinder-coherence test the hypothesis itself demands (see p5k8_tie_probe.py).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k8_gautschi.sbatch
#
# Optional overrides -- export them in the SUBMITTING SHELL, e.g.
#   P5K8_CELLS=1024 sbatch p5k8_gautschi.sbatch
# and NEVER as `sbatch --export=ALL,P5K8_CELLS=512,1024 ...`: slurm splits that
# list on commas, so a multi-field value arrives mangled (measured 2026-08-06).
#   P5K8_CELLS=1024           the divergent cell only
#   P5K8_TOP=10               how many divergent voxels to fingerprint
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J torch_p5k8
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k8_%j.log

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

# The kernels are irrelevant to this question and the parallel pair may now be
# default-on: the kill switch keeps direct_recon on the torch bodies, so the
# measurement is of the LIBRARY chains and one less thing can move.
export MBIRTORCH_DISABLE_TRITON=1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
"$TORCHPY" -u ../p5k8_tie_probe.py
echo "P5K8 TIE PROBE DONE"
