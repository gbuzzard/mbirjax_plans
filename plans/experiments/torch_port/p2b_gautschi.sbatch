#!/bin/bash
# Phase 2 follow-up on gautschi: (1) the warm-vcd gate readout rerun (both
# frameworks record a warm second vcd run at every cell; mbirtorch now carries
# the scaled gather budget), and (2) the FIRST real gpu-torch harness run via
# torch_backend_writer.py (results land in torch_p2/results_writer/ for
# fetching into mbirjax_metrics/results/gpu-torch/).
# Submit: cd /scratch/gautschi/buzzard/torch_p2 && sbatch p2b_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:00:00
#SBATCH -J torch_p2b
#SBATCH -o /scratch/gautschi/buzzard/torch_p2/p2b_%j.log

set -x
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q -e /scratch/gautschi/buzzard/torch_p1/mbirtorch
"$TORCHPY" -c "import torch, mbirtorch; assert torch.cuda.is_available()" || exit 1

cd /scratch/gautschi/buzzard/torch_p2 || exit 1
export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export P0_TORCH_DEVICES=cuda
"$TORCHPY" -u p2_gate_readout.py

# The harness writer: REPO_ROOT resolution needs the script under a
# tooling/scaling_tests-like depth; run it from a staged copy with the results
# redirected via a local mbirjax_metrics-shaped tree.
mkdir -p writer_repo/tooling/scaling_tests writer_repo/results
cp torch_backend_writer.py writer_repo/tooling/scaling_tests/
MBIRTORCH_ROOT=/scratch/gautschi/buzzard/torch_p1/mbirtorch \
  "$TORCHPY" -u writer_repo/tooling/scaling_tests/torch_backend_writer.py
echo "P2B DONE"
