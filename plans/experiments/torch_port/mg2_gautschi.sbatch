#!/bin/bash
# mg2: THE LEDGER AT n>1 on 4 H100s -- modeled against measured per-device
# peaks at both gate cells, both geometries, n = 2 and 4, plus the unweighted
# pair at parallel 1024.  10 calibrated arms.  See mg2_ledger_calib.py and
# multigpu_plan.md §4 (mg2).
#
# §7 estimates 1-1.5 h; the walltime below is twice the upper estimate, per
# protocol 11.
#
# CHAINING (§4: "mg2 chains after mg1 on the same allocation class and the same
# staged tree"):
#     sbatch --dependency=afterok:<mg1 jobid> mg2_gautschi.sbatch
# Do NOT sync into mbirtorch_src between the two jobs: they share the tree, and
# no sync happens into a tree a running job imports from.
#
# SCHEDULING (protocol 11).  A four-GPU block, so it must clear the 02:00 jax
# nightly and the 03:00 torch nightly; the mg1 dependency usually does that on
# its own, but check before submitting a same-night pair.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg2_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:00:00
#SBATCH -J torch_mg2
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg2_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The ledger's view-batch charge follows the BOUND body, so this reading is the
# kernels-on cell and the kill switch is explicitly OFF rather than merely
# unset.  Protocol note: compile_mode='off' does NOT disable the Triton
# kernels -- selection is availability-driven, not compile-driven.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1: each arm pins through MBIRTORCH_NUM_DEVICES, which the SCRIPT
# sets per arm, and which keeps the model on the automatic branch where the
# preflight -- the ledger's only consumer -- still runs.  An inherited value
# here would make every arm single-count.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode OWNS torch.cuda.max_memory_allocated, so it is set by
# the SCRIPT for the measurement arms only; the generator must not reset the
# counter it is not reading.
unset MBIRTORCH_MEMORY_CALIBRATION

export MG2_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG2_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg2_ledger_calib.py
echo "MG2 DONE"
