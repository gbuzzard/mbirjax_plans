#!/bin/bash
# mg1: THE GATE READOUT on 4 H100s -- the full n = 1, 2, 4 matrix, both
# frameworks, both geometries, both gate cells, warm seeded 3-iteration VCD,
# one subprocess per arm, plus the four auto arms and the twelve instrumented
# twins that carry the three-region attribution.  See mg1_readout.py and
# multigpu_plan.md §4 (mg1) under the eleven protocols of §3.
#
# 40 measured arms + 4 untimed generator arms.  §7 estimates 2.5-4 h; the
# walltime below is twice the upper estimate, per protocol 11.
#
# SCHEDULING (protocol 11).  This job is a four-GPU block, so it must clear the
# 02:00 jax nightly and the 03:00 torch nightly (itself up to four hours on a
# changed-tip night).  Submit outside that window, or add --begin.  mg2 chains
# after this job on the same allocation class and the same staged tree:
#     sbatch --dependency=afterok:<mg1 jobid> mg2_gautschi.sbatch
#
# STAGING (protocol 11).  Every file is scp'd PER FILE and md5-verified; no
# sync happens into a tree a running job imports from, so do not touch
# mbirtorch_src while any nightly or trial job is live.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg1_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 08:00:00
#SBATCH -J torch_mg1
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg1_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this campaign a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1
"$JAXPY" -c "import jax; assert len(jax.devices()) >= 4, jax.devices()" || exit 1

# The kernels are the shipped production configuration and this readout
# measures what a real multi-GPU user gets, so the kill switch is explicitly
# OFF rather than merely unset.  Protocol note: compile_mode='off' does NOT
# disable the Triton kernels -- selection is availability-driven, not
# compile-driven.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1: every pinned arm pins through MBIRTORCH_NUM_DEVICES, which the
# SCRIPT sets per arm.  An inherited value here would blur every arm, and would
# silently un-pin nothing while silently pinning the auto arms.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: the calibration mode owns torch.cuda.max_memory_allocated, and
# mg1's peaks are read straight off that counter.  mg2 is the only job that
# sets it.
unset MBIRTORCH_MEMORY_CALIBRATION

export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export MG1_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG1_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg1_readout.py
echo "MG1 DONE"
