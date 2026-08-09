#!/bin/bash
# mg6: THE DIRECT-RECON BACK LOOP AT n>1 on 4 H100s -- charter B, step one.
# Four arms: the cone-512 n=1 ruler first, then parallel 1024 at n = 2 and 4,
# then cone 512 at n=2.  See mg6_backloop_probe.py and multigpu_findings.md
# §6.3.
#
# WHAT IT COSTS.  Each arm is one cold pass plus one instrumented warm pass at
# MG6_ITERATIONS=1, so the wall is dominated by the 1024 cold passes and the
# per-device Triton compiles (protocol 7).  Upper estimate one hour; the
# walltime below is twice that, per protocol 11.
#
# SCHEDULING (protocol 11).  A four-GPU block, so it must clear the 02:00 jax
# nightly and the 03:00 torch nightly.  It reads no tree that another job
# writes, so it needs no --dependency unless it shares a staging window with a
# job that syncs mbirtorch_src -- no sync happens into a tree a running job
# imports from.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg6_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg6
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg6_%j.log
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

# The charged view batch follows the BOUND body, and term 1 of the charter is
# only readable beside the realized batch, so this is the kernels-on cell and
# the kill switch is explicitly OFF rather than merely unset.  Protocol note:
# compile_mode='off' does NOT disable the Triton kernels -- selection is
# availability-driven, not compile-driven.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1: each arm pins through MBIRTORCH_NUM_DEVICES, which the SCRIPT
# sets per arm, and which keeps the model on the automatic branch where the
# ledger -- the thing this probe is auditing -- is actually built.  An
# inherited value here would make every arm single-count.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: mg2 is the ONE job that owns MBIRTORCH_MEMORY_CALIBRATION.  This
# probe owns torch.cuda.max_memory_allocated sub-phase by sub-phase, and two
# owners resetting the same counter would destroy both readings.  The arm
# ASSERTS this absent and fails loudly rather than measuring nonsense.
unset MBIRTORCH_MEMORY_CALIBRATION

export MG6_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG6_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg6_backloop_probe.py
echo "MG6 DONE"
