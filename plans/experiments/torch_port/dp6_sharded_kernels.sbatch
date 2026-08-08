#!/bin/bash
# dp6: the standing kernel-times-sharding gate, on 2 H100s.
#
# This combination had no gate before.  A single device never runs the banded
# multi-device drivers, so the composed n=1 gates cannot see it, and the
# multi-device value gates predate the Triton kernels.  The gap is what hid
# the forward-kernel defect that the interim selection rule now works around.
#
# The file it runs, tests/test_kernels_sharded.py, is also the acceptance bar
# for the repaired forward kernel: restoration means its arms join the torch
# arms at the multi-device float floor and the interim retires.
#
# Two devices are enough.  The isolation matrix found the defect equally at
# two and four, and a 2-GPU allocation schedules far sooner, so this gate is
# cheap enough to run on every kernel or engine change.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch dp6_sharded_kernels.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 02:00:00
#SBATCH -J torch_dp6
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/dp6_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2" || exit 1

# The kernels must be LIVE for this gate: it exists to check what a real
# multi-GPU user gets from the shipped selection.  The kill switch is
# explicitly off rather than merely unset, and the torch-body reference arms
# force their bodies in the test itself -- compile_mode='off' does NOT
# disable the kernels, because selection is availability-driven.
export MBIRTORCH_DISABLE_TRITON=0
# The gate configures its own device counts, so the process-wide pin must not
# be set; an inherited pin would silently make every arm single-device.
unset MBIRTORCH_NUM_DEVICES
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest tests/test_kernels_sharded.py -v 2>&1 | tail -40
echo "DP6 DONE"
