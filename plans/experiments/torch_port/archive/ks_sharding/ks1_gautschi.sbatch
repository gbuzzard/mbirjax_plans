#!/bin/bash
# ks1: where does a Triton forward launch go under the banded multi-device
# drivers, and which one change repairs its values?  Reading the launch path
# names a candidate mechanism -- triton launches on the launching THREAD's
# current CUDA device (a fresh worker thread's is 0), torch ops on the
# TENSOR's device -- and the arms discriminate launch-context, same-device
# async, view-lifetime, and banded-contract classes with one change each.
# Two GPUs suffice: the defect showed equally at two and four.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch ks1_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 01:30:00
#SBATCH -J torch_ks1
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/ks1_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2" || exit 1

# The toolchain record for the findings: the versions, and the launcher's
# device/stream source read from the LIVE objects rather than recalled --
# triton's driver binds get_current_device to torch.cuda.current_device
# (thread-current), which is the seam the probe arms test.
"$TORCHPY" -c "
import torch, triton
from triton.runtime import driver
print('torch', torch.__version__, '| triton', triton.__version__)
print('launcher device fn is torch.cuda.current_device:',
      driver.active.get_current_device is torch.cuda.current_device)
import torch.cuda
print('peer access 0<->1:', torch.cuda.can_device_access_peer(0, 1),
      torch.cuda.can_device_access_peer(1, 0))
"

# MBIRTORCH_DISABLE_TRITON is set PER ARM by the script, in each arm's own
# environment (the availability probe caches its verdict per process).
# Every arm binds its bodies and pins its device count explicitly, so
# neither the automatic layout choice nor the interim selection plays any
# part in this probe.
unset MBIRTORCH_DISABLE_TRITON
export MBIRTORCH_NUM_DEVICES=1
export KS1_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$KS1_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u ks1_launch_context.py
echo "KS1 DONE"
