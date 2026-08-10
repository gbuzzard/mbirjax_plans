#!/bin/bash
# dp4: the device-policy flip gate on 4 H100s.  Correctness only -- the
# automatic count, its identity with an explicit configure_devices call, the
# eager n=2/n=4 value floor against n=1, the process-wide pin, and the
# preflight's refusal of an impossible budget.  Item 3 owns the multi-GPU
# performance readout; nothing here is a timing gate.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch dp4_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 03:00:00
#SBATCH -J torch_dp4
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/dp4_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4" || exit 1

# The value arms are EAGER by construction (the script builds them with
# compile_mode='off'), per the compile-latitude policy.  The kernels' own
# availability is irrelevant to an eager arm, so nothing is forced here.
# MBIRTORCH_NUM_DEVICES must stay UNSET: the whole point is to observe the
# automatic choice.  The pinned_env arm sets it inside its own subprocess.
unset MBIRTORCH_NUM_DEVICES
export DP4_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$DP4_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u dp4_flip_gate.py
echo "DP4 DONE"
