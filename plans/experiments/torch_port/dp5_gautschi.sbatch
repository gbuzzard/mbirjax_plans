#!/bin/bash
# dp5: does the Triton kernel set hold its values under sharding?  The flip
# gate read n=2/n=4 against n=1 at order-1 relative difference where the
# floor is ~1e-3, and a CPU bisection cleared the device-policy changes.
# CPU does not exercise the kernels, and compile_mode='off' does not disable
# them, so this varies the kernels alone at fixed device counts.  It reuses
# the sinograms dp4 cached, so run it AFTER dp4.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch dp5_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 03:00:00
#SBATCH -J torch_dp5
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/dp5_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4" || exit 1

# MBIRTORCH_DISABLE_TRITON is set PER ARM by the script, in each arm's own
# environment, because the availability probe caches its verdict per process.
# Nothing is exported here: an inherited value would blur the two arms.
# Every arm pins its device count explicitly, so the automatic choice plays
# no part in this probe.
unset MBIRTORCH_DISABLE_TRITON
export MBIRTORCH_NUM_DEVICES=1
export DP5_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$DP5_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u dp5_kernel_shard_probe.py
echo "DP4 DONE"
