#!/bin/bash
# mg29: THE VIEW-LOOP VARIANT OF THE PARALLEL FORWARD KERNEL -- THE SPIKE.
# One H100.  The full statement is in mg29_pfwd_viewloop.py.
#
# WHAT THIS RUN IS.  mg28 measured the parallel forward kernel load-bound on
# the per-view reload of its values block: DRAM delivers the 3.1 GB block
# about 130 times per 128-view launch while the arithmetic units run at 16
# percent (findings section 1.26).  The remedy the kernel's own docstring
# reserved for this measurement moves the view axis from the launch grid
# into an in-program loop, so each values tile is read once.  Greg ruled
# 2026-08-18: proceed (open item B7).  This spike writes that variant INSIDE
# the harness -- no library file is touched -- sweeps eight tile and warp
# configurations, gates each on values against the shipped wrapper at 1e-5,
# and times both on mg20's pixel ladder.  One Nsight Compute reading of the
# winner then confirms the mechanism: DRAM read over the values block
# should fall from 130x toward the few-times class.
#
# WHAT IT DOES NOT DECIDE.  The spike picks no default and ships nothing.
# A winning configuration goes to the library step, which re-gates end to
# end (the standing suites, the two-CUDA-device arms, and a composed A/B),
# because a kernel spike's win is not the driver's (the recorded lesson).
#
# WALL ESTIMATE.  Eight variant compiles at seconds each, a four-point
# ladder with four timed calls per (configuration, point) at up to 0.9 s a
# call, the shipped baseline on the same ladder, one model build, and one
# bounded ncu attempt: about 15 minutes.  The walltime below is 00:45:00.
#
# DEVICE MEMORY.  About 7 GB, as mg28: the 3.1 GB values block, the 0.5 GB
# output view batch per call, and the contract tensors.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: the shipped baseline within
# 25 percent of mg20's 859 ms anchor, every configuration's values within
# 1e-5 of the shipped wrapper, and no configuration raising.  The counter
# attempt never changes it.  Nothing is chained to this job.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that installs
# into the same environment; chain with --dependency=afterany when one is
# queued.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg29_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_mg29
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg29_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu
# on PATH for the counter attempt.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
command -v ncu >/dev/null 2>&1 && ncu --version | head -2

# The shipped configuration, and the assumption the baseline rests on: the
# parallel forward really binds the Triton kernel on this device (mg20's
# preflight, unchanged).  Every speedup this spike reports divides by that
# baseline.
export MBIRTORCH_DISABLE_TRITON=0
"$TORCHPY" -c "
import numpy as np, torch, mbirtorch
cell = (8, 24, 20)
model = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('parallel forward body:', fwd.__name__)
assert fwd.__name__ == '_parallel_forward_view_batch_triton', fwd.__name__
" || { echo "the parallel forward does NOT bind the Triton kernel on this node; the spike's baseline is that kernel, so the job stops here"; exit 1; }

unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG29_SMOKE
unset MG29_DRY
unset MG29_CONFIGS
unset MG29_NCU
unset MG29_REPEATS

export MG29_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG29_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG29_DRY=1
"$TORCHPY" -u mg29_pfwd_viewloop.py || exit 1
unset MG29_DRY

"$TORCHPY" -u mg29_pfwd_viewloop.py
rc=$?
echo "MG29 DONE rc=$rc"
exit $rc
