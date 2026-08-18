#!/bin/bash
# mg32: THE VIEW-CHUNK VARIANT OF THE PARALLEL FORWARD KERNEL -- THE
# SECOND SPIKE.
# One H100.  The full statement is in mg32_pfwd_segmented.py.
#
# WHAT THIS RUN IS.  mg29 measured the two ends of a trade and both lose:
# the shipped kernel re-reads the values block once per view (377 GB of
# DRAM reads), and mg29's whole-batch view loop collapsed the reads but
# thrashed the output planes out of L2 instead (262 GB of DRAM writes),
# for only 1.10x.  This spike sweeps the middle the two ends bracket: a
# view CHUNK per program (4 to 32, at the two best mg29 tiles), sized so
# the tile load amortizes while the active output planes stay cache
# resident.  Still inside the harness; no library file is touched.  The
# counter reading on the winner confirms both traffic sides at once.
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
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg32_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_mg32
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg32_%j.log
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
unset MG32_SMOKE
unset MG32_DRY
unset MG32_CONFIGS
unset MG32_NCU
unset MG32_REPEATS

export MG32_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG32_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG32_DRY=1
"$TORCHPY" -u mg32_pfwd_segmented.py || exit 1
unset MG32_DRY

"$TORCHPY" -u mg32_pfwd_segmented.py
rc=$?
echo "MG32 DONE rc=$rc"
exit $rc
