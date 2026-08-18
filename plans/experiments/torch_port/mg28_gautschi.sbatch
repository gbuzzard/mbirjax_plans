#!/bin/bash
# mg28: THE COUNTER RUN ON THE PARALLEL FORWARD KERNEL.  One H100.  The full
# statement is in mg28_pfwd_counters.py; the essentials are repeated here.
#
# WHAT THIS RUN IS.  vcd parallel trails mbirjax at every device count and
# the records localize the gap to the parallel forward kernel, whose device
# span alone exceeds mbirjax's whole one-device reconstruction (open item
# B7).  mbirjax's forward moved to sorted, segmented accumulation and gained
# 1.4 to 2x; the torch kernel scatters with float atomic adds, and findings
# section 1.19 explicitly left open whether that atomic path sets the
# kernel's absolute rate.  This job reads the deciding counters with mg25's
# instrument: speed-of-light throughputs, the stall profile, and both memory
# paths priced -- the atomic write path AND the per-view reload of the
# values tile, which the kernel's own docstring names as the other candidate
# ("atomic-bound or load-bound... measure, then specialize").  It decides
# nothing and implements nothing.
#
# ONE RECORDED NUMBER IS RE-DERIVED.  Section 1.19's "192 times the ideal
# L2 sector traffic" was priced without the launch's view factor; the
# corrected formula divides by the 128-view batch, so the write path may be
# only 1.5x from coalesced.  This job's table prices both ways and the
# finding corrects the record if the re-derivation holds.
#
# THE VARIANTS, both through the SHIPPED wrapper at width 1008 on mg20's
# cell (1024, 1008, 992), view batch 128:
#     w1008_fullpix   the full 771,240-pixel mask (mg20's timed shape)
#     w1008_sub64     12,051 pixels, the most common VCD call size
#
# WALL ESTIMATE.  The timing leg is one model build plus four full-pixel
# launches of about 0.9 s each: under 5 minutes.  The counter leg is one
# discovery run plus two profiled variants at up to three attempts each,
# bounded at 420 s per attempt and 1200 s for the leg.  At most about 25
# minutes; the walltime below is 00:45:00.
#
# DEVICE MEMORY.  About 7 GB: the 3.1 GB values block, the 0.5 GB output
# view batch, and the contract tensors.  The profiled workers are smaller.
#
# DISK.  One jsonl and one ncu log per attempt, megabytes.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: the timing row exists and
# sits within 25 percent of mg20's 859 ms anchor, the model binds the
# Triton forward, and the device is the one asked for.  The counter leg's
# absence never changes it.  Nothing is chained to this job.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that installs
# into the same environment; chain with --dependency=afterany when one is
# queued.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg28_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_mg28
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg28_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu
# on PATH.  The counter leg needs it; the timing leg does not.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# NSIGHT COMPUTE.  Its absence is a finding, not a failure; this block only
# puts the answer at the top of the log.  The binary lives in the cuda
# module's own bin, appended as a fallback (mg20's discovery).
command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
if command -v ncu >/dev/null 2>&1; then
    echo "ncu found at $(command -v ncu)"
    ncu --version
else
    echo "NOTE: ncu is NOT on PATH.  The counter leg will record"
    echo "profiler_permitted as unavailable and the timing leg will run"
    echo "alone.  This does not fail the job."
fi

# The shipped configuration, set explicitly.  The kernel the kill switch
# would turn off is the entire subject of this job.
export MBIRTORCH_DISABLE_TRITON=0
# THE ASSUMPTION EVERY VARIANT RESTS ON: the parallel forward really binds
# the Triton kernel on this device (mg20's preflight, unchanged).
"$TORCHPY" -c "
import numpy as np, torch, mbirtorch
cell = (8, 24, 20)
model = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('parallel forward body:', fwd.__name__)
from mbirtorch.kernel_availability import parallel_forward_kernel_usable
print('forward gate:', parallel_forward_kernel_usable(model))
assert fwd.__name__ == '_parallel_forward_view_batch_triton', fwd.__name__
" || { echo "the parallel forward does NOT bind the Triton kernel on this node; the whole run measures that kernel, so the job stops here"; exit 1; }

# This job reads no memory ledger, so it must not own the peak counters.
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG28_SMOKE
unset MG28_DRY
unset MG28_NCU
unset MG28_REPEATS

export MG28_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG28_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The plan first, so a mis-set variable is caught before any GPU work.
export MG28_DRY=1
"$TORCHPY" -u mg28_pfwd_counters.py || exit 1
unset MG28_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the findings are in the tables in this log.
"$TORCHPY" -u mg28_pfwd_counters.py
rc=$?
echo "MG28 DONE rc=$rc"
exit $rc
