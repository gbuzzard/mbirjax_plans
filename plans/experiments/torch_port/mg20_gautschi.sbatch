#!/bin/bash
# mg20: WHY THE PARALLEL FORWARD KERNEL COSTS THE SAME AT VALUES WIDTH 504 AS
# AT WIDTH 1008.  One H100.  The full statement is in mg20_width_mechanism.py;
# the essentials are repeated here because a reader of this file may never open
# that one.
#
# WHERE THIS SITS.  mg10 measured the hand-written Triton parallel forward
# kernel against the width of the values block it is handed, at one device, and
# multigpu_findings.md section 1.9 records the result:
#
#     values-block width       per launch    per slice
#     1008 (the shipped call)     41.38 ms    0.0411 ms
#      504                        41.46 ms    0.0823 ms
#      252                        21.18 ms    0.0840 ms
#       63                         4.69 ms    0.0744 ms
#
# Below 504 the cost per launch falls roughly in proportion to the width, and
# from 504 to 1008 it does not fall at all.  A 504-wide launch therefore does
# half the work in the same time.  Section 1.9 named grid occupancy as the
# leading candidate and left the mechanism open.  This job is the probe.
#
# TWO QUESTIONS, AND THEY ARE INDEPENDENT.
#   1. Does the width regime point at how the launch fills the machine, or at
#      how it uses memory?
#   2. What do the write-path counters say about accumulating the detector
#      channels in sorted order, AT EVERY WIDTH?  The kernel scatters its
#      results with one float atomic add per channel tap.  That reading does
#      not depend on question 1, and an occupancy answer to question 1 does not
#      close it.
#
# TWO LEGS.  The timing leg needs no profiler and always runs.  The profiler
# leg uses Nsight Compute and is optional: many clusters keep GPU performance
# counters closed to unprivileged users, and this job probes for that first and
# records a refusal as a finding rather than as a failure.
#
# THE ARMS, eight of them, each in its own process, all at sinogram
# (1024, 1008, 992) on one device with the view batch pinned at 128:
#     w1008            1008                the shipped call
#     w504             504, 504            section 1.9's doubling point
#     w252             252 x 4
#     w63              63 x 16
#     w512_496         512, 496            both widths divisible by 16
#     w504_alloc1008   504 inside a 1008-wide allocation
#     w512_stride1008  512 columns of work at row stride 1008
#     w512_stride2016  512 columns of work at row stride 2016
# The first four are section 1.9's, and reproducing them is the validity check.
# The last two are the ablation the question turns on: they give the kernel
# 504-sized work over 1008-sized memory, so work and layout move separately for
# the first time.
#
# NO LIBRARY FILE IS TOUCHED.  The contiguous arms call the shipped wrapper.
# The strided arms cannot: the wrapper packs its values argument, and the
# kernel has no separate stride argument for the values block.  Those two arms
# copy the wrapper's launch code into the harness and change the LAUNCH only --
# they pass the wide block with num_cols set to its true row stride and
# truncate the grid's column axis to four blocks, which is 512 columns of work.
# The copy is checked at the full width against the library wrapper before its
# truncated launches are believed.
#
# WALL ESTIMATE.
#   * EIGHT TIMING ARMS.  Each pays a torch import, a model build at the 1024
#     cell, one Triton compile per new launch shape, a reference projection and
#     a values comparison, and then its launches.  The launches themselves are
#     seconds.  One full-pixel launch at width 1008 should cost about 0.9 s:
#     mg10 measured 28.14 s of forward busy time per reconstruction over a
#     launch mixture that adds up to 32 full-pixel launches.  That is an
#     inference from mg10's rows, and this job is the first to time such a
#     launch on its own.  An arm runs a four-point pixel ladder four times
#     over.  About 2 min each, so 16 min.
#   * THE PROFILER LEG.  One discovery run at about 2 min, then up to five
#     profiled arms.  ncu replays each kernel several times to collect its
#     counters and saves and restores the memory it wrote between passes, and
#     nobody has timed a replay of this kernel, so the harness bounds the leg
#     twice: 360 s per attempt and 1200 s for the whole leg.  Whatever is left
#     unprofiled is recorded as unprofiled.  At most 22 min.
# That is at most about 40 min, and typically nearer 25.  The walltime below is
# 01:30:00, more than twice the estimate, because the ncu replay cost of this
# kernel is an estimate rather than a measurement.
#
# DISK.  A few megabytes in $MG20_RESULTS: one jsonl and one ncu log per
# profiled attempt.  No arrays are staged -- each arm computes its own
# reference in its own process, which is cheaper than writing a half-gigabyte
# file and checksumming it.
#
# DEVICE MEMORY.  About 12 GB at the widest arm: a 3.1 GB values block, the
# per-piece copies, one 0.5 GB sinogram view batch as the reference, and for
# w512_stride2016 a 6.2 GB double-width buffer.  Well inside one H100.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every arm produced a row, every
# arm's values agreed with the full-width reference at 1e-5, and every section
# 1.9 arm reproduced its anchor within 15 percent.  The profiler leg's absence
# never changes it, and neither does what either leg found.  The mechanism
# verdict is read by a person from the two tables in this log.  Nothing is
# chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg20_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J torch_mg20
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg20_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu on
# PATH.  The profiler leg needs it; the timing leg does not.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# NSIGHT COMPUTE.  Its absence is a finding, not a failure: the timing leg is
# the leg that always runs, and the harness probes for ncu itself and records
# what it found.  This block only puts the answer at the top of the log.
#
# The first run (job 15316533) recorded ncu absent: the batch environment's
# module load does not export Nsight Compute's directory.  The binary lives in
# the cuda module's own bin, so that directory is appended as a fallback;
# harmless when a later module does export it.
command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
if command -v ncu >/dev/null 2>&1; then
    echo "ncu found at $(command -v ncu)"
    ncu --version
else
    echo "NOTE: ncu is NOT on PATH after load_conda_cuda.sh.  The profiler leg"
    echo "will record profiler_permitted as unavailable and the timing leg will"
    echo "run alone.  This does not fail the job."
fi

# The shipped configuration, set explicitly.  This is NOT a variable under
# test.  MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and the
# kernel it would turn off is the entire subject of this job.
export MBIRTORCH_DISABLE_TRITON=0
# THE ASSUMPTION EVERY ARM RESTS ON: the parallel forward really binds the
# Triton kernel on this device.  The availability gate runs a value self-check
# on the actual hardware, so a driver, a triton version or a kill switch that
# moved would silently hand every arm the compiled TORCH body instead.  That
# body has no width regime to explain, so all eight arms would produce numbers
# that look like measurements and answer a different question.  This check is
# on a tiny cell, which is cheap and reaches the same gate.
"$TORCHPY" -c "
import numpy as np, torch, mbirtorch
cell = (8, 24, 20)
model = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('parallel forward body:', fwd.__name__)
print('parallel back body:   ', back.__name__)
from mbirtorch.kernel_availability import parallel_forward_kernel_usable
print('forward gate:', parallel_forward_kernel_usable(model))
assert fwd.__name__ == '_parallel_forward_view_batch_triton', fwd.__name__
" || { echo "the parallel forward does NOT bind the Triton kernel on this node; every arm of mg20 measures that kernel, so the job stops here"; exit 1; }

# This job reads no memory ledger, so it must not own the peak counters.
unset MBIRTORCH_MEMORY_CALIBRATION
# One device is what the job allocates; the harness pins nothing and records
# the layout it settled on.  A value inherited from the shell would pin it.
unset MBIRTORCH_NUM_DEVICES
# The process-wide column-gather override changes which driver runs.  This job
# calls the projection BODY directly and never enters a driver, but an
# inherited value would still change what the model builds around it.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG20_SMOKE
unset MG20_ARMS
unset MG20_DRY
unset MG20_NCU
unset MG20_NCU_PIXEL_DIV
unset MG20_REPEATS

export MG20_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG20_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The arm plan first, so the log opens with the eight arms, their widths and
# their strides, and a mis-set variable is caught before any GPU work.  The dry
# run imports no torch and touches no device.
export MG20_DRY=1
"$TORCHPY" -u mg20_width_mechanism.py || exit 1
unset MG20_DRY

# The probe.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the findings are in the two tables in this log.
"$TORCHPY" -u mg20_width_mechanism.py
rc=$?
echo "MG20 DONE rc=$rc"
exit $rc
