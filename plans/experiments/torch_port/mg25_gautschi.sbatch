#!/bin/bash
# mg25: THE COUNTER RUN ON THE CONE BACK KERNEL.  One H100.  The full
# statement is in mg25_back_counters.py; the essentials are repeated here
# because a reader of this file may never open that one.
#
# WHAT THIS RUN IS.  Open item B3 (in-kernel sorted or segmented accumulation)
# names one precondition in back_remedy_design.md section 6: a counter run on
# the back kernel itself, after the padding's confirmation runs.  This job is
# that run: it reads Nsight Compute counters on the Triton cone back kernel at
# five production band lengths and prints what they say.  It decides nothing
# and implements nothing -- the verdict is read by a person from the tables in
# this log, and acting on it needs Greg's approval.
#
# WHERE THIS SITS.  The band-padding remedy landed as mbirtorch 64dedb8, so
# every production band now launches at a multiple of 16 and therefore at the
# divisible rate; findings section 1.23 measured the 2.44x cliff closing, with
# the non-divisible bands falling from about 47,000 to about 20,400 ns per
# view-slice.  The band-length question is settled.  What is left of the cone
# back question is the efficiency of a SINGLE call at that rate.  mg20's
# counters measured the parallel FORWARD kernel's atomic write path, and this
# kernel has no atomics -- it gathers sinogram taps and stores once per output
# element -- so those counters describe a different kernel.
#
# THE THREE QUESTIONS, each answered by named counter columns:
#   1. Are the gathers transaction-bound at the divisible rate?  (memory
#      against SM speed-of-light throughput, the long-scoreboard stall ratio,
#      the L1 and L2 hit rates)
#   2. What is the gather path's sector efficiency?  (sectors per request, and
#      the L1 load sectors against one 32-byte sector per eight float32 taps)
#   3. What limits occupancy?  (the three occupancy limits, registers per
#      thread, achieved occupancy)
# Plus one zero-witness: the atomic and reduction sector counters, which the
# design note says are absent from this kernel by construction, measured here
# rather than asserted.
#
# TWO LEGS.  The timing leg needs no profiler and always runs; it calls the
# production route (Projectors.sparse_back_project_view_range) once per band
# and checks that each band ran at the divisible rate.  The counter leg uses
# Nsight Compute and is optional: many clusters keep GPU performance counters
# closed to unprivileged users, and the harness probes for that first and
# records a refusal as a finding.
#
# THE VARIANTS, five bands at cone sinogram (256, 2016, 1984) on one device,
# all of them already multiples of 16 so each launches at exactly its length:
#     1008   the 1024-class one-device band, and the 2048-class two-device band
#      672   the 2048-class three-device band
#      512   the padded form of 504 (2048-class four-device, 1024-class
#            two-device)
#      336   the 1024-class three-device band
#      256   the padded form of 252 (1024-class four-device)
#
# WALL ESTIMATE.
#   * THE TIMING LEG.  One torch import, one model build at the 2048-class
#     cell, then per band one warm call (which pays that band's Triton
#     compile) and three timed calls.  mg21b's calls at this cell are seconds
#     each.  About 10 minutes including the build and the per-band compiles.
#   * THE COUNTER LEG.  One discovery run, then up to six profiled variants at
#     up to three attempts each.  ncu replays each kernel several times and
#     saves and restores the memory it wrote between passes, and nobody has
#     timed a replay of THIS kernel, so the harness bounds the leg twice: 420 s
#     per attempt and 1800 s for the whole leg.  Whatever is left unprofiled is
#     recorded as unprofiled.  At most 30 minutes.
# That is at most about 40 minutes.  The walltime below is 01:30:00, more than
# twice the estimate, because the ncu replay cost of this kernel is an estimate
# rather than a measurement.
#
# DEVICE MEMORY.  About 45 GB at the peak, which is the 1008-band variant: the
# witness reference partial 12.4 GB, the current partial up to another 12.4 GB,
# the sinogram and the channel-major copy the body makes 8.2 GB together, and
# the per-chunk contract about 3 GB, plus the chunked float64 comparison.  Well
# inside one H100.  The profiled workers are much smaller: they subsample the
# pixel mask and launch one view batch.
#
# DISK.  A few megabytes in $MG25_RESULTS: one jsonl, and one ncu log per
# profiled attempt.  Nothing is staged.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every variant produced a timing
# row, every witness was at or below 1e-6, every rate was at or below 30,000 ns
# per view-slice, the model bound no torch body in either direction on CUDA,
# and the realized device is the one asked for.  The counter leg's absence
# never changes it, and neither does what either leg found.  Nothing is chained
# to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg25_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J torch_mg25
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg25_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu on
# PATH.  The counter leg needs it; the timing leg does not.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this one
# and leaves a torn .pth (the cluster notes' recorded failure).  Chain instead,
# with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# NSIGHT COMPUTE.  Its absence is a finding, not a failure: the timing leg is
# the leg that always runs, and the harness probes for ncu itself and records
# what it found.  This block only puts the answer at the top of the log.
#
# mg20's first run (job 15316533) recorded ncu absent: the batch environment's
# module load does not export Nsight Compute's directory.  The binary lives in
# the cuda module's own bin, so that directory is appended as a fallback;
# harmless when a later module does export it.
command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
if command -v ncu >/dev/null 2>&1; then
    echo "ncu found at $(command -v ncu)"
    ncu --version
else
    echo "NOTE: ncu is NOT on PATH after load_conda_cuda.sh.  The counter leg"
    echo "will record profiler_permitted as unavailable and the timing leg will"
    echo "run alone.  This does not fail the job."
fi

# The shipped configuration, set explicitly.  This is NOT a variable under
# test.  MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and the
# kernel it would turn off is the entire subject of this job.
export MBIRTORCH_DISABLE_TRITON=0
# THE ASSUMPTION EVERY VARIANT RESTS ON: the cone back projection really binds
# the Triton kernel on this device.  The availability gate runs a value
# self-check on the actual hardware, so a driver, a triton version or a kill
# switch that moved would silently hand every variant the compiled TORCH body
# instead.  That body has no gather counters to read, so the whole run would
# produce numbers that look like measurements and answer a different question.
# This check is on a tiny cell, which is cheap and reaches the same gate.
"$TORCHPY" -c "
import numpy as np, torch, mbirtorch
cell = (8, 24, 20)
angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4.0 * cell[2],
                                source_iso_dist=2.0 * cell[2])
model.skip_memory_preflight = True
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('cone forward body:', fwd.__name__)
print('cone back body:   ', back.__name__)
from mbirtorch.kernel_availability import cone_back_kernel_usable
print('back gate:', cone_back_kernel_usable(model))
assert back.__name__ == '_cone_back_view_batch_triton', back.__name__
" || { echo "the cone back projection does NOT bind the Triton kernel on this node; the whole run measures that kernel, so the job stops here"; exit 1; }

# This job reads no memory ledger, so it must not own the peak counters.
unset MBIRTORCH_MEMORY_CALIBRATION
# One device is what the job allocates; the harness configures cuda:0 and
# records the device it realized.  A value inherited from the shell would
# change the layout under it.
unset MBIRTORCH_NUM_DEVICES
# The process-wide column-gather override changes which driver runs.  This job
# calls one back-projection route and never enters a multi-device driver, but
# an inherited value would still change what the model builds around it.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG25_SMOKE
unset MG25_DRY
unset MG25_BANDS
unset MG25_NCU
unset MG25_NCU_PIXEL_DIV
unset MG25_REPEATS

export MG25_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG25_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The variant plan first, so the log opens with the five bands, their launch
# bands, the profiled-variant list, the metric sets and the gates, and a
# mis-set variable is caught before any GPU work.  The dry run imports no torch
# and touches no device.
export MG25_DRY=1
"$TORCHPY" -u mg25_back_counters.py || exit 1
unset MG25_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what they
# found; the findings are in the tables in this log.
"$TORCHPY" -u mg25_back_counters.py
rc=$?
echo "MG25 DONE rc=$rc"
exit $rc
