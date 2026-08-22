#!/bin/bash
# mg51: WHAT LIMITS THE MULTIAXIS GEOMETRY'S COMPILED PROJECTION BODIES.  One
# H100.  The full statement is in mg51_multiaxis_counters.py; the essentials
# are repeated here because a reader of this file may never open that one.
#
# WHY THIS RUN EXISTS.  MultiAxisParallelModel has no hand-written kernels.
# Both of its projection directions run bodies that torch.compile generated
# from ordinary python, and the model says so itself: its _view_batch_bodies
# returns _multiaxis_forward_view_batch and _multiaxis_back_view_batch with no
# kernel in sight.  Every other geometry in this tree that anyone has tried to
# make faster got a hand-written kernel first and a counter reading afterwards.
# This job takes the counter reading FIRST, so that whether multiaxis is worth
# a kernel at all is a question with measurements under it rather than an
# assumption.  It writes no kernel, changes no library file, and decides
# nothing; the verdict is read by a person from the tables in this log, and
# acting on it needs Greg's approval.
#
# THE FIVE THINGS IT NAMES, per direction and at two sizes:
#   1. How many DISTINCT generated kernels one projection call runs.  A
#      compiled body is not one kernel; inductor splits it into as many as its
#      fusion decides on, and a body of three kernels and a body of forty are
#      different engineering problems.
#   2. How many LAUNCHES one call issues in total.
#   3. WHERE THE DEVICE TIME CONCENTRATES: the top ten kernels by device time,
#      with their call counts and mean durations.  One dominant kernel is a
#      target; twenty kernels at five percent each is not.
#   4. HOST ENQUEUE TIME AGAINST DEVICE BUSY TIME.  If the host spends longer
#      issuing launches than the device spends running them, the body is
#      launch-bound and no kernel tuning moves it; the remedy would be fewer,
#      larger launches.  Read twice -- from the profiler's own host and device
#      totals, and from a profiler-free pair of clocks -- so neither reading
#      has to be taken on faith.
#   5. WHAT THE HARDWARE COUNTERS SAY about the kernels that matter: Nsight
#      Compute on the top three generated kernels in each direction.
#      Memory-bound, latency-bound and compute-bound are three different
#      verdicts with three different remedies, and the counter columns are
#      what separate them.
#
# THE TWO CELLS.  Multiaxis sinogram (512, 448, 384) and (1024, 1008, 992),
# both cells the multiaxis widening floors were measured at, so a wall here
# sits beside a recorded wall without adjustment.  One device, named
# explicitly, and the realized device is recorded.
#
# THE PROTOCOL IS THE WIDENING FLOORS REFRESH'S, deliberately: the same model
# construction (azimuths evenly spaced over half a turn, elevations swept
# across +/- 0.5 radians), the same shepp-logan low-dynamic-range phantom, the
# same seed 13 reset before every call.  The floors refresh times a whole
# reconstruction; this job times ONE projection call in each direction,
# through the public funnel the reconstruction itself calls
# (TomographyModel.sparse_forward_project and sparse_back_project).  The
# phantom is forward projected once to make the sinogram the back projection
# is given, which is the same array the floors refresh stages, so nothing is
# projected twice for bookkeeping.
#
# THREE LEGS.  The timing leg needs no profiler and always runs: one cold call
# per direction, then warm calls on a plain host clock with a device
# synchronize, plus the enqueue clock and a CUDA-event device span.  The
# profiler leg wraps warm calls in torch.profiler with CPU and CUDA activities
# -- three calls at the 512 cell, one at the 1024 cell, because one profile is
# enough to bound an expensive call.  The counter leg uses Nsight Compute at
# the 512 cell only and is optional: many clusters keep GPU performance
# counters closed to unprivileged users, and the harness probes for that first
# and records a refusal as a finding.
#
# THE KERNEL NAMES ARE DISCOVERED, NOT GUESSED.  Compiled kernel names are
# generated (triton_poi_fused_*, triton_red_fused_*, triton_per_fused_*) and
# nobody has written them down for this geometry.  The counter leg runs the
# script's own small runner mode once per direction with torch.profiler
# attached, takes the top generated names out of THAT process -- the process
# ncu will attach to -- and builds its --kernel-name filter from them.  A
# direction that produces no generated kernel is recorded as such and the leg
# falls back to the top device kernels of any kind rather than profiling
# nothing.
#
# WALL ESTIMATE.
#   * THE 512 CELL.  A model build, a phantom, two inductor compiles (one per
#     direction), then a cold and three warm calls each and three profiled
#     calls each.  The recorded warm wall for a 512-class multiaxis
#     reconstruction is about 11 s at one device, and that is a whole
#     three-iteration reconstruction; a single projection call is a fraction
#     of it, so this leg is dominated by the two compiles.  About 10 minutes.
#   * THE 1024 CELL.  A phantom the size of the volume built on the device and
#     gathered to the host, the host-side mask index, two more compiles at the
#     new shapes, then the same calls with one profiled call per direction.
#     The recorded warm wall for a 1024-class multiaxis reconstruction is
#     about 310 s at one device, again for a whole three-iteration
#     reconstruction.  About 25 minutes.
#   * THE COUNTER LEG.  Two discovery runs, then up to six kernels at up to
#     three attempts each.  ncu replays each profiled kernel several times and
#     saves and restores the memory it wrote between passes, and nobody has
#     timed a replay of THESE kernels, so the harness bounds the leg twice:
#     480 s per attempt and 3000 s for the whole leg.  Whatever is left
#     unprofiled is recorded as unprofiled, and the two legs that matter have
#     already finished by then.  Expect 20 to 30 minutes; the bound allows
#     rather more.
# So about an hour is the expectation and the bounded worst case is close to
# two.  The walltime below is 02:30:00, because the ncu replay cost of a
# compiled body's kernels is an estimate rather than a measurement and because
# nothing is chained to this job.
#
# DEVICE MEMORY.  The 1024 cell holds a 3.5 GB voxel array, a 4.1 GB sinogram
# and a similar output per call, so about 12 GB at the peak -- far inside one
# H100 on paper.  Paper is not a measurement, so every leg at that cell is
# wrapped: an out-of-memory is caught, recorded as that leg's result, and the
# run continues.  Host memory peaks around 8 GB while the phantom is indexed
# down to the mask.
#
# DISK.  A few megabytes in $MG51_RESULTS: one jsonl, and one ncu log per
# profiled attempt.  Nothing is staged, and no input is written to scratch.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every planned leg produced a
# row or a recorded out-of-memory, the realized device is the one asked for,
# both directions bound a torch body, both bodies really compiled rather than
# falling back to eager, and the tree witnesses hold.  The counter leg's
# absence never changes it, and neither does what any leg found.  Nothing is
# chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg51_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 02:30:00
#SBATCH -J torch_mg51
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg51_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu on
# PATH.  The counter leg needs it; the other two legs do not.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L

# ── the mbirtorch preflight ──────────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this one
# and leaves a torn .pth (the cluster notes' recorded failure).  Chain instead,
# with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# The tree witnesses.  The multiaxis bodies run no hand-written kernel and no
# padded wrapper, so neither of the first two touches what this job measures
# directly; they are what lets a reader tell from the log which tree produced
# these numbers.  The third one does matter here: the recompile remedy is what
# keeps a compiled body compiled across shapes, and a tree without it would
# hand this job eager python to measure.  The harness records all three on its
# header row as well.
"$TORCHPY" -c "
import inspect
from mbirtorch._utils import padded_kernel_width
from mbirtorch import projectors
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
print('tree witnesses ok')
" || exit 1

# NSIGHT COMPUTE.  Its absence is a finding, not a failure: the timing and
# profiler legs are the legs that always run, and the harness probes for ncu
# itself and records what it found.  This block only puts the answer at the top
# of the log.
#
# An earlier run in this series recorded ncu absent: the batch environment's
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
    echo "will record profiler_permitted as unavailable and the other two legs"
    echo "will run alone.  This does not fail the job."
fi

# The shipped configuration, set explicitly.  This is NOT a variable under
# test.  MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and this
# geometry has none, so it changes nothing here; it is set so that the
# environment is the one every other run in this series used.
export MBIRTORCH_DISABLE_TRITON=0

# THE ASSUMPTION EVERY MEASUREMENT BELOW RESTS ON: the multiaxis geometry
# really does run compiled TORCH bodies in both directions on this device.  If
# a hand-written kernel ever appears for this geometry, or if the geometry
# starts falling back to one, the tables in this log would describe a different
# thing under the same headings.  The check is on a tiny cell, which is cheap
# and reaches the same selection.
"$TORCHPY" -c "
import numpy as np, mbirtorch
from mbirtorch import _memory_ledger
cell = (8, 24, 20)
azimuth = np.linspace(0, np.pi, cell[0], endpoint=False)
elevation = np.linspace(-0.5, 0.5, cell[0])
model = mbirtorch.MultiAxisParallelModel(cell, np.stack([azimuth, elevation], axis=1))
model.skip_memory_preflight = True
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('multiaxis forward body:', fwd.__name__)
print('multiaxis back body:   ', back.__name__)
directions = tuple(_memory_ledger.torch_body_directions(model))
print('torch-body directions:', directions)
print('compile enabled:', model.compile_enabled)
assert directions == ('forward', 'back'), directions
assert model.compile_enabled, 'compilation is off, so there are no compiled bodies to measure'
" || { echo "multiaxis does NOT run compiled torch bodies in both directions on this node; that is the whole subject of this job, so it stops here"; exit 1; }

# This job reads no memory ledger, so it must not own the peak counters.
unset MBIRTORCH_MEMORY_CALIBRATION
# One device is what the job allocates; the harness configures cuda:0 and
# records the device it realized.  A value inherited from the shell would
# change the layout under it.
unset MBIRTORCH_NUM_DEVICES
# An explicit device list bypasses the speed floors anyway; unset so no
# inherited value can change what the model builds around it.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted, and this job
# measures compiled bodies across two cell shapes.
unset MBIRTORCH_RECOMPILE_LIMIT
# The process-wide column-gather override changes which driver runs.  This job
# calls two projection routes on one device and never enters a multi-device
# driver, but an inherited value would still change what the model builds.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
unset PYTHONPATH
unset MG51_SMOKE
unset MG51_DRY
unset MG51_CELLS
unset MG51_DIRECTIONS
unset MG51_NCU
unset MG51_WARM
# The runner variable is the internal mode the counter leg spawns.  A stray
# value in the submitting shell would turn the whole run into a single runner
# that prints one line and exits 0.
unset MG51_NCU_RUNNER

export MG51_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg51_multiaxis_counters
mkdir -p "$MG51_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg51_multiaxis_counters.py || { echo "mg51_multiaxis_counters.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with the two cells, their mirrored recon
# shapes, the leg settings, the metric sets and the bounds, and a mis-set
# variable is caught before any GPU work.  The dry plan imports no torch and
# touches no device.
export MG51_DRY=1
"$TORCHPY" -u mg51_multiaxis_counters.py || exit 1
unset MG51_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what they
# found; the kernel counts, the launch counts, the device-time concentration,
# the host-against-device split and the counter tables are all in this log.
"$TORCHPY" -u mg51_multiaxis_counters.py
rc=$?
echo "MG51 DONE rc=$rc"
exit $rc
