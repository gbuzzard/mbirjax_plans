#!/bin/bash
# mg53: WHERE THE HOST TIME OF ONE COMPILED PROJECTION CALL GOES.  One H100.
# The full statement is in mg53_host_cost_split.py; the essentials are repeated
# here because a reader of this file may never open that one.
#
# WHY THIS RUN EXISTS.  A separate measurement (mg51) timed one compiled
# multiaxis projection call on two clocks at once: the host time to return from
# the public call without synchronizing, and the profiler's account of what the
# host spent that time on.  At the (1024, 1008, 992) cell the two do not add
# up.  About 35 ms of host time PER VIEW BATCH is neither the kernel-launch API
# nor the final synchronize.  At the (512, 448, 384) cell the same unaccounted
# host time is about 0.35 ms per view batch.  A hundredfold difference in
# per-batch host cost is large enough that guessing at it is not worth doing,
# and nothing in the launch counts explains it.  This job attributes it.  It
# writes no kernel, changes no library file, and decides nothing; the verdict
# is read by a person from the tables in this log, and acting on it needs
# Greg's approval.
#
# THE FOUR CANDIDATE MECHANISMS, and the leg that separates each from the rest:
#   1. WORK INSIDE THE COMPILED CALLABLE.  A compiled body is not only its
#      kernels: calling it costs a dynamo cache lookup, a guard evaluation over
#      the arguments, and the inductor wrapper's own python before any launch
#      is issued.  That is charged inside the body call, so the split leg puts
#      it on one side of its split and the host ops leg names it.
#   2. ALLOCATOR CALLS THAT BLOCK.  cudaMalloc and cudaFree are synchronizing
#      driver calls, and an allocation that misses the caching allocator's
#      free-block pools triggers a retry that frees cached blocks first.  At
#      the 1024 cell the per-batch transients are large enough for this to be
#      plausible where it would not be at 512.  Read three ways: the
#      allocator's own counters across one call, the profiler's cudaMalloc and
#      cudaFree rows, and a re-timing under a different allocator setting.
#   3. A HIDDEN PER-BATCH SYNCHRONIZATION.  A device-to-host copy, an .item(),
#      or anything else that reads a device value on the host makes the host
#      wait once per view batch.  That is host time with no host work under it.
#      The synchronization leg turns on torch's own synchronization warnings,
#      so this candidate arrives named rather than inferred.
#   4. HOST WORK IN THE DRIVER LOOP.  The view-batch loop around the compiled
#      body is ordinary python: it slices the view parameters, allocates or
#      indexes the output block, and assigns or adds each batch's result.  That
#      is outside the body call, so the split leg puts it on the other side.
#
# THE TWO CELLS.  Multiaxis sinogram (512, 448, 384) and (1024, 1008, 992), the
# two cells mg51 measured, so the split here lands on numbers that already
# exist.  One device, named explicitly, and the realized device is recorded.
#
# THE PROTOCOL IS MG51'S, deliberately: the same model construction (azimuths
# evenly spaced over half a turn, elevations swept across +/- 0.5 radians), the
# same shepp-logan low-dynamic-range phantom, the same seed 13 reset before
# every call, and the same public funnel
# (TomographyModel.sparse_forward_project and sparse_back_project).  The
# phantom is forward projected once to make the sinogram the back projection is
# given, so nothing is projected twice for bookkeeping.
#
# FOUR LEGS, per cell and direction, each after ONE untimed warm-up call that
# pays the compile:
#   * TIMING reproduces mg51's timing semantics -- warm calls on a host clock
#     with a device synchronize, plus the enqueue clock, medians recorded -- so
#     this job's rows lay beside mg51's numbers without adjustment.
#   * BODY_SPLIT is the central measurement.  The projector object holds its
#     per-device bound bodies in two lists, and the single-device driver reads
#     index 0 of the relevant list at every public call and invokes it once per
#     view batch.  The leg wraps index 0 in a host clock, makes ONE call, and
#     puts the original back in a finally.  The enqueue time minus the summed
#     in-body time is the host time spent in the driver loop OUTSIDE the
#     compiled callable, which is candidate 1 against candidate 4 on one call.
#     The same call carries the allocator counters, which is candidate 2 read
#     where it would happen.
#   * HOST_OPS runs torch.profiler around one warm call and reports the top
#     events by self CPU time plus, by name, the runtime calls that settle
#     candidates 1, 2 and 3.
#   * SYNC_DETECTOR sets the synchronization debug mode to warn around one warm
#     call and records what the call trips, by name.
#
# THE ALLOCATOR ABLATION runs LAST and in its own process.  The caching
# allocator reads its segment policy from the environment when CUDA
# initializes and cannot be changed afterwards, so the harness spawns one
# subprocess of itself with expandable segments on.  The child runs the timing
# leg only, at the same cells and directions, and its rows are folded into the
# main file under a second leg name.  A child that fails is recorded as failed
# and changes nothing else, because the main legs are already on disk when it
# starts.
#
# WALL ESTIMATE.
#   * THE 512 CELL.  A model build, a phantom, two inductor compiles (one per
#     direction), then a warm-up call and six more calls in each direction
#     (three timed warm, one split, one profiled, one with synchronization
#     warnings on).  Dominated by the two compiles.  About 10 minutes.
#   * THE 1024 CELL.  A phantom the size of the volume built and gathered to
#     the host, the host-side mask index, two more compiles at the new shapes,
#     then the same seven calls per direction.  The recorded warm wall for a
#     1024-class multiaxis reconstruction is about 310 s at one device, and
#     that is a whole three-iteration reconstruction; one projection call is a
#     fraction of it.  About 25 minutes.
#   * THE ABLATION CHILD.  A second process that repeats the model builds, the
#     compiles and the timing leg at both cells, so roughly the setup cost of
#     the two above plus four short call sets.  About 20 minutes, bounded at
#     2400 s by the harness itself.
# So about an hour is the expectation.  The walltime below is 01:30:00, and
# nothing is chained to this job.  If the child is the part that runs out of
# time, it is recorded as unfinished and the main legs are already written.
#
# DEVICE MEMORY.  The 1024 cell holds a 3.5 GB voxel array, a 4.1 GB sinogram
# and a similar output per call, so about 12 GB at the peak -- far inside one
# H100 on paper.  Paper is not a measurement, so every leg is wrapped: an
# out-of-memory is caught, recorded as that leg's result, and the run
# continues.  Host memory peaks around 8 GB while the phantom is indexed down
# to the mask, and the ablation child holds its own copy of that peak while it
# runs, which is after the parent's cells have been released.
#
# DISK.  A few megabytes in $MG53_RESULTS: one jsonl from this run and one from
# the ablation child.  Nothing is staged, and no input is written to scratch.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every planned leg produced a
# row or a recorded out-of-memory, the realized device is the one asked for,
# both directions bound a torch body, both bodies really compiled rather than
# falling back to eager, the wrapped body was called and put back, and the tree
# witnesses hold.  The ablation arm's absence never changes it, and neither
# does what any leg found.  Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg53_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:30:00
#SBATCH -J mg53_host_cost_split
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg53_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
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
# hand this job eager python, which has no compiled dispatch to charge and so
# would answer a different question under the same headings.  The harness
# records all three on its header row as well.
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

# The shipped configuration, set explicitly.  This is NOT a variable under
# test.  MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and this
# geometry has none, so it changes nothing here; it is set so that the
# environment is the one every other run in this series used.
export MBIRTORCH_DISABLE_TRITON=0

# THE ASSUMPTION EVERY MEASUREMENT BELOW RESTS ON: the multiaxis geometry
# really does run compiled TORCH bodies in both directions on this device.  A
# hand-written kernel body carries its own view-batch cost attribute and would
# be batched by a different rule, and an eager body has no compiled dispatch to
# charge at all; either way the split in this log would describe something else
# under the same headings.  The check is on a tiny cell, which is cheap and
# reaches the same selection.
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
# The allocator setting is the ablation arm's own variable.  The parent process
# must run under the default so the two timing legs differ in exactly one
# thing; the harness puts the setting in the child's environment itself.
unset PYTORCH_CUDA_ALLOC_CONF
unset MG53_SMOKE
unset MG53_DRY
unset MG53_CELLS
unset MG53_DIRECTIONS
unset MG53_WARM
# The child variables are the ablation arm's internal mode.  A stray value in
# the submitting shell would make the whole run measure the timing leg alone
# and write its rows somewhere no table reads.
unset MG53_CHILD
unset MG53_CHILD_OUT

export MG53_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg53_host_cost_split
mkdir -p "$MG53_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg53_host_cost_split.py || { echo "mg53_host_cost_split.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with the two cells, their mirrored recon
# shapes, the legs and their settings, and the ablation arm's bounds, and a
# mis-set variable is caught before any GPU work.  The dry plan imports no
# torch and touches no device.
export MG53_DRY=1
"$TORCHPY" -u mg53_host_cost_split.py || exit 1
unset MG53_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what they
# found; the split between the compiled callable and the driver loop, the
# allocator counters, the host operator table, the synchronization warnings and
# the allocator ablation are all in this log.
"$TORCHPY" -u mg53_host_cost_split.py
rc=$?
echo "MG53 DONE rc=$rc"
exit $rc
