#!/bin/bash
# mg12: DOES A DEDICATED COPY STREAM CLOSE THE FORWARD PROJECTION'S TRANSFER
# STALL?  Three source trees, fifteen arms, four H100s, all measured in one job.
# The full statement of what is measured and why is in mg12_stream_gate.py; the
# essentials are repeated here because a reader of this file may never open that
# one.
#
# WHAT THIS JOB IS ABOUT.  On more than one GPU the forward projection gathers
# batches of voxel columns from the GPUs that own the slices onto the GPU that
# owns the views, and then projects them.  That shape -- the column gather -- is
# the library default on the tree this job starts from.  mg11 measured it and
# left one thing unexplained: on four H100s at the 1024 cell the forward
# funnel's per-device wall was 16.3 s while the projection kernels themselves
# were only busy for 12.4 s of it (pixel batch 4096; at batch 32768 the same
# pair reads 10.0 and 9.3).  The ~4 s difference is the device sitting idle
# while values move to it.  This job measures whether two candidate changes
# close it.
#
# THE THREE TREES, ALL BUILT IN THIS JOB ON THE COMPUTE NODE.  They are built
# here rather than staged from the login node because the login filesystem
# client has intermittently returned truncated reads for files whose bytes are
# correct on disk, and a tree that is subtly wrong is exactly the failure this
# job cannot survive: it would report NO CHANGE and look tidy doing it.
#   control    a clone of https://github.com/cabouman/mbirtorch.git checked out
#              at 3e9c183c192b8c6f320a2cacabcf5ba43fea6fbf.  Each batch is
#              gathered, then projected, then the next is gathered.  Nothing
#              overlaps by construction.
#   prefetch   a copy of control with prefetch_only.patch applied.  The driver
#              issues batch k+1's gather BEFORE projecting batch k, but every
#              copy still rides the device's default CUDA stream -- the same
#              stream the projection runs on.  THIS ARM IS A PREDICTION CHECK,
#              and the prediction is that it changes NOTHING: a stream runs its
#              work one item at a time and in order, so issuing a copy earlier
#              only moves its place in a queue that still has to run everything
#              in turn.  It exists so that the streams arm's result cannot be
#              credited to the prefetch ordering it also contains.
#   streams    a copy of control with five files replaced.  On top of the
#              prefetch ordering, each device gets a CUDA stream used for
#              nothing but these copies, and each batch carries its own event so
#              a projection waits for its own batch's copies and not for the
#              batch being gathered ahead of it.  THIS IS THE ARM EXPECTED TO
#              CLOSE THE STALL.
#
# WHAT MUST BE ON THE CLUSTER BEFORE THIS IS SUBMITTED.  The clone comes from
# GitHub, so only the overlays have to be shipped, and they go under
# $MG12_FILES in the SAME relative layout as the repository -- a flat directory
# would let a mis-copy land silently, and a layout that mirrors the repo cannot.
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/prefetch_only.patch
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/mbirtorch/_sharding.py
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/mbirtorch/tomography_model.py
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/mbirtorch/_memory_ledger.py
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/tests/test_memory_ledger.py
#     /scratch/gautschi/buzzard/torch_p3/mg12_files/tests/test_sharding.py
# and the harness itself:
#     /scratch/gautschi/buzzard/torch_p3/mg12_stream_gate.py
# The job checks every one of those is present before it spends any GPU time,
# echoes each file's sha256 as it copies it, and echoes the sha256 of what
# landed in the tree, so the log carries both ends of every copy.
#
# WHAT THE JOB BUILDS AND CHECKS, in order, all before the first arm runs.
#   1. The clone, with a clean retry: the tree is REMOVED between attempts, so a
#      half-written clone from a failed attempt cannot be finished by the next
#      one into something that is neither commit.
#   2. The checked-out hash is read back and compared against the pinned one.
#   3. .git is removed, so copying the tree twice costs the source and nothing
#      else.  The hash is written beside the tree first, so the log and the
#      directory both record what it is.
#   4. prefetch = copy + patch; streams = copy + five file copies.
#   5. compileall on all three trees AND on the harness.  This is the integrity
#      gate for the truncated-read failure above: a module an arm imports lazily
#      would otherwise fail forty minutes in instead of here.
#   6. A per-tree seam check that a live model of each geometry resolves to the
#      column gather with the switch UNSET, that the pixel-batch attribute is
#      the one the library reads, and that each tree's driver carries exactly the
#      shape it is supposed to.  A tree in the right place with the wrong
#      contents dies here rather than producing fifteen tidy rows.
#
# THE ARMS.  Five configurations, each on all three trees, 15 measured arms plus
# two untimed sinogram generators, all at mg9's, mg10's and mg11's cell
# (1024, 1008, 992), which gives 1008 slices -- divisible by 1, 2 and 4, so
# nothing is padded.
#     parallel  n=4  batch  4096      the cell the stall was measured at
#     parallel  n=4  batch  8192
#     parallel  n=2  batch  8192
#     cone      n=4  batch  8192
#     cone      n=2  batch  8192
# ARM ORDER is configuration-major: the three trees of one configuration run
# back-to-back.  A truncated job then yields COMPLETE comparisons rather than a
# control column and no treatment, and any drift in the node over the hours the
# job runs lands inside a comparison rather than between the columns.
#
# THE ONE GATE THAT CAN FAIL THIS INCREMENT IS MEMORY.  Both overlays already
# charge one more gathered cylinder in the memory ledger
# (COLUMN_GATHER_RESIDENTS goes from 2 to 3), because a device that gathers
# ahead really does hold the next batch while it projects this one.  The streams
# arm adds two further ways the MEASURED peak could outrun that charge: torch's
# caching allocator keeps its blocks per stream, so a copy stream is a second
# pool on the same device, and record_stream defers a block's reuse until the
# reading stream is done with it.  mg11 measured this ledger's margin at 0.4 to
# 1.0 percent, so there is very little room.  modeled/measured at or above 1.00
# passes; below 1.00 is printed as FLOOR VIOLATION.
#
# WALL ARITHMETIC, from mg11's MEASURED per-arm subprocess walls on this cell
# and this node class (row mg11_flip_gates_h001_20260811_041522.jsonl), less one
# composed reconstruction per arm -- mg11 ran 1 cold plus 3 warm passes and this
# job runs 1 cold plus 2.
#     2 generators                                     140 s
#     5 control arms (mg11's walls, measured)          691 s
#     5 prefetch arms (never run: low = control)       691 s
#     5 streams arms  (never run: low = control)       691 s
#     cold inductor cache allowance             600 to 1200 s
#     TOTAL                              about 47 to 69 minutes
# The overlay arms have never been run at all, so their high end assumes half
# again as much, which is mg10's and mg11's own rule for an unmeasured arm.  The
# cache allowance is real: this job gets its OWN inductor cache (see below), so
# the first arm at each launch shape compiles from cold where mg11 reused a warm
# one; that cost lands in each arm's discarded cold pass.  --dry-run prints this
# same arithmetic before any GPU time is spent.
#     The walltime below is 04:00:00, about 3.5x the upper estimate.  That much
# headroom is deliberate: ten of the fifteen arms have no prior reading at all,
# and if a copy stream makes things WORSE rather than better the wall grows in
# the direction the estimate cannot see.
#
# IF THIS JOB MUST BE SPLIT OR CUT, use MG12_CONFIGS, which takes the tokens the
# dry-run prints, and trim WHOLE CONFIGURATIONS in this order:
#     MG12_CONFIGS=p4b04096,p4b08192,p2b08192,c4b08192      drop c2b08192
#     MG12_CONFIGS=p4b04096,p4b08192,c4b08192               drop p2b08192 too
#     MG12_CONFIGS=p4b04096                                 the core alone
# Never trim a single tree out of a configuration: a configuration missing any
# of its three arms has no comparison block at all, and the control arm is
# where the value floor comes from.  MG12_ARMS drops single arms and exists for
# re-running one, not for trimming.
#
# THE COMPILER CACHE is this job's own, torch_cache_mg12, and NOT mg9's.  mg11
# could reuse mg9's because it ran mg9's tree; this job runs three trees, one of
# which changes _sharding.py, and sharing a cache across concurrent or
# successive jobs is a recorded way to get a corrupt artifact and a mystery
# crash.  The price is paid in the discarded cold passes and is in the estimate
# above.
#
# DISK.  Two shared sinograms at the 1024 cell, about 4 GB each, plus two
# strided reconstruction samples per arm at about 1.2 MB each, all in
# $MG12_RESULTS during the run and deleted at the end unless
# MG12_KEEP_ARTIFACTS=1 is exported.  The distances are computed before anything
# is removed.  The three trees are a few hundred MB together and are left in
# place, so a failed job can be examined and a re-run can skip the build.
#
# SCHEDULING.  This is a four-GPU block that can run to 04:00:00, so it cannot
# be submitted blind against the 02:00 and 03:00 nightlies: at the estimated 47
# to 69 minutes it would fit between them, but the WALLTIME does not, and slurm
# schedules on the walltime.  Submit it with an explicit start time that clears
# both, for example
#     sbatch --begin=04:15 mg12_gautschi.sbatch
# or any hour that leaves four clear hours ahead of it.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch --begin=<time> mg12_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 04:00:00
#SBATCH -J torch_mg12
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg12_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

P3=/scratch/gautschi/buzzard/torch_p3
MG12_FILES=$P3/mg12_files
MG12_TREES=$P3/mg12_trees
HARNESS=$P3/mg12_stream_gate.py
CONTROL_HASH=3e9c183c192b8c6f320a2cacabcf5ba43fea6fbf
REPO_URL=https://github.com/cabouman/mbirtorch.git

# This job's OWN inductor cache (see the header): three trees, one of which
# changes a library module, and a shared cache is a recorded way to get a
# corrupt artifact.
export TORCHINDUCTOR_CACHE_DIR=$P3/torch_cache_mg12
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1

# ── WHAT MUST ALREADY BE HERE ────────────────────────────────────────────────
test -f "$HARNESS" || { echo "the harness is not at $HARNESS"; exit 1; }
for f in prefetch_only.patch \
         mbirtorch/_sharding.py mbirtorch/tomography_model.py \
         mbirtorch/_memory_ledger.py \
         tests/test_memory_ledger.py tests/test_sharding.py; do
    test -f "$MG12_FILES/$f" || {
        echo "MISSING: $MG12_FILES/$f.  This job builds the control tree from"
        echo "GitHub but the two overlays are shipped, and they go under"
        echo "$MG12_FILES in the same relative layout as the repository."
        exit 1; }
done
echo "── the shipped overlay files, as they arrived ──"
sha256sum "$MG12_FILES/prefetch_only.patch" \
          "$MG12_FILES/mbirtorch/_sharding.py" \
          "$MG12_FILES/mbirtorch/tomography_model.py" \
          "$MG12_FILES/mbirtorch/_memory_ledger.py" \
          "$MG12_FILES/tests/test_memory_ledger.py" \
          "$MG12_FILES/tests/test_sharding.py" || exit 1

# ── TREE 1: THE CONTROL, cloned here with a clean retry ──────────────────────
# The tree is REMOVED between attempts.  A half-written clone left by a failed
# attempt must not be finished by the next one into something that is neither
# commit, and that is the whole reason the rm is inside the loop.
mkdir -p "$MG12_TREES"
CONTROL=$MG12_TREES/control
cloned=0
for attempt in 1 2 3; do
    rm -rf "$CONTROL"
    if git clone --quiet "$REPO_URL" "$CONTROL" \
       && git -C "$CONTROL" checkout --quiet --detach "$CONTROL_HASH"; then
        cloned=1
        break
    fi
    echo "clone attempt $attempt failed; removing the partial tree and retrying"
    sleep 20
done
test "$cloned" = 1 || { echo "could not clone $REPO_URL at $CONTROL_HASH in three attempts"; exit 1; }

# The hash is read back rather than trusted: a checkout that silently landed
# somewhere else would give every control arm the wrong baseline.
GOT_HASH=$(git -C "$CONTROL" rev-parse HEAD)
test "$GOT_HASH" = "$CONTROL_HASH" || {
    echo "the control tree is at $GOT_HASH and this job is pinned to $CONTROL_HASH"; exit 1; }
echo "$CONTROL_HASH" > "$MG12_TREES/control_hash.txt"
# .git goes now, so copying the tree twice below costs the source and nothing
# else.  The hash is already recorded in the log and in the file above.
rm -rf "$CONTROL/.git"
test -d "$CONTROL/mbirtorch" || { echo "no mbirtorch package in $CONTROL"; exit 1; }

# ── TREE 2: THE PREFETCH OVERLAY ─────────────────────────────────────────────
# patch(1) rather than git apply, so nothing in this job depends on the copied
# tree still being a git repository -- .git was removed above.
PREFETCH=$MG12_TREES/prefetch
rm -rf "$PREFETCH"
cp -a "$CONTROL" "$PREFETCH"
( cd "$PREFETCH" && patch -p1 < "$MG12_FILES/prefetch_only.patch" ) || {
    echo "prefetch_only.patch did not apply to the control tree at $CONTROL_HASH."
    echo "The patch was made against that commit; if the pin moved, remake it."
    exit 1; }
echo "── the prefetch tree, after the patch ──"
sha256sum "$PREFETCH/mbirtorch/tomography_model.py" \
          "$PREFETCH/mbirtorch/_memory_ledger.py" \
          "$PREFETCH/mbirtorch/_sharding.py" \
          "$PREFETCH/tests/test_memory_ledger.py" \
          "$PREFETCH/tests/test_sharding.py"

# ── TREE 3: THE COPY-STREAM OVERLAY ──────────────────────────────────────────
# Five files replaced outright.  Each is echoed at both ends -- the file that
# was shipped and the file that landed -- so the log shows the copy took.
STREAMS=$MG12_TREES/streams
rm -rf "$STREAMS"
cp -a "$CONTROL" "$STREAMS"
for f in mbirtorch/_sharding.py mbirtorch/tomography_model.py \
         mbirtorch/_memory_ledger.py \
         tests/test_memory_ledger.py tests/test_sharding.py; do
    cp -f "$MG12_FILES/$f" "$STREAMS/$f" || {
        echo "could not copy $f into the streams tree"; exit 1; }
done
echo "── the streams tree, after the five copies ──"
sha256sum "$STREAMS/mbirtorch/_sharding.py" \
          "$STREAMS/mbirtorch/tomography_model.py" \
          "$STREAMS/mbirtorch/_memory_ledger.py" \
          "$STREAMS/tests/test_memory_ledger.py" \
          "$STREAMS/tests/test_sharding.py"
# The copies must have CHANGED something.  A cp that silently did nothing would
# leave three control trees and the job would report NO CHANGE three times.
cmp -s "$CONTROL/mbirtorch/_sharding.py" "$STREAMS/mbirtorch/_sharding.py" && {
    echo "the streams tree's _sharding.py is identical to the control's; the copy did not take"; exit 1; }
cmp -s "$CONTROL/mbirtorch/tomography_model.py" "$PREFETCH/mbirtorch/tomography_model.py" && {
    echo "the prefetch tree's driver is identical to the control's; the patch did not take"; exit 1; }

export MG12_TREE_CONTROL=$CONTROL
export MG12_TREE_PREFETCH=$PREFETCH
export MG12_TREE_STREAMS=$STREAMS

# ── THE INTEGRITY GATE: every python file in all three trees must PARSE ──────
# This node's filesystem client has intermittently returned truncated reads for
# files whose bytes are correct on disk.  A module an arm imports lazily would
# otherwise fail mid-job instead of here.  The harness is checked first, because
# the seam check below imports it indirectly through the same interpreter.
"$TORCHPY" -m compileall -q "$HARNESS" || {
    echo "mg12_stream_gate.py does not parse on this node; re-copy it"; exit 1; }
for tree in "$CONTROL" "$PREFETCH" "$STREAMS"; do
    "$TORCHPY" -m compileall -q "$tree" || {
        echo "a file in $tree does not parse on this node; the build is not sound"; exit 1; }
done

# ── THE SEAM CHECK, one pass per tree, all before any GPU time ───────────────
# Each tree is asked, in a fresh process on its own PYTHONPATH, to prove three
# things about itself.  A tree in the right place with the wrong contents dies
# here rather than producing tidy rows.
#   THE SHAPE.  The tokens in the live driver source, and whether _sharding
#   carries the copy-stream entry points.  control has none of it, prefetch has
#   the ordering and not the streams, streams has both.
#   THE LEDGER.  COLUMN_GATHER_RESIDENTS: 2 on control, 3 on both overlays.  If
#   this were wrong the memory gate would price the wrong shape.
#   THE DEFAULT.  A live model of each geometry must resolve to the COLUMN
#   GATHER with the switch UNSET, and must read the pixel batch off the
#   documented attribute.  Every arm in this job runs the library's default with
#   the variable absent, so a tree whose default was the banded walk would give
#   fifteen arms of the wrong shape.
# The switch is removed from the environment first, because the check is about
# the default.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_WIDENING_GUARD
export MBIRTORCH_DISABLE_TRITON=0
for tree_name in control prefetch streams; do
    case "$tree_name" in
        control)  tree=$CONTROL;  want_ahead=0; want_async=0; want_res=2 ;;
        prefetch) tree=$PREFETCH; want_ahead=1; want_async=0; want_res=3 ;;
        streams)  tree=$STREAMS;  want_ahead=1; want_async=1; want_res=3 ;;
    esac
    MG12_SEAM_TREE=$tree MG12_SEAM_NAME=$tree_name \
    MG12_WANT_AHEAD=$want_ahead MG12_WANT_ASYNC=$want_async \
    MG12_WANT_RES=$want_res \
    PYTHONPATH=$tree "$TORCHPY" -c "
import inspect, os, sys
import numpy as np
import mbirtorch
from mbirtorch import _memory_ledger, _sharding
from mbirtorch.tomography_model import (TomographyModel, COLUMN_GATHER_ENV_VAR,
                                        FORWARD_PIXEL_BATCH)

name = os.environ['MG12_SEAM_NAME']
root = os.path.realpath(os.environ['MG12_SEAM_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(root + os.sep):
    sys.exit(name + ': mbirtorch imported from ' + got + ', NOT from ' + root +
             '.  An editable install of another tree is winning through a '
             'meta-path finder; check what else is running before fixing this '
             'with pip.')

# FACT 1: the switch is named what the harness removes from every child env.
if COLUMN_GATHER_ENV_VAR != 'MBIRTORCH_FORWARD_COLUMN_GATHER':
    sys.exit(name + ': the library switch is named %r; the harness removes '
             'MBIRTORCH_FORWARD_COLUMN_GATHER.' % (COLUMN_GATHER_ENV_VAR,))

# FACT 2: the names the harness reaches through are all present.
for attr in ('_column_gather_forward', '_forward_pixel_batch',
             '_sparse_forward_project_columns', '_build_memory_ledger',
             'configure_devices', '_view_batch_bodies'):
    if not hasattr(TomographyModel, attr):
        sys.exit(name + ': TomographyModel has no ' + attr +
                 '; the harness reads or drives it.')
for attr in ('gather_column_band', 'broadcast_band_to_views', 'move_shard',
             'run_per_device'):
    if not hasattr(_sharding, attr):
        sys.exit(name + ': _sharding has no ' + attr +
                 '; the harness wraps it as a witness.')

# FACT 3: THE SHAPE OF THIS TREE'S DRIVER.  'batch_bounds' is the list the
# prefetch ordering builds and the control loop does not mention;
# 'gather_column_band_async' is the copy-stream gather.
src = inspect.getsource(TomographyModel._sparse_forward_project_columns)
ahead = 'batch_bounds' in src
asyncg = 'gather_column_band_async' in src
if ahead != bool(int(os.environ['MG12_WANT_AHEAD'])):
    sys.exit(name + ': this driver %s gather one batch ahead, and the tree '
             'labeled %r %s.  It is not the tree it is labeled as.'
             % ('does' if ahead else 'does not', name,
                'is supposed to' if os.environ['MG12_WANT_AHEAD'] == '1'
                else 'is not supposed to'))
if asyncg != bool(int(os.environ['MG12_WANT_ASYNC'])):
    sys.exit(name + ': this driver %s use the copy-stream gather, and the tree '
             'labeled %r %s.  It is not the tree it is labeled as.'
             % ('does' if asyncg else 'does not', name,
                'is supposed to' if os.environ['MG12_WANT_ASYNC'] == '1'
                else 'is not supposed to'))
if asyncg:
    for attr in ('copy_stream', 'gather_column_band_async',
                 'open_copy_streams', 'close_copy_streams',
                 'wait_for_column_band'):
        if not hasattr(_sharding, attr):
            sys.exit(name + ': the driver calls the copy-stream gather but '
                     '_sharding has no ' + attr + '.')
elif hasattr(_sharding, 'copy_stream'):
    sys.exit(name + ': _sharding carries copy_stream on a tree whose driver '
             'does not use it; this is not the tree it is labeled as.')

# FACT 4: THE LEDGER CHARGE.  A device that gathers ahead holds one more
# cylinder, and the ledger has to charge it or the memory gate prices the wrong
# shape.
res = int(_memory_ledger.COLUMN_GATHER_RESIDENTS)
if res != int(os.environ['MG12_WANT_RES']):
    sys.exit(name + ': COLUMN_GATHER_RESIDENTS is %d and this tree ships %s.'
             % (res, os.environ['MG12_WANT_RES']))

# FACT 5: THE DEFAULT IS THE COLUMN GATHER, with the switch absent, on both
# geometries -- which is what every arm in this job runs.
if 'MBIRTORCH_FORWARD_COLUMN_GATHER' in os.environ:
    sys.exit(name + ': the switch is set in the environment of this check, so '
             'the check would not be about the default.')
cell = (8, 24, 20)
par = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
cone = mbirtorch.ConeBeamModel(cell, np.linspace(0, 2*np.pi, cell[0], endpoint=False),
                               source_detector_dist=80.0, source_iso_dist=40.0)
for geom, model in (('parallel', par), ('cone', cone)):
    if not model.column_gather_geometry:
        sys.exit(name + ': ' + geom + '.column_gather_geometry is False; this '
                 'tree cannot run the shape the job measures.')
    if not model._column_gather_forward():
        sys.exit(name + ': ' + geom + ' does NOT resolve to the column gather '
                 'with the switch UNSET.  Every arm in this job runs the '
                 'library default with the variable absent, so all fifteen '
                 'would measure the banded walk.')
    model.forward_project_pixel_batch = 4096
    if int(model._forward_pixel_batch()) != 4096:
        sys.exit(name + ': forward_project_pixel_batch is not the attribute '
                 'TomographyModel._forward_pixel_batch reads.')
print('%-9s seam check passed: gather-ahead=%s async-gather=%s residents=%d '
      'shipped default batch=%d, and both geometries default to the column '
      'gather' % (name, ahead, asyncg, res, int(FORWARD_PIXEL_BATCH)))
" || { echo "the $tree_name tree failed its seam check; do not run the job against it"; exit 1; }
done

# ── THE ENVIRONMENT, set explicitly, nothing inherited ───────────────────────
# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, because the measurement must be taken in the shipped
# environment and not in a variant of it; every arm re-asserts it.
export MBIRTORCH_DISABLE_TRITON=0
# THE SWITCH STAYS UNSET, here and in every child.  The column gather is the
# DEFAULT on all three trees, so an absent variable IS the shipped
# configuration.  The harness removes it from each child environment as well and
# every arm asserts it is absent; a stale '0' in a submitting shell would
# otherwise turn all fifteen arms into the banded walk without a word.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
# The device count is pinned on the MODEL with configure_devices, so a process
# wide count here would be a second, silent opinion.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, and the memory
# gate reads that counter after every timed reconstruction.  This job never sets
# it; it builds its own ledger instead, from the library's same entry point.
unset MBIRTORCH_MEMORY_CALIBRATION
# The speed floors are bypassed by a configure_devices pin anyway; unset so
# nothing inherited changes an automatic path no arm takes.
unset MBIRTORCH_WIDENING_GUARD
# PYTHONPATH is cleared HERE and set per arm by the harness.  A value inherited
# from a submitting shell could put another mbirtorch ahead of the tree an arm
# is supposed to import, which is the one failure that would look like a clean
# result.  Each arm proves where its mbirtorch came from anyway.
unset PYTHONPATH

# MG12_CONFIGS and MG12_ARMS are deliberately NOT unset: they are the knobs the
# header tells a reader to export from the SUBMITTING SHELL, and slurm's default
# --export=ALL is what carries them in.  Both are echoed so the log records what
# the job was asked for.
echo "MG12_CONFIGS=${MG12_CONFIGS:-<unset: all five configurations>}"
echo "MG12_ARMS=${MG12_ARMS:-<unset: all fifteen arms>}"
# The shaping knobs ARE unset, so nothing stale in a submitting shell can
# silently change what is measured or how long each arm takes.
unset MG12_ITERATIONS
unset MG12_WARM_REPEATS
unset MG12_MAX_EVENT_PAIRS
unset MG12_KEEP_ARTIFACTS
unset MG12_SMOKE
unset MG12_DEVICE
unset MG12_PIN

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the harness, which is the
# convention the rows are collected under.
export MG12_RESULTS=$P3
cd "$P3" || exit 1
# The arm plan first, so the log opens with the seventeen arms, the three tree
# roots, the cylinder each arm assembles, and the wall arithmetic -- and a
# mis-set environment variable is caught before any GPU work.
"$TORCHPY" -u mg12_stream_gate.py --dry-run || exit 1
"$TORCHPY" -u mg12_stream_gate.py
echo 'MG12 DONE'
