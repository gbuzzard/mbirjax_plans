#!/bin/bash
# mg14: DOES FUSING THE COLUMN GATHER'S PER-BATCH ACCUMULATION INTO THE
# PROJECTOR MAKE THE FORWARD CHEAPER, AND DOES IT COST ANY MEMORY?  Two source
# trees, six measured arms, four H100s, all measured in one job.  The full
# statement of what is measured and why is in mg14_accum_gate.py; the essentials
# are repeated here because a reader of this file may never open that one.
#
# WHAT CHANGED.  On more than one GPU the forward projection gathers batches of
# voxel columns onto the GPU that owns the views, projects each batch, and sums
# the batches into that owner's sinogram block.  Until now each batch got its
# OWN full block: the projector allocated one, copied its view batches into it,
# and handed it back for the driver to add into the running total -- two
# full-block passes and one full-block allocation per batch.  The block is
# shard-sized, so that cost is the same whatever the pixel batch is; a bigger
# batch HIDES it rather than removing it, and a 1024-class pass runs 125 batches
# at the default and 249 at half of it.  Now the first batch's projection
# allocates the block and every later batch adds into that same block from
# inside the projector's own view loop, where the block was going to be written
# anyway: one pass, no per-batch allocation.
#
# WHAT IS CLAIMED, and what this job measures.
#   SPEED.  A few to fifteen percent of the forward, growing with the number of
#   batches; a CPU reading gave 13.3 percent at 64 batches and 4.4 percent at 16.
#   BUSY IS THE PRIMARY READING HERE, which is a change from mg12.  The work
#   that was removed is projector work: it ran on the compute stream between the
#   kernel launches, inside the per-body-call brackets that busy is the sum of.
#   The bracket is reported beside it because the forward funnel is what a user
#   waits for, but the effect should appear in busy first.  Both trees carry the
#   copy streams, so the copies are off the compute stream and the busy reading
#   is clean.
#   VALUE.  Bit-identical, by construction: per element the same summands are
#   added in the same order, and only where the addition happens changed.  The
#   distance to the control arm should sit at the control's own repeat floor or
#   under it.  The harness prints it either way and draws no gate from it.
#   MEMORY.  A view-owner no longer holds a separate incoming block beside the
#   one it is accumulating into, so the widest instant is one shard-sized block
#   narrower.  THE LEDGER STILL CHARGES TWO, on purpose: that charge is shared
#   with the banded path, which really does hold both, and over-charging is
#   legal where under-charging is not.  So the modeled peak is IDENTICAL on the
#   two trees -- verified before this job was written, and asserted again in
#   every comparison block -- and the reading of interest is whether the
#   MEASURED peak or the allocator's reserved bytes falls.
#
# THE TWO TREES, BOTH BUILT IN THIS JOB ON THE COMPUTE NODE.  They are built
# here rather than staged from the login node because the login filesystem
# client has intermittently returned truncated reads for files whose bytes are
# correct on disk, and a tree that is subtly wrong is exactly the failure this
# job cannot survive: it would report NO CHANGE and look tidy doing it.
#   control  a clone of the tip at eaccf55bc6b215d11e978802f4db0f6f8fdb2695.
#   fused    the same clone with four files replaced.
#
# WHAT MUST BE ON THE CLUSTER BEFORE THIS IS SUBMITTED.  The clone comes from
# GitHub, so only the overlay has to be shipped, and it goes under $MG14_FILES
# in the SAME relative layout as the repository -- a flat directory would let a
# mis-copy land silently, and a layout that mirrors the repo cannot.
#     /scratch/gautschi/buzzard/torch_p3/mg14_files/mbirtorch/projectors.py
#     /scratch/gautschi/buzzard/torch_p3/mg14_files/mbirtorch/tomography_model.py
#     /scratch/gautschi/buzzard/torch_p3/mg14_files/mbirtorch/_memory_ledger.py
#     /scratch/gautschi/buzzard/torch_p3/mg14_files/tests/test_sharding.py
# and the harness itself:
#     /scratch/gautschi/buzzard/torch_p3/mg14_accum_gate.py
# The job checks every one of those is present before it spends any GPU time,
# echoes each file's sha256 as it copies it, and echoes the sha256 of what
# landed in the tree, so the log carries both ends of every copy.
#
# WHAT THE JOB BUILDS AND CHECKS, in order, all before the first arm runs.
#   1. The clone, with a clean retry: the tree is REMOVED between attempts, so a
#      half-written clone from a failed attempt cannot be finished by the next
#      one into something that is neither commit.
#   2. The checked-out hash is read back and compared against the pinned one.
#   3. .git is removed, so copying the tree costs the source and nothing else.
#      The hash is written beside the tree first, so the log and the directory
#      both record what it is.
#   4. fused = copy + four file copies, with a cmp guard that the copy CHANGED
#      something -- a cp that silently did nothing would leave two control trees
#      and the job would report NO CHANGE at every configuration.
#   5. compileall on both trees AND on the harness.  This is the integrity gate
#      for the truncated-read failure above: a module an arm imports lazily
#      would otherwise fail twenty minutes in instead of here.
#   6. A per-tree seam check.  The fused tree's proves ``accumulate_into`` IS a
#      parameter of the live Projectors.sparse_forward_project_view_range and
#      the control's proves it is NOT, which is how mg12 proved its tree shapes.
#      Both also prove the shared ground -- the copy streams, the gather-ahead
#      ordering, the three-cylinder charge, and that both geometries default to
#      the column gather with the switch unset -- because a tree that quietly
#      lost any of it would still produce numbers and they would silently mean
#      something else.
#
# THE ARMS.  Three configurations, each on both trees, 6 measured arms plus two
# untimed sinogram generators, all at mg9's through mg12's cell
# (1024, 1008, 992), which gives 1008 slices -- divisible by 1, 2 and 4, so
# nothing is padded.
#     parallel  n=4  batch  4096      about 249 batches: the widest lever
#     parallel  n=4  batch  8192      about 125 batches: the library default
#     cone      n=4  batch  8192      about 125 batches
# The two parallel configurations differ only in the batch, so they differ by a
# factor of two in the batch COUNT, which is the quantity the saving is supposed
# to scale with; the log prints that pair side by side at the end.
# ARM ORDER is configuration-major: the two trees of one configuration run
# back-to-back.  A truncated job then yields COMPLETE comparisons rather than a
# control column and no treatment, and any drift in the node over the job lands
# inside a comparison rather than between the columns.
#
# WALL ARITHMETIC, from mg11's MEASURED per-arm subprocess walls on this cell
# and this node class (row mg11_flip_gates_h001_20260811_041522.jsonl), less one
# composed reconstruction per arm -- mg11 ran 1 cold plus 3 warm passes and this
# job runs 1 cold plus 2.
#     2 generators                                     140 s
#     3 control arms (mg11's walls, measured)          360 s
#     3 fused arms   (never run: low = control)        360 s
#     cold inductor cache allowance              400 to 800 s
#     TOTAL                              about 21 to 31 minutes
# The fused arms have never been run, so their high end assumes half again as
# much -- mg10's, mg11's and mg12's rule for an unmeasured arm -- and that is
# kept even though the change is expected to make them FASTER, because an
# estimate that assumes its own result is not an estimate.  --dry-run prints this
# same arithmetic before any GPU time is spent.
#     The walltime below is 02:00:00, about 4x the upper estimate.  mg12 took
# 04:00:00 for fifteen arms across three trees; six arms across two need half
# that, and a shorter walltime schedules sooner.
#
# IF THIS JOB MUST BE CUT, use MG14_CONFIGS, which takes the tokens the dry-run
# prints, and trim WHOLE CONFIGURATIONS in this order:
#     MG14_CONFIGS=p4b04096,p4b08192      drop cone
#     MG14_CONFIGS=p4b04096               the widest lever alone
# Never trim a single tree out of a configuration: a configuration missing
# either arm has no comparison block at all, and the control arm is where the
# value floor comes from.  MG14_ARMS drops single arms and exists for re-running
# one, not for trimming.
#
# THE COMPILER CACHE is this job's own, torch_cache_mg14.  Two trees whose
# projector differs, and sharing a cache across concurrent or successive jobs is
# a recorded way to get a corrupt artifact and a mystery crash.  The price is
# paid in the discarded cold passes and is in the estimate above.
#
# DISK.  Two shared sinograms at the 1024 cell, about 4 GB each, plus two
# strided reconstruction samples per arm at about 1.2 MB each, all in
# $MG14_RESULTS during the run and deleted at the end unless
# MG14_KEEP_ARTIFACTS=1 is exported.  The distances are computed before anything
# is removed.  The two trees are a few hundred MB together and are left in
# place, so a failed job can be examined and a re-run can skip the build.
#
# SCHEDULING.  This is a four-GPU block that can run to 02:00:00, so it cannot
# be submitted blind against the 02:00 and 03:00 nightlies: slurm schedules on
# the walltime, not on the estimate.  Submit it with an explicit start time that
# clears both, for example
#     sbatch --begin=04:15 mg14_gautschi.sbatch
# or any hour that leaves two clear hours ahead of it.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch --begin=<time> mg14_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg14
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg14_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

P3=/scratch/gautschi/buzzard/torch_p3
MG14_FILES=$P3/mg14_files
MG14_TREES=$P3/mg14_trees
HARNESS=$P3/mg14_accum_gate.py
CONTROL_HASH=eaccf55bc6b215d11e978802f4db0f6f8fdb2695
REPO_URL=https://github.com/cabouman/mbirtorch.git

# This job's OWN inductor cache (see the header): two trees whose projector
# differs, and a shared cache is a recorded way to get a corrupt artifact.
export TORCHINDUCTOR_CACHE_DIR=$P3/torch_cache_mg14
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1

# ── WHAT MUST ALREADY BE HERE ────────────────────────────────────────────────
test -f "$HARNESS" || { echo "the harness is not at $HARNESS"; exit 1; }
for f in mbirtorch/projectors.py mbirtorch/tomography_model.py \
         mbirtorch/_memory_ledger.py tests/test_sharding.py; do
    test -f "$MG14_FILES/$f" || {
        echo "MISSING: $MG14_FILES/$f.  This job builds the control tree from"
        echo "GitHub but the overlay is shipped, and it goes under"
        echo "$MG14_FILES in the same relative layout as the repository."
        exit 1; }
done
echo "── the shipped overlay files, as they arrived ──"
sha256sum "$MG14_FILES/mbirtorch/projectors.py" \
          "$MG14_FILES/mbirtorch/tomography_model.py" \
          "$MG14_FILES/mbirtorch/_memory_ledger.py" \
          "$MG14_FILES/tests/test_sharding.py" || exit 1

# ── TREE 1: THE CONTROL, cloned here with a clean retry ──────────────────────
# The tree is REMOVED between attempts.  A half-written clone left by a failed
# attempt must not be finished by the next one into something that is neither
# commit, and that is the whole reason the rm is inside the loop.
mkdir -p "$MG14_TREES"
CONTROL=$MG14_TREES/control
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
# somewhere else would give both arms of every configuration the wrong baseline.
GOT_HASH=$(git -C "$CONTROL" rev-parse HEAD)
test "$GOT_HASH" = "$CONTROL_HASH" || {
    echo "the control tree is at $GOT_HASH and this job is pinned to $CONTROL_HASH"; exit 1; }
echo "$CONTROL_HASH" > "$MG14_TREES/control_hash.txt"
# .git goes now, so copying the tree below costs the source and nothing else.
# The hash is already recorded in the log and in the file above.
rm -rf "$CONTROL/.git"
test -d "$CONTROL/mbirtorch" || { echo "no mbirtorch package in $CONTROL"; exit 1; }

# ── TREE 2: THE FUSED OVERLAY ────────────────────────────────────────────────
# Four files replaced outright.  Each is echoed at both ends -- the file that
# was shipped and the file that landed -- so the log shows the copy took.
FUSED=$MG14_TREES/fused
rm -rf "$FUSED"
cp -a "$CONTROL" "$FUSED"
for f in mbirtorch/projectors.py mbirtorch/tomography_model.py \
         mbirtorch/_memory_ledger.py tests/test_sharding.py; do
    cp -f "$MG14_FILES/$f" "$FUSED/$f" || {
        echo "could not copy $f into the fused tree"; exit 1; }
done
echo "── the fused tree, after the four copies ──"
sha256sum "$FUSED/mbirtorch/projectors.py" \
          "$FUSED/mbirtorch/tomography_model.py" \
          "$FUSED/mbirtorch/_memory_ledger.py" \
          "$FUSED/tests/test_sharding.py"
# The copies must have CHANGED something.  A cp that silently did nothing would
# leave two control trees and the job would report NO CHANGE at every
# configuration and look tidy doing it.  The projector and the driver are the
# two that carry the change; the ledger's difference is comment only, so it is
# not checked here.
for f in mbirtorch/projectors.py mbirtorch/tomography_model.py; do
    cmp -s "$CONTROL/$f" "$FUSED/$f" && {
        echo "the fused tree's $f is identical to the control's; the copy did not take"; exit 1; }
done

export MG14_TREE_CONTROL=$CONTROL
export MG14_TREE_FUSED=$FUSED

# ── THE INTEGRITY GATE: every python file in both trees must PARSE ───────────
# This node's filesystem client has intermittently returned truncated reads for
# files whose bytes are correct on disk.  A module an arm imports lazily would
# otherwise fail mid-job instead of here.  The harness is checked first, because
# the seam check below runs in the same interpreter.
"$TORCHPY" -m compileall -q "$HARNESS" || {
    echo "mg14_accum_gate.py does not parse on this node; re-copy it"; exit 1; }
for tree in "$CONTROL" "$FUSED"; do
    "$TORCHPY" -m compileall -q "$tree" || {
        echo "a file in $tree does not parse on this node; the build is not sound"; exit 1; }
done

# ── THE SEAM CHECK, one pass per tree, all before any GPU time ───────────────
# Each tree is asked, in a fresh process on its own PYTHONPATH, to prove what it
# is.  A tree in the right place with the wrong contents dies here rather than
# producing tidy rows.
#   THE DISCRIMINATOR.  ``accumulate_into`` is a parameter of the live
#   Projectors.sparse_forward_project_view_range on the fused tree and is absent
#   on the control, read with inspect.signature off the RUNNING function.  The
#   driver carries the matching pair: it either adds the returned block in
#   itself or hands the projector the block to add into.
#   THE SHARED GROUND.  Both trees must carry the copy streams, the
#   gather-ahead ordering and the three-cylinder charge, and both geometries
#   must default to the column gather with the switch UNSET.  None of that
#   separates the trees; all of it changes what the numbers MEAN.  With the
#   copies back on the compute stream the per-body-call brackets would start
#   covering transfer, and busy -- this job's primary reading -- would stop
#   meaning what it says.
# The switch is removed from the environment first, because the check is about
# the default.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_WIDENING_GUARD
export MBIRTORCH_DISABLE_TRITON=0
for tree_name in control fused; do
    case "$tree_name" in
        control) tree=$CONTROL; want_accum=0 ;;
        fused)   tree=$FUSED;   want_accum=1 ;;
    esac
    MG14_SEAM_TREE=$tree MG14_SEAM_NAME=$tree_name \
    MG14_WANT_ACCUM=$want_accum \
    PYTHONPATH=$tree "$TORCHPY" -c "
import inspect, os, sys
import numpy as np
import mbirtorch
from mbirtorch import _memory_ledger, _sharding
from mbirtorch.projectors import Projectors
from mbirtorch.tomography_model import (TomographyModel, COLUMN_GATHER_ENV_VAR,
                                        FORWARD_PIXEL_BATCH)

name = os.environ['MG14_SEAM_NAME']
root = os.path.realpath(os.environ['MG14_SEAM_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(root + os.sep):
    sys.exit(name + ': mbirtorch imported from ' + got + ', NOT from ' + root +
             '.  Either the working directory holds another mbirtorch, which '
             'python puts ahead of PYTHONPATH, or an editable install is '
             'winning through a meta-path finder.')

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
for attr in ('gather_column_band', 'gather_column_band_async',
             'broadcast_band_to_views', 'move_shard', 'run_per_device',
             'copy_stream', 'open_copy_streams', 'close_copy_streams',
             'wait_for_column_band'):
    if not hasattr(_sharding, attr):
        sys.exit(name + ': _sharding has no ' + attr + '.  The harness wraps '
                 'it as a witness, and the copy-stream names are what keep '
                 'the busy reading free of transfer time.')
if not hasattr(Projectors, 'sparse_forward_project_view_range'):
    sys.exit(name + ': Projectors has no sparse_forward_project_view_range; '
             'that is the loop this job measures.')

# FACT 3: THE DISCRIMINATOR, read off the running function rather than a file.
sig = inspect.signature(Projectors.sparse_forward_project_view_range)
takes = 'accumulate_into' in sig.parameters
want = bool(int(os.environ['MG14_WANT_ACCUM']))
if takes != want:
    sys.exit(name + ': the projector %s take accumulate_into, and the tree '
             'labeled %r %s.  It is not the tree it is labeled as.  Signature: '
             '%s' % ('does' if takes else 'does not', name,
                     'is supposed to' if want else 'is not supposed to',
                     sig))
src = inspect.getsource(TomographyModel._sparse_forward_project_columns)
passes = 'accumulate_into=owned' in src
adds = 'owned.add_(part)' in src
if passes != want or adds == want:
    sys.exit(name + ': the driver passes_accumulate_into=%s and '
             'adds_the_returned_block=%s, which does not match a %r tree.'
             % (passes, adds, name))

# FACT 4: THE SHARED GROUND.  The gather-ahead ordering and the copy-stream
# gather are in the driver of BOTH trees, and the ledger charges three
# cylinders on both.
if 'batch_bounds' not in src:
    sys.exit(name + ': the driver does not gather one batch ahead; both trees '
             'in this job are supposed to, and the bracket reading assumes it.')
if 'gather_column_band_async' not in src:
    sys.exit(name + ': the driver does not use the copy-stream gather.  With '
             'the copies back on the compute stream the per-body-call brackets '
             'would cover transfer, and busy is the primary reading of this '
             'job.')
res = int(_memory_ledger.COLUMN_GATHER_RESIDENTS)
if res != 3:
    sys.exit(name + ': COLUMN_GATHER_RESIDENTS is %d and both trees here ship '
             '3.' % res)

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
                 'library default with the variable absent, so all six would '
                 'measure the banded walk.')
    model.forward_project_pixel_batch = 4096
    if int(model._forward_pixel_batch()) != 4096:
        sys.exit(name + ': forward_project_pixel_batch is not the attribute '
                 'TomographyModel._forward_pixel_batch reads.')
print('%-7s seam check passed: accumulate_into=%s, driver passes it=%s / adds '
      'the returned block=%s, gather-ahead and copy streams present, '
      'residents=%d, shipped default batch=%d, both geometries default to the '
      'column gather' % (name, takes, passes, adds, res, int(FORWARD_PIXEL_BATCH)))
" || { echo "the $tree_name tree failed its seam check; do not run the job against it"; exit 1; }
done

# ── THE MODELED PEAK MUST BE THE SAME ON BOTH TREES ─────────────────────────
# The overlay's only change to the ledger is a comment, so the two trees must
# price an identical configuration identically.  If they do not, the memory
# ratios in the two columns are not comparable and the memory reading of this
# job is void -- so it is checked here, on a tiny cell, before any GPU time.
for tree_name in control fused; do
    case "$tree_name" in
        control) tree=$CONTROL ;;
        fused)   tree=$FUSED ;;
    esac
    MG14_LEDGER_OUT=$P3/_mg14_ledger_$tree_name.txt \
    PYTHONPATH=$tree "$TORCHPY" -c "
import os
import numpy as np, torch, mbirtorch
m = mbirtorch.ParallelBeamModel((64, 48, 40),
                                np.linspace(0, np.pi, 64, endpoint=False))
m.forward_project_pixel_batch = 64
devices = [torch.device('cpu')] * 4
led = m._build_memory_ledger(devices=devices, partition_sequence=None,
                             weights=None, init_recon=None, fm_hessian=None,
                             prox_input=None, init_error_sinogram=None)
open(os.environ['MG14_LEDGER_OUT'], 'w').write(repr(list(led.per_device_peaks())))
print('ledger', list(led.per_device_peaks()))
" || { echo "could not build a ledger on the $tree_name tree"; exit 1; }
done
cmp -s "$P3/_mg14_ledger_control.txt" "$P3/_mg14_ledger_fused.txt" || {
    echo "the two trees model DIFFERENT peaks for the same configuration."
    echo "The overlay's only ledger change is a comment, so this means the"
    echo "trees are not what they are labeled as, or the ledger really moved."
    echo "Either way the memory columns would not be comparable; do not run."
    exit 1; }
echo "both trees model the same peak for the same configuration, as intended"
rm -f "$P3/_mg14_ledger_control.txt" "$P3/_mg14_ledger_fused.txt"

# ── THE ENVIRONMENT, set explicitly, nothing inherited ───────────────────────
# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, because the measurement must be taken in the shipped
# environment and not in a variant of it; every arm re-asserts it.
export MBIRTORCH_DISABLE_TRITON=0
# THE SWITCH STAYS UNSET, here and in every child.  The column gather is the
# DEFAULT on both trees, so an absent variable IS the shipped configuration.
# The harness removes it from each child environment as well and every arm
# asserts it is absent; a stale '0' in a submitting shell would otherwise turn
# all six arms into the banded walk without a word.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
# The device count is pinned on the MODEL with configure_devices, so a process
# wide count here would be a second, silent opinion.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, and the memory
# reading uses that counter after every timed reconstruction.  This job never
# sets it; it builds its own ledger instead, from the library's same entry point.
unset MBIRTORCH_MEMORY_CALIBRATION
# The speed floors are bypassed by a configure_devices pin anyway; unset so
# nothing inherited changes an automatic path no arm takes.
unset MBIRTORCH_WIDENING_GUARD
# PYTHONPATH is cleared HERE and set per arm by the harness.  A value inherited
# from a submitting shell could put another mbirtorch ahead of the tree an arm
# is supposed to import.  Note that PYTHONPATH is not the whole story: python
# puts the running script's OWN directory ahead of it, so the harness also puts
# each arm's tree at the front of sys.path before importing, and proves where
# its mbirtorch came from either way.  Keep $P3 free of an mbirtorch directory.
unset PYTHONPATH

# MG14_CONFIGS and MG14_ARMS are deliberately NOT unset: they are the knobs the
# header tells a reader to export from the SUBMITTING SHELL, and slurm's default
# --export=ALL is what carries them in.  Both are echoed so the log records what
# the job was asked for.
echo "MG14_CONFIGS=${MG14_CONFIGS:-<unset: all three configurations>}"
echo "MG14_ARMS=${MG14_ARMS:-<unset: all six arms>}"
# The shaping knobs ARE unset, so nothing stale in a submitting shell can
# silently change what is measured or how long each arm takes.
unset MG14_ITERATIONS
unset MG14_WARM_REPEATS
unset MG14_MAX_EVENT_PAIRS
unset MG14_KEEP_ARTIFACTS
unset MG14_SMOKE
unset MG14_DEVICE
unset MG14_PIN

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the harness, which is the
# convention the rows are collected under.
export MG14_RESULTS=$P3
cd "$P3" || exit 1
# The arm plan first, so the log opens with the eight arms, the two tree roots,
# the batch count each configuration runs, and the wall arithmetic -- and a
# mis-set environment variable is caught before any GPU work.
"$TORCHPY" -u mg14_accum_gate.py --dry-run || exit 1
# The sentinel means DONE: it prints only when the harness exited cleanly.
# mg13's sentinel printed after a crash, and its watcher read a dead job as
# finished; this one cannot.
if "$TORCHPY" -u mg14_accum_gate.py; then
  echo 'MG14 DONE'
else
  echo "MG14 FAILED (harness exit $?)"
fi
