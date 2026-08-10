#!/bin/bash
# mg10: THE TWO PROPOSED FORWARD REMEDIES, MEASURED ON OUR OWN KERNELS BEFORE
# EITHER ONE IS WRITTEN INTO THE LIBRARY, on four H100s.  The full statement of
# what is measured and why is in mg10_shape_sweep.py; the essentials are
# repeated here because a reader of this file may never open that one.
#
# WHY THIS JOB EXISTS.  mg9 settled the mechanism: the forward projection's
# per-device time is kernel-BUSY time -- 97 to 98 percent of the bracket at
# every device count -- so copying and waiting are not what keeps the forward
# from getting faster when GPUs are added.  A reading of mbirjax, the reference
# implementation, then supplied the SHAPE of the fix, and it is a different
# shape in each geometry.  Neither shape carries a number that is ours, and the
# remedy memo's section 8.6 says so in as many words: the reading transfers a
# shape, not a number, and three things have to be established on our own
# kernels.  This job establishes them.
#
#   EXPERIMENT 1, THE PARALLEL BAND KNEE.  Today each slice-owner's whole shard
#   is walked as ONE band: 504 slices wide at two devices, 252 at four.
#   mbirjax walks fixed 256-slice bands and its source says why -- "the
#   measured knee; whole-shard bands are WORSE at scale".  mg9 measured two
#   points on our kernel, 41.5 ms per launch at a 504-slice band and 21.4 ms at
#   252.  Those two points sit almost exactly on the break-even line, because
#   sub-banding multiplies the launch count by the number of sub-bands, so a
#   band half as wide has to cost LESS than half as much per launch to pay for
#   itself.  Whether it does below 252 is unmeasured.  This experiment sweeps
#   the sub-band length and reads per-launch time and per-device span against
#   it.  It changes nothing in mbirtorch: it sets the model attribute the
#   forward driver already reads, which is exactly the walk the proposed
#   library change would produce.
#
#   EXPERIMENT 2, THE CONE COLUMN GATHER.  Today cone walks slice bands like
#   parallel and adds one partial sinogram per band.  mbirjax does the
#   opposite: for each batch of pixel columns it gathers that batch's
#   FULL-HEIGHT cylinder from every slice-owner and makes ONE projector call
#   spanning every slice.  mbirjax measured that form at about 2x faster for
#   about 10 percent more transient memory.  This experiment builds that shape
#   as a harness prototype -- the same kernel bodies, the same transfer
#   primitive, the same return contract -- and reads time, memory AND VALUE
#   against the banded path we ship.  The value column exists because the
#   gather changes the order the vertical contributions are summed in, which is
#   the one gate the memo says cone's remedy must pass.
#
# WHAT THE LOG IS BUILT TO MAKE OBVIOUS, including a null result.  A narrower
# launch always has a smaller per-launch time, which on its own says nothing at
# all, so every table carries a NORMALIZED cost column: ms_per_slice for the
# band sweep and ms_per_kpix for the column-batch sweep.  mg9's own readings in
# that column are 0.0412 ms per slice at a 1008-slice band against 0.0823 at
# 504 and 0.0849 at 252, so the widest band was the most efficient per slice
# and a sweep in which NO length beats the shipped whole-shard walk is a real
# possible outcome of this job rather than a sign that something broke.  The
# log says so in those words when it happens.  The last block of the summary
# prints the design note's slot names verbatim with this run's values beside
# them, and prints NOT MEASURED IN THIS RUN for any slot whose arms did not
# run, so a narrowed or truncated job cannot leave a slot looking answered.
#
# WHAT EVERY ARM PROVES ABOUT ITSELF.  A patch that silently fails to engage
# produces a plausible row that means nothing, and a whole sweep can be read
# off such rows.  So every patched arm carries a counter checked against
# arithmetic derived from its own parameters, and every control arm proves the
# patch was NOT active; an arm that disagrees aborts loudly instead of
# reporting.  Experiment 1's witness is the band fan-out -- its call count and
# the width of every band it carried.  Experiment 2's witness is the
# prototype's own counters, of which the load-bearing one is the HEIGHT of
# every assembled cylinder: a single value equal to the full slice count is
# what separates a column gather from a band by another name.
#
# THE ARMS.  Eighteen measured arms plus two untimed sinogram generators, all
# at mg9's cell (1024, 1008, 992), which gives 1008 slices -- divisible by 1, 2
# and 4, so nothing is padded.
#     parallel, 2 devices: the shipped control (one 504-slice band), then
#                          sub-bands of 64, 128, 192, 256 and 384
#     cone,     1 device:  the value anchor (the shipped single-device path,
#                          which already makes the full-height calls the
#                          prototype imitates)
#     cone,     2 devices: the shipped banded control, then column batches of
#                          2048, 4096 and 8192
#     cone,     4 devices: the same four
#     parallel, 4 devices: the shipped control (252), then sub-bands of 128
#                          and 256
# Two of those requests land on a walk another arm already runs, and the plan
# print names both out loud: the driver caps a band request at the shard and
# its balanced tiling snaps to a divisor, so at two devices 256 and 384 both
# walk 2 x 252, and at four devices 256 is capped to 252, which IS the control.
# They are kept deliberately -- a repeated configuration is the only
# arm-to-arm noise floor this job has, and the sweep needs one to read small
# differences against -- but they are two runs of one measurement, not two
# points, and nobody should plot them as two.
#
# ARM ORDER is by decreasing value, because the rows are written incrementally
# and a truncated job should lose the least: the parallel two-device sweep
# first, then the cone value anchor, then the two-device column batches, then
# cone at four, then the parallel four-device arms.
#
# WALL ARITHMETIC.  mg9 measured its own per-arm subprocess walls on this cell
# and this node class, cold pass and torch import included: parallel 200 s at
# two devices and 145 s at four, cone 308 s at two, each generator 65 s.  A
# PATCHED arm's wall is unknown by construction -- that is what this job
# measures -- so the estimate is a range: the low end assumes a patched arm
# costs what its control costs, the high end assumes half again as much.
#     2 generators                              130 s
#     parallel n=2, 6 arms          1200 to    1700 s
#     cone n=1, 1 arm                310 to     330 s
#     cone n=2, 4 arms              1230 to    1710 s
#     cone n=4, 4 arms              1200 to    1670 s
#     parallel n=4, 3 arms           435 to     590 s
#     TOTAL                      75 min to  102 min
# The walltime below is 03:00:00, about 1.8x the upper estimate, which also
# covers a shared node.  --dry-run prints this same arithmetic before any GPU
# time is spent, so a mis-set knob is caught there.
#
# IF THIS JOB MUST BE SPLIT OR CUT, use MG10_ARMS, which takes the tokens the
# dry-run prints.  Trim in this order: the parallel four-device arms first,
# then the largest column batch.  The parallel two-device sweep and the cone
# two-device column batches are the CORE and must not be trimmed, and the cone
# one-device anchor is what every cone value row is measured against, so it
# goes wherever the cone arms go.
#     MG10_ARMS=p2_shard,p2_64,p2_128,p2_192,p2_256,p2_384
#     MG10_ARMS=c1,c2_banded,c2_2048,c2_4096,c2_8192
#     MG10_ARMS=c4_banded,c4_2048,c4_4096,c4_8192
#     MG10_ARMS=p4_shard,p4_128,p4_256
# A split has one cost worth naming: the value distances are computed across
# arms in ONE process, so a cone split that leaves the anchor in another job
# loses the vs_anchor column for those arms.  Keep c1 with whichever cone arms
# run.  Two further knobs extend rather than narrow the sweep, if the measured
# curve says the interesting region is outside the swept one:
#     MG10_PARALLEL_BANDS=32,48,64,96      MG10_CONE_BATCHES=8192,16384,32768
#
# DISK.  Two shared sinograms at the 1024 cell, about 4 GB each, plus two
# strided reconstruction samples per arm at about 1.2 MB each (the value
# columns), all in $MG10_RESULTS during the run and deleted at the end unless
# MG10_KEEP_ARTIFACTS=1 is exported.  The distances are computed before
# anything is removed.  mg9 deleted its own sinograms at the end of its job, so
# this one regenerates them; they are not shared between jobs.
#
# THE TREE THIS JOB IMPORTS, and the reason it is not installed.  mg10 runs out
# of /scratch/gautschi/buzzard/torch_p3/mbirtorch_mg9 -- the same export mg9
# ran on, already staged and verified, and NOT re-shipped (see SHIP.md).  Using
# mg9's tree is the point: mg10's control arms have to reproduce mg9's numbers,
# and they cannot do that against a different tree.
#   This file does NOT `pip install -e` that tree, which is what mg5 and mg8 do.
# An editable install rewrites a path pointer inside the SHARED environment at
# torch_p0/env, so installing here would repoint any other job's imports
# mid-flight.  mg10 puts the tree on PYTHONPATH instead, which changes nothing
# outside this job's own processes, and then ASSERTS that the mbirtorch it
# imported really came from that tree.  If that assert fails, an editable
# install of some other tree is winning through a meta-path finder; check what
# else is running before fixing it with pip.
#
# THE COMPILER CACHE is mg9's, deliberately reused:
# TORCHINDUCTOR_CACHE_DIR=.../torch_cache_mg9.  Same tree, same cell, same
# geometries, so the cache is warm for everything the control arms compile.
# The patched arms launch at new SHAPES, so they will compile some new
# artifacts; that cost lands in each arm's discarded cold pass, which is why
# the patch goes in before the cold pass and not after it.  Reusing the cache
# is only safe while no other job is writing it -- two concurrent jobs sharing
# one inductor cache is a recorded way to get a corrupt artifact and a mystery
# crash -- so check the queue before submitting.
#
# SCHEDULING.  This is a four-GPU block that can run to 03:00:00, so it must be
# submitted with the 02:00 and 03:00 nightlies in view; at the estimated 75 to
# 102 minutes it fits between them, but the walltime does not, so start it with
# enough clearance.  It shares the node with nothing it needs.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg10_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:00:00
#SBATCH -J torch_mg10
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg10_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
MG10_TREE=/scratch/gautschi/buzzard/torch_p3/mbirtorch_mg9
# mg9's cache, reused on purpose (see the header).
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache_mg9
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

# The import path, in place of an editable install (see the header).  The arm
# subprocesses inherit this, so every arm imports the same tree.
export PYTHONPATH=$MG10_TREE${PYTHONPATH:+:$PYTHONPATH}

nvidia-smi -L
test -d "$MG10_TREE/mbirtorch" || { echo "the tree is not staged at $MG10_TREE; mg9 staged and verified it -- do NOT re-ship it, check the path (see SHIP.md)"; exit 1; }
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# THE PRELUDE ASSERTS, in four parts, all before any GPU time is spent.
# Part 1 is mg8's and mg9's: the tree imports and four GPUs are visible.
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1
# Part 2 is mg9's provenance check: the mbirtorch that got imported must be the
# staged tree and not some other tree an editable install points at.
export MG10_TREE
"$TORCHPY" -c "
import os, sys, mbirtorch
tree = os.path.realpath(os.environ['MG10_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(tree + os.sep):
    sys.exit('mbirtorch imported from ' + got + ', NOT from the staged tree ' + tree +
             '.  An editable install of another tree is winning through a '
             'meta-path finder.  Check what else is running before fixing this '
             'with pip.')
print('mbirtorch from', got)
" || exit 1
# Part 3: the names this job reaches through the driver.  An older checkout
# would import cleanly and then fail every arm one subprocess at a time.
"$TORCHPY" -c "
from mbirtorch import _sharding
from mbirtorch.projectors import Projectors
from mbirtorch.tomography_model import TomographyModel
for name in ('broadcast_band_to_views', 'move_shard', 'run_per_device'):
    assert hasattr(_sharding, name), name
assert hasattr(Projectors, 'sparse_forward_project_view_range')
for name in ('_sparse_forward_project_sharded', '_banded_setup',
             '_slice_band_length', '_balanced_slice_bounds', '_band_pool'):
    assert hasattr(TomographyModel, name), name
print('the names mg10 wraps and calls are all present')
" || { echo "this mbirtorch does not carry the names mg10 needs; stage the f985a6e export into $MG10_TREE"; exit 1; }
# Every python file in the tree must parse, checked on THIS node before any GPU
# time.  The login node's filesystem client has intermittently returned
# truncated reads for files whose bytes are correct on disk; a module the arms
# import lazily would otherwise fail mid-job instead of here.  The harness
# itself is checked first, because part 4 below IMPORTS it and would otherwise
# report a truncated copy as a moved seam.
"$TORCHPY" -m compileall -q /scratch/gautschi/buzzard/torch_p3/mg10_shape_sweep.py || { echo "mg10_shape_sweep.py does not parse on this node; re-copy and re-verify it (SHIP.md step 2)"; exit 1; }
"$TORCHPY" -m compileall -q "$MG10_TREE" || { echo "a file in $MG10_TREE does not parse on this node; re-verify the tree against its manifest"; exit 1; }
# Part 4, and this one is mg10's own and matters most.  BOTH patch seams are
# checked against the driver's actual source and behavior, because each
# experiment rests on one line of it.  Experiment 1 rests on the forward driver
# reading `forward_project_slice_band`: if that getattr moved or was renamed,
# every sub-band arm would silently run the control and the sweep would read as
# a flat line.  Experiment 2 rests on the funnel resolving
# `_sparse_forward_project_sharded` on the INSTANCE, which is what lets the
# prototype be installed without editing the package.  The band tiling is also
# checked value by value against the harness's own re-derivation of it, since
# every witness in experiment 1 is compared against that re-derivation.
"$TORCHPY" -c "
import inspect, sys
sys.path.insert(0, '/scratch/gautschi/buzzard/torch_p3')
from mbirtorch.tomography_model import TomographyModel
import mg10_shape_sweep as mg10
src = inspect.getsource(TomographyModel._sparse_forward_project_sharded)
for token in ('forward_project_slice_band', 'broadcast_band_to_views',
              '_slice_band_length', '_balanced_slice_bounds',
              'sparse_forward_project_view_range'):
    if token not in src:
        sys.exit('the forward driver no longer mentions ' + token +
                 '; mg10 experiment 1 patches through that seam and would '
                 'silently measure the control instead.')
funnel = inspect.getsource(TomographyModel.sparse_forward_project)
if 'self._sparse_forward_project_sharded' not in funnel:
    sys.exit('the forward funnel no longer resolves '
             '_sparse_forward_project_sharded on the instance; mg10 '
             'experiment 2 installs its prototype there and would never run.')
for n_dev, requested in ((2, None), (2, 64), (2, 128), (2, 192), (2, 256),
                         (2, 384), (4, None), (4, 128), (4, 256)):
    shard = 1008 // n_dev
    band = TomographyModel._slice_band_length(shard, n_dev, 771240, requested)
    lib = [b - a for a, b in TomographyModel._balanced_slice_bounds(shard, band)]
    ours = mg10.realized_band_lengths(shard, requested)
    if lib != ours:
        sys.exit('the driver tiles a %d-slice shard at request %r as %r while '
                 'mg10 predicts %r; every experiment-1 witness compares against '
                 'the prediction, so the sweep would be mislabeled.'
                 % (shard, requested, lib, ours))
print('both mg10 patch seams are where mg10 expects them, and the band tiling '
      'matches the harness prediction at every swept length')
" || { echo "a mg10 patch seam moved in this tree; do not run the sweep against it"; exit 1; }
# Every python file in the tree must parse, checked on THIS node before any GPU
# time.  The login node's filesystem client has intermittently returned
# truncated reads for files whose bytes are correct on disk; a module the arms
# import lazily would otherwise fail mid-job instead of here.
"$TORCHPY" -m compileall -q "$MG10_TREE" || { echo "a file in $MG10_TREE does not parse on this node; re-verify the tree against its manifest"; exit 1; }
"$TORCHPY" -m compileall -q /scratch/gautschi/buzzard/torch_p3/mg10_shape_sweep.py || { echo "mg10_shape_sweep.py does not parse on this node; re-copy and re-verify it"; exit 1; }

# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, because the measurement must be taken in the shipped
# environment and not in a variant of it; every arm re-asserts it in its own
# environment.
export MBIRTORCH_DISABLE_TRITON=0
# Each arm pins its own device count, which the SCRIPT pops and then sets per
# subprocess.  A value inherited from here would pin the whole job to one count
# without saying so.  Each arm then asserts the device list it actually got,
# and a wrong count is FATAL in mg10 rather than a note: every shape number in
# an arm is derived from the shard length, so a mislabeled count would
# mislabel the whole row.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, and this job
# reads that counter after every timed reconstruction for its memory column.
# mg10 never sets it, here or anywhere.
unset MBIRTORCH_MEMORY_CALIBRATION
# The speed guard is bypassed by a device-count pin anyway; unset so nothing
# inherited changes the automatic path an arm never takes.
unset MBIRTORCH_WIDENING_GUARD
# The arm-narrowing, sweep-extending and artifact knobs, named here only so a
# re-run needs no edit to the file.  See the header for the trim order.
unset MG10_ARMS
unset MG10_PARALLEL_BANDS
unset MG10_PARALLEL_BANDS_N4
unset MG10_CONE_BATCHES
unset MG10_KEEP_ARTIFACTS
unset MG10_SMOKE

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the script, which is the
# convention the rows are collected under.
export MG10_RESULTS=/scratch/gautschi/buzzard/torch_p3
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the twenty arms, the walk each one
# actually performs, the two collapsed requests and the wall arithmetic -- and
# a mis-set environment variable is caught before any GPU work.
"$TORCHPY" -u mg10_shape_sweep.py --dry-run || exit 1
"$TORCHPY" -u mg10_shape_sweep.py
echo "MG10 DONE"
