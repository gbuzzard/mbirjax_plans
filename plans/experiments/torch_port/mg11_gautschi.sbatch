#!/bin/bash
# mg11: THE COMBINED FLIP GATES for the forward column gather, on cone AND
# parallel, run against the SHIPPED implementation on four H100s.  The full
# statement of what is measured and why is in mg11_flip_gates.py; the essentials
# are repeated here because a reader of this file may never open that one.
#
# WHY THIS JOB EXISTS, and what makes it different from every job before it.
# The column gather is written, tested and committed.  It is off by default.
# This job is what decides whether the default flips.  mg10 measured a
# PROTOTYPE -- it built the shape on the live model object because the shape did
# not exist in the library yet -- and a prototype's numbers are the prototype's.
# mg11 patches NOTHING.  Every arm drives the library exactly as a user would,
# through one environment variable and one documented model attribute:
#     MBIRTORCH_FORWARD_COLUMN_GATHER=1   the switch, read per call
#     model.forward_project_pixel_batch = B   the documented batch override
# There is no install_* function in mg11_flip_gates.py and there must never be
# one.  The moment an arm patches the driver, the thing being gated stops being
# the thing that will ship.
#
# THE RULINGS THIS EXECUTES.  forward_remedy_design.md section 13 (RULED, Greg,
# 2026-08-10, late): the parallel extension is approved, the two geometries'
# gates run as ONE combined campaign, and the defaults flip when the gates pass,
# without a further ruling per geometry.  Section 7.2 carries the value gates
# and the bar: the SHIPPED PARITY FLOOR governs, and mg10's measured e-6 class
# is a LOGGED EXPECTATION that is never a failing assertion.
#
# THE THREE GATE COLUMNS, one per arm, and the verdict the log ends with.
#   TIMING.  mg9's and mg10's instrument unchanged: a bracket around the forward
#   funnel, a busy probe around every forward body call keyed by device
#   position, the per-launch time, and the composed reconstruction wall.  The
#   speed verdict is the best flag-on arm against the flag-off control at the
#   same (geometry, device count), both measured IN THIS JOB.
#   VALUE.  Per-recon checksums and strided samples; each arm's distances to the
#   same-count control and to its geometry's one-device anchor, each read
#   against that arm's OWN repeat floor (both forward kernels accumulate with
#   float atomics and are not bit-reproducible, so the floor is not zero).  The
#   FAILING bar is the shipped parity floor, 5e-3 relative, which the prelude
#   below re-reads out of the test files rather than trusting.
#   MEMORY.  Per-device measured peak AND the ledger's own modeled peak for that
#   arm's exact configuration, from the library's own planning entry point
#   (TomographyModel._build_memory_ledger, called with the same call arrays
#   recon hands _apply_device_policy).  modeled/measured at or above 1.00
#   passes; below 1.00 is a FLOOR VIOLATION and fails that geometry's gate.
#   This is the calibration confirmation the implementation deferred to
#   hardware, and it is the column with no prior reading at all.
#
# The log ends with one block per geometry carrying those three readings and one
# of "GATES PASS - FLIP AUTHORIZED: <geometry>" or "GATES FAIL: <geometry>:
# <which gate>".  Nothing in the harness draws a conclusion beyond that line.
#
# WHAT EVERY ARM PROVES ABOUT ITSELF, and why it is harder here.  A patched arm
# can count its own patch; a patchless arm cannot.  So every witness is built
# out of something the LIBRARY does, observed from outside, and the four are
# recorded by four different pieces of code:
#     the RESOLVER      model._column_gather_forward(), the library's own answer
#     the LEDGER PLAN   plan.column_pixel_batch -- the batch on the gather path
#                       and None on the banded one, written by a different module
#     the PRIMITIVES    _sharding.gather_column_band is called only by the
#                       column gather and _sharding.broadcast_band_to_views only
#                       by the banded walk, so counting them separates the two
#                       shapes at the transfer layer.  This is the same witness
#                       the library's own test uses (test_sharding.py::
#                       test_the_banded_walk_is_what_runs_with_the_switch_off)
#     the KERNEL'S VIEW the shape of every values block a forward body was
#                       handed: (batch) x (the WHOLE slice axis) on the gather
#                       path, (all the pass's pixels) x (one shard) on the
#                       banded one.  A single column count equal to the full
#                       slice count is what separates a column gather from a
#                       band by another name.
# The last two are cross-checked against EACH OTHER rather than each against an
# expectation, so a witness that agreed with itself cannot pass.  Controls carry
# the mirror image of all four.  An arm that disagrees with any of them ABORTS.
# Twelve negative tests were run on this Mac before shipping, each breaking one
# witness and requiring the arm to abort; all twelve aborted.
#
# THE ARMS.  Twenty measured arms plus two untimed sinogram generators, all at
# mg9's and mg10's cell (1024, 1008, 992), which gives 1008 slices -- divisible
# by 1, 2 and 4, so nothing is padded.
#     cone,     1 device:  c1, the value anchor (the shipped single-device path)
#     cone,     2 devices: c2_off, then flag on at 8192, 16384, 32768
#     parallel, 1 device:  p1, the value anchor
#     parallel, 2 devices: p2_off, then flag on at 4096, 8192, 16384, 32768
#     cone,     4 devices: c4_off, then the three cone batches
#     parallel, 4 devices: p4_off, then the four parallel batches
# Cone's sweep starts at the library's own default (FORWARD_PIXEL_BATCH = 8192,
# which mg10 measured as the largest value tried and still falling).  Parallel's
# reaches lower as well as higher, and that is section 13's caveat made
# concrete: the two-times prediction for parallel was measured with the PIXEL
# COUNT HELD FULL, and the gather cuts pixels to the batch, so a sweep starting
# at 8192 could not tell a batch effect from a shape effect.  The prediction is
# judged against the SWEPT BEST.  One gathered cylinder holds batch x 1008 x
# 4 B, which is 132.1 MB at batch 32768, so the whole sweep stays under the
# 150 MB bound the ruling names; the dry-run prints that arithmetic per arm.
#
# ARM ORDER is by decreasing value, because rows are written incrementally and a
# truncated job should lose the least: each geometry's anchor comes before the
# arms measured against it, and both two-device blocks -- where the cone
# measurement and the parallel prediction both live -- come before the
# four-device ones.
#
# WALL ARITHMETIC, from mg10's MEASURED per-arm subprocess walls on this cell
# and this node class (rows mg10_shape_sweep_h004_20260810_174925.jsonl and
# _h008_20260810_201612.jsonl), torch import and cold pass included.
#     2 generators                                136 s
#     cone     n=1 anchor              281 to    295 s
#     cone     n=2, 1 off + 3 on      1091 to   1225 s   (on: mg10 measured 264)
#     parallel n=1 anchor              187 to    196 s
#     parallel n=2, 1 off + 4 on       955 to   1347 s
#     cone     n=4, 1 off + 3 on       841 to    941 s   (on: mg10 measured 193)
#     parallel n=4, 1 off + 4 on       675 to    951 s
#     TOTAL                        69 min to  85 min
# Cone's flag-on arms are the same shape mg10 already ran at batch 8192, so
# their base is measured and their band is narrow; parallel's have never been
# run, which is what this job is for, so only they take mg10's rule for an
# unmeasured arm (the high end assumes half again as much).  85 minutes is well
# under the 2.5-hour ceiling this campaign was given, so NOTHING IS TRIMMED and
# every swept batch runs.  --dry-run prints this same arithmetic before any GPU
# time is spent.
#     The walltime below is 04:00:00, about 2.8x the upper estimate.  That is
# more headroom than mg10 took, deliberately: mg10 could bound its patched arms
# from its own prototype timings, and the parallel flag-on arms here have no
# prior reading at all -- if the shape does not pay for parallel, those nine
# arms are SLOWER than their controls rather than faster, and the wall grows in
# the direction the estimate cannot see.
#
# IF THIS JOB MUST BE SPLIT OR CUT, use MG11_ARMS, which takes the tokens the
# dry-run prints.  Trim in this order: the four-device arms first, then the
# middle batches of each sweep (keep the ends -- the sweep exists to find where
# the curve turns).  Each geometry's anchor must stay with its own arms: the
# value distances are computed across arms in ONE process, so a job without the
# anchor loses the vs_anchor column for every arm in it.
#     MG11_ARMS=c1,c2_off,c2_on08192,c2_on16384,c2_on32768
#     MG11_ARMS=p1,p2_off,p2_on04096,p2_on08192,p2_on16384,p2_on32768
#     MG11_ARMS=c1,c4_off,c4_on08192,c4_on16384,c4_on32768
#     MG11_ARMS=p1,p4_off,p4_on04096,p4_on08192,p4_on16384,p4_on32768
# A split also costs the verdict block: the block for a geometry is built from
# the arms present in ONE run, so a geometry split across two jobs has no single
# quotable verdict.  Prefer trimming batches to splitting geometries.
#
# DISK.  Two shared sinograms at the 1024 cell, about 4 GB each, plus two
# strided reconstruction samples per arm at about 1.2 MB each (the value
# columns), all in $MG11_RESULTS during the run and deleted at the end unless
# MG11_KEEP_ARTIFACTS=1 is exported.  The distances are computed before anything
# is removed.
#
# THE TREE THIS JOB IMPORTS, and why it is a NEW one.  mg11 runs out of
# /scratch/gautschi/buzzard/torch_p3/mbirtorch_mg11.  mg9 and mg10 both ran out
# of mbirtorch_mg9, which is the pre-implementation export: it has no column
# gather in it at all, so mg11 cannot use it.  The tree ship is the session
# lead's step (see SHIP.md) and it MUST carry the parallel extension -- the
# prelude below proves it does, by asserting that
# ParallelBeamModel.column_gather_geometry is True and that a live parallel
# model resolves to the gather with the switch on.  A tree without the extension
# would import cleanly, pass every other check, and then fail all nine parallel
# flag-on arms one subprocess at a time.
#   This file does NOT `pip install -e` that tree, which is what mg5 and mg8 do.
# An editable install rewrites a path pointer inside the SHARED environment at
# torch_p0/env, so installing here would repoint any other job's imports
# mid-flight.  mg11 puts the tree on PYTHONPATH instead, which changes nothing
# outside this job's own processes, and then ASSERTS that the mbirtorch it
# imported really came from that tree.  If that assert fails, an editable
# install of some other tree is winning through a meta-path finder; check what
# else is running before fixing it with pip.
#
# THE COMPILER CACHE is mg9's, deliberately reused:
# TORCHINDUCTOR_CACHE_DIR=.../torch_cache_mg9.  Same cell, same geometries, same
# kernels, so the cache is warm for everything the control and anchor arms
# compile.  The flag-on arms launch at new SHAPES -- a different pixel count and
# a different column count per launch -- so they will compile some new
# artifacts; that cost lands in each arm's discarded cold pass, which is why the
# knobs go in before the cold pass and not after it.  Reusing the cache is only
# safe while no other job is writing it -- two concurrent jobs sharing one
# inductor cache is a recorded way to get a corrupt artifact and a mystery crash
# -- so check the queue before submitting.
#
# SCHEDULING, and why submission uses --begin.  This is a four-GPU block that
# can run to 04:00:00, so it cannot be submitted blind against the 02:00 and
# 03:00 nightlies: at the estimated 69 to 85 minutes it would fit between them,
# but the WALLTIME does not, and slurm schedules on the walltime.  Submit it
# with an explicit start time that clears both, for example
#     sbatch --begin=04:15 mg11_gautschi.sbatch
# or any hour that leaves four clear hours ahead of it.  SHIP.md carries the
# same instruction beside the submit command.  The job shares the node with
# nothing it needs.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch --begin=<time> mg11_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 04:00:00
#SBATCH -J torch_mg11
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg11_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
MG11_TREE=/scratch/gautschi/buzzard/torch_p3/mbirtorch_mg11
# mg9's cache, reused on purpose (see the header).
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache_mg9
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

# The import path, in place of an editable install (see the header).  The arm
# subprocesses inherit this, so every arm imports the same tree.
export PYTHONPATH=$MG11_TREE${PYTHONPATH:+:$PYTHONPATH}

nvidia-smi -L
test -d "$MG11_TREE/mbirtorch" || { echo "the tree is not staged at $MG11_TREE; mg11 needs the POST-IMPLEMENTATION export -- mbirtorch_mg9 is the pre-implementation tree and has no column gather in it (see SHIP.md)"; exit 1; }
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1

# ── THE PRELUDE ASSERTS, in six parts, all before any GPU time is spent ──────
# Part 1 is mg8's, mg9's and mg10's: the tree imports and four GPUs are visible.
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# Part 2 is mg9's and mg10's provenance check: the mbirtorch that got imported
# must be the staged tree and not some other tree an editable install points at.
export MG11_TREE
"$TORCHPY" -c "
import os, sys, mbirtorch
tree = os.path.realpath(os.environ['MG11_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(tree + os.sep):
    sys.exit('mbirtorch imported from ' + got + ', NOT from the staged tree ' + tree +
             '.  An editable install of another tree is winning through a '
             'meta-path finder.  Check what else is running before fixing this '
             'with pip.')
print('mbirtorch from', got)
" || exit 1

# Part 3: every python file in the tree and the harness itself must PARSE on
# THIS node.  The login node's filesystem client has intermittently returned
# truncated reads for files whose bytes are correct on disk; a module the arms
# import lazily would otherwise fail mid-job instead of here.  The harness is
# checked first, because part 6 below imports it.
"$TORCHPY" -m compileall -q /scratch/gautschi/buzzard/torch_p3/mg11_flip_gates.py || { echo "mg11_flip_gates.py does not parse on this node; re-copy and re-verify it (SHIP.md step 2)"; exit 1; }
"$TORCHPY" -m compileall -q "$MG11_TREE" || { echo "a file in $MG11_TREE does not parse on this node; re-verify the tree against its manifest"; exit 1; }

# Part 4, THE ONE THIS JOB TURNS ON.  mg11 patches nothing, so it has no patch
# seam to check -- what it has instead is a set of facts about the SHIPPED
# implementation that every arm and every witness rests on.  Each check below
# names the fact and what a silent failure would cost.  The environment is
# cleared of the switch first, because two of these read the DEFAULT.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
"$TORCHPY" -c "
import inspect, os, sys
import numpy as np
sys.path.insert(0, '/scratch/gautschi/buzzard/torch_p3')
import mbirtorch
from mbirtorch import _memory_ledger, _sharding
from mbirtorch.tomography_model import (TomographyModel, COLUMN_GATHER_ENV_VAR,
                                        FORWARD_PIXEL_BATCH)

# FACT 1: the switch is named what mg11 exports.  A rename here would leave
# every flag-on arm running the control under a treatment's name.
if COLUMN_GATHER_ENV_VAR != 'MBIRTORCH_FORWARD_COLUMN_GATHER':
    sys.exit('the library switch is named %r; mg11 exports '
             'MBIRTORCH_FORWARD_COLUMN_GATHER.' % (COLUMN_GATHER_ENV_VAR,))

# FACT 2: the names the arms reach through, all present.
for name in ('_column_gather_forward', '_forward_pixel_batch',
             '_sparse_forward_project_columns', '_sparse_forward_project_sharded',
             '_build_memory_ledger', '_banded_setup'):
    if not hasattr(TomographyModel, name):
        sys.exit('TomographyModel has no %s; mg11 reads or drives it.' % name)
for name in ('gather_column_band', 'broadcast_band_to_views', 'move_shard',
             'run_per_device'):
    if not hasattr(_sharding, name):
        sys.exit('_sharding has no %s; mg11 wraps it as a witness.' % name)

# FACT 3: the driver really consults the switch, and each shape really uses the
# primitive mg11 counts for it.  If the forward driver stopped calling
# _sparse_forward_project_columns, or the gather stopped going through
# gather_column_band, the witnesses would read zero on a healthy run.
src = inspect.getsource(TomographyModel._sparse_forward_project_sharded)
for token in ('_column_gather_forward', '_sparse_forward_project_columns',
              'broadcast_band_to_views'):
    if token not in src:
        sys.exit('the forward driver no longer mentions ' + token +
                 '; mg11 witnesses the two shapes through that name.')
cols = inspect.getsource(TomographyModel._sparse_forward_project_columns)
for token in ('gather_column_band', '_forward_pixel_batch',
              'sparse_forward_project_view_range'):
    if token not in cols:
        sys.exit('the column gather no longer mentions ' + token +
                 '; mg11 counts that call as its proof the gather ran.')

# FACT 4: the ledger reads the switch OFF THE MODEL rather than re-deriving it.
# This is what makes plan.column_pixel_batch an independent second witness AND
# what makes a flag-on arm's modeled peak price the ACTIVE path.
plan_src = inspect.getsource(_memory_ledger.plan_from_model)
for token in ('_column_gather_forward', '_forward_pixel_batch',
              'column_pixel_batch'):
    if token not in plan_src:
        sys.exit('_memory_ledger.plan_from_model no longer mentions ' + token +
                 '; the memory gate would price the wrong shape for every '
                 'flag-on arm and the witness would not exist.')
if not hasattr(_memory_ledger, 'estimate_peak_device_bytes'):
    sys.exit('_memory_ledger.estimate_peak_device_bytes is gone; that is the '
             'entry point the modeled peak comes from.')

# FACT 5: the shipped default batch is what cone's sweep starts at.
if int(FORWARD_PIXEL_BATCH) != 8192:
    sys.exit('the shipped FORWARD_PIXEL_BATCH is %r, not 8192; cone\'s sweep '
             'is built to start at the library default.' % (FORWARD_PIXEL_BATCH,))

# FACT 6, AND THE PARALLEL EXTENSION IS EXACTLY THIS ONE.  Both geometries must
# declare the capability, and a live model of each must resolve to the gather
# with the switch on and to the banded walk with it off.  A tree missing the
# parallel extension passes every check above and then fails all nine parallel
# flag-on arms one subprocess at a time.
if not mbirtorch.ConeBeamModel.column_gather_geometry:
    sys.exit('ConeBeamModel.column_gather_geometry is False; the cone gate '
             'cannot run against this tree.')
if not mbirtorch.ParallelBeamModel.column_gather_geometry:
    sys.exit('ParallelBeamModel.column_gather_geometry is False: THIS TREE DOES '
             'NOT CARRY THE PARALLEL EXTENSION.  Ship the tree that does (see '
             'SHIP.md); do not run the parallel half against this one.')
cell = (8, 24, 20)
par = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
cone = mbirtorch.ConeBeamModel(cell, np.linspace(0, 2*np.pi, cell[0], endpoint=False),
                               source_detector_dist=80.0, source_iso_dist=40.0)
for name, model in (('parallel', par), ('cone', cone)):
    # DEFAULT OFF.  If a geometry resolved to the gather with the switch unset,
    # the flag-off arms would not be the shipped configuration and there would
    # be no control in this job at all.
    if model._column_gather_forward():
        sys.exit('%s resolves to the column gather with the switch UNSET.  The '
                 'default is supposed to be off -- that is what mg11 exists to '
                 'change -- so this job would have no control arms.' % name)
    os.environ['MBIRTORCH_FORWARD_COLUMN_GATHER'] = '1'
    try:
        if not model._column_gather_forward():
            sys.exit('%s does NOT resolve to the column gather with the switch '
                     'forced on.  For parallel this is the row-aligned refusal '
                     'in the resolver, which the extension lifts for the gather '
                     'path; for cone it is a capability flag.  Every flag-on arm '
                     'of that geometry would abort.' % name)
    finally:
        del os.environ['MBIRTORCH_FORWARD_COLUMN_GATHER']
    if model._column_gather_forward():
        sys.exit('%s stayed on the gather after the switch was removed; the '
                 'override is not read per call and the arms cannot be run in '
                 'one session.' % name)
print('mg11 seam check passed: the switch is named as exported, both drivers '
      'and both primitives are where mg11 witnesses them, the ledger reads the '
      'switch off the model, the shipped default batch is 8192, and BOTH '
      'geometries carry the capability, default off and resolve on demand')
" || { echo "an mg11 seam moved in this tree, or the tree is missing the parallel extension; do not run the gates against it"; exit 1; }

# Part 5: THE BAR THE HARNESS QUOTES, re-read out of the test files.  mg11
# hard-codes the shipped parity floor as its FAILING bar and prints the citation
# beside every verdict.  A stale quoted bar is worse than no bar at all, so the
# number is checked against the tests it is quoted from before the job runs.
"$TORCHPY" -c "
import os, re, sys
tree = os.environ['MG11_TREE']
sys.path.insert(0, '/scratch/gautschi/buzzard/torch_p3')
import mg11_flip_gates as mg11
path = os.path.join(tree, 'tests', 'test_kernels_sharded.py')
if not os.path.exists(path):
    sys.exit('the standing kernel parity suite is not in this tree at ' + path +
             '; the value bar mg11 quotes cannot be verified.')
text = open(path).read()
match = re.search(r'^FLOOR\s*=\s*([0-9.eE+-]+)', text, re.M)
if not match:
    sys.exit('tests/test_kernels_sharded.py no longer defines FLOOR at module '
             'level; mg11 quotes that constant as its failing bar.')
floor = float(match.group(1))
if floor != mg11.PARITY_FLOOR:
    sys.exit('the standing suite holds FLOOR=%r and mg11 quotes %r as the '
             'failing bar.  Update the harness rather than running it with a '
             'stale bar.' % (floor, mg11.PARITY_FLOOR))
sharding = open(os.path.join(tree, 'tests', 'test_sharding.py')).read()
if 'test_column_gather_recon_matches_single_device' not in sharding:
    sys.exit('tests/test_sharding.py no longer carries '
             'test_column_gather_recon_matches_single_device; that test is half '
             'the citation mg11 prints beside its value verdict.')
print('the failing bar mg11 quotes (%g) is the one the standing suites hold' % floor)
" || { echo "the value bar mg11 quotes does not match the tests in this tree"; exit 1; }

# Part 6: the harness's own plan, printed before any GPU work, so a mis-set
# environment variable is caught here and not forty minutes in.
# (run at the end of the prelude, after the environment below is settled)

# ── THE ENVIRONMENT, set explicitly, nothing inherited ───────────────────────
# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, because the measurement must be taken in the shipped
# environment and not in a variant of it; every arm re-asserts it.
export MBIRTORCH_DISABLE_TRITON=0
# THE SWITCH IS UNSET HERE AND STAYS UNSET.  The harness sets it per arm -- '1'
# on a flag-on arm and '0' on every control and anchor -- and each arm asserts
# the value it got.  A value inherited from here would apply to the whole job
# and every arm would abort on the assert, which is the right failure but a
# needlessly expensive one.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
# Each arm pins its own device count, which the SCRIPT pops and then sets per
# subprocess.  A value inherited from here would pin the whole job to one count
# without saying so.  Each arm then asserts the device list it actually got, and
# a wrong count is FATAL: every shape number in an arm is derived from the shard
# length, so a mislabeled count would mislabel the whole row.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, and the memory
# gate reads that counter after every timed reconstruction.  mg11 never sets it,
# here or anywhere; it builds its own ledger instead, from the same entry point.
unset MBIRTORCH_MEMORY_CALIBRATION
# The speed guard is bypassed by a device-count pin anyway; unset so nothing
# inherited changes the automatic path an arm never takes.
unset MBIRTORCH_WIDENING_GUARD
# MG11_ARMS is deliberately NOT unset: it is the one knob the header and SHIP.md
# tell a reader to export from the SUBMITTING SHELL, and slurm's default
# --export=ALL is what carries it in.  It is echoed so the log records what the
# job was asked for.
echo "MG11_ARMS=${MG11_ARMS:-<unset: the full 20-arm list>}"
# The sweep-shaping and artifact knobs ARE unset, so nothing stale in a
# submitting shell can silently change what the swept batches are or how long
# each arm runs.
unset MG11_CONE_BATCHES
unset MG11_PARALLEL_BATCHES
unset MG11_ITERATIONS
unset MG11_WARM_REPEATS
unset MG11_KEEP_ARTIFACTS
unset MG11_SMOKE
unset MG11_DEVICE

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the script, which is the
# convention the rows are collected under.
export MG11_RESULTS=/scratch/gautschi/buzzard/torch_p3
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first (prelude part 6), so the log opens with the twenty-two
# arms, the cylinder each flag-on arm assembles in megabytes, and the wall
# arithmetic -- and a mis-set environment variable is caught before any GPU work.
"$TORCHPY" -u mg11_flip_gates.py --dry-run || exit 1
"$TORCHPY" -u mg11_flip_gates.py
echo "MG11 DONE"
