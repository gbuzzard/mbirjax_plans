#!/bin/bash
# mg54: how fast is a multiaxis reconstruction with the two hand-written
# multiaxis kernels bound, against the same reconstruction with them off?
# One H100.  The full statement is in mg54_multiaxis_kernel_ab.py; the
# essentials are repeated here because a reader of this file may never open
# that one.
#
# WHY THIS JOB EXISTS.  The multiaxis geometry now selects two hand-written
# Triton kernels wherever their availability checks pass.  Those checks answer
# one question only -- whether each kernel REPRODUCES the torch body it
# replaces on the device that will run it -- and the geometry's own
# _view_batch_bodies says so at the selection: the tile constants were adopted
# from the cone kernels rather than swept, and nothing has yet timed a
# multiaxis reconstruction with the kernels bound.  There is no composed speed
# measurement for them at all.  This job makes the first one: seeded
# 3-iteration reconstructions at the three recorded multiaxis cells, the kernel
# route against the torch-body route, the same staged sinograms and weights on
# both sides, warm medians, on one device.
#
# It decides nothing and implements nothing.  It edits no library file and
# writes no kernel.  The ONE thing it varies is MBIRTORCH_DISABLE_TRITON, per
# arm, in a fresh process -- the library's own kill switch, not a knob invented
# for this job.  The exit code reports whether the instrument worked; the
# readings are read by a person, and acting on them needs Greg's approval.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE ONE DELIBERATE DIFFERENCE FROM EVERY OTHER JOB IN THIS SERIES: THIS JOB
# TESTS A CANDIDATE TREE THROUGH PYTHONPATH AND RUNS NO EDITABLE INSTALL.
#
# Every earlier job here pip-installed its tree editable into the shared
# torch_p0 environment and then imported mbirtorch normally.  This one must
# not.  The tree under test is the CANDIDATE at
# /scratch/gautschi/buzzard/torch_p3/mbirtorch_dev, which is the tree that
# routes these multiaxis kernels; the tree installed under torch_p0/env is the
# COMMITTED one, which does not.  Installing the candidate would repoint that
# shared environment, so every other job on this cluster and the nightly
# regression would silently start importing an unreviewed tree -- for as long
# as it stayed installed, which is until somebody noticed.  That is not a risk
# worth taking for a measurement.
#
# So the candidate is selected by PYTHONPATH alone, the shared install is left
# exactly as it was, and the block below ASSERTS that mbirtorch.__file__
# resolves under mbirtorch_dev before anything else runs.  Two consequences
# follow, and both are deliberate:
#   * PYTHONPATH is NOT unset in the hygiene block below, unlike mg53's, and
#     the harness does NOT pop it from its children's environment, unlike
#     mg52's.  It is the whole selection mechanism here.  Every job the harness
#     spawns records the mbirtorch file it actually imported, so which tree ran
#     is a fact on every row rather than an assumption.
#   * This job installs nothing, so it does not have to be serialized against
#     other torch_p3 jobs the way an installing job does.  It still reads the
#     shared environment's torch and triton.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IT MEASURES.  Three multiaxis cells, cheapest first so a defect shows up
# in minutes rather than after an hour:
#
#   multiaxis  (512, 448, 384)
#   multiaxis  (768, 672, 576)
#   multiaxis  (1024, 1008, 992)
#
# and two routes at each: "kernel", which is the default environment where the
# availability checks decide, and "torch", which sets MBIRTORCH_DISABLE_TRITON=1
# in that arm's environment.  Six arms, each a fresh subprocess under a
# 45-minute cap.  Each arm ASSERTS its premise before any clock starts -- a
# kernel arm that _view_batch_bodies returned two names ending
# _view_batch_triton, a torch arm that both bodies are the torch ones -- and an
# arm whose premise fails times nothing at all.  A wall taken on the other
# route's bodies and filed under this route's name is the one way this
# comparison could be quietly wrong.
#
# THE STAGED INPUTS ARE MG52'S, REUSED.  mg52 staged exactly these three cells
# and its run record says the npz files remain on scratch under
# results/mg52_framework_anchor for repeat runs.  Their contents are everything
# this job needs, so the harness uses the same filenames and the same keys,
# searches its own results directory and then mg52's, verifies the md5 of what
# it finds, and reuses it; only if no verified copy exists anywhere does it
# stage fresh, into its own results directory, under the same name.  A fresh
# stage runs with MBIRTORCH_DISABLE_TRITON=1 on purpose: mg52's sinograms are
# torch-body forward projections, because they were staged before these kernels
# were routed, and a re-stage should be the same input rather than a new one.
# Both arms of a cell read one file either way.
#
# THE PROTOCOL PER ARM is mg52's, unchanged, because the recorded one-device
# walls this job's torch arms should reproduce were measured under it: seed 13
# before every reconstruction, three VCD iterations with the stopping threshold
# disabled, one cold pass then three timed warm ones, warm median and spread
# recorded.  Each arm also records the peak allocated and reserved bytes reset
# before the cold pass and again before the warm passes, the bound body names,
# what both availability checks answered and why, and a float64 value
# fingerprint of the final reconstruction taken on the device.  The parent then
# prints, per cell, the kernel/torch warm ratio, the peak-memory ratio and the
# relative fingerprint differences.  Fingerprints are RECORDED, never gated: the
# availability checks already hold each kernel to a tolerance against its torch
# body at a tiny shape, and what a composed difference over three iterations
# shows is float summation order.
#
# THE SINGLE-CALL LEG runs last, in the parent, on the kernel route only, at the
# (1024, 1008, 992) cell.  mg53 measured the torch-body driver choosing a view
# batch of ONE view there -- 1024 body calls per projection -- against 9 views
# (57 body calls) at the (512, 448, 384) cell, and most of one call's host time
# went inside those 1024 body calls.  The batch is chosen by the cost model of
# the body actually bound, and a kernel body carries its own _view_batch_cost
# attribute declaring a much smaller per-view charge than the gather transient
# the driver prices a torch body at.  So the same driver should choose far
# fewer, far larger view batches with the kernels bound.  This leg makes one
# untimed warm-up call and then one measured call per direction with a host
# clock over index 0 of the projector's bound-body list, and records the
# body-call count beside the driver's chosen batch.  That is the number that
# says whether the one-view batches became large ones.
#
# WALL ESTIMATE, and its arithmetic.
#   * THE TORCH ARMS ARE KNOWN.  The recorded one-device warm walls for these
#     three cells are 11.4 s, 56.3 s and 309.9 s, and mg52's run record shows
#     its own torch arms reproducing them at 11.41, 56.30 and 310.06 under this
#     same protocol.  Each arm is a cold pass plus three warm ones, and mg52
#     recorded torch cold passes from 23 to 322 s.  So the 512-class arm is
#     about a minute, the 768-class about four, and the 1024-class about 21 --
#     a bit over half an hour for the three.
#   * THE KERNEL ARMS ARE THE UNKNOWN.  That is the entire reason for the job.
#     Each is capped at 45 minutes by the harness.
#   * STAGING.  If mg52's three multiaxis npz files are still on scratch -- 0.7,
#     2.2 and 7.6 GiB -- staging is an md5 of about 10.5 GiB and costs a few
#     minutes.  If they are gone it is three phantoms, three forward
#     projections and 10.5 GB of writes, which is maybe fifteen minutes.
#   * THE SINGLE-CALL LEG.  A model, a phantom the size of the 1024-class
#     volume, its host-side mask index, then two calls in each direction.  mg53
#     measured the torch bodies' single warm calls at that cell at 38.053 s
#     forward and 23.647 s back; the kernel route's are part of what this job
#     is measuring.  Call it fifteen minutes with the setup.
# So the known parts come to about an hour, and 03:00:00 leaves roughly two
# hours for the three kernel arms.  If all three ran all the way into their
# caps the job would not finish, and that is an honest possibility rather than
# a padded-away one: the jsonl is flushed row by row, so a job cut short yields
# everything it finished, and MG54_CELLS and MG54_ROUTES re-run only the rest,
# reusing every staged file whose md5 still matches.
#
# DEVICE MEMORY.  mg52 recorded one-device peaks of 11.38, 15.12 and 34.75 GB
# for these three cells on the torch route, all well inside an 80 GB H100.  The
# kernel route's peaks are part of what this job reports.  An arm that runs out
# of device memory, or hits its cap, is recorded as that arm's result and the
# run continues; that is a reading, not a broken instrument.
#
# DISK.  A few megabytes of jsonl in $MG54_RESULTS, plus about 10.5 GB of
# staged npz files only if mg52's are gone and this job has to rebuild them.
# Anything it stages is KEPT, under mg52's own filenames, so the next run of
# either reconstructs the same bytes rather than a regenerated sinogram that is
# not bit-identical.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY, and it covers four things:
# every planned arm produced a row or a recorded out-of-memory or timeout,
# every kernel arm really bound both kernels and every torch arm really bound
# both torch bodies, the staged inputs md5-verified, and the tree witnesses
# hold.  What any arm MEASURED never changes it.  Nothing is chained to this
# job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg54_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:00:00
#SBATCH -J mg54_kernel_ab
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg54_%j.log
# One GPU and 14 cores: the one-GPU shape every single-device job in this
# series has used, which is a quarter of the node.
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
CANDIDATE=/scratch/gautschi/buzzard/torch_p3/mbirtorch_dev
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L

# ── the candidate tree, and the assertion that it is the one being imported ──
# See the block at the top of this file.  PYTHONPATH is the ONLY selection
# mechanism here: no editable install runs, so the shared torch_p0 environment
# keeps pointing at the committed tree for every other job and the nightly.
test -d "$CANDIDATE" || { echo "the candidate tree is missing at $CANDIDATE: rsync it there before submitting"; exit 1; }
export PYTHONPATH="$CANDIDATE"
# BEFORE ANYTHING ELSE.  If mbirtorch resolves anywhere but under the candidate
# tree, every number below would describe the installed tree instead -- which
# does not route these kernels at all, so the whole job would silently measure
# one route twice.  Stop here rather than measure that.
"$TORCHPY" -c "
import os, mbirtorch
path = os.path.abspath(mbirtorch.__file__)
print('library under test:', path)
assert '/mbirtorch_dev/' in path, ('mbirtorch did not resolve under the candidate tree', path)
" || { echo "mbirtorch does not resolve under $CANDIDATE; PYTHONPATH selection failed and this job would measure the installed tree"; exit 1; }

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -c "import torch; assert torch.cuda.device_count() >= 1, torch.cuda.device_count(); print('cuda devices', torch.cuda.device_count(), torch.cuda.get_device_name(0))" || exit 1

# ── the tree witnesses ───────────────────────────────────────────────────────
# The first three are the block every job in this series carries: they say this
# is the padded, recompile-remedied tree the recorded one-device walls were
# measured on, and the third matters directly because a tree without it would
# hand the TORCH arms eager python rather than the compiled bodies those walls
# were measured with.
#
# The last two are this job's own, and they are what makes the kernel route a
# route rather than a name: the kernel module imports and holds both wrappers
# with their per-view cost attributes, and the geometry's _view_batch_bodies
# consults both availability functions and reaches both wrappers.  Both are read
# by SOURCE INSPECTION and attribute lookup -- no model is built, no device is
# touched and no CUDA is initialized -- so this costs nothing and cannot itself
# perturb what the arms then measure.  The harness records all five on its
# header row as well.
"$TORCHPY" -c "
import inspect
from mbirtorch import projectors, triton_multiaxis
from mbirtorch._utils import padded_kernel_width
from mbirtorch.multiaxis_parallel import MultiAxisParallelModel
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
print('triton_multiaxis:', triton_multiaxis.__file__)
for attr in ('_multiaxis_forward_view_batch_triton', '_multiaxis_back_view_batch_triton'):
    body = getattr(triton_multiaxis, attr)
    assert body.__name__.endswith('_view_batch_triton'), body.__name__
    assert getattr(body, '_view_batch_cost', None) is not None, \
        (attr, 'carries no _view_batch_cost, so the driver would batch it as a torch body')
    print('  kernel body', body.__name__, 'carries _view_batch_cost')
selection = inspect.getsource(MultiAxisParallelModel._view_batch_bodies)
assert 'multiaxis_forward_kernel_usable' in selection and \
       'multiaxis_back_kernel_usable' in selection, \
    'the geometry selection does not consult the availability functions'
assert '_multiaxis_forward_view_batch_triton' in selection and \
       '_multiaxis_back_view_batch_triton' in selection, \
    'the geometry selection does not reach the kernel wrappers'
print('tree witnesses ok; the selection consults both availability checks and reaches both kernels')
" || exit 1

# ── the environment, set explicitly ──────────────────────────────────────────
# MBIRTORCH_DISABLE_TRITON IS DELIBERATELY NOT EXPORTED HERE.  It is the ONE
# variable under test, and the harness sets it per arm in each child's own
# environment: 0 for a kernel arm, 1 for a torch arm, and 1 for staging so a
# fresh sinogram is the torch-body forward projection the staged files already
# on disk hold.  A value exported at this level would reach every arm and make
# both routes of a cell the same route under two names -- which is exactly the
# failure this job is built to be unable to have.  It is unset instead, so an
# inherited value from the submitting shell cannot do the same thing.
unset MBIRTORCH_DISABLE_TRITON
# This job reads the peak memory counters itself, and the calibration mode owns
# them and does extra work at the settle and at the end of every call.
unset MBIRTORCH_MEMORY_CALIBRATION
# One device is what the job allocates; the harness sets MBIRTORCH_NUM_DEVICES=1
# per child, which is mg52's pin and keeps the model on the automatic branch.
# A value inherited from the shell would change the layout under it.
unset MBIRTORCH_NUM_DEVICES
# A one-device pin bypasses the speed floors anyway; unset so no inherited value
# can change what an arm builds around it.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted, and the torch
# arms measure compiled bodies across three cell shapes.
unset MBIRTORCH_RECOMPILE_LIMIT
# The process-wide column-gather and sorted-forward overrides change which
# driver runs.  These arms reconstruct on one device and never enter a
# multi-device driver, but an inherited value would still change what they build.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
# The allocator segment policy is read when CUDA initializes and changes what a
# peak reading means.  Both routes must run under the same one, and the default
# is the one every recorded peak in this series was measured under.
unset PYTORCH_CUDA_ALLOC_CONF
# NOTE: PYTHONPATH is NOT unset here, unlike every other sbatch in this series.
# It is the candidate tree's only selection mechanism; see the block at the top.
unset MG54_SMOKE
unset MG54_DRY
unset MG54_CELLS
unset MG54_ROUTES
unset MG54_WARM
unset MG54_ARM_TIMEOUT_MIN
unset MG54_SINGLE_CALL
# The child variables are the arm protocol's internal mode.  A stray value in
# the submitting shell would make the whole run execute one arm's job
# description and write its row somewhere no table reads.
unset MG54_CHILD
unset MG54_CHILD_OUT

export MG54_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg54_multiaxis_kernel_ab
mkdir -p "$MG54_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg54_multiaxis_kernel_ab.py || { echo "mg54_multiaxis_kernel_ab.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with the ten jobs in run order, the staged
# files it will look for and where, the two routes and what each asserts, the
# single-call leg and its bounds -- and so a mis-set variable is caught before
# any GPU work.  The dry plan imports no torch and touches no device.
export MG54_DRY=1
"$TORCHPY" -u mg54_multiaxis_kernel_ab.py || exit 1
unset MG54_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what it
# measured: the side-by-side table, the warm and peak ratios, the fingerprint
# gaps, the view batch each route's cost model chose and the single-call split
# are all in this log, and an arm that ran out of memory or hit its cap is a
# recorded result rather than a failure.
"$TORCHPY" -u mg54_multiaxis_kernel_ab.py
rc=$?
echo "MG54 DONE rc=$rc"
exit $rc
