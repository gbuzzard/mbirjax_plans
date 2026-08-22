#!/bin/bash
# mg55: with the two hand-written multiaxis kernels bound, does sharding now
# divide the per-device memory peaks and pay in speed -- and does a 2048-class
# reconstruction complete on the standard four-GPU node?  The full statement is
# in mg55_multiaxis_scale.py; the essentials are repeated here because a reader
# of this file may never open that one.
#
# WHY THIS JOB EXISTS.  The two multiaxis kernels have now been timed against
# the torch bodies they replace, on one device, at three cells: the kernel
# route ran at 0.25, 0.23 and 0.22 of the torch route's warm wall -- between
# 4.0 and 4.6 times faster -- and its peak device memory fell with it, from
# 11.38 to 1.96 GB, 15.12 to 6.54 GB and 34.75 to 24.11 GB.  Time and memory
# moved for the same reason: the torch bodies held about fourteen slabs of
# (view batch, pixels, max(rows, slices)) floats at once, and a kernel body
# holds the problem arrays plus a small per-(view, pixel) contract instead, so
# the largest temporary class is gone.
#
# Two questions are left that only several devices can answer.
#   * WHETHER SHARDING NOW PAYS.  With the torch bodies bound, adding devices
#     did not divide the per-device peak -- the slab temporaries were sized by
#     the view batch and the pixel count rather than by the shard, so they
#     stayed on every device.  With them gone the per-device peak should fall
#     roughly as the shard does.  Nothing has measured it.
#   * WHETHER THE 2048-CLASS RUNS AT ALL.  Its volume is eight times the
#     1024-class volume and it has twice the views; no single device holds it,
#     and with the torch bodies bound it could not be run at any device count.
# This job measures both.  It decides nothing and implements nothing: it edits
# no library file and writes no kernel.  The one thing it varies is the device
# list handed to the model, per arm, in a fresh process.  The exit code reports
# whether the instrument worked; the readings are read by a person, and acting
# on them needs Greg's approval.
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS JOB TESTS A CANDIDATE TREE THROUGH PYTHONPATH AND RUNS NO EDITABLE
# INSTALL, exactly as the one-device kernel comparison did.
#
# The tree under test is the CANDIDATE at
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
#   * PYTHONPATH is NOT unset in the hygiene block below, and the harness does
#     NOT pop it from its children's environment.  It is the whole selection
#     mechanism here.  Every job the harness spawns records the mbirtorch file
#     it actually imported, so which tree ran is a fact on every row rather
#     than an assumption.
#   * This job installs nothing, so it does not have to be serialized against
#     other torch_p3 jobs the way an installing job does.  It still reads the
#     shared environment's torch and triton.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHAT IT MEASURES.
#
#   PART A, the 1024-class across device counts.  The cell is
#   (1024, 1008, 992), reconstructed at one, two and four devices on the
#   default kernel selection, from the SAME staged npz every earlier run of
#   this cell read (md5 798c72e1cf5bb7803b9f2b02294753c6, about 7.6 GiB).  The
#   harness searches its own results directory and then
#   results/mg54_multiaxis_kernel_ab and results/mg52_framework_anchor,
#   verifies the md5 of what it finds and reuses it; only if no verified copy
#   exists anywhere does it stage fresh, into its own results directory, under
#   the same name and with MBIRTORCH_DISABLE_TRITON=1 so the sinogram is the
#   torch-body forward projection the file on disk holds.  Protocol per arm:
#   seed 13, three VCD iterations with the stopping threshold disabled as a
#   cold pass, then three timed warm ones, warm median and spread.  Each arm
#   records the peak allocated and reserved bytes ON EVERY DEVICE IN ITS LIST,
#   reset before the cold pass and again before the warm passes, the busiest
#   device's reading beside them, the bound body names, and a float64 value
#   fingerprint of the reconstruction.  The parent computes the speedup against
#   the one-device arm and the fingerprint differences across counts.
#
#   PART B, the 2048-class demonstration.  The cell is (2048, 2016, 1984) --
#   this family's shapes doubled -- at four devices, which is the
#   demonstration, and at two, which is optional and on by default.  There is
#   no npz: each arm builds its own phantom, forward projects it through its
#   own model and makes the weights, because nothing else needs those bytes and
#   at this cell they are about 33 GB of sinogram and another 33 of weights.
#   The single-device case is NOT run; it is priced instead in part C.  Each
#   arm sets skip_memory_preflight, and the row carries the reason: the memory
#   ledger is recorded as modeling the 1024-class at 68 GB on one device
#   against a measured peak of 34.75 GB, so a preflight reading about twice the
#   real peak could refuse a run that fits.
#
#   PART C, the memory ledger's own numbers.  Last and in the parent process,
#   read only: for both cells at counts 1, 2 and 4, the modeled planned
#   per-device peak for a reconstruction, through the library's own
#   plan_from_model and estimate_peak_device_bytes, beside the measured peaks
#   from parts A and B wherever those exist.  Both entry points are pure
#   arithmetic -- no device is queried and nothing is allocated -- so a count
#   the model is not configured for can be priced.  These rows are the input to
#   the ledger correction that follows this job.
#
# THE DEVICE LIST IS EXPLICIT ON EVERY ARM, and that is deliberate.  Each arm
# calls configure_devices with the devices named one by one rather than asking
# for a count.  The automatic policy and the speed thresholds that order its
# candidates are not the subject here: they decide whether a layout is worth
# using, and this job is measuring what a layout costs and what it returns.
# Naming the devices makes the layout a fact on the row instead of a policy
# outcome, and keeps a two-device arm from settling on four.
#
# EVERY ARM ASSERTS ITS PREMISE BEFORE ANY CLOCK STARTS: that the geometry's
# selection returned two body names ending _view_batch_triton.  An arm whose
# premise fails times NOTHING.  A wall taken on the torch bodies and filed
# under a kernel arm's name is the one way this measurement could be quietly
# wrong.
#
# WALL ESTIMATE, and its arithmetic.
#   * THE 1024-CLASS ARMS ARE PART-KNOWN.  On one device this cell measured a
#     69.53 s cold pass and a 67.64 s warm median on the kernel route, and that
#     arm's whole subprocess took 299.1 s for a cold pass plus three warm ones
#     -- so about 27 s of process start-up, md5 verification and staged load on
#     top of the four reconstructions.  The one-device arm here is therefore
#     about five minutes.  The two- and four-device arms are the unknown; that
#     is the point of the job, and each is capped at 45 minutes.
#   * THE 2048-CLASS ARMS ARE AN ESTIMATE, and nothing at this size has been
#     measured.  The reconstruction volume goes from (992, 992, 1148) to
#     (1984, 1984, 2297) -- 1.13 billion voxels to 9.04 billion, a factor of
#     8.0 -- and the views double, so one reconstruction is about 16 times the
#     work IF the per-voxel-per-view cost holds.  On that basis a warm pass is
#     67.64 x 16 = about 18 minutes and the cold pass about 18.5, so a cold
#     pass plus two warm ones is about 55 minutes at the one-device rate.  Four
#     devices may be faster or slower than that, which is what the arm is for.
#     Each 2048 arm is capped at 75 minutes.
#   * THE IN-PROCESS 2048 STAGING IS ALSO AN ESTIMATE.  One full forward
#     projection call on this route at the 1024-class measured 8.972 s, so at
#     16 times the work it is about 2.4 minutes, plus the phantom build (which
#     runs on the devices in bounded blocks) and the host weights.  Call the
#     whole staging five minutes.
#   * STAGING FOR PART A.  If the 1024-class npz is still on scratch, staging
#     is an md5 of 7.6 GiB, which took 14.4 s the last time it was read.  If it
#     is gone it is a phantom, a forward projection and 7.6 GiB of writes,
#     which is maybe fifteen minutes.
# So the expected path is about two hours and twenty minutes, and 04:00:00
# leaves real headroom.  If every arm ran all the way into its cap (three at 45
# minutes and two at 75 comes to 285 minutes) the job would not finish, and
# that is an honest possibility rather than a padded-away one: the jsonl is
# flushed row by row, so a job cut short yields everything it finished, and
# MG55_ARMS re-runs only the rest, reusing the staged file whose md5 still
# matches.
#
# DEVICE MEMORY.  The 1024-class peaked at 24.11 GB allocated and 36.99 GB
# reserved on one device on this route, so it fits one H100 with room; what
# four devices do to that is the reading.  At the 2048-class the problem arrays
# alone are 30.5 GiB of sinogram and 33.7 GiB of volume, which over four
# devices is about 7.6 and 8.4 GiB each before any working set.  An arm that
# runs out of device memory, or hits its cap, is RECORDED as that arm's result
# and the run continues; that is a reading, not a broken instrument.
#
# HOST MEMORY.  The 2048-class arm holds its sinogram (30.5 GiB) and weights
# (30.5 GiB) while the reconstruction returns a volume (33.7 GiB): about 95 GiB
# at the peak.  The phantom is another 33.7 GiB and is released as soon as it
# has been forward projected, and the weights are built a block at a time so
# the host never holds three sinogram-sized arrays at once.  A host that cannot
# hold this lands as a worker killed with no row, which the harness classifies
# as a capacity reading rather than a fault.
#
# DISK.  A few megabytes of jsonl in $MG55_RESULTS, plus 7.6 GiB of staged npz
# only if the 1024-class file is gone and this job has to rebuild it.  The
# 2048-class arms write nothing: their inputs live and die inside the arm.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY, and it covers four things:
# every planned arm produced a row or a recorded out-of-memory or timeout,
# every arm really bound both kernels, the staged input md5-verified where it
# was reused, and the tree witnesses hold.  What any arm MEASURED never changes
# it -- including a 2048-class arm that ran out of memory.  Nothing is chained
# to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg55_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 04:00:00
#SBATCH -J mg55_multiaxis_scale
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg55_%j.log
# Four GPUs and 56 cores: the whole node.  The partition wants 14 CPUs per GPU
# -- the one-GPU jobs in this series ask for 14 -- so four GPUs take 56.
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
# does not route these kernels at all, so every arm would fail its premise and
# the job would measure nothing.  Stop here rather than measure that.
"$TORCHPY" -c "
import os, mbirtorch
path = os.path.abspath(mbirtorch.__file__)
print('library under test:', path)
assert '/mbirtorch_dev/' in path, ('mbirtorch did not resolve under the candidate tree', path)
" || { echo "mbirtorch does not resolve under $CANDIDATE; PYTHONPATH selection failed and this job would measure the installed tree"; exit 1; }

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# FOUR devices, not one: the four-device arms name cuda:0 through cuda:3
# explicitly, and an allocation with fewer would fail every one of them after
# the queue wait rather than before it.
"$TORCHPY" -c "
import torch
count = torch.cuda.device_count()
assert count >= 4, ('this job needs four CUDA devices; found', count)
print('cuda devices', count, [torch.cuda.get_device_name(i) for i in range(count)])
" || exit 1

# ── the tree witnesses ───────────────────────────────────────────────────────
# The first three are the block every job in this series carries: they say this
# is the padded, recompile-remedied tree the recorded one-device walls were
# measured on, and the third matters because a tree without it would hand a
# torch body eager python rather than a compiled one.
#
# The last two are what makes the kernel route a route rather than a name: the
# kernel module imports and holds both wrappers with their per-view cost
# attributes, and the geometry's _view_batch_bodies consults both availability
# functions and reaches both wrappers.  The cost attribute matters twice here:
# the driver reads it to choose a view batch, and the memory ledger reads it to
# decide how to price that batch -- which is exactly what part C reports.  Both
# witnesses are read by SOURCE INSPECTION and attribute lookup -- no model is
# built, no device is touched and no CUDA is initialized -- so this costs
# nothing and cannot itself perturb what the arms then measure.  The harness
# records all five on its header row as well.
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
        (attr, 'carries no _view_batch_cost, so the driver would batch it as a torch body and the ledger would price it as one')
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
# MBIRTORCH_DISABLE_TRITON IS DELIBERATELY NOT EXPORTED HERE.  Every arm in
# this job runs the default kernel selection, and the harness sets the variable
# per child: 0 for every arm, and 1 for a fresh 1024-class staging so its
# sinogram is the torch-body forward projection the staged file on disk already
# holds.  A value exported at this level would reach every arm and could turn
# the whole job into a measurement of the route this work has already measured.
# It is unset instead, so an inherited value from the submitting shell cannot
# do the same thing.
unset MBIRTORCH_DISABLE_TRITON
# This job reads the peak memory counters itself, and the calibration mode owns
# them and does extra work at the settle and at the end of every call.
unset MBIRTORCH_MEMORY_CALIBRATION
# EVERY ARM NAMES ITS DEVICES ONE BY ONE, so a process-wide count pin is not
# the mechanism here.  It is unset rather than set, and the harness pops it
# from every child's environment: a pin would not change an explicit layout but
# would cap the phantom build, which reads the same variable, at a count this
# job did not ask for.
unset MBIRTORCH_NUM_DEVICES
# The widening speed floors only reorder the AUTOMATIC branch's candidates, and
# no arm here takes that branch.  Unset so no inherited value changes what an
# arm builds around it.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted.
unset MBIRTORCH_RECOMPILE_LIMIT
# The process-wide column-gather and sorted-forward overrides change which
# driver runs.  These arms DO enter the multi-device drivers, so an inherited
# value here would change what two and four devices measure.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
# The allocator segment policy is read when CUDA initializes and changes what a
# peak reading means.  Every arm must run under the same one, and the default
# is the one every recorded peak in this series was measured under.
unset PYTORCH_CUDA_ALLOC_CONF
# NOTE: PYTHONPATH is NOT unset here.  It is the candidate tree's only
# selection mechanism; see the block at the top.
unset MG55_SMOKE
unset MG55_DRY
unset MG55_ARMS
unset MG55_WARM
unset MG55_2048_WARM
unset MG55_2048_N2
unset MG55_ARM_TIMEOUT_MIN
unset MG55_2048_TIMEOUT_MIN
# The child variables are the arm protocol's internal mode.  A stray value in
# the submitting shell would make the whole run execute one arm's job
# description and write its row somewhere no table reads.
unset MG55_CHILD
unset MG55_CHILD_OUT

export MG55_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg55_multiaxis_scale
mkdir -p "$MG55_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg55_multiaxis_scale.py || { echo "mg55_multiaxis_scale.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with every job in run order, the staged file
# it will look for and where, the device list each arm names, the per-arm caps
# and what each part records -- and so a mis-set variable is caught before any
# GPU work.  The dry plan imports no torch and touches no device.
export MG55_DRY=1
"$TORCHPY" -u mg55_multiaxis_scale.py || exit 1
unset MG55_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what it
# measured: the per-count table, the per-device peaks, the fingerprint gaps,
# the 2048-class walls and the ledger pricing rows are all in this log, and an
# arm that ran out of memory or hit its cap is a recorded result rather than a
# failure.
"$TORCHPY" -u mg55_multiaxis_scale.py
rc=$?
echo "MG55 DONE rc=$rc"
exit $rc
