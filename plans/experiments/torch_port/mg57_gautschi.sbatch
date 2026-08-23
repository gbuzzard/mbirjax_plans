#!/bin/bash
# mg57: what does a user wait through on a first reconstruction, and which
# parts of that wait are paid again?  The full statement is in
# mg57_cold_start.py; the essentials are repeated here because a reader of this
# file may never open that one.
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS JOB MUST NOT EXPORT TORCHINDUCTOR_CACHE_DIR AT THE TOP LEVEL, AND THAT
# IS THE ONE THING TO GET RIGHT ABOUT IT.
#
# Every other sbatch in this series opens with
#   export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
# which is correct for a job that wants its compiles to persist.  Here the
# CACHE STATE IS THE VARIABLE UNDER TEST.  A shared warm inductor cache would
# make the "first run" arms start warm and this job would measure nothing it
# set out to measure.  The harness owns the cache directories instead: it makes
# one pair per arm under its own results directory, empties them for the arms
# that must start cold, and sets BOTH TORCHINDUCTOR_CACHE_DIR and
# TRITON_CACHE_DIR in each arm's environment.
#
# Both, because Triton keeps a separate JIT cache and an unset TRITON_CACHE_DIR
# sends it to ~/.triton, where artifacts from every earlier job on this cluster
# are sitting.  An arm that started with an empty inductor directory and a full
# ~/.triton would report a cold start that was nothing of the kind.
#
# So the hygiene block below UNSETS both variables rather than exporting either.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS JOB EXISTS.  Every recorded measurement in this series reports a warm
# median and discards the cold pass.  That is right for comparing two routes,
# but it means the cold pass has never been attributed, and a user does not get
# to discard it: the first reconstruction of a session is the one they sit and
# watch.  This job takes that wait apart, once, at the parallel (1024, 1008, 992)
# cell -- the recorded production class -- and separates three costs that behave
# differently:
#   * work a compile cache removes, so a second session never repeats it;
#   * work every process pays once, which no cache touches -- the imports, the
#     CUDA context, the model build, the device settle, loading artifacts back
#     out of the cache;
#   * the reconstruction arithmetic itself, which is what the warm medians in
#     this series report.
# It decides nothing.  It edits no library file, changes no default and proposes
# no remedy; it prints numbers, and the reading is a person's job.
#
# WHAT IT MEASURES.
#
#   THE INPUT IS STAGED FIRST, IN ITS OWN PROCESS.  A measured arm LOADS its
#   sinogram from an npz and verifies the md5 beside it.  It never forward
#   projects one, because a forward projection compiles the projection bodies
#   and fills the very caches the arm exists to find empty.  The staging runs
#   before any arm, in cache directories no arm ever reads.  A staged file that
#   verifies is reused; only when none exists is one built (about 3.8 GiB).
#   The weights are NOT staged: each arm builds them on the host from the loaded
#   sinogram before the timed reconstruction, and times that separately.
#
#   FOUR ARMS, each a fresh subprocess under a 30-minute cap, each naming its
#   devices one by one:
#     first_run_n1   both caches empty, one device, ONE instrumented
#                    reconstruction -- a user's first-ever run.
#     cached_n1      both caches pointed at the pair first_run_n1 filled, one
#                    device, TWO instrumented reconstructions: a cold PROCESS
#                    with warm caches, then the in-process warm baseline.
#     first_run_n4   as first_run_n1 at four devices, with fresh directories of
#                    its own.  Four devices is what the automatic policy chooses
#                    at this cell, so this is the other cold start a user meets.
#     cached_n4      as cached_n1 at four devices.
#   Protocol on every arm: seed 13 before each reconstruction, three VCD
#   iterations, the stopping threshold disabled.
#
#   THE PHASE SPLIT is the central instrument.  A host clock is wrapped from
#   OUTSIDE the package around initialize_recon (and gen_set_of_pixel_partitions
#   inside it), _apply_device_policy, recon_direct, _initial_error_state,
#   compute_hessian_diagonal, create_vcd_subset_updater and _vcd_recon; the
#   wrappers go on immediately before the measured reconstruction and come off
#   in a finally.  Each timer synchronizes the arm's devices before it starts
#   and before it stops, so device work lands on the phase that issued it; that
#   removes some overlap and lengthens the reconstruction slightly, which is the
#   price of attribution, so each arm also runs one UN-INSTRUMENTED
#   reconstruction and the row carries its plain wall.
#
#   COMPILE ATTRIBUTION comes from the tools' own accounting: dynamo's
#   compile_times(repr='csv', aggregate=True) and its counters after each
#   reconstruction, the size of the shared Triton launch-key set before and
#   after, per-call timers on the two parallel Triton wrappers recording which
#   calls GREW that set, and the number of compiled callables the projector
#   layer holds.  Both cache directories' file counts and total bytes are
#   recorded before and after every arm.
#
# THIS JOB TESTS THE INSTALLED TREE, so it runs the editable install of
# /scratch/gautschi/buzzard/torch_p3/mbirtorch_src, exactly as the
# host-cost-split job and the floors refresh do.  (The two multiaxis-kernel jobs
# in this series select a CANDIDATE tree through PYTHONPATH instead and install
# nothing; that is the right form for an unreviewed tree and the wrong one here,
# because the subject is what a user's installed library costs on a first run.)
# The editable install repoints the shared torch_p0 environment, so THIS JOB
# MUST NOT RUN AT THE SAME TIME as another torch_p3 job that installs into the
# same environment: a concurrent install races this one and leaves a torn .pth.
# Chain instead, with --dependency=afterany.
#
# WALL ESTIMATE, and its arithmetic.
#   * THE WARM WALL AT THIS CELL IS KNOWN.  The widening-floors refresh (job
#     15435735) measured the parallel (1024, 1008, 992) cell at a 21.301 s warm
#     median on one device with a 0.1 percent spread, and 9.527 s on four with
#     a 5.2 percent spread, on this same seeded three-iteration protocol.
#   * THE COLD-START OVERHEAD IS THE UNKNOWN THIS JOB EXISTS TO MEASURE, so it
#     cannot be budgeted from a measurement.  What can be budgeted: the two
#     cached arms are each about two warm reconstructions plus one plain pass
#     plus their process overhead -- call it four minutes at one device and
#     three at four.  The two first-run arms are one cold reconstruction each
#     plus one plain pass; if a cold start were ten times its warm wall they
#     would be about four and two minutes.  Staging is an md5 of 3.8 GiB if the
#     file is there, or a phantom, a forward projection and 3.8 GiB of writes if
#     it is not -- maybe ten minutes.  So the expected path is well under half
#     an hour and 01:30:00 leaves real headroom, including the case where the
#     staged file has to be built.
#   * IF A COLD START IS FAR WORSE THAN THAT, each arm is capped at 30 minutes
#     and a timeout is RECORDED as that arm's result rather than failing the
#     job.  Four arms all running into their caps would exceed the walltime;
#     the jsonl is flushed row by row, so a job cut short yields everything it
#     finished and MG57_ARMS re-runs the rest against the staged file whose md5
#     still matches.
#
# DEVICE MEMORY.  This cell is a recorded production class at one device and at
# four, so no arm here is near a capacity boundary.  An arm that does run out is
# RECORDED as that arm's result and the run continues.
#
# DISK.  A few megabytes of jsonl in $MG57_RESULTS, plus 3.8 GiB of staged npz
# if it has to be built, plus the compile caches this job creates under
# $MG57_RESULTS/caches -- three pairs of directories, which is what the job is
# measuring the filling of.  They are left in place after the run so their
# contents can be inspected; delete them when the readings have been taken.
#
# ONE THING THE READINGS DEPEND ON, AND IT IS WORTH KNOWING BEFORE READING
# THEM.  $MG57_RESULTS is on the shared parallel filesystem, so the cache
# directories this job owns live there too, and both writing an artifact and
# reading one back cross that filesystem.  A cache on a node's local disk would
# behave differently.  Every arm's row carries the directory paths and the file
# counts and bytes it wrote, so which filesystem produced a number is a fact on
# the row rather than an assumption.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every planned arm produced a row
# or a recorded out-of-memory or timeout, every arm loaded the staged input with
# its md5 verified, every arm realized the device count it asked for, the
# wrapped phases and launch timers were restored, and the tree witnesses hold.
# What any arm MEASURED never changes it, including a cold start that turns out
# to be slow.  Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg57_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 01:30:00
#SBATCH -J mg57_cold_start
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg57_%j.log
# Four GPUs and 56 cores: the whole node.  The partition wants 14 CPUs per GPU,
# so four GPUs take 56.
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg

nvidia-smi -L

# ── the mbirtorch preflight ──────────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this one
# and leaves a torn .pth (the cluster notes' recorded failure).  Chain instead,
# with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch; print('library under test:', mbirtorch.__file__)" || exit 1
# FOUR devices, not one: the four-device arms name cuda:0 through cuda:3
# explicitly, and an allocation with fewer would fail both of them after the
# queue wait rather than before it.
"$TORCHPY" -c "
import torch
count = torch.cuda.device_count()
assert count >= 4, ('this job needs four CUDA devices; found', count)
print('cuda devices', count, [torch.cuda.get_device_name(i) for i in range(count)])
" || exit 1

# ── the tree witnesses ───────────────────────────────────────────────────────
# The first three are the block every job in this series carries: they say this
# is the padded, recompile-remedied tree the recorded walls were measured on.
# The third matters directly here, because compilation is the subject: a tree
# without it would hand a torch body eager python rather than a compiled one,
# and there would be no compile to attribute.
#
# The kernel pairs are what makes the Triton side of the attribution real.  The
# PARALLEL pair is the one this job times, so both wrappers must exist, both
# must carry the per-view cost attribute the driver reads to choose a view batch
# (the harness's own wrapper copies that attribute for the same reason), and the
# parallel geometry's selection must consult both availability functions and
# reach both wrappers.  The MULTIAXIS pair is checked too, because it identifies
# the committed tip this job is meant to be testing.  Everything here is read by
# source inspection and attribute lookup -- no model is built, no device is
# touched and no CUDA is initialized -- so it costs nothing and cannot perturb
# what the arms then measure.  The harness records all of it on its header row.
"$TORCHPY" -c "
import inspect
from mbirtorch import projectors, triton_multiaxis, triton_parallel
from mbirtorch._utils import padded_kernel_width
from mbirtorch.multiaxis_parallel import MultiAxisParallelModel
from mbirtorch.parallel_beam import ParallelBeamModel
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
print('triton_parallel: ', triton_parallel.__file__)
print('triton_multiaxis:', triton_multiaxis.__file__)
pairs = ((triton_parallel, '_parallel_forward_view_batch_triton', '_parallel_back_view_batch_triton'),
         (triton_multiaxis, '_multiaxis_forward_view_batch_triton', '_multiaxis_back_view_batch_triton'))
for module, fwd, back in pairs:
    for attr in (fwd, back):
        body = getattr(module, attr)
        assert body.__name__.endswith('_view_batch_triton'), body.__name__
        assert getattr(body, '_view_batch_cost', None) is not None, \
            (attr, 'carries no _view_batch_cost, so the driver would batch it as a torch body')
        print('  kernel body', body.__name__, 'carries _view_batch_cost')
selection = inspect.getsource(ParallelBeamModel._view_batch_bodies)
assert 'parallel_forward_kernel_usable' in selection and \
       'parallel_back_kernel_usable' in selection, \
    'the parallel selection does not consult the availability functions'
assert '_parallel_forward_view_batch_triton' in selection and \
       '_parallel_back_view_batch_triton' in selection, \
    'the parallel selection does not reach the kernel wrappers'
selection = inspect.getsource(MultiAxisParallelModel._view_batch_bodies)
assert 'multiaxis_forward_kernel_usable' in selection and \
       'multiaxis_back_kernel_usable' in selection, \
    'the multiaxis selection does not consult the availability functions'
print('tree witnesses ok')
" || exit 1

# ── the seams this job wraps, checked before the queue time is spent ─────────
# The phase split is the whole instrument.  A seam that has been renamed is
# recorded as absent and the run continues, which is right at run time and
# wrong at submit time: a job that measured four arms with half the seams
# missing would be a wasted node.  This costs nothing and touches no device.
"$TORCHPY" -c "
from mbirtorch import vcd_utils
from mbirtorch.tomography_model import TomographyModel
seams = ('initialize_recon', '_apply_device_policy', 'recon_direct',
         '_initial_error_state', 'compute_hessian_diagonal',
         'create_vcd_subset_updater', '_vcd_recon')
missing = [name for name in seams if not hasattr(TomographyModel, name)]
if not hasattr(vcd_utils, 'gen_set_of_pixel_partitions'):
    missing.append('vcd_utils.gen_set_of_pixel_partitions')
assert not missing, ('these seams do not exist in this tree', missing)
print('all', len(seams) + 1, 'phase seams resolve')
" || exit 1

# ── the environment, set explicitly ──────────────────────────────────────────
# THE TWO COMPILE CACHES ARE THE VARIABLE UNDER TEST, so neither is exported
# here.  See the block at the top of this file.  The harness makes one pair of
# directories per arm under $MG57_RESULTS/caches, empties the pair for each arm
# that must start cold, and sets both variables in that arm's environment.  A
# value inherited from the submitting shell would override the harness's own
# and hand a "first run" arm somebody else's compiles.
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
# The hand-written kernels are the shipped configuration and the harness sets
# this per child; unset so an inherited value cannot turn every arm into a
# measurement of the torch bodies instead.
unset MBIRTORCH_DISABLE_TRITON
# This job reads the peak memory counters itself, and the calibration mode owns
# them and does extra work at the settle and at the end of every call.
unset MBIRTORCH_MEMORY_CALIBRATION
# EVERY ARM NAMES ITS DEVICES ONE BY ONE, so a process-wide count pin is not the
# mechanism here.  It is unset rather than set, and the harness pops it from
# every child's environment.
unset MBIRTORCH_NUM_DEVICES
# The widening speed floors only reorder the AUTOMATIC branch's candidates, and
# no arm here takes that branch.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted, and this job
# is measuring compilation.
unset MBIRTORCH_RECOMPILE_LIMIT
# The process-wide column-gather and sorted-forward overrides change which
# driver runs, and the four-device arms do enter the multi-device drivers.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
# The allocator segment policy is read when CUDA initializes and changes what a
# peak reading and a context cost mean.  Every arm must run under the same one,
# and the default is the one every recorded number in this series used.
unset PYTORCH_CUDA_ALLOC_CONF
# The library under test is the INSTALLED tree, installed above.  A PYTHONPATH
# inherited from the shell would silently select a different one.
unset PYTHONPATH
unset MG57_SMOKE
unset MG57_DRY
unset MG57_ARMS
unset MG57_ITERATIONS
unset MG57_ARM_TIMEOUT_MIN
unset MG57_PLAIN_PASS
# The child variables are the arm protocol's internal mode.  A stray value in
# the submitting shell would make the whole run execute one arm's job
# description and write its row somewhere no table reads.
unset MG57_CHILD
unset MG57_CHILD_OUT

export MG57_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg57_cold_start
mkdir -p "$MG57_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg57_cold_start.py || { echo "mg57_cold_start.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with every job in run order, the cache
# directory pair each arm gets and whether it is emptied first, the device list
# each arm names, the seams the phase split wraps and the per-arm cap -- and so
# a mis-set variable is caught before any GPU work.  The dry plan imports no
# torch and touches no device.
export MG57_DRY=1
"$TORCHPY" -u mg57_cold_start.py || exit 1
unset MG57_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what it
# measured: the per-arm phase tables, the once-per-process costs, the cache
# directory readings, the three-way comparison at each device count and the
# compile-attribution table are all in this log, and a slow cold start is a
# recorded result rather than a failure.
"$TORCHPY" -u mg57_cold_start.py
rc=$?
echo "MG57 DONE rc=$rc"
exit $rc
