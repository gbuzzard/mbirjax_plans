#!/bin/bash
# mg59: which frames does dynamo trace on every reconstruction, and would
# marking the pixel-count dimension dynamic remove most of that tracing?  The
# full statement is in mg59_dynamo_tracing.py; the essentials are repeated here
# because a reader of this file may never open that one.
#
# ─────────────────────────────────────────────────────────────────────────────
# THIS JOB MUST NOT EXPORT TORCHINDUCTOR_CACHE_DIR OR TRITON_CACHE_DIR AT THE
# TOP LEVEL, AND THAT IS THE ONE THING TO GET RIGHT ABOUT IT.
#
# Every other sbatch in this series opens with
#   export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
# which is correct for a job that wants its compiles to persist.  Here every arm
# must run on a cache holding ITS OWN mode's artifacts and nothing else: the
# candidate's graphs are different graphs from the baseline's, so a candidate
# arm pointed at a shared cache the baseline filled would miss on every one of
# them and pay code generation the baseline did not, and the wall comparison
# would be about cache contents rather than about tracing.  The harness owns the
# cache directories instead: one pair per (mode, device count) under its own
# results directory, each filled by its own throwaway arm, with BOTH variables
# set in every arm's environment.
#
# Both, because Triton keeps a separate JIT cache and an unset TRITON_CACHE_DIR
# sends it to ~/.triton, where artifacts from every earlier job on this cluster
# are sitting.
#
# So the hygiene block below UNSETS both variables rather than exporting either.
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY THIS JOB EXISTS.  mg57 (job 15449106) measured that at the parallel
# (1024, 1008, 992) cell a reconstruction in a process with FULL compile caches
# still spends 4.42 s at one device and 8.00 s at four inside dynamo, by
# dynamo's own entire_frame_compile phase, and that the unique graph count is 9
# at one device and 36 at four -- exactly four times, which follows the
# per-device compiled instance design in projectors.py.  At four devices that
# 8.00 s precedes a 9.60 s reconstruction, so it is the largest remaining
# per-process cost, and a compile cache cannot remove it: the cache skips
# inductor code generation, not dynamo tracing.  What has never been said is
# WHICH frames those graphs belong to and WHY each variant was made.
#
# WHAT IT MEASURES.
#
#   PART 1, ATTRIBUTION.  Every measured arm records, from the tools' own
#   accounting: dynamo's counters (cleared before each reconstruction);
#   the length of dynamo's cache-entry list for every function the library
#   compiled, which is the authoritative per-frame variant count, enumerated by
#   READING the keys of projectors._COMPILE_CACHE; the TORCH_LOGS=recompiles
#   output, written to a per-arm file by TORCH_LOGS_OUT and read back and
#   summarized by the arm, which is where the GUARD FAILURE behind each
#   recompilation comes from; and compile_times(repr='csv') with the cumulative
#   per-phase totals as a before/after delta.  The deliverable is a table naming
#   every compiled frame, its variant count, and what varies across its
#   variants.
#
#   PART 2, THE CANDIDATE.  The hypothesis is that the compiled bodies
#   specialize on the PIXEL-COUNT dimension, which varies because each VCD
#   iteration projects a different subset size (the default granularity list is
#   [1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128] and a three-iteration run
#   visits levels 4, 16 and 64).  The candidate marks that dimension dynamic
#   with torch._dynamo.mark_dynamic before each wrapped body is called, from
#   OUTSIDE the package: on the projector functions' per-device body lists, on
#   model._get_update_direction as an instance attribute, and on the per-device
#   compiled bodies the subset updater holds, reached through the closure of the
#   updater model.create_vcd_subset_updater returns.  Nothing at module level in
#   the mbirtorch package is written and NO LIBRARY FILE IS EDITED; every
#   wrapper comes off in a finally and the row reports whether it did.
#
#   FOR PARALLEL BEAM THE TWO PROJECTION BODIES ARE HAND-WRITTEN TRITON KERNELS
#   AND DO NOT COMPILE, so marking their inputs is inert on this build.  The row
#   records which bound entries were the Triton wrappers, so that is visible
#   rather than assumed.
#
#   EIGHT ARMS, each a fresh subprocess under a 15-minute cap, each naming its
#   devices one by one, all with FULL compile caches:
#     fill_base_n1 / base_n1     baseline at one device; the fill arm empties
#                                and fills that pair, the measured arm reads it.
#     fill_cand_n1 / cand_n1     the candidate at one device, own pair.
#     fill_base_n4 / base_n4     baseline at four devices, own pair.
#     fill_cand_n4 / cand_n4     the candidate at four devices, own pair.
#   Each measured arm runs TWO reconstructions in its one process: the first
#   carries the tracing cost this job is about, the second shows whether the
#   candidate's one general graph is slower at steady state than the several
#   specialized graphs it replaces.
#   Protocol on every arm: seed 13 before each reconstruction, three VCD
#   iterations, the stopping threshold disabled.
#
#   THE INPUT IS STAGED FIRST, IN ITS OWN PROCESS.  A measured arm LOADS its
#   sinogram from an npz and verifies the md5 beside it.  It never forward
#   projects one, because a forward projection traces the very frames the arm
#   exists to count.  The harness looks for a verified staged copy of this cell
#   in its own results directory and in the cold-start job's results directory
#   beside it, and builds one (about 3.8 GiB) only when none is found.
#
#   THE VALUE CHECK.  Every reconstruction is fingerprinted in float64 -- the
#   sum of absolute values and the sum of squares over the volume -- and the
#   comparison prints the candidate's relative difference from the baseline at
#   the same device count against a declared 1e-6 tolerance.  A candidate that
#   changes the values is a failed candidate and the table says so.  That is a
#   FINDING, not an instrument failure, so it does not touch the exit code.
#
# THIS JOB TESTS THE INSTALLED TREE, so it runs the editable install of
# /scratch/gautschi/buzzard/torch_p3/mbirtorch_src, exactly as the cold-start
# job and the floors refresh do.  The editable install repoints the shared
# torch_p0 environment, so THIS JOB MUST NOT RUN AT THE SAME TIME as another
# torch_p3 job that installs into the same environment: a concurrent install
# races this one and leaves a torn .pth.  Chain instead, with
# --dependency=afterany.
#
# WALL ESTIMATE, and its arithmetic.
#   * THE WALLS AT THIS CELL ARE KNOWN.  mg57 measured a new process on full
#     caches at 26.02 s on one device and 19.51 s on four, and an in-process
#     warm reconstruction at 21.17 s and 9.60 s.
#   * A measured arm is one new-process reconstruction plus one in-process warm
#     one plus its process overhead: call it 75 s at one device and 50 s at
#     four.  A fill arm is one first-ever reconstruction -- mg57 measured 49.48 s
#     and 41.11 s -- plus overhead, so about 80 s and 70 s.  Four measured arms
#     and four fill arms come to roughly nine minutes.
#   * Staging is an md5 of 3.8 GiB if a verified file is there, or a phantom, a
#     forward projection and 3.8 GiB of writes if it is not -- maybe ten
#     minutes.  So the expected path is well under half an hour and 01:30:00
#     leaves real headroom, including the case where the staged file has to be
#     built.
#   * IF AN ARM IS FAR WORSE THAN THAT it is capped at 15 minutes and the
#     timeout is RECORDED as that arm's result rather than failing the job.  The
#     jsonl is flushed row by row, so a job cut short yields everything it
#     finished and MG59_ARMS re-runs the rest against the staged file whose md5
#     still matches.
#
# DEVICE MEMORY.  This cell is a recorded production class at one device and at
# four, so no arm here is near a capacity boundary.  An arm that does run out is
# RECORDED as that arm's result and the run continues.
#
# DISK.  A few megabytes of jsonl and recompile logs in $MG59_RESULTS, plus
# 3.8 GiB of staged npz if it has to be built, plus the compile caches this job
# creates under $MG59_RESULTS/caches -- six pairs of directories.  They are left
# in place after the run so their contents can be inspected; delete them when
# the readings have been taken.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every planned arm produced a row
# or a recorded out-of-memory or timeout, every arm loaded the staged input with
# its md5 verified, every arm realized the device count it asked for, every
# reuse arm found its caches already filled, every candidate arm installed its
# wrappers and restored them, every arm produced a readable recompile log and a
# readable per-frame variant count, and the tree witnesses hold.  What any arm
# MEASURED never changes it -- including a candidate that removes no graphs, or
# one that changes the reconstructed values.  Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg59_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 01:30:00
#SBATCH -J torch_mg59
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg59_%j.log
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
# explicitly, and an allocation with fewer would fail all four of them after the
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
# and there would be no tracing to attribute.
#
# The parallel kernel pair is checked because it is WHY the two projection
# bodies do not compile on this build: the wrappers exist, they carry the
# per-view cost attribute the driver reads to choose a view batch (the
# harness's own wrapper copies that attribute for the same reason), and the
# parallel geometry's selection consults both availability functions and reaches
# both wrappers.  Everything here is read by source inspection and attribute
# lookup -- no model is built, no device is touched and no CUDA is initialized --
# so it costs nothing and cannot perturb what the arms then measure.  The
# harness records all of it on its header row.
"$TORCHPY" -c "
import inspect
from mbirtorch import projectors, triton_parallel
from mbirtorch._utils import padded_kernel_width
from mbirtorch.parallel_beam import ParallelBeamModel
assert padded_kernel_width(504) == 512, 'not the padded tree'
assert projectors._RECOMPILE_LIMIT_FLOOR >= 64, 'the recompile remedy is missing'
source = inspect.getsource(projectors.maybe_compile)
assert '_raise_recompile_budget()' in source.split('_GLOBAL_COMPILE_LOCK:')[-1], \
    'the raise is not on the compiling-thread path'
assert 'instance_key' in source, 'maybe_compile has no per-device instance key'
print('triton_parallel: ', triton_parallel.__file__)
for attr in ('_parallel_forward_view_batch_triton', '_parallel_back_view_batch_triton'):
    body = getattr(triton_parallel, attr)
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
print('tree witnesses ok')
" || exit 1

# ── the seams the candidate wraps, checked before the queue time is spent ────
# Part 2 is the whole reason the job runs twice per device count.  A seam that
# has been renamed is recorded as a note and the arm continues, which is right
# at run time and wrong at submit time: a job that ran four candidate arms
# wrapping nothing would be a wasted node.  This costs nothing and touches no
# device.  The updater's per-device body lists are checked by NAME in the source
# of create_vcd_subset_updater, because that is where they are bound and the
# closure of the updater it returns is the only place they can be reached from.
"$TORCHPY" -c "
import inspect
from mbirtorch.projectors import Projectors
from mbirtorch.tomography_model import TomographyModel
init = inspect.getsource(Projectors.__init__)
assert '_fwd_body_per_dev' in init and '_back_body_per_dev' in init, \
    'the projector functions no longer hold per-device body lists'
for name in ('_get_update_direction', 'create_vcd_subset_updater'):
    assert hasattr(TomographyModel, name), ('this seam does not exist', name)
updater = inspect.getsource(TomographyModel.create_vcd_subset_updater)
lists = ('qggmrf_grad_hess', 'prior_line_terms', 'lin_quad_const',
         'lin_quad_weighted', 'apply_update')
missing = [name for name in lists if name not in updater]
assert not missing, ('the updater no longer binds these per-device lists', missing)
assert 'instance_key=i' in updater, 'the updater no longer binds one instance per device'
print('all', len(lists) + 3, 'candidate seams resolve')
" || exit 1

# ── the introspection this job reads its numbers from ────────────────────────
# Part 1 is only as good as these four.  If a torch upgrade moved any of them
# the job would still run and would print an empty table, so they are checked
# here where the failure is cheap.
"$TORCHPY" -c "
import torch
from torch._dynamo import utils as du
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
assert callable(torch._dynamo.mark_dynamic), 'no mark_dynamic'
assert callable(_debug_get_cache_entry_list), 'no cache entry list'
assert du.counters is not None, 'no dynamo counters'
assert callable(du.compile_times), 'no compile_times'
assert du.cumulative_time_spent_ns is not None, 'no cumulative phase totals'
def body(x, y):
    return (x * y).sum(0)
compiled = torch.compile(body)
for n in (4, 8):
    compiled(torch.randn(n, 3), torch.randn(n, 3))
entries = _debug_get_cache_entry_list(body.__code__)
assert len(entries) >= 2, ('the cache entry list does not see per-shape variants', len(entries))
print('dynamo introspection ok:', len(entries), 'variants on a two-shape probe')
" || exit 1

# ── the environment, set explicitly ──────────────────────────────────────────
# THE CACHE STATE IS PART OF THE DESIGN, so neither cache variable is exported
# here.  See the block at the top of this file.  The harness makes one pair of
# directories per (mode, device count) under $MG59_RESULTS/caches, empties the
# pair for each fill arm, and sets both variables in every arm's environment.  A
# value inherited from the submitting shell would override the harness's own and
# hand one mode the other mode's compiles.
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
# The harness sets these per arm, on baseline and candidate alike, and points
# them at a per-arm file it reads back.  An inherited value would send every
# arm's log to one file and the per-arm attribution would be lost.
unset TORCH_LOGS
unset TORCH_LOGS_OUT
# The hand-written kernels are the shipped configuration and the harness sets
# this per child; unset so an inherited value cannot turn every arm into a
# measurement of the torch bodies instead -- which would change WHICH frames
# compile, and that is the subject.
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
# The recompile floor stays at the shipped value: an override here would change
# how many variants a frame is allowed before dynamo gives up, and counting
# variants is the subject.
unset MBIRTORCH_RECOMPILE_LIMIT
# The process-wide column-gather and sorted-forward overrides change which
# driver runs, and the four-device arms do enter the multi-device drivers.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset MBIRTORCH_SORTED_FORWARD
# The allocator segment policy is read when CUDA initializes and changes what a
# peak reading means.  Every arm must run under the same one, and the default is
# the one every recorded number in this series used.
unset PYTORCH_CUDA_ALLOC_CONF
# The library under test is the INSTALLED tree, installed above.  A PYTHONPATH
# inherited from the shell would silently select a different one.
unset PYTHONPATH
unset MG59_SMOKE
unset MG59_DRY
unset MG59_ARMS
unset MG59_ITERATIONS
unset MG59_ARM_TIMEOUT_MIN
unset MG59_STAGE_DIRS
# The child variables are the arm protocol's internal mode.  A stray value in
# the submitting shell would make the whole run execute one arm's job
# description and write its row somewhere no table reads.
unset MG59_CHILD
unset MG59_CHILD_OUT

export MG59_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg59_dynamo_tracing
mkdir -p "$MG59_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

test -f mg59_dynamo_tracing.py || { echo "mg59_dynamo_tracing.py is not in $PWD"; exit 1; }

# The plan first, so the log opens with every job in run order, the cache
# directory pair each arm gets and whether it is emptied first, the device list
# each arm names, where the staged input is searched for, where the candidate's
# wrappers go and the per-arm cap -- and so a mis-set variable is caught before
# any GPU work.  The dry plan imports no torch and touches no device.
export MG59_DRY=1
"$TORCHPY" -u mg59_dynamo_tracing.py || exit 1
unset MG59_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what it
# measured: the per-frame variant tables, the guard-failure tables, the
# candidate's marking tables and the baseline-against-candidate comparison at
# each device count are all in this log, and a candidate that helps nothing is a
# recorded result rather than a failure.
"$TORCHPY" -u mg59_dynamo_tracing.py
rc=$?
echo "MG59 DONE rc=$rc"
exit $rc
