#!/bin/bash
# mg13: IS THE SORTED CHANNEL REDUCTION FASTER THAN THE ATOMIC SCATTER IN THE
# PARALLEL FORWARD?  One source tree, three bodies, six arms, four H100s, all
# measured in one job.  The full statement of what is measured and why is in
# mg13_sorted_probe.py; the essentials are repeated here because a reader of this
# file may never open that one.
#
# WHAT THIS JOB IS ABOUT.  The parallel forward bins each voxel's weighted rows
# into the detector channels it lands on.  Today the torch body does that with an
# atomic scatter -- one index_add_ per point-spread tap, with every pixel landing
# on a channel colliding with every other pixel landing there.  mbirjax does the
# same reduction by sorting the contributions by channel, summing each run of
# equal channels, and writing each channel once, and it measured that form faster
# on GPU wherever the mean collisions per channel is high enough.  This job puts
# the two side by side on the same inputs and the same substrate.
#
# WHY IT IS BEING RE-ASKED.  This port declined the sorted form once, on a
# threshold that has since moved: the forward now gathers full-width columns by
# default, so a body call carries the whole slice axis (1008 columns here) where
# the banded walk fed it a shard's width, and the cross-device copies have moved
# off the compute streams.  The recorded reasoning and the instruction for the
# re-gate are in section 8.5 of the forward remedy memo, and that instruction is
# explicit: probe the LIGHT PER-CALL FORM first.  That is what this job runs.
#
# THIS JOB DECIDES NOTHING.  The harness carries no stop threshold and no gate,
# on purpose.  It prints two mechanical lines and a table of value distances, and
# whether those clear the recorded threshold is a human re-read of the memo and
# of mbirjax's own selection constants against this reading.
#
# THE THREE BODIES, all on ONE source tree.  Unlike mg12 there is no source
# change here and therefore nothing to overlay: the candidate is installed by the
# harness inside its own process, which is legal for a probe, and the tree is
# never edited.  What separates the arms is two environment variables.
#   triton    MBIRTORCH_DISABLE_TRITON=0, nothing patched.  The production
#             reference: on CUDA the parallel forward normally runs the
#             hand-written Triton kernel, NOT the torch fan, so an arm that
#             measured only the torch fan would not be measuring what production
#             runs.  This arm says what the other two are being compared against.
#   scatter   MBIRTORCH_DISABLE_TRITON=1, nothing patched.  The BASELINE for the
#             candidate: the compiled torch fan with the shipped index_add_.
#   sorted    MBIRTORCH_DISABLE_TRITON=1, and the harness rebinds the name the
#             forward body calls -- mbirtorch.parallel_beam.fan_forward_batch --
#             to its own sort + segment-sum + one-write-per-run implementation.
#             The CANDIDATE.  It shares the baseline's substrate exactly, so the
#             only difference between those two arms is the reduction.
# The candidate is compared against the baseline; the kernel is measured beside
# them.  A sorted-against-triton ratio would confound the reduction with the
# whole compiled-body-versus-kernel difference, and is never drawn.
#
# WHAT MUST BE ON THE CLUSTER BEFORE THIS IS SUBMITTED.  The tree comes from
# GitHub and there are no overlay files at all, so exactly one file has to be
# shipped:
#     /scratch/gautschi/buzzard/torch_p3/mg13_sorted_probe.py
# The job checks it is present and that it parses before it spends any GPU time.
#
# WHAT THE JOB BUILDS AND CHECKS, in order, all before the first arm runs.
#   1. The clone, with a clean retry: the tree is REMOVED between attempts, so a
#      half-written clone from a failed attempt cannot be finished by the next one
#      into something that is neither commit.
#   2. The checked-out hash is read back and compared against the pinned one.
#      NOTE the pin is on origin/greg_dev rather than the default branch; a plain
#      clone fetches every branch, so the hash resolves, and the read-back is what
#      proves it did.
#   3. .git is removed once the hash is recorded, beside the tree and in the log.
#   4. compileall on the tree AND on the harness.  This is the integrity gate for
#      the truncated-read failure this node's filesystem client has produced
#      before: a module an arm imports lazily would otherwise fail forty minutes
#      in instead of here.
#   5. THE SEAM CHECK, which is where this job differs from its predecessors,
#      because the candidate is a patch rather than a tree.  A fresh process on
#      the clone's own path proves six things: that mbirtorch came from the clone;
#      that the forward body still calls fan_forward_batch BY THAT NAME out of
#      parallel_beam's globals, which is the harness's patch point and the one
#      thing that would make the candidate silently not run; that the rebinding
#      takes and the patched body really reaches the sorted reduction; that the
#      sorted reduction reproduces the shipped one's values on this tree; that
#      torch.segment_reduce exists in this environment; and that a live parallel
#      model resolves to the column gather with the switch unset, which is the
#      shape the four-device arms are supposed to measure.
#   6. THE KILL SWITCH, checked in both positions on the real GPU: with
#      MBIRTORCH_DISABLE_TRITON=1 the library must report the kernels unusable and
#      with 0 it must report them usable.  If that switch did not bite, all three
#      arms would run the same body and the job would report a very tidy nothing.
#
# THE ARMS.  Six measured arms plus one untimed sinogram generator, all at mg9's,
# mg10's, mg11's and mg12's cell (1024, 1008, 992), which gives 1008 slices --
# divisible by 1 and 4, so nothing is padded.
#     n=1   triton | scatter | sorted     the question: it is about one call
#     n=4   triton | scatter | sorted     the composed check: the candidate has to
#                                         survive the sharded driver and the
#                                         column gather, not just a bench
# ARM ORDER is device-count-major: the three bodies of one count run
# back-to-back.  A truncated job then yields COMPLETE three-way comparisons, and
# any drift in the node over the hour the job runs lands inside a comparison
# rather than between its columns.
#     The one-device arms also run an ISOLATED EAGER BENCH before their timed
# reconstructions: the same forward body called directly on shared inputs over a
# ladder of pixel counts, which sweeps the mean collisions per channel -- the
# variable mbirjax's own selection rule is written in.  The composed run measures
# the compiled path; the bench measures the two reductions with the compiler held
# out of it, and the harness records the dynamo graph-break counts so a
# disagreement between the two has somewhere to be attributed.
#
# WHAT COULD FAIL THIS INCREMENT.  Nothing here is a gate, but two readings are
# worth knowing about in advance.  The first is MEMORY: the sorted form holds all
# three point-spread taps' contributions at once where the scatter holds one at a
# time, so its transient is three times the scatter's -- 6.3 GB at the
# four-device launch shape.  The harness bounds that by splitting the sort along
# the VIEW axis (contributions from different views never share an accumulator
# row, so the split cannot change which contributions collide), using the
# library's own per-batch transient budget as the ceiling, and it records the
# realized chunk and transient on every row.  The second is VALUE: sorting
# changes the summation order, so the candidate cannot be bit-identical to the
# scatter.  The harness holds it to 1e-5 relative and reports the distances two
# independent ways -- on identical inputs inside the bench, and across whole
# reconstructions -- with each arm's own pass-to-pass distance printed as the
# floor to read them against.
#
# WALL ARITHMETIC.  The measured anchors are mg11's four-device parallel arm at
# this cell (21.8 s composed, row mg11_flip_gates_h001_20260811_041522.jsonl) and
# this campaign's one-device parallel reading (40.0 s composed).  A torch arm is
# scaled by the composed five-arm gate's ratio between the compiled torch bodies
# and the kernel path at this cell (5.56x of jax against 1.90x, so about 2.9x).
# Each arm is 1 cold discarded pass plus 2 warm ones.
#     1 generator                                        70 s
#     n1-triton   (40.0 s composed)                     270 s  including the bench
#     n1-scatter  (117 s composed)                      500 s  including the bench
#     n1-sorted   (never run; = scatter at the low end)  500 s  including the bench
#     n4-triton   (21.8 s composed)                     125 s
#     n4-scatter  (63.8 s composed)                     250 s
#     n4-sorted   (never run; = scatter at the low end)  250 s
#     cold inductor cache allowance              600 to 1200 s
#     TOTAL                              about 43 to 60 minutes
# The two sorted arms have never been run at all, so their high end assumes half
# again as much, which is mg10's and mg11's own rule for an unmeasured arm.  The
# cache allowance is real: this job gets its OWN inductor cache (see below), so
# the first arm at each launch shape compiles from cold; that cost lands in each
# arm's discarded cold pass.  --dry-run prints this same arithmetic before any
# GPU time is spent.
#     The walltime below is 02:00:00, about 2x the upper estimate.  That headroom
# is deliberate: three of the six arms have no prior reading at all, and if the
# sorted form is much SLOWER rather than faster the wall grows in the direction
# the estimate cannot see.
#
# IF THIS JOB MUST BE SPLIT OR CUT, use MG13_COUNTS, and trim WHOLE DEVICE
# COUNTS:
#     MG13_COUNTS=1        the question alone, three arms, about 25 minutes
# Never trim a single body out of a count: a count missing any of its three has
# no comparison block at all, and the torch scatter arm is where both verdict
# lines' denominator comes from.  MG13_ARMS drops single arms and exists for
# re-running one, not for trimming.  Trim n=4 before n=1 -- n=1 is the question
# and n=4 is the composed check on it.
#
# THE COMPILER CACHE is this job's own, torch_cache_mg13, and NOT mg9's or
# mg12's.  Sharing a cache across concurrent or successive jobs is a recorded way
# to get a corrupt artifact and a mystery crash.  Within this job the cache is
# shared across arms deliberately and safely: the two torch arms trace DIFFERENT
# graphs (one reduces with index_add_, the other with sort and segment-sum), so
# they key to different cache entries and cannot collide.
#
# DISK.  One shared sinogram at the 1024 cell, about 4 GB, plus two strided
# reconstruction samples per arm at about 1.2 MB each, all in $MG13_RESULTS
# during the run and deleted at the end unless MG13_KEEP_ARTIFACTS=1 is exported.
# The distances are computed before anything is removed.  The tree is a few
# hundred MB and is left in place, so a failed job can be examined and a re-run
# can skip the build.
#
# SCHEDULING.  This is a four-GPU block that can run to 02:00:00, so it cannot be
# submitted blind against the 02:00 and 03:00 nightlies: at the estimated 43 to
# 60 minutes it would fit between them, but the WALLTIME does not, and slurm
# schedules on the walltime.  Submit it with an explicit start time that clears
# both, for example
#     sbatch --begin=04:15 mg13_gautschi.sbatch
# or any hour that leaves two clear hours ahead of it.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch --begin=<time> mg13_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg13
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg13_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python

P3=/scratch/gautschi/buzzard/torch_p3
MG13_TREES=$P3/mg13_trees
HARNESS=$P3/mg13_sorted_probe.py
# The full hash of eaccf55, read with git rev-parse in the main checkout.  It
# sits on origin/greg_dev, not on the default branch; a plain clone fetches every
# branch, so the detached checkout below resolves it, and the read-back proves it.
PIN_HASH=eaccf55bc6b215d11e978802f4db0f6f8fdb2695
REPO_URL=https://github.com/cabouman/mbirtorch.git

# This job's OWN inductor cache (see the header).
export TORCHINDUCTOR_CACHE_DIR=$P3/torch_cache_mg13
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -c "import torch; print('torch', torch.__version__)" || exit 1

# ── WHAT MUST ALREADY BE HERE ────────────────────────────────────────────────
# One file.  There are no overlays in this job: the candidate is a patch the
# harness applies inside its own process, not a source change.
test -f "$HARNESS" || { echo "the harness is not at $HARNESS"; exit 1; }
echo "── the harness, as it arrived ──"
sha256sum "$HARNESS" || exit 1

# ── THE TREE, cloned here with a clean retry ─────────────────────────────────
# The tree is REMOVED between attempts.  A half-written clone left by a failed
# attempt must not be finished by the next one into something that is neither
# commit, and that is the whole reason the rm is inside the loop.
mkdir -p "$MG13_TREES"
TREE=$MG13_TREES/pinned
cloned=0
for attempt in 1 2 3; do
    rm -rf "$TREE"
    if git clone --quiet "$REPO_URL" "$TREE" \
       && git -C "$TREE" checkout --quiet --detach "$PIN_HASH"; then
        cloned=1
        break
    fi
    echo "clone attempt $attempt failed; removing the partial tree and retrying"
    sleep 20
done
test "$cloned" = 1 || { echo "could not clone $REPO_URL at $PIN_HASH in three attempts"; exit 1; }

# The hash is read back rather than trusted: a checkout that silently landed
# somewhere else would give every arm the wrong source.
GOT_HASH=$(git -C "$TREE" rev-parse HEAD)
test "$GOT_HASH" = "$PIN_HASH" || {
    echo "the tree is at $GOT_HASH and this job is pinned to $PIN_HASH"; exit 1; }
echo "$PIN_HASH" > "$MG13_TREES/pinned_hash.txt"
# .git goes now; the hash is already recorded in the log and in the file above.
rm -rf "$TREE/.git"
test -d "$TREE/mbirtorch" || { echo "no mbirtorch package in $TREE"; exit 1; }
echo "── the files the candidate and the substrate depend on ──"
sha256sum "$TREE/mbirtorch/horizontal_fan.py" \
          "$TREE/mbirtorch/parallel_beam.py" \
          "$TREE/mbirtorch/triton_parallel.py" \
          "$TREE/mbirtorch/tomography_model.py" \
          "$TREE/mbirtorch/kernel_availability.py"

export MG13_TREE=$TREE

# ── THE INTEGRITY GATE: every python file in the tree must PARSE ─────────────
# This node's filesystem client has intermittently returned truncated reads for
# files whose bytes are correct on disk.  A module an arm imports lazily would
# otherwise fail mid-job instead of here.  The harness is checked first, because
# the seam check below imports it.
"$TORCHPY" -m compileall -q "$HARNESS" || {
    echo "mg13_sorted_probe.py does not parse on this node; re-copy it"; exit 1; }
"$TORCHPY" -m compileall -q "$TREE" || {
    echo "a file in $TREE does not parse on this node; the build is not sound"; exit 1; }

# ── THE ENVIRONMENT, set explicitly, nothing inherited ───────────────────────
# THE COLUMN-GATHER SWITCH STAYS UNSET, here and in every child.  The column
# gather is the DEFAULT on this tree, so an absent variable IS the shipped
# configuration, and the four-device arms are supposed to measure the shipped
# forward.  The harness removes it from each child environment as well.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
# The device count is pinned on the MODEL with configure_devices, so a process
# wide count here would be a second, silent opinion.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, and the rows read
# that counter after every timed reconstruction.
unset MBIRTORCH_MEMORY_CALIBRATION
# The speed floors are bypassed by a configure_devices pin anyway.
unset MBIRTORCH_WIDENING_GUARD
# PYTHONPATH is cleared HERE and set per arm by the harness.  A value inherited
# from a submitting shell could put another mbirtorch ahead of the pinned tree,
# which is the one failure that would look like a clean result.  Each arm proves
# where its mbirtorch came from anyway.
unset PYTHONPATH
# The kill switch and the patch flag are set PER ARM by the harness, from the
# arm's own body, and each arm asserts the value it was given.  Anything stale
# here would be overwritten, but they are unset so the log shows them arriving
# from one place only.
unset MBIRTORCH_DISABLE_TRITON
unset MG13_PATCH_SORTED

# ── THE SEAM CHECK, one pass, all before any timed GPU work ──────────────────
# This is where mg13 differs from mg12.  mg12 checked that three TREES were the
# trees they were labelled as; here there is one tree and the treatment is a
# PATCH, so what has to be proved is that the patch point is still where the
# harness reaches for it and that the patched reduction computes the shipped
# one's values ON THIS TREE.  A patch that silently did not take would report a
# very tidy "no change".
MG13_TREE=$TREE PYTHONPATH=$TREE:$P3 "$TORCHPY" -c "
import inspect, os, sys
import numpy as np
import torch
import mbirtorch
from mbirtorch import horizontal_fan, kernel_availability, parallel_beam
from mbirtorch.projectors import Projectors
from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR, FORWARD_PIXEL_BATCH

root = os.path.realpath(os.environ['MG13_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(root + os.sep):
    sys.exit('mbirtorch imported from ' + got + ', NOT from ' + root + '.  An '
             'editable install of another tree is winning through a meta-path '
             'finder; check what else is running before fixing this with pip.')

# FACT 1: THE PATCH POINT.  The harness rebinds parallel_beam.fan_forward_batch,
# which works only because parallel_beam imports the name and the body resolves
# it out of parallel_beam's globals at call time.  If the module ever switched to
# calling it through horizontal_fan, or renamed it, the rebinding would take and
# change nothing.
if not hasattr(parallel_beam, 'fan_forward_batch'):
    sys.exit('parallel_beam has no fan_forward_batch attribute; the harness '
             'patches that name and would patch nothing.')
if parallel_beam.fan_forward_batch is not horizontal_fan.fan_forward_batch:
    sys.exit('parallel_beam.fan_forward_batch is not the function defined in '
             'horizontal_fan; this tree is not in the state the harness assumes.')
src = inspect.getsource(parallel_beam._parallel_forward_view_batch)
if 'fan_forward_batch(' not in src:
    sys.exit('the parallel forward body does not call fan_forward_batch by that '
             'name:\n' + src)

# FACT 2: THE REDUCTION API.  The candidate reduces with torch.segment_reduce
# when it is present and falls back to a cumsum run-difference otherwise; which
# one this environment gives is recorded before any arm runs.
print('torch.segment_reduce present:', hasattr(torch, 'segment_reduce'))

# FACT 3: THE PATCH TAKES, AND THE VALUES AGREE.  The harness owns the candidate,
# so it is imported here (PYTHONPATH carries both the tree and the run directory)
# and run against the shipped reduction on a small random case, through the real
# body, on this tree.  This is the check that would catch a contract change in
# the horizontal fan that left the signature intact.
import mg13_sorted_probe as probe

cell = (8, 24, 20)
model = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
model.set_params(no_warning=True, verbose=0)
args = model._view_batch_args()
idx = model.full_indices_device()
values = torch.rand((int(idx.shape[0]), int(model.get_params('recon_shape')[2])),
                    dtype=torch.float32, device=model.torch_device)
views = model.projector_functions._view_params_per_dev[0][:4]
shipped = parallel_beam._parallel_forward_view_batch(values, idx, views, **args)
record = probe.install_sorted_fan()
if parallel_beam.fan_forward_batch is not probe.fan_forward_batch_sorted:
    sys.exit('install_sorted_fan() did not take on this tree: the body would '
             'still call ' + repr(parallel_beam.fan_forward_batch))
before = probe.SORTED_STATE['calls']
candidate = parallel_beam._parallel_forward_view_batch(values, idx, views, **args)
if probe.SORTED_STATE['calls'] <= before:
    sys.exit('the body ran without entering the sorted reduction; the patch is '
             'installed but not on the path the body takes.')
# In float64 on the HOST: not every backend carries float64, and a check whose
# precision depended on where it ran would be a poor thing to hold a tolerance
# against.
a = candidate.detach().cpu().double()
b = shipped.detach().cpu().double()
rel = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))
if not (rel <= 1e-5):
    sys.exit('the sorted reduction disagrees with the shipped one by %.3e '
             'relative on this tree, over the 1e-5 the job holds it to.' % rel)
print('the sorted reduction reproduces the shipped one at %.3e relative, '
      'api=%s (resolved=%s)'
      % (rel, record['reduction_api'], probe.SORTED_STATE['reduction_api']))

# FACT 4: THE VIEW-CHUNK CEILING.  The harness derives the sorted form's memory
# bound from the library's own per-batch transient budget rather than a number of
# its own; that constant has to be there.
print('view batch transient budget:',
      int(Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES))

# FACT 5: THE KILL SWITCH IS NAMED WHAT THE HARNESS SETS.
if kernel_availability.DISABLE_ENV_VAR != 'MBIRTORCH_DISABLE_TRITON':
    sys.exit('the library kill switch is named %r; the harness sets '
             'MBIRTORCH_DISABLE_TRITON.' % (kernel_availability.DISABLE_ENV_VAR,))

# FACT 6: THE FOUR-DEVICE ARMS RUN THE SHIPPED FORWARD SHAPE.  With the switch
# absent, a live parallel model must resolve to the column gather.
if COLUMN_GATHER_ENV_VAR in os.environ:
    sys.exit('the column-gather switch is set in the environment of this check, '
             'so the check would not be about the default.')
if not model.column_gather_geometry or not model._column_gather_forward():
    sys.exit('parallel does NOT resolve to the column gather with the switch '
             'unset; the four-device arms would measure the banded walk.')
print('seam check passed: patch point present, patch takes, values agree, '
      'column gather is the default, shipped pixel batch=%d'
      % int(FORWARD_PIXEL_BATCH))
" || { echo "the tree failed its seam check; do not run the job against it"; exit 1; }

# ── THE KILL SWITCH, checked in BOTH positions on the real GPU ───────────────
# The two torch arms differ from the reference arm only by this variable, so a
# switch that did not bite would give three arms of the same body and a job that
# reported nothing while looking tidy.  Each position is checked in its own
# process because the library caches the probe result per process.
for want in 0 1; do
    MBIRTORCH_DISABLE_TRITON=$want PYTHONPATH=$TREE "$TORCHPY" -c "
import os, sys
from mbirtorch import kernel_availability
want_off = os.environ['MBIRTORCH_DISABLE_TRITON'] == '1'
usable, reason = kernel_availability.triton_available()
print('MBIRTORCH_DISABLE_TRITON=%s -> usable=%s (%s)'
      % (os.environ['MBIRTORCH_DISABLE_TRITON'], usable, reason))
if want_off and usable:
    sys.exit('the kill switch is set and the kernels are still usable; the two '
             'torch arms would run the Triton kernel and measure nothing about '
             'the reduction.')
if (not want_off) and (not usable):
    sys.exit('the kernels are NOT usable on this node with the switch off (%s); '
             'the production reference arm would silently be a third torch arm.'
             % reason)
" || { echo "the Triton kill switch does not behave as the arms require"; exit 1; }
done

# MG13_COUNTS and MG13_ARMS are deliberately NOT unset: they are the knobs the
# header tells a reader to export from the SUBMITTING SHELL, and slurm's default
# --export=ALL is what carries them in.  Both are echoed so the log records what
# the job was asked for.
echo "MG13_COUNTS=${MG13_COUNTS:-<unset: both device counts>}"
echo "MG13_ARMS=${MG13_ARMS:-<unset: all six arms>}"
# The shaping knobs ARE unset, so nothing stale in a submitting shell can
# silently change what is measured or how long each arm takes.
unset MG13_ITERATIONS
unset MG13_WARM_REPEATS
unset MG13_MAX_EVENT_PAIRS
unset MG13_KEEP_ARTIFACTS
unset MG13_SMOKE
unset MG13_DEVICE
unset MG13_NO_BENCH
# The candidate's two shaping knobs are unset too, so the measured form is the
# one the harness derives and documents: the view chunk comes from the library's
# transient budget, and the sort is the cheaper unstable one.
unset MG13_SORT_VIEW_CHUNK
unset MG13_SORT_STABLE

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the harness, which is the
# convention the rows are collected under.
export MG13_RESULTS=$P3
cd "$P3" || exit 1
# The arm plan first, so the log opens with the seven arms, the tree root and the
# wall arithmetic -- and a mis-set environment variable is caught before any GPU
# work.
"$TORCHPY" -u mg13_sorted_probe.py --dry-run || exit 1
"$TORCHPY" -u mg13_sorted_probe.py
echo 'MG13 DONE'
