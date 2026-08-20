#!/bin/bash
# mg49: does a denoise ever get faster on more devices, and where does the
# sharded sweep's time go?
#
# WHY THIS JOB EXISTS.  The QGGMRFDenoiser is held to one device by two
# sentinel rows in the shipped widening-floors table.  A sentinel records a
# device count with NO admission size: splitting lost at every size ever
# probed, so the automatic path never widens a denoise for speed and only
# the memory preflight ever widens one.  These are the warm walls those
# sentinels rest on, measured 2026-08-20 by the full floors refresh (mg48,
# job 15399595) and read from that job's log.  The ratio is warm(n1) over
# warm(n), and both denoiser rows are taken against ONE device:
#
#   (512, 448, 384)     n1 0.233 s  n2 0.394 s  n4 0.506 s   0.592x  0.461x
#   (768, 672, 576)     n1 0.664 s  n2 1.018 s  n4 1.253 s   0.652x  0.530x
#   (1024, 1008, 992)   n1 2.191 s  n2 3.269 s  n4 3.574 s   0.670x  0.613x
#
# The loss SHRINKS as the problem grows, so an admission size may sit above
# the largest cell anyone has probed.  This job extends the ladder to the
# 1280-, 1536- and 1664-class and re-anchors at the 1024-class.
#
# WHY THE LADDER STOPS AT THE 1664-CLASS, AND WHY THAT MAKES THE ANSWER
# FINAL.  The speed sentinel only governs where one device can still HOLD
# the problem; above that size the memory preflight widens a denoise
# whatever the sentinel says.  The denoiser's own memory ledger at one
# device (a closed form -- it queries no device and allocates nothing) puts
# that boundary just above the 1664-class: 59.5 GB peak there, 68.4 GB with
# the preflight's 15 percent margin, which an 80 GB H100 still holds, while
# the 1792-class needs 85.6 GB, which it does not.  So the 1664-class is
# the last size at which the speed question has any practical effect.  With
# it in the ladder, a run that finds no admission has closed the question
# across the sentinel's whole domain rather than pushing an open edge
# upward.  The script prints the same ledger table read from the tree under
# test, so nothing in the output depends on these numbers.
#
# THE SECOND QUESTION.  The single-device and multi-device denoisers are
# DIFFERENT implementations: at one device the whole per-subset update is
# one compiled call, and above one device the same update becomes two
# per-device fan-outs, four scalar reductions on the lead device, a
# step-size broadcast back and a halo exchange per pass.  Five arms at the
# 1024-class time every seam of that sequence on a host clock and a device
# clock and report how the sharded sweep's time divides.  Two of those five
# run the identical protocol with NO wrappers: they are the
# instrument-overhead control.
#
# THE INSTRUMENT IS IMPORTED, NOT COPIED.  mg49 imports its paired-clock
# probe from mg44_component_split.py, which must sit beside it in this
# directory; the preflight below checks that before any GPU work.
#
# THE PROTOCOL IS THE FLOORS REFRESH'S, deliberately: the same model, the
# same explicit configure_devices placement the recorded denoiser rows were
# measured under, the same seeded phantom-plus-noise input, the seed reset
# before every call, the same three-iteration denoise, one cold pass then
# three warm repeats.  The sentinels were measured by that tool, and a
# different protocol would measure a different thing.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that installs
# into the same environment.  Submit it with --dependency=afterany:<job id>
# behind any such job already queued.
#
# DISK AND HOST MEMORY.  The 1024-class input is REUSED from the mg48
# floors directory, so the arms at that cell denoise the same bytes the
# recorded walls were measured on.  The three new cells are staged here and
# write about 40 GB to scratch in all (8.1, 14.0 and 17.9 GB).  Building
# those noisy phantoms costs a few minutes of host time apiece, and the
# script builds each one a slab of rows at a time rather than in one
# expression: numpy's randn returns float64, so the whole-volume form would
# hold roughly 145 GB of host arrays at the 1664-class for an answer that
# is 18 GB.  The slabs draw from one seeded generator in C order, so the
# bytes are identical to the whole-volume form.
#
# WALL ESTIMATE.  Twenty-one arms and four staging jobs, each in its own
# process.  The compute is small -- a 1664-class denoise is under ten
# seconds warm at one device -- and the wall is dominated by the per-arm
# cost of loading and hashing an 8 to 18 GB input from scratch, plus the
# three phantom builds.  Roughly ninety minutes to two hours; two and a
# half are requested for margin.  NOTE: the plan this was sized for asked
# for 01:30:00, before the 1664-class cell (three arms and 18 GB of
# staging at the most expensive size) was added.  Lower it back to
# 01:30:00 if the margin is not wanted -- the jsonl is flushed row by row,
# so a job cut short still yields everything it completed, and MG49_ARMS
# re-runs the rest.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:30:00
#SBATCH -J torch_mg49
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg49_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L

# ── the mbirtorch preflight ──────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this
# one and leaves a torn .pth.  Chain instead, with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The tree witnesses.  These walls describe the padded, channel-sorted,
# recompile-remedied tree, which is the tree the mg48 walls this job
# anchors against were measured on.  The denoiser runs no projector, so
# neither witness touches what this job measures directly; they are what
# lets a reader tell from the log which tree produced these numbers.
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

# The kernels are the shipped configuration these readings describe.  The
# denoiser has no projectors, so neither of these changes what it does;
# they keep the environment the one every other run in this campaign used.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# Every arm places its model with an EXPLICIT configure_devices call, which
# is the protocol the recorded denoiser rows were measured under.  The
# count pin is a second mechanism claiming the same decision, so it is
# unset here and popped from every job environment by the script.
unset MBIRTORCH_NUM_DEVICES
# This is a TIMING probe.  The calibration mode does extra work at the
# settle and at the end of every call, so an arm that inherited it would
# not time what the other arms timed.
unset MBIRTORCH_MEMORY_CALIBRATION
# An explicit layout bypasses the speed floors anyway; unset so no
# inherited value can change what an arm does.
unset MBIRTORCH_WIDENING_GUARD
# The recompile floor stays at the shipped value: an override here would
# describe a different tree than the witness above just asserted.
unset MBIRTORCH_RECOMPILE_LIMIT
unset PYTHONPATH
unset MG49_SMOKE
unset MG49_DRY
unset MG49_ARMS

export MG49_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg49_denoiser
# The staged inputs come from the mg48 floors refresh's own results
# directory.  The 1024-class denoiser input is already there, so the arms
# at that cell denoise the same bytes the recorded walls were measured on;
# the three new cells are written there as well, so a later floors refresh
# reuses them instead of rebuilding 40 GB.
export MG49_SINO_DIR=/scratch/gautschi/buzzard/torch_p3/results/mg48_floors
mkdir -p "$MG49_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# mg49 imports its instrument from mg44_component_split, which must sit
# beside it.  Checked here rather than discovered by the first worker.
test -f mg49_denoiser_split.py || { echo "mg49_denoiser_split.py is not in $PWD"; exit 1; }
test -f mg44_component_split.py || { echo "mg44_component_split.py must sit beside mg49_denoiser_split.py in $PWD: mg49 imports its paired-clock instrument from it"; exit 1; }

# The plan first, so the log opens with the twenty-five jobs in run order
# and a mis-set variable is caught before any GPU work.  The dry plan
# imports no torch.
export MG49_DRY=1
"$TORCHPY" -u mg49_denoiser_split.py || exit 1
unset MG49_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what
# it measured; the ladder verdicts, the ledger table, the attribution
# split, the finiteness check and the overhead control are all in this log.
"$TORCHPY" -u mg49_denoiser_split.py
rc=$?
echo "MG49 DONE rc=$rc"
exit $rc
