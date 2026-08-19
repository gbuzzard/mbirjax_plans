#!/bin/bash
# mg42a: where a reconstruction's real peak sits, and which region carries it.
#
# WHY THIS JOB EXISTS.  The design note in the plans repository asks three
# measurement questions about the library's memory ledger, and this one job
# answers all three.  The ledger predicts each device's peak bytes before a
# reconstruction allocates anything; the calibration mode compares that
# prediction against one measured number per device.  One number says the
# model is high somewhere without saying where.  Every arm here runs with
# the calibration mode on AND with seven library seams wrapped, so each arm
# ends with the whole-run peak and a per-region attribution of what carried
# it.
#
#   question A, at the 1024 class on two devices: what the back-projection
#   worker region actually holds, against the ledger's back batch and back
#   output charges.
#   question B, at the 512 class on one, two and three devices: which term
#   over-charges as the count rises.
#   question C, at the 1024 class on four devices in a fresh process:
#   whether device 0 transiently holds more than the one-device arm's own
#   peak.
#
# THE INSTRUMENT.  The per-device peak counter is a monotone high-water
# mark and this probe never resets it, so the maximum over its samples is
# the whole-run peak by construction rather than by agreement.  The
# calibration mode's one reset lands at the settle, before any wrapped
# region holds anything, so the mode and the probe read the same counter
# over the same scope.  The script sets MBIRTORCH_MEMORY_CALIBRATION per
# subprocess: '1' for an arm, empty for a staging job, which reads no
# counter and must not own one.  This shell must therefore leave that
# variable alone -- see the unset block below, which deliberately does not
# name it.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that installs
# into the same environment.  Submit it with --dependency=afterany:<job id>
# behind any such job already queued.
#
# WALL ESTIMATE.  Six 3-iteration 512-class arms, two 1-iteration
# 1024-class arms and two 3-iteration 1024-class arms, each in its own
# process paying its own compiles, come to roughly an hour.  Two hours are
# requested for margin.
#
# DISK.  The four staged sinograms are REUSED from the reference-timing
# run's directory: about 4.1 GB each at the 1024 class and 0.34 GB each at
# the 512 class.  Nothing new is written there unless one is missing.  This
# job writes one jsonl of rows.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg42a
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg42a_%j.log
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

# The checkout must carry both kernel changes the campaign's rows describe:
# the width padding (504 -> 512 is the production band it exists for) and
# the channel-sorted parallel forward, whose enable switch only exists on
# the sorted tree.  Without them this job would attribute the peaks of a
# different implementation than the ledger is written against.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
from mbirtorch.triton_parallel import sorted_forward_enabled
msg = 'the tree is missing a kernel change these readings describe'
assert padded_kernel_width(504) == 512, msg
assert sorted_forward_enabled(), msg
" || exit 1

# The kernels are the shipped configuration these readings describe, and
# the sorted route must take its shipped default.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# The script pins every arm through MBIRTORCH_NUM_DEVICES in its own
# subprocess; a value inherited from this shell would reach a job that
# asked for something else.
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
# MBIRTORCH_MEMORY_CALIBRATION is deliberately NOT unset and NOT exported
# here.  The script pops whatever it inherits and sets it per subprocess:
# '1' for an arm, empty for a staging job.  Setting it in this shell would
# only invite a reader to think the shell owns it.
unset MG42_SMOKE
unset MG42_DRY
unset MG42_ARMS
# Left unset so the staged sinograms come from the reference-timing run's
# directory, which is the script's own default.
unset MG42_SINO_DIR

export MG42_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg42a_ledger
mkdir -p "$MG42_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The plan first, so the log opens with the fourteen jobs in run order and
# a mis-set variable is caught before any GPU work.
export MG42_DRY=1
"$TORCHPY" -u mg42a_ledger_probe.py || exit 1
unset MG42_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what
# they measured; the ratios, the region attributions and the question-C
# comparison are in this log.
"$TORCHPY" -u mg42a_ledger_probe.py
rc=$?
echo "MG42A DONE rc=$rc"
exit $rc
