#!/bin/bash
# mg44: which component of a reconstruction carries the two-device loss.
#
# WHY THIS JOB EXISTS.  The multiaxis geometry reconstructs SLOWER on two
# devices at some problem sizes and faster at others, and no recorded row
# says which part of the reconstruction the difference lives in.  These
# are the warm walls this job is measured against, measured 2026-08-18 by
# the floors refresh (mg26, job 15342578) and read on 2026-08-19 from
# /scratch/gautschi/buzzard/torch_p3/results/mg26_floors/_out_arm_*.json:
#
#   multiaxis   (512, 448, 384)     n1  11.4 s   n2  32.6 s   0.35x
#   multiaxis   (768, 672, 576)     n1  56.3 s   n2  38.5 s   1.46x
#   multiaxis  (1024, 1008, 992)    n1 309.9 s   n2 388.8 s   0.80x
#   translation (256, 1900, 3000)   n1  12.6 s   n2  14.2 s   0.89x
#
# The ratio is warm(n1) over warm(n2), so above 1.00 means two devices are
# faster.  This job times every component call of the reconstruction at one
# and two devices and reports which component's warm time carries the
# difference, and how that moves from the 512 class to the 1024 class.
#
# THE INSTRUMENT.  Around each named library seam the script records both a
# host clock and a pair of CUDA timing events on each relevant device's
# default stream.  The host clock says how long the call took to ENQUEUE,
# which is all a host timer can say about a loop with no host
# synchronization in it; the events say how long that device's stream spent
# between the two markers.  Nothing is synchronized during a
# reconstruction: the events are recorded during the run and read
# afterwards, on the synchronize the timing protocol already does.  A
# synchronize inside the run would change the very overlap being measured.
#
# THE CONTROLS.  Two of the ten arms run the identical protocol with NO
# wrappers installed.  Their warm medians beside the wrapped arms' say what
# the instrument itself costs; a ratio near 1.00 means the wrapped numbers
# describe the library.
#
# THE PROTOCOL IS THE FLOORS REFRESH'S, deliberately: same model
# construction, same seed reset before every call, same weights, same
# three-iteration call, same timing envelope, one cold pass then three warm
# repeats.  The anomaly was measured by that tool, and a different protocol
# would measure a different thing.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment,
# so it must not run at the same time as another torch_p3 job that installs
# into the same environment.  Submit it with --dependency=afterany:<job id>
# behind any such job already queued.
#
# WALL ESTIMATE.  Four staging jobs that only re-hash files already on
# disk, eight wrapped timed arms and two control arms, each in its own
# process paying its own compiles.  The 1024-class pair is about
# three-quarters of it.  Roughly seventy minutes in all; two and a half
# hours are requested for margin.
#
# DISK.  The four staged sinograms are REUSED from the floors refresh's
# directory and nothing new is written there unless one is missing.  This
# job writes one jsonl of rows.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=2
#SBATCH -t 02:30:00
#SBATCH -J torch_mg44
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg44_%j.log
# No --mem: the ai partition allocates memory with the GPUs.
# Two GPUs and 28 cores: the same cores-per-GPU this campaign's four-GPU
# jobs ask for, halved with the GPU count.

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
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 2, torch.cuda.device_count()" || exit 1

# The checkout must carry the width padding (504 -> 512 is the production
# band it exists for).  Without it this job would split the time of a
# different implementation than the recorded walls describe.  The parallel
# forward's sorted-route witness that the memory jobs assert is NOT checked
# here: neither geometry this job measures goes through that route.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
assert padded_kernel_width(504) == 512, \
    'the tree is missing the width padding these readings describe'
" || exit 1

# The kernels are the shipped configuration these readings describe.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# The script pins every arm through MBIRTORCH_NUM_DEVICES in its own
# subprocess; a value inherited from this shell would reach a job that
# asked for something else.
unset MBIRTORCH_NUM_DEVICES
# This is a TIMING probe.  The calibration mode does extra work at the
# settle and at the end of every reconstruction, so an arm that inherited
# it would not time what the other arms timed.  The script pops it from
# every subprocess environment and never sets it; it is unset here as well
# so nothing in this shell suggests otherwise.
unset MBIRTORCH_MEMORY_CALIBRATION
# The device-count pin bypasses the widening guard anyway; unset so no
# inherited value can change what an arm does.
unset MBIRTORCH_WIDENING_GUARD
unset PYTHONPATH
unset MG44_SMOKE
unset MG44_DRY
unset MG44_ARMS

export MG44_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg44_component
# The staged sinograms come from the floors refresh's own results
# directory: these arms reconstruct the same bytes the recorded walls were
# measured on, and rebuilding them would cost an hour and, because the
# forward kernel's atomics are not bit-exact across runs, would also change
# what is being reconstructed.
export MG44_SINO_DIR=/scratch/gautschi/buzzard/torch_p3/results/mg26_floors
mkdir -p "$MG44_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The plan first, so the log opens with the fourteen jobs in run order and
# a mis-set variable is caught before any GPU work.
export MG44_DRY=1
"$TORCHPY" -u mg44_component_split.py || exit 1
unset MG44_DRY

# The run.  Its exit code reports whether the INSTRUMENT worked, not what
# it measured; the component split, the wall drifts, the host-bound list
# and the overhead check are in this log.
"$TORCHPY" -u mg44_component_split.py
rc=$?
echo "MG44 DONE rc=$rc"
exit $rc
