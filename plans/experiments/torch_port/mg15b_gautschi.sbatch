#!/bin/bash
# mg15b: ADJUDICATING THE ONE FAILING READING FROM mg15, on four H100s.  The
# full statement is in mg15b_multiaxis_probe.py; the essentials are repeated
# here because a reader of this file may never open that one.
#
# WHAT HAPPENED.  mg15 (slurm 15304566, h016, 2026-08-16) gated the removal of
# the sharding pad and passed every leg but one.  Its forward projector leg was
# clean on genuinely uneven layouts (cone_n3 5.696e-07, parallel_n3 5.420e-07,
# multiaxis_n4 1.585e-05, all against a 1e-4 gate), its ledger floor held on
# every device (lowest ratio 1.016), and the ma512_n4 row was measured.  The one
# failure was the multiaxis four-device RECONSTRUCTION backstop: 2.969e-02
# against a 1e-2 gate, with nrmse 7.624e-04, where cone_n3 read 6.861e-03 and
# parallel_n3 1.817e-03 and both passed.
#
# WHY THAT IS NOT YET AN ANSWER.  mg15's docstring predicted that its max-norm
# statistic would grow with problem size, and multiaxis was already the noisiest
# family, so trajectory divergence is the natural suspicion.  But mg15's fine
# instrument is the FORWARD projector, and a forward projection cannot see the
# adjoint.  A defect living only in the sharded BACK projection -- the direction
# that scatters into the sharded slice axis, and the direction whose blocks the
# pad removal changed -- would leave the forward leg clean and surface only
# after the reconstruction folded it in.  This job settles it with instruments.
#
# THE FOUR INSTRUMENTS.
#   1. BACK PROJECTION at one device against four, gated at 1e-4.  The adjoint
#      instrument mg15 lacked.  One linear operator, no iteration: a misplaced
#      block lands near one.  If this fails, the answer is a split defect.
#   2. SAME-COUNT NOISE FLOOR.  Two seeded 3-iteration reconstructions in ONE
#      process on ONE model at four devices, second against first.  Same count,
#      same partitions, same everything, so the reading is pure run-to-run
#      nondeterminism -- the scale that 2.969e-02 has to be read against.
#   3. ITERATION DECAY.  One-against-four at 3 iterations and at 10.  The
#      library documents cross-count differences falling from 6.1e-3 at 3 to
#      8.8e-4 at 10 (docs/source/usr_multi_gpu.rst).  Trajectory divergence
#      decays; a systematic split error is re-injected every iteration.
#   4. BOUNDARY CONCENTRATION, report only, at both iteration counts.  Read for
#      CHANGE and not for shape: every cross-count difference is seeded at the
#      seams, so clustering there is expected even when the split is correct --
#      the probe's own CPU smoke shows the boundary slices ranked #0 and #3 of
#      27 on a known-good split.  What discriminates is whether the boundary
#      peak decays with the whole difference or holds while the interior falls.
#
# VALUES ONLY, WHICH MAKES THIS SIMPLER THAN mg15.  No ledger, no calibration
# mode, no automatic device branch: every arm pins with an explicit
# configure_devices device list, which exists from construction.  mg15 had to
# pin through MBIRTORCH_NUM_DEVICES so its memory leg would run on the branch
# that consumes the ledger, and had to order its projector leg after the
# reconstruction because that pin only acts through the device policy and the
# projectors never call it.  Here that trap cannot arise.  The script POPS
# MBIRTORCH_MEMORY_CALIBRATION from every arm rather than setting it: this job
# reads no memory and must not take ownership of a peak counter.
#
# EXIT CODE IS INSTRUMENT HEALTH, NOT THE VERDICT.  0 when every arm ran and
# every comparison computed, 2 when an instrument failed to produce a reading.
# The adjudication is a judgement across four readings that point at each other,
# and is read by a human from the printed summary block.  Nothing is chained to
# this job.
#
# WALL ESTIMATE, and its arithmetic.  One generator plus six arms, all on
# mg15's multiaxis cell (512, 448, 384) -> recon (384, 384, 510).
#   * mg8 measured this cell at seconds per reconstruction pass, dominated by
#     torch compilation -- it budgeted ~4 minutes for three arms here.
#   * two backproject arms: a single back projection each, seconds plus one
#     compilation per device: ~3 minutes.
#   * two repeat arms: two 3-iteration passes each, but the compilation is paid
#     once per process, so the second pass is nearly free: ~5 minutes.
#   * two recon_long arms at 10 iterations: roughly 3x the iteration work of a
#     3-iteration pass on top of the same one-time compilation: ~7 minutes.
#   * one generator (a forward projection) plus md5 work over ~0.6 GiB of
#     staged phantom and sinogram and ~1.8 GiB of staged volumes, each re-read
#     by the driver's comparisons: ~4 minutes.
#   * seven subprocess starts, each paying a torch import: ~2 minutes.
# Total ~20-25 minutes.  The walltime below is 00:45:00, comfortably under an
# hour so it clears the 02:00 and 03:00 nightlies.
#
# DISK.  About 2.4 GiB in $MG15B_RESULTS at peak: 0.29 GiB phantom, 0.34 GiB
# sinogram, two back projections and four reconstruction volumes at 0.29 GiB
# each.  Deleted at the end unless MG15B_KEEP_ARTIFACTS=1 is exported; the jsonl
# is always kept.  Export MG15B_KEEP_ARTIFACTS=1 if the difference volumes may
# need re-examining, because regenerating them costs the whole job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg15b_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 00:45:00
#SBATCH -J torch_mg15b
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg15b_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1
# THE ASSUMPTION THIS JOB RESTS ON, checked before any GPU time is spent: that
# the checkout staged in mbirtorch_src is the same unpadded one mg15 measured.
# An older checkout would import cleanly, run every arm, and adjudicate a
# reading that came from different code -- a wrong answer rather than a missing
# one.  512 over three devices must come back as 171/171/170.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, axis=0, axis_len=512).shard_ranges()]
assert blocks == [171, 171, 170], blocks
print('unpadded split confirmed:', blocks)" || { echo "this mbirtorch still pads its shards: stage the same checkout mg15 measured into mbirtorch_src"; exit 1; }

# The shipped configuration, set explicitly rather than left to whatever the
# submitting shell held: an adjudication has to be taken in the same
# environment as the reading it adjudicates.  The script re-asserts it per arm.
export MBIRTORCH_DISABLE_TRITON=0
# This job measures VALUES ONLY.  It reads no memory, so it must not own the
# peak counter -- the calibration mode RESETS torch.cuda.max_memory_allocated,
# and a job that resets a counter it never reads is taking a resource from
# whatever else is measuring.  Unset here and popped again per subprocess.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins with an explicit configure_devices list, so an inherited
# count pin would describe a mechanism this job does not use.  The script pops
# it per subprocess as well.
unset MBIRTORCH_NUM_DEVICES
unset MG15B_SMOKE
unset MG15B_DEVICE
unset MG15B_KEEP_ARTIFACTS

export MG15B_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG15B_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the seven subprocesses and the
# reading being adjudicated, and a mis-set environment variable is caught before
# any GPU work.
"$TORCHPY" -u mg15b_multiaxis_probe.py --dry-run || exit 1
# The probe.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the finding is in the summary block in this log.
"$TORCHPY" -u mg15b_multiaxis_probe.py
rc=$?
echo "MG15B DONE rc=$rc"
exit $rc
