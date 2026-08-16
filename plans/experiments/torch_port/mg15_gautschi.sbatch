#!/bin/bash
# mg15: THE CLUSTER GATE FOR THE SHARDING-PAD REMOVAL, on four H100s.  The full
# statement of what is measured and why is in mg15_p5_gate.py; the essentials
# are repeated here because a reader of this file may never open that one.
#
# WHY THIS JOB EXISTS.  mbirtorch used to shard by padding: an axis split across
# devices was first rounded UP to a multiple of the device count, every device
# got an equal block, and the padding was carried through the arithmetic and
# thrown away at the end.  That is gone.  A sharded axis is now split into
# contiguous blocks whose lengths differ by at most one, longer blocks first, so
# no device holds padding and the blocks are no longer all the same length.
#
# The local test suite already covers the split and the values it produces on
# virtual CPU devices at counts that do not divide the axes.  This job adds what
# a laptop cannot show, and nothing else.
#
# THE THREE LEGS.
#   1. VALUES.  Real multi-GPU runs at counts that do not divide the sharded
#      axes, against the single-device reference.  Two parts: one forward
#      projection of a staged phantom on the arm's own layout against that
#      phantom projected on one device (gate 1e-4, and the part that actually
#      resolves the split -- it reproduces to ~1e-7 and a misplaced block lands
#      near one), and the 3-iteration reconstruction against the n=1 arm (gate
#      1e-2, a gross-error backstop; multi-device VCD is not reproducible to
#      better than ~2e-3 at these sizes for reasons that have nothing to do with
#      the padding, which was measured before this job was written).
#   2. MEMORY.  Per-device peak memory against the library's memory ledger.  The
#      ledger now prices the block a device actually owns rather than a padded
#      block that is the same everywhere.  Its one hard rule is that it may
#      over-estimate but must never UNDER-estimate, because too small means a
#      doomed run starts and then dies inside the allocator.  A modeled peak
#      below a measured peak on any device FAILS this job.  An over-estimate
#      beyond the declared band is reported as a warning and does not fail.
#   3. ONE LEDGER ROW.  The measured arm 'ma512_n4' in tests/test_memory_ledger.py
#      MEASURED_ARMS was recorded on the PADDED split: 510 recon slices were
#      rounded to 512 and cut into four blocks of 128, where they are now cut
#      into 128, 128, 127, 127.  The job re-measures that arm under mg8's exact
#      protocol and prints the replacement line ready to paste.
#
# THE NON-DIVIDING ARITHMETIC, which is the point of the device counts chosen:
#     cone       512 views % 3 = 2   and  448 recon slices % 3 = 1
#     parallel   512 views % 3 = 2   and  448 recon slices % 3 = 1
#     multiaxis  512 views % 4 = 0   and  510 recon slices % 4 = 2
# Parallel beam is the row-aligned geometry, so at n=3 its detector-row axis
# shards unevenly along with its slices -- the path that used to carry the
# detector-row pad, and the one with no counterpart in the other two families.
# The multiaxis count is 4 rather than 3 because the ledger row it replaces is a
# four-device row.
#
# THE EXIT CODE IS THE VERDICT.  A follow-on floors-measurement job is chained
# to this one with --dependency=afterok, so this script must not swallow the
# python exit status: it captures it and exits with it.  A failing gate stops
# the follow-on rather than letting it measure on top of a split that does not
# reconstruct correctly.  Submit the chain as:
#     cd /scratch/gautschi/buzzard/torch_p3
#     jid=$(sbatch --parsable mg15_gautschi.sbatch)
#     sbatch --dependency=afterok:$jid <the floors job>
#
# WALL ESTIMATE, and its arithmetic.  Six measured arms plus three untimed
# generators, all at the 512-class cell (512, 448, 384).
#   * The reference is the measured warm 3-iteration reconstruction on one H100
#     at the 512-class parallel cell from the projector batching work: 4.87 s
#     with pure torch bodies, and less with the hand-written kernels cone and
#     parallel bind.  The reconstruction itself is seconds.
#   * What actually costs time is torch compilation, once per device on the
#     first (and here only) pass, which is why mg8 budgeted ~4 minutes for three
#     arms at this same cell.  Six arms here: ~8-10 minutes.
#   * Each arm also forward-projects the staged phantom once for the value leg:
#     seconds per arm, plus its own compilation on the multi-device layouts.
#     Call it ~3 minutes over the six.
#   * Three generators (one forward projection each) plus md5 work over ~1.8 GiB
#     of staged phantom and sinogram, each file re-verified by its two
#     consumers, plus ~1.6 GiB of reconstruction volumes written and read back
#     by the value gate: ~5-7 minutes.
#   * Nine subprocess starts, each paying a torch import: ~2 minutes.
# Total ~20-25 minutes, upper estimate ~35.  The walltime below is 01:30:00,
# comfortably over twice the upper estimate, which also covers a cold
# compilation cache and a shared node.  Being well under an hour, it clears the
# 02:00 and 03:00 nightlies with room on either side.
#
# DISK.  About 3.4 GiB in $MG15_RESULTS at peak (1.0 GiB of sinogram, 0.8 GiB of
# phantom, 1.6 GiB of reconstruction volumes), deleted at the end unless
# MG15_KEEP_ARTIFACTS=1 is exported.  The jsonl is always kept.
#
# IF THIS JOB MUST BE SPLIT, split it by FAMILY and by nothing else:
#     MG15_FAMILIES=cone        2 arms
#     MG15_FAMILIES=parallel    2 arms
#     MG15_FAMILIES=multiaxis   2 arms
# Every value comparison is against the n=1 arm of the SAME family in the SAME
# job, so the device-count axis may never cross a job boundary.  There is
# deliberately no environment knob for the counts.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg15_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 01:30:00
#SBATCH -J torch_mg15
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg15_%j.log
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
# THE ASSUMPTION THIS WHOLE JOB RESTS ON, checked before any GPU time is spent:
# that the checkout staged in mbirtorch_src is the one with the padding REMOVED.
# An older checkout would import cleanly, run every arm, and report a pass on
# code that still pads -- a wrong answer rather than a missing one.  512 over
# three devices must come back as 171/171/170, not as three equal blocks.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, axis=0, axis_len=512).shard_ranges()]
assert blocks == [171, 171, 170], blocks
print('unpadded split confirmed:', blocks)" || { echo "this mbirtorch still pads its shards: stage the padding-removal checkout into mbirtorch_src"; exit 1; }

# Neither switch below changes what the arms compute, but both are set
# explicitly rather than left to whatever the submitting shell happened to hold:
# the measurement has to be taken in the shipped production configuration and
# not in a variant of it.  The script re-asserts both inside every arm.
export MBIRTORCH_DISABLE_TRITON=0
# Each arm pins its own device count, which the SCRIPT pops and then sets per
# subprocess.  A value inherited from here would pin the whole job to one count
# without saying so.  Each arm then asserts the device list it actually got.
unset MBIRTORCH_NUM_DEVICES
# THE ONE JOB CLASS THAT OWNS THIS FLAG.  The calibration mode RESETS
# torch.cuda.max_memory_allocated so it can read it at the end of the
# reconstruction, which means it takes ownership of a counter any other measured
# job would be reading.  It is exported here, as 1, because the memory leg needs
# it -- every other job in this family unsets it.  The script still pops and
# re-sets it per subprocess, so the three generators run WITHOUT it and never
# reset a counter they are not reading.
export MBIRTORCH_MEMORY_CALIBRATION=1
# The re-run knobs, named here only so a narrowed re-run needs no edit to the
# file.  Split by FAMILY, which keeps both arms of a comparison together.
unset MG15_FAMILIES
unset MG15_ITERATIONS
unset MG15_KEEP_ARTIFACTS
unset MG15_SMOKE
unset MG15_DEVICE

export MG15_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG15_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the six arms and the gates they will
# be judged against, and a mis-set environment variable is caught before any GPU
# work.
"$TORCHPY" -u mg15_p5_gate.py --dry-run || exit 1
# The gate itself.  Its exit code is the job's exit code, because the follow-on
# job is chained with --dependency=afterok: 0 means values, memory floor and arm
# checks all passed, 2 means at least one did not.
"$TORCHPY" -u mg15_p5_gate.py
rc=$?
echo "MG15 DONE rc=$rc"
exit $rc
