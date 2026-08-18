#!/bin/bash
# mg21: WHERE THE CONE BACK PROJECTION SPENDS ITS TIME AT THE 2048 CELL, and why
# it spends more of it on four H100s than on three.  The full statement is in
# mg21_back_attrib.py; the essentials are repeated here because a reader of this
# file may never open that one.
#
# WHERE THIS SITS.  mg19 bracketed every projection call of the first
# 2048-class reconstructions (2026-08-17).  Its rows say the cone back
# projection's busy time ROSE with the device count: 137 s on three devices and
# 228 s on four, for the same three iterations of the same problem.  Adding a
# device made the back projection cost more, which is the opposite of what
# sharding is for.  The recorded hypothesis is that per-band-call costs which do
# not shrink with the band are what grow.  Nothing in mg19's rows tests that.
#
# WHAT THIS JOB DOES.  It instruments the inside of the sharded cone back
# projection at six places and splits each call's time into named parts, so a
# remedy design can rest on measured shares instead of the hypothesis:
#   * the pass, one call of the sharded funnel;
#   * one view-range call per (device, band);
#   * the two eager builders inside the projection body;
#   * the body itself;
#   * the Triton launch, through a proxy that also records the launch grid;
#   * the cross-device sum that closes each band.
# From those, per body call: builders, kernel, and a residual that is the body
# minus the other two.  The residual is mostly one copy -- the body writes a
# channel-major copy of its whole detector-row block before every launch, and
# that copy is sized by the DETECTOR, so it does not shrink when the band does.
# One microbench in the four-device arm prices that copy on its own.
#
# THE CELL, mg19's cone cell so the two runs describe the same problem:
#   sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
#   recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask
#
# THE ARMS, two of them, each in its own process: n3 and n4, pinned through
# MBIRTORCH_NUM_DEVICES.  Each arm runs six variants, each with one warmup
# repeat and two timed repeats:
#   full_p1     every pixel, coeff_power 1 -- the reference call
#   full_p2     every pixel, coeff_power 2 -- the Hessian call
#   sub4, sub16, sub64   one subset at each granularity the shipped
#                        three-iteration schedule visits
#   band_half   every pixel with the slice band halved
#
# band_half IS THE DISCRIMINATOR.  Halving the band doubles the number of band
# calls and halves the voxel work in each one.  A cost that scales with the band
# is then unchanged in total; a cost charged once per band call is doubled.  So
# if per-band-call costs drive the growth, band_half costs roughly twice
# full_p1; if the kernel's own work dominates, band_half barely moves.
#
# NOTHING IS FLIPPED HERE.  No library file is touched.  Every instrument is a
# monkey patch applied inside the arm's own subprocess, before the model is
# built, and every patch calls the original with its arguments unchanged.
#
# THE INPUT IS mg19's STAGED CONE SINOGRAM, read from $MG21_RESULTS, checksum
# verified on every arm, and placed on the devices ONCE per arm through the
# model's own sharding call.  Every timed call is then handed the already-placed
# input, so no host-to-device movement happens inside any bracket.  If that file
# is not there the arm makes its own sinogram with mg19's phantom seed and
# records that it did; the back projection's time depends on the sinogram's
# shape, not on its values.
#
# WALL ESTIMATE, built from mg19's two numbers above.  A three-iteration
# reconstruction's back projection is one Hessian call on every pixel plus, at
# each of granularities 4, 16 and 64, that many calls of about a quarter, a
# sixteenth and a sixty-fourth of the pixels.  Each of those three groups is
# about one full-pixel call's worth of work, so 137 s and 228 s put a single
# full-pixel call near 35 s at three devices and near 55 s at four.
#   * THE TIMED CALLS.  An arm runs nine calls at roughly full size (three
#     repeats each of full_p1, full_p2 and band_half) and nine smaller ones that
#     add up to about one more.  That is about 6 min at three devices and about
#     9 min at four, so 15 min in all.
#   * THE INPUT, per arm.  One md5 over a 30.5 GiB file, then that file placed
#     on the devices through a memory map.  About 5 min per arm, 10 min in all.
#   * FIRST-SHAPE TRITON COMPILES.  Each device compiles at its first launch of
#     each new shape, and the band length, the pixel count and the coefficient
#     power all key a shape, so each variant meets new ones.  The warmup repeat
#     is where that lands.  A few minutes per arm, 10 min in all.
#   * TWO SUBPROCESS STARTS, each paying a torch import, CUDA contexts on every
#     device, and the model build at this cell: 5 min.
# That is about 40 min, and 75 min is a comfortable upper bound.  The walltime
# below is 03:00:00, well over twice the estimate, because the compile cost and
# the parallel filesystem's read rate are both estimates rather than
# measurements here.
#
# DISK.  A few megabytes in $MG21_RESULTS: one jsonl.  No large artifact is
# written unless mg19's sinogram is missing, in which case the arm stages one
# 29.6 GiB phantom of its own under an mg21 name.  mg19's files are read and
# never modified.
#
# DEVICE MEMORY.  A device holds its sinogram shard, the band partial it is
# accumulating plus the block being added into it, the per-view precomputes for
# one view batch, and the result bands it already owns.  At three devices, which
# is the tighter of the two counts, that peaks near 35 GiB, well inside an H100.
#
# HOST MEMORY.  Nothing large is held on the host.  The sinogram is read through
# a memory map and copied to the devices shard by shard, and the outputs are
# reduced to two scalars on the devices rather than gathered.
#
# NO GATES ON THE EXIT CODE.  It reports INSTRUMENT HEALTH only: both arms ran
# and were recorded, the staged sinogram's checksum verified, each arm ran on its
# own device count and bound the kernel body through this run's wrapper, every
# variant produced the number of records its band structure predicts, and no arm
# ran out of event pairs.  What the parts add up to, whether halving the band
# doubled the cost, and how the reconstruction estimate compares with mg19's
# 137 s and 228 s are all printed in full and none of them changes the exit code
# -- they are the findings this job exists to produce.  Nothing is chained to
# this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg21_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 03:00:00
#SBATCH -J torch_mg21
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg21_%j.log
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
# THE FIRST ASSUMPTION: the checkout is the unpadded tree, in which 512 over
# three devices comes back as 171/171/170.  Three devices is one of the two
# counts this job compares, and a padded tree would give it a different shard
# shape, so the band structure this job attributes time to would not be the one
# in the rows.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, 0, 512).shard_ranges(512)]
assert blocks == [171, 171, 170], blocks
print('current tree: unpadded split confirmed:', blocks)" || { echo "the tree in mbirtorch_src is not the unpadded one mg21 was written for"; exit 1; }
# THE SECOND ASSUMPTION: this is the tree in which the banded multi-device
# forward has been removed and replaced by the cylinder transfer.  That merge
# also removed the column-gather names an earlier harness checked, so the check
# has to look at what the current tree has AND at what it no longer has.  A
# pre-merge checkout would import cleanly and run every arm, and its back
# projection would sit beside a different forward projection and a different
# memory structure than the one mg19 measured -- for an hour, at 2048 class,
# before anybody could tell.
"$TORCHPY" -c "
import mbirtorch
from mbirtorch import _sharding
from mbirtorch.tomography_model import TomographyModel
present = [('TomographyModel._sparse_forward_project_cylinders',
            hasattr(TomographyModel, '_sparse_forward_project_cylinders')),
           ('_sharding.transfer_cylinder_batch',
            hasattr(_sharding, 'transfer_cylinder_batch'))]
absent = [('TomographyModel.forward_project_slice_band',
           hasattr(TomographyModel, 'forward_project_slice_band')),
          ('TomographyModel._column_gather_forward',
           hasattr(TomographyModel, '_column_gather_forward')),
          ('_sharding.broadcast_band_to_views',
           hasattr(_sharding, 'broadcast_band_to_views'))]
for name, found in present:
    assert found, name + ' is MISSING'
for name, found in absent:
    assert not found, name + ' is still PRESENT'
print('current tree: cylinder-transfer forward present, banded-forward names removed')" || { echo "the tree in mbirtorch_src is not the merged banded-forward-removal tip mg21 was written for"; exit 1; }

# The shipped configuration, set explicitly.  This is NOT a variable under test:
# MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and four of this
# job's six instruments sit inside the cone back kernel's own wrapper.  With the
# kernels off, every arm would run the general torch body and record no parts at
# all.  The harness checks the binding on every arm and stops that arm within
# seconds of the settle, so this line is what keeps the check from being needed.
export MBIRTORCH_DISABLE_TRITON=0
# This job reads no memory ledger and must not own the per-device peak counters.
# The harness refuses an arm that finds this set.
unset MBIRTORCH_MEMORY_CALIBRATION
# Each arm pins its own device count through this variable, set per subprocess.
# A value inherited from the shell would pin both arms at once, and this job's
# whole reading is the difference between two device counts.
unset MBIRTORCH_NUM_DEVICES
# The process-wide column-gather override.  It would overrule the driver the
# memory structure of this job rests on, in either direction.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG21_SMOKE
unset MG21_ARMS
unset MG21_VARIANTS
unset MG21_DRY
# Where mg19's staged cone files are.  Unset so the harness's default applies,
# which is MG21_RESULTS -- mg19 wrote its files directly into that directory.
unset MG21_MG19_DIR

export MG21_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG21_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
ls -l "$MG21_RESULTS"/mg19_2k_cone_sinogram.npy "$MG21_RESULTS"/mg19_2k_cone_sinogram.npy.md5 || \
    echo "NOTE: mg19's staged cone sinogram is not there, so each arm will make its own.  That is recorded on the row and does not fail the job"

# The plan first, so the log opens with the two arms and the six variants, and a
# mis-set variable is caught before any GPU work.  The dry run imports no torch
# and touches no device.
export MG21_DRY=1
"$TORCHPY" -u mg21_back_attrib.py || exit 1
unset MG21_DRY

# The probe.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the findings are in the tables in this log.
"$TORCHPY" -u mg21_back_attrib.py
rc=$?
echo "MG21 DONE rc=$rc"
exit $rc
