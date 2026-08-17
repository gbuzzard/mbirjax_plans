#!/bin/bash
# mg19: THE FIRST 2048-CLASS RECONSTRUCTIONS, cone and parallel, on four H100s.
# The full statement is in mg19_two_k_baselines.py; the essentials are repeated
# here because a reader of this file may never open that one.
#
# WHERE THIS SITS.  A 2048-class reconstruction is the low end of production and
# nothing has ever been measured at that size.  What exists is a closed-form
# capacity table, built by evaluating the library's own memory model in the
# production configuration (plans/torch_port/active/two_k_design.md).  It says
# the problem does not fit on one or two H100s, fits on three with about 1.9 GiB
# to spare, and fits comfortably on four.  That table is a model calibrated at
# 512 and 1024 being asked about a problem four times larger in every dimension.
# This job is what validates it.
#
# THE CELL, one per geometry:
#   sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
#   recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask
#
# WHAT THIS JOB CARRIES, four readings in one run because the arms share their
# staged inputs:
#   * THE MEMORY VALIDATION.  Three-iteration reconstructions at three devices
#     (the model's edge) and at four (the operating point), in the ledger's
#     calibration mode, which reports the modeled peak beside the measured one
#     per device.  A two-device arm asserts the table's refusal for real.
#   * THE BATCH SWEEP.  The forward projection gathers pixel columns in batches
#     of 8192 by default.  The 1024-class sweep never bracketed an optimum, so
#     this one runs 8192, 16384, 32768 and 65536 at production scale.
#   * THE COMBINING SLAB.  Two cone arms move the back projection's streaming
#     slab from the shipped 64 MiB to 16 and 256 MiB.
#   * THE BACK PROJECTION'S SHARE OF TIME, from CUDA event pairs around every
#     forward and back projection call, per device.
#
# NOTHING IS FLIPPED HERE.  No library file is touched.  The pixel batch is set
# on each arm's own model instance before its layout settles, and the two slab
# arms patch a module attribute inside their own subprocess.
#
# THE ARMS.  Ten for cone, eight for parallel, eighteen in all, each in its own
# process, cone's whole block first:
#   gen (n=4, stages the phantom and the sinogram), refuse_n2 (n=2, expects the
#   preflight to refuse), n3_b8192, n4_b8192, n4_b8192_repeat, n4_b16384,
#   n4_b32768, n4_b65536, and for cone only n4_slab16 and n4_slab256.
# Every calibration arm forward-projects the staged phantom and gates it at 1e-4
# against the staged sinogram BEFORE its reconstruction runs; a gate failure
# skips that arm's reconstruction.
#
# WALL ESTIMATE, built from the 1024-class composed walls this campaign measured
# at four devices: 37.3 s for cone and 18.5 s for parallel.  Projection work
# scales as the fourth power of the linear size, so a 2048-class arm is 16 times
# a 1024-class one: about 10 min per cone composed arm and about 5 min per
# parallel one.
#   * COMPOSED ARMS, 16 of them.  8 cone at ~10 min and 8 parallel at ~5 min is
#     120 min.
#   * FIRST COMPILE, per new call shape.  Each geometry meets new shapes at
#     three devices and again at four, and each new pixel batch is another call
#     shape for the forward body.  A few minutes each, and the on-disk cache
#     makes only the first arm at a shape pay it: 25 min.
#   * THE TWO GENERATORS.  A 2048-class forward projection plus writing and
#     checksumming 60 GiB of phantom and sinogram: 8-10 min each, 20 min.
#   * ARTIFACT VERIFICATION.  Every arm re-checksums the phantom and the
#     sinogram it reads, which is 60 GiB of reading per composed arm on a
#     parallel filesystem: 30 min across the run.
#   * 18 SUBPROCESS STARTS, each paying a torch import and a CUDA context, and
#     the model build at this cell: 15 min.
# That is about 3.5 h.  The walltime below is 07:00:00, twice the estimate,
# because nothing at this size has ever been timed and the estimate rests on a
# scaling law rather than a measurement.
#
# DISK.  About 120 GiB in $MG19_RESULTS: two phantoms at 29.6 GiB, two sinograms
# at 30.5 GiB, and small fingerprint files.  It is NOT deleted at the end, so a
# re-run of a subset through MG19_ARMS can reuse it.  Remove it by hand when the
# rows have been read.
#
# HOST MEMORY.  The generator holds one 30.5 GiB sinogram in host memory between
# the projection and the write, and every composed arm gathers a 29.6 GiB volume
# to the host at the end.  Everything else is memory-mapped.  The -n 56 request
# below carries the node's memory with it on this partition.
#
# NO GATES ON THE EXIT CODE.  It reports INSTRUMENT HEALTH only: every arm ran
# and was recorded, every artifact staged, every checksum verified, and no arm
# ran on the wrong device count, bound the wrong kind of projection body, used a
# pixel batch other than its own, or ran under the wrong calibration setting.  A
# values gate failure, a preflight refusal, a missing refusal at two devices, and
# a calibration ratio outside the 1.00-to-1.30 band are all printed in full and
# none of them changes the exit code -- they are the findings this job exists to
# produce.  The verdicts are read by a person from the table in this log.
# Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg19_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 07:00:00
#SBATCH -J torch_mg19
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg19_%j.log
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
# three devices comes back as 171/171/170.  Three devices is one of the counts
# this job measures, and a padded tree would give it a different shard shape and
# therefore a different memory structure than the one being validated.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, 0, 512).shard_ranges(512)]
assert blocks == [171, 171, 170], blocks
print('current tree: unpadded split confirmed:', blocks)" || { echo "the tree in mbirtorch_src is not the unpadded one mg19 was written for"; exit 1; }
# THE SECOND ASSUMPTION: all four geometry classes declare
# column_gather_geometry True.  This job runs after the default flip on
# translation and multiaxis lands, and the shipping memory structure is the
# column gather's.  A pre-flip checkout would import cleanly, run every arm, and
# validate a capacity table against a memory structure that is not the one
# shipping -- for hours, at 2048 class, before anybody could tell.  Cone and
# parallel are checked in the same breath, because they are the two geometries
# this job actually measures and a regression on either would be worse still.
"$TORCHPY" -c "
import mbirtorch
classes = {'ConeBeamModel': mbirtorch.ConeBeamModel,
           'ParallelBeamModel': mbirtorch.ParallelBeamModel,
           'TranslationModel': mbirtorch.TranslationModel,
           'MultiAxisParallelModel': mbirtorch.MultiAxisParallelModel}
for name, cls in classes.items():
    assert cls.column_gather_geometry is True, (
        name + ' does not declare column_gather_geometry True; this is not the '
        'post-flip tree whose memory structure mg19 was written to validate')
print('current tree: all four geometries declare the column gather')" || { echo "the tree in mbirtorch_src is not the post-flip one mg19 was written for"; exit 1; }

# The shipped configuration, set explicitly.  This is NOT a variable under test:
# MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and this job's two
# geometries bind those kernels in BOTH directions on an H100 -- which every arm
# verifies and records.  It is pinned to the shipped value so the job varies one
# thing at a time.
export MBIRTORCH_DISABLE_TRITON=0
# The calibration mode is per arm: the composed arms need it and the generator
# must not have it, because it owns and resets the peak counters.  The harness
# sets it per subprocess, so a value inherited from the shell would put it on
# every arm at once.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins its own device count through this variable, set per subprocess.
# A value inherited from the shell would pin every arm at once.
unset MBIRTORCH_NUM_DEVICES
# The process-wide column-gather override.  It would overrule the driver the
# whole memory structure of this job rests on, in either direction.  Popped
# again per subprocess.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG19_SMOKE
unset MG19_GEOMS
unset MG19_ARMS

export MG19_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG19_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The arm plan first, so the log opens with the eighteen arms, their pins and
# their batches, and a mis-set variable is caught before any GPU work.  The dry
# run imports no torch and touches no device.
export MG19_DRY=1
"$TORCHPY" -u mg19_two_k_baselines.py || exit 1
unset MG19_DRY

# The measurement.  Its exit code reports whether the INSTRUMENTS worked, not
# what they found; the findings are in the table in this log.
"$TORCHPY" -u mg19_two_k_baselines.py
rc=$?
echo "MG19 DONE rc=$rc"
exit $rc
