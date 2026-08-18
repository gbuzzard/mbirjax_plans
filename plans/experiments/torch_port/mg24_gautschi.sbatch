#!/bin/bash
# mg24: THE CONFIRMATION RUNS FOR THE BAND-PADDING REMEDY, cone at the 2048
# class, on four H100s.
# The full statement is in mg24_padding_confirm.py; the essentials are repeated
# here because a reader of this file may never open that one.
#
# WHERE THIS SITS.  The cone back projection cost more on four devices than on
# three.  mg19 measured the composed back busy time at the 2048-class cell:
# 136.8 s at three devices and 227.9 s at four.  mg21 and mg21b found the
# cause.  Triton compiles a separate, faster kernel for each integer argument
# it can prove is a multiple of 16.  The back kernel's band argument is the
# slice block each device owns: 672 slices at three devices, which is a
# multiple of 16, and 504 at four, which is not.  The remedy rounds that
# argument up to the next multiple of 16 inside the kernel wrapper, computes
# the padded columns, and discards them.  The design note is
# plans/torch_port/active/back_remedy_design.md, and this job is its
# increment 3: the acceptance evidence.
#
# THE CELL, mg19's cone cell unchanged, so the two runs measure the same
# problem:
#   sinogram (2048, 2016, 1984) as (views, detector rows, channels), 30.5 GiB
#   recon    (1984, 1984, 2016), 29.6 GiB, 3,088,364 pixels inside the mask
#
# WHAT THIS JOB READS.  Three-iteration cone reconstructions at three and four
# devices, in the ledger's calibration mode, with CUDA event pairs around every
# forward and back projection call.  The back busy times are then read against
# the design note's section 3 projections:
#   three devices  136.8 s, and it should NOT move.  The band is 672, which is
#                  already a multiple of 16.
#   four devices   227.9 s, and it should FALL TO NEAR 100 s.  The band goes
#                  from 504 to a padded 512.
# The calibration rows carry a second reading: the ledger now charges the
# PADDED band length, so these ratios are how that charge is checked at
# production scale.
#
# NOTHING IS FLIPPED HERE.  No library file is touched and no default is
# changed.  Every arm runs the shipped configuration.
#
# THE ARMS, four, cone only, each in its own process:
#   gen (n=4, verifies the phantom and sinogram mg19 staged and rebuilds only
#   what is missing), n3_shipped, n4_shipped, n4_shipped_repeat.
# Every composed arm forward-projects the staged phantom and gates it at 1e-4
# against the staged sinogram BEFORE its reconstruction runs; a gate failure
# skips that arm's reconstruction.
#
# THE PIXEL BATCH.  The arms run the shipped batch, which is 32768 today, and
# mg19's comparison arms ran 8192.  The batch is a forward-projection knob and
# the back projection never reads it, so the back comparison is a comparison of
# the same work.  The forward busy time will differ, and the summary prints
# mg19's own reading at 32768 beside it.
#
# THE TREE CHECKS BELOW.  Two of them.  The first says the device sharding
# still splits a length with remainders.  The second says this is the merged
# tree AND that it carries the padding, which is the whole point of the job: an
# unpadded checkout would import cleanly, run every arm, and produce four
# hours of numbers that answer a question nobody asked.
#
# WALL ESTIMATE.  mg19's composed cone arms ran about 7 to 8 minutes each at
# four devices and about 8 minutes at three.  This job has three of them, and
# the padded back projection should make the four-device arms faster rather
# than slower.  Add the generator, which on a results directory that already
# holds mg19's files only re-checksums 60 GiB; add four subprocess starts, each
# paying a torch import, a CUDA context and the model build at this cell; and
# add the first compile at each new call shape.  The whole job should finish in
# under an hour.  The walltime below is 04:00:00, which is margin.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment, so
# it must not run at the same time as another torch_p3 job that installs into
# the same environment.  When another such job is queued, submit this one with
# --dependency=afterany:<that job id>.
#
# DISK.  This job reads the two large files mg19 left in $MG24_RESULTS, about
# 60 GiB, and writes only small fingerprint files and one jsonl.  It deletes
# nothing.  If either large file is missing, the generator arm rebuilds it and
# both the row and the summary say so loudly, because a rebuilt sinogram comes
# from THIS tree and not from the tree mg19 measured.
#
# HOST MEMORY.  Every composed arm gathers a 29.6 GiB volume to the host at the
# end.  Everything else is memory-mapped.  The -n 56 request below carries the
# node's memory with it on this partition.
#
# NO GATES ON THE EXIT CODE.  It reports INSTRUMENT HEALTH only: every arm ran
# and was recorded, every artifact verified, and no arm ran on the wrong tree,
# the wrong device count, the wrong kind of projection body, or the wrong
# calibration setting.  A values gate failure, a preflight refusal, a
# calibration ratio outside the 1.00-to-1.30 band, and a back busy time that
# misses the projection are all printed in full and none of them changes the
# exit code.  The verdict is read by a person from the comparison table in this
# log.  Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg24_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 04:00:00
#SBATCH -J torch_mg24
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg24_%j.log
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

# THE FIRST ASSUMPTION: the device sharding splits a length with remainders and
# rounds nothing, so 512 over three devices comes back as 171/171/170.  This is
# the sharding, not the kernel-width padding this job confirms; the two are
# different mechanisms and only the kernel one changed.  Three devices is a
# count this job measures, and a sharding that rounded its blocks would give
# that count a different memory structure than mg19 measured.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, 0, 512).shard_ranges(512)]
assert blocks == [171, 171, 170], blocks
print('current tree: the sharding splits with remainders:', blocks)" || { echo "the sharding in mbirtorch_src is not the one mg24 was written for"; exit 1; }
# THE SECOND ASSUMPTION: this is the merged tree AND it carries the band
# padding.  The first three checks are the 2026-08-17 merge that removed the
# banded multi-device forward.  The fourth is this job's own: 504 is the
# four-device band at this cell and 512 is what the remedy must turn it into.
# An unpadded checkout would import cleanly, run every arm, and spend hours
# measuring the problem the remedy was written to fix.  The harness repeats all
# four checks on every arm and records them on every row; they are here so a
# wrong checkout costs seconds instead of hours.
"$TORCHPY" -c "
from mbirtorch.tomography_model import TomographyModel
from mbirtorch import _sharding
merged = 'the tree is not the merged banded-forward-removal tip mg24 measures'
assert hasattr(TomographyModel, '_sparse_forward_project_cylinders'), merged
assert hasattr(_sharding, 'transfer_cylinder_batch'), merged
assert not hasattr(_sharding, 'broadcast_band_to_views'), merged
from mbirtorch._utils import padded_kernel_width
got = int(padded_kernel_width(504))
assert got == 512, (
    'padded_kernel_width(504) is ' + str(got) + ', not 512; this tree does not '
    'carry the band padding mg24 exists to confirm')
print('current tree: merged, and the band padding is present: 504 ->', got)" || { echo "the tree in mbirtorch_src is not the padded merged tree mg24 was written for"; exit 1; }

# The shipped configuration, set explicitly.  This is NOT a variable under
# test: MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and the
# padding lives inside one of those kernels' wrappers.  With the kernels off
# this job would measure nothing at all, which every arm verifies and records.
export MBIRTORCH_DISABLE_TRITON=0
# The calibration mode is per arm: the composed arms need it and the generator
# must not have it, because it owns and resets the peak counters.  The harness
# sets it per subprocess, so a value inherited from the shell would put it on
# every arm at once.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins its own device count through this variable, set per
# subprocess.  A value inherited from the shell would pin every arm at once,
# and the three-against-four comparison is the whole job.
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset MG24_SMOKE
unset MG24_ARMS
unset MG24_DRY

# The same results directory mg19 wrote to: this job reads mg19's staged
# phantom and sinogram from there.
export MG24_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG24_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The arm plan first, so the log opens with the four arms, their pins and the
# numbers this run will be read against, and a mis-set variable is caught
# before any GPU work.  The dry run imports no torch and touches no device.
export MG24_DRY=1
"$TORCHPY" -u mg24_padding_confirm.py || exit 1
unset MG24_DRY

# The measurement.  Its exit code reports whether the INSTRUMENTS worked, not
# what they found; the findings are in the table in this log.
"$TORCHPY" -u mg24_padding_confirm.py
rc=$?
echo "MG24 DONE rc=$rc"
exit $rc
