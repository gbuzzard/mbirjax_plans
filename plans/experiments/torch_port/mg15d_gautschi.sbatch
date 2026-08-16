#!/bin/bash
# mg15d: THE DECIDING SINGLE-VARIABLE TEST, on four H100s.  Is the uneven-split
# multiaxis adjoint difference produced by torch.compile's DYNAMIC-SHAPE
# specialization?  The full statement is in mg15d_compile_ablation.py; the
# essentials are repeated here because a reader of this file may never open
# that one.
#
# WHERE THIS SITS.  mg15 gated the pad removal; its one failing leg was
# adjudicated by mg15b as trajectory-shaped, but mg15b's new adjoint instrument
# read 6.289e-04.  mg15c (slurm 15304817) took that apart and the matrix was
# decisive in an unexpected direction:
#     multiaxis adjoint  n=1 vs n=2 (both axes even)          6.976e-07
#                        n=1 vs n=3 (views 171/171/170)       ~6.29e-04
#                        n=1 vs n=4 (slices 128/128/127/127)  ~6.29e-04
#                        n=3 vs n=4                            5.975e-04
#                        same-count repeats                    EXACTLY 0.0
#     cone     adjoint   n=1 vs n=3 (uneven, Triton kernels)   7.1e-07
#     parallel adjoint   n=1 vs n=3 (uneven, Triton kernels)   6.9e-07
#     PADDED tree 672edbd n=1 vs n=4 (equal blocks)            5.980e-07
#
# The difference appears whenever EITHER sharded axis carries two distinct
# block lengths and not when the blocks are equal; it is DETERMINISTIC (the
# same-count repeats are exactly zero, so not atomics and not scheduling); and
# it is confined to multiaxis, which runs a general torch body, while cone and
# parallel, which back-project through hand-written Triton kernels, are clean
# at the same uneven count.
#
# ONE MECHANISM FITS ALL OF THAT, and remove_padding.md section 2 predicted it
# -- as a SPEED cost only: "a second distinct shape makes torch.compile treat
# that dimension as dynamic, which can produce slightly slower code than a
# specialized graph."  A dynamically-compiled body is not merely slower;
# inductor generates different code for it, and different code contracts a
# reduction in a different order.  For a back projection that means a voxel's
# ~512 same-sign float32 contributions are summed in a different association.
# That explains determinism (compiled code is fixed), the either-axis trigger
# (any second block length makes the dimension dynamic), Triton immunity
# (hand-written kernels never reach inductor codegen), the padded tree's
# cleanliness (equal blocks stay static), and mg15's mildly elevated forward
# reading of 1.585e-05 at n=4, whose slice bands are uneven too.
#
# THE TEST.  Eager bodies have no shape specialization, so with compilation OFF
# the identical uneven split must collapse to the float32 floor if the
# mechanism is what it appears to be.  Four arms, one variable:
#     compile_mode='auto' n=1     compile_mode='off' n=1
#     compile_mode='auto' n=4     compile_mode='off' n=4
#   auto n=4 vs n=1   expected ~6.29e-04, reproducing mg15c (the positive
#                     control -- the script checks it before reading anything
#                     else, because an ablation whose baseline did not
#                     reproduce the effect is untested, not refuted)
#   off  n=4 vs n=1   THE DISCRIMINATOR: ~1e-7 confirms, ~6e-4 refutes
#   off n=1 vs auto n=1   context, no gate
#
# TWO CORRECTNESS REQUIREMENTS, both in the script and both worth knowing here.
# compile_mode is passed to the MODEL CONSTRUCTOR, because the projector layer
# reads model.compile_enabled once when the Projectors object is built and
# binds that decision into each device's body -- a compile-off arm arranged
# after model build keeps the bodies it already bound and is silently a
# compile-ON arm, which the repo's lessons record.  And each arm is its own
# subprocess, because maybe_compile caches compiled instances at module level
# for the life of the process and an ablation must not share a compile cache
# between its arms.  Each arm additionally verifies, after its projection, that
# its bound bodies really are compiled or really are not, and that
# projectors._COMPILE_ERRORS is empty -- maybe_compile falls back to eager on a
# compile failure, and an 'auto' arm that silently ran eager would read exactly
# like an 'off' arm and look like confirmation.
#
# NO GATES.  The exit code reports INSTRUMENT HEALTH only -- every arm ran,
# every comparison computed, every arm built in the mode it claimed, and the
# positive control reproduced.  The verdict is read by a human from the table
# in this log.  Nothing is chained to this job.
#
# WALL ESTIMATE.  One generator plus four single back projections on the
# (512, 448, 384) cell -> recon (384, 384, 510).
#   * mg15c measured a single back projection at this cell in seconds, with the
#     cost dominated by one torch compilation per device per process.  The two
#     'auto' arms pay that: ~2 minutes each including compilation.
#   * the two 'off' arms compile nothing and are pure eager execution.  Eager
#     is slower per call but pays no compile: ~2-3 minutes each, and the script
#     records the wall so the eager cost is on the record for any remedy
#     discussion.
#   * one generator (a forward projection) plus md5 work over ~0.6 GiB of
#     staged phantom and sinogram and ~1.2 GiB of staged volumes: ~3 minutes.
#   * five subprocess starts, each paying a torch import: ~2 minutes.
# Total ~12-15 minutes.  The walltime below is 00:30:00, comfortably over
# twice that, and it clears the 02:00 and 03:00 nightlies.
#
# DISK.  About 1.8 GiB in $MG15D_RESULTS at peak: one sinogram (~0.34 GiB), one
# phantom (~0.29 GiB), four back-projection volumes (~0.29 GiB each).  Deleted
# at the end unless MG15D_KEEP_ARTIFACTS=1 is exported; the jsonl is always
# kept.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg15d_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 00:30:00
#SBATCH -J torch_mg15d
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg15d_%j.log
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
# THE ASSUMPTION THIS JOB RESTS ON: the checkout is the same unpadded tree
# mg15c measured.  An older checkout would import cleanly, run four arms, and
# ablate a phenomenon that does not exist in it.  512 over three devices must
# come back as 171/171/170.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, 0, 512).shard_ranges(512)]
assert blocks == [171, 171, 170], blocks
print('current tree: unpadded split confirmed:', blocks)" || { echo "the tree in mbirtorch_src is not the unpadded one mg15c measured"; exit 1; }

# The shipped configuration, set explicitly.  NOTE this is NOT the variable
# under test: MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels,
# while this ablation turns off torch.compile, and multiaxis has no
# hand-written kernels for the Triton switch to affect in any case.  It is
# pinned to the shipped value so the ablation varies one thing only.
export MBIRTORCH_DISABLE_TRITON=0
# This job measures VALUES ONLY and reads no memory, so it must not own the
# peak counter.  Popped again per subprocess.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins with an explicit configure_devices list.  Popped per subprocess.
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset MG15D_SMOKE
unset MG15D_DEVICE
unset MG15D_KEEP_ARTIFACTS

export MG15D_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG15D_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the four arms and the discriminator
# they are built around, and a mis-set variable is caught before GPU work.
"$TORCHPY" -u mg15d_compile_ablation.py --dry-run || exit 1
# The ablation.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the finding is in the table in this log.
"$TORCHPY" -u mg15d_compile_ablation.py
rc=$?
echo "MG15D DONE rc=$rc"
exit $rc
