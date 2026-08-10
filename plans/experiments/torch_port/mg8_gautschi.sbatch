#!/bin/bash
# mg8: MEMORY CALIBRATION FOR THE TWO NEW GEOMETRIES -- translation and
# multiaxis parallel -- on four H100s.  Modeled peak memory against measured
# peak memory, per device, at one, two and four devices.  The full statement of
# what is measured and why is in mg8_geom_calib.py; the essentials are repeated
# here because a reader of this file may never open that one.
#
# WHY THIS JOB EXISTS.  Before a reconstruction starts, mbirtorch estimates how
# many bytes it will hold on each GPU (the "memory ledger"), and uses that
# estimate to decide how many GPUs to spread across and whether to run at all.
# The estimate may be too large; it must never be too small, because too small
# means a doomed run starts and then dies inside the allocator.
#
# A large part of that estimate is what one batch of projection views costs.
# For cone-beam and parallel-beam that cost comes from the hand-written (Triton)
# kernels, which declare their own real per-view cost.  TranslationModel and
# MultiAxisParallelModel have NO hand-written kernels: they run on general torch
# code, and for that code the cost falls back to a proxy -- one nominal slab,
# pixels x columns x 4 bytes.  That proxy has now been measured against the real
# bodies and it is short: 10x short on the multiaxis forward projection and 4.3x
# short on the multiaxis back projection, through the compiled bodies the driver
# actually binds.  The library's own source comment estimates only 2-5x.
#
# This job measures the composed reconstruction so the correction can be built
# from data.  It does not apply a correction.
#
# WHAT A "FAILING" ROW MEANS HERE.  Ratios BELOW the 1.00 floor are the expected
# result on the arms a projection phase dominates, and are the finding this job
# exists to produce -- not a fault.  The script says so on the summary line, and
# it keeps two separate lists: readings below the floor, and rows that cannot be
# believed (wrong device count, missing calibration flag, checksum mismatch,
# unexpected recon shape, or a projection body that turned out to declare its
# own cost after all).  Only the second list is a problem.
#
# THE ARMS.  12 measured arms plus 4 untimed sinogram generators.
#     multiaxis   (1024, 1008, 992)  and  (512, 448, 384)
#     translation (256, 1900, 3000)  and  (256, 950, 1500)
# each at 1, 2 and 4 devices.  The multiaxis shapes are the same two shapes the
# cone/parallel calibration used, so the new rows sit beside the existing ones.
# The large translation shape is the production translation-CT shape; the small
# one is that shape halved on both detector axes, so the trend over problem size
# is measured rather than inferred from a single point.
#
# WALL ESTIMATE, and its arithmetic.  The reference is the measured pure-torch
# (no kernels) warm 3-iteration reconstruction on one H100 from the projector
# batching work: 143.5 s at the 1024-class parallel cell and 4.87 s at the
# 512-class one.  These geometries run on exactly that kind of code.
#   * multiaxis (1024, 1008, 992) reconstructs into (992, 992, 1148), about
#     1.14x the slices of the parallel 1024 cell, so ~165 s per pass, plus one
#     torch compilation per device on the first pass.  Call it 3-5 minutes per
#     arm, three arms: ~12-15 minutes.
#   * multiaxis (512, 448, 384) reconstructs into (384, 384, 510): seconds per
#     pass, dominated by compilation.  Three arms: ~4 minutes.
#   * translation (256, 1900, 3000) reconstructs into only (118, 360, 240) --
#     42,480 pixels -- so its projection work is a few hundred times smaller
#     than the 1024 cell's, but its sinogram is 1.46e9 elements (5.44 GiB, or
#     5.84 GB) and the sinogram-shaped algebra dominates instead.  It is the
#     largest single artifact this job writes.  1-3 minutes per arm, three
#     arms: ~6-9 minutes.
#   * translation (256, 950, 1500) is a quarter of that: ~4 minutes for three.
#   * four generators (one forward projection each) plus the checksum work over
#     ~11 GiB of sinogram, each file re-verified by its three consumers:
#     ~8-10 minutes.
#   * 16 subprocess starts, each paying a torch import: ~3 minutes.
# Total ~40-45 minutes, upper estimate ~1 hour.  The walltime below is 02:00:00,
# roughly twice the upper estimate, which also covers a cold compilation cache
# and a shared node.
#
# ONE OUTCOME WORTH EXPECTING.  If the shortfall is as large as the single-body
# measurement suggests, an arm can run out of device memory -- that is exactly
# what "the ledger under-predicts" means in practice.  The script records such
# an arm as a row carrying its traceback and moves to the next one, so a single
# failure does not cost the job its other eleven arms.
#
# DISK.  About 11 GiB of sinogram in $MG8_RESULTS during the run (3.81 + 0.33 +
# 5.44 + 1.36 GiB), deleted at the end unless MG8_KEEP_ARTIFACTS=1 is exported.
#
# IF THIS JOB MUST BE SPLIT, split it by GEOMETRY and by nothing else:
#     MG8_GEOMETRIES=multiaxis      6 arms
#     MG8_GEOMETRIES=translation    6 arms
# Every reading compares device counts within one shape, so the count axis may
# never cross a job boundary.  MG8_COUNTS exists to re-run a single arm, not to
# shard the job.  A cross-job comparison should pin the node.
#
# SCHEDULING.  This is a four-GPU block, so it must clear the 02:00 and 03:00
# nightlies.  At around an hour it fits comfortably between them.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg8_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg8
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg8_%j.log
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
# The two geometries measured here are new enough to be worth naming before any
# GPU time is spent: an older checkout would import cleanly and then fail every
# arm one subprocess at a time.
"$TORCHPY" -c "import mbirtorch; mbirtorch.MultiAxisParallelModel; mbirtorch.TranslationModel; mbirtorch.gen_translation_vectors" || { echo "this mbirtorch has no MultiAxisParallelModel / TranslationModel: stage the merged tip (1a2deb0 or later) into mbirtorch_src"; exit 1; }

# Neither of these geometries has hand-written kernels, so this switch changes
# nothing they do.  It is exported anyway, and explicitly OFF rather than merely
# unset, because the measurement has to be taken in the shipped production
# environment and not in a variant of it.  The script re-asserts it inside every
# arm.
export MBIRTORCH_DISABLE_TRITON=0
# Each arm pins its own device count, which the SCRIPT pops and then sets per
# subprocess.  A value inherited from here would pin the whole job to one count
# without saying so.  Each arm then asserts the device list it actually got.
unset MBIRTORCH_NUM_DEVICES
# THE ONE JOB CLASS THAT OWNS THIS FLAG.  The calibration mode RESETS
# torch.cuda.max_memory_allocated so it can read it at the end of the
# reconstruction, which means it takes ownership of a counter any other measured
# job would be reading.  It is exported here, as 1, because owning that counter
# is this job's entire purpose -- every other job in this family unsets it.  The
# script still pops and re-sets it per subprocess, so the four sinogram
# generators run WITHOUT it and never reset a counter they are not reading.
export MBIRTORCH_MEMORY_CALIBRATION=1
# The split and re-run knobs, named here only so a narrowed re-run needs no edit
# to the file.  Split by GEOMETRY; MG8_COUNTS is for re-running one arm.
unset MG8_GEOMETRIES
unset MG8_SHAPES
unset MG8_COUNTS
unset MG8_KEEP_ARTIFACTS

export MG8_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG8_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the 12 arms and the registered
# expectation, and a mis-set environment variable is caught before any GPU work.
"$TORCHPY" -u mg8_geom_calib.py --dry-run || exit 1
"$TORCHPY" -u mg8_geom_calib.py
echo "MG8 DONE"
