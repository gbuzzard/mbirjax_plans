#!/bin/bash
# mg3: THE VALUE COMPARISON on 4 H100s -- the shared-sinogram cross-framework
# and cross-configuration residuals at n = 1, 2, 4, both geometries, both gate
# cells, warm seeded VCD, one subprocess per arm.  Three arms per
# (geometry, cell, count): jax production, torch EAGER-PLAIN (kernels off AND
# compile off), torch production.  Parallel 1024 adds the ten-iteration tier and
# the optional compile-ordering tier.  See mg3_value.py and multigpu_plan.md §4
# (mg3) under the eleven protocols of §3.
#
# 45 measured arms + optional 3, plus 4 untimed generator arms.
#
# ─────────────────────────────────────────────────────────────────────────────
# RUN THE PRICING PROBE FIRST.  §4 and §5 increment 2: mg3's allocation is
# committed only after a probe, because 15 of its arms are eager-plain and no
# eager-plain composed time has ever been measured at the 1024 cells.  §7 calls
# the mg3 row the widest spread and the least trusted in the whole campaign.
# The probe RE-DERIVES the real need; the -t 08:00:00 below is twice §7's upper
# estimate and is a placeholder until the probe speaks.
#
# THE PROBE IS A 1-GPU JOB.  It runs exactly two measured arms -- torch
# eager-plain, n=1, 3 iterations, at each 1024 cell -- about ten minutes.
# Copy this file, replace the SBATCH block with the one below, and add
# `export MG3_PROBE=1` beside the other exports:
#
#     #SBATCH -A bouman
#     #SBATCH -p ai
#     #SBATCH -q normal
#     #SBATCH -N 1
#     #SBATCH --gpus-per-node=1
#     #SBATCH -n 14
#     #SBATCH -t 01:00:00
#     #SBATCH -J torch_mg3probe
#     #SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg3probe_%j.log
#
# THE SUBMITTER MAY SPLIT THIS JOB.  If the probe's projection exceeds the
# walltime below, run mg3 as two chained jobs rather than raising the walltime
# past what the queue will schedule:
#     MG3_SKIP_DEEP=1                       the depth-3 tiers (36 arms + 3)
#     MG3_CELLS=1024 MG3_GEOMS=parallel     the depth-10 tier (9 arms)
# chained with `sbatch --dependency=afterok:<jobid>`.  The probe's printout
# names both sub-walls explicitly and says which way to go.  Splitting is safe
# for the value columns: every column is computed WITHIN one (geometry, cell,
# tier, count) block, so no column crosses the split.
# ─────────────────────────────────────────────────────────────────────────────
#
# SCHEDULING (protocol 11).  This job is a four-GPU block, so it must clear the
# 02:00 jax nightly and the 03:00 torch nightly (itself up to four hours on a
# changed-tip night).  Submit outside that window, or add --begin.  §5
# increment 2 sequences mg3 behind mg1's VALIDITY RECORD -- not behind the
# checkpoint's ruling, because mg3 is measurement, not tuning:
#     sbatch --dependency=afterok:<mg1 jobid> mg3_gautschi.sbatch
#
# STAGING (protocol 11).  Every file is scp'd PER FILE and md5-verified; no sync
# happens into a tree a running job imports from, so do not touch mbirtorch_src
# while any nightly or trial job is live.  The shared sinogram ARTIFACTS this
# job writes are ~4 GB each at the 1024 cells and are md5-verified by every arm
# that consumes them (protocol 5: a corrupt Lustre read is a recorded failure
# mode).  They are deleted at the end of the run unless MG3_KEEP_ARTIFACTS=1.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg3_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 08:00:00
#SBATCH -J torch_mg3
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg3_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this campaign a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
JAXPY=$(conda run -n mbirjax which python)
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1
"$JAXPY" -c "import jax; assert len(jax.devices()) >= 4, jax.devices()" || exit 1

# Protocol 2: MBIRTORCH_DISABLE_TRITON is set PER ARM by the script -- 1 on the
# eager-plain and compiled-kernels-off arms, 0 on the production arms.  Nothing
# is exported here, because an inherited value would blur exactly the two arm
# classes this instrument exists to tell apart.  Note also that compile_mode
# does NOT disable the kernels: compile is turned off through the model
# CONSTRUCTOR (compile_mode='off'), which is the other half of "eager-plain".
unset MBIRTORCH_DISABLE_TRITON
# Protocol 1: every pinned arm pins through MBIRTORCH_NUM_DEVICES, which the
# SCRIPT sets per arm.  An inherited value here would blur every arm.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: the calibration mode owns torch.cuda.max_memory_allocated, and
# mg3's peaks are read straight off that counter.  mg2 is the only job that
# sets it.
unset MBIRTORCH_MEMORY_CALIBRATION

export P0_JAX_PYTHON=$JAXPY
export P0_TORCH_PYTHON=$TORCHPY
export MG3_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG3_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg3_value.py
echo "MG3 DONE"
