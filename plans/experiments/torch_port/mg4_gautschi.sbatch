#!/bin/bash
# mg4: THE CROSSOVER LADDER on 4 H100s -- where each device count stops paying.
# Six parallel-family cells at exact proportions of the 512 gate cell, the
# (1024,1008,992) off-family anchor, a three-size cone spot check bracketing the
# expected knee, and one thin-volume probe point sized so the candidate
# work-size metrics DISAGREE.  Every cell runs n = 1, 2, 4 pinned plus one auto
# arm, warm seeded 3-iteration VCD, one subprocess per arm.  See mg4_ladder.py
# and multigpu_plan.md §4 (mg4), §6 decision 1 and §8, under the eleven
# protocols of §3.
#
# 44 measured arms (28 parallel incl. the anchor + 12 cone + the 4-arm probe
# point) + 11 untimed generator arms.  §7 estimates 1.5-3 h; the walltime below
# is twice the upper estimate, per protocol 11.
#
# WHAT THIS JOB IS FOR.  §6 decision 1: the automatic path is capacity-only
# today and widens small problems onto counts that run slower.  The proposed
# guard is one floor per count, and each floor lands at this ladder's admission
# point for that count under §4's knee rule, with a margin toward FEWER devices.
# The KNEE RULE IS APPLIED IN ANALYSIS, not by this harness: it reads every
# larger ladder size, so no single job printout can decide it.  Every row
# carries what the rule needs -- warm median, spread, realized device count,
# per-device memory, and the health sample.
#
# THE THIN-VOLUME PROBE answers §8's open question about the guard's metric.
# Its point (1152, 336, 96) has the sinogram elements of the (384,336,288)
# ladder cell -- one step ABOVE the expected 256-class knee -- and fewer recon
# voxels than the (192,168,144) cell -- one step BELOW it.  The arithmetic is in
# mg4_ladder.py's docstring.  If this probe is dropped, §8's question cannot be
# answered from the ladder alone, because within the family the two metrics are
# proportional.
#
# SCHEDULING (protocol 11).  This job is a four-GPU block, so it must clear the
# 02:00 jax nightly and the 03:00 torch nightly (itself up to four hours on a
# changed-tip night).  Submit outside that window, or add --begin.  §5
# increment 2 sequences mg4 behind mg1's VALIDITY RECORD -- not behind the
# checkpoint's ruling, because mg4 is measurement, not tuning.  mg3 and mg4 are
# independent of each other and may run in either order, or chained:
#     sbatch --dependency=afterok:<mg1 jobid> mg4_gautschi.sbatch
#
# IF THIS JOB MUST BE SPLIT, split it by FAMILY, never by count: the knee rule
# compares counts within a cell, so a count must never cross a job boundary.
#     MG4_FAMILIES=parallel_family,thin_probe
#     MG4_FAMILIES=off_family_anchor,cone_spot
# Any cross-job comparison then pins the node or leans on the off-family anchor
# as the shared anchor cell, per the h010 precedent (protocol 11).
#
# STAGING (protocol 11).  Every file is scp'd PER FILE and md5-verified; no sync
# happens into a tree a running job imports from, so do not touch mbirtorch_src
# while any nightly or trial job is live.  This job writes one sinogram artifact
# per cell (the anchor's is ~4 GB), md5-verified by every arm that reads it, and
# deletes them at the end unless MG4_KEEP_ARTIFACTS=1.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg4_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 06:00:00
#SBATCH -J torch_mg4
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg4_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this campaign a queue delay before.

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

# The kernels are the shipped production configuration and this ladder measures
# what a real out-of-box multi-GPU recon() does, so the kill switch is
# explicitly OFF rather than merely unset.  Protocol note: compile_mode='off'
# does NOT disable the Triton kernels -- selection is availability-driven, not
# compile-driven.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1 and §6: every PINNED arm pins through MBIRTORCH_NUM_DEVICES, which
# the SCRIPT sets per arm; the AUTO arms must see it ABSENT, because the pin
# bypasses the automatic branch entirely and an inherited value here would turn
# every auto arm into a pinned one without saying so.  The script asserts the
# absence in the arm itself.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: the calibration mode owns torch.cuda.max_memory_allocated, and
# mg4's peaks are read straight off that counter.  mg2 is the only job that
# sets it.
unset MBIRTORCH_MEMORY_CALIBRATION

export P0_TORCH_PYTHON=$TORCHPY
export MG4_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG4_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg4_ladder.py
echo "MG4 DONE"
