#!/bin/bash
# mg5: CHARTER A -- ATTRIBUTE THE FORWARD AT n>1, on 4 H100s.
# The forward view chunk swept at the two anomalous coordinates, plus one
# torch-forward-body arm per coordinate as the discriminator.  See
# mg5_fwd_attrib.py, multigpu_findings.md §6.2 (the charter), §1.3 (the
# anomalies) and §3.5 (why these knees are the forward-sensitive ones), under
# the eleven protocols of multigpu_plan.md §3.
#
# 27 MEASURED ARMS + 2 untimed generator arms.  Projected wall 2.5-4 h on four
# GPUs, from mg1's MEASURED per-arm subprocess walls at the 1024 cells (2-5 min,
# cold pass included) times 27, with the torchfwd arms carried at the composed
# +19-24 percent the parallel selection note records for a torch forward.  The
# walltime below is 06:00:00.
#
# WHAT THIS JOB IS FOR.  mg1's three-region instrument found two anomalies and
# attributed neither:
#   * the CONE FORWARD IS FLAT IN DEVICE COUNT -- 31.0, 29.9, 28.6 s at
#     n = 1, 2, 4 on cone 1024, while per-device views drop fourfold.  That
#     region is 40-54 percent of the cell's composed wall, so its flatness is
#     why cone n=2 is a net regression (0.92x) and n=4 gains only 1.16x.
#   * the PARALLEL FORWARD RISES AT n=2 -- 15.9, 18.0, 7.3 s on parallel 1024.
#     A forward that costs MORE with half the views per device points at a
#     per-device regime change rather than at communication.
# The sweep separates three hypothesis classes: (a) kernel-specific behavior at
# shard shapes, (b) the realized per-device view batch shifting under the
# count-divided budget, (c) the driver fan-out and assembly around the kernel.
#
# THE ATTRIBUTION IS APPLIED IN ANALYSIS, not by this harness, exactly as mg4's
# knee rule was.  The job prints the three hypotheses' signatures and the
# numbers that discriminate them -- the forward region and its scaling per
# count, the realized forward view batch, and the forward funnel's host wall
# MINUS its largest per-device event span (the fan-out gap protocol 10's
# instrument exists to price) -- and prints no verdict.
#
# THE TORCHFWD ARM DOES NOT USE THE KILL SWITCH, and this is the one design
# point most likely to be "fixed" wrongly by a later editor.  MBIRTORCH_DISABLE_
# TRITON=1 disables BOTH directions, which would confound the forward change
# with a back change and answer no question.  The arm binds the bodies
# explicitly instead -- the torch forward body against the body the shipped
# selection itself returned for the back -- the way tests/test_kernels_sharded.py
# and the dp-era probes bound per-direction bodies.  Every arm in this job,
# torchfwd included, runs with MBIRTORCH_DISABLE_TRITON=0 and asserts it, and
# the torchfwd arms carry the launch-key witness in BOTH directions: back keys
# must grow and forward keys must NOT.
#
# STAGING, and the one dependency that is not obvious (protocol 11).  mg5
# IMPORTS mg1_readout.py for the region instrument, so BOTH FILES must be staged
# in this directory; the import failure says so by name.  The import is
# deliberate and departs from mg2/mg3/mg4, which copied the machinery: mg5's
# whole deliverable is its forward-region numbers set against mg1's recorded
# 31.0/29.9/28.6 and 15.9/18.0, and a copied-then-drifted instrument would
# produce numbers that look comparable and are not.  Every file is scp'd PER
# FILE and md5-verified; no sync happens into a tree a running job imports from,
# so do not touch mbirtorch_src while any nightly or trial job is live.
#
# ARTIFACTS.  One shared sinogram per geometry at the 1024 cell (~4 GB each),
# md5-verified by every arm that reads it, deleted at the end unless
# MG5_KEEP_ARTIFACTS=1 is exported.
#
# SCHEDULING (protocol 11).  This is a four-GPU block, so it must clear the
# 02:00 jax nightly and the 03:00 torch nightly (itself up to four hours on a
# changed-tip night).  Submit outside that window, or add --begin.  Charter A
# and charter B read different regions and may share a cluster window
# (multigpu_findings.md §6.7); mg5 waits on nothing else.
#
# IF THIS JOB MUST BE SPLIT, split it by GEOMETRY, never by count and never by
# chunk: every reading here compares counts within a coordinate and chunks
# within a count, so neither axis may cross a job boundary.
#     MG5_GEOMS=cone
#     MG5_GEOMS=parallel
# A cross-job comparison then pins the node, per the h010 precedent.  If the job
# runs long, the two shedding knobs are MG5_SKIP_TORCH_FWD=1 (drops the five
# discriminator arms; keeps the chunk sweep) and MG5_SKIP_DRIFT=1 (drops the two
# tail repeats) -- in that order of preference, since the drift arms are what
# price the n=1-first ordering's bias.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg5_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 06:00:00
#SBATCH -J torch_mg5
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg5_%j.log
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
# The staging dependency, asserted before four GPU-hours are spent on it.
test -f /scratch/gautschi/buzzard/torch_p3/mg1_readout.py || { echo "mg1_readout.py is not staged beside mg5_fwd_attrib.py; mg5 imports its region instrument"; exit 1; }

# The kernels are the shipped production configuration, and the torchfwd arms
# reach the torch forward body by BINDING it, never by this switch -- the switch
# would take the back kernel down with it.  Explicitly OFF rather than merely
# unset, and every arm re-asserts it in its own environment.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1: every arm pins through MBIRTORCH_NUM_DEVICES, which the SCRIPT
# sets per arm; an inherited value here would pin the whole job to one count
# without saying so.  The script asserts the per-arm value in the arm itself.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: the calibration mode owns torch.cuda.max_memory_allocated, and
# mg5's peaks are read straight off that counter.  mg2 is the only job that
# sets it.
unset MBIRTORCH_MEMORY_CALIBRATION
# The forward chunk ladder is PINNED by the charter and set in the script; it is
# named here only so a re-run can narrow it without editing the file.
unset MG5_CHUNKS

export P0_TORCH_PYTHON=$TORCHPY
export MG5_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG5_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg5_fwd_attrib.py
echo "MG5 DONE"
