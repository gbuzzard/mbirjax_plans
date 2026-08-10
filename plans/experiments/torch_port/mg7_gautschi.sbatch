#!/bin/bash
# mg7: THE CONE VIEW-BATCH PROBE -- does the realized per-device view batch FALL
# with the device count where the 2 GiB transient cap is NOT binding?  On four
# H100s.  See mg7_conebatch.py for the full reading, the registered prediction
# and its arithmetic, under the eleven protocols of multigpu_plan.md §3.
#
# 9 MEASURED ARMS + 3 untimed generator arms.  Projected wall 30-45 MINUTES on
# four GPUs, from mg1's MEASURED per-arm subprocess walls at the 512 cells
# (1 to 2.5 minutes, cold pass included -- multigpu_findings.md §1.4) times the
# nine arms, with the 384 cell cheaper than the 512 and the three generators at
# well under a minute each.  These are SMALL cells: the whole job is about a
# tenth of mg5's.  The walltime below is 02:00:00 -- headroom for a cold Triton
# compile per arm per count and for a shared node, not an estimate.
#
# WHAT THIS JOB IS FOR.  mg1 measured the realized forward view batch at 52 at
# EVERY device count at both 1024 cells.  That is not evidence of invariance: at
# those cells 8x the per-device sinogram exceeds the 2 GiB cap at n = 1, 2 AND
# 4, so the budget is pinned at the cap and the count-divided budget cannot
# express itself at all.  The 384- and 512-class cells are where the cap stops
# binding.  This job measures the batch there.
#
# THE MECHANISM UNDER TEST.  Projectors._transient_budget_bytes
# (projectors.py:231) scales the transient budget by the PER-DEVICE view shard,
# so the budget falls as 1/n between a 256 MB floor and the 2 GiB cap.  The
# per-view cost divides it: cone's forward cost reads num_rows_r out of the
# PARAMS (triton_cone.py:692) and does NOT shrink with the shard, while
# parallel's reads the call's column count (triton_parallel.py:483), which under
# sharding IS the slice band and does shrink.  Parallel at the 512 cell is the
# control.
#
# THE REGISTERED PREDICTION IS IN THE PYTHON FILE, computed from those cost
# functions before this job runs, and each arm recomputes it live and fails a
# check if the two disagree -- so the readout is a test of a stated prediction
# and not a fishing trip.  Registered: cone 384 -> 128/128/85, cone 512 ->
# 128/128/113, parallel 512 -> 128/128/128 at n = 1/2/4.  The same arithmetic
# reproduces mg1's MEASURED 52 at cone 1024 to the digit, which is why it is
# worth registering.
#
# THREE PLACES THE ARITHMETIC CONTRADICTS THE OBVIOUS READING, spelled out in
# the python docstring and repeated here because a reader of this file may never
# open that one:
#   * the batch does NOT fall as 1/n -- the shipped chunk of 128 floors it until
#     the budget cap dips below 128, which happens at n=4 ONLY;
#   * per-device launch counts do NOT grow with n -- they fall and then flatten
#     (cone 512: 4, 2, 2).  What grows is the SUMMED launch count across devices
#     (cone 4, 4, 8 against parallel's 4, 4, 4), which is the cap's real cost;
#   * a batch EVENT is not a launch -- _effective_view_batch is called once per
#     view-range CALL (projectors.py:382) and that call then loops
#     ceil(span/batch) times (projectors.py:386).  The launch counts printed are
#     DERIVED, and every row says so and names the assumption.
#
# NOTHING IS PINNED.  mg5 swept the forward view chunk by setattr on the kernel
# module; mg7 must NOT, and this is the design point most likely to be "fixed"
# wrongly by a later editor.  The question is what the SHIPPED constant realizes
# as the budget divides, so a pinned chunk would make the answer the harness's
# rather than the code's.  Every arm READS both chunk constants, asserts each is
# 128 on entry, and re-reads them after the timed passes to prove nothing moved
# them.  MG7_CHUNKS does not exist.
#
# PRODUCTION BODIES ONLY.  Every arm runs with MBIRTORCH_DISABLE_TRITON=0,
# asserts it in its own environment, and asserts a kernel body bound in BOTH
# directions on EVERY device.  There is no torch-body arm here -- that
# discriminator was mg5's question, not this one.
#
# STAGING, and the one dependency that is not obvious (protocol 11).  mg7
# IMPORTS mg1_readout.py -- for the region instrument, for the static
# view-batch probe (_view_batch_static) AND for the realized-batch observer
# (observe_view_batches) -- so BOTH FILES must be staged in this directory:
#     mg7_conebatch.py
#     mg1_readout.py
# The import failure names that remedy by file.  The import is deliberate and
# departs from mg2/mg3/mg4, which copied the machinery: mg7's whole deliverable
# is a realized view batch set against mg1's recorded 52, and a copied-then-
# drifted probe would produce a number that looks comparable and is not.  Every
# file is scp'd PER FILE and md5-verified; no sync happens into a tree a running
# job imports from, so do not touch mbirtorch_src while any nightly or trial job
# is live.
#
# ARTIFACTS.  One shared sinogram per GEOMETRY-CELL -- three of them, ~0.6 GB
# (384) and ~1.4 GB (512) each -- md5-verified by every arm that reads it, and
# deleted at the end unless MG7_KEEP_ARTIFACTS=1 is exported.
#
# SCHEDULING (protocol 11).  This is a four-GPU block, so it must clear the
# 02:00 jax nightly and the 03:00 torch nightly (itself up to four hours on a
# changed-tip night).  Submit outside that window, or add --begin.  At well
# under an hour this job fits comfortably between the windows.
#
# IF THIS JOB MUST BE SPLIT, split it by GEOMETRY and by nothing else.  Every
# reading here compares COUNTS within a coordinate, so the count axis may never
# cross a job boundary; MG7_COUNTS exists to re-run a single arm after a
# throttle flag, never to shard the job.
#     MG7_GEOMS=cone        the two cone cells, 6 arms
#     MG7_GEOMS=parallel    the control cell, 3 arms
# A cross-job comparison then pins the node, per the h010 precedent.  There is
# no shedding knob beyond that: nine arms is already the minimum that carries
# the question, and dropping a count would drop the comparison itself.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg7_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg7
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg7_%j.log
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
# The staging dependency, asserted before any GPU time is spent on it.
test -f /scratch/gautschi/buzzard/torch_p3/mg1_readout.py || { echo "mg1_readout.py is not staged beside mg7_conebatch.py; mg7 imports its region instrument, its static view-batch probe and its realized-batch observer"; exit 1; }

# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, and every arm re-asserts it in its own environment; mg7 has
# no torch-body arm, so a silent kernel decline would make every row vacuous.
export MBIRTORCH_DISABLE_TRITON=0
# Protocol 1: every arm pins through MBIRTORCH_NUM_DEVICES, which the SCRIPT
# pops and then sets per arm; an inherited value here would pin the whole job to
# one count without saying so.  The script asserts the per-arm value in the arm.
unset MBIRTORCH_NUM_DEVICES
# Protocol 6: the calibration mode owns torch.cuda.max_memory_allocated, and
# mg7's per-device peaks are read straight off that counter.  mg2 is the only
# job that sets it.
unset MBIRTORCH_MEMORY_CALIBRATION
# The two split/shed knobs, named here only so a re-run can narrow without
# editing the file.  Split by GEOMETRY; MG7_COUNTS is for re-running a single
# throttled arm, not for sharding the job.
unset MG7_GEOMS
unset MG7_COUNTS

export P0_TORCH_PYTHON=$TORCHPY
export MG7_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG7_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg7_conebatch.py
echo "MG7 DONE"
