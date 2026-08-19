#!/bin/bash
# mg38: THE CONE PIXEL-BATCHED FORWARD WITH TWO-AXIS GROUPING -- THE SPIKE.
# One H100.  The full statement is in mg38_cone_grouped.py.
#
# WHAT THIS RUN IS.  mg31 read the cone forward as gather-dominated: the
# vertical fan re-reads the whole flat recon about once per view, 175 GB of
# DRAM per 52-view launch, and nothing inside a view is worth reordering.
# Design note pfwd_segmented_design.md section 9 takes the only prize left,
# cross-view amortization: turn the kernel around so a work unit holds a
# (pixel tile x slice tile) block of the recon in registers and serves a
# CHUNK of views.  Turning it around makes the OUTPUT the moving target, and
# the two-axis grouping is the answer -- pixels sorted by (magnification
# bucket, channel center) give tiles whose detector footprint fits a small
# window, accumulated in registers and flushed once per view.  mg37, the
# paper check on the shipped builders, measured those windows closing at 99.8
# percent.  This job builds the kernel and measures it.  Still inside the
# harness; no library file is touched.
#
# WHAT IT DOES NOT DECIDE.  The spike picks no default and ships nothing.  A
# winning configuration goes to the composed re-gate, which re-measures end
# to end, because a kernel spike's win is not the driver's (the recorded
# lesson).
#
# WALL ESTIMATE.  Nine configurations over a three-point pixel ladder, with
# one values call, one discarded warm-up and three timed calls each, is 45
# variant calls at the full mask (the two subset points carry 1/64 and 1/128
# of the pixel work and cost minutes between them).  Budget three seconds a
# call at the widest window -- the design's own arithmetic puts the widest
# arm near 30x the shipped kernel's per-voxel work -- and that is about 135
# seconds.  Add the shipped baseline on the same ladder, the compound sort
# and gather inside every variant call at the tens-of-milliseconds class,
# the arithmetic preflight at seconds, up to 27 Triton compiles (three pixel
# counts x nine configurations) at tens of seconds each, and one ncu attempt
# the harness caps at 600 seconds: about 20 minutes.  The walltime below is
# 00:45:00.
#
# DEVICE MEMORY.  About 8 GB: the 3.11 GB values block, the seven
# per-(view, pixel) contract arrays at 64 views (1.38 GB) and their sorted
# copies (1.38 GB), two 0.26 GB output blocks (the variant's and the shipped
# reference's), and the permutation and odds and ends.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: the arithmetic preflight
# within 1e-5 of the torch body on both of its arms and each arm exercising
# the branch it exists for, every configuration's values within 1e-5 of the
# shipped wrapper at every ladder point, the expected body selection, and no
# configuration raising.  There is deliberately NO baseline anchor constant:
# no cone forward launch time for this cell exists in a form the harness can
# quote, so the shipped baseline is measured and printed and the reviewer
# compares it against mg31's rows.  The counter attempt never changes the
# exit code.  Nothing is chained to this job.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment, so
# it must not run at the same time as another torch_p3 job that installs into
# the same environment; chain with --dependency=afterany when one is queued.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg38_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_mg38
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg38_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
# This also does `module load cuda`, which is what puts Nsight Compute's ncu
# on PATH for the counter attempt.
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
command -v ncu >/dev/null 2>&1 && ncu --version | head -2

# The shipped configuration, and the assumption the baseline rests on: the
# CONE forward really binds the Triton kernel on this device.  Every speedup
# this spike reports divides by that baseline, and its values gate compares
# against that kernel's output, so a node that fell back to the torch body
# would report a meaningless run.
export MBIRTORCH_DISABLE_TRITON=0
"$TORCHPY" -c "
import numpy as np, torch, mbirtorch
cell = (8, 32, 20)
model = mbirtorch.ConeBeamModel(cell, np.linspace(0, 2*np.pi, cell[0], endpoint=False),
                                source_detector_dist=4.0*cell[2], source_iso_dist=2.0*cell[2])
model.configure_devices(devices=['cuda:0'])
model.set_params(no_warning=True, verbose=0)
fwd, back = model._view_batch_bodies()
print('cone forward body:', fwd.__name__)
assert fwd.__name__ == '_cone_forward_view_batch_triton', fwd.__name__
" || { echo "the cone forward does NOT bind the Triton kernel on this node; the spike's baseline and its values reference are that kernel, so the job stops here"; exit 1; }

unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG38_SMOKE
unset MG38_DRY
unset MG38_CONFIGS
unset MG38_NCU
unset MG38_REPEATS

export MG38_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG38_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG38_DRY=1
"$TORCHPY" -u mg38_cone_grouped.py || exit 1
unset MG38_DRY

"$TORCHPY" -u mg38_cone_grouped.py
rc=$?
echo "MG38 DONE rc=$rc"
exit $rc
