#!/bin/bash
# mg39: THE SORTED ORDER FED TO THE UNCHANGED SHIPPED CONE FORWARD.
# One H100.  The full statement is in mg39_cone_sorted_order.py.
#
# WHAT THIS RUN IS.  Design note pfwd_segmented_design.md section 9
# records Greg's decision after the mg38 loss: rework candidate A
# first.  This job feeds the two-axis sorted pixel order (32 W_p_r
# buckets, then channel center) to the SHIPPED per-tap cone forward
# kernel -- no kernel change at all -- against the raster-order
# baseline, per whole batch and per 8/16-view chunk, with a
# channel-only ablation arm.  A bounded profiler pass on the winner
# reads the locality mechanism against mg31's L1-hit 41.7 percent.
#
# WALL ESTIMATE.  The baseline full-mask call is about 0.5 s and the
# arms add sort and permute milliseconds: four timed calls per (arm,
# ladder point) over four arms and three points is a few minutes,
# plus the model build, one compile, and one bounded ncu pass.  The
# walltime below is 00:30:00.
#
# DEVICE MEMORY.  About 10 GB: the 2.9 GB values block, its permuted
# copy, the 64-view output slab, and the contract transients.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: every arm's values
# within 1e-5 of the raster baseline, the full mask realizing 771,240
# pixels, the Triton wrapper bound, and no arm raising.  The ncu pass
# never gates.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another
# torch_p3 job that installs into the same environment; chain with
# --dependency=afterany when one is queued.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg39_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:30:00
#SBATCH -J torch_mg39
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg39_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# The assumption every timing rests on: the cone forward really binds
# its Triton wrapper on this device.
"$TORCHPY" -c "
import numpy as np, mbirtorch
cell = (8, 24, 20)
m = mbirtorch.ConeBeamModel(cell, np.linspace(0, 2*np.pi, cell[0], endpoint=False), source_detector_dist=80.0, source_iso_dist=40.0)
m.configure_devices(devices=['cuda:0']); m.set_params(no_warning=True, verbose=0)
fwd, _ = m._view_batch_bodies()
assert fwd.__name__.endswith('_triton'), fwd.__name__
print('cone forward binds triton')
" || { echo "the cone forward does NOT bind the Triton kernel on this node; the arms would time the torch body, so the job stops here"; exit 1; }

command -v ncu >/dev/null 2>&1 || \
    export PATH="$PATH:/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
command -v ncu >/dev/null 2>&1 && ncu --version | head -2

unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_SORTED_FORWARD
unset MBIRTORCH_DISABLE_TRITON
unset PYTHONPATH
unset MG39_SMOKE
unset MG39_DRY
unset MG39_ARMS
unset MG39_NCU
unset MG39_NCU_INNER
unset MG39_REPEATS

export MG39_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG39_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG39_DRY=1
"$TORCHPY" -u mg39_cone_sorted_order.py || exit 1
unset MG39_DRY

"$TORCHPY" -u mg39_cone_sorted_order.py
rc=$?
echo "MG39 DONE rc=$rc"
exit $rc
