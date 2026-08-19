#!/bin/bash
# mg36: SINGLE-PROJECTION ANCHORS AND TABLE 1'S 1/128 ROW.
# One H100.  The full statement is in mg36_single_projection.py.
#
# WHAT THIS RUN IS.  The kernels document is being revised to end each
# kernel section with the cost of one full projection at the 1024-class
# example, and to add the 1/128 pixel row to its per-launch table.  No
# record holds those numbers, so this job measures them: whole-call
# forward and back projections for both geometries (parallel forward on
# both routes), and one 128-view launch at the 1/128 subset on both
# routes, all warm medians.
#
# WALL ESTIMATE.  Kernel compiles at seconds each; the timed region is
# five whole-call projections (the slowest about 8 s a call, four calls
# each) plus millisecond-scale subset launches: under ten minutes with
# the install and model builds.  The walltime below is 00:30:00.
#
# DEVICE MEMORY.  About 10 GB at peak: the 2.9 GB values block, the
# 3.8 GB sinogram, the 2.9 GB back output, and footprint transients.
#
# THE EXIT CODE REPORTS INSTRUMENT HEALTH ONLY: the two forward routes
# agree at 1e-5 (whole-call and the 1/128 launch), the full mask
# realizes 771,240 pixels, and no call raises.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment; chain with
# --dependency=afterany when one is queued.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg36_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:30:00
#SBATCH -J torch_mg36
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg36_%j.log
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

# The assumption every timing rests on: both geometries really bind
# their Triton forward wrappers on this device.
"$TORCHPY" -c "
import numpy as np, mbirtorch
cell = (8, 24, 20)
m = mbirtorch.ParallelBeamModel(cell, np.linspace(0, np.pi, cell[0], endpoint=False))
m.configure_devices(devices=['cuda:0']); m.set_params(no_warning=True, verbose=0)
fwd, _ = m._view_batch_bodies()
assert fwd.__name__ == '_parallel_forward_view_batch_triton', fwd.__name__
c = mbirtorch.ConeBeamModel(cell, np.linspace(0, 2*np.pi, cell[0], endpoint=False), source_detector_dist=80.0, source_iso_dist=40.0)
c.configure_devices(devices=['cuda:0']); c.set_params(no_warning=True, verbose=0)
cfwd, _ = c._view_batch_bodies()
assert cfwd.__name__.endswith('_triton'), cfwd.__name__
print('both forward wrappers bind triton')
" || { echo "a forward wrapper does NOT bind its Triton kernel on this node; the anchors would time the wrong body, so the job stops here"; exit 1; }

unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_SORTED_FORWARD
unset MBIRTORCH_DISABLE_TRITON
unset PYTHONPATH
unset MG36_SMOKE
unset MG36_DRY
unset MG36_REPEATS

export MG36_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG36_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

export MG36_DRY=1
"$TORCHPY" -u mg36_single_projection.py || exit 1
unset MG36_DRY

"$TORCHPY" -u mg36_single_projection.py
rc=$?
echo "MG36 DONE rc=$rc"
exit $rc
