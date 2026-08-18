#!/bin/bash
# mg21b: one variable -- the cone back kernel's band length -- swept across
# values divisible by 16 and not, on ONE device.  The full statement is in
# mg21b_band_divisibility.py.
#
# WHY.  mg21 found the Triton kernel itself is what grows when a fourth device
# joins the cone back projection at the 2048 class (24.3 s -> 45.8 s on the
# busiest device, about 2.5x slower per unit of work), with the per-band-call
# costs flat.  The three-device band is 672 slices (divisible by 16) and the
# four-device band is 504 (not), and Triton specializes a kernel on whether an
# integer argument is divisible by 16 -- the exact mechanism mg20 measured on
# the parallel forward kernel's width.  mg21's arms move band and device count
# together, so this job varies the band alone: same device, same sinogram,
# same pixels, same views, eight bands interleaved divisible / not.
#
# PREDICTION (recorded before the run): divisible bands share one rate,
# non-divisible bands run about 2 to 2.5x slower, and 496 / 504 / 512 land on
# opposite sides of the boundary though their work differs by under 2 percent.
#
# COST.  One GPU.  Eight arms, each one warmup (pays that band's Triton
# compile) plus three timed calls of seconds each; the whole job is minutes.
# The walltime below is 01:00:00 for queue-friendly margin.
#
# NO INSTALL.  The torch_p0 environment's editable install already points at
# mbirtorch_src, and mg22 (running when this was submitted) reinstalls the
# same tree at its start; a second concurrent install would race it (the torn
# .pth failure in the cluster notes).  This job only imports.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg21b_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 01:00:00
#SBATCH -J torch_mg21b
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg21b_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -c "import mbirtorch; print('library under test:', mbirtorch.__file__)" || exit 1
# The merged-tip witness, as in mg21: the cylinder transfer is present and the
# banded-forward names are gone.
"$TORCHPY" -c "
from mbirtorch import _sharding
msg = 'the tree is not the merged banded-forward-removal tip this probe was written for'
assert hasattr(_sharding, 'transfer_cylinder_batch'), msg
assert not hasattr(_sharding, 'broadcast_band_to_views'), msg
print('current tree: merged tip confirmed')" || exit 1

export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset MG21B_SMOKE

export MG21B_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG21B_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

"$TORCHPY" -u mg21b_band_divisibility.py
rc=$?
echo "MG21B DONE rc=$rc"
exit $rc
