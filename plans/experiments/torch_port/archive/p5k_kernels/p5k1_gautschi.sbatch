#!/bin/bash
# K1 validation: the cone back Triton kernel's CUDA test file on 1 H100 —
# the first real Triton compile of the kernel (the local run used a CPU
# emulator; the compile-time risks are enumerated in the K1 report).
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch p5k1_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_p5k1
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/p5k1_%j.log

set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg
export MBIRTORCH_ENABLE_TRITON=1

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest tests/test_triton_cone.py -v 2>&1 | tail -40
"$TORCHPY" -c "
import numpy as np, mbirtorch
from mbirtorch.kernel_availability import cone_back_kernel_usable
m = mbirtorch.ConeBeamModel((8, 8, 8), np.linspace(0, 6.28, 8, endpoint=False),
                            source_detector_dist=32, source_iso_dist=16)
m.set_params(no_warning=True, verbose=0)
print('USABLE:', cone_back_kernel_usable(m))
"
echo "P5K1 DONE"
