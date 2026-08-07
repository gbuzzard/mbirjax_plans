#!/bin/bash
# kb1: the CUDA battery re-run for kernel-aware view batching — both Triton
# test files on 1 H100, including the new driver-level batching tests (the
# chunked view-range loop against a single-batch reference, and the bound
# cost model's realized batches).  Baseline before this change: 68 passed
# (job 14921048); this run adds 4 CUDA tests.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch kb1_gautschi.sbatch
# Env note: pass any env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c — slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_kb1
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/kb1_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest tests/test_triton_cone.py tests/test_triton_parallel.py -v 2>&1 | tail -60
"$TORCHPY" -c "
import numpy as np, mbirtorch
from mbirtorch import kernel_availability as ka
cone = mbirtorch.ConeBeamModel((8, 8, 8), np.linspace(0, 6.28, 8, endpoint=False),
                               source_detector_dist=32, source_iso_dist=16)
cone.set_params(no_warning=True, verbose=0)
par = mbirtorch.ParallelBeamModel((8, 8, 8), np.linspace(0, 3.14, 8, endpoint=False))
par.set_params(no_warning=True, verbose=0)
print('USABLE cone back   :', ka.cone_back_kernel_usable(cone))
print('USABLE cone fwd    :', ka.cone_forward_kernel_usable(cone))
print('USABLE parallel back:', ka.parallel_back_kernel_usable(par))
print('USABLE parallel fwd :', ka.parallel_forward_kernel_usable(par))
"
echo "KB1 DONE"
