#!/bin/bash
# kb4: attribute the kb2 sweep's e-3 back-path value readings on 1 H100 --
# eager body vs compiled body vs kernel, pairwise, at the two loudest shapes
# and one clean control.  See kb4_value_attrib.py.
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch kb4_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:45:00
#SBATCH -J torch_kb4
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/kb4_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u kb4_value_attrib.py
echo "KB4 DONE"
