#!/bin/bash
# mg61: whether compiling the six update frames earns its tracing cost, at
# the 1024-class parallel cell.  Follows mg59, which named the frames and
# showed that making their shapes dynamic removes graphs but not seconds.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH -t 01:30:00
#SBATCH -J torch_mg61
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg61_%j.log
set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
# The library pins both compile caches itself; leave them unset so this runs
# with the shipped defaults, which is the state a user's later process sees.
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
export MG61_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg61_compile_worth_15iter.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg61_compile_worth.py
rc=$?
echo "MG61 DONE rc=$rc"
exit $rc
