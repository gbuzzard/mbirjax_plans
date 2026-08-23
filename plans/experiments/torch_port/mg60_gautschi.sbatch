#!/bin/bash
# mg60: verify the cheaper input validation once, at the 1024-class parallel
# cell.  Times both formulations side by side and re-times initialize_recon,
# which findings section 1.48 measured at 2.61 s before the change.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 00:25:00
#SBATCH -J torch_mg60
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg60_%j.log
set -x
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export MPLBACKEND=Agg
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
# The library pins both compile caches itself now; leave them unset so this
# job runs with the shipped defaults.
unset TORCHINDUCTOR_CACHE_DIR
unset TRITON_CACHE_DIR
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
export MG60_OUT=/scratch/gautschi/buzzard/torch_p3/results/mg60_verify.json
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg60_verify.py
rc=$?
echo "MG60 DONE rc=$rc"
exit $rc
