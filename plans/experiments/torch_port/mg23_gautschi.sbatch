#!/bin/bash
# mg23: the band-padding remedy's gate run (increment 2 of
# back_remedy_design.md, ruled 2026-08-18).  Three legs on one node:
#
#   1  THE SUITE ON GPUS.  The full pytest suite with two H100s visible, so
#      the CUDA-gated kernel tests run for real -- including the new padded
#      value and stride tests in test_triton_cone.py and
#      test_triton_parallel.py, gated at 1e-5 relative against the torch
#      bodies and windowed references.
#   2  THE CONE RATES.  mg21b's band sweep re-run on the padded tree.  On the
#      unpadded tree it measured 2.44x between divisible and non-divisible
#      bands; the prediction here is 1.0x -- every band at the fast rate --
#      because the wrapper now rounds the band argument up before the launch.
#   3  THE PARALLEL RATES.  mg23_parallel_band.py, the same sweep for the
#      parallel back kernel (the design note's audit arm), with the same
#      1.0x prediction.
#
# Legs 2 and 3 each need one GPU; the job requests two for leg 1's
# multi-device arms and runs the legs serially.
#
# WALL ESTIMATE.  The suite ran 4 minutes on the nightly's node at -n 4; the
# two sweeps are minutes each (mg21b measured 2 to 3 minutes per run).  The
# walltime below is 01:30:00 for compile and queue margin.
#
# This job pip-installs into the shared torch_p0 environment, so it must not
# run at the same time as another torch_p3 job that installs.  mg24 (the
# composed confirmation) is submitted with --dependency=afterany on this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg23_gautschi.sbatch
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=28
#SBATCH -t 01:30:00
#SBATCH -J torch_mg23
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg23_%j.log

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# pytest is not part of the measurement environment's editable install; the
# suite leg needs it.  Idempotent and small.
"$TORCHPY" -m pip install --no-cache-dir -q pytest || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 2, torch.cuda.device_count()" || exit 1
# The tree must be the merged tip WITH the padding.  The helper existing and
# rounding 504 to 512 is the padding's own witness.
"$TORCHPY" -c "
from mbirtorch import _sharding
from mbirtorch._utils import padded_kernel_width
msg = 'the tree is not the padded merged tip this gate run was written for'
assert hasattr(_sharding, 'transfer_cylinder_batch'), msg
assert not hasattr(_sharding, 'broadcast_band_to_views'), msg
assert padded_kernel_width(504) == 512, msg
print('current tree: merged tip with the band padding confirmed')" || exit 1

export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_NUM_DEVICES
unset PYTHONPATH
unset MG21B_SMOKE
unset MG23_SMOKE

export MG21B_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
export MG23_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG23_RESULTS"

# Leg 1: the suite, from the source tree so pytest finds the test files.
cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -m pytest -q -p no:cacheprovider
suite_rc=$?
echo "MG23 SUITE rc=$suite_rc"

# Legs 2 and 3: the two band sweeps, from the job directory.
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
"$TORCHPY" -u mg21b_band_divisibility.py
cone_rc=$?
echo "MG23 CONE SWEEP rc=$cone_rc"
"$TORCHPY" -u mg23_parallel_band.py
par_rc=$?
echo "MG23 PARALLEL SWEEP rc=$par_rc"

rc=$(( suite_rc | cone_rc | par_rc ))
echo "MG23 DONE rc=$rc (suite=$suite_rc cone=$cone_rc parallel=$par_rc)"
exit $rc
