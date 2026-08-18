#!/bin/bash
# mg22: the full widening-floors refresh, with multiaxis measured on its own
# geometry and translation measured for the first time.
#
# WHY THIS JOB EXISTS.  The automatic device count consults a table of
# measured speed floors (mbirtorch/_widening_floors.py): the problem size at
# which each device count starts paying for itself.  Two of its rows were
# never measured on the geometry they govern, one geometry has no rows at
# all, and the rest of the table was measured against code that has since
# changed.  This job addresses all three.
#
# First, the multiaxis n=2 row was never measured on multiaxis.  It is a copy
# of the parallel n=2 row, kept so two devices behaved exactly as they did
# when MultiAxisParallelModel borrowed the parallel floors.  This job replaces
# it with a threshold measured on the geometry itself.
#
# Second, the multiaxis n=4 row is a sentinel: no admission size has been
# found for four devices on that geometry.  It was written from a single
# composed reconstruction, so it records no bracket below the cell it lost at.
# This job probes the 512-, 768- and 1024-class cells for an admission size.
#
# Third, TranslationModel declares no floor family at all, so the parallel
# floors govern its automatic device count today.  The refresh script names
# translation in UNTABLED_FAMILIES and measures its first rows here.  A
# translation scan does not resemble anything on the shared ladder, so the
# script gives it a ladder of its own: three production-anchored cells that
# keep a 16x16 translation grid and halve the detector and the spacing
# together.  The largest is the production cell mg18 measured.  Those rows
# read in sinogram elements like the other projection families, and their
# notes must name the cell so nobody reads a translation floor as a size on
# the shared ladder.
#
# The same run re-measures the existing parallel, cone and denoiser rows
# against the merged tree.  Several cost inputs moved on 2026-08-17: the
# banded multi-device forward was removed, the forward driver was renamed, the
# forward pixel batch default became 32768, and the combining slab became
# 256 MiB.  The staleness note therefore fires on every automatic device
# selection, and pasting this job's output is what clears it.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0 environment, so
# it must not run at the same time as another torch_p3 job that installs into
# the same environment.  When another such job is queued, submit this one with
# --dependency=afterany:<that job id>.
#
# WALL ESTIMATE.  The previous full refresh (mg16) fit in 90 minutes on four
# H100s.  The multiaxis sentinel probes are the expensive addition.  A
# 1024-class multiaxis composed reconstruction measured 394 s at two devices
# and 951 s at four (job 15307729), and the protocol is one cold pass plus
# three warm passes per arm.  The three arms at that one cell therefore run
# close to two hours by themselves.  Translation's cells are minutes each.
# The whole run should take about 4.5 to 5.5 hours; 7 hours is requested for
# margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 07:00:00
#SBATCH -J torch_mg22
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg22_%j.log
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
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The checkout must be the tip that removed the banded multi-device forward,
# because that is the code these floors are being measured against.  The
# cylinder transfer is present there and the banded broadcast is gone.
"$TORCHPY" -c "
from mbirtorch import _sharding
msg = 'the tree is not the merged banded-forward-removal tip this refresh prices'
assert hasattr(_sharding, 'transfer_cylinder_batch'), msg
assert not hasattr(_sharding, 'broadcast_band_to_views'), msg
" || exit 1

# The kernels are the production configuration the floors describe.
export MBIRTORCH_DISABLE_TRITON=0
# The script pins each projection arm through MBIRTORCH_NUM_DEVICES in its
# own subprocess and places each denoiser arm with configure_devices; a value
# inherited from this shell would collapse every arm to one count.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION

# The staged inputs and per-arm rows go to scratch where they can be
# inspected after the run, rather than into the checkout.
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg22_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The arm list first, so the log opens with the plan, the translation ladder,
# the sentinel and first-measurement NOTE lines, and the metric reminders.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py
rc=$?
echo "MG22 DONE rc=$rc"
exit $rc
