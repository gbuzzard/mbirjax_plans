#!/bin/bash
# mg16: the full widening-floors refresh, including the FIRST denoiser rows.
#
# WHY THIS JOB EXISTS.  The automatic device count consults a table of
# measured speed floors (mbirtorch/_widening_floors.py): the problem size at
# which each device count starts paying for itself.  Two debts have
# accumulated against that table.  First, three of its cost inputs
# (_sharding.py, projectors.py, triton_cone.py) have changed since the floors
# were last measured, so the staleness note fires on every automatic device
# selection.  Second, QGGMRFDenoiser is about to come under the automatic
# policy, and its floor family ('denoiser') has never been measured -- the
# refresh script names it in UNTABLED_FAMILIES and probes its first admission
# sizes here.  One run pays both debts: it re-measures the four existing
# parallel/cone rows against the current code and gives the denoiser family
# its first rows, and the paste that follows clears the staleness note.
#
# The denoiser family differs from the projection families in three ways the
# script documents: its staged input is a noisy phantom rather than a forward
# projection, its timed call is denoise rather than recon, and its device
# count is set with configure_devices rather than the environment pin.  Its
# floors are read in IMAGE VOXELS -- the denoiser's sinogram shape IS its
# image shape -- and the pasted rows must say so in their notes.
#
# SCHEDULING.  This job is submitted with --dependency=afterok:<mg15 job id>:
# mg15 is the padding-removal verification, and its exit code is its verdict,
# so this measurement only starts on a split that mg15 has just verified on
# the same hardware.  If mg15 fails its gates, this job never runs.
#
# WALL ESTIMATE.  The previous full refresh (four rows, 2026-08-13) fit in
# 90 minutes on four H100s.  The denoiser adds nine arms (three probe cells
# at one, two and four devices) plus three phantom generators; a denoise
# sweep has no projections, so each arm is minutes at most.  02:30:00 covers
# both with margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:30:00
#SBATCH -J torch_mg16
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg16_%j.log
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

# The kernels are the production configuration the floors describe.
export MBIRTORCH_DISABLE_TRITON=0
# The script pins each projection arm through MBIRTORCH_NUM_DEVICES in its
# own subprocess and places each denoiser arm with configure_devices; a value
# inherited from this shell would collapse every arm to one count.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION

# The staged inputs and per-arm rows go to scratch where they can be
# inspected after the run, rather than into the checkout.
export REFRESH_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg16_floors
mkdir -p "$REFRESH_RESULTS"

cd /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
# The arm list first, so the log opens with the plan, the denoiser NOTE
# lines, and the image-voxel metric reminder.
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py --plan || exit 1
"$TORCHPY" -u dev_scripts/refresh_widening_floors.py
rc=$?
echo "MG16 DONE rc=$rc"
exit $rc
