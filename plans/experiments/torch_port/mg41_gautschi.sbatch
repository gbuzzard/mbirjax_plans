#!/bin/bash
# mg41: what one production reconstruction costs in each library, on one GPU.
#
# WHY THIS JOB EXISTS.  The open-items list asks for a production-shaped
# comparison of the two libraries: time and peak device memory for a
# 15-iteration VCD reconstruction with nontrivial weights, at the
# 1024-class cell, on one GPU, for both geometries.  Every cross-framework
# number recorded so far comes from a 3-iteration reconstruction, which is
# a good ruler for comparing device counts and is not what a person runs.
# This job measures the four rows that answer the plain question, and it
# does NOT replace the 3-iteration campaign tables; those stay as they
# are, and this is a separate reading at a separate iteration count.  Both
# libraries reconstruct the same staged sinograms, reused by md5 from the
# reference-timing runs, so the two columns differ only in the library.
#
# SCHEDULING.  This job pip-installs into the shared torch_p0
# environment, so it must not run at the same time as another torch_p3
# job that installs into the same environment.  Submit it with
# --dependency=afterany:<job id> behind any such job already queued.
#
# WALL ESTIMATE.  A 15-iteration warm pass is close to five times the
# recorded 3-iteration time, which puts mbirtorch parallel near 106 s,
# mbirtorch cone near 308 s, mbirjax parallel near 129 s, and mbirjax
# cone near 314 s.  Each variant runs four passes (one cold, three warm),
# so the passes alone are about 57 minutes; with the per-variant compiles
# and the staging check the job is roughly 75 minutes.  Three hours is
# requested for margin.
#
# DISK.  The two staged sinograms are REUSED from the reference-timing
# run's directory and are about 4.1 GB each; nothing new is written there
# unless one is missing.  This job writes one jsonl of rows.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH -n 14
#SBATCH -t 03:00:00
#SBATCH -J torch_mg41
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg41_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
JAXPY=$(conda run -n mbirjax which python)
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L

# ── the mbirtorch side ───────────────────────────────────────────────────
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# The editable install points the shared torch_p0 environment at this tree.
# NEVER run two such jobs at once: a second concurrent install races this
# one and leaves a torn .pth.  Chain instead, with --dependency=afterany.
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 1, torch.cuda.device_count()" || exit 1

# The checkout must carry both kernel changes the recorded columns
# describe: the width padding (504 -> 512 is the production band it exists
# for) and the channel-sorted parallel forward, whose enable switch only
# exists on the sorted tree.  Without them this job would price a
# different implementation than the one the tables are about.
"$TORCHPY" -c "
from mbirtorch._utils import padded_kernel_width
from mbirtorch.triton_parallel import sorted_forward_enabled
msg = 'the tree is missing a kernel change these timings describe'
assert padded_kernel_width(504) == 512, msg
assert sorted_forward_enabled(), msg
" || exit 1

# ── the mbirjax side ─────────────────────────────────────────────────────
"$JAXPY" -c "import jax, mbirjax; assert jax.devices()[0].platform == 'gpu'; print('mbirjax under test:', mbirjax.__file__)" || exit 1

# The kernels are the production configuration these timings describe, and
# the sorted route must take its shipped default.
export MBIRTORCH_DISABLE_TRITON=0
unset MBIRTORCH_SORTED_FORWARD
# The script pins every torch variant through MBIRTORCH_NUM_DEVICES in its
# own subprocess; a value inherited from this shell would reach a job that
# asked for something else.  The calibration mode owns the peak counters
# this run reads itself.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset PYTHONPATH
unset MG41_SMOKE
unset MG41_DRY
unset MG41_VARIANTS
unset MG41_REPEATS
unset MG41_SINO_DIR

export MG41_RESULTS=/scratch/gautschi/buzzard/torch_p3/results/mg41_production
mkdir -p "$MG41_RESULTS"
# One interpreter per library, because the two do not share an environment
# here.  The driver picks between them per variant.
export MG41_TORCH_PYTHON=$TORCHPY
export MG41_JAX_PYTHON=$JAXPY
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The plan first, so the log opens with the six jobs in run order and a
# mis-set variable is caught before any GPU work.
export MG41_DRY=1
"$TORCHPY" -u mg41_production_compare.py || exit 1
unset MG41_DRY

# The run.  Its exit code reports whether the INSTRUMENTS worked, not what
# they measured; the comparison table is in this log.
"$TORCHPY" -u mg41_production_compare.py
rc=$?
echo "MG41 DONE rc=$rc"
exit $rc
