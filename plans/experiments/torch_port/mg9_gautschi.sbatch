#!/bin/bash
# mg9: THE ONE MEASUREMENT THE FORWARD-REMEDY MEMO SAYS TO BUY FIRST, on four
# H100s.  The full statement of what is measured and why is in
# mg9_fwd_instrument.py; the essentials are repeated here because a reader of
# this file may never open that one.
#
# WHY THIS JOB EXISTS.  Adding GPUs does not make the forward projection
# faster.  Its per-device time holds at 32.18, 30.61 and 30.48 s on cone 1024 at
# one, two and four devices, and at 28.87 and 28.75 s on parallel 1024 at one and
# two.  Two different things could cause that, and nobody has measured which:
# either the devices are WAITING -- on the band copies, or on the single Python
# thread that issues them -- or they are BUSY the whole time, paying a cost per
# kernel launch that does not shrink when each device's share of the work
# shrinks.  The two fixes are different, and one is expensive, so the decision
# memo rules: measure before choosing.
#
# WHAT THE INSTRUMENT ADDS.  Three readings per device per reconstruction.
#   1. The existing bracket -- the CUDA event pair around the whole forward
#      projection region -- unchanged, so these numbers are comparable with the
#      ones above.
#   2. NEW: busy time.  An event pair around EACH individual projection body
#      call, summed, plus the call count.  Bracket minus busy is the time the
#      device spent not computing.
#   3. NEW: the band broadcast.  Host wall around each fan-out call, and a
#      device-side event pair around each individual copy inside it.
# If busy stays close to the bracket and flat as devices are added, the cost is
# per launch.  If the gap or the broadcast grows with the count, it is
# serialization.  The script prints both signatures and the numbers beside them
# and prints NO verdict; the attribution is made in analysis, as it was for mg4
# and mg5.
#
# THE ARMS.  Four measured arms plus two untimed sinogram generators, all at
# (1024, 1008, 992):
#     parallel at 1, 2 and 4 devices        (the four-device point has never
#                                            been measured; the memo's §6
#                                            follow-ups ask for it)
#     cone at 2 devices
# The one-device reference runs first, so a truncated job still yields it.
# Everything else -- phantom, shared md5-verified sinogram, weights, seed, three
# VCD iterations, one discarded cold pass, three timed reconstructions, and the
# SHIPPED forward view chunk of 128 -- is mg5's arm construction unchanged, which
# is what makes the spans line up with the memo's numbers.  This job moves no
# view chunk; it reads both constants and asserts the forward one is still 128.
#
# WALL ESTIMATE.  mg5 measured 2 to 5 minutes of subprocess wall per arm at
# these 1024 cells, cold pass included.  Four arms is 8 to 20 minutes; the two
# generators are one forward projection plus a multi-GB md5 each, call it 6 to 10
# minutes; six subprocess starts each pay a torch import, ~2 minutes.  Total 16
# to 32 minutes.  The walltime below is 02:00:00, roughly four times the upper
# estimate, which also covers a cold Triton cache and a shared node.
#
# DISK.  Two shared sinograms at the 1024 cell, about 4 GB each, in
# $MG9_RESULTS during the run, deleted at the end unless MG9_KEEP_ARTIFACTS=1
# is exported.
#
# THE TREE THIS JOB IMPORTS, and the reason it is not the usual one.  mg9 runs
# out of /scratch/gautschi/buzzard/torch_p3/mbirtorch_mg9, a fresh export the
# session lead ships.  It must NEVER reference mbirtorch_src: a job (mg8) is
# running out of that tree right now.
#   That is also why this file does NOT `pip install -e` its tree, which is what
# mg5 and mg8 do.  An editable install rewrites a path pointer inside the SHARED
# environment at torch_p0/env, so installing mbirtorch_mg9 there would repoint
# the running mg8 job's imports mid-flight.  mg9 puts its tree on PYTHONPATH
# instead, which changes nothing outside this job's own processes, and then
# ASSERTS that the mbirtorch it imported really came from the mg9 tree.  If that
# assert fails, an editable install of some other tree is winning through a
# meta-path finder; do NOT fix it with pip while mg8 is live -- wait for mg8 to
# finish and resubmit.
#
# IF THIS JOB MUST BE SPLIT, split it by ARM and keep the whole parallel count
# series together: the entire reading compares one geometry across device
# counts, so that axis may never cross a job boundary.
#     MG9_ARMS=parallel1,parallel2,parallel4
#     MG9_ARMS=cone2
# A cross-job comparison then pins the node.
#
# SCHEDULING.  This is a four-GPU block, so it must clear the 02:00 and 03:00
# nightlies.  At well under an hour it fits comfortably between them.  It shares
# the node with nothing it needs.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg9_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:00:00
#SBATCH -J torch_mg9
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg9_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
MG9_TREE=/scratch/gautschi/buzzard/torch_p3/mbirtorch_mg9
# Its OWN compiler cache directory.  mg8 is running with
# TORCHINDUCTOR_CACHE_DIR=.../torch_cache, and two concurrent jobs sharing one
# inductor cache is a recorded way to get a corrupt artifact and a mystery
# crash.
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache_mg9
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
export MPLBACKEND=Agg

# The import path, in place of an editable install (see the header).  The arm
# subprocesses inherit this, so every arm imports the same tree.
export PYTHONPATH=$MG9_TREE${PYTHONPATH:+:$PYTHONPATH}

nvidia-smi -L
test -d "$MG9_TREE/mbirtorch" || { echo "the mg9 tree is not staged at $MG9_TREE; ship it before submitting (see SHIP.md)"; exit 1; }
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
# THE PRELUDE ASSERT, in two halves.  The first is mg8's: the tree imports and
# four GPUs are visible, checked before any GPU time is spent.  The second is
# this job's own and matters more: the mbirtorch that got imported must be the
# mg9 tree and not some other tree an editable install is pointing at.
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1
export MG9_TREE
"$TORCHPY" -c "
import os, sys, mbirtorch
tree = os.path.realpath(os.environ['MG9_TREE'])
got = os.path.realpath(mbirtorch.__file__)
if not got.startswith(tree + os.sep):
    sys.exit('mbirtorch imported from ' + got + ', NOT from the mg9 tree ' + tree +
             '.  An editable install of another tree is winning through a '
             'meta-path finder.  Do NOT fix this with pip while mg8 is running '
             'out of mbirtorch_src -- wait for mg8 to finish, then resubmit.')
print('mbirtorch from', got)
" || exit 1
# The two names this job must reach through the driver.  An older checkout would
# import cleanly and then fail every arm one subprocess at a time.
"$TORCHPY" -c "
from mbirtorch import _sharding
from mbirtorch.projectors import Projectors
from mbirtorch.tomography_model import TomographyModel
assert hasattr(_sharding, 'broadcast_band_to_views')
assert hasattr(Projectors, 'sparse_forward_project_view_range')
assert hasattr(TomographyModel, '_sparse_forward_project_sharded')
print('the three names mg9 instruments are present')
" || { echo "this mbirtorch does not carry the names mg9 wraps (broadcast_band_to_views / sparse_forward_project_view_range / _sparse_forward_project_sharded); stage the f985a6e export into $MG9_TREE"; exit 1; }
# Every python file in the tree must parse, checked on THIS node before any
# GPU time.  The login node's filesystem client has intermittently returned
# truncated reads for files whose bytes are correct on disk; a module the arms
# import lazily would otherwise fail mid-job instead of here.
"$TORCHPY" -m compileall -q "$MG9_TREE" || { echo "a file in $MG9_TREE does not parse on this node; re-verify the tree against its manifest"; exit 1; }

# The kernels are the shipped production configuration.  Explicitly OFF rather
# than merely unset, because the measurement must be taken in the shipped
# environment and not in a variant of it; every arm re-asserts it in its own
# environment.
export MBIRTORCH_DISABLE_TRITON=0
# Each arm pins its own device count, which the SCRIPT pops and then sets per
# subprocess.  A value inherited from here would pin the whole job to one count
# without saying so.  Each arm then asserts the device list it actually got.
unset MBIRTORCH_NUM_DEVICES
# The calibration mode RESETS torch.cuda.max_memory_allocated, a counter this
# job reads for its per-device peaks.  mg9 never sets it, here or anywhere.
unset MBIRTORCH_MEMORY_CALIBRATION
# The arm-narrowing and artifact knobs, named here only so a re-run needs no
# edit to the file.  Split by ARM; keep the parallel count series together.
unset MG9_ARMS
unset MG9_KEEP_ARTIFACTS
unset MG9_SMOKE

export P0_TORCH_PYTHON=$TORCHPY
# The flat run directory: the jsonl lands beside the script, which is the
# convention the rows are collected under.
export MG9_RESULTS=/scratch/gautschi/buzzard/torch_p3
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the six arms and a mis-set
# environment variable is caught before any GPU work.
"$TORCHPY" -u mg9_fwd_instrument.py --dry-run || exit 1
"$TORCHPY" -u mg9_fwd_instrument.py
echo "MG9 DONE"
