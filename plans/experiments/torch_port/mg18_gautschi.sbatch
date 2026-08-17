#!/bin/bash
# mg18: THE BANDED WALK AGAINST THE COLUMN GATHER, on four H100s, for the two
# geometries that still run the banded walk: translation and multiaxis
# parallel.  The full statement is in mg18_column_gather_ab.py; the essentials
# are repeated here because a reader of this file may never open that one.
#
# WHERE THIS SITS.  mbirtorch has two drivers for a multi-device forward
# projection.  The banded walk visits the reconstruction's slice-owners in turn
# and makes one projector call per band.  The column gather walks the pixel
# axis in batches and makes one call per batch over the whole slice range.
# Cone beam and parallel beam measured the two against each other, the gather
# won the composed reconstruction by 1.2x to 1.6x, and both classes now declare
# column_gather_geometry = True.  TranslationModel and MultiAxisParallelModel
# still run the banded walk.  The recorded rule is that a geometry switches
# only on its OWN measurement, and nobody has measured these two.  This job is
# that measurement.
#
# NOTHING IS FLIPPED HERE.  Each arm forces the driver it wants by setting
# column_gather_geometry and forward_project_pixel_batch on its own model
# INSTANCE, before the device layout settles, and then records which driver the
# model reports it will run.  No library file is touched and no default moves.
#
# THE ARMS.  Twelve per geometry, each in its own process: a one-device anchor
# that stages the phantom and the reference sinogram; six forward arms (banded,
# gather at 8192, gather at 32768, at two and four devices); four composed
# three-iteration reconstructions (banded and gather at two and four devices);
# and a repeat of the first banded arm at the end as the cross-time drift
# witness.  Values are gated at 1e-3 against the one-device reference BEFORE
# anything is timed.  24 arms in total.
#
# THE CELLS are mg8's, built the same way, so these rows sit beside mg8's:
#   multiaxis   (1024, 1008, 992)  -> recon (992, 992, 1148), 771,240 pixels
#   translation ( 256, 1900, 3000) -> recon (118, 360,  240),  42,480 pixels
#
# WALL ESTIMATE, from mg8's measured three-iteration composed walls on these
# same two cells (slurm job mg8, four H100s):
#     multiaxis    347.5 s at one device,  500.4 at two, 1179.7 at four
#     translation   57.4 s at one device,  105.4 at two,  112.7 at four
#
#   * COMPOSED ARMS, 8 of them.  Priced at the banded wall for both drivers of
#     each pair -- the conservative reading, since the gather is the thing
#     under test -- multiaxis costs 2 x 500.4 + 2 x 1179.7 = 3360 s and
#     translation costs 2 x 105.4 + 2 x 112.7 = 436 s, so 63 min.  If the
#     gather wins by the 1.2x to 1.6x cone and parallel measured, the same set
#     lands near 55 min.  Take 55-63 min.
#   * FORWARD ARMS, 16 of them (the two anchors and the fourteen multi-device
#     forward arms).  Each pays one cold pass through the public entry point
#     with compilation included, one discarded warm-up, three timed passes, and
#     the md5 and staging of a multi-gigabyte output.  At about four minutes
#     each that is 65 min worst case.
#   * GENERATORS AND SUBPROCESS OVERHEAD.  Two seeded phantoms (one of them
#     4.2 GiB to draw and write), two reference projections, and 24 subprocess
#     starts each paying a torch import and a CUDA context: 20 min.
# Total about 2.3 h.  The walltime below is 04:30:00, close to twice that.
#
# DISK.  About 95 GiB in $MG18_RESULTS, and it is NOT deleted at the end: the
# staged outputs are what a later arm compares against, and re-running a subset
# through MG18_ARMS depends on them still being there.
#     multiaxis    4.2 GiB phantom + 3.8 GiB reference + 7 x 3.8 GiB forward
#                  outputs + 4 x 4.2 GiB reconstructions   = 51 GiB
#     translation  0.04 + 5.4 + 7 x 5.4 + 4 x 0.04         = 44 GiB
# Remove them by hand when the rows have been read.
#
# NO GATES ON THE EXIT CODE.  It reports INSTRUMENT HEALTH only: every arm ran
# and was recorded, every artifact staged, every md5 verified, and no arm ran
# on the wrong device count or the wrong driver.  A values gate failure and a
# memory preflight refusal are both printed in full and neither changes the
# exit code -- they are findings.  The verdict, which is whether either
# geometry should switch its default, is read by a person from the table in
# this log.  Nothing is chained to this job.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg18_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 04:30:00
#SBATCH -J torch_mg18
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg18_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

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
# THE ASSUMPTION THIS JOB RESTS ON: in the checkout being measured, translation
# and multiaxis still run the banded walk by default, and cone and parallel
# already run the gather.  A checkout where translation or multiaxis had
# already flipped would import cleanly, run every arm, and compare the gather
# against itself on the arms labelled banded.  The harness would catch that
# through its per-arm driver check, but only after hours of GPU time, so it is
# checked here first.
"$TORCHPY" -c "
import mbirtorch
banded = {'TranslationModel': mbirtorch.TranslationModel,
          'MultiAxisParallelModel': mbirtorch.MultiAxisParallelModel}
gather = {'ConeBeamModel': mbirtorch.ConeBeamModel,
          'ParallelBeamModel': mbirtorch.ParallelBeamModel}
for name, cls in banded.items():
    assert cls.column_gather_geometry is False, (
        name + ' already declares column_gather_geometry True, so its banded '
        'arms would not be banded')
for name, cls in gather.items():
    assert cls.column_gather_geometry is True, (
        name + ' no longer declares column_gather_geometry True; this is not '
        'the tree the cone and parallel measurement was made on')
print('current tree: translation and multiaxis still banded by default, '
      'cone and parallel already gathered')" || { echo "the tree in mbirtorch_src is not the one mg18 was written for"; exit 1; }

# The shipped configuration, set explicitly.  This is NOT the variable under
# test: MBIRTORCH_DISABLE_TRITON turns off the hand-written kernels, and
# neither of these two geometries HAS a hand-written kernel for it to affect --
# each arm verifies exactly that and records it.  It is pinned to the shipped
# value so the job varies one thing only.
export MBIRTORCH_DISABLE_TRITON=0
# This job reads torch.cuda.max_memory_allocated itself.  The calibration mode
# owns and resets those counters, so it must be off; each arm asserts its
# absence.  Popped again per subprocess.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins its own device count through this variable, set per
# subprocess.  A value inherited from the shell would pin every arm at once.
unset MBIRTORCH_NUM_DEVICES
# The process-wide column-gather override.  It would overrule the per-instance
# forcing every arm here depends on, in either direction.  Popped again per
# subprocess.
unset MBIRTORCH_FORWARD_COLUMN_GATHER
unset PYTHONPATH
unset MG18_SMOKE
unset MG18_GEOMS
unset MG18_ARMS

export MG18_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG18_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1

# The arm plan first, so the log opens with the 24 arms and their cells, and a
# mis-set variable is caught before any GPU work.  The dry run imports no torch
# and touches no device.
export MG18_DRY=1
"$TORCHPY" -u mg18_column_gather_ab.py || exit 1
unset MG18_DRY

# The measurement.  Its exit code reports whether the INSTRUMENTS worked, not
# what they found; the finding is in the table in this log.
"$TORCHPY" -u mg18_column_gather_ab.py
rc=$?
echo "MG18 DONE rc=$rc"
exit $rc
