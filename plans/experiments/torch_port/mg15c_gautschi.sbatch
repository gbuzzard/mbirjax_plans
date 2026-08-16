#!/bin/bash
# mg15c: ADJUDICATING THE ADJOINT, on four H100s.  Is the back-projection
# cross-count difference a float32 GROUPING property that predates the padding
# removal, or a SPLIT defect the removal introduced?  The full statement is in
# mg15c_adjoint_matrix.py; the essentials are repeated here because a reader of
# this file may never open that one.
#
# WHERE THIS SITS.  mg15 gated the pad removal and passed every leg but the
# multiaxis four-device reconstruction backstop (2.969e-02 against 1e-2).
# mg15b (slurm 15304687) answered that: the reconstruction difference is
# trajectory-shaped -- it decayed to 1.682e-02 at ten iterations, its boundary
# peak decayed with the interior, and the boundary slices ranked #271/#272/#120
# of 510.  But mg15b's own NEW instrument, the adjoint leg it added because
# mg15 had none, tripped: back projection at four devices against one read
# 6.289e-04 (nrmse 5.580e-06), where mg15's FORWARD leg on the same arm read
# 1.585e-05.  A single back projection has no trajectory in it, so that number
# needs its own adjudication.  This job is it.
#
# THE HYPOTHESIS, registered before the run.  Forward and back are not
# symmetric under a VIEW-sharded split.  A forward output element is one
# detector element of one view, and one view lives entirely on one device, so
# its sum never crosses the split and the forward leg reads scheduling noise.
# A back-projection output voxel receives every view, so its sum genuinely
# splits: at one device ~512 same-sign contributions accumulate in one long
# float32 chain, at four devices in four shorter chains combined pairwise.
# Float32 addition is not associative and same-sign sequential-against-blocked
# accumulation differs at exactly this scale.  If that is what 6.289e-04 is,
# it read the SAME on the padded code -- padding appended zeros, which change
# no sum, and the device count set the number of chains either way -- so it is
# a pre-existing property of splitting a reduction that the first cross-count
# value instrument at this size has surfaced, not a regression.
#
# THE THREE INSTRUMENTS.  Every arm is a SINGLE back projection: no
# reconstruction, no iteration, no ledger, no calibration mode.  Every arm pins
# with an explicit configure_devices list installed at construction, so the
# projectors run on the real layout from the first call and mg15's settle trap
# cannot arise.
#   1. COUNT MATRIX (current tree, multiaxis) at n=1,2,3,4, printed as the full
#      pairwise table, plus same-count repeats at n=1 and n=4 that measure the
#      adjoint's own run-to-run floor -- a number mg15b never took.  A GROUPING
#      property puts n=2/3/4 together, all a similar distance from n=1.  A
#      SPLIT defect singles out the non-dividing count: 512 views over 3 leaves
#      2, where 2 and 4 divide 512 exactly.
#   2. PRE-REMOVAL ABLATION (decisive): the same n=1-against-n=4 comparison on
#      the PADDED code, greg_dev 672edbd, the last commit before P3 changed the
#      split.  Within a factor of ~2 of the current tree means the adjoint
#      difference PREDATES the removal; the padded tree near 1e-7 while the
#      current reads 6e-4 means the removal changed the adjoint.
#   3. FAMILY CONTEXT (report only): the same comparison for cone and parallel
#      at n=3, whose back projections run the hand-written Triton kernels and,
#      for parallel, the row-aligned driver.  It says whether the adjoint
#      reading is torch-body-specific or universal.
#
# NO GATES ANYWHERE.  The exit code reports INSTRUMENT HEALTH only -- 0 when
# every arm ran and every comparison computed -- and the verdict is read by a
# human from the printed matrix in this log.  A threshold on any of these
# numbers would be the kind of borrowed constant that produced the question.
# Nothing is chained to this job.
#
# THE PADDED TREE MUST BE STAGED BEFORE THIS JOB RUNS.  It is not created here:
#     git -C <mbirtorch> worktree add /scratch/gautschi/buzzard/torch_p3/mbirtorch_p2 672edbd
# The job builds an overlay venv over TORCHPY's environment for it, idempotently
# (a re-run reuses it), and the two padded arms use that interpreter.  Both
# preflights below then confirm the two trees are what they claim: the current
# one splits 512 over three devices into 171/171/170, and the padded one into
# three EQUAL blocks.  The script re-checks both inside every arm and also
# asserts each padded arm's imported mbirtorch really came from mbirtorch_p2,
# because an overlay that silently resolved to the current checkout would print
# two identical columns and a confident wrong answer.
#
# WALL ESTIMATE, and its arithmetic.  Three generators plus ten single back
# projections, all on the (512, 448, 384) cell.
#   * mg15b measured a single back projection at this cell in seconds, with the
#     cost dominated by one torch compilation per device per process.  Ten arms
#     at ~1-2 minutes each including compilation: ~15 minutes.
#   * the two repeat arms project twice, but compilation is paid once per
#     process so the second projection is nearly free.
#   * the two padded arms pay their own first-time compilation in a separate
#     interpreter with a cold inductor cache: ~4 minutes.
#   * three generators (one forward projection each) plus md5 work over ~1.9
#     GiB of staged phantom and sinogram and ~2.8 GiB of staged volumes, each
#     re-read by the driver's comparisons: ~5 minutes.
#   * thirteen subprocess starts, each paying a torch import: ~3 minutes.
#   * the venv build itself, first time only: ~1 minute.
# Total ~25-30 minutes.  The walltime below is 00:45:00, which clears the 02:00
# and 03:00 nightlies.
#
# DISK.  About 4.7 GiB in $MG15C_RESULTS at peak: three sinograms (~1.0 GiB),
# three phantoms (~0.85 GiB), and ten back-projection volumes (~2.8 GiB).
# Deleted at the end unless MG15C_KEEP_ARTIFACTS=1 is exported; the jsonl is
# always kept.
#
# Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch mg15c_gautschi.sbatch
# Env note: pass env through the SUBMITTING SHELL, never through
# --export=ALL,VAR=a,b,c -- slurm splits exported values on commas.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 00:45:00
#SBATCH -J torch_mg15c
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg15c_%j.log
# No --mem: the ai partition allocates memory with the GPUs, and an explicit
# request has cost this work a queue delay before.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
P2_SRC=/scratch/gautschi/buzzard/torch_p3/mbirtorch_p2
P2_VENV=/scratch/gautschi/buzzard/torch_p3/venv_p2
P2_PY="$P2_VENV/bin/python"
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import triton; print('triton', triton.__version__)" || exit 1
"$TORCHPY" -m pip install --no-cache-dir -q --no-deps -e /scratch/gautschi/buzzard/torch_p3/mbirtorch_src || exit 1
"$TORCHPY" -c "import mbirtorch, torch; assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# ── the padded tree and its overlay interpreter ──────────────────────────────
# The checkout is NOT created here: it is staged before the job.  Fail early and
# say so, rather than running eight arms and then reporting two missing ones.
test -d "$P2_SRC" || { echo "the padded checkout is missing at $P2_SRC: stage 672edbd there before submitting (git worktree add $P2_SRC 672edbd)"; exit 1; }
# The overlay venv.  --system-site-packages does NOT work here: TORCHPY is
# itself a venv (over the conda base), and a venv stacked on a venv inherits
# the ORIGINAL base's site-packages, not the middle layer's -- torch lives in
# the middle layer, so the first build of this job died on `import torch`.
# PYTHONPATH is no substitute either: TORCHPY's environment carries an
# editable install of mbirtorch_src whose meta-path finder outranks any
# PYTHONPATH entry, in exactly the silent direction that would print two
# identical columns.  So the venv is built BARE, the p2 checkout is installed
# editable into it (its finder is the only mbirtorch finder registered), and
# TORCHPY's site-packages directory is appended through a .pth path line --
# a path line adds the directory to sys.path without processing that
# directory's own .pth files, so torch, triton and numpy resolve from the
# shared environment while mbirtorch_src's editable finder never registers.
# Only mbirtorch differs, which is the single variable instrument 2 varies.
# The venv is validated rather than trusted: a build that cannot import both
# torch and the p2 tree is torn down and rebuilt.
TORCH_SITE=$("$TORCHPY" -c "import torch, os; print(os.path.dirname(os.path.dirname(torch.__file__)))")
if ! "$P2_PY" -c "import torch, mbirtorch; assert 'mbirtorch_p2' in mbirtorch.__file__" 2>/dev/null; then
    rm -rf "$P2_VENV"
    "$TORCHPY" -m venv "$P2_VENV" || exit 1
    "$P2_PY" -m pip install --no-cache-dir -q --no-deps -e "$P2_SRC" || exit 1
    P2_SITE=$("$P2_PY" -c "import site; print(site.getsitepackages()[0])")
    echo "$TORCH_SITE" > "$P2_SITE/zz_torch_env.pth" || exit 1
fi
"$P2_PY" -c "import torch, mbirtorch; assert 'mbirtorch_p2' in mbirtorch.__file__, mbirtorch.__file__; print('overlay venv ok:', mbirtorch.__file__)" || exit 1

# ── both preflights: each tree must be what it claims ────────────────────────
# THE CURRENT TREE: the pad is gone, so 512 over three devices is 171/171/170.
"$TORCHPY" -c "
from mbirtorch._sharding import Placement
blocks = [e - s for _d, (s, e) in Placement(['cpu'] * 3, 0, 512).shard_ranges(512)]
assert blocks == [171, 171, 170], blocks
print('current tree: unpadded split confirmed:', blocks)" || { echo "the tree in mbirtorch_src is not the unpadded one mg15/mg15b measured"; exit 1; }
# THE PADDED TREE: the inverse.  Note the API differs and the check is written
# from the 672edbd source rather than assumed: that commit predates 7032535's
# rename of Placement's third argument (real_size -> axis_len), so the call is
# POSITIONAL; and on the padded tree shard_ranges takes a REQUIRED size and
# REFUSES a length its device count does not divide -- that refusal IS the
# padding contract -- so the blocks are reached through padded_shard_ranges,
# which pads 512 to 513 and returns three EQUAL blocks of 171.
"$P2_PY" -c "
import mbirtorch
from mbirtorch._sharding import Placement
assert 'mbirtorch_p2' in mbirtorch.__file__, mbirtorch.__file__
p = Placement(['cpu'] * 3, 0, 512)
blocks = [e - s for _d, (s, e), *rest in p.padded_shard_ranges()]
assert p.is_padded and len(set(blocks)) == 1 and p.padded_size == 513, (blocks, p.padded_size)
print('padded tree: equal-block split confirmed:', blocks, 'padded_size', p.padded_size)
print('padded tree import:', mbirtorch.__file__)" || { echo "the tree at $P2_SRC is not the padded 672edbd checkout, or its overlay venv did not take effect"; exit 1; }

# The shipped configuration, set explicitly rather than left to whatever the
# submitting shell held: an adjudication has to be taken in the same
# environment as the reading it adjudicates.  The script re-asserts it per arm.
export MBIRTORCH_DISABLE_TRITON=0
# This job measures VALUES ONLY and reads no memory, so it must not own the peak
# counter -- the calibration mode RESETS torch.cuda.max_memory_allocated, and a
# job that resets a counter it never reads takes it from whatever else is
# measuring.  Popped again per subprocess.
unset MBIRTORCH_MEMORY_CALIBRATION
# Every arm pins with an explicit configure_devices list, so an inherited count
# pin would describe a mechanism this job does not use.  Popped per subprocess.
unset MBIRTORCH_NUM_DEVICES
# PYTHONPATH must not point at either checkout: the padded arms select their
# tree through their INTERPRETER, and an inherited PYTHONPATH would shadow it
# silently, in the one direction that makes instrument 2 print two identical
# columns.  The script also pops it for those arms.
unset PYTHONPATH
unset MG15C_SMOKE
unset MG15C_DEVICE
unset MG15C_KEEP_ARTIFACTS

export MG15C_P2_ROOT="$P2_SRC"
export MG15C_P2_PYTHON="$P2_PY"
export MG15C_RESULTS=/scratch/gautschi/buzzard/torch_p3/results
mkdir -p "$MG15C_RESULTS"
cd /scratch/gautschi/buzzard/torch_p3 || exit 1
# The arm plan first, so the log opens with the thirteen subprocesses and the
# reading being adjudicated, and a mis-set variable is caught before GPU work.
"$TORCHPY" -u mg15c_adjoint_matrix.py --dry-run || exit 1
# The probe.  Its exit code reports whether the INSTRUMENTS worked, not what
# they found; the finding is in the matrix in this log.
"$TORCHPY" -u mg15c_adjoint_matrix.py
rc=$?
echo "MG15C DONE rc=$rc"
exit $rc
