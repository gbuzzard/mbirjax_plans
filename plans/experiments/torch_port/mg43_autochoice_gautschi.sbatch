#!/bin/bash
# mg43: verify the nightly's new automatic-device-choice check on real GPUs.
#
# WHY THIS JOB EXISTS.  The metrics repo's torch writer gained a check that
# builds one UNPINNED model and asserts the automatic device choice matches
# what the shipped widening floors say (open item C6).  Its skip path is
# verified on the Mac; the choice itself is a CUDA path that only exists at
# two or more visible devices, so the check's real branch has to be seen
# working on a multi-GPU node once before it rides a production nightly.
# This job runs the writer's own smoke (a toy one-cell sweep into a temp
# directory, no records touched) with four GPUs visible, which makes the
# auto-choice check run for real: an unpinned 512-class cone settle that
# must choose TWO devices, the middle count the floors admit there.
#
# The writer under test is the EDITED tooling staged at
# /scratch/gautschi/buzzard/torch_p3/mg43_metrics/, not the metrics
# checkout: the nightly clones from origin, and this change is not pushed
# yet.  The library under test is the torch_p3 tree, already installed
# editable in the shared environment.
#
# SCHEDULING.  This job does NOT pip-install anything, so it cannot race
# the torch_p3 install chain.  It reads the shared environment as it is.
#
# WALL ESTIMATE.  The toy sweep is about a minute; the auto-choice check
# builds one 512-class cone model, generates its sinogram on the host, and
# settles -- a few minutes with imports.  Thirty minutes is requested for
# margin.
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 00:30:00
#SBATCH -J torch_mg43
#SBATCH -o /scratch/gautschi/buzzard/torch_p3/mg43_%j.log
# No --mem: the ai partition allocates memory with the GPUs.

set -x
set -o pipefail
source ~/load_conda_cuda.sh
TORCHPY=/scratch/gautschi/buzzard/torch_p0/env/bin/python
export TORCHINDUCTOR_CACHE_DIR=/scratch/gautschi/buzzard/torch_p3/torch_cache
export MPLBACKEND=Agg

nvidia-smi -L
"$TORCHPY" -c "import mbirtorch, torch, ruamel.yaml; print('library under test:', mbirtorch.__file__); assert torch.cuda.device_count() >= 4, torch.cuda.device_count()" || exit 1

# The check must run the automatic branch, so no pin and no calibration mode
# may leak in from this shell.
unset MBIRTORCH_NUM_DEVICES
unset MBIRTORCH_MEMORY_CALIBRATION
unset MBIRTORCH_SORTED_FORWARD
unset PYTHONPATH
export MBIRTORCH_DISABLE_TRITON=0

OUT=$(mktemp -d /scratch/gautschi/buzzard/torch_p3/mg43_out_XXXX)
cd /scratch/gautschi/buzzard/torch_p3/mg43_metrics/scaling_tests || exit 1

REG_TORCH_SMOKE=1 REG_TORCH_GATE=1 REG_TORCH_PLATFORM=gpu-torch \
    REG_TORCH_LIB_ROOT=/scratch/gautschi/buzzard/torch_p3/mbirtorch_src \
    REG_TORCH_OUT_DIR="$OUT" \
    "$TORCHPY" -u torch_backend_writer.py
rc=$?

# The verdict this job exists for, pulled out of the run file so the log
# tail answers the question by itself.
"$TORCHPY" - "$OUT" <<'EOF'
import glob, sys
from ruamel.yaml import YAML
files = glob.glob(sys.argv[1] + "/regression_*.yaml")
files = [f for f in files if not f.endswith("_table.yaml")]
data = YAML(typ="safe").load(open(files[0])) if files else {}
check = data.get("auto_choice") or {}
print("AUTO-CHOICE RECORDED:", {k: check.get(k) for k in
      ("ok", "expected_n_devices", "realized_n_devices", "visible_devices",
       "skipped", "problems")})
EOF

echo "MG43 DONE rc=$rc"
exit $rc
