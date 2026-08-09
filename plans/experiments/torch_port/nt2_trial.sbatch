#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -n 56
#SBATCH -t 02:30:00
#SBATCH -J nt2-torch-trial
#SBATCH -o /scratch/gautschi/buzzard/nt1_trial/nt2_trial_%j.log
#
# nt2_trial.sbatch — the n>1 trial (nightly_plan.md §7, the increment-7 trial), on four H100s,
# from the same scratch staging as nt1 and with the same seams: REG_TORCH_NO_PUSH=1, per-file
# md5 verification on the compute node, TORCHPY with the editable repair at the end.
#
# Sequence:
#   A. mispin control at multi-GPU: with 2 devices visible, pin_devices(n=4) must RAISE.
#   1. the WRAPPER @ tip with REG_TORCH_DEVICE_COUNTS from the staged knobs ("1 2 4"): suite
#      on the 4-GPU node, then the widened sweep — n∈{1,2,4} at the two multi-device sizes,
#      n=1 everywhere else.  Gate vs the nt1 prior: n=1 cells compare (PASS expected), the
#      n=2/4 cells are new (soft "no baseline" notes — the designed cold start).
#   B. post controls: the §3(c-ii) ablation at n=2 and n=4 (five fresh subprocesses each),
#      the cross-count fingerprint check at the op tolerances, and the padding_zero assert.
#   R. repair TORCHPY's editable install and verify.
set -uo pipefail

STAGE=/scratch/gautschi/buzzard/nt1_trial
TORCHPY_ENV=/scratch/gautschi/buzzard/torch_p0/env
TORCH_P3_SRC=/scratch/gautschi/buzzard/torch_p3/mbirtorch_src
METRICS="$STAGE/metrics"
PY="$TORCHPY_ENV/bin/python"
PIP="$TORCHPY_ENV/bin/pip"

source "$HOME/load_conda_cuda.sh"
set -uo pipefail

echo "=== nt2 trial setup (all clones on the COMPUTE node — login-node mounts flap) ==="
date
cd "$STAGE"
# 1. Verify the scp'd OVERLAY first: the two files that are not yet at origin (the writer's
#    reduced n>1 cell set, and the knobs that request counts 1 2 4).  Everything else comes
#    from origin, so the trial measures committed code plus exactly this diff.
if ! md5sum -c nt2_manifest.md5; then
  echo "FATAL: overlay files fail md5 verification — re-scp and resubmit."; exit 2
fi
# 2. A FRESH metrics clone.  A pull of the staged clone was wedged twice by the login node's
#    flapping mount, and a half-fetched object store is not worth repairing — cloning here
#    costs seconds and is immune to it.
rm -rf metrics
git clone -q https://github.com/gbuzzard/mbirjax_metrics.git metrics
echo "metrics @ $(git -C metrics rev-parse --short HEAD)"
# 3. Apply the overlay, and prove each copy landed byte-identical.
while read -r _md5 rel; do
  dst="metrics/${rel#overlay/}"
  cp "$rel" "$dst"
  cmp -s "$rel" "$dst" || { echo "FATAL: overlay copy differs: $dst"; exit 2; }
  echo "  overlay -> $dst"
done < nt2_manifest.md5
# 4. The library tip.
rm -rf mbirtorch_tip
git clone -q --depth 2 --branch main https://github.com/cabouman/mbirtorch mbirtorch_tip
echo "mbirtorch tip: $(git -C mbirtorch_tip rev-parse HEAD)"
"$PIP" install -q ruamel.yaml pytest-xdist
mkdir -p "$STAGE/work"
rm -rf "$STAGE/work/.lock.d"      # a killed prior run can leave the single-instance lock held

echo "=== nt2 preflight ==="
nvidia-smi -L
"$PY" - <<'EOF'
import torch, sys
print("torch", torch.__version__, "| visible", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() == 4, "need 4 GPUs"
EOF

fail=0
step() { echo; echo "──────────────────────────────────────────────"; echo "== $*"; echo "──────────────────────────────────────────────"; }

# ── A. mispin control at multi-GPU (2 visible, ask 4) ───────────────────────────────────────
step "control A: mispin (2 visible, pin n=4 -> raise expected)"
"$PIP" install -q -e "$STAGE/mbirtorch_tip" --no-deps
set +e
CUDA_VISIBLE_DEVICES=0,1 NT2_STAGE="$STAGE" NT2_METRICS="$METRICS" NT2_MODE=mispin \
  "$PY" "$STAGE/nt2_trial_controls.py" 2>&1 | tee "$STAGE/nt2_controlA.log"
rcA=${PIPESTATUS[0]}
set -e
if [ "$rcA" -eq 0 ]; then echo "control A PASS."; else echo "control A FAIL (rc=$rcA)"; fail=1; fi
set -uo pipefail

# ── 1. the WRAPPER @ tip — the widened sweep through the production code path ───────────────
step "run: run_torch_regression.sh @ tip with counts from the staged knobs"
set +e
MBIRTORCH_REGRESSION_WORK="$STAGE/work" REG_TORCH_FRESH=1 REG_TORCH_NO_PUSH=1 REG_TORCH_FORCE=1 \
  REG_TORCH_VENV="$TORCHPY_ENV" \
  bash "$METRICS/tooling/regression/run_torch_regression.sh" </dev/null 2>&1 | tee "$STAGE/nt2_run.log"
rc1=${PIPESTATUS[0]}
set -e
if [ "$rc1" -eq 0 ]; then echo "run PASS (rc=0)."
elif [ "$rc1" -eq 1 ]; then echo "run: GATE FAIL (rc=1) — inspect nt2_run.log"; fail=1
else echo "run rc=$rc1 (harness error)"; fail=1; fi
set -uo pipefail

# ── B. post controls: ablation at n=2/4, cross-count values, padding ────────────────────────
step "controls B: n>1 ablation + cross-count value check"
"$PIP" install -q -e "$STAGE/mbirtorch_tip" --no-deps   # the wrapper's install pointed at its
#                                                         deleted clone; re-point for the controls
set +e
NT2_STAGE="$STAGE" NT2_METRICS="$METRICS" NT2_MODE=post \
  "$PY" "$STAGE/nt2_trial_controls.py" 2>&1 | tee "$STAGE/nt2_controls.log"
rcB=${PIPESTATUS[0]}
set -e
if [ "$rcB" -eq 0 ]; then echo "controls B PASS."; else echo "controls B FAIL (rc=$rcB)"; fail=1; fi
set -uo pipefail

# ── R. repair TORCHPY's editable install and verify ─────────────────────────────────────────
step "repair: restore TORCHPY's editable mbirtorch -> torch_p3"
"$PIP" install -q -e "$TORCH_P3_SRC" --no-deps
"$PY" - <<EOF
import mbirtorch
print("mbirtorch resolves to:", mbirtorch.__file__)
assert mbirtorch.__file__.startswith("$TORCH_P3_SRC"), "REPAIR FAILED"
print("repair PASS")
EOF

step "summary"
grep -aE "GATE: |HARD |n_devices= [24]" "$STAGE/nt2_run.log" | head -30 || true
if [ "$fail" -eq 0 ]; then echo "NT2 TRIAL: ALL STEPS PASS"; else echo "NT2 TRIAL: FAILURES PRESENT"; fi
exit "$fail"
