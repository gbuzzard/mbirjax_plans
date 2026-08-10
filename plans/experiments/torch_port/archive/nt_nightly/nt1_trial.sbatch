#!/bin/bash
#SBATCH -A bouman
#SBATCH -p ai
#SBATCH -q normal
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH -t 03:30:00
#SBATCH -J nt1-torch-trial
#SBATCH -o /scratch/gautschi/buzzard/nt1_trial/nt1_trial_%j.log
#
# nt1_trial.sbatch — the TRIAL-RUN GATE of nightly_plan.md §7: one real end-to-end mbirtorch
# nightly pass on one H100, with REG_TORCH_NO_PUSH=1, plus the gate's controls.  Everything runs
# from the scratch staging dir (per-file scp'd, md5-verified); nothing touches the production
# nightly, Greg's checkouts, or origin.
#
# Sequence (each step's PASS/FAIL is printed and the summary re-checks them):
#   A. platform-guard control: declare gpu-torch with CUDA hidden -> the writer must ABORT.
#   0. writer-direct run @ main~1 (prev): seeds the trial history (cold start).  No suite.
#   B-D. controls @ tip via nt1_trial_controls.py: mispinned-row raise; cone auto-shape capture
#        vs the writer's pin table; the §3(c-ii) memory ablation (5 fresh-subprocess repeats).
#   1. the WRAPPER @ tip: fire-on-change clone, install, suite, sweep, gate vs run 0 — the real
#      vs-prior pair (expected: pass/warn; the tip-vs-prev diff is the kernel repair, which only
#      touches non-trivial placements, and every trial row is n=1).
#   R. repair TORCHPY's editable install (the trial re-points it at throwaway clones) and verify.
#
# ENV NOTE (why TORCHPY, why the repair): the plan designates TORCHPY for the trial.  A venv
# layered over TORCHPY cannot work — venvs do not nest, so a layer would expose the conda BASE
# env's packages (the jax env), not TORCHPY's torch.  The trial therefore installs its two small
# missing deps (ruamel.yaml, pytest-xdist) into TORCHPY and lets the installs re-point TORCHPY's
# editable mbirtorch; step R restores the original torch_p3 pointer and proves it by import.
set -uo pipefail

STAGE=/scratch/gautschi/buzzard/nt1_trial
TORCHPY_ENV=/scratch/gautschi/buzzard/torch_p0/env
TORCH_P3_SRC=/scratch/gautschi/buzzard/torch_p3/mbirtorch_src
METRICS="$STAGE/metrics"
PY="$TORCHPY_ENV/bin/python"
PIP="$TORCHPY_ENV/bin/pip"

# Node preamble: conda/cuda modules + the squid proxy (git + pip on a compute node need it).
source "$HOME/load_conda_cuda.sh"
set -uo pipefail

echo "=== nt1 trial setup (on the compute node — the login nodes' mounts flap) ==="
date
cd "$STAGE"
# 0. The staged metrics clone must be a healthy git checkout (it was cloned via a login node).
if ! git -C metrics rev-parse HEAD >/dev/null 2>&1; then
  echo "FATAL: staged metrics clone is unhealthy — re-clone + re-scp, then resubmit."; exit 2
fi
# 1. md5-verify every scp'd file against the manifest BEFORE running anything (the standing
#    per-file-verification rule; a flapping login-node mount can corrupt an scp mid-write).
if ! md5sum -c nt1_manifest.md5; then
  echo "FATAL: staged files fail md5 verification — re-scp and resubmit."; exit 2
fi
# 2. The mbirtorch clones (tip + prev = tip~1), created HERE where the mounts are healthy.
rm -rf mbirtorch_tip mbirtorch_prev
git clone -q --depth 2 --branch main https://github.com/cabouman/mbirtorch mbirtorch_tip
git clone -q mbirtorch_tip mbirtorch_prev
git -C mbirtorch_prev checkout -q HEAD~1
echo "mbirtorch tip:  $(git -C mbirtorch_tip rev-parse HEAD)"
echo "mbirtorch prev: $(git -C mbirtorch_prev rev-parse HEAD)"
# 3. The two harness deps TORCHPY lacks (the plan's install-into-TORCHPY step).
"$PIP" install -q ruamel.yaml pytest-xdist
"$PY" -c "import ruamel.yaml, xdist; print('trial deps ok')"
# 4. The wrapper's work dir (its phase 1 would create this, but the trial enters at phase 2).
mkdir -p "$STAGE/work"

echo "=== nt1 trial preflight ==="
nvidia-smi -L
"$PY" - <<'EOF'
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| cuda_available", torch.cuda.is_available(),
      "| device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "NOT ON GPU"
EOF

fail=0
step() { echo; echo "──────────────────────────────────────────────"; echo "== $*"; echo "──────────────────────────────────────────────"; }

# ── A. platform-guard control: gpu-torch declared, CUDA hidden -> must abort ────────────────
step "control A: platform guard (declare gpu-torch on CPU-only -> abort expected)"
set +e
CUDA_VISIBLE_DEVICES="" REG_TORCH_PLATFORM=gpu-torch REG_TORCH_LIB_ROOT="$STAGE/mbirtorch_tip" \
  REG_TORCH_OUT_DIR="$STAGE/guard_out" REG_TORCH_SMOKE=1 \
  "$PY" "$METRICS/tooling/scaling_tests/torch_backend_writer.py" > "$STAGE/controlA.log" 2>&1
rcA=$?
set -e
if [ "$rcA" -ne 0 ] && grep -q "PLATFORM MISMATCH" "$STAGE/controlA.log"; then
  echo "control A PASS: writer aborted (rc=$rcA) with PLATFORM MISMATCH."
else
  echo "control A FAIL: rc=$rcA (expected nonzero + PLATFORM MISMATCH); see controlA.log"; fail=1
fi
set -uo pipefail

# ── 0. writer-direct run @ main~1 — seeds the vs-prior history ──────────────────────────────
step "run 0: writer-direct @ main~1 ($(git -C "$STAGE/mbirtorch_prev" rev-parse --short HEAD)) — cold start"
"$PIP" install -q -e "$STAGE/mbirtorch_prev" --no-deps
REG_TORCH_PLATFORM=gpu-torch REG_TORCH_LIB_ROOT="$STAGE/mbirtorch_prev" \
  REG_TORCH_OUT_DIR="$METRICS/results/gpu-torch/main" REG_TORCH_RUN_TAG=main \
  REG_TORCH_GATE=1 REG_TORCH_SKIP_TESTS=1 REG_TORCH_DEVICE_COUNTS="1" \
  "$PY" "$METRICS/tooling/scaling_tests/torch_backend_writer.py" 2>&1 | tee "$STAGE/run0.log"
rc0=${PIPESTATUS[0]}
if [ "$rc0" -eq 0 ]; then echo "run 0 PASS (rc=0, cold start)."; else echo "run 0 rc=$rc0 (expected 0)"; fail=1; fi

# ── B-D. controls @ tip: mispin raise, cone auto shapes, memory ablation ────────────────────
step "controls B-D @ tip (mispin, cone shapes, memory ablation)"
"$PIP" install -q -e "$STAGE/mbirtorch_tip" --no-deps
set +e
NT1_STAGE="$STAGE" NT1_METRICS="$METRICS" \
  "$PY" "$STAGE/nt1_trial_controls.py" 2>&1 | tee "$STAGE/controls.log"
rcC=${PIPESTATUS[0]}
set -e
if [ "$rcC" -eq 0 ]; then echo "controls B-D PASS."; else echo "controls B-D rc=$rcC — see controls.log"; fail=1; fi
set -uo pipefail

# ── 1. the WRAPPER @ tip — the real scheduled-path code, vs-prior gate on a real pair ───────
step "run 1: run_torch_regression.sh @ tip (suite + sweep + gate vs run 0)"
set +e
MBIRTORCH_REGRESSION_WORK="$STAGE/work" REG_TORCH_FRESH=1 REG_TORCH_NO_PUSH=1 REG_TORCH_FORCE=1 \
  REG_TORCH_VENV="$TORCHPY_ENV" \
  bash "$METRICS/tooling/regression/run_torch_regression.sh" </dev/null 2>&1 | tee "$STAGE/run1.log"
rc1=${PIPESTATUS[0]}
set -e
if [ "$rc1" -eq 0 ]; then echo "run 1 PASS (rc=0: no hard regression tip-vs-prev)."
elif [ "$rc1" -eq 1 ]; then echo "run 1: GATE FAIL (rc=1) — a HARD finding tip-vs-prev; inspect run1.log"; fail=1
else echo "run 1 rc=$rc1 (harness error)"; fail=1; fi
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

# ── summary ─────────────────────────────────────────────────────────────────────────────────
step "summary"
echo "results tree:"
ls -la "$METRICS/results/gpu-torch/main/" || true
echo
echo "gate lines from run 1:"
grep -aE "GATE: |HARD |warn " "$STAGE/run1.log" | head -20 || true
echo
echo "wall/cost: job $SLURM_JOB_ID; see sacct after completion."
if [ "$fail" -eq 0 ]; then echo "NT1 TRIAL: ALL STEPS PASS"; else echo "NT1 TRIAL: FAILURES PRESENT (see above)"; fi
exit "$fail"
