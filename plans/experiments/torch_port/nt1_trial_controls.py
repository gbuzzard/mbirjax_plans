"""nt1_trial_controls.py — controls B-D of the TRIAL-RUN GATE (nightly_plan.md §7), run on the
GPU node at the mbirtorch TIP, between run 0 and run 1 of nt1_trial.sbatch.

  B. mispinned-row control: pin_devices asking for MORE devices than the allocation holds must
     RAISE (never measure silently).  On the 1-GPU trial node, n=2 is the mispin.
  C. cone recon-shape capture: mbirtorch's AUTO shapes at all four GPU cone cells, compared to
     the writer's CONE_RECON_SHAPE_PINS — the §3(b) requirement that the pin table be seeded
     from measured auto shapes, verified rather than trusted.
  D. the §3(c-ii) memory-determinism ablation: ONE cell (parallel / vcd_nonconst / 512x448x384,
     n=1) repeated five times in FRESH subprocesses; the mem_mb spread decides
     TORCH_MEM_GATE_WINDOW (a few tenths of a percent -> 1; bimodal -> 3, matching jax).

Env: NT1_STAGE (staging dir), NT1_METRICS (the staged metrics checkout).
Exit 0 = all controls pass; nonzero = at least one failed (details on stdout).
"""
import os
import subprocess
import sys
import tempfile

STAGE = os.environ["NT1_STAGE"]
METRICS = os.environ["NT1_METRICS"]
SCALING = os.path.join(METRICS, "tooling", "scaling_tests")
sys.path.insert(0, SCALING)

import numpy as np                      # noqa: E402
import scaling_common as sc             # noqa: E402
import torch_backend_writer as twb      # noqa: E402

ABLATION_CELL = ("parallel", "vcd_nonconst", "512x448x384")
N_REPEATS = 5

failures = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ── B. mispinned-row control ──────────────────────────────────────────────────
print("\n[control B] mispinned row must raise (ask n=2 on a 1-GPU allocation)")
import torch  # noqa: E402
visible = torch.cuda.device_count()
print(f"  visible CUDA devices: {visible}")
if visible != 1:
    check("mispin-precondition", False, f"control expects a 1-GPU allocation, got {visible}")
else:
    cfg = twb.build_config("gpu-torch", tempfile.mkdtemp(), "00000000", "trial",
                           STAGE, [1], gate=False)
    model = twb.make_model(cfg, "parallel", (64, 64, 64), "gpu-torch")
    try:
        twb.pin_devices(model, 2, "gpu-torch")
        check("mispin-raises", False, "pin_devices(n=2) did NOT raise on a 1-GPU node")
    except Exception as e:                      # noqa: BLE001 — any loud refusal is the pass
        check("mispin-raises", True, f"{type(e).__name__}: {str(e)[:100]}")
    del model

# ── C. cone auto-shape capture vs the pin table ──────────────────────────────
print("\n[control C] cone AUTO recon shapes at the four GPU cells vs CONE_RECON_SHAPE_PINS")
import mbirtorch  # noqa: E402
gpu_cells = [(200, 208, 160), (512, 448, 384), (513, 449, 385), (1024, 1008, 992)]
for size in gpu_cells:
    n_views, n_rows, n_channels = size
    sdd = 4.0 * n_channels                      # the engine's cone convention (magnification 2)
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    m = mbirtorch.ConeBeamModel(size, angles, source_detector_dist=sdd,
                                source_iso_dist=sdd / 2.0)
    m.configure_devices(num_devices=1)          # pin before any read, per the row discipline
    auto = tuple(int(x) for x in m.get_params("recon_shape"))
    pin = twb.CONE_RECON_SHAPE_PINS.get(size)
    check(f"cone-pin {size}", auto == pin, f"auto={auto} pin={pin}")
    del m

# ── D. §3(c-ii) memory ablation: 5 fresh-subprocess repeats of one cell ──────
print(f"\n[control D] memory ablation: {ABLATION_CELL} n=1, {N_REPEATS} fresh subprocesses")
geom, op, size_label = ABLATION_CELL
cfg = twb.build_config("gpu-torch", tempfile.mkdtemp(), "00000000", "trial", STAGE, [1],
                       gate=False)
fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="nt1_abl_cfg_")
os.close(fd)
sc.save_yaml(cfg_path, cfg.to_dict())
writer = os.path.join(SCALING, "torch_backend_writer.py")
mems, times = [], []
for i in range(N_REPEATS):
    fd, out_f = tempfile.mkstemp(suffix=".yaml", prefix=f"nt1_abl_{i}_")
    os.close(fd)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [p for p in (os.path.join(STAGE, "mbirtorch_tip"), SCALING,
                     os.environ.get("PYTHONPATH")) if p])}
    r = subprocess.run([sys.executable, writer, "--worker", "--mode", "measure",
                        "--config", cfg_path, "--platform", "gpu-torch",
                        "--geometry", geom, "--op", op, "--size", size_label,
                        "--device-counts", "1", "--out-file", out_f],
                       env=env, capture_output=True, text=True)
    res = sc.load_yaml(out_f) if os.path.exists(out_f) else None
    rows = (res or {}).get("rows") or []
    if r.returncode != 0 or not rows:
        check(f"ablation-repeat-{i}", False,
              f"rc={r.returncode}; tail: {r.stdout[-200:]} {r.stderr[-200:]}")
        continue
    mems.append(float(rows[0]["mem_mb"]))
    times.append(float(rows[0]["min_ms"]))
    print(f"  repeat {i}: mem={mems[-1]:.1f} MB  min={times[-1]:.1f} ms")
if len(mems) == N_REPEATS:
    lo, hi = min(mems), max(mems)
    spread_pct = (hi - lo) / lo * 100.0 if lo else float("inf")
    print(f"  mem spread: min={lo:.1f} MB  max={hi:.1f} MB  spread={spread_pct:.3f}%")
    print(f"  time spread: min={min(times):.1f}  max={max(times):.1f} ms")
    # The decision rule from §3(c-ii): a spread inside a few tenths of a percent justifies
    # window=1 (hard gate, no detection lag); bimodality justifies 3.  8% is mem_hard_pct, so
    # anything near it would false-fire nightly — flag loudly.
    check("ablation-complete", True, f"spread {spread_pct:.3f}% over {N_REPEATS} repeats")
    print(f"  WINDOW RECOMMENDATION: {'1 (deterministic)' if spread_pct < 0.5 else '3 (spread like jax)'}")
else:
    check("ablation-complete", False, f"only {len(mems)}/{N_REPEATS} repeats produced a row")

print()
if failures:
    print(f"controls: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("controls: all pass")
