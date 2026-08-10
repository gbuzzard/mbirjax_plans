"""nt2_trial_controls.py — the n>1 trial's controls (nightly_plan.md §7 increment-7 trial),
run on a 4-GPU node around the wrapper's first n∈{1,2,4} sweep.

Modes (NT2_MODE):
  mispin  — run with CUDA_VISIBLE_DEVICES=0,1: pin_devices asking n=4 with 2 visible must
            RAISE (the nt1 mispin control repeated at multi-GPU).
  post    — after the sweep, using the run YAML it wrote:
            (a) the §3(c-ii) memory ablation at n=2 and n=4: one cell, five FRESH
                subprocesses each, spread recorded — the window number for the n>1 rows
                must rest on n>1 data, not the n=1 measurement;
            (b) the cross-count value check: for every multi-count cell, the n=2 and n=4
                fingerprint aggregates must sit within the op's own rtol of the n=1 row
                (the cross-device reference of the correctness design, checked in-trial);
            (c) padding_zero must remain True on every multi-count row (the §3(e)
                invariant; vacuous here because 2 and 4 divide both sharded axes, and
                asserted so a future non-dividing count cannot silently regress it).

Env: NT2_STAGE (staging dir), NT2_METRICS (staged metrics checkout), NT2_MODE.
Exit 0 = controls pass; nonzero = at least one failed (details on stdout).
"""
import glob
import os
import subprocess
import sys
import tempfile

STAGE = os.environ["NT2_STAGE"]
METRICS = os.environ["NT2_METRICS"]
MODE = os.environ.get("NT2_MODE", "post")
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


def finish():
    print()
    if failures:
        print(f"nt2 controls[{MODE}]: {len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print(f"nt2 controls[{MODE}]: all pass")
    sys.exit(0)


# ── mispin mode: n=4 with 2 visible devices must raise ────────────────────────
if MODE == "mispin":
    import torch
    visible = torch.cuda.device_count()
    print(f"[mispin] visible CUDA devices: {visible} (CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    if visible != 2:
        check("mispin-precondition", False, f"control expects 2 visible, got {visible}")
        finish()
    cfg = twb.build_config("gpu-torch", tempfile.mkdtemp(), "00000000", "trial",
                           STAGE, [1], gate=False)
    model = twb.make_model(cfg, "parallel", (64, 64, 64), "gpu-torch")
    try:
        twb.pin_devices(model, 4, "gpu-torch")
        check("mispin-raises-n4", False, "pin_devices(n=4) did NOT raise with 2 visible")
    except Exception as e:                      # noqa: BLE001
        check("mispin-raises-n4", True, f"{type(e).__name__}: {str(e)[:100]}")
    finish()

# ── post mode ─────────────────────────────────────────────────────────────────
# (a) memory ablation at n=2 and n=4.
geom, op, size_label = ABLATION_CELL
cfg = twb.build_config("gpu-torch", tempfile.mkdtemp(), "00000000", "trial", STAGE, [1],
                       gate=False)
fd, cfg_path = tempfile.mkstemp(suffix=".yaml", prefix="nt2_abl_cfg_")
os.close(fd)
sc.save_yaml(cfg_path, cfg.to_dict())
writer = os.path.join(SCALING, "torch_backend_writer.py")
spreads = {}
for n in (2, 4):
    print(f"\n[ablation] {ABLATION_CELL} n={n}, {N_REPEATS} fresh subprocesses")
    mems, times = [], []
    for i in range(N_REPEATS):
        fd, out_f = tempfile.mkstemp(suffix=".yaml", prefix=f"nt2_abl_{n}_{i}_")
        os.close(fd)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(
            [p for p in (os.path.join(STAGE, "mbirtorch_tip"), SCALING,
                         os.environ.get("PYTHONPATH")) if p])}
        r = subprocess.run([sys.executable, writer, "--worker", "--mode", "measure",
                            "--config", cfg_path, "--platform", "gpu-torch",
                            "--geometry", geom, "--op", op, "--size", size_label,
                            "--device-counts", str(n), "--out-file", out_f],
                           env=env, capture_output=True, text=True)
        res = sc.load_yaml(out_f) if os.path.exists(out_f) else None
        rows = (res or {}).get("rows") or []
        if r.returncode != 0 or not rows:
            check(f"ablation-n{n}-repeat-{i}", False,
                  f"rc={r.returncode}; tail: {r.stdout[-200:]} {r.stderr[-200:]}")
            continue
        mems.append(float(rows[0]["mem_mb"]))
        times.append(float(rows[0]["min_ms"]))
        print(f"  repeat {i}: mem={mems[-1]:.1f} MB  min={times[-1]:.1f} ms")
    if len(mems) == N_REPEATS:
        lo, hi = min(mems), max(mems)
        spread = (hi - lo) / lo * 100.0 if lo else float("inf")
        spreads[n] = spread
        print(f"  n={n} mem spread: min={lo:.1f}  max={hi:.1f} MB  spread={spread:.3f}%")
        check(f"ablation-n{n}-complete", True, f"spread {spread:.3f}%")
    else:
        check(f"ablation-n{n}-complete", False, f"only {len(mems)}/{N_REPEATS} rows")
if spreads:
    worst = max(spreads.values())
    print(f"\n  WINDOW RECOMMENDATION (n>1): {'1 (deterministic)' if worst < 0.5 else '3 (spread like jax)'}"
          f"  [worst spread {worst:.3f}%]")

# (b) + (c) cross-count values and padding, from the sweep's run file.
runs = sorted(glob.glob(os.path.join(METRICS, "results", "gpu-torch", "main",
                                     "regression_gpu-torch_*.yaml")))
runs = [r for r in runs if not r.endswith("_table.yaml")]
doc = sc.load_yaml(runs[-1])
print(f"\n[cross-count] run file: {os.path.basename(runs[-1])}")
cells = [c for c in doc["cells"] if not c.get("failed")]
multi = {}
for c in cells:
    if c["size"] in twb.MULTI_DEVICE_SIZE_LABELS and c["geometry"] != "denoiser":
        multi.setdefault((c["geometry"], c["op"], c["size"]), {})[int(c["n_devices"])] = c
n_groups = 0
for (g, o, s), by_n in sorted(multi.items()):
    if 1 not in by_n:
        check(f"crosscount-{g}|{o}|{s}", False, "no n=1 row to compare against")
        continue
    rtol = cfg.fp_rtol_iter if o in ("vcd_nonconst", "denoise") else cfg.fp_rtol_single
    ref = by_n[1]["fingerprint"]
    for n in (2, 4):
        if n not in by_n:
            check(f"crosscount-{g}|{o}|{s}|n{n}", False, "row missing")
            continue
        fp = by_n[n]["fingerprint"]
        worst_m, worst_rd = "", 0.0
        for m in ("sum", "mean", "l2norm"):
            rd = abs(float(fp[m]) - float(ref[m])) / (abs(float(ref[m])) or 1.0)
            if rd > worst_rd:
                worst_m, worst_rd = m, rd
        ok = worst_rd <= rtol
        check(f"crosscount-{g}|{o}|{s}|n{n}", ok,
              f"worst {worst_m} reldiff {worst_rd:.2e} vs rtol {rtol:g}")
        check(f"padding-{g}|{o}|{s}|n{n}", bool(fp.get("padding_zero", True) in (True, None)),
              f"padding_zero={fp.get('padding_zero')}")
        n_groups += 1
check("crosscount-coverage", n_groups == 32,
      f"{n_groups} of 32 expected (2 geoms x 4 ops x 2 sizes x 2 counts) cross-count rows")

finish()
