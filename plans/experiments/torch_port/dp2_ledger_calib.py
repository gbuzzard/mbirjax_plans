"""Memory-ledger calibration: modeled against measured per-device peaks.

The deliverable is one table.  For each (geometry, cell) it reports the
ledger's modeled peak, the measured `torch.cuda.max_memory_allocated`, and
their ratio, at n=1 with the kernels on.  The acceptance band is
1.00 <= modeled/measured <= 1.30: a ratio under 1.00 means the ledger
under-predicts, which is the direction that would let a doomed run start.

Two things make the reading trustworthy.  Each arm runs in its OWN
subprocess, so one arm's allocator state cannot leak into the next.  The
modeled number is read off the model's own `last_memory_calibration`, so
the table reports what the production path computed, never a
re-derivation by this script.

The measurement protocol matches the composed gates: a warm seeded
3-iteration vcd, weights supplied, and the peak read after a cold pass has
already paid the compiles.  The calibration mode resets the peak counter at
the top of vcd_recon, so the measured number covers the reconstruction and
not the phantom or the sinogram that preceded it.

One extra arm has no gate counterpart.  The `unweighted` arm reruns the
parallel 1024 cell with weights=None, which is the only path on which
vcd_recon builds its own all-ones sinogram into `hess_weights` and then
never releases it.  That residency cannot be measured on a weighted run,
where `hess_weights` is a bare alias of the caller's weights.

Environment:
    DP2_GEOMETRIES=parallel,cone      subset of the geometries
    DP2_CELLS=512,1024                subset of the cells (by view count)
    DP2_ARMS=weighted,unweighted      subset of the arms
    DP2_RESULTS=<dir>                 where the jsonl and the sinograms go
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

RESULTS_DIR = os.environ.get("DP2_RESULTS", ".")
CELLS = [(512, 448, 384), (1024, 1008, 992)]
GEOMETRIES = ["parallel", "cone"]
ARMS = ["weighted", "unweighted"]
VCD_ITERATIONS = 3
VCD_SEED = 12345
# The band the checkpoint-1 design fixed; a ratio below the floor is the
# failure this whole model exists to prevent.
BAND = (1.00, 1.30)
# The unweighted arm runs at one cell only: its purpose is to size a single
# residency, not to re-measure the matrix.
UNWEIGHTED_CELL = (1024, 1008, 992)
UNWEIGHTED_GEOMETRY = "parallel"


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_dp2_sino_{geometry}_{cell[0]}.npy")


def _weights(sinogram):
    """The gates' weighting formula, so the arms match the composed cells."""
    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def _build_model(geometry, cell):
    import mbirtorch

    num_views = cell[0]
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        return mbirtorch.ParallelBeamModel(cell, angles, device="cuda")
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    source_detector_dist = 4 * cell[2]
    source_iso_dist = source_detector_dist
    return mbirtorch.ConeBeamModel(cell, angles,
                                   source_detector_dist=source_detector_dist,
                                   source_iso_dist=source_iso_dist)


# ── the worker: one arm, one process ─────────────────────────────────────────
def run_arm(cfg):
    """One (geometry, cell, arm) measurement, in its own process."""
    import torch

    import mbirtorch

    geometry, cell, arm = cfg["geometry"], tuple(cfg["cell"]), cfg["arm"]
    result = dict(cfg)

    model = _build_model(geometry, cell)
    model.set_params(no_warning=True, verbose=0)
    # A single device explicitly, so the reading is the n=1 cell whatever the
    # allocation's GPU count is.
    model.configure_devices(1)

    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    result["num_pixels_full"] = int(model.full_index_count())

    sinogram = np.load(_sino_path(geometry, cell))
    weights = None if arm == "unweighted" else _weights(sinogram)

    # Which bodies actually ran, so a fallback cannot be mistaken for a
    # kernel reading (the arm check the gate campaigns established).
    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__
    result["kernels_on"] = ("triton" in fwd_body.__name__
                            and "triton" in back_body.__name__)

    def one_recon():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        return recon

    # Cold pass: pays every compile, so the warm pass measures steady state.
    start = time.time()
    one_recon()
    torch.cuda.synchronize()
    result["cold_s"] = time.time() - start

    # Warm pass under the calibration mode, which owns the peak counter.
    os.environ["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
    start = time.time()
    one_recon()
    torch.cuda.synchronize()
    result["warm_s"] = time.time() - start

    rows = model.last_memory_calibration or []
    result["calibration"] = [
        dict(device=str(device), modeled_bytes=int(modeled),
             measured_bytes=int(measured), ratio=float(ratio))
        for device, modeled, measured, ratio in rows]

    # The phase breakdown, so an out-of-band ratio can be attributed to a
    # phase rather than to the model as a whole.
    ledger = model.last_memory_ledger
    if ledger is not None:
        result["phases"] = [
            dict(name=phase.name, bytes=int(phase.per_device[0]))
            for phase in ledger.phases]
        result["dominant_phase"] = ledger.dominant_phase(0).name
    return result


def generate(cfg):
    """Build one cell's phantom and sinogram once, for every arm to share."""
    import torch

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    model = _build_model(geometry, cell)
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(1)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = model.forward_project(phantom)
    if not isinstance(sinogram, np.ndarray):
        sinogram = sinogram.detach().cpu().numpy()
    np.save(_sino_path(geometry, cell), np.asarray(sinogram, dtype=np.float32))
    del phantom, sinogram, model
    torch.cuda.empty_cache()
    return dict(cfg, path=_sino_path(geometry, cell))


# ── the driver ───────────────────────────────────────────────────────────────
def _spawn(cfg):
    """Run one configuration in a fresh interpreter and return its dict."""
    payload = json.dumps(cfg)
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:])


def _selected(env_var, allowed, cast=str):
    raw = os.environ.get(env_var, "")
    if not raw:
        return list(allowed)
    picked = [cast(v.strip()) for v in raw.split(",") if v.strip()]
    unknown = [v for v in picked if v not in allowed]
    if unknown:
        raise SystemExit(f"{env_var}: unknown {unknown}; allowed {list(allowed)}")
    return picked


def main():
    geometries = _selected("DP2_GEOMETRIES", GEOMETRIES)
    view_counts = _selected("DP2_CELLS", [c[0] for c in CELLS], int)
    cells = [c for c in CELLS if c[0] in view_counts]
    arms = _selected("DP2_ARMS", ARMS)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"dp2_ledger_calib_{stamp}.jsonl")
    rows = []

    def record(row):
        rows.append(row)
        with open(out_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    for geometry in geometries:
        for cell in cells:
            if not os.path.exists(_sino_path(geometry, cell)):
                print(f"generating {geometry} {cell}", flush=True)
                record(_spawn(dict(mode="generate", geometry=geometry,
                                   cell=list(cell))))

    for geometry in geometries:
        for cell in cells:
            if "weighted" not in arms:
                continue
            print(f"measuring {geometry} {cell} weighted", flush=True)
            record(_spawn(dict(mode="arm", geometry=geometry,
                               cell=list(cell), arm="weighted")))

    if "unweighted" in arms:
        cell = UNWEIGHTED_CELL
        if cell in cells and UNWEIGHTED_GEOMETRY in geometries:
            print(f"measuring {UNWEIGHTED_GEOMETRY} {cell} unweighted",
                  flush=True)
            record(_spawn(dict(mode="arm", geometry=UNWEIGHTED_GEOMETRY,
                               cell=list(cell), arm="unweighted")))

    summarize(rows, out_path)


def summarize(rows, out_path):
    print(f"\n===== ledger calibration ({out_path}) =====")
    header = (f'{"geometry":>10}{"views":>7}{"arm":>12}{"kernels":>9}'
              f'{"modeled":>11}{"measured":>11}{"ratio":>8}{"verdict":>9}'
              f'  dominant phase')
    print(header)
    print("-" * len(header))
    failures = []
    for row in rows:
        if row.get("mode") != "arm":
            continue
        if row.get("error"):
            print(f'{row["geometry"]:>10}{row["cell"][0]:>7}{row["arm"]:>12}'
                  f'   ERROR: {row["error"].splitlines()[-1][:60]}')
            failures.append(row)
            continue
        for entry in row.get("calibration", []):
            ratio = entry["ratio"]
            verdict = ("UNDER" if ratio < BAND[0]
                       else "over" if ratio > BAND[1] else "ok")
            if verdict != "ok":
                failures.append(row)
            print(f'{row["geometry"]:>10}{row["cell"][0]:>7}{row["arm"]:>12}'
                  f'{str(row.get("kernels_on")):>9}'
                  f'{entry["modeled_bytes"] / 2 ** 30:>10.2f}G'
                  f'{entry["measured_bytes"] / 2 ** 30:>10.2f}G'
                  f'{ratio:>8.3f}{verdict:>9}'
                  f'  {row.get("dominant_phase", "")}')
    print("-" * len(header))
    print(f"acceptance band {BAND[0]:.2f} <= modeled/measured <= {BAND[1]:.2f}; "
          f"{len(failures)} row(s) outside it or failed")

    # The residency the unweighted arm exists to size.
    weighted = {(r["geometry"], r["cell"][0]): r for r in rows
                if r.get("mode") == "arm" and r.get("arm") == "weighted"
                and not r.get("error")}
    for row in rows:
        if row.get("mode") != "arm" or row.get("arm") != "unweighted":
            continue
        if row.get("error"):
            continue
        pair = weighted.get((row["geometry"], row["cell"][0]))
        if not pair:
            continue
        un = row["calibration"][0]["measured_bytes"]
        wt = pair["calibration"][0]["measured_bytes"]
        print(f"\nhess_weights probe ({row['geometry']} {row['cell'][0]}): "
              f"unweighted measured {un / 2 ** 30:.2f} GB vs weighted "
              f"{wt / 2 ** 30:.2f} GB, difference {(un - wt) / 2 ** 30:+.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        out = generate(cfg) if cfg["mode"] == "generate" else run_arm(cfg)
        print("__RESULT__" + json.dumps(out))
    else:
        main()
