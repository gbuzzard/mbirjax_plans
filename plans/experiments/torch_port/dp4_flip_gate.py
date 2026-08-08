"""The device-policy flip gate: correctness of the automatic device count.

The flip makes a CUDA reconstruction spread across the devices that can hold
their share.  This gate asks four questions on real hardware, and none of
them is a performance question -- item 3 owns that readout.

1. Does the automatic path pick the device count it should, and does it
   install the SAME layout an explicit `configure_devices` call installs?
   The two arms differ only in WHEN the layout is installed, so their values
   must agree at the single-precision floor.  Not bitwise: the back kernel is
   bound in both arms and is not bit-reproducible across runs, which the
   kernel campaign already documented.  The floor below is six orders tighter
   than the multi-device divergence the other checks read, so it still
   catches any real difference in what the two paths install.
2. Do the n=2 and n=4 reconstructions agree with n=1 at the established
   multi-device float floor?  The comparison is EAGER-to-EAGER, per the
   compile-latitude policy.  The compiled bodies generate different float
   realizations per shape, so a compiled n>1 diff measures the compiler
   rather than the engine.

   Eager here means `compile_mode='off'`, and that does NOT disable the
   Triton kernels: selection is availability-driven, not compile-driven.
   These arms therefore run whatever the shipped selection binds, which is
   the point -- they measure what a user gets.  An arm that wants the plain
   torch engine must set `MBIRTORCH_DISABLE_TRITON=1`.
3. Does the process-wide pin hold on a multi-GPU node?
4. Does the preflight refuse a run that cannot fit, before allocating?

Each arm runs in its own subprocess so one arm's allocator and compile state
cannot reach the next.

Environment:
    DP4_GEOMETRIES=parallel,cone   subset of the geometries
    DP4_RESULTS=<dir>              where the jsonl goes
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

RESULTS_DIR = os.environ.get("DP4_RESULTS", ".")
# A DIVIDING cell: the per-cell tolerance must be calibrated against the
# dividing-case floor of the same cell, per the phase 4a stage 3 finding, and
# a dividing cell measures that floor directly.
CELL = (256, 64, 64)
GEOMETRIES = ["parallel", "cone"]
VCD_ITERATIONS = 3
VCD_SEED = 4321
# The established multi-device float-divergence scale for cells this size.
# The gate reports the measured floor and flags only a gross breach.
DIVERGENCE_CEILING = 5e-3
# Two arms that install the same layout by different routes.  Measured at
# 3.4e-07 to 3.9e-07, which is the single-precision floor of the bound
# kernels; the bar is set two orders above it.
SAME_LAYOUT_CEILING = 1e-5


def _build(geometry, cell, compile_mode="off"):
    import mbirtorch

    if geometry == "parallel":
        angles = np.linspace(0, np.pi, cell[0], endpoint=False)
        # NO configure_devices here: the `auto` arm exists to observe the
        # policy's own choice, and pinning the builder would erase it.  Every
        # other arm pins itself in run_arm.
        return mbirtorch.ParallelBeamModel(cell, angles,
                                           compile_mode=compile_mode)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    sdd = 4 * cell[2]
    return mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=sdd,
                                   source_iso_dist=sdd, compile_mode=compile_mode)


def _sino_path(geometry):
    return os.path.join(RESULTS_DIR, f"_dp4_sino_{geometry}_{CELL[0]}.npy")


def _recon_path(geometry, arm):
    return os.path.join(RESULTS_DIR, f"_dp4_recon_{geometry}_{arm}.npy")


def generate(cfg):
    """One phantom and sinogram per geometry, shared by every arm."""
    import mbirtorch

    geometry = cfg["geometry"]
    model = _build(geometry, CELL)
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(1)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = model.forward_project(phantom)
    if not isinstance(sinogram, np.ndarray):
        sinogram = sinogram.detach().cpu().numpy()
    np.save(_sino_path(geometry), np.asarray(sinogram, dtype=np.float32))
    return dict(cfg, recon_shape=list(recon_shape), path=_sino_path(geometry))


def run_arm(cfg):
    """One (geometry, arm) reconstruction, in its own process."""
    import torch

    geometry, arm = cfg["geometry"], cfg["arm"]
    result = dict(cfg, visible_devices=torch.cuda.device_count())

    model = _build(geometry, CELL)
    model.set_params(no_warning=True, verbose=0)

    # `auto` is the arm under test: it says nothing about devices and lets
    # the policy choose.  Every other arm pins itself.
    if arm == "auto":
        pass
    elif arm == "pinned_env":
        os.environ["MBIRTORCH_NUM_DEVICES"] = "1"
    elif arm == "doomed":
        # Nothing can satisfy this margin, so the preflight must refuse the
        # run rather than start it.
        model.memory_preflight_margin = 1e6
    else:
        model.configure_devices(int(arm.split("_")[1]))

    sinogram = np.load(_sino_path(geometry))
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)

    from mbirtorch._memory_ledger import MemoryPreflightError
    start = time.time()
    try:
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
    except MemoryPreflightError as error:
        result["preflight_error"] = str(error)
        result["refused"] = True
        return result
    result["refused"] = False
    result["seconds"] = time.time() - start
    result["devices_used"] = model.sino_placement.n_devices
    result["layout_is_automatic"] = model.device_layout_is_automatic
    np.save(_recon_path(geometry, arm), np.asarray(recon, dtype=np.float32))
    return result


def _spawn(cfg):
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker",
         json.dumps(cfg)], capture_output=True, text=True)
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:])


def _rel(a, b):
    denom = max(float(np.max(np.abs(a))), 1e-30)
    return float(np.max(np.abs(a - b)) / denom)


def main():
    import torch

    visible = torch.cuda.device_count()
    geometries = [g for g in GEOMETRIES
                  if g in os.environ.get("DP4_GEOMETRIES", ",".join(GEOMETRIES))]
    counts = [n for n in (1, 2, 4) if n <= visible]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"dp4_flip_gate_{stamp}.jsonl")
    rows = []

    def record(row):
        rows.append(row)
        with open(out_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

    print(f"visible CUDA devices: {visible}")
    for geometry in geometries:
        if not os.path.exists(_sino_path(geometry)):
            record(_spawn(dict(mode="generate", geometry=geometry)))
        arms = [f"pinned_{n}" for n in counts] + ["auto", "pinned_env",
                                                  "doomed"]
        for arm in arms:
            print(f"  {geometry} {arm}", flush=True)
            record(_spawn(dict(mode="arm", geometry=geometry, arm=arm)))

    summarize(rows, geometries, visible, counts, out_path)


def summarize(rows, geometries, visible, counts, out_path):
    by = {(r.get("geometry"), r.get("arm")): r for r in rows
          if r.get("mode") == "arm"}
    print(f"\n===== device-policy flip gate ({out_path}) =====")
    print(f"visible CUDA devices: {visible}\n")
    failures = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        if not ok:
            failures.append(label)

    for geometry in geometries:
        print(f"--- {geometry} {CELL} ---")
        auto = by.get((geometry, "auto"), {})
        if auto.get("error"):
            check(f"{geometry}: auto arm ran", False,
                  f" -- {auto['error'].splitlines()[-1][:70]}")
            continue

        # 1. the automatic count, and its identity with an explicit call
        used = auto.get("devices_used")
        check(f"{geometry}: automatic count == visible ({visible})",
              used == visible, f" -- chose {used}")
        check(f"{geometry}: automatic layout stays automatic",
              auto.get("layout_is_automatic") is True)
        explicit = by.get((geometry, f"pinned_{visible}"), {})
        if not explicit.get("error") and not explicit.get("refused"):
            a = np.load(_recon_path(geometry, "auto"))
            b = np.load(_recon_path(geometry, f"pinned_{visible}"))
            rel = _rel(a, b)
            check(f"{geometry}: automatic == explicit configure_devices"
                  f"({visible})", rel < SAME_LAYOUT_CEILING,
                  f" -- rel {rel:.2e}")

        # 2. the n>1 value floor, eager to eager
        if 1 in counts and not by.get((geometry, "pinned_1"), {}).get("error"):
            ref = np.load(_recon_path(geometry, "pinned_1"))
            for n in counts[1:]:
                row = by.get((geometry, f"pinned_{n}"), {})
                if row.get("error") or row.get("refused"):
                    check(f"{geometry}: n={n} ran", False)
                    continue
                rel = _rel(ref, np.load(_recon_path(geometry, f"pinned_{n}")))
                check(f"{geometry}: n={n} vs n=1 eager floor",
                      rel < DIVERGENCE_CEILING, f" -- rel {rel:.2e}")

        # 3. the process-wide pin
        pinned = by.get((geometry, "pinned_env"), {})
        check(f"{geometry}: MBIRTORCH_NUM_DEVICES=1 holds",
              pinned.get("devices_used") == 1,
              f" -- used {pinned.get('devices_used')}")

        # 4. the doomed run
        doomed = by.get((geometry, "doomed"), {})
        message = doomed.get("preflight_error", "")
        check(f"{geometry}: impossible budget is refused",
              doomed.get("refused") is True)
        check(f"{geometry}: refusal names the dominant phase",
              "dominant phase" in message)
        check(f"{geometry}: refusal names a remedy",
              "back_project_slice_band" in message)
        if message:
            print("      first lines of the refusal:")
            for line in message.splitlines()[:4]:
                print(f"        {line}")
        print()

    print(f"{len(failures)} check(s) failed"
          + ("" if not failures else ": " + "; ".join(failures)))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        out = generate(cfg) if cfg["mode"] == "generate" else run_arm(cfg)
        print("__RESULT__" + json.dumps(out))
    else:
        main()
