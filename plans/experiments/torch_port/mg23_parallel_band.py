"""mg23 -- the parallel back kernel's band rates on the padded tree.

WHY THIS RUN EXISTS.

The band-padding remedy landed in the kernel wrappers
(back_remedy_design.md, ruled 2026-08-18).  Its gate run owes one arm for the
parallel back kernel: the design note's audit point says the parallel
kernels are measured rather than assumed.  Before the padding, the parallel
back's sensitivity to non-divisible bands could only be bounded from
composed readings (about 1.5x per unit of work at the 512-class cells,
against cone's 2.5x); on the padded tree the un-padded configuration no
longer exists to measure, so the question this run answers is the one that
matters going forward: does the padded parallel back run every band at one
rate?

THE PREDICTION, stated before the run: all eight bands below run at one
per-unit-work rate, divisible and not alike, because the wrapper now rounds
the row count up to a multiple of 16 before the launch.  A band that still
ran slow would say the pad missed a use of the width argument.

THE CELL.  Parallel beam with the 2048-class detector, (256, 2016, 1984) as
(views, detector rows, channels), half a turn of views.  Parallel is
row-aligned, so a back call's band is a range of detector ROWS handed to it
as a sliced input; the call takes no band arguments.  The pixel set is the
full mask, 3,088,364 pixels.  The bands swept are mg21b's eight: 672, 504,
512, 496, 336, 344, 256, 252 -- four divisible by 16 and four not,
interleaved so a drift in clocks cannot line up with divisibility.

THE VALUES WITNESS.  Restricting the input rows to a band produces exactly
the first band columns of a wider call's output, so every arm compares
against the 672 arm on the columns they share, at 1e-6 relative.

THE EXIT CODE reports instrument health only; the rates and the verdict are
read by a person from the table this run prints.

Run:
    <torch python> mg23_parallel_band.py      on one GPU
    MG23_SMOKE=1 <python> mg23_parallel_band.py   tiny CPU pass

Configuration is by environment variable only; there is no command line.
    MG23_RESULTS=<dir>     where the jsonl goes
    MG23_SMOKE=1           tiny cell on the CPU: proves the plumbing only
"""

import json
import os
import platform
import subprocess
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG23_SMOKE", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (256, 2016, 1984)          # (views, detector rows, channels)
SMOKE_CELL = (16, 24, 20)

#: The swept row bands, mg21b's list: interleaved divisible / not divisible.
BANDS = [(672, True), (504, False), (512, True), (496, True),
         (336, True), (344, False), (256, True), (252, False)]
SMOKE_BANDS = [(16, True), (12, False), (8, True), (10, False)]

WARMUP_REPEATS = 1                # pays the arm's Triton compile
TIMED_REPEATS = 3
SINO_SEED = 20260818
WITNESS_REL = 1e-6

HOT_CORE_C = 85
HOT_HBM_C = 95

RESULTS_DIR = os.environ.get(
    "MG23_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def bands():
    return SMOKE_BANDS if SMOKE else BANDS


def cell():
    return SMOKE_CELL if SMOKE else CELL


def sample_gpu_health():
    fields = ("index,clocks.sm,temperature.gpu,temperature.memory,"
              "clocks_throttle_reasons.hw_thermal_slowdown,"
              "clocks_throttle_reasons.sw_thermal_slowdown,"
              "clocks_throttle_reasons.hw_power_brake_slowdown,"
              "clocks_throttle_reasons.sw_power_cap")
    names = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except Exception:                                             # noqa: BLE001
        return []
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 8:
            continue

        def gi(text):
            try:
                return int(float(text))
            except (TypeError, ValueError):
                return None

        out.append(dict(index=gi(parts[0]), sm_mhz=gi(parts[1]),
                        temp_c=gi(parts[2]), mem_temp_c=gi(parts[3]),
                        throttle=[name for name, value
                                  in zip(names, parts[4:8])
                                  if value.lower() == "active"]))
    return out


def build_model():
    """The geometry and the projectors on one explicit device.  The memory
    preflight prices a full reconstruction this run never performs, so it is
    skipped and both settings are recorded on the row."""
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles)
    model.skip_memory_preflight = True
    model.configure_devices(devices=[DEVICE + (":0" if DEVICE == "cuda"
                                               else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def main():
    import numpy as np
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    if DEVICE == "cuda" and not cuda:
        print("this run needs CUDA; use MG23_SMOKE=1 for the CPU plumbing "
              "pass")
        return 2

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = "smoke" if SMOKE else "gpu"
    out_path = os.path.join(
        RESULTS_DIR, f"mg23_pband_{label}_{RUN_LABEL}_{stamp}.jsonl")

    result = dict(kind="run", smoke=SMOKE, device=DEVICE,
                  cell=list(cell()), bands=[list(b) for b in bands()],
                  warmup=WARMUP_REPEATS, timed=TIMED_REPEATS,
                  sino_seed=SINO_SEED, witness_rel=WITNESS_REL,
                  torch=torch.__version__, node=platform.node(),
                  health_before=sample_gpu_health())

    model = build_model()
    from mbirtorch import _memory_ledger
    from mbirtorch._utils import padded_kernel_width
    result["torch_body_directions"] = list(
        _memory_ledger.torch_body_directions(model))
    result["torch_body_expected"] = [] if cuda else ["forward", "back"]
    result["bodies_ok"] = (result["torch_body_directions"]
                           == result["torch_body_expected"])
    result["padding_helper_check"] = (padded_kernel_width(504) == 512)
    device = model.torch_device
    result["device_realized"] = str(device)

    num_views, num_rows, num_channels = cell()
    generator = torch.Generator(device="cpu").manual_seed(SINO_SEED)
    sino = torch.rand((num_views, num_rows, num_channels),
                      generator=generator, dtype=torch.float32).to(device)

    indices = np.asarray(model._full_indices()).reshape(-1).astype(np.int64)
    idx = torch.as_tensor(indices, device=device)
    result["num_pixels"] = int(idx.shape[0])

    def one_call(band):
        # Row-aligned: the band is carried by the INPUT's row extent, and the
        # call takes no band arguments.  Slicing from row zero keeps every
        # arm's rows a prefix of the reference arm's.
        out = model.projector_functions.sparse_back_project_view_range(
            sino[:, :int(band), :], idx, (0, num_views), coeff_power=1,
            dev_index=0)
        if cuda:
            torch.cuda.synchronize(device)
        return out

    reference = None
    rows = []
    checks_failed = []
    for band, divisible in bands():
        arm = dict(kind="arm", band=int(band), divisible_by_16=bool(divisible))
        walls = []
        out = None
        for repeat in range(WARMUP_REPEATS + TIMED_REPEATS):
            if out is not None:
                del out
                if cuda:
                    torch.cuda.empty_cache()
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = one_call(band)
            walls.append(time.perf_counter() - start)
        arm["wall_warmup_s"] = walls[:WARMUP_REPEATS]
        timed = walls[WARMUP_REPEATS:]
        arm["wall_s"] = timed
        arm["wall_median_s"] = sorted(timed)[len(timed) // 2]
        arm["spread"] = ((max(timed) - min(timed)) / arm["wall_median_s"]
                         if arm["wall_median_s"] > 0 else None)
        arm["ns_per_view_row"] = (
            arm["wall_median_s"] / (num_views * band) * 1e9)

        if reference is None:
            reference = out.detach().clone()
            arm["witness_rel"] = 0.0
            arm["witness_ok"] = True
        else:
            shared = min(int(band), int(reference.shape[1]))
            diff = (out[:, :shared].double()
                    - reference[:, :shared].double())
            denom = float(reference[:, :shared].double().abs().max())
            rel = (float(diff.abs().max()) / denom) if denom > 0 else None
            arm["witness_rel"] = rel
            arm["witness_ok"] = (rel is not None and rel <= WITNESS_REL)
            if not arm["witness_ok"]:
                checks_failed.append(
                    f"band {band}: witness {rel} against {WITNESS_REL}")
            del diff
        rows.append(arm)
        print(f"  band {band:5d}  divisible {str(divisible):5s}  "
              f"median {arm['wall_median_s']:8.3f} s  "
              f"rate {arm['ns_per_view_row']:7.2f} ns/(view*row)  "
              f"spread {arm['spread']:.1%}  witness "
              f"{arm['witness_rel'] if arm['witness_rel'] is not None else float('nan'):.2e}",
              flush=True)
        del out
        if cuda:
            torch.cuda.empty_cache()

    del reference
    result["health_after"] = sample_gpu_health()
    hot = [g for g in result["health_after"]
           if (g.get("temp_c") or 0) >= HOT_CORE_C
           or (g.get("mem_temp_c") or 0) >= HOT_HBM_C or g.get("throttle")]
    result["gpu_hot_or_throttled"] = bool(hot)
    if not result["bodies_ok"]:
        checks_failed.append(
            f"bodies {result['torch_body_directions']} against "
            f"{result['torch_body_expected']}")
    if not result["padding_helper_check"]:
        checks_failed.append("padded_kernel_width(504) != 512; this is not "
                             "the padded tree")

    with open(out_path, "w") as handle:
        handle.write(json.dumps(result) + "\n")
        for arm in rows:
            handle.write(json.dumps(arm) + "\n")

    print("\n===== the verdict table: rate by divisibility, PADDED tree =====")
    fast = [a for a in rows if a["divisible_by_16"]]
    slow = [a for a in rows if not a["divisible_by_16"]]
    for group, name in ((fast, "divisible by 16"), (slow, "not divisible")):
        rates = sorted(a["ns_per_view_row"] for a in group)
        median = rates[len(rates) // 2] if rates else float("nan")
        print(f"  {name:>16}: bands {[a['band'] for a in group]}, "
              f"median rate {median:.2f} ns/(view*row)")
    if fast and slow:
        f_rates = sorted(a["ns_per_view_row"] for a in fast)
        s_rates = sorted(a["ns_per_view_row"] for a in slow)
        ratio = (s_rates[len(s_rates) // 2] / f_rates[len(f_rates) // 2])
        print(f"  non-divisible over divisible: {ratio:.2f}x.  The "
              f"prediction on the padded tree is 1.0x: the wrapper rounds "
              f"every band up before the launch, so divisibility should no "
              f"longer matter.")
    if result["gpu_hot_or_throttled"]:
        print("  NOTE: the device sampled hot or throttled; read the rates "
              "with that in mind.")
    print(f"\nwrote {out_path}")
    if checks_failed:
        print("instrument health: FAILED")
        for line in checks_failed:
            print("  " + line)
        return 1
    print("instrument health: ok.  The verdict is read from the table, not "
          "the exit code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
