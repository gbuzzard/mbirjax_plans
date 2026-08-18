"""mg21b -- does the cone back kernel slow down when its band length is not
divisible by 16?

WHY THIS RUN EXISTS.

mg21 split the cone back projection's time into named parts at the 2048-class
cell and found that the Triton kernel itself is what grows when a fourth
device is added: 24.3 s at three devices against 45.8 s at four on the busiest
device, while the builders, the channel-major copy, the accumulation, and the
reduce all stay flat or fall.  Per unit of voxel work the four-device kernel
runs about 2.5 times slower.  That refutes the recorded per-band-call
hypothesis (mg21's band_half variant moved the cost by 12 percent at three
devices and 2 percent at four), and it leaves the kernel's own efficiency as
the open question.

One difference between the two arms stands out.  At three devices each band is
672 slices, and 672 is divisible by 16.  At four devices each band is 504
slices, and 504 is not.  The band length is a runtime argument of the kernel,
and it is also the row stride of the kernel's output writes.  Triton
specializes a kernel on whether each integer argument is divisible by 16, and
mg20 measured exactly this mechanism on the parallel forward kernel's width
argument: the unspecialized compilation uses more registers, which caps
occupancy, and the kernel runs at about half efficiency (findings §1.19).
The 1024-class history fits the same reading: the cone back projection was
slower at two devices (band 504) than at one (band 1008), and splitting paid
at three (band 336), which is where the band happens to be divisible by 16
again.

mg21's own arms cannot settle this, because band length and device count moved
together there.  This run varies the band length alone: one device, one
sinogram, one pixel set, one view span, and eight band lengths -- four
divisible by 16 and four not, interleaved so a thermal drift cannot masquerade
as the effect.

THE PREDICTION, stated before the run: the four divisible bands share one
per-unit-work rate, the four non-divisible bands share a rate about 2 to 2.5
times worse, and the boundary is divisibility rather than size (496 and 512
fast, 504 slow, although the three differ by under 2 percent in work).

THE CELL.  Cone with the 2048-class detector, (256, 2016, 1984) as (views,
detector rows, channels).  256 views keep the sinogram at 3.8 GiB on one
device; the detector and the reconstruction grid are the production cell's, so
the kernel sees production-shaped work.  The pixel set is the full mask,
3,088,364 pixels.  The band lengths swept:

    672   16 x 42     the three-device band at the 2048 class
    504   not         the four-device band at the 2048 class, and the
                      two-device band at the 1024 class
    512   16 x 32     504 padded up to the next multiple of 16
    496   16 x 31     just under 504, so work is nearly 504's
    336   16 x 21     the three-device band at the 1024 class
    344   not         just over 336
    256   16 x 16     252 padded up
    252   not         the four-device band at the 1024 class

HOW THE CALL IS MADE.  Directly through
``Projectors.sparse_back_project_view_range`` with ``slice_start=0`` and
``band_slices`` set to the arm's band -- the same method the banded driver
calls once per (device, band), on the same two-fan path that hands the body
the full detector rows.  No multi-device driver runs, so there are no bands
to walk and no reduce; one call is one band.  The wall of one call is
builders + copy + kernel + accumulation, and mg21 measured the non-kernel
parts of such a call at about 5 ms per body call against per-call walls of
seconds, so the wall discriminates a 2x kernel effect cleanly.  The reported
rate divides the wall by (views x band slices), which is what makes bands of
different lengths comparable.

WHAT THE MODEL IS FOR.  The model supplies the geometry, the placement, and
the projectors; no reconstruction runs and no reconstruction-sized array is
allocated.  The device is set explicitly with ``configure_devices``, and
``skip_memory_preflight`` is set first, because the preflight prices a full
reconstruction this run never performs.  Both settings are recorded on the
row.

THE INPUT is a seeded uniform sinogram drawn directly on the device.  A back
projection's time does not depend on the sinogram's values, and no staged
artifact is needed at this size.

THE VALUES WITNESS.  Every arm's output agrees with the 672 arm's on the
slices they share, at 1e-6 relative.  The per-slice sums are the same
arithmetic in every arm; what may differ between compilations is instruction
scheduling and contraction, which is float-level.  A failure here means the
band plumbing is wrong, not that the timing is noisy.

THE EXIT CODE reports instrument health only: every arm ran, every witness
was computed, and the device was the one asked for.  The rates and the
verdict are read by a person from the table this run prints.  The throttle
sample before and after matters more than usual, because the finding is an
efficiency ratio; a hot or power-capped device is recorded on the row.

Run:
    <torch python> mg21b_band_divisibility.py      on one GPU
    MG21B_SMOKE=1 <python> mg21b_band_divisibility.py   tiny CPU pass

Configuration is by environment variable only; there is no command line.
    MG21B_RESULTS=<dir>     where the jsonl goes
    MG21B_SMOKE=1           tiny cell on the CPU: proves the plumbing only
"""

import json
import os
import platform
import subprocess
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG21B_SMOKE", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (256, 2016, 1984)          # (views, detector rows, channels)
SMOKE_CELL = (16, 24, 20)

#: The swept band lengths, interleaved divisible / not divisible so a drift in
#: clocks over the run cannot line up with the effect.  Each entry: (band,
#: divisible by 16).
BANDS = [(672, True), (504, False), (512, True), (496, True),
         (336, True), (344, False), (256, True), (252, False)]
SMOKE_BANDS = [(16, True), (12, False), (8, True), (10, False)]

#: The slice_start arms, added 2026-08-17 evening after the design review
#: asked whether the band start shares the band length's effect.  The start
#: is also an integer argument Triton specializes on divisibility by 16, and
#: two of the four production band starts at four devices (504 and 1512) are
#: not divisible.  Unlike the band length, the start never enters address
#: arithmetic: it is added to the slice index as a float.  Each entry:
#: (band, slice_start, start divisible by 16).  The (512, 0) row of the main
#: sweep is the baseline these compare against.  The witness is skipped for
#: these arms, because a shifted band overlaps the reference on a shifted
#: slice range; the question here is the rate alone.
SLICE_START_ARMS = [(512, 504, False), (512, 1008, True)]
SMOKE_SLICE_START_ARMS = [(8, 4, False)]

WARMUP_REPEATS = 1                # pays the arm's Triton compile
TIMED_REPEATS = 3
SINO_SEED = 20260817
WITNESS_REL = 1e-6                # reported and judged; never stops the run

HOT_CORE_C = 85
HOT_HBM_C = 95

RESULTS_DIR = os.environ.get(
    "MG21B_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def bands():
    return SMOKE_BANDS if SMOKE else BANDS


def slice_start_arms():
    return SMOKE_SLICE_START_ARMS if SMOKE else SLICE_START_ARMS


def cell():
    return SMOKE_CELL if SMOKE else CELL


def sample_gpu_health():
    """Clock, temperature, and throttle flags, because this run's finding is a
    ratio of efficiencies and a throttled device would fake one."""
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
    """The geometry and the projectors, on one explicit device.

    ``skip_memory_preflight`` is set before the device is configured: the
    preflight prices a full reconstruction, and this run allocates only a
    sinogram, one output band, and the per-view precomputes.
    """
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    num_views, channels = shape[0], shape[2]
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    model = mbirtorch.ConeBeamModel(
        shape, angles, source_detector_dist=4.0 * channels,
        source_iso_dist=2.0 * channels)
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
        print("this run needs CUDA; use MG21B_SMOKE=1 for the CPU plumbing "
              "pass")
        return 2

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    label = "smoke" if SMOKE else "gpu"
    out_path = os.path.join(
        RESULTS_DIR, f"mg21b_band_{label}_{RUN_LABEL}_{stamp}.jsonl")

    result = dict(kind="run", smoke=SMOKE, device=DEVICE,
                  cell=list(cell()), bands=[list(b) for b in bands()],
                  warmup=WARMUP_REPEATS, timed=TIMED_REPEATS,
                  sino_seed=SINO_SEED, witness_rel=WITNESS_REL,
                  torch=torch.__version__, node=platform.node(),
                  health_before=sample_gpu_health())

    model = build_model()
    from mbirtorch import _memory_ledger
    result["torch_body_directions"] = list(
        _memory_ledger.torch_body_directions(model))
    result["torch_body_expected"] = [] if cuda else ["forward", "back"]
    result["bodies_ok"] = (result["torch_body_directions"]
                           == result["torch_body_expected"])
    device = model.torch_device
    result["device_realized"] = str(device)

    num_views, num_rows, num_channels = cell()
    generator = torch.Generator(device="cpu").manual_seed(SINO_SEED)
    sino = torch.rand((num_views, num_rows, num_channels),
                      generator=generator, dtype=torch.float32).to(device)

    indices = np.asarray(model._full_indices()).reshape(-1).astype(np.int64)
    idx = torch.as_tensor(indices, device=device)
    result["num_pixels"] = int(idx.shape[0])

    def one_call(band, slice_start=0):
        out = model.projector_functions.sparse_back_project_view_range(
            sino, idx, (0, num_views), coeff_power=1,
            slice_start=int(slice_start), band_slices=int(band), dev_index=0)
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
        # The comparable rate: seconds per (view x band slice) of work.  The
        # kernel's voxel work is pixels x band x views, and pixels is the same
        # in every arm, so this rate differs only by efficiency.
        arm["ns_per_view_slice"] = (
            arm["wall_median_s"] / (num_views * band) * 1e9)

        # The witness: every arm agrees with the first arm on the slices they
        # share.  The first arm has this run's largest band, so every later
        # band is a prefix of it.
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
              f"rate {arm['ns_per_view_slice']:7.2f} ns/(view*slice)  "
              f"spread {arm['spread']:.1%}  witness "
              f"{arm['witness_rel'] if arm['witness_rel'] is not None else float('nan'):.2e}",
              flush=True)
        del out
        if cuda:
            torch.cuda.empty_cache()

    del reference

    # The slice_start arms: same band, the start moved across the
    # divisibility boundary.  Rate only; the docstring of SLICE_START_ARMS
    # says why the witness is skipped here.
    start_rows = []
    for band, start, start_divisible in slice_start_arms():
        arm = dict(kind="slice_start_arm", band=int(band),
                   slice_start=int(start),
                   start_divisible_by_16=bool(start_divisible))
        walls = []
        out = None
        for repeat in range(WARMUP_REPEATS + TIMED_REPEATS):
            if out is not None:
                del out
                if cuda:
                    torch.cuda.empty_cache()
            if cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = one_call(band, slice_start=start)
            walls.append(time.perf_counter() - t0)
        timed = walls[WARMUP_REPEATS:]
        arm["wall_s"] = timed
        arm["wall_median_s"] = sorted(timed)[len(timed) // 2]
        arm["spread"] = ((max(timed) - min(timed)) / arm["wall_median_s"]
                         if arm["wall_median_s"] > 0 else None)
        arm["ns_per_view_slice"] = (
            arm["wall_median_s"] / (num_views * band) * 1e9)
        start_rows.append(arm)
        print(f"  band {band:5d}  start {start:5d}  start divisible "
              f"{str(start_divisible):5s}  median {arm['wall_median_s']:8.3f} s"
              f"  rate {arm['ns_per_view_slice']:7.2f} ns/(view*slice)  "
              f"spread {arm['spread']:.1%}", flush=True)
        del out
        if cuda:
            torch.cuda.empty_cache()
    rows.extend(start_rows)

    result["health_after"] = sample_gpu_health()
    hot = [g for g in result["health_after"]
           if (g.get("temp_c") or 0) >= HOT_CORE_C
           or (g.get("mem_temp_c") or 0) >= HOT_HBM_C or g.get("throttle")]
    result["gpu_hot_or_throttled"] = bool(hot)
    if not result["bodies_ok"]:
        checks_failed.append(
            f"bodies {result['torch_body_directions']} against "
            f"{result['torch_body_expected']}")

    with open(out_path, "w") as handle:
        handle.write(json.dumps(result) + "\n")
        for arm in rows:
            handle.write(json.dumps(arm) + "\n")

    print("\n===== the verdict table: rate by divisibility =====")
    fast = [a for a in rows if a.get("kind") == "arm"
            and a["divisible_by_16"]]
    slow = [a for a in rows if a.get("kind") == "arm"
            and not a["divisible_by_16"]]
    for group, name in ((fast, "divisible by 16"), (slow, "not divisible")):
        rates = sorted(a["ns_per_view_slice"] for a in group)
        median = rates[len(rates) // 2] if rates else float("nan")
        print(f"  {name:>16}: bands {[a['band'] for a in group]}, "
              f"median rate {median:.2f} ns/(view*slice)")
    if fast and slow:
        f_rates = sorted(a["ns_per_view_slice"] for a in fast)
        s_rates = sorted(a["ns_per_view_slice"] for a in slow)
        ratio = (s_rates[len(s_rates) // 2] / f_rates[len(f_rates) // 2])
        print(f"  slow over fast: {ratio:.2f}x.  The prediction was 2.0 to "
              f"2.5x with the boundary at divisibility, not size.")
    if start_rows:
        base = next((a for a in rows if a.get("kind") == "arm"
                     and a["band"] == start_rows[0]["band"]), None)
        print("\n===== the slice_start arms, against the same band at "
              "start 0 =====")
        if base is not None:
            print(f"  baseline band {base['band']} start 0: "
                  f"{base['ns_per_view_slice']:.2f} ns/(view*slice)")
        for arm in start_rows:
            print(f"  band {arm['band']} start {arm['slice_start']} "
                  f"(divisible {arm['start_divisible_by_16']}): "
                  f"{arm['ns_per_view_slice']:.2f} ns/(view*slice)")
        print("  a start that matters shows here as a slow rate at a fast "
              "band; the band length's own verdict is above")
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
