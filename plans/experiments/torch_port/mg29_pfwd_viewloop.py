"""mg29 -- THE VIEW-LOOP VARIANT OF THE PARALLEL FORWARD KERNEL: THE SPIKE.

WHY THIS RUN EXISTS.  mg28 measured the parallel forward kernel load-bound
on the per-view reload of its values block: the launch grid carries the view
axis, so every (pixel, column, view) program re-reads its values tile, and
DRAM delivers the 3.1 GB block about 130 times per 128-view launch while the
arithmetic units run at 16 percent (findings §1.26).  The kernel's own
docstring reserved the remedy for exactly this measurement: under parallel
beam the values do not depend on the view, so moving the view axis from the
grid into an in-program loop reads each tile once instead of once per view.
Greg ruled 2026-08-18: proceed (open item B7).  This spike is the
increment's first step; the library change happens only if the spike gates,
and then re-gates composed -- a kernel spike's win is not the driver's
(lessons §5).

WHAT THE VARIANT IS.  A copy of the shipped ``_parallel_forward_kernel``
with three changes and nothing else: the grid drops its view axis, the body
loops over the views (the idiom the cone back kernel already uses), and the
per-view contract values are loaded inside that loop.  The values tile is
loaded ONCE, before the loop.  The tap arithmetic, the masks, and the
atomic adds are copied verbatim, so the variant computes the same sums in
the same per-view order and differs from the shipped kernel only by which
loop level owns the view.

THE SWEEP.  The tile now lives across 128 view iterations, so its size and
the warp count are the load-bearing knobs.  Eight configurations are swept,
starting at the shipped tile (8, 128) and growing the pixel axis; each
records Triton's own register and spill counts, because a spilled tile
lands in local memory, which is DRAM again and defeats the variant.

THE GATES, per configuration and per pixel-ladder point:
  * values: max-relative against the SHIPPED wrapper's output at 1e-5 (the
    atomic adds reorder float sums between compilations; lessons §2).
  * the baseline anchor: the shipped wrapper's full-pixel launch must sit
    within 25 percent of mg20's recorded 859.13 ms, so every speedup below
    divides by a reproduced baseline.

THE READINGS: per-launch medians on mg20's pixel ladder (full, /4, /16,
/64), the production-mixture mean rebuilt with mg20's weights, and the
speedup against the shipped wrapper per point.  The counter leg then takes
one Nsight Compute reading of the WINNING configuration at the full mask:
the confirmation is DRAM read falling from mg28's 130x of the values block
toward the few-times class, with the stall and throughput columns moving
with it.

WHAT THIS SPIKE DOES NOT DECIDE.  It does not touch a library file, it
does not pick the shipped default, and its speedup is not the composed
one: the library step re-gates end to end (the standing suites, the
two-CUDA-device arms, and a composed A/B) before anything ships.

Run:
    <torch python> mg29_pfwd_viewloop.py           on one GPU
    MG29_DRY=1 <python> mg29_pfwd_viewloop.py      print the plan and stop
    MG29_SMOKE=1 <python> mg29_pfwd_viewloop.py    tiny CPU plumbing pass
        (triton is unavailable on the CPU, so the smoke runs the ladder and
        report machinery on the torch body and records every variant
        configuration as skipped)

Configuration is by environment variable only; there is no command line.
    MG29_RESULTS=<dir>      where the jsonl and the ncu log go
    MG29_SMOKE=1 / MG29_DRY=1
    MG29_REPEATS=3          timed repeats per (configuration, ladder point)
    MG29_CONFIGS=a,b        a subset of the configurations, by name
    MG29_NCU=1              the counter attempt on the winner (default on)
"""

import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG29_SMOKE", "0") == "1"
DRY = os.environ.get("MG29_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (1024, 1008, 992)          # (views, detector rows, channels)
SMOKE_CELL = (8, 32, 20)          # width 32: a multiple of 16, so no padding
VIEW_BATCH = 128
VALUES_SEED = 20260818
NUM_PIXELS_EXPECTED = 771240      # recorded, not gated

WARMUP_REPEATS = 1
TIMED_REPEATS = max(1, int(os.environ.get("MG29_REPEATS", "3")))
VALUES_GATE_REL = 1e-5

#: mg20's measured full-pixel shipped launch (859.13 ms, job 15316533) and
#: the window the baseline must reproduce it within.  Every speedup this
#: spike reports divides by that baseline.
ANCHOR_FULLPIX_MS = 859.13
ANCHOR_WINDOW = 0.25

#: mg20's pixel ladder and production-mixture weights: (divisor of the full
#: mask, launches of one timed reconstruction at that count).
PIXEL_LADDER = ((64, 512), (16, 128), (4, 32), (1, 8))

#: The swept configurations: (name, BLOCK_P, BLOCK_R, num_warps, num_stages).
#: The first is the shipped tile carried onto the variant unchanged.  The
#: pixel axis grows from there because a taller tile amortizes the per-view
#: contract loads over more values rows; the warp count moves because the
#: tile's register residency per thread depends on it.
SWEEP_CONFIGS = (
    ("p8r128w8", 8, 128, 8, 1),
    ("p16r128w8", 16, 128, 8, 1),
    ("p32r128w8", 32, 128, 8, 1),
    ("p64r128w8", 64, 128, 8, 1),
    ("p8r128w4", 8, 128, 4, 1),
    ("p16r128w4", 16, 128, 4, 1),
    ("p16r64w4", 16, 64, 4, 1),
    ("p32r64w4", 32, 64, 4, 1),
)

# ── the counter attempt on the winner (mg28's machinery, one variant) ─────────
NCU_ENABLED = os.environ.get("MG29_NCU", "1") == "1"
NCU_LAUNCHES = 5
NCU_TIMEOUT_S = 420
NCU_PROBE_TIMEOUT_S = 180
METRICS_FULL = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
NCU_PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                          "insufficient permission")

HOT_CORE_C = 85
HOT_HBM_C = 95

RESULTS_DIR = os.environ.get(
    "MG29_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
NAME_COL = 12
# ──────────────────────────────────────────────────────────────────────────────


def cell():
    return SMOKE_CELL if SMOKE else CELL


def width():
    return int(cell()[1])


def _strict_subset(env_name, allowed):
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"{env_name}: {token!r} is not one of "
                             f"{list(allowed)}")
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return [name for name in allowed if name in chosen]


# ── GPU health (mg21b's sampler) ──────────────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


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
        out.append(dict(index=_gi(parts[0]), sm_mhz=_gi(parts[1]),
                        temp_c=_gi(parts[2]), mem_temp_c=_gi(parts[3]),
                        throttle=[n for n, v in zip(names, parts[4:8])
                                  if v.lower() == "active"]))
    return out


def health_is_hot(health):
    return any((g.get("temp_c") or 0) >= HOT_CORE_C
               or (g.get("mem_temp_c") or 0) >= HOT_HBM_C
               or g.get("throttle") for g in health)


# ── the model (mg20's construction) ───────────────────────────────────────────
def build_model():
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles)
    model.skip_memory_preflight = True
    model.configure_devices(
        devices=[DEVICE + (":0" if DEVICE == "cuda" else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def _seeded_values(torch_module, model, num_pixels, columns):
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(VALUES_SEED)
    block = torch_module.rand((int(num_pixels), int(columns)),
                              generator=generator,
                              dtype=torch_module.float32)
    return block.to(model.torch_device)


# ── the variant kernel ────────────────────────────────────────────────────────
_KERNEL_HOLDER = {}


def viewloop_kernel():
    """The view-loop variant, defined on first use so the dry run and the
    CPU smoke import no triton.

    The body is the shipped ``_parallel_forward_kernel`` with the view loop
    moved inside, in the idiom the cone back kernel already uses for its
    view loop.  The values tile is loaded once, before the loop; the
    per-(view, pixel) contract and the per-view scalars are loaded inside
    it; the tap arithmetic and the atomic-add masks are copied verbatim.
    """
    if "kernel" in _KERNEL_HOLDER:
        return _KERNEL_HOLDER["kernel"]
    from mbirtorch.triton_cone import _jit, _tap_range, _tl_abs, tl

    @_jit
    def _pfwd_viewloop_kernel(n_p_ptr, centers_ptr, w_p_c_ptr,
                              weight_scale_ptr, values_ptr, out_ptr,
                              num_views, num_pixels, num_channels, num_cols,
                              out_view_stride,
                              PSF_RADIUS: tl.constexpr,
                              BLOCK_P: tl.constexpr, BLOCK_R: tl.constexpr):
        p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        r_offs = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        p_mask = p_offs < num_pixels
        r_mask = r_offs < num_cols
        tile_mask = p_mask[:, None] & r_mask[None, :]

        # The voxel cylinders, read ONCE for all views: under parallel beam
        # the values do not depend on the view, which is the invariance this
        # variant exists to exploit.
        vals = tl.load(values_ptr + p_offs.to(tl.int64)[:, None] * num_cols
                       + r_offs[None, :], mask=tile_mask, other=0.0)

        for v in range(num_views):
            pix_base = v.to(tl.int64) * num_pixels + p_offs
            n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
            centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
            w_p_c = tl.load(w_p_c_ptr + v)
            weight_scale = tl.load(weight_scale_ptr + v)
            clip = tl.minimum(w_p_c, 1.0)
            out_view_ptr = out_ptr + v.to(tl.int64) * out_view_stride
            for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                n_tap = centers + (tc - PSF_RADIUS)
                w_chan = tl.maximum((w_p_c + 1.0) / 2.0
                                    - _tl_abs(n_p - n_tap.to(tl.float32)),
                                    0.0)
                w_chan = tl.minimum(w_chan, clip) * weight_scale
                n_chan = tl.minimum(tl.maximum(n_tap, 0), num_channels - 1)
                out_ptrs = (out_view_ptr
                            + n_chan.to(tl.int64)[:, None] * num_cols
                            + r_offs[None, :])
                tl.atomic_add(out_ptrs, w_chan[:, None] * vals,
                              mask=tile_mask
                              & ((n_tap >= 0)
                                 & (n_tap < num_channels))[:, None])

    _KERNEL_HOLDER["kernel"] = _pfwd_viewloop_kernel
    return _pfwd_viewloop_kernel


def viewloop_launch(values, pixel_indices, view_params_batch, args, config):
    """The shipped wrapper's launch, with the view axis moved out of the
    grid and the swept configuration applied.

    This is a copy of ``_parallel_forward_view_batch_triton``'s launch
    section with three changes: the grid is (pixel blocks, column blocks),
    the kernel is the view-loop variant with ``num_views`` as an argument,
    and the tile and warp constants come from ``config``.  The contract
    build, the zeroed channel-major output, the compile guard, the device
    bracket, and the returned permutation are the wrapper's.  The spike's
    cells have widths divisible by 16, asserted here, so the wrapper's
    padding branch has no work and is not copied.
    """
    import contextlib

    import torch

    from mbirtorch.parallel_beam import _parallel_hfan_math
    from mbirtorch.projectors import compile_serialized
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    _name, block_p, block_r, num_warps, num_stages = config
    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, args["num_rows"], args["num_cols"],
        args["num_channels"], args["delta_det_channel"],
        args["det_channel_offset"], args["delta_voxel"],
        args["delta_voxel_row"])
    num_views, num_pixels = n_p.shape
    num_value_cols = int(values.shape[1])
    assert num_value_cols % 16 == 0, num_value_cols
    values = values.contiguous()
    contract = [t.contiguous() for t in (n_p, centers)]
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    num_channels = int(args["num_channels"])
    out = torch.zeros((num_views, num_channels, num_value_cols),
                      dtype=torch.float32, device=values.device)
    grid = (-(-num_pixels // block_p), -(-num_value_cols // block_r))
    launch_key = ("mg29_viewloop", values.device.index,
                  int(args["psf_radius"]), block_p, block_r, num_warps,
                  num_stages, int(num_views), int(num_pixels), num_channels,
                  num_value_cols)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    kernel = viewloop_kernel()
    with torch.cuda.device(values.device), guard:
        kernel[grid](
            *contract, values, out,
            int(num_views), int(num_pixels), num_channels, num_value_cols,
            num_channels * num_value_cols,
            PSF_RADIUS=int(args["psf_radius"]), BLOCK_P=block_p,
            BLOCK_R=block_r, num_warps=num_warps, num_stages=num_stages)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    return out.permute(0, 2, 1)


def kernel_build_entries():
    """Registers and spills per compiled variant, from Triton's own cache
    (mg20's defensive reader, on the variant kernel)."""
    kernel = _KERNEL_HOLDER.get("kernel")
    if kernel is None:
        return []
    entries = []
    for attr in ("cache", "device_caches"):
        holder = getattr(kernel, attr, None)
        if not isinstance(holder, dict):
            continue
        for value in holder.values():
            group = value.values() if isinstance(value, dict) else [value]
            for compiled in group:
                if compiled is None or isinstance(compiled, (int, str)):
                    continue
                record = {}
                for field in ("n_regs", "n_spills", "shared", "num_warps",
                              "name"):
                    got = getattr(compiled, field, None)
                    if got is None:
                        meta = getattr(compiled, "metadata", None)
                        got = getattr(meta, field, None)
                        if got is None and isinstance(meta, dict):
                            got = meta.get(field)
                    if got is not None and not isinstance(got, (int, float,
                                                                str)):
                        got = str(got)
                    record[field] = got
                entries.append(record)
    return entries[:32]


# ── the comparison ────────────────────────────────────────────────────────────
def compare_blocks(candidate, reference, gate, chunks=8):
    """max|candidate - reference| / max|reference|, in view chunks (mg20's
    comparison, exact because the maximum of maxima is the maximum)."""
    if tuple(candidate.shape) != tuple(reference.shape):
        return dict(ok=False, rel=None, gate=gate,
                    reason=f"shape {list(candidate.shape)} is not the "
                           f"reference's {list(reference.shape)}")
    max_diff, max_ref = 0.0, 0.0
    step = max(1, int(reference.shape[0]) // max(1, chunks))
    for start in range(0, int(reference.shape[0]), step):
        a = candidate[start:start + step]
        b = reference[start:start + step]
        max_ref = max(max_ref, float(b.abs().max()))
        max_diff = max(max_diff, float((a - b).abs().max()))
    if max_ref <= 0.0:
        return dict(ok=False, rel=None, gate=gate,
                    reason="the reference is all zeros")
    rel = max_diff / max_ref
    return dict(ok=bool(rel <= gate), rel=rel, gate=gate,
                max_abs_diff=max_diff, max_abs_ref=max_ref)


def timed_calls(torch_module, cuda, call, repeats):
    """Median milliseconds over ``repeats`` warm calls after one discarded
    warm-up, device-synchronized around each call."""
    walls = []
    out = None
    for _repeat in range(WARMUP_REPEATS + repeats):
        if out is not None:
            del out
            out = None
            if cuda:
                torch_module.cuda.empty_cache()
        if cuda:
            torch_module.cuda.synchronize()
        start = time.perf_counter()
        out = call()
        if cuda:
            torch_module.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1e3)
    del out
    timed = walls[WARMUP_REPEATS:]
    return dict(median_ms=statistics.median(timed), all_ms=timed,
                warmup_ms=walls[:WARMUP_REPEATS],
                spread=((max(timed) - min(timed)) / statistics.median(timed)
                        if statistics.median(timed) > 0 else None))


# ── the sweep ─────────────────────────────────────────────────────────────────
def run_sweep(sink):
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    configs = _strict_subset("MG29_CONFIGS",
                             [c[0] for c in SWEEP_CONFIGS])
    header = dict(kind="run", smoke=SMOKE, device=DEVICE, cell=list(cell()),
                  width=width(), view_batch=VIEW_BATCH,
                  values_seed=VALUES_SEED, warmup=WARMUP_REPEATS,
                  timed=TIMED_REPEATS, values_gate=VALUES_GATE_REL,
                  anchor_fullpix_ms=ANCHOR_FULLPIX_MS,
                  anchor_window=ANCHOR_WINDOW, anchor_applies=not SMOKE,
                  configs=configs, torch=torch.__version__,
                  node=platform.node(), cuda=cuda, run_label=RUN_LABEL,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  health_before=sample_gpu_health())

    model = build_model()
    from mbirtorch import _memory_ledger
    header["torch_body_directions"] = list(
        _memory_ledger.torch_body_directions(model))
    header["torch_body_expected"] = [] if cuda else ["forward", "back"]
    header["bodies_ok"] = (header["torch_body_directions"]
                           == header["torch_body_expected"])
    fwd_body, _back = model._view_batch_bodies()
    header["forward_body"] = fwd_body.__name__
    header["triton_forward_bound"] = fwd_body.__name__.endswith("_triton")
    header["variants_runnable"] = bool(cuda
                                       and header["triton_forward_bound"])
    if not header["variants_runnable"]:
        header["variants_skipped_reason"] = (
            "the variant kernel needs triton on CUDA; the smoke proves the "
            "ladder, comparison, and report machinery on the torch body")

    args = model._view_batch_args()
    header["psf_radius"] = int(args["psf_radius"])
    pf = model.projector_functions
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    header["view_batch_used"] = view_batch
    view_params = pf._view_params_per_dev[0][:view_batch]
    idx_full = model.full_indices_device()
    header["num_pixels_full"] = int(idx_full.shape[0])
    header["num_pixels_matches_expected"] = (
        None if SMOKE else int(idx_full.shape[0]) == NUM_PIXELS_EXPECTED)
    values_full = _seeded_values(torch, model, header["num_pixels_full"],
                                 width())
    sink.write(json.dumps(header) + "\n")
    sink.flush()

    weight_sum = sum(w for _d, w in PIXEL_LADDER)
    baseline_points = []
    config_rows = {name: dict(kind="config", config=name,
                              params=next(c for c in SWEEP_CONFIGS
                                          if c[0] == name)[1:],
                              points=[], values_ok=True, worst_rel=0.0)
                   for name in configs}

    # Largest point first: the full-pixel point pays every configuration's
    # compile inside its discarded warm-up, and its baseline row is the
    # anchor check.
    for divisor, mixture_weight in sorted(PIXEL_LADDER):
        full = header["num_pixels_full"]
        want = max(1, full // divisor)
        step = max(1, full // want)
        idx = idx_full[::step][:want].contiguous()
        realized = int(idx.shape[0])
        values = (values_full if divisor == 1
                  else values_full[:realized].contiguous())

        def shipped():
            return fwd_body(values, idx, view_params, slice_start=0,
                            plan=None, **args)

        reference = shipped()
        base = timed_calls(torch, cuda, shipped, TIMED_REPEATS)
        base.update(kind="baseline_point", divisor=divisor,
                    mixture_weight=mixture_weight, num_pixels=realized)
        baseline_points.append(base)
        sink.write(json.dumps(base) + "\n")
        sink.flush()
        print(f'  baseline /{divisor:<3d} ({realized} px): '
              f'{base["median_ms"]:9.2f} ms', flush=True)

        for name in configs:
            row = config_rows[name]
            if not header["variants_runnable"]:
                row["skipped"] = header["variants_skipped_reason"]
                continue
            config = next(c for c in SWEEP_CONFIGS if c[0] == name)

            def variant():
                return viewloop_launch(values, idx, view_params, args,
                                       config)

            try:
                out = variant()
            except Exception:                                     # noqa: BLE001
                row["error"] = traceback.format_exc()[-2000:]
                row["values_ok"] = False
                print(f'    {name:<{NAME_COL}} ERROR (recorded)', flush=True)
                continue
            check = compare_blocks(out, reference, VALUES_GATE_REL)
            del out
            if cuda:
                torch.cuda.empty_cache()
            point = timed_calls(torch, cuda, variant, TIMED_REPEATS)
            point.update(divisor=divisor, mixture_weight=mixture_weight,
                         num_pixels=realized, values=check,
                         speedup=(base["median_ms"] / point["median_ms"]
                                  if point["median_ms"] > 0 else None))
            row["points"].append(point)
            row["values_ok"] = row["values_ok"] and bool(check.get("ok"))
            if check.get("rel") is not None:
                row["worst_rel"] = max(row["worst_rel"], check["rel"])
            print(f'    {name:<{NAME_COL}} {point["median_ms"]:9.2f} ms  '
                  f'{point["speedup"]:5.2f}x  values '
                  f'{check.get("rel", float("nan")):.2e}', flush=True)
        del reference
        if cuda:
            torch.cuda.empty_cache()

    # The mixture means: mg20's weights rebuild the production call mixture.
    base_mixture = sum(p["median_ms"] * p["mixture_weight"]
                       for p in baseline_points) / weight_sum
    for name in configs:
        row = config_rows[name]
        if row["points"]:
            row["mixture_mean_ms"] = sum(
                p["median_ms"] * p["mixture_weight"]
                for p in row["points"]) / weight_sum
            row["mixture_speedup"] = base_mixture / row["mixture_mean_ms"]
            full_point = next((p for p in row["points"]
                               if p["divisor"] == 1), None)
            row["fullpix_ms"] = (full_point or {}).get("median_ms")
            row["fullpix_speedup"] = (full_point or {}).get("speedup")
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    header["baseline_mixture_ms"] = base_mixture
    baseline_full = next((p for p in baseline_points if p["divisor"] == 1),
                         None)
    header["baseline_fullpix_ms"] = (baseline_full or {}).get("median_ms")
    if SMOKE or header["baseline_fullpix_ms"] is None:
        header["anchor_ok"] = None
    else:
        low = ANCHOR_FULLPIX_MS * (1 - ANCHOR_WINDOW)
        high = ANCHOR_FULLPIX_MS * (1 + ANCHOR_WINDOW)
        header["anchor_ok"] = bool(
            low <= header["baseline_fullpix_ms"] <= high)
    header["kernel_builds"] = kernel_build_entries()
    if cuda:
        header["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    header["health_after"] = sample_gpu_health()
    header["gpu_hot_or_throttled"] = bool(
        health_is_hot(header.get("health_before") or [])
        or health_is_hot(header["health_after"]))
    sink.write(json.dumps(dict(kind="run_tail",
                               baseline_mixture_ms=base_mixture,
                               baseline_fullpix_ms=header[
                                   "baseline_fullpix_ms"],
                               anchor_ok=header["anchor_ok"],
                               kernel_builds=header["kernel_builds"],
                               peak_bytes=header.get("peak_bytes"),
                               health_after=header["health_after"],
                               gpu_hot_or_throttled=header[
                                   "gpu_hot_or_throttled"])) + "\n")
    sink.flush()
    return header, baseline_points, config_rows


# ── the counter attempt on the winner ─────────────────────────────────────────
def one_launch(cfg):
    """Launch the winning configuration a few times at the full mask, for
    ncu.  Imports nothing that launches other kernels first."""
    import torch

    result = dict(cfg, mode="one_launch")
    model = build_model()
    args = model._view_batch_args()
    pf = model.projector_functions
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    view_params = pf._view_params_per_dev[0][:view_batch]
    idx = model.full_indices_device()
    values = _seeded_values(torch, model, int(idx.shape[0]), width())
    config = next(c for c in SWEEP_CONFIGS if c[0] == cfg["config"])
    result["num_pixels"] = int(idx.shape[0])
    result["view_batch"] = view_batch
    result["values_bytes"] = int(idx.shape[0]) * width() * 4
    for _index in range(int(cfg.get("launches", NCU_LAUNCHES))):
        out = viewloop_launch(values, idx, view_params, args, config)
        out = None
    torch.cuda.synchronize()
    result["kernel_builds"] = kernel_build_entries()
    return result


def trivial_kernel():
    import torch

    if not torch.cuda.is_available():
        return dict(mode="trivial_kernel", cuda=False)
    x = torch.ones(1 << 16, device="cuda")
    total = float((x * 2.0).sum())
    torch.cuda.synchronize()
    return dict(mode="trivial_kernel", cuda=True, checksum=total)


def _run(cmd, timeout, env=None):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return dict(returncode=proc.returncode, stdout=proc.stdout,
                    stderr=proc.stderr, wall_s=time.perf_counter() - start,
                    timed_out=False)
    except subprocess.TimeoutExpired:
        return dict(returncode=None, stdout="", stderr="",
                    wall_s=time.perf_counter() - start, timed_out=True)
    except FileNotFoundError as exc:
        return dict(returncode=None, stdout="", stderr=str(exc),
                    wall_s=time.perf_counter() - start, timed_out=False,
                    missing=True)


def _worker_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return None


def parse_ncu_csv(text):
    import csv as csv_module

    rows = list(csv_module.reader(text.splitlines()))

    def number(text_value):
        cleaned = str(text_value).replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return cleaned

    def parse_from(header_index):
        header = [c.strip() for c in rows[header_index]]
        body = rows[header_index + 1:]
        kernel_at = header.index("Kernel Name")
        if "Metric Name" in header and "Metric Value" in header:
            name_at = header.index("Metric Name")
            value_at = header.index("Metric Value")
            unit_at = (header.index("Metric Unit")
                       if "Metric Unit" in header else None)
            by_kernel = {}
            for row in body:
                if len(row) <= max(name_at, value_at, kernel_at):
                    continue
                key = row[kernel_at].strip()
                entry = by_kernel.setdefault(key, dict(kernel=key,
                                                       metrics={}, units={}))
                entry["metrics"][row[name_at].strip()] = number(row[value_at])
                if unit_at is not None and len(row) > unit_at:
                    entry["units"][row[name_at].strip()] = \
                        row[unit_at].strip()
            return list(by_kernel.values())
        out, units = [], {}
        for row in body:
            if len(row) < len(header):
                continue
            values = [number(c) for c in row]
            if all(isinstance(v, str) for v in values):
                if not units:
                    units = {name: str(v).strip()
                             for name, v in zip(header, values)
                             if name and str(v).strip()}
                continue
            entry = dict(kernel=row[kernel_at].strip(), metrics={},
                         units=dict(units))
            for name, v in zip(header, values):
                if name and name != "Kernel Name":
                    entry["metrics"][name] = v
            out.append(entry)
        return out

    best, best_score = [], (0, 0)
    for index, row in enumerate(rows):
        if not any(c.strip() == "Kernel Name" for c in row):
            continue
        try:
            parsed = parse_from(index)
        except (ValueError, IndexError):
            continue
        scored = [e for e in parsed
                  if any(isinstance(v, (int, float))
                         for v in e["metrics"].values())]
        numbers = sum(1 for e in scored for v in e["metrics"].values()
                      if isinstance(v, (int, float)))
        if (len(scored), numbers) > best_score:
            best, best_score = scored, (len(scored), numbers)
    return best


def variant_env():
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def ncu_on_winner(winner, results_dir):
    """One counter attempt on the winning configuration at the full mask.
    The confirmation is DRAM read against the values block falling from
    mg28's 130x class; everything records and nothing gates."""
    leg = dict(attempted=True, winner=winner)
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG29_NCU=0")
        return leg
    if DEVICE != "cuda" or winner is None:
        leg.update(attempted=False,
                   reason="no CUDA winner to profile" if winner is None
                   else "CPU run")
        return leg
    ncu = shutil.which("ncu")
    leg["ncu_path"] = ncu
    if ncu is None:
        leg.update(attempted=False, reason="ncu is not on PATH")
        return leg
    probe = _run([ncu, "--launch-count", "1", "--metrics",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  sys.executable, "-u", os.path.abspath(__file__),
                  "--trivial-kernel"], NCU_PROBE_TIMEOUT_S)
    blob = (probe["stdout"] or "") + (probe["stderr"] or "")
    if any(m.lower() in blob.lower() for m in NCU_PERMISSION_MARKERS) \
            or probe["timed_out"] or "sm__warps_active" not in blob:
        leg.update(profiler_permitted=False,
                   reason="counters unavailable; see the probe message",
                   probe_message=blob.strip()[-600:])
        return leg
    leg["profiler_permitted"] = True
    pattern = "regex:" + re.escape("_pfwd_viewloop_kernel")
    leg["kernel_name_filter"] = pattern
    cfg = dict(config=winner, launches=NCU_LAUNCHES)
    cmd = [ncu, "--csv", "--page", "raw", "--target-processes", "all",
           "--kernel-name", pattern, "--launch-skip",
           str(NCU_LAUNCHES - 1), "--launch-count", "1", "--metrics",
           ",".join(METRICS_FULL), sys.executable, "-u",
           os.path.abspath(__file__), "--one-launch", json.dumps(cfg)]
    got = _run(cmd, NCU_TIMEOUT_S, env=variant_env())
    log_path = os.path.join(results_dir, f"mg29_ncu_{winner}.log")
    with open(log_path, "w") as log_sink:
        log_sink.write(" ".join(cmd) + "\n\n")
        log_sink.write(got["stdout"] or "")
        log_sink.write("\n----- stderr -----\n")
        log_sink.write(got["stderr"] or "")
    leg["log"] = log_path
    leg["timed_out"] = got["timed_out"]
    leg["kernels"] = parse_ncu_csv(got["stdout"] or "")
    leg["worker"] = _worker_result(got["stdout"] or "")
    return leg


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, w=10, kind="f", prec=3):
    if value is None:
        return f'{"-":>{w}}'
    if isinstance(value, str):
        return f"{value:>{w}}"
    return f"{value:>{w}.{prec}{kind}}"


def _metric(kernel, name, default=None):
    metrics = (kernel or {}).get("metrics") or {}
    if name in metrics:
        return metrics[name]
    for key, value in metrics.items():
        if key.split(" ")[0] == name:
            return value
    return default


def _number(value):
    return value if isinstance(value, (int, float)) else None


def pick_winner(config_rows):
    """The best configuration by production-mixture speedup, among those
    whose values passed at every point; (None, None) when nothing
    qualifies."""
    winner, winner_mix = None, None
    for name, row in config_rows.items():
        if row.get("skipped") or row.get("error") or not row.get("values_ok"):
            continue
        mix = row.get("mixture_speedup")
        if mix is not None and (winner_mix is None or mix > winner_mix):
            winner, winner_mix = name, mix
    return winner, winner_mix


def summarize(header, baseline_points, config_rows, leg, out_path):
    print(f"\n===== mg29 view-loop spike ({out_path}) =====")
    print(f'\nbaseline (the shipped wrapper): full-pixel '
          f'{_fmt(header.get("baseline_fullpix_ms"), 0, "f", 2)} ms against '
          f'the {ANCHOR_FULLPIX_MS:.2f} anchor; production-mixture mean '
          f'{_fmt(header.get("baseline_mixture_ms"), 0, "f", 2)} ms')

    line = (f'{"config":<{NAME_COL}}{"P":>5}{"R":>5}{"warps":>7}'
            f'{"full ms":>10}{"full x":>8}{"mixture ms":>12}{"mix x":>8}'
            f'{"worst rel":>11}  check')
    print("\n===== the sweep, against the shipped wrapper =====")
    print(line)
    print("-" * len(line))
    for name, row in config_rows.items():
        if row.get("skipped"):
            print(f'{name:<{NAME_COL}}  SKIPPED: {row["skipped"][:80]}')
            continue
        if row.get("error"):
            print(f'{name:<{NAME_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            continue
        p, r, warps, _stages = row["params"]
        check = "ok" if row["values_ok"] else "VALUES GATE FAILED"
        print(f'{name:<{NAME_COL}}{p:>5}{r:>5}{warps:>7}'
              f'{_fmt(row.get("fullpix_ms"), 10, "f", 2)}'
              f'{_fmt(row.get("fullpix_speedup"), 8, "f", 2)}'
              f'{_fmt(row.get("mixture_mean_ms"), 12, "f", 2)}'
              f'{_fmt(row.get("mixture_speedup"), 8, "f", 2)}'
              f'{_fmt(row.get("worst_rel"), 11, "e", 2)}  {check}')
    print("-" * len(line))
    print("  'full x' and 'mix x' are the shipped wrapper's time over the "
          "variant's, at the full mask and over the production call "
          "mixture.")
    builds = header.get("kernel_builds") or []
    if builds:
        print("  Triton's compile record for the variant (one entry per "
              "compiled shape; a spilled tile lands in local memory, which "
          "is DRAM again, and disqualifies its configuration whatever the "
          "timing says):")
        for entry in builds:
            print(f'    warps {entry.get("num_warps")}: '
                  f'{entry.get("n_regs")} registers, '
                  f'{entry.get("n_spills")} spills, shared '
                  f'{entry.get("shared")}')
    winner, winner_mix = pick_winner(config_rows)
    if winner:
        print(f"\n  the winning configuration by mixture mean: {winner} "
              f"({winner_mix:.2f}x)")

    if leg.get("kernels"):
        kernel = leg["kernels"][0]
        worker = leg.get("worker") or {}
        dram_rd = _number(_metric(kernel, "dram__bytes_read.sum"))
        vals_bytes = worker.get("values_bytes")
        ratio = (dram_rd / vals_bytes
                 if (dram_rd is not None and vals_bytes) else None)
        print(f'\n===== the counter confirmation on {leg.get("winner")} '
              "=====")
        print(f'  DRAM read over the values block: {_fmt(ratio, 0, "f", 1)}'
              f' (mg28 measured the shipped kernel at 130)')
        print(f'  SM / memory throughput, percent of peak: '
              f'{_fmt(_number(_metric(kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed")), 0, "f", 1)}'
              f' / '
              f'{_fmt(_number(_metric(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")), 0, "f", 1)}'
              f'  (shipped: 15.6 / 52.3)')
        print(f'  warps waiting on memory per issue-active cycle: '
              f'{_fmt(_number(_metric(kernel, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio")), 0, "f", 1)}'
              f'  (shipped: 38.2)')
        print(f'  atomic-path sectors: '
              f'{_fmt((_number(_metric(kernel, "lts__t_sectors_op_atom.sum")) or 0) + (_number(_metric(kernel, "lts__t_sectors_op_red.sum")) or 0), 0, "e", 3)}'
              f'  (unchanged writes expected)')
    elif leg.get("attempted"):
        print(f'\ncounter confirmation: no profile '
              f'({leg.get("reason", "see the ncu log")})')

    checks = []
    if header.get("anchor_ok") is False:
        checks.append(
            f'the shipped baseline read '
            f'{header.get("baseline_fullpix_ms"):.1f} ms against the '
            f'{ANCHOR_FULLPIX_MS:.0f} ms anchor window')
    if not header.get("bodies_ok"):
        checks.append(f'torch bodies {header.get("torch_body_directions")} '
                      f'against {header.get("torch_body_expected")}')
    for name, row in config_rows.items():
        if row.get("skipped"):
            continue
        if row.get("error"):
            checks.append(f"{name} raised; the traceback is on its row")
        elif not row.get("values_ok"):
            checks.append(f"{name} failed the values gate "
                          f"(worst {row.get('worst_rel'):.2e})")
    if header.get("gpu_hot_or_throttled"):
        print("\nNOTE: the device sampled hot or throttled; the timings are "
              "read with that in mind.")

    healthy = not checks
    print(f'\nexit code reports INSTRUMENT HEALTH only: '
          f'{"healthy" if healthy else "BROKEN"}.  It covers the baseline '
          "anchor, the values gates, and no configuration raising.  The "
          "winner and its speedup are read by a person, and the LIBRARY "
          "decision waits for the composed re-gate; nothing ships from "
          "this spike.")
    for item in checks:
        print(f"  FAIL: {item}")
    return dict(kind="summary", healthy=healthy, checks=checks,
                winner=winner, winner_mixture_speedup=winner_mix,
                out_path=out_path)


def _dry_run():
    configs = _strict_subset("MG29_CONFIGS", [c[0] for c in SWEEP_CONFIGS])
    print(f"mg29 view-loop spike: device {DEVICE}, cell {tuple(cell())}, "
          f"width {width()}, view batch {VIEW_BATCH}")
    print("  open item B7's increment, step 1 of 2.  The spike decides "
          "nothing about the library; a winning configuration goes to the "
          "composed re-gate.")
    print(f"  results -> {RESULTS_DIR}")
    print(f"  baseline: the shipped wrapper on mg20's pixel ladder "
          f"{[d for d, _w in PIXEL_LADDER]}, anchored at "
          f"{ANCHOR_FULLPIX_MS:.2f} ms within {ANCHOR_WINDOW:.0%}")
    print(f'  {"config":<{NAME_COL}}{"BLOCK_P":>9}{"BLOCK_R":>9}'
          f'{"warps":>7}{"stages":>8}')
    for name, p, r, w, s in SWEEP_CONFIGS:
        marker = "" if name in configs else "  (not selected)"
        print(f"  {name:<{NAME_COL}}{p:>9}{r:>9}{w:>7}{s:>8}{marker}")
    print(f"  values gate {VALUES_GATE_REL:.0e} against the shipped "
          f"wrapper at every ladder point; {WARMUP_REPEATS} warm + "
          f"{TIMED_REPEATS} timed per point")
    print(f"  counter attempt on the winner: "
          f"{'on' if NCU_ENABLED else 'off (MG29_NCU=0)'}")
    print("  no library file is touched: the variant kernel and its launch "
          "live in this file")


def main():
    if DRY:
        _dry_run()
        return 0
    if not SMOKE:
        import torch
        if not torch.cuda.is_available():
            print("this run needs CUDA; use MG29_SMOKE=1 for the CPU "
                  "plumbing pass")
            return 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR,
                            f"mg29_pfwd_viewloop_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg29 view-loop spike on {RUN_LABEL} ({DEVICE}) -> {out_path}",
          flush=True)
    with open(out_path, "w") as sink:
        header, baseline_points, config_rows = run_sweep(sink)
        winner, _mix = pick_winner(config_rows)
        print("  counter attempt on the winner", flush=True)
        leg = ncu_on_winner(winner, RESULTS_DIR)
        sink.write(json.dumps(dict(kind="ncu_leg", **{
            k: v for k, v in leg.items() if k != "kernels"})) + "\n")
        if leg.get("kernels"):
            sink.write(json.dumps(dict(kind="ncu_kernels",
                                       kernels=leg["kernels"])) + "\n")
        summary = summarize(header, baseline_points, config_rows, leg,
                            out_path)
        sink.write(json.dumps(summary) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one-launch":
        worker_cfg = json.loads(sys.argv[2])
        try:
            worker_out = one_launch(worker_cfg)
        except Exception:                                         # noqa: BLE001
            worker_out = dict(worker_cfg,
                              error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(worker_out))
    elif len(sys.argv) > 1 and sys.argv[1] == "--trivial-kernel":
        print("__RESULT__" + json.dumps(trivial_kernel()))
    else:
        sys.exit(main())
