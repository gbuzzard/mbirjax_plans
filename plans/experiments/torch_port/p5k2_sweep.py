"""Phase 5 increment K2, part 1: the CONSTANT SWEEP for the Triton cone back
kernel -- the isolated per-view-batch bench that ranks (BLOCK_P, BLOCK_L,
num_warps, num_stages) at the two gate cells before the composed gate
(p5k2_gate.py) decides anything.

Protocol, following p3/p4: ONE (cell, config) per subprocess -- one ruler per
row, no cross-configuration cache contamination, and a config whose triton
compile blows up (registers, shared memory) costs one row instead of the run.
Each row rebuilds the same seeded inputs, so every row carries its own
kernel-vs-torch-body value check and its own honest peak.

What the row measures: the cone back BODY at one view batch, called exactly as
the driver calls it (`Projectors._effective_view_batch` supplies the view
count, `gen_full_indices` the pixel set, `_view_batch_args` the geometry) --
warmed once, then TRIALS timed calls, median reported.  The per-cell baseline
row times the torch body it replaces, eager and torch.compiled, at the same
inputs.  The optional jax row times mbirjax's pallas back projection at the
same cell for scale.

ISOLATION CAVEAT (the pallas E4 lesson, stated up front): an isolated body
bench flatters baselines and cannot settle the replacement rule.  It ranks
constants; p5k2_gate.py measures the composed recon, and that is the gate.

Run:
    <torch python> p5k2_sweep.py          on a CUDA node (see p5k2_gautschi.sbatch)
    python p5k2_sweep.py --dry-run        anywhere: print the pruned config plan
    python p5k2_sweep.py --help

Environment:
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the row subprocesses
    P5K2_SKIP_JAX=1                   skip the optional mbirjax reference rows
    P5K2_CELLS=512,1024               run a subset of the cells (by view count)
    P5K2_MAX_CONFIGS=30               cap on kept configurations per cell
    P5K2_VIEWS=<n>                    override the driver's view batch size
    P5K2_SMOKE=1                      tiny cell on P5K2_DEVICE (default cpu):
                                      exercises every step but the launch
    P5K2_DEVICE=cuda|cpu|mps          the torch device for the rows
"""

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The Phase 3/4 gate cells, with the harness magnification-2 cone convention
# (sdd = 4 * channels, sid = 2 * channels).
CELLS = [(512, 448, 384), (1024, 1008, 992)]

# Local smoke: a tiny cell on whatever device is at hand, so the harness --
# model build, the driver's view batch, the body call, the value check, the
# table -- can be exercised end to end without CUDA.  The kernel rows then
# fail with the wrapper's own "called without triton installed" error, which
# is the failure path worth seeing once before a cluster submission.
SMOKE = os.environ.get("P5K2_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K2_DEVICE", "cpu" if SMOKE else "cuda")

# The sweep grid (the design's K2 axes: pixel block, slice chunk, warps,
# stages).  The driver's view batch is the fifth axis and is NOT swept here --
# it is a memory knob owned by the transient budget, so the rows use the value
# the driver would choose and P5K2_VIEWS overrides it for a probe.
BLOCK_P_VALUES = (16, 32, 64)
BLOCK_L_VALUES = (32, 64, 128)
NUM_WARPS_VALUES = (2, 4, 8)
NUM_STAGES_VALUES = (1, 2)

# ── pruning: the register model the kernel's own comment reasons with ─────────
# The inner loop holds the (BLOCK_P, BLOCK_L) f32 accumulator plus ~6 live
# tiles of the same shape (m, inv_cos_phi, the row centers, the row weight, the
# row partial, the gathered values), all register-resident -- the kernel
# allocates no shared memory of its own, so registers, not SMEM, are the
# binding constraint (num_stages is the only axis that can put loads in shared
# memory at all, and the gathers here use computed pointers, which the
# pipeliner may decline to stage; the compiled `shared` bytes are read back per
# row rather than predicted).
LIVE_TILES = 7
# Hard cap: 255 registers/thread is the hardware ceiling on both H100 and A100,
# so an estimate past this spills by construction and would burn sweep budget
# measuring the spill.  224 keeps a little room for the estimate being crude.
MAX_REGS_PER_THREAD = 224
# A tile so small that each thread owns < 4 accumulator elements is index
# arithmetic with a rounding error attached: no reuse to amortize the tap
# loops' address math.
MIN_ELEMS_PER_THREAD = 4
# Software pipelining duplicates the in-flight tiles, so stages > 1 is swept
# only where half the budget is free.
MAX_REGS_PER_THREAD_PIPELINED = 112
# The pinned default's own estimate (triton_cone's "~96 registers per thread"),
# used to rank survivors when the cap truncates.
TARGET_REGS_PER_THREAD = 96
MAX_CONFIGS_PER_CELL = int(os.environ.get("P5K2_MAX_CONFIGS", "30"))
# Always measured, whatever the ranking says: the constants mbirtorch ships.
# Was (32, 64, 4, 1) when this sweep first ran; the sweep itself moved it.
PINNED_CONFIG = (16, 64, 4, 1)

WARMUP = 1          # pays the triton compile and the allocator's first pass
TRIALS = 3
SINO_SEED = 0       # a private generator: the recon gates read the global RNG
COEFF_POWER = 1     # the gradient path -- where the VCD loop spends its time
VALUE_REL_TOL = 1e-5                     # the design's gradient-path gate
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    """The cells to run: the smoke cell, all of CELLS, or the subset named by
    P5K2_CELLS as a comma-separated list of view counts."""
    wanted = os.environ.get("P5K2_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _drop_reason(elems_per_thread, est_regs, num_stages):
    """Why this grid point is not worth a row, or None to keep it."""
    if est_regs > MAX_REGS_PER_THREAD:
        reason = (f"est {est_regs:.0f} regs/thread > {MAX_REGS_PER_THREAD} "
                  f"(spills by construction)")
    elif elems_per_thread < MIN_ELEMS_PER_THREAD:
        reason = (f"{elems_per_thread:.1f} accumulator elements/thread < "
                  f"{MIN_ELEMS_PER_THREAD}")
    elif num_stages > 1 and est_regs > MAX_REGS_PER_THREAD_PIPELINED:
        reason = (f"pipelined (stages {num_stages}) at est {est_regs:.0f} > "
                  f"{MAX_REGS_PER_THREAD_PIPELINED}")
    else:
        reason = None
    return reason


def _grid_point(block_p, block_l, num_warps, num_stages):
    """One grid point with its cost estimates and its keep/drop verdict."""
    threads = 32 * num_warps
    elems_per_thread = block_p * block_l / threads
    est_regs = LIVE_TILES * elems_per_thread
    return dict(block_p=block_p, block_l=block_l, num_warps=num_warps,
                num_stages=num_stages,
                accumulator_bytes=4 * block_p * block_l,
                elems_per_thread=elems_per_thread,
                est_regs_per_thread=est_regs,
                drop_reason=_drop_reason(elems_per_thread, est_regs,
                                         num_stages))


def _rank_key(point):
    """Ranking for the cap: stages 1 before stages 2 (the view loop is
    gather-bound, so pipelining is the speculative axis), then closeness to the
    pinned default's register estimate, then a deterministic tie-break."""
    return (point["num_stages"] - 1,
            abs(math.log2(point["est_regs_per_thread"]
                          / TARGET_REGS_PER_THREAD)),
            point["block_p"], point["block_l"], point["num_warps"])


def build_plan():
    """(kept, dropped): the pruned, capped, ranked configuration plan.  The
    grid does not depend on the cell, so one plan serves every cell."""
    points = [_grid_point(bp, bl, w, st)
              for bp in BLOCK_P_VALUES
              for bl in BLOCK_L_VALUES
              for w in NUM_WARPS_VALUES
              for st in NUM_STAGES_VALUES]
    dropped = [p for p in points if p["drop_reason"] is not None]
    survivors = sorted((p for p in points if p["drop_reason"] is None),
                       key=_rank_key)
    kept = survivors[:MAX_CONFIGS_PER_CELL]
    pinned = [p for p in survivors if _config_tuple(p) == PINNED_CONFIG]
    if pinned and pinned[0] not in kept:
        kept = kept[:-1] + pinned          # the shipped constants always run
    for point in survivors[len(kept):]:
        point["drop_reason"] = f"beyond the {MAX_CONFIGS_PER_CELL}-config cap"
        dropped.append(point)
    return kept, dropped


def _config_tuple(point):
    return (point["block_p"], point["block_l"], point["num_warps"],
            point["num_stages"])


def print_plan(kept, dropped):
    total = (len(BLOCK_P_VALUES) * len(BLOCK_L_VALUES) * len(NUM_WARPS_VALUES)
             * len(NUM_STAGES_VALUES))
    print(f"config plan: {len(kept)} kept of {total} grid points "
          f"(cap {MAX_CONFIGS_PER_CELL}/cell), {len(dropped)} dropped")
    print(f"{'BP':>4}{'BL':>5}{'W':>3}{'ST':>3}{'acc_KB':>8}"
          f"{'elem/thr':>10}{'est_reg':>9}")
    for point in kept:
        print(f"{point['block_p']:>4}{point['block_l']:>5}"
              f"{point['num_warps']:>3}{point['num_stages']:>3}"
              f"{point['accumulator_bytes'] / 1024:>8.1f}"
              f"{point['elems_per_thread']:>10.1f}"
              f"{point['est_regs_per_thread']:>9.0f}")
    print("dropped:")
    for point in dropped:
        print(f"  {_config_tuple(point)}: {point['drop_reason']}")


# ── the row workers ───────────────────────────────────────────────────────────
def _is_kernel_cache(candidate):
    """Whether a dict holds compiled triton kernels.  Newer triton keeps the
    per-device cache inside a TUPLE alongside other launcher state, so
    membership is decided by the values (a compiled kernel carries
    ``metadata``/``n_regs``), never by position -- clearing the wrong member
    would corrupt the launcher instead of the cache.  An empty dict qualifies
    trivially, which is harmless: clearing it is a no-op."""
    return (isinstance(candidate, dict)
            and all(hasattr(v, "metadata") or hasattr(v, "n_regs")
                    for v in candidate.values()))


def _kernel_cache_dicts(jit_fn):
    """Every per-device compiled-kernel dict of a triton JITFunction, across
    the attribute renames (``cache`` -> ``device_caches``) between triton
    versions.  Empty when the shape is unrecognized -- the caller records that
    rather than assuming the clear worked."""
    dicts = []
    for attr in ("cache", "device_caches"):
        holder = getattr(jit_fn, attr, None)
        if isinstance(holder, dict):
            for value in holder.values():
                if _is_kernel_cache(value):
                    dicts.append(value)
                elif isinstance(value, (tuple, list)):
                    dicts.extend(v for v in value if _is_kernel_cache(v))
    return dicts


def _apply_constants(triton_cone, block_p, block_l, num_warps, num_stages):
    """Set the kernel's four tuning constants for the calls that follow, and
    clear both caches that sit behind a launch.

    The wrapper reads all four as MODULE GLOBALS at call time --
    ``_tile_size(CONE_BACK_BLOCK_P, num_pixels)`` and the
    ``num_warps=``/``num_stages=`` launch kwargs -- so a plain setattr takes
    effect on the next launch; nothing is baked at import.  Two caches sit
    behind that launch:

      - triton's own per-JITFunction compiled-kernel cache.  Current triton
        keys it on the constexprs AND the launch options, so a new (warps,
        stages) pair already compiles a new variant -- but a stale hit would
        silently measure the PREVIOUS configuration, which is the one failure
        mode a tuning sweep must not have, so it is cleared rather than
        trusted.  (Rows also run in separate processes, which makes this belt
        and braces; the belt is cheap.)
      - ``_COMPILED_LAUNCH_KEYS``, which decides only whether a launch takes
        the process-wide compile lock.  Its key includes the block sizes but
        NOT warps/stages, so a swept configuration would otherwise skip the
        lock that its first production launch takes.

    Returns:
        (bool, str): whether the triton cache was found and cleared, plus a
        note for the row.
    """
    triton_cone.CONE_BACK_BLOCK_P = int(block_p)
    triton_cone.CONE_BACK_BLOCK_L = int(block_l)
    triton_cone.CONE_BACK_NUM_WARPS = int(num_warps)
    triton_cone.CONE_BACK_NUM_STAGES = int(num_stages)
    triton_cone._COMPILED_LAUNCH_KEYS.clear()
    caches = _kernel_cache_dicts(triton_cone._cone_back_kernel)
    for cache in caches:
        cache.clear()
    if caches:
        note = f"cleared {len(caches)} triton cache dict(s)"
    else:
        note = ("no triton cache dict found on the JITFunction -- the row "
                "relies on triton keying its cache on warps/stages")
    return bool(caches), note


def _compiled_metadata(jit_fn):
    """What triton actually BUILT, read back from its cache: the requested
    warps/stages, the register count, and the spill count.

    This is the authoritative introspection the pruning model is only guessing
    at -- ``n_spills > 0`` is the real spill verdict, and a ``num_warps`` here
    that disagrees with the request means the constants never reached the
    compiler.  Best effort: an unrecognized triton version yields [].
    """
    entries = []
    try:
        for cache in _kernel_cache_dicts(jit_fn):
            for kernel in cache.values():
                metadata = getattr(kernel, "metadata", None)
                entries.append(dict(
                    num_warps=getattr(metadata, "num_warps", None),
                    num_stages=getattr(metadata, "num_stages", None),
                    n_regs=getattr(kernel, "n_regs", None),
                    n_spills=getattr(kernel, "n_spills", None),
                    shared=(getattr(kernel, "shared", None)
                            or getattr(metadata, "shared", None))))
    except Exception:                                             # noqa: BLE001
        entries = []
    return entries


def _effective_tile(triton_cone, cap, extent):
    """``_tile_size(cap, extent)`` across the signature it gained in K3, where a
    per-kernel minimum tile became a third parameter.  Reported, not assumed:
    the wrapper shrinks a requested cap to next_pow2(extent) on a small axis,
    and a row that recorded the REQUEST while the kernel launched something
    else would mislabel the whole configuration."""
    try:
        value = int(triton_cone._tile_size(cap, extent,
                                           triton_cone.CONE_BACK_MIN_TILE))
    except TypeError:
        value = int(triton_cone._tile_size(cap, extent))
    return value


def _timed_calls(fn, sync, trials):
    """``trials`` synchronized timings in ms.  The output is dropped inside the
    loop so the peak reflects ONE live output, not two."""
    times = []
    for _ in range(trials):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    return times


def torch_worker(cfg):
    """One torch row: the baseline (both torch-body forms) or one kernel
    configuration, at one cell's view batch."""
    import numpy as np
    import torch

    import mbirtorch
    import mbirtorch.triton_cone as triton_cone
    from mbirtorch.cone_beam import _cone_back_view_batch
    from mbirtorch.kernel_availability import triton_available
    from mbirtorch.projectors import maybe_compile

    cell = tuple(cfg["cell"])
    num_views, num_det_rows, num_channels = cell
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    # compile_mode='off' so the MODEL compiles nothing behind the bench: the
    # compiled torch baseline below is compiled explicitly, by the driver's own
    # maybe_compile, and is the only compiled thing in the row.
    model = mbirtorch.ConeBeamModel(cell, angles,
                                    source_detector_dist=4.0 * num_channels,
                                    source_iso_dist=2.0 * num_channels,
                                    device=DEVICE, compile_mode="off")
    model.set_params(no_warning=True, verbose=0)
    device = model.torch_device

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elif DEVICE == "mps":
            torch.mps.synchronize()

    def reset_peak():
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_bytes():
        if DEVICE == "cuda":
            value = int(torch.cuda.max_memory_allocated())
        else:
            value = 0
        return value

    recon_shape = tuple(model.get_params("recon_shape"))
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    num_pixels = int(pixel_indices.shape[0])
    band_len = int(recon_shape[2])
    # The driver's own view-batch rule, not a guess: the transient budget is
    # what production would hand this body at this cell and pixel set.
    driver_views = model.projector_functions._effective_view_batch(num_pixels,
                                                                   band_len)
    view_batch = int(os.environ.get("P5K2_VIEWS", "0")) or driver_views
    view_batch = max(1, min(view_batch, num_views))

    generator = torch.Generator().manual_seed(SINO_SEED)
    sino_batch = torch.rand((view_batch, num_det_rows, num_channels),
                            generator=generator).to(device)
    view_params = torch.as_tensor(model.get_params("view_params_array"),
                                  dtype=torch.float32,
                                  device=device)[:view_batch]
    args = model._view_batch_args()

    def torch_body():
        return _cone_back_view_batch(sino_batch, pixel_indices, view_params,
                                     coeff_power=COEFF_POWER, **args)

    def kernel_body():
        return triton_cone._cone_back_view_batch_triton(
            sino_batch, pixel_indices, view_params, coeff_power=COEFF_POWER,
            **args)

    result = dict(mode=cfg["mode"], cell=list(cell), recon_shape=list(recon_shape),
                  num_pixels=num_pixels, band_len=band_len,
                  view_batch=view_batch, driver_view_batch=int(driver_views),
                  psf_radius=int(args["psf_radius"]), coeff_power=COEFF_POWER,
                  version=f"torch {torch.__version__}",
                  triton_available=list(triton_available()),
                  device=DEVICE,
                  device_name=(torch.cuda.get_device_name(0)
                               if DEVICE == "cuda" else DEVICE))

    if cfg["mode"] == "baseline":
        reference = torch_body()
        sync()
        result["checksum"] = float(reference.abs().sum().item())
        del reference
        # Peaks are reset AFTER the warmup everywhere in this script: a first
        # call pays compilation (inductor autotuning allocates real buffers),
        # and the number wanted here is the steady-state peak of the call.
        _timed_calls(torch_body, sync, WARMUP)
        reset_peak()
        result["eager_times_ms"] = _timed_calls(torch_body, sync, TRIALS)
        result["eager_peak_bytes"] = peak_bytes()
        # The production torch body is COMPILED (the driver compiles bodies
        # unless compile_mode='off'), so the honest baseline for a speedup
        # ratio is the compiled one; the eager number stays for reference.
        try:
            compiled_body_fn = maybe_compile(_cone_back_view_batch, True,
                                             instance_key=0)

            def compiled_body():
                return compiled_body_fn(sino_batch, pixel_indices, view_params,
                                        coeff_power=COEFF_POWER, **args)

            _timed_calls(compiled_body, sync, WARMUP)
            reset_peak()
            result["compiled_times_ms"] = _timed_calls(compiled_body, sync,
                                                       TRIALS)
            result["compiled_peak_bytes"] = peak_bytes()
        except Exception as e:                                    # noqa: BLE001
            result["compiled_error"] = f"{type(e).__name__}: {e}"[:300]
        from mbirtorch.projectors import _COMPILE_ERRORS
        result["compile_fallbacks"] = dict(_COMPILE_ERRORS)
    else:
        cleared, clear_note = _apply_constants(
            triton_cone, cfg["block_p"], cfg["block_l"], cfg["num_warps"],
            cfg["num_stages"])
        result.update(cfg_block_p=cfg["block_p"], cfg_block_l=cfg["block_l"],
                      cfg_num_warps=cfg["num_warps"],
                      cfg_num_stages=cfg["num_stages"],
                      cache_cleared=cleared, cache_note=clear_note)
        # The tile the wrapper will actually launch: _tile_size shrinks the
        # requested cap to next_pow2(extent) on a small axis, so the effective
        # tile is recorded rather than assumed equal to the request.
        # K3 gave _tile_size a third parameter (the per-kernel minimum tile),
        # so the call goes through the arity probe rather than being pinned to
        # the signature this script was first written against.
        result["effective_block_p"] = _effective_tile(
            triton_cone, cfg["block_p"], num_pixels)
        result["effective_block_l"] = _effective_tile(
            triton_cone, cfg["block_l"], band_len)

        reference = torch_body()
        kernel_out = kernel_body()
        sync()
        scale = max(float(reference.abs().max()), 1e-30)
        result["value_rel"] = float((kernel_out - reference).abs().max()) / scale
        result["value_pass"] = bool(result["value_rel"] <= VALUE_REL_TOL)
        result["checksum"] = float(kernel_out.abs().sum().item())
        del reference, kernel_out

        _timed_calls(kernel_body, sync, WARMUP)
        reset_peak()
        result["kernel_times_ms"] = _timed_calls(kernel_body, sync, TRIALS)
        result["kernel_peak_bytes"] = peak_bytes()
        result["compiled_metadata"] = _compiled_metadata(
            triton_cone._cone_back_kernel)
    return result


def jax_worker(cfg):
    """The optional scale reference: mbirjax's pallas cone back projection at
    the same cell, FULL sinogram (its driver batches internally), warm.  Not
    comparable call-for-call with a single torch view batch -- the summary
    derives an extrapolated full-back number for that, clearly labeled."""
    import numpy as np

    import jax
    import mbirjax

    cell = tuple(cfg["cell"])
    num_views, _, num_channels = cell
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    model = mbirjax.ConeBeamModel(cell, angles,
                                  source_detector_dist=4.0 * num_channels,
                                  source_iso_dist=2.0 * num_channels)
    model.set_params(no_warning=True, verbose=0)
    rng = np.random.default_rng(SINO_SEED)
    sinogram = jax.device_put(rng.random(cell, dtype=np.float32))

    def back():
        return model.back_project(sinogram, output_sharded=True)

    times = []
    for _ in range(WARMUP + TRIALS):
        t0 = time.perf_counter()
        out = back()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    # memory_stats() is None on the CPU backend (the local smoke path).
    stats = jax.devices()[0].memory_stats() or {}
    return dict(mode="jax", cell=list(cell), version=f"jax {jax.__version__}",
                back_times_ms=times[WARMUP:],
                gpu_peak_bytes=int(stats.get("peak_bytes_in_use", 0)),
                recon_shape=[int(x) for x in model.get_params("recon_shape")])


def run_one(python, cfg):
    """One row in its own process (the p3/p4 ruler-per-row protocol)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k2.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k2.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path])
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(error=f"worker exited {proc.returncode}", **cfg)
    else:
        with open(out_path) as f:
            row = json.load(f)
    return row


def _median_ms(row, key):
    times = row.get(key)
    if times:
        value = statistics.median(times)
    else:
        value = None
    return value


def print_cell_table(cell, rows):
    baseline = next((r for r in rows if r.get("mode") == "baseline"
                     and "error" not in r), None)
    if baseline is not None:
        base_ms = (_median_ms(baseline, "compiled_times_ms")
                   or _median_ms(baseline, "eager_times_ms"))
    else:
        base_ms = None
    shaped = next((r for r in rows if r.get("view_batch")), None)
    if shaped is None:
        shape_text = ""
    else:
        # The view batch is the driver's, so a cell where the budget forces 1
        # view says so here: the kernel's view reduction runs once per call
        # there, and its amortization across views is not what this row saw.
        shape_text = (f"{shaped['num_pixels']} pixels x {shaped['band_len']} "
                      f"slices, view batch {shaped['view_batch']} of {cell[0]}; ")
    print(f"\n=== cell {'x'.join(map(str, cell))} "
          f"({shape_text}baseline torch body "
          f"{base_ms if base_ms is None else round(base_ms, 3)}"
          f" ms/view-batch) ===", flush=True)
    print(f"{'BP':>4}{'BL':>5}{'W':>3}{'ST':>3}{'ms':>10}{'x_body':>8}"
          f"{'n_regs':>8}{'spill':>6}{'peakGB':>8}{'rel':>10}  value", flush=True)
    kernel_rows = [r for r in rows if r.get("mode") == "kernel"]
    for row in sorted(kernel_rows,
                      key=lambda r: (_median_ms(r, "kernel_times_ms")
                                     or float("inf"))):
        if "error" in row:
            print(f"{row.get('block_p', '?'):>4}{row.get('block_l', '?'):>5}"
                  f"{row.get('num_warps', '?'):>3}{row.get('num_stages', '?'):>3}"
                  f"   FAILED: {row['error'][:60]}", flush=True)
        else:
            ms = _median_ms(row, "kernel_times_ms")
            speedup = "-" if base_ms is None else f"{base_ms / ms:.2f}"
            meta = (row.get("compiled_metadata") or [{}])[0]
            n_regs = meta.get("n_regs")
            n_spills = meta.get("n_spills")
            verdict = "PASS" if row.get("value_pass") else "FLAG"
            print(f"{row['cfg_block_p']:>4}{row['cfg_block_l']:>5}"
                  f"{row['cfg_num_warps']:>3}{row['cfg_num_stages']:>3}"
                  f"{ms:>10.3f}{speedup:>8}"
                  f"{('-' if n_regs is None else n_regs):>8}"
                  f"{('-' if n_spills is None else n_spills):>6}"
                  f"{row['kernel_peak_bytes'] / 2**30:>8.2f}"
                  f"{row['value_rel']:>10.1e}  {verdict}", flush=True)


def summarize_cell(cell, rows):
    """The per-cell readout the reviewer reads first: the winner, its margin
    over the torch body, and the (indicative) extrapolation against jax."""
    kernel_rows = [r for r in rows if r.get("mode") == "kernel"
                   and "error" not in r]
    baseline = next((r for r in rows if r.get("mode") == "baseline"
                     and "error" not in r), None)
    jax_row = next((r for r in rows if r.get("mode") == "jax"
                    and "error" not in r), None)
    summary = dict(cell=list(cell), rows_ok=len(kernel_rows),
                   rows_failed=len([r for r in rows if "error" in r]))
    if kernel_rows:
        best = min(kernel_rows, key=lambda r: _median_ms(r, "kernel_times_ms"))
        best_ms = _median_ms(best, "kernel_times_ms")
        summary.update(best_config=[best["cfg_block_p"], best["cfg_block_l"],
                                    best["cfg_num_warps"],
                                    best["cfg_num_stages"]],
                       best_ms=best_ms,
                       best_peak_bytes=best["kernel_peak_bytes"],
                       view_batch=best["view_batch"],
                       value_flags=[[r["cfg_block_p"], r["cfg_block_l"],
                                     r["cfg_num_warps"], r["cfg_num_stages"],
                                     r["value_rel"]]
                                    for r in kernel_rows
                                    if not r.get("value_pass")])
        if baseline is not None:
            eager_ms = _median_ms(baseline, "eager_times_ms")
            compiled_ms = _median_ms(baseline, "compiled_times_ms")
            summary.update(torch_eager_ms=eager_ms,
                           torch_compiled_ms=compiled_ms,
                           torch_eager_peak_bytes=baseline.get("eager_peak_bytes"),
                           torch_compiled_peak_bytes=baseline.get(
                               "compiled_peak_bytes"))
            base_ms = compiled_ms or eager_ms
            summary["kernel_speedup_vs_torch_body"] = base_ms / best_ms
        if jax_row is not None:
            jax_ms = statistics.median(jax_row["back_times_ms"])
            batches = math.ceil(cell[0] / best["view_batch"])
            extrapolated = best_ms * batches
            summary.update(
                jax_full_back_ms=jax_ms,
                jax_gpu_peak_bytes=jax_row.get("gpu_peak_bytes"),
                torch_full_back_ms_extrapolated=extrapolated,
                torch_vs_jax_full_back_extrapolated=extrapolated / jax_ms)
    return summary


def main():
    cells = selected_cells()
    kept, dropped = build_plan()
    skip_jax = (os.environ.get("P5K2_SKIP_JAX", "0") == "1"
                or not os.path.exists(JAX_PYTHON))
    print(f"p5k2 sweep on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, "
          f"{len(kept)} configs/cell, jax reference "
          f"{'SKIPPED' if skip_jax else 'ON'}", flush=True)
    print("isolated body bench -- it ranks constants; p5k2_gate.py is the gate",
          flush=True)
    print_plan(kept, dropped)

    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="cone", coeff_power=COEFF_POWER,
                       grid=dict(block_p=list(BLOCK_P_VALUES),
                                 block_l=list(BLOCK_L_VALUES),
                                 num_warps=list(NUM_WARPS_VALUES),
                                 num_stages=list(NUM_STAGES_VALUES)),
                       pruning=dict(live_tiles=LIVE_TILES,
                                    max_regs_per_thread=MAX_REGS_PER_THREAD,
                                    min_elems_per_thread=MIN_ELEMS_PER_THREAD,
                                    max_regs_pipelined=MAX_REGS_PER_THREAD_PIPELINED,
                                    max_configs=MAX_CONFIGS_PER_CELL),
                       plan=kept, dropped=dropped, rows=[], summaries=[])

    for cell in cells:
        cell_rows = []
        label = "x".join(map(str, cell))
        print(f"\n{label}/baseline ...", flush=True)
        cell_rows.append(run_one(TORCH_PYTHON, dict(mode="baseline",
                                                    cell=list(cell))))
        if not skip_jax:
            print(f"{label}/jax-back ...", flush=True)
            cell_rows.append(run_one(JAX_PYTHON, dict(mode="jax",
                                                      cell=list(cell))))
        for point in kept:
            cfg = dict(mode="kernel", cell=list(cell), **_config_dict(point))
            print(f"{label}/kernel {_config_tuple(point)} ...", flush=True)
            row = run_one(TORCH_PYTHON, cfg)
            cell_rows.append(row)
            if "error" in row:
                print(f"  FAILED: {row['error'][:200]}", flush=True)
            else:
                print(f"  median {statistics.median(row['kernel_times_ms']):.3f} ms"
                      f"  rel {row['value_rel']:.1e}"
                      f"  peak {row['kernel_peak_bytes'] / 2**30:.2f}G",
                      flush=True)
        print_cell_table(cell, cell_rows)
        summary = summarize_cell(cell, cell_rows)
        all_results["rows"].extend(cell_rows)
        all_results["summaries"].append(summary)
        print(f"summary {label}: {json.dumps(summary)}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5k2_sweep_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


def _config_dict(point):
    return dict(block_p=point["block_p"], block_l=point["block_l"],
                num_warps=point["num_warps"], num_stages=point["num_stages"])


def _run_worker(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        if cfg["mode"] == "jax":
            result = jax_worker(cfg)
        else:
            result = torch_worker(cfg)
    except Exception as e:                                        # noqa: BLE001
        traceback.print_exc()
        result = dict(error=f"{type(e).__name__}: {e}"[:400], **cfg)
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _run_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--dry-run":
        print_plan(*build_plan())
    elif len(sys.argv) >= 2:
        print(f"unknown argument {sys.argv[1]!r}; try --help")
        sys.exit(2)
    else:
        main()
