"""The CONSTANT SWEEP for the Triton PARALLEL pair -- the isolated
per-view-batch bench that ranks (BLOCK_P, BLOCK_R, num_warps, num_stages) for
BOTH directions at the two gate cells, before the composed gate (p5k6_pgate.py)
decides anything.

The cone twins are p5k2_sweep.py (back) and p5k4_fwd_sweep.py (forward); this
is their parallel-beam counterpart, one file for both directions because the
two parallel kernels share a grid shape, a contract, and a tile axis, so a
`direction` column is cheaper than a third near-copy.

What is different here, and what the sweep is for:

  - The shipped constants are SEEDED FROM THE CONE WINNERS, not measured: the
    parallel back kernel is the cone back's inner tile with the row-tap loop
    removed, the parallel forward is the cone forward's scatter phase with the
    vertical fan removed.  Nothing has been measured at these shapes.
  - Register pressure is strictly LOWER (see the pruning block: 4 live tiles
    for back and 3 for forward, against the cone kernels' 7 and 12), so the
    grid pushes BLOCK_R to 256 -- the axis the implementation flagged as the
    one that should now stay clear of spilling.
  - Both directions are opt-in (MBIRTORCH_ENABLE_TRITON_PBACK / _PFWD), so
    every row sets the switch for the direction it measures and then ARM-CHECKS
    the model: `_view_batch_bodies()` must hand back the triton body.  A row
    still TIMES the wrapper directly, but a device whose availability
    self-check declines would otherwise report happy numbers for a body
    production would never bind.
  - The forward scatters with ATOMICS, so its output is not bit-reproducible
    between launches.  Every row of BOTH directions measures its own
    kernel-vs-kernel repeat: for forward it is the floor the value column must
    be read against, and for back it is the assertion that the value column has
    no floor at all (it should be exactly 0).

Protocol otherwise identical to the cone sweeps: ONE (cell, direction, config)
per subprocess, warmed then TRIALS timed calls, median reported, honest peak,
the per-(cell, direction) torch baseline in both eager and COMPILED forms
(compiled is the production body and the honest ratio), and ptxas metadata read
back from triton's own cache.

ISOLATION CAVEAT (the pallas E4 lesson, stated up front): an isolated body
bench flatters baselines and cannot settle the replacement rule.  It ranks
constants; p5k6_pgate.py measures the composed recon, and that is the gate.

Run:
    <torch python> p5k6_psweep.py       on a CUDA node (see p5k6_psweep_gautschi.sbatch)
    python p5k6_psweep.py --dry-run     anywhere: print the pruned config plans
    python p5k6_psweep.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas):
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the row subprocesses
    P5K6_SKIP_JAX=1                   skip the optional mbirjax reference rows
    P5K6_CELLS=512,1024               run a subset of the cells (by view count)
    P5K6_DIRECTIONS=back,fwd          run a subset of the directions
    P5K6_MAX_CONFIGS=30               cap on kept configurations per (cell, dir)
    P5K6_VIEWS=<n>                    override the driver's view batch size
    P5K6_SMOKE=1                      tiny cell on P5K6_DEVICE (default cpu):
                                      exercises every step but the launch
    P5K6_DEVICE=cuda|cpu|mps          the torch device for the rows
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

# The Phase 2/3/4 gate cells.  Parallel beam holds rows == slices, so the
# detector row count IS the recon slice count at every cell.
CELLS = [(512, 448, 384), (1024, 1008, 992)]
DIRECTIONS = ("back", "fwd")

SMOKE = os.environ.get("P5K6_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K6_DEVICE", "cpu" if SMOKE else "cuda")

# ── the two grids and their pruning models ────────────────────────────────────
# The register model, counted from the kernel sources in triton_parallel.py
# rather than inherited from the cone estimates:
#
#   BACK, at the channel-tap loop peak, these (BLOCK_P, BLOCK_R) tiles are live
#     tile_mask, acc (the accumulator), vals (the gather), and the product
#     w_chan[:, None] * vals                                        = 4 tiles
#   FORWARD, at the atomic-tap loop peak
#     tile_mask, vals (loaded ONCE and held across every tap -- the design's
#     point), and the atomic product                                = 3 tiles
#
# against the cone back's 7 and the cone forward's 12.  The per-pixel vectors
# (n_p, centers, n_tap, w_chan, n_chan) cost BLOCK_P elements each, a
# 1/BLOCK_R fraction of a tile, and are ignored.  Both loops also build an i64
# POINTER tile (2 registers per element) that this estimate does not model --
# which is precisely why every row reads n_regs/n_spills back from ptxas: the
# estimate RANKS configurations, the readback JUDGES them.
#
# The ranking anchor is an ABSOLUTE register estimate, not a shape: the shipped
# constants here are seeded from the cone winners rather than measured, so
# ranking toward them would defeat the sweep's purpose.  56 is the cone BACK
# winner's estimate on this hardware (7 tiles at 16x64x4 warps) and 96 the cone
# forward's pinned one -- the only measured points there are.  The anchor only
# decides which stages-2 rows survive the cap; every stages-1 survivor runs.
GRIDS = {
    "back": dict(block_p=(8, 16, 32), block_r=(64, 128, 256),
                 num_warps=(2, 4, 8), num_stages=(1, 2),
                 live_tiles=4, target_regs=56, pinned=(16, 64, 4, 1)),
    "fwd": dict(block_p=(8, 16, 32), block_r=(64, 128, 256),
                num_warps=(2, 4, 8), num_stages=(1, 2),
                live_tiles=3, target_regs=96, pinned=(8, 128, 8, 1)),
}

# Hard cap: 255 registers/thread is the hardware ceiling on H100 and A100, so
# an estimate past this spills by construction.  224 leaves room for the
# estimate being crude.
MAX_REGS_PER_THREAD = 224
# A tile so small that each thread owns < 4 accumulator elements is index
# arithmetic with a rounding error attached.
MIN_ELEMS_PER_THREAD = 4
# Software pipelining duplicates the in-flight tiles, so stages > 1 is swept
# only where half the budget is free.
MAX_REGS_PER_THREAD_PIPELINED = 112
MAX_CONFIGS_PER_CELL = int(os.environ.get("P5K6_MAX_CONFIGS", "30"))

WARMUP = 1          # pays the triton compile and the allocator's first pass
TRIALS = 3
INPUT_SEED = 0      # a private generator: the recon gates read the global RNG
COEFF_POWER = 1     # the gradient path -- where the VCD loop spends its time
VALUE_REL_TOL = 1e-5                     # the design's gradient-path gate
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]

# What each direction needs from mbirtorch, in one place: the module constants
# to override, the wrapper to time, the torch body to compare against, and the
# JITFunction whose cache is cleared and whose metadata is read back.
DIRECTION_SPEC = {
    "back": dict(constants=("PARALLEL_BACK_BLOCK_P", "PARALLEL_BACK_BLOCK_R",
                            "PARALLEL_BACK_NUM_WARPS",
                            "PARALLEL_BACK_NUM_STAGES"),
                 min_tile="PARALLEL_BACK_MIN_TILE",
                 kernel="_parallel_back_kernel",
                 wrapper="_parallel_back_view_batch_triton",
                 torch_body="_parallel_back_view_batch",
                 enable_env="MBIRTORCH_ENABLE_TRITON_PBACK"),
    "fwd": dict(constants=("PARALLEL_FWD_BLOCK_P", "PARALLEL_FWD_BLOCK_R",
                           "PARALLEL_FWD_NUM_WARPS",
                           "PARALLEL_FWD_NUM_STAGES"),
                min_tile="PARALLEL_FWD_MIN_TILE",
                kernel="_parallel_forward_kernel",
                wrapper="_parallel_forward_view_batch_triton",
                torch_body="_parallel_forward_view_batch",
                enable_env="MBIRTORCH_ENABLE_TRITON_PFWD"),
}
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    """The cells to run: the smoke cell, all of CELLS, or the subset named by
    P5K6_CELLS as a comma-separated list of view counts."""
    wanted = os.environ.get("P5K6_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def selected_directions():
    wanted = os.environ.get("P5K6_DIRECTIONS", "").strip()
    if wanted:
        keep = {d.strip() for d in wanted.split(",") if d.strip()}
        directions = [d for d in DIRECTIONS if d in keep]
    else:
        directions = list(DIRECTIONS)
    return directions


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


def _grid_point(direction, block_p, block_r, num_warps, num_stages):
    """One grid point with its cost estimates and its keep/drop verdict."""
    live_tiles = GRIDS[direction]["live_tiles"]
    threads = 32 * num_warps
    elems_per_thread = block_p * block_r / threads
    est_regs = live_tiles * elems_per_thread
    return dict(direction=direction, block_p=block_p, block_r=block_r,
                num_warps=num_warps, num_stages=num_stages,
                accumulator_bytes=4 * block_p * block_r,
                elems_per_thread=elems_per_thread,
                est_regs_per_thread=est_regs,
                drop_reason=_drop_reason(elems_per_thread, est_regs,
                                         num_stages))


def _rank_key(point):
    """Ranking for the cap: stages 1 before stages 2 (both loops are gather- or
    atomic-bound, so pipelining is the speculative axis), then closeness to the
    anchor register estimate, then a deterministic tie-break."""
    target = GRIDS[point["direction"]]["target_regs"]
    return (point["num_stages"] - 1,
            abs(math.log2(point["est_regs_per_thread"] / target)),
            point["block_p"], point["block_r"], point["num_warps"])


def _config_tuple(point):
    return (point["block_p"], point["block_r"], point["num_warps"],
            point["num_stages"])


def _config_dict(point):
    return dict(block_p=point["block_p"], block_r=point["block_r"],
                num_warps=point["num_warps"], num_stages=point["num_stages"])


def build_plan(direction):
    """(kept, dropped): the pruned, capped, ranked plan for one direction.  The
    grid does not depend on the cell, so one plan serves every cell."""
    grid = GRIDS[direction]
    points = [_grid_point(direction, bp, br, w, st)
              for bp in grid["block_p"]
              for br in grid["block_r"]
              for w in grid["num_warps"]
              for st in grid["num_stages"]]
    dropped = [p for p in points if p["drop_reason"] is not None]
    survivors = sorted((p for p in points if p["drop_reason"] is None),
                       key=_rank_key)
    kept = survivors[:MAX_CONFIGS_PER_CELL]
    pinned = [p for p in survivors if _config_tuple(p) == grid["pinned"]]
    if pinned and pinned[0] not in kept:
        kept = kept[:-1] + pinned          # the shipped constants always run
    for point in survivors[len(kept):]:
        point["drop_reason"] = f"beyond the {MAX_CONFIGS_PER_CELL}-config cap"
        dropped.append(point)
    return kept, dropped


def print_plan(direction, kept, dropped):
    grid = GRIDS[direction]
    total = (len(grid["block_p"]) * len(grid["block_r"])
             * len(grid["num_warps"]) * len(grid["num_stages"]))
    print(f"\n{direction} plan: {len(kept)} kept of {total} grid points "
          f"(cap {MAX_CONFIGS_PER_CELL}), {len(dropped)} dropped; "
          f"{grid['live_tiles']} live tiles, anchor {grid['target_regs']} "
          f"regs/thread, shipped {grid['pinned']}")
    print(f"{'BP':>4}{'BR':>5}{'W':>3}{'ST':>3}{'tile_KB':>9}"
          f"{'elem/thr':>10}{'est_reg':>9}")
    for point in kept:
        print(f"{point['block_p']:>4}{point['block_r']:>5}"
              f"{point['num_warps']:>3}{point['num_stages']:>3}"
              f"{point['accumulator_bytes'] / 1024:>9.1f}"
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


def _apply_constants(triton_parallel, direction, block_p, block_r, num_warps,
                     num_stages):
    """Set one direction's four tuning constants for the calls that follow, and
    clear both caches that sit behind a launch.

    The wrappers read all four as MODULE GLOBALS at call time --
    ``_tile_size(PARALLEL_*_BLOCK_P, num_pixels, PARALLEL_*_MIN_TILE)`` and the
    ``num_warps=``/``num_stages=`` launch kwargs -- so a plain setattr takes
    effect on the next launch; nothing is baked at import.  Two caches sit
    behind that launch:

      - triton's own per-JITFunction compiled-kernel cache.  Current triton
        keys it on the constexprs AND the launch options, so a new (warps,
        stages) pair already compiles a new variant -- but a stale hit would
        silently measure the PREVIOUS configuration, which is the one failure
        mode a tuning sweep must not have, so it is cleared rather than
        trusted.  Clearing it here also drops the variant the availability
        self-check compiled at the tiny probe shape, which is why the arm check
        runs BEFORE this call: the metadata readback afterwards then describes
        the swept launch alone.
      - ``_COMPILED_LAUNCH_KEYS``, shared by all four cone and parallel
        kernels, which decides only whether a launch takes the process-wide
        compile lock.  Its keys include the block sizes but NOT warps/stages,
        so a swept configuration would otherwise skip the lock that its first
        production launch takes.

    Returns:
        (bool, str): whether a triton cache was found and cleared, plus a note.
    """
    names = DIRECTION_SPEC[direction]["constants"]
    for name, value in zip(names, (block_p, block_r, num_warps, num_stages)):
        setattr(triton_parallel, name, int(value))
    triton_parallel._COMPILED_LAUNCH_KEYS.clear()
    jit_fn = getattr(triton_parallel, DIRECTION_SPEC[direction]["kernel"])
    caches = _kernel_cache_dicts(jit_fn)
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

    The authoritative introspection the pruning model is only guessing at --
    ``n_spills > 0`` is the real spill verdict (and the i64 pointer tiles are
    exactly what the estimate does not model), while a ``num_warps`` here that
    disagrees with the request means the constants never reached the compiler.
    Best effort: an unrecognized triton version yields [].
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
    configuration, for one direction at one cell's view batch."""
    import numpy as np
    import torch

    import mbirtorch
    import mbirtorch.parallel_beam as parallel_beam
    import mbirtorch.triton_parallel as triton_parallel
    from mbirtorch.kernel_availability import (parallel_back_kernel_usable,
                                               parallel_forward_kernel_usable,
                                               triton_available)
    from mbirtorch.projectors import maybe_compile

    direction = cfg["direction"]
    spec = DIRECTION_SPEC[direction]
    cell = tuple(cfg["cell"])
    num_views, num_det_rows, num_channels = cell
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    # compile_mode='off' so the MODEL compiles nothing behind the bench: the
    # compiled torch baseline below is compiled explicitly, by the driver's own
    # maybe_compile, and is the only compiled thing in the row.
    model = mbirtorch.ParallelBeamModel(cell, angles, 
                                        compile_mode="off")
    model.configure_devices(devices=[DEVICE])
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

    # ARM CHECK, before any constant is applied (see _apply_constants): with
    # this direction's opt-in set in the process, would production actually
    # bind the triton body here?  The row times the wrapper directly either
    # way, but a device whose self-check declines makes those numbers a
    # measurement of a body nobody would run.
    fwd_hook, back_hook = model._view_batch_bodies()
    hook_names = dict(fwd=getattr(fwd_hook, "__name__", str(fwd_hook)),
                      back=getattr(back_hook, "__name__", str(back_hook)))

    recon_shape = tuple(model.get_params("recon_shape"))
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    num_pixels = int(pixel_indices.shape[0])
    # rows == slices under parallel beam, so one number is both the detector
    # row count of a sinogram block and the column count of a voxel cylinder.
    band_rows = int(model.get_params("sinogram_shape")[1])
    # The driver's own view-batch rule, not a guess: the transient budget is
    # what production would hand this body at this cell and pixel set.
    driver_views = model.projector_functions._effective_view_batch(num_pixels,
                                                                   band_rows)
    view_batch = int(os.environ.get("P5K6_VIEWS", "0")) or driver_views
    view_batch = max(1, min(view_batch, num_views))

    generator = torch.Generator().manual_seed(INPUT_SEED)
    # Through the driver's own indirection: a parallel model's view parameters
    # are its ANGLES, a cone model's are a (angle, z_shift) array, and the
    # geometry names which ('view_params_name') -- hardcoding the cone name
    # here is a NameError, which is how the local smoke found this line.
    view_params_name = model.get_params("view_params_name")
    view_params = torch.as_tensor(model.get_params(view_params_name),
                                  dtype=torch.float32,
                                  device=device)[:view_batch]
    args = model._view_batch_args()
    torch_body_fn = getattr(parallel_beam, spec["torch_body"])
    kernel_fn = getattr(triton_parallel, spec["wrapper"])

    if direction == "back":
        sino_batch = torch.rand((view_batch, band_rows, num_channels),
                                generator=generator).to(device)
        call_args = (sino_batch, pixel_indices, view_params)
        call_kwargs = dict(coeff_power=COEFF_POWER, **args)
        tile_extent = band_rows
    else:
        values = torch.rand((num_pixels, band_rows),
                            generator=generator).to(device)
        call_args = (values, pixel_indices, view_params)
        call_kwargs = dict(args)
        tile_extent = band_rows

    def torch_body():
        return torch_body_fn(*call_args, **call_kwargs)

    def kernel_body():
        return kernel_fn(*call_args, **call_kwargs)

    result = dict(mode=cfg["mode"], direction=direction, cell=list(cell),
                  recon_shape=list(recon_shape), num_pixels=num_pixels,
                  band_rows=band_rows, view_batch=view_batch,
                  driver_view_batch=int(driver_views),
                  psf_radius=int(args["psf_radius"]),
                  coeff_power=(COEFF_POWER if direction == "back" else None),
                  version=f"torch {torch.__version__}",
                  triton_available=list(triton_available()),
                  fwd_body=hook_names["fwd"], back_body=hook_names["back"],
                  kernel_bound=("triton" in hook_names[direction]),
                  enable_env=os.environ.get(spec["enable_env"], ""),
                  device=DEVICE,
                  device_name=(torch.cuda.get_device_name(0)
                               if DEVICE == "cuda" else DEVICE))
    if direction == "back":
        result["usable"] = list(parallel_back_kernel_usable(model))
    else:
        result["usable"] = list(parallel_forward_kernel_usable(model))

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
            compiled_body_fn = maybe_compile(torch_body_fn, True,
                                             instance_key=0)

            def compiled_body():
                return compiled_body_fn(*call_args, **call_kwargs)

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
            triton_parallel, direction, cfg["block_p"], cfg["block_r"],
            cfg["num_warps"], cfg["num_stages"])
        result.update(cfg_block_p=cfg["block_p"], cfg_block_r=cfg["block_r"],
                      cfg_num_warps=cfg["num_warps"],
                      cfg_num_stages=cfg["num_stages"],
                      cache_cleared=cleared, cache_note=clear_note)
        # The tile the wrapper will actually launch: _tile_size shrinks the
        # requested cap to next_pow2(extent) on a small axis, so the effective
        # tile is recorded rather than assumed equal to the request.
        min_tile = getattr(triton_parallel, spec["min_tile"])
        result["effective_block_p"] = int(
            triton_parallel._tile_size(cfg["block_p"], num_pixels, min_tile))
        result["effective_block_r"] = int(
            triton_parallel._tile_size(cfg["block_r"], tile_extent, min_tile))

        reference = torch_body()
        kernel_out = kernel_body()
        repeat_out = kernel_body()          # the same kernel, launched again
        sync()
        scale = max(float(reference.abs().max()), 1e-30)
        result["value_rel"] = float((kernel_out - reference).abs().max()) / scale
        result["value_pass"] = bool(result["value_rel"] <= VALUE_REL_TOL)
        # The forward scatters with atomics, so two identical launches need not
        # agree bit for bit: this is the floor its value column is read
        # against.  The back kernel has no atomics, so its floor should be
        # exactly 0 -- recorded as an assertion, not as a caveat.
        result["value_rel_selfrepeat"] = (
            float((repeat_out - kernel_out).abs().max()) / scale)
        result["checksum"] = float(kernel_out.abs().sum().item())
        del reference, kernel_out, repeat_out

        _timed_calls(kernel_body, sync, WARMUP)
        reset_peak()
        result["kernel_times_ms"] = _timed_calls(kernel_body, sync, TRIALS)
        result["kernel_peak_bytes"] = peak_bytes()
        result["compiled_metadata"] = _compiled_metadata(
            getattr(triton_parallel, spec["kernel"]))
    return result


def jax_worker(cfg):
    """The optional scale reference: mbirjax's pallas parallel projection in
    the same direction at the same cell, full array (its driver batches views
    internally), warm.  Not comparable call-for-call with a single torch view
    batch -- the summary derives an extrapolated full-projection number for
    that, clearly labeled."""
    import numpy as np

    import jax
    import mbirjax

    direction = cfg["direction"]
    cell = tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(cell, angles)
    model.set_params(no_warning=True, verbose=0)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    rng = np.random.default_rng(INPUT_SEED)
    if direction == "back":
        payload = jax.device_put(rng.random(cell, dtype=np.float32))

        def project():
            return model.back_project(payload, output_sharded=True)
    else:
        payload = jax.device_put(rng.random(recon_shape, dtype=np.float32))

        def project():
            return model.forward_project(payload, output_sharded=True)

    times = []
    for _ in range(WARMUP + TRIALS):
        t0 = time.perf_counter()
        out = project()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    # memory_stats() is None on the CPU backend (the local smoke path).
    stats = jax.devices()[0].memory_stats() or {}
    return dict(mode="jax", direction=direction, cell=list(cell),
                version=f"jax {jax.__version__}",
                project_times_ms=times[WARMUP:],
                gpu_peak_bytes=int(stats.get("peak_bytes_in_use", 0)),
                recon_shape=list(recon_shape))


def run_one(python, cfg):
    """One row in its own process (the p3/p4 ruler-per-row protocol).  A torch
    row carries the opt-in switch for the direction it measures -- and only
    that one, so the other direction's availability self-check never runs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k6.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k6.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    if cfg["mode"] == "jax":
        env = dict(os.environ)
    else:
        env = dict(os.environ, MBIRTORCH_DISABLE_TRITON="0")
        env[DIRECTION_SPEC[cfg["direction"]]["enable_env"]] = "1"
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=env)
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


def print_cell_table(cell, direction, rows):
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
        shape_text = (f"{shaped['num_pixels']} pixels x {shaped['band_rows']} "
                      f"rows, view batch {shaped['view_batch']} of {cell[0]}; ")
    print(f"\n=== cell {'x'.join(map(str, cell))} {direction} "
          f"({shape_text}baseline compiled torch "
          f"{base_ms if base_ms is None else round(base_ms, 3)}"
          f" ms/view-batch) ===", flush=True)
    print(f"{'BP':>4}{'BR':>5}{'W':>3}{'ST':>3}{'ms':>10}{'x_body':>8}"
          f"{'n_regs':>8}{'spill':>6}{'peakGB':>8}{'rel':>10}{'repeat':>10}"
          f"  value", flush=True)
    kernel_rows = [r for r in rows if r.get("mode") == "kernel"]
    for row in sorted(kernel_rows,
                      key=lambda r: (_median_ms(r, "kernel_times_ms")
                                     or float("inf"))):
        if "error" in row:
            print(f"{row.get('block_p', '?'):>4}{row.get('block_r', '?'):>5}"
                  f"{row.get('num_warps', '?'):>3}{row.get('num_stages', '?'):>3}"
                  f"   FAILED: {row['error'][:60]}", flush=True)
        else:
            ms = _median_ms(row, "kernel_times_ms")
            speedup = "-" if base_ms is None else f"{base_ms / ms:.2f}"
            meta = (row.get("compiled_metadata") or [{}])[0]
            n_regs = meta.get("n_regs")
            n_spills = meta.get("n_spills")
            verdict = "PASS" if row.get("value_pass") else "FLAG"
            print(f"{row['cfg_block_p']:>4}{row['cfg_block_r']:>5}"
                  f"{row['cfg_num_warps']:>3}{row['cfg_num_stages']:>3}"
                  f"{ms:>10.3f}{speedup:>8}"
                  f"{('-' if n_regs is None else n_regs):>8}"
                  f"{('-' if n_spills is None else n_spills):>6}"
                  f"{row['kernel_peak_bytes'] / 2**30:>8.2f}"
                  f"{row['value_rel']:>10.1e}"
                  f"{row['value_rel_selfrepeat']:>10.1e}  {verdict}",
                  flush=True)


def summarize(cell, direction, rows):
    """The per-(cell, direction) readout: the winner, its margin over the
    compiled torch body, and the (indicative) extrapolation against jax."""
    kernel_rows = [r for r in rows if r.get("mode") == "kernel"
                   and "error" not in r]
    baseline = next((r for r in rows if r.get("mode") == "baseline"
                     and "error" not in r), None)
    jax_row = next((r for r in rows if r.get("mode") == "jax"
                    and "error" not in r), None)
    summary = dict(cell=list(cell), direction=direction,
                   rows_ok=len(kernel_rows),
                   rows_failed=len([r for r in rows if "error" in r]))
    arm = next((r for r in rows if "kernel_bound" in r), None)
    if arm is not None:
        summary["kernel_bound"] = arm["kernel_bound"]
        summary["usable"] = arm["usable"]
    if kernel_rows:
        best = min(kernel_rows, key=lambda r: _median_ms(r, "kernel_times_ms"))
        best_ms = _median_ms(best, "kernel_times_ms")
        summary.update(best_config=[best["cfg_block_p"], best["cfg_block_r"],
                                    best["cfg_num_warps"],
                                    best["cfg_num_stages"]],
                       best_ms=best_ms,
                       best_peak_bytes=best["kernel_peak_bytes"],
                       best_n_regs=(best.get("compiled_metadata")
                                    or [{}])[0].get("n_regs"),
                       best_n_spills=(best.get("compiled_metadata")
                                      or [{}])[0].get("n_spills"),
                       view_batch=best["view_batch"],
                       worst_repeat_floor=max(r["value_rel_selfrepeat"]
                                              for r in kernel_rows),
                       value_flags=[[r["cfg_block_p"], r["cfg_block_r"],
                                     r["cfg_num_warps"], r["cfg_num_stages"],
                                     r["value_rel"], r["value_rel_selfrepeat"]]
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
            jax_ms = statistics.median(jax_row["project_times_ms"])
            batches = math.ceil(cell[0] / best["view_batch"])
            extrapolated = best_ms * batches
            summary.update(
                jax_full_ms=jax_ms,
                jax_gpu_peak_bytes=jax_row.get("gpu_peak_bytes"),
                torch_full_ms_extrapolated=extrapolated,
                torch_vs_jax_full_extrapolated=extrapolated / jax_ms)
    return summary


def main():
    cells = selected_cells()
    directions = selected_directions()
    plans = {d: build_plan(d) for d in directions}
    skip_jax = (os.environ.get("P5K6_SKIP_JAX", "0") == "1"
                or not os.path.exists(JAX_PYTHON))
    print(f"p5k6 parallel sweep on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, directions "
          f"{directions}, jax reference {'SKIPPED' if skip_jax else 'ON'}",
          flush=True)
    print("isolated body bench -- it ranks constants; p5k6_pgate.py is the gate",
          flush=True)
    for direction in directions:
        print_plan(direction, *plans[direction])

    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel",
                       grids={d: {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in GRIDS[d].items()}
                              for d in directions},
                       pruning=dict(max_regs_per_thread=MAX_REGS_PER_THREAD,
                                    min_elems_per_thread=MIN_ELEMS_PER_THREAD,
                                    max_regs_pipelined=MAX_REGS_PER_THREAD_PIPELINED,
                                    max_configs=MAX_CONFIGS_PER_CELL),
                       plans={d: plans[d][0] for d in directions},
                       dropped={d: plans[d][1] for d in directions},
                       rows=[], summaries=[])

    for cell in cells:
        for direction in directions:
            rows = []
            label = f"{'x'.join(map(str, cell))}/{direction}"
            print(f"\n{label}/baseline ...", flush=True)
            row = run_one(TORCH_PYTHON, dict(mode="baseline", cell=list(cell),
                                             direction=direction))
            rows.append(row)
            if "error" not in row and not row.get("kernel_bound"):
                print(f"  ARM CHECK FAILED: production would bind "
                      f"{row.get(direction + '_body', '?')} here -- "
                      f"{row.get('usable')}", flush=True)
            if not skip_jax:
                print(f"{label}/jax ...", flush=True)
                rows.append(run_one(JAX_PYTHON, dict(mode="jax",
                                                     cell=list(cell),
                                                     direction=direction)))
            for point in plans[direction][0]:
                cfg = dict(mode="kernel", cell=list(cell), direction=direction,
                           **_config_dict(point))
                print(f"{label}/kernel {_config_tuple(point)} ...", flush=True)
                row = run_one(TORCH_PYTHON, cfg)
                rows.append(row)
                if "error" in row:
                    print(f"  FAILED: {row['error'][:200]}", flush=True)
                else:
                    print(f"  median {statistics.median(row['kernel_times_ms']):.3f}"
                          f" ms  rel {row['value_rel']:.1e}"
                          f"  repeat {row['value_rel_selfrepeat']:.1e}"
                          f"  peak {row['kernel_peak_bytes'] / 2**30:.2f}G",
                          flush=True)
            print_cell_table(cell, direction, rows)
            summary = summarize(cell, direction, rows)
            all_results["rows"].extend(rows)
            all_results["summaries"].append(summary)
            print(f"summary {label}: {json.dumps(summary)}", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5k6_psweep_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


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
        for _direction in selected_directions():
            print_plan(_direction, *build_plan(_direction))
    elif len(sys.argv) >= 2:
        print(f"unknown argument {sys.argv[1]!r}; try --help")
        sys.exit(2)
    else:
        main()
