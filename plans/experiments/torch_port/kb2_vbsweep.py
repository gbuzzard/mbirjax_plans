"""The VIEW-CHUNK sweep for kernel-aware view batching -- the driver-level
bench that picks the four ``*_VIEW_CHUNK`` constants (triton_parallel.py,
triton_cone.py) before the composed gate (kb3_gate.py) decides anything.

What is being swept, and why it is a DRIVER bench.  The batching change gives
each Triton kernel body its own view-batch rule (the ``_view_batch_cost``
attribute): a nominal view chunk capped by the transient budget.  The cost of
the old batching was per-call overhead -- a launch and a fresh hfan contract
build per view batch -- so the quantity that decides the chunk is the whole
view-range loop of ``Projectors`` (``sparse_forward_project_view_range`` /
``sparse_back_project_view_range``), timed through the driver exactly as
production runs it, never an isolated wrapper call.  Rows cover both
geometries, both directions, both gate cells, and two pixel classes (the full
ROR set, where the old rule collapsed to view batch 1 at the 1024 cell, and a
stride-64 subset standing in for the VCD fine tail).

The arms, per (cell, geometry, direction, pixel class):

  - ``torch_ref``: the production torch path -- kill switch set, bodies
    COMPILED by the driver -- timed through the same loop.  Its output is
    saved as the combo's reference artifact; every kernel row's value column
    reads against it.  This is also the "today" torch column of the summary.
  - ``kernel_chunk``: the kernel path with the direction's ``*_VIEW_CHUNK``
    module constant overridden to each swept value.  The realized batch is
    read back from ``_effective_view_batch`` (the driver's own rule, not a
    guess) and duplicate realized batches are SKIPPED by the runner: at the
    cells where the budget cap binds below several chunk values, those rows
    would measure the same configuration twice.
  - ``kernel_legacy``: the kernel bodies under TODAY's charged batching,
    emulated exactly by setting ``model.view_batch_size`` to the old rule's
    batch (min(64, budget // torch charge)); an explicit nominal caps kernel
    batches by design, and the row asserts the realized batch equals it.
    This is the "kernels before this change" comparison arm.

Value protocol.  Every kernel row reports the max-rel difference of its
driver output against the combo's torch_ref artifact, at the same inputs
(one seeded generator per combo).  The comparison crosses different float
summation orders (different batchings, and atomics on the forward), so the
flag tolerance is the design's coeff_power-2 figure, 1e-4, and each row also
reports its own kernel-repeat floor (two identical driver calls diffed):
the forward's atomic floor is the number its value column is read against,
and the back path's floor must be exactly zero at a fixed batch.

Peaks and compiles.  Every timed section reports the CUDA peak (reset after
warmup), because the chunk pin spends real memory headroom and the pin is
chosen from the joint time-and-peak readout.  Each row also reads back
triton's compiled-variant metadata (n_regs / n_spills / count), so variant
churn across batch sizes is observed rather than assumed.

No jax arm: the composed gate (kb3) is the cross-framework ruler, with the
shared-sinogram value protocol; this sweep ranks chunk values within the
torch stack.

Run:
    <torch python> kb2_vbsweep.py        on a CUDA node (see kb2_gautschi.sbatch)
    python kb2_vbsweep.py --dry-run      anywhere: print the row plan
    python kb2_vbsweep.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  Every list is
parsed STRICTLY: an unrecognized token is a hard error, never a silent skip.
    P0_TORCH_PYTHON                   interpreter for the row subprocesses
    KB2_CELLS=512,1024                subset of the cells (by view count)
    KB2_GEOMS=parallel,cone           subset of the geometries
    KB2_DIRECTIONS=back,fwd           subset of the directions
    KB2_PCLASSES=full,subset          subset of the pixel classes
    KB2_CHUNKS=16,32,64,128,256       swept chunk values (positive ints)
    KB2_SMOKE=1                       tiny cell on KB2_DEVICE (default cpu):
                                      exercises every step but the launches
    KB2_DEVICE=cuda|cpu               the torch device for the rows
"""

import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The Phase 2/3/4 gate cells, (views, rows, channels).
CELLS = [(512, 448, 384), (1024, 1008, 992)]
GEOMETRIES = ("parallel", "cone")
DIRECTIONS = ("back", "fwd")
PCLASSES = ("full", "subset")
DEFAULT_CHUNKS = (16, 32, 64, 128, 256)
SUBSET_STRIDE = 64          # the "fine tail" pixel class: full_indices[::64]

SMOKE = os.environ.get("KB2_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("KB2_DEVICE", "cpu" if SMOKE else "cuda")

WARMUP = 1          # pays the triton/inductor compiles and the allocator
TRIALS = 3
INPUT_SEED = 0      # a private generator: the recon gates read the global RNG
COEFF_POWER = 1     # the gradient path -- where the VCD loop spends its time
VALUE_REL_TOL = 1e-4
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]

# Per (geometry, direction): the kernel module's names for the chunk constant,
# the wrapper, the torch body, and the JITFunction whose metadata is read.
SPEC = {
    ("parallel", "back"): dict(chunk_const="PARALLEL_BACK_VIEW_CHUNK",
                               wrapper="_parallel_back_view_batch_triton",
                               torch_body="_parallel_back_view_batch",
                               kernel="_parallel_back_kernel"),
    ("parallel", "fwd"): dict(chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                              wrapper="_parallel_forward_view_batch_triton",
                              torch_body="_parallel_forward_view_batch",
                              kernel="_parallel_forward_kernel"),
    ("cone", "back"): dict(chunk_const="CONE_BACK_VIEW_CHUNK",
                           wrapper="_cone_back_view_batch_triton",
                           torch_body="_cone_back_view_batch",
                           kernel="_cone_back_kernel"),
    ("cone", "fwd"): dict(chunk_const="CONE_FWD_VIEW_CHUNK",
                          wrapper="_cone_forward_view_batch_triton",
                          torch_body="_cone_forward_view_batch",
                          kernel="_cone_forward_kernel"),
}
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """The env-list parse that refuses garbage: every token must name a member
    of ``allowed`` after ``cast``, and an unparsable or unknown token raises
    (the slurm --export comma-split lesson: a truncated list must fail loud,
    never sweep a silently smaller plan)."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = cast(token)
        except ValueError:
            raise ValueError(f"{env_name}: unparsable token {token!r}")
        if value not in allowed:
            raise ValueError(f"{env_name}: {value!r} is not one of "
                             f"{sorted(allowed)}")
        chosen.append(value)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return chosen


def selected_plan():
    """The ordered row plan: one combo dict per (cell, geometry, direction,
    pixel class), each carrying its torch_ref, legacy, and chunk rows."""
    if SMOKE:
        cells = [SMOKE_CELL]
    else:
        keep = _strict_subset("KB2_CELLS", {c[0] for c in CELLS}, int)
        cells = [c for c in CELLS if c[0] in keep]
    geometries = _strict_subset("KB2_GEOMS", set(GEOMETRIES))
    directions = _strict_subset("KB2_DIRECTIONS", set(DIRECTIONS))
    pclasses = _strict_subset("KB2_PCLASSES", set(PCLASSES))
    raw_chunks = os.environ.get("KB2_CHUNKS", "").strip()
    if raw_chunks:
        chunks = []
        for token in raw_chunks.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit() or int(token) <= 0:
                raise ValueError(f"KB2_CHUNKS: bad chunk {token!r}")
            chunks.append(int(token))
    else:
        chunks = list(DEFAULT_CHUNKS)
    combos = []
    for cell in cells:
        for geometry in geometries:
            for direction in directions:
                for pclass in pclasses:
                    combos.append(dict(cell=cell, geometry=geometry,
                                       direction=direction, pclass=pclass,
                                       chunks=chunks))
    return combos


# ── triton cache introspection (the p5k6 helpers, verbatim in intent) ─────────
def _is_kernel_cache(candidate):
    return (isinstance(candidate, dict)
            and all(hasattr(v, "metadata") or hasattr(v, "n_regs")
                    for v in candidate.values()))


def _kernel_cache_dicts(jit_fn):
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


def _compiled_metadata(jit_fn):
    entries = []
    try:
        for cache in _kernel_cache_dicts(jit_fn):
            for kernel in cache.values():
                metadata = getattr(kernel, "metadata", None)
                entries.append(dict(
                    num_warps=getattr(metadata, "num_warps", None),
                    n_regs=getattr(kernel, "n_regs", None),
                    n_spills=getattr(kernel, "n_spills", None)))
    except Exception:                                             # noqa: BLE001
        entries = []
    return entries


# ── the row worker ────────────────────────────────────────────────────────────
def _build_model(geometry, cell, compile_mode):
    import numpy as np

    import mbirtorch

    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles, device=DEVICE,
                                            compile_mode=compile_mode)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels,
                                        device=DEVICE,
                                        compile_mode=compile_mode)
    model.set_params(no_warning=True, verbose=0)
    return model


def _rel_max_vs_artifact(out_cpu, ref_path):
    """Max-rel difference against the combo's reference artifact, read in
    slabs through a memory map so a 4 GB reference never doubles in host
    memory.  The scale is the artifact's own max, computed in the same pass;
    an identically zero reference is refused rather than passed."""
    import numpy as np

    ref = np.load(ref_path, mmap_mode="r")
    flat_out = out_cpu.reshape(-1)
    flat_ref = ref.reshape(-1)
    if flat_out.shape != flat_ref.shape:
        raise ValueError(f"shape mismatch vs artifact: {out_cpu.shape} "
                         f"vs {ref.shape}")
    slab = 1 << 26
    worst = 0.0
    scale = 0.0
    for start in range(0, flat_ref.shape[0], slab):
        r = np.asarray(flat_ref[start:start + slab], dtype=np.float64)
        o = np.asarray(flat_out[start:start + slab], dtype=np.float64)
        worst = max(worst, float(np.abs(o - r).max()))
        scale = max(scale, float(np.abs(r).max()))
    if scale == 0.0:
        raise ValueError("the reference artifact is identically zero")
    return worst / scale


def torch_worker(cfg):
    """One row: torch_ref, kernel_chunk, or kernel_legacy, for one combo."""
    import numpy as np
    import torch

    import mbirtorch
    from mbirtorch import kernel_availability

    geometry, direction = cfg["geometry"], cfg["direction"]
    spec = SPEC[(geometry, direction)]
    cell = tuple(cfg["cell"])
    num_views, num_det_rows, num_channels = cell
    mode = cfg["mode"]

    if geometry == "parallel":
        import mbirtorch.triton_parallel as kernel_module
    else:
        import mbirtorch.triton_cone as kernel_module

    compile_mode = "auto" if mode == "torch_ref" else "off"
    model = _build_model(geometry, cell, compile_mode)
    device = model.torch_device

    def sync():
        if DEVICE == "cuda":
            torch.cuda.synchronize()

    def reset_peak():
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def peak_bytes():
        return int(torch.cuda.max_memory_allocated()) if DEVICE == "cuda" else 0

    # ARM CHECK (the reworked-gate lesson: pin what each arm binds, at run
    # time).  torch_ref runs under the kill switch and must bind the torch
    # body; kernel rows must bind the triton wrapper.  In smoke mode on a
    # non-CUDA device the kernel rows report the miss and stop cleanly.
    fwd_hook, back_hook = model._view_batch_bodies()
    hook = {"fwd": fwd_hook, "back": back_hook}[direction]
    hook_name = getattr(hook, "__name__", str(hook))
    kernel_bound = hook_name == spec["wrapper"]
    result = dict(cfg, recon_shape=list(model.get_params("recon_shape")),
                  bound_body=hook_name, kernel_bound=kernel_bound,
                  triton_available=list(kernel_availability.triton_available()),
                  version=f"torch {torch.__version__}", device=DEVICE,
                  device_name=(torch.cuda.get_device_name(0)
                               if DEVICE == "cuda" else DEVICE))
    if mode == "torch_ref":
        if kernel_bound:
            result["error"] = ("arm check failed: torch_ref bound the kernel "
                               "body despite the kill switch")
            return result
    elif not kernel_bound:
        result["error"] = f"arm check failed: kernel row bound {hook_name}"
        result["smoke_expected"] = SMOKE and DEVICE != "cuda"
        return result

    recon_shape = tuple(model.get_params("recon_shape"))
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    if cfg["pclass"] == "subset":
        pixel_indices = pixel_indices[::SUBSET_STRIDE].contiguous()
    num_pixels = int(pixel_indices.shape[0])
    num_slices = int(recon_shape[2])
    pf = model.projector_functions
    args = model._view_batch_args()

    generator = torch.Generator().manual_seed(INPUT_SEED)
    if direction == "back":
        sino = torch.rand(cell, generator=generator).to(device)
        cols = int(sino.shape[1])

        def driver_call():
            return pf.sparse_back_project_view_range(
                sino, pixel_indices, (0, num_views), coeff_power=COEFF_POWER)
    else:
        values = torch.rand((num_pixels, num_slices),
                            generator=generator).to(device)
        cols = int(values.shape[1])

        def driver_call():
            return pf.sparse_forward_project_view_range(
                values, pixel_indices, (0, num_views))

    # The old rule's batch at these inputs, computed from the same pieces the
    # old driver used: min(64, budget // (P * transient_cols * 4)), floored
    # at 1.  Recorded on every row; the legacy row also RUNS at it.
    budget = pf._transient_budget_bytes()
    legacy_charge = num_pixels * model._transient_cols(cols) * 4
    legacy_vb = max(1, min(64, budget // max(1, legacy_charge)))
    result.update(num_pixels=num_pixels, cols=cols, budget_bytes=int(budget),
                  legacy_view_batch=int(legacy_vb))

    if mode == "kernel_chunk":
        setattr(kernel_module, spec["chunk_const"], int(cfg["chunk"]))
    elif mode == "kernel_legacy":
        model.view_batch_size = int(legacy_vb)

    bound = (pf._back_body_per_dev[0] if direction == "back"
             else pf._fwd_body_per_dev[0])
    realized_vb = int(pf._effective_view_batch(bound, num_pixels, cols, args))
    result["realized_view_batch"] = realized_vb
    if mode == "kernel_chunk":
        cost_fn = bound._view_batch_cost
        bytes_pv, _ = cost_fn(num_pixels, cols, args)
        result["kernel_bytes_per_view"] = int(bytes_pv)
        expected = max(1, min(int(cfg["chunk"]), budget // max(1, bytes_pv)))
        if realized_vb != expected:
            result["error"] = (f"realized batch {realized_vb} != expected "
                               f"{expected} at chunk {cfg['chunk']}")
            return result
    elif mode == "kernel_legacy" and realized_vb != legacy_vb:
        result["error"] = (f"legacy emulation realized {realized_vb} != "
                           f"{legacy_vb}")
        return result

    ref_path = cfg["ref_path"]
    first = driver_call()
    sync()
    result["checksum"] = float(first.abs().sum().item())
    if mode == "torch_ref":
        np.save(ref_path, first.detach().cpu().numpy())
        result["ref_saved"] = True
    else:
        second = driver_call()
        sync()
        scale = max(float(first.abs().max()), 1e-30)
        result["value_rel_selfrepeat"] = (
            float((second - first).abs().max()) / scale)
        del second
        result["value_rel"] = _rel_max_vs_artifact(
            first.detach().cpu().numpy(), ref_path)
        result["value_pass"] = bool(result["value_rel"] <= VALUE_REL_TOL)
    del first

    # Peaks reset AFTER the warmup (the steady-state number is the one the
    # composed run pays every iteration; the first call pays compiles).
    _ = driver_call()
    sync()
    reset_peak()
    times = []
    for _ in range(TRIALS):
        sync()
        t0 = time.perf_counter()
        out = driver_call()
        sync()
        times.append((time.perf_counter() - t0) * 1e3)
        del out
    result["driver_times_ms"] = times
    result["driver_peak_bytes"] = peak_bytes()
    if mode != "torch_ref":
        result["compiled_variants"] = _compiled_metadata(
            getattr(kernel_module, spec["kernel"]))
    return result


# ── the runner ────────────────────────────────────────────────────────────────
def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_kb2.json")
    out_path = os.path.join(RESULTS_DIR, "_out_kb2.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = dict(os.environ)
    if cfg["mode"] == "torch_ref":
        env["MBIRTORCH_DISABLE_TRITON"] = "1"
    else:
        env["MBIRTORCH_DISABLE_TRITON"] = "0"
    proc = subprocess.run([TORCH_PYTHON, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=env)
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(error=f"worker exited {proc.returncode}", **cfg)
    else:
        with open(out_path) as f:
            row = json.load(f)
    return row


def _median(row, key):
    times = row.get(key)
    return statistics.median(times) if times else None


def _combo_name(combo):
    return (f"{combo['cell'][0]}-{combo['geometry']}-{combo['direction']}-"
            f"{combo['pclass']}")


def print_combo_table(combo, rows):
    ref = next((r for r in rows if r.get("mode") == "torch_ref"), None)
    ref_ms = _median(ref, "driver_times_ms") if ref else None
    header = ref if (ref and not ref.get("error")) else rows[-1]
    print(f"\n== {_combo_name(combo)}  "
          f"(P={header.get('num_pixels')}, "
          f"legacy vb={header.get('legacy_view_batch')}) ==")
    print(f"{'mode':>14}{'chunk':>7}{'vb':>6}{'ms':>10}{'peak_GB':>9}"
          f"{'vs_ref':>8}{'value_rel':>11}{'floor':>10}{'spills':>7}")
    for row in rows:
        if row.get("error"):
            print(f"{row['mode']:>14}{str(row.get('chunk', '')):>7}  "
                  f"ERROR: {row['error']}")
            continue
        ms = _median(row, "driver_times_ms")
        peak = row.get("driver_peak_bytes", 0) / 2 ** 30
        ratio = f"{ref_ms / ms:.2f}x" if (ref_ms and ms) else "-"
        value = row.get("value_rel")
        floor = row.get("value_rel_selfrepeat")
        spills = sum(1 for v in row.get("compiled_variants", [])
                     if v.get("n_spills"))
        print(f"{row['mode']:>14}{str(row.get('chunk', '')):>7}"
              f"{row.get('realized_view_batch', 0):>6}"
              f"{ms:>10.1f}{peak:>9.2f}{ratio:>8}"
              f"{(f'{value:.1e}' if value is not None else '-'):>11}"
              f"{(f'{floor:.1e}' if floor is not None else '-'):>10}"
              f"{spills:>7}")


def main():
    if "--dry-run" in sys.argv:
        for combo in selected_plan():
            print(_combo_name(combo), "chunks", combo["chunks"])
        return
    combos = selected_plan()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR,
                            f"kb2_vbsweep_{RUN_LABEL}_{stamp}.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"kb2 view-chunk sweep on {RUN_LABEL} ({DEVICE}); "
          f"{len(combos)} combos -> {out_path}")
    with open(out_path, "w") as sink:
        for combo in combos:
            ref_path = os.path.join(
                RESULTS_DIR, f"_kb2_ref_{_combo_name(combo)}.npy")
            base = dict(cell=list(combo["cell"]), geometry=combo["geometry"],
                        direction=combo["direction"], pclass=combo["pclass"],
                        ref_path=ref_path)
            rows = []

            def run_and_log(cfg):
                row = run_one(cfg)
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                return row

            ref_row = run_and_log(dict(base, mode="torch_ref", chunk=None))
            if ref_row.get("error"):
                print(f"{_combo_name(combo)}: torch_ref failed -- "
                      f"{ref_row['error']}; skipping the combo")
                print_combo_table(combo, rows)
                continue
            run_and_log(dict(base, mode="kernel_legacy", chunk=None))
            seen_realized = set()
            for chunk in combo["chunks"]:
                row = run_and_log(dict(base, mode="kernel_chunk", chunk=chunk))
                realized = row.get("realized_view_batch")
                if realized is not None and realized in seen_realized:
                    # The budget cap bound below this chunk value, so the row
                    # measured a configuration an earlier row already did;
                    # logged (no silent caps) and the duplicate is kept in the
                    # JSONL for the record.
                    print(f"  note: chunk {chunk} realized vb={realized}, "
                          f"duplicate of an earlier row")
                seen_realized.add(realized)
            print_combo_table(combo, rows)
            if os.path.exists(ref_path):
                os.remove(ref_path)
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        row = torch_worker(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(error=traceback.format_exc()[-2000:], **cfg)
    with open(out_path, "w") as f:
        json.dump(row, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        main()
