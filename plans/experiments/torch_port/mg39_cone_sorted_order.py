"""mg39 -- THE SORTED ORDER FED TO THE UNCHANGED SHIPPED CONE FORWARD.

WHY THIS RUN EXISTS.  mg38 measured the grouped cone kernel losing
(findings §1.32): the window mechanics validated but the register
residency never materialized, and the arithmetic tax was paid on top.
Design note §9 records Greg's decision on the rework candidates: start
with A -- feed the two-axis sorted pixel order to the SHIPPED per-tap
kernel, changing no kernel at all -- then evaluate B.  The premise is
mg31's profile: the shipped cone forward's pathology is locality (L1
hit 41.7 percent, nine warps parked), not DRAM bandwidth, and a tile
of 32 pixels adjacent in (channel center, W_p_r) order gathers
overlapping slice ranges and adjacent channels.

WHAT IT MEASURES, at the 1024-class cone cell, one H100:
  * baseline        -- the shipped wrapper in mask (raster) order.
  * sorted_batch    -- one compound sort (32 W_p_r buckets, then
                       channel center) keyed at the batch's middle
                       view, one 64-view call.
  * sorted_chunk8   -- 8-view sub-calls, each with its own sort keyed
  * sorted_chunk16     at the chunk's first view (16 likewise); the
                       ordering staleness shrinks with the chunk.
  * np_only_chunk8  -- the ablation: channel-center-only key at chunk
                       8, discriminating whether the W_p_r axis of
                       the key matters for the vertical gather.
Ladder: full mask, /64 and /128 strided subsets.  Timed region per
arm: key build + sort + values/index permute + wrapper call(s),
mg38's like-for-like convention; the sort's own cost is also recorded
separately.  Values gate: every arm against the baseline output at
1e-5 relative (same kernel, same sums, different summation order).
A bounded profiler pass runs on the winning arm's full-mask call
(never gates): the mechanism reading this spike exists to take.

Run:  <torch python> mg39_cone_sorted_order.py     on one GPU
      MG39_DRY=1 / MG39_SMOKE=1 / MG39_RESULTS=<dir> / MG39_REPEATS=3
      MG39_ARMS=a,b   a subset of the arms, by name
      MG39_NCU=0      skip the profiler pass (default on)
      MG39_NCU_INNER=<arm>  internal: one warm + one measured call
"""

import inspect
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG39_SMOKE", "0") == "1"
DRY = os.environ.get("MG39_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (1024, 1008, 992)          # (views, det rows, channels)
SMOKE_CELL = (8, 32, 20)
VIEW_BATCH = 64                   # mg38's batch, for baseline comparability
BUCKETS = 32                      # mg37's winning bucket count
VALUES_SEED = 20260818
NUM_PIXELS_EXPECTED = 771240
LADDER_DIVISORS = (1, 64, 128)

WARMUP_REPEATS = 1
TIMED_REPEATS = max(1, int(os.environ.get("MG39_REPEATS", "3")))
VALUES_GATE_REL = 1e-5

ARMS = ("sorted_batch", "sorted_chunk8", "sorted_chunk16",
        "np_only_chunk8")
ARM_CHUNK = {"sorted_batch": None, "sorted_chunk8": 8,
             "sorted_chunk16": 16, "np_only_chunk8": 8}
ARM_COMPOUND = {"sorted_batch": True, "sorted_chunk8": True,
                "sorted_chunk16": True, "np_only_chunk8": False}

NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_read.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
])


def cell():
    return SMOKE_CELL if SMOKE else CELL


def build_model():
    """mg31's construction, verbatim."""
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    num_views, channels = shape[0], shape[2]
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    model = mbirtorch.ConeBeamModel(
        shape, angles, source_detector_dist=4.0 * channels,
        source_iso_dist=2.0 * channels)
    model.skip_memory_preflight = True
    model.configure_devices(
        devices=[DEVICE + (":0" if DEVICE == "cuda" else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def call_with_args(fn, first_args, args):
    alias = {"num_rows": "num_recon_rows", "num_cols": "num_recon_cols"}
    params = list(inspect.signature(fn).parameters)[len(first_args):]
    missing = [p for p in params if alias.get(p, p) not in args]
    if missing:
        raise KeyError(f"{fn.__name__} needs {missing} not in args")
    return fn(*first_args, **{p: args[alias.get(p, p)] for p in params})


def seeded_values(torch, model, num_pixels, columns):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VALUES_SEED)
    block = torch.rand((int(num_pixels), int(columns)), generator=generator,
                       dtype=torch.float32)
    return block.to(model.torch_device)


def sort_key_arrays(idx, view_params_row, args):
    """n_p and W_p_r at ONE view, from the shipped builders -- the
    compound key's two axes."""
    from mbirtorch.cone_beam import (_cone_horizontal_data,
                                     _cone_vertical_affine)

    angles = view_params_row[:, 0]
    z_shifts = view_params_row[:, 1]
    n_p, _c, _w, _s, pixel_mag = call_with_args(
        _cone_horizontal_data, (idx, angles), args)
    _m0, w_p_r, _z = call_with_args(
        _cone_vertical_affine, (pixel_mag, z_shifts), args)
    return n_p[0], w_p_r[0]


def compound_perm(torch, n_p, w_p_r, compound):
    """The two-axis ordering: stable sort by channel center, then by
    W_p_r bucket -- a lexsort with the bucket as the primary key.  The
    ablation arm sorts by channel center alone."""
    if not compound:
        return torch.argsort(n_p)
    lo, hi = torch.min(w_p_r), torch.max(w_p_r)
    edges = torch.linspace(lo.item(), hi.item() + 1e-6, BUCKETS + 1,
                           device=w_p_r.device)
    bucket = torch.clamp(torch.bucketize(w_p_r, edges) - 1, 0, BUCKETS - 1)
    ord1 = torch.argsort(n_p)
    return ord1[torch.argsort(bucket[ord1], stable=True)]


def timed(torch, fn):
    cuda = DEVICE == "cuda"
    out = None
    for _ in range(WARMUP_REPEATS):
        out = fn()
    times = []
    for _ in range(TIMED_REPEATS):
        del out
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, dict(median_ms=statistics.median(times), all_ms=times)


def rel_diff(a, b):
    denom = max(float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / denom


def make_arm_fn(torch, fwd_body, args, idx, values, view_params, arm):
    """One call of the arm over the whole view batch: sort (per batch
    or per chunk), permute, call the SHIPPED wrapper, concatenate."""
    chunk = ARM_CHUNK[arm]
    compound = ARM_COMPOUND[arm]
    num_views = int(view_params.shape[0])
    sort_ms_box = [0.0]

    def run():
        sort_ms = 0.0
        outs = []
        step = num_views if chunk is None else chunk
        for i in range(0, num_views, step):
            vp = view_params[i:i + step]
            key_row = (vp[len(vp) // 2:len(vp) // 2 + 1] if chunk is None
                       else vp[:1])
            t0 = time.perf_counter()
            n_p, w_p_r = sort_key_arrays(idx, key_row, args)
            perm = compound_perm(torch, n_p, w_p_r, compound)
            idx_s = idx[perm]
            vals_s = values[perm].contiguous()
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            sort_ms += (time.perf_counter() - t0) * 1000.0
            outs.append(fwd_body(vals_s, idx_s, vp, slice_start=0,
                                 plan=None, **args))
        sort_ms_box[0] = sort_ms
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    return run, sort_ms_box


def profiler_pass(arm, log_path):
    """A bounded ncu attempt on the arm's full-mask call.  Never gates."""
    ncu = shutil.which("ncu")
    if ncu is None:
        extra = "/apps/spack/gautschi-gpu/apps/cuda/12.6.1-gcc-11.4.1-jtjtgkd/bin"
        cand = os.path.join(extra, "ncu")
        ncu = cand if os.path.exists(cand) else None
    if ncu is None:
        return dict(kind="ncu", arm=arm, skipped="no ncu on PATH")
    env = dict(os.environ)
    env["MG39_NCU_INNER"] = arm
    cmd = [ncu, "--metrics", NCU_METRICS, "--launch-skip", "1",
           "--launch-count", "1", "-k", "regex:_cone_forward_kernel",
           "--csv", sys.executable, os.path.abspath(__file__)]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=1200)
        open(log_path, "w").write(proc.stdout + "\n=== stderr ===\n"
                                  + proc.stderr)
        lines = [l for l in proc.stdout.splitlines()
                 if any(m.split(".")[0] in l for m in NCU_METRICS.split(","))]
        return dict(kind="ncu", arm=arm, rc=proc.returncode,
                    lines=lines[-24:], log=log_path)
    except Exception:                                             # noqa: BLE001
        return dict(kind="ncu", arm=arm, error=traceback.format_exc()[-1500:])


def inner_once(arm):
    """The ncu target: one warm call, one measured call, full mask."""
    import torch
    model = build_model()
    args = model._view_batch_args()
    fwd_body, _ = model._view_batch_bodies()
    pf = model.projector_functions
    view_params = pf._view_params_per_dev[0][:VIEW_BATCH]
    idx = model.full_indices_device()
    values = seeded_values(torch, model,
                           int(idx.shape[0]),
                           int(model.get_params("recon_shape")[2]))
    if arm == "baseline":
        def run():
            return fwd_body(values, idx, view_params, slice_start=0,
                            plan=None, **args)
    else:
        run, _ = make_arm_fn(torch, fwd_body, args, idx, values,
                             view_params, arm)
    out = run()
    del out
    torch.cuda.synchronize()
    out = run()
    torch.cuda.synchronize()
    del out
    return 0


def main():
    inner = os.environ.get("MG39_NCU_INNER", "")
    if inner:
        return inner_once(inner)

    arms = [a.strip() for a in os.environ.get("MG39_ARMS", "").split(",")
            if a.strip()] or list(ARMS)
    bad = [a for a in arms if a not in ARMS]
    if bad:
        raise ValueError(f"unknown arms {bad}; known: {list(ARMS)}")

    results_dir = os.environ.get("MG39_RESULTS", ".")
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    node = platform.node().split(".")[0] or "local"
    out_path = os.path.join(
        results_dir, f"mg39_cone_sorted_{node}_{stamp}.jsonl")

    if DRY:
        print(json.dumps(dict(arms=arms, ladder=LADDER_DIVISORS,
                              cell=cell(), buckets=BUCKETS,
                              view_batch=VIEW_BATCH), indent=2))
        return 0

    import torch
    rc = 0
    sink = open(out_path, "w")

    def emit(row):
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    model = build_model()
    args = model._view_batch_args()
    fwd_body, _ = model._view_batch_bodies()
    triton_bound = fwd_body.__name__.endswith("_triton")
    pf = model.projector_functions
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    view_params = pf._view_params_per_dev[0][:view_batch]
    idx_full = model.full_indices_device()
    num_pixels = int(idx_full.shape[0])
    num_slices = int(model.get_params("recon_shape")[2])
    pixels_ok = SMOKE or num_pixels == NUM_PIXELS_EXPECTED
    rc = rc if pixels_ok else 1
    emit(dict(kind="header", cell=cell(), node=node, arms=arms,
              buckets=BUCKETS, view_batch=view_batch,
              num_pixels=num_pixels, pixels_ok=pixels_ok,
              forward_body=fwd_body.__name__,
              triton_bound=triton_bound,
              device=(torch.cuda.get_device_name(0)
                      if DEVICE == "cuda" else "cpu")))
    print(f"mg39 on {node}: body {fwd_body.__name__}, "
          f"{num_pixels} pixels, batch {view_batch}", flush=True)
    values_full = seeded_values(torch, model, num_pixels, num_slices)

    winner = (None, 0.0)
    try:
        for divisor in LADDER_DIVISORS:
            want = max(1, num_pixels // divisor)
            step = max(1, num_pixels // want)
            idx = idx_full[::step][:want].contiguous()
            values = (values_full if divisor == 1
                      else values_full[:int(idx.shape[0])].contiguous())

            def base_call():
                return fwd_body(values, idx, view_params, slice_start=0,
                                plan=None, **args)

            reference, t = timed(torch, base_call)
            base_ms = t["median_ms"]
            emit(dict(kind="point", arm="baseline", divisor=divisor,
                      num_pixels=int(idx.shape[0]), **t))
            print(f'  baseline /{divisor:<3d} ({int(idx.shape[0])} px): '
                  f'{base_ms:9.2f} ms', flush=True)

            for arm in arms:
                run, sort_box = make_arm_fn(torch, fwd_body, args, idx,
                                            values, view_params, arm)
                try:
                    out, t = timed(torch, run)
                except Exception:                                 # noqa: BLE001
                    emit(dict(kind="point", arm=arm, divisor=divisor,
                              error=traceback.format_exc()[-2000:]))
                    rc = 1
                    print(f'    {arm:<18} ERROR (recorded)', flush=True)
                    continue
                rel = rel_diff(out, reference)
                ok = rel <= VALUES_GATE_REL
                rc = rc if ok else 1
                speedup = base_ms / t["median_ms"]
                emit(dict(kind="point", arm=arm, divisor=divisor,
                          rel_diff=rel, values_ok=ok, speedup=speedup,
                          sort_ms=sort_box[0], **t))
                print(f'    {arm:<18} {t["median_ms"]:9.2f} ms '
                      f'{speedup:5.2f}x  values {rel:.2e} '
                      f'{"ok" if ok else "FAIL"}  sort {sort_box[0]:.1f} ms',
                      flush=True)
                del out
                if divisor == 1 and ok and speedup > winner[1]:
                    winner = (arm, speedup)
            del reference
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        if (DEVICE == "cuda" and winner[0] is not None
                and os.environ.get("MG39_NCU", "1") == "1"):
            print(f"profiler pass on {winner[0]} "
                  f"({winner[1]:.2f}x at the full mask)", flush=True)
            row = profiler_pass(winner[0],
                                out_path.replace(".jsonl", "_ncu.log"))
            emit(row)
            for line in row.get("lines", []):
                print("   ", line, flush=True)
    except Exception:                                             # noqa: BLE001
        emit(dict(kind="error", trace=traceback.format_exc()[-4000:]))
        rc = 1
    finally:
        emit(dict(kind="done", rc=rc, winner=winner[0],
                  winner_speedup=winner[1]))
        sink.close()
        print(f"rows: {out_path}", flush=True)
        print(f"MG39 rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
