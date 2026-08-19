"""mg36 -- SINGLE-PROJECTION ANCHORS AND TABLE 1'S 1/128 ROW.

WHY THIS RUN EXISTS.  The kernels document
(plans/torch_port/triton_kernels.md) is being revised to end each kernel
section with the cost of ONE full projection at the 1024-class example,
and to add the 1/128 pixel row to its per-launch table.  Neither number
is recorded anywhere: the run records hold per-launch times only (mg31's
cone anchor is a single descriptive observation, the parallel back has
no timing at all), and mg33's ladder stopped at /64.  This run measures
exactly those cells and nothing else.

WHAT IT MEASURES, all at mg20's 1024-class construction, one H100,
each number the median of 3 timed repeats after 1 discarded warm-up,
with a device synchronize around every timed call:

  1. WHOLE-CALL projections through the model API, full mask, all views:
     * parallel forward (`sparse_forward_project`), sorted route and
       per-tap route (the per-call MBIRTORCH_SORTED_FORWARD switch);
     * parallel back (`sparse_back_project`);
     * cone forward and cone back, same calls on mg31's cone model.
  2. ONE 128-view wrapper launch at the 1/128 subset (6,025 pixels,
     mg33's strided construction: idx_full[::step][:want]), sorted and
     per-tap -- the missing row of the kernels document's Table 1,
     measured exactly as mg33 measured the /1 to /64 rows.

THE GATES (exit code = instrument health): the sorted and per-tap
routes agree at 1e-5 relative, both whole-call and at the 1/128 launch
(the sort reorders the same commutative sums); the full mask realizes
771,240 pixels; no call raises.  The cone rows and the back rows have
no pair to gate against and are recorded as timings only.

Run:
    <torch python> mg36_single_projection.py       on one GPU
    MG36_DRY=1    print the plan and stop
    MG36_SMOKE=1  tiny CPU plumbing pass
    MG36_RESULTS=<dir>   where the jsonl goes
    MG36_REPEATS=3       timed repeats per measurement
"""

import json
import os
import platform
import statistics
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG36_SMOKE", "0") == "1"
DRY = os.environ.get("MG36_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (1024, 1008, 992)          # (views, detector rows, channels)
SMOKE_CELL = (8, 32, 20)
VIEW_BATCH = 128
VALUES_SEED = 20260818            # mg33's seed
NUM_PIXELS_EXPECTED = 771240      # gated on the real cell
SUBSET_DIVISOR = 128

WARMUP_REPEATS = 1
TIMED_REPEATS = max(1, int(os.environ.get("MG36_REPEATS", "3")))
VALUES_GATE_REL = 1e-5


def cell():
    return SMOKE_CELL if SMOKE else CELL


# ── models (mg20's constructions, verbatim from mg33 and mg31) ────────────────
def build_parallel():
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


def build_cone():
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


def seeded_values(torch, model, num_pixels, columns):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(VALUES_SEED)
    block = torch.rand((int(num_pixels), int(columns)), generator=generator,
                       dtype=torch.float32)
    return block.to(model.torch_device)


# ── timing and comparison ─────────────────────────────────────────────────────
def timed(torch, fn):
    """Median of TIMED_REPEATS wall times after WARMUP_REPEATS discarded
    calls; the last call's output is returned for value gates."""
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
    return out, {"median_ms": statistics.median(times), "all_ms": times,
                 "repeats": TIMED_REPEATS}


def rel_diff(a, b):
    denom = max(float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / denom


def set_route(name):
    """The sorted/per-tap switch is read per call, so flipping the
    environment variable between calls flips the route."""
    os.environ["MBIRTORCH_SORTED_FORWARD"] = "1" if name == "sorted" else "0"


# ── the run ───────────────────────────────────────────────────────────────────
def main():
    results_dir = os.environ.get("MG36_RESULTS", ".")
    os.makedirs(results_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    node = platform.node().split(".")[0] or "local"
    out_path = os.path.join(
        results_dir, f"mg36_single_projection_{node}_{stamp}.jsonl")

    plan = {
        "whole_call": ["parallel fwd sorted", "parallel fwd per-tap",
                       "parallel back", "cone fwd", "cone back"],
        "launch_128": ["1/128 subset, sorted and per-tap, 128 views"],
        "cell": cell(), "repeats": TIMED_REPEATS,
    }
    if DRY:
        print(json.dumps(plan, indent=2))
        return 0

    import torch
    cuda = DEVICE == "cuda"
    rc = 0
    sink = open(out_path, "w")

    def emit(row):
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        keys = {k: v for k, v in row.items()
                if k in ("kind", "geometry", "direction", "route",
                         "median_ms", "rel_diff", "values_ok",
                         "num_pixels")}
        print("  " + json.dumps(keys), flush=True)

    header = dict(kind="header", plan=plan, node=node, smoke=SMOKE,
                  device=torch.cuda.get_device_name(0) if cuda else "cpu",
                  torch=torch.__version__)
    emit(header)

    try:
        # ── parallel, whole-call ──────────────────────────────────────
        model = build_parallel()
        num_slices = int(model.get_params("recon_shape")[2])
        idx_full = model.full_indices_device()
        num_pixels = int(idx_full.shape[0])
        pixels_ok = SMOKE or num_pixels == NUM_PIXELS_EXPECTED
        emit(dict(kind="mask", geometry="parallel", num_pixels=num_pixels,
                  values_ok=pixels_ok))
        rc = rc if pixels_ok else 1
        values_full = seeded_values(torch, model, num_pixels, num_slices)

        set_route("sorted")
        sino_sorted, t = timed(
            torch, lambda: model.sparse_forward_project(values_full,
                                                        idx_full))
        emit(dict(kind="whole_call", geometry="parallel",
                  direction="forward", route="sorted", **t))

        set_route("per_tap")
        sino_tap, t = timed(
            torch, lambda: model.sparse_forward_project(values_full,
                                                        idx_full))
        emit(dict(kind="whole_call", geometry="parallel",
                  direction="forward", route="per_tap", **t))

        rel = rel_diff(sino_sorted, sino_tap)
        ok = rel <= VALUES_GATE_REL
        emit(dict(kind="gate", geometry="parallel", direction="forward",
                  rel_diff=rel, values_ok=ok))
        rc = rc if ok else 1
        del sino_sorted

        set_route("sorted")
        back_out, t = timed(
            torch, lambda: model.sparse_back_project(sino_tap, idx_full))
        emit(dict(kind="whole_call", geometry="parallel", direction="back",
                  route="only", **t))
        del back_out, sino_tap

        # ── parallel, the 1/128 launch (mg33's quantity) ──────────────
        fwd_body, _back_body = model._view_batch_bodies()
        args = model._view_batch_args()
        pf = model.projector_functions
        view_batch = min(VIEW_BATCH, int(cell()[0]))
        view_params = pf._view_params_per_dev[0][:view_batch]
        want = max(1, num_pixels // SUBSET_DIVISOR)
        step = max(1, num_pixels // want)
        idx_sub = idx_full[::step][:want].contiguous()
        vals_sub = values_full[:int(idx_sub.shape[0])].contiguous()
        emit(dict(kind="mask", geometry="parallel",
                  num_pixels=int(idx_sub.shape[0]),
                  values_ok=True))

        def launch():
            return fwd_body(vals_sub, idx_sub, view_params, slice_start=0,
                            plan=None, **args)

        set_route("sorted")
        out_sorted, t = timed(torch, launch)
        emit(dict(kind="launch_128", divisor=SUBSET_DIVISOR,
                  route="sorted", **t))
        set_route("per_tap")
        out_tap, t = timed(torch, launch)
        emit(dict(kind="launch_128", divisor=SUBSET_DIVISOR,
                  route="per_tap", **t))
        rel = rel_diff(out_sorted, out_tap)
        ok = rel <= VALUES_GATE_REL
        emit(dict(kind="gate", geometry="parallel", direction="launch_128",
                  rel_diff=rel, values_ok=ok))
        rc = rc if ok else 1
        del out_sorted, out_tap, vals_sub, values_full, model
        if cuda:
            torch.cuda.empty_cache()

        # ── cone, whole-call ──────────────────────────────────────────
        set_route("sorted")            # the shipped default; cone ignores it
        model = build_cone()
        num_slices = int(model.get_params("recon_shape")[2])
        idx_full = model.full_indices_device()
        emit(dict(kind="mask", geometry="cone",
                  num_pixels=int(idx_full.shape[0]), values_ok=True))
        values_full = seeded_values(torch, model, int(idx_full.shape[0]),
                                    num_slices)
        sino, t = timed(
            torch, lambda: model.sparse_forward_project(values_full,
                                                        idx_full))
        emit(dict(kind="whole_call", geometry="cone", direction="forward",
                  route="only", **t))
        del values_full
        back_out, t = timed(
            torch, lambda: model.sparse_back_project(sino, idx_full))
        emit(dict(kind="whole_call", geometry="cone", direction="back",
                  route="only", **t))
        del back_out, sino, model
    except Exception:                                             # noqa: BLE001
        emit(dict(kind="error", trace=traceback.format_exc()[-4000:]))
        rc = 1
    finally:
        os.environ.pop("MBIRTORCH_SORTED_FORWARD", None)
        emit(dict(kind="done", rc=rc))
        sink.close()
        print(f"rows: {out_path}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
