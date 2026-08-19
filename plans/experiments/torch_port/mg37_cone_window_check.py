"""mg37 -- THE PAPER CHECK FOR THE CONE TWO-AXIS GROUPING (design note
`pfwd_segmented_design.md` section 9).

WHY THIS RUN EXISTS.  The cone pixel-batched form holds a (pixel tile x
slice tile) block of the flat recon and serves a chunk of views; its
feasibility rests on whether pixels grouped by (channel center, W_p_r)
give tiles whose detector footprint fits a small (channel x row)
window.  This check computes those windows EXACTLY from the shipped
geometry builders (`_cone_horizontal_data`, `_cone_vertical_affine` --
the sanctioned affine bridge row = m0 + W_p_r x slice) on the CPU, at
the 1024-class cone cell.  No GPU, no kernel: it decides cheaply
whether the spike is worth building.

WHAT IT COMPUTES, at reference angles 0/45/90/135 degrees:
  1. Sort the full mask by the compound key (W_p_r bucket, n_p) at the
     chunk's first view; tile 32 consecutive pixels.
  2. For each (buckets B, chunk C, slice tile S, probe slice l): the
     EXACT per-tile channel window (n_p span over tile x chunk views,
     plus horizontal taps) and row window (span of m0 + W_p_r x l over
     tile x chunk at the slice tile's two endpoint slices, plus the
     vertical footprint pad).  Distributions over tiles, and the
     fraction fitting candidate windows.
  3. The accumulator bill for fitting windows (Wc x Wr floats per
     tile patch) and the count of bucket-boundary tiles (mixed
     buckets; at most B-1 of them).

Run:  PYTHONPATH=<mbirtorch checkout> python mg37_cone_window_check.py
      MG37_RESULTS=<dir> for the jsonl (default: beside this file).
"""

import inspect
import json
import os
import platform
import time

import numpy as np
import torch

CELL = (1024, 1008, 992)              # (views, det rows, channels)
REF_VIEWS = (0, 128, 256, 384)        # 0/45/90/135 degrees at 2pi/1024
CHUNKS = (8, 16)
BUCKETS = (16, 32, 64)
SLICE_TILES = (4, 8, 16)
PROBE_SLICES = (0, 250, 500, 750, 991)
TILE_P = 32
ROW_FITS = (16, 32)
CHAN_FITS = (8, 16)


def build_model():
    """mg31's construction, verbatim."""
    import mbirtorch

    shape = CELL
    num_views, channels = shape[0], shape[2]
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    model = mbirtorch.ConeBeamModel(
        shape, angles, source_detector_dist=4.0 * channels,
        source_iso_dist=2.0 * channels)
    model.skip_memory_preflight = True
    model.configure_devices(devices=["cpu"])
    model.set_params(no_warning=True, verbose=0)
    return model


def call_with_args(fn, first_args, args):
    """Call a builder, filling its named geometry parameters from the
    model's `_view_batch_args()` dict; missing names are an error."""
    alias = {"num_rows": "num_recon_rows", "num_cols": "num_recon_cols"}
    params = list(inspect.signature(fn).parameters)[len(first_args):]
    missing = [p for p in params if alias.get(p, p) not in args]
    if missing:
        raise KeyError(f"{fn.__name__} needs {missing} not in args")
    return fn(*first_args, **{p: args[alias.get(p, p)] for p in params})


def tile_min_max(values, num_tiles):
    """Per-tile min and max over (chunk views x 32 pixels) JOINTLY --
    the union footprint a per-CHUNK flush would need: values is (C, P)
    in SORTED pixel order; returns two (num_tiles,) arrays."""
    v = values[:, :num_tiles * TILE_P]
    per_pixel_min = v.min(axis=0)
    per_pixel_max = v.max(axis=0)
    return (per_pixel_min.reshape(num_tiles, TILE_P).min(axis=1),
            per_pixel_max.reshape(num_tiles, TILE_P).max(axis=1))


def per_view_tile_span(list_of_values, num_tiles):
    """The window a per-VIEW flush needs: within each view, the tile's
    span over the given arrays jointly (e.g. a slice tile's two
    endpoint rows), then the MAX over the chunk's views.  Common-mode
    drift across views moves the window without widening it, so it is
    excluded here by construction.  Each array is (C, P) in sorted
    order; returns one (num_tiles,) array."""
    C = list_of_values[0].shape[0]
    n = num_tiles * TILE_P
    stack = np.stack([v[:, :n] for v in list_of_values])   # (A, C, P)
    tiles = stack.reshape(len(list_of_values), C, num_tiles, TILE_P)
    span = tiles.max(axis=(0, 3)) - tiles.min(axis=(0, 3))  # (C, T)
    return span.max(axis=0)


def dist(a):
    return dict(median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def main():
    out_dir = os.environ.get("MG37_RESULTS",
                             os.path.dirname(os.path.abspath(__file__)))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    node = platform.node().split(".")[0]
    out_path = os.path.join(out_dir, f"rows/mg37_cone_window_{node}_{stamp}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sink = open(out_path, "w")

    def emit(row):
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    from mbirtorch.cone_beam import (_cone_horizontal_data,
                                     _cone_vertical_affine)

    model = build_model()
    args = model._view_batch_args()
    pf = model.projector_functions
    view_params = pf._view_params_per_dev[0]
    idx = model.full_indices_device()
    num_pixels = int(idx.shape[0])
    psf_radius = int(args["psf_radius"])
    h_tap_pad = 2 * psf_radius + 1
    num_tiles = num_pixels // TILE_P
    emit(dict(kind="header", cell=CELL, num_pixels=num_pixels,
              psf_radius=psf_radius, tile_p=TILE_P, node=node))
    print(f"mask {num_pixels} pixels, {num_tiles} tiles of {TILE_P}, "
          f"horizontal tap span +{h_tap_pad}")

    for ref in REF_VIEWS:
        for chunk in CHUNKS:
            vp = view_params[ref:ref + chunk]
            angles = vp[:, 0]
            z_shifts = vp[:, 1]
            n_p, _c, _w, _s, pixel_mag = call_with_args(
                _cone_horizontal_data, (idx, angles), args)
            m0, w_p_r, _z = call_with_args(
                _cone_vertical_affine, (pixel_mag, z_shifts), args)
            n_p = n_p.numpy()
            m0 = m0.numpy()
            w_p_r = w_p_r.numpy()
            wpr_lo, wpr_hi = float(w_p_r.min()), float(w_p_r.max())

            for B in BUCKETS:
                # The compound key at the chunk's FIRST view: coarse
                # W_p_r bucket, then channel center.
                t0 = time.perf_counter()
                edges = np.linspace(wpr_lo, wpr_hi, B + 1)
                bucket = np.clip(np.digitize(w_p_r[0], edges) - 1, 0, B - 1)
                order = np.lexsort((n_p[0], bucket))
                sort_ms = (time.perf_counter() - t0) * 1000
                n_p_s = n_p[:, order]
                m0_s = m0[:, order]
                wpr_s = w_p_r[:, order]
                bucket_s = bucket[order][:num_tiles * TILE_P]
                boundary_tiles = int((np.ptp(
                    bucket_s.reshape(num_tiles, TILE_P), axis=1) > 0).sum())

                # Per-VIEW flush sizing (the design's case): within-view
                # span, maxed over the chunk.  Union sizing (a per-chunk
                # flush) kept beside it for comparison.
                chan_win = per_view_tile_span([n_p_s], num_tiles) + h_tap_pad
                cmin_u, cmax_u = tile_min_max(n_p_s, num_tiles)
                chan_union = (cmax_u - cmin_u) + h_tap_pad
                # Vertical footprint pad per tile: one slice spans about
                # W_p_r rows; +2 covers the interpolation edges.
                _wmin, wmax_t = tile_min_max(wpr_s, num_tiles)
                v_pad = np.ceil(wmax_t) + 2

                for S in SLICE_TILES:
                    worst = None
                    for l in PROBE_SLICES:
                        l2 = min(l + S - 1, CELL[1] - 1)
                        rows_a = m0_s + wpr_s * l
                        rows_b = m0_s + wpr_s * l2
                        row_win = per_view_tile_span([rows_a, rows_b],
                                                     num_tiles) + v_pad
                        amin, amax = tile_min_max(rows_a, num_tiles)
                        bmin, bmax = tile_min_max(rows_b, num_tiles)
                        row_union = (np.maximum(amax, bmax)
                                     - np.minimum(amin, bmin)) + v_pad
                        fits = {f"rows_le_{r}": float((row_win <= r).mean())
                                for r in ROW_FITS}
                        fits.update({f"chan_le_{c}": float((chan_win <= c).mean())
                                     for c in CHAN_FITS})
                        both = {f"both_le_{r}x{c}":
                                float(((row_win <= r) & (chan_win <= c)).mean())
                                for r in ROW_FITS for c in CHAN_FITS}
                        row = dict(kind="combo", ref=ref, chunk=chunk,
                                   buckets=B, slice_tile=S, probe_slice=l,
                                   row_window=dist(row_win),
                                   chan_window=dist(chan_win),
                                   row_union=dist(row_union),
                                   chan_union=dist(chan_union),
                                   boundary_tiles=boundary_tiles,
                                   sort_ms=sort_ms, **fits, **both)
                        emit(row)
                        if worst is None or row_win.mean() > worst[0]:
                            worst = (row_win.mean(), row)
                    w = worst[1]
                    print(f'ref {ref:>3} chunk {chunk:>2} B {B:>2} S {S:>2} '
                          f'worst probe l={w["probe_slice"]:>4}: '
                          f'row p99 {w["row_window"]["p99"]:6.1f} '
                          f'max {w["row_window"]["max"]:6.1f}  '
                          f'chan p99 {w["chan_window"]["p99"]:5.1f}  '
                          f'fit32x16 {w["both_le_32x16"]*100:5.1f}%  '
                          f'fit16x8 {w["both_le_16x8"]*100:5.1f}%  '
                          f'(union row {w["row_union"]["p99"]:5.1f} '
                          f'chan {w["chan_union"]["p99"]:5.1f})')

    sink.close()
    print(f"rows: {out_path}")


if __name__ == "__main__":
    main()
