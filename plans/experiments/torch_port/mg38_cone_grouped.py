"""mg38 -- THE CONE PIXEL-BATCHED SPIKE: TWO-AXIS GROUPING (design note
`pfwd_segmented_design.md` section 9).

WHY THIS RUN EXISTS.  mg31's counters read the cone forward as
gather-dominated: the vertical fan re-reads the whole flat recon roughly once
per view, so a 52-view launch pulled 175 GB from DRAM.  Nothing inside one
view is worth reordering; the only prize is CROSS-VIEW amortization.  Section
9's design turns the kernel inside out to take it: a work unit holds a
(pixel tile x slice tile) block of the recon in registers and serves a whole
CHUNK of views, so the recon is read once per chunk instead of once per view.
mg37, the paper check, closed the feasibility question on the shipped
geometry builders: with 32 magnification buckets, a chunk of 8 or 16 and a
slice tile of 4 or 8, at least 99.8 percent of 32-pixel tiles have a detector
footprint fitting a 16-row x 8-channel window, and every swept combination
fits 32 x 16.  This spike builds that kernel inside the harness and measures
it against the shipped wrapper.  No library file is touched.

THE DESIGN, as implemented.

  * OUTSIDE the kernel, once per view chunk: the compound ordering of section
    9.  At the chunk's FIRST view the pixels are bucketed by W_p_r (the
    rows-per-slice slope) into 32 equal-width buckets and sorted by
    (bucket, channel center) with a stable two-pass sort.  The contract
    arrays are gathered into that one order for every view of the chunk, and
    an int32 permutation maps each sorted position back to its values row.
    One sort per chunk, not one per view -- the factor-of-chunk saving over
    the parallel forward's per-view sorts.

  * INSIDE the kernel, per work unit (32 pixels x S slices x one view
    chunk): the values tile is gathered through the permutation ONCE, before
    the view loop.  Per view the tile's window bases are recomputed from that
    view's contract -- the lowest detector row any voxel of the tile can
    reach and the lowest channel any pixel can reach -- and the tile's whole
    contribution is accumulated into a compile-time (rows x channels) patch,
    which is flushed with one atomic add per patch element.  Recomputing the
    base per view is what makes chunk drift free: rotation (and, on a helical
    scan, the per-view z shift) moves the window's POSITION between views,
    never its SIZE.

  * THE VERTICAL ARITHMETIC IS THE SHIPPED KERNEL'S, INCLUDING ITS FLOAT
    ORDER.  This is the one place where a plausible shortcut is wrong, and it
    was measured on CPU before this file was written.  The obvious inverted
    form evaluates the trapezoid on (m0 + W_p_r * slice) - row directly; at
    the 1024-class cell that difference of two numbers near 1000 loses enough
    of the float32 mantissa to move the answer 5.1e-5 relative from the
    shipped body -- a values-gate failure with no bug behind it.  So the
    kernel inverts the affine at the ROW exactly as `_cone_forward_kernel`
    does (k_m = (row - m0)/W_p_r, k_center = floor(k_m + 0.5),
    m_p = W_p_r * (k_center - k_m), then the trapezoid of m_p + W_p_r * k_off
    with the |k_off| <= bp_psf_radius window), which is one division per
    (pixel, window row) hoisted out of the slice loop.  The same CPU check
    then reads 1.5e-7.  The cone-angle divisor and the per-view z_offset are
    carried exactly as the shipped kernel carries them
    (v_det = pixel_mag * (delta_voxel_slice * (k - slice_center) + z_offset)):
    the test cell is non-helical, but a helical z shift is a per-view
    common-mode row displacement that the per-view flush absorbs as window
    position, so the arithmetic stays intact for it.

  * THE PER-TILE GUARD.  Each tile measures its actual row and channel spans
    at run time; a tile that does not fit its window takes a per-tap path
    that adds straight into the sinogram with no window at all.  That path
    uses the same weights in the same float order, so correctness never rests
    on the grouping -- only speed does.  One swept arm sets the windows below
    any real span, so every tile takes the fallback and the values gate
    covers it.

WHAT THIS RUN DOES NOT DECIDE.  It picks no default and ships nothing.  A
winning configuration goes to the composed re-gate; a kernel spike's win is
not the driver's (the recorded lesson).

THE EXPECTED SHAPE OF THE RESULT, stated before the run.  The full mask is
the domain: raster pixels sorted two ways give the compact tiles mg37
measured, and the values read should fall by about the chunk length.  The
subset points are expected to LOSE and are kept as the disengagement
evidence section 9's rider asks for: a thinned pixel set spreads each tile
over many channels, the span guard fires, and the per-tap fallback carries
the call at a cost above the shipped kernel's.

THE ARITHMETIC BILL is the number to watch beside the speedup, because this
design buys traffic with arithmetic and the cone forward already ran at 50.4
percent of arithmetic peak.  Per voxel the shipped kernel evaluates three
vertical taps and three horizontal ones, six operations of which three are
atomics.  The window path evaluates WINDOW_ROWS vertical weights (only about
three of them can be nonzero -- the window spans the whole TILE, while one
voxel reaches only its own two or three rows) and, amortized over the slice
tile, WINDOW_ROWS * WINDOW_CHANNELS / SLICE_TILE horizontal products.  That
is 32 operations per voxel at (16 rows, 8 channels, 8 slices), about 5x the
shipped kernel, and 160 at (32, 16, 4), about 27x.  The atomics move the
other way and not always favourably: the flush issues
WINDOW_ROWS * WINDOW_CHANNELS / (SLICE_TILE * 32) per voxel, which is 0.5 at
(16, 8, 8) -- six times fewer than the shipped three -- but 4 at (32, 16, 4),
which is MORE.  So the small window with the large slice tile is the arm
that can win on both counts, and the wide window with the small slice tile
is in the sweep to price the fit rate, not to win.  If the cheap arm does not
win, no configuration here will.

WHAT WAS VALIDATED BEFORE SUBMISSION, on CPU, and what was not.  The kernel
body itself was executed against a torch stand-in for triton.language --
real pointer arithmetic, real masks, real reductions -- on two small cone
cells across every swept configuration and three ladder points, and at the
1024-class cell on a run of real full-mask tiles.  It reproduced the shipped
torch body to 3.9e-7 on the window path and 6.8e-7 on the forced fallback,
with no out-of-bounds address on any live lane.  What that stand-in cannot
see is everything Triton-specific: shape unification across the two
branches, register pressure and spilling, and performance.  The compile
record printed at the end of the run is where spilling would show.

THE GATES.  Values at 1e-5 relative against the shipped wrapper at every
ladder point for every configuration; the CPU arithmetic preflight (a torch
twin of the kernel's exact arithmetic, both branches) at 1e-5 against the
torch body before any kernel runs.  There is NO baseline anchor constant in
this file: no cone forward launch time is recorded for this cell in a form
this harness can quote, so the shipped baseline is measured and reported and
the reviewer compares it against mg31's rows.

THE FALLBACK RATE is computed on the HOST, in torch, from the sorted contract
-- the same span test the kernel applies -- and never inside the timed
region.  An in-kernel counter was deliberately not built: the last one
under-counted and was recorded as a defect, and a counter that cannot be
verified is worse than none.

Run:
    <torch python> mg38_cone_grouped.py           on one GPU
    MG38_DRY=1 <python> mg38_cone_grouped.py      print the plan and stop
    MG38_SMOKE=1 <python> mg38_cone_grouped.py    tiny CPU plumbing pass

Configuration is by environment variable only; there is no command line.
    MG38_RESULTS=<dir>      where the jsonl and the ncu log go
    MG38_SMOKE=1 / MG38_DRY=1
    MG38_REPEATS=3          timed repeats per (configuration, ladder point)
    MG38_CONFIGS=a,b        a subset of the configurations, by name
    MG38_NCU=1              the counter attempt on the winner (default on)
"""

import json
import math
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
SMOKE = os.environ.get("MG38_SMOKE", "0") == "1"
DRY = os.environ.get("MG38_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (1024, 1008, 992)          # (views, detector rows, channels) -- mg31's
SMOKE_CELL = (8, 32, 20)          # 32 rows: a multiple of 16, so no padding
VIEW_BATCH = 64                   # divides both swept chunks; ~7 GB at the cell
VALUES_SEED = 20260819
NUM_PIXELS_EXPECTED = 771240      # recorded, not gated

WARMUP_REPEATS = 1
TIMED_REPEATS = max(1, int(os.environ.get("MG38_REPEATS", "3")))
VALUES_GATE_REL = 1e-5

#: The pixel ladder: the full mask (the design's domain) and the two strided
#: subsets, which are the disengagement evidence rather than a target.
PIXEL_LADDER = (1, 64, 128)

#: mg37's winning grouping constants, fixed here: 32 equal-width W_p_r buckets
#: as the sort's primary key, 32 pixels per tile.
BUCKETS = 32
BLOCK_P = 32

#: The swept configurations: (name, VIEW_CHUNK, SLICE_TILE, WINDOW_ROWS,
#: WINDOW_CHANNELS, num_warps, num_stages).  mg37's verdict sets the windows
#: (32 x 16 fits every swept combination, 16 x 8 fits 99.8 percent and is
#: about six times cheaper in arithmetic), the chunks (8 and 16 price nearly
#: the same windows, so the longer chunk's read amortization is close to
#: free), and the slice tiles (4 and 8).  The last arm forces the per-tile
#: fallback by setting both windows below any real span; it exists for the
#: values gate, not for the timing.
SWEEP_CONFIGS = (
    ("grp_r16c8_v8_s8",    8, 8, 16,  8, 4, 1),
    ("grp_r16c8_v16_s8",  16, 8, 16,  8, 4, 1),
    ("grp_r16c8_v8_s4",    8, 4, 16,  8, 4, 1),
    ("grp_r16c8_v16_s4",  16, 4, 16,  8, 4, 1),
    ("grp_r32c16_v8_s8",   8, 8, 32, 16, 4, 1),
    ("grp_r32c16_v16_s8", 16, 8, 32, 16, 4, 1),
    ("grp_r32c16_v8_s4",   8, 4, 32, 16, 4, 1),
    ("grp_r32c16_v16_s4", 16, 4, 32, 16, 4, 1),
    ("forced_fallback",    8, 8,  4,  4, 4, 1),
)
FORCED_FALLBACK = "forced_fallback"

#: The CPU/host arithmetic preflight: a torch twin of the kernel's exact
#: arithmetic, run on a small pixel subset at the run's own cell before any
#: kernel launches.  It is what catches an arithmetic divergence that the
#: Triton path would otherwise report as a mysterious gate failure.
PREFLIGHT_PIXELS = 2050           # deliberately NOT a multiple of BLOCK_P
PREFLIGHT_VIEWS = 4
PREFLIGHT_SLICE_TILES = 4         # probe tiles spread over the slice axis
PREFLIGHT_ARMS = ("grp_r32c16_v8_s8", FORCED_FALLBACK)

#: The host-side fallback estimate: the same span test the kernel applies,
#: evaluated over the first view chunk at a few probe slice tiles.
FALLBACK_PROBE_TILES = 5

# ── the counter attempt on the winner (mg33's machinery, one variant) ─────────
NCU_ENABLED = os.environ.get("MG38_NCU", "1") == "1"
NCU_LAUNCHES = 3
NCU_TIMEOUT_S = 600
NCU_PROBE_TIMEOUT_S = 180
METRICS_FULL = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sectors_op_red.sum",
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
    "MG38_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
NAME_COL = 20
# ──────────────────────────────────────────────────────────────────────────────


def cell():
    return SMOKE_CELL if SMOKE else CELL


def config_named(name):
    return next(c for c in SWEEP_CONFIGS if c[0] == name)


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


# ── GPU health (mg21b's sampler, as mg33) ─────────────────────────────────────
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


# ── the model (mg31's cone construction, as mg37) ─────────────────────────────
def build_model():
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
    """Call a geometry builder, filling its named parameters from the model's
    ``_view_batch_args()`` dict; missing names are an error.  mg37's helper,
    including the two names the args dict spells differently."""
    import inspect

    alias = {"num_rows": "num_recon_rows", "num_cols": "num_recon_cols"}
    params = list(inspect.signature(fn).parameters)[len(first_args):]
    missing = [p for p in params if alias.get(p, p) not in args]
    if missing:
        raise KeyError(f"{fn.__name__} needs {missing} not in args")
    return fn(*first_args, **{p: args[alias.get(p, p)] for p in params})


def _seeded_values(torch_module, model, num_pixels, num_slices):
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(VALUES_SEED)
    block = torch_module.rand((int(num_pixels), int(num_slices)),
                              generator=generator,
                              dtype=torch_module.float32)
    return block.to(model.torch_device)


# ── the contract, the compound sort, and the gather ───────────────────────────
def build_contract(pixel_indices, view_params_batch, args):
    """The two shipped builders, hoisted exactly as every cone body hoists
    them: the horizontal fan contract and the vertical affine pair.  Returns
    the seven per-(view, pixel) arrays the kernel reads plus the per-view
    z_offset -- the same eight the shipped wrapper builds."""
    from mbirtorch.cone_beam import (_cone_horizontal_data,
                                     _cone_vertical_affine)

    angles = view_params_batch[:, 0]
    z_shifts = view_params_batch[:, 1]
    n_p, centers, w_p_c, weight_scale, pixel_mag = call_with_args(
        _cone_horizontal_data, (pixel_indices, angles), args)
    m0, w_p_r, z_offset = call_with_args(
        _cone_vertical_affine, (pixel_mag, z_shifts), args)
    return (n_p, centers, w_p_c, weight_scale, m0, w_p_r, pixel_mag), z_offset


def compound_sort(w_p_r, n_p, view_chunk, buckets=BUCKETS):
    """Design note section 9's compound ordering, one sort per view CHUNK.

    The primary key is the rows-per-slice slope W_p_r placed in ``buckets``
    equal-width buckets (coarse, so that the secondary key still has room);
    the secondary key is the channel center.  Both are read at the chunk's
    FIRST view and the resulting order serves every view of the chunk -- the
    whole point of the chunk, against the parallel route's per-view sorts.

    Returns an int32 (num_chunks, num_pixels) permutation whose entry [c, i]
    is the values row of the i-th pixel in chunk c's order.
    """
    import torch

    num_views, num_pixels = w_p_r.shape
    num_chunks = -(-num_views // view_chunk)
    perm = torch.empty((num_chunks, num_pixels), dtype=torch.int32,
                       device=w_p_r.device)
    for chunk in range(num_chunks):
        slope = w_p_r[chunk * view_chunk]
        lo = slope.min()
        width = (slope.max() - lo) / float(buckets)
        # A degenerate spread (every pixel in one bucket) is legal: the sort
        # then reduces to the channel-center sort, which is the parallel
        # route's ordering.
        width = torch.clamp(width, min=torch.finfo(width.dtype).tiny)
        bucket = torch.clamp(((slope - lo) / width).floor(), 0.0,
                             float(buckets - 1))
        first = torch.argsort(n_p[chunk * view_chunk], stable=True)
        order = first[torch.argsort(bucket[first], stable=True)]
        perm[chunk] = order.to(torch.int32)
    return perm


def gather_contract(arrays, perm, view_chunk):
    """Gather the per-(view, pixel) contract into its chunk's order.  The
    index is one expanded view of the chunk's permutation, so this costs the
    gathered copies and nothing else."""
    import torch

    num_views = int(arrays[0].shape[0])
    out = []
    for tensor in arrays:
        gathered = torch.empty_like(tensor)
        for chunk in range(int(perm.shape[0])):
            v0 = chunk * view_chunk
            v1 = min(v0 + view_chunk, num_views)
            index = perm[chunk].to(torch.int64).unsqueeze(0).expand(v1 - v0, -1)
            gathered[v0:v1] = torch.gather(tensor[v0:v1], 1, index)
        out.append(gathered.contiguous())
    return out


def row_tap_radius(w_p_r):
    """How many detector rows on each side of a voxel's row center can carry
    weight, from the trapezoid's own support: the weight is nonzero only
    where |m - row| < (W_p_r + 1)/2, and |m - row_center| <= 1/2, so the
    reach is the largest integer below (W_p_r + 1)/2 + 1/2.  Used only by the
    per-tap fallback, which has no window to bound it."""
    slope_max = float(w_p_r.max())
    return max(1, int(math.ceil((slope_max + 1.0) / 2.0 + 0.5)) - 1)


# ── the variant kernel ────────────────────────────────────────────────────────
_KERNEL_HOLDER = {}


def grouped_kernel():
    """The two-axis grouped cone forward, defined on first use so the dry run
    and the CPU smoke import no triton."""
    if "kernel" in _KERNEL_HOLDER:
        return _KERNEL_HOLDER["kernel"]
    from mbirtorch.triton_cone import (_jit, _tap_range, _tl_abs, _tl_floor,
                                       _tl_sqrt, tl)

    @_jit
    def _cone_grouped_kernel(n_p_ptr, centers_ptr, w_p_c_ptr, weight_scale_ptr,
                             m0_ptr, w_p_r_ptr, pixel_mag_ptr, z_offset_ptr,
                             perm_ptr, values_ptr, out_ptr,
                             num_views, num_pixels, num_channels, num_rows,
                             band_len, slice_start, out_view_stride,
                             out_row_stride, delta_voxel_slice, slice_center,
                             inv_sdd, k_center_lo, k_center_hi,
                             row_clamp_lo, row_clamp_hi,
                             VIEW_CHUNK: tl.constexpr,
                             SLICE_TILE: tl.constexpr,
                             WINDOW_ROWS: tl.constexpr,
                             WINDOW_CHANNELS: tl.constexpr,
                             PSF_RADIUS: tl.constexpr,
                             BP_PSF_RADIUS: tl.constexpr,
                             ROW_TAP_RADIUS: tl.constexpr,
                             BLOCK_P: tl.constexpr):
        """One program per (pixel tile, slice tile, view chunk).

        The shipped kernel's grid is (pixel block, detector row chunk, view):
        it walks detector rows and gathers the voxels that project onto each,
        so it reads the recon once per view.  This one walks VOXELS and
        scatters, which lets it hold its (pixel tile x slice tile) block of
        the recon in registers across a chunk of views -- the read
        amortization section 9 is after.  The cost of turning the kernel
        around is that the output becomes the moving target, and the two-axis
        grouping is what keeps it small: the pixels arrive sorted by
        (magnification bucket, channel center), so one tile's voxels land in
        a small patch of the detector, the patch accumulates in registers,
        and it is flushed once per view.

        The (WINDOW_CHANNELS x WINDOW_ROWS) patch is never materialized as
        one register tile.  The two axes separate cleanly -- the horizontal
        weight depends on the pixel alone and the vertical weight on the
        (pixel, slice, row) -- so the tile is first reduced over its slices
        into a (pixel, window row) block, and then one window channel at a
        time is reduced over the pixels and flushed.  That keeps the largest
        live tile at (BLOCK_P, WINDOW_ROWS), needs no tl.dot (and so no
        opinion about tensor-core rounding, which the 1e-5 gate would have),
        and puts each flush on one run of consecutive addresses, because the
        output is channel-major (view, channel, row) with the row axis
        contiguous.

        Padded pixel lanes carry sentinels into the window reductions so they
        cannot widen a window, load zero values so they add nothing, and are
        masked out of every atomic (the poison-the-padding rule).  A view
        chunk past the batch's end clamps its view index for every address
        and masks its atomics, so a batch shorter than the chunk is safe.
        """
        p_offs = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
        s_ar = tl.arange(0, SLICE_TILE)
        k_base = tl.program_id(1) * SLICE_TILE
        k_offs = k_base + s_ar
        chunk = tl.program_id(2)
        v0 = chunk * VIEW_CHUNK
        p_mask = p_offs < num_pixels
        k_mask = k_offs < band_len
        tile_mask = p_mask[:, None] & k_mask[None, :]

        # THE ONE VALUES READ.  The chunk's permutation maps this tile's
        # sorted positions back to their recon rows; the tile is gathered
        # here, before the view loop, and every view of the chunk is served
        # from these registers.
        row_idx = tl.load(perm_ptr + chunk.to(tl.int64) * num_pixels + p_offs,
                          mask=p_mask, other=0)
        vals = tl.load(values_ptr + row_idx.to(tl.int64)[:, None] * band_len
                       + k_offs[None, :], mask=tile_mask, other=0.0)

        for dv in range(VIEW_CHUNK):
            v = v0 + dv
            v_ok = v < num_views
            v_safe = tl.minimum(v, num_views - 1)
            pix_base = v_safe.to(tl.int64) * num_pixels + p_offs
            n_p = tl.load(n_p_ptr + pix_base, mask=p_mask, other=0.0)
            centers = tl.load(centers_ptr + pix_base, mask=p_mask, other=0)
            w_p_c = tl.load(w_p_c_ptr + pix_base, mask=p_mask, other=0.0)
            weight_scale = tl.load(weight_scale_ptr + pix_base, mask=p_mask,
                                   other=0.0)
            m0 = tl.load(m0_ptr + pix_base, mask=p_mask, other=0.0)
            # A padded lane DIVIDES by its slope, so its filler is 1.0 rather
            # than 0.0: 0/0 is a NaN and a NaN survives a clamp into an
            # undefined float-to-int conversion (the shipped kernel's reason).
            slope = tl.load(w_p_r_ptr + pix_base, mask=p_mask, other=1.0)
            pixel_mag = tl.load(pixel_mag_ptr + pix_base, mask=p_mask,
                                other=0.0)
            # The per-view z offset of _cone_vertical_affine, carried exactly
            # as the shipped kernel carries it.  It is zero on this cell, but
            # a helical scan enters the geometry through it and through m0,
            # both of which are per-view common-mode row shifts that the
            # per-view window base absorbs; keeping the term intact is what
            # makes the route cover helical scans.
            z_offset = tl.load(z_offset_ptr + v_safe)
            half = (slope + 1.0) / 2.0
            l_max_r = tl.minimum(slope, 1.0)
            out_view_ptr = out_ptr + v_safe.to(tl.int64) * out_view_stride

            # THE WINDOW BASES, recomputed for this view.  The row a voxel
            # reaches is the sanctioned affine m = m0 + W_p_r * slice; the
            # rows that can carry weight are those within (W_p_r + 1)/2 of
            # it, so the tile's row window runs from the floor of the least
            # such bound to just past the greatest.  The clamp keeps a
            # degenerate geometry's row out of the undefined float-to-int
            # range; it can only shrink a window towards the detector, and
            # every row it excludes was outside the panel anyway.
            m = m0[:, None] + slope[:, None] * (slice_start
                                                + k_offs).to(tl.float32)[None, :]
            m = tl.minimum(tl.maximum(m, row_clamp_lo), row_clamp_hi)
            big = 3.0e30
            row_lo = _tl_floor(tl.min(tl.where(tile_mask, m - half[:, None],
                                               big))).to(tl.int32)
            row_hi = _tl_floor(tl.max(tl.where(tile_mask, m + half[:, None],
                                               -big))).to(tl.int32) + 1
            big_i = 2147483647
            c_lo = tl.min(tl.where(p_mask, centers, big_i)) - PSF_RADIUS
            c_hi = tl.max(tl.where(p_mask, centers, -big_i)) + PSF_RADIUS
            fits = (((row_hi - row_lo + 1) <= WINDOW_ROWS)
                    & ((c_hi - c_lo + 1) <= WINDOW_CHANNELS))

            if fits:
                # ── THE WINDOW PATH ──────────────────────────────────────
                jr = tl.arange(0, WINDOW_ROWS)
                rows_i = row_lo + jr
                row_ok = (rows_i >= 0) & (rows_i < num_rows)
                # The shipped kernel's float order, and the reason for the
                # division: inverting the affine AT THE ROW keeps the
                # trapezoid's argument in the same catastrophic-cancellation
                # regime the shipped kernel is already in, so the two agree
                # to float rounding.  Hoisted out of the slice loop: k_center
                # and m_p depend on the row and the pixel, not the slice.
                k_m = (rows_i.to(tl.float32)[None, :] - m0[:, None]) \
                    / slope[:, None]
                k_c = _tl_floor(k_m + 0.5)
                k_c = tl.minimum(tl.maximum(k_c, k_center_lo), k_center_hi)
                m_p = slope[:, None] * (k_c - k_m)

                acc = tl.zeros((BLOCK_P, WINDOW_ROWS), dtype=tl.float32)
                for si in _tap_range(0, SLICE_TILE):
                    k_g = (slice_start + k_base + si).to(tl.float32)
                    # This tile's values column, taken from the registers the
                    # gather above filled.  Out-of-band columns loaded zero,
                    # so they contribute nothing here.
                    vals_s = tl.sum(tl.where(s_ar[None, :] == si, vals, 0.0),
                                    axis=1)
                    v_det = pixel_mag * (delta_voxel_slice
                                         * (k_g - slice_center) + z_offset)
                    inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd)
                                           * (v_det * inv_sdd))
                    u_s = vals_s * inv_cos_phi
                    k_off = k_g - k_c
                    w_row = tl.maximum(
                        half[:, None] - _tl_abs(m_p + slope[:, None] * k_off),
                        0.0)
                    w_row = tl.minimum(w_row, l_max_r[:, None])
                    # The shipped kernel reaches only bp_psf_radius slices
                    # from a row's own center; outside that the trapezoid is
                    # zero by construction of bp_psf_radius, and this mask
                    # makes the two enumerations identical rather than merely
                    # equal in exact arithmetic.
                    w_row = tl.where(_tl_abs(k_off) <= BP_PSF_RADIUS, w_row,
                                     0.0)
                    acc = acc + w_row * u_s[:, None]
                acc = tl.where(p_mask[:, None] & row_ok[None, :], acc, 0.0)

                # ── THE FLUSH: one atomic add per patch element ───────────
                # One window channel at a time, so the (channels x rows)
                # patch never has to exist as a register tile: each column is
                # the tile's pixels summed against that channel's trapezoid
                # weight, and it lands as one contiguous run of atomics.
                row_addr = tl.minimum(tl.maximum(rows_i, 0), num_rows - 1)
                for jc in _tap_range(0, WINDOW_CHANNELS):
                    chan = c_lo + jc
                    w_chan = tl.maximum(
                        (w_p_c + 1.0) / 2.0
                        - _tl_abs(n_p - chan.to(tl.float32)), 0.0)
                    w_chan = tl.minimum(w_chan, tl.minimum(w_p_c, 1.0)) \
                        * weight_scale
                    w_chan = tl.where(p_mask, w_chan, 0.0)
                    col = tl.sum(w_chan[:, None] * acc, axis=0)
                    chan_addr = tl.minimum(tl.maximum(chan, 0),
                                           num_channels - 1)
                    # Lanes past the tile's real channel span carry zero
                    # weight AND issue no atomic, so the add count is the
                    # span's rather than the window's.  The span test is
                    # written with the tensor on the LEFT: jc is the loop's
                    # compile-time index, and a comparison with that on the
                    # left is the constexpr's operator rather than the
                    # tensor's.
                    tl.atomic_add(out_view_ptr
                                  + chan_addr.to(tl.int64) * out_row_stride
                                  + row_addr, col,
                                  mask=(v_ok & (chan >= 0)
                                        & (chan < num_channels)
                                        & ((c_hi - c_lo) >= jc) & row_ok))
            else:
                # ── THE PER-TILE FALLBACK ────────────────────────────────
                # No window: every (voxel, row tap, channel tap) contribution
                # goes straight into the sinogram.  The weights and their
                # float order are the window path's, so a tile that lands
                # here is slower and never different.
                for si in _tap_range(0, SLICE_TILE):
                    k_g = (slice_start + k_base + si).to(tl.float32)
                    k_ok = (k_base + si) < band_len
                    vals_s = tl.sum(tl.where(s_ar[None, :] == si, vals, 0.0),
                                    axis=1)
                    v_det = pixel_mag * (delta_voxel_slice
                                         * (k_g - slice_center) + z_offset)
                    inv_cos_phi = _tl_sqrt(1.0 + (v_det * inv_sdd)
                                           * (v_det * inv_sdd))
                    u_s = vals_s * inv_cos_phi
                    m_one = tl.minimum(tl.maximum(m0 + slope * k_g,
                                                  row_clamp_lo), row_clamp_hi)
                    m_ctr = _tl_floor(m_one + 0.5)
                    for tr in _tap_range(0, 2 * ROW_TAP_RADIUS + 1):
                        r_f = m_ctr + (tr - ROW_TAP_RADIUS)
                        r_i = r_f.to(tl.int32)
                        k_m1 = (r_f - m0) / slope
                        k_c1 = _tl_floor(k_m1 + 0.5)
                        k_c1 = tl.minimum(tl.maximum(k_c1, k_center_lo),
                                          k_center_hi)
                        m_p1 = slope * (k_c1 - k_m1)
                        k_off1 = k_g - k_c1
                        # w_tap, not w_row: Triton's structured branching
                        # wants one shape per name across the two branches,
                        # and the window path's w_row is
                        # (BLOCK_P, WINDOW_ROWS) while this one is (BLOCK_P,).
                        # mg33 recorded a compilation failure on exactly that
                        # collision.
                        w_tap = tl.maximum(
                            half - _tl_abs(m_p1 + slope * k_off1), 0.0)
                        w_tap = tl.minimum(w_tap, l_max_r)
                        row_live = ((_tl_abs(k_off1) <= BP_PSF_RADIUS)
                                    & (r_i >= 0) & (r_i < num_rows)
                                    & p_mask & k_ok)
                        w_tap = tl.where(row_live, w_tap, 0.0)
                        r_addr = tl.minimum(tl.maximum(r_i, 0), num_rows - 1)
                        for tc in _tap_range(0, 2 * PSF_RADIUS + 1):
                            n_tap = centers + (tc - PSF_RADIUS)
                            w_chan = tl.maximum(
                                (w_p_c + 1.0) / 2.0
                                - _tl_abs(n_p - n_tap.to(tl.float32)), 0.0)
                            w_chan = tl.minimum(w_chan,
                                                tl.minimum(w_p_c, 1.0)) \
                                * weight_scale
                            n_addr = tl.minimum(tl.maximum(n_tap, 0),
                                                num_channels - 1)
                            tl.atomic_add(out_view_ptr
                                          + n_addr.to(tl.int64) * out_row_stride
                                          + r_addr,
                                          w_tap * w_chan * u_s,
                                          mask=(v_ok & row_live
                                                & (n_tap >= 0)
                                                & (n_tap < num_channels)))

    _KERNEL_HOLDER["kernel"] = _cone_grouped_kernel
    return _cone_grouped_kernel


def grouped_prep(values, pixel_indices, view_params_batch, args, config):
    """Everything the launch does before the kernel: the two builders, the
    compound sort, and the gather.  Returned so the host-side fallback
    estimate and the timed launch can share one implementation, and so the
    sort's own milliseconds are read separately from the launch's."""
    import torch

    (_name, view_chunk, _slice_tile, _wr, _wc, _warps, _stages) = config
    arrays, z_offset = build_contract(pixel_indices, view_params_batch, args)
    n_p, _centers, _w_p_c, _weight_scale, _m0, w_p_r, _pixel_mag = arrays
    cuda = values.is_cuda
    if cuda:
        torch.cuda.synchronize(values.device)
    start = time.perf_counter()
    perm = compound_sort(w_p_r, n_p, view_chunk)
    if cuda:
        torch.cuda.synchronize(values.device)
    sort_ms = (time.perf_counter() - start) * 1e3
    start = time.perf_counter()
    sorted_arrays = gather_contract(arrays, perm, view_chunk)
    if cuda:
        torch.cuda.synchronize(values.device)
    gather_ms = (time.perf_counter() - start) * 1e3
    return dict(arrays=sorted_arrays, z_offset=z_offset.contiguous(),
                perm=perm, sort_ms=sort_ms, gather_ms=gather_ms,
                row_tap_radius=row_tap_radius(w_p_r))


def grouped_launch(values, pixel_indices, view_params_batch, args, config):
    """The variant's whole call: builders, sort, gather, launch.

    Timed as a unit, exactly as mg33 timed its sorted parallel variant, so
    the comparison against the shipped wrapper is like for like -- that
    wrapper also builds its contract inside every call.  ``sort_ms`` and
    ``gather_ms`` are stashed on the function so the caller can report the
    ordering's own share; in production the orderings depend only on the
    (pixel set, view batch) pair and would memoize through the ``plan`` slot.
    """
    import contextlib

    import torch

    from mbirtorch._utils import padded_kernel_width
    from mbirtorch.projectors import compile_serialized
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    (_name, view_chunk, slice_tile, window_rows, window_channels, num_warps,
     num_stages) = config
    prep = grouped_prep(values, pixel_indices, view_params_batch, args, config)
    grouped_launch.last_sort_ms = prep["sort_ms"]
    grouped_launch.last_gather_ms = prep["gather_ms"]
    (n_p, centers, w_p_c, weight_scale, m0, w_p_r, pixel_mag) = prep["arrays"]

    num_views, num_pixels = n_p.shape
    band_len = int(values.shape[1])
    # The spike only ever calls unbanded, so slice_start is passed as 0 below.
    # The kernel keeps slice_start as an argument and anchors its z chain on
    # the full slice count, exactly as the shipped body does, so a banded
    # call would need only a caller that passes it -- but nothing here
    # exercises that, and the assert says so rather than letting a future
    # banded caller find out at the values gate.
    assert band_len == int(args["num_slices"]), (band_len, args["num_slices"])
    num_channels = int(args["num_channels"])
    num_rows_r = int(args["num_rows_r"])
    launch_rows = padded_kernel_width(num_rows_r)
    values = values.contiguous()
    out = torch.zeros((num_views, num_channels, launch_rows),
                      dtype=torch.float32, device=values.device)
    inv_sdd = (0.0 if math.isinf(float(args["source_detector_dist"]))
               else 1.0 / float(args["source_detector_dist"]))
    bp = int(args["bp_psf_radius"])
    tap_r = int(prep["row_tap_radius"])
    grid = (-(-num_pixels // BLOCK_P), -(-band_len // slice_tile),
            -(-num_views // view_chunk))
    launch_key = ("mg38_grouped", values.device.index, int(args["psf_radius"]),
                  bp, tap_r, view_chunk, slice_tile, window_rows,
                  window_channels, num_warps, num_stages, int(num_views),
                  int(num_pixels), num_channels, launch_rows, band_len)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    kernel = grouped_kernel()
    with torch.cuda.device(values.device), guard:
        kernel[grid](
            n_p, centers, w_p_c, weight_scale, m0, w_p_r, pixel_mag,
            prep["z_offset"], prep["perm"], values, out,
            int(num_views), int(num_pixels), num_channels, num_rows_r,
            band_len, 0, num_channels * launch_rows, launch_rows,
            float(args["delta_voxel_slice"]),
            (int(args["num_slices"]) - 1) / 2.0, inv_sdd,
            float(-bp - 1), float(band_len + bp),
            float(-2 - tap_r), float(num_rows_r + 1 + tap_r),
            VIEW_CHUNK=view_chunk, SLICE_TILE=slice_tile,
            WINDOW_ROWS=window_rows, WINDOW_CHANNELS=window_channels,
            PSF_RADIUS=int(args["psf_radius"]), BP_PSF_RADIUS=bp,
            ROW_TAP_RADIUS=tap_r, BLOCK_P=BLOCK_P,
            num_warps=num_warps, num_stages=num_stages)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    if launch_rows == num_rows_r:
        return out.permute(0, 2, 1)
    return out[:, :, :num_rows_r].permute(0, 2, 1)


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


# ── the host-side fallback estimate (never inside the timed region) ───────────
def fallback_fraction(prep, args, config, band_len, probes=FALLBACK_PROBE_TILES):
    """The share of tiles whose spans exceed the window, computed in torch
    from the sorted contract with the same test the kernel applies.

    Measured over the first view chunk at a few probe slice tiles spread over
    the slice axis, because the affine is linear in the slice index and the
    spread is widest at the ends.  It is an estimate of the kernel's own
    branch rate, not a count of it: no in-kernel counter is shipped here, on
    the recorded lesson that an unverifiable counter is worse than none.
    """
    import torch

    (_name, view_chunk, slice_tile, window_rows, window_channels, _w,
     _s) = config
    (n_p, centers, _w_p_c, _ws, m0, w_p_r, _pm) = prep["arrays"]
    num_views, num_pixels = n_p.shape
    views = min(view_chunk, num_views)
    tiles = num_pixels // BLOCK_P            # whole tiles only; the ragged
    if tiles == 0:                           # last tile is reported separately
        return dict(tiles=0, fraction=None)
    psf = int(args["psf_radius"])
    num_tiles_k = max(1, -(-band_len // slice_tile))
    probe_ids = sorted({int(round(i * (num_tiles_k - 1) / max(1, probes - 1)))
                        for i in range(probes)})
    slope = w_p_r[:views, :tiles * BLOCK_P].reshape(views, tiles, BLOCK_P)
    base = m0[:views, :tiles * BLOCK_P].reshape(views, tiles, BLOCK_P)
    half = (slope + 1.0) / 2.0
    ctr = centers[:views, :tiles * BLOCK_P].reshape(views, tiles, BLOCK_P)
    chan_span = (ctr.amax(dim=2) - ctr.amin(dim=2)) + 2 * psf + 1
    tap_r = int(prep["row_tap_radius"])
    clamp = (-2.0 - tap_r, float(args["num_rows_r"]) + 1.0 + tap_r)
    over = 0
    total = 0
    for tile_k in probe_ids:
        k_lo = float(tile_k * slice_tile)
        k_hi = float(min(tile_k * slice_tile + slice_tile, band_len) - 1)
        # The affine is monotone in the slice index, so the tile's row extent
        # sits at its two endpoint slices.  The clamp is the kernel's, kept
        # here so the estimate answers the same question the kernel asks.
        m_lo = (base + slope * k_lo).clamp(clamp[0], clamp[1])
        m_hi = (base + slope * k_hi).clamp(clamp[0], clamp[1])
        low = torch.minimum(m_lo, m_hi) - half
        high = torch.maximum(m_lo, m_hi) + half
        row_lo = torch.floor(low.amin(dim=2))
        row_hi = torch.floor(high.amax(dim=2)) + 1
        row_span = (row_hi - row_lo + 1)
        bad = (row_span > window_rows) | (chan_span > window_channels)
        over += int(bad.sum())
        total += int(bad.numel())
    return dict(tiles=tiles, views=views, probe_slice_tiles=probe_ids,
                fraction=(over / total if total else None), checked=total)


# ── the CPU arithmetic preflight: a torch twin of the kernel's arithmetic ─────
def grouped_twin(values, prep, args, config, slice_tiles, band_len):
    """A torch mirror of the kernel's arithmetic, both branches, used only by
    the preflight.  It exists because the one real hazard in this design is
    arithmetic, not plumbing: the inverted decomposition can be written in a
    float order that costs 5e-5 relative against the shipped body at this
    cell, and the twin is what proves the order this kernel uses does not.

    Accumulates only the given slice tiles, so the caller can compare it
    against the shipped torch body run on values zeroed outside them.
    Returns the block and a count of which branch each tile took, so the
    preflight can show that both were actually exercised rather than assumed.
    """
    import torch

    (_name, view_chunk, slice_tile, window_rows, window_channels, _w,
     _s) = config
    (n_p, centers, w_p_c, weight_scale, m0, w_p_r, pixel_mag) = prep["arrays"]
    z_offset = prep["z_offset"]
    perm = prep["perm"]
    num_views, num_pixels = n_p.shape
    tiles = -(-num_pixels // BLOCK_P)
    padded = tiles * BLOCK_P
    psf = int(args["psf_radius"])
    bp = int(args["bp_psf_radius"])
    tap_r = int(prep["row_tap_radius"])
    num_rows = int(args["num_rows_r"])
    num_channels = int(args["num_channels"])
    slice_center = (int(args["num_slices"]) - 1) / 2.0
    dvs = float(args["delta_voxel_slice"])
    inv_sdd = (0.0 if math.isinf(float(args["source_detector_dist"]))
               else 1.0 / float(args["source_detector_dist"]))
    row_clamp = (-2.0 - tap_r, num_rows + 1.0 + tap_r)
    k_clamp = (float(-bp - 1), float(band_len + bp))
    dev = values.device
    big = 3.0e30

    def pad(tensor, filler):
        if padded == num_pixels:
            return tensor.reshape(num_views, tiles, BLOCK_P)
        extra = torch.full((num_views, padded - num_pixels), filler,
                           dtype=tensor.dtype, device=dev)
        return torch.cat([tensor, extra], dim=1).reshape(num_views, tiles,
                                                         BLOCK_P)

    p_mask = pad(torch.ones((num_views, num_pixels), dtype=torch.bool,
                            device=dev), False)
    n_p_t, ctr_t = pad(n_p, 0.0), pad(centers, 0)
    wpc_t, ws_t = pad(w_p_c, 0.0), pad(weight_scale, 0.0)
    m0_t, slope_t = pad(m0, 0.0), pad(w_p_r, 1.0)
    mag_t = pad(pixel_mag, 0.0)
    out = torch.zeros((num_views, num_rows, num_channels),
                      dtype=torch.float32, device=dev)
    half_t = (slope_t + 1.0) / 2.0
    lmax_t = slope_t.clamp(max=1.0)
    jr = torch.arange(window_rows, dtype=torch.float32, device=dev)
    took = dict(window=0, fallback=0)

    for view in range(num_views):
        chunk = min(view // view_chunk, int(perm.shape[0]) - 1)
        order = perm[chunk].to(torch.int64)
        vals_sorted = values.index_select(0, order)
        vals_sorted = torch.cat(
            [vals_sorted,
             torch.zeros((padded - num_pixels, band_len), dtype=torch.float32,
                         device=dev)], dim=0).reshape(tiles, BLOCK_P, band_len)
        pm = p_mask[view]
        m0_v, slope_v = m0_t[view], slope_t[view]
        half_v, lmax_v = half_t[view], lmax_t[view]
        mag_v, n_p_v, ctr_v = mag_t[view], n_p_t[view], ctr_t[view]
        wpc_v, ws_v = wpc_t[view], ws_t[view]
        z_v = float(z_offset[view])
        for tile_k in slice_tiles:
            k_ids = [k for k in range(tile_k * slice_tile,
                                      min(tile_k * slice_tile + slice_tile,
                                          band_len))]
            k_g = torch.tensor([float(k) for k in k_ids], dtype=torch.float32,
                               device=dev)
            m = (m0_v[:, :, None] + slope_v[:, :, None] * k_g[None, None, :]
                 ).clamp(row_clamp[0], row_clamp[1])
            live = pm[:, :, None].expand_as(m)
            low = torch.where(live, m - half_v[:, :, None],
                              torch.full_like(m, big))
            high = torch.where(live, m + half_v[:, :, None],
                               torch.full_like(m, -big))
            row_lo = torch.floor(low.amin(dim=(1, 2)))
            row_hi = torch.floor(high.amax(dim=(1, 2))) + 1
            c_lo = torch.where(pm, ctr_v, torch.full_like(ctr_v, 2 ** 30)
                               ).amin(dim=1) - psf
            c_hi = torch.where(pm, ctr_v, torch.full_like(ctr_v, -(2 ** 30))
                               ).amax(dim=1) + psf
            fits = ((row_hi - row_lo + 1) <= window_rows) \
                & ((c_hi - c_lo + 1) <= window_channels)
            block = vals_sorted[:, :, k_ids[0]:k_ids[-1] + 1]
            for tile in range(tiles):
                u = []
                for pos, k in enumerate(k_ids):
                    v_det = mag_v[tile] * (dvs * (float(k) - slice_center)
                                           + z_v)
                    icp = torch.sqrt(1.0 + (v_det * inv_sdd) ** 2)
                    u.append(block[tile, :, pos] * icp)
                took["window" if bool(fits[tile]) else "fallback"] += 1
                if bool(fits[tile]):
                    rows = row_lo[tile] + jr
                    row_i = rows.to(torch.int64)
                    row_ok = (row_i >= 0) & (row_i < num_rows)
                    k_m = (rows[None, :] - m0_v[tile][:, None]) \
                        / slope_v[tile][:, None]
                    k_c = torch.floor(k_m + 0.5).clamp(k_clamp[0], k_clamp[1])
                    m_p = slope_v[tile][:, None] * (k_c - k_m)
                    acc = torch.zeros((BLOCK_P, window_rows),
                                      dtype=torch.float32, device=dev)
                    for pos, k in enumerate(k_ids):
                        k_off = float(k) - k_c
                        w = torch.clamp(
                            half_v[tile][:, None]
                            - (m_p + slope_v[tile][:, None] * k_off).abs(),
                            min=0.0)
                        w = torch.minimum(w, lmax_v[tile][:, None])
                        w = torch.where(k_off.abs() <= bp, w,
                                        torch.zeros((), device=dev))
                        acc = acc + w * u[pos][:, None]
                    acc = torch.where(pm[tile][:, None] & row_ok[None, :], acc,
                                      torch.zeros((), device=dev))
                    for jc in range(window_channels):
                        chan = int(c_lo[tile]) + jc
                        if chan < 0 or chan >= num_channels \
                                or jc > int(c_hi[tile] - c_lo[tile]):
                            continue
                        w_chan = torch.clamp(
                            (wpc_v[tile] + 1.0) / 2.0
                            - (n_p_v[tile] - float(chan)).abs(), min=0.0)
                        w_chan = torch.minimum(w_chan,
                                               wpc_v[tile].clamp(max=1.0)) \
                            * ws_v[tile]
                        w_chan = torch.where(pm[tile], w_chan,
                                             torch.zeros((), device=dev))
                        col = (w_chan[:, None] * acc).sum(dim=0)
                        keep = row_ok
                        out[view].index_put_(
                            (row_i.clamp(0, num_rows - 1)[keep],
                             torch.full((int(keep.sum()),), chan,
                                        dtype=torch.int64, device=dev)),
                            col[keep], accumulate=True)
                else:
                    for pos, k in enumerate(k_ids):
                        m_one = (m0_v[tile] + slope_v[tile] * float(k)
                                 ).clamp(row_clamp[0], row_clamp[1])
                        m_ctr = torch.floor(m_one + 0.5)
                        for tr in range(-tap_r, tap_r + 1):
                            r_f = m_ctr + tr
                            r_i = r_f.to(torch.int64)
                            k_m = (r_f - m0_v[tile]) / slope_v[tile]
                            k_c = torch.floor(k_m + 0.5).clamp(k_clamp[0],
                                                               k_clamp[1])
                            m_p = slope_v[tile] * (k_c - k_m)
                            k_off = float(k) - k_c
                            w = torch.clamp(
                                half_v[tile]
                                - (m_p + slope_v[tile] * k_off).abs(), min=0.0)
                            w = torch.minimum(w, lmax_v[tile])
                            live_r = (k_off.abs() <= bp) & (r_i >= 0) \
                                & (r_i < num_rows) & pm[tile]
                            w = torch.where(live_r, w,
                                            torch.zeros((), device=dev))
                            for tc in range(-psf, psf + 1):
                                n_tap = ctr_v[tile] + tc
                                w_chan = torch.clamp(
                                    (wpc_v[tile] + 1.0) / 2.0
                                    - (n_p_v[tile] - n_tap.to(torch.float32)
                                       ).abs(), min=0.0)
                                w_chan = torch.minimum(
                                    w_chan, wpc_v[tile].clamp(max=1.0)) \
                                    * ws_v[tile]
                                keep = live_r & (n_tap >= 0) \
                                    & (n_tap < num_channels)
                                if not bool(keep.any()):
                                    continue
                                out[view].index_put_(
                                    (r_i.clamp(0, num_rows - 1)[keep],
                                     n_tap.to(torch.int64).clamp(
                                         0, num_channels - 1)[keep]),
                                    (w * w_chan * u[pos])[keep],
                                    accumulate=True)
    return out, took


def run_preflight(model, args, view_params, idx_full, sink):
    """The arithmetic gate, before any kernel exists: the torch twin of the
    kernel's arithmetic against the shipped TORCH body, on a small pixel
    subset at this run's own cell, for the window path and for a forced
    fallback.  Runs on CPU and on GPU alike, and gates the exit code."""
    import torch

    from mbirtorch.cone_beam import _cone_forward_view_batch

    rows = []
    views = min(PREFLIGHT_VIEWS, int(view_params.shape[0]))
    vp = view_params[:views]
    band_len = int(args["num_slices"])
    # THE PIXEL SET IS A RUN OF REAL FULL-MASK TILES, not a strided subset.
    # A tile's channel span is set by how many pixels share a (bucket,
    # channel) cell, and at this cell the full mask puts about two dozen
    # there; thin the mask and the window stops closing, which is the design's
    # own subset finding rather than an arithmetic fact.  A strided subset
    # therefore sends every tile down the per-tap path and would leave the
    # window branch ungated.  So the whole mask is sorted once here and one
    # contiguous, tile-aligned run of that order is handed to both the twin
    # and the reference.  The run is a ragged length, so padded lanes ride
    # too.
    arrays, z_offset = build_contract(idx_full, vp, args)
    perm_full = compound_sort(arrays[5], arrays[0], views)
    sorted_arrays = gather_contract(arrays, perm_full, views)
    full = int(idx_full.shape[0])
    want = min(PREFLIGHT_PIXELS, full)
    start = (max(0, (full - want) // 2) // BLOCK_P) * BLOCK_P
    order0 = perm_full[0, start:start + want].to(torch.int64)
    idx = idx_full[order0].contiguous()
    prep_arrays = [t[:, start:start + want].contiguous() for t in sorted_arrays]
    # The values rows already arrive in the sorted order, so the kernel's
    # permutation is the identity for this set.
    identity = torch.arange(int(idx.shape[0]), dtype=torch.int32,
                            device=idx.device)[None, :]
    tap_r = row_tap_radius(prep_arrays[5])
    del arrays, sorted_arrays, perm_full
    values = _seeded_values(torch, model, int(idx.shape[0]), band_len)
    for name in PREFLIGHT_ARMS:
        config = config_named(name)
        slice_tile = config[2]
        num_tiles_k = max(1, -(-band_len // slice_tile))
        probes = sorted({int(round(i * (num_tiles_k - 1)
                                   / max(1, PREFLIGHT_SLICE_TILES - 1)))
                         for i in range(PREFLIGHT_SLICE_TILES)})
        # Zero every slice outside the probe tiles, so the shipped body's
        # output is exactly the contribution the twin accumulates.
        masked = torch.zeros_like(values)
        for tile_k in probes:
            lo = tile_k * slice_tile
            hi = min(lo + slice_tile, band_len)
            masked[:, lo:hi] = values[:, lo:hi]
        reference = _cone_forward_view_batch(masked, idx, vp, slice_start=0,
                                             plan=None, **args)
        # One chunk covering the preflight's views, so the twin's ordering is
        # the ordering the kernel would use for them.
        prep_config = (config[0], views) + tuple(config[2:])
        prep = dict(arrays=prep_arrays, z_offset=z_offset.contiguous(),
                    perm=identity, sort_ms=None, gather_ms=None,
                    row_tap_radius=tap_r)
        got, took = grouped_twin(masked, prep, args, prep_config, probes,
                                 band_len)
        check = compare_blocks(got, reference, VALUES_GATE_REL)
        seen = took["window"] + took["fallback"]
        # Which branch each tile took, so the arms prove their coverage
        # instead of asserting it: the forced arm must send every tile down
        # the per-tap path, and the window arm must send at least some down
        # the window path, or the preflight has gated only one of the two.
        row = dict(kind="preflight", arm=name, num_pixels=int(idx.shape[0]),
                   views=views, probe_slice_tiles=probes,
                   forced_fallback=(name == FORCED_FALLBACK),
                   row_tap_radius=prep["row_tap_radius"], values=check,
                   tiles_seen=seen, tiles_window=took["window"],
                   tiles_fallback=took["fallback"],
                   window_share=(took["window"] / seen if seen else None))
        row["coverage_ok"] = bool(
            seen and (took["fallback"] == seen if name == FORCED_FALLBACK
                      else took["window"] > 0))
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        print(f'  preflight {name:<{NAME_COL}} values '
              f'{check.get("rel", float("nan")):.2e}  '
              f'{"ok" if check.get("ok") else "FAILED"}   window path on '
              f'{took["window"]}/{seen} tiles, per-tap on '
              f'{took["fallback"]}', flush=True)
        del masked, reference, got
        if values.is_cuda:
            torch.cuda.empty_cache()
    return rows


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
    configs = _strict_subset("MG38_CONFIGS", [c[0] for c in SWEEP_CONFIGS])
    header = dict(kind="run", smoke=SMOKE, device=DEVICE, cell=list(cell()),
                  view_batch=VIEW_BATCH, values_seed=VALUES_SEED,
                  warmup=WARMUP_REPEATS, timed=TIMED_REPEATS,
                  values_gate=VALUES_GATE_REL, buckets=BUCKETS,
                  block_p=BLOCK_P, ladder=list(PIXEL_LADDER),
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
            "ladder, the compound sort, the fallback estimate, the "
            "arithmetic preflight and the report machinery on the torch body")

    args = model._view_batch_args()
    header["psf_radius"] = int(args["psf_radius"])
    header["bp_psf_radius"] = int(args["bp_psf_radius"])
    header["num_slices"] = int(args["num_slices"])
    pf = model.projector_functions
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    header["view_batch_used"] = view_batch
    view_params = pf._view_params_per_dev[0][:view_batch]
    idx_full = model.full_indices_device()
    header["num_pixels_full"] = int(idx_full.shape[0])
    header["num_pixels_matches_expected"] = (
        None if SMOKE else int(idx_full.shape[0]) == NUM_PIXELS_EXPECTED)
    band_len = int(args["num_slices"])
    values_full = _seeded_values(torch, model, header["num_pixels_full"],
                                 band_len)
    sink.write(json.dumps(header) + "\n")
    sink.flush()

    print("  the arithmetic preflight (torch twin against the torch body)",
          flush=True)
    preflight_rows = run_preflight(model, args, view_params, idx_full, sink)

    baseline_points = []
    config_rows = {name: dict(kind="config", config=name,
                              params=config_named(name)[1:],
                              points=[], values_ok=True, worst_rel=0.0)
                   for name in configs}

    # Largest point first: the full-mask point pays every configuration's
    # compile inside its discarded warm-up.
    for divisor in PIXEL_LADDER:
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
                    num_pixels=realized)
        baseline_points.append(base)
        sink.write(json.dumps(base) + "\n")
        sink.flush()
        print(f'  baseline /{divisor:<4d} ({realized} px): '
              f'{base["median_ms"]:9.2f} ms', flush=True)

        for name in configs:
            row = config_rows[name]
            config = config_named(name)
            # The ordering and its span test are host work, read outside the
            # timed region; the timed launch below builds its own.
            prep = grouped_prep(values, idx, view_params, args, config)
            fb = fallback_fraction(prep, args, config, band_len)
            del prep
            if cuda:
                torch.cuda.empty_cache()
            if not header["variants_runnable"]:
                row["skipped"] = header["variants_skipped_reason"]
                row.setdefault("fallback", []).append(
                    dict(divisor=divisor, **fb))
                print(f'    {name:<{NAME_COL}} SKIPPED (fallback estimate '
                      f'{fb.get("fraction")})', flush=True)
                continue

            def variant():
                return grouped_launch(values, idx, view_params, args, config)

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
            point.update(divisor=divisor, num_pixels=realized, values=check,
                         fallback=fb,
                         sort_ms=getattr(grouped_launch, "last_sort_ms", None),
                         gather_ms=getattr(grouped_launch, "last_gather_ms",
                                           None),
                         speedup=(base["median_ms"] / point["median_ms"]
                                  if point["median_ms"] > 0 else None))
            row["points"].append(point)
            row["values_ok"] = row["values_ok"] and bool(check.get("ok"))
            if check.get("rel") is not None:
                row["worst_rel"] = max(row["worst_rel"], check["rel"])
            print(f'    {name:<{NAME_COL}} {point["median_ms"]:9.2f} ms  '
                  f'{point["speedup"]:5.2f}x  values '
                  f'{check.get("rel", float("nan")):.2e}  fallback '
                  f'{(fb.get("fraction") if fb.get("fraction") is not None else float("nan")):.3f}',
                  flush=True)
        del reference
        if cuda:
            torch.cuda.empty_cache()

    for name in configs:
        row = config_rows[name]
        for point in row["points"]:
            if point["divisor"] == 1:
                row["fullmask_ms"] = point["median_ms"]
                row["fullmask_speedup"] = point["speedup"]
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    baseline_full = next((p for p in baseline_points if p["divisor"] == 1),
                         None)
    header["baseline_fullmask_ms"] = (baseline_full or {}).get("median_ms")
    header["kernel_builds"] = kernel_build_entries()
    if cuda:
        header["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    header["health_after"] = sample_gpu_health()
    header["gpu_hot_or_throttled"] = bool(
        health_is_hot(header.get("health_before") or [])
        or health_is_hot(header["health_after"]))
    sink.write(json.dumps(dict(kind="run_tail",
                               baseline_fullmask_ms=header[
                                   "baseline_fullmask_ms"],
                               kernel_builds=header["kernel_builds"],
                               peak_bytes=header.get("peak_bytes"),
                               health_after=header["health_after"],
                               gpu_hot_or_throttled=header[
                                   "gpu_hot_or_throttled"])) + "\n")
    sink.flush()
    return header, baseline_points, config_rows, preflight_rows


# ── the counter attempt on the winner (mg33's machinery) ──────────────────────
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
    values = _seeded_values(torch, model, int(idx.shape[0]),
                            int(args["num_slices"]))
    config = config_named(cfg["config"])
    result["num_pixels"] = int(idx.shape[0])
    result["view_batch"] = view_batch
    result["values_bytes"] = int(idx.shape[0]) * int(args["num_slices"]) * 4
    for _index in range(int(cfg.get("launches", NCU_LAUNCHES))):
        out = grouped_launch(values, idx, view_params, args, config)
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
            by_kernel = {}
            for row in body:
                if len(row) <= max(name_at, value_at, kernel_at):
                    continue
                key = row[kernel_at].strip()
                entry = by_kernel.setdefault(key, dict(kernel=key, metrics={}))
                entry["metrics"][row[name_at].strip()] = number(row[value_at])
            return list(by_kernel.values())
        out = []
        for row in body:
            if len(row) < len(header):
                continue
            values = [number(c) for c in row]
            if all(isinstance(v, str) for v in values):
                continue
            entry = dict(kernel=row[kernel_at].strip(), metrics={})
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
    The confirmation this run is after is the DRAM read falling by roughly the
    view chunk; everything records and nothing gates."""
    leg = dict(attempted=True, winner=winner)
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG38_NCU=0")
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
    pattern = "regex:" + re.escape("_cone_grouped_kernel")
    leg["kernel_name_filter"] = pattern
    cfg = dict(config=winner, launches=NCU_LAUNCHES)
    cmd = [ncu, "--csv", "--page", "raw", "--target-processes", "all",
           "--kernel-name", pattern, "--launch-skip", str(NCU_LAUNCHES - 1),
           "--launch-count", "1", "--metrics", ",".join(METRICS_FULL),
           sys.executable, "-u", os.path.abspath(__file__),
           "--one-launch", json.dumps(cfg)]
    got = _run(cmd, NCU_TIMEOUT_S, env=variant_env())
    log_path = os.path.join(results_dir, f"mg38_ncu_{winner}.log")
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
    """The best configuration at the FULL MASK -- the design's domain --
    among those whose values passed at every ladder point.  The forced
    fallback arm never wins; it is a gate, not a candidate."""
    winner, winner_speed = None, None
    for name, row in config_rows.items():
        if row.get("skipped") or row.get("error") or not row.get("values_ok"):
            continue
        if name == FORCED_FALLBACK:
            continue
        speed = row.get("fullmask_speedup")
        if speed is not None and (winner_speed is None or speed > winner_speed):
            winner, winner_speed = name, speed
    return winner, winner_speed


def summarize(header, baseline_points, config_rows, preflight_rows, leg,
              out_path):
    print(f"\n===== mg38 cone two-axis grouping spike ({out_path}) =====")
    print(f'\nbaseline (the shipped wrapper, {header.get("forward_body")}): '
          f'full mask '
          f'{_fmt(header.get("baseline_fullmask_ms"), 0, "f", 2)} ms at '
          f'{header.get("view_batch_used")} views.  There is no anchor '
          "constant for this cell in this file; the reviewer compares this "
          "against mg31's rows.")

    print("\n===== the arithmetic preflight (no kernel involved) =====")
    for row in preflight_rows:
        print(f'  {row["arm"]:<{NAME_COL}} '
              f'{"forced fallback" if row["forced_fallback"] else "window path"}'
              f'  rel {_fmt((row["values"] or {}).get("rel"), 9, "e", 2)}  '
              f'{"ok" if (row["values"] or {}).get("ok") else "FAILED"}'
              f'   tiles {row.get("tiles_window")} window / '
              f'{row.get("tiles_fallback")} per-tap'
              f'{"" if row.get("coverage_ok") else "   COVERAGE SHORT"}')
    print("  This is the torch twin of the kernel's arithmetic against the "
          "torch body.  It is here because the inverted decomposition has "
          "one float-order trap: evaluating the trapezoid on "
          "(m0 + W_p_r * slice) - row instead of inverting at the row costs "
          "about 5e-5 relative at this cell, which reads as a values-gate "
          "failure with no bug behind it.")

    line = (f'{"config":<{NAME_COL}}{"chunk":>7}{"slices":>7}{"rows":>6}'
            f'{"chans":>6}{"full ms":>10}{"full x":>8}{"sort ms":>9}'
            f'{"gather ms":>10}{"fallback":>10}{"worst rel":>11}  check')
    print("\n===== the sweep, against the shipped wrapper =====")
    print(line)
    print("-" * len(line))
    for name, row in config_rows.items():
        if row.get("skipped"):
            rates = [f.get("fraction") for f in row.get("fallback", [])
                     if f.get("fraction") is not None]
            extra = (f'  (host fallback estimate max {max(rates):.3f})'
                     if rates else "")
            print(f'{name:<{NAME_COL}}  SKIPPED: {row["skipped"][:60]}{extra}')
            continue
        if row.get("error"):
            print(f'{name:<{NAME_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            continue
        chunk, slice_tile, wr, wc, _warps, _stages = row["params"]
        check = "ok" if row["values_ok"] else "VALUES GATE FAILED"
        full = next((p for p in row["points"] if p["divisor"] == 1), None)
        rates = [p.get("fallback", {}).get("fraction") for p in row["points"]
                 if p.get("fallback", {}).get("fraction") is not None]
        print(f'{name:<{NAME_COL}}{chunk:>7}{slice_tile:>7}{wr:>6}{wc:>6}'
              f'{_fmt(row.get("fullmask_ms"), 10, "f", 2)}'
              f'{_fmt(row.get("fullmask_speedup"), 8, "f", 2)}'
              f'{_fmt((full or {}).get("sort_ms"), 9, "f", 1)}'
              f'{_fmt((full or {}).get("gather_ms"), 10, "f", 1)}'
              f'{_fmt((full or {}).get("fallback", {}).get("fraction"), 10, "f", 3)}'
              f'{_fmt(row.get("worst_rel"), 11, "e", 2)}  {check}')
    print("-" * len(line))
    print("  'full x' is the shipped wrapper's time over the variant's at "
          "the full mask.  'sort ms' and 'gather ms' are the compound "
          "ordering's own share of that time, which production would "
          "memoize; 'fallback' is the HOST estimate of the share of tiles "
          "taking the per-tap path, not an in-kernel count.")
    print("\n===== every ladder point =====")
    line2 = (f'{"config":<{NAME_COL}}{"divisor":>9}{"pixels":>10}'
             f'{"base ms":>10}{"variant ms":>12}{"x":>7}{"fallback":>10}')
    print(line2)
    print("-" * len(line2))
    base_by_div = {p["divisor"]: p["median_ms"] for p in baseline_points}
    for name, row in config_rows.items():
        for point in row.get("points", []):
            print(f'{name:<{NAME_COL}}{point["divisor"]:>9}'
                  f'{point["num_pixels"]:>10}'
                  f'{_fmt(base_by_div.get(point["divisor"]), 10, "f", 2)}'
                  f'{_fmt(point["median_ms"], 12, "f", 2)}'
                  f'{_fmt(point.get("speedup"), 7, "f", 2)}'
                  f'{_fmt(point.get("fallback", {}).get("fraction"), 10, "f", 3)}')
    print("-" * len(line2))
    print("  The subset points are expected to lose: a thinned pixel set has "
          "no channel locality, the span guard fires, and the per-tap "
          "fallback carries the call.  That is the disengagement evidence "
          "section 9's rider asks for, not a defect.")

    builds = header.get("kernel_builds") or []
    if builds:
        print("  Triton's compile record for the variant (a spilled tile "
              "lands in local memory, which is DRAM again, and disqualifies "
              "its configuration whatever the timing says):")
        for entry in builds:
            print(f'    warps {entry.get("num_warps")}: '
                  f'{entry.get("n_regs")} registers, '
                  f'{entry.get("n_spills")} spills, shared '
                  f'{entry.get("shared")}')
    winner, winner_speed = pick_winner(config_rows)
    if winner:
        print(f"\n  the winning configuration at the full mask: {winner} "
              f"({winner_speed:.2f}x)")

    if leg.get("kernels"):
        kernel = leg["kernels"][0]
        worker = leg.get("worker") or {}
        dram_rd = _number(_metric(kernel, "dram__bytes_read.sum"))
        vals_bytes = worker.get("values_bytes")
        ratio = (dram_rd / vals_bytes
                 if (dram_rd is not None and vals_bytes) else None)
        print(f'\n===== the counter confirmation on {leg.get("winner")} '
              "=====")
        print(f'  DRAM read over the recon block: {_fmt(ratio, 0, "f", 1)}  '
              "(mg31 measured the shipped kernel at about one per view; the "
              "design predicts one per view CHUNK)")
        print(f'  SM / memory throughput, percent of peak: '
              f'{_fmt(_number(_metric(kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed")), 0, "f", 1)}'
              f' / '
              f'{_fmt(_number(_metric(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")), 0, "f", 1)}'
              '  (mg31 read the shipped cone forward at 50.4 / 63.6)')
        print(f'  atomic-path sectors: '
              f'{_fmt((_number(_metric(kernel, "lts__t_sectors_op_atom.sum")) or 0) + (_number(_metric(kernel, "lts__t_sectors_op_red.sum")) or 0), 0, "e", 3)}')
    elif leg.get("attempted"):
        print(f'\ncounter confirmation: no profile '
              f'({leg.get("reason", "see the ncu log")})')

    checks = []
    for row in preflight_rows:
        if not (row.get("values") or {}).get("ok"):
            checks.append(f'the arithmetic preflight arm {row["arm"]} read '
                          f'{(row["values"] or {}).get("rel")} against the '
                          f'{VALUES_GATE_REL:.0e} gate')
        if not row.get("coverage_ok"):
            checks.append(f'the arithmetic preflight arm {row["arm"]} did not '
                          f'exercise the branch it exists for '
                          f'({row.get("tiles_window")} window, '
                          f'{row.get("tiles_fallback")} per-tap)')
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
          f'{"healthy" if healthy else "BROKEN"}.  It covers the arithmetic '
          "preflight, the values gates, the body selection, and no "
          "configuration raising.  The winner and its speedup are read by a "
          "person, and the LIBRARY decision waits for the composed re-gate; "
          "nothing ships from this spike.")
    for item in checks:
        print(f"  FAIL: {item}")
    return dict(kind="summary", healthy=healthy, checks=checks,
                winner=winner, winner_fullmask_speedup=winner_speed,
                out_path=out_path)


def _dry_run():
    configs = _strict_subset("MG38_CONFIGS", [c[0] for c in SWEEP_CONFIGS])
    print(f"mg38 cone two-axis grouping spike: device {DEVICE}, cell "
          f"{tuple(cell())}, view batch {VIEW_BATCH}")
    print("  design note pfwd_segmented_design.md section 9, the spike step. "
          " It decides nothing about the library; a winning configuration "
          "goes to the composed re-gate.")
    print(f"  results -> {RESULTS_DIR}")
    print(f"  baseline: the shipped cone forward wrapper on the pixel ladder "
          f"{list(PIXEL_LADDER)} (full mask, then the strided subsets)")
    print(f"  grouping: {BUCKETS} W_p_r buckets, {BLOCK_P} pixels per tile, "
          "one compound sort per view chunk at the chunk's first view")
    print(f'  {"config":<{NAME_COL}}{"chunk":>7}{"slices":>8}'
          f'{"rows":>7}{"chans":>7}{"warps":>7}{"stages":>8}')
    for name, chunk, slice_tile, wr, wc, warps, stages in SWEEP_CONFIGS:
        marker = "" if name in configs else "  (not selected)"
        print(f"  {name:<{NAME_COL}}{chunk:>7}{slice_tile:>8}{wr:>7}{wc:>7}"
              f"{warps:>7}{stages:>8}{marker}")
    print(f"  values gate {VALUES_GATE_REL:.0e} against the shipped wrapper "
          f"at every ladder point; {WARMUP_REPEATS} warm + {TIMED_REPEATS} "
          "timed per point")
    print(f"  arithmetic preflight on {PREFLIGHT_PIXELS} pixels, "
          f"{PREFLIGHT_VIEWS} views, {PREFLIGHT_SLICE_TILES} probe slice "
          f"tiles, arms {list(PREFLIGHT_ARMS)}")
    print(f"  counter attempt on the winner: "
          f"{'on' if NCU_ENABLED else 'off (MG38_NCU=0)'}")
    print("  no library file is touched: the variant kernel, its launch, the "
          "compound sort and the fallback estimate all live in this file")


def main():
    if DRY:
        _dry_run()
        return 0
    if not SMOKE:
        import torch
        if not torch.cuda.is_available():
            print("this run needs CUDA; use MG38_SMOKE=1 for the CPU "
                  "plumbing pass")
            return 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR,
                            f"mg38_cone_grouped_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg38 cone grouped spike on {RUN_LABEL} ({DEVICE}) -> {out_path}",
          flush=True)
    with open(out_path, "w") as sink:
        header, baseline_points, config_rows, preflight_rows = run_sweep(sink)
        winner, _speed = pick_winner(config_rows)
        print("  counter attempt on the winner", flush=True)
        leg = ncu_on_winner(winner, RESULTS_DIR)
        sink.write(json.dumps(dict(kind="ncu_leg", **{
            k: v for k, v in leg.items() if k != "kernels"})) + "\n")
        if leg.get("kernels"):
            sink.write(json.dumps(dict(kind="ncu_kernels",
                                       kernels=leg["kernels"])) + "\n")
        summary = summarize(header, baseline_points, config_rows,
                            preflight_rows, leg, out_path)
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
