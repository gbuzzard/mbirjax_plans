"""mg25 -- THE COUNTER RUN ON THE CONE BACK KERNEL.

WHAT THIS RUN IS.  Open item B3 (in-kernel sorted or segmented accumulation)
names one precondition in back_remedy_design.md section 6: a counter run on
the back kernel itself, taken after the padding's confirmation runs.  This is
that run.  It reads Nsight Compute counters on the Triton cone back kernel
(mbirtorch/triton_cone.py, ``_cone_back_kernel``) at five production band
lengths, and prints what they say.

WHERE THIS SITS.  The band-padding remedy landed as mbirtorch 64dedb8.  Every
production band is now rounded up to a multiple of 16 before the launch, so
every production band launches at the divisible rate.  Findings section 1.23
measured that: the three non-divisible bands fell from about 47,000 to about
20,400 ns per view-slice, which closes the 2.44x divisibility cliff section
1.21 recorded.  The band-length question is therefore settled, and it is not
what this run asks.  What is left of the cone back question is the efficiency
of a SINGLE call at that rate: whether the kernel's gathers leave headroom
worth a kernel-campaign increment.

WHY MG20'S COUNTERS DO NOT ANSWER IT.  mg20 read the write path of the
parallel FORWARD kernel, which scatters its results with one float atomic add
per detector-channel tap.  The cone back kernel has no atomics.  It gathers
sinogram taps and stores once per output element (back_remedy_design.md
section 6).  mg20's atomic and reduction counters therefore describe a
different kernel, and this run reads the back kernel itself.

## THE THREE QUESTIONS, AND THE COLUMNS THAT ANSWER EACH

  1. ARE THE GATHERS TRANSACTION-BOUND AT THE DIVISIBLE RATE?  Read the memory
     throughput against the SM throughput (both as a percent of speed of
     light), the long-scoreboard stall ratio, and the L1 and L2 hit rates.
     Memory throughput far above SM throughput, with a high stall ratio, is a
     kernel waiting on its loads.

  2. WHAT IS THE GATHER PATH'S SECTOR EFFICIENCY?  Read the average sectors
     per request on the global load path, and the L1 load sector count against
     the ideal.  The ideal is one 32-byte sector per eight float32 taps,
     because the kernel gathers along the slice axis of a channel-major
     sinogram, where consecutive slice lanes are consecutive addresses.  A
     sector count near that ideal means the gathers are already coalesced.

  3. WHAT LIMITS OCCUPANCY?  Read the three occupancy limits (registers,
     shared memory, blocks per multiprocessor), the registers per thread, and
     the achieved occupancy.  The tile is fixed at 16 x 64 with 4 warps and 1
     stage, so the limiter should not move with the band.

PLUS ONE ZERO-WITNESS.  The atomic and reduction sector counters are read at
every variant.  The design note states they are absent from this kernel by
construction; this run measures them instead of asserting them.  A number that
is not near zero means the reading of the kernel is wrong somewhere, and the
whole gather story above would have to be re-examined.

## THIS RUN DECIDES NOTHING

It implements nothing, and it starts no remedy.  The verdict -- whether
sorting or reordering the gathers has headroom worth building -- is read by a
PERSON from the tables this job prints.  Acting on that verdict needs Greg's
approval.  The exit code reports instrument health only: whether the
measurement was taken correctly, never what it found.  The B3 remedy is not
started here.

## THE CELL

Cone with the 2048-class detector, sinogram (256, 2016, 1984) as (views,
detector rows, channels).  This is mg21b's cell verbatim, so a rate here sits
beside mg21b's and mg23's rates without adjustment.  The pixel set is the full
region-of-reconstruction mask, expected to be 3,088,364 pixels; the count is
recorded and does not gate anything.  One device.

## THE BAND VARIANTS

Five bands, all divisible by 16, so the wrapper's ``padded_kernel_width`` call
returns each unchanged and the kernel launches at exactly these lengths:

    1008   the 1024-class one-device band, and the 2048-class two-device band
     672   the 2048-class three-device band
     512   the padded form of 504 (2048-class four-device, 1024-class
           two-device)
     336   the 1024-class three-device band
     256   the padded form of 252 (1024-class four-device)

That every one of them is already divisible is the point: these are the
lengths production launches at today, so the counters describe production.

## THE TIMING LEG

Always runs, in this process, by mg21b's method.  One seeded uniform sinogram
on the device, then per variant one warm call and three timed calls of
``Projectors.sparse_back_project_view_range`` -- the production route, the
same method the banded driver calls once per (device, band).  The reported
rate is the median wall divided by (views x band slices), in nanoseconds,
which is what makes bands of different lengths comparable.

THE WITNESS.  The first variant has the largest band, so every later band is a
prefix of it along the slice axis.  Each variant's output is compared against
that reference on the slices they share, in float64, at 1e-6 relative.  A
failure means the band plumbing is wrong, not that the timing is noisy.

THE RATE GATE is an instrument check, not a finding.  Every variant's rate
must be at or below 30,000 ns per view-slice.  The divisible-rate class
measured 18,600 to 21,500 ns per view-slice (mg21b, unpadded tree, divisible
bands) and about 20,400 on the padded tree (mg23); the non-divisible class
read about 46,000 to 51,000.  A rate above 30,000 means the padded wrapper did
not deliver the divisible rate on this node, and the counters below would then
describe the wrong kernel.  Each rate is also printed as a ratio against the
20,400 anchor, which is reported and not gated.

## THE COUNTER LEG

Optional, and it never changes the exit code.  It is mg20's Nsight Compute
machinery pointed at this kernel: a permission probe, then kernel-name
discovery at runtime, then one profiled warm launch per variant.

THE PROFILED LAUNCH IS PRODUCTION-SHAPED.  The single-launch worker asks the
driver's own rule (``Projectors._effective_view_batch``) how many views one
body call takes at this cell, and launches with that view count; mg21
measured 13 views per body call at this detector.  The worker imports
``_cone_back_view_batch_triton`` DIRECTLY rather than taking it from
``model._view_batch_bodies()``, because that selection runs an availability
self-check which launches the same kernel once on a tiny problem, and that
launch would be the first thing ncu's filter matched.  Which body the model
binds is witnessed by the timing leg instead.

THE PIXEL DIVISOR.  The profiled worker subsamples the pixel mask by
MG25_NCU_PIXEL_DIV (4 by default).  ncu replays each kernel several times and
saves and restores the memory it wrote between passes, so a smaller output
partial is a much shorter profile.  What the counters read are intensities
(rates, hit rates, sectors per request) and per-launch shape, and the divisor
leaves both representative.  One control variant profiles the same band at the
FULL pixel mask so that assumption can be checked; it runs last, because it is
the likeliest to exhaust the attempt timeout.

THE METRIC SETS.  Several of the metric names this run wants have not been
collected on this cluster's ncu before, so the leg falls back.  It tries the
full set first, then the core set that mg20 proved on this cluster (job
15316589) plus one confirmed addition, then the core set again profiling the
first launch instead of the last.  Which set succeeded is recorded on every
counter row, and a column whose metric was not in that set prints as blank
rather than as zero.  The registers per thread is the one column that does not
depend on the metric set: Triton's own compile cache reports it, so the
occupancy-limiter question is answered even when only the core set collects.

NSIGHT COMPUTE DURATIONS ARE NOT WALL TIMES.  ncu serializes kernels and
replays each one to collect its counters.  The durations in the counter table
compare variants within that table only.  The timing leg owns time.

## THE EXIT CODE

Instrument health only.  It is 0 when every selected variant produced a timing
row, every witness was at or below 1e-6, every rate was at or below 30,000 ns
per view-slice (skipped in the smoke), the model bound no torch body in either
direction on CUDA, and the realized device is the one asked for.  The counter
leg's absence, its refusal, and whatever it finds all leave the exit code
alone.

## THE LOCAL SMOKE

MG25_SMOKE=1 runs the whole variant plan on a tiny cone cell on the CPU.
Triton is unavailable there, so the model binds its torch bodies, the rate
gate is skipped and recorded (a torch body's rate is not this kernel's), and
the counter leg is skipped and recorded.  What the smoke exercises is the
harness: the variant plan, the production-route call, the witness, the rows
and the tables.  It is not a measurement.

Run:
    <torch python> mg25_back_counters.py           on a 1-GPU node
    MG25_DRY=1 <python> mg25_back_counters.py      print the plan and stop
    MG25_SMOKE=1 <python> mg25_back_counters.py    the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized band is an error, not a silent skip.
    MG25_RESULTS=<dir>          where the jsonl and the ncu logs go
    MG25_BANDS=1008,672         subset of the bands, by band length
    MG25_DRY=1                  print the plan and exit; imports no torch
    MG25_SMOKE=1                the local CPU smoke
    MG25_NCU=0                  skip the counter leg entirely
    MG25_NCU_PIXEL_DIV=4        profile at a quarter of the pixels
    MG25_REPEATS=3              timed calls per variant
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


def _flag(name, default="0"):
    """An environment flag that must read exactly "0" or "1".

    Accepting "true" or "yes" silently as false has cost this work a repeat
    before: the run prints the plan it was asked for and measures another one.
    """
    raw = os.environ.get(name, default).strip()
    if raw not in ("0", "1"):
        raise ValueError(f"{name}: {raw!r} is not 0 or 1")
    return raw == "1"


def _positive_int(name, default):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}: {raw!r} is not an integer") from None
    if value < 1:
        raise ValueError(f"{name}: {value} is not at least 1")
    return value


SMOKE = _flag("MG25_SMOKE")
DRY = _flag("MG25_DRY")
DEVICE = "cpu" if SMOKE else "cuda"

#: (views, detector rows, channels).  mg21b's cell verbatim.
CELL = (256, 2016, 1984)
SMOKE_CELL = (16, 24, 20)
#: The full region-of-reconstruction mask at this cell.  Recorded, not gated.
NUM_PIXELS_EXPECTED = 3088364

#: The swept configurations, in descending band order.  Every band here is
#: divisible by 16, so ``padded_kernel_width`` returns it unchanged and the
#: kernel launches at exactly this length -- which is the point, because these
#: are the lengths production launches at on the padded tree.
BAND_VARIANTS = (
    dict(variant="b1008", band=1008, fullpix_control=False,
         note="the 1024-class one-device band, and the 2048-class two-device "
              "band"),
    dict(variant="b672", band=672, fullpix_control=True,
         note="the 2048-class three-device band"),
    dict(variant="b512", band=512, fullpix_control=False,
         note="the padded form of 504: the 2048-class four-device band and "
              "the 1024-class two-device band"),
    dict(variant="b336", band=336, fullpix_control=False,
         note="the 1024-class three-device band"),
    dict(variant="b256", band=256, fullpix_control=False,
         note="the padded form of 252: the 1024-class four-device band"),
)
SMOKE_BAND_VARIANTS = (
    dict(variant="b16", band=16, fullpix_control=False,
         note="the smoke's largest band, so it is the witness reference"),
    dict(variant="b8", band=8, fullpix_control=True,
         note="a prefix of the reference band"),
)

WARMUP_REPEATS = 1                # pays the variant's Triton compile
TIMED_REPEATS = _positive_int("MG25_REPEATS", 3)
SINO_SEED = 20260818
WITNESS_REL = 1e-6

#: The instrument gate on the timing leg, in nanoseconds per view-slice.  The
#: divisible-rate class measured 18,600 to 21,500 (mg21b, unpadded tree,
#: divisible bands) and about 20,400 on the padded tree (mg23); the
#: non-divisible class read about 46,000 to 51,000.  A rate above 30,000 means
#: the padded wrapper did not deliver the divisible rate on this node, and the
#: counters would then describe the wrong kernel.  Skipped in the smoke, where
#: the torch body runs and its rate is not this kernel's.
RATE_GATE_NS = 30000.0
#: The padded tree's measured rate (mg23, findings section 1.23), printed
#: beside every variant as a ratio.  Reported, never gated.
RATE_ANCHOR_NS = 20400.0

# ── the counter leg ───────────────────────────────────────────────────────────
NCU_ENABLED = _flag("MG25_NCU", "1")
NCU_PIXEL_DIV = _positive_int("MG25_NCU_PIXEL_DIV", 4)
NCU_LAUNCHES = 5                  # the single-launch worker's launch count
# Two bounds, because ncu replays each kernel several times to collect its
# counters and nobody has timed a replay of THIS kernel.  One attempt cannot
# run longer than the first, and the whole leg cannot run longer than the
# second; whatever is left unprofiled is recorded as unprofiled.  The timing
# leg has already finished by then, so a leg that runs out of budget costs the
# run nothing it needed.
NCU_TIMEOUT_S = 420
NCU_LEG_BUDGET_S = 1800
NCU_PROBE_TIMEOUT_S = 180

#: Everything this run wants.  Four of these names -- the two throughput
#: percentages, the L1 sector counters and the long-scoreboard ratio -- have
#: not been used on this cluster's ncu before, which is why there is a
#: fallback set below.
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
    "lts__t_sectors_op_write.sum",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct",
    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
#: The set mg20 proved on this cluster (job 15316589), plus one addition:
#: lts__t_sectors_op_read.sum.  This is the set that must not fail, so nothing
#: unproven goes in it.  The one addition is confirmed listed by this
#: cluster's ``ncu --query-metrics --chip gh100`` (ncu 2024.3.1), alongside the
#: op_red and op_atom names mg20 already collected.
#: launch__registers_per_thread is deliberately NOT here even though the full
#: set wants it: launch attributes are not listed by --query-metrics, so its
#: validity on this ncu is unproven, and the register count is available from
#: Triton's own compile cache anyway (see kernel_build_record).
METRICS_CORE = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "lts__t_sectors_op_read.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
#: The attempts, in order: (metric set name, metric set, launch skip).  The
#: last launch is the warm one, so it is aimed at first.  A skip of zero
#: profiles the FIRST launch, which is warm in every sense except the kernel
#: cache, and is worth having if the aimed skip matched nothing.
NCU_ATTEMPTS = (
    ("full", METRICS_FULL, NCU_LAUNCHES - 1),
    ("core", METRICS_CORE, NCU_LAUNCHES - 1),
    ("core", METRICS_CORE, 0),
)
NCU_PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                          "insufficient permission")

# ── GPU health ────────────────────────────────────────────────────────────────
# A thermally throttled or power-capped device produces valid values and an
# invalid rate, and the rate is this run's instrument gate.
HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

RESULTS_DIR = os.environ.get(
    "MG25_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
VARIANT_COL = 15
#: The library's rounding multiple, mirrored here so the DRY plan can print
#: launch bands without importing torch.  The real run reads the library's own
#: ``padded_kernel_width`` and records whether the two agreed.
PAD_MULTIPLE = 16
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer variants than it printed has cost this work a
    repeat before.
    """
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


def cell():
    return SMOKE_CELL if SMOKE else CELL


def variant_table():
    return SMOKE_BAND_VARIANTS if SMOKE else BAND_VARIANTS


def _padded_band(band):
    """``band`` rounded up to the next multiple of 16, mirroring
    :func:`mbirtorch._utils.padded_kernel_width`.

    This copy exists for the dry run, which imports no torch.  The real run
    imports the library's own function and records whether it agreed with this
    one at every band, so the mirror is checked rather than trusted.
    """
    band = int(band)
    remainder = band % PAD_MULTIPLE
    return band if remainder == 0 else band + PAD_MULTIPLE - remainder


def build_plan():
    """One entry per selected variant, in table order (descending band)."""
    allowed = [str(spec["band"]) for spec in variant_table()]
    keep = _strict_subset("MG25_BANDS", allowed)
    plan = []
    for spec in variant_table():
        if str(spec["band"]) not in keep:
            continue
        band = int(spec["band"])
        plan.append(dict(variant=spec["variant"], band=band,
                         launch_band=_padded_band(band),
                         divisible_by_16=(band % PAD_MULTIPLE == 0),
                         fullpix_control=bool(spec["fullpix_control"]),
                         note=spec["note"], cell=list(cell())))
    if not plan:
        raise ValueError("MG25_BANDS selects no variant")
    return plan


def ncu_variants(plan):
    """The variants the counter leg profiles, in attempt order.

    Every selected band at the pixel divisor, then one control at the full
    pixel mask.  The control runs LAST on purpose: it is the largest profile
    and therefore the likeliest to exhaust the attempt timeout, and the leg's
    budget bound records whatever is left unprofiled.
    """
    entries = [dict(variant=entry["variant"], band=entry["band"],
                    pixel_div=NCU_PIXEL_DIV, note=entry["note"])
               for entry in plan]
    if NCU_PIXEL_DIV != 1:
        control = next((entry for entry in plan
                        if entry.get("fullpix_control")), None)
        if control is not None:
            entries.append(dict(
                variant=control["variant"] + "_fullpix", band=control["band"],
                pixel_div=1,
                note="the same band at the FULL pixel mask: the control on "
                     "whether the pixel divisor left the counters "
                     "representative"))
    return entries


# ── GPU health ────────────────────────────────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Clocks, temperatures, and throttle flags, sampled before and after the
    timing leg."""
    for fields in (_GPU_FIELDS_FULL, _GPU_FIELDS_MIN):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + fields,
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
        except Exception:                                         # noqa: BLE001
            return []
        if proc.returncode != 0:
            continue
        full = fields is _GPU_FIELDS_FULL
        out = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            entry = {"index": _gi(parts[0]), "sm_mhz": _gi(parts[1])}
            if full and len(parts) >= 9:
                entry["mem_mhz"] = _gi(parts[2])
                entry["temp_c"] = _gi(parts[3])
                entry["mem_temp_c"] = _gi(parts[4])
                entry["throttle"] = [name for name, value
                                     in zip(_THROTTLE_NAMES, parts[5:9])
                                     if value.lower() == "active"]
            else:
                entry["temp_c"] = _gi(parts[2])
            out.append(entry)
        if out:
            return out
    return []


def health_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C) or gpu.get("throttle"):
            return True
    return False


# ── the model and the pixel set ───────────────────────────────────────────────
def build_model():
    """mg21b's construction at mg21b's cell, unchanged.

    ``skip_memory_preflight`` is set BEFORE the device is configured: the
    preflight prices a full reconstruction, and this run allocates a sinogram,
    one output partial, and the per-view precomputes.
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
    model.configure_devices(
        devices=[DEVICE + (":0" if DEVICE == "cuda" else "")])
    model.set_params(no_warning=True, verbose=0)
    return model


def full_indices(torch_module, model):
    """The full region-of-reconstruction mask as int64 on the model's device.

    Built the same way in both legs, so the pixel set the counters describe is
    the pixel set the timing leg measured.
    """
    import numpy as np

    indices = np.asarray(model._full_indices()).reshape(-1).astype(np.int64)
    return torch_module.as_tensor(indices, device=model.torch_device)


def launch_geometry(num_pixels, band):
    """The grid, the tile, and the launched band, derived from the library's
    own rules so the tables can print the launch shape without a profiler.

    Every rule here is imported rather than restated: ``padded_kernel_width``
    rounds the band, ``_tile_size`` picks each tile, and the grid is the
    wrapper's own ceiling division.  A band already divisible by 16 gets its
    own value back, which is true of every variant in this run and is checked
    on the row.
    """
    try:
        from mbirtorch._utils import padded_kernel_width
        from mbirtorch.triton_cone import (CONE_BACK_BLOCK_L,
                                           CONE_BACK_BLOCK_P,
                                           CONE_BACK_MIN_TILE, _tile_size)
    except Exception as exc:                                      # noqa: BLE001
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    band = int(band)
    launch_band = int(padded_kernel_width(band))
    block_p = _tile_size(CONE_BACK_BLOCK_P, int(num_pixels), CONE_BACK_MIN_TILE)
    block_l = _tile_size(CONE_BACK_BLOCK_L, launch_band, CONE_BACK_MIN_TILE)
    grid = (-(-int(num_pixels) // block_p), -(-launch_band // block_l))
    return dict(available=True, block_p=int(block_p), block_l=int(block_l),
                grid=[int(grid[0]), int(grid[1])],
                blocks=int(grid[0]) * int(grid[1]),
                launch_band=launch_band,
                launch_band_equals_band=(launch_band == band),
                mirror_agrees=(launch_band == _padded_band(band)))


def kernel_build_record():
    """What Triton compiled for the cone back kernel, read from its own cache:
    registers, spills and shared memory per compiled variant.

    Registers per thread is what caps occupancy for this tile, so this answers
    part of question 3 whatever ncu does: the counter leg may fall back to a
    metric set without launch__registers_per_thread, and the leg may not run
    at all, and this record stands in both cases.  It also
    supplies candidate kernel NAMES for the ncu filter.  The attribute names
    have moved between Triton versions, so every lookup is defensive and a
    miss is recorded rather than raised.
    """
    try:
        from mbirtorch.triton_cone import _cone_back_kernel as back
    except Exception as exc:                                      # noqa: BLE001
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    entries, names = [], []
    caches = []
    for attr in ("cache", "device_caches"):
        holder = getattr(back, attr, None)
        if isinstance(holder, dict):
            caches.append(holder)
    for holder in caches:
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
                    if got is not None and not isinstance(got, (int, float, str)):
                        got = str(got)
                    record[field] = got
                if record.get("name"):
                    names.append(str(record["name"]))
                entries.append(record)
    return dict(available=bool(entries), entries=entries[:24],
                names=sorted(set(names)),
                python_name=getattr(back, "__name__", None))


# ── the witness ───────────────────────────────────────────────────────────────
def compare_prefix(candidate, reference, gate, chunks=8):
    """max|candidate - reference| / max|reference| over the slices the two
    share, taken in float64 on the device in pixel-row chunks.

    The maximum of maxima is the maximum, so chunking is exact.  It is chunked
    rather than taken whole because promoting a (3.1M, 1008) partial to
    float64 allocates 25 GB per operand, which is more device memory than the
    rest of this run uses together; in chunks the comparison costs a few
    gigabytes and the stated memory budget holds.
    """
    shared = min(int(candidate.shape[1]), int(reference.shape[1]))
    rows = int(reference.shape[0])
    if int(candidate.shape[0]) != rows:
        return dict(ok=False, rel=None, gate=gate, shared=shared,
                    reason=f"the candidate has {int(candidate.shape[0])} "
                           f"pixels and the reference has {rows}")
    step = max(1, -(-rows // max(1, int(chunks))))
    max_diff, max_ref = 0.0, 0.0
    for start in range(0, rows, step):
        left = candidate[start:start + step, :shared].double()
        right = reference[start:start + step, :shared].double()
        max_ref = max(max_ref, float(right.abs().max()))
        max_diff = max(max_diff, float((left - right).abs().max()))
        left = None
        right = None
    if max_ref <= 0.0:
        return dict(ok=False, rel=None, gate=gate, shared=shared,
                    reason="the reference is all zeros, so a relative "
                           "comparison has no denominator")
    rel = max_diff / max_ref
    return dict(ok=bool(rel <= gate), rel=rel, gate=gate, shared=shared,
                max_abs_diff=max_diff, max_abs_ref=max_ref)


# ── the timing leg ────────────────────────────────────────────────────────────
def timing_leg(plan, sink):
    """Every variant timed through the production route, in this process.

    Returns ``(header, rows)``.  Each row is written to the jsonl as soon as it
    is complete, so a job killed partway still leaves the variants it finished.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    header = dict(kind="run", smoke=SMOKE, device=DEVICE, cell=list(cell()),
                  bands=[entry["band"] for entry in plan],
                  warmup=WARMUP_REPEATS, timed=TIMED_REPEATS,
                  sino_seed=SINO_SEED, witness_rel=WITNESS_REL,
                  rate_gate_ns=RATE_GATE_NS, rate_anchor_ns=RATE_ANCHOR_NS,
                  rate_gate_applies=not SMOKE,
                  ncu_enabled=NCU_ENABLED, ncu_pixel_div=NCU_PIXEL_DIV,
                  torch=torch.__version__, node=platform.node(),
                  cuda=cuda, run_label=RUN_LABEL,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  health_before=sample_gpu_health())

    model = build_model()
    from mbirtorch import _memory_ledger
    header["torch_body_directions"] = list(
        _memory_ledger.torch_body_directions(model))
    header["torch_body_expected"] = [] if cuda else ["forward", "back"]
    header["bodies_ok"] = (header["torch_body_directions"]
                           == header["torch_body_expected"])
    fwd_body, back_body = model._view_batch_bodies()
    header["forward_body"] = fwd_body.__name__
    header["back_body"] = back_body.__name__

    device = model.torch_device
    header["device_realized"] = str(device)
    header["device_expected"] = "cuda:0" if cuda else DEVICE
    header["device_ok"] = (str(device) == header["device_expected"])

    num_views, num_rows, num_channels = cell()
    args = model._view_batch_args()
    header["psf_radius"] = int(args["psf_radius"])
    header["taps"] = 2 * int(args["psf_radius"]) + 1
    header["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]

    generator = torch.Generator(device="cpu").manual_seed(SINO_SEED)
    sino = torch.rand((num_views, num_rows, num_channels),
                      generator=generator, dtype=torch.float32).to(device)

    idx = full_indices(torch, model)
    header["num_pixels"] = int(idx.shape[0])
    header["num_pixels_expected"] = None if SMOKE else NUM_PIXELS_EXPECTED
    header["num_pixels_matches_expected"] = (
        None if SMOKE else int(idx.shape[0]) == NUM_PIXELS_EXPECTED)
    header["driver_view_batch"] = int(model.projector_functions
                                      ._effective_view_batch(
                                          back_body, int(idx.shape[0]),
                                          num_rows, args))
    sink.write(json.dumps(header) + "\n")
    sink.flush()

    def one_call(band):
        out = model.projector_functions.sparse_back_project_view_range(
            sino, idx, (0, num_views), coeff_power=1, slice_start=0,
            band_slices=int(band), dev_index=0)
        if cuda:
            torch.cuda.synchronize(device)
        return out

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    reference = None
    rows = []
    for entry in plan:
        row = dict(kind="variant", **{k: v for k, v in entry.items()
                                      if k != "cell"})
        row["geometry"] = launch_geometry(int(header["num_pixels"]),
                                          entry["band"])
        walls = []
        out = None
        for _repeat in range(WARMUP_REPEATS + TIMED_REPEATS):
            if out is not None:
                del out
                out = None
                if cuda:
                    torch.cuda.empty_cache()
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = one_call(entry["band"])
            walls.append(time.perf_counter() - start)
        row["wall_warmup_s"] = walls[:WARMUP_REPEATS]
        timed = walls[WARMUP_REPEATS:]
        row["wall_s"] = timed
        row["wall_median_s"] = statistics.median(timed)
        row["spread"] = ((max(timed) - min(timed)) / row["wall_median_s"]
                         if row["wall_median_s"] > 0 else None)
        # The comparable rate: nanoseconds per (view x band slice).  The
        # kernel's voxel work is pixels x band x views and the pixel set is the
        # same in every variant, so this rate differs only by efficiency.
        row["ns_per_view_slice"] = (
            row["wall_median_s"] / (num_views * int(entry["band"])) * 1e9)
        row["rate_vs_anchor"] = row["ns_per_view_slice"] / RATE_ANCHOR_NS
        if SMOKE:
            row["rate_ok"] = None
            row["rate_gate_skipped"] = (
                "the smoke runs the torch body on the CPU, and its rate is not "
                "this kernel's")
        else:
            row["rate_ok"] = bool(row["ns_per_view_slice"] <= RATE_GATE_NS)

        # The witness.  The first variant has the largest band, so every later
        # band is a prefix of it along the slice axis.  The reference is kept
        # as the returned tensor rather than a clone: the production route
        # allocates a fresh partial per call, and a clone would hold a second
        # 12 GB copy at this cell for nothing.
        if reference is None:
            reference = out
            out = None
            row["witness_rel"] = 0.0
            row["witness_ok"] = True
            row["witness_shared"] = int(reference.shape[1])
        else:
            witness = compare_prefix(out, reference, WITNESS_REL)
            row["witness_rel"] = witness.get("rel")
            row["witness_ok"] = bool(witness.get("ok"))
            row["witness_shared"] = witness.get("shared")
            row["witness"] = witness
            del out
            out = None
            if cuda:
                torch.cuda.empty_cache()
        rate_text = ("skipped" if row["rate_ok"] is None
                     else ("ok" if row["rate_ok"] else "RATE GATE FAILED"))
        witness_value = (float("nan") if row["witness_rel"] is None
                         else row["witness_rel"])
        print(f'  {row["variant"]:<8} band {entry["band"]:5d}  launch '
              f'{row["geometry"].get("launch_band", entry["launch_band"]):5d}  '
              f'median {row["wall_median_s"]:8.3f} s  '
              f'rate {row["ns_per_view_slice"]:9.1f} ns/(view*slice)  '
              f'witness {witness_value:.2e}  {rate_text}', flush=True)
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()

    # Everything the timing leg allocated is dead here, and the counter leg's
    # subprocesses need the device.  Dropped explicitly rather than left to the
    # interpreter, because this process stays alive for the whole counter leg.
    del reference
    del sino
    del idx
    if cuda:
        torch.cuda.empty_cache()
        header["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    header["health_after"] = sample_gpu_health()
    header["gpu_hot_or_throttled"] = bool(
        health_is_hot(header.get("health_before") or [])
        or health_is_hot(header["health_after"]))
    # The header row went to the jsonl before the leg ran, so the readings
    # taken after it get their own row; without this they would exist only in
    # the printed log.
    sink.write(json.dumps(dict(kind="run_tail",
                               peak_bytes=header.get("peak_bytes"),
                               health_after=header["health_after"],
                               gpu_hot_or_throttled=header[
                                   "gpu_hot_or_throttled"])) + "\n")
    sink.flush()
    return header, rows


# ── the single-launch worker the profiler drives ──────────────────────────────
def one_launch(cfg):
    """Build the model, launch the cone back kernel a few times, exit.

    Kept as small as it can be, because ncu profiles what it is given: a whole
    sweep under the profiler would be a sweep of replayed kernels.

    THE BODY IS IMPORTED DIRECTLY here rather than taken from
    ``model._view_batch_bodies()``.  That selection runs the kernel's
    availability self-check, which launches the same back kernel once on a tiny
    problem, and that launch would be the first thing ncu's kernel filter
    matched.  Importing the body makes this worker's launch count exactly the
    number below, so the launch skip lands where it is aimed.  Which body the
    model would have bound is witnessed by the timing leg.
    """
    import torch

    from mbirtorch.triton_cone import _cone_back_view_batch_triton

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, mode="one_launch", device=DEVICE, cuda=cuda,
                  launches=int(cfg.get("launches", NCU_LAUNCHES)))
    band = int(cfg["band"])
    num_views, num_rows, num_channels = cell()

    model = build_model()
    args = model._view_batch_args()
    pf = model.projector_functions
    # The batching rule rides on the BODY as a ``_view_batch_cost`` attribute
    # (projectors._effective_view_batch), and the directly imported wrapper
    # carries it.  Asking the model's own selection instead
    # (``_view_batch_bodies``) would run the first-use availability
    # self-check, which launches this same kernel once on a tiny problem --
    # and the skip-0 fallback attempt would then profile that tiny launch.
    back_body = _cone_back_view_batch_triton
    result["worker_back_body"] = back_body.__name__

    idx_full = full_indices(torch, model)
    full_pixels = int(idx_full.shape[0])
    pixel_div = max(1, int(cfg.get("pixel_div", 1)))
    want = max(1, full_pixels // pixel_div)
    step = max(1, full_pixels // want)
    idx = idx_full[::step][:want].contiguous()
    del idx_full
    used = int(idx.shape[0])
    result["num_pixels_full"] = full_pixels
    result["num_pixels_used"] = used
    result["pixel_div"] = pixel_div

    # The driver's own view-batch rule, so the profiled launch has the shape a
    # production launch has.  mg21 measured 13 views per body call at this
    # detector.
    view_batch = min(int(pf._effective_view_batch(back_body, used, num_rows,
                                                  args)), num_views)
    result["view_batch"] = view_batch
    result["psf_radius"] = int(args["psf_radius"])
    result["taps"] = 2 * int(args["psf_radius"]) + 1
    result["launch_geometry"] = launch_geometry(used, band)
    result["sino_batch_bytes"] = (view_batch * int(num_rows)
                                  * int(num_channels) * 4)
    result["out_partial_bytes"] = used * band * 4

    generator = torch.Generator(device="cpu").manual_seed(SINO_SEED)
    sino_batch = torch.rand((view_batch, num_rows, num_channels),
                            generator=generator,
                            dtype=torch.float32).to(model.torch_device)
    view_params = pf._view_params_per_dev[0][:view_batch]

    def call():
        return _cone_back_view_batch_triton(
            sino_batch, idx, view_params, coeff_power=1, slice_start=0,
            band_slices=band, plan=None, **args)

    names = []
    for index in range(result["launches"]):
        last = index == result["launches"] - 1
        if last and cfg.get("profile_names") and cuda:
            # torch.profiler names the kernel exactly as the CUDA runtime does,
            # which is what ncu's --kernel-name filter matches against.  Run on
            # the last launch only, and never under ncu.
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = call()
                torch.cuda.synchronize()
            for event in prof.key_averages():
                device_time = getattr(event, "self_device_time_total", None)
                if device_time is None:
                    device_time = getattr(event, "self_cuda_time_total", 0.0)
                if device_time:
                    names.append(dict(name=str(event.key),
                                      device_time_us=float(device_time)))
        else:
            out = call()
        out = None
    if cuda:
        torch.cuda.synchronize()
    names.sort(key=lambda entry: -entry["device_time_us"])
    result["profiler_kernel_names"] = names[:12]
    result["kernel_build"] = kernel_build_record()
    return result


def trivial_kernel():
    """One tiny CUDA kernel, for the profiler's permission probe.  Nothing here
    depends on mbirtorch."""
    import torch

    if not torch.cuda.is_available():
        return dict(mode="trivial_kernel", cuda=False)
    x = torch.ones(1 << 16, device="cuda")
    total = float((x * 2.0).sum())
    torch.cuda.synchronize()
    return dict(mode="trivial_kernel", cuda=True, checksum=total)


# ── the counter leg ───────────────────────────────────────────────────────────
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
    """One dict of metric name to value per profiled kernel.

    Two layouts are accepted because the two ncu pages disagree and the format
    has moved between releases.  The wide layout puts one metric per column and
    one kernel per row.  The long layout puts one metric per ROW, with the
    metric name and its value in their own columns.  A unit row directly under
    the header is skipped when it appears.  Numbers may carry thousands
    separators, which are stripped.
    """
    import csv as csv_module

    rows = list(csv_module.reader(text.splitlines()))

    def number(text_value):
        cleaned = str(text_value).replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return cleaned

    def parse_from(header_index):
        header = [cell_text.strip() for cell_text in rows[header_index]]
        body = rows[header_index + 1:]
        kernel_at = header.index("Kernel Name")
        if "Metric Name" in header and "Metric Value" in header:
            name_at = header.index("Metric Name")
            value_at = header.index("Metric Value")
            unit_at = (header.index("Metric Unit") if "Metric Unit" in header
                       else None)
            by_kernel = {}
            for row in body:
                if len(row) <= max(name_at, value_at, kernel_at):
                    continue
                key = row[kernel_at].strip()
                entry = by_kernel.setdefault(key, dict(kernel=key, metrics={},
                                                       units={}))
                entry["metrics"][row[name_at].strip()] = number(row[value_at])
                if unit_at is not None and len(row) > unit_at:
                    entry["units"][row[name_at].strip()] = row[unit_at].strip()
            return list(by_kernel.values())
        out, units = [], {}
        for row in body:
            if len(row) < len(header):
                continue
            values = [number(cell_text) for cell_text in row]
            if all(isinstance(value, str) for value in values):
                # The unit row directly under the header.  It is the only place
                # the wide layout says what its numbers are in, so it is kept
                # rather than skipped.
                if not units:
                    units = {name: str(value).strip()
                             for name, value in zip(header, values)
                             if name and str(value).strip()}
                continue
            entry = dict(kernel=row[kernel_at].strip(), metrics={},
                         units=dict(units))
            for name, value in zip(header, values):
                if name and name != "Kernel Name":
                    entry["metrics"][name] = value
            out.append(entry)
        return out

    # Several rows can carry "Kernel Name": ncu prints a header per page, and
    # some releases repeat it.  Every candidate is parsed and the parse that
    # recovered the most kernels, and then the most numbers, wins.  The choice
    # of header row is therefore not a guess about the release.
    best, best_score = [], (0, 0)
    for index, row in enumerate(rows):
        if not any(cell_text.strip() == "Kernel Name" for cell_text in row):
            continue
        try:
            parsed = parse_from(index)
        except (ValueError, IndexError):
            continue
        scored = [entry for entry in parsed
                  if any(isinstance(value, (int, float))
                         for value in entry["metrics"].values())]
        numbers = sum(1 for entry in scored for value in entry["metrics"].values()
                      if isinstance(value, (int, float)))
        if (len(scored), numbers) > best_score:
            best, best_score = scored, (len(scored), numbers)
    return best


def variant_env():
    """The environment a profiled subprocess runs under, set explicitly so
    nothing is inherited.

    Three variables are popped and one is set.  The device pin and the
    calibration mode would both change what the worker builds, and this run
    pins neither: it uses the single device the job allocates.  The Triton kill
    switch is set to the shipped value, because the kernel it would turn off is
    the whole subject of this run.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def ncu_leg(plan, torch_python, results_dir):
    """The optional counter leg: probe, discover, profile.

    Every step records what it saw.  A refusal, a missing ncu, an unmatched
    kernel name and a timeout are all findings on the record, and none of them
    changes the exit code.
    """
    profiled = ncu_variants(plan)
    leg = dict(attempted=True, enabled=NCU_ENABLED, pixel_div=NCU_PIXEL_DIV,
               launches=NCU_LAUNCHES,
               metrics_full=list(METRICS_FULL), metrics_core=list(METRICS_CORE),
               planned=[entry["variant"] for entry in profiled], variants={})
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG25_NCU=0")
        return leg
    if DEVICE != "cuda":
        leg.update(attempted=False,
                   reason="this run is on the CPU, where there are no GPU "
                          "performance counters to read")
        return leg
    ncu = shutil.which("ncu")
    leg["ncu_path"] = ncu
    if ncu is None:
        leg.update(attempted=False,
                   reason="ncu is not on PATH; the sbatch loads the cuda "
                          "module, which is where Nsight Compute lives")
        return leg
    version = _run([ncu, "--version"], 60)
    leg["ncu_version"] = (version["stdout"].strip().splitlines()[:2]
                          if version["stdout"] else version["stderr"][:200])

    # ── step 1: the permission probe ────────────────────────────────────────
    probe = _run([ncu, "--launch-count", "1", "--metrics",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  torch_python, "-u", os.path.abspath(__file__),
                  "--trivial-kernel"], NCU_PROBE_TIMEOUT_S)
    blob = (probe["stdout"] or "") + (probe["stderr"] or "")
    refused = any(marker.lower() in blob.lower()
                  for marker in NCU_PERMISSION_MARKERS)
    # The probe's own worker prints a result line, so a probe that produced no
    # metric can be told apart from one whose python never ran at all.  The two
    # have different remedies and reporting them as one would send a reader to
    # the wrong place.
    worker_ran = "__RESULT__" in blob
    empty = "sm__warps_active" not in blob
    leg["permission_probe"] = dict(
        returncode=probe["returncode"], timed_out=probe["timed_out"],
        refused=refused, profile_empty=empty, worker_ran=worker_ran,
        message=blob.strip()[-800:])
    leg["profiler_permitted"] = not (refused or empty or probe["timed_out"])
    if not leg["profiler_permitted"]:
        if refused:
            leg["reason"] = "the driver refused performance counters to this user"
        elif probe["timed_out"]:
            leg["reason"] = (f"the permission probe did not finish within "
                             f"{NCU_PROBE_TIMEOUT_S} s")
        elif not worker_ran:
            leg["reason"] = ("the probe's own python never ran, so this says "
                             "nothing about counter permission; read the "
                             "message on the row")
        else:
            leg["reason"] = ("the probe ran and produced no metric, so the "
                             "counters are unavailable")
        return leg

    # ── step 2: the kernel name, discovered at runtime ──────────────────────
    if not profiled:
        leg.update(reason="no variant is in this run's plan")
        return leg
    cfg = dict(variant=profiled[0]["variant"], band=profiled[0]["band"],
               pixel_div=NCU_PIXEL_DIV, profile_names=True, launches=2)
    plain = _run([torch_python, "-u", os.path.abspath(__file__),
                  "--one-launch", json.dumps(cfg)], NCU_TIMEOUT_S,
                 env=variant_env())
    row = _worker_result(plain["stdout"] or "")
    leg["discovery"] = dict(variant=cfg["variant"],
                            returncode=plain["returncode"],
                            error=(None if row else
                                   (plain["stderr"] or "")[-1500:]))
    candidates = []
    if row:
        leg["discovery"]["profiler_kernel_names"] = row.get(
            "profiler_kernel_names")
        leg["discovery"]["triton_names"] = (row.get("kernel_build") or {}).get(
            "names")
        leg["discovery"]["view_batch"] = row.get("view_batch")
        leg["discovery"]["launch_geometry"] = row.get("launch_geometry")
        for entry in row.get("profiler_kernel_names") or []:
            candidates.append(str(entry.get("name", "")))
        candidates.extend((row.get("kernel_build") or {}).get("names") or [])
    match = None
    for name in candidates:
        if "cone_back" in name.lower():
            match = name
            break
    leg["kernel_name"] = match
    if match is None:
        leg["reason"] = (
            "no kernel name reported by torch.profiler or by Triton's own "
            "cache contains 'cone_back'; the names seen are on the row, and a "
            "filter is not guessed")
        return leg
    # A bare name is used as a regular expression, so its regex characters are
    # escaped.  ncu matches the expression against the kernel's mangled name.
    pattern = "regex:" + re.escape(match)
    leg["kernel_name_filter"] = pattern

    # ── step 3: one warm launch per variant ─────────────────────────────────
    leg_start = time.perf_counter()
    leg["budget_s"] = NCU_LEG_BUDGET_S
    for entry in profiled:
        variant = entry["variant"]
        spent = time.perf_counter() - leg_start
        if spent > NCU_LEG_BUDGET_S:
            leg["variants"][variant] = dict(
                variant=variant, band=entry["band"],
                pixel_div=entry["pixel_div"],
                reason=f"the counter leg's {NCU_LEG_BUDGET_S} s budget was "
                       f"already spent ({spent:.0f} s); this variant was not "
                       "profiled")
            continue
        cfg = dict(variant=variant, band=entry["band"],
                   pixel_div=entry["pixel_div"], launches=NCU_LAUNCHES)
        record = dict(variant=variant, band=entry["band"],
                      pixel_div=entry["pixel_div"], note=entry["note"],
                      attempts=[])
        print(f"    {variant} (band {entry['band']}, pixel divisor "
              f"{entry['pixel_div']})", flush=True)
        for set_name, metrics, skip in NCU_ATTEMPTS:
            cmd = [ncu, "--csv", "--page", "raw", "--target-processes", "all",
                   "--kernel-name", pattern, "--launch-skip", str(skip),
                   "--launch-count", "1", "--metrics", ",".join(metrics),
                   torch_python, "-u", os.path.abspath(__file__),
                   "--one-launch", json.dumps(cfg)]
            got = _run(cmd, NCU_TIMEOUT_S, env=variant_env())
            log_path = os.path.join(
                results_dir, f"mg25_ncu_{variant}_{set_name}_skip{skip}.log")
            with open(log_path, "w") as log_sink:
                log_sink.write(" ".join(cmd) + "\n\n")
                log_sink.write(got["stdout"] or "")
                log_sink.write("\n----- stderr -----\n")
                log_sink.write(got["stderr"] or "")
            parsed = parse_ncu_csv(got["stdout"] or "")
            worker = _worker_result(got["stdout"] or "")
            record["attempts"].append(dict(
                metric_set=set_name, launch_skip=skip,
                returncode=got["returncode"], timed_out=got["timed_out"],
                wall_s=got["wall_s"], kernels=len(parsed), log=log_path))
            if parsed:
                record.update(metric_set=set_name, launch_skip=skip,
                              kernels=parsed, wall_s=got["wall_s"],
                              log=log_path, worker=worker)
                break
            if got["timed_out"]:
                # Retrying a timeout with a smaller metric set would spend the
                # same minutes again for the same reason.  MG25_NCU_PIXEL_DIV
                # is the knob that makes the profiled launch smaller.
                record["reason"] = (
                    f'the profile did not finish within {NCU_TIMEOUT_S} s; '
                    "re-run with a larger MG25_NCU_PIXEL_DIV to profile fewer "
                    "pixels")
                break
        if "kernels" not in record and "reason" not in record:
            record["reason"] = ("no kernel matched the filter, or the profile "
                                "was empty; the raw output is in the logs above")
        leg["variants"][variant] = record
    return leg


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=3):
    if value is None:
        return f'{"-":>{width}}'
    if isinstance(value, str):
        return f"{value:>{width}}"
    return f"{value:>{width}.{prec}{kind}}"


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


def _triton_registers(record):
    """Registers and spills per thread from Triton's own compile cache, as
    ``(registers, spills)``.

    Read from the profiled worker's result rather than from ncu, so it is
    there whichever metric set collected -- launch__registers_per_thread is a
    launch attribute that ncu does not list in --query-metrics, so it is in
    the full set only.  Several compiled variants may sit in one cache; the
    largest register count is taken, because that is the one that would cap
    occupancy.
    """
    entries = (((record or {}).get("worker") or {}).get("kernel_build")
               or {}).get("entries") or []
    registers = [entry.get("n_regs") for entry in entries
                 if isinstance(entry.get("n_regs"), (int, float))]
    spills = [entry.get("n_spills") for entry in entries
              if isinstance(entry.get("n_spills"), (int, float))]
    return (max(registers) if registers else None,
            max(spills) if spills else None)


def _duration_ms(kernel):
    """gpu__time_duration.sum in milliseconds, using the unit ncu reported.

    ncu emits this metric in nanoseconds, microseconds or milliseconds
    depending on the release and the page, so the unit is read rather than
    assumed.  An unrecognized unit returns None and the raw value stays on the
    row.
    """
    value = _metric(kernel, "gpu__time_duration.sum")
    if not isinstance(value, (int, float)):
        return None
    unit = ((kernel or {}).get("units") or {}).get("gpu__time_duration.sum")
    if unit is None:
        for key in (kernel.get("metrics") or {}):
            if key.startswith("gpu__time_duration.sum") and "(" in key:
                unit = key[key.index("(") + 1:key.rindex(")")]
    scale = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3,
             "msecond": 1.0, "ms": 1.0, "second": 1e3, "s": 1e3}
    factor = scale.get(str(unit).strip()) if unit is not None else None
    return None if factor is None else value * factor


def timing_table(header, rows):
    """Table 1: what each band cost through the production route."""
    print("\n===== table 1: the production route, one call per band =====")
    print(f'  cell {tuple(header.get("cell") or cell())}, '
          f'{header.get("num_pixels")} pixels, {TIMED_REPEATS} timed call'
          f'{"" if TIMED_REPEATS == 1 else "s"} after one discarded warm-up')
    print(f'  back body bound: {header.get("back_body")}; driver view batch '
          f'{header.get("driver_view_batch")}')
    line = (f'{"variant":<{VARIANT_COL}}{"band":>6}{"launch":>8}'
            f'{"median s":>11}{"spread":>9}{"ns/view-slice":>15}'
            f'{"vs 20400":>10}{"witness rel":>13}  check')
    print(line)
    print("-" * len(line))
    for row in rows:
        geometry = row.get("geometry") or {}
        launch = geometry.get("launch_band", row.get("launch_band"))
        if row.get("rate_ok") is None:
            check = "rate gate skipped"
        elif row.get("rate_ok"):
            check = "ok"
        else:
            check = f"RATE GATE FAILED (> {RATE_GATE_NS:.0f})"
        if not row.get("witness_ok"):
            check = "WITNESS FAILED; " + check
        print(f'{row.get("variant", "?"):<{VARIANT_COL}}'
              f'{int(row.get("band", 0)):>6}{int(launch or 0):>8}'
              f'{_fmt(row.get("wall_median_s"), 11, "f", 3)}'
              f'{_fmt(row.get("spread"), 9, "f", 4)}'
              f'{_fmt(row.get("ns_per_view_slice"), 15, "f", 1)}'
              f'{_fmt(row.get("rate_vs_anchor"), 10, "f", 2)}'
              f'{_fmt(row.get("witness_rel"), 13, "e", 2)}  {check}')
    print("-" * len(line))
    print(f"  'vs 20400' is the rate against the padded tree's measured rate "
          f"(mg23, findings section 1.23).  Reported, never gated.")
    print(f"  the gate is {RATE_GATE_NS:.0f} ns per view-slice: above it the "
          "padded wrapper did not deliver the divisible rate, and the counters "
          "below would describe the wrong kernel.")
    for row in rows:
        geometry = row.get("geometry") or {}
        if geometry.get("available") and not geometry.get("launch_band_equals_band"):
            print(f'  NOTE {row.get("variant")}: the launch band '
                  f'{geometry.get("launch_band")} is not the requested band '
                  f'{row.get("band")}; this run was designed so every band is '
                  "already a multiple of 16")


def counter_table(leg):
    """Table 2: the counters, one profiled launch per variant."""
    print("\n===== table 2: Nsight Compute counters, one warm launch per "
          "variant =====")
    if not leg.get("attempted"):
        print(f'  counter leg NOT ATTEMPTED: {leg.get("reason")}')
        return False
    if not leg.get("profiler_permitted"):
        probe = leg.get("permission_probe") or {}
        print("  profiler_permitted = false")
        print(f'  reason: {leg.get("reason")}')
        for line in (probe.get("message") or "").strip().splitlines()[-6:]:
            print(f"    {line}")
        print("  the timing leg stands alone; this is a recorded finding and "
              "not a failure")
        return False
    if not leg.get("variants"):
        print(f'  no variant profiled: {leg.get("reason")}')
        return False
    print(f'  kernel filter {leg.get("kernel_name_filter")}, pixel divisor '
          f'{leg.get("pixel_div")}, {leg.get("launches")} launches per worker')
    line = (f'{"variant":<{VARIANT_COL}}{"dur ms":>9}{"occup %":>9}'
            f'{"limiter":>20}{"reg/thr":>9}{"SM %":>8}{"mem %":>8}'
            f'{"L2 hit %":>10}{"L1 hit %":>10}{"sec/req":>9}{"stall":>8}'
            f'{"DRAM rd GB":>12}{"DRAM wr GB":>12}  set')
    print(line)
    print("-" * len(line))
    printed = False
    for variant, record in leg["variants"].items():
        kernels = record.get("kernels")
        if not kernels:
            print(f'{variant:<{VARIANT_COL}}  NO PROFILE: '
                  f'{str(record.get("reason", ""))[:90]}')
            continue
        kernel = kernels[0]
        limits = {
            "registers": _number(_metric(kernel,
                                         "launch__occupancy_limit_registers")),
            "shared mem": _number(_metric(kernel,
                                          "launch__occupancy_limit_shared_mem")),
            "blocks": _number(_metric(kernel,
                                      "launch__occupancy_limit_blocks"))}
        known = {name: value for name, value in limits.items()
                 if value is not None}
        limiter = min(known, key=known.get) if known else None
        limiter_text = f"{limiter} {known[limiter]:g}" if limiter else "-"
        read = _number(_metric(kernel, "dram__bytes_read.sum"))
        write = _number(_metric(kernel, "dram__bytes_write.sum"))
        print(f'{variant:<{VARIANT_COL}}'
              f'{_fmt(_duration_ms(kernel), 9, "f", 2)}'
              f'{_fmt(_number(_metric(kernel, "sm__warps_active.avg.pct_of_peak_sustained_active")), 9, "f", 1)}'
              f'{limiter_text:>20}'
              f'{_fmt(_number(_metric(kernel, "launch__registers_per_thread")), 9, "f", 0)}'
              f'{_fmt(_number(_metric(kernel, "sm__throughput.avg.pct_of_peak_sustained_elapsed")), 8, "f", 1)}'
              f'{_fmt(_number(_metric(kernel, "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")), 8, "f", 1)}'
              f'{_fmt(_number(_metric(kernel, "lts__t_sector_hit_rate.pct")), 10, "f", 1)}'
              f'{_fmt(_number(_metric(kernel, "l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct")), 10, "f", 1)}'
              f'{_fmt(_number(_metric(kernel, "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio")), 9, "f", 2)}'
              f'{_fmt(_number(_metric(kernel, "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio")), 8, "f", 2)}'
              f'{_fmt(read / 2 ** 30 if read is not None else None, 12, "f", 2)}'
              f'{_fmt(write / 2 ** 30 if write is not None else None, 12, "f", 2)}'
              f'  {record.get("metric_set", "-")}')
        printed = True
    print("-" * len(line))
    if printed:
        print("  A blank column means that metric was not in the set that "
              "collected; the 'set' column names which set that was, and only "
              "the full set carries the throughput, L1, stall and reg/thr "
              "columns.")
        print("  Durations here are ncu's, not wall times: ncu serializes and "
              "replays each kernel to collect its counters.  The timing leg "
              "owns time.")
        # The register count does not depend on ncu at all.  Printed
        # separately rather than folded into 'reg/thr', so the two sources are
        # never confused for one another.
        triton_lines = []
        for variant, record in leg["variants"].items():
            registers, spills = _triton_registers(record)
            if registers is not None:
                triton_lines.append(
                    f"    {variant:<{VARIANT_COL}} {registers:g} registers per "
                    f"thread"
                    + (f", {spills:g} spilled" if spills is not None else ""))
        if triton_lines:
            print("  registers per thread from Triton's own compile cache, "
                  "which is available whichever metric set collected:")
            for entry in triton_lines:
                print(entry)
    return printed


def gather_table(leg):
    """Table 3: the gather path, priced against what the kernel asks for.

    The kernel issues one global load per (pixel, slice, view, row tap, channel
    tap).  Along the slice axis of a channel-major sinogram those addresses are
    near unit stride, so eight consecutive float32 taps fill one 32-byte
    sector: a perfectly coalesced gather needs one sector per eight loads.  The
    ratio of the sectors ncu counted to that ideal is how far the gather path
    is from coalesced, and it is the number a sorted or reordered gather would
    move.
    """
    if not leg.get("variants"):
        return
    line = (f'{"variant":<{VARIANT_COL}}{"taps-loads":>13}'
            f'{"L1 ld sectors":>15}{"sec/ideal":>11}{"L2 rd sectors":>15}'
            f'{"DRAM rd/sino":>14}{"DRAM wr/partial":>17}'
            f'{"atom+red sectors":>18}')
    print("\n===== table 3: the gather path priced, per profiled launch =====")
    print(line)
    print("-" * len(line))
    for variant, record in leg["variants"].items():
        kernels = record.get("kernels")
        worker = record.get("worker") or {}
        if not kernels:
            continue
        kernel = kernels[0]
        band = int(record.get("band") or 0)
        pixels = worker.get("num_pixels_used")
        view_batch = worker.get("view_batch")
        taps = worker.get("taps")
        loads = None
        if pixels and view_batch and taps and band:
            loads = float(pixels) * band * float(view_batch) * float(taps) ** 2
        l1_sectors = _number(_metric(
            kernel, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"))
        over_ideal = (l1_sectors / (loads / 8.0)
                      if (l1_sectors is not None and loads) else None)
        l2_read = _number(_metric(kernel, "lts__t_sectors_op_read.sum"))
        dram_read = _number(_metric(kernel, "dram__bytes_read.sum"))
        dram_write = _number(_metric(kernel, "dram__bytes_write.sum"))
        sino_bytes = worker.get("sino_batch_bytes")
        partial_bytes = worker.get("out_partial_bytes")
        read_ratio = (dram_read / sino_bytes
                      if (dram_read is not None and sino_bytes) else None)
        write_ratio = (dram_write / partial_bytes
                       if (dram_write is not None and partial_bytes) else None)
        atom = _number(_metric(kernel, "lts__t_sectors_op_atom.sum"))
        red = _number(_metric(kernel, "lts__t_sectors_op_red.sum"))
        zero_witness = (None if (atom is None and red is None)
                        else (atom or 0.0) + (red or 0.0))
        print(f'{variant:<{VARIANT_COL}}'
              f'{_fmt(loads, 13, "e", 3)}'
              f'{_fmt(l1_sectors, 15, "e", 3)}'
              f'{_fmt(over_ideal, 11, "f", 2)}'
              f'{_fmt(l2_read, 15, "e", 3)}'
              f'{_fmt(read_ratio, 14, "f", 2)}'
              f'{_fmt(write_ratio, 17, "f", 2)}'
              f'{_fmt(zero_witness, 18, "e", 3)}')
    print("-" * len(line))
    print("  'taps-loads' is pixels x band x views x taps squared, the loads "
          "the kernel issues for live pixel lanes.  'sec/ideal' is the L1 load "
          "sectors over one sector per eight loads: 1.00 is a fully coalesced "
          "gather.")
    print("  'DRAM rd/sino' is the DRAM read traffic over the channel-major "
          "sinogram batch the launch reads; 1.00 means each byte was fetched "
          "once.  'DRAM wr/partial' is the same for the output partial it "
          "writes.")
    print("  'atom+red sectors' is the ZERO-WITNESS.  This kernel has no "
          "atomics -- it gathers taps and stores once per output element -- so "
          "these should read at or near zero.  A number here means the reading "
          "of the kernel is wrong somewhere and the columns to its left have "
          "to be re-examined.")


def reading_guide(rows, leg, profiled):
    """The three questions, and which columns answer each."""
    print("\n===== how to read the tables =====")
    print("\n(1) ARE THE GATHERS TRANSACTION-BOUND AT THE DIVISIBLE RATE?")
    print("    Table 1 first: every rate must sit in the divisible class "
          f"(the gate is {RATE_GATE_NS:.0f} ns per view-slice, and the padded "
          f"tree measured about {RATE_ANCHOR_NS:.0f}).  If it does not, stop; "
          "the counters describe a kernel production does not run.")
    if profiled:
        print("    Then table 2, columns 'mem %' against 'SM %' (both are a "
              "percent of speed of light), 'stall' (the average warps stalled "
              "on a long scoreboard per issue-active cycle), 'L2 hit %' and "
              "'L1 hit %'.  Memory throughput far above SM throughput, with a "
              "high stall ratio and low hit rates, is a kernel waiting on its "
              "loads.")
    else:
        print("    The counter columns did not collect, so this question is "
              "answered only as far as the rate: UNANSWERED beyond that.")

    print("\n(2) WHAT IS THE GATHER PATH'S SECTOR EFFICIENCY?")
    if profiled:
        print("    Table 2 column 'sec/req' (average sectors per global load "
              "request) and table 3 columns 'L1 ld sectors' and 'sec/ideal'.  "
              "The ideal is one 32-byte sector per eight float32 taps, because "
              "the gather walks the slice axis of a channel-major sinogram "
              "where consecutive slice lanes are consecutive addresses.")
        print("    Table 3's 'DRAM rd/sino' says how many times the launch "
              "re-fetched its sinogram batch from memory; a value near 1 means "
              "the caches held it.")
    else:
        print("    These counters are the counter leg's, and the leg did not "
              "produce them.  UNANSWERED.")

    print("\n(3) WHAT LIMITS OCCUPANCY?")
    if profiled:
        print("    Table 2 columns 'limiter' (the smallest of the three "
              "occupancy limits, named with its value), 'reg/thr' and "
              "'occup %'.  The tile is fixed at 16 x 64 with 4 warps and 1 "
              "stage, so the limiter should not move with the band; a limiter "
              "that does move means something other than the band changed "
              "between variants.")
        print("    The 'reg/thr' column is blank when only the core metric "
              "set collected, because ncu does not list that launch attribute "
              "in --query-metrics and it is therefore not in the set that must "
              "not fail.  The register count printed under table 2 comes from "
              "Triton's own compile cache instead, so this question is "
              "answered either way.")
    else:
        print("    The three occupancy limits are the counter leg's.  What is "
              "available without it is the launch shape and Triton's own "
              "register count, printed below.")
    geometry_lines = []
    for row in rows:
        geometry = row.get("geometry") or {}
        if geometry.get("available"):
            geometry_lines.append(
                f'    {row.get("variant"):<{VARIANT_COL}} band '
                f'{row.get("band")}, launch band {geometry.get("launch_band")}, '
                f'grid {geometry.get("grid")}, {geometry.get("blocks")} blocks, '
                f'tile {geometry.get("block_p")} x {geometry.get("block_l")}')
    if geometry_lines:
        print("    the launch shapes, derived from the library's own tile "
              "rule:")
        for entry in geometry_lines:
            print(entry)

    print("\n(4) THE ZERO-WITNESS.  Table 3's last column.  It should be at or "
          "near zero at every variant, and it is measured here rather than "
          "assumed.")

    print("\nWHAT WOULD SAY THERE IS HEADROOM for sorted or reordered "
          "gathers: 'sec/ideal' well above 1 together with a low L2 hit rate, "
          "a high long-scoreboard stall ratio, and memory throughput far above "
          "SM throughput.  That is a kernel spending its time fetching sectors "
          "it barely uses.")
    print("WHAT WOULD SAY THERE IS LITTLE: 'sec/ideal' near 1, or SM "
          "throughput at or above memory throughput, or occupancy capped by "
          "registers with the memory columns quiet.  Any one of those says the "
          "gathers are not where the time goes.")
    print("\nTHE VERDICT IS READ BY A PERSON from the tables above.  This run "
          "decides nothing and implements nothing.  The B3 remedy (in-kernel "
          "sorted or segmented accumulation) is NOT started here, and acting "
          "on whatever these columns say needs Greg's approval.")


def summarize(header, rows, plan, leg, out_path):
    """The tables a person reads the answer from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and this function keeps them apart.  No
    verdict is computed here.  What is computed here is whether the
    measurement was taken correctly.
    """
    print(f"\n===== mg25 cone back counters ({out_path}) =====")
    timing_table(header, rows)
    profiled = counter_table(leg)
    if profiled:
        gather_table(leg)
    reading_guide(rows, leg, profiled)

    missing = [entry["variant"] for entry in plan
               if entry["variant"] not in {row.get("variant") for row in rows}]
    witness_failed = [row.get("variant") for row in rows
                      if not row.get("witness_ok")]
    rate_failed = [row.get("variant") for row in rows
                   if row.get("rate_ok") is False]
    checks = []
    if missing:
        checks.append(f"variants with no timing row: {missing}")
    if witness_failed:
        checks.append(f"variants whose witness exceeded {WITNESS_REL:g}: "
                      f"{witness_failed}")
    if rate_failed:
        checks.append(f"variants above the {RATE_GATE_NS:.0f} ns per "
                      f"view-slice rate gate: {rate_failed}")
    if not header.get("bodies_ok"):
        checks.append(f'torch bodies {header.get("torch_body_directions")} '
                      f'against the expected '
                      f'{header.get("torch_body_expected")}')
    if not header.get("device_ok"):
        checks.append(f'the realized device is {header.get("device_realized")} '
                      f'and not {header.get("device_expected")}')

    if SMOKE:
        print("\nsmoke: the rate gate is SKIPPED and recorded.  The torch body "
              "runs here and its rate is not this kernel's.")
    if header.get("num_pixels_matches_expected") is False:
        print(f'\nNOTE: the pixel mask holds {header.get("num_pixels")} '
              f'pixels, not the recorded {NUM_PIXELS_EXPECTED}.  Recorded, and '
              "it does not gate anything.")
    if header.get("gpu_hot_or_throttled"):
        print("\nNOTE: the device sampled hot or throttled.  The rates are an "
              "efficiency reading, so read them with that in mind.")
    if header.get("peak_bytes"):
        print(f'\ntiming leg peak device memory: '
              f'{header["peak_bytes"] / 2 ** 30:.1f} GiB')
    unprofiled = [name for name, record in (leg.get("variants") or {}).items()
                  if not record.get("kernels")]
    if unprofiled:
        print(f"\n{len(unprofiled)} variant(s) produced no counter row: "
              f"{unprofiled}.  The counter leg never changes the exit code.")

    healthy = not checks
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"}.  It covers five things: '
          "every selected variant produced a timing row, every witness was at "
          f"or below {WITNESS_REL:g}, every rate was at or below "
          f"{RATE_GATE_NS:.0f} ns per view-slice (skipped in the smoke), the "
          "model bound no torch body on CUDA, and the realized device is the "
          "one asked for.")
    for line in checks:
        print(f"  FAIL: {line}")
    print("The counter leg's absence never changes it, and neither does what "
          "either leg found.  The verdict is read by a person from the tables "
          "above and the rows in the jsonl.")
    return dict(kind="summary", healthy=healthy, checks=checks,
                missing=missing, witness_failed=witness_failed,
                rate_failed=rate_failed,
                bodies_ok=header.get("bodies_ok"),
                device_ok=header.get("device_ok"),
                gpu_hot_or_throttled=header.get("gpu_hot_or_throttled"),
                profiler_permitted=leg.get("profiler_permitted"),
                profiler_reason=leg.get("reason"),
                unprofiled=unprofiled, variants=len(rows),
                out_path=out_path)


# ── the runner ────────────────────────────────────────────────────────────────
def _dry_run(plan):
    print(f"mg25 cone back counters: {len(plan)} variant(s), device {DEVICE}, "
          f"cell {tuple(cell())}")
    print(f"  the counter run named as open item B3's precondition "
          f"(back_remedy_design.md section 6).  It decides nothing.")
    print(f"  results, and one ncu log per attempt -> {RESULTS_DIR}")
    print(f'  {"variant":<{VARIANT_COL}}{"band":>6}{"launch":>8}{"div16":>7}  '
          f"note")
    for entry in plan:
        print(f'  {entry["variant"]:<{VARIANT_COL}}{entry["band"]:>6}'
              f'{entry["launch_band"]:>8}'
              f'{str(entry["divisible_by_16"]):>7}  {entry["note"]}')
    print("  every band above is already a multiple of 16, so the wrapper's "
          "padded_kernel_width returns it unchanged and the kernel launches at "
          "exactly that length.  The launch bands here come from this file's "
          "mirror of that rule; the real run reads the library's own function "
          "and records whether the two agreed.")
    print(f"\n  timing leg: {WARMUP_REPEATS} warm + {TIMED_REPEATS} timed "
          "call(s) per variant through "
          "Projectors.sparse_back_project_view_range")
    print(f"  witness gate {WITNESS_REL:.0e} relative, against the largest "
          "band's output on the slices they share")
    print(f"  rate gate {RATE_GATE_NS:.0f} ns per view-slice"
          + (" (SKIPPED in the smoke)" if SMOKE else "")
          + f"; the padded tree's anchor is {RATE_ANCHOR_NS:.0f}")
    print(f"\n  counter leg: {'on' if NCU_ENABLED else 'off (MG25_NCU=0)'}")
    if NCU_ENABLED:
        for entry in ncu_variants(plan):
            print(f'    {entry["variant"]:<{VARIANT_COL}}band '
                  f'{entry["band"]:>5}  pixel divisor {entry["pixel_div"]}')
        print(f"    {NCU_LAUNCHES} launches per worker; attempts per variant, "
              "in order:")
        for set_name, metrics, skip in NCU_ATTEMPTS:
            print(f"      metric set {set_name} ({len(metrics)} metrics), "
                  f"page raw, launch skip {skip}")
        print(f"    bounds: {NCU_TIMEOUT_S} s per attempt, "
              f"{NCU_LEG_BUDGET_S} s for the whole leg, "
              f"{NCU_PROBE_TIMEOUT_S} s for the permission probe")
        print("    the counter leg never changes the exit code")
    print("\n  exit code = instrument health: every variant timed, every "
          "witness in gate, every rate in gate (skipped in the smoke), no "
          "torch body on CUDA, and the realized device is the one asked for")
    print("  no library file is touched: the timing leg calls the production "
          "route and the counter leg calls the shipped kernel wrapper")


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    if not SMOKE:
        import torch
        if not torch.cuda.is_available():
            print("this run needs CUDA; use MG25_SMOKE=1 for the CPU plumbing "
                  "pass")
            return 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        RESULTS_DIR, f"mg25_back_counters_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg25 cone back counters on {RUN_LABEL} ({DEVICE}); "
          f"{len(plan)} variant(s) -> {out_path}", flush=True)
    with open(out_path, "w") as sink:
        print("  timing leg", flush=True)
        header, rows = timing_leg(plan, sink)
        print("  counter leg", flush=True)
        leg = ncu_leg(plan, sys.executable, RESULTS_DIR)
        sink.write(json.dumps(dict(kind="ncu_leg", **leg)) + "\n")
        sink.flush()
        summary = summarize(header, rows, plan, leg, out_path)
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
            worker_out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(worker_out))
    elif len(sys.argv) > 1 and sys.argv[1] == "--trivial-kernel":
        print("__RESULT__" + json.dumps(trivial_kernel()))
    else:
        sys.exit(main())
