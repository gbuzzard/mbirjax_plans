"""mg31 -- THE COUNTER RUN ON THE CONE FORWARD KERNEL.

WHY THIS RUN EXISTS.  The segmented-accumulation design note (open item B7's
follow-on) proposes replacing the forward kernels' per-tap atomic scatter
with in-tile accumulation, and the two forward kernels share that scatter:
the parallel kernel is the cone forward's horizontal scatter with the
vertical fan deleted, and the cone kernel's own comment calls its version
"the plain per-tap atomic form".  mg28 measured the parallel forward
memory-side with the atomic path binding.  Whether the CONE forward is also
atomic-bound is unmeasured, and its situation differs: the vertical tap
loop gives cone much more arithmetic per atomic add, so the atomic share is
diluted and the kernel may sit compute-side, as the cone BACK kernel does
(findings §1.24).  This run takes that reading, so the design note commits
to cone on a measurement rather than an analogy.  Greg ordered the run
2026-08-18.

THE THREE READINGS, mg28's:
  1. Compute-side or memory-side?  (SM against memory speed-of-light
     throughput, the long-scoreboard stall ratio)
  2. The two memory paths.  (the atomic-path sectors against their
     coalesced ideal; the load path reported raw -- the cone gather's tap
     structure differs from parallel's and its ideal is not priced here,
     because the design note's question is the write side)
  3. The occupancy limiter.  (limits, registers per thread, achieved)

THE CELL is the 1024-class cone of the comparison table: sinogram
(1024, 1008, 992), mg1's construction, recon (992, 992, 1008), full mask.
The values block is (pixels, 1008 slices), seeded.  The view batch per
launch is the driver's own rule for this body, read at run time and
recorded, not pinned: the cone forward's transient budget sets it.

TWO VARIANTS through the SHIPPED wrapper (no launch code is copied): the
full pixel mask, and the mask subsampled 64x -- the most common VCD call
size.  Two variants reading the same intensities is the internal
consistency check; no single-launch cone anchor exists in the record, so
the timing rows are descriptive and the exit code gates only on the
instruments (the Triton forward bound, the realized device, both timing
rows produced).  The counter leg never gates.

Run:
    <torch python> mg31_cfwd_counters.py           on one GPU
    MG31_DRY=1 <python> mg31_cfwd_counters.py      print the plan and stop
    MG31_SMOKE=1 <python> mg31_cfwd_counters.py    tiny CPU plumbing pass

Configuration is by environment variable only; there is no command line.
    MG31_RESULTS=<dir>      where the jsonl and the ncu logs go
    MG31_SMOKE=1 / MG31_DRY=1
    MG31_NCU=0              skip the counter leg entirely
    MG31_REPEATS=3          timed repeats in the timing leg
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
SMOKE = os.environ.get("MG31_SMOKE", "0") == "1"
DRY = os.environ.get("MG31_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

CELL = (1024, 1008, 992)          # (views, detector rows, channels)
SMOKE_CELL = (8, 24, 20)
RECON_SHAPE_EXPECTED = (992, 992, 1008)
NUM_PIXELS_EXPECTED = 771240      # recorded, not gated

#: The view batch every launch uses: the forward kernel's nominal view chunk
#: and the driver's choice at this cell (mg20 pinned the same value).
VIEW_BATCH = 128
VALUES_SEED = 20260818

WARMUP_REPEATS = 1
TIMED_REPEATS = max(1, int(os.environ.get("MG31_REPEATS", "3")))

#: mg20's measured full-pixel launch at width 1008 on an H100 (job 15316533,
#: rows/mg20_width_h007_20260817_092645.jsonl): median 859.13 ms.  The gate
#: is a 25 percent window around it -- an instrument check that the profiled
#: kernel is the recorded kernel, not a finding.  Skipped in the smoke.
ANCHOR_FULLPIX_MS = 859.13
ANCHOR_WINDOW = 0.25

#: The two profiled variants: (name, pixel divisor).  The divisor subsamples
#: the full mask by stride, mg20's construction.  64 gives 12,051 pixels,
#: the most common VCD call size in the production mixture (section 1.9).
NCU_VARIANTS = (("cfwd_fullpix", 1), ("cfwd_sub64", 64))

# ── the counter leg (mg25's machinery) ────────────────────────────────────────
NCU_ENABLED = os.environ.get("MG31_NCU", "1") == "1"
NCU_LAUNCHES = 5
NCU_TIMEOUT_S = 420
NCU_LEG_BUDGET_S = 1200
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
#: The set that must not fail: the names mg20 and mg25 both collected on this
#: cluster, plus lts__t_sectors_op_read.sum (proven by mg25).
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
NCU_ATTEMPTS = (
    ("full", METRICS_FULL, NCU_LAUNCHES - 1),
    ("core", METRICS_CORE, NCU_LAUNCHES - 1),
    ("core", METRICS_CORE, 0),
)
NCU_PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                          "insufficient permission")

HOT_CORE_C = 85
HOT_HBM_C = 95

RESULTS_DIR = os.environ.get(
    "MG31_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
VARIANT_COL = 15
# ──────────────────────────────────────────────────────────────────────────────


def cell():
    return SMOKE_CELL if SMOKE else CELL


def width():
    """The values-block width: the reconstruction's slice count, which under
    parallel beam equals the detector row count."""
    return int(cell()[1])


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
                        throttle=[name for name, value
                                  in zip(names, parts[4:8])
                                  if value.lower() == "active"]))
    return out


def health_is_hot(health):
    return any((g.get("temp_c") or 0) >= HOT_CORE_C
               or (g.get("mem_temp_c") or 0) >= HOT_HBM_C
               or g.get("throttle") for g in health)


# ── the model (mg20's construction) ───────────────────────────────────────────
def build_model():
    """mg20's construction at mg20's cell: half a turn is a full
    parallel-beam scan.  ``skip_memory_preflight`` is set because this run
    allocates a values block and one sinogram view batch, not a
    reconstruction."""
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


def _seeded_values(torch_module, model, num_pixels, columns):
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(VALUES_SEED)
    block = torch_module.rand((int(num_pixels), int(columns)),
                              generator=generator,
                              dtype=torch_module.float32)
    return block.to(model.torch_device)


def _subsample(idx_full, divisor):
    """mg20's subset construction: every ``stride``-th pixel of the mask."""
    full = int(idx_full.shape[0])
    want = max(1, full // max(1, int(divisor)))
    step = max(1, full // want)
    return idx_full[::step][:want].contiguous()


def kernel_build_record():
    """Registers, spills and names from Triton's own cache for the parallel
    forward kernel -- the occupancy question's ncu-independent half, and the
    name candidates for the ncu filter (mg20's reader, defensively)."""
    try:
        from mbirtorch.triton_cone import _cone_forward_kernel as fwd
    except Exception as exc:                                      # noqa: BLE001
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    entries, names = [], []
    caches = []
    for attr in ("cache", "device_caches"):
        holder = getattr(fwd, attr, None)
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
                    if got is not None and not isinstance(got, (int, float,
                                                                str)):
                        got = str(got)
                    record[field] = got
                if record.get("name"):
                    names.append(str(record["name"]))
                entries.append(record)
    return dict(available=bool(entries), entries=entries[:24],
                names=sorted(set(names)),
                python_name=getattr(fwd, "__name__", None))


# ── the timing leg ────────────────────────────────────────────────────────────
def timing_leg(sink):
    """One warm plus TIMED_REPEATS timed full-pixel launches through the
    shipped wrapper, in this process, with the anchor gate."""
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    header = dict(kind="run", smoke=SMOKE, device=DEVICE, cell=list(cell()),
                  width=width(), view_batch=VIEW_BATCH,
                  values_seed=VALUES_SEED, warmup=WARMUP_REPEATS,
                  timed=TIMED_REPEATS,
                  anchor_fullpix_ms=ANCHOR_FULLPIX_MS,
                  anchor_window=ANCHOR_WINDOW,
                  anchor_applies=not SMOKE,
                  ncu_enabled=NCU_ENABLED,
                  torch=torch.__version__, node=platform.node(),
                  cuda=cuda, run_label=RUN_LABEL,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  env_disable_triton=os.environ.get(
                      "MBIRTORCH_DISABLE_TRITON"),
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
    header["triton_forward_bound"] = fwd_body.__name__.endswith("_triton")

    device = model.torch_device
    header["device_realized"] = str(device)
    header["device_expected"] = "cuda:0" if cuda else DEVICE
    header["device_ok"] = (str(device) == header["device_expected"])
    header["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]
    header["recon_shape_matches_expected"] = (
        None if SMOKE
        else tuple(header["recon_shape"]) == RECON_SHAPE_EXPECTED)

    args = model._view_batch_args()
    header["psf_radius"] = int(args["psf_radius"])
    header["taps"] = 2 * int(args["psf_radius"]) + 1
    pf = model.projector_functions
    idx_full = model.full_indices_device()
    view_batch = min(int(pf._effective_view_batch(
        fwd_body, int(idx_full.shape[0]), width(), args)),
        int(cell()[0]))
    header["view_batch_used"] = view_batch
    view_params = pf._view_params_per_dev[0][:view_batch]
    header["num_pixels_full"] = int(idx_full.shape[0])
    header["num_pixels_matches_expected"] = (
        None if SMOKE else int(idx_full.shape[0]) == NUM_PIXELS_EXPECTED)
    values = _seeded_values(torch, model, header["num_pixels_full"], width())

    def launch():
        out = fwd_body(values, idx_full, view_params, slice_start=0,
                       plan=None, **args)
        if cuda:
            torch.cuda.synchronize(device)
        return out

    if cuda:
        torch.cuda.reset_peak_memory_stats()
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
        out = launch()
        walls.append(time.perf_counter() - start)
    del out
    row = dict(kind="timing", warmup_s=walls[:WARMUP_REPEATS],
               timed_s=walls[WARMUP_REPEATS:])
    row["median_ms"] = statistics.median(row["timed_s"]) * 1e3
    row["ms_per_slice"] = row["median_ms"] / width()
    row["spread"] = ((max(row["timed_s"]) - min(row["timed_s"]))
                     / statistics.median(row["timed_s"]))
    # No single-launch cone forward anchor exists in the record, so the
    # timing row is descriptive and never gates; the smoke's torch-body
    # rate is not this kernel's either way.
    row["anchor_ok"] = None
    row["anchor_skipped"] = "no recorded single-launch cone forward anchor"
    print(f'  full-pixel launch: median {row["median_ms"]:8.2f} ms '
          f'({row["ms_per_slice"]:.4f} ms per slice), '
          f'spread {row["spread"]:.1%}', flush=True)

    del values
    del idx_full
    if cuda:
        torch.cuda.empty_cache()
        header["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    header["health_after"] = sample_gpu_health()
    header["gpu_hot_or_throttled"] = bool(
        health_is_hot(header.get("health_before") or [])
        or health_is_hot(header["health_after"]))
    sink.write(json.dumps(header) + "\n")
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return header, row


# ── the single-launch worker the profiler drives ──────────────────────────────
def one_launch(cfg):
    """Build the model, launch the shipped forward wrapper a few times, exit.

    THE BODY IS IMPORTED DIRECTLY rather than taken from
    ``model._view_batch_bodies()``: that selection runs the availability
    self-check, which launches this same kernel once on a tiny problem, and
    the skip-0 fallback attempt would then profile that tiny launch.  Which
    body the model binds is witnessed by the timing leg.
    """
    import torch

    from mbirtorch.triton_cone import _cone_forward_view_batch_triton

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, mode="one_launch", device=DEVICE, cuda=cuda,
                  launches=int(cfg.get("launches", NCU_LAUNCHES)))
    model = build_model()
    args = model._view_batch_args()
    pf = model.projector_functions
    idx = _subsample(model.full_indices_device(),
                     int(cfg.get("pixel_div", 1)))
    used = int(idx.shape[0])
    # The driver's own view-batch rule for this body at this pixel count,
    # read through the wrapper's _view_batch_cost attribute; the cone
    # forward's transient budget sets it, so it is not pinned here.
    view_batch = min(int(pf._effective_view_batch(
        _cone_forward_view_batch_triton, used, width(), args)),
        int(cell()[0]))
    view_params = pf._view_params_per_dev[0][:view_batch]
    result["num_pixels_used"] = used
    result["view_batch"] = view_batch
    result["psf_radius"] = int(args["psf_radius"])
    result["taps"] = 2 * int(args["psf_radius"]) + 1
    result["width"] = width()
    result["values_bytes"] = used * width() * 4
    result["out_slab_bytes"] = (view_batch * int(args["num_channels"])
                                * width() * 4)
    values = _seeded_values(torch, model, used, width())

    def call():
        return _cone_forward_view_batch_triton(
            values, idx, view_params, slice_start=0, plan=None, **args)

    names = []
    for index in range(result["launches"]):
        last = index == result["launches"] - 1
        if last and cfg.get("profile_names") and cuda:
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
    import torch

    if not torch.cuda.is_available():
        return dict(mode="trivial_kernel", cuda=False)
    x = torch.ones(1 << 16, device="cuda")
    total = float((x * 2.0).sum())
    torch.cuda.synchronize()
    return dict(mode="trivial_kernel", cuda=True, checksum=total)


# ── the counter leg (mg25's, retargeted) ──────────────────────────────────────
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
    """mg20's parser: both ncu CSV layouts, best header wins."""
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
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def ncu_leg(results_dir):
    """Probe, discover, profile -- mg25's leg with the forward kernel's name
    filter.  Every step records what it saw; nothing here gates."""
    leg = dict(attempted=True, enabled=NCU_ENABLED, launches=NCU_LAUNCHES,
               metrics_full=list(METRICS_FULL),
               metrics_core=list(METRICS_CORE),
               planned=[name for name, _d in NCU_VARIANTS], variants={})
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG31_NCU=0")
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

    probe = _run([ncu, "--launch-count", "1", "--metrics",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  sys.executable, "-u", os.path.abspath(__file__),
                  "--trivial-kernel"], NCU_PROBE_TIMEOUT_S)
    blob = (probe["stdout"] or "") + (probe["stderr"] or "")
    refused = any(m.lower() in blob.lower() for m in NCU_PERMISSION_MARKERS)
    worker_ran = "__RESULT__" in blob
    empty = "sm__warps_active" not in blob
    leg["permission_probe"] = dict(
        returncode=probe["returncode"], timed_out=probe["timed_out"],
        refused=refused, profile_empty=empty, worker_ran=worker_ran,
        message=blob.strip()[-800:])
    leg["profiler_permitted"] = not (refused or empty or probe["timed_out"])
    if not leg["profiler_permitted"]:
        leg["reason"] = (
            "the driver refused performance counters to this user" if refused
            else "the permission probe did not finish in time"
            if probe["timed_out"]
            else "the probe's own python never ran" if not worker_ran
            else "the probe ran and produced no metric")
        return leg

    cfg = dict(variant=NCU_VARIANTS[0][0], pixel_div=NCU_VARIANTS[0][1],
               profile_names=True, launches=2)
    plain = _run([sys.executable, "-u", os.path.abspath(__file__),
                  "--one-launch", json.dumps(cfg)], NCU_TIMEOUT_S,
                 env=variant_env())
    row = _worker_result(plain["stdout"] or "")
    leg["discovery"] = dict(returncode=plain["returncode"],
                            error=(None if row
                                   else (plain["stderr"] or "")[-1500:]))
    candidates = []
    if row:
        leg["discovery"]["profiler_kernel_names"] = row.get(
            "profiler_kernel_names")
        leg["discovery"]["triton_names"] = (row.get("kernel_build")
                                            or {}).get("names")
        for entry in row.get("profiler_kernel_names") or []:
            candidates.append(str(entry.get("name", "")))
        candidates.extend((row.get("kernel_build") or {}).get("names") or [])
    match = next((n for n in candidates if "cone_forward" in n.lower()),
                 None)
    leg["kernel_name"] = match
    if match is None:
        leg["reason"] = ("no kernel name reported by torch.profiler or by "
                         "Triton's own cache contains 'parallel_forward'; a "
                         "filter is not guessed")
        return leg
    pattern = "regex:" + re.escape(match)
    leg["kernel_name_filter"] = pattern

    leg_start = time.perf_counter()
    leg["budget_s"] = NCU_LEG_BUDGET_S
    for name, divisor in NCU_VARIANTS:
        spent = time.perf_counter() - leg_start
        if spent > NCU_LEG_BUDGET_S:
            leg["variants"][name] = dict(
                variant=name, pixel_div=divisor,
                reason=f"the counter leg's {NCU_LEG_BUDGET_S} s budget was "
                       f"already spent ({spent:.0f} s)")
            continue
        cfg = dict(variant=name, pixel_div=divisor, launches=NCU_LAUNCHES)
        record = dict(variant=name, pixel_div=divisor, attempts=[])
        print(f"    {name} (pixel divisor {divisor})", flush=True)
        for set_name, metrics, skip in NCU_ATTEMPTS:
            cmd = [ncu, "--csv", "--page", "raw", "--target-processes",
                   "all", "--kernel-name", pattern, "--launch-skip",
                   str(skip), "--launch-count", "1", "--metrics",
                   ",".join(metrics), sys.executable, "-u",
                   os.path.abspath(__file__), "--one-launch",
                   json.dumps(cfg)]
            got = _run(cmd, NCU_TIMEOUT_S, env=variant_env())
            log_path = os.path.join(
                results_dir, f"mg31_ncu_{name}_{set_name}_skip{skip}.log")
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
                record["reason"] = (f"the profile did not finish within "
                                    f"{NCU_TIMEOUT_S} s")
                break
        if "kernels" not in record and "reason" not in record:
            record["reason"] = ("no kernel matched the filter, or the "
                                "profile was empty; the raw output is in "
                                "the logs above")
        leg["variants"][name] = record
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


def _duration_ms(kernel):
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


def counter_table(leg):
    print("\n===== table 1: Nsight Compute counters, one warm launch per "
          "variant =====")
    if not leg.get("attempted"):
        print(f'  counter leg NOT ATTEMPTED: {leg.get("reason")}')
        return False
    if not leg.get("profiler_permitted"):
        print(f'  profiler_permitted = false: {leg.get("reason")}')
        return False
    if not leg.get("variants"):
        print(f'  no variant profiled: {leg.get("reason")}')
        return False
    print(f'  kernel filter {leg.get("kernel_name_filter")}, '
          f'{leg.get("launches")} launches per worker')
    line = (f'{"variant":<{VARIANT_COL}}{"dur ms":>9}{"occup %":>9}'
            f'{"limiter":>20}{"reg/thr":>9}{"SM %":>8}{"mem %":>8}'
            f'{"L2 hit %":>10}{"L1 hit %":>10}{"sec/req":>9}{"stall":>8}'
            f'{"DRAM rd GB":>12}{"DRAM wr GB":>12}  set')
    print(line)
    print("-" * len(line))
    printed = False
    for name, record in leg["variants"].items():
        kernels = record.get("kernels")
        if not kernels:
            print(f'{name:<{VARIANT_COL}}  NO PROFILE: '
                  f'{str(record.get("reason", ""))[:90]}')
            continue
        kernel = kernels[0]
        limits = {
            "registers": _number(_metric(
                kernel, "launch__occupancy_limit_registers")),
            "shared mem": _number(_metric(
                kernel, "launch__occupancy_limit_shared_mem")),
            "blocks": _number(_metric(
                kernel, "launch__occupancy_limit_blocks"))}
        known = {k: v for k, v in limits.items() if v is not None}
        limiter = min(known, key=known.get) if known else None
        limiter_text = f"{limiter} {known[limiter]:g}" if limiter else "-"
        read = _number(_metric(kernel, "dram__bytes_read.sum"))
        write = _number(_metric(kernel, "dram__bytes_write.sum"))
        print(f'{name:<{VARIANT_COL}}'
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
        print("  Durations here are ncu's, not wall times: ncu serializes "
              "and replays each kernel to collect its counters.  The timing "
              "leg owns time.")
        for name, record in leg["variants"].items():
            build = ((record.get("worker") or {}).get("kernel_build")
                     or {})
            for entry in build.get("entries") or []:
                if entry.get("n_regs") is not None:
                    print(f"  {name:<{VARIANT_COL}} "
                          f"{entry['n_regs']:g} registers per thread from "
                          f"Triton's own compile cache"
                          + (f", {entry['n_spills']:g} spilled"
                             if entry.get("n_spills") is not None else ""))
                    break
    return printed


def traffic_table(leg):
    """Table 2: both memory paths priced, with the VIEW FACTOR in.

    The kernel's grid is (pixel blocks, column blocks, views), so per launch
    it loads the values tile once per (pixel, column, VIEW) and issues one
    atomic add per (pixel, column, live channel tap, VIEW).  mg20's pricing
    omitted the view factor, which overstated its sectors-over-ideal by the
    128-view batch; the formulas here carry it, and the columns say which
    path the traffic actually takes.
    """
    if not leg.get("variants"):
        return
    line = (f'{"variant":<{VARIANT_COL}}{"atomics":>12}{"atom+red sec":>14}'
            f'{"atom/ideal":>12}{"loads":>12}{"L1 ld sec":>12}'
            f'{"ld/ideal":>10}{"L2 rd sec":>12}{"DRAM rd/vals":>14}'
            f'{"DRAM wr/slab":>14}')
    print("\n===== table 2: the two memory paths priced, per profiled "
          "launch =====")
    print(line)
    print("-" * len(line))
    for name, record in leg["variants"].items():
        kernels = record.get("kernels")
        worker = record.get("worker") or {}
        if not kernels:
            continue
        kernel = kernels[0]
        pixels = worker.get("num_pixels_used")
        vb = worker.get("view_batch")
        taps = worker.get("taps")
        w = worker.get("width")
        atomics = loads = None
        if pixels and vb and taps and w:
            atomics = float(vb) * float(pixels) * float(taps) * float(w)
            loads = float(vb) * float(pixels) * float(w)
        atom = _number(_metric(kernel, "lts__t_sectors_op_atom.sum")) or 0.0
        red = _number(_metric(kernel, "lts__t_sectors_op_red.sum")) or 0.0
        atom_sectors = atom + red
        atom_ratio = (atom_sectors / (atomics / 8.0)
                      if atomics else None)
        l1_ld = _number(_metric(
            kernel, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"))
        ld_ratio = (l1_ld / (loads / 8.0)
                    if (l1_ld is not None and loads) else None)
        l2_rd = _number(_metric(kernel, "lts__t_sectors_op_read.sum"))
        dram_rd = _number(_metric(kernel, "dram__bytes_read.sum"))
        dram_wr = _number(_metric(kernel, "dram__bytes_write.sum"))
        vals_bytes = worker.get("values_bytes")
        slab_bytes = worker.get("out_slab_bytes")
        rd_ratio = (dram_rd / vals_bytes
                    if (dram_rd is not None and vals_bytes) else None)
        wr_ratio = (dram_wr / slab_bytes
                    if (dram_wr is not None and slab_bytes) else None)
        print(f'{name:<{VARIANT_COL}}'
              f'{_fmt(atomics, 12, "e", 3)}'
              f'{_fmt(atom_sectors, 14, "e", 3)}'
              f'{_fmt(atom_ratio, 12, "f", 2)}'
              f'{_fmt(loads, 12, "e", 3)}'
              f'{_fmt(l1_ld, 12, "e", 3)}'
              f'{_fmt(ld_ratio, 10, "f", 2)}'
              f'{_fmt(l2_rd, 12, "e", 3)}'
              f'{_fmt(rd_ratio, 14, "f", 2)}'
              f'{_fmt(wr_ratio, 14, "f", 2)}')
    print("-" * len(line))
    print("  'atomics' is view_batch x pixels x taps x width, the adds the "
          "kernel issues for live lanes; edge taps whose mask drops them "
          "make the true count slightly smaller.  'atom/ideal' is the "
          "atomic-path sectors over one 32-byte sector per eight adds: 1.00 "
          "is a fully coalesced write path.")
    print("  'loads' and 'ld/ideal' use the PARALLEL kernel's pricing "
          "(one values read per pixel, column, view).  The cone gather "
          "reads values once per slice TAP, about three times that, so "
          "its ld/ideal reads high by that factor and is reported for "
          "scale only; this run's question is the write side.")
    print("  'DRAM rd/vals' is DRAM read over the values block's bytes: how "
          "many times the block was fetched from memory.  'DRAM wr/slab' is "
          "DRAM write over the output view-batch slab.")


def reading_guide(leg, profiled):
    print("\n===== how to read the tables =====")
    print("\n(1) ATOMIC-BOUND, LOAD-BOUND, OR COMPUTE-BOUND?")
    print("    Table 1 'SM %' against 'mem %' says whether the kernel is "
          "closer to compute or to memory speed of light; 'stall' is the "
          "average warps waiting on a long scoreboard per issue-active "
          "cycle.  Table 2 splits the memory side: the atomic columns "
          "against the load columns say which path carries the sectors.")
    print("    Memory-side and atomic-heavy: the write path binds, and "
          "in-kernel segmented accumulation (mbirjax's design) is aimed at "
          "the limiter.  Memory-side and load-heavy: the per-view reload "
          "of the values tile binds, and the kernel docstring's view-loop "
          "respecialization is the aimed remedy.  Compute-side: neither "
          "memory remedy has much to win, as mg25 found for the back "
          "kernel.")
    print("\n(2) HOW FAR FROM COALESCED IS EACH PATH?")
    print("    'atom/ideal' and 'ld/ideal' in table 2.  Near 1.00 means "
          "the path is already coalesced and reordering it has little to "
          "win.  This is also the check on section 1.19's recorded '192x': "
          "that figure was priced without the view factor, and the "
          "corrected ratio should read near 192 divided by the view batch "
          "if the re-derivation holds.")
    print("\n(3) WHAT LIMITS OCCUPANCY?")
    print("    Table 1 'limiter', 'reg/thr' and 'occup %', with Triton's "
          "own register count printed under the table whichever metric set "
          "collected.")
    if not profiled:
        print("\n    The counter leg did not produce tables, so every "
              "question above is UNANSWERED by this job; the timing leg "
              "alone says the kernel ran at its recorded rate.")
    print("\nTHE VERDICT IS READ BY A PERSON from the tables above.  This "
          "run decides nothing and implements nothing (open item B7).")


def summarize(header, timing, leg, out_path):
    print(f"\n===== mg31 parallel forward counters ({out_path}) =====")
    profiled = counter_table(leg)
    if profiled:
        traffic_table(leg)
    reading_guide(leg, profiled)

    checks = []
    if timing is None:
        checks.append("the timing leg produced no row")
    if not header.get("bodies_ok"):
        checks.append(f'torch bodies {header.get("torch_body_directions")} '
                      f'against {header.get("torch_body_expected")}')
    if DEVICE == "cuda" and not header.get("triton_forward_bound"):
        checks.append(f'the model bound {header.get("forward_body")}, not '
                      "the Triton forward kernel")
    if not header.get("device_ok"):
        checks.append(f'realized device {header.get("device_realized")} is '
                      f'not {header.get("device_expected")}')

    if header.get("gpu_hot_or_throttled"):
        print("\nNOTE: the device sampled hot or throttled; the timing "
              "anchor is read with that in mind.")
    if header.get("peak_bytes"):
        print(f'\ntiming leg peak device memory: '
              f'{header["peak_bytes"] / 2 ** 30:.1f} GiB')

    healthy = not checks
    print(f'\nexit code reports INSTRUMENT HEALTH only: '
          f'{"healthy" if healthy else "BROKEN"}.  It covers the timing '
          "row, the anchor window (skipped in the smoke), the Triton "
          "forward binding on CUDA, and the realized device.  The counter "
          "leg never changes it, and neither does what any column says.")
    for line in checks:
        print(f"  FAIL: {line}")
    return dict(kind="summary", healthy=healthy, checks=checks,
                profiler_permitted=leg.get("profiler_permitted"),
                profiler_reason=leg.get("reason"),
                unprofiled=[n for n, r in (leg.get("variants") or {}).items()
                            if not r.get("kernels")],
                out_path=out_path)


def _dry_run():
    print(f"mg31 cone forward counters: device {DEVICE}, cell "
          f"{tuple(cell())}, width {width()}, view batch {VIEW_BATCH}")
    print("  the segmented design note's cone input: is the cone forward "
          "atomic-bound like parallel's, or compute-side like the cone "
          "back?  It decides nothing.")
    print(f"  results and ncu logs -> {RESULTS_DIR}")
    print(f"  timing leg: {WARMUP_REPEATS} warm + {TIMED_REPEATS} timed "
          "full-pixel launches through the shipped wrapper; descriptive "
          "only -- no recorded single-launch cone anchor exists")
    print(f"  counter leg: {'on' if NCU_ENABLED else 'off (MG31_NCU=0)'}")
    if NCU_ENABLED:
        for name, divisor in NCU_VARIANTS:
            print(f"    {name:<{VARIANT_COL}}pixel divisor {divisor}")
        print(f"    {NCU_LAUNCHES} launches per worker; attempts per "
              "variant, in order:")
        for set_name, metrics, skip in NCU_ATTEMPTS:
            print(f"      metric set {set_name} ({len(metrics)} metrics), "
                  f"page raw, launch skip {skip}")
        print(f"    bounds: {NCU_TIMEOUT_S} s per attempt, "
              f"{NCU_LEG_BUDGET_S} s for the leg, {NCU_PROBE_TIMEOUT_S} s "
              "for the permission probe")
    print("  no library file is touched: both legs call the shipped "
          "wrapper, and the counter formulas carry the view factor mg20's "
          "pricing omitted")


def main():
    if DRY:
        _dry_run()
        return 0
    if not SMOKE:
        import torch
        if not torch.cuda.is_available():
            print("this run needs CUDA; use MG31_SMOKE=1 for the CPU "
                  "plumbing pass")
            return 2
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR,
                            f"mg31_pfwd_counters_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg31 cone forward counters on {RUN_LABEL} ({DEVICE}) -> "
          f"{out_path}", flush=True)
    with open(out_path, "w") as sink:
        print("  timing leg", flush=True)
        header, timing = timing_leg(sink)
        print("  counter leg", flush=True)
        leg = ncu_leg(RESULTS_DIR)
        sink.write(json.dumps(dict(kind="ncu_leg", **leg)) + "\n")
        sink.flush()
        summary = summarize(header, timing, leg, out_path)
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
