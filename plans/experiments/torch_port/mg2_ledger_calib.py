"""mg2 -- THE LEDGER AT n>1: modeled against measured per-device peaks.

The charter is `multigpu_plan.md` §4 (mg2) under the eleven protocols of §3.
The widening rule trusts the ledger at exactly the counts where the ledger is
unmeasured: every one of the five dp2 calibration cells ran at ONE device, and
the preflight consumes the ledger at n>1.  mg2 closes that gap.

This is dp2_ledger_calib.py extended to counts.  What carries over unchanged:
one arm per subprocess so no allocator state leaks; the modeled number read off
the model's own ``last_memory_calibration`` so the table reports what the
production path computed, never a re-derivation here; the composed measurement
protocol (warm seeded 3-iteration vcd, weights supplied, peak read after a cold
pass has already paid the compiles); and the acceptance band.

THE ARMS (10 calibrated arms, §7's count):

    weighted    {parallel, cone} x {(512,448,384), (1024,1008,992)}
                x n in {2, 4}                                        = 8
    unweighted  parallel (1024,1008,992) x n in {2, 4}               = 2

The unweighted pair carries the configuration whose dominant phase differs and
whose ``hess_weights`` sharding has never been measured: weights=None is the
only path on which vcd_recon builds its own all-ones sinogram into
``hess_weights`` and then never releases it, a residency a weighted run (where
``hess_weights`` is a bare alias of the caller's weights) cannot show.

THE BAND.  1.00 <= modeled/measured <= 1.30, PER DEVICE.  The lower bound is
the one that matters: a ledger that under-predicts would let a doomed run
start, which is the failure the whole model exists to prevent.  A cell below
the floor is fixed by ADDING THE MISSING TERM, never by a factor (the
checkpoint-2 rule), so every sub-band row here is reported with the phase
attribution this harness can capture: the dominant phase per device, its
largest terms, and the ``band reduce`` term specifically -- the term the ledger
names as flat in the count and the one mg5's seam lever would move.

dp3_phase_probe.py is the follow-up instrument for a sub-band reading and
stages ALONGSIDE this harness (the n=1 calibration needed it to attribute every
sub-band reading).  It is not run here.

DEVIATION FROM dp2, DELIBERATE (protocol 1).  dp2 pinned with
``model.configure_devices(1)``.  That takes the EXPLICIT branch, where the
preflight no longer runs; the env pin keeps the model on the automatic branch,
where it does.  Since the preflight is the ledger's only consumer, mg2 must
measure the ledger on the branch the preflight uses, so every arm here pins
through ``MBIRTORCH_NUM_DEVICES`` and then asserts the realized list.

THE GATHER CONTRACT (nt2_local_shard_check.py).  ``Shards.gather()`` ALREADY
returns numpy; re-detaching its result is the recorded failure that cost the
nightly's first 4-GPU trial all 32 of its n>1 rows.  ``_to_numpy`` below is the
only host exit and never re-detaches a gather.

Run:
    <torch python> mg2_ledger_calib.py       on a 4-GPU node (mg2_gautschi.sbatch)
    python mg2_ledger_calib.py --dry-run     anywhere: print the arm plan
    python mg2_ledger_calib.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list).  List values are parsed STRICTLY.
    MG2_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG2_GEOMETRIES=parallel,cone      subset of the geometries
    MG2_CELLS=512,1024                subset of the cells (by view count)
    MG2_COUNTS=2,4                    subset of the device counts
    MG2_ARMS=weighted,unweighted      subset of the arms
    MG2_ITERATIONS=3                  VCD iterations per recon
    MG2_WARM_REPEATS=1                calibrated warm passes per arm
    MG2_SMOKE=1 / MG2_DEVICE=cpu      the local smoke
"""

import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELLS = [(512, 448, 384), (1024, 1008, 992)]
GEOMETRIES = ("parallel", "cone")
COUNTS = (2, 4)
ARMS = ("weighted", "unweighted")

# The unweighted pair runs at ONE cell: its purpose is to size a single
# residency at n>1, not to re-measure the matrix.
UNWEIGHTED_CELL = (1024, 1008, 992)
UNWEIGHTED_GEOMETRY = "parallel"

SMOKE = os.environ.get("MG2_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG2_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG2_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 12345                      # dp2's seed, so the rows compare
# dp2's protocol: one cold pass (which pays the compiles) then the calibrated
# warm pass.  mg2's reading is a MEMORY reading, not a timing one, so protocol
# 9's three-repeat rule does not bind it; the knob is here so a spread can be
# priced if a reviewer wants one.
WARM_REPEATS = max(1, int(os.environ.get("MG2_WARM_REPEATS", "1")))

# The band the checkpoint-1 design fixed; a ratio below the floor is the
# failure this whole model exists to prevent.
BAND = (1.00, 1.30)
# The ledger term the plan names as flat in the count, and the one mg5's seam
# lever would move.  Surfaced per device so the mg5 trigger is decidable.
BAND_REDUCE_TERM = "band reduce"

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
    "MG2_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """Refuse garbage: every token must name a member of ``allowed`` (kb3's
    slurm --export comma-split lesson)."""
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
    if SMOKE:
        cells = [SMOKE_CELL]
    else:
        keep = _strict_subset("MG2_CELLS", {c[0] for c in CELLS}, int)
        cells = [c for c in CELLS if c[0] in keep]
    chosen = _strict_subset("MG2_GEOMETRIES", set(GEOMETRIES))
    # Normalized to the DECLARED order so the job order is reproducible.
    geometries = [g for g in GEOMETRIES if g in chosen]
    counts = _strict_subset("MG2_COUNTS", set(COUNTS), int)
    counts = [n for n in COUNTS if n in counts]
    arms = _strict_subset("MG2_ARMS", set(ARMS))
    arms = [a for a in ARMS if a in arms]
    return geometries, cells, counts, arms


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg2_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _md5(path, chunk=8 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(x):
    """The ONE host exit.  ``Shards.gather()`` already returns numpy -- the
    nt2 shard check's recorded failure class is re-detaching that result."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    gather = getattr(x, "gather", None)
    if callable(gather) and hasattr(x, "placement"):
        return gather()                       # ALREADY numpy: do not re-detach
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weights(sinogram):
    """The gates' weighting formula, so the arms match the composed cells."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


# ── the GPU health sample (protocol 11) ───────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
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


def row_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── model construction ────────────────────────────────────────────────────────
def _build_model(geometry, cell, pin_devices=None):
    """The model.  On CUDA nothing is configured here: protocol 1 pins through
    MBIRTORCH_NUM_DEVICES, which keeps the model on the AUTOMATIC branch where
    the preflight -- the ledger's only consumer -- still runs.  ``pin_devices``
    is the smoke's CPU virtual-device path only."""
    import numpy as np

    import mbirtorch

    num_views = cell[0]
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        sdd = 4 * cell[2]
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=sdd,
                                        source_iso_dist=sdd)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _ledger_record(model):
    """The modeled side, PER DEVICE: the phase table, the dominant phase, its
    largest terms, and the band-reduce term.  All read off the model's own
    ledger, never re-derived here."""
    ledger = model.last_memory_ledger
    if ledger is None:
        return None
    n_dev = len(ledger.devices)
    phases = [dict(name=phase.name,
                   per_device_bytes=[int(b) for b in phase.per_device])
              for phase in ledger.phases]
    dominant, band_reduce, top_terms = [], [], []
    for i in range(n_dev):
        phase = ledger.dominant_phase(i)
        dominant.append(phase.name)
        top_terms.append([[name, int(value)]
                          for name, value in phase.dominant_terms(i, count=3)])
        # The band-reduce term wherever it is charged, at its largest.
        largest = 0
        for candidate in ledger.phases:
            for name, values in candidate.terms:
                if name == BAND_REDUCE_TERM:
                    largest = max(largest, int(values[i]))
        band_reduce.append(largest)
    return dict(devices=[str(d) for d in ledger.devices],
                modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
                dominant_phase=dominant, dominant_terms=top_terms,
                band_reduce_term_bytes=band_reduce,
                num_pixels_full=int(ledger.num_pixels_full),
                phases=phases)


# ── the worker: one arm, one process ──────────────────────────────────────────
def run_arm(cfg):
    """One (geometry, cell, arm, n) calibration measurement, in its own
    process.  MBIRTORCH_MEMORY_CALIBRATION is set in this process's
    ENVIRONMENT by the runner, so the mode owns the peak counter from the
    first recon entry and both the cold and the warm passes are calibrated;
    the WARM row is the reading, and the cold row is kept beside it because it
    carries the compile transients the warm pass no longer pays."""
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    arm, n_dev = cfg["arm"], cfg["n_dev"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = cfg.get("cpu_devices") if not cuda else None
    if not cuda and pin_devices is None:
        pin_devices = [DEVICE]

    model = _build_model(geometry, cell, pin_devices=pin_devices)
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    # The one job where the calibration mode IS expected to be on.
    result["calibration_env_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1")

    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    result["num_pixels_full"] = int(model.full_index_count())

    # The bodies bound, so a fallback cannot be mistaken for a kernel reading
    # (the arm-check discipline the gate campaigns established).  The ledger's
    # view-batch charge follows the BOUND body, so this is load-bearing here.
    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__
    result["kernels_on"] = ("triton" in fwd_body.__name__
                            and "triton" in back_body.__name__)
    result["bodies_ok"] = result["kernels_on"] if cuda else True

    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    result["sino_md5"] = actual_md5
    result["sino_md5_ok"] = (actual_md5 == expected_md5)
    if not result["sino_md5_ok"]:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5}")
    sinogram = np.load(sino_path)
    weights = None if arm == "unweighted" else _weights(sinogram)

    def one_recon():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return recon

    def calibration_rows():
        return [dict(device=str(device), modeled_bytes=int(modeled),
                     measured_bytes=int(measured), ratio=float(ratio))
                for device, modeled, measured, ratio
                in (model.last_memory_calibration or [])]

    health = [sample_gpu_health()]
    # Cold pass: pays every compile, so the warm pass measures steady state.
    # Triton compiles PER DEVICE, so an n-device cold pass pays n compiles per
    # shape (protocol 7); the wall is recorded because §6's cadence decision
    # needs exactly the costs the warm protocol discards.
    start = time.perf_counter()
    one_recon()
    result["cold_s"] = time.perf_counter() - start
    result["calibration_cold"] = calibration_rows()
    health.append(sample_gpu_health())

    warm, warm_rows = [], []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        one_recon()
        warm.append(time.perf_counter() - start)
        warm_rows.append(calibration_rows())
        health.append(sample_gpu_health())
    result["warm_all_s"] = warm
    result["warm_s"] = statistics.median(warm)
    result["calibration"] = warm_rows[-1]
    result["calibration_all"] = warm_rows

    # ── arm check: the realized device list, after the timed call ────────────
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))

    result["ledger"] = _ledger_record(model)
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def generate(cfg):
    """One shared phantom and sinogram per (geometry, cell), md5'd, for every
    arm at that cell to read.  Pinned to a single device so the generator
    cannot itself become a multi-device run."""
    import numpy as np
    import torch

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    model = _build_model(geometry, cell,
                         pin_devices=(cfg.get("cpu_devices") or [DEVICE]))
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _sino_path(geometry, cell)
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(geometry, cell), "w") as handle:
        handle.write(digest + "\n")
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(cfg, path=path, sino_md5=digest, recon_shape=list(recon_shape))


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.
    Protocol 1: the pin is MBIRTORCH_NUM_DEVICES and nothing else.  mg2 is the
    ONE job that sets MBIRTORCH_MEMORY_CALIBRATION, and it does so for the
    measurement arms only -- the generator must not own the peak counter."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg["mode"] == "arm":
        env["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
        if cfg.get("n_dev") and DEVICE == "cuda":
            env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter (protocol 6: memory is
    re-measured per arm in a fresh subprocess, never inferred)."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def build_plan(geometries, cells, counts, arms):
    """The arm plan, in job order.  Generators first, then the weighted matrix,
    then the unweighted pair."""
    plan = []
    for geometry in geometries:
        for cell in cells:
            entry = dict(mode="generate", geometry=geometry, cell=list(cell),
                         arm="generate", n_dev=None)
            if DEVICE != "cuda":
                entry["cpu_devices"] = [DEVICE]
            plan.append(entry)
    if "weighted" in arms:
        for geometry in geometries:
            for cell in cells:
                for n in counts:
                    plan.append(_arm_cfg(geometry, cell, "weighted", n))
    if "unweighted" in arms:
        for n in counts:
            cell, geometry = UNWEIGHTED_CELL, UNWEIGHTED_GEOMETRY
            if SMOKE:
                cell, geometry = cells[0], geometries[0]
            if cell in cells and geometry in geometries:
                plan.append(_arm_cfg(geometry, cell, "unweighted", n))
    return plan


def _arm_cfg(geometry, cell, arm, n):
    entry = dict(mode="arm", geometry=geometry, cell=list(cell), arm=arm,
                 n_dev=n, arm_id=f"{geometry}_{cell[0]}_{arm}_n{n}")
    if DEVICE != "cuda":
        # SMOKE ONLY: the env pin is a CUDA-only mechanism (the policy
        # short-circuits at `visible < 2`), so the CPU path pins by device
        # LIST and says so on the row.
        entry["cpu_devices"] = [DEVICE] * n
    return entry


def main():
    geometries, cells, counts, arms = selected_plan()
    plan = build_plan(geometries, cells, counts, arms)
    if "--dry-run" in sys.argv:
        measured = [c for c in plan if c["mode"] == "arm"]
        print(f"mg2 plan: {len(measured)} calibrated arms "
              f"({len(plan) - len(measured)} generators)")
        for cfg in plan:
            print(f"  {cfg.get('arm_id', cfg['arm']):<40} "
                  f"{cfg['geometry']:>9} {cfg['cell'][0]:>5} n={cfg['n_dev']}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg2_ledger_calib_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg2 ledger at n>1 on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, cells {[c[0] for c in cells]}, counts {counts}, "
          f"arms {arms} -> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY (protocol 11), so a truncated job still yields
    # every arm it finished.
    with open(out_path, "w") as sink:
        for cfg in plan:
            if cfg["mode"] == "generate" and \
                    os.path.exists(_md5_path(cfg["geometry"],
                                             tuple(cfg["cell"]))):
                continue
            label = cfg.get("arm_id", f"generate {cfg['geometry']} "
                                      f"{cfg['cell'][0]}")
            print(f"  {label}", flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        summary = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    for geometry in geometries:
        for cell in cells:
            for path in (_sino_path(geometry, cell), _md5_path(geometry, cell)):
                if os.path.exists(path):
                    os.remove(path)
    print(f"\nwrote {out_path}")


def summarize(rows, out_path):
    print(f"\n===== mg2 ledger calibration at n>1 ({out_path}) =====")
    header = (f'{"geometry":>10}{"views":>7}{"arm":>12}{"n":>3}{"dev":>7}'
              f'{"kern":>6}{"modeled":>11}{"measured":>11}{"ratio":>8}'
              f'{"verdict":>9}  dominant phase')
    print(header)
    print("-" * len(header))
    failures, under, summary_rows = [], [], []
    for row in rows:
        if row.get("mode") != "arm":
            continue
        if row.get("error"):
            print(f'{row["geometry"]:>10}{row["cell"][0]:>7}{row["arm"]:>12}'
                  f'{row["n_dev"]:>3}   ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:70]}')
            failures.append(row.get("arm_id"))
            continue
        ledger = row.get("ledger") or {}
        dominant = ledger.get("dominant_phase") or []
        band_reduce = ledger.get("band_reduce_term_bytes") or []
        if not row.get("calibration"):
            # The MEASURED side is torch.cuda.max_memory_allocated, which is
            # CUDA-only, so a CPU smoke exercises the modeled side alone.  Say
            # so rather than printing an empty row that reads as a failure.
            modeled = ledger.get("modeled_peak_bytes") or []
            print(f'{row["geometry"]:>10}{row["cell"][0]:>7}{row["arm"]:>12}'
                  f'{row["n_dev"]:>3}      -{str(row.get("kernels_on")):>6}'
                  f'{(max(modeled) / 2 ** 30 if modeled else 0):>10.2f}G'
                  f'{"n/a":>11}{"n/a":>8}{"no-cuda":>9}  '
                  f'{dominant[0] if dominant else ""}')
        entry = dict(arm_id=row.get("arm_id"), geometry=row["geometry"],
                     cell=row["cell"], arm=row["arm"], n_dev=row["n_dev"],
                     devices_ok=row.get("devices_ok"),
                     kernels_on=row.get("kernels_on"),
                     warm_s=row.get("warm_s"), cold_s=row.get("cold_s"),
                     band_reduce_bytes=band_reduce,
                     modeled_peak_bytes=ledger.get("modeled_peak_bytes"),
                     per_device=[])
        for i, cal in enumerate(row.get("calibration", [])):
            ratio = cal["ratio"]
            verdict = ("UNDER" if ratio < BAND[0]
                       else "over" if ratio > BAND[1] else "ok")
            if verdict != "ok":
                failures.append(f'{row.get("arm_id")}|{cal["device"]}')
            if verdict == "UNDER":
                under.append((row, i, cal))
            entry["per_device"].append(
                dict(device=cal["device"], modeled_bytes=cal["modeled_bytes"],
                     measured_bytes=cal["measured_bytes"], ratio=ratio,
                     verdict=verdict,
                     dominant_phase=(dominant[i] if i < len(dominant) else None),
                     band_reduce_bytes=(band_reduce[i]
                                        if i < len(band_reduce) else None)))
            print(f'{row["geometry"]:>10}{row["cell"][0]:>7}{row["arm"]:>12}'
                  f'{row["n_dev"]:>3}{cal["device"]:>7}'
                  f'{str(row.get("kernels_on")):>6}'
                  f'{cal["modeled_bytes"] / 2 ** 30:>10.2f}G'
                  f'{cal["measured_bytes"] / 2 ** 30:>10.2f}G'
                  f'{ratio:>8.3f}{verdict:>9}  '
                  f'{dominant[i] if i < len(dominant) else ""}')
        if row.get("devices_ok") is False:
            print(f'    ARM CHECK FAIL: realized {row.get("realized_devices")} '
                  f'for n={row["n_dev"]}')
            failures.append(f'{row.get("arm_id")}|devices')
        if row.get("calibration_env_ok") is False:
            print("    ARM CHECK FAIL: MBIRTORCH_MEMORY_CALIBRATION not set")
            failures.append(f'{row.get("arm_id")}|calibration_env')
        summary_rows.append(entry)
    print("-" * len(header))
    print(f"acceptance band {BAND[0]:.2f} <= modeled/measured <= {BAND[1]:.2f} "
          f"PER DEVICE; {len(failures)} reading(s) outside it or failed")

    # A sub-band reading is fixed by ADDING THE MISSING TERM, never by a
    # factor (the checkpoint-2 rule), so every one is reported here with the
    # attribution this harness can capture.  dp3_phase_probe.py is the
    # follow-up instrument and stages alongside this file.
    for row, i, cal in under:
        ledger = row.get("ledger") or {}
        terms = (ledger.get("dominant_terms") or [[]])[i] \
            if i < len(ledger.get("dominant_terms") or []) else []
        print(f"\nUNDER-PREDICTION {row.get('arm_id')} on {cal['device']}: "
              f"modeled {cal['modeled_bytes'] / 2 ** 30:.2f}G < measured "
              f"{cal['measured_bytes'] / 2 ** 30:.2f}G (ratio "
              f"{cal['ratio']:.3f}); missing "
              f"{(cal['measured_bytes'] - cal['modeled_bytes']) / 2 ** 30:.2f}G")
        print(f"  dominant phase: "
              f"{(ledger.get('dominant_phase') or [None])[i]}")
        for name, value in terms:
            print(f"    {name:<40}{value / 2 ** 30:>8.2f}G")
        print("  next instrument: dp3_phase_probe.py (staged alongside), "
              "which attributes a sub-band reading to a phase")

    # The band-reduce term, so mg5's seam trigger is decidable at the
    # checkpoint: the plan prices it at ~1.5x cyl(P, S_pad), FLAT in the count.
    print("\nband-reduce term (the mg5 seam lever's subject), per device:")
    for entry in summary_rows:
        values = entry.get("band_reduce_bytes") or [0]
        peaks = [d["measured_bytes"] for d in entry["per_device"]]
        against = "measured"
        if not peaks or not max(peaks):
            peaks, against = entry.get("modeled_peak_bytes") or [0], "modeled"
        share = (max(values) / max(peaks)) if peaks and max(peaks) else 0.0
        print(f'  {entry["arm_id"]:<34} max '
              f'{max(values, default=0) / 2 ** 30:.3f}G '
              f'({share:.1%} of the {against} peak)')

    # The residency the unweighted pair exists to size, now at n>1.
    weighted = {(r["geometry"], r["cell"][0], r["n_dev"]): r for r in rows
                if r.get("mode") == "arm" and r.get("arm") == "weighted"
                and not r.get("error")}
    for row in rows:
        if row.get("mode") != "arm" or row.get("arm") != "unweighted" \
                or row.get("error"):
            continue
        pair = weighted.get((row["geometry"], row["cell"][0], row["n_dev"]))
        if not pair:
            continue
        un = [c["measured_bytes"] for c in row.get("calibration", [])]
        wt = [c["measured_bytes"] for c in pair.get("calibration", [])]
        if not un or not wt:
            continue
        print(f"\nhess_weights probe ({row['geometry']} {row['cell'][0]} "
              f"n={row['n_dev']}): unweighted measured "
              f"{max(un) / 2 ** 30:.2f} GB vs weighted {max(wt) / 2 ** 30:.2f} "
              f"GB, difference {(max(un) - max(wt)) / 2 ** 30:+.2f} GB")
    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}")
    return dict(rows=summary_rows, failures=failures,
                under=[r.get("arm_id") for r, _i, _c in under], hot=hot)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        try:
            out = generate(cfg) if cfg["mode"] == "generate" else run_arm(cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        main()
