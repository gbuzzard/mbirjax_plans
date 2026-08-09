"""mg4 -- THE CROSSOVER LADDER: where each device count stops paying, measured
across a size sweep, which is the data decision 1 needs.

The charter is `multigpu_plan.md` §4 (mg4) and §6 decision 1, under the eleven
protocols of §3.  The subject is the OUT-OF-BOX `recon()`, so every arm is warm
seeded 3-iteration VCD, the same protocol as the readout.

WHAT DECISION 1 NEEDS FROM THIS FILE (§6).  The automatic path is capacity-only
today, so it widens small problems onto counts that run slower.  The proposed
guard is a FLOOR PER COUNT, below which the automatic path does not admit that
count.  Each count's floor lands at this ladder's admission point for that count
under §4's knee rule, with a margin toward FEWER devices -- the margin direction
follows from the measured asymmetry (widening a below-knee cell cost about 2.7x
in the phase-4 prior; holding a just-above-knee cell at a smaller count costs a
few percent).  The floors' metric must be a quantity the decision site already
knows, and §8 leaves open WHICH: sinogram elements, recon voxels, or their
product.  The thin-volume probe below is what tells them apart.

THE KNEE RULE IS NOT COMPUTED HERE.  §4 states it against noise -- a count is
admitted at a size only if it wins by more than the protocol-9 spread THERE and
at EVERY LARGER ladder size, so one noisy cell cannot move a floor.  That is an
analysis rule over the whole table, so this harness only PRINTS the raw speedups
beside their spreads and emits every field the rule needs: warm median, spread,
count, realized devices, per-device memory, and the health sample.

THE CELLS.

    parallel family    EXACT proportions of the 512 gate cell, so the ladder is
                       one geometric sequence and nothing but size varies:
                       (128,112,96) (192,168,144) (256,224,192) (384,336,288)
                       (512,448,384) (768,672,576).  Views and slices divide by
                       4 at every cell, so no padding effect confounds the knee.
    off-family anchor  (1024,1008,992), labelled off-family, re-measured INSIDE
                       mg4 so the ladder ties to mg1's matrix without a
                       cross-job comparison.
    cone spot check    (256,224,192) (384,336,288) (512,448,384) -- a pinned
                       default BRACKETING the expected knee region.
                       `lessons.md` §6 records that vcd has a problem-size floor
                       near 256-class cells: per-subset work is num_subsets
                       times smaller than a bare projection, and the per-subset
                       host scalar syncs are the GPU-specific limiter.  The
                       bracket is a top-of-file constant so a follow-up can move
                       it if the parallel knee lands outside it.
    thin probe         ONE point, sized to DISCRIMINATE (see below).

Three of the six family cells sit at or below the 256 class, which is where §2
says the ladder should find the knee.

THE THIN-VOLUME PROBE -- (1152, 336, 96), and the arithmetic that picks it.
The family scales all three axes together, so within it

    sinogram elements / recon voxels = (views x rows x channels)
                                     / (channels x channels x rows)
                                     = views / channels = 512/384 = 4/3

is CONSTANT: the family cannot tell the two candidate metrics apart, which is
exactly what §8 says.  The probe breaks the tie by making them disagree, with
the expected knee at the 256-class cell:

    metric                    probe          one step BELOW    knee        one step ABOVE
                              (1152,336,96)  (192,168,144)     (256,...)   (384,336,288)
    sinogram elements         37,158,912      4,644,864        11,010,048  37,158,912
    recon voxels               3,096,576      3,483,648         8,257,536  27,869,184
    their product              1.151e14        1.618e13         9.092e13    1.036e15

By SINOGRAM ELEMENTS the probe is EXACTLY the (384,336,288) cell -- one full
ladder step ABOVE the knee, so a sinogram-element guard admits there whatever it
admits at (384,336,288).  By RECON VOXELS the probe is SMALLER than
(192,168,144) -- one full ladder step BELOW the knee, so a voxel guard refuses
there whatever it refuses at (192,168,144).  The two metrics therefore land on
OPPOSITE SIDES of the knee, each a full step clear of it, and the measurement
records which one the outcome vindicated.  Their product gives a third answer
again (just above the knee, 1.27x it), so all three of §8's candidates are
separated by this one point.

Engine constraints respected: views 1152 and rows 336 both divide by 4, so no
padding at n = 1, 2 or 4; the recon slice axis is the 336 detector rows, giving
84 slices per device at n=4, so the thin-volume empty-shard extension is not
even reached (it is what PERMITS few slices; this shape does not need it).  The
engine's own `auto_set_recon_geometry` was queried to confirm the recon shape is
(96, 96, 336) rather than assumed.

THE AUTO ARMS (§4, and where today's harm lands).  One per cell, riding beside
the pinned arms and doubling as the protocol-9 repeat of the pinned all-device
arm.  An auto arm sets NO pin and NEVER calls `configure_devices`; under the
one-bit rule any call is explicit and would silently disable the very automatic
path being observed (tomography_model.py:803).  Two arm checks keep the
observation honest -- `MBIRTORCH_NUM_DEVICES` absent from the environment, and a
counted wrapper proving `configure_devices` was not called.  The arm records the
policy's chosen count and the per-candidate rejection reasons the selection loop
already logs at verbose 2 (`_settle`, tomography_model.py:985), read BOTH from
the log stream and from `model.device_choice_rejections`, which is where the
loop parks them.  AN UNEXPECTED CHOICE IS A RECORDED FINDING, NOT A CRASH.

WHY EVERY CELL STILL GETS A SHARED SINOGRAM ARTIFACT (an addition beyond §4,
flagged for review).  §4 does not require protocol 5 here, because mg4 is
single-framework and measures time rather than value.  But a sinogram generated
inside each arm would be generated AT THAT ARM'S DEVICE COUNT, so the n=1, n=2,
n=4 and auto arms of one cell would reconstruct four different arrays and the
comparison the knee rule reads would not be controlled.  One artifact per cell,
built once at one device and md5-verified by every consumer, costs one forward
projection and removes the confound.

THE GATHER CONTRACT (nt2_local_shard_check.py).  `Shards.gather()` ALREADY
returns numpy (_sharding.py:181).  Re-detaching its result is the recorded
failure that cost the nightly's first 4-GPU trial all 32 of its n>1 rows.  Every
host exit here goes through `_to_numpy`, which never re-detaches a gather.

ARM COUNT (§7: "28 parallel + 12 cone + 1 probe"):

    parallel family  6 cells x (n=1,2,4 pinned + auto)          = 24
    off-family anchor  1 cell x 4                               =  4   } 28
    cone spot check    3 cells x 4                              = 12
    thin probe         1 POINT x 4 arms                         =  4
                                                          total  44 arms
                                                       + 11 untimed generators

§7's "1 probe" counts the probe POINT; it is four arms, like every other cell.

Run:
    <torch python> mg4_ladder.py             on a 4-GPU node (mg4_gautschi.sbatch)
    python mg4_ladder.py --dry-run           anywhere: print the arm plan
    python mg4_ladder.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON                   interpreter for the arm subprocesses
    MG4_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG4_FAMILIES=parallel_family,...  subset of the cell families
    MG4_COUNTS=1,2,4                  subset of the pinned device counts
    MG4_ITERATIONS=3                  VCD iterations per recon
    MG4_WARM_REPEATS=3                warm repeats after the discarded cold pass
    MG4_SKIP_AUTO=1                   drop the auto arms
    MG4_KEEP_ARTIFACTS=1              keep the .npy files after the run
    MG4_SMOKE=1                       the local smoke (tiny cells, few iters)
    MG4_DEVICE=cpu                    smoke device
"""

import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The parallel family: EXACT proportions of the 512 gate cell (x1/4, x3/8, x1/2,
# x3/4, x1, x3/2).  Views and slices divide by 4 at every one, so padding never
# confounds the knee.
PARALLEL_LADDER = [(128, 112, 96), (192, 168, 144), (256, 224, 192),
                   (384, 336, 288), (512, 448, 384), (768, 672, 576)]
# The off-family size anchor: mg1's second gate cell, re-measured HERE so the
# ladder ties to mg1's matrix without a cross-job comparison.
OFF_FAMILY_ANCHOR = (1024, 1008, 992)
# The cone spot check: a PINNED DEFAULT bracketing the expected knee region.
# `lessons.md` §6 records vcd's problem-size floor near 256-class cells (the
# per-subset host scalar syncs are the GPU-specific limiter), so the bracket
# straddles it.  A follow-up moves this list if the parallel knee lands outside.
CONE_SPOT_CHECK = [(256, 224, 192), (384, 336, 288), (512, 448, 384)]
# The thin-volume probe: see the module docstring for the discrimination
# arithmetic.  sinogram elements put it one ladder step ABOVE the knee, recon
# voxels one step BELOW, and their product a third place again.
THIN_PROBE = (1152, 336, 96)

# The two-probe ADDENDUM (Greg, 2026-08-09), covering the two off-family
# classes where the guard could be WRONG rather than merely conservative,
# first measured in the mg4b addendum job:
#   thin_volume_probe  few slices, many views (the flat-panel class, the
#                      empty-VIEW-shard extension's regime): view-axis work
#                      dominates, so widening should pay at a SMALLER
#                      sinogram size than the family knee -- a single
#                      sinogram-elements floor risks holding n=1 where n=2
#                      wins.  32 slices divide by 4, so no shard is empty
#                      and no padding fires.
#   sparse_view_probe  few views on the 512-cell volume (the sparse-view
#                      regime, short of the empty-shard extreme: 16 real
#                      views per device at n=4): the sinogram says hold
#                      n=1 while the recon side is 512-class -- the
#                      opposite stress of the thin probe.
THIN_VOLUME_PROBE = (1024, 32, 768)
SPARSE_VIEW_PROBE = (64, 448, 384)

COUNTS = (1, 2, 4)
FAMILIES = ("parallel_family", "thin_probe", "off_family_anchor", "cone_spot",
            "thin_volume_probe", "sparse_view_probe")
GEOMETRY_OF = {"parallel_family": "parallel", "thin_probe": "parallel",
               "off_family_anchor": "parallel", "cone_spot": "cone",
               "thin_volume_probe": "parallel", "sparse_view_probe": "parallel"}

SMOKE = os.environ.get("MG4_SMOKE", "0") == "1"
# Two tiny cells so the CROSS-CELL machinery runs, plus a tiny probe point and a
# tiny cone cell: the smoke's job is that every code path executes, not that any
# number means anything.
SMOKE_LADDER = [(8, 24, 20), (16, 24, 20)]
SMOKE_ANCHOR = (24, 24, 20)
SMOKE_CONE = [(8, 24, 20)]
SMOKE_THIN_PROBE = (16, 24, 8)
SMOKE_THIN_VOLUME = (16, 8, 20)
SMOKE_SPARSE_VIEW = (8, 24, 20)
DEVICE = os.environ.get("MG4_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG4_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # kb3's seed, the same protocol as the readout
SAMPLE_ROWS = 16          # kb3's / p5k6's / mg1's sample convention
WARM_REPEATS = int(os.environ.get("MG4_WARM_REPEATS", "1" if SMOKE else "3"))

# The phase-4 prior the ladder re-measures (§2, §6): at the 512 cell n=2 beat
# n=1 by 18 percent while n=4 collapsed to about 2.7x SLOWER.  Recorded here so
# the printout can say whether the shape reproduced; the numbers are PRE-KERNEL
# and are not a gate.
PRIOR_512_SPEEDUP_OVER_N1 = {2: 3.15 / 2.58, 4: 3.15 / 8.59}

# The throttle rule (protocol 11 / nightly_plan.md §10.5): sw_power_cap at a
# normal temperature is the boost governor -- recorded and KEPT.  A row is
# marked for re-run only when it is hot AND its clock is depressed.
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
    "MG4_RESULTS",
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
    """(cells, counts), where cells is a list of (family, geometry, cell) in
    JOB ORDER of the families."""
    families = [f for f in FAMILIES
                if f in _strict_subset("MG4_FAMILIES", set(FAMILIES))]
    ladder = SMOKE_LADDER if SMOKE else PARALLEL_LADDER
    anchor = SMOKE_ANCHOR if SMOKE else OFF_FAMILY_ANCHOR
    cone = SMOKE_CONE if SMOKE else CONE_SPOT_CHECK
    probe = SMOKE_THIN_PROBE if SMOKE else THIN_PROBE
    thin_volume = SMOKE_THIN_VOLUME if SMOKE else THIN_VOLUME_PROBE
    sparse_view = SMOKE_SPARSE_VIEW if SMOKE else SPARSE_VIEW_PROBE
    by_family = {"parallel_family": ladder, "thin_probe": [probe],
                 "off_family_anchor": [anchor], "cone_spot": cone,
                 "thin_volume_probe": [thin_volume],
                 "sparse_view_probe": [sparse_view]}
    cells = [(family, GEOMETRY_OF[family], cell)
             for family in families for cell in by_family[family]]
    if SMOKE and not os.environ.get("MG4_COUNTS", "").strip():
        # The env pin is a CUDA-only mechanism (the policy short-circuits at
        # `visible < 2`, tomography_model.py:894), so a pinned n>1 arm on CPU
        # would silently measure n=1 and the smoke would prove nothing.
        counts = [1]
    else:
        chosen = _strict_subset("MG4_COUNTS", set(COUNTS), int)
        counts = [n for n in COUNTS if n in chosen]
    return cells, counts


# ── staged-artifact mechanics (protocol 5's mechanics, see the docstring) ─────
def _cell_tag(geometry, cell):
    return f"{geometry}_{cell[0]}x{cell[1]}x{cell[2]}"


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg4_sino_{_cell_tag(geometry, cell)}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id):
    return os.path.join(RESULTS_DIR, f"_mg4_sample_{arm_id}.npy")


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, chunked: the anchor's artifact is a ~4 GB array and
    a corrupt Lustre read is a recorded failure mode."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(x):
    """The ONE host exit.  ``Shards.gather()`` already returns numpy
    (_sharding.py:181) -- the nt2 shard check's recorded failure class is
    re-detaching that result -- so a gather is never followed by ``.detach()``."""
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
    """One weighting formula, one dtype, every arm (kb3's rule)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def sino_elements(cell):
    return int(cell[0]) * int(cell[1]) * int(cell[2])


def recon_voxels_parallel(cell):
    """channels x channels x rows -- the parallel recon shape at unit
    magnification, confirmed against the engine's own
    ``auto_set_recon_geometry``.  Cone magnifies, so this is reported for the
    parallel cells only and the engine's realized shape is recorded on every
    row regardless."""
    return int(cell[2]) * int(cell[2]) * int(cell[1])


# ── the GPU health sample (protocol 11) ───────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks (SM + memory), temps (core + HBM), and active throttle
    reasons, via nvidia-smi.  ``[]`` when nvidia-smi is unavailable.  The fields
    mirror the nightly's own sample so the two are comparable."""
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


def worst_health(samples):
    """Per-GPU worst case across samples: MIN clocks, MAX temps, the union of
    throttle reasons.  A single post-run snapshot misses the dips."""
    agg = {}
    for snapshot in samples:
        for gpu in snapshot:
            index = gpu.get("index")
            slot = agg.setdefault(index, {"index": index, "sm_mhz": None,
                                          "mem_mhz": None, "temp_c": None,
                                          "mem_temp_c": None, "throttle": []})
            for key in ("sm_mhz", "mem_mhz"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else min(slot[key],
                                                                    value)
            for key in ("temp_c", "mem_temp_c"):
                value = gpu.get(key)
                if value is not None:
                    slot[key] = value if slot[key] is None else max(slot[key],
                                                                    value)
            for reason in gpu.get("throttle", []):
                if reason not in slot["throttle"]:
                    slot["throttle"].append(reason)
    return [agg[k] for k in sorted(agg, key=lambda i: (i is None, i))]


def row_is_hot(health):
    """Hot by TEMPERATURE alone (the reliable signal).  Not a re-run verdict on
    its own: protocol 11's re-run rule needs a depressed clock too, and
    sw_power_cap at a normal temperature is the boost governor -- recorded and
    KEPT."""
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── the torch side ────────────────────────────────────────────────────────────
def _build_torch_model(geometry, cell, pin_devices=None):
    """The model, in the PRODUCTION configuration (kernels on, compile on),
    because the guard's subject is the out-of-box ``recon()``.

    ``pin_devices`` is a device LIST for the smoke's CPU paths only; on CUDA
    nothing is configured here, because protocol 1 pins through
    MBIRTORCH_NUM_DEVICES -- which keeps the model on the AUTOMATIC branch where
    the preflight still runs -- and an explicit ``configure_devices`` call would
    take the explicit branch and get no preflight."""
    import numpy as np

    import mbirtorch

    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels)
    if pin_devices is not None:
        model.configure_devices(devices=pin_devices)
    model.set_params(no_warning=True, verbose=0)
    return model


def _launch_key_counts(geometry):
    """Per-kernel launch-key counts (kb3's positive witness): all four kernels
    share one key set and every key leads with its kernel's name."""
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    names = (("pback", "pfwd") if geometry == "parallel" else ("back", "fwd"))
    back = sum(1 for k in _COMPILED_LAUNCH_KEYS
               if isinstance(k, tuple) and k and k[0] == names[0])
    fwd = sum(1 for k in _COMPILED_LAUNCH_KEYS
              if isinstance(k, tuple) and k and k[0] == names[1])
    return back, fwd


def _view_batch_static(model, expect_kernels):
    """kb3's realized-view-batch check, PER DEVICE (protocol 3: the batch budget
    divides by the count while the kernel cost models do not all follow, so the
    realized batch is not invariant in n).  Every arm records it, because on
    this ladder the batch is one of the few things that can change the shape of
    a scaling curve without changing the code."""
    import mbirtorch

    pf = model.projector_functions
    args = model._view_batch_args()
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    budget = pf._transient_budget_bytes()
    n_dev = model.sino_placement.n_devices
    cols = dict(fwd=int(-(-recon_shape[2] // n_dev)),
                back=int(sinogram_shape[1]))
    bodies = dict(fwd=pf._fwd_body_per_dev, back=pf._back_body_per_dev)
    expected_is_kernel = dict(fwd=expect_kernels[0], back=expect_kernels[1])

    record, ok = {}, True
    for direction in ("fwd", "back"):
        realized, expected_all = [], []
        for body in bodies[direction]:
            value = int(pf._effective_view_batch(body, num_pixels,
                                                 cols[direction], args))
            legacy = max(1, min(64, budget
                                // max(1, num_pixels
                                       * model._transient_cols(cols[direction])
                                       * 4)))
            cost = getattr(body, "_view_batch_cost", None)
            if expected_is_kernel[direction]:
                if cost is None:
                    expected = None          # the body check already failed
                else:
                    bytes_pv, chunk = cost(num_pixels, cols[direction], args)
                    expected = max(1, min(int(chunk),
                                          budget // max(1, bytes_pv)))
            else:
                expected = int(legacy)
            realized.append(value)
            expected_all.append(expected)
            ok = ok and (expected is not None) and (value == expected)
        record[f"{direction}_view_batch_per_device"] = realized
        record[f"{direction}_view_batch_expected_per_device"] = expected_all
    record["num_pixels_full"] = num_pixels
    record["budget_bytes"] = int(budget)
    return record, ok


def torch_worker(cfg):
    """One ladder arm: one DISCARDED cold pass, then WARM_REPEATS warm repeats
    (protocols 7 and 9).  ``arm_class`` is 'pinned' or 'auto'."""
    import logging

    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    arm_class, n_dev = cfg["arm_class"], cfg.get("n_dev")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    # The auto arm pins NOTHING, on any backend: pinning the builder would
    # erase the very choice it exists to observe (the dp4 auto-arm rule).
    pin_devices = None if (cuda or arm_class == "auto") else [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=pin_devices)

    # ── the auto arm's two arm checks (§4) ───────────────────────────────────
    # (1) nothing pinned it, and (2) nothing called configure_devices -- under
    # the one-bit rule any call is explicit and would silently disable the
    # automatic path (tomography_model.py:803).
    configure_calls = []
    original_configure = model.configure_devices

    def counted_configure(*args, **kwargs):
        configure_calls.append(traceback.format_stack(limit=4))
        return original_configure(*args, **kwargs)
    model.configure_devices = counted_configure

    log_records = []

    class _Collector(logging.Handler):
        def emit(self, record):
            log_records.append(record.getMessage())
    collector = _Collector()
    collector.setLevel(logging.DEBUG)

    expect_kernels = (cuda, cuda)     # production: kernels on in both directions

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  pin_mechanism=("none (auto arm)" if arm_class == "auto" else
                                 ("MBIRTORCH_NUM_DEVICES" if cuda else
                                  "configure_devices(devices=[...]) "
                                  "-- CPU smoke only")),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  sino_elements=sino_elements(cell))
    # Protocol 6: the calibration mode owns max_memory_allocated; mg2 alone
    # sets it.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    if arm_class == "auto":
        result["auto_env_unpinned_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") is None)

    # ── arm check: the bodies bound, per direction (kb3's witnesses) ─────────
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_kernel_selected="triton" in fwd_name,
                  back_kernel_selected="triton" in back_name,
                  expected_bodies=list(expect_kernels))
    result["bodies_ok"] = ((result["fwd_kernel_selected"],
                            result["back_kernel_selected"]) == expect_kernels)

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    # The two candidate work-size metrics of §8, recorded on EVERY row from the
    # engine's own realized recon shape rather than from the formula, so the
    # thin probe's discrimination is read off measured quantities.
    result["recon_voxels"] = int(recon_shape[0] * recon_shape[1]
                                 * recon_shape[2])
    result["work_metric_product"] = (result["sino_elements"]
                                     * result["recon_voxels"])

    # ── the shared sinogram artifact, md5-verified ───────────────────────────
    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    result["sino_md5"] = actual_md5
    result["sino_md5_ok"] = (actual_md5 == expected_md5)
    if not result["sino_md5_ok"]:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5} (the Lustre "
                           f"corrupt-read failure mode)")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram),
                                               dtype=np.float64))

    def peaks():
        if not cuda:
            return []
        return [int(torch.cuda.max_memory_allocated(d))
                for d in model.sino_placement.devices]

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return _to_numpy(recon)

    keys_before = _launch_key_counts(geometry) if cuda else (0, 0)
    health = [sample_gpu_health()]

    # ── the cold pass, DISCARDED from the warm statistics (protocol 7) ───────
    # The auto arm takes its cold pass at VERBOSE 2, which is where the
    # selection loop logs its per-candidate rejections (tomography_model.py:985);
    # the warm repeats then run at verbose 0, so the extra logging cannot taint
    # the wall this arm contributes as the protocol-9 repeat.
    if arm_class == "auto":
        model.set_params(verbose=2, no_warning=True)
        model.logger.addHandler(collector)
        model.logger.setLevel(logging.DEBUG)
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    if arm_class == "auto":
        model.logger.removeHandler(collector)
        model.set_params(verbose=0, no_warning=True)
        # Both capture routes, because they fail differently: the log stream
        # can be swallowed by a handler change, and the attribute is only
        # populated when the search actually ran.
        result["auto_rejections"] = [
            [int(count), str(why)]
            for count, why in getattr(model, "device_choice_rejections", [])]
        result["auto_rejection_log"] = [line for line in log_records
                                        if "rejected" in line]
        result["auto_log_lines"] = len(log_records)
        result["auto_layout_is_automatic"] = bool(
            getattr(model, "device_layout_is_automatic", False))
        result["auto_torch_device"] = str(model.torch_device)
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the warm repeats (protocol 9) ────────────────────────────────────────
    if cuda:
        for device in model.sino_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    warm = []
    for _ in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        warm.append(time.perf_counter() - start)
        health.append(sample_gpu_health())
    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_min"] = min(warm)
    result["vcd_warm_max"] = max(warm)
    result["vcd_warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    # ── arm check: the REALIZED device list after the timed call (protocol 1) ─
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    if arm_class == "auto":
        # AN UNEXPECTED CHOICE IS A FINDING, NOT A CRASH (§4).  The expectation
        # is the pre-guard policy's: all visible devices.  On CPU/MPS the policy
        # short-circuits at `visible < 2` and the only possible answer is 1.
        expected_auto = result["visible_devices"] if cuda else 1
        result["auto_chosen_count"] = len(realized)
        result["auto_choice_as_expected"] = (len(realized) == expected_auto)
        result["auto_expected_count"] = expected_auto
        result["auto_configure_calls"] = len(configure_calls)
        result["auto_configure_never_called_ok"] = (len(configure_calls) == 0)
        result["devices_ok"] = True
    else:
        result["devices_ok"] = (len(realized) == n_dev) if cuda else \
            (len(realized) == len(pin_devices or [DEVICE]))

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (
            result["back_launch_keys_delta"] > 0
            and result["fwd_launch_keys_delta"] > 0)

    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    # ── arm check, POST-RUN: did a compile fall back to eager DURING the run? ─
    # `maybe_compile`'s guard is lazy: it compiles at the FIRST call and, on any
    # exception there, retries eagerly and rebinds permanently
    # (projectors.py:111-123).  Every arm here is the PRODUCTION configuration,
    # so a silent fallback would change a ladder time without changing anything
    # this harness otherwise records -- and the knee is read off exactly those
    # times.
    from mbirtorch import projectors as _projectors
    result["compile_mode"] = model.compile_mode
    result["compile_enabled"] = bool(model.compile_enabled)
    result["compile_errors_after_run"] = dict(_projectors._COMPILE_ERRORS)
    result["compile_fallback_free_ok"] = not _projectors._COMPILE_ERRORS

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build ONE sinogram per cell: phantom -> sinogram -> .npy, plus its md5
    sidecar.  Pinned to ONE device, so the artifact does not itself depend on
    the device count the arms are varying (see the module docstring)."""
    import numpy as np

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    model = _build_torch_model(geometry, cell, pin_devices=[DEVICE])
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
    return dict(cfg, framework="torch", role="generator", path=path,
                sino_md5=digest, sino_md5_ok=(_md5(path) == digest),
                sinogram_shape=list(sinogram.shape),
                recon_shape=list(recon_shape),
                sino_elements=sino_elements(cell),
                recon_voxels=int(recon_shape[0] * recon_shape[1]
                                 * recon_shape[2]),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


def _finish(result, out, cfg):
    """The common tail: checksum, a strided row sample (the ladder is a TIMING
    instrument, but a sample costs nothing and makes any surprise arguable), and
    the host peak."""
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result["recon_checksum"] = float(np.sum(np.abs(out), dtype=np.float64))
    step = max(1, out.shape[0] // SAMPLE_ROWS)
    path = _sample_path(cfg["arm_id"])
    np.save(path, out[::step])
    result["sample_path"] = path
    result["sample_step"] = step
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ── the runner ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits (kb3's
    rule).

    Protocol 1: a pinned arm pins ONLY through MBIRTORCH_NUM_DEVICES, which
    keeps the model on the automatic branch where the preflight still runs.
    The AUTO arm gets no pin at all -- and §6 records why that matters here
    specifically: both pin mechanisms bypass the proposed guard, so a guard that
    consulted floors on the pinned branch would silently override every pinned
    arm in this campaign.  This ladder measures the un-guarded policy, so the
    auto arm's environment must be clean."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # protocol 6
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # production: kernels on
    if cfg["arm_class"] != "auto" and cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg4_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg4_{cfg['arm_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    start = time.perf_counter()
    proc = subprocess.run([TORCH_PYTHON, "-u", os.path.abspath(__file__),
                           "_worker", cfg_path, out_path], env=arm_env(cfg))
    subprocess_wall = time.perf_counter() - start
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(cfg, error=f"worker exited {proc.returncode}")
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    # Protocol 7: the total subprocess wall is recorded even though the warm
    # protocol discards it, because §6's cadence decision needs exactly that.
    row["subprocess_wall_s"] = subprocess_wall
    return row


def _arm(arm_class, family, geometry, cell, n_dev):
    tag = f"n{n_dev}" if n_dev else "auto"
    return dict(framework="torch", arm_class=arm_class, family=family,
                geometry=geometry, cell=list(cell), n_dev=n_dev,
                arm_id=f"{_cell_tag(geometry, cell)}_{tag}")


def build_plan(cells, counts):
    """The arm plan, in JOB ORDER.

    PHASE 0 is every generator and every n=1 arm, so a truncated job still
    yields the n=1 column the whole ladder is normalized against (protocol 11).

    PHASE 1 alternates the direction of each cell's n>1 block -- (2, 4, auto)
    then (auto, 4, 2) -- which is protocol 9's blocked-and-reversed pattern
    applied ACROSS adjacent ladder cells, the axis the knee is read along.  A
    literal within-cell palindrome is not available: §4 budgets four arms per
    cell (n=1, 2, 4, auto), so no count appears twice within one cell except the
    all-device pair (pinned n=max and auto), which is exactly the pair §4
    assigns the protocol-9 repeat role to.  Every row carries `block_index` /
    `block_size` so the analysis can regress out what the alternation cannot
    cancel."""
    skip_auto = os.environ.get("MG4_SKIP_AUTO", "0") == "1"
    higher = [n for n in counts if n != 1]
    phase0, phase1 = [], []
    for family, geometry, cell in cells:
        phase0.append(dict(framework="torch", arm_class="generator",
                           family=family, geometry=geometry, cell=list(cell),
                           n_dev=None,
                           arm_id=f"{_cell_tag(geometry, cell)}_generator"))
    if 1 in counts:
        for family, geometry, cell in cells:
            phase0.append(_arm("pinned", family, geometry, cell, 1))
    for index, (family, geometry, cell) in enumerate(cells):
        block = [_arm("pinned", family, geometry, cell, n) for n in higher]
        if not skip_auto:
            block.append(_arm("auto", family, geometry, cell, None))
        phase1.extend(block if index % 2 == 0 else list(reversed(block)))
    return phase0, phase1


# ── the summary ───────────────────────────────────────────────────────────────
def print_cell(family, geometry, cell, rows):
    """The raw ladder readout for one cell.  Speedups are PRINTED beside their
    spreads; the admission rule of §4 is applied in ANALYSIS over the whole
    table, never here, because it reads every LARGER cell too."""
    header = (f"\n===== {family}  {geometry} {tuple(cell)}  "
              f"sino_elements {sino_elements(cell):,} =====")
    print(header)
    print(f"{'arm':>10}{'n':>4}{'cold_s':>9}{'warm_s':>9}{'spread':>8}"
          f"{'peakGB':>8}{'devs':>6}{'dev':>5}{'body':>6}{'vb':>4}{'hot':>5}")
    baseline = None
    for row in rows:
        if row.get("error"):
            print(f"{row.get('arm_class', '?'):>10}"
                  f"{str(row.get('n_dev', '-')):>4}  ERROR: "
                  f"{str(row['error'])[:100]}")
            continue

        def flag(key):
            value = row.get(key)
            return "-" if value is None else ("ok" if value else "FAIL")
        if row.get("arm_class") == "pinned" and row.get("n_dev") == 1:
            baseline = row.get("vcd_warm")
        spread = row.get("vcd_warm_spread")
        print(f"{row['arm_class']:>10}{str(row.get('n_dev') or '-'):>4}"
              f"{row.get('vcd_cold', 0):>9.2f}{row.get('vcd_warm', 0):>9.2f}"
              f"{(f'{spread:.1%}' if spread is not None else '-'):>8}"
              f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:>8.2f}"
              f"{row.get('realized_n_devices', 0):>6}"
              f"{flag('devices_ok'):>5}{flag('bodies_ok'):>6}"
              f"{flag('vb_ok'):>4}"
              f"{('HOT' if row.get('gpu_hot') else '-'):>5}")
    if baseline:
        for row in rows:
            warm = row.get("vcd_warm")
            if not warm or row.get("n_dev") == 1:
                continue
            label = (f"n={row['n_dev']}" if row.get("n_dev")
                     else f"auto(n={row.get('realized_n_devices')})")
            spread = row.get("vcd_warm_spread") or 0.0
            print(f"    speedup over n=1, {label:>14}: "
                  f"{baseline / warm:.3f}x   (this arm's spread "
                  f"{spread:.1%}; §4's admission rule is applied in analysis)")
    for row in rows:
        if row.get("arm_class") != "auto" or row.get("error"):
            continue
        visible = row.get("visible_devices")
        seen = (f"{visible} CUDA-visible" if row.get("cuda") else
                f"{visible} CUDA-visible -- non-CUDA host, so the policy "
                f"short-circuits at `visible < 2` and 1 is the only answer")
        print(f"    AUTO: chose {row.get('auto_chosen_count')} of {seen} "
              f"(expected {row.get('auto_expected_count')}; "
              f"as expected: {row.get('auto_choice_as_expected')}), "
              f"device {row.get('auto_torch_device')}, "
              f"configure_devices calls {row.get('auto_configure_calls')}, "
              f"env unpinned {row.get('auto_env_unpinned_ok')}")
        for entry in row.get("auto_rejections") or []:
            print(f"          rejected {entry[0]} devices: {entry[1]}")
        for line in row.get("auto_rejection_log") or []:
            print(f"          log: {line}")
        if not (row.get("auto_rejections") or row.get("auto_rejection_log")):
            print("          (no rejections recorded: the policy took the "
                  "largest count, or the search short-circuited)")


def print_thin_probe(rows_by_cell, probe_cell):
    """§8's discrimination read-out: what the probe's outcome says about which
    work-size metric the guard should use.  The verdict itself belongs to the
    analysis; this prints the numbers the verdict is made from."""
    key = ("thin_probe", "parallel", tuple(probe_cell))
    rows = rows_by_cell.get(key)
    if not rows:
        return
    print(f"\n===== THIN-VOLUME PROBE {tuple(probe_cell)} -- §8's "
          f"discrimination =====")
    first = next((r for r in rows if not r.get("error")), None)
    if first:
        print(f"  sinogram elements {first.get('sino_elements'):,}  "
              f"recon voxels {first.get('recon_voxels'):,}  "
              f"recon_shape {first.get('recon_shape')}")
    if tuple(probe_cell) != THIN_PROBE:
        # The smoke's probe point is a plumbing stand-in, not a discriminating
        # shape; claiming otherwise here would be a lie in the log.
        print("  (SMOKE stand-in shape: the discrimination arithmetic below "
              "applies to the real probe point only.)")
    elif first:
        # Defaults, never a StopIteration: a summary must not die after a
        # six-hour job because a ladder constant moved.
        above = next((c for c in PARALLEL_LADDER
                      if sino_elements(c) == first.get("sino_elements")), None)
        below = next((c for c in PARALLEL_LADDER
                      if recon_voxels_parallel(c) > first.get("recon_voxels")),
                     None)
        print(f"  by SINOGRAM ELEMENTS this point equals the {above} ladder "
              f"cell (one step ABOVE the expected 256-class knee).")
        print(f"  by RECON VOXELS it is smaller than the {below} ladder cell "
              f"(one step BELOW the expected knee).")
        print("  Whichever side the measured best count lands on is the "
              "metric the outcome vindicates.")
    baseline = next((r.get("vcd_warm") for r in rows
                     if r.get("arm_class") == "pinned" and r.get("n_dev") == 1),
                    None)
    for row in rows:
        warm = row.get("vcd_warm")
        if not (baseline and warm):
            continue
        label = (f"n={row['n_dev']}" if row.get("n_dev")
                 else f"auto(n={row.get('realized_n_devices')})")
        print(f"    {label:>16}  warm {warm:.3f}s  "
              f"speedup over n=1 {baseline / warm:.3f}x")


def _cleanup(paths):
    if os.environ.get("MG4_KEEP_ARTIFACTS", "0") == "1":
        return
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def main():
    cells, counts = selected_plan()
    phase0, phase1 = build_plan(cells, counts)
    plan = phase0 + phase1
    measured = [c for c in plan if c["arm_class"] != "generator"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR,
                            f"mg4_ladder_{RUN_LABEL}_{stamp}.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"mg4 crossover ladder on {RUN_LABEL} ({DEVICE}); "
          f"{len(cells)} cells, counts {counts} + auto")
    print(f"  {len(measured)} measured arms + "
          f"{len(plan) - len(measured)} generators -> {out_path}")
    print("  (§7 expects 28 parallel + 12 cone + 1 probe POINT = 44 arms at "
          "the full plan)")

    rows_by_cell, all_rows = {}, []
    with open(out_path, "w") as sink:
        for index, cfg in enumerate(plan):
            cfg = dict(cfg, block_index=index, block_size=len(plan))
            print(f"  [{index + 1}/{len(plan)}] {cfg['arm_id']} "
                  f"({cfg['arm_class']}) ...", flush=True)
            row = run_one(cfg)
            all_rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()                       # protocol 11: incremental jsonl
            if cfg["arm_class"] != "generator":
                key = (cfg["family"], cfg["geometry"], tuple(cfg["cell"]))
                rows_by_cell.setdefault(key, []).append(row)

        for (family, geometry, cell), rows in rows_by_cell.items():
            ordered = sorted(rows, key=lambda r: (r.get("n_dev") or 99))
            print_cell(family, geometry, cell, ordered)
        probe = SMOKE_THIN_PROBE if SMOKE else THIN_PROBE
        print_thin_probe(rows_by_cell, probe)

        summary = dict(kind="summary", run_label=RUN_LABEL,
                       host=platform.node(), measured_arms=len(measured),
                       cells=[[f, g, list(c)] for f, g, c in cells],
                       errors=[r.get("arm_id") for r in all_rows
                               if r.get("error")],
                       arm_check_failures=[
                           r.get("arm_id") for r in all_rows
                           if False in (r.get("devices_ok"),
                                        r.get("bodies_ok"),
                                        r.get("vb_ok"),
                                        r.get("sino_md5_ok"),
                                        r.get("kernels_launched_ok"),
                                        r.get("calibration_absent_ok"),
                                        r.get("compile_fallback_free_ok"),
                                        r.get("auto_env_unpinned_ok"),
                                        r.get("auto_configure_never_called_ok"))],
                       auto_findings=[
                           dict(arm_id=r.get("arm_id"),
                                chose=r.get("auto_chosen_count"),
                                expected=r.get("auto_expected_count"),
                                rejections=r.get("auto_rejections"))
                           for r in all_rows
                           if r.get("arm_class") == "auto"
                           and r.get("auto_choice_as_expected") is False],
                       hot_rows=[r.get("arm_id") for r in all_rows
                                 if r.get("gpu_hot")])
        sink.write(json.dumps(summary) + "\n")

    print(f"\nerrors: {summary['errors'] or 'none'}")
    print(f"arm-check failures: {summary['arm_check_failures'] or 'none'}")
    print(f"auto-arm FINDINGS (an unexpected choice is recorded, not a crash): "
          f"{summary['auto_findings'] or 'none'}")
    print(f"hot rows (protocol 11: re-run only if the clock is ALSO "
          f"depressed): {summary['hot_rows'] or 'none'}")
    _cleanup([_sino_path(g, c) for _f, g, c in cells]
             + [_sample_path(c["arm_id"]) for c in measured])
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        if cfg["arm_class"] == "generator":
            row = generator_worker(cfg)
        else:
            row = torch_worker(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)


def _dry_run():
    cells, counts = selected_plan()
    phase0, phase1 = build_plan(cells, counts)
    print("cells, in job order:")
    for family, geometry, cell in cells:
        voxels = (recon_voxels_parallel(cell) if geometry == "parallel"
                  else None)
        print(f"  {family:>18} {geometry:>8} {str(cell):>18}  "
              f"sino {sino_elements(cell):>14,}  "
              f"voxels {('%14s' % f'{voxels:,}') if voxels else '%14s' % '-'}")
    for label, block in (("PHASE 0 (generators + every n=1 arm)", phase0),
                         ("PHASE 1 (alternating n>1 blocks)", phase1)):
        print(f"\n{label}: {len(block)} arms")
        for cfg in block:
            print(f"   {cfg['arm_id']:>34}  {cfg['arm_class']:>10}  "
                  f"n={cfg['n_dev']}")
    measured = [c for c in phase0 + phase1 if c["arm_class"] != "generator"]
    print(f"\ntotal {len(phase0) + len(phase1)} arms ({len(measured)} measured, "
          f"{len(phase0) + len(phase1) - len(measured)} generators)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    elif "--dry-run" in sys.argv:
        _dry_run()
    else:
        main()
