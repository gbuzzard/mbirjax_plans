"""mg11 -- the combined cluster gate campaign for the forward column gather, on
BOTH geometries, against the SHIPPED implementation.

WHAT THIS FILE IS FOR.  The column gather is written, tested and committed.  It
is off by default.  This job is what decides whether the default flips.  Its
defining property is stated first because everything else follows from it:

    mg11 PATCHES NOTHING.  Every arm drives the library exactly as a user would
    -- one environment variable and one documented model attribute -- and
    measures what the shipped code does.  mg10 had to build its own prototype
    because the shape did not exist yet, and a prototype's numbers are its own.
    mg11's numbers are the library's.  There is no install_* function in this
    file and there must never be one: the moment an arm patches the driver, the
    thing being gated stops being the thing that will ship.

THE TWO KNOBS, and nothing else.
    MBIRTORCH_FORWARD_COLUMN_GATHER=1  in the ARM's child environment forces the
        gather on; =0 forces it off.  The library reads it per call
        (tomography_model.COLUMN_GATHER_ENV_VAR), which is what lets one job run
        both shapes over the same inputs.
    model.forward_project_pixel_batch = B  sets the pixel-column batch, the
        documented override of tomography_model.FORWARD_PIXEL_BATCH.
Every flag-on arm sets both.  Every control sets the environment variable to
'0' -- explicitly OFF rather than merely unset, so a stale export in a
submitting shell cannot turn a control into a treatment.

THE RULINGS THIS EXECUTES.  forward_remedy_design.md section 13 (RULED, Greg,
2026-08-10, late): the parallel extension is approved, the two geometries' gates
run as ONE combined campaign, and the defaults flip when the gates pass without
a further ruling per geometry.  Section 7.2 carries the value gates and, with
them, the bar: the SHIPPED PARITY FLOOR governs, and the e-6 class measured by
mg10 is a LOGGED EXPECTATION that is never a failing assertion.  One caveat is
recorded with the approval and this job answers it -- see THE PARALLEL CAVEAT
below.

THE THREE GATE COLUMNS, one per open question, computed per arm.

  1. TIMING -- mg9's and mg10's instrument, unchanged.  A bracket around the
     forward funnel (host wall plus a CUDA event pair per device), a busy probe
     around every forward projection BODY call keyed by DEVICE POSITION, the
     per-launch time that busy over calls gives, and the composed reconstruction
     wall.  The speed verdict compares the best flag-on arm against the flag-off
     control at the SAME (geometry, device count), both measured in this job, on
     this node, in this process ordering.

  2. VALUE -- two independent distance families, each read against a floor
     measured in the same arm.  The CHECKSUM family is the relative difference
     of whole-volume sums of absolute value, three per arm.  The SAMPLE family
     is a relative L2 and a max-relative-to-peak over a strided sample of the
     reconstruction, which is section 7.2's own metric.  Each arm carries its
     own repeat-to-repeat distance -- both forward kernels accumulate with float
     atomics and are not bit-reproducible, so the floor is not zero -- and its
     distances to the same-count control and to the one-device anchor.
       The FAILING BAR is the shipped parity floor and nothing else.  That floor
     is 5e-3 relative, read out of the standing suites: tests/test_kernels_
     sharded.py line 60 `FLOOR = 5e-3` (the two-device CUDA kernel arms), and
     tests/test_sharding.py `assert rel < 5e-3` in
     test_column_gather_recon_matches_single_device and in the sparse-view cone
     cases beside it.  Those tests take max|a-b| / max|b|, so this file reads
     the bar against the SAME functional form (max_rel_of_peak) and not against
     the L2.
       The REGISTERED EXPECTATION is separate and never fails anything.  mg10
     measured the gather 1.47e-06 to 1.54e-06 in relative L2 from the one-device
     anchor.  A reading that leaves that class by an order of magnitude gets a
     printed marker so a human weighing the tradeoff can see it.  Section 7.2
     ruled that the shipped parity floor governs; a marker is not a gate.

  3. MEMORY -- the calibration confirmation the implementation deferred to
     hardware.  Every arm reports its per-device MEASURED peak
     (torch.cuda.max_memory_allocated) and the ledger's own MODELED peak for
     that arm's exact configuration, and prints modeled/measured per device.
     A ratio at or above 1.00 means the ledger over-predicts, which is what a
     preflight must do; a ratio below 1.00 is a FLOOR VIOLATION and fails that
     geometry's gate.
       The modeled number is not re-derived here.  It comes from the library's
     own planning entry point, called the way the library calls it:
         model._build_memory_ledger(devices=<the settled placement>,
                                    partition_sequence=None, weights=weights,
                                    init_recon=None, fm_hessian=None,
                                    prox_input=None, init_error_sinogram=None)
     which is tomography_model.TomographyModel._build_memory_ledger, itself
     _memory_ledger.estimate_peak_device_bytes(_memory_ledger.plan_from_model(
     ...)).  Those are exactly the call arrays TomographyModel.recon hands
     _apply_device_policy for the call this arm makes.  Because plan_from_model
     reads `model._column_gather_forward()` and `model._forward_pixel_batch()`
     off the model rather than re-deriving them, a flag-on arm's plan prices the
     ACTIVE path automatically -- and the arm asserts that it did, by requiring
     plan.column_pixel_batch to equal the batch it asked for.

THE PARALLEL CAVEAT, section 13, and how this job answers it.  The two-times
prediction for parallel was measured with the PIXEL COUNT HELD FULL: cutting a
one-device call into two 504-wide pieces doubled its cost, so full-width blocks
at every device count should halve the forward busy.  The column gather does
give full-width blocks, but it also cuts the pixel axis to the batch, and no
measurement covers that.  So the parallel batch sweep is wider than cone's --
4096, 8192, 16384 and 32768 -- and the prediction is judged against the SWEPT
BEST rather than against batch 8192 alone.  The transient one gathered cylinder
holds is batch x slices x 4 B, which is 132.1 MB at batch 32768 and the 1008-
slice cell, so the whole sweep stays under the 150 MB the ruling names.  The
summary prints the caveat check as measured-against-predicted and draws no
conclusion from it; the analysis is the session lead's.

WHAT EVERY ARM PROVES ABOUT ITSELF, and why this is harder without patches.  A
patched arm can count its own patch.  A patchless arm has no counter of its own,
so its witnesses are built out of things the LIBRARY does, observed from
outside:

    THE RESOLVER.  model._column_gather_forward() is the library's own answer to
    "which forward runs".  Every arm reads it and requires it to agree with the
    arm's name.
    THE LEDGER PLAN.  _memory_ledger.plan_from_model(...).column_pixel_batch is
    the batch on the gather path and None on the banded path.  It is written by
    a different module than the driver, so it is an independent second reading
    of the same switch.
    THE PRIMITIVES.  _sharding.gather_column_band is called only by
    _sparse_forward_project_columns, and _sharding.broadcast_band_to_views only
    by the banded _sparse_forward_project_sharded (grep says both, in the
    shipped tree).  Counting them separates the two shapes at the transfer
    layer.  This is the same witness the library's own test uses: test_sharding.
    py::test_the_banded_walk_is_what_runs_with_the_switch_off replaces
    gather_column_band with a function that raises.
    THE KERNEL'S OWN VIEW.  The busy probe records the shape of every values
    block a forward body was handed.  On the gather path that block is
    (pixel batch) x (the WHOLE device-form slice axis); on the banded path it is
    (all the pass's pixels) x (one owner's shard).  A single column count equal
    to the full slice count is what separates a column gather from a band by
    another name, and it is recorded by a different wrapper, on a different
    function, than the gather count is -- so a witness that agreed with itself
    cannot pass.

    Controls carry the mirror image of all four: the resolver False, the plan's
    column_pixel_batch None, ZERO gathers with a positive fan-out count, and
    values blocks one shard wide.  An arm that disagrees with any of these
    ABORTS; it does not report.

THE ARMS.  Twenty measured arms plus two untimed sinogram generators, all at
mg9's and mg10's cell (1024, 1008, 992), which gives 1008 slices -- divisible by
1, 2 and 4, so nothing is padded and no arm's arithmetic carries a padding term.
    cone,     1 device:  c1, the value anchor (the shipped single-device path,
                         which already makes the full-height calls)
    cone,     2 devices: c2_off (flag off), then flag on at 8192, 16384, 32768
    parallel, 1 device:  p1, the value anchor
    parallel, 2 devices: p2_off, then flag on at 4096, 8192, 16384, 32768
    cone,     4 devices: c4_off, then the three cone batches
    parallel, 4 devices: p4_off, then the four parallel batches
ARM ORDER is by decreasing value, because rows are written incrementally and a
truncated job should lose the least: each geometry's anchor comes before the
arms that are measured against it, and the two-device blocks -- where both
predictions live -- come before the four-device ones.

THE VERDICT BLOCK.  The log ends with one block per geometry, mechanical and
quotable: the speed comparison with the batch that won, the value maximum
against the floor bar with the expectation marker beside it, the smallest
modeled/measured ratio, and one of
    GATES PASS - FLIP AUTHORIZED: <geometry>
    GATES FAIL: <geometry>: <which gate>
and nothing else.  There is no prose verdict in this file, deliberately: the
harness states the readings and the rule, and the reading of them is the session
lead's.

TERMS OF ART, each defined once, here.
    arm          one subprocess run at fixed parameters -- one geometry, one
                 device count, and the flag either on at one batch or off.
    control      an arm with MBIRTORCH_FORWARD_COLUMN_GATHER=0 at a device count
                 above one: the shipped banded walk, which is what the flag-on
                 arms at that count are compared against.
    anchor       the one-device arm of a geometry.  Its placement is trivial, so
                 neither multi-device forward runs at all; it is the value both
                 shapes are trying to reproduce.
    flag-on      an arm with the environment variable set to 1 and a pixel
                 batch set on the model.
    composed     one whole timed reconstruction's wall.
    bracket      the forward funnel's per-device CUDA event span.
    busy         the sum of per-body-call event spans on one device.
    per-launch   busy over the body call count, on one device.
    repeat floor an arm's own pass-to-pass distance, on either metric.
    modeled peak the ledger's per-device peak for this arm's configuration.
    measured peak torch.cuda.max_memory_allocated on that device.

ENVIRONMENT KNOBS (all optional).
    MG11_ARMS=c1,c2_off,...        run only these tokens (the dry-run prints
                                   them); the trim order is in the sbatch header
    MG11_CONE_BATCHES=8192,32768   override cone's swept batches
    MG11_PARALLEL_BATCHES=...      override parallel's swept batches
    MG11_ITERATIONS=3              VCD iterations per reconstruction
    MG11_WARM_REPEATS=3            timed reconstructions after the cold pass
    MG11_MAX_EVENT_PAIRS=400000    per-reconstruction event budget
    MG11_KEEP_ARTIFACTS=1          keep the sinograms and the value samples
    MG11_SMOKE=1                   the local CPU smoke (tiny cell, few iters)
    MG11_DEVICE=cpu                smoke device
"""

import functools
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# mg9's and mg10's cell, and nothing else.  cell = (num_views, num_det_rows,
# num_det_channels); at this cell both geometries give recon (992, 992, 1008),
# so the slice count is 1008 and it divides 1, 2 and 4 exactly.
CELL = (1024, 1008, 992)

SMOKE = os.environ.get("MG11_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG11_DEVICE", "cpu" if SMOKE else "cuda")

# The library's own name for the switch, imported in the worker rather than
# spelled here, so a rename in the library is a loud import error and not a
# silently ineffective export.  This string is the fallback used by the parent
# process (which never imports torch) and is asserted against the library's in
# every worker.
COLUMN_GATHER_ENV_VAR = "MBIRTORCH_FORWARD_COLUMN_GATHER"


def _int_list(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(default)
    out = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out or list(default)


# The swept pixel batches.  Cone's three start at the library's own default
# (FORWARD_PIXEL_BATCH = 8192, which mg10 measured as the largest value tried
# and still falling) and go up from there.  Parallel's four reach lower as well
# as higher, because section 13's caveat is precisely that the batch cuts the
# pixel axis and the two-times prediction did not: a sweep that started at 8192
# could not tell a batch effect from a shape effect.
CONE_BATCHES = _int_list("MG11_CONE_BATCHES",
                         (16,) if SMOKE else (8192, 16384, 32768))
PARALLEL_BATCHES = _int_list("MG11_PARALLEL_BATCHES",
                             (16, 64) if SMOKE else (4096, 8192, 16384, 32768))

VCD_ITERATIONS = int(os.environ.get("MG11_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13          # mg1's / mg5's / mg9's / mg10's seed; the arms stay comparable
WARM_REPEATS = max(1, int(os.environ.get("MG11_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# ── THE BARS AND THE PREDICTIONS, all quoted, none invented here ──────────────
# THE FAILING BAR.  The relative tolerance the standing multi-device parity
# suites hold, read out of the tests themselves:
#   tests/test_kernels_sharded.py line 60:  FLOOR = 5e-3
#       "The established multi-device float-divergence scale for cells of this
#        class", applied to the forward and back kernel arms on two CUDA devices.
#   tests/test_sharding.py, test_column_gather_recon_matches_single_device:
#       assert rel < 5e-3     # the shipped parity floor, as above
#       with rel = max|out - ref| / max|ref|.
# That functional form is why the bar below is read against max_rel_of_peak and
# not against the relative L2.
PARITY_FLOOR = 5e-3
PARITY_FLOOR_CITATION = (
    "tests/test_kernels_sharded.py:60 FLOOR = 5e-3, and tests/test_sharding.py "
    "test_column_gather_recon_matches_single_device 'assert rel < 5e-3' with "
    "rel = max|out-ref| / max|ref|")

# THE REGISTERED EXPECTATION, which fails nothing.  mg10 measured the column
# gather 1.47e-06 to 1.54e-06 in relative L2 from the one-device anchor at this
# cell (design note 7.2), against banded distances of 2.83e-07 and 5.41e-07.
# The marker below fires when a reading leaves that class by an order of
# magnitude.  It is a marker, not a threshold: section 7.2 ruled that the
# shipped parity floor governs shape C, and nothing here asserts this number.
EXPECTATION_REL_L2 = 1.5e-06
EXPECTATION_MARKER_AT = 1e-05

# SECTION 13's parallel prediction, stated before the measurement so it cannot
# be fitted to it.  "parallel 1024 at two devices should fall from 28.2 to about
# 14.1 s of forward busy and from 39.2 to about 25 s composed."
PARALLEL_PREDICTION = {
    2: dict(busy_from_s=28.2, busy_to_s=14.1, composed_from_s=39.2,
            composed_to_s=25.0),
}
# The transient bound the ruling names for the swept parallel batches.
TRANSIENT_BOUND_BYTES = 150 * 1000 * 1000

# mg10's measured per-arm subprocess walls on this cell and this node class,
# cold pass and torch import included (rows mg10_shape_sweep_h004_20260810_
# 174925.jsonl and _h008_20260810_201612.jsonl).  These are what the wall
# arithmetic is built on, and the cone flag-on entries are MEASURED rather than
# assumed: mg10's c2_8192 and c4_8192 ran this exact shape.
MG10_WALL_S = {
    ("cone", 1, "off"): 281, ("cone", 2, "off"): 299, ("cone", 4, "off"): 262,
    ("cone", 2, "on"): 264, ("cone", 4, "on"): 193,
    ("parallel", 1, "off"): 187, ("parallel", 2, "off"): 191,
    ("parallel", 4, "off"): 135,
}
GENERATOR_S = 68
# Cone's flag-on walls are measured, so their band is narrow.  Parallel's are
# not measured at all -- that is what this job is for -- so they take mg10's own
# rule for an unmeasured arm: the low end assumes the arm costs what its control
# costs, the high end assumes half again as much.
CONE_ON_HIGH_FACTOR = 1.15
PARALLEL_ON_HIGH_FACTOR = 1.50
CONTROL_HIGH_FACTOR = 1.05

# Reconciliation tolerance, mg1's constant.
RECONCILE_SLACK = 0.02

RESULTS_DIR = os.environ.get(
    "MG11_RESULTS", os.path.dirname(os.path.abspath(__file__)))
RUN_LABEL = platform.node().split(".")[0]

SPEC = {
    "parallel": dict(kernel_module="mbirtorch.triton_parallel",
                     fwd_chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                     back_chunk_const="PARALLEL_BACK_VIEW_CHUNK"),
    "cone": dict(kernel_module="mbirtorch.triton_cone",
                 fwd_chunk_const="CONE_FWD_VIEW_CHUNK",
                 back_chunk_const="CONE_BACK_VIEW_CHUNK"),
}

# The value sample: one reconstruction voxel out of every VALUE_SAMPLE_TARGET
# per axis, so the sample is a few hundred thousand values whatever the cell.
# This is section 7.2's metric -- "a relative L2 over every fifteenth voxel".
VALUE_SAMPLE_TARGET = 64

# ── the region definitions and GPU-health machinery, COPIED from mg9/mg10 ─────
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
NESTED_REGIONS = ("band_reduce",)
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
MAX_EVENT_PAIRS = int(os.environ.get("MG11_MAX_EVENT_PAIRS", "400000"))

HOT_CORE_C = 85
HOT_HBM_C = 95
CLOCK_DEPRESSED_FRAC = 0.90
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")
# ──────────────────────────────────────────────────────────────────────────────


# ── the arm plan ──────────────────────────────────────────────────────────────
def build_arms():
    """Every arm, in job order (least valuable last -- rows write
    incrementally).  Each entry is a dict describing one subprocess run.

    Order: each geometry's anchor before the arms measured against it, and both
    two-device blocks -- where the cone measurement and the parallel prediction
    both live -- before the four-device ones."""
    arms = []

    def add(token, geometry, n_dev, batch=None, role="control"):
        arms.append(dict(token=token, geometry=geometry, n_dev=n_dev,
                         pixel_batch=batch, role=role,
                         gather=(batch is not None)))

    def block(prefix, geometry, n_dev, batches):
        add(f"{prefix}{n_dev}_off", geometry, n_dev, role="control")
        for batch in batches:
            add(f"{prefix}{n_dev}_on{batch:05d}", geometry, n_dev, batch=batch,
                role="gather")

    add("c1", "cone", 1, role="anchor")
    block("c", "cone", 2, CONE_BATCHES)
    add("p1", "parallel", 1, role="anchor")
    block("p", "parallel", 2, PARALLEL_BATCHES)
    block("c", "cone", 4, CONE_BATCHES)
    block("p", "parallel", 4, PARALLEL_BATCHES)
    return arms


ARMS = build_arms()


def selected_arms():
    """The arms to run, narrowed by MG11_ARMS (comma-separated tokens)."""
    by_token = {a["token"]: a for a in ARMS}
    raw = os.environ.get("MG11_ARMS", "").strip()
    if not raw:
        chosen = list(ARMS)
    else:
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in by_token:
                raise ValueError(f"MG11_ARMS: {token!r} is not one of "
                                 f"{sorted(by_token)}")
            wanted.add(token)
        if not wanted:
            raise ValueError(f"MG11_ARMS: no valid tokens in {raw!r}")
        chosen = [a for a in ARMS if a["token"] in wanted]      # declared order
    if SMOKE and not os.environ.get("MG11_ARMS", "").strip():
        # The smoke keeps one arm of every KIND at every geometry -- anchor,
        # control and flag-on -- so every witness, every negative witness and
        # every gate column is exercised on CPU, and drops the four-device arms,
        # which add nothing the two-device ones do not.
        keep = {"c1", "c2_off", f"c2_on{CONE_BATCHES[0]:05d}",
                "p1", "p2_off", f"p2_on{PARALLEL_BATCHES[0]:05d}"}
        chosen = [a for a in ARMS if a["token"] in keep]
    return chosen


def anchor_token(geometry):
    """The one-device arm of a geometry: what every value row is read against."""
    return "c1" if geometry == "cone" else "p1"


def control_token(geometry, n_dev):
    """The flag-off arm at this device count."""
    return f"{'c' if geometry == 'cone' else 'p'}{n_dev}_off"


def cell_for(_geometry):
    return SMOKE_CELL if SMOKE else CELL


def num_slices_for(cell):
    """The reconstruction's slice count at this cell.  Both geometries derive it
    from the detector row count; the worker asserts this against the model's own
    recon_shape before it is used for anything."""
    return int(cell[1])


def cylinder_bytes(batch, num_slices):
    """What ONE gathered cylinder holds: the batch's pixel columns at every
    slice, float32.  This is the transient the ruling's 150 MB bound is on."""
    return int(batch) * int(num_slices) * 4


# ── staged-artifact mechanics (mg5's / mg9's / mg10's md5 discipline) ─────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg11_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id, index):
    return os.path.join(RESULTS_DIR, f"_mg11_sample_{arm_id}_p{index}.npy")


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
    recorded failure class is re-detaching that result."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    gather = getattr(x, "gather", None)
    if callable(gather) and hasattr(x, "placement"):
        return gather()
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weights(sinogram):
    """One weighting formula, one dtype, every arm (mg1's rule)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks, temperatures and active throttle reasons."""
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
        rows = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 9:
                rows.append(dict(
                    index=_gi(parts[0]), sm_mhz=_gi(parts[1]),
                    mem_mhz=_gi(parts[2]), core_c=_gi(parts[3]),
                    hbm_c=_gi(parts[4]),
                    throttle=[name for name, value
                              in zip(_THROTTLE_NAMES, parts[5:9])
                              if value.lower() in ("active", "1")]))
            elif len(parts) >= 3:
                rows.append(dict(index=_gi(parts[0]), sm_mhz=_gi(parts[1]),
                                 mem_mhz=None, core_c=_gi(parts[2]),
                                 hbm_c=None, throttle=[]))
        if rows:
            return rows
    return []


def worst_health(samples):
    """The worst reading per device over the samples taken around an arm."""
    worst = {}
    for sample in samples:
        for row in sample:
            index = row.get("index")
            if index is None:
                continue
            slot = worst.setdefault(index, dict(index=index, sm_mhz=None,
                                                mem_mhz=None, core_c=None,
                                                hbm_c=None, throttle=[]))
            for key, better in (("core_c", max), ("hbm_c", max),
                                ("sm_mhz", min), ("mem_mhz", min)):
                value = row.get(key)
                if value is None:
                    continue
                slot[key] = value if slot[key] is None else better(slot[key],
                                                                   value)
            for name in row.get("throttle") or ():
                if name not in slot["throttle"]:
                    slot["throttle"].append(name)
    return [worst[k] for k in sorted(worst)]


def row_is_hot(health):
    """Whether any device ran hot, throttled, or at a depressed clock."""
    if not health:
        return None
    peak_sm = max((h["sm_mhz"] for h in health if h.get("sm_mhz")), default=None)
    for row in health:
        if row.get("throttle"):
            return True
        if row.get("core_c") and row["core_c"] >= HOT_CORE_C:
            return True
        if row.get("hbm_c") and row["hbm_c"] >= HOT_HBM_C:
            return True
        if (peak_sm and row.get("sm_mhz")
                and row["sm_mhz"] < peak_sm * CLOCK_DEPRESSED_FRAC):
            return True
    return False


# ── INSTRUMENT 0: per-region host walls and device spans (mg9's, unchanged) ───
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    reconstruction loop's calling thread.  Copied from mg10_shape_sweep.py,
    which copied it from mg9_fwd_instrument.py, which copied it from
    mg1_readout.py without change -- so mg11's forward bracket IS mg5's, mg9's
    and mg10's forward bracket, and the control arms are comparable with theirs.

    CUDA path: for each device in the region's placement a start and an end
    event are CREATED AND RECORDED inside ``with torch.cuda.device(dev)``, on
    that device's current stream.  The end event is recorded AFTER the call
    returns, so it queues behind everything the call enqueued.  Elapsed times
    are read only in :meth:`finish`, after a per-device synchronize.

    CPU path (the local smoke ONLY): perf_counter walls stand in behind the same
    interface.  Two smoke artifacts follow from virtual cpu devices sharing one
    name: the per-device map collapses to a single ``'cpu'`` key, and its span
    sum is the host wall times the device count.
    """

    def __init__(self, torch_module, cuda):
        self.torch = torch_module
        self.cuda = cuda
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.devices_seen = {region: [] for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"

    def reset(self):
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False

    def _start(self, region, devices):
        for device in devices:
            name = str(device)
            if name not in self.devices_seen[region]:
                self.devices_seen[region].append(name)
        if not self.cuda:
            return None
        if self.pair_count + len(devices) > MAX_EVENT_PAIRS:
            self.cap_hit = True
            return None
        events = []
        for device in devices:
            with self.torch.cuda.device(device):
                start = self.torch.cuda.Event(enable_timing=True)
                start.record()
            events.append((device, start))
        return events

    def _stop(self, region, events):
        if events is None:
            return
        for device, start in events:
            with self.torch.cuda.device(device):
                end = self.torch.cuda.Event(enable_timing=True)
                end.record()
            self._pairs[region].setdefault(str(device), []).append((start, end))
            self.pair_count += 1

    def wrap(self, region, resolve_devices, func):
        def wrapped(*args, **kwargs):
            devices = resolve_devices(*args, **kwargs)
            events = self._start(region, devices)
            host0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                self._stop(region, events)
                self.calls[region] += 1
                self.host_wall[region] += host
                if not self.cuda:
                    for device in devices:
                        self._cpu_spans[region].setdefault(
                            str(device), []).append(host * 1e3)
        return wrapped

    def finish(self, devices):
        """Per-device synchronize, THEN read the spans (never inside the loop)."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        record = {}
        for region in REGIONS:
            per_device = {}
            if self.cuda:
                for name, pairs in self._pairs[region].items():
                    per_device[name] = sum(s.elapsed_time(e) for s, e in pairs)
            else:
                for name, spans in self._cpu_spans[region].items():
                    per_device[name] = float(sum(spans))
            record[region] = dict(
                calls=self.calls[region],
                host_wall_s=self.host_wall[region],
                device_span_ms=per_device,
                device_span_max_ms=(max(per_device.values())
                                    if per_device else 0.0),
                device_span_sum_ms=float(sum(per_device.values())),
                devices=self.devices_seen[region])
        return dict(regions=record, event_backend=self.backend,
                    event_pairs=self.pair_count, event_cap_hit=self.cap_hit)


def attach_region_instrument(model, torch_module, cuda):
    """Wrap the five regions on THIS model instance (mg1's attach_instrument).
    Nothing in the mbirtorch package is edited: the funnels are shadowed as
    instance attributes and the sharding seams as module attributes, all of
    which the engine looks up at call time.  This is observation, not a patch:
    every wrapper calls the original and returns its value unchanged."""
    from mbirtorch import _sharding

    instrument = RegionInstrument(torch_module, cuda)

    model.sparse_forward_project = instrument.wrap(
        "forward_funnel",
        lambda *a, **k: list(model.sino_placement.devices),
        model.sparse_forward_project)
    model.sparse_back_project = instrument.wrap(
        "back_funnel",
        lambda *a, **k: list(model.recon_placement.devices),
        model.sparse_back_project)

    original_halos = _sharding.exchange_qggmrf_halos
    _sharding.exchange_qggmrf_halos = instrument.wrap(
        "halo",
        lambda shards, *a, **k: list(shards.placement.devices),
        original_halos)

    original_reduce = _sharding.sum_band_to_owner
    _sharding.sum_band_to_owner = instrument.wrap(
        "band_reduce",
        lambda partials, owner, *a, **k: [owner],
        original_reduce)

    original_run = _sharding.run_per_device
    wrapped_prior = instrument.wrap(
        "prior",
        lambda devices, *a, **k: list(devices),
        original_run)

    def run_per_device(devices, worker_fn, executor=None):
        if getattr(worker_fn, "__name__", "") == "prior_worker":
            return wrapped_prior(devices, worker_fn, executor=executor)
        return original_run(devices, worker_fn, executor=executor)

    _sharding.run_per_device = run_per_device

    def detach():
        _sharding.exchange_qggmrf_halos = original_halos
        _sharding.sum_band_to_owner = original_reduce
        _sharding.run_per_device = original_run

    return instrument, detach


def _concrete_device(torch_module, device, cuda):
    """A device carrying a concrete index, for recording events on and for
    comparing against a tensor's device."""
    if not cuda or getattr(device, "type", None) != "cuda":
        return device
    if device.index is None:
        return torch_module.device("cuda", torch_module.cuda.current_device())
    return device


# ── INSTRUMENT 1: busy time, per device, per body call (mg9's / mg10's) ───────
class BusyProbe:
    """Times each individual forward projection BODY call, in buckets keyed by
    DEVICE INDEX -- never by object identity, because every entry of
    ``_fwd_body_per_dev`` is the same object.

    Busy divided by the call count is the PER-LAUNCH time.

    THE SHAPE HISTOGRAMS ARE MG11'S LOAD-BEARING WITNESS.  A forward body's
    first positional argument is ``band_values``, of shape (pixels, columns)
    (Projectors.sparse_forward_project_view_range).  On the BANDED path that is
    (all the pass's pixels) x (one owner's slice band); on the COLUMN-GATHER
    path it is (the pixel batch) x (the WHOLE device-form slice axis).  So a
    single column value equal to the full slice count is what separates a column
    gather from a band by another name -- recorded here, by a wrapper on the
    body, entirely independently of the gather counter in the transfer probe.

    Threading.  Body calls arrive on the per-device worker threads of
    ``run_per_device``, which runs at most one thread per device index at a time
    and waits for all of them before the next fan-out, so each bucket has a
    single writer and needs no lock.  That holds for the column gather too: it
    fans out once per view-owner and each worker touches only its own index.
    """

    def __init__(self, torch_module, cuda, n_dev, pairs_per_device):
        self.torch = torch_module
        self.cuda = cuda
        self.n_dev = n_dev
        self.pairs_per_device = pairs_per_device
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"
        self._pairs = [[] for _ in range(n_dev)]
        self._cpu_ms = [0.0] * n_dev
        self.calls = [0] * n_dev
        self.host_s = [0.0] * n_dev
        self.cap_hit = [False] * n_dev
        # Positive witness for the positional key: a forward body's first tensor
        # argument is the values block, which lives on the device that position
        # is supposed to name.
        self.device_mismatch = [0] * n_dev
        self.cols_hist = [{} for _ in range(n_dev)]
        self.pixels_hist = [{} for _ in range(n_dev)]

    def wrap(self, body, dev_index, device):
        """Return ``body`` bracketed, carrying its device index EXPLICITLY.

        ``functools.wraps`` copies the wrapped function's ``__dict__``, which is
        where a kernel body keeps ``_view_batch_cost`` and
        ``_mbirtorch_no_compile``; the caller asserts both survived."""
        torch_module, cuda = self.torch, self.cuda
        device = _concrete_device(torch_module, device, cuda)

        @functools.wraps(body)
        def wrapped(*args, **kwargs):
            if args and torch_module.is_tensor(args[0]):
                if args[0].device != device:
                    self.device_mismatch[dev_index] += 1
                cols = int(args[0].shape[-1])
                bucket = self.cols_hist[dev_index]
                bucket[cols] = bucket.get(cols, 0) + 1
                pixels = int(args[0].shape[0])
                bucket = self.pixels_hist[dev_index]
                bucket[pixels] = bucket.get(pixels, 0) + 1
            if not cuda:
                host0 = time.perf_counter()
                try:
                    return body(*args, **kwargs)
                finally:
                    host = time.perf_counter() - host0
                    self.calls[dev_index] += 1
                    self.host_s[dev_index] += host
                    self._cpu_ms[dev_index] += host * 1e3
            budget = len(self._pairs[dev_index]) < self.pairs_per_device
            start = None
            if budget:
                with torch_module.cuda.device(device):
                    start = torch_module.cuda.Event(enable_timing=True)
                    start.record()
            else:
                self.cap_hit[dev_index] = True
            host0 = time.perf_counter()
            try:
                return body(*args, **kwargs)
            finally:
                host = time.perf_counter() - host0
                if start is not None:
                    with torch_module.cuda.device(device):
                        end = torch_module.cuda.Event(enable_timing=True)
                        end.record()
                    self._pairs[dev_index].append((start, end))
                self.calls[dev_index] += 1
                self.host_s[dev_index] += host

        wrapped._mg11_dev_index = dev_index
        wrapped._mg11_wrapped_body = body
        return wrapped

    def drain(self, devices):
        """Read the elapsed times and reset, ONCE PER TIMED RECONSTRUCTION."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        busy_ms, calls, host_s, mismatch = [], [], [], []
        cols, pixels = [], []
        for i in range(self.n_dev):
            if self.cuda:
                busy_ms.append(float(sum(s.elapsed_time(e)
                                         for s, e in self._pairs[i])))
            else:
                busy_ms.append(float(self._cpu_ms[i]))
            calls.append(int(self.calls[i]))
            host_s.append(float(self.host_s[i]))
            mismatch.append(int(self.device_mismatch[i]))
            cols.append({str(k): v for k, v in sorted(self.cols_hist[i].items())})
            pixels.append({str(k): v for k, v
                           in sorted(self.pixels_hist[i].items())})
        record = dict(busy_ms_per_device=busy_ms,
                      busy_calls_per_device=calls,
                      busy_host_s_per_device=host_s,
                      busy_device_mismatch_per_device=mismatch,
                      busy_value_cols_per_device=cols,
                      busy_value_pixels_per_device=pixels,
                      busy_cap_hit=any(self.cap_hit))
        self._pairs = [[] for _ in range(self.n_dev)]
        self._cpu_ms = [0.0] * self.n_dev
        self.calls = [0] * self.n_dev
        self.host_s = [0.0] * self.n_dev
        self.device_mismatch = [0] * self.n_dev
        self.cols_hist = [{} for _ in range(self.n_dev)]
        self.pixels_hist = [{} for _ in range(self.n_dev)]
        self.cap_hit = [False] * self.n_dev
        return record


# ── INSTRUMENT 2: the transfer layer, and MG11'S PATH WITNESS ────────────────
class TransferProbe:
    """Times the cross-device copies the forward makes, in both shapes, and --
    the part mg11 turns on -- COUNTS WHICH SHAPE MADE THEM.

    The two shapes reach the wire through two different library primitives, and
    each is called from exactly one place in the shipped tree:
        _sharding.broadcast_band_to_views  <- the banded forward driver,
            TomographyModel._sparse_forward_project_sharded
        _sharding.gather_column_band       <- the column gather,
            TomographyModel._sparse_forward_project_columns
    Counting them therefore says which forward ran, at the transfer layer,
    without asking either driver anything.  This is the same witness the
    library's own test uses: test_sharding.py::
    test_the_banded_walk_is_what_runs_with_the_switch_off replaces
    gather_column_band with a function that raises, and asserts the banded walk
    still broadcasts.

    Both primitives are WRAPPED, never reimplemented: the wrapper calls the
    original and returns its result, and records the shape of what came back.
    For the gather that shape is the load-bearing number -- the assembled
    cylinder's HEIGHT must be the whole device-form slice axis at every call,
    and its WIDTH is the pixel batch.

    The copies are timed with event pairs recorded on the SOURCE device's
    current stream, which is the stream torch enqueues a cross-device copy on:
    ``move_shard``'s direct path is ``x.to(target)``, and torch runs that on the
    source's stream and makes the destination's stream wait on a barrier.
    Events on the destination's stream would have measured the barrier wait and
    read as ~0, which is exactly what a wrong-stream bracket looks like, so the
    summary flags a ~0 reading rather than reporting it as a fast copy.

    THE LOCK.  The gather issues its copies from the per-view-owner worker
    threads, so the shared counters take one.  It is uncontended on the banded
    path.
    """

    def __init__(self, torch_module, cuda, max_pairs):
        self.torch = torch_module
        self.cuda = cuda
        self.max_pairs = max_pairs
        self.lock = threading.Lock()
        # the banded fan-out
        self.broadcast_calls = 0
        self.broadcast_host_s = 0.0
        self.band_cols = {}           # band width -> how many fan-outs
        # the column gather
        self.gather_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}          # cylinder height -> how many gathers
        self.cyl_width = {}           # cylinder width  -> how many gathers
        # both
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        self._pairs = []              # (src name, dst name, start, end)
        self.copy_measurement = None

    def drain(self, devices):
        """Read the copy spans and reset, once per timed reconstruction."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        by_src, by_dst, total = {}, {}, 0.0
        for src, dst, start, end in self._pairs:
            span = float(start.elapsed_time(end))
            by_src[src] = by_src.get(src, 0.0) + span
            by_dst[dst] = by_dst.get(dst, 0.0) + span
            total += span
        record = dict(broadcast_calls=int(self.broadcast_calls),
                      broadcast_host_wall_s=float(self.broadcast_host_s),
                      band_cols_hist={str(k): v for k, v
                                      in sorted(self.band_cols.items())},
                      gather_calls=int(self.gather_calls),
                      gather_host_wall_s=float(self.gather_host_s),
                      cyl_height_hist={str(k): v for k, v
                                       in sorted(self.cyl_height.items())},
                      cyl_width_hist={str(k): v for k, v
                                      in sorted(self.cyl_width.items())},
                      copy_count=int(self.copy_count),
                      copy_noop_count=int(self.copy_noop_count),
                      copy_bytes=int(self.copy_bytes),
                      copy_device_ms_total=total,
                      copy_device_ms_by_src=by_src,
                      copy_device_ms_by_dst=by_dst,
                      copy_cap_hit=bool(self.cap_hit),
                      copy_measurement=self.copy_measurement)
        self.broadcast_calls = 0
        self.broadcast_host_s = 0.0
        self.band_cols = {}
        self.gather_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}
        self.cyl_width = {}
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        self._pairs = []
        return record


def attach_forward_probes(model, torch_module, cuda, max_pairs,
                          gather_expected=False):
    """Install the busy and transfer instruments; return
    ``(busy, transfer, verify, detach, observed)``.

    Call this AFTER the discarded cold pass.  The body wrappers live inside the
    projector object, and a device-count settle during the first reconstruction
    rebuilds that object (``_install_device_layout`` -> ``create_projectors``),
    which would throw them away.  ``verify()`` re-checks, before and after every
    timed reconstruction, that the projector object and the wrapped list are
    still the ones the driver will call."""
    from mbirtorch import _sharding

    pf = model.projector_functions
    devices = list(model.sino_placement.devices)
    n_dev = len(devices)
    bodies = pf._fwd_body_per_dev
    if len(bodies) != n_dev:
        raise RuntimeError(
            f"the projector holds {len(bodies)} forward bodies for {n_dev} "
            f"devices; the positional device key would be wrong")

    busy = BusyProbe(torch_module, cuda, n_dev,
                     max(1, max_pairs // max(1, 2 * n_dev)))
    originals = list(bodies)
    wrappers = []
    for index, body in enumerate(originals):
        wrapper = busy.wrap(body, index, devices[index])
        # The driver chooses the view batch by reading this attribute OFF THE
        # BODY, so a wrapper that lost it would silently move the kernel onto
        # the torch batching rule.
        if getattr(body, "_view_batch_cost", None) is not \
                getattr(wrapper, "_view_batch_cost", None):
            raise RuntimeError(
                "the body wrapper did not carry _view_batch_cost through; the "
                "realized view batch would change and this arm would not be "
                "comparable with mg9's or mg10's")
        wrappers.append(wrapper)
    # Mutated IN PLACE rather than rebound, so any other reference to the list
    # sees the wrappers too.
    for index, wrapper in enumerate(wrappers):
        bodies[index] = wrapper

    # The realized view batch, observed PER DEVICE by the positional key.  The
    # column gather hands the driver far fewer pixels per call, which RAISES the
    # batch the transient budget allows; that is a real property of the shipped
    # shape and belongs on the row.
    observed = {}
    observed_lock = threading.Lock()
    original_effective = pf._effective_view_batch

    def effective_view_batch(body, num_pixels, band_cols, args):
        value = original_effective(body, num_pixels, band_cols, args)
        index = getattr(body, "_mg11_dev_index", None)
        key = (f"fwd_dev{index}" if index is not None
               else "back_body_device_not_recoverable")
        with observed_lock:
            bucket = observed.setdefault(key, {})
            bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = effective_view_batch

    # -- the copies, and the path witness -------------------------------------
    transfer = TransferProbe(torch_module, cuda, max_pairs)
    if not cuda:
        transfer.copy_measurement = (
            "host_wall_only: the CPU smoke has no CUDA events, and its virtual "
            "devices make every copy a no-op")
    elif getattr(model, "dev2dev_safe", True):
        transfer.copy_measurement = (
            "device_events_on_source_stream: move_shard's direct path is "
            "x.to(dst), and torch runs a cross-device copy on the SOURCE "
            "device's current stream (the destination's stream only waits on a "
            "barrier), so each copy is bracketed on the source device")
        if gather_expected:
            # The banded fan-out issues its copies from one thread, so its
            # per-copy brackets never overlap.  The gather issues them from
            # every view-owner's worker thread, and two workers pulling from the
            # SAME slice-owner enqueue onto that one source stream at the same
            # time, so their brackets can cover each other's copies.  The
            # per-copy device total is therefore an upper bound on a flag-on arm
            # rather than a sum of disjoint intervals.  The byte total and the
            # gather host wall carry no such caveat, and the fix -- holding the
            # lock across the copy -- is refused because it would serialize the
            # very issue pattern this job exists to measure.
            transfer.copy_measurement += (
                "; UPPER BOUND on this arm: several view-owners copy from one "
                "slice-owner concurrently, so per-copy brackets on that "
                "source's stream can overlap.  Read copy_bytes and the gather "
                "host wall as exact and copy_device_s as a bound")
    else:
        transfer.copy_measurement = (
            "host_wall_only: dev2dev_safe is False, so move_shard routes "
            "through host memory and a source-stream bracket would cover only "
            "the device-to-host half of each copy; the device numbers below "
            "are partial and must not be read as copy time")

    state = threading.local()
    original_move = _sharding.move_shard
    original_broadcast = _sharding.broadcast_band_to_views
    original_gather = _sharding.gather_column_band

    def move_shard(x, target, dev2dev_safe=True):
        if not getattr(state, "timing", False):
            return original_move(x, target, dev2dev_safe=dev2dev_safe)
        with transfer.lock:
            timed = cuda and len(transfer._pairs) < transfer.max_pairs
            if cuda and not timed:
                transfer.cap_hit = True
        start = end = None
        source = x.device
        if timed:
            with torch_module.cuda.device(source):
                start = torch_module.cuda.Event(enable_timing=True)
                start.record()
        out = original_move(x, target, dev2dev_safe=dev2dev_safe)
        if timed:
            with torch_module.cuda.device(source):
                end = torch_module.cuda.Event(enable_timing=True)
                end.record()
        # A copy to the tensor's own device returns the tensor itself, so one
        # move per fan-out (and one per column batch) is free.  Identity is the
        # exact test.  A free copy is counted but kept OUT of the spans and the
        # bytes, so the copy columns mean what actually landed on a device.
        noop = out is x
        with transfer.lock:
            transfer.copy_count += 1
            if noop:
                transfer.copy_noop_count += 1
            else:
                transfer.copy_bytes += int(x.numel()) * int(x.element_size())
                if start is not None:
                    transfer._pairs.append((str(source), str(target), start, end))
        return out

    def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
        state.timing = True
        host0 = time.perf_counter()
        try:
            return original_broadcast(band, view_owners,
                                      dev2dev_safe=dev2dev_safe)
        finally:
            state.timing = False
            cols = int(band.shape[-1])
            with transfer.lock:
                transfer.broadcast_host_s += time.perf_counter() - host0
                transfer.broadcast_calls += 1
                transfer.band_cols[cols] = transfer.band_cols.get(cols, 0) + 1

    def gather_column_band(shard_tensors, p0, p1, target, dev2dev_safe=True):
        """The column gather's own primitive, wrapped exactly as the fan-out is.

        The RETURNED cylinder is measured rather than the arguments, because the
        claim being witnessed is about what was assembled: height is the whole
        device-form slice axis (shape[-1]) and width is the pixel batch
        (shape[0]).  Reading p1 - p0 instead would only repeat the caller's
        arithmetic back at it."""
        state.timing = True
        host0 = time.perf_counter()
        cyl = None
        try:
            cyl = original_gather(shard_tensors, p0, p1, target,
                                  dev2dev_safe=dev2dev_safe)
            return cyl
        finally:
            state.timing = False
            with transfer.lock:
                transfer.gather_host_s += time.perf_counter() - host0
                transfer.gather_calls += 1
                if cyl is not None:
                    height = int(cyl.shape[-1])
                    width = int(cyl.shape[0])
                    transfer.cyl_height[height] = \
                        transfer.cyl_height.get(height, 0) + 1
                    transfer.cyl_width[width] = \
                        transfer.cyl_width.get(width, 0) + 1

    _sharding.move_shard = move_shard
    _sharding.broadcast_band_to_views = broadcast_band_to_views
    _sharding.gather_column_band = gather_column_band

    def verify():
        """Is the instrument still on the path the driver takes?"""
        live = model.projector_functions
        return dict(
            projector_object_same=(live is pf),
            body_list_same=(live._fwd_body_per_dev is bodies),
            wrappers_in_place=all(
                bodies[i] is wrappers[i] for i in range(len(wrappers))),
            broadcast_wrapped=(
                _sharding.broadcast_band_to_views is broadcast_band_to_views),
            gather_wrapped=(_sharding.gather_column_band is gather_column_band),
            move_shard_wrapped=(_sharding.move_shard is move_shard))

    def detach():
        for index, body in enumerate(originals):
            bodies[index] = body
        pf._effective_view_batch = original_effective
        _sharding.move_shard = original_move
        _sharding.broadcast_band_to_views = original_broadcast
        _sharding.gather_column_band = original_gather

    return busy, transfer, verify, detach, observed


# ── the model, and the ONLY two things an arm does to it ─────────────────────
def _build_torch_model(geometry, cell, pin_devices=None):
    """The model.  ``pin_devices`` is a device LIST for the smoke's CPU paths
    only; on CUDA nothing is configured here, because an arm pins through
    MBIRTORCH_NUM_DEVICES and an explicit configure_devices call would take the
    explicit branch, skip the preflight, and leave last_memory_ledger unset."""
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


def configure_arm(model, pixel_batch):
    """THE WHOLE OF WHAT AN ARM DOES TO THE LIBRARY.

    One documented attribute is set on the model, and only on a flag-on arm.
    The switch itself is not touched here at all: it is the environment variable
    the runner exported into this process, which the library reads per call.
    A control gets neither, which is what makes it the shipped configuration.

    Returns the record of what was set, for the row."""
    if pixel_batch is None:
        return dict(pixel_batch_attribute=None,
                    knobs="none: the shipped configuration")
    model.forward_project_pixel_batch = int(pixel_batch)
    return dict(pixel_batch_attribute=int(pixel_batch),
                knobs=("MBIRTORCH_FORWARD_COLUMN_GATHER=1 in the child env "
                       "(exported by the runner) + "
                       "model.forward_project_pixel_batch = "
                       f"{int(pixel_batch)}"))


def read_switch(model):
    """The library's own answer to 'which forward runs', plus the inputs it
    read to get there.  Nothing here re-derives the rule -- the resolver is
    called."""
    from mbirtorch.tomography_model import (COLUMN_GATHER_ENV_VAR as LIB_VAR,
                                            FORWARD_PIXEL_BATCH)
    return dict(
        env_var_name_in_library=LIB_VAR,
        env_var_value=os.environ.get(LIB_VAR),
        resolver_says_gather=bool(model._column_gather_forward()),
        resolver_pixel_batch=int(model._forward_pixel_batch()),
        shipped_default_batch=int(FORWARD_PIXEL_BATCH),
        column_gather_geometry=bool(getattr(model, "column_gather_geometry",
                                            False)),
        rows_track_slices=bool(getattr(model, "rows_track_slices", False)),
        model_attribute=getattr(model, "forward_column_gather", None))


def ledger_reading(model, weights):
    """THE MODELED PEAK, from the library's own planning entry point, plus the
    ledger PLAN, which is one of this job's four witnesses.

    THE ENTRY POINT.  ``TomographyModel.vcd_recon`` calls
    ``self._apply_device_policy(partition_sequence=..., weights=...,
    init_recon=..., fm_hessian=..., prox_input=..., init_error_sinogram=...)``,
    which forwards those call arrays to ``self._build_memory_ledger(
    devices=devices, **call_arrays)``; that is
    ``_memory_ledger.estimate_peak_device_bytes(_memory_ledger.plan_from_model(
    model, devices, **call_arrays))``.  The ledger it returns is stored on the
    model as ``last_memory_ledger``.

    WHICH NUMBER THE GATE READS, and why it is the library's own.  This
    function builds a second ledger from the same entry point, and the two can
    differ in exactly one input: the PARTITION SEQUENCE.  ``recon`` generates
    the sequence the run will visit (``initialize_recon``, truncated to
    ``max_iterations``) and hands THAT to the policy, while a call from outside
    passes None and ``plan_from_model`` falls back to the model's full
    parameter sequence.  A longer sequence visits more granularities, and the
    ledger's peak is a maximum over phases, so the outside call can only ever
    price the same or MORE.  The gate therefore reads ``last_memory_ledger``
    when it exists -- the ledger the run itself was decided with, which is the
    number the floor is a statement about -- and falls back to this function's
    own build when the library made none (an explicit device layout takes the
    branch that skips the preflight, which is how the CPU smoke pins).  Both
    numbers are recorded either way.

    THE PLAN IS A WITNESS.  ``plan.column_pixel_batch`` is the batch on the
    gather path and None on the banded path, and ``plan_from_model`` resolves
    it by calling the model's own ``_column_gather_forward()`` and
    ``_forward_pixel_batch()`` rather than re-deriving the rule.  So it is an
    independent second reading of the switch, written by a different module
    than the driver -- and it does not depend on the partition sequence at all,
    which is why the witness is sound whichever ledger the gate reads.
    """
    from mbirtorch import _memory_ledger

    devices = list(model.sino_placement.devices)
    call_arrays = dict(partition_sequence=None, weights=weights,
                       init_recon=None, fm_hessian=None, prox_input=None,
                       init_error_sinogram=None)
    plan = _memory_ledger.plan_from_model(model, devices, **call_arrays)
    harness = model._build_memory_ledger(devices=devices, **call_arrays)
    harness_peaks = [int(b) for b in harness.per_device_peaks()]
    live = getattr(model, "last_memory_ledger", None)
    library_peaks = ([int(b) for b in live.per_device_peaks()]
                     if live is not None else None)
    chosen = live if live is not None else harness
    modeled = library_peaks if library_peaks is not None else harness_peaks
    return dict(
        entry_point=("TomographyModel._build_memory_ledger(devices=..., "
                     "weights=weights, everything else None) -> "
                     "_memory_ledger.estimate_peak_device_bytes("
                     "_memory_ledger.plan_from_model(...)); the gate reads the "
                     "library's own last_memory_ledger from the same entry "
                     "point when the run built one"),
        modeled_source=("library last_memory_ledger (the ledger this run was "
                        "decided with)" if library_peaks is not None else
                        "harness _build_memory_ledger (the library built none "
                        "on this branch)"),
        modeled_peak_per_device=modeled,
        harness_modeled_peak_per_device=harness_peaks,
        library_modeled_peak_per_device=library_peaks,
        modeled_agrees=(None if library_peaks is None
                        else library_peaks == harness_peaks),
        # A library number ABOVE the harness's would contradict the partition
        # sequence argument above and is worth seeing.
        library_above_harness=(
            None if library_peaks is None
            else bool(any(a > b for a, b in zip(library_peaks, harness_peaks)))),
        dominant_phase_per_device=[chosen.dominant_phase(i).name
                                   for i in range(len(devices))],
        plan_column_pixel_batch=plan.column_pixel_batch,
        plan_forward_band=plan.forward_band,
        plan_back_band=plan.back_band,
        plan_num_pixels_full=int(plan.num_pixels_full),
        plan_rows_track_slices=bool(plan.rows_track_slices),
        plan_slice_blocks=[list(map(int, b)) for b in plan.slice_blocks],
        phase_table=chosen.format_table())


def _launch_key_counts(geometry):
    """Per-kernel launch-key counts (the positive witness that the kernels ran):
    all four kernels share one key set and every key leads with its kernel's
    name."""
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS

    names = (("pback", "pfwd") if geometry == "parallel" else ("back", "fwd"))
    back = sum(1 for k in _COMPILED_LAUNCH_KEYS
               if isinstance(k, tuple) and k and k[0] == names[0])
    fwd = sum(1 for k in _COMPILED_LAUNCH_KEYS
              if isinstance(k, tuple) and k and k[0] == names[1])
    return back, fwd


def _view_batch_static(model, expect_kernels, arm_fwd_cols):
    """The realized view batch per direction per device, against the formula of
    the body EXPECTED to be bound (mg1's static probe).  Run BEFORE the probes
    are installed, so the probe's own observer cannot see this traffic.

    ``arm_fwd_cols`` is this arm's own forward block shape -- the pixel batch
    and the full slice count on the gather path, the shard band otherwise."""
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
                    expected = None
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
    record["view_batch_cols"] = cols
    arm_pixels = int(arm_fwd_cols.get("pixels") or num_pixels)
    arm_cols = int(arm_fwd_cols.get("cols") or cols["fwd"])
    record["arm_fwd_pixels"] = arm_pixels
    record["arm_fwd_cols"] = arm_cols
    record["arm_fwd_view_batch"] = int(
        pf._effective_view_batch(bodies["fwd"][0], arm_pixels, arm_cols, args))
    return record, ok


def _per_device_body_names(model):
    pf = model.projector_functions
    return ([getattr(b, "__name__", str(b)) for b in pf._fwd_body_per_dev],
            [getattr(b, "__name__", str(b)) for b in pf._back_body_per_dev])


def _merge_regions(total, part):
    for region in REGIONS:
        block = part[region]
        slot = total.setdefault(region, dict(calls=0, host_wall_s=0.0,
                                             device_span_ms={}, devices=[]))
        slot["calls"] += block["calls"]
        slot["host_wall_s"] += block["host_wall_s"]
        for name, span in block["device_span_ms"].items():
            slot["device_span_ms"][name] = \
                slot["device_span_ms"].get(name, 0.0) + span
        for name in block["devices"]:
            if name not in slot["devices"]:
                slot["devices"].append(name)
    return total


def _sample_steps(shape):
    """A stride per axis giving roughly VALUE_SAMPLE_TARGET samples along it."""
    return tuple(max(1, int(n) // VALUE_SAMPLE_TARGET) for n in shape)


def _hist_total(hist):
    return sum(int(v) for v in (hist or {}).values())


def _hist_keys(hist):
    return {int(k) for k in (hist or {})}


def _merge_hists(hists):
    out = {}
    for hist in hists or ():
        for key, count in (hist or {}).items():
            out[int(key)] = out.get(int(key), 0) + int(count)
    return out


def _hist_mean(hist):
    """The count-weighted mean of a {value: how many} histogram, with string
    keys as the jsonl carries them.  ``None`` for an empty histogram."""
    total = weight = 0
    for key, count in (hist or {}).items():
        total += int(key) * int(count)
        weight += int(count)
    return (total / weight) if weight else None


# ── one arm ───────────────────────────────────────────────────────────────────
def torch_worker(cfg):
    """One arm: the two knobs, one discarded cold pass, then WARM_REPEATS timed
    reconstructions with every instrument live.

    TWO ORDERINGS, both load-bearing, and they are mg10's for the same reasons.
      The KNOBS go in BEFORE the cold pass, because the pixel batch changes the
    shapes the Triton kernels launch at and the launch key includes the pixel
    count and the column count; set later, every first launch and the compile
    lock it takes would land inside the first timed reconstruction.
      The PROBES go in AFTER the cold pass, because the body wrappers live
    inside the projector object and a device-count settle during the first
    reconstruction rebuilds it.  Both are re-verified around every timed pass.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg.get("n_dev")
    pixel_batch = cfg.get("pixel_batch")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    import importlib
    spec = SPEC[geometry]
    kernel_module = importlib.import_module(spec["kernel_module"])
    shipped_fwd_chunk = int(getattr(kernel_module, spec["fwd_chunk_const"]))
    shipped_back_chunk = int(getattr(kernel_module, spec["back_chunk_const"]))

    pin_devices = None
    if not cuda and smoke_cpu_devices:
        pin_devices = list(smoke_cpu_devices)
    elif not cuda:
        pin_devices = [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=pin_devices)

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  compile_enabled=bool(getattr(model, "compile_enabled", False)),
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  shipped_fwd_chunk=shipped_fwd_chunk,
                  shipped_back_chunk=shipped_back_chunk)

    # ── the env arm checks ───────────────────────────────────────────────────
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))
    if cuda:
        result["pin_env_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") == str(n_dev))

    # THE SWITCH, read from the library's own module rather than spelled here.
    # A rename in the library shows up as an ImportError in read_switch, never
    # as an export that quietly does nothing.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR as LIB_VAR
    result["env_var_name_ok"] = (LIB_VAR == COLUMN_GATHER_ENV_VAR)
    if not result["env_var_name_ok"]:
        raise RuntimeError(
            f"the library's switch is named {LIB_VAR!r} and this harness "
            f"exports {COLUMN_GATHER_ENV_VAR!r}.  Every flag-on arm would run "
            f"the control under a treatment's name.")
    want_env = "1" if pixel_batch is not None else "0"
    result["env_column_gather"] = os.environ.get(LIB_VAR)
    result["env_column_gather_ok"] = (result["env_column_gather"] == want_env)
    if not result["env_column_gather_ok"]:
        raise RuntimeError(
            f"this arm needs {LIB_VAR}={want_env!r} in its child environment "
            f"and got {result['env_column_gather']!r}.  A flag-on arm without "
            f"the export measures the control, and a control WITH it measures "
            f"the treatment; either way the row would be mislabeled.")

    expect_kernels = (cuda, cuda)
    result["expected_bodies_kernel"] = list(expect_kernels)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    num_slices = int(recon_shape[2])
    result["num_slices"] = num_slices
    result["num_slices_planned_ok"] = (num_slices == num_slices_for(cell))

    # ── the arm's shape arithmetic, derived from the arm's own parameters ────
    n_owners = n_dev if cuda else len(pin_devices or [DEVICE])
    slices_per_dev = num_slices // max(1, n_owners)
    result["slices_per_dev"] = slices_per_dev
    result["pixel_batch"] = pixel_batch
    result["cylinder_bytes"] = (cylinder_bytes(pixel_batch, num_slices)
                                if pixel_batch else None)
    result["cylinder_under_bound"] = (
        None if not pixel_batch
        else bool(result["cylinder_bytes"] <= TRANSIENT_BOUND_BYTES))
    result["arm_kind"] = ("anchor" if n_owners == 1 else
                          ("gather" if pixel_batch else "control"))

    # ── THE KNOBS.  This is the whole of what mg11 does to the library ───────
    result["knob_record"] = configure_arm(model, pixel_batch)
    result["switch_at_install"] = read_switch(model)
    want_gather = pixel_batch is not None
    if bool(result["switch_at_install"]["resolver_says_gather"]) != want_gather:
        raise RuntimeError(
            f"the library's own resolver, model._column_gather_forward(), says "
            f"gather={result['switch_at_install']['resolver_says_gather']} "
            f"where this arm is {'flag-on' if want_gather else 'flag-off'}.  "
            f"switch={result['switch_at_install']}.  For a parallel flag-on arm "
            f"this is what a tree WITHOUT the parallel extension looks like: "
            f"ParallelBeamModel.column_gather_geometry must be True and the "
            f"row-aligned refusal in the resolver must be lifted for the gather "
            f"path.  For any arm it means the row would be measuring the other "
            f"shape under this arm's name.")
    if want_gather and (result["switch_at_install"]["resolver_pixel_batch"]
                        != int(pixel_batch)):
        raise RuntimeError(
            f"the library resolved a pixel batch of "
            f"{result['switch_at_install']['resolver_pixel_batch']} where this "
            f"arm set {pixel_batch}; the override attribute is not the one "
            f"TomographyModel._forward_pixel_batch reads.")

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

    # The region instrument goes in BEFORE the cold pass and is drained after
    # every timed pass.
    instrument, detach_regions = attach_region_instrument(model, torch, cuda)

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

    if cuda:
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", index))
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the checks that need the SETTLED layout ──────────────────────────────
    result["switch_after_cold"] = read_switch(model)
    if bool(result["switch_after_cold"]["resolver_says_gather"]) != want_gather:
        raise RuntimeError(
            "the library's resolver changed its answer across the cold pass: "
            f"{result['switch_after_cold']}")
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    fwd_per_dev, back_per_dev = _per_device_body_names(model)
    n_realized = model.sino_placement.n_devices
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_body_per_device=fwd_per_dev,
                  back_body_per_device=back_per_dev)
    want_fwd_kernel, want_back_kernel = expect_kernels
    result["bodies_per_device_ok"] = (
        len(fwd_per_dev) == n_realized and len(back_per_dev) == n_realized
        and all(("triton" in name) == want_fwd_kernel for name in fwd_per_dev)
        and all(("triton" in name) == want_back_kernel for name in back_per_dev))
    result["bodies_ok"] = (
        result["bodies_per_device_ok"]
        and ("triton" in fwd_name) == want_fwd_kernel
        and ("triton" in back_name) == want_back_kernel)
    pf = model.projector_functions
    result["fwd_bodies_distinct_objects"] = (
        len({id(b) for b in pf._fwd_body_per_dev}) == len(pf._fwd_body_per_dev))

    # A wrong device count invalidates every shape number in this arm.
    realized_devices = [str(d) for d in model.sino_placement.devices]
    expected_count = n_dev if cuda else len(pin_devices or [DEVICE])
    if len(realized_devices) != expected_count:
        raise RuntimeError(
            f"this arm asked for {expected_count} device(s) and settled on "
            f"{len(realized_devices)} ({realized_devices}).  Every shape number "
            f"in the arm is derived from the shard length, so the whole row "
            f"would be mislabeled.")
    if not result["num_slices_planned_ok"]:
        raise RuntimeError(
            f"the plan assumed {num_slices_for(cell)} slices at this cell and "
            f"the model built {num_slices}; the cylinder-height arithmetic in "
            f"the plan print is wrong for this tree.")

    # ── THE MEMORY GATE'S MODELED SIDE, from the library's own entry point ───
    result["ledger"] = ledger_reading(model, weights)
    plan_batch = result["ledger"]["plan_column_pixel_batch"]
    want_plan_batch = int(pixel_batch) if (want_gather and n_owners > 1) else None
    if want_gather and n_owners > 1 and plan_batch != want_plan_batch:
        raise RuntimeError(
            f"the memory ledger's plan priced column_pixel_batch={plan_batch} "
            f"where this arm runs the gather at {want_plan_batch}.  The modeled "
            f"peak would be the BANDED path's, so the floor ratio would be "
            f"meaningless for this arm.")
    if not want_gather and plan_batch is not None:
        raise RuntimeError(
            f"a flag-OFF arm's ledger plan priced a column batch of "
            f"{plan_batch}; the plan is reading a switch this arm did not set.")
    # Which of the two ledgers the gate is reading, hoisted onto the row so the
    # memory table can say so without digging.
    result["library_ledger_peaks"] = \
        result["ledger"]["library_modeled_peak_per_device"]
    result["library_ledger_agrees"] = result["ledger"]["modeled_agrees"]
    result["modeled_source"] = result["ledger"]["modeled_source"]

    arm_cols = dict(
        pixels=(pixel_batch if want_gather else None),
        cols=(num_slices if want_gather else
              (num_slices if n_owners == 1 else slices_per_dev)))
    vb_record, vb_ok = _view_batch_static(model, expect_kernels, arm_cols)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    result["dev2dev_safe"] = bool(getattr(model, "dev2dev_safe", True))
    result["fwd_chunk_after"] = int(getattr(kernel_module,
                                            spec["fwd_chunk_const"]))
    result["back_chunk_after"] = int(getattr(kernel_module,
                                             spec["back_chunk_const"]))
    # mg11 moves no module constant anywhere, so both must be the shipped ones
    # on EVERY arm.
    result["chunks_unchanged_ok"] = (
        result["fwd_chunk_after"] == shipped_fwd_chunk
        and result["back_chunk_after"] == shipped_back_chunk)

    # ── the instruments, installed on the settled projector ──────────────────
    busy, transfer, verify, detach_probes, observed = attach_forward_probes(
        model, torch, cuda, MAX_EVENT_PAIRS, gather_expected=want_gather)
    result["probe_verify_before"] = verify()
    if not all(result["probe_verify_before"].values()):
        raise RuntimeError(
            f"the forward probes are not on the driver's path before the timed "
            f"passes: {result['probe_verify_before']}")

    # ── the timed reconstructions ────────────────────────────────────────────
    instrument.reset()
    if cuda:
        for device in model.sino_placement.devices:
            torch.cuda.reset_peak_memory_stats(device)
    devices = list(model.sino_placement.devices)
    device_names = [str(d) for d in devices]
    warm, per_recon, region_totals = [], [], {}
    pair_total, cap_hit = 0, False
    checksums, sample_paths = [], []
    steps = None
    for repeat in range(WARM_REPEATS):
        start = time.perf_counter()
        out = vcd()
        wall = time.perf_counter() - start
        warm.append(wall)
        # Drained AFTER the wall is recorded, so the read-out's synchronize and
        # elapsed_time calls never land inside a timed number.
        readout = instrument.finish(devices)
        _merge_regions(region_totals, readout["regions"])
        pair_total += readout["event_pairs"]
        cap_hit = cap_hit or readout["event_cap_hit"]
        instrument.reset()
        record = dict(recon_index=repeat, wall_s=wall)
        record.update(busy.drain(devices))
        record.update(transfer.drain(devices))
        forward = readout["regions"]["forward_funnel"]
        record["bracket_ms_per_device"] = [
            float(forward["device_span_ms"].get(name, 0.0))
            for name in device_names]
        record["bracket_host_wall_s"] = float(forward["host_wall_s"])
        record["bracket_calls"] = int(forward["calls"])
        back = readout["regions"]["back_funnel"]
        record["back_bracket_ms_per_device"] = [
            float(back["device_span_ms"].get(name, 0.0))
            for name in device_names]
        record["gap_ms_per_device"] = [
            b - u for b, u in zip(record["bracket_ms_per_device"],
                                  record["busy_ms_per_device"])]
        record["probe_verify"] = verify()
        record["switch"] = read_switch(model)
        record["view_batch_observed"] = {
            key: sorted(bucket.items()) for key, bucket in observed.items()}
        # THE MEASURED SIDE OF THE MEMORY GATE, per timed reconstruction.  No
        # reset between passes, so the series is a running maximum.
        record["peak_bytes_per_device"] = peaks()
        per_recon.append(record)
        health.append(sample_gpu_health())

        # ── the value column ─────────────────────────────────────────────────
        checksums.append(float(np.sum(np.abs(out), dtype=np.float64)))
        if steps is None:
            steps = _sample_steps(out.shape)
        if repeat < 2:
            # Two samples: one for the cross-arm distances and a second so the
            # summary can state this arm's OWN pass-to-pass distance, which is
            # the floor every cross-arm number has to be read against (both
            # forward kernels accumulate with float atomics and are not
            # bit-reproducible).
            path = _sample_path(cfg["arm_id"], repeat)
            np.save(path, np.ascontiguousarray(
                out[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32))
            sample_paths.append(path)

        # ── the witnesses, on the first timed reconstruction ─────────────────
        if repeat == 0:
            _check_witnesses(result, record, device_names, n_owners, num_slices,
                             slices_per_dev, pixel_batch)

    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_min"] = min(warm)
    result["vcd_warm_max"] = max(warm)
    result["vcd_warm_spread"] = ((max(warm) - min(warm))
                                 / statistics.median(warm)) if warm else None
    result["per_recon"] = per_recon
    result["device_names"] = device_names
    result["view_batch_observed_per_device"] = {
        k: sorted(v.items()) for k, v in observed.items()}
    result["busy_backend"] = busy.backend
    result["probe_verify_after"] = verify()
    result["switch_after"] = read_switch(model)
    detach_probes()
    detach_regions()

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    result["realized_devices"] = realized_devices
    result["realized_n_devices"] = len(realized_devices)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = True          # fatal above if it were not

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (
            result["back_launch_keys_delta"] > 0
            and result["fwd_launch_keys_delta"] > 0)

    # ── the region read-out, in mg5's field names so the rows diff ───────────
    regions = {}
    for region in REGIONS:
        slot = region_totals.get(region, dict(calls=0, host_wall_s=0.0,
                                              device_span_ms={}, devices=[]))
        spans = slot["device_span_ms"]
        regions[region] = dict(
            calls=slot["calls"], host_wall_s=slot["host_wall_s"],
            device_span_ms=spans,
            device_span_max_ms=(max(spans.values()) if spans else 0.0),
            device_span_sum_ms=float(sum(spans.values())),
            devices=slot["devices"])
    result["regions"] = regions
    result["event_backend"] = instrument.backend
    result["event_pairs"] = pair_total
    result["event_cap_hit"] = cap_hit

    count = len(realized_devices)
    missing = []
    for region in REGIONS:
        if region in REGIONS_ABSENT_AT_N1 and count == 1:
            continue
        if regions[region]["host_wall_s"] <= 0.0:
            missing.append(f"{region}:host")
            continue
        if region in REGIONS_HOST_ONLY_AT_N1 and count == 1:
            continue
        if regions[region]["device_span_sum_ms"] <= 0.0:
            missing.append(f"{region}:device")
    result["region_nonzero_ok"] = not missing
    result["region_missing"] = missing

    passes = max(1, WARM_REPEATS)
    composed = result["vcd_warm"]
    for region in REGIONS:
        block = regions[region]
        result[f"{region}_wall_per_pass_s"] = block["host_wall_s"] / passes
        result[f"{region}_dev_span_max_per_pass_s"] = (
            block["device_span_max_ms"] / 1e3 / passes)
        result[f"{region}_calls"] = block["calls"]
        result[f"{region}_gap_per_pass_s"] = (
            result[f"{region}_wall_per_pass_s"]
            - result[f"{region}_dev_span_max_per_pass_s"])
    top = sum(regions[r]["host_wall_s"] for r in REGIONS
              if r not in NESTED_REGIONS)
    per_pass = top / passes
    result["region_wall_total_s"] = top
    result["region_wall_per_pass_s"] = per_pass
    result["region_remainder_per_pass_s"] = composed - per_pass
    result["region_remainder_frac"] = ((composed - per_pass) / composed
                                       if composed else None)
    result["reconcile_ok"] = (per_pass <= composed * (1.0 + RECONCILE_SLACK))
    result["forward_share_of_composed"] = (
        result["forward_funnel_wall_per_pass_s"] / composed if composed else None)

    result["recon_checksums"] = checksums
    result["recon_checksum"] = checksums[-1] if checksums else None
    result["recon_checksum_spread"] = (
        (max(checksums) - min(checksums)) / statistics.median(checksums)
        if len(checksums) > 1 and statistics.median(checksums) else None)
    result["value_sample_paths"] = sample_paths
    result["value_sample_steps"] = list(steps or ())
    result["recon_shape_out"] = list(out.shape)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result.update(summarize_arm(result))
    result.update(memory_reading(result))
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def _check_witnesses(result, record, device_names, n_owners, num_slices,
                     slices_per_dev, pixel_batch):
    """THE PROOF THAT THIS ARM MEASURED WHAT ITS NAME SAYS, run on the first
    timed reconstruction and fatal on any disagreement.

    mg11 installs no patch, so it has no patch counter.  Every witness below is
    built out of something the LIBRARY does, observed from outside, and each of
    the four is recorded by different code than the others:

        the RESOLVER         model._column_gather_forward()      (checked at
                             install and after the cold pass, above)
        the LEDGER PLAN      plan.column_pixel_batch             (checked in the
                             worker, above)
        the PRIMITIVES       _sharding.gather_column_band and
                             _sharding.broadcast_band_to_views call counts, plus
                             the assembled cylinder's shape (transfer probe)
        the KERNEL'S VIEW    the shape of every values block a forward body was
                             handed (busy probe)

    The last two are the pair that cannot agree by accident: one is recorded at
    the transfer layer, the other inside the projector's view loop, and they are
    cross-checked against each other rather than each against an expectation.

    Every expected number here comes from the arm's own parameters -- the device
    count, the slice count and the batch it set -- and never from asking the
    library what it did.
    """
    funnel_calls = int(record["bracket_calls"])
    n_dev = len(device_names)
    want_gather = pixel_batch is not None
    checks = {}

    # -- shared: the instrument itself is on the path -------------------------
    missing = [name for name, calls
               in zip(device_names, record["busy_calls_per_device"])
               if calls <= 0]
    if missing:
        raise RuntimeError(
            f"the busy instrument recorded NO forward body calls on {missing} "
            f"in the first timed reconstruction.  The wrappers are not on the "
            f"path the driver takes, and every per-device number in this arm "
            f"would be empty.  verify={record['probe_verify']}")
    if not all(record["probe_verify"].values()):
        raise RuntimeError(
            f"the forward probes left the driver's path during the first timed "
            f"reconstruction: {record['probe_verify']}")
    if funnel_calls <= 0:
        raise RuntimeError(
            "the forward funnel was never called in a timed reconstruction; "
            "the region instrument is not on the model's own attribute.")
    if bool(record["switch"]["resolver_says_gather"]) != want_gather:
        raise RuntimeError(
            f"the library's resolver said gather="
            f"{record['switch']['resolver_says_gather']} DURING the first timed "
            f"reconstruction where this arm is "
            f"{'flag-on' if want_gather else 'flag-off'}: {record['switch']}")

    gathers = int(record["gather_calls"])
    broadcasts = int(record["broadcast_calls"])
    cols_seen = _merge_hists(record["busy_value_cols_per_device"])
    pixels_seen = _merge_hists(record["busy_value_pixels_per_device"])
    heights = {int(k): int(v) for k, v in record["cyl_height_hist"].items()}
    widths = {int(k): int(v) for k, v in record["cyl_width_hist"].items()}
    bands = {int(k): int(v) for k, v in record["band_cols_hist"].items()}
    checks.update(gather_calls=gathers, broadcast_calls=broadcasts,
                  values_block_cols=sorted(cols_seen),
                  cylinder_heights=sorted(heights),
                  band_widths=sorted(bands))

    # -- the ANCHOR: a trivial placement moves nothing at all -----------------
    if n_owners == 1:
        if n_dev != 1:
            raise RuntimeError(
                f"an anchor arm settled on {n_dev} devices ({device_names}); "
                f"the anchor exists precisely because its placement is trivial.")
        for name, count in (("band fan-outs", broadcasts),
                            ("column gathers", gathers),
                            ("cross-device copies", int(record["copy_count"])),
                            ("cross-device bytes", int(record["copy_bytes"]))):
            if count:
                raise RuntimeError(
                    f"a one-device anchor recorded {count} {name}.  At a "
                    f"trivial placement the forward driver returns before it "
                    f"reaches either multi-device shape, so this row is not the "
                    f"single-device measurement it claims to be.")
        if set(cols_seen) != {num_slices}:
            raise RuntimeError(
                f"the one-device anchor handed the kernel values blocks of "
                f"widths {sorted(cols_seen)}; the shipped single-device call is "
                f"one block of the whole {num_slices}-slice volume.")
        checks["anchor_block_cols"] = num_slices
        result["witnesses"] = checks
        return checks

    # -- a CONTROL: the banded walk ran, and the gather did NOT ---------------
    if not want_gather:
        if gathers:
            raise RuntimeError(
                f"a flag-OFF control called _sharding.gather_column_band "
                f"{gathers} times.  That primitive is called only by "
                f"TomographyModel._sparse_forward_project_columns, so this row "
                f"is the treatment under a control's name and every comparison "
                f"read against it would be wrong.")
        if broadcasts <= 0:
            raise RuntimeError(
                "a flag-off control recorded ZERO band fan-outs.  The banded "
                "forward broadcasts once per sub-band per slice-owner, so a "
                "control that broadcast nothing did not run the banded walk.")
        if set(bands) != {slices_per_dev}:
            raise RuntimeError(
                f"the control fanned out bands of widths {sorted(bands)} where "
                f"its shard is {slices_per_dev} slices and the shipped rule "
                f"walks the whole shard as one band.  Either an inherited "
                f"forward_project_slice_band is set, or the shard is not the "
                f"length this arm's device count implies.")
        if set(cols_seen) != {slices_per_dev}:
            raise RuntimeError(
                f"the control handed the kernel values blocks of widths "
                f"{sorted(cols_seen)} where the banded walk hands it one "
                f"{slices_per_dev}-slice shard.  A block as wide as the whole "
                f"{num_slices}-slice volume is what the column gather hands it, "
                f"so this control may be running the gather.")
        if broadcasts % n_dev:
            raise RuntimeError(
                f"{broadcasts} band fan-outs do not divide by the {n_dev} "
                f"slice-owners; the banded walk broadcasts once per owner per "
                f"sub-band.")
        checks["control_band_width"] = slices_per_dev
        result["witnesses"] = checks
        return checks

    # -- a FLAG-ON arm: the gather ran, and the banded walk did NOT -----------
    if broadcasts:
        raise RuntimeError(
            f"a flag-ON arm called _sharding.broadcast_band_to_views "
            f"{broadcasts} times.  That primitive is called only by the BANDED "
            f"forward, so part or all of this reconstruction ran the shape this "
            f"arm exists to replace.")
    if gathers <= 0:
        raise RuntimeError(
            "a flag-ON arm called _sharding.gather_column_band ZERO times.  "
            "The resolver said gather, so the driver reached "
            "_sparse_forward_project_columns and would have gathered; a zero "
            "here means the primitive this harness wrapped is not the one the "
            "driver calls (it resolves the module global at call time).")
    # THE LOAD-BEARING ONE.  Every assembled cylinder spans the WHOLE
    # device-form slice axis -- that single value is what separates a column
    # gather from a band by another name.
    if set(heights) != {num_slices}:
        raise RuntimeError(
            f"the assembled cylinders had heights {sorted(heights)} where the "
            f"whole device-form slice axis is {num_slices}.  A cylinder shorter "
            f"than the volume is a band, not a column gather, and every timing "
            f"number in this row would belong to a shape nobody proposed.")
    # The batch bounds every cylinder's width, and a width below it is a tail:
    # at most one per (forward call, view-owner).
    if max(widths) > int(pixel_batch):
        raise RuntimeError(
            f"a gathered cylinder was {max(widths)} pixel columns wide where "
            f"this arm set a batch of {pixel_batch}; the batch is what bounds "
            f"the transient, so the memory column would be understated.")
    tails = sum(count for width, count in widths.items()
                if width != int(pixel_batch))
    if tails > funnel_calls * n_dev:
        raise RuntimeError(
            f"{tails} gathered cylinders were narrower than the {pixel_batch} "
            f"batch, against at most one tail per forward call per view-owner "
            f"({funnel_calls} x {n_dev}).  The pixel walk is not the "
            f"range(0, num_pixels, batch) the driver runs.")
    if gathers % n_dev:
        raise RuntimeError(
            f"{gathers} column gathers do not divide by the {n_dev} "
            f"view-owners; the driver fans out once per owner and each owner "
            f"walks the same pixel axis, so every owner makes the same number "
            f"of gathers.  ONE EXCEPTION EXISTS and it is not this cell's: an "
            f"owner with no real views returns before gathering (the "
            f"sparse-view extension).  The 1024-view cell divides by 1, 2 and "
            f"4, so every owner here owns real views; at a cell that does not "
            f"divide, this check needs the owner count that actually projects.")
    if gathers < funnel_calls * n_dev:
        raise RuntimeError(
            f"{gathers} column gathers against {funnel_calls} forward calls on "
            f"{n_dev} view-owners; every owner gathers at least once per call.")
    # -- the cross-check: two recorders, two layers, one structure -----------
    # Every gathered cylinder is handed straight to
    # sparse_forward_project_view_range as its band_values, which slices the
    # VIEW axis and hands the same block to the body once per view batch.  So
    # the set of pixel counts the BODY saw must be exactly the set of cylinder
    # widths the TRANSFER layer assembled, and the body's column count must be
    # the cylinder height.  These two are recorded by different wrappers on
    # different functions; a witness that agreed with itself could not fail.
    if set(cols_seen) != {num_slices}:
        raise RuntimeError(
            f"the kernel was handed values blocks of widths {sorted(cols_seen)} "
            f"where every gathered cylinder is {num_slices} slices tall.  The "
            f"transfer layer and the projector's view loop disagree about what "
            f"was projected.")
    if set(pixels_seen) != set(widths):
        raise RuntimeError(
            f"the kernel saw pixel counts {sorted(pixels_seen)} against "
            f"gathered cylinder widths {sorted(widths)}.  Each cylinder is "
            f"handed to exactly one view-range call, so the two sets are the "
            f"same set; a disagreement means part of the forward took another "
            f"path.")
    per_width = {}
    for width, count in widths.items():
        launches = pixels_seen.get(width)
        if launches is None or launches % count:
            raise RuntimeError(
                f"{launches} body launches at pixel count {width} do not divide "
                f"by the {count} cylinders of that width; every cylinder is "
                f"projected in the same number of view batches.")
        per_width[width] = launches // count
    checks.update(cylinder_height=num_slices, cylinder_widths=sorted(widths),
                  tails=tails, launches_per_cylinder=per_width,
                  batch=int(pixel_batch))
    result["witnesses"] = checks
    return checks


# ── the per-arm reductions ────────────────────────────────────────────────────
def summarize_arm(result):
    """Per device: the bracket, the busy sum, the call count, the per-launch
    time, the gap and the normalized cost columns -- each the median over the
    timed reconstructions -- plus the transfer totals.  Medians, because a
    single reconstruction can carry a scheduling artifact.

    ms_per_kpix is the normalized cost column that matters here: a call handed
    fewer pixels always takes less time, which on its own says nothing, so the
    per-launch time is divided by the mean pixel count in thousands of the
    values blocks the kernel was actually handed.  ms_per_slice is kept beside
    it because the parallel comparison moves the column axis instead."""
    names = result["device_names"]
    passes = result["per_recon"]
    if not passes:
        return dict(per_device=[], transfer=None)

    def median(values):
        return float(statistics.median(values)) if values else None

    per_device = []
    for index, name in enumerate(names):
        bracket = [p["bracket_ms_per_device"][index] / 1e3 for p in passes]
        busy = [p["busy_ms_per_device"][index] / 1e3 for p in passes]
        calls = [p["busy_calls_per_device"][index] for p in passes]
        host = [p["busy_host_s_per_device"][index] for p in passes]
        back = [p["back_bracket_ms_per_device"][index] / 1e3 for p in passes]
        copy_in = [p["copy_device_ms_by_dst"].get(name, 0.0) / 1e3
                   for p in passes]
        copy_out = [p["copy_device_ms_by_src"].get(name, 0.0) / 1e3
                    for p in passes]
        peak = [(p.get("peak_bytes_per_device") or [0] * len(names))[index]
                for p in passes]
        bracket_med, busy_med = median(bracket), median(busy)
        calls_med = median(calls)
        per_launch = ((busy_med * 1e3 / calls_med)
                      if busy_med and calls_med else None)
        cols_hist = passes[0]["busy_value_cols_per_device"][index]
        pix_hist = passes[0].get("busy_value_pixels_per_device",
                                 [{}] * len(names))[index]
        mean_cols, mean_pixels = _hist_mean(cols_hist), _hist_mean(pix_hist)
        per_device.append(dict(
            device=name,
            bracket_span_s=bracket_med,
            busy_sum_s=busy_med,
            busy_calls=calls_med,
            per_launch_ms=per_launch,
            mean_cols_per_launch=mean_cols,
            mean_pixels_per_launch=mean_pixels,
            ms_per_slice=(per_launch / mean_cols
                          if per_launch and mean_cols else None),
            ms_per_kpix=(per_launch / (mean_pixels / 1000.0)
                         if per_launch and mean_pixels else None),
            busy_host_s=median(host),
            gap_s=(bracket_med - busy_med),
            busy_frac_of_bracket=(busy_med / bracket_med if bracket_med else None),
            back_bracket_span_s=median(back),
            copy_device_in_s=median(copy_in),
            copy_device_out_s=median(copy_out),
            peak_bytes=median(peak),
            value_cols=cols_hist,
            value_pixels=pix_hist,
            device_mismatch=sum(p["busy_device_mismatch_per_device"][index]
                                for p in passes)))

    bytes_med = median([p["copy_bytes"] for p in passes])
    dev_ms_med = median([p["copy_device_ms_total"] for p in passes])
    transfer = dict(
        broadcast_calls_per_recon=median([p["broadcast_calls"] for p in passes]),
        broadcast_host_s_per_recon=median([p["broadcast_host_wall_s"]
                                           for p in passes]),
        gather_calls_per_recon=median([p["gather_calls"] for p in passes]),
        gather_host_s_per_recon=median([p["gather_host_wall_s"]
                                        for p in passes]),
        copy_count_per_recon=median([p["copy_count"] for p in passes]),
        copy_noop_count_per_recon=median([p["copy_noop_count"] for p in passes]),
        copy_bytes_per_recon=bytes_med,
        copy_device_s_per_recon=(dev_ms_med / 1e3 if dev_ms_med is not None
                                 else None),
        copy_gb_per_s=((bytes_med / 1e9) / (dev_ms_med / 1e3)
                       if bytes_med and dev_ms_med else None),
        copy_measurement=passes[0].get("copy_measurement"),
        dev2dev_safe=result.get("dev2dev_safe"))
    # A device-side copy time of ~0 at two or more devices is what a bracket on
    # the WRONG STREAM looks like, so it is flagged rather than reported as a
    # fast copy.
    transfer["copy_device_plausible"] = (
        None if len(names) < 2 or not result.get("cuda")
        else bool(dev_ms_med and dev_ms_med > 0.0))
    # The headline speed columns, so the verdict block reads one place.
    per_dev_busy = [d["busy_sum_s"] for d in per_device if d["busy_sum_s"]]
    per_dev_bracket = [d["bracket_span_s"] for d in per_device
                       if d["bracket_span_s"]]
    return dict(per_device=per_device, transfer=transfer,
                forward_busy_max_s=(max(per_dev_busy) if per_dev_busy else None),
                forward_bracket_max_s=(max(per_dev_bracket)
                                       if per_dev_bracket else None),
                composed_s=result.get("vcd_warm"))


def memory_reading(result):
    """The memory gate's two sides, and their ratio per device.

    The MEASURED peak is max(cold, warm): torch.cuda.max_memory_allocated is a
    running maximum and both readings are taken, so the gate reads the largest
    allocation the process actually reached while reconstructing.  mg10's rows
    show the two equal to two decimal places at every arm, so this is strict
    without being a compile-artifact trap -- and when they do differ, the row
    carries both ratios and the summary says which one carried the violation.

    The MODELED peak is the ledger's, from the library's own entry point (see
    ledger_reading).  modeled / measured at or above 1.00 is a preflight that
    over-predicts, which is what a preflight must do.  Below 1.00 is a FLOOR
    VIOLATION: the library would have admitted a layout that does not fit."""
    modeled = (result.get("ledger") or {}).get("modeled_peak_per_device") or []
    cold = result.get("gpu_peak_cold_per_device") or []
    warm = result.get("gpu_peak_warm_per_device") or []
    both = result.get("gpu_peak_per_device") or []
    if not modeled or not both or len(modeled) != len(both):
        return dict(memory=dict(
            available=False,
            why=("no CUDA peak counter on this run (the CPU smoke has none), "
                 "so the floor ratio is not computed here"),
            modeled_peak_per_device=modeled))

    def ratios(measured):
        return [(m / x) if x else None for m, x in zip(modeled, measured)]

    ratio = ratios(both)
    live = [r for r in ratio if r is not None]
    cold_ratio = ratios(cold) if cold else []
    warm_ratio = ratios(warm) if warm else []
    worst_index = (min(range(len(ratio)), key=lambda i: (ratio[i] is None,
                                                         ratio[i]))
                   if live else None)
    return dict(memory=dict(
        available=True,
        modeled_peak_per_device=modeled,
        measured_peak_per_device=both,
        measured_cold_per_device=cold,
        measured_warm_per_device=warm,
        ratio_per_device=ratio,
        ratio_cold_per_device=cold_ratio,
        ratio_warm_per_device=warm_ratio,
        min_ratio=(min(live) if live else None),
        min_ratio_device=(result["device_names"][worst_index]
                          if worst_index is not None else None),
        floor_violation=bool(live and min(live) < 1.0),
        violation_only_in_cold=bool(
            live and min(live) < 1.0 and warm_ratio
            and min(r for r in warm_ratio if r is not None) >= 1.0),
        dominant_phase_per_device=(result.get("ledger") or {}).get(
            "dominant_phase_per_device")))


def generator_worker(cfg):
    """Build ONE shared sinogram per geometry: phantom -> sinogram -> .npy, plus
    its md5 sidecar.  Every arm at that geometry reconstructs THAT array, so no
    arm's timing or value carries an input difference.  Pinned to one device so
    the generator cannot itself become a multi-device run, and run with the
    switch forced OFF, because a generator that forward-projected through the
    shape under test would put the treatment inside every arm's INPUT."""
    import numpy as np

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    devices = cfg.get("cpu_devices") or [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=devices)
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
                sino_md5=digest, sinogram_shape=list(sinogram.shape),
                recon_shape=list(recon_shape),
                env_column_gather=os.environ.get(COLUMN_GATHER_ENV_VAR),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


# ── the value read-out ────────────────────────────────────────────────────────
def _rel_distance(path_a, path_b):
    """Two distances between two strided reconstruction samples, and why there
    are two.

    ``rel_l2`` is section 7.2's own metric -- a relative L2 over the strided
    sample -- and it is the one the REGISTERED EXPECTATION (mg10's e-6 class)
    is stated in.
    ``max_rel_of_peak`` is max|a-b| / max|b|, which is the functional form the
    standing parity suites take, so it is the one the FAILING BAR
    (PARITY_FLOOR) is read against.  Reading a 5e-3 bar built from a max-abs
    ratio against an L2 would be comparing two different quantities.
    """
    import numpy as np

    if not path_a or not path_b:
        return None
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
    a = np.load(path_a).astype(np.float64)
    b = np.load(path_b).astype(np.float64)
    if a.shape != b.shape:
        return None
    denom = float(np.linalg.norm(b))
    if denom == 0.0:
        return None
    diff = a - b
    scale = float(np.max(np.abs(b)))
    return dict(rel_l2=float(np.linalg.norm(diff) / denom),
                max_abs=float(np.max(np.abs(diff))),
                max_rel_of_peak=(float(np.max(np.abs(diff)) / scale)
                                 if scale else None))


def _checksum_stats(row):
    """The per-reconstruction checksums of one arm, reduced: the median, and the
    REPEAT-TO-REPEAT spread as a fraction of it.  That spread is the run-to-run
    noise floor every cross-arm checksum distance is read against; both forward
    kernels accumulate with float atomics, so it is not zero on the GPU even for
    two runs of the identical configuration."""
    values = row.get("recon_checksums") or []
    if not values:
        return None, None
    mid = statistics.median(values)
    spread = (max(values) - min(values)) / mid if mid else None
    return mid, spread


def _checksum_distance(row, other):
    """Relative distance between two arms' median checksums."""
    a, _ = _checksum_stats(row)
    b, _ = _checksum_stats(other or {})
    if a is None or b is None or not b:
        return None
    return abs(a - b) / abs(b)


def _first_sample(row):
    return ((row or {}).get("value_sample_paths") or [None])[0]


def value_table(rows):
    """Every distance the flip's value gate is priced on, computed two ways so
    neither has to be trusted alone.

    For each arm: its OWN repeat-to-repeat distance (the floor), its distance to
    the flag-off control at the same device count, and its distance to its
    geometry's one-device anchor.  A cross-arm distance at the level of an arm's
    own floor is the strongest statement this instrument can make."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by_token = {r.get("token"): r for r in live}
    out = []
    for row in live:
        token = row.get("token")
        geometry, n_dev = row.get("geometry"), row.get("n_dev")
        samples = row.get("value_sample_paths") or []
        own = _rel_distance(samples[1], samples[0]) if len(samples) > 1 else None
        ctrl_token = control_token(geometry, n_dev)
        anch_token = anchor_token(geometry)
        control = by_token.get(ctrl_token)
        anchor = by_token.get(anch_token)
        vs_control = (_rel_distance(_first_sample(row), _first_sample(control))
                      if control is not None and control is not row else None)
        vs_anchor = (_rel_distance(_first_sample(row), _first_sample(anchor))
                     if anchor is not None and anchor is not row else None)
        median_checksum, repeat_spread = _checksum_stats(row)
        entry = dict(
            token=token, geometry=geometry, n=n_dev,
            kind=row.get("arm_kind"), batch=row.get("pixel_batch"),
            label=(f"column gather, batch {row.get('pixel_batch')}"
                   if row.get("pixel_batch") else
                   ("the shipped single-device call" if n_dev == 1 else
                    "the shipped banded walk (flag off)")),
            checksums=row.get("recon_checksums"),
            checksum_median=median_checksum,
            checksum_repeat_spread=repeat_spread,
            checksum_vs_control=(_checksum_distance(row, control)
                                 if control is not None and control is not row
                                 else None),
            checksum_vs_anchor=(_checksum_distance(row, anchor)
                                if anchor is not None and anchor is not row
                                else None),
            own_pass_to_pass=own,
            vs_control=vs_control, vs_control_token=ctrl_token,
            vs_anchor=vs_anchor, vs_anchor_token=anch_token)
        # Recorded, never asserted: the parallel gather is order-preserving at
        # the driver level (each detector row keeps a single producing piece --
        # design note section 13), so a parallel flag-on checksum "should" match
        # its control.  The kernel's own atomics are not bit-reproducible, so
        # the honest reading is against this arm's own floor and not against
        # zero, and this field says only whether the exact equality happened.
        if geometry == "parallel" and row.get("pixel_batch"):
            cs = row.get("recon_checksums") or []
            ctrl_cs = (control or {}).get("recon_checksums") or []
            entry["checksum_equals_control_exactly"] = bool(
                cs and ctrl_cs and cs[0] == ctrl_cs[0])
        out.append(entry)
    return out


# ── the runner (mg5's / mg9's / mg10's subprocess pattern) ───────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.

    Two variables carry the whole experiment.
      MBIRTORCH_NUM_DEVICES pins the count.  An arm pins ONLY this way, which
    keeps the model on the automatic branch where the preflight still runs and
    where the library builds its own ledger for the run; an explicit
    configure_devices call would take the explicit branch, get no preflight, and
    leave last_memory_ledger unset -- so the two are not interchangeable.
      MBIRTORCH_FORWARD_COLUMN_GATHER is the switch, set to '1' on a flag-on arm
    and to '0' -- explicitly, never merely unset -- on every control and anchor.
    Unset would leave the model attribute to decide, and a model attribute
    inherited from nowhere is exactly the ambiguity a control must not have."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    env[COLUMN_GATHER_ENV_VAR] = "1" if cfg.get("pixel_batch") else "0"
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg11_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg11_{cfg['arm_id']}.json")
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
    row["subprocess_wall_s"] = subprocess_wall
    return row


def build_plan(arms):
    """The generator arms, then the measured arms in declared order."""
    plan = []
    for geometry in sorted({a["geometry"] for a in arms}):
        cell = cell_for(geometry)
        gen = dict(framework="torch", arm_class="generator", geometry=geometry,
                   cell=list(cell), n_dev=None, token=f"gen_{geometry}",
                   pixel_batch=None, role="generator",
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if SMOKE and DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        plan.append(gen)
    measured = []
    for arm in arms:
        cell = cell_for(arm["geometry"])
        if arm["pixel_batch"] is not None:
            tag = f"on{arm['pixel_batch']:05d}"
        elif arm["role"] == "anchor":
            tag = "anchor"
        else:
            tag = "off"
        cfg = dict(framework="torch", arm_class="instrument",
                   geometry=arm["geometry"], cell=list(cell),
                   n_dev=arm["n_dev"], token=arm["token"], role=arm["role"],
                   pixel_batch=arm["pixel_batch"],
                   arm_id=f"{arm['geometry']}_{cell[0]}_n{arm['n_dev']}_{tag}")
        if SMOKE and DEVICE != "cuda":
            # SMOKE ONLY: virtual cpu devices, so the n>1 wiring -- the band
            # fan-out, the column gather, the per-device workers, the witnesses
            # -- is exercised without CUDA.  The env pin is CUDA-only, so this
            # pins by device LIST and says so on the row.
            cfg["cpu_devices"] = [DEVICE] * arm["n_dev"]
        measured.append(cfg)
    return plan, measured


# ── the summary ───────────────────────────────────────────────────────────────
def _fmt(value, spec, dash="-"):
    """``format(value, spec)``, with a missing value rendered as a dash padded
    to the SAME width -- an unpadded dash shifts every column to its right."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    width = ""
    for char in spec:
        if char.isdigit():
            width += char
        else:
            break
    return f"{dash:>{int(width) if width else 1}}"


def _gb(value):
    return None if value is None else value / 2 ** 30


def print_arm_table(row):
    """One arm's per-device rows: the bracket, the busy sum, the launch count,
    the per-launch time and both normalized columns, plus the memory pair."""
    token = row.get("token")
    print(f"\n  [{token}] {row.get('geometry')} n={row.get('n_dev')} "
          f"{row.get('arm_kind')}"
          + (f" batch {row.get('pixel_batch')}" if row.get("pixel_batch")
             else "")
          + f"  composed {_fmt(row.get('vcd_warm'), '.2f')} s "
            f"(cold {_fmt(row.get('vcd_cold'), '.2f')} s)")
    knob = (row.get("knob_record") or {}).get("knobs")
    print(f"      knobs: {knob}")
    switch = row.get("switch_after_cold") or row.get("switch_at_install") or {}
    print(f"      resolver: gather={switch.get('resolver_says_gather')} "
          f"batch={switch.get('resolver_pixel_batch')} "
          f"env={switch.get('env_var_value')!r} "
          f"geometry_capable={switch.get('column_gather_geometry')} "
          f"rows_track_slices={switch.get('rows_track_slices')}")
    wit = row.get("witnesses") or {}
    print(f"      witnesses: gathers={wit.get('gather_calls')} "
          f"fan-outs={wit.get('broadcast_calls')} "
          f"cylinder heights={wit.get('cylinder_heights')} "
          f"values-block cols={wit.get('values_block_cols')}")
    header = (f"      {'device':<10}{'bracket_s':>11}{'busy_s':>10}"
              f"{'calls':>8}{'per_launch_ms':>15}{'ms_per_kpix':>13}"
              f"{'ms_per_slice':>14}{'peak_GB':>10}{'modeled_GB':>12}"
              f"{'mod/meas':>10}")
    print(header)
    mem = (row.get("memory") or {})
    modeled = mem.get("modeled_peak_per_device") or []
    ratios = mem.get("ratio_per_device") or []
    for index, dev in enumerate(row.get("per_device") or []):
        model_gb = _gb(modeled[index]) if index < len(modeled) else None
        ratio = ratios[index] if index < len(ratios) else None
        mark = ""
        if isinstance(ratio, float) and ratio < 1.0:
            mark = " <-FLOOR"
        print(f"      {dev['device']:<10}"
              f"{_fmt(dev['bracket_span_s'], '11.3f')}"
              f"{_fmt(dev['busy_sum_s'], '10.3f')}"
              f"{_fmt(dev['busy_calls'], '8.0f')}"
              f"{_fmt(dev['per_launch_ms'], '15.3f')}"
              f"{_fmt(dev['ms_per_kpix'], '13.5f')}"
              f"{_fmt(dev['ms_per_slice'], '14.5f')}"
              f"{_fmt(_gb(dev['peak_bytes']), '10.2f')}"
              f"{_fmt(model_gb, '12.2f')}"
              f"{_fmt(ratio, '10.3f')}{mark}")


def _rows_for(rows, geometry, n_dev=None):
    out = [r for r in rows if not r.get("error")
           and r.get("arm_class") != "generator"
           and r.get("geometry") == geometry]
    if n_dev is not None:
        out = [r for r in out if r.get("n_dev") == n_dev]
    return out


def speed_block(rows, geometry):
    """The speed column, per device count: the flag-off control against the best
    flag-on arm, on the composed wall and on the forward busy.

    THE RULE, stated before the numbers.  The speed gate passes at a device
    count when the best flag-on arm's composed wall is BELOW the control's by
    more than the control's own pass-to-pass spread.  A win inside that spread
    is not a win this instrument can resolve, and is reported as such rather
    than counted."""
    out = []
    for n_dev in sorted({r.get("n_dev") for r in _rows_for(rows, geometry)
                         if (r.get("n_dev") or 0) > 1}):
        block = _rows_for(rows, geometry, n_dev)
        control = next((r for r in block if not r.get("pixel_batch")), None)
        gathers = [r for r in block if r.get("pixel_batch")]
        if control is None or not gathers:
            out.append(dict(n=n_dev, resolvable=False,
                            why=("no control in this run" if control is None
                                 else "no flag-on arm in this run")))
            continue
        best = min(gathers, key=lambda r: (r.get("vcd_warm") is None,
                                           r.get("vcd_warm")))
        walls = control.get("vcd_warm_all") or []
        spread = ((max(walls) - min(walls)) if len(walls) > 1 else 0.0)
        c_comp, b_comp = control.get("vcd_warm"), best.get("vcd_warm")
        c_busy = control.get("forward_busy_max_s")
        b_busy = best.get("forward_busy_max_s")
        out.append(dict(
            n=n_dev, resolvable=True,
            control_token=control.get("token"), best_token=best.get("token"),
            best_batch=best.get("pixel_batch"),
            control_composed_s=c_comp, best_composed_s=b_comp,
            composed_speedup=((c_comp / b_comp) if c_comp and b_comp else None),
            control_busy_s=c_busy, best_busy_s=b_busy,
            busy_speedup=((c_busy / b_busy) if c_busy and b_busy else None),
            control_spread_s=spread,
            margin_s=((c_comp - b_comp) if c_comp and b_comp else None),
            passes=bool(c_comp and b_comp and (c_comp - b_comp) > spread),
            all_batches=[(r.get("pixel_batch"), r.get("vcd_warm"),
                          r.get("forward_busy_max_s")) for r in gathers]))
    return out


def value_block(values, geometry):
    """The value column: the largest distance any flag-on arm of this geometry
    sits from its control or from the anchor, on each metric, with the floors
    beside it.

    The bar and the expectation are two different things and are kept apart.
    The FAILING BAR is PARITY_FLOOR, read against max_rel_of_peak, which is the
    form the standing suites use.  The EXPECTATION is mg10's e-6 relative L2 to
    the anchor; it is printed, marked when it is left behind by an order of
    magnitude, and fails nothing."""
    entries = [v for v in values if v.get("geometry") == geometry
               and v.get("batch")]
    floors = [v for v in values if v.get("geometry") == geometry]

    def biggest(key, metric):
        best = None
        for entry in entries:
            block = entry.get(key)
            if not block:
                continue
            value = block.get(metric)
            if value is None:
                continue
            if best is None or value > best[0]:
                best = (value, entry.get("token"), entry.get("batch"))
        return best

    def floor_range(metric):
        seen = [v["own_pass_to_pass"][metric] for v in floors
                if v.get("own_pass_to_pass")
                and v["own_pass_to_pass"].get(metric) is not None]
        return (min(seen), max(seen)) if seen else None

    worst_control = biggest("vs_control", "max_rel_of_peak")
    worst_anchor = biggest("vs_anchor", "max_rel_of_peak")
    bar_candidates = [b for b in (worst_control, worst_anchor) if b]
    worst_bar = max(bar_candidates, key=lambda b: b[0]) if bar_candidates else None
    exp_anchor = biggest("vs_anchor", "rel_l2")
    checksum_worst = None
    for entry in entries:
        for key in ("checksum_vs_control", "checksum_vs_anchor"):
            value = entry.get(key)
            if value is None:
                continue
            if checksum_worst is None or value > checksum_worst[0]:
                checksum_worst = (value, entry.get("token"), key)
    checksum_floor = [v.get("checksum_repeat_spread") for v in floors
                      if v.get("checksum_repeat_spread") is not None]
    return dict(
        arms=len(entries),
        worst_vs_control=worst_control, worst_vs_anchor=worst_anchor,
        worst_for_bar=worst_bar,
        bar=PARITY_FLOOR, bar_citation=PARITY_FLOOR_CITATION,
        passes=(bool(worst_bar and worst_bar[0] < PARITY_FLOOR)
                if worst_bar else None),
        sample_floor_max_rel=floor_range("max_rel_of_peak"),
        sample_floor_rel_l2=floor_range("rel_l2"),
        expectation_rel_l2=EXPECTATION_REL_L2,
        expectation_marker_at=EXPECTATION_MARKER_AT,
        expectation_reading=exp_anchor,
        expectation_exceeded=(bool(exp_anchor
                                   and exp_anchor[0] >= EXPECTATION_MARKER_AT)
                              if exp_anchor else None),
        checksum_worst=checksum_worst,
        checksum_floor=((min(checksum_floor), max(checksum_floor))
                        if checksum_floor else None))


def memory_block(rows, geometry):
    """The memory column: the smallest modeled/measured ratio over this
    geometry's arms, and which arm and device carried it.

    THE GATING SCOPE, stated before the numbers.  The gate is on the FLAG-ON
    arms, because the column-gather path is the pricing this campaign is
    confirming.  A control or anchor below 1.00 is recorded and marked, and does
    not fail the flip: it would be a pre-existing ledger defect on a path the
    library already ships, which is a different finding and a different fix."""
    worst_gather = worst_other = None
    per_arm = []
    for row in _rows_for(rows, geometry):
        mem = row.get("memory") or {}
        if not mem.get("available"):
            per_arm.append(dict(token=row.get("token"), available=False,
                                why=mem.get("why")))
            continue
        entry = dict(token=row.get("token"), n=row.get("n_dev"),
                     batch=row.get("pixel_batch"), kind=row.get("arm_kind"),
                     min_ratio=mem.get("min_ratio"),
                     device=mem.get("min_ratio_device"),
                     violation=mem.get("floor_violation"),
                     cold_only=mem.get("violation_only_in_cold"),
                     modeled=mem.get("modeled_peak_per_device"),
                     measured=mem.get("measured_peak_per_device"),
                     available=True)
        per_arm.append(entry)
        if entry["min_ratio"] is None:
            continue
        target = "gather" if row.get("pixel_batch") else "other"
        current = worst_gather if target == "gather" else worst_other
        if current is None or entry["min_ratio"] < current["min_ratio"]:
            if target == "gather":
                worst_gather = entry
            else:
                worst_other = entry
    return dict(per_arm=per_arm, worst_gather=worst_gather,
                worst_other=worst_other,
                passes=(None if worst_gather is None
                        else bool(worst_gather["min_ratio"] >= 1.0)))


def caveat_block(rows):
    """Section 13's caveat, answered: the parallel prediction against the swept
    best.  Stated as measured-against-predicted and nothing else."""
    out = []
    for n_dev, predicted in sorted(PARALLEL_PREDICTION.items()):
        block = _rows_for(rows, "parallel", n_dev)
        control = next((r for r in block if not r.get("pixel_batch")), None)
        gathers = [r for r in block if r.get("pixel_batch")]
        if control is None or not gathers:
            out.append(dict(n=n_dev, resolvable=False, predicted=predicted))
            continue
        best_busy = min(gathers, key=lambda r: (r.get("forward_busy_max_s")
                                                is None,
                                                r.get("forward_busy_max_s")))
        best_comp = min(gathers, key=lambda r: (r.get("vcd_warm") is None,
                                                r.get("vcd_warm")))
        c_busy = control.get("forward_busy_max_s")
        c_comp = control.get("vcd_warm")
        out.append(dict(
            n=n_dev, resolvable=True, predicted=predicted,
            measured_busy_from_s=c_busy,
            measured_busy_to_s=best_busy.get("forward_busy_max_s"),
            busy_batch=best_busy.get("pixel_batch"),
            predicted_busy_factor=(predicted["busy_from_s"]
                                   / predicted["busy_to_s"]),
            measured_busy_factor=((c_busy / best_busy["forward_busy_max_s"])
                                  if c_busy and best_busy.get(
                                      "forward_busy_max_s") else None),
            measured_composed_from_s=c_comp,
            measured_composed_to_s=best_comp.get("vcd_warm"),
            composed_batch=best_comp.get("pixel_batch"),
            predicted_composed_factor=(predicted["composed_from_s"]
                                       / predicted["composed_to_s"]),
            measured_composed_factor=((c_comp / best_comp["vcd_warm"])
                                      if c_comp and best_comp.get("vcd_warm")
                                      else None),
            swept_batches=sorted(r.get("pixel_batch") for r in gathers)))
    return out


def print_verdict(rows, values, geometry):
    """ONE block per geometry: mechanical, quotable, and carrying no prose."""
    speed = speed_block(rows, geometry)
    value = value_block(values, geometry)
    memory = memory_block(rows, geometry)
    print(f"\n=== GEOMETRY: {geometry} ===")

    print("  SPEED  (rule: the best flag-on arm's composed wall must be below "
          "the control's by")
    print("          more than the control's own pass-to-pass spread)")
    speed_pass = bool(speed) and all(s.get("passes") for s in speed)
    for entry in speed:
        if not entry.get("resolvable"):
            print(f"    n={entry['n']}: NOT RESOLVABLE -- {entry.get('why')}")
            continue
        print(f"    n={entry['n']}: control {entry['control_token']} "
              f"{_fmt(entry['control_composed_s'], '.2f')} s composed  ->  best "
              f"flag-on {entry['best_token']} (batch {entry['best_batch']}) "
              f"{_fmt(entry['best_composed_s'], '.2f')} s  "
              f"= {_fmt(entry['composed_speedup'], '.3f')}x")
        print(f"           forward busy {_fmt(entry['control_busy_s'], '.2f')} "
              f"-> {_fmt(entry['best_busy_s'], '.2f')} s = "
              f"{_fmt(entry['busy_speedup'], '.3f')}x; margin "
              f"{_fmt(entry['margin_s'], '.2f')} s against a control spread of "
              f"{_fmt(entry['control_spread_s'], '.2f')} s  "
              f"[{'PASS' if entry['passes'] else 'FAIL'}]")
        for batch, composed, busy in sorted(entry["all_batches"]):
            print(f"           batch {batch:>7}: composed "
                  f"{_fmt(composed, '.2f')} s, forward busy "
                  f"{_fmt(busy, '.2f')} s")

    print(f"  VALUE  (failing bar {PARITY_FLOOR:.0e} relative, read against "
          f"max|a-b|/max|b|;")
    print(f"          the bar is the shipped parity floor: {PARITY_FLOOR_CITATION})")
    worst = value.get("worst_for_bar")
    if worst is None:
        print("    NOT RESOLVABLE -- no flag-on arm carried a distance in this "
              "run")
        value_pass = None
    else:
        value_pass = bool(worst[0] < PARITY_FLOOR)
        print(f"    max distance over the flag-on arms: {worst[0]:.3e} "
              f"({worst[1]}, batch {worst[2]}) against the bar "
              f"{PARITY_FLOOR:.0e}  [{'PASS' if value_pass else 'FAIL'}]")
        floor = value.get("sample_floor_max_rel")
        if floor:
            print(f"    the arms' own repeat floors on that metric: "
                  f"{floor[0]:.3e} to {floor[1]:.3e}")
        cs = value.get("checksum_worst")
        csf = value.get("checksum_floor")
        if cs:
            print(f"    checksum family: max {cs[0]:.3e} ({cs[1]}, {cs[2]})"
                  + (f" against repeat floors {csf[0]:.3e} to {csf[1]:.3e}"
                     if csf else ""))
    reading = value.get("expectation_reading")
    if reading is None:
        print("    EXPECTATION: no relative-L2 distance to the anchor in this "
              "run")
    else:
        mark = ("  [ABOVE THE REGISTERED EXPECTATION]"
                if value.get("expectation_exceeded") else "  [within it]")
        print(f"    EXPECTATION (logged, never a failure): relative L2 to the "
              f"anchor {reading[0]:.3e}")
        print(f"      registered at {EXPECTATION_REL_L2:.2e} by mg10; marker "
              f"fires at {EXPECTATION_MARKER_AT:.0e}{mark}")

    print("  MEMORY (rule: modeled/measured >= 1.00 passes; below 1.00 is a "
          "FLOOR VIOLATION.")
    print("          The gate is on the flag-on arms; a control below 1.00 is "
          "recorded, not gating)")
    worst_gather = memory.get("worst_gather")
    memory_pass = memory.get("passes")
    if worst_gather is None:
        print("    NOT RESOLVABLE -- no flag-on arm reported a device peak "
              "(the CPU smoke has none)")
    else:
        mark = "FLOOR VIOLATION" if worst_gather["min_ratio"] < 1.0 else "PASS"
        print(f"    min modeled/measured over the flag-on arms: "
              f"{worst_gather['min_ratio']:.3f} "
              f"({worst_gather['token']} on {worst_gather['device']}, modeled "
              f"{_fmt(_gb(worst_gather['modeled'][0]), '.2f')} GB vs measured "
              f"{_fmt(_gb(worst_gather['measured'][0]), '.2f')} GB on device 0)"
              f"  [{mark}]")
        if worst_gather.get("cold_only"):
            print("      the violation is in the COLD pass only; the timed "
                  "reconstructions stayed at or above 1.00")
    worst_other = memory.get("worst_other")
    if worst_other is not None and worst_other["min_ratio"] < 1.0:
        print(f"    RECORDED, NOT GATING: the control/anchor arm "
              f"{worst_other['token']} reads {worst_other['min_ratio']:.3f} on "
              f"{worst_other['device']} -- a ledger defect on the shipped path, "
              f"which is a different finding")

    if geometry == "parallel":
        print("  SECTION 13 CAVEAT CHECK (measured against predicted; no "
              "verdict is drawn here)")
        for entry in caveat_block(rows):
            predicted = entry["predicted"]
            if not entry.get("resolvable"):
                print(f"    n={entry['n']}: NOT RESOLVABLE in this run; "
                      f"predicted forward busy "
                      f"{predicted['busy_from_s']} -> {predicted['busy_to_s']} s")
                continue
            print(f"    n={entry['n']} forward busy: predicted "
                  f"{predicted['busy_from_s']:.1f} -> "
                  f"{predicted['busy_to_s']:.1f} s "
                  f"({entry['predicted_busy_factor']:.2f}x); measured "
                  f"{_fmt(entry['measured_busy_from_s'], '.2f')} -> "
                  f"{_fmt(entry['measured_busy_to_s'], '.2f')} s "
                  f"({_fmt(entry['measured_busy_factor'], '.2f')}x at batch "
                  f"{entry['busy_batch']})")
            print(f"    n={entry['n']} composed:     predicted "
                  f"{predicted['composed_from_s']:.1f} -> "
                  f"{predicted['composed_to_s']:.1f} s "
                  f"({entry['predicted_composed_factor']:.2f}x); measured "
                  f"{_fmt(entry['measured_composed_from_s'], '.2f')} -> "
                  f"{_fmt(entry['measured_composed_to_s'], '.2f')} s "
                  f"({_fmt(entry['measured_composed_factor'], '.2f')}x at batch "
                  f"{entry['composed_batch']})")
            print(f"    the prediction was measured with the pixel count held "
                  f"FULL, and the gather cuts pixels to the batch; the swept "
                  f"batches were {entry['swept_batches']}")

    failed = []
    if not speed_pass:
        failed.append("speed")
    if value_pass is not True:
        failed.append("value")
    if memory_pass is not True:
        failed.append("memory")
    if failed:
        print(f"  GATES FAIL: {geometry}: {', '.join(failed)}")
    else:
        print(f"  GATES PASS - FLIP AUTHORIZED: {geometry}")
    return dict(geometry=geometry, speed=speed, value=value, memory=memory,
                caveat=(caveat_block(rows) if geometry == "parallel" else None),
                speed_pass=speed_pass, value_pass=value_pass,
                memory_pass=memory_pass,
                gates_pass=not failed, failed_gates=failed)


def summarize(rows, out_path):
    """The whole read-out: the per-arm tables, the value table, the memory
    table, and then one verdict block per geometry."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    failed = [r for r in rows if r.get("error")]
    print("\n" + "=" * 78)
    print(f"mg11 -- the combined flip gates, {len(live)} arms on {RUN_LABEL} "
          f"({DEVICE})")
    print("=" * 78)
    print("  mg11 patches NOTHING.  Every arm below drives the shipped library "
          "through the")
    print("  environment switch and the documented pixel-batch attribute, so "
          "these numbers")
    print("  are the library's own and not a prototype's.")
    if DEVICE != "cuda":
        print()
        print("  SMOKE RUN -- READ THE GATE LINES AS EXERCISE, NOT AS RESULT.")
        print("  This is the local CPU run.  Two of the three gate columns "
              "cannot mean anything")
        print("  here and the blocks below will say so in their own words: "
              "there is no device")
        print("  peak counter, so the MEMORY ratio is NOT RESOLVABLE; and the "
              "smoke cell is a few")
        print("  dozen voxels, far below the size at which either shape pays, "
              "so a SPEED verdict")
        print("  either way is noise.  What the smoke does establish is the "
              "part that does not")
        print("  depend on scale: every witness fires, every negative witness "
              "holds, the two")
        print("  recorders agree, and the value distances are computed.  An "
              "authorization is a")
        print("  cluster reading and only a cluster reading.")

    for row in live:
        print_arm_table(row)

    values = value_table(rows)
    print("\n  -- the value table (two metrics, each read against a floor "
          "measured in the same arm) --")
    header = (f"      {'token':<14}{'n':>3}{'batch':>8}"
              f"{'own_L2':>11}{'ctl_L2':>11}{'anch_L2':>11}"
              f"{'anch_maxrel':>13}{'cs_repeat':>12}{'cs_anchor':>12}")
    print(header)
    for entry in values:
        own = (entry.get("own_pass_to_pass") or {}).get("rel_l2")
        ctl = (entry.get("vs_control") or {}).get("rel_l2")
        anch = (entry.get("vs_anchor") or {}).get("rel_l2")
        anch_max = (entry.get("vs_anchor") or {}).get("max_rel_of_peak")
        print(f"      {str(entry['token']):<14}{_fmt(entry['n'], '3.0f')}"
              f"{_fmt(entry['batch'], '8.0f')}"
              f"{_fmt(own, '11.3e')}{_fmt(ctl, '11.3e')}{_fmt(anch, '11.3e')}"
              f"{_fmt(anch_max, '13.3e')}"
              f"{_fmt(entry.get('checksum_repeat_spread'), '12.3e')}"
              f"{_fmt(entry.get('checksum_vs_anchor'), '12.3e')}")

    print("\n  -- the memory table (modeled by the library's own ledger; "
          "measured by torch) --")
    print(f"      {'token':<14}{'n':>3}{'batch':>8}{'min mod/meas':>14}"
          f"{'device':>10}  dominant phase")
    for geometry in ("cone", "parallel"):
        for entry in memory_block(rows, geometry)["per_arm"]:
            if not entry.get("available"):
                print(f"      {str(entry['token']):<14}   "
                      f"{'(no device peak counter)':>30}")
                continue
            row = next((r for r in live if r.get("token") == entry["token"]),
                       {})
            phase = ((row.get("ledger") or {}).get("dominant_phase_per_device")
                     or [None])[0]
            mark = " <-FLOOR VIOLATION" if entry.get("violation") else ""
            print(f"      {str(entry['token']):<14}{_fmt(entry['n'], '3.0f')}"
                  f"{_fmt(entry['batch'], '8.0f')}"
                  f"{_fmt(entry['min_ratio'], '14.3f')}"
                  f"{str(entry['device']):>10}  {phase}{mark}")
    # WHICH LEDGER THE COLUMN ABOVE IS.  The gate reads the library's own
    # last_memory_ledger -- the ledger the run itself was decided with -- and
    # falls back to mg11's build of the same entry point where the library made
    # none.  The two can differ in one input, the partition sequence (see
    # ledger_reading), and mg11's build can only price the same or more, so a
    # library number ABOVE mg11's would contradict that and is called out.
    absent = [r.get("token") for r in live
              if r.get("library_ledger_agrees") is None]
    disagreed = [r.get("token") for r in live
                 if r.get("library_ledger_agrees") is False]
    above = [r.get("token") for r in live
             if (r.get("ledger") or {}).get("library_above_harness")]
    if absent:
        print(f"      the library built no ledger of its own on {absent}, so "
              "the column above is mg11's")
        print("      build of the same entry point (an explicit device layout "
              "takes the branch that")
        print("      skips the preflight -- the CPU smoke pins that way)")
    if disagreed:
        print(f"      the library's ledger and mg11's differ on {disagreed}; "
              "the column above is the")
        print("      LIBRARY's, which is the number the run was decided with.  "
              "The expected cause is")
        print("      the partition sequence: the run visits the sequence recon "
              "generated, mg11's")
        print("      build falls back to the model's full parameter sequence "
              "and so prices >= it.")
    if above:
        print(f"      UNEXPECTED on {above}: the library's modeled peak is "
              "ABOVE mg11's on some device.")
        print("      The partition-sequence argument says it cannot be, so an "
              "input other than the")
        print("      sequence differs between the two calls; read the plan "
              "fields on the row.")
    if not absent and not disagreed:
        print("      the library's own last_memory_ledger agrees with mg11's "
              "build on every arm")

    if failed:
        print(f"\n  {len(failed)} ARM(S) FAILED and are not in any block above:")
        for row in failed:
            first = str(row.get("error", "")).strip().splitlines()
            print(f"      {row.get('token')}: "
                  f"{first[-1] if first else 'unknown'}")

    verdicts = []
    print("\n" + "=" * 78)
    print("  THE VERDICT BLOCKS")
    print("=" * 78)
    for geometry in ("cone", "parallel"):
        if not _rows_for(rows, geometry):
            print(f"\n=== GEOMETRY: {geometry} ===")
            print("  NOT RUN IN THIS JOB -- no arm of this geometry produced a "
                  "row, so no gate is")
            print("  answered and nothing here may be read as an authorization.")
            print(f"  GATES FAIL: {geometry}: not run")
            verdicts.append(dict(geometry=geometry, gates_pass=False,
                                 failed_gates=["not run"]))
            continue
        verdicts.append(print_verdict(rows, values, geometry))
    print(f"\nrows: {out_path}")
    return dict(verdicts=verdicts, values=values,
                speed={g: speed_block(rows, g) for g in ("cone", "parallel")},
                memory={g: memory_block(rows, g) for g in ("cone", "parallel")},
                caveat=caveat_block(rows))


# ── the wall arithmetic ───────────────────────────────────────────────────────
def wall_estimate(generators, measured):
    """Low and high wall estimates, in seconds, from mg10's MEASURED per-arm
    subprocess walls on this cell and node class.

    Cone's flag-on arms are the same shape mg10 ran at batch 8192, so their base
    is measured and the band is narrow.  Parallel's flag-on arms have never been
    run, which is what this job is for, so they take mg10's own rule for an
    unmeasured arm: the low end assumes the arm costs what its control costs and
    the high end assumes half again as much."""
    low = high = GENERATOR_S * len(generators)
    for cfg in measured:
        geometry, n_dev = cfg["geometry"], cfg["n_dev"]
        gather = cfg["pixel_batch"] is not None
        base = MG10_WALL_S.get((geometry, n_dev, "on" if gather else "off"))
        if base is None:
            base = MG10_WALL_S.get((geometry, n_dev, "off"), 300)
        low += base
        if not gather:
            high += int(base * CONTROL_HIGH_FACTOR)
        elif geometry == "cone":
            high += int(base * CONE_ON_HIGH_FACTOR)
        else:
            high += int(base * PARALLEL_ON_HIGH_FACTOR)
    return low, high


def main():
    arms = selected_arms()
    generators, measured = build_plan(arms)
    if "--dry-run" in sys.argv:
        low, high = wall_estimate(generators, measured)
        num_slices = num_slices_for(cell_for("cone"))
        print(f"mg11 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  cell {cell_for('cone')}, slices {num_slices}, warm repeats "
              f"{WARM_REPEATS}, iterations {VCD_ITERATIONS}, device {DEVICE}, "
              f"results {RESULTS_DIR}")
        print(f"  cone batches swept {CONE_BATCHES}; parallel batches swept "
              f"{PARALLEL_BATCHES}")
        print("  mg11 PATCHES NOTHING: every arm below is the shipped library "
              "driven through")
        print(f"  {COLUMN_GATHER_ENV_VAR} and model.forward_project_pixel_batch.")
        for cfg in generators:
            print(f"  {cfg['arm_id']:<44} {'(generator, switch forced OFF)':>34}")
        for cfg in measured:
            n_dev = cfg["n_dev"]
            shard = num_slices // max(1, n_dev or 1)
            if cfg["pixel_batch"]:
                cyl = cylinder_bytes(cfg["pixel_batch"], num_slices)
                note = (f"FLAG ON, batch {cfg['pixel_batch']:>5}: cylinders "
                        f"{cfg['pixel_batch']} x {num_slices} = "
                        f"{cyl / 1e6:.1f} MB"
                        + ("" if cyl <= TRANSIENT_BOUND_BYTES
                           else "  [ABOVE THE 150 MB BOUND THE RULING NAMES]"))
            elif n_dev == 1:
                note = ("value anchor: the shipped single-device path, switch "
                        "forced off")
            else:
                note = (f"flag OFF control: the shipped banded walk, one "
                        f"{shard}-slice band per slice-owner")
            print(f"  {cfg['arm_id']:<44} n={n_dev} [{cfg['token']}] {note}")
        print(f"  wall estimate {low / 60:.0f} to {high / 60:.0f} minutes "
              f"(mg10's MEASURED per-arm walls as the base; cone's flag-on "
              f"arms are the shape mg10 ran, parallel's are not, so only "
              f"parallel's high end assumes half again as much).")
        if high > 2.5 * 3600:
            print("  THAT IS OVER 2.5 HOURS: trim the four-device batch sweeps "
                  "to the ends of the range,")
            print("    MG11_CONE_BATCHES / MG11_PARALLEL_BATCHES, or drop arms "
                  "by token with MG11_ARMS.")
        else:
            print("  That is under the 2.5-hour ceiling, so no arm is trimmed "
                  "and every swept batch runs.")
        print("  if it must be cut: MG11_ARMS drops arms by token.  Trim the "
              "four-device arms")
        print("  first (c4_*, p4_*), then the middle batches of each sweep.  "
              "Both two-device")
        print("  blocks are the core, and each geometry's anchor (c1, p1) must "
              "stay with its own")
        print("  arms: the value distances are computed across arms in ONE "
              "process, so a job")
        print("  without the anchor loses the vs_anchor column for every arm "
              "in it.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg11_flip_gates_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg11 flip gates on {RUN_LABEL} ({DEVICE}); {len(measured)} arms "
          f"-> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY: a truncated job still yields the arms it
    # finished, which is why the arm order runs the core first.
    with open(out_path, "w") as sink:
        for cfg in generators + measured:
            print(f"  [{cfg['arm_id']}]", flush=True)
            row = run_one(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        summary = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    if os.environ.get("MG11_KEEP_ARTIFACTS", "0") != "1":
        # The sinograms and the value samples are internal to this job -- the
        # distances are computed above, before anything is removed.
        for geometry in sorted({a["geometry"] for a in arms}):
            for path in (_sino_path(geometry, cell_for(geometry)),
                         _md5_path(geometry, cell_for(geometry))):
                if os.path.exists(path):
                    os.remove(path)
        for row in rows:
            for path in row.get("value_sample_paths") or []:
                if os.path.exists(path):
                    os.remove(path)
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


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker_main(sys.argv[2], sys.argv[3])
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        main()
