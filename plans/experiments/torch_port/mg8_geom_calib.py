"""mg8 -- MEMORY CALIBRATION FOR THE TWO NEW GEOMETRIES (translation and
multiaxis parallel): modeled peak memory against measured peak memory, on one
to four GPUs.

WHY THIS RUN EXISTS.

Before a reconstruction starts, mbirtorch computes a "memory ledger": a
closed-form estimate, phase by phase and device by device, of how many bytes
the run will hold.  The ledger decides how many GPUs the run spreads across and
refuses a run that cannot fit anywhere.  It has one hard rule -- it may
over-estimate but it must never UNDER-estimate, because an under-estimate lets
a doomed run start and then die inside the allocator, which is precisely the
failure the ledger exists to prevent.

A large part of that estimate is the "view batch" term.  Projecting a
sinogram is done a few views at a time (a "view batch"), and each batch
materializes a large temporary array.  How many bytes one view costs is decided
by a per-view cost model shared by two consumers: the driver, which uses it to
pick the batch size, and the ledger, which uses it to price that batch's
memory.

For cone-beam and parallel-beam that cost model is a real measurement, because
those geometries have hand-written (Triton) kernels that declare their own
per-view cost.  TranslationModel and MultiAxisParallelModel have NO hand-written
kernels.  They run on general torch code ("torch bodies"), and for a torch body
the cost model falls back to a proxy: one nominal slab, num_pixels x columns x
4 bytes.  That proxy has now been measured against the real bodies, and it is
short -- 10x short on the multiaxis forward projection and 4.3x short on the
multiaxis back projection, measured through the compiled bodies the driver
actually binds (prerelease_review_multiaxis_2026-08-10.md; the same review
records 16.0x and 4.5x for the translation bodies in eager form).  The library's
own source comment estimates only 2-5x, so the exposure is larger than the code
believes.

The batch term is not a rounding error in these ledgers.  At the production
translation shape the back-projection batch term is 54.3 GiB of a 115.7 GiB
modeled four-device peak.  A 4-10x shortfall on a term that size puts the true
peak far above the model, in the one direction the ledger may not err.

So: this run measures the modeled peak against the measured peak, per device,
for both new geometries at three device counts, so the ledger's torch-body
terms can be corrected FROM DATA rather than from a guess.

REGISTERED EXPECTATION, stated before the run so the result cannot be
rationalized afterwards.  The ratio reported below is modeled / measured, and
the ledger's acceptance band is 1.00 to 1.30.  Ratios BELOW 1.00 are EXPECTED
here on the arms whose modeled peak is set by a projection phase, and possibly
by a wide margin.  An under-floor reading in this job is the FINDING, not a
failure: it is the measurement the correction will be built from.  Arms whose
modeled peak is set by a non-projection phase (several translation arms are
priced by sinogram-shaped algebra, not by the batch term) should land inside
the band, and a badly under-floor reading THERE would mean something other than
the per-view cost model is missing.

The summary prints, per geometry, the worst (lowest) ratio and the multiplier
the per-view cost model would have needed for the modeled peak to reach the
measured one.

TERMS USED BELOW, defined once here:
    arm            one measured configuration: one geometry, one shape, one
                   device count, run in its own fresh process.
    sinogram       the measured projection data, shaped
                   (views, detector rows, detector channels).
    recon shape    the reconstruction volume, (rows, columns, slices).
    modeled peak   the ledger's estimate of peak bytes on one device.
    measured peak  torch.cuda.max_memory_allocated on that device, which is
                   CUDA-only -- there is no such counter on CPU.
    ratio          modeled / measured on one device.
    dominant phase the phase whose per-device bytes set the modeled peak.
    batch term     the ledger's 'forward batch' or 'back batch' entry: the
                   bytes one view batch holds, which is exactly
                   view_batch x per-view cost, and therefore exactly the
                   quantity a corrected cost model would scale.
    implied        the multiplier on the per-view cost that would have lifted
    multiplier     the modeled peak to the measured one on that device:
                   1 + (measured - modeled) / batch term.

THE ARMS.  12 measured arms plus 4 untimed generators.

    multiaxis    (1024, 1008, 992)  ->  recon (992, 992, 1148)
    multiaxis    ( 512,  448, 384)  ->  recon (384, 384,  510)
    translation  ( 256, 1900, 3000) ->  recon (118, 360,  240)
    translation  ( 256,  950, 1500) ->  recon ( 59, 180,  120)

each at 1, 2 and 4 devices.  The two multiaxis shapes are the shapes the
cone/parallel calibration used, so the new rows sit beside the existing ones.
The large translation shape is the production translation-CT shape the
prerelease review priced; the smaller one is that shape halved on both detector
axes (and on both translation spacings), so the trend over problem size is
visible rather than inferred from a single point.

Every arm supplies weights.  There is no unweighted arm: the unweighted
configuration exists to size one particular resident array, which the
cone/parallel calibration already measured and which has nothing to do with the
per-view cost model this run is about.

HOW THE MEASUREMENT IS TAKEN.  One arm per subprocess, so no allocator state
carries from one arm to the next.  The device count is pinned by setting
MBIRTORCH_NUM_DEVICES in that subprocess's environment (popped first, then set,
so nothing is inherited), and the arm then asserts the device list the model
actually realized.  Pinning by environment rather than by an explicit
configure_devices call matters: an explicit device list turns off the automatic
device-count search, and the ledger is what that search consumes, so the
measurement has to be taken on the branch that uses it.

MBIRTORCH_MEMORY_CALIBRATION=1 is set for the measurement arms.  That mode
computes the ledger at any device count including one, resets
torch.cuda.max_memory_allocated at the start of the reconstruction, and reads
it at the end.  Because it RESETS that counter it owns it, which is why it is
off by default and why the generator subprocesses do not get it.

DELIBERATE DIFFERENCES FROM mg2_ledger_calib.py, the file this one is modeled
on, each stated so a reader does not take them for oversights:

  1. ONE reconstruction per arm, and that reconstruction is the reading.  mg2
     ran a cold pass to pay the compilation cost and then read a warm pass.
     Here the single pass is read, compilation included, because a production
     run pays that compilation exactly once and the ledger is supposed to cover
     the peak a production run actually reaches.  The consequence is that these
     ratios are NOT directly comparable to mg2's warm-pass ratios; they are a
     stricter reading of the same quantity.  MG8_REPEATS>1 adds further passes,
     recorded separately, for anyone who wants the warm number too -- the first
     pass remains the reading.

  2. A ratio below the band is NOT counted as a failure.  Only things that
     would invalidate a row are: a wrong realized device count, a missing
     calibration environment, a sinogram checksum mismatch, an unexpected recon
     shape, or a projection body that turned out to carry its own cost model.
     The band verdict is reported on every row and drives the summary, but it
     does not make the job "fail" -- see the registered expectation above.

  3. The arm check on the bodies is inverted.  mg2 asserted that hand-written
     kernels were bound.  Here the load-bearing fact is the opposite: neither
     body may carry a '_view_batch_cost' attribute, because a body that carried
     one would not be using the proxy this run exists to measure.  If a kernel
     is ever written for these geometries, that check fires and says so.

WHAT IS NOT MEASURED HERE.  This run does not attempt to correct the cost
model, and it does not measure the per-view transient in isolation -- the
prerelease review already did that with a single-body ablation.  It measures
the composed reconstruction, which is the quantity the ledger claims to
predict.  Note also that the implied multiplier printed at the end is a
first-order attribution: it scales the batch term while holding the batch COUNT
fixed.  A real correction would feed back into the batch chooser (a larger
per-view cost means a smaller batch), so the correction that finally lands will
not simply be this number multiplied through.

ARTIFACTS.  One sinogram per shape, written once, checksummed, and re-verified
by every arm that reads it: about 11 GiB in total (3.81 + 0.33 + 5.44 + 1.36
GiB).  Deleted at the end unless MG8_KEEP_ARTIFACTS=1 is exported.

Run:
    <torch python> mg8_geom_calib.py       on a 4-GPU node (mg8_gautschi.sbatch)
    python mg8_geom_calib.py --dry-run     anywhere: print the arm plan
    python mg8_geom_calib.py --help

Environment (export from the SUBMITTING SHELL, never through an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed strictly: an unrecognized token is an error, not a silent skip.
    MG8_RESULTS=<dir>                     where the jsonl and the sinograms go
    MG8_GEOMETRIES=multiaxis,translation  subset of the geometries
    MG8_SHAPES=ma1024,ma512,tct2k,tct1k   subset of the shapes, by name
    MG8_COUNTS=1,2,4                      subset of the device counts
    MG8_ITERATIONS=3                      reconstruction iterations per arm
    MG8_REPEATS=1                         passes per arm; pass 1 is the reading
    MG8_KEEP_ARTIFACTS=1                  keep the sinograms after the run
    MG8_SMOKE=1 / MG8_DEVICE=cpu          the local CPU smoke
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Each shape carries everything needed to rebuild its model, plus the recon
# shape and pixel count it produced when this file was written (mbirtorch at
# merged tip 1a2deb0, 2026-08-10).  The arm re-derives both and flags a
# mismatch: a change in either would silently move every number in the table,
# and the geometry defaults these two models inherit are new enough to be worth
# pinning down.
#
# translation shapes are built from a grid of translations.  The translation
# vectors, the source distances and the detector spacings together fix the
# recon shape; the values below are the ones that reproduce the production
# shape the prerelease review priced, (118, 360, 240) at 42,480 pixels.  The
# source distance convention -- source-to-isocenter = source-to-detector =
# half the smaller detector dimension -- is mbirtorch's own convention for
# generated translation data (utilities.py, the demo-data builder).
SHAPES = (
    dict(name="ma1024", geometry="multiaxis", cell=(1024, 1008, 992),
         recon_shape=(992, 992, 1148), num_pixels=771240),
    dict(name="ma512", geometry="multiaxis", cell=(512, 448, 384),
         recon_shape=(384, 384, 510), num_pixels=115164),
    dict(name="tct2k", geometry="translation", cell=(256, 1900, 3000),
         translations=(16, 16), spacing=(24.0, 16.0),
         recon_shape=(118, 360, 240), num_pixels=42480),
    dict(name="tct1k", geometry="translation", cell=(256, 950, 1500),
         translations=(16, 16), spacing=(12.0, 8.0),
         recon_shape=(59, 180, 120), num_pixels=10620),
)
GEOMETRIES = ("multiaxis", "translation")
COUNTS = (1, 2, 4)

# The CPU smoke's shapes are the two geometries' own test shapes, so the smoke
# exercises a configuration already known to reconstruct.  Its counts stop at
# two: at these sizes a four-way split of the translation recon would leave one
# virtual device owning no real data, which the layout validation rejects --
# correctly, and for reasons unrelated to anything this run measures.
SMOKE = os.environ.get("MG8_SMOKE", "0") == "1"
SMOKE_SHAPES = (
    dict(name="ma_tiny", geometry="multiaxis", cell=(16, 24, 20),
         recon_shape=(20, 20, 27), num_pixels=276),
    dict(name="tct_tiny", geometry="translation", cell=(16, 40, 32),
         translations=(4, 4), spacing=(3.0, 2.0),
         recon_shape=(2, 9, 6), num_pixels=18),
)
SMOKE_COUNTS = (1, 2)

DEVICE = os.environ.get("MG8_DEVICE", "cpu" if SMOKE else "cuda")
VCD_ITERATIONS = int(os.environ.get("MG8_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 12345               # the cone/parallel calibration's seed
# One pass is the reading (see difference 1 in the docstring).  Extra passes are
# recorded beside it and never replace it.
REPEATS = max(1, int(os.environ.get("MG8_REPEATS", "1")))

# The band the ledger itself declares (mbirtorch/_memory_ledger.py,
# CALIBRATION_BAND).  Repeated rather than imported so the harness can print the
# band it judged against even if the library's value moves underneath it; the
# arm records the library's value too, and a mismatch is reported.
BAND = (1.00, 1.30)
# The ledger entries that are exactly view_batch x per-view cost, and therefore
# exactly what a corrected per-view cost model would scale.  'forward block' is
# deliberately NOT here: it scales with the batch SIZE but not with the per-view
# cost, so it moves only if the batch chooser's answer changes.
BATCH_TERMS = ("forward batch", "back batch")

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
    "MG8_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed, cast=str):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
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
    """The shapes and device counts this run will measure, in declared order so
    the job order is reproducible."""
    all_shapes = SMOKE_SHAPES if SMOKE else SHAPES
    counts_all = SMOKE_COUNTS if SMOKE else COUNTS
    keep_names = _strict_subset("MG8_SHAPES", {s["name"] for s in all_shapes})
    keep_geoms = _strict_subset("MG8_GEOMETRIES", set(GEOMETRIES))
    shapes = [s for s in all_shapes
              if s["name"] in keep_names and s["geometry"] in keep_geoms]
    if not shapes:
        raise ValueError("MG8_SHAPES and MG8_GEOMETRIES together select no "
                         "shape")
    counts = _strict_subset("MG8_COUNTS", set(counts_all), int)
    counts = [n for n in counts_all if n in counts]
    return shapes, counts


def _shape_by_name(name):
    for spec in (SMOKE_SHAPES if SMOKE else SHAPES):
        if spec["name"] == name:
            return spec
    raise KeyError(f"no shape named {name!r}")


def _sino_path(name):
    return os.path.join(RESULTS_DIR, f"_mg8_sino_{name}.npy")


def _md5_path(name):
    return _sino_path(name) + ".md5"


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
    """The ONE host exit.

    A sharded array's ``gather()`` ALREADY returns a numpy array; calling
    ``.detach()`` on that result is a recorded failure that once cost a whole
    multi-device run its rows.  Nothing else in this file leaves the device.
    """
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
    """The weighting formula the other calibration runs use, so these arms are
    weighted the same way theirs are."""
    import numpy as np

    scale = float(np.max(sinogram))
    if scale <= 0:
        raise RuntimeError("sinogram is all zeros; the phantom did not project")
    return np.exp(-sinogram / (2 * scale)).astype(np.float32)


# ── the GPU health sample ─────────────────────────────────────────────────────
# A GPU that is thermally throttled produces a valid memory reading but an
# invalid timing one, and a hot node is worth knowing about even here, because
# it usually means a neighbour job is sharing the hardware -- which can also
# move the free-memory reading the ledger's preflight takes.
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
def _build_model(spec, pin_devices=None):
    """Build the model for one shape.

    On CUDA nothing is configured here: the device count is pinned through the
    environment, which leaves the model on the automatic branch where the
    ledger is actually consumed.  ``pin_devices`` is the CPU smoke's path only,
    where the environment pin does not apply -- the automatic search
    short-circuits when fewer than two CUDA devices are visible -- so the smoke
    pins by explicit device list and every row says which mechanism it used.
    """
    import numpy as np

    import mbirtorch

    cell = tuple(spec["cell"])
    if spec["geometry"] == "multiaxis":
        # Two angles per view: azimuth around the object, elevation (tilt) out
        # of the plane.  These are the geometry's own test defaults -- azimuths
        # evenly spaced over half a turn, elevations swept across +/- 0.5
        # radians.  The elevation range matters for the recon shape: the
        # automatic geometry divides the detector height by the smallest
        # |cos(elevation)|, and clamps that divisor at 0.1, so a range wide
        # enough to reach the clamp would inflate the slice count roughly
        # tenfold.  0.5 radians is far from the clamp.
        num_views = cell[0]
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            cell, np.stack([azimuth, elevation], axis=1))
    else:
        # A translation scan moves the object across a fixed source/detector
        # pair on a grid, so the "views" are grid positions rather than angles.
        num_x, num_z = spec["translations"]
        x_spacing, z_spacing = spec["spacing"]
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != cell[0]:
            raise RuntimeError(
                f'{spec["name"]}: {num_x}x{num_z} translations give '
                f"{vectors.shape[0]} views, but the sinogram has {cell[0]}")
        source_iso_dist = min(cell[1], cell[2]) / 2
        model = mbirtorch.TranslationModel(
            cell, vectors, source_detector_dist=source_iso_dist,
            source_iso_dist=source_iso_dist)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def _shape_check(model, spec):
    """Did the geometry defaults produce the recon shape this file registered?

    Recorded rather than raised: a moved default is worth knowing about, but it
    does not make the memory reading itself wrong.
    """
    realized = tuple(int(s) for s in model.get_params("recon_shape"))
    pixels = int(model.full_index_count())
    expected = tuple(spec["recon_shape"])
    return dict(recon_shape=list(realized), num_pixels_full=pixels,
                recon_shape_expected=list(expected),
                recon_shape_ok=(realized == expected),
                num_pixels_expected=int(spec["num_pixels"]),
                num_pixels_ok=(pixels == int(spec["num_pixels"])))


def _reference_view_charge(model, n_dev):
    """The per-view cost model's own numbers, at the full pixel set and the
    full column count.

    This is a SCALE REFERENCE, not the charge the sharded calls actually pay:
    under sharding each call passes its own band length as the column count, so
    the realized charge is smaller.  It is recorded because it is the quantity
    the correction will be applied to, and reading it here means the table can
    show what one view is currently charged without re-deriving the formula.
    """
    projector_functions = model.projector_functions
    fwd_body, back_body = model._view_batch_bodies()
    args = model._view_batch_args()
    num_pixels = int(model.full_index_count())
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    out = {}
    for label, body, cols in (("forward", fwd_body, int(recon_shape[2])),
                              ("back", back_body, int(sinogram_shape[1]))):
        batch, per_view = projector_functions.view_batch_charge(
            body, num_pixels, cols, args, n_devices=n_dev)
        out[label] = dict(view_batch=int(batch),
                          bytes_per_view=int(per_view),
                          band_cols=int(cols))
    return out


def _ledger_record(model):
    """The modeled side, per device, read off the model's own ledger.

    Nothing here re-derives a number: the table reports what the production
    path computed, so a disagreement between this harness and production is
    impossible by construction.
    """
    ledger = model.last_memory_ledger
    if ledger is None:
        return None
    n_dev = len(ledger.devices)
    phases = [dict(name=phase.name,
                   per_device_bytes=[int(b) for b in phase.per_device])
              for phase in ledger.phases]
    dominant, top_terms, batch_in_dominant, batch_names, batch_anywhere = \
        [], [], [], [], []
    for i in range(n_dev):
        phase = ledger.dominant_phase(i)
        dominant.append(phase.name)
        top_terms.append([[name, int(value)]
                          for name, value in phase.dominant_terms(i, count=4)])
        # The batch term INSIDE the phase that sets the peak: that is the term
        # a corrected per-view cost would scale on this device.  Both
        # directions are summed in case a phase ever carries both; the names
        # are kept so the row says which it found.
        found = [(name, int(values[i])) for name, values in phase.terms
                 if name in BATCH_TERMS and values[i] > 0]
        batch_in_dominant.append(sum(v for _n, v in found))
        batch_names.append([n for n, _v in found])
        # The largest batch term anywhere in the ledger, for scale.
        largest = 0
        for candidate in ledger.phases:
            for name, values in candidate.terms:
                if name in BATCH_TERMS:
                    largest = max(largest, int(values[i]))
        batch_anywhere.append(largest)
    return dict(devices=[str(d) for d in ledger.devices],
                modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
                dominant_phase=dominant, dominant_terms=top_terms,
                dominant_batch_bytes=batch_in_dominant,
                dominant_batch_terms=batch_names,
                max_batch_bytes=batch_anywhere,
                num_pixels_full=int(ledger.num_pixels_full),
                phases=phases)


def implied_multiplier(modeled, measured, batch_bytes):
    """The multiplier on the per-view cost that would have lifted ``modeled``
    to ``measured``, or None when the peak carries no batch term.

    Holding the batch COUNT fixed, the modeled peak is (everything else) plus
    (batch term), so reaching the measured peak needs the batch term scaled by
    1 + (measured - modeled) / batch term.  See the docstring: a real
    correction also changes the batch the driver chooses, so this is a
    first-order attribution and not the final number.
    """
    if not batch_bytes:
        return None
    return 1.0 + (float(measured) - float(modeled)) / float(batch_bytes)


# ── the worker: one arm, one process ──────────────────────────────────────────
def run_arm(cfg):
    """One (shape, device count) measurement, in its own process.

    MBIRTORCH_MEMORY_CALIBRATION is set in this process's environment by the
    runner, so the mode owns the peak counter from the moment the
    reconstruction starts.
    """
    import numpy as np
    import torch

    import mbirtorch
    from mbirtorch import _memory_ledger

    spec = _shape_by_name(cfg["shape"])
    n_dev = cfg["n_dev"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = cfg.get("cpu_devices") if not cuda else None
    if not cuda and pin_devices is None:
        pin_devices = [DEVICE]

    model = _build_model(spec, pin_devices=pin_devices)
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, repeats=REPEATS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    # This is the one job class where the calibration mode is expected to be on.
    result["calibration_env_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") == "1")
    # The band the library declares, beside the one this file judged against.
    library_band = tuple(float(x) for x in _memory_ledger.CALIBRATION_BAND)
    result["library_band"] = list(library_band)
    result["band_matches_library"] = (library_band == BAND)

    result.update(_shape_check(model, spec))

    # The bodies actually bound, and the fact this whole run rests on: neither
    # may declare its own per-view cost.  A body that declared one would be
    # priced by a real measurement instead of the proxy, and the arm would be
    # measuring something other than what it claims to.
    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__
    result["forward_has_own_cost"] = hasattr(fwd_body, "_view_batch_cost")
    result["back_has_own_cost"] = hasattr(back_body, "_view_batch_cost")
    result["bodies_use_proxy"] = not (result["forward_has_own_cost"]
                                      or result["back_has_own_cost"])
    result["reference_view_charge"] = _reference_view_charge(model, n_dev)

    sino_path = _sino_path(spec["name"])
    with open(_md5_path(spec["name"])) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    result["sino_md5"] = actual_md5
    result["sino_md5_ok"] = (actual_md5 == expected_md5)
    if not result["sino_md5_ok"]:
        raise RuntimeError(f"shared sinogram checksum mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5}")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)

    def one_recon():
        np.random.seed(VCD_SEED)
        # logfile_path=None keeps the results directory free of log files, and
        # print_logs=False keeps the subprocess's output to the one result
        # line.  Nothing is lost: the calibration rows the library would print
        # are read straight off the model below.
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0,
                                   logfile_path=None, print_logs=False)
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
    passes, walls = [], []
    for _ in range(REPEATS):
        start = time.perf_counter()
        one_recon()
        walls.append(time.perf_counter() - start)
        passes.append(calibration_rows())
        health.append(sample_gpu_health())
    # Pass 1 is the reading, compilation included (see the docstring).
    result["recon_s"] = walls[0]
    result["recon_all_s"] = walls
    result["calibration"] = passes[0]
    result["calibration_all"] = passes
    if not result["calibration"]:
        # The measured side is torch.cuda.max_memory_allocated, which exists
        # only on CUDA.  Say so, so an empty column is never read as a pass.
        result["calibration_skipped_reason"] = (
            "torch.cuda.max_memory_allocated is CUDA-only and this arm ran on "
            f"{DEVICE}; the modeled column is still real, the measured column "
            "does not exist here")

    # ── arm check: the device list the model actually realized ───────────────
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
    """One sinogram per shape, checksummed, for every arm at that shape to
    read.

    Pinned to a single device so the generator cannot itself become a
    multi-device run, and run WITHOUT the calibration mode so it never resets
    the peak counter an arm is about to read.

    The phantom is the sparse-dot volume both geometries' own tests use.  It is
    also by far the leanest option at these sizes: it builds one float32 array,
    where the Shepp-Logan builder holds six volume-shaped grids at once, which
    at the largest recon shape here would be tens of gigabytes of host memory.
    """
    import numpy as np
    import torch

    import mbirtorch

    spec = _shape_by_name(cfg["shape"])
    model = _build_model(spec, pin_devices=(cfg.get("cpu_devices") or [DEVICE]))
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.gen_translation_phantom(recon_shape, "dots", None,
                                                fill_rate=0.05)
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _sino_path(spec["name"])
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(spec["name"]), "w") as handle:
        handle.write(digest + "\n")
    out = dict(cfg, path=path, sino_md5=digest, recon_shape=list(recon_shape),
               sino_bytes=int(sinogram.nbytes))
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm, set explicitly so nothing is
    inherited.

    The device pin is MBIRTORCH_NUM_DEVICES and nothing else.  Both it and the
    calibration flag are popped first and then set, so a value exported by the
    submitting shell or the job script cannot leak into an arm that did not ask
    for it -- in particular the generators, which must not own the peak
    counter.
    """
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
    """Run one configuration in a FRESH interpreter, so memory is re-measured
    per arm and never inferred from a process that has already run one."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        # An arm that runs out of device memory lands here.  That is a real
        # reading, not a harness fault -- it says the ledger let a run start
        # that could not finish -- so it is recorded as a row and the job
        # continues to the next arm.
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def _arm_cfg(spec, n):
    entry = dict(mode="arm", shape=spec["name"], geometry=spec["geometry"],
                 cell=list(spec["cell"]), n_dev=n,
                 arm_id=f'{spec["name"]}_n{n}')
    if DEVICE != "cuda":
        # SMOKE ONLY.  The environment pin is a CUDA-only mechanism -- the
        # automatic search short-circuits below two visible CUDA devices -- so
        # the CPU path pins by device LIST and the row says so.
        entry["cpu_devices"] = [DEVICE] * n
    return entry


def build_plan(shapes, counts):
    """The plan in job order: every generator first, then the arms grouped by
    shape so a truncated job still holds whole shapes."""
    plan = []
    for spec in shapes:
        entry = dict(mode="generate", shape=spec["name"],
                     geometry=spec["geometry"], cell=list(spec["cell"]),
                     n_dev=None)
        if DEVICE != "cuda":
            entry["cpu_devices"] = [DEVICE]
        plan.append(entry)
    for spec in shapes:
        for n in counts:
            plan.append(_arm_cfg(spec, n))
    return plan


def main():
    shapes, counts = selected_plan()
    plan = build_plan(shapes, counts)
    if "--dry-run" in sys.argv:
        measured = [c for c in plan if c["mode"] == "arm"]
        print(f"mg8 plan: {len(measured)} measured arms "
              f"({len(plan) - len(measured)} generators), "
              f"{VCD_ITERATIONS} iterations each, device {DEVICE}")
        total = 0.0
        for spec in shapes:
            gib = (spec["cell"][0] * spec["cell"][1] * spec["cell"][2] * 4
                   / 2 ** 30)
            total += gib
            print(f'  sinogram {spec["name"]:<9} {spec["geometry"]:>11} '
                  f'{tuple(spec["cell"])!s:>20} -> recon '
                  f'{tuple(spec["recon_shape"])!s:<18} {gib:7.2f} GiB')
        print(f"  {total:.2f} GiB of sinogram written to {RESULTS_DIR}"
              f'{"" if os.environ.get("MG8_KEEP_ARTIFACTS") == "1" else ", deleted at the end"}')
        for cfg in plan:
            if cfg["mode"] != "arm":
                continue
            print(f'  {cfg["arm_id"]:<16} {cfg["geometry"]:>11} '
                  f'{tuple(cfg["cell"])!s:>20} n={cfg["n_dev"]}')
        print("registered expectation: ratios below 1.00 are expected on the "
              "arms a projection phase dominates; they are the finding")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg8_geom_calib_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg8 memory calibration for translation and multiaxis on "
          f"{RUN_LABEL} ({DEVICE}); shapes {[s['name'] for s in shapes]}, "
          f"counts {counts} -> {out_path}", flush=True)
    rows = []
    # Rows are written as they finish, so a job that runs out of wall time
    # still yields every arm it completed.
    with open(out_path, "w") as sink:
        for cfg in plan:
            if cfg["mode"] == "generate" and \
                    os.path.exists(_md5_path(cfg["shape"])):
                continue
            label = cfg.get("arm_id", f'generate {cfg["shape"]}')
            print(f"  {label}", flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        summary = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    if os.environ.get("MG8_KEEP_ARTIFACTS", "0") != "1":
        for spec in shapes:
            for path in (_sino_path(spec["name"]), _md5_path(spec["name"])):
                if os.path.exists(path):
                    os.remove(path)
    else:
        print("MG8_KEEP_ARTIFACTS=1: the sinograms and their checksums are "
              f"left in {RESULTS_DIR}")
    print(f"\nwrote {out_path}")


def _verdict(ratio):
    return ("UNDER" if ratio < BAND[0]
            else "over" if ratio > BAND[1] else "ok")


def summarize(rows, out_path):
    print(f"\n===== mg8 modeled against measured peaks, translation and "
          f"multiaxis ({out_path}) =====")
    header = (f'{"geometry":>12}{"shape":>10}{"n":>3}{"device":>8}'
              f'{"modeled":>11}{"measured":>11}{"ratio":>8}{"band":>8}'
              f'{"implied":>9}  dominant phase')
    print(header)
    print("-" * len(header))
    # Two separate lists, deliberately.  "invalid" is a row that cannot be
    # believed; "under" is a row that is believed and reads below the floor,
    # which is what this run went looking for.
    invalid, under, summary_rows = [], [], []
    for row in rows:
        if row.get("mode") != "arm":
            continue
        if row.get("error"):
            print(f'{row["geometry"]:>12}{row["shape"]:>10}{row["n_dev"]:>3}'
                  f'   ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:74]}')
            invalid.append(f'{row.get("arm_id")}|error')
            continue
        ledger = row.get("ledger") or {}
        dominant = ledger.get("dominant_phase") or []
        batch_bytes = ledger.get("dominant_batch_bytes") or []
        entry = dict(arm_id=row.get("arm_id"), geometry=row["geometry"],
                     shape=row["shape"], cell=row["cell"], n_dev=row["n_dev"],
                     devices_ok=row.get("devices_ok"),
                     bodies_use_proxy=row.get("bodies_use_proxy"),
                     recon_shape=row.get("recon_shape"),
                     recon_shape_ok=row.get("recon_shape_ok"),
                     recon_s=row.get("recon_s"),
                     modeled_peak_bytes=ledger.get("modeled_peak_bytes"),
                     dominant_batch_bytes=batch_bytes, per_device=[])
        if not row.get("calibration"):
            modeled = ledger.get("modeled_peak_bytes") or []
            print(f'{row["geometry"]:>12}{row["shape"]:>10}{row["n_dev"]:>3}'
                  f'{"-":>8}'
                  f'{(max(modeled) / 2 ** 30 if modeled else 0):>10.2f}G'
                  f'{"n/a":>11}{"n/a":>8}{"no-cuda":>8}{"n/a":>9}  '
                  f'{dominant[0] if dominant else ""}')
            entry["calibration_skipped_reason"] = row.get(
                "calibration_skipped_reason")
        for i, cal in enumerate(row.get("calibration", [])):
            ratio = cal["ratio"]
            verdict = _verdict(ratio)
            batch = batch_bytes[i] if i < len(batch_bytes) else 0
            implied = implied_multiplier(cal["modeled_bytes"],
                                         cal["measured_bytes"], batch)
            if verdict == "UNDER":
                under.append((row, i, cal, implied))
            entry["per_device"].append(
                dict(device=cal["device"], modeled_bytes=cal["modeled_bytes"],
                     measured_bytes=cal["measured_bytes"], ratio=ratio,
                     verdict=verdict, implied_multiplier=implied,
                     batch_term_bytes=batch,
                     dominant_phase=(dominant[i] if i < len(dominant) else None)))
            print(f'{row["geometry"]:>12}{row["shape"]:>10}{row["n_dev"]:>3}'
                  f'{cal["device"]:>8}'
                  f'{cal["modeled_bytes"] / 2 ** 30:>10.2f}G'
                  f'{cal["measured_bytes"] / 2 ** 30:>10.2f}G'
                  f'{ratio:>8.3f}{verdict:>8}'
                  # Printed on under-floor rows only.  The same arithmetic on
                  # an in-band or over-band row yields a multiplier below one,
                  # which would read as an invitation to SHRINK a per-view cost
                  # that is already short; the value is kept in the row for
                  # anyone who wants it, but the table does not suggest it.
                  f'{(f"{implied:.2f}x" if (verdict == "UNDER" and implied is not None) else "-"):>9}  '
                  f'{dominant[i] if i < len(dominant) else ""}')
        # ── the checks that decide whether a row can be believed ─────────────
        if row.get("devices_ok") is False:
            print(f'    ARM CHECK FAIL: realized {row.get("realized_devices")} '
                  f'for n={row["n_dev"]}')
            invalid.append(f'{row.get("arm_id")}|devices')
        if row.get("calibration_env_ok") is False:
            print("    ARM CHECK FAIL: MBIRTORCH_MEMORY_CALIBRATION was not "
                  "set in the arm's own environment")
            invalid.append(f'{row.get("arm_id")}|calibration_env')
        if row.get("bodies_use_proxy") is False:
            print(f'    ARM CHECK FAIL: a projection body now declares its own '
                  f'per-view cost (forward '
                  f'{row.get("forward_has_own_cost")}, back '
                  f'{row.get("back_has_own_cost")}); this arm is no longer '
                  f'measuring the proxy this run is about')
            invalid.append(f'{row.get("arm_id")}|bodies')
        if row.get("recon_shape_ok") is False:
            print(f'    ARM CHECK FAIL: recon shape {row.get("recon_shape")} '
                  f'is not the registered {row.get("recon_shape_expected")}; '
                  f'a geometry default has moved')
            invalid.append(f'{row.get("arm_id")}|recon_shape')
        if row.get("band_matches_library") is False:
            print(f'    NOTE: the library now declares band '
                  f'{row.get("library_band")}, not {list(BAND)}; the verdicts '
                  f'above were judged against {list(BAND)}')
        summary_rows.append(entry)
    print("-" * len(header))
    print(f"band {BAND[0]:.2f} <= modeled/measured <= {BAND[1]:.2f} per device."
          f"  {len(under)} reading(s) below the floor -- EXPECTED here, see "
          f"the module docstring -- and {len(invalid)} row(s) that cannot be "
          f"believed.")

    # ── the deliverable: how far short, and by what multiplier ───────────────
    for row, i, cal, implied in under:
        ledger = row.get("ledger") or {}
        terms = ((ledger.get("dominant_terms") or [[]])[i]
                 if i < len(ledger.get("dominant_terms") or []) else [])
        batch = (ledger.get("dominant_batch_bytes") or [0])[i] \
            if i < len(ledger.get("dominant_batch_bytes") or []) else 0
        names = (ledger.get("dominant_batch_terms") or [[]])[i] \
            if i < len(ledger.get("dominant_batch_terms") or []) else []
        print(f"\nUNDER THE FLOOR {row.get('arm_id')} on {cal['device']}: "
              f"modeled {cal['modeled_bytes'] / 2 ** 30:.2f}G < measured "
              f"{cal['measured_bytes'] / 2 ** 30:.2f}G (ratio "
              f"{cal['ratio']:.3f}); short by "
              f"{(cal['measured_bytes'] - cal['modeled_bytes']) / 2 ** 30:.2f}G")
        print(f"  dominant phase: "
              f"{(ledger.get('dominant_phase') or [None])[i]}")
        for name, value in terms:
            print(f"    {name:<40}{value / 2 ** 30:>8.2f}G")
        if batch:
            print(f"  batch term in that phase ({', '.join(names)}): "
                  f"{batch / 2 ** 30:.2f}G")
            print(f"  per-view cost would need {implied:.2f}x to cover this "
                  f"reading (batch count held fixed)")
        else:
            print("  that phase carries NO batch term, so the shortfall is "
                  "NOT the per-view cost model: something else is missing "
                  "here, and it needs its own attribution")

    # ── the two headline numbers, per geometry ───────────────────────────────
    print("\nworst reading and required multiplier, per geometry:")
    for geometry in GEOMETRIES:
        readings = [(row, i, cal, implied) for row, i, cal, implied in under
                    if row["geometry"] == geometry]
        measured_any = [e for e in summary_rows
                        if e["geometry"] == geometry and e["per_device"]]
        if not measured_any:
            print(f"  {geometry:<12} no measured reading (no CUDA arm ran)")
            continue
        if not readings:
            best = min(d["ratio"] for e in measured_any
                       for d in e["per_device"])
            print(f"  {geometry:<12} nothing below the floor; lowest ratio "
                  f"{best:.3f}")
            continue
        worst = min(readings, key=lambda r: r[2]["ratio"])
        multipliers = [m for _r, _i, _c, m in readings if m is not None]
        print(f'  {geometry:<12} worst ratio {worst[2]["ratio"]:.3f} at '
              f'{worst[0]["arm_id"]} on {worst[2]["device"]} '
              f'({len(readings)} reading(s) below the floor)')
        if multipliers:
            print(f'  {"":<12} the per-view cost model needs up to '
                  f'{max(multipliers):.2f}x to cover every reading here '
                  f'(largest of {len(multipliers)}); the single-body ablation '
                  f'in the prerelease review measured 10x forward and 4.3x '
                  f'back on multiaxis')
        else:
            print(f'  {"":<12} no reading below the floor sits in a phase with '
                  f'a batch term, so no multiplier is implied')

    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} row(s) sampled hot: {hot}")
    return dict(rows=summary_rows, invalid=invalid,
                under=[f'{r.get("arm_id")}|{c["device"]}'
                       for r, _i, c, _m in under],
                hot=hot, band=list(BAND))


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
