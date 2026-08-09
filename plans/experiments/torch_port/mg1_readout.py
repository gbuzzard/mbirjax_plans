"""mg1 -- THE GATE READOUT: the full n = 1, 2, 4 matrix, both frameworks, both
geometries, both gate cells, with the three-region attribution instrument.

The charter is `multigpu_plan.md` §4 (mg1) under the eleven protocols of §3.
Goals 1 and 4: the readout itself, and the forward region's share of composed
wall that gates item 13.

THE ARMS (§7's count: 24 matrix + 4 auto + 12 instrumented = 40 measured arms,
plus 4 untimed generator arms).

    matrix        pinned production arms: {torch, jax} x {parallel, cone}
                  x {(512,448,384), (1024,1008,992)} x n in {1,2,4}   = 24
    instrumented  the torch twins: the SAME production configuration with the
                  region instrument attached, one per (geometry, cell, n) = 12
    auto          torch, NO pin: the automatic policy observed, one per
                  (geometry, cell)                                     =  4
    generator     one shared sinogram artifact per (geometry, cell), untimed

WHY EVERY TIMED ARM READS THE SHARED ARTIFACT (a design decision, flagged for
review).  §4 requires a cross-framework production value column built from ONE
shared sinogram (protocol 5), and §7 budgets no separate arms for it.  Both are
satisfied by making the shared artifact the INPUT of the matrix arms: per
(geometry, cell) a generator arm builds phantom -> sinogram -> .npy once, md5s
it, and every torch and jax arm at that cell reconstructs THAT array.  The
timed arms are therefore also the value arms, at every count, for free.  For
torch this is bit-identical to what kb3's timed arm built in-process (kb3's
generator is the same torch forward projection), so the n=1 torch rows stay
directly comparable to the recorded baselines; for jax it is kb3's `shared_jax`
input rather than mbirjax's own phantom, which changes values at boundary ties
and not shapes, so jax times stay comparable too.

WHAT THE n=1 ARMS MUST REPRODUCE (`multigpu_plan.md` §2, kb3 job 14975410):

    cell             torch/jax warm time     torch peak
    parallel  512    1.13                    1.93 GB
    parallel 1024    1.55                   23.22 GB
    cone      512    0.87                    2.15 GB
    cone     1024    0.99                   23.68 GB

The n=1 rows are the campaign's validity check: nothing in the n>1 columns
means anything until they land in that class.  Protocol 11 therefore runs
every n=1 arm FIRST (PHASE 0), so a truncated job still yields the check.

ARM ORDER (protocol 9, and the one place THREE requirements cannot all be
literal).  Protocol 9 asks for counts blocked-and-reversed at each cell
(1, 2, 4, 4, 2, 1); protocol 11 asks for the n=1 reproduction arms first; §4
asks for each twin ADJACENT to its uninstrumented sibling.  Adjacency and
n=1-first win, so the built order per cell is:

    PHASE 0:  generator, torch n=1, jax n=1
    PHASE 1:  twin n=1, torch n=2, twin n=2, torch n=4, twin n=4,
              auto, jax n=4, jax n=2

The torch arms therefore run 1, 1, 2, 2, 4, 4 (sibling pairs adjacent) and
the jax n>1 arms run REVERSED at the block's tail.  Linear drift is not
cancelled by symmetry in this order; it is MEASURED instead: each arm's own
warm repeats carry the within-arm spread, and the auto arm repeats the
all-device sibling at the block's far end, so the within-job drift at the
largest count is read directly from that pair.

THE INSTRUMENT (§3 protocol 10, §4).  Five regions, wrapped from the recon
loop's calling thread, never from a worker thread:

    forward_funnel   model.sparse_forward_project   (the public funnel)
    back_funnel      model.sparse_back_project      (the public funnel)
    prior            the prior fan-out, tomography_model.py:2159
    halo             _sharding.exchange_qggmrf_halos, tomography_model.py:2110
    band_reduce      _sharding.sum_band_to_owner    (SUPPLEMENTARY, nested
                                                     inside back_funnel)

For each device in the relevant placement, a start and an end event are created
and recorded inside ``with torch.cuda.device(dev)`` on that device's default
stream, immediately before and after the wrapped call.  Elapsed times are read
only after a per-device synchronize at recon end, so the instrument cannot
serialize the overlap it measures.  Every region also records the host wall of
the call (perf_counter), and wall-minus-span prices orchestration.

RECONCILIATION.  band_reduce is NESTED inside back_funnel and is excluded from
the top-level sum; the reconciliation is

    forward_funnel + back_funnel + prior + halo + remainder == composed wall

TWO STRUCTURAL FACTS, recorded rather than asserted away:
  * band_reduce is NEVER called at n=1: the trivial placement short-circuits at
    tomography_model.py:548 before the banded driver runs.  A zero at n=1 is
    the engine's shape, not an instrument failure.
  * halo's DEVICE span is ~0 at n=1: exchange_qggmrf_halos (_sharding.py:318)
    moves nothing with one shard.  Its host wall is still nonzero.

THE VALUE COLUMNS (§4).  Within-framework: each production arm against its own
framework's n=1, same cell, same job; torch's divergence is gated against jax's
at the same coordinates (the phase-4 ruler).  Cross-framework: torch against
jax at equal n from the shared artifact -- REPORTED, not gated, until mg3
supplies the eager floors.

Protocol 4 note: mg1 carries NO eager arm, by design.  Every torch arm here is
the production configuration, so the within-framework column is read against
the documented amplified envelope -- jax's own divergence at the same
coordinates -- and never against the eager floor.  The eager-to-eager gate the
protocol names is mg3's instrument, not this one's.

THE ATTRIBUTION EXPECTATION (§4).  The n=1 forward reading at parallel 1024
must be commensurate with the kb3 close-out, which attributed that cell's
composed remainder over jax (about 14.4 s) to the forward.  A forward region
far from that class says the brackets sit in the wrong place; the summary
prints that one reading on its own line.

THE GATHER CONTRACT (nt2_local_shard_check.py).  ``Shards.gather()`` ALREADY
returns numpy.  Re-detaching its result is the recorded failure that cost the
nightly's first 4-GPU trial all 32 of its n>1 rows.  Every host exit in this
file goes through ``_to_numpy``, which never re-detaches a gather.

Run:
    <torch python> mg1_readout.py            on a 4-GPU node (mg1_gautschi.sbatch)
    python mg1_readout.py --dry-run          anywhere: print the arm plan
    python mg1_readout.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the arm subprocesses
    MG1_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG1_GEOMS=parallel,cone           subset of the geometries
    MG1_CELLS=512,1024                subset of the cells (by view count)
    MG1_COUNTS=1,2,4                  subset of the device counts
    MG1_ITERATIONS=3                  VCD iterations per recon
    MG1_WARM_REPEATS=3                warm repeats after the discarded cold pass
    MG1_SKIP_JAX=1                    torch arms only
    MG1_SKIP_INSTRUMENTED=1           drop the twins
    MG1_SKIP_AUTO=1                   drop the auto arms
    MG1_SMOKE=1                       the local smoke (tiny cell, few iters)
    MG1_DEVICE=cpu                    smoke device
    MG1_SMOKE_CPU_N2=1                smoke only: a 2-device CPU arm that
                                      exercises the instrument's n>1 wiring
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
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

CELLS = [(512, 448, 384), (1024, 1008, 992)]
GEOMETRIES = ("parallel", "cone")
COUNTS = (1, 2, 4)

SMOKE = os.environ.get("MG1_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG1_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG1_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # kb3's seed, so the n=1 rows compare to its baselines
SAMPLE_ROWS = 16          # kb3's / p5k6's sample convention
# Protocol 9: one DISCARDED cold pass plus at least three warm repeats.  The
# smoke drops to two only to keep the local round trip short.
WARM_REPEATS = max(1, int(os.environ.get("MG1_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# The kb3 close-out baselines the n=1 arms must reproduce (plan §2).
BASELINE_TORCH_OVER_JAX = {("parallel", 512): 1.13, ("parallel", 1024): 1.55,
                           ("cone", 512): 0.87, ("cone", 1024): 0.99}
BASELINE_TORCH_PEAK_GB = {("parallel", 512): 1.93, ("parallel", 1024): 23.22,
                          ("cone", 512): 2.15, ("cone", 1024): 23.68}

# HARNESS CONSTANT, NOT A PLAN NUMBER.  §4 gates torch's within-framework
# divergence "against jax's at the same coordinates" without fixing a factor.
# The raw ratio is always reported; this only decides the printed verdict.
# 3.0 by review ruling: the phase-4 matrix put torch's own-n divergence at
# 1.92x of jax's at the 1024 cell (2.98e-5 vs 1.55e-5), so 2.0 would flap on
# a ratio the record already shows; 3.0 still catches the order-one class the
# gate exists for by three orders of magnitude.
VALUE_RULER_FACTOR = 3.0
# Below this the two frameworks are both inside the documented benign
# divergence class (own-count divergences read 8.7e-5 to 3.0e-4 in the mg1
# gate table) and the ratio above is meaningless, so no verdict is printed.
# Raised from 1e-6 per the increment-1 checkpoint ruling: the ruler tripped
# at ratio 3.19 with BOTH divergences in the e-5 class (parallel 1024 n=2),
# a benign flag the old floor sat two decades below.
VALUE_RULER_FLOOR = 1e-4

# The five instrumented regions.  band_reduce is NESTED inside back_funnel and
# is excluded from the reconciliation sum.
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
NESTED_REGIONS = ("band_reduce",)
# Structurally absent at n=1 (see the module docstring); a zero there is the
# engine's shape, not an instrument failure.
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
# Device spans are structurally ~0 at n=1; the host wall is still nonzero.
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
# The instrument stops recording new pairs past this, and says so, rather than
# growing without bound on a fine partition.
MAX_EVENT_PAIRS = int(os.environ.get("MG1_MAX_EVENT_PAIRS", "400000"))
# Reconciliation tolerance: the region walls must account for the composed wall
# to within this fraction (the remainder is what is left over, and must be >=0
# up to the same slack).
RECONCILE_SLACK = 0.02

# The throttle rule (protocol 11 / nightly_plan.md §10.5): sw_power_cap at a
# normal temperature is the boost governor -- recorded and KEPT.  A row is
# marked for re-run only when it is hot AND its clock is depressed.
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

RESULTS_DIR = os.environ.get(
    "MG1_RESULTS",
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
        keep = _strict_subset("MG1_CELLS", {c[0] for c in CELLS}, int)
        cells = [c for c in CELLS if c[0] in keep]
    chosen = _strict_subset("MG1_GEOMS", set(GEOMETRIES))
    # Normalized to the DECLARED order: the arm order is load-bearing here
    # (protocol 9's reversal design), so it must not depend on set iteration.
    geometries = [g for g in GEOMETRIES if g in chosen]
    if SMOKE and not os.environ.get("MG1_COUNTS", "").strip():
        # The env pin is a CUDA-only mechanism (the policy short-circuits at
        # `visible < 2`), so a pinned n>1 arm on CPU would silently measure
        # n=1.  The smoke therefore runs n=1 plus the dedicated CPU virtual
        # 2-device arm, which pins by device LIST and says so.
        counts = [1]
    else:
        counts = _strict_subset("MG1_COUNTS", set(COUNTS), int)
        counts = [n for n in COUNTS if n in counts]
    return geometries, cells, counts


# ── staged-artifact mechanics (protocol 5) ────────────────────────────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg1_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id):
    """The arm's strided row sample; ``arm_id`` already names the geometry,
    the cell, the arm class, the framework and the count."""
    return os.path.join(RESULTS_DIR, f"_mg1_sample_{arm_id}.npy")


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, chunked: at the 1024 cells the artifact is a
    multi-GB array and a corrupt read is a recorded Lustre failure mode."""
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
    nt2 shard check's recorded failure class is re-detaching that result -- so
    a gather is never followed by ``.detach()``."""
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
    """One weighting formula, one dtype, every arm and both frameworks
    (kb3's rule)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


# ── the GPU health sample (protocol 11) ───────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks (SM + memory), temps (core + HBM), and active throttle
    reasons, via nvidia-smi.  ``[]`` when nvidia-smi is unavailable.  The
    fields mirror the nightly's own sample so the two are comparable."""
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
    its own: the re-run rule needs a depressed clock too."""
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── THE INSTRUMENT (protocol 10) ──────────────────────────────────────────────
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    recon loop's calling thread.

    CUDA path (the cluster path): for each device in the region's placement,
    a start and an end event are CREATED AND RECORDED inside
    ``with torch.cuda.device(dev)``, on that device's default stream, so the
    launch context is the device's own (the ks1 mechanism arriving through the
    instrument -- a worker thread's current device is 0, and an event recorded
    there measures nothing).  The end event is recorded AFTER the call returns,
    so it queues behind everything the call enqueued and async spillover past
    the return is covered.  Elapsed times are read only in :meth:`finish`,
    after a per-device synchronize, so the instrument never serializes the
    overlap it measures.

    CPU path (the local smoke ONLY, clearly labelled): perf_counter walls stand
    in for the event spans behind the same interface.  The CUDA event path is
    cluster-only and cannot be smoked on a Mac.  Two smoke artifacts follow
    from virtual cpu devices sharing one name: the per-device map collapses to
    a single ``'cpu'`` key, and its span sum is therefore the host wall times
    the device count.  On CUDA the keys are ``cuda:0..n-1`` and the spans are
    the real per-device windows.
    """

    def __init__(self, torch_module, cuda):
        self.torch = torch_module
        self.cuda = cuda
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        # region -> {device string -> [(start, end), ...]}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.devices_seen = {region: [] for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False
        self.backend = "cuda_events" if cuda else \
            "perf_counter (CPU smoke; the CUDA event path is cluster-only)"

    def reset(self):
        """Drop everything accumulated so far, keeping the devices seen.

        Called after the DISCARDED cold pass (protocol 7): a cold pass pays n
        per-device Triton compiles inside the very regions being measured, and
        folding that into the per-pass share would put the region walls orders
        of magnitude above the warm composed wall.  After this call the region
        walls cover the warm repeats and nothing else, which is what the
        reconciliation reads against.
        """
        self.calls = {region: 0 for region in REGIONS}
        self.host_wall = {region: 0.0 for region in REGIONS}
        self._pairs = {region: {} for region in REGIONS}
        self._cpu_spans = {region: {} for region in REGIONS}
        self.pair_count = 0
        self.cap_hit = False

    # -- the bracket -----------------------------------------------------------
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
        """Return ``func`` bracketed for ``region``.  ``resolve_devices`` is
        called per invocation (with the call's own args) so a mid-run
        re-settle of the layout cannot leave the instrument on a stale list."""
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

    # -- the read-out ----------------------------------------------------------
    def finish(self, devices):
        """Per-device synchronize, THEN read the spans (never inside the
        loop).  Returns the jsonl-ready region record."""
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


def attach_instrument(model, torch_module, cuda):
    """Wrap the five regions on THIS model instance.  Nothing in the mbirtorch
    package is edited: the funnels are shadowed as instance attributes, and the
    three sharding seams are shadowed as module attributes, all of which the
    engine looks up at call time.

    The prior seam is the one that is not a public name.  The engine's prior
    fan-out is ``_sharding.run_per_device(devices, prior_worker, ...)`` at
    tomography_model.py:2159, on the loop thread; the wrapper therefore
    discriminates on the worker's ``__name__``.  That is a private closure
    name, so a rename would silently empty the region -- which is exactly why
    the ``prior`` region carries a nonzero arm check at every count.
    """
    from mbirtorch import _sharding

    instrument = RegionInstrument(torch_module, cuda)

    # forward funnel: the output side is the sino placement.
    model.sparse_forward_project = instrument.wrap(
        "forward_funnel",
        lambda *a, **k: list(model.sino_placement.devices),
        model.sparse_forward_project)
    # back funnel: the output side is the recon placement.
    model.sparse_back_project = instrument.wrap(
        "back_funnel",
        lambda *a, **k: list(model.recon_placement.devices),
        model.sparse_back_project)

    # halo: a module-level function, called from the loop thread through
    # vcd_subset_updater.stage_halos (tomography_model.py:2110/2326).
    original_halos = _sharding.exchange_qggmrf_halos
    _sharding.exchange_qggmrf_halos = instrument.wrap(
        "halo",
        lambda shards, *a, **k: list(shards.placement.devices),
        original_halos)

    # band reduce: the named mg5 seam lever, called from the loop thread at
    # tomography_model.py:605 (inside the back funnel, so NESTED).  Its device
    # is the band's owner.
    original_reduce = _sharding.sum_band_to_owner
    _sharding.sum_band_to_owner = instrument.wrap(
        "band_reduce",
        lambda partials, owner, *a, **k: [owner],
        original_reduce)

    # prior: the single loop-thread wrap point, discriminated by worker name.
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


# ── the torch side ────────────────────────────────────────────────────────────
def _build_torch_model(geometry, cell, pin_devices=None):
    """The model.  ``pin_devices`` is a device LIST for the smoke's CPU paths
    only; on CUDA nothing is configured here, because protocol 1 pins through
    MBIRTORCH_NUM_DEVICES and an explicit configure_devices call would take the
    explicit branch and skip the preflight."""
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
    """kb3's realized-view-batch check, extended PER DEVICE (protocol 3: the
    batch budget divides by the count while the kernel cost models do not all
    follow, so the batch is not invariant in n).  Computed at the full-pixel-set
    inputs against the formula of the body EXPECTED to be bound."""
    import mbirtorch

    pf = model.projector_functions
    args = model._view_batch_args()
    recon_shape = tuple(model.get_params("recon_shape"))
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    num_pixels = int(mbirtorch.gen_full_indices(recon_shape).shape[0])
    budget = pf._transient_budget_bytes()
    n_dev = model.sino_placement.n_devices
    # Under sharding the forward's band is the slice-owner's whole shard and
    # the back's column count is the (unsharded) detector-row count.
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
    record["view_batch_cols"] = cols
    return record, ok


def observe_view_batches(model):
    """The REALIZED view batch per direction per device, as the drivers
    actually chose it during the run.  ``_effective_view_batch`` is shadowed on
    the projector instance and the direction/device are recovered from the body
    IDENTITY, so this records what was bound rather than what was computed."""
    pf = model.projector_functions
    original = pf._effective_view_batch
    fwd_ids = {id(b): i for i, b in enumerate(pf._fwd_body_per_dev)}
    back_ids = {id(b): i for i, b in enumerate(pf._back_body_per_dev)}
    seen = {}

    def wrapped(body, num_pixels, band_cols, args):
        value = original(body, num_pixels, band_cols, args)
        if id(body) in fwd_ids:
            key = f"fwd_dev{fwd_ids[id(body)]}"
        elif id(body) in back_ids:
            key = f"back_dev{back_ids[id(body)]}"
        else:
            key = "unattributed"
        bucket = seen.setdefault(key, {})
        bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = wrapped
    return seen


def torch_worker(cfg):
    """One torch arm: cold pass discarded, then WARM_REPEATS warm repeats.

    ORDERING NOTE, load-bearing.  Every projector-dependent check runs AFTER
    the cold pass.  The automatic branch settles the layout inside the first
    ``recon`` call, and a settle that changes the count calls
    ``_install_device_layout`` -> ``create_projectors`` (tomography_model.py:842),
    which REPLACES ``model.projector_functions``.  A view-batch reading taken
    before that would describe a one-device projector set under an n-device
    label, and an observer installed before it would be discarded with the old
    instance.  The instrument itself is immune: it shadows instance and module
    attributes the engine resolves at call time.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    arm_class, n_dev = cfg["arm_class"], cfg.get("n_dev")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    pin_devices = None
    if not cuda and smoke_cpu_devices:
        pin_devices = list(smoke_cpu_devices)
    elif not cuda and arm_class != "auto":
        pin_devices = [DEVICE]
    model = _build_torch_model(geometry, cell, pin_devices=pin_devices)

    # ── the auto arm's two arm checks (§4): nothing pinned it, and nothing
    # called configure_devices -- under the one-bit rule any call is explicit
    # and would silently disable the automatic path being observed.
    configure_calls = []
    original_configure = model.configure_devices

    def counted_configure(*args, **kwargs):
        configure_calls.append(traceback.format_stack(limit=4))
        return original_configure(*args, **kwargs)
    model.configure_devices = counted_configure

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda and
                                 arm_class != "auto" else
                                 ("none (auto arm)" if arm_class == "auto"
                                  else "configure_devices(devices=[...]) "
                                       "-- CPU smoke only")),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"))
    # Protocol 6: the calibration mode owns max_memory_allocated, so it must be
    # absent everywhere in mg1.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    if arm_class == "auto":
        result["auto_env_unpinned_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") is None)

    expect_kernels = (cuda, cuda)     # production: kernels on in both directions
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)

    # ── the shared sinogram artifact (protocol 5), md5-verified ───────────────
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

    # The instrument is attached BEFORE the cold pass -- it shadows instance
    # and module attributes the engine resolves at call time, so a mid-run
    # settle cannot lose it -- and reset after, so its walls cover the warm
    # repeats alone.
    instrument = detach = None
    if arm_class == "instrumented":
        instrument, detach = attach_instrument(model, torch, cuda)

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

    # ── the cold pass, DISCARDED from the warm statistics (protocol 7) ────────
    # The auto arm takes its cold pass at verbose 2, which is where the
    # selection loop logs its per-candidate rejections; the warm repeats then
    # run at verbose 0 so the per-iteration memory-stat syncs cannot taint the
    # wall this arm contributes as the protocol-9 all-device repeat.
    if arm_class == "auto":
        model.set_params(verbose=2, no_warning=True)
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    if arm_class == "auto":
        # The rejections come from the model's own attribute, which _settle
        # fills whatever the verbosity is; the verbose-2 lines are read from
        # the model's ALWAYS-ON in-memory log buffer.  A handler of our own
        # would not survive: setup_logger closes and removes every existing
        # handler at the top of each run (parameter_handler.py:69).
        result["auto_rejections"] = [
            [int(count), str(why)]
            for count, why in getattr(model, "device_choice_rejections", [])]
        buffer = getattr(model, "log_buffer", None)
        text = buffer.getvalue() if buffer is not None else ""
        result["auto_rejection_log"] = [line.strip() for line in
                                        text.splitlines() if "rejected" in line]
        result["auto_device_report"] = [line.strip() for line in
                                        text.splitlines()
                                        if "Reconstruction devices" in line]
        result["auto_layout_is_automatic"] = bool(
            getattr(model, "device_layout_is_automatic", False))
        result["auto_torch_device"] = str(model.torch_device)
        model.set_params(verbose=0, no_warning=True)
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the projector-dependent checks, now that the layout has SETTLED ──────
    # (see the ordering note in this function's docstring)
    fwd_hook, back_hook = model._view_batch_bodies()
    fwd_name = getattr(fwd_hook, "__name__", str(fwd_hook))
    back_name = getattr(back_hook, "__name__", str(back_hook))
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_kernel_selected="triton" in fwd_name,
                  back_kernel_selected="triton" in back_name,
                  expected_bodies=list(expect_kernels))
    result["bodies_ok"] = ((result["fwd_kernel_selected"],
                            result["back_kernel_selected"]) == expect_kernels)
    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok
    # Installed only now, so the observer sits on the projector instance the
    # settled layout built, and so the static probe above cannot pollute it.
    observed_batches = observe_view_batches(model)

    # ── the warm repeats ─────────────────────────────────────────────────────
    # Protocol 7: the cold pass is DISCARDED, from the memory counters and
    # from the instrument alike -- it pays n per-device Triton compiles inside
    # the very regions being bracketed.
    if instrument is not None:
        instrument.reset()
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
    result["vcd_warm_spread"] = ((max(warm) - min(warm))
                                 / statistics.median(warm)) if warm else None

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    # ── arm check: the realized device list after the timed call ─────────────
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    if arm_class == "auto":
        expected_auto = result["visible_devices"] if cuda else None
        result["auto_chosen_count"] = len(realized)
        result["auto_choice_as_expected"] = (
            None if expected_auto in (None, 0)
            else len(realized) == expected_auto)
        result["auto_configure_calls"] = len(configure_calls)
        result["auto_configure_never_called_ok"] = (len(configure_calls) == 0)
        # An unexpected choice is a FINDING with the rejection log attached,
        # never a crash (§4).
        result["devices_ok"] = True
    else:
        result["devices_ok"] = (len(realized) == n_dev) if cuda else \
            (len(realized) == (len(pin_devices) if pin_devices else 1))

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (
            result["back_launch_keys_delta"] > 0
            and result["fwd_launch_keys_delta"] > 0)

    # ── arm check: the realized view batches, as the warm passes chose them ──
    result["view_batch_observed"] = {k: sorted(v.items())
                                     for k, v in observed_batches.items()}

    # ── the instrument read-out and its own checks ────────────────────────────
    if instrument is not None:
        readout = instrument.finish(model.sino_placement.devices)
        result.update(readout)
        detach()
        regions = readout["regions"]
        count = len(realized)
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
        top = sum(regions[r]["host_wall_s"] for r in REGIONS
                  if r not in NESTED_REGIONS)
        # The instrument was reset after the cold pass, so its walls cover the
        # WARM repeats and nothing else.
        passes = max(1, WARM_REPEATS)
        per_pass = top / passes
        composed = result["vcd_warm"]
        result["region_wall_total_s"] = top
        result["region_wall_per_pass_s"] = per_pass
        result["region_remainder_per_pass_s"] = composed - per_pass
        result["region_remainder_frac"] = ((composed - per_pass) / composed
                                           if composed else None)
        # The regions are a proper subset of the composed wall, so the sum
        # must not exceed it and the remainder must be non-negative, both to
        # within the same slack.  A sum ABOVE the composed wall would mean the
        # brackets double-count; a remainder near zero would mean nothing runs
        # outside them, which the loop's stats and glue contradict.
        result["reconcile_ok"] = (
            per_pass <= composed * (1.0 + RECONCILE_SLACK))
        # The coverage side of the same guard (review ruling): a future call
        # path that SKIPS the funnels would leave the regions tiny and the
        # remainder near 100 percent, which the upper bound alone cannot see.
        # At the gate cells the projections dominate the composed wall, so a
        # remainder above half the wall at n>1 is a warning flag (not a
        # failure -- small cells are legitimately orchestration-heavy).
        result["region_coverage_warn"] = (
            count > 1 and result["region_remainder_frac"] is not None
            and result["region_remainder_frac"] > 0.5)
        forward_share = (regions["forward_funnel"]["host_wall_s"] / passes
                         / composed) if composed else None
        result["forward_share_of_composed"] = forward_share
        back = regions["back_funnel"]["host_wall_s"]
        result["band_reduce_share_of_back"] = (
            regions["band_reduce"]["host_wall_s"] / back if back else None)

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


# ── the jax side ──────────────────────────────────────────────────────────────
def jax_worker(cfg):
    """One jax arm, same protocol: same job, same node, same shared sinogram,
    so every ratio is same-run."""
    import numpy as np

    import jax
    import mbirjax

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg.get("n_dev")
    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirjax.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirjax.ConeBeamModel(cell, angles,
                                      source_detector_dist=4.0 * num_channels,
                                      source_iso_dist=2.0 * num_channels)
    model.set_params(no_warning=True, verbose=0)
    # mbirjax has no environment pin: configure_devices(int n) IS its pin, and
    # the realized list is asserted below exactly as the torch arms assert
    # theirs.
    model.configure_devices(n_dev)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))

    sino_path = _sino_path(geometry, cell)
    with open(_md5_path(geometry, cell)) as handle:
        expected_md5 = handle.read().strip()
    actual_md5 = _md5(sino_path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}: "
                           f"{actual_md5} != {expected_md5}")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)

    shard_devices = getattr(model, "shard_devices", None)
    if shard_devices is None:
        # No silent fallback: a fallback to jax.devices()[:n] would make
        # devices_ok vacuously true if mbirjax ever renames the property --
        # the vacuity class the arm checks exist to kill.
        raise RuntimeError("mbirjax model has no shard_devices property; "
                           "the realized-device assertion cannot run")
    realized = [str(d) for d in shard_devices]
    result = dict(cfg, framework="jax", version=f"jax {jax.__version__}",
                  recon_shape=list(recon_shape),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  pin_mechanism="mbirjax configure_devices(n)",
                  realized_devices=realized, realized_n_devices=len(realized),
                  devices_ok=(len(realized) == n_dev),
                  sino_md5=actual_md5, sino_md5_ok=True,
                  sinogram_checksum=float(np.sum(np.abs(sinogram),
                                                 dtype=np.float64)))

    def vcd():
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        return np.asarray(recon)

    health = [sample_gpu_health()]
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    health.append(sample_gpu_health())
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

    per_device = []
    for device in jax.devices():
        stats = device.memory_stats() or {}
        per_device.append(int(stats.get("peak_bytes_in_use", 0)))
    result["gpu_peak_per_device"] = per_device
    result["gpu_peak_bytes"] = max(per_device, default=0)
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build ONE shared sinogram per (geometry, cell): phantom -> sinogram ->
    .npy, plus its md5 sidecar (protocol 5).  Torch builds it; the choice is
    arbitrary because every arm reconstructs the same array.  Pinned to one
    device so the generator cannot itself become a multi-device run."""
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
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


def _finish(result, out, cfg):
    """The common tail: checksum, the strided row sample for the value
    columns, and the host peak."""
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
    rule).  Protocol 1: a pinned arm pins ONLY through MBIRTORCH_NUM_DEVICES,
    which keeps the model on the automatic branch where the preflight still
    runs; an explicit configure_devices call would take the explicit branch and
    get no preflight, so the two are not interchangeable."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # protocol 6
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    if cfg["framework"] == "torch":
        env["MBIRTORCH_DISABLE_TRITON"] = "0"         # production: kernels on
        if cfg["arm_class"] != "auto" and cfg.get("n_dev") and DEVICE == "cuda":
            env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg1_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg1_{cfg['arm_id']}.json")
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    python = JAX_PYTHON if cfg["framework"] == "jax" else TORCH_PYTHON
    start = time.perf_counter()
    proc = subprocess.run([python, "-u", os.path.abspath(__file__), "_worker",
                           cfg_path, out_path], env=arm_env(cfg))
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


def build_plan(geometries, cells, counts):
    """The arm plan, in JOB ORDER (see the module docstring)."""
    skip_jax = os.environ.get("MG1_SKIP_JAX", "0") == "1"
    skip_twins = os.environ.get("MG1_SKIP_INSTRUMENTED", "0") == "1"
    skip_auto = os.environ.get("MG1_SKIP_AUTO", "0") == "1"
    smoke_cpu_n2 = SMOKE and os.environ.get("MG1_SMOKE_CPU_N2", "1") == "1"
    phase0, phase1 = [], []

    def arm(framework, arm_class, geometry, cell, n_dev, suffix="", **extra):
        tag = f"{arm_class}_{framework}_n{n_dev}" if n_dev else \
            f"{arm_class}_{framework}"
        return dict(framework=framework, arm_class=arm_class,
                    geometry=geometry, cell=list(cell), n_dev=n_dev,
                    arm_id=f"{geometry}_{cell[0]}_{tag}{suffix}", **extra)

    for geometry in geometries:
        for cell in cells:
            gen = arm("torch", "generator", geometry, cell, None)
            if SMOKE and DEVICE != "cuda":
                gen["cpu_devices"] = [DEVICE]
            phase0.append(gen)
            # PHASE 0: the n=1 reproduction arms, first (protocol 11).
            if 1 in counts:
                phase0.append(arm("torch", "matrix", geometry, cell, 1))
                if not skip_jax:
                    phase0.append(arm("jax", "matrix", geometry, cell, 1))
            # PHASE 1: the reversal block, each twin adjacent to its sibling.
            if 1 in counts and not skip_twins:
                phase1.append(arm("torch", "instrumented", geometry, cell, 1))
            for n in [n for n in counts if n != 1]:
                phase1.append(arm("torch", "matrix", geometry, cell, n))
                if not skip_twins:
                    phase1.append(arm("torch", "instrumented", geometry,
                                      cell, n))
            if smoke_cpu_n2:
                # SMOKE ONLY: two virtual cpu devices, so the instrument's n>1
                # wiring (band_reduce, halo, the sharded funnels) is exercised
                # without CUDA.  The env pin is CUDA-only, so this arm pins by
                # device LIST and says so.
                phase1.append(arm("torch", "instrumented", geometry, cell, 2,
                                  suffix="_smokecpu",
                                  cpu_devices=[DEVICE, DEVICE]))
            if not skip_auto:
                phase1.append(arm("torch", "auto", geometry, cell, None))
            if not skip_jax:
                for n in reversed([n for n in counts if n != 1]):
                    phase1.append(arm("jax", "matrix", geometry, cell, n))
    return phase0, phase1


# ── the summary ───────────────────────────────────────────────────────────────
def _rel_max(path_a, path_b):
    import numpy as np

    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return None
    a, b = np.load(path_a), np.load(path_b)
    if a.shape != b.shape:
        return None
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-30))


def summarize(rows, geometries, cells, counts, out_path):
    by = {}
    for row in rows:
        if row.get("error") or row.get("arm_class") == "generator":
            continue
        by[(row["geometry"], row["cell"][0], row["arm_class"],
            row["framework"], row.get("n_dev"))] = row
    summaries = []
    print(f"\n===== mg1 gate readout ({out_path}) =====")
    for geometry in geometries:
        for cell in cells:
            key = (geometry, cell[0])
            print(f"\n--- {geometry} {cell} ---")
            print(f"{'arm':>26}{'n':>3}{'cold_s':>9}{'warm_s':>9}"
                  f"{'spread':>8}{'peak_GB':>9}{'dev':>5}{'checks':>26}")
            summary = dict(geometry=geometry, cell=list(cell), rows=[])
            for row in rows:
                if row.get("geometry") != geometry or \
                        row.get("cell", [None])[0] != cell[0]:
                    continue
                if row.get("arm_class") == "generator":
                    continue
                if row.get("error"):
                    print(f"{row.get('arm_id', '?'):>26}  ERROR: "
                          f"{str(row['error'])[:100]}")
                    summary.setdefault("errors", []).append(row.get("arm_id"))
                    continue
                checks = []
                for name, flag in (("dev", row.get("devices_ok")),
                                   ("bod", row.get("bodies_ok")),
                                   ("vb", row.get("vb_ok")),
                                   ("cal", row.get("calibration_absent_ok")),
                                   ("md5", row.get("sino_md5_ok")),
                                   ("rgn", row.get("region_nonzero_ok")),
                                   ("rec", row.get("reconcile_ok")),
                                   ("auto", row.get("auto_configure_never_called_ok"))):
                    if flag is False:
                        checks.append(f"{name}:FAIL")
                spread = row.get("vcd_warm_spread")
                print(f"{row['arm_id'].split('_', 2)[-1]:>26}"
                      f"{str(row.get('n_dev') or '-'):>3}"
                      f"{row.get('vcd_cold', 0):>9.2f}"
                      f"{row.get('vcd_warm', 0):>9.2f}"
                      f"{(f'{spread:.1%}' if spread is not None else '-'):>8}"
                      f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:>9.2f}"
                      f"{row.get('realized_n_devices', '-'):>5}"
                      f"{(','.join(checks) if checks else 'ok'):>26}")

            # -- the gate table numbers ---------------------------------------
            def matrix(framework, n):
                return by.get((geometry, cell[0], "matrix", framework, n))

            for n in counts:
                torch_row, jax_row = matrix("torch", n), matrix("jax", n)
                cell_out = dict(n=n)
                if torch_row:
                    cell_out["torch_warm"] = torch_row.get("vcd_warm")
                    cell_out["torch_peak_gb"] = (
                        torch_row.get("gpu_peak_bytes", 0) / 2 ** 30)
                    cell_out["torch_peak_per_device_gb"] = [
                        p / 2 ** 30 for p in
                        torch_row.get("gpu_peak_per_device", [])]
                if jax_row:
                    cell_out["jax_warm"] = jax_row.get("vcd_warm")
                    cell_out["jax_peak_gb"] = (
                        jax_row.get("gpu_peak_bytes", 0) / 2 ** 30)
                if torch_row and jax_row and jax_row.get("vcd_warm"):
                    cell_out["torch_over_jax_time"] = (
                        torch_row["vcd_warm"] / jax_row["vcd_warm"])
                base_t, base_j = matrix("torch", 1), matrix("jax", 1)
                if torch_row and base_t and torch_row.get("vcd_warm"):
                    cell_out["torch_scaling_vs_n1"] = (
                        base_t["vcd_warm"] / torch_row["vcd_warm"])
                if jax_row and base_j and jax_row.get("vcd_warm"):
                    cell_out["jax_scaling_vs_n1"] = (
                        base_j["vcd_warm"] / jax_row["vcd_warm"])
                # value columns
                if torch_row and base_t and n != 1:
                    cell_out["torch_value_vs_own_n1"] = _rel_max(
                        torch_row["sample_path"], base_t["sample_path"])
                if jax_row and base_j and n != 1:
                    cell_out["jax_value_vs_own_n1"] = _rel_max(
                        jax_row["sample_path"], base_j["sample_path"])
                tv, jv = (cell_out.get("torch_value_vs_own_n1"),
                          cell_out.get("jax_value_vs_own_n1"))
                if tv is not None and jv is not None:
                    cell_out["value_ruler_ratio"] = tv / max(jv, 1e-30)
                    cell_out["value_ruler_pass"] = (
                        True if max(tv, jv) < VALUE_RULER_FLOOR
                        else tv <= jv * VALUE_RULER_FACTOR)
                if torch_row and jax_row:
                    # REPORTED, not gated (§4): the shared-artifact column.
                    cell_out["shared_value_torch_vs_jax"] = _rel_max(
                        torch_row["sample_path"], jax_row["sample_path"])
                summary["rows"].append(cell_out)
                bits = [f"n={n}"]
                for label, key_name, fmt in (
                        ("torch", "torch_warm", "{:.2f}s"),
                        ("jax", "jax_warm", "{:.2f}s"),
                        ("t/j", "torch_over_jax_time", "{:.2f}"),
                        ("scale_t", "torch_scaling_vs_n1", "{:.2f}x"),
                        ("scale_j", "jax_scaling_vs_n1", "{:.2f}x"),
                        ("val_t", "torch_value_vs_own_n1", "{:.2e}"),
                        ("val_j", "jax_value_vs_own_n1", "{:.2e}"),
                        ("t_vs_j", "shared_value_torch_vs_jax", "{:.2e}")):
                    value = cell_out.get(key_name)
                    if value is not None:
                        bits.append(f"{label}=" + fmt.format(value))
                if cell_out.get("value_ruler_pass") is False:
                    bits.append("VALUE RULER FAIL")
                print("   " + "  ".join(bits))

            # -- the n=1 reproduction check -----------------------------------
            base_t, base_j = matrix("torch", 1), matrix("jax", 1)
            if base_t and base_j and base_j.get("vcd_warm"):
                ratio = base_t["vcd_warm"] / base_j["vcd_warm"]
                expect = BASELINE_TORCH_OVER_JAX.get(key)
                summary["n1_torch_over_jax"] = ratio
                summary["n1_torch_over_jax_baseline"] = expect
                print(f"   n=1 reproduction: torch/jax {ratio:.2f} "
                      f"(kb3 recorded {expect})")
            if base_t:
                peak = base_t.get("gpu_peak_bytes", 0) / 2 ** 30
                summary["n1_torch_peak_gb"] = peak
                summary["n1_torch_peak_baseline_gb"] = \
                    BASELINE_TORCH_PEAK_GB.get(key)
                print(f"   n=1 reproduction: torch peak {peak:.2f} GB "
                      f"(kb3 recorded {BASELINE_TORCH_PEAK_GB.get(key)} GB)")

            # -- the twin / sibling spread and the three-region attribution ---
            for n in counts:
                twin = by.get((geometry, cell[0], "instrumented", "torch", n))
                sibling = matrix("torch", n)
                if not twin or twin.get("error"):
                    continue
                entry = dict(n=n)
                if sibling and sibling.get("vcd_warm"):
                    entry["twin_over_sibling"] = (twin["vcd_warm"]
                                                  / sibling["vcd_warm"])
                    band = max(sibling.get("vcd_warm_spread") or 0.0,
                               twin.get("vcd_warm_spread") or 0.0)
                    entry["sibling_spread"] = band
                    entry["twin_within_spread_ok"] = (
                        abs(entry["twin_over_sibling"] - 1.0) <= band + 1e-9)
                for region in REGIONS:
                    block = (twin.get("regions") or {}).get(region)
                    if block:
                        entry[f"{region}_wall_per_pass_s"] = (
                            block["host_wall_s"] / max(1, WARM_REPEATS))
                        entry[f"{region}_device_span_max_ms"] = \
                            block["device_span_max_ms"]
                        entry[f"{region}_calls"] = block["calls"]
                entry["forward_share_of_composed"] = \
                    twin.get("forward_share_of_composed")
                entry["remainder_frac"] = twin.get("region_remainder_frac")
                entry["band_reduce_share_of_back"] = \
                    twin.get("band_reduce_share_of_back")
                summary.setdefault("attribution", []).append(entry)
                if geometry == "parallel" and cell[0] == 1024 and n == 1:
                    # §4's placement check for the brackets themselves.
                    summary["kb3_forward_attribution_reading_s"] = \
                        entry.get("forward_funnel_wall_per_pass_s")
                    print(f"   §4 CHECK: parallel 1024 n=1 forward region = "
                          f"{entry.get('forward_funnel_wall_per_pass_s', 0):.2f}s "
                          f"per pass; kb3 attributed that cell's ~14.4 s "
                          f"composed remainder over jax to the forward, so a "
                          f"reading far from that class says the brackets sit "
                          f"in the wrong place")
                print(f"   attribution n={n}: "
                      f"fwd={entry.get('forward_funnel_wall_per_pass_s', 0):.3f}s "
                      f"back={entry.get('back_funnel_wall_per_pass_s', 0):.3f}s "
                      f"prior={entry.get('prior_wall_per_pass_s', 0):.3f}s "
                      f"halo={entry.get('halo_wall_per_pass_s', 0):.3f}s "
                      f"reduce={entry.get('band_reduce_wall_per_pass_s', 0):.3f}s "
                      f"(nested) remainder="
                      f"{(entry.get('remainder_frac') or 0):.1%} "
                      f"fwd_share={(entry.get('forward_share_of_composed') or 0):.1%}"
                      + ("  TWIN OUT OF SPREAD"
                         if entry.get("twin_within_spread_ok") is False else ""))

            # -- the auto arm's finding ---------------------------------------
            auto = by.get((geometry, cell[0], "auto", "torch", None))
            if auto and not auto.get("error"):
                summary["auto"] = dict(
                    chosen=auto.get("auto_chosen_count"),
                    as_expected=auto.get("auto_choice_as_expected"),
                    rejections=auto.get("auto_rejections"),
                    configure_never_called=auto.get(
                        "auto_configure_never_called_ok"),
                    env_unpinned=auto.get("auto_env_unpinned_ok"),
                    layout_is_automatic=auto.get("auto_layout_is_automatic"))
                pinned4 = matrix("torch", max(counts))
                if pinned4 and pinned4.get("vcd_warm") and auto.get("vcd_warm"):
                    summary["auto_vs_pinned_spread"] = abs(
                        auto["vcd_warm"] - pinned4["vcd_warm"]) \
                        / pinned4["vcd_warm"]
                print(f"   auto: chose {summary['auto']['chosen']} device(s), "
                      f"as_expected={summary['auto']['as_expected']}, "
                      f"rejections={summary['auto']['rejections']}, "
                      f"within-job spread vs pinned n={max(counts)}: "
                      f"{summary.get('auto_vs_pinned_spread')}")
                if summary["auto"]["as_expected"] is False:
                    print("   AUTO FINDING: the policy did not choose every "
                          "visible device; the rejection log is on the row.")
            summaries.append(summary)

    # -- the throttle rule (protocol 11) ---------------------------------------
    best_clock = {}
    for row in rows:
        for gpu in row.get("gpu_health") or []:
            if gpu.get("sm_mhz"):
                best_clock[gpu["index"]] = max(best_clock.get(gpu["index"], 0),
                                               gpu["sm_mhz"])
    rerun = []
    for row in rows:
        if row.get("error"):
            continue
        depressed = any(
            gpu.get("sm_mhz") and best_clock.get(gpu["index"])
            and gpu["sm_mhz"] < CLOCK_DEPRESSED_FRAC * best_clock[gpu["index"]]
            for gpu in row.get("gpu_health") or [])
        if row.get("gpu_hot") and depressed:
            rerun.append(row.get("arm_id"))
    print(f"\nthrottle rule: sw_power_cap at normal temperature is recorded "
          f"and KEPT; {len(rerun)} row(s) hot AND clock-depressed -> re-run: "
          f"{rerun}")
    return summaries, rerun


def main():
    geometries, cells, counts = selected_plan()
    phase0, phase1 = build_plan(geometries, cells, counts)
    if "--dry-run" in sys.argv:
        print(f"mg1 plan: {len(phase0)} phase-0 arms, {len(phase1)} phase-1 "
              f"arms")
        for cfg in phase0 + phase1:
            print(f"  {cfg['arm_id']:<44} {cfg['framework']:>5} "
                  f"n={cfg['n_dev']}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg1_readout_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg1 gate readout on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, cells {[c[0] for c in cells]}, counts {counts} "
          f"-> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY (protocol 11): a truncated job still yields its
    # n=1 validity check, which is why phase 0 runs first.
    with open(out_path, "w") as sink:
        for phase, plan in (("phase0", phase0), ("phase1", phase1)):
            for cfg in plan:
                print(f"  [{phase}] {cfg['arm_id']}", flush=True)
                row = run_one(dict(cfg, phase=phase))
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        summaries, rerun = summarize(rows, geometries, cells, counts, out_path)
        for summary in summaries:
            sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.write(json.dumps(dict(thermal_rerun=rerun)) + "\n")
        sink.flush()
    # The shared artifacts are KEPT (review ruling): the forward kernel's
    # atomics make a regenerated artifact non-identical at the e-7 class, so
    # the md5s recorded in the rows are only re-verifiable against these
    # exact files.  Scratch is purge-eligible; provenance beats tidiness.
    print(f"\nwrote {out_path}")


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        if cfg["arm_class"] == "generator":
            row = generator_worker(cfg)
        elif cfg["framework"] == "jax":
            row = jax_worker(cfg)
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
