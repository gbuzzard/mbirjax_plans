"""mg9 -- the one measurement the forward-remedy memo says to buy first.

WHAT THIS FILE IS FOR.  The forward projection's per-device GPU time does not
fall when GPUs are added: cone 1024 reads 32.18, 30.61, 30.48 s at one, two and
four devices, and parallel 1024 reads 28.87 and 28.75 s at one and two.  Two
mechanisms could produce that, and the campaign has not measured which one does.
Either the devices are WAITING -- on the band copies, or on the single Python
thread that issues them -- or they are BUSY the whole time, paying a per-launch
cost that does not shrink when each device's share of the work shrinks.  The
remedies are different and one of them is expensive, so the decision memo
(`forward_remedy_memo.md` §6) rules: measure first.

This harness is that measurement.  It changes nothing in mbirtorch; every
instrument here is a wrapper installed on the live objects at run time.

TERMS OF ART, each defined once, here.
    arm         one subprocess run at a fixed geometry and a fixed device
                count.  Four arms: parallel 1024 at 1, 2 and 4 devices, and
                cone 1024 at 2 devices.  (The parallel four-device point has
                never been measured; the memo's §6 follow-ups say to add it to
                whichever sweep runs next, and this is that sweep.)
    bracket     mg5's existing per-device CUDA event pair around the WHOLE
                forward projection call.  Unchanged here, so mg9's spans are
                directly comparable with the memo's numbers.
    busy time   NEW.  The sum of per-call CUDA event pairs around each
                individual forward projection BODY call on one device, plus
                the count of those calls.  Body = the geometry's per-view-batch
                function, the thing that launches the projection kernel.
    the gap     bracket minus busy: time inside the bracket that no body call
                covers.  That is time the device spent not computing -- waiting
                on a band copy, or waiting for the loop thread to issue the
                next call, or sitting under the driver's assembly work.
    broadcast   `_sharding.broadcast_band_to_views`, the fan-out that copies
                each slice band from its owner to every view-owner.  NEW here:
                host wall around each call, and device-side event pairs around
                each individual copy.

THE READING THIS PRODUCES, and what each outcome means (memo §6).  The harness
prints the numbers and states both signatures; it prints NO verdict, because the
attribution is analysis, not a harness output (the mg4 and mg5 precedent).
    * If the busy sum stays close to the bracket AND stays flat as devices are
      added, the devices are computing the whole time and a per-launch cost
      dominates -- the Option A4 direction (coalesce the per-owner walk).
    * If the gap or the broadcast wall dominates and grows with the device
      count, the devices are waiting and serialization dominates -- the Option
      A2 direction (issue each copy from the thread that consumes it), then A3
      (overlap a band's copy with the previous band's compute).

WHY THIS FILE IS SELF-CONTAINED, and mg5 is not.  mg5 imports mg1_readout for
the region instrument, deliberately, so its numbers cannot drift from mg1's.
mg9 copies that instrument instead, for two reasons.  The staged copy of
mg1_readout on the cluster is not guaranteed to be the current one, and this
job's whole value is that its bracket is the SAME bracket -- a stale import is
exactly the failure that would destroy that.  And mg1_readout's
`observe_view_batches` carries the collapsed-key defect described below, which
mg9 must not inherit.  The copied region instrument below is byte-for-byte
mg1's `RegionInstrument` and its five region definitions; a reviewer should
diff them.

THE COLLAPSED-KEY DEFECT, which this file must not repeat (memo §2.2).  The
per-device forward bodies in `Projectors._fwd_body_per_dev` are built by
`maybe_compile(fwd_body, use_compile, instance_key=i)` (projectors.py:262), and
`maybe_compile` returns the function UNCHANGED when it carries
`_mbirtorch_no_compile` -- which both Triton forward bodies do.  Every entry of
that list is therefore the SAME OBJECT.  mg1's observer keys its records by
`id(body)`, so the map collapses to one entry and every device's calls land
under the last device's name.  mg9 never keys anything by object identity: it
replaces each list element BY POSITION with a closure that carries its device
index explicitly, and every reading is attributed through that index.  The same
positional key is used to fix the realized-view-batch observation as a
by-product, so this row carries a correct per-device batch where mg7's did not.

WHERE THE INSTRUMENTS ATTACH, and why the order is load-bearing.  The region
instrument shadows instance and module attributes that the engine resolves at
call time, so it is attached BEFORE the discarded cold pass and reset after it
(mg5's order, unchanged).  The two NEW instruments are different: the body
wrappers live INSIDE the projector object, and a device-count settle during the
first reconstruction calls `create_projectors()` and would throw them away.
They are therefore installed AFTER the cold pass, at exactly the point where mg5
installs its realized-batch observer, and the arm re-verifies before and after
every timed reconstruction that the projector object and the wrapped list are
still the ones the driver will call.

FOUR CORRECTNESS TRAPS, each of which has already cost this campaign a reading.
    1. Object identity.  See the collapsed-key defect above.  Position only.
    2. Event mechanics.  A worker thread's current CUDA device is 0, so an
       event recorded there without a device context measures the wrong device
       (or nothing).  Every event here is created and recorded inside
       `with torch.cuda.device(dev)`, in the thread that issues the work, so it
       lands on the stream that carries the kernels.  Nothing synchronizes
       inside the hot path; `elapsed_time` is read only after every device has
       been synchronized at the end of a timed reconstruction.
    3. Attribute pass-through.  `Projectors._effective_view_batch` reads
       `_view_batch_cost` OFF THE BODY to choose the view batch
       (projectors.py:338).  A wrapper that hides that attribute would silently
       switch the kernel bodies onto the torch batching rule, change the
       realized view batch, and make every span here incomparable with mg5's.
       The wrappers use `functools.wraps` (which copies the wrapped function's
       `__dict__`, and therefore that attribute) and then ASSERT that the
       attribute survived.
    4. A silent no-op instrument.  After the first timed reconstruction of each
       arm the harness asserts that every active device's body-call counter is
       nonzero and that the broadcast wrapper fired at least once at two or
       more devices, and aborts the arm loudly otherwise.  An instrument that
       measures nothing must fail, not emit plausible empty rows.

WHICH STREAM THE BROADCAST COPIES USE, determined from the code rather than
guessed.  `broadcast_band_to_views` (_sharding.py:252) copies through
`move_shard`, which for the direct path is `x.to(target)` -- a cross-device
tensor copy.  Torch performs a cross-device copy on the SOURCE device's current
stream and inserts a two-way event barrier so the destination's current stream
waits for it; no copy work is enqueued on the destination's stream.  Every
destination in one broadcast call copies from the SAME source tensor, so all of
a call's copies are serialized on that one stream.  The device-side event pairs
are therefore recorded on the SOURCE device, around each individual copy, which
times each destination's copy unambiguously.  Events on the destination's
stream would have measured the barrier wait and not the copy, and the memo's
own warning applies: a pair of events on a stream the copies do not use reads
as ~0 and would silently mislead the ruling.  The row records which
measurement was taken in `copy_measurement`, and the harness prints a warning
if the device-side copy time reads ~0 at two or more devices, since that is
what a wrong stream would look like.
    The determination holds for the DIRECT device-to-device path.  When
`dev2dev_safe` is False the primitive routes through host memory instead, and
the source-stream bracket then covers only the device-to-host half; the row
records `dev2dev_safe` and downgrades `copy_measurement` accordingly.

ARM CONSTRUCTION is mg5's, unchanged, so the spans line up: the same cell
(1024, 1008, 992), the same phantom and shared md5-verified sinogram per
geometry, the same weights formula, the same seed, three VCD iterations, three
timed reconstructions after one discarded cold pass, and the SHIPPED forward
view chunk (128) -- which is what mg5's chunk-128 anchor arm ran.  This file
never moves a chunk constant; it reads both and asserts the forward one is the
shipped 128.

ARM ORDER: the one-device reference first (a truncated job still yields it),
then two, then four, then the cone arm.

Run:
    <torch python> mg9_fwd_instrument.py        on a 4-GPU node (mg9_gautschi.sbatch)
    python mg9_fwd_instrument.py --dry-run      anywhere: print the arm plan
    MG9_SMOKE=1 python mg9_fwd_instrument.py    the local CPU smoke
    python mg9_fwd_instrument.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas):
    P0_TORCH_PYTHON=<python>       interpreter for the arm subprocesses
    MG9_RESULTS=<dir>              where the jsonl and the artifacts go
    MG9_ARMS=parallel1,parallel2,parallel4,cone2   subset of the arms
    MG9_ITERATIONS=3               VCD iterations per reconstruction
    MG9_WARM_REPEATS=3             timed reconstructions after the cold pass
    MG9_MAX_EVENT_PAIRS=400000     per-reconstruction event budget
    MG9_KEEP_ARTIFACTS=1           keep the shared sinograms after the run
    MG9_SMOKE=1                    the local CPU smoke (tiny cell, few iters)
    MG9_DEVICE=cpu                 smoke device
"""

import functools
import json
import hashlib
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

# The memo's cell, and nothing else.
CELL = (1024, 1008, 992)
# The four arms, in job order: the one-device reference first, then the counts
# that the flatness is measured across, then the cone arm.  Each entry is
# (geometry, device count); the token in MG9_ARMS is geometry + count.
ARMS = (("parallel", 1), ("parallel", 2), ("parallel", 4), ("cone", 2))

SMOKE = os.environ.get("MG9_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG9_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG9_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # mg1's / mg5's seed, so the arms stay comparable
WARM_REPEATS = max(1, int(os.environ.get("MG9_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# The shipped forward view chunk.  mg5's anchor arm pinned this value, which is
# also the shipped one, so "pin to 128" and "leave it alone" are the same
# configuration.  mg9 leaves it alone and asserts the constant is still 128; an
# arm whose chunk had drifted would produce spans that are not mg5's.
SHIPPED_CHUNK = 128

# What the memo recorded for the forward's per-device span, from the mg5 rows.
# The mg9 arms must reproduce these; a large disagreement means the node or the
# tree moved and the whole reading is suspect.  (mg5 reports both a host wall
# and a device-event span for the forward region and the memo does not say
# which column these came from, so the summary prints mg9's readings of BOTH
# beside them and lets the reviewer match.)
MEMO_FORWARD_SPAN_S = {("cone", 1): 32.18, ("cone", 2): 30.61, ("cone", 4): 30.48,
                       ("parallel", 1): 28.87, ("parallel", 2): 28.75}
MEMO_COMPOSED_S = {("cone", 1): 61.57, ("cone", 2): 67.23,
                   ("parallel", 1): 40.00, ("parallel", 2): 39.40}

# Reconciliation tolerance, mg1's constant: the bracketed regions are a proper
# subset of the composed wall.
RECONCILE_SLACK = 0.02

RESULTS_DIR = os.environ.get(
    "MG9_RESULTS", os.path.dirname(os.path.abspath(__file__)))
RUN_LABEL = platform.node().split(".")[0]

# Per geometry: the kernel module carrying the forward view chunk constant that
# this file READS (and never sets), and the module's back twin.
SPEC = {
    "parallel": dict(kernel_module="mbirtorch.triton_parallel",
                     fwd_chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                     back_chunk_const="PARALLEL_BACK_VIEW_CHUNK"),
    "cone": dict(kernel_module="mbirtorch.triton_cone",
                 fwd_chunk_const="CONE_FWD_VIEW_CHUNK",
                 back_chunk_const="CONE_BACK_VIEW_CHUNK"),
}

# ── the region definitions and the GPU-health machinery, COPIED from ──────────
# ── mg1_readout.py so this file stands alone (see the module docstring) ────────

# The five instrumented regions.  band_reduce is NESTED inside back_funnel and
# is excluded from the reconciliation sum.
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
NESTED_REGIONS = ("band_reduce",)
# Structurally absent at one device; a zero there is the engine's shape, not an
# instrument failure.
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
# Device spans are structurally ~0 at one device; the host wall is still nonzero.
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
# The instruments stop recording new pairs past this, and say so, rather than
# growing without bound.  The budget is per reconstruction: everything is
# drained and reset after each timed reconstruction.
MAX_EVENT_PAIRS = int(os.environ.get("MG9_MAX_EVENT_PAIRS", "400000"))

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


def selected_arms():
    """The arms to run, narrowed by MG9_ARMS.  Tokens are geometry + count."""
    tokens = {f"{g}{n}": (g, n) for g, n in ARMS}
    raw = os.environ.get("MG9_ARMS", "").strip()
    if not raw:
        chosen = list(ARMS)
    else:
        chosen = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in tokens:
                raise ValueError(f"MG9_ARMS: {token!r} is not one of "
                                 f"{sorted(tokens)}")
            chosen.append(tokens[token])
        if not chosen:
            raise ValueError(f"MG9_ARMS: no valid tokens in {raw!r}")
        chosen = [a for a in ARMS if a in chosen]        # declared order
    if SMOKE and not os.environ.get("MG9_ARMS", "").strip():
        # The device-count pin is a CUDA-only mechanism, so a pinned n>1 arm on
        # CPU would silently measure one device.  The smoke runs the one-device
        # arm plus a two-VIRTUAL-device CPU arm that pins by device LIST and
        # says so, which is what exercises the broadcast wrapper and the n>1
        # asserts without a GPU.
        chosen = [("parallel", 1), ("parallel", 2), ("cone", 2)]
    return chosen


def cell_for(_geometry):
    return SMOKE_CELL if SMOKE else CELL


# ── staged-artifact mechanics (mg5's md5 discipline) ──────────────────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg9_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _md5(path, chunk=8 << 20):
    """md5 of a staged file, chunked: at the 1024 cell the artifact is a
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
    recorded failure class is re-detaching that result -- so a gather is never
    followed by ``.detach()``."""
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
    """One weighting formula, one dtype, every arm (mg1's rule)."""
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)


def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    """Per-GPU clocks, temperatures and active throttle reasons, via
    nvidia-smi.  ``[]`` when nvidia-smi is unavailable."""
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
    """Hot by TEMPERATURE alone.  Not a re-run verdict on its own: the re-run
    rule needs a depressed clock too."""
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── THE EXISTING BRACKET: mg1's region instrument, copied verbatim ────────────
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    reconstruction loop's calling thread.  Copied from mg1_readout.py without
    change so mg9's forward bracket IS mg5's forward bracket.

    CUDA path (the cluster path): for each device in the region's placement, a
    start and an end event are CREATED AND RECORDED inside
    ``with torch.cuda.device(dev)``, on that device's current stream, so the
    launch context is the device's own -- a worker thread's current device is 0,
    and an event recorded there measures nothing.  The end event is recorded
    AFTER the call returns, so it queues behind everything the call enqueued.
    Elapsed times are read only in :meth:`finish`, after a per-device
    synchronize, so the instrument never serializes the overlap it measures.

    CPU path (the local smoke ONLY, clearly labelled): perf_counter walls stand
    in for the event spans behind the same interface.  Two smoke artifacts
    follow from virtual cpu devices sharing one name: the per-device map
    collapses to a single ``'cpu'`` key, and its span sum is the host wall times
    the device count.
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
        """Drop everything accumulated so far, keeping the devices seen."""
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
        called per invocation (with the call's own args) so a mid-run re-settle
        of the layout cannot leave the instrument on a stale list."""
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
        """Per-device synchronize, THEN read the spans (never inside the loop).
        Returns the jsonl-ready region record."""
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
    """Wrap the five regions on THIS model instance (mg1's ``attach_instrument``,
    copied).  Nothing in the mbirtorch package is edited: the funnels are
    shadowed as instance attributes and the sharding seams as module attributes,
    all of which the engine looks up at call time."""
    from mbirtorch import _sharding

    instrument = RegionInstrument(torch_module, cuda)

    # forward funnel: the output side is the sino placement.  THIS is mg9's
    # bracket -- the span the memo's 28.87 / 30.61 numbers came from.
    model.sparse_forward_project = instrument.wrap(
        "forward_funnel",
        lambda *a, **k: list(model.sino_placement.devices),
        model.sparse_forward_project)
    # back funnel: the output side is the recon placement.
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

    # prior: the single loop-thread wrap point, discriminated by worker name.
    # That is a private closure name, which is why the prior region carries a
    # nonzero arm check at every count.
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
    comparing against a tensor's device.

    This is not a formality.  A model that resolved its device automatically
    holds ``torch.device('cuda')`` with NO index, and at one device the
    placement is built straight from that lead device (a one-device pin leaves
    the count unchanged, so the layout is never reinstalled with an indexed
    list).  A tensor's ``.device`` always carries an index, so comparing the two
    directly would report every single call as landing on the wrong device.
    """
    if not cuda or getattr(device, "type", None) != "cuda":
        return device
    if device.index is None:
        return torch_module.device("cuda", torch_module.cuda.current_device())
    return device


# ── THE NEW INSTRUMENT 1: busy time, per device, per body call ────────────────
class BusyProbe:
    """Times each individual forward projection BODY call, in buckets keyed by
    DEVICE INDEX -- never by object identity, because every entry of
    ``_fwd_body_per_dev`` is the same object (the collapsed-key defect).

    What one bucket holds, per device, per reconstruction: the summed device
    time of the per-call event pairs (the busy sum), the number of calls, and
    the summed host time of those same calls (what the issuing thread spent
    getting the work onto the device).

    Threading.  The body calls arrive on the per-device worker threads of
    ``run_per_device``, which runs at most one thread per device index at a
    time and waits for all of them before the next fan-out, so each bucket has
    a single writer at any moment and needs no lock.  The per-device event
    budget is fixed at construction for the same reason -- a shared counter
    would need one.

    Events are created and recorded inside ``with torch.cuda.device(dev)``,
    which is what puts them on the stream that carries this device's kernels;
    the body itself is called OUTSIDE that context, so it runs in exactly the
    device context it runs in without the instrument.

    What this costs the measurement.  Two device-context switches and two event
    records per body call, on the order of twenty microseconds, against a call
    that at these cells takes tens of milliseconds -- under a tenth of a
    percent, and paid identically at every device count, so it cannot bias the
    comparison across counts.  An event record puts a marker on the stream and
    no work, so the enclosing bracket reads the same as it does without this
    probe.
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
        # Positive witness for the positional key: the first tensor argument of
        # a forward body call is the band, which lives on the device that
        # position is supposed to name.  A nonzero count here means the
        # position-to-device map is wrong and every per-device number is
        # misattributed.
        self.device_mismatch = [0] * n_dev

    def wrap(self, body, dev_index, device):
        """Return ``body`` bracketed, carrying its device index EXPLICITLY.

        ``functools.wraps`` copies the wrapped function's ``__dict__``, which is
        where a kernel body keeps ``_view_batch_cost`` and
        ``_mbirtorch_no_compile``; the caller asserts that both survived, since
        losing the first would silently change the realized view batch and make
        this arm incomparable with mg5's.
        """
        torch_module, cuda = self.torch, self.cuda
        device = _concrete_device(torch_module, device, cuda)

        @functools.wraps(body)
        def wrapped(*args, **kwargs):
            if args and torch_module.is_tensor(args[0]) \
                    and args[0].device != device:
                self.device_mismatch[dev_index] += 1
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

        wrapped._mg9_dev_index = dev_index
        wrapped._mg9_wrapped_body = body
        return wrapped

    def drain(self, devices):
        """Read the elapsed times and reset, ONCE PER TIMED RECONSTRUCTION.

        Every device is synchronized first, so no ``elapsed_time`` is read on an
        event the stream has not reached.  The reconstruction already
        synchronized; this repeat costs nothing and makes the requirement local
        to the read.
        """
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        busy_ms, calls, host_s, mismatch = [], [], [], []
        for i in range(self.n_dev):
            if self.cuda:
                busy_ms.append(float(sum(s.elapsed_time(e)
                                         for s, e in self._pairs[i])))
            else:
                busy_ms.append(float(self._cpu_ms[i]))
            calls.append(int(self.calls[i]))
            host_s.append(float(self.host_s[i]))
            mismatch.append(int(self.device_mismatch[i]))
        record = dict(busy_ms_per_device=busy_ms,
                      busy_calls_per_device=calls,
                      busy_host_s_per_device=host_s,
                      busy_device_mismatch_per_device=mismatch,
                      busy_cap_hit=any(self.cap_hit))
        self._pairs = [[] for _ in range(self.n_dev)]
        self._cpu_ms = [0.0] * self.n_dev
        self.calls = [0] * self.n_dev
        self.host_s = [0.0] * self.n_dev
        self.device_mismatch = [0] * self.n_dev
        self.cap_hit = [False] * self.n_dev
        return record


# ── THE NEW INSTRUMENT 2: the band broadcast and its copies ───────────────────
class BroadcastProbe:
    """Times ``_sharding.broadcast_band_to_views`` -- host wall around each
    call -- and, separately, each individual copy that call issues.

    The copies are timed with device event pairs recorded on the SOURCE
    device's current stream, which is the stream torch enqueues a cross-device
    copy on (see the module docstring for the determination and for what makes
    the destination's stream the wrong place to look).  All of one call's
    copies come from the same source tensor, so they are serialized on that one
    stream and an event pair around each individual copy attributes device time
    to each destination.

    The original ``broadcast_band_to_views`` is CALLED, not reimplemented: the
    per-copy bracket is installed on ``move_shard``, the primitive the original
    resolves as a module global at call time.  Nothing about the fan-out's
    order, its arguments, or its results changes.  A thread-local flag keeps the
    bracket confined to copies issued from inside a broadcast, so the other
    users of ``move_shard`` (the halo exchange, the band reduce, the view
    parameter placement) run untouched and untimed.
    """

    def __init__(self, torch_module, cuda, max_pairs):
        self.torch = torch_module
        self.cuda = cuda
        self.max_pairs = max_pairs
        self.calls = 0
        self.host_s = 0.0
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        # (source name, destination name, start event, end event)
        self._pairs = []
        self.copy_measurement = None      # set by attach_forward_probes

    def drain(self, devices):
        """Read the copy spans and reset, once per timed reconstruction (see
        :meth:`BusyProbe.drain` for why the synchronize is repeated here)."""
        if self.cuda:
            for device in devices:
                self.torch.cuda.synchronize(device)
        by_src, by_dst, total = {}, {}, 0.0
        for src, dst, start, end in self._pairs:
            span = float(start.elapsed_time(end))
            by_src[src] = by_src.get(src, 0.0) + span
            by_dst[dst] = by_dst.get(dst, 0.0) + span
            total += span
        record = dict(broadcast_calls=int(self.calls),
                      broadcast_host_wall_s=float(self.host_s),
                      copy_count=int(self.copy_count),
                      copy_noop_count=int(self.copy_noop_count),
                      copy_bytes=int(self.copy_bytes),
                      copy_device_ms_total=total,
                      copy_device_ms_by_src=by_src,
                      copy_device_ms_by_dst=by_dst,
                      copy_cap_hit=bool(self.cap_hit),
                      copy_measurement=self.copy_measurement)
        self.calls = 0
        self.host_s = 0.0
        self.copy_count = 0
        self.copy_noop_count = 0
        self.copy_bytes = 0
        self.cap_hit = False
        self._pairs = []
        return record


def attach_forward_probes(model, torch_module, cuda, max_pairs):
    """Install the two new instruments and return
    ``(busy, broadcast, verify, detach)``.

    Call this AFTER the discarded cold pass.  The body wrappers live inside the
    projector object, and a device-count settle during the first reconstruction
    rebuilds that object (``_install_device_layout`` -> ``create_projectors``),
    which would throw them away.  ``verify()`` re-checks, before and after every
    timed reconstruction, that the projector object and the wrapped list are
    still the ones the driver will call.
    """
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
        # Trap 3: the driver chooses the view batch by reading this attribute
        # OFF THE BODY, so a wrapper that lost it would silently move the
        # kernel onto the torch batching rule.  functools.wraps copies the
        # function's __dict__; this asserts that it did.
        if getattr(body, "_view_batch_cost", None) is not \
                getattr(wrapper, "_view_batch_cost", None):
            raise RuntimeError(
                "the body wrapper did not carry _view_batch_cost through; the "
                "realized view batch would change and this arm would not be "
                "comparable with mg5's")
        wrappers.append(wrapper)
    # Mutated IN PLACE rather than rebound, so any other reference to the list
    # sees the wrappers too.
    for index, wrapper in enumerate(wrappers):
        bodies[index] = wrapper

    # The realized view batch, observed PER DEVICE by the positional key -- the
    # reading mg7's row could not carry.  ``_effective_view_batch`` is called
    # from the per-device worker threads as well as the loop thread, so this one
    # takes a lock; it runs once per view-range call, not once per body call.
    observed = {}
    observed_lock = threading.Lock()
    original_effective = pf._effective_view_batch

    def effective_view_batch(body, num_pixels, band_cols, args):
        value = original_effective(body, num_pixels, band_cols, args)
        index = getattr(body, "_mg9_dev_index", None)
        # The back bodies pass through here too.  They are NOT wrapped, and the
        # back list has the same collapsed identity, so their device cannot be
        # recovered -- which the key says out loud instead of guessing.
        key = (f"fwd_dev{index}" if index is not None
               else "back_body_device_not_recoverable")
        with observed_lock:
            bucket = observed.setdefault(key, {})
            bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = effective_view_batch

    # -- the broadcast and its copies ------------------------------------------
    broadcast = BroadcastProbe(torch_module, cuda, max_pairs)
    if not cuda:
        broadcast.copy_measurement = (
            "host_wall_only: the CPU smoke has no CUDA events, and its virtual "
            "devices make every copy a no-op")
    elif getattr(model, "dev2dev_safe", True):
        broadcast.copy_measurement = (
            "device_events_on_source_stream: move_shard's direct path is "
            "band.to(dst), and torch runs a cross-device copy on the SOURCE "
            "device's current stream (the destination's stream only waits on a "
            "barrier), so each copy is bracketed on the source device")
    else:
        broadcast.copy_measurement = (
            "host_wall_only: dev2dev_safe is False, so move_shard routes "
            "through host memory and a source-stream bracket would cover only "
            "the device-to-host half of each copy; the device numbers below "
            "are partial and must not be read as copy time")

    state = threading.local()
    original_move = _sharding.move_shard
    original_broadcast = _sharding.broadcast_band_to_views

    def move_shard(x, target, dev2dev_safe=True):
        if not getattr(state, "in_broadcast", False):
            return original_move(x, target, dev2dev_safe=dev2dev_safe)
        timed = cuda and len(broadcast._pairs) < broadcast.max_pairs
        start = end = None
        source = x.device
        if timed:
            with torch_module.cuda.device(source):
                start = torch_module.cuda.Event(enable_timing=True)
                start.record()
        elif cuda:
            broadcast.cap_hit = True
        out = original_move(x, target, dev2dev_safe=dev2dev_safe)
        if timed:
            with torch_module.cuda.device(source):
                end = torch_module.cuda.Event(enable_timing=True)
                end.record()
        broadcast.copy_count += 1
        # A copy to the tensor's own device returns the tensor itself: the band's
        # owner is one of the destinations, so one copy per fan-out is free.
        # Identity is the exact test and needs no device comparison.  A free copy
        # is counted but kept OUT of the spans and the bytes, so `copy_in` means
        # what actually landed on a device.
        noop = out is x
        if noop:
            broadcast.copy_noop_count += 1
        else:
            broadcast.copy_bytes += int(x.numel()) * int(x.element_size())
        if start is not None and not noop:
            broadcast._pairs.append((str(source), str(target), start, end))
        return out

    def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
        state.in_broadcast = True
        host0 = time.perf_counter()
        try:
            return original_broadcast(band, view_owners,
                                      dev2dev_safe=dev2dev_safe)
        finally:
            state.in_broadcast = False
            broadcast.host_s += time.perf_counter() - host0
            broadcast.calls += 1

    _sharding.move_shard = move_shard
    _sharding.broadcast_band_to_views = broadcast_band_to_views

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
            move_shard_wrapped=(_sharding.move_shard is move_shard))

    def detach():
        for index, body in enumerate(originals):
            bodies[index] = body
        pf._effective_view_batch = original_effective
        _sharding.move_shard = original_move
        _sharding.broadcast_band_to_views = original_broadcast

    return busy, broadcast, verify, detach, observed


# ── the torch side (mg5's model builder and checks, unchanged) ────────────────
def _build_torch_model(geometry, cell, pin_devices=None):
    """The model.  ``pin_devices`` is a device LIST for the smoke's CPU paths
    only; on CUDA nothing is configured here, because an arm pins through
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


def _view_batch_static(model, expect_kernels):
    """The realized view batch per direction per device at the full-pixel-set
    inputs, against the formula of the body EXPECTED to be bound (mg1's static
    probe, copied).  Run BEFORE the probes are installed, so the probe's own
    observer cannot see this traffic."""
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
    record["view_batch_cols"] = cols
    return record, ok


def _per_device_body_names(model):
    """The bodies actually bound, per direction per device."""
    pf = model.projector_functions
    return ([getattr(b, "__name__", str(b)) for b in pf._fwd_body_per_dev],
            [getattr(b, "__name__", str(b)) for b in pf._back_body_per_dev])


def _merge_regions(total, part):
    """Fold one reconstruction's region record into the running total, so the
    accumulated fields are exactly what mg5 reports (mg5 accumulates over the
    warm repeats and divides by the count; mg9 drains per reconstruction and
    sums, which gives the same numbers plus the per-reconstruction detail)."""
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


def torch_worker(cfg):
    """One arm: one discarded cold pass, then WARM_REPEATS timed
    reconstructions with all three instruments live.

    ORDERING NOTE, load-bearing.  Every projector-dependent step runs AFTER the
    cold pass: the automatic branch settles the device layout inside the first
    ``recon`` call, and a settle that changes the count calls
    ``_install_device_layout`` -> ``create_projectors``, which REPLACES
    ``model.projector_functions``.  The region instrument is immune (it shadows
    attributes the engine resolves at call time); the body wrappers are not,
    which is why they go in here and are re-verified around every timed pass.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg.get("n_dev")
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
    # The calibration mode owns max_memory_allocated, so it must be absent here.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))
    if cuda:
        result["pin_env_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") == str(n_dev))
    # This job runs the SHIPPED configuration and moves no constant.  mg5's
    # anchor arm pinned the forward chunk to this same value, so the arms are
    # the same configuration; if the shipped constant has drifted they are not.
    result["shipped_chunk_is_the_anchor_ok"] = (
        shipped_fwd_chunk == SHIPPED_CHUNK)

    expect_kernels = (cuda, cuda)
    result["expected_bodies_kernel"] = list(expect_kernels)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)

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
    # every timed pass (see the module docstring).
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

    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    health.append(sample_gpu_health())

    # ── the projector-dependent checks, now that the layout has SETTLED ──────
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
    # The recorded witness for the collapsed-key defect: at n>1 every entry of
    # the forward body list is the SAME object, which is why nothing here is
    # keyed by identity.  Recorded rather than asserted -- if a later tree makes
    # the bodies distinct, this row says so instead of failing.
    pf = model.projector_functions
    result["fwd_bodies_distinct_objects"] = (
        len({id(b) for b in pf._fwd_body_per_dev}) == len(pf._fwd_body_per_dev))

    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    result["dev2dev_safe"] = bool(getattr(model, "dev2dev_safe", True))
    result["fwd_chunk_after"] = int(getattr(kernel_module,
                                            spec["fwd_chunk_const"]))
    result["back_chunk_after"] = int(getattr(kernel_module,
                                             spec["back_chunk_const"]))
    result["chunks_unchanged_ok"] = (
        result["fwd_chunk_after"] == shipped_fwd_chunk
        and result["back_chunk_after"] == shipped_back_chunk)

    # ── the NEW instruments, installed on the settled projector ──────────────
    busy, broadcast, verify, detach_probes, observed = attach_forward_probes(
        model, torch, cuda, MAX_EVENT_PAIRS)
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
        record.update(broadcast.drain(devices))
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
        per_recon.append(record)
        health.append(sample_gpu_health())

        # ── the silent-no-op assert (trap 4) ─────────────────────────────────
        if repeat == 0:
            missing = [name for name, calls
                       in zip(device_names, record["busy_calls_per_device"])
                       if calls <= 0]
            if missing:
                raise RuntimeError(
                    f"the busy instrument recorded NO forward body calls on "
                    f"{missing} in the first timed reconstruction.  The "
                    f"wrappers are not on the path the driver takes (a "
                    f"projector rebuild, or a driver that no longer reaches "
                    f"the bodies through _fwd_body_per_dev), and every "
                    f"per-device number in this arm would be empty.  "
                    f"verify={record['probe_verify']}")
            if len(devices) > 1 and record["broadcast_calls"] <= 0:
                raise RuntimeError(
                    "the broadcast wrapper never fired at "
                    f"{len(devices)} devices.  The band fan-out is either no "
                    "longer routed through _sharding.broadcast_band_to_views "
                    "or the shadow was replaced; the copy columns of this arm "
                    "would be empty.  (At one device there is no fan-out and "
                    "this check does not apply.)")
            if not all(record["probe_verify"].values()):
                raise RuntimeError(
                    f"the forward probes left the driver's path during the "
                    f"first timed reconstruction: {record['probe_verify']}")

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
    detach_probes()
    detach_regions()

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev) if cuda else \
        (len(realized) == (len(pin_devices) if pin_devices else 1))

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

    # ── THE DERIVED SUMMARY: one entry per device, medians over the passes ───
    result.update(summarize_arm(result))

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def summarize_arm(result):
    """Per device: the bracket, the busy sum, the call count, and the gap
    between them -- each the median over the timed reconstructions -- plus the
    broadcast's per-reconstruction totals.  Medians, because a single
    reconstruction can carry a scheduling artifact and the campaign reads
    medians everywhere else."""
    names = result["device_names"]
    passes = result["per_recon"]
    if not passes:
        return dict(per_device=[], broadcast=None)

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
        bracket_med, busy_med = median(bracket), median(busy)
        per_device.append(dict(
            device=name,
            bracket_span_s=bracket_med,
            busy_sum_s=busy_med,
            busy_calls=median(calls),
            busy_host_s=median(host),
            gap_s=(bracket_med - busy_med),
            busy_frac_of_bracket=(busy_med / bracket_med if bracket_med else None),
            back_bracket_span_s=median(back),
            copy_device_in_s=median(copy_in),
            copy_device_out_s=median(copy_out),
            device_mismatch=sum(p["busy_device_mismatch_per_device"][index]
                                for p in passes)))

    bytes_med = median([p["copy_bytes"] for p in passes])
    dev_ms_med = median([p["copy_device_ms_total"] for p in passes])
    broadcast = dict(
        calls_per_recon=median([p["broadcast_calls"] for p in passes]),
        host_wall_s_per_recon=median([p["broadcast_host_wall_s"]
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
    broadcast["copy_device_plausible"] = (
        None if len(names) < 2 or not result.get("cuda")
        else bool(dev_ms_med and dev_ms_med > 0.0))
    return dict(per_device=per_device, broadcast=broadcast)


def generator_worker(cfg):
    """Build ONE shared sinogram per geometry: phantom -> sinogram -> .npy, plus
    its md5 sidecar.  Every arm at that geometry reconstructs THAT array, so no
    arm's timing carries an input difference.  Pinned to one device so the
    generator cannot itself become a multi-device run."""
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
    """The common tail: the reconstruction checksum and the host peak.

    ONE DELIBERATE DIFFERENCE FROM mg5, so a reader diffing them is not left
    guessing.  mg5 also saves a strided row sample of every arm, because it has
    a value column: its arms differ in what they compute (a view chunk moves, a
    projection body is swapped), so their outputs have to be diffed against each
    other.  mg9's arms differ only in how many devices they use.  Every wrapper
    here passes its arguments through and returns its result unchanged, so there
    is nothing to diff, and a sample would be about 63 MB per arm of scratch
    that nothing reads.  The checksum below is kept as the cheap witness that an
    arm reconstructed something and not garbage.
    """
    import numpy as np

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result["recon_checksum"] = float(np.sum(np.abs(out), dtype=np.float64))
    result["recon_shape_out"] = list(out.shape)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


# ── the runner (mg5's subprocess pattern, unchanged) ──────────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.

    An arm pins ONLY through MBIRTORCH_NUM_DEVICES, which keeps the model on the
    automatic branch where the preflight still runs; an explicit
    configure_devices call would take the explicit branch and get no preflight,
    so the two are not interchangeable and mixing them would make the arms
    incomparable with mg5's.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg9_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg9_{cfg['arm_id']}.json")
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
    """The arm plan, in job order: the generators, then the arms as declared
    (the one-device reference first, so a truncated job still yields it)."""
    plan = []
    for geometry in sorted({g for g, _n in arms}):
        cell = cell_for(geometry)
        gen = dict(framework="torch", arm_class="generator", geometry=geometry,
                   cell=list(cell), n_dev=None,
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if SMOKE and DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        plan.append(gen)
    measured = []
    for geometry, n_dev in arms:
        cell = cell_for(geometry)
        arm = dict(framework="torch", arm_class="instrument", geometry=geometry,
                   cell=list(cell), n_dev=n_dev,
                   arm_id=f"{geometry}_{cell[0]}_n{n_dev}_instrument")
        if SMOKE and DEVICE != "cuda":
            # SMOKE ONLY: virtual cpu devices, so the n>1 wiring (the band
            # broadcast, the per-device fan-out, the asserts) is exercised
            # without CUDA.  The env pin is CUDA-only, so this pins by device
            # LIST and says so on the row.
            arm["cpu_devices"] = [DEVICE] * n_dev
        measured.append(arm)
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


SIGNATURE_TEXT = (
    "  SIGNATURE, per-launch cost (the Option A4 direction): the busy sum "
    "stays CLOSE TO the\n"
    "    bracket and stays FLAT as devices are added.  The devices compute the "
    "whole time, and\n"
    "    what does not shrink is the cost of each launch.\n"
    "  SIGNATURE, serialization (the Option A2 then A3 direction): the gap "
    "(bracket minus busy)\n"
    "    or the broadcast wall DOMINATES and GROWS with the device count.  The "
    "devices are\n"
    "    waiting -- on the band copies, or on the single thread that issues "
    "them.\n"
    "  No verdict is printed here: the attribution is analysis, not a harness "
    "output.")


def print_arm_table(row):
    """The per-device table for one arm, in plain English."""
    geometry, n_dev = row.get("geometry"), row.get("n_dev")
    print(f"\n--- {geometry} {row['cell'][0]}, {n_dev} device(s): the forward, "
          f"per device (median of {len(row.get('per_recon') or [])} timed "
          f"reconstructions) ---")
    print(f"   {'device':>10}{'bracket_s':>11}{'busy_s':>9}{'busy/brk':>10}"
          f"{'gap_s':>8}{'calls':>8}{'issue_s':>9}{'copy_in_s':>11}"
          f"{'copy_out_s':>12}")
    for entry in row.get("per_device") or []:
        print(f"   {entry['device']:>10}"
              f"{_fmt(entry['bracket_span_s'], '11.2f')}"
              f"{_fmt(entry['busy_sum_s'], '9.2f')}"
              f"{_fmt(entry['busy_frac_of_bracket'], '10.2f')}"
              f"{_fmt(entry['gap_s'], '8.2f')}"
              f"{_fmt(entry['busy_calls'], '8.0f')}"
              f"{_fmt(entry['busy_host_s'], '9.2f')}"
              f"{_fmt(entry['copy_device_in_s'], '11.2f')}"
              f"{_fmt(entry['copy_device_out_s'], '12.2f')}")
    print("   bracket_s = the whole forward projection region, this device's "
          "event span.")
    print("   busy_s    = the same device's time inside the per-call brackets "
          "around the")
    print("               projection body: what it spent computing.")
    print("   gap_s     = bracket minus busy: time inside the region this "
          "device spent NOT")
    print("               computing -- waiting on a copy, or on the thread "
          "that issues the work.")
    print("   issue_s   = host time the issuing thread spent inside those "
          "same calls.")
    print("   copy_in / copy_out = band-broadcast copy time landing on / "
          "leaving this device,")
    print("               timed on the SOURCE device's stream (see "
          "copy_measurement).")
    bcast = row.get("broadcast") or {}
    if bcast:
        rate = bcast.get("copy_gb_per_s")
        rate_text = f" = {rate:.2f} GB/s" if rate else ""
        print(f"   broadcast per reconstruction: "
              f"{_fmt(bcast.get('calls_per_recon'), '.0f')} calls, host wall "
              f"{_fmt(bcast.get('host_wall_s_per_recon'), '.2f')} s, "
              f"{_fmt(bcast.get('copy_count_per_recon'), '.0f')} copies "
              f"({_fmt(bcast.get('copy_noop_count_per_recon'), '.0f')} to the "
              f"band's own device, which are free),")
        print(f"      device time on the source stream "
              f"{_fmt(bcast.get('copy_device_s_per_recon'), '.2f')} s for "
              f"{_fmt((bcast.get('copy_bytes_per_recon') or 0) / 1e9, '.2f')} "
              f"GB{rate_text}")
        print(f"      copy_measurement: {bcast.get('copy_measurement')}")
        if bcast.get("copy_device_plausible") is False:
            print("      WARNING: the device-side copy time reads ~0 at more "
                  "than one device.  That is")
            print("      what a bracket on a stream the copies do not use "
                  "looks like; read the host wall")
            print("      instead and re-check the stream determination before "
                  "ruling on these numbers.")
    memo = MEMO_FORWARD_SPAN_S.get((geometry, n_dev))
    if memo is not None:
        print(f"   memo's recorded forward span at this point: {memo:.2f} s.  "
              f"mg9 reads host wall "
              f"{_fmt(row.get('forward_funnel_wall_per_pass_s'), '.2f')} s and "
              f"a largest per-device event span of "
              f"{_fmt(row.get('forward_funnel_dev_span_max_per_pass_s'), '.2f')}"
              f" s.")
    if row.get("view_batch_observed_per_device"):
        print(f"   realized forward view batch, per device, keyed BY POSITION "
              f"(the reading mg7's row")
        print(f"      could not carry): "
              f"{row['view_batch_observed_per_device']}")


def summarize(rows, out_path):
    """The per-arm tables, then the one table the memo asks for: the forward's
    bracket, busy sum, gap and broadcast wall across the device counts."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    print(f"\n===== mg9 forward instrument ({out_path}) =====")

    for row in rows:
        if row.get("arm_class") == "generator":
            continue
        if row.get("error"):
            print(f"\n--- {row.get('arm_id', '?')}: ERROR "
                  f"{str(row['error'])[-400:]}")
            continue
        checks = []
        for name, flag in (("dev", row.get("devices_ok")),
                           ("pin", row.get("pin_env_ok")),
                           ("bod", row.get("bodies_ok")),
                           ("bpd", row.get("bodies_per_device_ok")),
                           ("vb", row.get("vb_ok")),
                           ("chunk", row.get("shipped_chunk_is_the_anchor_ok")),
                           ("chunk_same", row.get("chunks_unchanged_ok")),
                           ("kern", row.get("kernels_launched_ok")),
                           ("kill", row.get("kill_switch_off_ok")),
                           ("cal", row.get("calibration_absent_ok")),
                           ("md5", row.get("sino_md5_ok")),
                           ("rgn", row.get("region_nonzero_ok")),
                           ("rec", row.get("reconcile_ok"))):
            if flag is False:
                checks.append(f"{name}:FAIL")
        print_arm_table(row)
        mismatch = sum(e.get("device_mismatch", 0)
                       for e in row.get("per_device") or [])
        if mismatch:
            checks.append(f"device_key:{mismatch} MISATTRIBUTED CALLS")
        # An exhausted event budget under-reports the busy sum, which would
        # inflate the gap and push the reading toward the serialization
        # signature for no reason.  It has to be visible on the arm's line.
        if any(p.get("busy_cap_hit") for p in row.get("per_recon") or []):
            checks.append("busy_events:CAPPED (raise MG9_MAX_EVENT_PAIRS)")
        if any(p.get("copy_cap_hit") for p in row.get("per_recon") or []):
            checks.append("copy_events:CAPPED (raise MG9_MAX_EVENT_PAIRS)")
        memo_composed = MEMO_COMPOSED_S.get((row.get("geometry"),
                                             row.get("n_dev")))
        print(f"   composed reconstruction "
              f"{_fmt(row.get('vcd_warm'), '.2f')} s"
              f"{'' if memo_composed is None else f' (memo {memo_composed:.2f} s)'}"
              f" (spread "
              f"{_fmt((row.get('vcd_warm_spread') or 0) * 100, '.1f')}%), peak "
              f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:.2f} GB, checks: "
              f"{','.join(checks) if checks else 'ok'}")

    # ── THE DISCRIMINATION ───────────────────────────────────────────────────
    print("\n===== the forward across device counts: what the memo asks for "
          "=====")
    print(f"{'geometry':>10}{'n':>3}{'bracket_s':>11}{'busy_s':>9}"
          f"{'gap_s':>8}{'busy/brk':>10}{'calls':>8}{'bcast_wall_s':>14}"
          f"{'bcast_dev_s':>13}{'memo_s':>9}")
    series = []
    for row in live:
        entries = row.get("per_device") or []
        if not entries:
            continue
        bcast = row.get("broadcast") or {}
        # The per-device numbers are summarized by their MAXIMUM over devices:
        # the reconstruction waits for the slowest device, so the largest span
        # is the one that sets the wall.
        bracket = max((e["bracket_span_s"] or 0.0) for e in entries)
        busy = max((e["busy_sum_s"] or 0.0) for e in entries)
        calls = max((e["busy_calls"] or 0) for e in entries)
        entry = dict(geometry=row["geometry"], n=row["n_dev"],
                     bracket_s=bracket, busy_s=busy, gap_s=bracket - busy,
                     busy_frac=(busy / bracket if bracket else None),
                     calls=calls,
                     bcast_host_s=bcast.get("host_wall_s_per_recon"),
                     bcast_dev_s=bcast.get("copy_device_s_per_recon"),
                     memo_s=MEMO_FORWARD_SPAN_S.get((row["geometry"],
                                                     row["n_dev"])))
        series.append(entry)
        print(f"{entry['geometry']:>10}{entry['n']:>3}"
              f"{_fmt(entry['bracket_s'], '11.2f')}"
              f"{_fmt(entry['busy_s'], '9.2f')}"
              f"{_fmt(entry['gap_s'], '8.2f')}"
              f"{_fmt(entry['busy_frac'], '10.2f')}"
              f"{_fmt(entry['calls'], '8.0f')}"
              f"{_fmt(entry['bcast_host_s'], '14.2f')}"
              f"{_fmt(entry['bcast_dev_s'], '13.2f')}"
              f"{_fmt(entry['memo_s'], '9.2f')}")
    print("   Each row is the LARGEST reading over that arm's devices: the "
          "reconstruction waits")
    print("   for the slowest device, so the largest span is the one that "
          "sets the wall.")
    print("   memo_s is what the decision memo recorded for this point, from "
          "the mg5 rows; a")
    print("   bracket far from it means the node or the tree moved and the "
          "whole reading is")
    print("   suspect.  (The memo does not say whether its number is the host "
          "wall or the")
    print("   device span; each arm's block above prints both of mg9's.)")
    print(SIGNATURE_TEXT)

    backends = {r.get("event_backend") for r in live if r.get("event_backend")}
    if any(b != "cuda_events" for b in backends):
        print(f"\nNOTE: event backend {sorted(backends)}.  On the CPU path the "
              f"per-device span map collapses to a single 'cpu' key whose span "
              f"IS the host wall, and every band copy is a no-op, so the busy, "
              f"gap and copy columns price nothing there.  Those columns are "
              f"meaningful only under cuda_events.")

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
    print(f"\nthrottle rule: sw_power_cap at normal temperature is recorded and "
          f"KEPT; {len(rerun)} row(s) hot AND clock-depressed -> re-run: "
          f"{rerun}")
    return series, rerun


def main():
    arms = selected_arms()
    generators, measured = build_plan(arms)
    if "--dry-run" in sys.argv:
        print(f"mg9 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  arms {arms}, warm repeats {WARM_REPEATS}, iterations "
              f"{VCD_ITERATIONS}, device {DEVICE}, results {RESULTS_DIR}")
        for cfg in generators + measured:
            print(f"  {cfg['arm_id']:<44} n={cfg['n_dev']}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg9_fwd_instrument_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg9 forward instrument on {RUN_LABEL} ({DEVICE}); arms {arms} "
          f"-> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY: a truncated job still yields the arms it
    # finished, which is why the one-device reference runs first.
    with open(out_path, "w") as sink:
        for cfg in generators + measured:
            print(f"  [{cfg['arm_id']}]", flush=True)
            row = run_one(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        series, rerun = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=dict(series=series))) + "\n")
        sink.write(json.dumps(dict(thermal_rerun=rerun)) + "\n")
        sink.flush()
    # The shared sinograms are internal to this job -- nothing outside mg9
    # re-verifies these md5s -- and the pair is several GB of scratch.  Export
    # MG9_KEEP_ARTIFACTS=1 to keep them.
    if os.environ.get("MG9_KEEP_ARTIFACTS", "0") != "1":
        for geometry in sorted({g for g, _n in arms}):
            for path in (_sino_path(geometry, cell_for(geometry)),
                         _md5_path(geometry, cell_for(geometry))):
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
