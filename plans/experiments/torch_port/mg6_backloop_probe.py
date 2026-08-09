"""mg6 -- THE DIRECT-RECON BACK LOOP AT n>1: which sub-step is the peak?

The charter is `multigpu_findings.md` §6.3 (charter B), step one, under the
eleven protocols of `multigpu_plan.md` §3.  mg2 found the ledger over-charging
by 1.31 to 1.59 at n=4 with the direct-recon back loop named as the modeled
dominant phase at every n>1 weighted cell.  The over-charge admits two
readings and the distinction decides the remedy: a phantom charge (fix the
ledger) or a real avoidable residency (fix the loop).

THE QUESTION, made measurable.  At n>1 the banded back driver
(`TomographyModel._sparse_back_project_sharded`) walks slice-owners
SEQUENTIALLY, and each owner's band pass has two consecutive sub-steps:

    (a) the WORKERS -- every device back-projects its own views onto this
        owner's band, holding the view-loop accumulator plus its live blocks
        plus the batch transient;
    (b) the band REDUCE -- `_sharding.sum_band_to_owner` moves all n partials
        onto the owner and sums them there.

The ledger charges `back output` (2 cylinders at n>1) and `band reduce`
(n+1 or n+2 cylinders) and `back batch` into ONE phase total.  If (a) and (b)
are consecutive rather than co-live, that sum is a phantom of exactly the
species correction five already fixed one level up (the loop and the scatter).
This probe measures (a) and (b) separately, per device, per band pass.

THE THREE DECISIVE NUMBERS, and what each settles:

 1. WORKER BLOCK COUNT.  `(worker peak - worker entry) / cyl_dev`, minus the
    realized back-batch charge.  The n=1-era finding is three cylinders --
    `block = back_body(...)` then `out.add_(block)`, so the previous block is
    still alive while the kernel produces the next one.  That third cylinder
    needs at least THREE view batches in the loop.  At n>1 each device owns
    V/n views, so the batch count falls with n: the probe records the realized
    view batch and the batch count beside the measurement, because a loop that
    runs twice can hold only two cylinders however the ledger charges it.

 2. THE BAND-PASS ENTRY STEP.  `entry(pass 1) - entry(pass 0)` per device.
    The driver writes `partials = _sharding.run_per_device(...)`, and python
    evaluates the call before it rebinds the name, so the PREVIOUS band pass's
    partials -- one (P, L) array on EVERY device -- are still alive while the
    next band's kernels run.  Nothing in the ledger charges them.  Predicted
    step, in cyl_dev: +2 on device 0 (its finished own band, plus the stale
    partial) and +1 on the last device (the stale partial alone).  If instead
    the step reads +1 and 0, the stale partials are not live and there is
    nothing to fix.

 3. THE REDUCE COPY COUNT.  `(reduce peak - reduce entry) / cyl_dev` on the
    owner, which the ledger models as n+1 at two devices and n+2 above.  The
    owner enters holding its own partial, so the predicted reading is n at
    n=2 and n+1 at n=4.

THE ARMS (4).  The n=1 reproduction arm runs FIRST (protocol 11), and it is
the instrument's ruler: cone 512 at n=1 is the cell whose dp2/dp3 dominant
phase IS the direct-recon back loop (at 1.104), so the three-cylinder finding
must reproduce there or the instrument is wrong before any n>1 row is read.

    cone     (512, 448, 384)   n=1     the ruler, and the n=1 anchor
    parallel (1024,1008, 992)  n=2     mg2's 1.05-class cell
    parallel (1024,1008, 992)  n=4     mg2's 1.42-1.43 cell, the worst reading
    cone     (512, 448, 384)   n=2     the same cell as the ruler, widened

PROTOCOLS.  1: every n>1 arm pins through MBIRTORCH_NUM_DEVICES and asserts
the realized list, so the model stays on the automatic branch where the ledger
is built.  3: the realized view batch is recorded per arm, because the batch
budget divides by the device count while the kernel cost model does not, and
term 1 above is only readable beside it.  6: memory is re-measured per arm in
a fresh subprocess, and MBIRTORCH_MEMORY_CALIBRATION is ASSERTED ABSENT -- mg2
is the one job that owns that mode, and this probe owns the peak counter
itself, sub-phase by sub-phase.  10: every mark is recorded from the LOOP
thread with an explicit device argument; nothing is read inside a worker
lambda, whose current device is 0.  11: rows write incrementally to jsonl, the
n=1 arm runs first, and the GPU health sample rides on every row.

NOT A TIMING INSTRUMENT.  The sub-phase brackets synchronize every device
before each reading, which serializes the fan-out the band loop overlaps.  The
walls recorded here are therefore NOT comparable with mg1's regions and are
kept only to price the job.

Run:
    <torch python> mg6_backloop_probe.py     on a 4-GPU node (mg6_gautschi.sbatch)
    python mg6_backloop_probe.py --dry-run   anywhere: print the arm plan
    MG6_SMOKE=1 python mg6_backloop_probe.py the local CPU smoke

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list).  List values are parsed STRICTLY.
    MG6_RESULTS=<dir>              where the jsonl and the artifacts go
    MG6_ARMS=cone_512_n1,...       subset of the arm ids below
    MG6_ITERATIONS=1               vcd iterations (the direct recon runs once
                                   whatever this is; 1 keeps the arm cheap)
    MG6_SMOKE=1 / MG6_DEVICE=cpu   the local smoke
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
PARALLEL_1024 = (1024, 1008, 992)
CONE_512 = (512, 448, 384)

# (arm_id, geometry, cell, n).  The n=1 ruler first (protocol 11).
ARMS = (
    ("cone_512_n1", "cone", CONE_512, 1),
    ("parallel_1024_n2", "parallel", PARALLEL_1024, 2),
    ("parallel_1024_n4", "parallel", PARALLEL_1024, 4),
    ("cone_512_n2", "cone", CONE_512, 2),
    # The reviewer's widening: the code-reading attribution reproduces every
    # mg2 reading except cone 1024 at n=4, where it lands UNDER the measured
    # peak by about 0.5 GB -- the one cell where the liveness account is
    # incomplete, and the one direction (a charge below measured) the ledger
    # may not err in.  Costs roughly six minutes: one cold and one warm pass
    # at the 1024 cone cell, plus that cell's generator.
    ("cone_1024_n4", "cone", (1024, 1008, 992), 4),
)

SMOKE = os.environ.get("MG6_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG6_DEVICE", "cpu" if SMOKE else "cuda")

# The direct recon runs ONCE per reconstruction, before the loop, so the
# region this probe measures does not depend on the iteration count.  One
# iteration still compiles the subset shapes on the cold pass, which keeps the
# warm pass free of compile transients inside a measured region.
VCD_ITERATIONS = int(os.environ.get("MG6_ITERATIONS", "1"))
VCD_SEED = 12345                      # dp2/dp3/mg2's seed, so the rows compare

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
    "MG6_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_arms():
    """The arm plan, filtered STRICTLY (kb3's slurm comma-split lesson)."""
    arms = [(aid, g, SMOKE_CELL if SMOKE else c, n) for aid, g, c, n in ARMS]
    raw = os.environ.get("MG6_ARMS", "").strip()
    if not raw:
        return arms
    allowed = {a[0] for a in arms}
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError(f"MG6_ARMS: unknown arm {token!r}; "
                             f"known arms are {sorted(allowed)}")
        chosen.append(token)
    return [a for a in arms if a[0] in chosen]


# ── the shared sinogram per cell (generated once, md5-verified) ───────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg6_sino_{geometry}_{cell[0]}.npy")


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
    """The ONE host exit.  Shards.gather() ALREADY returns numpy; re-detaching
    its result is the recorded failure that cost the nightly's first 4-GPU
    trial all 32 of its n>1 rows (nt2_local_shard_check.py)."""
    import numpy as np
    if hasattr(x, "gather"):
        return x.gather()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _weights(sinogram):
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
    the ledger is built.  ``pin_devices`` is the smoke's CPU path only."""
    import numpy as np

    import mbirtorch

    if geometry == "parallel":
        angles = np.linspace(0, np.pi, cell[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    else:
        angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
        sdd = 4 * cell[2]
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=sdd,
                                        source_iso_dist=sdd)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    return model


# ── the instrument ────────────────────────────────────────────────────────────
class SubPhaseProbe:
    """Sub-phase memory marks for the direct-recon back region.

    Every mark records, per device, ``memory_allocated`` at entry and
    ``max_memory_allocated`` over the bracket.  The counter is reset per
    bracket, so a mark is the true allocator high-water mark while that
    sub-step ran, with the resident set it inherited included -- the dp3
    contract, at one more level of resolution.

    Protocol 10: every read here happens on the LOOP thread with an explicit
    device argument.  The per-device worker lambdas are NOT instrumented; a
    worker thread's current device is 0 and a device-less read there measures
    nothing.
    """

    def __init__(self, model, devices, cuda):
        self.model = model
        self.devices = list(devices)
        self.cuda = cuda
        self.marks = []
        self.region = None          # None | 'back loop' | 'scatter'
        self.in_direct_recon = False
        self.band_index = 0
        self.filter_method = None
        self.arm_start = time.perf_counter()

    # -- the readings ---------------------------------------------------------
    def _sync(self):
        if not self.cuda:
            return
        import torch
        for device in self.devices:
            torch.cuda.synchronize(device)

    def _allocated(self):
        if not self.cuda:
            return [0] * len(self.devices)
        import torch
        return [int(torch.cuda.memory_allocated(d)) for d in self.devices]

    def _reset(self):
        if not self.cuda:
            return
        import torch
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)

    def _peak(self):
        if not self.cuda:
            return [0] * len(self.devices)
        import torch
        return [int(torch.cuda.max_memory_allocated(d)) for d in self.devices]

    def bracket(self, label, fn, **extra):
        """Run ``fn`` inside one mark."""
        self._sync()
        entry = self._allocated()
        self._reset()
        start = time.perf_counter()
        out = fn()
        self._sync()
        mark = dict(label=label, entry_bytes=entry, peak_bytes=self._peak(),
                    wall_s=time.perf_counter() - start,
                    t_s=time.perf_counter() - self.arm_start)
        mark.update(extra)
        self.marks.append(mark)
        return out

    # -- the wrapping ---------------------------------------------------------
    def install(self):
        from mbirtorch import _sharding

        model = self.model
        probe = self

        # SCOPE (the charter's: the direct-recon phase alone).  The vcd loop
        # calls the same funnels for the hessian and for every subset step, so
        # every bracket below is gated on `in_direct_recon`.  Without the gate
        # the band marks of six back projections at three different pixel
        # counts would land in one list and the cylinder arithmetic -- which
        # is normalized by cyl(P_full, S_dev) -- would be meaningless.
        def wrap_method(name, label, region=None, opens_region=False):
            original = getattr(model, name)

            def wrapped(*args, **kwargs):
                if opens_region:
                    probe.in_direct_recon = True
                elif not probe.in_direct_recon:
                    return original(*args, **kwargs)
                previous = probe.region
                if region is not None:
                    probe.region = region
                    probe.band_index = 0
                try:
                    return probe.bracket(label, lambda: original(*args, **kwargs))
                finally:
                    probe.region = previous
                    if opens_region:
                        probe.in_direct_recon = False
            setattr(model, name, wrapped)

        # The region, then its two halves, then the banded call inside them.
        # The filter's method name is the geometry's own -- parallel's
        # fbp_recon calls fbp_filter, cone's fdk_recon calls fdk_filter -- and
        # the direct recon calls that method, not the `direct_filter` alias,
        # so the wrap follows the name the pipeline actually uses.
        wrap_method("direct_recon", "direct recon (region)", opens_region=True)
        for filter_name in ("fbp_filter", "fdk_filter"):
            if hasattr(model, filter_name):
                wrap_method(filter_name, "direct recon filter")
                self.filter_method = filter_name
                break
        wrap_method("back_project", "back projection (region)", region="scatter")
        wrap_method("sparse_back_project", "sparse back project (banded)",
                    region="back loop")

        # The band loop's two consecutive sub-steps.  `_sparse_back_project_
        # sharded` reaches both through the MODULE (`_sharding.run_per_device`,
        # `_sharding.sum_band_to_owner`), so patching the module attributes
        # catches them without touching the package's source.
        original_run = _sharding.run_per_device
        original_sum = _sharding.sum_band_to_owner

        def run_per_device(devices, worker_fn, executor=None):
            if not probe.in_direct_recon or probe.region != "back loop":
                # The forward driver, the scatter's per-device worker, and the
                # qGGMRF halo exchange all share this entry point.  Only the
                # back loop's band passes are this probe's business; the
                # scatter is bracketed as a whole by back_project above.
                return original_run(devices, worker_fn, executor=executor)
            index = probe.band_index
            probe.band_index += 1
            return probe.bracket(
                "band workers",
                lambda: original_run(devices, worker_fn, executor=executor),
                band_pass=index)

        def sum_band_to_owner(partials, owner, dev2dev_safe=True):
            if not probe.in_direct_recon or probe.region != "back loop":
                return original_sum(partials, owner, dev2dev_safe=dev2dev_safe)
            return probe.bracket(
                "band reduce",
                lambda: original_sum(partials, owner,
                                     dev2dev_safe=dev2dev_safe),
                band_pass=probe.band_index - 1, owner=str(owner),
                owner_index=(self.devices.index(owner)
                             if owner in self.devices else None))

        _sharding.run_per_device = run_per_device
        _sharding.sum_band_to_owner = sum_band_to_owner
        self._originals = (original_run, original_sum)

    def remove(self):
        from mbirtorch import _sharding
        _sharding.run_per_device, _sharding.sum_band_to_owner = self._originals


# ── the modeled side ──────────────────────────────────────────────────────────
def ledger_record(model, weights):
    """The ledger for the layout that actually ran.

    At n>1 the automatic branch already built one and left it on the model.
    At n=1 no layout is chosen, so the ledger is computed here from the same
    pure function the preflight uses -- it allocates nothing and never touches
    the peak counter this probe owns (which is why the calibration mode, whose
    job is to reset that counter, stays absent per protocol 6)."""
    from mbirtorch import _memory_ledger as ML

    ledger = model.last_memory_ledger
    if ledger is None:
        plan = ML.plan_from_model(model, model.sino_placement.devices,
                                  weights=weights)
        ledger = ML.estimate_peak_device_bytes(plan)
    out = dict(devices=[str(d) for d in ledger.devices],
               modeled_peak_bytes=[int(b) for b in ledger.per_device_peaks()],
               dominant_phase=[ledger.dominant_phase(i).name
                               for i in range(len(ledger.devices))],
               num_pixels_full=int(ledger.num_pixels_full),
               phases=[dict(name=p.name,
                            per_device_bytes=[int(b) for b in p.per_device])
                       for p in ledger.phases])
    for phase in ledger.phases:
        if phase.name == "direct recon (back loop)":
            out["back_loop_terms"] = [[name, [int(v) for v in vals]]
                                      for name, vals in phase.terms]
    return out


def realized_back_batch(model):
    """The view batch the back driver actually chooses for the direct recon's
    full-index call, and how many batches that is per device (protocol 3).
    Term 1 of the charter is only readable beside these two numbers: three
    cylinders need three view batches."""
    pf = model.projector_functions
    args = model._view_batch_args()
    _fwd_body, back_body = model._view_batch_bodies()
    num_pixels = int(model.full_index_count())
    sp, rp = model.sino_placement, model.recon_placement
    n = sp.n_devices
    if n == 1:
        band_cols = int(model.get_params("sinogram_shape")[1])
    elif getattr(model, "rows_track_slices", False):
        band_cols = rp.padded_size // n
    else:
        band_cols = int(model.get_params("sinogram_shape")[1])
    view_batch = pf._effective_view_batch(back_body, num_pixels, band_cols, args)
    _batch, bytes_per_view = pf.view_batch_charge(back_body, num_pixels,
                                                  band_cols, args)
    views_per_dev = max(n_valid for _d, _r, n_valid in sp.padded_shard_ranges())
    return dict(back_view_batch=int(view_batch),
                back_bytes_per_view=int(bytes_per_view),
                back_band_cols=int(band_cols),
                views_per_device=int(views_per_dev),
                view_batches_per_band_pass=-(-int(views_per_dev)
                                             // int(view_batch)),
                back_body=back_body.__name__)


# ── the worker: one arm, one process ──────────────────────────────────────────
def run_arm(cfg):
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg["n_dev"]
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    pin_devices = cfg.get("cpu_devices") if not cuda else None
    if not cuda and pin_devices is None:
        pin_devices = [DEVICE]

    model = _build_model(geometry, cell, pin_devices=pin_devices)
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS,
                  pin_mechanism=("MBIRTORCH_NUM_DEVICES" if cuda else
                                 "configure_devices(devices=[...]) "
                                 "-- CPU smoke only"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    # Protocol 6: mg2 is the ONE job that owns MBIRTORCH_MEMORY_CALIBRATION.
    # This probe owns the peak counter sub-phase by sub-phase, so the mode
    # must be absent or the two owners would clobber each other's resets.
    result["calibration_absent_ok"] = not os.environ.get(
        "MBIRTORCH_MEMORY_CALIBRATION")
    if not result["calibration_absent_ok"]:
        raise RuntimeError("MBIRTORCH_MEMORY_CALIBRATION is set; this probe "
                           "owns the peak counter (protocol 6)")

    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    result["num_pixels_full"] = int(model.full_index_count())

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
        raise RuntimeError(f"shared sinogram md5 mismatch at {sino_path}")
    sinogram = np.load(sino_path)
    weights = _weights(sinogram)

    def one_recon(probe=None):
        np.random.seed(VCD_SEED)
        recon, _info = model.recon(sinogram, weights=weights,
                                   max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.sino_placement.devices:
                torch.cuda.synchronize(device)
        return recon

    health = [sample_gpu_health()]
    # Cold pass, UNINSTRUMENTED: it pays every compile (per device, per shape),
    # and a compile transient inside a measured bracket would be charged to the
    # sub-step that happened to trigger it.
    start = time.perf_counter()
    one_recon()
    result["cold_s"] = time.perf_counter() - start
    health.append(sample_gpu_health())

    # The layout is settled now, so the probe can name the devices it reads.
    devices = list(model.sino_placement.devices)
    probe = SubPhaseProbe(model, devices, cuda)
    probe.install()
    try:
        start = time.perf_counter()
        one_recon()
        result["warm_s"] = time.perf_counter() - start
    finally:
        probe.remove()
    health.append(sample_gpu_health())

    result["marks"] = probe.marks
    result["filter_method"] = probe.filter_method
    # The instrument's own check, and the one the CPU smoke can make: the
    # wrappers must have fired at every site, once per band pass per owner.
    counts = {}
    for mark in probe.marks:
        counts[mark["label"]] = counts.get(mark["label"], 0) + 1
    result["mark_counts"] = counts
    result["marks_ok"] = (
        counts.get("direct recon (region)") == 1
        and counts.get("direct recon filter") == 1
        and counts.get("sparse back project (banded)") == 1
        and counts.get("band workers", 0) == (0 if n_dev == 1 else n_dev)
        and counts.get("band reduce", 0) == (0 if n_dev == 1 else n_dev))
    result["batching"] = realized_back_batch(model)
    result["ledger"] = ledger_record(model, weights)

    # ── arm check: the realized device list, after the timed call ────────────
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    result["layout_is_automatic"] = bool(
        getattr(model, "device_layout_is_automatic", False))

    # ── the derived quantities, in cylinders ────────────────────────────────
    slices_per_dev = model.recon_placement.padded_size // len(devices)
    cyl_dev = result["num_pixels_full"] * slices_per_dev * 4
    result["cyl_dev_bytes"] = int(cyl_dev)
    result["slices_per_device"] = int(slices_per_dev)
    result["attribution"] = attribute(result)

    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def attribute(row):
    """The three decisive numbers, per device, in units of cyl_dev."""
    cyl = row.get("cyl_dev_bytes") or 0
    marks = row.get("marks") or []
    n_dev = row.get("realized_n_devices") or 1
    if not cyl or not row.get("cuda"):
        return None
    workers = [m for m in marks if m["label"] == "band workers"]
    reduces = [m for m in marks if m["label"] == "band reduce"]
    batch = (row.get("batching") or {})
    batch_bytes = (batch.get("back_view_batch", 0)
                   * batch.get("back_bytes_per_view", 0))

    def per_dev(values):
        return [round(v / cyl, 3) for v in values]

    out = dict(
        view_batches_per_band_pass=batch.get("view_batches_per_band_pass"),
        back_batch_cyl=round(batch_bytes / cyl, 3),
        worker_peak_cyl=[], reduce_peak_cyl=[], worker_transient_cyl=[],
        reduce_transient_cyl=[], band_pass_entry_cyl=[])
    for pass_index, mark in enumerate(workers):
        out["worker_transient_cyl"].append(
            per_dev([p - e for p, e in zip(mark["peak_bytes"],
                                           mark["entry_bytes"])]))
        out["band_pass_entry_cyl"].append(per_dev(mark["entry_bytes"]))
    for mark in reduces:
        out["reduce_transient_cyl"].append(
            per_dev([p - e for p, e in zip(mark["peak_bytes"],
                                           mark["entry_bytes"])]))
    for i in range(n_dev):
        out["worker_peak_cyl"].append(
            round(max([m["peak_bytes"][i] for m in workers] or [0]) / cyl, 3))
        out["reduce_peak_cyl"].append(
            round(max([m["peak_bytes"][i] for m in reduces] or [0]) / cyl, 3))
    # 1. blocks live in the view loop = worst worker transient minus the batch
    out["live_blocks_cyl"] = [
        round(max((t[i] for t in out["worker_transient_cyl"]), default=0.0)
              - out["back_batch_cyl"], 3) for i in range(n_dev)]
    # 2. the band-pass entry step: pass 1 against pass 0
    if len(out["band_pass_entry_cyl"]) >= 2:
        out["entry_step_cyl"] = [
            round(out["band_pass_entry_cyl"][1][i]
                  - out["band_pass_entry_cyl"][0][i], 3) for i in range(n_dev)]
    else:
        out["entry_step_cyl"] = None
    # 3. which sub-step is the peak, per device
    out["peak_substep"] = [
        ("reduce" if out["reduce_peak_cyl"][i] > out["worker_peak_cyl"][i]
         else "workers") for i in range(n_dev)]
    out["measured_phase_peak_cyl"] = [
        round(max(out["worker_peak_cyl"][i], out["reduce_peak_cyl"][i]), 3)
        for i in range(n_dev)]
    # The verdict number: what the ledger charges this phase, against the
    # larger of the two sub-steps that actually run.  A positive value is the
    # over-charge, and if it tracks the smaller sub-step then the ledger is
    # summing two things that never coexist.
    modeled = None
    for phase in (row.get("ledger") or {}).get("phases") or []:
        if phase["name"] == "direct recon (back loop)":
            modeled = phase["per_device_bytes"]
    if modeled:
        out["modeled_phase_cyl"] = [round(modeled[i] / cyl, 3)
                                    for i in range(n_dev)]
        out["ledger_over_cyl"] = [
            round(out["modeled_phase_cyl"][i] - out["measured_phase_peak_cyl"][i], 3)
            for i in range(n_dev)]
        out["ledger_over_ratio"] = [
            round(out["modeled_phase_cyl"][i]
                  / max(1e-9, out["measured_phase_peak_cyl"][i]), 3)
            for i in range(n_dev)]
    return out


def generate(cfg):
    """One shared phantom and sinogram per (geometry, cell), md5'd, for every
    arm at that cell to read.  Pinned to a single device so the generator
    cannot itself become a multi-device run, and run with the peak counter
    untouched."""
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
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    # Protocol 6: absent everywhere in this job, generators included.
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"       # the shipped configuration
    if cfg["mode"] == "arm" and cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter (protocol 6)."""
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


def build_plan(arms):
    plan, seen = [], set()
    for _aid, geometry, cell, _n in arms:
        if (geometry, cell) in seen:
            continue
        seen.add((geometry, cell))
        entry = dict(mode="generate", geometry=geometry, cell=list(cell),
                     arm_id=f"generate_{geometry}_{cell[0]}", n_dev=None)
        if DEVICE != "cuda":
            entry["cpu_devices"] = [DEVICE]
        plan.append(entry)
    for arm_id, geometry, cell, n in arms:
        entry = dict(mode="arm", geometry=geometry, cell=list(cell),
                     arm_id=arm_id, n_dev=n)
        if DEVICE != "cuda":
            # SMOKE ONLY: the env pin is a CUDA-only mechanism (the policy
            # short-circuits at `visible < 2`), so the CPU path pins by device
            # LIST and says so on the row.
            entry["cpu_devices"] = [DEVICE] * n
        plan.append(entry)
    return plan


def main():
    arms = selected_arms()
    plan = build_plan(arms)
    if "--dry-run" in sys.argv:
        measured = [c for c in plan if c["mode"] == "arm"]
        print(f"mg6 plan: {len(measured)} probe arms "
              f"({len(plan) - len(measured)} generators)")
        for cfg in plan:
            print(f"  {cfg['arm_id']:<26}{cfg['geometry']:>9} "
                  f"{cfg['cell'][0]:>5} n={cfg['n_dev']}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg6_backloop_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg6 back-loop probe on {RUN_LABEL} ({DEVICE}); arms "
          f"{[a[0] for a in arms]} -> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY (protocol 11), and the n=1 ruler is first, so a
    # truncated job still yields its validity check.
    with open(out_path, "w") as sink:
        for cfg in plan:
            if cfg["mode"] == "generate" and os.path.exists(
                    _md5_path(cfg["geometry"], tuple(cfg["cell"]))):
                continue
            print(f"  {cfg['arm_id']}", flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        sink.write(json.dumps(dict(summary=summarize(rows, out_path))) + "\n")
        sink.flush()
    for _aid, geometry, cell, _n in arms:
        for path in (_sino_path(geometry, cell), _md5_path(geometry, cell)):
            if os.path.exists(path):
                os.remove(path)
    print(f"\nwrote {out_path}")


def summarize(rows, out_path):
    print(f"\n===== mg6 direct-recon back loop, sub-phase attribution "
          f"({out_path}) =====")
    header = (f'{"arm":>18}{"n":>3}{"dev":>6}{"vb":>5}{"nb":>4}'
              f'{"blocks":>8}{"entry+":>8}{"wpeak":>8}{"rpeak":>8}'
              f'{"peak@":>9}{"charged":>9}{"over":>7}')
    print(header)
    print("-" * len(header))
    print("  units: cylinders cyl(P, S_dev).  vb = realized back view batch, "
          "nb = view\n  batches per band pass, blocks = live cylinders in the "
          "view loop, entry+ =\n  band-pass entry step (pass 1 - pass 0), "
          "w/rpeak = the two sub-steps' peaks,\n  charged = what the ledger "
          "charges this phase, over = charged - max(w, r).")
    print("-" * len(header))
    summary = []
    for row in rows:
        if row.get("mode") != "arm":
            continue
        if row.get("error"):
            print(f'{row["arm_id"]:>18}   ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:60]}')
            summary.append(dict(arm_id=row["arm_id"], error=True))
            continue
        att = row.get("attribution")
        if not att:
            # The MEASURED side is torch.cuda.max_memory_allocated, which is
            # CUDA-only, so a CPU smoke exercises the INSTRUMENT and the
            # modeled side alone.  Say what fired rather than printing an
            # empty row that reads as a failure.
            marks = ", ".join(f"{k} x{v}" for k, v in
                              sorted((row.get("mark_counts") or {}).items()))
            print(f'{row["arm_id"]:>18}{row["n_dev"]:>3}   no CUDA measurement;'
                  f' marks_ok={row.get("marks_ok")}  [{marks}]')
            summary.append(dict(arm_id=row["arm_id"], cuda=False,
                                marks_ok=row.get("marks_ok"),
                                mark_counts=row.get("mark_counts"),
                                devices_ok=row.get("devices_ok")))
            continue
        batch = row["batching"]
        nan = float("nan")
        charged = att.get("modeled_phase_cyl") or []
        over = att.get("ledger_over_cyl") or []
        for i in range(row["realized_n_devices"]):
            step = (att["entry_step_cyl"][i]
                    if att["entry_step_cyl"] else nan)
            print(f'{row["arm_id"] if i == 0 else "":>18}'
                  f'{row["n_dev"] if i == 0 else "":>3}{i:>6}'
                  f'{batch["back_view_batch"] if i == 0 else "":>5}'
                  f'{batch["view_batches_per_band_pass"] if i == 0 else "":>4}'
                  f'{att["live_blocks_cyl"][i]:>8.2f}{step:>8.2f}'
                  f'{att["worker_peak_cyl"][i]:>8.2f}'
                  f'{att["reduce_peak_cyl"][i]:>8.2f}'
                  f'{att["peak_substep"][i]:>9}'
                  f'{(charged[i] if i < len(charged) else nan):>9.2f}'
                  f'{(over[i] if i < len(over) else nan):>7.2f}')
        summary.append(dict(arm_id=row["arm_id"], n_dev=row["n_dev"],
                            devices_ok=row.get("devices_ok"),
                            bodies_ok=row.get("bodies_ok"),
                            calibration_absent_ok=row.get(
                                "calibration_absent_ok"),
                            gpu_hot=row.get("gpu_hot"),
                            cyl_dev_bytes=row.get("cyl_dev_bytes"),
                            attribution=att))
    print("-" * len(header))
    print("READING IT.  blocks ~ min(3, nb) says the three-cylinder finding "
          "survives only\n  where the view loop runs three times.  entry+ ~ "
          "+2 on device 0 and +1 on the\n  last device says the previous band "
          "pass's partials are still live -- a real,\n  avoidable "
          "co-residency worth releasing.  `over` near the SMALLER of wpeak "
          "and\n  rpeak says the charge is a phantom: the ledger is summing "
          "two sub-steps that\n  never coexist, and the remedy is a charge "
          "correction on top of the code fix.")
    return summary


if __name__ == "__main__":
    if "--worker" in sys.argv:
        cfg = json.loads(sys.argv[sys.argv.index("--worker") + 1])
        try:
            out = generate(cfg) if cfg["mode"] == "generate" else run_arm(cfg)
        except Exception:                                         # noqa: BLE001
            traceback.print_exc()
            sys.exit(1)
        print("__RESULT__" + json.dumps(out))
    else:
        main()
