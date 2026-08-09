"""mg5 -- CHARTER A: attribute the forward at n>1.

The charter is `multigpu_findings.md` §6.2, fired by the two anomalies of §1.3,
under the eleven protocols of `multigpu_plan.md` §3.  mg5 is the plan's
tuning-menu prefix (§4, mg5); this is its FIRST fired lever, the per-device view
chunk, and the sweep exists to ATTRIBUTE before any constant moves.

THE TWO ANOMALIES THIS SWEEP EXPLAINS (mg1, job 15011662, rows
`mg1_readout_h014_20260809_050121.jsonl`):

    cone 1024      forward region 31.0, 29.9, 28.6 s at n = 1, 2, 4
                   -- FLAT in the device count while per-device views drop 4x.
                   40-54 percent of that cell's composed wall, so the flatness
                   is why cone n=2 is a net regression (0.92x).
    parallel 1024  forward region 15.9, 18.0, 7.3 s at n = 1, 2, 4
                   -- RISES at n=2.  A forward that costs MORE with half the
                   views per device points at a per-device regime change rather
                   than at communication.

THE THREE HYPOTHESIS CLASSES, and the reading that discriminates each.  The
harness prints the numbers side by side and states each signature; it prints NO
verdict, because attribution is analysis (the mg4 precedent: the knee rule is
applied in analysis, not by the harness).

    (a) KERNEL-SPECIFIC behavior at shard shapes -- launch shapes, atomics
        contention on the forward's output plane.
        Signature: the kernel forward is flat/rising in n while the TORCH
        forward body, run through the same drivers at the same counts, scales.
        Read from: the torchfwd arm's forward region against the prod arm's,
        across counts.

    (b) The REALIZED per-device view batch shifting under the count-divided
        budget (protocol 3: the batch budget divides by the count while the
        kernel cost models do not all follow, so the realized batch is not
        invariant in n).
        Signature: the realized forward batch moves with n, and the forward
        region tracks the chunk ladder.
        Read from: `fwd_view_batch_per_device` (the static probe at the
        full-pixel set) and `view_batch_observed` (what the warm passes
        actually chose, per direction per device), against the chunk ladder's
        forward-region curve.
        ARITHMETIC REGISTERED IN ADVANCE, so the reading is a test and not a
        fishing trip.  `Projectors._transient_budget_bytes` (projectors.py:231)
        scales the budget by the PER-DEVICE view shard and then caps it at
        2 GiB; at both 1024 cells 8 x the per-device sinogram exceeds 2 GiB at
        n = 1, 2 AND 4, so the budget is PINNED AT THE CAP at every count here
        and hypothesis (b) predicts NO shift from the budget side.  What can
        still move with n is `bytes_per_view`: parallel's forward plane term is
        4 * num_channels * num_value_cols and the value columns ARE the slice
        shard, so it shrinks with n; cone's plane term reads `num_rows_r` from
        the params and does not.  The prediction this sweep tests is therefore
        parallel-forward batch 128 at every count (chunk-bound) and cone-forward
        batch ~52 at every count (cap-bound) -- a flat realized batch under a
        flat forward region, which would leave (a) and (c) holding the field.

    (c) The DRIVER fan-out / assembly around the kernel -- the banded forward's
        per-band broadcast, the per-owner thread fan-out, the concatenate.
        Signature: the forward funnel's HOST WALL minus its largest per-device
        EVENT SPAN -- the gap protocol 10's instrument exists to price -- grows
        with n while the device span itself falls.  A torchfwd arm that is
        ALSO flat says the same thing from the other side: the anomaly does not
        live in the kernel.
        Read from: `fwd_gap_per_pass_s`, printed beside the region wall.

THE ARMS (27 measured + 2 untimed generators).  Five coordinates: cone 1024 at
n = 1, 2, 4 and parallel 1024 at n = 1, 2, the n=1 arms being the references.

    prod       the production configuration -- kernels bound in BOTH directions
               -- with the FORWARD view chunk pinned to each of {32, 64, 128,
               256} and the BACK chunk left at its shipped constant.  Four arms
               per coordinate = 20.  The chunk-128 arm reproduces mg1's
               production arm exactly and is this job's validity anchor.
    torchfwd   the TORCH forward body bound against the SELECTED (kernel) back
               body.  One arm per coordinate = 5.  This is the (a)-versus-(c)
               discriminator.
    drift      the chunk-128 n=1 prod arm re-run as the job's LAST arm, one per
               cell = 2.  See ARM ORDER below: the n=1-first rule puts every
               reference at the job's start and every n>1 arm after it, so a
               linear drift would bias every scaling number in the readout.
               These two arms price that bias directly instead of assuming it
               away.

MBIRTORCH_DISABLE_TRITON IS NOT THE TOOL for the torchfwd arm, and this file
never sets it to 1.  The kill switch kills BOTH directions (it is read inside
the availability probe, `kernel_availability.py`), which would confound the
forward change with a back change.  The torchfwd arm instead binds the bodies
EXPLICITLY, the way the standing kernel-times-sharding gate does
(tests/test_kernels_sharded.py:94 `_force_one_kernel`, and dp5's `_force_bodies`
at dp5_kernel_shard_probe.py:72): `model._view_batch_bodies` is shadowed with a
pair, and `create_projectors()` rebinds the per-device instances.  The back half
of the pair is the body the SHIPPED SELECTION ITSELF returned, read out of
`_view_batch_bodies()` before the shadow goes in, so the back direction is
production by construction rather than by a matching import.  Every arm in this
file, torchfwd included, runs with `MBIRTORCH_DISABLE_TRITON=0` and asserts it.

THE CHUNK MECHANISM, reused from kb2 (kb2_vbsweep.py:369, with its SPEC table at
kb2_vbsweep.py:108).  The kernel wrappers' `_view_batch_cost` attributes read
their module-level chunk constant AT CALL TIME
(triton_cone.py:693 `return 48 * num_pixels + plane_bytes, CONE_FWD_VIEW_CHUNK`;
triton_parallel.py:484, the parallel twin), so a chunk arm sets the module
attribute in its own subprocess and nothing in the mbirtorch package is edited.
Only the FORWARD constant moves; the arm reads the back constant before and
after and fails loudly if it drifted.

WHAT EVERY ARM CARRIES (protocols 9-11).  mg1's three-region instrument
(`RegionInstrument` + `attach_instrument`, imported from mg1_readout so the
region definitions cannot drift from the readings mg5 is compared against), the
realized-batch observer (`observe_view_batches`), one discarded cold pass plus
three warm repeats with the median and spread, per-device peaks, the GPU health
sample and the throttle rule, and incremental jsonl with the n=1 arms first.

THE IMPORT OF mg1_readout IS DELIBERATE, and is a departure from mg2/mg3/mg4,
which each copied the machinery.  mg5's entire deliverable is its forward-region
numbers set against mg1's recorded 31.0/29.9/28.6 and 15.9/18.0, so a
copied-and-drifted instrument would produce numbers that look comparable and are
not.  The cost is a staging dependency: mg1_readout.py MUST be staged beside
this file (the sbatch says so, and the import failure names the remedy).

ARM ORDER (protocol 9 and protocol 11, and the one place they cannot both be
literal).  Protocol 11 wants the n=1 reference arms first, so a truncated job
still yields its validity check; protocol 9 wants counts blocked and reversed.
n=1-first wins, exactly as it did in mg1, and the reversal is applied to the
SWEPT axis instead: within a cell the chunk ladder runs ascending at the first
n>1 count and DESCENDING at the next, so ladder position and time position are
decorrelated inside each block.  Between blocks the drift arms measure what the
ordering cannot cancel.

    PHASE 0   both generators, then every n=1 arm (ladder ascending, torchfwd)
    PHASE 1   cone n=2 (ascending), cone n=4 (descending), parallel n=2
    PHASE 2   the two drift arms

THE VALUE COLUMN is reported, not gated.  Each arm's strided row sample is
diffed against the chunk-128 prod arm AT THE SAME COUNT, so the column reads the
chunk's own float-summation-order effect (and, on the torchfwd arm, the
kernel-versus-torch forward divergence) with the partition order held fixed.
A reading above MG5_VALUE_FLAG is flagged as structural, not as a failure.

THE GATHER CONTRACT (nt2_local_shard_check.py).  ``Shards.gather()`` ALREADY
returns numpy; re-detaching its result is the recorded failure that cost the
nightly's first 4-GPU trial all 32 of its n>1 rows.  Every host exit here goes
through mg1's ``_to_numpy``, which never re-detaches a gather.

Run:
    <torch python> mg5_fwd_attrib.py         on a 4-GPU node (mg5_gautschi.sbatch)
    python mg5_fwd_attrib.py --dry-run       anywhere: print the arm plan
    MG5_SMOKE=1 python mg5_fwd_attrib.py     the local CPU smoke
    python mg5_fwd_attrib.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON=<python>          interpreter for the arm subprocesses
    MG5_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG5_GEOMS=cone,parallel           subset of the geometries
    MG5_COUNTS=1,2,4                  subset of the device counts
    MG5_CHUNKS=32,64,128,256          the forward chunk ladder
    MG5_ITERATIONS=3                  VCD iterations per recon
    MG5_WARM_REPEATS=3                warm repeats after the discarded cold pass
    MG5_SKIP_TORCH_FWD=1              drop the torchfwd discriminator arms
    MG5_SKIP_DRIFT=1                  drop the two drift arms
    MG5_KEEP_ARTIFACTS=1              keep the shared sinograms after the run
    MG5_SMOKE=1                       the local smoke (tiny cell, few iters)
    MG5_DEVICE=cpu                    smoke device
    MG5_SMOKE_CPU_N2=1                smoke only: a 2-virtual-device CPU arm
                                      that exercises the instrument's n>1 wiring
"""

import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback

# ── the mg1 machinery (see the module docstring: this import is deliberate) ────
# ``RegionInstrument`` is imported although ``attach_instrument`` is what this
# file calls: the name is the WITNESS that mg1's instrument is the one in use,
# so a rename there fails here at import rather than silently downstream.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from mg1_readout import (NESTED_REGIONS, REGIONS, REGIONS_ABSENT_AT_N1,
                             REGIONS_HOST_ONLY_AT_N1, RegionInstrument,
                             attach_instrument, observe_view_batches,
                             _launch_key_counts, _md5, _rel_max, _strict_subset,
                             _to_numpy, _view_batch_static, _weights,
                             row_is_hot, sample_gpu_health, worst_health,
                             CLOCK_DEPRESSED_FRAC)
    assert RegionInstrument is not None
except ImportError as exc:                                        # noqa: BLE001
    raise SystemExit(
        f"mg5 could not import mg1_readout ({exc}).  mg5 reuses mg1's region "
        f"instrument verbatim so its forward-region readings are comparable "
        f"with mg1's recorded ones; mg1_readout.py must be staged in the same "
        f"directory as this file.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The two anomalous coordinates of §1.3, and nothing else: this is an
# attribution sweep, not a matrix.
CELL = (1024, 1008, 992)
COORDINATES = (("cone", CELL, (1, 2, 4)),
               ("parallel", CELL, (1, 2)))
GEOMETRIES = tuple(g for g, _c, _n in COORDINATES)
COUNTS = (1, 2, 4)
# The PINNED ladder (charter A).  The shipped constant is 128 in both
# directions and at both geometries; the 128 arm is therefore the anchor that
# reproduces mg1's production configuration, and the arm asserts that identity
# against the module constant rather than assuming it.
CHUNKS = (32, 64, 128, 256)
SHIPPED_CHUNK = 128

SMOKE = os.environ.get("MG5_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
# On CPU the kernels are unavailable, so the chunk constant is inert; the smoke
# exercises every other step (the override path, the body binding, the
# instrument's n>1 wiring, the checks, the ordering, the jsonl, the summary).
SMOKE_CHUNKS = (32, 128)
DEVICE = os.environ.get("MG5_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG5_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # kb3's / mg1's seed, so the arms stay comparable
SAMPLE_ROWS = 16
WARM_REPEATS = max(1, int(os.environ.get("MG5_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# mg1's recorded readings at the SHIPPED configuration (§1.2, §1.3), which the
# chunk-128 prod arms must reproduce.  This job's validity check: mg5 crosses a
# job boundary, and protocol 11 requires such a comparison to pin the node or
# carry a shared anchor -- the chunk-128 arms ARE the shared anchor, being mg1's
# production arm rebuilt.
MG1_FORWARD_REGION_S = {("cone", 1): 31.0, ("cone", 2): 29.9, ("cone", 4): 28.6,
                        ("parallel", 1): 15.9, ("parallel", 2): 18.0,
                        ("parallel", 4): 7.3}
MG1_BACK_REGION_S = {("cone", 1): 25.3, ("cone", 2): 30.8, ("cone", 4): 17.5,
                     ("parallel", 1): 3.5, ("parallel", 2): 2.2,
                     ("parallel", 4): 3.0}
MG1_COMPOSED_S = {("cone", 1): 61.57, ("cone", 2): 67.23, ("cone", 4): 53.10,
                  ("parallel", 1): 40.00, ("parallel", 2): 39.40,
                  ("parallel", 4): 23.36}

# The value column is REPORTED.  This only decides whether a reading is printed
# as structural: mg1 recorded torch's own-count divergences in the e-4 class at
# these cells and the cross-framework column at 6.1e-3, and kb4 bounded the
# kernels' own contribution at the e-6 parity class against eager bodies, so a
# same-count chunk diff above 1e-2 is a different animal from float order.
VALUE_FLAG = float(os.environ.get("MG5_VALUE_FLAG", "1e-2"))
# Reconciliation tolerance, mg1's constant: the bracketed regions are a proper
# subset of the composed wall.
RECONCILE_SLACK = 0.02

RESULTS_DIR = os.environ.get(
    "MG5_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]

# Per geometry: the kernel module carrying the swept FORWARD chunk constant and
# its untouched BACK twin, the wrapper names the arm checks match against, and
# the torch forward body the torchfwd arm binds.  kb2's SPEC table
# (kb2_vbsweep.py:108) narrowed to the forward direction, which is the only one
# this charter varies.
SPEC = {
    "parallel": dict(kernel_module="mbirtorch.triton_parallel",
                     fwd_chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                     back_chunk_const="PARALLEL_BACK_VIEW_CHUNK",
                     fwd_wrapper="_parallel_forward_view_batch_triton",
                     back_wrapper="_parallel_back_view_batch_triton",
                     body_module="mbirtorch.parallel_beam",
                     torch_fwd_body="_parallel_forward_view_batch"),
    "cone": dict(kernel_module="mbirtorch.triton_cone",
                 fwd_chunk_const="CONE_FWD_VIEW_CHUNK",
                 back_chunk_const="CONE_BACK_VIEW_CHUNK",
                 fwd_wrapper="_cone_forward_view_batch_triton",
                 back_wrapper="_cone_back_view_batch_triton",
                 body_module="mbirtorch.cone_beam",
                 torch_fwd_body="_cone_forward_view_batch"),
}
# ──────────────────────────────────────────────────────────────────────────────


def selected_plan():
    """(geometries, counts, chunks), each narrowed by its env knob."""
    chosen = _strict_subset("MG5_GEOMS", set(GEOMETRIES))
    geometries = [g for g in GEOMETRIES if g in chosen]
    if SMOKE and not os.environ.get("MG5_COUNTS", "").strip():
        # The env pin is a CUDA-only mechanism (the policy short-circuits at
        # `visible < 2`), so a pinned n>1 arm on CPU would silently measure
        # n=1.  The smoke therefore runs n=1 plus the dedicated CPU virtual
        # 2-device arms, which pin by device LIST and say so.
        counts = [1]
    else:
        keep = _strict_subset("MG5_COUNTS", set(COUNTS), int)
        counts = [n for n in COUNTS if n in keep]
    default_chunks = SMOKE_CHUNKS if SMOKE else CHUNKS
    raw = os.environ.get("MG5_CHUNKS", "").strip()
    if raw:
        chunks = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit() or int(token) <= 0:
                raise ValueError(f"MG5_CHUNKS: bad chunk {token!r}")
            chunks.append(int(token))
        if not chunks:
            raise ValueError(f"MG5_CHUNKS: no valid tokens in {raw!r}")
    else:
        chunks = list(default_chunks)
    return geometries, counts, chunks


def cell_for(_geometry):
    return SMOKE_CELL if SMOKE else CELL


# ── staged-artifact mechanics (protocol 5's md5 discipline) ───────────────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg5_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id):
    return os.path.join(RESULTS_DIR, f"_mg5_sample_{arm_id}.npy")


# ── THE CHUNK MECHANISM (kb2_vbsweep.py:369) ──────────────────────────────────
def pin_forward_chunk(geometry, chunk):
    """Pin the geometry's FORWARD view chunk for this process, and read back
    both chunk constants so the arm can prove what it changed.

    Nothing in the mbirtorch package is edited.  The kernel wrappers'
    ``_view_batch_cost`` attributes return their module constant AT CALL TIME
    (triton_cone.py:693, triton_parallel.py:484), so setting the module
    attribute before the projectors run is the whole mechanism -- kb2's, at
    kb2_vbsweep.py:369, narrowed here to the forward direction.

    Returns the ``(module, shipped_fwd, shipped_back)`` triple; the caller
    records the shipped values and re-reads the back constant after the run.
    """
    import importlib

    spec = SPEC[geometry]
    module = importlib.import_module(spec["kernel_module"])
    shipped_fwd = int(getattr(module, spec["fwd_chunk_const"]))
    shipped_back = int(getattr(module, spec["back_chunk_const"]))
    if chunk is not None:
        setattr(module, spec["fwd_chunk_const"], int(chunk))
    return module, shipped_fwd, shipped_back


# ── THE BODY-BINDING MECHANISM (tests/test_kernels_sharded.py:94) ─────────────
def force_torch_forward(model, geometry):
    """Bind the TORCH forward body against the SELECTED back body.

    The mirror of the standing gate's ``_force_one_kernel``
    (tests/test_kernels_sharded.py:94), and of dp5's ``_force_bodies``
    (dp5_kernel_shard_probe.py:72): ``_view_batch_bodies`` is shadowed on the
    instance and ``create_projectors()`` rebinds the per-device instances
    through it.

    Two differences from the test, both deliberate.  The back half of the pair
    is the body the SHIPPED SELECTION returned -- read out of
    ``_view_batch_bodies()`` before the shadow goes in -- so this arm's back
    direction is production by construction, not by a matching import.  And the
    default pair is returned to the caller as the LOUD WITNESS the standing gate
    also carries (tests/test_kernels_sharded.py:243): an arm that exists to
    measure the torch forward against the kernel must fail loudly if the
    kernels silently declined, rather than compare torch with torch and pass
    vacuously.

    ``create_projectors()`` is called here rather than left to the layout
    settle because the two pinning paths differ: the CUDA env pin settles the
    layout inside the first ``recon`` and rebuilds the projectors through the
    shadow, but the smoke's explicit ``configure_devices(devices=[...])`` has
    already installed its layout by now, and its projectors would keep the
    default pair.  Calling it makes both paths bind the same way; on the CUDA
    path the settle simply rebuilds through the same shadow.
    """
    import importlib

    spec = SPEC[geometry]
    torch_fwd = getattr(importlib.import_module(spec["body_module"]),
                        spec["torch_fwd_body"])
    default_fwd, default_back = model._view_batch_bodies()
    pair = (torch_fwd, default_back)
    model._view_batch_bodies = lambda: pair
    model.create_projectors()
    return dict(default_fwd_body=getattr(default_fwd, "__name__",
                                         str(default_fwd)),
                default_back_body=getattr(default_back, "__name__",
                                          str(default_back)),
                forced_fwd_body=getattr(torch_fwd, "__name__", str(torch_fwd)))


def _per_device_body_names(model):
    """The bodies actually bound, PER DIRECTION PER DEVICE.  The selection hook
    says what was chosen; these say what the driver holds, one compiled
    instance per device (projectors.py:261).  ``maybe_compile`` prefixes a
    compiled torch body with ``compiled_`` (projectors.py:129) and returns a
    kernel wrapper untouched, so both forms are recognizable by name."""
    pf = model.projector_functions
    return ([getattr(b, "__name__", str(b)) for b in pf._fwd_body_per_dev],
            [getattr(b, "__name__", str(b)) for b in pf._back_body_per_dev])


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


def torch_worker(cfg):
    """One arm: cold pass discarded, then WARM_REPEATS warm repeats, with mg1's
    three-region instrument attached and the realized-batch observer installed.

    ORDERING NOTE, load-bearing (mg1's, unchanged).  Every projector-dependent
    check runs AFTER the cold pass.  The automatic branch settles the layout
    inside the first ``recon`` call, and a settle that changes the count calls
    ``_install_device_layout`` -> ``create_projectors``
    (tomography_model.py:842), which REPLACES ``model.projector_functions``.  A
    view-batch or body reading taken before that would describe a one-device
    projector set under an n-device label.  The instrument is immune: it shadows
    instance and module attributes the engine resolves at call time.  So is the
    body shadow, for the same reason.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    arm_class, n_dev, chunk = cfg["arm_class"], cfg.get("n_dev"), cfg.get("chunk")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    # The chunk pin goes in BEFORE the model is built, so no projector, ledger,
    # or preflight can read a stale constant.
    kernel_module, shipped_fwd_chunk, shipped_back_chunk = pin_forward_chunk(
        geometry, chunk if arm_class == "prod" else None)

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
                  requested_chunk=chunk,
                  shipped_fwd_chunk=shipped_fwd_chunk,
                  shipped_back_chunk=shipped_back_chunk)

    # ── the env arm checks ───────────────────────────────────────────────────
    # Protocol 6: the calibration mode owns max_memory_allocated, so it must be
    # absent everywhere in mg5.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    # THE CHARTER'S EXPLICIT POINT: the kill switch is not the tool for the
    # torchfwd arm, because it kills both directions.  Every arm here, torchfwd
    # included, must see it OFF.
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))
    if cuda:
        result["pin_env_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") == str(n_dev))
    # The chunk arms may move the FORWARD constant only.
    result["shipped_chunk_is_the_anchor_ok"] = (
        None if arm_class != "prod" or chunk != SHIPPED_CHUNK
        else shipped_fwd_chunk == SHIPPED_CHUNK)

    # ── the body binding (the torchfwd arm's whole mechanism) ────────────────
    if arm_class == "torchfwd":
        binding = force_torch_forward(model, geometry)
        result.update(binding)
        # The loud witness: this arm exists to measure the torch forward
        # against the kernel, so a silent availability decline must fail here
        # rather than compare torch with torch and pass vacuously.
        result["kernels_were_available_ok"] = (
            None if not cuda
            else ("triton" in binding["default_fwd_body"]
                  and "triton" in binding["default_back_body"]))
        # Said out loud rather than left to be inferred from two equal strings:
        # on the CPU smoke the shipped selection ALREADY returns the torch
        # forward, so this arm binds what would have been bound anyway and
        # measures nothing about the kernel.  It still exercises the binding
        # path, which is what the smoke is for.
        result["torchfwd_degenerate"] = (
            binding["default_fwd_body"] == binding["forced_fwd_body"])
        expect_kernels = (False, cuda)
    else:
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

    # The instrument is attached BEFORE the cold pass -- it shadows instance and
    # module attributes the engine resolves at call time, so a mid-run settle
    # cannot lose it -- and reset after, so its walls cover the warm repeats
    # alone (protocol 7: a cold pass pays n per-device Triton compiles inside
    # the very regions being bracketed).
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
    # PER DIRECTION PER DEVICE, the charter's requirement: one instance per
    # device, each the intended kind, and for the torchfwd arm each forward
    # instance additionally NAMING the torch body -- so a wrong body cannot
    # pass merely by not being a kernel.
    want_fwd_kernel, want_back_kernel = expect_kernels
    torch_fwd_name = SPEC[geometry]["torch_fwd_body"]
    result["bodies_per_device_ok"] = (
        len(fwd_per_dev) == n_realized and len(back_per_dev) == n_realized
        and all(("triton" in name) == want_fwd_kernel for name in fwd_per_dev)
        and all(("triton" in name) == want_back_kernel for name in back_per_dev)
        and (want_fwd_kernel
             or all(torch_fwd_name in name for name in fwd_per_dev)))
    result["bodies_ok"] = (
        result["bodies_per_device_ok"]
        and ("triton" in fwd_name) == want_fwd_kernel
        and ("triton" in back_name) == want_back_kernel)

    # The realized view batch per direction per device, at the full-pixel-set
    # inputs, against the formula of the body EXPECTED to be bound (mg1's
    # static probe, verbatim).
    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok

    # THE CHUNK ARM CHECK (kb2_vbsweep.py:378): the realized forward batch
    # equals what this arm's requested chunk implies under the live budget.
    # Skipped -- with the reason RECORDED, never a vacuous pass -- wherever the
    # forward body carries no chunk (the torchfwd arm by design, the CPU smoke
    # by availability).
    result["fwd_chunk_realized"] = None
    result["fwd_chunk_expected"] = None
    result["fwd_chunk_ok"] = None
    result["fwd_chunk_skip_reason"] = None
    pf = model.projector_functions
    args = model._view_batch_args()
    num_pixels = int(vb_record["num_pixels_full"])
    fwd_cols = int(vb_record["view_batch_cols"]["fwd"])
    budget = int(vb_record["budget_bytes"])
    bound_fwd = pf._fwd_body_per_dev[0]
    cost = getattr(bound_fwd, "_view_batch_cost", None)
    realized_fwd = int(pf._effective_view_batch(bound_fwd, num_pixels, fwd_cols,
                                                args))
    result["fwd_chunk_realized"] = realized_fwd
    if arm_class != "prod":
        result["fwd_chunk_skip_reason"] = (
            "torchfwd arm: the torch forward body carries no view chunk, so "
            "its batch follows the torch rule and the chunk constant does not "
            "apply")
    elif cost is None:
        result["fwd_chunk_skip_reason"] = (
            "the forward body bound carries no _view_batch_cost (kernels "
            "unavailable -- the CPU smoke): the chunk constant is inert")
    else:
        bytes_per_view, chunk_seen = cost(num_pixels, fwd_cols, args)
        expected = max(1, min(int(chunk), budget // max(1, int(bytes_per_view))))
        result["fwd_bytes_per_view"] = int(bytes_per_view)
        result["fwd_chunk_seen_by_cost_fn"] = int(chunk_seen)
        result["fwd_chunk_expected"] = expected
        result["fwd_chunk_ok"] = (realized_fwd == expected
                                  and int(chunk_seen) == int(chunk))
        # The budget cap, not the chunk, may be what binds -- which is
        # hypothesis (b)'s reading and must be visible on the row.
        result["fwd_chunk_binds"] = int(chunk) <= (budget
                                                   // max(1, int(bytes_per_view)))
    # The BACK constant must not have moved: this sweep varies the forward only.
    result["back_chunk_after"] = int(
        getattr(kernel_module, SPEC[geometry]["back_chunk_const"]))
    result["back_chunk_unchanged_ok"] = (
        result["back_chunk_after"] == shipped_back_chunk)
    result["fwd_chunk_after"] = int(
        getattr(kernel_module, SPEC[geometry]["fwd_chunk_const"]))

    # Installed only now, so the observer sits on the projector instance the
    # settled layout built, and so the static probe above cannot pollute it.
    observed_batches = observe_view_batches(model)

    # ── the warm repeats ─────────────────────────────────────────────────────
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

    # ── arm check: the realized device list after the timed call (protocol 1) ─
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev) if cuda else \
        (len(realized) == (len(pin_devices) if pin_devices else 1))

    # ── arm check: the launch-key witness, POSITIVE and NEGATIVE ─────────────
    # kb3's positive witness, plus the negative half this charter needs: a
    # torchfwd arm must add BACK keys and NO forward keys.  Nothing else proves
    # the torch forward body actually ran the forward.
    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        if arm_class == "torchfwd":
            result["kernels_launched_ok"] = (
                result["back_launch_keys_delta"] > 0
                and result["fwd_launch_keys_delta"] == 0)
        else:
            result["kernels_launched_ok"] = (
                result["back_launch_keys_delta"] > 0
                and result["fwd_launch_keys_delta"] > 0)

    # ── arm check: the realized view batches, as the warm passes chose them ──
    result["view_batch_observed"] = {k: sorted(v.items())
                                     for k, v in observed_batches.items()}

    # ── the instrument read-out and its own checks (protocol 10) ─────────────
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

    passes = max(1, WARM_REPEATS)
    composed = result["vcd_warm"]
    for region in REGIONS:
        block = regions[region]
        result[f"{region}_wall_per_pass_s"] = block["host_wall_s"] / passes
        result[f"{region}_dev_span_max_per_pass_s"] = (
            block["device_span_max_ms"] / 1e3 / passes)
        result[f"{region}_calls"] = block["calls"]
        # THE ORCHESTRATION GAP -- hypothesis (c)'s direct reading.  Host wall
        # minus the largest per-device event span is what protocol 10's
        # instrument exists to price: the fan-out, the broadcast, and the
        # assembly around the kernel.  On the CPU smoke the per-device map
        # collapses to one 'cpu' key whose span IS the host wall, so the gap is
        # an artifact there and the backend field says so.
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
    result["region_coverage_warn"] = (
        count > 1 and result["region_remainder_frac"] is not None
        and result["region_remainder_frac"] > 0.5)
    result["forward_share_of_composed"] = (
        result["forward_funnel_wall_per_pass_s"] / composed if composed else None)
    back = regions["back_funnel"]["host_wall_s"]
    result["band_reduce_share_of_back"] = (
        regions["band_reduce"]["host_wall_s"] / back if back else None)

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build ONE shared sinogram per geometry: phantom -> sinogram -> .npy,
    plus its md5 sidecar.  Every arm at that coordinate reconstructs THAT array,
    so no arm's timing or value column carries an input difference.  Pinned to
    one device so the generator cannot itself become a multi-device run."""
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
    """The common tail: checksum, the strided row sample for the value column,
    and the host peak."""
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
    keeps the model on the automatic branch where the preflight still runs; an
    explicit configure_devices call would take the explicit branch and get no
    preflight, so the two are not interchangeable and mixing them would make the
    arms incomparable.

    MBIRTORCH_DISABLE_TRITON is "0" for EVERY arm, torchfwd included.  The kill
    switch disables both directions at once (it is read inside the availability
    probe, which caches its verdict per process), so it cannot express a
    single-direction change; the torchfwd arm binds its bodies instead.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # protocol 6
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg5_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg5_{cfg['arm_id']}.json")
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
    # protocol discards it.
    row["subprocess_wall_s"] = subprocess_wall
    return row


def build_plan(geometries, counts, chunks):
    """The arm plan, in JOB ORDER (see the module docstring's ARM ORDER)."""
    skip_torch_fwd = os.environ.get("MG5_SKIP_TORCH_FWD", "0") == "1"
    skip_drift = os.environ.get("MG5_SKIP_DRIFT", "0") == "1"
    smoke_cpu_n2 = SMOKE and os.environ.get("MG5_SMOKE_CPU_N2", "1") == "1"
    coordinates = [(g, cell_for(g), [n for n in ns if n in counts])
                   for g, _c, ns in COORDINATES if g in geometries]
    phase0, phase1, phase2 = [], [], []

    def arm(arm_class, geometry, cell, n_dev, chunk=None, suffix="", **extra):
        tag = f"c{chunk}" if chunk is not None else arm_class
        return dict(framework="torch", arm_class=arm_class, geometry=geometry,
                    cell=list(cell), n_dev=n_dev, chunk=chunk,
                    arm_id=f"{geometry}_{cell[0]}_n{n_dev}_{arm_class}"
                           f"{'_' + tag if chunk is not None else ''}{suffix}",
                    **extra)

    def block(geometry, cell, n, ladder, cpu_devices=None, suffix=""):
        """One coordinate's arms: the chunk ladder in the given direction, then
        the torchfwd discriminator."""
        extra = dict(cpu_devices=cpu_devices) if cpu_devices else {}
        out = [arm("prod", geometry, cell, n, chunk=c, suffix=suffix, **extra)
               for c in ladder]
        if not skip_torch_fwd:
            out.append(arm("torchfwd", geometry, cell, n, suffix=suffix,
                           **extra))
        return out

    for geometry, cell, ns in coordinates:
        gen = dict(framework="torch", arm_class="generator", geometry=geometry,
                   cell=list(cell), n_dev=None, chunk=None,
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if SMOKE and DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        phase0.append(gen)

    for geometry, cell, ns in coordinates:
        # PHASE 0: the n=1 REFERENCE arms, first (protocol 11), ladder ascending.
        if 1 in ns:
            phase0.extend(block(geometry, cell, 1, list(chunks)))
        # PHASE 1: the n>1 blocks, ladder direction alternating per count so
        # ladder position and time position are decorrelated (protocol 9's
        # reversal, applied to the swept axis).
        for index, n in enumerate([n for n in ns if n != 1]):
            ladder = list(chunks) if index % 2 == 0 else list(reversed(chunks))
            phase1.extend(block(geometry, cell, n, ladder))
        if smoke_cpu_n2:
            # SMOKE ONLY: two virtual cpu devices, so the instrument's n>1
            # wiring (band_reduce, halo, the sharded funnels) and the body
            # binding under an explicit layout are exercised without CUDA.  The
            # env pin is CUDA-only, so this arm pins by device LIST and says so.
            phase1.extend(block(geometry, cell, 2, list(chunks),
                                cpu_devices=[DEVICE, DEVICE],
                                suffix="_smokecpu"))
        # PHASE 2: the drift arm -- the n=1 anchor re-run at the job's tail.
        # ``is_drift`` keeps it out of the summary's arm index, which is keyed
        # by (geometry, count, class, chunk): a repeat has the same key as the
        # arm it repeats, and without the flag it would silently REPLACE the
        # reference it exists to measure against.
        if not skip_drift and 1 in ns and SHIPPED_CHUNK in chunks:
            phase2.append(arm("prod", geometry, cell, 1, chunk=SHIPPED_CHUNK,
                              suffix="_drift", is_drift=True))
    return phase0, phase1, phase2


# ── the summary ───────────────────────────────────────────────────────────────
def _key(row):
    return (row.get("geometry"), row.get("n_dev"), row.get("arm_class"),
            row.get("chunk"))


def _fmt(value, spec, dash="-"):
    """``format(value, spec)``, with a missing value rendered as a dash padded
    to the SAME width -- an unpadded dash shifts every column to its right and
    makes a table with one absent arm unreadable."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format(value, spec)
    width = ""
    for char in spec:
        if char.isdigit():
            width += char
        else:
            break
    return f"{dash:>{int(width) if width else 1}}"


def summarize(rows, geometries, counts, chunks, out_path):
    """Three blocks: the per-coordinate arm table with its checks, the
    forward-attribution table that carries the charter's question, and the mg1
    anchor reproduction.  NO verdict is printed -- the hypothesis signatures are
    stated and the numbers are placed beside them, and the attribution itself is
    analysis (the mg4 precedent)."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    # The drift arms are repeats and share their reference's key, so they are
    # indexed by arm_id alone and read out separately.
    by = {}
    for row in live:
        if not row.get("is_drift"):
            by[_key(row)] = row
    summaries = []
    print(f"\n===== mg5 forward attribution ({out_path}) =====")

    for geometry in geometries:
        cell = cell_for(geometry)
        # The counts actually measured, read off the rows: in the smoke the
        # CPU 2-virtual-device arms carry an n the coordinate table does not.
        ns = sorted({r["n_dev"] for r in live
                     if r.get("geometry") == geometry and r.get("n_dev")})
        print(f"\n--- {geometry} {cell} ---")
        print(f"{'arm':>26}{'n':>3}{'chunk':>7}{'vb_f':>6}{'cold_s':>9}"
              f"{'warm_s':>9}{'spread':>8}{'fwd_s':>8}{'fwd_dev':>9}"
              f"{'fwd_gap':>9}{'back_s':>8}{'rem':>7}{'pk_GB':>7}"
              f"{'checks':>22}")
        summary = dict(geometry=geometry, cell=list(cell), rows=[],
                       attribution=[])
        for row in rows:
            if row.get("geometry") != geometry or \
                    row.get("arm_class") == "generator":
                continue
            if row.get("error"):
                print(f"{row.get('arm_id', '?'):>26}  ERROR: "
                      f"{str(row['error'])[:90]}")
                summary.setdefault("errors", []).append(row.get("arm_id"))
                continue
            checks = []
            for name, flag in (("dev", row.get("devices_ok")),
                               ("pin", row.get("pin_env_ok")),
                               ("bod", row.get("bodies_ok")),
                               ("bpd", row.get("bodies_per_device_ok")),
                               ("vb", row.get("vb_ok")),
                               ("chk", row.get("fwd_chunk_ok")),
                               ("back_c", row.get("back_chunk_unchanged_ok")),
                               ("kern", row.get("kernels_launched_ok")),
                               ("avail", row.get("kernels_were_available_ok")),
                               ("kill", row.get("kill_switch_off_ok")),
                               ("cal", row.get("calibration_absent_ok")),
                               ("md5", row.get("sino_md5_ok")),
                               ("rgn", row.get("region_nonzero_ok")),
                               ("rec", row.get("reconcile_ok"))):
                if flag is False:
                    checks.append(f"{name}:FAIL")
            vb_fwd = (row.get("fwd_view_batch_per_device") or [None])[0]
            spread = row.get("vcd_warm_spread")
            remainder = row.get("region_remainder_frac")
            spread_txt = f"{spread:.1%}" if spread is not None else "-"
            rem_txt = f"{remainder:.0%}" if remainder is not None else "-"
            print(f"{row['arm_id'].split('_', 2)[-1]:>26}"
                  f"{str(row.get('n_dev') or '-'):>3}"
                  f"{str(row.get('chunk') or '-'):>7}"
                  f"{str(vb_fwd if vb_fwd is not None else '-'):>6}"
                  f"{_fmt(row.get('vcd_cold'), '9.2f')}"
                  f"{_fmt(row.get('vcd_warm'), '9.2f')}"
                  f"{spread_txt:>8}"
                  f"{_fmt(row.get('forward_funnel_wall_per_pass_s'), '8.2f')}"
                  f"{_fmt(row.get('forward_funnel_dev_span_max_per_pass_s'), '9.2f')}"
                  f"{_fmt(row.get('forward_funnel_gap_per_pass_s'), '9.2f')}"
                  f"{_fmt(row.get('back_funnel_wall_per_pass_s'), '8.2f')}"
                  f"{rem_txt:>7}"
                  f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:>7.2f}"
                  f"{(','.join(checks) if checks else 'ok'):>22}")
            summary["rows"].append(dict(
                arm_id=row["arm_id"], n=row.get("n_dev"),
                arm_class=row.get("arm_class"), chunk=row.get("chunk"),
                warm=row.get("vcd_warm"), spread=spread,
                fwd_wall=row.get("forward_funnel_wall_per_pass_s"),
                fwd_dev_span=row.get("forward_funnel_dev_span_max_per_pass_s"),
                fwd_gap=row.get("forward_funnel_gap_per_pass_s"),
                back_wall=row.get("back_funnel_wall_per_pass_s"),
                reduce_wall=row.get("band_reduce_wall_per_pass_s"),
                fwd_vb=row.get("fwd_view_batch_per_device"),
                fwd_vb_observed=row.get("view_batch_observed"),
                remainder=row.get("region_remainder_frac"),
                fwd_share=row.get("forward_share_of_composed"),
                peak_gb=row.get("gpu_peak_bytes", 0) / 2 ** 30,
                checks=checks))

        # ── the value column, same count, against the shipped-chunk arm ──────
        for n in ns:
            base = by.get((geometry, n, "prod", SHIPPED_CHUNK))
            if not base:
                continue
            for row in live:
                if row.get("geometry") != geometry or row.get("n_dev") != n:
                    continue
                if row is base:
                    continue
                value = _rel_max(row.get("sample_path", ""),
                                 base.get("sample_path", ""))
                if value is None:
                    continue
                row["value_vs_shipped_chunk_same_n"] = value
                row["value_flagged"] = value > VALUE_FLAG
                summary.setdefault("value", []).append(
                    dict(arm_id=row["arm_id"], n=n, value=value,
                         flagged=value > VALUE_FLAG))
                if value > VALUE_FLAG:
                    print(f"   VALUE FLAG {row['arm_id']}: {value:.2e} vs the "
                          f"chunk-{SHIPPED_CHUNK} arm at n={n} "
                          f"(above {VALUE_FLAG:.0e})")

        # ── THE FORWARD ATTRIBUTION, the charter's question ──────────────────
        print(f"\n   == the forward region across counts, {geometry} "
              f"{cell[0]} ==")
        print(f"   {'arm':>12}" + "".join(f"{f'n={n}':>26}" for n in ns))
        print(f"   {'':>12}" + "".join(
            f"{'fwd_s  scale  gap_s  vb':>26}" for _n in ns))
        classes = [("prod", c) for c in chunks] + [("torchfwd", None)]
        for arm_class, chunk in classes:
            label = (f"kernel@{chunk}" if arm_class == "prod" else "torch_fwd")
            base = by.get((geometry, 1, arm_class, chunk))
            base_fwd = base.get("forward_funnel_wall_per_pass_s") if base else None
            cells_out, entry = [], dict(arm_class=arm_class, chunk=chunk,
                                        label=label, per_count={})
            for n in ns:
                row = by.get((geometry, n, arm_class, chunk))
                if not row:
                    cells_out.append(f"{'-':>26}")
                    continue
                fwd = row.get("forward_funnel_wall_per_pass_s")
                gap = row.get("forward_funnel_gap_per_pass_s")
                vb = (row.get("fwd_view_batch_per_device") or [None])[0]
                scale = (base_fwd / fwd) if (base_fwd and fwd) else None
                entry["per_count"][n] = dict(
                    fwd_wall=fwd, fwd_scale_vs_n1=scale, fwd_gap=gap,
                    fwd_dev_span=row.get(
                        "forward_funnel_dev_span_max_per_pass_s"),
                    fwd_vb=vb, composed=row.get("vcd_warm"),
                    fwd_share=row.get("forward_share_of_composed"))
                cells_out.append(
                    f"{_fmt(fwd, '7.2f')}{(f'{scale:.2f}x' if scale else '-'):>7}"
                    f"{_fmt(gap, '7.2f')}{str(vb if vb is not None else '-'):>5}")
            summary["attribution"].append(entry)
            print(f"   {label:>12}" + "".join(cells_out))
        print("   scale = the n=1 forward region over this count's; >1 is "
              "speedup, 1.00x is FLAT.")
        print("   gap   = forward host wall MINUS the largest per-device event "
              "span, per pass:")
        print("           the fan-out / broadcast / assembly around the kernel "
              "(protocol 10).")
        print("   vb    = the realized forward view batch per device at the "
              "full-pixel set.")
        print("   HYPOTHESIS SIGNATURES (stated, not adjudicated here):")
        print("     (a) kernel-specific  -- torch_fwd scales where kernel@* is "
              "flat.")
        print("     (b) realized batch   -- vb moves with n, and fwd_s tracks "
              "the chunk ladder.")
        print("     (c) driver fan-out   -- gap grows with n while the device "
              "span falls; torch_fwd flat too.")

        # ── the mg1 anchor reproduction (this job's validity check) ──────────
        print(f"\n   == mg1 anchor: the chunk-{SHIPPED_CHUNK} arms are mg1's "
              f"production arms rebuilt ==")
        anchor = []
        for n in ns:
            row = by.get((geometry, n, "prod", SHIPPED_CHUNK))
            if not row:
                continue
            recorded = MG1_FORWARD_REGION_S.get((geometry, n))
            entry = dict(n=n, fwd=row.get("forward_funnel_wall_per_pass_s"),
                         fwd_mg1=recorded,
                         back=row.get("back_funnel_wall_per_pass_s"),
                         back_mg1=MG1_BACK_REGION_S.get((geometry, n)),
                         composed=row.get("vcd_warm"),
                         composed_mg1=MG1_COMPOSED_S.get((geometry, n)))
            anchor.append(entry)
            print(f"   n={n}: fwd {_fmt(entry['fwd'], '.2f')} s (mg1 "
                  f"{entry['fwd_mg1']}), back {_fmt(entry['back'], '.2f')} s "
                  f"(mg1 {entry['back_mg1']}), composed "
                  f"{_fmt(entry['composed'], '.2f')} s (mg1 "
                  f"{entry['composed_mg1']})")
        summary["mg1_anchor"] = anchor

        # ── the drift arms ───────────────────────────────────────────────────
        drift = next((r for r in live if r.get("geometry") == geometry
                      and r.get("is_drift")), None)
        first = by.get((geometry, 1, "prod", SHIPPED_CHUNK))
        if drift and first and first.get("vcd_warm"):
            ratio = drift["vcd_warm"] / first["vcd_warm"]
            fwd_ratio = None
            if first.get("forward_funnel_wall_per_pass_s"):
                fwd_ratio = (drift.get("forward_funnel_wall_per_pass_s", 0)
                             / first["forward_funnel_wall_per_pass_s"])
            summary["drift"] = dict(composed_ratio=ratio,
                                    forward_ratio=fwd_ratio,
                                    within_arm_spread=max(
                                        first.get("vcd_warm_spread") or 0.0,
                                        drift.get("vcd_warm_spread") or 0.0))
            print(f"   DRIFT (the n=1 anchor re-run at the job tail): composed "
                  f"{ratio:.3f}x, forward region "
                  f"{(f'{fwd_ratio:.3f}x' if fwd_ratio else '-')}, "
                  f"within-arm spread "
                  f"{summary['drift']['within_arm_spread']:.1%}.  Every "
                  f"scaling number above is read against a reference measured "
                  f"at the job's START, so this ratio is the bias that "
                  f"ordering could not cancel.")
        summaries.append(summary)

    # -- the instrument backend caveat -----------------------------------------
    backends = {r.get("event_backend") for r in live if r.get("event_backend")}
    if any(b != "cuda_events" for b in backends):
        print(f"\nNOTE: event backend {sorted(backends)}.  On the CPU path the "
              f"per-device span map collapses to a single 'cpu' key whose span "
              f"IS the host wall, so gap = wall - wall*n is NEGATIVE by "
              f"construction and prices nothing.  The gap column is meaningful "
              f"only under cuda_events.")
    if any(r.get("torchfwd_degenerate") for r in live):
        print("NOTE: at least one torchfwd arm is DEGENERATE -- the shipped "
              "selection already returned the torch forward there (the CPU "
              "smoke), so that arm discriminates nothing and only exercises "
              "the binding path.")

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
    print(f"\nthrottle rule: sw_power_cap at normal temperature is recorded and "
          f"KEPT; {len(rerun)} row(s) hot AND clock-depressed -> re-run: "
          f"{rerun}")
    return summaries, rerun


def main():
    geometries, counts, chunks = selected_plan()
    phase0, phase1, phase2 = build_plan(geometries, counts, chunks)
    if "--dry-run" in sys.argv:
        measured = [c for c in phase0 + phase1 + phase2
                    if c["arm_class"] != "generator"]
        generators = [c for c in phase0 if c["arm_class"] == "generator"]
        print(f"mg5 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  geometries {geometries}, counts {counts}, forward chunk "
              f"ladder {chunks}, warm repeats {WARM_REPEATS}, iterations "
              f"{VCD_ITERATIONS}, device {DEVICE}")
        for label, plan in (("phase0", phase0), ("phase1", phase1),
                            ("phase2", phase2)):
            for cfg in plan:
                print(f"  [{label}] {cfg['arm_id']:<44} "
                      f"n={cfg['n_dev']} chunk={cfg['chunk']}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg5_fwd_attrib_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg5 forward attribution on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, counts {counts}, chunks {chunks} -> {out_path}",
          flush=True)
    rows = []
    # Rows write INCREMENTALLY (protocol 11): a truncated job still yields its
    # n=1 reference arms, which is why phase 0 runs first.
    with open(out_path, "w") as sink:
        for phase, plan in (("phase0", phase0), ("phase1", phase1),
                            ("phase2", phase2)):
            for cfg in plan:
                print(f"  [{phase}] {cfg['arm_id']}", flush=True)
                row = run_one(dict(cfg, phase=phase))
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        summaries, rerun = summarize(rows, geometries, counts, chunks, out_path)
        for summary in summaries:
            sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.write(json.dumps(dict(thermal_rerun=rerun)) + "\n")
        sink.flush()
    # TWO PRECEDENTS CONFLICT HERE, and the choice is recorded rather than
    # silent.  mg1 KEEPS its shared artifacts by review ruling, because the
    # forward kernel's atomics make a regenerated artifact non-identical at the
    # e-7 class, so the md5s in its rows are re-verifiable only against those
    # exact files.  mg4 DELETES unless MG4_KEEP_ARTIFACTS=1.  mg5 follows mg4:
    # its value column is internal to the job (every arm is diffed against
    # another arm of this job, at the same count), nothing outside mg5
    # re-verifies these md5s, and the pair is ~8 GB of scratch.  Export
    # MG5_KEEP_ARTIFACTS=1 for mg1's behavior.
    if os.environ.get("MG5_KEEP_ARTIFACTS", "0") != "1":
        for geometry in geometries:
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
