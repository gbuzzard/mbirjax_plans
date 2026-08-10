"""mg7 -- the CONE VIEW-BATCH probe: does the realized per-device view batch
FALL with the device count where the 2 GiB transient cap is NOT binding?

THE QUESTION.  At the 1024-class cells the transient budget is pinned at its
2 GiB cap at every device count, so the count-divided budget cannot express
itself and mg1 measured the same realized batch (52) at n = 1, 2 and 4.  The
384- and 512-class cells are the cells where the cap stops binding, so they are
where the budget's 1/n proportionality can show.  This probe measures the
realized forward view batch at those cells, against parallel at the 512 cell as
the control.

THE CODE READING, verified against the source and quoted here so a drift in
either place is visible.

    Projectors._transient_budget_bytes  (projectors.py:231-249) scales the
    budget by the PER-DEVICE view shard, then floors and caps it:

        n_dev = (self.model.sino_placement.n_devices if n_devices is None
                 else int(n_devices))
        local_views = -(-int(num_views) // n_dev)
        sino_bytes = local_views * num_rows * num_channels * 4
        return max(self.VIEW_BATCH_TRANSIENT_FLOOR_BYTES,
                   min(self.VIEW_BATCH_TRANSIENT_BUDGET_BYTES,
                       self.VIEW_BATCH_SINO_MULTIPLE * sino_bytes))

    with VIEW_BATCH_TRANSIENT_BUDGET_BYTES = 2 * 2**30 (the cap),
    VIEW_BATCH_TRANSIENT_FLOOR_BYTES = 256 * 2**20 (the floor) and
    VIEW_BATCH_SINO_MULTIPLE = 8 (projectors.py:222-224).  So the budget falls
    as 1/n between the floor and the cap.

    view_batch_charge  (projectors.py:309-350) divides that budget by the
    body's own per-view cost and clamps to the body's nominal chunk:

        bytes_per_view, view_chunk = cost(num_pixels, band_cols, args)
        if nominal is None:
            nominal = view_chunk
        budget = self._transient_budget_bytes(n_devices=n_devices)
        cap = budget // max(1, int(bytes_per_view))
        return max(1, min(int(nominal), int(cap))), int(bytes_per_view)

    CONE's forward cost  (triton_cone.py:685-693) reads the detector-row count
    out of the PARAMS, so it does not shrink with n:

        plane_bytes = 4 * int(args['num_channels']) * int(args['num_rows_r'])
        return 48 * int(num_pixels) + plane_bytes, CONE_FWD_VIEW_CHUNK

    PARALLEL's forward cost  (triton_parallel.py:478-484) reads the CALL's
    column count, which under sharding is the slice band, so it does shrink:

        plane_bytes = 4 * int(args['num_channels']) * int(num_value_cols)
        return 16 * int(num_pixels) + plane_bytes, PARALLEL_FWD_VIEW_CHUNK

    Both chunks ship at 128 (triton_cone.py:188, triton_parallel.py:130).  mg7
    PINS NOTHING: the question is what the SHIPPED constant realizes, so the
    arms read both constants, assert they are 128, and re-read them at the end
    to prove nothing moved.  (mg5 swept the forward chunk; mg7 must not, or the
    realized batch would be the harness's choice rather than the code's.)

THE REGISTERED PREDICTION, computed from those cost functions before the job
runs, so the readout is a test and not a fishing trip.  num_pixels is the FULL
index set, ``gen_full_indices(recon_shape).shape[0]`` under the ROR mask --
64652 at the 384 cell (recon 288x288x336) and 115164 at the 512 cell (recon
384x384x448).  ``cols`` is the forward's band width, ceil(recon_shape[2] / n).

  cone (384,336,288), bytes_per_view = 48*64652 + 4*288*336 = 3,490,368 (n-free)
    n=1  8x sino = 8*384*336*288*4 = 1,189,085,184 -> budget 1.189 GB (uncapped)
         cap = 1,189,085,184 // 3,490,368 = 340   -> batch min(128, 340) =  128
    n=2  8x sino =                 594,542,592 -> budget 0.595 GB (uncapped)
         cap =   594,542,592 // 3,490,368 = 170   -> batch min(128, 170) =  128
    n=4  8x sino =                 297,271,296 -> budget 0.297 GB (uncapped)
         cap =   297,271,296 // 3,490,368 =  85   -> batch min(128,  85) =   85

  cone (512,448,384), bytes_per_view = 48*115164 + 4*384*448 = 6,216,000 (n-free)
    n=1  8x sino = 2,818,572,288 > 2 GiB -> budget CAPPED at 2,147,483,648
         cap = 2,147,483,648 // 6,216,000 = 345   -> batch min(128, 345) =  128
    n=2  8x sino = 1,409,286,144 -> budget 1.409 GB (uncapped)
         cap = 1,409,286,144 // 6,216,000 = 226   -> batch min(128, 226) =  128
    n=4  8x sino =   704,643,072 -> budget 0.705 GB (uncapped)
         cap =   704,643,072 // 6,216,000 = 113   -> batch min(128, 113) =  113

  parallel (512,448,384) -- THE CONTROL, bytes_per_view = 16*115164 + 4*384*cols
    n=1  cols=448  bpv = 1,842,624 + 688,128 = 2,530,752
         cap = 2,147,483,648 // 2,530,752 = 848   -> batch min(128, 848) =  128
    n=2  cols=224  bpv = 1,842,624 + 344,064 = 2,186,688
         cap = 1,409,286,144 // 2,186,688 = 644   -> batch min(128, 644) =  128
    n=4  cols=112  bpv = 1,842,624 + 172,032 = 2,014,656
         cap =   704,643,072 // 2,014,656 = 349   -> batch min(128, 349) =  128

THE MODEL'S OWN VALIDATION, which is why the table above is worth trusting.
Run the same arithmetic at cone (1024,1008,992): num_pixels 771240, plane
4*992*1008, bytes_per_view 41,019,264; 8x the per-device sinogram exceeds 2 GiB
at n = 1, 2 AND 4, so the budget is 2,147,483,648 at every count and the cap is
2,147,483,648 // 41,019,264 = 52 -- which is exactly the batch mg1 MEASURED at
every count there.  The arithmetic reproduces a recorded measurement to the
digit before mg7 spends a GPU-minute.

THREE PLACES THE CODE CONTRADICTS THE CHARTER PROMPT'S READING.  Reported, not
worked around; the arms measure what the code does and the analysis decides
what it means.

  (1) THE REALIZED BATCH DOES NOT FALL AS 1/n.  The budget does, but the batch
      is min(chunk, cap) and the SHIPPED CHUNK OF 128 FLOORS IT until the cap
      dips below 128.  Registered: 128 / 128 / 85 at cone 384 and 128 / 128 /
      113 at cone 512 -- flat through n=2, then a 34 percent and a 12 percent
      fall at n=4.  Nothing here is proportional to 1/n at any measured count.
      The prompt's "budget approximately 2.15 / 1.41 / 0.70 GB" is right (it is
      this file's budget column); the step from budget to batch is where the
      1/n reading breaks.

  (2) PER-DEVICE LAUNCH COUNTS DO NOT GROW WITH n; they fall and then flatten.
      Launches per device per forward call = ceil(local_views / batch), and
      local_views itself falls as 1/n:
        cone 512:     n=1 ceil(512/128)=4,  n=2 ceil(256/128)=2,
                      n=4 ceil(128/113)=2   (it would be 1 if the cap did not bind)
        cone 384:     3, 2, 2               (2 instead of 1 at n=4)
        parallel 512: 4, 2, 1               -- the control, no cap effect anywhere
      What DOES grow is the TOTAL launch count summed over devices: cone goes
      4 -> 4 -> 8 while parallel goes 4 -> 4 -> 4.  The cap's cost at n=4 is a
      DOUBLING of the forward launches the job issues, and a per-device launch
      count 2x the control's.  That is the real, smaller, sharper effect, and
      the summary prints both the per-device and the summed columns.

  (3) A "BATCH EVENT" IS NOT A LAUNCH.  ``_effective_view_batch`` is called ONCE
      per ``sparse_forward_project_view_range`` call (projectors.py:382), and
      that one call then loops ``for v in range(v0, v1, vb_size)``
      (projectors.py:386) issuing ceil((v1-v0)/vb_size) body launches.  So the
      observer's event COUNT is the number of view-range CALLS, not the launch
      count, and the launch count must be DERIVED as
      sum over events of ceil(local_views / batch).  Every row carries
      ``launch_count_is_derived`` and the assumption it rests on -- the banded
      driver's contract that each owner gets ONE contiguous real-view span
      (projectors.py:367-368, tomography_model.py:417-418) -- rather than
      passing a derived number off as a measured one.

ONE MORE THING THE READOUT MUST NOT CONFUSE, and the reason every row carries
BOTH numbers.  The registered table is the batch at the FULL pixel set, which
is what mg1's static probe reads and what the recon's initial error-sinogram
forward actually uses (tomography_model.py:1256 -> forward_project ->
sparse_forward_project over gen_full_indices).  The VCD subset steps project
FEWER pixels: the shipped partition_sequence [2, 4, 6] over granularity
[1,2,4,8,16,32,64,...] visits granularities 4, 16 and 64 in a 3-iteration run,
so their bytes_per_view is 3x to 8x smaller and their cap never binds --
predicted batch 128 for every geometry, cell and count on the subset steps.
The OBSERVED distribution should therefore read, per device:

    cone 512 n=4    a 113 population (the full-pixel forward) and a 128
                    population (the subset forwards)
    cone 512 n=1,2  a single 128 population
    parallel 512    a single 128 population at every count

A second batch value below 128 appearing ONLY on cone and ONLY at n=4 is the
observation this probe exists to make; its absence refutes the reading.

THE ARMS.  Nine measured arms plus three untimed generator arms, all at the
SHIPPED chunk constants -- no pinning anywhere, in either direction.

    cone     (384,336,288)  n = 1, 2, 4
    cone     (512,448,384)  n = 1, 2, 4
    parallel (512,448,384)  n = 1, 2, 4     the control

WHAT EVERY ARM CARRIES (protocols 1, 3, 6, 7, 9, 10, 11).  mg1's three-region
instrument (``RegionInstrument`` / ``attach_instrument``), reset after the
discarded cold pass; mg1's static view-batch probe (``_view_batch_static``) and
its realized-batch observer (``observe_view_batches``) -- BOTH, because the
static probe says what the formula gives and the observer says what the passes
chose; the live ``_transient_budget_bytes()`` at the realized count;
``bytes_per_view`` and the chunk the cost function saw; three warm repeats with
the median and spread; the forward and back region walls and per-device event
spans; per-device peak bytes; the GPU health sample with the hot-AND-clock-
depressed re-run rule; one fresh subprocess per arm with the count pinned
through MBIRTORCH_NUM_DEVICES (popped, then set, in the subprocess env) and the
realized device list asserted AFTER the timed call.

THE IMPORT OF mg1_readout IS DELIBERATE, as it was for mg5.  mg7's whole
deliverable is a realized view batch set against mg1's recorded 52 at the 1024
cells, so the probe and the observer must be mg1's own and not a copy that has
drifted.  The cost is a staging dependency: mg1_readout.py MUST be staged
beside this file, and the import failure below names that remedy.

ARM ORDER.  Protocol 11 puts the n=1 reference arms first, so a truncated job
still yields the references every scaling reading is taken against.  Protocol
9's reversal is then applied to the count axis WITHIN each coordinate: the
first coordinate runs n=2 then n=4, the second n=4 then n=2, the third n=2 then
n=4, so count position and time position are decorrelated.

    PHASE 0   the three generators, then every n=1 arm
    PHASE 1   the n>1 arms, counts blocked per coordinate, direction alternating

NO VERDICT IS PRINTED.  The two discriminating signatures are STATED beside the
numbers and the decision is made in analysis -- the house rule, mg4's knee rule
and mg5's attribution both.

THE GATHER CONTRACT (nt2_local_shard_check.py).  ``Shards.gather()`` ALREADY
returns numpy; re-detaching its result is the recorded failure that cost the
nightly's first 4-GPU trial all 32 of its n>1 rows.  Every host exit here goes
through mg1's ``_to_numpy``, which never re-detaches a gather.

Run:
    <torch python> mg7_conebatch.py          on a 4-GPU node (mg7_gautschi.sbatch)
    python mg7_conebatch.py --dry-run        anywhere: print the arm plan
    MG7_SMOKE=1 python mg7_conebatch.py      the local CPU smoke
    python mg7_conebatch.py --help

Environment (export from the SUBMITTING SHELL; never in an sbatch
--export=ALL,VAR=a,b,c list, which slurm splits on commas).  List values are
parsed STRICTLY: an unrecognized token is a hard error.
    P0_TORCH_PYTHON=<python>          interpreter for the arm subprocesses
    MG7_RESULTS=<dir>                 where the jsonl and the artifacts go
    MG7_GEOMS=cone,parallel           subset of the geometries (the ONLY split)
    MG7_COUNTS=1,2,4                  subset of the device counts
    MG7_ITERATIONS=3                  VCD iterations per recon
    MG7_WARM_REPEATS=3                warm repeats after the discarded cold pass
    MG7_KEEP_ARTIFACTS=1              keep the shared sinograms after the run
    MG7_SMOKE=1                       the local smoke (tiny cells, few iters)
    MG7_DEVICE=cpu                    smoke device
    MG7_SMOKE_CPU_N2=1                smoke only: a 2-virtual-device CPU arm
                                      that exercises the sharded path
"""

import json
import math
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
                             _launch_key_counts, _md5, _strict_subset,
                             _to_numpy, _view_batch_static, _weights,
                             row_is_hot, sample_gpu_health, worst_health,
                             CLOCK_DEPRESSED_FRAC)
    assert RegionInstrument is not None
except ImportError as exc:                                        # noqa: BLE001
    raise SystemExit(
        f"mg7 could not import mg1_readout ({exc}).  mg7 reuses mg1's region "
        f"instrument, its static view-batch probe and its realized-batch "
        f"observer VERBATIM, so its batch readings are comparable with mg1's "
        f"recorded 52 at the 1024 cells; mg1_readout.py must be staged in the "
        f"same directory as this file.")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# The cells, read from mg4's families: (384,336,288) and (512,448,384) are the
# fourth and fifth rungs of mg4's PARALLEL_LADDER (mg4_ladder.py:158-159) and
# the second and third of its CONE_SPOT_CHECK (mg4_ladder.py:167).  Both cells
# therefore already carry mg4 walls at both geometries.
CELL_384 = (384, 336, 288)
CELL_512 = (512, 448, 384)
COORDINATES = (("cone", CELL_384, (1, 2, 4)),
               ("cone", CELL_512, (1, 2, 4)),
               ("parallel", CELL_512, (1, 2, 4)))
GEOMETRIES = ("cone", "parallel")
COUNTS = (1, 2, 4)

# The SHIPPED chunk, in BOTH directions and at BOTH geometries.  mg7 never sets
# it -- see the module docstring.  Every arm asserts the constant is this value
# on entry and unchanged on exit.
SHIPPED_CHUNK = 128

# The budget constants, mirrored from Projectors so the harness can price a
# count it is not currently running (projectors.py:222-224).  An arm asserts the
# mirror against the live class attributes, so a change there fails loudly here
# instead of silently invalidating the registered table.
BUDGET_CAP_BYTES = 2 * 2 ** 30
BUDGET_FLOOR_BYTES = 256 * 2 ** 20
BUDGET_SINO_MULTIPLE = 8

# THE REGISTERED PREDICTION (the module docstring carries the arithmetic).
# Keyed (geometry, num_views, n_dev) -> the realized forward view batch per
# device at the FULL pixel set.  Registered in advance; each arm ALSO recomputes
# it from the live cost function and fails the check if the two disagree, so a
# source change cannot quietly turn a prediction into a postdiction.
PREDICTED_FWD_BATCH = {
    ("cone", 384, 1): 128, ("cone", 384, 2): 128, ("cone", 384, 4): 85,
    ("cone", 512, 1): 128, ("cone", 512, 2): 128, ("cone", 512, 4): 113,
    ("parallel", 512, 1): 128, ("parallel", 512, 2): 128,
    ("parallel", 512, 4): 128,
}
# Per-device forward launches per view-range call, ceil(local_views / batch).
PREDICTED_LAUNCHES_PER_DEVICE = {
    ("cone", 384, 1): 3, ("cone", 384, 2): 2, ("cone", 384, 4): 2,
    ("cone", 512, 1): 4, ("cone", 512, 2): 2, ("cone", 512, 4): 2,
    ("parallel", 512, 1): 4, ("parallel", 512, 2): 2,
    ("parallel", 512, 4): 1,
}
# The batch every VCD SUBSET step is predicted to realize, at every cell, count
# and geometry: the subsets carry 4x to 64x fewer pixels, so their cap never
# binds (see the module docstring's distribution note).
PREDICTED_SUBSET_BATCH = 128

SMOKE = os.environ.get("MG7_SMOKE", "0") == "1"
# Two tiny cells so the cone coordinate really carries TWO cells and the
# cross-cell machinery runs.  The smoke's job is that every path executes, not
# that any number means anything.
SMOKE_CELLS = ((8, 24, 20), (16, 24, 20), (16, 24, 20))
DEVICE = os.environ.get("MG7_DEVICE", "cpu" if SMOKE else "cuda")

VCD_ITERATIONS = int(os.environ.get("MG7_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13             # kb3's / mg1's / mg5's seed, so the arms stay comparable
SAMPLE_ROWS = 16
WARM_REPEATS = max(1, int(os.environ.get("MG7_WARM_REPEATS",
                                         "2" if SMOKE else "3")))

# mg1's recorded composed warm walls at the 512 cells (findings §1.2), printed
# beside this job's so a reader can see at a glance whether the node is behaving.
# NOT a gate: mg7 crosses a job boundary and these are context, not a check.
MG1_COMPOSED_S = {("cone", 512): {1: 2.74, 2: 2.78, 4: 4.07},
                  ("parallel", 512): {1: 1.91, 2: 1.57, 4: 2.52}}
# The realized batch mg1 measured at the 1024 cells, at EVERY count -- the
# cap-bound regime this probe is the uncapped counterpart of.
MG1_BATCH_AT_1024 = 52

# Reconciliation tolerance, mg1's constant: the bracketed regions are a proper
# subset of the composed wall.
RECONCILE_SLACK = 0.02

RESULTS_DIR = os.environ.get(
    "MG7_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]

# Per geometry: the kernel module carrying the chunk constants mg7 READS (and
# never writes), and the wrapper names the body checks match against.  kb2's
# SPEC table (kb2_vbsweep.py:108) with the setattr half deliberately absent.
SPEC = {
    "parallel": dict(kernel_module="mbirtorch.triton_parallel",
                     fwd_chunk_const="PARALLEL_FWD_VIEW_CHUNK",
                     back_chunk_const="PARALLEL_BACK_VIEW_CHUNK",
                     fwd_cost="_parallel_forward_view_batch_cost"),
    "cone": dict(kernel_module="mbirtorch.triton_cone",
                 fwd_chunk_const="CONE_FWD_VIEW_CHUNK",
                 back_chunk_const="CONE_BACK_VIEW_CHUNK",
                 fwd_cost="_cone_forward_view_batch_cost"),
}
# ──────────────────────────────────────────────────────────────────────────────


def coordinates():
    """The (geometry, cell, counts) triples, smoke cells substituted in place so
    the cone coordinate still carries two distinct cells."""
    out = []
    for index, (geometry, cell, ns) in enumerate(COORDINATES):
        out.append((geometry, SMOKE_CELLS[index] if SMOKE else cell, ns))
    return out


def selected_plan():
    """(geometries, counts), each narrowed by its env knob.

    MG7_GEOMS is the ONLY split axis: every reading here compares counts within
    a coordinate, so the count axis may never cross a job boundary."""
    chosen = _strict_subset("MG7_GEOMS", set(GEOMETRIES))
    geometries = [g for g in GEOMETRIES if g in chosen]
    if SMOKE and not os.environ.get("MG7_COUNTS", "").strip():
        # The env pin is a CUDA-only mechanism (the policy short-circuits at
        # `visible < 2`), so a pinned n>1 arm on CPU would silently measure
        # n=1.  The smoke therefore runs n=1 plus the dedicated CPU virtual
        # 2-device arms, which pin by device LIST and say so.
        counts = [1]
    else:
        keep = _strict_subset("MG7_COUNTS", set(COUNTS), int)
        counts = [n for n in COUNTS if n in keep]
    return geometries, counts


# ── the registered arithmetic, recomputable ───────────────────────────────────
def budget_bytes_for(cell, n_dev):
    """``Projectors._transient_budget_bytes`` mirrored (projectors.py:231-249),
    so the harness can price the budget for a count it is not running.  An arm
    asserts this against the LIVE method at its own count."""
    num_views, num_rows, num_channels = cell
    local_views = -(-int(num_views) // int(n_dev))
    sino_bytes = local_views * int(num_rows) * int(num_channels) * 4
    return max(BUDGET_FLOOR_BYTES,
               min(BUDGET_CAP_BYTES, BUDGET_SINO_MULTIPLE * sino_bytes))


def predict_fwd_batch(geometry, cell, recon_shape, num_pixels, args, n_dev):
    """Recompute the registered prediction from the LIVE kernel cost function.

    Imported from the kernel module directly rather than read off the bound
    body, so this works on the CPU smoke (where no kernel body is bound) and so
    the prediction is the KERNEL's even when the availability probe declined.
    Returns the record the arm records and checks."""
    import importlib

    spec = SPEC[geometry]
    module = importlib.import_module(spec["kernel_module"])
    cost = getattr(module, spec["fwd_cost"])
    cols = int(-(-int(recon_shape[2]) // int(n_dev)))
    bytes_per_view, chunk = cost(int(num_pixels), cols, args)
    budget = budget_bytes_for(cell, n_dev)
    cap_views = budget // max(1, int(bytes_per_view))
    batch = max(1, min(int(chunk), int(cap_views)))
    local_views = int(-(-int(cell[0]) // int(n_dev)))
    return dict(predicted_budget_bytes=int(budget),
                predicted_budget_is_capped=bool(
                    BUDGET_SINO_MULTIPLE * (local_views * cell[1] * cell[2] * 4)
                    > BUDGET_CAP_BYTES),
                predicted_budget_is_floored=bool(
                    BUDGET_SINO_MULTIPLE * (local_views * cell[1] * cell[2] * 4)
                    < BUDGET_FLOOR_BYTES),
                predicted_bytes_per_view=int(bytes_per_view),
                predicted_chunk=int(chunk),
                predicted_cap_views=int(cap_views),
                predicted_cap_binds=bool(int(cap_views) < int(chunk)),
                predicted_fwd_batch=int(batch),
                predicted_fwd_cols=cols,
                predicted_local_views=local_views,
                predicted_launches_per_device=int(
                    math.ceil(local_views / max(1, batch))))


# ── staged-artifact mechanics (protocol 5's md5 discipline) ───────────────────
def _cell_tag(geometry, cell):
    return f"{geometry}_{cell[0]}x{cell[1]}x{cell[2]}"


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg7_sino_{_cell_tag(geometry, cell)}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id):
    return os.path.join(RESULTS_DIR, f"_mg7_sample_{arm_id}.npy")


# ── the chunk constants: READ ONLY (see the module docstring) ─────────────────
def read_shipped_chunks(geometry):
    """The geometry's forward and back view-chunk constants, READ and never
    written.

    mg5 swept the forward constant by setattr on the kernel module
    (kb2_vbsweep.py:369's mechanism).  mg7 must NOT: its whole question is what
    the realized batch does under the SHIPPED constant as the budget divides, so
    a pinned chunk would make the answer the harness's rather than the code's.
    The arm reads both constants here, asserts each is SHIPPED_CHUNK, and
    re-reads them after the timed passes to prove nothing in the run moved them.
    """
    import importlib

    spec = SPEC[geometry]
    module = importlib.import_module(spec["kernel_module"])
    return (module,
            int(getattr(module, spec["fwd_chunk_const"])),
            int(getattr(module, spec["back_chunk_const"])))


def _per_device_body_names(model):
    """The bodies actually bound, PER DIRECTION PER DEVICE.  The selection hook
    says what was chosen; these say what the driver holds, one compiled
    instance per device (projectors.py:261).  ``maybe_compile`` prefixes a
    compiled torch body with ``compiled_`` (projectors.py:129) and returns a
    kernel wrapper untouched, so both forms are recognizable by name."""
    pf = model.projector_functions
    return ([getattr(b, "__name__", str(b)) for b in pf._fwd_body_per_dev],
            [getattr(b, "__name__", str(b)) for b in pf._back_body_per_dev])


def _local_view_counts(model, cell):
    """Each view-owner's REAL view count, from the placement itself.

    ``padded_shard_ranges`` (\\_sharding.py:127-142) returns
    ``(device, (start, end), n_valid)`` per owner, and ``_banded_setup``
    (tomography_model.py:417-418) builds the forward's view spans from exactly
    that ``n_valid`` -- so this is the span each ``sparse_forward_project_view_
    range`` call walks.  A trivial placement has no padded size; there the whole
    view axis is the single owner's span."""
    placement = model.sino_placement
    try:
        return [int(n_valid)
                for _dev, (_v0, _v1), n_valid in placement.padded_shard_ranges()]
    except Exception:                                             # noqa: BLE001
        n_dev = int(placement.n_devices)
        return [int(-(-int(cell[0]) // n_dev))] * n_dev


def summarize_observed(observed, local_views, passes):
    """Fold the observer's ``{key: {batch: count}}`` into the per-direction,
    per-device readout the charter asks for.

    THE DISTINCTION THAT MATTERS, and the reason this is a function with a
    docstring rather than an inline comprehension.  Each entry in ``observed``
    is one call to ``_effective_view_batch``, which the driver makes ONCE per
    ``sparse_forward_project_view_range`` call (projectors.py:382) -- so the
    entry count is the number of view-range CALLS, a BATCH EVENT, not a launch.
    That one call then loops ``for v in range(v0, v1, vb_size)``
    (projectors.py:386), issuing ``ceil(span / batch)`` body launches.  The
    launch count is therefore DERIVED here, and the row says so.

    The derivation rests on one assumption, named so it can be checked: each
    owner walks ONE contiguous real-view span of ``local_views[dev]`` views --
    the banded driver's stated contract (projectors.py:367-368, and
    tomography_model.py:417-418 where the spans are built).
    """
    out = {}
    for key, buckets in observed.items():
        items = sorted((int(b), int(c)) for b, c in buckets.items())
        events = sum(count for _batch, count in items)
        index = None
        if key.startswith("fwd_dev") or key.startswith("back_dev"):
            try:
                index = int(key.rsplit("dev", 1)[1])
            except ValueError:
                index = None
        span = (local_views[index]
                if index is not None and index < len(local_views)
                else (local_views[0] if local_views else 0))
        launches = sum(count * math.ceil(span / max(1, batch))
                       for batch, count in items)
        modal = max(items, key=lambda pair: pair[1])[0] if items else None
        out[key] = dict(
            distribution=items,
            batch_values=[batch for batch, _c in items],
            batch_min=(items[0][0] if items else None),
            batch_max=(items[-1][0] if items else None),
            batch_modal=modal,
            events_total=events,
            events_per_pass=(events / passes if passes else None),
            launches_total=launches,
            launches_per_pass=(launches / passes if passes else None),
            span_views_assumed=span)
    return out


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
    three-region instrument attached and its realized-batch observer installed.

    ORDERING NOTE, load-bearing (mg1's and mg5's, unchanged).  Every
    projector-dependent check runs AFTER the cold pass.  The automatic branch
    settles the layout inside the first ``recon`` call, and a settle that
    changes the count calls ``_install_device_layout`` -> ``create_projectors``
    (tomography_model.py:842), which REPLACES ``model.projector_functions``.  A
    view-batch or body reading taken before that would describe a one-device
    projector set under an n-device label.  The instrument is immune: it shadows
    instance and module attributes the engine resolves at call time.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    n_dev = cfg.get("n_dev")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    # Read (never write) the shipped chunk constants BEFORE the model is built,
    # so nothing downstream can be blamed for a value read late.
    kernel_module, shipped_fwd_chunk, shipped_back_chunk = \
        read_shipped_chunks(geometry)

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
    # Protocol 6: the calibration mode owns max_memory_allocated, so it must be
    # absent everywhere in mg7 -- the per-device peaks are read off that counter.
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    # PRODUCTION BODIES ONLY: mg7 has no torch-body arm, so the kill switch must
    # be off at every arm and the bound bodies must be the kernels on CUDA.
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))
    if cuda:
        result["pin_env_ok"] = (
            os.environ.get("MBIRTORCH_NUM_DEVICES") == str(n_dev))
    # NO CHUNK PINNING ANYWHERE: both constants must be the shipped 128 on entry.
    result["chunks_are_shipped_ok"] = (shipped_fwd_chunk == SHIPPED_CHUNK
                                       and shipped_back_chunk == SHIPPED_CHUNK)
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
    want_fwd_kernel, want_back_kernel = expect_kernels
    result.update(fwd_body=fwd_name, back_body=back_name,
                  fwd_body_per_device=fwd_per_dev,
                  back_body_per_device=back_per_dev)
    result["bodies_per_device_ok"] = (
        len(fwd_per_dev) == n_realized and len(back_per_dev) == n_realized
        and all(("triton" in name) == want_fwd_kernel for name in fwd_per_dev)
        and all(("triton" in name) == want_back_kernel for name in back_per_dev))
    result["bodies_ok"] = (
        result["bodies_per_device_ok"]
        and ("triton" in fwd_name) == want_fwd_kernel
        and ("triton" in back_name) == want_back_kernel)

    # ── THE STATIC PROBE: mg1's, verbatim (protocol 3) ───────────────────────
    # The realized view batch per direction per device at the FULL pixel set,
    # against the formula of the body EXPECTED to be bound.
    vb_record, vb_ok = _view_batch_static(model, expect_kernels)
    result.update(vb_record)
    result["vb_ok"] = vb_ok
    realized_fwd_per_dev = list(vb_record["fwd_view_batch_per_device"])
    result["fwd_batch_realized"] = (realized_fwd_per_dev[0]
                                    if realized_fwd_per_dev else None)
    result["fwd_batch_uniform_across_devices"] = (
        len(set(realized_fwd_per_dev)) <= 1)

    # ── THE LIVE BUDGET at the realized count, and the mirror check ──────────
    pf = model.projector_functions
    args = model._view_batch_args()
    num_pixels = int(vb_record["num_pixels_full"])
    fwd_cols = int(vb_record["view_batch_cols"]["fwd"])
    budget = int(vb_record["budget_bytes"])          # pf._transient_budget_bytes()
    result["budget_bytes_live"] = budget
    result["budget_gb_live"] = budget / 1e9
    # The mirror this file prices absent counts with must match the class it
    # mirrors, or the registered table is priced off stale constants.
    result["budget_constants_ok"] = (
        int(pf.VIEW_BATCH_TRANSIENT_BUDGET_BYTES) == BUDGET_CAP_BYTES
        and int(pf.VIEW_BATCH_TRANSIENT_FLOOR_BYTES) == BUDGET_FLOOR_BYTES
        and int(pf.VIEW_BATCH_SINO_MULTIPLE) == BUDGET_SINO_MULTIPLE)
    # On CPU ``_transient_budget_bytes`` short-circuits to the flat cap
    # (projectors.py:232-233), so the smoke's budget is n-INDEPENDENT by
    # construction and can neither confirm nor refute the mechanism.  Said out
    # loud on the row rather than left to be inferred from equal numbers.
    result["budget_is_cpu_constant"] = (not cuda)
    result["budget_mirror_ok"] = (
        None if not cuda else budget == budget_bytes_for(cell, n_realized))

    # ── THE COST FUNCTION: bytes_per_view and the chunk it saw ───────────────
    bound_fwd = pf._fwd_body_per_dev[0]
    cost = getattr(bound_fwd, "_view_batch_cost", None)
    result["fwd_bytes_per_view"] = None
    result["fwd_chunk_seen_by_cost_fn"] = None
    result["fwd_cost_skip_reason"] = None
    if cost is None:
        result["fwd_cost_skip_reason"] = (
            "the bound forward body carries no _view_batch_cost (the kernels "
            "are unavailable -- the CPU smoke): the batch follows the torch "
            "gather-transient rule and the chunk constant is inert")
    else:
        bytes_per_view, chunk_seen = cost(num_pixels, fwd_cols, args)
        result["fwd_bytes_per_view"] = int(bytes_per_view)
        result["fwd_chunk_seen_by_cost_fn"] = int(chunk_seen)
        result["fwd_cap_views_live"] = int(budget // max(1, int(bytes_per_view)))
        result["fwd_cap_binds"] = bool(result["fwd_cap_views_live"]
                                       < int(chunk_seen))
        result["fwd_chunk_unpinned_ok"] = (int(chunk_seen) == SHIPPED_CHUNK)

    # ── THE REGISTERED PREDICTION, recomputed live and checked ───────────────
    prediction = predict_fwd_batch(geometry, cell, recon_shape, num_pixels,
                                   args, n_realized)
    result.update(prediction)
    registered = PREDICTED_FWD_BATCH.get((geometry, cell[0], n_realized))
    result["registered_fwd_batch"] = registered
    result["registered_launches_per_device"] = \
        PREDICTED_LAUNCHES_PER_DEVICE.get((geometry, cell[0], n_realized))
    # The registry is the ADVANCE commitment; the recomputation is the guard
    # against a source change turning it into a postdiction.
    result["registry_matches_live_prediction_ok"] = (
        None if registered is None
        else registered == prediction["predicted_fwd_batch"])
    # And the prediction against what the code actually realized.  Skipped --
    # with the reason RECORDED, never a vacuous pass -- where no kernel body is
    # bound, because the registered table is the KERNEL's arithmetic.
    if cost is None or not cuda:
        result["prediction_vs_realized_ok"] = None
        result["prediction_skip_reason"] = (
            "no kernel forward body is bound (the CPU smoke), and on CPU the "
            "budget is the flat 2 GiB constant regardless of count: the "
            "registered table is the kernel's arithmetic under the "
            "count-divided budget and neither applies here")
    else:
        result["prediction_vs_realized_ok"] = (
            result["fwd_batch_realized"] == prediction["predicted_fwd_batch"])

    # ── the per-owner real view spans (the launch derivation's input) ────────
    local_views = _local_view_counts(model, cell)
    result["local_views_per_device"] = local_views

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

    # ── the chunk constants, RE-READ: nothing in the run moved them ──────────
    result["fwd_chunk_after"] = int(
        getattr(kernel_module, SPEC[geometry]["fwd_chunk_const"]))
    result["back_chunk_after"] = int(
        getattr(kernel_module, SPEC[geometry]["back_chunk_const"]))
    result["chunks_unchanged_ok"] = (
        result["fwd_chunk_after"] == shipped_fwd_chunk
        and result["back_chunk_after"] == shipped_back_chunk)

    # ── arm check: the realized device list after the timed call (protocol 1) ─
    realized = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["devices_ok"] = (len(realized) == n_dev) if cuda else \
        (len(realized) == (len(pin_devices) if pin_devices else 1))

    # ── arm check: the launch-key witness (kb3's positive witness) ───────────
    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (result["back_launch_keys_delta"] > 0
                                         and result["fwd_launch_keys_delta"] > 0)

    # ── THE OBSERVED BATCH DISTRIBUTION, and the DERIVED launch count ────────
    passes = max(1, WARM_REPEATS)
    result["view_batch_observed"] = {k: sorted(v.items())
                                     for k, v in observed_batches.items()}
    folded = summarize_observed(observed_batches, local_views, passes)
    result["view_batch_observed_folded"] = folded
    result["launch_count_is_derived"] = True
    result["launch_count_derivation"] = (
        "launches = sum over batch EVENTS of ceil(local_views / batch).  An "
        "event is one _effective_view_batch call, made ONCE per "
        "sparse_forward_project_view_range call (projectors.py:382); that call "
        "then loops for v in range(v0, v1, vb_size) (projectors.py:386).  The "
        "assumption: each owner walks ONE contiguous real-view span "
        "(projectors.py:367-368, tomography_model.py:417-418), whose length is "
        "local_views_per_device.")
    fwd_keys = sorted(k for k in folded if k.startswith("fwd_dev"))
    back_keys = sorted(k for k in folded if k.startswith("back_dev"))
    result["fwd_observed_batches_per_device"] = [
        folded[k]["batch_values"] for k in fwd_keys]
    result["fwd_observed_modal_per_device"] = [
        folded[k]["batch_modal"] for k in fwd_keys]
    result["fwd_observed_min_per_device"] = [
        folded[k]["batch_min"] for k in fwd_keys]
    result["fwd_events_per_pass_per_device"] = [
        folded[k]["events_per_pass"] for k in fwd_keys]
    result["fwd_launches_per_pass_per_device"] = [
        folded[k]["launches_per_pass"] for k in fwd_keys]
    result["fwd_launches_per_pass_all_devices"] = (
        sum(folded[k]["launches_per_pass"] for k in fwd_keys)
        if fwd_keys else None)
    result["back_launches_per_pass_all_devices"] = (
        sum(folded[k]["launches_per_pass"] for k in back_keys)
        if back_keys else None)
    result["unattributed_batch_events"] = (
        folded.get("unattributed", {}).get("events_total", 0))
    # The distribution note of the module docstring, as a CHECK: the VCD subset
    # steps are predicted to realize the full chunk at every cell and count, so
    # the modal observed batch should be SHIPPED_CHUNK everywhere and any second
    # population below it is the full-pixel forward.
    result["fwd_modal_is_subset_prediction_ok"] = (
        None if not fwd_keys or not cuda
        else all(folded[k]["batch_modal"] == PREDICTED_SUBSET_BATCH
                 for k in fwd_keys))
    # Gated on cuda for the same reason as the check above: on CPU the torch
    # legacy rule returns 64, which is below the chunk for a reason that has
    # nothing to do with the count-divided budget, and a bare True there would
    # read as the signal this probe exists to find.
    result["fwd_second_population_below_chunk"] = (
        None if not fwd_keys or not cuda
        else any(folded[k]["batch_min"] < SHIPPED_CHUNK for k in fwd_keys))

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

    composed = result["vcd_warm"]
    for region in REGIONS:
        block = regions[region]
        result[f"{region}_wall_per_pass_s"] = block["host_wall_s"] / passes
        result[f"{region}_dev_span_max_per_pass_s"] = (
            block["device_span_max_ms"] / 1e3 / passes)
        result[f"{region}_dev_span_sum_per_pass_s"] = (
            block["device_span_sum_ms"] / 1e3 / passes)
        result[f"{region}_calls"] = block["calls"]
        # The orchestration gap: host wall minus the largest per-device event
        # span, per pass -- the fan-out, broadcast and assembly around the
        # kernel (protocol 10).  On the CPU smoke the per-device map collapses
        # to one 'cpu' key whose span IS the host wall, so the gap is an
        # artifact there and the backend field says so.
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
    # The forward wall PER LAUNCH: if the cap's extra launches at n=4 cost
    # anything, this is the column that prices them.
    launches = result["fwd_launches_per_pass_all_devices"]
    result["forward_wall_per_launch_ms"] = (
        result["forward_funnel_wall_per_pass_s"] / launches * 1e3
        if launches else None)

    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    _finish(result, out, cfg)
    return result


def generator_worker(cfg):
    """Build ONE shared sinogram per GEOMETRY-CELL: phantom -> sinogram -> .npy,
    plus its md5 sidecar.  Every arm at that coordinate reconstructs THAT array,
    so no arm's timing carries an input difference.  Pinned to one device so the
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
    """The common tail: checksum, the strided row sample, and the host peak."""
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
    arms incomparable.  The variable is POPPED and then SET, so an inherited
    value can never survive into an arm that did not ask for it.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)     # protocol 6
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"             # production bodies only
    if cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg7_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg7_{cfg['arm_id']}.json")
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


def build_plan(geometries, counts):
    """The arm plan, in JOB ORDER (see the module docstring's ARM ORDER)."""
    smoke_cpu_n2 = SMOKE and os.environ.get("MG7_SMOKE_CPU_N2", "1") == "1"
    coords = [(g, cell, [n for n in ns if n in counts])
              for g, cell, ns in coordinates() if g in geometries]
    phase0, phase1 = [], []

    def arm(geometry, cell, n_dev, suffix="", **extra):
        return dict(framework="torch", arm_class="prod", geometry=geometry,
                    cell=list(cell), n_dev=n_dev,
                    arm_id=f"{geometry}_{cell[0]}_n{n_dev}_prod{suffix}",
                    **extra)

    for geometry, cell, _ns in coords:
        gen = dict(framework="torch", arm_class="generator", geometry=geometry,
                   cell=list(cell), n_dev=None,
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if SMOKE and DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        phase0.append(gen)

    # PHASE 0: every n=1 REFERENCE arm first (protocol 11), so a truncated job
    # still yields the references every scaling reading is taken against.
    for geometry, cell, ns in coords:
        if 1 in ns:
            phase0.append(arm(geometry, cell, 1))

    # PHASE 1: the n>1 arms, counts blocked per coordinate with the direction
    # alternating between coordinates (protocol 9's reversal on the count axis),
    # so count position and time position are decorrelated.
    for index, (geometry, cell, ns) in enumerate(coords):
        higher = [n for n in ns if n != 1]
        if index % 2 == 1:
            higher = list(reversed(higher))
        for n in higher:
            phase1.append(arm(geometry, cell, n))
        if smoke_cpu_n2:
            # SMOKE ONLY: two virtual cpu devices, so the sharded path (the
            # banded forward, the per-owner view spans, band_reduce, halo) is
            # exercised without CUDA.  The env pin is CUDA-only, so this arm
            # pins by device LIST and says so on the row.
            phase1.append(arm(geometry, cell, 2, suffix="_smokecpu",
                              cpu_devices=[DEVICE, DEVICE]))
    return phase0, phase1


# ── the summary ───────────────────────────────────────────────────────────────
def _key(row):
    cell = row.get("cell") or [None]
    return (row.get("geometry"), cell[0], row.get("n_dev"),
            bool(row.get("cpu_devices")))


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


def _short(values, limit=4):
    if not values:
        return "-"
    text = ",".join(str(v) for v in values[:limit])
    return text + ("+" if len(values) > limit else "")


def summarize(rows, geometries, counts, out_path):
    """One table per GEOMETRY-CELL, then the closing block.

    NO verdict is printed.  The two discriminating signatures are STATED and the
    numbers are placed beside them; the decision is made in analysis (the house
    rule -- mg4's knee rule and mg5's attribution both)."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by = {_key(r): r for r in live}
    summaries = []
    print(f"\n===== mg7 cone view-batch probe ({out_path}) =====")

    for geometry, cell, _ns in coordinates():
        if geometry not in geometries:
            continue
        cell_rows = [r for r in live
                     if r.get("geometry") == geometry
                     and (r.get("cell") or [None])[0] == cell[0]]
        if not cell_rows:
            continue
        ns = sorted({r["n_dev"] for r in cell_rows if r.get("n_dev")})
        print(f"\n--- {geometry} {cell} ---")
        print(f"{'arm':>22}{'n':>3}{'vb_f':>6}{'pred':>6}{'reg':>5}"
              f"{'bud_GB':>8}{'bpv_MB':>8}{'cap':>6}{'obs':>12}"
              f"{'ev/pass':>9}{'L/pass':>8}{'L/dev':>7}"
              f"{'fwd_s':>8}{'fwd_dev':>9}{'warm_s':>8}{'pk_GB':>7}"
              f"{'checks':>20}")
        summary = dict(geometry=geometry, cell=list(cell), rows=[])
        for row in rows:
            if row.get("geometry") != geometry or \
                    (row.get("cell") or [None])[0] != cell[0] or \
                    row.get("arm_class") == "generator":
                continue
            if row.get("error"):
                print(f"{row.get('arm_id', '?'):>22}  ERROR: "
                      f"{str(row['error'])[:90]}")
                summary.setdefault("errors", []).append(row.get("arm_id"))
                continue
            checks = []
            for name, flag in (("dev", row.get("devices_ok")),
                               ("pin", row.get("pin_env_ok")),
                               ("bod", row.get("bodies_ok")),
                               ("bpd", row.get("bodies_per_device_ok")),
                               ("vb", row.get("vb_ok")),
                               ("pred", row.get("prediction_vs_realized_ok")),
                               ("reg", row.get(
                                   "registry_matches_live_prediction_ok")),
                               ("chunk", row.get("chunks_are_shipped_ok")),
                               ("chunk2", row.get("chunks_unchanged_ok")),
                               ("bud", row.get("budget_mirror_ok")),
                               ("budc", row.get("budget_constants_ok")),
                               ("kern", row.get("kernels_launched_ok")),
                               ("kill", row.get("kill_switch_off_ok")),
                               ("cal", row.get("calibration_absent_ok")),
                               ("md5", row.get("sino_md5_ok")),
                               ("rgn", row.get("region_nonzero_ok")),
                               ("rec", row.get("reconcile_ok"))):
                if flag is False:
                    checks.append(f"{name}:FAIL")
            bpv = row.get("fwd_bytes_per_view")
            obs = _short(sorted({b for values in
                                 (row.get("fwd_observed_batches_per_device")
                                  or []) for b in values}))
            ev = row.get("fwd_events_per_pass_per_device") or []
            ldev = row.get("fwd_launches_per_pass_per_device") or []
            print(f"{row['arm_id'].split('_', 2)[-1]:>22}"
                  f"{str(row.get('n_dev') or '-'):>3}"
                  f"{str(row.get('fwd_batch_realized') or '-'):>6}"
                  f"{str(row.get('predicted_fwd_batch') or '-'):>6}"
                  f"{str(row.get('registered_fwd_batch') or '-'):>5}"
                  f"{_fmt(row.get('budget_gb_live'), '8.3f')}"
                  f"{_fmt(bpv / 1e6 if bpv else None, '8.2f')}"
                  f"{str(row.get('fwd_cap_views_live') or '-'):>6}"
                  f"{obs:>12}"
                  f"{_fmt(ev[0] if ev else None, '9.1f')}"
                  f"{_fmt(row.get('fwd_launches_per_pass_all_devices'), '8.1f')}"
                  f"{_fmt(ldev[0] if ldev else None, '7.1f')}"
                  f"{_fmt(row.get('forward_funnel_wall_per_pass_s'), '8.2f')}"
                  f"{_fmt(row.get('forward_funnel_dev_span_max_per_pass_s'), '9.2f')}"
                  f"{_fmt(row.get('vcd_warm'), '8.2f')}"
                  f"{row.get('gpu_peak_bytes', 0) / 2 ** 30:>7.2f}"
                  f"{(','.join(checks) if checks else 'ok'):>20}")
            summary["rows"].append(dict(
                arm_id=row["arm_id"], n=row.get("n_dev"),
                fwd_batch_realized=row.get("fwd_batch_realized"),
                fwd_batch_per_device=row.get("fwd_view_batch_per_device"),
                predicted=row.get("predicted_fwd_batch"),
                registered=row.get("registered_fwd_batch"),
                budget_bytes=row.get("budget_bytes_live"),
                bytes_per_view=bpv,
                chunk_seen=row.get("fwd_chunk_seen_by_cost_fn"),
                cap_views=row.get("fwd_cap_views_live"),
                cap_binds=row.get("fwd_cap_binds"),
                observed=row.get("view_batch_observed"),
                observed_folded=row.get("view_batch_observed_folded"),
                events_per_pass=ev, launches_per_pass_per_device=ldev,
                launches_per_pass=row.get("fwd_launches_per_pass_all_devices"),
                local_views=row.get("local_views_per_device"),
                fwd_wall=row.get("forward_funnel_wall_per_pass_s"),
                fwd_dev_span=row.get(
                    "forward_funnel_dev_span_max_per_pass_s"),
                fwd_gap=row.get("forward_funnel_gap_per_pass_s"),
                back_wall=row.get("back_funnel_wall_per_pass_s"),
                back_dev_span=row.get(
                    "back_funnel_dev_span_max_per_pass_s"),
                wall_per_launch_ms=row.get("forward_wall_per_launch_ms"),
                warm=row.get("vcd_warm"), spread=row.get("vcd_warm_spread"),
                peaks=row.get("gpu_peak_per_device"),
                checks=checks))

        # The observed distributions, printed in full: the table above shows the
        # set of values, this shows how many calls chose each.
        print("   observed forward batches per device "
              "(batch x events over all warm passes):")
        for row in cell_rows:
            folded = row.get("view_batch_observed_folded") or {}
            parts = []
            for key in sorted(k for k in folded if k.startswith("fwd_dev")):
                dist = ", ".join(f"{b}x{c}"
                                 for b, c in folded[key]["distribution"])
                parts.append(f"{key}={{{dist}}}")
            print(f"     n={row.get('n_dev')}: {'  '.join(parts) or '-'}")
        print(f"   local views per device: " + "; ".join(
            f"n={r.get('n_dev')}: {r.get('local_views_per_device')}"
            for r in cell_rows))

        recorded = MG1_COMPOSED_S.get((geometry, cell[0]))
        if recorded:
            print("   mg1 composed warm walls at this cell (context, not a "
                  "gate): " + ", ".join(f"n={n} {recorded[n]:.2f} s"
                                        for n in sorted(recorded)))
        summary["mg1_composed"] = recorded
        summaries.append(summary)

    # ── the cross-cell view of the one number this probe is about ────────────
    print("\n== the realized forward view batch, all coordinates ==")
    print(f"   {'coordinate':>22}" + "".join(f"{f'n={n}':>16}" for n in counts))
    print(f"   {'':>22}" + "".join(f"{'batch (pred)':>16}" for _n in counts))
    for geometry, cell, _ns in coordinates():
        if geometry not in geometries:
            continue
        cells_out = []
        for n in counts:
            row = by.get((geometry, cell[0], n, False))
            if not row:
                cells_out.append(f"{'-':>16}")
                continue
            cells_out.append(
                f"{str(row.get('fwd_batch_realized') or '-'):>9}"
                f"{'(' + str(row.get('predicted_fwd_batch') or '-') + ')':>7}")
        print(f"   {f'{geometry} {cell[0]}':>22}" + "".join(cells_out))
    print(f"   For contrast, mg1 measured {MG1_BATCH_AT_1024} at EVERY count at "
          f"both 1024 cells, where the 2 GiB cap binds at n = 1, 2 and 4.")

    print("\n== launches per pass (DERIVED, not measured) ==")
    print(f"   {'coordinate':>22}" + "".join(f"{f'n={n}':>18}" for n in counts))
    print(f"   {'':>22}" + "".join(f"{'per dev / all dev':>18}"
                                   for _n in counts))
    for geometry, cell, _ns in coordinates():
        if geometry not in geometries:
            continue
        cells_out = []
        for n in counts:
            row = by.get((geometry, cell[0], n, False))
            ldev = (row or {}).get("fwd_launches_per_pass_per_device") or []
            total = (row or {}).get("fwd_launches_per_pass_all_devices")
            if not row:
                cells_out.append(f"{'-':>18}")
                continue
            cells_out.append(f"{_fmt(ldev[0] if ldev else None, '9.1f')}"
                             f"{_fmt(total, '9.1f')}")
        print(f"   {f'{geometry} {cell[0]}':>22}" + "".join(cells_out))
    print("   A batch EVENT is one view-range CALL, not a launch: the driver "
          "calls")
    print("   _effective_view_batch once per call (projectors.py:382) and then "
          "loops")
    print("   ceil(span / batch) times (projectors.py:386).  These launch "
          "counts are")
    print("   DERIVED from the events and the per-owner real view spans; every "
          "row")
    print("   carries the derivation and the assumption it rests on.")

    # ── THE CLOSING BLOCK: the two signatures, STATED and not adjudicated ────
    print("\n== THE TWO DISCRIMINATING SIGNATURES (stated, not adjudicated -- "
          "the decision is made in analysis) ==")
    print("   (a) The cone batch FALLS with n at the 384 and 512 cells while "
          "parallel's HOLDS")
    print("       at 128 across n = 1, 2, 4.  Then the budget's 1/n "
          "proportionality is the")
    print("       mechanism: cone's per-view cost is params-derived and does "
          "not shrink with")
    print("       the shard, so the falling budget divides into fewer views, "
          "while parallel's")
    print("       cost tracks the slice band and its cap stays clear of the "
          "chunk.")
    print(f"       REGISTERED MAGNITUDE, so the fall is not read as larger "
          f"than it is: the")
    print(f"       chunk of {SHIPPED_CHUNK} floors the batch until the cap "
          f"drops below it, which the")
    print("       arithmetic puts at n=4 ONLY -- 128/128/85 at cone 384 and "
          "128/128/113 at")
    print("       cone 512, not a 1/n fall.  Its cost shows in the LAUNCH "
          "columns: cone's")
    print("       summed forward launches double at n=4 (4/4/8) while "
          "parallel's hold (4/4/4).")
    print("   (b) BOTH hold -- cone's batch is 128 at every count too.  Then "
          "the reading is")
    print("       REFUTED: the count-divided budget never reaches the chunk at "
          "these cells,")
    print("       nothing about the realized batch distinguishes cone from "
          "parallel, and the")
    print("       small-cell harm the ladder measured needs another cause "
          "(the fan-out and")
    print("       glue that findings §3.3 already names below the 384 cell "
          "are the standing")
    print("       alternative).")

    # -- the instrument backend caveat -----------------------------------------
    backends = {r.get("event_backend") for r in live if r.get("event_backend")}
    if any(b != "cuda_events" for b in backends):
        print(f"\nNOTE: event backend {sorted(backends)}.  On the CPU path the "
              f"per-device span map collapses to a single 'cpu' key whose span "
              f"IS the host wall, so the device-span and gap columns price "
              f"nothing.  They are meaningful only under cuda_events.")
    if any(r.get("budget_is_cpu_constant") for r in live):
        print("NOTE: at least one arm ran on CPU, where "
              "_transient_budget_bytes short-circuits to the flat 2 GiB "
              "constant (projectors.py:232-233).  The budget there is "
              "n-INDEPENDENT by construction, so a CPU arm can neither "
              "confirm nor refute the mechanism -- it exercises the paths.")

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
    geometries, counts = selected_plan()
    phase0, phase1 = build_plan(geometries, counts)
    if "--dry-run" in sys.argv:
        measured = [c for c in phase0 + phase1
                    if c["arm_class"] != "generator"]
        generators = [c for c in phase0 if c["arm_class"] == "generator"]
        print(f"mg7 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  geometries {geometries}, counts {counts}, warm repeats "
              f"{WARM_REPEATS}, iterations {VCD_ITERATIONS}, device {DEVICE}")
        print(f"  chunk constants are READ ONLY at the shipped "
              f"{SHIPPED_CHUNK}; nothing is pinned")
        for label, plan in (("phase0", phase0), ("phase1", phase1)):
            for cfg in plan:
                registered = PREDICTED_FWD_BATCH.get(
                    (cfg["geometry"], cfg["cell"][0], cfg.get("n_dev")))
                print(f"  [{label}] {cfg['arm_id']:<34} "
                      f"cell={tuple(cfg['cell'])} n={cfg['n_dev']} "
                      f"registered_batch="
                      f"{registered if registered is not None else '-'}")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg7_conebatch_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg7 cone view-batch probe on {RUN_LABEL} ({DEVICE}); geometries "
          f"{geometries}, counts {counts} -> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY (protocol 11): a truncated job still yields its
    # n=1 reference arms, which is why phase 0 runs first.
    with open(out_path, "w") as sink:
        for phase, plan in (("phase0", phase0), ("phase1", phase1)):
            for cfg in plan:
                print(f"  [{phase}] {cfg['arm_id']}", flush=True)
                row = run_one(dict(cfg, phase=phase))
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
                sink.flush()
        summaries, rerun = summarize(rows, geometries, counts, out_path)
        for summary in summaries:
            sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.write(json.dumps(dict(thermal_rerun=rerun)) + "\n")
        sink.flush()
    # mg4's and mg5's artifact rule, not mg1's: mg7's shared sinograms are
    # internal to the job (nothing outside re-verifies these md5s) and the three
    # of them are several GB of scratch.  Export MG7_KEEP_ARTIFACTS=1 to keep
    # them -- which is what a follow-up job re-verifying an md5 would need.
    if os.environ.get("MG7_KEEP_ARTIFACTS", "0") != "1":
        for geometry, cell, _ns in coordinates():
            if geometry not in geometries:
                continue
            for path in (_sino_path(geometry, cell),
                         _md5_path(geometry, cell)):
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
