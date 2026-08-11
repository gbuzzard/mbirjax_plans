"""mg14 -- does fusing the column gather's per-batch accumulation into the
projector make the forward projection cheaper, and does it cost any memory?

WHAT CHANGED, in one paragraph.  On more than one GPU the forward projection
gathers batches of voxel columns onto the GPU that owns the views and projects
each batch, summing the batches into that owner's sinogram block.  Until now
each batch got its OWN full block: the projector allocated one, copied its view
batches into it, and handed it back for the driver to add into the running
total.  That is two full-block passes and one full-block allocation per batch,
and the block is shard-sized, so the cost is the same whatever the pixel batch
is -- a bigger batch HIDES it rather than removing it, and there are on the
order of a hundred batches in a 1024-class pass at the default.  Now the first
batch's projection allocates the block and every later batch adds into that same
block from inside the projector's own view loop, where the block was going to be
written anyway: one pass, no per-batch allocation.

WHAT IS CLAIMED, and what this job measures.
    SPEED.  A few to fifteen percent of the forward, growing with the number of
    batches -- a CPU reading gave 13.3 percent at 64 batches and 4.4 percent at
    16.  BUSY IS THE PRIMARY READING HERE, which is a change from the jobs
    before this one.  The work that was removed is projector work: it ran on the
    compute stream between the kernel launches, inside the per-body-call
    brackets.  The bracket is reported beside it because the forward funnel is
    what a user waits for, but the effect should appear in busy first.
    VALUE.  Bit-identical, by construction.  Per element the sequence is still
    batch 0's contribution, then batch 1's added to it, then batch 2's -- the
    same summands in the same order.  Only where the addition happens changed.
    MEMORY.  A view-owner no longer holds a separate incoming block beside the
    one it is accumulating into, so the widest instant is one shard-sized block
    narrower.  THE LEDGER STILL CHARGES TWO.  That charge is shared with the
    banded path, which really does hold both, and the ledger's rule is that it
    may charge more than a run needs but never less, so the column-gather path
    is deliberately over-charged by one block rather than given a term of its
    own.  The modeled peak is therefore IDENTICAL on the two trees -- verified
    before this job was written -- and the interesting reading is whether the
    MEASURED peak, or the allocator's reserved bytes, falls.

THE TWO TREES, both built in-job on the compute node.
    control   a clone of the tip at eaccf55bc6b215d11e978802f4db0f6f8fdb2695.
              The projector allocates a block per batch and the driver adds it
              in: ``owned.add_(part)``.
    fused     the same clone with four files replaced.  The projector takes an
              ``accumulate_into`` block and adds into it in place; the driver
              hands it ``owned`` from the second batch onward.
Both trees already carry the copy streams and the gather-ahead ordering that
mg12 measured, so those are shared ground here rather than a variable, and they
are what makes the busy reading clean: the cross-device copies ride streams of
their own, so the per-body-call brackets contain projection work and not
transfer.

THE ARMS.  Three configurations, each on both trees, 6 measured arms, plus two
untimed sinogram generators.
    parallel  n=4  batch  4096      the most batches, so the widest lever
    parallel  n=4  batch  8192      the library default
    cone      n=4  batch  8192
All at mg9's through mg12's cell (1024, 1008, 992), which gives 1008 slices --
divisible by 1, 2 and 4, so no arm's arithmetic carries a padding term.  ARM
ORDER is configuration-major: the two trees of one configuration run
back-to-back, so a truncated job yields COMPLETE comparisons and any drift in
the node lands inside a comparison rather than between the columns.

THE VERDICT LINE, and the rule it is drawn by.  A "spread" is one arm's own max
minus min over its two warm passes, floored at SPREAD_FLOOR_FRAC of the control
arm's reading -- two passes is a weak estimator of run-to-run noise, so the
comparison refuses to resolve a difference smaller than that floor.
    ACCUMULATION SAVES   when control_busy - fused_busy exceeds the two arms'
                         busy spreads added together.
    NO CHANGE            otherwise.
Two further lines print only when they fire, and neither overrules a verdict --
each says something about the reading a verdict rests on.
    COMPOSED CHANGED     when the composed reconstruction wall moves OPPOSITE to
                         busy, both moves being outside their own spreads.
                         Removing work from the forward cannot make the whole
                         reconstruction slower, so the two moving in opposite
                         directions means something outside the forward moved
                         and the block has to be read with that in mind.  This
                         is the mirror of mg12's BUSY CHANGED advisory: there a
                         busy change was the surprise, here a busy DROP is the
                         expected result and a composed wall that disagrees with
                         it is the surprise.
    UNSTABLE ARM(S)      when an arm's own two warm passes disagree by more than
                         WARM_INSTABILITY_FRAC of its own reading.  The
                         combined-spread rule would otherwise let such an arm
                         confirm whatever was predicted.

WHAT EVERY ARM PROVES ABOUT ITSELF.  A wrong PYTHONPATH would run both arms of
the same tree and quietly report NO CHANGE, so the tree an arm actually imported
is witnessed four separate ways, and any disagreement ABORTS the arm rather than
reporting it.
    THE PATH.  os.path.realpath(mbirtorch.__file__) must sit inside the tree
    root the runner exported -- and the tree root is put at the FRONT of
    sys.path before anything imports mbirtorch, because the running script's own
    directory otherwise precedes everything PYTHONPATH names (see
    put_tree_first).
    THE BYTES.  hashlib sha256, first 12 hex, of projectors.py,
    tomography_model.py and _memory_ledger.py read from beside that __file__.
    The summary then checks these ACROSS arms: control and fused must differ in
    all three.
    THE SIGNATURE.  ``accumulate_into`` is a parameter of the live
    ``Projectors.sparse_forward_project_view_range`` on the fused tree and is
    absent on the control, read out of the running function rather than off
    disk.  The driver source carries the matching pair: ``owned.add_(part)`` on
    control, ``accumulate_into=owned`` on fused.
    THE BEHAVIOUR.  Counts taken while the arm runs: how many projector calls
    were handed a block to accumulate into (zero on control, one per batch after
    the first on fused), and whether the block a call returned is the SAME
    OBJECT the previous call on that worker returned -- which is the whole of
    what the change does, observed from outside.
Those abort an arm that disagrees.  A fifth is recorded and printed but never
gating: the order in which one worker issued its gathers and projections, which
both trees open the same way here and which is the only reading that depends on
how threads happen to be scheduled.

THE TRANSFER ROUTE, witnessed on every arm.  The library probes the hardware
once per configuration and falls back to routing every transfer through host
memory when a direct copy fails to round-trip.  That does not bear on the
accumulation directly, but it changes what the forward is spending its time on,
so every arm prints model.dev2dev_safe, records whether the library's own
host-bounce warning fired during the first forward, and reads the module flag
that warning sets.  The interpretation line changes NO verdict.

TERMS OF ART, each defined once, here.
    arm          one subprocess run at fixed parameters: one tree, one geometry,
                 one device count, one pixel batch.
    tree         one of the two source trees, selected by PYTHONPATH.
    configuration  a (geometry, device count, pixel batch) triple.  Each is run
                 on both trees and gets one comparison block.
    composed     one whole timed reconstruction's wall.
    busy         the sum of per-body-call event spans on one device -- the
                 projector's view loop, which is where the removed work was.
                 The reading is the largest device's.  THE PRIMARY READING.
    bracket      the forward funnel's per-device CUDA event span -- copies,
                 waits and projections together.  The reading is the largest
                 device's.
    per-launch   busy divided by the body call count.
    stall        bracket minus busy: the part of the forward a device spent not
                 projecting.
    spread       one arm's max minus min over its two warm passes.
    repeat floor an arm's own pass-to-pass value distance.  Both forward kernels
                 accumulate with float atomics and are not bit-reproducible, so
                 this is not zero, and it is the floor the cross-tree distance
                 has to be read against.
    modeled peak the memory ledger's per-device peak for this arm's
                 configuration.  Identical on the two trees, on purpose.
    measured peak torch.cuda.max_memory_allocated on that device.

ENVIRONMENT KNOBS (all optional; the two tree roots are required).
    MG14_TREE_CONTROL / MG14_TREE_FUSED   the tree roots (required)
    MG14_ARMS=p4b04096-control,...   run only these arm tokens
    MG14_CONFIGS=p4b04096,c4b08192   run only these configurations
    MG14_ITERATIONS=3                VCD iterations per reconstruction
    MG14_WARM_REPEATS=2              timed reconstructions after the cold pass
    MG14_MAX_EVENT_PAIRS=400000      per-reconstruction event budget
    MG14_KEEP_ARTIFACTS=1            keep the sinograms and the value samples
    MG14_PIN=configure|env           how the device count is pinned; 'configure'
                                     is the default and the documented choice.
                                     'env' pins with MBIRTORCH_NUM_DEVICES
                                     instead, which leaves the model on the
                                     automatic branch so the library builds its
                                     own ledger.  Recorded on every row.
    MG14_SMOKE=1                     the local CPU smoke (tiny cell, few iters)
    MG14_DEVICE=cpu                  smoke device
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
import warnings
import weakref

# ── CONFIG ────────────────────────────────────────────────────────────────────
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

# mg9's / mg10's / mg11's cell, and nothing else.  cell = (num_views,
# num_det_rows, num_det_channels); at this cell both geometries give recon
# (992, 992, 1008), so the slice count is 1008 and it divides 1, 2 and 4 exactly.
CELL = (1024, 1008, 992)

SMOKE = os.environ.get("MG14_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG14_DEVICE", "cpu" if SMOKE else "cuda")

# The two trees.  The runner puts one of these on the child's PYTHONPATH and the
# child proves it imported from there; nothing else selects a tree.
TREES = ("control", "fused")
TREE_ENV = {"control": "MG14_TREE_CONTROL",
            "fused": "MG14_TREE_FUSED"}
TREE_NOTE = {
    "control": "the tip: the projector allocates a block per batch and the "
               "driver adds it into the running total",
    "fused": "the projector takes an accumulate_into block and adds into it in "
             "place, inside the view loop that was going to write it anyway",
}

# The library's own name for the switch.  Spelled here only so the parent
# process (which never imports torch) can remove it from every child
# environment; every worker asserts this string against the library's own.
COLUMN_GATHER_ENV_VAR = "MBIRTORCH_FORWARD_COLUMN_GATHER"

# The three configurations, in job order: the widest lever first.  The cost the
# change removes is one full-block pass and one full-block allocation PER BATCH,
# so it grows with the batch COUNT, and the batch count is the pixel count over
# the pixel batch.  Batch 4096 therefore has twice the lever of the default
# 8192 at the same cell, which is why it leads.  Each runs on both trees.
CONFIGS = (
    dict(token="p4b04096", geometry="parallel", n_dev=4, pixel_batch=4096),
    dict(token="p4b08192", geometry="parallel", n_dev=4, pixel_batch=8192),
    dict(token="c4b08192", geometry="cone", n_dev=4, pixel_batch=8192),
)
SMOKE_CONFIGS = (
    dict(token="p2b00016", geometry="parallel", n_dev=2, pixel_batch=16),
    dict(token="c2b00016", geometry="cone", n_dev=2, pixel_batch=16),
)

VCD_ITERATIONS = int(os.environ.get("MG14_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13        # mg1's / mg5's / mg9's / mg10's / mg11's seed
WARM_REPEATS = max(2, int(os.environ.get("MG14_WARM_REPEATS", "2")))

# How the device count is pinned.  'configure' is the documented choice (see the
# module docstring); 'env' is kept as a one-word alternative because it is the
# only way to make the library build its own ledger for the run.
PIN_MECHANISM = os.environ.get("MG14_PIN", "configure").strip().lower()

# ── THE COMPARISON RULE, one constant, stated before any number is read ───────
# Two warm passes is a weak estimator of run-to-run noise, so the comparison
# refuses to resolve a difference smaller than this fraction of the control
# arm's own reading, however tight the observed spread happened to be.  The
# effect this job is looking for is a few to fifteen percent of the forward, so
# the floor sits below the low end of the claim rather than an order of
# magnitude below it -- a 4 percent saving is a real reading here and must not
# be floored away, and the price of that is that this job can resolve less than
# mg12 could.
SPREAD_FLOOR_FRAC = 0.02
# The other end of the same problem.  A spread floor stops the comparison
# resolving a difference that is too small; this stops it being decided by an
# arm whose own two passes disagree wildly.  An arm whose warm spread exceeds
# this fraction of its own reading is not a usable measurement of anything, and
# the combined-spread rule above would quietly turn it into a confirmation of
# whatever was predicted.  Such an arm gets a printed UNSTABLE ARM line
# immediately above the verdict lines of its block and again beside them in the
# recap, so the line is never read without it.  The verdict lines themselves are
# unchanged: this flags a reading, it does not overrule one.
WARM_INSTABILITY_FRAC = 0.25

# ── WHAT mg11 MEASURED, quoted, and used for nothing but the wall arithmetic ──
# Row mg11_flip_gates_h001_20260811_041522.jsonl, four H100s, this cell, 1 cold
# pass plus 3 warm passes per arm.  These are the READINGS THIS JOB IS ABOUT --
# they are printed in the plan and NEVER compared against a measurement here,
# because every comparison in this file is between arms measured in this job.
MG11_READING = {
    ("parallel", 4, 4096): dict(bracket_s=16.31, busy_s=12.41, composed_s=24.8,
                                wall_s=133),
    ("parallel", 4, 8192): dict(bracket_s=13.33, busy_s=10.83, composed_s=21.8,
                                wall_s=121),
    ("cone", 4, 8192): dict(bracket_s=16.47, busy_s=15.04, composed_s=39.0,
                            wall_s=192),
}
# THE CPU READING THE CLAIM COMES FROM, quoted so the plan can print it, and
# compared against NOTHING here: a CPU measurement of a GPU change is a reason
# to run this job, not a number to check it against.
CPU_CLAIM = {64: 0.133, 16: 0.044}    # batch count -> fraction of the forward
GENERATOR_S = 70
# mg11's walls above cover 4 reconstructions per arm and this job runs 3, so a
# control arm's base is mg11's wall less one composed reconstruction.  The two
# overlay arms have never been run at all, so they take mg10's and mg11's own
# rule for an unmeasured arm: the low end assumes the arm costs what its control
# costs, the high end assumes half again as much.
CONTROL_HIGH_FACTOR = 1.05
OVERLAY_HIGH_FACTOR = 1.50
# This job gets its OWN inductor cache, so the first arm at each launch shape
# compiles from cold where mg11 reused a warm cache.  That cost lands in each
# arm's discarded cold pass; this is the allowance the plan prints for it.
# Three launch shapes here rather than mg12's five, so the allowance is smaller.
COLD_CACHE_LOW_S = 400
COLD_CACHE_HIGH_S = 800

RESULTS_DIR = os.environ.get(
    "MG14_RESULTS", os.path.dirname(os.path.abspath(__file__)))
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
VALUE_SAMPLE_TARGET = 64

# How many (gather, project) events one worker's issue-order trace keeps.  A
# handful is enough to tell G P G P from G G P G P, and a bound keeps the
# recorder off the hot path for the rest of the run.
TRACE_LIMIT = 12

# The library files whose bytes are fingerprinted per arm: the projector that
# gained the accumulation parameter, the driver that hands it a block, and the
# ledger.  The overlay changes all three, so all three must differ between the
# trees -- the ledger's change is comment only and moves no number, which is
# checked separately by the modeled peaks coming out equal.
FINGERPRINT_FILES = ("projectors.py", "tomography_model.py", "_memory_ledger.py")

# ── the region definitions and GPU-health machinery, COPIED from mg9/mg11 ─────
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
NESTED_REGIONS = ("band_reduce",)
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
MAX_EVENT_PAIRS = int(os.environ.get("MG14_MAX_EVENT_PAIRS", "400000"))

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
def configurations():
    return list(SMOKE_CONFIGS if SMOKE else CONFIGS)


def build_arms():
    """Every arm, in job order.  CONFIGURATION-MAJOR: the two trees of one
    configuration run back-to-back, so a truncated job yields whole comparisons
    and any drift in the node lands inside a comparison rather than between the
    control column and the treatment column."""
    arms = []
    for config in configurations():
        for tree in TREES:
            arms.append(dict(config, tree=tree,
                             config_token=config["token"],
                             token=f"{config['token']}-{tree}"))
    return arms


ARMS = build_arms()


def selected_arms():
    """The arms to run, narrowed by MG14_CONFIGS then by MG14_ARMS."""
    chosen = list(ARMS)
    raw = os.environ.get("MG14_CONFIGS", "").strip()
    if raw:
        known = {c["token"] for c in configurations()}
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in known:
                raise ValueError(f"MG14_CONFIGS: {token!r} is not one of "
                                 f"{sorted(known)}")
            wanted.add(token)
        chosen = [a for a in chosen if a["config_token"] in wanted]
    raw = os.environ.get("MG14_ARMS", "").strip()
    if raw:
        known = {a["token"] for a in ARMS}
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in known:
                raise ValueError(f"MG14_ARMS: {token!r} is not one of "
                                 f"{sorted(known)}")
            wanted.add(token)
        chosen = [a for a in chosen if a["token"] in wanted]
    if not chosen:
        raise ValueError("no arms selected")
    return chosen


def tree_root(tree):
    """The directory a tree's mbirtorch package sits in, from the environment.
    Required: this file will not guess a tree, because guessing wrong is the
    one failure that would look like a clean result."""
    root = os.environ.get(TREE_ENV[tree], "").strip()
    if not root:
        raise RuntimeError(
            f"{TREE_ENV[tree]} is not set, so the {tree!r} arms have no tree to "
            f"import.  The sbatch builds both trees and exports both roots; "
            f"running this file by hand needs the same two exports.")
    return os.path.abspath(root)


def cell_for(_geometry):
    return SMOKE_CELL if SMOKE else CELL


def num_slices_for(cell):
    """The reconstruction's slice count at this cell.  Both geometries derive it
    from the detector row count; the worker asserts this against the model's own
    recon_shape before it is used for anything."""
    return int(cell[1])


def cylinder_bytes(batch, num_slices):
    """What ONE gathered cylinder holds: the batch's pixel columns at every
    slice, float32."""
    return int(batch) * int(num_slices) * 4


# ── staged-artifact mechanics (mg5's / mg9's / mg11's md5 discipline) ─────────
def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_mg14_sino_{geometry}_{cell[0]}.npy")


def _md5_path(geometry, cell):
    return _sino_path(geometry, cell) + ".md5"


def _sample_path(arm_id, index):
    return os.path.join(RESULTS_DIR, f"_mg14_sample_{arm_id}_p{index}.npy")


def _md5(path, chunk=8 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_12(path):
    """First 12 hex of a file's sha256 -- the per-arm provenance witness."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()[:12]


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


# ── THE PROVENANCE WITNESS ────────────────────────────────────────────────────
_PATH_SHADOWS = []


def put_tree_first(tree):
    """Put this arm's tree at the FRONT of sys.path before anything imports
    mbirtorch, and record every other place on the path that holds one.

    PYTHONPATH IS NOT ENOUGH ON ITS OWN, which is easy to miss.  Python puts the
    running script's own directory ahead of everything PYTHONPATH names, and
    this harness runs out of the same directory the trees and the results sit
    beside.  An mbirtorch package left there -- an unpacked tree, a stray
    checkout, a copy made while debugging -- would then be imported by every
    arm no matter what PYTHONPATH said, and all the arms would agree with each
    other perfectly.  Measured on this Mac: with a tree on PYTHONPATH and
    another in the working directory, the working directory won every time.
    The fingerprint below would catch it and abort the arm, so nothing false
    could be reported either way; this makes the arm RUN CORRECTLY rather than
    merely fail loudly, and names the shadowing directory on the row.
    """
    root = tree_root(tree)
    del _PATH_SHADOWS[:]
    for entry in list(sys.path):
        directory = entry or os.getcwd()
        if not os.path.isdir(directory):
            continue
        if os.path.realpath(directory) == os.path.realpath(root):
            continue
        if os.path.exists(os.path.join(directory, "mbirtorch", "__init__.py")):
            _PATH_SHADOWS.append(entry or "<the working directory>")
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    return list(_PATH_SHADOWS)


def tree_fingerprint(expected_tree):
    """Where the mbirtorch this process imported actually came from, and what
    its bytes are.

    Two readings, and they are independent of each other.  The PATH says the
    package sits inside the tree root the runner exported for this arm, which
    catches a PYTHONPATH that did not take -- including an editable install of
    some other tree winning through a meta-path finder.  The BYTES are sha256
    prefixes of the files read from beside that path, which catch a tree that is
    in the right place with the wrong contents (a copy that did not land, a
    patch that did not apply).  The summary then compares the byte readings
    ACROSS arms, where a run that accidentally used one tree three times shows
    up as three identical fingerprints.
    """
    import mbirtorch

    # The package directory is where __init__.py sits, so the files beside it
    # are the ones this process really imported.
    package = os.path.dirname(os.path.realpath(mbirtorch.__file__))
    root = os.path.realpath(tree_root(expected_tree))
    inside = package.startswith(root + os.sep)
    digests = {}
    for name in FINGERPRINT_FILES:
        path = os.path.join(package, name)
        digests[name] = _sha256_12(path) if os.path.exists(path) else None
    return dict(expected_tree=expected_tree, expected_root=root,
                package_dir=package,
                mbirtorch_file=os.path.realpath(mbirtorch.__file__),
                inside_expected_root=bool(inside),
                path_shadows=list(_PATH_SHADOWS),
                sha256_12=digests)


def source_tokens():
    """Which shape the live code carries, read out of the tree this process
    imported rather than out of any record of what was built.

    THE DISCRIMINATOR IS THE PROJECTOR'S SIGNATURE.  ``accumulate_into`` is a
    parameter of ``Projectors.sparse_forward_project_view_range`` on the fused
    tree and is absent on the control, and it is read here with
    ``inspect.signature`` off the RUNNING function -- so a tree that passed the
    byte fingerprint but somehow ran another projector would still be caught.
    The driver carries the matching pair: it either adds the returned block in
    itself (``owned.add_(part)``) or hands the projector the block to add into
    (``accumulate_into=owned``), and exactly one of those is present.

    The rest are SHARED GROUND rather than discriminators: both trees carry the
    copy streams and the gather-ahead ordering that mg12 measured, and both
    charge three gathered cylinders.  They are read anyway, because a tree that
    quietly lost them would change what the busy reading contains -- with the
    copies back on the compute stream the per-body-call brackets would start
    covering transfer, and the primary reading of this job would stop meaning
    what it says."""
    import inspect

    from mbirtorch import _memory_ledger, _sharding
    from mbirtorch.projectors import Projectors
    from mbirtorch.tomography_model import TomographyModel

    driver = inspect.getsource(TomographyModel._sparse_forward_project_columns)
    signature = inspect.signature(Projectors.sparse_forward_project_view_range)
    return dict(
        projector_takes_accumulate_into=("accumulate_into"
                                         in signature.parameters),
        driver_passes_accumulate_into=("accumulate_into=owned" in driver),
        driver_adds_the_returned_block=("owned.add_(part)" in driver),
        projector_signature=str(signature),
        # shared ground, checked on both trees
        driver_has_batch_bounds=("batch_bounds" in driver),
        driver_has_async_gather=("gather_column_band_async" in driver),
        sharding_has_copy_stream=hasattr(_sharding, "copy_stream"),
        sharding_has_async_gather=hasattr(_sharding, "gather_column_band_async"),
        sharding_has_open_close=(hasattr(_sharding, "open_copy_streams")
                                 and hasattr(_sharding, "close_copy_streams")),
        sharding_has_wait=hasattr(_sharding, "wait_for_column_band"),
        column_gather_residents=int(
            getattr(_memory_ledger, "COLUMN_GATHER_RESIDENTS", -1)))


# The shape each tree's source is required to have.  An arm whose reading
# disagrees ABORTS: a mislabeled tree is the one failure mode that would look
# like a clean "no change".  The first three entries separate the trees; the
# rest are the shared ground both must stand on.
_SHARED_GROUND = dict(driver_has_batch_bounds=True,
                      driver_has_async_gather=True,
                      sharding_has_copy_stream=True,
                      sharding_has_async_gather=True,
                      sharding_has_open_close=True,
                      sharding_has_wait=True,
                      column_gather_residents=3)
TREE_SOURCE_EXPECTED = {
    "control": dict(_SHARED_GROUND,
                    projector_takes_accumulate_into=False,
                    driver_passes_accumulate_into=False,
                    driver_adds_the_returned_block=True),
    "fused": dict(_SHARED_GROUND,
                  projector_takes_accumulate_into=True,
                  driver_passes_accumulate_into=True,
                  driver_adds_the_returned_block=False),
}


# ── INSTRUMENT 0: per-region host walls and device spans (mg9's, unchanged) ───
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    reconstruction loop's calling thread.  Copied from mg11_flip_gates.py, which
    copied it from mg10, which copied it from mg9, which copied it from
    mg1_readout.py without change -- so mg14's forward bracket IS mg9's and
    mg11's forward bracket, and a control arm here is comparable with theirs.

    CUDA path: for each device in the region's placement a start and an end
    event are CREATED AND RECORDED inside ``with torch.cuda.device(dev)``, on
    that device's current stream -- which is the COMPUTE stream, the one the
    projections run on, on every tree.  The end event is recorded AFTER the call
    returns, so it queues behind everything the call enqueued, including the
    waits a copy stream imposes.  That is what makes this span the right reading
    of the stall: it covers the time the device could not project because values
    had not arrived.  Elapsed times are read only in :meth:`finish`, after a
    per-device synchronize.

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


class IssueTrace:
    """The order in which one worker issued its gathers and its projections.

    'G' is written by the gather wrapper and 'P' by the forward body wrapper --
    two wrappers on two different functions that do not consult each other.  The
    control tree gathers and projects in turn, so a worker's trace opens 'GP';
    both overlays gather one batch ahead, so theirs opens 'GG'.

    THE KEY IS (device name, thread), and both halves are needed.  The device
    name alone is enough on CUDA, where every device has its own name.  It is
    NOT enough in the CPU smoke, where every virtual device is called 'cpu' and
    two workers would write into one list and scramble it -- the same
    collapsed-name artifact the region instrument documents.  The thread alone
    is not enough either, because ``run_per_device`` may be handed a pool whose
    threads serve different devices on different fan-outs.  Together they give
    one writer per key within a fan-out, and a key that outlives a fan-out only
    ever concatenates whole clean sequences, which leaves the opening intact.

    The dict is written from worker threads without a lock.  Each key has a
    single writer, and dict insertion is atomic under the interpreter lock, so
    the only shared mutation is adding a key -- which each thread does once.
    """

    def __init__(self):
        self.marks = {}          # (device name, thread) -> list of 'G'/'P'
        self.device_index = {}   # the same key -> device position, when known

    def note(self, device_name, mark, dev_index=None):
        key = (device_name, threading.get_ident())
        bucket = self.marks.get(key)
        if bucket is None:
            bucket = self.marks[key] = []
        if len(bucket) < TRACE_LIMIT:
            bucket.append(mark)
        if dev_index is not None:
            self.device_index[key] = dev_index

    def read(self):
        """The traces, as (device name, device position, sequence) triples in a
        stable order."""
        out = []
        for key in sorted(self.marks, key=lambda k: (k[0], k[1])):
            out.append(dict(device=key[0],
                            device_index=self.device_index.get(key),
                            sequence="".join(self.marks[key])))
        return out


# ── INSTRUMENT 1: busy time, per device, per body call (mg9's / mg11's) ───────
class BusyProbe:
    """Times each individual forward projection BODY call, in buckets keyed by
    DEVICE POSITION -- never by object identity, because every entry of
    ``_fwd_body_per_dev`` can be the same object.

    Busy divided by the call count is the PER-LAUNCH time.

    WHY THIS IS THE RIGHT READING OF "THE KERNELS ALONE" ON ALL THREE TREES.
    The start event is recorded on the device's compute stream immediately
    before the body runs, and a stream reaches its work in order.  On the
    control tree the copies were enqueued on that same stream before this point,
    so the start event is reached only once they have finished.  Both trees here
    enqueue a wait on that same stream before this point, so the start event is
    reached only once this batch's copies have landed.  Either way the span that
    follows is the projection and not the transfer, which is what lets the busy
    column be compared across the trees -- and it is why busy is this job's
    primary reading: the work the change removes ran right here, between the
    kernel launches, inside this bracket.

    THE SHAPE HISTOGRAMS ARE A WITNESS, not decoration.  A forward body's first
    positional argument is ``band_values``, of shape (pixels, columns).  On the
    column-gather path that is (the pixel batch) x (the WHOLE device-form slice
    axis), so a single column count equal to the full slice count is what
    separates a column gather from a band by another name -- and it is recorded
    here, by a wrapper on the body, entirely independently of the cylinder
    shapes the gather probe records.

    Threading.  Body calls arrive on the per-device worker threads of
    ``run_per_device``, which runs at most one thread per device index at a time
    and waits for all of them before the next fan-out, so each bucket has a
    single writer and needs no lock.
    """

    def __init__(self, torch_module, cuda, n_dev, pairs_per_device, trace,
                 device_names):
        self.torch = torch_module
        self.cuda = cuda
        self.n_dev = n_dev
        self.pairs_per_device = pairs_per_device
        self.trace = trace
        self.device_names = device_names
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
        ``_mbirtorch_no_compile``; the caller asserts the first survived."""
        torch_module, cuda = self.torch, self.cuda
        device = _concrete_device(torch_module, device, cuda)

        @functools.wraps(body)
        def wrapped(*args, **kwargs):
            self.trace.note(self.device_names[dev_index], "P",
                            dev_index=dev_index)
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

        wrapped._mg14_dev_index = dev_index
        wrapped._mg14_wrapped_body = body
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


# ── INSTRUMENT 2: the gather, counted -- and deliberately NOT timed per copy ──
class GatherProbe:
    """Counts what the forward's transfer layer did, and records the shape of
    every cylinder it assembled.

    WHAT IS COUNTED, and why each count is here.
        gather_calls        entries into the gather from the driver, on every
                            tree.  Zero would mean the column gather did not run.
        gather_async_calls  entries into ``gather_column_band_async``, which
                            which BOTH trees here use.  It is not a
                            discriminator; it is the check that the copies are
                            still on streams of their own, which is what keeps
                            the busy reading free of transfer time.
        wait_calls          entries into ``wait_for_column_band``: one per batch
                            on both trees, and the count must match the gather
                            count.
        open_calls / close_calls   the once-per-forward stream handshakes.
        broadcast_calls     entries into the BANDED fan-out.  This must be zero
                            on every arm: a positive count would mean the banded
                            walk ran and the arm is not measuring the gather at
                            all.
    The cylinder height and width histograms are the shape witness: height is
    the whole device-form slice axis at every gather, width is the pixel batch
    (with one short tail batch per pass).

    WHY THE COPIES ARE NOT TIMED ONE BY ONE.  mg9 and mg11 bracket every
    ``move_shard`` with a CUDA event pair and take a lock around the counters.
    Doing that here would put extra event records and a shared lock into the
    exact code path this job exists to measure -- the issue of copies that are
    supposed to be overlapping with a projection -- and a probe that serializes
    the thing it measures cannot answer the question.  So this probe touches
    only the two per-batch entry points, once each, and the transfer time is
    read where it actually shows: in the difference between the bracket and the
    busy time.  The gather's HOST wall is still recorded, and it carries a real
    signal of its own -- the gather returns once its copies are issued rather
    than once they have landed, so it should be small on both trees.

    THE LOCK is taken once per batch, not once per copy, and only around
    integer updates.

    THE ACCUMULATION COUNTERS ARE MG14'S OWN, and they are what the whole job
    turns on.  ``Projectors.sparse_forward_project_view_range`` is wrapped once
    per call -- once per pixel batch per view-owner, not per view batch -- and
    three things are recorded.
        view_range_calls    how many times the projector's view loop ran.
        accumulate_calls    how many of those were handed a block to add into.
                            Zero on the control tree, where the parameter does
                            not exist; one per batch after the first on the
                            fused tree.
        block_reuses        how many times the block a call returned was the
                            SAME OBJECT the previous call on that worker
                            returned.  This is the change itself, observed from
                            outside and without asking the driver anything:
                            keeping one block across batches is exactly what
                            fusing the accumulation means.
    The identity is held as a WEAK reference between calls.  A strong one would
    keep the previous block alive and change the residency this job is also
    measuring, and comparing raw id() values would be worse than useless -- on
    the control tree the previous block is freed each batch and a new one can
    land on the same address, which would read as a reuse that never happened.
    A weak reference to a freed block returns None instead, so the control tree
    cannot report a reuse at all.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.gather_calls = 0
        self.gather_inner_calls = 0
        self.gather_async_calls = 0
        self.wait_calls = 0
        self.open_calls = 0
        self.close_calls = 0
        self.broadcast_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}
        self.cyl_width = {}
        self.view_range_calls = 0
        self.accumulate_calls = 0
        self.block_reuses = 0

    def drain(self):
        """Read the counts and reset, once per timed reconstruction."""
        record = dict(gather_calls=int(self.gather_calls),
                      gather_inner_calls=int(self.gather_inner_calls),
                      gather_async_calls=int(self.gather_async_calls),
                      wait_calls=int(self.wait_calls),
                      open_copy_streams_calls=int(self.open_calls),
                      close_copy_streams_calls=int(self.close_calls),
                      broadcast_calls=int(self.broadcast_calls),
                      gather_host_wall_s=float(self.gather_host_s),
                      view_range_calls=int(self.view_range_calls),
                      accumulate_calls=int(self.accumulate_calls),
                      block_reuses=int(self.block_reuses),
                      cyl_height_hist={str(k): v for k, v
                                       in sorted(self.cyl_height.items())},
                      cyl_width_hist={str(k): v for k, v
                                      in sorted(self.cyl_width.items())})
        self.gather_calls = 0
        self.gather_inner_calls = 0
        self.gather_async_calls = 0
        self.wait_calls = 0
        self.open_calls = 0
        self.close_calls = 0
        self.broadcast_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}
        self.cyl_width = {}
        self.view_range_calls = 0
        self.accumulate_calls = 0
        self.block_reuses = 0
        return record


def attach_forward_probes(model, torch_module, cuda, max_pairs):
    """Install the busy probe and the gather probe; return
    ``(busy, gathers, trace, verify, detach, observed)``.

    Call this AFTER the discarded cold pass.  The body wrappers live inside the
    projector object, and a device-count settle during the first reconstruction
    would rebuild that object and throw them away.  With the device count pinned
    by configure_devices no settle can happen, but ``verify()`` re-checks before
    and after every timed reconstruction anyway, because a silent detachment
    would read as a busy time of zero rather than as an error.
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

    # THE ISSUE-ORDER TRACE (see IssueTrace for the key and why it has two
    # halves).  It is a CORROBORATING witness, not a decisive one: it is the
    # only witness in this file that depends on how threads happen to be
    # scheduled, so it is recorded and printed and an arm is never aborted for
    # it.  Tree identity is settled by the byte fingerprint, the source tokens
    # and the async gather count, none of which touch threading.
    trace = IssueTrace()
    device_names = [str(_concrete_device(torch_module, d, cuda))
                    for d in devices]

    busy = BusyProbe(torch_module, cuda, n_dev,
                     max(1, max_pairs // max(1, 2 * n_dev)), trace,
                     device_names)
    originals = list(bodies)
    wrappers = []
    for index, body in enumerate(originals):
        wrapper = busy.wrap(body, index, devices[index])
        # The driver chooses the view batch by reading this attribute OFF THE
        # BODY, so a wrapper that lost it would silently move the kernel onto
        # the torch batching rule and change what "busy" means.
        if getattr(body, "_view_batch_cost", None) is not \
                getattr(wrapper, "_view_batch_cost", None):
            raise RuntimeError(
                "the body wrapper did not carry _view_batch_cost through; the "
                "realized view batch would change and this arm would not be "
                "comparable with the other two trees")
        wrappers.append(wrapper)
    # Mutated IN PLACE rather than rebound, so any other reference to the list
    # sees the wrappers too.
    for index, wrapper in enumerate(wrappers):
        bodies[index] = wrapper

    # The realized view batch, observed PER DEVICE by the positional key.  This
    # is recorded so that a busy time that moved between trees can be explained:
    # a changed view batch would mean the kernels were launched differently, and
    # that is a different finding from a changed transfer schedule.
    observed = {}
    observed_lock = threading.Lock()
    original_effective = pf._effective_view_batch

    def effective_view_batch(body, num_pixels, band_cols, args):
        value = original_effective(body, num_pixels, band_cols, args)
        index = getattr(body, "_mg14_dev_index", None)
        key = (f"fwd_dev{index}" if index is not None
               else "back_body_device_not_recoverable")
        with observed_lock:
            bucket = observed.setdefault(key, {})
            bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = effective_view_batch

    # -- the gather, counted once per batch ----------------------------------
    gathers = GatherProbe()
    depth = threading.local()

    # -- THE ACCUMULATION WITNESS (see GatherProbe) --------------------------
    # One wrapper on the projector's view loop, entered once per pixel batch per
    # view-owner.  It records whether the call was handed a block to add into
    # and whether the block it returned is the one the previous call on this
    # worker returned, held weakly so the recorder neither keeps a block alive
    # nor mistakes a recycled address for the same object.
    original_view_range = pf.sparse_forward_project_view_range
    last_block = threading.local()

    def sparse_forward_project_view_range(*args, **kwargs):
        accumulating = kwargs.get("accumulate_into") is not None
        out = original_view_range(*args, **kwargs)
        previous = getattr(last_block, "ref", None)
        reused = previous is not None and previous() is out
        try:
            last_block.ref = weakref.ref(out)
        except TypeError:                     # not weak-referenceable
            last_block.ref = None
        with gathers.lock:
            gathers.view_range_calls += 1
            if accumulating:
                gathers.accumulate_calls += 1
            if reused:
                gathers.block_reuses += 1
        return out

    pf.sparse_forward_project_view_range = sparse_forward_project_view_range

    def _outermost():
        """True when this is the driver's entry into a gather rather than the
        async wrapper calling the plain one underneath it.  Both are wrapped,
        and without this every batch would be counted and traced twice."""
        return getattr(depth, "level", 0) == 0

    original_gather = _sharding.gather_column_band
    original_broadcast = _sharding.broadcast_band_to_views

    def _record_shape(cyl):
        if cyl is None:
            return
        height = int(cyl.shape[-1])
        width = int(cyl.shape[0])
        gathers.cyl_height[height] = gathers.cyl_height.get(height, 0) + 1
        gathers.cyl_width[width] = gathers.cyl_width.get(width, 0) + 1

    def gather_column_band(shard_tensors, p0, p1, target, dev2dev_safe=True):
        """The gather's inner primitive, wrapped exactly as mg11 wraps it.

        The RETURNED cylinder is measured rather than the arguments, because the
        claim being witnessed is about what was assembled: its height is the
        whole device-form slice axis and its width is the pixel batch.  Reading
        p1 - p0 instead would only repeat the caller's arithmetic back at it."""
        outer = _outermost()
        if outer:
            trace.note(str(target), "G")
        depth.level = getattr(depth, "level", 0) + 1
        host0 = time.perf_counter()
        cyl = None
        try:
            cyl = original_gather(shard_tensors, p0, p1, target,
                                  dev2dev_safe=dev2dev_safe)
            return cyl
        finally:
            depth.level -= 1
            span = time.perf_counter() - host0
            with gathers.lock:
                gathers.gather_inner_calls += 1
                _record_shape(cyl)
                if outer:
                    gathers.gather_calls += 1
                    gathers.gather_host_s += span

    _sharding.gather_column_band = gather_column_band

    def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
        """The BANDED fan-out.  Every arm in this job runs the column gather, so
        this must never be entered; it is wrapped so that "never" is measured
        rather than assumed."""
        with gathers.lock:
            gathers.broadcast_calls += 1
        return original_broadcast(band, view_owners, dev2dev_safe=dev2dev_safe)

    _sharding.broadcast_band_to_views = broadcast_band_to_views

    # The copy-stream entry points.  Both trees here have them, but each is
    # still wrapped only when it is present: a tree that had lost them would
    # then read zero rather than fail to attach, and the witness check turns
    # that zero into a clear abort.
    restore = [("gather_column_band", original_gather),
               ("broadcast_band_to_views", original_broadcast)]

    original_async = getattr(_sharding, "gather_column_band_async", None)
    if original_async is not None:
        def gather_column_band_async(shard_tensors, p0, p1, target,
                                     dev2dev_safe=True):
            outer = _outermost()
            if outer:
                trace.note(str(target), "G")
            depth.level = getattr(depth, "level", 0) + 1
            host0 = time.perf_counter()
            try:
                return original_async(shard_tensors, p0, p1, target,
                                      dev2dev_safe=dev2dev_safe)
            finally:
                depth.level -= 1
                span = time.perf_counter() - host0
                with gathers.lock:
                    gathers.gather_async_calls += 1
                    if outer:
                        gathers.gather_calls += 1
                        gathers.gather_host_s += span
        _sharding.gather_column_band_async = gather_column_band_async
        restore.append(("gather_column_band_async", original_async))

    original_wait = getattr(_sharding, "wait_for_column_band", None)
    if original_wait is not None:
        def wait_for_column_band(target, ready):
            with gathers.lock:
                gathers.wait_calls += 1
            return original_wait(target, ready)
        _sharding.wait_for_column_band = wait_for_column_band
        restore.append(("wait_for_column_band", original_wait))

    original_open = getattr(_sharding, "open_copy_streams", None)
    if original_open is not None:
        def open_copy_streams(devs):
            with gathers.lock:
                gathers.open_calls += 1
            return original_open(devs)
        _sharding.open_copy_streams = open_copy_streams
        restore.append(("open_copy_streams", original_open))

    original_close = getattr(_sharding, "close_copy_streams", None)
    if original_close is not None:
        def close_copy_streams(devs):
            with gathers.lock:
                gathers.close_calls += 1
            return original_close(devs)
        _sharding.close_copy_streams = close_copy_streams
        restore.append(("close_copy_streams", original_close))

    def verify():
        """Is the instrument still on the path the driver takes?"""
        live = model.projector_functions
        return dict(
            projector_object_same=(live is pf),
            body_list_same=(live._fwd_body_per_dev is bodies),
            wrappers_in_place=all(
                bodies[i] is wrappers[i] for i in range(len(wrappers))),
            gather_wrapped=(_sharding.gather_column_band is gather_column_band),
            broadcast_wrapped=(
                _sharding.broadcast_band_to_views is broadcast_band_to_views),
            view_range_wrapped=(pf.sparse_forward_project_view_range
                                is sparse_forward_project_view_range))

    def detach():
        for index, body in enumerate(originals):
            bodies[index] = body
        pf._effective_view_batch = original_effective
        # Removing the instance attribute restores the bound method from the
        # class rather than leaving a copy of it shadowing one.
        try:
            del pf.sparse_forward_project_view_range
        except AttributeError:
            pf.sparse_forward_project_view_range = original_view_range
        for name, original in restore:
            setattr(_sharding, name, original)

    return busy, gathers, trace, verify, detach, observed


def copy_streams_created():
    """How many dedicated copy streams the library made, read out of its own
    cache.  Both trees here have them, so this reads one per CUDA device the
    gather touched on both -- it is a check that the copies are still off the
    compute stream, not a way of telling the trees apart."""
    from mbirtorch import _sharding

    return len(getattr(_sharding, "_COPY_STREAMS", {}) or {})


def host_bounce_flag():
    """The library's own record of whether any transfer was routed through host
    memory.  ``move_shard`` sets this module flag the first time it takes that
    path, so reading it is a direct answer that does not depend on catching a
    warning on a worker thread."""
    from mbirtorch import _sharding

    return bool(getattr(_sharding, "_warned_host_bounce", False))


# ── the model, and the knobs an arm sets ──────────────────────────────────────
def _build_torch_model(geometry, cell, n_dev, cpu_devices=None):
    """The model, with the device count pinned.

    THE PIN.  ``configure_devices`` is the documented way a caller names the
    layout, and calling it takes the choice out of the library's hands: the
    automatic device-count search never runs, and neither do the widening SPEED
    FLOORS that order it.  That is why the two trees' floors tables are
    irrelevant to every number in this job -- no arm consults one.  (In this
    build they are identical anyway: the overlay replaces four files and
    _widening_floors.py is not among them.)
    A second effect matters to the instruments: with the layout fixed before the
    first reconstruction there is no device-count settle to rebuild the
    projector object the probes attach to.
    The cost is that the library builds no memory ledger of its own for a layout
    the caller named, so ``last_memory_ledger`` stays None and the modeled peak
    comes from this file's call to the same entry point (see ``ledger_reading``).
    MG14_PIN=env pins with MBIRTORCH_NUM_DEVICES instead, which leaves the model
    on the automatic branch; the row records which was used.
    """
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
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    elif PIN_MECHANISM == "configure" and n_dev:
        model.configure_devices(num_devices=int(n_dev))
    model.set_params(no_warning=True, verbose=0)
    return model


def configure_arm(model, pixel_batch):
    """THE WHOLE OF WHAT AN ARM DOES TO THE LIBRARY, besides the device pin.

    One documented attribute is set on the model.  The switch is not touched at
    all: the column gather is the DEFAULT on both trees, so the shipped
    configuration is the environment variable being absent, and the runner
    removes it from every child environment rather than setting it to anything."""
    model.forward_project_pixel_batch = int(pixel_batch)
    return dict(pixel_batch_attribute=int(pixel_batch),
                knobs=("model.configure_devices(num_devices=n) + "
                       "model.forward_project_pixel_batch = "
                       f"{int(pixel_batch)}; "
                       "MBIRTORCH_FORWARD_COLUMN_GATHER left UNSET because the "
                       "column gather is the default on both trees"))


def read_switch(model):
    """The library's own answer to 'which forward runs', plus the inputs it read
    to get there.  Nothing here re-derives the rule -- the resolver is called."""
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
    """THE MODELED PEAK, from the library's own planning entry point.

    THE ENTRY POINT.  ``vcd_recon`` calls ``self._apply_device_policy(
    partition_sequence=..., weights=..., init_recon=..., fm_hessian=...,
    prox_input=..., init_error_sinogram=...)``, which forwards those call arrays
    to ``self._build_memory_ledger(devices=devices, **call_arrays)``; that is
    ``_memory_ledger.estimate_peak_device_bytes(_memory_ledger.plan_from_model(
    ...))``.  This function calls the same method with the same call arrays.

    WHICH NUMBER THE GATE READS.  ``last_memory_ledger`` when the library built
    one -- the ledger the run itself was decided with -- and this function's own
    build otherwise.  With the device count pinned by ``configure_devices`` the
    library builds none, because an explicit layout is the caller's, so the
    fallback is the normal case here rather than the exception.  The two can
    differ in one input, the partition sequence, and mg11 measured them
    IDENTICAL on every arm at this cell; both are recorded either way.

    WHY THE LEDGER MOVES BETWEEN TREES, and why that is correct.  Both overlays
    raise ``_memory_ledger.COLUMN_GATHER_RESIDENTS`` from 2 to 3, because a
    driver that gathers one batch ahead really does hold three cylinders at the
    widest instant.  Each arm's ledger comes from its OWN tree, so each arm is
    priced by the rule that tree ships, which is the only way the ratio means
    anything.  The constant itself is recorded on the row.
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
                     "_memory_ledger.plan_from_model(...))"),
        modeled_source=("library last_memory_ledger (the ledger this run was "
                        "decided with)" if library_peaks is not None else
                        "harness _build_memory_ledger (an explicit device "
                        "layout is the caller's, so the library built none)"),
        modeled_peak_per_device=modeled,
        harness_modeled_peak_per_device=harness_peaks,
        library_modeled_peak_per_device=library_peaks,
        modeled_agrees=(None if library_peaks is None
                        else library_peaks == harness_peaks),
        column_gather_residents=int(
            getattr(_memory_ledger, "COLUMN_GATHER_RESIDENTS", -1)),
        dominant_phase_per_device=[chosen.dominant_phase(i).name
                                   for i in range(len(devices))],
        plan_column_pixel_batch=plan.column_pixel_batch,
        plan_forward_band=plan.forward_band,
        plan_back_band=plan.back_band,
        plan_num_pixels_full=int(plan.num_pixels_full),
        plan_rows_track_slices=bool(plan.rows_track_slices),
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


def _hist_keys(hist):
    return {int(k) for k in (hist or {})}


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
    """One arm: prove the tree, set the knobs, one discarded cold pass, then
    WARM_REPEATS timed reconstructions with the instruments live.

    THREE ORDERINGS, all load-bearing.
      THE TREE IS PROVED FIRST, before anything is built, so an arm that imported
    the wrong source spends no GPU time at all.
      THE KNOBS GO IN BEFORE THE COLD PASS, because the pixel batch changes the
    shapes the Triton kernels launch at and the launch key includes the pixel
    count and the column count; set later, every first launch and the compile
    lock it takes would land inside the first timed reconstruction.
      THE PROBES GO IN AFTER THE COLD PASS, because the body wrappers live inside
    the projector object and anything that rebuilt it would throw them away.
    """
    import numpy as np
    import torch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    tree = cfg["tree"]
    n_dev = cfg.get("n_dev")
    pixel_batch = int(cfg["pixel_batch"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    # ── THE TREE, proved before anything else ───────────────────────────────
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  pin_mechanism=PIN_MECHANISM,
                  env_pythonpath=os.environ.get("PYTHONPATH"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_calibration=os.environ.get("MBIRTORCH_MEMORY_CALIBRATION"))
    result["fingerprint"] = tree_fingerprint(tree)
    if not result["fingerprint"]["inside_expected_root"]:
        raise RuntimeError(
            f"this arm is labeled {tree!r} but imported mbirtorch from "
            f"{result['fingerprint']['mbirtorch_file']}, which is not inside "
            f"{result['fingerprint']['expected_root']}.  Either PYTHONPATH did "
            f"not take or an editable install of another tree is winning "
            f"through a meta-path finder.  A mislabeled tree is the one failure "
            f"that would look like a clean 'no change'.")
    result["source_tokens"] = source_tokens()
    expected = TREE_SOURCE_EXPECTED[tree]
    disagreed = {key: (result["source_tokens"].get(key), want)
                 for key, want in expected.items()
                 if result["source_tokens"].get(key) != want}
    result["source_tokens_ok"] = not disagreed
    if disagreed:
        raise RuntimeError(
            f"the source this arm imported does not look like the {tree!r} "
            f"tree.  got vs expected: {disagreed}.  The bytes are "
            f"{result['fingerprint']['sha256_12']} at "
            f"{result['fingerprint']['package_dir']}.")

    import importlib
    spec = SPEC[geometry]
    kernel_module = importlib.import_module(spec["kernel_module"])
    shipped_fwd_chunk = int(getattr(kernel_module, spec["fwd_chunk_const"]))
    shipped_back_chunk = int(getattr(kernel_module, spec["back_chunk_const"]))
    result["shipped_fwd_chunk"] = shipped_fwd_chunk
    result["shipped_back_chunk"] = shipped_back_chunk

    # ── the env arm checks ───────────────────────────────────────────────────
    result["calibration_absent_ok"] = (
        os.environ.get("MBIRTORCH_MEMORY_CALIBRATION") in (None, "", "0"))
    result["kill_switch_off_ok"] = (
        os.environ.get("MBIRTORCH_DISABLE_TRITON", "0") in ("", "0"))

    # THE SWITCH, read from the library's own module rather than spelled here.
    from mbirtorch.tomography_model import COLUMN_GATHER_ENV_VAR as LIB_VAR
    result["env_var_name_ok"] = (LIB_VAR == COLUMN_GATHER_ENV_VAR)
    if not result["env_var_name_ok"]:
        raise RuntimeError(
            f"the library's switch is named {LIB_VAR!r} and this harness "
            f"removes {COLUMN_GATHER_ENV_VAR!r} from the child environment; "
            f"the two must be the same name for the removal to mean anything.")
    result["env_column_gather"] = os.environ.get(LIB_VAR)
    result["env_column_gather_ok"] = (result["env_column_gather"] is None)
    if not result["env_column_gather_ok"]:
        raise RuntimeError(
            f"{LIB_VAR} is set to {result['env_column_gather']!r} in this "
            f"arm's environment.  Every arm in this job runs the library's "
            f"DEFAULT forward, which is the column gather with the variable "
            f"absent; a value of '0' here would run the banded walk in all "
            f"fifteen arms without a word.")

    # ── the model, with the device count pinned ──────────────────────────────
    model = _build_torch_model(geometry, cell, n_dev,
                               cpu_devices=smoke_cpu_devices)
    expect_kernels = (cuda, cuda)
    result["expected_bodies_kernel"] = list(expect_kernels)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    num_slices = int(recon_shape[2])
    result["num_slices"] = num_slices
    result["num_slices_planned_ok"] = (num_slices == num_slices_for(cell))
    if not result["num_slices_planned_ok"]:
        raise RuntimeError(
            f"the plan assumed {num_slices_for(cell)} slices at this cell and "
            f"the model built {num_slices}; the cylinder-height arithmetic in "
            f"the plan print is wrong for this tree.")

    n_owners = len(model.sino_placement.devices)
    slices_per_dev = num_slices // max(1, n_owners)
    result["slices_per_dev"] = slices_per_dev
    result["pixel_batch"] = pixel_batch
    result["cylinder_bytes"] = cylinder_bytes(pixel_batch, num_slices)

    # A wrong device count invalidates every shape number in this arm.
    realized_devices = [str(d) for d in model.sino_placement.devices]
    expected_count = (len(smoke_cpu_devices) if smoke_cpu_devices
                      else int(n_dev))
    if len(realized_devices) != expected_count:
        raise RuntimeError(
            f"this arm asked for {expected_count} device(s) and settled on "
            f"{len(realized_devices)} ({realized_devices}).  Every shape number "
            f"in the arm is derived from the shard length, so the whole row "
            f"would be mislabeled.")

    # ── THE TRANSFER ROUTE, witnessed before any timing ─────────────────────
    # The library probes the hardware once per configuration and falls back to
    # routing every transfer through host memory when a direct copy fails to
    # round-trip.  A host bounce synchronizes by construction, so a copy stream
    # can buy nothing over one; this is recorded so that a NO CHANGE result can
    # be read for what it is.
    result["dev2dev_safe"] = bool(getattr(model, "dev2dev_safe", True))
    result["host_bounce_flag_before"] = host_bounce_flag()

    # ── THE KNOBS ────────────────────────────────────────────────────────────
    result["knob_record"] = configure_arm(model, pixel_batch)
    result["switch_at_install"] = read_switch(model)
    if not result["switch_at_install"]["resolver_says_gather"]:
        raise RuntimeError(
            f"the library's own resolver, model._column_gather_forward(), says "
            f"the column gather does NOT run: "
            f"{result['switch_at_install']}.  Every arm in this job measures "
            f"the gather, so there is nothing here to time.")
    if result["switch_at_install"]["resolver_pixel_batch"] != pixel_batch:
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

    def reserved():
        """The allocator's RESERVED bytes, recorded beside the allocated peak.
        The memory floor is stated against allocated bytes, so allocated is what
        the gate reads; reserved is here because a per-stream pool -- which the
        allocator can hold on to -- grows the reservation before it grows any
        single allocation.  It is the column most likely to show the block the
        fused tree no longer allocates, so a reader looking for that effect
        should not have to infer it."""
        if not cuda:
            return []
        return [int(torch.cuda.max_memory_reserved(d))
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

    # ── the discarded cold pass, with the warning recorder around it ─────────
    # The library warns once per process the first time it routes a transfer
    # through host memory, and the first forward is inside this pass.  The
    # recorder is a backstop for the module flag read below it: the warning is
    # raised on a worker thread and the filter state is process-global, so it is
    # caught here, but the flag is the answer that does not depend on that.
    if cuda:
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", index))
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    reserved_cold = reserved()
    health.append(sample_gpu_health())

    messages = [str(w.message) for w in caught]
    result["warnings_during_cold"] = messages[:20]
    result["host_bounce_warned"] = any(
        "host memory" in m or "device-to-device" in m for m in messages)
    result["host_bounce_flag_after_cold"] = host_bounce_flag()
    # One reading, from two places that do not depend on each other.
    result["host_bounce_in_use"] = bool(result["host_bounce_warned"]
                                        or result["host_bounce_flag_after_cold"]
                                        or not result["dev2dev_safe"])

    # ── the checks that need a settled, warmed model ─────────────────────────
    result["switch_after_cold"] = read_switch(model)
    if not result["switch_after_cold"]["resolver_says_gather"]:
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
    pf = model.projector_functions
    result["fwd_bodies_distinct_objects"] = (
        len({id(b) for b in pf._fwd_body_per_dev}) == len(pf._fwd_body_per_dev))
    result["fwd_chunk_after"] = int(getattr(kernel_module,
                                            spec["fwd_chunk_const"]))
    result["back_chunk_after"] = int(getattr(kernel_module,
                                             spec["back_chunk_const"]))
    # This file moves no module constant anywhere, so both must be the shipped
    # ones on EVERY arm of EVERY tree.
    result["chunks_unchanged_ok"] = (
        result["fwd_chunk_after"] == shipped_fwd_chunk
        and result["back_chunk_after"] == shipped_back_chunk)

    # ── THE MEMORY GATE'S MODELED SIDE, from the library's own entry point ───
    result["ledger"] = ledger_reading(model, weights)
    plan_batch = result["ledger"]["plan_column_pixel_batch"]
    if n_owners > 1 and plan_batch != pixel_batch:
        raise RuntimeError(
            f"the memory ledger's plan priced column_pixel_batch={plan_batch} "
            f"where this arm runs the gather at {pixel_batch}.  The modeled "
            f"peak would be the wrong shape's and the floor ratio would be "
            f"meaningless for this arm.")
    residents = result["ledger"]["column_gather_residents"]
    want_residents = TREE_SOURCE_EXPECTED[tree]["column_gather_residents"]
    if residents != want_residents:
        raise RuntimeError(
            f"this arm's ledger charges COLUMN_GATHER_RESIDENTS={residents} "
            f"and the {tree!r} tree ships {want_residents}; the modeled peak "
            f"would be another tree's.")
    result["modeled_source"] = result["ledger"]["modeled_source"]

    # ── the instruments, installed on the settled projector ──────────────────
    busy, gathers, trace, verify, detach_probes, observed = \
        attach_forward_probes(model, torch, cuda, MAX_EVENT_PAIRS)
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
        record.update(gathers.drain())
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
        # THE STALL, per device: the part of the forward funnel a device spent
        # not projecting.  This is the number the whole job is about.
        record["stall_ms_per_device"] = [
            b - u for b, u in zip(record["bracket_ms_per_device"],
                                  record["busy_ms_per_device"])]
        record["bracket_max_s"] = max(record["bracket_ms_per_device"]) / 1e3
        record["busy_max_s"] = max(record["busy_ms_per_device"]) / 1e3
        record["stall_max_s"] = record["bracket_max_s"] - record["busy_max_s"]
        record["probe_verify"] = verify()
        # THE MEASURED SIDE OF THE MEMORY GATE, per timed reconstruction.  No
        # reset between warm passes, so the series is a running maximum.
        record["peak_bytes_per_device"] = peaks()
        record["reserved_bytes_per_device"] = reserved()
        per_recon.append(record)
        health.append(sample_gpu_health())

        # ── the value column ─────────────────────────────────────────────────
        checksums.append(float(np.sum(np.abs(out), dtype=np.float64)))
        if steps is None:
            steps = _sample_steps(out.shape)
        if repeat < 2:
            # Two samples: one for the cross-tree distances and a second so the
            # summary can state this arm's OWN pass-to-pass distance, which is
            # the floor every cross-tree number has to be read against (both
            # forward kernels accumulate with float atomics and are not
            # bit-reproducible).
            path = _sample_path(cfg["arm_id"], repeat)
            np.save(path, np.ascontiguousarray(
                out[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32))
            sample_paths.append(path)

        # ── the witnesses, on the first timed reconstruction ─────────────────
        if repeat == 0:
            result["issue_trace"] = trace.read()
            result["copy_streams_created"] = copy_streams_created()
            _check_witnesses(result, record, tree, n_owners, num_slices,
                             pixel_batch, cuda)

    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_spread_s"] = (max(warm) - min(warm)) if len(warm) > 1 else 0.0
    result["per_recon"] = per_recon
    result["device_names"] = device_names
    result["view_batch_observed_per_device"] = {
        k: sorted(v.items()) for k, v in observed.items()}
    result["busy_backend"] = busy.backend
    result["probe_verify_after"] = verify()
    result["switch_after"] = read_switch(model)
    result["host_bounce_flag_after_warm"] = host_bounce_flag()
    detach_probes()
    detach_regions()

    peaks_warm = peaks()
    result["gpu_peak_cold_per_device"] = peaks_cold
    result["gpu_peak_warm_per_device"] = peaks_warm
    result["gpu_peak_per_device"] = [max(a, b) for a, b
                                     in zip(peaks_cold or [0] * len(peaks_warm),
                                            peaks_warm)]
    result["gpu_reserved_cold_per_device"] = reserved_cold
    result["gpu_reserved_warm_per_device"] = reserved()
    result["gpu_peak_bytes"] = max(result["gpu_peak_per_device"], default=0)

    result["realized_devices"] = realized_devices
    result["realized_n_devices"] = len(realized_devices)
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]

    if cuda:
        keys_after = _launch_key_counts(geometry)
        result["back_launch_keys_delta"] = keys_after[0] - keys_before[0]
        result["fwd_launch_keys_delta"] = keys_after[1] - keys_before[1]
        result["kernels_launched_ok"] = (
            result["back_launch_keys_delta"] > 0
            and result["fwd_launch_keys_delta"] > 0)

    # ── the region read-out, in mg9's field names so the rows diff ───────────
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
    result["forward_share_of_composed"] = (
        result["forward_funnel_wall_per_pass_s"] / composed if composed else None)

    result["recon_checksums"] = checksums
    result["recon_checksum"] = checksums[-1] if checksums else None
    result["value_sample_paths"] = sample_paths
    result["value_sample_steps"] = list(steps or ())
    result["recon_shape_out"] = list(out.shape)
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    result.update(summarize_arm(result))
    result.update(memory_reading(result))
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def _check_witnesses(result, record, tree, n_owners, num_slices, pixel_batch,
                     cuda):
    """THE PROOF THAT THIS ARM MEASURED WHAT ITS NAME SAYS, run on the first
    timed reconstruction and FATAL on any disagreement.

    Six groups of checks, each covering a different way the row could be wrong.
    """
    checks = {}

    # 1. THE GATHER RAN.  Zero gathers with a positive broadcast count is the
    #    banded walk under a gather's name.
    checks["gathers_positive"] = (record["gather_calls"] > 0)
    checks["no_banded_fanout"] = (record["broadcast_calls"] == 0)

    # 2. THE CYLINDER IS A COLUMN GATHER, not a band by another name.  Its
    #    height must be the WHOLE device-form slice axis at every gather; a
    #    height equal to one owner's shard would be a band.
    heights = _hist_keys(record["cyl_height_hist"])
    checks["cylinder_height_is_full_slice_axis"] = (heights == {num_slices})
    #    Its width is the pixel batch, with at most one shorter tail per pass.
    widths = _hist_keys(record["cyl_width_hist"])
    checks["cylinder_width_within_batch"] = bool(
        widths and all(0 < w <= pixel_batch for w in widths))

    # 3. THE KERNEL SAW THE SAME THING.  The busy probe records the values block
    #    each body was handed, by a wrapper on a different function.  Its column
    #    count must equal the cylinder height, so the two recorders agree
    #    without either being compared to an expectation.
    body_cols = set()
    for hist in record["busy_value_cols_per_device"]:
        body_cols |= _hist_keys(hist)
    checks["body_cols_match_cylinder_height"] = bool(body_cols
                                                     and body_cols == heights)

    # 4. THE COPY STREAMS, which BOTH trees carry here and which are what makes
    #    the busy reading clean: with the copies on streams of their own, the
    #    per-body-call brackets contain projection and not transfer.  A tree
    #    that quietly lost them would still produce numbers, and they would
    #    silently mean something else.
    checks["async_gather_entered"] = (record["gather_async_calls"] > 0)
    checks["one_wait_per_gather"] = (
        record["wait_calls"] == record["gather_calls"])
    checks["streams_opened_and_closed"] = (
        record["open_copy_streams_calls"] > 0
        and record["open_copy_streams_calls"]
        == record["close_copy_streams_calls"])
    if cuda:
        checks["copy_streams_created"] = (
            result.get("copy_streams_created", 0) >= n_owners)

    # 5. THE ACCUMULATION, which is what this job is about.  The projector's
    #    view loop runs once per pixel batch per view-owner, and one of the two
    #    shapes must be in evidence:
    #      control  no call is handed a block to add into, and no call returns
    #               the block the previous one returned -- each batch gets its
    #               own, and the driver adds them up.
    #      fused    every batch after the first is handed the block, and returns
    #               that same object, so the reuse count and the accumulate
    #               count agree with each other.  They are recorded by two
    #               different readings of the same wrapper -- one off the
    #               arguments, one off the returned object's identity -- so a
    #               witness that agreed with itself cannot pass.
    #    A pass whose pixels fit in ONE batch never accumulates on either tree,
    #    so the fused checks are made only when a second batch existed.
    checks["view_loop_ran"] = (record["view_range_calls"] > 0)
    # One view-loop call per (forward funnel call) x (view-owner) x (pixel
    # batch), so more than one batch means more calls than owners times funnel
    # calls.
    multi_batch = (record["view_range_calls"]
                   > max(1, record.get("bracket_calls", 1)) * max(1, n_owners))
    if tree == "control":
        checks["no_accumulate_into"] = (record["accumulate_calls"] == 0)
        checks["no_block_reuse"] = (record["block_reuses"] == 0)
    else:
        checks["accumulate_into_used"] = (
            record["accumulate_calls"] > 0 if multi_batch else True)
        checks["reuse_matches_accumulate"] = (
            record["block_reuses"] == record["accumulate_calls"])

    # 6. THE PROBES ARE STILL THE ONES THE DRIVER CALLS.
    checks["probes_attached"] = all(record["probe_verify"].values())

    # THE ISSUE ORDER, recorded and reported but NOT part of the fatal set.
    # BOTH trees here gather one batch ahead of the projection that reads it, so
    # both open 'GG'; this is shared ground and not a discriminator, and it is
    # kept because a tree that lost the gather-ahead would change what the
    # bracket contains.  A pass with a single batch has nothing to gather ahead,
    # so there is nothing to read then.  This is the only witness here that
    # depends on thread scheduling, and tree identity is already settled by
    # witnesses that do not, so a disagreement is printed rather than fatal.
    traces = [t["sequence"] for t in (result.get("issue_trace") or ())
              if len(t.get("sequence") or "") >= 2]
    want_open = "GG"
    readable = [t for t in traces if t.count("G") > 1]
    result["issue_order_expected"] = want_open
    result["issue_order_ok"] = (None if not readable
                                else all(t.startswith(want_open)
                                         for t in readable))

    result["witnesses"] = dict(
        checks=checks,
        gather_calls=record["gather_calls"],
        gather_async_calls=record["gather_async_calls"],
        wait_calls=record["wait_calls"],
        broadcast_calls=record["broadcast_calls"],
        open_copy_streams_calls=record["open_copy_streams_calls"],
        view_range_calls=record["view_range_calls"],
        accumulate_calls=record["accumulate_calls"],
        block_reuses=record["block_reuses"],
        cylinder_heights=record["cyl_height_hist"],
        cylinder_widths=record["cyl_width_hist"],
        values_block_cols=record["busy_value_cols_per_device"],
        issue_trace=result.get("issue_trace"),
        issue_order_expected=want_open,
        issue_order_ok=result["issue_order_ok"],
        copy_streams_created=result.get("copy_streams_created"))
    failed = [name for name, ok in checks.items() if not ok]
    result["witnesses_ok"] = not failed
    if failed:
        raise RuntimeError(
            f"witness(es) {failed} failed on the {tree!r} arm.  The full "
            f"reading is {result['witnesses']}.  This arm is not measuring "
            f"what its name says, so it reports nothing.")


# ── the per-arm reductions ────────────────────────────────────────────────────
def summarize_arm(result):
    """The arm's headline readings and its per-device table.

    THE HEADLINE READINGS are the largest device's, because a reconstruction
    waits for the slowest device, and the median over the warm passes, because a
    single pass can carry a scheduling artifact.  Each carries its own SPREAD --
    max minus min over the warm passes -- which is what every comparison in this
    job is resolved against."""
    names = result["device_names"]
    passes = result["per_recon"]
    if not passes:
        return dict(per_device=[], transfer=None)

    def median(values):
        return float(statistics.median(values)) if values else None

    def spread(values):
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    per_device = []
    for index, name in enumerate(names):
        bracket = [p["bracket_ms_per_device"][index] / 1e3 for p in passes]
        busy = [p["busy_ms_per_device"][index] / 1e3 for p in passes]
        calls = [p["busy_calls_per_device"][index] for p in passes]
        host = [p["busy_host_s_per_device"][index] for p in passes]
        back = [p["back_bracket_ms_per_device"][index] / 1e3 for p in passes]
        peak = [(p.get("peak_bytes_per_device") or [0] * len(names))[index]
                for p in passes]
        res = [(p.get("reserved_bytes_per_device") or [0] * len(names))[index]
               for p in passes]
        bracket_med, busy_med = median(bracket), median(busy)
        calls_med = median(calls)
        per_launch = ((busy_med * 1e3 / calls_med)
                      if busy_med and calls_med else None)
        cols_hist = passes[0]["busy_value_cols_per_device"][index]
        pix_hist = passes[0].get("busy_value_pixels_per_device",
                                 [{}] * len(names))[index]
        per_device.append(dict(
            device=name,
            bracket_span_s=bracket_med,
            busy_sum_s=busy_med,
            stall_s=((bracket_med - busy_med)
                     if bracket_med is not None and busy_med is not None
                     else None),
            busy_calls=calls_med,
            per_launch_ms=per_launch,
            mean_cols_per_launch=_hist_mean(cols_hist),
            mean_pixels_per_launch=_hist_mean(pix_hist),
            busy_host_s=median(host),
            busy_frac_of_bracket=(busy_med / bracket_med if bracket_med else None),
            back_bracket_span_s=median(back),
            peak_bytes=median(peak),
            reserved_bytes=median(res),
            value_cols=cols_hist,
            value_pixels=pix_hist,
            device_mismatch=sum(p["busy_device_mismatch_per_device"][index]
                                for p in passes)))

    bracket_max = [p["bracket_max_s"] for p in passes]
    busy_max = [p["busy_max_s"] for p in passes]
    per_launch_all = [d["per_launch_ms"] for d in per_device
                      if d["per_launch_ms"]]
    transfer = dict(
        gather_calls_per_recon=median([p["gather_calls"] for p in passes]),
        gather_host_s_per_recon=median([p["gather_host_wall_s"]
                                        for p in passes]),
        gather_async_calls_per_recon=median([p["gather_async_calls"]
                                             for p in passes]),
        wait_calls_per_recon=median([p["wait_calls"] for p in passes]),
        broadcast_calls_per_recon=median([p["broadcast_calls"] for p in passes]),
        copy_streams_created=result.get("copy_streams_created"),
        dev2dev_safe=result.get("dev2dev_safe"),
        host_bounce_in_use=result.get("host_bounce_in_use"))
    return dict(
        per_device=per_device, transfer=transfer,
        forward_bracket_max_s=median(bracket_max),
        forward_bracket_spread_s=spread(bracket_max),
        forward_busy_max_s=median(busy_max),
        forward_busy_spread_s=spread(busy_max),
        forward_stall_max_s=(median(bracket_max) - median(busy_max)),
        per_launch_ms=(max(per_launch_all) if per_launch_all else None),
        composed_s=result.get("vcd_warm"),
        composed_spread_s=result.get("vcd_warm_spread_s"))


def memory_reading(result):
    """The memory gate's two sides, and their ratio per device.

    The MEASURED peak is max(cold, warm): torch.cuda.max_memory_allocated is a
    running maximum, the counter is reset before each phase, and both readings
    are taken, so the gate reads the largest allocation the process actually
    reached while reconstructing.

    The MODELED peak is the ledger's, from the library's own entry point and
    from THIS arm's tree (see ledger_reading).  modeled / measured at or above
    1.00 is a preflight that over-predicts, which is what a preflight must do.
    Below 1.00 is a FLOOR VIOLATION: the library would admit a layout that does
    not fit."""
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
        reserved_warm_per_device=result.get("gpu_reserved_warm_per_device"),
        ratio_per_device=ratio,
        ratio_cold_per_device=cold_ratio,
        ratio_warm_per_device=warm_ratio,
        min_ratio=(min(live) if live else None),
        min_ratio_index=worst_index,
        min_ratio_device=(result["device_names"][worst_index]
                          if worst_index is not None else None),
        # The pair that PRODUCED the printed ratio.  The largest measured peak
        # and the smallest ratio are not always the same device, and printing
        # one device's bytes beside another device's ratio invites the reader to
        # divide two numbers that were never divided.
        modeled_at_min=(modeled[worst_index] if worst_index is not None
                        else None),
        measured_at_min=(both[worst_index] if worst_index is not None
                         else None),
        reserved_at_min=((result.get("gpu_reserved_warm_per_device") or
                          [None] * len(both))[worst_index]
                         if worst_index is not None else None),
        floor_violation=bool(live and min(live) < 1.0),
        violation_only_in_cold=bool(
            live and min(live) < 1.0 and warm_ratio
            and min(r for r in warm_ratio if r is not None) >= 1.0),
        column_gather_residents=(result.get("ledger") or {}).get(
            "column_gather_residents"),
        dominant_phase_per_device=(result.get("ledger") or {}).get(
            "dominant_phase_per_device")))


def generator_worker(cfg):
    """Build ONE shared sinogram per geometry, from the CONTROL tree, pinned to
    a single device.

    Every arm of that geometry -- on both trees -- reconstructs THAT array, so
    no arm's timing or value carries an input difference.  One device is enough
    and is the point: a single-device forward never enters the sharded driver at
    all, so both trees would produce the identical array here and the choice of
    the control tree records provenance rather than making a difference."""
    import numpy as np

    import mbirtorch

    geometry, cell = cfg["geometry"], tuple(cfg["cell"])
    devices = cfg.get("cpu_devices") or [DEVICE]
    model = _build_torch_model(geometry, cell, 1, cpu_devices=devices)
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
                fingerprint=tree_fingerprint(cfg["tree"]),
                env_column_gather=os.environ.get(COLUMN_GATHER_ENV_VAR),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


# ── the value read-out ────────────────────────────────────────────────────────
def _rel_distance(path_a, path_b):
    """Two distances between two strided reconstruction samples, and why there
    are two.

    ``rel_l2`` is a relative L2 over the strided sample, which is the metric the
    design notes state value expectations in.
    ``max_rel_of_peak`` is max|a-b| / max|b|, which is the functional form the
    standing parity suites take.  Both are reported because a single number
    cannot say whether a difference is spread thin or concentrated in a few
    voxels, and the two overlays claim the values do not move at all."""
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
    """One arm's per-reconstruction checksums, reduced: the median, and the
    REPEAT-TO-REPEAT spread as a fraction of it.  That spread is the run-to-run
    noise floor every cross-tree checksum distance is read against; both forward
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
    """Every value distance in this job, computed two ways.

    For each arm: its OWN repeat-to-repeat distance -- the floor -- and its
    distance to the CONTROL TREE'S ARM AT THE SAME CONFIGURATION.  A control arm
    has only the first, and that is what makes it the floor the other two are
    read against: it is the same configuration reconstructed twice by the same
    code, so whatever distance it shows is what two runs cost with nothing
    changed at all."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by_token = {r.get("token"): r for r in live}
    out = []
    for row in live:
        config = row.get("config_token")
        control = by_token.get(f"{config}-control")
        samples = row.get("value_sample_paths") or []
        own = _rel_distance(samples[1], samples[0]) if len(samples) > 1 else None
        vs_control = (_rel_distance(_first_sample(row), _first_sample(control))
                      if control is not None and control is not row else None)
        median_checksum, repeat_spread = _checksum_stats(row)
        out.append(dict(
            token=row.get("token"), config=config, tree=row.get("tree"),
            geometry=row.get("geometry"), n=row.get("n_dev"),
            batch=row.get("pixel_batch"),
            checksums=row.get("recon_checksums"),
            checksum_median=median_checksum,
            checksum_repeat_spread=repeat_spread,
            checksum_vs_control=(_checksum_distance(row, control)
                                 if control is not None and control is not row
                                 else None),
            own_pass_to_pass=own,
            vs_control=vs_control,
            vs_control_token=(control or {}).get("token")))
    return out


# ── the runner (mg5's / mg9's / mg11's subprocess pattern) ────────────────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.

    ONE variable carries the experiment: PYTHONPATH names the tree, and it is
    SET rather than prepended, so an inherited path cannot put another mbirtorch
    ahead of it.  The child then proves it imported from that root anyway,
    because an editable install can still win through a meta-path finder.

    MBIRTORCH_FORWARD_COLUMN_GATHER is REMOVED.  The column gather is the
    default on both trees, so its absence is the shipped configuration; setting
    it to '1' would measure the same thing but would also mean a rename in the
    library could silently stop mattering, and setting it to '0' would measure
    the banded walk.  Removal plus the child's assert is the only form
    with no ambiguity.

    MBIRTORCH_NUM_DEVICES is removed too, unless MG14_PIN=env asked for it: the
    device count is pinned on the model, and a variable saying something else
    would be a second, silent opinion.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop(COLUMN_GATHER_ENV_VAR, None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    env["PYTHONPATH"] = tree_root(cfg["tree"])
    if PIN_MECHANISM == "env" and cfg.get("n_dev") and DEVICE == "cuda":
        env["MBIRTORCH_NUM_DEVICES"] = str(cfg["n_dev"])
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg14_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg14_{cfg['arm_id']}.json")
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
                   cell=list(cell), n_dev=1, token=f"gen_{geometry}",
                   config_token=None, pixel_batch=None,
                   # Built on the control tree; a one-device forward never
                   # enters the sharded driver, so the array is the same
                   # whichever tree builds it.
                   tree="control",
                   arm_id=f"{geometry}_{cell[0]}_generator")
        if DEVICE != "cuda":
            gen["cpu_devices"] = [DEVICE]
        plan.append(gen)
    measured = []
    for arm in arms:
        cell = cell_for(arm["geometry"])
        cfg = dict(framework="torch", arm_class="instrument",
                   geometry=arm["geometry"], cell=list(cell),
                   n_dev=arm["n_dev"], token=arm["token"],
                   config_token=arm["config_token"], tree=arm["tree"],
                   pixel_batch=arm["pixel_batch"],
                   arm_id=(f"{arm['geometry']}_{cell[0]}_n{arm['n_dev']}"
                           f"_b{arm['pixel_batch']:05d}_{arm['tree']}"))
        if DEVICE != "cuda":
            # SMOKE ONLY: virtual cpu devices, so the multi-device wiring -- the
            # column gather, the per-device workers, the witnesses -- is
            # exercised without CUDA.  The accumulation itself is not a CUDA
            # feature, so the smoke really does run both shapes and can tell
            # them apart; what the smoke cannot show is the cost, the cell being
            # far too small for a full-block pass to matter.
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
    """One arm's per-device rows: the bracket, the busy sum, the stall between
    them, the launch count, the per-launch time and the memory pair."""
    print(f"\n  [{row.get('token')}] {row.get('geometry')} "
          f"n={row.get('n_dev')} batch {row.get('pixel_batch')} "
          f"tree={row.get('tree')}"
          f"  composed {_fmt(row.get('vcd_warm'), '.2f')} s "
          f"(cold {_fmt(row.get('vcd_cold'), '.2f')} s)")
    fingerprint = row.get("fingerprint") or {}
    digests = fingerprint.get("sha256_12") or {}
    print(f"      tree witness: {fingerprint.get('package_dir')}")
    if fingerprint.get("path_shadows"):
        print(f"      NOTE: another mbirtorch sits on this arm's import path at "
              f"{fingerprint['path_shadows']}; the tree above was put ahead of "
              f"it (see put_tree_first)")
    print("      bytes: " + "  ".join(f"{k}={v}" for k, v in digests.items()))
    tokens = row.get("source_tokens") or {}
    print(f"      source: projector_takes_accumulate_into="
          f"{tokens.get('projector_takes_accumulate_into')} "
          f"driver_passes_it={tokens.get('driver_passes_accumulate_into')} "
          f"driver_adds_returned_block="
          f"{tokens.get('driver_adds_the_returned_block')}")
    print(f"      shared ground: async_gather="
          f"{tokens.get('driver_has_async_gather')} "
          f"copy_stream={tokens.get('sharding_has_copy_stream')} "
          f"residents={tokens.get('column_gather_residents')}")
    wit = row.get("witnesses") or {}
    print(f"      run witness: gathers={wit.get('gather_calls')} "
          f"async={wit.get('gather_async_calls')} "
          f"waits={wit.get('wait_calls')} "
          f"fan-outs={wit.get('broadcast_calls')} "
          f"copy streams={wit.get('copy_streams_created')}")
    print(f"      accumulation: view-loop calls={wit.get('view_range_calls')} "
          f"handed a block to add into={wit.get('accumulate_calls')} "
          f"same block returned again={wit.get('block_reuses')}")
    sequences = [f"{t.get('device')}#{t.get('device_index')}:{t['sequence']}"
                 for t in (wit.get("issue_trace") or ())]
    order_ok = wit.get("issue_order_ok")
    mark = ("" if order_ok is not False else
            f"   <-EXPECTED TO OPEN {wit.get('issue_order_expected')!r}; "
            f"this witness depends on thread scheduling and is reported, "
            f"not gating")
    print(f"      issue order (want {wit.get('issue_order_expected')!r}): "
          f"{sequences}{mark}")
    transfer = row.get("transfer") or {}
    print(f"      route: dev2dev_safe={transfer.get('dev2dev_safe')} "
          f"host_bounce_in_use={transfer.get('host_bounce_in_use')} "
          f"gather host wall {_fmt(transfer.get('gather_host_s_per_recon'), '.2f')} "
          f"s per reconstruction")
    header = (f"      {'device':<10}{'bracket_s':>11}{'busy_s':>10}"
              f"{'stall_s':>10}{'calls':>8}{'per_launch_ms':>15}"
              f"{'peak_GB':>10}{'resv_GB':>9}{'modeled_GB':>12}{'mod/meas':>10}")
    print(header)
    mem = (row.get("memory") or {})
    modeled = mem.get("modeled_peak_per_device") or []
    ratios = mem.get("ratio_per_device") or []
    for index, dev in enumerate(row.get("per_device") or []):
        model_gb = _gb(modeled[index]) if index < len(modeled) else None
        ratio = ratios[index] if index < len(ratios) else None
        mark = ""
        if isinstance(ratio, float) and ratio < 1.0:
            mark = " <-FLOOR VIOLATION"
        print(f"      {dev['device']:<10}"
              f"{_fmt(dev['bracket_span_s'], '11.3f')}"
              f"{_fmt(dev['busy_sum_s'], '10.3f')}"
              f"{_fmt(dev['stall_s'], '10.3f')}"
              f"{_fmt(dev['busy_calls'], '8.0f')}"
              f"{_fmt(dev['per_launch_ms'], '15.3f')}"
              f"{_fmt(_gb(dev['peak_bytes']), '10.2f')}"
              f"{_fmt(_gb(dev['reserved_bytes']), '9.2f')}"
              f"{_fmt(model_gb, '12.2f')}"
              f"{_fmt(ratio, '10.3f')}{mark}")


def _floored(spread, reference):
    """A spread, floored at SPREAD_FLOOR_FRAC of the control arm's reading.  Two
    warm passes is a weak estimator of run-to-run noise, and without a floor a
    freakishly tight pair would let the comparison resolve a difference it has
    no business resolving."""
    if spread is None:
        spread = 0.0
    if reference:
        return max(float(spread), SPREAD_FLOOR_FRAC * abs(float(reference)))
    return float(spread)


def comparison(rows, config):
    """The two trees at ONE configuration, reduced to the numbers the verdict
    line is drawn from.  Every reading here was measured in this job."""
    by_tree = {r.get("tree"): r for r in rows
               if not r.get("error") and r.get("arm_class") != "generator"
               and r.get("config_token") == config["token"]}
    entry = dict(config=config["token"], geometry=config["geometry"],
                 n=config["n_dev"], batch=config["pixel_batch"],
                 present=sorted(by_tree), resolvable=False)
    control = by_tree.get("control")
    if control is None or not all(t in by_tree for t in TREES):
        entry["why"] = (f"this configuration needs both trees and has "
                        f"{sorted(by_tree) or 'none'}")
        return entry
    entry["resolvable"] = True

    def read(tree, key, spread_key):
        row = by_tree[tree]
        return row.get(key), row.get(spread_key)

    for tree in TREES:
        row = by_tree[tree]
        entry[tree] = dict(
            token=row.get("token"),
            bracket_s=row.get("forward_bracket_max_s"),
            bracket_spread_s=row.get("forward_bracket_spread_s"),
            busy_s=row.get("forward_busy_max_s"),
            busy_spread_s=row.get("forward_busy_spread_s"),
            stall_s=row.get("forward_stall_max_s"),
            per_launch_ms=row.get("per_launch_ms"),
            composed_s=row.get("composed_s"),
            composed_spread_s=row.get("composed_spread_s"),
            min_ratio=(row.get("memory") or {}).get("min_ratio"),
            ratio_device=(row.get("memory") or {}).get("min_ratio_device"),
            floor_violation=(row.get("memory") or {}).get("floor_violation"),
            residents=((row.get("memory") or {}).get("column_gather_residents")
                       or (row.get("ledger") or {}).get(
                           "column_gather_residents")),
            peak_bytes=(row.get("memory") or {}).get("measured_peak_per_device"),
            modeled_bytes=(row.get("memory") or {}).get(
                "modeled_peak_per_device"),
            reserved_bytes=(row.get("memory") or {}).get(
                "reserved_warm_per_device"),
            # The three numbers ON the device that carried the printed ratio.
            measured_at_min=(row.get("memory") or {}).get("measured_at_min"),
            modeled_at_min=(row.get("memory") or {}).get("modeled_at_min"),
            reserved_at_min=(row.get("memory") or {}).get("reserved_at_min"),
            dev2dev_safe=row.get("dev2dev_safe"),
            host_bounce_in_use=row.get("host_bounce_in_use"),
            gpu_hot=row.get("gpu_hot"))

    c_bracket, c_bracket_sp = read("control", "forward_bracket_max_s",
                                   "forward_bracket_spread_s")
    c_busy, c_busy_sp = read("control", "forward_busy_max_s",
                             "forward_busy_spread_s")

    c_composed, c_composed_sp = read("control", "composed_s",
                                     "composed_spread_s")
    f_bracket, f_bracket_sp = read("fused", "forward_bracket_max_s",
                                   "forward_bracket_spread_s")
    f_busy, f_busy_sp = read("fused", "forward_busy_max_s",
                             "forward_busy_spread_s")
    f_composed, f_composed_sp = read("fused", "composed_s", "composed_spread_s")

    # THE VERDICT LINE, drawn on BUSY.  The work the change removes is
    # projector work -- a full-block pass and a full-block allocation per batch
    # -- and it ran on the compute stream between the kernel launches, inside
    # the per-body-call brackets that busy is the sum of.  So busy is where the
    # saving has to appear if it is real, and the bracket rides beside it
    # because the forward funnel is what a user waits for.  The fall has to
    # exceed the two arms' busy spreads together, each floored at
    # SPREAD_FLOOR_FRAC of the control reading.
    margin = _floored(c_busy_sp, c_busy) + _floored(f_busy_sp, c_busy)
    entry["busy_margin_s"] = margin
    entry["busy_delta_s"] = ((c_busy - f_busy)
                             if c_busy is not None and f_busy is not None
                             else None)
    entry["busy_saved_frac"] = ((entry["busy_delta_s"] / c_busy)
                                if entry["busy_delta_s"] is not None and c_busy
                                else None)
    entry["accumulation_saves"] = bool(
        entry["busy_delta_s"] is not None and entry["busy_delta_s"] > margin)
    entry["verdict_line"] = (
        f"ACCUMULATION SAVES: {config['token']}: busy {_fmt(c_busy, '.2f')} -> "
        f"{_fmt(f_busy, '.2f')} s, bracket {_fmt(c_bracket, '.2f')} -> "
        f"{_fmt(f_bracket, '.2f')} s"
        if entry["accumulation_saves"] else
        f"NO CHANGE: {config['token']}: busy {_fmt(c_busy, '.2f')} -> "
        f"{_fmt(f_busy, '.2f')} s, bracket {_fmt(c_bracket, '.2f')} -> "
        f"{_fmt(f_bracket, '.2f')} s")

    # THE COMPOSED FLAG, which is mg12's busy flag turned around.  There a busy
    # change was the surprise; here a busy DROP is the expected result and the
    # surprise is a composed wall that disagrees with it.  Taking work out of
    # the forward cannot make the whole reconstruction slower, so busy and
    # composed moving in OPPOSITE directions -- both moves outside their own
    # spreads, so neither is noise -- means something outside the forward moved
    # and every reading in the block has to be read with that in mind.
    composed_flags = []
    if None not in (c_busy, f_busy, c_composed, f_composed):
        busy_allowed = margin
        composed_allowed = (_floored(c_composed_sp, c_composed)
                            + _floored(f_composed_sp, c_composed))
        busy_move = c_busy - f_busy              # positive means fused is faster
        composed_move = c_composed - f_composed
        busy_resolved = abs(busy_move) > busy_allowed
        composed_resolved = abs(composed_move) > composed_allowed
        if (busy_resolved and composed_resolved
                and (busy_move > 0) != (composed_move > 0)):
            composed_flags.append(
                f"COMPOSED CHANGED: busy "
                f"{'fell' if busy_move > 0 else 'rose'} "
                f"{c_busy:.2f} -> {f_busy:.2f} s but the composed wall "
                f"{'rose' if composed_move < 0 else 'fell'} "
                f"{c_composed:.2f} -> {f_composed:.2f} s, both outside their "
                f"own spreads.  Removing work from the forward cannot make the "
                f"reconstruction slower, so something outside the forward moved "
                f"between these two arms; read every reading in this block with "
                f"that in mind")
    entry["composed_flags"] = composed_flags

    # THE MEMORY READING.  The ledger charges the same on both trees on purpose
    # (see the module docstring), so the modeled column should be IDENTICAL and
    # the question is whether the measured peak or the reserved bytes fell.  A
    # ratio below 1.00 on either tree is still a FLOOR VIOLATION.
    entry["floor_violations"] = [
        tree for tree in TREES if entry[tree].get("floor_violation")]
    c_model = entry["control"].get("modeled_bytes") or []
    f_model = entry["fused"].get("modeled_bytes") or []
    entry["modeled_identical"] = (bool(c_model) and c_model == f_model)
    c_peak = entry["control"].get("peak_bytes") or []
    f_peak = entry["fused"].get("peak_bytes") or []
    entry["peak_delta_bytes"] = ((max(c_peak) - max(f_peak))
                                 if c_peak and f_peak else None)
    c_res = entry["control"].get("reserved_bytes") or []
    f_res = entry["fused"].get("reserved_bytes") or []
    entry["reserved_delta_bytes"] = ((max(c_res) - max(f_res))
                                     if c_res and f_res else None)

    # UNSTABLE ARMS.  An arm whose own two warm passes disagree by more than
    # WARM_INSTABILITY_FRAC of its own reading is not a usable measurement, and
    # the combined-spread rule would let it confirm whatever was predicted.  The
    # verdict lines are drawn exactly as stated either way; this line is printed
    # beside them so neither is read on its own.
    unstable = []
    for tree in TREES:
        block = entry[tree]
        for name, value_key, spread_key in (
                ("bracket", "bracket_s", "bracket_spread_s"),
                ("busy", "busy_s", "busy_spread_s"),
                ("composed", "composed_s", "composed_spread_s")):
            value, spread_value = block.get(value_key), block.get(spread_key)
            if not value or spread_value is None:
                continue
            frac = abs(spread_value) / abs(value)
            if frac > WARM_INSTABILITY_FRAC:
                unstable.append(dict(tree=tree, reading=name, frac=frac,
                                     value=value, spread=spread_value))
    entry["unstable"] = unstable
    entry["unstable_line"] = (
        "" if not unstable else
        "UNSTABLE ARM(S) IN THIS COMPARISON: "
        + "; ".join(f"{u['tree']} {u['reading']} {u['value']:.3f} s with a warm "
                    f"spread of {u['spread']:.3f} s ({u['frac']:.0%} of its own "
                    f"reading)" for u in unstable)
        + f".  Above {WARM_INSTABILITY_FRAC:.0%} the two passes are not "
          f"measuring the same thing, so the line(s) below rest on a reading "
          f"that cannot carry them.")
    return entry


def route_note(entry):
    """One line saying which transfer route this configuration's arms took, and
    one line saying what a result therefore means.  Neither changes a verdict."""
    safe = [entry[tree].get("dev2dev_safe") for tree in TREES]
    bounced = [entry[tree].get("host_bounce_in_use") for tree in TREES]
    fact = (f"dev2dev_safe={safe}, host bounce in use={bounced} "
            f"(control, fused)")
    if any(bounced) or not all(safe):
        reading = (
            "READ THIS BLOCK ACCORDINGLY: at least one arm routed its "
            "cross-device transfers through host memory.  That does not stop "
            "the accumulation change from working -- the work it removes is "
            "projector work and does not touch the wire -- but a host bounce "
            "synchronizes, so the forward spends far more of itself waiting "
            "and the same absolute saving is a smaller share of the bracket.  "
            "Read the BUSY column, which is where the saving lives, and treat "
            "the bracket beside it as diluted.  This line explains; it changes "
            "no verdict.")
    else:
        reading = ("Direct device-to-device transfer is in use on both arms, "
                   "which is the route both were meant to be measured on.")
    return fact, reading


def print_comparison(entry, values):
    """ONE block per configuration: mechanical, quotable, and carrying no prose
    beyond the two verdict lines and the reading of the transfer route."""
    print(f"\n=== {entry['geometry']} n={entry['n']} "
          f"batch {entry['batch']}  [{entry['config']}] ===")
    if not entry.get("resolvable"):
        print(f"  NOT RESOLVABLE -- {entry.get('why')}")
        return
    fact, reading = route_note(entry)
    print(f"  ROUTE: {fact}")
    print(f"         {reading}")

    print("  SPEED  (largest device, median of the warm passes; the spread "
          "beside each reading is")
    print(f"          that arm's own max minus min over those passes.  The "
          f"comparison never resolves")
    print(f"          a difference smaller than {SPREAD_FLOOR_FRAC:.0%} of the "
          f"control reading.")
    print("          BUSY IS THE PRIMARY READING: the removed work ran on the "
          "compute stream")
    print("          between the kernel launches, inside the brackets busy is "
          "the sum of.)")
    print(f"      {'tree':<10}{'bracket_s':>11}{'(spread)':>10}{'busy_s':>10}"
          f"{'(spread)':>10}{'stall_s':>10}{'per_launch_ms':>15}"
          f"{'composed_s':>12}{'(spread)':>10}")
    for tree in TREES:
        block = entry[tree]
        print(f"      {tree:<10}"
              f"{_fmt(block['bracket_s'], '11.3f')}"
              f"{_fmt(block['bracket_spread_s'], '10.3f')}"
              f"{_fmt(block['busy_s'], '10.3f')}"
              f"{_fmt(block['busy_spread_s'], '10.3f')}"
              f"{_fmt(block['stall_s'], '10.3f')}"
              f"{_fmt(block['per_launch_ms'], '15.3f')}"
              f"{_fmt(block['composed_s'], '12.2f')}"
              f"{_fmt(block['composed_spread_s'], '10.2f')}")
    if entry.get("busy_saved_frac") is not None:
        print(f"      busy moved {entry['busy_saved_frac']:+.1%} against a "
              f"combined spread of {entry.get('busy_margin_s') or 0.0:.3f} s "
              f"(positive means the fused tree is faster)")
    for flag in entry.get("composed_flags") or ():
        print(f"      {flag}")
    if not entry.get("composed_flags"):
        print("      busy and the composed wall moved in the same direction, "
              "or neither move was")
        print("      outside its own spread, which is what a change confined "
              "to the forward does.")

    print("  MEMORY (modeled/measured; at or above 1.00 passes, below 1.00 is a "
          "FLOOR VIOLATION.")
    print("          The ledger charges the SAME on both trees on purpose -- "
          "the two-block charge is")
    print("          shared with the banded path, which really does hold two, "
          "and over-charging is")
    print("          legal where under-charging is not.  So the modeled column "
          "should be identical")
    print("          and the reading of interest is whether the MEASURED peak "
          "or the reserved bytes")
    print("          fell.  The three byte columns are the readings ON the "
          "device that carried the")
    print("          ratio.)")
    print(f"      {'tree':<10}{'residents':>11}{'measured_GB':>13}"
          f"{'modeled_GB':>12}{'reserved_GB':>13}{'min mod/meas':>14}"
          f"{'device':>10}")
    for tree in TREES:
        block = entry[tree]
        mark = " <-FLOOR VIOLATION" if block.get("floor_violation") else ""
        print(f"      {tree:<10}"
              f"{_fmt(block.get('residents'), '11.0f')}"
              f"{_fmt(_gb(block.get('measured_at_min')), '13.2f')}"
              f"{_fmt(_gb(block.get('modeled_at_min')), '12.2f')}"
              f"{_fmt(_gb(block.get('reserved_at_min')), '13.2f')}"
              f"{_fmt(block.get('min_ratio'), '14.3f')}"
              f"{str(block.get('ratio_device')):>10}{mark}")
    if entry.get("modeled_identical") is False:
        print("      NOTE: the modeled peaks are NOT identical on the two "
              "trees.  The overlay's only")
        print("      change to the ledger is a comment, so a moved number here "
              "means the two arms")
        print("      priced different configurations and the ratios are not "
              "comparable.")
    if entry.get("peak_delta_bytes") is not None:
        print(f"      measured peak fell by "
              f"{_gb(entry['peak_delta_bytes']):+.3f} GB and reserved by "
              f"{_fmt(_gb(entry.get('reserved_delta_bytes')), '+.3f')} GB "
              f"(positive means the fused tree used less)")

    print("  VALUE  (distance to the control arm of THIS configuration, with "
          "the repeat floor beside")
    print("          it.  The design claim is that the values are BIT-"
          "IDENTICAL: per element the")
    print("          same summands are added in the same order, and only where "
          "the addition happens")
    print("          changed.  So the distance below should sit at the control "
          "arm's own repeat")
    print("          floor or under it.  That floor is the same code "
          "reconstructing the same input")
    print("          twice, and it is not zero, because both forward kernels "
          "accumulate with float")
    print("          atomics.)")
    print(f"      {'tree':<10}{'own_rel_L2':>13}{'own_maxrel':>13}"
          f"{'vs_ctl_L2':>13}{'vs_ctl_maxrel':>15}{'own_cs':>12}"
          f"{'vs_ctl_cs':>12}")
    exact_zero = False
    for tree in TREES:
        found = next((v for v in values
                      if v.get("config") == entry["config"]
                      and v.get("tree") == tree), None)
        if found is None:
            continue
        own = found.get("own_pass_to_pass") or {}
        against = found.get("vs_control") or {}
        print(f"      {tree:<10}"
              f"{_fmt(own.get('rel_l2'), '13.3e')}"
              f"{_fmt(own.get('max_rel_of_peak'), '13.3e')}"
              f"{_fmt(against.get('rel_l2'), '13.3e')}"
              f"{_fmt(against.get('max_rel_of_peak'), '15.3e')}"
              f"{_fmt(found.get('checksum_repeat_spread'), '12.3e')}"
              f"{_fmt(found.get('checksum_vs_control'), '12.3e')}")
        if (tree == "fused" and against
                and against.get("rel_l2") == 0.0
                and against.get("max_abs") == 0.0):
            exact_zero = True
    if exact_zero and DEVICE == "cuda":
        # Exact agreement is what "bit-identical" sounds like it should give,
        # and on this hardware it is the one reading that should NOT happen.
        print("      NOTE: the fused arm reads EXACTLY zero against the "
              "control.  Both forward")
        print("      kernels accumulate with float atomics, which are not "
              "run-to-run reproducible on")
        print("      CUDA -- the control's own repeat floor in the same row is "
              "the evidence, and it")
        print("      is not zero.  Two DIFFERENT runs therefore cannot normally "
              "agree bit for bit,")
        print("      whatever the source says, so an exact zero here is more "
              "likely a harness fault")
        print("      -- the same sample file read twice -- than a result.  "
              "Compare the two rows'")
        print("      value_sample_paths before reading it as agreement.")

    if entry.get("unstable_line"):
        print(f"  {entry['unstable_line']}")
    print(f"  {entry['verdict_line']}")


def fingerprint_check(rows):
    """THE CROSS-ARM PROVENANCE CHECK, which no single arm can make.

    Each arm proves it imported from the root it was given.  What no arm can see
    is whether the two roots hold two different trees, and a job that pointed
    both at one directory would report NO CHANGE at every configuration and look
    tidy doing it.  So the digests are compared across arms against what the
    build is supposed to have produced: control and fused must differ in all
    three fingerprinted files.  The ledger's difference is comment only and
    moves no number, which is checked separately -- the comparison blocks assert
    the two trees' MODELED peaks come out identical.
    """
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by_tree = {}
    for row in live:
        digests = ((row.get("fingerprint") or {}).get("sha256_12") or {})
        by_tree.setdefault(row.get("tree"), set()).add(
            tuple(sorted(digests.items())))
    findings, ok = [], True
    for tree, seen in sorted(by_tree.items()):
        if len(seen) > 1:
            ok = False
            findings.append(f"the {tree!r} arms did not all import the same "
                            f"bytes: {sorted(seen)}")
    one = {tree: dict(next(iter(seen))) for tree, seen in by_tree.items() if seen}

    def compare(a, b, name, want_same):
        if a not in one or b not in one:
            return
        same = one[a].get(name) == one[b].get(name)
        if same != want_same:
            findings.append(
                f"{name}: {a} and {b} are "
                f"{'the same bytes' if same else 'different bytes'} and the "
                f"build says they should be "
                f"{'the same' if want_same else 'different'} "
                f"({a}={one[a].get(name)}, {b}={one[b].get(name)})")

    for name in FINGERPRINT_FILES:
        compare("control", "fused", name, False)
    ok = ok and not findings
    return dict(ok=ok, per_tree=one, findings=findings)


def summarize(rows, out_path):
    """The whole read-out: the per-arm tables, then one comparison block per
    configuration, then the recap."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    failed = [r for r in rows if r.get("error")]
    print("\n" + "=" * 78)
    print(f"mg14 -- the fused accumulation measurement, {len(live)} arms on "
          f"{RUN_LABEL} ({DEVICE})")
    print("=" * 78)
    print("  Two trees, measured in this job, against one another: the tip, "
          "where the projector")
    print("  allocates a block per pixel batch and the driver adds it into the "
          "running total, and")
    print("  the same tree with that addition moved INSIDE the projector's view "
          "loop, into the")
    print("  block it was going to write anyway.  Nothing here is compared "
          "against a number")
    print("  measured anywhere else.")
    for tree in TREES:
        print(f"    {tree:<9} {TREE_NOTE[tree]}")
    if DEVICE != "cuda":
        print()
        print("  SMOKE RUN -- READ THE VERDICT LINES AS EXERCISE, NOT AS "
              "RESULT.  The cell is a few")
        print("  dozen voxels, far below the size at which a full-block pass "
              "costs anything, so a")
        print("  speed reading either way is noise; and there is no device peak "
              "counter, so the")
        print("  memory ratio is NOT RESOLVABLE.  What the smoke does establish "
              "is the part that")
        print("  does not depend on scale: every witness fires, the two trees "
              "are told apart, the")
        print("  accumulation is seen happening, and the value distances are "
              "computed.")

    for row in live:
        print_arm_table(row)

    print("\n  -- the tree fingerprints, compared ACROSS arms --")
    check = fingerprint_check(rows)
    for tree, digests in sorted(check["per_tree"].items()):
        print(f"      {tree:<10}" + "  ".join(f"{k}={v}"
                                              for k, v in sorted(digests.items())))
    if check["ok"]:
        print("      the two trees differ in every file the overlay "
              "replaces, which is what the")
        print("      build says they should do")
    else:
        for finding in check["findings"]:
            print(f"      PROVENANCE PROBLEM: {finding}")
        print("      A job whose two roots are not two different trees would "
              "report NO CHANGE at")
        print("      every configuration and look tidy doing it.  Do not read "
              "the blocks below.")

    values = value_table(rows)
    entries = []
    print("\n" + "=" * 78)
    print("  THE COMPARISON BLOCKS, one per configuration")
    print("=" * 78)
    for config in configurations():
        entry = comparison(rows, config)
        entries.append(entry)
        print_comparison(entry, values)

    if failed:
        print(f"\n  {len(failed)} ARM(S) FAILED and are in no block above:")
        for row in failed:
            first = str(row.get("error", "")).strip().splitlines()
            print(f"      {row.get('token')}: "
                  f"{first[-1] if first else 'unknown'}")

    print("\n" + "=" * 78)
    print("  THE LINES")
    print("=" * 78)
    for entry in entries:
        if entry.get("resolvable"):
            if entry.get("unstable_line"):
                print(f"  {entry['unstable_line']}")
            for flag in entry.get("composed_flags") or ():
                print(f"  {flag}")
            print(f"  {entry['verdict_line']}")
        else:
            print(f"  NOT RESOLVABLE: {entry['geometry']} n={entry['n']} "
                  f"batch={entry['batch']}: {entry.get('why')}")

    # The claim was that the saving GROWS with the batch count, so the two
    # parallel configurations at one device count and two batches are the pair
    # that speaks to it.  Stated as two readings side by side and nothing else.
    swept = [e for e in entries
             if e.get("resolvable") and e.get("busy_saved_frac") is not None
             and e["geometry"] == "parallel" and e["n"] == 4]
    if len(swept) > 1:
        print()
        print("  THE BATCH-COUNT READING (the claim was that the saving grows "
              "with the number of")
        print("  batches, and halving the pixel batch doubles that number):")
        for entry in sorted(swept, key=lambda e: e["batch"]):
            print(f"    batch {entry['batch']:>6}: busy "
                  f"{entry['control']['busy_s']:.3f} -> "
                  f"{entry['fused']['busy_s']:.3f} s, "
                  f"{entry['busy_saved_frac']:+.1%}")

    violations = sorted({f"{e['config']}-{tree}"
                         for e in entries
                         for tree in (e.get("floor_violations") or ())})
    print()
    if violations:
        print(f"  MEMORY FLOOR VIOLATION on {violations}: on those arms the "
              f"ledger modeled less")
        print("  than the run measured, which is the one gate in this job that "
              "can fail an increment.")
    else:
        print("  MEMORY: no arm read modeled/measured below 1.00.")

    off_order = [r.get("token") for r in live
                 if r.get("issue_order_ok") is False]
    if off_order:
        print(f"\n  ISSUE ORDER: {off_order} did not open the way their tree's "
              f"driver issues work.")
        print("  This witness is recorded, not gating: it is the only reading "
              "here that depends on")
        print("  thread scheduling, and the byte fingerprint, the projector "
              "signature and the")
        print("  accumulation counts all agreed on which tree ran.  Read the "
              "per-arm traces above")
        print("  before concluding anything from it.")

    hot = [r.get("token") for r in live if r.get("gpu_hot")]
    print(f"\n  throttle rule: sw_power_cap at normal temperature is recorded "
          f"and KEPT; {len(hot)} arm(s) ran hot or clock-depressed: {hot}")

    print(f"\nrows: {out_path}")
    return dict(comparisons=entries, values=values, fingerprints=check,
                floor_violations=violations, hot_arms=hot)


# ── the wall arithmetic ───────────────────────────────────────────────────────
def wall_estimate(generators, measured):
    """Low and high wall estimates, in seconds.

    The base is mg11's MEASURED per-arm subprocess wall for the same
    configuration on the same node class, less one composed reconstruction --
    mg11 ran 1 cold plus 3 warm passes per arm and this job runs 1 cold plus 2.
    A control arm is that base; the fused arms have never been run, so they take
    the same rule mg10, mg11 and mg12 used for an unmeasured arm: the low end
    assumes the arm costs what its control costs, the high end assumes half
    again as much.  That high end is deliberately pessimistic here -- the change
    is expected to make the fused arm FASTER, not slower -- and it is kept
    anyway, because an estimate that assumes its own result is not an estimate.
    On top of that sits an allowance for a COLD INDUCTOR CACHE: this job gets
    its own, so the first arm at each launch shape compiles from scratch.  The
    allowance is smaller than mg12's because there are three launch shapes here
    rather than five."""
    low = high = GENERATOR_S * len(generators)
    high = int(high * CONTROL_HIGH_FACTOR)
    for cfg in measured:
        key = (cfg["geometry"], cfg["n_dev"], cfg["pixel_batch"])
        reading = MG11_READING.get(key)
        if reading is None:
            base = 300
        else:
            base = max(60, reading["wall_s"] - reading["composed_s"])
        low += base
        high += int(base * (CONTROL_HIGH_FACTOR if cfg["tree"] == "control"
                            else OVERLAY_HIGH_FACTOR))
    return low + COLD_CACHE_LOW_S, high + COLD_CACHE_HIGH_S


def main():
    arms = selected_arms()
    generators, measured = build_plan(arms)
    if "--dry-run" in sys.argv:
        low, high = wall_estimate(generators, measured)
        num_slices = num_slices_for(cell_for("cone"))
        print(f"mg14 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arms")
        print(f"  cell {cell_for('cone')}, slices {num_slices}, warm repeats "
              f"{WARM_REPEATS}, iterations {VCD_ITERATIONS}, device {DEVICE}, "
              f"results {RESULTS_DIR}")
        print(f"  device pin: {PIN_MECHANISM}; "
              f"{COLUMN_GATHER_ENV_VAR} is removed from every child environment "
              f"because the column gather is the default on both trees")
        print(f"  the claim, from a CPU reading and compared against nothing "
              f"here: "
              + ", ".join(f"{frac:.1%} of the forward at {count} batches"
                          for count, frac in sorted(CPU_CLAIM.items())))
        print("  the two trees:")
        for tree in TREES:
            try:
                root = tree_root(tree)
            except RuntimeError as exc:
                print(f"    {tree:<9} NOT SET -- {exc}")
                continue
            present = os.path.isdir(os.path.join(root, "mbirtorch"))
            print(f"    {tree:<9} {root}"
                  f"{'' if present else '   [NO mbirtorch PACKAGE THERE]'}")
            print(f"    {'':<9} {TREE_NOTE[tree]}")
        for cfg in generators:
            print(f"  {cfg['arm_id']:<48} (generator, control tree, one device)")
        # The batch COUNT is what the saving is supposed to scale with, so the
        # plan prints it rather than leaving the reader to divide.
        pixels = num_slices * num_slices    # the full pixel grid at this cell
        for cfg in measured:
            cyl = cylinder_bytes(cfg["pixel_batch"], num_slices)
            batches = -(-pixels // cfg["pixel_batch"])
            reading = MG11_READING.get((cfg["geometry"], cfg["n_dev"],
                                        cfg["pixel_batch"]))
            memo = ("" if reading is None else
                    f"  [mg11 read busy {reading['busy_s']:.2f} s, bracket "
                    f"{reading['bracket_s']:.2f} s here]")
            print(f"  {cfg['arm_id']:<48} tree={cfg['tree']:<7} "
                  f"cylinder {cfg['pixel_batch']} x {num_slices} = "
                  f"{cyl / 1e6:.1f} MB, about {batches} batches per pass"
                  + (memo if cfg["tree"] == "control" else ""))
        print(f"  wall estimate {low / 60:.0f} to {high / 60:.0f} minutes "
              f"(mg11's measured walls less one reconstruction as the base; "
              f"the fused arms have never been run, so their high end assumes "
              f"half again as much even though the change is expected to make "
              f"them faster; plus {COLD_CACHE_LOW_S / 60:.0f} to "
              f"{COLD_CACHE_HIGH_S / 60:.0f} minutes for a cold inductor "
              f"cache).")
        print("  if it must be cut: MG14_CONFIGS drops whole configurations by "
              "token and MG14_ARMS")
        print("  drops single arms.  Trim WHOLE CONFIGURATIONS -- a "
              "configuration missing either")
        print("  tree has no comparison block at all, and the control arm is "
              "where the value floor")
        print("  comes from.  Trim c4b08192 first, then p4b08192; p4b04096 has "
              "the most batches and")
        print("  so the widest lever, and it is the last thing to go.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg14_accum_gate_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg14 accumulation gate on {RUN_LABEL} ({DEVICE}); {len(measured)} arms "
          f"-> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY: a truncated job still yields the configurations
    # it finished, which is why the arm order runs a configuration's two trees
    # back-to-back.
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
    if os.environ.get("MG14_KEEP_ARTIFACTS", "0") != "1":
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
    # BEFORE ANYTHING IMPORTS mbirtorch: put this arm's tree ahead of the
    # harness's own directory on sys.path (see put_tree_first).
    put_tree_first(cfg["tree"])
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
