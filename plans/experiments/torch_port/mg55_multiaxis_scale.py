"""mg55 -- THE MULTIAXIS KERNELS ACROSS SEVERAL DEVICES: DOES SHARDING NOW
DIVIDE THE PER-DEVICE PEAKS AND PAY IN SPEED, AND DOES A 2048-CLASS
RECONSTRUCTION COMPLETE ON A FOUR-GPU NODE?

WHY THIS RUN EXISTS.  The two hand-written multiaxis kernels have now been
timed against the torch bodies they replace, on one device, at three cells:
the kernel route ran at 0.25, 0.23 and 0.22 of the torch route's warm wall --
between 4.0 and 4.6 times faster -- and its peak device memory fell with it,
from 11.38 to 1.96 GB, 15.12 to 6.54 GB and 34.75 to 24.11 GB.  The reason the
memory moved is the same reason the time did: the torch bodies held about
fourteen slabs of (view batch, pixels, max(rows, slices)) floats at once, and a
kernel body holds the problem arrays plus a small per-(view, pixel) contract
instead, so the largest temporary class is simply gone.

Two questions are left that only several devices can answer, and this run
measures both.

    FIRST, WHETHER SHARDING NOW PAYS.  With the torch bodies bound, adding
    devices did not divide the per-device peak: the slab temporaries were sized
    by the view batch and the pixel count rather than by the shard, so they
    stayed on every device.  With those temporaries gone the per-device peak
    should fall roughly as the shard does, and the time should improve or at
    worst stay flat.  Neither has been measured.  This run reconstructs the
    same 1024-class problem at one, two and four devices on the kernel route
    and records the walls and the per-device peaks.

    SECOND, WHETHER THE 2048-CLASS RUNS AT ALL.  Its volume is eight times the
    1024-class volume and it has twice as many views, and no single device can
    hold it.  With the torch bodies bound it could not be run at any device
    count, because sharding did not divide their peaks.  This run tries it on
    the standard four-GPU node and records what happens -- including running
    out of memory, which is a reading and not a fault.

THIS RUN DECIDES NOTHING.  It edits no library file and writes no kernel.  It
varies one thing, the device list handed to the model, and it varies it in a
fresh process per arm.  The exit code reports whether the instrument worked;
the readings are read by a person.

TERMS, defined once here.
    cell        one problem to measure: a multiaxis geometry and a sinogram
                shape.
    staging     the untimed step that builds one cell's sinogram and weights.
                For the 1024-class that is a file on disk with an md5 beside
                it, written once and reused; for the 2048-class it happens
                inside the arm's own process (see PART B).
    arm         one device layout reconstructing one cell, in its own process.
    cold pass   the first reconstruction in a process.  It pays whatever
                first-call cost the route has, so it is timed and reported but
                never used for a ratio.
    warm pass   a reconstruction after the cold one.  The row carries their
                median and their spread.
    fingerprint two float64 reductions of a reconstruction -- the sum of
                absolute values and the sum of squares -- accumulated on the
                DEVICE in float64, in fixed-size chunks, and read back.

THE DEVICE LIST IS EXPLICIT, AND THAT IS DELIBERATE.  Every arm calls
``configure_devices(devices=[...])`` with the devices named one by one, rather
than asking for a count and letting the library choose.  The automatic policy
and the speed thresholds that order its candidates are NOT the subject here:
they decide whether a layout is worth using, and this run is measuring what a
layout costs and what it returns.  Naming the devices makes the layout a fact
on the row instead of a policy outcome, and it keeps a two-device arm from
quietly settling on four because the policy preferred them.

PART A -- THE 1024-CLASS ACROSS DEVICE COUNTS.  The cell is
(1024, 1008, 992), built exactly as the one-device measurement built it: two
angles per view, azimuths evenly spaced over half a turn, elevations swept
across +/- 0.5 radians, seed 13.  Three arms, one per device count -- 1, 2 and
4 -- each a fresh process on the default kernel selection.  Each arm runs a
seeded 3-iteration cold reconstruction with the stopping threshold disabled,
then MG55_WARM more, and records the bound body names, the walls, the PER-DEVICE
peak allocated and reserved bytes (reset and read on every device in the list,
kept as a list, with the busiest device's reading beside them) and a float64
fingerprint of the reconstruction.  The parent then computes the speedup of
each count against the one-device arm and the fingerprint differences across
counts.  A difference above 1e-3 relative gets a note and never a failure:
what accumulates over three iterations at different device counts is summation
order, which the library's own multi-device tolerance already covers.

    THE STAGED INPUT IS REUSED.  The 1024-class sinogram and weights are the
    same npz the one-device measurements read, md5
    798c72e1cf5bb7803b9f2b02294753c6, about 7.6 GiB.  This run searches its own
    results directory and then the two directories that hold it,
    results/mg54_multiaxis_kernel_ab and results/mg52_framework_anchor,
    verifies the md5 of whatever it finds and reuses it; only when no verified
    copy exists anywhere does it stage fresh, under this run's own results
    directory and under the same filename.  A fresh stage runs with
    MBIRTORCH_DISABLE_TRITON=1, which is the library's own kill switch, so its
    sinogram is a torch-body forward projection like the file already on disk
    rather than a new input nobody else has reconstructed.

PART B -- THE 2048-CLASS DEMONSTRATION.  The cell is (2048, 2016, 1984), this
family's shapes doubled, with azimuths over a half turn and elevations across
+/- 0.5 radians as everywhere else, seed 13.  Two arms: four devices, which is
the demonstration, and two devices, which is optional and on by default.  Each
is a fresh process under its own 75-minute cap.

    THERE IS NO NPZ.  The arm builds its own input: the phantom exactly as the
    staging builds one, forward projected through the same model, and the
    weights from the same formula.  A staged file would be about 33 GB of
    sinogram and another 33 GB of weights, no other arm needs those same bytes,
    and writing and re-reading them would buy nothing.  The staging wall is
    recorded on the row, separately from the reconstruction walls.

    THE SINGLE-DEVICE CASE IS NOT RUN.  It is priced instead, in PART C.  The
    volume alone is about 34 GB and the sinogram about 33, so one device
    holding the whole problem plus a reconstruction's working set is not a
    measurement worth an hour of a four-GPU node.

    THE MEMORY PREFLIGHT IS TURNED OFF ON THESE ARMS, and the row says so in
    one sentence: the memory ledger is recorded as modeling the 1024-class at
    68 GB on one device against a measured peak of 34.75 GB, so a preflight
    reading about twice the real peak could refuse a run that fits.  The rows
    PART C records are the input to the correction that follows this run.

    Each 2048 arm records the bound bodies, the staging wall, the cold and warm
    reconstruction walls, the per-device peaks, the fingerprint, and whether
    the forward-model error metric fell over the iterations, read from the
    convergence record the reconstruction returns.  All of it is recorded and
    none of it is a threshold.  An out-of-memory or a timeout is RECORDED as
    that arm's result and the run continues.

PART C -- WHAT THE MEMORY LEDGER MODELS.  Last, in the parent process, read
only: for both cells at device counts 1, 2 and 4, the modeled planned
per-device peak for a reconstruction, taken from the library's own ledger
through its public entry points -- a plan built from the model over a candidate
device list, and the estimate made from that plan.  Both are pure arithmetic:
no device is queried and nothing is allocated, so a count the model is not
configured for can be priced.  One row per (cell, count), carrying the modeled
bytes beside the measured peaks from PARTS A and B wherever those exist.

    IT RUNS LAST FOR A REASON.  It is the only part that does its work in the
    parent process, and a parent that has already initialized CUDA and built
    models holds a context and allocator state on the devices.  Run first, that
    would sit inside every arm's peak reading.

OUTPUT.  One jsonl under MG55_RESULTS, named
mg55_multiaxis_scale_<node>_<stamp>.jsonl: a header row carrying the torch and
mbirtorch identity, the GPUs, the tree witnesses and the staged file's md5; the
staging row; one row per arm; the scaling comparison across device counts; one
ledger pricing row per (cell, count); and a summary row.  Rows are flushed as
they are written, so a job that runs out of wall time still yields everything
it finished, and MG55_ARMS re-runs the rest.

Run:
    <torch python> mg55_multiaxis_scale.py           on a four-GPU node
    MG55_DRY=1 <any python> mg55_multiaxis_scale.py       the plan, then stop
    MG55_SMOKE=1 <python> mg55_multiaxis_scale.py         the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized arm name is an error, not a silent
skip.
    MG55_RESULTS=<dir>         where the jsonl and any fresh staging go
    MG55_ARMS=a,b              subset of the arms, by arm name
    MG55_WARM=3                warm reconstructions after the 1024-class cold
    MG55_2048_WARM=2           warm reconstructions after the 2048-class cold
    MG55_2048_N2=1             set 0 to skip the optional two-device 2048 arm
    MG55_ARM_TIMEOUT_MIN=45    the per-arm cap for the 1024-class arms
    MG55_2048_TIMEOUT_MIN=75   the per-arm cap for the 2048-class arms
    MG55_DRY=1                 print the plan and exit; imports no torch
    MG55_SMOKE=1               the local CPU smoke
    MG55_CHILD=<path>          internal: an arm's job description.  Its
                               presence puts this process in child mode.
    MG55_CHILD_OUT=<path>      internal: where that child writes its row

THE LOCAL SMOKE runs the whole flow at two tiny cells on the CPU, and it
degrades in three places rather than pretending otherwise.  There is no triton
on a CPU install at all, so both projection bodies are the torch ones and every
arm row says so.  There is one device, so PART A collapses to a single
subprocess arm -- one device count, not three -- and the scaling comparison it
feeds has nothing to compare.  PART B runs one arm at a tiny doubled cell, so
its in-process staging, its convergence reading and the child protocol are all
exercised, and its row records that the cell and the device count are not the
ones the demonstration is about.  PART C prices counts 1, 2 and 4 with the one
CPU device repeated, which exercises the ledger's public entry points but
models bytes no CUDA layout would see, and its rows say so.  The smoke is
plumbing, not a measurement.
"""

import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────


def _flag(name, default="0"):
    """An environment flag that must read exactly "0" or "1".

    Accepting "true" or "yes" silently as false has cost this work a repeat
    before: the run prints the plan it was asked for and measures another one.
    """
    raw = os.environ.get(name, default).strip()
    if raw not in ("0", "1"):
        raise ValueError("{}: {!r} is not 0 or 1".format(name, raw))
    return raw == "1"


def _positive_int(name, default):
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError("{}: {!r} is not an integer".format(name, raw)) from None
    if value < 1:
        raise ValueError("{}: {} is not at least 1".format(name, value))
    return value


SMOKE = _flag("MG55_SMOKE")
DRY = _flag("MG55_DRY")
#: The arm's subprocess mode: the path to the job description this process is
#: to run.  Non-empty means child mode.  The sbatch unsets it, so a stray value
#: in the submitting shell cannot turn the real run into an arm.
CHILD = os.environ.get("MG55_CHILD", "").strip()
CHILD_OUT = os.environ.get("MG55_CHILD_OUT", "").strip()
DEVICE = "cpu" if SMOKE else "cuda"

#: The 1024-class cell, and the device counts it is measured at.  These are the
#: shapes the one-device kernel reading was taken at, so this run's one-device
#: arm lies beside a number that already exists.
CELL_1024 = dict(name="multiaxis_1024", cell=(1024, 1008, 992), part="A",
                 counts=(1, 2, 4))
#: The 2048-class cell: this family's shapes doubled.  Four devices is the
#: demonstration; two is optional and on by default.  One device is not run at
#: all -- it is priced in PART C instead.
CELL_2048 = dict(name="multiaxis_2048", cell=(2048, 2016, 1984), part="B",
                 counts=(4, 2))
#: The smoke's stand-ins, in the same relationship: the second is the first
#: doubled.  Both are tiny enough that the whole flow runs in under a minute on
#: a laptop CPU.
SMOKE_1024 = dict(name="multiaxis_smoke", cell=(128, 16, 16), part="A",
                  counts=(1,))
SMOKE_2048 = dict(name="multiaxis_smoke_2x", cell=(256, 32, 32), part="B",
                  counts=(1,))

#: Whether the optional two-device 2048-class arm runs.  On by default: a
#: 2048-class reconstruction that completes on two devices says something the
#: four-device arm cannot, and one that fails on two while completing on four
#: says where the boundary is.
RUN_2048_N2 = _flag("MG55_2048_N2", "1")

# ── the reconstruction protocol ───────────────────────────────────────────────
#: A seeded 3-iteration reconstruction with the stopping threshold disabled, so
#: every arm does exactly the same amount of work.  Not a knob: the recorded
#: one-device walls this run's n=1 arm should reproduce were measured at three.
VCD_ITERATIONS = 3
#: The seed, reset immediately before every reconstruction.  The library draws
#: its pixel partitions from numpy's global generator, so this is the same
#: mechanism at every device count.
VCD_SEED = 13
WARM_REPEATS = _positive_int("MG55_WARM", 3)
#: Fewer warm passes at the 2048-class, because each one is estimated at about
#: eighteen minutes and the arm has a cap to fit inside.
WARM_REPEATS_2048 = _positive_int("MG55_2048_WARM", 2)
#: The multiaxis elevation sweep, in radians, and the geometry's clamp on the
#: smallest |cos(elevation)|.  Both are mirrored here so the dry plan can print
#: the recon shape without importing anything; the real run reads the model's
#: own recon_shape and records whether the two agreed.
ELEVATION_HALF_RANGE = 0.5
MIN_COS_ELEVATION = 0.1

#: The per-arm hard time caps.  An arm that exceeds its cap is killed and the
#: timeout is recorded as that arm's result.
ARM_TIMEOUT_S = 60.0 * float(os.environ.get("MG55_ARM_TIMEOUT_MIN", "45"))
ARM_2048_TIMEOUT_S = 60.0 * float(os.environ.get("MG55_2048_TIMEOUT_MIN", "75"))
#: The identity probe imports torch and mbirtorch and reads the tree witnesses.
#: It does no real work, so ten minutes is generous even with a cold module
#: cache on a shared filesystem.
PROBE_TIMEOUT_S = 600.0
#: The staging job verifies an existing file's md5 -- about 14 seconds for the
#: 7.6 GiB 1024-class npz when it was last read -- or builds and writes one.
#: Given its own cap so a slow filesystem does not eat an arm's budget.
STAGE_TIMEOUT_S = 3600.0

# ── how a body is recognized ──────────────────────────────────────────────────
#: Every hand-written multiaxis body's name ends in this.  The kernel wrappers
#: are _multiaxis_forward_view_batch_triton and _multiaxis_back_view_batch_triton
#: in mbirtorch/triton_multiaxis.py; the torch bodies they replace are
#: _multiaxis_forward_view_batch and _multiaxis_back_view_batch.
KERNEL_BODY_SUFFIX = "_view_batch_triton"

# ── recorded context, not thresholds ──────────────────────────────────────────
#: What one device measured on the kernel route at the 1024-class cell, read in
#: session from the run record and rows of the one-device comparison
#: (results row file mg54_multiaxis_kernel_ab_h001_20260822_094414.jsonl).
#: Printed beside this run's one-device arm so a reader can see at a glance
#: whether it landed where it landed before.  Nothing is gated on it.
RECORDED_ONE_DEVICE_KERNEL = dict(
    cold_s=69.53, warm_s=67.64,
    peak_bytes=25889317888, reserved_bytes=39717961728)
#: And the torch route at the same cell, which is what the kernels replaced.
RECORDED_ONE_DEVICE_TORCH = dict(
    cold_s=316.10, warm_s=309.92,
    peak_bytes=37311663616, reserved_bytes=50537168896)
#: The md5 the 1024-class staged npz has carried through every run that read
#: it.  A staged file that hashes to this is the same input those runs
#: reconstructed, and the staging row says so.  A file that does not match is
#: not wrong -- a regenerated sinogram is never bit-identical -- so this is
#: recorded and never gated.
RECORDED_STAGE_MD5 = {"multiaxis_1024": "798c72e1cf5bb7803b9f2b02294753c6"}
#: What the memory ledger is recorded as modeling for the 1024-class on one
#: device, in bytes, beside the measured peak of the same reconstruction.  The
#: two differ by about a factor of two, which is why the 2048-class arms turn
#: the preflight off and why PART C exists.
RECORDED_LEDGER_1024_ONE_DEVICE_GB = 68.0
RECORDED_MEASURED_1024_TORCH_GB = 34.75
#: Fingerprint differences above this print a NOTE.  Nothing fails on it.
FINGERPRINT_NOTE_LEVEL = 1e-3
#: Elements promoted to float64 at a time when a fingerprint is taken.  Eight
#: million float64 is 64 MiB, which bounds the reading's own device memory at
#: any volume size.
FINGERPRINT_CHUNK_ELEMS = 1 << 23
#: Elements the host weights formula evaluates at a time in the 2048-class
#: arm's own staging.  The whole-array expression holds three sinogram-sized
#: host arrays at once, which is about 98 GB at that cell; in blocks it holds
#: two and a block.  The arithmetic is elementwise, so the values are identical
#: either way.
WEIGHTS_CHUNK_ELEMS = 1 << 26

#: Substrings that mark an arm's failure as a CAPACITY reading rather than a
#: harness fault.  A cell that does not fit the devices it was given is an
#: outcome, not a broken instrument.  Matched case-insensitively.
CAPACITY_MARKERS = ("out of memory", "outofmemory", "cuda error: out of memory",
                    "failed to allocate", "memoryerror", "cannot allocate",
                    "memorypreflighterror")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.environ.get("MG55_RESULTS",
                             os.path.join(SCRIPT_DIR, "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 26                  # wide enough for the longest arm id printed
# ──────────────────────────────────────────────────────────────────────────────


def cells():
    """The two cells, in run order: the 1024-class first, because it is the
    cheaper one and a harness defect should show up in minutes rather than
    after an hour."""
    return (SMOKE_1024, SMOKE_2048) if SMOKE else (CELL_1024, CELL_2048)


def cell_by_name(name):
    for spec in cells():
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def counts_for(spec):
    """The device counts this cell is measured at.  The optional two-device
    2048-class arm drops out when it is switched off."""
    counts = tuple(int(c) for c in spec["counts"])
    if spec["part"] == "B" and not RUN_2048_N2:
        counts = tuple(c for c in counts if c != 2)
    return counts


def arm_id(spec, count):
    return "{}_n{}".format(spec["name"], count)


def device_list(count):
    """The devices an arm of this size is given, named one by one.

    On CUDA that is cuda:0 .. cuda:<count-1>.  On the smoke's CPU there is one
    device and only one count, so the list is that device.
    """
    if DEVICE != "cuda":
        return [DEVICE]
    return ["cuda:{}".format(i) for i in range(count)]


def device_label(devices):
    """The device list in a width a table column can hold: cuda:0,1,2,3."""
    if not devices:
        return "-"
    if all(str(d).startswith("cuda:") for d in devices):
        return "cuda:" + ",".join(str(d).split(":", 1)[1] for d in devices)
    return ",".join(str(d) for d in devices)


def arm_timeout_s(spec):
    return ARM_2048_TIMEOUT_S if spec["part"] == "B" else ARM_TIMEOUT_S


def warm_repeats_for(spec):
    return WARM_REPEATS_2048 if spec["part"] == "B" else WARM_REPEATS


def stage_search_dirs():
    """Where the 1024-class staged npz may already be, in search order.

    This run's own results directory first, then the two sibling directories
    that hold it on the cluster, then the two local defaults a laptop smoke
    would have used.  The sibling lookup is what makes the reuse work without
    anybody naming a path.  Duplicates are dropped and order is kept.
    """
    parent = os.path.dirname(os.path.abspath(RESULTS_DIR))
    candidates = [
        RESULTS_DIR,
        os.path.join(parent, "mg54_multiaxis_kernel_ab"),
        os.path.join(parent, "mg52_framework_anchor"),
        os.path.join(SCRIPT_DIR, "results", "mg54_multiaxis_kernel_ab"),
        os.path.join(SCRIPT_DIR, "results", "mg52_framework_anchor"),
        os.path.join(SCRIPT_DIR, "results"),
    ]
    seen, out = set(), []
    for path in candidates:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.

    A silently ignored token would shrink the run without saying so, and a run
    that quietly measured fewer arms than it printed has cost this work a
    repeat before.  The error names the full valid list, because the caller who
    mistyped one id needs to see the others.
    """
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError("{}: {!r} is not one of this run's: {}".format(
                env_name, token, ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError("{}: no valid tokens in {!r}.  The valid ones are: {}"
                         .format(env_name, raw, ", ".join(allowed)))
    # Normalized to the DECLARED order: the run order is load-bearing (the
    # cheaper cell first, the smaller device count first), so it must not
    # depend on the order somebody typed the tokens in.
    return [name for name in allowed if name in chosen]


def mirrored_recon_shape(cell):
    """The recon shape this cell should produce, derived from the geometry's
    own rule without importing anything.

    The rule is multiaxis_parallel's auto_set_recon_geometry: the in-plane
    extent comes from the channel coverage, the slice extent from the row
    coverage divided by the smallest |cos(elevation)|, clamped at 0.1.  With the
    detector pitches and voxel aspects at their defaults of 1.0 the whole rule
    is arithmetic, which is what lets the dry plan print it.  This is a MIRROR:
    every arm records the recon shape the model really realized and whether the
    two agreed.
    """
    _views, num_rows, num_channels = cell
    max_u = num_channels / 2.0
    max_v = num_rows / 2.0
    min_cos = max(math.cos(ELEVATION_HALF_RANGE), MIN_COS_ELEVATION)
    return (int(math.floor(2 * max_u)), int(math.floor(2 * max_u)),
            int(math.floor(2 * (max_v / min_cos))))


def voxels(cell):
    shape = mirrored_recon_shape(cell)
    return shape[0] * shape[1] * shape[2]


def sinogram_bytes(cell):
    return int(cell[0]) * int(cell[1]) * int(cell[2]) * 4


# ── the staged file for the 1024-class ────────────────────────────────────────
def stage_name(spec):
    """The staged filename, reproduced exactly so the file already on disk is
    found by name and a file this run writes is found by the next run."""
    return "mg52_stage_multiaxis_{}x{}x{}.npz".format(*tuple(spec["cell"]))


def md5_path(path):
    return path + ".md5"


def file_md5(path, chunk=8 << 20):
    """md5 of a staged file, read in chunks: this npz is about 7.6 GiB and
    reading it whole to hash it would be wasteful."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def staged_present(path):
    return os.path.exists(path) and os.path.exists(md5_path(path))


def recorded_md5(path):
    with open(md5_path(path)) as handle:
        return handle.read().strip()


def find_staged(spec):
    """The first staged copy of this cell that exists with its md5 sidecar, or
    None.  Existence only -- the md5 is VERIFIED by whoever reads it."""
    name = stage_name(spec)
    for directory in stage_search_dirs():
        path = os.path.join(directory, name)
        if staged_present(path):
            return path
    return None


def stage_write_path(spec):
    """Where a fresh stage goes: this run's own results directory, under the
    same filename every run of this cell has used."""
    return os.path.join(RESULTS_DIR, stage_name(spec))


def read_staged(path, with_arrays=True):
    """The npz read itself, WITHOUT the md5 check.  Callers that have just
    hashed the file use this; everything else goes through ``load_staged``, so a
    file is never read twice to verify it once.

    ``with_arrays=False`` reads everything EXCEPT the sinogram and the weights.
    An npz member is only read when it is asked for, and those two are about
    7.6 GiB together.
    """
    import numpy as np

    with np.load(path, allow_pickle=False) as handle:
        meta = dict(
            view_params=handle["view_params"],
            sinogram_shape=[int(v) for v in handle["sinogram_shape"]],
            recon_shape=[int(v) for v in handle["recon_shape"]],
            geometry=str(handle["geometry"].item()),
            name=str(handle["name"].item()),
            delta_voxel=float(handle["delta_voxel"]),
            voxel_row_aspect=float(handle["voxel_row_aspect"]),
            voxel_slice_aspect=float(handle["voxel_slice_aspect"]),
            psf_radius=int(handle["psf_radius"]),
            phantom_fallback=str(handle["phantom_fallback"].item()))
        if with_arrays:
            meta["sinogram"] = handle["sinogram"]
            meta["weights"] = handle["weights"]
    return meta


def load_staged(path, with_arrays=True):
    """Read the staged npz into a plain dict, after verifying its md5.

    Raises on a mismatch.  An arm that reconstructed different bytes than its
    siblings did not measure what the plan said, and a truncated read on a
    shared parallel filesystem is a recorded failure mode of this work.
    """
    expected = recorded_md5(path)
    actual = file_md5(path)
    if actual != expected:
        raise ValueError("the staged file at {} hashes to {}, not the recorded "
                         "{}".format(path, actual, expected))
    meta = read_staged(path, with_arrays=with_arrays)
    meta["md5"] = actual
    return meta


# ── the model and its inputs ──────────────────────────────────────────────────
def view_params_for(cell):
    """One cell's per-view parameters: two angles per view, azimuths over half
    a turn and elevations across +/- 0.5 radians.  The elevation range is part
    of what a multiaxis cell measures -- the geometry divides the detector
    height by the smallest |cos(elevation)| -- so a wider sweep would inflate
    the slice count."""
    import numpy as np

    num_views = int(cell[0])
    azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.linspace(-ELEVATION_HALF_RANGE, ELEVATION_HALF_RANGE,
                            num_views)
    return np.stack([azimuth, elevation], axis=1)


def build_model(sinogram_shape, view_params, devices, skip_preflight=False):
    """The multiaxis model an arm reconstructs with, on the devices it names.

    The device list is EXPLICIT here, which turns the automatic choice off.
    That is the point: this run is measuring what a layout costs, so the layout
    has to be the one the plan asked for rather than the one a policy preferred.

    ``skip_preflight`` is set on the 2048-class arms only; the arm's row
    carries the sentence that says why.  It is set BEFORE the devices are
    configured, because the flag is read wherever a layout is settled.  It is
    belt and braces rather than the mechanism: a model given an explicit device
    list takes the caller's-layout branch, where the capacity check does not
    run at all.  Setting it means no path can refuse these arms, and the reason
    is on the row either way.
    """
    import mbirtorch

    model = mbirtorch.MultiAxisParallelModel(
        tuple(int(v) for v in sinogram_shape), view_params)
    if skip_preflight:
        model.skip_memory_preflight = True
    model.configure_devices(devices=list(devices))
    model.set_params(no_warning=True, verbose=0)
    return model


def build_phantom(recon_shape):
    """The phantom, as a host float32 array, with the staging's own fallback.

    The shepp-logan builder places its ellipsoids as fractions of the volume,
    and on a volume only a few voxels deep every one of them can miss, leaving
    the phantom all zeros.  An all-zero phantom forward projects to an all-zero
    sinogram, so an arm would time a reconstruction of nothing.  The fallback is
    a seeded uniform volume, and the row records that it was used.
    """
    import numpy as np

    import mbirtorch

    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    fallback = ""
    if float(np.max(phantom)) == 0.0:
        phantom = np.asarray(np.random.RandomState(VCD_SEED).rand(*recon_shape),
                             dtype=np.float32)
        fallback = "seeded uniform (shepp-logan returned all zeros)"
    return np.asarray(phantom, dtype=np.float32), fallback


def to_numpy(x):
    """The one host exit.  A gathered container ALREADY returns numpy, so a
    gather is never followed by ``.detach()`` -- re-detaching one is a recorded
    way to lose rows."""
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    if callable(getattr(x, "gather", None)) and hasattr(x, "placement"):
        return x.gather()
    return (x.detach().cpu().numpy()
            if callable(getattr(x, "detach", None)) else np.asarray(x))


def weights_whole(sinogram):
    """The staging's weights formula, evaluated whole.

    Used by the 1024-class staging path only, where the arrays are small enough
    for it and where the point is to reproduce the bytes the file on disk
    already holds.
    """
    import numpy as np

    return np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32,
                                                             copy=False)


def weights_blocked(sinogram):
    """The same weights formula, evaluated a block at a time.

    Whole-array, this expression holds the sinogram, the divided temporary and
    the exponentiated result at once -- about 98 GB at the 2048-class cell.  In
    blocks it holds the sinogram, the result and one block.  Every operation
    here is elementwise, so the values are the same ones the whole-array
    expression produces.
    """
    import numpy as np

    denom = 2 * np.max(sinogram)
    flat_in = sinogram.reshape(-1)
    weights = np.empty(sinogram.shape, dtype=np.float32)
    flat_out = weights.reshape(-1)
    for start in range(0, flat_in.size, WEIGHTS_CHUNK_ELEMS):
        stop = min(start + WEIGHTS_CHUNK_ELEMS, flat_in.size)
        flat_out[start:stop] = np.exp(-flat_in[start:stop] / denom)
    return weights


# ── the value fingerprint ─────────────────────────────────────────────────────
def _flat_tensor(volume, torch_module):
    """One flat torch view of a reconstruction, whatever ``recon`` handed back.

    ``recon`` returns a host numpy array in this library, and a gathered
    container returns numpy too, so this is normally a zero-copy wrap of host
    memory.  A device tensor is accepted and left where it is.
    """
    import numpy as np

    if isinstance(volume, torch_module.Tensor):
        return volume.detach().reshape(-1)
    if callable(getattr(volume, "gather", None)) and hasattr(volume, "placement"):
        volume = volume.gather()
    return torch_module.as_tensor(
        np.ascontiguousarray(np.asarray(volume))).reshape(-1)


def fingerprint(volume, torch_module, device):
    """Two float64 reductions of a reconstruction -- the sum of absolute values
    and the sum of squares -- accumulated ON THE DEVICE in float64 and read
    back.

    Two numbers rather than one, because a sum of absolute values alone cannot
    see a rearrangement that preserves magnitudes.  Accumulated in fixed-size
    chunks with plain ``torch.sum``: a float32 sum over a nine-billion-element
    volume loses the digits this comparison needs, promoting a whole volume at
    once would double what the reading costs, and this library's own notes
    record that the one-line norm reductions are both inaccurate at scale and
    slow on some backends.  Each chunk's partial is read back to the host
    immediately, so nothing but one chunk is ever resident.
    """
    flat = _flat_tensor(volume, torch_module)
    abs_sum = 0.0
    sq_sum = 0.0
    total = int(flat.numel())
    for start in range(0, total, FINGERPRINT_CHUNK_ELEMS):
        block = flat[start:start + FINGERPRINT_CHUNK_ELEMS].to(
            device=device, dtype=torch_module.float64)
        abs_sum += float(torch_module.sum(torch_module.abs(block)))
        sq_sum += float(torch_module.sum(block * block))
        block = None
    return abs_sum, sq_sum, total


def relative_gap(value, reference):
    """|value - reference| / |reference|, with a zero reference reported as an
    absolute gap rather than as infinity."""
    if value is None or reference is None:
        return None
    scale = abs(reference)
    return abs(value - reference) / (scale if scale > 0.0 else 1.0)


def _is_oom(exc):
    """Whether an exception is a device out-of-memory.

    The class name is checked as well as the message because torch raises its
    own OutOfMemoryError on some paths and a plain RuntimeError on others, and
    a run that mistook one for a real failure would report a broken instrument
    when the reading was only that the problem did not fit.
    """
    if type(exc).__name__ in ("OutOfMemoryError", "CUDAOutOfMemoryError"):
        return True
    return "out of memory" in str(exc).lower()


def git_identity(path):
    """The commit a checkout sits at, and whether it is dirty.  ``None`` when
    the directory is not a git checkout or git is unavailable, which is a
    recorded state rather than an error: an rsynced export has no commit."""
    out = dict(commit=None, dirty=None, root=path)
    try:
        proc = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            out["commit"] = proc.stdout.strip()
    except Exception:                                             # noqa: BLE001
        return out
    try:
        proc = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                              capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            out["dirty"] = bool(proc.stdout.strip())
    except Exception:                                             # noqa: BLE001
        pass
    return out


# ── the tree under test ───────────────────────────────────────────────────────
def tree_witnesses():
    """What tree produced these numbers, measured rather than asserted.

    The first three say this is the padded, recompile-remedied tree the
    recorded one-device walls were measured on, and the third matters directly
    because a tree without it would hand a torch body eager python rather than
    a compiled one.

    The last two are what makes the kernel route a real route rather than a
    name: the kernel module must import -- it is written to import without
    triton, so this works on a laptop too -- and the geometry's selection hook
    must consult the two availability functions and reach the two Triton
    wrappers.  Both are read by SOURCE INSPECTION and by attribute lookup: no
    model is built, no device is touched and no CUDA is initialized, so the
    witness block costs nothing and can run anywhere.

    A lookup that fails is recorded as failed; nothing here raises.
    """
    record = dict(available=True)
    try:
        import inspect

        from mbirtorch import projectors
        from mbirtorch._utils import padded_kernel_width

        record["padded_kernel_width_504"] = int(padded_kernel_width(504))
        record["recompile_limit_floor"] = int(
            getattr(projectors, "_RECOMPILE_LIMIT_FLOOR", -1))
        source = inspect.getsource(projectors.maybe_compile)
        record["raise_on_compiling_thread"] = bool(
            "_raise_recompile_budget()"
            in source.split("_GLOBAL_COMPILE_LOCK:")[-1])

        from mbirtorch import triton_multiaxis
        from mbirtorch.multiaxis_parallel import MultiAxisParallelModel

        record["triton_multiaxis_file"] = triton_multiaxis.__file__
        # The two kernel wrappers exist, are named the way the premise check
        # recognizes them, and carry the per-view cost attribute the driver
        # reads to choose a view batch and the ledger reads to price it.
        kernels = {}
        for direction, attr in (("forward",
                                 "_multiaxis_forward_view_batch_triton"),
                                ("back",
                                 "_multiaxis_back_view_batch_triton")):
            body = getattr(triton_multiaxis, attr, None)
            kernels[direction] = dict(
                present=body is not None,
                name=getattr(body, "__name__", None),
                named_like_a_kernel=bool(
                    getattr(body, "__name__", "").endswith(
                        KERNEL_BODY_SUFFIX)),
                has_view_batch_cost=getattr(body, "_view_batch_cost",
                                            None) is not None)
        record["multiaxis_kernels"] = kernels
        selection = inspect.getsource(
            MultiAxisParallelModel._view_batch_bodies)
        record["selection_consults_availability"] = bool(
            "multiaxis_forward_kernel_usable" in selection
            and "multiaxis_back_kernel_usable" in selection)
        record["selection_reaches_kernels"] = bool(
            "_multiaxis_forward_view_batch_triton" in selection
            and "_multiaxis_back_view_batch_triton" in selection)

        record["ok"] = bool(
            record["padded_kernel_width_504"] == 512
            and record["recompile_limit_floor"] >= 64
            and record["raise_on_compiling_thread"]
            and all(entry["present"] and entry["named_like_a_kernel"]
                    and entry["has_view_batch_cost"]
                    for entry in kernels.values())
            and record["selection_consults_availability"]
            and record["selection_reaches_kernels"])
    except Exception as exc:                                      # noqa: BLE001
        record.update(available=False, ok=False,
                      reason="{}: {}".format(type(exc).__name__, exc))
    return record


# ── the premise: which bodies did this model really bind ──────────────────────
def route_expectation():
    """What every arm must have bound before it may time anything, and the
    reason when that is not what this run is about.

    On a CPU install there is no triton at all, so the model binds the torch
    bodies and the premise DEGRADES.  The smoke says that on the row rather
    than pretending otherwise; what the smoke exercises is the flow, not the
    kernels.
    """
    if DEVICE != "cuda":
        return "torch", ("this run is on {}, where there is no triton at all, "
                         "so the model binds the torch bodies and the kernel "
                         "premise is degraded".format(DEVICE))
    return "kernel", ""


def bound_body_report(model):
    """Which bodies this model bound, what the availability checks answered,
    and why.

    ``_view_batch_bodies`` is the geometry's selection hook and returns the two
    plain functions, before compilation and before any per-device binding, so
    their names are readable whatever torch did with them afterwards.  The two
    availability functions are asked separately, because the geometry asks them
    separately and a machine may bind one kernel and keep the other direction's
    torch body; their reasons are the record of WHY a node is not using a
    kernel, which is the first question anybody reading these walls will ask.
    """
    from mbirtorch import _memory_ledger
    from mbirtorch.kernel_availability import (multiaxis_back_kernel_usable,
                                               multiaxis_forward_kernel_usable)

    fwd_body, back_body = model._view_batch_bodies()
    fwd_ok, fwd_reason = multiaxis_forward_kernel_usable(model)
    back_ok, back_reason = multiaxis_back_kernel_usable(model)
    names = dict(forward=fwd_body.__name__, back=back_body.__name__)
    report = dict(
        forward_body=names["forward"], back_body=names["back"],
        forward_kernel_available=bool(fwd_ok),
        back_kernel_available=bool(back_ok),
        forward_availability_reason=str(fwd_reason),
        back_availability_reason=str(back_reason),
        # A kernel body carries a per-view cost attribute and general torch
        # code carries none, so this is the same question asked a second way,
        # through the library's own reader rather than through a name.  It is
        # also what the memory ledger reads to decide how to price the views.
        torch_body_directions=list(
            _memory_ledger.torch_body_directions(model)),
        env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
        compile_enabled=bool(model.compile_enabled),
        compile_mode=str(model.compile_mode))
    report["kernels_bound"] = all(
        name.endswith(KERNEL_BODY_SUFFIX) for name in names.values())
    report["torch_bodies_bound"] = (
        report["torch_body_directions"] == ["forward", "back"])
    return report


def check_premise(report):
    """Whether an arm bound what this run is about, and the sentence that goes
    on the row when it did not.

    An arm whose premise fails times NOTHING.  Every arm here is on the default
    kernel selection, so an arm that quietly bound the torch bodies would be
    measuring the route this work already measured, under a name that says
    otherwise -- and a wrong number that looks right is worse than no number.
    """
    expected, degraded_reason = route_expectation()
    if expected == "kernel":
        ok = report["kernels_bound"]
        reason = ("" if ok else
                  "this arm bound {} and {}, and a kernel body's name ends in "
                  "{}.  The availability checks said: forward {}, back {}"
                  .format(report["forward_body"], report["back_body"],
                          KERNEL_BODY_SUFFIX,
                          report["forward_availability_reason"],
                          report["back_availability_reason"]))
    else:
        ok = report["torch_bodies_bound"] and not report["kernels_bound"]
        reason = ("" if ok else
                  "this arm was to bind the torch bodies in both directions "
                  "and bound {} and {}".format(report["forward_body"],
                                               report["back_body"]))
    return ok, reason, expected, degraded_reason


# ── the shared measurement ────────────────────────────────────────────────────
def peak_readings(torch_module, devices, cuda):
    """The two peak counters on EVERY device in the list, as two lists.

    Allocated is what the library asked for and reserved is what the caching
    allocator kept from the driver; a layout that fragments differently moves
    the second without moving the first, so neither stands in for the other.
    Per device rather than per process because whether sharding divides the
    peak is the question this run exists to answer, and a single maximum cannot
    show a layout that loaded one device and left the others idle.
    """
    if not cuda:
        return None, None
    allocated = [int(torch_module.cuda.max_memory_allocated(d)) for d in devices]
    reserved = [int(torch_module.cuda.max_memory_reserved(d)) for d in devices]
    return allocated, reserved


def reset_peaks(torch_module, devices, cuda):
    if not cuda:
        return
    for device in devices:
        torch_module.cuda.reset_peak_memory_stats(device)


def convergence_record(recon_dict):
    """The forward-model error metric the reconstruction reports, and whether
    it fell.

    The library returns ``(recon, recon_dict)``, and ``recon_dict``'s
    'recon_params' entry carries the per-iteration traces.  'fm_rmse' is the
    forward-model error, one value per iteration.  Only that trace and the
    stopping percentages are kept here: the same dict also carries a
    per-iteration, per-slice norm, which at these volumes is a large array
    nobody reads off a row.

    RECORDED, NEVER GATED.  A reconstruction whose error did not fall is a
    reading about the problem or the layout, and this run is not the instrument
    that should judge it.
    """
    out = dict(fm_rmse=None, fm_rmse_decreased=None, num_iterations=None,
               stop_pct=None, convergence_error=None)
    try:
        params = (recon_dict or {}).get("recon_params") or {}
        rmse = [float(v) for v in (params.get("fm_rmse") or [])]
        out["fm_rmse"] = rmse
        out["num_iterations"] = params.get("num_iterations")
        out["stop_pct"] = [float(v) for v in (params.get(
            "stop_threshold_change_pct") or [])]
        if len(rmse) >= 2:
            out["fm_rmse_decreased"] = bool(rmse[-1] < rmse[0])
            out["fm_rmse_monotone"] = bool(
                all(b <= a for a, b in zip(rmse, rmse[1:])))
    except Exception as exc:                                      # noqa: BLE001
        out["convergence_error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def measure(model, sinogram, weights, devices, warm_repeats, cuda, row):
    """One arm's reconstructions: a cold pass, then ``warm_repeats`` warm ones.

    The peak counters are reset before the cold pass and again before the warm
    passes, so the cold peaks and the warm peaks are two separate readings, both
    per device.  The fingerprint is taken from the LAST reconstruction AFTER the
    peaks are read, so the reading's own device memory cannot land in the number
    it is reported beside.

    The previous output is dropped before each new call.  ``recon`` returns a
    host volume, and at the 2048-class cell that is about 34 GB; holding the old
    one while the next is built would double it for no reason.
    """
    import numpy as np
    import torch

    out = None
    info = None

    def one_recon():
        nonlocal out, info
        out = None                     # release the previous volume first
        info = None
        np.random.seed(VCD_SEED)
        volume, recon_dict = model.recon(sinogram, weights=weights,
                                         max_iterations=VCD_ITERATIONS,
                                         stop_threshold_change_pct=0.0)
        if cuda:
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        out, info = volume, recon_dict

    reset_peaks(torch, devices, cuda)
    start = time.perf_counter()
    one_recon()
    row["cold_s"] = time.perf_counter() - start

    # The layout has settled, so this describes the layout the timed passes
    # actually ran on.
    realized = [str(d) for d in model.recon_placement.devices]
    row["realized_devices"] = realized
    row["realized_n_devices"] = len(realized)
    row["realized_sino_devices"] = [str(d) for d
                                    in model.sino_placement.devices]
    row["devices_as_asked"] = (realized == [str(d) for d in devices])

    row["cold_peak_bytes_per_device"], row["cold_reserved_bytes_per_device"] = \
        peak_readings(torch, devices, cuda)
    reset_peaks(torch, devices, cuda)

    warm = []
    for _ in range(warm_repeats):
        start = time.perf_counter()
        one_recon()
        warm.append(time.perf_counter() - start)
    row["warm_all"] = warm
    row["warm_s"] = statistics.median(warm)
    row["warm_min"] = min(warm)
    row["warm_max"] = max(warm)
    row["warm_spread"] = (max(warm) - min(warm)) / statistics.median(warm)

    row["warm_peak_bytes_per_device"], row["warm_reserved_bytes_per_device"] = \
        peak_readings(torch, devices, cuda)

    if cuda:
        # The comparison uses the PROCESS-LIFETIME peak per device, which is
        # the larger of the two readings: a layout whose cold pass costs more
        # than its warm passes still needed that much device memory to run.
        row["peak_bytes_per_device"] = [
            max(a, b) for a, b in zip(row["cold_peak_bytes_per_device"],
                                      row["warm_peak_bytes_per_device"])]
        row["reserved_bytes_per_device"] = [
            max(a, b) for a, b in zip(row["cold_reserved_bytes_per_device"],
                                      row["warm_reserved_bytes_per_device"])]
        row["busiest_peak_bytes"] = max(row["peak_bytes_per_device"])
        row["busiest_reserved_bytes"] = max(row["reserved_bytes_per_device"])
        row["busiest_device"] = str(devices[
            row["peak_bytes_per_device"].index(row["busiest_peak_bytes"])])
    else:
        for key in ("peak_bytes_per_device", "reserved_bytes_per_device",
                    "busiest_peak_bytes", "busiest_reserved_bytes",
                    "busiest_device"):
            row[key] = None
    row["peak_kind"] = ("torch.cuda.max_memory_allocated and "
                        "max_memory_reserved on every device in the list, "
                        "reset before the cold pass and again before the warm "
                        "passes")

    row.update(convergence_record(info))

    fingerprint_device = (model.recon_placement.devices[0]
                          if model.recon_placement.devices else DEVICE)
    abs_sum, sq_sum, elements = fingerprint(out, torch, fingerprint_device)
    row["fingerprint_abs_sum"] = abs_sum
    row["fingerprint_sq_sum"] = sq_sum
    row["fingerprint_elements"] = elements
    row["fingerprint_where"] = "float64 chunked sums on {}".format(
        fingerprint_device)
    return row


# ── the workers: the identity probe, the staging, or one arm ──────────────────
def run_identity(cfg):
    """Which library this run is about to measure, and on what.

    Run before anything else and in its own process, so the header row can name
    the torch version, the devices, the tree under test and its witnesses
    without the driver importing torch at all until the ledger pricing.
    """
    import torch

    import mbirtorch

    row = dict(cfg)
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda else 1
    row.update(torch_version=torch.__version__,
               library_file=mbirtorch.__file__,
               device_count=count,
               device_names=[torch.cuda.get_device_name(i)
                             for i in range(count)] if cuda else [DEVICE],
               cuda=cuda,
               triton_version=None,
               python=platform.python_version())
    try:
        import triton
        row["triton_version"] = triton.__version__
    except Exception as exc:                                      # noqa: BLE001
        row["triton_version"] = "{}: {}".format(type(exc).__name__, exc)
    package_root = os.path.dirname(os.path.dirname(
        os.path.abspath(mbirtorch.__file__)))
    row["git"] = git_identity(package_root)
    row["tree_witnesses"] = tree_witnesses()
    return row


def run_stage(cfg):
    """Build the 1024-class sinogram and weights once and write them to one npz
    with an md5 sidecar -- or verify and reuse the copy already on disk.

    Every arm of that cell then loads the same file, so the three device counts
    reconstruct identical input and the comparison between them is controlled
    rather than incidental.  A file already on disk is VERIFIED rather than
    rebuilt: a regenerated sinogram is not bit-identical, so a rebuild would
    silently change what the arms reconstruct and would break the tie to the
    one-device numbers measured on the same bytes.
    """
    import numpy as np

    spec = cfg["spec"]
    row = dict(cfg)
    cell = tuple(int(v) for v in spec["cell"])
    row["mirrored_recon_shape"] = list(mirrored_recon_shape(cell))
    row["recorded_md5_elsewhere"] = RECORDED_STAGE_MD5.get(spec["name"])

    found = find_staged(spec)
    if found is not None:
        expected = recorded_md5(found)
        actual = file_md5(found)
        row.update(stage_path=found, reused=True, md5=actual,
                   md5_ok=(actual == expected), recorded_md5=expected,
                   bytes_on_disk=os.path.getsize(found),
                   reused_from=os.path.dirname(os.path.abspath(found)),
                   same_bytes_as_recorded=(
                       actual == row["recorded_md5_elsewhere"]))
        if actual == expected:
            # Hashed once, just above; read the metadata only -- the sinogram
            # and the weights are read by the arms, not here.
            meta = read_staged(found, with_arrays=False)
            row.update(recon_shape=meta["recon_shape"],
                       sinogram_shape=meta["sinogram_shape"],
                       geometry=meta["geometry"],
                       delta_voxel=meta["delta_voxel"],
                       psf_radius=meta["psf_radius"],
                       phantom_fallback=meta["phantom_fallback"])
            row["recon_shape_mirror_agrees"] = (
                meta["recon_shape"] == row["mirrored_recon_shape"])
            row["shape_ok"] = (meta["sinogram_shape"] == list(cell)
                               and meta["geometry"] == "multiaxis")
            if not row["shape_ok"]:
                row["invalid_reasons"] = [
                    "the staged file at {} holds a {} sinogram of shape {}, "
                    "not a multiaxis {}".format(found, meta["geometry"],
                                                meta["sinogram_shape"],
                                                list(cell))]
        else:
            row["invalid_reasons"] = [
                "the staged file at {} hashes to {}, not the recorded {}"
                .format(found, actual, expected)]
        return row

    # Nothing verified anywhere, so build it here, under this run's own results
    # directory and under the same filename.  ONE device: the file this
    # replaces holds a one-device torch-body forward projection, and a re-stage
    # should be the same input rather than a new one.
    path = stage_write_path(spec)
    view_params = np.asarray(view_params_for(cell), dtype=np.float32)
    model = build_model(cell, view_params, device_list(1))
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    phantom, phantom_fallback = build_phantom(recon_shape)

    sinogram = np.ascontiguousarray(
        np.asarray(to_numpy(model.forward_project(phantom)), dtype=np.float32))
    phantom = None
    weights = weights_whole(sinogram)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(path, sinogram=sinogram, weights=weights,
             view_params=view_params,
             sinogram_shape=np.asarray(cell, dtype=np.int64),
             recon_shape=np.asarray(recon_shape, dtype=np.int64),
             distances=np.asarray((), dtype=np.float64),
             geometry=np.asarray("multiaxis"), name=np.asarray(spec["name"]),
             delta_voxel=np.asarray(float(model.get_params("delta_voxel"))),
             voxel_row_aspect=np.asarray(
                 float(model.get_params("voxel_row_aspect"))),
             voxel_slice_aspect=np.asarray(
                 float(model.get_params("voxel_slice_aspect"))),
             psf_radius=np.asarray(int(model.get_psf_radius())),
             phantom_fallback=np.asarray(phantom_fallback))
    digest = file_md5(path)
    with open(md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    row.update(stage_path=path, reused=False, md5=digest, md5_ok=True,
               shape_ok=True, recon_shape=list(recon_shape),
               sinogram_shape=list(cell), geometry="multiaxis",
               recon_shape_mirror_agrees=(
                   list(recon_shape) == row["mirrored_recon_shape"]),
               delta_voxel=float(model.get_params("delta_voxel")),
               psf_radius=int(model.get_psf_radius()),
               phantom_fallback=phantom_fallback,
               bytes_on_disk=os.path.getsize(path),
               same_bytes_as_recorded=(
                   digest == row["recorded_md5_elsewhere"]),
               # Which bodies made these bytes.  A fresh stage runs with the
               # kernels off, so this should name the torch bodies; the row
               # carries the answer rather than the intention.
               stage_bodies=bound_body_report(model))
    return row


def _arm_row_skeleton(cfg, torch_module, mbirtorch_module, cuda, devices):
    return dict(cfg, invalid_reasons=[],
                vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
                warm_repeats=cfg["warm_repeats"],
                requested_devices=[str(d) for d in devices],
                requested_n_devices=len(devices),
                torch_version=torch_module.__version__,
                library_file=mbirtorch_module.__file__,
                device=DEVICE, cuda=cuda,
                visible_devices=(torch_module.cuda.device_count()
                                 if cuda else 1),
                env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))


def run_arm_staged(cfg):
    """PART A: one device count reconstructing the 1024-class cell from the
    staged npz."""
    import torch

    import mbirtorch

    devices = list(cfg["devices"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    path = cfg["stage_path"]
    meta = load_staged(path)              # raises on an md5 mismatch
    row = _arm_row_skeleton(cfg, torch, mbirtorch, cuda, devices)
    row.update(stage_path=path, staged_md5=meta["md5"],
               staged_recon_shape=meta["recon_shape"],
               phantom_fallback=meta["phantom_fallback"],
               staging_kind="staged npz, md5-verified on load")
    if cfg.get("recorded_one_device"):
        row["recorded_one_device_kernel"] = RECORDED_ONE_DEVICE_KERNEL
        row["recorded_one_device_torch"] = RECORDED_ONE_DEVICE_TORCH

    model = build_model(meta["sinogram_shape"], meta["view_params"], devices)
    realized = [int(s) for s in model.get_params("recon_shape")]
    row["recon_shape"] = realized
    row["recon_shape_ok"] = (realized == list(meta["recon_shape"]))
    if not row["recon_shape_ok"]:
        row["invalid_reasons"].append(
            "this arm's model realized recon shape {}, but the staged cell was "
            "built at {}".format(realized, meta["recon_shape"]))
        return row

    report = bound_body_report(model)
    row.update(report)
    ok, reason, expected, degraded = check_premise(report)
    row.update(premise_ok=ok, expected_bodies=expected,
               route_degraded=bool(degraded), route_degraded_reason=degraded)
    if not ok:
        # Nothing is timed.  A row carrying a wall taken on the other route's
        # bodies would be worse than this row, which carries none.
        row["invalid_reasons"].append(reason)
        return row

    row.update(view_batch_report(model, meta["sinogram_shape"], realized))
    measure(model, meta["sinogram"], meta["weights"], devices,
            cfg["warm_repeats"], cuda, row)
    return row


def run_arm_inprocess(cfg):
    """PART B: one device count reconstructing the 2048-class cell, with its
    input built here.

    No npz: the phantom is built exactly as the staging builds one, forward
    projected through this same model, and the weights come from the same
    formula.  Nothing else needs these bytes, and at this cell they are about
    33 GB of sinogram and another 33 of weights.

    ONE CONSEQUENCE IS WORTH STATING.  The sinogram here is a KERNEL forward
    projection, because the arm is on the kernel route, where the 1024-class
    staged file holds a torch-body one.  Nothing compares the two cells' inputs
    with each other, so this changes no comparison; the row records which
    bodies made the bytes.

    AND ONE READING CAVEAT.  The staging's own peaks are read before the
    counters are reset for the cold pass, so the reconstruction's readings
    describe the reconstruction.  The allocator's RESERVED pool still carries
    over from the staging, which is what reserved means; both numbers are on
    the row and neither stands in for the other.
    """
    import numpy as np
    import torch

    import mbirtorch

    devices = list(cfg["devices"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    cell = tuple(int(v) for v in cfg["spec"]["cell"])
    row = _arm_row_skeleton(cfg, torch, mbirtorch, cuda, devices)
    row.update(staging_kind="built in this process: phantom, forward "
                            "projection, weights",
               skip_memory_preflight=True,
               skip_memory_preflight_reason=cfg["preflight_reason"],
               sinogram_bytes=sinogram_bytes(cell),
               mirrored_recon_shape=list(mirrored_recon_shape(cell)))

    view_params = np.asarray(view_params_for(cell), dtype=np.float32)
    model = build_model(cell, view_params, devices, skip_preflight=True)
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    row["recon_shape"] = list(recon_shape)
    row["recon_shape_mirror_agrees"] = (
        list(recon_shape) == row["mirrored_recon_shape"])
    row["recon_bytes"] = int(recon_shape[0] * recon_shape[1]
                             * recon_shape[2] * 4)

    report = bound_body_report(model)
    row.update(report)
    ok, reason, expected, degraded = check_premise(report)
    row.update(premise_ok=ok, expected_bodies=expected,
               route_degraded=bool(degraded), route_degraded_reason=degraded)
    if not ok:
        row["invalid_reasons"].append(reason)
        return row

    row.update(view_batch_report(model, cell, recon_shape))

    # ── the staging, timed and reported separately from the walls ────────────
    reset_peaks(torch, devices, cuda)
    start = time.perf_counter()
    phantom, fallback = build_phantom(recon_shape)
    row["phantom_s"] = time.perf_counter() - start
    row["phantom_fallback"] = fallback
    sinogram = np.ascontiguousarray(
        np.asarray(to_numpy(model.forward_project(phantom)), dtype=np.float32))
    phantom = None                       # 34 GB; the weights need the room
    row["forward_project_s"] = time.perf_counter() - start - row["phantom_s"]
    weights = weights_blocked(sinogram)
    row["staging_s"] = time.perf_counter() - start
    row["staging_peak_bytes_per_device"], \
        row["staging_reserved_bytes_per_device"] = peak_readings(
            torch, devices, cuda)
    row["staging_bodies"] = dict(forward_body=report["forward_body"],
                                 back_body=report["back_body"])
    row["sinogram_shape"] = [int(s) for s in sinogram.shape]
    row["sinogram_max"] = float(np.max(sinogram))

    measure(model, sinogram, weights, devices, cfg["warm_repeats"], cuda, row)
    return row


def view_batch_report(model, sinogram_shape, recon_shape):
    """The view batch the driver will choose, per direction, by its own rule
    and off the bodies this arm actually bound.

    Recorded rather than derived: the batch is chosen by the cost model of the
    body that will run, and a kernel body declares its own per-view cost where
    a torch body is charged the transient the driver prices itself.  The number
    is context for the walls beside it, and nothing is gated on it.
    """
    out = {}
    try:
        args = model._view_batch_args()
        indices = model.full_indices_device()
        num_pixels = int(indices.shape[0])
        pf = model.projector_functions
        out["num_pixels"] = num_pixels
        out["view_batch"] = dict(
            forward=int(pf._effective_view_batch(
                pf._fwd_body_per_dev[0], num_pixels, int(recon_shape[2]),
                args)),
            back=int(pf._effective_view_batch(
                pf._back_body_per_dev[0], num_pixels,
                int(sinogram_shape[1]), args)))
        out["view_batch_iterations"] = {
            name: int(math.ceil(int(sinogram_shape[0]) / max(1, batch)))
            for name, batch in out["view_batch"].items()}
        indices = None
    except Exception as exc:                                      # noqa: BLE001
        out["view_batch"] = None
        out["view_batch_error"] = "{}: {}".format(type(exc).__name__, exc)
    return out


def run_job(cfg):
    started = time.time()
    if cfg["kind"] == "identity":
        row = run_identity(cfg)
    elif cfg["kind"] == "stage":
        row = run_stage(cfg)
    elif cfg["part"] == "A":
        row = run_arm_staged(cfg)
    else:
        row = run_arm_inprocess(cfg)
    row["worker_wall_s"] = time.time() - started
    return row


# ── the driver ────────────────────────────────────────────────────────────────
def job_env(cfg):
    """The environment that DEFINES one job, set explicitly so nothing leaks in
    from the submitting shell.

    MBIRTORCH_DISABLE_TRITON is set per job and never inherited: 0 for every
    arm, because every arm here is on the default kernel selection, and 1 for a
    fresh 1024-class staging, so its sinogram is the torch-body forward
    projection the file already on disk holds.

    MBIRTORCH_NUM_DEVICES IS REMOVED rather than set.  Every arm names its
    devices one by one, so a process-wide count pin is not the mechanism here
    and setting one would only cap something else -- the phantom build reads it
    too -- at a count this run did not ask for.

    PYTHONPATH IS INHERITED.  On the cluster the library under test is a
    candidate tree reached only through PYTHONPATH; popping it would silently
    run the installed tree instead.  Every job records the mbirtorch file it
    actually imported, so which tree ran is a fact on the row.
    """
    env = dict(os.environ)
    env.pop("MG55_DRY", None)           # a worker never prints a plan
    env.pop("MG55_ARMS", None)          # a worker runs its cfg, not a plan
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)  # it owns the peak counters
    env.pop("MBIRTORCH_WIDENING_GUARD", None)      # explicit layouts bypass it
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "1" if cfg["kind"] == "stage" else "0"
    return env


def spawn(cfg, timeout_s):
    """Run one job in a NEW interpreter, with a hard time cap.

    A new process per job is not tidiness.  The kernel availability probe and
    the per-device value self-checks are cached for the life of a process, and
    compiled bodies and allocator state are cached the same way -- and the
    allocator's high-water marks are what this run reads, so one process per
    layout is the only way each layout's peaks describe that layout alone.  The
    row travels through a file rather than through stdout, so the worker's own
    output streams into the job log while it runs.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR,
                            "_mg55_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR,
                            "_mg55_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    env = job_env(cfg)
    env["MG55_CHILD"] = cfg_path
    env["MG55_CHILD_OUT"] = out_path
    start = time.perf_counter()
    timed_out = False
    returncode = None
    try:
        proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                              env=env, timeout=timeout_s)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        # The cap is a READING, not a fault: an arm that cannot finish inside it
        # has told us something about this layout at this size.
        timed_out = True
    wall = time.perf_counter() - start
    if timed_out:
        row = dict(cfg, error="timed out after {:.1f} min".format(
            timeout_s / 60.0), timed_out=True)
    elif not os.path.exists(out_path):
        # A job killed by the operating system's memory manager lands here,
        # having written nothing.  That is a reading too, and the classifier
        # below decides.
        row = dict(cfg, error="worker exited {} and wrote no row"
                   .format(returncode), worker_returncode=returncode)
    else:
        with open(out_path) as handle:
            row = json.load(handle)
        row["worker_returncode"] = returncode
    row["subprocess_wall_s"] = wall
    return row


def is_capacity_reading(row):
    """Whether an arm's failure is a capacity or timeout READING rather than a
    harness fault.  A cell that does not fit the devices it was given is an
    outcome, so it must not be reported as a broken instrument."""
    if row.get("timed_out"):
        return True
    text = str(row.get("error", "")).lower()
    if not text:
        return False
    if any(marker in text for marker in CAPACITY_MARKERS):
        return True
    # A worker killed outright by the operating system's memory manager writes
    # no traceback at all.  Signal 9 with no row is the shape that leaves.
    return "wrote no row" in text and str(row.get("worker_returncode")) == "-9"


# ── PART C: what the memory ledger models ─────────────────────────────────────
def price_cell(spec, counts):
    """The modeled planned per-device peak for a reconstruction of one cell, at
    each candidate device count.

    Read-only, through the library's own public entry points: a plan built from
    the model over a candidate device list, and the estimate made from that
    plan.  Both are pure arithmetic -- no device is queried and nothing is
    allocated -- which is exactly what lets a count the model is not configured
    for be priced.  The model itself is built once and configured on the first
    device only; building it is the one piece of real work here, and the
    kernels' availability self-check runs on that device the first time the
    selection hook is asked.

    Weights are declared as supplied without being built.  The plan reads only
    WHETHER weights were passed, so a one-element array says "supplied", and
    every arm in this run does supply them.
    """
    import numpy as np
    import torch

    from mbirtorch import _memory_ledger

    cell = tuple(int(v) for v in spec["cell"])
    rows = []
    base = dict(row="ledger_price", name=spec["name"], cell=list(cell),
                part=spec["part"])
    try:
        # Nothing is reconstructed here, so the preflight is turned off for the
        # same reason the single-call setups in this series turn it off: it
        # prices a whole reconstruction, and this model only answers questions.
        model = build_model(cell, view_params_for(cell), device_list(1),
                            skip_preflight=True)
        report = bound_body_report(model)
        recon_shape = [int(s) for s in model.get_params("recon_shape")]
        weights_sentinel = np.zeros(1, dtype=np.float32)
    except Exception as exc:                                      # noqa: BLE001
        return [dict(base, count=None, error=str(exc)[:1200],
                     traceback=traceback.format_exc()[-2000:])]

    for count in counts:
        row = dict(base, count=int(count), recon_shape=recon_shape,
                   forward_body=report["forward_body"],
                   back_body=report["back_body"],
                   torch_body_directions=report["torch_body_directions"])
        try:
            devices = [torch.device(d) for d in device_list(count)]
            if len(devices) < count:
                # The smoke has one device, so a higher count is priced with
                # that device repeated.  The bytes are real ledger arithmetic
                # but they are not a layout any CUDA machine would run.
                devices = devices * count
                row["degraded"] = True
                row["degraded_reason"] = (
                    "there is one {} device here, so this count is priced with "
                    "it repeated; the transient budget on a non-CUDA device is "
                    "a constant rather than the sinogram-scaled one, so these "
                    "bytes are not what a CUDA layout would model"
                    .format(DEVICE))
            plan = _memory_ledger.plan_from_model(
                model, devices, workload="recon", weights=weights_sentinel)
            ledger = _memory_ledger.estimate_peak_device_bytes(plan)
            per_device = [int(b) for b in ledger.per_device_peaks()]
            busiest = max(per_device)
            index = per_device.index(busiest)
            row.update(
                modeled_peak_bytes_per_device=per_device,
                modeled_busiest_peak_bytes=busiest,
                modeled_devices=[str(d) for d in devices],
                dominant_phase=ledger.dominant_phase(index).name,
                dominant_terms=[[str(name), int(value)] for name, value
                                in ledger.dominant_phase(index)
                                .dominant_terms(index)],
                plan_weights_supplied=bool(plan.weights_supplied),
                plan_num_pixels_full=int(plan.num_pixels_full),
                plan_view_blocks=[int(v) for v in plan.view_blocks],
                plan_slice_blocks=[int(v) for v in plan.slice_blocks],
                plan_torch_body_directions=list(plan.torch_body_directions),
                entry_points="_memory_ledger.plan_from_model then "
                             "_memory_ledger.estimate_peak_device_bytes")
        except Exception as exc:                                  # noqa: BLE001
            row.update(error=str(exc)[:1200],
                       traceback=traceback.format_exc()[-2000:])
        rows.append(row)
    return rows


def ledger_rows(measured):
    """Every pricing row, one per (cell, count), with the measured peak beside
    the modeled one wherever an arm produced it."""
    out = []
    for spec in cells():
        # Every count this run cares about, whether or not an arm ran it: the
        # single-device 2048-class case is priced precisely because it is not
        # run.
        counts = sorted(set(tuple(int(c) for c in spec["counts"]) + (1, 2, 4)))
        for row in price_cell(spec, counts):
            arm = measured.get(arm_id(spec, row.get("count")))
            if arm and not arm.get("error"):
                row["measured_busiest_peak_bytes"] = arm.get(
                    "busiest_peak_bytes")
                row["measured_peak_bytes_per_device"] = arm.get(
                    "peak_bytes_per_device")
                row["measured_busiest_reserved_bytes"] = arm.get(
                    "busiest_reserved_bytes")
                modeled = row.get("modeled_busiest_peak_bytes")
                measured_busiest = row.get("measured_busiest_peak_bytes")
                row["modeled_over_measured"] = (
                    modeled / measured_busiest
                    if modeled and measured_busiest else None)
            elif arm:
                row["measured_status"] = str(arm.get("error"))[:200]
            else:
                row["measured_status"] = "no arm ran this count"
            out.append(row)
    return out


# ── the plan ──────────────────────────────────────────────────────────────────
def build_plan():
    """Every job, in run order: the identity probe, the 1024-class staging, then
    the arms, cheaper cell first and smaller device count first.

    Staging runs BEFORE the header row is written, because the header records
    the staged file's md5.  It is cheap to repeat: a staged file whose md5
    matches is reused, so a re-run stages nothing.
    """
    all_arms = []
    for spec in cells():
        for count in counts_for(spec):
            all_arms.append((spec, count))
    allowed = [arm_id(spec, count) for spec, count in all_arms]
    keep = _strict_subset("MG55_ARMS", allowed)

    probe = dict(kind="identity", job_id="identity")
    stages, arms = [], []
    for spec, count in all_arms:
        name = arm_id(spec, count)
        if name not in keep:
            continue
        cfg = dict(kind="arm", part=spec["part"], spec=spec, name=spec["name"],
                   count=count, devices=device_list(count), arm=name,
                   job_id=name, warm_repeats=warm_repeats_for(spec),
                   recorded_one_device=(spec["part"] == "A" and count == 1))
        if spec["part"] == "B":
            cfg["preflight_reason"] = (
                "the memory ledger is recorded as modeling the 1024-class cell "
                "at {:.0f} GB on one device against a measured peak of {:.2f} "
                "GB, so a preflight reading about twice the real peak could "
                "refuse a run that fits".format(
                    RECORDED_LEDGER_1024_ONE_DEVICE_GB,
                    RECORDED_MEASURED_1024_TORCH_GB))
        arms.append(cfg)
        if spec["part"] == "A" and not any(s["name"] == spec["name"]
                                           for s in stages):
            stages.append(dict(kind="stage", spec=spec, name=spec["name"],
                               job_id="stage_" + spec["name"]))
    if not arms:
        raise ValueError("MG55_ARMS selects no arm")
    return probe, stages, arms


def staged_gib(spec):
    """What the staged npz costs on disk: the sinogram and the weights, both
    float32, at the sinogram shape."""
    return 2.0 * sinogram_bytes(tuple(spec["cell"])) / 2 ** 30


def print_plan(probe, stages, arms):
    print("mg55 the multiaxis kernels across device counts, and one "
          "2048-class demonstration: {} arm(s), device {}, {} VCD "
          "iteration(s)".format(len(arms), DEVICE, VCD_ITERATIONS))
    print("  the kernels ran at 0.25, 0.23 and 0.22 of the torch bodies' warm "
          "wall on ONE device, with the largest temporary class gone from the "
          "peaks.  Whether sharding now divides those peaks and pays in speed, "
          "and whether the 2048-class runs at all, are the two things only "
          "several devices can answer.  This run answers both and decides "
          "nothing.")
    print("  rows -> {}".format(RESULTS_DIR))
    print("  interpreter: {}".format(sys.executable))
    print("  PYTHONPATH:  {}".format(os.environ.get("PYTHONPATH") or "(none)"))
    print("  every arm names its devices one by one: the automatic policy and "
          "its speed thresholds are not the subject, the layout is")
    print("  fingerprints are recorded, not gated; a gap above {:.0e} relative "
          "prints a note".format(FINGERPRINT_NOTE_LEVEL))

    print("\n  {:<{w}}{:>22}{:>22}{:>16}{:>9}  what it does".format(
        "job", "sinogram", "recon (mirrored)", "devices", "cap min",
        w=ARM_COL))
    print("  {:<{w}}{:>22}{:>22}{:>16}{:>9}  {}".format(
        probe["job_id"], "-", "-", "-", int(PROBE_TIMEOUT_S / 60),
        "names torch, mbirtorch, the devices and the tree witnesses",
        w=ARM_COL))
    for cfg in stages:
        spec = cfg["spec"]
        print("  {:<{w}}{:>22}{:>22}{:>16}{:>9}  {}".format(
            cfg["job_id"], str(tuple(spec["cell"])),
            str(tuple(mirrored_recon_shape(spec["cell"]))),
            device_label(device_list(1)),
            int(STAGE_TIMEOUT_S / 60),
            "reuses the staged npz ({:.1f} GiB), or builds it once with the "
            "kernels off".format(staged_gib(spec)), w=ARM_COL))
    for cfg in arms:
        spec = cfg["spec"]
        print("  {:<{w}}{:>22}{:>22}{:>16}{:>9}  {}".format(
            cfg["job_id"], str(tuple(spec["cell"])),
            str(tuple(mirrored_recon_shape(spec["cell"]))),
            device_label(cfg["devices"]),
            int(arm_timeout_s(spec) / 60),
            "cold pass then {} warm; {}".format(
                cfg["warm_repeats"],
                "input staged on disk" if spec["part"] == "A"
                else "input built in this arm, preflight off"), w=ARM_COL))
    print("  the recon shapes above come from this file's mirror of the "
          "geometry's own rule; every arm records the shape its model really "
          "realized and whether the two agreed")

    print("\n  part A, the 1024-class across device counts: the staged npz "
          "(md5 {}) reconstructed at each count on the default kernel "
          "selection.  Recorded per arm: the bound bodies, the walls, the peak "
          "allocated and reserved bytes on EVERY device in the list, the "
          "busiest device's reading, and a float64 fingerprint.  The parent "
          "computes the speedup against the one-device arm and the fingerprint "
          "gaps across counts.".format(
              RECORDED_STAGE_MD5.get("multiaxis_1024")))
    print("    the one-device kernel route was measured here at cold {:.2f} s, "
          "warm {:.2f} s, peak {:.2f} GB; the torch bodies at warm {:.2f} s, "
          "peak {:.2f} GB.  Context on the row, never a threshold.".format(
              RECORDED_ONE_DEVICE_KERNEL["cold_s"],
              RECORDED_ONE_DEVICE_KERNEL["warm_s"],
              RECORDED_ONE_DEVICE_KERNEL["peak_bytes"] / 2 ** 30,
              RECORDED_ONE_DEVICE_TORCH["warm_s"],
              RECORDED_ONE_DEVICE_TORCH["peak_bytes"] / 2 ** 30))

    print("\n  part B, the 2048-class demonstration: no npz -- each arm builds "
          "its own phantom, forward projects it through its own model and "
          "makes the weights, then reconstructs.  The single-device case is "
          "not run; it is priced in part C.  Each arm turns the memory "
          "preflight off, and the row carries the reason.")
    print("    an out-of-memory or a timeout is RECORDED as that arm's result "
          "and the run continues")

    print("\n  part C, the memory ledger's own numbers: for both cells at "
          "counts 1, 2 and 4, the modeled planned per-device peak for a "
          "reconstruction, read through plan_from_model and "
          "estimate_peak_device_bytes, beside the measured peaks where they "
          "exist.  It runs LAST and in the parent, because a parent that has "
          "initialized CUDA would otherwise sit inside every arm's peak "
          "reading.")

    if DEVICE != "cuda":
        print("\n  ON {} THE RUN DEGRADES, in three places: there is no triton "
              "at all, so both bodies are the torch ones; there is one device, "
              "so part A is a single arm and its scaling comparison has "
              "nothing to compare; and part C prices counts 2 and 4 with that "
              "one device repeated.  Every affected row says so."
              .format(DEVICE.upper()))

    print("\n  exit code = INSTRUMENT HEALTH ONLY: every planned arm produced "
          "a row or a recorded out-of-memory or timeout, every arm bound both "
          "kernels, the staged input md5-verified where it was reused, and the "
          "tree witnesses hold.  What any arm MEASURED never changes it -- "
          "including a 2048-class arm that ran out of memory.")
    print("  no library file is edited: what varies is the device list handed "
          "to the model")


# ── rows ──────────────────────────────────────────────────────────────────────
def write_row(sink, row):
    """One jsonl row, flushed.

    Flushed per row because a job that is killed mid-run should leave every row
    it had already finished.
    """
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    return row


# ── the comparison and the report ─────────────────────────────────────────────
def arm_status(row):
    if not row:
        return "not planned"
    if row.get("error") and row.get("timed_out"):
        return "timeout"
    if row.get("error") and is_capacity_reading(row):
        return "capacity"
    if row.get("error"):
        return "error"
    if row.get("invalid_reasons"):
        return "premise"
    if row.get("warm_s") is None:
        return "no wall"
    return "ok"


def compare_scaling(spec, arm_rows):
    """One cell across its device counts: the speedup against the smallest
    count that produced a wall, the per-device peaks, and the fingerprint gaps.

    The REFERENCE is the one-device arm where there is one, because that is the
    layout every recorded number for this problem was measured on.  When no
    one-device arm ran -- the 2048-class, which cannot have one -- the smallest
    count that produced a wall is the reference and the row says which.
    """
    # Ascending for the table and the reference, whatever order the arms ran
    # in: the 2048-class runs its four-device demonstration first.
    counts = sorted(counts_for(spec))
    out = dict(row="scaling", name=spec["name"], cell=list(spec["cell"]),
               counts=list(counts), per_count={})
    for count in counts:
        row = arm_rows.get(arm_id(spec, count)) or {}
        out["per_count"][str(count)] = dict(
            status=arm_status(row) if row else "not planned",
            cold_s=row.get("cold_s"), warm_s=row.get("warm_s"),
            warm_spread=row.get("warm_spread"),
            staging_s=row.get("staging_s"),
            peak_bytes_per_device=row.get("peak_bytes_per_device"),
            busiest_peak_bytes=row.get("busiest_peak_bytes"),
            busiest_reserved_bytes=row.get("busiest_reserved_bytes"),
            forward_body=row.get("forward_body"),
            back_body=row.get("back_body"),
            abs_sum=row.get("fingerprint_abs_sum"),
            sq_sum=row.get("fingerprint_sq_sum"),
            fm_rmse=row.get("fm_rmse"),
            fm_rmse_decreased=row.get("fm_rmse_decreased"),
            error=(str(row.get("error")).strip().splitlines()[-1][:300]
                   if row.get("error") else None))

    with_walls = [c for c in counts
                  if out["per_count"][str(c)].get("warm_s")]
    reference = 1 if 1 in with_walls else (min(with_walls) if with_walls
                                           else None)
    out["reference_count"] = reference
    ref = out["per_count"].get(str(reference)) if reference else None
    for count in counts:
        entry = out["per_count"][str(count)]
        entry["speedup_over_reference"] = (
            ref["warm_s"] / entry["warm_s"]
            if ref and entry.get("warm_s") else None)
        entry["peak_ratio_over_reference"] = (
            entry["busiest_peak_bytes"] / ref["busiest_peak_bytes"]
            if ref and ref.get("busiest_peak_bytes")
            and entry.get("busiest_peak_bytes") else None)
        entry["abs_sum_rel_gap"] = (relative_gap(entry.get("abs_sum"),
                                                 ref.get("abs_sum"))
                                    if ref else None)
        entry["sq_sum_rel_gap"] = (relative_gap(entry.get("sq_sum"),
                                                ref.get("sq_sum"))
                                   if ref else None)
    gaps = [g for count in counts
            for g in (out["per_count"][str(count)]["abs_sum_rel_gap"],
                      out["per_count"][str(count)]["sq_sum_rel_gap"])
            if g is not None]
    out["max_fingerprint_gap"] = max(gaps) if gaps else None
    out["fingerprint_note"] = (
        "" if not gaps or max(gaps) <= FINGERPRINT_NOTE_LEVEL else
        "the reconstructions differ across device counts by {:.2e} relative, "
        "above the {:.0e} this run treats as worth a second look.  What "
        "accumulates over {} iterations at different shard boundaries is "
        "summation order, which the library's own multi-device tolerance "
        "already covers.  Recorded, not a failure.".format(
            max(gaps), FINGERPRINT_NOTE_LEVEL, VCD_ITERATIONS))
    return out


def _fmt(value, width=10, kind="f", prec=2):
    if value is None:
        return "{:>{w}}".format("-", w=width)
    if isinstance(value, str):
        return "{:>{w}}".format(value, w=width)
    if kind == "d":
        return "{:>{w}d}".format(int(round(float(value))), w=width)
    return "{:>{w}.{p}{k}}".format(value, w=width, p=prec, k=kind)


def _gb_list(values):
    if not values:
        return "-"
    return " / ".join("{:.2f}".format(v / 2 ** 30) for v in values)


def print_scaling(item):
    print("\n### {} {} across device counts, warm median of a seeded "
          "{}-iteration reconstruction".format(
              item["name"], tuple(item["cell"]), VCD_ITERATIONS))
    print("| devices | cold s | warm s | spread | speedup | peak per device GB "
          "| busiest GB | peak ratio | abs gap |")
    print("|---|---|---|---|---|---|---|---|---|")
    for count in item["counts"]:
        entry = item["per_count"][str(count)]
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            count,
            "{:.2f}".format(entry["cold_s"]) if entry.get("cold_s")
            else entry["status"],
            "{:.2f}".format(entry["warm_s"]) if entry.get("warm_s") else "-",
            "{:.1%}".format(entry["warm_spread"])
            if entry.get("warm_spread") is not None else "-",
            "{:.2f}x".format(entry["speedup_over_reference"])
            if entry.get("speedup_over_reference") else "-",
            _gb_list(entry.get("peak_bytes_per_device")),
            "{:.2f}".format(entry["busiest_peak_bytes"] / 2 ** 30)
            if entry.get("busiest_peak_bytes") else "-",
            "{:.2f}x".format(entry["peak_ratio_over_reference"])
            if entry.get("peak_ratio_over_reference") else "-",
            "{:.2e}".format(entry["abs_sum_rel_gap"])
            if entry.get("abs_sum_rel_gap") is not None else "-"))
    print("Speedup and peak ratio are against the {}-device arm.  A speedup "
          "above 1 means more devices ran faster; a peak ratio below 1 means "
          "the busiest device held less.".format(item["reference_count"]))
    if item["fingerprint_note"]:
        print("NOTE {}".format(item["fingerprint_note"]))


def print_ledger_table(rows):
    print("\n### what the memory ledger models for a reconstruction, beside "
          "what was measured")
    print("| cell | devices | modeled per device GB | modeled busiest GB | "
          "measured busiest GB | modeled/measured | dominant phase |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        if row.get("error"):
            print("| {} | {} | {} | - | - | - | - |".format(
                tuple(row.get("cell") or ()), row.get("count"),
                str(row["error"]).strip().splitlines()[-1][:120]))
            continue
        measured = row.get("measured_busiest_peak_bytes")
        ratio = row.get("modeled_over_measured")
        print("| {} | {} | {} | {} | {} | {} | {} |".format(
            tuple(row["cell"]), row["count"],
            _gb_list(row.get("modeled_peak_bytes_per_device")),
            "{:.2f}".format(row["modeled_busiest_peak_bytes"] / 2 ** 30)
            if row.get("modeled_busiest_peak_bytes") else "-",
            "{:.2f}".format(measured / 2 ** 30) if measured
            else (row.get("measured_status") or "-"),
            "{:.2f}x".format(ratio) if ratio else "-",
            row.get("dominant_phase") or "-"))
    print("These rows are read only: the ledger is asked what it models, "
          "through its own plan and estimate entry points, and nothing is "
          "changed.  A modeled/measured ratio well above 1 is the ledger "
          "pricing the worst concurrency it charges for; that is the "
          "conservative direction, and it is also what makes a preflight "
          "refuse runs that fit.")


def summarize(identity, stage_rows, arm_rows, scalings, arms, ledger,
              findings, out_path):
    """The tables a person reads, and the instrument-health accounting the exit
    code comes from.

    These are two different things and this function keeps them apart.  A slow
    arm, a wide spread, an arm that ran out of memory and an arm that hit the
    time cap are FINDINGS: they are printed and none of them touches the exit
    code.  A missing row, an md5 that did not verify, an arm that bound the
    wrong bodies, and an error that is not a capacity or timeout reading are
    instrument failures, because they mean the run did not measure what the
    plan said it would.
    """
    print("\n===== mg55 the multiaxis kernels across device counts ({}) "
          "=====".format(out_path))
    broken = []
    findings = list(findings)

    header = ("{:<{w}}{:>9}{:>10}{:>9}{:>12}{:>11}{:>10}"
              .format("arm", "cold s", "warm s", "spread", "busiest GB",
                      "resvd GB", "state", w=ARM_COL))
    print(header)
    print("-" * len(header))
    for cfg in arms:
        name = cfg["arm"]
        row = arm_rows.get(name)
        if row is None:
            print("{:<{w}}  no row".format(name, w=ARM_COL))
            broken.append("{}|no row".format(name))
            continue
        if row.get("error"):
            capacity = is_capacity_reading(row)
            state = "timeout" if row.get("timed_out") else (
                "capacity" if capacity else "ERROR")
            print("{:<{w}}  {}: {}".format(
                name, state,
                str(row["error"]).strip().splitlines()[-1][:90], w=ARM_COL))
            if capacity:
                findings.append("{}: {}".format(name, state))
            else:
                broken.append("{}|{}".format(
                    name, str(row["error"]).strip().splitlines()[-1][:200]))
            continue
        print("{:<{w}}{}{}{:>9}{}{}{:>10}".format(
            name, _fmt(row.get("cold_s"), 9), _fmt(row.get("warm_s"), 10),
            "-" if row.get("warm_spread") is None
            else "{:.1%}".format(row["warm_spread"]),
            _fmt(None if row.get("busiest_peak_bytes") is None
                 else row["busiest_peak_bytes"] / 2 ** 30, 12),
            _fmt(None if row.get("busiest_reserved_bytes") is None
                 else row["busiest_reserved_bytes"] / 2 ** 30, 11),
            "ok" if not row.get("invalid_reasons") else "PREMISE",
            w=ARM_COL))
        print("{:<{w}}  devices {} | bodies {} / {}{}".format(
            "", device_label(row.get("realized_devices") or []),
            row.get("forward_body"), row.get("back_body"),
            "  [route degraded]" if row.get("route_degraded") else "",
            w=ARM_COL))
        if row.get("peak_bytes_per_device"):
            print("{:<{w}}  peak per device GB {}".format(
                "", _gb_list(row["peak_bytes_per_device"]), w=ARM_COL))
        if row.get("staging_s") is not None:
            print("{:<{w}}  staging {:.1f} s (phantom {:.1f}, forward "
                  "projection {:.1f})".format(
                      "", row["staging_s"], row.get("phantom_s") or 0.0,
                      row.get("forward_project_s") or 0.0, w=ARM_COL))
        if row.get("fm_rmse"):
            print("{:<{w}}  forward-model error {} ({})".format(
                "", ", ".join("{:.4g}".format(v) for v in row["fm_rmse"]),
                "fell" if row.get("fm_rmse_decreased") else "did not fall",
                w=ARM_COL))
        for reason in row.get("invalid_reasons") or []:
            print("    ARM CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(name, reason))
        if row.get("devices_as_asked") is False:
            findings.append("{}: asked for {} and realized {}".format(
                name, row.get("requested_devices"),
                row.get("realized_devices")))
        if row.get("route_degraded"):
            findings.append("{}: {}".format(name,
                                            row.get("route_degraded_reason")))
        if row.get("fm_rmse_decreased") is False:
            findings.append("{}: the forward-model error did not fall over "
                            "the iterations ({})".format(name,
                                                         row.get("fm_rmse")))

    if stage_rows:
        print("\n-- the staged input --")
    for row in stage_rows:
        print("  {} {}: md5 {}{}{}".format(
            row.get("name"), tuple(row.get("sinogram_shape") or ()),
            row.get("md5", "-"),
            "  (reused from {})".format(row.get("reused_from"))
            if row.get("reused") else "  (built here)",
            "  same bytes the earlier runs measured"
            if row.get("same_bytes_as_recorded") else ""))
        if row.get("phantom_fallback"):
            print("    phantom: {}".format(row["phantom_fallback"]))
        if row.get("error"):
            broken.append("{}|{}".format(row.get("job_id"),
                                         str(row["error"])[:200]))
        for reason in row.get("invalid_reasons") or []:
            print("    STAGE CHECK FAIL: {}".format(reason))
            broken.append("{}|{}".format(row.get("job_id"), reason))
        if row.get("recon_shape_mirror_agrees") is False:
            findings.append("{}: the staged recon shape {} is not this file's "
                            "mirror of the geometry's rule, {}".format(
                                row.get("name"), row.get("recon_shape"),
                                row.get("mirrored_recon_shape")))

    for item in scalings:
        print_scaling(item)
        if item["fingerprint_note"]:
            findings.append("{}: {}".format(item["name"],
                                            item["fingerprint_note"]))
    print_ledger_table(ledger)
    for row in ledger:
        if row.get("error"):
            findings.append("the ledger pricing for {} at {} devices failed: "
                            "{}".format(row.get("name"), row.get("count"),
                                        str(row["error"])[:200]))
        elif row.get("degraded"):
            findings.append("the ledger pricing for {} at {} devices is "
                            "degraded: {}".format(row.get("name"),
                                                  row.get("count"),
                                                  row.get("degraded_reason")))

    print("\n-- what ran --")
    row = identity or {}
    commit = (row.get("git") or {}).get("commit")
    print("  torch {} | triton {} | {} | {} device(s)".format(
        row.get("torch_version", "?"), row.get("triton_version", "?"),
        ", ".join(row.get("device_names") or ["?"]),
        row.get("device_count", "?")))
    print("  mbirtorch {} | commit {}{}".format(
        row.get("library_file", "?"), commit or "unknown",
        " (dirty)" if (row.get("git") or {}).get("dirty") else ""))
    if row.get("error"):
        print("    PROBE FAILED: {}".format(str(row["error"])[-300:]))
        broken.append("identity|{}".format(
            str(row["error"]).strip().splitlines()[-1][:200]))
    witnesses = (row.get("tree_witnesses") or {})
    if witnesses.get("ok"):
        print("  tree witnesses ok: the padded, recompile-remedied tree, the "
              "kernel module imports, and the geometry's selection consults "
              "both availability checks and reaches both kernels")
    else:
        print("  TREE WITNESSES: {}".format(witnesses))
        broken.append("tree witnesses|{}".format(witnesses))

    print("\n-- instrument health --")
    print("  the exit code covers four things: every planned arm produced a "
          "row or a recorded out-of-memory or timeout, every arm really bound "
          "both kernels, the staged input md5-verified where it was reused, "
          "and the tree witnesses hold.  What any arm MEASURED never changes "
          "it, including a 2048-class arm that ran out of memory.")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm produced a result, every arm bound the "
              "bodies this run is about, the staged input verified its md5, "
              "and the tree witnesses hold")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings outside the tables")

    return dict(row="summary", healthy=not broken, broken=broken,
                findings=findings, scalings=scalings,
                ledger_rows=len(ledger),
                arms={name: dict(warm_s=row.get("warm_s"),
                                 cold_s=row.get("cold_s"),
                                 staging_s=row.get("staging_s"),
                                 busiest_peak_bytes=row.get(
                                     "busiest_peak_bytes"),
                                 peak_bytes_per_device=row.get(
                                     "peak_bytes_per_device"),
                                 forward_body=row.get("forward_body"),
                                 back_body=row.get("back_body"),
                                 error=row.get("error"))
                      for name, row in arm_rows.items()},
                out_path=out_path)


# ── the child entry point ─────────────────────────────────────────────────────
def _child_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = run_job(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)
    return 0


def main():
    probe, stages, arms = build_plan()
    print_plan(probe, stages, arms)
    if DRY:
        return 0
    findings = []

    # ── the identity probe and the staging, both before the header row ───────
    print("\n-- identity probe --", flush=True)
    identity = spawn(probe, PROBE_TIMEOUT_S)
    print("  {}".format(identity.get("error") or
                        "torch {} | {} | {} device(s) | {}".format(
                            identity.get("torch_version"),
                            ", ".join(identity.get("device_names") or []),
                            identity.get("device_count"),
                            identity.get("library_file"))), flush=True)

    print("\n-- staged input (reused when the md5 verifies) --", flush=True)
    stage_rows = []
    for index, cfg in enumerate(stages):
        print("  [{}/{}] {}".format(index + 1, len(stages), cfg["job_id"]),
              flush=True)
        row = spawn(cfg, STAGE_TIMEOUT_S)
        stage_rows.append(row)
        if row.get("error"):
            print("    ERROR: {}".format(str(row["error"])[:400]), flush=True)
        else:
            print("    md5 {} {} recon {}".format(
                row.get("md5"),
                "(reused from {})".format(row.get("reused_from"))
                if row.get("reused") else "(built here)",
                row.get("recon_shape")), flush=True)
    staged_by_name = {row.get("name"): row for row in stage_rows}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR,
        "mg55_multiaxis_scale_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    arm_rows = {}
    scalings = []
    ledger = []
    with open(out_path, "w") as sink:
        write_row(sink, dict(
            row="run_header", script="mg55_multiaxis_scale.py",
            node=RUN_LABEL, stamp=stamp, device=DEVICE, smoke=SMOKE,
            driver_python=sys.executable,
            pythonpath=os.environ.get("PYTHONPATH"),
            results_dir=RESULTS_DIR, stage_search_dirs=stage_search_dirs(),
            identity=identity, tree_witnesses=identity.get("tree_witnesses"),
            staged_md5={row.get("name"): row.get("md5")
                        for row in stage_rows},
            recorded_stage_md5=RECORDED_STAGE_MD5,
            recorded_one_device_kernel=RECORDED_ONE_DEVICE_KERNEL,
            recorded_one_device_torch=RECORDED_ONE_DEVICE_TORCH,
            recorded_ledger_1024_one_device_gb=(
                RECORDED_LEDGER_1024_ONE_DEVICE_GB),
            cells=[dict(spec) for spec in cells()],
            vcd_iterations=VCD_ITERATIONS, vcd_seed=VCD_SEED,
            warm_repeats=WARM_REPEATS, warm_repeats_2048=WARM_REPEATS_2048,
            arm_timeout_s=ARM_TIMEOUT_S, arm_2048_timeout_s=ARM_2048_TIMEOUT_S,
            run_2048_n2=RUN_2048_N2,
            fingerprint_note_level=FINGERPRINT_NOTE_LEVEL,
            plan=[dict(kind=cfg["kind"], job_id=cfg["job_id"],
                       name=cfg.get("name"), count=cfg.get("count"),
                       devices=cfg.get("devices"))
                  for cfg in [probe] + list(stages) + list(arms)]))
        for row in stage_rows:
            write_row(sink, dict(row="stage", **row))

        for index, cfg in enumerate(arms):
            print("\n  [{}/{}] {} on {}".format(
                index + 1, len(arms), cfg["job_id"],
                device_label(cfg["devices"])), flush=True)
            spec = cfg["spec"]
            if spec["part"] == "A":
                stage_row = staged_by_name.get(cfg["name"]) or {}
                if not stage_row.get("md5_ok") or not stage_row.get(
                        "stage_path"):
                    # No verified input, so there is nothing honest to time.
                    # The staging row already carries the reason; this arm
                    # records that it never ran rather than running on
                    # unverified bytes.
                    row = dict(cfg, error="this cell has no md5-verified "
                                          "staged input; see its staging row")
                else:
                    row = spawn(dict(cfg, stage_path=stage_row["stage_path"],
                                     stage_md5=stage_row["md5"]),
                                arm_timeout_s(spec))
            else:
                row = spawn(cfg, arm_timeout_s(spec))
            arm_rows[cfg["arm"]] = row
            write_row(sink, dict(row="arm", **row))
            if row.get("error"):
                print("    {}: {}".format(
                    "READING" if is_capacity_reading(row) else "ERROR",
                    str(row["error"]).strip().splitlines()[-1][:200]),
                    flush=True)
            elif row.get("invalid_reasons"):
                print("    PREMISE: {}".format(
                    str(row["invalid_reasons"][0])[:250]), flush=True)
            else:
                print("    bodies {} / {}".format(row.get("forward_body"),
                                                  row.get("back_body")),
                      flush=True)
                print("    cold {:.2f}s  warm {:.2f}s  spread {:.1%}  "
                      "busiest peak {}  per device {}".format(
                          row.get("cold_s", 0.0), row.get("warm_s", 0.0),
                          row.get("warm_spread", 0.0),
                          "-" if row.get("busiest_peak_bytes") is None
                          else "{:.2f} GB".format(
                              row["busiest_peak_bytes"] / 2 ** 30),
                          _gb_list(row.get("peak_bytes_per_device"))),
                      flush=True)
            # The scaling row goes out as soon as every PLANNED arm of a cell
            # is in, so a job cut short still carries every cell it finished.
            planned = [other["arm"] for other in arms
                       if other["name"] == spec["name"]]
            if all(name in arm_rows for name in planned):
                item = compare_scaling(spec, arm_rows)
                scalings.append(item)
                write_row(sink, item)
                print_scaling(item)

        # ── the ledger pricing, last and in this process ─────────────────────
        print("\n  the memory ledger's own numbers, read only", flush=True)
        try:
            ledger = ledger_rows(arm_rows)
        except Exception as exc:                                  # noqa: BLE001
            ledger = [dict(row="ledger_price", error=str(exc)[:1200],
                           traceback=traceback.format_exc()[-2000:])]
        for row in ledger:
            write_row(sink, row)

        summary = summarize(identity, stage_rows, arm_rows, scalings, arms,
                            ledger, findings, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        write_row(sink, summary)
    print("\nwrote {}".format(out_path))
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


if __name__ == "__main__":
    if CHILD:
        sys.exit(_child_main(CHILD, CHILD_OUT))
    sys.exit(main())
