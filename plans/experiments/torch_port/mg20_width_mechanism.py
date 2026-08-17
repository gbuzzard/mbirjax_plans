"""mg20 -- WHY THE PARALLEL FORWARD KERNEL COSTS THE SAME AT VALUES WIDTH 504
AS AT WIDTH 1008.

WHAT IS ALREADY MEASURED.  The parallel-beam forward projection runs a
hand-written Triton kernel (mbirtorch/triton_parallel.py, the body
``_parallel_forward_view_batch_triton``).  mg10 measured that kernel's cost
against the width of the values block it is handed, at one device, and the
result is multigpu_findings.md section 1.9:

    values-block width       per launch    per slice
    1008 (the shipped call)     41.38 ms    0.0411 ms
     504                        41.46 ms    0.0823 ms
     252                        21.18 ms    0.0840 ms
      63                         4.69 ms    0.0744 ms

Below 504 the cost per launch falls roughly in proportion to the width.  From
504 to 1008 it does not fall at all.  A launch covering 504 columns therefore
does half the work of a launch covering 1008 columns in the same time, and its
cost per slice is twice as high.  Section 1.9 named grid occupancy as the
leading candidate and left the mechanism open.

THE TWO QUESTIONS THIS RUN ASKS.  They are independent, and the run keeps them
apart on purpose.

  1. THE WIDTH REGIME.  Does the doubling point at how the launch fills the
     machine, or at how the launch uses memory?  Occupancy falling as the
     width falls, with the memory counters flat, points one way.  Memory
     counters degrading at the narrow widths, with occupancy flat, points the
     other.

  2. THE SORTED-CHANNEL QUESTION.  The forward kernel scatters each pixel's
     contribution into the sinogram with one atomic add per detector channel
     tap.  Neighbouring pixels land on neighbouring channels, but the order is
     not enforced, so the writes are scattered.  An in-kernel accumulation
     that sorted the channels first would make those writes ordered.  Whether
     that is worth building is read from four write-path counters: the atomic
     sector count, the reduction sector count, the L2 hit rate, and the DRAM
     write traffic.  Those four are read AT EVERY WIDTH.  That reading does
     not depend on the answer to question 1.  An occupancy answer to question
     1 does not close question 2, and this file does not present it as if it
     did.

THE CELL.  Parallel beam at sinogram (1024, 1008, 992), which gives recon
(992, 992, 1008) and 771,240 pixels inside the region-of-reconstruction mask.
This is mg9's and mg10's cell, built the same way, so a row here sits beside
their rows.  One device.

THE VALUES BLOCK.  Every timed launch is handed a values block of
(pixels, columns) float32, seeded, resident on the device.  The columns are
reconstruction slices, which under parallel beam are detector rows one for
one.  A launch's column count is the width this run sweeps.

THE VIEW BATCH IS PINNED AT 128 and is not a variable here.  128 is what the
driver chose at this cell in mg10 (the kernel's nominal view chunk), so the
launches have the shape production launches have, and the width is the only
thing that moves.

THE TWO LEGS.  The timing leg needs no profiler and always runs.  The
profiler leg uses Nsight Compute and is optional.  A cluster that refuses
performance counters to unprivileged users leaves the timing leg standing
alone, and that refusal is recorded as a finding rather than treated as a
failure.

## THE TIMING LEG

Each arm cuts the 1008-column values block into pieces of a fixed width,
projects every piece, and reassembles the sinogram.  Rows track slices under
parallel beam, so piece k's output is detector rows [l0, l1) and the pieces
concatenate along the row axis in slice order.  This is mg10's construction,
reused.

    arm              piece widths        layout of the values memory
    w1008            1008                contiguous, row stride 1008
    w504             504, 504            contiguous, row stride 504
    w252             252 x 4             contiguous, row stride 252
    w63              63 x 16             contiguous, row stride 63
    w512_496         512, 496            contiguous, row stride 512 / 496
    w504_alloc1008   504, 504            contiguous inside a 1008-wide buffer
    w512_stride1008  512, 512            row stride 1008, 512 columns of work
    w512_stride2016  512, 512            row stride 2016, 512 columns of work

The first four arms are section 1.9's, and reproducing them on the current
tree is this run's validity check.  The last four are the new ablations.

THE STRIDED ARMS ARE THE ONES THE QUESTION TURNS ON.  They give the kernel
504-sized work over 1008-sized memory, so work and layout move separately for
the first time.  If a 512-column launch over a 1008-stride buffer costs what a
1008-column launch costs, the regime travels with the work.  If it costs half
that, the regime travels with the layout.

WHAT THE LIBRARY ALLOWS, read before the arms were designed.  The wrapper
packs its values argument.  It reads ``num_value_cols = int(values.shape[1])``
and then calls ``values = values.contiguous()`` (triton_parallel.py lines 436
to 437).  The kernel has no separate stride argument for the values block.
Its ``num_cols`` argument is three things at once: the values row stride, the
mask that sets the logical width, and the output row stride.

    vals = tl.load(values_ptr + p_offs[:, None] * num_cols + r_offs[None, :], ...)
    out_ptrs = out_view_ptr + n_chan[:, None] * num_cols + r_offs[None, :]

A values tensor whose row stride exceeds its column count therefore cannot be
passed to the shipped kernel at all.  The strided arms are built the other way
round, and the change is on the launch side only.  They pass the whole
1008-wide block, or the 2016-wide one, with ``num_cols`` set to its true row
stride.  They then truncate the grid's column axis to four blocks.  Four
blocks of BLOCK_R=128 is 512 columns of work, which is within 1.6 percent of
504.  The kernel reads and writes at the wide stride while doing the narrow
width's work, which is the intended ablation.

No library file is touched by any of this.  The arm copies the wrapper's
launch code into this file instead.  A fidelity check then runs that copy at
the full width against the library wrapper, before any truncated launch is
believed.

Two launches cover the whole 1008 columns for a strided arm: one at column
offset 0 and one at offset 496.  They overlap on columns 496 to 511, so the
assembly takes columns 0 to 495 from the first and 496 to 1007 from the
second.  Each column is therefore computed once in the assembled sinogram.

w504_alloc1008 is the fallback that would have carried this question if the
strided arms had turned out to be impossible.  It is kept because it separates
one thing the strided arms do not.  Its values are contiguous and 504 wide,
inside a buffer allocated 1008 wide whose second half is never touched.  It
moves the allocation size and nothing else, so it says whether allocation size
alone matters.  It is not a layout test.

w512_496 is a single-variable ablation of one specific mechanism.  Triton
specializes an integer kernel argument on whether it is divisible by 16, and
the alignment it can then assume for the addresses derived from that argument
sets how wide a vector access it may use.  ``num_cols`` is such an argument,
and it is the values row stride, the width mask and the output row stride at
once.  Of the four widths in section 1.9 only 1008 is divisible by 16; 504,
252 and 63 are not.  w512_496 cuts the same volume into two pieces that ARE
both divisible by 16 (512 and 496) at a width within 1.6 percent of w504's.
If the per-slice cost drops to w1008's, that specialization is the mechanism.

## THE PIXEL LADDER, AND WHAT "PER LAUNCH" MEANS IN SECTION 1.9

Section 1.9's per-launch numbers are means over a mixture of pixel counts,
not the cost of one launch shape.  mg10 timed every forward body call of a
three-iteration reconstruction and divided by the call count.  Its own rows
record the mixture: of the 680 calls in one timed reconstruction, 512 ran on
12,051 pixels, 128 on 48,203, 32 on 192,810 and 8 on the full 771,240, which
is 36,294 pixels per call on average.  A single full-pixel-set launch is
therefore NOT 41.4 ms.

mg20 times each arm at all four of those pixel counts.  The full-pixel point
is the headline reading and is the shape the profiler leg profiles.  The four
points together let this run rebuild the same weighted mean mg10 reported, so
the absolute anchors can be compared as well as their ratios.

## THE VALUES CHECK

Timing an arm that computes a different sinogram measures nothing, so every
arm assembles its sinogram and compares it against a full-width reference
before its timing is read.  The reference is one full-width launch through the
shipped library wrapper, taken in the arm's own process at the full pixel set.
The comparison is

    rel = max|assembled - reference| / max|reference|

and the gate is 1e-5.  The forward kernel accumulates with float atomic adds,
which are commutative but not associative, so two launches of the same call do
not agree bit for bit.  The w1008 arm's own comparison is that repeat floor
and is printed beside the others.

## THE PROFILER LEG

Nsight Compute is at the CUDA module's bin, so the sbatch loads the cuda
module and ``ncu`` is on PATH.  The leg runs in three steps.

  1. A PERMISSION PROBE.  One trivial kernel is profiled for one metric.  Many
     clusters leave GPU performance counters closed to unprivileged users, and
     the probe's failure message says so (ERR_NVGPUCTRPERM).  A refusal is
     recorded and the leg stops there.

  2. KERNEL NAME DISCOVERY.  The name is read at runtime and never guessed.
     One plain run of this file's single-launch worker reports the CUDA kernel
     names torch.profiler saw, and the names Triton's own compiled-kernel
     cache holds.  The name that matches the parallel forward kernel becomes
     the ``--kernel-name`` filter.

  3. ONE WARM LAUNCH PER WIDTH.  The single-launch worker builds the model,
     builds the values, launches the arm's first piece a few times and exits,
     so the profile holds one kernel rather than a whole sweep.  That worker
     imports the Triton body directly instead of going through the model's
     body selection, because the selection's availability self-check launches
     the same kernel once on a tiny problem and would otherwise be the first
     thing ncu matched.  ``--launch-skip`` then lands on a warm launch.

NSIGHT COMPUTE DURATIONS ARE NOT WALL TIMES.  ncu serializes kernels and
replays each one several times to collect its counters.  The durations in the
counter table are useful for comparing widths within that table and are not
comparable to the timing leg's milliseconds.  The timing leg owns time.

The metrics fall into two groups, one per question.  Question 1 is answered by
the launch shape, the three occupancy limits and the achieved occupancy.
Question 2 is answered by the L2 hit rate, the L2 sectors that arrived as
atomics and as reductions, and the DRAM bytes read and written.

## THE EXIT CODE

The exit code reports INSTRUMENT HEALTH only.  It is 0 when every selected
timing arm produced a row, every arm passed its values check, and every
section 1.9 arm reproduced its anchor within 15 percent.

The anchor test is taken on the COST PER SLICE RELATIVE TO WIDTH 1008.  That
ratio is the relationship section 1.9 states.  It is also the quantity that
survives the difference between mg10's launches and this run's.  mg10 timed
calls embedded in a reconstruction and averaged them over a pixel mixture,
and this run times isolated warm launches at a fixed pixel count.  The
absolute per-launch milliseconds are reported beside the ratios, and so is the
rebuilt mixture mean.  Neither absolute number is gated.

The profiler leg's absence never changes the exit code.  Neither does what
either leg finds.  The mechanism verdict is read by a person from the two
tables and the reading guide this job prints.

## THE LOCAL SMOKE

MG20_SMOKE=1 runs the whole arm plan on a tiny parallel cell on the CPU.  The
Triton kernel is unavailable there, so the model binds its torch forward body,
the two strided arms are skipped and recorded, and the profiler leg is skipped
and recorded.  What the smoke exercises is the harness: the piece cutting, the
reassembly, the values check, the pixel ladder, the subprocess protocol, the
rows and the tables.  It is not a measurement.

Run:
    <torch python> mg20_width_mechanism.py          on a 1-GPU node
    MG20_DRY=1 <python> mg20_width_mechanism.py     print the plan and stop
    MG20_SMOKE=1 <python> mg20_width_mechanism.py   the local CPU smoke

Configuration is by environment variable only; there is no command line.
Export from the SUBMITTING SHELL, never through an sbatch --export list, which
slurm splits on commas.  An unrecognized arm name is an error, not a silent
skip.
    MG20_RESULTS=<dir>              where the jsonl and the ncu logs go
    MG20_ARMS=w1008,w504            subset of the arms, by arm name
    MG20_DRY=1                      print the plan and exit
    MG20_SMOKE=1                    the local CPU smoke
    MG20_NCU=0                      skip the profiler leg entirely
    MG20_NCU_PIXEL_DIV=4            profile at a quarter of the pixels
    MG20_REPEATS=5                  timed repeats per arm per pixel point
"""

import contextlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG20_SMOKE", "0") == "1"
DRY = os.environ.get("MG20_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

# mg9's and mg10's cell: (views, detector rows, detector channels).  The recon
# is (992, 992, 1008), so the values block has 1008 columns.
CELL = (1024, 1008, 992)
SMOKE_CELL = (8, 24, 20)
RECON_SHAPE_EXPECTED = (992, 992, 1008)
NUM_PIXELS_EXPECTED = 771240

# The view batch every launch uses.  128 is the forward kernel's nominal view
# chunk (triton_parallel.PARALLEL_FWD_VIEW_CHUNK) and is what the driver chose
# at this cell in mg10, at every width.  Pinned so the width is the only
# variable; the driver's own choice is recorded beside it on every row.
VIEW_BATCH = 128

# Four blocks of the forward kernel's BLOCK_R=128 is 512 columns of work.  This
# is the grid truncation the strided arms use; see the module docstring.
STRIDE_GRID_COLS = 4

TIMED_REPEATS = max(1, int(os.environ.get("MG20_REPEATS", "2" if SMOKE else "3")))
WARM_PASSES = 1                  # discarded before the timed repeats
VALUES_SEED = 20260817

# mg10's own pixel mixture, from the p1_shipped row of
# rows/mg10_shape_sweep_h008_20260810_201612.jsonl: divisor of the full pixel
# count -> how many launches of one timed reconstruction ran at that count.
# Used two ways: as the ladder this run times, and as the weights that rebuild
# section 1.9's mean per launch.
PIXEL_LADDER = ((64, 512), (16, 128), (4, 32), (1, 8))

# Section 1.9's readings, one device, on an H100 (job 15159551 on h008).
# per_launch_ms is a mean over the pixel mixture above; ms_per_slice is that
# mean divided by the width.  The gated quantity is per_slice_ratio, the cost
# per slice relative to width 1008 -- see the module docstring.
ANCHOR_1_9 = {
    "w1008": dict(per_launch_ms=41.38, ms_per_slice=0.0411, per_slice_ratio=1.00),
    "w504": dict(per_launch_ms=41.46, ms_per_slice=0.0823, per_slice_ratio=2.00),
    "w252": dict(per_launch_ms=21.18, ms_per_slice=0.0840, per_slice_ratio=2.04),
    "w63": dict(per_launch_ms=4.69, ms_per_slice=0.0744, per_slice_ratio=1.81),
}
ANCHOR_TOL = 0.15

VALUES_GATE_REL = 1e-5
VALUES_EXPECTATION = (
    "near machine zero on CPU (the torch body has no atomics)" if SMOKE else
    "1e-6 class: the kernel accumulates with float atomic adds, so two "
    "launches of the same call differ by float rounding")

# ── the arm plan ──────────────────────────────────────────────────────────────
# widths must sum to the slice count; the strided arms are described by their
# work width instead, because their launches overlap.
ARM_TABLE = (
    dict(arm="w1008", layout="contig", widths=(1008,), smoke_widths=(24,),
         anchor="w1008",
         note="the shipped call"),
    dict(arm="w504", layout="contig", widths=(504, 504), smoke_widths=(12, 12),
         anchor="w504",
         note="section 1.9's doubling point"),
    dict(arm="w252", layout="contig", widths=(252,) * 4, smoke_widths=(6,) * 4,
         anchor="w252", note=""),
    dict(arm="w63", layout="contig", widths=(63,) * 16, smoke_widths=(3,) * 8,
         anchor="w63", note="section 1.9's ten percent bonus width"),
    dict(arm="w512_496", layout="contig", widths=(512, 496),
         smoke_widths=(16, 8), anchor=None,
         note="both widths divisible by 16, against w504's 504"),
    dict(arm="w504_alloc1008", layout="alloc_full", widths=(504, 504),
         smoke_widths=(12, 12), anchor=None,
         note="504 wide inside a 1008-wide allocation"),
    dict(arm="w512_stride1008", layout="stride", stride_multiple=1,
         smoke_widths=None, anchor=None, needs_triton=True,
         note="512 columns of work at row stride 1008"),
    dict(arm="w512_stride2016", layout="stride", stride_multiple=2,
         smoke_widths=None, anchor=None, needs_triton=True,
         note="512 columns of work at row stride 2016"),
)
ARM_NAMES = tuple(spec["arm"] for spec in ARM_TABLE)

# ── the profiler leg ──────────────────────────────────────────────────────────
NCU_ENABLED = os.environ.get("MG20_NCU", "1") == "1"
NCU_PIXEL_DIV = max(1, int(os.environ.get("MG20_NCU_PIXEL_DIV", "1")))
NCU_ARMS = ("w1008", "w504", "w252", "w63", "w512_stride1008")
NCU_LAUNCHES = 5                 # the single-launch worker's launch count
# Two bounds on the profiler leg, because ncu replays each kernel several
# times to collect its counters and nobody has timed a replay of THIS kernel.
# One attempt cannot run longer than the first, and the whole leg cannot run
# longer than the second; whatever is left unprofiled is recorded as such.  The
# timing leg has already finished by then, so a leg that runs out of budget
# costs the run nothing it needed.
NCU_TIMEOUT_S = 360
NCU_LEG_BUDGET_S = 1200
NCU_PROBE_TIMEOUT_S = 180
NCU_METRICS = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_blocks",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_red.sum",
    "lts__t_sectors_op_atom.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
)
NCU_PERMISSION_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission",
                          "insufficient permission")

# ── GPU health ────────────────────────────────────────────────────────────────
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
    "MG20_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 17
# ──────────────────────────────────────────────────────────────────────────────


def _strict_subset(env_name, allowed):
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
        if token not in allowed:
            raise ValueError(f"{env_name}: {token!r} is not one of "
                             f"{list(allowed)}")
        if token not in chosen:
            chosen.append(token)
    if not chosen:
        raise ValueError(f"{env_name}: no valid tokens in {raw!r}")
    return [name for name in allowed if name in chosen]


def cell():
    return SMOKE_CELL if SMOKE else CELL


def num_slices():
    """The values block's column count: the reconstruction's slice count, which
    for a row-aligned geometry is also the sinogram's detector row count."""
    return int(cell()[1])


def arm_pieces(spec):
    """The pieces one arm launches, as absolute column ranges.

    A piece carries four things: where its launch starts in the values block,
    how many columns of work that launch covers, and which absolute columns of
    the assembled sinogram it contributes.  For the contiguous arms the launch
    covers exactly its own columns.  For the strided arms two launches of 512
    columns cover 1008 columns with an overlap, and the assembly takes each
    column from exactly one of them.
    """
    total = num_slices()
    if spec["layout"] == "stride":
        stride = total * int(spec["stride_multiple"])
        # BLOCK_R is 128 for any width at or above 128, which both strides are.
        work = STRIDE_GRID_COLS * 128
        if not work <= total <= 2 * work:
            raise ValueError(
                f'{spec["arm"]}: {work} columns of work cannot cover {total} '
                "columns in two launches")
        cut = total - work
        return [dict(index=0, start=0, work=work, stride=stride,
                     keep_local=(0, cut), keep_abs=(0, cut)),
                dict(index=1, start=cut, work=work, stride=stride,
                     keep_local=(cut, total), keep_abs=(cut, total))]
    widths = spec["smoke_widths"] if SMOKE else spec["widths"]
    if sum(widths) != total:
        raise ValueError(f'{spec["arm"]}: widths {widths} sum to {sum(widths)}, '
                         f"not the {total} columns of the values block")
    pieces, start = [], 0
    for index, width in enumerate(widths):
        pieces.append(dict(index=index, start=start, work=width, stride=width,
                           keep_local=(0, width),
                           keep_abs=(start, start + width)))
        start += width
    return pieces


def build_plan():
    """One entry per selected arm, in table order."""
    keep = _strict_subset("MG20_ARMS", ARM_NAMES)
    plan = []
    for spec in ARM_TABLE:
        if spec["arm"] not in keep:
            continue
        pieces = None
        skip = None
        if spec.get("needs_triton") and SMOKE:
            skip = ("the strided launch needs the Triton kernel, and the smoke "
                    "runs on the CPU where the model binds its torch body")
        else:
            pieces = arm_pieces(spec)
        plan.append(dict(arm=spec["arm"], layout=spec["layout"],
                         anchor=spec["anchor"], note=spec["note"],
                         pieces=pieces, skip=skip,
                         cell=list(cell()), num_slices=num_slices()))
    if not plan:
        raise ValueError("MG20_ARMS selects no arm")
    return plan


# ── GPU health, copied from mg18 ──────────────────────────────────────────────
# A thermally throttled GPU produces a valid values reading and an invalid
# timing one, and this run is entirely a timing run.
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


# ── the model, and the launch machinery ───────────────────────────────────────
def build_model(pin_devices=None):
    """mg19's construction at this run's cell, unchanged.

    The memory preflight is skipped and that is recorded.  The preflight
    prices a whole reconstruction, and this run allocates a few values blocks
    and one sinogram view batch; mg18's one-device staging arm skips it for the
    same reason.
    """
    import numpy as np

    import mbirtorch

    shape = tuple(cell())
    # Half a turn is a full parallel-beam scan: the second half repeats the
    # first, reflected.
    angles = np.linspace(0, np.pi, shape[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(shape, angles)
    if pin_devices is not None:
        model.configure_devices(devices=list(pin_devices))
    model.set_params(no_warning=True, verbose=0)
    model.skip_memory_preflight = True
    model._apply_device_policy()
    return model


def _seeded_values(torch_module, model, num_pixels, columns, row_stride=None):
    """The values block one arm projects, seeded so every arm sees the same
    numbers.

    ``row_stride`` allocates a wider buffer and fills only its first
    ``columns`` columns.  That is how the strided arms get 1008-wide memory
    under 512-wide work, and how w504_alloc1008 gets its untouched second half.
    """
    generator = torch_module.Generator(device=model.torch_device)
    generator.manual_seed(VALUES_SEED)
    width = int(row_stride or columns)
    block = torch_module.zeros((int(num_pixels), width), dtype=torch_module.float32,
                               device=model.torch_device)
    block[:, :int(columns)] = torch_module.rand(
        (int(num_pixels), int(columns)), generator=generator,
        dtype=torch_module.float32, device=model.torch_device)
    return block


def strided_forward_launch(values_buf, pixel_indices, view_params_batch, args,
                           num_cols, col_offset, grid_cols):
    """The library wrapper's launch, copied here, with the grid's column axis
    truncated and the pointers offset.

    THIS IS A COPY OF ``_parallel_forward_view_batch_triton`` (triton_parallel.py
    lines 403 to 470) and it exists because the shipped wrapper cannot express
    what this arm needs.  The wrapper packs its values argument with
    ``.contiguous()``, and the kernel takes no separate stride argument for the
    values block: its ``num_cols`` is at once the values row stride, the mask
    that sets the logical width, and the output row stride.  So a values tensor
    whose row stride exceeds its column count cannot reach the shipped kernel.

    What this function changes against the wrapper is the LAUNCH and nothing
    else.  ``num_cols`` is the buffer's true row stride, so the kernel strides
    correctly through both the values and the output.  The grid's column axis
    is truncated to ``grid_cols`` blocks, so the launch does
    ``grid_cols * BLOCK_R`` columns of work instead of the whole width.  The
    values and output pointers are offset by ``col_offset`` columns, so the
    truncated launch can be placed anywhere along the width.  Everything else
    -- the hfan build, the contract tensors, the tile sizes, the zeroed
    channel-major output, the compile guard, the device bracket, the returned
    permutation -- is the wrapper's.

    The whole (views, stride, channels) output is returned, permuted as the
    wrapper permutes it.  Only the columns this launch covered hold results;
    the rest are the zeros the allocation started with.  The caller knows which
    columns those are and slices them out.
    """
    import torch

    from mbirtorch.parallel_beam import _parallel_hfan_math
    from mbirtorch.projectors import compile_serialized
    from mbirtorch.triton_cone import _COMPILED_LAUNCH_KEYS, _tile_size
    from mbirtorch.triton_parallel import (PARALLEL_FWD_BLOCK_P,
                                           PARALLEL_FWD_BLOCK_R,
                                           PARALLEL_FWD_MIN_TILE,
                                           PARALLEL_FWD_NUM_STAGES,
                                           PARALLEL_FWD_NUM_WARPS,
                                           _parallel_forward_kernel)

    num_cols = int(num_cols)
    col_offset = int(col_offset)
    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        pixel_indices, view_params_batch, args["num_rows"], args["num_cols"],
        args["num_channels"], args["delta_det_channel"],
        args["det_channel_offset"], args["delta_voxel"],
        args["delta_voxel_row"])
    num_views, num_pixels = n_p.shape
    contract = [t.contiguous() for t in (n_p, centers)]
    contract += [t.reshape(num_views).contiguous()
                 for t in (w_p_c, weight_scale)]
    num_channels = int(args["num_channels"])
    out = torch.zeros((num_views, num_channels, num_cols), dtype=torch.float32,
                      device=values_buf.device)

    block_p = _tile_size(PARALLEL_FWD_BLOCK_P, num_pixels, PARALLEL_FWD_MIN_TILE)
    block_r = _tile_size(PARALLEL_FWD_BLOCK_R, num_cols, PARALLEL_FWD_MIN_TILE)
    cols_launched = int(grid_cols) if grid_cols else -(-num_cols // block_r)
    grid = (-(-num_pixels // block_p), cols_launched, num_views)
    launch_key = ("mg20_pfwd_strided", values_buf.device.index,
                  int(args["psf_radius"]), block_p, block_r, int(num_views),
                  int(num_pixels), num_channels, num_cols, cols_launched)
    first_launch = launch_key not in _COMPILED_LAUNCH_KEYS
    guard = compile_serialized() if first_launch else contextlib.nullcontext()
    values_view = values_buf[:, col_offset:] if col_offset else values_buf
    out_view = out[:, :, col_offset:] if col_offset else out
    with torch.cuda.device(values_buf.device), guard:
        _parallel_forward_kernel[grid](
            *contract, values_view, out_view,
            int(num_pixels), num_channels, num_cols,
            num_channels * num_cols,
            PSF_RADIUS=int(args["psf_radius"]), BLOCK_P=block_p,
            BLOCK_R=block_r, num_warps=PARALLEL_FWD_NUM_WARPS,
            num_stages=PARALLEL_FWD_NUM_STAGES)
    _COMPILED_LAUNCH_KEYS.add(launch_key)
    return out.permute(0, 2, 1)


def launch_geometry(num_pixels, columns, view_batch, grid_cols=None):
    """The grid and tile the kernel will use, derived here from the library's
    own rule so the tables can print it without a profiler.

    ``_tile_size`` shrinks the column tile below its 128 cap only when the
    width is smaller, which is why width 63 runs a 64-wide tile and every
    other width in this run runs a 128-wide one.
    """
    try:
        from mbirtorch.triton_cone import _tile_size
        from mbirtorch.triton_parallel import (PARALLEL_FWD_BLOCK_P,
                                               PARALLEL_FWD_BLOCK_R,
                                               PARALLEL_FWD_MIN_TILE)
    except Exception:                                             # noqa: BLE001
        return dict(available=False)
    block_p = _tile_size(PARALLEL_FWD_BLOCK_P, int(num_pixels),
                         PARALLEL_FWD_MIN_TILE)
    block_r = _tile_size(PARALLEL_FWD_BLOCK_R, int(columns),
                         PARALLEL_FWD_MIN_TILE)
    cols = int(grid_cols) if grid_cols else -(-int(columns) // block_r)
    grid = (-(-int(num_pixels) // block_p), cols, int(view_batch))
    return dict(available=True, block_p=block_p, block_r=block_r,
                grid=list(grid), blocks=int(grid[0]) * int(grid[1]) * int(grid[2]),
                columns_of_work=cols * block_r,
                num_cols_divisible_by_16=(int(columns) % 16 == 0))


def kernel_build_record():
    """What Triton compiled, read from its own cache: registers, spills and
    shared memory per compiled kernel.

    These are the numbers occupancy per block is computed from, so they answer
    part of question 1 without a profiler.  The attribute names have moved
    between Triton versions, so every lookup here is defensive and a miss is
    recorded rather than raised.
    """
    try:
        from mbirtorch.triton_parallel import _parallel_forward_kernel as fwd
    except Exception as exc:                                      # noqa: BLE001
        return dict(available=False, reason=f"{type(exc).__name__}: {exc}")
    entries, names = [], []
    caches = []
    for attr in ("cache", "device_caches"):
        holder = getattr(fwd, attr, None)
        if isinstance(holder, dict):
            caches.append(holder)
    for holder in caches:
        for value in holder.values():
            group = value.values() if isinstance(value, dict) else [value]
            for compiled in group:
                if compiled is None or isinstance(compiled, (int, str)):
                    continue
                record = {}
                for field in ("n_regs", "n_spills", "shared", "num_warps",
                              "name"):
                    got = getattr(compiled, field, None)
                    if got is None:
                        meta = getattr(compiled, "metadata", None)
                        got = getattr(meta, field, None)
                        if got is None and isinstance(meta, dict):
                            got = meta.get(field)
                    if got is not None and not isinstance(got, (int, float, str)):
                        got = str(got)
                    record[field] = got
                if record.get("name"):
                    names.append(str(record["name"]))
                entries.append(record)
    return dict(available=bool(entries), entries=entries[:24],
                names=sorted(set(names)),
                python_name=getattr(fwd, "__name__", None))


def wrapper_contract_record():
    """What the shipped wrapper and kernel say about a strided values tensor.

    Read at runtime rather than asserted from a reading taken once: the arm
    plan depends on this answer, and a library that grew a stride argument
    should say so on the row instead of being silently ablated the old way.
    """
    import inspect

    record = dict(wrapper_packs_values=None, kernel_arg_names=None,
                  kernel_has_values_row_stride=None, source_lines=[])
    try:
        from mbirtorch.triton_parallel import (
            _parallel_forward_kernel as kernel,
            _parallel_forward_view_batch_triton as wrapper)
    except Exception as exc:                                      # noqa: BLE001
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    try:
        source = inspect.getsource(wrapper)
        record["wrapper_packs_values"] = "values = values.contiguous()" in source
        record["source_lines"] = [line.strip() for line in source.splitlines()
                                  if "contiguous()" in line
                                  and "values" in line][:4]
    except Exception:                                             # noqa: BLE001
        pass
    names = getattr(kernel, "arg_names", None)
    if names is None:
        # Without triton installed the kernel is the undecorated python
        # function (triton_cone._jit), so its own code object carries the
        # argument names.  The smoke reads the contract this way.
        inner = getattr(kernel, "fn", kernel)
        code = getattr(inner, "__code__", None)
        if code is not None:
            names = list(code.co_varnames[:code.co_argcount])
    if names is not None:
        names = [str(n) for n in names]
        record["kernel_arg_names"] = names
        record["kernel_has_values_row_stride"] = any(
            "stride" in n and "out" not in n for n in names)
    return record


# ── the values comparison ─────────────────────────────────────────────────────
def compare_blocks(torch_module, candidate, reference, gate):
    """max|candidate - reference| / max|reference|, walked in view slabs.

    The maximum of maxima is the maximum, so slabbing is exact.  It is taken on
    the device in float32: a maximum carries none of the summation error a norm
    would, and moving two half-gigabyte blocks to the host to promote them
    would cost more than the comparison.
    """
    if tuple(candidate.shape) != tuple(reference.shape):
        return dict(ok=False, rel=None, gate=gate,
                    reason=f"shape {list(candidate.shape)} is not the "
                           f"reference's {list(reference.shape)}")
    max_diff, max_ref = 0.0, 0.0
    step = max(1, int(reference.shape[0]) // 8)
    for start in range(0, int(reference.shape[0]), step):
        a = candidate[start:start + step]
        b = reference[start:start + step]
        max_ref = max(max_ref, float(b.abs().max()))
        max_diff = max(max_diff, float((a - b).abs().max()))
    if max_ref <= 0.0:
        return dict(ok=False, rel=None, gate=gate,
                    reason="the reference is all zeros, so a relative "
                           "comparison has no denominator")
    rel = max_diff / max_ref
    return dict(ok=bool(rel <= gate), rel=rel, gate=gate,
                max_abs_diff=max_diff, max_abs_ref=max_ref)


# ── timing ────────────────────────────────────────────────────────────────────
def timed_launches(torch_module, cuda, calls, repeats):
    """Milliseconds per launch, one entry per (repeat, piece).

    On CUDA each launch gets its own event pair and the device is synchronized
    once at the end, so the reading is device time and the host's queueing does
    not enter it.  Each call's output is dropped as soon as it is timed, so
    memory stays flat across the repeats.
    """
    for _ in range(WARM_PASSES):
        for call in calls:
            out = call()
            out = None
    spans = []
    if not cuda:
        for _ in range(repeats):
            for index, call in enumerate(calls):
                start = time.perf_counter()
                out = call()
                spans.append((index, (time.perf_counter() - start) * 1e3))
                out = None
        return spans
    torch_module.cuda.synchronize()
    pairs = []
    for _ in range(repeats):
        for index, call in enumerate(calls):
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record()
            out = call()
            end.record()
            pairs.append((index, start, end))
            out = None
    torch_module.cuda.synchronize()
    return [(index, start.elapsed_time(end)) for index, start, end in pairs]


def summarize_spans(spans, pieces):
    """Per-piece-width medians and spreads from one pixel point's launches.

    Widths are kept apart because one arm may launch two of them: w512_496
    cuts 1008 into a 512 and a 496, and averaging those two would report a
    width that neither launch had.
    """
    by_width, strides = {}, {}
    for index, ms in spans:
        width = int(pieces[index]["work"])
        by_width.setdefault(width, []).append(float(ms))
        strides[width] = int(pieces[index]["stride"])
    out = {}
    for width, values in by_width.items():
        out[str(width)] = dict(
            width=width, stride=strides[width], launches=len(values),
            median_ms=statistics.median(values), min_ms=min(values),
            max_ms=max(values), spread_ms=max(values) - min(values),
            ms_per_slice=statistics.median(values) / width)
    return out


# ── one timing arm ────────────────────────────────────────────────────────────
def run_arm(cfg):
    """One arm, in its own process.

    A fresh process per arm is not tidiness.  Triton caches its compiled
    kernels at module level for the life of a process, and so does the
    library's first-launch compile guard, so two arms in one process would
    share both and the second would be timed under the first's cache.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    health = [sample_gpu_health()]
    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  view_batch=VIEW_BATCH, timed_repeats=TIMED_REPEATS,
                  values_seed=VALUES_SEED,
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"),
                  invalid_reasons=[])
    if cfg.get("skip"):
        result["skipped"] = cfg["skip"]
        result["gpu_health"] = [g for snap in health for g in snap]
        return result

    model = build_model(pin_devices=None if cuda else [DEVICE])
    result["recon_shape"] = [int(s) for s in model.get_params("recon_shape")]
    result["num_pixels_full"] = int(model.full_index_count())
    if not SMOKE:
        result["recon_shape_ok"] = (tuple(result["recon_shape"])
                                    == RECON_SHAPE_EXPECTED)
        result["num_pixels_ok"] = (result["num_pixels_full"]
                                   == NUM_PIXELS_EXPECTED)
    devices = [str(d) for d in model.sino_placement.devices]
    result["realized_devices"] = devices
    if len(devices) != 1:
        result["invalid_reasons"].append(
            f"the layout settled on {len(devices)} devices; every launch in "
            "this run is a one-device launch")

    # WHICH BODY THE MODEL BINDS.  The whole probe is about the Triton forward
    # kernel, so an arm that quietly ran the torch body would produce numbers
    # that look like measurements and answer a different question.
    fwd_body, back_body = model._view_batch_bodies()
    result["forward_body"] = fwd_body.__name__
    result["back_body"] = back_body.__name__
    triton_bound = fwd_body.__name__.endswith("_triton")
    result["triton_forward_bound"] = triton_bound
    if cuda and not triton_bound:
        result["invalid_reasons"].append(
            f"the model bound {fwd_body.__name__} for the forward projection, "
            "not the Triton kernel; every arm of this run measures that kernel "
            "and this one measured something else")
    result["wrapper_contract"] = wrapper_contract_record()

    args = model._view_batch_args()
    result["psf_radius"] = int(args["psf_radius"])
    result["psf_width"] = 2 * int(args["psf_radius"]) + 1
    pf = model.projector_functions
    total = int(cfg["num_slices"])
    full_pixels = int(model.full_index_count())
    driver_batch, bytes_per_view = pf.view_batch_charge(
        fwd_body, full_pixels, total, args)
    result["driver_view_batch"] = int(driver_batch)
    result["driver_bytes_per_view"] = int(bytes_per_view)
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    result["view_batch_used"] = view_batch
    view_params = pf._view_params_per_dev[0][:view_batch]

    idx_full = model.full_indices_device()
    pieces = cfg["pieces"]
    layout = cfg["layout"]
    stride = int(pieces[0]["stride"])
    result["piece_widths"] = [int(p["work"]) for p in pieces]
    result["layout_stride"] = stride
    result["launch_geometry"] = launch_geometry(
        full_pixels, stride, view_batch,
        grid_cols=STRIDE_GRID_COLS if layout == "stride" else None)

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # THE VALUES BLOCKS.  One per piece, built before any timing, so no arm
    # pays a copy inside a timed launch.  mg10's arms paid exactly that copy
    # (the wrapper packs a strided piece) and measured it at 0.2 to 1.3 ms
    # against 21 to 41 ms launches; here it is outside the bracket instead.
    reference_values = _seeded_values(torch, model, full_pixels, total)
    piece_values = []
    if layout == "stride":
        # Both strided launches read the same buffer at different offsets, so
        # there is one buffer.  Its first ``total`` columns are copied from the
        # reference block, which is what makes the two arms' values identical
        # by construction rather than by both drawing the same seed.
        shared = torch.zeros((full_pixels, stride), dtype=torch.float32,
                             device=model.torch_device)
        shared[:, :total].copy_(reference_values)
        piece_values = [shared for _ in pieces]
    else:
        for piece in pieces:
            if layout == "contig":
                block = reference_values[
                    :, piece["start"]:piece["start"] + piece["work"]].contiguous()
            else:
                # A contiguous block of the piece's width, living inside an
                # allocation of the FULL width whose second half is never
                # touched.
                buffer = torch.zeros((full_pixels, total), dtype=torch.float32,
                                     device=model.torch_device)
                block = buffer.view(-1)[:full_pixels * piece["work"]].view(
                    full_pixels, piece["work"])
                block.copy_(reference_values[:, piece["start"]:
                                             piece["start"] + piece["work"]])
                block._mg20_backing = buffer
            piece_values.append(block)

    def make_call(piece, block, pixel_indices):
        if layout == "stride":
            return lambda: strided_forward_launch(
                block, pixel_indices, view_params, args,
                num_cols=piece["stride"], col_offset=piece["start"],
                grid_cols=STRIDE_GRID_COLS)
        return lambda: fwd_body(block, pixel_indices, view_params,
                                slice_start=0, plan=None, **args)

    # ── the reference, and this arm's values check ──────────────────────────
    # The reference is one full-width launch through the SHIPPED wrapper, taken
    # here rather than staged: an arm is cheap, and a reference computed in the
    # arm's own process cannot drift from it.
    reference = fwd_body(reference_values, idx_full, view_params,
                         slice_start=0, plan=None, **args)
    assembled = []
    for piece, block in zip(pieces, piece_values):
        out = make_call(piece, block, idx_full)()
        low, high = piece["keep_local"]
        assembled.append(out[:, low:high, :].clone())
        out = None
    assembled = (assembled[0] if len(assembled) == 1
                 else torch.cat(assembled, dim=1))
    result["assembled_shape"] = list(assembled.shape)
    result["values"] = compare_blocks(torch, assembled, reference,
                                      VALUES_GATE_REL)
    assembled = None

    # The copy of the wrapper's launch code, run at the FULL width against the
    # wrapper itself.  It separates two things the arm's own values check
    # cannot: whether the copied launch is faithful, and whether the grid
    # truncation is placed correctly.
    if layout == "stride" and cuda:
        copy_out = strided_forward_launch(
            piece_values[0], idx_full, view_params, args,
            num_cols=piece_values[0].shape[1], col_offset=0, grid_cols=None)
        result["copy_fidelity"] = compare_blocks(
            torch, copy_out[:, :total, :], reference, VALUES_GATE_REL)
        copy_out = None
    # The reference block and the reference sinogram are both dead once the
    # comparison is taken, and together they are several gigabytes.  Dropped
    # before the timing starts so the timed launches allocate into a quiet
    # device.  Each piece holds its own values, so nothing here is still in use.
    reference = None
    reference_values = None
    if cuda:
        torch.cuda.empty_cache()

    if not result["values"].get("ok"):
        result["values_failed"] = True
        result["timing_skipped_reason"] = (
            f'the values gate failed at rel {result["values"].get("rel")} '
            f"against {VALUES_GATE_REL:g}; timing a launch set that computes a "
            "different sinogram would measure nothing")
        health.append(sample_gpu_health())
        result["gpu_health"] = [g for snap in health for g in snap]
        result["gpu_hot"] = row_is_hot(result["gpu_health"])
        return result

    # ── the two named per-call costs, so a reader can subtract them ─────────
    # The horizontal-fan contract is rebuilt on every body call and its cost
    # does not depend on the width, so it is a fixed offset on every launch in
    # every arm.  The output slab is zeroed on every call and its cost is
    # proportional to the width.
    result["fixed_costs"] = measure_fixed_costs(
        torch, cuda, model, idx_full, view_params, args, stride)

    # ── the pixel ladder ────────────────────────────────────────────────────
    ladder = []
    for divisor, weight in PIXEL_LADDER:
        want = max(1, -(-full_pixels // divisor))
        step = max(1, full_pixels // want)
        subset = idx_full[::step][:want].contiguous()
        realized = int(subset.shape[0])
        calls = [make_call(piece, block[:realized], subset)
                 for piece, block in zip(pieces, piece_values)]
        spans = timed_launches(torch, cuda, calls, TIMED_REPEATS)
        entry = dict(divisor=divisor, mg10_launches=weight,
                     num_pixels=realized, by_width=summarize_spans(spans, pieces))
        entry["per_launch_median_ms"] = statistics.median(
            [ms for _index, ms in spans])
        ladder.append(entry)
        calls = None
    result["ladder"] = ladder

    full_point = ladder[-1]
    result["full_pixel_point"] = full_point
    result["by_width"] = full_point["by_width"]
    # Section 1.9's quantity, rebuilt: the mean per launch over mg10's mixture
    # of pixel counts.  It is reported and not gated -- see the module
    # docstring on what the exit code tests instead.
    weight_sum = sum(w for _d, w in PIXEL_LADDER)
    result["mixture_mean_per_launch_ms"] = sum(
        entry["per_launch_median_ms"] * entry["mg10_launches"]
        for entry in ladder) / weight_sum

    result["kernel_build"] = kernel_build_record()
    if cuda:
        result["peak_bytes"] = int(torch.cuda.max_memory_allocated())
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def measure_fixed_costs(torch_module, cuda, model, pixel_indices, view_params,
                        args, columns):
    """The two per-call costs that are not the kernel: the horizontal-fan
    contract build and the zeroed output slab.

    Both sit inside every timed launch because both sit inside the library's
    body.  The contract build does not depend on the values width, so it is the
    same offset on every arm.  The slab is proportional to the width.  Measured
    once per arm, outside every timed region, so a reader can subtract either.
    """
    from mbirtorch.parallel_beam import _parallel_hfan_math

    def hfan():
        return _parallel_hfan_math(
            pixel_indices, view_params, args["num_rows"], args["num_cols"],
            args["num_channels"], args["delta_det_channel"],
            args["det_channel_offset"], args["delta_voxel"],
            args["delta_voxel_row"])

    def slab():
        return torch_module.zeros(
            (int(view_params.shape[0]), int(args["num_channels"]), int(columns)),
            dtype=torch_module.float32, device=model.torch_device)

    out = {}
    for name, call in (("hfan_ms", hfan), ("out_slab_ms", slab)):
        spans = timed_launches(torch_module, cuda, [call], 3)
        out[name] = statistics.median([ms for _index, ms in spans])
    out["out_slab_bytes"] = int(view_params.shape[0]) * int(
        args["num_channels"]) * int(columns) * 4
    return out


# ── the single-launch worker the profiler drives ──────────────────────────────
def one_launch(cfg):
    """Build the model, build one arm's first piece, launch it a few times,
    exit.

    Kept as small as it can be because ncu profiles what it is given: a whole
    sweep under the profiler would be a sweep of replayed kernels.

    THE FORWARD BODY IS IMPORTED DIRECTLY here rather than taken from
    ``model._view_batch_bodies()``.  That selection runs the kernel's
    availability self-check, which launches the same forward kernel once on a
    tiny problem, and that launch would be the first thing ncu's kernel filter
    matched.  Importing the body makes this worker's launch count exactly the
    number below, so ``--launch-skip`` lands where it is aimed.  Which body the
    model would have bound is witnessed by every timing arm.
    """
    import torch

    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    result = dict(cfg, mode="one_launch", device=DEVICE, cuda=cuda,
                  launches=int(cfg.get("launches", NCU_LAUNCHES)))
    spec = None
    for candidate in ARM_TABLE:
        if candidate["arm"] == cfg["arm"]:
            spec = candidate
    if spec is None:
        raise ValueError(f'unknown arm {cfg["arm"]!r}')
    pieces = arm_pieces(spec)
    piece = pieces[0]

    from mbirtorch.triton_parallel import _parallel_forward_view_batch_triton

    model = build_model(pin_devices=None if cuda else [DEVICE])
    args = model._view_batch_args()
    pf = model.projector_functions
    view_batch = min(VIEW_BATCH, int(cell()[0]))
    view_params = pf._view_params_per_dev[0][:view_batch]
    full_pixels = int(model.full_index_count())
    want = max(1, full_pixels // max(1, int(cfg.get("pixel_div", 1))))
    step = max(1, full_pixels // want)
    idx = model.full_indices_device()[::step][:want].contiguous()
    realized = int(idx.shape[0])
    result["num_pixels"] = realized
    result["piece"] = piece
    result["launch_geometry"] = launch_geometry(
        realized, piece["stride"], view_batch,
        grid_cols=STRIDE_GRID_COLS if spec["layout"] == "stride" else None)

    if spec["layout"] == "stride":
        block = _seeded_values(torch, model, realized, int(cfg["num_slices"]),
                               row_stride=piece["stride"])

        def call():
            return strided_forward_launch(
                block, idx, view_params, args, num_cols=piece["stride"],
                col_offset=piece["start"], grid_cols=STRIDE_GRID_COLS)
    else:
        block = _seeded_values(torch, model, realized, piece["work"])

        def call():
            return _parallel_forward_view_batch_triton(
                block, idx, view_params, slice_start=0, plan=None, **args)

    names = []
    for index in range(result["launches"]):
        last = index == result["launches"] - 1
        if last and cfg.get("profile_names") and cuda:
            # torch.profiler names the kernel exactly as the CUDA runtime does,
            # which is what ncu's --kernel-name filter matches against.  Run on
            # the last launch only, and never under ncu.
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                out = call()
                torch.cuda.synchronize()
            for event in prof.key_averages():
                device_time = getattr(event, "self_device_time_total", None)
                if device_time is None:
                    device_time = getattr(event, "self_cuda_time_total", 0.0)
                if device_time:
                    names.append(dict(name=str(event.key),
                                      device_time_us=float(device_time)))
        else:
            out = call()
        out = None
    if cuda:
        torch.cuda.synchronize()
    names.sort(key=lambda entry: -entry["device_time_us"])
    result["profiler_kernel_names"] = names[:12]
    result["kernel_build"] = kernel_build_record()
    return result


def trivial_kernel():
    """One tiny CUDA kernel, for the profiler's permission probe.  Nothing here
    depends on mbirtorch."""
    import torch

    if not torch.cuda.is_available():
        return dict(mode="trivial_kernel", cuda=False)
    x = torch.ones(1 << 16, device="cuda")
    total = float((x * 2.0).sum())
    torch.cuda.synchronize()
    return dict(mode="trivial_kernel", cuda=True, checksum=total)


# ── the profiler leg ──────────────────────────────────────────────────────────
def _run(cmd, timeout, env=None):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return dict(returncode=proc.returncode, stdout=proc.stdout,
                    stderr=proc.stderr, wall_s=time.perf_counter() - start,
                    timed_out=False)
    except subprocess.TimeoutExpired:
        return dict(returncode=None, stdout="", stderr="",
                    wall_s=time.perf_counter() - start, timed_out=True)
    except FileNotFoundError as exc:
        return dict(returncode=None, stdout="", stderr=str(exc),
                    wall_s=time.perf_counter() - start, timed_out=False,
                    missing=True)


def _worker_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return None


def parse_ncu_csv(text):
    """One dict of metric name to value per profiled kernel.

    Two layouts are accepted because the two ncu pages disagree and the format
    has moved between releases.  The wide layout puts one metric per column and
    one kernel per row.  The long layout puts one metric per ROW, with the
    metric name and its value in their own columns.  A unit row directly under
    the header is skipped when it appears.  Numbers may carry thousands
    separators, which are stripped.
    """
    import csv as csv_module

    rows = list(csv_module.reader(text.splitlines()))

    def number(text_value):
        cleaned = str(text_value).replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return cleaned

    def parse_from(header_index):
        header = [cell.strip() for cell in rows[header_index]]
        body = rows[header_index + 1:]
        kernel_at = header.index("Kernel Name")
        if "Metric Name" in header and "Metric Value" in header:
            name_at = header.index("Metric Name")
            value_at = header.index("Metric Value")
            unit_at = (header.index("Metric Unit") if "Metric Unit" in header
                       else None)
            by_kernel = {}
            for row in body:
                if len(row) <= max(name_at, value_at, kernel_at):
                    continue
                key = row[kernel_at].strip()
                entry = by_kernel.setdefault(key, dict(kernel=key, metrics={},
                                                       units={}))
                entry["metrics"][row[name_at].strip()] = number(row[value_at])
                if unit_at is not None and len(row) > unit_at:
                    entry["units"][row[name_at].strip()] = row[unit_at].strip()
            return list(by_kernel.values())
        out, units = [], {}
        for row in body:
            if len(row) < len(header):
                continue
            values = [number(cell) for cell in row]
            if all(isinstance(value, str) for value in values):
                # The unit row directly under the header.  It is the only place
                # the wide layout says what its numbers are in, so it is kept
                # rather than skipped.
                if not units:
                    units = {name: str(value).strip()
                             for name, value in zip(header, values)
                             if name and str(value).strip()}
                continue
            entry = dict(kernel=row[kernel_at].strip(), metrics={},
                         units=dict(units))
            for name, value in zip(header, values):
                if name and name != "Kernel Name":
                    entry["metrics"][name] = value
            out.append(entry)
        return out

    # Several rows can carry "Kernel Name": ncu prints a header per page, and
    # some releases repeat it.  Every candidate is parsed and the parse that
    # recovered the most kernels, and then the most numbers, wins.  The choice
    # of header row is therefore not a guess about the release.
    best, best_score = [], (0, 0)
    for index, row in enumerate(rows):
        if not any(cell.strip() == "Kernel Name" for cell in row):
            continue
        try:
            parsed = parse_from(index)
        except (ValueError, IndexError):
            continue
        scored = [entry for entry in parsed
                  if any(isinstance(value, (int, float))
                         for value in entry["metrics"].values())]
        numbers = sum(1 for entry in scored for value in entry["metrics"].values()
                      if isinstance(value, (int, float)))
        if (len(scored), numbers) > best_score:
            best, best_score = scored, (len(scored), numbers)
    return best


def ncu_leg(plan, torch_python, results_dir):
    """The optional counter leg: probe, discover, profile.

    Every step records what it saw.  A refusal, a missing ncu, an unmatched
    kernel name and a timeout are all findings on the record, and none of them
    changes the exit code.
    """
    leg = dict(attempted=True, enabled=NCU_ENABLED, pixel_div=NCU_PIXEL_DIV,
               metrics=list(NCU_METRICS), arms={})
    if not NCU_ENABLED:
        leg.update(attempted=False, reason="MG20_NCU=0")
        return leg
    if DEVICE != "cuda":
        leg.update(attempted=False,
                   reason="this run is on the CPU, where there are no GPU "
                          "performance counters to read")
        return leg
    ncu = shutil.which("ncu")
    leg["ncu_path"] = ncu
    if ncu is None:
        leg.update(attempted=False,
                   reason="ncu is not on PATH; the sbatch loads the cuda "
                          "module, which is where Nsight Compute lives")
        return leg
    version = _run([ncu, "--version"], 60)
    leg["ncu_version"] = (version["stdout"].strip().splitlines()[:2]
                          if version["stdout"] else version["stderr"][:200])

    # ── step 1: the permission probe ────────────────────────────────────────
    probe = _run([ncu, "--launch-count", "1", "--metrics",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  torch_python, "-u", os.path.abspath(__file__),
                  "--trivial-kernel"], NCU_PROBE_TIMEOUT_S)
    blob = (probe["stdout"] or "") + (probe["stderr"] or "")
    refused = any(marker.lower() in blob.lower()
                  for marker in NCU_PERMISSION_MARKERS)
    # The probe's own worker prints a result line, so a probe that produced no
    # metric can be told apart from one whose python never ran at all.  The two
    # have different remedies and reporting them as one would send a reader to
    # the wrong place.
    worker_ran = "__RESULT__" in blob
    empty = "sm__warps_active" not in blob
    leg["permission_probe"] = dict(
        returncode=probe["returncode"], timed_out=probe["timed_out"],
        refused=refused, profile_empty=empty, worker_ran=worker_ran,
        message=blob.strip()[-800:])
    leg["profiler_permitted"] = not (refused or empty or probe["timed_out"])
    if not leg["profiler_permitted"]:
        if refused:
            leg["reason"] = "the driver refused performance counters to this user"
        elif probe["timed_out"]:
            leg["reason"] = (f"the permission probe did not finish within "
                             f"{NCU_PROBE_TIMEOUT_S} s")
        elif not worker_ran:
            leg["reason"] = ("the probe's own python never ran, so this says "
                             "nothing about counter permission; read the "
                             "message on the row")
        else:
            leg["reason"] = ("the probe ran and produced no metric, so the "
                             "counters are unavailable")
        return leg

    # ── step 2: the kernel name, discovered at runtime ──────────────────────
    discover_arm = next((entry["arm"] for entry in plan
                         if entry["arm"] in NCU_ARMS and not entry.get("skip")),
                        None)
    if discover_arm is None:
        leg.update(reason="no profiled arm is in this run's plan")
        return leg
    cfg = dict(arm=discover_arm, num_slices=num_slices(),
               pixel_div=NCU_PIXEL_DIV, profile_names=True, launches=2)
    plain = _run([torch_python, "-u", os.path.abspath(__file__),
                  "--one-launch", json.dumps(cfg)], NCU_TIMEOUT_S,
                 env=arm_env())
    row = _worker_result(plain["stdout"] or "")
    leg["discovery"] = dict(arm=discover_arm, returncode=plain["returncode"],
                            error=(None if row else
                                   (plain["stderr"] or "")[-1500:]))
    candidates = []
    if row:
        leg["discovery"]["profiler_kernel_names"] = row.get(
            "profiler_kernel_names")
        leg["discovery"]["triton_names"] = (row.get("kernel_build") or {}).get(
            "names")
        for entry in row.get("profiler_kernel_names") or []:
            candidates.append(str(entry.get("name", "")))
        candidates.extend((row.get("kernel_build") or {}).get("names") or [])
    match = None
    for name in candidates:
        if "parallel_forward" in name.lower():
            match = name
            break
    leg["kernel_name"] = match
    if match is None:
        leg["reason"] = (
            "no kernel name reported by torch.profiler or by Triton's own "
            "cache contains 'parallel_forward'; the names seen are on the row, "
            "and a filter is not guessed")
        return leg
    # A bare name is used as a regular expression, so its regex characters are
    # escaped.  ncu matches the expression against the kernel's mangled name.
    pattern = "regex:" + re.escape(match)
    leg["kernel_name_filter"] = pattern

    # ── step 3: one warm launch per arm ─────────────────────────────────────
    leg_start = time.perf_counter()
    leg["budget_s"] = NCU_LEG_BUDGET_S
    for entry in plan:
        arm = entry["arm"]
        if arm not in NCU_ARMS or entry.get("skip"):
            continue
        spent = time.perf_counter() - leg_start
        if spent > NCU_LEG_BUDGET_S:
            leg["arms"][arm] = dict(
                arm=arm, reason=f"the profiler leg's {NCU_LEG_BUDGET_S} s "
                                f"budget was already spent ({spent:.0f} s); "
                                "this arm was not profiled")
            continue
        cfg = dict(arm=arm, num_slices=num_slices(), pixel_div=NCU_PIXEL_DIV,
                   launches=NCU_LAUNCHES)
        record = dict(arm=arm, attempts=[])
        # The warm launch first, then two fallbacks.  A skip of zero profiles
        # the arm's FIRST launch, which is warm in every sense except the
        # kernel cache, and is worth having if the aimed skip matched nothing.
        # The details page is tried last because the raw page is the one whose
        # CSV is wide; both are parsed the same way here.
        for page, skip in (("raw", NCU_LAUNCHES - 1), ("raw", 0),
                           ("details", NCU_LAUNCHES - 1)):
            cmd = [ncu, "--csv", "--page", page, "--target-processes", "all",
                   "--kernel-name", pattern, "--launch-skip", str(skip),
                   "--launch-count", "1", "--metrics", ",".join(NCU_METRICS),
                   torch_python, "-u", os.path.abspath(__file__),
                   "--one-launch", json.dumps(cfg)]
            got = _run(cmd, NCU_TIMEOUT_S, env=arm_env())
            log_path = os.path.join(
                results_dir, f"mg20_ncu_{arm}_{page}_skip{skip}.log")
            with open(log_path, "w") as sink:
                sink.write(" ".join(cmd) + "\n\n")
                sink.write(got["stdout"] or "")
                sink.write("\n----- stderr -----\n")
                sink.write(got["stderr"] or "")
            parsed = parse_ncu_csv(got["stdout"] or "")
            record["attempts"].append(dict(
                page=page, launch_skip=skip, returncode=got["returncode"],
                timed_out=got["timed_out"], wall_s=got["wall_s"],
                kernels=len(parsed), log=log_path))
            if parsed:
                record.update(page=page, launch_skip=skip, kernels=parsed,
                              wall_s=got["wall_s"], log=log_path)
                break
            if got["timed_out"]:
                # Retrying a timeout with a different page would spend the same
                # minutes again for the same reason.  MG20_NCU_PIXEL_DIV=4 is
                # the knob that makes the profiled kernel smaller.
                record["reason"] = (
                    f'the profile did not finish within {NCU_TIMEOUT_S} s; '
                    "re-run with MG20_NCU_PIXEL_DIV=4 to profile a quarter of "
                    "the pixels")
                break
        if "kernels" not in record and "reason" not in record:
            record["reason"] = ("no kernel matched the filter, or the profile "
                                "was empty; the raw output is in the logs above")
        leg["arms"][arm] = record
    return leg


# ── the runner ────────────────────────────────────────────────────────────────
def arm_env():
    """The environment an arm runs under, set explicitly so nothing is
    inherited.

    Three variables are popped and one is set.  The device pin and the
    calibration mode would both change what an arm measures, and this run pins
    neither: it uses the single device the job allocates.  The Triton kill
    switch is set to the shipped value, because the kernel it would turn off is
    the whole subject of this run.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def _spawn(cfg):
    """Run one arm in a fresh interpreter."""
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env())
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    row = _worker_result(proc.stdout)
    if row is None:
        return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                    subprocess_wall_s=wall)
    row["subprocess_wall_s"] = wall
    return row


def _dry_run(plan):
    print(f"mg20 width mechanism: {len(plan)} arms, device {DEVICE}, "
          f"cell {tuple(cell())}, {TIMED_REPEATS} timed repeats per pixel point")
    print(f"  results and ncu logs -> {RESULTS_DIR}")
    print(f"  values gate {VALUES_GATE_REL:.0e}, expected {VALUES_EXPECTATION}")
    print(f"  view batch pinned at {VIEW_BATCH}; the pixel ladder is "
          f'{[d for d, _w in PIXEL_LADDER]} as divisors of the full pixel set')
    print(f'  {"arm":<{ARM_COL}}{"widths":>26}{"stride":>8}{"pieces":>8}  note')
    for entry in plan:
        if entry.get("skip"):
            print(f'  {entry["arm"]:<{ARM_COL}}{"-":>26}{"-":>8}{"-":>8}  '
                  f'SKIPPED: {entry["skip"]}')
            continue
        widths = [p["work"] for p in entry["pieces"]]
        shown = (f"{widths[0]} x {len(widths)}"
                 if len(set(widths)) == 1 and len(widths) > 1
                 else ", ".join(str(w) for w in widths))
        print(f'  {entry["arm"]:<{ARM_COL}}{shown:>26}'
              f'{entry["pieces"][0]["stride"]:>8}{len(widths):>8}  '
              f'{entry["note"]}')
    print(f"  profiler leg: {'on' if NCU_ENABLED else 'off (MG20_NCU=0)'}"
          + (f", arms {list(NCU_ARMS)}" if NCU_ENABLED else ""))
    print("no library file is touched: the contiguous arms call the shipped "
          "wrapper, and the strided arms copy its launch code into this file")


def main():
    plan = build_plan()
    if DRY:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"mg20_width_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg20 width mechanism on {RUN_LABEL} ({DEVICE}); {len(plan)} arms "
          f"-> {out_path}", flush=True)
    rows = []
    with open(out_path, "w") as sink:
        for cfg in plan:
            print(f'  {cfg["arm"]}', flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        print("  profiler leg", flush=True)
        leg = ncu_leg(plan, sys.executable, RESULTS_DIR)
        sink.write(json.dumps(dict(ncu_leg=leg)) + "\n")
        sink.flush()
        summary = summarize(rows, leg, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    print(f"\nwrote {out_path}")
    return 0 if summary["healthy"] else 2


# ── the report ────────────────────────────────────────────────────────────────
def _fmt(value, width=10, kind="f", prec=3):
    if value is None:
        return f'{"-":>{width}}'
    if isinstance(value, str):
        return f"{value:>{width}}"
    return f"{value:>{width}.{prec}{kind}}"


def _metric(kernel, name, default=None):
    metrics = (kernel or {}).get("metrics") or {}
    if name in metrics:
        return metrics[name]
    for key, value in metrics.items():
        if key.split(" ")[0] == name:
            return value
    return default


def _duration_ms(kernel):
    """gpu__time_duration.sum in milliseconds, using the unit ncu reported.

    ncu emits this metric in nanoseconds, microseconds or milliseconds
    depending on the release and the page, so the unit is read rather than
    assumed.  An unrecognized unit returns None and the raw value stays on the
    row.
    """
    value = _metric(kernel, "gpu__time_duration.sum")
    if not isinstance(value, (int, float)):
        return None
    unit = ((kernel or {}).get("units") or {}).get("gpu__time_duration.sum")
    if unit is None:
        for key in (kernel.get("metrics") or {}):
            if key.startswith("gpu__time_duration.sum") and "(" in key:
                unit = key[key.index("(") + 1:key.rindex(")")]
    scale = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3,
             "msecond": 1.0, "ms": 1.0, "second": 1e3, "s": 1e3}
    factor = scale.get(str(unit).strip()) if unit is not None else None
    return None if factor is None else value * factor


def widest_reading(row):
    """The arm's widest piece, as (width, ms per slice).

    One arm may hold two widths (w512_496 cuts 1008 into a 512 and a 496), and
    a comparison has to name which one it used.  The widest is taken, and both
    are printed in table 1.
    """
    widths = row.get("by_width") or {}
    best = None
    for entry in widths.values():
        if best is None or int(entry.get("width", 0)) > int(best.get("width", 0)):
            best = entry
    if best is None:
        return None, None
    return int(best.get("width")), best.get("ms_per_slice")


def timing_table(rows):
    """Table 1: what each launch shape cost, and whether its values agreed."""
    header = (f'{"arm":<{ARM_COL}}{"width":>7}{"stride":>8}{"pieces":>7}'
              f'{"per launch ms":>15}{"spread ms":>11}{"per slice ms":>14}'
              f'{"vs w1008":>10}{"values rel":>12}  check')
    print("\n===== table 1: per-launch cost against values-block width =====")
    print(f"full pixel set, view batch pinned at {VIEW_BATCH}, "
          f"{TIMED_REPEATS} timed repeat"
          f'{"" if TIMED_REPEATS == 1 else "s"} after a discarded warm-up')
    print(header)
    print("-" * len(header))
    base = None
    for row in rows:
        if row.get("arm") == "w1008":
            _width, base = widest_reading(row)
    for row in rows:
        arm = row.get("arm", "?")
        if row.get("error"):
            print(f'{arm:<{ARM_COL}}  ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:80]}')
            continue
        if row.get("skipped"):
            print(f'{arm:<{ARM_COL}}  SKIPPED: {row["skipped"][:96]}')
            continue
        values = row.get("values") or {}
        rel = values.get("rel")
        check = "ok" if values.get("ok") else "VALUES GATE FAILED"
        widths = row.get("by_width") or {}
        if not widths:
            print(f'{arm:<{ARM_COL}}{"-":>7}{"-":>8}{"-":>7}{"-":>15}{"-":>11}'
                  f'{"-":>14}{"-":>10}{_fmt(rel, 12, "e", 2)}  {check}')
            continue
        pieces = row.get("piece_widths") or []
        for key in sorted(widths, key=lambda k: -int(k)):
            entry = widths[key]
            per_slice = entry.get("ms_per_slice")
            ratio = (per_slice / base) if (base and per_slice) else None
            count = sum(1 for w in pieces if int(w) == int(key))
            print(f'{arm:<{ARM_COL}}{int(key):>7}'
                  f'{entry.get("stride", row.get("layout_stride", "-")):>8}'
                  f'{count:>7}'
                  f'{_fmt(entry.get("median_ms"), 15, "f", 3)}'
                  f'{_fmt(entry.get("spread_ms"), 11, "f", 3)}'
                  f'{_fmt(per_slice, 14, "f", 4)}'
                  f'{_fmt(ratio, 10, "f", 2)}'
                  f'{_fmt(rel, 12, "e", 2)}  {check}')
    print("-" * len(header))


def anchor_table(rows):
    """Section 1.9 beside this run, and the gated comparison.

    Returns the per-arm verdicts, or None when the comparison does not apply.
    Section 1.9's readings are H100 readings of the Triton kernel, so the CPU
    smoke cannot be judged against them: the smoke runs the torch body on
    another machine entirely, and it exists to test this harness rather than to
    measure anything.
    """
    print("\nsection 1.9 (job 15159551 on h008) beside this run:")
    if SMOKE:
        print("  NOT COMPARED: section 1.9 measured the Triton kernel on an "
              "H100, and this is the CPU smoke running the torch body.  The "
              "anchors do not apply and they do not touch the exit code.")
        return None
    base = None
    for row in rows:
        if row.get("arm") == "w1008":
            _width, base = widest_reading(row)
    if base is None:
        print("  NOT COMPARED: w1008 is not in this run, and every anchor is a "
              "cost per slice relative to width 1008.  Select w1008 to have "
              "the anchors checked.")
        return None
    header = (f'{"arm":<{ARM_COL}}{"1.9 per launch":>16}{"1.9 per slice":>15}'
              f'{"1.9 ratio":>11}{"this ratio":>12}{"off by":>9}'
              f'{"this mixture mean":>19}')
    print(header)
    print("-" * len(header))
    verdicts = []
    for row in rows:
        anchor = ANCHOR_1_9.get(row.get("anchor") or "")
        if anchor is None or row.get("error") or row.get("skipped"):
            continue
        _width, per_slice = widest_reading(row)
        ratio = (per_slice / base) if per_slice else None
        off = (abs(ratio - anchor["per_slice_ratio"]) / anchor["per_slice_ratio"]
               if ratio else None)
        verdicts.append((row.get("arm"), ratio, anchor["per_slice_ratio"], off))
        print(f'{row.get("arm"):<{ARM_COL}}'
              f'{anchor["per_launch_ms"]:>16.2f}{anchor["ms_per_slice"]:>15.4f}'
              f'{anchor["per_slice_ratio"]:>11.2f}{_fmt(ratio, 12, "f", 2)}'
              f'{_fmt(off, 9, "f", 2)}'
              f'{_fmt(row.get("mixture_mean_per_launch_ms"), 19, "f", 2)}')
    print("-" * len(header))
    print("The gated column is 'off by': the cost per slice relative to width "
          f"1008, against section 1.9's, tolerance {ANCHOR_TOL:.0%}.")
    print("The mixture mean rebuilds section 1.9's quantity from this run's "
          "pixel ladder.  Section 1.9's per-launch numbers are means over 680 "
          "calls of a reconstruction at four different pixel counts (512 calls "
          "at 12051 pixels, 128 at 48203, 32 at 192810, 8 at 771240), so they "
          "are not the cost of one full-pixel launch.  The mixture mean is "
          "reported and not gated: this run's launches are isolated and warm, "
          "and mg10's were embedded in a reconstruction.")
    return verdicts


def counter_table(leg, rows):
    """Table 2: what the counters said, one profiled launch per width."""
    print("\n===== table 2: Nsight Compute counters, one warm launch per "
          "width =====")
    if not leg.get("attempted"):
        print(f'  profiler leg NOT ATTEMPTED: {leg.get("reason")}')
        return False
    if not leg.get("profiler_permitted"):
        probe = leg.get("permission_probe") or {}
        print("  profiler_permitted = false")
        print(f'  reason: {leg.get("reason")}')
        message = (probe.get("message") or "").strip().splitlines()
        for line in message[-6:]:
            print(f"    {line}")
        print("  the timing leg stands alone; this is a recorded finding and "
              "not a failure")
        return False
    if not leg.get("arms"):
        print(f'  no arm profiled: {leg.get("reason")}')
        return False
    print(f'  kernel filter {leg.get("kernel_name_filter")}, '
          f'pixel divisor {leg.get("pixel_div")}')
    header = (f'{"arm":<{ARM_COL}}{"width":>7}{"occup %":>9}{"limiter":>22}'
              f'{"L2 hit %":>10}{"red sectors":>14}{"atom sectors":>14}'
              f'{"DRAM rd GB":>12}{"DRAM wr GB":>12}{"dur ms":>9}')
    print(header)
    print("-" * len(header))
    by_arm = {row.get("arm"): row for row in rows}
    printed = False
    for arm, record in leg["arms"].items():
        kernels = record.get("kernels")
        if not kernels:
            print(f'{arm:<{ARM_COL}}  NO PROFILE: {record.get("reason", "")[:80]}')
            continue
        kernel = kernels[0]
        row = by_arm.get(arm) or {}
        width, _per_slice = widest_reading(row)
        limits = {
            "registers": _metric(kernel, "launch__occupancy_limit_registers"),
            "shared mem": _metric(kernel, "launch__occupancy_limit_shared_mem"),
            "blocks": _metric(kernel, "launch__occupancy_limit_blocks")}
        known = {k: v for k, v in limits.items() if isinstance(v, (int, float))}
        limiter = min(known, key=known.get) if known else None
        limiter_text = (f"{limiter} {known[limiter]:g}" if limiter else "-")
        read = _metric(kernel, "dram__bytes_read.sum")
        write = _metric(kernel, "dram__bytes_write.sum")
        occupancy = _metric(
            kernel, "sm__warps_active.avg.pct_of_peak_sustained_active")
        print(f'{arm:<{ARM_COL}}{(width if width else "-"):>7}'
              f'{_fmt(occupancy, 9, "f", 1)}'
              f'{limiter_text:>22}'
              f'{_fmt(_metric(kernel, "lts__t_sector_hit_rate.pct"), 10, "f", 1)}'
              f'{_fmt(_metric(kernel, "lts__t_sectors_op_red.sum"), 14, "e", 3)}'
              f'{_fmt(_metric(kernel, "lts__t_sectors_op_atom.sum"), 14, "e", 3)}'
              f'{_fmt(read / 2 ** 30 if isinstance(read, (int, float)) else None, 12, "f", 2)}'
              f'{_fmt(write / 2 ** 30 if isinstance(write, (int, float)) else None, 12, "f", 2)}'
              f'{_fmt(_duration_ms(kernel), 9, "f", 2)}')
        printed = True
    print("-" * len(header))
    if printed:
        print("Durations here are ncu's, not wall times: ncu serializes and "
              "replays each kernel to collect its counters.  The timing leg "
              "owns time.")
    return printed


def write_path_table(leg, rows):
    """The write path, priced against what the kernel actually asks for.

    The kernel issues one float atomic add per (pixel, tap, column).  Eight
    consecutive floats fill one 32-byte L2 sector, so a perfectly ordered write
    path would need one sector per eight atomics.  The ratio of the sectors ncu
    counted to that ideal is how far the write path is from ordered, and it is
    the number the sorted-channel proposal would move.
    """
    if not leg.get("arms"):
        return
    by_arm = {row.get("arm"): row for row in rows}
    header = (f'{"arm":<{ARM_COL}}{"width":>7}{"atomics issued":>16}'
              f'{"sectors":>13}{"sectors/ideal":>15}{"DRAM wr / slab":>16}')
    print("\nthe write path, per profiled launch:")
    print(header)
    print("-" * len(header))
    for arm, record in leg["arms"].items():
        kernels = record.get("kernels")
        row = by_arm.get(arm) or {}
        if not kernels or not row:
            continue
        kernel = kernels[0]
        width, _per_slice = widest_reading(row)
        pixels = row.get("num_pixels_full")
        taps = row.get("psf_width")
        if not (width and pixels and taps):
            continue
        pixels = pixels // max(1, leg.get("pixel_div", 1))
        atomics = float(taps) * float(pixels) * float(width)
        red = _metric(kernel, "lts__t_sectors_op_red.sum") or 0.0
        atom = _metric(kernel, "lts__t_sectors_op_atom.sum") or 0.0
        sectors = float(red) + float(atom)
        ideal = atomics / 8.0
        write = _metric(kernel, "dram__bytes_write.sum")
        slab = (row.get("fixed_costs") or {}).get("out_slab_bytes")
        amplification = (write / slab if isinstance(write, (int, float)) and slab
                         else None)
        print(f'{arm:<{ARM_COL}}{width:>7}{atomics:>16.3e}{sectors:>13.3e}'
              f'{_fmt(sectors / ideal if ideal else None, 15, "f", 2)}'
              f'{_fmt(amplification, 16, "f", 2)}')
    print("-" * len(header))
    print("'sectors/ideal' is 1.00 when every atomic lands in a fully used "
          "32-byte sector.  'DRAM wr / slab' is the DRAM write traffic divided "
          "by the size of the output slab the launch writes; 1.00 means each "
          "byte of the slab reached DRAM once.")


def reading_guide(rows, leg, profiled):
    """The two questions, kept apart, with what to look at for each."""
    print("\n===== how to read the two tables =====")
    print("\n(a) THE WIDTH REGIME.  Table 1 says whether the doubling "
          "reproduced: the 'vs w1008' column is the cost per slice relative to "
          "the full-width launch, and section 1.9 read 2.00 at width 504.")
    print("    The two new layouts separate work from memory.  w512_stride1008 "
          "does 512 columns of work over a 1008-wide layout.  If its cost per "
          "slice sits near w1008's, the regime travels with the LAYOUT, and a "
          "narrow launch is slow because of how it addresses memory.  If it "
          "sits near w504's, the regime travels with the WORK, and the layout "
          "is innocent.")
    print("    w512_496 varies one further thing on its own: both of its "
          "widths are divisible by 16 and w504's is not, at nearly the same "
          "work.  Triton specializes an integer kernel argument on divisibility "
          "by 16, and num_cols is such an argument.  If w512_496 reads near "
          "w1008, that specialization and the vector width it buys are the "
          "mechanism.")
    print("    w504_alloc1008 moves the allocation size alone.  It is the "
          "weakest of the three and it separates allocation size from layout.")
    grid_lines = []
    for row in rows:
        geometry = row.get("launch_geometry") or {}
        if geometry.get("available"):
            grid_lines.append(
                f'    {row.get("arm"):<{ARM_COL}} grid {geometry.get("grid")}, '
                f'{geometry.get("blocks")} blocks, tile '
                f'{geometry.get("block_p")} x {geometry.get("block_r")}, '
                f'num_cols divisible by 16: '
                f'{geometry.get("num_cols_divisible_by_16")}')
    if grid_lines:
        print("    the launch shapes, derived from the library's own tile rule:")
        for line in grid_lines:
            print(line)
        print("    Note that the contiguous arms launch the SAME total number "
              "of blocks across their pieces, because halving the width halves "
              "the blocks per launch and doubles the launches.  A block count "
              "too small to fill the machine cannot by itself explain a per-"
              "launch cost that does not fall.")
    if profiled:
        print("    In table 2, occupancy falling as the width falls, with the "
              "L2 and DRAM columns flat, points at how the launch fills the "
              "machine.  The L2 hit rate falling or the DRAM columns rising at "
              "the narrow widths, with occupancy flat, points at memory.  The "
              "limiter column names which of registers, shared memory or "
              "blocks per multiprocessor caps occupancy; it should not move "
              "with the width, because the tile and the kernel do not.")
    else:
        print("    Table 2 did not run, so question (a) is answered from the "
              "timing leg and the launch shapes alone.")

    print("\n(b) THE SORTED-CHANNEL QUESTION, at every width, and independent "
          "of (a).  The kernel scatters its results with one float atomic add "
          "per (pixel, tap, column).  Sorting the channels inside the kernel "
          "would replace scattered atomics with ordered accumulation.")
    if profiled:
        print("    The signature that sorting would attack is high atomic and "
              "reduction sector counts against a low L2 hit rate, with DRAM "
              "write traffic well above the output slab's size.  The write-"
              "path table prices exactly that: 'sectors/ideal' near 1 means "
              "the writes are already coalesced and sorting has little to win, "
              "and a large ratio means each atomic is touching a sector it "
              "barely uses.")
        print("    Read this at EVERY width.  A width answer to (a) does not "
              "settle it: an occupancy-bound kernel with a badly used write "
              "path is still a candidate for sorted accumulation, and the two "
              "readings come from different columns.")
    else:
        print("    The counters this reading needs are the profiler leg's, and "
              "the leg did not run.  The sorted-channel question is therefore "
              "UNANSWERED by this job, and it is not closed by whatever the "
              "timing leg says about the width.")


def summarize(rows, leg, out_path):
    """The tables a person reads the mechanism from, and the instrument-health
    accounting the exit code comes from.

    These are two different things and this function keeps them apart.  The
    mechanism verdict is not computed here.  What is computed here is whether
    the instruments worked: every selected arm produced a row, every arm's
    values agreed with the full-width reference, and every section 1.9 arm
    reproduced its anchor.
    """
    print(f"\n===== mg20 width mechanism ({out_path}) =====")
    timing_table(rows)
    verdicts = anchor_table(rows)
    profiled = counter_table(leg, rows)
    if profiled:
        write_path_table(leg, rows)
    reading_guide(rows, leg, profiled)

    broken, values_failed, anchors_off, skipped = [], [], [], []
    for row in rows:
        arm = row.get("arm", "?")
        if row.get("error"):
            broken.append(f"{arm}|error")
            continue
        if row.get("skipped"):
            skipped.append(arm)
            continue
        for reason in row.get("invalid_reasons") or []:
            print(f"\nARM CHECK FAIL {arm}: {reason}")
            broken.append(f"{arm}|check")
        if not (row.get("values") or {}).get("ok"):
            values_failed.append(arm)
        if not row.get("by_width"):
            broken.append(f"{arm}|no timing")
    for arm, ratio, expected, off in verdicts or ():
        if off is None or off > ANCHOR_TOL:
            shown = f"{ratio:.2f}" if ratio else "no reading"
            anchors_off.append(f"{arm} ({shown} against {expected:.2f})")

    fixed = [(row.get("arm"), (row.get("fixed_costs") or {}).get("hfan_ms"),
              (row.get("fixed_costs") or {}).get("out_slab_ms"))
             for row in rows if row.get("fixed_costs")]
    if fixed:
        print("\nthe two per-call costs inside every timed launch, so they can "
              "be subtracted (ms):")
        for arm, hfan, slab in fixed:
            print(f"  {arm:<{ARM_COL}} horizontal-fan contract "
                  f"{_fmt(hfan, 8, 'f', 3)}   zeroed output slab "
                  f"{_fmt(slab, 8, 'f', 3)}")

    for row in rows:
        contract = row.get("wrapper_contract") or {}
        if contract.get("kernel_arg_names"):
            print("\nwhat the shipped forward wrapper allows, read at runtime:")
            print(f'  the wrapper packs its values argument: '
                  f'{contract.get("wrapper_packs_values")}')
            for line in contract.get("source_lines") or []:
                print(f"    {line}")
            print(f'  the kernel takes a values row-stride argument: '
                  f'{contract.get("kernel_has_values_row_stride")}')
            print(f'  kernel arguments: {contract.get("kernel_arg_names")}')
            print("  A values tensor whose row stride exceeds its column count "
                  "therefore cannot reach the shipped kernel.  The strided "
                  "arms pass the wide block with num_cols set to its true row "
                  "stride and truncate the grid instead, which is a launch-"
                  "side change and touches no library file.")
            break

    for row in rows:
        if row.get("copy_fidelity"):
            check = row["copy_fidelity"]
            print(f'\nthe copied launch code, run at the full width against '
                  f'the library wrapper ({row.get("arm")}): rel '
                  f'{_fmt(check.get("rel"), 0, "e", 2)}, '
                  f'{"agrees" if check.get("ok") else "DISAGREES"}')

    hot = [row.get("arm") for row in rows if row.get("gpu_hot")]
    if hot:
        print(f"\nGPU health: {len(hot)} arm(s) sampled hot: {hot}")
    if skipped:
        print(f"\n{len(skipped)} arm(s) skipped by plan: {skipped}.  A planned "
              "skip does not change the exit code")
    if values_failed:
        print(f"\n{len(values_failed)} arm(s) failed the values gate: "
              f"{values_failed}.  Their timings mean nothing and the exit code "
              "says so")
    if anchors_off:
        print(f"\n{len(anchors_off)} section 1.9 anchor(s) did not reproduce "
              f"within {ANCHOR_TOL:.0%}: {anchors_off}.  The width regime this "
              "job exists to explain was not reproduced on this tree and this "
              "node, so the ablations have nothing to be read against")

    healthy = not (broken or values_failed or anchors_off)
    print(f"\nexit code reports INSTRUMENT HEALTH only: "
          f'{"healthy" if healthy else "BROKEN"}.  It covers three things: '
          "every selected arm produced a row, every arm's values agreed with "
          f"the full-width reference at {VALUES_GATE_REL:g}, and every section "
          f"1.9 arm reproduced its anchor within {ANCHOR_TOL:.0%}.")
    if verdicts is None:
        print("The anchor test did not apply in this run, so the exit code "
              "rests on the first two conditions alone.")
    print("The profiler leg's absence never changes it, and neither does what "
          "either leg found.  The mechanism verdict is read by a person from "
          "the tables above and the rows in the jsonl.")
    return dict(healthy=healthy, broken=broken, values_failed=values_failed,
                anchors_off=anchors_off, skipped=skipped, hot=hot,
                profiler_permitted=leg.get("profiler_permitted"),
                profiler_reason=leg.get("reason"),
                arms=len(rows))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        worker_cfg = json.loads(sys.argv[2])
        try:
            out = run_arm(worker_cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif len(sys.argv) > 2 and sys.argv[1] == "--one-launch":
        worker_cfg = json.loads(sys.argv[2])
        try:
            out = one_launch(worker_cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(worker_cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif len(sys.argv) > 1 and sys.argv[1] == "--trivial-kernel":
        print("__RESULT__" + json.dumps(trivial_kernel()))
    else:
        sys.exit(main())
