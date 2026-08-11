"""mg13 -- is the SORTED channel reduction faster than the ATOMIC SCATTER in
the parallel forward, on the data flow the library actually runs today?

WHAT IS BEING MEASURED, in one paragraph.  The parallel forward bins each
voxel's weighted rows into the detector channels it lands on.  Today the torch
body does that with an atomic scatter: for each of the psf taps it builds the
weighted rows and calls ``index_add_`` into a flat accumulator, and several
pixels landing on one channel collide inside that one call.  mbirjax does the
same reduction a different way -- it sorts the contributions by channel, sums
each run of equal channels, and writes each channel once -- and it measured that
form faster on GPU wherever the mean number of collisions per channel is high
enough.  This job puts the two side by side on the same inputs, on the same
substrate, at the library's own 1024-class cell, and reads out per-launch time,
the forward funnel's wall, and the composed reconstruction.

WHY THIS IS BEING RE-ASKED.  This port declined the sorted form once, and the
decision rested on a threshold that has since moved: the forward now gathers
full-width columns by default (so a body call carries the whole slice axis, 1008
columns here, where the banded walk fed it a shard's width), and the cross-device
copies have moved off the compute streams.  Both change the numbers the earlier
ruling was drawn against.  The recorded reasoning is in section 8.5 of the
forward remedy memo, and its instruction for the re-gate is explicit: probe the
LIGHT PER-CALL FORM first -- mbirjax's shipped win is a per-call sort inside the
kernel, not the cached pre-sorted streams this port's own design sketched.

THIS FILE DRAWS NO STOP-OR-GO CONCLUSION, AND THAT IS DELIBERATE.  There is no
threshold, no stop rule and no gate anywhere below.  The recorded threshold lives
in the memo and in mbirjax's own constants, and deciding whether these numbers
clear it is a human re-read of that record against this reading.  What this file
produces is two mechanical lines and a table of distances.

THE THREE BODIES, and why the comparison has to control which one runs.  On CUDA
the parallel forward normally runs a hand-written Triton kernel, NOT the torch
fan -- so a sorted form patched into the torch fan would not even be on the
production path.  The comparison is therefore three-way:
    triton scatter   the production reference: the kernel the library selects by
                     default, whose forward scatters with per-tap atomics inside
                     the kernel.  Nothing is patched.  This arm exists to say
                     what the other two are being compared against.
    torch scatter    the BASELINE for the candidate: the compiled torch fan,
                     reached by setting MBIRTORCH_DISABLE_TRITON=1, running the
                     shipped ``index_add_`` reduction.
    torch sorted     the CANDIDATE: the same compiled torch fan with its channel
                     reduction replaced by sort + segment-sum + one write per
                     run.  Same kill switch, so the substrate is identical to the
                     baseline's and the ONLY difference between the two arms is
                     the reduction.
The candidate and its baseline share a substrate; the production reference is
measured beside them rather than against them.  A sorted-vs-triton ratio would
confound the reduction with the whole compiled-body-versus-kernel difference.

HOW THE CANDIDATE IS INSTALLED.  This is a PROBE, so it patches a live model in
its own process and edits no repository -- the mg-series convention (mg12 wraps
library functions by rebinding module attributes: the region instrument shadows
``_sharding.exchange_qggmrf_halos`` and friends, the busy probe swaps the entries
of ``projector_functions._fwd_body_per_dev`` in place).  The same mechanism
serves here: ``mbirtorch.parallel_beam.fan_forward_batch`` is the name the
forward body actually calls, so that module attribute is rebound to the sorted
implementation defined below.  The rebinding happens BEFORE the model is built,
so the compiled body traces the sorted reduction from its first compile rather
than tripping a guard and recompiling mid-run.  Every arm records which function
its body would call, by name and by identity, and the sorted implementation
counts its own calls, so a patch that did not take fails the arm instead of
reporting a tidy "no change".

THE ARMS.  Six measured arms plus one untimed sinogram generator, all at mg9's /
mg10's / mg11's / mg12's cell (1024, 1008, 992), which gives 1008 slices --
divisible by 1 and 4, so no arm's arithmetic carries a padding term.
    n=1   triton scatter | torch scatter | torch sorted
    n=4   triton scatter | torch scatter | torch sorted
One device answers the question, because the question is about what happens
inside one projector call.  Four devices are the composed check: the candidate
has to survive the sharded driver, the column gather and the per-device workers,
not just a bench.  ARM ORDER is device-count-major, so the three bodies of one
device count run back-to-back: a truncated job then yields a COMPLETE three-way
comparison rather than one body at both counts, and any drift in the node over
the hours the job runs lands inside a comparison rather than between its columns.

WHAT EACH DEVICE COUNT MEASURES.
    n=1   the isolated per-launch bench (below), the forward funnel's wall and
          per-launch time from the composed run, and the composed reconstruction.
    n=4   the composed reconstruction, with the same funnel and per-launch
          instruments riding along -- but no isolated bench, because a bench of
          one call on one device answers the same question more cheaply.

THE ISOLATED BENCH, and why it is EAGER.  Beside the composed run, the n=1 arms
time the forward body directly on inputs built once and shared by every body:
same pixel indices, same view angles, same values block.  It runs the bodies
EAGERLY, without torch.compile, for a reason.  The composed run already measures
the compiled path -- that is what the per-launch column is -- so an eager bench
adds the one reading the composed run cannot give: the cost of the two
reductions with the compiler held out of it.  If the two readings disagree in
direction, the compiler is doing something to one of them, and the recorded
dynamo counters and compile errors on each row are where to look first.
    The bench also SWEEPS THE PIXEL COUNT, because the pixel count is what the
mean channel-collision count is made of: psf_width * num_pixels / num_channels.
That ratio is the variable mbirjax's own selection rule is written in terms of,
and a single point would say nothing about where the crossover sits.  Each bench
point records its ratio beside its times.
    In the n=1 TRITON arm all three bodies exist in one process, so that arm's
bench is the three-way comparison with no cross-process drift in it at all.  The
two torch arms run the same bench without the kernel, which is a check that the
process-to-process readings agree.

THE VALUE QUESTION.  Sorting changes the order in which a channel's
contributions are summed, so the sorted form cannot be bit-identical to the
scatter and is not expected to be.  It is expected to agree to float tolerance,
and the tolerance this job holds is 1e-5 relative.  Two independent readings:
    the bench distance   sorted against scatter on the SAME inputs in the SAME
                         process, which is the clean measurement -- one variable,
                         no reconstruction in between.
    the composed distance  each arm's reconstruction sampled on a stride, against
                         the torch scatter arm's, plus each arm's OWN
                         pass-to-pass distance.  That own-distance is the floor
                         everything else is read against: the triton forward
                         accumulates with float atomics and is not
                         bit-reproducible, so two runs of one arm already differ.

THE TWO VERDICT LINES, and the rule each is drawn by.  Neither is a gate.
    SORTED vs SCATTER (torch substrate)   the ratio of the torch scatter arm's
        reading to the torch sorted arm's, at n=1 per-launch and at n=4 composed.
        Above 1.00 the sorted form is faster.  Printed with both raw numbers
        beside it so the ratio is never read alone.
    TRITON REFERENCE   the ratio of the torch scatter arm's reading to the triton
        arm's, at the same two places -- what the production kernel is worth
        against the substrate the candidate was measured on.
A "spread" is one arm's own max-minus-min over its two warm passes, floored at
SPREAD_FLOOR_FRAC of the torch scatter arm's reading.  Two passes is a weak
estimator of run-to-run noise, so a difference smaller than that floor is
reported as WITHIN THE SPREAD beside the ratio rather than being resolved.  An
arm whose own two passes disagree by more than WARM_INSTABILITY_FRAC of its own
reading gets an UNSTABLE ARM line printed beside the verdict lines; it changes
no number, it says the number should not be leaned on.

TERMS OF ART, each defined once, here.
    arm          one subprocess run at fixed parameters: one device count, one
                 body.
    body         which forward runs inside a view batch: the Triton kernel, the
                 torch fan with the scatter, or the torch fan with the sorted
                 reduction.
    composed     one whole timed reconstruction's wall.
    bracket      the forward funnel's per-device CUDA event span -- everything
                 the forward does, kernels and copies together.  The reading is
                 the largest device's.
    busy         the sum of per-body-call event spans on one device -- the
                 projection launches alone.  The reading is the largest device's.
    per-launch   busy divided by the body call count.
    per-slice    per-launch divided by the mean column count of the values blocks
                 that arm's body was handed.  The mg-series precedent for this
                 number is 0.0411 ms per slice, measured 2026-08-10 on one H100
                 for the parallel forward kernel on a full-width 1008-column
                 block (against 0.0823 on a 504-wide one).
    collision ratio  psf_width * num_pixels / num_channels: the mean number of
                 contributions landing on one detector channel in one call.  This
                 is the variable mbirjax's selection rule for the sorted form is
                 written in, and it is recorded for every launch shape here.
    spread       one arm's max minus min over its two warm passes.
    repeat floor an arm's own pass-to-pass value distance.

ENVIRONMENT KNOBS (all optional; the tree root is required).
    MG13_TREE=/path/to/tree          the source tree every arm imports (required)
    MG13_ARMS=n1-sorted,n4-triton    run only these arm tokens
    MG13_COUNTS=1,4                  run only these device counts
    MG13_ITERATIONS=3                VCD iterations per reconstruction
    MG13_WARM_REPEATS=2              timed reconstructions after the cold pass
    MG13_MAX_EVENT_PAIRS=400000      per-reconstruction event budget
    MG13_SORT_VIEW_CHUNK=0           views per sort inside the sorted fan; 0 means
                                     derive it from the library's own transient
                                     budget (see the implementation's docstring)
    MG13_SORT_STABLE=1               ask for a stable sort inside the sorted fan
                                     (off by default; see SORT_STABLE_ENV)
    MG13_NO_BENCH=1                  skip the isolated bench
    MG13_KEEP_ARTIFACTS=1            keep the sinogram and the value samples
    MG13_SMOKE=1                     the local CPU smoke (tiny cell, few iters)
    MG13_DEVICE=cpu                  smoke device
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

# mg9's / mg10's / mg11's / mg12's cell, and nothing else.  cell = (num_views,
# num_det_rows, num_det_channels); at this cell parallel beam gives recon
# (992, 992, 1008), so the slice count is 1008 and it divides 1 and 4 exactly.
CELL = (1024, 1008, 992)

SMOKE = os.environ.get("MG13_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("MG13_DEVICE", "cpu" if SMOKE else "cuda")

# The single source tree.  One tree, three bodies: unlike mg12 nothing here is a
# source change, so there is nothing to overlay and no cross-tree fingerprint to
# compare.  The root is still required and still witnessed per arm, because an
# arm that imported some other checkout would report a difference that is not the
# one being measured.
TREE_ENV = "MG13_TREE"

# The three bodies, and the environment each one needs.  The kill switch is the
# library's own name for it; every arm asserts the value it was given.
BODIES = ("triton", "scatter", "sorted")
BODY_NOTE = {
    "triton": "the production reference: the Triton parallel forward kernel, "
              "per-tap atomics inside the kernel, nothing patched",
    "scatter": "the baseline: the compiled torch fan with the shipped "
               "index_add_ scatter (MBIRTORCH_DISABLE_TRITON=1)",
    "sorted": "the candidate: the same compiled torch fan with sort + "
              "segment-sum + one write per run (MBIRTORCH_DISABLE_TRITON=1)",
}
# Which kill-switch value each body needs, and whether the fan is patched.
BODY_ENV = {"triton": dict(disable_triton="0", patch_sorted="0"),
            "scatter": dict(disable_triton="1", patch_sorted="0"),
            "sorted": dict(disable_triton="1", patch_sorted="1")}
KILL_SWITCH = "MBIRTORCH_DISABLE_TRITON"
PATCH_ENV_VAR = "MG13_PATCH_SORTED"

DEVICE_COUNTS = (1, 4)
SMOKE_COUNTS = (1, 2)

GEOMETRY = "parallel"      # item 13 is scoped to parallel; see memo section 8.5

VCD_ITERATIONS = int(os.environ.get("MG13_ITERATIONS", "1" if SMOKE else "3"))
VCD_SEED = 13        # mg1's / mg5's / mg9's / mg10's / mg11's / mg12's seed
WARM_REPEATS = max(2, int(os.environ.get("MG13_WARM_REPEATS", "2")))

# ── THE COMPARISON RULE, one constant, stated before any number is read ───────
# Two warm passes is a weak estimator of run-to-run noise, so a difference
# smaller than this fraction of the torch scatter arm's own reading is reported
# as WITHIN THE SPREAD rather than resolved.
SPREAD_FLOOR_FRAC = 0.02
# The other end of the same problem: an arm whose own two passes disagree by more
# than this fraction of its own reading is not a usable measurement of anything,
# and it gets an UNSTABLE ARM line printed beside the verdict lines.  This flags
# a reading; it overrules none.
WARM_INSTABILITY_FRAC = 0.25
# The value tolerance this job holds the sorted form to.  Sorting changes the
# summation order, so equality is not expected; agreement at this level is.
VALUE_REL_TOL = 1e-5

# ── WHAT EARLIER JOBS MEASURED, quoted, and used for nothing but arithmetic ───
# Used ONLY for the wall estimate and for the per-slice precedent printed beside
# this job's own numbers.  Nothing here is ever compared against a measurement
# taken in this job: every comparison below is between arms measured here.
#
# mg11 row mg11_flip_gates_h001_20260811_041522.jsonl, four H100s, this cell,
# column gather, both kernels on, 1 cold plus 3 warm passes per arm.
MG11_PARALLEL_N4 = dict(bracket_s=13.33, busy_s=10.83, composed_s=21.8, wall_s=121)
# The one-device composed reading this campaign carries for parallel 1024 with
# the kernels on (forward remedy memo section 1).  It predates the column gather,
# so it is a wall estimate only.
ONE_DEVICE_COMPOSED_S = 40.0
# How much slower the compiled torch bodies ran than the kernel path at this cell
# in the five-arm composed gate: 5.56x of jax against the kernels' 1.90x
# (parallel_beam._view_batch_bodies).  The torch arms in this job run both
# directions on the torch bodies, so their wall is scaled by this.
TORCH_BODY_SLOWDOWN = 5.56 / 1.90
# An arm never measured before takes mg10's and mg11's rule: the low end assumes
# it costs what its nearest measured neighbour costs, the high end half again.
MEASURED_HIGH_FACTOR = 1.05
UNMEASURED_HIGH_FACTOR = 1.50
GENERATOR_S = 70
BENCH_S = 90
# This job gets its OWN inductor cache, so the first arm at each launch shape
# compiles from cold.  That cost lands in each arm's discarded cold pass.
COLD_CACHE_LOW_S = 600
COLD_CACHE_HIGH_S = 1200

# The mg-series precedent for the per-slice column, measured 2026-08-10 on one
# H100: the parallel forward kernel at 0.0411 ms per slice on a full-width
# 1008-column block of values (0.0823 on a 504-wide one).  Printed beside this
# job's own per-slice numbers; never compared arithmetically against them.
PRECEDENT_MS_PER_SLICE = 0.0411
PRECEDENT_NOTE = ("the parallel forward kernel, one H100, full-width 1008-column "
                  "block, measured 2026-08-10")

RESULTS_DIR = os.environ.get(
    "MG13_RESULTS", os.path.dirname(os.path.abspath(__file__)))
RUN_LABEL = platform.node().split(".")[0]

# The value sample: one reconstruction voxel out of every VALUE_SAMPLE_TARGET per
# axis, so the sample is a few hundred thousand values whatever the cell.
VALUE_SAMPLE_TARGET = 64

# The library files whose bytes are fingerprinted per arm.  Every arm runs one
# tree, so these prove the tree is the pinned one -- horizontal_fan.py is here
# because it is the file the candidate replaces a function from.
FINGERPRINT_FILES = ("horizontal_fan.py", "parallel_beam.py",
                     "triton_parallel.py", "tomography_model.py")

# ── the region definitions and GPU-health machinery, COPIED from mg9/mg11/mg12 ─
REGIONS = ("forward_funnel", "back_funnel", "prior", "halo", "band_reduce")
REGIONS_ABSENT_AT_N1 = ("band_reduce",)
REGIONS_HOST_ONLY_AT_N1 = ("halo",)
MAX_EVENT_PAIRS = int(os.environ.get("MG13_MAX_EVENT_PAIRS", "400000"))

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


# ══════════════════════════════════════════════════════════════════════════════
# THE CANDIDATE: the sorted channel reduction, as a drop-in for the shipped fan
# ══════════════════════════════════════════════════════════════════════════════
# Everything in this block runs inside the worker process only.  It imports torch
# lazily, because the parent process never does.

SORT_VIEW_CHUNK_ENV = int(os.environ.get("MG13_SORT_VIEW_CHUNK", "0"))
# Whether the sort is asked to be stable.  Default off: an unstable sort is the
# cheaper one and is what a per-call form would ship with.  Stable makes the
# reduction bit-reproducible across chunk sizes and across runs, and mbirjax's
# ``lax.sort_key_val`` is stable, so the switch exists for a reader who wants
# that property and is willing to pay whatever it costs here.
SORT_STABLE_ENV = os.environ.get("MG13_SORT_STABLE", "0") == "1"

# What the sorted fan records about itself, read out onto every row.
SORTED_STATE = dict(calls=0, view_chunk_used=None, stable_sort=None,
                    transient_bytes=None, reduction_api=None,
                    segment_reduce_error=None, shapes={})
# One-element list rather than a bare name so the reduction can flip it without
# a global statement: True until torch.segment_reduce refuses a call, then False
# for the rest of the process (see _segment_sum).
_SEGMENT_REDUCE_OK = [True]


def _sorted_reduction_api():
    """Which torch reduction the sorted form uses, decided once and recorded.

    ``torch.segment_reduce`` is present in torch 2.13 (verified locally against
    the same 2.13.0 build the cluster environment carries) and takes the segment
    LENGTHS rather than jax's arbitrary segment ids, which is why the
    implementation below turns the sorted channel indices into a length per
    output row with ``bincount``: one length per accumulator row, zero for the
    rows nothing landed on, summing to the contribution count by construction.
    Empty segments reduce to 0.0 for 'sum', with and without an explicit
    ``initial``; the call passes ``initial=0.0`` anyway so the answer does not
    depend on a default.

    PRESENCE IS NOT USABILITY.  The operator is a beta one and is not
    implemented for every backend: it raises NotImplementedError on MPS, which
    is what the local smoke on this developer's Mac runs into.  So the answer
    this function gives is provisional -- it says which reduction will be TRIED
    -- and :func:`_segment_sum` overwrites it the moment a call refuses.

    The cumsum fallback reduces the same runs with a cumulative sum and a
    difference at the run boundaries, then writes each run once: the same
    values, one more full-size transient.  Which one actually ran is on every
    row, beside the error that caused the switch when there was one.
    """
    import torch

    return "torch.segment_reduce" if hasattr(torch, "segment_reduce") \
        else "cumsum run-difference (torch.segment_reduce absent)"


def _default_sort_view_chunk(vb, num_pixels, num_cols, taps):
    """How many views one sort covers, when MG13_SORT_VIEW_CHUNK is 0.

    WHY CHUNKING IS LEGAL HERE, and exactly how far the claim goes.  A
    contribution's accumulator row is ``view * num_channels + channel``, so two
    contributions from DIFFERENT views can never share a row.  Splitting the sort
    along the view axis therefore splits the runs exactly where no run exists:
    every run the whole-batch sort would form lies entirely inside one view's
    slice, and the chunked sort forms the same run from the same contributions.
    What chunking does NOT guarantee is the ORDER of equal keys inside a run,
    because ``torch.sort`` is not required to be stable -- so the two forms sum
    the same numbers and may sum them in a different order, which makes them
    equal to float rounding rather than bit for bit.  (``MG13_SORT_STABLE=1``
    asks for a stable sort, which does make them bit-identical, at whatever a
    stable sort costs.)  The bench measures the chunked form against the
    unchunked one directly, so the claim is a reading rather than an argument.

    WHY IT IS ON BY DEFAULT.  The sorted form holds all psf taps' contributions
    at once, where the scatter holds one tap's at a time, so its transient is
    psf_width times the scatter's: 3 x 64 x 8192 x 1008 x 4 bytes is 6.3 GB at
    the four-device launch shape.  The chunk is derived from the library's OWN
    per-batch transient budget (``Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES``)
    so that the candidate is charged the same ceiling the driver already charges
    the bodies, rather than a number invented here.  The realized chunk and the
    realized transient are recorded on every row: a chunk below the view batch
    means the sorted form ran several sorts per body call, and that is part of
    what it costs, not something hidden.
    """
    from mbirtorch.projectors import Projectors

    if SORT_VIEW_CHUNK_ENV:
        return max(1, min(int(vb), SORT_VIEW_CHUNK_ENV))
    budget = int(Projectors.VIEW_BATCH_TRANSIENT_BUDGET_BYTES)
    per_view = max(1, taps * int(num_pixels) * int(num_cols) * 4)
    return max(1, min(int(vb), budget // per_view))


def fan_forward_batch_sorted(hfan_data, values, num_channels, psf_radius,
                             view_chunk=None, record=True, stable=None):
    """The candidate reduction: a drop-in for
    :func:`mbirtorch.horizontal_fan.fan_forward_batch`.

    Same arguments, same (Vb, num_channels, num_cols) channel-major return, same
    weights from the same :func:`tap_weights` -- the ONLY thing that changes is
    how the weighted rows reach the accumulator.

    The shipped fan loops over the psf taps and calls ``index_add_`` once per
    tap, so every pixel landing on a channel is an atomic add colliding with
    every other pixel landing there.  This form instead:
      1. builds all taps' (accumulator row, weight) pairs at once,
      2. sorts them by accumulator row,
      3. gathers the weighted value rows in that sorted order,
      4. sums each run of equal rows and writes each row once.
    That is mbirjax's shipped shape (``lax.sort_key_val`` then ``segment_sum``,
    projectors.py), written with the torch ops that correspond to those.

    THE FLAT LAYOUT.  Contributions are laid out tap-major: entry
    ``k * (Vb * P) + v * P + p`` is tap k of pixel p in view v.  So a sorted
    entry's weight is ``flat_A[order]`` and its VALUE row is ``order % P`` when
    the values are shared across views (parallel beam's (P, cols) voxel
    cylinders) and ``order % (Vb * P)`` when they are per view (a two-fan
    geometry's (Vb, P, cols)).  Gathering in sorted order rather than sorting a
    materialized product is what keeps this to one full-size transient.

    SUMMATION ORDER.  A channel's contributions are summed in sorted-index
    order here and in atomic-arrival order in the shipped fan, so the two agree
    to float rounding and not bit for bit.  That is the whole reason this job
    measures a value distance.
    """
    import torch

    from mbirtorch.horizontal_fan import tap_weights

    n_p, centers, W_p_c, weight_scale = hfan_data
    centers = centers.to(torch.int64)
    vb, num_pixels = n_p.shape
    num_cols = values.shape[-1]
    dev = values.device
    taps = 2 * psf_radius + 1
    per_view = values.dim() == 3

    if view_chunk is None:
        view_chunk = _default_sort_view_chunk(vb, num_pixels, num_cols, taps)
    chunk = max(1, min(int(vb), int(view_chunk)))
    stable = SORT_STABLE_ENV if stable is None else bool(stable)

    # The two per-view scalars of the parallel contract are (Vb, 1); a geometry
    # could pass them per pixel.  Broadcasting to (Vb, P) FIRST makes the view
    # slice below unambiguous and costs nothing: broadcast_to returns a
    # zero-stride view, so no bytes are copied and tap_weights reads it exactly
    # as it reads the original.
    W_full = torch.broadcast_to(W_p_c, (vb, num_pixels))
    S_full = torch.broadcast_to(weight_scale, (vb, num_pixels))

    acc = torch.zeros((vb * num_channels, num_cols), dtype=torch.float32,
                      device=dev)
    transient = 0
    for v0 in range(0, vb, chunk):
        v1 = min(v0 + chunk, vb)
        nv = v1 - v0
        row_base = torch.arange(v0, v1, device=dev)[:, None] * num_channels
        idx_parts, weight_parts = [], []
        for offset in range(-psf_radius, psf_radius + 1):
            A, n = tap_weights(n_p[v0:v1], centers[v0:v1] + offset,
                               W_full[v0:v1], S_full[v0:v1], num_channels)
            idx_parts.append((row_base + n).reshape(-1))
            weight_parts.append(A.reshape(-1))
        flat_idx = torch.cat(idx_parts)
        flat_A = torch.cat(weight_parts)
        sorted_idx, order = torch.sort(flat_idx, stable=stable)
        if per_view:
            rows = order % (nv * num_pixels)
            src = values[v0:v1].reshape(nv * num_pixels, num_cols)
        else:
            rows = order % num_pixels
            src = values
        updates = flat_A[order].unsqueeze(-1) * src[rows]
        transient = max(transient, updates.numel() * 4)
        # One length per accumulator row of this chunk, zero where nothing
        # landed; the lengths sum to the contribution count by construction, so
        # the reduction below is exact rather than a subset.
        lengths = torch.bincount(sorted_idx - v0 * num_channels,
                                 minlength=nv * num_channels)
        acc[v0 * num_channels:v1 * num_channels] = _segment_sum(
            updates, lengths, nv * num_channels, sorted_idx - v0 * num_channels)

    if record:
        SORTED_STATE["calls"] += 1
        SORTED_STATE["view_chunk_used"] = chunk
        SORTED_STATE["stable_sort"] = stable
        SORTED_STATE["transient_bytes"] = transient
        key = f"{vb}x{num_pixels}x{num_cols}"
        SORTED_STATE["shapes"][key] = SORTED_STATE["shapes"].get(key, 0) + 1
    return acc.view(vb, num_channels, num_cols)


def _segment_sum(updates, lengths, num_rows, sorted_idx):
    """Sum each run of equal rows: ``torch.segment_reduce`` where it works, the
    cumsum run-difference otherwise (see :func:`_sorted_reduction_api`).

    THE FALLBACK IS TAKEN ON A RAISE, not only on a missing attribute.  The
    operator exists in torch 2.13 but is not implemented for every backend --
    on MPS it raises NotImplementedError -- so presence is not the same as
    usability, and the only reliable test is the call.  The first raise switches
    the process over permanently and records the reason on the row, so the
    exception is paid once and the read-out says which reduction produced the
    numbers.
    """
    import torch

    if _SEGMENT_REDUCE_OK[0]:
        try:
            return torch.segment_reduce(updates, "sum", lengths=lengths, axis=0,
                                        unsafe=True, initial=0.0)
        except Exception as exc:                                  # noqa: BLE001
            _SEGMENT_REDUCE_OK[0] = False
            SORTED_STATE["reduction_api"] = (
                "cumsum run-difference (torch.segment_reduce raised)")
            SORTED_STATE["segment_reduce_error"] = str(exc)[:300]
    # The fallback: a cumulative sum down the sorted axis, then the difference
    # across each run's boundaries, then one write per run.  Same values, one
    # more full-size transient than the primary path.
    out = torch.zeros((num_rows, updates.shape[1]), dtype=updates.dtype,
                      device=updates.device)
    if updates.shape[0] == 0:
        return out
    csum = torch.cat([torch.zeros((1, updates.shape[1]), dtype=updates.dtype,
                                  device=updates.device),
                      updates.cumsum(0)], dim=0)
    ends = torch.nonzero(
        torch.cat([sorted_idx[1:] != sorted_idx[:-1],
                   torch.ones(1, dtype=torch.bool, device=updates.device)]),
        as_tuple=False).reshape(-1)
    starts = torch.cat([torch.zeros(1, dtype=ends.dtype, device=ends.device),
                        ends[:-1] + 1])
    out.index_copy_(0, sorted_idx[ends], csum[ends + 1] - csum[starts])
    return out


def install_sorted_fan():
    """Rebind the name the forward body calls, and prove the rebinding took.

    ``parallel_beam.py`` does ``from .horizontal_fan import fan_forward_batch``,
    so the body resolves the name out of the PARALLEL_BEAM module's globals at
    call time.  That attribute is the patch point.

    ONLY PARALLEL BEAM IS PATCHED, and that is deliberate.  Four geometry
    modules bind this name -- parallel, cone, translation and multiaxis -- and
    item 13 is scoped to parallel on measured need (forward remedy memo, section
    8.5: cone runs at 0.98x of jax at one device, parallel runs 14.4 s above it).
    Rebinding ``horizontal_fan``'s own attribute as well would leave those three
    importing the shipped function anyway, since they bound it at import time,
    so the only thing it would change is what a LATER importer sees.  Leaving it
    alone keeps the process in one state that is easy to state: parallel's
    forward runs the candidate and nothing else in the library does.

    CALLED BEFORE THE MODEL IS BUILT.  The torch bodies are compiled, and the
    compiled artifact traces the fan it saw at its first call.  Patching first
    means the sorted reduction is what gets traced, rather than the shipped one
    being traced and then a global-identity guard firing a recompile in the
    middle of a timed pass.
    """
    from mbirtorch import horizontal_fan, parallel_beam

    original = horizontal_fan.fan_forward_batch
    SORTED_STATE["reduction_api"] = _sorted_reduction_api()
    parallel_beam.fan_forward_batch = fan_forward_batch_sorted
    return dict(
        original_name=getattr(original, "__qualname__", str(original)),
        installed_name=fan_forward_batch_sorted.__qualname__,
        reduction_api=SORTED_STATE["reduction_api"],
        sort_view_chunk_env=SORT_VIEW_CHUNK_ENV)


def fan_identity():
    """What the forward body would actually call, by name and by identity.

    Read on every arm, patched or not: on a scatter arm the two must be the same
    object, and on a sorted arm the parallel_beam name must be this file's
    function.  A patch that silently did not take would otherwise report a very
    tidy 'no change'."""
    from mbirtorch import horizontal_fan, parallel_beam

    live = parallel_beam.fan_forward_batch
    return dict(
        body_calls=getattr(live, "__qualname__", str(live)),
        body_calls_module=getattr(live, "__module__", None),
        is_library_original=(live is horizontal_fan.fan_forward_batch
                             and live.__module__ == "mbirtorch.horizontal_fan"),
        is_this_files_sorted=(live is fan_forward_batch_sorted))


# ══════════════════════════════════════════════════════════════════════════════


# ── staged-artifact mechanics (mg5's / mg9's / mg11's / mg12's md5 discipline) ─
def _sino_path(cell):
    return os.path.join(RESULTS_DIR, f"_mg13_sino_{GEOMETRY}_{cell[0]}.npy")


def _md5_path(cell):
    return _sino_path(cell) + ".md5"


def _sample_path(arm_id, index):
    return os.path.join(RESULTS_DIR, f"_mg13_sample_{arm_id}_p{index}.npy")


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
    """Per-GPU clocks, temperatures and active throttle reasons (mg12's)."""
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
    """Whether any device ran hot, throttled, or at a depressed clock.

    THE THROTTLE RULE, mg9's and kept verbatim: a sw_power_cap at a normal
    temperature is what an H100 does when it is working, so such a row is
    RECORDED and KEPT rather than discarded.  Nothing here drops a measurement;
    the flag travels with it."""
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


# ── the arm plan ──────────────────────────────────────────────────────────────
def device_counts():
    return list(SMOKE_COUNTS if SMOKE else DEVICE_COUNTS)


def build_arms():
    """Every arm, in job order.  DEVICE-COUNT-MAJOR: the three bodies of one
    device count run back-to-back, so a truncated job yields a complete
    three-way comparison and node drift lands inside a comparison rather than
    between its columns."""
    arms = []
    for count in device_counts():
        for body in BODIES:
            arms.append(dict(n_dev=count, body=body,
                             count_token=f"n{count}",
                             token=f"n{count}-{body}"))
    return arms


ARMS = build_arms()


def selected_arms():
    """The arms to run, narrowed by MG13_COUNTS then by MG13_ARMS."""
    chosen = list(ARMS)
    raw = os.environ.get("MG13_COUNTS", "").strip()
    if raw:
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if int(token) not in device_counts():
                raise ValueError(f"MG13_COUNTS: {token!r} is not one of "
                                 f"{device_counts()}")
            wanted.add(int(token))
        chosen = [a for a in chosen if a["n_dev"] in wanted]
    raw = os.environ.get("MG13_ARMS", "").strip()
    if raw:
        known = {a["token"] for a in ARMS}
        wanted = set()
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            if token not in known:
                raise ValueError(f"MG13_ARMS: {token!r} is not one of "
                                 f"{sorted(known)}")
            wanted.add(token)
        chosen = [a for a in chosen if a["token"] in wanted]
    if not chosen:
        raise ValueError("no arms selected")
    return chosen


def tree_root():
    """The directory the mbirtorch package sits in, from the environment.
    Required: this file will not guess a tree, because guessing wrong is the one
    failure that would look like a clean result."""
    root = os.environ.get(TREE_ENV, "").strip()
    if not root:
        raise RuntimeError(
            f"{TREE_ENV} is not set, so no arm has a tree to import.  The "
            f"sbatch clones the pinned tree and exports this; running this file "
            f"by hand needs the same export.")
    return os.path.abspath(root)


def cell_for():
    return SMOKE_CELL if SMOKE else CELL


def num_slices_for(cell):
    """The reconstruction's slice count at this cell; the worker asserts this
    against the model's own recon_shape before it is used for anything."""
    return int(cell[1])


# ── THE PROVENANCE WITNESS ────────────────────────────────────────────────────
def tree_fingerprint():
    """Where the mbirtorch this process imported came from, and what its bytes
    are.  Two independent readings: the PATH says the package sits inside the
    root the runner exported (catching a PYTHONPATH that did not take, including
    an editable install winning through a meta-path finder), and the BYTES are
    sha256 prefixes of the files read from beside that path (catching a tree in
    the right place with the wrong contents).  The summary then compares the byte
    readings across arms, where all six must be identical -- this job runs ONE
    tree, so a difference means an arm imported something else."""
    import mbirtorch

    package = os.path.dirname(os.path.realpath(mbirtorch.__file__))
    root = os.path.realpath(tree_root())
    inside = package.startswith(root + os.sep)
    digests = {}
    for name in FINGERPRINT_FILES:
        path = os.path.join(package, name)
        digests[name] = _sha256_12(path) if os.path.exists(path) else None
    return dict(expected_root=root, package_dir=package,
                mbirtorch_file=os.path.realpath(mbirtorch.__file__),
                inside_expected_root=bool(inside), sha256_12=digests)


def compile_diagnostics():
    """What the compiler did, read out of torch's own registries rather than
    inferred from a time.

    ``projectors._COMPILE_ERRORS`` is the library's own record of a body whose
    compile failed and was permanently rebound to eager -- if the sorted
    reduction cannot be compiled, that is where it shows, and every number in
    that arm has to be read as an eager number.  The dynamo counters carry the
    graph-break tally: ``torch.segment_reduce`` is a beta operator, and a break
    around it would split the compiled body in two.  Neither reading gates an
    arm; both are recorded so a surprising time has somewhere to be attributed
    before it is attributed to the algorithm."""
    out = dict(compile_errors=None, graph_breaks=None, unique_graph_breaks=None)
    try:
        from mbirtorch import projectors

        out["compile_errors"] = {k: str(v)[:400] for k, v
                                 in dict(projectors._COMPILE_ERRORS).items()}
    except Exception as exc:                                      # noqa: BLE001
        out["compile_errors"] = f"unreadable: {exc}"
    try:
        from torch._dynamo.utils import counters

        breaks = dict(counters.get("graph_break", {}))
        out["graph_breaks"] = int(sum(breaks.values()))
        out["unique_graph_breaks"] = sorted(breaks)[:20]
    except Exception as exc:                                      # noqa: BLE001
        out["graph_breaks"] = f"unreadable: {exc}"
    return out


# ── INSTRUMENT 0: per-region host walls and device spans (mg9's, unchanged) ───
class RegionInstrument:
    """Per-region host walls and per-device event spans, recorded from the
    reconstruction loop's calling thread.  Copied from mg12_stream_gate.py, which
    copied it from mg11, which copied it from mg10, which copied it from mg9,
    which copied it from mg1_readout.py without change -- so mg13's forward
    bracket IS mg9's, mg11's and mg12's forward bracket, and an arm here is
    comparable with theirs.

    CUDA path: for each device in the region's placement a start and an end event
    are CREATED AND RECORDED inside ``with torch.cuda.device(dev)``, on that
    device's current stream, which is the compute stream the projections run on.
    The end event is recorded AFTER the call returns, so it queues behind
    everything the call enqueued.  Elapsed times are read only in
    :meth:`finish`, after a per-device synchronize.

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
    instance attributes and the sharding seams as module attributes, all of which
    the engine looks up at call time.  This is observation, not a patch: every
    wrapper calls the original and returns its value unchanged."""
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


# ── INSTRUMENT 1: busy time, per device, per body call (mg9's / mg11's / mg12's)
class BusyProbe:
    """Times each individual forward projection BODY call, in buckets keyed by
    DEVICE POSITION -- never by object identity, because every entry of
    ``_fwd_body_per_dev`` can be the same object.

    Busy divided by the call count is the PER-LAUNCH time, which is the reading
    the n=1 verdict line is drawn from.

    WHY THIS IS THE RIGHT READING OF "THE LAUNCHES ALONE" ON ALL THREE BODIES.
    The start event is recorded on the device's compute stream immediately before
    the body runs, and a stream reaches its work in order, so anything the driver
    enqueued earlier -- including the waits the column gather's copy stream
    imposes at four devices -- has finished by the time the start event is
    reached.  The span that follows is the projection and not the transfer, which
    is what lets the busy column be compared across the three bodies.

    THE SHAPE HISTOGRAMS ARE A WITNESS, not decoration.  A forward body's first
    positional argument is the values block, of shape (pixels, columns).  Its
    column count is what the per-slice number divides by, and its pixel count is
    what the collision ratio is computed from -- so the two numbers the whole
    question turns on are recorded per launch rather than assumed from the plan.
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

        wrapped._mg13_dev_index = dev_index
        wrapped._mg13_wrapped_body = body
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


# ── INSTRUMENT 2: which forward SHAPE ran (mg11's / mg12's gather counters) ───
class GatherProbe:
    """Counts what the forward's transfer layer did, so an arm's row states which
    forward shape produced it rather than assuming the default held.

        gather_calls     entries into the column gather.  Zero at one device (the
                         trivial placement never enters the sharded driver at
                         all) and positive at four.
        broadcast_calls  entries into the BANDED fan-out.  Zero at every device
                         count here: a positive count would mean the banded walk
                         ran and the four-device arms are not the shape the
                         question is about.
    The cylinder height and width histograms are the shape witness: height is the
    whole device-form slice axis at every gather, width is the pixel batch with
    one short tail batch per pass.

    The copies are NOT timed one by one, for mg12's reason: putting extra event
    records and a shared lock into the copy path changes the thing being measured.
    The lock below is taken once per batch and only around integer updates.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.gather_calls = 0
        self.broadcast_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}
        self.cyl_width = {}

    def drain(self):
        record = dict(gather_calls=int(self.gather_calls),
                      broadcast_calls=int(self.broadcast_calls),
                      gather_host_wall_s=float(self.gather_host_s),
                      cyl_height_hist={str(k): v for k, v
                                       in sorted(self.cyl_height.items())},
                      cyl_width_hist={str(k): v for k, v
                                      in sorted(self.cyl_width.items())})
        self.gather_calls = 0
        self.broadcast_calls = 0
        self.gather_host_s = 0.0
        self.cyl_height = {}
        self.cyl_width = {}
        return record


def attach_forward_probes(model, torch_module, cuda, max_pairs):
    """Install the busy probe and the gather probe; return
    ``(busy, gathers, verify, detach, observed)``.

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

    busy = BusyProbe(torch_module, cuda, n_dev,
                     max(1, max_pairs // max(1, 2 * n_dev)))
    originals = list(bodies)
    wrappers = []
    for index, body in enumerate(originals):
        wrapper = busy.wrap(body, index, devices[index])
        # The driver chooses the view batch by reading this attribute OFF THE
        # BODY, so a wrapper that lost it would silently move a kernel onto the
        # torch batching rule and change what "busy" means.
        if getattr(body, "_view_batch_cost", None) is not \
                getattr(wrapper, "_view_batch_cost", None):
            raise RuntimeError(
                "the body wrapper did not carry _view_batch_cost through; the "
                "realized view batch would change and this arm would not be "
                "comparable with the other two bodies")
        wrappers.append(wrapper)
    # Mutated IN PLACE rather than rebound, so any other reference to the list
    # sees the wrappers too.
    for index, wrapper in enumerate(wrappers):
        bodies[index] = wrapper

    # The realized view batch, observed PER DEVICE.  Recorded because a
    # per-launch time that moved between bodies has to be readable: a changed
    # view batch means the launches are different sizes, which is a different
    # finding from a changed reduction.
    observed = {}
    observed_lock = threading.Lock()
    original_effective = pf._effective_view_batch

    def effective_view_batch(body, num_pixels, band_cols, args):
        value = original_effective(body, num_pixels, band_cols, args)
        index = getattr(body, "_mg13_dev_index", None)
        key = (f"fwd_dev{index}" if index is not None
               else "back_body_device_not_recoverable")
        with observed_lock:
            bucket = observed.setdefault(key, {})
            bucket[int(value)] = bucket.get(int(value), 0) + 1
        return value

    pf._effective_view_batch = effective_view_batch

    gathers = GatherProbe()
    original_gather = getattr(_sharding, "gather_column_band", None)
    original_broadcast = _sharding.broadcast_band_to_views
    restore = [("broadcast_band_to_views", original_broadcast)]

    def _record_shape(cyl):
        if cyl is None:
            return
        height = int(cyl.shape[-1])
        width = int(cyl.shape[0])
        gathers.cyl_height[height] = gathers.cyl_height.get(height, 0) + 1
        gathers.cyl_width[width] = gathers.cyl_width.get(width, 0) + 1

    if original_gather is not None:
        def gather_column_band(shard_tensors, p0, p1, target, dev2dev_safe=True):
            """The gather's inner primitive, wrapped exactly as mg11 and mg12
            wrap it.  The RETURNED cylinder is measured rather than the
            arguments, because the claim being witnessed is about what was
            assembled."""
            host0 = time.perf_counter()
            cyl = None
            try:
                cyl = original_gather(shard_tensors, p0, p1, target,
                                      dev2dev_safe=dev2dev_safe)
                return cyl
            finally:
                span = time.perf_counter() - host0
                with gathers.lock:
                    gathers.gather_calls += 1
                    gathers.gather_host_s += span
                    _record_shape(cyl)

        _sharding.gather_column_band = gather_column_band
        restore.append(("gather_column_band", original_gather))

    def broadcast_band_to_views(band, view_owners, dev2dev_safe=True):
        """The BANDED fan-out.  Every multi-device arm here runs the column
        gather, so this must never be entered; it is wrapped so that 'never' is
        measured rather than assumed."""
        with gathers.lock:
            gathers.broadcast_calls += 1
        return original_broadcast(band, view_owners, dev2dev_safe=dev2dev_safe)

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
                _sharding.broadcast_band_to_views is broadcast_band_to_views))

    def detach():
        for index, body in enumerate(originals):
            bodies[index] = body
        pf._effective_view_batch = original_effective
        for name, original in restore:
            setattr(_sharding, name, original)

    return busy, gathers, verify, detach, observed


# ── the model, and the knobs an arm sets ──────────────────────────────────────
def _build_torch_model(cell, n_dev, cpu_devices=None):
    """The model, with the device count pinned.

    THE PIN.  ``configure_devices`` is the documented way a caller names the
    layout, and calling it takes the choice out of the library's hands: the
    automatic device-count search never runs, and neither do the widening speed
    floors that order it.  A second effect matters to the instruments: with the
    layout fixed before the first reconstruction there is no device-count settle
    to rebuild the projector object the probes attach to.
    """
    import numpy as np

    import mbirtorch

    num_views = cell[0]
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirtorch.ParallelBeamModel(cell, angles)
    if cpu_devices is not None:
        model.configure_devices(devices=list(cpu_devices))
    elif n_dev:
        model.configure_devices(num_devices=int(n_dev))
    model.set_params(no_warning=True, verbose=0)
    return model


def read_switch(model):
    """The library's own answer to which forward SHAPE runs, plus the inputs it
    read to get there.  Nothing here re-derives the rule -- the resolver is
    called.  At one device the sharded driver is never entered at all, so the
    resolver's answer is about the multi-device shape only."""
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
        rows_track_slices=bool(getattr(model, "rows_track_slices", False)))


def kernel_state():
    """What the library says about the Triton kernels in THIS process: the kill
    switch as the arm set it, the probe's own answer and its reason."""
    from mbirtorch import kernel_availability

    usable, reason = kernel_availability.triton_available()
    return dict(kill_switch=os.environ.get(KILL_SWITCH),
                triton_usable=bool(usable), triton_reason=str(reason))


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


def _hist_mean(hist):
    """The count-weighted mean of a {value: how many} histogram, with string keys
    as the jsonl carries them.  ``None`` for an empty histogram."""
    total = weight = 0
    for key, count in (hist or {}).items():
        total += int(key) * int(count)
        weight += int(count)
    return (total / weight) if weight else None


def collision_ratio(psf_width, num_pixels, num_channels):
    """The mean number of contributions landing on one detector channel in one
    call: ``psf_width * num_pixels / num_channels``.  This is the variable
    mbirjax's own selection rule for the sorted form is written in, so it is
    computed here for every launch shape rather than being reasoned about."""
    if not num_pixels or not num_channels:
        return None
    return float(psf_width) * float(num_pixels) / float(num_channels)


# ══════════════════════════════════════════════════════════════════════════════
# THE ISOLATED BENCH: the two reductions and the kernel, on identical inputs
# ══════════════════════════════════════════════════════════════════════════════
BENCH_WARMUP = 3
BENCH_REPEATS = 5
# A bench point is skipped when the sorted form's single transient would exceed
# this fraction of the device's total memory.  The sorted form holds all psf
# taps at once where the scatter holds one, so its transient is psf_width times
# the scatter's, and at the widest pixel counts that is tens of gigabytes.  A
# skipped point is RECORDED with its arithmetic, because "it did not fit" is a
# reading about the candidate, not a gap in the data.
BENCH_MAX_TRANSIENT_FRAC = 0.25


class _fan_bound:
    """Bind ``parallel_beam.fan_forward_batch`` to one implementation for the
    duration of a block, then put back what was there.

    The bench times the LIBRARY'S OWN forward body rather than a copy of it, so
    the geometry chain, the permute and the allocation pattern are identical for
    every arm of the bench and the reduction is the only thing that varies.
    Rebinding the module attribute is how that single variable is set -- the same
    mechanism the sorted arm uses, applied for a few launches at a time.
    """

    def __init__(self, impl):
        self.impl = impl

    def __enter__(self):
        from mbirtorch import parallel_beam

        self.previous = parallel_beam.fan_forward_batch
        parallel_beam.fan_forward_batch = self.impl
        return self

    def __exit__(self, *exc):
        from mbirtorch import parallel_beam

        parallel_beam.fan_forward_batch = self.previous
        return False


def _time_launches(torch_module, cuda, call, warmup=BENCH_WARMUP,
                   repeats=BENCH_REPEATS):
    """Median and minimum milliseconds for one call, timed launch by launch.

    On CUDA each timed launch gets its own event pair and the device is
    synchronized once at the end, so the reading is device time and the host's
    queueing does not enter it.  The minimum is reported beside the median
    because a single launch can carry a scheduling artifact and the minimum is
    the cleanest estimate of the work itself."""
    out = None
    for _ in range(warmup):
        out = call()
    if not cuda:
        spans = []
        for _ in range(repeats):
            start = time.perf_counter()
            out = call()
            spans.append((time.perf_counter() - start) * 1e3)
        return dict(median_ms=statistics.median(spans), min_ms=min(spans),
                    spans_ms=spans), out
    torch_module.cuda.synchronize()
    pairs = []
    for _ in range(repeats):
        start = torch_module.cuda.Event(enable_timing=True)
        end = torch_module.cuda.Event(enable_timing=True)
        start.record()
        out = call()
        end.record()
        pairs.append((start, end))
    torch_module.cuda.synchronize()
    spans = [s.elapsed_time(e) for s, e in pairs]
    return dict(median_ms=statistics.median(spans), min_ms=min(spans),
                spans_ms=spans), out


def bench_point(model, torch_module, cuda, num_pixels, num_cols, args,
                bodies_available, total_memory):
    """One launch shape, every available body, on ONE set of inputs.

    The view batch is not chosen here: it is read from the library's own
    ``view_batch_charge``, the same rule the driver applies, so a bench launch is
    the shape a production launch would be at this pixel count and column count.
    """
    from mbirtorch.horizontal_fan import fan_forward_batch
    from mbirtorch.parallel_beam import _parallel_forward_view_batch

    pf = model.projector_functions
    psf_radius = int(args["psf_radius"])
    taps = 2 * psf_radius + 1
    num_channels = int(args["num_channels"])

    # The pixel subset: a stride through the model's own ROR-masked indices, so
    # the geometry the bench projects is the geometry the model reconstructs.
    full = model.full_indices_device()
    step = max(1, int(full.shape[0]) // max(1, num_pixels))
    pixel_indices = full[::step][:num_pixels].contiguous()
    realized_pixels = int(pixel_indices.shape[0])

    # The view batch the driver would use for this shape, from the library's own
    # cost model.  The torch bodies and a kernel body charge differently, so the
    # charge is read for the body that will run; the torch body's charge is the
    # one the two reductions share and is what the bench reports.
    view_batch, bytes_per_view = pf.view_batch_charge(
        _parallel_forward_view_batch, realized_pixels, num_cols, args)
    sorted_transient = taps * view_batch * realized_pixels * num_cols * 4
    scatter_transient = view_batch * realized_pixels * num_cols * 4
    point = dict(
        num_pixels=realized_pixels, num_cols=num_cols, view_batch=view_batch,
        psf_width=taps, num_channels=num_channels,
        collision_ratio=collision_ratio(taps, realized_pixels, num_channels),
        torch_body_bytes_per_view=int(bytes_per_view),
        scatter_transient_bytes=int(scatter_transient),
        sorted_transient_bytes=int(sorted_transient),
        sorted_over_scatter_transient=taps, results={}, skipped=None)
    if total_memory and sorted_transient > BENCH_MAX_TRANSIENT_FRAC * total_memory:
        point["skipped"] = (
            f"the sorted form's single transient here is "
            f"{sorted_transient / 2 ** 30:.1f} GB, over "
            f"{BENCH_MAX_TRANSIENT_FRAC:.0%} of the device's "
            f"{total_memory / 2 ** 30:.0f} GB; the point is recorded and not run")
        return point

    generator = torch_module.Generator(device=model.torch_device)
    generator.manual_seed(VCD_SEED)
    values = torch_module.rand((realized_pixels, num_cols),
                               generator=generator, dtype=torch_module.float32,
                               device=model.torch_device)
    view_params = pf._view_params_per_dev[0][:view_batch]

    def run(body):
        return body(values, pixel_indices, view_params, slice_start=0,
                    plan=None, **args)

    outputs = {}
    # The two reductions, through the SAME library body, one variable between
    # them: which function the body's ``fan_forward_batch`` name resolves to.
    variants = [("torch_scatter", fan_forward_batch),
                ("torch_sorted", fan_forward_batch_sorted)]
    if view_batch > 1:
        # The chunked sorted form, forced to one view per sort.  Its value must
        # equal the unchunked sorted form's EXACTLY (different views never share
        # an accumulator row), and its time is what the memory bound costs.
        variants.append(
            ("torch_sorted_chunk1",
             functools.partial(fan_forward_batch_sorted, view_chunk=1,
                               record=False)))
    for name, impl in variants:
        with _fan_bound(impl):
            timing, out = _time_launches(torch_module, cuda,
                                         lambda: run(_parallel_forward_view_batch))
        point["results"][name] = timing
        outputs[name] = out
    if "triton" in bodies_available:
        from mbirtorch.triton_parallel import _parallel_forward_view_batch_triton

        timing, out = _time_launches(
            torch_module, cuda,
            lambda: run(_parallel_forward_view_batch_triton))
        point["results"]["triton"] = timing
        outputs["triton"] = out

    # The value distances, on these inputs, in this process: the clean reading.
    def distance(candidate, reference):
        """Distances in float64 ON THE HOST.  The bit-identical test is taken on
        the device tensors, where it is exact; the norms move to the host first
        because float64 is not available on every backend the smoke runs on, and
        a distance that silently changed precision with the backend would be a
        poor thing to read a tolerance against.  This is outside every timed
        region."""
        identical = bool(torch_module.equal(candidate, reference))
        # The move and the cast are two steps on purpose: MPS refuses a float64
        # cast even when the same call is moving the tensor off the device.
        a = candidate.detach().cpu().double()
        b = reference.detach().cpu().double()
        denom = float(torch_module.linalg.vector_norm(b))
        peak = float(b.abs().max())
        diff = a - b
        return dict(
            rel_l2=(float(torch_module.linalg.vector_norm(diff)) / denom
                    if denom else None),
            max_abs=float(diff.abs().max()),
            max_rel_of_peak=(float(diff.abs().max()) / peak if peak else None),
            bit_identical=identical)

    reference = outputs["torch_scatter"]
    point["value"] = {}
    for name, out in outputs.items():
        if name == "torch_scatter":
            continue
        point["value"][name] = distance(out, reference)
    # The chunked sorted form against the UNCHUNKED one, which is the claim the
    # view chunk rests on: chunking splits the sort where no run crosses, so the
    # same numbers reach each row.  Bit-identical only when the sort is stable
    # (see the chunk helper's docstring); equal to float rounding otherwise.
    if "torch_sorted_chunk1" in outputs:
        point["value"]["chunk1_against_unchunked_sorted"] = distance(
            outputs["torch_sorted_chunk1"], outputs["torch_sorted"])
    del outputs, values
    if cuda:
        torch_module.cuda.empty_cache()
    for name, timing in point["results"].items():
        timing["ms_per_slice"] = timing["median_ms"] / num_cols
    point["ms_per_slice_precedent"] = PRECEDENT_MS_PER_SLICE
    return point


def isolated_bench(model, torch_module, cuda, bodies_available):
    """The eager per-launch bench: a ladder of pixel counts at the full slice
    width, plus one narrow-column point.

    THE LADDER IS THE POINT.  The mean collisions per channel is
    ``psf_width * num_pixels / num_channels``, so sweeping the pixel count sweeps
    the one variable the sorted form's value is known to depend on, from the
    column gather's own batch size up to the whole ROR.  A single point would
    give a ratio with nothing to read it against.

    THE NARROW-COLUMN POINT is there because the column count is the OTHER
    variable in mbirjax's selection rule -- its sorted form is withheld above
    1280 columns, and the full-width column gather feeds 1008 here, just under
    that ceiling.  One point at a quarter of the width says whether the column
    count matters on this substrate at all.
    """
    args = model._view_batch_args()
    num_slices = int(model.get_params("recon_shape")[2])
    full_pixels = int(model.full_index_count())
    total_memory = 0
    if cuda:
        total_memory = int(
            torch_module.cuda.get_device_properties(
                model.torch_device).total_memory)

    ladder = []
    for target in (4096, 8192, 32768, max(1, full_pixels // 8), full_pixels):
        target = min(int(target), full_pixels)
        if target not in ladder:
            ladder.append(target)
    ladder.sort()

    points = []
    for num_pixels in ladder:
        points.append(bench_point(model, torch_module, cuda, num_pixels,
                                  num_slices, args, bodies_available,
                                  total_memory))
    narrow_cols = max(1, num_slices // 4)
    points.append(bench_point(model, torch_module, cuda,
                              min(8192, full_pixels), narrow_cols, args,
                              bodies_available, total_memory))
    return dict(points=points, ladder=ladder, full_pixels=full_pixels,
                num_slices=num_slices, narrow_cols=narrow_cols,
                warmup=BENCH_WARMUP, repeats=BENCH_REPEATS,
                mode="eager (the compiled reading comes from the composed run's "
                     "per-launch column)",
                bodies=sorted(bodies_available))


# ── one arm ───────────────────────────────────────────────────────────────────
def torch_worker(cfg):
    """One arm: install the candidate if this is the candidate's arm, prove what
    ran, one discarded cold pass, then WARM_REPEATS timed reconstructions with
    the instruments live.

    FOUR ORDERINGS, all load-bearing.
      THE PATCH GOES IN FIRST, before the model exists, so the compiled body
    traces the reduction this arm is about rather than tracing the shipped one
    and recompiling when a global-identity guard fires mid-pass.
      THE TREE IS PROVED NEXT, before anything is built, so an arm that imported
    the wrong source spends no GPU time at all.
      THE BENCH RUNS BEFORE THE COMPOSED PASSES, on a model whose projectors are
    built but whose reconstruction has not started, so its allocations are not
    competing with a reconstruction's.
      THE PROBES GO IN AFTER THE COLD PASS, because the body wrappers live inside
    the projector object and anything that rebuilt it would throw them away.
    """
    import numpy as np
    import torch

    cell = tuple(cfg["cell"])
    body_kind = cfg["body"]
    n_dev = cfg.get("n_dev")
    cuda = DEVICE == "cuda" and torch.cuda.is_available()
    smoke_cpu_devices = cfg.get("cpu_devices")

    result = dict(cfg, framework="torch", version=f"torch {torch.__version__}",
                  device=DEVICE, cuda=cuda,
                  device_name=(torch.cuda.get_device_name(0) if cuda else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  vcd_iterations=VCD_ITERATIONS, warm_repeats=WARM_REPEATS,
                  env_pythonpath=os.environ.get("PYTHONPATH"),
                  env_kill_switch=os.environ.get(KILL_SWITCH),
                  env_patch_sorted=os.environ.get(PATCH_ENV_VAR),
                  env_num_devices=os.environ.get("MBIRTORCH_NUM_DEVICES"))
    # Recorded on EVERY arm, not only the candidate's: the bench runs the sorted
    # reduction in every one-device arm, so which torch op it reduces with is
    # part of every one-device row.
    SORTED_STATE["reduction_api"] = _sorted_reduction_api()

    # ── THE CANDIDATE, installed before anything else exists ────────────────
    want_patch = BODY_ENV[body_kind]["patch_sorted"] == "1"
    result["patch_requested"] = want_patch
    result["patch_env_agrees"] = (
        (os.environ.get(PATCH_ENV_VAR, "0") == "1") == want_patch)
    if not result["patch_env_agrees"]:
        raise RuntimeError(
            f"this arm is the {body_kind!r} body, which "
            f"{'needs' if want_patch else 'must not have'} the sorted fan, and "
            f"{PATCH_ENV_VAR} is {os.environ.get(PATCH_ENV_VAR)!r}.  The runner "
            f"sets that variable from the arm, so a disagreement means the "
            f"child environment is not this arm's.")
    result["patch_record"] = install_sorted_fan() if want_patch else None

    # ── THE TREE, proved before any model is built ───────────────────────────
    result["fingerprint"] = tree_fingerprint()
    if not result["fingerprint"]["inside_expected_root"]:
        raise RuntimeError(
            f"this arm imported mbirtorch from "
            f"{result['fingerprint']['mbirtorch_file']}, which is not inside "
            f"{result['fingerprint']['expected_root']}.  Either PYTHONPATH did "
            f"not take or an editable install of another tree is winning "
            f"through a meta-path finder.")

    # ── WHICH FAN THE BODY WOULD CALL, before the model, and again after ─────
    result["fan_at_install"] = fan_identity()
    if want_patch and not result["fan_at_install"]["is_this_files_sorted"]:
        raise RuntimeError(
            f"the sorted fan was installed and the forward body would still "
            f"call {result['fan_at_install']}.  The patch did not take, and an "
            f"unpatched arm reporting 'no change' is the one failure this job "
            f"cannot survive.")
    if not want_patch and not result["fan_at_install"]["is_library_original"]:
        raise RuntimeError(
            f"this arm must run the shipped reduction and the forward body "
            f"would call {result['fan_at_install']}.")

    # ── THE KILL SWITCH, as the arm needs it ─────────────────────────────────
    want_switch = BODY_ENV[body_kind]["disable_triton"]
    result["kill_switch_ok"] = (os.environ.get(KILL_SWITCH) == want_switch)
    if not result["kill_switch_ok"]:
        raise RuntimeError(
            f"this arm is the {body_kind!r} body and needs "
            f"{KILL_SWITCH}={want_switch}; the environment carries "
            f"{os.environ.get(KILL_SWITCH)!r}.  The two torch arms must run the "
            f"SAME substrate or the sorted-against-scatter ratio is not about "
            f"the reduction.")
    result["kernel_state"] = kernel_state()

    from mbirtorch import triton_parallel
    result["shipped_fwd_chunk"] = int(triton_parallel.PARALLEL_FWD_VIEW_CHUNK)
    result["shipped_back_chunk"] = int(triton_parallel.PARALLEL_BACK_VIEW_CHUNK)

    # ── the model, with the device count pinned ──────────────────────────────
    model = _build_torch_model(cell, n_dev, cpu_devices=smoke_cpu_devices)
    recon_shape = tuple(model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    num_slices = int(recon_shape[2])
    result["num_slices"] = num_slices
    if num_slices != num_slices_for(cell):
        raise RuntimeError(
            f"the plan assumed {num_slices_for(cell)} slices at this cell and "
            f"the model built {num_slices}.")
    result["psf_radius"] = int(model.get_psf_radius())
    result["psf_width"] = 2 * result["psf_radius"] + 1
    result["num_channels"] = int(cell[2])
    result["full_pixel_count"] = int(model.full_index_count())

    realized_devices = [str(d) for d in model.sino_placement.devices]
    expected_count = (len(smoke_cpu_devices) if smoke_cpu_devices
                      else int(n_dev))
    if len(realized_devices) != expected_count:
        raise RuntimeError(
            f"this arm asked for {expected_count} device(s) and settled on "
            f"{len(realized_devices)} ({realized_devices}).")
    n_owners = len(realized_devices)
    result["realized_devices"] = realized_devices
    result["realized_n_devices"] = n_owners
    result["switch_at_install"] = read_switch(model)
    result["dev2dev_safe"] = bool(getattr(model, "dev2dev_safe", True))

    # WHICH BODY THE LIBRARY ACTUALLY BOUND.  This is the check that the kill
    # switch did what the arm needs: a triton arm whose bodies are torch bodies
    # is not the production reference, and a torch arm whose bodies are kernels
    # never touches the fan this job patches.
    fwd_per_dev, back_per_dev = _per_device_body_names(model)
    result["fwd_body_per_device"] = fwd_per_dev
    result["back_body_per_device"] = back_per_dev
    want_kernel = (body_kind == "triton") and cuda
    result["bodies_match_arm"] = (
        len(fwd_per_dev) == n_owners
        and all(("triton" in name) == want_kernel for name in fwd_per_dev)
        and all(("triton" in name) == want_kernel for name in back_per_dev))
    if not result["bodies_match_arm"]:
        raise RuntimeError(
            f"this arm is the {body_kind!r} body and the library bound "
            f"{fwd_per_dev} / {back_per_dev}.  Kernel bodies were "
            f"{'expected' if want_kernel else 'not expected'} here, and an arm "
            f"running the other substrate answers a different question.")

    # ── the shared sinogram artifact, md5-verified ───────────────────────────
    sino_path = _sino_path(cell)
    with open(_md5_path(cell)) as handle:
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

    # ── THE ISOLATED BENCH, one device only ──────────────────────────────────
    result["bench"] = None
    if cfg.get("bench") and os.environ.get("MG13_NO_BENCH", "0") != "1":
        available = {"torch"}
        if want_kernel and result["kernel_state"]["triton_usable"]:
            available.add("triton")
        sorted_calls_before = SORTED_STATE["calls"]
        result["bench"] = isolated_bench(model, torch, cuda, available)
        result["bench"]["sorted_calls_during_bench"] = (
            SORTED_STATE["calls"] - sorted_calls_before)

    # The region instrument goes in BEFORE the cold pass and is drained after
    # every timed pass.
    instrument, detach_regions = attach_region_instrument(model, torch, cuda)

    def peaks():
        if not cuda:
            return []
        return [int(torch.cuda.max_memory_allocated(d))
                for d in model.sino_placement.devices]

    def reserved():
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

    health = [sample_gpu_health()]

    # ── the discarded cold pass ──────────────────────────────────────────────
    if cuda:
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device("cuda", index))
    sorted_calls_before_cold = SORTED_STATE["calls"]
    start = time.perf_counter()
    out = vcd()
    result["vcd_cold"] = time.perf_counter() - start
    peaks_cold = peaks()
    reserved_cold = reserved()
    health.append(sample_gpu_health())
    result["sorted_calls_in_cold"] = SORTED_STATE["calls"] - sorted_calls_before_cold

    # THE CANDIDATE REALLY RAN.  On the sorted arm the fan counts its own calls,
    # so a positive count in the cold pass is the behavioural witness that the
    # patched reduction is what the reconstruction used -- independent of the
    # name and identity readings taken before the model existed.
    #
    # THE COUNT IS A DELTA ACROSS THE COLD PASS, not the running total, because
    # the isolated bench above deliberately runs BOTH reductions in every n=1
    # arm.  The reconstruction is what these checks are about.
    if want_patch and result["sorted_calls_in_cold"] <= 0:
        raise RuntimeError(
            "the sorted fan recorded no calls during the cold reconstruction, "
            "so the reconstruction did not use it.  Every timing in this arm "
            "would be the shipped reduction's under the candidate's name.")
    if not want_patch and result["sorted_calls_in_cold"] != 0:
        raise RuntimeError(
            f"the sorted fan recorded {result['sorted_calls_in_cold']} calls "
            f"inside the {body_kind!r} arm's reconstruction, which must not use "
            f"it at all.")

    result["fan_after_cold"] = fan_identity()
    result["switch_after_cold"] = read_switch(model)

    # ── the instruments, installed on the settled projector ──────────────────
    busy, gathers, verify, detach_probes, observed = \
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
        sorted_before = SORTED_STATE["calls"]
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
        record = dict(recon_index=repeat, wall_s=wall,
                      sorted_fan_calls=SORTED_STATE["calls"] - sorted_before)
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
        record["stall_ms_per_device"] = [
            b - u for b, u in zip(record["bracket_ms_per_device"],
                                  record["busy_ms_per_device"])]
        record["bracket_max_s"] = max(record["bracket_ms_per_device"]) / 1e3
        record["busy_max_s"] = max(record["busy_ms_per_device"]) / 1e3
        record["stall_max_s"] = record["bracket_max_s"] - record["busy_max_s"]
        record["probe_verify"] = verify()
        record["peak_bytes_per_device"] = peaks()
        record["reserved_bytes_per_device"] = reserved()
        per_recon.append(record)
        health.append(sample_gpu_health())

        checksums.append(float(np.sum(np.abs(out), dtype=np.float64)))
        if steps is None:
            steps = _sample_steps(out.shape)
        if repeat < 2:
            # Two samples: one for the cross-arm distances and a second so the
            # summary can state this arm's OWN pass-to-pass distance, which is
            # the floor every cross-arm number has to be read against (the
            # triton forward accumulates with float atomics and is not
            # bit-reproducible).
            path = _sample_path(cfg["arm_id"], repeat)
            np.save(path, np.ascontiguousarray(
                out[::steps[0], ::steps[1], ::steps[2]], dtype=np.float32))
            sample_paths.append(path)

        if repeat == 0:
            _check_witnesses(result, record, body_kind, n_owners, num_slices,
                             want_patch)

    result["vcd_warm_all"] = warm
    result["vcd_warm"] = statistics.median(warm)
    result["vcd_warm_spread_s"] = (max(warm) - min(warm)) if len(warm) > 1 else 0.0
    result["per_recon"] = per_recon
    result["device_names"] = device_names
    result["view_batch_observed_per_device"] = {
        k: sorted(v.items()) for k, v in observed.items()}
    result["busy_backend"] = busy.backend
    result["probe_verify_after"] = verify()
    result["fan_after_warm"] = fan_identity()
    result["sorted_state"] = dict(SORTED_STATE)
    # The reconstruction's own count, separated from the bench's: this is the
    # number that says which reduction the timed passes used.
    result["sorted_calls_in_recon"] = (
        result["sorted_calls_in_cold"]
        + sum(p["sorted_fan_calls"] for p in per_recon))
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
    result["recon_devices"] = [str(d) for d in model.recon_placement.devices]
    result["compile"] = compile_diagnostics()

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
    result["gpu_health"] = worst_health([h for h in health if h])
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def _check_witnesses(result, record, body_kind, n_owners, num_slices,
                     want_patch):
    """THE PROOF THAT THIS ARM MEASURED WHAT ITS NAME SAYS, run on the first
    timed reconstruction and FATAL on any disagreement."""
    checks = {}

    # 1. THE REDUCTION.  On the sorted arm the fan counts itself; on the other
    #    two it must never be entered.
    checks["sorted_fan_used"] = ((record["sorted_fan_calls"] > 0) == want_patch)

    # 2. THE FORWARD SHAPE.  One device never enters the sharded driver, so it
    #    has no gathers; more than one runs the column gather and no banded
    #    fan-out at all.
    if n_owners > 1:
        checks["gathers_positive"] = (record["gather_calls"] > 0)
        heights = {int(k) for k in (record["cyl_height_hist"] or {})}
        checks["cylinder_height_is_full_slice_axis"] = (heights == {num_slices})
    else:
        checks["no_gathers_at_one_device"] = (record["gather_calls"] == 0)
    checks["no_banded_fanout"] = (record["broadcast_calls"] == 0)

    # 3. THE LAUNCHES HAPPENED, and the busy probe saw them.
    checks["body_calls_positive"] = any(
        c > 0 for c in record["busy_calls_per_device"])

    # 4. THE PROBES ARE STILL THE ONES THE DRIVER CALLS.
    checks["probes_attached"] = all(record["probe_verify"].values())

    # The launch shape, recorded for the collision ratio and the per-slice
    # number.  Not a check -- a reading the summary needs from the run rather
    # than from the plan.
    cols, pixels = set(), set()
    for hist in record["busy_value_cols_per_device"]:
        cols |= {int(k) for k in (hist or {})}
    for hist in record["busy_value_pixels_per_device"]:
        pixels |= {int(k) for k in (hist or {})}
    result["launch_cols_seen"] = sorted(cols)
    result["launch_pixels_seen"] = sorted(pixels)

    result["witnesses"] = dict(
        checks=checks, body=body_kind,
        sorted_fan_calls=record["sorted_fan_calls"],
        gather_calls=record["gather_calls"],
        broadcast_calls=record["broadcast_calls"],
        cylinder_heights=record["cyl_height_hist"],
        cylinder_widths=record["cyl_width_hist"],
        launch_cols=result["launch_cols_seen"],
        launch_pixels=result["launch_pixels_seen"])
    failed = [name for name, ok in checks.items() if not ok]
    result["witnesses_ok"] = not failed
    if failed:
        raise RuntimeError(
            f"witness(es) {failed} failed on the {body_kind!r} arm.  The full "
            f"reading is {result['witnesses']}.  This arm is not measuring what "
            f"its name says, so it reports nothing.")


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
        return dict(per_device=[])

    def median(values):
        return float(statistics.median(values)) if values else None

    def spread(values):
        return float(max(values) - min(values)) if len(values) > 1 else 0.0

    psf_width = int(result.get("psf_width") or 3)
    num_channels = int(result.get("num_channels") or 0)
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
        mean_cols = _hist_mean(cols_hist)
        mean_pixels = _hist_mean(pix_hist)
        per_device.append(dict(
            device=name,
            bracket_span_s=bracket_med,
            busy_sum_s=busy_med,
            stall_s=((bracket_med - busy_med)
                     if bracket_med is not None and busy_med is not None
                     else None),
            busy_calls=calls_med,
            per_launch_ms=per_launch,
            mean_cols_per_launch=mean_cols,
            mean_pixels_per_launch=mean_pixels,
            # per-slice: the per-launch time divided by the columns that launch
            # carried, which is the form the 0.0411 ms precedent is stated in.
            ms_per_slice=((per_launch / mean_cols)
                          if per_launch and mean_cols else None),
            collision_ratio=collision_ratio(psf_width, mean_pixels or 0,
                                            num_channels),
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
    slice_all = [d["ms_per_slice"] for d in per_device if d["ms_per_slice"]]
    return dict(
        per_device=per_device,
        forward_bracket_max_s=median(bracket_max),
        forward_bracket_spread_s=spread(bracket_max),
        forward_busy_max_s=median(busy_max),
        forward_busy_spread_s=spread(busy_max),
        forward_stall_max_s=(median(bracket_max) - median(busy_max)),
        # The per-launch headline is the SLOWEST device's, matching every other
        # headline here: the reconstruction waits for that device.
        per_launch_ms=(max(per_launch_all) if per_launch_all else None),
        per_launch_spread_ms=_per_launch_spread(passes),
        ms_per_slice=(max(slice_all) if slice_all else None),
        gather_calls_per_recon=statistics.median(
            [p["gather_calls"] for p in passes]),
        broadcast_calls_per_recon=statistics.median(
            [p["broadcast_calls"] for p in passes]),
        composed_s=result.get("vcd_warm"),
        composed_spread_s=result.get("vcd_warm_spread_s"))


def _per_launch_spread(passes):
    """The pass-to-pass spread of the slowest device's per-launch time."""
    values = []
    for record in passes:
        best = None
        for busy_ms, calls in zip(record["busy_ms_per_device"],
                                  record["busy_calls_per_device"]):
            if calls:
                value = busy_ms / calls
                best = value if best is None else max(best, value)
        if best is not None:
            values.append(best)
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def generator_worker(cfg):
    """Build ONE shared sinogram, on the reference body, pinned to one device.

    Every arm reconstructs THAT array, so no arm's timing or value carries an
    input difference.  One device is enough and is the point: a single-device
    forward never enters the sharded driver at all."""
    import numpy as np

    import mbirtorch

    cell = tuple(cfg["cell"])
    devices = cfg.get("cpu_devices") or [DEVICE]
    model = _build_torch_model(cell, 1, cpu_devices=devices)
    recon_shape = tuple(model.get_params("recon_shape"))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = np.ascontiguousarray(
        np.asarray(_to_numpy(model.forward_project(phantom)), dtype=np.float32))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _sino_path(cell)
    np.save(path, sinogram)
    digest = _md5(path)
    with open(_md5_path(cell), "w") as handle:
        handle.write(digest + "\n")
    return dict(cfg, framework="torch", role="generator", path=path,
                sino_md5=digest, sinogram_shape=list(sinogram.shape),
                recon_shape=list(recon_shape), fingerprint=tree_fingerprint(),
                kernel_state=kernel_state(), fan=fan_identity(),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


# ── the value read-out ────────────────────────────────────────────────────────
def _rel_distance(path_a, path_b):
    """Two distances between two strided reconstruction samples, and why there
    are two.  ``rel_l2`` is a relative L2 over the sample, which is the metric
    the design notes state value expectations in.  ``max_rel_of_peak`` is
    max|a-b| / max|b|, the functional form the standing parity suites take."""
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
    """One arm's per-reconstruction checksums, reduced: the median and the
    repeat-to-repeat spread as a fraction of it.  That spread is the run-to-run
    noise floor every cross-arm checksum distance is read against."""
    values = row.get("recon_checksums") or []
    if not values:
        return None, None
    mid = statistics.median(values)
    spread = (max(values) - min(values)) / mid if mid else None
    return mid, spread


def _first_sample(row):
    return ((row or {}).get("value_sample_paths") or [None])[0]


def value_table(rows):
    """Every composed value distance in this job, computed two ways.

    For each arm: its OWN repeat-to-repeat distance -- the floor -- and its
    distance to the TORCH SCATTER arm at the same device count.  The torch
    scatter arm is the reference because it is the candidate's baseline: the
    sorted form's whole claim is that it computes the same thing that arm does."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    by_token = {r.get("token"): r for r in live}
    out = []
    for row in live:
        count = row.get("count_token")
        reference = by_token.get(f"{count}-scatter")
        samples = row.get("value_sample_paths") or []
        own = _rel_distance(samples[1], samples[0]) if len(samples) > 1 else None
        vs_ref = (_rel_distance(_first_sample(row), _first_sample(reference))
                  if reference is not None and reference is not row else None)
        median_checksum, repeat_spread = _checksum_stats(row)
        out.append(dict(
            token=row.get("token"), count=count, body=row.get("body"),
            n=row.get("n_dev"),
            checksums=row.get("recon_checksums"),
            checksum_median=median_checksum,
            checksum_repeat_spread=repeat_spread,
            own_pass_to_pass=own,
            vs_scatter=vs_ref,
            vs_scatter_token=(reference or {}).get("token"),
            within_tolerance=(None if vs_ref is None
                              else bool(vs_ref["rel_l2"] <= VALUE_REL_TOL))))
    return out


# ── the runner (mg5's / mg9's / mg11's / mg12's subprocess pattern) ───────────
def arm_env(cfg):
    """The env that DEFINES an arm, set EXPLICITLY so nothing inherits.

    THREE variables carry the experiment.  PYTHONPATH names the tree and is SET
    rather than prepended, so an inherited path cannot put another mbirtorch
    ahead of it; the child proves it imported from that root anyway, because an
    editable install can still win through a meta-path finder.
    MBIRTORCH_DISABLE_TRITON selects the substrate: 0 for the production
    reference, 1 for both torch arms, and the child asserts the value its body
    needs.  MG13_PATCH_SORTED selects the reduction, and the child asserts that
    too, so an arm can never disagree with its own name.

    MBIRTORCH_FORWARD_COLUMN_GATHER is REMOVED.  The column gather is the shipped
    default on this tree, so its absence IS the shipped configuration, and the
    four-device arms are supposed to measure the shipped forward.
    MBIRTORCH_NUM_DEVICES is removed too: the device count is pinned on the
    model, and a variable saying something else would be a second, silent
    opinion.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_FORWARD_COLUMN_GATHER", None)
    env.pop("MBIRTORCH_WIDENING_GUARD", None)
    body = cfg.get("body", "triton")
    env[KILL_SWITCH] = BODY_ENV[body]["disable_triton"]
    env[PATCH_ENV_VAR] = BODY_ENV[body]["patch_sorted"]
    env["PYTHONPATH"] = tree_root()
    return env


def run_one(cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, f"_cfg_mg13_{cfg['arm_id']}.json")
    out_path = os.path.join(RESULTS_DIR, f"_out_mg13_{cfg['arm_id']}.json")
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
    """The generator arm, then the measured arms in declared order."""
    cell = cell_for()
    gen = dict(framework="torch", arm_class="generator", geometry=GEOMETRY,
               cell=list(cell), n_dev=1, token="gen_parallel",
               count_token=None, body="triton",
               arm_id=f"{GEOMETRY}_{cell[0]}_generator")
    if DEVICE != "cuda":
        gen["cpu_devices"] = [DEVICE]
    plan = [gen]
    measured = []
    for arm in arms:
        cfg = dict(framework="torch", arm_class="instrument", geometry=GEOMETRY,
                   cell=list(cell), n_dev=arm["n_dev"], token=arm["token"],
                   count_token=arm["count_token"], body=arm["body"],
                   # The isolated bench runs at ONE device only: a bench of one
                   # call on one device answers the same question the four-device
                   # arms would answer more expensively, and the four-device arms
                   # are there for the composed check.
                   bench=(arm["n_dev"] == 1),
                   arm_id=f"{GEOMETRY}_{cell[0]}_n{arm['n_dev']}_{arm['body']}")
        if DEVICE != "cuda":
            cfg["cpu_devices"] = [DEVICE] * arm["n_dev"]
        measured.append(cfg)
    return plan, measured


# ── the summary ───────────────────────────────────────────────────────────────
def _fmt(value, spec, dash="-"):
    """``format(value, spec)``, with a missing value rendered as a dash padded to
    the SAME width -- an unpadded dash shifts every column to its right."""
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
    them, the launch count, the per-launch and per-slice times, and the launch
    shape the collision ratio is computed from."""
    print(f"\n  [{row.get('token')}] {row.get('geometry')} "
          f"n={row.get('n_dev')} body={row.get('body')}"
          f"  composed {_fmt(row.get('vcd_warm'), '.2f')} s "
          f"(cold {_fmt(row.get('vcd_cold'), '.2f')} s)")
    print(f"      {BODY_NOTE.get(row.get('body'), '')}")
    fingerprint = row.get("fingerprint") or {}
    digests = fingerprint.get("sha256_12") or {}
    print(f"      tree: {fingerprint.get('package_dir')}")
    print("      bytes: " + "  ".join(f"{k}={v}" for k, v in digests.items()))
    fan = row.get("fan_after_warm") or row.get("fan_at_install") or {}
    kernels = row.get("kernel_state") or {}
    print(f"      reduction: the body calls {fan.get('body_calls')} "
          f"(library original={fan.get('is_library_original')}, "
          f"sorted={fan.get('is_this_files_sorted')}); sorted fan calls inside "
          f"the reconstructions={row.get('sorted_calls_in_recon')} "
          f"(bench calls are separate)")
    print(f"      substrate: {KILL_SWITCH}={kernels.get('kill_switch')} "
          f"triton_usable={kernels.get('triton_usable')} "
          f"bodies={row.get('fwd_body_per_device')}")
    sorted_state = row.get("sorted_state") or {}
    if sorted_state.get("reduction_api"):
        print(f"      sorted form: api={sorted_state.get('reduction_api')} "
              f"view chunk={sorted_state.get('view_chunk_used')} "
              f"stable sort={sorted_state.get('stable_sort')} "
              f"largest transient "
              f"{_fmt(_gb(sorted_state.get('transient_bytes')), '.2f')} GB")
    compile_info = row.get("compile") or {}
    print(f"      compiler: graph breaks={compile_info.get('graph_breaks')} "
          f"compile errors={compile_info.get('compile_errors')}")
    wit = row.get("witnesses") or {}
    print(f"      run witness: gathers={wit.get('gather_calls')} "
          f"fan-outs={wit.get('broadcast_calls')} "
          f"launch pixels={wit.get('launch_pixels')} "
          f"launch cols={wit.get('launch_cols')}")
    header = (f"      {'device':<10}{'bracket_s':>11}{'busy_s':>10}"
              f"{'stall_s':>10}{'calls':>8}{'per_launch_ms':>15}"
              f"{'ms_per_slice':>14}{'collisions':>12}{'peak_GB':>10}")
    print(header)
    for dev in row.get("per_device") or []:
        print(f"      {dev['device']:<10}"
              f"{_fmt(dev['bracket_span_s'], '11.3f')}"
              f"{_fmt(dev['busy_sum_s'], '10.3f')}"
              f"{_fmt(dev['stall_s'], '10.3f')}"
              f"{_fmt(dev['busy_calls'], '8.0f')}"
              f"{_fmt(dev['per_launch_ms'], '15.3f')}"
              f"{_fmt(dev['ms_per_slice'], '14.4f')}"
              f"{_fmt(dev['collision_ratio'], '12.1f')}"
              f"{_fmt(_gb(dev['peak_bytes']), '10.2f')}")
    print(f"      per-slice precedent for comparison: "
          f"{PRECEDENT_MS_PER_SLICE:.4f} ms per slice ({PRECEDENT_NOTE})")


def print_bench(row):
    """One arm's isolated bench: the ladder, and what each body cost on it."""
    bench = row.get("bench")
    if not bench:
        return
    print(f"\n  -- the isolated bench, {row.get('token')} "
          f"({bench.get('mode')}) --")
    print(f"     bodies present in this process: {bench.get('bodies')}; "
          f"warmup {bench.get('warmup')}, timed launches {bench.get('repeats')}")
    names = []
    for point in bench["points"]:
        for name in point.get("results", {}):
            if name not in names:
                names.append(name)
    header = (f"     {'pixels':>9}{'cols':>7}{'Vb':>5}{'collisions':>12}"
              + "".join(f"{(n + '_ms')[-23:]:>24}" for n in names))
    print(header)
    for point in bench["points"]:
        if point.get("skipped"):
            print(f"     {point['num_pixels']:>9}{point['num_cols']:>7}"
                  f"{point['view_batch']:>5}"
                  f"{_fmt(point['collision_ratio'], '12.1f')}"
                  f"   SKIPPED: {point['skipped']}")
            continue
        line = (f"     {point['num_pixels']:>9}{point['num_cols']:>7}"
                f"{point['view_batch']:>5}"
                f"{_fmt(point['collision_ratio'], '12.1f')}")
        for name in names:
            timing = point["results"].get(name)
            line += _fmt(timing["median_ms"] if timing else None, "24.3f")
        print(line)
    print("     transient per point (the sorted form holds every psf tap at "
          "once, the scatter one at a time):")
    for point in bench["points"]:
        print(f"       pixels {point['num_pixels']:>9} cols "
              f"{point['num_cols']:>5}: scatter "
              f"{_gb(point['scatter_transient_bytes']):.2f} GB, sorted "
              f"{_gb(point['sorted_transient_bytes']):.2f} GB "
              f"({point['sorted_over_scatter_transient']}x)")
    print("     value against the scatter, on identical inputs in this process "
          f"(tolerance {VALUE_REL_TOL:g} relative):")
    for point in bench["points"]:
        for name, dist in (point.get("value") or {}).items():
            flag = "" if dist["rel_l2"] is None or dist["rel_l2"] <= VALUE_REL_TOL \
                else "   <-OUTSIDE THE STATED TOLERANCE"
            print(f"       pixels {point['num_pixels']:>9} {name:<20} "
                  f"rel_l2 {_fmt(dist['rel_l2'], '.3e')} "
                  f"max_rel_of_peak {_fmt(dist['max_rel_of_peak'], '.3e')} "
                  f"bit-identical={dist['bit_identical']}{flag}")


def _floored(spread, reference):
    """A spread, floored at SPREAD_FLOOR_FRAC of the torch scatter arm's reading.
    Two warm passes is a weak estimator of run-to-run noise, and without a floor
    a freakishly tight pair would let the comparison resolve a difference it has
    no business resolving."""
    if spread is None:
        spread = 0.0
    if reference:
        return max(float(spread), SPREAD_FLOOR_FRAC * abs(float(reference)))
    return float(spread)


def _ratio_line(label, reference_value, candidate_value, reference_spread,
                candidate_spread, unit):
    """One 'A against B' reading: the ratio, both raw numbers, and whether the
    difference is larger than the two spreads together.

    The ratio is reference / candidate, so above 1.00 means the CANDIDATE is
    faster.  Nothing here is a gate: the WITHIN THE SPREAD note says the two
    passes cannot resolve the difference, not that the difference is acceptable
    or unacceptable."""
    if reference_value is None or candidate_value is None or not candidate_value:
        return dict(ratio=None, resolved=None,
                    text=f"{label}: NOT RESOLVABLE (a reading is missing)")
    ratio = reference_value / candidate_value
    margin = (_floored(reference_spread, reference_value)
              + _floored(candidate_spread, reference_value))
    resolved = abs(reference_value - candidate_value) > margin
    direction = "faster" if ratio > 1.0 else "slower"
    shown = ratio if ratio >= 1.0 else (1.0 / ratio if ratio else None)
    note = "" if resolved else "  [WITHIN THE SPREAD: the two warm passes " \
                               "cannot resolve this difference]"
    return dict(ratio=ratio, resolved=resolved, margin=margin,
                reference=reference_value, candidate=candidate_value,
                text=(f"{label}: {direction} by {shown:.2f}x "
                      f"({candidate_value:.3f} against {reference_value:.3f} "
                      f"{unit}){note}"))


def comparison(rows, count):
    """The three bodies at ONE device count, reduced to the numbers the two
    verdict lines are drawn from.  Every reading here was measured in this job."""
    by_body = {r.get("body"): r for r in rows
               if not r.get("error") and r.get("arm_class") != "generator"
               and r.get("n_dev") == count}
    entry = dict(count=count, present=sorted(by_body), resolvable=False)
    if not all(b in by_body for b in BODIES):
        entry["why"] = (f"this device count needs all three bodies and has "
                        f"{sorted(by_body) or 'none'}")
        return entry
    entry["resolvable"] = True
    for body in BODIES:
        row = by_body[body]
        entry[body] = dict(
            token=row.get("token"),
            bracket_s=row.get("forward_bracket_max_s"),
            bracket_spread_s=row.get("forward_bracket_spread_s"),
            busy_s=row.get("forward_busy_max_s"),
            busy_spread_s=row.get("forward_busy_spread_s"),
            stall_s=row.get("forward_stall_max_s"),
            per_launch_ms=row.get("per_launch_ms"),
            per_launch_spread_ms=row.get("per_launch_spread_ms"),
            ms_per_slice=row.get("ms_per_slice"),
            composed_s=row.get("composed_s"),
            composed_spread_s=row.get("composed_spread_s"),
            forward_wall_s=row.get("forward_funnel_wall_per_pass_s"),
            peak_bytes=row.get("gpu_peak_per_device"),
            graph_breaks=(row.get("compile") or {}).get("graph_breaks"),
            gpu_hot=row.get("gpu_hot"))

    scatter, sorted_arm, triton = (entry["scatter"], entry["sorted"],
                                   entry["triton"])
    entry["sorted_vs_scatter"] = dict(
        per_launch=_ratio_line(
            "per-launch", scatter["per_launch_ms"], sorted_arm["per_launch_ms"],
            scatter["per_launch_spread_ms"], sorted_arm["per_launch_spread_ms"],
            "ms"),
        forward_bracket=_ratio_line(
            "forward bracket", scatter["bracket_s"], sorted_arm["bracket_s"],
            scatter["bracket_spread_s"], sorted_arm["bracket_spread_s"], "s"),
        composed=_ratio_line(
            "composed", scatter["composed_s"], sorted_arm["composed_s"],
            scatter["composed_spread_s"], sorted_arm["composed_spread_s"], "s"))
    entry["triton_vs_scatter"] = dict(
        per_launch=_ratio_line(
            "per-launch", scatter["per_launch_ms"], triton["per_launch_ms"],
            scatter["per_launch_spread_ms"], triton["per_launch_spread_ms"],
            "ms"),
        forward_bracket=_ratio_line(
            "forward bracket", scatter["bracket_s"], triton["bracket_s"],
            scatter["bracket_spread_s"], triton["bracket_spread_s"], "s"),
        composed=_ratio_line(
            "composed", scatter["composed_s"], triton["composed_s"],
            scatter["composed_spread_s"], triton["composed_spread_s"], "s"))

    unstable = []
    for body in BODIES:
        block = entry[body]
        for name, value_key, spread_key in (
                ("bracket", "bracket_s", "bracket_spread_s"),
                ("busy", "busy_s", "busy_spread_s"),
                ("per-launch", "per_launch_ms", "per_launch_spread_ms"),
                ("composed", "composed_s", "composed_spread_s")):
            value, spread_value = block.get(value_key), block.get(spread_key)
            if not value or spread_value is None:
                continue
            frac = abs(spread_value) / abs(value)
            if frac > WARM_INSTABILITY_FRAC:
                unstable.append(dict(body=body, reading=name, frac=frac,
                                     value=value, spread=spread_value))
    # WHERE A SLOW SORTED ARM SHOULD BE ATTRIBUTED FIRST.  The two torch arms run
    # the same compiled body with one function swapped, so they should compile
    # the same way.  If the candidate's arm broke its graph more times than the
    # baseline's did, part of the composed and per-launch difference is the break
    # and not the reduction -- and the eager bench is the reading that separates
    # them.  This is a note, not a correction: no number below is adjusted.
    breaks_scatter = scatter.get("graph_breaks")
    breaks_sorted = sorted_arm.get("graph_breaks")
    entry["graph_break_note"] = ""
    if (isinstance(breaks_scatter, int) and isinstance(breaks_sorted, int)
            and breaks_sorted > breaks_scatter):
        entry["graph_break_note"] = (
            f"GRAPH BREAKS DIFFER: the sorted arm broke its compiled graph "
            f"{breaks_sorted} times against the scatter arm's "
            f"{breaks_scatter}.  Part of the compiled difference is that break "
            f"rather than the reduction; the eager bench above is the reading "
            f"that separates the two.")

    entry["unstable"] = unstable
    entry["unstable_line"] = (
        "" if not unstable else
        "UNSTABLE ARM(S) AT THIS DEVICE COUNT: "
        + "; ".join(f"{u['body']} {u['reading']} {u['value']:.3f} with a warm "
                    f"spread of {u['spread']:.3f} ({u['frac']:.0%} of its own "
                    f"reading)" for u in unstable)
        + " -- the verdict lines are drawn exactly as stated; this says the "
          "reading should not be leaned on")
    return entry


def print_comparison(entry, values):
    print("\n" + "-" * 78)
    if not entry.get("resolvable"):
        print(f"  n={entry['count']}: NOT RESOLVABLE -- {entry.get('why')}")
        return
    print(f"  n={entry['count']}: the three bodies")
    print("-" * 78)
    header = (f"      {'body':<10}{'per_launch_ms':>15}{'ms_per_slice':>14}"
              f"{'fwd_bracket_s':>15}{'fwd_wall_s':>12}{'composed_s':>12}")
    print(header)
    for body in BODIES:
        block = entry[body]
        print(f"      {body:<10}"
              f"{_fmt(block['per_launch_ms'], '15.3f')}"
              f"{_fmt(block['ms_per_slice'], '14.4f')}"
              f"{_fmt(block['bracket_s'], '15.3f')}"
              f"{_fmt(block['forward_wall_s'], '12.3f')}"
              f"{_fmt(block['composed_s'], '12.2f')}")
    if entry.get("unstable_line"):
        print(f"\n  {entry['unstable_line']}")
    if entry.get("graph_break_note"):
        print(f"\n  {entry['graph_break_note']}")
    print("\n      sorted against scatter (the candidate against its baseline, "
          "same substrate):")
    for key in ("per_launch", "forward_bracket", "composed"):
        print(f"        {entry['sorted_vs_scatter'][key]['text']}")
    print("      triton against scatter (what the production kernel is worth "
          "on this cell):")
    for key in ("per_launch", "forward_bracket", "composed"):
        print(f"        {entry['triton_vs_scatter'][key]['text']}")
    print("\n      composed value distances at this device count:")
    for value in values:
        if value.get("n") != entry["count"]:
            continue
        own = value.get("own_pass_to_pass") or {}
        vs = value.get("vs_scatter") or {}
        flag = ""
        if value.get("within_tolerance") is False:
            flag = f"   <-OUTSIDE {VALUE_REL_TOL:g} RELATIVE"
        print(f"        {value['token']:<14} own pass-to-pass "
              f"{_fmt(own.get('rel_l2'), '.3e')}   against the scatter arm "
              f"{_fmt(vs.get('rel_l2'), '.3e')}{flag}")


def fingerprint_check(rows):
    """THE CROSS-ARM PROVENANCE CHECK, which no single arm can make.  This job
    runs ONE tree, so every arm's bytes must be identical; a difference means an
    arm imported a different checkout and its numbers are not comparable."""
    live = [r for r in rows if not r.get("error")]
    seen = {}
    for row in live:
        digests = ((row.get("fingerprint") or {}).get("sha256_12") or {})
        seen.setdefault(tuple(sorted(digests.items())), []).append(
            row.get("token"))
    ok = len(seen) <= 1
    return dict(ok=ok, groups=[dict(bytes=dict(k), arms=v)
                               for k, v in seen.items()])


def verdict_lines(entries):
    """THE TWO LINES THIS JOB EXISTS TO PRINT.

    Both are ratios of readings measured in this job, and NEITHER IS A GATE.
    There is no stop threshold anywhere in this file, deliberately: the recorded
    threshold for the sorted form lives in the forward remedy memo (section 8.5)
    and in mbirjax's own constants, and deciding whether these numbers clear it
    is a human re-read of that record against this reading.  This function turns
    the measurements into the two sentences that re-read needs, and stops."""
    by_count = {e["count"]: e for e in entries if e.get("resolvable")}

    def ratio(entry, family, key):
        if entry is None:
            return None
        return (entry.get(family) or {}).get(key, {}).get("ratio")

    def describe(value):
        """A ratio as a magnitude with its direction attached.  The ratio is
        always reference / candidate, so a value above 1.00 means the candidate
        is the faster one; printing '0.41x' would invite the reader to
        misattribute the direction, so the word is carried with the number."""
        if value is None:
            return "not resolvable"
        if value >= 1.0:
            return f"{value:.2f}x faster"
        return f"{1.0 / value:.2f}x slower"

    # The single-device count carries the per-launch clause (the question is
    # about one call) and the largest configured count carries the composed
    # clause (the composed check).  Both are read off the plan rather than
    # spelled, so a trimmed run's lines say which counts they are about.
    one_count = device_counts()[0]
    many_count = device_counts()[-1]
    one, many = by_count.get(one_count), by_count.get(many_count)
    launch_ratio = ratio(one, "sorted_vs_scatter", "per_launch")
    composed_ratio = ratio(many, "sorted_vs_scatter", "composed")
    direction, magnitude = "not resolvable", "-"
    if launch_ratio is not None:
        direction = "faster" if launch_ratio > 1.0 else "slower"
        magnitude = (f"{launch_ratio:.2f}x" if launch_ratio >= 1.0
                     else f"{1.0 / launch_ratio:.2f}x")
    sorted_line = (
        f"SORTED vs SCATTER (torch substrate): {direction} by "
        f"{magnitude} at n={one_count} per-launch, "
        f"{describe(composed_ratio)} composed n={many_count}")

    triton_launch = ratio(one, "triton_vs_scatter", "per_launch")
    triton_composed = ratio(many, "triton_vs_scatter", "composed")
    triton_line = (
        f"TRITON REFERENCE: the production kernel is "
        f"{describe(triton_launch)} vs the torch scatter at n={one_count} "
        f"per-launch, {describe(triton_composed)} composed n={many_count}")
    return [sorted_line, triton_line], dict(
        sorted_per_launch_n1=launch_ratio, sorted_composed_n4=composed_ratio,
        triton_per_launch_n1=triton_launch, triton_composed_n4=triton_composed)


def summarize(rows, out_path):
    """The whole read-out: the per-arm tables and benches, then one comparison
    block per device count, then the two lines."""
    live = [r for r in rows if not r.get("error")
            and r.get("arm_class") != "generator"]
    failed = [r for r in rows if r.get("error")]
    print("\n" + "=" * 78)
    print(f"mg13 -- the sorted channel reduction, {len(live)} arms on "
          f"{RUN_LABEL} ({DEVICE})")
    print("=" * 78)
    print("  Three bodies, measured in this job, against one another at the "
          "same cell: the")
    print("  production Triton kernel, the compiled torch fan with its shipped "
          "atomic scatter,")
    print("  and the same torch fan with the channel reduction replaced by "
          "sort + segment-sum")
    print("  + one write per run.  The candidate's baseline is the torch "
          "scatter, which shares")
    print("  its substrate; the kernel is measured beside them as what "
          "production runs.")
    for body in BODIES:
        print(f"    {body:<9} {BODY_NOTE[body]}")
    if DEVICE != "cuda":
        print()
        print("  SMOKE RUN -- READ THE LINES AS EXERCISE, NOT AS RESULT.  Off "
              "CUDA there is no")
        print("  Triton kernel at all, so the reference arm runs the torch fan "
              "and the three-body")
        print("  comparison collapses to two.  What the smoke establishes is "
              "the plumbing: the")
        print("  patch takes, the witnesses fire, and the value distances are "
              "computed.")

    for row in live:
        print_arm_table(row)
        print_bench(row)

    print("\n  -- the tree fingerprints, compared ACROSS arms --")
    check = fingerprint_check(rows)
    for group in check["groups"]:
        print("      " + "  ".join(f"{k}={v}" for k, v
                                   in sorted(group["bytes"].items())))
        print(f"        arms: {group['arms']}")
    if check["ok"]:
        print("      every arm imported the same bytes, which is what one tree "
              "means")
    else:
        print("      PROVENANCE PROBLEM: the arms did not all import the same "
              "bytes.  Do not read")
        print("      the blocks below -- a comparison across two checkouts is "
              "not the comparison")
        print("      this job is about.")

    values = value_table(rows)
    entries = []
    print("\n" + "=" * 78)
    print("  THE COMPARISON BLOCKS, one per device count")
    print("=" * 78)
    for count in device_counts():
        entry = comparison(rows, count)
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
        if not entry.get("resolvable"):
            continue
        if entry.get("unstable_line"):
            print(f"  {entry['unstable_line']}")
        if entry.get("graph_break_note"):
            print(f"  n={entry['count']}: {entry['graph_break_note']}")
    lines, ratios = verdict_lines(entries)
    for line in lines:
        print(f"  {line}")
    print()
    print("  These two lines are readings, not rulings.  This file carries no "
          "stop threshold")
    print("  and no gate: the recorded threshold for the sorted form is in the "
          "forward remedy")
    print("  memo (section 8.5) and in mbirjax's own selection constants, and "
          "whether these")
    print("  numbers clear it is a human re-read of that record against this "
          "reading.")

    print("\n  -- the value distances, every arm --")
    for value in values:
        own = value.get("own_pass_to_pass") or {}
        vs = value.get("vs_scatter") or {}
        print(f"      {value['token']:<14} own pass-to-pass "
              f"{_fmt(own.get('rel_l2'), '.3e')}   against "
              f"{value.get('vs_scatter_token')} {_fmt(vs.get('rel_l2'), '.3e')}"
              f"   within {VALUE_REL_TOL:g}: {value.get('within_tolerance')}")
    print(f"      the tolerance is {VALUE_REL_TOL:g} relative, and it is stated "
          f"here because sorting")
    print("      changes the summation order: the sorted form cannot be "
          "bit-identical to the")
    print("      scatter, and each arm's own pass-to-pass distance is the floor "
          "to read the")
    print("      cross-arm numbers against.")
    deterministic = [v["token"] for v in values
                     if (v.get("own_pass_to_pass") or {}).get("rel_l2") == 0.0]
    if deterministic:
        print(f"      {deterministic} repeated their own value EXACTLY across "
              f"the warm passes.  That is")
        print("      a property of the reduction rather than an artifact: a "
              "sort followed by a")
        print("      segment-sum adds a channel's contributions in one fixed "
              "order, where an atomic")
        print("      scatter adds them in whatever order the adds arrive.  It "
              "is a reading about")
        print("      the candidate, and it is not part of either verdict line.")

    hot = [r.get("token") for r in live if r.get("gpu_hot")]
    print(f"\n  throttle rule: sw_power_cap at normal temperature is recorded "
          f"and KEPT; {len(hot)} arm(s) ran hot or clock-depressed: {hot}")

    print(f"\nrows: {out_path}")
    return dict(comparisons=entries, values=values, fingerprints=check,
                verdict_lines=lines, verdict_ratios=ratios, hot_arms=hot)


# ── the wall arithmetic ───────────────────────────────────────────────────────
def arm_base_seconds(cfg):
    """The estimated subprocess wall for one arm, in seconds.

    The measured anchors are mg11's four-device parallel arm at this cell
    (21.8 s composed) and this campaign's one-device parallel reading (40.0 s
    composed), both with the kernels on.  A torch arm is scaled by the composed
    five-arm gate's ratio between the compiled torch bodies and the kernel path
    at this cell.  A sorted arm has never been run at all, so it is the scatter
    arm's estimate -- and the high end takes mg10's and mg11's rule for an
    unmeasured arm and adds half again."""
    composed = (ONE_DEVICE_COMPOSED_S if cfg["n_dev"] == 1
                else MG11_PARALLEL_N4["composed_s"])
    if cfg["body"] != "triton":
        composed *= TORCH_BODY_SLOWDOWN
    passes = 1 + WARM_REPEATS                    # one cold, discarded, plus warm
    overhead = 60                                # import, model build, sinogram
    base = composed * passes + overhead
    if cfg.get("bench"):
        base += BENCH_S
    return base


def wall_estimate(generators, measured):
    """Low and high wall estimates, in seconds."""
    low = GENERATOR_S * len(generators)
    high = int(low * MEASURED_HIGH_FACTOR)
    for cfg in measured:
        base = arm_base_seconds(cfg)
        low += base
        high += base * (UNMEASURED_HIGH_FACTOR if cfg["body"] == "sorted"
                        else MEASURED_HIGH_FACTOR)
    return int(low) + COLD_CACHE_LOW_S, int(high) + COLD_CACHE_HIGH_S


def main():
    arms = selected_arms()
    generators, measured = build_plan(arms)
    if "--dry-run" in sys.argv:
        low, high = wall_estimate(generators, measured)
        cell = cell_for()
        print(f"mg13 plan: {len(measured)} measured arms + {len(generators)} "
              f"untimed generator arm")
        print(f"  cell {cell}, slices {num_slices_for(cell)}, warm repeats "
              f"{WARM_REPEATS}, iterations {VCD_ITERATIONS}, seed {VCD_SEED}, "
              f"device {DEVICE}, results {RESULTS_DIR}")
        try:
            print(f"  tree: {tree_root()}"
                  + ("" if os.path.isdir(os.path.join(tree_root(), "mbirtorch"))
                     else "   [NO mbirtorch PACKAGE THERE]"))
        except RuntimeError as exc:
            print(f"  tree: NOT SET -- {exc}")
        print("  the three bodies:")
        for body in BODIES:
            print(f"    {body:<9} {KILL_SWITCH}="
                  f"{BODY_ENV[body]['disable_triton']} "
                  f"{PATCH_ENV_VAR}={BODY_ENV[body]['patch_sorted']}")
            print(f"    {'':<9} {BODY_NOTE[body]}")
        for cfg in generators:
            print(f"  {cfg['arm_id']:<44} (generator, one device, untimed)")
        for cfg in measured:
            print(f"  {cfg['arm_id']:<44} body={cfg['body']:<8} "
                  f"n={cfg['n_dev']}  bench={bool(cfg.get('bench'))}  "
                  f"~{arm_base_seconds(cfg) / 60:.0f} min")
        print(f"  wall estimate {low / 60:.0f} to {high / 60:.0f} minutes "
              f"(a sorted arm has never been run, so its high end assumes half "
              f"again as much; plus {COLD_CACHE_LOW_S / 60:.0f} to "
              f"{COLD_CACHE_HIGH_S / 60:.0f} minutes for a cold inductor cache)")
        print("  if it must be cut: MG13_COUNTS drops whole device counts and "
              "MG13_ARMS drops")
        print("  single arms.  Trim WHOLE DEVICE COUNTS -- a count missing any "
              "of its three")
        print("  bodies has no comparison block at all, and the torch scatter "
              "arm is where both")
        print("  verdict lines' denominator comes from.  Trim n=4 first: n=1 "
              "is the question.")
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            f"mg13_sorted_probe_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg13 sorted probe on {RUN_LABEL} ({DEVICE}); {len(measured)} arms "
          f"-> {out_path}", flush=True)
    rows = []
    # Rows write INCREMENTALLY: a truncated job still yields the device counts it
    # finished, which is why the arm order runs a count's three bodies
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
    if os.environ.get("MG13_KEEP_ARTIFACTS", "0") != "1":
        # The sinogram and the value samples are internal to this job -- the
        # distances are computed above, before anything is removed.
        for path in (_sino_path(cell_for()), _md5_path(cell_for())):
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
