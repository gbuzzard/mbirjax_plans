# Kernel-aware view batching — design

**Status:** APPROVED, IN EXECUTION (2026-08-07).  Checkpoint 1 of the
mbirtorch kernel follow-ups (current_plans.md item 1 — formerly §5; renumbered 2026-08-07) passed Fable
review with three notes, all folded in below: the 512 "today" batch
corrected to the measured 10 (the earlier ~5 assumed cubic cells; the
gate cells are (views, rows, channels) tuples), the `view_batch_size`
docstrings updated to int-or-None where users read them, and the chunk
pins chosen from the joint time-and-memory readout with the 1024
forward's co-residency named.
**Prior art:** the "batching rule is implementation-supplied" paragraph
of `phase5_kernel_design.md`, the driver's view-range loop and budget
in `mbirtorch/projectors.py`, and the parallel-1024 decomposition in
`phase5_findings.md`.

## The defect

The driver charges every body the torch body's transient when it
chooses a view batch.  `Projectors._effective_view_batch` computes the
batch as `min(view_batch_size, budget // (P * transient_cols * 4))`,
where `transient_cols` comes from the geometry's `_transient_cols`
hook.  That hook models the torch bodies' materialized gather: the
runtime band length for parallel beam, and `max(num_slices, num_rows)`
for cone.  The model is real and calibrated for the torch bodies.  It
is fiction for the Triton kernels, whose whole design is that the
gather transient is never written.

The miscount is most expensive at the 1024 gate cells.  There the
torch body's transient is about 3.1 GB per view against the 2 GiB
budget, so the driver runs view batch 1 for every full-pixel-set call:
the error-sinogram initialization forward, the Hessian back
projection, and the public `forward_project` / `back_project`.  Each
such projection becomes 1024 separate kernel launches, and each launch
pays its own hfan contract build.  Subset calls are miscounted too:
the counted charge caps them near view batch 40 at the 1024 cell, and
the 64 default caps them at 512, while the phase5 design says these
register-tile kernels want view chunks near 128.

The composed stakes are set by the close-out's decomposition.  It
measures the back kernel at about 12 s of the parallel-1024 composed
51.1 s, and it notes that even a free back projector would leave about
39 s against jax's 26.9 s, so the forward is the larger gap.  Per-view
launch and build overhead from this defect inflates both directions.
What its repair actually recovers is the checkpoint-2 measurement, and
the sorted-stream decision (current_plans item 1, goal 2) is made on the remainder.

## What the design must satisfy

The work order fixes four constraints.  (a) Mixed selection works: the
forward and back bodies are selected independently, the self-check can
fall either one back to the torch body at `create_projectors` time,
and the cost model must follow the body actually bound, never the
geometry.  (b) The torch bodies keep exactly their current batching,
including under the kill switch: their transient model is calibrated,
and behavior with kernels off changes zero.  (c) The kernel cost model
states what a kernel batch is actually resident in and derives a
view-chunk choice with a swept-constant cap, starting near 128.
(d) The calibration warning on `_transient_cols` applies: changing
batch sizes changes float summation order and measured peaks, so the
re-gates are named in this document.

## The mechanism: a cost attribute on the body function

The batching rule rides as an attribute on the kernel body function.
Each Triton wrapper module attaches a small module-level cost function
to its wrapper, under the name `_view_batch_cost`.  The driver reads
that attribute off the body it is about to call, and it falls back to
the current `_transient_cols` path when the attribute is absent.

The attribute contract is one function per kernel:

    _view_batch_cost(num_pixels, cols, args) -> (bytes_per_view, view_chunk)

`num_pixels` and `cols` are the two values the driver already passes
to `_effective_view_batch` (`cols` is `band_values.shape[-1]` on the
forward site and `local_sino.shape[1]` on the back site).  `args` is
the geometry's `_view_batch_args()` dict, which the driver holds at
both sites; the cost function lives in the same module as its kernel
and reads its own geometry's keys (`num_channels`, and `num_rows_r`
for the cone forward).  `bytes_per_view` is the charged residency of
one view in a kernel batch.  `view_chunk` is the kernel's nominal
batch, a swept module constant pinned beside the kernel's other
pinned constants.

The driver rule becomes:

    cost = getattr(body, '_view_batch_cost', None)
    nominal = model.view_batch_size          # None means automatic
    if cost is None:
        cols_t = model._transient_cols(cols)
        cap = budget // max(1, num_pixels * cols_t * 4)
        vb = max(1, min(64 if nominal is None else nominal, cap))
    else:
        bytes_per_view, view_chunk = cost(num_pixels, cols, args)
        cap = budget // max(1, bytes_per_view)
        vb = max(1, min(view_chunk if nominal is None else nominal, cap))

The `cost is None` branch is today's rule, with `None` resolving to
the 64 the default has always been (a named constant in the
implementation).  `_effective_view_batch` gains the bound body and the
args dict as parameters; both call sites have them in hand.  The body
consulted is `self._fwd_body_per_dev[dev_index]` or
`self._back_body_per_dev[dev_index]`, the function object actually
about to be called.  `maybe_compile` returns a marker-carrying kernel
body unchanged, so the attribute survives binding.  A torch body comes
back either as itself or as a compiled wrapper, and neither carries
the attribute, so absence of the attribute is exactly the condition
"the torch body is bound".

This shape satisfies constraint (a) without any new bookkeeping.  The
forward and back bodies are consulted separately, so a kernel back
batches by its cost function while a fallen-back torch forward batches
by `_transient_cols`.  A self-check fallback at `create_projectors`
time binds the torch body, and the attribute read follows that binding
with no second decision.  If selection ever became per device, the
per-`dev_index` read would follow it unchanged.

The phase5 design sketched this rule as "`_transient_cols` generalized
to the kernel's cost model", and this design deliberately deviates
from that sketch in one way: `_transient_cols` itself is untouched.
The hook is a geometry method, so it would have to re-derive the
selection instead of following the bound function, and re-derivation
is the drift constraint (a) warns about.  The intent of the phase5
sentence — the batching rule is implementation-supplied — is preserved
by moving the supply point onto the implementation itself.

Two alternative shapes were considered and rejected.  A parallel
geometry hook returning per-body cost models would have to repeat the
selection logic of `_view_batch_bodies`, and the two copies could
disagree; disagreement between selection and cost model is
unrepresentable when the cost rides on the selected function.  A
driver-side registry mapping known kernel functions to cost models
would keep the knowledge outside the module that owns it and would
need maintenance for every new kernel.  The attribute also has direct
precedent: `_mbirtorch_no_compile` already travels on these same
functions, and the driver already honors it.

## The kernel cost model

Each kernel batch is resident in three things: the hfan contract (plus
the vertical affine pair for cone), one sinogram-shaped tile per view,
and the call's fixed tensors.  The charged `bytes_per_view` covers the
first two, which are the terms that scale with the batch:

- Parallel contract: `n_p` (f32) and `centers` (i32) are (Vb, P), 8
  bytes per (view, pixel); `W_p_c` and `weight_scale` ride as per-view
  scalars.  The builder holds `x` and one expression temporary
  alongside them at its peak.  Charge: **16 bytes per (view, pixel)**.
- Cone contract: `n_p`, `W_p_c`, `weight_scale`, `pixel_mag`, `m0`,
  `W_p_r` (f32) and `centers` (i32) are (Vb, P), 28 bytes per
  (view, pixel); the builders hold `x`, `y`, `u`, `theta` and
  expression temporaries alongside them at their peak.  Charge:
  **48 bytes per (view, pixel)**.
- Sinogram tile: the back wrappers copy the batch channel-major
  (`sino_t`), and the forward wrappers zero their channel-major
  output.  Both are 4 * num_channels * R bytes per view, where R is
  `cols` for the parallel pair and the cone back, and `num_rows_r`
  (band-independent) for the cone forward.

Call-fixed tensors are excluded from the per-view charge: the back
output (P, band), the forward `values` (P, band), and the driver's
assembled full-range output.  These exist at any batch size, so the
batch choice cannot control them, and the torch-body budget never
charged their analogs either.

The charges are counted from the builder and wrapper source, and a
count is an estimate, not a measurement.  The existing budget comment
in `projectors.py` records the lesson: recalibrate against a measured
per-view delta, never against the nominal slab.  The division of labor
follows that lesson: the charged coefficients protect the budget
boundary, and the swept chunk constant is the performance chooser.
The gates re-measure the real peaks (constraint (d)).  A modest
coefficient error therefore moves nothing except calls already near
the budget.

The budget itself is unchanged: the kernel path divides the same
`_transient_budget_bytes()` the torch path uses.  One
memory-protection concept serves both paths.  Its small-cell scaling
binds harmlessly under the kernels, because the chunk cap binds first
at small cells, and its 2 GiB large-cell cap is what bounds the new
larger kernel batches.

Four new pinned constants carry the chunk caps, one per kernel, named
like the existing pinned constants (for example
`PARALLEL_BACK_VIEW_CHUNK`), starting at 128 per the phase5 design and
pinned by the checkpoint-2 sweep.  128 also divides both gate cells'
view counts exactly, so full-range projections run without a tail
batch at the starting value.

## The one default change: view_batch_size becomes None

`view_batch_size` is the user's single memory/speed knob, and today it
defaults to 64.  A 64 nominal would silently cap every kernel below
the 128-class chunks the sweep is meant to explore, so the constructor
default (in `TomographyModel`, `ParallelBeamModel`, and
`ConeBeamModel`) becomes `None`, meaning automatic.  The driver
resolves `None` per body: 64 for a torch body, and the chunk constant
for a kernel body.  The 64 is the value that has always been the
default, kept as a named constant.  An explicit integer keeps its
exact current meaning for every body: a nominal that the budget may
cap further.  A memory-constrained user's small explicit value
therefore still bounds the kernel batches.  Torch-body behavior is
unchanged in every case, which is what constraint (b) requires.  This
is the design's only default change, called out here so its approval
is explicit.  The constructor docstrings change with it: the
`view_batch_size` entries become int-or-None and state the resolution
rule, so the default change is documented where users read it
(review note 2).

## What does not change

The following stay untouched, per constraint (b):

- the `_transient_cols` implementations and their calibration;
- the torch bodies and their compile path;
- the budget constants and `_transient_budget_bytes`;
- the kill switch: with kernels unselected no attribute is ever
  present, so batching is bit-identical to today;
- the availability self-checks, which call bodies directly and never
  batch;
- the banded sharded drivers, which pass through the same two
  view-range loops;
- the `plan` slot.

## Predicted operating points (counted; measurement decides)

The gate cells are (views, rows, channels) = (512, 448, 384) and
(1024, 1008, 992), the Phase 2/3/4 cells the sweep and gate harnesses
carry.  The counted model gives the following view batches, with P the
full ROR pixel set of each cell (about 1.16e5 and 7.73e5 pixels):

| call, cell | today | designed |
|---|---|---|
| parallel back/fwd, full P, 1024 | 1 | 128 (chunk-capped; budget cap ~131) |
| cone back/fwd, full P, 1024 | 1 | ~52 (budget-capped) |
| parallel and cone, full P, 512 | 10 | 128 (chunk-capped) |
| subset calls, 1024 | ~44 | 128 (chunk-capped) |
| subset calls, 512 | 64 | 128 (chunk-capped) |

An earlier draft read ~5 in the 512 "today" row, from assuming cubic
cells.  The real 512 cell's torch charge is P x 448 x 4 ≈ 208 MB per
view against the 2 GiB budget, which gives 10 — exactly the batch the
p5k6 sweep measured for both directions.  The same correction anchors
the 1024 row: the real cell's charge is ~3.1 GB per view, matching the
close-out.  The legacy sweep row records the actuals this table is
checked against either way.

These numbers predict the 1024-cell full-range projections collapsing
from 1024 launches to 8 (parallel) and 20 (cone), and the subset calls
doubling to tripling their batch.  The predicted risk runs the other
way on memory.  A kernel batch may now hold transients up to the
2 GiB budget where today's defective batches held tens of MB, so the
composed peak can rise.  The rise is real, not hypothetical: at the
full-range 1024 forward the contract alone is ~1.6 GB at chunk 128
(16 bytes x 7.7e5 pixels x 128 views), and it is co-resident with the
4 GB assembled full-range output, so that call's peak grows by roughly
2 GB over today's.  The current kernel arms sit at 0.56–0.63x of jax's
memory against a ~1.5x replacement-rule ceiling, which is the headroom
the chunk pin spends.  The gate's memory columns are re-measured,
never inferred, and the pin is chosen from the sweep's joint time and
peak readout, never from time alone.

## Compile and launch-key consequences

New batch sizes compile new kernel variants, and this is expected, not
a bug.  Each wrapper's launch key includes the view-batch size, so a
new batch size adds one launch key; a missed key costs one uncontended
lock acquisition.  Triton itself specializes integer arguments by
bucket (equal-to-1, divisibility by 16), so the distinct compile count
stays small.  The checkpoint-2 sweep records compile counts per row;
if variant churn appears, the budget-capped batch can floor to a
multiple of 16, and that refinement is deliberately not baked in now.
When kernels are off, no shape any torch body sees changes.

## Validation and the named re-gates

Changing batch sizes changes float summation order and measured peaks,
so the calibration warning on `_transient_cols` applies to this change
in full.  The required validation:

1. **Unit tests, CPU-runnable.**  `_effective_view_batch` becomes
   directly testable with any function object: a stub body carrying
   `_view_batch_cost` gets the derived min/floor batch; a body without
   the attribute reproduces today's numbers exactly (pinned cases for
   the parallel base rule and the cone override); mixed stubs batch
   each direction by its own rule; `view_batch_size=None` resolves to
   64 for torch bodies; an explicit value caps both paths.
2. **The CUDA battery.**  `tests/test_triton_cone.py` and
   `tests/test_triton_parallel.py` re-run green on H100, extended
   with: a driver-level equality test (a kernel body through the
   view-range loop, chunked versus a single reference batching,
   within summation-order tolerance), and a regression test that the
   kernel batch at a body-transient-bound shape exceeds 1.
3. **The view-chunk sweep.**  Both geometries, both directions, both
   gate cells, at a full-P and a subset-P pixel class, chunk swept
   over {16, 32, 64, 128, 256} plus a legacy row that reproduces
   today's charged batch for comparison.  Harness modeled on
   `p5k6_psweep.py`: subprocess-isolated rows, strict env parse,
   per-row value checks against the torch body at the same inputs,
   kernel-repeat floors for the atomic forwards, and per-row peak
   memory and compile counts.  The sweep pins the four chunk
   constants.
4. **The composed five-arm gates**, both geometries, both cells,
   modeled on `p5k6_pgate.py` with runtime arm checks.  The arm
   checks pin each arm's expected bodies and additionally assert the
   realized view batch per direction, so a silently-inactive cost
   attribute fails the arm check rather than shipping a null result
   (the phase5 arm-check lesson).  Value columns: kernel-vs-body
   within the composed envelopes beside measured repeat floors, and
   the cross-framework comparison hands ONE shared sinogram artifact
   to both frameworks per `p5ka_shared_gate.py`.  Memory columns
   re-measured.  The pure-torch arm doubles as the constraint-(b)
   control: it must land on the phase5 pure-torch numbers within
   noise, since nothing in its path changed.

The baselines the gate table reads against: parallel 1.21x / 1.90x,
cone 1.00x / 1.18x of jax time, memory 0.56–0.63x.  The
parallel-1024 result is the number the sorted-stream go/no-go
(current_plans item 1, goal 2) is decided on, and that decision is Fable's.

## Checkpoint-2 increments

1. Implement the driver rule, the four cost functions and chunk
   constants, and the `view_batch_size=None` default; add the unit
   tests; local suite green against the 279-passed / 52-skipped
   baseline.
2. Sync to gautschi under the per-file scp + md5 rule; run the CUDA
   battery.
3. Run the view-chunk sweep; pin the four constants.
4. Run the composed five-arm gates for both geometries; report the
   gate table against the baselines and STOP for Fable review.

---

**Checkpoint-1 staged files:**
`plans/torch_port/kernel_batching_design.md` (this document, in
mbirjax_plans).
