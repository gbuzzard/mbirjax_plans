# Device policy checkpoint 2: the memory ledger — findings

**Status:** IMPLEMENTED AND MEASURED, awaiting Fable review at the
checkpoint-2 STOP.  The ledger of `device_policy_design.md` is implemented
with the checkpoint-1 scope change applied.  Under that change the preflight
gates only the automatic multi-device path, and the calibration mode computes
the ledger at any device count.  The local suite is green at 387 passed
against a re-measured 354 baseline.  The H100 calibration passes the
acceptance band at all five cells.

**Terms.**  This page keeps the design's names.  The LEDGER is the per-device
peak model.  The PREFLIGHT is the runtime check that consumes it.  The
INITIAL ERROR STATE is the phase that builds the error sinogram, and the
ERROR SINOGRAM ASSIGNMENT is the line inside it that forms the array.

## The headline

The ledger envelops the measured peak at every cell, and it does so after
five attributed corrections rather than after a fitted constant.  The first
measurement put four of five cells below the acceptance floor, at ratios of
0.805 to 0.911.  A phase-resolved probe attributed every one of those gaps to
a specific array or a specific line of the engine.  The corrected ledger
reads 1.001 to 1.169.

| geometry, cell | arm | modeled | measured | ratio | error | dominant phase |
|---|---|---|---|---|---|---|
| parallel 512 | weighted | 2.29 GB | 2.26 GB | **1.014** | +1.4% | initial error state |
| parallel 1024 | weighted | 26.71 GB | 26.68 GB | **1.001** | +0.1% | initial error state |
| cone 512 | weighted | 2.64 GB | 2.26 GB | **1.169** | +16.9% | hessian diagonal |
| cone 1024 | weighted | 26.71 GB | 26.68 GB | **1.001** | +0.1% | initial error state |
| parallel 1024 | unweighted | 24.53 GB | 23.71 GB | **1.034** | +3.4% | hessian diagonal |

The acceptance band is 1.00 to 1.30, and every cell is inside it.  The table
comes from one run: job 14948975 on an H100, n=1, kernels on, warm seeded
3-iteration vcd.  Job 14948118 produced the same five ratios to three
decimals on a different allocation, which bounds run-to-run drift.

Two controls anchor the table.  The measured warm peak at both 1024 cells is
26.68 GB, which reproduces the composed gate's 26.7 GB to the printed digit.
The warm times reproduce the composed gate arms within 1.5 percent: 1.90
against 1.87 s, 40.2 against 40.3 s, 2.78 against 2.79 s, and 62.7 against
62.9 s.  These results indicate that the calibration run measures the same
configuration the gates measure, so the ratios are comparable to the recorded
baselines.

## Every phase envelops, not only the peak

The band is stated on the peak, but a phase that under-charges today could
dominate under another configuration.  The phase probe's measured phases and
the final ledger share the same engine, because only the ledger changed
between them, so the two can be compared phase by phase.

| phase, parallel 1024 weighted | measured | modeled | ratio |
|---|---|---|---|
| direct recon | 21.45 GB | 22.21 GB | 1.035 |
| initial error state | 26.68 GB | 26.71 GB | 1.001 |
| hessian diagonal | 23.71 GB | 24.53 GB | 1.035 |
| subset step, coarsest granularity | 23.22 GB | 23.43 GB | 1.009 |

Every phase envelops its measurement.  The tightest is the initial error
state, which is the phase the peak sits on.

## The design named the wrong dominant phase, and the probe found why

The checkpoint-1 design's worked calculation built its peak from the subset
back projection.  The measurement names the initial error state at three of
the five cells.  That phase is pure state.  It holds seven named arrays and
no projector transient at its peak.

The phase-resolved probe made the attribution possible.  It ran as job
14946798.  The probe wraps the production methods rather than
re-implementing them, resetting the peak counter on entry and reading it on
exit.  Its entry column reports `memory_allocated` at the same instant, which
separates a wrong persistent set from a wrong transient.

At parallel 1024 weighted, the probe read 26.68 GB inside
`_initial_error_state` against an entry of 11.43 GB.  That one function's
transient is therefore 15.25 GB.

Counting the arrays live at the error sinogram assignment reproduces the
reading closely.  Six sinogram-shaped arrays are live there: the sinogram,
the weights, the forward projection, the weighted forward projection, the
alpha-scaled projection, and the error sinogram.  The initial volume and the
partitions bring the enumeration to 26.65 GB against the 26.68 GB measured,
an agreement of 0.12 percent.

## The five corrections

**One: the back projection holds three cylinder arrays, not two.**  The
driver's loop is `block = back_body(...)` followed by `out.add_(block)`.
Python evaluates the call before it rebinds the name, so the previous block
is still alive while the kernel produces the next one.  Charging two put the
measured direct recon at 0.892 and the measured hessian at 0.876.  One
mechanism accounts for both phases at both cells.

**Two: supplied weights are live in the phases before the loop.**  The
weights are placed at the top of `vcd_recon`, so they are resident through
the direct recon and the initial error state.  The ledger charged them only
from the hessian onward.

The probe's entry column also pinned WHEN the weights array exists, which is
not the same question as whether it exists.  An unweighted run has no
weights-shaped array until the hessian block builds its all-ones sinogram.
The unweighted direct recon enters at 3.92 GB and the weighted one at
7.73 GB.  That difference is exactly one sinogram.  The ledger now
distinguishes a supplied weights array from an internally built one.

**Three: the single-device error state holds a weighted forward projection it
never releases.**  `_initial_error_state` binds
`weighted_fwd = weights * fwd` at `tomography_model.py:1042` and uses it for
two dot products.  The name stays bound until the function returns, so the
array is still resident at the error sinogram assignment on line 1049.  The
sharded branch has no such array, because it fuses the weights into per-shard
dot products whose locals die on worker return.  This is a second avoidable
residency of the same class as `hess_weights`.

**Four: the error sinogram assignment allocates two sinogram-shaped arrays.**
`sinogram - alpha * fwd` allocates the scaled projection, then allocates the
difference.  Both are live at the assignment.  The ledger charged only the
result.

**Five: the direct recon's back loop and its scatter are not co-live.**  This
is the one correction in the over-charging direction.  The back accumulator
is consumed by the scatter, so the two are consecutive sub-peaks rather than
one sum.

The first version of correction five carried a defect of its own, found in
review.  It picked one whole sub-phase by comparing cross-device TOTALS,
which would under-charge a device whose larger sub-phase lost the comparison.
The ledger now emits both sub-phases and lets the per-device maximum choose,
so the selection is per device.  The n=1 numbers are unchanged, which the
two matching calibration runs confirm.

## The residual was a constant, so it is a term and not a factor

After the five corrections three cells still read just under the floor, at
0.986, 0.9988 and 0.9988.  The residual was +32.0, +33.0 and +33.0 MiB.

That residual is flat across a twelvefold range of peak size, from a
2.26 GiB cell to a 26.68 GiB cell.  These results indicate a fixed
per-process allocation rather than a missing array term, because a missing
array would scale with the problem.  The size and the architecture match the
cuBLAS workspace, which is 32 MiB per stream on this class of device.  That
workspace is allocated through torch's own caching allocator, so
`max_memory_allocated` counts it and the ledger's array enumeration cannot
see it.

The ledger charges it as a named term at 64 MiB.  That covers the measurement
with headroom, at a cost of 2.8 percent at the smallest gate cell and 0.2
percent at the largest.  This satisfies the design's rule that a cell below
the floor is fixed by adding the missing term and never by a scale factor.
Here the missing term happens to be constant.

## The two residencies, sized — and only one of them pays

The unweighted probe cell exists to size the `hess_weights` residency, which
a weighted run cannot show.  On a weighted run `hess_weights` is a bare alias
of the caller's weights, so no all-ones sinogram is ever built.

The unweighted arm measured 23.71 GB against the weighted arm's 26.68 GB.
The unweighted run therefore uses 2.97 GB LESS, not more.  The reason is that
the weighted run holds two extra sinogram-shaped arrays that the unweighted
run does not: the caller's weights, and the weighted forward projection of
correction three.  The unweighted run holds one that the weighted run does
not, which is the all-ones sinogram.

The validated ledger was then asked what each proposed fix would buy, and it
answered that only one of them pays.  Releasing `weighted_fwd` before the
error sinogram assignment moves the weighted 1024 peak from 26.71 GB to
24.53 GB.  That is an 8.2 percent reduction, and it hands the dominant phase
to the hessian.

Releasing `hess_weights` after the hessian call moves no peak at all.  The
all-ones sinogram is unavoidably live INSIDE the hessian phase, which is
where it is built and consumed, and that phase is larger than every subset
phase at both 1024 arms.  Releasing it afterwards shrinks the subset phases
by one sinogram each and leaves the peak where it is.  The fix becomes worth
making at a cell whose dominant phase is a subset step, and it is not worth
making at these cells.

This correction matters beyond the two fixes.  The first draft of this page
predicted a 12 percent gain from the `hess_weights` fix, and the ledger
itself refuted the prediction.  These results indicate the ledger is already
useful as a design instrument and not only as a gate.

## Cone 512 is the loosest cell, and it is loose in the safe direction

Cone 512 reads 1.169, which is inside the band and is the largest
over-charge.  Its dominant phase is the hessian diagonal, where the ledger
over-charges by 327 MiB before the workspace term.  Cone's `_transient_cols`
is params-derived rather than runtime-derived, so its modeled batch charge
exceeds the realized one at the small cell.  No cell is near the 1.30
ceiling.

## What did not change

The n=1 path is untouched in behavior.  The ledger is built only under the
calibration mode or on the automatic multi-device path, and neither is active
for a default single-device or explicitly configured run.  The refactor of
`_effective_view_batch` into `view_batch_charge` preserves the existing
signature and its default argument, so every production call site chooses the
same view batch as before.  `tests/test_view_batching.py` pins that and
passes unchanged.

The local suite is green.  The charter's 295-passed baseline predates other
landed work, so the baseline was re-measured rather than assumed.  The
re-measurement used a scratch copy of the working tree with only this
checkpoint's files reverted.  That copy runs 354 passed and 56 skipped.  The
working tree runs 387 passed and 56 skipped.  The delta is exactly the 33
tests of the new file, and no pre-existing test changed status.

## What checkpoint 2 did NOT ship

The preflight has no production call site yet, and that follows from the
checkpoint-1 scope change rather than from being unfinished.  The preflight
gates only the automatic multi-device path, and that path is checkpoint 3's
work.  Four pieces are implemented and unit-tested against synthetic budgets:
the budget reader, the resident-array credits, the per-device verdict, and
the readable shortfall message.  Checkpoint 3 wires them to the widening
rule.

## Open items for review

**1. The projector batch charge under-states the kernels' real residency.**
The hessian phase at parallel 1024 left about 2.7 GB unexplained by the
enumerated arrays before correction one.  That works out to roughly 45 bytes
per view-pixel.  The parallel back kernel's cost model charges 16 bytes per
view-pixel.  Correction one absorbed the gap at every measured cell, so no
cell needs this today.  The under-statement is real nonetheless.
`projectors.py` already carries a TODO estimating the true per-view transient
at two to five times the nominal slab.  Recommendation: record it, and
revisit if a future cell's dominant phase is projector-bound.

**2. The two residencies.**  Recommendation: land the `weighted_fwd` fix as
its own change and check the measured result against the predicted 26.71 to
24.53 GB move.  Leave `hess_weights` alone for now, and record that its fix
buys nothing at any current gate cell.

**3. The acceptance band held without adjustment.**  The upper bound of 1.30
was flagged at checkpoint 1 as the number most likely to need revision.  It
did not need revision.  The loosest cell is 1.169.

**4. A shared-checkout note.**  Concurrent sessions committed most of this
checkpoint's files while the work was in progress.  `9cf6252` swept in
`mbirtorch/projectors.py`, and `d11a253` swept in the remaining mbirtorch
files.  Every change is correct as committed and nothing needs undoing.  It
is recorded because it makes the file list below read differently from the
usual checkpoint record.

## Files

Every file below carries this checkpoint's work.  Most are already committed,
by the concurrent sessions of open item 4 rather than by this one, so the
list separates the two states as they stand at the time of writing.

Committed in mbirtorch:
`mbirtorch/_memory_ledger.py` (new), `mbirtorch/projectors.py`,
`mbirtorch/tomography_model.py`, `tests/test_memory_ledger.py` (new).  The
mbirtorch working tree is clean, and the committed `_memory_ledger.py`
includes the per-device sub-phase fix.

Committed in mbirjax_plans:
`plans/torch_port/device_policy_design.md` (the checkpoint-1 ruling),
`plans/experiments/torch_port/dp2_ledger_calib.py`,
`plans/experiments/torch_port/dp2_gautschi.sbatch`,
`plans/experiments/torch_port/dp3_phase_probe.py`,
`plans/experiments/torch_port/dp3_gautschi.sbatch`.

Staged in mbirjax_plans:
`plans/torch_port/device_policy_findings.md` (this document).

Raw rows stay on scratch per convention.  The accepted table is
`/scratch/gautschi/buzzard/torch_p3/results/dp2_ledger_calib_20260807_212442.jsonl`.
Its confirming duplicate is `.../dp2_ledger_calib_20260807_210333.jsonl`, the
first pre-correction table is `.../dp2_ledger_calib_20260807_202856.jsonl`,
and the attribution is `.../dp3_phase_probe_*.json`.  The slurm logs are
`dp2_14948975.log`, `dp2_14948118.log`, `dp2_14946338.log`, and
`dp3_14946798.log`.
