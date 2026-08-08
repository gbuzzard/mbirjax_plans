# Device policy checkpoint 2: the memory ledger — findings

**Status:** IMPLEMENTED AND MEASURED, awaiting Fable review at the
checkpoint-2 STOP.  The ledger of `device_policy_design.md` is implemented
with the checkpoint-1 scope change applied: the preflight gates only the
automatic multi-device path, and the calibration mode computes the ledger at
any device count.  The local suite is green at 386 passed against a measured
354 baseline.  The H100 calibration passes the acceptance band at all five
cells.

## The headline

The ledger envelopes the measured peak at every cell, and it does so after
five attributed corrections rather than after a fitted constant.  The first
measurement put four of five cells BELOW the acceptance floor, at ratios of
0.805 to 0.911.  A phase-resolved probe attributed every one of those gaps to
a specific array or a specific line of the engine.  The corrected ledger
reads 1.001 to 1.169.

| geometry, cell | arm | modeled | measured | ratio | error | dominant phase |
|---|---|---|---|---|---|---|
| parallel 512 | weighted | 2.29 GB | 2.26 GB | **1.014** | +1.4% | error sinogram formation |
| parallel 1024 | weighted | 26.71 GB | 26.68 GB | **1.001** | +0.1% | error sinogram formation |
| cone 512 | weighted | 2.64 GB | 2.26 GB | **1.169** | +16.9% | hessian diagonal |
| cone 1024 | weighted | 26.71 GB | 26.68 GB | **1.001** | +0.1% | error sinogram formation |
| parallel 1024 | unweighted | 24.53 GB | 23.71 GB | **1.034** | +3.4% | hessian diagonal |

The acceptance band is 1.00 to 1.30 and every cell is inside it.  Job
14948118, H100, n=1, kernels on, warm seeded 3-iteration vcd.

Two controls anchor the table.  The measured warm peak at both 1024 cells is
26.68 GB, which reproduces the composed gate's 26.7 GB to the printed digit.
The warm times reproduce the composed gate arms within one percent: 1.9
against 1.87 s, 40.2 against 40.3 s, 2.8 against 2.79 s, and 62.7 against
62.9 s.  These results indicate that the calibration run measures the same
configuration the gates measure, so the ratios are comparable to the recorded
baselines.

## The design's dominant phase was wrong, and the probe found why

The checkpoint-1 design named the subset back projection as the dominant
phase.  The measurement names the INITIAL ERROR STATE at three of the five
cells.  That phase is pure state: it holds seven named arrays and no
projector transient at its peak.

The phase-resolved probe is what made the attribution possible (job
14946798).  It wraps the production methods rather than re-implementing them,
resetting the peak counter on entry and reading it on exit.  Its entry column
reports `memory_allocated` at the same instant, which separates a wrong
PERSISTENT set from a wrong transient.  At parallel 1024 weighted the probe
read 26.68 GB inside `_initial_error_state` against an entry of 11.43 GB, so
that one function's transient is 15.25 GB.

Counting the arrays live at `error_sinogram = sinogram - alpha * fwd`
reproduces the reading to 0.2 percent.  Six sinogram-shaped arrays are live
there: the sinogram, the weights, the forward projection, the weighted
forward projection, the alpha-scaled projection, and the error sinogram.  The
initial volume and the partitions bring the total to 26.63 GB against the
26.68 GB measured.

## The five corrections

**One: the back projection holds three cylinder arrays, not two.**  The
driver's loop is `block = back_body(...)` followed by `out.add_(block)`.
Python evaluates the call before it rebinds the name, so the previous block
is still alive while the kernel produces the next one.  Charging two put the
direct recon at 0.892 and the hessian at 0.876 of their measured peaks.
Charging three puts them at 1.040 and 1.039, and the same correction fits the
512 cell at 1.068 and 1.068.  One mechanism, four cells.

**Two: supplied weights are live in the phases before the loop.**  The
weights are placed at the top of `vcd_recon`, so they are resident through
the direct recon and the initial error state.  The ledger charged them only
from the hessian onward.

The probe's entry column also pinned WHEN the weights array exists, which is
not the same question as whether it exists.  An unweighted run has no
weights-shaped array until the hessian block builds its all-ones sinogram.
The unweighted direct recon enters at 3.92 GB and the weighted one at
7.73 GB, a difference of exactly one sinogram.  The ledger now distinguishes
a supplied weights array from an internally built one.

**Three: the single-device error state holds a weighted forward projection it
never releases.**  `_initial_error_state` binds
`weighted_fwd = weights * fwd` and uses it for two dot products.  The name
stays bound until the function returns, so the array is still resident when
the error sinogram is formed.  The SHARDED branch has no such array, because
it fuses the weights into per-shard dot products whose locals die on worker
return.  This is a second avoidable residency of the same class as
`hess_weights`, and it is the larger of the two on a weighted run.

**Four: forming the error sinogram allocates two sinogram-shaped arrays.**
`sinogram - alpha * fwd` allocates the scaled projection, then allocates the
difference.  Both are live at the assignment.  The ledger charged only the
result.

**Five: the direct recon's back loop and its scatter are not co-live.**  This
is the one correction in the over-charging direction.  The back accumulator
is consumed by the scatter, so the two are consecutive sub-peaks rather than
one sum.  The ledger now takes their maximum.

## The residual was a constant, so it is a term and not a factor

After the five corrections three cells still read just under the floor, at
0.986, 0.9988 and 0.9988.  The residual was +32.0, +33.0 and +33.0 MiB.

That residual is flat across a twelvefold range of peak size, from a 2.4 GB
cell to a 28.6 GB cell.  These results indicate a fixed per-process
allocation rather than a missing array term, because a missing array would
scale with the problem.  The size and the architecture match the cuBLAS
workspace, which is 32 MiB per stream on this class of device.  The workspace
is allocated through torch's own caching allocator, so
`max_memory_allocated` counts it and the ledger's array enumeration cannot
see it.

The ledger charges it as a named term at 64 MiB.  That covers the measurement
with headroom, at a cost of 2.6 percent at the smallest gate cell and 0.2
percent at the largest.  This satisfies the design's rule that a cell below
the floor is fixed by adding the missing term and never by a scale factor:
the missing term here happens to be constant.

## The two residencies, sized

The unweighted probe cell exists to size the `hess_weights` residency, which
a weighted run cannot show.  On a weighted run `hess_weights` is a bare alias
of the caller's weights, so no ones array is ever built.

The unweighted arm measured 23.71 GB against the weighted arm's 26.68 GB.
The unweighted run therefore uses 2.97 GB LESS, not more.  The reason is that
the weighted run holds two extra sinogram-shaped arrays that the unweighted
run does not: the caller's weights, and the weighted forward projection of
correction three.  The unweighted run holds one that the weighted run does
not, which is the all-ones array.

The now-validated ledger predicts what each fix would buy, and these are the
predictions the code fixes should be checked against.  Releasing
`weighted_fwd` before the error formation moves the weighted 1024 peak from
26.63 GB to 24.60 GB, a 7.6 percent reduction, and hands the dominant phase
to the hessian.  Releasing `hess_weights` after the hessian moves the
unweighted 1024 peak from 24.60 GB to 21.60 GB, a 12 percent reduction, and
hands the dominant phase to the qGGMRF prior.  Neither fix changes any value.

## Cone 512 is the loosest cell, and it is loose in the safe direction

Cone 512 reads 1.169, which is inside the band and is the largest
over-charge.  Its dominant phase is the hessian diagonal, where the ledger
over-charges by 327 MiB before the workspace term.  Cone's `_transient_cols`
is params-derived rather than runtime-derived, so its modeled batch charge
exceeds the realized one at the small cell.  The over-charge is on the safe
side of the band and no cell is near the 1.30 ceiling.

## What did not change

The n=1 path is untouched in behavior.  The ledger is built only under the
calibration mode or on the automatic multi-device path, and neither is active
for a default single-device or explicitly configured run.  The refactor of
`_effective_view_batch` into `view_batch_charge` preserves the existing
signature and its default argument, so every production call site chooses the
same view batch as before.  `tests/test_view_batching.py` pins that and
passes unchanged.

The local suite is green.  The charter's 295-passed baseline predates other
landed work, so the baseline was re-measured rather than assumed: a scratch
copy of the working tree with only this checkpoint's files reverted runs 354
passed and 56 skipped.  The working tree runs 386 passed and 56 skipped.  The
delta is exactly the 32 tests of the new file, and no pre-existing test
changed status.

## What checkpoint 2 did NOT ship

The preflight has no production call site yet, and that follows from the
checkpoint-1 scope change rather than from being unfinished.  The preflight
gates only the automatic widening path, and that path is checkpoint 3's work.
Everything the call site needs is implemented and unit-tested against
synthetic budgets: the budget reader, the resident-array credits, the
per-device verdict, and the readable shortfall message.  Checkpoint 3 wires
them to the widening rule.

## Open items for review

**1. The projector batch charge under-states the kernels' real residency.**
The hessian phase at parallel 1024 leaves about 2.7 GB unexplained by the
enumerated arrays, which works out to roughly 45 bytes per view-pixel against
the 16 the parallel back kernel's cost model charges.  The three-cylinder
correction absorbed this at every measured cell, so no cell needs it today.
The under-statement is real nonetheless, and `projectors.py` already carries
a TODO estimating the true per-view transient at two to five times the
nominal slab.  Recommendation: record it, and revisit if a future cell's
dominant phase is projector-bound.

**2. The two residencies.**  Both are one-line fixes with predicted effects
above.  Recommendation: land them as a single change after this checkpoint,
and check the measured result against the predictions rather than merely
against a lower number.

**3. The acceptance band held without adjustment.**  The upper bound of 1.30
was flagged at checkpoint 1 as the number most likely to need revision.  It
did not: the loosest cell is 1.169.

## Staged files

mbirtorch:
`mbirtorch/_memory_ledger.py` (new), `mbirtorch/projectors.py`,
`mbirtorch/tomography_model.py`, `tests/test_memory_ledger.py` (new).

mbirjax_plans:
`plans/torch_port/device_policy_design.md` (checkpoint-1 ruling recorded),
`plans/torch_port/device_policy_findings.md` (this document),
`plans/experiments/torch_port/dp2_ledger_calib.py`,
`plans/experiments/torch_port/dp2_gautschi.sbatch`,
`plans/experiments/torch_port/dp3_phase_probe.py`,
`plans/experiments/torch_port/dp3_gautschi.sbatch`.

Raw rows stay on scratch per convention:
`/scratch/gautschi/buzzard/torch_p3/results/dp2_ledger_calib_20260807_210333.jsonl`
(the accepted table), `.../dp2_ledger_calib_20260807_202856.jsonl` (the first,
pre-correction table), `.../dp3_phase_probe_*.json` (the attribution), and the
slurm logs `dp2_14948118.log`, `dp2_14946338.log`, `dp3_14946798.log`.
