# Kernel-aware view batching — findings (checkpoint 2)

**Status:** IMPLEMENTED AND MEASURED, awaiting Fable review at the
checkpoint-2 STOP.  The mechanism of `kernel_batching_design.md` is
implemented and default-active; the local suite is green (295 passed,
52 + 4-new skipped, against the 279/52 baseline); the CUDA battery is
green on H100 (72 passed, job 14931291); the view-chunk sweep ran 112
rows (job 14931348) and pinned all four chunk constants at 128, the
design's starting value; and the sweep's one surprise, an e-3-class
value column on the back paths, was attributed end to end to the
COMPILED reference body's per-shape sum reassociation, not to the
kernels and not to the batching (job 14932618).  The composed five-arm
gates for both geometries are recorded below.  No default beyond the
approved design changed.

## What shipped

The driver now batches each body by the body's own cost model.  A
Triton kernel body carries a `_view_batch_cost` attribute (attached in
its own module) returning its charged bytes per view and its nominal
view chunk; `Projectors._effective_view_batch` reads the attribute off
the body actually bound per direction and per device, and a body
without the attribute batches exactly as before through the geometry's
`_transient_cols`.  The four chunk constants ship beside the kernels'
other pinned constants.  `view_batch_size` defaults to `None`
(automatic: 64 for a torch body, the chunk constant for a kernel
body), and an explicit integer still caps every body; the three
constructor docstrings document the rule (review note 2).  New tests
pin both dispatch paths, the mixed selection, the `None` resolution,
and the explicit ceiling on CPU, and two new CUDA tests per battery
file check the chunked view-range loop against a single-batch
reference and the realized batches through a real default selection.

## The CUDA battery

The battery re-ran green on the first submission after one test fix:
72 passed (68 baseline + 4 new), with all four availability
self-checks reading 1.4e-7 to 6.8e-7.  The one fix was to a NEW test
of this change, not to shipped code: at the tiny cone cell the torch
charge (P x 12 cols x 4 bytes) numerically equals the kernel charge
(48 bytes per view-pixel), so a `kernel batch != torch batch`
assertion failed on a coincidence of the two formulas.  The dispatch
proof did not need the inequality (the same test pins kernel-chunk-128
against torch-default-64 at the same inputs), so the fragile assertion
was dropped and the coincidence is recorded in the test's comment.

## The view-chunk sweep

The sweep measured the production view-range loop, not an isolated
wrapper call: 112 subprocess-isolated rows over both geometries, both
directions, both gate cells, and two pixel classes (the full ROR set
and a stride-64 subset), with a torch-reference arm, a legacy-batching
arm (the kernels under the OLD charged batches, emulated through an
explicit `view_batch_size`), and chunk values {16, 32, 64, 128, 256}.
Every row passed its arm check, its realized batch matched its
formula, and the duplicate rows where the budget cap binds below
several chunk values (cone full-1024 realizes 52 at chunks 64 to 256)
measured a 0.6 percent repeat noise floor.

The batching repair itself is the legacy-vs-chunked comparison at the
same kernels.  The full-range 1024 back projections, which the old
rule ran at view batch 1, moved most: parallel 4406 to 792 ms (5.56x)
and cone 7477 to 4517 ms (1.66x).  The full-range forwards moved
almost nothing (parallel 7246 to 7189 ms, cone 8270 to 7996 ms).
These forwards are atomic-kernel-bound rather than launch-bound, which
is the same decomposition the close-out predicted, and the parallel
forward still LOSES its full-pixel-set bench to the compiled body
(0.80x) exactly as phase5 measured in isolation.  Subset rows improved
1.02x to 1.46x.

The pin is 128 for all four kernels, the design's starting value.  At
the 1024 cells chunk 128 sits within 1.7 percent of the best chunk on
every row, inside twice the measured noise floor.  Chunk 256 wins
several subset rows, by 1.0 to 14.7 percent, but its wins concentrate
on 4-6 ms calls at the 512 cells while it costs 0.4 to 1.0 GB more
peak per call class; priced jointly (review note 3), the memory buys
nothing back, so 256 was declined.  Chunk 64 wins only the two
full-1024 back rows, by 0.6 to 1.3 percent, which is within noise, and
it loses subset rows by 4 to 20 percent.  One pinned value for all
four kernels follows the phase5 pin-one-config discipline.

Two operating-point notes for the record.  The realized batches
matched the design table exactly: full-1024 runs at 128 (parallel,
budget cap 131) and 52 (cone, budget-capped), full-512 at 128, and
every subset class at 128.  The peak columns behaved as the design
predicted: the chunked full-1024 back rows peak at 13.8 GB against
the legacy rows' 12.6 GB, and the composed peaks are re-measured at
the gate, never inferred from these rows.

## The e-3 value column, attributed to the ruler

Three sweep combos read a value column of 1.7e-3 to 2.2e-3 against the
1e-4 flag: parallel back at every multi-view-reference shape, and cone
back at full-512.  Two facts in the sweep data already bounded the
cause.  The reading was IDENTICAL at every chunk value and in the
legacy row, so the batching change could not be the cause.  The
back-path repeat floors were exactly zero, so nondeterminism could not
be either.

The attribution ran as one single-variable probe (kb4, job 14932618):
the same full-range driver output computed three ways -- eager torch
body, compiled torch body (the sweep's reference), and kernel -- and
diffed pairwise.  At 512-parallel-back-full the kernel matches the
EAGER body at max-rel 7.4e-7 with zero of 51.6M elements over 1e-4,
while compiled-vs-eager reads 1.74e-3 with 77k elements over, and
kernel-vs-compiled reads the same 1.74e-3.  At 512-cone-back-full the
same pattern holds (kernel-vs-eager 2.5e-6; compiled-vs-eager 2.15e-3
at 580 elements).  At the clean control (1024-cone-back-subset) all
three pairs sit at or under 5.4e-6.  These results close the
attribution: the e-3 column measured the compiled reference's
per-shape reassociation of the long view/tap sums, the kernels
reproduce the eager bodies they were validated against, and the
reading is a property of the ruler, not the measured.  The norm-rel of
the compiled-vs-eager difference is 3.3e-5 with 0.15 percent of
elements over 1e-4, an isolated-tail shape, and the composed gates'
5e-3 recon-level envelope already absorbs exactly this class (the
phase5 composed kernel-vs-body columns were 2.3e-3 to 3.2e-3).

## The composed five-arm gates

(Pending: job 14932647 in flight; this section is filled from its
table before the checkpoint record is staged.)

## Staged files

(Filled with the checkpoint report.)
