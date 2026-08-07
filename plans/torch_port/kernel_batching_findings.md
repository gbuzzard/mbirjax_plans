# Kernel-aware view batching — findings (checkpoint 2)

**Status:** IMPLEMENTED AND MEASURED, awaiting Fable review at the
checkpoint-2 STOP.  The mechanism of `kernel_batching_design.md` is
implemented and default-active.  The local suite is green at 295
passed against the 279 baseline, and the CUDA battery is green on H100
at 72 passed (job 14931291).  The view-chunk sweep ran 112 rows (job
14931348) and pinned all four chunk constants at 128, the design's
starting value.  The sweep's one surprise, an e-3-class value column
on the back paths, was attributed end to end to the COMPILED reference
body's per-shape sum reassociation (job 14932618); the kernels and the
batching are exonerated by that attribution.  The composed five-arm
gates passed both cells of both geometries, every cell improved, and
the tables are below.  No default beyond the approved design changed.

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
These forwards are atomic-kernel-bound rather than launch-bound,
which re-measures the close-out's decomposition.  The parallel
forward also still loses its full-pixel-set bench to the compiled
body at 0.80x, matching the phase5 isolated reading.  Subset rows
improved 1.02x to 1.46x.

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
attribution.  The e-3 column measured the compiled reference's
per-shape reassociation of the long view and tap sums.  The kernels
reproduce the eager bodies they were validated against, so the
reading is a property of the ruler, not of the measured.  The norm-rel of
the compiled-vs-eager difference is 3.3e-5 with 0.15 percent of
elements over 1e-4, an isolated-tail shape, and the composed gates'
5e-3 recon-level envelope already absorbs exactly this class (the
phase5 composed kernel-vs-body columns were 2.3e-3 to 3.2e-3).

## The composed five-arm gates

The gates re-ran for both geometries at both cells (job 14932647, H100,
warm seeded 3-iteration vcd, pinned constants, chunk 128 everywhere).
Every arm check passed: the bodies bound matched each arm's
expectation, the realized view batches matched each bound body's own
formula in both directions, and the launch-key deltas witnessed the
kernels inside the recons.  The back-only arms ran the mixed selection
in production form, with the back direction batching at its chunk
while the fallen-back torch forward batched at the legacy 1 or 10.

| geometry, cell | both kernels | back only | pure torch | jax | torch/jax | was | mem/jax |
|---|---|---|---|---|---|---|---|
| parallel 512 | 1.87 s | 2.33 s | 4.87 s | 1.65 s | **1.13x** | 1.21x | 0.63x |
| parallel 1024 | 40.3 s | 51.0 s | 143.5 s | 25.9 s | **1.56x** | 1.90x | 0.59x |
| cone 512 | 2.79 s | 4.79 s | 9.46 s | 3.18 s | **0.88x** | 1.00x | 0.60x |
| cone 1024 | 62.9 s | 118.5 s | 245.1 s | 63.2 s | **1.00x** | 1.18x | 0.57x |

The replacement rule passes every cell with margin, and every cell
improved.  Cone now runs AT or BELOW jax at both cells (0.88x and
1.00x).  Parallel-1024, the cell the sorted-stream decision reads,
moved from 1.90x to 1.56x.  The batching recovered 10.8 s of the
composed 51.1 s at that cell.  The remaining 14.4 s over jax belongs
to the forward, which is the close-out's decomposition re-measured:
the back kernel is now worth 2.81x composed (143.5 s pure torch to
51.0 s back-only), and the forward kernel's marginal win over the
back-only arm is 1.27x.

Two controls anchor the table.  The pure-torch arms landed on the
phase5 numbers within 0.4 percent (143.5 vs 143.6 s, 245.1 vs
245.8 s, 4.87 vs 4.86 s, 9.46 vs 9.46 s).  These numbers are the
measured form of constraint (b): nothing in the torch-body path
changed.  The jax arms re-measured in the same run, so every ratio is
same-run and same-node.  jax's cone-512 read 3.18 s here against
phase5's 3.08 s, which is why the ratio, not the absolute, is the
gate's number.

The memory columns eased rather than rising.  The composed kernel-arm
peaks at the 1024 cells are 26.7 GB, at 0.57-0.59x of jax.  The
design named the chunked batches' transients as the risk that could
spend the 0.56-0.63x headroom, and that risk did not materialize in
composition: the batch transients sit under the recon state that
already dominates the composed peak.  The peaks are measured, never
inferred.

The value columns hold their established classes.  Kernel-vs-body:
3.17e-3 and 2.32e-3 (parallel), 2.75e-4 and 9.74e-5 (cone), all
within the 5e-3 envelope beside floors of 1.3e-6 to 1.8e-5.  The
shared-sinogram cross-framework columns: 5.4e-4 (parallel 512),
8.4e-5 and 3.0e-4 (cone), and 6.11e-3 at parallel-1024 -- the one
over-envelope number, and it REPRODUCES the phase5 closure's shared
sinogram residual (6.094e-3 there) to three digits, with the kernels
invisible in it (kernel arm 6.11e-3 vs body arm 6.10e-3).  That
residual is the documented cross-framework trajectory spread of
unconverged 3-iteration runs, not a kernel or batching effect.

## The sorted-stream input (for the goal-2 decision)

The decision number is parallel-1024 at 1.56x of jax after batching.
The composed remainder belongs to the forward.  The both-kernels arm
spends 40.3 s where jax spends 25.9 s, and the back kernel's composed
share is now small (the 51.0 s back-only arm against the 40.3 s
both-kernels arm).  The sweep's full-range forward rows are flat
across every batch size, so batching has nothing further to give
there; the forward is atomic-kernel-bound and runs at 0.80x of the
compiled body at full P.  Two candidates stand where the phase5
design left them: the cheap view-loop forward variant recorded in the
kernel's docstring, and the two-phase sorted-stream forward with plan
caching.  The decision is Fable's on these numbers.

## Files this checkpoint staged

Every file below was staged by explicit name; Greg committed the
mbirtorch set (b6fe992) and the earlier plans-repo set mid-session,
and the final edits to this document are the staged remainder.

Checkpoint-2 files, mbirtorch:
`mbirtorch/projectors.py`, `mbirtorch/tomography_model.py`,
`mbirtorch/parallel_beam.py`, `mbirtorch/cone_beam.py`,
`mbirtorch/triton_parallel.py`, `mbirtorch/triton_cone.py`,
`tests/test_view_batching.py` (new), `tests/test_triton_parallel.py`,
`tests/test_triton_cone.py`.

Checkpoint-2 files, mbirjax_plans:
`plans/torch_port/kernel_batching_design.md` (checkpoint-1 notes folded
in), `plans/torch_port/kernel_batching_findings.md` (this document),
`plans/experiments/torch_port/kb1_gautschi.sbatch`,
`plans/experiments/torch_port/kb2_vbsweep.py`,
`plans/experiments/torch_port/kb2_gautschi.sbatch`,
`plans/experiments/torch_port/kb3_gate.py`,
`plans/experiments/torch_port/kb3_gautschi.sbatch`,
`plans/experiments/torch_port/kb4_value_attrib.py`,
`plans/experiments/torch_port/kb4_gautschi.sbatch`.

Raw rows stay on scratch per convention:
`/scratch/gautschi/buzzard/torch_p3/results/kb2_vbsweep_h003_20260807_135908.jsonl`,
`.../kb3_gate_h*.jsonl`, `.../kb4_value_attrib_h*.json`, and the slurm
logs `kb1_14931291.log`, `kb2_14931348.log`, `kb3_14932647.log`,
`kb4_14932618.log`.
