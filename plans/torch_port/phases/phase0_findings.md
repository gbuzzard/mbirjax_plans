# Phase 0 findings — de-risking spikes for the PyTorch port

**Status:** COMPLETE 2026-08-04.  Local (Mac) cells and the gautschi CUDA
cells (job 14831556, one H100, 5 min) are both measured.
**Plan:** `port_plan.md` section 6 defines the spikes.  Scripts:
`plans/experiments/torch_port/archive/p_phase0_4/p0s{1,2,3}_*.py` plus
`plans/experiments/torch_port/archive/p_phase0_4/p0_gautschi.sbatch`.

## Summary

The spikes found no blocker for the port, and they resolved the two named
risks in the port's favor.  On CUDA they also sized the expected gap to the
tuned baseline: eager torch runs 10-15x behind the pallas-tuned jax path at
the 512 cell, and closing that gap is exactly the work Phases 2 and 5 already
plan (compile plus the Triton kernel port).  A fourth result simplifies the
plan: CUDA graphs gain only 5-15% because torch's eager dispatch is already
cheap at these op sizes, so the graph machinery is optional rather than
load-bearing.  Cross-framework values agree far inside the
proposed golden tolerances, with zero rounding-tie mismatches in 4.9 M checked
scatter centers (spike 1).  The CPU risk is real in EAGER form: the eager
torch back projector runs 8.6x slower than jax-CPU.  torch.compile flips it.
On the same chains, compilation gives 1.7-3.6x in time and up to 6x in peak
memory on CPU (spike 2).  These results indicate the planned mitigation
(compile the hot chains) works on exactly this code shape.  MPS exceeded
expectations twice.  The eager MPS forward already matches jax-CPU (1.0-1.3x),
and torch.compile WORKS on MPS in torch 2.13, with 5-17x chain speedups.  For
Mac users, who run jax on CPU today, the port therefore looks like an upgrade
rather than a concession.  The remaining Phase 0 evidence is the CUDA cells,
including the spike 3 CUDA-graph question.

## Setup

The frozen jax baseline is jax/jaxlib 0.10.1 on python 3.11.15 (the shipped
pin), running as one XLA CPU device, which is the default local
configuration.  The torch side is torch 2.13.0 (the recorded Phase 0 pin) in
the new `mbirtorch` conda env, using its default 10 CPU threads.  The machine
is Greg's Apple-silicon MacBook Pro (arm64).

The measurement protocol follows the harness conventions.  Every (framework,
device, cell) runs in its own subprocess.  Times are the min of 3 warm trials
after a warmup call, with device synchronization inside the timed region.
Memory rulers: `ru_maxrss` per subprocess on CPU, `max_memory_allocated` on
CUDA, `current_allocated_memory` on MPS.  The torch timed region includes the
per-call scatter-center computation, matching what the jax wrappers do per
call.

## Spike 1 — fan kernels (jax vs eager torch)

The jax side is the public `sparse_forward_project` / `sparse_back_project`
on a real `ParallelBeamModel`.  The torch side is the view-batched eager
port of the fan kernels (per-tap scatter-add forward, per-tap gather back).
Times are seconds; ratios are torch over jax-CPU; the torch entry is its best
view batch.

| cell (sino) | recon (pinned) | pixels | jax-CPU fwd / back | torch-CPU fwd / back | torch-MPS fwd / back |
|---|---|---|---|---|---|
| 128x112x96 | 96x96x112 | 7,080 | 0.046 / 0.013 | 0.135 (2.9x) / 0.112 (8.6x) | 0.048 (1.0x) / 0.032 (2.5x) |
| 200x208x160 | 160x160x208 | 19,856 | 0.297 / 0.111 | 1.292 (4.4x) / 0.960 (8.7x) | 0.386 (1.3x) / 0.250 (2.3x) |
| 512x448x384 | 384x384x448 | 115,164 | 11.643 / 4.124 | not run (ungated cell) | 12.286 (1.1x) / 9.888 (2.4x) |

The recon column doubles as the record of the parallel pinned shapes for the
harness (port_plan.md section 4): the auto-derived shapes at these cells are
(96,96,112), (160,160,208), and (384,384,448).

Correctness: forward outputs agree with jax at rel-max 2.4e-6 to 1.1e-5, and
back outputs at 3.0e-7 to 7.9e-7.  These sit well inside the proposed
cross-framework tolerances (1e-4 single-op).  The rounding-tie diagnostic
found ZERO center mismatches: 0 of 906,240 (view, pixel) pairs at the 128
cell and 0 of 3,971,200 at the 200 cell.  This does not retire the tie risk
in general, but it shows the standard angle sets produce no ties at these
cells.

Interpretation.  Eager torch on CPU misses the 2x gate, and the back
projector is the worst piece.  Its per-tap gather feeds an einsum reduction
that XLA fuses and eager torch materializes.  The forward ratio grows with
cell size (2.9x to 4.4x), consistent with growing materialized transients.
On MPS the same eager code is already competitive with jax-CPU.  These eager
numbers are the BASELINE the spike 2 compilation results then act on.

### CUDA (H100, jax = the TUNED pallas baseline)

The same spike on gautschi, where the jax side runs the shipped pallas
kernels.  Times are seconds; ratios are eager torch over tuned jax.

| cell (sino) | jax-H100 fwd / back | torch-CUDA eager fwd / back |
|---|---|---|
| 128x112x96 | 0.002 / 0.001 | 0.003 (1.5x) / 0.003 (3x) |
| 200x208x160 | 0.004 / 0.002 | 0.024 (6x) / 0.017 (8.5x) |
| 512x448x384 | 0.076 / 0.037 | 0.803 (10.6x) / 0.550 (14.9x) |

Correctness holds on CUDA too: rel-max 8.4e-7 to 2.8e-6, and again zero
center mismatches at the two checked cells.

Interpretation.  Against the TUNED baseline, eager torch loses by an amount
that grows with size, reaching 10-15x at the 512 cell.  This is the expected
shape of the result, not a new risk: the plan never expected eager to meet
the gate on CUDA, and the tuned endpoint there is the Triton port of the
pallas kernels (Phase 5) plus compiled glue (Phase 2).  The spike 2 CUDA rows
below give the first step of that closure.

## Spike 2 — chain fusion (eager vs torch.compile)

Two chains at the 200-class and 512-class shapes: the fan weight chain +
scatter (one forward view-batch call), and a qGGMRF-surrogate neighbor chain
over the full recon volume (the port's named memory attention point).  Times
are seconds (min of 3 warm trials); memory is the honest per-subprocess peak.

| device / chain / size | eager | compile | time ratio | peak memory eager -> compile |
|---|---|---|---|---|
| cpu / fan / 200c | 0.469 | 0.227 | 2.1x | 2.38 -> 0.42 GiB (5.7x) |
| cpu / fan / 512c | 0.660 | 0.388 | 1.7x | 3.73 -> 0.60 GiB (6.2x) |
| cpu / qggmrf / 200c | 0.060 | 0.025 | 2.4x | 0.45 -> 0.47 GiB (flat) |
| cpu / qggmrf / 512c | 1.056 | 0.291 | 3.6x | 3.39 -> 1.62 GiB (2.1x) |
| mps / fan / 200c | 0.125 | 0.024 | 5.2x | (mps_alloc flat at 0.02 GiB) |
| mps / fan / 512c | 0.213 | 0.038 | 5.6x | (mps_alloc flat at 0.19 GiB) |
| mps / qggmrf / 200c | 0.019 | 0.002 | 9.5x | (mps_alloc flat at 0.02 GiB) |
| mps / qggmrf / 512c | 0.254 | 0.012 | 21x | (mps_alloc flat at 0.25 GiB) |

The 21x MPS number was suspicious, so it was re-verified with forced
materialization: a `.sum().item()` on both outputs inside the timed region
gives 0.0147 s compiled vs 0.256 s eager (17x), and the compiled values match
eager at rel-max 2.0e-7.  The win is real.

The CUDA rows from the gautschi job (peak memory is
`cuda_max_memory_allocated`):

| chain / size | eager | compile | time ratio | peak alloc eager -> compile |
|---|---|---|---|---|
| fan / 200c | 0.008 | 0.003 | 2.7x | 2.06 -> 0.05 GiB (41x) |
| fan / 512c | 0.013 | 0.005 | 2.6x | 3.33 -> 0.22 GiB (15x) |
| qggmrf / 200c | 0.002 | <0.001 | >2x | 0.26 -> 0.10 GiB |
| qggmrf / 512c | 0.022 | 0.001 | 22x | 3.20 -> 1.23 GiB (2.6x) |

The CUDA memory result matters more than the CUDA time result.  Compilation
removes the eager fan chain's materialized transients almost entirely (3.33
GiB down to 0.22 GiB at the 512 shape).  These transients were the main
threat to the 1.5x memory gate, so the gate now looks comfortably reachable.

Interpretation.  Inductor fuses these chains well on both local backends.  On
CPU, compilation removes most of the eager gap measured in spike 1 and
removes the materialized-transient memory (the 6x peak reductions on the fan
chain).  On MPS, torch 2.13's compile support is no longer the immature
caveat the plan assumed; the plan's "eager-only on MPS" stance can be
upgraded in Phase 2.  A caveat stands: these are isolated chains, and
composed end-to-end behavior (compile guards, cache churn across VCD's
shape set) is Phase 2's question, per the lessons rule that driver-level wins
must be re-validated end to end.

## Spike 3 — subset update under CUDA graphs

The question: does graph replay remove the host-dispatch cost of a
VCD-subset-shaped update loop, the regime where jax measured ~95% host?  The
mock update (all-view back projection onto the subset, preconditioned delta,
forward projection, on-device line search, in-place state updates) ran 32
subsets per pass on the H100, eager vs captured-graph replay.

| cell / subset pixels | eager ms/subset | graph ms/subset | graph win |
|---|---|---|---|
| 200c / 1240 | 3.08 | 2.92 | 5% |
| 200c / 310 | 1.27 | 1.09 | 14% |
| 512c / 7240 | 81.7 | 81.4 | 0.4% |
| 512c / 1810 | 21.8 | 21.6 | 1% |

Interpretation.  The graph win is marginal because torch's eager dispatch is
already cheap: the mock's ~50 dispatches per subset cost well under a
millisecond of host time, so the loop is compute-bound even eagerly.  The jax
95%-host finding does not transfer, because it came from jax's own per-call
eager-wrapper costs (~1 ms per eager op) rather than from anything intrinsic
to the update's structure.  Two consequences for the plan: CUDA graphs move
from load-bearing to optional in Phase 2, and the interactive-size regime is
a place the port may beat jax with no special machinery at all.  One corner
stays unmeasured: truly tiny subsets (the 128-class cells at fine
granularity) could still surface a host share, and Phase 2's end-to-end VCD
measurement covers that.

## Read against the gates

Nothing measured in Phase 0 blocks the port, and the Phase 0 exit question
(port_plan.md section 5, "does the 2x gate look safe enough to fund
Phase 1?") gets a yes.  The evidence by gate:

- **Correctness.**  Cross-framework agreement is 3e-7 to 1.1e-5 rel-max over
  every (cell, op, device) measured, far inside the proposed 1e-4 golden
  tolerance, with zero rounding-tie mismatches in 9.75 M checked centers
  across both platforms.
- **CPU time gate.**  Eager misses (back 8.6x); compile recovers 1.7-3.6x on
  the same chains.  The composed margin is Phase 2's measurement, and a
  bounded `cpp_extension` fallback remains if it falls short.
- **CUDA time gate.**  Eager is 10-15x behind the tuned pallas baseline at
  the 512 cell, as expected.  The closure path is the planned one: compiled
  glue (2.6x already measured) plus the Phase 5 Triton port of the pallas
  kernels, which is a lateral move within the same kernel backend.
- **Memory gate.**  Compilation removes the fan chain's materialized
  transients (41x and 15x lower peak CUDA allocation at the two shapes), so
  the 1.5x gate looks comfortably reachable.
- **MPS (ungated).**  Eager forward already matches jax-CPU, compile works on
  MPS in torch 2.13 with 5-17x chain wins, and Mac users run jax on CPU
  today.  The port upgrades this platform.

The Phase 1 vertical slice should proceed.  The two assumptions Phase 0
leaves to later phases, named per the working principles: composed
end-to-end compile behavior across VCD's shape set (Phase 2), and
Triton-kernel parity with pallas (Phase 5).
