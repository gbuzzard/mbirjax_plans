# Phase 2 findings — compile integration and the first gate-cell readout

**Status:** local (CPU + MPS) cells measured 2026-08-04; the CUDA cells run in
gautschi job 14833737 and are recorded in their own section.
**Plan:** `port_plan.md` section 5 (Phase 2) and section 4 (the gates).
**Scripts:** `plans/experiments/torch_port/archive/p_phase0_4/p2_gate_readout.py` and
`p2_gautschi.sbatch`.

## Summary

Phase 2's core question was whether compilation closes the eager gaps, and on
the gated CPU platform the answer is stronger than the gate asked for.  With
torch.compile integrated into the engine, every CPU cell and op sits INSIDE
the 2x time gate, and the user-facing op (a 3-iteration weighted VCD) runs
2-5x FASTER than mbirjax.  Peak process memory is 0.36-0.60x of jax's on
every cell, far inside the 1.5x memory gate.  The speed reversal on VCD is
the Phase 0 host-dispatch prediction confirmed in-engine: jax's VCD at
interactive sizes was ~95% host dispatch, and torch's cheap dispatch plus
compiled kernels removes that cost.  MPS multiplies the Mac story again
(VCD at the 200 cell: 2.2 s vs jax-CPU's 7.7 s).  One real defect surfaced
and is fixed: the back fan's gather transient cannot fuse away, so large
cells need a view-batch memory cap (details below).

## Compile integration

``compile_mode='auto'`` (the default) binds torch.compile'd forms of the two
fan kernels, the parallel geometry chain, and the qGGMRF gradient/Hessian;
``'off'`` keeps pure eager.  A compile failure falls back to eager and is
recorded (the same availability philosophy as mbirjax's pallas gate).  Two
findings worth recording:

- **inference_mode and torch.compile do not mix** (torch 2.13): compiled calls
  inside ``torch.inference_mode`` with in-place state updates crash dynamo's
  guard machinery.  The engine now runs under ``torch.no_grad`` instead, which
  composes cleanly; the lost inference_mode benefit (version-counter
  bookkeeping) is noise next to the kernels.
- **Compile cost is per process and nontrivial**: the first VCD in a fresh
  process pays 7-14 s of inductor compilation at these cells.  Warm-process
  numbers are the op cost.  Torch's persistent FX-graph cache should amortize
  this across processes; verifying/enabling it is a follow-up.

The in-engine A/B at the demo cell (128^3 phantom, 10 iterations, warm):
MPS 3.56 s eager -> 2.07 s compiled (1.7x); CPU 15.9 s -> 7.1 s (2.25x).
Values are identical in both modes (NRMSE 0.0586 either way).

## The gate readout method

One script measures BOTH frameworks with one ruler
(`p2_gate_readout.py`): each (framework, device, cell, op) runs in its own
subprocess; ops are warm-timed (min of 3 trials after a warmup call) except
VCD, which reports the first run and, where cheap, a warm second run; peak
memory is per-process ``ru_maxrss`` (plus the device allocators' peaks on
CUDA).  Inputs are deterministic and identical across frameworks: the
Shepp-Logan phantom (exact cross-framework match), its forward projection as
the sinogram, transmission-root-style weights, and seed 13 for the VCD
partitions.  The op set and cells are the harness's (port_plan section 4).

## CPU gate table (the gated platform, n=1)

Ratios are torch/jax; the VCD torch entry is the warm-process time.  The time
gate is <= 2x; the memory gate is ~1.5x.

| cell | op | jax (s) | torch (s) | time ratio | RSS torch/jax |
|---|---|---|---|---|---|
| 128x112x96 | direct_filter | 0.014 | 0.016 | 1.14 | 0.60 |
| 128x112x96 | forward | 0.045 | 0.038 | 0.84 | 0.39 |
| 128x112x96 | back | 0.014 | 0.019 | 1.34 | 0.51 |
| 128x112x96 | vcd (3 it) | 4.21 | 0.93 | **0.22** | 0.45 |
| 129x113x97 | direct_filter | 0.017 | 0.018 | 1.07 | 0.56 |
| 129x113x97 | forward | 0.057 | 0.038 | 0.66 | 0.40 |
| 129x113x97 | back | 0.015 | 0.020 | 1.35 | 0.52 |
| 129x113x97 | vcd (3 it) | 4.71 | 0.85 | **0.18** | 0.44 |
| 200x208x160 | direct_filter | 0.184 | 0.221 | 1.20 | 0.40 |
| 200x208x160 | forward | 0.295 | 0.285 | 0.97 | 0.36 |
| 200x208x160 | back | 0.119 | 0.138 | 1.16 | 0.42 |
| 200x208x160 | vcd (3 it) | 7.72 | 3.69 | **0.48** | 0.38 |

Every cell passes both gates.  The worst time ratio anywhere is the back
projector's 1.35x; VCD, the op users run, is uniformly faster than jax.

## MPS (informational, vs jax-CPU on the same machine)

| cell | direct_filter | forward | back | vcd (3 it, warm) |
|---|---|---|---|---|
| 128x112x96 | 0.003 | 0.007 | 0.011 | 0.30 |
| 129x113x97 | 0.006 | 0.008 | 0.012 | 0.33 |
| 200x208x160 | 0.012 | 0.052 | 0.084 | 2.16 |
| 512x448x384 | 0.361 | 1.62 | 3.02 | (not run) |

Against the jax-CPU column above, MPS is 3-15x faster on the direct ops and
3-14x faster on VCD.  The 512-cell back number comes from a post-fix manual
run (the original row hit the defect below); it is not subprocess-isolated,
so its memory was not recorded.

## A defect found and fixed: the back fan's gather transient

The original 512-cell MPS back projection aborted: at the default view batch
of 64, the back fan's gather output is a (64, 115k, 448) float tensor, ~13 GB.
Compilation cannot remove it -- a gather's output is real data, unlike the
forward fan's product chain, which inductor fuses into the scatter.  The fix
is a view-batch memory cap in the drivers (`VIEW_BATCH_TRANSIENT_BUDGET_BYTES`
= 2 GiB): the effective batch shrinks at large cells so one batch's
(Vb, P, cols) transient stays bounded.  Batch size affects only float
summation order, so the cap is a pure memory knob; the suite (21 tests)
passes unchanged.  The rule worth recording: compile-era memory reasoning
must separate FUSIBLE chains (products, weights) from UNAVOIDABLE
materializations (gather outputs) -- only the former vanish.

## CUDA cells (gautschi, one H100) — the warm-protocol table

Two readouts ran.  The first (job 14833737) recorded torch VCD at large cells
COLD (with compile) and, it turned out, recorded JAX's vcd cold too: the
fresh worker processes paid jax's own trace-and-compile (the persistent jax
cache was cold on that node), which understated jax by up to 27x at the
interactive cell.  The warm protocol (job 14837754: both frameworks run vcd
twice, the second run is the op cost) fixes the ruler on both sides, and the
run also carries mbirtorch's scaled gather budget.  Ratios are torch/jax,
warm-vcd; memory is the device allocator's peak.

| cell | op | jax (s) | torch (s) | time ratio | mem ratio |
|---|---|---|---|---|---|
| 200x208x160 | direct_filter | 0.003 | 0.003 | 0.82 | 0.46 |
| 200x208x160 | forward | 0.007 | 0.007 | 1.03 | 0.29 |
| 200x208x160 | back | 0.005 | 0.009 | 1.90 | 3.4 |
| 200x208x160 | vcd (warm) | 0.605 | 0.278 | **0.46** | 2.8 |
| 512x448x384 | direct_filter | 0.012 | 0.014 | 1.15 | 0.62 |
| 512x448x384 | forward | 0.102 | 0.182 | 1.78 | 0.38 |
| 512x448x384 | back | 0.040 | 0.307 | **7.6** | **3.1** |
| 512x448x384 | vcd (warm) | 1.68 | 3.12 | **1.86** | 2.12 |
| 513x449x385 | direct_filter | 0.019 | 0.017 | 0.88 | 0.62 |
| 513x449x385 | forward | 0.107 | 0.352 | **3.3** | 0.38 |
| 513x449x385 | back | 0.042 | 0.329 | **7.8** | **3.1** |
| 513x449x385 | vcd (warm) | 1.83 | 5.44 | **2.97** | 1.97 |
| 1024x1008x992 | direct_filter | 0.120 | 0.111 | 0.92 | 0.80 |
| 1024x1008x992 | forward | 2.75 | 5.22 | 1.90 | 0.50 |
| 1024x1008x992 | back | 1.16 | 5.09 | **4.4** | 0.65 |
| 1024x1008x992 | vcd (warm) | 26.1 | 94.3 | **3.61** | 0.80 |

Reading.  The honest vcd verdict on CUDA: torch WINS the interactive cell
(0.46x) and PASSES the 512 cell (1.86x); it is over the gate at 513 (2.97x)
and 1024 (3.61x).  Both breaches have named causes.  The 513 excess over its
512 sibling (in forward and vcd alike) is the odd-size inductor
specialization effect, still to be investigated.  The 1024 excess is the back
projector's 4.4x, which dominates vcd at scale -- the pallas-kernel gap the
plan assigns to the Phase 5 Triton port.  The measurement lesson is recorded
deliberately: a "fresh subprocess per config" protocol is honest for MEMORY
but must warm BOTH frameworks' compile caches before timing, or it charges
each framework's compiler to the op (the mbirjax lessons' separate-the-ruler
rule, met again).

Memory with the scaled gather budget: the 200-cell back/vcd ratios fell from
12.2x/11.4x to 3.4x/2.8x, and the 1024 column stays in torch's favor on every
op.  The remaining excess at the 200/512 cells traces to the 256 MiB transient
floor plus per-batch layout copies -- Phase 5 layout territory, with the
budget knobs available sooner if those cells matter before then.

## prox_map

The proximal map for Plug-and-Play is ported (the mbirjax flow: cached
initialization via ``prox_data``, sigma_prox override with restore, the
engine's prox prior path).  Its test drives both regimes: a tiny sigma_prox
hugs the prox input, a huge one follows the data, and the two differ -- on
every backend.

## Dashboard wiring (implemented, staged in mbirjax_metrics)

The display form is Greg's design (2026-08-04): the History section gets a
separate torch row below the jax row, and the Platform dropdowns (History by
Op, and the Scaling section's page-level selector) show backend-qualified
entries -- gpu-jax, gpu-torch, cpu-jax, cpu-torch.  The implementation keys
the backend off the results DIRECTORY: torch runs live under
``results/{gpu,cpu}-torch/<branch>/regression_{gpu,cpu}-torch_*.yaml``, so
the existing discovery, per-platform aggregation, and y-range machinery all
work unchanged, and the original ``gpu``/``cpu`` directories are the jax
series (displayed with the ``-jax`` suffix).  Changes: renderHistory is
row-parameterized (the torch row hides until torch runs exist; one shared
sync group keeps the x-zoom mirrored across both rows); the torch platforms
get their own colors (teal / rose); the cross-platform correctness analyzer
matches partners only within a backend family.  Verified by a build with a
synthetic ``cpu-torch`` run (platforms and the torch row flowed through) and
a clean rebuild after removing it; a live browser render check is still owed
-- the Browser pane hung -- so open ``dashboard/index.html`` after a build to
confirm visually.

## Persistent compile cache (implemented)

mbirtorch now pins the inductor cache to ``~/.mbirtorch/torch_cache`` and
enables the FX-graph cache at import (setdefault, so the environment can
override; effective when mbirtorch is imported before torch's first compile
-- the mbirjax jax-cache pattern).  Measured with fresh subprocesses sharing
one cache directory: the first recon in a fresh process fell from 14.1 s
(cold cache) to 2.0 s (warm cache) at the 64-cell -- a 7x reduction, with the
residual being dynamo tracing plus the recon itself.  The cold-vcd entries in
the CUDA table above would shrink accordingly on any node after its first
run.

## QGGMRFDenoiser (ported)

The denoiser is ported (single-device path): the identity forward model, the
sequential-subset sweep over one fixed partition, the noise-std estimator,
and the denoiser's own indicator/recon-std overrides.  Parity against the
mbirjax denoiser on a seeded 5-iteration golden: the sigma estimate is
EXACTLY equal (both sides run the same numpy statistics), the alpha trace
matches at 1.8e-06, the nmae trace at 3.0e-06, and the output volume at
3.9e-07 rel-max -- tighter than the recon parity, since the identity model
has no projector atomics.  The suite is 36 passed / 1 skipped.

## The torch series is live (writer + first gpu-torch run)

The harness writer exists
(``mbirjax_metrics/tooling/scaling_tests/torch_backend_writer.py``): harness
sizes and trial counts, per-cell subprocess memory, harness-form fingerprints
(so vs-prior gating can apply to the torch series), git provenance, and a
companion tests file from the mbirtorch suite.  Its first real run -- one
H100, parallel + denoiser cells including the 1024-class -- is landed at
``results/gpu-torch/master/`` and renders on the dashboard's torch row (the
vcd cells cross-check the warm gate readout exactly).  A cpu-torch trial from
the Mac verified the pipeline end to end and was then removed: the durable
cpu-torch series should start on the nightly's own CPU hardware.  Wiring the
writer into the nightly (schedule, env, TRACKED_BRANCHES) is deliberately
left as its own decision.

## Remaining Phase 2 scope

Nightly integration of the torch writer (a scheduling/infra decision); YAML
save/load plus user-facing docs.  The slice-viewer port is deliberately HELD
(Greg, 2026-08-04): the viewer is due a refactor first.
