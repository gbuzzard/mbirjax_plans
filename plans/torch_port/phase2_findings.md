# Phase 2 findings — compile integration and the first gate-cell readout

**Status:** local (CPU + MPS) cells measured 2026-08-04; the CUDA cells run in
gautschi job 14833737 and are recorded in their own section.
**Plan:** `port_plan.md` section 5 (Phase 2) and section 4 (the gates).
**Scripts:** `plans/experiments/torch_port/p2_gate_readout.py` and
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

## CUDA cells (gautschi, one H100)

Job 14833737 (the resubmission with the view-batch cap; the first attempt,
14833387, predated the fix) measured both frameworks on the GPU cells.  The
jax side is the TUNED pallas path.  Ratios are torch/jax; memory is the
device allocator's peak (torch max_memory_allocated vs jax
peak_bytes_in_use).  The torch VCD at the 512-class and 1024 cells had no
warm rerun in this readout, so those entries INCLUDE the one-time inductor
compile (a readout gap to close; the 200-cell warm entry shows the shape of
the correction).

| cell | op | jax (s) | torch (s) | time ratio | mem ratio |
|---|---|---|---|---|---|
| 200x208x160 | direct_filter | 0.003 | 0.003 | 0.83 | 0.54 |
| 200x208x160 | forward | 0.007 | 0.006 | 0.90 | 0.35 |
| 200x208x160 | back | 0.005 | 0.008 | 1.79 | 12.2 |
| 200x208x160 | vcd (warm) | 16.46 | 0.271 | **0.02** | 11.4 |
| 512x448x384 | direct_filter | 0.012 | 0.014 | 1.16 | 0.62 |
| 512x448x384 | forward | 0.103 | 0.184 | 1.79 | 0.38 |
| 512x448x384 | back | 0.040 | 0.310 | **7.7** | **3.1** |
| 512x448x384 | vcd (cold+compile) | 14.64 | 22.35 | 1.53 | 2.14 |
| 513x449x385 | direct_filter | 0.019 | 0.015 | 0.82 | 0.62 |
| 513x449x385 | forward | 0.108 | 0.356 | **3.3** | 0.38 |
| 513x449x385 | back | 0.042 | 0.333 | **8.0** | **3.1** |
| 513x449x385 | vcd (cold+compile) | 22.54 | 29.37 | 1.30 | 2.04 |
| 1024x1008x992 | direct_filter | 0.120 | 0.112 | 0.93 | 0.80 |
| 1024x1008x992 | forward | 2.74 | 5.23 | 1.91 | 0.50 |
| 1024x1008x992 | back | 1.16 | 5.10 | **4.4** | 0.65 |
| 1024x1008x992 | vcd (cold+compile) | 39.01 | 114.4 | 2.93 | 1.00 |

Reading.  The interactive regime inverts hard in torch's favor: warm VCD at
the 200 cell is 0.27 s against jax's 16.5 s, the Phase 0 host-dispatch
prediction at its most extreme.  The filter passes everywhere, the compiled
forward sits at 0.9-1.9x on the even cells, and the 1024 memory column favors
torch on every op (the compiled forward runs at HALF jax's peak).  Three
breaches remain, all with named causes: the BACK projector at large cells
(4.4-8x) is the pallas-kernel gap the plan assigns to the Phase 5 Triton
port; the 513 forward (3.3x, against 1.79x at 512) is an odd-size inductor
specialization effect to investigate; and the back/vcd MEMORY ratios at the
200/512 cells trace to the 2 GiB gather budget being generous where jax
streams smaller -- a knob to scale with cell size, not a structural problem
(the 1024 cell, where memory matters most, already favors torch).  The
cold-vcd entries would also shrink under the warm protocol the harness uses.

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

## Remaining Phase 2 scope

In likely order: the harness writer for torch runs (the measurement side that
fills ``results/*-torch/``); the persistent torch compile cache (the cold-vcd
entries above are the motivation); a warm-vcd rerun protocol for the large
CUDA cells; the gather-budget scaling noted above; the denoiser port; and
YAML save/load plus user-facing docs.
