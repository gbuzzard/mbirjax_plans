# PyTorch port — assessment and phase plan

**Status:** DRAFT, revised per Greg's review (2026-08-04): pinned recon
shapes, the dashboard display item, the Phase 0 repo scaffold; §9 decisions
confirmed.  Exploratory only; no port code exists yet.
**Home for the port:** the `mbirtorch` repo (created 2026-08-04, currently empty),
checked out parallel to `mbirjax`.
**Supporting scripts for this area:** `plans/experiments/torch_port/` (none yet).

## Summary

We are evaluating a port of mbirjax from JAX to PyTorch.  The primary motivation
is interoperability with the deep-learning ecosystem.  The decision is
performance-gated: if the tuned port comes within 2x of jax wall time and about
1.5x of jax peak memory on the standard metrics cells, mbirtorch replaces
mbirjax.  This document records the assessment (§2–§3), the parity gates (§4),
a six-phase incremental plan (§5), the Phase 0 de-risking experiments (§6), and
the semantic differences to pre-register (§7).  The port is more tractable than
the package size suggests, for three structural reasons given in §2.  The main
open performance risk is CPU; for Mac users the MPS backend is the likely
mitigation (§3).

## 1. Motivation and decision rule

The primary motivation is interoperability with the deep-learning ecosystem
(Greg, 2026-08-04).  Learned-prior methods are overwhelmingly written in
PyTorch: plug-and-play, diffusion models, and learned denoisers.  Coupling such
a model to mbirjax today means two frameworks fighting over one GPU.  The XLA
pool reserves most of device memory (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.94`).
Torch brings its own caching allocator.  Tensors cross between the two only by
bridging or copies.  A torch-native engine would share one allocator and pass
tensors with no copies.

A secondary motivation is flexibility.  Torch runs eagerly, so dynamic shapes
need no retrace, data-dependent control flow needs no recompile, and in-place
mutation is native.  Compilation becomes an incremental choice per hot spot,
with eager as a permanent escape hatch.  JAX offers no equivalent escape hatch;
everything must fit the traced functional model.

The decision rule was set by Greg (2026-08-04).  If the tuned port comes within
2x of jax wall time, and within about 1.5x of jax peak memory, on the gate
cells of §4, then mbirtorch replaces mbirjax.  "Tuned" includes the Triton
kernels, compile or CUDA-graph work, and batch-size sweeps, not the eager-only
first version.  If the port misses the gates, we re-evaluate rather than
replace.

One consequence for current plans: the LEAP/SVMBIR wrapper goal
(current_plans §2) moves to the back burner (Greg, 2026-08-04).  A jax-side
wrapper would be throwaway work if replacement proceeds, and LEAP itself
presents as a PyTorch front end, so the wrapper is thinner on mbirtorch.

## 2. What a port touches

The package is about 24k lines, but the jax coupling is concentrated.  The
table below groups the package by layer and coupling.

| Layer | Approx. size | JAX coupling |
|---|---|---|
| VCD engine (`tomography_model.py` core) | ~1.5k of 4.1k lines | High: jitted subset updater, donation, placements |
| Projectors (`projectors.py`, `parallel_beam.py`) | ~1.6k | High: jit/vmap/`lax.map`, scatter semantics, static params |
| `qggmrf.py`, `vcd_utils.py` | ~1.1k | Medium: mostly elementwise math |
| Perf layer (`_pallas_kernels.py`, tile policy, `_sharding/`) | ~1.7k | Total, and deferred to Phases 4–5 |
| Other geometries (cone, translation, multiaxis) | ~3.4k | High, Phase 3 and later |
| `preprocess/`, `parameter_handler`, `viewer`, most of `utilities` | ~8k | Low: largely numpy/scipy/cv2/OSQP |

Three structural facts make the port smaller than the line count suggests.

First, the core has no autodiff dependence.  The forward and back projectors
are hand-written adjoint pairs, and the qGGMRF gradient and Hessian are
closed-form.  The only autodiff call in the package is one
`jax.linear_transpose` inside the `get_transpose` test utility.  A port
therefore translates numerical kernels, not an autodiff graph.  Torch autograd
instead becomes a cheap new feature.  A `torch.autograd.Function` can define
the backward of the forward projector as the existing back projector, and vice
versa.  That differentiable pair is the deliverable DL collaborators actually
want (§5, Phase 1).

Second, the jax API surface is narrow and disciplined.  A frequency census
over the package is dominated by numpy-alike ops with direct torch
equivalents.  The genuinely jax-specific machinery is jit statics,
vmap/`lax.map`, `.at[]` scatter, donation, and sharding.  That machinery is
concentrated in the projector wrappers, the VCD engine, and the perf layer.

Third, the sharding layer is already torch-shaped.  mbirjax deliberately
bypasses GSPMD for the hard collective patterns and uses explicit per-device
dispatch with a banded broadcast/reduce pair (`_sharding/`).  Explicit
per-device code is the native multi-GPU style in torch.  The jax-specific
sharding hazards (reference-cycle leaks, donation, the sharded-axis slicing
trap) have no torch counterpart and simply do not port.

The public API also favors replacement.  `recon()` and `fbp_recon()` take and
return numpy by default, with device forms opt-in.  Most user scripts would
survive a backend swap untouched; the breakage surface is the explicitly
jax-typed touchpoints.

## 3. Expected performance by regime

The 2x gate is not uniformly hard.  This section records where the risk
concentrates, regime by regime.

**GPU, large problems: strong odds.**  The tuned endpoint is a Triton port of
the pallas kernels.  The pallas kernels already target the Triton backend, so
the kernel port is a lateral move, and we expect near-parity at the kernel
level.  That expectation is to be verified, not assumed; the pallas lowering
and hand-written Triton can differ.  Between kernels, jax wins today through
jit fusion of the per-subset update.  The subset update has fixed shapes per
partition granularity and no host syncs in the line search, which are exactly
the properties CUDA graph capture requires.  The glue cost is therefore
addressable, and this regime should land well inside the gate.

**Interactive sizes: torch could beat jax.**  Our own attribution showed VCD
at interactive sizes (~200-class) to be roughly 95% host dispatch (the 2026-07
performance-goals revision of current_plans).  A CUDA-graph replay per subset
makes the host cost near zero.  The port could therefore be faster than jax
exactly where users feel latency.

**CPU: the one genuinely open cell.**  XLA fuses the fan-kernel chains on CPU,
while eager torch materializes each intermediate.  Torch scatter paths have
also historically had serial CPU implementations.  Two bounded mitigations
exist:
Inductor CPU compilation (C++/OpenMP codegen) of the fan chains, and a small
`cpp_extension` custom op for the two fan kernels if Inductor misses.  The
Phase 0 spikes measure CPU shapes for exactly this reason.

**Apple MPS: the likely answer for Mac users.**  Most local users run Macs,
and torch's MPS backend should serve them better than jax does today (Greg,
2026-08-04).  jax on Metal is experimental, so Mac users currently run jax on
CPU.  The user-relevant comparison is therefore torch-MPS versus jax-CPU on
the same machine, and no 2x-rule baseline exists for it.  Three MPS
constraints carry into the design: the backend is float32-only (acceptable;
the engine is f32 by design); compile support is immature, so plan for eager;
and op-coverage gaps fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK`.  A
Phase 0 spike cell on an M-series Mac calibrates the scatter-heavy kernels
there.

**Memory: achievable with two named attention points.**  In-place updates and
refcount freeing replace the donation and `.delete()` machinery, which makes
transients more controllable, not less.  Attention point one is eager chain
width.  An eager op chain holds two to three live buffers where XLA fused to
one or two.  The qGGMRF neighbor-shift chain over volume-sized arrays is the
one worth fusing early; it is elementwise, which is Inductor's sweet spot.
Attention point two is the ruler.  Compare jax `peak_bytes_in_use` against torch
`max_memory_allocated` (allocated, not reserved), per device, with the same
isolated-subprocess discipline the harness already uses.  Int64 index arrays
double the scatter-centers array; at volume scale this is small, but it
belongs in the ledger.

## 4. Parity gates

The gate cells are the mbirjax_metrics scaling cells (Greg, 2026-08-04), so
the comparison instrument is the harness and dashboard we already trust.  The
gate has seven parts:

- **Cells.**  The standard sinogram sizes from
  `mbirjax_metrics/tooling/scaling_tests/performance_tracking.py`:
  CPU (128,112,96), (129,113,97), (200,208,160);
  GPU (200,208,160), (512,448,384), (513,449,385), plus (1024,1008,992) as the
  single-trial capacity probe.  Recon shapes are the harness's PINNED shapes,
  not library defaults (Greg, 2026-08-04).  Pinning decouples every cell from
  either library's auto-geometry policy.  The cone pins in
  `performance_tracking.py` are the existing example; the parallel cells get
  pinned values recorded the same way at Phase 0.
- **Ops.**  The harness op set: direct_filter, forward, back, and
  vcd_nonconst.  The denoiser cells join when the denoiser ports (Phase 2).
- **Device counts.**  n=1 through Phase 3; n=1, 2, 4 from Phase 4.
- **Thresholds.**  Wall time within 2x of jax per gated cell, and peak device
  memory within about 1.5x.
- **Gated platforms.**  CPU and CUDA, the platforms with jax baselines.  MPS
  is reported alongside against jax-CPU on the same machine, ungated.
- **Frozen baseline.**  Record the jax/jaxlib version at Phase 0 start and
  hold every gate comparison to baselines from that version.  jax keeps
  moving; the pyproject version exclusions record past regressions.  A
  drifting baseline makes the gate unmeasurable.
- **Correctness tolerances.**  The harness gates same-framework fingerprints
  at relative 1e-5 single-shot and 1e-4 iterated.  Cross-framework comparisons
  add libm and summation-order differences.  Start at 1e-4 single-op and 1e-3
  iterated, calibrate on measured floors, and keep the scale-invariant
  rel-max form from lessons §2.

The harness itself needs two extensions (Phase 2).  The first is a backend
column, so torch rows land in the same results files as the jax baselines.
The second is a dashboard display design: torch series must be readable
without making the whole dashboard too busy (Greg, 2026-08-04).  The form (a
backend selector, an overlay mode, or a dedicated comparison view) will be
decided with a small mockup before wiring.

## 5. Phase plan

Estimates assume the current working style (Claude-drafted, Greg-reviewed) and
are FTE-equivalent.  Calendar time roughly doubles for a background effort.

**Phase 0 — de-risking spikes plus repo scaffold (3–5 days).**  Standalone
scripts answering the top performance unknowns before any package code;
specifications in §6.  Phase 0 also scaffolds the mbirtorch repo (Greg,
2026-08-04): pyproject with the recorded torch pin, .gitignore, dev_scripts,
the environment file, and an empty package plus test skeleton.  The
deliverables are the scaffold and a findings page with a numbers table
against the frozen jax baselines.  The exit question: does the 2x gate look
safe enough to fund Phase 1?

**Phase 1 — parallel-beam vertical slice (~1 week to a converging prototype;
3–5 weeks to gated).**  The scope is the smallest end-to-end recon: geometry
math (`compute_proj_data`); the horizontal fan forward and back in the
portable forms only (per-tap scatter-add and per-tap gather; no sorted reduce,
no stacked gather); qGGMRF gradient and Hessian; the VCD engine with numpy
partitions, the subset updater, the line search, and positivity; FBP via
torch.fft; the parameter handler nearly verbatim; phantom and weights
utilities as needed.  Promoted into Phase 1 for the interop motivation: the
differentiable projector pair as a `torch.autograd.Function`, plus a thin
`nn.Module` adapter in the LEAP style.  Deferred: sharding (single device
only), Triton, the tile policy (one fixed batching scheme), other geometries,
prox/denoiser, and preprocess.  Phase gates: adjointness
(&lt;Ax, y&gt; = &lt;x, A'y&gt;); cross-framework goldens against jax at the §4
tolerances; convergence parity on shepp-logan (per-iteration alpha and RMSE
traces at loose tolerance, final NRMSE tight); smoke runs on CPU, CUDA, and
MPS.

**Phase 2 — performance pass on parallel (2–4 weeks).**  Apply what Phase 0
chose: torch.compile and/or CUDA graphs on the subset update, batch-knob
sweeps, and qGGMRF fusion for the memory point.  Add the harness backend
column and produce the first full gate-cell readout at n=1.  The denoiser
ports here (the VCD loop with identity projectors) and adds its gate rows.

**Phase 3 — cone beam (2–3 weeks).**  The banded vertical fan and the cone
kernels, using the harness cone recon-shape pins.  This phase validates that
the base-class kernel contract survived the port.

**Phase 4 — multi-device (2–4 weeks, after a design spike).**  The
`Placement` and banded broadcast/reduce design transfers conceptually.  The
implementation choice needs a spike: single-process threads, where GIL
contention on eager dispatch is the analog of the jax host-dispatch findings,
versus one process per GPU with NCCL for the two banded collectives.

**Phase 5 — Triton kernels and sweeps (2–4 weeks).**  Port the pallas kernels
to Triton and re-run the measured-constant sweeps (tile bands, batch sizes) on
the cluster.  Produce the final gate readout at n=1, 2, 4, which is the
replacement decision input.

**Beyond the gates.**  The gate decision is made on parallel and cone;
replacement requires the rest of the package.  The tail is: translation and
multiaxis geometries (+2–4 weeks), preprocess (mostly numpy, small), and
docs, demos, and nightly-soak migration (+2–3 weeks).  Full replacement
readiness is therefore roughly 1.5–2.5 months beyond the gate decision, and
the go/no-go information itself is front-loaded in the first month
(Phases 0–1).

## 6. Phase 0 spike specifications

Conventions for all spikes: scripts live in `plans/experiments/torch_port/`;
run parameters sit at the top of each script, not on the command line; each
measured config runs in an isolated subprocess for honest memory; every
results file records the toolchain (torch version, device, driver).  Hardware:
Greg's M-series Mac for CPU and MPS cells; gautschi H100 for CUDA cells
(standing cluster authorization applies once this plan is approved;
coordinate before heavy use).

**Spike 1 — fan kernels.**  Question: what do the eager torch fan kernels
cost, relative to jax, at production shapes?  Method: implement
`horizontal_fan_project` and `horizontal_fan_back` in eager torch; run the
(200,208,160) and (512,448,384) cell shapes with realistic pixel batches on
CPU, CUDA, and MPS; A/B against the jax XLA path and the jax tuned path.
This spike also pins down torch's CPU scatter parallelism, the §3 CPU risk.

**Spike 2 — chain fusion.**  Question: what do torch.compile and fusion buy
on the weight chain and on the qGGMRF chain, in both time and peak memory?
Method: eager versus torch.compile on the two chains at volume scale,
recording wall time and `max_memory_allocated`.

**Spike 3 — subset-update host overhead.**  Question: can a VCD-subset-shaped
update loop run with near-zero host cost under CUDA graph capture?  Method: a
mock subset updater with the real per-granularity shapes, built from the
spike-1 kernels; eager versus graph replay at the 200-class and 512-class
sizes; measure wall time and host occupancy.

The Phase 0 deliverable is `plans/torch_port/phase0_findings.md`: the numbers
table for all three spikes against the frozen jax baselines, plus a go/no-go
recommendation against the §4 gates.

## 7. Semantic differences to pre-register

These are the places a transliteration breaks silently.  Each becomes a
checklist item for review, and the value-critical ones get targeted tests.

1. **Out-of-bounds scatter and gather.**  jax drops out-of-bounds scatter
   indices and clamps out-of-bounds gathers, and the per-tap fan loops rely on
   both.  torch `index_add_` device-asserts on out-of-bounds.  Rule: every
   ported fan loop adopts the clip-plus-zero-weight pattern already present in
   the sorted branch.
2. **Rounding ties.**  `torch.round` and `jnp.round` are both half-to-even,
   but the upstream cos/sin libms differ by ULPs, so a tie-adjacent scatter
   center can flip between frameworks at symmetric geometries.  The
   center-slice study showed exactly this class of deterministic tipping.
   Rule: cross-framework golden configs avoid tie-prone symmetric setups, or
   compare center arrays with a tie-tolerant comparator.
3. **Autograd overhead.**  The engine does not use autograd, so it runs under
   `torch.inference_mode()`.  The differentiable projector wrapper is the one
   explicit grad-enabled path.
4. **Dtype promotion.**  torch keeps the tensor dtype against python scalars,
   so the f64-scalar-promotes-volume trap disappears.  The few deliberate
   float64 sites need a survival check.
5. **Index dtype.**  torch index ops want int64.  The scatter-centers array
   doubles (about 0.5 to 1 GB at the 1024-class full grid), and the existing
   chunking-threshold pattern carries over.
6. **Determinism diagnostics.**  `torch.use_deterministic_algorithms(True)`
   replaces the `--xla_gpu_deterministic_ops` discriminator for atomics noise.
   The lessons §2 tolerance methodology transfers unchanged.
7. **Allocator hygiene.**  The caching allocator fragments under shape churn.
   The fixed-shape-per-granularity discipline (padding, fixed batches) remains
   load-bearing, and `expandable_segments` is the fallback knob.
8. **RNG.**  Partitions draw from numpy RNG and stay numpy.  The seeding rules
   for cross-config comparisons transfer unchanged.

## 8. Interaction with current release goals

The port coexists with the jax release goals until the gate decision.  The
sharpness-schedule work (§1) and MAR H-caching (§3) continue on mbirjax.  The
LEAP/SVMBIR goal (§2) is back-burnered per §1 above.  A feature freeze on
mbirjax starts only if the replacement decision triggers; until then the
policy is bugfixes-mirrored, features-unfrozen.  The metrics harness gains its
backend column in Phase 2, and the jax baselines freeze at the version
recorded in Phase 0.

## 9. Decisions (confirmed by Greg, 2026-08-04)

- **torch version pin.**  Record the current stable torch at Phase 0 start,
  and adopt the pyproject pinning style (floor plus known-bad exclusions).
- **API stance.**  mbirtorch keeps the mbirjax public API shape, numpy at the
  boundary, so replacement is near-drop-in.
- **prox_map scope.**  Out of Phase 1, into Phase 2; it shares the engine and
  ports cheaply.
- **Golden-data mechanism.**  A jax-side script writes golden HDF5 per gate
  cell, and mbirtorch tests read them, so the mbirtorch test env never
  imports jax.
- **MPS gating status.**  Informational only, reported against jax-CPU on the
  same machine.
