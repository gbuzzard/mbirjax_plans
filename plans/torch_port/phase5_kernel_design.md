# Phase 5 kernel design — Triton projector bodies for mbirtorch

**Status:** APPROVED AND IN EXECUTION (2026-08-07).  K1 (cone back
kernel) and K3 (cone forward kernel) are implemented and green on H100;
K2's sweep and composed gate PASSED and the back kernel is DEFAULT-ON
with constants (16, 64, 4, 1) pinned; K4 (forward sweep + five-arm gate)
is in flight.  The increment ledger and measurements live in
phase5_findings.md.  **Prior art:** the mbirjax pallas campaign
(`plans/projector_kernels/`: gpu_headroom_summary/findings, the E3/E4
records, the A100 tuning findings) — the designs transfer, the numbers
do not.  **Delegation:** Opus implements each increment against this
design; Fable reviews adversarially and holds the GPU gates.

## Goal and targets

Close the known single-device kernel gaps with hand-written Triton
kernels, in measured order: cone BACK projection first (3.4–6.2x behind
jax's tuned pallas kernels — the top target from the Phase 3 readout),
then cone forward, then the parallel pair.  The eager/compiled torch
bodies stay compiled-in everywhere as the permanent value reference and
fallback, exactly as mbirjax keeps its XLA path.

## Integration: kernels are alternative bodies

A kernel arrives as a module-level function with the SAME signature as
the torch body it replaces, selected at `create_projectors` time by the
geometry's `_view_batch_bodies` hook when `triton_available()` says so
(the probe and the `MBIRTORCH_DISABLE_TRITON` kill switch shipped in the
Phase 5 prep).  Four function swaps, no driver changes: the driver's
view-range loop, lazy assembly, banded seams (`slice_start`,
`band_slices`, `dev_index`), and `plan` slot all pass through unchanged.
The batching rule is implementation-supplied through `_transient_cols`
generalized to the kernel's cost model: a register-tile kernel holds no
gather transient, wants view chunks near 128 (the pallas constant; swept
per arch), and declares its own resident bytes per (view, pixel).

## The cone back kernel (the flagship)

The structure is the pallas back design re-expressed in Triton, with
the two mechanisms that carried its 9.07x: a register accumulator
across the view reduction (the view sum never touches memory), and a
grid ordered so concurrent programs gather from the same sinogram slice
(L2 residency for the transaction-bound gathers).

- Grid: (pixel-block, slice-chunk), slice chunk near 128 (pallas
  CONE_LC; swept).  Each program owns an (P_blk, L_chunk) accumulator.
- Loop nest per program: views in the chunk, then channel taps
  (psf_radius), then row taps (psf_radius on the back path).  All tap
  loops are compile-time constant trip counts.
- Inputs: the trimmed hfan contract (n_p f32, centers i32, W_p_c,
  weight_scale — per (view, pixel), precomputed by the existing torch
  chain), the vertical affine pair (m0, W_p_r) from
  `_cone_vertical_affine` (the sanctioned bridge; recomputed as (Vb, P)
  tensors alongside the hfan data), the local sinogram, and the
  geometry scalars.  Channel tap weights are derived IN-KERNEL from the
  contract (the trapezoid rule of horizontal_fan.py) — the tap axis is
  never materialized, which is the compression the pallas campaign
  flagged but never needed (their streams cost 24 B per (view, pixel);
  this contract costs 8–16).
- The vertical map is inline arithmetic: m = m0 + W_p_r * l per slice,
  row taps around round(m), the 1/cos_phi divisor, and coeff_power
  applied AFTER the divisor (the mbirjax rule, stated in the torch body
  and preserved bit-for-bit in intent).
- Output: the (P, L) partial the body contract requires — functional,
  freshly written once per program.  Cross-chunk accumulation stays in
  the driver's lazy `add_`, the torch analog of pallas's donated
  accumulate.

## The other three kernels

Cone forward reuses the same affine inline in its inverse form
(k = (m - m0) / W_p_r, the detector-side gather the torch body already
implements) and the horizontal scatter becomes the two-phase
sorted-stream walk ONLY if the plain per-tap atomic form measures short
of the bar — the pallas evidence says atomics are the forward's
limiter, but Triton's atomics and scheduling differ enough that the
plain form is measured first (measure, then specialize).  The parallel
pair is the degenerate case: no vertical fan, rows ride as the vector
axis, and the pallas parallel numbers (2.13x subset batches for the
sorted forward, 9x-class back) set expectations.  The sorted/CSR
forward variant, when it comes, builds its streams from the same hfan
contract (the zero-weight/clamp-index convention makes the stream
bound static) and caches them through the `plan` argument per
(pixel subset x view chunk) — fixed per recon, amortized across the
fine tail exactly as in mbirjax (their 16–26x regime).

## Rules carried from the campaigns

- STATIC SHAPES: no data-dependent launch geometry (the pallas
  increment-2 0.68x episode).  Pixel blocks pad to the block size with
  zero-weight taps; the row/band axis pads only to the chunk multiple
  (the row-chunked grid removes the power-of-two blowup).
- HOISTED BUILDERS: every per-call precompute (hfan, affine, plans) is
  built once per (subset, view-chunk) outside the kernel loop — a
  per-chunk rebuild is the bench artifact that hid 3.54x-vs-9.07x.
- PINNED CONFIGS, NOT RUNTIME AUTOTUNE: the compile-latitude finding
  showed per-instance variant scatter amplifies through the VCD loop.
  Kernels ship with constants pinned per (architecture, shape class)
  from explicit sweeps; `@triton.autotune` is a sweep-time tool only.
  Compilation runs under `compile_serialized()` (the exported lock).
- PER-ARCH VALIDATION: constants MAY transfer across architectures but
  are verified, never assumed.  The A100 campaign found the H100-tuned
  pallas constants already optimal on A100 -- and the same sweeps
  caught a single-arch win that reversed +28 percent on H100, which is
  the hazard the verification exists for (the SXM4-vs-PCIe form-factor
  spread is the same lesson within one arch).  Performance NUMBERS are
  per-arch regardless.  Protocol (Greg, 2026-08-06): kernels default ON
  wherever the probe passes, on ANY architecture, with the H100-pinned
  constants -- the structural wins are architecture-generic and an
  untuned kernel beating a 3.4-6.2x baseline gap is the likely case, so
  an arch allowlist is overly conservative.  The guard moves to the
  correct axis: a FIRST-USE VALUE SELF-CHECK at create_projectors time
  runs one kernel-vs-torch-body comparison at a small shape on the
  actual device (milliseconds) and falls back only on a tolerance
  breach -- the is_dev2dev_safe philosophy (probe the hardware you are
  on, never trust a vendor list), and it catches a broken toolchain
  even on swept architectures.  Per-arch spot-validation of the
  constants remains a TUNING activity, not a gate.  ROCm rides the same
  mechanism (Triton backend, probe plus self-check).
- ADVERSARIAL MEMORY HONESTY: peaks re-measured, never inferred; the
  poison-the-padding test class carries over.

## Gates (per increment, before the next starts)

Value: kernel-vs-torch-body equality at rel 1e-5 (gradient path) and
1e-4 (Hessian, coeff_power=2) per the pallas precedent, including its
documented rounding carve-out (cos(atan2) vs sqrt(1+t^2) forms); the
explicit adjointness test; the goldens unchanged; n>1 value gates run
eager-to-eager per the compile-latitude policy.  Performance: warm
model-level A/B at the 512 and 1024 cells on gautschi under the
replacement rule (torch within 2x of jax time, ~1.5x memory), with the
cone back target being to CLOSE its 3.4–6.2x gap, not merely narrow
it; the n4@1024 +16 percent timing attribution rides this re-baseline.
Composition is measured, not extrapolated: the isolated kernel bench
flatters baselines (the pallas E4 lesson), so the gate is the composed
recon call.

## Increments for Opus

K1: the cone back kernel + its parity/adjoint harness (CPU-skippable,
CUDA-gated), behind the probe, values first.  K2: the H100 sweep
(pixel block, slice chunk, view chunk, warps) + the gautschi composed
gate; pin constants.  K3: cone forward (plain atomic form first).
K4: the parallel pair.  K5 (optional, measured need): sorted streams +
plan caching.  Each increment ends suite-green with the kernel OFF by
default until its gate passes; the switch-on is a Fable checkpoint.
