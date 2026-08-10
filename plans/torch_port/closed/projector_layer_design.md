# Projector-layer design — mbirtorch

**Status:** IMPLEMENTED 2026-08-06 (Greg-approved; suite 260 passed with
zero test-file edits -- the acceptance the adversarial review predicted).
Panel-reviewed by four reviewers: mbirjax alignment, Triton consumer,
new-geometry walkthrough, consumer-seam adversarial.  Two small design
elements are recorded follow-ups rather than implemented: the named
`_cone_vertical_affine` extraction and the `plan` body argument (both
Phase-5 conveniences with no behavior today; the hook shape shipped as a
single `_view_batch_bodies()` returning the (forward, back) pair).
**Motivation:** Greg's review found the
current division of labor unclear — projectors.py mixes driver with
kernels, parallel and cone expose their geometry at different seams, and
cone bypasses the geometry hook it cannot satisfy.

## The division of labor

Three layers replace the current two-and-a-half.

The DRIVER (projectors.py) owns iteration and memory only: the view-batch
loop for each direction, the transient budget arithmetic, per-device
compiled-instance management, and the compile lock.  It knows nothing of
any geometry: it slices `view_params_batch` rows from the model's
view-parameter array, calls the geometry's body, and assembles outputs it
sizes lazily from the first block.  The loop is written as a two-axis
tile walk with the pixel axis at one chunk today, and the forward
assembles by accumulation, so mbirjax's pixel tiling drops in later
without a rewrite and without silent-wrongness (the current empty-plus-
assign forward would corrupt under a pixel loop).

The GEOMETRY BODIES are the single seam.  Every geometry model supplies
two per-view-batch bodies through hooks `_forward_view_batch` and
`_back_view_batch`.  A body owns its geometry math, its tap loops, its
layout, and its output orientation; it returns sinogram-layout blocks.
Bodies are MODULE-LEVEL PURE FUNCTIONS taking parameter VALUES, paired
with a thin eager unpacker on the model that builds the argument dict
outside the traced region.  Never bound methods (they pin models in the
module-level compile cache and collide its error keys), never
`self.get_params` inside the graph (dynamo would guard on the parameter
machinery and silently fall back to eager past its cache limit).  The
compiled unit is the BODY — this fuses parallel's hfan chain with its
fan kernels, removing today's materialized tuple boundary AND a latent
defect: `_parallel_hfan_math` is currently compiled as one shared
instance executed concurrently by every device thread.

The SHARED KERNELS (new module `horizontal_fan.py`) hold the math several
geometries reuse but none owns: `tap_weights`, `fan_forward_batch`,
`fan_back_batch`, and the documented hfan contract.  Two generalizations
make them serve every fan geometry instead of parallel alone: `values`
takes a broadcastable leading view axis (`shape[-1]`, not `shape[1]`),
and the back kernel's view reduction becomes an explicit flag — cone
currently duplicates both tap loops only because the shared forms bake
the reduction in.  The module is named for the horizontal fan because
this codebase names two fans; cone's vertical affine becomes a named
pure function `_cone_vertical_affine(...) -> (m0, W_p_r)` used by both
cone bodies (today the same affine is derived twice in two algebraic
forms), and it is the sanctioned bridge a fused kernel consumes.

## The hfan contract

The tuple is `(n_p, centers, W_p_c, weight_scale)` per (view, pixel),
with `centers` int32 and `L_max` dropped (recomputed at the use site,
one instruction).  This trims cone from 28 to 16 and parallel from 12 to
8 bytes per (view, pixel) — the compression mbirjax's kernel campaign
flagged as its known unused lever — and matters because small-cell
device memory is where torch's cone back already trails jax by 4-6x.
The tap axis is NEVER materialized; precompute stays per-(view, pixel).
Values are unchanged: int32 indices are exact, and the recomputed
`min(1, W_p_c)` is the same operation.

Precompute-only is confirmed by measurement, not assumption.  mbirjax's
`compute_hfan_data` exists solely for its pallas kernels; its in-kernel
recompute path survives only because XLA fuses it for free.  At the gate
cells the precompute costs 0.2-1 percent of the tuned back-kernel
budget, and for cone the inline alternative LOSES: the horizontal chain
is transcendental-heavy (two atan2 per (view, pixel)) while the vertical
chain is a pure affine — so kernels consume precomputed horizontal data
and re-express only the vertical affine inline, and the geometry stays
single-sourced in torch.

## Contracts the driver holds

The batching rule is implementation-supplied, not driver-owned.  Today's
rule models the eager gather transient, which is geometry-specific
(parallel: the runtime band length; cone: `max(num_slices, num_rows)`
from params) and which fused kernels eliminate entirely — a register-
tile kernel wants view chunks an order of magnitude larger.  The
geometry (later, the kernel implementation) supplies bytes-per-
(view, pixel) and preferred chunk; the driver does the division.
ACCEPTANCE: the effective view batch is bit-unchanged for both current
geometries at the golden cells before anything else moves, because
vb changes move float summation order and calibrated peak memory.

The body signature carries the union of banding arguments
(`slice_start`, `band_slices`, `coeff_power`, `dev_index`) plus an
optional opaque `plan` argument, unused today, for the sorted/CSR
variant's per-(subset, view-range) memoization — the pixel subsets are
fixed per recon, so the plan cache hits ~100 percent after the first
pass.  A row-aligned geometry's body asserts `slice_start == 0` rather
than silently ignoring it.  The view-range form is the one loop; the
full-range public form is an adapter that coerces inputs, pins
`dev_index=0`, and asserts parallel's rows==slices invariant.  The
banded drivers' one assumption — one contiguous real-view span per
owner — is recorded at the adapter.

Row alignment becomes a model property (`rows_track_slices` or similar),
REQUIRED with no getattr default: the current `getattr(..., True)`
default silently takes the aligned branch for any geometry that forgets
to declare it, and every geometry still to port wants False.  The
property unifies with `_sino_row_padding`, which states the same
geometric fact in a second vocabulary.  The banded capability guard
re-points from the driver method to the model hook so its clean
NotImplementedError survives.

## Rules carried from the review

- `create_projectors` overrides survive: the denoiser's `pass` is
  load-bearing (its view_params_name is the string 'None').
- `get_psf_radius` becomes pure FIRST: it currently assigns
  `bp_psf_radius` as a side effect that only keyword-argument evaluation
  order keeps correct.  The direction asymmetry (forward uses
  bp_psf_radius, back uses psf_radius) and cone's coeff_power order
  (squared AFTER the 1/cos divisor) go in the body-contract docstring.
- Per-device compiled instances build EAGERLY in create_projectors and
  cache per (function, device index) so projector rebuilds stop
  re-tracing devices 1..n-1; cone's damping profile joins the eager
  build (its lazy build currently races worker threads on first use).
- View parameters are read from the model at every rebuild and
  PRE-PLACED per device through move_shard — today's `.to(dev)` per
  band-pass bypasses the dev2dev-safe policy and re-copies.
- The compile lock is exported for Phase 5 (triton.jit compiles outside
  torch.compile but races the same way), together with an availability
  probe modeled on mbirjax's (`(usable, reason)`, arch allowlist, env
  kill switch) — required for the AMD/ROCm goal.
- Bodies stay functional (return partials, no caller-buffer mutation):
  maybe_compile's eager retry re-runs the first call, and an
  accumulate-into-buffer body would double-count.
- Cone's plain back accumulate becomes in-place, matching the banded
  form (one fewer (P, S) allocation per view batch).

## The unported geometries, walked through

Only two remain: translation and multiaxis parallel.  Both are two-fan
compositions; both declare rows_track_slices False; both carry per-view
data as fixed-width rows (translation (V, 3) vectors, multiaxis (V, 2)
azimuth/elevation) that the driver slices opaquely — translation's
non-angle parameter needs no special-casing anywhere.  Translation's
bodies are cone's with the horizontal chain swapped (~15 lines) and
`z_offset = -t_z`; its direct filter is cone's minus the helical
z-weight.  Multiaxis's horizontal fan is parallel's VERBATIM; its
vertical fans are new — the forward must be a slice-side scatter because
its slope legitimately reaches zero at top-down elevation, so cone's
detector-side gather has no inverse there.  Two recorded prerequisites:
translation's production panels (~1900 x 3000) exceed the transient
budget even at one view per batch, making pixel-axis chunking a
REQUIREMENT for that port; multiaxis needs the scatter primitive
validated under torch.compile.

Detector growth upgrades that requirement (Greg, 2026-08-06): the
current estimate is 6K x 10K views.  At that size ONE view's gather
transient against a 512-class pixel set is ~6 GB, so no view batch can
bound memory and the pixel axis must chunk.  Pixel-axis chunking is
therefore first-class driver structure, not a translation-port detail --
the two-axis tile walk and the accumulating forward exist so it lands
without touching the geometry contract.

## The n>=3 divergence finding (GPU verification, 2026-08-06)

The verification matrix read torch n4-vs-n1 at 2.16e-03 at 512 (prior
runs: 3.03e-04) with n=2 and every jax row exact.  A five-probe bisect
attributed it fully, and it is NOT a correctness bug.  The evidence
chain: projections at n=4 on CUDA sit at their established floors; the
divergence is deterministic in-process (repeats, a clone toggle, and
CUDA_LAUNCH_BLOCKING all read identically, ruling out races and buffer
aliasing); eager n4 vs eager n1 -- bit-identical math on every device --
reads 5.16e-04, the old floor, so the banding logic is clean; and eager
n1 vs compiled n1 alone reads 1.07e-02, calibrating how strongly the VCD
feedback loop amplifies ANY kernel-level float difference at this cell.

The decomposition: the restructure's large fused bodies give
torch.compile room to generate DIFFERENT float realizations per compiled
instance and per input shape, where the old few-line fan kernels
compiled identically everywhere.  Per-device instances contributed
7.49e-03; sharing instances removes that (2.76e-03); the remainder is
per-shape specialization (n=4's band shapes versus n=1's full shapes)
and is unreachable by instance sharing.  Every realization is a
legitimate float trajectory to the same MAP fixed point.

The policy this sets: n>1 VALUE gates compare eager-to-eager (bounding
logic at the true float floor, ~5e-04 here), while compiled n>1
trajectories are accepted within the amplified compile-latitude
envelope, now documented.  Reducing the envelope by sharing compiled
instances (a 2.7x reduction; the shared runs executed concurrently
without incident, but the per-device split guarded launcher state) is a
recorded option gated on a thread soak.

## Exit gates

Value gates unchanged and untouched: the goldens, adjointness, cone
suites, and the sharded parity tests all pass with zero test-file edits
(the review verified no test touches a projector internal — which also
means values are the only net under this seam, so the migration runs
suite-green per chunk).  The effective view batch is verified
bit-unchanged per geometry at the golden cells before the driver
unifies.  Peak memory is re-measured at the gate cells, not assumed.
GPU verification: the full suite passes on a gautschi GPU node (258
passed, 3 skipped -- the two extras are MPS-gated), and the matrix's jax
rows and torch n=2 reproduce exactly; the n>=3 reading is the compile-
latitude finding above.
