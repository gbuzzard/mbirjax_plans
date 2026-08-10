# Phase 3 findings — the cone-beam port

**Status:** COMPLETE 2026-08-05.  The geometry is parity-validated on every
tested path, including helical and curved-detector goldens; the demo carries
a cone variant; and the cone gate-cell readout ran on one H100 against tuned
jax.  The CUDA performance gaps it measured are Phase 5 scope and are
recorded in its target list.
**Plan:** `port_plan.md` section 5 (Phase 3).  **Code:** `mbirtorch/cone_beam.py`
plus the cone sections of the goldens generator and `tests/test_cone.py`.

## Summary

The cone-beam port is complete as a converging, parity-validated geometry:
the magnification-dependent horizontal fan (flat and curved detectors), the
detector-side forward vertical fan and its adjoint banded gather, the
auto-recon geometry with per-end axial padding, FDK with the cosine
pre-weight, the helical z-weight, and the DC-damping preconditioner.  Every
single-op golden matches mbirjax at ~1e-6 rel-max, and the seeded five-
iteration convergence parity holds at ~1e-5 -- but only after a port lesson
worth recording: the first parity run DIVERGED (final recon 9.5e-2) with
every projector op matching, because mbirjax cone ships the "C4" DC-damping
preconditioner ON by default through the `_get_update_direction` seam.  The
divergence signature -- per-op values at float noise, trajectories apart --
is exactly what a missing preconditioner looks like, and the seam the port
had preserved for architectural parity turned out to be load-bearing on the
first geometry that overrides it.

## What was ported

`ConeBeamModel` with its own drivers (`ConeProjectors`): cone projection is
two separable fans, and the drivers batch over views exactly like the
parallel drivers, with the (view_batch, P, S) and (view_batch, P, R)
transients governed by the same scaled budget.  The forward vertical fan is
formulated from the DETECTOR side (for each detector row, which voxels
project onto it), matching the back projector by construction; the elementwise
torch form of mbirjax's affine-from-first-row row map is algebraically
identical (float differences at the ULP level, absorbed by the golden
floors).  Ported alongside: `get_support_radius` (into vcd_utils), the FDK
cosine pre-weight path in the shared row filter, `detector_mn_to_uv`, the
axial-padding auto geometry (verbatim math, including the per-end excess
formula), the >45-degree cone-angle warning, and the DC-damping profile with
its cache.  Not ported: the multi-device banding seams and the pallas
kernels (the jax perf layer).

## Gate results (local, CPU golden comparisons)

| quantity | rel-max vs mbirjax |
|---|---|
| sparse forward | 1.93e-06 |
| sparse back | 4.14e-07 |
| full forward (phantom sinogram) | 2.58e-06 |
| Hessian diagonal | 1.25e-06 |
| FDK recon | 2.53e-06 |
| auto recon shape + slice offset | exact |
| convergence parity (5 it): alpha / fm / final | 2.2e-06 / 9.1e-06 / 6.7e-06 |

Adjointness holds on every backend (rel ~6e-08 measured), the recon smoke
converges on CPU and MPS, and the full suite stands at 47 passed / 1 skipped.
The golden cell uses the harness geometry convention (sdd = 4x channels,
magnification 2) at a 48-cell with full-circle angles.

## The DC-damping lesson

The parity gate is exactly the instrument that catches a missing
preconditioner: single-op goldens can never see it (the damping reshapes the
update direction, not the operator values), and an end-state NRMSE comparison
would blur it into "close enough".  Iteration-for-iteration traces failed
immediately and pointed at the engine rather than the math.  The port rule
recorded: when a geometry overrides `_get_update_direction` (or any engine
seam), the override IS part of the geometry's behavior and ports with it.

## CUDA smoke (gautschi, one H100)

Job 14843141 ran the full suite -- all cone gates included -- on one H100:
47 passed / 1 skipped in 2:39 (the wall is dominated by first-run inductor
compiles for the new cone kernels; the persistent cache absorbs them on
subsequent runs).  One advisory from the run worth noting for the tuning
work: torch suggests `torch.set_float32_matmul_precision('high')` to enable
TF32 tensor cores -- deliberately NOT enabled, since TF32 would loosen the
float32 parity floors; revisit only with the golden gates re-calibrated.

## Helical and curved-detector goldens (2026-08-05)

The two untested cone paths were closed with additional golden groups at the
48-cell.  The helical group uses z-shifts spanning +-8 ALU against a
detector half-height of 12 at iso, so the z-weight is nontrivial, and it
includes a seeded 3-iteration recon trace.  The curved group exercises the
theta = u/sdd horizontal-fan branch on a circular scan.  Every comparison
landed at the established floors on the first run:

| golden | helical | curved |
|---|---|---|
| sparse forward | 1.9e-06 | 1.9e-06 |
| sparse back | 3.4e-07 | 4.8e-07 |
| full forward | 2.8e-06 | 2.3e-06 |
| FDK | 2.3e-06 | 2.3e-06 |
| 3-iteration parity (alpha / fm / final) | 2.1e-06 / 3.3e-06 / 7.4e-06 | -- |

These paths were ported verbatim from mbirjax before any golden covered
them.  The first-run floors confirm the verbatim-port discipline held.

## The cone demo variant (2026-08-05)

demo_1 gained a MODEL_TYPE switch ('parallel' or 'cone'), mirroring the
mbirjax demo, with the cone using full-circle angles and the goldens'
magnification-2 distance convention.  The full-size check ran 256^3 cone on
MPS.  It converged in 11 iterations at the 0.2 percent threshold, in 101 s
including first-call compiles, with NRMSE 0.061 against the ground truth
phantom; the new get_memory_stats tail reported an 8.7 GiB MPS driver
high-water.

## The cone gate-cell readout (2026-08-05)

The readout is p3_cone_readout.py: the p2 protocol unchanged (one
framework/cell/op per subprocess, warm timing, the warm-vcd second run for
both frameworks) with cone models at the harness magnification-2 convention.
Both frameworks auto-set the recon geometry.  The goldens pin those shapes
as exactly equal, so the cells match across frameworks.

One H100 (gautschi job 14867702, node h008).  Entries are torch/jax ratios;
memory is the device peak during vcd:

| cell | filter | forward | back | vcd warm | mem (vcd) |
|---|---|---|---|---|---|
| 200x208x160 | 1.09 | 1.32 | 3.44 | **0.53** | 5.85 |
| 512x448x384 | 1.66 | 2.14 | 6.23 | 2.96 | 4.46 |
| 513x449x385 | 1.15 | 2.71 | 5.45 | 3.61 | 4.33 |
| 1024x1008x992 | 1.14 | 1.73 | 5.98 | 3.90 | **0.92** |

Local, cone 128-cell: CPU warm vcd 1.38x with peak RSS 1.24x, both within
the gates; MPS warm vcd 1.0 s, about 3x faster than jax on the same CPU.

The readout separates correctness from performance cleanly.  Correctness is
settled: every golden and parity gate holds on all three backends.  The CUDA
performance picture mirrors the parallel-beam one with a larger
back-projector gap.  Warm vcd passes the 2x gate at the 200 cell (0.53x, a
torch win) and exceeds it from 512 up (2.96-3.90x).  The driver is the cone
back projector at 3.4-6.2x across all cells.  That path is the horizontal
gather plus the banded vertical gather, both of which materialize their
taps.  Device memory follows the same mechanism.  At the small and mid
cells the gather transients and the channel-major transpose copy give
4.3-5.9x against jax's small footprints.  At the 1024 capacity cell the
view-batch budget clamps the torch transient while jax's own working set
grows, and torch comes in UNDER jax at 0.92x.  The cone back projector
therefore joins the Phase 5 Triton target list at the top, with the
small-cell transient memory as its second axis.

## Cluster staging lesson (three runs to a clean readout)

The first two H100 submissions failed on the torch side with
"module 'mbirtorch' has no attribute 'ConeBeamModel'".  The cause was
Python namespace-package shadowing: the repo checkout directory named
mbirtorch sat on sys.path and shadowed the installed package as an empty
namespace package.  The trap has two faces, which is why one fix round was
not enough.  python -c places the CURRENT directory first on sys.path, so a
clean run subdirectory fixed the preflight check; python script.py places
the SCRIPT'S directory first, so every worker subprocess still saw the
checkout sitting beside the script.  The durable fix stages the checkout
under a different name (mbirtorch_src).  The p2 runs never hit this only
because their scratch directory did not contain the checkout.
