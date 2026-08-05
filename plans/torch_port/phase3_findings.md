# Phase 3 findings — the cone-beam port

**Status:** COMPLETE as a converging, parity-validated geometry 2026-08-05
(local CPU + MPS, and the H100 CUDA smoke); the remaining-scope items below
carry into the next stretch.
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

## Remaining Phase 3 scope

Cone gate-cell readout against the tuned jax baseline (the p2 readout script
extended with the cone geometry -- the harness cone recon-shape pins apply
there); helical and curved-detector golden coverage (the code paths exist and
follow mbirjax verbatim, but only circular/flat is golden-tested so far); and
the demo's cone variant.
