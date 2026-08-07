"""n=4 CUDA value-anomaly discriminator (post projector restructure): the
gate matrix read torch n4-vs-n1 at 2.16e-03 @512 (was 3.03e-04) with n=2
exact and all logic clean on virtual CPUs.  Three readings on 4 real GPUs:
(a) pure projection parity n4 vs n1 (driver/bodies isolation), (b) seeded
3-iter vcd parity (what the matrix measures), (c) both again with the
per-device view params rebuilt from HOST numpy instead of the pre-placed
cross-device copies -- the one new cross-GPU data motion the restructure
introduced.  Run: <torch python> p4c_n4_probe.py
"""
import numpy as np
import torch

import mbirtorch

assert torch.cuda.device_count() >= 4

cell = (512, 448, 384)
angles = np.linspace(0, np.pi, cell[0], endpoint=False)


def build(n):
    m = mbirtorch.ParallelBeamModel(cell, angles)
    m.set_params(no_warning=True, verbose=0)
    if n > 1:
        m.configure_devices(n)
    return m


def host_built_view_params(m):
    """Replace the pre-placed per-device view params with host-built copies
    (no cross-device hop)."""
    name = m.get_params('view_params_name')
    vp = np.asarray(m.get_params(name), dtype=np.float32)
    m.projector_functions._view_params_per_dev = [
        torch.as_tensor(vp, dtype=torch.float32, device=d)
        for d in m.sino_placement.devices]


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


m1 = build(1)
rs = tuple(m1.get_params('recon_shape'))
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
sino = np.asarray(m1.forward_project(phantom))
weights = np.exp(-sino / (2 * np.max(sino))).astype(np.float32)
rng = np.random.default_rng(4)
vol = rng.standard_normal(rs).astype(np.float32)
f1 = np.asarray(m1.forward_project(vol))
b1 = np.asarray(m1.back_project(sino))
np.random.seed(13)
r1, _ = m1.recon(sino, weights=weights, max_iterations=3,
                 stop_threshold_change_pct=0.0)

for label, patch in (("pre-placed", False), ("host-built", True)):
    m4 = build(4)
    if patch:
        host_built_view_params(m4)
    f4 = m4._gather_sinogram(m4.forward_project(vol, output_sharded=True))
    b4 = m4._gather_recon(m4.back_project(sino, output_sharded=True))
    print(f"{label}: fwd n4 rel {rel(f4, f1):.2e}  back n4 rel {rel(b4, b1):.2e}",
          flush=True)
    np.random.seed(13)
    r4, _ = m4.recon(sino, weights=weights, max_iterations=3,
                     stop_threshold_change_pct=0.0)
    print(f"{label}: vcd n4 rel {rel(r4, r1):.2e}", flush=True)

print("P4C PROBE DONE", flush=True)
