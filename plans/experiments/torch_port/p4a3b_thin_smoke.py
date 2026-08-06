"""Extreme-aspect extension smokes on 4 real GPUs: more devices than one
axis can fill.  Thin volume (3 slices, many views: one all-padded SLICE
shard) and sparse view (3 views, many slices: one all-padded VIEW shard),
each n=4 vs its n=1 reference.  Run: <torch python> p4a3b_thin_smoke.py
"""
import numpy as np
import torch

import mbirtorch

assert torch.cuda.device_count() >= 4


def run_case(label, sino_shape, empty_axis):
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build(n):
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.set_params(no_warning=True, verbose=0)
        if n > 1:
            m.configure_devices(n)
        return m

    m1 = build(1)
    rs = tuple(m1.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
    sino = np.asarray(m1.forward_project(phantom))
    weights = np.exp(-sino / (2 * np.max(sino))).astype(np.float32)
    np.random.seed(17)
    ref, _ = m1.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)

    m2 = build(4)
    placement = (m2.recon_placement if empty_axis == 'slice'
                 else m2.sino_placement)
    assert placement.padded_shard_ranges()[-1][2] == 0
    np.random.seed(17)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))
    print(f"{label} n4 vs n1: rel {rel:.2e}", flush=True)


def run_floor_case(label, sino_shape):
    """The dividing-case floor of the same cell class: no padding, no empty
    shard -- the inherent n=4 divergence the extension gates calibrate
    against."""
    angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)

    def build(n):
        m = mbirtorch.ParallelBeamModel(sino_shape, angles)
        m.set_params(no_warning=True, verbose=0)
        if n > 1:
            m.configure_devices(n)
        return m

    m1 = build(1)
    rs = tuple(m1.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
    sino = np.asarray(m1.forward_project(phantom))
    weights = np.exp(-sino / (2 * np.max(sino))).astype(np.float32)
    np.random.seed(17)
    ref, _ = m1.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    m2 = build(4)
    assert not m2.sino_placement.is_padded
    np.random.seed(17)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    rel = float(np.max(np.abs(out - ref)) / max(np.max(np.abs(ref)), 1e-30))
    print(f"{label} n4 vs n1: rel {rel:.2e}", flush=True)


run_case("thin parallel (3 slices, empty slice shard)", (512, 3, 256),
         'slice')
run_case("sparse-view parallel (3 views, empty view shard)", (3, 256, 256),
         'view')
run_floor_case("dividing floor (4 views, no padding)", (4, 256, 256))
print("P4A3B SMOKE DONE", flush=True)
