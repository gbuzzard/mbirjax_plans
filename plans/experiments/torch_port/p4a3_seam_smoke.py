"""Phase 4a stage 3 seam smoke on real GPUs: the padded (non-dividing)
parallel engine and the un-guarded cone multi-device engine, each n=2 vs
its own n=1 reference.  Prints rel diffs; the CPU parity tests bound the
values, this run proves the seams on CUDA hardware (real transfers,
compiled per-device kernels, threads).

Run inside an sbatch with >= 2 CUDA devices: <torch python> p4a3_seam_smoke.py
"""
import numpy as np
import torch

import mbirtorch

assert torch.cuda.device_count() >= 2


def rel_diff(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def parallel_padded():
    # 255 views / 97 slices: both axes pad at n=2 (rows pad with slices).
    sino_shape = (255, 97, 128)
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
    np.random.seed(7)
    ref, _ = m1.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    m2 = build(2)
    assert m2.sino_placement.is_padded and m2.recon_placement.is_padded
    np.random.seed(7)
    out, _ = m2.recon(sino, weights=weights, max_iterations=3,
                      stop_threshold_change_pct=0.0)
    print(f"parallel padded n2 vs n1: rel {rel_diff(out, ref):.2e}", flush=True)


def cone_n2():
    cell = (128, 128, 128)
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)

    def build(n):
        m = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=4 * cell[2],
                                    source_iso_dist=2 * cell[2])
        m.set_params(no_warning=True, verbose=0)
        if n > 1:
            m.configure_devices(n)
        return m

    m1 = build(1)
    rs = tuple(m1.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
    sino = np.asarray(m1.forward_project(phantom))
    np.random.seed(11)
    ref, _ = m1.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)
    m2 = build(2)
    np.random.seed(11)
    out, _ = m2.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0)
    print(f"cone n2 vs n1 (DC damping per shard): rel {rel_diff(out, ref):.2e}",
          flush=True)


parallel_padded()
cone_n2()
print("P4A3 SMOKE DONE", flush=True)
