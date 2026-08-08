"""n>=3 vcd CUDA nondeterminism bisect: (1) same-process repeat (nondet
scale), (2) n=3 (band-count vs device-set), (3) back-driver clone-on-
accumulate toggle (inductor output aliasing), all vs one n=1 reference.
CUDA_LAUNCH_BLOCKING variant runs as a second job.
Run: <torch python> p4d_n4_bisect.py
"""
import numpy as np
import torch

import mbirtorch
from mbirtorch import projectors as _proj

assert torch.cuda.device_count() >= 4

cell = (512, 448, 384)
angles = np.linspace(0, np.pi, cell[0], endpoint=False)


def build(n):
    m = mbirtorch.ParallelBeamModel(cell, angles)
    m.set_params(no_warning=True, verbose=0)
    # UNCONDITIONAL: a CUDA model without an explicit
    # configure_devices call now spreads across every visible
    # device, so the n=1 arm must pin itself or it silently
    # becomes the very multi-device run it is the baseline for.
    m.configure_devices(n)
    return m


def vcd(m, sino, weights):
    np.random.seed(13)
    r, _ = m.recon(sino, weights=weights, max_iterations=3,
                   stop_threshold_change_pct=0.0)
    return r


m1 = build(1)
rs = tuple(m1.get_params('recon_shape'))
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
sino = np.asarray(m1.forward_project(phantom))
weights = np.exp(-sino / (2 * np.max(sino))).astype(np.float32)
r1 = vcd(m1, sino, weights)


def rel(r):
    return float(np.max(np.abs(r - r1)) / np.max(np.abs(r1)))


m3 = build(3)
print(f"n3 vcd rel {rel(vcd(m3, sino, weights)):.2e}", flush=True)
del m3
m4 = build(4)
print(f"n4 vcd rel A {rel(vcd(m4, sino, weights)):.2e}", flush=True)
print(f"n4 vcd rel B {rel(vcd(m4, sino, weights)):.2e}", flush=True)

# Clone-on-accumulate: never alias the compiled body's output buffer.
orig = _proj.Projectors.sparse_back_project_view_range


def cloned(self, local_sino, pixel_indices, view_range, coeff_power=1,
           slice_start=0, band_slices=None, dev_index=0):
    m = self.model
    v0, v1 = view_range
    args = m._view_batch_args()
    vb_size = self._effective_view_batch(pixel_indices.shape[0],
                                         local_sino.shape[1])
    view_params = self._view_params_per_dev[dev_index]
    out = None
    for v in range(v0, v1, vb_size):
        view_params_batch = view_params[v:min(v + vb_size, v1)]
        block = self._back_body_per_dev[dev_index](
            local_sino[v - v0:v - v0 + view_params_batch.shape[0]],
            pixel_indices, view_params_batch, coeff_power=coeff_power,
            slice_start=slice_start, band_slices=band_slices, **args)
        if out is None:
            out = block.clone()
        else:
            out.add_(block)
    return out


_proj.Projectors.sparse_back_project_view_range = cloned
m4c = build(4)
print(f"n4 vcd rel CLONED {rel(vcd(m4c, sino, weights)):.2e}", flush=True)
_proj.Projectors.sparse_back_project_view_range = orig
print("P4D BISECT DONE", flush=True)
