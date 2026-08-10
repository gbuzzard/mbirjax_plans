"""Shared-compiled-instances probe: force maybe_compile to ignore
instance_key everywhere (driver bodies AND the VCD updater's units), so all
device threads execute ONE compiled instance per function -- variant
identity across devices; the global lock still serializes cold compiles.
Expect the n=4 divergence to return to the eager floor (~5e-04) if
per-device variant divergence is the whole story.  Runs vcd twice
(stability + no launcher crash under repeated concurrent shared
execution).  Run: <torch python> p4f_shared_probe.py
"""
import numpy as np
import torch

import mbirtorch
from mbirtorch import projectors as _proj
from mbirtorch import tomography_model as _tm
from mbirtorch import cone_beam as _cb
from mbirtorch import parallel_beam as _pb

assert torch.cuda.device_count() >= 4

_orig = _proj.maybe_compile


def shared_maybe_compile(fn, enabled, instance_key=None):
    return _orig(fn, enabled)


for mod in (_proj, _tm, _cb, _pb):
    if hasattr(mod, 'maybe_compile'):
        mod.maybe_compile = shared_maybe_compile

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

m4 = build(4)
for run in ("A", "B"):
    r4 = vcd(m4, sino, weights)
    rel = float(np.max(np.abs(r4 - r1)) / np.max(np.abs(r1)))
    print(f"shared-all n4 rel {run} {rel:.2e}", flush=True)
print("P4F PROBE DONE", flush=True)
