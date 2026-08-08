"""Compile-variant discriminator for the deterministic n>=3 divergence:
(1) n=4 vcd with compile OFF (eager = bit-identical math on every device);
(2) n=4 vcd with compile ON but ONE shared driver-body instance serving all
devices (variant identity across devices; execution stays serialized enough
by the GIL for a probe).  Clean (1) + clean (2) => per-device compiled
instances pick different kernel variants and the VCD loop amplifies the
float difference.  Run: <torch python> p4e_variant_probe.py
"""
import numpy as np
import torch

import mbirtorch

assert torch.cuda.device_count() >= 4

cell = (512, 448, 384)
angles = np.linspace(0, np.pi, cell[0], endpoint=False)


def build(n, compile_mode='auto'):
    m = mbirtorch.ParallelBeamModel(cell, angles, compile_mode=compile_mode)
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


m1e = build(1, compile_mode='off')
r1e = vcd(m1e, sino, weights)
print(f"eager n1 vs compiled n1 rel {rel(r1e):.2e}", flush=True)
m4e = build(4, compile_mode='off')
r4e = vcd(m4e, sino, weights)
print(f"eager n4 vs eager n1 rel "
      f"{float(np.max(np.abs(r4e - r1e)) / np.max(np.abs(r1e))):.2e}",
      flush=True)

m4s = build(4)
pf = m4s.projector_functions
pf._fwd_body_per_dev = [pf._fwd_body_per_dev[0]] * 4
pf._back_body_per_dev = [pf._back_body_per_dev[0]] * 4
print(f"shared-driver-instance n4 rel {rel(vcd(m4s, sino, weights)):.2e}",
      flush=True)
print("P4E PROBE DONE", flush=True)
