"""Phase 4 spike 3: the REAL sharded engine on 2 H100s -- the fine-grained
dispatch measurement that the coarse p4s2 spike could not provide, plus the
jax study's 1-DEVICE DIAGNOSTIC (the sharded code path on one device prices
the orchestration overhead with no data movement in the frame).

Rows (each with a value check against the n=1 reference):
  n1        : plain single-device recon (the baseline).
  n1-diag   : the SHARDED code path on ONE device (devices=['cuda:0']) --
              wait, a 1-device placement is trivial by design, so the diag
              instead uses two placements on the SAME device
              (['cuda:0', 'cuda:0']): full orchestration, zero transfers.
  n2        : two H100s.
Timings: cold (first recon; per-process compile) and warm (second recon).
"""

import time

import numpy as np
import torch
import mbirtorch

CELL = (256, 256, 256)
ITERS = 4


def build():
    angles = np.linspace(0, np.pi, CELL[0], endpoint=False)
    m = mbirtorch.ParallelBeamModel(CELL, angles, device="cuda")
    m.set_params(no_warning=True, verbose=0)
    return m


def run(label, devices, sino, weights, ref=None):
    m = build()
    if devices is not None:
        m.configure_devices(devices=devices)
    times = []
    for _ in range(2):
        np.random.seed(13)
        t0 = time.perf_counter()
        recon, _ = m.recon(sino, weights=weights, max_iterations=ITERS,
                           stop_threshold_change_pct=0.0)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peaks = [torch.cuda.max_memory_allocated(i) / 2**30
             for i in range(torch.cuda.device_count())]
    diff = (np.max(np.abs(recon - ref)) / max(np.max(np.abs(ref)), 1e-30)
            if ref is not None else 0.0)
    print(f"{label:>8}: cold {times[0]:7.2f}s  warm {times[1]:7.2f}s  "
          f"gpu_peaks {['%.2f' % p for p in peaks]} GiB  rel_diff {diff:.2e}",
          flush=True)
    return recon, times[1]


def main():
    m = build()
    rs = tuple(m.get_params('recon_shape'))
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(rs)
    sino = m.forward_project(phantom)
    weights = np.exp(-np.asarray(sino) / (2 * np.max(sino)))
    del m
    torch.cuda.reset_peak_memory_stats()

    ref, t1 = run("n1", None, sino, weights)
    torch.cuda.reset_peak_memory_stats()
    _, td = run("n1-diag", ["cuda:0", "cuda:0"], sino, weights, ref)
    torch.cuda.reset_peak_memory_stats()
    _, t2 = run("n2", ["cuda:0", "cuda:1"], sino, weights, ref)
    print(f"\norchestration overhead (n1-diag / n1): {td / t1:.2f}x")
    print(f"n2 scaling vs n1 (warm):               {t1 / t2:.2f}x of 1.0 "
          f"(2.0 = ideal)", flush=True)


if __name__ == "__main__":
    main()
