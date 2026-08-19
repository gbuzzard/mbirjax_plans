"""mg42b -- does a multi-device reconstruction's end-state free at refcount,
or only at garbage collection?

This is the discriminating check behind the nightly's 26.6 GiB lead-device
reading (multigpu_findings.md section 1.35).  The candidate mechanism was a
reference cycle holding each call's end-state until garbage collection.  Two
virtual CPU devices reproduce the sharded containers exactly, and automatic
gc is disabled so anything cycle-held stays visible instead of vanishing
whenever the allocator happens to trip gc.

RESULT (2026-08-19, run locally on the Mac): the hypothesis is REFUTED.
After deleting a reconstruction's results, nine small tensors remain tracked
(0.2 MiB); across three back-to-back reconstructions, garbage collection
frees ZERO tensor bytes.  The end-state frees at refcount.  So no library
cycle fix is owed, and the nightly's reading has some other origin; the
nightly's own trial loop was separately read and is disciplined (it drops
the previous result and collects before each timed call), so the reading's
mechanism is recorded as unexplained.  The accuracy fix that needs no
mechanism: reset the peak counters before each trial and report the warm
trial's peak.

Run:  <python with mbirtorch> mg42b_cycle_check.py     (CPU, seconds)
"""
import gc

import numpy as np
import torch

import mbirtorch


def tracked_tensor_bytes():
    total = 0
    count = 0
    for obj in gc.get_objects():
        try:
            if type(obj) is torch.Tensor:
                total += int(obj.numel()) * int(obj.element_size())
                count += 1
        except (ReferenceError, TypeError, RuntimeError):
            continue
    return count, int(total)


gc.disable()

cell = (48, 40, 32)
angles = np.linspace(0, np.pi, cell[0], endpoint=False)
model = mbirtorch.ParallelBeamModel(cell, angles)
model.configure_devices(devices=['cpu', 'cpu'])
model.set_params(no_warning=True, verbose=0)

recon_shape = tuple(model.get_params('recon_shape'))
phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
sino = np.asarray(model.forward_project(phantom), dtype=np.float32)
weights = np.exp(-sino / (2 * np.max(sino))).astype(np.float32)

gc.collect()
base_count, base_bytes = tracked_tensor_bytes()
print(f"baseline after setup: {base_count} tensors, {base_bytes/2**20:.1f} MiB")

for trial in range(3):
    np.random.seed(13)
    out, stats = model.recon(sino, weights=weights, max_iterations=1,
                             stop_threshold_change_pct=0.0)
    del out, stats
    count_after, bytes_after = tracked_tensor_bytes()
    freed = gc.collect()
    count_gc, bytes_gc = tracked_tensor_bytes()
    print(f"trial {trial}: after del  {count_after} tensors "
          f"{bytes_after/2**20:7.1f} MiB | gc collected {freed} objects | "
          f"after gc {count_gc} tensors {bytes_gc/2**20:7.1f} MiB | "
          f"cycle-held {(bytes_after-bytes_gc)/2**20:7.1f} MiB")
