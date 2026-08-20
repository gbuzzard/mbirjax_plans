"""mg46 -- WHO SETS torch's RECOMPILE LIMIT, ON THE CLUSTER, DURING A
TWO-DEVICE MULTIAXIS RECONSTRUCTION.

WHY THIS PROBE EXISTS.  The library raises torch's per-function recompile
budget to 64 before it compiles anything (projectors.py,
_raise_recompile_budget), and the raise verifiably works on a local CPU
run: assigning the config caps or frees recompilation immediately, before
or after a compiled wrapper is created.  On the cluster gate (job
15394465) the same library produced torch's recompile-limit warning
reading (8) -- the value at conversion time was the default, so on that
node either the raise never ran or something assigned the default back.
This probe reruns the failing flow with two instruments: a checkpoint
print of the config value at each phase, and a tracer on the config
module's attribute setter that prints every assignment to
``recompile_limit`` or ``cache_size_limit`` with the stack that made it.

WHAT IT RUNS.  One process, two pinned devices, the multiaxis 512-class
cell, the staged mg26 sinogram, one reconstruction at one iteration --
the smallest flow that fired the warning on the gate.

HOW TO READ THE OUTPUT.  Lines starting ``LIMIT`` are the checkpoints.
Lines starting ``SETATTR`` are the tracer: every write to either config
name, its value, and the last frames of the stack that wrote it.  The
verdict is mechanical: if no SETATTR line shows the library's raise, the
raise never ran; if a SETATTR line shows 64 followed by one showing 8,
the stack on the second line names the restorer.

Run on a 2-GPU node:  <torch python> mg46_limit_probe.py
"""

import os
import sys
import time
import traceback

os.environ.setdefault("MBIRTORCH_NUM_DEVICES", "2")
os.environ.setdefault("MBIRTORCH_DISABLE_TRITON", "0")
os.environ.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
os.environ.pop("MBIRTORCH_RECOMPILE_LIMIT", None)

SINO_PATH = ("/scratch/gautschi/buzzard/torch_p3/results/mg26_floors/"
             "_sino_multiaxis_512x448x384.npy")
CELL = (512, 448, 384)
SEED = 13

import torch                                                      # noqa: E402
import torch._dynamo.config as dynamo_config                      # noqa: E402


def stamp(label):
    print(f"LIMIT {label}: recompile_limit={dynamo_config.recompile_limit} "
          f"cache_size_limit={dynamo_config.cache_size_limit}", flush=True)


# ── the tracer: every assignment to either limit name, with its stack ────────
_config_type = type(dynamo_config)
_original_setattr = _config_type.__setattr__


def _traced_setattr(self, name, value):
    if name in ("recompile_limit", "cache_size_limit"):
        stack = "".join(traceback.format_stack(limit=8)[:-1])
        print(f"SETATTR {getattr(self, '__name__', self)}.{name} = {value} "
              f"at {time.strftime('%H:%M:%S')}\n{stack}", flush=True)
    _original_setattr(self, name, value)


_config_type.__setattr__ = _traced_setattr

stamp("after torch import")
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)

import numpy as np                                                # noqa: E402

import mbirtorch                                                  # noqa: E402
from mbirtorch import projectors                                  # noqa: E402

print("library under test:", mbirtorch.__file__, flush=True)
print("recompile floor in the tree:",
      getattr(projectors, "_RECOMPILE_LIMIT_FLOOR", "ABSENT"), flush=True)
assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
stamp("after mbirtorch import")

azimuth = np.linspace(0, np.pi, CELL[0], endpoint=False)
elevation = np.linspace(-0.5, 0.5, CELL[0])
model = mbirtorch.MultiAxisParallelModel(
    tuple(CELL), np.stack([azimuth, elevation], axis=1))
model.set_params(no_warning=True, verbose=0)
stamp("after model build")

sinogram = np.load(SINO_PATH)
weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
stamp("after sinogram load")

np.random.seed(SEED)
out, _stats = model.recon(sinogram, weights=weights, max_iterations=1,
                          stop_threshold_change_pct=0.0)
for device in model.recon_placement.devices:
    torch.cuda.synchronize(device)
stamp("after 1-iteration recon")
print("realized devices:",
      [str(d) for d in model.recon_placement.devices], flush=True)
print("MG46 DONE", flush=True)
