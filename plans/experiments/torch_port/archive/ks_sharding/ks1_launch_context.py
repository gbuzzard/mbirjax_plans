"""Single-variable probes: WHERE does a Triton forward launch go under the
banded multi-device drivers, and which one change repairs its values?

The isolation matrix (job 14954801) showed every kernel-FORWARD arm off by
order one at n=2 and n=4 in both geometries, non-reproducibly, while the
back-kernel arms reproduce the pure-torch arms to four significant figures.
Reading the launch path names a candidate mechanism: a Triton launch targets
the launching THREAD's current CUDA device and that device's thread-current
stream (triton 3.7.1 binds ``get_current_device = torch.cuda.current_device``
in backends/driver.py), while every torch op dispatches on the TENSOR's
device.  The banded drivers issue per-device work from fresh worker threads
(``_sharding.run_per_device``) that never set a device, and a fresh thread's
current CUDA device is 0 -- so every worker's kernel launch would land on
device 0's default stream, unordered against the shard's own producers and
consumers on its device's stream.

Each arm below varies exactly one thing against the plain kernel-forward
arm, and the outcomes discriminate the classes:

  worker log      the observation: thread-current device vs tensor device
                  inside every launch wrapper (no behavior change).
  plain           the defect reproduced (kernel fwd + TORCH back, so the
                  forward is blamed on its own), plus the same forward call
                  twice IN-PROCESS -- a race differs, a contract bug repeats.
  devctx          the launch bracketed in ``with torch.cuda.device(dev)``:
                  repairs values iff the mechanism is the launch context.
  sync_tensor     ``torch.cuda.synchronize(tensor.device)`` around the
                  launch: repairs iff the kernel ran on the tensor's device
                  and merely raced (same-device async class).
  sync_all        synchronize EVERY visible device around the launch:
                  repairs any ordering mechanism by brute force (an upper
                  bound, not a fix).
  contig          return a contiguous COPY instead of the permute view (the
                  back wrappers' input path forces a copy; the forward
                  returns a view): repairs iff the mechanism is view
                  lifetime, not launch placement.
  banded1         the full banded driver on ONE device (a two-shard
                  placement listing cuda:0 twice): wrong values here mean a
                  banded-contract bug in the wrapper; clean values exonerate
                  the contract and pin the defect to multi-device placement.

Every kernel arm binds the TORCH back body, so the forward carries the whole
comparison.  The torch arms bind torch both ways (MBIRTORCH_DISABLE_TRITON=1
in the arm's own environment -- compile_mode='off' does NOT disable kernels;
the standing protocol note).  Bodies are bound explicitly through the
``_view_batch_bodies`` hook, which also bypasses the interim torch-forward
selection that currently guards sharded layouts.

Environment:
    KS1_RESULTS=<dir>   where the jsonl, sinograms, and per-arm npys go
"""

import functools
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

RESULTS_DIR = os.environ.get("KS1_RESULTS", ".")
CELL = (256, 64, 64)
VCD_ITERATIONS = 3
VCD_SEED = 4321
# The recon-level floor for cells of this class (the dp5 ruler), and the
# forward-only kernel-vs-torch parity class measured by the kernel batteries.
RECON_FLOOR = 5e-3
FWD_FLOOR = 1e-4

_LAUNCH_LOG = []


def _build(geometry, cell):
    import mbirtorch

    if geometry == "parallel":
        angles = np.linspace(0, np.pi, cell[0], endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles, compile_mode="off")
    else:
        angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
        sdd = 4 * cell[2]
        model = mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=sdd,
                                        source_iso_dist=sdd,
                                        compile_mode="off")
    model.set_params(no_warning=True, verbose=0)
    return model


def _configure(model, n_label):
    if n_label == "dup2":
        model.configure_devices(devices=["cuda:0", "cuda:0"])
    else:
        model.configure_devices(int(n_label))


def _sino_path(geometry):
    return os.path.join(RESULTS_DIR, f"_ks1_sino_{geometry}_{CELL[0]}.npy")


def _out_path(geometry, variant, n_label, kind):
    return os.path.join(
        RESULTS_DIR, f"_ks1_{geometry}_{variant}_{n_label}_{kind}.npy")


def _wrap_forward(fn, variant):
    """The variant wrappers: one behavior change each, plus the launch-site
    observation (thread-current device vs tensor device) in all of them.
    The batching attributes ride along so every arm batches identically."""
    import torch

    @functools.wraps(fn)
    def wrapped(values, *args, **kwargs):
        dev = values.device
        _LAUNCH_LOG.append((threading.current_thread().name,
                            int(torch.cuda.current_device()), str(dev)))
        if variant == "devctx":
            with torch.cuda.device(dev):
                return fn(values, *args, **kwargs)
        if variant == "sync_tensor":
            torch.cuda.synchronize(dev)
            out = fn(values, *args, **kwargs)
            torch.cuda.synchronize(dev)
            return out
        if variant == "sync_all":
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            out = fn(values, *args, **kwargs)
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            return out
        out = fn(values, *args, **kwargs)
        return out.contiguous() if variant == "contig" else out

    wrapped._view_batch_cost = fn._view_batch_cost
    wrapped._mbirtorch_no_compile = True
    return wrapped


def _bind_bodies(model, geometry, variant):
    """Bind the arm's body pair explicitly: torch both ways for the 'torch'
    arm, else the (possibly wrapped) Triton forward against the TORCH back.
    Explicit binding bypasses both the availability probe and the interim
    sharded-layout selection, so the arm measures what it names."""
    if geometry == "parallel":
        from mbirtorch.parallel_beam import (_parallel_back_view_batch,
                                             _parallel_forward_view_batch)
        torch_fwd, torch_back = (_parallel_forward_view_batch,
                                 _parallel_back_view_batch)
        if variant != "torch":
            from mbirtorch.triton_parallel import \
                _parallel_forward_view_batch_triton as triton_fwd
    else:
        from mbirtorch.cone_beam import (_cone_back_view_batch,
                                         _cone_forward_view_batch)
        torch_fwd, torch_back = _cone_forward_view_batch, _cone_back_view_batch
        if variant != "torch":
            from mbirtorch.triton_cone import \
                _cone_forward_view_batch_triton as triton_fwd

    fwd = torch_fwd if variant == "torch" else _wrap_forward(triton_fwd, variant)
    model._view_batch_bodies = lambda: (fwd, torch_back)
    model.create_projectors()


def generate(cfg):
    """One phantom and torch-body sinogram per geometry, shared by every arm
    (the generate subprocess runs with MBIRTORCH_DISABLE_TRITON=1)."""
    import mbirtorch

    geometry = cfg["geometry"]
    model = _build(geometry, CELL)
    model.configure_devices(1)
    recon_shape = model.get_params("recon_shape")
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    sinogram = model.forward_project(phantom)
    if not isinstance(sinogram, np.ndarray):
        sinogram = sinogram.detach().cpu().numpy()
    np.save(_sino_path(geometry), np.asarray(sinogram, dtype=np.float32))
    return dict(cfg, recon_shape=list(recon_shape), path=_sino_path(geometry))


def run_arm(cfg):
    import torch

    import mbirtorch

    geometry, variant, n_label = cfg["geometry"], cfg["variant"], cfg["n"]
    model = _build(geometry, CELL)
    _configure(model, n_label)
    _bind_bodies(model, geometry, variant)

    fwd, back = model._view_batch_bodies()
    result = dict(cfg, forward_body=fwd.__name__, back_body=back.__name__,
                  visible=torch.cuda.device_count())

    recon_shape = model.get_params("recon_shape")
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)

    # Forward-only, twice in-process: a race differs between the two calls,
    # a deterministic contract bug repeats exactly (to the atomic float
    # floor, which the forward kernel's atomics set at ~1e-7).
    start = time.time()
    for rep in (1, 2):
        sino = model.forward_project(phantom)
        if not isinstance(sino, np.ndarray):
            sino = sino.detach().cpu().numpy()
        np.save(_out_path(geometry, variant, n_label, f"fwd{rep}"),
                np.asarray(sino, dtype=np.float32))
    result["fwd_seconds"] = time.time() - start

    # The recon is the proven reproducer (the dp5 kernel-forward arm at this
    # cell read order one), so it carries the verdict per variant.
    sinogram = np.load(_sino_path(geometry))
    weights = np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
    np.random.seed(VCD_SEED)
    start = time.time()
    recon, _info = model.recon(sinogram, weights=weights,
                               max_iterations=VCD_ITERATIONS,
                               stop_threshold_change_pct=0.0)
    result["recon_seconds"] = time.time() - start
    np.save(_out_path(geometry, variant, n_label, "recon"),
            np.asarray(recon, dtype=np.float32))

    # The launch-site observation, condensed: every distinct (thread-current
    # device, tensor device) pair seen at a kernel launch, with its count.
    # An unindexed 'cuda' tensor device means index 0.
    pairs, mismatched = {}, 0
    for _thread, cur, dev in _LAUNCH_LOG:
        key = f"current={cur} tensor={dev}"
        pairs[key] = pairs.get(key, 0) + 1
        tensor_index = int(dev.split(":")[1]) if ":" in dev else 0
        mismatched += int(cur != tensor_index)
    result["launch_pairs"] = pairs
    result["mismatched_launches"] = mismatched
    return result


def _spawn(cfg):
    env = dict(os.environ)
    # The kill switch is read inside the availability probe, which caches per
    # process, so it must be set before the interpreter starts.  Kernel arms
    # bind explicitly and never consult the probe, but the switch keeps any
    # OTHER selection in the arm honest.
    env["MBIRTORCH_DISABLE_TRITON"] = \
        "1" if cfg["variant"] == "torch" or cfg.get("mode") == "generate" \
        else "0"
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker",
         json.dumps(cfg)], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return dict(cfg, error=(proc.stderr[-2000:] or "") +
                    (proc.stdout[-500:] or ""))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    return dict(cfg, error="no result line")


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))


ARMS = [("torch", "1"), ("torch", "2"),
        ("plain", "1"), ("plain", "2"), ("plain", "dup2"),
        ("devctx", "2"), ("sync_tensor", "2"), ("sync_all", "2"),
        ("contig", "2")]


def main():
    import torch

    if torch.cuda.device_count() < 2:
        print(f"needs 2 CUDA devices, found {torch.cuda.device_count()}")
        sys.exit(1)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ks1_launch_context_{stamp}.jsonl")

    def record(row):
        with open(out_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")
        if "error" in row:
            print(f"    ERROR: {row['error'][:400]}", flush=True)
        return row

    rows = []
    for geometry in ("parallel", "cone"):
        if not os.path.exists(_sino_path(geometry)):
            print(f"generate {geometry}", flush=True)
            record(_spawn(dict(mode="generate", geometry=geometry,
                               variant="torch")))
        for variant, n_label in ARMS:
            print(f"  {geometry} {variant} n={n_label}", flush=True)
            rows.append(record(_spawn(dict(geometry=geometry, variant=variant,
                                           n=n_label))))

    print(f"\n===== forward launch probes ({out_path}) =====")
    print("Launch sites observed (kernel arms; current vs tensor device):")
    for row in rows:
        if row.get("launch_pairs"):
            print(f'  {row["geometry"]:>9} {row["variant"]:>11} '
                  f'n={row["n"]:<5} {row["launch_pairs"]}')

    header = (f'{"geometry":>9}{"variant":>12}{"n":>6}{"recon rel":>12}'
              f'{"fwd rel":>12}{"fwd repeat":>12}  verdict vs floor')
    print("\nEach arm against the TORCH arm at the same device count")
    print("(plain@dup2 reads against plain@1: the banded-contract probe).")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f'{row["geometry"]:>9}{row["variant"]:>12}{row["n"]:>6}'
                  f'{"failed":>12}')
            continue
        geometry, variant, n_label = row["geometry"], row["variant"], row["n"]
        ref_variant, ref_n = (("plain", "1") if n_label == "dup2"
                              else ("torch", n_label))
        try:
            ref_recon = np.load(_out_path(geometry, ref_variant, ref_n, "recon"))
            ref_fwd = np.load(_out_path(geometry, ref_variant, ref_n, "fwd1"))
            recon = np.load(_out_path(geometry, variant, n_label, "recon"))
            fwd1 = np.load(_out_path(geometry, variant, n_label, "fwd1"))
            fwd2 = np.load(_out_path(geometry, variant, n_label, "fwd2"))
        except FileNotFoundError as exc:
            print(f'{geometry:>9}{variant:>12}{n_label:>6}  missing: {exc}')
            continue
        recon_rel = _rel(ref_recon, recon)
        fwd_rel = _rel(ref_fwd, fwd1)
        repeat_rel = _rel(fwd1, fwd2)
        verdict = ("ref" if (variant, n_label) in (("torch", "1"), ("torch", "2"))
                   else ("ok" if recon_rel < RECON_FLOOR and fwd_rel < FWD_FLOOR
                         else "WRONG"))
        print(f'{geometry:>9}{variant:>12}{n_label:>6}{recon_rel:>12.3e}'
              f'{fwd_rel:>12.3e}{repeat_rel:>12.3e}  {verdict}')
    print("\nReading: devctx repaired alone = launch-context class; "
          "sync_tensor repaired = same-device async class; only sync_all "
          "repaired = ordering, placement unresolved; contig repaired = "
          "view-lifetime class; plain@dup2 wrong = banded-contract bug.")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        out = generate(cfg) if cfg.get("mode") == "generate" else run_arm(cfg)
        print("__RESULT__" + json.dumps(out))
    else:
        main()
