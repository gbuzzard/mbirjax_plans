"""mg15d -- THE DECIDING SINGLE-VARIABLE TEST: is the uneven-split adjoint
difference produced by torch.compile's DYNAMIC-SHAPE specialization?

WHERE THIS SITS, in one paragraph.  mg15 gated the removal of the sharding pad;
its one failing leg (multiaxis reconstruction at four devices, 2.969e-02) was
adjudicated by mg15b as trajectory-shaped, but mg15b's new adjoint instrument
tripped at 6.289e-04.  mg15c (slurm 15304817) took that apart with a count
matrix and a pre-removal ablation, and the result was decisive in an unexpected
direction:

    current tree, multiaxis adjoint     n=1 vs n=2      6.976e-07
      (n=2: both axes even)             n=1 vs n=3      ~6.29e-04
      (n=3: views 171/171/170 uneven)   n=1 vs n=4      ~6.29e-04
      (n=4: slices 128/128/127/127)     n=3 vs n=4      5.975e-04
    same-count repeats                                  EXACTLY 0.0
    cone adjoint, uneven n=3                            7.1e-07
    parallel adjoint, uneven n=3                        6.9e-07
    PADDED tree (672edbd), n=4                          5.980e-07

Five facts, and only one mechanism fits all five.  The difference appears
whenever EITHER sharded axis carries two distinct block lengths, and not when
the blocks are equal (n=2 is clean, the padded tree is clean).  It is
DETERMINISTIC -- the same-count repeats are exactly zero, so it is not atomics
and not scheduling.  It is confined to multiaxis, the family that runs a
general torch body, while cone and parallel, which back-project through
hand-written Triton kernels, are clean at the same uneven count.

That is the signature of the one cost remove_padding.md §2 predicted when the
pad was removed, and it predicted it as a SPEED cost only:

    "One small cost remains: a second distinct shape makes torch.compile treat
    that dimension as dynamic, which can produce slightly slower code than a
    specialized graph."

A dynamically-compiled body is not merely slower.  Inductor generates different
code for it, and different code contracts a reduction in a different order --
which for a back projection means a voxel's ~512 same-sign float32
contributions are summed in a different association than the
shape-specialized graph would use.  The mechanism explains every one of the
five facts: determinism (compiled code is fixed once generated), the
either-axis trigger (any second block length makes that dimension dynamic),
Triton immunity (hand-written kernels are marked no-compile and never reach
inductor codegen), the padded tree's cleanliness (equal blocks stay static),
and even mg15's mildly elevated forward reading of 1.585e-05 at n=4, whose
slice bands are uneven too.

THE TEST.  Eager bodies have no shape specialization at all -- there is no
graph to specialize and no inductor codegen to change -- so if the mechanism is
what it appears to be, running the identical uneven split with compilation OFF
must collapse the difference to the float32 floor, while compilation on
reproduces mg15c's reading.  Four arms, one variable:

    compile_mode='auto', n=1        compile_mode='off', n=1
    compile_mode='auto', n=4        compile_mode='off', n=4

    auto n=4 vs n=1   expected ~6.29e-04, reproducing mg15c
    off  n=4 vs n=1   THE DISCRIMINATOR.  ~1e-7 confirms the dynamic-shape
                      mechanism; ~6e-4 refutes it and sends the question back
                      to the split itself.
    off n=1 vs auto n=1   context, no gate: how far eager and compiled differ
                      at the SAME device count, which is the scale any
                      compiled-against-eager comparison sits on.

WHY compile_mode GOES THROUGH THE CONSTRUCTOR, and why each arm is its own
process.  Both are correctness requirements here, not hygiene:

  * The projector layer reads ``model.compile_enabled`` ONCE, when the
    Projectors object is built, and binds that decision into each device's
    body (projectors.py, ``Projectors.__init__`` -> ``bind``).  Setting
    ``model.compile_mode = 'off'`` after the model exists does not rebind
    anything -- the bodies are already bound, and the repo's own lessons record
    a compile-off arm set after model build as invalid for exactly this reason.
    So the mode is threaded through the geometry constructor, which is the only
    place that reaches the binding.

  * ``maybe_compile`` caches compiled instances at MODULE level, keyed by
    (function, device index), deliberately, so rebuilding a model's projectors
    reuses them.  A process-wide cache is exactly what an ablation must not
    share between its arms, and one subprocess per arm removes the question.
    (For the record, the disabled path returns the raw function before the
    cache is consulted, so an 'off' arm could not pick up a compiled instance
    even in a shared process; the separation is kept because relying on that
    detail would make the ablation depend on an implementation detail rather
    than on its own design.)

THE CONFOUND THIS ARM GUARDS AGAINST.  ``maybe_compile`` wraps the first call
so that if compilation fails at runtime -- a broken toolchain, a bad triton --
it retries EAGERLY, records the failure in ``projectors._COMPILE_ERRORS``, and
permanently rebinds to eager.  That is good behavior in production and a trap
here: an 'auto' arm that silently fell back to eager would read exactly like an
'off' arm and would look like confirmation.  So every arm records, after its
projection, whether each bound back-projection body is actually a compiled
object (identity against the raw body) and the contents of _COMPILE_ERRORS.  An
'auto' arm that did not really compile is reported as an instrument failure,
not as a reading.

NO GATES.  The exit code reports INSTRUMENT HEALTH only -- every arm ran, every
comparison computed, and every arm really built in the mode it claimed.  The
verdict is read by a human from the printed table.  Wall time per projection is
recorded because eager will be slower and that number is context for any
remedy discussion; it is not a verdict about anything.

Run:
    <torch python> mg15d_compile_ablation.py     on a 4-GPU node
    python mg15d_compile_ablation.py --dry-run   anywhere: print the arm plan
    python mg15d_compile_ablation.py --help

Environment:
    MG15D_RESULTS=<dir>            where the jsonl and the artifacts go
    MG15D_KEEP_ARTIFACTS=1         keep the staged volumes after the run
    MG15D_SMOKE=1 / MG15D_DEVICE=cpu   the local CPU smoke
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
# mg15c's multiaxis cell and angles, unchanged, so the 'auto' arm here is
# directly comparable to the reading it is reproducing.
CELL = (512, 448, 384)
SMOKE = os.environ.get("MG15D_SMOKE", "0") == "1"
SMOKE_CELL = (16, 24, 20)
CELL_IN_USE = SMOKE_CELL if SMOKE else CELL

DEVICE = os.environ.get("MG15D_DEVICE", "cpu" if SMOKE else "cuda")
REF_COUNT = 1
# The multi-device count carrying an uneven split.  On the cluster n=4 leaves
# the recon slices at 128/128/127/127; on the smoke n=3 leaves the views at
# 6/5/5.  Either is a second distinct block length, which is all the mechanism
# under test needs.
MULTI_COUNT = 3 if SMOKE else 4
COMPILE_MODES = ("auto", "off")
PHANTOM_SEED = 12345

# mg15c's readings, echoed so the summary prints them beside the new ones.
# Source: slurm 15304817.
MG15C_JOB = "slurm 15304817"
MG15C_READINGS = (
    ("multiaxis adjoint n=1 vs n=2 (both even)", 6.976e-07, "clean"),
    ("multiaxis adjoint n=1 vs n=3 (views uneven)", 6.29e-04, "the effect"),
    ("multiaxis adjoint n=1 vs n=4 (slices uneven)", 6.29e-04, "the effect"),
    ("multiaxis adjoint n=3 vs n=4", 5.975e-04, "the effect"),
    ("multiaxis adjoint same-count repeats", 0.0, "EXACTLY zero: deterministic"),
    ("cone adjoint n=1 vs n=3 (uneven, Triton)", 7.1e-07, "kernels are clean"),
    ("parallel adjoint n=1 vs n=3 (uneven, Triton)", 6.9e-07, "kernels are clean"),
    ("multiaxis adjoint PADDED tree n=1 vs n=4", 5.980e-07, "equal blocks: clean"),
)
PREDICTION = ('remove_padding.md section 2: "a second distinct shape makes '
              'torch.compile treat that dimension as dynamic, which can '
              'produce slightly slower code than a specialized graph"')

HOT_CORE_C = 85
HOT_HBM_C = 95
_GPU_FIELDS_FULL = ("index,clocks.sm,clocks.mem,temperature.gpu,temperature.memory,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown,"
                    "clocks_throttle_reasons.hw_power_brake_slowdown,"
                    "clocks_throttle_reasons.sw_power_cap")
_GPU_FIELDS_MIN = "index,clocks.sm,temperature.gpu"
_THROTTLE_NAMES = ("hw_thermal", "sw_thermal", "hw_power_brake", "sw_power_cap")

RESULTS_DIR = os.environ.get(
    "MG15D_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def _sino_path():
    return os.path.join(RESULTS_DIR, "_mg15d_sino.npy")


def _phantom_path():
    return os.path.join(RESULTS_DIR, "_mg15d_phantom.npy")


def _bp_path(mode, n_dev):
    return os.path.join(RESULTS_DIR, f"_mg15d_bp_{mode}_n{n_dev}.npy")


def _md5_path(path):
    return path + ".md5"


def _md5(path, chunk=8 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(x):
    """The ONE host exit.

    A sharded array's ``gather()`` ALREADY returns a numpy array; calling
    ``.detach()`` on that result is a recorded failure that once cost a whole
    multi-device run its rows.
    """
    import numpy as np

    if isinstance(x, np.ndarray):
        return x
    gather = getattr(x, "gather", None)
    if callable(gather) and hasattr(x, "placement"):
        return gather()                       # ALREADY numpy: do not re-detach
    detach = getattr(x, "detach", None)
    if callable(detach):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compare_arrays(out, ref, budget_bytes=64 << 20):
    """max|out - ref| / max|ref| in float64, with a normalized RMS beside it.

    mg15b's and mg15c's comparator, unchanged, so a number printed here means
    the same thing as a number printed there.  Walked in slabs along the first
    axis; the maximum is accumulated slab by slab, which is exact.
    """
    import numpy as np

    if tuple(out.shape) != tuple(ref.shape):
        return dict(rel=None,
                    reason=f"shape {list(out.shape)} is not the reference's "
                           f"{list(ref.shape)}")
    row_bytes = max(1, int(np.prod(ref.shape[1:])) * 8)
    step = max(1, int(budget_bytes // row_bytes))
    max_diff, max_ref, sq_diff, sq_ref = 0.0, 0.0, 0.0, 0.0
    for start in range(0, ref.shape[0], step):
        a = np.asarray(ref[start:start + step], dtype=np.float64)
        b = np.asarray(out[start:start + step], dtype=np.float64)
        diff = b - a
        max_ref = max(max_ref, float(np.max(np.abs(a))))
        max_diff = max(max_diff, float(np.max(np.abs(diff))))
        sq_diff += float(np.sum(diff * diff))
        sq_ref += float(np.sum(a * a))
    if max_ref <= 0.0:
        return dict(rel=None,
                    reason="the reference is all zeros, so a relative "
                           "comparison has no denominator")
    return dict(rel=max_diff / max_ref, max_abs_diff=max_diff,
                max_abs_ref=max_ref, shape=list(ref.shape),
                nrmse=((sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else None))


# ── the GPU health sample ─────────────────────────────────────────────────────
def _gi(text):
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def sample_gpu_health():
    for fields in (_GPU_FIELDS_FULL, _GPU_FIELDS_MIN):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=" + fields,
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
        except Exception:                                         # noqa: BLE001
            return []
        if proc.returncode != 0:
            continue
        full = fields is _GPU_FIELDS_FULL
        out = []
        for line in proc.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            entry = {"index": _gi(parts[0]), "sm_mhz": _gi(parts[1])}
            if full and len(parts) >= 9:
                entry["mem_mhz"] = _gi(parts[2])
                entry["temp_c"] = _gi(parts[3])
                entry["mem_temp_c"] = _gi(parts[4])
                entry["throttle"] = [name for name, value
                                     in zip(_THROTTLE_NAMES, parts[5:9])
                                     if value.lower() == "active"]
            else:
                entry["temp_c"] = _gi(parts[2])
            out.append(entry)
        if out:
            return out
    return []


def row_is_hot(health):
    for gpu in health:
        core, hbm = gpu.get("temp_c"), gpu.get("mem_temp_c")
        if (core is not None and core >= HOT_CORE_C) or \
                (hbm is not None and hbm >= HOT_HBM_C):
            return True
    return False


# ── model construction ────────────────────────────────────────────────────────
def device_list(n_dev):
    """The explicit device list an arm pins to, installed at construction."""
    if DEVICE == "cuda":
        return [f"cuda:{i}" for i in range(n_dev)]
    return [DEVICE] * n_dev


def build_model(n_dev, compile_mode):
    """mg15c's multiaxis model, with the compile mode threaded through the
    CONSTRUCTOR.

    The mode has to be set here and cannot be flipped afterwards: the projector
    layer reads ``model.compile_enabled`` once, when the Projectors object is
    built, and binds that decision into each device's body.  A model whose
    compile_mode is changed after construction keeps the bodies it already
    bound, so a compile-off arm arranged that way would silently be a
    compile-ON arm -- the failure the repo's lessons record.
    """
    import numpy as np

    import mbirtorch

    cell = tuple(CELL_IN_USE)
    num_views = cell[0]
    model = mbirtorch.MultiAxisParallelModel(
        cell, np.stack([np.linspace(0, np.pi, num_views, endpoint=False),
                        np.linspace(-0.5, 0.5, num_views)], axis=1),
        compile_mode=compile_mode)
    model.configure_devices(devices=device_list(n_dev))
    model.set_params(no_warning=True, verbose=0)
    return model


def _blocks(model):
    """The block each device owns on the two sharded axes.

    The whole hypothesis is about a SECOND distinct block length, so the block
    lists are what say whether an arm exercised the condition at all.
    """
    sinogram_shape = tuple(model.get_params("sinogram_shape"))
    recon_shape = tuple(model.get_params("recon_shape"))
    views = [end - start for _d, (start, end)
             in model.sino_placement.shard_ranges(sinogram_shape[0])]
    slices = [end - start for _d, (start, end)
              in model.recon_placement.shard_ranges(recon_shape[2])]
    return dict(view_blocks=views, slice_blocks=slices,
                distinct_view_lengths=len(set(views)),
                distinct_slice_lengths=len(set(slices)),
                # The condition under test: two distinct lengths on EITHER
                # axis is what makes a dimension dynamic.
                uneven=(len(set(views)) > 1 or len(set(slices)) > 1))


def _compile_state(model):
    """Did this arm REALLY build in the mode it claims?

    ``maybe_compile`` returns the raw function when compilation is off, and
    also permanently rebinds to the raw function if a compile fails at runtime
    and the eager retry succeeds.  So the honest test is identity: a bound body
    that IS the raw body did not compile.  Read after the projection, so a
    first-call fallback has already had its chance to happen.
    """
    from mbirtorch import projectors

    raw_fwd, raw_back = model._view_batch_bodies()
    functions = model.projector_functions
    bound_back = list(getattr(functions, "_back_body_per_dev", []) or [])
    bound_fwd = list(getattr(functions, "_fwd_body_per_dev", []) or [])
    return dict(
        compile_mode=model.compile_mode,
        compile_enabled=bool(model.compile_enabled),
        back_body_compiled=[b is not raw_back for b in bound_back],
        forward_body_compiled=[b is not raw_fwd for b in bound_fwd],
        back_body_name=getattr(raw_back, "__name__", str(raw_back)),
        # A non-empty registry means a compile failed and the arm silently ran
        # eager.  That invalidates an 'auto' arm rather than being a reading.
        compile_errors=dict(getattr(projectors, "_COMPILE_ERRORS", {}) or {}))


def _verify_staged(result):
    """Re-check the staged phantom and sinogram before using them.  All four
    arms read these bytes; a file that changed underneath the run would make
    the ablation compare two different problems."""
    for label, path in (("sino", _sino_path()), ("phantom", _phantom_path())):
        with open(_md5_path(path)) as handle:
            expected = handle.read().strip()
        actual = _md5(path)
        result[f"{label}_md5"] = actual
        result[f"{label}_md5_ok"] = (actual == expected)
        if actual != expected:
            raise RuntimeError(f"staged {label} checksum mismatch at {path}: "
                               f"{actual} != {expected}")


def _save(path, volume):
    """Stage a volume as float32 with a checksum sidecar."""
    import numpy as np

    volume = np.ascontiguousarray(np.asarray(volume, dtype=np.float32))
    np.save(path, volume)
    digest = _md5(path)
    with open(_md5_path(path), "w") as handle:
        handle.write(digest + "\n")
    return dict(path=path, md5=digest, shape=list(volume.shape),
                abs_sum=float(np.sum(np.abs(volume, dtype=np.float64))))


# ── the workers ───────────────────────────────────────────────────────────────
def run_backproject(cfg):
    """One back projection of the staged sinogram, at one device count, in one
    compile mode.  That is the entire arm."""
    import numpy as np
    import torch

    n_dev, mode = cfg["n_dev"], cfg["compile_mode"]
    result = dict(cfg, device=DEVICE)
    model = build_model(n_dev, mode)
    realized = [str(d) for d in model.sino_placement.devices]
    result.update(requested_devices=device_list(n_dev),
                  realized_devices=realized, realized_n_devices=len(realized),
                  devices_ok=(len(realized) == n_dev),
                  pin_mechanism="configure_devices(devices=[...])",
                  recon_shape=list(model.get_params("recon_shape")),
                  blocks=_blocks(model), interpreter=sys.executable,
                  env_disable_triton=os.environ.get("MBIRTORCH_DISABLE_TRITON"))
    _verify_staged(result)
    sinogram = np.load(_sino_path())
    health = [sample_gpu_health()]

    start = time.perf_counter()
    volume = _to_numpy(model.back_project(sinogram))
    if DEVICE == "cuda" and torch.cuda.is_available():
        for device in model.sino_placement.devices:
            torch.cuda.synchronize(device)
    # The wall clock includes compilation on the 'auto' arms, because this is
    # the only projection the arm performs.  It is recorded as context for a
    # remedy discussion and is NOT a timing measurement of the two modes --
    # that would need a warm pass, which this probe deliberately does not take.
    result["back_project_s"] = time.perf_counter() - start
    result["staged"] = _save(_bp_path(mode, n_dev), volume)
    # Read AFTER the projection, so a first-call compile failure that fell back
    # to eager has already been recorded.
    result["compile_state"] = _compile_state(model)
    health.append(sample_gpu_health())
    result["gpu_health"] = [g for snap in health for g in snap]
    result["gpu_hot"] = row_is_hot(result["gpu_health"])
    return result


def generate(cfg):
    """The staged phantom and its one-device sinogram.

    Built with compilation on, the shipped configuration: this is the input to
    all four arms, not one of the measurements, and it must be the same bytes
    for every arm.  The dots draw is seeded and both files are checksummed.
    """
    import numpy as np
    import torch

    import mbirtorch

    result = dict(cfg, device=DEVICE)
    model = build_model(1, "auto")
    recon_shape = tuple(model.get_params("recon_shape"))
    np.random.seed(PHANTOM_SEED)
    phantom = np.ascontiguousarray(np.asarray(
        mbirtorch.gen_translation_phantom(recon_shape, "dots", None,
                                          fill_rate=0.05), dtype=np.float32))
    sinogram = _to_numpy(model.forward_project(phantom))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result.update(recon_shape=list(recon_shape),
                  num_pixels_full=int(model.full_index_count()))
    result["phantom"] = _save(_phantom_path(), phantom)
    result["sino"] = _save(_sino_path(), sinogram)
    del phantom, sinogram, model
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


WORKERS = dict(generate=generate, backproject=run_backproject)


# ── the driver ────────────────────────────────────────────────────────────────
def arm_env(cfg):
    """The environment that DEFINES an arm.

    Both mbirtorch knobs are popped: this probe reads no memory, and every arm
    pins by explicit device list.  MBIRTORCH_DISABLE_TRITON stays 0, the
    shipped configuration -- note that it is NOT the variable under test here.
    That switch turns off the hand-written kernels; this ablation turns off
    torch.compile, which is a different mechanism, and multiaxis has no
    hand-written kernels for the Triton switch to affect anyway.
    """
    env = dict(os.environ)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def _spawn(cfg):
    """Run one configuration in a FRESH interpreter.

    Required, not merely tidy: maybe_compile caches compiled instances at
    module level for the life of the process, and an ablation must not share a
    compile cache between its arms.
    """
    payload = json.dumps(cfg)
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", os.path.abspath(__file__), "--worker", payload],
        capture_output=True, text=True, env=arm_env(cfg))
    wall = time.perf_counter() - start
    if proc.returncode != 0:
        return dict(cfg, error=proc.stderr[-3000:], subprocess_wall_s=wall)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("__RESULT__"):
            row = json.loads(line[len("__RESULT__"):])
            row["subprocess_wall_s"] = wall
            return row
    return dict(cfg, error="no result line\n" + proc.stdout[-3000:],
                subprocess_wall_s=wall)


def build_plan():
    plan = [dict(mode="generate", arm_id="generate", n_dev=1)]
    for compile_mode in COMPILE_MODES:
        for n in (REF_COUNT, MULTI_COUNT):
            plan.append(dict(mode="backproject",
                             arm_id=f"{compile_mode}_n{n}", n_dev=n,
                             compile_mode=compile_mode))
    return plan


def _dry_run(plan):
    arms = [c for c in plan if c["mode"] == "backproject"]
    print(f"mg15d compile ablation: {len(arms)} back-projection arms "
          f"({len(plan) - len(arms)} generator), device {DEVICE}, cell "
          f"{tuple(CELL_IN_USE)}")
    print(f"  testing whether mg15c's {MG15C_READINGS[2][1]:.3e} adjoint "
          f"difference is torch.compile dynamic-shape specialization")
    for cfg in plan:
        if cfg["mode"] == "generate":
            print(f'  {cfg["arm_id"]:<14} stage phantom + one-device sinogram')
            continue
        print(f'  {cfg["arm_id"]:<14} n={cfg["n_dev"]}  '
              f'compile_mode={cfg["compile_mode"]!r} (set on the CONSTRUCTOR)')
    print(f"  the discriminator: off n={MULTI_COUNT} vs n={REF_COUNT}.  "
          f"~1e-7 confirms the dynamic-shape mechanism; ~6e-4 refutes it.")
    print(f"  prediction on record -- {PREDICTION}")
    print("no gates: the exit code reports INSTRUMENT HEALTH only.")


def main():
    plan = build_plan()
    if "--dry-run" in sys.argv:
        _dry_run(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(
        RESULTS_DIR, f"mg15d_compile_ablation_{RUN_LABEL}_{stamp}.jsonl")
    print(f"mg15d compile ablation on {RUN_LABEL} ({DEVICE}) -> {out_path}",
          flush=True)
    rows = []
    with open(out_path, "w") as sink:
        for cfg in plan:
            print(f'  {cfg["arm_id"]}', flush=True)
            row = _spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
        summary = summarize(rows, out_path)
        sink.write(json.dumps(dict(summary=summary)) + "\n")
        sink.flush()
    if os.environ.get("MG15D_KEEP_ARTIFACTS", "0") != "1":
        paths = [_sino_path(), _phantom_path()]
        for compile_mode in COMPILE_MODES:
            for n in (REF_COUNT, MULTI_COUNT):
                paths.append(_bp_path(compile_mode, n))
        for path in list(paths):
            paths.append(_md5_path(path))
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
    else:
        print(f"MG15D_KEEP_ARTIFACTS=1: the staged volumes are left in "
              f"{RESULTS_DIR}")
    print(f"\nwrote {out_path}")
    return 0 if summary["instruments_healthy"] else 2


def _staged(by_id, arm_id):
    entry = (by_id.get(arm_id) or {}).get("staged") or {}
    path = entry.get("path")
    return path if path and os.path.exists(path) else None


def _pair(by_id, arm_a, arm_b, problems, label):
    import numpy as np

    a, b = _staged(by_id, arm_a), _staged(by_id, arm_b)
    if not a or not b:
        problems.append(f"{label}|missing volume")
        return None
    return compare_arrays(np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r"))


def _fmt(check, key="rel"):
    if not check or check.get(key) is None:
        return "n/a"
    return f'{check[key]:.3e}'


def summarize(rows, out_path):
    print(f"\n===== mg15d: is the uneven-split adjoint difference "
          f"torch.compile dynamic-shape specialization? ({out_path}) =====")
    by_id = {r.get("arm_id"): r for r in rows}
    problems = []

    print("\nARMS")
    for row in rows:
        arm_id = row.get("arm_id")
        if row.get("error"):
            print(f'  {arm_id:<14} ERROR: '
                  f'{str(row["error"]).splitlines()[-1][:78]}')
            problems.append(f"{arm_id}|error")
            continue
        if row.get("mode") == "generate":
            print(f'  {arm_id:<14} staged recon shape {row.get("recon_shape")}')
            continue
        blocks = row.get("blocks") or {}
        state = row.get("compile_state") or {}
        compiled = state.get("back_body_compiled") or []
        print(f'  {arm_id:<14} mode={state.get("compile_mode"):<5} '
              f'on {row.get("realized_devices")}  views '
              f'{blocks.get("view_blocks")} slices {blocks.get("slice_blocks")}'
              f'{"  UNEVEN" if blocks.get("uneven") else "  even"}  '
              f'back body compiled={compiled}  '
              f'{row.get("back_project_s", 0):.1f}s')
        if row.get("devices_ok") is False:
            print(f'    ARM CHECK FAIL: realized {row.get("realized_devices")} '
                  f'for n={row.get("n_dev")}')
            problems.append(f"{arm_id}|devices")
        # The arm must really have built in the mode it claims.  An 'auto' arm
        # that fell back to eager would read exactly like an 'off' arm.
        wanted = (state.get("compile_mode") == "auto")
        if compiled and any(c is not wanted for c in compiled):
            print(f'    ARM CHECK FAIL: compile_mode='
                  f'{state.get("compile_mode")!r} but the bound back bodies '
                  f'report compiled={compiled}; this arm did not build in the '
                  f'mode it claims')
            problems.append(f"{arm_id}|compile_state")
        if state.get("compile_errors"):
            print(f'    ARM CHECK FAIL: torch.compile failed and the arm fell '
                  f'back to eager: {state["compile_errors"]}')
            problems.append(f"{arm_id}|compile_error")

    # ── the three readings ───────────────────────────────────────────────────
    auto = _pair(by_id, f"auto_n{MULTI_COUNT}", f"auto_n{REF_COUNT}", problems,
                 "auto")
    off = _pair(by_id, f"off_n{MULTI_COUNT}", f"off_n{REF_COUNT}", problems,
                "off")
    same = _pair(by_id, f"off_n{REF_COUNT}", f"auto_n{REF_COUNT}", problems,
                 "eager-vs-compiled")

    print(f"\nTHE ABLATION -- one variable, compile_mode, everything else held")
    print(f'  auto  n={MULTI_COUNT} vs n={REF_COUNT}      {_fmt(auto):>11}   '
          f'nrmse {_fmt(auto, "nrmse")}   reproduces mg15c '
          f'({MG15C_READINGS[2][1]:.3e})')
    print(f'  off   n={MULTI_COUNT} vs n={REF_COUNT}      {_fmt(off):>11}   '
          f'nrmse {_fmt(off, "nrmse")}   THE DISCRIMINATOR')
    print(f'  off n={REF_COUNT} vs auto n={REF_COUNT}    {_fmt(same):>11}   '
          f'nrmse {_fmt(same, "nrmse")}   eager against compiled, same count '
          f'(context, no gate)')

    print("\nHOW TO READ IT")
    uneven_arms = [r.get("arm_id") for r in rows
                   if (r.get("blocks") or {}).get("uneven")]
    if not uneven_arms:
        # Without a second block length there is no dynamic dimension and the
        # ablation has nothing to turn off.  Say so rather than reporting a
        # confirmation that was never tested.
        print("  NO ARM EXERCISED AN UNEVEN SPLIT: every arm's blocks are "
              "equal on both axes, so the condition under test never arose "
              "and neither reading means anything here.")
        problems.append("ablation|no uneven arm")
    elif auto and off and auto.get("rel") is not None and off.get("rel") is not None:
        # THE POSITIVE CONTROL, checked BEFORE either reading is interpreted.
        # An ablation is only readable if its baseline reproduced the effect:
        # turning something off cannot refute a mechanism for a phenomenon that
        # was never present.  The scale to judge against is measured in this
        # same run -- the eager-against-compiled comparison at the SAME device
        # count, which changes no split and is therefore the floor a
        # split-related effect has to stand above.
        floor = (same or {}).get("rel")
        if floor and floor > 0:
            reproduced = auto["rel"] >= 10.0 * floor
        else:
            reproduced = auto["rel"] > 0
        print(f'  positive control: the compiled uneven split reads '
              f'{_fmt(auto)} against a same-count eager/compiled floor of '
              f'{_fmt(same)}.')
        if not reproduced:
            print("  -> THE EFFECT DID NOT REPRODUCE IN THIS RUN.  The "
                  "compiled arm sits at the same scale as the floor, so the "
                  "phenomenon mg15c measured is absent here and the ablation "
                  "is UNTESTED -- neither confirmed nor refuted.")
            print("     Expected on the CPU smoke, whose tiny shapes and CPU "
                  "inductor put every reading at the float32 floor.  On the "
                  "cluster this would mean the positive control failed and "
                  "nothing below it could be believed.")
            if not SMOKE:
                problems.append("ablation|baseline did not reproduce mg15c")
        else:
            ratio = (float("inf") if off["rel"] <= 0.0
                     else auto["rel"] / off["rel"])
            print(f'  compiled reads {_fmt(auto)} and eager reads {_fmt(off)} '
                  f'on the SAME uneven split ({ratio:.1f}x).')
            if ratio >= 10.0:
                print("  -> CONFIRMED.  Turning compilation off collapses the "
                      "difference while nothing about the split changed, so "
                      "the difference is produced by torch.compile's shape "
                      "specialization and not by where the blocks fall.  The "
                      "split places its data correctly; the compiled code "
                      "sums it in a different order.")
                print(f"     this is the cost {PREDICTION.split(':')[0]} "
                      f"predicted, appearing as a VALUE difference rather "
                      f"than only as the speed difference anticipated there.")
            elif ratio <= 2.0:
                print("  -> REFUTED.  Eager reproduces the difference, so it "
                      "is not compilation: the question returns to the split "
                      "itself and the torch body's own reduction over uneven "
                      "blocks.")
            else:
                print(f"  -> PARTIAL: eager reduces the difference by "
                      f"{ratio:.1f}x but does not remove it.  Compilation is "
                      f"implicated but is not the whole story; read the eager "
                      f"reading against the same-count context line above.")
    walls = {r.get("arm_id"): r.get("back_project_s") for r in rows
             if r.get("mode") == "backproject" and r.get("back_project_s")}
    if walls:
        print(f"\n  wall per projection (first call, compilation included on "
              f"the auto arms; context only, not a timing measurement):")
        for arm_id in sorted(walls):
            print(f"    {arm_id:<14} {walls[arm_id]:7.1f}s")

    # ── everything on one page ───────────────────────────────────────────────
    print(f"\n{'READING':<46}{'rel':>12}  note")
    print("  " + "-" * 74)
    print(f"  -- mg15c, {MG15C_JOB} --")
    for label, rel, note in MG15C_READINGS:
        print(f'  {label:<44} {rel:>11.3e}  {note}')
    print("  -- mg15d, this run --")
    print(f'  {"multiaxis adjoint auto n%d vs n%d" % (MULTI_COUNT, REF_COUNT):<44} '
          f'{_fmt(auto):>11}  compiled, uneven split')
    print(f'  {"multiaxis adjoint off  n%d vs n%d" % (MULTI_COUNT, REF_COUNT):<44} '
          f'{_fmt(off):>11}  EAGER, same uneven split')
    print(f'  {"multiaxis adjoint off n%d vs auto n%d" % (REF_COUNT, REF_COUNT):<44} '
          f'{_fmt(same):>11}  eager vs compiled, same count')
    print("  " + "-" * 74)

    healthy = not problems
    print(f"\n===== INSTRUMENT HEALTH: {'OK' if healthy else 'DEGRADED'} =====")
    if problems:
        print(f"  {len(problems)} problem(s): {', '.join(problems)}")
    else:
        print("  every arm ran, every comparison computed, and every arm "
              "built in the mode it claimed")
    hot = [r.get("arm_id") for r in rows if r.get("gpu_hot")]
    if hot:
        print(f"  GPU health: {len(hot)} row(s) sampled hot: {hot}")
    print(f"  exit code   {0 if healthy else 2}")
    return dict(instruments_healthy=healthy, problems=problems,
                auto=auto, off=off, eager_vs_compiled=same, walls=walls,
                multi_count=MULTI_COUNT, ref_count=REF_COUNT,
                mg15c=[list(r) for r in MG15C_READINGS], hot=hot)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.argv[2])
        try:
            out = WORKERS[cfg["mode"]](cfg)
        except Exception:                                         # noqa: BLE001
            out = dict(cfg, error=traceback.format_exc()[-3000:])
        print("__RESULT__" + json.dumps(out))
    elif "--help" in sys.argv:
        print(__doc__)
    else:
        sys.exit(main())
