"""mg50 -- HOW MUCH OF THE DENOISER'S MULTI-DEVICE SLOWDOWN IS THE OUTPUT
GATHER, AND WHAT DOES A DENOISE COST WHEN NOBODY GATHERS?

WHY THIS RUN EXISTS.  The component split (mg49, job 15402884) attributed a
1024-class denoise at two devices and found that the sharded sweep itself --
the loop everyone has been arguing about -- is 175 ms of a 3,401 ms call.
Setup accounts for another 1,165 ms.  That leaves about 2,060 ms, sixty
percent of the call, inside no region the probe wrapped.  The largest
unwrapped candidate by far is the OUTPUT GATHER: ``denoise`` returns
``self._gather_recon(denoised)`` unless the caller asks for the device form,
and on a sharded model that call pulls every shard to host and then
concatenates them on the LAST axis, which is the worst axis for memory
locality.  A one-device denoise pays a single contiguous device-to-host copy
instead, with no concatenate at all.

That is a hypothesis from a residual, not a measurement, and this run turns
it into one.  It also answers the question the hypothesis raises, which
matters more than the accounting: a plug-and-play or ADMM loop keeps the
volume on the devices between steps and never gathers, so what a denoise
costs THERE is not what the floors protocol measures.

THE MEASUREMENT is a single-variable ablation.  Every arm runs the floors
protocol exactly -- same model construction, same seed, same noisy phantom,
same three iterations, same cold pass and three warm repeats -- and varies
ONE thing: whether ``denoise`` is asked for the host form or the device
form.

    output_sharded=False   what the floors rows measure: the sweep plus the
                           gather back to a host numpy array.
    output_sharded=True    what an ADMM loop pays: the same reconstruction,
                           left where it was computed.

The difference between the two walls at one device count is that count's
gather cost.  The ratio of the one-device wall to the wider one, computed
inside each mode, is the speed verdict for that mode.  Both are printed.

WHAT WOULD MAKE THIS RUN WRONG, and how each is guarded.  If the device-form
arm simply deferred work rather than avoiding it, its wall would be
meaningless; every arm therefore synchronizes every one of its devices
before stopping the clock, so no arm can hide unfinished work in the timer.
If the two modes reconstructed different volumes the comparison would be
empty; every arm records a checksum, computed on device in the sharded mode
so that reading it does not perform the very gather under test, and the
report compares them.  And if a mode changed the amount of arithmetic the
verdict would not be about gathering; it does not -- the two paths differ
only in the final return.

WHAT THIS RUN DOES NOT DO.  It wraps nothing and edits no library file.  It
is walls only, because a wall is the whole question here.

Run:
    <torch python> mg50_denoiser_gather.py        on a 4-GPU node
    MG50_DRY=1 <python> mg50_denoiser_gather.py   print the plan and stop
    MG50_SMOKE=1 <python> mg50_denoiser_gather.py tiny, CPU, seconds

Configuration is by environment variable only.  Export from the SUBMITTING
SHELL, never through an sbatch --export list, which slurm splits on commas.
    MG50_RESULTS=<dir>    where the jsonl goes
    MG50_SINO_DIR=<dir>   where the staged noisy images are read from;
                          defaults to the floors runs' directory, whose
                          files this run reuses so its walls compare with
                          the recorded ones
    MG50_ARMS=a,b         a subset, by arm id
"""

import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG50_SMOKE", "0") == "1"
DRY = os.environ.get("MG50_DRY", "0") == "1"
DEVICE = "cpu" if SMOKE else "cuda"

#: The protocol, the floors refresh's.  Changing any of these would measure a
#: different amount of work than the recorded walls describe.
SEED = 13
DENOISE_SIGMA = 0.1
ITERATIONS = 1 if SMOKE else 3
WARM_REPEATS = 1 if SMOKE else 3

#: The cells.  For a denoiser the cell IS the image shape.  The 1024-class is
#: the cell mg49 attributed, so it is where the hypothesis was formed; the
#: 1664-class is the largest size at which the speed question has any
#: practical effect, and it says whether the gather's share grows with volume.
CELL_1024 = (1024, 1008, 992)
CELL_1664 = (1664, 1648, 1632)
CELLS = (CELL_1024, CELL_1664)
SMOKE_CELL = (8, 24, 20)

#: The device counts, and the two output modes that are the single variable.
COUNTS = (1, 2, 4)
SMOKE_MAX_DEVICES = 2
OUTPUT_MODES = (False, True)          # output_sharded

#: Warm walls measured 2026-08-20 by mg49 (job 15402884) at output_sharded
#: False, read from that job's log.  Quoted so the report can print this
#: run's host-form walls beside them; nothing gates on them.
RECORDED_SOURCE = ("mg49, job 15402884 (the denoiser ladder and component "
                   "split; measured 2026-08-20, read from the job log)")
RECORDED_HOST_FORM_WALLS = {
    CELL_1024: {1: 2.209, 2: 3.451, 4: 3.718},
    CELL_1664: {1: 8.784, 2: 13.380, 4: 14.157},
}

RESULTS_DIR = os.environ.get(
    "MG50_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
#: Where the staged noisy images live.  The default is the floors runs'
#: directory: these arms denoise the same bytes the recorded walls were
#: measured on, and every one of these files already exists there.
SINO_DIR = os.environ.get(
    "MG50_SINO_DIR",
    RESULTS_DIR if SMOKE
    else "/scratch/gautschi/buzzard/torch_p3/results/mg48_floors")

RUN_LABEL = platform.node().split(".")[0]
ARM_COL = 22
# ──────────────────────────────────────────────────────────────────────────────


def cell_for(cell):
    """The cell an arm actually runs at.  The smoke collapses both cells onto
    one tiny stand-in, so the arm ids stay the same either way."""
    return SMOKE_CELL if SMOKE else cell


def arm_id(cell, n_dev, sharded):
    return "d{}_n{}_{}".format(cell[0], n_dev, "dev" if sharded else "host")


def all_arms():
    """Every arm, in RUN order: the cheap cell first, and within a cell the
    two output modes side by side at each count, so a defect in one mode
    surfaces beside its own control rather than an hour later."""
    arms = []
    for cell in CELLS:
        for n_dev in COUNTS:
            if SMOKE and n_dev > SMOKE_MAX_DEVICES:
                continue
            for sharded in OUTPUT_MODES:
                arms.append(dict(
                    kind="arm", arm=arm_id(cell, n_dev, sharded),
                    job_id=arm_id(cell, n_dev, sharded),
                    cell=list(cell_for(cell)), declared_cell=list(cell),
                    n_dev=n_dev, output_sharded=sharded,
                    iterations=ITERATIONS, warm_repeats=WARM_REPEATS))
    return arms


def all_arm_ids():
    return [cfg["arm"] for cfg in all_arms()]


def _strict_subset(env_name, allowed):
    """Refuse garbage: every token must name a member of ``allowed``.  A
    silently ignored token would shrink the run without saying so."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return list(allowed)
    chosen = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token not in allowed:
            raise ValueError("{}: {!r} is not an arm of this run.  The valid "
                             "ids are: {}".format(env_name, token,
                                                  ", ".join(allowed)))
        if token not in chosen:
            chosen.append(token)
    return [name for name in allowed if name in chosen]


def _input_path(cell):
    """One file per cell, under the shared input directory, named as the
    floors refresh names it."""
    return os.path.join(SINO_DIR,
                        "_sino_denoiser_{}x{}x{}.npy".format(*cell))


def build_model(cell, n_dev):
    """The model the floors refresh times, built its way.

    The denoiser family is placed EXPLICITLY rather than through the
    environment pin: that is the protocol its recorded rows were measured
    under, and an explicit layout is never second-guessed by the device
    policy.  Every row records the count it realized, so a placement that
    did not take is visible rather than assumed.
    """
    import mbirtorch

    model = mbirtorch.QGGMRFDenoiser(tuple(cell))
    if DEVICE == "cuda":
        model.configure_devices(
            devices=["cuda:{}".format(i) for i in range(n_dev)])
    else:
        model.configure_devices(devices=["cpu"] * n_dev)
    model.set_params(no_warning=True, verbose=0)
    return model


def _staged(path, recon_shape):
    """The staged noisy image, or one built here by the floors tool's recipe.

    The files this run wants already exist, so this normally only loads.  The
    builder is kept for the smoke, whose tiny cell has no staged file, and it
    draws its noise a slab of rows at a time from one seeded generator, which
    consumes the same stream a whole-volume draw would.
    """
    import numpy as np

    import mbirtorch

    if os.path.exists(path):
        return np.load(path), False
    phantom = mbirtorch.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
    if float(np.max(phantom)) == 0.0:
        # A volume only a few voxels deep can miss every ellipsoid, leaving a
        # phantom of zeros; a seeded uniform volume stands in, as in the
        # floors tool.
        phantom = np.asarray(np.random.RandomState(SEED).rand(*recon_shape),
                             dtype=np.float32)
    generator = np.random.RandomState(SEED)
    staged = np.empty(tuple(int(s) for s in recon_shape), dtype=np.float32)
    for start in range(0, int(recon_shape[0])):
        staged[start] = np.asarray(
            phantom[start] + DENOISE_SIGMA * generator.randn(
                int(recon_shape[1]), int(recon_shape[2])),
            dtype=np.float32)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, staged)
    return staged, True


def _checksum(out):
    """The sum of absolute values, computed WITHOUT gathering.

    The sharded arm must not perform the very transfer this run is pricing
    just to describe its own result, so a device-form output is reduced on
    its own devices and only the per-shard scalars come to the host.  Read
    after the clock has stopped, so it is outside every timed number.
    """
    import numpy as np
    import torch

    if hasattr(out, "tensors"):
        return float(sum(float(torch.sum(torch.abs(t))) for t in out.tensors))
    if torch.is_tensor(out):
        return float(torch.sum(torch.abs(out)))
    return float(np.sum(np.abs(out), dtype=np.float64))


def run_arm(cfg):
    """One arm: a cold pass discarded, then the warm repeats, varying only
    whether the result is asked for in host form or device form."""
    import numpy as np
    import torch

    cell = tuple(cfg["cell"])
    n_dev = int(cfg["n_dev"])
    sharded = bool(cfg["output_sharded"])
    cuda = DEVICE == "cuda" and torch.cuda.is_available()

    result = dict(cfg, framework="torch", version="torch " + torch.__version__,
                  device=DEVICE, cuda=cuda, seed=SEED,
                  device_name=(torch.cuda.get_device_name(0) if cuda
                               else DEVICE),
                  visible_devices=(torch.cuda.device_count() if cuda else 0),
                  env_calibration=os.environ.get(
                      "MBIRTORCH_MEMORY_CALIBRATION"))
    result["invalid_reasons"] = []
    if result["env_calibration"] not in (None, "", "0"):
        result["invalid_reasons"].append(
            "the memory calibration mode is on; it does extra work at the "
            "settle and at the end of every call, so this arm did not time "
            "what the others timed")

    model = build_model(cell, n_dev)
    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    result["recon_shape"] = list(recon_shape)
    path = _input_path(cell)
    result["input_path"] = path
    staged, built = _staged(path, recon_shape)
    result["input_built_here"] = built

    def one():
        np.random.seed(SEED)
        out, _info = model.denoise(staged, sigma_noise=DENOISE_SIGMA,
                                   max_iterations=ITERATIONS,
                                   stop_threshold_change_pct=0.0,
                                   print_logs=False,
                                   output_sharded=sharded)
        if DEVICE == "cuda":
            # Every device, so no arm can leave work in flight and call it a
            # faster wall.  The host-form arm has already synchronized itself
            # by copying to host; this makes both modes stop the clock on the
            # same condition.
            for device in model.recon_placement.devices:
                torch.cuda.synchronize(device)
        return out

    started = time.perf_counter()
    out = one()
    result["cold_s"] = time.perf_counter() - started
    warm = []
    for _ in range(WARM_REPEATS):
        started = time.perf_counter()
        out = one()
        warm.append(time.perf_counter() - started)
    result["warm_all"] = warm
    result["warm_s"] = statistics.median(warm)
    result["spread"] = ((max(warm) - min(warm)) / result["warm_s"]
                        if result["warm_s"] else None)

    # After the clock: describing the result must not enter any timed number.
    result["output_form"] = type(out).__name__
    result["output_is_host"] = isinstance(out, np.ndarray)
    result["checksum"] = _checksum(out)
    if result["output_is_host"] == sharded:
        result["invalid_reasons"].append(
            "this arm asked for output_sharded={} and got a {}, so it did "
            "not measure the mode it claims".format(sharded,
                                                    type(out).__name__))

    realized = [str(d) for d in model.recon_placement.devices]
    result["realized_devices"] = realized
    result["realized_n_devices"] = len(realized)
    result["devices_ok"] = (len(realized) == n_dev)
    if not result["devices_ok"]:
        result["invalid_reasons"].append(
            "configured for {} device(s) and realized {}: {}".format(
                n_dev, len(realized), realized))
    return result


def job_env(cfg):
    """The environment that DEFINES a job, set explicitly so nothing is
    inherited from the submitting shell.  The count is realized by an
    explicit device list, so the process-wide pin is popped and never set:
    two mechanisms claiming the same decision is how a run measures a count
    it did not ask for."""
    env = dict(os.environ)
    env.pop("MBIRTORCH_NUM_DEVICES", None)
    env.pop("MBIRTORCH_MEMORY_CALIBRATION", None)
    env.pop("MBIRTORCH_WIDENING_GUARD", None)
    env["MBIRTORCH_DISABLE_TRITON"] = "0"
    return env


def spawn(cfg):
    """Run one arm in a NEW interpreter.

    A fresh process per arm is not tidiness: compiled bodies are cached at
    module level for the life of a process and the allocator keeps its pools,
    so a second arm in the same interpreter would inherit the first arm's
    state and time something else.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_mg50_cfg_{}.json".format(cfg["job_id"]))
    out_path = os.path.join(RESULTS_DIR, "_mg50_out_{}.json".format(cfg["job_id"]))
    with open(cfg_path, "w") as handle:
        json.dump(cfg, handle)
    if os.path.exists(out_path):
        os.remove(out_path)
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__),
                           "--worker", cfg_path, out_path], env=job_env(cfg))
    wall = time.perf_counter() - started
    if not os.path.exists(out_path):
        # A job that ran out of device memory lands here.  That is a reading,
        # not a harness fault, so it is recorded and the run goes on.
        row = dict(cfg, error="worker exited {} and wrote no row".format(
            proc.returncode))
    else:
        with open(out_path) as handle:
            row = json.load(handle)
    row["subprocess_wall_s"] = wall
    return row


def build_plan():
    keep = _strict_subset("MG50_ARMS", all_arm_ids())
    plan = [cfg for cfg in all_arms() if cfg["arm"] in keep]
    if not plan:
        raise ValueError("MG50_ARMS selects no arm")
    return plan


def print_plan(plan):
    print("mg50 the denoiser's output gather: {} arm(s), device {}".format(
        len(plan), DEVICE))
    print("  jsonl -> {}".format(RESULTS_DIR))
    print("  staged images read from -> {}".format(SINO_DIR))
    print("  protocol: seed {} reset before every call, {} iteration(s), one "
          "cold pass then {} warm repeat(s), every device synchronized before "
          "the clock stops -- the floors refresh's protocol, so these walls "
          "compare with its recorded ones".format(SEED, ITERATIONS,
                                                  WARM_REPEATS))
    print("  the single variable is output_sharded: 'host' asks for a numpy "
          "array and pays the gather, 'dev' leaves the result on the devices "
          "and is what a plug-and-play loop pays")
    header = "  {:<{}}{:>5}{:>8}{:>22}".format("arm", ARM_COL, "n", "output",
                                               "cell")
    print(header)
    for cfg in plan:
        print("  {:<{}}{:>5}{:>8}{:>22}".format(
            cfg["arm"], ARM_COL, cfg["n_dev"],
            "device" if cfg["output_sharded"] else "host",
            str(tuple(cfg["cell"]))))
    print("  nothing is wrapped and no library file is edited: this run is "
          "walls only")


def _fmt(value, width=9, prec=3):
    if value is None:
        return "{:>{}}".format("-", width)
    return "{:>{}.{}f}".format(value, width, prec)


def summarize(rows, plan, out_path):
    """The blocks a person reads, and the instrument health the exit code
    comes from.  Which mode is faster, how large the gather is and what it
    means for a plug-and-play loop are all FINDINGS: printed, never gated."""
    print("\n===== mg50 the denoiser's output gather ({}) =====".format(out_path))
    broken, findings = [], []
    by_arm = {}
    for row in rows:
        if row.get("error"):
            print("  {:<{}}  ERROR: {}".format(
                row.get("job_id", "?"), ARM_COL,
                str(row["error"]).splitlines()[-1][:100]))
            broken.append("{}|error".format(row.get("job_id", "?")))
            continue
        by_arm[row["arm"]] = row
        for reason in row.get("invalid_reasons") or []:
            broken.append("{}|{}".format(row["arm"], reason))
    for cfg in plan:
        if cfg["arm"] not in by_arm and not any(
                item.startswith(cfg["arm"] + "|") for item in broken):
            broken.append("{}|no row".format(cfg["arm"]))

    print("\n===== the walls, by output mode =====")
    print("Warm medians in seconds.  'host form' is what the floors protocol "
          "measures: the denoise plus the gather back to a host numpy array.  "
          "'device form' is what a plug-and-play or ADMM loop pays: the same "
          "reconstruction, left on the devices.  The gather column is the "
          "difference between them, which is that count's cost of coming "
          "home.")
    print("  The recorded column is from {}.".format(RECORDED_SOURCE))
    header = ("  {:<22}{:>4}{:>11}{:>13}{:>11}{:>11}".format(
        "cell", "n", "host form", "device form", "gather", "recorded"))
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cell in CELLS:
        for n_dev in COUNTS:
            host = by_arm.get(arm_id(cell, n_dev, False))
            dev = by_arm.get(arm_id(cell, n_dev, True))
            if host is None and dev is None:
                continue
            host_s = host.get("warm_s") if host else None
            dev_s = dev.get("warm_s") if dev else None
            gather = (host_s - dev_s) if (host_s and dev_s) else None
            recorded = (RECORDED_HOST_FORM_WALLS.get(tuple(cell)) or {}).get(n_dev)
            print("  {:<22}{:>4}{}{}{}{}".format(
                str(tuple(cell)), n_dev, _fmt(host_s, 11), _fmt(dev_s, 13),
                _fmt(gather, 11), _fmt(recorded, 11)))

    print("\n===== the speed verdict, computed INSIDE each mode =====")
    print("The ratio is warm(n=1) over warm(n), so above 1.00 means the wider "
          "count is faster.  Both denoiser floor rows are read against one "
          "device, so that is what these compare.  The two columns are the "
          "same reconstruction judged by two different definitions of when it "
          "is finished.")
    header = "  {:<22}{:>4}{:>16}{:>16}".format("cell", "n", "host form",
                                                "device form")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cell in CELLS:
        base_host = by_arm.get(arm_id(cell, 1, False))
        base_dev = by_arm.get(arm_id(cell, 1, True))
        for n_dev in COUNTS:
            if n_dev == 1:
                continue
            host = by_arm.get(arm_id(cell, n_dev, False))
            dev = by_arm.get(arm_id(cell, n_dev, True))
            r_host = (base_host["warm_s"] / host["warm_s"]
                      if base_host and host and host.get("warm_s") else None)
            r_dev = (base_dev["warm_s"] / dev["warm_s"]
                     if base_dev and dev and dev.get("warm_s") else None)
            print("  {:<22}{:>4}{}{}".format(
                str(tuple(cell)), n_dev, _fmt(r_host, 16), _fmt(r_dev, 16)))
            if r_host is not None and r_dev is not None and r_dev > 1.0 >= r_host:
                findings.append(
                    "{} at n={}: the wider count LOSES in host form ({:.3f}x) "
                    "and WINS in device form ({:.3f}x), so the verdict depends "
                    "on whether the caller gathers".format(
                        tuple(cell), n_dev, r_host, r_dev))

    print("\n===== the reconstructions are the same in both modes =====")
    print("A checksum computed on device for the sharded arms, so reading it "
          "does not perform the transfer under test.  Small differences "
          "across device counts are float32 reordering; a large one would be "
          "a finding about the library.")
    for cell in CELLS:
        sums = []
        for n_dev in COUNTS:
            for sharded in OUTPUT_MODES:
                row = by_arm.get(arm_id(cell, n_dev, sharded))
                if row and row.get("checksum"):
                    sums.append((arm_id(cell, n_dev, sharded), row["checksum"]))
        if not sums:
            continue
        values = [value for _name, value in sums]
        spread = ((max(values) - min(values)) / abs(max(values))
                  if max(values) else 0.0)
        print("  {:<22} {} arm(s), relative spread {:.2e}".format(
            str(tuple(cell)), len(sums), spread))
        if spread > 1e-3:
            findings.append("{}: the checksums across arms spread {:.2e}, "
                            "which is far more than float32 reordering "
                            "explains".format(tuple(cell), spread))

    print("\n-- instrument health --")
    if broken:
        for item in broken:
            print("  BROKEN {}".format(item))
    else:
        print("  every planned arm ran, realized its configured device count, "
              "returned the output form it asked for, and ran with the "
              "calibration mode off")
    for item in findings:
        print("  finding (not gated) {}".format(item))
    if not findings:
        print("  no findings")
    return dict(healthy=not broken, broken=broken, findings=findings,
                arms={name: dict(n_dev=row.get("n_dev"),
                                 output_sharded=row.get("output_sharded"),
                                 warm_s=row.get("warm_s"),
                                 checksum=row.get("checksum"))
                      for name, row in by_arm.items()})


def main():
    plan = build_plan()
    if DRY:
        print_plan(plan)
        return 0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR,
                            "mg50_gather_{}_{}.jsonl".format(RUN_LABEL, stamp))
    print_plan(plan)
    print("\nrunning -> {}".format(out_path), flush=True)
    started = time.time()
    rows = []
    with open(out_path, "w") as sink:
        sink.write(json.dumps(dict(
            row="run_header", script="mg50_denoiser_gather.py", node=RUN_LABEL,
            stamp=stamp, device=DEVICE, smoke=SMOKE, python=sys.executable,
            results_dir=RESULTS_DIR, sino_dir=SINO_DIR, seed=SEED,
            iterations=ITERATIONS, warm_repeats=WARM_REPEATS,
            recorded_source=RECORDED_SOURCE,
            plan=[dict(c) for c in plan])) + "\n")
        sink.flush()
        for index, cfg in enumerate(plan):
            print("\n  [{}/{}] {}".format(index + 1, len(plan), cfg["job_id"]),
                  flush=True)
            row = spawn(cfg)
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if row.get("error"):
                print("    ERROR: {}".format(str(row["error"])[:300]), flush=True)
            else:
                print("    cold {:.2f}s  warm {:.3f}s  spread {:.1%}  "
                      "{} device(s)  {} output".format(
                          row.get("cold_s", 0), row.get("warm_s", 0),
                          row.get("spread") or 0,
                          row.get("realized_n_devices", "-"),
                          row.get("output_form", "?")), flush=True)
        summary = summarize(rows, plan, out_path)
        summary["elapsed_min"] = (time.time() - started) / 60.0
        sink.write(json.dumps(dict(row="summary", **summary)) + "\n")
    print("\nwrote {}".format(out_path))
    print("elapsed {:.1f} min".format(summary["elapsed_min"]))
    return 0 if summary["healthy"] else 2


def _worker_main(cfg_path, out_path):
    with open(cfg_path) as handle:
        cfg = json.load(handle)
    try:
        row = run_arm(cfg)
    except Exception:                                             # noqa: BLE001
        row = dict(cfg, error=traceback.format_exc()[-3000:])
    with open(out_path, "w") as handle:
        json.dump(row, handle)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
