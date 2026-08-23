"""mg61 -- do the six compiled update frames earn their compile?

mg59 named every frame torch compiles during a parallel-beam reconstruction:
six small frames, all in the VCD subset updater (the qGGMRF gradient and
Hessian, the update apply, the update direction, the prior line terms, and
two line-search terms).  The projection bodies are not among them, because
parallel beam runs hand-written kernels.  Those six cost 4.15 s of tracing
per process at one device and 7.74 s at four, and mg59 showed the cost is
not the graph count: making the shapes dynamic removed variants without
removing seconds.

So the question this run asks is whether compiling those frames is worth
what it costs.  The recorded justification for compiling is a measured
chain-level win, but it was measured on projection chains, and on this
geometry the projections do not go through the compiler at all.  Each arm
reconstructs twice in a fresh process, once with the shipped setting and
once with compilation off, and reports what each costs cold and warm.

It changes nothing and decides nothing.
"""
import json, os, sys, time
import numpy as np, torch

STAGED = os.environ.get(
    "MG61_STAGED",
    "/scratch/gautschi/buzzard/torch_p3/results/mg57_cold_start/"
    "mg57_stage_parallel_1024x1008x992.npz")
OUT = os.environ.get("MG61_OUT", "mg61_compile_worth.json")
# The library's own default is 15 iterations, and that is the workload a
# policy about compiling has to be judged against: the compile is paid once
# per process while what it buys is spent once per iteration, so a short run
# charges the full cost against a fraction of the benefit.
ITERATIONS = int(os.environ.get("MG61_ITERATIONS", "15"))
SEED = 13


def dynamo_seconds():
    """The outermost compile phase torch reports, in seconds, or None.  Its
    phases nest, so the outer one is taken rather than a sum."""
    try:
        from torch._dynamo.utils import compile_times
        rows = compile_times(repr='csv', aggregate=True)
        best = None
        for line in str(rows).splitlines():
            parts = line.split(',')
            if len(parts) >= 2 and parts[0].strip() == 'entire_frame_compile':
                try:
                    best = float(parts[1])
                except ValueError:
                    pass
        return best
    except Exception as exc:                                      # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def unique_graphs():
    try:
        from torch._dynamo.utils import counters
        return int(counters['stats'].get('unique_graphs', 0))
    except Exception:                                             # noqa: BLE001
        return None


def fingerprint(volume):
    flat = torch.as_tensor(np.asarray(volume)).reshape(-1).to(torch.float64)
    return float(flat.abs().sum()), float((flat * flat).sum())


def run_arm(devices, compile_mode):
    import mbirtorch
    with np.load(STAGED) as data:
        sino = data['sinogram']
        angles = data['angles']
    weights = np.ones_like(sino)
    model = mbirtorch.ParallelBeamModel(sino.shape, angles,
                                        compile_mode=compile_mode)
    model.skip_memory_preflight = True
    model.configure_devices(devices=devices)
    model.set_params(no_warning=True, verbose=0)
    row = {"devices": devices, "compile_mode": compile_mode,
           "cell": list(sino.shape), "iterations": ITERATIONS}
    walls = []
    for index in range(2):
        np.random.seed(SEED)
        start = time.perf_counter()
        volume, _ = model.recon(sino, weights=weights,
                                max_iterations=ITERATIONS,
                                stop_threshold_change_pct=0.0)
        for device in model.sino_placement.devices:
            torch.cuda.synchronize(device)
        walls.append(time.perf_counter() - start)
        if index == 1:
            row["fingerprint"] = fingerprint(volume)
        volume = None
    row["cold_s"], row["warm_s"] = walls[0], walls[1]
    row["dynamo_entire_frame_compile_s"] = dynamo_seconds()
    row["unique_graphs"] = unique_graphs()
    row["peak_gib"] = [float(torch.cuda.max_memory_allocated(d)) / 2 ** 30
                       for d in model.sino_placement.devices]
    return row


def main():
    if os.environ.get("MG61_CHILD"):
        devices = os.environ["MG61_DEVICES"].split(",")
        row = run_arm(devices, os.environ["MG61_MODE"])
        with open(os.environ["MG61_CHILD_OUT"], "w") as sink:
            json.dump(row, sink)
        return 0

    import subprocess
    arms = []
    for devices in (["cuda:0"], ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]):
        for mode in ("auto", "off"):
            child_out = f"/tmp/mg61_{ITERATIONS}_{len(devices)}_{mode}.json"
            env = dict(os.environ, MG61_CHILD="1", MG61_MODE=mode,
                       MG61_DEVICES=",".join(devices),
                       MG61_CHILD_OUT=child_out)
            print(f"  arm {len(devices)} device(s), compile_mode={mode}",
                  flush=True)
            proc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)],
                                  env=env, timeout=2400)
            if proc.returncode != 0 or not os.path.exists(child_out):
                arms.append({"devices": devices, "compile_mode": mode,
                             "error": f"child exited {proc.returncode}"})
                continue
            with open(child_out) as source:
                arms.append(json.load(source))

    print(f"\n### does compiling the six update frames earn its cost? ({ITERATIONS} iterations)")
    print("| devices | compile | iterations | cold s | warm s | graphs |")
    print("|---|---|---|---|---|---|")
    for row in arms:
        if "error" in row:
            print(f"| {len(row['devices'])} | {row['compile_mode']} | "
                  f"{row['error']} | | | |")
            continue
        print(f"| {len(row['devices'])} | {row['compile_mode']} | "
              f"{row['iterations']} | {row['cold_s']:.2f} | "
              f"{row['warm_s']:.2f} | {row['unique_graphs']} |")
    for count in (1, 4):
        pair = {r["compile_mode"]: r for r in arms
                if "error" not in r and len(r["devices"]) == count}
        if len(pair) == 2:
            fa, fo = pair["auto"]["fingerprint"], pair["off"]["fingerprint"]
            rel = [abs(a - b) / max(abs(a), 1e-30) for a, b in zip(fa, fo)]
            print(f"  {count} device(s): value difference between the two "
                  f"settings, relative: {rel[0]:.3e} and {rel[1]:.3e}")
    with open(OUT, "w") as sink:
        json.dump(arms, sink, indent=2)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
