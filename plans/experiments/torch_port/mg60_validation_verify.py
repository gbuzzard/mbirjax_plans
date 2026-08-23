"""mg60 -- one verification of the cheaper input validation.

Findings section 1.48 measured initialize_recon at 2.61 s warm at the
parallel (1024, 1008, 992) cell, unchanged between one device and four, and
mg58 attributed most of it to host-side scans: np.isfinite over the whole
sinogram and, for the weights, isfinite and a negative test and an all-zero
test, one full pass and one full-length temporary each.  The library now
reads a minimum and a maximum through torch instead, which answers all of
those questions from two reductions on the array's own device.

This script times the phase after that change, times both formulations side
by side on the same arrays, and checks that the two agree on what they
accept and reject.  It changes nothing.
"""
import json, os, time, warnings
import numpy as np, torch, mbirtorch
from mbirtorch.tomography_model import _array_extremes

CELL = (1024, 1008, 992)
OUT = os.environ.get("MG60_OUT", "mg60_verify.json")


def clock(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def old_sino_check(a):
    return bool(np.isfinite(a).all())


def old_weight_checks(a):
    return (bool(np.isfinite(a).all()), bool((a < 0).any()), bool((a == 0).all()))


def new_checks(a):
    low, high = _array_extremes(a)
    import math
    return (math.isfinite(low) and math.isfinite(high), low < 0,
            low == 0 and high == 0)


def main():
    rows = {"cell": list(CELL), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "torch_threads": torch.get_num_threads()}

    angles = np.linspace(0, np.pi, CELL[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(CELL, angles)
    model.skip_memory_preflight = True
    model.configure_devices(devices=['cuda:0'])
    model.set_params(no_warning=True, verbose=0)

    rng = np.random.RandomState(13)
    sino = rng.rand(*CELL).astype(np.float32)
    weights = np.ones(CELL, dtype=np.float32)
    rows["array_gib"] = sino.nbytes / 2 ** 30

    # The two formulations, on the same arrays, three passes each.
    for label, fn in (("old_sinogram", lambda: old_sino_check(sino)),
                      ("old_weights", lambda: old_weight_checks(weights)),
                      ("new_sinogram", lambda: new_checks(sino)),
                      ("new_weights", lambda: new_checks(weights))):
        passes = [clock(fn)[0] for _ in range(3)]
        rows[f"{label}_s"] = passes
        rows[f"{label}_median_s"] = float(np.median(passes))

    # The phase as the reconstruction calls it, which is what section 1.48
    # measured at 2.61 s before the change.
    passes = [clock(lambda: model.initialize_recon(sino, weights=weights))[0]
              for _ in range(3)]
    rows["initialize_recon_s"] = passes
    rows["initialize_recon_median_s"] = float(np.median(passes))

    # The two formulations must agree on what they accept and reject, on
    # arrays small enough to hold several copies.
    small = (4, 8, 8)
    agree = []
    base = np.ones(small, dtype=np.float32)
    for name, arr in (("clean", base.copy()),
                      ("nan", np.where(np.arange(base.size).reshape(small) == 3,
                                       np.nan, base).astype(np.float32)),
                      ("posinf", np.where(np.arange(base.size).reshape(small) == 3,
                                          np.inf, base).astype(np.float32)),
                      ("neginf", np.where(np.arange(base.size).reshape(small) == 3,
                                          -np.inf, base).astype(np.float32)),
                      ("negative", np.where(np.arange(base.size).reshape(small) == 3,
                                            -1.0, base).astype(np.float32)),
                      ("all_zero", np.zeros(small, dtype=np.float32)),
                      ("part_zero", np.where(np.arange(base.size).reshape(small) < 3,
                                             0.0, base).astype(np.float32))):
        old = old_weight_checks(arr)
        new = new_checks(arr)
        # The old finite test reports True for finite; the new one likewise.
        agree.append({"case": name, "old": [bool(v) for v in old],
                      "new": [bool(v) for v in new],
                      "same": [bool(o) == bool(n) for o, n in zip(old, new)]})
    rows["agreement"] = agree
    rows["all_cases_agree"] = all(all(c["same"]) for c in agree)

    print(json.dumps(rows, indent=2))
    with open(OUT, "w") as sink:
        json.dump(rows, sink, indent=2)
    return 0 if rows["all_cases_agree"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
