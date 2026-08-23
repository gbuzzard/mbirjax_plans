"""mg58 -- what the recon setup spends on validation, and what the device
would spend on the same check.

mg57 measured initialize_recon at 2.6 to 2.7 seconds warm at the parallel
(1024, 1008, 992) cell, unchanged between one device and four, of which the
pixel partitions are 0.4 to 0.5 seconds.  Host-side work is what does not
change with device count, and initialize_recon's host-side work is the input
validation: it runs np.isfinite over the whole sinogram and over the whole
weights array before anything reaches a device.  This probe times those steps
directly, and times the same check on a device beside them, so the remainder
is attributed rather than inferred.

Nothing here is a remedy and nothing is changed: every call is a public one on
arrays this script owns.
"""
import json
import os
import time

import numpy as np
import torch

import mbirtorch

CELL = (1024, 1008, 992)
SEED = 13
OUT = os.environ.get("MG58_OUT", "mg58_probe.json")


def clock(fn):
    start = time.perf_counter()
    value = fn()
    return time.perf_counter() - start, value


def main():
    rows = {"cell": list(CELL), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0)}

    angles = np.linspace(0, np.pi, CELL[0], endpoint=False)
    model = mbirtorch.ParallelBeamModel(CELL, angles)
    model.skip_memory_preflight = True
    model.configure_devices(devices=['cuda:0'])
    model.set_params(no_warning=True, verbose=0)

    # Host arrays of the shapes recon() is handed.  The values do not matter to
    # a finiteness scan or to a transfer, and building them is not timed.
    rng = np.random.RandomState(SEED)
    sino = rng.rand(*CELL).astype(np.float32)
    weights = np.ones(CELL, dtype=np.float32)
    rows["array_gib"] = sino.nbytes / 2 ** 30

    # The two host scans initialize_recon runs today, each timed alone and
    # repeated, because the first pass over a fresh array also pays its page
    # faults and the later passes are what a warm process sees.
    for name, array in (("sinogram", sino), ("weights", weights)):
        passes = []
        for _ in range(3):
            seconds, ok = clock(lambda: bool(np.isfinite(array).all()))
            passes.append(seconds)
        rows[f"host_isfinite_{name}_s"] = passes
        rows[f"host_isfinite_{name}_median_s"] = float(np.median(passes))

    # The same statement, made where the data is going anyway.  The transfer is
    # timed separately from the check, because recon() moves the sinogram to
    # the device regardless: only the CHECK is the cost in question.
    transfer_s, sino_dev = clock(
        lambda: torch.as_tensor(sino, device='cuda:0'))
    torch.cuda.synchronize()
    rows["host_to_device_transfer_s"] = transfer_s

    device_passes = []
    for _ in range(3):
        start = time.perf_counter()
        ok = bool(torch.isfinite(sino_dev).all())
        torch.cuda.synchronize()
        device_passes.append(time.perf_counter() - start)
    rows["device_isfinite_s"] = device_passes
    rows["device_isfinite_median_s"] = float(np.median(device_passes))
    sino_dev = None
    torch.cuda.empty_cache()

    # The regularization parameters, the other host-side step of the same
    # phase.
    seconds, _ = clock(
        lambda: model.auto_set_regularization_params(sino, weights=weights))
    rows["auto_set_regularization_params_s"] = seconds

    # The pixel partitions, for the record beside them: the default
    # granularity list has eleven levels and a three-iteration run visits
    # three of them.  Both numbers are read from the model rather than assumed.
    from mbirtorch import vcd_utils
    recon_shape, granularity, use_ror_mask = model.get_params(
        ['recon_shape', 'granularity', 'use_ror_mask'])
    rows["granularity"] = list(granularity)
    sequence = vcd_utils.gen_partition_sequence(
        model.get_params('partition_sequence'), max_iterations=3)
    rows["levels_visited_by_three_iterations"] = sorted(
        {int(granularity[i]) for i in sequence})
    seconds, partitions = clock(
        lambda: vcd_utils.gen_set_of_pixel_partitions(
            recon_shape, granularity, device=model.torch_device,
            use_ror_mask=use_ror_mask))
    rows["gen_set_of_pixel_partitions_all_levels_s"] = seconds
    rows["num_partitions_built"] = len(partitions)
    partitions = None

    visited = sorted({int(granularity[i]) for i in sequence})
    seconds, _ = clock(
        lambda: vcd_utils.gen_set_of_pixel_partitions(
            recon_shape, visited, device=model.torch_device,
            use_ror_mask=use_ror_mask))
    rows["gen_set_of_pixel_partitions_visited_only_s"] = seconds

    print(json.dumps(rows, indent=2))
    with open(OUT, "w") as sink:
        json.dump(rows, sink, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
