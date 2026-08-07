"""kb4: attribute the kb2 sweep's e-3-class back-path value readings.

The kb2 view-chunk sweep read its kernel rows against a COMPILED torch
reference (compile_mode='auto', the production form) and three combos read
1.7-2.2e-3 where the rest read 1e-5-class: parallel back at every
multi-view-reference shape, and cone back at the full-512 shape.  The
reading was IDENTICAL across every chunk value and in the legacy-batching
row, so the batching change cannot be the cause; the candidates are the
kernel itself (diverging from the body at these shapes) and the compiled
reference (inductor reassociating the long view/tap sums per shape).

One variable at a time: for each probed shape this script computes the SAME
full-view-range driver output three ways -- the eager torch body, the
compiled torch body, and the Triton kernel -- and reads every pairwise
max-rel (plus a norm-rel and an over-1e-4 element count, which say whether a
difference is broad or a few isolated outputs).  If eager-vs-compiled
carries the e-3 and kernel-vs-eager does not, the reading was the RULER
(the compiled reference's reassociation) and the kernels are faithful to
the eager body they were validated against; if kernel-vs-eager carries it,
the kernel diverges at these shapes and the sweep flagged something real.

Shapes probed: the two loudest readings (512-parallel-back-full,
512-cone-back-full) and one clean multi-view control
(1024-cone-back-subset).

Run on a CUDA node:  <torch python> kb4_value_attrib.py
(see kb4_gautschi.sbatch; no env beyond the interpreters).
"""

import json
import os
import platform
import time

PROBES = [
    dict(cell=(512, 448, 384), geometry="parallel", pclass="full"),
    dict(cell=(512, 448, 384), geometry="cone", pclass="full"),
    dict(cell=(1024, 1008, 992), geometry="cone", pclass="subset"),
]
SUBSET_STRIDE = 64
INPUT_SEED = 0
COEFF_POWER = 1
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _build(geometry, cell, compile_mode, disable_kernels):
    import numpy as np

    os.environ["MBIRTORCH_DISABLE_TRITON"] = "1" if disable_kernels else "0"
    from mbirtorch import kernel_availability
    kernel_availability._reset_probe_cache()
    kernel_availability._reset_self_check_cache()
    import mbirtorch

    num_views, _, num_channels = cell
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles, device="cuda",
                                            compile_mode=compile_mode)
    else:
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(cell, angles,
                                        source_detector_dist=4.0 * num_channels,
                                        source_iso_dist=2.0 * num_channels,
                                        device="cuda",
                                        compile_mode=compile_mode)
    model.set_params(no_warning=True, verbose=0)
    return model


def _back_output(model, pclass):
    import torch

    import mbirtorch

    recon_shape = tuple(model.get_params("recon_shape"))
    device = model.torch_device
    pixel_indices = torch.as_tensor(mbirtorch.gen_full_indices(recon_shape),
                                    dtype=torch.int64, device=device)
    if pclass == "subset":
        pixel_indices = pixel_indices[::SUBSET_STRIDE].contiguous()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    cell = tuple(int(s) for s in model.get_params("sinogram_shape"))
    sino = torch.rand(cell, generator=generator).to(device)
    out = model.projector_functions.sparse_back_project_view_range(
        sino, pixel_indices, (0, cell[0]), coeff_power=COEFF_POWER)
    torch.cuda.synchronize()
    result = out.detach().cpu().to(dtype=torch.float64)
    del out, sino, pixel_indices
    torch.cuda.empty_cache()
    return result


def _bound_back_name(model):
    body = model.projector_functions._back_body_per_dev[0]
    return getattr(body, "__name__", str(body))


def _pair(a, b):
    import numpy as np

    a, b = a.numpy(), b.numpy()
    scale = float(np.abs(b).max())
    diff = np.abs(a - b)
    return dict(max_rel=float(diff.max()) / scale,
                norm_rel=float(np.linalg.norm(diff) / np.linalg.norm(b)),
                over_1e4=int((diff > 1e-4 * scale).sum()),
                n=int(diff.size))


def main():
    import torch

    rows = []
    for probe in PROBES:
        name = (f"{probe['cell'][0]}-{probe['geometry']}-back-"
                f"{probe['pclass']}")
        print(f"\n== {name} ==", flush=True)
        t0 = time.time()
        # Three arms, one variable apart: eager body, compiled body (the kb2
        # reference), kernel.  Each builds a fresh model; the back path has
        # no atomics, so each output is deterministic per configuration.
        eager_model = _build(probe["geometry"], probe["cell"], "off", True)
        assert "triton" not in _bound_back_name(eager_model)
        eager = _back_output(eager_model, probe["pclass"])
        del eager_model

        compiled_model = _build(probe["geometry"], probe["cell"], "auto", True)
        assert "triton" not in _bound_back_name(compiled_model)
        compiled = _back_output(compiled_model, probe["pclass"])
        del compiled_model

        kernel_model = _build(probe["geometry"], probe["cell"], "off", False)
        assert "triton" in _bound_back_name(kernel_model), \
            _bound_back_name(kernel_model)
        kernel = _back_output(kernel_model, probe["pclass"])
        del kernel_model

        row = dict(probe=name,
                   kernel_vs_eager=_pair(kernel, eager),
                   kernel_vs_compiled=_pair(kernel, compiled),
                   compiled_vs_eager=_pair(compiled, eager),
                   seconds=round(time.time() - t0, 1))
        for key in ("kernel_vs_eager", "kernel_vs_compiled",
                    "compiled_vs_eager"):
            entry = row[key]
            print(f"  {key:>20}: max_rel {entry['max_rel']:.2e}  "
                  f"norm_rel {entry['norm_rel']:.2e}  "
                  f"over_1e-4 {entry['over_1e4']}/{entry['n']}", flush=True)
        rows.append(row)
        del eager, compiled, kernel

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        RESULTS_DIR,
        f"kb4_value_attrib_{platform.node().split('.')[0]}_{stamp}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
