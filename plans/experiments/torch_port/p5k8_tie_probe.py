"""Job C: is the torch-vs-jax direct_recon divergence a ROUNDING-TIE class?

Where this sits: p5k7 attributed the parallel gate's 1024 value gap to the
init, not to the VCD iteration and not to any Triton kernel.  Recomputing on
the saved samples then showed the gap is not a field-wide error at all -- norm
rel 1.1e-3, max rel 0.375, carried by ~24 isolated voxels -- and the two
frameworks' fbp_filter agrees to 2-3e-6 at both channel counts, so the filter
is exonerated and the back-projection step of direct_recon is the remaining
suspect.  The hypothesis under test: voxels whose projected channel center
lands on a half-integer boundary take a different tap in the two frameworks
(floor(x+0.5) vs round-half-even), the same class as the cone kernel's
documented rounding carve-out.

NO TIMING.  Three measurements, and a fourth the hypothesis itself demands:

  1. THE DIVERGENCE, on direct_recon alone under the gate's phantom protocol:
     norm rel, max rel, the count of voxels over 1 / 5 / 20 percent of max, and
     the (row, col, slice) coordinates of the top 10 -- reported in BOTH the
     sampled-row index of the artifact and the full-volume row it maps to, so
     the coordinates are unambiguous.
  2. THE TIE FINGERPRINT: for each top voxel's pixel, the torch hfan projected
     centre n_p across ALL views, reduced to the minimum distance of its
     fractional part from 0.5.  A tie voxel sits within ~1e-3 of 0.5 for at
     least one view.  Ten non-divergent voxels are measured the same way as the
     control -- without them a small distance means nothing, since some voxel
     is always closest.
  3. PARAMETER PARITY: both frameworks' geometry parameters at both cells, so
     "the two libraries disagree about the geometry" is excluded by
     measurement rather than by assumption.
  4. CYLINDER COHERENCE -- the hypothesis's own prediction.  Under parallel
     beam the projected channel centre depends on (row, col) and the view
     angle ALONE: the slice axis is inert (slice k is detector row k).  So a
     channel-tie voxel cannot be isolated -- every slice of that (row, col)
     cylinder must take the same wrong tap.  For each top voxel this reports
     how many of its cylinder's slices diverge.  A whole cylinder confirms the
     channel-tie class; a genuinely isolated voxel REFUTES it and moves the
     suspect to the row/slice path, whatever the fractional parts say.

Run:
    <torch python> p5k8_tie_probe.py      on a CUDA node (see p5k8_gautschi.sbatch)
    python p5k8_tie_probe.py --dry-run    anywhere: print the measurement plan
    python p5k8_tie_probe.py --help

Environment (export from the SUBMITTING SHELL, never in an --export list):
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the subprocesses
    P5K8_CELLS=512,1024               run a subset of the cells
    P5K8_TOP=10                       how many divergent voxels to fingerprint
    P5K8_SMOKE=1                      tiny cell on P5K8_DEVICE (default cpu)
    P5K8_DEVICE=cuda|cpu|mps          the torch device
"""

import json
import os
import platform
import subprocess
import sys
import traceback

# ── CONFIG ────────────────────────────────────────────────────────────────────
JAX_PYTHON = os.environ.get(
    "P0_JAX_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python")
TORCH_PYTHON = os.environ.get(
    "P0_TORCH_PYTHON", "/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python")

CELLS = [(512, 448, 384), (1024, 1008, 992)]
FRAMEWORKS = ("torch", "jax")

SMOKE = os.environ.get("P5K8_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K8_DEVICE", "cpu" if SMOKE else "cuda")

TOP_VOXELS = int(os.environ.get("P5K8_TOP", "10"))
CONTROL_VOXELS = 10
CONTROL_SEED = 5
SAMPLE_ROWS = 16          # the p5k6/p5k7 artifact convention, kept so these
                          # numbers explain THAT gate's value line
# A tie shows as a fractional part within this of 0.5.
TIE_TOLERANCE = 1e-3
# Divergence thresholds, as fractions of the reference's max.
THRESHOLDS = (0.01, 0.05, 0.20)

# The geometry parameters compared for parity.  Names absent from a framework
# are reported as n/a rather than raising -- the two libraries need not expose
# identical parameter sets for the comparison to be meaningful.
PARITY_PARAMS = ("recon_shape", "sinogram_shape", "delta_voxel",
                 "delta_det_channel", "delta_det_row", "det_channel_offset",
                 "det_row_offset", "recon_slice_offset", "voxel_row_aspect",
                 "voxel_slice_aspect", "use_ror_mask", "magnification")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    wanted = os.environ.get("P5K8_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _artifact(framework, cell):
    return os.path.join(RESULTS_DIR, f"_p5k8_{framework}_{cell[0]}_direct.npy")


def worker(cfg):
    """One framework's direct_recon at one cell, under the gate's phantom
    protocol, plus its geometry parameters."""
    import numpy as np

    framework, cell = cfg["framework"], tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if framework == "jax":
        import jax
        import mbirjax as mj

        model = mj.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mj.generate_3d_shepp_logan_low_dynamic_range
        version = f"jax {jax.__version__}"
        psf_radius = None
    else:
        import torch

        import mbirtorch as mt

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mt.generate_3d_shepp_logan_low_dynamic_range
        version = f"torch {torch.__version__}"
        psf_radius = int(model.get_psf_radius())

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    params = {}
    for name in PARITY_PARAMS:
        try:
            value = model.get_params(name)
        except Exception:                                         # noqa: BLE001
            value = "n/a"
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        params[name] = value
    if psf_radius is not None:
        params["psf_radius"] = psf_radius

    phantom = phantom_fn(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    del phantom
    direct = np.asarray(model.direct_recon(sinogram))
    step = max(1, recon_shape[0] // SAMPLE_ROWS)
    np.save(_artifact(framework, cell), direct[::step])
    return dict(framework=framework, cell=list(cell), version=version,
                recon_shape=list(recon_shape), params=params,
                sample_step=step, device=DEVICE,
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)),
                direct_checksum=float(np.sum(np.abs(direct),
                                             dtype=np.float64)))


def _tie_distances(cell, pixels, num_cols):
    """For each (row, col) pixel, the minimum over ALL views of
    |frac(n_p) - 0.5| -- how close the torch hfan projected channel centre
    comes to a half-integer boundary -- with the view that attains it.

    The torch chain is the one measured because it is the one that can be
    changed; the question is whether these voxels sit on a tie at all, and a
    tie is a property of the geometry both frameworks share.
    """
    import numpy as np
    import torch

    import mbirtorch as mt
    from mbirtorch.parallel_beam import _parallel_hfan_math

    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    model = mt.ParallelBeamModel(cell, angles, device="cpu")
    model.set_params(no_warning=True, verbose=0)
    args = model._view_batch_args()
    flat = torch.as_tensor([int(r) * num_cols + int(c) for r, c in pixels],
                           dtype=torch.int64)
    view_params = torch.as_tensor(model.get_params("angles"),
                                  dtype=torch.float32)
    n_p, centers, w_p_c, weight_scale = _parallel_hfan_math(
        flat, view_params, args["num_rows"], args["num_cols"],
        args["num_channels"], args["delta_det_channel"],
        args["det_channel_offset"], args["delta_voxel"],
        args["delta_voxel_row"])
    frac = n_p - torch.floor(n_p)                       # (views, pixels)
    distance = torch.abs(frac - 0.5)
    best = torch.min(distance, dim=0)
    return [dict(min_distance_from_half=float(best.values[i]),
                 at_view=int(best.indices[i]),
                 n_p_there=float(n_p[best.indices[i], i]))
            for i in range(len(pixels))]


def analyze(cell, rows):
    """The whole readout for one cell (see the module docstring)."""
    import numpy as np

    by_fw = {r.get("framework"): r for r in rows if "error" not in r}
    label = "x".join(map(str, cell))
    print(f"\n=== direct_recon divergence, cell {label} ===", flush=True)
    if len(by_fw) < 2:
        print(f"  MISSING framework rows: "
              f"{[f for f in FRAMEWORKS if f not in by_fw]}", flush=True)
        return dict(cell=list(cell), error="missing framework rows")

    torch_row, jax_row = by_fw["torch"], by_fw["jax"]
    summary = dict(cell=list(cell), sample_step=torch_row["sample_step"])

    # 3. parameter parity first: it decides whether anything below is even
    # comparable.
    mismatches = {}
    for name in PARITY_PARAMS:
        t_val, j_val = (torch_row["params"].get(name),
                        jax_row["params"].get(name))
        if t_val != j_val and "n/a" not in (t_val, j_val):
            mismatches[name] = [t_val, j_val]
    summary["param_mismatches"] = mismatches
    summary["torch_params"] = torch_row["params"]
    summary["jax_params"] = jax_row["params"]
    print(f"  3. parameter parity: "
          f"{'ALL MATCH' if not mismatches else 'MISMATCHES ' + json.dumps(mismatches)}",
          flush=True)
    print(f"     torch psf_radius {torch_row['params'].get('psf_radius')}, "
          f"recon_shape {torch_row['recon_shape']} / {jax_row['recon_shape']}",
          flush=True)

    a = np.load(_artifact("torch", cell))
    b = np.load(_artifact("jax", cell))
    if a.shape != b.shape:
        summary["error"] = f"sample shapes differ: {a.shape} vs {b.shape}"
        print(f"  SHAPES DIFFER: {a.shape} vs {b.shape}", flush=True)
        return summary

    diff = np.abs(a - b)
    ref_max = max(float(np.max(np.abs(b))), 1e-30)
    norm_rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
    max_rel = float(np.max(diff) / ref_max)
    counts = {f"over_{int(t * 100)}pct": int(np.sum(diff > t * ref_max))
              for t in THRESHOLDS}
    summary.update(norm_rel=norm_rel, max_rel=max_rel, counts=counts,
                   sample_shape=list(a.shape), ref_max=ref_max,
                   sample_voxels=int(a.size))
    print(f"  1. norm rel {norm_rel:.3e}   max rel {max_rel:.3e}   "
          f"counts {counts} of {a.size} sampled voxels", flush=True)

    step = torch_row["sample_step"]
    num_cols = torch_row["recon_shape"][1]
    flat_order = np.argsort(diff, axis=None)[::-1][:TOP_VOXELS]
    top = []
    for flat_index in flat_order:
        i, j, k = np.unravel_index(int(flat_index), diff.shape)
        # 4. cylinder coherence: how much of this (row, col) cylinder diverges.
        cylinder = diff[i, j, :]
        top.append(dict(
            sample_row=int(i), full_row=int(i) * step, col=int(j),
            slice=int(k), torch=float(a[i, j, k]), jax=float(b[i, j, k]),
            abs_diff=float(diff[i, j, k]),
            rel_of_max=float(diff[i, j, k] / ref_max),
            cylinder_slices_over_1pct=int(np.sum(cylinder > 0.01 * ref_max)),
            cylinder_len=int(cylinder.size)))
    summary["top_voxels"] = top
    print(f"  1b. top {len(top)} voxels (full_row, col, slice): "
          f"torch vs jax, and how many of the (row,col) CYLINDER's slices "
          f"also diverge", flush=True)
    for entry in top:
        print(f"      ({entry['full_row']:>5}, {entry['col']:>4}, "
              f"{entry['slice']:>4})  torch {entry['torch']:+.5f}  jax "
              f"{entry['jax']:+.5f}  rel {entry['rel_of_max']:.3f}   "
              f"cylinder {entry['cylinder_slices_over_1pct']:>4}/"
              f"{entry['cylinder_len']}", flush=True)
    coherent = [e["cylinder_slices_over_1pct"] for e in top]
    if coherent:
        summary["cylinder_median_over_1pct"] = float(np.median(coherent))
        whole = sum(1 for c in coherent if c > 0.5 * top[0]["cylinder_len"])
        summary["top_voxels_in_whole_cylinders"] = whole
        print(f"  4. cylinder coherence: {whole} of {len(top)} top voxels sit "
              f"in cylinders whose majority of slices diverge (median "
              f"{np.median(coherent):.0f} of {top[0]['cylinder_len']} slices)",
              flush=True)

    # 2. the tie fingerprint, on the divergent pixels and on a control set.
    divergent_pixels = [(e["full_row"], e["col"]) for e in top]
    rng = np.random.default_rng(CONTROL_SEED)
    quiet = np.argwhere(diff <= 1e-9 * ref_max)
    if len(quiet) >= CONTROL_VOXELS:
        picks = rng.choice(len(quiet), CONTROL_VOXELS, replace=False)
        control_pixels = [(int(quiet[p][0]) * step, int(quiet[p][1]))
                          for p in picks]
    else:
        control_pixels = []
    try:
        summary["tie_divergent"] = _tie_distances(cell, divergent_pixels,
                                                  num_cols)
        summary["tie_control"] = (_tie_distances(cell, control_pixels, num_cols)
                                  if control_pixels else [])
    except Exception as e:                                        # noqa: BLE001
        summary["tie_error"] = f"{type(e).__name__}: {e}"[:300]
        print(f"  2. tie fingerprint FAILED: {summary['tie_error']}",
              flush=True)
        return summary

    div_d = [t["min_distance_from_half"] for t in summary["tie_divergent"]]
    ctl_d = [t["min_distance_from_half"] for t in summary["tie_control"]]
    summary["tie_divergent_max_distance"] = max(div_d) if div_d else None
    summary["tie_control_min_distance"] = min(ctl_d) if ctl_d else None
    ties = sum(1 for d in div_d if d <= TIE_TOLERANCE)
    ctl_ties = sum(1 for d in ctl_d if d <= TIE_TOLERANCE)
    summary["tie_divergent_count"] = ties
    summary["tie_control_count"] = ctl_ties
    print(f"  2. tie fingerprint (min |frac(n_p) - 0.5| over all views):",
          flush=True)
    print(f"     divergent voxels: {[f'{d:.2e}' for d in div_d]}", flush=True)
    print(f"     control voxels:   {[f'{d:.2e}' for d in ctl_d]}", flush=True)
    print(f"     within {TIE_TOLERANCE:.0e} of a tie: {ties}/{len(div_d)} "
          f"divergent vs {ctl_ties}/{len(ctl_d)} control", flush=True)
    if div_d and ctl_d:
        if max_rel <= THRESHOLDS[0]:
            # The "top" voxels here are float noise, so the fingerprint has no
            # divergence to explain: a control, not a verdict.
            verdict = (f"no divergence at this cell (max rel {max_rel:.1e} <= "
                       f"{THRESHOLDS[0]:.0%}); fingerprint is a control only")
        elif ties >= max(1, len(div_d) // 2) and ctl_ties == 0:
            verdict = "TIE FINGERPRINT CONFIRMED"
        elif ties == 0:
            verdict = ("TIE FINGERPRINT ABSENT: no divergent voxel sits near a "
                       "half-integer channel boundary")
        else:
            verdict = "TIE FINGERPRINT PARTIAL"
        summary["tie_verdict"] = verdict
        print(f"     -> {verdict}", flush=True)
    return summary


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k8.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k8.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    if os.path.exists(out_path):
        os.remove(out_path)
    proc = subprocess.run([python, os.path.abspath(__file__), "_worker",
                           cfg_path, out_path])
    if proc.returncode != 0 and not os.path.exists(out_path):
        row = dict(error=f"worker exited {proc.returncode}", **cfg)
    else:
        with open(out_path) as f:
            row = json.load(f)
    return row


def print_plan():
    print(f"cells:      {[('x'.join(map(str, c))) for c in selected_cells()]}")
    print(f"frameworks: {list(FRAMEWORKS)}  (torch on {DEVICE})")
    print(f"measures:   direct_recon only, no timing; top {TOP_VOXELS} voxels "
          f"fingerprinted against {CONTROL_VOXELS} controls")
    print(f"tie window: |frac(n_p) - 0.5| <= {TIE_TOLERANCE:.0e}")
    print(f"results:    {RESULTS_DIR}/p5k8_tie_probe_{RUN_LABEL}.json")


def main():
    cells = selected_cells()
    print(f"p5k8 tie probe on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, direct_recon only, "
          f"no timing", flush=True)
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", tie_tolerance=TIE_TOLERANCE,
                       thresholds=list(THRESHOLDS), rows=[], summaries=[])
    for cell in cells:
        rows = []
        for framework in FRAMEWORKS:
            python = JAX_PYTHON if framework == "jax" else TORCH_PYTHON
            print(f"\n{'x'.join(map(str, cell))}/{framework} ...", flush=True)
            row = run_one(python, dict(framework=framework, cell=list(cell)))
            rows.append(row)
            if "error" in row:
                print(f"  FAILED: {row['error'][:200]}", flush=True)
            else:
                print(f"  direct checksum {row['direct_checksum']:.8g}",
                      flush=True)
        summary = analyze(cell, rows)
        all_results["rows"].extend(rows)
        all_results["summaries"].append(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5k8_tie_probe_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


def _run_worker(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        result = worker(cfg)
    except Exception as e:                                        # noqa: BLE001
        traceback.print_exc()
        result = dict(error=f"{type(e).__name__}: {e}"[:400], **cfg)
    with open(out_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _run_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--dry-run":
        print_plan()
    elif len(sys.argv) >= 2:
        print(f"unknown argument {sys.argv[1]!r}; try --help")
        sys.exit(2)
    else:
        main()
