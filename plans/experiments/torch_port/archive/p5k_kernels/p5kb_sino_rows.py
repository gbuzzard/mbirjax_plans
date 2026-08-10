"""The decisive element-wise probe: do the two frameworks' gate-protocol
PHANTOMS and SINOGRAMS differ at the six recon-divergent slices/rows -- and is
either chain z-flipped relative to the other?

Why this exists.  Two independent reductions disagree about the phantom:

    reduction-based (p5k9, per-slice fingerprints):  slice 139 differs by 1.6,
        868 by 1.6, 239/768 by 0.8, 496/511 by 0.2 -- mirror-symmetric, and
        every recon-divergent slice covered
    element-wise (coordinator, local):  EXACTLY ZERO at {139,239,496,511,768},
        1.6 at {868} only -- not mirror-closed

They cannot both be right, and the answer decides whether a SECOND CAUSE exists:
direct recon is per-slice separable under parallel beam, so a divergent slice
whose phantom is bit-identical must get its divergence through the sinogram at
that detector row, which would be a real finding about the projector rather
than about the phantom generator.

So this job stops reducing and compares ELEMENT-WISE, the stronger method, and
does it for both arrays in one place: one framework writes its phantom and
sinogram, the other loads them beside its own and computes every difference
itself.  No cross-job artifact assumptions, no fingerprint standing in for a
diff.

Reported, per cell:
  1. PROTOCOL -- exactly how the gate builds its sinogram, printed verbatim, so
     the findings can pin it (analytic vs forward-projected, any scaling, any
     noise, any seeding).
  2. PHANTOM, element-wise per slice: sum |d|, max |d|, and the count of
     differing voxels -- separating VALUE changes from presence/absence flips,
     which are different mechanisms with the same slice index.
  3. SINOGRAM, element-wise per detector row: sum |d| and max |d|, the set of
     rows above float noise, and whether that set covers the six divergent
     rows.  This is the readout that decides second-cause-or-not.
  4. THE Z-FLIP TEST, on both arrays: the same per-row/per-slice reduction of
     |a - flip(b)|.  A chain that flipped the slice axis somewhere would
     agree better flipped than unflipped, which no amount of amplitude
     argument can fake.  Run on both because the mirror-pair structure is the
     observation that motivated it.

Run:
    <torch python> p5kb_sino_rows.py      on a CUDA node (see p5kb_gautschi.sbatch)
    python p5kb_sino_rows.py --dry-run    anywhere: print the plan
    python p5kb_sino_rows.py --help

Environment:
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the subprocesses
    P5KB_CELLS=1024                   cells (default 1024, the divergent one)
    P5KB_KEEP=1                       keep the ~8 GiB scratch arrays
    P5KB_SMOKE=1 / P5KB_DEVICE=cpu    local smoke
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

ALL_CELLS = [(512, 448, 384), (1024, 1008, 992)]
CELLS = [(1024, 1008, 992)]

SMOKE = os.environ.get("P5KB_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5KB_DEVICE", "cpu" if SMOKE else "cuda")

# The slices/rows the composed gate's direct recon diverges at (p5k8).
DIVERGENT = [139, 239, 496, 511, 768, 868]
# Above this fraction of the array max, a difference is not float noise.
NOISE_FRACTION = 1e-9

# The gate's sinogram protocol, printed verbatim into the log so the findings
# can pin it without re-reading the harness.
PROTOCOL = """  phantom  = generate_3d_shepp_logan_low_dynamic_range(recon_shape)
             -- each framework's OWN generator, deterministic, no seed, no
                target_max_attenuation or other scaling argument
  sinogram = model.forward_project(phantom)      -- FORWARD-PROJECTED, not
             analytic; no noise added anywhere; no post-scaling
  weights  = exp(-sinogram / (2 * max(sinogram)))
             -- cast to float32 in the torch arm ONLY in the original gate
                (an asymmetry; p5ka casts in both)"""

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    wanted = os.environ.get("P5KB_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in ALL_CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _path(kind, cell):
    return os.path.join(RESULTS_DIR, f"_p5kb_torch_{cell[0]}_{kind}.npy")


def _build(framework, cell):
    """(phantom, sinogram) under the gate protocol, as float32 numpy."""
    import numpy as np

    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    if framework == "jax":
        import mbirjax as mj

        model = mj.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mj.generate_3d_shepp_logan_low_dynamic_range
    else:
        import mbirtorch as mt

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mt.generate_3d_shepp_logan_low_dynamic_range
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    phantom = np.asarray(phantom_fn(recon_shape), dtype=np.float32)
    sinogram = np.asarray(model.forward_project(phantom), dtype=np.float32)
    return phantom, sinogram, recon_shape


def producer_worker(cfg):
    """The torch side: build under the gate protocol and write both arrays."""
    import numpy as np

    cell = tuple(cfg["cell"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    phantom, sinogram, recon_shape = _build("torch", cell)
    np.save(_path("phantom", cell), phantom)
    np.save(_path("sino", cell), sinogram)
    return dict(role="producer", framework="torch", cell=list(cell),
                recon_shape=list(recon_shape),
                phantom_shape=list(phantom.shape),
                sinogram_shape=list(sinogram.shape),
                phantom_checksum=float(np.sum(np.abs(phantom),
                                              dtype=np.float64)),
                sinogram_checksum=float(np.sum(np.abs(sinogram),
                                               dtype=np.float64)))


def _axis_report(diff, axis_index, label):
    """Element-wise reductions along one axis: sum |d|, max |d|, count."""
    import numpy as np

    axes = tuple(a for a in range(diff.ndim) if a != axis_index)
    return dict(label=label,
                sum_abs=np.sum(np.abs(diff), axis=axes, dtype=np.float64),
                max_abs=np.max(np.abs(diff), axis=axes),
                count=np.sum(diff != 0, axis=axes))


def comparer_worker(cfg):
    """The jax side: build its own, load torch's, and do every element-wise
    comparison here so nothing depends on a reduction computed elsewhere."""
    import numpy as np

    cell = tuple(cfg["cell"])
    phantom_j, sino_j, recon_shape = _build("jax", cell)
    result = dict(role="comparer", framework="jax", cell=list(cell),
                  recon_shape=list(recon_shape),
                  phantom_checksum=float(np.sum(np.abs(phantom_j),
                                                dtype=np.float64)),
                  sinogram_checksum=float(np.sum(np.abs(sino_j),
                                                 dtype=np.float64)))

    # ── the phantom, element-wise per SLICE (axis 2) ──────────────────────────
    phantom_t = np.load(_path("phantom", cell))
    result["phantom_shapes_match"] = (phantom_t.shape == phantom_j.shape)
    diff = phantom_t - phantom_j
    report = _axis_report(diff, 2, "phantom_per_slice")
    result["phantom_sum_abs"] = report["sum_abs"].tolist()
    result["phantom_max_abs"] = report["max_abs"].tolist()
    result["phantom_count"] = report["count"].tolist()
    # the z-flip test on the phantom
    flip_diff = phantom_t - phantom_j[:, :, ::-1]
    result["phantom_flip_sum_abs"] = np.sum(
        np.abs(flip_diff), axis=(0, 1), dtype=np.float64).tolist()
    result["phantom_total_abs"] = float(np.sum(np.abs(diff), dtype=np.float64))
    result["phantom_flip_total_abs"] = float(np.sum(np.abs(flip_diff),
                                                    dtype=np.float64))
    del phantom_t, phantom_j, diff, flip_diff

    # ── the sinogram, element-wise per DETECTOR ROW (axis 1) ──────────────────
    sino_t = np.load(_path("sino", cell))
    result["sinogram_shapes_match"] = (sino_t.shape == sino_j.shape)
    diff = sino_t - sino_j
    report = _axis_report(diff, 1, "sinogram_per_row")
    result["sino_sum_abs"] = report["sum_abs"].tolist()
    result["sino_max_abs"] = report["max_abs"].tolist()
    result["sino_count"] = report["count"].tolist()
    flip_diff = sino_t - sino_j[:, ::-1, :]
    result["sino_flip_sum_abs"] = np.sum(
        np.abs(flip_diff), axis=(0, 2), dtype=np.float64).tolist()
    result["sino_total_abs"] = float(np.sum(np.abs(diff), dtype=np.float64))
    result["sino_flip_total_abs"] = float(np.sum(np.abs(flip_diff),
                                                 dtype=np.float64))
    result["sino_abs_max"] = float(np.max(np.abs(sino_j)))
    return result


def analyze(cell, rows):
    import numpy as np

    label = "x".join(map(str, cell))
    print(f"\n=== element-wise phantom and sinogram, cell {label} ===",
          flush=True)
    comparer = next((r for r in rows if r.get("role") == "comparer"
                     and "error" not in r), None)
    producer = next((r for r in rows if r.get("role") == "producer"
                     and "error" not in r), None)
    if comparer is None or producer is None:
        print("  MISSING a side", flush=True)
        return dict(cell=list(cell), error="missing side")

    summary = dict(cell=list(cell))
    print(f"  1. gate sinogram protocol:\n{PROTOCOL}", flush=True)
    print(f"     checksums: phantom torch {producer['phantom_checksum']:.10g} "
          f"vs jax {comparer['phantom_checksum']:.10g}", flush=True)
    print(f"                sinogram torch {producer['sinogram_checksum']:.10g}"
          f" vs jax {comparer['sinogram_checksum']:.10g}", flush=True)

    # 2. the phantom
    p_sum = np.array(comparer["phantom_sum_abs"])
    p_max = np.array(comparer["phantom_max_abs"])
    p_cnt = np.array(comparer["phantom_count"])
    p_slices = sorted(int(k) for k in np.nonzero(p_sum > 0)[0])
    value_only = [k for k in p_slices if p_cnt[k] > 0 and p_max[k] < 0.999]
    summary.update(phantom_diff_slices=p_slices,
                   phantom_diff_slice_count=len(p_slices),
                   phantom_total_abs=comparer["phantom_total_abs"],
                   phantom_flip_total_abs=comparer["phantom_flip_total_abs"])
    print(f"  2. PHANTOM differs at {len(p_slices)} slices "
          f"(element-wise, |d| > 0)", flush=True)
    print(f"     at the six recon-divergent slices:", flush=True)
    print(f"     {'slice':>6} {'sum|d|':>12} {'max|d|':>10} {'voxels':>8}",
          flush=True)
    rows_out = []
    for k in DIVERGENT:
        if k < len(p_sum):
            rows_out.append(dict(slice=k, sum_abs=float(p_sum[k]),
                                 max_abs=float(p_max[k]), count=int(p_cnt[k])))
            print(f"     {k:>6} {p_sum[k]:>12.6g} {p_max[k]:>10.6g} "
                  f"{p_cnt[k]:>8d}", flush=True)
    summary["phantom_at_divergent"] = rows_out
    covered = [r["slice"] for r in rows_out if r["sum_abs"] > 0]
    summary["phantom_covers_divergent"] = (len(covered) == len(rows_out))
    print(f"     -> phantom differs at {len(covered)}/{len(rows_out)} of the "
          f"divergent slices", flush=True)

    # 3. the sinogram -- the decisive readout
    s_sum = np.array(comparer["sino_sum_abs"])
    s_max = np.array(comparer["sino_max_abs"])
    # SELF-CALIBRATING threshold.  The two forward projectors differ at float
    # level on essentially every row (measured: identical phantoms still give
    # ~1e-7 row differences), so a fixed floor would flag all of them and the
    # decisive readout would say nothing.  The median row is that float-noise
    # level; a row carrying a real phantom-sourced difference stands orders
    # above it, and the ratio is reported so the separation is visible rather
    # than asserted.
    baseline = float(np.median(s_max))
    # 10x, not 100x.  MEASURED at the 1024 cell 2026-08-07: the six
    # phantom-sourced rows sit at 223x, 112x and 28x the median, so a 100x cut
    # called the two smallest "no difference" and printed SECOND CAUSE for
    # rows whose sinogram difference was in fact 28x the noise floor AND
    # matched their phantom difference to four significant figures.  A
    # threshold able to manufacture the headline finding is the wrong
    # instrument: this cut now only screens the float-noise floor, and the
    # phantom-vs-sinogram correspondence printed below is the real test.
    noise = max(baseline * 10.0, NOISE_FRACTION * comparer["sino_abs_max"])
    s_rows = sorted(int(k) for k in np.nonzero(s_max > noise)[0])
    summary.update(sino_diff_rows=s_rows, sino_diff_row_count=len(s_rows),
                   sino_noise_threshold=noise, sino_row_baseline=baseline,
                   sino_total_abs=comparer["sino_total_abs"])
    print(f"  3. SINOGRAM: median row max|d| = {baseline:.3g} (the float-noise "
          f"level); {len(s_rows)} of {len(s_max)} rows exceed 10x that "
          f"({noise:.3g})", flush=True)
    # THE CHAIN TEST, stronger than any threshold: under parallel beam a ray
    # through the differing voxels of slice k accumulates their difference into
    # detector row k, so the row's max|d| should reproduce the slice's sum|d|.
    # Agreement here is positive evidence that the sinogram difference IS the
    # phantom difference, which no cut on magnitude alone can establish.
    print(f"     {'row':>6} {'sum|d|':>14} {'max|d|':>12} {'x median':>10}"
          f" {'phantom sum|d|':>15} {'ratio':>8}", flush=True)
    chain = []
    for k in DIVERGENT:
        if k < len(s_sum):
            predicted = float(p_sum[k]) if k < len(p_sum) else 0.0
            ratio = float(s_max[k]) / max(predicted, 1e-30)
            chain.append(dict(row=k, sino_max_abs=float(s_max[k]),
                              phantom_sum_abs=predicted, ratio=ratio))
            print(f"     {k:>6} {s_sum[k]:>14.6g} {s_max[k]:>12.6g} "
                  f"{s_max[k] / max(baseline, 1e-30):>10.1f} "
                  f"{predicted:>15.6g} {ratio:>8.4f}", flush=True)
    summary["chain_test"] = chain
    live = [c for c in chain if c["phantom_sum_abs"] > 0]
    if live:
        worst = max(abs(c["ratio"] - 1.0) for c in live)
        summary["chain_worst_deviation"] = worst
        print(f"     chain test: sinogram row max|d| reproduces the phantom "
              f"slice sum|d| to within {worst:.2%} at all {len(live)} rows"
              if worst < 0.05 else
              f"     chain test: WORST deviation {worst:.1%} -- the sinogram "
              f"difference does not simply carry the phantom difference",
              flush=True)
    # Only rows this cell actually HAS can be evaluated: at a smaller cell the
    # divergent indices fall outside the array, and an empty "missing" list
    # would otherwise read as full coverage -- a verdict the data never earned.
    in_range = [k for k in DIVERGENT if k < len(s_max)]
    missing = [k for k in in_range if s_max[k] <= noise]
    summary["divergent_rows_evaluated"] = in_range
    summary["divergent_rows_without_sino_diff"] = missing
    summary["sino_covers_divergent"] = (bool(in_range) and not missing)
    if not in_range:
        print(f"     -> not applicable at this cell: the divergent rows "
              f"{DIVERGENT} lie outside its {len(s_max)} rows", flush=True)
    elif not missing:
        print(f"     -> the sinogram differs at ALL {len(in_range)} divergent "
              f"rows: the divergence enters through the sinogram, NO second "
              f"cause", flush=True)
    else:
        print(f"     *** SECOND CAUSE: rows {missing} diverge in the recon "
              f"with NO sinogram difference -- the back-projection/filter "
              f"path differs at those rows ***", flush=True)

    # 4. the z-flip test
    for name, plain, flipped in (
            ("phantom", comparer["phantom_total_abs"],
             comparer["phantom_flip_total_abs"]),
            ("sinogram", comparer["sino_total_abs"],
             comparer["sino_flip_total_abs"])):
        ratio = flipped / max(plain, 1e-30)
        summary[f"{name}_flip_ratio"] = ratio
        # Two arrays that already agree exactly give 0/0: there is no
        # disagreement to explain, so neither answer is available.
        if plain <= 0.0 and flipped <= 0.0:
            verdict = "not applicable: the two arrays are identical"
        elif ratio < 0.5:
            verdict = "A Z-FLIP IS PRESENT"
        else:
            verdict = "no z-flip (flipped agrees no better, as it must not)"
        print(f"  4. {name} z-flip test: total |a - b| {plain:.6g} vs "
              f"|a - flip(b)| {flipped:.6g}  ratio {ratio:.3g} -> {verdict}",
              flush=True)

    # The mirror symmetry of the phantom difference itself.
    mirror_pairs = []
    n = len(p_sum)
    for k in DIVERGENT:
        partner = n - 1 - k
        if k < n and partner < n:
            mirror_pairs.append(dict(
                slice=k, partner=int(partner), sum_abs=float(p_sum[k]),
                partner_sum_abs=float(p_sum[partner]),
                symmetric=bool(abs(p_sum[k] - p_sum[partner])
                               <= 1e-6 * max(p_sum[k], 1e-30))))
    summary["phantom_mirror_check"] = mirror_pairs
    # Only pairs with a difference to be symmetric ABOUT say anything.
    live = [m for m in mirror_pairs
            if max(m["sum_abs"], m["partner_sum_abs"]) > 0]
    symmetric = bool(live) and all(m["symmetric"] for m in live)
    summary["phantom_diff_mirror_symmetric"] = symmetric
    summary["phantom_mirror_pairs_live"] = len(live)
    if not live:
        text = "not applicable: no phantom difference at these slices"
    elif symmetric:
        text = "SYMMETRIC (explains the mirror pairs with no z-flip)"
    else:
        text = "NOT symmetric"
    print(f"  5. phantom difference mirror symmetry at the divergent slices "
          f"({len(live)} with a difference): {text}", flush=True)
    return summary


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5kb.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5kb.json")
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
    print(f"protocol printed into the log:\n{PROTOCOL}")
    print(f"divergent slices/rows under test: {DIVERGENT}")
    print(f"noise floor: {NOISE_FRACTION:.0e} of the array max")
    print(f"results:    {RESULTS_DIR}/p5kb_sino_rows_{RUN_LABEL}.json")


def main():
    cells = selected_cells()
    print(f"p5kb element-wise phantom/sinogram probe on {platform.node()} "
          f"({DEVICE}{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}", flush=True)
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", protocol=PROTOCOL,
                       divergent=DIVERGENT, rows=[], summaries=[])
    for cell in cells:
        label = "x".join(map(str, cell))
        print(f"\n{label}/producer (torch) ...", flush=True)
        rows = [run_one(TORCH_PYTHON, dict(role="producer", cell=list(cell)))]
        if "error" in rows[0]:
            print(f"  FAILED: {rows[0]['error'][:200]}", flush=True)
        else:
            print(f"  wrote phantom + sinogram", flush=True)
            print(f"{label}/comparer (jax) ...", flush=True)
            rows.append(run_one(JAX_PYTHON, dict(role="comparer",
                                                 cell=list(cell))))
            if "error" in rows[-1]:
                print(f"  FAILED: {rows[-1]['error'][:200]}", flush=True)
        summary = analyze(cell, rows)
        # The per-slice/per-row vectors are the evidence; keep them out of the
        # printed table but IN the json.
        all_results["rows"].extend(rows)
        all_results["summaries"].append(summary)
        if os.environ.get("P5KB_KEEP", "0") != "1":
            for kind in ("phantom", "sino"):
                try:
                    os.remove(_path(kind, cell))
                except OSError:
                    pass

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5kb_sino_rows_{RUN_LABEL}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=1)
    print(f"\nwrote {out}", flush=True)


def _run_worker(cfg_path, out_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        if cfg["role"] == "producer":
            result = producer_worker(cfg)
        else:
            result = comparer_worker(cfg)
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
