"""The linkage check: are the direct-recon divergent SLICES a subset of the
PHANTOM-diff slices?

Where this sits.  p5k7 attributed the parallel gate's 1024 value gap to the
init; p5k8 then refuted the channel-tie hypothesis on its own prediction --
under parallel beam a channel tie must corrupt the whole (row, col) cylinder,
and the measured cylinders diverge at a median of 2 slices out of 1008, with
the tie fingerprint firing on 7 of 10 CONTROL voxels as well (at 992 channels
almost every voxel is near a half-integer, so the metric does not
discriminate).  The remaining explanation is upstream of both: the two
frameworks' PHANTOM GENERATORS differ at f32-vs-f64 ellipsoid boundaries (the
documented divergence at the top of mbirtorch/utilities.py), and the gate had
each framework project its own phantom.

Direct recon is per-slice separable under parallel beam -- detector row k feeds
recon slice k and nothing else -- so that explanation makes a falsifiable
prediction:

    every slice where direct_recon diverges must be a slice where the
    PHANTOMS differ.

A divergent slice with no phantom difference would mean a SECOND cause, and
this script says so loudly rather than reporting a subset ratio.

The phantom comparison is exhaustive, not sampled: each framework reduces its
FULL phantom to three per-slice fingerprints -- sum |p|, the nonzero count, and
a row-position-weighted sum -- so a single flipped voxel anywhere in a slice
moves at least one of them, while nothing the size of a volume crosses the
filesystem.  (A sampled row comparison can miss a flip whose row is not in the
sample, which is a way to under-count the phantom-diff slices and manufacture
an apparent second cause.)

The divergent-slice set is read from the p5k8 artifacts, so this runs AFTER
that job and re-measures nothing it already measured.

Run:
    <torch python> p5k9_phantom_link.py     on a CUDA node (see p5k9_gautschi.sbatch)
    python p5k9_phantom_link.py --dry-run   anywhere: print the plan
    python p5k9_phantom_link.py --help

Environment:
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the subprocesses
    P5K9_CELLS=1024                   cells to check (default: both)
    P5K9_THRESHOLD=0.01               divergence threshold, fraction of max
    P5K9_SMOKE=1 / P5K9_DEVICE=cpu    local smoke
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

SMOKE = os.environ.get("P5K9_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K9_DEVICE", "cpu" if SMOKE else "cuda")
THRESHOLD = float(os.environ.get("P5K9_THRESHOLD", "0.01"))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    wanted = os.environ.get("P5K9_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _phantom_artifact(framework, cell):
    return os.path.join(RESULTS_DIR,
                        f"_p5k9_{framework}_{cell[0]}_slicefp.npy")


def _direct_artifact(framework, cell):
    """The p5k8 direct_recon sample -- this job reads, never rewrites it."""
    return os.path.join(RESULTS_DIR, f"_p5k8_{framework}_{cell[0]}_direct.npy")


def worker(cfg):
    """One framework's phantom at one cell, reduced to per-slice fingerprints.

    Three reductions, because one is not enough: the absolute sum misses a pair
    of flips that cancel, the nonzero count misses a change of value that keeps
    a voxel nonzero, and the row-weighted sum misses neither.
    """
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
    else:
        import torch

        import mbirtorch as mt

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        phantom_fn = mt.generate_3d_shepp_logan_low_dynamic_range
        version = f"torch {torch.__version__}"

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    phantom = np.asarray(phantom_fn(recon_shape), dtype=np.float64)
    rows = np.arange(1, phantom.shape[0] + 1, dtype=np.float64)[:, None, None]
    fingerprints = np.stack([
        np.sum(np.abs(phantom), axis=(0, 1)),
        np.count_nonzero(phantom, axis=(0, 1)).astype(np.float64),
        np.sum(np.abs(phantom) * rows, axis=(0, 1)),
    ])
    np.save(_phantom_artifact(framework, cell), fingerprints)
    return dict(framework=framework, cell=list(cell), version=version,
                recon_shape=list(recon_shape),
                phantom_checksum=float(np.sum(np.abs(phantom))),
                num_slices=int(phantom.shape[2]))


def _mirror_pairs(slices, num_slices):
    """Slices that pair as k <-> num_slices - 1 - k.  The Shepp-Logan volume is
    symmetric about its central slice, so a generator difference at an
    ellipsoid boundary should appear at both members of a pair; the pairing is
    therefore evidence about the CAUSE, not decoration."""
    remaining = set(slices)
    pairs, singles = [], []
    for k in sorted(slices):
        partner = num_slices - 1 - k
        if k in remaining:
            if partner in remaining and partner != k:
                pairs.append([k, partner])
                remaining.discard(k)
                remaining.discard(partner)
            else:
                singles.append(k)
                remaining.discard(k)
    return pairs, singles


def analyze(cell, rows):
    import numpy as np

    by_fw = {r.get("framework"): r for r in rows if "error" not in r}
    label = "x".join(map(str, cell))
    print(f"\n=== phantom linkage, cell {label} ===", flush=True)
    summary = dict(cell=list(cell))
    if len(by_fw) < 2:
        summary["error"] = "missing framework rows"
        print(f"  MISSING framework rows", flush=True)
        return summary

    # 1. the phantom-diff slices, from the exhaustive per-slice fingerprints.
    fp_t = np.load(_phantom_artifact("torch", cell))
    fp_j = np.load(_phantom_artifact("jax", cell))
    per_slice_diff = np.max(np.abs(fp_t - fp_j), axis=0)
    phantom_slices = sorted(int(k) for k in np.nonzero(per_slice_diff > 0)[0])
    num_slices = int(fp_t.shape[1])
    summary["num_slices"] = num_slices
    summary["phantom_diff_slices"] = phantom_slices
    summary["phantom_diff_slice_count"] = len(phantom_slices)
    summary["phantom_checksum_torch"] = by_fw["torch"]["phantom_checksum"]
    summary["phantom_checksum_jax"] = by_fw["jax"]["phantom_checksum"]
    print(f"  phantom checksums: torch {by_fw['torch']['phantom_checksum']:.8g}"
          f"  jax {by_fw['jax']['phantom_checksum']:.8g}", flush=True)
    print(f"  phantom differs at {len(phantom_slices)} of {num_slices} slices",
          flush=True)

    # 2. the direct-recon divergent slices, from the p5k8 artifacts.
    path_t, path_j = (_direct_artifact("torch", cell),
                      _direct_artifact("jax", cell))
    if not (os.path.exists(path_t) and os.path.exists(path_j)):
        summary["error"] = "p5k8 direct_recon artifacts missing; run p5k8 first"
        print(f"  MISSING p5k8 artifacts -- run p5k8_tie_probe.py first",
              flush=True)
        return summary
    a, b = np.load(path_t), np.load(path_j)
    diff = np.abs(a - b)
    ref_max = max(float(np.max(np.abs(b))), 1e-30)
    direct_slices = sorted(int(k) for k in
                           np.nonzero(np.max(diff, axis=(0, 1))
                                      > THRESHOLD * ref_max)[0])
    summary["direct_diff_slices"] = direct_slices
    summary["direct_threshold"] = THRESHOLD
    print(f"  direct_recon diverges (> {THRESHOLD:.0%} of max) at "
          f"{len(direct_slices)} slices: {direct_slices}", flush=True)

    # 3. THE LINKAGE.
    missing = [k for k in direct_slices if k not in set(phantom_slices)]
    summary["divergent_not_in_phantom"] = missing
    summary["linkage_holds"] = (len(missing) == 0)
    if direct_slices and not missing:
        print(f"  LINKAGE HOLDS: every divergent slice is a phantom-diff "
              f"slice ({len(direct_slices)}/{len(direct_slices)})", flush=True)
    elif not direct_slices:
        print(f"  no divergent slices at this cell -- nothing to link",
              flush=True)
    else:
        print(f"  *** SECOND CAUSE: {len(missing)} divergent slice(s) have NO "
              f"phantom difference: {missing} ***", flush=True)

    pairs, singles = _mirror_pairs(direct_slices, num_slices)
    summary["divergent_mirror_pairs"] = pairs
    summary["divergent_unpaired"] = singles
    if direct_slices:
        print(f"  mirror structure (k <-> {num_slices - 1} - k): {len(pairs)} "
              f"pair(s) {pairs}, {len(singles)} unpaired {singles}", flush=True)
    p_pairs, p_singles = _mirror_pairs(phantom_slices, num_slices)
    summary["phantom_mirror_pairs"] = len(p_pairs)
    summary["phantom_unpaired"] = p_singles
    print(f"  phantom-diff slices: {len(p_pairs)} mirror pair(s), "
          f"{len(p_singles)} unpaired", flush=True)
    return summary


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k9.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k9.json")
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
    print(f"measures:   exhaustive per-slice phantom fingerprints, then the "
          f"subset check against the p5k8 direct_recon divergent slices")
    print(f"threshold:  {THRESHOLD:.0%} of max")
    print(f"reads:      {RESULTS_DIR}/_p5k8_<fw>_<cell>_direct.npy")
    print(f"results:    {RESULTS_DIR}/p5k9_phantom_link_{RUN_LABEL}.json")


def main():
    cells = selected_cells()
    print(f"p5k9 phantom linkage on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}", flush=True)
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", threshold=THRESHOLD,
                       rows=[], summaries=[])
    for cell in cells:
        rows = []
        for framework in FRAMEWORKS:
            python = JAX_PYTHON if framework == "jax" else TORCH_PYTHON
            print(f"\n{'x'.join(map(str, cell))}/{framework} phantom ...",
                  flush=True)
            row = run_one(python, dict(framework=framework, cell=list(cell)))
            rows.append(row)
            if "error" in row:
                print(f"  FAILED: {row['error'][:200]}", flush=True)
            else:
                print(f"  phantom checksum {row['phantom_checksum']:.8g}",
                      flush=True)
        summary = analyze(cell, rows)
        all_results["rows"].extend(rows)
        all_results["summaries"].append(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5k9_phantom_link_{RUN_LABEL}.json")
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
