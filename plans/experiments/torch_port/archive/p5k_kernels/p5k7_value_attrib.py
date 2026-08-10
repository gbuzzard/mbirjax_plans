"""Attribution of the torch-vs-jax RECON VALUE divergence at the parallel 1024
cell -- the single-variable ablation behind the p5k6 gate's value line.

The signature to explain (p5k6_pgate_h004.json): the two frameworks' sinogram
checksums agree to 1.9e-7, the torch arms agree among themselves to float
noise, torch-vs-jax recon samples agree to 5.5e-4 at the 512 cell, and diverge
by rel 0.375 at 1024 -- in the PURE-BODY arm as much as in the kernel arm, so
the cause is upstream of every Triton kernel.  Cone at 1024 passes the same
protocol at 9.7e-5, so it is not the protocol either.

NO TIMING HERE.  Every number is a value, and every comparison varies exactly
one thing.  Four measurements, in the order that narrows fastest:

  1. THE PARTITION SEQUENCE each framework actually uses -- the granularity
     table, the partition sequence, and the resulting subset count per
     iteration for iterations 0-2 -- and, beyond the counts, whether the
     PARTITIONS THEMSELVES match: same seed, same generator signature, so an
     index-for-index comparison says whether the two frameworks update the same
     voxels in the same order.  (The counts were already checked locally on the
     host and agree at both cells: granularity [1,2,4,...,128], sequence
     [2,4,6,...] -> 4, 16, 64 subsets.  The job re-measures them on the node
     for the record, and adds the contents check the host probe could not
     settle.)
  2. WHAT recon() INITIALIZES FROM.  Both libraries call direct_recon when
     init_recon is None (parallel beam: FBP), so the init is not zero in either
     -- which makes the init a candidate rather than a control.  The job
     reports the frameworks' direct_recon rel diff at both cells.  This is the
     leading hypothesis after 1: a size-dependent difference in the FBP path
     would put the two reconstructions in different places before iteration 0,
     and three unconverged iterations would not erase it.
  3. THE FIRST DIVERGING ITERATION: recon(max_iterations=k) for k = 1, 2, 3,
     compared across frameworks at each k.  With stop_threshold_change_pct=0
     and the sequence truncated (not resampled) by max_iterations, run k is
     exactly the first k iterations of run 3, so this is a per-iteration trace
     without touching either library's internals.
  4. THE ABLATION: the same k = 1, 2, 3 with init_recon PINNED TO 0 in both
     frameworks (both accept an int init and both then take the same
     scale_recon_to_sinogram branch).  This removes the direct-recon difference
     and nothing else.  If the divergence collapses, it is attributed to the
     init; if it survives, the init is exonerated and the VCD iteration itself
     is the target.

Run:
    <torch python> p5k7_value_attrib.py     on a CUDA node (see p5k7_gautschi.sbatch)
    python p5k7_value_attrib.py --dry-run   anywhere: print the measurement plan
    python p5k7_value_attrib.py --help

Environment (export from the SUBMITTING SHELL, never in an --export list):
    P0_TORCH_PYTHON / P0_JAX_PYTHON   interpreters for the subprocesses
    P5K7_CELLS=512,1024               run a subset of the cells
    P5K7_ITERATIONS=3                 the iteration depth to trace
    P5K7_SKIP_ABLATION=1              measurements 1-3 only
    P5K7_SMOKE=1                      tiny cell on P5K7_DEVICE (default cpu)
    P5K7_DEVICE=cuda|cpu|mps          the torch device
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

SMOKE = os.environ.get("P5K7_SMOKE", "0") == "1"
SMOKE_CELL = (8, 24, 20)
DEVICE = os.environ.get("P5K7_DEVICE", "cpu" if SMOKE else "cuda")

ITERATIONS = int(os.environ.get("P5K7_ITERATIONS", "3"))
SKIP_ABLATION = os.environ.get("P5K7_SKIP_ABLATION", "0") == "1"
VCD_SEED = 13             # the p5k6 gate's seed, so this traces THAT run
SAMPLE_ROWS = 16          # recon rows kept per artifact for the comparisons
# The gate's envelope, repeated here so a rel diff can be read against it.
VALUE_REL_TOL = 5e-3

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RUN_LABEL = platform.node().split(".")[0]
# ──────────────────────────────────────────────────────────────────────────────


def selected_cells():
    wanted = os.environ.get("P5K7_CELLS", "").strip()
    if SMOKE:
        cells = [SMOKE_CELL]
    elif wanted:
        keep = {int(v) for v in wanted.split(",") if v.strip()}
        cells = [c for c in CELLS if c[0] in keep]
    else:
        cells = list(CELLS)
    return cells


def _artifact(framework, cell, tag):
    return os.path.join(RESULTS_DIR, f"_p5k7_{framework}_{cell[0]}_{tag}.npy")


def worker(cfg):
    """Everything one framework can measure at one cell, in one process: the
    sequence, the partitions, the init, and the iteration trace."""
    import numpy as np

    framework, cell = cfg["framework"], tuple(cfg["cell"])
    angles = np.linspace(0, np.pi, cell[0], endpoint=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if framework == "jax":
        import jax
        import mbirjax as mj

        model = mj.ParallelBeamModel(cell, angles)
        model.set_params(no_warning=True, verbose=0)
        gen_sequence = mj.gen_partition_sequence
        gen_partition = mj.gen_pixel_partition
        phantom_fn = mj.generate_3d_shepp_logan_low_dynamic_range
        version = f"jax {jax.__version__}"

        def direct(sino):
            return np.asarray(model.direct_recon(sino))
    else:
        import torch

        import mbirtorch as mt

        model = mt.ParallelBeamModel(cell, angles, device=DEVICE)
        model.set_params(no_warning=True, verbose=0)
        gen_sequence = mt.gen_partition_sequence
        gen_partition = mt.gen_pixel_partition
        phantom_fn = mt.generate_3d_shepp_logan_low_dynamic_range
        version = f"torch {torch.__version__}"

        def direct(sino):
            return np.asarray(model.direct_recon(sino))

    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    granularity = [int(g) for g in model.get_params("granularity")]
    partition_sequence = [int(p) for p in model.get_params("partition_sequence")]
    sequence = [int(s) for s in gen_sequence(partition_sequence,
                                             max_iterations=ITERATIONS)]
    subset_counts = [granularity[s] for s in sequence]

    result = dict(framework=framework, cell=list(cell), version=version,
                  recon_shape=list(recon_shape), granularity=granularity,
                  partition_sequence_head=partition_sequence[:8],
                  sequence=sequence, subset_counts=subset_counts,
                  init_source="direct_recon (init_recon=None)",
                  iterations=ITERATIONS, device=DEVICE)

    # 1b. The partitions THEMSELVES, at this run's seed.  Saved for the
    # parent's index-for-index comparison: matching counts do not imply
    # matching subsets, and a different assignment of voxels to subsets is a
    # different algorithm even with an identical granularity table.
    use_ror_mask = bool(model.get_params("use_ror_mask"))
    for index, count in enumerate(dict.fromkeys(subset_counts)):
        np.random.seed(VCD_SEED)
        partition = np.asarray(gen_partition(recon_shape, count,
                                             use_ror_mask=use_ror_mask))
        np.save(_artifact(framework, cell, f"part{count}"), partition)
        result[f"partition_shape_{count}"] = list(partition.shape)

    phantom = phantom_fn(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    weights = np.exp(-sinogram / (2 * np.max(sinogram)))
    result["sinogram_checksum"] = float(np.sum(np.abs(sinogram),
                                               dtype=np.float64))
    del phantom

    step = max(1, recon_shape[0] // SAMPLE_ROWS)
    result["sample_step"] = step

    # 2. The init both libraries actually start from.
    init = direct(sinogram)
    np.save(_artifact(framework, cell, "direct"), np.asarray(init)[::step])
    result["direct_checksum"] = float(np.sum(np.abs(init), dtype=np.float64))
    del init

    # 3 and 4. The iteration trace, from scratch at each depth so that run k is
    # exactly the first k iterations of run ITERATIONS (the sequence is
    # truncated, not resampled, and early stopping is off).
    inits = [("default", None)]
    if not SKIP_ABLATION:
        inits.append(("zeroinit", 0))
    for tag, init_recon in inits:
        for k in range(1, ITERATIONS + 1):
            np.random.seed(VCD_SEED)
            kwargs = dict(weights=weights, max_iterations=k,
                          stop_threshold_change_pct=0.0)
            if init_recon is not None:
                kwargs["init_recon"] = init_recon
            recon, _ = model.recon(sinogram, **kwargs)
            recon = np.asarray(recon)
            np.save(_artifact(framework, cell, f"{tag}{k}"), recon[::step])
            result[f"checksum_{tag}{k}"] = float(np.sum(np.abs(recon),
                                                        dtype=np.float64))
            del recon
    return result


def run_one(python, cfg):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg_path = os.path.join(RESULTS_DIR, "_cfg_p5k7.json")
    out_path = os.path.join(RESULTS_DIR, "_out_p5k7.json")
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


def _rel_max(path_a, path_b):
    """max |a - b| / max |b| between two saved artifacts, or None."""
    import numpy as np

    if os.path.exists(path_a) and os.path.exists(path_b):
        a, b = np.load(path_a), np.load(path_b)
        if a.shape == b.shape:
            value = float(np.max(np.abs(a - b))
                          / max(float(np.max(np.abs(b))), 1e-30))
        else:
            value = None
    else:
        value = None
    return value


def _partition_verdict(cell, count):
    """How the two frameworks' partitions at ``count`` subsets compare: equal,
    the same voxel set differently assigned, or different sets."""
    import numpy as np

    path_t = _artifact("torch", cell, f"part{count}")
    path_j = _artifact("jax", cell, f"part{count}")
    if not (os.path.exists(path_t) and os.path.exists(path_j)):
        verdict = "missing"
    else:
        a, b = np.load(path_t), np.load(path_j)
        if a.shape != b.shape:
            verdict = f"different shapes {a.shape} vs {b.shape}"
        elif np.array_equal(a, b):
            verdict = "IDENTICAL"
        elif np.array_equal(np.sort(a, axis=None), np.sort(b, axis=None)):
            verdict = "same voxel set, DIFFERENT subset assignment"
        else:
            verdict = "DIFFERENT voxel sets"
    return verdict


def compare_cell(cell, rows):
    """The attribution readout for one cell."""
    by_fw = {r.get("framework"): r for r in rows if "error" not in r}
    torch_row, jax_row = by_fw.get("torch"), by_fw.get("jax")
    label = "x".join(map(str, cell))
    print(f"\n=== attribution, cell {label} ===", flush=True)
    if torch_row is None or jax_row is None:
        print(f"  MISSING framework rows: "
              f"{[f for f in FRAMEWORKS if f not in by_fw]}", flush=True)
        return dict(cell=list(cell), error="missing framework rows")

    summary = dict(cell=list(cell))
    # 1. the sequence
    same_sequence = (torch_row["subset_counts"] == jax_row["subset_counts"])
    summary["subset_counts_torch"] = torch_row["subset_counts"]
    summary["subset_counts_jax"] = jax_row["subset_counts"]
    summary["same_sequence"] = same_sequence
    print(f"  1. subset counts per iteration: torch "
          f"{torch_row['subset_counts']}  jax {jax_row['subset_counts']}  "
          f"-> {'MATCH' if same_sequence else 'DIFFER'}", flush=True)
    summary["partition_verdicts"] = {}
    for count in dict.fromkeys(torch_row["subset_counts"]):
        verdict = _partition_verdict(cell, count)
        summary["partition_verdicts"][str(count)] = verdict
        print(f"     partitions at {count} subsets: {verdict}", flush=True)

    # 2. the init
    summary["init_source_torch"] = torch_row["init_source"]
    summary["init_source_jax"] = jax_row["init_source"]
    summary["sinogram_rel_checksum"] = abs(
        torch_row["sinogram_checksum"] - jax_row["sinogram_checksum"]) / max(
            abs(jax_row["sinogram_checksum"]), 1e-30)
    rel_direct = _rel_max(_artifact("torch", cell, "direct"),
                          _artifact("jax", cell, "direct"))
    summary["value_rel_direct_recon"] = rel_direct
    print(f"  2. init: torch = {torch_row['init_source']}, "
          f"jax = {jax_row['init_source']}", flush=True)
    print(f"     sinogram checksum rel {summary['sinogram_rel_checksum']:.2e}"
          f"   direct_recon rel "
          f"{'n/a' if rel_direct is None else format(rel_direct, '.3e')}",
          flush=True)

    # 3 and 4. the iteration traces
    for tag, name in (("default", "default init (direct_recon)"),
                      ("zeroinit", "init pinned to 0 in BOTH")):
        rels = []
        for k in range(1, ITERATIONS + 1):
            rels.append(_rel_max(_artifact("torch", cell, f"{tag}{k}"),
                                 _artifact("jax", cell, f"{tag}{k}")))
        if any(r is not None for r in rels):
            summary[f"value_rel_{tag}"] = rels
            text = "  ".join(
                f"iter{k + 1} {'n/a' if r is None else format(r, '.3e')}"
                for k, r in enumerate(rels))
            print(f"  {'3' if tag == 'default' else '4'}. {name}: {text}",
                  flush=True)
    default_rels = summary.get("value_rel_default") or []
    zero_rels = summary.get("value_rel_zeroinit") or []
    if default_rels and zero_rels and default_rels[-1] and zero_rels[-1]:
        ratio = zero_rels[-1] / default_rels[-1]
        summary["ablation_ratio"] = ratio
        if default_rels[-1] <= VALUE_REL_TOL:
            # Nothing to attribute here: this cell already agrees inside the
            # gate's envelope, so the ablation is a control, not a verdict.
            verdict = (f"no divergence to attribute at this cell "
                       f"({default_rels[-1]:.1e} <= {VALUE_REL_TOL:.0e})")
        elif ratio < 0.1:
            verdict = ("ATTRIBUTED TO THE INIT: pinning it collapses the "
                       "divergence")
        elif ratio > 0.5:
            verdict = ("INIT EXONERATED: the divergence survives a common "
                       "init, so it is in the iteration")
        else:
            verdict = "PARTIAL: the init explains some of the divergence"
        summary["ablation_verdict"] = verdict
        print(f"     ablation: rel at iter{ITERATIONS} falls from "
              f"{default_rels[-1]:.3e} to {zero_rels[-1]:.3e} "
              f"({ratio:.2f}x) -- {verdict}", flush=True)
    return summary


def print_plan():
    print(f"cells:      {[('x'.join(map(str, c))) for c in selected_cells()]}")
    print(f"frameworks: {list(FRAMEWORKS)}  (torch on {DEVICE})")
    print(f"iterations: {ITERATIONS} (seed {VCD_SEED}), no timing measured")
    print(f"ablation:   {'SKIPPED' if SKIP_ABLATION else 'init pinned to 0 in both'}")
    print("measurements: 1 sequence + partition contents, 2 init and its rel "
          "diff, 3 per-iteration rel diff, 4 the pinned-init ablation")
    print(f"results:    {RESULTS_DIR}/p5k7_value_attrib_{RUN_LABEL}.json")


def main():
    cells = selected_cells()
    print(f"p5k7 value attribution on {platform.node()} ({DEVICE}"
          f"{', SMOKE' if SMOKE else ''}): cells "
          f"{[('x'.join(map(str, c))) for c in cells]}, {ITERATIONS} "
          f"iterations, no timing", flush=True)
    all_results = dict(run_label=RUN_LABEL, host=platform.node(),
                       geometry="parallel", iterations=ITERATIONS,
                       vcd_seed=VCD_SEED, value_rel_tol=VALUE_REL_TOL,
                       rows=[], summaries=[])
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
                print(f"  sequence {row['subset_counts']}  "
                      f"direct checksum {row['direct_checksum']:.6g}",
                      flush=True)
        summary = compare_cell(cell, rows)
        all_results["rows"].extend(rows)
        all_results["summaries"].append(summary)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"p5k7_value_attrib_{RUN_LABEL}.json")
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
