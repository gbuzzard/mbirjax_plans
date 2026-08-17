"""mg17 -- THE 2048-CLASS CAPACITY TABLE, from the library's own ledger.

WHAT THIS COMPUTES.  Per-device peak memory for 2048-class reconstructions,
evaluated with the library's own closed-form model: `plan_from_model` builds
the plan and `estimate_peak_device_bytes` prices it.  Nothing is
reconstructed and nothing large is allocated.  The one GPU this job holds
serves two purposes only: the kernel availability probe binds the same
projection bodies a production run would bind, and the device budget is read
from a real idle H100 rather than assumed.

WHY A JOB AT ALL.  The ledger prices the bodies that are actually bound.  On
a machine without CUDA the Triton kernels are unavailable, so cone and
parallel would be priced as general torch bodies, which carry a different
(much larger) view-batch charge.  Evaluating on an H100 node makes the table
describe the production configuration.  The evaluation itself is pure
arithmetic and takes seconds.

THE REFERENCE SHAPE.  Sinogram (2048, 2016, 1984) as (views, detector rows,
channels).  The default recon shape follows the geometry: (1984, 1984, 2016)
for cone and parallel.  At this shape the sinogram is 32.8 GB, the recon
31.7 GB, and one full-pixel-set cylinder stack 24.9 GB, so no single H100
can hold a reconstruction and the question is which device COUNTS can.

THE VARIANTS.  Each (geometry, count) cell is priced four ways.  The four
differ only in what the back projection's combining step -- the step that
sums per-device partial results onto each slice-owner -- holds on the owner
device.

  * today        The shipped code.  `sum_band_to_owner` streams each
                 arriving partial in bounded row slabs (commit 413aeb0,
                 2026-08-11), so at the default band the owner holds its own
                 shard twice plus one slab per peer.
  * band_knob    The shipped code with `back_project_slice_band` set to a
                 quarter of the shard, the existing memory lever: the owner
                 holds shard + band instead of two shards.
  * reduce_min   The combining transient replaced by the bare output: the
                 owner holds only the reduced shard itself.  This is the
                 floor any further combining restructure could reach, and
                 the with/without comparison the A1 task asks for.
  * pre_stream   The pre-2026-08-11 form, for context: the owner held every
                 peer's partial beside the running totals, (n+1) band copies
                 at two devices and (n+2) at three or more.  This is the
                 form the 37-GB-flat premise in the open-items file
                 describes.

Only the 'today' variant is the code; the other three are computed by
replacing the 'band reduce' term inside the ledger's own phases, using the
plan's fields.  The script verifies its reconstruction of the 'today' term
against the ledger's stored value on every phase of every row, so the three
derived variants cannot drift from the closed forms they edit.

WHAT A ROW REPORTS.  The per-device modeled peaks, the largest one, the
phase that sets it and that phase's three largest terms, and whether the
demand (peak times the preflight's 1.15 margin) fits the measured idle-H100
budget.  Rows go to a jsonl beside a printed table.

ANCHOR ROWS.  The same evaluation at the 1024-class gate cells, where the
ledger is calibrated (mg11 measured every gather arm inside the 1.00-1.30
band; mg15 re-verified at the 512 class after the pad removal).  The 2048
rows are the same closed forms at a larger shape, and A3 -- the first
composed 2048-class runs -- is what will validate them there.

APPENDIX ROWS.  Translation and multiaxis at their mg8 production shapes,
'today' variant only, one, two and four devices.  These price the banded
torch-body path those geometries still run, as context for the B4
measurement; they are not part of the A1 cone/parallel table.

CONFIGURATION.  Constants below; no command-line arguments.  MG17_RESULTS
names the output directory (default ./results).  MG17_SMOKE=1 shrinks every
shape for a CPU-only correctness pass of the table machinery; smoke rows are
labeled and never mixed with real ones.
"""

import datetime
import json
import math
import os
import socket

import numpy as np
import torch

# ── configuration ────────────────────────────────────────────────────────────
SMOKE = os.environ.get("MG17_SMOKE", "") not in ("", "0")
RESULTS_DIR = os.environ.get("MG17_RESULTS", "results")

# Device counts to price.  An H100 node carries eight GPUs, so eight is the
# widest single-node layout; five through seven are included because the
# capacity question is "what fits at which counts", not "which counts divide
# evenly".
COUNTS = (1, 2, 3, 4, 5, 6, 7, 8)

# The preflight's own margin (layout_fits default): demand = 1.15 * peak.
MARGIN = 0.15

# The 2048-class reference cell and the 1024-class anchor cell, as
# (views, detector rows, channels).
CELL_2K = (2048, 2016, 1984)
CELL_1K = (1024, 1008, 992)

# The main table: cone and parallel at the 2048 class, plus the 1024-class
# anchors where the ledger is calibrated.
MAIN_SPECS = [
    dict(name="cone_2k", geometry="cone", cell=CELL_2K, counts=COUNTS),
    dict(name="parallel_2k", geometry="parallel", cell=CELL_2K,
         counts=COUNTS),
    dict(name="cone_1k", geometry="cone", cell=CELL_1K, counts=(1, 2, 4)),
    dict(name="parallel_1k", geometry="parallel", cell=CELL_1K,
         counts=(1, 2, 4)),
]

# The appendix: the two torch-body geometries at their mg8 production
# shapes, banded path, 'today' variant only.
APPENDIX_SPECS = [
    dict(name="ma1024", geometry="multiaxis", cell=(1024, 1008, 992),
         counts=(1, 2, 4)),
    dict(name="tct2k", geometry="translation", cell=(256, 1900, 3000),
         translations=(16, 16), spacing=(24.0, 16.0), counts=(1, 2, 4)),
]

SMOKE_MAIN = [
    dict(name="cone_smoke", geometry="cone", cell=(32, 24, 20),
         counts=(1, 2, 3)),
    dict(name="parallel_smoke", geometry="parallel", cell=(32, 24, 20),
         counts=(1, 2, 3)),
]
SMOKE_APPENDIX = [
    dict(name="ma_smoke", geometry="multiaxis", cell=(16, 24, 20),
         counts=(1, 2)),
    dict(name="tct_smoke", geometry="translation", cell=(16, 40, 32),
         translations=(4, 4), spacing=(3.0, 2.0), counts=(1, 2)),
]

VARIANTS = ("today", "band_knob", "reduce_min", "pre_stream")

_GB = 1024.0 ** 3
_F32_BYTES = 4


# ── model construction (the mg15 conventions) ────────────────────────────────
def build_model(spec):
    """One model per geometry spec, with the campaign's standard parameters:
    cone at magnification two, parallel over half a turn, multiaxis with
    mg8's azimuth/elevation sweep, translation from mg8's production
    translation grid."""
    import mbirtorch

    cell = tuple(spec["cell"])
    num_views, channels = cell[0], cell[2]
    geometry = spec["geometry"]
    if geometry == "cone":
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        model = mbirtorch.ConeBeamModel(
            cell, angles, source_detector_dist=4.0 * channels,
            source_iso_dist=2.0 * channels)
    elif geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirtorch.ParallelBeamModel(cell, angles)
    elif geometry == "multiaxis":
        azimuth = np.linspace(0, np.pi, num_views, endpoint=False)
        elevation = np.linspace(-0.5, 0.5, num_views)
        model = mbirtorch.MultiAxisParallelModel(
            cell, np.stack([azimuth, elevation], axis=1))
    elif geometry == "translation":
        # mg8's production translation grid: the "views" are grid positions,
        # and the source distances follow mg8's rule so the recon shape
        # reproduces the shape the prerelease review priced.
        num_x, num_z = spec["translations"]
        x_spacing, z_spacing = spec["spacing"]
        vectors = mbirtorch.gen_translation_vectors(
            num_x, num_z, x_spacing=x_spacing, z_spacing=z_spacing)
        if vectors.shape[0] != cell[0]:
            raise RuntimeError(
                f"{spec['name']}: {num_x}x{num_z} translations give "
                f"{vectors.shape[0]} views, but the sinogram has {cell[0]}")
        source_iso_dist = min(cell[1], cell[2]) / 2
        model = mbirtorch.TranslationModel(
            cell, vectors, source_detector_dist=source_iso_dist,
            source_iso_dist=source_iso_dist)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    model.set_params(no_warning=True, verbose=0)
    return model


def body_witness(model):
    """Which projection bodies this model bound, read the way the ledger
    reads it.  For cone and parallel on a CUDA node both directions must be
    kernel bodies; a row priced from a torch-body fallback would describe a
    configuration production does not run, so that is an error, not a note."""
    from mbirtorch import _memory_ledger

    directions = _memory_ledger.torch_body_directions(model)
    return dict(torch_body_directions=list(directions))


# ── the band-reduce closed forms this script may substitute ──────────────────
def reduce_terms_for_variant(plan, phase_pixels, variant):
    """Per-device replacement values for the 'band reduce' term of one
    reduce sub-phase, under one variant.

    The 'today' expression mirrors `_memory_ledger.estimate_peak_device_bytes`
    exactly, and the caller asserts it against the ledger's stored term, so a
    drift in either place fails loudly here.
    """
    from mbirtorch import _sharding

    n = plan.n_devices
    values = []
    for i in range(n):
        shard = int(plan.slice_blocks[i])
        if n == 1 or shard <= 0:
            values.append(0)
            continue
        band = int(plan.band_length(i, "back"))
        row_bytes = band * _F32_BYTES
        p = int(phase_pixels)
        if variant in ("today", "band_knob"):
            slab_rows = _sharding.reduce_slab_rows(p, row_bytes)
            value = (p * (shard + band) * _F32_BYTES
                     + (n - 1) * slab_rows * row_bytes)
        elif variant == "reduce_min":
            # The owner holds only its reduced output: the combining
            # transient is gone.  The floor a restructure could reach.
            value = p * shard * _F32_BYTES
        elif variant == "pre_stream":
            # The pre-413aeb0 gather-then-sum: every peer's partial beside
            # the running totals -- (n+1) band copies at n=2, (n+2) at n>=3
            # (the backloop-attribution reading, which measured 1.5 full
            # cylinder sets at both two and four devices).
            copies = n + 1 if n == 2 else n + 2
            value = copies * p * band * _F32_BYTES
        else:
            raise ValueError(f"unknown variant {variant!r}")
        values.append(int(value))
    return values


def phase_pixel_count(plan, phase_name):
    """The pixel count behind one reduce sub-phase, recovered from its name.
    The three phases that carry a band reduce are the direct-recon back
    loop and the hessian (full-index counts) and the subset back projection
    (the granularity's share)."""
    if phase_name.startswith("direct recon (back loop)"):
        return int(plan.num_pixels_full)
    if phase_name.startswith("hessian diagonal"):
        return int(plan.num_pixels_full if plan.hessian_masked
                   else plan.num_pixels_grid)
    if phase_name.startswith("subset back projection (granularity "):
        inside = phase_name.split("granularity ", 1)[1]
        granularity = int(inside.split(")", 1)[0])
        return math.ceil(plan.num_pixels_full / max(1, granularity))
    raise ValueError(f"no pixel count rule for phase {phase_name!r}")


def variant_peaks(plan, ledger, variant):
    """Per-device peaks under one variant, by replacing the 'band reduce'
    term inside each reduce sub-phase and taking the same per-device maximum
    over phases the ledger takes.  The 'today' variant must reproduce the
    ledger's stored term exactly; that assertion is the guard on every
    derived number this script prints."""
    n = plan.n_devices
    peaks = [0] * n
    binding = [None] * n
    for phase in ledger.phases:
        totals = list(phase.per_device)
        if phase.name.endswith("[band reduce]"):
            stored = dict(phase.terms).get("band reduce")
            if stored is None:
                raise AssertionError(
                    f"phase {phase.name!r} carries no band-reduce term")
            pixels = phase_pixel_count(plan, phase.name)
            todays = reduce_terms_for_variant(plan, pixels, "today")
            if list(stored) != todays:
                raise AssertionError(
                    "the script's band-reduce expression no longer matches "
                    f"the ledger's at {phase.name!r}: {todays} vs "
                    f"{list(stored)}")
            replacement = reduce_terms_for_variant(plan, pixels, variant)
            totals = [totals[i] - int(stored[i]) + replacement[i]
                      for i in range(n)]
        for i in range(n):
            if totals[i] > peaks[i]:
                peaks[i] = totals[i]
                binding[i] = (phase, totals[i])
    return peaks, binding


def binding_report(binding_entry, dev_index):
    """The binding phase's name and its three largest terms on one device.
    For a variant-edited phase the term list still names the shipped terms;
    the substituted value is reported through the phase total instead, so
    the term names are context, not arithmetic."""
    phase, _total = binding_entry
    top = phase.dominant_terms(dev_index, count=3)
    return dict(
        phase=phase.name,
        top_terms=[[name, int(value)] for name, value in top])


# ── one row ──────────────────────────────────────────────────────────────────
def price_row(model, spec, count, variant, budget_bytes):
    """One (geometry, count, variant) row: build the candidate plan, price
    it, and report peaks, the binding phase, and the fit verdict."""
    from mbirtorch import _memory_ledger

    devices = [torch.device(f"cuda:{i}") for i in range(count)]
    if SMOKE:
        devices = [torch.device("cpu")] * count

    knob = None
    if variant == "band_knob":
        # A quarter of the largest shard, at least one slice.  Set on the
        # model so plan_from_model reads it exactly as a user's setting.
        largest_shard = math.ceil(
            int(model.get_params("recon_shape")[2]) / count)
        knob = max(1, math.ceil(largest_shard / 4))
        model.back_project_slice_band = knob
    try:
        plan = _memory_ledger.plan_from_model(model, devices,
                                              workload="recon")
        ledger = _memory_ledger.estimate_peak_device_bytes(plan)
    finally:
        if variant == "band_knob":
            del model.back_project_slice_band

    peaks, binding = variant_peaks(plan, ledger, variant)
    worst = max(range(count), key=lambda i: peaks[i])
    demand = int((1.0 + MARGIN) * peaks[worst])
    fits = None if budget_bytes is None else bool(demand <= budget_bytes)

    recon_shape = tuple(int(s) for s in model.get_params("recon_shape"))
    sino_shape = tuple(int(s) for s in model.get_params("sinogram_shape"))
    cylinder = plan.num_pixels_full * recon_shape[2] * _F32_BYTES
    row = dict(
        script="mg17", spec=spec["name"], geometry=spec["geometry"],
        smoke=SMOKE, cell=list(sino_shape), recon_shape=list(recon_shape),
        num_pixels_full=int(plan.num_pixels_full),
        n=count, variant=variant, back_band_knob=knob,
        column_pixel_batch=plan.column_pixel_batch,
        torch_body_directions=list(plan.torch_body_directions),
        view_blocks=list(plan.view_blocks),
        slice_blocks=list(plan.slice_blocks),
        peak_per_device_gb=[round(p / _GB, 3) for p in peaks],
        peak_max_gb=round(peaks[worst] / _GB, 3),
        demand_gb=round(demand / _GB, 3),
        budget_gb=(None if budget_bytes is None
                   else round(budget_bytes / _GB, 3)),
        fits=fits,
        sinogram_gb=round(math.prod(sino_shape) * _F32_BYTES / _GB, 3),
        recon_gb=round(math.prod(recon_shape) * _F32_BYTES / _GB, 3),
        cylinder_gb=round(cylinder / _GB, 3),
        binding=binding_report(binding[worst], worst),
    )
    return row


# ── the run ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    host = socket.gethostname().split(".")[0]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        RESULTS_DIR, f"mg17_capacity_{host}_{stamp}.jsonl")

    import mbirtorch
    from mbirtorch import _memory_ledger

    print(f"mbirtorch from {mbirtorch.__file__}", flush=True)
    if not SMOKE:
        assert torch.cuda.is_available(), (
            "mg17 prices the production configuration and needs one CUDA "
            "device for the kernel probe and the budget reading")
        budget = _memory_ledger.device_budget_bytes(torch.device("cuda:0"))
        name = torch.cuda.get_device_name(0)
        print(f"budget device: {name}, idle budget "
              f"{budget / _GB:.2f} GB", flush=True)
    else:
        budget = None
        print("SMOKE: cpu-only table-machinery pass, no budget", flush=True)

    main_specs = SMOKE_MAIN if SMOKE else MAIN_SPECS
    appendix = SMOKE_APPENDIX if SMOKE else APPENDIX_SPECS

    rows = []
    with open(out_path, "w") as out:
        def emit(row):
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()

        for spec in main_specs:
            model = build_model(spec)
            witness = body_witness(model)
            if not SMOKE and witness["torch_body_directions"]:
                raise AssertionError(
                    f"{spec['name']}: expected kernel bodies in both "
                    f"directions, got torch bodies for "
                    f"{witness['torch_body_directions']}; this row would "
                    "not describe the production configuration")
            for count in spec["counts"]:
                for variant in VARIANTS:
                    if count == 1 and variant != "today":
                        continue        # one device has no combining step
                    emit(price_row(model, spec, count, variant, budget))
            del model

        for spec in appendix:
            model = build_model(spec)
            witness = body_witness(model)
            for count in spec["counts"]:
                row = price_row(model, spec, count, "today", budget)
                row["appendix"] = True
                row.update(witness)
                emit(row)
            del model

    print(f"\nrows written to {out_path}\n", flush=True)
    print_tables(rows)
    return 0


def print_tables(rows):
    """One table per spec: counts down, variants across, peak GB in the
    cells, with the binding phase of the 'today' column named per count."""
    specs = []
    for row in rows:
        if row["spec"] not in specs:
            specs.append(row["spec"])
    for spec in specs:
        spec_rows = [r for r in rows if r["spec"] == spec]
        counts = sorted({r["n"] for r in spec_rows})
        variants = [v for v in VARIANTS
                    if any(r["variant"] == v for r in spec_rows)]
        head = spec_rows[0]
        print(f"== {spec}  cell {tuple(head['cell'])}  recon "
              f"{tuple(head['recon_shape'])}  sinogram "
              f"{head['sinogram_gb']} GB  recon {head['recon_gb']} GB  "
              f"cylinder {head['cylinder_gb']} GB")
        header = f"{'n':>3}" + "".join(f"{v:>13}" for v in variants)
        header += "   binding phase (today)  fits(today)"
        print(header)
        for count in counts:
            cells = []
            today = None
            for variant in variants:
                match = [r for r in spec_rows
                         if r["n"] == count and r["variant"] == variant]
                if not match:
                    cells.append(f"{'-':>13}")
                    continue
                cells.append(f"{match[0]['peak_max_gb']:>12.2f} ")
                if variant == "today":
                    today = match[0]
            fits = ("-" if today is None or today["fits"] is None
                    else ("yes" if today["fits"] else "NO"))
            phase = today["binding"]["phase"] if today else "-"
            print(f"{count:>3}" + "".join(cells) + f"   {phase}  {fits}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
