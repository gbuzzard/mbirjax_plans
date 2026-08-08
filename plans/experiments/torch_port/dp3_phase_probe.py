"""Phase-resolved attribution for the memory-ledger under-charge.

The calibration table reports one measured peak per cell.  When that peak
exceeds the modeled peak, one number cannot say WHICH phase is under-charged.
This probe measures the real reconstruction phase by phase and puts each
measured phase beside the ledger's own term for it.

It instruments the production path rather than re-implementing it.  Each
wrapped method resets the peak counter on entry and reads it on exit, so a
phase's number is the true allocator high-water mark while that phase ran,
with the resident set it inherited included.  The entry column is
`memory_allocated` at the same instant, which separates a wrong PERSISTENT
set from a wrong TRANSIENT: if the entry column matches the ledger's
persistent terms and the peak does not, the miss is a transient.

The warm pass is the measured one.  A cold pass runs first so no compile
lands inside a measured phase.

Environment:
    DP3_GEOMETRY=parallel|cone     which geometry (default parallel)
    DP3_VIEWS=512|1024             which cell (default 1024)
    DP3_WEIGHTED=1|0               supply weights (default 1)
    DP3_RESULTS=<dir>              where the sinograms live
"""

import json
import os

import numpy as np
import torch

import mbirtorch
from mbirtorch import _memory_ledger as ML

RESULTS_DIR = os.environ.get("DP3_RESULTS", ".")
CELLS = {512: (512, 448, 384), 1024: (1024, 1008, 992)}
VCD_ITERATIONS = 3
VCD_SEED = 12345


def _sino_path(geometry, cell):
    return os.path.join(RESULTS_DIR, f"_dp2_sino_{geometry}_{cell[0]}.npy")


def _build_model(geometry, cell):
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, cell[0], endpoint=False)
        return mbirtorch.ParallelBeamModel(cell, angles, device="cuda")
    angles = np.linspace(0, 2 * np.pi, cell[0], endpoint=False)
    sdd = 4 * cell[2]
    return mbirtorch.ConeBeamModel(cell, angles, source_detector_dist=sdd,
                                   source_iso_dist=sdd)


def main():
    geometry = os.environ.get("DP3_GEOMETRY", "parallel")
    views = int(os.environ.get("DP3_VIEWS", "1024"))
    weighted = os.environ.get("DP3_WEIGHTED", "1") not in ("0", "", "false")
    cell = CELLS[views]

    model = _build_model(geometry, cell)
    model.set_params(no_warning=True, verbose=0)
    model.configure_devices(1)
    sinogram = np.load(_sino_path(geometry, cell))
    weights = (np.exp(-sinogram / (2 * np.max(sinogram))).astype(np.float32)
               if weighted else None)

    marks = []

    def mark(label, entry, peak):
        marks.append(dict(label=label, entry_bytes=int(entry),
                          peak_bytes=int(peak)))

    def wrap(name, label):
        original = getattr(model, name)

        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            entry = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = original(*args, **kwargs)
            torch.cuda.synchronize()
            mark(label, entry, torch.cuda.max_memory_allocated())
            return out
        setattr(model, name, wrapped)

    wrap("direct_recon", "direct recon")
    wrap("_initial_error_state", "initial error state")
    wrap("compute_hessian_diagonal", "hessian diagonal")

    # The subset step, labelled by its subset size so the granularity that
    # produced each reading is unambiguous.
    original_create = model.create_vcd_subset_updater

    def create(*args, **kwargs):
        updater = original_create(*args, **kwargs)

        def wrapped(flat_recon, error_sinogram, pixel_indices):
            torch.cuda.synchronize()
            entry = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = updater(flat_recon, error_sinogram, pixel_indices)
            torch.cuda.synchronize()
            mark(f"subset P={int(pixel_indices.shape[0])}", entry,
                 torch.cuda.max_memory_allocated())
            return out
        wrapped.stage_halos = updater.stage_halos
        return wrapped
    model.create_vcd_subset_updater = create

    def one_recon():
        np.random.seed(VCD_SEED)
        return model.recon(sinogram, weights=weights,
                           max_iterations=VCD_ITERATIONS,
                           stop_threshold_change_pct=0.0)

    one_recon()                      # cold: pays every compile
    marks.clear()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    one_recon()                      # warm: the measured pass
    torch.cuda.synchronize()
    whole_run_peak = int(torch.cuda.max_memory_allocated())

    ledger = model.last_memory_ledger
    if ledger is None:
        os.environ["MBIRTORCH_MEMORY_CALIBRATION"] = "1"
        one_recon()
        ledger = model.last_memory_ledger

    # Collapse the repeated subset readings to the worst per subset size.
    worst = {}
    for entry in marks:
        key = entry["label"]
        if key not in worst or entry["peak_bytes"] > worst[key]["peak_bytes"]:
            worst[key] = entry

    print(f"\n===== phase probe: {geometry} {cell} "
          f"{'weighted' if weighted else 'unweighted'} =====")
    print(f'{"phase":>26}{"entry":>12}{"measured":>12}{"modeled":>12}'
          f'{"ratio":>9}')
    print("-" * 71)
    modeled_by_phase = {p.name: p.per_device[0] for p in ledger.phases}
    rows = []
    for label, entry in sorted(worst.items(), key=lambda kv: -kv[1]["peak_bytes"]):
        modeled = _match(label, modeled_by_phase, ledger)
        ratio = (modeled / entry["peak_bytes"]) if entry["peak_bytes"] else 0.0
        rows.append(dict(entry, modeled_bytes=int(modeled), ratio=float(ratio)))
        print(f'{label:>26}{entry["entry_bytes"] / 2 ** 30:>11.2f}G'
              f'{entry["peak_bytes"] / 2 ** 30:>11.2f}G'
              f'{modeled / 2 ** 30:>11.2f}G{ratio:>9.3f}')
    print("-" * 71)
    print(f'{"whole warm run":>26}{"":>12}{whole_run_peak / 2 ** 30:>11.2f}G'
          f'{ledger.peak_bytes(0) / 2 ** 30:>11.2f}G'
          f'{ledger.peak_bytes(0) / whole_run_peak:>9.3f}')

    print("\n--- the ledger's own phase table ---")
    print(ledger.format_table())

    out_path = os.path.join(
        RESULTS_DIR,
        f"dp3_phase_probe_{geometry}_{views}_"
        f"{'w' if weighted else 'u'}.json")
    with open(out_path, "w") as handle:
        json.dump(dict(geometry=geometry, cell=list(cell), weighted=weighted,
                       whole_run_peak_bytes=whole_run_peak, phases=rows,
                       ledger=[dict(name=p.name, bytes=int(p.per_device[0]),
                                    terms=[(t, int(v[0])) for t, v in p.terms])
                               for p in ledger.phases]), handle, indent=1)
    print(f"\nwrote {out_path}")


def _match(label, modeled_by_phase, ledger):
    """The ledger phase that corresponds to one measured phase."""
    if label.startswith("subset P="):
        # A subset step's measured peak spans every sub-phase of that step, so
        # its counterpart is the largest sub-phase at the matching granularity.
        num_pixels = int(label.split("=")[1])
        best, best_gap = 0, None
        for phase in ledger.phases:
            if not phase.name.startswith("subset "):
                continue
            granularity = int(phase.name.rsplit(" ", 1)[1].rstrip(")"))
            modeled_pixels = -(-ledger.num_pixels_full // granularity)
            gap = abs(modeled_pixels - num_pixels)
            if best_gap is None or gap < best_gap:
                best_gap, best = gap, phase.per_device[0]
            elif gap == best_gap:
                best = max(best, phase.per_device[0])
        return best
    for name, value in modeled_by_phase.items():
        if label in name or name in label:
            return value
    if label == "initial error state":
        return max(v for n, v in modeled_by_phase.items()
                   if "initial forward" in n or "error sinogram" in n
                   or "init recon scaling" in n)
    return 0


if __name__ == "__main__":
    main()
