"""Verification script for the artifacts-branch PR (viewer / segmentation / MAR).

Runs against WHICHEVER mbirjax is importable, and reports which behavior it
finds -- so it can be run twice for a before/after comparison:

    old code:  copy this file anywhere, activate an env with the PRERELEASE
               mbirjax installed (or sys.path a prerelease checkout), run it.
    new code:  same, with the artifacts branch installed.

Checks (each prints OLD-code or NEW-code behavior; total runtime a few
minutes on CPU, under a minute on GPU):

  1. slice_viewer nonblocking option -- signature introspection only (no GUI).
  2. Segmentation mask margins -- at a 64-wide volume, the old fixed
     radial_margin=10 caps the plastic mask at radius 22 and cuts the corners
     of a 0.6-width slab (corner radius ~27); the new size-relative default
     (margin 2) keeps them.
  3. MAR num_metal=1 on a single-metal beam-hardened case -- the old code
     collapses the plastic region (plastic NRMSE ~ 1.0, i.e. plastic mostly
     erased); the fixed code corrects it (plastic NRMSE well below the
     uncorrected recon's).

The reconstructions are saved to ./verify_artifacts_out_{old|new}/ (labeled
by the detected code version), so after running under both versions the
results can be compared side by side in the slice viewer:

    import numpy as np, mbirjax as mj
    mj.slice_viewer(np.load('verify_artifacts_out_old/ground_truth.npy'),
                    np.load('verify_artifacts_out_old/recon_mar.npy'),
                    np.load('verify_artifacts_out_new/recon_mar.npy'),
                    slice_label=['ground truth', 'MAR (old)', 'MAR (new)'])

No command line arguments.
"""

import inspect
import os

import numpy as np

import mbirjax as mj
import mbirjax.preprocess as mjp

SIZE = 64
NUM_VIEWS = 40
RECON_ITERATIONS = 8
SEVERITY = 0.5          # beam-hardening strength for check 3


# ---- compact polychromatic model (self-contained; mirrors the artifacts demo) --
def _klein_nishina(e_kev):
    k = np.asarray(e_kev, dtype=np.float64) / 511.0
    return ((1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
            + np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2)


def _spectrum(kvp=140.0, filtration_mm_al=2.0, n_bins=24, e_min=20.0, e_ref=60.0):
    edges = np.linspace(e_min, kvp, n_bins + 1)
    e = 0.5 * (edges[:-1] + edges[1:])
    mu_al = 0.075 * (0.35 * (e / e_ref) ** -3
                     + 0.65 * _klein_nishina(e) / _klein_nishina(e_ref))
    s = np.maximum(kvp - e, 0.0) / e * np.exp(-mu_al * filtration_mm_al)
    return e, s / s.sum()


def _poly_sinogram(path_sinos, w_pe_list, severity, e_ref=60.0):
    energies, weights = _spectrum()
    ratios = []
    for w_pe in w_pe_list:
        r = w_pe * (energies / e_ref) ** -3 \
            + (1 - w_pe) * _klein_nishina(energies) / _klein_nishina(e_ref)
        rs = (1.0 - severity) + severity * r
        ratios.append(rs / np.sum(weights * rs))
    total = np.zeros_like(path_sinos[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        expo = sum(float(rs[k]) * t for t, rs in zip(path_sinos, ratios))
        total += wk * np.exp(-expo)
    return (-np.log(total)).astype(np.float32)


def _case():
    """Plastic slab + one large central metal ball; polychromatic noisy sino."""
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sdd = 2.5 * SIZE
    model = mj.ConeBeamModel((NUM_VIEWS, SIZE, SIZE), angles,
                             source_detector_dist=sdd, source_iso_dist=sdd / 2)
    model.set_params(verbose=0, sharpness=1.0, snr_db=30.0)
    shape = model.get_params('recon_shape')
    slab = np.zeros(shape, np.float32)
    metal = np.zeros(shape, np.float32)
    rc = (shape[0] - 1) / 2.0
    half = int(0.3 * SIZE)
    zc = shape[2] // 2
    hz = int(0.25 * shape[2])
    slab[int(rc) - half:int(rc) + half + 1, int(rc) - half:int(rc) + half + 1,
         zc - hz:zc + hz + 1] = 1.0
    radius = 0.12 * SIZE
    rad_i = int(np.ceil(radius)) + 1
    rr = np.arange(int(rc) - rad_i, int(rc) + rad_i + 1)
    sph = ((rr - rc)[:, None, None] ** 2 + (rr - rc)[None, :, None] ** 2
           + (rr - rc)[None, None, :] ** 2) <= radius ** 2
    w = slab[np.ix_(rr, rr, rr)]; w[sph] = 0.0; slab[np.ix_(rr, rr, rr)] = w
    w = metal[np.ix_(rr, rr, rr)]; w[sph] = 6.0; metal[np.ix_(rr, rr, rr)] = w

    t = [np.asarray(model.forward_project(v), np.float64) for v in (slab, metal)]
    scale = 5.0 / float(sum(t).max())
    t = [x * scale for x in t]
    sino = _poly_sinogram(t, [0.05, 0.8], SEVERITY)
    rng = np.random.default_rng(0)
    counts = rng.poisson(1.0e4 * np.exp(-np.float64(sino)))
    sino = (-np.log(np.maximum(counts, 1) / 1.0e4)).astype(np.float32)
    gt = (slab + metal) * scale
    return model, gt, sino, scale


def main():
    print('mbirjax under test:', mj.__file__)

    # ---- check 1: nonblocking viewer option (signature only) ----
    has_block = 'block' in inspect.signature(mj.slice_viewer).parameters
    print(f'\n[1] slice_viewer block= parameter: '
          f'{"present -> NEW code" if has_block else "absent -> OLD code"}')

    # ---- build the shared case ----
    print('\nBuilding the single-metal beam-hardened case '
          f'({SIZE}^3, {NUM_VIEWS} views)...')
    model, gt, sino, scale = _case()
    weights = mj.gen_weights(sino, weight_type='transmission_root')
    recon_std, _ = model.recon(sino, weights=weights,
                               max_iterations=RECON_ITERATIONS)
    recon_std = np.asarray(recon_std)

    # ---- check 2: segmentation mask margins ----
    pm, _, _, _ = mjp.segment_plastic_metal(recon_std, num_metal=1)
    sl = np.asarray(pm)[:, :, gt.shape[2] // 2]
    r = np.hypot(*np.meshgrid(np.arange(SIZE) - (SIZE - 1) / 2.0,
                              np.arange(SIZE) - (SIZE - 1) / 2.0))
    rmax = float(r[sl > 0].max()) if sl.any() else 0.0
    verdict = ('size-relative -> NEW code' if rmax > 24.0
               else 'fixed 10 -> OLD code')
    print(f'\n[2] plastic-mask max radius at {SIZE}^3: {rmax:.1f} '
          f'(slab corner ~{0.3 * SIZE * np.sqrt(2):.0f}; old cap = '
          f'{SIZE // 2 - 10}) -> {verdict}')

    # ---- check 3: MAR num_metal=1 plastic collapse ----
    print('\n[3] recon_plastic_metal(num_metal=1) on the hardened case...')
    recon_mar = np.asarray(mjp.recon_plastic_metal(
        model, sino, weights, num_BH_iterations=1, num_metal=1,
        max_iterations=RECON_ITERATIONS))
    metal_floor = 3.0 * scale
    plastic = (gt > 0) & (gt < metal_floor)

    def perr(rec):
        return float(np.linalg.norm((rec - gt)[plastic])
                     / np.linalg.norm(gt[plastic]))

    p_std, p_mar = perr(recon_std), perr(recon_mar)
    verdict = ('plastic PRESERVED/IMPROVED -> NEW code (collapse fixed)'
               if p_mar < 0.6 else 'plastic COLLAPSED -> OLD code (the bug)')
    print(f'    plastic-region NRMSE: standard {p_std:.3f} | MAR {p_mar:.3f} '
          f'-> {verdict}')

    # Save the reconstructions, labeled by the detected code version, so the
    # two runs' results can be viewed side by side (snippet in the docstring).
    out_dir = f'verify_artifacts_out_{"new" if has_block else "old"}'
    os.makedirs(out_dir, exist_ok=True)
    for name, vol in (('ground_truth', gt), ('recon_standard', recon_std),
                      ('recon_mar', recon_mar)):
        np.save(os.path.join(out_dir, f'{name}.npy'), vol.astype(np.float32))
    print(f'\nReconstructions saved to ./{out_dir}/ '
          '(ground_truth, recon_standard, recon_mar)')

    print('\nSummary: run this script under the prerelease install and under '
          'the artifacts branch; checks 1-3 should each flip from OLD to NEW, '
          'and the saved recons can be compared in the slice viewer.')


if __name__ == '__main__':
    main()
