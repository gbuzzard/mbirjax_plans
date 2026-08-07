# Preprocessing port — plan for Charlie's session

**Status:** PLAN, ready to execute (written 2026-08-07 for a session working
with Charlie; reviewed by Greg).  This is one piece of `current_plans.md`
item 8 (the remaining utility API surface).  The work is the
`mbirjax.preprocess` package plus the coupled main-package pieces — the MAR
weights, the blocked median, the download utilities, and the HDF5 save/load
family — ported into the `mbirtorch` repository.

**Context in one paragraph.**  mbirtorch is a PyTorch port of mbirjax.  The
core is done and performance-gated: parallel and cone geometries, the VCD
engine, multi-device sharding, and hand-written Triton projector kernels that
put the port within the replacement rule (2x of jax time, 1.5x of memory) at
every gate cell.  The public API mirrors mbirjax — numpy in, numpy out, device
tensors opt-in — so preprocessing ports as the same functions with the same
signatures.  mbirjax remains the read-only reference throughout.

## Scope

The port target is `mbirjax/preprocess/` (~6k lines) plus two main-package
functions.  The inventory, with the jax coupling that determines porting
effort:

| module | lines | jax use | content |
|---|---|---|---|
| `utilities.py` | 1573 | heavy in kernels | the scan-to-sinogram chain: transmission, defective-pixel interpolation, rotation and background correction, downsample/crop, TIFF readers, helpers |
| `mar.py` | 882 | heavy | metal artifact reduction: Huber weights, BH correction, the plastic/metal fit (OSQP), `recon_plastic_metal` |
| `stripe.py` | 411 | heavy | ring/stripe removal (sorting-based and wavelet-FFT) |
| `segmentation.py` | 395 | light | multi-threshold Otsu, `segment_plastic_metal` |
| `pipeline.py` | 113 | yes | `map_view_batches`, the batched-kernel driver |
| `nsi.py`, `zeiss.py`, `zeiss_tct.py`, `pymbir.py`, `_xradia_ole.py` | ~2600 | light | vendor loaders: file parsing plus calls into the chain above |

Several main-package pieces ride along, all documented mbirjax API that
mbirtorch currently lacks: `gen_weights_mar` (in `mbirjax/vcd_utils.py`;
depends on segmentation and a ct_model), `median_filter3d` (in
`mbirjax/denoising.py`; a blocked median), the download utilities
(`download_and_extract` in `mbirjax/utilities.py`), and the HDF5 save/load
family — `export_recon_hdf5`/`import_recon_hdf5` and
`save_data_hdf5`/`load_data_hdf5` in `mbirjax/utilities.py`, plus the model
methods `save_recon_hdf5`/`load_recon_hdf5` and their metadata providers
`get_recon_dict`/`get_all_params` in `mbirjax/tomography_model.py`.  The
HDF5 family belongs here because its export path calls into preprocess (the
flash-removal margins use `apply_cylindrical_mask`).

**The end-to-end target.**  The reference application script is
`mbirjax_applications/nsi/Lilly_recon.py`: NSI dataset in via
`nsi.get_sino_and_model` (which constructs the ConeBeamModel from the vendor
geometry files), transmission weights, `recon_plastic_metal`, HDF5 export.
An mbirtorch translation of that script running end to end is the port's
integration test, and increment 4 gates on it.

Out of scope, staying with the mbirtorch team: the model factories, the
phantom/demo-data generators, `device_summary`, multi-device batching (the
`devices=` parameter of `map_view_batches` ports as single-device batching;
the sharding engine integration comes later), and the Sphinx pages (see
Coordination below).

## What already exists to build on

`mbirtorch` has the full model layer: `ParallelBeamModel`, `ConeBeamModel`,
`recon()`, `direct_recon()`, `gen_weights`, and the projectors — everything
`mar.py` and `gen_weights_mar` call.  Device handling is
`device='auto'` at model construction; preprocessing tensors follow the same
rule (compute on the model's device when a model is in hand, else on the best
available device, numpy at every public boundary).  The package is
float32-throughout by design.

## Porting rules

These are the rules the rest of the port followed; they apply verbatim.

1. **Translate, do not redesign.**  Copy each function's structure and its
   docstring from mbirjax, adapting wording (jax array -> tensor) and nothing
   else.  Where mbirjax batches with `map_view_batches`, port the driver once
   (a plain loop over view batches with host-to-device staging) and reuse it.
2. **numpy at the boundary.**  Public functions accept and return numpy;
   torch is an internal detail.  Scalars returned to users are python floats
   (note `compute_scaling_factor` returns a float in mbirjax; keep that).
3. **Out-of-bounds discipline.**  jax drops out-of-bounds scatters and clamps
   out-of-bounds gathers; torch `index_add_` asserts instead.  Every ported
   gather/scatter adopts the clip-plus-zero-weight pattern.
4. **Index dtype is int64** for anything that indexes a tensor.
5. **The 2^31 lesson.**  Element counts of large arrays overflow int32
   arithmetic; use `math.prod` and the `float()` idiom for products that feed
   further arithmetic.  `mar.py` had exactly this bug (fixed 2026-07-10);
   port the fixed form.
6. **Port the CURRENT mbirjax.**  Two recent MAR fixes must carry over: the
   OSQP infeasible-constraint guard (an infeasible fit returned a finite
   sentinel that poisoned theta; the fix is support-restricted selection, an
   RHS clamp, and a solver-status guard) and the size-relative segmentation
   mask margins.  Both are in mbirjax main as of 2026-07-26.
7. **Memory discipline.**  `interpolate_defective_pixels` is ported from its
   CHUNKED form (the unchunked per-batch NaN gather once peaked ~57 GB per
   GPU); keep the chunking and its batch-size parameters.  OSQP stays a
   host-side numpy dependency — no torch involvement in the fit's solver.
8. **No plan notation in code.**  Comments describe the code, never this
   plan's increment names or dates.

## Testing protocol

The established golden mechanism: a jax-side script (run in a jax
environment) writes golden HDF5 inputs and outputs per function; the
mbirtorch tests read them, so the mbirtorch test environment never imports
jax.  Extend the existing goldens generator rather than inventing a second
mechanism (`dev_scripts/` in mbirtorch has the pattern).

Two rules from measured experience:

- **Shared inputs, always.**  Never generate test inputs independently in
  each framework — generators can differ at float boundary ties across
  frameworks and even across platforms (a measured lesson from the kernel
  campaign).  The golden file carries the exact input array both sides use.
- **Tolerances are measured, not asserted.**  Deterministic host-side chains
  should sit at the f32 floor (rel-max ~1e-6); start gates at 1e-5 and record
  each function's measured floor beside its gate.  The MAR end-to-end path
  has reconstructions in the loop, so it gates at a documented looser
  tolerance on the corrected sinogram and the fitted theta, not bitwise.

Every public function gets a golden parity test.  Functions with batch-size
parameters also get a batch-invariance test (two batch sizes, same result),
which is the cheap guard against seam bugs in the batching driver.

## Increments

Each increment ends with the suite green and a short PR-sized review unit.

1. **Scaffold and the scan-to-sinogram chain.**  The `mbirtorch/preprocess/`
   package mirroring mbirjax's layout and `__init__` surface; the batching
   driver; then `utilities.py`: `compute_sino_transmission`,
   `interpolate_defective_pixels`, `correct_det_rotation`,
   `correct_background_offset`, `downsample_view_data`, `crop_view_data`,
   `scan_to_sino`, the TIFF readers, and the small helpers.  Gate: golden
   parity per function plus the batch-invariance tests.
2. **Stripe removal and segmentation.**  `stripe.py` (note the `pywt`
   dependency) and `segmentation.py`.  Gate: goldens; the Otsu thresholds
   should match exactly on shared inputs (integer bin arithmetic).
3. **MAR and the coupled main-package functions.**  `mar.py` against
   mbirtorch models, `gen_weights_mar` into `mbirtorch/vcd_utils.py`, and
   `median_filter3d` into `mbirtorch/denoising.py`.  Gate: function goldens
   for the pieces; an end-to-end `recon_plastic_metal` comparison on a small
   shared case at the documented tolerance; the OSQP guard behavior covered
   by tests ported from mbirjax's.
4. **Vendor loaders, the download utilities, and the end-to-end run.**
   `download_and_extract` into `mbirtorch/utilities.py`; then `nsi.py` first
   (the reference-script path), then the other loaders as data availability
   allows — `_xradia_ole` and the zeiss pair are mostly file parsing.  The
   NSI loader constructs a ConeBeamModel and sets physical-units parameters
   (e.g. `alu_value`); any parameter it sets that mbirtorch lacks is a
   finding to report, not to work around.  Gates: the translated
   `Lilly_recon.py` runs end to end on the NSI dataset and its MAR recon
   matches the mbirjax run at the increment-3 tolerance; a smoke load of
   real sample data per other vendor where a file exists, with loaders
   lacking accessible data ported but marked untested-with-real-data in
   their docstrings (Charlie decides which vendors merit chasing files).
5. **The HDF5 save/load family.**  `export_recon_hdf5`/`import_recon_hdf5`
   and `save_data_hdf5`/`load_data_hdf5` into `mbirtorch/utilities.py`, and
   `save_recon_hdf5`/`load_recon_hdf5` with `get_recon_dict`/`get_all_params`
   as additive methods on `TomographyModel`.  This increment is independent
   of 2–3 and should land BEFORE increment 4's end-to-end gate, so the
   translated reference script calls the real `export_recon_hdf5`.  Gate:
   round-trip tests (save then load reproduces array and metadata) plus a
   golden read of an mbirjax-written file, which pins the on-disk format as
   shared between the two packages.

## Coordination

- **Files owned by this work:** `mbirtorch/preprocess/` (new),
  `tests/test_preprocess_*.py` (new), the goldens-generator additions, and
  small additions to `mbirtorch/vcd_utils.py` and `mbirtorch/denoising.py`
  (increment 3), `mbirtorch/utilities.py` (increments 4 and 5), and
  `mbirtorch/tomography_model.py` (increment 5 ONLY, under a strict rule:
  purely additive methods in one bounded block, in their own commit, with no
  edit to any existing line — that file is under active change by the
  batching and device-policy work).  Please do not otherwise touch the
  engine, projector, or kernel files
  (`tomography_model.py`, `projectors.py`, `parallel_beam.py`,
  `cone_beam.py`, `horizontal_fan.py`, `triton_*.py`,
  `kernel_availability.py`, `_sharding.py`) — active work is in flight there.
- **Package exports.**  Mirror mbirjax: `mbirtorch.preprocess` is a
  subpackage namespace (`from . import preprocess` in the main `__init__`),
  so the main package's `__all__` does not change.
- **Dependencies.**  The port adds `osqp`, `pywt` (PyWavelets), `tifffile`,
  `cv2` (opencv-python), and `h5py`.  Mirror mbirjax's dependency choices in
  `pyproject.toml` (an optional `[preprocess]` extra is acceptable if Charlie
  prefers a lean core install); `pyproject.toml` is a shared file, so keep
  its edit minimal and isolated in its own commit.
- **Docs.**  A separate docs session maintains the Sphinx pages, which carry
  `PENDING` markers for exactly these functions.  When an increment lands,
  note it in the PR description so the markers can be restored; do not edit
  `docs/` from this work.
- **Review.**  PRs against the mbirtorch repository, one per increment,
  reviewed by Greg.  Questions and anything surprising found in the mbirjax
  source (the port has repeatedly surfaced real upstream findings) go to
  Greg rather than being silently worked around.
