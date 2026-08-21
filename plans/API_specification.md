# API OVERVIEW

## Design philosophy

The API is designed so that a user's script does not depend on the number of GPUs.
The same script runs with no GPU, one GPU, or many GPUs, and the number of GPUs
changes only the speed and the feasible problem size.  Four principles carry this.

**1. Users work with CPU arrays.**  Every user-facing function accepts numpy arrays,
and returns numpy arrays by default.  Torch tensors are also accepted, for robustness.
CPU arrays are the right resting place for data: CPU memory is much larger than GPU
memory, results are ultimately viewed and saved from the CPU, and a CPU array is
valid no matter how many GPUs are present.

**2. GPU use is the library's responsibility, not the user's.**  A reconstruction
chooses its GPU layout automatically, once per model, balancing speed and memory.
A preprocessing function divides its work across the available GPUs internally.
In both cases the user sees only numpy in and numpy out.

**3. The divided GPU form is internal.**  When work is spread across several GPUs,
each GPU holds one piece of each array (a `Shards` object).  This form appears at
the API surface in exactly one way: a reconstruction method can return it on request
(`output_sharded=True`), so that a following on-device step does not pay a round
trip through CPU memory.  Functions that cannot operate on the divided form reject
it with a clear error rather than computing a wrong answer.

**4. Data crosses between stages through CPU memory.**  Preprocessing writes numpy;
reconstruction reads numpy and places it on the GPUs it chose.  At production scale
this transfer costs seconds, while a reconstruction costs minutes to hours, so the
simplicity is nearly free.

Explicit control is available but never required: `configure_devices` fixes a
model's layout, the `MBIRTORCH_NUM_DEVICES` environment variable caps the count
process-wide, and preprocessing functions accept a `devices=` argument.

## Function categories

The public surface forms the tree below.  Each subcategory shows one or two
representative functions, not the full list; the detailed sections that follow are
complete.

```
mbirtorch API
├── 1. Reconstruction
│   ├── One-call functions ......... recon_simple_parallel, recon_simple_cone
│   ├── Projection ................. forward_project, back_project
│   ├── Iterative reconstruction ... recon, prox_map
│   ├── Direct reconstruction ...... recon_fbp, recon_fdk
│   ├── Specialized reconstruction . recon_split_sino, recon_plastic_metal
│   └── Denoising .................. denoise
├── 2. Model management
│   ├── Construction and copying ... ConeBeamModel(...), copy_ct_model
│   ├── Parameters ................. set_params, get_params
│   └── Device control ............. configure_devices, MBIRTORCH_NUM_DEVICES
├── 3. Preprocessing
│   ├── Scanner readers ............ get_sino_and_model, load_scans_and_params
│   ├── Sinogram correction ........ scan_to_sino, correct_zinger_pixels
│   ├── Beam hardening and metal ... BH_correction, segment_plastic_metal
│   └── Model-using corrections .... align_sino_views
├── 4. Utilities
│   ├── Weights .................... gen_weights
│   ├── Synthetic data ............. generate_demo_data
│   ├── File input/output .......... save_recon_hdf5, export_recon_hdf5
│   ├── Viewing .................... slice_viewer
│   └── General .................... median_filter3d, stitch_arrays
├── 5. Differentiable projectors ... forward_project_differentiable
└── 6. Application packages ........ hsnt, vcls
```

Each category has one input/output rule:

1. **Reconstruction**: numpy / tensor / divided GPU form in; numpy out by default;
   divided form on request (`output_sharded=True`).  The specialized reconstructions
   always return numpy, because they exist to keep the full volume off the GPUs.
2. **Model management**: these functions take and return parameters -- scalars,
   tuples, and strings -- not data arrays.  Construction touches no GPU.
3. **Preprocessing**: numpy in, numpy out; GPU use is internal.
4. **Utilities**: array utilities return the form they were given; file functions
   read and write host numpy.
5. **Differentiable projectors**: tensor in, tensor out, gradients preserved, single
   device enforced.
6. **Application packages**: numpy in, numpy out.  Like preprocessing, an
   application package may divide its work across the GPUs internally.

Scalars returned to users are python floats.


# OPEN ISSUES

Each issue below names a place where the code does not yet follow the rules
above, with a proposed solution.  The detailed tables mark the affected rows
as "open issue N".

**Issue 1 (FIXED 2026-08-16): `interpolate_defective_pixels` accepted only a torch tensor.**

**Issue 2 (FIXED 2026-08-21): `stitch_arrays` mishandles mixed inputs.**
It now raises a named error in both cases: tensors on more than one device,
and any element in the divided GPU form.

**Issue 3 (FIXED 2026-08-21): the saved-file format tag names the old package.**
New files carry `'mbirtorch_preprocessing_v1'`; the loader accepts both tags
and files with no tag, and rejects an unknown tag by name.

**Issue 4 (FIXED 2026-08-21): the FBP theory derivation has no pointer.**
The derivation download link is at the top of the documentation's theory
page (`theory.rst`).

**Issue 5 (FIXED 2026-08-16): `denoise` does not yet join the automatic GPU choice.**
The thresholds were measured (both are sentinels: splitting never paid up to
a billion voxels) and `denoise` now calls the device policy
(entry_point_plan.md §9.9).

**Issue 6 (FIXED 2026-08-21): two helpers can refuse a problem they could handle.**
`prepare_sino_for_devices` and `compute_hessian_diagonal` now pass the
direct workload to the device policy.  The layout is still sized for a full
reconstruction when one fits; on a problem too large for any full
reconstruction, the check that can refuse is made against the helper's own
smaller footprint, and a later full `recon` on such a layout re-runs the
memory check rather than reusing it.  One recorded residue: a standalone
`compute_hessian_diagonal` with default indices back-projects the unmasked
grid, which the direct plan under-prices by up to 1.29x on that fallback
path; noted for a ledger follow-up.

**Issue 7 (FIXED 2026-08-21): reconstruction method names.**
The reconstruction methods are renamed for consistency with `recon`:
`recon_direct`, `recon_split_sino`, `recon_fbp`, and `recon_fdk`, and the
iteration engine is private as `_vcd_recon`.  No aliases were kept, since
the package has not had a release.

# DETAILED DESCRIPTIONS

The sections below follow the taxonomy above.  Each table lists every public
function in its subcategory.

## 1. Reconstruction (model methods)

### Projection, iterative, direct, specialized, denoising

| Function | Primary input | Accepted forms | Default output | `output_sharded=True` |
|---|---|---|---|---|
| `recon_simple_parallel` / `recon_simple_cone` (module functions) | sinogram | numpy / tensor | (numpy recon, recon_dict) | — |
| `forward_project` | recon | numpy / tensor / Shards | numpy sinogram | device-form sinogram |
| `back_project` | sinogram | numpy / tensor / Shards | numpy recon | device-form recon |
| `recon` | sinogram | numpy / tensor / Shards | (numpy recon, recon_dict) | (device-form recon, dict) |
| `prox_map` | prox_input, sinogram | numpy / tensor / Shards | (numpy recon, dict) | (device-form recon, dict) |
| `recon_direct` / `recon_fbp` / `recon_fdk` | sinogram | numpy / tensor / Shards | numpy recon | device-form recon |
| `recon_split_sino` | sinogram | numpy / tensor (tensor converted to numpy at entry) | (numpy recon, dict) — no `output_sharded` kwarg | — |
| `recon_plastic_metal` | sinogram | numpy / tensor (tensor converted to numpy at entry) | (numpy recon, dict) — no `output_sharded` kwarg | — |
| `denoise` | image | numpy / tensor / Shards | (numpy image, dict) | (device form, dict) |

### Internal reconstruction functions

| Function | Primary input | Accepted forms | Default output | `output_sharded=True` |
|---|---|---|---|---|
| `_vcd_recon` | sinogram | numpy / tensor / Shards | **always device form** + losses — no gather, no kwarg | — |
| `compute_hessian_diagonal` | weights (optional) | numpy / tensor / Shards | numpy | device form |
| `prepare_sino_for_devices` | sinogram | numpy / tensor / Shards | **always device form** (that is its purpose); a pair when weights given | — |
| `sparse_forward_project` / `sparse_back_project` | voxel values / sinogram | tensor / Shards | **mirrors the input form** — the one true form-mirroring pair on this surface | — |

## 2. Model management

No data arrays cross this surface; inputs and outputs are scalars, tuples,
strings, dicts, and models.

| Function | Input | Output |
|---|---|---|
| `ParallelBeamModel`, `ConeBeamModel`, `TranslationModel`, `MultiAxisParallelModel` | shapes and geometry parameters | model |
| `get_ct_model` | geometry name and parameters | model |
| `build_model` | the dicts from `get_all_params` | model |
| `copy_ct_model` | model (+ parameter overrides) | new model; copies an explicitly configured device layout, leaves an automatic model automatic |
| `set_params` | keyword parameters | — |
| `get_params` / `print_params` | parameter name(s) | values / printed text |
| `get_all_params` | — | three dicts (constructor, optional, regularization) |
| `auto_set_recon_geometry` | — | — (sets recon parameters) |
| `scale_recon_shape` | scale factors | pixels added per axis |
| `configure_devices` | device count or list | — (fixes the model's layout) |
| `MBIRTORCH_NUM_DEVICES` (env variable) | count | caps the device count process-wide |

## 3. Preprocessing

Numpy in, numpy out throughout; GPU use is internal.  The functions that use
internal GPU batching accept two optional arguments: `batch_size` (views per
GPU call) and `devices` (which GPUs to use; `None` means all permitted).  The
"GPU batching" column below says which functions those are.

### Scanner readers (nsi, zeiss, zeiss_tct, pymbir)

| Function | Input | Output | GPU batching |
|---|---|---|---|
| `get_sino_and_model` | string = path | numpy sinogram, model (zeiss_tct also returns numpy weights) | yes, internally (no `devices=` argument today) |
| `load_scans_and_params` | string = path | numpy scan arrays, parameter dict | no |

### Sinogram computation and correction

| Function | Input | Output | GPU batching |
|---|---|---|---|
| `scan_to_sino` | numpy scans | numpy sinogram | yes |
| `compute_sino_transmission` | numpy scans | numpy | yes |
| `correct_det_rotation` | numpy | numpy | yes |
| `downsample_view_data` | numpy scans | numpy | yes |
| `crop_view_data` | numpy scans | numpy | no |
| `correct_zinger_pixels` | numpy | numpy | yes |
| `correct_background_offset` | numpy | numpy | no |
| `interpolate_defective_pixels` | numpy | numpy | no |
| `remove_all_stripe` / `remove_stripe_fw` / `remove_sino_offset` | numpy | numpy | no (CPU threads for the stripe removals) |
| `detect_blank_margins` | numpy | margin counts (ints) | no |

### Beam hardening and metal

| Function | Input | Output | GPU batching |
|---|---|---|---|
| `BH_correction` | numpy | numpy | yes |
| `fit_beam_hardening_curve` / `fit_inverse_beam_hardening_curve` | numpy | numpy coefficients | no |
| `apply_beam_hardening_curve` / `apply_inverse_beam_hardening_curve` | numpy | numpy | no |
| `segment_plastic_metal` | numpy / tensor / Shards (recon volume) | masks in the same form as the input | no; follows the input's placement |
| `multi_threshold_otsu` | numpy / tensor / Shards | thresholds (python floats) | no; follows the input's placement |
| `correct_sino_plastic_metal` | model + numpy / tensor / Shards | always numpy | no; uses the model's GPU layout |

### Model-using corrections

| Function | Input | Output | GPU batching |
|---|---|---|---|
| `estimate_sino_view_offset` | model + numpy / tensor | numpy shifts (scalars per view) | no; uses the model's GPU layout |
| `align_sino_views` | model + numpy / tensor | numpy | no; uses the model's GPU layout |

## 4. Utilities

Array utilities return the form they were given; file functions read and write
host numpy.

### Weights

| Function | Input | Output |
|---|---|---|
| `gen_weights` | numpy / tensor sinogram | weights in the same form and place as the input.  No model argument. |
| `gen_weights_mar` | model + numpy | numpy |
| `gen_huber_weights` | numpy / tensor | same form as the input |

### Synthetic data

| Function | Input | Output |
|---|---|---|
| `generate_demo_data` | sizes and options (`devices=` pins the GPUs) | numpy phantom, numpy sinogram, parameter dict |
| `generate_3d_shepp_logan_reference` | shape | numpy |
| `generate_3d_shepp_logan_low_dynamic_range` | shape | numpy |
| `gen_translation_phantom` | shape and options | numpy |

### File input/output

| Function | Input | Output |
|---|---|---|
| `save_recon_hdf5` / `load_recon_hdf5` | path + numpy / tensor / Shards | file / numpy + dict |
| `export_recon_hdf5` / `import_recon_hdf5` | path + numpy / tensor / Shards | file / numpy + dict |
| `save_data_hdf5` / `load_data_hdf5` | path + numpy / tensor / Shards | file / numpy + dict |
| `save_cone_preprocessing` / `load_cone_preprocessing` | path + numpy arrays and model | file / numpy arrays + model parameters |
| `read_tif_img` / `read_tif_stack_dir` | path | numpy float32 |
| `download_and_extract` | url | path string |

### Viewing

| Function | Input | Output |
|---|---|---|
| `slice_viewer` | numpy / tensor volumes | interactive display; returns nothing |

### General

| Function | Input | Output |
|---|---|---|
| `median_filter3d` | numpy / tensor | same form as the input |
| `stitch_arrays` | list of numpy / tensor | one array (issue 2 fixed: mixed devices and the divided form now raise) |
| `apply_cylindrical_mask` | numpy / tensor / Shards | same form as the input |
| `clear_cache` | — | — (empties the on-disk compile cache) |
| `get_memory_stats` | — | per-device memory report |

## 5. Differentiable projectors

Tensor in, tensor out, gradients preserved.  A model using more than one device
is refused with a clear error.

| Function | Input | Output |
|---|---|---|
| `forward_project_differentiable` | model + tensor recon | tensor sinogram (differentiable) |
| `back_project_differentiable` | model + tensor sinogram | tensor recon (differentiable) |
| `TorchProjector` | model | a torch module wrapping both directions |

## 6. Application packages

Numpy in, numpy out.  Like preprocessing, an application package may divide its
work across the GPUs internally.

| Function | Input | Output |
|---|---|---|
| `hsnt.dehydrate` / `hsnt.rehydrate` | numpy hyperspectral data | numpy + basis / numpy |
| `hsnt.hyper_denoise` | numpy | numpy |
| `hsnt.generate_hyper_data` | sizes and options | numpy arrays |
| `hsnt.create_hsnt_metadata` | parameters | dict |
| `hsnt.export_hsnt_data_hdf5` / `hsnt.import_hsnt_data_hdf5` | path + numpy | file / numpy |
| `vcls.get_opt_views` | model + numpy | view indices (ints) and a score |
| `vcls.show_image_with_projection_rays` | model + numpy | display |
