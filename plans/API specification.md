# TOP-LEVEL OVERVIEW

## Recon-related functions

Public facing recon functions accept numpy / tensor / Shards and return numpy by default.  Can opt in to sharded form.  
```
forward_project, back_project, recon, prox_map, 
direct_recon (+fbp+fdk), denoise
```

Exceptions: The two below accept numpy or tensor input (a tensor is converted to numpy at entry, for robustness) and ALWAYS return numpy.  They have no `output_sharded` option: both are host-side drivers whose purpose is to keep the full volume off the devices, and a device-form output at exit would undo that.
```
split_sino_recon, recon_plastic_metal
```

## Preprocessing functions

Public facing functions return numpy arrays
```
get_sino_and_model, load_scans_and_params
```

Scalars returned to users are python floats (note compute_scaling_factor returns a float in mbirjax; keep that).

Internal pre-processing:  sinograms are sharded by view across all devices and processed in parallel. 


# MORE DETAILED DESCRIPTIONS

# Recon-related functions

## Primary user-facing functions:

| Function | Primary input | Accepted forms                                                                                            | Default output                                                                     | `output_sharded=True`     |
|---|---|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|---------------------------|
| `forward_project` | recon | numpy / tensor / Shards                                                                                   | numpy sinogram                                                                     | device-form sinogram      |
| `back_project` | sinogram | numpy / tensor / Shards                                                                                   | numpy recon                                                                        | device-form recon         |
| `recon` | sinogram | numpy / tensor / Shards                                                                                   | (numpy recon, recon_dict)                                                          | (device-form recon, dict) |
| `prox_map` | prox_input, sinogram | numpy / tensor / Shards                                                                                   | (numpy recon, dict)                                                                | (device-form recon, dict) |
| `direct_recon` / `fbp_recon` / `fdk_recon` | sinogram | numpy / tensor / Shards                                                                                   | numpy recon                                                                        | device-form recon         |
| `denoise` | image | numpy / tensor / Shards                                                                                   | (numpy image, dict)                                                                | (device form, dict)       |
| `split_sino_recon` | sinogram | numpy / tensor (tensor converted to numpy at entry)                                                       | (numpy recon, dict) — no `output_sharded` kwarg                                    | —                         |
| `recon_plastic_metal` | sinogram | numpy / tensor (tensor converted to numpy at entry)  | (numpy recon, dict) — no `output_sharded` kwarg | —                         |

## Secondary/internal functions

| Function | Primary input | Accepted forms | Default output | `output_sharded=True` |
|---|---|---|---|---|
| `vcd_recon` | sinogram | numpy / tensor / Shards | **always device form** + losses — no gather, no kwarg | — |
| `compute_hessian_diagonal` | weights (optional) | numpy / tensor / Shards | numpy | device form |
| `prepare_sino_for_devices` | sinogram | numpy / tensor / Shards | **always device form** (that is its purpose); a pair when weights given | — |
| `sparse_forward_project` / `sparse_back_project` | voxel values / sinogram | tensor / Shards | **mirrors the input form** — the one true form-mirroring pair on this surface | — |



# Preprocessing functions


## Primary user facing functions

| Function | Input                   | Output                  |
|---|-------------------------|-------------------------|
|`get_sino_and_model`| string = path   | numpy array, ct model   |
|`load_scans_and_params`| string = path | numpy arrays, dict |

## Internal functions without a model

These functions take only arrays and scalars — no `TomographyModel` argument.  All functions take numpy arrays as input and produce numpy arrays as output. However, internally they may divide the views across the GPUs for parallel processing.

| Function | Input | Output |
|---|---|---|
| `scan_to_sino` | numpy | numpy |
| `compute_sino_transmission` | numpy | numpy |
| `correct_det_rotation` | numpy | numpy |
| `downsample_view_data` | numpy | numpy |
| `correct_zinger_pixels` | numpy | numpy |
| `BH_correction` | numpy | numpy |


## Internal functions with a model

These functions take a `TomographyModel` argument because they need the scanner geometry (typically to forward- or back-project).

| Function | Primary input | Output |
|---|---|---|
| `correct_sino_plastic_metal` | numpy / tensor / Shards (via `prepare_sino_for_devices`) | always host numpy |
| `estimate_sino_view_offset` | numpy / tensor | numpy shifts (scalars per view) |
| `align_sino_views` | numpy / tensor | always numpy |
| `gen_weights_mar` | numpy | numpy |

## Internal functions on reconstruction volumes (no model)

| Function | Primary input | Output |
|---|---|---|
| `segment_plastic_metal` | numpy / tensor / Shards (recon volume) | masks in the same form as the input |
| `apply_cylindrical_mask` | numpy / tensor / Shards | same form as the input |

# Utilities

| Function | Input | Output |
|---|---|---|
| `gen_weights` | numpy / tensor sinogram | weights in the same form and place as the input (numpy in, numpy out; tensor in, tensor out on the same device).  No model argument. |
| `median_filter3d` | numpy / tensor | same form as the input |

