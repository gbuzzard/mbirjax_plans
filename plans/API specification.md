# Recon-related functions

Public facing recon functions accept numpy / tensor / Shards and return numpy by default.  Can opt in to sharded form.  
```
forward_project, back_project, recon, prox_map, 
direct_recon (+fbp+fdk), denoise
```

Exceptions: The two below accept numpy input only, and have numpy output by default but can opt-in to sharded output
```
split_sino_recon, recon_plastic_metal
```

# Preprocessing functions

Public facing functions return numpy arrays
```
get_sino_and_model, load_scans_and_params
```

Scalars returned to users are python floats (note compute_scaling_factor returns a float in mbirjax; keep that).

Internal pre-processing:  sinograms are sharded by view across all devices and processed in parallel. 


