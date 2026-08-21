## Functional Interface Proposal

**Motivation:**  The mbirtorch object-oriented interface is powerful but
can be overwhelming for some people.  

**Goal:** Develop a simple, functional interface that can let a naive
user get a recon with a single line.  Separate functions for basic parallel and cone,
no other geometries, few options. 

### Parallel beam signature and body proposal: 

(Note that cone beam would have an analogous function.)

```python
def parallel_recon(sinogram, angles, weights=None, sharpness=1, max_iterations=15):
    """
    Functional interface for a basic parallel beam reconstruction.
    For a full range of reconstruction options, use :meth:`TomographyModel.recon <mbirtorch.TomographyModel.recon>`

    To perform a filtered back projection recon (scaled to minimize the error sinogram), use `max_iterations=0'

    Args:
        sinogram (numpy or tensor): 3D sinogram data with shape
            (num_views, num_det_rows, num_det_channels).
        angles (numpy or tensor): 1D array of angles (radians) with shape (num_views, )
        weights (numpy or tensor, optional): 3D positive weights with the
            same shape as the sinogram.  Defaults to None (all 1s).
        sharpness (float, optional): sharpness parameter. Defaults to 1.0.  Set lower to for softer edges, less noise, higher for crisper edges, more noise.
        max_iterations (int, optional): maximum number of VCD iterations.  Use 0 for FBP reconstruction.

    Returns:
        (recon, recon_dict): the reconstruction volume, and a dict
        with entries 'recon_params' (per-iteration traces and settings),
        'recon_log' (the run's log text), 'notes', and
        'model_params' (a snapshot of the model parameters).
    """
    model = mbirtorch.ParallelBeamModel(sinogram.shape, angles)
    model.set_params(sharpness=sharpness)
    recon, recon_dict = model.recon(sinogram, weights=weights, max_iterations=max_iterations)
    return recon, recon_dict
```
### Parallel beam example use:  
```python
sinogram, angles, weights = get_data()
recon, recon_dict = parallel_recon(sinogram, angles, weights)
```

