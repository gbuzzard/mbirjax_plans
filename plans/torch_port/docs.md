# Sphinx docs port — findings for the code session

## Status

The mbirjax docs are being ported to mbirtorch in `mbirtorch/docs`.  The build
scaffold, the background pages, the user-API pages, and the two sharding pages
are done.  The docs build succeeds with 15 warnings, all of them references to
pages not yet written.  This page records what the port found about the mbirtorch
PACKAGE, not about the documentation.  Every finding below needed a decision in
the code session, because every fix lay outside `docs/`.

### TBD — pages written against work still in progress

Four things in the shipped pages are expected to change.  Each was written against
the code as it stood on 2026-08-07.

**The `configure_devices` interface.**  `usr_multi_gpu.rst` documents two call
forms, a device count and an explicit device list, because those are the two the
signature accepts today.  A tuple form indexing into the visible devices is
planned, matching mbirjax.  The examples in that page need a third form when it
lands.

**An automatic device path.**  The same page states that multi-device is opt-in,
and gives three reasons drawn from decision 4 below.  A zero-effort path that
widens the device count automatically may be added after all.  If it is, the
"Turning it on" section reverts toward mbirjax's wording, and so do the
corresponding paragraphs in `overview.rst`, `advanced_features.rst`, and
`usr_tomography_model.rst`.

**The scaling numbers.**  The measured table in `usr_multi_gpu.rst` comes from the
phase 4 gate matrix and is expected to improve.  The entry most likely to change
is the 512 cell, where four devices currently measure about three times slower
than one.  That table is the only place a user sees the number, so it must be
re-read against the current gate matrix before any release.

**The denoiser.**  `dev_sharding_overview.rst` states that the QGGMRF denoiser is
single-device only, and calls sharding it possible future work.  Whether the
denoiser is ever sharded is undecided.  The affected text is one paragraph in that
page's "Single- and multi-device paths" section, plus the scope note at the top.

The evidence in both sharding pages may need adjusting as measurements land.  The
provenance of every number is recorded in the "Measured evidence in the sharding
pages" section near the end of this page.  That section is the place to start when
a measurement changes.

### On hold — pages not yet written

Six pages remain.  Two are blocked on the code session: `dev_projector_kernels.rst`
and `dev_api.rst`, together with the `new_model_template.py` code sample.  All
three depend on kernel work in progress.  Four are held at the user's request:
`install.rst`, `dev_maintenance.rst`, `dev_performance_dashboard.rst`, and the demo
portion of `demos_and_faqs.rst`.  The FAQ portion of that last page is mostly
portable and was scoped as the smaller half of the work.

The port is deliberately mechanical.  Pages are copied from mbirjax and changed
only where a sentence would otherwise be false for mbirtorch.  That constraint is
what makes this page useful: each divergence below is a place where mbirtorch and
mbirjax genuinely differ, surfaced by copying rather than by searching for it.

## The largest finding: `__all__` changes what autodoc renders

An explicit `__all__` in `mbirtorch/__init__.py` makes autodoc document every
name it lists.  `mbirjax/__init__.py` star-imports its submodules and declares no
`__all__`.  Autodoc then classifies every name in mbirjax as an imported member
and documents none of them.  The two packages therefore render very differently
from the same directive.

The size difference is large.  Both `usr_api.rst` files carry the same
`automodule` directive with `:members:` and `:undoc-members:`.  mbirjax's
rendered page is 16.7 KB.  mbirtorch's first build of the same page was 155 KB.
The extra 138 KB is the internal VCD and qGGMRF helper functions, printed in full
on the page a new user is pointed at first.

The dumped page also produced most of the build warnings.  Those helper
docstrings cross-reference private functions such as `_get_estimate_of_recon_std`
and `_sharding.run_per_device`.  Rendering the helpers dragged those references
onto the page, where they could not resolve.  The first mbirtorch build reported
39 warnings; mbirjax's own build reports 1.

The docs side is already handled, and the handling is recorded in the file.  The
`:members:` and `:undoc-members:` options were dropped from that one directive on
`usr_api.rst` and `usr_api_overview.rst`.  A `DIVERGENCE(automodule members)`
comment at each site explains why and states the condition for restoring them.
The rendered page is now 17.3 KB, and 18 of the 20 real warnings are gone.

The package question is separate, and it is the reason this section exists.
`mbirtorch.__all__` lists nine names that look internal: `gen_full_indices`,
`gen_pixel_partition`, `gen_set_of_pixel_partitions`, `gen_partition_sequence`,
`get_2d_ror_mask`, `qggmrf_gradient_and_hessian_at_indices`, `get_b_from_nbr_wts`,
`b_tilde_by_definition`, and `qggmrf_loss`.  Note that mbirjax exports the same
names at package level through its star imports.  The difference is that
mbirtorch DECLARES them, and a declared name is a promise to users.  If the
intended public surface is narrower, narrowing `__all__` would let the docs
restore the mbirjax directive verbatim.

## Absent API

Twenty-one names documented in mbirjax do not exist in mbirtorch.  Each was
checked by import, not by grep.  Their documentation is commented out in place on
the live pages, with a `PENDING(<topic>)` marker naming what must be ported to
restore it.  The pages that are pure autodoc against absent modules are staged
whole under `docs/source/_pending/`.

The absent names group as follows:

- `TomographyModel`: `get_all_params`, `get_recon_dict`, `device_summary`,
  `save_recon_hdf5`, `load_recon_hdf5`.
- `ConeBeamModel`: `split_sino_recon`.
- Denoising: `median_filter3d`.
- Weights: `gen_weights_mar`.
- Parameters: `use_gpu`.
- `utilities`: `stitch_arrays`, `get_ct_model`, `copy_ct_model`, `build_model`,
  `download_and_extract`, `save_data_hdf5`, `load_data_hdf5`,
  `export_recon_hdf5`, `import_recon_hdf5`, `generate_demo_data`,
  `generate_3d_shepp_logan_reference`, `gen_translation_phantom`.

The `utilities` group is the one worth a decision.  Twelve of the twenty-one
absent names are in it, and they leave the Utilities page with three entries:
the slice viewer, `gen_weights`, and one phantom generator.  Saving and loading a
reconstruction is entirely absent, across both `TomographyModel` and `utilities`.

Three markers are coupled and must be restored together.  The `_SaveLoadDocs`
label is defined inside the commented block in `usr_tomography_model.rst`.
`usr_utilities.rst` and `usr_api_overview.rst` each reference that label from
their own commented blocks.  Restoring any one alone will break the build with an
undefined label.  This coupling is recorded in `docs/source/_pending/README.rst`.

## Missing docstrings

Four documented methods have no docstring, so their pages render a signature with
no body.  They do not fall into one pattern, and the four split into three cases.

`direct_recon` is undocumented in all three classes that define it:
`TomographyModel`, `ParallelBeamModel`, and `ConeBeamModel`.  No override
supplies text.  This one is a genuine gap rather than an artifact of inheritance.

`auto_set_recon_geometry` and `get_magnification` are undocumented only on
`TomographyModel`.  Both geometry subclasses override them with docstrings.  The
base declarations are the stubs, and the pages document the base.

`ParameterHandler.print_params` is the real implementation, with no override
anywhere.  It simply has no docstring.

## A dangling cross-reference

The docstring of `TomographyModel.forward_project` refers to
`:func:`vcd_utils.get_2d_ror_mask``, at `mbirtorch/tomography_model.py:1054`.
That function is documented on no page, so the reference cannot resolve.  It
accounts for both of the two real warnings that remain in the mbirtorch build.

mbirjax's docstring for the same method carries no such reference.  This one was
added during the port to mbirtorch.  Two fixes work: demote the role to a literal,
or document `get_2d_ror_mask` on the Utilities page.  The choice depends on
whether `get_2d_ror_mask` is meant to be public, which is the same question the
`__all__` section raises.

## Multi-GPU is opt-in, not automatic

`configure_devices` defaults to `num_devices=1`.  Nothing widens that default
outside the test suite.  A plain `model.recon(sinogram)` on a machine with eight
GPUs therefore uses one GPU.

mbirjax behaves differently, and its documentation says so in three places.  Its
pages state that a reconstruction is spread across available GPUs automatically,
with no change to the user's script.  Those sentences were rewritten in
`overview.rst`, `advanced_features.rst`, and `usr_tomography_model.rst` to name
`configure_devices(num_devices=n)` as an explicit step.

This is recorded as a divergence, not as a defect.  The remaining question for the
code session is whether the opt-in default is intended to persist.  The answer
shapes the `usr_multi_gpu.rst` rewrite, which is step 4 of the port.

## Two smaller items

`ParameterHandler` is not exported at package level in mbirtorch.  Its methods are
documented as `mbirtorch.parameter_handler.ParameterHandler.*`.  The class also
had to be given an explicit `autoclass` directive, because `:show-inheritance:`
on every model class needs it as a cross-reference target.  mbirjax obtains that
target for free from its package-level `automodule`.

`slice_viewer` resolves through `view_utils`, not `viewer`.  Both modules define a
function of that name, and they are not the same object.  `mbirtorch.slice_viewer`
is the one in `view_utils`, so that is the path the docs use.

## Build state

The mbirtorch build succeeds with 21 warnings.  Nineteen are references to the
five developer pages that step 4 will create, and they will clear on their own.
The remaining two are the dangling cross-reference described above.

The comparison build is the useful number to keep.  mbirjax's docs build with 1
warning, using its own environment.  That build is the control that made the
`__all__` finding visible, and rerunning it is the cheapest way to tell an
mbirtorch problem from an inherited one.

## Decisions and instructions for the package changes

This section answers the questions raised above and gives the executable
instructions.  The decisions were made in the code session on 2026-08-07 after
reading the package source and the mbirjax reference.  The instructions are
self-contained: every edit names its file, and every docstring to add is quoted
in full.  Follow the standing repo rules: stage each edited file by name with
`git add`, and never run `git commit`.

### 1. Narrow `__all__` (decision: the nine names are not public)

The nine internal-looking names come out of `__all__`.  The public surface of
mbirtorch should match the surface mbirjax documents, and mbirjax documents
none of the nine.  The import lines in `mbirtorch/__init__.py` stay exactly as
they are.  Keeping the imports keeps attribute access working, so
`mbirtorch.gen_full_indices` and the other eight remain reachable; seven test
files use that spelling today, and none of them breaks.

Edit `mbirtorch/__init__.py` and replace the `__all__` list with:

```python
__all__ = [
    "ParallelBeamModel", "ConeBeamModel", "TomographyModel", "QGGMRFDenoiser",
    "TorchProjector", "forward_project_differentiable",
    "back_project_differentiable", "gen_weights",
    "generate_3d_shepp_logan_low_dynamic_range", "clear_cache",
    "get_memory_stats", "SliceViewer", "VolumeStack", "slice_viewer",
]
```

The restoration condition in the `DIVERGENCE(automodule members)` comments is
now met.  Restore `:members:` and `:undoc-members:` on the directives in
`usr_api.rst` and `usr_api_overview.rst`, remove the two DIVERGENCE comments,
and rebuild.  The rendered `usr_api` page should land near mbirjax's 16.7 KB.

### 2. Demote the dangling cross-reference

With `get_2d_ror_mask` non-public, the reference demotes to a literal.  In
`mbirtorch/tomography_model.py` (the `forward_project` docstring, near line
1054), change:

```
reconstruction (the pixels selected by the ROR mask; see
:func:`vcd_utils.get_2d_ror_mask`).
```

to:

```
reconstruction (the pixels selected by the ROR mask; see
``vcd_utils.get_2d_ror_mask``).
```

This clears the last two real build warnings.

### 3. Add the four missing docstrings

Each docstring below is the mbirjax text with two adaptations: "jax array"
becomes "tensor", and the `output_sharded` wording matches mbirtorch's
device-form convention.  Insert each verbatim.

**3a. `TomographyModel.direct_recon`** (`mbirtorch/tomography_model.py`, the
base stub near line 192).  Keep the `raise NotImplementedError` body and add:

```python
        """
        Do a direct (non-iterative) reconstruction, typically using a form of
        filtered backprojection.  The implementation details are geometry
        specific, and direct_recon may not be available for all geometries.

        Args:
            sinogram (numpy or tensor): 3D sinogram data with shape
                (num_views, num_det_rows, num_det_channels).
            filter_name (string or None, optional): The name of the filter to
                use.  Defaults to None, in which case the geometry-specific
                method chooses a default, typically 'ramp'.
            output_sharded (bool, optional): If False (default), return a
                numpy array.  If True, return the internal device form
                (slice shards on a multi-device model; on a single-device
                model the output is the same tensor either way).

        Returns:
            recon (numpy or tensor): The reconstructed volume.
        """
```

**3b. The subclass overrides of `direct_recon`**
(`mbirtorch/parallel_beam.py` near line 339 and `mbirtorch/cone_beam.py` near
line 764).  Autodoc inherits the base docstring once 3a lands, but the
inherited text cannot name the algorithm.  Add one short docstring to each
override.  Parallel:

```python
        """Direct reconstruction by filtered backprojection (FBP); equivalent
        to :meth:`fbp_recon`.  See :meth:`TomographyModel.direct_recon` for
        the argument and return conventions."""
```

Cone:

```python
        """Direct reconstruction by the FDK algorithm; equivalent to
        :meth:`fdk_recon`.  See :meth:`TomographyModel.direct_recon` for the
        argument and return conventions."""
```

**3c. The base geometry stubs** (`mbirtorch/tomography_model.py`,
`get_magnification` near line 157 and `auto_set_recon_geometry` near line
163).  Keep the `raise NotImplementedError` bodies and add:

```python
        """Return the magnification for this geometry.  Each geometry model
        defines this; parallel beam returns 1.0."""
```

```python
        """
        Set the automatic value of the recon shape and voxel pitch using the
        geometry parameters and sinogram shape.  Each geometry model defines
        this.

        Note: This function should be run after changing geometry parameters
        such as ``delta_det_channel``.  It will set reconstruction parameters
        such as ``recon_shape`` and ``delta_voxel`` to reasonable values.

        Args:
            no_compile (bool, optional): If True, do not rebuild the
                projectors.  Defaults to False.
            no_warning (bool, optional): If True, do not issue warnings.
                Defaults to False.
        """
```

**3d. `ParameterHandler.print_params`** (`mbirtorch/parameter_handler.py`,
near line 149).  The mbirtorch implementation also lost a behavior the mbirjax
docstring describes: mbirjax hides the view-params entry below verbosity 3, so
a 1024-view model does not dump its full angle array.  Port the guard along
with the docstring.  Replace the method with:

```python
    def print_params(self):
        """
        Print the current parameter values in the model.

        This method prints all parameters stored in the model's internal
        dictionary.  If the model's verbosity level is less than 3, the view
        parameter array (e.g. the angles) is summarized rather than printed
        in full.

        Example:
            >>> ct_model = mbirtorch.ParallelBeamModel(sinogram_shape, angles)
            >>> ct_model.set_params(sharpness=0.7)
            >>> ct_model.print_params()
        """
        verbose, view_params_name = self.get_params(['verbose',
                                                     'view_params_name'])
        print('----')
        for key, entry in self.params.items():
            if verbose < 3 and key == view_params_name:
                val = np.asarray(entry.val)
                print(f'{key} = array(shape={val.shape}, '
                      f'dtype={val.dtype})')
            else:
                print(f'{key} = {entry.val}')
        print('----')
```

Check the imports at the top of `parameter_handler.py`: the replacement uses
`np`, so add `import numpy as np` if the module does not already import it.

### 4. Multi-GPU: the default WILL change to inherit mbirjax's auto-spread

**Decision revised by Greg 2026-08-07** (superseding the earlier
opt-in-persists decision recorded here).  mbirtorch will adopt the mbirjax
policy: a reconstruction spreads across the available GPUs by default, with
`configure_devices(num_devices=1)` remaining the way to pin a run to one
device.  The reasoning: a user who has four GPUs and silently gets one — or
gets a late out-of-memory failure — is worse off than a user who pays a
small device-count-dependent float difference (measured to decay from
6.1e-3 at 3 iterations to 8.8e-4 at 10) or a modest time penalty on small
problems.  The change ships together with a memory preflight check, whose
priority is raised for exactly this reason; the charter is
`current_plans.md` item 2.

For the docs, until the code lands: the three sentences already rewritten
to name `configure_devices` as an explicit step remain CORRECT for the
current package and stay as they are.  Do not write `usr_multi_gpu.rst`
against either behavior yet — stage it as `PENDING(auto-spread)` with a
one-line note that the default is changing, and write it when the flip
lands.  When it does, its content: the auto-spread default, the preflight
that guards it, `configure_devices(num_devices=1)` as the reproducibility
pin, and one sentence stating that results can differ slightly with device
count and that the difference decays with iterations.

### 5. Classify the absent API for the PENDING markers

The twenty-one absent names fall in three buckets.  The docs treatment
differs by bucket.

**Planned — keep the PENDING markers.**  The HDF5 save/load family
(`save_recon_hdf5`, `load_recon_hdf5`, `save_data_hdf5`, `load_data_hdf5`,
`export_recon_hdf5`, `import_recon_hdf5`), `get_recon_dict`, `get_all_params`,
the model factories (`get_ct_model`, `copy_ct_model`, `build_model`), the demo
utilities (`download_and_extract`, `generate_demo_data`,
`generate_3d_shepp_logan_reference`), `stitch_arrays`, and `median_filter3d`.
These are on the port roadmap.

**Blocked on an unported geometry or module — keep PENDING, name the
blocker.**  `gen_translation_phantom` (translation geometry) and
`gen_weights_mar` (the MAR module).

**Replaced — remove the PENDING markers and the commented blocks.**  The
`use_gpu` parameter is replaced by `configure_devices`; `device_summary` is
replaced by `get_memory_stats`.  Where a replaced name appears in prose, name
the replacement.

**AMENDED 2026-08-07:** `split_sino_recon` moves from this bucket back to
PLANNED.  The application-compatibility review found the nsi split-sinogram
demo calls it directly, and it is a capacity feature in its own right (it
nearly doubles the feasible cone recon size at a fixed GPU count), so it
will be ported with the full mbirjax logic rather than dropped.  Flip its
`REPLACED(...)` marker to `PENDING(split_sino_recon)`.

### 6. Confirmed as intended (no change)

Two of the smaller findings need no code edit.  The `slice_viewer` pair is
deliberate: `viewer.py` is package-independent so the identical file can later
serve mbirjax, and `view_utils.slice_viewer` is the mbirtorch-facing wrapper
that converts device tensors and serializes recon dicts on the way in.  The
docs correctly target `view_utils`.  `ParameterHandler` likewise stays
unexported at package level; the explicit `autoclass` workaround stands.

### 7. Verify

Run the test suite from the mbirtorch env after the edits; the expected
result is the current baseline (279 passed, 52 skipped locally).  Then
rebuild the docs.  The expected warning count is 19, all of them references
to the step-4 developer pages: the two cross-reference warnings clear with
item 2, and item 1's restored directives must not reintroduce the helper
dump (the page-size check against mbirjax's 16.7 KB is the guard).

## Outcome of the instructions (docs session, 2026-08-07)

All seven items were applied.  Six landed as written.  Item 1 landed only in
part, because its stated mechanism does not reach its stated goal.  The section
below records what was measured, so the next attempt starts from evidence rather
than from the same model.

### Item 1 did not restore the mbirjax directive

Narrowing `__all__` was applied and is kept.  The list now holds fourteen names,
and the nine internal helpers are gone from it.  The imports are unchanged, so
attribute access still works and the test suite is unaffected.

Restoring `:members:` and `:undoc-members:` was applied, measured, and reverted.
The guard in item 7 is what reverted it.  With the options restored, the rendered
`usr_api` page measured 140 KB against mbirjax's 16.7 KB, and the build reported
28 warnings against the expected 19.  Item 7 names the page-size check as the
guard on item 1, so the guard failed and the change did not stand.

The premise behind item 1 was that the nine helpers caused the 155 KB page.  The
helpers were a small part of it.  Removing all nine moved the page from 155 KB to
140 KB, a saving of 15 KB.  The remaining 138 KB is something else.

That something else is class methods.  `automodule` with `:members:` documents
each class AND every public method of that class.  The seven classes in `__all__`
carry 250 public methods between them: `ParallelBeamModel` 42, `ConeBeamModel`
45, `TomographyModel` 39, `QGGMRFDenoiser` 41, `TorchProjector` 54, `VolumeStack`
20, and `SliceViewer` 2.  No narrowing of `__all__` reduces that count, because
the classes are the public surface and must stay in the list.

mbirjax renders 16.7 KB for a different reason than a narrow public surface.  It
renders that because it has NO `__all__` at all, so autodoc classifies every name
as an imported member and documents zero of them.  Any non-empty `__all__`
containing the model classes produces a page near 140 KB.  Matching mbirjax's
rendered page therefore requires dropping the options, deleting `__all__`
entirely, or a third mechanism such as an explicit `:exclude-members:` list.

Restoring the options also re-raised eight warnings that the dropped options had
been suppressing.  They come from method docstrings that reference private
helpers: `_get_estimate_of_recon_std` in `auto_set_sigma_x` and
`auto_set_sigma_prox`, `_sharding.run_per_device` in `create_vcd_subset_updater`,
and `get_psf_radii` in `ConeBeamModel.get_psf_radius`.  These are latent today
because the directive does not render those methods.  Any future decision to
render class members will surface all eight, and they would need fixing first.

The `DIVERGENCE(automodule members)` comments in `usr_api.rst` and
`usr_api_overview.rst` now carry these numbers, so the next reader does not repeat
the measurement.

### Items 2, 3, 5, 6 and 7 landed as written

Item 2 demoted the `get_2d_ror_mask` role to a literal.  Both cross-reference
warnings cleared.

Item 3 added all four docstrings plus the two subclass overrides.  The
`print_params` replacement was checked against its preconditions before it was
applied.  Two preconditions hold: `parameter_handler.py` already imports numpy,
and `view_params_name` is set by every geometry constructor.  One risk was
considered and dismissed: `view_params_name` is not in the defaults dict, so a
model lacking it would raise in `get_params`.  A bare `TomographyModel` cannot be
constructed at all, since its constructor raises `NotImplementedError`, so every
reachable instance is a geometry model or the denoiser, and all of them set the
parameter.  The denoiser sets it to the string `'None'`, which matches no key and
prints every parameter, which is correct for a model with no view array.

Item 5 was applied with a new marker.  A `PENDING(...)` marker means the content
is expected back.  The three replaced names are recorded as `REPLACED(...)`
instead, each naming its replacement: `use_gpu` by `configure_devices`,
`device_summary` by `get_memory_stats`, and `split_sino_recon` by the
multi-device engine.  The blocked names now name their blockers.  Both
conventions are documented in `docs/source/_pending/README.rst`.

Item 7 passes at the reverted state.  The test suite reports 279 passed and 52
skipped, matching the stated baseline.  The docs build reports 19 warnings, all
of them references to the step-4 developer pages, and `usr_api` renders at
17.3 KB.

### One question back to the code session

Item 1's goal needs restating before it can be met.  Three options exist, and
they differ in what the User API page becomes.  Option one keeps the current
state: the options stay dropped, the page stays at 17.3 KB, and it matches
mbirjax.  Option two accepts a richer page than mbirjax's: the options return,
the page runs to 140 KB, and the eight private-helper references must be fixed
first.  Option three keeps the options and adds an explicit `:exclude-members:`
list to hold the page down, which is the most work to maintain.

The docs session recommends option one, on the grounds that the port is meant to
be mechanical and a 140 KB wall of methods is worse for a new user than the
per-class pages that already document the same methods with prose around them.

### Decision on the question above (code session, 2026-08-07)

Option one is confirmed: the `:members:` and `:undoc-members:` options stay
dropped.  The User API page stays at 17.3 KB and matches mbirjax's rendered page.
The narrowed `__all__` stays as well, on its own merits as the declared public
surface.

Two consequences follow, and both are deliberate.  The eight private-helper
references in method docstrings stay latent, since the directive never renders
those methods; they become real work only if a future change renders class
members.  The `DIVERGENCE(automodule members)` comments stay in place as the
record of why the directive differs from mbirjax's.

## Measured evidence in the sharding pages (docs session, 2026-08-07)

`usr_multi_gpu.rst` and `dev_sharding_overview.rst` are drafted.  Every number in
them is an mbirtorch measurement, and every mbirjax number that had no mbirtorch
counterpart was dropped rather than carried over.  This section records which is
which, so a later reader can tell a sourced claim from a removed one.

### Numbers used, with their source

The device-scaling table in `usr_multi_gpu.rst` is the gate matrix from
`phase4_findings.md`: H100, warm 3-iteration VCD, torch frame.  It reports 3.12 /
2.96 / 9.11 s at the 512 cell and 94.0 / 70.7 / 60.2 s at the 1024 cell, for one,
two, and four devices.

The band-length numbers come from the `_slice_band_length` docstring in
`tomography_model.py`, which cites the H100 gate matrix.  Two appear: a sub-band
default costs 47 to 66 percent more warm time at the two-device cells, and a
four-device 512-cell run trades 6.6 GiB down to 2.6 GiB for about 8 percent more
time.

### Claims dropped for lack of mbirtorch evidence

MBIRJAX states that reconstruction time scales nearly linearly with the device
count at large problem sizes.  That is false for mbirtorch, and the gate matrix
says so: four devices give 1.56x, not 4x.  The claim was replaced with the
measured table and an explicit warning that small problems get worse.

MBIRJAX states that peak memory per GPU scales roughly as 1/N.  The gate matrix
reports peak memory only at four devices, so there is no mbirtorch series to
support a 1/N claim.  The pages state the memory benefit structurally instead,
from the fact that the recon is sharded by slice, and give no ratio.

MBIRJAX's thread-pool section carries a four-row comparison table (thread pool,
`shard_map`, threads-through-host, `pmap`) and an accompanying "approximately 2x
on 2 GPUs" figure.  That is a JAX substrate study with no torch counterpart, so
the table and the figure were both dropped.  The torch substrate decision that
replaced it -- DTensor rejected as immature for index-heavy kernels -- is stated
without a number, since the supporting spikes live in `phase4_design.md` rather
than in a form suitable for a docs table.

MBIRJAX's cone-beam section cites an A/B test showing whole-cylinder projection
costing about 10 percent more transient memory for about 2x less time.  No
mbirtorch A/B exists, so the rationale is kept and the numbers are dropped.

### One item to confirm

The 512-cell result deserves a second look before this ships.  Four devices are
about three times *slower* than one at that size, which the phase 4 page
attributes to the per-subset band loop growing as bands times devices.  If the
band-loop work has since changed that number, `usr_multi_gpu.rst` needs updating;
the table is the only place a user sees it.
