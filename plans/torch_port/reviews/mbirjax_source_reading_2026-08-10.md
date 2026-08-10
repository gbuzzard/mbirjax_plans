# mbirjax source reading (2026-08-10, evening)

Archived verbatim from the session's source-reading agent, so the
file-and-line evidence behind the remedy memo's §8 survives outside
the session scratchpad.  The questions were the repo owner's three,
asked after the mg9 instrument read; the memo's §8 is the ruling record
and this file is its evidence.


Verbatim report of the source-reading agent that answered the repo owner's
three questions about mbirjax's multi-GPU orchestration and kernel
formulation.  Every claim carries file:line evidence from the mbirjax
checkout at "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax".
Items the reader derived rather than read are flagged as inference.

## A.1 Dispatch mechanism

A reused `ThreadPoolExecutor`, one worker thread per device.  Not pmap, not
`shard_map`, not SPMD.  The pool is reused across all bands within one
projection call, created and shut down per projection (not process-level).

`mbirjax/_sharding/thread_execution.py:19,43,80-88`:

```python
from concurrent.futures import ThreadPoolExecutor
...
def _run_on_device(i):
    with jax.default_device(devices[i]):
        return worker_fn(i, devices[i])

if executor is not None:
    return list(executor.map(_run_on_device, range(n)))
```

`device_pool` is the reuse mechanism (`thread_execution.py:24-47`): "Pass the
yielded pool as `run_per_device(..., executor=pool)` so a loop of many
per-device fan-outs ... reuses one pool instead of creating and tearing down
a fresh `ThreadPoolExecutor` per call."  Consumers wrap the whole band loop:
`tomography_model.py:1801` `with mjs.device_pool(len(devices)) as pool:`
encloses both the slice-owner and band loops, and `tomography_model.py:1711`
does the same for the cone path.

Results reassemble with no host round trip: `thread_execution.py:107-109`
`jax.make_array_from_single_device_arrays(global_shape, sharding,
list(per_device_arrays))`.

Dispatch is deliberately asynchronous — `thread_execution.py:58-62`: "this
function does NOT call block_until_ready on the results.  JAX dispatch is
asynchronous; leaving the results unblocked lets a caller overlap the next
batch's data transfer with the current batch's compute."

Alternatives were measured and rejected — `docs/source/dev_sharding_overview.rst:207-235`:
thread pool = "used (~2x on 2 GPUs)"; `shard_map` = "wrong on L40S (cause
unknown); correct on H100 -> not hardware-safe"; `pmap` = "deprecated";
threads-through-host = "dominated by transfer".  Also
`mbirjax/preprocess/segmentation.py:127` notes shard_map "invokes XLA's SPMD
partitioner, whose lowering has bitten" them.

Hardware workaround: `mbirjax/_sharding/transfer.py:36-58` empirically probes
whether direct device-to-device `device_put` is correct (on L40S it "silently
produces zeros on the destination — no error is raised"), and falls back to a
host bounce when not.

## A.2 Per-device residency during a sharded forward

It depends on geometry, and the two paths are structurally different.  Recon
is sharded by slice, sinogram by view (`dev_sharding_overview.rst:26-31`).
The sinogram never moves: `_sharding/placement.py` module docstring — "the
ONLY data that crosses the recon<->sino boundary is voxel-cylinder
slice-bands (the sinogram is written locally on its view-shard and never
moves)."

PARALLEL BEAM — bands streamed one at a time (the shape mbirtorch copied).
`tomography_model.py:1800-1813`:

```python
for slice_owner in slice_owners:
    cyl_shard, _ = recon_shard_info[slice_owner]   # (num_pixels, slices_per_dev)
    for (l0, l1) in band_bounds:
        band = cyl_shard[:, l0:l1]                 # (num_pixels, L) on slice_owner
        band_on_views = mjs.broadcast_band_to_views(band, devices, self.dev2dev_safe)
        row_bands = self._forward_project_band_to_local_views(...)
```

Each device holds: its own slice shard, one foreign band of shape
(num_pixels, L) at a time (`transfer.py:130-156`), and its accumulating list
of row-bands.  `band_on_views` rebinds each iteration, so the previous band
drops on refcount (inference; the analogous refcount reasoning is stated at
`tomography_model.py:1721-1724`).  Total received per device over one
forward = the whole recon once.

CONE — the opposite: no banding at all; full-height cylinders gathered per
pixel-batch.  `tomography_model.py:1695-1709`:

```python
for p0 in range(0, n_local, pixel_batch):
    p1 = min(p0 + pixel_batch, n_local)
    full_cyl = jnp.concatenate(
        [mjs.move_shard(recon_shard_info[so][0][p0:p1], view_owner, ...)
         for so in slice_owners], axis=1)
    full_cyl = full_cyl[:, :real_slices]
    part = self.projector_functions.sparse_forward_project(
        full_cyl, idx[p0:p1], owned_view_indices=view_ranges[view_owner])
    owned = part if owned is None else owned + part
```

A cone view-owner holds (fwd_pixel_batch, num_slices) — a full-height
cylinder for a narrow column of pixels — never a slice band.  The docstring
at `tomography_model.py:1654-1660` states the reason: "A slice can project to
a RANGE of detector rows (cone), so every view-owner needs ALL slices to
produce its own views' rows -- it cannot stream slice-bands."  The A/B result
is recorded in `dev_sharding_overview.rst:119-124`: the whole-cylinder form
"increased transient memory for the projector by about 10% but decreased time
by about 2x" over per-band.

## A.3 What bounds per-device memory at 2K^3 on 4 GPUs

Sharding plus in-kernel tiling; no separate large-problem code path.  Four
mechanisms, none size-switched:

1. Both arrays sharded, always.  Peak per GPU ~ 1/N (`usr_multi_gpu.rst:69-70`).
2. Bounded cross-device transient.  Parallel: fixed 256-slice band
   (`parallel_beam.py:116` `_FWD_SLICE_BAND_GPU = 256  # the measured knee;
   whole-shard bands are WORSE at scale`).  Cone: `fwd_pixel_batch` (4096 at
   >=768 slices, `cone_beam.py:415-416,436-437`), so the transient is
   ~4096 x num_slices x 4 B ~ 34 MB at 2048^3 regardless of device count.
3. Rolled tiling INSIDE each compiled program.  `_jit_sparse_forward_project`
   batches pixels with `jax.lax.scan` (`projectors.py:816`) and views with
   `jax.lax.map` (`projectors.py:710`); live vmap width capped by
   `fwd_view_batch` <= 128 (`tomography_model.py:572` `_FWD_VIEW_CAP = 128`).
4. Exits never re-gather onto one device.  `tomography_model.py:784-785`: "a
   large volume sharded across N devices is never re-gathered onto one device
   (which would OOM, e.g. a 32 GiB 2048^3 volume on one GPU)".

Size-driven thresholds (not alternate algorithms): `cone_beam.py:416`
`_FWD_PIXEL_BATCH_MIN_SLICES = 768`; `projectors.py:171`
`N_PC_SINGLE_CALL_MAX_BYTES = 256 * 2**20`; `projectors.py:66-67`
`SORTED_CHANNEL_REDUCE_MIN_COLS = 48` / `MAX_COLS = 1280`.

The one separate large-problem path is opt-in and user-invoked:
`ConeBeamModel.split_sino_recon` (`cone_beam.py:1339-1345`).

## A.4 Invocation count per device (the key structural difference)

mbirjax's per-device host dispatch count for the parallel-beam forward is
essentially independent of device count — the opposite of the torch port.

Arithmetic (derived by the reader from the code, flagged as inference):
`parallel_beam.py:186-192` sets `slices_per_dev = num_slices // n_dev`,
`band_len = min(256, slices_per_dev)`, `num_bands =
ceil(slices_per_dev/band_len)` (`_balanced_slice_bounds`,
`tomography_model.py:2373`).  `_forward_project_all_bands` loops n_dev
slice-owners x num_bands, calling `run_per_device` once per (owner, band).
Per-device calls = n_dev x ceil(slices_per_dev/256) ~ num_slices/256,
independent of n_dev.  At 1024 slices: n=1 -> 1x4 = 4; n=2 -> 2x2 = 4;
n=4 -> 4x1 = 4.  Adding devices shrinks each call in the VIEW dimension
(`view_ranges[view_owner]`); it does not multiply the calls.

Each call is ONE XLA program: `_jit_sparse_forward_project` is a single
`jax.jit` (`projectors.py:896-898`) whose pixel and view loops are
`lax.scan`/`lax.map` inside the compiled program.  One host dispatch; the
tiling happens on-device.

Cone: single-device is exactly one call; multi-device is
ceil(num_pixels/fwd_pixel_batch).  The n_dev==1 short-circuit comment,
`tomography_model.py:1686-1694`: "Single device: the whole cylinder is
already local on this device, so there is NO cross-device gather to bound --
the per-pixel-batch loop below is pure overhead and, by issuing many small
rigid dispatches, it defeats the XLA rematerialization that the monolithic
single-device forward relies on (measured: 1024^3 single-device peak ~16 GB
one-shot vs ~32 GB looped).  Project the full cylinder in ONE call."

Two measured statements bearing on the launch-cost finding:
- `tomography_model.py:2336-2338`: "The price -- ~n_dev bands per
  slice-owner, hence more dispatches -- is cheap on GPU (launch throughput
  hides it; time is flat across B) and carries no extra FLOPs."
- Dispatch floor, `tomography_model.py:2218-2225`: "~50M elements/band scaled
  fine and even sped up ... while ~0.8M/band added dispatch overhead with no
  memory benefit.  4M is a safe knee" -> `_BACK_PROJECT_MIN_BAND_WORK =
  4_000_000` (elements = num_pixels x band).

## B.1 Forward scatters, back gathers — both geometries

`docs/source/dev_projector_kernels.rst:38-45`: "Back projection reads each
detector value and adds it into the voxels along the ray -- a gather, with no
collisions.  Forward projection does the opposite: many voxels land on the
same detector channel, so the kernel's inner loop is a scatter-add into
shared locations, and on a GPU those colliding atomic adds were essentially
the entire forward-kernel cost."

Both geometries call the same shared helpers.  Forward scatter:
`projectors.py:221-267` `horizontal_fan_project`, fallback body
`sinogram_view_T.at[n, :].add(...)` (`:266`); called from
`parallel_beam.py:340` and `cone_beam.py:558`.  Back gather:
`projectors.py:270-320` `horizontal_fan_back` (`:319`), called from
`parallel_beam.py:388` and `cone_beam.py:763`.

Nuance: cone's forward is separable and mixed — a vertical fan first
(a GATHER driven by detector rows, `cone_beam.py:629-656`), then the
horizontal fan (the scatter).  Only the horizontal stage needs collision
handling.

## B.2 Sorted-channel structure: per-call in-kernel sort, not cached streams

The sort is `jax.lax.sort_key_val` executed inside the compiled kernel on
every call — `projectors.py:125-143`:

```python
sorted_n, order = jax.lax.sort_key_val(flat_n, jnp.arange(flat_n.shape[0]))
updates = A.reshape(-1)[order][:, None] * values[order % num_pixels]
return jax.ops.segment_sum(updates, sorted_n, num_segments=num_out, indices_are_sorted=True)
```

Selected by the `sort_by_channel` flag (`projectors.py:83-114`), used for
BOTH geometries: parallel at `parallel_beam.py:143`, cone at
`cone_beam.py:449` (`sort_by_channel=num_det_rows >=
SORTED_CHANNEL_REDUCE_MIN_COLS`).  Translation deliberately not —
`dev_projector_kernels.rst:46-48`: "at its real detector shapes the sorted
form measured 4.5--6.5x slower."

Why no caching: `parallel_beam.py:136-137`: "its per-call setup is ~free --
the centers exist for the XLA path anyway."

Precomputed outside the jit: integer channel centers,
`_jit_compute_scatter_centers` (`projectors.py:174-190`), per projector call,
not cached; motive is a known XLA miscompilation (`projectors.py:147-154`),
and forward/back consume one shared center value "so the pair stays adjoint
even at rounding ties" (`:156-157`).

Guard constants: `projectors.py:66-81` — MIN_COLS=48, MAX_COLS=1280 (above
which "the segment-sum lowering hits a register/vectorization ceiling and
collapses"), MAX_PSF_RADIUS=2, MIN_COLLISION_RATIO=4.

## B.3 Cone exploits the z-independence of the in-plane mapping

The channel mapping is computed once per (x,y) pixel per view at a single
dummy slice, then reused across every detector row.  `cone_beam.py:992-996`
(`slice_index = jnp.arange(1)` is the proof).  `geometry_xyz_to_uv_mag`
(`cone_beam.py:1067-1072`): `u = pixel_mag * x` — neither term touches z.
The per-pixel result feeds the SAME helper parallel uses
(`cone_beam.py:557-560` vs `parallel_beam.py:339-342`).
`dev_projector_kernels.rst:73-75`: the trapezoid-tap machinery lives once in
`projectors.py`; geometry files keep only coordinate stages and weight
scales.

## B.4 The vertical direction is factored: exact affine, two scalars per (view, pixel)

XLA forward kernel (`cone_beam.py:598-616`): per pixel, one y and pixel_mag;
`scaled_voxel_values = voxel_cylinder / cos_phi_p` in one shot; a single
scalar slope `W_p_r = (pixel_mag * delta_voxel_slice) / gp.delta_det_row`.
Row loop rolled via `jax.lax.map` over batches of
`CONE_FORWARD_DET_ROW_BATCH = 128` rows (`cone_beam.py:34,656`); the
voxel-to-row map is affine (`:640`).  No per-slice table.

The Pallas kernels name the factorization (`_pallas_kernels.py:558-563`):
"the projected detector-row center is EXACTLY affine in the slice index,
m(v,p,l) = m0(v,p) + W_p_r(v,p) * l ... so the vertical fan needs no
per-slice precompute: the kernel forms row centers, trapezoid row weights,
and the 1/cos(phi) divisor in-kernel from two scalars per (view, pixel)."
`_jit_compute_vfan_scalars` (`_pallas_kernels.py:579-594`) materializes
(m0, W_p_r); the forward kernel inverts the affine per output row (`:775`).

## Two things to carry back (the reader's synthesis)

1. The parallel-beam band structure mbirtorch copied is the right shape, but
   mbirjax pins the band to a fixed 256 slices rather than to the shard,
   which makes the per-device host dispatch count constant in n_dev instead
   of growing with it.  `parallel_beam.py:116` — "the measured knee;
   whole-shard bands are WORSE at scale."

2. Everything that does not require a cross-device copy stays inside one
   compiled program (`lax.scan` over pixel batches, `lax.map` over view
   batches).  Only the loop that actually moves data between devices is
   hoisted to Python.  Plus the explicit per-band work floor
   (`_BACK_PROJECT_MIN_BAND_WORK = 4_000_000` elements), which exists
   precisely because ~0.8M elements/band "added dispatch overhead with no
   memory benefit."
