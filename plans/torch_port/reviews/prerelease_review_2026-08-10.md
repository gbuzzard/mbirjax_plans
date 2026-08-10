# Review: five incoming commits on origin/prerelease

Reviewed 2026-08-10 against `origin/greg_dev` (tip `a880d9c`). All five fork from
`f7e08da`, the merge base. Read-only review: no checkout, no merge, no edits
outside the scratchpad. Tests were **not** executed on a merged tree (that would
require a merge); all runtime claims below are from code reading plus one
standalone numerical experiment noted in §1.

Commit order confirmed by parentage (oldest first):

| # | sha | subject |
|---|-----|---------|
| 1 | `a5b04ce` | Port the sharded case of segment_plastic_metal |
| 2 | `c5b5438` | Accept sharded volumes in the HDF5 export |
| 3 | `712c523` | Spread scan preprocessing across the visible devices |
| 4 | `2b1d02c` | Port the sharded case of QGGMRFDenoiser |
| 5 | `944aec2` | Port the sharded case of the beam-hardening correction |

They are a dependency chain, not five independent changes — see §7.

The mbirjax parity anchors quoted throughout come from the reference checkout at
`/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax` (HEAD `b74ffc8`).

---

## 1. `a5b04ce` — sharded `segment_plastic_metal`

### What it does and why

A full-resolution MAR run OOM'd on GPU 0 (job 15001292) because the segmentation
took the volume only as one tensor on one device. The commit makes the
segmentation chain form-polymorphic: numpy, tensor, one-shard `Shards`, and
several-shard `Shards` all accepted, results returned in the form given.

Four pieces:

- `mbirtorch/preprocess/segmentation.py:5-95` — new
  `_shard_valid_masks` / `_shard_chunks` / `_sharded_masked_histogram`. Each
  shard is histogrammed on its own device in chunks bounded to
  `_HISTOGRAM_CHUNK_ELEMENTS = 1 << 24`; only the `num_bins`-long int64 count
  tables leave a device, and they sum on the host.
- `mbirtorch/preprocess/segmentation.py:302-372` — `segment_plastic_metal` grows
  a `Shards` branch; a *trivial* (one-shard) container recurses through the
  tensor path and is rewrapped, which is the cheapest correct way to honor the
  array-forms rule.
- `mbirtorch/preprocess/utilities.py:642-682` — `compute_scaling_factor` gains a
  sharded branch; the chunked inner-product loop is extracted to `_dot_sums` and
  shared by both branches.
- `mbirtorch/preprocess/utilities.py:754-774` — `apply_cylindrical_mask` gains a
  sharded branch that masks each shard on its own device and zeroes each shard's
  overlap with the global top/bottom margin ranges.
- `mbirtorch/preprocess/mar.py:184-210` — three seams in
  `_est_plastic_metal_sinos_from_recon` that assumed a tensor (`.ndim`,
  `mask * recon`, `scalar * sinogram`) now handle both forms.

### Do the claims hold?

Mostly yes, with one over-claim.

**Over-claim (the headline finding for this commit).** The message says "the
small count tables are summed on the host as exact integers, so the thresholds
equal the unsharded ones exactly." The summing is exact, but the *binning* is
not the same rule as the reference path's. The single-device path goes through
`np.histogram` (`segmentation.py:96-114`, `_masked_histogram`), which computes
the bin index in float64 and then runs numpy's ULP edge-correction pass
(`decrement`/`increment` against the actual `bin_edges`). The sharded path
(`segmentation.py:79-84`) hand-rolls the binning:

```python
idx = torch.clamp(((vals - lo) * scale).to(torch.int64), max=num_bins - 1)
```

— float32 arithmetic, truncation toward zero, no correction pass.

I tested this directly (standalone, no repo imports; script at
`scratchpad/hist_check.py`, run under the miniforge env). Over 200 trials of
20 000 float32 uniform samples into 1024 bins, **up to 8 counts per trial land
in a different bin** than `np.histogram` gives. Values sitting exactly on bin
edges happened to agree (0 mismatches), so the divergence is ordinary float
rounding in the interior, not an edge-rule bug.

Practical impact is low: the Otsu DP maximizes between-class variance over 1024
bins whose counts are in the millions at production scale, so a handful of
misplaced counts will essentially never move a boundary — and the test in this
commit does pass. But "exactly" is not true by construction, and mbirjax is
explicitly honest about the same divergence in its own port
(`mbirjax/preprocess/segmentation.py:131-134` records that a value exactly on an
interior bin edge can differ from numpy by one bin, "deemed irrelevant at Otsu's
granularity"). **The code is at mbirjax parity; the commit message is stricter
than the behavior.** Worth softening the message/docstring so a future reader
does not build on a guarantee that isn't there.

**Parity is otherwise good.** mbirjax's `_sharded_histogram`
(`segmentation.py:113-161`) uses the same two-pass structure: masked min/max
combined on the host, then its own bucketize-and-scatter per slab, int64 host
accumulation, no cross-device collectives. mbirtorch's chunk bound (2^24) is
more conservative than mbirjax's (2^28). The edge derivation differs in
mechanism but not in result: mbirtorch calls `np.histogram` on an *empty* array
of the right dtype to borrow numpy's exact edge arithmetic
(`segmentation.py:88-92`), mbirjax recomputes `np.linspace(lo, hi, num_bins+1,
dtype=image.dtype)`. Both land on the same float32 edges.

### Defects and risks

- **`segmentation.py:75` — degenerate constant volume gives a silently wrong
  answer (LOW severity, but silent).** When `hi == lo` the code sets
  `scale = 0.0`, so every value lands in bin 0. But the edges on the next lines
  come from `np.histogram(empty, bins=num_bins, range=(lo, hi))`, and numpy
  *expands* a zero-width range to `(lo-0.5, lo+0.5)`. Measured: for a constant
  0.3 volume with 8 bins, `np.histogram` puts all 1000 counts in bin 4 with
  edges `[-0.2, 0.8]`; the sharded rule puts them all in bin 0 against those
  same expanded edges. Counts and edges therefore disagree, and the derived
  thresholds are wrong rather than raising. The single-device path is coherent
  in this case because numpy does both halves. A constant volume is degenerate
  input for segmentation anyway, so this is a robustness nit, not a live bug —
  but it fails silently, which is the wrong failure mode.
- **`utilities.py:757-772` — `apply_cylindrical_mask`'s sharded branch hardcodes
  the slice axis (LOW).** It writes `masked[:, :, lo-s0:hi-s0] = 0`, i.e. axis 2,
  while taking the shard bounds from `pl.shard_ranges(...)` for whatever
  `pl.axis` happens to be. Correct today because `recon_placement.axis` is always
  `-1`, but the two facts are not tied together and nothing asserts it. The
  unsharded path below has the same axis-2 assumption, so this is consistent
  rather than new.
- **No aliasing bug here — checked.** `apply_cylindrical_mask` allocates a fresh
  array at `utilities.py:801` (`recon = recon * circular_mask[:, :, None]`), so
  the sharded branch's in-place `= 0` writes hit that new array, not the
  caller's shard. Confirmed by reading, since an in-place write into a caller's
  device tensor would have been the serious defect to find in this diff.
- **`mar.py:184-185` — pre-existing shape hazard now reachable (LOW).**
  `valid_mask = pl.real_mask(ndim)` takes its length from
  `ct_model.recon_placement.padded_size`. If a caller hands
  `correct_sino_plastic_metal` a *host* recon while the model is multi-device,
  the mask is `padded_size` long and the volume is `real_size` deep → broadcast
  error. This predates the commit (the old line was the same modulo `.ndim`),
  but multi-device MAR only became runnable now, so the path is newly reachable.
  Inside `recon_plastic_metal` it cannot happen (`direct_recon(output_sharded=
  True)` and `recon(output_sharded=True)` both return device form).

### Ledger

The sharded `class_mask` (`segmentation.py:352-360`) allocates one **full-volume
float array per class** — `plastic_mask` plus `num_metal` metal masks — spread
across the devices and live simultaneously with the recon itself. The histogram
chunking additionally allocates up to two chunk-sized temporaries per chunk
(`chunk[mc]`, then the `(vals >= lo) & (vals <= hi)` filter), bounded at 2^24
elements ≈ 64 MB float32. None of this is in `_memory_ledger`. See §6.

### Test coverage delta

New file `tests/test_sharded_segmentation.py` (+68): whole-vs-2-CPU-shard
identity on masks (exact `array_equal`), zero padding on the masks' padded
slices, scale factors to `rel=1e-5`, and a one-shard round-trip for the
array-forms rule. Good gates, well-chosen bar. Gaps: CPU-only and 2 shards only
(no ≥3, no unpadded sharding), no regression test that the refactored *numpy*
branch of `class_mask` still behaves (that branch was rewritten at
`segmentation.py:365-369` and is only covered incidentally), and nothing
exercises the degenerate `hi == lo` case above.

**Verdict: ISSUES (minor).** Sound port at mbirjax parity; the "exactly" claim
should be softened and the `hi == lo` case should raise instead of silently
mis-binning.

---

## 2. `c5b5438` — sharded volumes in the HDF5 export

### What it does and why

`export_recon_hdf5` / `save_data_hdf5` accepted only numpy and plain tensors, so
a volume from `recon(output_sharded=True)` across several GPUs could not be
written. `_to_host` (`mbirtorch/utilities.py:159-172`), the helper both use, now
gathers a `Shards` and crops the zero-padding of the sharded axis.

### Do the claims hold?

Yes. The implementation is small and correct: `gather()` already returns numpy
so it is not re-converted, the crop is guarded on
`pl.real_size is not None and pl.padded_size > pl.real_size`, and the axis is
normalized with `pl.axis % out.ndim`. The `save_data_hdf5` dispatch guard
(`utilities.py:269-271`) correctly widens from `hasattr(array, 'detach')` to
also admit `Shards`.

No circular-import risk from the new module-level `from . import _sharding` in
`utilities.py`: `_sharding.py` imports only stdlib, numpy and torch
(`_sharding.py:27-31`). Verified because `utilities.py` is imported early.

### Defects and risks

- **`utilities.py:159-172` + `:269-278` — the gather defeats the slab streaming
  this module was built around (MEDIUM at production scale).** `save_data_hdf5`
  deliberately streams: its comment at `utilities.py:273` says "Stream the array
  to disk slab-by-slab (no full contiguous copy, even for a strided view)", and
  `produce_slab(i0, i1)` exists for exactly that. But `_to_host(Shards)` builds
  the **entire** host array up front before the streaming loop starts, so for a
  sharded input the peak host footprint is the full volume plus the slab — the
  thing the streaming design exists to avoid. At 2K³ float32 that is ~32 GB
  materialized on the host in one allocation. Given "production scale is 2K³ and
  larger", this is worth fixing rather than deferring: the natural fix is to make
  `produce_slab` gather per slab from the shards instead of gathering once.
  Not a correctness bug, and it is strictly better than the previous behavior
  (which was a crash), so it need not block.

### Test coverage delta

`tests/test_sharded_segmentation.py` +18: exports the same volume whole and as
two padded CPU shards, loads both back, asserts shape (padding cropped 12→11)
and `array_equal`. Right test for the claim. No test of `save_data_hdf5`
directly (only `export_recon_hdf5`), and none of a non-`-1` sharded axis.

**Verdict: ISSUES (minor).** Correct and well-tested; the full-volume host
gather quietly undoes the module's streaming guarantee at production scale.

---

## 3. `712c523` — "Spread scan preprocessing across the visible devices"

This is the commit flagged for hardest scrutiny, and it is indeed the one
device-count decision made outside `_apply_device_policy`. Full treatment of the
policy question is in §5; this section covers the code.

### What it does and why

`map_view_batches` accepted a `devices` list but ran everything on the first
entry. It now has two modes (`mbirtorch/preprocess/pipeline.py:52-121`): the
existing single-device sequential view-batch loop, and a multi-device mode that
splits the views into contiguous in-order shards (`np.array_split`), one per
device, each filled by its own thread via `_sharding.run_per_device` into its
disjoint slice of the one shared host output. Separately,
`scan_to_sino` (`mbirtorch/preprocess/utilities.py:433-435`) changes its
`devices=None` default from "one device" to "all visible CUDA devices".

### Do the claims hold?

**Yes, and it is faithful mbirjax parity — including the split of defaults.**
mbirjax's `scan_to_sino` does `devices = jax.devices()` when `devices is None`
(`mbirjax/preprocess/utilities.py:449-450`), and its shared driver
`map_view_batches` keeps the opposite default, `[jax.devices()[0]]`
(`mbirjax/preprocess/pipeline.py:71`). mbirtorch reproduces both defaults
exactly, so the legacy per-stage public functions (`compute_sino_transmission`,
`downsample_view_data`, `correct_det_rotation`) stay single-device. The threading
mechanism, the probe-then-preallocate structure, and the `np.array_split`
contiguous in-order shards all match.

The probe/worker boundary is handled correctly: the probe fills `output[0:probe_hi]`
on `devices[0]`, and every worker starts at `max(rng[0], probe_hi)`
(`pipeline.py:113`), so views already filled by the probe are not recomputed and
no view is skipped even when `batch_size` exceeds one shard's width.

### Defects and risks

- **`pipeline.py:95` + `:110-118` — "byte identical" holds only for homogeneous
  devices (LOW).** The probe batch runs on `devices[0]` regardless of which shard
  owns those views, so views `[0:batch_size]` are computed on device 0 while the
  rest of shard 0's range may run elsewhere. On identical GPUs this is exact; on
  a heterogeneous box (different arch, different fast-math paths) the probe
  boundary is a place where results could differ by an ULP. The commit's flat
  "byte identical to the single-device one" should be qualified.
- **No memory bound and no speed floor (MEDIUM — see §5).** Device residency is
  `batch_size` views per device, so it is bounded and small, but nothing checks
  it and nothing asks whether spreading is *worth it* at this problem size. A
  short scan fanned across 8 GPUs pays 8 CUDA context creations plus per-device
  H2D setup for possibly negative net gain — which is precisely the pathology
  `_widening_floors` was built to prevent, reintroduced in a module the floors
  do not cover.
- **Side effect on the recon preflight (LOW, and self-correcting).** Touching
  every visible device creates a CUDA context on each (~300–600 MB). A later
  recon's `device_budget_bytes` (`mbirtorch/_memory_ledger.py`) reads live
  `torch.cuda.mem_get_info` free bytes, so it *observes* this rather than being
  fooled by it — the preflight becomes slightly more conservative, not wrong.
  The allocator's cached-but-free blocks are also credited back
  (`reserved - allocated`), so preprocessing's freed batch buffers do not
  penalize the recon. Worth knowing, not worth blocking.

### Test coverage delta

New `tests/test_sharded_pipeline.py` (+26): the same per-view kernel through
`devices=['cpu']` and `devices=['cpu','cpu','cpu']`, asserting `array_equal`.
This is an honest and useful gate on the *split arithmetic* — view ranges,
probe boundary, no gaps or overlaps — but three CPU "devices" are one physical
device, so it cannot exercise real concurrency, real cross-device determinism, or
the thread-safety of the shared-output writes under actual parallelism. There is
also **no test that `scan_to_sino`'s changed default actually fans out** — the
one behavioral change users will see is untested (understandably: it needs
multi-GPU hardware).

**Verdict: ISSUES.** Code is correct and mirrors mbirjax exactly; the concern is
governance (a second device-count rule, unguarded by ledger or floors), not
correctness. See §5.

---

## 4. `2b1d02c` — sharded `QGGMRFDenoiser`

### What it does and why

The denoiser was single-device: a multi-device model crashed at `.clone()` on the
sharded container before any denoising ran. `denoise` now branches
(`mbirtorch/denoising.py:270-296` single device, `:311-411` sharded). The sharded
path slice-shards the image, stages the qGGMRF halos once per pass, has each
device update its own shard, and combines four line-search sums on the host into
one step size. `denoise` also gains `logfile_path` / `print_logs` and returns its
dict through `get_recon_dict`.

### Do the claims hold?

**Yes on the algorithm, with strong parity.**

- **Halo cadence.** Halos are staged once per outer pass
  (`denoising.py:373-374`), *not* per subset, and are not recomputed after each
  subset's update. This matches mbirjax's denoiser exactly
  (`mbirjax/denoising.py:488-495`) and matches mbirtorch's own `vcd_recon`, whose
  `stage_halos` is called by the partition iterator before the subset loop
  (`tomography_model.py:2250-2252`). The staleness this admits within a pass is
  the documented, accepted design, bounded by the seeded n=2-vs-n=1 parity test.
  Note mbirjax's `vcd_recon` keeps an A/B escape hatch
  (`_vcd_halo_per_subset`) that neither denoiser has — same gap in both, so not a
  port defect.
- **Alpha.** `alpha = min(max(alpha, _F32_EPS), 1.5)` (`denoising.py:388-389`)
  uses the same hardcoded 1.5 as the single-device kernel
  (`denoising.py:69-70`) and as mbirjax (`mbirjax/denoising.py:438-439`, which
  likewise does not read the `max_alpha` model param). Consistent.
- **Interface masks and the padded slice axis** are taken from
  `self._qggmrf_interface_masks()` (`denoising.py:365`), the same cached
  per-device masks `vcd_recon` uses (`tomography_model.py:755-787`). Correct.
- **No input mutation.** `flat_image` is a `.clone()` of the init shards
  (`denoising.py:345-347`) and `flat_error` is a fresh subtraction, so the
  in-place `index_add_` updates never touch the caller's tensors. When
  `init_image is None`, `init_sh is image_sh` and the residual is exactly zero —
  matching the single-device path. Checked carefully; clean.
- **Return-shape change is backward compatible.** `denoiser_dict` moves from a
  hand-built `{'recon_params', 'model_params'}` to `get_recon_dict(...)`, which
  returns `{'recon_params', 'recon_log', 'notes', 'model_params'}`
  (`tomography_model.py:3244-3300`) — a superset. Existing readers of either old
  key keep working.

### Defects and risks

- **`denoising.py:376-395` — the line-search sums sync to host per subset per
  device (MEDIUM; the one place this port drops below the code it mirrors).**
  `terms_worker` returns four `float(torch.sum(...))` values and `apply_worker`
  returns a fifth, so every subset of every pass forces 5 × `n_devices`
  device→host synchronizations from inside worker threads. mbirtorch's own
  `vcd_recon` deliberately does the opposite — it keeps the line search on device
  via `combine_on_lead`, with the explicit comment "The line search stays ON
  DEVICE (alpha is a scalar tensor; no host synchronization per subset)"
  (`tomography_model.py:2382-2387`). mbirjax's denoiser needs no reconciliation
  at all, because the identity forward model means all four scalars live on the
  single recon mesh (`mbirjax/denoising.py:434-436, 461-470`) — so mbirjax pays
  *nothing* here and mbirtorch pays the most expensive available option. This is
  also the reason bitwise parity is lost (the commit acknowledges the host sum
  order). Recommend routing these through the same `combine_on_lead` path
  `vcd_recon` uses.
- **`denoising.py:381, 397` — `run_per_device` called without `executor=`
  (LOW-MED).** Each of the two fan-outs per subset builds and tears down a
  private `ThreadPoolExecutor` (`_sharding.py:311-315`). `vcd_recon` reuses
  `self._per_device_pool` for exactly this reason
  (`tomography_model.py:2300, 2344, 2375, 2419`). At 15 iterations × N subsets ×
  2 fan-outs this is thousands of pool create/destroy cycles.
- **`denoising.py:367-368` — the whole partition is materialized on every device
  up front (LOW, and a ledger item).**
  `idx_per_dev = [[torch.as_tensor(partition[k]).to(d) for d in devices] for k in
  range(partition.shape[0])]` puts `num_pixels × 8 bytes` of int64 indices on
  *each* device, resident for the entire denoise (~34 MB/device at 2048²
  pixels). `vcd_recon` builds its per-subset indices transiently inside the
  updater instead (`tomography_model.py:2277-2278`). Hoisting is a defensible
  speed choice — it avoids repeated H2D — but it should be a deliberate,
  documented one, and it is uncharged.
- **`denoising.py:233-237` — new full-volume host gather (LOW).** A sharded input
  is passed through `_to_host` purely so `estimate_image_noise_std` and
  `auto_set_regularization_params` can index it on the host. Correct (and the
  crop means padding is properly excluded from the statistics), but it is a new
  full-volume host allocation on the sharded path.
- **`denoising.py:227` — new default side effect on the *single-device* path
  (LOW).** `denoise` now calls `_log_run_header(first_iteration, logfile_path,
  print_logs)` with `logfile_path='~/.mbirtorch/logs/recon.log'` by default, so
  every existing `denoise()` call now creates/appends a log file where it
  previously wrote none. This is deliberate consistency with `recon`, and the
  tests pass `logfile_path=None`, but it is a user-visible default change that
  the commit message frames only as "gains the arguments the other entry points
  have."

### Test coverage delta

`tests/test_denoiser.py` +33: same seeded problem on 1 device vs 2 padded CPU
shards, gated at `rel < 1e-4` (the measured iterated-comparison floor), plus a
check that the dict now carries `recon_log` and `notes`. The right bar and the
right gate. Gaps: 2 shards only, 5 iterations, `verbose=0` so the logging path is
only key-checked, and no test that a denoiser with an *unpadded* multi-shard
layout works (the interface-mask `None` branch).

**Verdict: ISSUES (minor).** Algorithmically sound and at mbirjax parity; the
per-subset host syncs and the throwaway thread pools are avoidable and diverge
from the standard `vcd_recon` sets in the same repo.

---

## 5. `944aec2` — sharded beam-hardening correction

### What it does and why

`correct_sino_plastic_metal` and everything under it operated on plain tensors,
so multi-device MAR failed downstream of the now-sharded segmentation. The chain
becomes form-polymorphic through a helper set at the top of
`mbirtorch/preprocess/mar.py:13-90`: `_ps_map` (elementwise, same form out),
`_ps_sum` / `_ps_max` / `_ps_numel` (reductions combining per-piece partials on
the host), `_ps_item` (global 3-D index → float), `_ps_argmin3d` (global argmin
preserving tie-breaking), `_ps_view_mask` (host view mask sliced per piece).

### Do the claims hold?

Mostly, with one exception on "single-device behavior unchanged."

- **Argmin tie-breaking — holds.** `_ps_argmin3d` (`mar.py:60-72`) visits pieces
  in view order and compares strictly (`float(val) < best_val`), so a global tie
  resolves to the first view, and `_argmin_3d` within a piece resolves to the
  first index. Equivalent to the single-tensor row-major argmin. mbirjax reaches
  the same guarantee by a different route — a per-view reduce then a global
  argmin over the small `(V,)` vector (`mbirjax/preprocess/mar.py:283-302`) —
  which is shard-order-independent by construction. Both are correct; mbirjax's
  is structurally more robust.
- **"Reductions combine per-piece partials on the host in double precision" —
  holds, and is *better* than mbirjax.** `_ps_sum` sums `float(...)` per-piece
  results as Python floats (float64), and `HtH`/`Hty` are already
  `dtype=np.float64` (`mar.py:508-509`). mbirjax accumulates `HtH`/`Hty` in
  float32 (`mbirjax/preprocess/mar.py:431-450`); only its OSQP `theta` touches
  float64.
- **"The scale maxima are exact by construction" — holds.** Max is
  order-invariant, and `_ps_max` returning a Python float instead of a 0-d tensor
  is numerically identical: `float()` of a float32 is exact, and dividing a
  float32 tensor by a Python float keeps float32 weak-scalar semantics.
- **Form alignment is safe.** `_ps_map` keys on `xs[0]`'s form and assumes all
  arguments share it. This holds because `measured_sino` is normalized by
  `ct_model.prepare_sino_for_devices(...)` at `mar.py:783` and every estimate
  comes from `forward_project(..., output_sharded=True)` on the same
  `sino_placement`. Verified, but unasserted — see below.

**The exception.** `mar.py:713-717` changes the unmasked Sp mean from
`torch.mean(Sp)` to `_ps_sum(torch.sum, Sp) / _ps_numel(Sp)`, and `_ps_sum`
returns a Python float on *both* branches — so this is a **single-device change**,
not only a sharded addition. Two differences follow: the division moves from
float32 to float64, and the clamp below moves from
`torch.maximum(Sp, Sp_floor_tensor)` to `torch.clamp(Sp, min=Sp_floor_float)`.
mbirjax keeps `jnp.mean` here (`mbirjax/preprocess/mar.py:621-624`). The value
feeds `Sp_floor = gamma * mean_plastic_coef`, a stabilization floor that then
divides the corrected sinogram, so it is not a cosmetic path. The 62 goldens
passing bounds the effect below golden tolerance, but the commit's flat
"single-device behavior unchanged" is not exact and this is the line that
falsifies it. Low risk, but it should be stated rather than claimed away — or
the unsharded branch should simply keep `torch.mean`.

### Defects and risks

- **`mar.py:19-25` — `_ps_map` has no form/placement validation (LOW).** If a
  caller ever passes a mixed pair (tensor first, `Shards` later), the non-sharded
  branch calls `fn` with `Shards` objects and produces garbage or an obscure
  error rather than failing at the seam. One `assert` on matching forms/placements
  would make the invariant explicit; every current call site satisfies it.
- **`mar.py:938-943` — full host gather per BH pass (MEDIUM at production
  scale).** `init = ct_model._gather_recon(recon)` runs on every pass because the
  recon entry points validate `init_recon` as a plain array. At 2K³ × 3 passes
  that is ~32 GB gathered and re-scattered three times. The commit records this
  as a known gap with the engine-side one-line fix noted in the plan, which is the
  right disposition — flagging it here only so it stays visible.
- **Full-sinogram temporaries per piece.** `Sp`, `y_minus_Sm`, `Sp_masked`,
  `ymSm_masked`, and the `h_i`/`h_j` column pair in `_compute_entry_for_OSQP` are
  each a full sinogram per piece, and the OSQP entry loop is O(num_cols²) column
  builds inside a `num_constraint_update_iter`-deep loop. This is *not* a
  regression — the single-device code had the same structure — but it is a large
  uncharged residency that sharding now spreads rather than reduces. See §6.

### A hazard I checked and cleared

`recon_plastic_metal` calls `recon_function` once per BH pass, and on greg_dev
`ct_model.recon` → `vcd_recon` → `_apply_device_policy`, which **re-evaluates the
device count on every entry** while the layout is automatic. I traced whether a
mid-loop placement change could leave the pass's `recon` shards disagreeing with
`ct_model.recon_placement` (which would corrupt `valid_mask`, the interface
masks, and `forward_project`). It cannot: `recon` is always the return value of
the most recent `recon_function` call, the policy runs at the *start* of that
call, and nothing between the policy and the return changes the placement — so
the shards and the placement are always in sync when
`correct_sino_plastic_metal` reads them. `direct_recon` and `forward_project` do
**not** run the policy (`_apply_device_policy` has exactly one call site,
`tomography_model.py:2574`, inside `vcd_recon`), so they inherit. **MAR is
policy-neutral and coherent under the new per-entry re-evaluation.**

### Test coverage delta

`tests/test_sharded_segmentation.py` +40: the full MAR pipeline (`recon_plastic_metal`,
2 BH passes, 2 iterations) on 1 CPU device vs 2 CPU shards with seeded
partitions, gated at `rel < 1e-3`. Appropriate given that discrete constraint
selection can amplify float differences. Gaps: a 16³ cell is very small (the
constraint search may never exercise interesting branches), the corrected
sinogram itself is not gated separately despite the message quoting ~1e-6 for it,
and there is no direct test of `_ps_argmin3d`'s cross-shard tie-breaking — the
subtlest thing in the diff and the one most worth a unit test.

**Verdict: ISSUES (minor).** Careful, well-reasoned port; the Sp-mean change
quietly touches the single-device path against the commit's own claim.

---

## 6. Cross-cutting: is any of this a second device-policy site?

**Answer, per commit:**

| commit | verdict |
|---|---|
| `a5b04ce` segmentation | **Inherits.** Reads `ct_model.recon_placement` / operates on whatever shards it is handed. No count chosen. |
| `c5b5438` HDF5 export | **Device-neutral.** Reads a container's own placement at a file boundary. |
| `712c523` scan preprocessing | **CHOOSES.** `scan_to_sino` defaults to all visible CUDA devices by its own rule. |
| `2b1d02c` denoiser | **Inherits.** Uses `image_sh.placement.devices`; explicitly disclaims the automatic count. |
| `944aec2` beam hardening | **Inherits.** Uses `ct_model.sino_placement`; `forward_project` and `direct_recon` do not run the policy. |

### The one fork, judged

`712c523` is a genuine second device-count decision site: `scan_to_sino` with
`devices=None` fans out over every visible CUDA device
(`mbirtorch/preprocess/utilities.py:433-435`), and no ledger, preflight, or speed
floor sees it.

Three things argue it is nonetheless acceptable, and one argues it needs
follow-up.

1. **It structurally cannot route through the policy.** `_apply_device_policy` is
   a method on `TomographyModel`. `scan_to_sino` takes no model — it is a pure
   function over raw scans that runs *before* any model exists. There is no
   object to ask. The uniformity ruling ("public entries should route through it")
   is about reconstruction entries on a model; this entry has no model to route
   through.
2. **It is exact mbirjax parity, including the asymmetry.** mbirjax's
   `scan_to_sino` is `devices = jax.devices()`
   (`mbirjax/preprocess/utilities.py:449-450`) while its shared driver
   `map_view_batches` defaults to one device
   (`mbirjax/preprocess/pipeline.py:71`). mbirtorch reproduces both. The commit
   message states the choice openly ("Preprocessing does not consult the
   reconstruction device policy... per the plan's design note"), so it is a
   declared design decision, not an accident.
3. **The resource profile is different in kind.** Recon widening is about whether
   a volume-sized working set *fits*; preprocessing residency is
   `batch_size` views per device — bounded, small, and independent of the recon
   geometry. The ledger's phase model (`_memory_ledger.plan_from_model`) has
   nothing to say about it.

**The follow-up it needs:** the *speed* half of the guard does apply in spirit.
The widening floors exist because widening a small problem makes it slower, and
"all visible devices, unconditionally" is exactly the rule the floors replaced on
the recon side. A short scan across 8 GPUs pays 8 context creations and 8
H2D setups for possibly negative gain, and nothing measures or holds that back.
This does not make the commit wrong — it makes it the one place in the library
where "how many devices" is answered without a floor. Recommend a follow-up that
either gives preprocessing a cheap size threshold or documents explicitly why
preprocessing is exempt, so the exemption is a recorded decision rather than an
inherited default.

### The QGGMRFDenoiser caveat (dimension 2)

**Does the denoiser's loop now genuinely run multi-device?** Yes. `_denoise_sharded`
(`mbirtorch/denoising.py:311-411`) is a real sharded loop: per-device prior
gradients/Hessians, staged halos across shard boundaries, per-device in-place
`index_add_` updates, and a combined step size. It is structurally the same loop
as `vcd_recon`'s sharded path and as mbirjax's `_denoise_sharded`.

**Does the policy docstring become stale?** Yes — its *first and heaviest* stated
reason does. `tomography_model.py:894-897` reads:

> "Three reasons, in order of weight. A construction-time choice would give
> ``QGGMRFDenoiser`` a multi-device layout its own loop cannot run."

After `2b1d02c` that sentence is false: the loop *can* run it. The other two
reasons are untouched and still carry the ruling — the ledger needs a free-memory
reading that is only meaningful when a reconstruction is about to start, and a
developer calling a projector directly has not asked for a layout change. **The
ruling survives; the docstring must be rewritten**, and it should be rewritten in
the same change that lands `2b1d02c`, because a stale "cannot run" is exactly the
kind of comment a future reader will build on.

**Would the denoiser now want the policy call?** No — and the current arrangement
is already consistent with the policy's own rules. After this commit,
multi-device denoising is reachable *only* through an explicit
`configure_devices` call (the denoiser never enters `vcd_recon`, the sole
`_apply_device_policy` call site). `_apply_device_policy`'s own first branch says
an explicit layout is the caller's and is neither searched nor reduced. So an
explicitly-configured multi-device denoiser is being treated exactly as the policy
would treat it anyway.

**Does anything break if a denoiser inherits a wide layout today?** No, and it
cannot silently acquire one. `copy_ct_model` (`mbirtorch/utilities.py:687`)
propagates devices only when the user set them explicitly ("If the user
explicitly set the devices on ct_model with configure_devices, the copy gets the
same devices. Otherwise the copy chooses its own devices when it is used"), and it
copies geometry models, not denoisers. I also checked the logging seam: the
denoiser now calls `_log_device_report()` (`denoising.py:228`), and greg_dev
changed `_device_report`'s formatting in `mbirtorch/parameter_handler.py:154-166`
to label the in-use count. For a denoiser, `device_choice_rejections` is `[]` and
(after `configure_devices`) `device_layout_is_automatic` is `False`, so the new
clause is skipped entirely. No crash, no misleading line. The guard's
`_speed_floor_held` / `_speed_floor_fallback` are initialized in `__init__`
(`tomography_model.py:148, 151`) and read via `getattr(..., None)` in `_settle`,
so a denoiser that never runs the policy leaves no landmine.

**The real gap** is not the caveat but the consequence: explicit multi-device
denoising now runs with **no memory preflight and no ledger at all**. That is
consistent with how explicit layouts are treated everywhere, so it is defensible
— but it is new surface area for the uncharged-residency class.

---

## 7. Ledger coverage (dimension 4)

None of the five commits adds a `_memory_ledger` term, and four of them add
per-device allocations the ledger does not know about.

| path | new per-device allocations | ledger knows? |
|---|---|---|
| sharded segmentation (`a5b04ce`) | `plastic_mask` + `num_metal` metal masks — **one full-volume float array per class**, live simultaneously with the recon; plus ≤2 chunk temporaries ≤64 MB | No |
| HDF5 export (`c5b5438`) | host only (full-volume gather); no device allocation | N/A |
| scan preprocessing (`712c523`) | `batch_size` views per device × every visible device, plus a CUDA context per device | No |
| sharded denoiser (`2b1d02c`) | `flat_image` + `flat_error` (2 volume-equivalents across devices), per-subset `deltas`, and `idx_per_dev` = `num_pixels × 8 B` **on every device, persistent** (~34 MB/device at 2048²) | No — the denoiser never calls `_build_memory_ledger` |
| sharded BH (`944aec2`) | `Sp`, `y_minus_Sm`, `Sp_masked`, `ymSm_masked`, `h_i`/`h_j` — each a full sinogram per piece, inside an O(num_cols²) × `num_constraint_update_iter` loop | No |

**How bad is this, honestly?** Less bad than "uncharged residency" usually
implies, for one specific reason: `device_budget_bytes`
(`mbirtorch/_memory_ledger.py`) is a **live reading** — `torch.cuda.mem_get_info`
free bytes plus the allocator's reclaimable `reserved - allocated`. Anything
allocated *before* a recon entry is therefore observed by the preflight rather
than modeled by it, and cached-but-freed blocks are correctly credited back. So
segmentation masks and BH temporaries that are live when the next pass's
`ct_model.recon(...)` runs its preflight make that preflight *conservative*, not
wrong.

The exposure that remains is real but different from the campaign's core case:
**these paths can OOM with no preflight at all**, because none of them runs one.
A production MAR run that dies inside `_compute_entry_for_OSQP` gets torch's raw
allocator error, not the ledger's readable shortfall report with a dominant
phase and a remedy. The denoiser is the cleanest instance — an explicitly
4-device denoise of a large volume has no capacity check anywhere in its path.

I would not treat this as a merge blocker. It is a new charter's worth of work
(preprocessing/denoiser ledger phases), and it is *additive* to the campaign's
existing scope rather than a regression of it: before these commits these paths
did not run multi-device at all.

---

## 8. Merge collision with greg_dev (dimension 3)

### Textual: clean, and not marginally so

The two branches touch **completely disjoint file sets**. Verified two ways:

```
comm -12 <(git diff --name-only f7e08da origin/prerelease | sort) \
         <(git diff --name-only f7e08da origin/greg_dev    | sort)
→ empty
```

and `git merge-tree f7e08da origin/greg_dev origin/prerelease` produces a merged
tree with **zero conflict markers** and zero "changed in both" entries.

- prerelease touches: `denoising.py`, `preprocess/{mar,pipeline,segmentation,utilities}.py`,
  `utilities.py`, `tests/{test_denoiser,test_sharded_pipeline,test_sharded_segmentation}.py`
- greg_dev touches: `_memory_ledger.py`, `_widening_floors.py`, `projectors.py`,
  `tomography_model.py`, `parameter_handler.py`, `cone_beam.py`, `parallel_beam.py`,
  `.gitignore`, `docs/source/usr_multi_gpu.rst`, `dev_scripts/`,
  `tests/{test_device_policy,test_memory_ledger,test_widening_floors}.py`

Notably, prerelease touches **none** of `_apply_device_policy`, `_settle`, the
placements, the banded drivers, or the ledger's phases. The two branches were
working in genuinely separate parts of the tree.

### Semantic: four interactions, three benign, one doc fix

1. **Stale policy docstring (needs a fix, not a conflict).**
   `tomography_model.py:894-897` asserts the denoiser's loop "cannot run" a
   multi-device layout. `2b1d02c` makes that false. Git will merge silently and
   leave a comment that is now wrong about the heaviest of its three reasons.
   **This is the only semantic collision that requires an edit.**
2. **Logging seam — compatible, verified.** `2b1d02c` makes the denoiser call
   `_log_device_report()`; `72e121f` changed `_device_report`'s rejection
   formatting in `parameter_handler.py:154-166`. A denoiser has empty
   `device_choice_rejections` and a non-automatic layout, so the new
   "used/rejected" clause never fires. Guard state (`_speed_floor_held`,
   `_speed_floor_fallback`) is `__init__`-initialized and `getattr`-read, so a
   model that never runs the policy is safe. No crash, no wrong log line.
3. **Per-entry policy re-evaluation vs the MAR loop — coherent, verified.**
   Traced in §5. `recon_plastic_metal`'s per-pass `recon` re-enters
   `_apply_device_policy`, but the returned shards and `ct_model.recon_placement`
   can never disagree, because the policy runs before the shards are produced and
   nothing in between changes the layout.
4. **Ledger blind spots — additive, not a collision.** §7.

### Test interaction

The prerelease tests pin devices explicitly (`configure_devices(devices=['cpu'])`
/ `['cpu','cpu']`), which sets `device_layout_is_automatic = False` and short-
circuits `_apply_device_policy`'s first branch — and on this CPU-only Mac
`torch.cuda.is_available()` is False anyway, so `visible < 2` short-circuits
too. The widening floors never run in them. Conversely, greg_dev's three test
files touch neither preprocessing nor denoising. I expect both suites to pass
post-merge unchanged, but **this was not executed** (it would require a merge)
and should be confirmed by running the full suite under
`/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python` once merged.

---

## 9. Merge-order recommendation

**Merge all five, in their existing order, as a unit. Nothing blocks.**

The order is not cosmetic — these are a dependency chain and must not be
cherry-picked apart:

- `2b1d02c` (denoiser) calls `from .utilities import _to_host` on a `Shards`,
  which only works because `c5b5438` taught `_to_host` about `Shards`.
- `944aec2` (BH) runs downstream of the sharded segmentation from `a5b04ce`;
  its `_ps_*` helper set is a direct generalization of `a5b04ce`'s `_per_shard`.
- `a5b04ce`'s own message names the `correct_sino_plastic_metal` gap that
  `944aec2` closes, so landing 1 without 5 leaves multi-device MAR half-built.

Because the file sets are disjoint and `merge-tree` is clean, a plain merge is
safe; no rebase and no conflict resolution is needed.

**Land-then-fix (recommended, in this order after the merge):**

1. **Rewrite the `_apply_device_policy` docstring** at
   `tomography_model.py:894-897`. The denoiser's loop now runs multi-device;
   reasons 2 and 3 still carry the ruling. Highest priority because it is the
   one thing the merge silently falsifies.
2. **Soften `a5b04ce`'s exactness claim** in the commit-derived docstring at
   `segmentation.py:60-72`, and make the `hi == lo` case raise instead of
   silently mis-binning against expanded edges.
3. **Route the denoiser's line-search sums through `combine_on_lead`** and pass
   `executor=self._per_device_pool` to its `run_per_device` calls
   (`denoising.py:376-397`) — brings it up to the standard `vcd_recon` already
   sets in the same repo.
4. **Restore `torch.mean` on the unsharded Sp-mean branch** (`mar.py:713-717`),
   or state the single-device change in the message rather than claiming none.
5. **Make `save_data_hdf5` gather per slab** rather than up front
   (`utilities.py:159-172`), so the streaming design survives sharded input at
   2K³.
6. **Decide preprocessing's device rule explicitly** (§6) — either a cheap size
   threshold or a recorded exemption, so "all visible devices" is a decision
   rather than an inherited default.
7. **Ledger phases for the denoiser and preprocessing** — a separate charter, not
   a fix.

**Nothing here justifies blocking.** Every defect found is either (a) a claim in
a commit message that is stronger than the code, (b) a performance divergence
from a standard the same repo sets elsewhere, or (c) a pre-existing gap that
these commits make reachable rather than create. All five commits move the
library from "crashes on multi-device" to "runs multi-device," and the disjoint
file sets mean the merge itself carries essentially no integration risk.
