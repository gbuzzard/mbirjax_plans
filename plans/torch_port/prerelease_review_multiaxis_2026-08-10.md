# Prerelease review: `16ff97c` (multiaxis port) and `410ccf8` (segmentation overlap)

Reviewed 2026-08-10 against `greg_dev` at `c4aa556`, not the fork point.
Read-only: nothing in the repo working tree was touched. All execution happened
in merged copies built in the scratchpad — `scratchpad/merged2` (the recommended
resolution) and `scratchpad/merged_theirsB` (commit B resolved THEIR way, to
test it).

Method reused from `scratchpad/prerelease_review_translation.md`, which covers
`1a2ced7` (translation) and is treated here as context: the two ports sit on the
same stack, share the same base hooks, and share several of the same gaps.

**Verdicts**

| commit | verdict |
|---|---|
| `16ff97c` MultiAxisParallelModel | **ISSUES — 0 HIGH, 4 MEDIUM, 8 LOW/NOTE. LAND NOW** |
| `410ccf8` segmentation softening | **ISSUES — 1 MEDIUM (merge conflict + a wrong message). LAND ONLY AS A RESOLVED MERGE**, taking ours wholesale |

Nothing found in either commit is a correctness defect in the projection path,
the device policy, or the ledger.

---

## 0. State of the branches

| | |
|---|---|
| commits under review | `16ff97c` (parent `1a2ced7`) and `410ccf8` (tip of `origin/prerelease`) |
| merge base with `greg_dev` | `944aec2` — the stack forks from BEFORE `a880d9c` (device policy) and BEFORE `c4aa556` (floors/ledger set, segmentation fix) |
| `greg_dev` | `c4aa556`, clean apart from `demo/demo_1_shepp_logan.py` |

`16ff97c`'s changed set: `mbirtorch/multiaxis_parallel.py` (new, 418 lines),
`mbirtorch/__init__.py`, `mbirtorch/utilities.py` (one 2-line hunk),
`tests/test_multiaxis.py` (new, 190 lines), `tests/generate_goldens.py`, and
three docs files (one a rename out of `_pending/`).

`410ccf8`'s changed set: `mbirtorch/preprocess/segmentation.py`,
`tests/test_sharded_segmentation.py` (docstring only).

---

# PART A — `16ff97c` "Port MultiAxisParallelModel from mbirjax"

## 1. The guard's unlisted-geometry fallback — CLEAN, and SOUND rather than merely conservative, but for a different reason than the framing suggests

**Declarations, verified live.** `MultiAxisParallelModel` declares NO
`_floor_family` (`'_floor_family' in MultiAxisParallelModel.__dict__` is False;
it inherits `None` from `tomography_model.py:1199`) — correct, multiaxis is
unmeasured. It declares no `rows_track_slices`, inheriting the base `False` at
`tomography_model.py:1191`, whose own comment already names "cone, translation,
multiaxis" as the False cases. No `_dc_damping`, no `split_sino_recon`
(`_memory_remedies()` returns `[]`).

**Inheritance to the policy site.** Verified at runtime that `recon`,
`vcd_recon`, `_apply_device_policy` and `_speed_ordered_candidates` are all the
base implementations (`m.recon.__func__ is TomographyModel.recon` etc.).
`vcd_recon` is the sole call site of `_apply_device_policy`, so multiaxis reaches
the one policy site.

**Own device-choosing code: NONE.** `multiaxis_parallel.py` contains ZERO
occurrences of `torch_device`, `projector_functions`, any `_sparse_*` driver,
`triton`, or `kernel_availability`. It is cleaner on this axis than the
translation port, which at least had one `self.torch_device` placement for its
FDK pre-weight; multiaxis has no direct-recon weight array at all
(`row_weight=None`). Every `device=` in the file is derived from an input
tensor's device inside a body. The uniformity ruling holds: one policy site.

**The family lookup, traced live** on the merged tree:

| sinogram | elements | candidate order | held |
|---|---|---|---|
| (16, 24, 20) toy | 7,680 | `[1, 4, 3, 2]` | 4, 3, 2 |
| (512, 448, 384) | 88,080,384 | `[2, 1, 4, 3]` | 4, 3 |
| (256, 700, 500) | 89,600,000 | `[2, 1, 4, 3]` | 4, 3 |
| (1024, 1008, 992) | 1,023,934,464 | `[4, 3, 2, 1]` | none |

Sample reason string, verbatim:

> `held by the speed floor: 0.0M sinogram elements < 88.1M (the parallel n=2 floor); configure_devices(num_devices=2) overrides (this model names no _floor_family, so the parallel floors apply)`

and the verbose-2 log line, captured off the model's own logger:

> `MultiAxisParallelModel names no _floor_family, so the parallel widening speed floors apply to its automatic device count.`

`_widening_floors.stale_note()` returns `None` after the merge: the port touches
none of `COST_INPUT_FILES` (`triton_parallel.py`, `triton_cone.py`,
`projectors.py`) nor `COST_INPUT_METHODS`, so `BLESSED_COST_HASHES` and
`TABLE_CHECKSUM` stay valid.

### Is the parallel-floors fallback the RIGHT family for multiaxis?

The brief asks whether, for a PARALLEL-family geometry, the parallel-floors
fallback is arguably the right family outright, and whether multiaxis's cost
structure matches parallel's closely enough. Three findings, in order of weight:

**(a) Today the question has no operational content: the two measured families'
floors are numerically IDENTICAL at every measured count.** Read live off
`_widening_floors.FLOORS`:

```
parallel floors: {2: 88_080_384, 4: 1_023_934_464}
cone floors    : {2: 88_080_384, 4: 1_023_934_464}
```

Both `n=2` floors are the 512-class shape and both `n=4` floors are the
1024-class shape. So substituting `parallel` for a hypothetical `cone` or
`multiaxis` family cannot change a single admission decision at present. The
fallback is sound, not merely conservative — but the soundness comes from the
table, not from a cost-structure argument.

**(b) On cost structure, multiaxis is cone-shaped, not parallel-shaped.** Its
projection is two separable fans through compiled TORCH bodies with an explicit
`psf` tap loop and a materialized `(Vb, P, R)` detector-column tensor. Parallel
beam's floors were measured with hand-written Triton kernels on
(`_widening_floors.MEASURED_CONFIG` says so explicitly: "Triton kernels on").
Parallel beam is also genuinely 2-D — `rows_track_slices=True`, no vertical fan
at all. So the mechanism that sets parallel's crossover is not multiaxis's
mechanism. Calling it "a parallel-family geometry" is right about the physics and
wrong about the cost model.

**(c) The direction of the residual error is the safe one.** Multiaxis does
strictly more work per sinogram element than parallel: an extra fan, no fused
kernels, and (see §4 below) an auto-set recon that can be far larger than the
detector. More per-device compute at the same sinogram size means the widening
crossover arrives EARLIER for multiaxis than for parallel, so parallel's floors
hold multiaxis at a smaller count slightly longer than optimal. By the module's
own measured asymmetry ("widening too early has cost multiples, while holding a
just-large-enough problem at a smaller count costs a few percent"), that is the
cheap direction.

**Conclusion:** the fallback is sound today and conservative in the right
direction. The exposure is forward-looking, and it is worth one sentence in
`_widening_floors.py`: if a future refresh ever separates the parallel and cone
numbers, multiaxis silently follows the wrong one, and nothing in the code or the
tests would notice. That is recorded as LOW-1 below rather than as a defect in
this commit.

### MEDIUM-1 — the refresh tooling still cannot see an unmeasured geometry (one fix covers translation AND multiaxis)

Unchanged from the translation review, now with two subjects instead of one.
`dev_scripts/refresh_widening_floors.py:178-182`:

```python
family = (getattr(cls, '_floor_family', None)
          if isinstance(cls, type) and issubclass(cls, TomographyModel) else None)
if family is not None and family not in known:
```

`MultiAxisParallelModel` arrives with `_floor_family = None` and is skipped, so
`unmeasured_families()` stays `{}` with BOTH new geometries importable. Secondary:
`_build_model` at `:236-243` has `if family == 'cone': ... else: ParallelBeamModel(...)`,
so a `'multiaxis'` family label would silently measure parallel beam.

**One fix covers both geometries** — this is the same code, not two instances.

---

## 2. The funnel — CLEAN

`multiaxis_parallel.py`'s complete set of engine touches, by line:

| line | call | verdict |
|---|---|---|
| 255-258 | `_view_batch_bodies()` returns the two module-level torch bodies | the geometry hook |
| 382 | `self._apply_direct_recon_filter(...)` | base shared FBP filter |
| 400 | `self.back_project(filtered_sinogram, output_sharded=True)` | routes through `sparse_back_project` |
| 401 | `self._gather_recon(recon)` | base chokepoint |

No reference to `projector_functions`, to any `_sparse_*` driver, or to a body
function anywhere else. Every engine projection routes through the public
`sparse_forward_project` / `sparse_back_project` pair, and the bodies are reached
only through `Projectors.sparse_{forward,back}_project_view_range`.

**Signature conformance against the driver's actual calls** (`projectors.py:388`
and `:433`) — checked term by term:

* forward: driver passes `(band_values, pixel_indices, view_params_batch,
  slice_start=, plan=, **args)`; body takes
  `(values, pixel_indices, view_params_batch, ..., slice_start=0, plan=None)`. ✓
* back: driver passes `(local_sino, pixel_indices, view_params_batch,
  coeff_power=, slice_start=, band_slices=, plan=, **args)`; body takes
  exactly those. ✓
* `_view_batch_args()` supplies exactly the 14 remaining names the bodies
  declare — no extra key, no missing key (verified by sorted-key comparison).

Note that multiaxis's `fbp_filter` does NOT call `_shard_sinogram` itself; it
relies on `_apply_direct_recon_filter` to shard a plain input, matching parallel
beam and mbirjax. That is correct — `fbp_recon` passes `output_sharded=True`
through and `back_project` consumes the sharded form directly, so there is no
host round-trip in the middle.

---

## 3. The memory ledger — it prices multiaxis; `rows_track_slices` is False and that is correct; one MEDIUM, now MEASURED

**It never returns None and it never raised.** `_build_memory_ledger` →
`plan_from_model` → `estimate_peak_device_bytes` built successfully at 1, 2 and 4
devices on both a toy and a production shape, so `_apply_device_policy` does NOT
reach the `ledger is None` branch at `tomography_model.py:991` and does not skip
the capacity preflight for multiaxis.

**`rows_track_slices` is False, and the shapes hold.** This is the brief's
question, so it is worth being explicit about why False is right and what it
buys:

* At elevation 0 multiaxis IS parallel beam and rows would track slices 1:1. At
  nonzero elevation the slice-to-row map is `m(k) = m0 + slope*k` with `m0`
  carrying the pixel's in-plane depth `y` (`multiaxis_parallel.py:79-80`), so one
  slice lands on different rows for different pixels. Row-alignment is false for
  the general geometry, and the base value is the correct declaration.
* The consequence in the ledger is `forward_block_rows(i) = plan.sino_rows`
  rather than the band (`_memory_ledger.py:411`). That is exactly right for this
  body: `_multiaxis_forward_view_batch` allocates
  `det_col = torch.zeros((vb, num_pixels, num_rows_r))` with `num_rows_r` read
  from the params, so its output spans every detector row whatever slice band it
  was handed. A row-aligned declaration would have under-charged the forward by
  `(rows - band)` per view, which is the specific failure
  `_memory_ledger.py:392-410` warns about.
* It also means `_sino_row_padding()` returns None, so the sinogram keeps its
  real detector rows under sharding — correct, since nothing ties them to the
  padded slice axis.

Every other plan field, read live:

| plan field | value | why it is right |
|---|---|---|
| `helical` | `False` | `'view_params_array' not in params` (multiaxis stores `angles`), so no spurious helical z-weight term |
| `hessian_masked` | `True` | multiaxis leaves `use_ror_mask` at the default True, matching mbirjax (checked on the jax side: `use_ror_mask True`), so the hessian is priced at the masked index set and the `hessian scatter` sub-phase is emitted |
| `num_pixels_full` / `num_pixels_grid` | 276 / 400 on the toy | consistent with an active ROR mask; the in-plane circular ROR is geometrically right for a parallel in-plane geometry |
| `sino_rows` | `sinogram_shape[1]` | follows from `rows_track_slices=False` |
| `view_charge` | resolves | closure over `_view_batch_bodies()` + `_view_batch_args()` binds cleanly at every count |

Priced at (256, 1900, 3000), auto recon (3000, 3000, 2062):

```
n=1  peak 297.74 GiB   dominant "hessian diagonal"          (back output 162.8, init recon 69.1, back batch 54.3)
n=2  peak 176.35 GiB   dominant "hessian diagonal [back workers]"
n=4  peak 115.71 GiB   dominant "hessian diagonal [back workers]"  (back batch 54.3, back output 27.2)
```

### MEDIUM-2 — the ledger's projection terms rest on the torch-body slab proxy, and the shortfall is now MEASURED at ~10x

Neither multiaxis body carries a `_view_batch_cost` (verified:
`hasattr(fwd, '_view_batch_cost')` is False for both), so
`Projectors.view_batch_charge` falls to the gather-slab rule at
`projectors.py:340-341`: `bytes_per_view = num_pixels * _transient_cols(band_cols) * 4`
— ONE slab. `projectors.py:196-201` already says the proxy under-states reality
by "a small multiple (~2-5x)".

The translation review counted live tensors and cited that 2-5x. Here I measured
it instead, with a single-variable ablation: identical process, identical
pre-allocated inputs and outputs, one body call or none, `ru_maxrss` high-water
difference. P = 20,000, S = 277, R = 256:

```
                        ledger charge   measured body peak   ratio
multiaxis forward vb=1      21.1 MiB          286.2 MiB      13.5x     (eager)
multiaxis forward vb=4      84.5 MiB         1088.8 MiB      12.9x     (eager)
multiaxis back    vb=1      21.1 MiB          379.2 MiB      17.9x     (eager)
multiaxis back    vb=4      84.5 MiB         1249.6 MiB      14.8x     (eager)
translation forward vb=4    78.1 MiB         1246.6 MiB      16.0x     (eager)
translation back    vb=4    78.1 MiB          351.6 MiB       4.5x     (eager)

through the bodies the DRIVER binds (maybe_compile applied):
multiaxis forward vb=4      84.5 MiB          843.7 MiB      10.0x
multiaxis back    vb=4      84.5 MiB          360.8 MiB       4.3x
```

The ratio is stable across `vb=1` and `vb=4`, which is the meaningful signal: it
is a genuine per-view multiple, not a fixed overhead. Compile fusion helps
(13x → 10x on the forward) but does not close it.

**This is a documented exposure, now quantified, and it is worse than the
`projectors.py` comment's own 2-5x estimate on the forward side.** Two
consequences:

1. `VIEW_BATCH_SINO_MULTIPLE = 8` sizes the batch CHOICE with a multiplier baked
   in, so the driver is protected. The LEDGER is not: it consumes
   `view_batch * bytes_per_view` directly as its `forward batch` / `back batch`
   charge with no multiplier. At the production shape `back batch` is 54.3 GiB of
   the 115.7 GiB modeled n=4 peak; a 4-10x shortfall on that term puts the true
   peak far above the model — the one direction `_memory_ledger`'s own comments
   (`:316-320`, `:745-748`) say it may not err in.
2. Cone and parallel are shielded because their Triton bodies carry a real
   `_view_batch_cost`. Multiaxis and translation have no kernels, so the proxy is
   their permanent state, not a fallback.

**Not a port defect** — the bodies are a structural clone of cone's torch bodies.
It is an exposure the port makes permanent, and this review supplies the number
the earlier one could only bound. Action: one `MBIRTORCH_MEMORY_CALIBRATION=1`
reconstruction on CUDA at a multiaxis shape, checked against `CALIBRATION_BAND`,
before any production multiaxis run. Not a blocker for landing.

### NOTE — auto geometry sizes the recon from the CHANNEL count, and the elevation clamp can inflate slices 10x

`auto_set_recon_geometry` (`multiaxis_parallel.py:330-363`, verbatim from
mbirjax) sets the in-plane extent from `max_u = num_det_channels * delta/2` and
the slice count from `max_v / max(min|cos el|, 0.1)`. Consequences, measured:

```
sinogram (256, 1900, 3000)      -> recon (3000, 3000, 2062)   = 18.6e9 voxels
64-row detector, max_el 20 deg  -> 68 slices
64-row detector, max_el 80 deg  -> 368 slices
64-row detector, max_el 89 deg  -> 640 slices   (the 0.1 clamp, 10x the rows)
```

Nothing is wrong here — it is faithful to mbirjax and the `>45 deg` constructor
warning fires — but a production-scale multiaxis panel implies a recon several
times larger than a cone recon at the same detector, and the ledger's 297 GiB at
n=1 says so. Worth a line in the user doc.

---

## 4. Body and kernel selection — CLEAN; the kernel probes correctly stay out

`_view_batch_bodies` (`multiaxis_parallel.py:255-258`) returns the two
module-level torch bodies unconditionally, with a plain-English one-line reason,
and imports nothing from `kernel_availability`.

**Do the Triton kernel-availability probes engage for multiaxis?** No, and that
is sound rather than a gap. The port does NOT reuse the parallel bodies — it has
its own two-fan bodies, and the probes are invoked ONLY from `cone_beam.py:370-391`
and parallel's equivalent, never generically (confirmed by grep across
`mbirtorch/`). So the parallel/cone-specific probes stay out of multiaxis's way by
construction, and there is nothing for multiaxis to self-check: it has no kernel
variant whose value would need proving. This differs from the case the brief
raises — if multiaxis had reused `_parallel_forward_view_batch_triton`, the probes
would matter, because parallel's kernels assume `rows_track_slices` semantics
that multiaxis does not have. It does not reuse them; it reuses only the shared
`horizontal_fan.fan_forward_batch` / `fan_back_batch`, which are geometry-agnostic
trapezoid fans consumed the same way cone and translation consume them.

What multiaxis DOES omit relative to mbirjax is the sorted-channel-reduce tile
policy: `mbirjax/multiaxis_parallel.py:_select_tile_policy` enables
`sort_by_channel` on GPU (mbirjax measured 1.2-1.4x forward) behind three guards.
mbirtorch's shared fan has no sorted branch to enable, so this is a performance
gap in the shared kernel layer, not a port omission. NOTE.

`_transient_cols` (`multiaxis_parallel.py:276-281`) is the params-derived
`max(recon_shape[2], sinogram_shape[1])`, band-independent — verified:
`_transient_cols(1)` and `_transient_cols(999)` both return 27 on the toy model
(S=27, R=24). Correct, because both bodies hold `(Vb, P, R)` and `(Vb, P, S)`
transients regardless of the band handed to them. Note this is the one place
multiaxis needs `max(S, R)` rather than cone's `max(recon_shape[2],
sinogram_shape[1])` for a different reason: the auto geometry can make S much
larger than R (2062 vs 1900 at production), so the `max` is load-bearing here
rather than a formality.

The torch-body path engages cleanly: `Projectors.__init__` runs `maybe_compile`
on both bodies per device and the full merged suite is green. (One false alarm
investigated and dismissed: an early probe script that built many models in one
process tripped `torch._dynamo` `recompile_limit` on `_multiaxis_back_view_batch`.
A single-model-per-process ablation across all four geometries shows 0 hits for
multiaxis, parallel and cone, and 2 for translation on an unrelated
`qggmrf_gradient_and_hessian_at_indices`. Artifact of the probe, not the port.)

---

## 5. Merge collision vs `origin/greg_dev` (c4aa556)

**Textual.** `git merge-tree 944aec2 HEAD 410ccf8` (the whole three-commit stack,
which is what will actually merge):

* **Conflict markers appear in exactly two files, both from commit B** —
  `mbirtorch/preprocess/segmentation.py` and `tests/test_sharded_segmentation.py`.
  See Part B.
* `mbirtorch/utilities.py` is "changed in both" and merges CLEAN. The multiaxis
  hunk is a 2-line edit at old line 396 (`_resolve_geometry_class`'s name tuple);
  translation's are at 1109 and 1409; `greg_dev`'s are at 244-286 and 1367-1412.
  `git merge-file` produced a clean merge, used for all testing below.
* `mbirtorch/__init__.py`, `tests/generate_goldens.py`, `tests/test_demo_data.py`,
  `docs/source/usr_utilities.rst` all merge clean.
* `docs/source/_pending/usr_multiaxis_parallel_beam_model.rst` shows as "removed
  in remote" — it is the rename into `docs/source/`, resolved by rename detection.
* `mbirtorch/multiaxis_parallel.py`, `tests/test_multiaxis.py` are pure additions.
* The port touches none of `_memory_ledger.py`, `_widening_floors.py`,
  `projectors.py`, `tomography_model.py`, `vcls.py`, `parameter_handler.py`,
  `cone_beam.py`, `parallel_beam.py`, `horizontal_fan.py`.

**Suite on the merged tree** (`scratchpad/merged2`, commit B resolved as ours):

```
449 passed, 68 skipped, 91 deselected   in 98.46s
```

against `437 passed, 68 skipped` on `HEAD` alone — re-measured in this session on
a fresh `git archive HEAD` tree, `437 passed, 68 skipped, 79 deselected`, not
carried over from the earlier review — and `442 passed, 68 skipped` on
HEAD+translation. **+7 tests from multiaxis** (adjointness, zero-elevation
equivalence and recon smoke on each of the two backends, plus the 2-shard parity
test), **zero regressions**. `tests/test_device_policy.py`,
`tests/test_memory_ledger.py`, `tests/test_widening_floors.py`,
`tests/test_sharded_*` all pass unchanged.

**Semantic.** The port assumes nothing about the files `greg_dev` moved — it
consumes only base hooks that all survive (`_view_batch_bodies`,
`_view_batch_args`, `_transient_cols`, `_apply_direct_recon_filter`,
`_gather_recon`, `back_project`) plus `horizontal_fan.fan_{forward,back}_batch`.

### MEDIUM-3 — geometry enumerations in `greg_dev`'s own docs go stale (shared with translation, ONE fix covers both, plus a fourth site)

* `docs/source/usr_multi_gpu.rst:9` — "It works for the parallel-beam and
  cone-beam geometries." False after the merge (four geometries).
* `docs/source/usr_multi_gpu.rst:26` — "per-geometry, because parallel-beam and
  cone-beam reach the crossover at different sizes." Now doubly incomplete, and
  this is exactly the page that should explain the substituted-floors case that
  BOTH new geometries now exercise. (It should also say what §1(a) found: the two
  measured families' floors are currently identical, so "different sizes" is not
  even true of the two it names.)
* `docs/source/overview.rst:39` — "MBIRTorch supports the *parallel-beam* and
  *cone-beam* imaging geometries".
* **`docs/source/dev_sharding_overview.rst:14` — "It works for the parallel-beam
  and cone-beam geometries."** NOT in the translation review's list; found here.

One editing pass fixes all four for both new geometries.

---

## 6. Port checks

### 6.1 mbirjax parity — verified independently, PASSES

The local `tests/goldens/golden_64x64x64.npz` predates both ports (no `ma_*` and
no `tct_*` keys), so all six multiaxis golden gates SKIP in a normal run here. I
regenerated the multiaxis golden section from the mbirjax sibling checkout
(`/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python`, cwd = the mbirjax repo,
jax 0.10.1) with my own script rather than the port's, and ran the gates against
the merged tree:

```
auto geometry     torch (28, 28, 34) == jax (28, 28, 34), delta_voxel 1.0 both
sparse_forward    rel_max = 8.69e-07
sparse_back       rel_max = 3.30e-07
hessian (p=2)     rel_max = 1.05e-06     [not gated by the port's tests -- added here]
full forward      rel_max = 9.46e-07
fbp               rel_max = 8.92e-07
recon@3   alpha rel = 6.64e-06, fm rel = 4.69e-06, volume rel_max = 1.10e-05
recon@10  volume rel_max = 6.77e-06
```

plus checks the port's tests do not make:

```
adjointness (torch pair, exact)            rel = 0.00e+00
banded forward, tiling over slices         rel_max = 3.25e-07
banded back, concatenating bands           rel_max = 0.00e+00
zero elevation vs mbirtorch ParallelBeam   rel_max = 8.93e-08
```

The banded checks matter because the port ADDS `slice_start` / `band_slices` that
mbirjax's forward does not have (mbirjax's forward is monolithic by design), so
the tiling-invariance property is a mbirtorch-only claim and is now gated
independently. It holds exactly.

**An independent line-by-line read of both sources found no divergence** in: the
coordinate stages (`recon_ijk_to_xyz` → `geometry_xyz_to_uv` → `detector_uv_to_mn`,
including `v = z*cos(el) + y*sin(el)` and the offset signs); `z_0 =
-delta_voxel_slice*(num_slices-1)/2 + recon_slice_offset` anchored on the REAL
slice count; `slope = delta_voxel_slice*cos(el)/delta_det_row`; the max-of-edges
vertical footprint (`_vertical_footprint_phys` vs `_multiaxis_vertical_terms`,
all three edges term-for-term) with NO floor; `L_max = min(1, W_p_r)`; the
trapezoid `clip((W_p_r+1)/2 - |m_p - m|, 0, L_max)` then the validity mask;
the mass-conserving amplitude `(delta_voxel_slice/delta_det_row)/W_p_r` folded
into the forward's VALUES and applied to the back's cylinder at `coeff_power`;
the horizontal `footprint_xy = max(|cos az|*delta_voxel, |sin az|*delta_voxel_row)`
and `weight_scale = delta_voxel*delta_voxel_row/footprint_xy`, both per-VIEW;
`get_psf_radius` including the max-of-each-edge-over-actual-elevations rule and
the `angles is None` fallback; `auto_set_recon_geometry` including the
`min_cos_el` 0.1 clamp; and the FBP filter's `1/(delta_voxel*delta_voxel_row)`
scale with `row_weight=None`.

Two algebraic re-orderings, both exactly equivalent and worth naming because they
look like differences: mbirjax applies `scaling ** coeff_power` AFTER the row sum
while the port folds it into each tap (`A * scale_pow`); and mbirjax clips to
`[0, L_max]` in one call while the port does `clamp(min=0)` then `minimum(L_max)`.
Neither changes a value.

**Deliberate deltas, all in the safe direction:** the port adds the banded
forward and banded back (mbirjax has a banded back but a monolithic forward), and
adds `view_batch_size` / `compile_mode` constructor arguments matching the other
mbirtorch geometries. The port OMITS mbirjax's two memory knobs
(`MULTIAXIS_SLICE_BAND_SIZE`, `MULTIAXIS_FORWARD_SLICE_BATCH`) and the
`_select_tile_policy` sorted-channel-reduce.

### 6.2 The port's stated "measured parity floor" is honest and reproduces — but the gate is applied to a case 100x tighter than the case it was measured on

The commit message and `tests/test_multiaxis.py`'s module docstring claim: "at the
dividing case (16 views, elevations to 29 deg) the seeded 3-iteration recon
differs from mbirjax by 1.2e-3 max, decaying to 4.2e-4 by 10 iterations". I
generated that exact configuration from the jax side and measured:

```
_small_ma config (16 views, 24x20, el +-0.5 rad = 28.6 deg), recon (20, 20, 27):
   forward vs mbirjax          rel = 1.64e-06
   recon@3  vs mbirjax         rel = 9.39e-04     (port claims 1.2e-3)
   recon@10 vs mbirjax         rel = 4.27e-04     (port claims 4.2e-4 -- exact)
   alpha rel 1.07e-05, fm rel 7.55e-06
```

**The claim reproduces.** And a cheap ablation settles the mechanism: the SAME
code run as 2 CPU shards instead of 1 device — a pure float-summation-order
perturbation, no cross-framework difference at all — diverges from single-device
by `9.387e-04`, essentially identical to the `9.386e-04` cross-framework number.
Meanwhile the raw projector pair is exact to float rounding under sharding:

```
2 shards vs 1: sparse_fwd 3.67e-07  sparse_back 1.78e-07  hessian 1.47e-07
3 shards vs 1: sparse_fwd 3.67e-07  sparse_back 1.78e-07  hessian 2.21e-07
                forward 1.37e-07  back 1.23e-07  fbp 8.88e-08 / 1.78e-07
```

So the divergence is 3-4 orders of magnitude of trajectory amplification of 1e-7
float noise over 3 VCD iterations at this configuration, not a systematic
geometry or weight difference. The port's characterization ("trajectory float
noise around one fixed point") is correct and is now corroborated by an
independent mechanism.

### MEDIUM-4 — the golden convergence gate is ~450x looser than the parity it gates

`test_multiaxis_recon_convergence_parity` asserts `final_rel < 5e-3`, but it runs
on the GOLDEN configuration (24 views, elevations to +-0.4 rad), not the
`_small_ma` configuration the 5e-3 was measured from. Measured parity on the
golden configuration is `1.10e-05` at 3 iterations and `6.77e-06` at 10 — the gate
is 450x above it and would pass a 100x regression in the port's recon path
unnoticed.

The docstring is honest about where the number came from ("at the dividing case
(16 views, elevations to 29 deg)"), so this is a calibration slip rather than a
misrepresentation: one volume tolerance was set from the worst case and applied to
both tests. It is correct for `test_multiaxis_sharded_recon_matches_single_device`
(measured 9.4e-4, a 5.3x margin, which is right for a chaotic trajectory) and
much too loose for the golden test.

Recommended: keep 5e-3 on the sharded test; tighten the golden test to `1e-3`
(100x margin over the measured 1.1e-5, still 100x above it for platform headroom)
and add one clause saying the two tests are gated at different configurations for
that reason. Optionally add the `coeff_power=2` hessian gate — it is a
golden-cheap addition and it is the one operator the port's tests do not compare
(I checked it independently at 1.05e-06).

### 6.3 Test coverage delta

Added: `tests/test_multiaxis.py` (11 tests — adjointness, zero-elevation
equivalence with `ParallelBeamModel` and a recon smoke on each backend, 2-shard
parity, and six mbirjax golden gates), plus the `tests/generate_goldens.py`
multiaxis section.

The zero-elevation test is a genuinely good addition the translation port had no
analogue for: it pins the reduction-to-parallel-beam property that the class
docstring claims, and it passes at 8.9e-08.

Gaps (LOW):
* `tests/test_device_policy.py:700-704` still uses the synthetic
  `_UnlistedGeometry` stand-in whose docstring says "as TranslationModel and any
  future class would arrive". TWO real classes now arrive that way and neither is
  the subject of the standing test for the fallback path.
* No multiaxis case in `tests/test_memory_ledger.py`, `tests/test_view_batching.py`,
  `tests/test_adjoint.py` or `tests/test_sharded_pipeline.py` (grep count 0 in all
  four). Same gap translation has.
* No test for the geometry's own edge cases, all of which I verified by hand and
  all of which behave: 1-D angles, 3-column angles and a view-count mismatch all
  raise `ValueError` from the constructor; `verify_valid_params` catches a
  post-hoc `set_params(angles=...)` mismatch; the `>45 deg` warning fires at 46,
  80 and 89 degrees and not at 20; and a full 90-degree (top-down, slope 0)
  elevation set produces a finite forward projection — which is the port's
  central structural claim (scatter forward rather than a row-to-slice inversion)
  and is currently ungated.

### 6.4 Goldens impact

`tests/goldens/` is gitignored (generated on demand), so the commit ships a
generator change, not data. Every parity test is `skipif`-guarded on
`"ma_sino" in golden.files`, so on any unrefreshed machine the six strongest
gates skip silently in a normal run. Correct design, but CI/nightly must
regenerate or both new geometries' parity claims are dark. Same NOTE as
translation, now doubled.

### 6.5 Integrations — verified by running each against a live model

#### MEDIUM-5 — the exported `MultiAxisParallelBeamModel` alias silently resolves to `ParallelBeamModel`

`multiaxis_parallel.py:418` exports
`MultiAxisParallelBeamModel = MultiAxisParallelModel` as "the backward-compatible
public API name used throughout docs/examples", and `__init__.py` imports it. But
the commit's own `utilities.py` hunk lists the CANONICAL name:

```python
for name in ('ConeBeamModel', 'MultiAxisParallelModel', 'ParallelBeamModel',
             'TranslationModel'):
    if name in geometry_type:
```

`'MultiAxisParallelModel'` is NOT a substring of `'MultiAxisParallelBeamModel'`
(`Parallel` + `Model` vs `Parallel` + `BeamModel`), but `'ParallelBeamModel'` IS.
Observed live:

```
_resolve_geometry_class(str(type(m)))                 -> MultiAxisParallelModel   (correct)
_resolve_geometry_class("MultiAxisParallelBeamModel") -> ParallelBeamModel        (WRONG, silent)
```

Worth noting the reorder in the commit is inert for the real class string —
neither `'MultiAxisParallelModel'` nor `'ParallelBeamModel'` is a substring of the
other, so the ordering change accomplishes nothing. It reads as if it were meant
to guard exactly this case and does not.

**Blast radius today is small but the failure is silent and it is a
wrong-geometry reconstruction, not an error.** Every in-tree producer writes
`str(cls)` (`preprocess/nsi.py:149`, `pymbir.py:88`, `zeiss.py:147,149`,
`zeiss_tct.py:120`, `get_all_params`), and `str()` of the alias is the same string
as `str()` of the class, so the round trip is safe (verified: `get_all_params` →
`build_model` rebuilds a `MultiAxisParallelModel` with an identical
`recon_shape`). The trap is a hand-written or doc-copied
`geometry_type='MultiAxisParallelBeamModel'` in a params dict or a saved file —
which is precisely the usage the alias exists to serve.

Fix: add `'MultiAxisParallelBeamModel'` to the tuple BEFORE `'ParallelBeamModel'`,
or match on `'MultiAxisParallel'`. One line.

#### `copy_ct_model` — refuses multiaxis, but a ONE-LINE gate widening is enough (unlike translation)

`mbirtorch/utilities.py:791-796` raises
`TypeError: copy_ct_model() is restricted to ConeBeam and ParallelBeam Models`,
observed verbatim. The brief asks whether one fix covers translation and
multiaxis. It does not, and the asymmetry is worth stating:

* multiaxis's `get_all_params()` required dict is
  `['angles', 'geometry_type', 'sinogram_shape']` — `angles` IS the key
  `copy_ct_model:803` reads, and `len(new_angles)` on a `(V, 2)` array is `V`, so
  `:824`'s `new_shape[0] = len(new_angles)` is correct too. I patched the type
  gate alone and ran it: `copy_ct_model` returned a `MultiAxisParallelModel` with
  `sinogram_shape (8, 24, 20)`, `angles (8, 2)`, `recon_shape (20, 20, 24)`.
  **Correct end to end.**
* translation's required dict is
  `['geometry_type', 'sinogram_shape', 'source_detector_dist', 'source_iso_dist',
  'translation_vectors']` — no `angles` at all, so `:803` `KeyError`s and the
  `angles` assumption must be generalized.

So: one line for multiaxis, a real generalization for translation. Doing multiaxis
first is cheap and low-risk.

#### `get_ct_model` — refuses multiaxis; a plain `elif` suffices (unlike translation)

`mbirtorch/utilities.py:1013-1021` raises
`ValueError: Invalid geometry type. Expected cone or parallel, got multiaxis`,
observed for `'multiaxis'`, `'multiaxis_parallel'` and `'MultiAxisParallelModel'`.
Its existing `angles` positional takes a `(V, 2)` array unchanged, so multiaxis
needs only an `elif` with no signature change; translation needs a new
`translation_vectors` parameter. Again one fix does not cover both, and multiaxis's
half is trivial.

#### What DOES work (checked, no change needed)

* Construction, `auto_set_recon_geometry` (from `TomographyModel.__init__`),
  `verify_valid_params`, `recon`, `fbp_recon`, `fbp_filter`, `direct_recon`,
  `direct_filter`.
* `get_all_params` → `build_model` round trip reproduces `recon_shape` exactly and
  keeps the `(V, 2)` angles array.
* Sharded recon at 2 and 3 CPU shards; sharded projector pair exact to float
  rounding (§6.2).
* `preprocess/mar.py:920` correctly falls through
  (`'cone' in '...MultiAxisParallelModel...'` is False), so multiaxis takes the
  plain `recon` path.
* `_memory_remedies()` returns `[]` (no `split_sino_recon`), correct.
* Doc plumbing: `usr_api_overview.rst` and `usr_geometry_models.rst` both add the
  entry, the `_pending/` page is promoted, and `_pending/README.rst` correctly
  drops to "(none currently)" — the staging area is now empty, which is the
  intended end state of that mechanism.

#### LOW / inherited

* `hsnt.py:386` `VALIDATION_RULES` allows only `(None, 'parallel', 'cone')` for
  `dataset_geometry`; a multiaxis HSNT dataset would warn. Same as translation.
* `mbirtorch.__all__` gains `MultiAxisParallelModel` but not
  `MultiAxisParallelBeamModel`, while `__init__.py` imports both. The
  `_pending/README.rst` invariant is "documented if and only if declared"; the
  alias is imported, not declared, and not documented. Consistent, but combined
  with MEDIUM-5 it means the alias is a half-supported name.
* `vcls.get_opt_views` takes a `ct_model`, so it reads `view_params` off the model;
  its docstring enumerates only Parallel/Cone. Not exercised for multiaxis.

---

## 7. The comment guideline — CLEAN

No `checklist`, `campaign`, `item N`, `mg3`/`mg4`, `floor_N`, `decision X`,
`phase N`, plan notation, `TODO`, `FIXME` or `XXX` anywhere in
`mbirtorch/multiaxis_parallel.py` or `tests/test_multiaxis.py`. "checklist B"
appears only in the commit message, which is where the guideline allows it.

This is cleaner than the translation port, which carried three borderline lines.
Notably, mbirjax's own source at the corresponding places DOES carry plan
notation ("decision C", a `plans/sharding/...` path in `_sharded_histogram`), and
the port dropped all of it rather than copying it through — the right call, and
worth noticing as evidence the guideline was applied deliberately rather than by
luck.

The one repeated coined phrase, "``plan`` is the memoization slot for a future
sorted/CSR stream variant; unused today" (`:109-110`, `:169-170`), is verbatim
from `projectors.py:372-373` and cone's and translation's bodies — consistent
house style, now in a fourth file. Same LOW as the translation review noted; the
fix is to define it once in `projectors.py` and stop restating it.

---

# PART B — `410ccf8` "Soften the sharded-histogram exactness claim; raise on a constant volume"

## B.1 The two versions side by side, and what `git merge-tree` produces

Both sides make the SAME TWO fixes to `_sharded_masked_histogram` — soften the
exactness claim, and raise instead of binning a degenerate range. They are
**overlapping, not conflicting in intent**, but they are **textually conflicting**
because they rewrite the same two regions.

`git merge-tree 944aec2 HEAD 410ccf8` produces **CONFLICT MARKERS, not a clean
merge and not a silent double-application** — three `<<<<<<<` markers across two
files:

| file | result |
|---|---|
| `mbirtorch/preprocess/segmentation.py` | 2 conflict hunks: the docstring block, and the raise block |
| `tests/test_sharded_segmentation.py` | 1 conflict hunk: the module docstring |

That is the good outcome. A silent double-application (two raises stacked, or two
softening paragraphs) is the failure mode to fear here, and git does not produce
it: the hunks overlap so the conflict is forced. Nothing else in the file
collides — `scale = num_bins / (hi - lo)` is IDENTICAL on both sides, so that line
merges without a marker.

Side by side:

| | ours (`c4aa556`) | theirs (`410ccf8`) |
|---|---|---|
| non-finite (all-masked / empty) | `if not (np.isfinite(lo) and np.isfinite(hi)): raise ValueError('The sharded volume has no valid entries to histogram: every shard was empty or entirely excluded by valid_mask.')` | none — falls into the constant-volume branch |
| degenerate range | `if hi <= lo: raise ValueError(f'The valid entries span the degenerate range [{lo}, {hi}] (min == max), so there are no intensity classes to separate. Segmentation needs a volume that takes more than one value.')` | `if not hi > lo: raise ValueError('Cannot segment a constant volume: all valid voxels ' f'equal {lo}.')` |
| why it must raise | 6-line comment: numpy EXPANDS a zero-width range to `(lo-0.5, lo+0.5)`, so counts and edges would describe different partitions | 2-line comment, same reason, more compact |
| divergence claim | a MEASUREMENT: 200 trials of 20 000 uniform float32 into 1024 bins, at most 3 per trial (mean 0.36), most sitting exactly ON an interior edge | an unmeasured rate: "a few counts per thousand can land one bin over" |
| framing | names the mechanism (float64 + ULP correction pass vs float32 truncate), states "expected not to" is the claim, not "cannot" | adds a genuinely good point ours lacks: "(Any histogram-based threshold is approximate to begin with — an order statistic of float data cannot be recovered from a fixed number of bins.)" |
| mbirjax citation | "mbirjax records the same divergence in its own port and deems it irrelevant at Otsu's granularity; this is at parity with that, not stricter than it" | "Irrelevant at Otsu's granularity; mbirjax records the same divergence for its sharded histogram" |
| `Raises:` section | yes | no |
| public `multi_threshold_otsu` docstring | updated (a 5-line paragraph at `:164` telling a USER the sharded path bins in float32 and raises on a constant volume) | not updated |
| tests | +2 (`test_degenerate_sharded_histogram_raises`, `test_all_padding_sharded_histogram_raises`) | none; docstring-only test-file change |

## B.2 Which is better on each axis

**The raise's condition — ours, clearly.** Ours splits two genuinely different
failures; theirs collapses them and mis-describes one. Verified by running our
tests against their code (`scratchpad/merged_theirsB`):

```
FAILED test_degenerate_sharded_histogram_raises   -- expected 'degenerate range',
        got 'Cannot segment a constant volume: all valid voxels equal 0.3.'
FAILED test_all_padding_sharded_histogram_raises  -- expected 'no valid entries',
        got 'Cannot segment a constant volume: all valid voxels equal inf.'
2 failed, 7 passed
```

The second line is the substantive defect, not a wording preference. With
`valid_mask` all False, `lo` stays `+inf` and `hi` stays `-inf`; theirs evaluates
`not (-inf > inf)` → True and reports **"all valid voxels equal inf"** for a
volume that has no valid voxels at all. A user chasing a bad mask is told their
data is constant and infinite. Ours names the actual condition.

**The raise's condition, edge cases.** `hi <= lo` and `not hi > lo` are equivalent
on finite non-NaN values, so neither raises where the other returns on ordinary
data. They differ only on NaN, where `hi <= lo` is False but `not hi > lo` is
True — and ours still raises there, because `np.isfinite(nan)` is False catches
it first. (Ours' message would be the misleading one in that specific case; a
NaN-only volume reports "no valid entries". Worth one clause, LOW.) Both drop NaN
values from pass 2 identically via `vals[(vals >= lo) & (vals <= hi)]`.

**The docstring's claim — ours on accuracy, theirs on one framing point.**

I reproduced our docstring's measurement on the merged tree:

```
trials=200 N=20000 bins=1024: worst displaced per trial = 3, mean = 0.360
  as a rate: worst 0.150 per thousand, mean 0.018 per thousand
```

Exactly the numbers our docstring records. Theirs says "a few counts per thousand
can land one bin over", which at N=20 000 would be 60-100 displaced values per
trial. The measured worst is **3**. Theirs overstates the divergence by roughly
20x (against the worst case) to 200x (against the mean), and it is unsourced.

**Against mbirjax's language.** mbirjax's `_sharded_histogram`
(`mbirjax/preprocess/segmentation.py:132-135`) says:

> A value exactly on an interior bin edge can differ from numpy's edge arithmetic
> by one bin (float rounding of the scaled index) -- irrelevant at Otsu's bin
> granularity.

mbirjax scopes the divergence to values sitting exactly ON an interior edge and
calls it irrelevant at Otsu's granularity. **Ours reproduces both halves** ("most
of those are values sitting exactly ON an interior bin edge, where the two rules
break the tie differently"; "mbirjax ... deems it irrelevant at Otsu's
granularity; this is at parity with that, not stricter than it"). **Theirs
reproduces the conclusion but not the scope** — it says "a few counts per
thousand" without the on-edge mechanism, which is why its number is wrong.

Worth noting for context: mbirjax's own HEADLINE is still the strong claim
("matching np.histogram, with EXACT int64 counts"; `multi_threshold_otsu` says
"bit-identical counts"), with the softening in a later paragraph. Both mbirtorch
versions are more careful than mbirjax's top line.

**The one thing theirs has that ours does not** is the parenthetical:

> (Any histogram-based threshold is approximate to begin with -- an order
> statistic of float data cannot be recovered from a fixed number of bins.)

That is a true and useful reframing — it says the whole exactness question is
downstream of a coarser approximation nobody objects to. It belongs in the merged
result.

**Coverage.** Ours updates the PUBLIC `multi_threshold_otsu` docstring; theirs
does not. That is where a user looks, and it is the difference between an internal
note and a documented contract.

## B.3 Recommended merge resolution — exact

**Take OURS wholesale for both files, then add one sentence from theirs.**

1. `mbirtorch/preprocess/segmentation.py`, docstring hunk — **take `.our`
   entirely** (the measurement, the mechanism, the `Raises:` section), then insert
   theirs' one parenthetical into our paragraph. Concretely, after our sentence
   ending `...where the two rules break the tie differently.` add:

   > `(Any histogram-based threshold is approximate to begin with -- an order`
   > `statistic of float data cannot be recovered from a fixed number of bins.)`

   Take NOTHING else from `.their`: its "a few counts per thousand" is
   contradicted by measurement, its `Returns:` block drops information ours keeps,
   and its `The range is the masked min/max...` line is already in ours.
2. `mbirtorch/preprocess/segmentation.py`, raise hunk — **take `.our` entirely,
   discard `.their`**. Ours' condition is a strict superset, its messages are
   correct in both branches, and theirs' all-masked message is factually wrong.
3. `tests/test_sharded_segmentation.py` — **take `.our` entirely, discard
   `.their`**. Ours' docstring says the same thing at more length and, unlike
   theirs, is consistent with the two tests ours adds below it. (Theirs' phrasing
   "on this volume the two paths happen to bin identically" is slightly better
   than ours' "Identical is the right bar for THIS volume" as an opening; a
   one-clause borrow is optional and cosmetic.)
4. Keep ours' `multi_threshold_otsu` paragraph at `:164` — theirs has nothing
   there to merge.

Net effect: commit `410ccf8` contributes **one parenthetical sentence** to the
merged tree. Everything else in it is already present in `c4aa556` in a better
form. It is worth recording in the merge commit message that the prerelease author
independently reached the same two fixes — that is corroboration of the
diagnosis, and the second independent derivation of the same failure mode is a
result even when the code is not taken.

## B.4 Do the tests pass on the merged result?

Yes, with the resolution above. `tests/test_sharded_segmentation.py` is
`9 passed` on `scratchpad/merged2`, and the whole suite is `449 passed, 68
skipped`. Their commit adds no tests, so there is nothing of theirs to keep
passing. With their resolution instead, our two tests fail as shown in B.2.

---

## 8. Recommendation

**LAND NOW, as a resolved merge.**

`16ff97c` is mathematically verified against mbirjax from scratch on this machine
(single ops, hessian, FBP, auto geometry and seeded convergence all at or below
1.1e-5), adds seven tests and breaks none, touches nothing in the device-policy,
ledger, kernel or driver surface, merges with zero conflicts, and both the
widening guard's fallback and the memory ledger handle it correctly on first
contact. Its two structurally novel claims — the scatter forward that survives a
90-degree tilt, and the banded slice arithmetic mbirjax does not have — both hold
under direct test.

`410ccf8` cannot be landed as-is: it conflicts, and the conflict must be resolved
in ours' favour on both hunks.

### Fix list

At merge time (blocking the merge commit, not the port):

1. **Resolve `410ccf8` per §B.3** — ours for both files, plus theirs' one
   parenthetical. Do not accept the conflict markers and do not take theirs' raise.
2. `docs/source/usr_multi_gpu.rst:9` and `:26`, `docs/source/overview.rst:39`,
   `docs/source/dev_sharding_overview.rst:14` — four geometry enumerations that
   become wrong the moment the merge lands. One pass covers both new geometries.
3. `mbirtorch/utilities.py:396-399` — add `'MultiAxisParallelBeamModel'` to
   `_resolve_geometry_class`'s tuple BEFORE `'ParallelBeamModel'` (MEDIUM-5). One
   line, and it closes a silent wrong-geometry path opened by this very commit.

Fix-forward, in priority order:

4. Tighten `test_multiaxis_recon_convergence_parity`'s volume gate from `5e-3` to
   `~1e-3` and say why the two tests differ (MEDIUM-4); optionally add the
   `coeff_power=2` hessian golden.
5. `mbirtorch/utilities.py:791-796` — widen `copy_ct_model`'s type gate to
   multiaxis (verified: the gate alone is sufficient, the rest of the function is
   already correct). Translation still needs the `angles` assumption generalized
   separately.
6. `mbirtorch/utilities.py:1013-1021` — add a `'multiaxis'` branch to
   `get_ct_model` (no signature change needed). Translation's branch needs a new
   `translation_vectors` argument.
7. `dev_scripts/refresh_widening_floors.py:178-182` — report classes whose
   `_floor_family` is None, so the now TWO unmeasured geometries are visible; and
   guard `:236-243`'s `else`, which would silently build a `ParallelBeamModel`
   under a `'multiaxis'` or `'translation'` label. One fix, both geometries.
8. One `MBIRTORCH_MEMORY_CALIBRATION=1` reconstruction on CUDA at a multiaxis
   shape before any production run — the ledger's `forward batch` / `back batch`
   terms come entirely from the torch-body slab proxy, measured here at **10x**
   (forward, compiled) and 4.3x (back, compiled) short (MEDIUM-2). This is now a
   number, not an estimate, and it exceeds the 2-5x `projectors.py` assumes.
9. Point `tests/test_device_policy.py:700-737` at the two REAL unlisted
   geometries instead of (or beside) the `_UnlistedGeometry` stand-in.
10. Ensure CI/nightly regenerates `tests/goldens/`, or twelve mbirjax parity gates
    (six translation, six multiaxis) skip silently.
11. Optional: add a sentence to `_widening_floors.py` noting that the parallel and
    cone floors are currently identical, so the `DEFAULT_FAMILY` substitution is
    inert today and would need revisiting if a refresh ever separates them
    (§1(a)/LOW-1); add a multiaxis case to `tests/test_memory_ledger.py`; gate the
    90-degree-elevation forward; note the auto-geometry slice inflation in
    `usr_multiaxis_parallel_beam_model.rst`; extend `hsnt.py:386`'s geometry list.
