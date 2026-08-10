# Prerelease review: `1a2ced7` "Port TranslationModel from mbirjax (checklist B, translation)"

Reviewed 2026-08-10 against the CURRENT `greg_dev` tree, not the fork point.
Read-only: nothing in the repo working tree was touched. All execution happened
in a merged copy built in the scratchpad
(`scratchpad/merged`, from `git archive HEAD` + the commit's files + a
`git merge-file` three-way merge of `mbirtorch/utilities.py`).

**Verdict: ISSUES — 0 HIGH, 5 MEDIUM, 7 LOW/NOTE. Recommendation: LAND NOW**, with
a merge-time doc fix and a short fix-forward list. Nothing found is a correctness
defect in the projection path, the device policy, or the ledger.

---

## 0. State of the branches (this moved during the review)

| | |
|---|---|
| commit under review | `1a2ced7`, parent `944aec2` |
| merge base with `greg_dev` | `944aec2` — i.e. the port forks from BEFORE `a880d9c` (device policy) and BEFORE `c4aa556` (floors/ledger set) |
| `greg_dev` at review start | `60ae515`, with 16 files STAGED |
| `greg_dev` now | `c4aa556` "Refine the multi-gpu device selection." — another session committed and pushed those staged files mid-review; tree is clean apart from `demo/demo_1_shepp_logan.py` |

So the "staged but uncommitted" set named in the brief is now committed. Every
statement below is against `c4aa556`.

The port's changed set: `mbirtorch/translation_model.py` (new, 460 lines),
`mbirtorch/__init__.py`, `mbirtorch/utilities.py`, `tests/test_translation.py`
(new), `tests/generate_goldens.py`, `tests/test_demo_data.py`, and five docs files
(one a rename out of `_pending/`).

---

## 1. The guard's unlisted-geometry fallback — CLEAN, verified live

This is the first geometry to exercise `_widening_floors.DEFAULT_FAMILY`, and it
works.

**Declarations.** `TranslationModel` declares NO `_floor_family` — correct, it has
no measured floors — and no `rows_track_slices`, correctly inheriting the base
`False` at `mbirtorch/tomography_model.py:1191`, whose own comment already names
"cone, translation, multiaxis" as the False cases. It declares no `_dc_damping`
(cone-only) and no `split_sino_recon`.

**Inheritance to the policy site.** Verified at runtime that
`recon`, `vcd_recon` and `_apply_device_policy` are all the base implementations
(`m.recon.__func__ is TomographyModel.recon` etc.). `vcd_recon` is the sole call
site of `_apply_device_policy`, so translation reaches the one policy site.

**Own device-choosing code: NONE.** The only `self.torch_device` reference in the
whole file is `translation_model.py:409`, placing the FDK cosine pre-weight on the
model's already-chosen device — placement, not selection, and identical to cone's
`fdk_filter`. Every other `device=` is derived from an input tensor's device
inside a body. The uniformity ruling holds: one policy site.

**The family lookup, traced live.** `_speed_ordered_candidates` reads
`self._floor_family` → `None` → `_widening_floors.admitted(None, n, elements)` →
`resolved = DEFAULT_FAMILY = 'parallel'`, with the substitution clause appended to
every reason string, and the verbose-2 log line emitted from
`tomography_model.py:1056-1060`. Measured outputs:

| sinogram | elements | candidate order | held |
|---|---|---|---|
| (16, 40, 32) toy | 20,480 | `[1, 4, 3, 2]` | 4, 3, 2 |
| (256, 700, 500) | 89,600,000 | `[2, 1, 4, 3]` | 4, 3 |
| (256, 1900, 3000) production TCT | 1.459e9 | `[4, 3, 2, 1]` | none |

Sample reason string, verbatim:

> `held by the speed floor: 0.0M sinogram elements < 88.1M (the parallel n=2 floor); configure_devices(num_devices=2) overrides (this model names no _floor_family, so the parallel floors apply)`

and the matching fallback note:

> `chosen past its speed floor because no admitted count fits: 0.0M sinogram elements < 88.1M (the parallel n=2 floor)`

At production TCT scale every count is admitted, so the substitution is permissive
exactly where it needs to be — it does not hold translation at one device on real
problems.

`_widening_floors.stale_note()` returns `None` after the merge: the port touches
none of `COST_INPUT_FILES` (`triton_parallel.py`, `triton_cone.py`,
`projectors.py`) nor `COST_INPUT_METHODS`, so `BLESSED_COST_HASHES` and
`TABLE_CHECKSUM` stay valid and no spurious staleness note appears.

### MEDIUM-1 — the refresh tooling cannot see the first unmeasured geometry

`dev_scripts/refresh_widening_floors.py:178-182`:

```python
family = (getattr(cls, '_floor_family', None)
          if isinstance(cls, type) and issubclass(cls, TomographyModel) else None)
if family is not None and family not in known:
```

A class arriving exactly as `TranslationModel` does — `_floor_family = None` — is
skipped. Verified live on the merged tree: `unmeasured_families()` returns `{}`
with `TranslationModel` imported and reachable. The one tool whose job is to
report "this geometry needs measurement" is silent about the first geometry that
does. Secondary: `_build_model` at `:236-243` has `if family == 'cone': ... else:
ParallelBeamModel(...)`, so if a `'translation'` floor family is ever declared the
`else` would silently measure the wrong geometry under a translation label.

---

## 2. The funnel — CLEAN

`translation_model.py` contains no reference to `projector_functions`, to any
`_sparse_*` driver, or to a body function. Its complete set of engine touches:

| line | call | verdict |
|---|---|---|
| 386 | `self._shard_sinogram(sinogram)` | base chokepoint, above the funnel |
| 412 | `self._apply_direct_recon_filter(...)` | base shared FBP filter |
| 429 | `self.back_project(filtered_sinogram, output_sharded=True)` | routes through `sparse_back_project` at `tomography_model.py:1618` |
| 430 | `self._gather_recon(recon)` | base chokepoint |

Every engine projection therefore routes through the public
`sparse_forward_project` / `sparse_back_project` pair. The bodies are reached only
through `Projectors.sparse_{forward,back}_project_view_range` via
`_view_batch_bodies()`, exactly as cone and parallel. Both body signatures match
the driver's call exactly (forward takes `slice_start`, `plan`, `**args`; back
takes `coeff_power`, `slice_start`, `band_slices`, `plan`, `**args`) — confirmed
by the sharded test passing, which drives the banded path.

---

## 3. The memory ledger — CLEAN (it prices translation), one MEDIUM

**It never returns None, and it never raised.** `_build_memory_ledger` →
`plan_from_model` → `estimate_peak_device_bytes` are total functions; there is no
`return None` path for a live model. Built successfully at 1, 2 and 4 devices on
both a toy and a production shape. So `_apply_device_policy` does NOT reach the
`ledger is None` branch at `tomography_model.py:991` and does NOT silently skip
the capacity preflight for translation. (That branch exists for hand-built plans
and explicit layouts.) **No MEDIUM to record on that front.**

Every plan assumption holds for translation geometry:

| plan field | value | why it is right |
|---|---|---|
| `rows_track_slices` | `False` | two-fan geometry; drives `sino_rows = sinogram_shape[1]`, `back_cols = num_rows_dev`, `forward_block_rows = sino_rows` — all correct, its forward output spans every detector row whatever slice band it was handed |
| `helical` | `False` | `'view_params_array' not in params` (translation stores `translation_vectors`), so no spurious `helical z-weight` recon-shaped term |
| `hessian_masked` | `False` | translation sets `use_ror_mask=False`, so the hessian is priced at the full grid and no `hessian scatter` sub-phase is emitted |
| `num_pixels_full` / `num_pixels_grid` | equal | consistent with no ROR mask |
| `view_charge` | resolves | closure over `_view_batch_bodies()` + `_view_batch_args()` binds cleanly |

Priced at (256, 1900, 3000), recon (118, 360, 240), P = 42,480:

```
n=1  peak 21.89 GiB   dominant "per-iteration statistics"
n=2  peak 13.54 GiB   dominant "subset delta forward projection (granularity 16)"
n=4  peak  8.03 GiB   dominant "subset delta forward projection (granularity 4)"
                      top term: forward batch 1.95 GiB
```

**New per-device allocations the ledger does not know: essentially none.** The
port allocates nothing outside the bodies except `fdk_filter`'s `(rows, channels)`
float32 `weight_t` — 21.7 MB at a 1900x3000 panel, on the model device only,
during the direct-recon phase, which the ledger prices in GB. Cone's `fdk_filter`
carries the identical unmodelled array, so this is not a new class of gap. NOTE
only.

### MEDIUM-2 — the ledger's projection terms rest entirely on the unvalidated torch-body proxy

Neither translation body carries a `_view_batch_cost`, so
`Projectors.view_batch_charge` falls to the gather-slab rule:
`bytes_per_view = num_pixels * _transient_cols(band_cols) * 4`, one slab.
`projectors.py:196-201` already says the proxy under-states reality:

> the kernels hold several slab-scale tensors at once ... so the actual per-view
> transient is a small multiple (~2-5x) of the nominal slab; the 8x sinogram
> multiple was calibrated empirically with that multiplier baked in

That multiplier is baked into `VIEW_BATCH_SINO_MULTIPLE = 8`, which sizes the
**batch choice**. The **ledger** consumes `view_batch * bytes_per_view` directly as
its `forward batch` / `back batch` charge, with no such multiplier. At the
production shape: `view_batch = 6`, `bytes/view = 3.23e8` → 1.80 GiB charged,
and at n=4 `forward batch` (1.95 GiB) is the largest single term of the 8.03 GiB
modeled peak. A 2-5x under-statement of that term puts the true peak 2-8 GiB
above the model — the one direction `_memory_ledger` says it may not err in (its
own comments at `_memory_ledger.py:316-320` and `:745-748`).

Cone and parallel are shielded because their Triton bodies carry a real
`_view_batch_cost`, and `_widening_floors.MEASURED_CONFIG` records the floors and
the calibration were taken with "Triton kernels on". Translation has no kernels,
so the torch-body proxy is its permanent state, not a fallback.

Counting the live tensors in `_translation_back_view_batch` at the tap loop's
peak — `det_col` (Vb,P,R), `v_slices`/`cos_phi`/`m_p` (Vb,P,S), `m_center` (Vb,P,S)
int64, plus per-iteration `mm` int64, `L`, `A`, the clamp temporary and the gather
output — gives roughly 12 slab-equivalents against a 1-slab charge, consistent
with the documented 2-5x once compile fusion is allowed for.

**This is not a port defect** — the bodies are a structural clone of cone's torch
bodies and the same arithmetic applies there. It is an exposure the port makes
permanent. Action: one `MBIRTORCH_MEMORY_CALIBRATION=1` reconstruction on CUDA at
a translation shape, checked against `CALIBRATION_BAND`, before any production TCT
run. Not a blocker for landing.

---

## 4. Kernel and body selection — CLEAN

`_view_batch_bodies` exists (`translation_model.py:267-271`) and returns the two
module-level torch bodies unconditionally, with a plain-English one-line reason.
It imports nothing from `kernel_availability`.

The kernel-availability probes are invoked ONLY from `cone_beam.py:370-391` and
parallel's equivalent — never generically — so the parallel/cone-specific probes
stay entirely out of translation's way. Confirmed by grep across `mbirtorch/`.

The torch-body path engages cleanly: `Projectors.__init__` runs `maybe_compile` on
both bodies per device, and the full merged suite (which exercises the compiled
paths) is green.

`view_batch_charge` prices the transient sanely by the gather-slab rule.
`_transient_cols` (`translation_model.py:289-294`) is a verbatim structural copy of
cone's (`cone_beam.py:415-421`), returning the params-derived
`max(recon_shape[2], sinogram_shape[1])` rather than the runtime band, with the
same "whatever the requested band" rationale — correct, because both bodies do
hold `(Vb, P, R)` and `(Vb, P, S)` transients regardless of the band they are
handed. Verified: `_transient_cols(1)` and `_transient_cols(999)` both return 40 on
the toy model (rows=40, slices=24).

---

## 5. Merge collision — CLEAN textually; two stale-doc gaps

**Textual.** `git merge-tree 944aec2 HEAD 1a2ced7`: **zero conflict markers**.

* One file "changed in both": `mbirtorch/utilities.py`. The port's hunks are at old
  lines 1046 (insert 180 lines of phantom builders) and 1347-1354 (delete the
  `NotImplementedError`); `greg_dev`'s are at 244-286 (`save_data_hdf5` sharded
  streaming) and 1367-1412 (`device_provenance`). Nearest gap is 12 lines, so
  three-line context never overlaps. `git merge-file` produced a clean merge, used
  for all testing below.
* `docs/source/_pending/usr_translation_model.rst` shows as "removed in remote" —
  it is the rename into `docs/source/`, resolved by rename detection.
* Every other file is disjoint. The port touches none of `_memory_ledger.py`,
  `_widening_floors.py`, `projectors.py`, `tomography_model.py`, `vcls.py`,
  `preprocess/*`, `parameter_handler.py`, `cone_beam.py`, `parallel_beam.py`.

**Suite on the merged tree:** `442 passed, 68 skipped` vs `437 passed, 68 skipped`
on `HEAD` alone. +5 tests (the five non-golden translation tests), zero
regressions. `tests/test_device_policy.py`, `tests/test_memory_ledger.py`,
`tests/test_widening_floors.py`, `tests/test_sharded_*` all pass unchanged.

**Semantic.** The port assumes nothing about the moved files — it consumes only
base hooks that all survive (`_view_batch_bodies`, `_view_batch_args`,
`_transient_cols`, `_apply_direct_recon_filter`, `_shard_sinogram`,
`_gather_recon`, `_check_lateral_truncation`, `back_project`). Two doc collisions:

### MEDIUM-3 — geometry enumerations in `greg_dev`'s own device-policy docs go stale

* `docs/source/usr_multi_gpu.rst:9` — "It works for the parallel-beam and cone-beam
  geometries." False after the merge.
* `docs/source/usr_multi_gpu.rst:26` — "per-geometry, because parallel-beam and
  cone-beam reach the crossover at different sizes." Now incomplete: a third
  geometry exists whose floors are the substituted parallel set, which is exactly
  the case the page should explain.
* `docs/source/overview.rst:39` — "MBIRTorch supports the *parallel-beam* and
  *cone-beam* imaging geometries". Pre-existing on both sides.

The port could not have known about the first two (they were written after its
fork point), but the merge must fix them.

---

## 6. Port checks

### 6.1 mbirjax parity — verified independently, PASSES

I regenerated the translation golden section from the mbirjax sibling checkout on
this machine (`/Users/gbuzzard/miniforge3/envs/mbirjax/bin/python`, cwd = the
mbirjax repo) and ran the parity gates against the merged tree:

```
translation sparse_fwd  rel_max = 2.14e-06
translation sparse_back rel_max = 7.20e-07
translation forward     rel_max = 1.61e-06
translation fdk         rel_max = 1.01e-06
translation recon parity: alpha rel = 2.31e-06, fm rel = 5.09e-07, final rel_max = 5.35e-05
```

mbirjax's auto geometry — recon_shape (2, 36, 24), delta_voxel 0.25,
voxel_row_aspect 64.0 — matches the port exactly. The commit message's claims
("sparse fwd/back and FDK at float rounding vs mbirjax; seeded 3-iteration recon
1.8e-5") are independently confirmed.

An independent line-by-line math read of both sources found **no divergence** in:
the in-plane coordinates and `pixel_mag`; the hfan inputs (`n_p`, `W_p_c`,
`weight_scale = delta_voxel_row / cos(theta)`, `(num_channels-1)/2` centering,
round-half-to-even int32 centers); the vertical affine including the signs of
`t_z` (`z_offset = -t_z`) and `det_row_offset`; the `1/cos(phi)` placement
(forward on the values, back inside the coefficient) in both directions; the
trapezoid `clamp((slope+1)/2 - |.|, 0)` then `min(., min(1, W_p_r))` with **no
normalization on either side**; `coeff_power` raised on the FULL coefficient
including `1/cos(phi)` and after masking, matching
`mbirjax/projectors.py:379-382`; `get_psf_radius` (`amax(translation_vectors,
axis=0)[1]`, `max(delta_voxel, delta_voxel_slice)` excluding the row pitch, the
ceil-of-ceil-over-2 form, the `>4` warning); the FDK cosine pre-weight and
`alpha = delta_det_row / (voxel_volume * M_0)`; and the constructor defaults
(`qggmrf_nbr_wts=[0.1,1,1]`, `use_ror_mask=False`, `max_alpha=1.3`,
`fraction_of_fill=0.5`).

mbirjax has **no** separate back-projection psf radius for translation (unlike
cone), so the port's single `psf_radius` and its docstring claim at
`translation_model.py:12-15` are accurate.

Deliberate deltas, all in the safe direction: the port ADDS `translation_vectors`
shape validation the jax side lacks (constructor and `verify_valid_params`;
verified live that it rejects a wrong view count), raises on
`source_to_closest_pixel <= 0` BEFORE the divide where jax tests `< 0` after, and
adds the banded forward (`slice_start`) that jax deliberately does not have. The
port OMITS jax's two memory knobs (`TRANSLATION_SLICE_BAND_SIZE`,
`TRANSLATION_FORWARD_DET_ROW_BATCH`) — which is exactly the "known scale limit"
its own module docstring records at `translation_model.py:17-21`.

### 6.2 Sharded correctness

`test_translation_sharded_recon_matches_single_device` runs 2 CPU shards against 1
device on the same seeded problem end to end and passes, which drives the banded
multi-device drivers. No CUDA shard test exists (no CUDA on this machine); nothing
in `tests/test_kernels_sharded.py` applies, since there are no kernels.

### 6.3 Test coverage delta

Added: `tests/test_translation.py` (11 tests — adjointness and recon smoke on each
backend, 2-shard parity, and six mbirjax golden gates), the
`tests/generate_goldens.py` translation section, and
`test_generate_demo_data_translation` replacing
`test_generate_demo_data_translation_is_not_available`.

Gaps (LOW):
* `tests/test_device_policy.py:700-704` still uses the synthetic
  `_UnlistedGeometry` stand-in whose docstring says "as TranslationModel and any
  future class would arrive". The real class now exists and should be the subject
  (or a case beside it) — that is the standing coverage for the fallback path.
* No translation case in `tests/test_memory_ledger.py`.

### 6.4 Goldens impact

`tests/goldens/` is gitignored (generated on demand), so the commit ships a
generator change, not data. Every parity test is `skipif`-guarded on
`"tct_sino" in golden.files` with a clear regenerate message. The local golden file
predates the port, so on any unrefreshed machine the six strongest gates **skip
silently** in a normal run. Correct design, but CI/nightly must regenerate or the
parity claims are dark. NOTE.

### 6.5 Integrations — the real gaps

Verified by running each against a live `TranslationModel` on the merged tree.

#### MEDIUM-4 — `copy_ct_model` still refuses translation

`mbirtorch/utilities.py:791-796`:

```python
raise TypeError('copy_ct_model() is restricted to ConeBeam and ParallelBeam Models')
```

Observed verbatim. And widening the type gate is not enough: `:803`
`old_angles = required['angles']` would then `KeyError` (translation's required
dict carries `translation_vectors`), and `:824` `new_shape[0] = len(new_angles)`
mis-sizes. Public API, reached indirectly by `vcls` and by
`split_sino_recon`-style flows.

#### MEDIUM-5 — `get_ct_model('translation', ...)` still refuses translation

`mbirtorch/utilities.py:1013-1021`:

```
ValueError: Invalid geometry type.  Expected cone or parallel, got translation
```

Observed verbatim. Its signature has no `translation_vectors` parameter, so adding
the branch is a signature change, not a one-line `elif`.

#### What DOES work (checked, no change needed)

* Construction, `auto_set_recon_geometry` (called from `TomographyModel.__init__`
  at `:186`), `verify_valid_params`, `recon`, `fdk_recon`, `direct_recon`.
* `get_all_params` → `_resolve_geometry_class` → `build_model` round-trips and
  reproduces `recon_shape` exactly. `utilities.py:399` already enumerated
  `'TranslationModel'` in anticipation; the port's `__init__.py` change is what
  unblocks it.
* `get_recon_dict` / `save_recon_hdf5` round trip.
* `mbirtorch/preprocess/zeiss_tct.py:120` (`str(mt.TranslationModel)`) is DEAD in
  the current tree (raises `AttributeError`), and this commit FIXES it — a real
  net win the commit message does not mention.
* `preprocess/mar.py:920` correctly falls through (`'cone' in 'translation'` is
  False), so translation takes the plain `recon` path.
* `_memory_remedies()` returns `[]` (no `split_sino_recon`), correct.
* `gen_translation_phantom` both options, including the Pillow/font-fallback path.

#### LOW / inherited

* `generate_demo_data(model_type='translation')` works but produces a degenerate
  demo: recon_shape (1, 132, 132) and an **all-zero** phantom (verified,
  `nonzero == 0`). It also ignores the `voxel_row_aspect` / `voxel_slice_aspect`
  arguments, and returns `params = {'translation_vectors': ...}` only — omitting
  `source_detector_dist` / `source_iso_dist`, so unlike the cone/parallel branches
  its `params` cannot rebuild the model. All three behaviours are
  character-identical to `mbirjax/utilities.py:1406-1414`. Inherited, and the
  port's own test says so honestly.
* `gen_dot_phantom` calls `np.random.seed(42)`, silently reseeding the caller's
  global numpy RNG. Verbatim from `mbirjax/utilities.py:953`.
* `vcls.get_opt_views` treats `view_params[:, 0]` as an angle (`vcls.py:151`); on a
  translation model that is the x translation. It does not crash, and it was
  already loose for cone's two-column array, but its docstring enumerates only
  Parallel/Cone.
* `hsnt.py:386` `VALIDATION_RULES` allows only `(None, 'parallel', 'cone')` for
  `dataset_geometry`; a translation HSNT dataset would warn. Only matters if
  translation HSNT data is in scope.
* `TomographyModel._apply_direct_recon_filter` folds `filter_scale * pi/num_views`
  into the filter and its docstring states this assumes views EQUALLY SPACED over
  `[0, pi)`. Translation's views are translations, not angles, so that stated
  assumption is silently untrue for the new caller. mbirjax does exactly the same
  and the goldens match to 1e-6, so it is a faithful scale factor on an
  explicitly approximate initializer — a docstring gap, not a numerical one.

---

## 7. The comment guideline — essentially CLEAN

No `checklist`, `campaign`, `item N`, `mg3`/`mg4`, `floor_4`, plan notation,
`TODO`, `FIXME` or `XXX` anywhere in `translation_model.py` or
`tests/test_translation.py`. "checklist B" appears only in the commit message,
which is where the guideline allows it.

The one coined-looking term, **"the hfan contract"** (`translation_model.py:63`),
is established house vocabulary — `triton_cone.py:16,441,610`,
`triton_parallel.py:25,224`, `tests/test_triton_parallel.py:7` — and the line names
its defining file inline. Not a violation.

Three borderline lines worth a clause each:

* `mbirtorch/translation_model.py:19-21` — "A planned engine change may restore
  pixel batching; no workaround is built here." References an unnamed plan. The
  equivalent note in `projectors.py:206-224` names what it means (pixel-axis
  chunking / the fused Triton kernels). One clause would close it.
* `mbirtorch/translation_model.py:113-115` and `:179-180` — "``plan`` is the
  memoization slot for a future sorted/CSR stream variant; unused today." Verbatim
  from `projectors.py:372-373` and cone's bodies, so consistent house style, but it
  is an undefined term now repeated in a third file.
* `docs/source/usr_translation_model.rst:11-13` — the new text editorializes about
  the upstream project ("The mbirjax page's statement that no direct
  reconstruction exists is outdated there as well."). Accurate, but a user doc is
  an odd place to correct another project's page.

---

## 8. Recommendation

**LAND NOW.** The port is mathematically verified against mbirjax from scratch on
this machine, merges with zero conflicts, adds five tests and breaks none, touches
nothing in the device-policy or ledger surface, and both the widening guard's
fallback and the memory ledger handle it correctly on first contact.

### Fix list

At merge time (blocking the merge commit, not the port):

1. `docs/source/usr_multi_gpu.rst:9` and `:26`, `docs/source/overview.rst:39` —
   the geometry enumerations become wrong the moment the merge lands.

Fix-forward, in priority order:

2. `mbirtorch/utilities.py:791-796` (+`:803`, `:824`) — widen `copy_ct_model` to
   translation (needs the `angles` assumption generalized), or at minimum make the
   refusal name translation rather than implying only two geometries exist.
3. `mbirtorch/utilities.py:1013-1021` — add the `'translation'` branch to
   `get_ct_model`, with a `translation_vectors` argument.
4. `dev_scripts/refresh_widening_floors.py:178-182` — report classes whose
   `_floor_family` is None, so the first unmeasured geometry is visible; and guard
   `:236-243`'s `else`, which would silently build a `ParallelBeamModel` under a
   `'translation'` label.
5. One `MBIRTORCH_MEMORY_CALIBRATION=1` reconstruction on CUDA at a translation
   shape, checked against `CALIBRATION_BAND`, before any production TCT run — the
   ledger's `forward batch` / `back batch` terms for translation come entirely
   from the torch-body slab proxy, which has never been calibrated as the sole
   charge for a geometry.
6. Point `tests/test_device_policy.py:700-737` at the real `TranslationModel`
   instead of (or beside) the `_UnlistedGeometry` stand-in.
7. Ensure CI/nightly regenerates `tests/goldens/`, or the six mbirjax parity gates
   skip silently.
8. Optional: name the "planned engine change" at `translation_model.py:19-21`;
   widen `vcls.get_opt_views`'s docstring or guard it; extend `hsnt.py:386`'s
   geometry list if translation HSNT data is in scope.
