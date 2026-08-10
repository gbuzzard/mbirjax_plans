# Prerelease review: four convergent fixes (`b96c7e7`, `4b10686`, `81351ef`, `dd3d8d8`)

Method follows `scratchpad/prerelease_review_multiaxis.md` §PART B: side-by-side,
`git merge-tree`, cross-wise test runs, exact resolution, named cherries.

## 0. State of the branches

```
merge-base(greg_dev, prerelease) = 944aec2  "Port the sharded case of the beam-hardening correction"
ours   = c4aa556  (HEAD tree; one unrelated demo edit on top)
theirs = origin/prerelease = dd3d8d8

commits on prerelease not in HEAD, oldest first:
  1a2ced7  Port TranslationModel                       (precedent review)
  16ff97c  Port MultiAxisParallelModel                 (precedent review)
  410ccf8  Soften the sharded-histogram claim          (precedent review, PART B)
  b96c7e7  Stream sharded volumes to HDF5              <- pair 1
  81351ef  Restore torch.mean for the single-device Sp floor   <- pair 3
  4b10686  Keep the sharded denoiser's line search on device   <- pair 2
  dd3d8d8  Name the unsupported geometries in the errors       <- pair 4
```

The real merge (`git merge origin/prerelease` from `c4aa556`, run in
`scratchpad/cv_clone2`) produces **5 conflicted files, 7 conflict hunks**:

| file | hunks | owner |
|---|---|---|
| `mbirtorch/denoising.py` | 2 | `4b10686` |
| `mbirtorch/preprocess/mar.py` | 1 | `81351ef` |
| `mbirtorch/preprocess/segmentation.py` | 2 | `410ccf8` (precedent) |
| `mbirtorch/utilities.py` | 1 | `b96c7e7` |
| `tests/test_sharded_segmentation.py` | 1 | `410ccf8` (precedent) |

`dd3d8d8` contributes **no conflict at all**.

Working trees built for this review (all under `scratchpad/`):

| tree | contents |
|---|---|
| `cv_head` | `c4aa556` verbatim (baseline) |
| `cv_theirsHDF5` | `c4aa556` + `b96c7e7` resolved fully toward theirs, with a shim giving theirs' arithmetic ours' `_sharded_slab_source` signature so ours' every-boundary test runs against it |
| `cv_theirsDEN` | `c4aa556` with `denoising.py` = `4b10686`'s |
| `cv_theirsMAR` | `c4aa556` with `mar.py` = `81351ef`'s |
| `cv_baseDEN`, `cv_baseMAR` | `c4aa556` with those files reverted to `944aec2` (controls) |
| `cv_premergeMAR` | `c4aa556` with `mar.py` = `a5b04ce`'s — the true PRE-MERGE arithmetic |
| `cv_merged` | the recommended resolution of these four pairs only |
| `cv_fullmerge` | the whole `origin/prerelease` merge with every recommended resolution |

---

# PAIR 1 — `b96c7e7` "Stream sharded volumes to HDF5 one slab at a time"

## 1.1 The key question, answered: YES, theirs fixes `export_recon_hdf5`

Ours deliberately did not, and the gap is real. Measured on `cv_head` vs
`cv_theirsHDF5` with the two host-allocation sites instrumented
(`scratchpad/cv_probe_export_mem.py`), 48x32x21 float32 volume, 2 shards:

```
                                       largest single host array   whole-volume gather?
ours    slice-sharded, plain                129024 B (1.00 x vol)   YES
ours    slice-sharded, remove_flash         129024 B (1.00 x vol)   YES
ours    view-sharded,  plain                129024 B (1.00 x vol)   YES
theirs  slice-sharded, plain                  6144 B (0.05 x vol)   no  (21 slabs)
theirs  slice-sharded, remove_flash           6144 B (0.05 x vol)   no  (21 slabs)
theirs  view-sharded,  plain                129024 B (1.00 x vol)   YES (falls back)
```

`export_recon_hdf5` in ours still opens with `recon = _to_host(recon)`
(`mbirtorch/utilities.py:370`), so a sharded recon lands on the host whole
before a single slab is written. At the production sizes this library targets
(2048³ float32 = 34.4 GB) that is a 34 GB host allocation on the export path,
in a design whose entire point is that no full volume is ever held. Theirs
closes it for the slice-sharded case — which is the case that arises, since
`recon_placement` is always the slice axis.

Theirs' `save_data_hdf5` change, by contrast, is **the same fix ours already
has**: measured peak host bytes per slab are identical (0.125 x volume at an
8-row slab on both), and the files are byte-identical.

## 1.2 Side by side

| | ours (`c4aa556`) | theirs (`b96c7e7`) |
|---|---|---|
| helper | one `_sharded_slab_source(shards) -> (out_shape, dtype, produce_slab)` (`utilities.py:289`) | two: `_shard_axis_block(shards, i0, i1)` and `_sharded_host_shape_dtype(shards)` |
| `save_data_hdf5` sharded | delegates to `_sharded_slab_source` | inlines the two-branch slab construction in the function body |
| `export_recon_hdf5` sharded | **not fixed** — `_to_host` gathers whole | streams when `placement.axis % 3 == 2`, gathers otherwise |
| `export_recon_hdf5` structure | two code paths (`remove_flash` or not) | one path with `get_block` + an `if remove_flash` inside the slab producer — strictly simpler, and it drops the `save_data_hdf5(transpose(...))` indirection |
| slab size | hard-coded `(1 << 30)` inline | `_HDF5_SLAB_BYTES = 1 << 30` module constant, so a test can shrink it |
| shard extents | read off the TENSORS (`sizes = [int(t.shape[axis]) ...]`, `starts = np.cumsum(...)`) | read off the PLACEMENT (`pl.shard_ranges(pl.padded_size)`, `pl.real_size or pl.padded_size`) |
| host copies | `.detach().cpu().numpy()` | `.cpu().numpy()` (no `detach`) |
| dtype | `shards.tensors[0][:0].detach().cpu().numpy().dtype` — no torch import needed | `torch.empty((), dtype=shards.dtype).numpy().dtype`, with a module-level-absent `import torch` inside the function (and a second, entirely unused `import torch` in `_shard_axis_block`) |
| tests | 8 tests incl. `test_sharded_slab_source_matches_a_full_gather_at_every_boundary`, parametrized over both sharded axes, comparing EVERY (i0,i1) slab against a full gather | 1 test: an end-to-end `save_data_hdf5` + `export_recon_hdf5` round trip with `_HDF5_SLAB_BYTES` monkeypatched to 256 so several slabs are really written |
| docstring | a 5-line paragraph in the public `save_data_hdf5` docstring stating the byte-identity contract | none |

## 1.3 Correctness at shard boundaries — cross-wise runs

**Ours' tests on THEIRS' implementation** (`cv_theirsHDF5`, ours'
every-boundary test driven through a verbatim repackaging of theirs'
`save_data_hdf5` body):

```
tests/test_sharded_segmentation.py + tests/test_hdf5_family.py -> 10 passed
```

Theirs' boundary arithmetic is correct on every slab of every width, on both
sharded axes, padded and unpadded. And the reverse: files produced by the two
implementations for the same inputs are **byte-identical** across six cases
(`save` on axes 0/1/-1, `export` plain / `remove_flash` / view-sharded).

The one place they diverge is a placement with `real_size=None`
(`scratchpad/cv_probe_hdf5.py`, section 2):

```
                                                        ours      theirs
real_size=None, axis=-1, n=1  (the trivial placement)    PASS   FAIL  shape (8,4,None)
real_size=None, axis=0,  n=1                             PASS   FAIL  shape (None,4,6)
real_size=None, axis=-1, n=2 divisible                   PASS   FAIL  shape (8,4,None)
real_size=None, axis=0,  n=2 divisible                   PASS   FAIL  shape (None,4,6)
```

`Placement(devices, axis)` without `real_size` sets `padded_size = None`
(`_sharding.py:70-71`), so theirs' `pl.real_size if ... else pl.padded_size`
yields a `None`-length axis and `h5py.create_dataset` gets a `None` dimension;
`pl.shard_ranges(pl.padded_size)` is likewise called with `None`. Ours derives
the extents from the tensors and is immune.

**Severity, honestly:** LATENT, not live. The only `real_size=None` placements
the library builds are the trivial single-device ones at
`tomography_model.py:221` and `:232`, and `_shard_recon` / `_shard_sinogram`
return a plain tensor rather than a `Shards` on a trivial placement, so no
`real_size=None` `Shards` reaches these functions through the public API today.
It is reachable by hand (a test, or a user building a `Placement` directly).
Ours' rule costs nothing and removes the trap, so it should be the rule that
survives the merge.

**Theirs' test on OURS' implementation:**

```
FAILED test_sharded_save_and_export_stream_by_slab
  AttributeError: module 'mbirtorch.utilities' has no attribute '_HDF5_SLAB_BYTES'
1 failed, 9 passed
```

That is not a nitpick. Ours' every-boundary test exercises
`_sharded_slab_source` *directly*; nothing in ours ever drives
`save_data_hdf5` through more than one slab, because a test-sized array is
always under 1 GiB. Theirs' constant is the only thing in either commit that
makes the multi-slab write path itself testable. Cherry it.

## 1.4 `git merge-tree` — and the silent double-application

```
git merge-tree b96c7e7^ c4aa556 b96c7e7   ->  1 conflict marker, in mbirtorch/utilities.py
git cherry-pick b96c7e7 onto c4aa556      ->  1 unmerged path: mbirtorch/utilities.py
```

`tests/test_sharded_segmentation.py` merges CLEAN: both sides append at the
end of the file and git takes both, so the merged test file has ours' 8 tests
plus theirs' 1.

**The one marked hunk is the `save_data_hdf5` Shards branch. Everything else in
`b96c7e7` applies silently**, and this is the failure mode to flag:

- `_shard_axis_block` + `_sharded_host_shape_dtype` are ADDED clean at `:217`
  and `:238`, while ours' `_sharded_slab_source` stays at `:328`;
- `_HDF5_SLAB_BYTES` and its use in `_write_hdf5_streaming` land clean;
- **the whole `export_recon_hdf5` rewrite lands clean**, because ours never
  touched that function.

Verified in `scratchpad/cv_clone` mid-cherry-pick: with the single marked hunk
resolved toward ours, the file carries **two independent slab-gather
mechanisms** — `_sharded_slab_source` driving `save_data_hdf5`, and
`_shard_axis_block` / `_sharded_host_shape_dtype` driving `export_recon_hdf5`
— with **different extent arithmetic** (tensor-derived vs placement-derived),
different padding rules, and different detach behaviour. Nothing warns about
it. A reviewer who resolves only the conflict marker ships that.

There is no *runtime* double-application (theirs' export no longer routes
through `save_data_hdf5`, so no slab is written twice), but the divergent
duplicate mechanism is the thing that will drift, and it is invisible in the
conflict.

## 1.5 Verdict — **MERGE**

Ours' slab arithmetic + theirs' export fix + theirs' constant, with theirs'
two helpers repaired to ours' extent rule so there is one convention rather
than two.

Exact resolution:

1. `mbirtorch/utilities.py`, the conflicted `save_data_hdf5` Shards branch —
   **take `.our`**:
   ```python
   if isinstance(array, _sharding.Shards):
       out_shape, dtype, produce_slab = _sharded_slab_source(array)
       _write_hdf5_streaming(file_path, array_name, out_shape, dtype, produce_slab, attributes_dict)
       return
   ```
   Discard theirs' inlined body. Keep ours' `_sharded_slab_source` and its
   public-docstring paragraph.
2. **KEEP theirs' clean-applied `export_recon_hdf5` rewrite.** Do not revert
   it, and do NOT use `git checkout --ours mbirtorch/utilities.py`, which
   would silently throw it away along with items 3 and 6.
3. **KEEP theirs' clean-applied `_HDF5_SLAB_BYTES`** and its use in
   `_write_hdf5_streaming`.
4. **Repair `_shard_axis_block`** to read the shard extents off the tensors
   (ours' rule), and to `detach()`; drop its dead `import torch`:
   ```python
       sizes = [int(t.shape[axis]) for t in shards.tensors]
       starts = np.cumsum([0] + sizes)
       real_end = int(pl.real_size) if pl.real_size is not None else int(starts[-1])
       pieces = []
       for t, s0, s1 in zip(shards.tensors, starts[:-1], starts[1:]):
           lo, hi = max(i0, int(s0)), min(i1, min(int(s1), real_end))
           if lo < hi:
               sel = [slice(None)] * ndim
               sel[axis] = slice(lo - int(s0), hi - int(s0))
               pieces.append(t[tuple(sel)].detach().cpu().numpy())
   ```
5. **Repair `_sharded_host_shape_dtype`** the same way, and take the dtype off
   an empty slice so the function needs no torch import:
   ```python
       shape = [int(v) for v in shards.tensors[0].shape]
       total = sum(int(t.shape[axis]) for t in shards.tensors)
       shape[axis] = int(pl.real_size) if pl.real_size is not None else total
       np_dtype = shards.tensors[0][:0].detach().cpu().numpy().dtype
   ```
6. **Add a rank guard to theirs' export gate**, so a flat
   `(num_pixels, num_slices)` container cannot reach the three-way shape
   unpacking (theirs' `% 3` hard-codes rank 3):
   ```python
   if (isinstance(recon, _sharding.Shards) and recon.tensors[0].ndim == 3
           and recon.placement.axis % 3 == 2):
   ```
7. Keep theirs' new test (it merges clean and is the only multi-slab gate).

Fix-forward, not blocking: fold `_shard_axis_block` and `_sharded_slab_source`
onto one shared extent helper; and extend ours' every-boundary test to cover
theirs' block reader as well.

---

# PAIR 2 — `4b10686` "Keep the sharded denoiser's line search on device; reuse one thread pool"

## 2.1 Side by side

The two are the SAME PATCH almost line for line — same `combine_on_lead` with a
character-identical docstring, same five 0-d tensors out of `terms_worker`,
same `torch.clamp(alpha, _F32_EPS, 1.5)`, same `alpha_per_device`, same three
`float()` reads per pass. Both derive from `vcd_recon`'s combine. The material
differences are three:

| | ours | theirs |
|---|---|---|
| pool ownership | `owns_pool = n > 1 and self._per_device_pool is None`; installs it on `self`; passes `executor=self._per_device_pool`; `finally: shutdown(wait=True); self._per_device_pool = None` | `pool = _sharding.device_pool(n) if n > 1 else None`; local only; passes `executor=pool`; `finally: pool.shutdown()` |
| caller's pool | **reused** — a recon that already installed one gets no second pool, matching `_band_pool`'s convention (`tomography_model.py:357-363`) and `vcd_recon`'s (`:2752-2757`, `:2827-2829`) | **ignored** — a second pool of `n` threads is built alongside the caller's live one |
| indentation | normal | `try:` then `  with torch.no_grad():` at a 2-space indent, so the entire 70-line loop body keeps its old column — valid Python, but the block's nesting no longer matches its indentation |
| docstring | 10 lines naming the mechanism, the cost avoided (5 x n_devices D2H per subset), the tie to `vcd_subset_denoiser`'s float32, and what syncs remain | 4 lines |
| module docstring | updated to match | not updated |

## 2.2 Measured, not asserted

`scratchpad/cv_probe_denoise2.py`, 2 CPU shards, 24x24x21, 5 iterations,
16 subsets. `Tensor.item` AND `Tensor.__float__` both counted and attributed to
the frame that caused them; `ThreadPoolExecutor` constructions counted.

| | base `944aec2` | ours `c4aa556` | theirs `4b10686` |
|---|---|---|---|
| host reads inside the sweep | **1758** | **963** | **963** |
| — in `terms_worker` | 1588 | 948 | 948 |
| — in `apply_worker` | 160 | 0 | 0 |
| — in `_denoise_sharded` | 10 | 15 | 15 |
| ThreadPoolExecutors constructed | **161** | **2** | **2** |
| sharded output digest | `2318aa4cab436b41` | `8685d7e89783719d` | `8685d7e89783719d` |
| sharded-vs-single `rel_max` | 1.057e-07 | 1.057e-07 | 1.057e-07 |

The 948 reads in `terms_worker` are the qGGMRF kernel's own, present in all
three — that is the floor, not the line search. Base's excess is exactly the
line search: 4 sums x 2 devices x 16 subsets x 5 passes = 640, plus
1 x 2 x 16 x 5 = 160 in `apply_worker`. Both patches remove all 800 and add
5 (three `float()` per pass instead of two).

**Ours and theirs are bitwise identical in output and identical on every
counted axis.** The only measured difference is caller-pool reuse:

```
with a caller pool installed (a recon driving the denoiser):
  ours    extra pools created = 0   self._per_device_pool is the caller pool = True
  theirs  extra pools created = 1   self._per_device_pool is the caller pool = True
```

Exception safety is fine on both — an injected `RuntimeError` mid-fan-out
leaves 0 pools running in either, and ours additionally restores
`self._per_device_pool = None`. Neither leaks.

Ours' tests pass unchanged on theirs (`tests/test_denoiser.py`: 3 passed on
both trees) — the suite does not discriminate here, which is expected given the
bitwise identity.

## 2.3 `git merge-tree`

```
git merge-tree 4b10686^ c4aa556 4b10686  ->  2 conflict markers, mbirtorch/denoising.py
```
Both hunks are wholly-overlapping rewrites (the docstring, and the loop). No
silent application anywhere: theirs touches only this one function.

## 2.4 Verdict — **KEEP OURS**

Same arithmetic, same host-read count, byte-identical output; ours additionally
reuses a caller-installed pool (theirs builds a redundant one), follows the
`_per_device_pool` convention the rest of the codebase already uses, keeps the
indentation honest, and documents the mechanism. Theirs is simpler only in the
sense that it does less: its simplicity IS the missing reuse. Nothing in it is
better where equally correct.

Take `.our` for both hunks. `git checkout --ours mbirtorch/denoising.py` is
safe here — theirs changes nothing else in the file.

**Independent-derivation note worth recording in the merge message:** the
prerelease author reached the same fix, with the same `combine_on_lead`
helper and the same clamp, from the same `vcd_recon` source. That is
corroboration of the diagnosis even though none of the code is taken.

**A cherry against OURS, from reading theirs:** ours' comment at
`denoising.py:442` says *"The one host synchronization per pass"*. Measured:
three host READS per pass (`float(ell1_accum)`, `float(image_l1)`,
`float(alpha_accum)`), at one synchronization point. Theirs' *"The only host
reads: once per iteration"* is closer. One word: make ours say "reads".

---

# PAIR 3 — `81351ef` "Restore torch.mean for the single-device Sp floor"

## 3.1 The three expressions lined up

True pre-merge, `a5b04ce:mbirtorch/preprocess/mar.py`:

```python
if view_mask is None:
    mean_plastic_coef = torch.mean(Sp)                                  # f32 0-d tensor
else:
    mean_plastic_coef = torch.sum(Sp * view_mask) / float(num_real_pixels)   # f32 0-d tensor
Sp_floor = gamma * mean_plastic_coef                                    # f32 0-d tensor
...
clamped_plastic_coef = torch.maximum(Sp, Sp_floor)
```

Base `944aec2` (the A7 commit) replaced BOTH branches with
`_ps_sum(...) / _ps_numel(...)`, which returns a **Python float** on a single
device (`_ps_sum` ends `return float(fn(*xs))`, `mar.py:34`), and changed the
application to `torch.clamp(sp, min=Sp_floor)` so the float scalar would be
accepted.

Ours `c4aa556` (`mar.py:713-731`) branches on the FORM of `Sp` and restores the
pre-merge expression **verbatim on both sub-branches**, then restores
`torch.maximum` on both branches by materialising the sharded Python-float
floor into a 0-d tensor of the piece's dtype and device; it also changes the
negativity test to `float(mean_plastic_coef) <= 0` so it reads the same on
either form.

Theirs `81351ef` branches on `isinstance(Sp, torch.Tensor)` **inside the
`view_mask is None` arm only**. The masked arm keeps base's
`_ps_sum(...) / float(num_real_pixels)`; the clamp stays
`torch.clamp(sp, min=Sp_floor)`; the negativity test stays
`if mean_plastic_coef <= 0`.

| | pre-merge | base `944aec2` | ours | theirs |
|---|---|---|---|---|
| unmasked, plain tensor | `torch.mean` (f32) | sum/numel (f64) | `torch.mean` (f32) | `torch.mean` (f32) |
| **masked, plain tensor** | f32 tensor divide | f64 host divide | **f32 tensor divide** | **f64 host divide** |
| sharded, either | n/a | `_ps_sum` | `_ps_sum` | `_ps_sum` |
| application | `torch.maximum` | `torch.clamp(min=float)` | `torch.maximum` both branches | `torch.clamp` |
| branch predicate | — | — | `not isinstance(Sp, Shards)` | `isinstance(Sp, torch.Tensor)` |

## 3.2 Bitwise check — 200 seeds, each tree's own function

`scratchpad/cv_seedhunt.py` / `cv_seedhunt_unmasked.py`, sha256 of the returned
sinogram, `gamma = 1.5` so the floor actually binds:

```
UNMASKED single-device path (the reachable one):
  ours vs pre-merge     0/200 seeds differ
  theirs vs ours        0/200 seeds differ
  base  vs ours        77/200 seeds differ  (38.5%)   <- the A7 defect, reproduced

MASKED single-device path:
  ours vs pre-merge     0/200 seeds differ
  theirs vs ours       46/200 seeds differ  (23.0%)   <- theirs == base here
  base  vs ours        46/200 seeds differ
```

So **ours is bitwise the pre-merge arithmetic on both sub-branches; theirs
restores one of the two and leaves the other on the very f64 divide its own
commit message names as the defect.** Its message — *"A plain tensor now takes
torch.mean again"* — is true, and incomplete: a plain tensor with a view mask
still takes the host-combined f64 form.

Isolating the mechanism (`scratchpad/cv_sweep_mar2.py`, 2000 trials,
N = 20000, gamma = 0.9): the two floor values differ as float32 in **44.4%**
(unmasked) and **43.5%** (masked) of draws, at a maximum relative difference of
1.3e-07 — one ULP. Where the floor binds, every clamped element carries that
1-ULP difference. `torch.maximum(x, 0-d tensor)` and `torch.clamp(x, min=float)`
are identical when the floor VALUE is identical (checked directly, including
NaN), so the divergence is entirely in how the mean is computed and rounded,
not in how it is applied.

**Reachability, honestly:** `view_mask` is non-`None` only when
`pl.real_mask(3)` is non-`None` (`mar.py:808`), i.e. only for a padded
multi-device placement, which implies `measured_sino` is a `Shards`. So the
masked-plain-tensor case is **not reachable through `bh_correction` today**. It
is reachable by a direct call to `_correct_plastic_sinogram`, and it is what
the branch is FOR. Neither side's tests cover it — `tests/test_preprocess_mar.py`
+ `tests/test_sharded_segmentation.py` give `9 passed` on both `cv_head` and
`cv_theirsMAR`.

## 3.3 Edge cases — does either raise or mis-branch where the other does not?

Checked directly (`scratchpad/cv_mar_edge.py`):

| input | ours | theirs |
|---|---|---|
| plain `torch.Tensor` (f32) | single-device branch | single-device branch |
| plain `torch.Tensor` (f64) | OK, same branch | OK, same branch |
| `Shards` | sharded branch; all trees bitwise equal (`c4ca51ba0b8487a8`) | sharded branch, same digest |
| `numpy.ndarray` | `TypeError` from `zeros_like` — raised much earlier in `_get_column_H`, before either predicate is reached | identical `TypeError` |

`Sp` is always exactly one of `{Tensor, Shards}` at that point (it comes from
`_ps_map`), so `not isinstance(Sp, Shards)` and `isinstance(Sp, Tensor)` are
complements on every reachable input. **Neither mis-branches.** Ours' predicate
is the safer one to keep only because it names the thing the branch is about
(the sharded form), and because it keeps a hypothetical third container on the
combine path rather than on `torch.mean`.

## 3.4 `git merge-tree`

```
git merge-tree 81351ef^ c4aa556 81351ef  ->  1 conflict marker, mbirtorch/preprocess/mar.py
```
Nothing else in the commit; no silent application.

## 3.5 Verdict — **KEEP OURS**

Ours is a strict superset: it restores both sub-branches (measured bitwise
against `a5b04ce`, 0/200), restores `torch.maximum`, and makes the negativity
test form-independent. Theirs restores half and, on the other half, is
bit-for-bit the arithmetic it set out to remove (46/200 seeds). Take `.our`;
`git checkout --ours mbirtorch/preprocess/mar.py` is safe (theirs touches
nothing else in the file).

**Corroboration worth recording:** two independent reviews found the same A7
defect in the same expression. Both diagnoses are right; only ours' fix is
complete.

**Optional cherry (cosmetic):** theirs' 2-line inline comment

> `# Plain tensor: torch.mean, the original single-device arithmetic.`
> `# Sharded: per-piece sums combined on the host, divided by the count.`

is a compact restatement of ours' 8-line comment. Ours' version says WHY the
forms are not interchangeable, which is the load-bearing part; if the long
comment is trimmed, keep the "would move the divide to float64" clause.

**Fix-forward:** add a test for the masked single-device path — it is the only
part of this branch nothing covers, and it is exactly where theirs went wrong.

---

# PAIR 4 — `dd3d8d8` "Name the unsupported geometries in get_ct_model / copy_ct_model errors"

Nothing landed on our side; ours still carries
`'copy_ct_model() is restricted to ConeBeam and ParallelBeam Models'`
(`utilities.py:796`) and mbirjax's verbatim
`'Invalid geometry type.  Expected cone or parallel, got {}'` (`:1021`).

## 4.1 Is the message accurate against the merged reality?

Run on `cv_fullmerge` — the whole prerelease merge, ports included
(`scratchpad/cv_probe_msgs.py`):

```
exports: TranslationModel, MultiAxisParallelModel, MultiAxisParallelBeamModel
get_ct_model('cone')        -> OK ConeBeamModel
get_ct_model('parallel')    -> OK ParallelBeamModel
get_ct_model('translation') -> ValueError: get_ct_model() supports geometry_type 'cone' and
                               'parallel' only; got 'translation'.  For the translation and
                               multiaxis geometries (not yet supported here, matching mbirjax),
                               construct TranslationModel or MultiAxisParallelModel directly.
get_ct_model('multiaxis')   -> same
copy_ct_model(ParallelBeamModel)      -> OK
copy_ct_model(TranslationModel)       -> TypeError: ... got TranslationModel. ...
copy_ct_model(MultiAxisParallelModel) -> TypeError: ... got MultiAxisParallelModel. ...
```

Accurate on every clause:

- both helpers really do still refuse both geometries after the ports land;
- `TranslationModel` and `MultiAxisParallelModel` are the real exported class
  names and both are directly constructible — the message's advice works;
- *"matching mbirjax"* checks out: mbirjax ships `translation_model.py` and
  `multiaxis_parallel.py` yet its `get_ct_model` still ends
  `raise ValueError('Invalid geometry type.  Expected cone or parallel, ...')`
  (`mbirjax/utilities.py:1399`) and its `copy_ct_model` still tests only
  `ConeBeamModel` / `ParallelBeamModel` (`:1427`). The restriction is inherited,
  not invented here.

Two small imprecisions: for a plain typo (`get_ct_model('bogus')`) the message
still lectures about translation and multiaxis; and it drops mbirjax's verbatim
string, so the two libraries' messages now differ in wording (not in behaviour).
Neither is worth blocking on.

## 4.2 Does it collide with the widening to come?

**Textually, no** — `git merge-tree dd3d8d8^ c4aa556 dd3d8d8` gives **0 conflict
markers**, and the full merge leaves the file clean at both sites.

**Semantically, partly.** The precedent review's fix list already schedules the
widening, and it is asymmetric:

- `copy_ct_model` — verified there that widening the type gate ALONE is enough
  for multiaxis. When that lands, `MultiAxisParallelModel` must come out of
  this message; `TranslationModel` stays (translation still needs the `angles`
  assumption generalised).
- `get_ct_model` — a plain `elif` suffices for multiaxis; translation needs the
  per-view-params API discussion this commit's own message points at.

So the message goes half-stale at the next step, in a way that is obvious and
one-line to fix, at two known sites. That is the normal cost of a stopgap, not
a collision.

## 4.3 Verdict — **KEEP THEIRS**

It merges clean, every claim in it is verified true against the merged tree, it
names classes that exist and are constructible, and it replaces two messages
that tell a user nothing about what to do next. The interim is not short — the
widening waits on an API discussion — so carrying it is worth it.

**Adopt regardless (a cherry that pairs with it), with a correction to the
precedent:** `16ff97c` ALREADY adds `'MultiAxisParallelModel'` to
`_resolve_geometry_class`'s tuple, ahead of `'ParallelBeamModel'`. What it does
not add is the **alias** name, and `'MultiAxisParallelModel'` is not a substring
of `'MultiAxisParallelBeamModel'` while `'ParallelBeamModel'` is. Checked on
`cv_fullmerge`:

```
"<class 'mbirtorch.multiaxis_parallel.MultiAxisParallelModel'>"      -> MultiAxisParallelModel
"<class 'mbirtorch.multiaxis_parallel.MultiAxisParallelBeamModel'>"  -> ParallelBeamModel   <- wrong
```

So the precedent's MEDIUM-5 stands but is NARROWER than it was framed: the
alias resolves to the same class object, so `type(model).__name__` on any live
instance is always `MultiAxisParallelModel` and the normal
`get_all_params` -> `build_model` round trip is safe. The silent
wrong-resolution needs a `geometry_type` string written as the alias — a
hand-built params dict, or a file recorded by code that used the alias. One
extra tuple entry closes it:
`('ConeBeamModel', 'MultiAxisParallelBeamModel', 'MultiAxisParallelModel', 'ParallelBeamModel', 'TranslationModel')`.

**At the widening, edit both messages in the same commit** — they are the
canonical statement of what the helpers accept, and a stale one is worse than
the vague one it replaced.

---

# 5. The merged tree, and what the suite says

`scratchpad/cv_merged` = `c4aa556` + the recommended resolution of these four
pairs (ours' `save_data_hdf5` slab source, theirs' export streaming and slab
constant with the two helpers repaired, ours' denoiser, ours' mar, theirs'
messages). Full diff at `scratchpad/cv_recommended.diff`
(`utilities.py` +106/-20, `tests/test_sharded_segmentation.py` +37).

```
              FULL SUITE                                    -m goldens
cv_head       437 passed, 68 skipped, 79 deselected   |  79 passed, 505 deselected
cv_merged     438 passed, 68 skipped, 79 deselected   |  79 passed, 506 deselected
cv_fullmerge  450 passed, 68 skipped, 91 deselected   |  79 passed, 12 skipped, 518 deselected
```

`cv_merged` is `+1` test (theirs' multi-slab gate) and zero regressions.
`cv_fullmerge` is the whole `origin/prerelease` merge with every recommended
resolution including the precedent's; its 12 skipped goldens are the
multiaxis/translation goldens that have not been generated yet, exactly as the
precedent recorded.

Behaviour re-verified on the merged tree:

```
every-boundary + real_size=None + end-to-end battery        14 passed, 0 failed
export_recon_hdf5, slice-sharded, largest host array        0.05 x volume, 21 slabs, no whole-volume gather
export_recon_hdf5, view-sharded                             falls back to a gather (documented)
```

---

# 6. Verdicts and cherries

| commit | verdict | one line |
|---|---|---|
| `b96c7e7` | **MERGE** | theirs fixes the `export_recon_hdf5` gap ours left open (1.00 x volume -> 0.05 x, measured) and adds the only testable slab constant; ours' slab arithmetic is the more robust of the two identical-output implementations |
| `4b10686` | **KEEP OURS** | bitwise-identical output and identical host-read count, but theirs builds a second thread pool when the caller already has one |
| `81351ef` | **KEEP OURS** | ours is bitwise pre-merge on both sub-branches (0/200 seeds); theirs restores only the unmasked one and stays on the f64 divide for the masked one (46/200) |
| `dd3d8d8` | **KEEP THEIRS** | every claim verified true against the merged tree, names real constructible classes, merges clean; half-stale at the widening, one line to fix at each of two sites |

## Cherries — take regardless of the verdict above

1. **`_HDF5_SLAB_BYTES`** (from `b96c7e7`) — the module constant plus its use in
   `_write_hdf5_streaming`. Ours has no way to drive `save_data_hdf5` through
   more than one slab in a test; this is it. Comes with theirs' test.
2. **Theirs' `export_recon_hdf5` rewrite** (from `b96c7e7`) — the streaming
   export, and the single-path `get_block` + `if remove_flash` structure, which
   is genuinely simpler than ours' two-path version.
3. **"reads", not "synchronization"** — `denoising.py:442`, ours' own comment.
   Measured: 3 host reads per pass at one sync point.
4. **`_resolve_geometry_class` + the `'MultiAxisParallelBeamModel'` ALIAS** —
   the precedent's MEDIUM-5, narrowed: `16ff97c` already added the real class
   name; the alias still resolves to `ParallelBeamModel` (§4.3).
5. **`tests/test_denoiser.py:73`** — ours' own test docstring still says the
   sums "combine on the host". They combine on the lead device now.
6. **A test for the masked single-device Sp floor** — the one branch neither
   side covers, and exactly where theirs went wrong.

## Merge-time instruction set

Ordered, for `git merge origin/prerelease` on `greg_dev`:

1. `git merge origin/prerelease` -> 5 conflicted files, 7 hunks.
2. `git checkout --ours mbirtorch/denoising.py` — safe, theirs touches nothing
   else in it. (Pair 2: KEEP OURS.)
3. `git checkout --ours mbirtorch/preprocess/mar.py` — safe, same reason.
   (Pair 3: KEEP OURS.)
4. `mbirtorch/preprocess/segmentation.py` — resolve per the precedent review
   §B.3: ours for both hunks, plus theirs' one parenthetical sentence. **Do not**
   `checkout --ours` blindly if you also want that sentence.
5. `mbirtorch/utilities.py` — **hand-resolve; never `checkout --ours`.** Four
   prerelease commits touch this file (`1a2ced7`, `16ff97c`, `b96c7e7`,
   `dd3d8d8`) and only one of them conflicts, so `--ours` here would silently
   drop `gen_translation_phantom` (+180 lines) and the
   `_resolve_geometry_class` widening as well. Take `.our` for the single marked
   hunk (the `save_data_hdf5` Shards branch), then KEEP every clean-applied
   block from theirs: `_HDF5_SLAB_BYTES`, the `export_recon_hdf5` rewrite,
   `dd3d8d8`'s two error messages, the ports' phantom generator and geometry
   tuple. Then apply the three repairs in §1.5 items 4-6
   (tensor-derived extents in `_shard_axis_block` and
   `_sharded_host_shape_dtype`, `detach()`, drop the two dead `import torch`,
   rank guard on the export gate).
6. `tests/test_sharded_segmentation.py` — **hand-resolve; never
   `checkout --ours`.** The marked hunk is `410ccf8`'s module docstring: take
   ours. `b96c7e7`'s `test_sharded_save_and_export_stream_by_slab` was appended
   cleanly below and must survive.
7. Before committing, grep the merged `mbirtorch/utilities.py` for
   `_sharded_slab_source`, `_shard_axis_block`, `_HDF5_SLAB_BYTES` and
   `supports geometry_type` — all four must be present. Their absence means a
   `--ours` checkout ate a clean-applied hunk.
8. Apply the merge-time items from the precedent review's fix list (the four
   stale geometry enumerations in `docs/`, and `_resolve_geometry_class`).
9. `PYTHONPATH=. python -m pytest -q` and `-m goldens`. Expect
   **450 passed, 68 skipped** and **79 passed, 12 skipped** (the 12 are the
   ungenerated multiaxis/translation goldens).
10. In the merge commit message, record that the prerelease author independently
    reached the same diagnosis for the denoiser line search and the A7 Sp-floor
    regression. Two independent derivations of the same failure mode is a
    result, even where the code is not taken.
