# The direct-recon back loop at n>1: the attribution memo

Charter B, step one (multigpu_findings.md §6.3), authored 2026-08-09 by
the charter's implementation agent.  Verified against source line by
line at the increment-1 checkpoint ruling (2026-08-09) and ENDORSED; it
holds in full.  Status: a code-reading attribution plus a local CPU
reproduction of the ledger's published numbers.  The GPU confirmation
is mg6 (job 15034661, mg6_backloop_probe.py), whose readout either
confirms this account or names the cell where it fails; the shallow fix
in §2 is DESCRIBED, not applied, and lands paired with the ledger's
sub-phase split only after mg6 confirms.  The ruling adds one OPTIONAL
RIDER, at the charter's discretion, only after mg6 confirms, and as its
own ledger-visible change: release `block` after `out.add_(block)` in
the worker loop, taking the live-block count from min(3, nb) to
min(2, nb) — worth about one cylinder per device wherever nb ≥ 3.

**Verdict: neither of the findings page's two §2 readings is right.  The
residency is REAL but it is not the three-cylinder charge; the
over-charge is the ledger SUMMING TWO SUB-STEPS THAT NEVER COEXIST.**
The band reduce is real and at n=4 it *is* the peak.  On top of that
there is one genuinely avoidable co-live cylinder the ledger does not
charge at all, and it is a one-line fix of the `weighted_fwd` species.

## 1. The attribution

### The path, and what is live at each step

Chain: `vcd_recon` (`tomography_model.py:2464`) → `direct_recon` →
`fbp_recon`/`fdk_recon` (`parallel_beam.py:335-346`,
`cone_beam.py:767-774`) → `back_project` (`tomography_model.py:1444-1478`)
→ `sparse_back_project` (`336-346`) → `_sparse_back_project_sharded`
(`533-615`).

Units: `cyl = cyl(P, S_dev)` = P·S_dev·4 bytes; `sino` = the per-device
sinogram shard.  B = live blocks in the view loop = `min(3,
view_batches)`.  Per band pass over owner `oi`, with the default band =
whole shard (so n passes, one per owner):

| step | code | live on device i | live on the owner | ledger term |
|---|---|---|---|---|
| region residents | `fbp_recon` holds `sinogram` + `filtered_sinogram`; `vcd_recon` placed `weights` | 3·sino | 3·sino | `sinogram`+`weights`+`filtered sinogram` — **REAL, exact** |
| per-device index copy | `_banded_setup` `tomography_model.py:420-421` | P·8 B | P·8 B | charged on **lead device only** (`_memory_ledger.py:361`) — small **under-charge at n>1** |
| **(a) workers** | `partials = _sharding.run_per_device(...)` `tomography_model.py:580 / 593` → `sparse_back_project_view_range` `projectors.py:397-442` | `out` (=first block, `:439`) + previous `block` + incoming block = **B·cyl** (`:433` evaluates `back_body` before rebinding `block`) + the batch transient + `recon_tensors[i]` if i<oi (1 cyl) + **the previous pass's `partials[i]` (1 cyl)** | same | `back output` charges a flat **2·cyl** at n>1 (`_memory_ledger.py:303`), `back batch` charges the batch |
| **(b) reduce** | `sum_band_to_owner` `tomography_model.py:605-607` → `_sharding.py:237-249` | its own partial only (1 cyl) | `contribs` (n, `:245`; `move_shard` to self is a no-op, `:216-218`) + old + new `total` (`:246-248`) = **(n+1) at n=2, (n+2) at n≥3** — and **no blocks, no batch: the worker locals died on return** | `band reduce` (`:268-284`) — **REAL, exact** |
| concat | `:608-609` | none (one band ⇒ `owner_parts[0]` taken directly, no copy) | — | not charged; correct at the default band |
| scatter | `back_project` `:1463-1472` | `c` (1 cyl, still held by `cylinders`) + `r` (recon-shaped shard) | same | separate `direct recon (scatter)` phase — **REAL, correctly separated** |

The kernel body returns a **fresh** (P, cols) tensor per call
(`triton_parallel.py:261`, `out = torch.empty(...)`), so each `block` is
a full cylinder-shard.

### Per-term verdict

| term | ledger charge | actually live | verdict |
|---|---|---|---|
| `back output` = 2·cyl at n>1 (`_memory_ledger.py:286-303`) | 2·cyl flat | **B·cyl**, B = min(3, view batches): 3 at parallel-1024 n=2, **2** at n=4, **1** at parallel-512 n=4, 2 at cone-512 n=2 | **PHANTOM as charged** — not the n=1 three-cylinder charge carried forward (that would be 3), but a flat 2 that is wrong in both directions.  The docstring's stated reason ("accumulates its per-band parts in a list and concatenates them") does not describe the default single-band path. |
| `band reduce` = (n+1 \| n+2)·cyl (`:268-284`) | 3·cyl (n=2), 6·cyl (n=4) | identical, **on the owner during its own pass only** | **REAL, and exact.**  At parallel 1024 n=4 it IS the phase peak. |
| `back output` + `band reduce` **summed in one phase** (`:390-397`) | added | **(a) and (b) are consecutive, never co-live** | **PHANTOM — the dominant error.**  Same species as correction five, one level down. |
| `back batch` (`:258-261`) | vb·bytes/view, co-charged with the reduce | co-live with (a) only | **REAL in kind, PHANTOM in placement**; magnitude likely conservative (§5) — **UNDECIDED, needs the probe** |
| previous pass's `partials` (1 cyl on **every** device) | **not charged** | **live** — `partials = run_per_device(...)` evaluates before rebinding, so pass k−1's list survives all of pass k's kernels | **REAL, avoidable, uncharged** ← the fix |
| `recon_tensors[i]`, device i's finished own band | **not charged** | live from its own pass onward | **REAL, unavoidable, uncharged** |

### Charged vs live, at the mg2 cells

The ledger was reproduced locally on CPU (patching only the CUDA
transient budget and the Triton cost model, in a scratch script); it
matches the published numbers exactly: parallel 1024 n=1 → 1.009, cone
512 n=1 → 1.104, parallel 1024 n=4 → 1.419 (mg2 reports 1.42–1.43).
Whole-run peak, GB, against mg1 §1.2:

| cell | n | cyl_dev | view batches | shipped ledger | **live (this attribution)** | measured | shipped/meas | live/meas |
|---|---|---|---|---|---|---|---|---|
| parallel 1024 | 2 | 1.448 | 4 (B=3) | 14.803 | 14.803 / 13.355 | 14.04 | 1.054 | 1.054 |
| parallel 1024 | 4 | 0.724 | **2** (B=2) | 10.375 | **7.479 / 7.336** | 7.31 | **1.419** | **1.023** |
| parallel 512 | 4 | 0.048 | **1** (B=1) | 0.943 | 0.771 | 0.65 | 1.451 | 1.186 |
| parallel 512 | 2 | 0.096 | 2 | 1.306 | 1.250 | 1.11 | 1.177 | 1.126 |
| cone 512 | 2 | 0.096 | 2 | 1.786 | 1.730 | 1.70 | 1.051 | 1.018 |
| cone 512 | 4 | 0.048 | 2 | 1.357 | 1.183 | 1.07 | 1.269 | 1.105 |
| cone 1024 | 4 | 0.724 | 5 (B=3) | 10.771 | 8.599 | 9.08 | 1.186 | **0.947 ← under** |

At parallel 1024 n=4 the ledger charges 2 + 6 + 2.20 = 10.2 cyl above
the residents where max(worker 6.20, reduce 6.00) = 6.20 cyl is live:
**the entire 1.42 over-charge is the phantom sum.**  Correcting it pulls
the n=4 cells from 1.19–1.45 to 1.02–1.19 and moves nothing at n=1.

## 2. The shallow fix (described, NOT applied)

Diff, `mbirtorch/tomography_model.py`, inside
`_sparse_back_project_sharded`, immediately after line 607:

```python
                     owner_parts.append(_sharding.sum_band_to_owner(
                         [p for p in partials if p is not None], odev,
                         self.dev2dev_safe))
+                    # The partials are consumed by the reduce.  Release the
+                    # name here: the next band's `partials = run_per_device(...)`
+                    # evaluates its call BEFORE rebinding, so without this the
+                    # previous band's partial stays live on every device for the
+                    # whole of the next band's projection -- one cylinder per
+                    # device, uncharged.  (The `weighted_fwd` treatment.)
+                    partials = None
```

Per the checkpoint-2 precedent this lands **paired with the ledger
change**, in one commit, because the current ledger cannot see the term:
split `direct recon (back loop)` (and identically `hessian diagonal` and
`subset back projection`, which share `back_fixed`/`band_reduce`) into
two sub-phases — `[workers]` charging `min(3, view_batches)·cyl` + the
stale partial + the finished own band + `back batch`, and `[reduce]`
charging only the n+1/n+2 copies — then the fix drops the stale-partial
term, exactly as fix 1 added a sub-phase and dropped a term in the same
change.

Ledger-predicted per-device effect (corrected-ledger basis, GB):

| cell | n | corrected | +fix | move | binding phase after |
|---|---|---|---|---|---|
| parallel 1024 | 2 | 14.803 | **13.355** | **−9.8%** | back loop [workers] |
| cone 1024 | 2 | 15.080 | **13.638** | **−9.6%** | subset delta forward projection |
| cone 1024 | 4 | 8.599 | **7.875** | **−8.4%** | back loop [workers] |
| parallel 1024 | 4 | 7.479 | 7.336 | −1.9% | **back loop [reduce]** |
| parallel/cone 512 | 2, 4 | — | **no move** | 0% | already `initial forward projection` |

This is the `hess_weights` shape of ruling: it pays at the 1024 cells
and buys nothing at the 512 cells, where the peak has already passed to
another phase.

**Second finding for the record:** at parallel 1024 n=4 the binding
sub-peak after the fix is the **band reduce**, whose gather-then-sum
(`_sharding.py:245-248`) holds n+2 copies where a streaming reduce would
hold 3 — worth 2.17 GB/device there.  That is §6.5's declined seam
restructure and §6.4's 2K reduce leg, arriving as the wall one scale
earlier than expected.

## 3. The confirming probe

`mg6_backloop_probe.py`, submitted as job 15034661 (chained after mg5).
It brackets the region, the filter, and — the new resolution — each band
pass's workers and its reduce separately, per device, and reports three
decisive numbers in cyl units: `live_blocks_cyl` (3, or min(3, nb)?),
`entry_step_cyl` (+2 on device 0 / +1 on the last device iff the stale
partials are live), and `worker_peak` vs `reduce_peak` beside the
ledger's charge.  The review widened its arms with `cone_1024_n4`, the
one cell where this account lands UNDER the measured peak — the one
direction the ledger may not err in — so the corrected charge cannot
ship on the code reading alone.

## 4. Scoping note: the structural remedy (pixel batching)

mbirjax's `_sparse_back_project` (`mbirjax/projectors.py:901-946`) nests
the two axes: the OUTER loop batches views and **sums**
(`sum_function_in_batches`, `:944`), and INSIDE each view batch it
batches pixels and **concatenates** (`concatenate_function_in_batches`,
`:940-942`), with the per-view kernel vmapped over the view batch and
reduced (`:930-938`) inside the pixel chunk.  The transient is therefore
bounded at (view_batch × pixel_batch × cols) rather than (view_batch ×
P × cols); the (P, cols) result is still assembled whole, so pixel
batching bounds the **transient**, not the accumulator.  Static shapes
come from a ragged FIRST batch (`:694`) so `lax.map` sees exactly two
shapes.  Default `back_pixel_batch = 2048` (`mbirjax/tomography_model.py:574`)
— at P = 771,240 that is 377 chunks.

In mbirtorch the slot is `Projectors.sparse_back_project_view_range`,
`projectors.py:430-441`: the loop is already shaped as the outer half of
a two-axis tile walk (`projectors.py:212-221` says so explicitly).  The
pixel loop goes **inside** the view loop around the `back_body` call,
writing into row slices of an `out` allocated once up front instead of
lazily from the first block — which as a side effect **removes the
three-cylinder residency entirely**, leaving one accumulator plus one
chunk-sized block.  Costs are known in kind: sinogram traffic amplifies
by roughly the chunk count (each pixel chunk re-reads the same view
batch's `sino_t` slab, materialized per call at `triton_parallel.py:256`),
and the chunk shape must be static with a padded tail or the per-chunk
relaunch/recompile eats the win.

**The one number a traffic probe must measure before this could become
default:** the back projection's wall-time multiplier `t(k pixel chunks)
/ t(1 pixel chunk)` at fixed view batch, at parallel 1024 with the
production P, for the k that bounds the transient to target.  Everything
else in the decision is already known in kind; only whether the k-fold
sinogram re-read is absorbed by cache or paid in full is not.

## 5. Flagged in review

1. **cone 1024 n=4 is the one cell the attribution under-explains**
   (0.947 against measured).  Covered: the review widened mg6's arms
   with that cell.
2. **The `back batch` charge looks ~0.8 GB conservative at parallel
   1024 n=2.**  `_parallel_hfan_math`'s intermediates die on return, so
   at the launch instant only `n_p` + `centers` + `sino_t` are live, not
   the 4 slabs the cost model counts (`triton_parallel.py:295-313`,
   itself flagged as "a counted estimate").  The probe measures this
   directly as `worker transient − blocks`.  This is the residual behind
   the model's 5.4% over-read at n=2.
3. **The same evaluate-before-rebind pattern exists in the forward
   driver** (`tomography_model.py:502`, the two-fan branch's
   `partials =`).  Out of charter B's scope; recorded because it is the
   same species and charter A is reading that path now.
