# Device policy at the entry points — plan

## Executive summary 

Every entry point function (e.g., `recon` or `get_sino_and_model`) follows one of two rules (policies).  
 - Preprocessing functions work on view
batches, so mapping to multiple devices is simple.  Each such function uses 
all visible CUDA devices by default.  This default can 
be overridden in two ways: by setting the env variable `MBIRTORCH_NUM_DEVICES` in advance or 
by passing an explicit `devices=` argument.  
 - Reconstruction-related functions use a policy that depends on geometry and shape.  This policy is determined 
in `_apply_device_policy`.  The device choice is 
settled once per model and re-decided only when
the model's shapes change.  `ct_model.configure_devices()` can be used
to override the policy to set a layout explicitly.

This approach is designed to 
keep the code simple and minimize data movement.  

Two consequences for users:  
- The public API barely
changes.  Three preprocessing functions gain the `devices=` argument that
`scan_to_sino` already has, and two others change only their default. 
- The path from preprocessing to reconstruction is via host memory.
Preprocessing writes its result to host memory, and the reconstruction
moves that result to the devices it chose.  This is exactly what the code
does today.  At production sizes the transfer takes seconds, while a
reconstruction takes minutes to hours.

The work has nine increments, each reviewed before the next starts.
 - Increment 1 implements the once-per-model rule in
   `_apply_device_policy`.  No entry point function changes in this increment.
 - Increments 2 through 5 add the call to `_apply_device_policy` to the functions that lack
   it: direct reconstructions, `recon_plastic_metal`,
   `generate_demo_data`, and four helpers that allocate full-size arrays.
   One helper, `gen_weights`, first needs a per-shard form, because its
   arithmetic cannot accept a sharded sinogram today.
 - Increment 6 develops a policy for `QGGMRFDenoiser.denoise` and then implements it.  
 - Increments 7 through 9 update the preprocessing defaults and the
   documentation.


## Scope and status

This plan sets one device policy across the package's entry points.  It
records three things: the principles Greg decided on 2026-08-13, the
rulings those principles give on the survey's five design questions, and
the increments that implement the rulings.  It supersedes the draft
principles at the head of the survey's §4.  The survey
(`active/entry_point_survey.md`) remains the reference for per-entry
detail, and rows cited here (A9, D7, ...) are its rows.  Greg has decided
every ruling in this plan.  §7 and §8 record the two decided last, in the
discussion of 2026-08-13.

The survey's file:line references were recorded at `greg_dev` commit
`a880d9c`.  The tree has changed since then, and the five prerelease
commits the survey cross-referenced are now merged.  This plan cites
current locations where they differ.  A three-seat review (accuracy,
reasoning, style) ran on 2026-08-13, and its corrections are folded in.

## 1. The two categories

Device policy divides into two categories: preprocessing and
reconstruction.  The property that separates them is the per-device
working set.  A preprocessing entry streams view batches.  Its working set
per device is therefore one batch, whose size `batch_size` sets, at every
problem size.  Bounded work needs no memory preflight, so a preprocessing
entry can use every permitted device without consulting a ledger.  A
reconstruction entry instead holds full arrays resident on the devices:
the sinogram and the volume.  The iterative entries also hold the weights
and the Hessian.  Resident full arrays make the layout a capacity and
speed decision.  A reconstruction entry therefore calls the device policy,
which sizes candidate layouts with the memory ledger and orders the
candidate device counts with the widening floors.  The floors were
measured on three-iteration VCD reconstructions, and "the floors" in this
plan always means these.

Category membership follows the working set, not the module.  Most survey
rows fall into the category their letter suggests.  Categories A, B, and C
are reconstruction, and the streaming entries of category D are
preprocessing.  A few rows sort against their letter: §3 names the
model-bound preprocessing rows, and §5 lists every row whose treatment
changes.

The categories are handled separately, and data may move between them as a
result.  §8 records how that transition works.

## 2. The reconstruction category

A reconstruction entry calls the device policy, `_apply_device_policy` at
`tomography_model.py:1229`.  The layout the policy chooses is settled once
per model and kept.  Later calls on the same model inherit the settled
layout instead of re-deciding.  A settled layout survives changes in free
device memory.  It is re-decided only when the model's shapes change,
because a shape change invalidates the ledger inputs the decision came
from.

Keeping the layout reverses the policy's current contract.  The policy
docstring (`tomography_model.py:1254`) states that the choice is
re-evaluated at every automatic call, so that a long-lived model can adapt
to changed conditions.  Under this plan the stable layout is the goal, and
the remedy for changed conditions is `configure_devices` or a new model.
The reversal removes a failure class.  When a re-decision changes the
device count, every `Shards` object a caller still holds becomes unusable.
The survey names three such holders: the prepared sinogram of row A9, the
precomputed Hessian of row A8, and the per-pass `init_recon` of row D7.

Three layout states implement the rule: automatic and unsettled, automatic
and settled, and pinned.  A new model is automatic and unsettled.  The
first policy call settles the model's layout.  `configure_devices` pins
the model instead, exactly as today, and pinning also clears any settled
state, so the pin and the state cannot disagree.  On a settled model the
policy returns without searching, the same way the pinned branch returns
today.  This return also removes a cost the survey notes at row B5.  On a
cone model given no `init_recon`, `vcd_recon` calls `direct_recon`
internally, so the policy runs twice per reconstruction today.  The second
run repeats the free-memory query; after increment 1 it returns at once.
The settled return must leave the calibration mode correct.  Re-arming the
calibration counters on a nested call resets the peak mid-run, so
increment 1 arms them once per outermost call and carries a calibration
test.

Settling records the shapes the decision came from.  At each call the
policy compares the model's current `sinogram_shape` and `recon_shape` to
the recorded pair, and a difference clears the settled state and
re-decides.  The comparison deliberately does not hook
`refresh_device_bindings`.  That method fires on every recompile-flagged
parameter, including detector offsets and voxel pitch, and the denoiser
sets one such parameter (`sigma_noise`) on every call.  Hooking it would
therefore unsettle the layout on changes that leave the ledger inputs
intact, and mid-pipeline re-decisions are the failure class this plan
removes.  The shape comparison also covers a shape change delivered with
`no_compile=True`, which never reaches `refresh_device_bindings`.

`copy_ct_model` keeps its current rule: a pinned parent produces a pinned
copy, and an automatic parent produces an automatic, unsettled copy.  The
halves of `split_sino_recon` (row B7) therefore continue to decide at
their own, smaller sinogram.  The survey endorses that behavior, and
`tests/test_device_policy.py` covers it.

Every reconstruction entry sizes its layout with the full-reconstruction
ledger and the floors.  This includes the direct reconstructions and,
once calibrated, the denoiser (row C1).  Three reasons led to this ruling.
First, the settled layout must serve the model's lifetime.  The largest
workload in that lifetime is a later `recon` on the same model, so each
call sizes the layout for the model rather than for itself.  Second, the
library cannot predict whether a user will only ever run direct
reconstructions, and a per-workload user setting is not wanted.  Third,
the cost of the ruling has already shrunk once, in the direction Greg
expected.  At the survey's commit the cone floors admitted no two-device
layout.  The floor refresh of 2026-08-11 (commit `4a222c7`) admits two
cone devices from 88,080,384 sinogram elements and four from
1,023,934,464.  The remaining cost is therefore narrower: below those
floors a direct reconstruction is held at a crossover measured on VCD
iterations rather than one measured on a single filtered backprojection.
`dev_scripts/refresh_widening_floors.py` re-measures the floors when the
kernels change, so this cost keeps shrinking without a policy change.

One consequence of settling deserves its own statement.  The capacity
preflight runs once per model, at whichever call settles the layout.  A
later `recon` on the settled model repeats no fit check, so memory taken
by a neighbor process after settling surfaces as the allocator's own
error, not as the preflight's message.  That trade is deliberate: the
preflight's margin absorbs small movements, and a caller who needs a fresh
decision builds a new model.

One driver receives a documented exception to the full-size sizing.  For
cone models `recon_plastic_metal` reconstructs through `split_sino_recon`,
whose purpose is to run problems the full-size reconstruction cannot fit.
Sizing the driver's entry at the full sinogram would refuse exactly those
runs, with an error message recommending the function the driver was about
to use.  When the driver selects the split branch, its entry call
therefore settles the layout with the capacity preflight skipped.  The
per-half policy calls (row B7) then carry the capacity check, and the
halves are the arrays actually reconstructed.  The driver's initial
full-size `direct_recon` returns on the settled layout, so no full-size
check refuses it either.

These entries call the policy: `recon` (A1), `prox_map` (A2), `vcd_recon`
(A3), `fbp_recon` (B1), `fdk_recon` (B4), the halves of `split_sino_recon`
(B7), `denoise` (C1), `recon_plastic_metal` (D7), and `generate_demo_data`
(E1).  `vcd_recon` and `fdk_recon` make the call today.  The calls at B1,
C1, D7, and E1 are new.  Two direct reconstructions outside the survey's
table are new as well: `MultiAxisParallelModel.fbp_recon` and
`TranslationModel.fdk_recon`.  Both classes subclass `TomographyModel`
directly, so changes to `parallel_beam.py` and `cone_beam.py` do not reach
them.  Neither names a floor family, so the parallel floors govern them.
The `direct_recon` delegates (B2, B5) reach the call through their
targets, so a delegated call runs the policy once, not twice.  The base
`TomographyModel.direct_recon` docstring (A4) records this contract for
the remaining and future geometries.

Four further entries settle an unsettled model before allocating.  These
are the full-array allocators: `compute_hessian_diagonal` (A8),
`prepare_sino_for_devices` (A9), `gen_weights` when given a model (F1),
and `gen_weights_mar` (F2).  A8, A9, and F1 each place a full
sinogram-sized array on the devices, and A8 also back-projects it into a
full volume.  F2 touches the devices only on its `init_recon` branch,
where it forward-projects a full metal mask, so it settles only on that
branch.  Today each of these runs on the lead device of a model that has
not spread across devices.  For A9 and F1 the survey reports the result as
a doubled peak memory use on the lead device at production scale.
Settling first places the arrays on the final layout.  Settling first also
resolves row A9's contested status.  Preparing and then reconstructing
becomes safe, because the reconstruction inherits the layout the
preparation settled.  One port is a prerequisite: `gen_weights` cannot
operate on a sharded sinogram today, so increment 5 gives it a per-shard
form before adding the settle.

Two consequences of the allocator settle are documented rather than
avoided.  First, these helpers become able to raise the preflight's error
and to rebuild the model's device state.  Their docstrings say so after
increment 5.  Second, a caller who settles through a helper and then calls
`configure_devices` invalidates the helper's returned `Shards`.  That
invalidation is inherent to an explicit layout change, and the
`configure_devices` docstring records it.

Every other row keeps the survey's column-six ruling.  The projectors and
their sub-steps inherit the model's current layout.  The policy
docstring's rationale still holds.  A developer calling a projector
directly has not asked for a layout change.  Argument-following entries
follow their argument.  Host entries stay host.  The autograd wrappers
(F6) keep their explicit single-device check.  That check remains the
pattern for refusing a multi-device model with a clear message.

## 3. The preprocessing category

A preprocessing entry runs on all permitted devices, or on the devices the
caller names.  Permitted means the visible CUDA devices, capped by the
device-count pin `MBIRTORCH_NUM_DEVICES` when it is set.  The pin is
process-wide state the caller established, so a default must respect it.
`scan_to_sino` already implements exactly this rule
(`preprocess/utilities.py:435`, added after the survey).  No ledger and no
floors apply, because a streaming entry's device memory is bounded by its
batch size at any problem size.

Model-bound preprocessing follows the model instead of this rule.
`correct_sino_plastic_metal` (D5) shards through the model's layout.
`segment_plastic_metal` (D6) follows its argument's layout.
`estimate_sino_view_offset` and `align_sino_views` (D9) project through
the model, so they run on the model's settled layout.  Their `direct_recon`
argument is supplied by the caller, and the inheritance holds when that
argument came from the same model.  `recon_plastic_metal` (D7) is not
preprocessing.  It is a reconstruction driver, and §2 gives it the policy
call.

§7 records which functions make up the public preprocessing surface and
why.

## 4. Rulings on the survey's design questions

The survey's §4 poses five design questions labeled D1 through D5.  Those
labels collide with the survey table's rows D1 through D11, so this
section restates each question in words and retires the question labels.
Row labels keep their meaning.

**Model-free entries.**  The survey asks whether an entry that runs before
any model exists consults a policy at all, and whether it must honor the
device-count pin.  Ruling: such an entry follows the preprocessing rule
and honors the pin.  This is the survey's option (b), and `scan_to_sino`
already implements it.

**Re-choosing.**  The survey asks whether an entry that inherits a layout
from a previous reconstruction on the same model may re-choose.  Ruling:
an automatic layout is settled once per model and kept, and it is
re-decided only when the model's shapes change.  The policy never
re-decides because the free device memory changed.  §2 gives the mechanism
and the failure class this removes.

**The denoiser.**  The survey asks whether `QGGMRFDenoiser.denoise` should
call the policy now that its loop runs sharded.  Ruling: yes.  Row C1
joins the reconstruction category.  It cannot make the call as sized
today, for three reasons.  The denoiser sets `sinogram_shape` equal to its
image shape, so the floors would read image voxels as sinogram elements.
The denoiser names no floor family, so the parallel floors would govern
it.  And the ledger would charge projector terms the denoiser never
allocates.  The denoiser therefore gets its own ledger and its own floors
before the call is added.  That calibration is increment 6.
`median_filter3d` (C2) stays exempt.  It is single-device by construction
and bounds itself through `max_block_gb`.

**Floors and ledger beyond VCD.**  The survey asks whether the
VCD-measured floors and the full-reconstruction ledger apply to entries
that are not VCD reconstructions.  Ruling: they apply unchanged to every
reconstruction entry.  The sizing paragraph of §2 gives the three reasons.
This also answers the question commit `72208bb` left open in the
`fdk_recon` change.  Two documented exceptions attach to the ruling, both
recorded where they arise: the split branch of `recon_plastic_metal` (§2)
and the internal model of `generate_demo_data` (below).

**The two explicit device choices outside the policy.**  The survey asks
what happens to `generate_demo_data`'s `devices=` argument (E1) and the
vcls single-view sibling's pin (G1).  Ruling: E1 moves into the
reconstruction category.  It calls the policy on the model it constructs,
and its `devices=` argument stays an explicit pin routed through
`configure_devices`.  Its internal model settles with the capacity
preflight skipped, because that model exists only for one projection and
has no reconstruction lifetime to size for.  The floors still choose the
count, and a genuine overflow surfaces as the allocator's error.  G1 is
deferred with the rest of category G.  The provisional answer is to keep
its pin and record the reason in its docstring, following the F6 pattern.

## 5. Entries whose treatment changes

The table lists every row whose behavior this plan changes, with its
increment:

| Row | Entry | Change | Increment |
|---|---|---|---|
| A2 | `prox_map` | first call settles the layout; later passes inherit it | 1 |
| A8 | `compute_hessian_diagonal` | settles an unsettled model before allocating | 5 |
| A9 | `prepare_sino_for_devices` | settles an unsettled model before placing the sinogram | 5 |
| B1 | `fbp_recon` | policy call at entry, mirroring `fdk_recon`'s | 2 |
| (none) | `MultiAxisParallelModel.fbp_recon` | policy call at entry; no survey row, found in review | 2 |
| (none) | `TranslationModel.fdk_recon` | policy call at entry; no survey row, found in review | 2 |
| C1 | `denoise` | its own ledger and floors, then the policy call at entry | 6 |
| D3 | unfused siblings | gain `devices=` with the permitted-devices default (§7) | 7 |
| D4 | `correct_zinger_pixels`, `BH_correction` | existing `devices=` default becomes the permitted devices (§7) | 7 |
| D7 | `recon_plastic_metal` | policy call at entry; the split branch skips the capacity preflight (§2) | 3 |
| E1 | `generate_demo_data` | policy call on its internal model, preflight skipped (§4); `devices=` stays an explicit pin | 4 |
| F1 | `gen_weights` (with a model) | per-shard form, then settles an unsettled model before sharding | 5 |
| F2 | `gen_weights_mar` | settles an unsettled model on its projecting branch | 5 |

Every row not listed keeps the survey's column-six ruling, and category G
stays deferred for separate evaluation.

## 6. Already in the tree

Four of the survey's findings are already resolved on `greg_dev`, and the
increments must not redo them:

* `scan_to_sino` honors the pin (`preprocess/utilities.py:435`).  This
  fixes the survey's most serious finding: a preprocessing default that
  ignored `MBIRTORCH_NUM_DEVICES`.
* `fdk_recon` makes the policy call (`cone_beam.py:780`, commit
  `72208bb`).  That commit's message asks whether the VCD floors are the
  right rule for FDK and whether the other geometries' direct
  reconstructions get the same call.  §4 answers yes to both.  The same
  commit closes the survey's row-D5 ordering hazard for cone models,
  because `recon_plastic_metal`'s initial `direct_recon` now settles the
  layout before the first beam-hardening pass.
* The cone floors now admit a two-device layout (commit `4a222c7`,
  2026-08-11).  The survey's statement that the cone family never admits
  two devices is out of date, and §2's cost discussion uses the current
  floors.
* The policy docstring no longer cites the denoiser's missing
  multi-device loop.  The rewrite the survey required is in place
  (`tomography_model.py:1241`).

## 7. The preprocessing entry surface (decided)

The question was which preprocessing functions form the public
device-parallel surface.  Today `scan_to_sino` accepts `devices=` and runs
its fused pipeline across devices.  Its three unfused siblings accept only
`batch_size` and run on one device: `compute_sino_transmission`,
`correct_det_rotation`, and `downsample_view_data`.  The asymmetry is
inherited from mbirjax.  The mbirjax siblings have the same single-device
signatures (`mbirjax/preprocess/utilities.py:30, :251, :389`), and
mbirjax's `scan_to_sino` likewise defaults to all visible devices.

Greg decided on 2026-08-13 that the function surface stays.  The three
siblings gain the same `devices=` parameter and permitted-devices default
as `scan_to_sino`.  The same default reaches the two pass-through
functions of row D4, `correct_zinger_pixels` and `BH_correction`, whose
`devices=` parameter already exists and today defaults to one device.
This is increment 7.

Three facts led to the decision:

* The siblings are genuinely public, so narrowing the surface was
  rejected.  All three are documented API
  (`docs/source/usr_preprocess.rst:73, :78, :80`), and the preprocess
  goldens test them.  `correct_det_rotation` also has real unfused
  callers.  `pymbir.get_sino_and_model` applies it, through a private
  helper, to a sinogram that arrives already computed
  (`preprocess/pymbir.py:84`).  The NSI reader's tilt-angle docstring
  directs users to call it themselves (`preprocess/nsi.py:617`).
* The change is small because the driver makes no device choices of its
  own beyond a single-device default.  `map_view_batches`
  (`preprocess/pipeline.py:52`) runs on whatever device list it receives
  and consults no ledger, floors, or pin; `devices=None` means one
  default device.  The increment is a signature and documentation change.
* The parity argument favors the parameter over a class.  A preprocessing
  class would diverge from mbirjax's released surface and break public
  API.  `devices=` on the siblings is a small divergence mbirjax can
  adopt back.  A class stays possible later, as a joint mbirjax and
  mbirtorch API design with its own justification.  That design is out of
  scope here.

One caveat is recorded with the decision.  The siblings are single-stage
operations dominated by host transfer, so consistency of the rule, not
measured speed, justifies the parameter.  No measurement time is spent on
it.

## 8. The transition between the categories (decided)

The question was how data crosses from preprocessing to reconstruction
when the two categories chose different device sets.

Greg decided on 2026-08-13 to gather to host and re-shard.  Model-free
preprocessing consumes and produces host arrays, and the reconstruction
entry shards the host array under its own settled layout.  This ratifies
the boundary the code already has, so increment 8 is documentation only.

The current tree already works this way on the model-free path.
`scan_to_sino` writes its output into one pre-allocated host array, and
its workers each fill a disjoint view slice (`preprocess/pipeline.py`).
The decision adds no new transfer.  Model-bound preprocessing never
crosses this boundary, because it already follows the model's layout.

Three numbers support the decision.  A 2048-cubed float32 sinogram is
32 GiB.  At host-to-device rates of tens of GB/s, moving it takes
seconds, compared with reconstruction times of minutes to hours at that
scale.  During preprocessing the host holds the scan and the sinogram
together, roughly 64 GiB at that size, and gautschi provides about
126 GiB of host memory per requested GPU.

Two interactions were checked during the discussion and need no work:

* The caching-allocator interaction is already handled.  Preprocessing on
  all devices leaves torch's caching allocator holding reserved memory.
  The ledger's budget counts driver-free memory plus reclaimable cached
  segments (`_memory_ledger.device_budget_bytes`), so a later free-memory
  query on the same devices is not distorted.
* A device-resident handoff was considered and rejected.  Settling the
  model first and writing preprocessing output directly into recon shards
  would save one host round trip.  That round trip costs seconds at the
  2048 scale.  The handoff would also couple the categories and invert
  the data flow, so it stays rejected until a measurement shows the
  saving matters.

## 9. Increments

The work proceeds in gated increments, each reviewed before the next
starts:

1. **The settled state.**  Add the third layout state.  The policy
   returns without searching on a settled model.  Settling records the
   model's `sinogram_shape` and `recon_shape`.  At each call the policy
   compares the current shapes to the recorded pair and re-decides on a
   difference; §2 records why the comparison must not hook
   `refresh_device_bindings`.  `configure_devices` clears the settled
   state when it pins.  Calibration is armed once per outermost call, so
   a nested call on a settled model does not reset the peak counters.
   Rewrite the policy docstring's re-evaluation paragraph, and correct
   its claim that `vcd_recon` is the sole call site
   (`tomography_model.py:1246`), which predates the `fdk_recon` call.
   `copy_ct_model` semantics are unchanged.  Setting
   `skip_memory_preflight` after a model has settled does nothing until
   the model re-decides, and the flag's docstring says so.  Six tests
   cover this increment: a repeated automatic call does not re-decide
   when free memory changes; a shape change re-decides; a non-shape
   recompile parameter such as `det_channel_offset` does not re-decide; a
   pin is unaffected; calibration measures a whole run correctly across
   nested calls; the existing copy and split-half tests still pass.
2. **The three remaining direct reconstructions call the policy**:
   `ParallelBeamModel.fbp_recon` (B1), `MultiAxisParallelModel.fbp_recon`,
   and `TranslationModel.fdk_recon`.  Each call is placed inside the
   reconstruction method, so the `direct_recon` delegate runs the policy
   once.  This mirrors the `fdk_recon` change, with matching tests.  The
   multiaxis and translation models name no floor family, so the parallel
   floors govern them, and the tests record that fact.
3. **`recon_plastic_metal` calls the policy at entry** (D7), with the
   split-branch exception §2 records: when the driver selects
   `split_sino_recon`, the entry settles with the capacity preflight
   skipped, and the per-half policy calls carry the capacity check.  The
   ordering failure at row D5 is already closed for cone models by
   `72208bb`.  The entry call makes the settle explicit at the start of
   the driver and covers every geometry.  Two tests cover this increment:
   the layout is settled before the first beam-hardening pass and stable
   across passes, and a problem that fits only the split path is not
   refused at entry.
4. **`generate_demo_data` calls the policy on its internal model** (E1),
   with the capacity preflight skipped per §4's documented exception.
   Its `devices=` argument keeps pinning through `configure_devices`.
5. **The full-array allocators settle first** (A8, A9, F1, F2).  This
   increment has four parts: port `gen_weights` to a per-shard form,
   because its arithmetic cannot accept a sharded sinogram today; add the
   settle to the four entries, with F2 settling only on its projecting
   branch; teach `split_sino_recon` to gather a device-form sinogram or
   weights at entry, so prepared arrays remain usable there; update the
   helper docstrings to name the new behavior, namely that the model's
   layout may change and the preflight's error can be raised.  Tests
   assert on returned values, not only on the layout, because the silent
   failure mode is `gen_weights('unweighted')` returning a
   zero-dimensional array.  Multi-device tests take the suite's
   `unpinned` fixture.
6. **Denoiser calibration, then the call** (C1).  This increment has
   three steps: build the denoiser's own ledger; measure its widening
   floors with an extension of `dev_scripts/refresh_widening_floors.py`;
   add the policy call to `denoise`.  The floor measurement is the one
   cluster measurement in the plan and is reviewed on its own.
   Increment 1's shape comparison is a prerequisite, because `denoise`
   sets a recompile-flagged parameter (`sigma_noise`) on every call, and
   a refresh-hook design would unsettle the layout every pass.  A script
   that holds a denoiser and a reconstruction model settles each
   independently.  The plan assumes their device residencies are
   sequential; a script where both must stay resident should pin both
   explicitly.
7. **The preprocessing surface** (§7).  Give `compute_sino_transmission`,
   `correct_det_rotation`, and `downsample_view_data` the `devices=`
   parameter and permitted-devices default of `scan_to_sino`, with the
   docstrings and tests to match.  Change the default of
   `correct_zinger_pixels` and `BH_correction` (row D4) to the same
   permitted devices; their parameter already exists.
8. **The transition** (§8).  Documentation only: record the host boundary
   as the contract between the categories.
9. **Documentation.**  This increment has four parts.  Correct
   `usr_multi_gpu.rst`'s statement that a reconstruction spreads across
   the visible GPUs with no change to the script; today that holds only
   for `recon`, `prox_map`, and `fdk_recon`, and it holds for the rest
   after increments 2 through 4.  Correct the stale single-device
   docstrings in `preprocess/mar.py`.  Correct `generate_demo_data`'s
   claim that `devices=` does not affect the result; summation order
   shifts results at the 1e-6 level, so golden regeneration must run
   under the suite's device-count pin.  Write the base `direct_recon`
   contract (A4).

Increment 1 goes first, because every other reconstruction increment
depends on the settled state.  Increments 2 through 5 are independent of
each other once 1 is in.  Increment 6 follows 1 and its own calibration.
Increments 7 and 8 touch only preprocessing and documentation, so they can
run at any point.  Increments 3 and 5 overlap without conflict: after
increment 5, the first beam-hardening pass would settle through
`prepare_sino_for_devices` even without increment 3's entry call.
Documentation changes accompany the increments they describe, and
increment 9 covers whatever remains.

## 10. Testing

Two local test methods cover nearly all of the verification.  The
automatic path is tested with the faked device counts already used in
`tests/test_device_policy.py`.  That path has three behaviors to check:
settling, the return on a settled model, and the re-decision after a
shape change.  `tests/conftest.py:9-11` starts CUDA before any test fakes
the count, and the same file pins the suite to one device, so multi-device
tests take the suite's `unpinned` fixture.  Sharded values are tested with
virtual CPU devices, `configure_devices(devices=['cpu'] * n)`, which
reproduces the sharded path on the Mac.  The lessons file's tooling
section (§9) records that results agree across device counts to well
within the test tolerances.  Tests run at a device count that does not
divide the sharded axis, per the lessons file's sharded-code section (§3).

Cluster time is needed only for increment 6's floor measurement.  Before
that measurement is submitted, the virtual-CPU equivalent of the new code
path runs locally, per the lessons rule that finding a host-side mistake
through a GPU queue is slow.
