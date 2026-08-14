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

The work has eight increments, each reviewed before the next starts.
 - Increment 1 implements the once-per-model rule in
   `_apply_device_policy`.  No entry point function changes in this increment.
 - Increments 2 through 4 add the call to `_apply_device_policy` to the functions that lack
   it: direct reconstructions, `generate_demo_data`, and four helpers
   that allocate full-size arrays.  One helper, `gen_weights`, first
   needs a per-shard form, because its arithmetic cannot accept a
   sharded sinogram today.
 - Increment 5 develops a policy for `QGGMRFDenoiser.denoise` and then implements it.  
 - Increment 6 updates the preprocessing defaults.
 - Increment 7 ports the sharded phantom build from mbirjax, which demo
   data at production scale needs.
 - Increment 8 updates the documentation.


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
preprocessing.  A few rows sort against their letter: §3.2 names the
model-bound preprocessing rows, and §5 lists every row whose treatment
changes.

The categories are handled separately, and data may move between them as a
result.  §8.1 records how that transition works.

## 2. The reconstruction category

### 2.1 The rule

A reconstruction function calls `_apply_device_policy`
(`tomography_model.py:1229`) before it allocates.  The policy chooses a
device layout, and that choice is settled once per model.  Later calls on
the same model reuse the settled layout.  The policy re-decides only when
the model's `sinogram_shape` or `recon_shape` changes.  A change in free
device memory never re-decides.  `configure_devices` overrides the policy
at any time.

Settling once keeps caller-held sharded arrays valid.  A re-decision that
changed the device count would invalidate every `Shards` object a caller
still holds: a prepared sinogram (A9), a precomputed Hessian (A8), or a
beam-hardening pass's `init_recon` (D7).

### 2.2 The three layout states

| State | Set by | The next policy call |
|---|---|---|
| automatic, unsettled | a new model | decides a layout and settles it |
| automatic, settled | the first policy call | returns the settled layout without searching |
| pinned | `configure_devices` | returns the pinned layout without searching |

Increment 1 implements the states inside the policy:

* Settling records the model's `sinogram_shape` and `recon_shape`.
* Each later call compares the current shapes with the recorded pair.
  Equal shapes return at once.  Different shapes clear the settled state
  and re-decide.
* `configure_devices` clears the settled state when it pins, so the state
  and the pin cannot disagree.
* The calibration counters are armed once per outermost policy call.  A
  nested call must not reset the peak mid-run.

The shape comparison belongs in the policy, not in
`refresh_device_bindings`.  That method fires on every recompile-flagged
parameter, including detector offsets, voxel pitch, and the denoiser's
`sigma_noise`.  Hooking it would unsettle the layout on parameter changes
that leave the shapes intact.  The comparison also catches a shape change
made with `no_compile=True`, which never reaches
`refresh_device_bindings`.

`copy_ct_model` is unchanged.  A pinned parent gives a pinned copy, and an
automatic parent gives an automatic, unsettled copy.  The halves of
`split_sino_recon` (B7) therefore decide at their own, smaller size.

### 2.3 Sizing the layout

Every reconstruction function sizes its layout with the
full-reconstruction ledger and the floors, direct reconstructions
included.  The settled layout serves every later call on the model, and
the most demanding of those is a full `recon`.

The capacity preflight runs once per model, at the call that settles the
layout.  A later `recon` repeats no fit check.  Memory taken by another
process after settling surfaces as the allocator's error rather than as
the preflight's message.

Sizing every call for a full `recon` has one known cost.  A model that
will only ever run a direct reconstruction is still sized for a `recon`
it will never run, so a large problem can be refused for a workload it
was not going to attempt.  The sharpest case is a cone problem too large
for a full `recon`: its initial direct reconstruction is refused, and the
error recommends `split_sino_recon`, which is the path that would have
fit.  That case arrived with commit `72208bb` and is not introduced by
this plan.  The remedy in both cases is `configure_devices`, which names
the layout and runs no preflight.  Increment 2 proposes to remove this
cost by checking capacity against the work in progress while still
choosing the count for a full `recon`; §9.2 specifies the change.

### 2.4 Which functions call the policy

| Function | Today | After this plan |
|---|---|---|
| `recon` (A1), `prox_map` (A2) | through `vcd_recon` | unchanged |
| `vcd_recon` (A3) | calls the policy | unchanged |
| `ConeBeamModel.fdk_recon` (B4) | calls the policy | unchanged |
| `ParallelBeamModel.fbp_recon` (B1) | no call | calls the policy (increment 2) |
| `MultiAxisParallelModel.fbp_recon` | no call | calls the policy (increment 2) |
| `TranslationModel.fdk_recon` | no call | calls the policy (increment 2) |
| `split_sino_recon` halves (B7) | through each half's `recon` | unchanged |
| `recon_plastic_metal` (D7) | late, once per pass | through its initial `direct_recon`, once per model (increments 1 and 2) |
| `generate_demo_data` (E1) | no call | calls the policy (increment 4) |
| `QGGMRFDenoiser.denoise` (C1) | no call | calls the policy (increment 6) |

Three notes on that table.  `MultiAxisParallelModel` and
`TranslationModel` subclass `TomographyModel` directly, so the calls added
to `parallel_beam.py` and `cone_beam.py` do not reach them; neither names
a floor family, so the parallel floors govern both.  The `direct_recon`
delegates (B2, B5) call their targets, so a delegated call runs the policy
once.  The base `TomographyModel.direct_recon` docstring records the
contract for future geometries.

### 2.5 The full-array allocators

Four helpers settle an unsettled model before they allocate:

* `compute_hessian_diagonal` (A8) — a full sinogram of ones,
  back-projected into a full volume.
* `prepare_sino_for_devices` (A9) — the whole sinogram.
* `gen_weights` with a model (F1) — a full sinogram-sized weights array.
* `gen_weights_mar` (F2) — a full forward projection, on its `init_recon`
  branch only.  It settles only on that branch.

Today each of these runs on the lead device of a model that has not spread
across devices, which doubles peak lead-device memory at production scale.
Settling first places the arrays on the final layout.  It also makes
preparing and then reconstructing safe, which resolves row A9's contested
status.

Increment 5 carries two prerequisites and two documentation changes:

* `gen_weights` needs a per-shard form.  Its arithmetic cannot accept a
  sharded sinogram today.
* `split_sino_recon` must gather a device-form sinogram or weights at
  entry, because it host-slices its inputs.
* The four helper docstrings state that the call may change the model's
  layout and may raise the preflight's error.
* The `configure_devices` docstring states that pinning after a helper has
  settled invalidates that helper's returned `Shards`.

### 2.6 Everything else

Every survey row not named above keeps its column-six ruling:

* The projectors and their sub-steps inherit the model's current layout.
  A developer calling a projector directly has not asked for a layout
  change.
* Argument-following functions follow their argument's layout.
* Host functions stay on the host.
* The autograd wrappers (F6) keep their single-device check, which
  refuses a multi-device model with a clear message.

## 3. The preprocessing category

### 3.1 The rule

A preprocessing function runs on all permitted devices by default.
Permitted means the visible CUDA devices, capped by
`MBIRTORCH_NUM_DEVICES` when that variable is set.  A `devices=` argument
overrides the default.  No ledger and no floors apply, because a streaming
function's device memory is bounded by its batch size at any problem size.

`scan_to_sino` already implements this rule
(`preprocess/utilities.py:435`).  Increment 7 gives the same rule to the
five remaining preprocessing functions, which §7.1 names.

### 3.2 Model-bound preprocessing

Four functions receive a model or a sharded array, so they follow the
layout that argument carries rather than the rule above:

* `correct_sino_plastic_metal` (D5) shards through the model's layout.
* `segment_plastic_metal` (D6) follows its argument's layout.
* `estimate_sino_view_offset` and `align_sino_views` (D9) project through
  the model.  Their `direct_recon` argument comes from the caller, so they
  run on the model's settled layout when that argument came from the same
  model.

`recon_plastic_metal` (D7) is not preprocessing.  It is a reconstruction
driver, and §2.4 gives it the policy call.

## 4. Rulings on the survey's design questions

The survey's §4 poses five design questions labeled D1 through D5.  Those
labels collide with the survey table's rows D1 through D11.  This section
restates each question in words and retires the question labels.  Row
labels keep their meaning.

| Question | Ruling |
|---|---|
| Model-free functions: do they consult a policy, and must they honor the device-count pin? | They follow the preprocessing rule and honor the pin.  `scan_to_sino` already does. |
| Re-choosing: may a function that inherits a layout decide it again? | No.  The layout is settled once per model and re-decided only on a shape change (§2.1). |
| The denoiser: should `QGGMRFDenoiser.denoise` call the policy? | Yes, once it has a ledger and floors of its own (§4.1). |
| Beyond VCD: do the floors and the full-reconstruction ledger apply to functions that are not VCD reconstructions? | Yes, to every reconstruction function.  §2.3 records the one known cost. |
| Outside the policy: what happens to `generate_demo_data`'s `devices=` (E1) and the vcls sibling's pin (G1)? | E1 joins the reconstruction category and keeps `devices=` as a pin.  G1 is deferred (§4.3). |

### 4.1 The denoiser

The denoiser cannot call the policy as sized today, for three reasons:

* It sets `sinogram_shape` equal to its image shape, so the floors would
  read image voxels as sinogram elements.
* It names no floor family, so the parallel floors would govern it.
* The ledger would charge projector terms the denoiser never allocates.

Increment 6 therefore builds a denoiser ledger and measures denoiser
floors before adding the call.  `median_filter3d` (C2) stays exempt.  It
is single-device by construction and bounds itself through
`max_block_gb`.

### 4.2 Floors and ledger beyond VCD

Two reasons led to this ruling.  The settled layout serves every later
call on the model, and the most demanding of those is a full `recon`.  The
library also cannot predict whether a user will only ever run direct
reconstructions, and a per-workload user setting is not wanted.  This
answers the question commit `72208bb` left open in the `fdk_recon` change.

The cost of the ruling has already begun to shrink.  The cone floors now
admit a two-device layout (§6), so a cone direct reconstruction is held at
one device over a narrower range of sizes than the survey reports.

### 4.3 The two explicit device choices

`generate_demo_data` (E1) joins the reconstruction category.  It calls the
policy on the model it builds, and its `devices=` argument stays an
explicit pin routed through `configure_devices`.  A caller who hits the
preflight cost of §2.3 on a large generation uses that same argument to
name the layout.

The vcls single-view sibling (G1) is deferred with the rest of category G.
The provisional answer is to keep its pin and record the reason in its
docstring, following the F6 pattern.

## 5. Entries whose treatment changes

The table lists every row whose behavior this plan changes, with its
increment:

| Row | Entry | Change | Increment |
|---|---|---|---|
| A2 | `prox_map` | first call settles the layout; later passes inherit it | 1 |
| A8 | `compute_hessian_diagonal` | settles an unsettled model before allocating | 4 |
| A9 | `prepare_sino_for_devices` | settles an unsettled model before placing the sinogram | 4 |
| B1 | `fbp_recon` | policy call at entry, mirroring `fdk_recon`'s | 2 |
| (none) | `MultiAxisParallelModel.fbp_recon` | policy call at entry; no survey row, found in review | 2 |
| (none) | `TranslationModel.fdk_recon` | policy call at entry; no survey row, found in review | 2 |
| C1 | `denoise` | its own ledger and floors, then the policy call at entry | 5 |
| D3 | unfused siblings | gain `devices=` with the permitted-devices default (§7) | 6 |
| D4 | `correct_zinger_pixels`, `BH_correction` | existing `devices=` default becomes the permitted devices (§7) | 6 |
| D7 | `recon_plastic_metal` | the device count stops changing between beam-hardening passes; no edit to the driver | 1 |
| E1 | `generate_demo_data` | policy call on its internal model; `devices=` stays an explicit pin | 3 |
| F1 | `gen_weights` (with a model) | per-shard form, then settles an unsettled model before sharding | 4 |
| F2 | `gen_weights_mar` | settles an unsettled model on its projecting branch | 4 |

Every row not listed keeps the survey's column-six ruling, and category G
stays deferred for separate evaluation.

## 6. Already in the tree

Four of the survey's findings are already resolved on `greg_dev`.  The
increments must not redo them.

| Already done | Where | What it means for this plan |
|---|---|---|
| `scan_to_sino` honors the device-count pin | `preprocess/utilities.py:435` | Closes the survey's most serious finding, a preprocessing default that ignored `MBIRTORCH_NUM_DEVICES`. |
| `fdk_recon` calls the policy | `cone_beam.py:780`, commit `72208bb` | Increment 2 extends the same call to the three remaining direct reconstructions. |
| The cone floors admit a two-device layout | commit `4a222c7`, 2026-08-11 | Two devices from 88,080,384 sinogram elements, four from 1,023,934,464.  §4.2 uses these values. |
| The policy docstring drops its stale denoiser reason | `tomography_model.py:1241` | The rewrite the survey required is done. |

Commit `72208bb` carries one further consequence.  Its `fdk_recon` call
closes the survey's row-D5 ordering hazard for cone models, because
`recon_plastic_metal` runs `direct_recon` before its first beam-hardening
pass.  Increment 3 still adds the entry call, which makes the settle
explicit and covers the other geometries.

## 7. The preprocessing entry surface (decided)

### 7.1 The decision

The public function surface stays as it is, and five functions get the
permitted-devices default (Greg, 2026-08-13).  This is increment 7:

* `compute_sino_transmission`, `correct_det_rotation`, and
  `downsample_view_data` gain the `devices=` parameter that
  `scan_to_sino` already has, with the same default.
* `correct_zinger_pixels` and `BH_correction` (row D4) already have the
  parameter.  Only their default changes, from one device to the
  permitted devices.

### 7.2 Why the surface stays

The question was which preprocessing functions form the public
device-parallel surface.  Today `scan_to_sino` accepts `devices=` and runs
its fused pipeline across devices.  Its three unfused siblings accept only
`batch_size` and run on one device.  The asymmetry is inherited from
mbirjax, whose siblings carry the same single-device signatures
(`mbirjax/preprocess/utilities.py:30, :251, :389`).

Three facts led to the decision:

* The siblings are genuinely public, so narrowing the surface was
  rejected.  All three are documented API
  (`docs/source/usr_preprocess.rst:73, :78, :80`), and the preprocess
  goldens test them.  `correct_det_rotation` also has real unfused
  callers.  `pymbir.get_sino_and_model` applies it through a private
  helper (`preprocess/pymbir.py:84`).  The NSI reader's tilt-angle
  docstring directs users to call it themselves (`preprocess/nsi.py:617`).
* The change is small.  `map_view_batches` (`preprocess/pipeline.py:52`)
  runs on whatever device list it receives and consults no ledger, floors,
  or pin.  Increment 7 is therefore a signature and documentation change,
  not a driver rewrite.
* A preprocessing class was rejected for now.  It would diverge from
  mbirjax's released surface and break public API, while `devices=` on the
  siblings is a small divergence mbirjax can adopt back.  A class stays
  possible later, as a joint mbirjax and mbirtorch API design with its own
  justification.

One caveat is recorded with the decision.  The siblings are single-stage
operations dominated by host transfer, so consistency of the rule, not
measured speed, justifies the parameter.  No measurement time is spent on
it.

## 8. The transition between the categories (decided)

### 8.1 The decision

Data crosses from preprocessing to reconstruction through host memory
(Greg, 2026-08-13).  Model-free preprocessing consumes and produces host
arrays.  The reconstruction then shards the host array under its own
settled layout.  This is what the code already does, so increment 8
records the boundary and changes no code.

`scan_to_sino` writes its output into one pre-allocated host array, and
its workers each fill a disjoint view slice (`preprocess/pipeline.py`).
Model-bound preprocessing never crosses this boundary, because it already
follows the model's layout.

### 8.2 The cost

Three numbers bound the cost at production scale:

* A 2048-cubed float32 sinogram is 32 GiB.
* Moving it takes seconds, at host-to-device rates of tens of GB/s,
  compared with reconstruction times of minutes to hours.
* The host holds the scan and the sinogram together during preprocessing,
  roughly 64 GiB, and gautschi provides about 126 GiB of host memory per
  requested GPU.

### 8.3 Checked and left alone

Two interactions were checked during the discussion and need no work:

* The caching allocator does not distort a later device choice.
  Preprocessing on all devices leaves torch's caching allocator holding
  reserved memory.  The ledger's budget counts driver-free memory plus
  reclaimable cached segments (`_memory_ledger.device_budget_bytes`).
* A device-resident handoff was rejected.  Settling the model first and
  writing preprocessing output directly into reconstruction shards would
  save one host round trip, which costs seconds at the 2048 scale.  The
  handoff would also couple the categories and invert the data flow.  It
  stays rejected until a measurement shows the saving matters.

## 9. Increments

The work proceeds in increments, each reviewed before the next starts.
Increment 1 goes first, because every other reconstruction increment
depends on the settled state.  Increments 2 through 4 then follow in any
order.  Increment 5 follows increment 1 and its own calibration.
Increments 6 and 7 touch only preprocessing and documentation, so they
can run at any point.  Each increment carries the documentation it makes
stale, and increment 8 collects what is left.  Increment 7 ports the
sharded phantom build and follows increment 6, whose shared
permitted-devices helper it uses.

`recon_plastic_metal` (D7) needs no increment of its own.  It runs
`direct_recon` before its beam-hardening loop (`preprocess/mar.py:942`),
so increment 2 settles the layout before the first pass, and increment 1
keeps the device count fixed across passes.  On the `num_metal == 0` path
the single reconstruction calls the policy itself.

Deciding a device layout only helps where the computation can use it, so
each increment below rests on a multi-device implementation that already
exists.  One does not:

| Function the increment touches | The multi-device implementation it uses |
|---|---|
| the three direct reconstructions (increment 2) | `_apply_direct_recon_filter` fans the filter out per device (`tomography_model.py:2374`), and both classes inherit the banded sharded `back_project` |
| `generate_demo_data` (increment 3) | `forward_project(..., output_sharded=True)`, the core sharded projector path |
| `compute_hessian_diagonal`, `prepare_sino_for_devices` (increment 4) | `_shard_sinogram` and the banded `sparse_back_project` |
| `gen_weights` (increment 4) | **none today.**  Increment 4 writes it, which is why that step comes first. |
| `gen_weights_mar` (increment 4) | `forward_project` only.  The weights themselves are host NumPy (`vcd_utils.py:355`), so settling helps the projection and nothing after it. |
| `QGGMRFDenoiser.denoise` (increment 5) | `_denoise_sharded` (`denoising.py:321`), added by the 2026-08 prerelease |
| the five preprocessing functions (increment 6) | `map_view_batches` splits the views across devices (`preprocess/pipeline.py:52`) |
| the phantom build (increment 7) | **none today.**  Increment 7 ports the sharded and blocked builds from mbirjax. |

One open risk has a proposed fix.  Every reconstruction call is sized
today for a full `recon`, so a cone problem too large for one is refused
at its initial direct reconstruction rather than reaching
`split_sino_recon`.  Commit `72208bb` introduced that behavior.
Increment 2 proposes to narrow the capacity check to the work in
progress, which removes the refusal for all four direct reconstructions.
Until that lands, `configure_devices` is the remedy.

A larger idea is recorded but not planned: the policy could choose
`split_sino_recon` itself when a cone problem does not fit, which would
also remove the unconditional split branch in `recon_plastic_metal`
(`preprocess/mar.py:920`) that today splits every cone run whatever its
size.  Splitting changes the reconstruction and not only its placement,
so that choice needs its own design and an explicit caller opt-in.

### 9.1 Increment 1 — the settled state

Increment 1 changes `_apply_device_policy` and the helpers it calls.  No
function gains or loses a policy call.  §2.2 states the rule this
implements, including why the shape comparison does not hook
`refresh_device_bindings`.

**Step 1: add the state.**  `TomographyModel.__init__` gains two
attributes beside `device_layout_is_automatic`
(`tomography_model.py:165`):

* `_settled_shapes` holds the `(sinogram_shape, recon_shape)` pair the
  current automatic layout was decided from.  It is `None` until an
  automatic decision is made.
* `_calibration_scope_open` is `False`.  Step 5 uses it.

**Step 2: record the shapes when the policy settles.**  `_settle`
(`:1432`) already installs the layout and logs the choice.  It now also
stores the model's current `sinogram_shape` and `recon_shape` in
`_settled_shapes`.  This is the only place that attribute is set.

**Step 3: return early on a settled model.**  `_apply_device_policy`
(`:1229`) gains one branch, placed after the existing
`device_layout_is_automatic` branch:

* When `_settled_shapes` equals the model's current shapes, return
  without searching.  The branch body is the explicit-layout branch's
  body: build a ledger only when the calibration mode is on, then call
  `_arm_calibration`.  Pass `**call_arrays` through, so a calibration
  ledger stays accurate.
* When `_settled_shapes` is set and differs from the current shapes, set
  it to `None` and fall through to the existing search.

Two existing branches keep their behavior.  The explicit-layout branch
returns as it does today.  The `visible < 2` branch also returns as it
does today and does not settle, so a host with one visible device never
enters the settled state.

The environment pin changes behavior slightly.  A model pinned by
`MBIRTORCH_NUM_DEVICES` reaches `_settle` today, so it now becomes
settled, and its later calls skip the ledger and the preflight.  That
matches the once-per-model preflight of §2.3.

**Step 4: clear the state when the caller pins.**  `configure_devices`
(`:1129`) sets `_settled_shapes` to `None` where it already clears
`device_choice_rejections`.  The explicit-layout branch never reads
`_settled_shapes`, so this keeps the two from disagreeing rather than
changing behavior.

**Step 5: arm the calibration counters once per reconstruction.**
`_arm_calibration` (`:1474`) calls `_memory_ledger.calibration_start`,
which resets the CUDA peak counters.  `vcd_recon` calls the policy at
`:2945` and calls `direct_recon` at `:2976`, which re-enters the policy
on a cone model.  That second entry resets the peak after the sinogram
and the weights are already placed, so the calibration report at `:3189`
reads a partial run.  Two edits close this:

* `_arm_calibration` calls `calibration_start` only when
  `_calibration_scope_open` is `False`, and sets the flag when it does.
* `vcd_recon` sets `_calibration_scope_open` back to `False` where it
  reads the calibration report (`:3189`).  That is where the measured
  scope ends.

The flag is read only under the calibration mode, so an ordinary run is
unaffected.

**Step 6: correct three docstrings.**

* `_apply_device_policy`'s re-evaluation paragraph (`:1254`) says the
  choice is re-evaluated on every automatic entry.  Replace it with the
  settled-once rule and the shape-change exception.
* The same docstring calls `vcd_recon` the sole call site (`:1246`).
  `fdk_recon` has called the policy since commit `72208bb`.
* `skip_memory_preflight` (`:181`) gains a sentence.  Setting it on a
  model that has already settled has no effect until that model
  re-decides.

**What does not change.**  `refresh_device_bindings` keeps rebuilding the
placements on the current device list and does not touch the settled
state.  `_install_device_layout` stays free of policy.  The floors, the
ledger, and the candidate search are untouched.  `copy_ct_model` needs no
edit: it constructs a new model, which starts unsettled, so the halves of
`split_sino_recon` still decide at their own size.

**Step 7: tests** (`tests/test_device_policy.py`).  Six cases:

* A second automatic call does not re-decide when the free memory
  reading changes.  Fake a smaller reading between the two calls and
  assert the device count is unchanged.
* A `sinogram_shape` change re-decides at the new shape.
* A recompile-flagged parameter that leaves the shapes alone does not
  re-decide.  `det_channel_offset` is the case §2.2 names.
* An explicit `configure_devices` still wins after a settle.
* Under the calibration mode, a cone reconstruction reports one peak for
  the whole run.  The nested `direct_recon` must not reset the counter.
* The existing `copy_ct_model` and `split_sino_recon` tests pass
  unchanged.

Multi-device cases take the suite's `unpinned` fixture, because
`tests/conftest.py` pins the suite to one device.

### 9.2 Increment 2 — the direct reconstructions

Increment 2 gives `ParallelBeamModel.fbp_recon` (`parallel_beam.py:335`),
`MultiAxisParallelModel.fbp_recon` (`multiaxis_parallel.py:386`), and
`TranslationModel.fdk_recon` (`translation_model.py:423`) the policy call
that `ConeBeamModel.fdk_recon` already makes.

**Step 1: add the call.**  `self._apply_device_policy()` becomes the first
statement of each of the three methods, before any placement or filtering,
which is where `cone_beam.py:780` puts it.  The call takes no arguments,
so the ledger prices a plain reconstruction plan at the model's shapes.

**Step 2: nothing else in those methods changes.**  The bodies already
work in device form and stay as they are.  The `direct_recon` delegates
reach the new call through their targets, so each still runs the policy
once.

**Step 3: tests for the call.**  Mirror the existing `fdk_recon` test for
each geometry: on a model with a faked multi-device count and no explicit
layout, a bare direct reconstruction spreads across devices instead of
running on the lead device.  The multiaxis and translation models name no
`_floor_family`, so the parallel floors govern them, and one test records
that.  `usr_multi_gpu.rst` already documents that rule for geometries
without floors of their own.

**Step 4: narrow the capacity check to the work in progress (proposal).**
Today the policy prices every call with a full `recon` plan, so a direct
reconstruction is refused whenever no device count fits a `recon` that
call is not going to run.  §2.3 records that cost.  This step removes it,
and it applies to all four direct reconstructions, `ConeBeamModel`
included, so the behavior that arrived with `72208bb` is corrected in the
same place it is extended.

The change separates two decisions the ledger currently makes together:

* The **device count** is still chosen with the full `recon` plan, as
  §2.3 rules.  The settled layout serves the model's whole life, so the
  count must suit the largest workload the model may later run.
* The **capacity check that can refuse** is made against the plan for the
  call in progress.  A direct reconstruction allocates the sinogram
  shard, the filtered sinogram shard, the recon shard, the
  back-projection reduce term, and the filter's per-device temporaries.
  It allocates no prior, no Hessian, no partition sequence, and no
  reconstruction view batch.

Four edits carry it:

* `_memory_ledger.plan_from_model` gains a `workload` argument, default
  `'recon'`, and a `'direct'` plan that charges only the terms above.
* `_apply_device_policy` gains a matching `workload='recon'` parameter,
  declared before `**call_arrays`.  It passes the value to the ledger and
  records it beside `_settled_shapes` as `_settled_workload`.
* The four direct reconstructions pass `workload='direct'`.
* A call whose workload differs from the recorded one re-runs the
  capacity check on the settled layout.  The layout does not change; only
  the check runs.  This keeps a later `recon` on a model that settled
  under a direct reconstruction from reaching the allocator without the
  preflight's message and its remedies.

Two notes on the shape of this change.  The `workload` argument tells the
ledger what is about to be allocated, which is information the ledger
needs to be correct; it is not a caller-supplied override of the policy.
And it generalizes: increment 5 gives the denoiser its own plan, which
becomes `workload='denoise'` rather than a separate mechanism.

**Step 5: tests for the narrowed check.**  Two cases.  A geometry too
large for a full `recon` still runs a direct reconstruction, where today
it is refused.  A `recon` on that same model is then refused with the
preflight's message, not by the allocator.

### 9.3 Increment 3 — `generate_demo_data` calls the policy

Increment 3 settles the internal generation model before it projects
(`utilities.py`), with the capacity preflight skipped per §4.3.

**Step 1: call the policy after the model is built.**  The generation
model is constructed at `utilities.py:1626-1723` and pinned at `:1732`
when the caller passed `devices=`.  After that block, the function calls
`ct_model_for_generation._apply_device_policy()`.  The call is
unconditional: on a pinned model the policy's explicit-layout branch
returns at once, so no branch is needed here.

**Step 2: nothing about the `devices=` argument changes.**  It still pins
through `configure_devices`, and a pinned layout still wins.

**Step 3: tests.**  With a faked multi-device count and no `devices=`, the
forward projection at `:1779` runs on the settled layout rather than on
the lead device.  With `devices=`, the pin is what the projection uses.

### 9.4 Increment 4 — the full-array allocators settle first

Increment 4 has a strict internal order.  Two existing gaps must be closed
before the settle is added, because settling is what first routes real
callers into them.

**Step 1: give `gen_weights` a per-shard form** (`vcd_utils.py:203`).
Today the function places the sinogram with `ct_model._shard_sinogram` and
then selects its array module with `xp = torch if isinstance(sinogram,
torch.Tensor) else np`.  A `Shards` object is neither, so `xp` becomes
numpy: `'transmission'` raises a `TypeError` and `'unweighted'` returns a
zero-dimensional object array.  The fix builds the weights per shard and
returns `Shards` when the placement has more than one device.  The
single-device and host paths keep their present behavior exactly.

**Step 2: let `split_sino_recon` accept device-form inputs**
(`cone_beam.py:1048-1049`).  It host-slices `sino[:, lo:hi, :]` and
`weights[:, lo:hi, :]`, and `Shards` does not support subscripting.  The
fix gathers a device-form sinogram or weights at entry.  Without it, a
prepared sinogram followed by `split_sino_recon` would raise, and
`recon_plastic_metal` reaches exactly that pair on cone models.

**Step 3: add the settle to the four helpers.**  Each settles an unsettled
model before it allocates:

* `compute_hessian_diagonal` (`tomography_model.py:1992`)
* `prepare_sino_for_devices` (`tomography_model.py:1576`)
* `gen_weights` (`vcd_utils.py:203`), only when `ct_model` is given
* `gen_weights_mar` (`vcd_utils.py:303`), only on the `init_recon` branch
  that forward-projects; the Otsu branch does no device work and does not
  settle

**Step 4: update the docstrings.**  The four helpers state that the call
may change the model's device layout and may raise the preflight's error.
`gen_weights` also loses its claim that the result "stays where the input
is", which a settle can now change.  `configure_devices` states that
pinning after a helper has settled invalidates the `Shards` that helper
returned.

**Step 5: tests.**  Assert on the returned arrays, not only on the model's
layout.  A placement-only assertion would pass while
`gen_weights('unweighted')` returned a zero-dimensional array, which is
the silent failure step 1 removes.  Cover the prepared-then-split pair
from step 2.  Multi-device cases take the `unpinned` fixture.

### 9.5 Increment 5 — the denoiser

Increment 5 brings `QGGMRFDenoiser.denoise` under the policy.  It runs in
three steps, and the floor measurement is reviewed on its own before the
call is added.  Increment 1 is a prerequisite, because `denoise` sets the
recompile-flagged `sigma_noise` on every call (`denoising.py:242`).

**Step 1: build the denoiser's ledger.**  `plan_from_model`
(`_memory_ledger.py`) prices view batches, projector transients, and a
partition sequence.  The denoiser has no projectors at all: its
`create_projectors` is a no-op (`denoising.py:126-128`), and it fixes a
single partition (`granularity=[16]`, `partition_sequence=[0]`).  The
denoiser therefore gets a plan shaped to what it allocates: the image, the
residual, and the qGGMRF working set, with halos for the sharded sweep.

**Step 2: measure denoiser floors.**  Add a `'denoiser'` family to
`_widening_floors.py` and set `_floor_family = 'denoiser'` on
`QGGMRFDenoiser`, which today names none and so inherits the parallel
floors.  Measure the family with an extension of
`dev_scripts/refresh_widening_floors.py`, which already carries the size
ladder, the sentinel probes, and the per-family model builder.  Record in
the family's rows that the denoiser's `sinogram_shape` equals its image
shape, so its floors are read in image voxels while every other family's
are read in sinogram elements.  This is the one cluster measurement in the
plan.

**Step 3: add the call.**  `denoise` calls `self._apply_device_policy()`
before it places the image.

**What this increment assumes.**  A script that holds a denoiser and a
reconstruction model settles each independently, and neither ledger knows
about the other.  The plan assumes their device residencies are
sequential.  A script that needs both resident at once should pin both
with `configure_devices`.

### 9.6 Increment 6 — the preprocessing surface

Increment 6 gives five preprocessing functions the permitted-devices
default, per §7.1.

**Step 1: factor out the default.**  `scan_to_sino` resolves the default
inline (`preprocess/utilities.py:435-447`): all visible CUDA devices,
capped by `MBIRTORCH_NUM_DEVICES`, or the default device when there are
none.  Five more call sites need the same rule, so it moves into a small
shared helper that `scan_to_sino` then calls.

**Step 2: add the parameter to three functions.**
`compute_sino_transmission` (`:38`), `correct_det_rotation` (`:235`), and
`downsample_view_data` (`:373`) gain `devices=None` and pass it to
`map_view_batches`.

**Step 3: change the default for two functions.**
`correct_zinger_pixels` (`:1533`) and `BH_correction`
(`preprocess/mar.py:154`) already accept `devices=`.  Only their
resolution of `None` changes, from one device to the permitted devices.

**Step 4: nothing in the driver changes.**  `map_view_batches`
(`preprocess/pipeline.py:52`) already runs on whatever device list it
receives.

**Step 5: tests and goldens.**  The preprocessing kernels are per-view
with no cross-view reduction, so results do not depend on the device
count and the preprocess goldens must not move.  A test asserts that
agreement across device counts, which is a stronger check than the
reconstruction path can make.

### 9.7 Increment 7 — the sharded phantom build

`generate_3d_shepp_logan_low_dynamic_range` (`utilities.py:117`) builds
the whole phantom on the host with several phantom-sized transients, and
its docstring says the mbirjax blocked and sharded builds "are not
ported".  This increment ports them.  It bounds what increment 3 can
achieve: settling the generation model helps the forward projection, but
a 2048-cubed phantom is 32 GiB on the host before the projection starts,
so demo data at production scale is limited by the build rather than by
the projection.

**Step 1: port the sharded build.**  mbirjax's
`_generate_3d_shepp_logan_sharded` (`mbirjax/utilities.py:859`) gives each
device a contiguous band of slices on the recon shard axis, padded up to
the device count with a zero tail.  Every voxel is independent, so there
are no halos and no cross-device communication.  The torch port uses
`_sharding.Placement` and `_sharding.run_per_device`
(`_sharding.py:556`).  One difference from the source matters: mbirjax
loops over devices without a thread pool because its dispatch is
asynchronous, while torch needs the thread pool to run the bands
concurrently.

**Step 2: port the blocked single-device build.**  mbirjax's
`_generate_3d_shepp_logan_blocked` (`mbirjax/utilities.py:812`) splits the
rows into fixed-size blocks so only one block's transients are live, which
is what keeps a large single-device build inside memory.  `max_block_gb`
bounds the block size.

**Step 3: adopt the mbirjax signature.**  The function gains `devices=None`
and `max_block_gb=4.0`, matching
`mbirjax/utilities.py:713`.  `devices=None` resolves through the shared
preprocessing helper of increment 6, so the phantom follows the
preprocessing rule of §3.1: all permitted devices, no ledger, no floors.
Each device's band is bounded work, which is what puts this function in
that category.

**Step 4: the return stays a host array.**  The phantom is a reference
object.  The build gathers to the host, crops the padded slice tail, and
frees the device arrays, as mbirjax does.  Callers see no change in type.

**Step 5: tests.**  The sharded build matches the single-device build on
the real slices, and the padded tail is exactly zero.  Run at a device
count that does not divide the slice axis.  The blocked build is
bit-identical to the unblocked one, because only the loop structure
differs.  Existing phantom goldens must not move.

Survey row E3 leaves the host-only column when this lands, and §3.1's
rule then covers the phantom build.

### 9.8 Increment 8 — documentation

Increment 8 collects the documentation the earlier increments make stale,
plus the contract from §8.

* Record the host boundary as the contract between the two categories:
  model-free preprocessing consumes and produces host arrays, and the
  reconstruction places them under its own settled layout.
* Correct `usr_multi_gpu.rst`.  Its statement that a reconstruction
  spreads across the visible GPUs with no change to the script holds
  today for `recon`, `prox_map`, and `fdk_recon`.  It holds for the
  direct reconstructions after increment 2 and for the drivers after
  increment 3.
* Correct the stale single-device docstrings in `preprocess/mar.py`,
  which increment 6 makes wrong.
* Correct `generate_demo_data`'s claim that `devices=` affects only where
  the work runs and not the result.  The device count changes summation
  order in the projection, which moves results at the 1e-6 level, so
  golden regeneration must run under the suite's device-count pin.
* Write the base `TomographyModel.direct_recon` contract (A4): a direct
  reconstruction calls the policy, so a new geometry inherits one answer.

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
