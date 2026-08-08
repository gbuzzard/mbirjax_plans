# Device policy: the memory preflight and the all-device default — design

**Status:** DRAFT, awaiting Fable review at the checkpoint-1 STOP.  No
implementation code has been written.

Three records govern this design.  The charter is `current_plans.md`
item 2.  The parked mbirjax ledger design it inherits is item 11.  The
decision record for the default flip is `docs.md` §4.

Four sources are prior art.  The uniform-Shards engine is described in
`phase4_findings.md`, in its Phase 4a stage 1 through 3 sections.  The
banded drivers and the placement chokepoints are in
`mbirtorch/tomography_model.py`.  The shared per-view cost models are in
`mbirtorch/projectors.py` and in the two Triton modules.  The composed
peak measurements are in `kernel_batching_findings.md`.

**Terms.**  The per-device peak model designed here is called the
LEDGER.  The runtime check that consumes it is called the PREFLIGHT.
The word MODEL always means a `TomographyModel` object.  The per-view
transient models the driver already carries are always called COST
MODELS in full.

## What the preflight is for

The preflight exists to make a doomed reconstruction fail in seconds
rather than in half an hour.  The incident that motivates it is recorded
in item 11.  A 2-GPU full recon at 1600×1617×1422 spent 32 minutes inside
XLA's allocator retry loop.  It emitted about 1,900 warning lines and
then surfaced `RESOURCE_EXHAUSTED`.  The allocator's retry policy is not
user-tunable in either framework, so the only place to intervene is
before the first doomed allocation.

The preflight has a second job in mbirtorch that it did not have in
mbirjax.  It is the criterion for the all-device default.  A
reconstruction spreads onto a device only when that device can hold its
share.  Widening without the preflight would convert a clean
single-device out-of-memory failure into a multi-device one, which is
strictly worse to diagnose.  Building the preflight first is therefore
what makes the flip safe.

The ledger must be closed-form rather than a compile query.  The torch
updater is eager Python, so no single compiled artifact ever sees the
cross-call lineup of live tensors.  The ledger must also run before the
compiles it would otherwise wait for.  Both reasons carry over from the
mbirjax design unchanged.

## (a) The per-device peak ledger

### The verified inventory

The persistent set differs from the mbirjax list, so it was read out of
`vcd_recon` rather than copied.  Four differences matter.

First, the measured sinogram is not persistent.  `vcd_recon` folds it
into the error sinogram and then drops its reference at
`sinogram = None` (line 2123).  That happens before the Hessian and
before the loop.  The sinogram is live during the direct recon and the
initial error state, and not afterwards.

Second, the weighted error sinogram is not persistent either.  mbirjax
must hold its weights product through the line search.  The torch updater
frees it immediately after the back projection (line 1825), because
`_forward_lin_quad_weighted` fuses the weights into its reductions.  On
the constant-weights path no weighted array is materialized at all, since
line 1809 binds a bare alias of the error sinogram.  The sinogram-shaped
pair that killed the student's run is therefore one array here, or none,
and never two.

Third, the weights array and the Hessian's weight array are the same
array whenever the caller supplies weights.  Under supplied weights
`hess_weights` is a bare alias (line 2169).  Under constant weights it
is a freshly built all-ones sinogram (line 2167), and `vcd_recon` never
rebinds or deletes that name, so the ones array stays resident for the
whole VCD loop.  The ledger therefore charges exactly ONE
sinogram-shaped weights term, never two:

```
weights_term = sino_dev  if (weights supplied or fm_hessian is None)
               else 0
```

Charging both would over-count a full `sino_dev` on every weighted run.
Weighted runs are the common case, and they are the case the gate cells
measure.

Fourth, the pixel partitions are device-resident and are charged to the
lead device only.  `gen_set_of_pixel_partitions` builds one int64 tensor
per granularity on `model.torch_device`.  It builds one for every entry
of `granularity`, not only for the entries the sequence visits.  The
default granularity list has eleven entries, so the charge is about
`11 × P_full × 8` bytes.  At the 1024 gate cell that is roughly 68 MB.

Four arrays make up the persistent set on each device: the error
sinogram, the weights term above, the flat recon, and the Hessian
diagonal.  The first two are sinogram-shaped and the last two are
recon-shaped.

### Notation

The ledger is a pure function of shapes, a placement, and a call plan.
It never reads a model's live placement, because the widening rule must
price candidate device counts the model is not configured for.

Let `V`, `R`, `C` be the sinogram shape and `Rr`, `Rc`, `S` the recon
shape.  Let `n` be the candidate device count.  The device-form axis
lengths come from `Placement`: `Vd` is the padded view block per device,
and `Sd` is the padded slice block per device.  For a `rows_track_slices`
geometry with a padded slice axis the sinogram's row axis pads with it,
so `Rd` is `S_pad` there and `R` everywhere else.

Three derived sizes carry most of the ledger.  A sinogram-shaped array
costs `sino_dev = Vd × Rd × C × 4` bytes per device.  A recon-shaped
array costs `recon_dev = Rr × Rc × Sd × 4` bytes per device.  A cylinder
array over `P` pixels costs `cyl(P, cols) = P × cols × 4` bytes.

Three pixel counts appear, and they are not interchangeable.  `P_full` is
the ROR-masked count that the projections and the partitions use.
`P_grid` is `Rr × Rc`, the unmasked count that `compute_hessian_diagonal`
uses.  `P_grid` exceeds `P_full` by about 27 percent for a square grid.
A subset at granularity `g` holds `P_g = ceil(P_full / g)` pixels.

### The phases

The per-device peak is the maximum over phases of that phase's
persistent set plus its co-live transients.  Five phases make up a
reconstruction: the direct recon, the initial error state, the Hessian
diagonal, the subset step at each granularity in the sequence, and the
per-iteration statistics.  The statistics phase is dominated by the
others and is charged as zero.

**Phase B, the direct recon.**  This phase runs only when `init_recon`
is None.  Both geometries have the same structure: place the sinogram,
filter it out-of-place, then back-project over `P_full`.  Six terms are
live at the peak: the sinogram (`sino_dev`), the filtered sinogram
(`sino_dev`), the back accumulator and one block
(`2 × cyl(P_full, Sd)`), the projector batch charge, and the scatter
buffer (`recon_dev`).  The filtered array is a separate allocation
because `apply_row_filter` is out-of-place.  The unfiltered sinogram
stays referenced through the back projection in both `fbp_recon` and
`fdk_recon`.  Helical cone adds one more `recon_dev` for the z-weight
pass.

**Phase C, the initial error state.**  Six terms are live at the peak:
the sinogram, the init recon (`recon_dev`), the voxel gather
(`cyl(P_full, Sd)`), the forward assembly buffer (`sino_dev`), one
forward block, and the projector batch charge.  A second, slightly lower
peak follows when the error sinogram is formed.  Four arrays are co-live
there: the sinogram, the init recon, the forward projection, and the
error sinogram.  The scaling step `alpha * init_recon` briefly doubles
the init recon.

**Phase D, the Hessian diagonal.**  This phase runs only when
`fm_hessian` is None.  Five terms are live at the peak: the error
sinogram, the init recon, the Hessian weight array (`sino_dev`), the back
accumulator and one block at the UNMASKED pixel count
(`2 × cyl(P_grid, Sd)`), and the projector batch charge.  This is the one
phase where `P_grid` rather than `P_full` is charged.

**Phase E, one subset step at granularity `g`.**  The step has five
sub-phases and the ledger takes their maximum.  E1 is the qGGMRF prior.
E2 is the back projection of the weighted error.  E3 is the update
direction.  E4 is the forward projection of the delta.  E5 is the state
application.

E1, E3, and E5 charge cylinders only, each of size `cyl(P_g, Sd)`.  E1
charges nine, E3 charges seven, and E5 charges two.  E2 charges two
cylinders plus the back accumulator, one block, and the batch charge.
E2 also charges `sino_dev` for the weighted error sinogram when the
caller supplied weights.  E4 charges one cylinder plus `sino_dev` for
the delta sinogram, one forward block, and the batch charge.

The E1 count of nine is the qGGMRF kernel's co-live array inventory
under `torch.compile`.  The kernel holds four arrays across its in-slice
loop: the central cylinders, the running gradient, the running Hessian,
and `b_tilde_2_delta`.  That last array is built at `qggmrf.py:151`,
read once at line 159, and then never rebound or deleted, so it survives
to the function's return.  Each loop iteration adds five more: the
neighbor gather, the difference, the `b_tilde_2`, and the incoming
buffers of the two accumulator rebinds.  The neighbor gather cannot fuse
away, so the count does not collapse under compilation.  The code
comment at `create_vcd_subset_updater` already names this chain as the
updater's memory attention point, and the ledger agrees with it.

E1 carries the ledger's largest modelling uncertainty, and its bound is
worth stating.  Under the eager fallback the `b_tilde_by_definition`
chain materializes six or seven more same-shape temporaries.  A silent
compile fallback would therefore leave E1 under-charged, which is the
forbidden direction.  Two facts bound the exposure.  `maybe_compile`
records every fallback in `projectors._COMPILE_ERRORS` and rebinds
permanently to eager, so the condition is detectable.  E2 exceeds E1 by
a full `sino_dev` on any weighted run, so E1 is not the dominant
sub-phase in the common case.  Checkpoint 2 measures E1 in isolation and
pins the count.

The E3 count of seven is an upper bound on `direction_worker`.  Six
arrays are simultaneously live at its peak: the prior gradient and
Hessian, the back-projected error shard, the forward gradient, the
forward Hessian, and the update direction.  The seventh covers the
Hessian row gather at line 1835, which is live only while the forward
Hessian is being formed.

The positivity constraint, when enabled, adds one cylinder to E4 and
repeats E4's forward projection.

The maximum is taken over the granularities the sequence actually
visits, not over the whole granularity list.  At the default
`partition_sequence` and `max_iterations=15`, the visited granularities
are 4, 16, 64, and 128.  The coarsest visited granularity is therefore
4, and `P_g = P_full / 4`.  At the gates' 3-iteration setting the
visited granularities are 4, 16, and 64, and the coarsest is the same.

### The projector term through the shared cost models

The batch transient is computed by calling the same cost model the
driver calls, so the batch chooser and the ledger cannot drift apart.  A
kernel body carries `_view_batch_cost(num_pixels, band_cols, args)`
returning `(bytes_per_view, view_chunk)`.  A torch body has no such
attribute and is charged `num_pixels × _transient_cols(band_cols) × 4`
bytes per view.  `Projectors._effective_view_batch` already implements
both branches and already applies the size-scaled budget.

One small refactor makes the cost model shareable.
`_effective_view_batch` returns only the batch, and the ledger needs the
charged bytes as well, so the charge computation moves into a new
method:

```python
def view_batch_charge(self, body, num_pixels, band_cols, args, n_devices=None):
    """(view_batch, bytes_per_view) for one call of `body`."""

def _effective_view_batch(self, body, num_pixels, band_cols, args):
    return self.view_batch_charge(body, num_pixels, band_cols, args)[0]
```

The `n_devices` argument is what lets the ledger price a candidate
layout.  `_transient_budget_bytes` currently reads
`self.model.sino_placement` directly, so it takes the same optional
argument and keeps its present behavior when the argument is omitted.
Existing call sites are unchanged, which keeps the n=1 batch selection
bit-identical.

The cost models deliberately exclude the call-fixed output tensors, and
their docstrings say so.  The ledger therefore adds those outputs itself,
as the accumulator and block terms listed per phase above.  This division
is the correct one.  The cost model answers what one more view in the
batch costs.  The ledger answers what is resident at once.

Which body is bound is answered without running a projection.
`model._view_batch_bodies()` returns the same two function objects
`Projectors.__init__` binds.  The first call triggers the per-device
kernel self-check, which costs milliseconds and would run at projector
build anyway.

### Per-device differences

Both placements are built over the same device list, so every device is
both a view-owner and a slice-owner.  The role terms therefore do not
differ by device index.  Three real per-device differences remain.

The first is the empty-shard extensions.  A device with no real views
skips every projector call in the banded drivers, and a device with no
real slices skips its band.  The ledger charges a device its view-owner
terms only when `n_valid > 0` on the view axis, and its slice-owner
terms only when `n_valid > 0` on the slice axis.  These are the same
predicates `padded_shard_ranges()` already returns and the drivers
already test.

The second is the lead device's extras: the pixel partitions, the
`full_indices_device` cache, and the 0-d line-search combines.  Only the
first two have size worth charging.

The third is the band reduce.  It is the most important term in the
ledger.  `sum_band_to_owner` moves ALL `n` partials onto the owner before
summing them, because the list comprehension at `_sharding.py:245`
completes before the summation loop begins.  The owner therefore holds
`n` arrays of `cyl(P, band_len)` plus the running total.  At `n >= 3` the
old and the new total coexist during a rebind, so the count is `n + 2`
there and `n + 1` at n=2.  With the default one-band-per-owner setting
`band_len = Sd = S_pad / n`, so the term is:

```
n = 2:  1.50 × cyl(P, S_pad)
n = 4:  1.50 × cyl(P, S_pad)
n = 8:  1.25 × cyl(P, S_pad)
```

These results indicate that the band reduce is flat in the device count
over the range that matters.  Adding devices shrinks the persistent set
by `1/n` and leaves this term where it was.  `back_project_slice_band`
is the only lever that reduces it, and the error message below names
that knob for exactly this reason.

The forward driver has a smaller analog.  A row-aligned geometry
accumulates every slice-owner's row-band and then concatenates them, so
`2 × sino_dev` is live at the concatenation.  A two-fan geometry
accumulates in place and holds one partial plus the accumulator, also
`2 × sino_dev`.

### The direct-recon path

Phase B above is the direct-recon path, and it precedes the loop rather
than living inside it.  Its filter is a `(C,)` array and is negligible.
Its cost is the full-index back projection.  That is the largest single
projection of the run, at `P_full` pixels over the full view range.  It
is charged only when `init_recon` is None, which is the default for
`recon()` and is not the default for a resumed or chained call.

## (b) Where the preflight runs

### The site

The preflight runs once at the top of `vcd_recon`, immediately after the
sinogram shape check and before the weights placement.  That is the
single site every reconstruction entry funnels through.  `recon()`
reaches it through `initialize_recon`, and `prox_map()` reaches it
directly.  Nothing large has been allocated by `vcd_recon` at that point.

The denoiser is a deliberate exclusion, and the reason is worth
recording.  `QGGMRFDenoiser.denoise` does not call `vcd_recon`.  It runs
its own loop over a single fixed partition with an identity forward
model.  It also does not work under a non-trivial placement today,
because `denoise` calls `.clone()` on the result of `_shard_recon`, and
`Shards` has no `clone`.  The denoiser is therefore single-device in
practice: nothing stops a user from calling `configure_devices` on it,
and `denoise` then raises.  It has no projector transients and no
widening, and its peak is a handful of recon-shaped arrays.  It gets no
preflight.  Its sharding gap is flagged below as a separate item.

This is also the strongest argument for the `vcd_recon` site over a
construction-time site.  Auto-widening at construction would give
`QGGMRFDenoiser` a non-trivial placement on a multi-GPU machine and break
`denoise` outright.

### Free memory, other processes, and the caching allocator

The budget per device is the bytes a new allocation can still obtain:

```
budget(d) = torch.cuda.mem_get_info(d)[0]
          + torch.cuda.memory_reserved(d) - torch.cuda.memory_allocated(d)
```

The first term is what the CUDA driver will still hand out.  It already
excludes memory held by other processes and memory this process has
reserved.  The second term is the caching allocator's reserved-but-unused
pool, which torch can reuse without a new `cudaMalloc`.  Including it is
correct because torch releases cached segments before it reports an
out-of-memory error.

Memory held by other processes is treated as unavailable.  That is the
right default.  The preflight cannot evict a neighbor, and a run that
depends on a neighbor exiting is not a run that should be started.

The reading is a snapshot, and that is its limitation.  A neighbor
process that grows after the preflight can still exhaust the device.  The
error message does not claim otherwise, and the margin is not sized to
absorb it.

Fragmentation is the second thing the budget cannot see.  The reserved
pool may be split into blocks too small to serve a large request even
when the total is sufficient.  The margin absorbs modest fragmentation.
A run that fails after passing the preflight is expected to be rare
rather than impossible.

The demand side needs one correction.  Arrays the caller already placed
on a device are already counted in `memory_allocated`, so charging them
again would double-count them.  The preflight therefore credits every
array it was handed that is already CUDA-resident, at its true byte size,
on the device where it lives.  The set of such arrays is exactly the
argument list of `vcd_recon`: the sinogram, the weights, the init recon,
the prox input, the resume error sinogram, the precomputed Hessian, and
the partitions.  Nothing else the ledger charges can be pre-allocated.

The preflight is then, per device:

```
(1 + margin) × (modeled_peak(d) - credit(d))  <=  budget(d)
```

The margin defaults to 0.15 and is exposed as
`model.memory_preflight_margin`.  A user near the boundary can lower it
rather than disable the preflight entirely.  The default sits at the top
of item 11's 10-to-15 percent range, because the ledger under-counts two
things that scale with the working set.  Those two are allocator
fragmentation, and library workspaces that CUDA allocates outside
torch's allocator.

### The override

The override is `model.skip_memory_preflight`, a plain attribute
defaulting to False.  It is an attribute rather than a keyword argument,
for two reasons.  It reaches `recon` and `prox_map` without changing
three signatures.  It also sits beside the memory knobs it belongs with:
`view_batch_size`, `forward_project_slice_band`, and
`back_project_slice_band`.

An environment-variable form was considered and declined.  The device pin
below is process-wide, so it earns an environment variable.  Skipping the
preflight is a per-model decision.

### The error message

The preflight raises one error before any large allocation.  The message
names the dominant phase and the remedy that matches it.  A worked
example follows:

```
MemoryPreflightError: this reconstruction needs more memory than cuda:1 has free.

  device      modeled peak      available     shortfall
  cuda:0           18.4 GB        31.2 GB             -
  cuda:1           18.4 GB        12.7 GB        5.7 GB

  The dominant phase on cuda:1 is the Hessian diagonal back projection
  (11.9 GB of the 18.4 GB): a 2-device layout over 1024 views, with the
  band reduce holding 3.2 GB and the batch transient 2.1 GB.

  Remedies, most effective first for this shortfall:
    model.back_project_slice_band = 128   # the band reduce is flat in the
                                          # device count; this is its only lever
    model.view_batch_size = 32            # caps the projector batch transient
    model.configure_devices(num_devices=4)# more devices shrink the persistent
                                          # set but not the band reduce
    model.set_params(granularity=[...])   # a finer coarsest granularity shrinks
                                          # the subset transients
  To run anyway: model.skip_memory_preflight = True
```

Three properties of that message are deliberate.  It reports every device
rather than only the failing one, because the shortfall's shape tells the
user whether the problem is the layout or one busy device.  It names the
dominant phase with its share, because the remedy that helps depends on
which phase dominates.  It does not lead with "use fewer devices",
because the band reduce analysis shows that fewer devices is often the
wrong answer.  `split_sino_recon` will become a named remedy when it is
ported.  It is `PENDING` today and is not named.

## (c) Calibration, not user-facing

Two paths keep the ledger honest, and neither is on by default.

The verbose path prints the ledger.  At `verbose >= 2`, on a successful
run, `vcd_recon` prints the per-device phase table it already computed.
The printout costs nothing extra and sits beside the existing
`get_memory_stats` calls at the same verbosity.  It keeps the ledger
inspectable without a debugger.

The calibration path compares the ledger against the measurement.  Under
`MBIRTORCH_MEMORY_CALIBRATION=1`, the preflight calls
`torch.cuda.reset_peak_memory_stats` on each device.  `vcd_recon` then
reads `torch.cuda.max_memory_allocated` per device at exit and warns when
the ratio falls outside the acceptance band.  The environment variable is
required rather than optional, because resetting the peak counter would
otherwise clobber the very measurement the gate scripts read.  When the
flag is on, the calibration path owns that counter and says so in its
warning text.

The acceptance band is `1.00 <= modeled / measured <= 1.30`.  The lower
bound is the one that matters.  A ledger that under-predicts would let
the preflight pass a doomed run.  That failure is the one this whole
design exists to prevent.  A cell below 1.00 blocks checkpoint 2 and is
fixed by adding the missing term, never by a scale factor.  A cell above
1.30 must have its over-charge attributed before shipping.  The
preflight's 15 percent margin applies on top of the ledger and is not
part of this band.

A first-pass hand calculation supports the band being reachable.  The
parallel 1024 gate cell runs with supplied weights, so its persistent set
is 16.1 GB: the error sinogram at 4.09 GB, the weights at 4.09 GB, and
the flat recon and Hessian diagonal at 3.97 GB each.  Adding E2's terms
at the coarsest visited granularity gives about 24.2 GB, against the
measured 26.7 GB composed peak.  That first pass under-predicts by about
9 percent.  The residual is expected to sit in three terms this design
enumerates but the hand pass approximated: the compiled qGGMRF chain's
fused temporaries, the forward assembly block, and the cold-pass compile
workspace.  The gate's peak is a `max(cold, warm)`, so it includes that
workspace.  Checkpoint 2 will attribute the residual per phase before
reporting the table, per the standard set by
`kernel_batching_findings.md` and `phase5_findings.md`.

The unit tests run on CPU with synthetic device budgets.  The ledger is a
pure function of shapes, a placement, a call plan, and the two cost
models, so nothing in it needs CUDA.  The tests feed it a fabricated
`Placement` over `["cpu", "cpu"]` and a fabricated per-device budget.
They assert five things: the phase ordering, the per-device symmetry, the
empty-shard skips, the granularity maximum, and that the preflight raises
with the dominant phase named.  This is a checkpoint-2 requirement from
the charter, and it is what the pure-function shape is for.

## (d) The widening rule

The rule is to be implemented in checkpoint 3.  It is stated here so its
decisions are reviewed alongside the ledger that serves it.

### The selection

With no explicit `configure_devices` call, the device count is chosen as
the largest visible CUDA count that passes both the empty-shard
validation and the per-device ledger, with fallback toward n=1:

```python
for n in range(torch.cuda.device_count(), 1, -1):
    if not layout_valid(n):        # _check_no_empty_shard, no mutation
        continue
    if ledger_fits(n):             # every device 0..n-1 passes its own budget
        return n
return 1
```

Every device in the candidate set must pass its own budget.  Shards are
equal-sized by construction, so the device with the least free memory
binds.  That is the intended treatment of heterogeneous devices.  A
machine with one busy GPU and three idle ones falls back, rather than
starting a run that only three devices could hold.  Rejected candidates
are logged at `verbose >= 2` with the reason, which is what a support
question will need.

Devices are the prefix `cuda:0 .. cuda:n-1`, unchanged from the present
`configure_devices`.  One limitation follows.  A machine whose `cuda:0`
is busy cannot reach `cuda:1..3` through the automatic path.  The
remedies are `CUDA_VISIBLE_DEVICES` and the explicit `devices=`
argument, and both already work.

### Where it triggers

The selection runs at the top of `vcd_recon`, at the same site as the
preflight, and not at model construction.  Three reasons, in order of
weight.

The denoiser reason is decisive and is given above.  Construction-time
widening would break `QGGMRFDenoiser.denoise` on any multi-GPU machine.

The free-memory reason is next.  The ledger needs a budget, and the
budget is only knowable when the reconstruction is about to run.  A
reading taken at construction can be stale by the time `recon` is called.
A stale reading is worse than no reading, because it would be trusted.

The scope reason is last.  Widening at `vcd_recon` means the
reconstruction entries widen and standalone `forward_project` calls do
not.  That is the correct scope.  The user story in the charter is a
reconstruction.  A developer calling the projector directly has not asked
for a layout change.

The widening is sticky.  The chosen layout is installed through the same
internal path `configure_devices` uses, so a later `forward_project` on
the same model inherits it.  The selection is re-evaluated on each
`vcd_recon` entry while the policy is automatic, and the placements are
rebuilt only when the chosen count differs from the current one.
Re-evaluation is a closed-form ledger pass plus one `mem_get_info` call
per device, so an unchanged count costs microseconds.

An explicit `configure_devices` call switches the model out of automatic
mode permanently.  It keeps exactly its current meaning, including
`num_devices=1` as the reproducibility pin.  A geometry change still
routes through `refresh_device_bindings`, which rebuilds the placements
at the current count.  Under automatic mode the next `vcd_recon` then
re-evaluates against the new shapes.

### Eligibility

Automatic widening applies only when the model's device is CUDA and was
not named with an index.  `_resolve_device` returns
`torch.device('cuda')` with no index for both `device='auto'` and
`device='cuda'`, so both are eligible.  `device='cuda:2'` gives an
indexed device and is not eligible.  Naming a device is treated as naming
a device.  This is a deliberate choice, and the docs page will state it.

### Test-suite determinism

The suite must not silently change device count on a multi-GPU machine.
The mechanism is an environment variable, `MBIRTORCH_NUM_DEVICES`, read
by the automatic path.  When set, it pins the count exactly as an
explicit `configure_devices(num_devices=n)` call would.  Three things
still hold: the empty-shard validation still applies, the preflight still
runs and can still fail the run, and the count is never reduced below the
pin.  An explicit pin means explicit.

The suite uses the variable through one autouse fixture in
`tests/conftest.py` that sets it to `1` for the whole session.  It is a
fixture rather than a monkeypatch, so the suite's determinism is a
visible, auditable line rather than hidden state.  Every existing n>1
test already calls `configure_devices(devices=[...])` explicitly, and
every model in `tests/test_sharding.py` is built with `device="cpu"`, so
none of them change.  One new test asserts the determinism itself:
construct a model, run a tiny recon, and assert
`sino_placement.n_devices == 1`.

### One experiment script needs a change

The measurement scripts are not uniformly safe, and the exception was
found by checking rather than by assuming.

`kb3_gate.py` is safe.  It never calls `configure_devices`, and its job
runs under `--gpus-per-node=1`, so `torch.cuda.device_count()` is 1 and
the automatic path cannot widen.  The four composed n=1 gate cells are
therefore structurally unaffected by the flip.

`p4_gate_readout.py` is NOT safe, and its n=1 arm is the one at risk.
Its jax branch calls `configure_devices(n_dev)` unconditionally at line
43.  Its torch branch calls it only when `n_dev > 1`, at lines 65 and 66,
and builds the model with the unindexed `device="cuda"` at line 63.  The
job requests 2 or 4 GPUs, and each arm runs as a bare subprocess with no
`CUDA_VISIBLE_DEVICES` restriction.  The torch n=1 arm would therefore
auto-widen to n=2 or n=4 under the flip.  That arm is also the reference
the value diffs are taken against, so the parity columns would silently
compare a widened run against itself.  Checkpoint 3 fixes the script by
pinning its torch n=1 arm, either with `MBIRTORCH_NUM_DEVICES=1` in the
arm's environment or with an unconditional `configure_devices(n_dev)`.

The general lesson for checkpoint 3: any script that builds an unindexed
CUDA model on a multi-GPU allocation and relies on getting one device
must be pinned.  Checkpoint 3 audits all of them rather than spot-checking.

### CUDA_VISIBLE_DEVICES

`torch.cuda.device_count()` already respects `CUDA_VISIBLE_DEVICES`, and
the visible devices are indexed `0 .. count-1` in the restricted
ordering.  The automatic path therefore never looks past the variable,
and the existing `configure_devices` device construction needs no change.
A user restricting the pool gets exactly the pool they asked for.

## (e) The re-gates the flip requires

The n=1 gates must not move.  Those gates are the batching tables.  The
composed peaks of `kernel_batching_findings.md` are the memory
baselines: 26.7 GB at both 1024 cells, and 2.3 to 2.4 GB at the 512
cells.  The time baselines are the four ratios against jax: 1.13x, 1.56x,
0.88x, and 1.00x.  Two properties protect them.  The preflight is
read-only and adds one closed-form pass plus one `mem_get_info` call per
device.  The widening cannot fire where `device_count()` is 1.
Checkpoint 3 re-runs the four composed cells and expects them within
noise of those numbers.

The n>1 value gates compare eager to eager.  That is the compile-latitude
policy of `projector_layer_design.md`.  The large fused bodies let
`torch.compile` generate different float realizations per instance and
per input shape.  A compiled n>1 comparison would therefore measure the
compiler's latitude rather than the engine's correctness.  Eager-to-eager
bounds the logic at the true float floor, about 5e-4 at the cells
measured there.  Compiled n>1 trajectories are accepted within the
documented amplified envelope.

The n=2 and n=4 gates for this work are correctness plus a basic sanity
check, not tuning.  Item 3 owns the full multi-GPU performance readout.
The correctness gates follow the existing patterns in
`tests/test_sharding.py`.  Each gate compares a seeded n>1 run against
n=1.  Each cell's tolerance is calibrated against the DIVIDING-case floor
of that same cell, per the phase 4a stage 3 calibration finding.  The
basic sanity check is that the automatic path selects the expected count
on a 2-GPU and a 4-GPU node and that the run completes.

One new gate is specific to this work: the preflight blocks a doomed run.
A synthetic per-device budget is injected through the ledger's
pure-function seam.  The preflight must then raise before any large
allocation, and the message must name the dominant phase.  This test runs
on CPU.

The docs handoff is a flag, not an edit.  `usr_multi_gpu.rst` is written
by the docs session AFTER this lands, per `docs.md` §4, and this work
does not touch `docs/`.  Checkpoint 3's record will state that the flip
has landed and that the page can be written.  The record will also carry
the four facts §4 lists for the page: the auto-spread default, the
preflight that guards it, `configure_devices(num_devices=1)` as the
reproducibility pin, and a note that results can differ slightly with
device count.  That last note should add that the difference decays with
iterations.  The three constructor docstrings and `configure_devices` are
updated by checkpoint 3 itself.

## Open decisions for review

**1. The `hess_weights` residency.**  Under constant weights only,
`vcd_recon` builds an all-ones sinogram into `hess_weights` (line 2167)
and never releases it, so it stays resident for the whole VCD loop.  Were
the 1024 gate cell run unweighted, that array would be 4.09 GB.  The gate
cells supply weights, so this residency is NOT part of the measured
26.7 GB and cannot be measured by the gate as configured.  The fix is one
line.  The recommendation is to not fix it in checkpoint 2.  Checkpoint
2's deliverable is ledger-versus-measured agreement, and changing the code
under the measurement would muddle it.  Checkpoint 2 should instead do
three things: charge the residency in the ledger on the constant-weights
path, measure it with one unweighted probe cell, and report it.  The fix
then lands afterwards as a small change whose effect the now-validated
ledger predicts exactly.

**2. The acceptance band.**  `1.00 <= modeled / measured <= 1.30` is
proposed above.  The lower bound is not negotiable in this design's
logic.  The upper bound is a judgement call, and it is the number most
likely to want adjustment after checkpoint 2's first table.

**3. The margin default.**  0.15, the top of item 11's range.  It is
exposed as `model.memory_preflight_margin` so it is tunable without
disabling the preflight.

**4. E1's compiled-versus-eager count.**  The ledger charges nine
cylinder arrays for the qGGMRF prior under compilation, and the eager
path holds six or seven more.  A silent compile fallback would therefore
under-charge.  The recommendation is to charge the compiled count and
rely on the calibration path plus `_COMPILE_ERRORS` to detect the
fallback.  The alternative is to always charge the eager count, which
over-charges a weighted 1024 run by roughly 4 GB and risks the 1.30 upper
bound.

**5. The denoiser's sharding gap.**  `QGGMRFDenoiser.denoise` raises
under any non-trivial placement, at the `.clone()` on a `Shards` result.
This design leaves it single-device and excludes it from both the
preflight and the widening.  It is flagged as a separate item rather than
fixed here.

**6. Re-evaluation on every entry.**  The automatic count is re-evaluated
on each `vcd_recon` entry rather than fixed at the first one.  The
alternative is a first-recon decision that persists.  Re-evaluation is
recommended, because a long-lived model in a Plug-and-Play loop can
outlive the memory conditions its first decision was made under.

## Checkpoint-1 review ruling (Fable, 2026-08-07): APPROVED, with a scope change

Every load-bearing claim was verified in code before approval: the four
vcd_recon inventory differences, the band-reduce co-residency (the list
comprehension at `_sharding.py` really does move all n partials before the
sum loop), the `b_tilde_2_delta` residency, the denoiser `.clone()`-on-Shards
bug (real; now recorded in `current_plans.md` item 11 as its own follow-up),
and the `p4_gate_readout.py` n=1-arm hazard (real).

**Scope change (Greg): the preflight is MULTI-DEVICE-CUDA-ONLY — its point
is to determine the device count.**  Torch's caching allocator does not have
XLA's slow retry-loop death: a single-device torch OOM raises quickly with a
readable message, so the fail-fast job the mbirjax design needed is largely
provided by torch itself.  Consequences for the design: the ledger runs only
on the AUTOMATIC widening path (unindexed CUDA device, `device_count() >= 2`,
no explicit `configure_devices`).  Single-GPU machines and explicitly
configured layouts get no preflight; an explicit count is the user's count.
One nuance is retained: when the automatic path finds that NO candidate
count fits — including n=1 — it raises the readable dominant-phase error
rather than launching a known-doomed run (the answer to "which count" is
"none", and the ledger has already been computed).  The error-message design
above survives for exactly that case; the per-model `skip_memory_preflight`
attribute survives as the force-run escape for it.  The calibration path
(`MBIRTORCH_MEMORY_CALIBRATION=1`) computes the ledger at ANY count,
including n=1, since checkpoint 2's modeled-versus-measured table needs it —
calibration is a measurement mode, not the production gate.

**Rulings on the six open decisions:** (1) `hess_weights` — as recommended:
charge it, measure it with one unweighted probe cell, land the code fix
after checkpoint 2 as its own predicted-effect change.  (2) The acceptance
band 1.00–1.30 — approved; the lower bound is non-negotiable.  (3) Margin
0.15 — approved.  (4) E1 — charge the compiled count of nine, with one
refinement: the ledger consults `projectors._COMPILE_ERRORS` at evaluation
time and charges the eager count once a qGGMRF fallback has been recorded
in the process.  (5) The denoiser gap — separate item, recorded.  (6)
Per-entry re-evaluation — approved (the Plug-and-Play argument), with a
verbose log line whenever the chosen count CHANGES between entries.

**Three additions for checkpoint 2:** state that widening and the ledger
are CUDA-only with MPS/CPU models unchanged; one sentence on the resharding
transient (a CUDA-resident caller array's source device briefly holds
full-plus-shard; it rides the margin); and the checkpoint-3 script audit
extends to the mbirjax_metrics harness scripts, not only
`plans/experiments/`.

## Checkpoint-2 review ruling (Fable, 2026-08-07): APPROVED, and the two residency fixes are greenlit

Checkpoint 2's record is `device_policy_findings.md`, approved as written:
the ledger calibrates inside the 1.00–1.30 band at all five cells
(1.001–1.169) through five attributed corrections and one named constant
term, and the acceptance controls anchor it to the composed-gate
configuration.  The open items resolve as the findings recommend, with the
`weighted_fwd` and masked-Hessian fixes approved as follow-on changes under
the instructions below.  Land them in this order, each as its own change.

### Fix 1: release `weighted_fwd` before the error sinogram assignment

One line in `_initial_error_state`: rebind `weighted_fwd = None` after its
second dot product and before the `error_sinogram = sinogram - alpha * fwd`
assignment.  The ledger's initial-error-state enumeration drops the term in
the SAME change, and the calibration re-run must confirm the predicted move
at the weighted 1024 cells: 26.71 to 24.53 GB modeled, with the dominant
phase passing to the Hessian diagonal.  The sharded branch already has no
such array and is untouched.

### Fix 2: the masked Hessian, with the scatter-into-zeros step

`compute_hessian_diagonal` gains an `indices=None` argument.  `None` keeps
today's exact behavior — the full-grid `arange`, whose dense back projection
IS the flat recon, finished by a plain reshape — so the public contract, the
goldens on the public method, and mbirjax parity on that surface are
untouched, bit for bit.  `vcd_recon` passes the ROR-masked index set at its
one internal call site.

The masked path needs the step the full-grid path never did.  A masked back
projection returns a `(P_full, num_slices)` cylinder array that is NOT
reshapeable to the grid, so the implementation allocates a full flat
`(P_grid, num_slices)` zeros array, `index_copy_`s the rows in at the masked
indices, and reshapes as before.  Downstream flat indexing is unchanged.
Every index the engine ever reads is ROR-masked, and the back projection is
per-pixel independent, so the values at every read site are bitwise
identical; the outside-ROR entries become zeros instead of
computed-but-never-read values.

The accounting to encode in the ledger, in the same change: the Hessian
phase's cylinder terms move from `P_grid` to `P_full` (the transient saving,
about 2.4–2.6 GB at the 1024 cells), and a scatter co-residency term is
added for the moment the full zeros output and the `P_full` accumulator
coexist (about 1.78 cylinders).  That moment sits below the loop's own
three-cylinder peak, so it must not become the phase maximum — the
calibration re-run checks this along with the headline move.

Tests for fix 2: a masked-vs-full agreement test asserting equality AT THE
MASKED INDICES (the only places the engine reads), and a bitwise recon
parity test (one small seeded recon before and after the internal call-site
change must produce identical results).

### The compounding, for the record

The findings priced the fixes separately (8 percent, then 4 percent at the
unweighted cell).  They compound at the weighted gate cells: after fix 1
the Hessian dominates there too, so fix 2 then moves the weighted-1024 peak
down to the subset phase, about 23.4 GB — roughly 12 percent combined.  The
post-fix calibration table is the check on that arithmetic.

### Also recorded at this checkpoint

The projector-batch-charge under-statement (about 45 bytes per view-pixel
realized against the 16 charged) is recorded with its revisit trigger: a
future cell whose dominant phase is projector-bound.  The `hess_weights`
release stays unmade, with the ledger's own refutation as the reason.  A
concurrent-session note: the preprocess goldens were regenerated mid-review
and exposed an over-tight tolerance on a non-converging fit
(`test_beam_hardening_family`, fixed separately); golden regeneration in the
shared checkout should be announced between sessions.

## Post-merge notes for checkpoint 3 (2026-08-07, late)

`split_sino_recon` is now ported and reviewed, so the preflight error
message adds it as a named remedy for cone models, as this design
anticipated.  Its half models pin their devices UNCONDITIONALLY to the
parent's layout (made so in review), so a half never enters the automatic
widening path; the checkpoint-3 audit can treat `split_sino_recon` as
pinned by construction.

## Checkpoint-2 closure ruling (Fable, 2026-08-08): CLOSED; checkpoint 3 proceeds

Both residency fixes landed and beat their prediction (13.0 and 11.2
percent at the weighted 1024 cells against the predicted ~12, with a
14.6 percent bonus at parallel 512), and the post-fix ledger envelops at
1.001–1.104.  Three rulings on the closure record:

1. The EAGER form of the whole-recon parity proof is ACCEPTED as stronger
   than the compiled form the earlier ruling asked for.  The reasoning
   stands: the masked and full paths compile different shapes, and dynamo's
   shape specialization perturbs unrelated kernels, so a compiled
   end-to-end diff measures the compiler.  Bitwise identity at every masked
   index under both execution modes, the partitions-inside-mask test, and
   the eager end-to-end bitwise identity settle the value claim.
2. The per-iteration-statistics term is APPROVED as charged (persistent
   set plus two sinogram-shaped arrays).  A second declined residency is
   RECORDED beside `hess_weights`: fusing the statistics phase's two
   squared-error products would cut about 7.6 GB where that phase
   dominates, which today is only the unweighted cell — not a gate cell.
   The revisit trigger is a stats-dominated or unweighted production
   workload.
3. The cold-compile control (four repeats: warm runs bitwise identical,
   the cold run off at 2.1e-4) is a standing fact for every future value
   protocol: warm compiled runs are reproducible, and a cold-vs-warm diff
   measures the compile.  The warm protocols already discard the cold run;
   now the record says why that is load-bearing.

The probe's own two failures — the tail reading masquerading as a
whole-run peak, and the uninstrumented fifth region — are the checkpoint's
most durable lesson: every phase enveloping is not the peak enveloping.

## Checkpoint-1 staged files

`plans/torch_port/device_policy_design.md` (this document).
