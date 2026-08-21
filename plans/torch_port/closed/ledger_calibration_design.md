# Design: the memory-ledger calibration pass

**Status: RULED 2026-08-19 (§6): the probe ran, the term changes are
tabled, and the projection-entry preflight is closed without action.**
Opened 2026-08-19; the probe (§2) ran the same day as
mg42a (jobs 15376256 and 15377054, findings §1.35, run record
mg42a_ledger_probe.md).  Three of its verdicts, in brief: the
three-device over-read reproduced and is cone's back batch charge plus
parallel's deliberate forward covers; the lead-device transient does
NOT exist in a single reconstruction, so no ledger term is owed --
though the nightly's own 26.6 GiB reading is not yet attributed, and
the candidate mechanism (multi-device end-states freeing at garbage
collection, one per repeated call) would be a library finding with a
cheap discriminating probe, findings §1.35; and one NEW
under-read appeared -- parallel at one device reads 0.935, below the
band, reproduced twice -- which changes §6(d)'s premise: the
under-charge record is no longer clean, and the term increment (§3)
now leads with that gap.  This is open item C5.  The five original
inputs are collected, with their citations, in
`ledger_calibration_inputs.md`; this note says what the pass does
about each and in what order.

## 1. What the five inputs are, read together

The five inputs are not five independent defects.  Read against the
code, they group into three questions.

The first question is where the ledger's slack sits.  Input 1 says the
back-projection batch charge counts four per-view slabs where three are
live at the launch instant, about 0.8 GB high at parallel 1024 on two
devices.  Input 2 says the model over-reads the measured peak by up to
1.417 at the 512-class three-device arms, against a band top of 1.30.
These two are connected, and the connection is deliberate: the ledger's
`back_fixed` term charges two live blocks where the code's reading
suggests up to three, and its own comment says the shortfall is
absorbed by the batch charge beside it, which measured 30 to 45 percent
larger than what is held.  So the slack is a designed compensation
between two terms, not an accident, and trimming either term alone
could push some cell below the 1.00 floor.  The pass must measure the
worker phase whole and rebalance the pair together.

The second question is whether a cost exists that no term names.
Input 5 reports the nightly's four-device VCD arm pushing device 0
about 3.1 GB above the one-device arm's own peak, transiently, in both
geometries.  The reading came out of a harness whose watermark is
process-cumulative, so the first thing the pass must establish is
whether the transient exists in a fresh process with its own counters.
If it does, the probe's phase instrumentation names the region, and the
region's arrays name the term.  The candidate the arithmetic points at:
3.1 GB is close to a full 1024-class sinogram minus one shard, so a
placement or assembly step that briefly materializes a sinogram-shaped
array on the lead device fits the size.

The third question is a policy gap with no measurement needed.  Input 4
records a 2048-class forward projection that neither widened nor was
refused, and then died in the allocator.  The code explains it exactly:
`forward_project` and `back_project` never call
`_apply_device_policy`, so on a fresh automatic model they run on the
trivial single-device placement with no preflight.  Every
reconstruction entry gained that call in the entry-point campaign; the
bare projection entries were left out.  The fix is the same pattern
those entries use, plus a `'project'` workload for the ledger to price.

Input 3 is not a defect at all.  The 2048-class runs read 1.10 to 1.19,
inside the band, and the pass keeps them there: every stored measured
peak is re-priced under any changed model before anything lands, which
costs no GPU time because the ledger is closed form.

## 2. The probe (mg42a), one job

One job on four H100s answers the three measurement questions.  Every
arm runs in a fresh subprocess with the calibration mode on, so the
whole-run peak comes from the mode's own counter, and the per-phase
readings come from separate wrappers -- the whole-run row is never a
tail reading.  The residual check from the device-policy campaign
applies: whole-run peak and max-over-phases must agree within the
model's band, or the instrumentation is missing a region.

The arms:

* **The back-batch attribution (input 1).**  Parallel and cone at the
  1024-class, two devices, weighted, one full back projection
  instrumented per phase.  The reading wanted is the worker phase's
  transient minus its named blocks, which is the batch's true live
  set.  The cited estimate says three slab-equivalents of the four
  charged (`_parallel_back_view_batch_cost`'s 16 bytes per view and
  pixel).
* **The three-device band arms (input 2).**  Cone and parallel at the
  512-class, at one, two, and three devices.  The calibration table
  plus the ledger's per-device term breakdown say which term
  over-charges the three-device arms; the suspects that do not shrink
  with the shard are the batch charges, the transferred cylinders, and
  the fixed workspace.
* **The lead-device transient (input 5).**  Parallel and cone at the
  1024-class, four devices, in a fresh process, with the placement
  region (settle, sinogram and weights placement, partition build) and
  the reconstruction phases wrapped separately on device 0.  First
  verdict: does the 26.6 GiB reading reproduce outside the nightly's
  cumulative watermark?  If yes, the phase that carries it names the
  term to add.  If no, input 5 closes as a harness-semantics reading
  and the nightly's comparison method gets the note instead.

Cost: the arms are 3-iteration weighted reconstructions at the 512- and
1024-classes plus two full back projections, roughly one GPU-hour.

## 3. The term changes, gated by the probe

No charge moves before mg42a's rows are read.  The expected shape of
the changes, stated now so the increment is mechanical afterwards:

* **Rebalance the back pair.**  Set the batch charge to what the probe
  measures live (the kernel-side constant in
  `_parallel_back_view_batch_cost` and its cone twin), and in the same
  change re-examine `back_fixed`'s two-block count against the same
  rows, so the pair's sum still covers every measured peak.  The
  under-charge rule is absolute: if the rebalanced pair sits below any
  stored or fresh measured peak, the change does not land.  One caveat
  the ruling should know: the same 16-byte constant is also the
  driver's view-batch sizing rule, so trimming it would ALSO raise the
  production view batch.  The safe form is to leave the driver's
  batching number alone and correct only what the LEDGER prices for
  the batch; the proposal is that form, with the alternative (change
  both, re-gate performance) named for completeness.
* **The n=3 over-read fix follows the breakdown**, not a guess.  If the
  dominant over-charge is the back pair, the rebalance above may close
  input 2 by itself; the pass re-reads the three-device arms after it
  and only then decides whether any other term moves.
* **A new term only if input 5 reproduces**, sized by the probe's
  reading and charged on the lead device in the phase the probe names.

After any change: every stored measured row (`MEASURED_ARMS` in
tests/test_memory_ledger.py, the mg15 gate rows, the mg19 2048-class
rows) is re-priced closed-form against the new model, and the band must
hold on both sides everywhere.  One small confirm job (mg42b) re-runs
the calibration mode on the cells whose charges moved.

## 4. The projection-entry preflight (input 4), a code increment

`forward_project` and `back_project` gain
`self._apply_device_policy(workload='project')` as their first
statement, before `_shard_recon` or `_shard_sinogram` places anything.
The ledger gains a `'project'` workload: for the forward, the state
placement plus the existing initial-forward-projection phase terms
without the recon-loop residents; for the back, the existing back
phases without the filter.  The settle still sizes the layout for a
full reconstruction, exactly as the direct entries do, and the
workload-versus-sizing retry already in `_apply_device_policy` prices
the projection alone when no count fits a recon.  `workload_covers`
adds one pair: a layout checked for `'recon'` covers `'project'`;
nothing else is claimed.

The behavior change this ships: a bare projection on a fresh automatic
multi-device model now widens or is refused with the shortfall named,
where today it runs whole on the lead device and dies in the
allocator.  mg35's staging failure becomes the acceptance case: a
2048-class `forward_project` on the automatic branch must widen.  The
tests are CPU-side with synthetic budgets, like the existing preflight
tests, plus one line in mg42b's job asserting the 2048-class staging
case end to end.

## 5. What this pass does not do

The preprocessing entries' array-form gaps stay with their own item
(D2).  The denoiser's charges are untouched: no input names them.  The
capacity override, the floors, and the settle-once rule are untouched.
The nightly's own watermark semantics are out of scope except as
input 5's alternative verdict, in which case the note lands in the
nightly's comparison method rather than in the library.

## 6. The ruling

* **(a) Approve the probe** (mg42a, §2) as specified.
* **(b) Approve the gated term changes** (§3), including the safe form
  of the back-pair rebalance (ledger-side only; the driver's batching
  number stays).
* **(c) Approve the projection-entry preflight** (§4) with the
  `'project'` workload.
* **(d) Or direct otherwise.**  Nothing here is urgent: every input is
  an over-charge, a policy gap at a scale users have not yet hit, or a
  possible harness artifact, and the ledger's under-charge record is
  clean.

**RULED 2026-08-19 (Greg).**  (a) is moot: the probe ran the same day.
(b) is TABLED as not urgent.  The one under-read the probe found
(parallel at one device, 0.935) is absorbed by the preflight's 15
percent margin: a candidate is charged at 1.15 times its modeled peak,
so a model at 0.935 of the true peak still demands 1.075 times the
true peak, and no doomed layout is admitted through this gap.  The
over-reads cost nothing but headroom.  (c) is CLOSED WITHOUT ACTION:
no additional workloads join the ledger, so `forward_project` and
`back_project` keep their current behavior, and the explicit device
list remains the documented way to run a bare projection at a scale
one device cannot hold.  What stays live from this pass is outside the
ledger entirely: the nightly's multi-device memory columns
(findings §1.35 carries the state and the mechanism-independent
remedy).
