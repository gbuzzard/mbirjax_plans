# mbirtorch in the nightly, including multi-GPU — plan

**Status:** IMPLEMENTED and LIVE (2026-08-08).  Increments 1 through 5
and increment 7 have shipped; the schedule runs at 03:00 on four GPUs.
The findings are in §10 (the n=1 series) and §11 (the n>1 rows).  Only
increment 6, cpu-torch on the Mac, remains.

The charter is `current_plans.md` item 4.  Two of its sentences set the
scope: wire the torch writer into the nightly so mbirtorch gets the same
regression protection as mbirjax, and add n>1 rows under the established
memory-gate discipline.

Five records are prior art.  The open tail item and the dashboard wiring
are in `phase2_findings.md`.  The gate cells, the op set, and the
correctness tolerances are in `port_plan.md` §4.  The shared-sinogram
protocol and the arm-check discipline are in `phase5_findings.md`.  The
current composed baselines are in `kernel_batching_findings.md`.  The
device pin and the coming default flip are in `device_policy_design.md`.

**Terms.**  The JAX ENGINE means
`mbirjax_metrics/tooling/scaling_tests/performance_tracking.py`.  The
TORCH WRITER means
`mbirjax_metrics/tooling/scaling_tests/torch_backend_writer.py`.  The
JAX NIGHTLY means the running scheduled job named `mbirjax-nightly` on
gautschi, plus its launchd sibling on the Mac.  A PLATFORM KEY is the
`results/` subdirectory name: `gpu`, `cpu`, `gpu-torch`, `cpu-torch`.  A
CELL is one `(geometry, op, size, n_devices)` coordinate.

---

## 1. What this plan does

This plan wires mbirtorch into the nightly regression as a second,
independently scheduled job that writes into the platform keys the
dashboard already understands.  The jax nightly is not modified in any
behavioral way.  The torch writer is rebuilt so that its gate, its
record book, its filenames, and its op definitions are the jax engine's,
not private copies of them.

The investigation found three things that change the shape of the work
from what the charter assumes.  Each is stated in §2 with its evidence.
The design decisions the charter asks for are in §3.  The architecture
that follows from them is in §4, the file inventory in §5, the
deployment seam in §6, and the increments in §7.

---

## 2. Three findings from the investigation

### 2.1 The torch writer measures different operations than the jax ops of the same name

The torch writer emits cells named `forward`, `back`, and
`vcd_nonconst`, and the jax engine emits cells with those same names.
The operations behind the names differ.  The table gives the two
definitions side by side.

| cell op | jax engine | torch writer |
|---|---|---|
| `direct_filter` | `model.direct_filter(sino, output_sharded=True)` | same |
| `forward` | `model.sparse_forward_project(cylinders, pixel_indices)` | `model.forward_project(phantom, output_sharded=True)` |
| `back` | `model.sparse_back_project(sino, pixel_indices)` | `model.back_project(sino, output_sharded=True)` |
| `vcd_nonconst` | `model.vcd_recon(sino, partitions, partition_sequence, ...)`, partitions built outside the timed region | `model.recon(sino, weights, max_iterations=3, ...)`, which builds partitions and the direct-recon initialisation inside the timed region |
| `denoise` | seeded standard-normal image, partition seed 7 | seeded uniform image, partition seed 0 |

The inputs differ as well.  The jax engine's sinogram is a seeded
uniform random array, and its VCD weights are a seeded uniform draw on
`[0.5, 1.5]`.  The torch writer's sinogram is the forward projection of
a Shepp-Logan phantom, and its weights are a transmission-style
`exp(-sino / (2 max(sino)))`.

Four of the five ops therefore measure a different quantity on the two
backends.  The nightly's own gates do not care, because every gate is
history-based and compares a series against itself.  The dashboard does
care.  It places the torch History row directly below the jax row, and
it puts `gpu-jax` and `gpu-torch` in one Platform dropdown.  A reader
comparing those two rows under one op name is therefore comparing two
different operations.

The measured sizes of the gap support the concern.  The torch writer's
`back` at the 512 cell read 308 ms against the jax engine's 38.3 ms, and
its `forward` read 182 ms against 77.3 ms.  Those ratios do not
reproduce any ratio in `kernel_batching_findings.md`, whose composed
gate reads parallel 1.13x and 1.56x of jax.  The campaign harness
measured matched operations; the torch writer does not.

These findings indicate that the op definitions should be aligned before
history accumulates.  The cost of aligning them now is one discarded
run.  That run is the whole of the existing `results/gpu-torch/master/`
series, and it was measured on a dirty working tree two days before the
batching change landed.  The cost of aligning them later is a
re-baseline of however much history exists at that point.

### 2.2 The memory gate would be silently soft for torch

Memory is the only hard performance gate in the harness, and it is hard
only on GPU.  The rationale is recorded in `performance_tracking_plan.md`
§10: `peak_bytes_in_use` is deterministic on GPU, whereas CPU `mem_mb` is
coarse whole-process RSS.

The rule is implemented as a literal string comparison at
`performance_tracking.py:937-938`:

```python
bucket = hard if plat == "gpu" else soft
cpu_note = "" if plat == "gpu" else " [CPU RSS, coarse]"
```

A run filed under `gpu-torch` fails that test.  Its memory findings
would land in the soft bucket and carry the annotation
`[CPU RSS, coarse]`.  That annotation is false.  The torch writer reads
`torch.cuda.max_memory_allocated`, which is a device-side counter of the
same class as the jax one.  `port_plan.md` §3 names that counter as the
intended torch-side memory ruler.

So the letter of the code and the intent of the design disagree for the
new platform key.  §4.3 states the two-line change that resolves the
disagreement and the control that proves it is a no-op for `gpu` and
`cpu`.

### 2.3 The run filename carries the measurement time, not the commit time

The jax engine names each run file from `_file_tag`, which is the
library commit's UTC timestamp followed by its short SHA.  Three
behaviors depend on that choice.  A lexicographic sort of the filenames
is chronological by commit time, which is what makes `_find_priors`
return the immediately preceding commit's run.  Re-measuring one commit
overwrites its file rather than adding a second point.  The dashboard
places the run on the timeline at the commit's time.

The torch writer names its file from `datetime.now()` at line 224.
Re-measuring one mbirtorch commit would therefore add a new file every
night.  The timeline would grow duplicate points on one commit, and once
a real gate exists it would compare each run against the previous
night's run of the same code.

The fix is to call the engine's `_file_tag`.  It is small, and it must
happen before any nightly run lands, because the filenames are the time
series.

### 2.4 Three smaller findings, recorded here so the plan can carry them

The platform-mismatch guard is inert for torch platform keys.
`_assert_platform_matches_out_dir` selects the claimed platform with
`p in ("gpu", "cpu")`, and `gpu-torch` matches neither, so the function
returns without checking.  The torch writer needs its own guard; §3(f)
specifies it.

The vs-main correctness reference will stay dark unless the tracked
branch is literally named `main`.  The dashboard's analyzer tests
`r["branch"] == "main"` and keys the reference by exact platform string.
mbirtorch's remote carries `main`, `master`, and `greg_dev`, and its
`HEAD` points at `main`.  The existing `results/gpu-torch/master/`
directory is a leftover from the manual run.  Tracking `main` activates
the reference; tracking `master` does not.

The dashboard's default landing view selects the run with the newest
commit time across all platforms.  mbirtorch is under active
development, so a nightly torch series will usually own the newest
commit and the page will open on `gpu-torch`.  This is a real change to
how the dashboard behaves for a reader who opens it expecting the jax
view.  It is a display question, not a correctness one, and §9 lists it
as an open decision rather than assuming an answer.

---

## 3. The design decisions

### (a) Scheduling — a separate job, one GPU, staggered an hour after the jax nightly

**Options.**  Three arrangements were considered: extending
`run_regression.sh` with a torch step inside the existing job; adding a
second wrapper invoked sequentially by the same scrontab block; and a
separate scrontab block running a separate wrapper.

**Costs.**  The estimate below is for the §3(b) cell set on one H100,
built from the measured numbers available today.  The dominant term is
the eight VCD cells, whose warm times are the
`kernel_batching_findings.md` composed-gate numbers: parallel 1.87 s at
512 and 40.3 s at 1024, cone 2.79 s and 62.9 s.  Each VCD cell runs one
warmup plus one trial, so those four cells alone cost about 206 s.  The
remaining VCD cells and all single-shot cells add about 100 s.  Each
cell also pays a fixed cost for four things: torch import, CUDA
initialisation, model construction, and host input generation.  That
fixed cost is roughly 15 s at the small cells and 35 s at the 1024
cells, or about 11 minutes over 35 cells.  Adding a steady-state compile
allowance of 3 minutes gives about 20 minutes of measurement.  The
mbirtorch suite adds 5 to 10 minutes.

**Estimated added GPU-hours: about 0.5 per changed branch per night, on
a one-GPU allocation.**  Fire-on-change means a night on which no
tracked mbirtorch branch moved costs seconds.  With two tracked branches
both moving, the worst night is about 1.0 GPU-hour.

**Recommendation: a separate scrontab block, a separate wrapper, a
separate work directory, and one GPU.**  Four reasons, in order of
weight.

Failure isolation becomes structural rather than a matter of careful
guarding.  The charter requires that a torch failure not cost the jax
nightly its run.  Inside one job that requirement is met only by
trapping every torch failure mode, including the ones that leave the
GPUs in a bad state.  That failure mode is not hypothetical: the
2026-07-10 cascade in `run_regression.sh`'s comments records crashed
test workers leaving the GPUs unusable for the perf engine that
followed.  Two jobs cannot do that to each other.

The environments must be separate anyway.  The jax nightly's dedicated
env receives a per-branch `pip install -e "$WT[cuda13,test]"` on every
changed branch.  Installing mbirtorch into that same env would put torch
and jax's CUDA plugins in one resolver problem, and the per-branch
reinstall would churn both.  §3(f) makes the separate env the
recommendation on its own merits; a separate job follows naturally.

The allocation sizes differ.  The jax nightly requests four GPUs because
it sweeps n=1, 2, 4.  A torch n=1 sweep needs one.  Sharing the job
would charge the torch sweep at four GPUs for no benefit.

The alert identity is clearer.  A Slurm job named `mbirtorch-nightly`
with its own `--mail-type=FAIL` says which backend regressed before the
mail is opened.

**Cost of the recommendation.**  Some machinery is duplicated: a
wrapper, an env file, and the three schedule scripts.  §5 keeps the
duplication to the shell layer and shares every Python component.  The
two jobs must not share `WORK_DIR`, because `run_regression.sh` uses
`$WORK_DIR/.lock.d` as a single-instance lock and `$WORK_DIR/metrics` as
the clone it pushes from.  The torch wrapper gets its own.

**Schedule.**  The jax nightly wakes at `0 2 * * *`.  The torch nightly
should wake at `0 3 * * *`.  Slurm would allocate the two jobs
separately in any case, so the stagger is insurance against queueing
contention on the `ai` partition rather than a correctness requirement.

**The Mac.**  `phase2_findings.md` records the decision that the durable
cpu-torch series starts on the nightly's own CPU hardware.  A Mac trial
was created and then deliberately removed to hold that line.  The same
wrapper covers the Mac through the launchd path, exactly as the jax one
does.  §7 places cpu-torch after gpu-torch has soaked, because the Mac
adds no new design questions and every design question is easier to
answer with GPU data in hand.

### (b) The cell set and the ops

`port_plan.md` §4 already decided this, and the decision was made with
the metrics cells specifically so that the comparison instrument would
be the harness we trust.  This plan follows it rather than re-opening
it.

**Geometries: parallel, cone, denoiser.**  Cone is the notable addition
over the existing torch writer, which measures parallel and denoiser
only.  Cone is where the kernel campaign's numbers are closest to jax
(0.88x at 512 and 1.00x at 1024), so it is where a regression would show
first.  mbirtorch has no translation or multiaxis geometry, so those two
rows do not exist and their absence is not a gap.

**Ops: `direct_filter`, `forward`, `back`, `vcd_nonconst` for parallel
and cone; `denoise` for the denoiser.**  This is the harness op set.
The writer supports the denoiser cells today, and `port_plan.md` §4
records that they join once the denoiser ports, which it has.

**Sizes (GPU): (200, 208, 160), (512, 448, 384), (513, 449, 385), and
(1024, 1008, 992) at trials=1.**  Denoiser sizes: (225, 241, 257),
(512, 448, 384), (1024, 1008, 992).  These are the engine's own
`Config.sizes["gpu"]` and `Config.geom_sizes["denoiser"]["gpu"]`.  Using
them unchanged puts the torch and jax rows at identical coordinates.
Identical coordinates are what make the dashboard's shared axes honest,
and they are what `port_plan.md` §4 requires of the replacement gate.

**Sizes (CPU), when cpu-torch starts: (128, 112, 96), (129, 113, 97),
(200, 208, 160)**, denoiser (128, 144, 160) and (225, 241, 257).  The
(200, 208, 160) cell appears on both platforms deliberately.  It is the
cross-platform shared cell of design note D7, and within the torch
family it gives the `cpu-torch` to `gpu-torch` correctness check
something to compare.

**Cone recon shapes must be pinned.**  The jax engine pins cone
`recon_shape` per sinogram size in `CONE_RECON_SHAPE_PINS`.  The pin
exists so that a change to the library's axial-padding policy cannot
silently move a cell's memory and time baselines.  mbirtorch has its own
padding policy and will acquire its own reasons to change it.  A local
probe read mbirtorch's current cone auto shapes at the three smaller
cells as (160, 160, 208), (384, 384, 448), and (385, 385, 449).  Those
are the parallel shapes rather than the jax cone pins, so the torch
writer needs a pin table of its own.  The table is seeded from
mbirtorch's current auto shapes at all four GPU cells and all three CPU
cells, captured during the trial run.  Pinning does not move any
baseline.  It only decouples the baseline from a future policy change.

**n=1 first.**  The device-count question is (c).

### (c) Multi-GPU rows — n=2 and n=4, in a second increment, not waiting for item 3

**The recommendation: start n=2 and n=4 rows as soon as the device pin
is stable, on a reduced cell set, seeding their own history and
therefore ungated on their first nights.  Do not wait for item 3's
tuning.**

**Counts and cells.**  n ∈ {1, 2, 4} for parallel and cone at
(512, 448, 384) and (1024, 1008, 992), all four ops.  The two smaller
sizes stay n=1.  The denoiser stays n=1 at every size.

Three reasons for that shape.  The 512 and 1024 cells are the cells the
whole torch campaign gates on, so multi-device history there is directly
comparable to the campaign record.  The small cells at n>1 measure
mostly communication overhead and would add cost for little regression
signal.  The denoiser is excluded because `current_plans.md` §11 records
that `QGGMRFDenoiser.denoise` raises under any non-trivial placement,
and `device_policy_design.md` deliberately leaves the denoiser
single-device.  A row that is expected to fail teaches nothing until the
gap is closed, and §9 lists its return as a follow-up.

**Timing: after the device pin is stable, not after item 3.**  The
charter's overview says multi-GPU rows join once item 3 establishes
their baselines.  The investigation argues for the opposite order, and
the argument is worth stating plainly.

The nightly's value is regression protection during active development,
and items 2 and 3 are active development on exactly the multi-device
path.  Rows that exist during that work catch accidents while they
happen.  Rows that arrive after it starts from an already-tuned baseline
and protected nothing during the risky interval.

The cost of starting early is one night of seeding.  The gate is
history-based and vs-prior, so the first n>1 run has no reference and is
all-soft by the cold-start rule.  Item 3's tuning then appears as
whatever it turns out to be.  A memory increase fires the hard gate
once and auto-advances the baseline, which is the designed behavior for
an expected change.  A memory decrease, or a time change of either sign,
is not gated at all.

The real prerequisite is not item 3.  It is two things.  The first is
that `configure_devices` behaves the same before and after the item-2
flip, which §3(d) establishes.  The second is that the torch memory
ruler is characterised at n>1, which §3(c-ii) below makes an explicit
trial-run measurement.

**(c-ii) The rolling-min window must be measured, not inherited.**  The
`mem_gate_window = 3` rule exists because jax's `peak_bytes_in_use` on
the sharded path is bimodal per run.  A sporadic 30 to 100 MB scratch
transient rides on a stable floor there.  The ablation measured n=2
wandering about 12 percent while n=1 and n=4 were byte-frozen.  That
bimodality is a property of jax's ruler, and the torch ruler is a
different instrument.  `torch.cuda.max_memory_allocated` reports
allocated bytes rather than reserved bytes, which is a plausible reason
to expect it to be steadier.  That expectation is a hypothesis and not a
measurement.

The plan therefore treats the window as a swept parameter.  The trial
run repeats one n=2 and one n=4 cell five times in fresh subprocesses
and records the spread.  A spread inside a few tenths of a percent
justifies `MEM_GATE_WINDOW=1` for torch and a hard gate with no
detection lag.  A bimodal spread justifies 3, matching jax.  The number
goes in the torch run-knobs file with the measurement cited beside it.

**Cost of the n>1 increment.**  The added measurements are modest, but
the allocation is not.  A job that measures n=4 must request four GPUs,
and that request multiplies every other cell in the job.  The n=1 job
runs about 30 minutes on one GPU, or about 0.5 GPU-hours.  The same job
with n=2 and n=4 rows runs about 40 minutes on four GPUs, or about
**2.7 GPU-hours per changed branch per night**.  That fivefold increase
is the honest reason the increments are separate.  §9 raises the cadence
question it implies.

### (d) Device-count pinning — every row calls `configure_devices`, then asserts what it got

This is a hard requirement, and the investigation shows the current
writer does not meet it.  `torch_backend_writer.py` builds every model
with `device="cuda"` and never calls `configure_devices`.  Today that
yields one device, because mbirtorch's default placement is trivial and
nothing widens it.  After the item-2 flip, `device="cuda"` is an
unindexed CUDA device and therefore eligible for automatic widening.
Every existing torch row would then silently start measuring an
all-device run under a cell labelled n=1.

That is not a hypothetical class.  `device_policy_design.md` records the
same defect in `p4_gate_readout.py`, whose torch branch calls
`configure_devices` only when `n_dev > 1`.  It observes that the n=1 arm
is also the reference the value diffs are taken against, so the parity
columns would silently compare a widened run against itself.  Fable's
checkpoint-1 ruling extends the checkpoint-3 script audit to the
mbirjax_metrics harness scripts specifically.  This plan discharges that
obligation for the nightly.

**How each row pins.**  Three layers, in order of authority.

The per-model pin is `model.configure_devices(num_devices=n)`, called
immediately after construction and before any warmup.  Its first
statement sets `device_layout_is_automatic = False` permanently, which
is precisely the flag the automatic widening path will consult.  The
design states the contract directly: an explicit `configure_devices`
call switches the model out of automatic mode permanently, and
`num_devices=1` remains the reproducibility pin.  Order matters, because
`configure_devices` reads `sinogram_shape` and `recon_shape` at call
time and recreates the projectors, so it must follow any geometry
`set_params` and precede any timed call.

The per-process pin is `MBIRTORCH_NUM_DEVICES=<n>`, exported into each
cell worker's environment.  The variable does not exist in mbirtorch
today, and is checkpoint-3 work in the device-policy design.  That
design specifies that when the variable is set it pins the count exactly
as an explicit `configure_devices(num_devices=n)` would, and that the
count is never reduced below the pin.  Setting it now is harmless,
because an unread environment variable is inert.  It becomes a second
independent pin the moment the flip lands.

The assertion is the layer that makes the other two verifiable.  After
`configure_devices`, and again after the timed call, the worker reads
`model.sino_placement.devices` and raises unless its length equals the
intended `n`.  The row records the realized list.  This is the
`phase5_findings.md` arm-check discipline applied to device count: the
row verifies what was BOUND, never what was requested.

**What the row records.**  Each row carries `n_devices` (the intended
count), `n_shard_devices` (`len(model.sino_placement.devices)`),
`devices` (the string form of each device), and `is_sharded`
(`not model.sino_placement.is_trivial`).  The first two must agree or
the cell fails.  The existing writer hardcodes `is_sharded: False` and
`n_shard_devices: 1`, which would become a lie at n>1.

**One consequence to record.**  The memory preflight is
multi-device-CUDA-only and runs only on the automatic path, so an
explicitly pinned row never invokes it.  The nightly therefore does not
exercise the preflight.  That coverage lives in the unit tests and in
calibration mode, and this plan does not move it.

**One interaction to avoid.**  `MBIRTORCH_MEMORY_CALIBRATION` must be
unset in every cell worker.  Calibration mode calls
`reset_peak_memory_stats` at the top of `vcd_recon` and owns the peak
counter, which would clobber the measurement the row is taking.  The
worker asserts the variable is absent rather than trusting the ambient
environment.

### (e) Correctness gating — same-framework only; no cross-framework column in the nightly

**Recommendation: gate the torch series against its own history at the
established tolerances, and put no torch-versus-jax column in the
nightly.**

**The same-framework gate.**  The fingerprint form is already correct in
the torch writer: float64 reductions, twelve deterministic samples, and
exact shape and dtype.  The tolerances are the harness's own, and
`port_plan.md` §4 names them for exactly this purpose: relative 1e-5 for
the single-shot ops, and 1e-4 for the iterated ones.  One deviating
sample of the twelve is allowed before a soft flag.  Reaching them
requires only that the
torch writer stop hardcoding `gate: {result: pass}` and call the
engine's `gate_run` against its own prior run.  §4 covers the mechanism.

The `padding_zero` invariant is vacuous at n=1, because torch outputs
are unpadded.  It stops being vacuous at n>1, where a non-dividing
count pads the sharded axis.  The n>1 increment must crop to the true
shape and assert the overhang is exactly zero, exactly as the jax engine
does, so the fingerprint stays comparable across device counts.

**Which references are live for torch.**  Of the four references in the
correctness design, prior-run is live from night one.  Cross-device
activates with the n>1 increment and is the design's highest-value
check, because it catches a sharding bug that both device counts would
otherwise agree on being wrong about.  Cross-platform activates when
cpu-torch starts and compares torch to torch, guarded by the analyzer's
backend-family test.  Vs-main activates when the tracked branch is
named `main`, which §3(b) and §9 both depend on.

**The cross-framework question.**  The recommendation is no, for three
reasons.

The comparison would need a protocol the nightly does not have.
`phase5_findings.md` records that a cross-framework value gate must hand
one sinogram artifact to both frameworks.  Two measured reasons force
that rule: the per-framework phantom differs across frameworks at
boundary ties, and the jax phantom additionally differs across
platforms.  The repair moved the parallel 1024 residual from 0.375 to
6.1e-3.  A nightly column would have to produce and share that artifact
between two jobs, in two environments, on two schedules.

The comparison would fire on things that are not regressions.  The
residual that survives the shared-sinogram protocol is real and is
attributed: it is the compile-latitude term, measured at 6.1e-3 at three
iterations and decaying to 8.8e-4 by ten.  Separating that term from the
shared partition-order term is item 3's deferred value comparison.  A
nightly gate would fire on it nightly and teach nothing.

The coverage already exists in the right place.  mbirtorch's own
`test_vs_goldens.py` carries the cross-framework check, and the nightly
runs the mbirtorch suite, so a golden divergence surfaces as a test
failure on the torch row.  The dashboard's analyzer is already
family-guarded and will not compare `gpu` to `gpu-torch`.  The guard's
own comment says a cross-backend value comparison belongs to the port's
golden gates, not the analyzer.

The cheap answer and the right answer coincide.  Cross-framework
comparison is campaign work.

### (f) Environment — a nightly-owned env, with the env identity recorded and asserted in every row

**Options.**  Two: reuse `TORCHPY`, the hand-built cluster env at
`/scratch/gautschi/buzzard/torch_p0/env`; or create a dedicated
`mbirtorch_regression` env the harness owns and auto-creates, mirroring
the jax nightly's `mbirjax_regression`.

**Recommendation: a nightly-owned `mbirtorch_regression` env for the
scheduled job, and `TORCHPY` for the pre-schedule trial run only.**

`TORCHPY` is a development environment that Greg mutates between
campaigns.  A scheduled job that depends on it inherits every mutation
as an unattributed change to its own measurements.  That is exactly the
class of drift the run header's toolchain block exists to catch.  The
jax nightly's discipline is instead a dedicated env that the harness
creates and installs into.  The reasoning in `lib_env.sh` applies
unchanged to torch: a per-branch editable install into a shared dev env
would churn that env.

`TORCHPY` remains the right vehicle for the trial run, for three
reasons: it is reachable by per-file scp into the existing scratch
checkout; it already carries the cu130 torch build the node needs; and
using it keeps the trial from depending on the env-creation step that
has not been exercised yet.

**One difference from jax that the plan must carry.**  jax selects its
CUDA build through a pip extra, `jax[cuda13]`, which is why
`INSTALL_EXTRAS_gpu` exists and why `run_configs.env` warns that it must
match the node's CUDA module.  torch selects its CUDA build through the
wheel index, not through an extra.  mbirtorch's pyproject declares
`torch>=2.13` with no CUDA extra, so a plain `pip install -e "$WT[test]"`
resolves whatever the default index serves.  The torch run-knobs file
therefore needs a `TORCH_INDEX_URL` knob, defaulted to the pytorch cu130
index to match the existing `2.13.0+cu130` build, with the same
keep-in-sync-with-the-node warning the jax file carries.

**How the env's identity is recorded in each row.**  The run header
carries a toolchain block with six fields: `torch`, the full build
string including the CUDA tag; `torch_cuda`, from `torch.version.cuda`;
`python`; `triton`; the interpreter path; and the `LOADEDMODULES` signal
from the node preamble.  The header also carries
`installed_packages()`, which the jax engine already provides and which
is backend-agnostic, so a night-to-night drift can be attributed to the
specific package that moved.

**The platform-mismatch guard.**  This is the equivalent of the guard
that exists because the GPU nightly once measured on CPU and filed the
results under `gpu`.  The torch writer's version asserts three things
before writing any file, and aborts loudly on any failure.

The intended platform is declared by the wrapper, not inferred.  The
wrapper exports `REG_TORCH_PLATFORM=gpu-torch` or `cpu-torch`, and the
writer raises if `torch.cuda.is_available()` disagrees with the
declaration.  Inferring the key from availability, as the current writer
does, cannot fail loudly.  A GPU night on which CUDA did not initialise
would simply file itself under `cpu-torch`, and the gpu-torch charts
would go quiet with no other symptom.  That is the 2026-07-21 failure
reproduced exactly.

The device name and count are recorded and checked.  The header carries
`torch.cuda.get_device_name(0)` and `torch.cuda.device_count()`, and the
writer raises on a GPU platform if the count is zero.

Every row's realized device list is checked against its intended count,
per §3(d).

### (g) Failure reporting — the dashboard, plus a Slurm mail with its own identity

The dashboard's existing machinery covers the torch rows with no change,
once the writer emits a real gate block.  The investigation confirmed
this end to end.  The hard-gate cell regex parses `geom|op|size|ndev` and
is unaffected by a hyphen in the platform key, because the platform
appears only inside the basis prefix that is stripped first.  The
severity split classifies a hard string as correctness or performance by
its text, and the strings come from the engine's own
`_gate_fingerprint`.  The correctness banner, the browser-tab badge, the
History gate panel, and the red markers on the scaling plots all iterate
platforms generically.

Two payload fields must be emitted or two surfaces degrade.  The gate
block needs `compared_to`, or the correctness basis label falls back to
the literal string "prior run" instead of resolving to the run it
compared against.  The config block needs the five gate-threshold keys
`mem_hard_pct`, `speedup_warn_pct`, `time_soft_pct`, `fp_rtol_single`,
and `fp_rtol_iter`, or the gate-threshold explanation renders as `?%`
whenever a torch run is the anchor.  Both come for free if the writer
builds a real `pt.Config` and serializes it, which §4 recommends for
independent reasons.

Beyond the dashboard, the torch wrapper reuses the jax wrapper's alert
mail path, addressed from a job named `mbirtorch-nightly` so the subject
line names the backend.  The wrapper's exit code tracks hard-gate
failures only, matching the jax convention that a test failure emails
its detail without flipping the exit code.

One surface needs a small fix rather than nothing.  `recent_runs.py`
prints the platform into a four-character field, so `Gpu-torch` shifts
every column after it on torch rows.  Widening the field is a one-line,
display-only change.

---

## 4. Architecture — one gate model, two measurement layers

### 4.1 The shape

The torch writer becomes a torch measurement layer over the jax engine's
decision layer.  It imports the engine and calls its pure-dictionary
functions for everything that is not measurement: `Config`, `gate_run`,
`update_records`, `_find_priors`, `_apply_mem_window`, `_cell_key`,
`_expected_cells`, `_file_tag`, and `regression_to_table.write_table`.
It keeps its own model construction, its own op bodies, its own timing
loop, and its own memory read.

The split is possible because the engine's decision layer never touches
jax.  `scaling_common` imports numpy, ruamel, and matplotlib at module
level and defers every jax import into the functions that need it.  So
`import performance_tracking` from a torch environment initialises no
jax backend.  The torch env must carry matplotlib and ruamel, which is
one line in the torch run-knobs file's `HARNESS_DEPS`.

The split is also the whole point.  A second private gate implementation
would drift from the first, and the drift would be invisible until a
torch regression was silently not caught.  One gate model means the
memory threshold, the fingerprint tolerances, the status-transition
rules, the cold-start rule, and the rolling-min window are defined once.

### 4.2 What the torch layer supplies

The torch layer supplies six things.  It builds an mbirtorch model of a
given geometry at a given sinogram size, with the cone `recon_shape`
pinned.  It pins and verifies the device count per §3(d).  It builds the
op inputs with the jax engine's own generators, which are pure numpy and
therefore reusable verbatim: `make_sinogram`, `make_cylinders`,
`make_weights`, `make_noisy_image`, and the seeds beside them.  It binds
each op to the mbirtorch call that matches the jax definition in the
left-hand column of the §2.1 table; the bindings are tabulated below.
It times with
`torch.cuda.synchronize` in place of `jax.block_until_ready`, dropping
the previous result before the next allocation exactly as `sc.time_op`
does.  It reads peak memory as the maximum of
`torch.cuda.max_memory_allocated(d)` over the row's pinned devices,
after a `reset_peak_memory_stats(d)` on each of them before the timed
region.

The aligned op bindings are these.  Every one of the five mbirtorch
names exists and mirrors its mbirjax counterpart, so the alignment costs
no new library surface.  Two signatures differ in a way the
implementation must absorb: mbirtorch's `initialize_recon` and `denoise`
take no `print_logs` argument, while the jax engine passes one to both.

| cell op | mbirtorch call | input |
|---|---|---|
| `direct_filter` | `model.direct_filter(sino, output_sharded=True)` | `make_sinogram` |
| `forward` | `model.sparse_forward_project(cylinders, pixel_indices)` | `make_cylinders`, `gen_full_indices` |
| `back` | `model.sparse_back_project(sino, pixel_indices)` | `make_sinogram`, `gen_full_indices` |
| `vcd_nonconst` | `model.vcd_recon(sino, partitions, partition_sequence, stop_threshold_change_pct=0.0, weights=w, init_recon=None)` | `make_sinogram`, `make_weights`, partitions from `initialize_recon` under `np.random.seed(measure_seed)` |
| `denoise` | `model.denoise(image, sigma_noise=0.1, max_iterations=20, stop_threshold_change_pct=0.0, output_sharded=True)` | `make_noisy_image`, `np.random.seed(measure_seed)` |

The isolated-subprocess-per-cell discipline is kept, for the reason the
plan gives: a peak counter is a high-water mark, so it is only a ruler
inside a fresh process.

`sc.sample_gpu_health`, `sc.throttled_gpus`, `sc.annotate_speedups`,
`sc.size_label`, `sc.is_oom`, `sc.save_yaml`, and `sc.load_yaml` are
nvidia-smi or pure-python and are reused unchanged, so the thermal flag
and the OOM-descent classification behave identically on both backends.

### 4.3 The one change to the shared jax engine

The memory gate's platform test becomes a named predicate:

```python
def _memory_is_device_peak(plat):
    """True when mem_mb is a per-device accelerator peak, False when it is
    coarse whole-process RSS.  'gpu' is jax reading peak_bytes_in_use;
    'gpu-torch' is torch reading max_memory_allocated -- the same class of
    deterministic device-side counter, and the ruler port_plan.md §3 names.
    Everything else ('cpu', 'cpu-torch') is process RSS."""
    return plat in ("gpu", "gpu-torch")
```

The two call sites at `performance_tracking.py:937-938` use it in place
of `plat == "gpu"`.

**The pure control.**  For `plat` in `("gpu", "cpu")` the predicate
returns exactly what the string comparison returned, so the jax path is
unchanged by construction.  The control turns that argument into a
measurement.  Before and after the edit, run `gate_run` over two
consecutive committed jax GPU runs and over two consecutive CPU runs
from `results/`.  Then diff the resulting gate dictionaries.  The diff
must be empty.  This runs on the Mac in seconds and needs no cluster.

No other change is made to `performance_tracking.py`, to
`run_regression.sh`, to `regression.env`, to `run_configs.env`, to
`lib_env.sh`, or to the three jax scheduling scripts.

### 4.4 What the torch writer stops doing

It stops hardcoding the gate.  It stops naming files by measurement
time.  It stops inferring its platform key from hardware availability.
It stops writing `sizes` keyed by platform family, and writes them keyed
by the full platform key, so `_expected_cells` resolves the sweep it was
supposed to attempt.  It stops omitting the record book and the
`_table.yaml` companion.

---

## 5. File inventory

### mbirjax_metrics

| file | change |
|---|---|
| `tooling/scaling_tests/torch_backend_writer.py` | rewritten per §4 |
| `tooling/scaling_tests/performance_tracking.py` | the two-line predicate of §4.3, with the pure control |
| `tooling/regression/run_torch_regression.sh` | NEW — the torch wrapper, sibling of `run_regression.sh` |
| `tooling/regression/torch_regression.env` | NEW — torch infrastructure config |
| `tooling/regression/lib_torch_env.sh` | NEW — torch env creation and install, sibling of `lib_env.sh` |
| `tooling/regression/enable_torch_nightly.sh` | NEW — installs the `mbirtorch-nightly` scrontab block or launchd agent |
| `tooling/regression/disable_torch_nightly.sh` | NEW |
| `tooling/regression/status_torch_nightly.sh` | NEW |
| `tooling/regression/nightly_torch_regression.slurm` | NEW — manual batch test |
| `tooling/regression/com.mbirtorch.regression.plist` | NEW — launchd template for the Mac |
| `tooling/regression/recent_runs.py` | widen the platform column (display only) |
| `action_scripts/torch_run_configs.env` | NEW — the torch run knobs |
| `action_scripts/run_one_torch_night.sh` | NEW — thin wrapper |
| `action_scripts/enable_torch_nightly.sh`, `disable_torch_nightly.sh`, `status_torch_nightly.sh` | NEW — thin wrappers |
| `tooling/regression/README.md` | a torch section |
| `action_scripts/README.md` | the torch entry points and knobs |
| `results/gpu-torch/master/` | proposed for removal, pending Greg's call; see §9 |

### mbirtorch

No changes.  The device pin uses the existing `configure_devices`, the
test step uses the existing `dev_scripts/run_tests.sh`, and the
`MBIRTORCH_NUM_DEVICES` pin is inert until the device-policy work lands
it.  This is deliberate: it keeps the deployment seam entirely inside
mbirjax_metrics.

### mbirjax_plans

| file | change |
|---|---|
| `plans/torch_port/nightly_plan.md` | this file, plus the findings and closing sections at the second STOP |
| `plans/current_plans.md` | item 4 state, at the second STOP |
| `plans/experiments/torch_port/nt1_gate_control.py` | NEW — the §4.3 pure control, re-runnable |
| `plans/experiments/torch_port/nt1_trial.sbatch` | NEW — the trial-run submission script |

---

## 6. The deployment seam

The production nightly runs from a clone it refreshes itself.  Phase 1
of the wrapper updates `$WORK_DIR/metrics` by `git pull --rebase` from
origin, then re-execs that copy.  Harness code therefore reaches
production only after Greg commits and pushes to `mbirjax_metrics` main.

The live installation was read on 2026-08-07 and is this:

```
# mbirjax-nightly-BEGIN
#SCRON -A bouman -p ai -q normal -N1 --gpus-per-node=4 -n 56 -t 04:00:00 \
       -J mbirjax-nightly --mail-user=buzzard@purdue.edu --mail-type=FAIL \
       -o /home/buzzard/.mbirjax/regression/nightly-%j.log
0 2 * * * bash /home/buzzard/PycharmProjects/mbirjax_metrics/tooling/regression/run_regression.sh
# mbirjax-nightly-END
```

Two checkouts matter, and they are at different ages.  The scrontab's
entrypoint is Greg's own working checkout,
`/home/buzzard/PycharmProjects/mbirjax_metrics`, sitting on `main` at
`7965e83` from 2026-07-12.  The clone the wrapper re-execs,
`/home/buzzard/.mbirjax/regression/metrics`, is on `main` at `8efb299`,
the current origin tip.  So the nightly measures with current code
despite a month-stale entrypoint, which is the bootstrap-then-re-exec
design working exactly as intended.

**That staleness is an oversight to fix, not a constraint to design
around, and pulling it is provably safe.**  The entrypoint executes only
phase 1 before it `exec`s the fresh clone, and phase 1 is
BYTE-IDENTICAL between `7965e83` and current `main` — 46 lines, no diff.
Every configuration variable phase 1 reads is unchanged too: `ENABLED`,
`WORK_DIR`, `METRICS_URL`, `MBIRJAX_URL`, `PREAMBLE_FILE`, `TOKEN_FILE`.
The 86 lines that do differ in `run_regression.sh` are all in phase 2,
which the entrypoint never reaches, and which already runs every night
from the `$WORK_DIR` clone.  A pull therefore changes nothing about what
executes.  The working tree is clean apart from two untracked output
directories, so the pull has nothing to conflict with.

**So the torch nightly uses the same standing-checkout convention as the
jax one.**  `enable_torch_nightly.sh` points the `mbirtorch-nightly`
block at
`/home/buzzard/PycharmProjects/mbirjax_metrics/tooling/regression/run_torch_regression.sh`,
exactly as the jax block points at its sibling.  One convention, one
place to look.  An earlier draft of this plan invented a second standing
checkout under scratch to avoid the pull; that was unnecessary
complexity built on an unchecked assumption about where the 86 lines
were.

Three more facts from the same reconnaissance shape the trial run.
`TORCHPY` is an isolated venv over the `mbirjax` conda env, with
`include-system-site-packages = false`, so it does not inherit that
env's `ruamel.yaml`; the trial must install the harness deps into it.
It does already carry torch 2.13.0, triton 3.7.1, and matplotlib 3.11.1.
And `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src` is a plain
scp'd directory rather than a git checkout, so the trial must clone
mbirtorch from origin rather than measure that tree; cloning is the real
nightly path anyway, and it is the only way the run gets true provenance.

This gives a clean division of what can be trialed and what must wait.

**Trialable by per-file scp, with no schedule change and no production
impact:** the rewritten torch writer, the torch wrapper, the torch env
files, and the trial sbatch script.  They go into a scratch checkout,
not into the production clone, and the trial submits them directly with
`sbatch`.  The trial exports `REG_TORCH_NO_PUSH=1` so it writes its
results locally and pushes nothing.  Every scp'd file is verified by
md5, per the standing sync rule, because a partial rsync has silently
dropped files before.

**Verifiable on the Mac, with no cluster at all:** the pure control of
§4.3, and a `REG_SMOKE`-equivalent plumbing pass of the torch wrapper
against a one-cell sweep.

**Landing only after Greg commits and pushes:** everything, before the
schedule is installed.  The order is fixed and is the last increment's
whole content.  Greg commits and pushes the staged files.  The standing
cluster checkout is pulled.  `enable_torch_nightly.sh` is run once.
`status_torch_nightly.sh` confirms both layers.  The production nightly
never runs uncommitted code, because the wrapper it runs came from
origin.

**One consequence to state.**  The scrontab block is installed by a
script that must run on the cluster.  That is reachable: the cluster
account is `buzzard`, not the local Mac account name, and key auth works
with the default `~/.ssh/id_rsa` and no `-i` flag.  The conventions live
in `mbirjax_plans/.claude/cluster_use.md`, which is the file to read
before any cluster work on this project.

---

## 7. Implementation increments

Each increment ends in a verifiable state.  The trial-run gate sits
between increment 4 and increment 5, and no schedule change happens
before it passes.

**Increment 1 — the shared-engine change and its control.**  Add
`_memory_is_device_peak` and use it at the two call sites.  Run the pure
control of §4.3 on the Mac against committed jax runs from both
platforms.  Stage only when the gate-dictionary diff is empty.

**Increment 2 — the torch writer rewrite.**  Eight changes: build a real
`pt.Config`; align the five op definitions; pin and verify device
counts; pin cone recon shapes; add the platform guard; name files with
`_file_tag`; call `gate_run` and `update_records`; and write the
`_table.yaml` companion.
Verify locally on the Mac in a CPU-only, one-cell configuration: a run
file appears with a real gate block, a second run of the same commit
overwrites rather than adds, a third run against a deliberately
perturbed prior fires the expected hard finding, and
`build_dashboard.py` parses all of it.

**Increment 3 — the wrapper and the schedule scripts.**  Write
`run_torch_regression.sh` and its env files, modelled on the jax wrapper
and keeping its structure: the bootstrap re-exec, the single-instance
lock, fire-on-change via `git ls-remote` against mbirtorch, the
throwaway shallow clone per changed branch, the test step, the perf
step, the rebase-retry push, and the alert mail.  Verify with the
smoke path on the Mac.

**Increment 4 — the trial-run package.**  Write the trial sbatch script.
scp every changed file into a scratch staging dir with md5 verification
per file.  Install `ruamel.yaml` into `TORCHPY`'s venv, which is
isolated and lacks it.  Pass all environment through the submitting
shell, never through `--export`, because Slurm splits exported values on
commas.  Honor the two standing cluster rules: one GPU with
`--cpus-per-task=14`, and no `--mem` on the `ai` partition.

**TRIAL-RUN GATE.**  One real end-to-end run on gautschi, on one H100,
with `REG_TORCH_NO_PUSH=1`.  It must produce all of the following before
any schedule change.

A complete `results/gpu-torch/main/` run file with every §3(b) cell
measured, a real gate block reading cold-start (all-soft), a records
file, a `_table.yaml` companion, and a tests file.

A measured wall time and GPU-hour figure, replacing the §3(a) estimate.

The device-pin assertion exercised: every row's realized device list has
length one, and a deliberately mispinned control row raises rather than
measuring silently.

The platform guard exercised: a control invocation declaring
`gpu-torch` on a CPU-only allocation aborts.

The cone recon shapes captured at all four GPU cells, for the pin table.

The memory-determinism ablation of §3(c-ii): one cell repeated five
times in fresh subprocesses, with the spread recorded.

A second run against the first, to confirm the vs-prior gate fires
correctly on a real pair rather than on a cold start.

**Increment 5 — the gpu-torch schedule change.**  Greg commits and
pushes.  The standing checkout at `~/PycharmProjects/mbirjax_metrics` is
pulled, which §6 shows changes nothing about what the jax nightly
executes.  `enable_torch_nightly.sh` installs the `mbirtorch-nightly`
block.  `status_torch_nightly.sh` confirms both layers.  The jax nightly
is then re-verified as untouched: `scrontab -l` still shows its own
block byte-for-byte, and its `ENABLED` flag is unchanged.

**Increment 6 — cpu-torch on the Mac.**  Same wrapper, launchd path, CPU
cell set, and the same commit-then-enable order as increment 5.  This is
where the `cpu-torch` to `gpu-torch` cross-platform correctness
reference comes alive.

**Increment 7 — the n>1 rows.**  Three preconditions gate it: the
device-policy flip has landed; the §3(c-ii) window measurement is in
hand; and the n=1 series has soaked for at least three nights, so a
rolling window has a history to work with.  The increment raises the
torch job's allocation to four GPUs and adds the reduced n ∈ {2, 4} cell
set of §3(c).  Its own trial run repeats the device-pin assertion at n=2
and n=4, and adds the padding-zero check.

---

## 8. What could break the running nightly, and why it will not

The charter's standing rule is that the jax nightly must not break.
Four mechanisms keep it safe, and each is stated as a claim that can be
checked.

Nothing the jax nightly executes is modified except two lines whose
behavior is provably identical for `gpu` and `cpu`, with a diff-the-gate
control that proves it.

The two jobs share no mutable state.  They use different work
directories, different locks, different persistent metrics clones,
different conda environments, and different scrontab blocks.  They write
disjoint paths under `results/`, which is the same property that already
lets the Mac and the cluster push concurrently.

The push contention is already solved.  Both wrappers commit only their
own platform's paths and both use the same rebase-and-retry loop, and a
failed push is non-fatal and self-heals because the next run re-measures.

The dashboard build is additive.  Platform discovery is a glob over
`results/`, the `_table.yaml` exclusion already covers the new key, and
the gate and correctness machinery iterate platforms generically.  The
one behavior that does change is the default landing view, which §9
raises as a decision rather than a side effect.

---

## 9. Open questions

**Which mbirtorch branches to track.**  The recommendation is `main`
alone to start, because it activates the vs-main correctness reference
and because a second tracked branch doubles the worst-night cost.
`greg_dev` is the natural second once the series is established.
`master` should not be tracked; it is behind `main`, and the dashboard's
vs-main reference tests the branch name literally.

**Whether to remove the existing `results/gpu-torch/master/` run.**  The
recommendation is to remove it.  It was measured on a dirty tree, with
op definitions that §2.1 changes, two days before the batching change
landed, under a branch name that will not be tracked.  Keeping it would
put one incomparable point at the head of the new series.  Removing
results is a deletion of committed data, so it is Greg's call and not
something this plan does unilaterally.

**The dashboard's default landing platform.**  A nightly torch series
will usually own the newest commit, so the page will open on
`gpu-torch`.  Three options: accept it, prefer the jax family in the
default selection, or remember the reader's last choice.  This is a
one-line change in `dashboard.js` either way and is display-only.

**The n>1 cadence.**  The n>1 increment takes the torch job from about
0.5 to about 2.7 GPU-hours on a changed-branch night, because the
allocation must grow to four GPUs.  The options are nightly at that
cost, or n=1 nightly with a separate weekly four-GPU job for n>1.  The
recommendation is nightly, for two reasons.  A weekly gate gives a
week-wide window in which a multi-device regression can land and be
forgotten.  Fire-on-change means the cost is paid only on nights when
mbirtorch actually moved.  Greg's read on the GPU-hour budget should
decide it.

**A torch dependency canary, deferred.**  mbirtorch's pyproject already
anticipates one, in the comment beside its `torch>=2.13` floor:
re-test on each torch bump via the metrics harness backend cells before
advancing it.  The jax analog is the `DEP_CANARY_ENABLED` machinery and
the `JAX_LAST_REVIEWED` watermark, which together re-measure a fixed
commit when a new jax ships.  The recommendation is to defer the torch
version and record the reason.  A dep-step is only readable against
established history, and the torch series has none.  The plan does
carry the cheap half now: the run header records the full torch build
string, so a torch bump is at least attributable after the fact.

**The denoiser at n>1.**  Excluded until the sharding gap in
`current_plans.md` §11 is closed or the denoiser is formally scoped to
one device.  Whichever way that resolves, the nightly row follows it.

**The installed scrontab walltime does not match the configured one.**
Found while reading the live schedule, and separate from this plan.  The
installed `mbirjax-nightly` block carries `-t 04:00:00`, while
`run_configs.env` has said `SLURM_WALLTIME="06:00:00"` since at least
2026-07-12.  So the block was written by an `enable_nightly.sh` run that
predates the current value, and the live ceiling is 4 hours rather than
the intended 6.  `enable_nightly.sh` itself is unchanged since then, so
re-running it after the §7 increment-5 pull rewrites the block from the
current knobs and closes the gap.  This matters only on a worst-case
all-branches-changed night, which is exactly the night the 6 hours was
chosen for.

---

## 10. Findings and closing (2026-08-08)

The plan is implemented through the first real scheduled-path run.  The
gpu-torch series now exists at origin: commit `ee1b249` carries
`results/gpu-torch/main/` with 35 measured cells at mbirtorch `ae9bb6f9`,
a records book, a `_table.yaml` companion, a tests file, and
`state/gpu-torch/main`.  The dashboard renders the series with the
GPU-TORCH tile column live and the torch History row below the jax row.
The jax nightly ran untouched through the whole change; its own
2026-08-08 gpu run (`e37bc93`) landed mid-implementation.

### 10.1 What shipped, against the increments

Increments 1 and 2 were committed by the prior session and re-verified
here.  The gate-control re-run at HEAD reproduced the recorded
post-edit state exactly: 16 cases, 523 hard, 504 soft, with FORCED
gpu-torch at 168 hard and 0 soft.  The writer's local verification
passed all four checks: a cold-start run file with a real gate block, a
same-commit rerun that overwrites, a perturbed prior that fires a hard
fingerprint finding with exit 1, and a clean `build_dashboard.py` parse.

The writer then needed one adaptation the plan did not anticipate.  The
device-policy flip landed in mbirtorch (`aa9644b`) nine hours after the
writer was committed, and it removed the constructor `device` argument
the writer passed.  The adaptation has three parts: models are built
with no device argument, every pin goes through `configure_devices`, and
the pin is platform-aware.  Platform-aware means cpu-torch pins
`devices=['cpu']` explicitly.  The count-only pin would bind MPS on a
Mac, because the lazy device preference is cuda, then mps, then cpu — so
a cpu-torch row would silently measure Apple's GPU.  Each row now also
verifies the KIND of every realized device, not just the count.

Increment 3 shipped as planned: the wrapper, the two env files,
`lib_torch_env.sh`, the three schedule scripts on both platforms, the
manual slurm file, the launchd template, the four `action_scripts` entry
points, the `recent_runs.py` column widening, and README sections.  The
kickoff decisions are applied: `main` is the sole tracked branch, the
incomparable `results/gpu-torch/master/` run is deleted, and the
dashboard's default landing view prefers the newest JAX-family run, with
torch one Platform-dropdown click away.

### 10.2 The trial-run gate, item by item

The gate ran as job 14987177 on one H100: 30m21s wall, exit 0, all
steps passing.  Its evidence, against the §7 checklist:

* **A complete run file.**  Two, in fact: the trial measured the real
  pair `aa9644b` (writer-direct) then `ae9bb6f9` (through the wrapper),
  35 of 35 cells each, zero failures, with records, `_table.yaml`, and a
  tests file beside them.
* **A measured wall time.**  A full changed-branch pass is 15m48s:
  install 8 s, suite 4m37s, sweep 11m01s.  That is about 0.26 GPU-hours
  per changed night, half the §3(a) estimate.
* **The device-pin assertion.**  Every row's realized list has length
  one on a CUDA device, and the mispinned control raised
  `configure_devices(2) needs 2 CUDA devices; found 1` rather than
  measuring.
* **The platform guard.**  Declaring gpu-torch with CUDA hidden aborted
  with `PLATFORM MISMATCH` at exit 2.
* **The cone recon shapes.**  mbirtorch's auto shapes at all four GPU
  cells match the writer's pin table exactly, through (1024, 1008, 992)
  → (992, 992, 1008).
* **The §3(c-ii) memory ablation.**  Five fresh-subprocess repeats of
  parallel/vcd_nonconst at 512×448×384 read 1974.6 MB every time: a
  0.000% spread.  `max_memory_allocated` is deterministic at n=1, so
  `TORCH_MEM_GATE_WINDOW=1` is the measured setting.  The jax window of
  3 exists for a jax artefact the torch ruler does not share.  The n>1
  increment must repeat this ablation at n=2 and n=4 before trusting 1
  there.
* **A second run against the first.**  The wrapper's run gated the tip
  against the prior with `GATE: PASS`, zero hard and zero soft findings.
  The tip-vs-prev diff is the forward-kernel repair, which touches only
  non-trivial placements, so a clean n=1 pair is the expected reading.
  The clean pair also shows that day-over-day noise sits inside the
  tolerances.

The trial's seed baselines for the headline VCD cells: parallel 1.51 s
and 36.4 s at 512 and 1024, cone 2.40 s and 57.8 s, with 23.2 to
23.7 GB peaks at the 1024 cells.  These sit below the campaign's
composed-gate numbers by construction, because the nightly times
`vcd_recon` with initialization outside the measured region, while the
campaign's gate timed whole reconstructions.

### 10.3 The first scheduled-path run

The first real run was job 14989680, triggered through
`run_one_torch_night.sh --sbatch` from the standing checkout after its
pull from `7965e83` to the current tip.  The run exercised the entire
production path on its own: the phase-1 bootstrap re-exec, the one-time
auto-creation of the `mbirtorch_regression` env, fire-on-change reading
`main @ ae9bb6f9: CHANGED (was none)`, the suite, the sweep, the
cold-start gate, and the rebase-push that produced `ee1b249`.  Wall
time was 20m58s including the env creation.

The run surfaced one provenance defect.  The wrapper wrote its install
log inside the library clone, and mbirtorch's `.gitignore` has no
`*.log` rule, so `git_provenance` read the pristine origin tip as
`git_dirty: true` and the dashboard stamped the row "dirty".  mbirjax
never hit this because its `.gitignore` covers `*.log`.  The fix
(`3ada7ba`) moves the log outside the clone, which also lets a FAILED
install's log survive the clone's deletion.  A forced re-measure (job
14991549) reseeded the same commit with clean provenance.

### 10.4 The goldens gap: the suite's cross-framework check skips in the nightly

`tests/goldens/` is gitignored in mbirtorch, so a fresh clone has no
goldens and `test_vs_goldens` skips.  The nightly runs from fresh
clones, so the suite's torch-vs-jax value check — the §3(e) reason this
plan put no cross-framework column in the nightly — is inert on the
nightly path.  The cluster suite read 396 passed, 99 skipped, and the
skips include both golden families.  The gap does not affect the gate,
and the check still runs in dev checkouts, where the goldens exist.
Three options, in rising cost: accept dev-checkout-plus-campaign
coverage as sufficient; commit the goldens to mbirtorch so clones carry
them; or have the nightly generate goldens, which would put jax inside
the torch env against the §3(f) separation.  This is Greg's call, and
the item-3 session's deferred value comparison bears on it.

**Resolution (2026-08-08, Greg + Fable): the goldens are development and
release instruments, not nightly ones.**  The nightly's question — did
tonight's commit change behavior — is answered by its own history and
vs-main references, the same machinery that replaced goldens in mbirjax.
The port-fidelity question is asked at development time and at releases,
where the archives exist.  The goldens are furthermore OPT-IN everywhere (Greg, same day,
superseding the first form of this ruling): mbirjax development is
expected to stop in favor of mbirtorch, intentional
non-backward-compatible changes will follow, and goldens that sit in
every run's path would then be an impediment.  Every archive-consuming
test carries a `goldens` pytest marker, and mbirtorch's pyproject
`addopts` deselects the marker by default (landed 2026-08-08).  A plain
`pytest tests` therefore runs the self-contained suite on every
machine, including the nightly's fresh clones, so this plan's owner has
NO wiring step.  The fresh-clone skip set is hardware gates alone, and
any new skip is signal.  Opting in is a run-time flag: `-m goldens` for
the parity tests alone, `-m "goldens or not goldens"` for the full
suite, or `RUN_GOLDENS=1` for dev_scripts/run_tests.sh.  Porting
charters (the Lilly gate, item 6's translation parity) and the release
gate opt in explicitly.  One correction to the count above: the marker covers 79 tests,
not 65 — the 65 missed the `golden_*.npz` consumers in `test_cone`,
`test_denoiser`, and `test_phantom`.  The archives stay gitignored and
regenerate on announcement, as now.  The release workflow
(current_plans item 7) carries the backstop: a release candidate passes
the full suite, goldens included, in a provisioned dev environment.

### 10.5 Smaller observations

The H100 flags heavy torch kernels as throttled.  Two of the trial's 35
rows carry the flag at `sw_power_cap` and 45°C, which is the boost
governor, not thermal distress.  The production night sampled up to
87°C during the 1024 VCD cells and flagged them; history-based gating
absorbs the noise, and the flag is informational on the tiles.

Row records show the device as unindexed `cuda` rather than `cuda:0`.
The unindexed form is what `configure_devices(num_devices=1)` binds at
n=1; the kind check accepts it and the measurement reads the current
device.  Cosmetic, and worth normalizing if it ever matters.

An sbatch shell has no `module` function, so the preamble's `module
load` lines fail in the torch job's log.  The torch stack does not
need them: the cu130 wheels bundle their CUDA runtime, conda resolves
through `lib_torch_env.sh`'s fallback, and the proxy exports still run.
The lines are noise, not a defect.

A killed scp can leave a Lustre file that fails every read with
`Input/output error`.  The staging hit this once; the remove-and-rewrite
remedy and the verify-on-the-compute-node rule are recorded in
`cluster_use.md`'s failure table.

### 10.6 What remains

The scrontab installation is the one step left of increment 5.  Running
`action_scripts/enable_torch_nightly.sh` on gautschi installs the
`mbirtorch-nightly` block at 03:00 with one GPU; `status_torch_nightly.sh`
then confirms both layers, and the §9 note about re-running the JAX
`enable_nightly.sh` to close its walltime gap still stands.

Increment 6 (cpu-torch on the Mac) is unblocked: the writer's
platform-aware pin already guards the MPS hole it would otherwise hit,
and the launchd template and schedule scripts are in place.

Increment 7 (the n>1 rows) has its preconditions measured: the
device-policy flip has landed, the window ablation exists and must be
repeated at n=2 and n=4, and the n=1 soak clock started 2026-08-08.
The trial's 0.26 GPU-hour figure halves the §3(c) cost basis for the
cadence decision.

The dashboard's vs-main correctness reference is active by
construction, because the tracked branch is literally `main`; the
cross-device reference activates with increment 7 and the
cross-platform reference with increment 6.

---

## 11. Increment 7: the n>1 rows (2026-08-08)

The multi-device rows shipped the same day as the n=1 series, at Greg's
direction.  The plan's three-night soak was waived deliberately, so that
item 3's multi-GPU campaign would have a measured nightly baseline to
start from rather than an empty series.  §11.4 records what the waiver
costs.

### 11.1 The cell set, and the one writer change it needed

The n>1 rows exist only where §3(c) placed them.  The writer gained one
function, `cell_device_counts`, which returns the counts a given
(geometry, size) sweeps: the requested list at parallel and cone for
512×448×384 and 1024×1008×992, and `[1]` everywhere else.  The denoiser
returns `[1]` at every size.  So `TORCH_DEVICE_COUNTS="1 2 4"` widens
exactly sixteen cells and leaves the other thirty-five untouched.

The reduction is a property of the writer rather than of the knob.  A
future edit that widens the knob cannot accidentally widen the small
cells or the denoiser, and the trial's sweep headers confirmed the
behavior directly: `n=[1, 2, 4]` printed at the two large sizes and
`n=[1]` at every other one.

### 11.2 One bug, and the reason single-device testing could not find it

The first 4-GPU trial (job 15004425) lost every one of its 32 n>1 rows
to one line of the writer.  `Shards.gather()` already returns numpy,
because it detaches and concatenates on the host internally, and
`to_numpy` detached the result again.  Every multi-device cell therefore
failed with `'numpy.ndarray' object has no attribute 'detach'`.

The n=1 path returns a plain tensor and never reaches that branch.  No
amount of single-device verification could have found this, and the
Mac's earlier checks were all single-device.

The remedy is a check, not just a fix.  `nt2_local_shard_check.py` pins
all four projector ops at n=1 and n=2 on virtual cpu devices, compares
the two fingerprints at the op's own tolerance, and runs in about a
minute on a laptop.  It reproduces the whole Shards seam with no
allocation at all.  Its measured cross-count agreement is 0.0 to 8.6e-9,
against tolerances of 1e-5 and 1e-4.  The rule this establishes: no
cluster n>1 submission without a local shard check first.

### 11.3 The trial evidence

Job 15005811 ran on four H100s in 32m59s and passed every step.

The device pin holds at multi-device.  The mispin control asked for four
devices with two visible and raised rather than measuring.  Every n=2
row bound `cuda:0, cuda:1` and every n=4 row bound all four.

The values agree across device counts.  All 32 multi-device fingerprints
matched their n=1 row inside the op tolerances, which is the
cross-device correctness reference of §3(e) exercised in the trial
rather than waiting for the dashboard.

The memory ruler is deterministic at n>1 as well.  The §3(c-ii) ablation
repeated five fresh subprocesses at n=2 and again at n=4, reading
1138.1 MB and 663.4 MB with a 0.000 percent spread at both counts.  So
`TORCH_MEM_GATE_WINDOW=1` is measured at every count the nightly
sweeps, not inherited from the n=1 measurement.

The sweep produced 67 cells with zero failures: 35 at n=1 and 16 each at
n=2 and n=4.

### 11.4 The multi-GPU baseline the campaign starts from

Memory shards well at every cell.  Time does not, and the pattern is
the interesting one.

| cell | n=1 | n=2 | n=4 |
|---|---|---|---|
| parallel 512 | 1.59 s / 1.94 GB | 1.35 s (1.18x) / 1.11 GB | 2.32 s (0.69x) / 0.65 GB |
| parallel 1024 | 37.41 s / 25.99 GB | 35.16 s (1.06x) / 14.04 GB | 18.55 s (2.02x) / 7.31 GB |
| cone 512 | 2.50 s / 2.15 GB | 2.40 s (1.04x) / 1.70 GB | 3.62 s (0.69x) / 1.07 GB |
| cone 1024 | 59.13 s / 25.99 GB | 63.31 s (0.93x) / 14.32 GB | 49.69 s (1.19x) / 9.08 GB |

Three readings follow.  The 512 cells lose time at n=4, which is the
communication-dominated behavior §3(c) predicted when it kept the small
sizes single-device.  The parallel 1024 cell scales 2.02x on four
devices while cone reaches only 1.19x, and cone at n=2 is slower than at
n=1.  Memory falls by 3.55x and 2.86x at the two 1024 cells, so capacity
sharding works even where time does not.

These results indicate that multi-device time on torch has real
headroom, and that cone is where it is largest.  That is item 3's
charter, and these rows are its starting baseline.  The rows are
descriptive, not a gate: history-based gating compares each cell against
its own future, so a slow n=4 cell is a baseline rather than a finding.

### 11.5 Cost, and what the waived soak costs

A changed-branch night now allocates four GPUs for about 30 minutes, or
roughly 2 GPU-hours, against 0.26 for the n=1 job.  The walltime is 4
hours at Greg's request, a ceiling the measured run uses about an eighth
of.  Fire-on-change means an unchanged night still costs seconds.

The waived soak costs one specific protection.  Three unattended nights
would have shown the schedule surviving contact with nights nobody
watches, and the n>1 rows now seed before that evidence exists.  The
compensating evidence is that the same wrapper has completed four
supervised end-to-end runs, two of them through the production
scheduled path.  The first two unattended nights deserve a look at
`status_torch_nightly.sh` rather than assumed success.
