# Kernel sharding findings — the Triton forward under the banded drivers

**Status: COMPLETE (2026-08-08).**  The mechanism is isolated and repaired,
the interim selection rule is retired, and every acceptance gate has
passed: the standing kernel-times-sharding gate 12/12 (twice, the second
run with strengthened arm checks), the flip gate 18/18, the composed n=1
cells reproducing the recorded baselines in full, and the full suite on
H100 at 477 passed with none failed.  This page records the diagnosis and
repair of the Triton FORWARD kernels' incorrect results under the banded
multi-device drivers.  The
defect was found and bounded by the device-policy campaign, and its evidence
lives in the checkpoint-3 section of `device_policy_findings.md`: order-1
relative errors at n=2 and n=4 in both geometries, non-reproducible run to
run, with the back kernels and the torch bodies clean.  The mechanism is the
LAUNCH CONTEXT: a Triton launch targets the launching thread's current CUDA
device, the banded drivers launch from worker threads whose current device
is 0, and the shard's own consumers race the misplaced kernel.  The
isolation is job 14973047, a seven-arm single-variable matrix.  The
device-context arm alone repaired the values to the kernel-parity class,
and every rival class was refuted.  The repair brackets all four wrapper
launches in `with torch.cuda.device(...)`, the interim selection rule is
retired, and the acceptance gates are jobs 14973464-14973467.

## The launch path, read before any probe

The candidate mechanism came from reading the launch path, and it is a
device-targeting asymmetry between Triton and torch.  A Triton launch targets
the launching THREAD's current CUDA device.  A torch op targets the TENSOR's
device, whatever thread issues it.

The Triton half of that claim was verified in the installed toolchain rather
than recalled.  On the cluster env (torch 2.13.0, triton 3.7.1),
`triton/runtime/jit.py` line 713 reads `device =
driver.active.get_current_device()`, with the stream from
`get_current_stream(device)`.  In `triton/backends/driver.py` lines 57-61,
those bind to `torch.cuda.current_device` and
`torch._C._cuda_getCurrentRawStream`.  Both are thread-current lookups.  A
fresh thread's current CUDA device is 0.

The banded drivers put those two dispatch rules on opposite sides of a thread
boundary.  `_sharding.run_per_device` issues per-device work from fresh
worker threads and sets no device in them, by a design note that says torch
needs no thread-local device because tensors carry theirs.  That note is true
for every torch op in the worker.  It is false for the Triton launch inside
the kernel wrappers, and none of the four wrappers
(`triton_parallel.py`, `triton_cone.py`) brackets its launch in a device
context.  Under this reading, every worker's kernel launch lands on device
0's default stream, whichever shard the worker was handed.

A launch on the wrong device still computes, which is why nothing crashed.
The H100 nodes have peer access between devices, and the drivers' own
device-to-device copies enable it, so a kernel in device 0's context can
dereference device 1's memory over NVLink.  What the wrong-device launch
loses is ORDERING: the shard's producers and consumers are torch ops on the
shard device's default stream, and the kernel is on device 0's, with no
ordering edge between them.

## Why the forward breaks and the back survives, under this reading

The forward's output is consumed on the shard's own stream with nothing
ordering the read after the kernel.  The driver's view-range loop assigns
each kernel block into the assembled output on the shard device's stream.
The banded assembly then concatenates or accumulates on that same stream,
and the final gather copies to host from it.  Each of those reads can
observe the atomics' target mid-flight, which matches both the order-1
magnitude and the run-to-run variance the checkpoint-3 matrix recorded.

The back kernels take the same wrong-context launch, so their four-digit
agreement needs its own explanation.  The explanation this reading offers is
an accident of reduce topology.  A back partial is consumed only through
`sum_band_to_owner`, which moves the device-0 partial first.  That first
copy runs on device 0's default stream, behind every misplaced kernel
launch.  Torch's cross-device copy protocol fences the copy against the
destination's current stream in both directions, so the owner's stream comes
to happen-after every kernel, and each later per-device copy inherits that
edge transitively through the same two-way fences.  Every read of every
partial therefore lands after every kernel, and the back path is correct by
the shape of its reduce rather than by construction.  This asymmetry
explains the isolation matrix without any appeal to the wrappers' input
paths, and it predicts that the back wrappers are one driver refactor away
from the same defect.

The n=1 blindness follows from the same mechanics.  A trivial placement
short-circuits the banded drivers, so no worker threads exist.  Even
threaded, the thread-current device and the tensor device would agree at one
device.  Every n=1 gate therefore exercised the kernels in the one
configuration that cannot show the seam.

## The probe design, with predictions registered in advance

The mechanism above is a reading of code, and the probes test it with one
variable each.  All arms run on two GPUs at the dp5 cell (256 views, 64
rows, 64 channels), in both geometries, with bodies bound explicitly through
the `_view_batch_bodies` hook.  Every kernel arm binds the TORCH back body,
so the forward carries the whole comparison.  The script is
`plans/experiments/torch_port/ks1_launch_context.py` (job 14973047), and
each arm runs a forward projection twice in-process, then the dp5-style
3-iteration recon.

The arms and their registered predictions:

1. **The observation.**  Every kernel arm logs the thread-current device
   against the tensor device at each launch.  Predicted: current device 0 at
   every launch, including shards on cuda:1.
2. **plain, n=2.**  The defect reproduced under explicit binding.
   Predicted: order-1 recon error against the torch arm, and the two
   in-process forward calls differ (a race varies; a contract bug repeats).
3. **devctx, n=2.**  The launch bracketed in `with torch.cuda.device(dev)`.
   Predicted: repaired to the kernel-parity class, because the launch
   context is the whole mechanism.
4. **sync_tensor, n=2.**  `torch.cuda.synchronize(tensor.device)` around
   the launch.  Predicted: NOT repaired, because the kernel is not on that
   device, and a sync of the right device cannot order a kernel on the
   wrong one.
5. **sync_all, n=2.**  Synchronize every visible device around the launch.
   Predicted: repaired, because a full barrier orders any placement.  This
   arm is an upper bound that localizes the class, not a candidate fix.
6. **contig, n=2.**  The forward's permute view returned as a contiguous
   copy, the shape of the back wrappers' input asymmetry.  Predicted: NOT
   repaired, because the copy runs on the shard's stream and is as
   unordered against the kernel as the view was.
7. **banded1.**  The full banded driver on one device, via a two-shard
   placement listing cuda:0 twice.  Predicted: clean, exonerating the
   banded contract (band anchoring, `slice_start`, sub-band assembly) and
   pinning the defect to multi-device placement.

The discrimination rule was fixed before the run.  If devctx repairs alone,
the mechanism is the launch context.  If sync_tensor repairs, the mechanism
is a same-device async race instead.  If only sync_all repairs, the defect
is ordering with the placement unresolved.  If contig repairs, the mechanism
is view lifetime.  If banded1 is wrong, the wrapper disobeys the banded
contract, which no other arm would localize.

## The probe results: every prediction held, every rival refuted

The launch-site observation came first, and it was unanimous.  Every n=2
kernel arm logged 174 launches at thread-current device 0 against tensors
on cuda:1, in both geometries, alongside 174 correctly-placed launches for
the cuda:0 shard.  The dup2 arm logged all 348 launches at current 0
against cuda:0 tensors.  These counts are the direct sighting of the
wrong-context launch, before any value is read.

The value matrix then discriminated exactly as registered (job 14973047,
recon and forward-only max-rel against the torch arm at the same layout):

| arm | parallel recon | cone recon | parallel fwd repeat | cone fwd repeat |
|---|---|---|---|---|
| plain, n=1 | 1.302e-06 | 8.678e-07 | 7.1e-07 | 4.7e-07 |
| plain, n=2 | 1.384e+00 | 1.326e+00 | 9.9e-01 | 9.2e-01 |
| plain, dup2 | 4.466e-04 | 4.575e-04 | 4.7e-07 | 5.3e-07 |
| devctx, n=2 | 3.375e-07 | 1.109e-06 | 6.5e-07 | 4.7e-07 |
| sync_tensor, n=2 | 1.176e+00 | 1.335e+00 | 9.8e-01 | 9.4e-01 |
| sync_all, n=2 | 3.375e-07 | 1.012e-06 | 6.5e-07 | 4.7e-07 |
| contig, n=2 | 1.498e+00 | 1.071e+00 | 1.0e+00 | 8.5e-01 |

The plain arm reproduced the defect under explicit binding, at order one
with an order-one in-process repeat spread.  The repeat spread is the race
signature: the same call in the same process does not return the same
wrong answer.  The devctx arm repaired both geometries to the kernel-parity
class, at 3.4e-07 and 1.1e-06 against the torch arm.  The sync_tensor arm
stayed at order one, which refutes the same-device async class: a
synchronize of the tensor's device cannot order a kernel that is not on
that device.  The contig arm stayed at order one, which refutes the
view-lifetime class; its forward-only reading of 39.5 in parallel says the
copy consumed the buffer even earlier than the assembly did.  The sync_all
arm repaired, as any full barrier must, and localizes nothing further.

The dup2 arm exonerated the banded contract, and it carried a cross-check
that was not designed in.  Its forward-only reading sits at the parity
floor.  The banded forward assembly is therefore value-correct on one
device.  Its recon reading is 4.466e-04 in parallel and 4.575e-04 in cone,
against the plain n=1 arm.  Those two numbers reproduce the isolation
matrix's TORCH arms at n=2 (4.466e-04, 4.575e-04) to every printed digit.
These agreements indicate the banded float floor is a property of the
PARTITION arithmetic, not of the devices.  Two view-spans summed in the
banded order produce the same float divergence whether they run on two
devices or on one device twice.

## Why the back kernels never showed it

The reduce-topology account survives contact with the measurements, and it
stands as the explanation of record.  Both back wrappers took the same
wrong-context launch, which the launch logs now confirm directly.  Their
sole consumer is `sum_band_to_owner`, whose first contribution move is
device-0-sourced.  That copy runs on device 0's default stream, behind
every misplaced kernel, and torch's two-way cross-device copy fences then
chain every later read behind it.  The forward has no reduce, so its
consumers read unfenced.  The back kernels were therefore correct by an
accident of reduce topology, and the repair brackets their launches too,
because the next driver refactor should not be able to un-save them.

## The repair

The fix is the device context, applied at every launch site.  All four
wrappers now bracket their launch in `with torch.cuda.device(...)` on the
tensors' device, which places the kernel on the device's default stream,
where every torch producer and consumer of the same tensors already runs.
No synchronization is added anywhere: the ordering comes from stream
placement, exactly as it does for the torch bodies.

The launch key now leads with the device index, and this is a second
defect the diagnosis exposed.  Triton caches compiled kernels per device,
so under the repair a shape compiled on device 0 compiles AGAIN on device
1's first launch.  The `_COMPILED_LAUNCH_KEYS` set decides whether a launch
takes the process-wide compile lock, and it was device-blind.  Device
blindness was accidentally consistent while every launch claimed device 0.
Once launches spread across devices, it would have readmitted the
concurrent-cold-compile crash class the lock exists for.  With the device
in the key, each device's first compile of a shape serializes.

Two records were corrected where the seam was documented wrong.
`run_per_device`'s docstring claimed torch needs no thread-local device;
that claim is true for torch ops and was false for raw kernel launches,
and it now states the Triton caveat and points here.  The selection sites
in both geometries carry the measured basis for the restored rule.

The interim selection rule is retired.  Both geometries'
`_view_batch_bodies` select the forward kernel by availability alone, at
any placement.  The CPU selection tests now pin layout-independence in
both directions, keeping the follows-the-layout structure so a rebuilt
layout re-selects rather than latching.

The standing gate grew the arms the repair owes it.  The forward-kernel
arm and the default-selection arm join the torch and back-kernel arms at
the multi-device floor.  A new trivial-placement arm pins the third
exposure the mechanism implies.  A model pinned to cuda:1 launches its
kernels from the main thread, whose current device is 0, with no banded
driver involved.  No gate had ever run the kernels on a nonzero single
device.  The mechanism says both directions were wrong there, and the back
was unprotected, because the trivial path has no reduce to fence it.

## The launch-site sweep

The ruling asked for a sweep of the other launch sites, and the sweep is
short.  A grep for Triton launches over the package finds exactly four, the
four wrapper launches, plus one docstring example in `projectors.py`.  All
four are now bracketed.  The torch bodies compile through inductor, whose
generated launchers carry their own device guards, and the torch arms of
every matrix here measured clean, so no other site carries this class.

## Acceptance gates

Four jobs carry the acceptance record: dp6, the standing
kernel-times-sharding gate on two GPUs; the full suite on one H100; dp4,
the flip gate at n=1/2/4 on four GPUs; and kb3, the composed five-arm n=1
gate at all four cells against the recorded baselines.

**dp6 (14973464): 12 of 12 pass in 28 seconds.**  The forward-kernel arm
and the default-selection arm joined the torch arms under the 5e-3 floor in
both geometries, which is the bar the checkpoint-3 ruling set for retiring
the interim.  The new trivial-placement arm passed on cuda:1, closing the
third exposure.  After this run the two default-selection tests gained an
arm check that fails loudly if the availability gates silently decline and
leave a torch-vs-torch comparison, the same vacuity class the p5 gate
rework closed.  The strengthened file re-ran as job 14973644 and passed 12
of 12, with the arm checks confirming both kernels bound.

**dp4, the flip gate (14973466): 18 of 18 pass.**  The engine floors
reproduce the recorded values to every printed digit: n=2 and n=4 against
n=1 read 4.47e-04 and 9.49e-04 in parallel, 4.57e-04 and 4.95e-04 in cone.
The automatic-versus-explicit check is the n=4 signature flip.  Both of its
arms now run the repaired forward kernel on four devices, and they agree at
3.37e-07 and 4.34e-07.  Before the repair, the same pair inside the same
gate disagreed at 4.58e-01.

**The suite job (14973465) failed at pip, not at pytest.**  Four jobs
launched together, and two editable installs into the shared env raced;
the loser read a torn `.pth` and exited before collecting a test.  The
suite resubmitted as 14973643, chained so no install races a running job's
subprocess imports, and the failure signature went into
`.claude/cluster_use.md`.

**The kb3 first attempt (14973467) failed on a stale ruler, not on the
repair.**  Every torch arm at every cell raised
`ConeBeamModel.__init__() got an unexpected keyword argument 'device'`,
while the jax arms ran.  The scratch copy of `kb3_gate.py` predated the
constructor amendment; the migrated copy lived in the plans repo and had
never been re-synced.  This is the per-file sync rule biting from a new
side: a migration that edits staged scripts must re-stage them.  The
migrated script is synced and kb3 re-runs as 14973798, chained after the
dp6 re-run (14973644).

The stale-ruler class was then swept rather than fixed one specimen at a
time, and the sweep found it was larger than one script.  An md5 sweep of
every script and sbatch on the torch_p3 scratch staging area against the
plans repo found twenty-one stale files, kb3's among them.  A second
sweep, of the whole `mbirtorch_src` tree against the local HEAD, found
thirty-five more: thirteen stale files, mostly pre-amendment test files,
and the `preprocess` subpackage, `hsnt`, and their tests never staged at
all.  The resubmitted suite (14973643) had already failed on exactly those
files, every failure a constructor `TypeError` or a missing module, with
the kernel and sharding batteries passing beside them.  The files that
carried the probes and the gates were verified current before those jobs
ran, so no measured result here rests on the stale files.  All fifty-six
are synced and verified per file.  One transfer hit a Lustre
`Input/output error` on read-back, and the per-file md5 verify is what
caught it; a delete and fresh write repaired it, which is the sync rule
earning its keep.

**The kb3 re-run (14973798) reproduced the recorded baselines wherever it
ran on a stable tree, and I broke its cone arms myself.**  The tree sweep
above synced files WHILE this job was running, and a cone worker imported
the new `__init__.py` against the not-yet-copied `vcd_utils.py`.  That is
the same mid-run mutation hazard as the pip race, arriving through scp,
and it is now recorded beside it.  The arms that ran outside the sync
window carry the readout, and they hold: parallel-512 peak 1.93 GB and
kernel-versus-body 3.17e-03 against recorded 1.93 and 3.17e-03;
parallel-1024 peak 23.22 GB, kernel-versus-body 2.32e-03, and the
shared-sinogram cross-framework 6.11e-03 against recorded 23.22, 2.31e-03,
and 6.11e-03 (the documented 3-iteration trajectory spread); cone-512 peak
2.15 GB and 2.76e-04 against recorded 2.15 and 2.76e-04.  The warm-time
ratios match the recorded table within two percent at those cells.  A
clean kb3 ran as 14975410 on the stable tree, and it closed the readout:
all four cells completed with every arm check ok and no errors.

The clean run reproduces the recorded n=1 baselines in full.  The
kernel-arm peaks read 1.93, 23.22, 2.15 and 23.68 GB across the four
cells, each equal to its recorded value to the printed digit.  The
kernel-versus-body values read 3.17e-03, 2.31e-03, 2.76e-04 and 9.74e-05,
against recorded 3.17e-03, 2.31e-03 to 2.32e-03, 2.76e-04 and 9.74e-05.
The warm torch-over-jax time ratios land at 1.13, 1.55, 0.87 and 0.99,
inside the recorded 1.11-1.13, 1.54-1.56, 0.88 and 0.98-1.00 bands.  The
parallel-1024 shared-sinogram cross-framework value reads 6.08e-03
against its recorded 6.11e-03, the documented trajectory spread of
3-iteration runs.  These agreements say the launch-context repair changed
nothing on the n=1 path, which is what a bracket around an
already-correctly-placed launch must do.

**The suite's second failure (14974065) named a dependency the env never
had.**  Completing the tree brought in the `preprocess` subpackage, whose
`scipy` import aborted collection.  The pyproject declares scipy and eight
more as core dependencies, and the torch_p0 env, built with `--no-deps`
installs, carried none of them, which is why the partial staging had
never tripped it.  The nine declared dependencies installed as job
14975396, restoring the env to the package's own contract.

**The full suite on H100 (14975409): 476 passed, 18 skipped, and one
latent stale test.**  This was the first full-suite run on a CUDA node
since the constructor amendment, and it surfaced exactly one failure:
`test_only_an_unindexed_cuda_model_is_eligible_for_the_automatic_count`
asserts the PRE-amendment eligibility rule, that an unindexed
`devices=['cuda']` stays automatic.  The amended rule is one bit, any
`configure_devices` call is explicit, and the test's CUDA gate meant it
had never run anywhere since.  It is not touched by the kernel repair.
The test now asserts the amended contract, including the untouched-model
case that runs on every backend.  The final suite run (14975757) is
green: 477 passed, 18 skipped, none failed.

## Files and rows

The repair, in mbirtorch: `mbirtorch/triton_cone.py` and
`mbirtorch/triton_parallel.py` (the four launch brackets and the
device-leading launch keys), `mbirtorch/parallel_beam.py` and
`mbirtorch/cone_beam.py` (the interim retired at both selection sites),
and `mbirtorch/_sharding.py` (the `run_per_device` docstring correction).

The tests, in mbirtorch: `tests/test_kernels_sharded.py` (the forward,
default-selection, and cuda:1 trivial-placement arms, the arm checks, and
the layout-independent selection contract), `tests/test_triton_parallel.py`
and `tests/test_triton_cone.py` (the selection tests restored to
layout-independence), and `tests/test_memory_ledger.py` (the latent
pre-amendment eligibility test brought to the amended contract).

The probes and gates, in mbirjax_plans:
`plans/experiments/torch_port/ks1_launch_context.py` and
`ks1_gautschi.sbatch` (the isolation matrix), the refreshed
`dp6_sharded_kernels.sbatch` header, and this document.
`.claude/cluster_use.md` gained the concurrent-install and mid-run-sync
failure signatures.

Raw rows stay on scratch per convention.  The isolation matrix is
`/scratch/gautschi/buzzard/torch_p3/results/ks1_launch_context_20260808_080450.jsonl`
with its per-arm sinogram and recon arrays beside it; the clean composed
readout is `.../results/kb3_gate_h001_20260808_091915.jsonl`; the flip-gate
rows are `.../results/dp4_flip_gate_20260808_081819.jsonl`.  The job logs
are `ks1_14973047.log`, `dp6_14973464.log`, `dp6_14973644.log`,
`dp4_14973466.log`, `kb3_14975410.log`, and `suite_14975757.log`, all in
`/scratch/gautschi/buzzard/torch_p3/`.
