# Phase 4 design — multi-device mbirtorch

**Status:** DESIGN SPIKE IN PROGRESS (2026-08-05).  The gloo spike is done;
the 2-H100 discriminator (gautschi job 14870898) fills the decision section.
The substrate decision goes to Greg with those numbers before implementation.
**Plan:** port_plan.md section 5, Phase 4.  **Spike scripts:**
`plans/experiments/torch_port/archive/p_phase0_4/p4s1_collectives_gloo.py`,
`plans/experiments/torch_port/archive/p_phase0_4/p4s2_dual_gpu.py`.

## Goal and gates

Phase 4 brings mbirtorch to n = 2 and 4 GPUs with mbirjax's sharding
architecture: view-sharded sinograms, slice-sharded recons, and the two
banded collectives.  The gates are the multi-device metrics cells against
tuned multi-device jax, under the same replacement rule.

## What transfers, and what is already paid for

mbirjax's design transfers conceptually intact.  A `Placement` maps each
array axis to device bands.  The forward projection broadcasts each
slice-owner's recon band to the view-owners, which accumulate per-band
partial sinogram shards.  The back projection reduces per-band partials
across view-owners onto each band's slice-owner.  The engine stays
placement-agnostic because every array touch already routes through the
chokepoints built this week: `_shard_sinogram` / `_shard_recon` place, pad,
and validate; `_gather_sinogram` / `_gather_recon` crop and assemble.  The
deferred-seams list in port_plan.md is the Phase 4 work inventory: the
qGGMRF interface masks and halos, the prepare-weights zero-fill, the
device-form ones sinogram body, the relaxed entry validations, and the
per-device view-batch budget.

## The substrate question

The one open decision is how devices are driven.  Option A is a single
process: one python loop (or thread per device) issues work to each GPU,
with band transfers as peer-to-peer `tensor.to()` copies.  It keeps the
engine structure, the chokepoints, and the checkpoint semantics unchanged;
its risk is the GIL serializing eager dispatch, the torch analog of the jax
host-dispatch findings.  Option B is one process per GPU with NCCL: the
engine runs SPMD with `dist.broadcast` and `dist.reduce` for the two banded
collectives; it scales cleanly but restructures the driver loop, the RNG
seeding contract, and the stats reductions across ranks.

## Spike 1 — the collectives on gloo (DONE, local)

Both banded collectives express directly in torch.distributed and run on
CPU ranks with the gloo backend.  Forward matches a single-process
reference exactly, and back matches at 1.7e-6, at both 2 and 4 ranks.
This also settles the local-test story: gloo CPU ranks are the torch analog
of mbirjax's XLA host-device split, so multi-device correctness tests run
on any machine, in CI, with no GPU.

## Spike 2 — the discriminator on 2 H100s (job 14870898)

Three measurements.  (a) Threaded dual-GPU scaling of a warm compiled
matmul + gather + index_add chain: scaling near 2x keeps Option A alive;
scaling well below ~1.8x means GIL contention and points to Option B.
(b) Peer-to-peer `tensor.to()` bandwidth at band sizes, the unit transfer
of Option A's broadcast.  (c) NCCL broadcast and reduce at the same sizes
under torchrun, the unit transfers of Option B.

RESULTS (job 14870898, 2 H100s).  Transfers first: p2p `tensor.to()` reaches
141-154 GiB/s at the 4 MiB and 16 MiB band sizes, and NCCL reaches 128-155
GiB/s at the same sizes.  Both are NVLink-class and within 1.2x of each
other, so the transfer axis does not discriminate.  The threading
measurement needs an honest caveat: the wall-clock "scaling" number is a
harness artifact (the wall included per-thread tensor allocation and
warmup, ~190 ms, around a 5 ms workload).  The per-device timings inside
the threads are the valid reading, and they show both GPUs running the
compiled chain at exactly the solo speed (0.005 s each, concurrently) --
no GIL contention at ~170 us per launch.  What the spike cannot rule out
is contention at VCD's fine-grained dispatch (many small launches per
subset), the regime the jax host-dispatch findings flag.  That question is
answerable only with the real engine on two devices.

## Decision (recommendation to Greg)

Option A, single-process, with a recorded fallback.  The reasoning: the
transfer speeds are a wash; the per-device thread timings show no coarse
contention; the engine, chokepoints, and checkpoint semantics survive
unchanged; and the Phase 2 compiled-glue work already coalesced the
per-subset launches that would stress the GIL.  The risk that remains --
fine-grained dispatch contention -- concentrates at small cells, where
multi-device matters least, and it gets measured for real in increment 3.
The fallback is pre-paid: the spike-1 collectives skeleton is the NCCL SPMD
implementation seed, and the placement/chokepoint layer is
substrate-agnostic, so a mid-course switch to Option B changes the driver
loop, not the architecture.

## Decision rule

Choose Option A if the threaded chain scales at or above ~1.8x AND p2p
bandwidth is within ~2x of NCCL at the dominant band sizes; the simplicity
and the untouched engine are worth a modest transfer penalty.  Otherwise
choose Option B and take the SPMD restructure.  Either way the collectives
code from spike 1 is the implementation skeleton, and the gloo path remains
the correctness harness.

## Prior art: the jax fbp-filter parallelism study

`plans/sharding/parallel_performance/fbp_filter_parallelism_comparison.md`
answered this same substrate question for jax, and its results transfer as
evidence and as method.  Its production winner on GPUs was path G: python
THREADS driving per-device work on pre-sharded, device-resident data --
zero host roundtrips, correct, and super-linear at small sizes from reduced
per-GPU memory pressure.  That is Option A.  Its losers map to paths this
design already excludes: threading through host-memory copies died on PCIe
roundtrips (the reason Option A's bands move device-to-device, the p2p
numbers above), and jax's SPMD layer (shard_map) produced INCORRECT
multi-GPU results on the L40S with the root cause never found -- the
substrate-maturity risk whose torch analog (DTensor) this design avoids.
Two of its methods carry into the Phase 4 harness: the 1-DEVICE DIAGNOSTIC
(run the multi-device code path on one device to price orchestration
overhead with no data movement in the frame) and a per-path value-diff
column beside every timing.

## The compile race, and what others do about it (2026-08-05)

Implementing the sharded engine hit a torch.compile thread-safety failure on
CUDA (triton launcher asserts under concurrent cold compiles), fixed here
with a process-wide compile lock keyed on unseen input shapes.  The research
question was whether others found non-serialization answers.  Findings: this
is a KNOWN, acknowledged, unresolved class -- a PyTorch maintainer states
PT2 has "no thread-safety guarantees" in a multithreaded-compile issue, the
PT2 lead's usage guide says multithreading "is currently buggy", a recent
issue catalogues five distinct global-state races inside inductor (with
fixes proposed as thread-locals plus LOCKS -- upstream's own medicine is the
same as ours), and the matching triton issue is open with no fix.  The
alternatives in production use are: (a) WARMUP/PRECOMPILE before threading
-- compile every shape on the main thread, threads only execute; the
standard serving pattern, and the strongest form is AOT precompilation;
(b) dynamic=True / mark_dynamic -- one dynamic kernel instead of per-shape
specializations shrinks both the race window and recompile stalls, at a
kernel-speed cost that would need re-gating; (c) PROCESS-PER-GPU -- the
industry sidestep (vLLM-class systems), which is exactly this design's
Option B fallback.  Assessment: our shape-keyed lock is the lazy form of
(a); an explicit warmup pass (the engine's shape set is enumerable: one
subset size per granularity plus the full-index size, per device) would
remove even the first-touch serialization from the iteration loop and is
the recorded polish; (b) is a Phase 5 measurement candidate; (c) remains
the pre-paid fallback.

## Increments after the decision

1. Placement objects and the chokepoint bodies (place, pad, validate, crop).
2. The forward band broadcast and back band reduce in the sparse drivers,
   with the per-device view-batch budget.
3. The engine pass: per-shard subset updates, qGGMRF halos and interface
   masks, cross-device stats, seeded-partition consistency.
4. The deferred-seams sweep (the port_plan list), then the multi-device
   gate readout at n = 2 and 4.
