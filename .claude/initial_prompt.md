We're continuing the mbirtorch port (the `mbirtorch` repo, parallel checkout
to `mbirjax`; mbirjax is READ-ONLY reference).  This session's task:
`current_plans.md` item 3 — the multi-GPU performance investigation and
tuning.  This is a measurement campaign: PLAN FIRST (write
`plans/torch_port/multigpu_plan.md`, stage it, STOP for Fable review via
Greg before cluster campaigns), then execute in gated increments.

**IMPORTANT — workflow reminders:** stage only (`git add` by explicit file
name), never `git commit` (Greg commits from PyCharm).  Shared checkouts —
never `git add -A`.  No plan notation in code or tests.  Cluster jobs are
pre-authorized during the agreed investigation.  Durable records in Alley
style — reread `.claude/writing_style.md` before drafting.  Announce golden
regeneration if you touch `tests/goldens/`.

Read for orientation (code and measured results over recollection):
1. `.claude/claude_prompt.md`, `.claude/lessons.md` (§5, §6),
   `.claude/cluster_use.md`.
2. `plans/torch_port/kernel_batching_findings.md` — the n=1 composed
   baselines (parallel 1.13x/1.56x of jax, cone 0.88x/1.00x) and the gate
   discipline (arm checks, measured floors).
3. `plans/torch_port/device_policy_findings.md` and the closing rulings in
   `device_policy_design.md` — the live widening rule, the ledger, the
   forward-kernel defect story and its repair, and the permanent
   kernel-times-sharding gate (its harness is reusable here).
4. `plans/torch_port/kernel_sharding_findings.md` — the repair record; the
   restored forward kernels are what this campaign measures.
5. `plans/torch_port/phase5_findings.md` — the shared-sinogram value
   protocol and the composed-gate patterns.

## The four goals (from current_plans item 3)

1. A full n=1/2/4 gate readout with the repaired kernels on, at the live
   all-device default, both geometries, the standard gate cells.
2. Attribute any gaps and tune what the data indicates: per-device view
   chunks, the seam and streaming hooks (`back_project_slice_band` is the
   named lever — the band reduce is FLAT in device count, about
   1.5 x cyl(P, S_pad) at n=2–4, so more devices do not shrink it), and
   the widening rule's parameters.
3. The deferred value comparison: jax vs torch across device counts,
   separating the shared partition-order term (~5e-4 eager floor) from
   torch's compile-latitude term (~5e-3 class).
4. The forward-attribution arm: the parallel forward's measured share of
   composed time at n=1/2/4.  This number is item 13's entry gate (the
   sorted-stream decision) and also feeds the widening speed guard below.

## Protocols that are load-bearing here

- Every arm PINS its device count (`configure_devices` or
  `MBIRTORCH_NUM_DEVICES`) and asserts the realized device list — the
  automatic default is live, and the phase-4-era scripts were audited for
  exactly this hazard; new scripts follow the pinned pattern.
- An arm that intends the plain torch engine sets
  `MBIRTORCH_DISABLE_TRITON=1`; `compile_mode` does NOT disable kernels.
- n>1 VALUE gates run eager-to-eager (the compile-latitude policy);
  cross-framework value comparisons hand ONE shared sinogram artifact to
  both frameworks; memory is re-measured per arm, never inferred.
- Known priors to test rather than assume: the n>1 back limiter was the
  band transpose (pre-kernels); the old 512-cell reading of four devices
  about 3x SLOWER than one predates the kernels and the batching repair,
  so the crossover has likely moved — measure it fresh.

## Two decisions this campaign owes

1. **The widening speed guard**: the automatic path is capacity-only today
   and will widen small problems that run faster on one device.  From the
   measured crossover, recommend a work-size floor below which the
   automatic path stays at n=1 (one robust knob, data-picked knee — not a
   communication model), for Fable review.
2. **The nightly n>1 cadence**: the readout's costs inform whether the
   nightly's n>1 rows run nightly or weekly (`nightly_plan.md` §9 raised
   it).  Report the numbers; Greg decides.

The nightly (item 4) is live and seeding history — its rows are regression
protection for this campaign's tuning, not its instrument; campaign
measurements use your own gated harnesses.

## Standing context

- Cluster: gautschi (ssh BatchMode; accepted key `~/.ssh/id_rsa` — if key
  files are unreadable in your environment, ask Greg to run
  `ssh-add ~/.ssh/id_rsa` once).  sbatch on partition `ai`, account
  `bouman`, --cpus-per-task=14 per GPU, --gpus-per-node=2 or 4 for the
  multi-device cells.  mbirtorch scratch checkout:
  `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`;
  TORCHPY=`/scratch/gautschi/buzzard/torch_p0/env/bin/python`; results in
  `/scratch/gautschi/buzzard/torch_p3/results/`.  SYNC RULE: per-file scp +
  md5 verify of every changed file.  Slurm `--export` splits on commas —
  pass env via the submission shell.
- Scripts to `plans/experiments/torch_port/` (suggested prefix `mg*_`);
  findings to `plans/torch_port/`.
- Concurrent sessions may be active (Charlie's, on the utility API and
  docs).  Terminology: "variants" (never arms/cells for variant sets).
