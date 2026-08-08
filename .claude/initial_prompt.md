We're continuing the mbirtorch port (the `mbirtorch` repo, parallel checkout to
`mbirjax`; mbirjax is READ-ONLY reference).  This session's task: diagnose and
repair the Triton FORWARD kernels' incorrect results under the banded
multi-device drivers — stages 2 and 3 of the plan recorded at the end of
`plans/torch_port/device_policy_findings.md` (Checkpoint 3) and in the
closing rulings of `plans/torch_port/device_policy_design.md`.  Kernel
diagnosis is Fable-lane work: run it in this session, not a delegated one.

**IMPORTANT — workflow reminders:** stage only (`git add` by explicit file
name), never `git commit` (Greg commits from PyCharm).  Shared checkouts with
other sessions — never `git add -A`.  No plan notation (stage numbers,
checkpoint names) in code or tests.  Cluster jobs are pre-authorized during
this investigation.  Durable records in Alley style — reread
`.claude/writing_style.md` before drafting any findings page.  Announce
golden regeneration if you touch `tests/goldens/`.

Read for orientation (code and measured results over recollection):
1. `plans/torch_port/device_policy_findings.md`, the Checkpoint 3 section —
   the defect's evidence: the body-controlled matrix (job 14954801), the
   non-reproducibility, the CPU bisection clearing the engine.
2. `plans/torch_port/device_policy_design.md`, the closing rulings — the
   interim (torch forward at non-trivial placements; the flip ships) and
   the permanent kernel-times-sharding gate this repair must pass.
3. `plans/torch_port/phase5_findings.md` and `phase5_kernel_design.md` — the
   kernels' design, protocol, and value history.
4. `.claude/lessons.md` §5 and `.claude/cluster_use.md`.

## The defect, in brief

At n=2 and n=4 the Triton FORWARD kernels produce order-1 relative errors
against n=1, in BOTH geometries, under the banded multi-device drivers.  The
back kernels reproduce the pure-torch arms to four significant figures.  The
torch bodies are correct (multi-device float floor ~1e-3; the engine was
cleared by a CPU bisection).  The errors are NON-REPRODUCIBLE run to run
(1.413 vs 1.438 across identical jobs; two arms within one job disagreeing
at 4.58e-01), which rules out benign atomic reordering (measured atomic
repeat floor ~1e-7) and points at a race or partially-observed memory.  n=1
never sees it because a trivial placement short-circuits the banded drivers
entirely — every n=1 gate was structurally blind to this seam.

## Stage 2 — isolation (single-variable probes, before any kernel edit)

Ranked hypotheses and the probes that discriminate them (2-GPU jobs on
gautschi; the checkpoint-3 probe harness is the template):

1. **Stream/device-context race in threaded multi-GPU launches.**  The
   banded drivers launch per-device workers in threads; torch ops dispatch
   on the tensor's device, but a Triton launch targets the launching
   thread's CURRENT device/stream.  A mismatch lets the driver's assembly
   copy observe the forward's atomically-accumulated output before the
   kernel finishes.  The back kernels' survival may itself be the tell: the
   back wrapper's INPUT path forces a `.contiguous()` copy where the
   forward returns a `permute` VIEW with no synchronizing copy on the
   output path.  Probes: (b) `torch.cuda.synchronize(dev)` bracketing the
   forward launch — if that alone repairs the values, it is this class;
   (c) `with torch.cuda.device(d):` around the launch — separates
   wrong-context from wrong-stream; also log `torch.cuda.current_device()`
   inside each worker against the tensor's device, and try returning a
   contiguous COPY from the forward wrapper (the back-asymmetry test).
2. **A banded-contract mismatch in the forward wrappers** (band anchoring,
   `slice_start`, output-row semantics under sub-bands).  Predicts
   DETERMINISTIC wrongness, so it cannot explain the run-to-run variance
   alone, but may coexist.  Probe (a): banded-at-n=1 — force sub-banding on
   ONE device (the `forward_project_slice_band` hook) and compare against
   the plain path; wrong = contract bug, right = multi-device-specific.
3. Probe (d), always: the same banded call twice IN-PROCESS, diffed —
   separates race (differs) from deterministic error (identical), per
   mechanism and per geometry.

Write the findings as you go (`plans/torch_port/kernel_sharding_findings.md`;
scripts as `plans/experiments/torch_port/ks*_*.py` plus sbatch files).
Attribute before fixing — the standard is the campaign record:
single-variable probes, arm checks, and the ruler suspected first.

## Stage 3 — the repair, gated

Fix per the isolated mechanism (likely in the two forward wrappers in
`mbirtorch/triton_parallel.py` / `mbirtorch/triton_cone.py`, or the banded
driver seam in `mbirtorch/projectors.py` / `mbirtorch/_sharding.py`
threading).  Then:
1. The repaired forward must pass the permanent kernel-times-sharding gate
   at the multi-device float floor (the gate the checkpoint-3 ruling
   promoted; its harness is the acceptance bar).
2. Retire the interim: restore forward-kernel selection at non-trivial
   placements in both geometries' `_view_batch_bodies`, and update the
   selection tests' non-trivial-placement branch to the restored contract.
3. Re-run the flip gate (n=2/4 vs n=1 at the float floor) and the n=1
   composed cells (they must reproduce the recorded baselines; the current
   table is in the checkpoint-3 findings).
4. If the mechanism was launch context/stream: sweep the OTHER launch sites
   for the same class (the back wrappers may be correct only by the
   accident of their `.contiguous()`), and say so in the findings either
   way.

Gate-script protocol note (already ruled): `compile_mode='off'` does not
disable kernels; an arm that intends the plain torch engine sets
`MBIRTORCH_DISABLE_TRITON=1`.

## Standing context

- Cluster: gautschi (ssh BatchMode; the accepted key is `~/.ssh/id_rsa`);
  sbatch on partition `ai`, account `bouman`, --cpus-per-task=14 per GPU,
  --gpus-per-node=2 for the probe jobs.  mbirtorch checkout:
  `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`;
  TORCHPY=`/scratch/gautschi/buzzard/torch_p0/env/bin/python`; results in
  `/scratch/gautschi/buzzard/torch_p3/results/`.  SYNC RULE: per-file scp +
  md5 verify of every changed file (rsync has silently dropped files).
  Slurm `--export` splits on commas — pass env via the submission shell.
- Concurrent sessions may be active (the nightly wiring; docs).  The
  device-policy session is closing checkpoint 3 with the interim; do not
  start stage 3's selection changes until the interim has landed.
- Terminology: "variants" (never arms/cells for variant sets).
