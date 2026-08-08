We're continuing the mbirtorch port (the `mbirtorch` repo, parallel checkout
to `mbirjax`; mbirjax is READ-ONLY reference).  This session's task:
implement `current_plans.md` item 4 — mbirtorch in the nightly regression —
against the APPROVED plan at `plans/torch_port/nightly_plan.md`
(Fable-reviewed; implement it, do not re-open its decisions).  A prior
session started this work and ended mid-flight: your FIRST step is to
establish what already exists (staged or committed, in mbirjax_metrics and
in `plans/experiments/torch_port/`) rather than assuming a clean start.

**IMPORTANT — workflow reminders:** stage only (`git add` by explicit file
name), never `git commit` (Greg commits from PyCharm).  Shared checkouts
with other sessions — never `git add -A`.  No plan notation (increment
numbers, item numbers) in code or harness files that live in mbirtorch or
mbirjax_metrics.  Durable records in Alley style — reread
`.claude/writing_style.md` before drafting findings text.  Announce golden
regeneration if you touch `tests/goldens/`.

Read for orientation (rely on code and results over recollection):
1. `plans/torch_port/nightly_plan.md` — THE plan; its increments 1–7 and
   the TRIAL-RUN GATE between increments 4 and 5 are the work order.
2. `.claude/claude_prompt.md` — collaboration style and code-change
   workflow.
3. `.claude/lessons.md` — engineering playbook (§5 and §9 especially).
4. `.claude/cluster_use.md` — gautschi usage.
5. `plans/torch_port/kernel_batching_findings.md` and
   `device_policy_findings.md` — the current composed baselines and the
   device-policy state the rows will measure.

## Three kickoff decisions — confirm with Greg in your FIRST reply

The plan left three calls to Greg.  State the recommendations and ask him
to confirm or override before increment 5 (nothing before the trial gate
depends on them): (1) track `main` alone to start; (2) remove the
incomparable `results/gpu-torch/master/` run (committed-data deletion, his
call); (3) the dashboard's default landing view prefers the jax family.

## Standing constraints (hard, from the plan and its review)

- THE RUNNING MBIRJAX NIGHTLY MUST NOT BREAK.  Wiring is additive; the one
  shared-engine edit ships with its diff-the-gate pure control.
- Every row PINS its device count explicitly (`configure_devices` plus the
  `MBIRTORCH_NUM_DEVICES` worker export) and ASSERTS the realized device
  list — the all-device default is live, so an unpinned row silently
  measures a widened run.
- The platform guard: the wrapper DECLARES the platform key and the writer
  raises on disagreement with the hardware; never infer the key.
- Deployment seam: the production nightly runs committed code only.  Trials
  go by per-file scp with md5 verification of every changed file into the
  scratch checkout, submitted directly; `REG_TORCH_NO_PUSH=1`.
- Timing note in this session's favor: the forward-kernel repair has
  landed, so the history the first runs seed reflects the configuration
  that will persist.  Do not start before verifying the mbirtorch suite is
  green at HEAD.

## The finish line

Increments through the trial-run gate, the gate's evidence reported, then —
after Greg commits and enables — TRIGGER THE FIRST REAL SCHEDULED-PATH RUN
and report its results (run file, gate block, records, dashboard render).
That first run is the milestone the item-3 session starts from.  Update
`plans/torch_port/nightly_plan.md` with a findings/closing section and
`current_plans.md` item 4 at the end, in Alley style.

## Standing context

- Cluster: gautschi (ssh BatchMode; the accepted key is `~/.ssh/id_rsa` —
  if key files are unreadable in your environment, ask Greg to run
  `ssh-add ~/.ssh/id_rsa` once).  sbatch on partition `ai`, account
  `bouman`, --cpus-per-task=14 per GPU.  mbirtorch scratch checkout:
  `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`;
  TORCHPY=`/scratch/gautschi/buzzard/torch_p0/env/bin/python`.  Slurm
  `--export` splits on commas — pass env via the submission shell.
- Concurrent sessions may be active (Charlie's, on the utility API and
  docs).  The mbirjax_metrics repo is yours this round; coordinate only on
  `results/` pushes, which the wrappers already handle by rebase-retry.
- Terminology: "variants" (never arms/cells for variant sets).
