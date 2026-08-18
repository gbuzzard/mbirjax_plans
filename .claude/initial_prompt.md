We're continuing the mbirtorch port in the `mbirtorch` repo, parallel checkout
to `mbirjax`; mbirjax is READ-ONLY reference. `mbirjax_plans` is parallel to
both and contains plans related to both.  `mbirjax_metrics` is the nightly
regression engine and dashboard and is also parallel to both.

This session's task: the opening list in
`mbirjax_plans/plans/open_items_v3.md`, following that file's
"Start here" note (ordered 2026-08-18: parallel and cone performance
first).  The item entries in v3 carry the detail; work from them, not
from this file.

Do not start the B3 remedy; the counter run is this session's item, and
acting on what it finds requires Greg's approval.  The coarser-table
proposal (item 2) likewise ends in a ruling request, not an
implementation.  The band-padding remedy is closed (B1, B2; findings
§1.23) — do not reopen it.  Update
`mbirjax_plans/plans/open_items_v3.md` as you complete items.

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` unless Greg directs it (he commits from
PyCharm).  Shared checkouts — never `git add -A`; verify staged-file
lists at report time.  No plan notation in
code or tests.  Cluster jobs are pre-authorized during the agreed
investigation.  Durable records in Alley style — reread
`.claude/writing_style.md` before drafting; plan entries and chat
summaries stay short and plain, with run detail in script comments or a
companion `.md` beside the script.  Have opus carry out well-defined
plans, then review.

Read for orientation (code and measured results over recollection or .md files):
1. `.claude/claude_prompt.md`, `.claude/lessons.md` (§2, §5, §6),
   `.claude/cluster_use.md`.
2. `plans/open_items_v3.md` — the task source, with the Start-here
   order this session follows and per-item status labels.
3. `plans/torch_port/active/multigpu_findings.md` §1.19, §1.21, and
   §1.23 — the width mechanism, its measurement on the back kernel,
   and the padding remedy's landing; §1.22 for the floors state and
   the multiaxis anomaly.
4. `plans/torch_port/active/back_remedy_design.md` — the landed
   padding design; its §6 names the counter-run precondition item 1
   tests.
5. For item 2: `mbirtorch/_widening_floors.py` (the table and its
   notes) and `dev_scripts/refresh_widening_floors.py` (the tool;
   mg22 was the last full run).
6. `plans/API_specification.md` for reference.

The nightly dashboard is live and seeding history — its rows are regression
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
  md5 verify of every changed file — and the scratch tree lags the
  repository, so sync it to the current tip before the first job.  Slurm
  `--export` splits on commas — pass env via the submission shell.  The
  torch_p3 sbatch files pip-install into the shared environment at
  start, so never run two such jobs at once; chain with
  `--dependency=afterany:<jobid>`.
- Scripts to `plans/experiments/torch_port/` (suggested prefix `mg*_`;
  mg1 through mg24 are used, so new scripts start at mg25); findings to
  `plans/torch_port/active/`, with measured rows filed under
  `plans/experiments/torch_port/rows/`.
- Concurrent sessions may be active (the test-helper repair, and
  Charlie's on the utility API and docs).  Terminology: "variants"
  (never arms/cells for variant sets); the multi-device forward's
  mechanism is the "cylinder transfer" — pre-2026-08-17 records call it
  the "column gather".  The kernel width rule: hand-written kernels
  round width-class arguments up to the next multiple of 16
  (`mbirtorch/_utils.padded_kernel_width`, landed 2026-08-18), so
  records from before that date describe the unpadded kernels.
