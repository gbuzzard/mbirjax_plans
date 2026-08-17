We're continuing the mbirtorch port in the `mbirtorch` repo, parallel checkout
to `mbirjax`; mbirjax is READ-ONLY reference. `mbirjax_plans` is parallel to
both and contains plans related to both.  `mbirjax_metrics` is the nightly
regression engine and dashboard and is also parallel to both.

This session's task: the opening list in
`mbirjax_plans/plans/open_items_v2.md`, following that file's
"Suggested execution order" note (updated 2026-08-17) — steps 1 through 4:
the nightly read, B1's attribution probe, the back-remedy design note, and
C2's thresholds.  In detail:

1. **Read the nightly** (desk work).  The 2026-08-17 session merged four
   changes together: the pixel-batch default at 32768, the combining slab
   at 256 MiB, the multiaxis two-device hold, and the banded-forward
   removal with the cylinder-transfer rename.  The first nightly on that
   tip is the regression check on all of them.  Report the gate line and
   anything that moved.  The floors re-bless should have run at merge
   time — verify, and if the staleness note still trips, say so before
   any cluster work.
2. **B1's attribution probe** (one GPU job, suggested mg21).  Instrument
   the cone back projection's interior at the 2048-class cell: per-band
   call and kernel-grid accounting, the workers' busy time against the
   combining step's, per device and per band pass.  The recorded
   hypothesis (findings §1.20) is that each band call pays the back
   kernel's full-detector-row grid, one band per slice-owner, so total
   back work grows with the count; the probe turns that reading into
   measured shares the design note can rest on.  mg19's brackets
   (`mg19_two_k_baselines.py`) are the instrument pattern; the staged
   2048-class artifacts on scratch can be reused if they still exist.
3. **The back-remedy design note** (desk work, no implementation).
   Follow the forward remedy's pattern (`closed/forward_remedy_design.md`):
   the candidate structures — a cylinder-transfer counterpart for the
   back, a band-sized kernel grid, and sorted accumulation inside the
   kernel with the mg20 counters (findings §1.19) as inputs — the
   2048-class arithmetic, the value and memory gates, and the ledger
   terms each candidate changes.  Product: a new note in
   `plans/torch_port/active/` ending in the decision Greg needs to make.
4. **C2's thresholds** (one GPU job).  Extend the floors refresh to the
   multiaxis and translation families and replace multiaxis's carried
   two-device row and four-device sentinel with measured thresholds
   (`mbirtorch/_widening_floors.py` carries both rows with their
   provenance; `dev_scripts/refresh_widening_floors.py` is the tool; the
   denoiser-family extension of 2026-08-16 is the pattern).  This can
   share a cluster window with item 2.

Do not start the back-remedy implementation; it follows Greg's ruling on
item 3's note.  Do not touch the device-policy budget-window test helper;
that repair runs as its own task in a separate session.  Update 
`mbirjax_plans/plans/open_items_v2.md` as you complete items.

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` unless Greg directs it (he commits from
PyCharm).  Shared checkouts — never `git add -A`.  No plan notation in
code or tests.  Cluster jobs are pre-authorized during the agreed
investigation.  Durable records in Alley style — reread
`.claude/writing_style.md` before drafting; plan entries and chat
summaries stay short and plain, with run detail in script comments or a
companion `.md` beside the script.  Have opus carry out well-defined
plans, then review.

Read for orientation (code and measured results over recollection or .md files):
1. `.claude/claude_prompt.md`, `.claude/lessons.md` (§2, §5, §6),
   `.claude/cluster_use.md`.
2. `plans/open_items_v2.md` — the task source, with the execution-order
   note this session follows and per-item status labels.
3. `plans/torch_port/active/two_k_design.md` — the 2048-class design
   record: the validated capacity table, the baseline verdicts (§6), and
   the combining-step ruling.
4. `plans/torch_port/active/multigpu_findings.md` §1.18 through §1.20 —
   the cylinder-transfer measurements, the width mechanism, and the
   2048-class baselines with the cone-back attribution.
5. `plans/torch_port/active/banded_forward_removal.md` — the removal and
   the cylinder-transfer name mapping (older records say "column
   gather").
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
  `--export` splits on commas — pass env via the submission shell.
- Scripts to `plans/experiments/torch_port/` (suggested prefix `mg*_`;
  mg1 through mg20 are used, so new scripts start at mg21); findings to
  `plans/torch_port/active/`, with measured rows filed under
  `plans/experiments/torch_port/rows/`.
- Concurrent sessions may be active (the test-helper repair, and
  Charlie's on the utility API and docs).  Terminology: "variants"
  (never arms/cells for variant sets); the multi-device forward's
  mechanism is the "cylinder transfer" — pre-2026-08-17 records call it
  the "column gather".
