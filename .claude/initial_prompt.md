We're continuing the mbirtorch port in the `mbirtorch` repo, parallel checkout
to `mbirjax`; mbirjax is READ-ONLY reference. `mbirjax_plans` is parallel to 
both and contains plans related to both.  `mbirjax_metrics` is the nightly 
regression engine and dashboard and is also parallel to both. 

This session's task: the first wave of
`mbirjax_plans/plans/open_items_2026_08_16.md`, following that file's
"Suggested execution order" note — items A1, B2, and B4, plus the A2
design draft.  In detail:

1. **A1 — the 2048-class capacity table** (desk work, no GPU).
   Compute per-device memory needs for 2048-class problems (the
   reference shape is (2048, 2016, 1984)) from the library's own
   ledger (`_memory_ledger.plan_from_model` and
   `estimate_peak_device_bytes`), across device counts, for cone and
   parallel first.  Model the table twice: with today's
   partial-result combining step, and with its cost removed, so the
   table shows whether A2 is a feasibility prerequisite or an
   optimization.  Product: the opening section of a new design note,
   `plans/torch_port/active/two_k_design.md` — what fits at which
   counts, and which memory term binds.
2. **B2 — the kernel-width discriminating run** (one small GPU job).
   Parallel beam, one device, the 1024-class cell, with
   `forward_project_slice_band` forced to half width (504) against
   the full-width reference.  The registered discrimination
   (`closed/forward_remedy_design.md` §12, question 1): a per-slice
   cost near 0.041 ms makes the width flatness a multi-device
   effect; near 0.082 ms makes it a kernel-width effect.  Report the
   reading and what it implies for B3; no remedy work.
3. **B4 — translation and multi-axis on the column-gather forward**
   (one two-arm job per geometry).  Both geometries still run the
   older slice-band forward, and the recorded rule is that a geometry
   switches only on its own measurement.  Compare the banded and
   column-gather forms at a production-representative cell per
   geometry; the selection point is
   `TomographyModel._column_gather_forward`, and findings §1.8 and
   §1.10 show how cone was measured.  Values gate before timing —
   and note the new lessons §2 entry: these geometries run general
   torch bodies, whose cross-device-count comparisons at uneven
   splits sit at ~6e-4 and gate at 1e-3, not 1e-5.  Deliverable: a
   measured verdict and recommendation per geometry.  Do not flip
   any default; that is a reviewed change after the numbers are read.
4. **A2 — the combining-step design draft** (no implementation).
   Draft the options for restructuring how partial back-projections
   are combined — today the owner device holds every peer's partial
   plus the running total, about 37 GB at the 2048 class, flat in
   device count.  Cover at least tree-shaped and in-place/streamed
   combining, the ledger terms each would change, and the A1 table's
   with/without comparison.  Product: a section of the same design
   note ending in the decision Greg needs to make.  Do not implement.

Do not start A3 or A4 (the 2048-class baseline runs); they follow
Greg's ruling on A2.

**IMPORTANT — workflow protocol:** stage only (`git add` by explicit file
name), never `git commit` (Greg commits from PyCharm).  Shared checkouts —
never `git add -A`.  No plan notation in code or tests.  Cluster jobs are
pre-authorized during the agreed investigation.  Durable records in Alley
style — reread `.claude/writing_style.md` before drafting.  Have opus carry out well-defined plans,
then review.  

Read for orientation (code and measured results over recollection or .md files):
1. `.claude/claude_prompt.md`, `.claude/lessons.md` (§2, §5, §6 — §2
   includes the 2026-08-16 entry on compiled cross-count differences),
   `.claude/cluster_use.md`.
2. `plans/open_items_2026_08_16.md` — the task source, with the
   execution-order note this session follows.
3. `plans/torch_port/active/multigpu_plan.md` §0 and §0a, and
   `plans/torch_port/active/greg_notes.md` — the campaign dashboard
   and Greg's ranked investigation notes behind these items.
4. `plans/torch_port/active/multigpu_findings.md` §1.16 (the
   compiled-reduction property), §6.4 and §6.5 (the 2048-class
   arithmetic and the combining-step memory premise).
5. `plans/API_specification.md` for reference.  

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
  md5 verify of every changed file.  Slurm `--export` splits on commas —
  pass env via the submission shell.
- Scripts to `plans/experiments/torch_port/` (suggested prefix `mg*_`;
  mg1 through mg16 are used, so new scripts start at mg17); findings to
  `plans/torch_port/active/`, with measured rows filed under
  `plans/experiments/torch_port/rows/`.
- Concurrent sessions may be active (Charlie's, on the utility API and
  docs).  Terminology: "variants" (never arms/cells for variant sets).
