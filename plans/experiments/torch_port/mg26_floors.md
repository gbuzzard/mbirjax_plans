# mg26 run record

One run.  The finding is in
`plans/torch_port/active/multigpu_findings.md` §1.25; the refreshed
rows are pasted into `mbirtorch/_widening_floors.py` (staged); the
coarsening proposal that reads from them is
`plans/torch_port/active/floors_coarsening_proposal.md`.  This file
holds the run detail.

## mg26, the floors refresh on the padded tree (job 15342578)

* Node h009, four H100s, 2 hours 57 minutes of wall (mg22, the same
  refresh on the unpadded tree, ran 3 hours 9 minutes), chained behind
  mg27 on the shared environment.  Exit 0.
* The tree is the padded 64dedb8 tip, verified per-file by md5 before
  the chain was submitted.  The scratch tree is not a git checkout, so
  the tool printed `commit='unknown'` in its paste; the pasted rows
  carry `commit='64dedb8'` by hand on that verification.
* The plan was 29 (family, count, cell) rows, 66 timed variants and 17
  generators, from the current table's brackets and sentinel probes.
* The crossover verdicts, per family:
  - cone n=2: 0.874x / 1.343x / 1.599x at the 384-, 512-, 768-class
    cells; floor unchanged at the 512-class.
  - cone n=4: 1.145x / 1.597x at the 768- and 1024-class cells; THE
    FLOOR MOVED DOWN from the 1024-class to the 768-class.
  - parallel n=2: 0.746x / 1.197x / 1.565x; floor unchanged at the
    512-class.
  - parallel n=4: 0.718x / 1.055x / 1.536x; floor unchanged at the
    768-class, the table's thinnest admission.
  - multiaxis n=2: 0.350x / 1.460x / 0.797x; floor unchanged at the
    768-class, and the 1024-class reading is NEW: a measured loss
    above the floor.
  - multiaxis n=4: 0.871x / 0.233x / 0.400x against n=2; sentinel.
  - denoiser n=2: 0.555x to 0.670x; n=4: 0.446x to 0.606x; sentinels.
  - translation n=2: 0.635x to 0.885x; n=4: 0.155x to 0.655x across
    the production-anchored cells; sentinels.
* The movements match the padding's band arithmetic, stated before the
  run: only cells where a count produces a band not divisible by 16
  could move, which is cone and parallel at the 384-class (both
  counts), the 768-class at four devices, and the 1024-class at two
  and four.  The one floor that moved (cone n=4) is the one whose
  losing cell sat exactly on such a band; the sentinel families
  reproduced within run noise.
* After the paste, `--bless` named the cost inputs that moved since
  mg22 as `triton_cone.py` and `triton_parallel.py` alone -- the two
  files the padding touched -- which is the family-scoped refresh
  mode's premise read off the tool itself.
* `tests/test_widening_floors.py` passes on the pasted table (28
  tests), and the full suite reads 583 passed / 82 skipped.
* Arm records: `results/mg26_floors/` on scratch (torch_p3); the
  verdicts and the paste are in the job log
  (`mg26_15342578.log`, overwritten only by a rerun of the same
  name).
