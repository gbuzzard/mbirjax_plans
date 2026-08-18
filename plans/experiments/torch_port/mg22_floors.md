# mg22 run record

One run.  The finding and its readings are in
`plans/torch_port/active/multigpu_findings.md` §1.22; this file holds
the run detail.  The tool is `dev_scripts/refresh_widening_floors.py`
in the mbirtorch repository, extended the same day to build multiaxis
and translation models; the job file is `mg22_floors_gautschi.sbatch`
beside this record.

## Run of 2026-08-17 (job 15327855)

* Node h017, four H100s, 3 hours 9 minutes, exit 0.  The tree was the
  merged 6d90601 tip, synced to scratch and verified by per-file md5
  before submission.
* 50 timed arms and 18 generators: the plan's 66 arm entries dedupe
  to 50 unique (family, cell, count) combinations.  Every arm
  realized its pinned count, and no arm errored.
* The shepp-logan phantom fallback never fired on a production cell;
  it exists for volumes only a few voxels deep, which only the smoke
  reconstructs.
* Arm records: one json per arm under
  `/scratch/gautschi/buzzard/torch_p3/results/mg22_floors/` (scratch,
  purge-eligible).  The numbers that drive decisions are in the
  pasted FLOORS rows and findings §1.22, which is the durable copy.
* The paste landed the same evening: all ten FLOORS rows rewritten
  with this run's provenance (commit 6d90601), the hashes re-recorded
  with the renamed forward-driver key, STALE_SINCE None, and the
  bound checksum.  `stale_note()` now returns None, which closes the
  re-bless the banded-forward removal had left owing.
* Library changes that rode the paste: `TranslationModel` declares
  `_floor_family = 'translation'`; the refresh script's
  `UNTABLED_FAMILIES` is empty again; five tests that used
  translation as the canonical unmeasured geometry now use a
  synthetic stand-in or assert the new state.  Full suite after the
  changes: 594 passed, 68 skipped.

Numbers a later reader may want, beyond the pasted rows:

* Multiaxis n=2 speedups against one device: 1.25x at the 384-class,
  0.35x at the 512-class, 1.46x at the 768-class.  The 512-class
  arm's warm repeats were 32.6 / 32.6 / 32.6 s against one device's
  11.4 / 11.4 / 11.5 s, and the reconstruction checksums agree at
  3e-8 relative, so the slowdown is real and not a values defect.
* Multiaxis n=4 against n=2: 0.87x at the 512-class, 0.23x at the
  768-class, 0.40x at the 1024-class.  The 0.40x reproduces mg18's
  0.41x production-cell reading under the full protocol.
* Translation n=2 against one device: 0.66x, 0.76x, 0.88x across its
  three production-anchored cells, rising with size.  n=4: 0.15x,
  0.41x, 0.64x.  The 14.7 percent spread on the smallest n=4 arm is
  the widest in the run; every other translation spread is under 3
  percent.
* Parallel and cone floors and brackets matched 2026-08-16 within
  the run spreads; the exact values are in the pasted rows.

Notes a later reader may want:

* The floor the tool PROPOSED for multiaxis n=2 was the 384-class
  (its smallest winning cell).  The pasted floor is the 768-class
  instead, because the tool's smallest-winning-cell rule assumes a
  monotone reading and this one is not: admitting the 384-class win
  would admit the measured three-times regression at the 512-class
  above it.  The row's note records the deviation and the reason.
* The multiaxis anomaly (0.35x at the 512-class n=2, 0.23x at the
  768-class n=4) is an open question with no probe yet.  It does not
  sort by band divisibility, so it is not §1.21's mechanism.
* The back-kernel padding remedy proposed in
  `back_remedy_design.md`, if approved, will move the cone
  crossovers at the counts whose bands are not divisible by 16; the
  staleness machinery will name the change and the next refresh
  re-anchors.  This run's cone rows describe the unpadded tree.
