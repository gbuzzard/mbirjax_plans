# mg18 run record

One run so far.  The finding and its tables are in
`plans/torch_port/active/multigpu_findings.md` §1.18; this file holds
the run detail.

## Run of 2026-08-17 (job 15307729)

* Node h014, four H100s, 1 hour 34 minutes of wall.
* Library state: the scratch tree at
  `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`, at local commit
  78b4f78, the same synced tree mg17 ran on.  The sbatch's own tree
  check confirmed translation and multiaxis still banded by default
  and cone and parallel already gathered.
* Output rows:
  `plans/experiments/torch_port/rows/mg18_ab_h014_20260816_231137.jsonl`
  (24 arm rows plus a summary row).  The job log carries the printed
  tables.
* Witness summary: no errors, no preflight refusals, every arm on its
  pinned count and claimed driver, torch bodies in both directions on
  every arm, both sharded axes split evenly at both counts, drift
  witnesses at 0.0 percent, warm-repeat floors 1.8e-6 (multiaxis) and
  3.2e-7 (translation).

Notes a later reader may want:

* The one-device anchors skip the memory preflight, and the skip is
  recorded on their rows.  The reason: the preflight prices a full
  reconstruction (68 GiB modeled at the multiaxis cell, within half a
  GiB of an idle budget), while the anchor runs one forward
  projection.  The anchor's measured peak was 22.9 GiB, so the skip
  changed nothing.
* The translation four-device banded forward read 3.69 s with a
  0.004 s spread over three warm passes, against 0.38 s at two
  devices and 0.36 s at one.  The reading is stable and its values
  passed, so it is the banded walk's real cost there, not a harness
  fault.  The per-arm `forward_view_charge` fields in the rows hold
  the batching detail if the mechanism is ever wanted; it becomes
  moot if the gather ships.
* The 32768-batch gather arms beat 8192 slightly at two devices on
  both geometries and on translation at four, and lost to 8192 on
  multiaxis at four while nearly doubling that arm's peak memory
  (23.2 GiB against 9.6 at batch 8192).
* Composed walls at the shipped batch, banded then gather: multiaxis
  447 to 394 s at two devices and 1141 to 951 s at four; translation
  44 to 32 s at two and 64 to 33 s at four.  mg8's banded composed
  walls at the same cells (500 and 1180 s on multiaxis) reproduce
  within the tree changes landed since.
* Artifacts: about 95 GiB under
  `/scratch/gautschi/buzzard/torch_p3/results/` (phantoms, reference
  sinograms, per-arm forward outputs, four reconstruction volumes per
  geometry).  Kept so that arm subsets can re-run against them;
  remove by hand once the rows have been read.
