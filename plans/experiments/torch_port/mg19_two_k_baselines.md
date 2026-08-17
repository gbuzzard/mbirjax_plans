# mg19 run record

One run.  The findings and their tables are in
`plans/torch_port/active/multigpu_findings.md` §1.20; this file holds
the run detail.

## Run of 2026-08-17 (job 15314401)

* Node h003, four H100s, 3 hours 18 minutes of wall.  Submitted the
  night before with a begin time that slipped a day because the
  submission crossed 07:00; released by hand in the morning.
* Library state: the scratch tree at commit 7cd32ed (the
  cylinder-transfer flip on translation and multiaxis; cone and
  parallel unchanged since 2026-08-11).  The sbatch's tree checks
  passed: unpadded split confirmed, all four geometries on the
  transfer default.
* Eighteen arms, all healthy: per geometry a generator, a two-device
  refusal check (both REFUSED, the expected reading), calibrated
  reconstructions at three and four devices, the four-batch sweep
  with a repeated shipped-batch arm, and for cone the two
  combining-slab arms.
* Values legs: 4.0e-6 to 9.6e-6 against the staged references, gate
  1e-4.  Same-count composed repeats: 5.4e-7 to 8.5e-7.
* Output rows:
  `plans/experiments/torch_port/rows/mg19_baselines_h003_20260817_082830.jsonl`.

Notes a later reader may want:

* The first arm at a new call shape pays the compilation the later
  arms read from the on-disk cache.  That is why cone's first
  four-device wall is 485 s where its repeat is 420 s, and why the
  batch comparisons are read on device-measured forward busy time.
* The three-device arms ran first per geometry, so their walls carry
  the compilation for their shapes; their "other" component is
  inflated by it.
* Busy columns are the busiest device's bracketed seconds per
  direction; per-device lists and sums are on the rows.
* The generators staged a 29.6 GiB phantom and a 30.5 GiB sinogram
  per geometry under
  `/scratch/gautschi/buzzard/torch_p3/results/`, kept for re-runs;
  remove by hand when no follow-up needs them.
* The parallel three-against-four fingerprint distance of 2.9e-4 is
  the compiled reduction-order latitude at the uneven view split; the
  even-count repeats sit at the float floor.
