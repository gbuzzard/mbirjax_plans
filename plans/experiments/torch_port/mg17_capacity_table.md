# mg17 run record

One run so far.  The table itself and its readings are in
`plans/torch_port/active/two_k_design.md`; this file holds the run
detail.

## Run of 2026-08-16 (job 15307591)

* Node h001, one H100, 27 seconds of wall.
* Library state: the scratch tree at
  `/scratch/gautschi/buzzard/torch_p3/mbirtorch_src`, synced to local
  commit 78b4f78 before the run.  Eight files changed in the sync
  (the denoiser device-allocation commit), each copied by scp and
  verified by md5.
* Idle device budget read on h001: 78.67 GiB.  The H100's total is
  79.65 GiB, so the reading is consistent with an idle card.
* Kernel witness: cone and parallel bound Triton bodies in both
  directions.  The appendix geometries bound torch bodies in both
  directions, as expected.
* Output rows:
  `plans/experiments/torch_port/rows/mg17_capacity_h001_20260816_223415.jsonl`
  (82 rows).  The job log carries the printed tables.
* The in-script assertion that rebuilds the shipped combining charge
  from the plan fields held on every reduce phase of every row, so
  the derived variant columns match the ledger's closed forms.

Numbers quoted elsewhere and their rows:

* Cone 2048-class, today variant: 181.5 / 97.9 / 66.7 / 51.1 GiB at
  one through four devices; 27.8 GiB at eight.  Parallel reads 0.03
  to 0.04 GiB lower everywhere.
* Demand at three devices 76.7 GiB against the 78.67 GiB budget;
  demand at four devices 58.8 GiB.
* Pre-streaming variant at three devices 69.2 GiB, demand 79.6 GiB,
  which does not fit; its binding phase at three or more devices is
  the hessian's reduce sub-phase.
* 1024-class anchors: cone 22.8 / 12.9 / 7.1 GiB at one, two, and
  four devices, consistent with the peaks mg11 measured in band.
* Appendix, today variant, banded path: ma1024 68.0 / 51.8 / 46.3
  GiB and tct2k 44.9 / 39.5 / 34.0 GiB at one, two, and four
  devices.
