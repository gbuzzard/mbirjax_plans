# mg41 run record

One run.  The finding lands in
`plans/torch_port/active/execution_overview.md` §5.4; this file holds
the run detail.

## mg41, the production-shaped comparison (job 15371081)

* 2026-08-19, node h001, one H100, 55.3 minutes, exit 0.  Every
  variant ran, realized one device, read an md5-verified sinogram,
  and bound hand-written kernels in both directions on the torch
  side.  No thermal or throttle findings.
* The question this run answers: at a production size, on one GPU,
  what does a real reconstruction cost in each library.  The
  protocol is the campaign's with one change: 15 VCD iterations
  instead of 3, with the stopping threshold disabled so every run
  does exactly 15.  Weights are the transmission-shaped exponential
  formula; the cell is the 1024-class sinogram (1024, 1008, 992).
* Both libraries reconstructed the same staged sinograms, reused by
  md5 from the reference-timing runs, so the columns differ only in
  the library.  mbirtorch ran the channel-sorted tip (c761b24 kernels
  on f9fde0a); mbirjax ran the cluster checkout at revision e99bc76,
  jax 0.10.1.
* The readings, warm median of three after a discarded cold pass,
  with the busiest-device peak:

  | geometry | library | cold s | warm median s | spread | peak GB |
  |---|---|---|---|---|---|
  | parallel | mbirtorch | 125.67 | 117.09 | 0.0% | 22.87 |
  | parallel | mbirjax | 156.09 | 139.75 | 0.3% | 48.66 |
  | cone | mbirtorch | 268.39 | 253.89 | 0.2% | 22.95 |
  | cone | mbirjax | 292.79 | 271.52 | 0.1% | 48.66 |

* Three readings carry the table.  mbirtorch is faster in both
  geometries at the production iteration count: 1.19x on parallel
  (139.75 against 117.09) and 1.07x on cone (271.52 against 253.89).
  mbirtorch holds 0.47x of mbirjax's device memory in both
  geometries.  The two libraries' answers agree at the 1e-7 class
  (worst gap 9.4e-7 relative), reduced by one shared host-side
  fingerprint on both volumes.
* The peak columns come from different instruments, named on every
  row: the torch counters reset after the cold pass, and jax's
  peak_bytes_in_use runs from process start, so the jax peak also
  covers compile-time allocations.  The recorded 3-iteration tables
  were read with the same pair, and the jax peaks here (48.66 GB)
  sit beside mg1's recorded 48.45 to 49.81 GB.
* The 15-iteration times run about 10 percent above five times the
  3-iteration times (117.09 against 5 x 21.26 on parallel torch).
  The package-default subset schedule changes across iterations, so
  per-iteration cost is not constant; the ratio is expected to be
  near five, not exactly five.
* Output rows: `rows/mg41_production_h001_20260819_141123.jsonl`
  (md5 1effe39ec14dcf2502d9c15823e96f61, verified after copy).  The
  harness and sbatch are beside this file; an Opus agent drafted
  both from the mg27 and mg1 templates, and the draft passed review
  and a two-library local smoke before submission.
