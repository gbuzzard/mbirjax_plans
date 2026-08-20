# mg49 — does a denoise ever get faster on more devices, and where does the sharded sweep's time go?

## Purpose

Two questions in one job.

**A, the ladder.** The QGGMRFDenoiser is held to one device by two sentinel rows
in the shipped widening-floors table. A sentinel records a device count with no
admission size: splitting lost at every size ever probed, so the automatic path
never widens a denoise for speed and only the memory preflight ever widens one.
The measured loss shrinks as the problem grows, so an admission size may sit
above the largest cell anyone has probed. This run extends the ladder to the
1280-, 1536- and 1664-class and re-anchors at the 1024-class.

**B, the attribution.** The single-device and multi-device denoisers are
different implementations. At one device the whole per-subset update is one
compiled call to `vcd_subset_denoiser`. Above one device the same update becomes
two per-device fan-outs, four 0-d scalar reductions combined on the lead device,
a step-size broadcast back, and a halo exchange per pass. So the comparison is
not "the same work, split" — it is one fused compiled kernel per subset against
a distributed sequence per subset. This run times each seam of that sequence at
the 1024-class on a host clock and a device clock, and reports how the sharded
sweep's time divides.

## The walls it is anchored against

Warm medians measured 2026-08-20 by the full floors refresh (mg48, job
15399595), read from that job's log. The ratio is warm(n1) over warm(n), so
above 1.00 means the wider count is faster; both denoiser floor rows are taken
against one device, because no smaller denoiser count is admitted.

| cell | n1 | n2 | n4 | n2/n1 | n4/n1 |
| --- | --- | --- | --- | --- | --- |
| (512, 448, 384) | 0.233 s | 0.394 s | 0.506 s | 0.592x | 0.461x |
| (768, 672, 576) | 0.664 s | 1.018 s | 1.253 s | 0.652x | 0.530x |
| (1024, 1008, 992) | 2.191 s | 3.269 s | 3.574 s | 0.670x | 0.613x |

## Why the ladder stops at the 1664-class

The speed sentinel only governs where one device can still *hold* the problem.
Above that size the memory preflight widens a denoise whatever the sentinel
says, because one device cannot take it. The denoiser's own memory ledger at one
device is a closed form — it queries no device and allocates nothing — and it
puts the boundary just above the 1664-class:

| cell | voxels | one-device peak | with the preflight's 15% margin | fits an 80 GB H100? |
| --- | --- | --- | --- | --- |
| (1024, 1008, 992) | 1.02e9 | 13.7 GB | 15.7 GB | yes |
| (1280, 1264, 1248) | 2.02e9 | 26.9 GB | 30.9 GB | yes |
| (1536, 1520, 1504) | 3.51e9 | 46.7 GB | 53.7 GB | yes |
| (1664, 1648, 1632) | 4.48e9 | 59.5 GB | 68.4 GB | yes |
| (1792, 1776, 1760) | 5.60e9 | 74.4 GB | 85.6 GB | **no** |

So the 1664-class is the last size at which the speed question has any practical
effect. With it in the ladder, a run that finds no admission anywhere has closed
the question across the sentinel's whole domain rather than leaving "maybe it
wins at some larger size" open; without it, the run can only push the open edge
upward. The figures above are quoted for the reader — the report prints the same
table read from the tree under test at run time
(`_build_memory_ledger(workload='denoise')`, and the margin read from
`_memory_ledger.layout_fits`), so nothing in the output depends on this note
staying current.

## The arms

Twenty-one arms and four staging jobs, each arm in a fresh subprocess, cheap
first.

1. `s1024_n1`, `s1024_n2`, `s1024_n4` — (1024, 1008, 992), wrapped. Question B.
2. `s1024_n1_control`, `s1024_n2_control` — the same, with no wrappers. The
   instrument-overhead check.
3. `d1024_n1/n2/n4` — (1024, 1008, 992), nothing wrapped. Re-anchors the
   recorded 0.670x and 0.613x.
4. `d1280_n1/n2/n4` — (1280, 1264, 1248).
5. `d1536_n1/n2/n4` — (1536, 1520, 1504).
6. `d1664_n1/n2/n4` — (1664, 1648, 1632). Last: the largest, and the one that
   may not fit one device.

The ladder arms are unwrapped on purpose. A plain wall measurement is the floors
protocol and nothing else, which is what makes its ratios comparable with the
recorded ones.

The protocol is the floors refresh's, copied rather than approximated: the model
is `QGGMRFDenoiser(tuple(cell))` where the cell *is* the image shape; the
placement is explicit through `configure_devices(devices=['cuda:0', ...])` at
every count, not the `MBIRTORCH_NUM_DEVICES` pin, because that is the protocol
the recorded denoiser rows were measured under; the input is a Shepp-Logan
phantom plus seeded gaussian noise at sigma 0.1; numpy's seed is reset to 13
immediately before every call; the call is `denoise(staged, sigma_noise=0.1,
max_iterations=3, stop_threshold_change_pct=0.0)`; and the timed envelope
includes the per-device synchronize and the gather. One cold pass, then three
warm repeats.

## The instrument

mg49 **imports** its paired-clock probe from `mg44_component_split` rather than
copying it: the event bookkeeping is subtle enough that a second copy would be a
second thing to get wrong. Around each named library seam the probe records the
host clock before and after, and a pair of CUDA timing events on each relevant
device's default stream. Nothing is synchronized during a call; the events are
read afterwards, on the synchronize the protocol already does.

Importing a module that configures itself from the environment has couplings,
and the script makes all of them explicit. mg44 reads `MG44_SMOKE` at import
time and its module-level `DEVICE` decides whether events are recorded at all,
so mg49 sets that variable before the import and asserts the mode matches right
after. mg44's sample store folds each sample's reporting level through its own
`REGION_LEVEL` and `COMPONENT_FANOUTS`, which name reconstruction regions, so
mg49 rebinds both to its denoiser taxonomy before any sample is taken.

### The seams

| region | seam | level |
| --- | --- | --- |
| denoise call | `QGGMRFDenoiser.denoise` | total |
| settle | `TomographyModel._apply_device_policy` | setup |
| place image | `TomographyModel._shard_recon` | setup |
| noise estimate | `QGGMRFDenoiser.estimate_image_noise_std` | setup |
| auto regularization | `TomographyModel.auto_set_regularization_params` | setup |
| sweep (sharded) | `QGGMRFDenoiser._denoise_sharded` | sweep |
| halo exchange | `_sharding.exchange_qggmrf_halos` | component |
| per-pass ell1 | `mbirtorch.denoising.image_ell1` | component |
| fan-out *worker* | `_sharding.run_per_device` | component |
| move_shard | `_sharding.move_shard` | host tally only |

Two details in that table are load-bearing.

`image_ell1` is patched at **`mbirtorch.denoising.image_ell1`**, not at
`mbirtorch._memory_ledger.image_ell1` where it is defined. `denoising.py` does
`from ._memory_ledger import image_ell1`, so the denoising module's own binding
is what the calls resolve through; patching the definition site would attach a
wrapper that nothing in the denoise path calls, and the region would report as
silently empty while looking attached. Verified by grep: that from-import is the
only one in the package, and `tomography_model`'s uses go through
`_memory_ledger.image_ell1`, so they are untouched. The `_sharding` seams are
safe for the mirror-image reason: no library module from-imports
`run_per_device`, `move_shard` or `exchange_qggmrf_halos`, so patching the
module attribute intercepts every call.

The sweep seam passes `marks_iteration=True`, which flags every sample taken
inside it. That flag is the only thing separating the sweep's components from
the setup regions.

### Two seams deliberately not taken

`denoising.vcd_subset_denoiser` and
`qggmrf.qggmrf_gradient_and_hessian_at_indices` are both handed to
`maybe_compile`. A wrapper on either would be **compiled**: dynamo would trace
the `perf_counter` and CUDA event calls and either graph-break or fail, and
either way it would change the thing being measured.

This is why the one-device arm has no internal seams at all — its whole
per-subset update *is* that one compiled call. The one-device sweep is therefore
reported as a **residual** (the whole denoise call minus the wrapped setup
regions), labeled as a residual everywhere it appears, and never as a
measurement. It bounds what the attribution can claim: the run can say precisely
where the *sharded* sweep's time goes, and can compare the two implementations
only at the total, at the setup, and at `per-pass ell1` — the one component both
paths run.

### What fires where

A one-device denoise never enters `_denoise_sharded`: `_shard_recon` returns a
plain tensor on a single-device placement, not a shard set. So the sweep region,
the halo exchange and both fan-outs are attached with zero calls at n=1, which
is correct and does not fail the arm. `per-pass ell1` is the exception — once
per pass at one device, once per shard per pass above one.

`noise estimate` resolves and never fires under this protocol: `denoise` reaches
it only when the caller supplies no `sigma_noise`, and the floors protocol
always supplies it, precisely so no arm pays a host-side estimate that does no
device work. The seam is taken anyway so a future protocol change shows up as a
region with calls rather than as silence, and the region is excluded from the
must-fire list.

## Staging

One input per cell, reused across that cell's arms. `MG49_SINO_DIR` defaults to
the mg48 floors results directory, whose name for a denoiser cell is
`_sino_denoiser_<v>x<r>x<c>.npy` — so the 1024-class file is already there and
reusing it makes this run's 1024 row directly comparable with the recorded one.
Each arm hashes what it loaded and the driver checks that every arm of a cell
read the same digest.

The three new cells write about 40 GB to scratch (8.1, 14.0 and 17.9 GB). The
script builds each noisy phantom a slab of rows at a time rather than in the
floors tool's one whole-volume expression: `randn` returns float64, so the
whole-volume form would hold roughly 145 GB of host arrays at the 1664-class for
an answer that is 18 GB. The slabs draw from one seeded `RandomState` in C
order and every operation is elementwise in float64, so the bytes are identical
to the whole-volume form and a later floors refresh that rebuilds one of these
files whole will get the same digest.

## What the report prints

1. **The ladder table** — walls at n=1/2/4, the ratios against one device, each
   cell's warm spread, and a verdict per (cell, count) under the shipped coarse
   rule: a win must clear 1.00 by more than the cell's spread *and* reach the
   admission margin, which is imported from
   `mbirtorch._widening_floors.ADMISSION_MARGIN` rather than written down. The
   1024-class row carries the recorded mg48 ratios beside it as an anchor check;
   a large disagreement there prints a warning about the ruler, not a finding
   about size. The table closes by naming the smallest cell and count that
   clears, or saying that none does and the sentinels stand.
2. **The ledger line** beneath it — the modeled one-device peak for every priced
   cell and the size at which capacity takes the decision away from the
   sentinel, both read from the tree under test.
3. **The output check** at and above 2^31 voxels — the checksum and finiteness of
   every ladder arm. The 1536- and 1664-class cells cross int32's boundary, a
   known trap class in this project; a non-finite or badly-off output there is a
   finding about the library, printed loudly, not an instrument failure.
4. **The attribution block** at the 1024-class — n=1 against n=2 and n=4, grouped
   by level, per warm call, showing device-ms on the busiest device, host-ms and
   call counts; then the residual (the sweep's own device span minus the sum of
   the component spans on that device, which is where dispatch and cross-device
   lock-step live); then the `move_shard` tally beside it.
5. **The one-device note** — that its sweep is a residual, and why.
6. **Instrument overhead** — wrapped against control at n=1 and n=2.
7. **Paste-ready observations** — one line per arm.

The exit code reports instrument health only: every planned arm produced a row,
realized its configured device count, read the same input as its siblings,
resolved every seam, ran with the calibration mode off, and exercised every
region its device count can exercise. Ratios, verdicts, finiteness, thermals:
findings, printed, never gated.

## Smoke

`MG49_SMOKE=1` collapses every cell onto one tiny stand-in on virtual CPU
devices, one iteration, one warm repeat, no arm above two devices, arm ids
unchanged. It measures nothing worth reading — there are no CUDA events there,
so every device column is empty. What it proves, and asserts as its exit code:
mg44 was imported in the matching mode, every seam resolved on every wrapped
arm, and at the two-device wrapped arm the regions `sweep (sharded)`,
`halo exchange`, `per-pass ell1`, `fan-out terms_worker` and
`fan-out apply_worker` all fired.

## Results

Run 2026-08-20, job 15402884 on h005, four H100s, 24.1 minutes, exit 0.
The rows are `rows/mg49_denoiser_h005_20260820_124825.jsonl`; the finding
is multigpu_findings.md section 1.39, which covers this run together with
mg50.  The full report is in the job log (`mg49_15402884.log`,
overwritten only by a rerun of the same name).

The ladder found no admission size.  Against one device, two devices read
0.640x, 0.654x, 0.643x and 0.657x at the 1024-, 1280-, 1536- and
1664-class, and four devices read 0.594x to 0.620x.  The ratio is flat
across a 4.4 times range in volume, so the earlier rise across the
smaller cells was a small-size effect that plateaus.  Capacity takes the
widening decision at the 1792-class, so this ladder covers every size at
which the speed sentinel has any effect, and the two sentinel rows stand.

Every ladder arm returned a finite output, and the checksums agree across
device counts at all four cells.  Two of those cells hold more than 2^31
voxels, so this also clears the integer-boundary trap for the denoiser.

The attribution found that the sharded sweep is 175 ms of a 3,401 ms call
at two devices and 190 ms at four, with setup at 1,165 ms.  That left
2,060 ms, sixty percent of the call, in no wrapped region.  The seam list
had no entry for the output gather, and that omission is this run's own
defect: the coverage of an attribution is only as complete as its seam
list, and a residual this large is the tell.  mg50 measured the missing
region directly and found it to be the whole multi-device penalty.

The instrument itself is free: the wrapped and control arms agree to
within 0.6 percent at both counts.
