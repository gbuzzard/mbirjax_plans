# Pricing a translation kernel campaign

**Written 2026-08-22, the multiaxis kernel campaign's closing
increment.**  The multiaxis campaign was ruled with translation
deferred, on the condition that the multiaxis work would price a
translation campaign from evidence.  This memo is that price.  The
decision rule is unchanged: need, not elegance, and the ruling is
Greg's and Charlie's.

## What the multiaxis campaign cost

The campaign ran increments 0 through 6 in one working day,
2026-08-22, with the implementation delegated and every increment
gated on real hardware before the next began.

* **Code.**  One new kernel module (mbirtorch/triton_multiaxis.py,
  904 lines: forward and back kernels, wrappers, cost
  declarations), one test battery (about 1,000 lines; 60 tests
  pass on a GPU), the availability gates and self-checks, the
  selection hook, and the floors rows.
* **Cluster.**  About 5.8 GPU-hours in seven jobs: three test
  gates (job 15430644 at 3m21s, 15431566 at 1m34s, and the
  two-GPU 15432210 at 3m40s), the host-cost attribution (mg53,
  12m40s), the composed A/B (mg54, 34m19s), the scale and
  capacity run (mg55, 32m38s on four devices), and the full
  floors refresh (mg56, 38m51s on four devices).  The refresh's
  2.6 GPU-hours are shared maintenance: it re-measured every
  family, not only multiaxis.
* **What made it fast.**  A one-time survey of the bodies and the
  existing kernels settled the design before any code; the
  wrappers reuse the bodies' own geometry builders, so no
  projection math was rewritten; and the cone and parallel
  kernels supplied the structure, the traps, and the test
  templates.

## What transferred, and stands ready

A translation campaign starts with all of this in place.

* The Triton language shims and the shared launch-key set, the
  launch-device bracket, the first-launch compile lock, and the
  double no-compile declaration.
* The width rule (padded_kernel_width) with three recorded
  padding-safety strategies, and the poison-the-padding
  discipline.
* The _view_batch_cost contract, which the driver and the memory
  ledger both consume; mg55 measured the ledger pricing a kernel
  route to within 7 percent through it.
* The availability-gate and self-check pattern, now in its third
  instance, with per-device caches and the re-entrancy guard.
* The test templates: parity across geometry variants, banded
  seams, pixel padding, delegation, repeat semantics split by
  atomics, adjointness including the kernel-pair form, and the
  independent coverage ruler the forward inversion needed.
* The measurement harnesses: the A/B shape (mg54), the
  device-count and capacity shape (mg55), and the floors refresh,
  each parameterized by family.
* One technique worth repeating: the shipped Triton kernel source
  was executed through a torch-backed shim on a machine with no
  GPU, with the already-gated back kernel as the harness control,
  which caught value errors before any cluster time was spent.

## What the multiaxis campaign returned

The returns, all measured and recorded in multigpu_findings.md
sections 1.45 to 1.47: 4.0x to 4.6x composed speed over the
compiled bodies on one device with identical values; the largest
temporary class gone from the memory peaks (11.38 GB down to 1.96
at the 512-class); device counts that finally pay (1.81x at two
and 2.97x at four at the 1024-class) with per-device peaks that
halve at each doubling; the first 2048-class multiaxis
reconstruction, which no torch-body route can run at any device
count; and 6.4x over mbirjax at the production class.  The tile
constants are still the adopted cone values, so a tuning sweep is
recorded upside on top of these numbers.

## Where translation stands today

* **Speed.**  The production scan (256, 1900, 3000) reconstructs
  in 12.59 s on one device, 1.29x ahead of mbirjax's 16.26 s
  (mg52).  The mg56 floors admit two devices from the half-scale
  scan (1.19x) and four from the production scan (1.15x).
* **Memory.**  The torch bodies' slab behavior is the recorded
  pain: the half-scale translation peak grew from 8.2 to 27.2 GB
  between one and four devices, and the production cell peaks at
  27.22 GB on one device against mbirjax's 41.46.
* **Structure.**  Translation is a point-source geometry
  (source-detector and source-iso distances), so its vertical
  terms carry magnification.  The cone kernels are therefore the
  likely template, with the footprint translated rather than
  rotated per view.  The multiaxis campaign's one structural
  discount, a 12-byte per-(view, pixel) contract because the
  slice-to-row slope is per view, probably does not transfer:
  magnification varies per pixel, which is cone's 28-byte class.
  A half-day survey in the multiaxis campaign's style settles
  this before any kernel is written.
* **Immunity.**  The recompile-budget mechanism named the
  translation back body alongside the multiaxis one (findings
  section 1.36).  Kernel bodies are outside that mechanism by
  construction, exactly as for multiaxis.

## The price, and the triggers

**The estimate: one to two working days and roughly three to six
GPU-hours**, on the multiaxis campaign's increments with the
reuse inventory above.  The survey is the only genuinely new
design work; the routing and the measurement harnesses are the
third instance of their patterns.

**The need test is not met today.**  The production scan is
already fast, already ahead of mbirjax, and has no recorded
demand above its current size, so the capacity argument that
decided the multiaxis ruling has no recorded translation
counterpart.  Three triggers would change that, and any one of
them re-opens this memo as a campaign plan:

1. A translation workload above the current production scan, or a
   throughput requirement the 12.59 s wall does not meet.
2. A multi-device translation demand, where today's returns are
   thin (1.19x and 1.15x at the floors) and kernels would both
   raise the speedups and let the per-device peaks divide, as
   they did for multiaxis.
3. Detector growth.  The driver's own notes record panels heading
   toward 6K by 10K, where one view's torch-body slab is about
   6 GB and the view axis alone cannot bound the transient
   (projectors.py, the tuning note).  Translation scans carry
   large detectors at fixed view counts, so this pressure lands
   on translation first among the torch-body geometries.

Until a trigger fires, the recommendation is to hold: the torch
bodies remain the translation route and stay fully supported by
the kill-switch-tested fallback machinery, and item D7's
translation half stays open, to be decided inside the kernel
design when the campaign runs, as it was for multiaxis.
