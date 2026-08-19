# Inputs for the next memory-ledger calibration pass

The memory ledger predicts a call's peak device memory, and the
preflight refuses or widens a call on that prediction.  Its charges
are re-calibrated in passes.  This page collects the inputs that have
accumulated since the last pass, so that pass has them in one place.
Three are measured charges: the conservative back-projection charge
that opened the item, and two over-readings found later.  Two are
policy gaps, where no ledger term prices the cost at all.  Open item
C5 in `plans/open_items_v4.md` is the entry these inputs belong to.

**Probe verdicts and the ruling, 2026-08-19 (mg42a; findings §1.35;
the design and the ruling are `ledger_calibration_design.md`).**
Inputs 1 and 2 measured as one defect on the cone side (the back batch
charge) plus the forward side's deliberate covers on the parallel
side; Greg TABLED the term changes as not urgent, since the preflight
margin absorbs input 6's under-read and the over-reads cost only
headroom.  Input 3 confirmed as the anchor.  Input 4 is CLOSED WITHOUT
ACTION by ruling: no additional workloads join the ledger, and an
explicit device list remains the way to run a bare projection at a
scale one device cannot hold.  Input 5: the transient does not exist
in a single reconstruction, so no ledger term is owed; the cycle and
trial-accumulation mechanisms behind the nightly's reading were both
refuted (mg42b locally; the harness loop read directly), so the
reading is unexplained and the accuracy remedy moved to the nightly
itself as open item G4.  Input 6 below was found by the probe.

Each input below keeps the citations it came with.

## 1. The back-projection batch charge is conservative

The estimate counts four slabs where only three are live at once,
about 0.8 GB high at parallel 1024 on two devices.

*(closed/backloop_attribution.md §5.)*

## 2. Three-device arms over-read the declared band's top

The 2026-08-16 gate run found cone and parallel at three devices
over-reading the declared band's top, up to 1.417 against 1.30.

*(multigpu_findings.md §1.16.)*

## 3. The 2048-class runs read inside the band

The 2048-class runs put every ratio there between 1.10 and 1.19.

*(multigpu_findings.md §1.20.)*

## 4. Policy gap: the direct-entry preflight under-prices a large call

Found 2026-08-18.  mg35's first staging job ran a 2048-class direct
forward projection under a four-device pin on the automatic branch,
and the settle chose one device and ran out of memory assembling the
30.5 GiB sinogram beside the 29.6 GiB phantom.  The call neither
widened nor was refused.  The preflight under-prices that call at
this scale, and the automatic settle honored the wrong arithmetic;
the explicit device list was the workaround.

*(the fix note in mg35_sorted_2k.py's staging; mg35 job 15347106.)*

## 5. Policy gap: a transient peak on the lead device

Found 2026-08-19.  It came out of the nightly comparison that exposed
the harness's cumulative-watermark semantics.  The n=4 VCD arm at the
1024 class transiently pushes device 0 about 3.1 GB above the n=1
arm's own peak: a 26.6 GiB watermark against 23.4 measured alone, in
both geometries, with both batch nightlies bit-identical.  A
multi-device arm's placement briefly costs the lead device more than
running the whole problem there, and no ledger term currently names
that cost.

*(regression rows 20260818T231349Z_c761b244 (2 GPUs, interactive)
against 20260819T020141Z_950beaaf and 20260818T120327Z_64dedb87
(4 GPUs, batch).)*

## 6. Parallel at one device reads UNDER the band

Found 2026-08-19, by the calibration probe itself.  The weighted
512-class parallel reconstruction at one device models 1.96 GB against
a measured 2.10, a ratio of 0.935 against the band floor of 1.00,
reproduced in both probe runs.  The model's peak phase is the initial
dot products; the measured watermark accumulates inside the back
worker region.  Cone at one device reads 1.104, in band.  An
under-read lets a doomed run start, so this input leads the term
increment; its first discriminating step is the same arm on the
per-tap forward route, which says whether the channel-sorted kernel
brought the gap or the model always had it.

*(mg42a jobs 15376256 and 15377054; multigpu_findings.md §1.35.)*
