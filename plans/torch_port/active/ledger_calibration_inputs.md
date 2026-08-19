# Inputs for the next memory-ledger calibration pass

The memory ledger predicts a call's peak device memory, and the
preflight refuses or widens a call on that prediction.  Its charges
are re-calibrated in passes.  This page collects the inputs that have
accumulated since the last pass, so that pass has them in one place.
Three are measured charges: the conservative back-projection charge
that opened the item, and two over-readings found later.  Two are
policy gaps, where no ledger term prices the cost at all.  Open item
C5 in `plans/open_items_v3.md` is the entry these inputs belong to.

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
