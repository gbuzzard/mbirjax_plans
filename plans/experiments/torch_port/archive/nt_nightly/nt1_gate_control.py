"""Pure control for the _memory_is_device_peak change in performance_tracking.py.

Run it BEFORE the edit and AFTER the edit; the two dumps must be byte-identical
for every 'gpu' and 'cpu' case.  The 'gpu-torch' / 'cpu-torch' cases are expected
to differ (that is the point of the change) and are reported separately.

RESULT 2026-08-07 (nightly_plan.md increment 1).  16 cases, 355 hard and 672 soft
findings before the edit.  15 of 16 cases byte-identical after it -- all 12 REAL
cases across both platforms, plus FORCED gpu, cpu, and cpu-torch.  The one case
that changed is FORCED gpu-torch, from 0 hard / 168 soft to 168 hard / 0 soft;
every moved finding is a memory finding, the message text is otherwise unchanged,
and the false '[CPU RSS, coarse]' note is gone from all 168.

Re-run this control whenever _memory_is_device_peak or _gate_metrics changes --
notably at increment 7, when the n>1 rows arrive.

Two families of case:

  REAL   -- committed (run, prior) pairs from results/{gpu,cpu}/<branch>/, driven
            through exactly the path performance_tracking.run() uses at gate time
            (Config from the run's own config block, _find_priors, _apply_mem_window,
            gate_run).  Proves the whole gate is unchanged on real data.

  FORCED -- a synthetic pair built from a real run whose every cell's mem_mb is
            inflated past mem_hard_pct, with the platform key relabelled.  The real
            data may never exercise the memory branch, in which case a REAL-only
            control would pass trivially.  FORCED makes the branch under edit fire
            on every ok->ok cell, so the hard/soft bucket decision is actually tested.

Usage:  <python-with-ruamel> gate_control.py <out.json>
"""
import io
import json
import os
import sys
import contextlib

REPO = "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics"
sys.path.insert(0, os.path.join(REPO, "tooling", "scaling_tests"))

import scaling_common as sc          # noqa: E402
import performance_tracking as pt    # noqa: E402

# (platform, branch) sources for the REAL cases; the two with the most history.
REAL_SOURCES = [("gpu", "prerelease"), ("cpu", "prerelease"),
                ("gpu", "greg_sharding_extensions"), ("cpu", "greg_sharding_extensions")]
N_REAL_PER_SOURCE = 3     # the newest N runs of each source that have a prior

MEM_INFLATE = 1.25        # > mem_hard_pct (8%), so every ok->ok cell fires the branch
FORCED_PLATS = ["gpu", "cpu", "gpu-torch", "cpu-torch"]


def run_files(out_dir, plat):
    """Committed run files for (out_dir, plat), oldest first; _table.yaml excluded."""
    if not os.path.isdir(out_dir):
        return []
    names = sorted(n for n in os.listdir(out_dir)
                   if n.startswith(f"regression_{plat}_") and n.endswith(".yaml")
                   and not n.endswith("_table.yaml"))
    return names


def tag_of(name, plat):
    return name[len(f"regression_{plat}_"):-len(".yaml")]


def gate_one_real(out_dir, plat, name):
    """Reproduce performance_tracking.run()'s gate step for one committed run."""
    doc = sc.load_yaml(os.path.join(out_dir, name)) or {}
    cfg = pt.Config.from_dict(doc.get("config") or {})
    W = max(1, int(getattr(cfg, "mem_gate_window", 1)))
    priors = pt._find_priors(out_dir, plat, tag_of(name, plat), W)
    if not priors:
        return None
    ref = sc.load_yaml(priors[0]) or {}
    gate_result, gate_ref = ((doc, ref) if W <= 1
                             else pt._apply_mem_window(doc, ref, priors, W))
    refs = [(f"prior:{os.path.basename(priors[0])}", gate_ref)]
    return pt.gate_run(gate_result, refs, cfg)


def gate_one_forced(out_dir, plat_src, name, plat_as):
    """A synthetic pair that forces the memory branch, relabelled to plat_as.

    Built with W=1 semantics (gate_run called directly on the two dicts), because
    the rolling-min window would otherwise erase the inflation.
    """
    ref = sc.load_yaml(os.path.join(out_dir, name)) or {}
    cfg = pt.Config.from_dict(ref.get("config") or {})
    today = {**ref, "platform": plat_as, "cells": []}
    for c in ref.get("cells", []):
        c2 = dict(c)
        if c2.get("mem_mb") is not None and not c2.get("failed") and not c2.get("skipped"):
            c2["mem_mb"] = float(c2["mem_mb"]) * MEM_INFLATE
        today["cells"].append(c2)
    ref2 = {**ref, "platform": plat_as}
    return pt.gate_run(today, [("prior:forced", ref2)], cfg)


def main():
    out_path = sys.argv[1]
    cases = {}
    for plat, branch in REAL_SOURCES:
        out_dir = os.path.join(REPO, "results", plat, branch)
        names = run_files(out_dir, plat)
        picked = 0
        for name in reversed(names):                       # newest first
            if picked >= N_REAL_PER_SOURCE:
                break
            with contextlib.redirect_stdout(io.StringIO()):  # hush gate-marker notes
                g = gate_one_real(out_dir, plat, name)
            if g is None:
                continue
            cases[f"REAL|{plat}|{branch}|{name}"] = g
            picked += 1

    # FORCED: one source run, replayed under each platform label.
    src_dir = os.path.join(REPO, "results", "gpu", "prerelease")
    src_name = run_files(src_dir, "gpu")[-1]
    for plat_as in FORCED_PLATS:
        with contextlib.redirect_stdout(io.StringIO()):
            g = gate_one_forced(src_dir, "gpu", src_name, plat_as)
        cases[f"FORCED|{plat_as}|{src_name}"] = g

    with open(out_path, "w") as f:
        json.dump(cases, f, indent=1, sort_keys=True)
    n_hard = sum(len(g.get("hard", [])) for g in cases.values())
    n_soft = sum(len(g.get("soft", [])) for g in cases.values())
    print(f"wrote {out_path}: {len(cases)} cases, {n_hard} hard, {n_soft} soft")
    for k in sorted(cases):
        g = cases[k]
        print(f"  {g['result']:>4}  hard={len(g.get('hard', [])):<3} "
              f"soft={len(g.get('soft', [])):<3} {k}")


if __name__ == "__main__":
    main()
