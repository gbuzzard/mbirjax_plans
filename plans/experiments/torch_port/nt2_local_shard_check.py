"""Local n>1 path check on CPU virtual devices — the multi-device ablation the Mac CAN run.

Exercises exactly the code the cluster n>1 rows use: pin to several cpu devices, place the
inputs as Shards, run each op, read the peak, and fingerprint the gathered result.  Its
target is the Shards seam, which the n=1 path never reaches.

WHY IT EXISTS.  The first 4-GPU trial (job 15004425) lost every one of its 32 n>1 rows to
one line of the writer: Shards.gather() already returns numpy, and to_numpy detached it
again.  Single-device verification cannot see that line, and a GPU allocation is a slow,
expensive way to learn it.  Virtual cpu devices reproduce the whole seam in about a minute.
Run this before any cluster n>1 submission.

It also checks the values: each op's n=2 fingerprint must match its n=1 fingerprint within
the op's own tolerance, which is the cross-device correctness reference in miniature.

Usage (an interpreter carrying mbirtorch, torch, ruamel.yaml and matplotlib):
    PYTHONPATH=<mbirtorch checkout> python nt2_local_shard_check.py
Exit 0 = every op passes at n=1 and n=2.
"""
import os, sys, tempfile
M = "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics/tooling/scaling_tests"
sys.path.insert(0, M)
import torch_backend_writer as twb
import scaling_common as sc

out_dir = tempfile.mkdtemp()
cfg = twb.build_config("cpu-torch", out_dir, "20260808", "local", ".", [1, 2], gate=False)
cfg.warmup = 0
size_label = "64x56x48"
fails = []
for geom, op in [("parallel", "direct_filter"), ("parallel", "forward"),
                 ("parallel", "back"), ("parallel", "vcd_nonconst")]:
    fd, out_f = tempfile.mkstemp(suffix=".yaml"); os.close(fd)
    try:
        res = twb.measure_cell_group(cfg, geom, op, size_label, [1, 2], "cpu-torch", out_f)
    except Exception as e:
        fails.append(f"{geom}/{op}: raised {type(e).__name__}: {e}"); continue
    rows = {int(r["n_devices"]): r for r in res["rows"]}
    for f in res.get("failures") or []:
        fails.append(f"{geom}/{op} n={f['n_devices']}: {f['error'][:120]}")
    if 1 in rows and 2 in rows:
        fp1, fp2 = rows[1]["fingerprint"], rows[2]["fingerprint"]
        rtol = cfg.fp_rtol_iter if op == "vcd_nonconst" else cfg.fp_rtol_single
        rd = abs(fp2["sum"] - fp1["sum"]) / (abs(fp1["sum"]) or 1.0)
        ok = rd <= rtol
        print(f"  {'ok ' if ok else 'BAD'} {geom}/{op:14s} n=1 vs n=2 sum reldiff {rd:.2e} (rtol {rtol:g})"
              f"  devices={rows[2]['devices']} sharded={rows[2]['is_sharded']} pad0={fp2['padding_zero']}")
        if not ok:
            fails.append(f"{geom}/{op}: cross-count reldiff {rd:.2e} > {rtol:g}")
    else:
        fails.append(f"{geom}/{op}: missing rows (have n={sorted(rows)})")
print()
print("LOCAL n>1 CHECK:", "FAILURES: " + "; ".join(fails) if fails else "all ops pass at n=1 and n=2")
sys.exit(1 if fails else 0)
