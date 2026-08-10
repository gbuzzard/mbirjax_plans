# torch_port experiments — what lives where

This folder holds the measurement harnesses for the PyTorch port of mbirjax.
Each harness is a Python file plus the Slurm batch file that submits it on the
Gautschi cluster.  The folder is split into two parts.  The `mg` series is at
the top level because it is still being edited and staged.  Every closed series
is under `archive/`.

The planning and findings documents for this program are in
`plans/torch_port/`, which has its own README.

## Live: the `mg` series

The 17 `mg*` files at the top level serve item 3, the multi-device campaign,
which is open.  They stay at the top level, flat, for as long as item 3 is open.

| files | what it measures |
|---|---|
| `mg1_readout.py`, `mg1_gautschi.sbatch` | The gate readout: the full one-, two-, and four-device matrix over both frameworks, both geometries, and all cells.  40 measured arms plus the three-region attribution instrument. |
| `mg2_ledger_calib.py`, `mg2_gautschi.sbatch` | The memory ledger above one device: modeled against measured peak per device, 10 arms. |
| `mg3_value.py`, `mg3_gautschi.sbatch`, `mg3probe_gautschi.sbatch` | Cross-framework residuals from a shared sinogram, 45 arms.  The pricing probe is a separate batch file so it can be chained. |
| `mg4_ladder.py`, `mg4_gautschi.sbatch` | The crossover ladder: the point at which each device count stops paying. |
| `mg5_fwd_attrib.py`, `mg5_gautschi.sbatch` | Charter A: attribution of the forward projector above one device, with the per-device view-chunk sweep. |
| `mg6_backloop_probe.py`, `mg6_gautschi.sbatch` | Charter B step one: which back-loop sub-step holds the peak above one device. |
| `mg7_conebatch.py`, `mg7_gautschi.sbatch` | Cone-beam view-batch probe, in the range where the 2 GiB transient cap stops binding. |
| `mg8_geom_calib.py`, `mg8_gautschi.sbatch` | Memory calibration for the translation and multiaxis geometries, 12 arms. |

## Archived: six closed series

`archive/` holds 78 files from campaigns that finished.  None of them will be
staged to the cluster again.  They are kept because the findings pages cite
them by name.

| folder | files | campaign |
|---|---|---|
| `archive/p_phase0_4/` | 33 | Phases 0 through 4 of the original port: the early spikes, the phase 2, 3, and 4 gate readouts, the collectives and dual-device substrate discriminators, and the four-device bisection tail. |
| `archive/p5k_kernels/` | 21 | Phase 5, the Triton kernel campaign: the kernel sweeps, gates, value attribution, tie probe, phantom link, shared gate, and sinogram rows. |
| `archive/kb_batching/` | 7 | Item 1, kernel-aware view batching. |
| `archive/dp_devicepolicy/` | 9 | Item 2, device policy. |
| `archive/ks_sharding/` | 2 | Item 14, the Triton launch-context probes that found and bounded the forward projector defect. |
| `archive/nt_nightly/` | 6 | Item 4, the nightly run. |

## The two data folders

`rows/` holds the archived result rows that are worth keeping, as `.jsonl` and
`.json` files.  It is tracked by git.  Three live documents name it by exact
path, so it does not move.  The `path` fields inside those rows point into
`/scratch` on the cluster.  Those paths are already dangling, and they are the
record of where each measurement was actually taken, so they are never
rewritten.

`results/` is where the harnesses write.  It is about 670 MB and git ignores it
through the exact path `plans/experiments/torch_port/results`.  It does not
move.

## Staging conventions

Four conventions govern how these files reach the cluster.  Each one is a
constraint on any future rearrangement of this folder.

**The cluster staging directory is flat.**  Every `*_gautschi.sbatch` file
submits from `/scratch/gautschi/buzzard/torch_p3`, and all harnesses sit
directly in that one directory.  The top level of this folder mirrors that flat
cluster directory for the live series, so a person comparing the two sees the
same list of names.

**Files are copied one at a time and verified.**  Protocol 11 requires that
every changed file is copied with `scp` individually and then checked by md5.
No copy is made into a tree that a running job is importing from.

**`mg5` and `mg7` import `mg1_readout`.**  Both files insert their own
directory onto the Python path and then import from `mg1_readout`, so that they
reuse mg1's region definitions rather than a drifted copy.  `mg1_readout.py`
must therefore be staged in the same directory as `mg5_fwd_attrib.py` and
`mg7_conebatch.py`.  Both harnesses check this before any GPU time is spent and
say so by name if the import fails.  The mg series stays together for this
reason.

**A harness writes beside itself unless an environment variable says
otherwise.**  Each harness sets its results directory to a `results` folder next
to its own file.  The git ignore rule covers only
`plans/experiments/torch_port/results`, by that exact path.  A script run from
inside `archive/` would therefore write into an `archive/<series>/results`
folder that git does not ignore.  The eight `mg` harnesses accept an override
environment variable named `MG1_RESULTS` through `MG8_RESULTS`, and their batch
files export it.  The archived p-series and `archive/kb_batching/kb3_gate.py`
have no override.  Before re-running any archived script, set its output path
explicitly or run it from a copy at the top level.

## When item 3 closes

Move the whole `mg` series to `archive/mg_multigpu/` in the same commit that
marks item 3 COMPLETE.  The series moves as a unit, because of the
`mg1_readout` import described above.
