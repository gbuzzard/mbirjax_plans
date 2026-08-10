# torch_port subfolder reorganization — PROPOSAL

**Status: PROPOSAL ONLY.** Nothing was moved, no git command was run, and no file
outside this scratchpad was edited. Everything below was read from the trees as
they stand on 2026-08-10.

**Problem.** `plans/torch_port/` holds 32 documents in one flat directory and
`plans/experiments/torch_port/` holds 95 scripts plus three directories in
another. Both accumulated across six phases and five numbered campaign items.
The overload is real and it is concentrated: in the experiments tree, **78 of
the 95 files belong to campaigns that closed** and will never be staged again.

**Recommendation in one line.** Lifecycle-first foldering of the plans tree into
four buckets under a new README map; and in the experiments tree, folder away
the four dead series only, leaving the live `mg` series, `rows/` and `results/`
exactly where they are.

---

## 1. Inventory — `plans/torch_port/` (32 files, ~800 KB)

Role lines are from each file's own opening (status line and first paragraph);
no file was read in full.

### Live — belongs to a `current_plans.md` item that is still open

| file | size | role |
|---|---|---|
| `multigpu_plan.md` | 48K | Item 3 contract: terms, protocols, triggers, the mg1–mg8 instrument charters. "INSTRUMENTS COMPLETE, implementation in progress." §0 maps done onto remaining. |
| `multigpu_findings.md` | 48K | Item 3's live results record. Increment-1 checkpoint; every section closed; §6 carries the two tuning charters. |
| `forward_remedy_memo.md` | 23K | **OPEN DECISION.** Dashboard step 5 of item 3: driver change vs item 13's sorted stream. "Nothing here is implemented." Draft for lead then owner. |
| `entry_point_survey.md` | 39K | Read-only survey of every device-allocating entry point; design input for extending `_apply_device_policy` uniformly (owner's ruling 2026-08-10). |
| `backloop_attribution.md` | 14K | Charter B step one: the direct-recon back loop at n>1, code-reading attribution + CPU repro. Verified and ENDORSED at the increment-1 ruling. |
| `preprocess_sharding_translation_multiaxis.md` | 17K | Item 6 charter/checklist: what parallelizes now vs what must ride the item-3 campaign. Carries a public GitHub URL to itself (line 9). |
| `release_workflow.md` | 15K | Item 7 proposal, setup in progress: 8 manual steps, steps 1/2/5 done (Trusted Publishing registered on both registries). |
| `docs.md` | 40K | Sphinx port findings-for-the-code-session; §5 census is item 5's work list. Item 5 LARGELY COMPLETE. |
| `preprocessing.md` | 14K | Item 8 plan for Charlie's session: the `mbirjax.preprocess` package plus coupled main-package pieces. Item 8 LARGELY COMPLETE. |
| `array_forms_rule.md` | 2.3K | **Standing rule**, not a record: the four input forms a user-facing transform accepts and returns. Adopted 2026-08-09 with three amendments. |

### Closed — completed campaign records, still cited

| file | size | role |
|---|---|---|
| `device_policy_design.md` | 49K | Item 2 design: the memory preflight and the all-device default. Carries both review rulings. |
| `device_policy_findings.md` | 35K | Item 2 findings: the ledger implemented and measured; 387 passed local; H100 calibration in band at 5 cells. |
| `kernel_batching_design.md` | 17K | Item 1 design: kernel-aware view batching, APPROVED and executed. |
| `kernel_batching_findings.md` | 13K | Item 1 findings: the four chunk constants pinned at 128; the e-3 back-path reading attributed to compiled-reference reassociation. |
| `kernel_sharding_findings.md` | 22K | Item 14: diagnosis and repair of the Triton FORWARD under the banded drivers. COMPLETE 2026-08-08; all gates green. |
| `nightly_plan.md` | 71K | Item 4: mbirtorch in the nightly incl. multi-GPU. COMPLETE and LIVE; all seven increments shipped. |
| `projector_layer_design.md` | 12K | The projector-layer division of labor. IMPLEMENTED 2026-08-06; panel-reviewed by four reviewers. |

### Phase records — the original six-phase port

| file | size | role |
|---|---|---|
| `phase0_findings.md` | 11K | De-risking spikes; no blocker found; sized the 10–15x eager gap. |
| `phase1_findings.md` | 5.0K | Parallel-beam vertical slice; every gate passes with margin. |
| `phase2_findings.md` | 12K | Compile integration and the first gate-cell readout. |
| `phase3_findings.md` | 8.3K | Cone-beam port; parity-validated incl. helical and curved detector. |
| `phase4_design.md` | 8.3K | Multi-device design spike; the gloo/2-H100 substrate decision. |
| `phase4_findings.md` | 17K | Multi-device measurements, increments 1–3 gate-measured at n=1,2,4. |
| `phase5_findings.md` | 14K | Triton kernels COMPLETE; four kernels default-on, replacement rule passes everywhere. Most-cited of the phase pages. |
| `phase5_kernel_design.md` | 8.4K | The four Triton bodies' design; K5 sorted streams deliberately not taken. |
| `dashboard_mockup.html` | 5.1K | Static display mockup for the torch backend on the metrics dashboard (2026-08-04, port_plan's dashboard display item). |

### Review archives — written once, read for reference, never updated

| file | size | role |
|---|---|---|
| `panel_review.md` | 28K | The 30-agent, five-lens panel review after Phase 2; 24 findings confirmed, 1 refuted. |
| `prerelease_review_2026-08-10.md` | 42K | Five incoming `origin/prerelease` commits reviewed against `greg_dev` @ `a880d9c`. |
| `prerelease_review_translation_2026-08-10.md` | 25K | `1a2ced7` TranslationModel port. ISSUES, LAND NOW. |
| `prerelease_review_multiaxis_2026-08-10.md` | 50K | `16ff97c` multiaxis + `410ccf8` segmentation overlap. |
| `prerelease_review_convergent_2026-08-10.md` | 34K | Four convergent fixes; exact resolution and named cherries. |

### The program root

| file | size | role |
|---|---|---|
| `port_plan.md` | 27K | ACTIVE plan of record: motivation, the replacement decision rule, parity gates, the six-phase plan. **Linked by exact path from `mbirtorch/README.md:6`.** |

---

## 2. Inventory — `plans/experiments/torch_port/` (95 files + 3 dirs)

### Live: the `mg` series — item 3 (17 files, ~570 KB)

| file(s) | size | role |
|---|---|---|
| `mg1_readout.py` / `mg1_gautschi.sbatch` | 74K / 3.3K | THE gate readout: full n=1,2,4 matrix, both frameworks/geometries/cells, 40 measured arms + the three-region attribution instrument. **Imported by mg5 and mg7.** |
| `mg2_ledger_calib.py` / `mg2_gautschi.sbatch` | 32K / 2.9K | The ledger at n>1: modeled vs measured per-device peaks, 10 arms. Chains after mg1. |
| `mg3_value.py` / `mg3_gautschi.sbatch` / `mg3probe_gautschi.sbatch` | 72K / 5.8K / 2.3K | Shared-sinogram cross-framework residuals at n=1,2,4; 45 arms, plus the pricing probe materialized as its own chainable file. |
| `mg4_ladder.py` / `mg4_gautschi.sbatch` | 54K / 5.3K | The crossover ladder: where each device count stops paying. Feeds decision 1 (the guard floor). |
| `mg5_fwd_attrib.py` / `mg5_gautschi.sbatch` | 66K / 7.0K | Charter A: attribute the forward at n>1; the per-device view-chunk sweep. **Imports `mg1_readout`.** |
| `mg6_backloop_probe.py` / `mg6_gautschi.sbatch` | 41K / 3.0K | Charter B step one: which back-loop sub-step is the peak at n>1. Has a Mac-local results row on disk. |
| `mg7_conebatch.py` / `mg7_gautschi.sbatch` | 73K / 8.3K | Cone view-batch probe where the 2 GiB transient cap stops binding. **Imports `mg1_readout`.** |
| `mg8_geom_calib.py` / `mg8_gautschi.sbatch` | 50K / 8.6K | Memory calibration for translation and multiaxis; 12 arms; the newest file in the tree (2026-08-10 14:03). |

### Dead: four closed series (78 files)

| series | count | role |
|---|---|---|
| `p0*`–`p4*` (33) | ~170K | Phase 0–4: fan-kernel/chain-fusion/subset-graph spikes, the p2/p3/p4 gate readouts, the p4 collectives + dual-GPU substrate discriminators, and the long p4a–p4f n=4 bisection tail. |
| `p5k*` (21) | ~340K | Phase 5 Triton campaign: the kernel sweeps, gates, value attribution, tie probe, phantom link, shared gate, sino rows. |
| `kb*` (7) | ~78K | Item 1 kernel-aware view batching: the CUDA battery re-run, the view-chunk sweep, the composed five-arm gate, the e-3 attribution. |
| `dp*` (9) | ~60K | Item 2 device policy: ledger calibration, phase probe, flip gate, kernel-shard probe, the sharded-kernels standing gate. |
| `ks*` (2) | ~18K | Item 14: the Triton launch-context probes that found and bounded the forward defect. |
| `nt*` (6) | ~37K | Item 4 nightly: the pure control, the 1-GPU and 4-GPU trial sbatches and their controls, the local CPU-virtual-device shard check. |

(`p*` rows above total 54; `kb`+`dp`+`ks`+`nt` total 24; 54+24 = 78.)

### The three directories

| dir | size | status |
|---|---|---|
| `rows/` | 896K, 13 `.jsonl`/`.json` | **Tracked archive.** Named by exact path from `multigpu_findings.md:36`, `multigpu_plan.md:95`, and `forward_remedy_memo.md:9–10`. |
| `results/` | 670 MB, 30 files | **Gitignored** — `.gitignore` line `plans/experiments/torch_port/results`. Every harness defaults `RESULTS_DIR` to `dirname(__file__)/results`. |
| `__pycache__/` | 844K, 30 `.pyc` | Gitignored build residue, including `cpython-310` and `cpython-314` pairs. Delete, do not move. |

---

## 3. Proposed layout — `plans/torch_port/`

```
plans/torch_port/
├── README.md      NEW — the map (see §3.1)
├── port_plan.md   the program plan of record; STAYS PUT
├── active/        10 — what you must read today
├── closed/         7 — completed campaign records, still citable
├── phases/         9 — the original six-phase port record
└── reviews/        5 — review archives
```

Six top-level entries. 31 files move, 1 stays, 1 is new.

**Why `port_plan.md` stays at the top level.** It is the only torch_port path
quoted from the *other* repo — `mbirtorch/README.md:6` reads
`mbirjax_plans/plans/torch_port/port_plan.md`. Leaving it put costs one
top-level entry and buys a zero-edit cross-repo link. It is also the right
place for the one document that indexes all six phases.

**Why lifecycle, not topic.** A topic split (`device_policy/`, `kernels/`,
`multigpu/`, `nightly/`) would put `device_policy_design.md` — a settled record
nobody opens — at the same visual weight as `forward_remedy_memo.md`, which is
an open decision awaiting a ruling. The whole overload is that closed work and
open work sit side by side. Lifecycle separates exactly that.

**The operating rule** (state it in the README, or `active/` becomes a second
junk drawer): *a document lives in `active/` while its `current_plans.md` item
is open, and moves to `closed/` in the same commit that marks the item
COMPLETE.* Closing item 3 is then one `git mv` of five known files.

### 3.1 The new `README.md`

One page, and it is not optional — see §4.3. It must contain: the four-bucket
rule; a one-line-per-file index mapping **every filename to its folder** (so the
72 in-tree bare-name references still resolve in one lookup); and the
move-on-close ritual.

### 3.2 File-by-file mapping

| file | → destination |
|---|---|
| `port_plan.md` | *(stays at top level)* |
| `array_forms_rule.md` | `active/` |
| `backloop_attribution.md` | `active/` |
| `docs.md` | `active/` |
| `entry_point_survey.md` | `active/` |
| `forward_remedy_memo.md` | `active/` |
| `multigpu_findings.md` | `active/` |
| `multigpu_plan.md` | `active/` |
| `preprocess_sharding_translation_multiaxis.md` | `active/` |
| `preprocessing.md` | `active/` |
| `release_workflow.md` | `active/` |
| `device_policy_design.md` | `closed/` |
| `device_policy_findings.md` | `closed/` |
| `kernel_batching_design.md` | `closed/` |
| `kernel_batching_findings.md` | `closed/` |
| `kernel_sharding_findings.md` | `closed/` |
| `nightly_plan.md` | `closed/` |
| `projector_layer_design.md` | `closed/` |
| `phase0_findings.md` | `phases/` |
| `phase1_findings.md` | `phases/` |
| `phase2_findings.md` | `phases/` |
| `phase3_findings.md` | `phases/` |
| `phase4_design.md` | `phases/` |
| `phase4_findings.md` | `phases/` |
| `phase5_findings.md` | `phases/` |
| `phase5_kernel_design.md` | `phases/` |
| `dashboard_mockup.html` | `phases/` |
| `panel_review.md` | `reviews/` |
| `prerelease_review_2026-08-10.md` | `reviews/` |
| `prerelease_review_convergent_2026-08-10.md` | `reviews/` |
| `prerelease_review_multiaxis_2026-08-10.md` | `reviews/` |
| `prerelease_review_translation_2026-08-10.md` | `reviews/` |

Two calls worth naming, because a reader will question them:

- **`phase5_*` go to `phases/`, not `closed/`.** They are the most-cited phase
  pages (`phase5_findings.md` alone has 9 in-tree mentions, and
  `.claude/initial_prompt.md:27` sends a new session to it), so `closed/` was
  tempting. Consistency wins: `phases/` means "the original six-phase port
  record" and phase 5 is one of the six. A rule with an exception is a rule
  nobody can apply.
- **`backloop_attribution.md` stays in `active/`** even though mg6 confirmed it
  and the increment-1 ruling endorsed it, because item 3 is open and
  `multigpu_findings.md` §6.3 sends readers to it. Under the operating rule it
  moves to `closed/` with the rest of item 3, in one commit.

---

## 4. Proposed layout — `plans/experiments/torch_port/`

**Recommendation: keep the live series flat; folder away only the dead series.**

```
plans/experiments/torch_port/
├── README.md          NEW — the series map and the staging rule
├── mg1_*  …  mg8_*    17 files, FLAT AND UNCHANGED
├── rows/              tracked archive, UNCHANGED
├── results/           gitignored, 670 MB, UNCHANGED
└── archive/           NEW
    ├── p_phase0_4/    33
    ├── p5k_kernels/   21
    ├── kb_batching/    7
    ├── dp_devicepolicy/ 9
    ├── ks_sharding/     2
    └── nt_nightly/      6
```

Top level goes from 98 visible entries to 21 (17 mg files + `rows/` + `results/`
+ `archive/` + `README.md`). 78 files move, 17 stay.

### 4.1 Why not the full restructure

The full restructure — an `mg/` folder too — would give a five-entry top level.
It was rejected on four measured grounds:

1. **`mg5_fwd_attrib.py:182` and `mg7_conebatch.py:235` do
   `sys.path.insert(0, dirname(__file__))` and then `from mg1_readout import …`.**
   `mg5_gautschi.sbatch` calls this out in its own STAGING header: *both files
   must be staged in this directory; the import failure says so by name.* Any
   layout that separates them breaks it. An `mg/` folder happens to preserve it
   — but it is a real constraint on any further subdivision of the series, and
   it means the mg files are already a unit whether or not a folder says so.

2. **The cluster tree is flat and staged per file.** Every `*_gautschi.sbatch`
   header reads `Submit: cd /scratch/gautschi/buzzard/torch_p3 && sbatch …`,
   and protocol 11 staging is *scp'd PER FILE and md5-verified* into that one
   flat directory. Today the local tree mirrors the cluster tree for the live
   series, so a human comparing the two sees a like-for-like list. Foldering the
   live series breaks that mirror on exactly the files that are still being
   edited and staged, where a staging slip costs GPU-hours. Foldering the dead
   series costs nothing, because they are never staged again.

3. **`RESULTS_DIR` is `dirname(__file__)/results` and `results/` is gitignored
   by its exact path.** Moving a script into a subfolder silently redirects its
   local writes to a new `<subfolder>/results` that the `.gitignore` line does
   *not* cover — a 670-MB-class footgun. The mg harnesses have `MG*_RESULTS`
   env overrides and the sbatches export them, so the cluster is safe; but
   `mg6` has already been run on the Mac (`results/mg6_backloop_Gregs-MacBook-Pro-2_20260809_101018.jsonl`),
   and the whole p-series and `kb3_gate.py` have **no** env override at all.
   Under the recommended layout, `results/` does not move, no script that will
   run again moves, and **the `.gitignore` needs no change**.

4. **The overload is caused by the 78 dead files, not the 17 live ones.**
   Foldering the dead series removes 80% of the directory from view. Foldering
   `mg/` as well buys 16 further entries in exchange for touching the only files
   still in motion. Bad trade while item 3 is open — and a trivial one later:
   when item 3 closes, the same command moves the series to
   `archive/mg_multigpu/`.

### 4.2 What is deliberately NOT touched

- `rows/` — three live documents name it by exact path.
- `results/` — gitignored, 670 MB, and every harness's default write target.
- All 17 `mg*` files.

### 4.3 The README is load-bearing, not decoration

Both trees rely heavily on **bare filename** references — `phase5_findings.md`,
`device_policy_design.md`, etc., with no path. Inside `plans/torch_port/` there
are **72 such bare-name mentions** across 27 target files (top offenders:
`port_plan.md` 16, `phase5_findings.md` 9, `kernel_batching_findings.md` 7,
`device_policy_design.md` 6). None of them *breaks* — they are prose, and there
are **zero markdown hyperlinks in the entire tree** (verified: `grep -c '](' `
over `plans/torch_port/*.md` returns 0). But today "bare name" and "sibling
file" are the same thing, and after the move they are not. The README's
filename→folder index is what keeps a bare name resolvable in one lookup, and
without it the reorg trades one kind of overload for another.

---

## 5. Cross-reference cost

Every reference is a plain text path in prose or a backticked path. There is not
a single markdown link, so nothing "breaks" in a renderer — what breaks is a
reader or an agent resolving a quoted path.

### 5.1 Breaking path references — 53 across 18 files

| file | breaking refs | targets |
|---|---|---|
| `plans/current_plans.md` | 11 | kernel_batching_findings ×2, device_policy_design, multigpu_findings, preprocess_sharding… ×2, release_workflow, preprocessing, docs, kernel_sharding_findings, phase5_findings |
| `plans/torch_port/kernel_batching_findings.md` | 9 | its own design + itself + all 7 `kb*` scripts |
| `plans/torch_port/device_policy_findings.md` | 7 | its own design + itself + 5 `dp*` scripts |
| `.claude/initial_prompt.md` | 5 | multigpu_plan, kernel_batching_findings, device_policy_findings, kernel_sharding_findings, phase5_findings |
| `plans/torch_port/nightly_plan.md` | 3 | itself, `nt1_gate_control.py`, `nt1_trial.sbatch` |
| `plans/torch_port/preprocess_sharding_translation_multiaxis.md` | 2 | multigpu_plan; the GitHub URL to itself |
| `plans/torch_port/kernel_sharding_findings.md` | 2 | `ks1_launch_context.py` ×2 |
| `plans/torch_port/forward_remedy_memo.md` | 2 | phase5_kernel_design, kernel_batching_findings |
| `.claude/initial_prompt_charlie.md` | 2 | preprocessing, docs |
| `.claude/lessons.md` | 2 | device_policy_findings, `nt2_local_shard_check.py` |
| `plans/torch_port/device_policy_design.md` | 1 | itself |
| `plans/torch_port/kernel_batching_design.md` | 1 | itself |
| `plans/torch_port/port_plan.md` | 1 | phase0_findings |
| `plans/torch_port/phase0_findings.md` | 1 | `p0s{1,2,3}_*.py` + `p0_gautschi.sbatch` — **glob, manual fix** |
| `plans/torch_port/phase2_findings.md` | 1 | `p2_gate_readout.py` |
| `plans/torch_port/phase4_design.md` | 1 | `p4s1…`, `p4s2_dual_gpu.py` — second name is bare, **manual fix** |
| `plans/README.md` | 1 | + a paragraph naming 5 files by bare name, **manual rewrite** |
| `mbirtorch/docs/source/usr_api.rst` | 1 | `plans/torch_port/docs.md` |
| **total** | **53** | across **18 files** |

### 5.2 Non-breaking but worth sharpening — 5

`current_plans.md:120` ("reviews archived in `plans/torch_port/`" →
`…/reviews/`); `.claude/initial_prompt.md:90` ("findings to
`plans/torch_port/`" → `…/active/`); `.claude/initial_prompt.md:89` and
`port_plan.md:11`/`:377` (`plans/experiments/torch_port/` — still correct for
the mg series, no edit needed); `plans/README.md:42`/`:46`;
`mbirtorch/docs/source/dev_projector_kernels.rst:22` (directory reference,
still valid).

`mbirtorch/README.md:6` needs **no** edit — that is the reason `port_plan.md`
stays at the top level.

### 5.3 Soft references — 72, not fixable by sed, mitigated by the README

The bare-name mentions of §4.3. Rewriting all 72 into full paths would be a
large, noisy diff across live documents; the README index is the cheaper and
more durable fix.

### 5.4 References that CANNOT be fixed

1. **The public GitHub URL already distributed.**
   `preprocess_sharding_translation_multiaxis.md:9` publishes
   `https://github.com/gbuzzard/mbirjax_plans/blob/main/plans/torch_port/preprocess_sharding_translation_multiaxis.md`.
   The in-file copy is fixable; any copy already sent to a collaborator is not.
   GitHub does not redirect moved paths. **This is the open question of §7.**
2. **Session memory outside both repos.** Three files under
   `~/.claude/projects/…-mbirtorch/memory/` quote torch_port paths:
   `production-scale-2k-and-beyond.md:29` and
   `item3-multigpu-campaign-state.md:27` name
   `plans/torch_port/multigpu_findings.md`; `item3-…:68` names the archived
   reviews. Editable by hand, but outside any repo-level fix — and
   `item3-…:81` already records "QUEUED: torch_port folder REORG (task #17)".
3. **Cluster-side copies and logs.** `/scratch/gautschi/buzzard/torch_p3/` holds
   flat staged copies of the harnesses, plus `mg*_%j.log` job logs. Local moves
   do not affect them (the scp *destination* stays flat; only the human's scp
   *source* path changes), but nothing on the cluster can be updated from here,
   and `/scratch` is purge-eligible.
4. **Provenance paths inside the archived rows.** Every row in `rows/*.jsonl`
   carries e.g. `"path": "/scratch/gautschi/buzzard/torch_p3/results/_mg7_sino_cone_384x336x288.npy"`.
   These are already dangling and **must not be rewritten** — they are the
   record of where a measurement was actually taken.
5. **Scratchpad citations in the review archives.**
   `prerelease_review_convergent_2026-08-10.md` and
   `…_multiaxis_2026-08-10.md` cite `scratchpad/prerelease_review_multiaxis.md`,
   `scratchpad/merged2`, `scratchpad/merged_theirsB` — session-scratch paths
   that no longer exist. Already dangling; leave them.
6. **Off-repo correspondence.** `array_forms_rule.md` records that the rule was
   "emailed to Greg"; `release_workflow.md` records registrations on PyPI and
   TestPyPI. Any path quoted in that correspondence is out of reach.
7. **A live collaborator session.** `.claude/initial_prompt_charlie.md` is a
   prompt handed to another session; if that session is running with those paths
   loaded, moving the files mid-session breaks it. Sequence the move around it.

---

## 6. Executable appendix — RUN ONLY AFTER APPROVAL

All commands assume the repo root:

```bash
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_plans"
```

Do this on a branch, in one commit per step, and check the shared checkout for
other sessions' staged files first.

### 6.1 Plans tree — `git mv`

```bash
mkdir -p plans/torch_port/active plans/torch_port/closed \
         plans/torch_port/phases plans/torch_port/reviews

# active/ (10)
git mv plans/torch_port/array_forms_rule.md                          plans/torch_port/active/
git mv plans/torch_port/backloop_attribution.md                      plans/torch_port/active/
git mv plans/torch_port/docs.md                                      plans/torch_port/active/
git mv plans/torch_port/entry_point_survey.md                        plans/torch_port/active/
git mv plans/torch_port/forward_remedy_memo.md                       plans/torch_port/active/
git mv plans/torch_port/multigpu_findings.md                         plans/torch_port/active/
git mv plans/torch_port/multigpu_plan.md                             plans/torch_port/active/
git mv plans/torch_port/preprocess_sharding_translation_multiaxis.md plans/torch_port/active/
git mv plans/torch_port/preprocessing.md                             plans/torch_port/active/
git mv plans/torch_port/release_workflow.md                          plans/torch_port/active/

# closed/ (7)
git mv plans/torch_port/device_policy_design.md      plans/torch_port/closed/
git mv plans/torch_port/device_policy_findings.md    plans/torch_port/closed/
git mv plans/torch_port/kernel_batching_design.md    plans/torch_port/closed/
git mv plans/torch_port/kernel_batching_findings.md  plans/torch_port/closed/
git mv plans/torch_port/kernel_sharding_findings.md  plans/torch_port/closed/
git mv plans/torch_port/nightly_plan.md              plans/torch_port/closed/
git mv plans/torch_port/projector_layer_design.md    plans/torch_port/closed/

# phases/ (9)
git mv plans/torch_port/phase0_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase1_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase2_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase3_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase4_design.md        plans/torch_port/phases/
git mv plans/torch_port/phase4_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase5_findings.md      plans/torch_port/phases/
git mv plans/torch_port/phase5_kernel_design.md plans/torch_port/phases/
git mv plans/torch_port/dashboard_mockup.html   plans/torch_port/phases/

# reviews/ (5)
git mv plans/torch_port/panel_review.md                            plans/torch_port/reviews/
git mv plans/torch_port/prerelease_review_2026-08-10.md            plans/torch_port/reviews/
git mv plans/torch_port/prerelease_review_convergent_2026-08-10.md plans/torch_port/reviews/
git mv plans/torch_port/prerelease_review_multiaxis_2026-08-10.md  plans/torch_port/reviews/
git mv plans/torch_port/prerelease_review_translation_2026-08-10.md plans/torch_port/reviews/
```

### 6.2 Experiments tree — `git mv`

```bash
E=plans/experiments/torch_port
mkdir -p $E/archive/{p_phase0_4,p5k_kernels,kb_batching,dp_devicepolicy,ks_sharding,nt_nightly}

git mv $E/p0*  $E/p1*  $E/p2*  $E/p3*  $E/p4*  $E/archive/p_phase0_4/   # 33
git mv $E/p5k*                                 $E/archive/p5k_kernels/  # 21
git mv $E/kb*                                  $E/archive/kb_batching/  #  7
git mv $E/dp*                                  $E/archive/dp_devicepolicy/ # 9
git mv $E/ks*                                  $E/archive/ks_sharding/  #  2
git mv $E/nt*                                  $E/archive/nt_nightly/   #  6

rm -rf $E/__pycache__        # gitignored residue; NOT a git mv
```

Verify before committing: `ls -1 $E | wc -l` should print 20 (17 `mg*` +
`archive` + `results` + `rows`; `README.md` makes 21 once written).

> `results/` is **not** moved and **not** `git mv`-able — it is gitignored by
> the exact path `plans/experiments/torch_port/results`. Leave both alone.

### 6.3 Link fixes — mechanical pass (48 of the 53)

Run from the repo root; then repeat the `mbirtorch` block from that repo's root.
BSD `sed` on macOS needs the empty `-i ''`.

```bash
fix () {  # fix <basename-with-extension> <destination-subdir>
  grep -rlI --include=*.md --include=*.rst "torch_port/$1" . \
    | xargs sed -i '' "s#torch_port/$1#torch_port/$2/$1#g"
}

# plans tree
for f in array_forms_rule.md backloop_attribution.md docs.md \
         entry_point_survey.md forward_remedy_memo.md multigpu_findings.md \
         multigpu_plan.md preprocess_sharding_translation_multiaxis.md \
         preprocessing.md release_workflow.md; do fix "$f" active; done

for f in device_policy_design.md device_policy_findings.md \
         kernel_batching_design.md kernel_batching_findings.md \
         kernel_sharding_findings.md nightly_plan.md \
         projector_layer_design.md; do fix "$f" closed; done

for f in phase0_findings.md phase1_findings.md phase2_findings.md \
         phase3_findings.md phase4_design.md phase4_findings.md \
         phase5_findings.md phase5_kernel_design.md dashboard_mockup.html; \
         do fix "$f" phases; done

for f in panel_review.md prerelease_review_2026-08-10.md \
         prerelease_review_convergent_2026-08-10.md \
         prerelease_review_multiaxis_2026-08-10.md \
         prerelease_review_translation_2026-08-10.md; do fix "$f" reviews; done

# experiments tree (the scripts actually cited by name)
for f in p2_gate_readout.py p0_gautschi.sbatch p4s1_collectives_gloo.py; \
         do fix "$f" archive/p_phase0_4; done
for f in kb1_gautschi.sbatch kb2_vbsweep.py kb2_gautschi.sbatch kb3_gate.py \
         kb3_gautschi.sbatch kb4_value_attrib.py kb4_gautschi.sbatch; \
         do fix "$f" archive/kb_batching; done
for f in dp2_ledger_calib.py dp2_gautschi.sbatch dp3_phase_probe.py \
         dp3_gautschi.sbatch dp4_flip_gate.py; do fix "$f" archive/dp_devicepolicy; done
fix ks1_launch_context.py archive/ks_sharding
for f in nt1_gate_control.py nt1_trial.sbatch nt2_local_shard_check.py; \
         do fix "$f" archive/nt_nightly; done
```

Then, in the `mbirtorch` repo:

```bash
cd "/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirtorch"
sed -i '' 's#plans/torch_port/docs.md#plans/torch_port/active/docs.md#' \
    docs/source/usr_api.rst
# README.md:6 (port_plan.md) needs NO edit.
```

### 6.4 Link fixes — manual (5)

1. `plans/torch_port/phases/phase0_findings.md:6` — the glob
   `plans/experiments/torch_port/p0s{1,2,3}_*.py plus p0_gautschi.sbatch` →
   `…/torch_port/archive/p_phase0_4/p0s{1,2,3}_*.py …`.
2. `plans/torch_port/phases/phase4_design.md:7` — `p4s2_dual_gpu.py` appears
   bare after the prefixed `p4s1…`; add the folder or the full path.
3. `plans/README.md:42–52` — rewrite the torch_port paragraph for the new
   layout (it names `port_plan.md`, `phase0`–`phase3_findings.md` and
   `panel_review.md` by bare name).
4. `plans/current_plans.md:120` — "reviews archived in `plans/torch_port/`" →
   `plans/torch_port/reviews/`.
5. `.claude/initial_prompt.md:90` — "findings to `plans/torch_port/`" →
   `plans/torch_port/active/`.

### 6.5 New files

- `plans/torch_port/README.md` — the four-bucket rule, the filename→folder index
  for all 32 documents, and the move-on-close ritual.
- `plans/experiments/torch_port/README.md` — the series map (`mg` live; the six
  archived series and which item each closed), plus the two standing rules the
  reorg must not erase: *the cluster staging dir is flat*, and *mg5/mg7 import
  mg1_readout, so the mg series stays together*.

### 6.6 Verification

```bash
# no surviving reference to a moved plans document at the old flat path
grep -rnI --include=*.md --include=*.rst -E \
  "torch_port/(array_forms_rule|backloop_attribution|docs|entry_point_survey|forward_remedy_memo|multigpu_findings|multigpu_plan|preprocess_sharding_translation_multiaxis|preprocessing|release_workflow|device_policy_|kernel_|nightly_plan|projector_layer_design|phase[0-5]|dashboard_mockup|panel_review|prerelease_review)" \
  . | grep -v "torch_port/\(active\|closed\|phases\|reviews\)/"
# expect: no output

# every quoted repo-relative path still resolves
grep -rhoI --include=*.md "plans/[A-Za-z0-9_/.-]*\.\(md\|py\|sbatch\|html\)" . \
  | sort -u | while read p; do [ -e "$p" ] || echo "DANGLING: $p"; done
```

---

## 7. The one open question the owner must answer first

**Is `plans/torch_port/` a surface whose paths have been handed to anyone
outside this machine?**

The concrete evidence that it might be:
`preprocess_sharding_translation_multiaxis.md:9` publishes its own GitHub URL
(`…/blob/main/plans/torch_port/preprocess_sharding_translation_multiaxis.md`),
which reads as a link that was *sent to someone* — and
`.claude/initial_prompt_charlie.md` is a prompt handed to a separate
collaborator session that quotes `plans/torch_port/preprocessing.md` and
`plans/torch_port/docs.md` by path.

It changes the mapping, so it must be answered before any `git mv`:

- **"No, it is internal"** → execute §6 exactly as written.
- **"Yes, those URLs are out"** → the affected files (at minimum the
  preprocess/translation/multiaxis charter, likely `preprocessing.md` and
  `docs.md` too) either stay at the top level like `port_plan.md`, or move with
  a one-line stub left behind at the old path pointing to the new one. GitHub
  does not redirect moved paths, and a collaborator following a dead link gets a
  404, not a hint.

A secondary sequencing point that follows from the same answer: if Charlie's
session is live right now, run the move when it is not.
