# mbirtorch viewer — build specification

**Status:** SPEC (2026-08-05); approved direction, execution delegated to a
dedicated session.
**Decision (Greg, 2026-08-05):** leave `mbirjax/viewer.py` unchanged for now.
Build the restructured viewer directly into mbirtorch.  After it has been used
for a while, retrofit mbirjax with the same module.
**Basis:** `slice_viewer_eval.md` in this directory.  That document carries the
evaluation, the line-referenced findings, and the reasoning; this one turns it
into a build plan.

## Goal

Produce `mbirtorch.slice_viewer` with the same user-facing behavior as
mbirjax's viewer, in the target architecture from the evaluation.  The target
architecture has four properties: a pure-numpy model (`VolumeStack`) separated
from the matplotlib view/controller; no import-time side effects; no easygui
(matplotlib-native dialogs); and blitting for slice and contrast updates.
The viewer module itself must import only numpy, matplotlib, and (lazily)
h5py.  This independence is the retrofit path: the identical file should later
drop into mbirjax.

## Reference sources

- Behavior reference: `mbirjax/viewer.py` (READ-ONLY; 1136 lines).  The
  public signature to preserve:
  `slice_viewer(*datasets, data_dicts=None, title='', vmin=None, vmax=None,
  slice_label=None, slice_axis=None, cmap='gray', show_instructions=True,
  block=True)`.
- Design reference: `slice_viewer_eval.md` — GUI decision (§ options),
  event-handling items 1–10, refactor recommendations 1–8.
- Demos that exercise the API: mbirjax `demo/demo_slice_viewer.py` and
  `demo/demo_10_artifacts.py` (nonblocking + multi-volume).

## Scope decisions

1. Greenfield build, not port-then-refactor.  The eval's eight-step sequence
   assumed an always-working mbirjax viewer; here the target structure is
   built directly, and mbirjax's file serves as the behavior oracle.
2. Feature parity covers what call sites actually use: multi-volume panels
   with colorbars, slice slider with proportional mapping for unequal depths,
   intensity range sliders plus exact-entry dialog, zoom/pan with
   couple/decouple sync, ROI circles with live stats, difference images,
   axis transposition, npy/npz/h5 load and h5 save, data-dict display,
   tooltips, Escape cleanup, and the nonblocking mode.
3. One deliberate UX change (eval event-item 10): the hand-rolled
   `fig.text` right-click context menu is replaced by a compact
   always-present control strip of matplotlib widgets.  Flag this at the
   stage-2 checkpoint for Greg's veto.
4. The torch shim (`detach().cpu().numpy()`), the data-dict-to-strings
   conversion, and the h5 save function are supplied by the mbirtorch-side
   wrapper.  The viewer file itself stays package-independent
   (eval recommendation 3).
5. Known mbirjax latent bugs must NOT be reproduced.  The checklist:
   `_update_slice` identity arithmetic (shallow volumes must map
   proportionally, not clip); `_display_mean` missing `continue` and the
   trailing-edge throttle drop; double tooltip construction; the
   `_update_axis` dead in-place block and its `axes_perms[0]` title bug;
   the `_syncing_axes` double-init and inverted name; set-iteration-order
   perms (use `sorted()`); construction-order wiring hazards (initialize all
   interaction state before connecting any callback).

## Stages and checkpoints

Work proceeds in four stages.  Each stage ends with a checkpoint: run the
full test suite headlessly, `git add` the new and changed files, and post a
short summary of what was built and verified.  Pause at each checkpoint for
Greg's go-ahead unless he has said to run through.

**Stage 1 — the model.**  `VolumeStack`, pure numpy: input normalization
(2D→3D, list/tuple handling, per-volume vmin/vmax/label broadcast), axis
permutations, proportional slice mapping across unequal depths, difference
and restore, ROI mask (via `np.ogrid`, no meshgrid cache) and statistics,
and file-load logic for npy/npz/h5 including the 4D branch.  Deliverable:
the class plus headless unit tests (perm round-trips, shallow-volume
mapping, difference shape/perm validation, ROI stats, load branches).
These are the first viewer tests in either repo.

**Stage 2 — the view and controller.**  Figure construction (N panels,
colorbars, sliders with `valstep=1` and single-slice hiding), ONE
`button_press` dispatcher with an interaction-mode enum (IDLE, DRAW_ROI,
MOVE_ROI, RESIZE_ROI, SELECT_COMPARISON), a single tooltip owner, one
in-place `refresh()` (no teardown/rebuild), one parameterized zoom-sync
implementation with the setting separated from the reentrancy latch,
blitting for slice and clim updates, the control strip, matplotlib-native
dialogs (TextBox and buttons) for range entry, data-dict display, array
choice, and file open/save, and a trailing-edge one-shot timer for ROI
stats.  Construction is separated from showing; `show(block=...)` displays.
Deliverable: the working viewer plus controller tests that synthesize
`MouseEvent`/`KeyEvent` through `fig.canvas.callbacks` on Agg.  Layout
verification: `savefig` snapshots on Agg.

**Stage 3 — mbirtorch integration.**  The `mbirtorch.slice_viewer` wrapper:
torch-tensor shim, data-dict string conversion, injected h5 save; lazy
export from `__init__` so headless `import mbirtorch` stays silent
(acceptance: zero warnings under `MPLBACKEND=Agg` import and pytest);
`slice_viewer` returns the viewer object, with a module registry as
keep-alive for nonblocking callers who drop it; demo_1's `SHOW_SLICES`
path switches to the viewer; docstrings adapted from mbirjax's in its
style.

**Stage 4 — parity pass and record.**  Side-by-side feature checklist
against mbirjax's viewer on identical volumes; the latent-bug
non-reproduction checklist confirmed by test or inspection; the full
mbirtorch suite green; a findings page `mbirtorch_viewer_findings.md` in
this directory (reread `.claude/writing_style.md` first); update the
Status line of `slice_viewer_eval.md` and the plans README entry.

## Working rules for the executing session

- Repos: mbirtorch is the only code target.  mbirjax is READ-ONLY reference.
  Durable docs go to `mbirjax_plans/plans/viewer/`.
- Git: `git add` only, in both repos.  Never commit; Greg commits from
  PyCharm.
- Environment: the `mbirtorch` conda env
  (`/Users/gbuzzard/miniforge3/envs/mbirtorch/bin/python`); tests via that
  env's pytest with `MPLBACKEND=Agg`.
- Narration: brief self-contained progress notes throughout, so the work
  reads clearly while streaming.
- Terminology: "variants", never "arms"; "ground truth phantom".
- Style for durable records: `mbirjax_plans/.claude/writing_style.md`.
