# mbirtorch viewer — build findings

**Status:** AS BUILT (2026-08-05); field-tested by Greg through four rounds of
interactive review.
**Basis:** `mbirtorch_viewer_build.md` (the build spec) and
`slice_viewer_eval.md` (the evaluation it executes).
**Code:** `mbirtorch/viewer.py` (2416 lines, package-independent) and
`mbirtorch/view_utils.py` (136 lines, the mbirtorch-side wrapper).
**Tests:** 165 viewer test items across three files; the full mbirtorch suite
passes at 227 passed, 1 skipped.

## Summary

The restructured slice viewer is built, integrated, and in daily-usable shape
in mbirtorch.  The public `mbirtorch.slice_viewer` matches the mbirjax
signature and feature set.  The four architecture properties from the
evaluation all hold: a pure-numpy model layer, no import-time side effects,
no easygui, and blitting for slice and contrast updates.  Field testing
forced one significant design revision.  The spec's matplotlib-native dialog
layer proved materially worse than native dialogs in use, so dialogs now
resolve through a native-first chain with in-figure fallbacks.  Field testing
also surfaced three matplotlib 3.11 macosx problems that the viewer now works
around.  They are recorded below because a later mbirjax retrofit will meet
them too.

## What was built

The viewer is two layers in one file.  `VolumeStack` is the model: volumes,
axis permutations, a shared master slice index with proportional per-volume
mapping, the intensity range, difference state, ROI statistics, and file
loading.  It imports numpy only, so every behavior is unit-testable headlessly.
`SliceViewer` is the matplotlib view/controller on top: it holds no data
logic, and pyplot is imported only at first construction.

The controller follows the evaluation's target structure.  There is one
`button_press` dispatcher driven by a five-state mode enum: IDLE, DRAW_ROI,
MOVE_ROI, RESIZE_ROI, and SELECT_COMPARISON.  Tooltips have a single owner
and are created exactly once.  All refreshes are in-place; no code path tears
down and rebuilds panels.  Zoom sync is one parameterized implementation, and
the user setting (`sync_limits`) is separate from the reentrancy latch
(`_in_sync_callback`).  All interaction state is initialized before any
callback is wired.  Construction never shows the window; `show(block=...)`
displays it, and `slice_viewer` returns the viewer object with a module
registry as the nonblocking keep-alive.

Blitting uses partial redraws rather than matplotlib's animated-artist
pattern.  A partial redraw paints an opaque rectangle over the affected
panel's region, redraws that axes and its colorbar in place, and blits the
region.  Nothing is marked animated, so `savefig` and the toolbar save button
need no special casing.  The fast path is enabled on Agg and TkAgg and falls
back to full redraws elsewhere; the reasons are in the field-problems section.

The wrapper supplies the mbirtorch-specific boundary conversions.  Torch
tensors, including CUDA and MPS tensors, convert via duck-typed
`detach().cpu().numpy()`, so the wrapper never imports torch.  Rich data
dicts serialize to display strings with stdlib `pprint`; mbirjax uses
ruamel-yaml here, but mbirtorch carries no yaml dependency, and the contract
is display-only.  The h5 save function is injectable (`save_fn`); the built-in
writer produces the mbirjax-compatible layout of one named dataset with
string attributes.  `slice_viewer`, `SliceViewer`, and `VolumeStack` export
from `mbirtorch.__init__` through a PEP 562 lazy hook.  A headless
`import mbirtorch` under `python -W error` is silent and loads no matplotlib
module; a subprocess test enforces this.

## Feature parity with the reference viewer

Every feature exercised by mbirjax call sites is present.  The table lists
the checklist and where each item is verified.

| Feature | Status | Verified by |
|---|---|---|
| Multi-volume panels with colorbars | parity | `test_panels_match_model` |
| Slice slider, `valstep=1` | parity plus proportional fix | `TestSliceMapping`, `test_slice_slider_updates_images` |
| Single-slice volumes | slider hidden (spec-sanctioned) | `test_single_slice_slider_hidden` |
| Intensity range slider | parity | `test_intensity_slider_sets_clim` |
| Exact range entry | revised dialog (deviation 4) | `TestRangeDialog` |
| Zoom/pan couple and decouple | parity | `TestZoomSync` |
| ROI circle with live stats | parity plus trailing-edge fix | `TestRoi` |
| Difference and error images | parity plus reorientation (deviation 5) | `TestDifference`, model tests |
| Axis radios, coupled and decoupled | parity | `TestAxisControls` |
| Transpose | parity | `test_transpose_action` |
| npy/npz/h5 load, 4D branch | parity | `TestFileLoad`, `TestFileDialogs` |
| h5 save | parity except dict editing (deviation 3) | `TestFileDialogs` |
| Data-dict display | parity | `TestDictDialogs` |
| Right-click context menu | parity via new mechanism (deviation 2) | `TestContextMenu` |
| Tooltips | parity | `test_tooltip_hover`, `test_tooltips_constructed_exactly_once` |
| Escape cleanup, help overlay | parity | `test_escape_clears_roi`, `TestResetAndHelp` |
| Nonblocking mode and registry | parity, and the viewer is returned | `TestShowAndWrapper`, `TestWrapperBehavior` |
| 2D promotion, None placeholders | parity | `TestNormalization` |

## Latent-bug checklist

The spec forbade reproducing seven mbirjax latent bugs.  Each is confirmed
absent by a test or by construction.

| Reference bug | Treatment here | Evidence |
|---|---|---|
| `_update_slice` identity arithmetic; shallow volumes clip | Proportional mapping from a master index | `test_shallow_volume_maps_proportionally_not_clipped` |
| `_display_mean` missing `continue` | Loop skips None circles | `test_stats_survive_none_circle` |
| Throttle drops the final stats update | Trailing one-shot timer plus force-on-release | `test_trailing_stats_update_fires` |
| Tooltips built twice, first set orphaned | Single construction site | `test_tooltips_constructed_exactly_once` |
| `_update_axis` dead in-place block | No rebuild path exists; `refresh()` is the only update | by construction |
| `axes_perms[0]` written into every title | Titles use the per-volume index | `test_titles_show_per_volume_perms` |
| `_syncing_axes` double-init and inverted name | One `sync_axes` setting, initialized once | `TestAxisControls`, inspection |
| Set-iteration-order permutations | `sorted({0, 1, 2} - {axis})` | `test_perm_from_slice_axis` |
| Construction-order wiring hazard | State initialized before any wiring | by construction; every test constructs |

## Deviations from the reference

Field use drove most of these; each records what changed and why.

1. **Dialogs are native-first.**  The spec called for matplotlib-native
   dialogs throughout.  Greg's field test found the in-figure file browser
   unacceptable for real directories.  File selection now tries Qt's dialog
   on Qt backends, the macOS panel via `osascript` on darwin, and
   `tkinter.filedialog` under TkAgg; the in-figure browser remains as the
   universal fallback.  The chain resolves lazily at click time, so the
   module keeps zero import-time GUI dependencies and easygui stays deleted.
2. **The context menu returned.**  The spec replaced the reference's
   right-click menu with a control strip of buttons.  Greg vetoed the strip
   after use.  It cluttered the layout, and its button text overflowed at
   small figure sizes.  Right-click now opens a menu with the reference's
   exact item list.  Under TkAgg it is a native Tk popup at the pointer; on other
   backends it is an in-figure popup at the click position.  The couple and
   decouple toggles live in this menu, as in the reference.
3. **Save-time dict editing is TkAgg-only.**  The reference edited every
   data-dict key through sequential easygui text boxes on save.  Multi-line
   editing has no safe host except Tk.  Under TkAgg a single editor window
   offers all keys at once.  Other backends save the dict unchanged.
4. **The range dialog starts blank.**  The reference prefilled the min and
   max fields, and clearing a field meant "use the data range".  Blank fields
   here keep the current bound, and an explicit Data-range button applies the
   data range.  This removes the select-all-then-retype friction that
   matplotlib text boxes cannot support.
5. **Differences reorient instead of refusing.**  The reference refused a
   difference when the two volumes' axis permutations differed.  That refusal
   trapped Greg after a transpose, with no visible cause.  The viewer now
   requires only equal original shapes and reorients the comparison volume
   into the baseline's frame with view-only transposes.
6. **Home always works.**  Stock matplotlib's Home is a silent no-op until
   a zoom or pan has pushed a view, and its saved views go stale after axis
   changes.  Home here falls back to the explicit full-extent reset when the
   navigation stack is empty, and structural changes clear the stack.
7. **Programmatic view changes hold the sync latch.**  Multi-panel restores
   (Home, Back, Forward) and per-panel resets each restore that panel's own
   extents.  Without the latch, the zoom sync stamped the last-restored
   panel's view onto all panels.  The reference shares this defect.
8. **Small input-validation improvements.**  Wrong-length `slice_axis` or
   `slice_label` lists raise a clear ValueError.  `vmin > vmax` at
   construction raises the same message the range dialog uses.  A file load
   clears any difference state on the replaced volumes.

## Field problems found and their causes

Interactive testing on Greg's Mac (macosx backend, matplotlib 3.11, Retina
display) surfaced three upstream behaviors.  A future mbirjax retrofit on
current matplotlib will meet all three.

1. **In-process Tk crashes under the macosx backend.**  Creating a
   `tkinter.Tk()` root inside the Cocoa event loop dies with SIGBUS.  No
   try/except can catch that failure.  A one-click probe confirmed this
   before it could ship.  The viewer therefore uses in-process Tk only when TkAgg is the
   backend; macOS gets its native panel through `osascript`, which runs in a
   separate process.
2. **Rubber-band zoom cancels on fast drags.**  matplotlib 3.10 added a
   guard that cancels a zoom when a motion event's `buttons` state does not
   match the pressed button.  The macosx backend fills that state from the
   live hardware value, `NSEvent.pressedMouseButtons`.  Motion events still
   queued at release therefore report no buttons, and quick zoom gestures
   died about two times in three.  The viewer patches its toolbar's
   `drag_zoom` on macosx to treat missing button state during an active zoom
   as still-held.
3. **The macosx blit path mishandles partial regions.**  The backend reports
   `supports_blit` but repaints the whole window per blit.  On Retina
   displays the renderer buffer is also twice the logical canvas size.
   Partial redraws clipped against the logical size froze the rightmost
   panel and ghosted text.  The fast path is therefore whitelisted to Agg and TkAgg,
   with regions clipped against the true renderer buffer and cleared against
   each region's previous extent.

One more interaction bug is worth recording for the retrofit.  With a
toolbar tool armed, the reference's dispatcher order swallows the click that
selects a difference comparison.  The same order initially shipped here and
cost a debugging round.  The selection check now outranks the toolbar guard.
Programmatic untoggling of the tool is not an alternative.  The macosx
backend's native buttons do not track `toolbar.zoom()` calls, so the attempt
inverted the button state.

## The mbirjax retrofit path

The viewer file drops into mbirjax unchanged; the wrapper is the only
mbirjax-side work.  The wrapper must supply three things: conversion of
inputs to numpy (the existing `np.asarray` handles jax arrays), serialization
of data dicts to strings (mbirjax already has
`TomographyModel.convert_subdicts_to_strings`), and optionally
`save_fn=mj.save_data_hdf5` for streaming saves.  The mbirjax `__init__`
should export the wrapper lazily and drop `from .viewer import *`; that
change alone delivers the headless-silent import.  The TkAgg paths (native
menu, Tk range dialog, save-time dict editor, Tk file dialog) are the ones
the ThinLinc cluster will exercise; they follow the reference's own
`window.after` deferral recipe but have not yet been field-tested over
remote X11.

## Tests and how to run them

The viewer carries 165 test items in three files: `test_viewer_model.py`
(the pure-numpy model), `test_viewer_controller.py` (construction, events,
dialogs, toolbar interplay, rendering), and `test_viewer_wrapper.py` (lazy
export, tensor shim, dict conversion).  Controller tests synthesize
`MouseEvent` and `KeyEvent` objects through `fig.canvas.callbacks` on Agg,
the technique matplotlib's own widget tests use.  Toolbar tests attach a real
`NavigationToolbar2` and drive full zoom gestures.  Layout is verified by
savefig snapshots whose panel pixels are checked for content.  The suite runs
headlessly:

```
MPLBACKEND=Agg python -m pytest tests/ -q
```

## Packaging notes

`pyproject.toml` gained `matplotlib` and `h5py`, unversioned to match
mbirjax's pinning of the same packages.  The conda env needed `h5py`
installed (3.16.0).  The demo entry point is `demo/demo_1_shepp_logan.py`
with `SHOW_SLICES = True`; it opens the ground truth phantom beside the
reconstruction with the recon dict attached.
