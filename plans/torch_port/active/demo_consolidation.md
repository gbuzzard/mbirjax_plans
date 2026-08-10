# Demo consolidation — plan

**Status:** STARTED 2026-08-10 (Charlie's session).  Critique phase; no
demo code is written until the consolidated design is approved.

## 1. Goal and ground rules

The mbirjax demos are ported to mbirtorch as a CONSOLIDATED set, not a
straight port.  Charlie's charge: clean up the demos, simplify them, and
make them easier to understand.

Three ground rules govern the work.

1. The demos are reviewed one at a time, with a written critique for
   each, before any design is fixed.
2. No demo code is written until the consolidated design is formulated
   and Charlie approves it.
3. This file is the running record: each critique is added here as it is
   done, and the design section is filled in last.

## 2. The existing demos (mbirjax/demo/)

| # | file | lines | what it shows |
|---|---|---|---|
| 1 | demo_1_shepp_logan.py | 124 | the basic pipeline: phantom, forward project, reconstruct, view |
| 2 | demo_2_large_object.py | 166 | an object larger than the field of view |
| 3 | demo_3_cropped_center_recon.py | 117 | reconstructing a cropped center region |
| 4 | demo_4_wrong_rotation_direction.py | 108 | diagnosing a reversed rotation direction |
| 5 | demo_5_fbp_fdk.py | 104 | direct (non-iterative) FBP and FDK reconstruction |
| 6 | demo_6_qggmrf_denoiser.py | 78 | the qGGMRF denoiser on its own |
| 7 | demo_7_multiaxis_parallelbeam.py | 121 | the multiaxis parallel geometry |
| 8 | demo_8_helical_recon.py | 136 | helical cone-beam reconstruction |
| 9 | demo_9_anisotropic_voxels.py | 174 | anisotropic voxel spacing |
| 10 | demo_10_artifacts.py | 401 | a gallery of common artifacts and their causes |
| 11 | demo_slice_viewer.py | 41 | the slice viewer |

1,570 lines in all.  Each script also has a Colab notebook twin linked
from the demos documentation page.  mbirtorch currently carries a
minimal port of demo 1 only.

## 3. The critique process

The demos are reviewed in numeric order, one per sitting, Charlie and
Claude together.  Each critique answers five questions and is recorded
in section 5 below.

1. What is this demo trying to teach?
2. Does that lesson earn its own demo, or does it belong inside another?
3. What in the current script is confusing, redundant, or stale?
4. What would the simplest version of this lesson look like?
5. Verdict: keep as its own demo, merge into another, or drop.

## 4. Draft principles for the new set

These are starting points for the critiques to confirm, sharpen, or
strike.  Nothing here is decided.

- One lesson per demo, stated in one sentence at the top of the script.
- Short: a demo someone can read top to bottom in a few minutes.
- Runs to completion on a laptop CPU in a few minutes, with a note on
  what changes on a GPU.
- A first-time user can run demo 1 and see a reconstruction with no
  editing of the script.
- Shared boilerplate (phantom creation, viewing) is either repeated
  plainly or factored into the library, never into a demo-local helper
  module a reader must chase.
- The script is the primary artifact; whether Colab notebook twins are
  kept is an open question below.

## 5. Critiques (filled in as reviewed)

Not started.

## 6. Open questions, settled during the critiques

1. How many demos should the consolidated set have?
2. Keep the Colab notebook twins, or scripts only?  (Colab offers free
   GPUs and zero installation, but every notebook is a second copy to
   maintain.)
3. Do the two new geometries (translation, multiaxis) get demos in the
   first consolidated set?
4. Does the artifact gallery (demo 10, 401 lines) survive as a gallery,
   become documentation prose, or split?
5. Where do demos live in the docs: the current demos-and-FAQs page
   structure, or something simpler?

## 7. After the critiques

The consolidated design is written as section 8 of this file: the list
of new demos, each with its one-sentence lesson and rough length.
Charlie approves or amends it.  Only then does implementation start,
one demo at a time, each reviewed as it lands.
