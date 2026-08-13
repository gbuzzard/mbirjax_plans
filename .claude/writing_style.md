# Writing style (chats, findings pages, reports, captions)

Goal: writing that is easy to parse on first read.  Optimize reader effort,
not word count.  This style is inspired by Michael Alley's *The Craft of
Scientific Writing*.

Where it applies.  Apply this style most strictly to durable records: findings
pages, reports, figure captions, and plan documents.  Apply it equally to the
summary that closes a chain of work — the message that explains what was found
after a long tool or analysis sequence.  Intermediate narration during such a
chain does not need this level of care; short status lines are fine there.

**Write plan entries and status reports in plain English.**
Technical terms are fine when they are
broadly understood by the community or defined where they are first
used.  Do not use invented or undefined jargon (for example, "merge
hygiene") or any plan notation.  The same rule applies to code comments.

**No metaphors or idioms in technical statements — state the literal
fact.**  (Charlie, 2026-08-10.  Example of the failure: describing a
helper function in the same repository as "sitting next door.")

**Do not use a word without fully understanding its precise meaning.**
(Charlie, 2026-08-10.  Example of the failure: a "fresh" thread pool.
"Fresh" applies to things whose quality degrades with time, so "stale
information" is legitimate and "fresh pool" is not.  Say the literal
fact: a new ThreadPoolExecutor is constructed, used once, and
destroyed.)

## Structure

* Order sections and paragraphs as a conceptual narrative: the big idea
  first, then support and detail.
* Start each paragraph with a topic sentence stating its one idea.  A
  sentence that introduces a new idea starts a new paragraph.

## Sentences

* One idea per sentence, at most two.
* State a result in one sentence.  Interpret it in the next ("These results
  indicate...").
* Give lists their own sentence: a complete clause, a colon, then the items.
* Link each sentence to the previous one, usually by reusing a word from it,
  as you would use a variable from a previous line of code.
* Use one term per object, consistently, as you would name a variable.
* A needed qualifier gets its own sentence; do not drop it and do not embed it.

## Warning signs

Treat each of these as a signal to split or restructure:

* an em-dash aside or parenthetical list inside a claim;
* a sentence carrying both evidence and interpretation;
* a colon chaining two complete clauses;
* a sentence much past thirty words.

## Process

* Before drafting any durable record, reread this file.  Style instructions
  read hours earlier decay over a long session; rereading resets them.
* After drafting, make a revision pass: reread each sentence alone and apply
  the warning signs.
* When a findings page gets a panel review, include a dedicated **style
  reviewer** alongside the accuracy and reasoning reviewers.  Its charge: check
  the page against the warning signs above and the example below, and report
  the specific sentences that need splitting or restructuring.

## Example

Instead of:

> A matched synthetic case — the same geometry class, a BGA-like phantom,
> realistic noise — shows no artifact anywhere on a 15-variant grid
> (sharpness 0–3, snr_db 25–45): every score sits at the floor — the
> fingerprint of the unique minimizer.

Write:

> In contrast to the real data, the streaks did not appear in the synthetic
> example.  The synthetic example matched the real data in many ways: the same
> geometry class, a BGA-like phantom, and realistic noise.  Reconstructions
> ran over a grid of conditions: sharpness 0–3, snr_db 25–45.  In every
> condition the severity score sat at the z-uncorrelated floor.  These results
> indicate that the score does not report false positives, at least in this
> case.

The second version is longer, and it takes less time to read and understand.
That trade is the point.

# A longer, more structured example

Instead of:

## Executive summary

Every device-allocating entry point follows one of two rules, chosen by
what the entry holds per device.  Preprocessing entries stream bounded
view batches, so they run on all visible CUDA devices with no memory
preflight, capped by the `MBIRTORCH_NUM_DEVICES` pin or by an explicit
`devices=` argument.  Reconstruction entries hold full arrays resident,
so they take the automatic device policy, and the layout the policy
chooses is settled once per model and kept.  A settled layout never
changes because free memory moved.  It is re-decided only when the
model's shapes change, and `configure_devices` stays the explicit
override.

This answers the floors question your `fdk_recon` commit (`72208bb`)
left open.  The VCD-calibrated floors and the full-recon ledger govern
every reconstruction entry, direct recons included.  The settled layout
must serve the model's lifetime, and a later `recon` is its largest
workload, so each entry prices the model rather than the call.

The work is nine gated increments.  The first builds the settled state.
Most of the rest add the policy call where it is missing: `fbp_recon`,
`recon_plastic_metal`, `generate_demo_data`, and the three full-sinogram
helpers that today place a whole sinogram on one device before the model
has widened.  The denoiser also joins, but only after it gets a ledger
shape and floors of its own, which is the one cluster measurement in the
plan.  The only signature change in the package is three preprocessing
functions gaining the `devices=` parameter `scan_to_sino` already has.
Data crosses from preprocessing to reconstruction through host memory,
which is the boundary the code already uses; at production sizes that
transfer costs seconds against reconstruction minutes to hours.

Write: 

## Executive summary 

Every entry point function (e.g., `recon` or `get_sino_and_model`) follows one of two rules (policies).  
 - Preprocessing functions work on view
batches, so mapping to multiple devices is simple.  Each such function uses 
all visible CUDA devices by default.  This default can 
be overridden in two ways: by setting the env variable `MBIRTORCH_NUM_DEVICES` in advance or 
by passing an explicit `devices=` argument.  
 - Reconstruction-related functions use a policy that depends on geometry and shape.  This policy is determined 
in `_apply_device_policy`.  The device choice is 
settled once per model and re-decided only when
the model's shapes change.  `ct_model.configure_devices()` can be used
to override the policy to set a layout explicitly.

This approach is designed to 
keep the code simple and minimize data movement.  

Two consequences for users:  
- The public API barely
changes.  Three preprocessing functions gain the `devices=` argument that
`scan_to_sino` already has, and two others change only their default. 
- The path from preprocessing to reconstruction is via host memory.
Preprocessing writes its result to host memory, and the reconstruction
moves that result to the devices it chose.  This is exactly what the code
does today.  At production sizes the transfer takes seconds, while a
reconstruction takes minutes to hours.

The work has nine increments, each reviewed before the next starts.
 - Increment 1 implements the once-per-model rule in
   `_apply_device_policy`.  No entry point function changes in this increment.
 - Increments 2 through 5 add the call to `_apply_device_policy` to the functions that lack
   it: direct reconstructions, `recon_plastic_metal`,
   `generate_demo_data`, and four helpers that allocate full-size arrays.
   One helper, `gen_weights`, first needs a per-shard form, because its
   arithmetic cannot accept a sharded sinogram today.
 - Increment 6 develops a policy for `QGGMRFDenoiser.denoise` and then implements it.  
 - Increments 7 through 9 update the preprocessing defaults and the
   documentation.