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
