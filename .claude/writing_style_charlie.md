# Writing style (chats, findings pages, reports, captions)

Goal: writing that is easy to parse on first read.  Optimize to reduce reader effort.  This style is inspired by Michael Alley's *The Craft of Scientific Writing*.

This applies to everything you write.
It particulary applies to any written communication to a user, program comments, and docstrings.

Keep all written material susinct, to-the-point, clear, and correct.
Think carefully before you write.


**Write plan entries and status reports in plain English.**
Technical terms are fine to use when their definition is understood by the broad community. If you would like to use a term that is not widely understood by the community, then you can define it.

Do not use invented or undefined jargon (for example, "merge hygiene").

Make sure to write sentences with subjects and verbs.
Do not communicate with sentence fragments and "bag of words".
"Bag of words" refers to a writting style in which an randomly ordered set of words is used to communicate an idea.

**No metaphors or idioms in technical statements — state the literal fact.**  
Example of the failure: describing a helper function in the same repository as "sitting next door."

**Do not use a word without fully understanding its precise meaning.**
Example of the failure: a "fresh" thread pool.
"Fresh" applies to things whose quality degrades with time, so "stale
information" is legitimate and "fresh pool" is not.  Say the literal
fact: a new ThreadPoolExecutor is constructed, used once, and
destroyed.

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

## Docstrings

Docstrings are ment to be read by humans, and their purpose is to explain how the function or methods is to be used, and what it does.

Docstrings should be very succinct and to-the-point.

They should not contain long meandering sentences about obscure details of the inner workings of the function or method.
