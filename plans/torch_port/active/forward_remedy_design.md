# Design note: the two forward remedies

**Status.** GATES PASSED AND THE DEFAULT FLIPPED (2026-08-11, morning).
The combined gate campaign the §13 ruling called for ran as job mg11 and
authorized the flip on both geometries.  Findings §1.10 carries the
readings, and the flip is implemented and staged in the library.  Every
increment below is complete.  That includes increment 9's floors
re-measure, which ran on the committed flip as job 15172987.

**Prior status.** RULED (Greg, 2026-08-10, evening).  Shape C is adopted.  The
value bar is ruled: the shipped kernel-parity floor governs acceptance, and
the ruling's two conditions were verified against the rows before it was
recorded — the column gather is faster than the banded walk (busy 0.65x at
two devices and 0.52x at four; composed 67.2 to 58.4 s and 53.0 to 39.1 s)
and its memory peaks are lower (12.76 to 12.81 against 14.31 GB at two
devices; 7.21 to 7.31 against 8.03 to 8.13 at four).  The measured e-6
distance class is registered as the expectation beside the floor, so drift
toward the e-4 class surfaces even though it would still pass the floor.
The registered class is not a second threshold (Greg, same evening): a
reading beyond it calls for a judgment on the tradeoffs, and only the parity
floor fails a gate.
Question 6 records the ruling; the increments in §9 are now the work plan,
with the default flip (increment 7) gated on the standing suites as §7.2
states.  The earlier history stands below: the mg10 sweep landed on
2026-08-10 and its numbers are folded in, so no SLOT markers remain.  The
sweep overruled one position the decision memo had ruled.  Memo §8.1 proposed
fixing the parallel band at a swept knee as parallel's time remedy, and mg10
declined that shape.  §2 carries the measured verdict.

**The four answers this note owes the checkpoint.**

* **Proposed:** shape C, the pixel-batched full-height column gather for cone,
  adopted subject to this checkpoint.  §3 gives the design, §9 gives the
  increments, and §7.2 gives the value gates and the one gate question the
  checkpoint must rule on.
* **Declined:** shape P as a time remedy, on mg10's measured totals (§2.1);
  memo §3's A4, the whole-cylinder coalescing, whose 2K residency objection the
  pixel tiling dissolves (§5.2); and memo §7's cone grid variant, which the
  column gather makes moot (memo §8.4).
* **Deferred:** parallel's factor of two from one device to two, which mg10
  relocated rather than closed.  §12's first question names the one cheap
  discriminating arm, which is parallel at one device with the band knob set
  to 504.
* **Cost:** the ledger terms of §5, the 2K capacity table of §6, and the
  widening-floors refresh of §8.

**What this note is.** The decision memo
(`plans/torch_port/active/forward_remedy_memo.md`) ruled the shape of the
remedy in its §8, and §8.6 named three things the design note owes.  This note
is that design.  It does not restate the memo, and every number it takes from
the memo or from the findings carries its citation.  Nothing here is
implemented.

**Sources.** The mbirjax evidence is
`plans/torch_port/reviews/mbirjax_source_reading_2026-08-10.md`.  The measured
base is `plans/torch_port/active/multigpu_findings.md`, §1.2, §1.5, §1.6, §1.7,
§1.8, §2.1, §3.3, and §6.4.  The mg10 rows are
`plans/experiments/torch_port/rows/mg10_shape_sweep_h004_20260810_174925.jsonl`,
and findings §1.8 is their reading.  The code read for this note is
`TomographyModel._sparse_forward_project_sharded` (lines 434 to 552),
`TomographyModel._slice_band_length` and `_balanced_slice_bounds` (lines 365 to
404), `mbirtorch/_sharding.py`, `mbirtorch/projectors.py`,
`mbirtorch/triton_parallel.py`, `mbirtorch/triton_cone.py`,
`mbirtorch/_memory_ledger.py`, and `mbirtorch/_widening_floors.py`, all in the
mbirtorch checkout.

---

## 1. The finding that shapes this note

The two geometries are flat for different reasons, and the code says which.
This section states that separation first, because it decides how much each
proposed shape can be expected to buy.

**Cone's banded walk multiplies work by the device count.**
`_cone_forward_view_batch_triton` sizes its launch grid as
`(ceil(num_pixels/BLOCK_P), ceil(num_rows_r/BLOCK_R), num_views)`, and
`num_rows_r` is read from the geometry's parameters rather than from the band
(`triton_cone.py` lines 644 to 646).  Inside the kernel the band appears only
in the `in_band` predicate, which zeroes a slice tap's weight and skips its
load (`triton_cone.py` lines 557 to 571).  The channel-tap loop and its
`tl.atomic_add` run over the full detector-row tile whatever the band contains
(`triton_cone.py` lines 576 to 589).  One launch therefore costs the same
whether it is handed 1008 slices or 252.  A device that walks n bands pays n
times one full-height call, so its total work is `P * R * V * taps`.  That
expression does not contain the device count.  It predicts exactly the flatness
§1.5 measured at cone 1024, at 32.18, 30.61, and 30.48 s over one, two, and
four devices.

**Parallel's banded walk does not multiply work.**
`_parallel_forward_view_batch_triton` sizes its grid as
`(ceil(num_pixels/BLOCK_P), ceil(num_value_cols/BLOCK_R), num_views)`, and
`num_value_cols` is the band the call was handed (`triton_parallel.py` lines
449 to 452).  The atomic count is proportional to that band.  A device's total
forward work is therefore `P * S * V / n`, which falls with the device count.
Parallel's flat leg from one device to two is a loss of efficiency per unit of
work, not extra work.

These two mechanisms are different, so the two shapes below buy different
things.  Shape C removes work.  Shape P moves the same work into more launches.

**The launch arithmetic that governs shape P.**  The forward's per-device
launch count is the band count times the view-batch count.  It has that form
because `Projectors.sparse_forward_project_view_range` runs one body call per
view batch (`projectors.py` lines 386 to 394), and because the driver calls
that function once per slice-owner band per device.  Narrowing the band
multiplies the launch count by the same factor it divides the band.  The
correct reading of a band sweep is therefore the per-device total, which is
`launches(B, n) * t(B)`, and not the per-launch time `t(B)` alone.

**mg10 measured that arithmetic, and it holds.**  The sweep varied the band at
a fixed device count, which is the one thing mg9 could not do (findings §1.8).
At two devices the per-launch time is 4.70, 10.99, 17.01, 21.21, and 41.51 ms
at realized widths 63, 126, 168, 252, and 504.  The per-slice times are 0.075,
0.087, 0.101, 0.084, and 0.082 ms.  Per-launch time is therefore linear in the
band within a device count.  The per-device total moves far less than the
launch count does.  The launch count spans a factor of eight over that sweep,
from 680 to 5440, and the total spans a factor of 1.36, from 25.55 to 34.70 s.
This note's draft predicted 29.1 s for a 252-slice band at two devices before
the sweep ran, and the sweep measured 28.85 s against a control of 28.23 s.

**The doubling that mg9 showed is not a band-width effect.**  The per-slice
time at two devices runs 0.075 to 0.101 ms over realized widths from 63 to
504, with no trend in the width.  At one device it is 0.041 ms at width 1008,
which is mg9's reading (findings §1.7).  The step therefore sits between one
device and more than one device, and not between one band width and another.
§12's first question carries the relocated unknown and the arm that would
settle it.

**One consequence decides shape P.**  A narrower band buys no time, because the
launch count rises by the same factor the per-launch time falls.  §2 records
the measured verdict, which is that shape P is declined as a time remedy and
kept only as the memory knob mbirjax's own record uses it for.  mbirjax's
campaign record measured that comparison at 1024³ on one device, at 0.97x time
and 3 GB more memory for the whole-shard form
(`plans/projector_kernels/fwd_back_findings.md`).  Our sweep now produces the
same reading on our own kernel.

---

## 2. Shape P: declined as a time remedy, kept as a memory knob

### 2.1 The measured verdict

**Shape P is declined as a time remedy.**  Three of the four swept bands were
slower than the control, by 2, 6, and 23 percent.  The fourth is the one
anomaly named below.  The measured per-device busy times at parallel 1024 with
two devices are these, against a control of 28.23 s.

| asked band | realized walk | busy s | against control | per launch ms | per slice ms |
|---|---|---|---|---|---|
| control | 1 x 504 | 28.23 | 1.000x | 41.51 | 0.082 |
| 64 | 8 x 63 | 25.55 | 0.905x | 4.70 | 0.075 |
| 128 | 4 x 126 | 29.89 | 1.059x | 10.99 | 0.087 |
| 192 | 3 x 168 | 34.70 | 1.229x | 17.01 | 0.101 |
| 256 | 2 x 252 | 28.85 | 1.022x | 21.21 | 0.084 |

At four devices the asked-128 band reads 1.049x against the four-device
control.  The asked-256 band at four devices realizes the control walk itself,
because a 252-slice shard caps the request, and it reads 1.000x.  A second
collapsed pair sits at two devices, where the asked-384 arm also realizes the
asked-256 walk.  Both collapsed pairs agree to within 0.02 percent, which is
this sweep's noise floor.

**The 9.5 percent win at the 63-slice walk is real and is not built on.**  Its
checksum distance to the control is 7.80e-10 against its own repeat floor of
7.22e-10, and its sample distance is 2.38e-07 against a floor of 2.31e-07.  It
therefore sits within 8 percent of its repeat floor on both metrics, which is
as clean as this sweep can read.  It is also non-monotonic,
because the next two bands are slower, and no mechanism in the code explains
it.  It is recorded here as an observation and it is not a basis for a default.

**Shape P survives as a memory knob, which is the use mbirjax's record
supports.**  Per-device peaks at parallel 1024 with two devices fall from
12.48 GB at the control to 11.84, 11.88, 11.91, and 11.97 GB at the 63, 126,
168, and 252 walks.  The total copied bytes are unchanged at 12.44 GB per
reconstruction, so the saving is the band-copy transient and not the traffic.
§5.1 carries the closed form and its 2K arithmetic, which is where the knob
matters.

**No library default change is proposed.**  The knob already exists as the
model attribute `forward_project_slice_band`, which the driver reads at line
461 of `tomography_model.py` and resolves through `_slice_band_length` at lines
471 to 472.  The ledger already reads the same attribute
(`_memory_ledger.py` line 908), so a caller who sets the knob is charged
correctly today.  What this note proposes for shape P is documentation of the
knob as a memory lever, and nothing more.

### 2.2 Three cautions, if a default is ever proposed

**A default would have to be forward-only.**  `_sparse_back_project_sharded`
calls the same `_slice_band_length` at line 585 with the `back` direction, so
the default would have to be split per direction first.

**A default would have to be parallel-only.**  `_slice_band_length` is a static
method of `TomographyModel`, so a new default written into it would reach every
geometry.  Cone, translation, and multiaxis pay per launch whatever the band
contains, for the reason §1 gives, so a narrower band multiplies their work.

**A default would open a ledger drift hazard.**  `LedgerPlan.band_length`
re-implements the band rule instead of calling it (`_memory_ledger.py` lines
252 to 257), and the plan builder records only the model attribute.  A default
that moved inside `_slice_band_length` alone would leave the ledger charging
the whole shard.  The hazard is latent while the attribute is the only source
of a band.  The fix is for `plan_from_model` to store the resolved band by
calling `model._slice_band_length`.

### 2.3 What the knob does not change

The single-device path does not change, because a trivial placement never
enters the banded driver (lines 448 to 452).  The whole-cylinder call at one
device is the value anchor for both geometries, and leaving it alone is what
keeps that anchor fixed.

`broadcast_band_to_views` keeps its single production call site at line 488,
and its arguments keep their meaning.  Only the size of the band it copies
changes.

The halo exchange is untouched.  `exchange_qggmrf_halos` reads the recon
shards' boundary columns (`_sharding.py` lines 318 to 344), and neither the
recon placement nor the shard contents move.

The realized band is a maximum rather than a request.  `_balanced_slice_bounds`
takes the fewest bands no longer than the asked band and makes their lengths
equal to within one (lines 391 to 404), which is why the sweep's asked values
realize the walks in the table above.

---

## 3. Shape C: the pixel-batched full-height column gather for cone

**Adopted, subject to the checkpoint ruling.**  §7.2 carries the one gate
question that ruling must settle.

### 3.1 The measured case

**mg10 broke cone's flatness.**  The sweep ran a column-gather prototype
against the banded control at cone 1024, at two and four devices, over three
pixel batches (findings §1.8).  The per-device busy times are these.

| pixel batch | n=2 busy s | against control | n=4 busy s | against control |
|---|---|---|---|---|
| banded control | 29.69 | 1.000x | 29.32 | 1.000x |
| 2048 | 24.57 | 0.828x | 20.61 | 0.703x |
| 4096 | 20.65 | 0.696x | 15.60 | 0.532x |
| 8192 | 19.40 | 0.654x | 15.27 | 0.521x |

**These results support the mechanism §1 derived from the code.**  The banded
control barely moves between two and four devices, at 29.69 s and 29.32 s.  The
column gather at batch 8192 falls from 19.40 s to 15.27 s over the same step.
Cone's forward now falls when devices are added, which it has never done in
this campaign.

**The batch is at or above 8192, so the knee is not yet bracketed.**  Busy time
was still falling at the largest batch measured.  The library increment must
therefore sweep higher rather than adopt 8192, and §9 carries that instruction.

**Busy time is the target, and the bracket is the prototype's conservative
floor.**  The prototype issues its gathers serially, so its brackets carry
unoverlapped transfer stalls of 1.4 to 9.4 s across the six column arms.  The
stall falls as the batch grows, from 9.38 s at batch 2048 to 2.46 s at batch
8192 at two devices.  The device time the gathers themselves cost is larger
than the stall they leave.  It reads as high as 16.3 s of copy-in on one device
at four devices with batch 2048, against a stall of 5.24 s on that arm.  Most
of the copy time therefore already overlaps something.  A real implementation
overlaps the rest, by starting the next batch's gather before waiting on the
current batch's compute.  That is what
`run_per_device` performs no synchronization for by design (`_sharding.py`
lines 286 to 292).  §9 carries the overlap as its own increment.

**One gap to the work argument remains open.**  The work argument in §1
predicts half the control at two devices, which is 14.85 s, and a quarter at
four, which is 7.33 s.  The measurements are 19.40 s and 15.27 s.  Two named
causes remain: the batch sits below the knee, and the prototype adds each
batch's contribution into the full sinogram shard, which is a per-batch cost
that a larger batch amortizes.  The higher-batch sweep tests both at once.

### 3.2 What changes

**The cone branch stops banding and gathers columns instead.**  The branch is
the `else` at lines 508 to 538 of `_sparse_forward_project_sharded`, selected by
`rows_track_slices` being false.  Today that branch broadcasts a band to every
view-owner, projects each owner's own views from the band with
`slice_start=s0+l0`, and accumulates the full-row partials with
`partial_shards[i].add_(partials[i])`.  Shape C replaces it with a loop over
pixel batches inside each view-owner's worker.  For each batch the worker moves
that batch's column range from every slice-owner, concatenates the pieces along
the slice axis, and makes one projector call at `slice_start=0` over the full
slice range and its own view range.  The per-batch results accumulate into the
owner's sinogram block.

**The pixel batch is the shape's one constant, and mg10 puts it at or above
8192.**  mbirjax uses 4096 at 768 slices and above (`cone_beam.py` lines 415
to 416 and 436 to 437, in the source reading).  Our kernel batches pixels
differently, and the sweep shows our knee sitting higher, at 8192 or beyond.
The library
increment adopts a value from the extended sweep rather than from mbirjax's
constant.

**The gather becomes a named primitive in `_sharding.py`.**  A new
`gather_column_band(shard_tensors, p0, p1, target, dev2dev_safe)` moves each
slice-owner's `[p0:p1]` rows to the target and concatenates them along the
slice axis.  Three reasons argue for a named function over inline calls: an
instrument can wrap it by name as mg9 wraps `broadcast_band_to_views`, the
ledger has one place to point at, and the release convention lives in one
place.  The function is the forward's second transfer primitive, and it is
built from `move_shard` exactly as `broadcast_band_to_views` is.

**The gathered cylinder keeps the padded slice tail.**  mbirjax trims with
`full_cyl[:, :real_slices]` (source reading, A.2).  Our padded tail is held at
zero by the model, and the back driver re-zeroes it after every band reduce
(`tomography_model.py` lines 563 to 568).  A zero voxel contributes nothing
through the kernel's tap loop.  The kernel's z anchor is the real slice count
read from the parameters, and not the width of the array it is handed
(`triton_cone.py` lines 630 to 632 and 671).  Keeping the tail is therefore
inert.  It also avoids the non-contiguous copy that a trim would force in the
kernel wrapper's `values.contiguous()` at line 636.

**One `run_per_device` fan-out per forward call, with the pixel loop inside the
worker.**  The alternative is a fan-out per pixel batch.  At the 1024 cell with
two devices that alternative would issue about 190 thread dispatches for one
full-pixel forward call, because 95 batches of 8192 pixels cover the call and
each fans out to two devices.  Putting the pixel loop inside the worker also
issues each device's gathers from the thread that consumes them.  That is what
memo §3's A2 proposed, and it costs nothing to adopt here.  mbirjax nests the
two loops the same way (source reading, A.2).

**Two existing skips have to survive the rewrite.**  A view-owner with no real
views receives no copies today and produces an empty block, and the driver
builds that set as `proj_devs` at lines 465 to 466.  Shape C keeps the same
test, and an owner outside the set runs no pixel-batch loop at all.  The
all-padding sub-band skip at lines 474 to 486 has no counterpart under shape C,
because a gathered cylinder spans every slice-owner at once.  The padding it
carries is inert for the reason given above.

**Shape C is selected by geometry, not by the branch.**  The `else` branch is
shared by three models.  `ConeBeamModel`, `TranslationModel`, and
`MultiAxisParallelModel` all leave `rows_track_slices` at its default of false,
and all three bind per-view-batch bodies, so all three take the banded cone
path today.  Shape C should be read from a model attribute that only
`ConeBeamModel` sets, so the two torch-body geometries keep today's walk until
their own measurement exists.  §11 records why they are out of scope.

### 3.3 What does not change

The cone kernel does not change.  It receives calls of exactly the
single-device shape, which is a full slice range at `slice_start=0`, so the
full-detector-rows grid axis carries real values instead of zero-weighted taps.

`broadcast_band_to_views` loses its cone caller and keeps its parallel one.  It
is called from exactly one place in production today (`tomography_model.py`
line 488), and its unit test pairs it with `sum_band_to_owner` directly
(`tests/test_sharding.py` lines 44 to 53).  Nothing else in mbirtorch calls it.

The cross-device byte volume does not change, and mg10 measured that directly.
The prototype moves 12.44 GB per reconstruction at two devices and 37.32 GB at
four, which are the banded walk's own totals.  Each view-owner receives the
`(n-1)/n` of the reconstruction it does not own, in both shapes.  What changes
is the number of transfers and their size.  At the 1024 cell with two devices
one full-pixel forward call issues two cross-device copies of 1.55 GB today.
Shape C at batch 8192 issues about 190 copies of 16.5 MB for the same call.
mg9 measured the copy path at 197 to 257 GB/s, at 48 to 189 ms per
reconstruction (findings §1.7).  These results indicate that the volume is not
the constraint.  What the
transfers cost when they are not overlapped is a separate matter, and §3.1
gives the prototype's stalls.

---

## 4. The primitives, the reduce, and the adjoint contract

**The forward driver never calls `sum_band_to_owner`, and neither shape changes
that.**  The band reduce belongs to the back driver, at `tomography_model.py`
line 626.  Memo §3 records the same fact.

**The documented transpose relationship survives both shapes.**
`broadcast_band_to_views` is documented as the transpose of `sum_band_to_owner`
(`_sharding.py` lines 252 to 263).  The band knob changes the size of the band
the function copies and not the function.  Shape C stops calling the function
in the cone forward, and the function itself is unchanged, so its documented
relationship to the reduce holds wherever it is still used.

**Adjointness is a property of the two operators, not of one transfer.**  Memo
§3's A5 was excluded because it changed which device computes which output,
which would require a matching change on the back side.  Shape C does not
change that assignment.  Every view-owner still produces its own views' full
sinogram block, from the same voxels, through the same kernel.  The operator is
unchanged, so the sharded forward stays the adjoint of the sharded back within
the float class the suite already holds.  `test_banded_projectors_adjoint` and
the cone inner-product check in
`test_cone_banded_projectors_match_single_device` are the standing evidence
(`tests/test_sharding.py` lines 165 to 172 and 203 to 215).

**The halo exchange is untouched by both shapes**, for the reason §2.2 gives.

---

## 5. The memory ledger

### 5.1 The parallel band knob

**One term changes, and two follow it.**  `forward_band_copy` charges one
cylinder shard on every view-owner, at `num_pixels * slice_blocks[i][0] * 4`
bytes (`_memory_ledger.py` lines 451 to 463).  With the band knob set the width
becomes the band instead of the shard, so the term reads `num_pixels * min(B,
slices_per_dev) * 4`.  Two further terms follow, because `forward_cols(i)`
returns the band length (`_memory_ledger.py` lines 301 to 304) and both
`forward_block_rows` and the forward view charge read it.  The ledger already
computes all three from the model attribute, so no ledger change is needed for
the knob.

**Values, with the band written as 252 for arithmetic.**

| cell | count | today | band 252 |
|---|---|---|---|
| 1024 | 2 | 1.55 GB | 0.78 GB |
| 1024 | 4 | 0.78 GB | 0.78 GB |
| 2K | 2 | 12.47 GB | 3.12 GB |
| 2K | 4 | 6.23 GB | 3.12 GB |

**mg10 measured the 1K row of that table at the whole-reconstruction peak.**
Per-device peaks at parallel 1024 with two devices read 12.48 GB at the control
and 11.84, 11.88, 11.91, and 11.97 GB at the 63, 126, 168, and 252 walks.  The
modeled saving at the 252 walk is 0.78 GB.  The measured saving is 0.50, in
the binary gigabytes the peak counter reports, and the modeled saving is 0.72
in those same units.  The measured saving is therefore about seventy percent
of the modeled one, which is the right size and the right direction for a term
that is one of several at the peak.

These results indicate that the knob's memory effect is largest exactly where
charter C needs it, because the term is proportional to the shard at 2K and
constant at the band.

### 5.2 Shape C

**Two terms go away and one arrives.**  `forward_band_copy` is zero for a cone
model under shape C, because the cone forward no longer broadcasts a band.  The
term that replaces it is the gathered column cylinder, and its closed form is
`resident * pixel_batch * num_slices * 4` bytes.

**The resident count is two, and it requires an explicit release.**  Python
evaluates the next batch's gather before it rebinds the name.  Without a
release, three things are then live together: the previous batch's cylinder,
the incoming pieces, and their concatenation.  The driver already makes this
release for the cone partials and states the reason (`tomography_model.py`
lines 528 to 538).  The back view loop makes the same release
(`projectors.py` lines 442 to 452).  Shape C adopts the same convention, and
the ledger charges two.  Increment 5 raises the count to three, because
overlapping the next batch's gather keeps one more cylinder in flight.

**mg10 measured the gathered cylinder, and it matches the closed form.**  The
prototype's cylinder reads 7.9, 15.8, and 31.5 MiB at batches 2048, 4096, and
8192, which is `batch * 1008 slices * 4 B` exactly.  The charged term is twice
those figures, because the ledger charges two residents.  The table below
states the same cylinders in decimal megabytes, to match the ledger's other
terms, so the 15.8 MiB cylinder reads 16.5 MB there.

| cell | pixel batch | one cylinder | charged term |
|---|---|---|---|
| 1024 | 4096 | 16.5 MB | 0.033 GB |
| 1024 | 8192 | 33.0 MB | 0.066 GB |
| 2K | 8192 | 66.1 MB | 0.132 GB |

**The whole-reconstruction peak fell rather than rose.**  Per-device peaks at
cone 1024 with two devices read 12.76 to 12.81 GB across the three batches,
against the banded control's 14.31 GB.  These results indicate that the column
gather is cheaper in peak than the banded walk at 1K, because it drops a
band-copy transient that is larger than the cylinder it adds.

The term does not depend on the device count.  That is what dissolves the 2K
objection memo §3 raised against A4, where the assembled cylinder was 24.9 GB
per device at 2K (findings §6.4).

**Two other forward terms move, and the ledger must carry both.**  Shape C
lowers the per-view charge, because the pixel count of one call falls to the
batch.  The charge is `48 * num_pixels + 4 * num_channels * num_rows_r`
(`_cone_forward_view_batch_cost`, `triton_cone.py` lines 685 to 693).  The
realized view batch therefore rises from the budget cap to the kernel's
nominal chunk of 128.  That rise is measured rather than derived.  The mg10
rows read 128 at the cone column arms, against the 52 the banded control reads
at the same cell.  At the 2K cell the same arithmetic takes the batch from 13
to 128.  `forward_batch` falls at 1K and holds at 2K.  `forward_block` rises
with the view batch, because a cone block spans the full detector rows
(`_memory_ledger.py` lines 477 to 491).  §6 gives the numbers.  One
consequence deserves its own sentence.  The pixel batch and the view batch
compete for the same 2 GiB transient budget (`projectors.py` lines 222 to
249), so increment 3's extended sweep must vary them together or hold one
fixed and say which.

### 5.3 The rules that govern the change

**The 1.00 floor governs.**  No modeled per-device peak may sit below the
measured one, and the acceptance band is 1.00 to 1.30 for the kernel path
(findings §2, §2.1).  A cell below the floor is fixed by adding the missing
term and never by a factor.

**A calibration re-run follows landing.**  The mg2 protocol measures modeled
against measured per-device peaks at both gate cells, both geometries, and two
and four devices.  Shape C changes the terms that set the forward phase's
modeled peak, and the forward phase set the modeled peak on nearly every run in
the 2026-08-10 check (`_memory_ledger.py` lines 431 to 434).

---

## 6. The 2K table

Per-device forward transients at charter C's 2K design point, which is a
(2048, 2016, 1984) sinogram at four devices (findings §6.4).  The persistent
set is unchanged and is not included.  The band-knob row is evaluated at a band
of 252 and the shape C row at a pixel batch of 4096.

| geometry and form | cross-device transient | forward batch | forward block | sum |
|---|---|---|---|---|
| parallel today | 6.23 GB | 2.14 GB | 0.16 GB | 8.53 GB |
| parallel, band knob at 252 | 3.12 GB | 2.11 GB | 0.08 GB | 5.31 GB |
| cone today | 6.23 GB | 2.14 GB | 0.21 GB | 8.58 GB |
| cone, shape C | 0.07 GB | 2.07 GB | 2.05 GB | 4.19 GB |

The forward batch is capped by the 2 GiB transient budget in three of the four
rows, which is why it barely moves.  The fourth row is cone under shape C,
whose batch is capped by the kernel's chunk of 128 instead and lands just under
the same budget.  Cone's block grows under shape C, from 0.21 to 2.05 GB,
because its view batch grows about tenfold.  That rise costs 1.77 GB once the
small fall in the batch is counted against it.  The cross-device transient
falls by 6.16 GB over the same change, so the row's sum still falls by
4.39 GB.  Both forms lower the forward's per-device transient at 2K, and shape
C lowers
it more.  A larger pixel batch raises the shape C row's first column
proportionally, at 0.13 GB for batch 8192.

---

## 7. The value gates

### 7.1 The parallel band knob

**The value class does not change, and the reason is structural.**  A
row-aligned geometry gives each detector row a single producing band.  Changing
the band length changes which call produces a row, and it does not change the
set of contributions summed into that row.  Memo §8.5 rules the same way.

**The gate compares band-to-band distance against run-to-run distance, and
mg10 ran it.**  The kernel accumulates with `tl.atomic_add`, so the order in
which one row's contributions are applied is already unspecified at a fixed
band.  Bit equality is therefore the wrong bar on a GPU, and it was not
asserted.  The sweep reports two distance families, and the band knob passes on
both.  On the checksum metric the repeat floors at two devices are 3.26e-10 to
7.44e-10 per arm, and the band-to-band distances against the control are
9.24e-11 to 7.80e-10.  Four of the five swept bands sit below their own repeat
floor, and the 63-slice walk sits 8 percent above its own.  On the sample
metric every swept band's relative distance to the control is 1.72e-07 to
2.38e-07, against pass-to-pass floors of 1.74e-07 to 2.31e-07.  These results
confirm order preservation on both metrics at every band.  The harness also
records whether a swept band's checksum equals its control's exactly, and every
arm reads false, which is what the kernel's atomics are expected to give.

**The standing parity suite has two parts.**  The first is
`tests/test_sharding.py::test_banded_projectors_match_single_device` and
`::test_banded_projectors_adjoint`, on virtual CPU devices.  The second is
`tests/test_kernels_sharded.py` on two CUDA devices, whose forward arm holds a
relative floor of 5e-3 and measured 3.4e-07 and 1.1e-06 on two H100s.  A
band-to-band distance well above the repeat distance is a defect rather than a
tolerance question, because it would say the row-to-band assignment is not what
this design assumes.  The 63-slice walk's 8 percent margin is not that.  A
distance and a floor of the same size are one reading, not two.

### 7.2 Shape C

**The gate is anchored on one device, because the one-device path already makes
full-height calls.**  Shape C makes a view-owner's calls the same shape as the
single-device call, so the one-device result is the value the multi-device
result should approach.

**mg10 measured the distances on two metrics, and the two do not agree.**  The
rule was stated before the measurement.  The column-gather distance to one
device may not exceed the banded distance to one device, and a tighter distance
is a win to be recorded.  The sweep reports two independent distance families.
The first is a relative distance between whole-volume checksums.  The second is
a relative L2 over every fifteenth voxel of the reconstruction.  Each family
carries its own repeat arms, so each distance is read against a measured floor.

**On the checksum the rule is met at two devices and is not resolvable at
four.**  The readings against the one-device anchor are these.

| pair | n=2 | n=4 |
|---|---|---|
| banded control against the anchor | 1.01e-09 | 7.20e-11 |
| column gather at 2048 against the anchor | 2.47e-10 | 4.13e-11 |
| column gather at 4096 against the anchor | 2.58e-11 | 1.59e-10 |
| column gather at 8192 against the anchor | 7.09e-11 | 9.25e-11 |
| repeat floors over those four arms | 1.91e-10 to 1.50e-09 | 6.59e-10 to 1.13e-09 |

At two devices every column batch reads below the banded control's 1.01e-09.
At four devices the banded control reads 7.20e-11, and two of the three column
batches read above it, at 1.59e-10 and 9.25e-11.  All four four-device readings
sit below their own repeat floors, so the ordering there is inside the noise
this metric can resolve.  The column gather's distance to the banded control
reads 9.42e-10 to 1.26e-09 at two devices and 2.05e-11 to 8.71e-11 at four,
against the same floors.

**On the sample metric the rule is not met, and the excess is explained.**  The
column gather sits 1.47e-06 to 1.54e-06 from the one-device anchor, at every
batch and both counts, against pass-to-pass floors of 1.26e-07 to 2.03e-07.
The banded control sits 2.83e-07 from the anchor at two devices and 5.41e-07 at
four, against its own floors of 2.37e-07 and 2.53e-07.  The column gather is
therefore three to five times further from the anchor than the banded walk is,
and seven to twelve times its own repeat floor.  The banded walk, by contrast,
sits at that floor.  This is the more sensitive of the two metrics, because a
checksum is a sum over the volume and cancels differences of opposite sign that
an L2 does not.  The last paragraph of this section names why shape C should
produce a distance above the floor.  Shape C changes the summation structure
twice, where the band knob changes it not at all.

**The verdict, and the one thing the checkpoint must rule.**  The prototype's
distance to the one-device anchor is 1.5e-06 relative, against the 5e-3
relative floor the standing kernel-parity suite holds, so it sits three orders
of magnitude inside the tolerance the library already ships.  The largest
single-voxel difference is 4.7e-06 in absolute value, which is 4.5e-06 of the
reconstruction's peak.  The prototype
does not, however, sit at or below the banded walk's own distance to the
anchor, which is what the rule quoted above required.  That rule was written
before either metric had been read, and it is met on the checksum metric at two
devices only.  §12's sixth question asks the checkpoint to rule on which bar
governs shape C.

**What must pass beside the distances.**  Four gates apply: the cone banded
parity and inner-product checks in `tests/test_sharding.py`, the two-device
kernel arms in `tests/test_kernels_sharded.py`, the cross-framework goldens in
`tests/test_vs_goldens.py`, and the ledger tests in
`tests/test_memory_ledger.py`.  One qualification belongs with the goldens.
They run on one CPU device and therefore exercise the unchanged single-device
path, so they confirm that the anchor did not move rather than gating the
change itself.

**Two order changes are in scope, and the memo named one.**  The vertical sum
moves from a host-side sum across bands into the kernel, which memo §8.5 names.
The pixel sum moves the other way, from the kernel's atomics into a host-side
sum across pixel batches, which the memo does not name.  Both sit inside a
value class the kernel already has, because the forward kernel's atomic
accumulation is itself order-unspecified.  The gate above prices both together,
because it compares whole reconstructions.

---

## 8. The staleness consequence

**The widening floors detect a changed cost input by hashing named sources.**
`COST_INPUT_FILES` is `triton_parallel.py`, `triton_cone.py`, and
`projectors.py`, hashed whole.  `COST_INPUT_METHODS` is
`_sparse_forward_project_sharded` and `_sparse_back_project_sharded`, hashed by
method source through `inspect.getsource` (`_widening_floors.py` lines 179 to
186 and 439 to 457).

**Shape C fires the note by itself.**  It rewrites
`_sparse_forward_project_sharded`, which is hashed.  The band knob fires
nothing, because setting a model attribute changes no source.  That is the
correct behaviour, because a caller's knob setting is not a library cost
change.

**One addition belongs in the same change as the code, and one is a latent
repair.**  `_sharding.py` joins `COST_INPUT_FILES` once the column gather lives
there.  `_slice_band_length` should join `COST_INPUT_METHODS` whenever a band
default is ever proposed, because that method is not hashed today and a default
moving inside it would go undetected.  The module docstring already states the
rule: code that determines projection cost and moves must be added to
`COST_INPUT_FILES` or `COST_INPUT_METHODS` in the same change
(`_widening_floors.py` lines 60 to 66).  Adding a name is itself detected,
because an unrecorded key reads as changed.

**A floors refresh follows landing.**  `dev_scripts/refresh_widening_floors.py`
is the sole writer of the floors, the recorded hashes, and the staleness date,
and pasting its output is the one thing that clears the note.  The full run took
31 minutes on four GPUs, and findings §3.3 prices a knee refresh of the three
or four cells nearest each floor at about fifteen minutes.

**Which knees should move is already predicted.**  Memo §6, "The validating
measurement", predicts that the forward-sensitive knees move down and the
others hold.  The forward-sensitive readings are cone at every count above one
and parallel 1024 at two devices (findings §3.3).  Shape C reaches only cone,
so the parallel reading in that set should hold rather than move.  The readings
below the 384 cell are limited by the fan-out and glue, which no forward change
touches.

---

## 9. Implementation increments

Each increment is small, independently testable, and independently revertible.
Increments 1 and 2 are complete, because mg10 ran them.  The order of the rest
runs from the cheapest reversible step to the one that changes a default.

**Increment 1, complete: the band sweep at a fixed device count.**  mg10 asked
five bands at two devices and two at four, reading per-launch time, per-device
total, per-device peak, and value distance.  Those requests realized four
distinct walks at two devices and one at four, because a request wider than the
shard collapses onto the shard.  §2.1 carries the result.

**Increment 2, complete: the column-gather prototype and its batch sweep.**
mg10 ran three batches at two and four devices, reading busy time, bracket,
copy volume, transient, peak, and value distance.  §3.1 carries the result.

**Increment 3, complete: the extended batch sweep.**  Sweep the pixel batch
above 8192, at 16384 and 32768, at both counts.  Busy time was still falling
at 8192, so the knee is not yet bracketed.  Read the realized view batch
alongside, because it moves with the pixel batch, per §5.2.  This increment
runs in the plans repository and edits no library code.  mg11 ran the sweep
on both geometries: composed wall kept improving through the new batches, by
4 to 15 percent over 8192 depending on the cell, so the knee is still not
bracketed.  The default stays at 8192 for the scale reason findings §1.10
records, and the 2K sweep belongs to the production-scale charter.

**Increment 4, complete: the column gather behind a switch, defaulting off.**
Add `gather_column_band` to `_sharding.py` with its own unit test, add the
cone branch that uses it, and select it from a model attribute that defaults
to false.  The banded branch stays in place and is the rollback.  The switch
is what lets the parity suite run both forms in one session for the distances
of §7.2.  This landed as commit 142b394, and the §13 extension brought
parallel beam onto the same driver as commit a33c7e8.

**Increment 5, complete: overlap the gather with the compute.**  The prototype
issues its gathers serially, and §3.1 gives the stalls that costs.  Issue the
next batch's gather before waiting on the current batch's projection, which
`run_per_device` already permits by performing no synchronization
(`_sharding.py` lines 286 to 292).  The increment is measured on the bracket
rather than on busy time, because busy time already excludes the stall.  It
also raises the resident count of §5.2 from two to three while one extra batch
is in flight, so the ledger term moves with it.  mg12 measured it on
2026-08-11 and findings §1.13 carries the result: the prefetch alone bought
0.8 to 2.4 s of forward wall, dedicated per-device copy streams on top of it
closed the stall and more (16.57 to 9.32 s at the widest configuration,
composed reconstructions 1.07x to 1.41x across five configurations), values
at the repeat floor, memory floors held.  The same measurement corrected this
increment's own premise: busy time did NOT exclude the stall at more than one
device, because peer-serving copies interleaved inside the busy brackets, so
the visible stall understated the transfer cost and the win exceeded it.

**Increment 6, complete: reduce the per-batch accumulation.**  The prototype
adds each batch's contribution into the full sinogram shard.  Accumulating
into a preallocated buffer, or accumulating less often, removes a per-batch
cost that the larger batches of increment 3 would otherwise hide rather than
remove.  The implementation found the literal preallocated buffer already
effectively in place and the real cost in the projector's per-batch block
allocation and hand-back; the fused form adds later batches into the running
total inside the projector's view loop, bit-identically.  mg14 measured it on
2026-08-11 and findings §1.15 carries the result: forward wall down 3.5 to
6.4 percent, per-device peak down about 0.9 GB, busy flat, values at the
repeat floor.  With this, increments 1 through 10 are all complete and the
forward remedy's implementation is finished; increment 11, the item-13
re-gate, closed the same day by measurement (findings §1.14).

**Increment 7, complete: the default, gated.**  Flip the switch on for cone
only after the §7.2 gates pass on the library implementation rather than on
the prototype.  The bar those gates are read against is §12's sixth question,
which the checkpoint must rule before this increment runs.  Translation and
multiaxis keep the banded walk.  The §13 ruling extended the flip to parallel
beam under one combined gate campaign.  That campaign ran as job mg11 on
2026-08-11 and passed all six gates, findings §1.10 carries the readings, and
the flip is implemented: unset selects the gather on cone and parallel, an
explicit False selects the banded walk, and translation and multiaxis are
unchanged.

**Increment 8, complete: the ledger terms.**  Land the closed forms of §5
with their tests, then re-run the mg2 calibration and re-read every cell
against the 1.00 to 1.30 band.  The closed forms landed with increment 4.
mg11 supplied the re-read: all 22 arms sat inside the band, from 1.003 to
1.158, and the library's own `last_memory_ledger` agreed with the harness's
independently built ledger on every arm.

**Increment 9, complete: the floors refresh and the record corrections.**
Run `dev_scripts/refresh_widening_floors.py` and paste its block.  Correct the
two documented band-length numbers, which are the 47 to 66 percent and the 8
percent readings quoted from the `_slice_band_length` docstring in the user
documentation (`plans/torch_port/active/docs.md`, "Numbers used, with their
source").  Both are pre-kernel measurements, as the plan's lever table already
notes (`plans/torch_port/active/multigpu_plan.md`, the streaming row).  mg10
now supplies the kernel-era replacement for both, at §2.1's table.  The record
corrections landed 2026-08-11, in the docstring, in both documentation pages,
and in the two plan records.  The floors refresh ran on the committed flip,
`4a222c7`, as job 15172987: a four-GPU node, 33 minutes, with its block pasted
into `_widening_floors.py` and the staleness note clean at that commit.
Findings §1.12 records which floors moved.  Commits after `4a222c7` have since
changed the projection-cost code, so the note names drifted inputs again.  That
drift is new measurement debt, tracked as item A1 of `plans/open_items.md`, and
not unfinished work of this increment.

**Increment 10, complete: the parallel band knob, documented.**  Record
`forward_project_slice_band` as a memory lever with §2.1's measured cost, which
is about 0.5 GB of peak for about 2 percent of time at the 252 walk.  No
default changes.  The record landed 2026-08-11 in the plan's lever paragraph
and in the user documentation's memory-lever list.

**Increment 11: the item-13 re-gate.**  Re-evaluate item 13's step-1 STOP
threshold on post-remedy numbers, and probe the light per-call sorted form
first, per memo §8.5.

**One cheap follow-up measurement, outside the increments.**  Run parallel at
one device with the band knob set to 504, which the existing knob supports.
§12's first question gives what the reading decides.

**Measurement hooks.**  mg9's probes attach by replacing named library
functions and by wrapping the per-device body list, and one of them needs an
addition.  The busy probe wraps `_fwd_body_per_dev` entries positionally, so it
reads shape C's calls without a change.  The copy probe is gated on a
thread-local flag set inside `broadcast_band_to_views`, so a cone arm under
shape C would record no copies unless the same flag is set inside
`gather_column_band`.  A named gather function is what makes that a one-line
addition.

---

## 10. Outcomes: what mg10 measured, and what remains predicted

The first three statements are measurements from the mg10 arms.  The rest are
predictions, stated in the form memo §6's validating-measurement section uses.

**Measured: cone's forward span now falls with the device count.**  The
column-gather prototype at batch 8192 reads 19.40 s of busy time at two devices
and 15.27 s at four, against a banded control of 29.69 s and 29.32 s.  The
prototype's bracket at two devices is 21.9 s, and the difference from busy is
its unoverlapped gathers.

**Measured: the composed wall at cone 1024 falls at both counts.**  mg10's own
walls at batch 8192 are 58.45 s at two devices and 39.13 s at four.  Its banded
controls at the same cell are 67.16 s and 52.96 s, and its one-device anchor is
61.54 s.  Cone's two-device scaling therefore turns from 0.92x into 1.05x, and
its four-device scaling from 1.16x into 1.57x.  Two devices stops being a
regression for the first time in this campaign.  Findings §1.2 measured the
same three controls on a different run, at 67.23, 53.10, and 61.57 s, so they
reproduce.  Against that run's jax rulers the ratio falls from 1.55 to 1.35 at
two devices and from 2.06 to 1.52 at four.  Those two ratios compare runs,
because mg10 ran no jax arm of its own.

**Measured: shape P moves no time.**  §2.1 gives the sweep.  The band knob is a
memory lever of about 0.5 GB at the 1024 cell, at a time cost of about 2
percent at the 252 walk.

**Predicted: the extended batch sweep should close part of the remaining gap.**
The work argument in §1 puts the floor at 14.85 s at two devices and 7.33 s at
four.  Busy time was still falling at batch 8192, so increments 3 and 6 should
move the measured 19.40 s and 15.27 s toward those floors.  How far is not
predicted here, because no measurement bounds it.

**Predicted: the library implementation should beat the prototype's bracket.**
Increment 5 overlaps the gathers, which the prototype does not, so the bracket
should approach the busy time.  The composed wall should improve by the part of
the 1.4 to 9.4 s stall that the overlap recovers.

**Predicted: cone should still not reach jax at two devices.**  Cone's back
projection rises to 30.33 s of device span at two devices, and no forward shape
touches it.

**Predicted: parallel's numbers should not move at all.**  Shape P is declined
as a time remedy, so nothing in this note changes parallel's timings.
Parallel's one-to-two leg stays open for §12's first and third questions.

**Predicted: charter C's 2K capacity table should change in the direction §6
gives.**  The ledger is closed-form, so that table is a design-instrument
computation rather than a measurement (findings §6.4).  It can therefore be
recomputed before any code moves.

---

## 11. What this note does not cover

**Cone's back projection.**  It rises from 23.59 s to 30.30 s of device span at
two devices, and mg9 reproduced that rise at 30.33 s.  It is now as large as
cone's forward at the same cell.  No forward shape touches it, and charter A
already names it as needing its own variant.

**Item 13 itself.**  Its design is recorded as K5 in
`plans/torch_port/phases/phase5_kernel_design.md`, and its position after the
driver remedy is unchanged.  This note touches it only through the re-gate in
§9.

**The two torch-body geometries.**  `TranslationModel` and
`MultiAxisParallelModel` have no hand-written kernels, so their forward runs
general torch code.  A band default must not reach them, and §2.2 gives the
mechanism that would otherwise carry it there.  Their calibration band is
(1.00, 5.80) from mg8, where the kernel path's is (1.00, 1.30) (findings §2.1).
A band change would therefore also move a charge that a different measurement
fixed.  Shape C's structure does apply to these two geometries.  Both declare
transients whose width is the detector rows or the call's slice band, whichever
is wider, so the width does not fall when the band narrows
(`translation_model.py` lines 295 to 300, `multiaxis_parallel.py` lines 276 to
281).  That is the same band-independence that makes cone's banded walk cost n
times one call.  Neither geometry should be switched on without its own
measurement.  mg8 measured a translation shape's peak growing from 8.2 to
27.2 GB between one and four devices, which is the reason to measure rather
than assume.

---

## 12. Open questions for the checkpoint

**1. Where does the parallel kernel's factor of two live?**  mg10 sharpened
this question rather than answering it.  Per-slice time at two devices runs
0.075 to 0.101 ms over bands from 63 slices to 504, with no trend in the width,
so the step is not a band-width effect within a device count.  Per-slice time
at one device is 0.041 ms at width 1008 (findings §1.7).  The step therefore
lies between the one-device arm and the
multi-device arms, and one device is also the only place a 1008-slice band was
ever measured.  Those two facts are still confounded.

**The discriminating arm has not been run, and it is cheap.**  Run parallel at
one device with the band knob set to 504, which `forward_project_slice_band`
already supports.  A per-slice reading near 0.041 ms would say the factor of
two is a multi-device effect, because the width changed and the cost did not.
A reading near 0.082 ms would say it is a kernel width effect spanning 504 to
1008, because the device count stayed at one and the cost moved.  Two devices
read 0.082 ms at that same width, which is what makes it the discriminating
value.

Two further measurements would explain whichever answer comes back.  The first
is a profiler read of one launch at each width, recording achieved occupancy,
L2 hit rate, atomic throughput, and DRAM traffic.  The second is a
single-variable ablation that separates the band from the array it lives in.
Project a 504-slice band whose `values` array is padded to 1008 columns, so
that the work is that of 504 and the layout is that of 1008.

The adopted design does not depend on the answer.  Shape P is declined on
measured totals whatever the mechanism, and shape C is adopted on measured
totals of its own.  The answer decides which further remedy is worth proposing
for parallel, which is the third question below.

**2. Is the band knob worth documenting as a memory lever?**  §2.1 measures it
at about 0.5 GB of per-device peak for about 2 percent of time at the 1024
cell, and §5.1 computes a four-fold reduction of the band-copy term at 2K.  The
checkpoint should rule on whether that is worth a documented knob now, or
whether it waits for charter C to need it.

**3. Should shape C's structure be applied to parallel as well?**  This goes
beyond memo §8, which assigns the band shape to parallel because mbirjax does.
The column gather is the only shape in hand that makes a full-slice-range call
at a bounded transient.  It is therefore the only way to give a multi-device
parallel arm the 1008-slice call shape that the one-device arm runs at 0.041 ms
per slice.  Whether that recovers the factor of two depends entirely on the
first question's arm.  Memo §3's A4 was the same idea without pixel tiling, and
its 2K residency objection is dissolved by the same tiling that dissolves it
for cone.

**4. How high does the pixel batch go?**  The concern the draft recorded was a
floor, and the measurement turned it into a ceiling.  mbirjax records a
dispatch floor of 4,000,000 elements per band (source reading, A.4), and a
batch of 4096 at 1008 slices sits at that floor.  Our port pays more per batch
than mbirjax does, because our pixel loop is a Python loop where mbirjax's is a
`lax.scan` inside one compiled program.  Our measured knee is at or above 8192,
so increment 3 sweeps up rather than down.  What bounds the batch from above is
the transient budget it shares with the view batch, per §5.2.

**5. Answered: the higher launch count under shape C costs nothing visible.**
The draft raised this as a risk, because shape C raises cone's per-device body
calls while cutting each call's work.  mg10 measured busy time falling at every
batch and both counts, so the added calls did not offset the work removed.  The
question is recorded as closed and is kept here for the record.

**6. Which value bar governs shape C?**  §7.2's rule, written before the
measurement, required the column gather's distance to the one-device anchor to
sit at or below the banded walk's.  The prototype meets that rule on the
checksum metric at two devices and misses it on the sample metric at both
counts, where it reads 1.5e-06 against the banded walk's 2.8e-07 to 5.4e-07.
The excess is expected rather than anomalous, because shape C changes the
summation structure twice and the banded walk changes it not at all.  The
alternative bar is the one the library already ships, which is the standing
kernel-parity suite's 5e-3 relative floor, and the prototype sits three orders
of magnitude inside it.  The checkpoint should rule which bar governs, because
increment 7 flips the cone default against whichever one is chosen.

RULED (Greg, 2026-08-10, evening): the shipped parity floor governs, on the
stated condition that the column gather is faster with comparable or better
memory, and both conditions hold in the rows (the status header carries the
numbers).  The measured e-6 class is registered as the expectation beside the
floor.  The class is not a hard threshold, by the same ruling: a reading
beyond it is a tradeoff judgment for the maintainers, and only the floor
fails a gate.

**7. Is the write slab's L2 residency the variable behind the sweep's one
anomaly?**  The walked-63 arm was the parallel sweep's only winner.  Its
per-launch scatter target is 128 views by 63 rows by 1024 channels at four
bytes, which is 33 MB, and that is the only swept width inside the H100's
50 MB L2; the next width up, 126 rows, is 66 MB.  The mbirjax kernel campaign
measured the same mechanism from both sides: its largest recorded win, 16.3 to
26x on the back projection, came from a design that phases the output through
L2-sized chunks, and the cone-back configuration whose working set crosses L2
costs a measured 9.1 percent on H100 (the campaign records,
`plans/projector_kernels/gpu_headroom_findings.md` and
`a100_tuning_findings.md`).  Two cheap probes would test the reading, and
neither is this note's work.  The first is the one-device narrowed-band arm of
question 1, which carries width 63, so it reports whether the win appears with
no multi-device effect in play.  The second shrinks the slab's other axis: a
view chunk of 8 at the full 1008-row band gives the same 33 MB slab, and the
recorded batching sweep never went below chunk 32, which is 132 MB and never
crossed L2.  Both readings belong to the tiling campaign's entry context,
because the phase-blocked accumulation in the pallas record is the structural
form of the same idea.

RULED (Greg, 2026-08-10, evening), on the cache directions this question
opens.  Four are worth exploring, each inside the campaign that owns it:
phase-blocked accumulation (the tiling charter's kernel leg), this question's
two probes, cache-eviction hints in the existing kernels (an increment inside
whichever kernel campaign runs), and on-chip accumulation through sorted
streams (the sorted-stream charter, which that framing strengthens).  Declined:
cross-stage fusion, the sinogram layout transpose, persistent kernels, and
full-scale accumulator privatization.

## 13. Addendum (2026-08-10, late): the discriminator's answer, and the parallel extension

The probes of questions 1 and 7 ran the same night as job 15159551, and
findings §1.9 carries the full read.  Both questions are answered.  The
parallel doubling is a kernel width effect and nothing else: cutting the
one-device call into two 504-wide pieces doubles its cost, a launch at width
504 takes the same 41.4 ms as a launch at width 1008, and the device count
contributes nothing.  The L2 reading resolved the other way: putting the
write slab inside L2 through the view chunk moved nothing at either count,
so L2 residency is the width-63 arm's ten percent and not the doubling.

The answer converts this note's parallel verdict from a decline into an
extension.  §2 declined SUB-BANDING because narrower blocks cannot help a
kernel whose efficient regime is the widest block, and the measurement now
says exactly that.  What parallel needs is the opposite of sub-banding:
full-width values blocks at every device count, which is what the column
gather already provides for cone.  For parallel the gather is
order-preserving, because each detector row keeps a single producing piece,
so the cone value question of §7.2 does not arise.  The transient is the
same bounded column cylinder, so the 2K arithmetic of §6 carries over
unchanged.  From the measured rates, parallel 1024 at two devices should
fall from 28.2 to about 14.1 s of forward busy and from 39.2 to about 25 s
composed, which is the scale §6 of the memo predicted for a successful
remedy.

The implementation is small by construction, because the mechanism landed
behind a geometry capability: `column_gather_geometry` moves to true on
`ParallelBeamModel`, the row-aligned refusal in the driver's resolver is
lifted for the gather path, the ledger's column term already prices the
shape, and the parallel parity tests join the flag-on suite.  The gates are
the same as cone's: the standing suites, the cluster value gates, and the
default flip only behind them.

RULED (Greg, 2026-08-10, late): the extension is approved, the two
geometries' gates run as one combined cluster campaign, and the defaults
flip when the gates pass, without a further ruling per geometry.  One caveat
is recorded with the approval: the two-times prediction was measured with
the pixel count held full, and the gather cuts pixels to the batch, so the
realized win at batch 8192 may land under two times; the combined campaign
sweeps larger parallel batches (the transient stays under 150 MB through
batch 32768) and the prediction is judged against the swept best.
