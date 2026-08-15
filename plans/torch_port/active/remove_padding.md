# Design note: removing the sharding pad

Written 2026-08-14, from a code review with Greg; revised the same day
on a three-seat review (accuracy, reasoning, style).  Status: proposed,
not decided.

## Summary

Every sharded axis in mbirtorch is padded up to a multiple of the
device count.  The pad was copied from mbirjax, and mbirtorch does not
require it.
 - mbirjax pads because jax requires it.  A `NamedSharding` partitions
   one global array, and jax requires the sharded axis to divide evenly
   across the devices.
 - mbirtorch has no such constraint.  A `Shards` is a plain list of
   per-device tensors, transfers are explicit `move_shard` copies, and
   the package uses no `torch.distributed` collectives.  The
   divisibility comes from one line of arithmetic in `_sharding.py:88`:
   `padded_size = ((real_size + n - 1) // n) * n`.
 - Nothing in torch requires shards of equal length.  A ceil/floor
   split gives shards that differ in length by at most one.

The recommendation is to remove the pad.  Removal deletes the code that
exists only to manage the pad, one live defect, and a recorded class of
failures.  It changes ten source files and three test files, so it is
separate work with its own review, not an addition to the entry-point
increments.  The decision itself belongs earlier: increments 4, 5, and
7 of the entry-point plan write code whose form depends on it (§5).

## 1. What the pad costs

Five pieces of code exist only to manage the pad, and the pad has
caused one live defect and a recorded class of failures.  Removal
deletes the code and retires the failures.
 - The inertness invariant: padded entries must stay exactly zero.  The
   sharded back projection re-zeroes its padded slices, because a back
   projection is a gather, so real detector data is written into the
   padded entries.
 - The qGGMRF interface masks.  A padded shard's last slice is not a
   real slice, so the prior must mask that boundary slice.  With no
   pad, every boundary slice is real.
 - The `real_mask` exclusions in the MAR segmentation statistics
   (`preprocess/mar.py`, the only `real_mask` consumer in the package).
   The reconstruction loop itself uses no mask: its statistics sum the
   padding's zeros and normalize by the real element count.
 - The crop step wherever shards are collected to the host.
 - The detector-row pad of the row-aligned geometries
   (`_sino_row_padding`): parallel beam pads the sinogram row axis to
   match the padded slice axis.  The `rows_track_slices` flag itself
   stays, because it selects the row-aligned driver with or without a
   pad.

The live defect: `helical_fdk_z_weight` passes the real slice count to
`shard_ranges`, which refuses a size the device count does not divide
(`cone_beam.py:725`).  A multi-device helical FDK therefore raises at
every non-dividing device count, and that branch is untested.  Removal
deletes the refusal that causes this.

The recorded failure class: a padded slice with zero projection
coverage turns `0 * inf` into NaN.  The lessons file also records the
general rule that assumptions about shard shapes appear only at device
counts that do not divide the axis.

The cost also appears in work that is planned and not yet written.  The
per-shard `gen_weights` of the entry-point plan's increment 4 must
re-zero its padded entries.  Three of its four weight types write
non-zeros exactly where the invariant requires zeros: `exp(-0) = 1` for
the transmission forms, ones for `'unweighted'`, and `1 / 0.1` for
`'emission'`.

## 2. What the pad buys

Two of the three claimed benefits are real and small.  The third is not
delivered by the current code.  All three were checked.
 - Simple offsets.  Every shard has the same length, so a global index
   maps to a device and a local index by one division.  `Placement`
   already computes explicit per-shard ranges, so uneven shards change
   this arithmetic in one place.
 - mbirjax parity of the shard lengths.  No cross-framework value
   comparison exists in the repository, and every stored golden is
   single-device, so nothing compares shard against shard today.  If
   such a comparison is ever wanted, it can run at a device count that
   divides the axis, where the two splits coincide.
 - Uniform kernel shapes: not delivered.  Both forward paths and the
   back driver restrict each device to its real view count before
   calling the projectors, so the compiled bodies already see
   per-device shapes today.  Each device holds its own compiled
   instance, so shard-length variation reaches one instance only
   through the slice bands, which already take two lengths.  The
   Triton tile sizes are capped powers of two, so two shard lengths at
   production sizes compile the same kernel.  The kernel launch cache
   keys on the runtime lengths, so a second length costs one cache miss
   and no new kernel.  One small cost remains: a second distinct shape makes
   `torch.compile` treat that dimension as dynamic, which can produce
   slightly slower code than a specialized graph.

## 3. What removal changes

Removal changes ten source files and three test files.  Each edit is
small, and the count is what makes this separate work.  The call sites
to change: `shard_ranges(padded_size)` in `preprocess/mar.py` (three
sites), `preprocess/segmentation.py`, `preprocess/utilities.py`,
`cone_beam.py` (two sites), and `tomography_model.py`; the padded-size
predicates in `utilities.py`; the device report in
`parameter_handler.py`; the drivers' span arithmetic; the memory
ledger; and parallel beam's row alignment.

Three code paths need a new empty-shard branch, because the pad does
real work there today.  A device with no real slices currently holds a
non-empty block of padding, so every per-device loop body runs and
every concatenation has an operand.  Under uneven splits that device
holds a zero-length tensor.
 - `_balanced_slice_bounds` computes a band count from a zero extent
   and a zero band length, and raises before it divides.
 - `exchange_qggmrf_halos` indexes the first slice of an empty shard.
   The guard carries a correctness question with it: the last shard
   that owns real slices must receive the reflected boundary condition,
   not a halo from an empty neighbor.  Under the pad the interface
   masks already neutralize that halo, so this matters only once the
   pad is gone.
 - The back driver concatenates an empty list of owner parts.

Two paths already handle emptiness: the column-gather concatenation
accepts a zero-width piece, and the segmentation histogram skips empty
value sets.

One policy decision is attached to the empty-shard rule.
`_check_no_empty_shard` refuses a layout only when a device is empty on
both axes, judged on padded ranges.  Under a ceil/floor split, some
layouts it refuses today become legal: five views and five slices over
four devices is refused today and admitted uneven.  Removal therefore
widens what the automatic device search may consider.  That widening
should be decided rather than inherited, and admitting the layouts
matches the advice the refusal message already gives.

Values move in the last floating-point digits at every non-dividing
device count, including counts where the real per-device split is
unchanged.  A measured example: summing the squares of a (7, 256, 256)
block gives 4.594492812e5 with a zero-padded eighth view and
4.594492500e5 without it, because torch blocks a reduction by the
element count.  The affected sums are the iteration statistics, the
line-search terms, and the in-band cone reductions.  The sharded
comparison tests gate these with fixed absolute tolerances today, which
the lessons file's own rule rejects for computed floats.  Converting
those gates to the relative form is part of this work.  Every stored
golden is single-device and does not move.  One recorded measurement
does move: the ledger-calibration arms in `tests/test_memory_ledger.py`
include one four-device arm measured on a padded split (`ma512_n4`),
and its assertions need re-verification against the new split.

The memory ledger needs almost nothing.  Its charges are already
per-device functions of the block lengths, so only the block
construction in `plan_from_model` changes.  Its view-batch estimate
stays a slight over-estimate for the shorter shards, which keeps the
batch uniform across devices.

Two bookkeeping steps follow automatically.  Editing `_sharding.py`
changes a hashed cost input of the widening floors, so the staleness
note fires until the hashes are re-blessed with
`refresh_widening_floors.py --bless`; the floors record timings, not
values, so no re-measurement is owed for this change alone.  And the
sharded test file asserts the padding machinery directly in about a
dozen places, with hand-padded guard tests in three more files that
lose their subject and are deleted.

## 4. What does not change

Three things are independent of shard length:
 - the widening floors, which read real sinogram elements;
 - the settled-layout rule of the entry-point plan, which records real
   shapes;
 - the contract that a `Shards` entering a public function must match
   the model's current placement by identity.

The device policy's rule is therefore unchanged, and two of its checks
move with the split: `_layout_is_valid` follows the new empty-shard
rule, and the ledger's charges follow the real shard sizes.

Six code paths were checked and need no work.  Paired operations on
two `Shards` stay shape-compatible, because both operands always share
one placement.  The pixel indices and partitions never touch the
sharded axis.  `gather_column_band` concatenates uneven pieces as it
is.  The HDF5 streaming path reads extents off the tensors rather than
the placement.  The projector transient budget derives its view batch
by rounding up, which stays the largest shard under the new split and
so remains a valid upper bound.  The per-slice crop of the iteration
statistics becomes a no-op rather than a wrong one.

## 5. Sequencing and the decision

The decision belongs before increments 4, 5, and 7 of the entry-point
plan are written.  Three pieces of that plan otherwise get written to
the pad and rewritten later.
 - Increment 4's per-shard `gen_weights` must re-zero padding that the
   removal then deletes.
 - Increment 5's denoiser ledger prices padded shard lengths.
 - Increment 7 ports mbirjax's phantom build, and three of its five
   steps are written to the pad: the zero tail, the crop, and the
   exactly-zero test.  This is the largest of the three rewrites.  The
   phantom build is also self-contained, so if the decision is to
   remove, that increment can build uneven slice bands from the start.

The removal itself runs as separate work with its own review, after the
entry-point increments.  The virtual-CPU instrument,
`configure_devices(devices=['cpu'] * n)`, reproduces uneven splits
locally, so the split and recombine paths are tested before any cluster
run.

The recommendation is to remove the pad.  The change removes a
recurring source of failures and one live defect, and it adds no
capability.  Later work ported from mbirjax would no longer need the
inertness invariant.

## 6. Implementation plan

The work runs as five increments, each reviewed before the next starts.
The order is chosen so that the split changes once, in increment P3,
with the code already tolerant of what the new split produces and the
value gates already trustworthy.  P1, P2, P4, and P5 each leave the test
suite green.  P3 lands as a single commit: the split is one contract
held by every caller at once, so the suite is red between its first edit
and its last, and the green boundary is the increment, not the step.

Every `file:line` reference below is against commit `0089d2b`, the tree
this plan was written from.  P1's edits shift later line numbers in
`tomography_model.py` and `_sharding.py`, so locate each site by the
symbol named beside it and read the number as a hint, not an address.

### P1 — Make the seam tolerate uneven and empty shards

P1 changes no split and no result.  It adds the handling the new split
will need while the pad still guarantees the old shapes, so each branch
is written and tested on its own.  Three sites reach a zero-length
shard:
 - `_slice_band_length` returns a band length of 0 for an extent of 0,
   and `_balanced_slice_bounds` then computes `-(-0 // 0)` and raises
   `ZeroDivisionError` before it reaches `divmod`.  Return an empty list
   of bounds when the extent is not positive; guarding the `divmod`
   alone does not reach the failure.
 - `exchange_qggmrf_halos` reads the first and last slice of each
   neighbor.  Skip empty shards, and give the last shard that owns real
   slices a `None` halo, which the prior maps to the reflected boundary
   condition.
 - The back driver concatenates its list of owner parts, which is empty
   when the owner has no slices.  Build the result directly as
   `torch.zeros((num_pixels, 0), dtype=sino_shards.dtype, device=odev)`;
   there is no part to return, so both dtype and device must be named.

All three edits are inert under the pad, which is what lets P1 land on
its own.  A padded block is never shorter than one entry, so neither the
zero extent nor the empty parts list occurs.  The halo edit is inert for
a subtler reason worth recording: the qGGMRF interface mask at the right
boundary is false exactly when the right neighbor is padding, so a
zero-valued halo and a `None` halo contribute equally today.  The edit
therefore does not change a result now, and it becomes load-bearing
after P3, when that neighbor is empty rather than padded and no mask
exists to neutralize it.

Tests construct the empty and uneven cases directly, by building a
`Placement` and per-device tensors in the test rather than through a
model.  That is what makes P1 testable before the split changes.

### P2 — Convert the sharded value gates to the relative form

The sharded-versus-single comparisons use fixed absolute tolerances
today, which the lessons file rejects for computed floats.  P2 converts
them to the scale-invariant relative maximum,
`max|out - ref| / max|ref|`, at the levels the lessons file already
calibrates: 1e-5 for a single-shot comparison that touches the
projectors, 1e-4 for an iterated one, 1e-6 for purely elementwise
kernels, and 1e-3 for a full-pipeline comparison.  P2 changes no library
code.  The absolute gates all live in `tests/test_sharding.py` (the
`atol=` comparisons, including the two mixed `rtol`-plus-`atol` forms);
`tests/test_kernels_sharded.py` and `tests/test_sharded_segmentation.py`
already compare in the relative form, and the remaining sharded files
carry no absolute gates.

Two kinds of assertion are deliberately excluded, because the lessons
file keeps exact equality for both.  Data-movement round trips stay
exact: shard, gather, and assemble identities compare bytes in against
bytes out, and a tolerance there would mask corruption.  Constructed-zero
assertions also stay exact for now; they are deleted with their subject
in P3, not loosened here.

P2 comes before the split so that P3's value drift is read against a
correct ruler.  It also stands on its own: the conversion is right
whether or not the pad is removed, so P2 can land even if the decision
goes the other way.

### P3 — Change the split

One commit, six steps.

Step 1, `Placement`.  `shard_ranges` takes the axis length as an
optional argument defaulting to `real_size`, and splits it into
contiguous blocks whose lengths differ by at most one, with the longer
blocks first.  It no longer refuses a non-dividing size, and it raises a
named error when no size is given and `real_size` is `None`, which is
the state of a lazily built trivial placement.  The convention matches
two things already in the tree: `_balanced_slice_bounds`, and
`preprocess/pipeline.py:108`, which already splits views this way with
`np.array_split` and already handles an empty range.  Using
`np.array_split` for the index arithmetic settles the convention by
construction.  `padded_size`, `is_padded`, `real_mask`, and
`padded_shard_ranges` are deleted.

Step 2, the consumers.  Every site below holds the deleted API and must
move in the same commit:

| File | Sites |
|---|---|
| `tomography_model.py` | `_banded_setup` `:520-521`, and the `view_spans` 3-tuple it returns, which becomes a 2-tuple at its consumers `:565, :580, :598, :615, :647, :751, :897, :952, :965`; the row count at `:742`; the padded branches at `:643`, `:890`, `:984-986`; `_check_no_empty_shard` `:1080-1083`; the interface masks `:1125-1131`; `_layout_is_valid` `:1471-1472`; `_sino_row_padding` `:1623-1626`; `_split_to_shards` `:1711-1737`; `_constant_recon` `:1801`; `_gather_shards` `:1906`; `_sino_ones_device_form` `:1962`; the all-padding skip branches `:573-585` and `:935-942`, which lose their input |
| `_memory_ledger.py` | `plan_from_model` `:969-977` |
| `cone_beam.py` | the damping cache key `:456`, the ones-fill `:475`, the split at `:479` |
| `parameter_handler.py` | `_device_report` `:141-147` |
| `utilities.py` | the crop predicates `:168` and `:360` |
| `preprocess/mar.py` | `:55`, `:70`, `:87`, the masks `:256` and `:808`, the real-pixel count `:812` |
| `preprocess/segmentation.py` | `:20` |
| `preprocess/utilities.py` | `:777` |

Two sites are exceptions to the "drop the argument" rule.
`cone_beam.py:479` keeps an explicit size, because its placement may
carry `real_size is None` on a single-device model that has never been
reconfigured, and dropping the argument there breaks every single-device
cone reconstruction that uses DC damping.  `helical_fdk_z_weight`
`:725` needs no edit at all: it already passes the real slice count,
which is why it raises today.

The `real_mask` arguments in the MAR statistics are removed rather than
replaced, because with no padding every entry is real.

Step 3, the inertness machinery.  Delete the padded-slice re-zero in the
sharded back projection.  Delete the qGGMRF interface masks and their
cache end to end: the builder `_qggmrf_interface_masks` (`:1104-1136`),
the two consumers that fetch the masks and pass one per device
(`tomography_model.py:2671-2714` and `denoising.py:365-402`), and the
kernel's `interface_mask` parameter (`qggmrf.py:78`, applied at
`:142-146`); the halo exchange itself stays as P1 left it.  Delete
`_sino_row_padding` and the row work at its three call sites: the
`row_pad` argument of `_shard_sinogram` (`:1571`) and the `row_pad`
parameter of `_split_to_shards` it feeds, the row crop in
`_gather_sinogram` (`:1769`), and the row fill in
`_sino_ones_device_form` (`:1959`).  Delete the axis crop in
`_gather_shards`.  In `_split_to_shards`, the comparison against the
padded length collapses because the two lengths are now equal; keep the
shape validation and its message, and drop only the clause that offers
the prepared device-form size as an alternative.  A genuinely prepared
array arrives as a `Shards` and is intercepted upstream by the
placement-identity checks, so no ambiguity arises.

Step 4, the ledger.  `plan_from_model` builds its per-device blocks from
the new split, and the pairs of padded length and real count collapse to
one number: `view_blocks` and `slice_blocks` become lists of per-device
block lengths.  The reads follow mechanically -- the block-length reads
(`[0]`) at `_memory_ledger.py:273`, `:294`, `:297`, `:300`, `:323`, and
`:422`, and the real-count reads (`[1]`) at `:303`, `:306`, `:431`, and
`:565`, where a positive real count becomes a positive length.
`sino_rows` loses its padded branch (`:976-978`).

Step 5, the empty-shard rule.  `_check_no_empty_shard` and
`_layout_is_valid` express the same rule against the new split, which
reduces to refusing a device count above both axis lengths.  This admits
layouts refused today, so it is the one user-visible behavior change in
the campaign and needs the ruling of §3.

Step 6, the tests that assert the deleted API.  These land in the same
commit, because they cannot run against it: the direct assertions in
`tests/test_sharding.py`, and the five fixtures that reach it --
`make_plan` and `_measured_arm_ledger` in `tests/test_memory_ledger.py`,
and the three pad-building helpers in
`tests/test_sharded_segmentation.py`: `_as_shards` (`:32`),
`_view_sharded` (`:114`), and the local `as_shards` inside
`test_sharded_save_and_export_stream_by_slab` (`:333`).  The first of
these, `tests/test_sharding.py:20-31`, asserts that a non-dividing
`shard_ranges` RAISES, which the new contract inverts.  One assertion
outside those files breaks with them:
`test_device_report_names_the_settled_layout`
(`tests/test_logging.py:131`) asserts the report's padded `11 -> 12`
clause, which step 2 deletes with `_device_report`'s padding lines; drop
that assertion and keep the layout assertion beside it.

### P4 — Replace the retired coverage and re-bless the bookkeeping

P3 deletes assertions.  P4 replaces the coverage they carried, which is
the larger job: the fixtures of step 6 feed roughly sixty tests between
them, most of the memory-ledger file among them.

Two assertions must be rewritten rather than deleted, because the new
split makes them vacuously true.  `tests/test_sharding.py:713` and
`:785` assert that an all-padding shard's tensor sums to zero; under the
new split those tensors have zero length, so the sum is zero by
definition and the assertion stops testing anything.  They become shape
assertions on the empty shard plus value assertions on the shards that
own data — on exactly the thin-volume and sparse-view extensions that
P3 step 5 widens.

New tests replace the deleted ones: the split is balanced, its lengths
differ by at most one, the longer blocks come first, and no gather crops
anything.

The hand-padded guard tests of §3 also retire here, because their
subject is gone: `test_helical_z_weight_zeroes_padded_slices`
(`tests/test_cone.py:157`) and the two padded-invariance tests of the
auto-regularization statistics (`tests/test_params_and_paths.py:192` and
`:205`).  All three still pass at P3, because the defensive branches
they drive survive the split as dead code, so they retire here rather
than in the split commit -- and the branches go with them: the helical
tail zeroing (`cone_beam.py:736-739`) and the auto-regularization row
crop (`tomography_model.py:2161-2169`).  The poison-the-padding tests of
the Triton suites are about the kernels' pixel-tile padding, which is
unrelated to the sharding pad; they stay.

Two bookkeeping steps close P4.  First the re-bless.  P1 and P3 between
them edit four hashed cost inputs of the widening floors --
`_sharding.py`, which is hashed whole, and the three sharded driver
methods -- so the staleness note fires from P1 on; it warns and passes
by design, and deferring the re-bless to here reddens nothing.  The
hashes are re-blessed with `refresh_widening_floors.py --bless`, and the
justification is that every measured floor cell divides on both axes at
the counts it was measured at, so the new split is identical there and
the recorded timings cannot move.  Second, the ledger-calibration arms
include one four-device arm measured on a padded split, whose modeled
peaks move by about one part in 128; its floor and band assertions are
re-read against the new split.

### P5 — Verify on the cluster

The virtual-CPU instrument covers the split and the values locally, at a
device count that does not divide the axis.  P5 adds what a laptop
cannot show: a multi-GPU run at a non-dividing count, checking values
against the single-device reference and per-device peak memory against
the ledger.  Re-verifying the four-device ledger arm from P4 belongs to
this run.

One local case belongs with P4, because no existing test can reach the
layouts the widened rule admits: a layout the old rule refuses cannot
appear in today's suite.  The thin-volume and sparse-view tests already
drive public entry points on virtual CPUs at a device count above one
axis (`tests/test_sharding.py:702` and `:770`), and after P3 those same
runs produce zero-length shards through a public entry.  What none of
them exercises is a layout that is refused today and admitted uneven,
so P4 adds a virtual-CPU case that drives a public entry point end to
end on §3's example: five views and five slices over four devices.

### Interaction with the entry-point increments

Three of the entry-point increments write code whose form depends on
this decision:
 - Increment 4 gives `gen_weights` a per-shard form.  Under the pad it
   must re-zero three of its four weight types; without the pad it must
   not.
 - Increment 5 builds the denoiser's ledger, which prices per-device
   block lengths.
 - Increment 7 ports the phantom build.  Three of its five steps are
   written to the pad: the zero tail, the crop, and the test that the
   tail is exactly zero.

The recommendation is to run this campaign between entry-point
increments 3 and 4.  Increment 1 is implemented, and increment 3 is
unaffected.  Increment 2 is nearly unaffected: it adds a direct-recon
ledger plan, and those new charges are written against the same
per-device block pairs that P3 step 4 collapses, so they follow that
step mechanically.  Running P1 through P5 at that point means increments
4, 5, and 7 are written once, against the final form.

Two alternatives are worth naming.  Running the campaign after all eight
increments costs the three rewrites above, of which the phantom port is
the largest.  Running it before increment 1 was not considered, because
increment 1 is already implemented.  If the campaign is deferred,
increment 7 should still build uneven slice bands from the start: its
bands are independent, it gathers to the host, and it needs no
interoperation with a model's placement, so the unpadded build is
simpler than the padded one in either world.
