# Removing the banded forward path

**Status: COMPLETED, 2026-08-17.**  Greg
ruled to remove the banded multi-device forward after weighing its
four retention arguments against maintenance cost and readability.
The merge waits on today's confirmation runs: the manual nightly on
the flipped tree and the 2048-class baselines.

## What is removed, and what stays

The column gather is now the measured winner on all four projection
geometries, so the banded walk becomes forward-path dead code behind
a switch that CI exercises and production never runs.  The removal
deletes, in the forward direction only:

* the banded driver body inside the sharded forward;
* the driver selection: `_column_gather_forward`, the
  `column_gather_geometry` class attributes, the
  `forward_column_gather` switch, and the environment variable;
* the `forward_project_slice_band` knob;
* `broadcast_band_to_views` in the sharding module, whose only caller
  is the banded forward;
* the ledger's banded-forward terms (the broadcast-band charge and
  the banded branches of the call-shape helpers), with
  `column_pixel_batch` becoming unconditional for projection models;
* the forced-banded test dimension, which collapses the suite's three
  environment states to one.

Nothing in the back projection moves.  The banded structure there is
the shipped adjoint: the band loop, the streamed combining step, the
`back_project_slice_band` knob, and the shared helpers they use all
stay.

## The property the change rests on

The removal is behavior-neutral for every default configuration.  The
gather is already the only path a default run takes, so the deletion
removes unreachable-by-default code and its selection logic, and no
golden, floor, or default value may move.  The gates below check that
directly.

## Increments

1. **Delete the forward driver and its selection.**  The sharded
   forward keeps its trivial-placement branch and otherwise always
   gathers.
2. **Simplify the ledger.**  Remove the banded-forward terms and the
   `forward_band` plan field.  The pinned calibration rows that were
   measured on the banded path can no longer be priced; replace them
   with rows measured on the gather (the 2026-08-17 comparison job
   measured per-device peaks for both torch-body geometries at two
   and four devices), with provenance.  The exact replacement row set
   is a review point.
3. **Collapse the tests.**  Remove the banded-forced parametrization
   and the environment-state matrix.  The three pre-existing failures
   in the forced-banded state disappear with the state; the fragile
   budget helper behind them is tracked separately.
4. **Documentation.**  The developer sharding page's forward
   description drops the banded walk; the back projection's banded
   description stays.
5. **Floors bookkeeping.**  The staleness machinery hashes the
   deleted method by name, so the removal trips it; re-bless with the
   defaults-unchanged defense recorded, no re-measurement.

## Gates

* The full suite passes in the single remaining environment state.
* CPU virtual-device forwards at two and three devices agree with one
  device on all four geometries, before and after the removal, at the
  same distances.
* A before-and-after check on one sharded forward shows bit-identical
  output, which is what behavior-neutral means for a compiled path
  that did not change.
* No golden moves, no floor value moves, no default changes.

## The rename that rides the removal

Ruled (Greg, 2026-08-17): the approach formerly named "column gather"
is renamed to **cylinder transfer**, in the same worktree.  The
reasons: "column" collides with the volume's (rows, columns, slices)
axis convention, and "gather" carries collective and indexed-read
meanings that do not describe this driver.  The moved object is a
batch of full-height voxel cylinders, the word the docstrings already
prefer.

The mapping, for readers of the historical records, which keep the
old vocabulary:

| old | new |
|---|---|
| the column gather (the approach) | the cylinder transfer |
| `_sparse_forward_project_columns` | `_sparse_forward_project_cylinders` |
| `gather_column_band` / `_async` | `transfer_cylinder_batch` / `_async` |
| `wait_for_column_band` | `wait_for_cylinder_batch` |
| `COLUMN_GATHER_RESIDENTS` | `CYLINDER_TRANSFER_RESIDENTS` |
| ledger helper `forward_column_cylinder` | `forward_transferred_cylinders` |
| ledger helper `column_gather_slices` | `whole_slice_extent` |
| the 'column cylinder' ledger term | 'transferred cylinders' |
| `LedgerPlan.column_pixel_batch` | `LedgerPlan.pixel_batch` |

One bookkeeping consequence: the recorded floors hash table still
keys the old method name, left byte-for-byte untouched so its tamper
checksum stays valid; the merge-time re-bless rewrites the key and
clears the staleness note's five entries at once.

Every findings section and mg script written before this ruling says
"column gather"; those records are not rewritten.

## Merge

The work lives in a worktree branched from the flip commit, left
uncommitted for Greg's review and commit.  Merge after the manual
nightly and the 2048-class baselines confirm the flipped tree.
