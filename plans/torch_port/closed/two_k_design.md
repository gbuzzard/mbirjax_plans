# The 2048-class design note

**Status: ACTIVE.  The combining-step ruling is in (§3.5, accepted),
the baseline runs have run, and the table is validated (§6).**
Opened 2026-08-16.  The capacity table in §2 was computed by
`plans/experiments/torch_port/mg17_capacity_table.py`, job 15307591 on
h001, on the synced 78b4f78 tree; the rows are
`plans/experiments/torch_port/rows/mg17_capacity_h001_20260816_223415.jsonl`.

## Executive summary

A 2048-class reconstruction is feasible on one node, with no new
mechanisms.  The library's memory ledger, evaluated in the production
configuration, gives the capacity table in §2.  Three results follow
from it:

* One and two GPUs cannot hold a 2048-class reconstruction.  Three
  GPUs fit with about 2 GiB to spare, and four or more fit
  comfortably.
* The memory bottleneck at every workable count is a set of six
  per-device arrays, and each shrinks as GPUs are added.  No tiling
  or restructuring is needed for capacity.
* The back-projection combining step is no longer a bottleneck.  Greg's 2026-08-11
  change already made its cost fall with the device count, so the
  restructuring this note was asked to design has in effect already
  landed.

One decision remains (§3.5): accept the landed combining step and
close that item, or commission further combining work.  The
recommendation is to accept.  The table is a model, calibrated at
smaller sizes, and the first 2048-class runs validate it at scale
(§4).

## 1. Scope and terms

This note is the design record for 2048-class reconstructions, which
are the low end of production.  It opens with the capacity table the
campaign plan names as the entry point, and it will grow as the
2048-class work proceeds.  

Definition: The combining step is the stage of a
sharded back projection that sums the per-device partial results onto
each slice-owner.

Units:  Sizes in this note are GiB (1024**3 bytes), the unit the ledger and the
harness print.  Earlier campaign notes quoted the same arrays in
decimal GB (1000**3), so the sinogram that appeared there as 32.8 GB appears
here as 30.5 GiB.

Method: The results below are based on calculation rather than cluster runs.

## 2. The capacity table

The reference problem is the 2048-class cell.  Its sinogram shape is
(2048, 2016, 1984) as (views, detector rows, channels), and the default
reconstruction shape is (1984, 1984, 2016).  At four bytes per value
the sinogram is 30.5 GiB, the reconstruction volume 29.6 GiB, and one
full-pixel-set cylinder stack 23.2 GiB.

The instrument is the library's own ledger, evaluated in the production
configuration.  `plan_from_model` built each candidate plan and
`estimate_peak_device_bytes` priced it, on an H100 node so that the
Triton kernel bodies bind exactly as production binds them.  No
reconstruction ran.  The verdict column applies the preflight's own
rule: a count fits when 1.15 times the modeled peak is at or under the
measured idle budget, which read 78.67 GiB on h001.  The binding
phase of a count is the phase whose charge sets that count's peak.

Each count was priced under four combining-step variants.  The variants
differ only in what the slice-owner holds while partial back
projections are combined onto it:

* **today** — the shipped code.  `sum_band_to_owner` streams each
  arriving partial in bounded row slabs (commit 413aeb0, 2026-08-11).
* **band_knob** — the shipped code with `back_project_slice_band` set
  to a quarter of the shard, the existing memory lever.
* **reduce_min** — the combining transient replaced by the bare
  output, the floor any further restructuring could reach.
* **pre_stream** — the pre-2026-08-11 form, in which the owner held
  every peer's whole partial beside the running totals.  This is the
  form the open-items file describes.

The table, for cone.  Parallel reads 0.03 to 0.04 GiB lower at every
entry and identical in every verdict and binding phase.

| n | today | band_knob | reduce_min | pre_stream | binding phase (today) | fits (today) |
|---|---|---|---|---|---|---|
| 1 | 181.5 | — | — | — | per-iteration statistics | NO |
| 2 | 97.9 | 97.9 | 97.9 | 97.9 | subset delta forward projection | NO |
| 3 | 66.7 | 66.7 | 66.7 | 69.2 | subset delta forward projection | yes (edge) |
| 4 | 51.1 | 51.1 | 51.1 | 57.8 | subset delta forward projection | yes |
| 5 | 41.8 | 41.8 | 41.8 | 51.0 | subset delta forward projection | yes |
| 6 | 35.6 | 35.6 | 35.6 | 46.4 | subset delta forward projection | yes |
| 7 | 31.1 | 31.1 | 31.1 | 43.1 | subset delta forward projection | yes |
| 8 | 27.8 | 27.8 | 27.8 | 40.7 | subset delta forward projection | yes |

Five readings.

1. **One device is impossible, and two devices do not fit either.**
   The two-device peak is 97.9 GiB against a 78.67 GiB budget.  The
   binding phase there holds six shard-sized arrays at once, so the
   shortfall is structural rather than marginal (§2.1).
2. **Three devices fit at the edge.**  The demand at three devices is
   76.7 GiB, which leaves 1.9 GiB of slack.  A neighbor process or
   allocator fragmentation could consume that slack, so three devices
   is the modeled floor, not the recommended operating point.
3. **Four devices fit comfortably**, at a 58.8 GiB demand, and every
   count above four fits with a growing margin.
4. **The combining step binds nowhere under today's code.**  The
   today, band_knob, and reduce_min columns are identical at every
   count.  Removing the combining transient entirely would change no
   peak, so no further combining work can move this table.
5. **The pre-streaming form was the binding term the earlier notes
   said it was.**  Under pre_stream the binding phase at three or more devices
   is the hessian's band reduce, and the three-device demand rises to
   79.6 GiB, which does not fit.  The streaming change of 2026-08-11
   therefore moved the feasibility floor from four devices to three.

### 2.1 The term that binds now

At every count from two through eight, the binding phase is the subset
delta forward projection at the coarsest visited granularity, which is
the four-subset partition.  Six arrays set its size, and each is a
1/n share of a full-size array: the error sinogram, the weights, the
delta sinogram, the forward assembly block, the flat reconstruction,
and the hessian diagonal.  The four sinogram-shaped shards read
7.6 GiB each at four devices, and the two reconstruction-shaped shards
7.4 GiB each.  The remaining terms of the phase are the subset's
cylinder arrays and the projector's bounded batch and block terms,
about 5.8 GiB together at four devices.  The six shard arrays fall as
1/n, and the remaining terms are bounded by fixed budgets, which is
why the whole column scales cleanly with the device count.

One known over-charge sits inside this phase.  The forward assembly
term charges a second sinogram-shaped block, and on the column-gather
path the driver adds into the owner's block instead of holding a
second one.  The ledger keeps the charge deliberately, because the
banded path really holds both.  Subtracting it changes no verdict: at
two devices the peak would still be 82.6 GiB and the demand 95.0 GiB.

### 2.2 What the table rests on

The table is the model's statement, not a measurement.  The closed
forms behind it are calibrated where measurements exist: the mg11 flip
gates measured every 1024-class gather arm inside the 1.00-to-1.30
band, and the mg15 gate re-verified the 512-class arms after the
padding removal, with the streamed combining charge in place, at a
lowest ratio of 1.016.  The 1024-class anchor rows in the mg17 output
reproduce the modeled peaks those measurements validated.  No
measurement exists at the 2048 class; the first composed 2048-class
runs are what will validate the table there, and they are the next
item in the campaign's order.

The harness verifies its own arithmetic.  The mg17 script recomputes
the shipped combining charge from the plan fields and asserts equality
with the ledger's stored term on every phase of every row, so the
derived variant columns cannot drift from the closed forms they edit.

## 3. The combining step

### 3.1 The premise has changed since the item was written

The open-items file describes the combining step this way: one GPU
collects and adds the partial results, holding about 37 GB at the 2048
class, and that cost does not fall as GPUs are added.  That was true
when the underlying notes were written on 2026-08-09 and 2026-08-10.
It stopped being true on 2026-08-11, when commit 413aeb0 ("Reduce
memory in cross device streaming") landed the streamed reduce, with
the ledger charge, the sharding tests, and the developer documentation
updated in the same commit.  The 2026-08-16 open-items compilation
carried the earlier description forward, so the item's premise is
stale, not wrong when written.

### 3.2 What the shipped combining step holds

`sum_band_to_owner` now forms the running total for a band on the
slice-owner and adds each arriving partial one bounded row slab at a
time.  A slab is at most `REDUCE_SLAB_BYTES`, 64 MiB when this note
was computed and 256 MiB since the 2026-08-17 sweep.  At the
widest instant the owner holds its shard-sized output, one band-sized
partial of its own that the driver keeps alive across the reduce, and
one slab per peer.  At the default band, which is the whole shard,
that is two shard-sized arrays plus the slabs.  The transient above
the output is therefore one band copy, 23.2/n GiB at the 2048 class,
plus (n−1) slabs, which is 6.0 GiB at four devices.  The combining
step runs inside three phases, and its largest carrier, the hessian's
reduce sub-phase, models at about 34.5 GiB at four devices, about
17 GiB below that count's peak.

The summation order is unchanged from the pre-streaming form, so the
change moved no values.

### 3.3 What the pre-streaming form held

The old form moved every peer's whole partial onto the owner before
summing.  The owner held n+1 band copies at two devices and n+2 at
three or more, which is 1.5 full cylinder sets at both two and four
devices, 34.8 GiB at the 2048 class.  That cost was flat in the device
count, and the pre_stream column shows what it would have done at the
2048 class: bound the peak from three devices upward and pushed the
feasibility floor to four.

### 3.4 The options, each against the table

**Accept the landed streamed reduce.**  No ledger term changes and no
code changes.  The table shows the combining step non-binding at every
count through eight, so no further combining change can lower any
peak at single-node scale.

**In-place accumulation.**  The reduce could accumulate into the
owner's own partial instead of forming a separate running total,
removing the one band copy the driver keeps alive.  The `band reduce`
term would fall from shard-plus-band to shard-plus-slabs.  The saving
is 23.2/n GiB on a phase that already sits about 17 GiB below the
peak, so the table assigns it no capacity effect at any count through
eight.  It would also change the summation order on every owner except
the first device, because the running total starts from the first
device's partial in device order.  The goldens and the value
expectations would therefore move at the float floor.  Not
recommended.

**Tree-shaped combining.**  A tree reduces the additions in log2(n)
rounds across the devices instead of serially on the owner.  It does
not lower the owner's modeled peak, because the streamed form already
bounds the owner's transient by slabs rather than by peer count.  Its
case would be time at large device counts.  The measured time of the
whole reduce is 0.04 s per pass at the 1024 class, so there is no time
case at single-node counts either.  Declined by this evidence through eight
devices; a multi-node design would re-open it.

**The band knob.**  `back_project_slice_band` already ships and
shrinks the combining phase further, to shard-plus-band with a band
smaller than the shard.  The table shows it changes no 2048-class
verdict.  It remains the documented emergency lever for a squeezed
device.

**The slab-size sweep.**  The 64 MiB slab is a reasoned default, and
the open measurement item for it still stands.  The slabs total
0.44 GiB at eight devices, so the sweep is a time calibration, not a
capacity question.  It can be measured inside the first 2048-class
baseline runs at no extra job cost.

### 3.5 The decision

The question the campaign registered was whether to restructure the
combining step so that 2048-class memory scales down with the device
count.  The code has answered it: the streamed reduce landed on
2026-08-11, and the table shows the combining step no longer binds at
any count through eight.  What remains is a bookkeeping ruling, and it
is Greg's:

* **(a) Accept the landed streamed reduce as this item's resolution.**
  Re-point the open item to this note, and carry one rider: the
  slab-size sweep joins the 2048-class baseline runs.  This is the
  recommendation.
* **(b) Commission further combining work now**, meaning in-place
  accumulation or tree-shaped combining.  The table gives neither a
  capacity case nor a time case at single-node counts, so this option
  costs review and validation effort and the model predicts no saving
  from it.

RULED (Greg, 2026-08-17): option (a).  The landed streamed reduce is
accepted as this item's resolution, and the slab-size sweep rides the
2048-class baseline runs.

## 4. What the table sets up for the baseline runs

The first composed 2048-class runs should target four devices as the
operating point and three devices as the model's edge.  Four is the
comfortable floor, and a three-device run validates the ledger exactly
where its slack is thinnest.  Two devices need not be attempted: the
modeled peak sits 25 percent over budget even with the known
over-charge subtracted.  The forward pixel-batch sweep is
memory-unconstrained at this scale, because the gathered column
cylinder is 0.19 GiB at batch 8192 and 0.74 GiB at batch 32768, so the
sweep can run inside the same baseline jobs.  No tiling work is required for
capacity at three or more devices; the binding structure is plain 1/n
shards, and the levers that would matter beyond it, such as supplying
the hessian or streaming the delta sinogram, are observations for the
record rather than proposals.

## 5. The baseline runs (the plan)

The first composed 2048-class runs validate the capacity table and
set the pixel batch.  One job carries them, on four H100s, for cone
and parallel.  The job is mg19, and its run record will live beside
the script.

Arms per geometry, each in its own process:

* A generator at four devices stages a seeded phantom and its
  sinogram, with checksums.
* A refusal check at two devices asserts that the memory preflight
  refuses, which is the table's two-device verdict tested for real.
* Composed three-iteration reconstructions at three devices (the
  model's edge) and at four devices (the operating point) run in the
  ledger's calibration mode.  Their modeled-against-measured peaks
  are the memory validation this note's table waits on (item A3).
* The four-device arm at the shipped pixel batch of 8192 runs twice,
  and the pair's spread is the ruler the batch comparisons read
  against.  Three more four-device reconstructions at batches 16384,
  32768, and 65536 extend the batch sweep to production scale
  (item A4).  The 1024-class sweep never bracketed the optimum, so
  this one reaches one doubling further.
* Two four-device cone arms with the combining slab at 16 and
  256 MiB carry the slab-size rider from the combining-step ruling.
  If the per-phase brackets cannot resolve the slab's effect, that
  is the reading, and the sweep gets its own instrument later.

Each composed arm also records per-call device brackets on the
forward and back projections, in the form the earlier forward
instrument used, so the back projection's share of time is read at
this scale.  That share is the input the open two-device cone
back-projection question needs.

Readings against rules: the calibration rows read against the
1.00-to-1.30 band; one forward projection of the staged phantom at
each count reads against the staged sinogram at 1e-4; and the batch
comparisons read against the repeated arm's spread.  A preflight
refusal at three devices would itself be a model-edge finding: it is
recorded, and only then may one repeat with the preflight skipped
obtain the measured peak, with the skip on the row.

## 6. What the baselines read (2026-08-17)

The runs happened and the full record is findings §1.20.  The
verdicts, one line each:

* **The table is validated.**  Every 2048-class calibration ratio
  sits between 1.10 and 1.19, inside the band and never under, and
  both two-device arms were refused by the preflight as §2 predicts.
* **The batch knee brackets at or just above 32768.**  Forward busy
  falls 17 to 18 percent from batch 8192 to 65536, with the last
  doubling worth 2 to 3 percent.  The recommendation is a reviewed
  default change to 32768.
* **The combining slab closes.**  Its whole 16-to-256 MiB range moves
  the back projection by 0.8 percent; 256 MiB is marginally best and
  the default is defensible.
* **The next target is the cone back projection.**  It is more than
  half of a four-device cone wall and its busy time rises from three
  devices to four.  The kernel campaign starts there, and the
  sorted-accumulation question moves there with it.
