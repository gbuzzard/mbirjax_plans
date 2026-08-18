# Design note: the cone back-projection remedy

**Status: DRAFT, awaiting Greg's ruling (§8).**  Opened 2026-08-17,
evening.  Nothing here is implemented.

**What this note is.**  The open item B1 asked why the cone back
projection costs more on more devices.  Three related items appear
below and are named once here: B1 is that anomaly, B2 is the
kernel-width padding increment left open by the width-mechanism
finding, and B3 is the proposal to sort or reorder accumulation
inside a kernel.  The campaign's plan called for a design note in the
forward remedy's pattern: the candidate structures, the 2048-class
arithmetic, the value and memory gates, and the ledger terms each
candidate changes.  The attribution probe ran first (mg21 and mg21b,
findings §1.21), and its answer reshapes this note.  The recorded
candidates were designed against a hypothesis the probe refuted, and
the measured mechanism has a much smaller remedy.  This note presents
that remedy as the proposal and evaluates each recorded candidate
against the measured shares.

**Sources.**  The measured base is
`plans/torch_port/active/multigpu_findings.md` §1.19, §1.20, and
§1.21, with run detail in
`plans/experiments/torch_port/mg21_back_attrib.md` and rows in
`plans/experiments/torch_port/rows/`.  The code read for this note is
`TomographyModel._sparse_back_project_sharded` and
`_slice_band_length` (tomography_model.py lines 576 to 828),
`Projectors.sparse_back_project_view_range` and the transient budget
(projectors.py lines 295 to 579), the cone back kernel and its
wrapper (triton_cone.py lines 202 to 457), `sum_band_to_owner`
(_sharding.py lines 206 to 296), and the back terms of
`_memory_ledger.py` (back_cols at lines 417 to 422, back_fixed at
lines 545 to 580), all on the merged 6d90601 tree.

---

## 1. The finding that shapes this note

The cone back projection's cost rises with the device count, and the
rise is inside the Triton kernel.  mg19 measured the composed back
busy time at the 2048-class cell: 137 s at three devices and 228 s at
four.  mg21 split one full-pixel back projection into named parts on
the busiest device:

| part | n=3 s | n=4 s | ratio |
|---|---|---|---|
| kernel | 24.3 | 45.8 | 1.88 |
| builders | 0.76 | 0.76 | 1.00 |
| channel-major copy and other residual | 0.07 | 0.08 | 1.01 |
| accumulation between body calls | 1.28 | 0.97 | 0.76 |
| cross-device reduce | 0.06 | 0.07 | 1.13 |
| whole pass | 26.5 | 47.7 | 1.80 |

The kernel is the only part that grows.  Its work per device falls to
0.75x from three devices to four while its time rises 1.88x.  These
readings put the kernel's time per unit of work 2.5 times higher at
four devices.

The recorded hypothesis said each band call pays the kernel's
full-detector-row grid.  The probe refuted it twice.  The launch grid
is band-sized: 193023x11 at n=3 and 193023x8 at n=4.  Halving the
band doubles the number of band calls, and so doubles the total of
every per-call cost; it moved the total by 12 percent at n=3 and 2
percent at n=4.

The mechanism is the divisibility specialization §1.19 found on the
parallel forward's width, now on the back kernel's band argument.
Triton compiles a separate kernel for each combination of integer
arguments it can prove divisible by 16.  The unspecialized
compilation uses more registers.  That caps occupancy at 5 blocks per
multiprocessor against the specialized 8, which §1.19 read as 60
percent against 90.  At three devices each band is 672 slices, and
672 is divisible by 16.  At four devices each band is 504, which is
not.  mg21b varied the band alone on one device.  The five divisible
bands ran at one rate and the three non-divisible bands at another, a
ratio of 2.44 on the medians.  The bands 496 and 512 both ran at the
fast rate, on either side of 504.

The 1024-class history is the same effect.  One and three devices
produce bands of 1008 and 336, both divisible; two and four produce
504 and 252, both not.  The cone back projection at two devices
measured 30.3 s at that class (findings §1.7), and its band is 504.

## 2. The proposal: pad the band argument to a multiple of 16

The remedy is B2's padding increment, applied to the cone back
kernel's band argument.  The kernel wrapper rounds the band argument
up to the next multiple of 16 before the launch, allocates the output
partial at the padded width, and returns the real-width slice of it.
The padded columns are computed and discarded.  Nothing above the
wrapper changes: the banded driver, the band knob, the combining
step, and the adjoint contract all keep their current form.

Five design points, decided here so the increment is mechanical:

* **The padded value goes into every use of the band argument.**  The
  band argument feeds the launch grid, the tile mask, the output row
  stride, and the tile-size choice.  The output row stride is the use
  the specialization acts on.  Splitting the argument into a padded
  stride and a real mask bound would still leave a non-divisible
  integer in the launch, and it would produce a compilation nothing
  has measured.  Padding the single argument keeps the kernel
  identical to the form mg21b measured fast; the tile-size choice
  cannot move, because the next power of two of a band and of its
  padded band agree for every band above 8 slices.
* **The padded columns hold values that no caller reads.**  A column
  here is one column of the (pixels, band) output partial, which is
  one slice position of the band -- not the volume's column axis.  A
  padded column's global slice index can point past the band and, at
  the last band, past the volume.  Every address it forms is already
  clamped in the kernel, so no load or store leaves its buffer, and
  the stores land in the padded output only.  The wrapper's sliced
  return discards the values.  The cost is small twice over: the pad
  is at most 15 slices, which is 1.6 percent at both production bands
  (8 in 504, 4 in 252), and at those bands the padded launch grid is
  unchanged (504 and 512 both need 8 tiles of 64), so the pad mostly
  turns already-launched masked lanes into live ones.
* **The kernel's padding contract moves with the change.**  The
  kernel docstring today promises that lanes beyond the band carry
  zeroed weights and masked stores.  Once the band argument is the
  padded value, that promise no longer covers the pad: its loads are
  live and its stores land.  Safety rests on the address clamps and
  the sliced return, and the docstring must say so in the same
  change.
* **The sliced return leaves the partials strided, and three
  consequences are accepted.**  First, the view-range loop's
  accumulation adds one strided view into another, which is correct
  and cheap (each row is a contiguous 2 KB run).  Second,
  `sum_band_to_owner`'s comment that a block of rows is one flat copy
  stops being exact: a strided slab stages one contiguity copy before
  the peer transfer.  The reduce's outputs stay contiguous, so the
  pad does not propagate past it.  Third, at a trivial placement the
  sliced view reaches the result list unreduced; nothing reads it
  wrongly today, and the increment adds a comment where it lands.
* **The increment measures the parallel kernels rather than assuming
  they are unaffected.**  The parallel back at the same cells rose
  1.13x from n=3 to n=4, from 32 s to 36 s (mg19).  Against the same
  0.75x fall in work, that is a 1.5x loss of efficiency, against
  cone's 2.5x.  Its kernel therefore carries much less of the effect
  at this size, and the gate run measures it with one arm rather
  than assuming it.  The parallel forward's production widths are
  already divisible (§1.19); the same wrapper-level pad there is the
  robustness half of B2, so arbitrary user shapes cannot pay the
  two-times penalty either.

One input to these points was measured after the first draft.  The
band start (`slice_start`) is also an integer argument the compiler
specializes, and two of the four production band starts at four
devices (504 and 1512) are not divisible by 16.  The start never
enters address arithmetic; it is added to the slice index as a
float.  The mg21b addendum measured it directly: at a fixed band of
512, starts of 0, 504, and 1008 ran at rates within 0.4 percent of
one another.  The start does not matter, and the band argument alone
governs.

## 3. The 2048-class arithmetic

The projected effect follows from the two measured rates.  At four
devices the band argument goes from 504 to a padded 512, so the
kernel returns to the divisible rate:

* One full-pixel back pass: the kernel falls from 45.8 s to about
  19 s, and the pass from 47.7 s to about 21 s.
* The composed back busy time at n=4: the bare-call estimate rebuilt
  from mg21's variants falls from 207 s to about 92 s.  The composed
  reading should therefore fall from 228 s to near 100 s.  Three
  devices sit on a divisible band already and do not move.  Back
  busy time then becomes monotone again: 137 s at n=3 against about
  100 at n=4, in place of today's rise to 228.
* The four-device cone wall: 420 s falls to about 290 s, and the
  back projection stops being more than half of it.
* The 1024-class two-device back (band 504) should fall by the same
  ratio, from 30.3 s to the 13 s class, which closes the anomaly
  that opened B1.

These are projections from measured rates, not measurements.  The
confirmation runs in §7 turn them into recorded numbers.

## 4. The ledger terms

The padding touches the terms that price the band-sized partials.
Two charges grow by the pad over the band: the live back blocks
(`back_fixed` charges two, the accumulator and the incoming block,
and the driver really holds both) and the band-reduce transient.
The cone back view charge does not move: for a two-fan geometry it
is priced from the detector rows, not the band
(`back_cols`, _memory_ledger.py lines 417 to 422).

The sizes at the 2048-class cell, four devices, full pixels: one
band partial is 5.80 GiB, its 8-slice pad is 94 MiB, and the two
live blocks together grow by 188 MiB.  The three-device slack in the
capacity table is 1.9 GiB, and the n=3 band takes no pad at all, so
no verdict in the table moves.  The ledger change is one rule.  The
charged band length becomes the padded band length, read from the
same helper the wrapper uses.  The code and the charge then cannot
disagree.

## 5. The value gates

The padded columns are discarded, so the per-element sums are the
same arithmetic as today.  The compilation itself is what can move
values, because the specialized kernel may schedule and contract
floats differently.  mg21b measured that difference directly.  Its
arms compared outputs across compilations, a 512-band launch against
a 672-band launch on the slices they share, and every comparison
read exactly zero.  That is the padding's own value question,
answered before the increment starts.

The gates do not rely on it.  The recorded calibration is lessons
§2: anything touching the projectors gates at 1e-5 relative
single-shot and 1e-4 iterated.  The increment gates there, against
the unpadded code, and the standing suites run unchanged: the banded
parity and adjoint tests at CPU virtual counts, the two-CUDA-device
kernel arms, and the goldens.

## 6. The recorded candidates, against the measured shares

**A cylinder-transfer counterpart for the back: declined.**  Its
case was the partial traffic, the per-band barriers, and the
channel-major copies.  mg21 measured those at 0.07 s, under 0.02 s,
and 0.08 s against a 47.7 s pass.  It would also change the
summation order, which moves every gate to the 1e-3 class.  It would
add a sinogram-window transfer primitive, and it is the largest
change of any option considered, for terms measured at under a fifth
of a second.

**A band-sized kernel grid: closed as moot.**  The grid is already
band-sized; the probe recorded every launch.  The one full-row cost
inside a band call is the channel-major copy, measured at 0.44 ms
per body call and 0.07 s per pass, which does not justify a change.

**Sorted accumulation inside the kernel (B3): deferred, with its
precondition named.**  The back kernel gathers and stores once per
output element, with no atomics.  The mg20 counters measured the
forward's atomic write path, so they do not describe this kernel.
After the padding is in place, the back kernel runs at the divisible
rate, and what remains is the efficiency of a single call.  Whether
sorting or reordering its gathers would improve that needs a counter
run on this kernel first.  That run belongs to the kernel campaign,
after the padding's confirmation runs.

## 7. Implementation increments

1. **The pad, both kernels.**  The cone back wrapper pads its band
   argument; the parallel kernels take the same wrapper-level pad on
   their width arguments.  The kernel docstring's padding contract
   and the ledger's band-length rule move in the same change (§2,
   §4).  The change is small and stays inside the kernel wrappers;
   no driver changes.
2. **The gates.**  The full suite, the CPU virtual-count parity
   tests, and a two-CUDA-device kernel arm run unchanged, with the
   1e-5 and 1e-4 value gates of §5.  One arm measures the parallel
   back's width sensitivity, which is the audit §2 names.  The band
   start needs no arm: the mg21b addendum measured it and it does
   not matter (§2).
3. **The confirmation runs.**  The 2048-class cone composed arms at
   three and four devices (mg19's arms, re-run on the padded tree)
   and the 1024-class two-device back.  The readings against §3's
   projections are the acceptance evidence.
4. **The floors consequence.**  The pad changes projection cost, so
   the staleness machinery will name it and the floors will owe a
   re-measure.  The mg22 refresh now running stays valid for the
   unpadded tree it measures.  The rows the padding will move are
   the cone and multiaxis rows at non-divisible bands.  A second
   refresh after the padding is committed re-anchors those rows.
   This is what the staleness rule is for.

## 8. The decision

The cone back projection's anti-scaling is the compiler's
divisibility specialization on the band argument.  Two runs measured
it, and their rates agree to within 3 percent.  That answers the
question B1 opened.  The ruling this note asks for:

* **(a) Approve the padding increment (§2, §7) as the back remedy.**
  This is the recommendation.  It is small, value-safe, and
  projected to restore monotone back scaling at the 2048 class and
  to close the 1024-class two-device anomaly.  The transfer
  counterpart is declined, the grid candidate is moot, and B3 waits
  behind a counter run on this kernel.
* **(b) Commission one of the structural candidates instead or in
  addition.**  The measured shares give none of them a term worth
  its cost today; §6 records the arithmetic each would need to
  overturn.

One question needs an answer either way: when to schedule the
confirmation runs.  They need one quiet 4-GPU window of about an
hour.
