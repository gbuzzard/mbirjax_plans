# The multiaxis kernel campaign

**Ruled 2026-08-22 (Greg).**  Hand-written kernels will be built for
the multiaxis geometry now.  Translation waits, and the multiaxis
work will be evaluated to price a translation campaign afterward.
The decision evidence is findings §1.42 and §1.43.  The decision
rule is the recorded one: need above the 1024-class, not elegance
(multigpu_plan_part_2.md, the escalation path).

## Why kernels, from the measurements

The case has three parts, each measured.

* **Capacity.**  The torch bodies hold about 14 slabs of
  (view batch, pixels, max(rows, slices)) floats at once, which is
  the memory ledger's calibrated multiplier
  (_memory_ledger.py:200-223).  The ledger models the multiaxis
  1024-class at 68 GB on one device, and sharding does not divide
  torch-body peaks.  The 2048-class therefore cannot run on
  the standard node.  A kernel body holds the problem arrays plus a
  12-byte per-(view, pixel) contract, so the slabs disappear.
* **Speed.**  The back direction's top kernels sit at the memory
  ceiling, and single launches move 7 to 9.7 GB of intermediates
  against 0.65 GB of problem arrays (findings §1.42).  mg53 then
  showed the apparent host pacing at the 1024-class is the
  device's own rate reflected through a full launch queue
  (findings §1.44), so both cells are device-bound and the speed
  target is the device time the slab traffic costs.  mbirtorch
  already meets or beats mbirjax at every torch-body cell
  (findings §1.43), so there is no catching-up motive; the target
  is the absolute limits both frameworks share.
* **Multi-device.**  The mg44 losses were the multiaxis and
  translation back bodies running uncompiled after torch's
  recompile budget filled (findings §1.36).  A hand-written kernel
  body carries the no-compile marker, so that whole mechanism
  class cannot apply to it.

## What the campaign must deliver

Four goals, each checkable.

1. **Values unchanged.**  The kernels pass the gate family of the
   first campaign, listed under Gates below.
2. **Capacity.**  A seeded 3-iteration 2048-class multiaxis
   reconstruction completes on the standard node, with its peak
   memory and wall recorded.  Today's tree cannot run it.
3. **Speed.**  The kernel route meets or beats the compiled bodies
   at the three multiaxis floors cells on one device.  The
   expectation is set honestly: both cells are device-bound
   (findings §1.44), so the win must come from removing the slab
   traffic; the 512-class compiled forward already runs near the
   arithmetic ceiling, and the best-supported headroom is the
   memory-ceiling back direction and the 1024-class.
4. **Multi-device.**  The 1024-class is re-measured at one, two,
   and four devices, and the mg44 back-projection loss class is
   absent.

## The design, and where D7 is decided

The survey of the bodies and the existing kernels (2026-08-22,
against the committed tip 4bb3be5) found the structure favorable.
Three of its facts determine the design.

* **The per-(view, pixel) contract is 12 bytes**: the channel
  coordinate, its integer center, and the row anchor m0.  The
  slice-to-row slope, both footprints, and both amplitude scales
  are per-view scalars, because parallel-beam geometry has no
  magnification (multiaxis_parallel.py:79-88).  Cone's contract is
  28 bytes held and 48 charged.  Multiaxis kernels therefore carry
  less traffic per tap than the cone kernels that already won.
* **The back kernel is a close cone-back adaptation.**  Gather on
  both axes into a register accumulator, no atomic adds, one store
  per output element, coeff_power as a compile-time constant, slice
  bands that concatenate, and clamp-style addressing with no padded
  sinogram copy (triton_cone.py:204-330).  Multiaxis adds three
  things cone lacks: the valid_k mask that zeroes slices past the
  real count, the mass-conserving vertical amplitude applied as
  scaling ** coeff_power after the weight power, and one shared
  psf_radius serving both fans.
* **The forward kernel has one open choice, on the vertical
  axis.**  The torch body scatters slices into detector rows and
  then scatters channels (multiaxis_parallel.py:144).  A kernel
  can mirror that scatter, which reproduces the body's arithmetic
  but multiplies the atomic adds by the vertical tap count, and
  the per-tap parallel history says atomic adds become the limit.
  Or it can gather on the vertical axis the way the cone forward
  does: invert the per-view affine from rows back to slices, sum
  slice taps in registers, and scatter only the channels.  The
  inversion divides by the per-view slope, which vanishes only at
  90-degree elevation; the model already warns above 45 degrees,
  and the kernel clamps the slice range to the band, so the cost
  of extreme elevation is extra loop iterations, not wrong values.

This choice is exactly item D7's question for multiaxis, and
Charlie's hypothesis (gather on the vertical axis, scatter on the
horizontal) is the cone-style candidate.  The campaign builds the
gather-vertical form as the primary, because the cone forward
ships that organization successfully and the scatter-mirror form
triples the atomic density.  The scatter-mirror form is built for
comparison only if the gather form's value behavior or speed
disappoints.  Either way the measured evidence is recorded, which
closes D7's multiaxis half.  D7's translation half stays open with
translation.

One speed option is recorded and not scheduled: the channel-sorted
contraction that earned the parallel forward 3.97x should port,
because the horizontal weights are per-view exactly as parallel's
are, with the vertical fan added on top.  It is attempted only if
the per-tap forward's measured limit is its atomic adds.  The cone
counterexample is the caution: the same rework lost (0.24x and
0.87x) on a kernel whose limit was its gathers.

## The attribution run (mg53, done 2026-08-22)

mg53 split the 35 ms per-batch host cost and found it is not host
work.  Kernel launches block when the runtime's command queue
fills, so at the 1024-class the host clock reads the device's own
rate: the blocked interval is 91 percent of the forward call's
host time and 81 percent of the back's, the compiled dispatch is
0.16 ms per batch, the allocator made zero device allocations,
and no synchronization fires (findings §1.44).  Two consequences
for this plan.  The speed case rests on the device-side evidence
alone, which is exactly what fusing the slabs away attacks.  And
no extra wrapper discipline is needed beyond the recorded
hoisted-builders rule, because there is no host-side per-batch
mechanism to avoid reintroducing.

## The increments

Each increment lands only after its gates pass review.

0. **mg53, the host-cost attribution.**  Done 2026-08-22 (job
   15429313); findings §1.44.
1. **The back kernel.**  A new mbirtorch/triton_multiaxis.py,
   importing the API-drift shims from triton_cone the way
   triton_parallel does.  The wrapper matches the torch body's
   signature exactly, carries both no-compile declarations,
   declares _view_batch_cost, brackets every launch with the
   tensor's device, and takes the shared compile lock on first
   sight of a launch key.  Tests follow the cone battery as the
   template, plus the parallel file's geometry-variant sweep with
   elevation as the variant axis.
   Landed 2026-08-22.  The GPU gate (job 15430644, one H100) ran
   the new battery at 26 passed: value parity at both coefficient
   powers across five elevation-and-contract variants, banded
   concatenation and repeated launches bit-identical, the padding
   strides asserted, adjointness against the torch forward at
   1e-4, and the overhanging-band mask.  The multiaxis,
   adjointness, and existing-kernel batteries passed 7, 7, and 92
   beside it, and the local suite reads 681 passed.  The kernel's
   one arithmetic carve-out is the cone back's row-center
   rounding, with the tie-inertness argument in the module
   docstring.
2. **The forward kernel**, gather-vertical form.  Adds the
   kernel-pair adjointness test (the parallel file has it, cone
   does not) and the repeat-consistency test for the atomic
   scatter.
   Landed 2026-08-22.  The gather enumerates the slices reaching
   each detector row with a per-call radius bound proved in
   _multiaxis_slice_tap_radius; a brute-force coverage test is
   the independent check, and it also records that the five
   parity variants alone would not have exposed an under-sized
   window, so a thin-slice cell carries that statement.  A
   vanishing slope past the enumeration cap delegates to the
   torch body.  The GPU gate (job 15431566, one H100) passed the
   grown battery at 51, with the regressions 7, 7, and 92 beside
   it, and the local suite reads 688.  One gate was retuned
   during review: at a 512-row cell the kernel and the body round
   the row coordinate differently (the compiler fuses the
   multiply-add), which costs about num_rows times float32 eps on
   the weight, so the multi-row-chunk test gates at 1e-4 with the
   scaling stated in place while the small cells hold 1e-5.  This
   closes D7's multiaxis half in construction: the shipped
   organization is gather on the vertical axis and scatter on the
   horizontal, and the speed comparison against the torch body's
   scatter organization arrives with increment 4's measurements.
3. **Integration.**  Selection in
   MultiAxisParallelModel._view_batch_bodies, two availability
   gates and self-checks in kernel_availability.py with a
   self-check cell that carries a real elevation spread, the
   ledger consuming _view_batch_cost, the floors cost-inputs list
   gaining the new file with hashes re-blessed, and the docs
   touched where they name which geometries have kernels.
   Landed 2026-08-22.  The two-GPU gate (job 15432210) passed the
   routed battery at 60, the model-level multiaxis tests running
   through the kernels at 7, the adjointness and existing-kernel
   batteries at 7 and 92, and the multi-device battery at 15,
   which includes a two-device multiaxis reconstruction and the
   device-pin test for the launch discipline.  The ledger needed
   no edit: with kernels bound it prices the declared per-view
   costs through the generic path.  One deliberate departure from
   this increment's written scope: the floors cost hashes were
   NOT re-blessed, because six of the eight drifted inputs
   predate this campaign and a bless is all-or-nothing, so it
   would clear re-measure-owed signals nothing measured; a dated
   note in _widening_floors.py records the state, and the closing
   floors refresh (increment 6) clears it honestly.  The kernels
   select on availability alone, so their composed performance
   gate is still owed; that is increment 4, and the selection
   comment in the code says so.
4. **One-device A/B on the cluster** at the three multiaxis floors
   cells, kernel route against torch bodies, with a single-call
   re-measure in mg51's style to confirm the batch count and the
   host time collapsed.
   Landed 2026-08-22 (mg54, job 15432699; findings §1.45).  The
   kernel route runs 4.0x to 4.6x faster than the torch route at
   every cell, with identical values (fingerprints at 1.6e-7 or
   better) and the slab class gone from the memory peaks (1.96
   against 11.38 GB at the 512-class).  Read against mg52, the
   1024-class now runs 6.4x faster than mbirjax on the same
   staged input.  The driver chose 128-view batches on the kernel
   route where the torch bodies were forced to 1, and the back's
   launch-queue pressure is gone.  The kernels therefore hold
   their default-on selection under the first campaign's
   standard, and D7's multiaxis half is closed by measurement:
   the gather-vertical organization won without the scatter-mirror
   form ever needing to be built.  The tile constants remain the
   adopted cone values; a tuning sweep is a recorded follow-up.
5. **Multi-device and capacity.**  The 1024-class at one, two, and
   four devices, then the 2048-class demonstration.
   Landed 2026-08-22 (mg55, job 15434826; findings §1.46).  The
   1024-class scales on the kernel route: 1.81x at two devices
   and 2.97x at four, with the busiest peak halving at each
   doubling (24.11 to 12.95 to 7.49 GB) and fingerprints agreeing
   across counts.  The 2048-class reconstruction completed on
   four devices in 298.81 s warm at 50.59 GB busiest, and its
   two-device arm's out-of-memory brackets the boundary.  This
   delivers campaign goals 2 and 4.  The ledger priced the kernel
   route to within 7 percent at every measured point and
   correctly separates the counts that fit from the one that does
   not, so the recorded 2x follow-up is resolved: that factor was
   the torch-body pricing, which now covers only the fallback
   path.  Increment 6's remaining scope is the floors refresh:
   the multiaxis sentinels are torch-body-era values, and real
   knees now exist to record.
6. **Floors refresh scoped to multiaxis, and the ledger
   re-price.**  This is where the recorded 2x conservatism at the
   multiaxis 1024-class is corrected against measured peaks.
   Landed 2026-08-22, widened to the FULL refresh (mg56, job
   15435735; findings §1.47) because eight cost inputs had
   drifted across every family.  The multiaxis two-device floor
   holds at the 512-class; the four-device floor rises to the
   1024-class, placed by hand on mg56's thin 768-class reading
   and mg55's 1.64x at the 1024-class.  Cone's four-device floor
   falls one class; parallel and translation reproduce; the
   denoiser stays a sentinel with its top-cell ratios up.  The
   cost hashes and table checksum are re-recorded, the staleness
   note is clear, and the floors test states the new split.  The
   ledger half needed no code change: mg55's pricing rows showed
   the kernel route modeled to within 7 percent, and the 2x note
   stands only for the torch-body fallback.
7. **The translation memo.**  What the campaign cost, what
   transferred (shims, gates, test templates, launch discipline),
   the measured multiaxis wins, and the translation-specific
   unknowns, so the translation decision is priced from evidence.
   Landed 2026-08-22: active/translation_kernel_memo.md.  The
   price is one to two working days and three to six GPU-hours on
   the reuse inventory; the need test is not met today, and the
   memo names the three triggers that would re-open it.  With
   this, every increment of the campaign has landed.  The
   follow-ups that remain are recorded in place: the tile-constant
   tuning sweep (increment 4's note), the denoiser ladder
   extension (the mg56 floors notes), and D7's translation half
   (the memo).

## Gates

The gate family is the first campaign's, carried verbatim.

* Value parity against the torch body: 1e-5 relative single-shot,
  1e-4 at coeff_power 2 and iterated, 1e-3 composed pipeline.
* Adjointness at 1e-4, including the kernel-pair form.
* The zero-elevation cell matches ParallelBeamModel at 1e-5, which
  is the strongest value reference multiaxis has
  (test_multiaxis.py:71-89).
* The stored multiaxis goldens (the ma_* keys of
  golden_64x64x64.npz) at their recorded tolerances.
* Banded parity in both directions: forward bands sum, back bands
  concatenate.
* Width padding: the vector axis rounds up to a multiple of 16
  through padded_kernel_width, and the tests assert the returned
  strides.
* The per-device value self-check gates binding, and
  MBIRTORCH_DISABLE_TRITON restores the torch bodies.
* The full suite and the docs build stay clean.

## Traps carried forward

Recorded in the first campaign and the survey; each is checked in
review rather than rediscovered.

* A Triton launch targets the launching thread's current device,
  so every launch is bracketed with the tensor's device, and the
  device leads the launch key.
* Triton compiles at first launch, outside torch.compile; the
  first launch of a key borrows the process-wide compile lock.
* Both no-compile declarations are needed: the decorator alone
  does not survive an explicit torch.compile.
* Padded lanes load zeros so their weights vanish; any quantity a
  padded lane divides by loads 1.0 instead, which is the NaN
  hazard the gather-vertical inversion inherits from the cone
  forward.
* Contract builders are hoisted outside every loop; a per-chunk
  rebuild is the recorded bench artifact from the pallas era.
* The forward's atomic adds are not bit-reproducible between
  launches; its repeat gate is a tolerance, and only the back is
  bit-exact.
* Divisibility specialization is real: an integer argument the
  compiler cannot prove divisible by 16 produces a half-rate
  kernel.

## Out of scope

Translation kernels wait for the increment-7 memo.  The
sort-ordering memoization stays the recorded follow-up it already
is.  D7's translation half stays open.
