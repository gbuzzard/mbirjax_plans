# Phase 5 findings — Triton projector kernels

**Status: PHASE 5 COMPLETE (2026-08-07).**  All four kernels are
DEFAULT-ON, the replacement rule passes at every gate cell of both
geometries (cone 1.00x/1.18x of jax, parallel 1.21x/1.90x, memory
0.56–0.63x), the flipped selection contracts are battery-green on
H100 (68 passed), and the value gate closes on the shared-sinogram
ruler, converging under the 5e-3 envelope by iteration 6.  The one
red cell along the way — value vs jax at parallel-1024 — was
attributed end to end to the test protocol (phantom-generator
boundary ties, framework- and platform-dependent), not to the kernels
or the reconstruction chain; the attribution and the protocol rules
it produced are recorded below.  Follow-ups (kernel-aware view
batching; sorted streams on measured need) are chartered in
`current_plans.md` §5.  The design is `phase5_kernel_design.md`; the
delegation model was Opus implementing against it with Fable
reviewing and holding the GPU gates.

## The increments so far

K1 built the cone back kernel as a drop-in view-batch body.  The
implementing agent validated the REAL kernel source before any GPU
existed for it: a torch-backed emulation of the Triton language ops
executed the actual kernel against the torch body on CPU, reading
parity at 1.6–2.4e-7 across flat, curved, and helical variants, exact
banding, and inert padding.  On the first genuine H100 compile the full
CUDA battery passed (17/17) and the value self-check read 6.8e-07.

K2 swept the kernel constants and ran the composed gate.  The sweep
(60/60 configs, subprocess-isolated, ptxas metadata read back) found
the isolated kernel 5.9x over the COMPILED torch body at the 512 cell
(3.38 vs 19.9 ms) and 3.8x at 1024 (4.46 vs 17.1 ms), with winners
differing across cells by under 0.6 percent -- so one config,
(16, 64, 4, 1), was pinned.  The composed gate then measured the whole
seeded warm vcd:

| cell | kernel | torch body | speedup | jax | torch/jax | mem/jax |
|---|---|---|---|---|---|---|
| 512 | 4.99 s | 9.47 s | 1.90x | 3.09 s | 1.61x | 1.49x |
| 1024 | 129.1 s | 246.3 s | 1.91x | 63.2 s | 2.04x | 0.57x |

The 512 cell passes the replacement rule outright (it stood at 3.06x
of jax before the kernel).  The 1024 cell sits 2 percent over the line
with cone FORWARD now the dominant term -- the K3/K4 target.  Values
passed at 2.76e-4 and 9.71e-5 against the 5e-3 envelope, with measured
repeat-run floors of 1.7e-6 and 7.0e-6 beside them.  On the gate pass
the back kernel's default flipped to probe-plus-self-check selection
(`MBIRTORCH_DISABLE_TRITON=1` is the kill switch).

K3 built the cone forward kernel in the plain-atomics form.  The view
axis became a grid dimension rather than a register reduction, because
the forward writes a separate output plane per view and has no view
sum to keep in registers.  The emulator (upgraded to module injection
running the actual source, ruler-checked against the K1 kernel) read
parity at 2.0–5.6e-7 everywhere; the first H100 compile passed the
full battery (32/32), including the masked two-dimensional atomic-add
tile that was the top named risk.

K4's forward sweep (30/30 configs, zero value flags) picked
(8, 128, 8, 1) at BOTH cells: the isolated forward kernel is 2.43x
over the compiled torch forward body at 512 (5.74 vs 13.96 ms) and
1.88x at 1024 (7.80 vs 14.70 ms), with kernel-repeat atomic floors
under 8e-7.

The five-arm composed gate then PASSED decisively, and the cone
replacement rule now holds at every gate cell:

| cell | both kernels | back only | pure torch | jax | torch/jax | mem/jax |
|---|---|---|---|---|---|---|
| 512 | 3.09 s | 4.94 s | 9.46 s | 3.08 s | 1.00x | 0.60x |
| 1024 | 74.4 s | 128.9 s | 245.8 s | 62.9 s | 1.18x | 0.56x |

The 512 cell runs at jax PARITY with both kernels; the 1024 cell sits
at 1.18x of jax's time at 0.56x of its memory.  Values passed at
2.75e-4 and 9.74e-5 against the 5e-3 envelope (body-repeat floors
2.0e-6 and 8.3e-6), and the arm checks confirmed both triton bodies
bound with the pinned constants.  The forward's default flipped to
probe-plus-self-check selection, its constants pinned, and the opt-in
environment variable retired.  The Phase 3 headline -- the cone back
projector 3.4-6.2x behind jax across all CUDA cells -- is CLOSED.

The K5 sorted-stream variant is NOT taken for cone: the measured need
it was contingent on did not materialize (jax parity at 512, 1.18x at
1024).  The remaining Phase 5 increment is the parallel pair.

The parallel pair arrived as the degenerate case the design predicted:
the cone kernels with the vertical fan deleted, not a second design.
Deleting the fan also deleted the entire rounding carve-out — these
kernels contain no atan2-vs-sqrt divisor and no round-vs-floor tie, and
several emulator readings were BIT-EXACT against the torch bodies
(worst 2.1e-7 across 48 checks, with the K1 cone kernel run alongside
as the ruler).  One contract specialization is new: under parallel
beam the projected footprint depends on the view angle alone, so
W_p_c and weight_scale ride as per-view scalars — two floats per view
instead of two (views, pixels) planes, a few hundred MB of traffic per
call at the 1024 cell — and the reshape that converts them is also the
guard (a contract that ever became per-pixel raises rather than
broadcasting).  The full CUDA battery (both kernel test files, 68
tests) passed on the first genuine Triton compile, extending the
emulator's first-compile record to four kernels.

The parallel sweep ran 60 configurations with zero value flags.  The
back kernel is 13–14x over the compiled torch body at the 512 cell and
7.8x at 1024, with bit-exact repeat launches (no atomics on the back
path).  The register-pressure prediction held: the parallel back
program holds about three live tiles where the cone back holds six, so
the taller BLOCK_R=256 rectangle runs at 50 registers with zero spills
and wins the 1024 cell.  The back winners split by cell — BLOCK_R=64
wins 512 by 9 percent and BLOCK_R=256 wins 1024 by 24 percent — and
one config was pinned anyway: (8, 256, 4, 1) sits within 0.6 percent
of best at 1024, where the back kernel dominates the composed time,
and its 9 percent concession at 512 is invisible there (the whole back
call is under 0.8 ms).  The forward pin (8, 128, 8, 1) is the
cone-seeded config, best at 1024 outright.

The forward kernel taught the E4 lesson in reverse.  Isolated, it is
1.21x over the compiled body at 512 and LOSES at 1024 (0.78x).
Composed, the both-kernels arm beats the back-only arm by 19–24
percent at both cells.  The isolated bench measures the body's best
regime — the full pixel set per view batch — while vcd calls the
forward on pixel subsets, where the body pays transients and per-shape
recompiles the kernel does not.  Composition is measured, not
extrapolated, in both directions.

The composed five-arm gate then passed both cells with the pinned
constants (warm seeded vcd, arm checks verified):

| cell | both kernels | back only | pure torch | jax | torch/jax | mem/jax |
|---|---|---|---|---|---|---|
| 512 | 2.05 s | 2.45 s | 4.86 s | 1.70 s | 1.21x | 0.63x |
| 1024 | 51.1 s | 61.4 s | 143.6 s | 26.9 s | 1.90x | 0.63x |

The pure-torch bodies stood at 2.98x and 5.56x of jax, so the pair
buys 2.4x and 2.8x composed.  Values against the torch bodies passed
at 3.2e-3 and 2.3e-3 (floors 1.7e-6 and 1.7e-5).  On the pass both
defaults flipped to probe-plus-self-check selection, the opt-in
environment variables were retired, and the flipped selection
contracts ran green on H100 (68 passed).  The Phase 2 headline — the
parallel back projector 7.6x/4.4x behind jax — is CLOSED.

## The 1024 value flag, attributed to the ruler

The parallel gate's one red cell was value-vs-jax at 1024: 0.375
against the 5e-3 envelope, identical in the pure-body arm.  The number
survived the kernels' own checks (kernel-vs-body 2.3e-3), so the cause
lay upstream, and the attribution ran as a chain of single-variable
probes.  The gate metric is a pointwise max-rel; recomputing on the
saved samples showed the volume agreeing at 1.1e-3 norm-rel, with the
max carried by about two dozen isolated voxels.  The filter step
matched across frameworks at 2e-6 on identical input.  The partition
sequences and the partitions themselves are index-identical, refuting
the first hypothesis (a default-sequence mismatch).  A per-iteration
trace showed the divergence inherited from the init at iteration 1,
and pinning a zero init collapsed it twentyfold.  A channel-tap
rounding tie was refuted by cylinder coherence: a tie must corrupt an
entire (row, col) cylinder under parallel beam, and the divergence
sits at isolated slices.

Isolated-slice divergence with a per-slice-separable operator means
the DATA differs, and it does: the two frameworks' phantom generators
disagree at ellipsoid-boundary voxels at the 1024 recon shape, and are
bit-identical at the 512 shape, matching where the gate was clean.
This is the documented boundary-tie divergence in the generator's own
module header — f32-vs-f64 grid arithmetic at exact ellipsoid
boundaries.  The gate had each framework project its own phantom, so
the tie voxels entered the sinograms and the value comparison
inherited them.

The linkage closed in situ, and closing it surfaced a second layer of
the same lesson.  The direct-recon census diverges at exactly six
slices, all in mirror pairs about the volume center, and the
protocol's own phantoms differ at every one of them with exactly
mirror-symmetric per-slice sums (1.6/1.6, 0.8/0.8, 0.2/0.2 — phantom
amplitudes, eight orders above float noise).  A local reduction had
first contradicted this, showing no phantom difference at five of the
six slices.  The contradiction resolved into a finding: the local jax
phantom came from macOS CPU jax, the protocol's from gautschi CUDA
jax, and the jax generator's tie resolution is PLATFORM-dependent (48
differing slices on the H100, 40 on the Mac, different sets; the f64
torch phantom is exactly z-symmetric and platform-stable).  The
mirror-pair structure needs no chain z-flip: the torch phantom is
exactly z-symmetric and the recon operators preserve the symmetry, so
any one-sided difference prints at both members of a pair.

The shared-sinogram re-run is the ruler repair, measured: handing one
forward-projected sinogram artifact to both frameworks collapses the
1024 value from 0.375 to 6.1e-3 (norm-rel 8.0e-4), with the kernels
invisible (kernel arm 6.094e-3 vs body arm 6.087e-3).  The residual
at 3 iterations is the ordinary cross-framework trajectory spread of
unconverged iterates, and extending the comparison settles it: the
kernel-arm max-rel vs jax falls 6.1e-3 at iteration 3, 2.2e-3 at 6,
and 8.8e-4 at 10 (norm-rel 8.0e-4, 1.4e-4, 4.1e-5), a sevenfold
monotone convergence toward the common minimizer.  The value gate
therefore closes under the standing 5e-3 envelope with no metric
change — the iteration-3 reading was the ruler again, this time its
convergence depth.  The re-run also removed a second protocol
asymmetry found while building it: the original gate cast weights to
float32 on the torch side only.

The protocol rules going forward, both measured lessons: a
cross-framework value gate hands ONE sinogram artifact to both
frameworks — the per-framework phantom differs across frameworks at
boundary ties, and the jax phantom additionally differs across
platforms — and in-framework generation stays fine for timing arms.

## Lessons recorded along the way

The battery submission for the parallel pair added a third specimen to
the silent-wrongness collection.  A partial rsync delivered the kernel
module but not its test file; pytest refused the missing path and
collected ZERO tests; and the sbatch's ``pytest | tail`` pipe reported
tail's exit status, so the job finished COMPLETED 0:0 having tested
nothing.  The md5 spot-check had verified only one file — per-file
verification of every changed file is now the sync rule, and the sbatch
carries ``set -o pipefail``.

The measurement discipline caught two silent-wrongness classes before
they cost anything.  Slurm's ``--export=ALL,VAR='a,b,c'`` splits
exported values on commas; the harness's strict constant parse refused
the truncated value instead of sweeping garbage, and every sbatch now
documents submission-shell env passing.  And after the back kernel's
default flip, the gate's original "pure torch" arm would have silently
run with the back kernel ON; the reworked gate pins every arm's
expected bodies and verifies them at run time (the arm check), which
is the same failure class the nightly platform-mismatch guard exists
for.

The attribution's element-wise probe added a lesson about thresholds.
The probe printed a SECOND CAUSE headline for two sinogram rows, and
the flag was an artifact of its own 100x-median cut — those two rows
carry the smallest phantom difference, real signal at 28x the noise
floor but under the chosen threshold.  The instrument could
manufacture the headline finding.  What settled it was a prediction
with no free parameter: each sinogram row's max difference reproduces
its phantom slice's summed difference to within 0.45 percent at all
six rows, exactly as a parallel-beam ray must accumulate it.  The
probe now records the measured ratios beside its lowered cut, so the
choice is auditable rather than tuned.

Two ruler notes carry forward.  The forward kernel's atomics make its
launches non-bit-reproducible, so each sweep row measures its own
kernel-repeat floor and the value column reads against that, not
against zero.  And the emulator validation held: across both kernels,
every real-hardware parity number landed within the range the CPU
emulation predicted, so first-compile risk concentrated where the
agents said it would (Triton API surface, not kernel math), and none
of it materialized.
