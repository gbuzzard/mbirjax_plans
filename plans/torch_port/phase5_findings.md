# Phase 5 findings — Triton projector kernels

**Status:** K1–K3 complete and green on H100; K2's composed gate PASSED
and the cone BACK kernel is DEFAULT-ON; the K4 forward sweep is
complete and its five-arm composed gate is in flight (job 14914632).
The design is `phase5_kernel_design.md`; the delegation model is Opus
implementing against it with Fable reviewing and holding the GPU gates.

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

## Lessons recorded along the way

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

Two ruler notes carry forward.  The forward kernel's atomics make its
launches non-bit-reproducible, so each sweep row measures its own
kernel-repeat floor and the value column reads against that, not
against zero.  And the emulator validation held: across both kernels,
every real-hardware parity number landed within the range the CPU
emulation predicted, so first-compile risk concentrated where the
agents said it would (Triton API surface, not kernel math), and none
of it materialized.
