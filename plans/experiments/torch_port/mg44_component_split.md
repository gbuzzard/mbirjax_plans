# mg44 — which component carries the two-device loss

## Purpose

The multiaxis geometry reconstructs slower on two devices at some problem sizes
and faster at others. A wall per arm says two devices lost. It does not say what
lost. This run times every component call of the reconstruction at one device
and at two, in fresh processes, and reports which component's warm time carries
the difference and how that changes with problem size.

## The walls it is measured against

Warm walls measured 2026-08-18 by the floors refresh (mg26, job 15342578) and
read on 2026-08-19 from
`/scratch/gautschi/buzzard/torch_p3/results/mg26_floors/_out_arm_*.json` on the
gautschi cluster. The ratio is warm(n1) over warm(n2), so above 1.00 means two
devices are faster.

| cell | n1 cold | n1 warm | n2 cold | n2 warm | ratio |
| --- | --- | --- | --- | --- | --- |
| multiaxis (512, 448, 384) | 20.4 s | 11.4 s | 48.2 s | 32.6 s | 0.35x |
| multiaxis (768, 672, 576) | 64.4 s | 56.3 s | 53.9 s | 38.5 s | 1.46x |
| multiaxis (1024, 1008, 992) | 323.8 s | 309.9 s | 402.8 s | 388.8 s | 0.80x |
| translation (256, 1900, 3000) | 24.5 s | 12.6 s | 31.1 s | 14.2 s | 0.89x |

## The arms

Ten arms, each in a fresh subprocess, cheap first so a harness defect surfaces
in minutes.

1. `ma512_n1`, `ma512_n2` — multiaxis (512, 448, 384), wrapped
2. `ma512_n1_control`, `ma512_n2_control` — the same, with no wrappers
3. `tr_n1`, `tr_n2` — translation (256, 1900, 3000), wrapped
4. `ma768_n1`, `ma768_n2` — multiaxis (768, 672, 576), wrapped
5. `ma1024_n1`, `ma1024_n2` — multiaxis (1024, 1008, 992), wrapped

The two control arms are the instrument check. They run the identical protocol
with nothing wrapped, so their warm medians beside the wrapped arms' say what
the probe itself costs.

The protocol is the floors refresh's, copied rather than approximated: the same
model construction, the seed reset to 13 immediately before every call, the same
weights, the same three-iteration call, the same timing envelope including the
per-device synchronize and the gather, one cold pass then three warm repeats.
The anomaly was measured by that tool, and a different protocol would measure a
different thing.

## The instrument

Around each named library function the probe records two things per call: the
host clock before and after, and a pair of CUDA timing events on each relevant
device's default stream. Both are needed because they answer different
questions. The subset loop has no host synchronization at any device count, so a
host timer around a region says how long that region took to enqueue, not how
long the device spent on it. The events say the second thing. A region whose
host time is large beside its device time is itself a finding: the cost there is
dispatch, a thread handoff or a compile guard rather than a kernel.

Events are used rather than a synchronize because a synchronize would change
what overlaps, and the overlap is what this run is trying to see. All library
compute lands on each device's default stream, so a pair of markers on that
stream brackets exactly the work enqueued between them. The markers are recorded
during the reconstruction and read only after it returns, on the synchronize the
timing protocol already does.

The regions are the reconstruction's phases, the two projection funnels, the
cross-device primitives, the per-call projector bodies, and one seam on the
per-device fan-out that labels each call by the worker it was handed — which is
what separates the prior, the update direction, the line search, the update
application and the statistics without touching any closure. Nesting is real
(bodies inside funnels inside iterations), so nothing is summed across levels:
the report groups regions into phases, in-iteration components and inner detail,
and prints a residual line for whatever an iteration spends outside its named
components.

The run edits no library file. The only runtime change is the wrappers, each of
which reads two clocks, calls through, and returns its original's result
unchanged. The exit code reports instrument health only. Every measured number
is a finding and none of them gates.

## Results

Run 2026-08-19, job 15391547 on h004, two H100s, 66 minutes, exit 0.
The rows are `rows/mg44_component_h004_20260819_210951.jsonl`; the
finding is multigpu_findings.md section 1.36. The full report is in the
job log (`mg44_15391547.log`, overwritten only by a rerun of the same
name).

Every arm reproduced its recorded warm wall at 1.00x (worst 0.99x), and
the wrapped and control arms agree within 0.4 percent, so the
instrument cost nothing visible.

The back projection carries the loss at every losing cell. Per warm
reconstruction, on the busiest device, in-iteration:

| cell | back n1 | back n2 | forward n1 | forward n2 |
| --- | --- | --- | --- | --- |
| multiaxis (512, 448, 384) | 3.0 s | 16.5 s | 4.1 s | 2.1 s |
| multiaxis (768, 672, 576) | 15.5 s | 13.4 s | 21.0 s | 10.6 s |
| multiaxis (1024, 1008, 992) | 92.1 s | 250.1 s | 126.2 s | 63.3 s |
| translation (256, 1900, 3000) | 0.9 s | 4.9 s | 2.5 s | 1.3 s |

The forward halves at two devices at every cell. Every other component
sits at or under a third of a second per reconstruction.

The mechanism: torch.compile's recompile limit (8) attaches to the
body's code object, which the per-device compiled instances share, and
the compiled variants guard on the device index. Where the budget
fills, later unmatched calls run eagerly. The job log carries torch's
own recompile-limit warning for the back body at exactly the losing
two-device arms (multiaxis 512 and 1024, translation production,
control arm included) and at no winning or one-device arm. Warm calls
compile nothing anywhere, so the warm loss is the eager end-state.

Two notes on the row fields. The `compiled_functions_delta` differs
between wrapped (65) and control (38) arms while `unique_graphs`,
`calls_captured`, and the walls are identical; that delta is an
artifact of how `torch._dynamo.utils.compile_times` counts entries, so
read `unique_graphs` and the per-call deltas instead. The host-bound
table's half-of-device threshold also lists many small benign regions;
the load-bearing entries are the back body rows.
