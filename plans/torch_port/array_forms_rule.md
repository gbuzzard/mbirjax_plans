# Array-forms rule for mbirtorch user-facing functions

Decided by Charlie 2026-08-09 (option A); emailed to Greg for the
project instructions.  Adopted 2026-08-09 (Greg), with three
amendments from Fable's review marked below.  First applied in the
segment_plastic_metal sharding port (checklist item A1).

**A user-facing function that transforms a volume or sinogram accepts
four input forms and returns its results in exactly the form it was
given.**

| Input                     | Output                    |
|---------------------------|---------------------------|
| numpy array               | numpy array               |
| torch tensor              | torch tensor              |
| Shards with one shard     | Shards with one shard     |
| Shards with several shards| Shards with several shards|

- The one-shard case computes through the plain-tensor path and is
  rewrapped on return (free; same memory).
- The several-shard case processes each shard on its own device and
  never gathers the volume.
- Scalars (thresholds, scale factors) return as ordinary Python
  numbers in every case.
- A function whose result is not array-shaped (an exporter) owes only
  the acceptance half of the contract.
- (Amendment 1) A function that does not yet implement the sharded
  case says so in its docstring AND raises `NotImplementedError` at
  entry on multi-shard input.  A docstring alone leaves the downstream
  failure in place, which is what this rule exists to prevent.  The
  translation branch set the raise-at-entry precedent.
- (Amendment 2) Scope: this rule governs transformers — preprocess,
  utility, and denoiser functions whose output is array-shaped like
  their input.  Producers (`recon`, `prox_map`, `direct_recon`, the
  phantom generators) keep the mbirjax convention instead: a host
  default with an `output_sharded` kwarg.  A producer has no input
  form to mirror, and mbirjax's geometry modules already carry the
  kwarg.
- (Amendment 3) Multi-input functions: the primary volume or sinogram
  argument sets the output form.  Secondary array arguments are
  accepted in any of the four forms and coerced internally.

Rejected alternative: returning a bare tensor for a one-shard input
(the engine's internal convention at `_as_device_form`).  Rejected for
public functions because "you get back what you put in" is the
simplest contract and never surprises a caller.
