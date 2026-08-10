# Panel review of the port (2026-08-04)

**Scope:** the mbirtorch repo as staged after Phase 2 (compile integration,
prox_map, the view-batch cap).  **Method:** five independent review lenses --
fidelity to mbirjax, standalone correctness, readability/API, performance, and
test coverage -- with every finding checked by an adversarial verifier that
read the code (and the mbirjax reference) and tried to refute it.  30 agents;
50 raw findings; the top 5 per lens were verified; 24 confirmed, 1 refuted.

**Headline reading.**  The port's numerics survived the panel untouched: no
confirmed finding challenges the projector, prior, or engine math (the golden
and parity gates stand).  The confirmed findings cluster in the SEAMS: parameter
-handling semantics that mbirjax implements and the port silently dropped
(the auto_regularize_flag side effects being the one silent-value-divergence
bug), robustness promises the code makes but cannot keep (the lazy
torch.compile fallback), input validation that mbirjax has and the port lost,
device/dtype normalization in the DL-interop wrapper, per-subset eager glue as
the next performance lever, and five named test gaps.


## Fidelity to mbirjax

**[high] set_params drops mbirjax's auto_regularize_flag side effects, silently changing recon values**  (`mbirtorch/parameter_handler.py:60`)

mbirjax's ParameterHandler.set_params (mbirjax/parameter_handler.py:519-540) has special-case handling the port omits: directly setting sigma_y/sigma_x/sigma_prox sets auto_regularize_flag=False (with a warning), and setting sharpness/snr_db re-enables a disabled flag. The port's set_params (mbirtorch/parameter_handler.py:57-69) only stores values. Consequence 1: `model.set_params(sigma_x=0.5); model.recon(sino)` uses sigma_x=0.5 in mbirjax (auto-reg disabled) but in mbirtorch auto_set_regularization_params (tomography_model.py:236) still runs and silently OVERWRITES the user's sigma_x, so the two frameworks reconstruct with different regularization from identical call sequences, with no warning. Consequence 2: after a user sets auto_regularize_flag=False, later setting sharpness/snr_db re-enables auto-reg in mbirjax (so sharpness takes effect) but is a silent no-op in mbirtorch. The port's parameter_handler docstring lists Phase-1 exclusions (YAML save/load, ParamNames typing, use_gpu shim) and does not list this, so it appears unintentional.

**[medium] set_params eagerly runs verify_valid_params, rejecting multi-step geometry changes mbirjax allows**  (`mbirtorch/parameter_handler.py:68`)

The port's set_params calls self.verify_valid_params() whenever no_warning is False (mbirtorch/parameter_handler.py:68-69). mbirjax never validates inside set_params -- neither the base (mbirjax/parameter_handler.py:474-547) nor the TomographyModel wrapper (mbirjax/tomography_model.py:2494-2508); validation is deferred to recon entry (vcd_recon calls verify_valid_params at tomography_model.py:3178). Behavior change: the standard mbirjax re-targeting flow `model.set_params(sinogram_shape=new_shape, angles=new_angles)` followed by `model.auto_set_recon_geometry()` works in mbirjax, but in mbirtorch the first call raises ValueError from ParallelBeamModel.verify_valid_params (mbirtorch/parallel_beam.py:164-168: recon_shape[2] still equals the OLD sinogram rows) before auto_set_recon_geometry can fix recon_shape. The user must pass no_warning=True, a signature deviation from mbirjax.

**[low] vcd_recon accepts prox_input without mbirjax's shape check, silently reshaping malformed input**  (`mbirtorch/tomography_model.py:863`)

mbirjax vcd_recon validates prox_input's shape against recon_shape and raises ValueError on mismatch (mbirjax/tomography_model.py:3307-3315). The port (mbirtorch/tomography_model.py:863-865) converts and reshapes prox_input to (-1, last_dim) with no check, so a wrong-shaped but size-compatible prox_input (e.g. a transposed volume) silently reshapes and yields a wrong proximal-map reconstruction where mbirjax fails loudly. Edge-handling deviation only (correct inputs behave identically).

**[low] prox_map sigma_prox override not persisted into prox_data, changing reported metadata on cached calls**  (`mbirtorch/tomography_model.py:1091`)

mbirjax prox_map mutates the stored dict in place -- `regularization_params['sigma_prox'] = sigma_prox` (mbirjax/tomography_model.py:3923) -- which also writes into self.prox_data, so a later prox_map(do_initialization=False, sigma_prox=None) REPORTS the previous call's override in recon_params['regularization_params']. The port copies instead -- `dict(regularization_params, sigma_prox=sigma_prox)` (mbirtorch/tomography_model.py:1091-1092) -- so the same later call reports the auto value. Recon values are identical in both (the model param is set and restored the same way, mbirtorch/tomography_model.py:1089-1113 vs mbirjax 3921-3944); only the reported metadata across repeated PnP calls differs. Arguably the port fixes a mbirjax reporting quirk, but it is a behavior deviation under the fidelity lens.

**[low] Truncation warning recommends scale_recon_shape, which was not ported**  (`mbirtorch/tomography_model.py:277`)

The port's _check_lateral_truncation warning text (mbirtorch/tomography_model.py:271-277) is copied verbatim from mbirjax (mbirjax/tomography_model.py:2593-2597) and advises 'Consider using scale_recon_shape(s, s)', but mbirjax's TomographyModel.scale_recon_shape (mbirjax/tomography_model.py:4002) does not exist anywhere in mbirtorch -- a user following the port's own advice gets AttributeError. Either port the method (it only rescales recon_shape) or reword the warning.

## Standalone correctness

**[medium] maybe_compile's eager fallback cannot trigger: torch.compile is lazy, so backend failures crash at first projector call**  (`mbirtorch/projectors.py:62`)

maybe_compile wraps only the torch.compile(fn) CALL in try/except, but torch.compile just returns a lazy wrapper -- actual Dynamo/Inductor compilation happens at the first invocation, inside fan_forward_batch/fan_back_batch/_parallel_hfan_math/qggmrf calls where no handler exists. The module comment promises 'A compile failure falls back to eager silently-but-recorded, so exotic backends/toolchains keep working', and compile_mode defaults to 'auto' on every backend. Concrete failure: on a machine without a working C++ toolchain (common: no Xcode CLT/clang, or a broken triton on CUDA), ParallelBeamModel(...).forward_project(x) raises BackendCompilerFailed from the first driver-loop call instead of falling back to eager; every projector call and recon() is unusable until the user discovers compile_mode='off'. _COMPILE_ERRORS stays empty forever (dead diagnostic). Fix direction: wrap the first invocation (or use torch._dynamo suppress_errors / a probe call) and rebind the cache entry to the eager fn on failure.

**[medium] Differentiable wrappers never normalize input device/dtype: CPU or float64 inputs crash, some only at backward time**  (`mbirtorch/autograd.py:63`)

forward_project_differentiable builds indices on model.torch_device and indexes the user's volume with them; sparse_* convert inputs to the model device internally, but returned gradients are model-device float32 tensors regardless of the input leaf. Concrete failures with a CUDA/MPS model: (1) volume = torch.randn(recon_shape, requires_grad=True) (CPU default) -> forward_project_differentiable raises RuntimeError at volume.reshape(-1,S)[indices] ('indices should be either on cpu or on the same device as the indexed tensor'). (2) back_project_differentiable(model, cpu_sinogram_requiring_grad): forward SUCCEEDS (sparse_back_project silently moves the data), the loss computes, and .backward() then fails with 'Function returned an invalid gradient at index 0 - expected device cpu but got cuda:0' because _BackProjectFunction.backward returns the grad on the model device. (3) A float64 volume similarly passes forward (cast to f32 inside) and fails at backward with a dtype-mismatch gradient error. The DL-interop deliverable should either move-and-validate inputs up front or cast the returned grads back to the input's device/dtype in backward.

**[medium] prox_map/vcd_recon dropped mbirjax's prox_input shape validation: mis-shaped input crashes obscurely mid-recon or runs silently**  (`mbirtorch/tomography_model.py:863`)

mbirjax vcd_recon raises ValueError('prox_input does not have the correct size...') when prox_input.shape != recon_shape; the port only does prox_input.reshape((-1, prox_input.shape[-1])) with no check. Concrete failures: recon_shape (128,128,64) with prox_input accidentally transposed to (64,128,128) reshapes to (8192,128); the first subset whose pixel_indices exceed 8191 raises IndexError inside prox_gradient_at_indices ('index out of bounds for dimension 0') -- after the expensive FBP init and Hessian computation, with no hint the input shape was wrong. Worse, shapes that keep first-dim >= rows*cols and last-dim == slices run to completion silently: e.g. a prox_input with a leading batch axis (2, rows, cols, slices) reshapes to (2*rows*cols, slices) and the recon silently uses only batch[0]. Restore the reference's explicit shape check before the reshape.

**[low] ZeroDivisionError in vcd_recon when the recon is identically zero (mbirjax nan-continues)**  (`mbirtorch/tomography_model.py:901`)

nmae_update[i] = float(ell1_for_partition) / float(recon_l1) converts both device scalars to Python floats, so recon_l1 == 0 raises ZeroDivisionError; mbirjax computes the same ratio in jnp arrays, gets nan, the nan fails the stop test, and the recon completes. Concrete failure: positivity_flag=True, init_recon=0, and a sinogram with no positive values (e.g. over-subtracted background, all entries <= 0). sigma_y is set from the RMS (positive) so initialization succeeds; every subset's update direction is negative and the positivity clip forces delta = max(0, negative) = 0, so flat_recon stays exactly 0; after the first partition, recon_l1 == 0.0 and ell1 == 0.0 -> 0.0/0.0 -> ZeroDivisionError mid-recon instead of returning the zero volume with a nan trace as the reference does. Guard the denominator (or compute the ratio in torch and let it be nan/inf).

**[low] Input sinogram shape is never validated: broadcasting silently accepts wrong-shaped sinograms; extra/missing views are silently dropped or crash obscurely**  (`mbirtorch/tomography_model.py:829`)

vcd_recon converts the sinogram with torch.as_tensor and never compares its shape to the model's sinogram_shape (only weights are shape-checked, inside compute_hessian_diagonal). Concrete failures: (1) with init_recon given, passing a broadcast-compatible wrong shape -- e.g. a single-view (1, R, C) or squeezed-channel (V, R, 1) array against a (V, R, C) model -- makes error_sinogram = sinogram - alpha*fwd broadcast to the model shape, and the recon runs to completion returning a wrong volume with no diagnostic. (2) In back_project/fbp_recon (projectors.py sparse_back_project view loop), a sinogram with FEWER views than the model yields a short slice against a full params_batch and fails with an opaque gather IndexError in fan_back_batch; one with MORE views silently ignores the extra views. A single shape check at the vcd_recon/back_project entries would convert all of these into clear errors.

## Readability and API

**[high] set_params silently drops mbirjax's manual-sigma semantics; vestigial compensation remains**  (`mbirtorch/parameter_handler.py:36`)

In mbirjax, set_params(sigma_x=...) (or sigma_y/sigma_prox) sets auto_regularize_flag=False and warns, so a manually chosen sigma survives recon(); setting sharpness/snr_db re-enables auto-regularization (mbirjax/parameter_handler.py:519-541). The port's set_params has none of this special-casing, so `model.set_params(sigma_x=0.5)` leaves auto_regularize_flag=True and the next recon() silently overwrites the user's value inside auto_set_regularization_params — same call, silently different outcome, and the docstring ('disables validity checking') gives no hint. Meanwhile the mbirjax-side compensation was faithfully ported and is now a no-op that reads as load-bearing: auto_set_sigma_y/x/prox pass auto_regularize_flag=True (tomography_model.py:314, 329, 344) and prox_map passes auto_regularize_flag=self.get_params('auto_regularize_flag') (tomography_model.py:1093-1094). Either port the special-casing or delete the vestigial kwargs and document the divergence. (Related symptom: the port's own demo and tests pass no_warning=True on every benign set_params, cargo-culted from mbirjax where it also suppressed these warnings; here it only disables validation.)

**[medium] Lateral-truncation warning recommends scale_recon_shape, which mbirtorch does not implement**  (`mbirtorch/tomography_model.py:274`)

_check_lateral_truncation warns: 'Consider using scale_recon_shape(s, s) where s >= 1.1...'. scale_recon_shape exists only in mbirjax (mbirjax/tomography_model.py:4002); grep finds no definition anywhere in mbirtorch. A user following the runtime advice gets AttributeError. Reword the warning (e.g. 'enlarge recon_shape via set_params') or port the method.

**[medium] output_sharded kept 'for API compatibility' but recon() dropped the parameter mbirjax has**  (`mbirtorch/tomography_model.py:971`)

The name reuse itself is defensible: every public site documents 'True returns the device tensor', and mbirjax already notes the sharded and unsharded forms coincide on one device. But the compatibility rationale is undercut on the flagship method: mbirjax.recon accepts output_sharded (mbirjax/tomography_model.py:3041) while mbirtorch.recon does not, so a ported script calling recon(..., output_sharded=True) raises TypeError anyway. Either add output_sharded to recon() (return the device tensor, skip .cpu().numpy() at line 1037) or state the asymmetry in the __init__.py note that defends the name.

**[medium] get_forward_model_loss treats numpy-array weights as a scalar, producing a sinogram-shaped 'loss'**  (`mbirtorch/tomography_model.py:525`)

The branch `elif not torch.is_tensor(weights) or weights.ndim == 0: avg_weight = weights` routes any non-tensor — including a full numpy weights array, plausible in this numpy-at-the-boundary API — into the scalar path. weighted_sq_sum / (avg_weight * numel) then broadcasts, returning a (views, rows, channels) tensor instead of a scalar; float() on it crashes later, or the caller silently gets nonsense. mbirjax's equivalent uses jnp.ndim(weights) == 0, which classifies numpy arrays correctly. Fix by testing np.ndim(weights) == 0, or by torch.as_tensor-converting weights first like every other public method in this file does.

## Performance

**[high] Per-subset eager glue: ~20 uncompiled launches per subset make interactive VCD dispatch-bound**  (`mbirtorch/tomography_model.py:663`)

Only the qggmrf chain and the fan bodies are compiled; everything between the projector calls in vcd_subset_updater runs as individual eager ops per subset: prior_linear/prior_quadratic reductions, the alpha chain (sub, add, div, clamp), the positivity max, alpha*delta_recon, index_add_, delta_sumsq reduction, ell1 reduction, and error_sinogram.sub_(alpha * delta_sinogram) which first allocates a full-sinogram temp for alpha*delta. Rough count per subset at the demo size (128 views, vb=64): back ~2 trips x ~8 kernels + forward same + ~20 glue kernels + qggmrf + gathers = ~60 launches; at 128-subset partitions that is ~7-8k launches per iteration, a ~50 ms/iteration floor at ~7 us/launch before any compute -- the torch analog of mbirjax's measured ~95% host-dispatch VCD overhead at interactive sizes. Added python plumbing per view-batch trip: compute_hfan_data_batched re-reads 6 params through get_params dict lookups every trip (parallel_beam.py:192-197), get_psf_radius recomputes from params every projector call, and each trip pays dynamo guard evaluation for 2-3 separate compiled regions. Fixes: compile the inter-projector scalar/update math as one or two fused regions (including a fused in-place error_sinogram update that eliminates the full-sino temp), hoist the param reads into the Projectors binding, and consider mode='reduce-overhead'/CUDA graphs for the fixed-shape fine-subset step (shapes are constant within a partition).

**[medium] Fan geometry recomputed 2-3x per subset update and materialized at a compile-region boundary**  (`mbirtorch/projectors.py:227`)

Within one vcd_subset_updater call, sparse_back_project (line 257) and sparse_forward_project (line 227) -- plus the positivity re-projection, a third pass -- each loop over all view batches calling compute_hfan_data_batched with the IDENTICAL (pixel_indices, angles): the full (V,P) geometry (n_p float32 + centers int64 + 3 per-view scalars, ~12 B/element) is recomputed and materialized 2-3x per subset, every subset, every iteration. jax also recomputes per call, but there the chain is fused inside one jitted program; here _parallel_hfan_math is a SEPARATE compiled region whose 5 outputs round-trip through memory into the fan bodies. Two fixes compose: (a) compute the subset's hfan tuple once per updater call and pass it through all 2-3 projector calls -- for every partition in the default sequence (>= 4 subsets) the cache is small (e.g. 180 views x 50k pixels x 12 B ~ 108 MB), gate it on the existing transient budget; (b) fuse the hfan math into the fan bodies (one compiled callable per direction, taking angles + scalars directly) so the (Vb,P) intermediates stay inside one inductor graph, which also removes the per-tap n.to(float32) cast and lets the per-view-only quantities (cos/sin, W_p_c, weight_scale, L_max -- functions of angle alone, precomputable once at Projectors build) fold away.

**[medium] Sinogram layout churn: full transpose copy per subset back-projection plus transposed writes and acc copies on forward**  (`mbirtorch/projectors.py:258`)

Every sparse_back_project call copies the whole (view-batched) sinogram via .permute(0,2,1).contiguous() (line 258), and every sparse_forward_project writes its result through a transposed view (line 230, sinogram[...] = block.permute(0,2,1) -- strided uncoalesced writes) after first zeroing and then copying out a separate acc buffer (line 117). The engine touches error/delta sinograms only elementwise outside the projectors (weights multiply, lin/quad reductions, sub_, stats), so keeping vcd_recon's error_sinogram, weights, and delta sinograms in channel-major (V, C, R) layout -- converting once at the numpy boundary -- eliminates: the per-subset full-sinogram transpose copy on the back path, the transposed write on the forward path, and (by letting fan_forward index_add_ directly into a preallocated flat (V*C, S) output with idx offset v0*C) the per-batch acc zeros + copy. That is roughly 4 full-sinogram-sized memory passes per subset today; at 128-subset partitions and a 3 GB sinogram (720x1024x1024) about 1.5 TB of pure copy traffic per iteration.

**[medium] Differentiable projector path recomputes ROR indices on host and re-uploads them every call**  (`mbirtorch/autograd.py:61`)

forward_project_differentiable and back_project_differentiable call gen_full_indices on EVERY invocation: a numpy meshgrid ellipse mask over the full (rows, cols) grid, boolean select, np.sort, then torch.as_tensor host-to-device upload of the int64 index array. For TorchProjector in a learned-prior training loop this is per-training-step host work plus an H2D transfer (~ms-scale at 512^2, ~1.6 MB upload) that serializes ahead of the projector. The result is deterministic (the num_subsets==1 partition path consumes no RNG, by design) and depends only on (recon_shape, use_ror_mask), so cache the device index tensor on the model (or in TorchProjector.__init__). The same per-call recompute sits in tomography_model.forward_project/back_project (lines 140-141 and 163-164), hit ~3x per recon (init projection, FBP back-projection); the one cache fixes both.

**[medium] fan_back einsum contraction lowers to a degenerate batched gemm with a permuted operand**  (`mbirtorch/projectors.py:158`)

torch.einsum("vp,vpr->pr", A, gathered) treats p as a batch dim and lowers to bmm((P,1,Vb) @ (P,Vb,R)): the (Vb,P,R) gathered tensor is permuted to (P,Vb,R) -- either materializing an extra full-transient contiguous copy per tap or running a poorly-coalesced strided-batch gemm (batch stride R, ld P*R) -- and the m=1 gemm shape is a worst case for cuBLAS. This survives torch.compile because matmuls stay extern in inductor. Two cheap alternatives worth a single-variable ablation: (a) gather directly in (P,Vb,R) order by transposing the index operands (sino_batch_T[arange(vb)[None,:], n.T] -- same gather cost, output already bmm-friendly and contiguous), or (b) replace the einsum with (A.unsqueeze(-1) * gathered).sum(dim=0), which inductor fuses into one multiply-reduce kernel with no permute and no extern gemm. Independent of (and complementary to) the planned Triton back-projector work.

## Test coverage

**[high] nmae parity trace is computed but never asserted**  (`tests/test_vs_goldens.py:121`)

test_recon_convergence_parity computes nmae_rel (line 121) and prints it, but the assert block (lines 125-127) only gates alpha_rel, fm_rel, and final_rel. The NMAE-per-iteration trace -- the quantity that drives the stop_threshold_change_pct stopping rule -- has no gate, so a drift in ell1_for_partition/recon_l1 accounting would pass silently. Suggested test: add `assert nmae_rel < 1e-2` (the printed floor from a golden run can tighten this later; nmae is a ratio of small differences, so match the alpha tolerance rather than fm's 1e-3).

**[high] Positivity path completely untested**  (`mbirtorch/tomography_model.py:714`)

positivity_flag defaults False and no test sets it, so the entire branch (clip delta to keep recon + alpha*delta >= 0, then re-project delta_sinogram) never executes. Note the guard is `if positivity_flag is True:` -- inherited verbatim from mbirjax (its line 3712), so `positivity_flag=1` silently disables the constraint; a test pinning True-vs-False behavior documents this. Suggested test: box-phantom sinogram (as in test_recon_smoke), np.random.seed(0), run recon twice with init_recon=0 (also covering the untested int-init path), max_iterations=4, stop_threshold_change_pct=0.0: once with set_params(positivity_flag=True), once False. Assert the unconstrained recon has min() < 0 (ringing negatives make the check non-vacuous) and the constrained recon has min() >= -1e-6, and that the two volumes differ.

**[high] Restart contract (first_iteration + init_recon) untested**  (`mbirtorch/tomography_model.py:948`)

The documented restart flow (recon docstring lines 978-981) and the RNG machinery built specifically for it (vcd_utils.py lines 117-121: single-subset partitions skip np.random so restarts reproduce partition draws) have zero coverage; first_iteration is only ever 0. Suggested test: seed np.random(0), run a continuous recon (max_iterations=4, stop_threshold_change_pct=0.0) recording the final volume and recon_params; then seed(0), run 2 iterations, seed(0) again, and run recon(sinogram, init_recon=first_half_result, max_iterations=4, first_iteration=2). Assert: num_iterations == 2 on the restart leg; the restart's partition_sequence equals the continuous run's tail (granularity continuation); fm_rmse continues decreasing across the handoff; and the restarted final volume is close to the continuous final (loose gate, e.g. relative L2 difference < 0.05 -- subset ORDER permutations differ between the runs, so exact equality is not the contract).

**[high] compile on/off value equality untested; silent eager fallback is invisible to the suite**  (`mbirtorch/projectors.py:57`)

compile_mode='auto' is the default, so every existing test exercises only whichever path torch.compile lands on -- and maybe_compile's failure handling (lines 64-67) falls back to eager while merely recording the error in _COMPILE_ERRORS, so a broken compile toolchain would leave the whole suite green while testing pure eager. The eager path itself (compile_mode='off') is never run. Suggested test: build two ParallelBeamModel instances on CPU with identical shapes/angles, compile_mode='off' and 'auto'; compare sparse_forward_project and sparse_back_project outputs on seeded random inputs (allclose rtol=1e-5, atol=1e-6 -- compile may reorder float ops) and one 2-iteration seeded recon (same np.random.seed before each). Then assert `mbirtorch.projectors._COMPILE_ERRORS == {}` after the compiled run, turning the silent fallback into a visible failure on platforms where compile is expected to work.

**[high] weights=None vs explicit all-ones weights equivalence untested; the None path has no value oracle**  (`mbirtorch/tomography_model.py:630`)

The const_weights fast path (skip the weights multiply at line 676, ones-sinogram Hessian at line 874, avg_weight shortcut in get_forward_model_loss) and the array-weights path are separate branches, but only the array path is value-anchored (convergence-parity golden uses transmission_root weights); weights=None is covered solely by the loose recon smoke. Since x*1.0 is an IEEE identity and the summation orders match, the two branches should agree essentially exactly. Suggested test: seeded recon (np.random.seed(0) before each call) on the box-phantom sinogram with weights=None vs weights=np.ones(sino_shape, np.float32), max_iterations=3, stop_threshold_change_pct=0.0, on CPU; assert final volumes and fm_rmse traces agree to max abs diff <= 1e-7 (empirically this should be bitwise). This pins the fast path to the golden-anchored path for free.

## Refuted

One finding was refuted by its verifier:

- readability: recon_dict key 'stop_threshold_change_pct' holds a different quantity than in mbirjax -- The reviewer misread the mbirjax reference. mbirjax's recon() rebinds the local variable at tomography_model.py:3095 — s

## Fix status (applied 2026-08-04, same day)

Greg approved the proposed fix order; the batch below is applied and staged in
mbirtorch, with the suite at 33 passed / 1 skipped and the golden parity
floors unchanged (final recon 1.13e-05).

**Fixed, with regression tests.**  The set_params semantics (manual sigmas
disable auto-regularization with a warning; sharpness/snr_db re-enable it;
unknown names raise; validation deferred to recon entry, so multi-step
geometry changes work).  The sinogram and prox_input shape validations.  The
numpy-weights loss bug.  The zero-recon nmae guard (nan-continue, matching
mbirjax; note the zero-SINOGRAM case drives sigma_y to 0 and raises in
mbirjax too, so it stays out of scope).  scale_recon_shape is ported, making
the truncation warning's advice actionable.  recon() regains output_sharded.
maybe_compile now guards the FIRST compiled call and falls back to eager on a
backend failure (the lazy-compile finding); _COMPILE_ERRORS records it.  The
differentiable wrappers normalize device and dtype through a differentiable
``.to`` (gradients return on the caller's device/dtype) and read the ROR
indices from a per-model cache.  The five test gaps are closed (nmae
asserted; positivity, restart, compile-on/off equality, and weights=None
equivalence all tested).

**Fixed, performance.**  The per-subset eager glue is compiled: the diagonal
direction, the prior line terms, the forward linear/quadratic reductions
(with the weights product fused, removing the per-subset weighted-sinogram
transient), and the in-place state application.  Measured effect at the
interactive cell (128^3, 10 iterations, MPS, warm): 2.07 s -> 1.82 s, for
1.96x total over eager.  Parity floors did not move.

**Kept as-is.**  The prox_map sigma_prox metadata behavior: the port reports
the current call's values rather than mbirjax's stale-override quirk; the
deviation is deliberate and documented here.

**Deferred to the planned perf work.**  Geometry recompute hoisting across a
subset's projector calls (a driver API change), the sinogram transpose churn,
and the einsum-to-bmm form -- all Phase 5 kernel/layout territory -- plus the
513-cell odd-size inductor specialization investigation.
