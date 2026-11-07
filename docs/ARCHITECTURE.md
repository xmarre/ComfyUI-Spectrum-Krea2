# Native Krea 2 Spectrum architecture

## Audited source

| Repository | Revision | Relevant evidence |
| --- | --- | --- |
| Comfy-Org/ComfyUI | `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe` | `comfy/ldm/krea2/model.py`, `model_base.py`, `model_detection.py`, `supported_models.py`, `text_encoders/krea2.py`, `samplers.py`, `model_patcher.py`, `patcher_extension.py` |
| xmarre/ComfyUI-Spectrum-MiniMax-H3 | `806fe498d574a726fecb8818917f00a4b993fd77` | `forecast.py`, `runtime.py`, `sampling.py`, `minimax_h3.py`, `config.py`, `nodes.py`, entrypoint, tests and packaging |
| hanjq17/Spectrum | `4e41f91c93ff66b045332530dd7bcb4c60eafbab` | `src/utils/basis_utils.py`, license, Chebyshev regression and scheduling |
| krea-ai/krea-2 | `db3984fbc6e13b34c0064990fc2d95ac64d00058` | `mmdit.py`, `sampling.py`, inference examples and Apache license |

Spectrum is Han et al.'s spectral forecasting algorithm, arXiv:2603.01623v1. This integration implements causal Chebyshev ridge forecasting of the post-transformer hidden image features. It adapts H3's history-weight formulation, bounded snapshots, invocation ownership and native fallback principles. H3 model-aware corrections, audio/video streams, offline smoothing and special solver policies do not apply here.

## Forward path and interception

The native path is `KSAMPLER → CFGGuider/PREDICT_NOISE → calc_cond_batch → BaseModel.apply_model → SingleStreamDiT.forward → _forward`. BaseModel converts the latent and sigma to the native input/timestep; Spectrum does not replace these transformations or the sampler update.

Native Krea 2 unpacks `(B, sequence, 12*2560)` Qwen3-VL conditioning, patchifies the image with circular padding, projects image patches and builds timestep embeddings. `txtfusion` has two layerwise blocks, a layer-stack projector and two refiner blocks; it consumes the twelve tapped encoder states. Text and image tokens then share all main transformer blocks. `LastLayer` performs RMS normalization, current-timestep scale/shift modulation and output projection. Native output reconstruction selects generated-image rows, unpatchifies and crops padding.

On an actual call the original `_forward` executes unchanged. A scoped LastLayer pre-hook snapshots **only** `combined[:, text_length:text_length + image_token_count]`. The hook never returns replacement inputs. The snapshot owns contiguous storage and retains neither text/reference rows nor a view of the larger joint sequence. Actual native output is returned verbatim.

On a forecast call the wrapper predicts those image rows and skips input projection, text fusion, positional work and transformer blocks. It recomputes `tmlp(timestep_embedding(current_timesteps))`, calls the loaded native `last`, unpatchifies and crops. LastLayer has no token interaction, so text/reference rows need not be projected. Token selection before versus after projection is checked against actual native output, including BF16 tolerances for GEMM shape differences. The sampler still receives the native output shape/dtype and performs its own unchanged update.

## References and layout

Native reference latents are patchified separately and appended after generated-image tokens. Their position index increases with reference number. In `index_timestep_zero`, native blocks use zero-timestep modulation for reference rows; text/generated rows retain current-time modulation. Actual calls preserve this whole path. Forecasting predicts generated-image hidden rows resulting from that interaction; the output head uses only their current-time embedding, exactly as native output selection does. Reference list layout, method and conditioning UUID identity are part of the stream identity.

Only native `index` and `index_timestep_zero` are explicitly supported for active references. A null native method ignores references just as native code does. Other methods fall back. This is an execution contract, not a claim that base checkpoints were trained for high-quality editing. The supplied reference pair must be judged against its native baseline.

4D inputs and 5D inputs with one temporal slice are forecast-enabled. Output channel ordering, odd spatial padding and cropping follow the inspected native implementation. Multiple temporal slices use native execution because this image model's context/batch folding is not a verified video contract.

## Sampling ownership

The MODEL clone installs keyed `SAMPLER_SAMPLE`, `PREDICT_NOISE` and `DIFFUSION_MODEL` wrappers. The original clone and any existing wrappers are preserved. Reapplying replaces only this package's keyed wrappers. No transformer copy or permanent forward monkeypatch is created.

Every `SAMPLER_SAMPLE` call allocates a fresh Session in a ContextVar and destroys its histories in `finally`. A nested sampler receives its own session and restores its parent's context. User callbacks run with the trajectory context cleared, so callback model calls cannot populate or forecast from the enclosing trajectory. Independent threads receive independent contexts; unbound worker calls execute natively. Context is never inferred from latent shape or a decreasing sigma alone.

Native Euler without churn and native DPM++ 2M each have one guider prediction followed by one outer-step callback. Audited version-neutral canonical AST digests check the solver functions. The native KSAMPLER container is checked by exact native class and bound entrypoint; its entire implementation is not hash-gated because callback logging differs across compatible releases. Parameter names, optional trailing parameters and decorator signatures do not affect admission: ComfyUI calls the container positionally. Signature introspection is diagnostic only, and unavailable signature metadata cannot disable acceleration. Both the current container and the earlier upstream container at `135abed8da169e33ab0b86550e05e3ae55d6df8c` are exercised with the current native Krea 2 model. That older whole ComfyUI revision itself predates native Krea 2 support. The `PREDICT_NOISE` wrapper marks each opportunity; the callback validates ordering and advances the outer step. Multiple conditioning branch calls can occur inside one opportunity. Repeated branch evaluation uses actual execution and invalidates its history. Repeated guider opportunities disable forecasting for the invocation. Other solvers remain native.

The complete received sigma schedule is copied to CPU once. It must be finite, nonnegative and strictly decreasing. Coordinate `u_i = 2*(sigma_0-sigma_i)/(sigma_0-sigma_end)-1` spans the received trajectory in sigma space; it is not a call count or assumed uniform timestep. Truncated schedules start a fresh coordinate domain and fresh warmup. Duplicate sigmas or reverse schedules are passed to native execution without being interpreted as distinct forecast steps.

Each history key includes the ordered processed-condition UUIDs, cond/uncond ordering, latent/context shape/dtype/device, and reference layout/method. Native ComfyUI creates UUIDs for processed immutable conditions. The run owns the actual model and active weight-patch UUID. Different conditioning UUIDs, batch packing, reference layout, precision or spatial layout cannot reuse a stream. Stream count and total retained bytes are bounded. Standard weight LoRAs execute through native modules; scheduled hooks, regional conditions, latent inpaint masks, controls, unknown transformer wrappers/custom CFG and multi-device dispatch use native fallback.

The capture pre-hook is installed only around an actual native call and removed even if that call raises. A second ContextVar marks capture ownership so unrelated nested calls cannot populate the history. Concurrent final-layer hooks conservatively select native bypass. No global model cache is used. Counters cover completed diffusion wrapper calls within the bound context; arbitrary external worker-thread calls are native and are outside that accounting scope. Cancellation and native exceptions propagate; forecast-only failures retry with one native call. A horizon or fit-precondition rejection retains valid actual anchors so the next observation can expand the fit span; nonfinite predictions or runtime failures invalidate cached history.

## Numerical and memory contract

For actual coordinates `u_j`, design matrix `D[j,m]=T_m(u_j)` uses `T_0=1`, `T_1=u`, `T_m=2u*T_(m-1)-T_(m-2)`. Ridge fits `(D^T D + lambda I) C = D^T H`. Instead of retaining full feature coefficients, prediction evaluates small history weights `w = phi(u) (D^T D + lambda I)^(-1) D^T`, then accumulates `sum_j w_j H_j` in bounded FP32 chunks. The small CPU system uses FP64 Cholesky with a bounded jitter ladder. Regression matrices are invalidated when actual history changes.

Forecasted features never enter actual history. Predictions require at least degree+1 actual anchors, strictly increasing coordinates, a target inside the full normalized domain, a nondegenerate retained history span, and finite bounded history weights (`L1 <= 8`). The additional extrapolation-distance guard defaults to `gap <= 0.5 * history_span` and is independently configurable using `extrapolation_guard_enabled` and `max_extrapolation_span_ratio` (0.05–8.0). Disabling it does not disable the weight bound, domain or nonfinite checks. A ratio other than 0.5 is an unvalidated quality tradeoff, not proof of accurate long-range feature forecasting. Predictions and projected outputs are checked for finite values before entering native sampler state, requiring two explicit device synchronizations per forecast. Safety checking has real overhead; benchmark it.

Retained snapshots stay in model dtype on VRAM or CPU. The configured byte budget is shared across streams and oldest entries from the largest stream are evicted before retention. A feature too large to retain the required anchors is not cached. Temporaries include one captured feature, predicted feature/output, model activations, finite-check tensors and <=4 MiB FP32 accumulation chunks plus source-transfer chunks. Peak allocator memory can exceed the retained-history budget. No RNG calls are introduced by capture, fitting, scheduling or forecasting.

## Boundaries

The native model class, sampler entrypoint and relevant model/solver AST contracts are checked at runtime. Known backend attention and weight-operation dispatch are retained on actual calls. Unknown forward replacements, hooks or execution contracts bypass acceleration. The compatibility checks cannot establish behavior for arbitrary third-party code mutating model internals during execution. GPU checkpoints, FP8/GGUF loaders, LoRA images, reference semantics and each attention backend need separate empirical validation.

### Additional native one-call deterministic solvers

The audited solver gate allows native `sample_res_multistep` (and its exact `res_multistep` delegate with `eta=0, cfg_pp=False`) and `sample_gradient_estimation` (with `cfg_pp=False`, default `ge_gamma=2`). Both perform exactly one denoiser call and one callback per input sigma step and consume only the denoised prediction for their state/history. Spectrum may replace that prediction with a feature forecast, preserving native solver integration and state update order. Source AST fingerprints pin each function; mutable `extra_options` are refused for these solvers. This is an empirical feature approximation, **not** mathematical equivalence to full native model evaluation.

### DEIS and iPNDM family (2026-11-06)

DEIS (`sample_deis`), iPNDM (`sample_ipndm`) and iPNDM-V (`sample_ipndm_v`) have one denoiser evaluation and one callback per native sigma step. Their finite multistep derivative histories consume the forecasted prediction through the original ComfyUI update equations; no solver reimplementation or bypass of previous-state bookkeeping is necessary. Their native function AST contracts are pinned. For DEIS, `get_deis_coeff_list`, `edm2t`, `cal_poly`, `cal_intergrand` and `t2alpha_fn` are also source-audited, so a modified coefficient implementation disables Spectrum. Default native options are required (DEIS max_order=3, deis_mode='tab'; iPNDM family max_order=4); custom sampler options fall back. Spectrum's history starts anew for each Scale-Locked invocation and never reuses features across resolution changes. The feature extrapolation is a model-output approximation, not mathematical equivalence to full native DiT evaluation.


### Native ER-SDE forecast qualification

ComfyUI's `sample_er_sde` performs one denoiser evaluation and one callback per native outer step. It then integrates a first-, second-, or third-order VP reverse-time SDE update from the denoised prediction, and optionally adds an independent Gaussian stochastic increment. Spectrum may forecast the denoised prediction at eligible steps; it never substitutes an Euler update, changes the native solver state, or consumes the stochastic noise stream itself. Independent per-run forecast histories are retained and cleared at solver invocation boundaries, including Scale-Locked's restarted resolution stages.

The native solver corrects an initial flow sigma of 1.0 to a finite-log-SNR coordinate. Spectrum computes its forecast coordinates using the same native `offset_first_sigma_for_snr` helper. Its source fingerprint is verified, as are `sample_er_sde` and `sigma_to_half_log_snr`; any modified source fails closed to ordinary native sampling. The only recognized sampler overrides are a finite nonnegative `s_noise`, a `max_stage` in 1–3, and callable `noise_scaler`/`noise_sampler`. Scale-Locked supplies patch-phase Gaussian-coupled per-step increments across resolutions, keyed by the absolute step and seed. These stochastic inputs do not enter Spectrum feature history. Native and no-forecast identity, repeatability, source mutation rejection, and CPU native Krea integration are necessary structural checks, not validation of stochastic visual quality or GPU acceleration.


### Qualified KJNodes preview-only wrapper

KJNodes' `_PreviewOverrideWrapper` is an `OUTER_SAMPLE` wrapper that decorates the progress callback. The audited source calls the underlying sampler once, forwards the original callback once per outer step, and restores temporary previewer method replacements in `finally`. It does not substitute a denoiser output, sigma schedule, model patch, or stochastic noise field. Consequently, it can coexist with Spectrum's `SAMPLER_SAMPLE`/ `PREDICT_NOISE` hooks and Sol-Krea2's distinct `OUTER_SAMPLE` sampling scope.

A callable registered exactly once under `outer_sample.kj_preview_override` is treated as an **outer-sampling observer**, not as a numerical modifier. No KJNodes class name, module source path, version, whole-file hash or callback AST is used in production admission. This boundary is intentional: Spectrum's SAMPLER_SAMPLE wrapper runs *inside* outer-sampling wrappers and therefore validates the **effective** ComfyUI KSAMPLER and source-audited native solver, selected options and sigma coordinates. Its inner denoiser wrapper separately validates model/weight identity, supported patches and conditioning, and each outer step's denoiser opportunity and callback sequence. A preview wrapper's implementation is irrelevant to those numerical contracts unless it changes the actual sampler/model behavior, which remains subject to the same checks and native fallback. Admission snapshots the specific registered KJ instance, its callable and its code-object identity for the active invocation, rejecting replacements or mutation while sampling. Unregistered transformer/diffusion wrappers and unknown outer-sample keys remain blocked. Future KJNodes implementations using the same registration enter the same effective-contract checks without source requalification. Historical upstream KJ source revisions are pinned **only for reproducible integration tests**, not for runtime compatibility. Structural tests cannot guarantee that every future external wrapper behavior is numerically benign; native solver validation and active runtime checks define the supported boundary.


### Native SA-Solver and PECE evaluation accounting

ComfyUI's SA-Solver uses stochastic Adams coefficient helpers and native
half-log-SNR scheduling, with an initial flow-sigma correction shared with
ER-SDE. Spectrum verifies the original native predictor, PECE entrypoint,
coefficient family and tau-window helper before forecasting, and accepts
validated sampler orders, stochastic noise and sampling window options.

Standard \`sa_solver\` evaluates the model once per outer step. PECE mode
(\`sa_solver_pece\` or the native \`sa_solver\` with \`use_pece=True\`) first
evaluates its predictor and **fires the outer-step callback**, then evaluates
the corrected state at the same sigma on eligible steps. Treating that second
call as a new outer step would corrupt forecast coordinates and PECE history.
The Spectrum sampler wrapper tracks a pending correction after the callback
and requires its sigma to match the prior step. It routes the correction
through the native model without any forecast, anchors or cached history
substitution. The next predictor resumes using the next sigma and existing
causal predictor history. Unexpected correction ordering disables forecasting.
The report records the additional PECE correction opportunity separately
from outer-step callbacks; native correction evaluations are still included
in the actual model-evaluation count.

#### Forecast isolation for SA-Solver

Inside the SDE window, SA adds a fresh Gaussian increment to its predictor
state before each evaluation. An actual Krea 2 call sees that increment and
removes it from `denoised = x - sigma * v`. A feature forecast is extrapolated
from earlier steps and cannot see it, so its denoised value keeps the increment.
SA would then carry it to the output as per-latent-cell colored speckles.

Forecast-enabled SA and PECE therefore run `sa_isolation.sample_sa_solver_isolated`.
It uses the native SA equations, coefficient helpers and noise sampler, with
two ownership rules:

* Only actual denoiser results and exact PECE corrections are persistent
  Adams history. A forecast is an ephemeral input to the current corrector
  and predictor only.
* On a forecast step whose input carries a fresh increment, SA consumes a
  causal value built only from actual anchors instead of the raw forecast.
  That value is the latest anchor after consecutive anchors. Otherwise it is
  a half-log-SNR extrapolation of at most one anchor interval, and it falls
  back to the latest anchor if the guard fails.

Without forecasts the adapter is the native solver bit for bit. The report
counts `sa_ephemeral_forecasts` and `sa_dense_outputs`. If another sampler
wrapper follows Spectrum's, forecasting falls back to native. The design
follows the SA and ER-SDE stochastic forecast isolation in Spectrum MiniMax H3.

#### Opt-in active-PECE acceleration

With `sa_pece_forecast_policy` set to `stable_start`, `balanced` or
`max_speed`, active PECE runs its predicted phase with the model only for the
first 3, 2 or 1 outer steps. After that the predicted phase is owned by the
solver: Spectrum returns a placeholder without running the DiT. SA ignores
that placeholder, and its corrector consumes a causal extrapolation of the
exact persistent PECE endpoints instead. Consecutive endpoints may be
extrapolated because each one is an exact corrected evaluation. The
one-interval guard still falls back to the latest endpoint. Every correction
remains an exact model evaluation and owns the Adams history and the next
predictor, so the trajectory is re-anchored at every step. The default `off`
keeps the normal Spectrum cadence. The summary reports the policy and
`sa_pece_skipped_predictions`. This follows the active-PECE policies of
Spectrum MiniMax H3.

#### Experimental continuous bounded corrected-endpoint forecast

The optional `balanced_detail` policy retains `balanced`'s two exact
predicted-phase prefix, skip count, native PECE corrections, and strict
source/weight/condition validation. It only changes what the solver consumes
when the half-log-SNR extrapolation ratio
`alpha = (lambda_i - lambda_last) / (lambda_last - lambda_prev)`
grows beyond one. The original policies abruptly switch at `alpha>1`
from a nearly full secant to a zero-displacement hold, introducing a
discontinuity even at tiny extrapolation-horizon overshoot.
For the opt-in variant, `w=alpha` within [0,1] and `w=1/alpha` above one.
The solver consumes `latest + w*(latest-previous)`. The weight is
nonnegative and at most one, continuous at alpha=1, and decays to zero
as the horizon grows without a separate hard threshold. It introduces
no raw stochastic predictor output or fresh Gaussian innovation, and
makes no additional model evaluations. Incompatible endpoints,
nonfinite geometry and backwards intervals still fail to an exact
anchor hold. Existing policies and defaults are unchanged.

Numerical CPU tests verify boundedness, continuity and independence
from raw stochastic placeholder values. The Krea 2 checkpoint's actual
CUDA texture/identity quality and runtime have **not** been validated
for this opt-in variant.

Scale-Locked PR #16 records planner predictor and corrected-state anchors
independently. Fully locked target predictor and correction steps receive
their corresponding planner values; the original native predictor/corrector
updates still execute. The SA stochastic sampler generates patch-coupled
Gaussian increments indexed by actual sigma transitions, since only the
selected SDE window invokes the noise sampler. Restarted resolution stages
reset Adams history as required by the native solver. CPU execution tests
cover predictor/corrector ordering, source mismatch fail-closed behavior,
stochastic-noise repeatability, callback counts and locked-step output parity.
These checks do not establish image fidelity or CUDA acceleration.
