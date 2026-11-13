# Validation evidence

Validation uses small random Krea 2 weights on CPU. Real checkpoint images, CUDA performance, peak VRAM and image fidelity are unvalidated.

| Check | Executed result | Scope |
| --- | --- | --- |
| Complete CPU suite, PyTorch 2.10 / Python 3.12 | 136 passed on current native sampler container (2026-11-05 CI); earlier-container lane also passed | Numerical fitting, native components, branch/lifecycle isolation, workflow graphs and node entrypoints |
| Full native ComfyUI CPU integration, PyTorch 2.15 | 16 cases passed with each native sampler container | Actual ModelPatcher, CFG batching, native BaseModel, modules and sampler updates |
| Scale-locked integration | Eight- and twelve-step planner/target passes completed with and without Spectrum | Finite singleton-frame outputs; execution compatibility only |
| Packaging | Source archive and wheel built | Entrypoint, source, notices, workflow examples and validation tools |

The native Krea 2 model source is audited at ComfyUI `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`. Sampler-container coverage includes that revision and the earlier native KSAMPLER implementation at `135abed8da169e33ab0b86550e05e3ae55d6df8c`. The older whole revision predates native Krea 2; tests combine its unchanged sampler-container method with the current native model and host infrastructure. Both versions execute identical audited Euler and DPM++ 2M functions. Runtime admission checks the native container class and bound entrypoint. Parameter names and decorator signatures do not gate execution. Signature metadata is logged diagnostically when debug is enabled; unavailable metadata does not disable acceleration.

Full-host cases cover CFG 1 and 4.5, batch one/two, references, non-square odd spatial geometry, standard native LoRA application/restoration, truncated denoise, native Heun fallback, foreign wrapper preservation and Turbo warmup/tail settings. All-actual output is bitwise native; accelerated repeats are reproducible. With Turbo warmup 1/tail 1 and the default one-point bootstrap, the tiny eight-step test completes six actual evaluations and two forecasts (one bootstrap); the twelve-step test completes seven actual evaluations and five forecasts (one bootstrap), for both containers. The bootstrap forecast equals the native projection of the held step-zero features at the second timestep exactly. These counts establish execution, not GPU speed or image quality.

The native-component fixture substitutes CPU host infrastructure, as documented in `tests/native_components.py`. `tests/full_comfy_smoke.py` separately runs actual ComfyUI infrastructure. Its `--sampler-source-root` option loads the earlier upstream container method for compatibility testing.

Container regressions cover renamed parameters, extra optional arguments, variadic wrappers and invalid signature metadata. The full-host tests exercise the optional-argument and variadic cases through real sampling, each completing eight actual calls and four forecasts at twelve steps.

Regression coverage distinguishes precondition rejection from corrupted prediction history: horizon misses retain genuine actual anchors, while nonfinite predictions clear them. Turbo respects explicit warmup and tail settings; the console report includes effective values. Regression forecasts always require at least degree+1 actual anchors and apply the in-domain, degenerate-span, weight-magnitude and nonfinite bounds. The optional one-point bootstrap is the exception: it may forecast the second step from a single actual anchor by holding that anchor's features, so the coordinate, span and weight checks do not apply to it; the nonfinite check still does. The additional history-span extrapolation-distance guard defaults to 0.5, accepts a configurable ratio or can be explicitly disabled. Native CPU integration cases for the Krea 2 `simple` 18-step sigma schedule, truncated to the last 7 and 8 steps, verify the two guard states and relaxed settings; these exercise execution decisions and do not establish visual fidelity.

GitHub Actions tests Python 3.10/3.12/3.13, an additional earlier-container lane, and both full-host container variants. [PR checks](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1/checks) show the current remote results.

Matched examples and the [GPU protocol](GPU_VALIDATION.md) remain the route to real-image acceptance. The benchmark runner has not executed GPU generation; no measured speedup, reduced-weight fidelity, perceptual quality or optimal default is claimed.

### Short partial-denoise schedules and Diff-Aid window reference (2026-11-13)

`tests/test_short_schedules.py` runs native ER-SDE (`max_stage=3`, `s_noise=1`, ER-SDE noise scaler, per-step seeded noise) through the real sampler wrappers on the trailing four and five steps of the native 14- and 16-step Krea 2 `simple` schedules, and asserts the actual/forecast decision of every step, the reported reasons and the anchors of every forecast. With Turbo `warmup=2`, `tail=1`, `window=1`, `bootstrap=false` and the distance guard off:

| Pass | Operator over the pass | Step decisions | Reported reason |
| --- | --- | --- | --- |
| 4 steps | constant (no Diff-Aid, full window, or partial window on absolute noise levels) | A A F A | none |
| 5 steps | constant | A A F A A | none |
| 4 / 5 steps | hard window normalized against the pass's first sigma, boundary 0.75 | all actual | `Diff-Aid sigma-window transition` x1 |
| 4 steps | gain change at the second step | all actual | x1 |
| 5 steps | gain change at the second step | A A A F A | x1 |
| 4 steps | gain change at the last step (hard or ramped) | A A F A | x1 |
| 4 / 5 steps | ramp shoulder spanning the pass | all actual | one per step |
| 4 steps | Sol dense warmup of one or two evaluations | all actual | `numerical attention backend transition` x1 |
| 5 steps | Sol dense warmup of one evaluation | A A A F A | x1 |
| 14 steps from the maximum sigma | hard window, either reference | 9 actual, 5 forecast | x1 |

Further cases cover warmup 3, tail 2, degree 2, window 0, the distance guard at 0.5 (forecast rejected and reported) and 1.5, bootstrap with warmup 0 and 1, split conditional/unconditional streams, four consecutive passes at growing latent sizes through one installed binding (repeated bitwise), interruption from the step callback and from the model after the first forecast, a nested pass inside a callback, and native fallback. In every forecast the anchors are actual evaluations under the forecast step's own gain and route. All-actual and fallback passes are bitwise equal to the plain native solver; a forecasting pass equals it up to the first forecast step.

`tests/full_comfy_smoke.py` repeats the four- and five-step passes with the real Diff-Aid Krea 2 node (`text_residual` and `values_only`), ComfyUI's ModelPatcher/CFGGuider and stochastic ER-SDE. With `sigma_reference=model_sigma_max` and a window boundary outside the pass, the all-actual result is bitwise equal to the fixed operator the gain selects (the unpatched model for gain 0, the full-window patch for gain 1), and the forecasting result is bitwise equal to Spectrum on that fixed operator, with one forecast. With the default reference the same window yields no forecast. On a schedule that starts at the model's maximum sigma both references give bitwise-equal results. The real Sol-Krea2 wrappers (CPU attention oracle) reproduce the dense-warmup rows.

These are CPU scheduling and identity results on small random weights. No Krea 2 checkpoint was sampled: decoded image quality, the visual effect of the window reference, wall time, VRAM and attention-backend behavior on CUDA are unverified.

### RES Multistep and Gradient Estimation (2026-11-06)

The native ComfyUI source-audit contracts now include `sample_res_multistep`, its `res_multistep` delegate, and `sample_gradient_estimation`. Native CPU tests compare all-actual Spectrum runs bit-for-bit with native sampling, enforce default-only solver options and delegate integrity, verify deterministic feature-forecast runs with the 20-step Turbo schedule (11 actual, 9 forecasts), and check callback ordering and hook cleanup. Full ComfyUI ModelPatcher/CFGGuider smoke tests exercise both solvers. The count parity and deterministic execution do not establish image fidelity for real checkpoint runs; revalidate anatomy after enabling forecasts.

### DEIS, iPNDM, and iPNDM-V (2026-11-06)

The remaining Scale-Locked sampler choices now have native source-contract checking and are admitted to Spectrum when their original one-call/one-callback solver semantics and default options hold. CPU tests cover native all-actual parity, feature-forecast determinism, 20-step Turbo execution counts (11 actual/9 forecast), altered DEIS coefficient helper rejection, option override rejection, and old native versions missing the new solver functions. The real ComfyUI integration tests all three through the original ModelPatcher/CFGGuider and sampling functions. The paired Scale-Locked workflow separately verifies the locked target pass (5 actual/3 forecasts from 8 unlocked steps); no full-checkpoint GPU throughput or perceptual fidelity claims follow from those CPU tests.
