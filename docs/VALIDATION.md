# Validation evidence

Validation uses small random Krea 2 weights on CPU. Real checkpoint images, CUDA performance, peak VRAM and image fidelity are unvalidated.

| Check | Executed result | Scope |
| --- | --- | --- |
| Complete CPU suite, PyTorch 2.10 / Python 3.12 | 124 passed with each native sampler container | Numerical fitting, native components, branch/lifecycle isolation, workflow graphs and node entrypoints |
| Full native ComfyUI CPU integration, PyTorch 2.15 | 16 cases passed with each native sampler container | Actual ModelPatcher, CFG batching, native BaseModel, modules and sampler updates |
| Scale-locked integration | Eight- and twelve-step planner/target passes completed with and without Spectrum | Finite singleton-frame outputs; execution compatibility only |
| Packaging | Source archive and wheel built | Entrypoint, source, notices, workflow examples and validation tools |

The native Krea 2 model source is audited at ComfyUI `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`. Sampler-container coverage includes that revision and the earlier native KSAMPLER implementation at `135abed8da169e33ab0b86550e05e3ae55d6df8c`. The older whole revision predates native Krea 2; tests combine its unchanged sampler-container method with the current native model and host infrastructure. Both versions execute identical audited Euler and DPM++ 2M functions. Runtime admission checks the native container class and bound entrypoint. Parameter names and decorator signatures do not gate execution. Signature metadata is logged diagnostically when debug is enabled; unavailable metadata does not disable acceleration.

Full-host cases cover CFG 1 and 4.5, batch one/two, references, non-square odd spatial geometry, standard native LoRA application/restoration, truncated denoise, native Heun fallback, foreign wrapper preservation and Turbo warmup/tail settings. All-actual output is bitwise native; accelerated repeats are reproducible. With Turbo warmup 1/tail 1 and the default one-point bootstrap, the tiny eight-step test completes six actual evaluations and two forecasts (one bootstrap); the twelve-step test completes seven actual evaluations and five forecasts (one bootstrap), for both containers. The bootstrap forecast equals the native projection of the held step-zero features at the second timestep exactly. These counts establish execution, not GPU speed or image quality.

The native-component fixture substitutes CPU host infrastructure, as documented in `tests/native_components.py`. `tests/full_comfy_smoke.py` separately runs actual ComfyUI infrastructure. Its `--sampler-source-root` option loads the earlier upstream container method for compatibility testing.

Container regressions cover renamed parameters, extra optional arguments, variadic wrappers and invalid signature metadata. The full-host tests exercise the optional-argument and variadic cases through real sampling, each completing eight actual calls and four forecasts at twelve steps.

Regression coverage distinguishes precondition rejection from corrupted prediction history: horizon misses retain genuine actual anchors, while nonfinite predictions clear them. Turbo respects explicit warmup and tail settings; the console report includes effective values. At least degree+1 actual anchors and numerical horizon bounds still apply.

GitHub Actions tests Python 3.10/3.12/3.13, an additional earlier-container lane, and both full-host container variants. [PR checks](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1/checks) show the current remote results.

Matched examples and the [GPU protocol](GPU_VALIDATION.md) remain the route to real-image acceptance. The benchmark runner has not executed GPU generation; no measured speedup, reduced-weight fidelity, perceptual quality or optimal default is claimed.
