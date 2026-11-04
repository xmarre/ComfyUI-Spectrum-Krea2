# Validation evidence

## Executed checks

Local validation used Python 3.12 on CPU with the audited ComfyUI source revision `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`. No Krea 2 model weights, text-encoder weights or GPU were available.

| Check | Observed result | Scope |
| --- | --- | --- |
| Numerical, component, lifecycle, node and workflow tests with PyTorch 2.10.0+cpu | **114 passed** | Complete final CPU suite, including repository-name import reproduction |
| Earlier complete suite with PyTorch 2.15.0+cpu | **110 passed** before the final callback/accounting/entrypoint regression cases were added | Additional PyTorch version evidence; final full-host integration below uses 2.15 |
| Full native ComfyUI integration with PyTorch 2.15.0+cpu | **12 cases passed** | Real ModelPatcher, CFGGuider, condition batching, Krea2 BaseModel transformations, native attention dispatch, native blocks and samplers; tiny random test weights |
| Build | Source archive and wheel built successfully | Archive includes entrypoint, notices, test helpers, workflows, docs and tools; wheel includes both license documents |
| CLI and image metrics smoke | Benchmark help and PSNR/SSIM comparison succeeded | Identical RGB pair produced SSIM 1 and the documented infinite-PSNR representation; no LPIPS execution |
| Example graphs | UI/API pairing, links, seed widgets and deterministic regeneration passed | Six UI workflows and six corresponding API prompts; browser rendering not exercised |

The full-host integration covers eight Euler/DPM++ 2M combinations with CFG 1 or 4.5, batch one/two, odd non-square spatial sizes and native `index_timestep_zero` references; a standard native LoRA weight adapter and restoration; denoise 0.55; native Heun fallback; and preservation of an external diffusion wrapper's modified output. All-actual wrapper execution is bitwise native in these cases. Repeated accelerated runs are reproducible. In the tiny twelve-step accelerated cases, completed diffusion calls were nine actual plus three forecast; Heun fallback completed twenty-three native calls and zero forecasts. These counts are structural execution evidence, **not performance measurements**.

Component-level tests additionally cover reference `index`, single-frame 5D and 4D layouts, BF16 projection parity, FP16/BF16 forecast accumulation, sliding memory bounds, separately evaluated CFG branches, changed UUID/layout/precision/weight identity, cancellation, native exceptions, nested callback samplers, missing identity, nonfinite predictions and unsupported options. Actual model calls preserve native output bitwise; projecting captured image rows alone uses dtype-appropriate tolerance because the GEMM row count differs.

The reduced component fixture uses unchanged upstream model/sampler code with CPU host infrastructure substitutes, documented in `tests/native_components.py`. The separate `tests/full_comfy_smoke.py` uses real ComfyUI infrastructure and does not install those substitutes. Neither test loads real checkpoint weights or validates image-generation quality.

## Review fixes

Review of the published PR diff resulted in callback isolation from trajectory context, consistent completed-call accounting, cleanup if hook registration fails, explicit native fallback for latent inpaint masks, and complete source-distribution contents. GitHub's first CI run exposed entrypoint importing under a hyphenated repository directory; package and direct-import forms are now explicitly covered and the CI directory layout was reproduced locally. The fixes remain on the implementation PR.

GitHub Actions runs the CPU contract suite on Python 3.10, 3.12 and 3.13 with CPU PyTorch 2.10, and a separate full-native CPU integration lane on Python 3.12/PyTorch 2.15. The [PR checks](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1/checks) are authoritative for their current remote status. The normal suite requires `COMFYUI_ROOT` to point at the audited source checkout; CI provides it explicitly. The full-host script takes `--comfy-root` and needs the integration dependencies declared in the workflow.

## Unverified

Real RAW/Turbo images, image editing capability, CUDA/Blackwell behavior, reduced-weight loaders, attention-backend differences, peak GPU memory, sampler wall time and PSNR/SSIM/LPIPS on generated pairs remain unmeasured. The benchmark runner has not executed a real GPU generation. The defaults have not passed empirical acceptance. Use [GPU_VALIDATION.md](GPU_VALIDATION.md) and the paired examples to collect that evidence before promoting a profile or making quality/speed claims.
