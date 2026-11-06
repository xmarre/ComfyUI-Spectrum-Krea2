# ComfyUI Spectrum Krea 2

**Spectrum feature forecasting for ComfyUI's native Krea 2 RAW and Turbo image models.** Insert **Spectrum Apply Krea 2** after the diffusion model loader and LoRAs, before the normal sampler. The node uses the loaded model, native conditioning, CFG, VAE and output projection; it does not load another transformer.

Based on **[Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration](https://arxiv.org/abs/2603.01623)** by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon, with the engineering foundation of [Spectrum MiniMax H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3). The original algorithm is Spectrum by Han et al.; this repository provides an independent Krea 2 integration.

## Status

Version **0.1.0**, initial GPU-validation candidate. CPU tests exercise native Krea 2 model components and sampler source with small test weights. **Real Krea 2 checkpoint image quality, CUDA precision behavior and wall-clock acceleration are not yet validated.** Defaults are conservative engineering choices, not measured optima. Forecast counts measure computation avoided; they do not establish a speedup or fidelity.

The audited ComfyUI revision is [`7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`](https://github.com/Comfy-Org/ComfyUI/tree/7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe). The integration checks the native sampler container by class and bound entrypoint, and checks model/solver source contracts rather than requiring that exact whole-repository version. Earlier native callback-container code is tested as well; logging changes alone do not disable forecasting. Changed contracts produce an explicit native fallback. Requires Python 3.10+, PyTorch 2.4+, and native Krea 2 support in ComfyUI. No additional runtime dependency or compiled extension is introduced.

## Install and use

1. Add `https://github.com/xmarre/ComfyUI-Spectrum-Krea2` in ComfyUI Patcher and apply the overlay for **[PR #1](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1)**. Restart ComfyUI.
2. Load a Krea 2 RAW or Turbo diffusion model, its compatible Qwen3-VL **4B / 12-layer-tap** text encoder with **CLIPLoader type `krea2`**, and a compatible Wan 2.1 VAE.
3. Connect the final loader/LoRA `MODEL` to **Spectrum Apply Krea 2**, then connect its `MODEL` output to the usual KSampler. Select `raw` or `turbo` to match the weights.
4. Start with **Euler**, **simple** scheduler, and the paired examples below. Compare native and Spectrum with identical seeds and all other inputs.

Disabling the node on an ordinary model returns the original MODEL unchanged. Reapplying the node replaces its own wrappers on a clone. Disabling a previously Spectrum-patched clone removes its Spectrum wrappers and preserves other wrappers.

| Initial test | Steps | ComfyUI CFG | Profile | Warmup / tail | Forecast window |
| --- | ---: | ---: | --- | --- | ---: |
| RAW, 1024 × 1024 | 52 | 4.5 | raw | 4 / 2 | 1 |
| Turbo, 1024 × 1024 | 8 | 1.0 | turbo | 4 / 2 | 1 |

The official RAW inference example uses `guidance=3.5` in `cond + guidance*(cond-uncond)`, equivalent to ComfyUI's `uncond + 4.5*(cond-uncond)`. Official Turbo disables CFG with zero in its own API; **ComfyUI CFG 1.0** selects the conditional prediction. These are different API conventions.

RAW and Turbo have the same native class and no verified distillation flag. Profile selection is explicit. At eight steps, Turbo permits at most one forecast with these defaults; numerical or memory checks can reduce that to zero. Short trajectories may remain entirely native.

## Compatibility

| Path | Implementation contract | Validation |
| --- | --- | --- |
| RAW / Turbo, Euler with no churn | Forecast-enabled | CPU source/component tests; GPU pending |
| Native DPM++ 2M | Forecast-enabled | CPU source/component tests; GPU pending |
| Strictly decreasing custom schedules, truncated denoise / partial stages | Fresh state per sampler invocation; warmup and tail use the actual received schedule | CPU tests; real workflows pending |
| Native CFG, combined or separate branches; batches | Independent histories keyed by ordered condition UUIDs and full layout | CPU tests; real checkpoints pending |
| Text-to-image / image-to-image | Same native image denoiser contract | CPU structural tests; GPU pending |
| Native reference `index` / `index_timestep_zero` | Capture only generated-image rows after native reference processing; project current-timestep image rows | CPU tests and paired example; edit quality pending |
| FP32 / BF16 / FP16 | Model-dtype snapshots; FP32 accumulation; native projection | CPU tests; CUDA/reduced-weight formats pending |
| Standard weight LoRAs installed before this node | Original patched modules execute; state belongs to the current weight-patch identity | Lifecycle tests; real LoRA images pending |
| 4D images / 5D with one temporal slice | Forecast-enabled; padding and output layout preserved | CPU tests |
| Other samplers, repeated/nonmonotonic sigmas, Euler churn | Native fallback | Parity/fallback tests |
| ControlNet, latent inpaint masks, regional masks/areas, scheduled hooks, custom CFG, transformer modifications, compiled model replacements, multi-device execution | Native fallback | Contract tests; no acceleration claim |
| Multi-frame 5D input | Native fallback | No video support claim |

Attention dispatch remains native. Unknown transformer wrappers fall back conservatively because skipping them could remove required side effects. Disabled and bypassed execution do not change the native model math. Native failures propagate with guaranteed hook/context cleanup.

### Experimental qualified attention-patch interoperability

The opt-in [Spectrum Krea 2 PR #2](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/2) overlay adds narrowly qualified support for two MODEL patch providers:

- [Sol-H3 PR #44](https://github.com/xmarre/ComfyUI-Sol-H3/pull/44), through its **Sol-Krea2 Sparse Attention (Experimental)** node. Its own first dense evaluation remains part of the numerical attention history. BF16 uses packaged Sana CuTe; native FP16 uses the ComfyUI Kitchen kernel with protected text-conditioning blocks and 256 additional selected tokens by default. The FP16 route is experimental and has not been validated on a real checkpoint.
- [Diff-Aid PR #12](https://github.com/xmarre/ComfyUI-DiffAid-Patches/pull/12), through its Krea 2 node in **text_residual**, **values_only**, or **keys_and_values** mode, for full nonramped sigma windows.

Install the relevant PR overlays using ComfyUI Patcher. MODEL wiring:

    Krea 2 loader / LoRA (Sage auto optional)
      -> Krea 2 Diff-Aid (optional)
      -> Sol-Krea2 Sparse Attention (Experimental; optional)
      -> Spectrum Apply Krea 2
      -> guider / sampler

A combined run requires **both** Spectrum PR #2 and Sol-H3 PR #44; add Diff-Aid PR #12 if Diff-Aid is used. The integration refuses unknown transformer replacements, unreviewed model-function wrappers, and partial/ramped Diff-Aid sigma windows. These paths continue to use native sampling without forecasts.

Spectrum does not extrapolate from anchors collected under a different attention backend. With Sol-Krea2's default `dense_evaluations=1`, the initial dense-to-sparse switch invalidates the first captured feature and may cost forecasts. To **align the two actual warmup steps** without an attention-route transition, explicitly set Sol-Krea2 `dense_evaluations=0`, retain Spectrum `warmup=2` and `bootstrap=false`, and leave Sol `dense_layers=2` and `extra_tokens=256` for initial quality tests. Spectrum still executes both initial denoiser evaluations; the first is now sparse rather than fully dense. This may alter initial image composition, so use a matched-seed dense comparison before accepting the change.

There is no mathematically exact, zero-cost conversion from one dense feature anchor to a sparse feature anchor: the dense and sparse operators produce different features, and a later sparse feature alone cannot reconstruct the former at the same timestep. Do not fabricate anchors or disable attention-route segregation. The *separate* distance-extrapolation guard is user-configurable; changing it permits farther feature predictions within the same numerical backend but may cause image artifacts. If sparse-from-start causes visible artifacts, restore Sol `dense_evaluations=1` and retain the safe transition. Actual forecast counts also depend on the sigma schedule, remaining token budget, and feature-history conditions.

Real-ComfyUI CPU integration verified the combined wrapper stack with an exact CPU substitute for Sol's CUDA attention kernel. **Real checkpoint CUDA execution, numerical fidelity, end-to-end acceleration, and decoded image quality remain unverified.** A nonzero forecast counter demonstrates skipped evaluations, not a measured speedup.

## Controls

- **degree / ridge**: Chebyshev polynomial degree and ridge regularization. Default degree 1 requires two actual anchors. Higher degrees require more history and can trigger numerical bounds.
- **warmup / tail**: initial/final actual outer steps. At least `degree + 1` actual observations are needed per conditioning stream.
- **bootstrap** (default on): with degree 1, warmup 0 or 1 and window of at least 1, the second outer step is forecast from the first actual step alone by holding its transformer features and projecting them with the current timestep, as Spectrum H3 does. Without it, the first forecast needs two actual anchors. The report counts these as `bootstrap_forecast`.
- **window / growth**: maximum consecutive forecasts between actual anchors. `window=0, growth=0` is an all-actual diagnostic. Growth is bounded at four forecasts; Turbo caps the window at one forecast and disables growth. Both profiles respect the chosen warmup and tail; defaults are four and two.
- **max_history / budget_mb**: sliding actual-only history. All conditioning streams share one retained-feature budget. Temporary snapshots, predictions, model activations and allocator reservations need additional memory.
- **storage**: `vram` avoids history transfers; `system_ram` moves retained snapshots to CPU and incurs transfers when predicting. Neither option retains full FP32 feature-space coefficients.
- **extrapolation_guard_enabled** (default **true**): controls only the history-span distance guard. Disable it to allow a regression forecast farther than the configured history-span ratio. This is experimental and may create incorrect anatomy, duplicates or other artifacts. The history-span degeneracy check, in-domain requirement, fitted-weight L1 bound and nonfinite checks still apply.
- **max_extrapolation_span_ratio** (default **0.50**, range 0.05–8.00): maximum allowed forecast distance as a multiple of the retained real-history span. For example, 1.5 allows a gap up to 150% of the observed span; 0.5 allows 50%. Only applies when the guard is enabled. Raising the ratio may improve forecast counts without a proportional reduction in model calls elsewhere because the sampling schedule, Turbo window, tail and stability gates remain authoritative.
- **debug**: explain actual-evaluation reasons. Every sampler invocation emits a JSON summary with completed actual/forecast call counts, effective guard settings and retained-history peak. Normal ComfyUI INFO logging must be enabled to see it.

All forecasts must be causal and inside the sampler's normalized sigma domain. With the distance guard enabled they must also remain within the selected fraction of the retained actual-history span. Even with that guard disabled, the regression-weight L1 bound (8), degenerate-history check, nonfinite checks and native fallback behavior remain active. These checks cannot establish image fidelity.

### Short high-resolution stages

In the observed Krea 2 Turbo 18-step `simple` scheduler, Scale-Locked Regenerate with `composition_lock_sigma=0.70` uses the final **7** steps as a high-resolution stage. With Spectrum `warmup=2`, `tail=1`, `window=1`, `bootstrap=false`, the existing 0.5 ratio admits **1 forecast**; raising the ratio to **1.5** or disabling the distance guard admits **2 forecasts** in a same-schedule CPU scheduling test. With `composition_lock_sigma=0.72`, the final **8** steps admit **2** and **3** forecasts, respectively. Retain the composition-lock threshold that produces the best anatomy; changing this Spectrum control does not change the resolution handoff. These counts are *schedule-specific execution tests*, not full-checkpoint CUDA quality claims. Compare identical seeds before adopting a more permissive guard.

## Examples and validation

Open a UI `.json` file from [examples](examples/README.md). Files ending `.api.json` are corresponding API prompts. Paired files differ only in Spectrum enablement and output filename. Choose your installed model/encoder/VAE filenames in both copies. The RAW reference pair also requires an input image.

[GPU validation](docs/GPU_VALIDATION.md) supplies the test matrix, paired image-review protocol, benchmark runner and PSNR/SSIM/optional LPIPS comparison tool. [Architecture](docs/ARCHITECTURE.md) documents interception, lifecycle, numerical formulation and compatibility boundaries. [Validation evidence](docs/VALIDATION.md) records checks actually run.

## Attribution and licensing

- **Spectrum research:** Han et al., [paper](https://arxiv.org/abs/2603.01623), [official implementation](https://github.com/hanjq17/Spectrum) (MIT). No Stanford implementation source is vendored or copied.
- **Engineering foundation:** xmarre's [Spectrum MiniMax H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3), GPL-3.0-or-later. History-weight ridge formulation and lifecycle principles were adapted for images; H3 audio, continuation and offline replay subsystems are omitted.
- **Native integration:** [ComfyUI](https://github.com/Comfy-Org/ComfyUI), GPL-3.0-or-later. Comfy contributors provide the native Krea 2 implementation, model patching, sampling and attention infrastructure.
- **Model:** [Krea AI](https://github.com/krea-ai/krea-2). Its inference repository is Apache-2.0; model weights have separate terms.

This repository's Python source is **GPL-3.0-or-later**. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model or text-encoder weights are distributed. The [Krea 2 Community License](https://www.krea.ai/krea-2-licensing) applies to downloaded weights and their use/deployment; it is not the Apache inference-code license. See the provenance record for the distinction and relevant conditions.
