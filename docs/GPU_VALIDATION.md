# Paired GPU validation

This candidate has no real-checkpoint GPU measurements. CPU structure/parity tests do not establish perceptual quality, prompt adherence, text rendering, or acceleration. Use the same model file, LoRAs, encoder, VAE, seed, resolution, scheduler, sampler, CFG, denoise and attention backend in each pair.

## Immediate ComfyUI Patcher test

Apply [PR #1](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1) to `https://github.com/xmarre/ComfyUI-Spectrum-Krea2` and restart ComfyUI. Load `examples/raw_native.json` and `examples/raw_spectrum.json`. Select installed model/encoder/VAE files in both workflows. Keep the text encoder type `krea2` and keep each pair's seed fixed.

Begin at 1024 × 1024, RAW 52 steps/CFG 4.5, then Turbo 8 steps/CFG 1.0, Euler/simple, degree 1, ridge 0.01, warmup 4, tail 2, window 1, growth 0. Use the node's profile matching the weights. Turbo may offer only one forecast; a native result with zero forecasts is valid when a bound is hit, but establishes no acceleration.

For references use the RAW reference pair, choose the same input image, and retain `index_timestep_zero` on positive and negative branches. Evaluate the baseline's editing capability first. A structurally valid reference path does not imply a base checkpoint is an effective edit model.

## Compact test matrix

Run seed 42 on all three scene prompts first. If visually acceptable, add seeds 1 and 12345. Compare crops at 100% as well as full composition. Do not run an exhaustive Cartesian product before the initial pairs pass.

| Stage | Settings | What to inspect |
| --- | --- | --- |
| Initial RAW | 52 steps, CFG 4.5, 1024 square, Euler/simple | Composition, conditioning, faces/hands, texture, text |
| Initial Turbo | 8 steps, CFG 1, 1024 square, Euler/simple | Whether the single forecast saves time; distilled-model sensitivity |
| Shape and batch | One accepted RAW prompt, 1216 × 832, batch 2 | Geometry, independent batch samples, packing and memory |
| Image-to-image | Encode an input with the same VAE, denoise 0.55, same seed | Truncated-schedule reset and image structure |
| Reference | RAW reference workflow pair, one image | Identity, retained details, native reference limitations |
| Sampler | One accepted RAW pair with DPM++ 2M | Forecast count and image stability under a different update rule |
| LoRA/precision | One accepted RAW pair with a normal weight LoRA, then the deployed reduced-weight format | Loaded patch identity, forecast dtype and artifacts |
| Lifecycle | Queue two different prompts, then repeat the first fixed-seed prompt; cancel once and requeue | Fresh warmup, repeatability, no stale history/hook failures |

Three prompts cover distinct defects without requiring a large matrix:

1. **People and fine mechanics:** A candid photograph of a watchmaker holding a tiny brass gear with both hands, detailed fingertips and engraved metal, daylight through a workshop window, a handwritten card reading SPECTRUM on the desk.
2. **Text and geometry:** A straight-on product photograph of a cream enamel sign with exactly two lines of crisp lettering, “KREA 2” and “SPECTRUM”, beside a precision-cut transparent acrylic cube and a striped ceramic cup. Soft studio illumination, realistic reflections, clean typography.
3. **Complex scene:** A wide architectural photograph of a busy greenhouse café in late afternoon. Thin iron rafters, dense climbing plants, patterned glass, overlapping chairs, customers at several tables, steam from an espresso machine, natural skin tones and fine leaf veins.

## Evidence and decisions

Collect complete console summaries/warnings, paired output PNGs and the run configuration. Record sampler elapsed seconds after equal warmup, completed actual and forecast diffusion calls, peak allocated VRAM, and retained-history peak. CFG batching changes the relationship between outer steps and diffusion calls: report both, and compare like-for-like batching.

A native bypass must match the native result under the same deterministic backend. For enabled forecasting, record PSNR, SSIM and LPIPS as descriptive paired fidelity evidence; **no universal numeric threshold is asserted**. Scores compare a changed trajectory to one native seed, not image quality itself. A high score cannot excuse broken text, anatomy or reference identity. A lower score may reflect a plausible composition shift that still requires explicit review.

For each pair, review global composition/prompt compliance, exact requested lettering, face and hand anatomy, straight edges/repeating structures, small texture, reference consistency and new noise/smearing. Mark pass/fail/uncertain with representative crops. Accept a profile only when the tested prompts/seeds show no material new defects and measured sampler time improves enough to justify memory and complexity. Record adverse cases; tune only against a separate set and retest previous accepted pairs after changes.

## Real-ComfyUI benchmark runner

`tools/benchmark.py` uses real native model loading, Krea 2 text encoding, `nodes.common_ksampler` and VAE decoding. It runs equal untimed warmups and alternates native/Spectrum order by seed. CUDA synchronization brackets sampler timing, peak allocated VRAM is reset per timed sample, and image decoding/saving is excluded. Loading/offloading performed inside sampling remains part of measured sampler time. The baseline uses a read-only native wrapper to count completed diffusion calls; Spectrum reports distinguish actual calls from forecasts. It writes `benchmark.json` and paired PNGs incrementally.

The runner is optional; the UI workflows provide the initial test route. It must be launched in the existing ComfyUI Python environment with `--comfy-root`, model/encoder/VAE filenames and `--preset raw` or `--preset turbo`. `--help` lists all arguments. No repository switching or manual Git operations are needed. The runner is source-checked and has CLI smoke coverage; actual CUDA execution still needs validation.

`tools/compare_images.py DIRECTORY` writes PSNR/SSIM JSON and side-by-side panels for paired PNGs. Add `--lpips` when LPIPS and compatible torchvision are installed; first use may download AlexNet feature weights. Optional metric dependencies are declared in `pyproject.toml`. Identical images use `identical=true` and a null PSNR value to represent infinite PSNR without invalid JSON. These metrics use saved 8-bit RGB images and do not measure latent-space error or perceptual quality in isolation.
