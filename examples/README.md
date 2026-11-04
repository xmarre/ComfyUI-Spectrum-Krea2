# Matched ComfyUI workflows

Import a UI workflow `.json` into ComfyUI. The corresponding `.api.json` contains the same graph in API format. Required custom node: **Spectrum Apply Krea 2**, supplied by this repository's [PR #1](https://github.com/xmarre/ComfyUI-Spectrum-Krea2/pull/1). All other nodes are native at ComfyUI `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`.

| Native baseline | Spectrum candidate | Defaults |
| --- | --- | --- |
| `raw_native.json` | `raw_spectrum.json` | RAW, 52 steps, CFG 4.5, Euler/simple, 1024 square |
| `turbo_native.json` | `turbo_spectrum.json` | Turbo, 8 steps, CFG 1.0, Euler/simple, 1024 square |
| `raw_reference_native.json` | `raw_reference_spectrum.json` | RAW with a native reference latent and `index_timestep_zero` |

Model filenames are editable placeholders: select the actual installed Krea 2 diffusion model, compatible Qwen3-VL 4B text encoder and Wan 2.1 VAE in each pair. Keep the text encoder loader's **type `krea2`**. A model with a matching filename but incompatible encoder architecture will fail native conditioning validation.

Reference workflows also require selecting an input image. They encode that image with the chosen VAE and set native reference conditioning on positive and negative branches. Reference execution is structurally tested; real editing quality is pending. If the native baseline does not perform the requested edit, the Spectrum pair cannot establish edit capability.

Within a pair, only Spectrum `enabled` and the output filename prefix differ. Seeds remain fixed. The disabled node passes the original MODEL unchanged. Default profile settings use degree 1, ridge 0.01, warmup 4, tail 2 and one forecast between actual anchors. GPU quality/performance have not yet been validated.

The graphs are generated together by `tools/generate_examples.py`; tests verify UI link integrity, widget order and paired API parity. Use [GPU_VALIDATION.md](../docs/GPU_VALIDATION.md) for the staged test matrix and evidence to collect.
