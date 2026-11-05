# Copyright, software provenance and model terms

## Project source

Copyright (c) 2026 xmarre. Distributed under **GPL-3.0-or-later**, with the full GNU GPL v3 text in [LICENSE](LICENSE). The SPDX headers express the option to use a later GPL version.

| Files / design | Source and treatment |
| --- | --- |
| `forecast.py` | Adaptation of the history-weight Chebyshev ridge formulation and bounded model-dtype accumulation from xmarre's GPL-3.0-or-later Spectrum MiniMax H3, revision `806fe498d574a726fecb8818917f00a4b993fd77`. Reimplemented in a small image-only module; no H3 model-aware/audio/replay code is copied. |
| `adapter.py` | Original capture/lifecycle integration and image-output reconstruction adapted to the native ComfyUI Krea 2 path. ComfyUI model integration source is GPL-3.0-or-later, Copyright (C) 2024 Comfy and contributors. Retains native modules by reference; no native model source is vendored. |
| Runtime, sampling, node, configuration, tests and tools | Krea 2-specific implementation by xmarre, informed by H3's wrapper ownership and testing principles. |
| Native-component tests | Load selected unchanged components from a separately checked-out ComfyUI revision. The test infrastructure is original; upstream model/sampler source is not redistributed here. |
| AST contracts | Digests of inspected native source; no source bodies embedded. |

The primary engineering reference is [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3). Native integration targets [ComfyUI](https://github.com/Comfy-Org/ComfyUI) revision `7ad7adb38f6cbb6c2b0713d3bd44c3fb2b9c8ebe`. The native tensor layout and model API determine this adapter's image slicing, timestep projection and output layout.

## Spectrum research and reference software

**Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo and Stefano Ermon. Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration. arXiv:2603.01623 (2026).** [Paper](https://arxiv.org/abs/2603.01623), [official implementation](https://github.com/hanjq17/Spectrum), inspected revision `4e41f91c93ff66b045332530dd7bcb4c60eafbab`.

The official implementation is MIT, Copyright (c) 2026 Jiaqi Han. It was inspected for scientific/mathematical comparison. No Stanford source file or protected implementation portion is incorporated in this repository; no Stanford code is vendored. Citation identifies the algorithm's authorship. The software adaptation described above derives from the owner's GPL H3 implementation. No third-party ComfyUI Spectrum port is used as an implementation source.

## Krea AI inference code and model weights

The [official Krea 2 inference repository](https://github.com/krea-ai/krea-2), inspected revision `db3984fbc6e13b34c0064990fc2d95ac64d00058`, is Apache-2.0. Its architecture and inference semantics were consulted; no Apache implementation code is copied or vendored here. Krea AI and the Krea 2 contributors developed the underlying model.

Krea 2 weights have the separate [Krea 2 Community License Agreement v1, June 22, 2026](https://www.krea.ai/krea-2-licensing), reviewed November 4, 2026. It permits commercial use only below $1 million total affiliated annual revenue; otherwise an enterprise license is required. Distribution of covered models/derivatives requires the agreement and specified notices/naming. Deployment includes reasonable content-filter measures and applicable disclosure/AUP obligations. Read the agreement for the complete conditions, restrictions and termination provisions.

This repository distributes accelerator source, without Krea 2 or encoder weights. Its GPL software license does not grant model-weight rights, and publishing an independent accelerator does not automatically impose every model-distribution condition on its source. Users downloading or deploying covered weights must separately comply with their applicable model terms. No claim is made that the model weights are Apache-2.0 or GPL.
