# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
from functools import partial
from .config import Config
from .compat import native_model
from .sampling import Binding, KEY
from .adapter import diffusion


class SpectrumApplyKrea2:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "preset": (["raw", "turbo"], {"default": "raw", "tooltip": "Select the loaded weights explicitly. Turbo limits forecasts to one between actual anchors."}),
            "degree": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Chebyshev degree; requires at least degree + 1 actual anchors."}),
            "ridge": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 10.0, "step": 0.001}),
            "warmup": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Initial actual outer steps. Forecasts additionally require degree + 1 actual anchors per conditioning stream."}),
            "tail": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "Final actual outer steps. The selected value is respected in both profiles."}),
            "window": ("INT", {"default": 1, "min": 0, "max": 4, "tooltip": "Maximum consecutive forecasts between actual anchors. Zero with growth zero is an all-actual diagnostic."}),
            "growth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Additional forecast budget per actual anchor, capped at four. Disabled for Turbo."}),
            "max_history": ("INT", {"default": 8, "min": 2, "max": 32}),
            "storage": (["vram", "system_ram"], {"tooltip": "VRAM avoids feature transfers. System RAM trades bandwidth for device memory."}),
            "budget_mb": ("INT", {"default": 4096, "min": 64, "max": 65536, "step": 64, "tooltip": "Total retained feature budget across all conditioning streams; temporary tensors require additional memory."}),
            "debug": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "bootstrap": ("BOOLEAN", {"default": True, "tooltip": "Forecast the second outer step by holding the first actual "
                                      "step's features, as Spectrum H3 does. Applies with degree 1, warmup 0 or 1 and window 1 or more."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patches/spectrum"
    DESCRIPTION = "Spectrum feature forecasting for native Krea 2 RAW/Turbo. GPU quality and performance require paired validation."

    def apply(self, model, enabled=True, preset="raw", degree=1, ridge=0.01,
              warmup=4, tail=2, window=1, growth=0.0, max_history=8,
              storage="vram", budget_mb=4096, debug=False, bootstrap=True):
        from comfy.patcher_extension import WrappersMP
        types = (WrappersMP.SAMPLER_SAMPLE, WrappersMP.PREDICT_NOISE, WrappersMP.DIFFUSION_MODEL)
        installed = any(model.get_wrappers(t, KEY) for t in types)
        if not enabled and not installed:
            return (model,)
        cloned = model.clone()
        for kind in types:
            cloned.remove_wrappers_with_key(kind, KEY)
        if not enabled:
            return (cloned,)
        if not native_model(getattr(getattr(model, "model", None), "diffusion_model", None)):
            raise ValueError("Spectrum Apply Krea 2 requires ComfyUI's native Krea 2 MODEL")
        config = Config(enabled, preset, degree, ridge, warmup, tail, window, growth,
                        max_history, storage, budget_mb, debug, bool(bootstrap))
        binding = Binding(config)
        cloned.add_wrapper_with_key(types[0], KEY, binding.sampler)
        cloned.add_wrapper_with_key(types[1], KEY, binding.predict)
        cloned.add_wrapper_with_key(types[2], KEY, partial(diffusion, binding))
        return (cloned,)


NODE_CLASS_MAPPINGS = {"SpectrumApplyKrea2": SpectrumApplyKrea2}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumApplyKrea2": "Spectrum Apply Krea 2"}
