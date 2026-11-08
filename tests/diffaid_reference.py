# SPDX-License-Identifier: MIT
# Copyright (c) 2026 xmarre
"""Diff-Aid sigma-window helpers, copied unchanged from ComfyUI-DiffAid-Patches.

Source: xmarre/ComfyUI-DiffAid-Patches nodes.py at 99a0ca7 (the Krea 2 Diff-Aid
revision exercised by CI). Spectrum binds these helpers from the installed
Diff-Aid module at runtime; unit tests install this copy in a stand-in module.
"""
import torch

STATE_KEY = "diffaid_runtime_state"
_EPSILON = 1.0e-12


def _smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sigma_window_gain(normalized_sigma: torch.Tensor, start: float, end: float, ramp: float) -> torch.Tensor:
    normalized_sigma = normalized_sigma.float().reshape(-1).clamp(0.0, 1.0)
    if start <= 0.0 and end >= 1.0:
        return torch.ones_like(normalized_sigma)
    if ramp <= 0.0:
        return ((normalized_sigma >= start) & (normalized_sigma <= end)).to(dtype=normalized_sigma.dtype)

    # Keep sigma_start/sigma_end as the full-strength window boundaries.
    # Ramp only softens the shoulders outside that window, so a boundary that
    # touches 0.0 or 1.0 is not accidentally reduced to half strength.
    ramp = max(float(ramp), _EPSILON)
    if start <= 0.0:
        left = torch.ones_like(normalized_sigma)
    else:
        left_edge = max(0.0, start - ramp)
        left_width = max(start - left_edge, _EPSILON)
        left = _smoothstep01((normalized_sigma - left_edge) / left_width)

    if end >= 1.0:
        right = torch.ones_like(normalized_sigma)
    else:
        right_edge = min(1.0, end + ramp)
        right_width = max(right_edge - end, _EPSILON)
        right = 1.0 - _smoothstep01((normalized_sigma - end) / right_width)

    return (left * right).clamp(0.0, 1.0)


def _coerce_batch_value(value, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        t = value.to(device=device, dtype=dtype).reshape(-1)
    else:
        t = torch.tensor([float(value)], device=device, dtype=dtype)
    if t.numel() == 1:
        return t.repeat(batch_size)
    if t.numel() == batch_size:
        return t
    return t[:1].repeat(batch_size)
