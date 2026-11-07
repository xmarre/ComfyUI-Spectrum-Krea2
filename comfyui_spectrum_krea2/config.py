# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
from dataclasses import dataclass, replace
import math


# Active SA-Solver PECE acceleration: number of leading outer steps whose
# predicted phase always runs the model. "off" keeps Spectrum's normal cadence.
SA_PECE_POLICIES = {"off": None, "stable_start": 3, "balanced": 2, "max_speed": 1,
                    "balanced_detail": 2}


@dataclass(frozen=True)
class Config:
    enabled: bool = True
    preset: str = "raw"
    degree: int = 1
    ridge: float = 0.01
    warmup: int = 4
    tail: int = 2
    window: int = 1
    growth: float = 0.0
    max_history: int = 8
    storage: str = "vram"
    budget_mb: int = 4096
    debug: bool = False
    bootstrap: bool = True
    extrapolation_guard_enabled: bool = True
    max_extrapolation_span_ratio: float = 0.5
    sa_pece_forecast_policy: str = "off"

    def __post_init__(self):
        if self.preset not in {"raw", "turbo"}:
            raise ValueError("Select the RAW or Turbo checkpoint profile explicitly")
        if not 1 <= self.degree <= 4:
            raise ValueError("degree must be between 1 and 4")
        if not math.isfinite(self.ridge) or not 0 <= self.ridge <= 10:
            raise ValueError("ridge must be finite and in [0, 10]")
        if not self.degree + 1 <= self.max_history <= 32:
            raise ValueError("max_history must be at least degree + 1 and at most 32")
        if not 0 <= self.window <= 4 or not math.isfinite(self.growth) or not 0 <= self.growth <= 1:
            raise ValueError("window must be in [0, 4]; growth must be in [0, 1]")
        if not 0 <= self.warmup <= 100 or not 1 <= self.tail <= 100:
            raise ValueError("warmup must be in [0, 100] and tail in [1, 100]")
        if self.storage not in {"vram", "system_ram"} or not 64 <= self.budget_mb <= 65536:
            raise ValueError("Invalid history storage or memory budget")
        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be a boolean")
        if type(self.extrapolation_guard_enabled) is not bool:
            raise ValueError("extrapolation_guard_enabled must be a boolean")
        ratio = self.max_extrapolation_span_ratio
        if (type(ratio) not in (int, float) or not math.isfinite(ratio)
                or not 0.05 <= ratio <= 8.0):
            raise ValueError("max_extrapolation_span_ratio must be finite and in [0.05, 8.0]")
        if self.sa_pece_forecast_policy not in SA_PECE_POLICIES:
            raise ValueError("sa_pece_forecast_policy must be one of: " + ", ".join(SA_PECE_POLICIES))

    @property
    def bootstrap_active(self):
        # The one-point hold replaces a degree-1 fit at the second outer step.
        # A longer warmup already covers that step; window 0 is all-actual.
        return self.bootstrap and self.degree == 1 and self.warmup <= 1 and self.window >= 1

    def resolved(self):
        # No reliable RAW/Turbo distinction exists in native model metadata.
        if self.preset == "turbo":
            return replace(self, window=min(1, self.window), growth=0.0)
        return self
