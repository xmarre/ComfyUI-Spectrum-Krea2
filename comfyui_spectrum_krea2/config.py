# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
from dataclasses import dataclass, replace
import math


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

    def resolved(self):
        # No reliable RAW/Turbo distinction exists in native model metadata.
        if self.preset == "turbo":
            return replace(self, window=min(1, self.window), growth=0.0)
        return self
