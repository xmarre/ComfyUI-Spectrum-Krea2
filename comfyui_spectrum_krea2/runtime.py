# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
import contextvars
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from .forecast import History, ForecastUnavailable

LOG = logging.getLogger(__name__)
CURRENT = contextvars.ContextVar("spectrum_krea2_session", default=None)
REPORTS = contextvars.ContextVar("spectrum_krea2_reports", default=None)


@dataclass
class Stream:
    history: History
    route: str | None = None
    last_step: int = -1
    last_actual: int = -1
    actual_anchors: int = 0


@dataclass
class Session:
    binding: object
    config: object
    coordinates: list
    model: object
    weight_identity: object
    batch: int
    spatial: tuple
    step: int = 0
    active_step: int | None = None
    predict_calls: int = 0
    streams: dict = field(default_factory=dict)
    counts: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    disabled_reason: str | None = None
    peak_history_bytes: int = 0
    weight_owner: object = None
    patch_policy: object = None
    pece_enabled: bool = False
    pece_corrector_order: int = 0
    pece_pending_step: int | None = None
    pece_sigmas: tuple = ()

    def disable(self, reason):
        if self.disabled_reason is None:
            self.disabled_reason = reason
            LOG.warning("Spectrum Krea 2: native fallback: %s", reason)
        self.streams.clear()

    def reason(self, reason):
        self.reasons[reason] += 1
        if self.config.debug and self.reasons[reason] == 1:
            LOG.info("Spectrum Krea 2: actual evaluation: %s", reason)

    @property
    def nbytes(self):
        return sum(s.history.nbytes for s in self.streams.values())

    def stream(self, key):
        if key not in self.streams:
            if len(self.streams) >= 16:
                self.disable("conditioning stream limit reached")
                return None
            self.streams[key] = Stream(History(self.config))
        return self.streams[key]

    def forecast_mode(self, stream):
        """None for an actual call, else "bootstrap" or "regression"."""
        step = self.active_step
        if step is None or stream.last_step >= step:
            return None
        if step >= len(self.coordinates) - self.config.tail:
            return None
        if (self.config.bootstrap_active and step == 1 and step >= self.config.warmup
                and stream.last_actual == 0 and len(stream.history.features) == 1):
            return "bootstrap"
        if step < max(self.config.warmup, self.config.degree + 1):
            return None
        if len(stream.history.features) < self.config.degree + 1:
            return None
        window = min(4, self.config.window + math.floor(self.config.growth * stream.actual_anchors))
        return "regression" if step - stream.last_actual <= window else None

    def should_forecast(self, stream):
        return self.forecast_mode(stream) is not None

    def observe(self, stream, feature):
        budget = self.config.budget_mb * 1024 * 1024
        required = feature.numel() * feature.element_size()
        if required * (self.config.degree + 1) > budget:
            stream.history.clear()
            self.reason("history budget too small for the required actual anchors")
            return
        # Evict before retaining the new feature. All streams share one budget.
        while self.nbytes + required > budget:
            largest = max(self.streams.values(), key=lambda s: s.history.nbytes)
            if not largest.history.features:
                break
            largest.history.trim()
        stream.history.observe(self.coordinates[self.active_step], feature)
        stream.last_actual = self.active_step
        stream.actual_anchors += 1
        self.peak_history_bytes = max(self.peak_history_bytes, self.nbytes)

    def report(self):
        return {"actual": self.counts["actual"], "forecast": self.counts["forecast"],
                "denoiser_calls": self.counts["actual"] + self.counts["forecast"],
                "outer_steps": self.step, "peak_history_bytes": self.peak_history_bytes,
                "bootstrap_forecast": self.counts["bootstrap"],
                "pece_corrector_evaluations": self.counts["pece_correction"],
                "effective_config": {"preset": self.config.preset, "warmup": self.config.warmup,
                                     "tail": self.config.tail, "window": self.config.window,
                                     "degree": self.config.degree,
                                     "bootstrap": self.config.bootstrap_active,
                                     "extrapolation_guard_enabled": self.config.extrapolation_guard_enabled,
                                     "max_extrapolation_span_ratio": self.config.max_extrapolation_span_ratio},
                "fallback": self.disabled_reason, "actual_reasons": dict(self.reasons)}
