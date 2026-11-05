# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
"""History-weight formulation adapted from Spectrum MiniMax H3.

Only the small CPU regression system uses float64. Feature accumulation uses
bounded float32 chunks, with no persistent feature-space coefficient matrix.
"""
import math
import torch


class ForecastUnavailable(RuntimeError):
    pass


class ForecastNonfinite(ForecastUnavailable):
    """Prediction evidence is nonfinite and retained history must be discarded."""


def chebyshev(x, degree):
    x = torch.as_tensor(x, dtype=torch.float64, device="cpu").reshape(-1, 1)
    columns = [torch.ones_like(x)]
    if degree:
        columns.append(x)
    for _ in range(2, degree + 1):
        columns.append(2 * x * columns[-1] - columns[-2])
    return torch.cat(columns, dim=1)


class History:
    def __init__(self, config):
        self.config = config
        self.coordinates = []
        self.features = []
        self._factor = None
        self.chunk_elements = 1024 * 1024  # <=4 MiB per FP32 chunk

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.features)

    def clear(self):
        self.coordinates.clear()
        self.features.clear()
        self._factor = None

    def trim(self):
        if self.features:
            self.features.pop(0)
            self.coordinates.pop(0)
            self._factor = None

    def observe(self, coordinate, feature):
        if not math.isfinite(coordinate) or not -1 <= coordinate <= 1:
            raise ForecastUnavailable("invalid sampling coordinate")
        if self.coordinates and coordinate <= self.coordinates[-1]:
            raise ForecastUnavailable("history coordinates must increase strictly")
        if not feature.is_floating_point():
            raise ForecastUnavailable("history requires floating-point features")
        if self.features and (feature.shape != self.features[-1].shape or feature.dtype != self.features[-1].dtype):
            raise ForecastUnavailable("feature layout changed")
        # Ownership is taken only after a detached snapshot was made by the adapter.
        target = "cpu" if self.config.storage == "system_ram" else feature.device
        self.features.append(feature.detach().to(device=target))
        self.coordinates.append(float(coordinate))
        while len(self.features) > self.config.max_history:
            self.trim()
        self._factor = None

    def weights(self, coordinate):
        n = len(self.features)
        if n < self.config.degree + 1:
            raise ForecastUnavailable("insufficient actual history")
        if not math.isfinite(coordinate) or not self.coordinates[-1] < coordinate <= 1:
            raise ForecastUnavailable("prediction must be causal and inside the sampling domain")
        span = self.coordinates[-1] - self.coordinates[0]
        gap = coordinate - self.coordinates[-1]
        if span < 1e-10 or gap > 0.5 * span + 1e-12:
            raise ForecastUnavailable("extrapolation exceeds half the retained history span")
        if self._factor is None:
            design = chebyshev(self.coordinates, self.config.degree)
            gram = design.T @ design
            eye = torch.eye(self.config.degree + 1, dtype=torch.float64)
            # A fixed bounded ladder covers zero-ridge degenerate fits.
            factor = None
            for jitter in (0.0, 1e-12, 1e-10, 1e-8):
                factor, info = torch.linalg.cholesky_ex(gram + (self.config.ridge + jitter) * eye)
                if int(info) == 0:
                    break
            else:
                raise ForecastUnavailable("ridge factorization failed")
            self._factor = (design, factor)
        design, factor = self._factor
        weights = (chebyshev([coordinate], self.config.degree) @
                   torch.cholesky_solve(design.T, factor)).flatten()
        if not bool(torch.isfinite(weights).all()) or float(weights.abs().sum()) > 8:
            raise ForecastUnavailable("unstable regression weights")
        return weights

    def predict_hold(self, device, dtype):
        """One-point bootstrap: reuse the only actual feature at the next step.

        The caller projects it with the current timestep, so only the
        transformer features are held, not the model output.
        """
        if len(self.features) != 1:
            raise ForecastUnavailable("one-point bootstrap requires exactly one actual anchor")
        out = self.features[0].to(device=device, dtype=dtype, copy=True)
        if not bool(torch.isfinite(out).all()):
            raise ForecastNonfinite("nonfinite predicted features")
        return out

    def predict(self, coordinate, device, dtype):
        weights = self.weights(coordinate).tolist()
        shape = self.features[-1].shape
        out = torch.empty(shape, device=device, dtype=dtype)
        flat = out.view(-1)
        for start in range(0, flat.numel(), self.chunk_elements):
            end = min(start + self.chunk_elements, flat.numel())
            acc = torch.zeros(end - start, device=device, dtype=torch.float32)
            for weight, feature in zip(weights, self.features):
                acc.add_(feature.view(-1)[start:end].to(device=device, dtype=torch.float32), alpha=weight)
            flat[start:end].copy_(acc)
        # One explicit synchronization is required to keep nonfinite predictions
        # out of the sampler. This is also checked after native output projection.
        if not bool(torch.isfinite(out).all()):
            raise ForecastNonfinite("nonfinite predicted features")
        return out
