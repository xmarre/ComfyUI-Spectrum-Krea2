# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
# SA-Solver equations adapted from ComfyUI, Copyright (C) 2024 Comfy.
"""Forecast-isolated native SA-Solver for Spectrum Krea 2.

Native SA-Solver adds a fresh Gaussian increment to its predictor state inside
the SDE window. An actual Krea 2 evaluation sees that increment and removes it.
A Spectrum forecast extrapolates transformer features from earlier steps and
cannot see it. The denoised value ``x - sigma * v`` then keeps the increment,
and the following steps carry it to the output as per-latent-cell colored
speckles ("confetti").

This adapter runs the reviewed native SA equations, the native coefficient
helpers, and the native noise sampler, with two ownership rules:

* Only actual denoiser results are persistent Adams history. A forecast is an
  ephemeral input to the current corrector and predictor only, so a forecast
  error is never reused by later stencils.
* When a forecast step's input carries a fresh stochastic increment, SA consumes
  a causal solver-space denoised value built only from actual anchors instead of
  the raw forecast: the latest anchor, or a bounded half-log-SNR extrapolation
  from two non-consecutive anchors.

Without forecasts the update is the native one, operation for operation.
"""
import torch


def causal_denoised(values, lambdas, steps, target_lambda, allow_consecutive=False,
                    soft_guard=False):
    """Predict SA's denoised variable from actual anchors only.

    Returns ``(value, mode)`` with mode ``hold`` (one anchor, or consecutive
    anchors unless ``allow_consecutive``: active PECE endpoints are exact), ``guard_hold`` (untrusted geometry or more than one anchor
    interval ahead) or ``extrapolation``. With opt-in ``soft_guard``,
    overlong forward steps retain a continuously attenuated, bounded secant
    instead of discontinuously dropping to zero. The guard reads one scalar.
    """
    if not values or not len(values) == len(lambdas) == len(steps):
        raise RuntimeError("stochastic SA dense output requires aligned actual anchors")
    latest = values[-1]
    if len(values) == 1 or (steps[-1] - steps[-2] == 1 and not allow_consecutive):
        return latest.clone(memory_format=torch.contiguous_format), "hold"
    previous = values[-2]
    if previous.shape != latest.shape or previous.dtype != latest.dtype or previous.device != latest.device:
        raise RuntimeError("stochastic SA dense-output anchors are incompatible")
    denominator = lambdas[-1] - lambdas[-2]
    advance = target_lambda - lambdas[-1]
    alpha = advance / denominator
    scale = torch.maximum(torch.maximum(lambdas[-2].abs(), lambdas[-1].abs()),
                          torch.ones_like(lambdas[-1]))
    valid = (torch.isfinite(alpha) & torch.isfinite(denominator)
             & (denominator.abs() > 1e-12 * scale) & (denominator * advance >= 0.0)
             & (alpha >= -1e-6))
    if not bool(valid) or (not soft_guard and bool(alpha > 1.0 + 1e-6)):
        return latest.clone(memory_format=torch.contiguous_format), "guard_hold"
    # The ordinary hard guard jumps from a near-full secant at alpha≈1 to
    # an exact hold at alpha>1. The opt-in detail mode replaces that abrupt
    # cutoff with w=1/alpha beyond the trustworthy horizon: continuous at
    # alpha=1, strictly bounded by one exact corrected-anchor increment,
    # and tending smoothly to a hold for very long extrapolation distances.
    # It never introduces the current stochastic predictor input or its
    # unobserved Gaussian increment into the solver-facing output.
    if soft_guard and bool(alpha > 1.0):
        weight = alpha.reciprocal().to(device=latest.device, dtype=latest.dtype)
        mode = "soft_extrapolation"
    else:
        weight = alpha.clamp(0.0, 1.0).to(device=latest.device, dtype=latest.dtype)
        mode = "extrapolation"
    return torch.lerp(latest, previous, -weight), mode


def sample_sa_solver_isolated(session, model, x, sigmas, extra_args=None, callback=None,
                              disable=False, tau_func=None, s_noise=1.0, noise_sampler=None,
                              predictor_order=3, corrector_order=4, use_pece=False,
                              simple_order_2=False):
    """Native ``sample_sa_solver`` with forecast-isolated Adams history."""
    from comfy.k_diffusion import sampling, sa_solver
    if len(sigmas) <= 1:
        return x
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = sampling.default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    s_noise = s_noise * getattr(model_sampling, "noise_scale", 1.0)
    sigmas = sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
    lambdas = sampling.sigma_to_half_log_snr(sigmas, model_sampling=model_sampling)

    if tau_func is None:
        start_sigma = model_sampling.percent_to_sigma(0.2)
        end_sigma = model_sampling.percent_to_sigma(0.8)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=1.0)

    max_used_order = max(predictor_order, corrector_order)
    x_pred = x
    h = 0.0
    tau_t = 0.0
    noise = 0.0
    # Persistent history: actual denoiser results (and exact PECE corrections).
    actual_preds, actual_lambdas, actual_steps = [], [], []
    lower_order_to_end = sigmas[-1].item() == 0

    # Active-PECE acceleration (opt-in): after a protected prefix, the
    # predicted phase skips the model. Its value only feeds this step's
    # corrector, and the exact corrected evaluation re-anchors the trajectory.
    from .config import SA_PECE_POLICIES
    prefix = SA_PECE_POLICIES[session.config.sa_pece_forecast_policy]
    accelerated = bool(use_pece and corrector_order > 0 and prefix is not None)

    for i in sampling.trange(len(sigmas) - 1, disable=disable):
        session.served_forecast = False
        session.solver_skip = accelerated and i >= max(prefix, 1) and bool(actual_preds)
        try:
            denoised = model(x_pred, sigmas[i] * s_in, **extra_args)
        finally:
            session.solver_skip = False
        forecast = session.served_forecast
        if forecast:
            session.counts["sa_ephemeral_forecast"] += 1
            if actual_preds and (accelerated or (tau_t > 0 and s_noise > 0)):
                # Stochastic input: x_pred holds a fresh increment the forecast
                # could not see. Accelerated PECE: the solver owns this value.
                denoised, mode = causal_denoised(actual_preds, actual_lambdas, actual_steps, lambdas[i],
                                                 allow_consecutive=accelerated,
                                                 soft_guard=(session.config.sa_pece_forecast_policy == "balanced_detail"))
                session.counts["sa_dense_output"] += 1
                session.counts["sa_dense_" + mode] += 1
                session.sa_dense_steps.append({"step": i, "sigma": round(float(sigmas[i]), 6),
                                               "mode": mode, "anchors": list(actual_steps[-2:])})
        if callback is not None:
            callback({"x": x_pred, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if not forecast:
            actual_preds.append(denoised)
            actual_lambdas.append(lambdas[i])
            actual_steps.append(i)
            del actual_preds[:-max_used_order], actual_lambdas[:-max_used_order], actual_steps[:-max_used_order]

        stencil_preds = list(actual_preds)
        stencil_lambdas = list(actual_lambdas)
        if forecast:
            stencil_preds.append(denoised)
            stencil_lambdas.append(lambdas[i])
        stencil_preds = stencil_preds[-max_used_order:]
        stencil_lambdas = stencil_lambdas[-max_used_order:]

        predictor_order_used = min(predictor_order, len(stencil_preds))
        if i == 0 or (sigmas[i + 1] == 0 and not use_pece):
            corrector_order_used = 0
        else:
            corrector_order_used = min(corrector_order, len(stencil_preds))

        if lower_order_to_end:
            predictor_order_used = min(predictor_order_used, len(sigmas) - 2 - i)
            corrector_order_used = min(corrector_order_used, len(sigmas) - 1 - i)

        # Corrector
        if corrector_order_used == 0:
            x = x_pred
        else:
            curr_lambdas = torch.stack(stencil_lambdas[-corrector_order_used:])
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i], curr_lambdas, lambdas[i - 1], lambdas[i], tau_t,
                simple_order_2, is_corrector_step=True,
            )
            pred_mat = torch.stack(stencil_preds[-corrector_order_used:], dim=1)
            corr_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            x = sigmas[i] / sigmas[i - 1] * (-(tau_t ** 2) * h).exp() * x + corr_res

            if tau_t > 0 and s_noise > 0:
                x = x + noise

            if use_pece:
                # The native corrected-state evaluation is never forecast; it
                # replaces this step's value in the persistent history.
                denoised = model(x, sigmas[i] * s_in, **extra_args)
                if forecast:
                    actual_preds.append(denoised)
                    actual_lambdas.append(lambdas[i])
                    actual_steps.append(i)
                    del (actual_preds[:-max_used_order], actual_lambdas[:-max_used_order],
                         actual_steps[:-max_used_order])
                else:
                    actual_preds[-1] = denoised
                stencil_preds = list(actual_preds)
                stencil_lambdas = list(actual_lambdas)

        # Predictor
        if sigmas[i + 1] == 0:
            x_pred = denoised
        else:
            tau_t = tau_func(sigmas[i + 1])
            curr_lambdas = torch.stack(stencil_lambdas[-predictor_order_used:])
            b_coeffs = sa_solver.compute_stochastic_adams_b_coeffs(
                sigmas[i + 1], curr_lambdas, lambdas[i], lambdas[i + 1], tau_t,
                simple_order_2, is_corrector_step=False,
            )
            pred_mat = torch.stack(stencil_preds[-predictor_order_used:], dim=1)
            pred_res = torch.tensordot(pred_mat, b_coeffs, dims=([1], [0]))
            h = lambdas[i + 1] - lambdas[i]
            x_pred = sigmas[i + 1] / sigmas[i] * (-(tau_t ** 2) * h).exp() * x + pred_res

            if tau_t > 0 and s_noise > 0:
                noise = noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * tau_t ** 2 * h).expm1().neg().sqrt() * s_noise
                x_pred = x_pred + noise
    return x_pred
