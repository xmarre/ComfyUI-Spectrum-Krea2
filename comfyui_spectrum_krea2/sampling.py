# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
import json
import logging
import math
import torch
from .compat import verified, model_contract
from .runtime import CURRENT, REPORTS, Session

KEY = "spectrum_krea2"
LOG = logging.getLogger(__name__)


def foreign_wrappers(options):
    for groups in options.get("wrappers", {}).values():
        for key, wrappers in groups.items():
            if key != KEY and wrappers:
                return True
    return False


def validate_sampler(executor, model_wrap, sigmas, extra_args):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    sampler = executor.class_obj
    if type(sampler) is not KSAMPLER or not verified(type(sampler).sample, "sampler.sample"):
        return "sampler execution contract is unsupported"
    function = sampler.sampler_function
    allowed = (sampling.sample_euler, sampling.sample_dpmpp_2m)
    if function not in allowed or not verified(function, "native." + function.__name__):
        return "only audited native Euler and DPM++ 2M are forecast-enabled"
    if function is sampling.sample_euler and sampler.extra_options.get("s_churn", 0) != 0:
        return "Euler churn changes the denoiser coordinate"
    if any(k not in {"s_churn", "s_tmin", "s_tmax", "s_noise"} for k in sampler.extra_options):
        return "unverified sampler options"
    options = extra_args.get("model_options", {})
    trans = options.get("transformer_options", {})
    if foreign_wrappers(trans) or trans.get("patches") or trans.get("patches_replace"):
        return "external transformer patches or wrappers are active"
    if any(options.get(k) for k in ("model_function_wrapper", "sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function", "multigpu_thread_pool")):
        return "custom CFG, model wrapper or multi-device execution is active"
    for conditions in model_wrap.conds.values():
        for cond in conditions or []:
            if any(cond.get(k) is not None for k in ("hooks", "control", "area", "mask", "gligen")):
                return "scheduled hooks, control or regional conditioning is active"
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return "sigma schedule must be a one-dimensional tensor"
    return None


class Binding:
    """Immutable installation identity; all mutable sampling state is invocation-local."""
    def __init__(self, config):
        self.config = config.resolved()

    def sampler(self, executor, model_wrap, sigmas, extra_args, callback, noise,
                latent_image=None, denoise_mask=None, disable_pbar=False):
        reason = validate_sampler(executor, model_wrap, sigmas, extra_args)
        inner = model_wrap.inner_model.diffusion_model
        if not model_contract(inner):
            reason = reason or "native Krea 2 source contract changed"
        values = sigmas.detach().to(device="cpu", dtype=torch.float64).tolist() if torch.is_tensor(sigmas) and sigmas.ndim == 1 else []
        if (len(values) < 2 or any(not math.isfinite(v) for v in values) or
                values[-1] < 0 or any(a <= b for a, b in zip(values, values[1:]))):
            reason = reason or "sigma schedule is empty, nonfinite or not strictly decreasing"
        if not torch.is_tensor(noise) or noise.ndim not in (4, 5) or (noise.ndim == 5 and noise.shape[2] != 1):
            reason = reason or "only image latents and single-frame 5D latents are forecast-enabled"
        coordinates = [] if reason else [2 * (values[0] - s) / (values[0] - values[-1]) - 1 for s in values[:-1]]
        session = Session(self, self.config, coordinates, inner,
                          getattr(model_wrap.inner_model, "current_weight_patches_uuid", None),
                          noise.shape[0], tuple(noise.shape[-2:]))
        session.weight_owner = model_wrap.inner_model
        if reason:
            session.disable(reason)
        token = CURRENT.set(session)

        def on_step(step, x0, x, total):
            if session.predict_calls != 1 or step != session.step or total != len(values) - 1:
                session.disable("sampler callback and denoiser opportunity ordering changed")
            session.step += 1
            session.predict_calls = 0
            if callback is not None:
                callback(step, x0, x, total)

        try:
            return executor(model_wrap, sigmas, extra_args, on_step, noise,
                            latent_image, denoise_mask, disable_pbar)
        finally:
            report = session.report()
            session.streams.clear()
            CURRENT.reset(token)
            sink = REPORTS.get()
            if sink is not None:
                sink.append(report)
            LOG.info("Spectrum Krea 2 summary: %s", json.dumps(report, sort_keys=True))

    def predict(self, executor, x, timestep, model_options=None, seed=None):
        session = CURRENT.get()
        if session is None or session.binding is not self:
            return executor(x, timestep, model_options or {}, seed)
        session.predict_calls += 1
        if session.predict_calls != 1 or session.step >= len(session.coordinates):
            session.disable("repeated or off-schedule denoiser opportunity")
        previous = session.active_step
        session.active_step = session.step
        try:
            return executor(x, timestep, model_options or {}, seed)
        finally:
            session.active_step = previous
