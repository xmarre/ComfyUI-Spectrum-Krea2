# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
import json
import inspect
import logging
import math
import torch
from .compat import verified, model_contract
from .runtime import CURRENT, REPORTS, FORCE_NATIVE, Session

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
    if type(sampler) is not KSAMPLER:
        return f"unsupported sampler container: {type(sampler).__module__}.{type(sampler).__name__}"
    # The native container still executes on every call. Its logging/callback
    # implementation varies across ComfyUI releases without changing the solver.
    # Validate the bound entrypoint here and the actual solver function below.
    # ComfyUI invokes this container positionally. Parameter spelling, optional
    # trailing arguments and decorator signatures do not define solver semantics.
    method = executor.original
    if getattr(method, "__self__", None) is not sampler or getattr(method, "__func__", None) is not KSAMPLER.sample:
        return "sampler entrypoint has been replaced outside native KSAMPLER"
    function = sampler.sampler_function
    # Only native, source-audited one-prediction-per-step implementations
    # may share Spectrum's one-outer-step/one-callback history contract.
    # ER-SDE additionally injects stochastic noise and offsets flow sigma=1;
    # those steps still make exactly one denoiser call and one callback.
    # RES requires auditing both its entrypoint and its
    # delegated native solver; eta=0 and cfg_pp=False are essential.
    # DEIS additionally uses a precomputed coefficient table. Audit that
    # delegate and its numerical helpers rather than only its wrapper.
    allowed = (
        sampling.sample_euler,
        sampling.sample_dpmpp_2m,
        getattr(sampling, "sample_res_multistep", None),
        getattr(sampling, "sample_gradient_estimation", None),
        getattr(sampling, "sample_deis", None),
        getattr(sampling, "sample_ipndm", None),
        getattr(sampling, "sample_ipndm_v", None),
        getattr(sampling, "sample_er_sde", None),
        getattr(sampling, "sample_sa_solver", None),
        getattr(sampling, "sample_sa_solver_pece", None),
    )
    if function not in allowed:
        return "only audited native Euler, DPM++ 2M, RES Multistep, Gradient Estimation, DEIS, iPNDM and ER-SDE solvers are forecast-enabled"
    if not verified(function, "native." + function.__name__):
        return "native solver source contract changed: " + function.__name__
    if (function is getattr(sampling, "sample_res_multistep", None)
            and not verified(getattr(sampling, "res_multistep", None), "native.res_multistep")):
        return "native RES Multistep solver source contract changed"
    if function is getattr(sampling, "sample_er_sde", None):
        if any(not verified(getattr(sampling, name, None), "native." + name)
               for name in ("sigma_to_half_log_snr", "offset_first_sigma_for_snr")):
            return "native ER-SDE sigma conversion source contract changed"
    if function in (getattr(sampling, "sample_sa_solver", None),
                    getattr(sampling, "sample_sa_solver_pece", None)):
        if any(not verified(getattr(sampling, name, None), "native." + name)
               for name in ("sigma_to_half_log_snr", "offset_first_sigma_for_snr",
                            "sample_sa_solver", "sample_sa_solver_pece")):
            return "native SA-Solver or sigma source contract changed"
        from comfy.k_diffusion import sa_solver
        if any(not verified(getattr(sa_solver, name, None), "sa_solver." + name)
               for name in ("compute_stochastic_adams_b_coeffs",
                            "compute_simple_stochastic_adams_b_coeffs",
                            "compute_exponential_coeffs", "get_tau_interval_func")):
            return "native SA-Solver coefficient or tau source contract changed"
    if function is getattr(sampling, "sample_deis", None):
        helper = getattr(sampling, "deis", None)
        if helper is None or any(
            not verified(getattr(helper, name, None), "deis." + name)
            for name in ("get_deis_coeff_list", "edm2t", "cal_poly",
                         "cal_intergrand", "t2alpha_fn")
        ):
            return "native DEIS coefficient source contract changed"
    # A caller-supplied override of the deterministic native options could
    # require different coordinates, introduce CFG++ side channels, or alter
    # the solver's history. Accept no overrides for these solvers.
    if function is sampling.sample_euler:
        if sampler.extra_options.get("s_churn", 0) != 0:
            return "Euler churn changes the denoiser coordinate"
        if any(k not in {"s_churn", "s_tmin", "s_tmax", "s_noise"} for k in sampler.extra_options):
            return "unverified sampler options"
    elif function is getattr(sampling, "sample_er_sde", None):
        er_options = sampler.extra_options
        if any(name not in {"s_noise", "max_stage", "noise_scaler", "noise_sampler"}
               for name in er_options):
            return "unverified sampler options"
        max_stage = er_options.get("max_stage", 3)
        s_noise = er_options.get("s_noise", 1.0)
        if (isinstance(max_stage, bool) or not isinstance(max_stage, int)
                or max_stage not in (1, 2, 3)
                or isinstance(s_noise, bool) or not isinstance(s_noise, (int, float))
                or not math.isfinite(s_noise) or s_noise < 0):
            return "unverified ER-SDE options"
        if (("noise_scaler" in er_options and
             not callable(er_options["noise_scaler"])) or
                ("noise_sampler" in er_options and
                 not callable(er_options["noise_sampler"]))):
            return "unverified ER-SDE options"
    elif function in (getattr(sampling, "sample_sa_solver", None),
                       getattr(sampling, "sample_sa_solver_pece", None)):
        sa_options = sampler.extra_options
        keys = {"tau_func", "s_noise", "noise_sampler", "predictor_order",
                "corrector_order", "simple_order_2"}
        is_pece_alias = function is getattr(sampling, "sample_sa_solver_pece", None)
        if not is_pece_alias:
            keys.add("use_pece")
        if set(sa_options) - keys:
            return "unverified SA-Solver options"
        predictor = sa_options.get("predictor_order", 3)
        corrector = sa_options.get("corrector_order", 4)
        s_noise = sa_options.get("s_noise", 1.0)
        if (type(predictor) is not int or not 1 <= predictor <= 6 or
                type(corrector) is not int or not 0 <= corrector <= 6 or
                isinstance(s_noise, bool) or not isinstance(s_noise, (int, float)) or
                not math.isfinite(s_noise) or not 0 <= s_noise <= 100 or
                type(sa_options.get("simple_order_2", False)) is not bool or
                (not is_pece_alias and type(sa_options.get("use_pece", False)) is not bool) or
                ("tau_func" in sa_options and not callable(sa_options["tau_func"])) or
                ("noise_sampler" in sa_options and not callable(sa_options["noise_sampler"]))):
            return "invalid SA-Solver options"
    elif sampler.extra_options:
        return "unverified sampler options"
    options = extra_args.get("model_options", {})
    trans = options.get("transformer_options", {})
    from .qualified_patches import qualify, explain_rejection
    patch_policy, patch_reason = qualify(options)
    if patch_reason:
        return explain_rejection(options, patch_reason)
    if (options.get("model_function_wrapper") is not None and
            patch_policy.diffaid_patch is None and patch_policy.diffaid_wrapper is None):
        return "custom CFG, model wrapper or multi-device execution is active"
    if any(options.get(k) for k in ("sampler_cfg_function", "sampler_pre_cfg_function", "sampler_post_cfg_function", "multigpu_thread_pool")):
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
        if self.config.debug:
            try:
                signature = str(inspect.signature(executor.original))
            except (TypeError, ValueError):
                signature = "<unavailable>"
            container = type(executor.class_obj)
            LOG.info("Spectrum Krea 2 sampler: container=%s.%s signature=%s solver=%s",
                     container.__module__, container.__name__, signature,
                     getattr(getattr(executor.class_obj, "sampler_function", None), "__name__", "<unknown>"))
        from .qualified_patches import CONTRACT_REVISION
        if self.config.debug:
            LOG.info("Spectrum Krea 2 compatibility contract: %s", CONTRACT_REVISION)
        reason = validate_sampler(executor, model_wrap, sigmas, extra_args)
        if denoise_mask is not None:
            reason = reason or "latent inpaint masks are not forecast-enabled"
        inner = model_wrap.inner_model.diffusion_model
        if not model_contract(inner):
            reason = reason or "native Krea 2 source contract changed"
        effective_sigmas = sigmas
        # Native ER-SDE evaluates the first CONST timestep at a slightly
        # reduced sigma to avoid infinite log-SNR. Forecast history must use
        # the actual denoiser coordinates while preserving the original
        # sampler's noise scaling and stochastic update.
        if reason is None and executor.class_obj.sampler_function.__name__ in (
                "sample_er_sde", "sample_sa_solver", "sample_sa_solver_pece"):
            from comfy.k_diffusion import sampling
            model_sampling = model_wrap.model_patcher.get_model_object("model_sampling")
            effective_sigmas = sampling.offset_first_sigma_for_snr(sigmas, model_sampling)
        values = effective_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist() if torch.is_tensor(effective_sigmas) and effective_sigmas.ndim == 1 else []
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
        sa_options = executor.class_obj.extra_options
        is_sa = executor.class_obj.sampler_function.__name__ in (
            "sample_sa_solver", "sample_sa_solver_pece"
        )
        # An unrelated nested Euler/ER-SDE sampler in the same context must
        # retain its own forecast policy; only SA scopes opt into native-only.
        session.native_only = is_sa and bool(FORCE_NATIVE.get())
        session.pece_enabled = (is_sa and
            (executor.class_obj.sampler_function.__name__ == "sample_sa_solver_pece" or
             sa_options.get("use_pece", False)))
        session.pece_corrector_order = (
            sa_options.get("corrector_order", 4) if session.pece_enabled else 0
        )
        session.pece_sigmas = tuple(values[:-1]) if session.pece_enabled else ()
        from .qualified_patches import qualify
        session.patch_policy, _ = qualify(extra_args.get("model_options", {}))
        if reason:
            session.disable(reason)
        token = CURRENT.set(session)

        def on_step(step, x0, x, total):
            if session.predict_calls != 1 or step != session.step or total != len(values) - 1:
                session.disable("sampler callback and denoiser opportunity ordering changed")
            if session.pece_pending_step is not None:
                session.disable("PECE correction evaluation was not observed before next callback")
            session.pece_pending_step = (
                step if session.pece_enabled and session.pece_corrector_order > 0
                and step > 0 else None
            )
            session.step += 1
            session.predict_calls = 0
            if callback is not None:
                # User callbacks may invoke models or nested samplers. They are
                # outside this trajectory's denoiser opportunity.
                callback_token = CURRENT.set(None)
                try:
                    callback(step, x0, x, total)
                finally:
                    CURRENT.reset(callback_token)

        try:
            return executor(model_wrap, sigmas, extra_args, on_step, noise,
                            latent_image, denoise_mask, disable_pbar)
        finally:
            if session.pece_pending_step is not None and session.disabled_reason is None:
                session.disable("PECE correction evaluation missing at sampler completion")
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
        if session.pece_pending_step is not None:
            step = session.pece_pending_step
            target = session.pece_sigmas[step]
            same_sigma = (
                torch.is_tensor(timestep) and timestep.numel() > 0
                and bool(torch.isfinite(timestep).all())
                and bool(torch.allclose(
                    timestep, torch.full_like(timestep, target), rtol=1e-5, atol=1e-6
                ))
            )
            session.pece_pending_step = None
            if same_sigma:
                # PECE corrects the *same* outer step after its callback.
                # Always evaluate the native denoiser: neither a predictor
                # forecast nor its history may substitute this correction.
                # Keep CURRENT so actual evaluation counters remain accurate.
                prior = session.active_step
                session.active_step = None
                session.counts["pece_correction"] += 1
                actual_before = session.counts["actual"]
                try:
                    return executor(x, timestep, model_options or {}, seed)
                finally:
                    session.counts["pece_correction_native"] += (
                        session.counts["actual"] - actual_before
                    )
                    session.active_step = prior
            session.disable("PECE correction sigma/order mismatch")
        session.predict_calls += 1
        if session.predict_calls != 1 or session.step >= len(session.coordinates):
            session.disable("repeated or off-schedule denoiser opportunity")
        previous = session.active_step
        session.active_step = session.step
        try:
            return executor(x, timestep, model_options or {}, seed)
        finally:
            session.active_step = previous
