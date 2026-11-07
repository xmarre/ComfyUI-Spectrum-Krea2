# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
# Native output reconstruction adapted from ComfyUI, Copyright (C) 2024 Comfy.
"""Native Krea 2 capture and current-timestep projection, without a second DiT."""
import contextvars
import torch
from .compat import native_model, verified
from .forecast import ForecastUnavailable, ForecastNonfinite
from .runtime import CURRENT
from .qualified_patches import matches

CAPTURE = contextvars.ContextVar("spectrum_krea2_capture", default=None)


def call_contract(session, inner, x, context, ref_latents, options, kwargs):
    if inner is not session.model or not native_model(inner):
        return "model identity changed"
    if torch.is_grad_enabled():
        return "gradient-enabled execution"
    if not torch.is_tensor(x) or x.ndim not in (4, 5) or (x.ndim == 5 and x.shape[2] != 1):
        return "unsupported latent layout"
    if tuple(x.shape[-2:]) != session.spatial:
        return "spatial layout differs from the sampler latent"
    ids, branches = options.get("uuids"), options.get("cond_or_uncond")
    if not ids or not branches or len(ids) != len(branches) or len(set(ids)) != len(ids):
        return "missing or ambiguous conditioning identity"
    if x.shape[0] != session.batch * len(ids):
        return "conditioning batch packing changed"
    if not torch.is_tensor(context) or context.ndim != 3 or context.shape[0] != x.shape[0]:
        return "conditioning layout changed"
    if context.shape[2] != inner.txtlayers * inner.txtdim:
        return "conditioning encoder is incompatible"
    if not matches(session.patch_policy, options):
        return "external transformer modifications are active"
    if kwargs.get("control") is not None:
        return "control conditioning is active"
    unknown = set(kwargs) - {"control", "ref_latents_method"}
    if unknown:
        return "unverified model inputs: " + ", ".join(sorted(unknown))
    method = kwargs.get("ref_latents_method", inner.default_ref_method)
    if ref_latents and method not in {None, "index", "index_timestep_zero"}:
        return "unverified reference-image modulation"
    if inner.last._forward_hooks or inner.last._forward_pre_hooks:
        return "external or concurrent final-layer hooks are active"
    return None


def stream_key(x, context, refs, options, kwargs, inner):
    # ComfyUI UUIDs identify the immutable processed conditions for this invocation.
    # Shape, packing order, dtype/device and reference layout complete that identity.
    layout = lambda t: (tuple(t.shape), t.dtype, t.device)
    return (tuple(options["uuids"]), tuple(options["cond_or_uncond"]), layout(x), layout(context),
            tuple(layout(r) for r in refs or []), kwargs.get("ref_latents_method", inner.default_ref_method))


def project(inner, feature, timesteps, x):
    from comfy.ldm.flux.layers import timestep_embedding
    if not verified(timestep_embedding, "native.timestep_embedding"):
        raise ForecastUnavailable("native timestep embedding contract changed")
    # The native LastLayer is tokenwise: dropping text/reference rows before it
    # preserves image-row math and avoids projecting tokens that are discarded.
    t = inner.tmlp(timestep_embedding(timesteps, inner.tdim).unsqueeze(1).to(feature.dtype))
    out = inner.last(feature, t)
    patch = inner.patch
    h, w = x.shape[-2:]
    hp, wp = (h + patch - 1) // patch, (w + patch - 1) // patch
    b = x.shape[0]
    out = out.reshape(b, hp, wp, inner.channels, patch, patch)
    out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, inner.channels, hp * patch, wp * patch)
    out = out[:, :, :h, :w]
    if x.ndim == 5:
        out = out.unsqueeze(2)
    if not bool(torch.isfinite(out).all()):
        raise ForecastNonfinite("nonfinite native projection of predicted features")
    return out


def diffusion(binding, executor, x, timesteps, context, attention_mask=None,
              ref_latents=None, transformer_options=None, **kwargs):
    options = transformer_options if transformer_options is not None else {}
    session = CURRENT.get()
    def native():
        result = executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
        if session is not None and session.binding is binding:
            session.counts["actual"] += 1
        return result
    if session is None or session.binding is not binding:
        return native()
    inner = executor.class_obj
    if session.disabled_reason or session.active_step is None:
        return native()
    reason = call_contract(session, inner, x, context, ref_latents, options, kwargs)
    # Weight identity is read from the active native BaseModel, never a stale clone.
    # The parent is supplied by the sampler rather than modifying the model object.
    if session.weight_owner.current_weight_patches_uuid != session.weight_identity:
        reason = "active weight patches changed"
    if reason:
        session.reason(reason)
        session.streams.clear()
        return native()
    key = stream_key(x, context, ref_latents, options, kwargs, inner)
    stream = session.stream(key)
    if stream is None:
        return native()
    step = session.active_step
    policy = session.patch_policy
    phase = policy.phase(x, context, ref_latents, inner) if policy is not None else "native"
    if phase is None:
        session.disable("qualified Sol-Krea2 sampling scope was lost")
        return native()
    planned_transition = stream.route is not None and stream.route != phase
    if planned_transition:
        stream.history.clear()
        stream.last_actual = -1
        stream.actual_anchors = 0
        session.reason("numerical attention backend transition")
    if stream.last_step >= step:
        stream.history.clear()
        session.reason("repeated branch evaluation at one outer step")
    mode = session.forecast_mode(stream)
    if mode is not None:
        prediction = None
        try:
            dtype = stream.history.features[-1].dtype
            if mode == "bootstrap":
                prediction = stream.history.predict_hold(x.device, dtype)
            else:
                prediction = stream.history.predict(session.coordinates[step], x.device, dtype)
            output = project(inner, prediction, timesteps, x)
        except (ForecastUnavailable, RuntimeError) as exc:
            # Retry only forecast-owned work. Errors from the native denoiser
            # below are allowed to propagate unchanged.
            session.reason("forecast rejected: " + str(exc))
            prediction = None
            # A horizon/fit precondition miss does not invalidate real anchors.
            # Keeping them lets the next actual call build enough fit span.
            if not isinstance(exc, ForecastUnavailable) or isinstance(exc, ForecastNonfinite):
                stream.history.clear()
        else:
            stream.last_step = step
            session.counts["forecast"] += 1
            if mode == "bootstrap":
                session.counts["bootstrap"] += 1
            return output

    capture_id = object()
    captured = []
    capture_error = []
    image_tokens = ((x.shape[-2] + inner.patch - 1) // inner.patch) * ((x.shape[-1] + inner.patch - 1) // inner.patch)
    text_tokens = context.shape[1]

    def capture(module, args):
        if CAPTURE.get() is not capture_id:
            return
        try:
            feature = args[0][:, text_tokens:text_tokens + image_tokens]
            # No alias to the joint sequence or native live tensor survives.
            captured.append(feature.detach().clone(memory_format=torch.contiguous_format))
        except RuntimeError as exc:
            capture_error.append(str(exc))
        return None

    before_sol = policy.sol_status() if policy is not None else None
    token = CAPTURE.set(capture_id)
    handle = None
    try:
        handle = inner.last.register_forward_pre_hook(capture)
        output = native()
    finally:
        if handle is not None:
            handle.remove()
        CAPTURE.reset(token)
    stream.last_step = step
    if before_sol is not None:
        after_sol = policy.sol_status()
        if (after_sol is None or
                after_sol["actual_evaluations"] != before_sol["actual_evaluations"] + 1):
            session.disable("Sol-Krea2 native execution was not witnessed")
            stream.history.clear()
            return output
        realized = "sol" if after_sol["sparse_calls"] > before_sol["sparse_calls"] else "dense"
    else:
        realized = "native"
    # The preflight already invalidated history when it predicted a route
    # change. Compare a witnessed actual against that prediction; do not
    # invalidate or report the same dense-to-sparse transition twice.
    if realized != phase or (
            not planned_transition and stream.route is not None and stream.route != realized):
        stream.history.clear()
        stream.last_actual = -1
        stream.actual_anchors = 0
        session.reason("numerical attention route differed from the previous anchor")
    stream.route = realized
    if len(captured) == 1 and not capture_error:
        try:
            session.observe(stream, captured[0])
        except (ForecastUnavailable, RuntimeError) as exc:
            session.reason("history capture rejected: " + str(exc))
            stream.history.clear()
    else:
        session.reason("native final-layer capture unavailable")
        stream.history.clear()
    return output
