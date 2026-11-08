# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
"""Narrow, runtime-verified Krea 2 attention-patch interoperability.

Only the reviewed Sol-Krea2 and Krea2 Diff-Aid implementations are eligible.
Unknown patches, wrapper owners and model-function decorators fail closed.
"""
from dataclasses import dataclass
import math
import sys

import torch

CONTRACT_REVISION = "qualified-krea2-v2"
SOL_KEY = "sol_krea2_sparse_attention_v1"
SPECTRUM_KEY = "spectrum_krea2"
DIFFAID_WRAPPER = "diffaid_krea2_text_residual"
KJ_PREVIEW_KEY = "kj_preview_override"
# KJNodes Model Preview Override is an OUTER_SAMPLE observer, not a
# denoiser or sampler replacement. Once the native sampler wrapper is entered,
# validate its *effective* solver, patches, conditioning and per-step callback
# ordering. Neither KJNodes' release nor its callback implementation is part
# of that contract. The stable wrapper registration key identifies this
# optional observer; unknown keys and all denoiser wrappers remain excluded.


def _registered_type(value, name):
    cls = type(value)
    module = sys.modules.get(cls.__module__)
    return cls.__name__ == name and getattr(module, name, None) is cls


def _wrappers(options):
    found = options.get("wrappers", {})
    if not isinstance(found, dict):
        return None
    return found


def _entries(mapping):
    for kind, entries in mapping.items():
        if not isinstance(entries, dict):
            raise ValueError("malformed wrapper group")
        for key, value in entries.items():
            if not isinstance(value, (list, tuple)):
                raise ValueError("malformed wrapper list")
            yield kind, key, value


def _diffaid_window(config):
    """Validated (start, end, ramp), or None for an invalid window."""
    try:
        window = tuple(float(getattr(config, name))
                       for name in ("sigma_start", "sigma_end", "sigma_ramp"))
    except (AttributeError, TypeError, ValueError):
        return None
    start, end, ramp = window
    if (not all(math.isfinite(v) for v in window) or
            not 0.0 <= start <= end <= 1.0 or ramp < 0.0):
        return None
    return window


class DiffAidGainUnavailable(RuntimeError):
    """Diff-Aid's sigma-window gain for an evaluation cannot be reproduced."""


@dataclass(frozen=True)
class DiffAidWindow:
    """Reproduces Diff-Aid's own per-evaluation sigma-window gain.

    Diff-Aid multiplies its modulation by ``_sigma_window_gain`` of the
    normalized sigma that its SharedTimestepWrapper injected into the
    transformer options of the same evaluation. Inside one constant gain the
    patched operator is a fixed function, so forecast history is valid. A gain
    change is a discontinuous operator change, so the caller treats it as a
    regime boundary that requires fresh actual anchors.
    """
    owner: object
    config: object
    window: tuple
    state_key: str
    coerce: object
    gain: object

    def regime(self, options, x):
        if self.owner.config is not self.config or _diffaid_window(self.config) != self.window:
            raise DiffAidGainUnavailable("Diff-Aid sigma-window configuration changed")
        # Same lookup order as Diff-Aid's _compute_alpha.
        normalized = options.get("normalized_sigma", None)
        if normalized is None:
            state = options.get(self.state_key, {}) or {}
            normalized = state.get("normalized_sigma", None)
        if normalized is None:
            normalized = 1.0
        # Diff-Aid casts the normalized sigma to the dtype of the hidden state
        # it modulates before the float32 window comparison. Krea 2's text
        # residual and text K/V use the DiT input dtype. The rounding and the
        # comparison are device-independent, so evaluate them on the CPU.
        try:
            sigma = self.coerce(normalized, x.shape[0], "cpu", x.dtype)
            gain = self.gain(sigma, self.window[0], self.window[1], self.window[2])
        except (TypeError, ValueError, RuntimeError) as exc:
            raise DiffAidGainUnavailable("Diff-Aid sigma-window gain failed: " + str(exc)) from exc
        values = tuple(float(v) for v in gain.reshape(-1).tolist())
        if len(values) != x.shape[0] or not all(math.isfinite(v) for v in values):
            raise DiffAidGainUnavailable("Diff-Aid sigma-window gain is malformed")
        return values


def _diffaid_gain_contract(owner, config, window):
    """Bind Diff-Aid's own gain helpers from the module of its verified owner."""
    module = sys.modules.get(type(owner).__module__)
    state_key = getattr(module, "STATE_KEY", None)
    coerce = getattr(module, "_coerce_batch_value", None)
    gain = getattr(module, "_sigma_window_gain", None)
    if type(state_key) is not str or not callable(coerce) or not callable(gain):
        return None
    return DiffAidWindow(owner, config, window, state_key, coerce, gain)


@dataclass(frozen=True)
class QualifiedPatches:
    sol_override: object = None
    diffaid_patch: object = None
    diffaid_wrapper: object = None
    kj_preview_wrapper: object = None
    kj_preview_method: object = None
    kj_preview_code: object = None
    verified_wrappers: tuple = ()
    diffaid_window: DiffAidWindow | None = None

    def diffaid_regime(self, options, x):
        """Diff-Aid gain of the NEXT evaluation; None while it cannot change."""
        if self.diffaid_window is None:
            return None
        return self.diffaid_window.regime(options, x)

    def sol_status(self):
        if self.sol_override is None:
            return None
        from sol_h3.krea2 import spectrum_runtime_status
        return spectrum_runtime_status(self.sol_override)

    def phase(self, x, context, refs, model):
        """Predict the route of the NEXT real Krea model evaluation."""
        if self.sol_override is None:
            return "native"
        status = self.sol_status()
        if status is None:
            return None
        settings = status["settings"]
        if status["actual_evaluations"] + 1 <= settings.dense_evaluations:
            return "dense"
        if (refs or status["kernel_unavailable"] or
                getattr(self.sol_override, "_sol_krea2_preprocess", False)):
            return "dense"
        # Krea2's native declared DiT compute dtype drives the route;
        # the latent dtype alone does not identify QKV precision.
        if x.device.type != "cuda" or x.shape[0] != 1:
            return "dense"
        dtype = getattr(model, "dtype", None)
        if dtype is torch.float16:
            from sol_h3 import krea2 as sol
            probe = getattr(sol, "fp16_backend_available", None)
            if not callable(probe) or not probe(x.device):
                return "dense"
        elif dtype is not torch.bfloat16:
            return "dense"
        tokens = context.shape[1]
        tokens += ((x.shape[-2] + model.patch - 1) // model.patch) * (
            (x.shape[-1] + model.patch - 1) // model.patch)
        if tokens < settings.min_tokens:
            return "dense"
        return "sol"


def _topology(model_options):
    """Bounded structural identifiers only; no tensor values, paths or prompts."""
    trans = model_options.get("transformer_options", {})
    if not isinstance(trans, dict):
        return "transformer_options=invalid"
    patches = trans.get("patches") or {}
    if isinstance(patches, dict):
        patch_names = [str(k)[:64] for k in patches]
    else:
        patch_names = ["<malformed>"]
    replacement = trans.get("patches_replace") or {}
    replace_names = [str(k)[:64] for k in replacement] if isinstance(replacement, dict) else ["<malformed>"]
    wraps = trans.get("wrappers") or {}
    wrappers = []
    if isinstance(wraps, dict):
        for kind, owners in wraps.items():
            if isinstance(owners, dict):
                for key, callbacks in owners.items():
                    if key != SPECTRUM_KEY:
                        wrappers.append(str(kind)[:32] + "." + str(key)[:64] +
                                        ("["+str(len(callbacks))+"]" if isinstance(callbacks, (tuple, list)) else "[malformed]"))
            else:
                wrappers.append(str(kind)[:32] + ".[malformed]")
    else:
        wrappers.append("<malformed>")
    override = trans.get("optimized_attention_override")
    fn = model_options.get("model_function_wrapper")
    owner = type(fn).__name__ if fn is not None else "none"
    return ("patches=" + ",".join(sorted(patch_names)[:12]) +
            "; replace=" + ",".join(sorted(replace_names)[:12]) +
            "; wrappers=" + ",".join(sorted(wrappers)[:16]) +
            "; sol_marker=" + str(SOL_KEY in trans) +
            "; sol_owner=" + str(getattr(override, "_sol_krea2_owner", None) is not None) +
            "; model_wrapper=" + owner[:64])


def explain_rejection(model_options, reason):
    """Keep fail-closed behavior and identify exactly which graph topology failed."""
    if reason and "external transformer patches or wrappers are active" in reason:
        return reason + " (" + _topology(model_options) + ")"
    return reason


def qualify(model_options):
    """Return (policy, reason). Does not mutate model options."""
    trans = model_options.get("transformer_options", {}) or {}
    wrappers = _wrappers(trans)
    if wrappers is None:
        return None, "external transformer patches or wrappers are active"
    try:
        entries = list(_entries(wrappers))
    except ValueError:
        return None, "malformed external transformer wrappers"
    if any(not isinstance(kind, str) for kind, _, _ in entries):
        return None, "external transformer patches or wrappers are active"
    if trans.get("patches_replace"):
        return None, "external transformer patches or wrappers are active"

    sol_override = None
    sol_mark = trans.get(SOL_KEY)
    sol_owner = None
    if sol_mark is None and getattr(trans.get("optimized_attention_override"), "_sol_krea2_owner", None) is not None:
        return None, "unverified Sol-Krea2 attention ownership"
    if sol_mark is not None:
        try:
            from sol_h3 import krea2 as sol
        except ImportError:
            return None, "unavailable Sol-Krea2 compatibility contract"
        sol_override = trans.get("optimized_attention_override")
        sol_owner = getattr(sol_override, "_sol_krea2_owner", None)
        status = sol.spectrum_runtime_status(sol_override)
        if (type(sol_mark) is not dict or sol_mark.get("api") != 1 or
                sol_owner is None or status is None or
                status["settings"].tau != sol_mark.get("tau") or
                status["settings"].min_tokens != sol_mark.get("min_tokens") or
                status["settings"].dense_evaluations != sol_mark.get("dense_evaluations") or
                status["settings"].dense_layers != sol_mark.get("dense_layers") or
                (sol_mark.get("extra_tokens") is not None and
                 status["settings"].extra_tokens != sol_mark["extra_tokens"])):
            return None, "unverified Sol-Krea2 attention ownership"

    diffaid_patch, diffaid_wrapper, diffaid_window = None, None, None
    fn_wrapper = model_options.get("model_function_wrapper")
    patches = trans.get("patches", {}) or {}
    if patches and (not isinstance(patches, dict) or set(patches) - {"attn1_patch"}):
        return None, "external transformer patches or wrappers are active"
    if fn_wrapper is not None or patches or any(k == DIFFAID_WRAPPER for _, k, _ in entries):
        if (not _registered_type(fn_wrapper, "SharedTimestepWrapper") or
                fn_wrapper.existing_wrapper is not None):
            return None, "unsupported model function wrapper"
        if not isinstance(patches, dict) or set(patches) - {"attn1_patch"}:
            return None, "external transformer patches or wrappers are active"
        attn = patches.get("attn1_patch", [])
        residuals = [v[0] for _, key, v in entries if key == DIFFAID_WRAPPER and len(v) == 1]
        if len(attn) == 1 and not residuals and _registered_type(attn[0], "Krea2AttentionPatch"):
            diffaid_patch = attn[0]
            if diffaid_patch.attention_target not in {"values_only", "keys_and_values"}:
                return None, "unsupported Diff-Aid Krea 2 attention mode"
            owner = diffaid_patch
            config = diffaid_patch.config
            blocks = diffaid_patch.selected_blocks
        elif not attn and len(residuals) == 1 and _registered_type(residuals[0], "Krea2TextResidualWrapper"):
            diffaid_wrapper = residuals[0]
            owner = diffaid_wrapper
            config = diffaid_wrapper.config
            blocks = diffaid_wrapper.selected_blocks
        else:
            return None, "unverified Krea 2 Diff-Aid patch ownership"
        window = _diffaid_window(config)
        if (window is None or not blocks or
                any(type(k) is not int or not 0 <= k < 28 for k in blocks)):
            return None, "invalid Krea 2 Diff-Aid patch configuration"
        # A full window has gain one at every sigma. A partial or ramped window
        # changes the patched operator over the trajectory; the adapter then
        # re-evaluates Diff-Aid's own gain on every call and starts fresh
        # forecast history whenever it changes.
        if not (window[0] <= 0.0 and window[1] >= 1.0):
            diffaid_window = _diffaid_gain_contract(owner, config, window)
            if diffaid_window is None:
                return None, "unverified Krea 2 Diff-Aid sigma-window contract"

    # OUTER_SAMPLE preview observers are outside the solver/denoiser path.
    # Admit the registered KJ observer regardless of its implementation
    # version. All numerical contracts are verified on the actual sampling
    # invocation and every subsequent model evaluation.
    kj_preview = None
    kj_entries = [(kind, values) for kind, key, values in entries if key == KJ_PREVIEW_KEY]
    if kj_entries:
        if (len(kj_entries) != 1 or kj_entries[0][0] != "outer_sample" or
                len(kj_entries[0][1]) != 1 or
                not callable(kj_entries[0][1][0])):
            return None, "invalid KJ PreviewOverride wrapper registration"
        kj_preview = kj_entries[0][1][0]

    allowed = {(kind, key) for kind, key, _ in entries if key == SPECTRUM_KEY}
    if kj_preview is not None:
        allowed.add(("outer_sample", KJ_PREVIEW_KEY))
    if sol_override is not None:
        allowed.update({("outer_sample", SOL_KEY), ("diffusion_model", SOL_KEY)})
    if diffaid_wrapper is not None:
        allowed.add(("diffusion_model", DIFFAID_WRAPPER))
    if any((kind, key) not in allowed or not values for kind, key, values in entries):
        return None, "external transformer patches or wrappers are active"
    if sol_override is not None:
        from sol_h3 import krea2 as sol
        expected = {("outer_sample", SOL_KEY): sol._SamplingScope,
                    ("diffusion_model", SOL_KEY): sol._DiffusionScope}
        for kind, key, values in entries:
            cls = expected.get((kind, key))
            if cls is not None and (
                    len(values) != 1 or type(values[0]) is not cls or
                    values[0].owner is not sol_owner):
                return None, "unverified Sol-Krea2 wrapper ownership"

    frozen = tuple((kind, key, tuple(values)) for kind, key, values in entries if key != SPECTRUM_KEY)
    return QualifiedPatches(
        sol_override, diffaid_patch, diffaid_wrapper, kj_preview,
        type(kj_preview).__call__ if kj_preview is not None else None,
        getattr(type(kj_preview).__call__, "__code__", None) if kj_preview is not None else None,
        frozen, diffaid_window,
    ), None


def matches(policy, trans):
    """Verify active invocation retained the audited patch identity."""
    if policy is None:
        wrappers = _wrappers(trans)
        if wrappers is None:
            return False
        try:
            entries = list(_entries(wrappers))
        except ValueError:
            return False
        return not (trans.get("patches") or trans.get("patches_replace") or
                    any(v for kind, key, v in entries if key != SPECTRUM_KEY))
    if trans.get("patches_replace"):
        return False
    patches = trans.get("patches", {}) or {}
    if policy.diffaid_patch is None:
        if patches:
            return False
    elif not isinstance(patches, dict) or (
            set(patches) != {"attn1_patch"} or
            len(patches["attn1_patch"]) != 1 or
            patches["attn1_patch"][0] is not policy.diffaid_patch):
        return False
    if policy.sol_override is not None and trans.get("optimized_attention_override") is not policy.sol_override:
        return False
    if (policy.kj_preview_wrapper is not None and
            (type(policy.kj_preview_wrapper).__call__ is not policy.kj_preview_method or
             getattr(policy.kj_preview_method, "__code__", None) is not policy.kj_preview_code)):
        return False
    try:
        wrapper_items = list(_entries(_wrappers(trans) or {}))
    except ValueError:
        return False
    current_wrappers = tuple((kind, key, tuple(values)) for kind, key, values in wrapper_items if key != SPECTRUM_KEY)
    if len(current_wrappers) != len(policy.verified_wrappers):
        return False
    for current, admitted in zip(current_wrappers, policy.verified_wrappers):
        if (current[:2] != admitted[:2] or len(current[2]) != len(admitted[2]) or
                any(a is not b for a, b in zip(current[2], admitted[2]))):
            return False
    allowed = {(kind, key) for kind, key, _ in wrapper_items if key == SPECTRUM_KEY}
    if policy.sol_override is not None:
        allowed.update({("outer_sample", SOL_KEY), ("diffusion_model", SOL_KEY)})
    if policy.kj_preview_wrapper is not None:
        allowed.add(("outer_sample", KJ_PREVIEW_KEY))
    if policy.diffaid_wrapper is not None:
        allowed.add(("diffusion_model", DIFFAID_WRAPPER))
    for kind, key, wrappers in wrapper_items:
        if (kind, key) not in allowed:
            return False
        if key == KJ_PREVIEW_KEY and (
                kind != "outer_sample" or len(wrappers) != 1 or
                wrappers[0] is not policy.kj_preview_wrapper):
            return False
        if key == DIFFAID_WRAPPER and (
                len(wrappers) != 1 or wrappers[0] is not policy.diffaid_wrapper):
            return False
    return True
