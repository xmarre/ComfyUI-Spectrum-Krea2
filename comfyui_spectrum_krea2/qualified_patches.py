# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
"""Narrow, runtime-verified Krea 2 attention-patch interoperability.

Only the reviewed Sol-Krea2 and Krea2 Diff-Aid implementations are eligible.
Unknown patches, wrapper owners and model-function decorators fail closed.
"""
from dataclasses import dataclass
import ast
import inspect
import textwrap
import types
from pathlib import Path
import sys

import torch

CONTRACT_REVISION = "qualified-krea2-v2"
SOL_KEY = "sol_krea2_sparse_attention_v1"
SPECTRUM_KEY = "spectrum_krea2"
DIFFAID_WRAPPER = "diffaid_krea2_text_residual"
KJ_PREVIEW_KEY = "kj_preview_override"
# KJNodes has several released preview_override_node.py revisions. A whole-file
# digest also changes for unrelated encoders, imports and UI controls, so it
# incorrectly rejects compatible preview-only callbacks. Qualify the *actual*
# registered method's sampler/callback protocol, not an arbitrary module key.
_KJ_ARGS = ("self", "executor", "noise", "latent_image", "sampler",
            "sigmas", "denoise_mask", "callback", "disable_pbar", "seed",
            "latent_shapes")
_KJ_FORWARD = ("noise", "latent_image", "sampler", "sigmas",
               "denoise_mask", "new_callback", "disable_pbar", "seed")
_KJ_CALLBACK = ("step", "x0", "x", "total_steps_")
_KJ_IMMUTABLE = frozenset({"noise", "latent_image", "sampler", "sigmas",
                          "denoise_mask", "seed", "latent_shapes",
                          "callback", "original_callback", "x0", "x"})


def _kj_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def _code_contract(code):
    """Compare executable semantics, not source-line offsets or qualnames."""
    return (
        code.co_code, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_nlocals, code.co_flags,
        code.co_names, code.co_varnames, code.co_cellvars,
        code.co_freevars,
        tuple(_code_contract(v) if isinstance(v, types.CodeType) else v
              for v in code.co_consts),
        getattr(code, "co_exceptiontable", b""),
    )


def _audited_kj_preview_type(cls):
    """Check a registered KJNodes preview-only OUTER_SAMPLE delegate.

    Versions may change encoders and UI without altering the sampling
    contract. Source and runtime callable must be the same registered method;
    any change to forwarding, callback, or cleanup falls back to native.
    """
    if cls.__name__ != "_PreviewOverrideWrapper":
        return False
    fn = cls.__dict__.get("__call__")
    module = sys.modules.get(cls.__module__)
    if (not inspect.isfunction(fn) or module is None or
            getattr(module, "_PreviewOverrideWrapper", None) is not cls or
            not (cls.__module__ == "preview_override_node" or
                 cls.__module__.endswith(".preview_override_node")) or
            fn.__module__ != cls.__module__):
        return False
    try:
        filename = inspect.getsourcefile(fn)
        if (filename is None or Path(filename).name != "preview_override_node.py"
                or Path(filename).resolve() != Path(module.__file__).resolve()):
            return False
        source = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(source)
    except (OSError, TypeError, ValueError, SyntaxError, AttributeError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    method = tree.body[0]
    # The file's checked syntax must be the in-memory method Python will
    # actually execute. inspect.getsource alone would miss a __code__ swap.
    try:
        compiled = compile(tree, "<kj_preview_contract>", "exec")
        expected = next(value for value in compiled.co_consts
                        if isinstance(value, types.CodeType)
                        and value.co_name == "__call__")
    except (ValueError, StopIteration, SyntaxError, TypeError):
        return False
    if _code_contract(fn.__code__) != _code_contract(expected):
        return False
    params = method.args
    if (method.name != "__call__" or
            tuple(a.arg for a in params.posonlyargs + params.args) != _KJ_ARGS or
            params.kwonlyargs or params.vararg or params.kwarg or
            params.defaults or params.kw_defaults or method.decorator_list):
        return False

    # Only one native sample invocation, with the original sampler, sigma,
    # noise, seed and masks untouched. The sole intervention is replacing
    # the callback with the KJ preview callback.
    calls = [n for n in ast.walk(method)
             if isinstance(n, ast.Call) and _kj_name(n.func, "executor")]
    if len(calls) != 1:
        return False
    call = calls[0]
    if (tuple(n.id if isinstance(n, ast.Name) else None for n in call.args)
            != _KJ_FORWARD or len(call.keywords) != 1 or
            call.keywords[0].arg != "latent_shapes" or
            not _kj_name(call.keywords[0].value, "latent_shapes")):
        return False
    if not any(
        isinstance(n, ast.Try) and n.finalbody and
        any(isinstance(stmt, ast.Return) and stmt.value is call
            for stmt in n.body)
        for n in ast.walk(method)
    ):
        return False

    # KJ forwards the outer solver callback unmodified, both when preview
    # decoding succeeds and when it produces no frame.
    callbacks = [n for n in method.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == "new_callback"]
    if len(callbacks) != 1 or isinstance(callbacks[0], ast.AsyncFunctionDef):
        return False
    cb = callbacks[0]
    if tuple(a.arg for a in cb.args.args) != _KJ_CALLBACK:
        return False
    forwards = [n for n in ast.walk(cb)
                if isinstance(n, ast.Call) and
                _kj_name(n.func, "original_callback")]
    if (len(forwards) != 2 or
            any(tuple(n.id if isinstance(n, ast.Name) else None for n in c.args)
                != _KJ_CALLBACK or c.keywords for c in forwards)):
        return False
    initial = [n for n in ast.walk(method) if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and
               _kj_name(n.targets[0], "original_callback") and
               _kj_name(n.value, "callback")]
    if len(initial) != 1:
        return False

    # Reject source revisions that rebind any original sampler input or
    # mutate the callback's original x0/x variables in-place. This deliberately
    # excludes unknown future KJ behavior rather than accepting its key.
    for node in ast.walk(method):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in _KJ_IMMUTABLE:
                if not (node.id == "original_callback" and node is initial[0].targets[0]):
                    # Formal parameters appear only as ast.arg, not ast.Name.
                    return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile"}:
                return False
    return True


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


@dataclass(frozen=True)
class QualifiedPatches:
    sol_override: object = None
    diffaid_patch: object = None
    diffaid_wrapper: object = None
    kj_preview_wrapper: object = None
    kj_preview_method: object = None
    kj_preview_code: object = None
    verified_wrappers: tuple = ()

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

    diffaid_patch, diffaid_wrapper = None, None
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
            config = diffaid_patch.config
            blocks = diffaid_patch.selected_blocks
        elif not attn and len(residuals) == 1 and _registered_type(residuals[0], "Krea2TextResidualWrapper"):
            diffaid_wrapper = residuals[0]
            config = diffaid_wrapper.config
            blocks = diffaid_wrapper.selected_blocks
        else:
            return None, "unverified Krea 2 Diff-Aid patch ownership"
        # A partially active sigma window changes the numerical operator
        # discontinuously. Keep it native until time-dependent patch history
        # and boundary receipts have been explicitly validated.
        if (config.sigma_start != 0.0 or config.sigma_end != 1.0 or
                config.sigma_ramp != 0.0 or not blocks or
                any(type(k) is not int or not 0 <= k < 28 for k in blocks)):
            return None, "dynamic or invalid Krea 2 Diff-Aid patch configuration"

    # A preview-only OUTER_SAMPLE wrapper may wrap the callback without
    # altering the denoiser, solver state or conditioning. Admit only the
    # exact upstream implementation audited above. Its source and instance
    # must both be verified; no key-only allowlist is sufficient.
    kj_preview = None
    kj_entries = [(kind, values) for kind, key, values in entries if key == KJ_PREVIEW_KEY]
    if kj_entries:
        if (len(kj_entries) != 1 or kj_entries[0][0] != "outer_sample" or
                len(kj_entries[0][1]) != 1 or
                not _audited_kj_preview_type(type(kj_entries[0][1][0]))):
            return None, "unverified KJ PreviewOverride wrapper source or ownership"
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
        type(kj_preview).__call__.__code__ if kj_preview is not None else None,
        frozen,
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
             policy.kj_preview_method.__code__ is not policy.kj_preview_code)):
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
