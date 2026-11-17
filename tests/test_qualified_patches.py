"""Compatibility recognition and fail-closed tests for Krea 2 patch owners."""
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

import diffaid_reference
from comfyui_spectrum_krea2.qualified_patches import qualify, matches, DiffAidGainUnavailable


@pytest.fixture
def providers(monkeypatch):
    sol_pkg = ModuleType("sol_h3")
    sol_mod = ModuleType("sol_h3.krea2")
    settings = SimpleNamespace(tau=1.0, min_tokens=4096, dense_evaluations=1, dense_layers=2, extra_tokens=256)
    counters = dict(actual_evaluations=0, sparse_calls=0, kernel_unavailable=False)
    owner = object()

    class _SamplingScope:
        def __init__(self):
            self.owner = owner

    class _DiffusionScope:
        def __init__(self):
            self.owner = owner

    def status(override):
        return {"api": 1, "settings": settings, **counters} if (
            getattr(override, "_sol_krea2_owner", None) is owner) else None

    sol_mod._SamplingScope = _SamplingScope
    sol_mod._DiffusionScope = _DiffusionScope
    sol_mod.spectrum_runtime_status = status
    sol_mod.fp16_backend_available = lambda device: True
    sol_pkg.krea2 = sol_mod
    monkeypatch.setitem(sys.modules, "sol_h3", sol_pkg)
    monkeypatch.setitem(sys.modules, "sol_h3.krea2", sol_mod)

    diff = ModuleType("test_krea2_diffaid_nodes")
    def bind(cls):
        cls.__module__ = diff.__name__
        setattr(diff, cls.__name__, cls)
        return cls

    @bind
    class SharedTimestepWrapper:
        def __init__(self):
            self.existing_wrapper = None

    @bind
    class Krea2AttentionPatch:
        def __init__(self):
            self.attention_target = "keys_and_values"
            self.selected_blocks = frozenset([1, 8, 15])
            self.config = SimpleNamespace(sigma_start=0.0, sigma_end=1.0, sigma_ramp=0.0)

    @bind
    class Krea2TextResidualWrapper:
        def __init__(self):
            self.selected_blocks = (1, 8, 15)
            self.config = SimpleNamespace(sigma_start=0.0, sigma_end=1.0, sigma_ramp=0.0)

    # Diff-Aid's own sigma-window helpers, as the installed module exposes them.
    diff.STATE_KEY = diffaid_reference.STATE_KEY
    diff._coerce_batch_value = diffaid_reference._coerce_batch_value
    diff._sigma_window_gain = diffaid_reference._sigma_window_gain
    monkeypatch.setitem(sys.modules, diff.__name__, diff)
    override = lambda *args, **kwargs: None
    override._sol_krea2_owner = owner
    override._sol_krea2_preprocess = False
    sol_wrappers = {
        "outer_sample": {"sol_krea2_sparse_attention_v1": [_SamplingScope()]},
        "diffusion_model": {"sol_krea2_sparse_attention_v1": [_DiffusionScope()]},
    }
    marker = {"api": 1, "tau": 1.0, "min_tokens": 4096,
              "dense_evaluations": 1, "dense_layers": 2, "extra_tokens": 256}
    return SimpleNamespace(
        override=override, wrappers=sol_wrappers, marker=marker,
        counters=counters, Shared=SharedTimestepWrapper,
        Attention=Krea2AttentionPatch, Residual=Krea2TextResidualWrapper,
        module=diff,
    )


def model_options(providers, sol=True, diff=True, residual=False):
    trans = {"wrappers": {"sampler_sample": {"spectrum_krea2": [lambda: None]}}}
    if sol:
        trans["optimized_attention_override"] = providers.override
        trans["sol_krea2_sparse_attention_v1"] = providers.marker
        for kind, group in providers.wrappers.items():
            trans["wrappers"].setdefault(kind, {}).update(group)
    opts = {"transformer_options": trans}
    if diff:
        opts["model_function_wrapper"] = providers.Shared()
        if residual:
            trans["wrappers"].setdefault("diffusion_model", {})["diffaid_krea2_text_residual"] = [providers.Residual()]
        else:
            trans["patches"] = {"attn1_patch": [providers.Attention()]}
    return opts


@pytest.mark.parametrize("sol,diff,residual", [
    (False, False, False), (True, False, False),
    (False, True, False), (True, True, False),
    (False, True, True), (True, True, True),
])
def test_qualified_known_patches_preserve_active_identity(providers, sol, diff, residual):
    opts = model_options(providers, sol=sol, diff=diff, residual=residual)
    policy, reason = qualify(opts)
    assert reason is None
    assert matches(policy, opts["transformer_options"])
    assert bool(policy.sol_override) == sol
    assert bool(policy.diffaid_patch or policy.diffaid_wrapper) == diff


def test_sol_phase_transitions_after_warmup_without_mutating_settings(providers):
    policy, error = qualify(model_options(providers, sol=True, diff=False))
    assert error is None
    x = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(1, 16, 128, 128))
    context = SimpleNamespace(shape=(1, 256, 1024))
    model = SimpleNamespace(patch=2, dtype=__import__("torch").bfloat16)
    assert policy.phase(x, context, None, model) == "dense"
    providers.counters["actual_evaluations"] = 1
    assert policy.phase(x, context, None, model) == "sol"
    providers.counters["kernel_unavailable"] = True
    assert policy.phase(x, context, None, model) == "dense"


@pytest.mark.parametrize("change", ["extra_patch", "extra_wrapper", "wrong_patch",
                                     "wrong_wrapper", "bad_config", "bad_owner", "unknown_fn",
                                     "nan_sigma", "reversed_sigma", "negative_ramp",
                                     "sigma_above_one", "partial_sigma_without_gain_contract"])
def test_unknown_owners_fail_closed(providers, change):
    opts = model_options(providers)
    trans = opts["transformer_options"]
    if change == "extra_patch":
        trans["patches"]["post_input"] = [lambda: None]
    elif change == "extra_wrapper":
        trans["wrappers"]["diffusion_model"]["unreviewed"] = [lambda: None]
    elif change == "wrong_patch":
        trans["patches"]["attn1_patch"] = [object()]
    elif change == "wrong_wrapper":
        trans["wrappers"]["diffusion_model"]["sol_krea2_sparse_attention_v1"] = [object()]
    elif change == "bad_config":
        trans["sol_krea2_sparse_attention_v1"] = {**providers.marker, "dense_layers": 3}
    elif change == "bad_owner":
        trans["optimized_attention_override"] = lambda: None
    elif change == "unknown_fn":
        opts["model_function_wrapper"] = object()
    elif change == "nan_sigma":
        trans["patches"]["attn1_patch"][0].config.sigma_end = float("nan")
    elif change == "reversed_sigma":
        trans["patches"]["attn1_patch"][0].config.sigma_start = 0.8
        trans["patches"]["attn1_patch"][0].config.sigma_end = 0.2
    elif change == "negative_ramp":
        trans["patches"]["attn1_patch"][0].config.sigma_ramp = -0.1
    elif change == "sigma_above_one":
        trans["patches"]["attn1_patch"][0].config.sigma_end = 1.5
    elif change == "partial_sigma_without_gain_contract":
        trans["patches"]["attn1_patch"][0].config.sigma_end = 0.5
        del providers.module._sigma_window_gain
    policy, reason = qualify(opts)
    assert policy is None
    assert reason
    if change == "partial_sigma_without_gain_contract":
        assert reason == "unverified Krea 2 Diff-Aid sigma-window contract"
    elif change.endswith(("sigma", "ramp", "one")):
        assert reason == "invalid Krea 2 Diff-Aid patch configuration"


def _state(value):
    return {diffaid_reference.STATE_KEY: {"normalized_sigma": value}}


@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("window", [(0.0, 0.9, 0.0), (0.1, 0.9, 0.0), (0.2, 0.7, 0.1)])
def test_partial_diffaid_window_qualifies_with_diffaid_gain(providers, residual, window):
    opts = model_options(providers, sol=False, residual=residual)
    trans = opts["transformer_options"]
    owner = (trans["wrappers"]["diffusion_model"]["diffaid_krea2_text_residual"][0]
             if residual else trans["patches"]["attn1_patch"][0])
    owner.config.sigma_start, owner.config.sigma_end, owner.config.sigma_ramp = window
    policy, reason = qualify(opts)
    assert reason is None and policy.diffaid_window is not None
    x = torch.zeros(2, 16, 4, 4, dtype=torch.float32)
    for value in (1.0, 0.95, 0.9, 0.75, 0.5, 0.15, 0.05, 0.0):
        expected = diffaid_reference._sigma_window_gain(
            torch.tensor([value, value]), *window).tolist()
        # Equal per-row gains collapse to one regime value.
        assert list(policy.diffaid_regime(_state(torch.tensor([value])), x)) == expected[:1]
        assert policy.diffaid_regime(_state(torch.tensor([value])), x[:1]) == tuple(expected[:1])
    assert matches(policy, trans)


def test_full_diffaid_window_has_no_dynamic_regime(providers):
    opts = model_options(providers, sol=False)
    opts["transformer_options"]["patches"]["attn1_patch"][0].config.sigma_ramp = 0.3
    policy, reason = qualify(opts)
    assert reason is None and policy.diffaid_window is None
    assert policy.diffaid_regime(_state(0.5), torch.zeros(1, 16, 2, 2)) is None


def test_diffaid_regime_matches_diffaid_lookup_and_dtype(providers):
    opts = model_options(providers, sol=False)
    config = opts["transformer_options"]["patches"]["attn1_patch"][0].config
    config.sigma_end = 0.899
    policy, _ = qualify(opts)
    f32 = torch.zeros(1, 16, 2, 2, dtype=torch.float32)
    bf16 = f32.to(torch.bfloat16)
    # Diff-Aid rounds the normalized sigma to the modulated hidden-state dtype
    # before comparing: 0.8995 is outside in FP32 but 0.8984375 in BF16.
    assert policy.diffaid_regime(_state(torch.tensor([0.8995])), f32) == (0.0,)
    assert policy.diffaid_regime(_state(torch.tensor([0.8995])), bf16) == (1.0,)
    # A top-level normalized_sigma takes precedence over the injected state.
    options = {"normalized_sigma": 0.5, **_state(0.95)}
    assert policy.diffaid_regime(options, f32) == (1.0,)
    # Diff-Aid falls back to normalized sigma 1.0 without injected state.
    assert policy.diffaid_regime({}, f32) == (0.0,)


def test_diffaid_window_reconfiguration_after_qualification_is_rejected(providers):
    opts = model_options(providers, sol=False)
    config = opts["transformer_options"]["patches"]["attn1_patch"][0].config
    config.sigma_end = 0.9
    policy, _ = qualify(opts)
    config.sigma_end = 0.5
    with pytest.raises(DiffAidGainUnavailable):
        policy.diffaid_regime(_state(0.7), torch.zeros(1, 16, 2, 2))


def test_patch_and_wrapper_mutations_fail_between_sampler_and_model_call(providers):
    opts = model_options(providers)
    trans = opts["transformer_options"]
    policy, reason = qualify(opts)
    assert reason is None
    trans["patches"]["attn1_patch"] = [providers.Attention()]
    assert not matches(policy, trans)
    trans["patches"]["attn1_patch"] = [policy.diffaid_patch]
    trans["wrappers"]["diffusion_model"]["sol_krea2_sparse_attention_v1"] = [
        providers.wrappers["diffusion_model"]["sol_krea2_sparse_attention_v1"][0].__class__()
    ]
    assert not matches(policy, trans)


def test_unknown_non_diffaid_patch_preserves_external_fallback_label(providers):
    opts = {"transformer_options": {"patches": {"post_input": []}}}
    policy, reason = qualify(opts)
    assert policy is None
    assert "external" in reason


@pytest.mark.parametrize("invalid", [
    {"wrappers": {"diffusion_model": []}},
    {"wrappers": {"diffusion_model": {"unknown": object()}}},
    {"wrappers": []},
])
def test_malformed_wrapper_shapes_fail_closed(invalid):
    policy, reason = qualify({"transformer_options": invalid})
    assert policy is None
    assert reason
    assert not matches(None, invalid)


def test_fp16_krea2_route_uses_native_fp16_kernel_capability(providers):
    import torch
    opts = model_options(providers, sol=True, diff=False)
    policy, reason = qualify(opts)
    assert reason is None
    providers.counters["actual_evaluations"] = 3
    x = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(1, 16, 128, 128))
    context = SimpleNamespace(shape=(1, 256, 1024))
    assert policy.phase(x, context, None, SimpleNamespace(patch=2, dtype=torch.float16)) == "sol"
    assert policy.phase(x, context, None, SimpleNamespace(patch=2, dtype=torch.bfloat16)) == "sol"
    import sol_h3.krea2 as sol
    old = sol.fp16_backend_available
    sol.fp16_backend_available = lambda device: False
    try:
        assert policy.phase(x, context, None, SimpleNamespace(patch=2, dtype=torch.float16)) == "dense"
    finally:
        sol.fp16_backend_available = old


def test_fallback_diagnostic_reports_only_names_and_owner(providers):
    from comfyui_spectrum_krea2.qualified_patches import explain_rejection, CONTRACT_REVISION
    opts = model_options(providers)
    opts["transformer_options"]["patches"]["unknown_patch"] = [object()]
    _, reason = qualify(opts)
    assert reason is not None
    detail = explain_rejection(opts, reason)
    assert "unknown_patch" in detail and "attn1_patch" in detail
    assert "sol_marker=True" in detail
    assert "SharedTimestepWrapper" in detail
    assert CONTRACT_REVISION == "qualified-krea2-v2"


def test_kj_preview_registration_is_version_independent(providers):
    from comfyui_spectrum_krea2.qualified_patches import qualify, matches
    # A future preview callback may be a callable object with any name,
    # module, signature or implementation. Its OUTER_SAMPLE identity does not
    # change the native sampler or transformer contracts observed inside.
    class PreviewVNext:
        def __call__(self, executor, *args, **kwargs):
            return executor(*args, **kwargs)

    preview = PreviewVNext()
    opts = model_options(providers, sol=True, diff=False)
    wraps = opts["transformer_options"]["wrappers"]
    wraps.setdefault("outer_sample", {})["kj_preview_override"] = [preview]
    policy, reason = qualify(opts)
    assert reason is None and policy.kj_preview_wrapper is preview
    assert matches(policy, opts["transformer_options"])

    # A method or code-object mutation during the active invocation must
    # invalidate its snapshot even when another invocation can requalify.
    original_method = PreviewVNext.__call__
    try:
        PreviewVNext.__call__ = lambda self, *a, **kw: None
        assert not matches(policy, opts["transformer_options"])
        next_policy, reason = qualify(opts)
        assert reason is None and next_policy.kj_preview_wrapper is preview
        assert matches(next_policy, opts["transformer_options"])
    finally:
        PreviewVNext.__call__ = original_method
    assert matches(policy, opts["transformer_options"])

    replacement = PreviewVNext()
    wraps["outer_sample"]["kj_preview_override"] = [replacement]
    assert not matches(policy, opts["transformer_options"])
    assert qualify(opts)[1] is None
    wraps["outer_sample"]["kj_preview_override"] = [preview, replacement]
    assert qualify(opts)[1] == "invalid KJ PreviewOverride wrapper registration"
    wraps["outer_sample"].pop("kj_preview_override")
    wraps["diffusion_model"]["kj_preview_override"] = [preview]
    assert qualify(opts)[1] == "invalid KJ PreviewOverride wrapper registration"


def test_kj_preview_still_rejects_unknown_numerical_modifiers(providers):
    from comfyui_spectrum_krea2.qualified_patches import qualify
    opts = model_options(providers, sol=True, diff=False)
    opts["transformer_options"]["wrappers"].setdefault("outer_sample", {})[
        "kj_preview_override"
    ] = [lambda executor, *a, **kw: executor(*a, **kw)]
    assert qualify(opts)[1] is None
    opts["transformer_options"]["wrappers"]["diffusion_model"]["untrusted"] = [
        lambda *args: None
    ]
    assert qualify(opts)[1] == "external transformer patches or wrappers are active"



def test_kj_preview_unverified_or_malformed_still_falls_back(providers):
    opts = model_options(providers, sol=True, diff=False)
    wraps = opts["transformer_options"]["wrappers"]
    wraps.setdefault("outer_sample", {})["kj_preview_override"] = [object()]
    policy, reason = qualify(opts)
    assert policy is None and reason == "invalid KJ PreviewOverride wrapper registration"
    wraps["outer_sample"]["kj_preview_override"] = []
    policy, reason = qualify(opts)
    assert policy is None and reason == "invalid KJ PreviewOverride wrapper registration"


def test_diffaid_regime_keeps_genuine_per_row_differences(providers):
    opts = model_options(providers, sol=False)
    opts["transformer_options"]["patches"]["attn1_patch"][0].config.sigma_end = 0.6
    policy, _ = qualify(opts)
    x = torch.zeros(2, 16, 2, 2)
    assert policy.diffaid_regime(_state(torch.tensor([0.9, 0.5])), x) == (0.0, 1.0)


@pytest.mark.parametrize("residual", [False, True])
def test_replaced_full_window_diffaid_config_fails_matches(providers, residual):
    opts = model_options(providers, sol=False, residual=residual)
    trans = opts["transformer_options"]
    policy, reason = qualify(opts)
    assert reason is None and policy.diffaid_window is None and matches(policy, trans)
    owner = policy.diffaid_wrapper if residual else policy.diffaid_patch
    owner.config = SimpleNamespace(sigma_start=0.0, sigma_end=0.5, sigma_ramp=0.0)
    assert not matches(policy, trans)
