"""Compatibility recognition and fail-closed tests for Krea 2 patch owners."""
from types import ModuleType, SimpleNamespace
import sys

import pytest

from comfyui_spectrum_krea2.qualified_patches import qualify, matches


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
                                     "partial_sigma"])
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
    elif change == "partial_sigma":
        trans["patches"]["attn1_patch"][0].config.sigma_end = 0.5
    policy, reason = qualify(opts)
    assert policy is None
    assert reason


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


def test_audited_kj_preview_is_allowed_only_at_outer_sample(providers, monkeypatch):
    from comfyui_spectrum_krea2 import qualified_patches as q
    class FakePreview:
        def __call__(self, *args):
            pass

    preview = FakePreview()
    # The real upstream source fingerprint is checked in the full-native
    # integration lane; isolate identity/placement checks in this fixture.
    monkeypatch.setattr(q, "_audited_kj_preview_type", lambda cls: cls is FakePreview)
    opts = model_options(providers, sol=True, diff=False)
    wraps = opts["transformer_options"]["wrappers"]
    wraps.setdefault("outer_sample", {})["kj_preview_override"] = [preview]
    policy, reason = qualify(opts)
    assert reason is None and policy.kj_preview_wrapper is preview
    assert matches(policy, opts["transformer_options"])

    wraps["outer_sample"]["kj_preview_override"] = [object()]
    assert not matches(policy, opts["transformer_options"])
    wraps["outer_sample"]["kj_preview_override"] = [preview, object()]
    assert qualify(opts)[1] == "unverified KJ PreviewOverride wrapper source or ownership"
    wraps["outer_sample"].pop("kj_preview_override")
    wraps["diffusion_model"]["kj_preview_override"] = [preview]
    assert qualify(opts)[1] == "unverified KJ PreviewOverride wrapper source or ownership"


def test_kj_preview_unverified_or_malformed_still_falls_back(providers):
    opts = model_options(providers, sol=True, diff=False)
    wraps = opts["transformer_options"]["wrappers"]
    wraps.setdefault("outer_sample", {})["kj_preview_override"] = [object()]
    policy, reason = qualify(opts)
    assert policy is None and reason == "unverified KJ PreviewOverride wrapper source or ownership"
    wraps["outer_sample"]["kj_preview_override"] = []
    policy, reason = qualify(opts)
    assert policy is None and reason == "unverified KJ PreviewOverride wrapper source or ownership"
