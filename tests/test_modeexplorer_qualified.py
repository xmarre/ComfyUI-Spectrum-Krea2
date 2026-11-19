"""Narrow trust and regime rules for ModeExplorer's Spectrum integration."""
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from comfyui_spectrum_krea2.qualified_patches import qualify, matches


@pytest.fixture
def mode_graph(monkeypatch):
    name = "test_plugin.mode_explorer.runtime"
    module = ModuleType(name)

    @dataclass(frozen=True)
    class Settings:
        feature: str = "feature_cosine"
        initialization: bool = False
        strength: float = 0.45
        fidelity: float = 0.55
        min_sigma: float = 0.82
        adaptive: bool = True
        similarity: float = 0.70
        operation: str = "auto"
        reference_budget: int = 2
        diagnostics: bool = False

    class SamplerWrapper:
        def __init__(self, settings):
            self.settings = settings

    Settings.__module__ = name
    SamplerWrapper.__module__ = name

    def diffusion_wrapper(*args):
        raise AssertionError("not called in policy-only tests")

    def attention_feature(*args):
        raise AssertionError("not called in policy-only tests")

    module.Settings = Settings
    module.SamplerWrapper = SamplerWrapper
    module.diffusion_wrapper = diffusion_wrapper
    module.attention_feature = attention_feature
    monkeypatch.setitem(sys.modules, name, module)

    sampler = SamplerWrapper(Settings())
    trans = {
        "wrappers": {
            "sampler_sample": {"mode_explorer_krea2": [sampler]},
            "diffusion_model": {"mode_explorer_krea2": [diffusion_wrapper]},
        },
        "patches": {"attn1_output_patch": [attention_feature]},
    }
    return sampler, trans


def test_verified_modeexplorer_is_admitted_and_regimes_are_scoped(mode_graph):
    sampler, trans = mode_graph
    policy, reason = qualify({"transformer_options": trans})
    assert reason is None
    assert policy.mode_explorer.sampler is sampler
    assert matches(policy, trans)
    guides = policy.mode_explorer.guides
    assert guides(1.0, 0, 1, "planner")
    assert guides(0.96, 1, 1, "planner")
    assert not guides(0.90, 2, 1, "planner")  # sequential reference budget ended
    assert not guides(0.81, 0, 1, "planner")  # sigma gate
    assert not guides(1.0, 0, 1, "highres")
    assert guides(0.9, 3, 2, "planner")  # batch has no reference budget
    assert not guides(0.9, 3, 2, "highres")


@pytest.mark.parametrize("change", ["patch", "sampler", "wrapper", "extra", "settings"])
def test_modified_modeexplorer_owner_fails_closed(mode_graph, change):
    sampler, trans = mode_graph
    policy, reason = qualify({"transformer_options": trans})
    assert reason is None
    if change == "patch":
        trans["patches"]["attn1_output_patch"] = [object()]
    elif change == "sampler":
        trans["wrappers"]["sampler_sample"]["mode_explorer_krea2"] = [object()]
    elif change == "wrapper":
        trans["wrappers"]["diffusion_model"]["mode_explorer_krea2"] = [object()]
    elif change == "extra":
        trans["patches"]["post_input"] = [object()]
    elif change == "settings":
        sampler.settings = type(sampler.settings)(feature="feature_volume")
    assert not matches(policy, trans)
    if change != "settings":
        rejected, reason = qualify({"transformer_options": trans})
        assert rejected is None and reason


@pytest.mark.parametrize("feature,batch,operation,active", [
    ("native", 1, "auto", False),
    ("feature_cosine", 1, "batch", False),
    ("feature_cosine", 2, "sequential", False),
    ("feature_cosine", 2, "batch", True),
])
def test_explicit_operation_guards(mode_graph, feature, batch, operation, active):
    sampler, trans = mode_graph
    sampler.settings = type(sampler.settings)(feature=feature, operation=operation)
    policy, reason = qualify({"transformer_options": trans})
    assert reason is None and policy.mode_explorer.guides(0.99, 0, batch, "planner") is active
