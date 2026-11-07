"""Forecast isolation for native SA-Solver.

A forecast cannot see the stochastic increment SA just added to its predictor
state. These tests pin the two ownership rules of the adapter: forecasts never
enter persistent Adams history, and on stochastic inputs SA consumes a value
built only from actual anchors. Without forecasts the adapter is the native
solver, bit for bit.
"""
import types
import pytest
import torch
from comfyui_spectrum_krea2.config import Config
from comfyui_spectrum_krea2.runtime import Session
from comfyui_spectrum_krea2.sa_isolation import causal_denoised, sample_sa_solver_isolated


class Model:
    """Deterministic nonlinear denoiser that can mark chosen steps as forecasts."""

    def __init__(self, session=None, forecast_steps=(), forecast_value=None):
        self.session = session
        self.forecast_steps = set(forecast_steps)
        self.forecast_value = forecast_value
        self.calls = 0
        self.predictor_step = -1
        self.inner_model = types.SimpleNamespace(model_patcher=types.SimpleNamespace(
            get_model_object=lambda name: types.SimpleNamespace(noise_scale=1.0)))

    def __call__(self, x, sigma, **kwargs):
        self.calls += 1
        sigma = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        actual = torch.tanh(x) * (1.0 - sigma) + 0.25 * x.roll(1, -1)
        return actual


def session_for(steps):
    session = Session(object(), Config(), [0.0] * steps, None, None, 1, (4, 6))
    return session


class Steps:
    """Per-predictor-step forecast marking, matching Spectrum's served flag."""

    def __init__(self, session, model, forecast_steps, forecast_value):
        self.session, self.model = session, model
        self.forecast_steps, self.forecast_value = set(forecast_steps), forecast_value
        self.step = -1
        self.last_sigma = None
        self.inner_model = model.inner_model

    def __call__(self, x, sigma, **kwargs):
        value = float(sigma.flatten()[0])
        if value != self.last_sigma:  # new predictor opportunity
            self.step += 1
            self.last_sigma = value
            if self.step in self.forecast_steps:
                self.session.served_forecast = True
                return self.forecast_value(x, self.step)
        return self.model(x, sigma, **kwargs)


def noise_sampler_factory(seed):
    generator = torch.Generator().manual_seed(seed)
    return lambda sigma, sigma_next: torch.randn(1, 2, 4, 6, generator=generator)


def run_native(native, sigmas, x, **options):
    from comfy.k_diffusion import sampling
    model = Model()
    return sampling.sample_sa_solver(model, x, sigmas, noise_sampler=noise_sampler_factory(7), **options)


def run_isolated(sigmas, x, forecast_steps=(), forecast_value=None, **options):
    session = session_for(len(sigmas) - 1)
    model = Steps(session, Model(), forecast_steps, forecast_value)
    out = sample_sa_solver_isolated(session, model, x, sigmas,
                                    noise_sampler=noise_sampler_factory(7), **options)
    return out, session


SIGMAS = torch.tensor([.98, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0.])


@pytest.mark.parametrize('use_pece', [False, True])
@pytest.mark.parametrize('stochastic', [False, True])
@pytest.mark.parametrize('simple_order_2,orders', [(False, (3, 4)), (True, (2, 1)), (False, (1, 0))])
def test_without_forecasts_matches_native_bitwise(native, use_pece, stochastic, simple_order_2, orders):
    x = torch.randn(1, 2, 4, 6, generator=torch.Generator().manual_seed(3))
    options = dict(tau_func=(lambda s: 0.8) if stochastic else (lambda s: 0.0),
                   s_noise=0.7 if stochastic else 0.0, predictor_order=orders[0],
                   corrector_order=orders[1], use_pece=use_pece, simple_order_2=simple_order_2)
    expected = run_native(native, SIGMAS, x, **options)
    actual, session = run_isolated(SIGMAS, x, **options)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert session.counts["sa_ephemeral_forecast"] == 0


@pytest.mark.parametrize('use_pece', [False, True])
def test_stochastic_forecasts_cannot_reach_the_trajectory(native, use_pece):
    """Raw forecast values on stochastic inputs are discarded entirely.

    The increment-blind forecast is exactly what produced confetti. Any two
    choices of raw forecast output must yield the identical SA result.
    """
    x = torch.randn(1, 2, 4, 6, generator=torch.Generator().manual_seed(5))
    options = dict(tau_func=lambda s: 1.0, s_noise=1.0, use_pece=use_pece)
    forecasts = (3, 5, 7)
    first, session = run_isolated(SIGMAS, x, forecasts, lambda x, i: torch.full_like(x, 1e3), **options)
    second, _ = run_isolated(SIGMAS, x, forecasts, lambda x, i: x.clone(), **options)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert torch.isfinite(first).all()
    assert session.counts["sa_dense_output"] == len(forecasts)
    modes = [entry["mode"] for entry in session.sa_dense_steps]
    assert [entry["step"] for entry in session.sa_dense_steps] == list(forecasts)
    assert sum(session.counts["sa_dense_" + m] for m in ("hold", "guard_hold", "extrapolation")) == len(forecasts)
    assert set(modes) <= {"hold", "guard_hold", "extrapolation"}
    assert session.counts["sa_ephemeral_forecast"] == len(forecasts)


def test_deterministic_forecast_is_used_but_never_persisted(native):
    """On ODE steps a forecast is the current endpoint, not later history."""
    x = torch.randn(1, 2, 4, 6, generator=torch.Generator().manual_seed(9))
    options = dict(tau_func=lambda s: 0.0, s_noise=0.0)
    reference, _ = run_isolated(SIGMAS, x, (4,), lambda x, i: Model()(x, SIGMAS[i:i + 1]), **options)
    native_out = run_native(native, SIGMAS, x, **options)
    # A forecast equal to the actual value changes only the Adams persistence:
    # the result differs from native exactly because the value is not reused.
    assert not torch.equal(reference, native_out)
    shifted, session = run_isolated(SIGMAS, x, (4,), lambda x, i: Model()(x, SIGMAS[i:i + 1]) + 0.5, **options)
    assert not torch.equal(shifted, reference)
    assert session.counts["sa_dense_output"] == 0


def test_causal_dense_output_is_bounded_and_causal():
    a, b = torch.zeros(3), torch.ones(3)
    held, mode = causal_denoised([a, b], [torch.tensor(0.), torch.tensor(1.)], [3, 4], torch.tensor(1.5))
    assert mode == "hold" and torch.equal(held, b)
    value, mode = causal_denoised([a, b], [torch.tensor(0.), torch.tensor(1.)], [2, 4], torch.tensor(1.5))
    assert mode == "extrapolation"
    torch.testing.assert_close(value, torch.full((3,), 1.5))
    # Beyond one anchor interval the guard holds the latest anchor (H3 semantics).
    overlong, mode = causal_denoised([a, b], [torch.tensor(0.), torch.tensor(1.)], [2, 4], torch.tensor(9.))
    assert mode == "guard_hold" and torch.equal(overlong, b)
    backwards, mode = causal_denoised([a, b], [torch.tensor(0.), torch.tensor(1.)], [2, 4], torch.tensor(.5))
    assert mode == "guard_hold" and torch.equal(backwards, b)
