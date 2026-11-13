# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2026 xmarre
"""Forecast scheduling on short partial-denoise schedules.

A refinement pass re-noises a finished image and runs only the trailing steps
of a schedule. With two warmup steps and a one-step tail, a four-step pass has
exactly one forecast-eligible step, so any change of the evaluated operator at
or before that step leaves the pass without a forecast. These tests run the
native ER-SDE solver through the real sampler wrappers and assert, per step,
which evaluations are actual, why, and that no forecast ever extrapolates
anchors evaluated under a different operator.
"""
import math
import types

import pytest
import torch

import diffaid_reference
from native_components import tiny_model
from test_native_integration import Owner
from comfyui_spectrum_krea2.config import Config
from comfyui_spectrum_krea2.forecast import History
from comfyui_spectrum_krea2.qualified_patches import DiffAidWindow, QualifiedPatches
from comfyui_spectrum_krea2.runtime import CURRENT, REPORTS
from comfyui_spectrum_krea2.sampling import Binding

TRANSITION = 'Diff-Aid sigma-window transition'
ROUTE = 'numerical attention backend transition'
GUARD = 'forecast rejected: extrapolation exceeds half the retained history span'


def simple_flow_schedule(steps, shift=math.exp(1.15)):
    """ComfyUI's ``simple`` scheduler on the Krea 2 flow sigma table (FP32)."""
    values = []
    for i in range(steps):
        t = (10000 - int(i * (10000 / steps))) / 10000
        values.append(shift * t / (1 - t + shift * t))
    return torch.tensor(values + [0.], dtype=torch.float32)


FULL = {14: simple_flow_schedule(14), 16: simple_flow_schedule(16)}
# Trailing steps of the 14- and 16-step schedules below sigma 0.60.
PASS = {4: FULL[14][10:], 5: FULL[16][11:]}
TURBO = dict(preset='turbo', degree=1, warmup=2, tail=1, window=1, bootstrap=False,
             extrapolation_guard_enabled=False)


def test_refinement_schedules_are_the_trailing_native_steps():
    torch.testing.assert_close(PASS[4], torch.tensor(
        [0.55826634, 0.46276948, 0.34492800, 0.19562392, 0.]), atol=2e-6, rtol=0)
    torch.testing.assert_close(PASS[5], torch.tensor(
        [0.58941382, 0.51284409, 0.42156866, 0.31090108, 0.17392661, 0.]), atol=2e-6, rtol=0)


class ScalerERSDE:
    """ER-SDE noise scaler family with eta 1."""
    def __call__(self, x):
        return x * ((x ** 0.3).exp() + 10.0)


class SeededNoise:
    """One reproducible Gaussian increment per stochastic step."""
    def __init__(self, shape, seed=1234):
        self.shape, self.seed, self.calls = tuple(shape), seed, 0

    def __call__(self, sigma, sigma_next):
        generator = torch.Generator(device='cpu').manual_seed(self.seed + self.calls)
        self.calls += 1
        return torch.randn(self.shape, generator=generator)


def diffaid_policy(window):
    config = types.SimpleNamespace(sigma_start=window[0], sigma_end=window[1], sigma_ramp=window[2])
    owner = types.SimpleNamespace(config=config)
    contract = None
    if not (window[0] <= 0.0 and window[1] >= 1.0):
        contract = DiffAidWindow(owner, config, tuple(window), diffaid_reference.STATE_KEY,
                                 diffaid_reference._coerce_batch_value,
                                 diffaid_reference._sigma_window_gain)
    return QualifiedPatches(diffaid_window=contract)


class SolPolicy:
    """Sol-Krea2 routing: the first ``dense_evaluations`` real evaluations of a
    sampler call are dense, later ones sparse. Counts are sampler-call local."""
    diffaid_window = None

    def __init__(self, dense_evaluations):
        self.dense_evaluations = dense_evaluations

    def diffaid_regime(self, options, x):
        return None

    def _count(self):
        return CURRENT.get().counts['actual']

    def phase(self, x, context, refs, inner):
        return 'dense' if self._count() + 1 <= self.dense_evaluations else 'sol'

    def sol_status(self):
        count = self._count()
        return {'actual_evaluations': count,
                'sparse_calls': max(0, count - self.dense_evaluations)}


class Stage(Owner):
    """Host model of one sampler call.

    ``reference`` reproduces the normalized sigma a Diff-Aid timestep wrapper
    injects into every model call: the sigma divided by the first sigma of
    this sampler call (``'schedule_start'``) or by a fixed sigma.
    """
    def __init__(self, model, binding, context, sigmas, window, reference, split):
        super().__init__(model, binding, {}, context, split)
        self.window = window
        self.divisor = float(sigmas[0]) if reference == 'schedule_start' else float(reference)
        self.trace = []

    def gain(self, t):
        if self.window is None:
            return None
        normalized = (t.reshape(-1)[:1].abs() / self.divisor).clamp(0.0, 1.0)
        return float(diffaid_reference._sigma_window_gain(normalized, *self.window))

    def predict(self, x, t, model_options, seed):
        session = CURRENT.get()
        before = session.counts['forecast']
        if self.window is not None:
            self.options = {diffaid_reference.STATE_KEY: {
                'normalized_sigma': (t.abs() / self.divisor).clamp(0.0, 1.0)}}
        out = super().predict(x, t, model_options, seed)
        self.trace.append(('F' if session.counts['forecast'] > before else 'A', self.gain(t)))
        return out


def run_stage(model, sigmas, config=None, *, window=None, reference='schedule_start', policy=None,
              split=False, binding=None, shape=(5, 7), callback=None, monkeypatch,
              model_options=None, seed=99, stochastic=True):
    """One native ER-SDE sampler call. Returns (latent, report, per-step trace)."""
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfy.patcher_extension import WrapperExecutor
    import comfyui_spectrum_krea2.adapter as adapter
    import comfyui_spectrum_krea2.qualified_patches as qualified

    if policy is None and window is not None:
        policy = diffaid_policy(window)
    monkeypatch.setattr(adapter, 'matches', lambda active, options: True)
    monkeypatch.setattr(qualified, 'qualify', lambda options: (
        policy if policy is not None else QualifiedPatches(), None))
    binding = binding or Binding(Config(**(config or TURBO)))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(1, model.channels, *shape, generator=generator, dtype=model.dtype)
    context = torch.randn(1, 3, model.txtlayers * model.txtdim, generator=generator, dtype=model.dtype)
    owner = Stage(model, binding, context, sigmas, window, reference, split)
    options = {'max_stage': 3, 's_noise': 1.0 if stochastic else 0.0, 'noise_scaler': ScalerERSDE()}
    if stochastic:
        options['noise_sampler'] = SeededNoise(x.shape)
    sampler = KSAMPLER(sampling.sample_er_sde, extra_options=options)
    executor = WrapperExecutor.new_class_executor(sampler.sample, sampler, [binding.sampler])
    reports = []
    token = REPORTS.set(reports)
    try:
        with torch.no_grad():
            result = executor.execute(
                owner, sigmas, {'model_options': {'transformer_options': {}, **(model_options or {})}},
                callback, x, torch.zeros_like(x))
    finally:
        REPORTS.reset(token)
    assert CURRENT.get() is None and not model.last._forward_pre_hooks
    return result, reports[0], owner.trace


def pattern(trace):
    return ''.join(kind for kind, _ in trace)


@pytest.fixture
def model(native):
    return tiny_model(native[0])


@pytest.fixture
def anchors(monkeypatch):
    """Record, for every regression forecast, the outer steps of its anchors."""
    used = []
    original = History.predict

    def spy(self, coordinate, device, dtype):
        session = CURRENT.get()
        steps = [min(range(len(session.coordinates)),
                     key=lambda i: abs(session.coordinates[i] - c)) for c in self.coordinates]
        used.append((session.active_step, steps))
        return original(self, coordinate, device, dtype)

    monkeypatch.setattr(History, 'predict', spy)
    return used


def assert_homogeneous(trace, anchors):
    """Every forecast used only anchors evaluated under its own gain."""
    for step, steps in anchors:
        assert trace[step][0] == 'F'
        assert {trace[i][0] for i in steps} == {'A'}
        assert {trace[i][1] for i in steps} == {trace[step][1]}, (step, steps, trace)


# -- Diff-Aid sigma windows ------------------------------------------------

@pytest.mark.parametrize('steps, expected', [(4, 'AAFA'), (5, 'AAFAA')])
@pytest.mark.parametrize('window', [None, (0.0, 1.0, 0.0), (0.0, 1.0, 0.2)])
def test_constant_operator_forecasts_the_single_eligible_step(model, monkeypatch, anchors, steps, expected, window):
    _, report, trace = run_stage(model, PASS[steps], window=window, monkeypatch=monkeypatch)
    assert pattern(trace) == expected and report['fallback'] is None
    assert (report['actual'], report['forecast']) == (steps - 1, 1)
    assert report['actual_reasons'] == {} and report['diffaid_gains'] == []
    assert_homogeneous(trace, anchors)


@pytest.mark.parametrize('steps', [4, 5])
@pytest.mark.parametrize('window', [(0.75, 1.0, 0.0), (0.0, 0.75, 0.0)])
def test_hard_window_restarted_by_the_pass_leaves_no_forecast(model, monkeypatch, anchors, steps, window):
    """Normalized against the pass's own first sigma, the boundary is crossed
    at the third step: exactly where the first forecast would be eligible."""
    _, report, trace = run_stage(model, PASS[steps], window=window, monkeypatch=monkeypatch)
    assert pattern(trace) == 'A' * steps and anchors == []
    assert (report['actual'], report['forecast'], report['fallback']) == (steps, 0, None)
    assert report['actual_reasons'] == {TRANSITION: 1}
    first, second = (1.0, 0.0) if window[0] else (0.0, 1.0)
    assert [gain for _, gain in trace] == [first, first] + [second] * (steps - 2)
    assert [(entry['step'], entry['gain']) for entry in report['diffaid_gains']] == [
        (0, [first]), (2, [second])]
    assert report['diffaid_gains'][1]['sigma'] == pytest.approx(float(PASS[steps][2]))


@pytest.mark.parametrize('steps, expected', [(4, 'AAFA'), (5, 'AAFAA')])
@pytest.mark.parametrize('window, gain', [((0.75, 1.0, 0.0), 0.0), ((0.0, 0.75, 0.0), 1.0),
                                          ((0.75, 1.0, 0.1), 0.0), ((0.0, 0.75, 0.1), 1.0)])
def test_window_on_absolute_noise_levels_keeps_one_gain_and_forecasts(
        model, monkeypatch, anchors, steps, expected, window, gain):
    """Normalized against the model's maximum sigma, every step of the pass is
    on one side of the boundary, so the operator is the same fixed function."""
    result, report, trace = run_stage(model, PASS[steps], window=window, reference=1.0,
                                      monkeypatch=monkeypatch)
    assert pattern(trace) == expected and {g for _, g in trace} == {gain}
    assert report['actual_reasons'] == {} and report['fallback'] is None
    assert [(entry['step'], entry['gain']) for entry in report['diffaid_gains']] == [(0, [gain])]
    assert_homogeneous(trace, anchors)
    # The gain record alone does not alter the trajectory or its forecasts.
    plain, plain_report, _ = run_stage(model, PASS[steps], monkeypatch=monkeypatch)
    torch.testing.assert_close(result, plain, atol=0, rtol=0)
    assert (plain_report['actual'], plain_report['forecast']) == (report['actual'], report['forecast'])


@pytest.mark.parametrize('window, steps, expected, transitions', [
    # Before the eligible step (during warmup): too few same-gain anchors remain.
    ((0.9, 1.0, 0.0), 4, 'AAAA', [1]),
    ((0.9, 1.0, 0.0), 5, 'AAAFA', [1]),
    # At the eligible step.
    ((0.75, 1.0, 0.0), 4, 'AAAA', [2]),
    # After it (at the tail step): the forecast was already served.
    ((0.5, 1.0, 0.0), 4, 'AAFA', [3]),
    ((0.5, 1.0, 0.3), 4, 'AAFA', [3]),
    # A wide ramp shoulder changes the gain at every step.
    ((0.9, 1.0, 0.4), 4, 'AAAA', [1, 2, 3]),
    ((0.9, 1.0, 0.4), 5, 'AAAAA', [1, 2, 3, 4]),
])
def test_transition_position_relative_to_the_eligible_step(
        model, monkeypatch, anchors, window, steps, expected, transitions):
    _, report, trace = run_stage(model, PASS[steps], window=window, monkeypatch=monkeypatch)
    assert pattern(trace) == expected and report['fallback'] is None
    assert [entry['step'] for entry in report['diffaid_gains']] == [0] + transitions
    assert report['actual_reasons'] == {TRANSITION: len(transitions)}
    for step in transitions:
        assert trace[step][0] == 'A' and trace[step][1] != trace[step - 1][1]
    assert_homogeneous(trace, anchors)


@pytest.mark.parametrize('reference', ['schedule_start', 1.0])
def test_full_schedule_is_identical_for_both_window_references(model, monkeypatch, anchors, reference):
    """A sampler call that starts at the model's maximum sigma crosses the
    boundary at the same step under either reference."""
    _, report, trace = run_stage(model, FULL[14], window=(0.75, 1.0, 0.0), reference=reference,
                                 monkeypatch=monkeypatch)
    assert pattern(trace) == 'AAFAFAFAAAFAFA'
    assert (report['actual'], report['forecast']) == (9, 5)
    assert report['actual_reasons'] == {TRANSITION: 1}
    assert [entry['step'] for entry in report['diffaid_gains']] == [0, 8]
    assert_homogeneous(trace, anchors)


# -- Sol attention routes ----------------------------------------------------

@pytest.mark.parametrize('dense, steps, expected, reasons', [
    (0, 4, 'AAFA', {}),
    (0, 5, 'AAFAA', {}),
    # The dense warmup evaluation and the first sparse one are different
    # operators: two sparse anchors are needed before a forecast.
    (1, 4, 'AAAA', {ROUTE: 1}),
    (1, 5, 'AAAFA', {ROUTE: 1}),
    (2, 4, 'AAAA', {ROUTE: 1}),
    (2, 5, 'AAAAA', {ROUTE: 1}),
])
def test_sol_dense_warmup_on_short_schedules(model, monkeypatch, anchors, dense, steps, expected, reasons):
    _, report, trace = run_stage(model, PASS[steps], policy=SolPolicy(dense), monkeypatch=monkeypatch)
    assert pattern(trace) == expected and report['fallback'] is None
    assert report['actual_reasons'] == reasons
    for step, used in anchors:
        # Anchors of a forecast are all sparse evaluations.
        assert all(sum(kind == 'A' for kind, _ in trace[:i]) >= dense for i in used)


# -- Spectrum controls ---------------------------------------------------------

@pytest.mark.parametrize('overrides, steps, expected, reasons, bootstrap', [
    ({'warmup': 3}, 4, 'AAAA', {}, 0),
    ({'warmup': 3}, 5, 'AAAFA', {}, 0),
    ({'tail': 2}, 4, 'AAAA', {}, 0),
    ({'tail': 2}, 5, 'AAFAA', {}, 0),
    ({'degree': 2, 'ridge': 0.0}, 4, 'AAAA', {}, 0),
    ({'window': 0}, 4, 'AAAA', {}, 0),
    ({'extrapolation_guard_enabled': True}, 4, 'AAAA', {GUARD: 1}, 0),
    ({'extrapolation_guard_enabled': True, 'max_extrapolation_span_ratio': 1.5}, 4, 'AAFA', {}, 0),
    # The one-point hold serves the second step and leaves one anchor for the third.
    ({'warmup': 1, 'bootstrap': True}, 4, 'AFAA', {}, 1),
    ({'warmup': 0, 'bootstrap': True}, 5, 'AFAFA', {}, 1),
])
def test_controls_on_short_schedules(model, monkeypatch, overrides, steps, expected, reasons, bootstrap):
    _, report, trace = run_stage(model, PASS[steps], {**TURBO, **overrides}, monkeypatch=monkeypatch)
    assert pattern(trace) == expected and report['fallback'] is None
    assert report['actual_reasons'] == reasons and report['bootstrap_forecast'] == bootstrap
    assert report['actual'] + report['forecast'] == report['outer_steps'] == steps


def test_bootstrap_hold_is_the_only_forecast_before_a_third_step_transition(model, monkeypatch, anchors):
    """With a gain change at the third step, only a one-anchor feature hold at
    the second, highest-noise step could forecast. It is never applied unless
    bootstrap and a warmup of at most one are selected explicitly."""
    held = {**TURBO, 'warmup': 1, 'bootstrap': True}
    _, report, trace = run_stage(model, PASS[4], held, window=(0.75, 1.0, 0.0), monkeypatch=monkeypatch)
    assert pattern(trace) == 'AFAA' and report['bootstrap_forecast'] == 1
    assert report['actual_reasons'] == {TRANSITION: 1} and anchors == []
    assert trace[0][1] == trace[1][1] == 1.0
    _, report, trace = run_stage(model, PASS[4], window=(0.75, 1.0, 0.0), monkeypatch=monkeypatch)
    assert pattern(trace) == 'AAAA' and report['bootstrap_forecast'] == 0


def test_all_actual_and_native_fallback_are_bitwise_native(model, monkeypatch):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    expected, report, _ = run_stage(model, PASS[4], {**TURBO, 'window': 0}, monkeypatch=monkeypatch)
    assert report['forecast'] == 0 and report['fallback'] is None
    # Plain native solver: the installed wrappers pass through without a session.
    generator = torch.Generator(device='cpu').manual_seed(99)
    x = torch.randn(1, model.channels, 5, 7, generator=generator)
    context = torch.randn(1, 3, model.txtlayers * model.txtdim, generator=generator)
    host = Owner(model, Binding(Config()), {}, context)
    sampler = KSAMPLER(sampling.sample_er_sde, extra_options={
        'max_stage': 3, 's_noise': 1.0, 'noise_scaler': ScalerERSDE(),
        'noise_sampler': SeededNoise(x.shape)})
    with torch.no_grad():
        native = sampler.sample(host, PASS[4], {'model_options': {}}, None, x, torch.zeros_like(x))
    torch.testing.assert_close(expected, native, atol=0, rtol=0)
    # An unqualified CFG function keeps the whole pass on the native path.
    fallback, report, trace = run_stage(model, PASS[4], window=(0.75, 1.0, 0.0), monkeypatch=monkeypatch,
                                        model_options={'sampler_cfg_function': lambda args: args['cond']})
    assert pattern(trace) == 'AAAA' and report['forecast'] == 0
    assert report['fallback'] == 'custom CFG, model wrapper or multi-device execution is active'
    assert report['actual_reasons'] == {} and report['diffaid_gains'] == []
    torch.testing.assert_close(fallback, native, atol=0, rtol=0)


def test_forecast_changes_only_the_forecast_step(model, monkeypatch):
    """Up to the first forecast the accelerated pass is the native pass."""
    states = {}
    def record(name):
        states[name] = []
        return lambda step, x0, x, total: states[name].append(x0.clone())
    run_stage(model, PASS[4], {**TURBO, 'window': 0}, monkeypatch=monkeypatch, callback=record('native'))
    run_stage(model, PASS[4], monkeypatch=monkeypatch, callback=record('forecast'))
    for step in (0, 1):
        torch.testing.assert_close(states['forecast'][step], states['native'][step], atol=0, rtol=0)
    assert not torch.equal(states['forecast'][2], states['native'][2])
    assert all(bool(torch.isfinite(value).all()) for value in states['forecast'])


# -- Conditioning streams, repeated and nested invocations ----------------------

@pytest.mark.parametrize('window, reference, expected', [
    (None, 'schedule_start', 'AAFA'),
    ((0.75, 1.0, 0.0), 'schedule_start', 'AAAA'),
    ((0.75, 1.0, 0.0), 1.0, 'AAFA'),
])
def test_separate_conditioning_streams_share_the_step_decision(
        model, monkeypatch, anchors, window, reference, expected):
    _, report, trace = run_stage(model, PASS[4], window=window, reference=reference, split=True,
                                 monkeypatch=monkeypatch)
    assert pattern(trace) == expected and report['fallback'] is None
    forecasts = expected.count('F')
    # Each outer step evaluates the conditional and unconditional stream.
    assert (report['actual'], report['forecast']) == (2 * (4 - forecasts), 2 * forecasts)
    # Each stream discards its own history at the gain change.
    assert report['actual_reasons'].get(TRANSITION, 0) == (2 if expected == 'AAAA' else 0)
    assert len(anchors) == 2 * forecasts and all(used == [0, 1] for _, used in anchors)


def test_consecutive_passes_of_one_model_start_without_stale_state(model, monkeypatch):
    """Full-schedule pass followed by refinement passes at growing sizes, all
    through one installed binding, then the whole sequence once more."""
    binding = Binding(Config(**TURBO))
    def sequence(window, reference):
        outputs = []
        outputs.append(run_stage(model, FULL[14], window=window, reference=reference, binding=binding,
                                 shape=(4, 6), monkeypatch=monkeypatch))
        for shape in ((5, 7), (6, 8), (8, 10)):
            outputs.append(run_stage(model, PASS[4], window=window, reference=reference, binding=binding,
                                     shape=shape, monkeypatch=monkeypatch))
        return outputs
    for reference, refinement in (('schedule_start', 'AAAA'), (1.0, 'AAFA')):
        first = sequence((0.75, 1.0, 0.0), reference)
        again = sequence((0.75, 1.0, 0.0), reference)
        assert [pattern(trace) for _, _, trace in first] == ['AAFAFAFAAAFAFA'] + [refinement] * 3
        for (a, report_a, trace_a), (b, report_b, trace_b) in zip(first, again):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            assert report_a == report_b and trace_a == trace_b
        for _, report, _ in first[1:]:
            # Nothing of the preceding pass is visible in a later one.
            assert report['outer_steps'] == 4 and report['diffaid_gains'][0]['step'] == 0
            assert report['peak_history_bytes'] > 0


@pytest.mark.parametrize('failure', ['callback', 'model'])
def test_interrupted_pass_leaves_no_state_for_the_next(model, monkeypatch, failure):
    binding = Binding(Config(**TURBO))
    kwargs = dict(window=(0.75, 1.0, 0.0), reference=1.0, binding=binding, monkeypatch=monkeypatch)
    expected, expected_report, expected_trace = run_stage(model, PASS[5], **kwargs)

    class Interrupted(Exception):
        pass

    if failure == 'callback':
        # Cancellation is raised from the per-step callback, after the first forecast.
        def interrupt(step, x0, x, total):
            if step == 2:
                raise Interrupted
        with pytest.raises(Interrupted):
            run_stage(model, PASS[5], callback=interrupt, **kwargs)
    else:
        calls = {'count': 0}
        original = type(model.blocks[0]).forward
        def failing(self, *args, **kw):
            calls['count'] += 1
            if calls['count'] == 2 * len(model.blocks) + 1:
                raise Interrupted
            return original(self, *args, **kw)
        with monkeypatch.context() as scoped:
            scoped.setattr(type(model.blocks[0]), 'forward', failing)
            with pytest.raises(Interrupted):
                run_stage(model, PASS[5], **kwargs)
    assert CURRENT.get() is None and not model.last._forward_pre_hooks
    result, report, trace = run_stage(model, PASS[5], **kwargs)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert report == expected_report and trace == expected_trace


def test_nested_pass_in_a_callback_is_independent(model, monkeypatch):
    kwargs = dict(window=(0.75, 1.0, 0.0), reference=1.0, monkeypatch=monkeypatch)
    alone, alone_report, alone_trace = run_stage(model, PASS[4], **kwargs)
    outer_alone, outer_alone_report, outer_alone_trace = run_stage(model, PASS[5], shape=(6, 8), **kwargs)
    nested = []
    def callback(step, x0, x, total):
        if step == 1:
            # Between the outer pass's two warmup anchors and its forecast.
            nested.append(run_stage(model, PASS[4], **kwargs))
    outer, outer_report, outer_trace = run_stage(model, PASS[5], shape=(6, 8), callback=callback, **kwargs)
    (inner, inner_report, inner_trace), = nested
    torch.testing.assert_close(inner, alone, atol=0, rtol=0)
    torch.testing.assert_close(outer, outer_alone, atol=0, rtol=0)
    assert inner_report == alone_report and inner_trace == alone_trace
    assert outer_report == outer_alone_report and outer_trace == outer_alone_trace
    assert pattern(outer_trace) == 'AAFAA' and pattern(inner_trace) == 'AAFA'


def test_gain_report_is_bounded(model, monkeypatch):
    from comfyui_spectrum_krea2.runtime import DIFFAID_GAIN_REPORT_LIMIT
    sigmas = torch.linspace(0.99, 0.0, 101)
    _, report, trace = run_stage(model, sigmas, window=(1.0, 1.0, 1.0), reference=1.0,
                                 monkeypatch=monkeypatch, stochastic=False)
    assert len(report['diffaid_gains']) == DIFFAID_GAIN_REPORT_LIMIT < len({g for _, g in trace})
