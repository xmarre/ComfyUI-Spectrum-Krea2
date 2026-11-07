import contextlib
from functools import partial
import copy
import dataclasses
import types
import uuid
import pytest
import torch
from native_components import tiny_model
from comfyui_spectrum_krea2.adapter import diffusion, project, CAPTURE
from comfyui_spectrum_krea2.compat import model_contract, verified
from comfyui_spectrum_krea2.config import Config
from comfyui_spectrum_krea2.forecast import ForecastUnavailable
from comfyui_spectrum_krea2.runtime import CURRENT, REPORTS, Session
from comfyui_spectrum_krea2.sampling import Binding


def inputs(model, batch=1, shape=(5,7), temporal=False):
    x=torch.randn(batch,model.channels,*shape,dtype=model.dtype)
    if temporal:
        x=x.unsqueeze(2)
    return x,torch.full((batch,),.7),torch.randn(batch,3,model.txtlayers*model.txtdim,dtype=model.dtype)


def setup_session(model,x,config=None):
    binding=Binding(config or Config())
    session=Session(binding,binding.config,[-1.,-.6,-.2,.2,.6],model,'weights',x.shape[0],tuple(x.shape[-2:]))
    session.weight_owner=types.SimpleNamespace(current_weight_patches_uuid='weights')
    session.active_step=0
    return binding,session


def call(model,binding,x,t,context,refs=None,options=None,**kw):
    from comfy.patcher_extension import WrapperExecutor
    options=options or {'uuids':[uuid.UUID(int=1)],'cond_or_uncond':[0]}
    executor=WrapperExecutor.new_class_executor(model._forward,model,[partial(diffusion,binding)])
    return executor.execute(x,t,context,ref_latents=refs,transformer_options=options,**kw)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('shape',[(4,6),(5,7),(6,4)])
@pytest.mark.parametrize('temporal',[False,True])
@pytest.mark.parametrize('ref_method',[None,'index','index_timestep_zero'])
def test_actual_native_parity_and_exact_image_projection(native,dtype,shape,temporal,ref_method):
    model=tiny_model(native[0],dtype)
    assert model_contract(model)
    x,t,context=inputs(model,shape=shape,temporal=temporal)
    refs=None if ref_method is None else [torch.randn(1,2,4,6,dtype=dtype)]
    binding,session=setup_session(model,x)
    with torch.no_grad():
        expected=model(x,t,context,ref_latents=refs,ref_latents_method=ref_method)
        rng=torch.random.get_rng_state().clone()
        token=CURRENT.set(session)
        try:
            actual=call(model,binding,x,t,context,refs,ref_latents_method=ref_method)
        finally:
            CURRENT.reset(token)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        assert torch.equal(rng,torch.random.get_rng_state())
        stream=next(iter(session.streams.values()))
        feature=stream.history.features[-1]
        assert feature.shape==(1,((shape[0]+1)//2)*((shape[1]+1)//2),32)
        projected=project(model,feature,t,x)
        torch.testing.assert_close(projected,expected,atol=1e-6 if dtype==torch.float32 else .001,rtol=1e-4 if dtype==torch.float32 else .01)
    assert len(model.last._forward_pre_hooks)==0
    assert CAPTURE.get() is None


class Owner:
    def __init__(self,model,binding,options,context,split=False):
        self.binding=binding
        self.options=options
        self.context=context
        self.split=split
        self.model_patcher=types.SimpleNamespace(
            model=types.SimpleNamespace(),
            get_model_object=lambda name: types.SimpleNamespace(noise_scale=1.0),
        )
        self.inner_model=types.SimpleNamespace(diffusion_model=model,current_weight_patches_uuid='weights',
             model_sampling=types.SimpleNamespace(noise_scaling=lambda sigma,noise,latent,max_denoise: noise,
                                                 inverse_noise_scaling=lambda sigma,x:x))
        self.conds={'positive':[],'negative':[]}
        self.cfg=3.5 if split else 1.

    def __call__(self,x,t,model_options,seed):
        from comfy.patcher_extension import WrapperExecutor
        executor=WrapperExecutor.new_class_executor(self.predict,self,[self.binding.predict])
        return executor.execute(x,t,model_options,seed)

    def predict(self,x,t,model_options,seed):
        branches=(0,1) if self.split else (0,)
        out=[]
        for branch in branches:
            opts={**self.options,'uuids':[uuid.UUID(int=branch+1)],'cond_or_uncond':[branch]}
            out.append(call(self.inner_model.diffusion_model,self.binding,x,t,self.context+branch,options=opts))
        return out[0] if not self.split else out[1]+self.cfg*(out[0]-out[1])


def run(model, config=None, steps=12, split=False, sigmas=None, options=None, callback=None,sampler_name='sample_euler',sampler_options=None):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfy.patcher_extension import WrapperExecutor
    binding=Binding(config or Config())
    x,t,context=inputs(model)
    options=options or {}
    owner=Owner(model,binding,options,context,split)
    sampler=KSAMPLER(getattr(sampling,sampler_name),extra_options=sampler_options or {})
    sigmas=torch.linspace(1.,0.,steps+1) if sigmas is None else sigmas
    executor=WrapperExecutor.new_class_executor(sampler.sample,sampler,[binding.sampler])
    reports=[]
    token=REPORTS.set(reports)
    try:
        with torch.no_grad():
            result=executor.execute(owner,sigmas,{'model_options':{'transformer_options':options}},callback,x,torch.zeros_like(x))
    finally:
        REPORTS.reset(token)
    return result,reports[0]


@pytest.mark.parametrize('sampler_name',[
    'sample_euler', 'sample_dpmpp_2m',
    'sample_res_multistep', 'sample_gradient_estimation',
    'sample_deis', 'sample_ipndm', 'sample_ipndm_v', 'sample_er_sde',
])
@pytest.mark.parametrize('split',[False,True])
def test_real_sampler_forecasts_and_fresh_runs(native,sampler_name,split):
    model=tiny_model(native[0])
    _,report=run(model,split=split,sampler_name=sampler_name)
    assert report['outer_steps']==12
    assert report['forecast']>0 and report['actual']>=6
    assert report['actual']+report['forecast']==12*(2 if split else 1)
    assert report['fallback'] is None
    _,second=run(model,split=split,sampler_name=sampler_name)
    assert second['actual']==report['actual'] and second['forecast']==report['forecast']
    assert not model.last._forward_pre_hooks and CURRENT.get() is None


@pytest.mark.parametrize('reason,options',[('external',{'patches':{'post_input':[]}}),('external',{'wrappers':{'diffusion_model':{'other':[lambda *a:None]}}})])
def test_unsupported_options_are_bitwise_native(native,reason,options):
    model=tiny_model(native[0])
    torch.manual_seed(123)
    expected,baseline=run(model,Config(window=0),options=options)
    torch.manual_seed(123)
    actual,report=run(model,options=options)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert report['forecast']==0 and reason in report['fallback']


@pytest.mark.parametrize('sigmas',[torch.tensor([1.,.8,.8,.4,0.]),torch.tensor([1.,.5,.6,0.])])
def test_duplicate_and_nonmonotonic_sigmas_preserve_native(native,sigmas):
    model=tiny_model(native[0])
    torch.manual_seed(99)
    baseline,_=run(model,Config(window=0),sigmas=sigmas)
    torch.manual_seed(99)
    actual,report=run(model,sigmas=sigmas)
    torch.testing.assert_close(actual,baseline,atol=0,rtol=0,equal_nan=True)
    assert report['forecast']==0 and 'strictly decreasing' in report['fallback']


def test_exception_cleans_context_and_hook_and_allows_reuse(native):
    model=tiny_model(native[0])
    def fail(*args):
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):
        run(model,callback=fail)
    assert CURRENT.get() is None and CAPTURE.get() is None
    assert not model.last._forward_pre_hooks
    _,report=run(model)
    assert report['forecast']>0


def test_native_error_propagates_and_hook_cleanup(native,monkeypatch):
    model=tiny_model(native[0])
    x,t,c=inputs(model)
    b,s=setup_session(model,x)
    def fail(*args,**kw):
        raise ValueError('native failure')
    monkeypatch.setattr(model.blocks[0],'forward',fail)
    token=CURRENT.set(s)
    try:
        with torch.no_grad(),pytest.raises(ValueError,match='native failure'):
            call(model,b,x,t,c)
    finally:
        CURRENT.reset(token)
    assert not model.last._forward_pre_hooks
    assert CAPTURE.get() is None


@pytest.mark.parametrize('changed',['uuid','size','dtype','weight','missing','patch','gradient','batch','reference'])
def test_call_identity_changes_do_not_reuse_history(native,changed):
    model=tiny_model(native[0])
    x,t,c=inputs(model)
    binding,session=setup_session(model,x,Config(warmup=0,tail=1,ridge=0,bootstrap=False))
    opts={'uuids':[uuid.UUID(int=1)],'cond_or_uncond':[0]}
    token=CURRENT.set(session)
    try:
        with torch.no_grad():
            for step in range(2):
                session.active_step=step
                call(model,binding,x,t,c,options=opts)
            session.active_step=2
            if changed=='uuid': opts['uuids']=[uuid.UUID(int=2)]
            if changed=='missing': opts['uuids']=[]
            if changed=='size': x=x[:,:,:3,:]
            if changed=='batch': x=x.repeat(2,1,1,1); t=t.repeat(2); c=c.repeat(2,1,1)
            if changed=='dtype': x=x.to(torch.bfloat16); c=c.to(torch.bfloat16); model.to(torch.bfloat16)
            if changed=='weight': session.weight_owner.current_weight_patches_uuid='new'
            if changed=='patch': opts['patches']={'attn1_patch':[]}
            refs=[torch.ones(1,2,4,4)] if changed=='reference' else None
            with torch.set_grad_enabled(changed=='gradient'):
                call(model,binding,x,t,c,refs,options=opts)
        assert session.counts['forecast']==0
        assert session.counts['actual']==3
    finally:
        CURRENT.reset(token)


def test_forecast_failure_retries_only_native(native,monkeypatch):
    from comfyui_spectrum_krea2.forecast import History
    model=tiny_model(native[0])
    monkeypatch.setattr(History,'predict',lambda *a: (_ for _ in ()).throw(ForecastUnavailable('bad prediction')))
    _,report=run(model)
    assert report['forecast']==0 and report['actual']==12
    assert any('bad prediction' in k for k in report['actual_reasons'])
    assert not model.last._forward_pre_hooks


def test_concurrent_contexts_ignore_other_capture_hooks(native):
    # A copied context simulates a second invocation on the same shared module.
    model=tiny_model(native[0])
    x,t,c=inputs(model)
    binding,session=setup_session(model,x)
    foreign=[]
    marker=object()
    handle=model.last.register_forward_pre_hook(lambda m,a: foreign.append(a[0]) if CAPTURE.get() is marker else None)
    token=CURRENT.set(session)
    try:
        with torch.no_grad():
            out=call(model,binding,x,t,c)
        assert not foreign and session.counts['forecast']==0
    finally:
        CURRENT.reset(token)
        handle.remove()


def test_nested_sampler_in_callback_has_independent_session(native):
    model=tiny_model(native[0])
    nested=[]
    def callback(step,*args):
        assert CURRENT.get() is None
        if step==0:
            _,report=run(model,steps=8)
            nested.append(report)
    _,report=run(model,callback=callback)
    assert report['actual']+report['forecast']==12
    assert nested[0]['actual']+nested[0]['forecast']==8
    assert report['forecast']>0 and nested[0]['forecast']>0
    assert CURRENT.get() is None


def test_failed_native_call_is_not_counted_as_completed(native,monkeypatch):
    model=tiny_model(native[0])
    x,t,c=inputs(model)
    binding,session=setup_session(model,x)
    session.disable('native test')
    def fail(*a,**kw):
        raise ValueError('native failure')
    monkeypatch.setattr(model.blocks[0],'forward',fail)
    token=CURRENT.set(session)
    try:
        with torch.no_grad(),pytest.raises(ValueError,match='native failure'):
            call(model,binding,x,t,c)
    finally:
        CURRENT.reset(token)
    assert session.counts['actual']==0


def test_turbo_early_warmup_builds_history_and_forecasts(native):
    model=tiny_model(native[0])
    _,report=run(model,Config(preset='turbo',warmup=1,tail=1),steps=8)
    assert report['fallback'] is None and report['forecast']>0
    assert report['actual']+report['forecast']==8
    assert report['effective_config']['warmup']==1
    assert report['effective_config']['tail']==1


@pytest.mark.parametrize('nonfinite',[False,True])
def test_prediction_failure_retains_only_valid_actual_history(native,monkeypatch,nonfinite):
    from comfyui_spectrum_krea2.forecast import History, ForecastNonfinite
    model=tiny_model(native[0])
    x,t,c=inputs(model)
    binding,session=setup_session(model,x,Config(warmup=0,tail=1,bootstrap=False))
    token=CURRENT.set(session)
    try:
        with torch.no_grad():
            for step in range(2):
                session.active_step=step
                call(model,binding,x,t,c)
            stream=next(iter(session.streams.values()))
            assert len(stream.history.features)==2
            def reject(*a,**kw):
                raise (ForecastNonfinite if nonfinite else ForecastUnavailable)('rejected test prediction')
            monkeypatch.setattr(History,'predict',reject)
            session.active_step=2
            call(model,binding,x,t,c)
            assert len(stream.history.features)==(1 if nonfinite else 3)
            assert session.counts['actual']==3
    finally:
        CURRENT.reset(token)


def test_replaced_sampler_entrypoint_is_rejected(native):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfyui_spectrum_krea2.sampling import validate_sampler
    sampler=KSAMPLER(sampling.sample_euler)
    executor=types.SimpleNamespace(class_obj=sampler,original=lambda *a:None)
    reason=validate_sampler(executor,None,torch.linspace(1,0,5),{})
    assert 'entrypoint' in reason


def test_non_native_sampler_container_is_identified(native):
    from comfyui_spectrum_krea2.sampling import validate_sampler
    executor=types.SimpleNamespace(class_obj=object())
    reason=validate_sampler(executor,None,torch.linspace(1,0,5),{})
    assert reason=='unsupported sampler container: builtins.object'


@pytest.mark.parametrize('variant',['optional','renamed','variadic'])
def test_sampler_signature_variants_preserve_native_and_allow_forecasting(native,monkeypatch,variant):
    from comfy.samplers import KSAMPLER
    from sampler_variants import wrap_sample
    model=tiny_model(native[0])
    torch.manual_seed(42)
    expected,_=run(model,Config(window=0))
    monkeypatch.setattr(KSAMPLER,'sample',wrap_sample(KSAMPLER.sample,variant))
    torch.manual_seed(42)
    actual,_=run(model,Config(window=0))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    _,report=run(model,Config(preset='turbo',warmup=1,tail=1))
    assert report['fallback'] is None and report['forecast']>0
    assert report['actual']+report['forecast']==12


def test_signature_introspection_failure_only_affects_diagnostics(native,monkeypatch,caplog):
    import logging
    from comfy.samplers import KSAMPLER
    from sampler_variants import wrap_sample
    wrapped=wrap_sample(KSAMPLER.sample,'variadic')
    wrapped.__signature__='invalid introspection metadata'
    monkeypatch.setattr(KSAMPLER,'sample',wrapped)
    model=tiny_model(native[0])
    with caplog.at_level(logging.INFO):
        _,report=run(model,Config(preset='turbo',warmup=1,tail=1,debug=True))
    assert report['fallback'] is None and report['forecast']>0
    assert 'signature=<unavailable>' in caplog.text


def test_bootstrap_holds_first_actual_features_at_second_step(native):
    model=tiny_model(native[0])
    x,t0,c=inputs(model)
    binding,session=setup_session(model,x,Config(warmup=1,tail=1))
    token=CURRENT.set(session)
    try:
        with torch.no_grad():
            session.active_step=0
            call(model,binding,x,t0,c)
            stream=next(iter(session.streams.values()))
            held=stream.history.features[0].clone()
            session.active_step=1
            t1=torch.full_like(t0,.55)
            output=call(model,binding,x,t1,c)
            # The held transformer features are projected with the current timestep.
            torch.testing.assert_close(output,project(model,held,t1,x),atol=0,rtol=0)
            assert session.counts['forecast']==1 and session.counts['bootstrap']==1
            assert len(stream.history.features)==1  # forecasts never enter history
            session.active_step=2
            call(model,binding,x,t1,c)
            assert session.counts['actual']==2
    finally:
        CURRENT.reset(token)


def test_bootstrap_adds_the_second_step_forecast_to_short_turbo_runs(native):
    model=tiny_model(native[0])
    _,without=run(model,Config(preset='turbo',warmup=1,tail=1,bootstrap=False),steps=8)
    _,with_bootstrap=run(model,Config(preset='turbo',warmup=1,tail=1),steps=8)
    assert with_bootstrap['fallback'] is None
    assert with_bootstrap['bootstrap_forecast']==1 and without['bootstrap_forecast']==0
    assert with_bootstrap['effective_config']['bootstrap'] is True
    assert with_bootstrap['forecast']>=without['forecast']+1
    assert with_bootstrap['actual']+with_bootstrap['forecast']==8



@pytest.mark.parametrize("route_sequence, expected_reasons", [
    # Predicted and observed dense->Sol change is one transition event.
    (("dense", "sol", "sol"), {"numerical attention backend transition": 1}),
    # A routing prediction that fails remains observable as a distinct event.
    (("dense", "dense"), {
        "numerical attention backend transition": 1,
        "numerical attention route differed from the previous anchor": 1,
    }),
])
def test_sol_attention_route_history_invalidation_is_single_event(
        native, monkeypatch, route_sequence, expected_reasons):
    import comfyui_spectrum_krea2.adapter as adapter

    model = tiny_model(native[0])
    x, t, context = inputs(model)
    binding, session = setup_session(
        model, x, Config(preset="turbo", warmup=2, tail=1, bootstrap=False),
    )
    # This test is about the adapter's route reconciliation, not the
    # qualification of a separately installed attention patch.
    monkeypatch.setattr(adapter, "matches", lambda policy, options: True)

    class SimulatedSol:
        def phase(self, x, context, refs, inner):
            return "dense" if session.counts["actual"] == 0 else "sol"

        def sol_status(self):
            count = session.counts["actual"]
            # First native call is dense. Later calls may or may not actually
            # use Sol; the mismatch case intentionally forces a dense result.
            realized = route_sequence[min(count, len(route_sequence) - 1)]
            return {
                "actual_evaluations": count,
                "sparse_calls": sum(1 for route in route_sequence[:count] if route == "sol"),
            }

    session.patch_policy = SimulatedSol()
    key = {"uuids": [uuid.UUID(int=1)], "cond_or_uncond": [0]}
    token = CURRENT.set(session)
    try:
        with torch.no_grad():
            for step in range(len(route_sequence)):
                session.active_step = step
                call(model, binding, x, t, context, options=key)
                stream = next(iter(session.streams.values()))
                # Never retain features from the old numerical attention
                # route once the first Sol actual has been witnessed.
                if step == 1:
                    assert len(stream.history.features) == 1
                    assert stream.actual_anchors == 1
            assert session.counts["actual"] == len(route_sequence)
            for reason, count in expected_reasons.items():
                assert session.reasons[reason] == count
            assert sum(
                count for reason, count in session.reasons.items()
                if reason.startswith("numerical attention")
            ) == sum(expected_reasons.values())
            assert session.disabled_reason is None
    finally:
        CURRENT.reset(token)
    assert not model.last._forward_pre_hooks



@pytest.mark.parametrize('remaining,expected_default,expected_relaxed', [
    (7, 1, 2),
    (8, 2, 3),
])
def test_krea_simple_high_resolution_schedule_guard_toggle(
        native, remaining, expected_default, expected_relaxed):
    # This is the native 18-step simple-scheduler tail used by
    # composition-lock handoffs, at 0.70 (7) and 0.72 (8).
    # ComfyUI's ModelSamplingFlux uses mu=1.15 and a 10,000-element
    # shifted sigma table. The native 'simple' scheduler reads table entries
    # from the end at int(i * 10000/18). Reproduce those exact FP32 values
    # here without importing unrelated ComfyUI host dependencies.
    import math
    shift = math.exp(1.15)
    full = []
    for i in range(18):
        t = (10000 - int(i * (10000 / 18))) / 10000
        full.append(shift * t / (1 - t + shift * t))
    full = torch.tensor(full + [0.], dtype=torch.float32)
    sigmas = full[-(remaining + 1):]
    model = tiny_model(native[0])
    baseline_config = dict(preset='turbo', warmup=2, tail=1,
                           window=1, bootstrap=False)
    reports = {}
    for name, overrides in (
        ('default', {}),
        ('relaxed', {'max_extrapolation_span_ratio': 1.5}),
        ('disabled', {'extrapolation_guard_enabled': False}),
    ):
        _, report = run(model, Config(**baseline_config, **overrides),
                        steps=remaining, sigmas=sigmas)
        assert report['fallback'] is None, report
        assert report['outer_steps'] == remaining, report
        assert report['actual'] + report['forecast'] == remaining, report
        assert report['bootstrap_forecast'] == 0, report
        assert report['effective_config']['extrapolation_guard_enabled'] == (
            name != 'disabled')
        assert report['effective_config']['max_extrapolation_span_ratio'] == (
            1.5 if name == 'relaxed' else 0.5)
        assert report['actual_reasons'].get('numerical attention backend transition', 0) == 0
        reports[name] = report
    assert reports['default']['forecast'] == expected_default, reports
    assert reports['relaxed']['forecast'] == expected_relaxed, reports
    assert reports['disabled']['forecast'] == expected_relaxed, reports
    assert reports['default']['actual_reasons'].get(
        'forecast rejected: extrapolation exceeds half the retained history span') == 2
    for name in ('relaxed', 'disabled'):
        assert not any('extrapolation exceeds' in reason
                       for reason in reports[name]['actual_reasons'])



@pytest.mark.parametrize('sampler_name', ['sample_res_multistep', 'sample_gradient_estimation',
    'sample_deis', 'sample_ipndm', 'sample_ipndm_v'])
def test_new_solver_exact_native_parity_when_forecasting_disabled(native, sampler_name):
    model = tiny_model(native[0])
    torch.manual_seed(31415)
    original, _ = run(model, Config(window=0), steps=20, sampler_name=sampler_name)
    torch.manual_seed(31415)
    actual, report = run(model, Config(preset='turbo', warmup=2, tail=1,
                                      window=0, extrapolation_guard_enabled=False),
                         steps=20, sampler_name=sampler_name)
    torch.testing.assert_close(actual, original, atol=0, rtol=0)
    assert report['fallback'] is None and report['forecast'] == 0
    assert report['actual'] == 20


@pytest.mark.parametrize('sampler_name,options', [
    ('sample_res_multistep', {'noise_sampler': object()}),
    ('sample_res_multistep', {'s_noise': 2.0}),
    ('sample_gradient_estimation', {'cfg_pp': True}),
    ('sample_gradient_estimation', {'ge_gamma': 1.0}),
    ('sample_deis', {'max_order': 2}),
    ('sample_deis', {'deis_mode': 'rhoab'}),
    ('sample_ipndm', {'max_order': 2}),
    ('sample_ipndm_v', {'max_order': 3}),
    ('sample_er_sde', {'max_stage': 4}),
    ('sample_er_sde', {'s_noise': float('nan')}),
    ('sample_er_sde', {'noise_scaler': 'invalid'}),
])
def test_solver_parameter_overrides_are_native_fallback(native, sampler_name, options):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfyui_spectrum_krea2.sampling import validate_sampler
    sampler = KSAMPLER(getattr(sampling, sampler_name), extra_options=options)
    executor = types.SimpleNamespace(class_obj=sampler, original=sampler.sample)
    reason = validate_sampler(executor, None, torch.linspace(1, 0, 4),
                              {'model_options': {}})
    expected = 'unverified ER-SDE options' if sampler_name == 'sample_er_sde' else 'unverified sampler options'
    assert reason == expected


def test_res_delegate_source_mismatch_disables_forecasting(native, monkeypatch):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfyui_spectrum_krea2.sampling import validate_sampler
    sampler = KSAMPLER(sampling.sample_res_multistep)
    executor = types.SimpleNamespace(class_obj=sampler, original=sampler.sample)
    monkeypatch.setattr(sampling, 'res_multistep', lambda *args, **kw: None)
    assert validate_sampler(executor, None, torch.linspace(1, 0, 4),
                            {'model_options': {}}) == 'native RES Multistep solver source contract changed'


@pytest.mark.parametrize('sampler_name', ['sample_res_multistep', 'sample_gradient_estimation',
    'sample_deis', 'sample_ipndm', 'sample_ipndm_v'])
def test_new_solver_20_step_turbo_forecast_counts(native, sampler_name):
    model = tiny_model(native[0])
    config = Config(preset='turbo', warmup=2, tail=1,
                    extrapolation_guard_enabled=False, bootstrap=False)
    _, report = run(model, config, steps=20, sampler_name=sampler_name)
    assert report['fallback'] is None, report
    assert report['actual'] == 11 and report['forecast'] == 9, report
    _, second = run(model, config, steps=20, sampler_name=sampler_name)
    assert second['actual'] == report['actual']
    assert second['forecast'] == report['forecast']
    assert CURRENT.get() is None and not model.last._forward_pre_hooks



def test_legacy_native_without_new_solvers_remains_compatible(native, monkeypatch):
    from comfy.k_diffusion import sampling
    for name in ("sample_res_multistep", "sample_gradient_estimation",
                 "res_multistep", "sample_deis", "sample_ipndm", "sample_ipndm_v"):
        monkeypatch.delattr(sampling, name, raising=False)
    model = tiny_model(native[0])
    _, report = run(model, Config(preset='turbo', warmup=2, tail=1,
                                  extrapolation_guard_enabled=False),
                    steps=12, sampler_name="sample_euler")
    assert report['fallback'] is None, report
    assert report['forecast'] > 0



@pytest.mark.parametrize("helper_name", [
    "get_deis_coeff_list", "edm2t", "cal_poly", "cal_intergrand", "t2alpha_fn",
])
def test_deis_coefficient_contract_mismatch_fails_closed(native, monkeypatch, helper_name):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfyui_spectrum_krea2.sampling import validate_sampler
    monkeypatch.setattr(sampling.deis, helper_name, lambda *a, **kw: None)
    sampler = KSAMPLER(sampling.sample_deis)
    executor = types.SimpleNamespace(class_obj=sampler, original=sampler.sample)
    reason = validate_sampler(executor, None, torch.linspace(1, 0, 4),
                              {"model_options": {}})
    assert reason == "native DEIS coefficient source contract changed"


@pytest.mark.parametrize("sampler_name", ["sample_deis", "sample_ipndm", "sample_ipndm_v"])
def test_alternative_multistep_solvers_20_step_determinism_and_parity(native, sampler_name):
    model = tiny_model(native[0])
    cfg = Config(preset="turbo", warmup=2, tail=1, bootstrap=False,
                 extrapolation_guard_enabled=False)
    torch.manual_seed(724)
    output, report = run(model, cfg, steps=20, sampler_name=sampler_name)
    assert report["fallback"] is None, report
    assert (report["actual"], report["forecast"]) == (11, 9), report
    torch.manual_seed(724)
    repeat, repeated = run(model, cfg, steps=20, sampler_name=sampler_name)
    torch.testing.assert_close(repeat, output, atol=0, rtol=0)
    assert repeated["actual"] == 11 and repeated["forecast"] == 9
    assert not model.last._forward_pre_hooks and CURRENT.get() is None



def test_er_sde_native_stochastic_history_and_ode_options(native):
    model = tiny_model(native[0])
    cfg = Config(preset='turbo', warmup=2, tail=1, bootstrap=False,
                 extrapolation_guard_enabled=False)
    for opts in (
        {'max_stage': 3, 's_noise': 1.0},
        {'max_stage': 2, 's_noise': 0.0, 'noise_scaler': lambda t: t},
    ):
        torch.manual_seed(123456)
        first, report = run(model, cfg, steps=20, sampler_name='sample_er_sde',
                            sampler_options=opts)
        assert report['fallback'] is None, report
        assert report['actual'] + report['forecast'] == 20
        assert report['forecast'] > 0
        torch.manual_seed(123456)
        again, second = run(model, cfg, steps=20, sampler_name='sample_er_sde',
                            sampler_options=opts)
        torch.testing.assert_close(first, again, atol=0, rtol=0)
        assert second['actual'] == report['actual']
        assert second['forecast'] == report['forecast']
        torch.manual_seed(123456)
        original, _ = run(model, Config(window=0), steps=20, sampler_name='sample_er_sde',
                          sampler_options=opts)
        torch.manual_seed(123456)
        native_path, native_report = run(model, Config(window=0), steps=20,
                                        sampler_name='sample_er_sde', sampler_options=opts)
        torch.testing.assert_close(original, native_path, atol=0, rtol=0)
        assert native_report['forecast'] == 0 and native_report['fallback'] is None
    assert CURRENT.get() is None and not model.last._forward_pre_hooks


@pytest.mark.parametrize('helper', ['sigma_to_half_log_snr', 'offset_first_sigma_for_snr'])
def test_er_sde_sigma_contract_mismatch_disables_forecasting(native, monkeypatch, helper):
    from comfy.samplers import KSAMPLER
    from comfy.k_diffusion import sampling
    from comfyui_spectrum_krea2.sampling import validate_sampler
    sampler = KSAMPLER(sampling.sample_er_sde)
    executor = types.SimpleNamespace(class_obj=sampler, original=sampler.sample)
    monkeypatch.setattr(sampling, helper, lambda *a, **kw: None)
    assert validate_sampler(executor, None, torch.linspace(1, 0, 5),
                            {'model_options': {}}) == 'native ER-SDE sigma conversion source contract changed'
