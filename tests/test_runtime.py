import concurrent.futures
import dataclasses
import types
import pytest
import torch
from comfyui_spectrum_krea2.config import Config
from comfyui_spectrum_krea2.runtime import CURRENT, Session
from comfyui_spectrum_krea2.sampling import Binding


def session(config=None):
    config=config or Config()
    b=Binding(config)
    return Session(b,b.config,[-1.+i*.1 for i in range(20)],object(),None,1,(4,4))


def test_shared_history_budget_and_eviction():
    s=session(Config(budget_mb=64))
    s.active_step=0
    a=s.stream('positive')
    b=s.stream('negative')
    feature=torch.ones(1,1024,4096)  # 16 MiB
    for step in range(6):
        s.active_step=step
        for stream in (a,b):
            s.observe(stream,feature.clone())
        assert s.nbytes<=64*1024*1024
    assert s.peak_history_bytes<=64*1024*1024
    assert len(a.history.features)+len(b.history.features)==4


def test_budget_too_small_for_fit_stays_actual():
    s=session(Config(budget_mb=64,degree=4))
    stream=s.stream('positive')
    s.active_step=0
    s.observe(stream,torch.ones(1,1024,4096))
    assert not stream.history.features
    assert 'history budget' in next(iter(s.reasons))


def test_condition_stream_limit_is_bounded():
    s=session()
    for i in range(16): assert s.stream(i) is not None
    assert s.stream(17) is None
    assert not s.streams and s.disabled_reason


def test_windows_warmup_tail_and_repeated_branch():
    s=session(Config(warmup=4,tail=2,window=1))
    a=s.stream('positive')
    for step in range(4):
        s.active_step=step
        assert not s.should_forecast(a)
        s.observe(a,torch.ones(1,2,3))
        a.last_step=step
    s.active_step=4
    assert s.should_forecast(a)
    a.last_step=4
    assert not s.should_forecast(a)
    s.active_step=5
    assert not s.should_forecast(a)
    s.active_step=18
    assert not s.should_forecast(a)


def test_context_lifetime_nested_and_thread_isolation():
    outer=session()
    token=CURRENT.set(outer)
    def thread_run():
        assert CURRENT.get() is None
        inner=session()
        t=CURRENT.set(inner)
        try:
            assert CURRENT.get() is inner
        finally:
            CURRENT.reset(t)
        return CURRENT.get()
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            assert pool.submit(thread_run).result() is None
        inner=session()
        t=CURRENT.set(inner)
        assert CURRENT.get() is inner
        CURRENT.reset(t)
        assert CURRENT.get() is outer
    finally:
        CURRENT.reset(token)
    assert CURRENT.get() is None


@pytest.mark.parametrize('changes,expected',[
    ({},'bootstrap'),({'warmup':0},'bootstrap'),({'bootstrap':False},None),({'warmup':2},None),
    ({'degree':2},None),({'window':0},None),({'preset':'turbo'},'bootstrap'),
])
def test_one_point_bootstrap_eligibility(changes,expected):
    s=session(Config(**{'warmup':1,'tail':1,**changes}))
    a=s.stream('positive')
    s.active_step=0
    assert s.forecast_mode(a) is None
    s.observe(a,torch.ones(1,2,3))
    a.last_step=0
    s.active_step=1
    assert s.forecast_mode(a)==expected
    if expected:
        a.last_step=1
        assert s.forecast_mode(a) is None  # one forecast per outer step
        s.active_step=2
        assert s.forecast_mode(a) is None  # a fit needs a second actual anchor


def test_bootstrap_respects_tail_and_requires_step_zero_anchor():
    s=Session(*(lambda b:(b,b.config))(Binding(Config(warmup=1,tail=1))),[-1.,0.],object(),None,1,(4,4))
    a=s.stream('positive')
    s.active_step=0
    s.observe(a,torch.ones(1,2,3)); a.last_step=0
    s.active_step=1
    assert s.forecast_mode(a) is None  # step 1 is the final actual tail
    s=session(Config(warmup=1,tail=1))
    a=s.stream('positive')
    s.active_step=1
    assert s.forecast_mode(a) is None  # no actual anchor yet
