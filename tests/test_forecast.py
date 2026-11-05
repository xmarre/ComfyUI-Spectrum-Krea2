import dataclasses
import pytest
import torch
from comfyui_spectrum_krea2.config import Config
from comfyui_spectrum_krea2.forecast import History, ForecastUnavailable, chebyshev


def test_chebyshev_recurrence():
    x = torch.tensor([-1., -.3, .4, 1.], dtype=torch.float64)
    basis = chebyshev(x, 4)
    torch.testing.assert_close(basis[:,2], 2*x*x-1)
    torch.testing.assert_close(basis[:,3], 4*x**3-3*x)
    torch.testing.assert_close(basis[:,4], 8*x**4-8*x*x+1)


@pytest.mark.parametrize('degree', [1,2,3,4])
def test_exact_irregular_polynomial(degree):
    h = History(Config(degree=degree, ridge=0, max_history=8))
    coords = [-1., -.83, -.36, -.07, .21, .55]
    for x in coords:
        value = sum((i+1)*x**i for i in range(degree+1))
        h.observe(x, torch.full((2,4,8), value, dtype=torch.float64))
    x = .58
    predicted = h.predict(x, 'cpu', torch.float64)
    torch.testing.assert_close(predicted, torch.full_like(predicted, sum((i+1)*x**i for i in range(degree+1))), atol=2e-5, rtol=2e-5)


def test_ridge_matches_direct_feature_fit():
    h=History(Config(degree=2, ridge=.1))
    xs=[-1.,-.8,-.5,-.2]
    features=torch.randn(4,2,3,4)
    for x,f in zip(xs,features):
        h.observe(x,f)
    design=chebyshev(xs,2)
    coeff=torch.linalg.solve(design.T@design+.1*torch.eye(3,dtype=torch.float64),design.T@features.double().flatten(1))
    expected=(chebyshev([0.],2)@coeff).reshape(2,3,4).float()
    torch.testing.assert_close(h.predict(0.,'cpu',torch.float32),expected)


@pytest.mark.parametrize('dtype', [torch.float32,torch.bfloat16,torch.float16])
@pytest.mark.parametrize('storage', ['vram','system_ram'])
def test_chunking_dtype_and_sliding_memory(dtype, storage):
    h=History(Config(max_history=3, storage=storage))
    h.chunk_elements=7
    for x in [-1.,-.8,-.6,-.4,-.2]:
        h.observe(x,torch.ones((2,3,5),dtype=dtype))
    assert h.coordinates==[-.6,-.4,-.2]
    assert h.nbytes==3*30*torch.empty((),dtype=dtype).element_size()
    out=h.predict(0.,'cpu',dtype)
    assert out.shape==(2,3,5) and out.dtype==dtype
    assert torch.isfinite(out).all()
    assert len(h.features)==3  # Predictions never become observations.
    h.clear()
    assert h.nbytes==0 and h._factor is None


@pytest.mark.parametrize('coords,target', [([-1.],0.),([-1.,-.9],1.),([-1.,-.5],-.5),([-1.,-.5],float('nan'))])
def test_preconditions(coords,target):
    h=History(Config())
    for x in coords:
        h.observe(x,torch.ones(1,2,3))
    with pytest.raises(ForecastUnavailable):
        h.predict(target,'cpu',torch.float32)


def test_nonfinite_and_cast_overflow_rejected():
    for value in [float('nan'),float('inf'),1e10]:
        h=History(Config(ridge=0))
        for x in [-1.,-.5]:
            h.observe(x,torch.full((1,2,3),value))
        with pytest.raises(ForecastUnavailable):
            h.predict(-.25,'cpu',torch.float16)


def test_history_alias_is_explicit_ownership_contract():
    h=History(Config())
    h.observe(-1.,torch.ones(1,2,3))
    with pytest.raises(ForecastUnavailable):
        h.observe(-1.,torch.ones(1,2,3))
    with pytest.raises(ForecastUnavailable):
        h.observe(-.5,torch.ones(2,2,3))


@pytest.mark.parametrize('change', [dict(degree=0),dict(degree=5),dict(ridge=float('nan')),dict(max_history=1),dict(window=5),dict(growth=float('inf')),dict(tail=0),dict(storage='auto'),dict(budget_mb=0),dict(bootstrap=1)])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        Config(**change)


def test_turbo_profile_limits():
    config=Config(preset='turbo',warmup=0,tail=1,window=4,growth=1).resolved()
    assert (config.warmup,config.tail,config.window,config.growth)==(0,1,1,0)


def test_source_contract_ignores_optional_empty_ast_fields():
    import ast
    from comfyui_spectrum_krea2.compat import canonical_ast, source_digest
    node=ast.parse('def f(x):\n    return x[..., 0]\n')
    before=canonical_ast(node)
    node.body[0].type_params=[]
    assert canonical_ast(node)==before
    # Meaningful source changes must still change the contract.
    assert source_digest('def f(x): return x[..., 0]')!=source_digest('def f(x): return x[..., 1]')


def test_one_point_hold_is_an_owned_copy_of_the_only_anchor():
    h=History(Config())
    with pytest.raises(ForecastUnavailable):
        h.predict_hold('cpu',torch.float32)
    feature=torch.randn(1,4,3)
    h.observe(-1.,feature)
    held=h.predict_hold('cpu',torch.float32)
    torch.testing.assert_close(held,feature,atol=0,rtol=0)
    assert held.data_ptr()!=h.features[0].data_ptr()
    h.observe(-.5,feature)
    with pytest.raises(ForecastUnavailable):
        h.predict_hold('cpu',torch.float32)
    bad=History(Config()); bad.observe(-1.,torch.full((1,2),float('nan')))
    with pytest.raises(ForecastUnavailable):
        bad.predict_hold('cpu',torch.float32)
