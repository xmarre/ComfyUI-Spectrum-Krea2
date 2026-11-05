import copy
import pytest
from native_components import tiny_model
from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
from comfyui_spectrum_krea2.sampling import KEY


class Patcher:
    def __init__(self, inner):
        self.model=type('Base',(),{'diffusion_model':inner})()
        self.wrappers={}
    def clone(self):
        other=copy.copy(self)
        other.wrappers={t:{k:list(w) for k,w in groups.items()} for t,groups in self.wrappers.items()}
        return other
    def get_wrappers(self,kind,key):
        return self.wrappers.get(kind,{}).get(key,[])
    def add_wrapper_with_key(self,kind,key,wrapper):
        self.wrappers.setdefault(kind,{})[key]=[wrapper]
    def remove_wrappers_with_key(self,kind,key):
        self.wrappers.get(kind,{}).pop(key,None)


def test_disabled_is_identity_and_does_not_require_supported_model(native):
    patcher=Patcher(object())
    assert SpectrumApplyKrea2().apply(patcher,enabled=False)[0] is patcher


def test_clone_patch_order_idempotence_and_disable(native):
    original=Patcher(tiny_model(native[0]))
    external=lambda *args:None
    original.add_wrapper_with_key('diffusion_model','external',external)
    node=SpectrumApplyKrea2()
    patched=node.apply(original)[0]
    assert patched is not original and patched.model is original.model
    assert not original.get_wrappers('diffusion_model',KEY)
    assert patched.get_wrappers('diffusion_model','external')==[external]
    again=node.apply(patched,preset='turbo')[0]
    assert len(again.get_wrappers('diffusion_model',KEY))==1
    disabled=node.apply(again,enabled=False)[0]
    assert not disabled.get_wrappers('diffusion_model',KEY)
    assert disabled.get_wrappers('diffusion_model','external')==[external]
    assert patched.get_wrappers('diffusion_model',KEY)


def test_unsupported_model_is_diagnosed(native):
    with pytest.raises(ValueError,match='native Krea 2'):
        SpectrumApplyKrea2().apply(Patcher(object()))


def test_entrypoint_direct_import_without_package():
    import pathlib
    import runpy
    namespace=runpy.run_path(str(pathlib.Path(__file__).resolve().parents[1]/'__init__.py'))
    assert 'SpectrumApplyKrea2' in namespace['NODE_CLASS_MAPPINGS']


def test_entrypoint_comfy_package_import():
    import pathlib
    import importlib.util
    import sys
    root=pathlib.Path(__file__).resolve().parents[1]
    name='test_spectrum_krea2_comfy_entry'
    spec=importlib.util.spec_from_file_location(name,root/'__init__.py',submodule_search_locations=[str(root)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        spec.loader.exec_module(module)
        assert 'SpectrumApplyKrea2' in module.NODE_CLASS_MAPPINGS
    finally:
        for key in list(sys.modules):
            if key==name or key.startswith(name+'.'):
                sys.modules.pop(key)
