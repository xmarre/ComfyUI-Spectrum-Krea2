# SPDX-License-Identifier: GPL-3.0-or-later
"""Full native CPU sampling integration with small random Krea 2 weights.

No ComfyUI infrastructure is mocked. This is an execution/parity test, not a
real-checkpoint image-quality or performance test. Run in a separate process
from the native-component fixture, which supplies reduced host infrastructure.
"""
import argparse
import json
import pathlib
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-root',type=pathlib.Path,required=True)
    p.add_argument('--sampler-source-root',type=pathlib.Path)
    args=p.parse_args()
    sys.path[:0]=[str(args.comfy_root.resolve()),str(pathlib.Path(__file__).resolve().parents[1])]
    sys.argv=[sys.argv[0],'--cpu','--use-pytorch-cross-attention','--disable-all-custom-nodes']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    torch.set_num_threads(1)
    import comfy.sample
    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models
    if args.sampler_source_root:
        import ast
        import comfy.samplers
        path=args.sampler_source_root.resolve()/'comfy/samplers.py'
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='KSAMPLER')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample')
        namespace=dict(vars(comfy.samplers))
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
        comfy.samplers.KSAMPLER.sample=namespace['sample']
    from comfy.weight_adapter.lora import LoRAAdapter
    from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
    from comfyui_spectrum_krea2.runtime import CURRENT, REPORTS
    from comfyui_spectrum_krea2.adapter import CAPTURE
    config=comfy.supported_models.Krea2(dict(image_model='krea2',features=32,tdim=8,txtdim=16,
        heads=2,kvheads=1,multiplier=1,layers=2,patch=2,channels=16,txtlayers=2,txtheads=2,txtkvheads=1))
    config.set_inference_dtype(torch.float32,None)
    torch.manual_seed(7)
    base=comfy.model_base.Krea2(config,device=torch.device('cpu'))
    with torch.no_grad():
        for param in base.diffusion_model.parameters():
            param.copy_(torch.randn_like(param)*.02)
    model=comfy.model_patcher.ModelPatcher(base,load_device=torch.device('cpu'),offload_device=torch.device('cpu'))
    node=SpectrumApplyKrea2()
    records=[]
    def sample(patcher,noise,positive,negative,cfg,sampler,steps=12,**kwargs):
        reports=[]; token=REPORTS.set(reports)
        try:
            with torch.inference_mode():
                result=comfy.sample.sample(patcher,noise,steps,cfg,sampler,'simple',positive,negative,
                                          torch.zeros_like(noise),disable_pbar=True,seed=42,**kwargs)
        finally:
            REPORTS.reset(token)
        assert CURRENT.get() is None and CAPTURE.get() is None
        assert not base.diffusion_model.last._forward_pre_hooks
        assert result.shape==noise.shape and torch.isfinite(result).all()
        return result,reports
    # Actual CFGGuider, calc_cond_batch, BaseModel transformations, model loading,
    # ModelPatcher.clone/pre_run and native sampler execution all participate.
    for sampler in ('euler','dpmpp_2m'):
        for batch,cfg,reference in ((1,1.,False),(1,4.5,False),(2,4.5,False),(1,4.5,True)):
            noise=torch.randn(batch,16,1,5,7)
            context=torch.randn(1,3,32)
            extra={}
            if reference:
                extra={'reference_latents':[torch.randn(1,16,1,4,6)],'reference_latents_method':'index_timestep_zero'}
            positive=[[context,dict(extra)]];negative=[[context*.2,dict(extra)]]
            native,_=sample(model,noise,positive,negative,cfg,sampler)
            all_actual,reports=sample(node.apply(model,window=0)[0],noise,positive,negative,cfg,sampler)
            torch.testing.assert_close(all_actual,native,atol=0,rtol=0)
            assert reports[0]['forecast']==0
            accelerated,reports=sample(node.apply(model)[0],noise,positive,negative,cfg,sampler)
            assert reports[0]['forecast']>0 and reports[0]['fallback'] is None
            repeat,_=sample(node.apply(model)[0],noise,positive,negative,cfg,sampler)
            torch.testing.assert_close(accelerated,repeat,atol=0,rtol=0)
            records.append({'sampler':sampler,'batch':batch,'cfg':cfg,'reference':reference,**reports[0]})
    # Exercise real weight patching and shared model restoration using a standard
    # native LoRA adapter. No custom forward or synthetic patcher is used.
    patched=model.clone()
    key='diffusion_model.blocks.0.attn.wq.weight'
    weight=base.state_dict()[key]
    adapter=LoRAAdapter(set(),(torch.randn(weight.shape[0],2)*.1,torch.randn(2,weight.shape[1])*.1,2.,None,None,None))
    assert patched.add_patches({key:adapter},strength_patch=.75)==[key]
    positive=[[torch.randn(1,3,32),{}]];negative=[[torch.randn(1,3,32),{}]]
    noise=torch.randn(1,16,1,4,6)
    original,_=sample(model,noise,positive,negative,4.5,'euler')
    lora_native,_=sample(patched,noise,positive,negative,4.5,'euler')
    assert not torch.equal(original,lora_native)
    actual,_=sample(node.apply(patched,window=0)[0],noise,positive,negative,4.5,'euler')
    torch.testing.assert_close(actual,lora_native,atol=0,rtol=0)
    _,reports=sample(node.apply(patched)[0],noise,positive,negative,4.5,'euler')
    assert reports[0]['forecast']>0 and reports[0]['fallback'] is None
    restored,_=sample(model,noise,positive,negative,4.5,'euler')
    torch.testing.assert_close(restored,original,atol=0,rtol=0)
    records.append({'lora':True,**reports[0]})
    # Truncated image-to-image schedule starts with its own warmup and histories.
    _,reports=sample(node.apply(model)[0],noise,positive,negative,4.5,'euler',denoise=.55)
    assert reports[0]['forecast']>0 and reports[0]['outer_steps']==12
    records.append({'denoise':.55,**reports[0]})
    # A multistage solver remains exactly native, including its extra calls.
    native,_=sample(model,noise,positive,negative,4.5,'heun')
    bypass,reports=sample(node.apply(model)[0],noise,positive,negative,4.5,'heun')
    torch.testing.assert_close(bypass,native,atol=0,rtol=0)
    assert reports[0]['forecast']==0 and reports[0]['fallback'] is not None
    assert reports[0]['actual']>12
    records.append({'unsupported_sampler':'heun',**reports[0]})
    foreign=model.clone()
    def external_wrapper(executor,*a,**kw):
        return executor(*a,**kw)+.125
    from comfy.patcher_extension import WrappersMP
    foreign.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL,'test_external',external_wrapper)
    native,_=sample(foreign,noise,positive,negative,4.5,'euler')
    bypass,reports=sample(node.apply(foreign)[0],noise,positive,negative,4.5,'euler')
    torch.testing.assert_close(bypass,native,atol=0,rtol=0)
    assert reports[0]['forecast']==0 and 'external' in reports[0]['fallback']
    records.append({'external_wrapper':True,**reports[0]})
    for steps in (8,12):
        selected=node.apply(model,preset='turbo',warmup=1,tail=1)[0]
        _,reports=sample(selected,noise,positive,negative,1.,'euler',steps=steps)
        assert reports[0]['fallback'] is None
        assert reports[0]['effective_config']['warmup']==1
        assert reports[0]['effective_config']['tail']==1
        assert reports[0]['forecast']>0
        records.append({'turbo_steps':steps,**reports[0]})
    print(json.dumps({'full_native_cases':len(records),'records':records},sort_keys=True))


if __name__=='__main__':
    main()
