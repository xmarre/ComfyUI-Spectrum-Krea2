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
    p.add_argument('--sol-root',type=pathlib.Path)
    p.add_argument('--diffaid-root',type=pathlib.Path)
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
    # Transparent native-container wrappers must not disable the real solver.
    from sampler_variants import wrap_sample
    import comfy.samplers
    original_sample=comfy.samplers.KSAMPLER.sample
    for variant in ('optional','variadic'):
        comfy.samplers.KSAMPLER.sample=wrap_sample(original_sample,variant)
        try:
            selected=node.apply(model,preset='turbo',warmup=1,tail=1)[0]
            _,reports=sample(selected,noise,positive,negative,1.,'euler')
            assert reports[0]['fallback'] is None and reports[0]['forecast']>0
            records.append({'container_wrapper':variant,**reports[0]})
        finally:
            comfy.samplers.KSAMPLER.sample=original_sample
    if args.sol_root and args.diffaid_root:
        # Real ComfyUI ModelPatcher, real Diff-Aid types, and the actual Sol
        # wrapper stack. Only the CUDA sparse arithmetic is replaced by its
        # mathematically exact CPU oracle to exercise numerical-route changes.
        import importlib.util
        sys.path.insert(0, str(args.sol_root.resolve()))
        path = args.diffaid_root.resolve() / 'nodes.py'
        spec = importlib.util.spec_from_file_location('diffaid_krea2_nodes', path)
        diff = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = diff
        spec.loader.exec_module(diff)
        from sol_h3 import krea2 as sol
        from comfy.patcher_extension import WrappersMP

        shared = diff.SharedConfig(.15, 0., 1., 0., 'none', .35, True)
        noise = torch.randn(1,16,1,5,7)
        context = torch.randn(1,3,32)
        positive = [[context,{}]]
        negative = [[context*.2,{}]]
        for mode in ('values_only', 'text_residual'):
            modded = model.clone()
            modded.set_model_unet_function_wrapper(diff.SharedTimestepWrapper())
            if mode == 'text_residual':
                modded.add_wrapper_with_key(
                    WrappersMP.DIFFUSION_MODEL,
                    diff.Krea2TextResidualWrapper.WRAPPER_KEY,
                    diff.Krea2TextResidualWrapper(shared, (0,)),
                )
            else:
                modded.set_model_attn1_patch(diff.Krea2AttentionPatch(shared, (0,), mode))
            with_sol = sol.install(
                modded, sol.Settings(dense_evaluations=1, dense_layers=0, min_tokens=0),
            )
            selected = node.apply(with_sol, preset='turbo', warmup=1, tail=1)[0]

            original_gate, original_attention = sol._can_sparse, sol.sol_attention
            def exact_cpu(q, k, v, prefix, cfg, state):
                state.sparse_calls += 1
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v,
                ).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

            def eligible(q, k, v, heads, mask, prec, skip, skip_out, kw,
                         settings, state, blocks):
                options = kw.get('transformer_options') or {}
                if options.get('block_type') != 'single' or 'img_slice' not in options:
                    return 'text_fusion', None
                if state.evaluations <= settings.dense_evaluations:
                    return 'dense_evaluation_warmup', None
                return None, options['img_slice'][0]

            from comfyui_spectrum_krea2.qualified_patches import QualifiedPatches
            original_phase = QualifiedPatches.phase
            def cpu_oracle_phase(self, x, context, refs, inner):
                status = self.sol_status()
                if status is None:
                    return 'native'
                return ('dense' if status['actual_evaluations'] + 1 <=
                        status['settings'].dense_evaluations else 'sol')
            sol._can_sparse = eligible
            sol.sol_attention = exact_cpu
            QualifiedPatches.phase = cpu_oracle_phase
            try:
                _, reports = sample(selected, noise, positive, negative, 1., 'euler', steps=10)
            finally:
                QualifiedPatches.phase = original_phase
                sol._can_sparse = original_gate
                sol.sol_attention = original_attention
            report = reports[0]
            assert report['fallback'] is None, report
            assert report['forecast'] > 0, report
            assert report['actual'] + report['forecast'] == 10
            # The backend changes from dense to Sol after the initial actual;
            # the first sparse anchor must never be forecasted from dense history.
            assert report['actual_reasons'].get('numerical attention backend transition', 0) >= 1, report
            records.append({'combined_overlay': mode, **report})

        # A real Krea Diff-Aid node (not a hand-built patch) requires the native
        # 12-layer text fusion model. This reproduces the deployed node install
        # and complete ModelPatcher cloning/wrapper registration sequence.
        config12 = comfy.supported_models.Krea2(dict(
            image_model='krea2', features=32, tdim=8, txtdim=16,
            heads=2, kvheads=1, multiplier=1, layers=2, patch=2,
            channels=16, txtlayers=12, txtheads=2, txtkvheads=1,
        ))
        config12.set_inference_dtype(torch.float32, None)
        base12 = comfy.model_base.Krea2(config12, device=torch.device('cpu'))
        with torch.no_grad():
            for p in base12.diffusion_model.parameters():
                p.copy_(torch.randn_like(p) * .02)
        model12 = comfy.model_patcher.ModelPatcher(
            base12, load_device=torch.device('cpu'), offload_device=torch.device('cpu'),
        )
        # KJ Sage auto is an inherited optimized-attention override. Use the
        # original ComfyUI kernel as a deterministic CPU stand-in for that owner.
        def inherited_attention(func, *args, **kwargs):
            return func(*args, **kwargs)
        model12.model_options['transformer_options']['optimized_attention_override'] = inherited_attention
        cond12 = torch.randn(1, 3, base12.diffusion_model.txtlayers * base12.diffusion_model.txtdim)
        positive12 = [[cond12, {}]]
        negative12 = [[cond12*.2, {}]]
        noise12 = torch.randn(1, 16, 1, 5, 7)
        from comfyui_spectrum_krea2.qualified_patches import qualify

        for mode in ('values_only', 'text_residual'):
            modded, summary = diff.Krea2DiffAidSparsePatchNode().patch(
                model12, enabled=True, block_preset='custom_block_indices',
                block_indices='1', strength=.15, modulation_target=mode,
                sigma_start=0., sigma_end=1., sigma_ramp=0.,
                token_weight_mode='none', token_tail=.35, cond_only=True,
            )
            assert 'krea2_sparse' in summary
            with_sol = sol.install(modded, sol.Settings(
                dense_evaluations=1, dense_layers=0, min_tokens=0,
            ))
            actual_node = node.apply(
                with_sol, preset='turbo', warmup=1, tail=1,
            )[0]
            from comfyui_spectrum_krea2.qualified_patches import QualifiedPatches
            original_phase = QualifiedPatches.phase
            original_gate, original_attention = sol._can_sparse, sol.sol_attention
            QualifiedPatches.phase = cpu_oracle_phase
            sol._can_sparse = eligible
            sol.sol_attention = exact_cpu
            try:
                _, reports = sample(
                    actual_node, noise12, positive12, negative12, 1., 'euler', steps=10,
                )
            finally:
                QualifiedPatches.phase = original_phase
                sol._can_sparse = original_gate
                sol.sol_attention = original_attention
            report = reports[0]
            assert report['fallback'] is None, report
            assert report['forecast'] > 0, report
            assert report['actual'] + report['forecast'] == 10, report
            assert report['actual_reasons'].get('numerical attention backend transition'), report
            assert not base12.diffusion_model.last._forward_pre_hooks
            records.append({'real_diffaid_node': mode, **report})

        # Startup alignment: Spectrum must still perform two REAL DiT
        # evaluations; setting Sol dense_evaluations=0 merely makes both
        # anchors use the same sparse attention operator. A dense->sparse
        # transition requires a new, homogeneous history. Never manufacture
        # the missing anchor, bypass the extrapolation limit, or bootstrap.
        aligned_reports = {}
        modded, _ = diff.Krea2DiffAidSparsePatchNode().patch(
            model12, enabled=True, block_preset='custom_block_indices',
            block_indices='1', strength=.15, modulation_target='text_residual',
            sigma_start=0., sigma_end=1., sigma_ramp=0.,
            token_weight_mode='none', token_tail=.35, cond_only=True,
        )
        for dense_evaluations in (1, 0):
            for steps in (8, 16):
                with_sol = sol.install(modded, sol.Settings(
                    dense_evaluations=dense_evaluations,
                    dense_layers=0, min_tokens=0,
                ))
                selected = node.apply(
                    with_sol, preset='turbo', warmup=2, tail=1, bootstrap=False,
                )[0]
                QualifiedPatches.phase = cpu_oracle_phase
                sol._can_sparse = eligible
                sol.sol_attention = exact_cpu
                try:
                    _, reports = sample(
                        selected, noise12, positive12, negative12,
                        1., 'euler', steps=steps,
                    )
                finally:
                    QualifiedPatches.phase = original_phase
                    sol._can_sparse = original_gate
                    sol.sol_attention = original_attention
                report = reports[0]
                assert report['fallback'] is None, report
                assert report['actual'] + report['forecast'] == steps, report
                assert report['bootstrap_forecast'] == 0, report
                assert report['actual'] >= 2, report
                transition_count = report['actual_reasons'].get(
                    'numerical attention backend transition', 0,
                )
                assert transition_count == (1 if dense_evaluations else 0), report
                assert not report['actual_reasons'].get(
                    'numerical attention route differed from the previous anchor', 0,
                ), report
                assert not base12.diffusion_model.last._forward_pre_hooks
                aligned_reports[dense_evaluations, steps] = report
                records.append({
                    'coherent_sol_start': dense_evaluations == 0,
                    'dense_evaluations': dense_evaluations,
                    'steps': steps,
                    **report,
                })
        for steps in (8, 16):
            aligned = aligned_reports[0, steps]
            transitioned = aligned_reports[1, steps]
            # Compare against the SAME weights, conditioning, solver and
            # sigma schedule with Spectrum alone. Absolute forecast counts
            # vary with sigma spacing; removing the attention-route change
            # must recover all opportunities available to that baseline.
            baseline_model = node.apply(
                modded, preset='turbo', warmup=2, tail=1, bootstrap=False,
            )[0]
            _, baseline_reports = sample(
                baseline_model, noise12, positive12, negative12,
                1., 'euler', steps=steps,
            )
            baseline = baseline_reports[0]
            assert baseline['fallback'] is None, baseline
            assert aligned['forecast'] == baseline['forecast'], (
                steps, aligned, baseline,
            )
            assert aligned['actual'] == baseline['actual'], (
                steps, aligned, baseline,
            )
            assert aligned['forecast'] >= transitioned['forecast'], (
                steps, aligned, transitioned,
            )
            records.append({
                'spectrum_only_reference': True, 'steps': steps, **baseline,
            })
    print(json.dumps({'full_native_cases':len(records),'records':records},sort_keys=True))


if __name__=='__main__':
    main()
