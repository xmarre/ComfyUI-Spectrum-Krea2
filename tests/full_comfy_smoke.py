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
    p.add_argument('--kj-root',type=pathlib.Path)
    p.add_argument('--kj-legacy-root',type=pathlib.Path)
    p.add_argument('--kj-older-root',type=pathlib.Path)
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
    for sampler in ('euler','dpmpp_2m','res_multistep','gradient_estimation',
                    'deis','ipndm','ipndm_v'):
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
    # Exact nonforecast native equivalence and forecast count on the user's
    # disabled-extrapolation-guard 20-step Turbo configuration. No artificial
    # sampling loop is substituted for the ComfyUI CFGGuider or native solver.
    for sampler in ('res_multistep', 'gradient_estimation',
                    'deis', 'ipndm', 'ipndm_v'):
        positive = [[torch.randn(1,3,32),{}]]
        negative = [[torch.randn(1,3,32),{}]]
        noise = torch.randn(1,16,1,4,6)
        native, _ = sample(model,noise,positive,negative,1.,sampler,steps=20)
        exact, reports = sample(node.apply(model,window=0)[0],noise,positive,negative,1.,sampler,steps=20)
        torch.testing.assert_close(exact,native,atol=0,rtol=0)
        assert reports[0]['fallback'] is None and reports[0]['actual']==20
        chosen = node.apply(model,preset='turbo',warmup=2,tail=1,
                            bootstrap=False,extrapolation_guard_enabled=False)[0]
        output, reports = sample(chosen,noise,positive,negative,1.,sampler,steps=20)
        report = reports[0]
        assert report['fallback'] is None and report['forecast']==9 and report['actual']==11, report
        repeated,_=sample(chosen,noise,positive,negative,1.,sampler,steps=20)
        torch.testing.assert_close(output,repeated,atol=0,rtol=0)
        records.append({'new_solver_turbo':sampler,**report})


    # PECE's corrected-state evaluations happen after the step callback.
    # Validate the real ComfyUI CFGGuider and native SA-Solver rather than
    # relying only on the isolated WrapperExecutor fixture.
    for solver in ("sa_solver", "sa_solver_pece"):
        noise_sa = torch.randn(1,16,1,4,6)
        context_sa = torch.randn(1,3,32)
        positive_sa = [[context_sa, {}]]
        negative_sa = [[context_sa*.2, {}]]
        native_sa, _ = sample(model, noise_sa, positive_sa, negative_sa, 1.,
                              solver, steps=14)
        exact_sa, exact_reports = sample(
            node.apply(model, window=0)[0], noise_sa, positive_sa,
            negative_sa, 1., solver, steps=14,
        )
        torch.testing.assert_close(exact_sa, native_sa, atol=0, rtol=0)
        assert exact_reports[0]["fallback"] is None
        assert exact_reports[0]["forecast"] == 0
        selected_sa = node.apply(
            model, preset="turbo", warmup=2, tail=1, bootstrap=False,
            extrapolation_guard_enabled=False,
        )[0]
        result, reports_sa = sample(
            selected_sa, noise_sa, positive_sa, negative_sa, 1.,
            solver, steps=14,
        )
        report = reports_sa[0]
        assert report["fallback"] is None and report["forecast"] > 0, report
        assert report["outer_steps"] == 14, report
        corrections = 13 if solver == "sa_solver_pece" else 0
        assert report["pece_corrector_evaluations"] == corrections, report
        assert report["actual"] + report["forecast"] == 14 + corrections, report
        repeat, repeated = sample(
            selected_sa, noise_sa, positive_sa, negative_sa, 1.,
            solver, steps=14,
        )
        torch.testing.assert_close(result, repeat, atol=0, rtol=0)
        assert repeated[0]["actual"] == report["actual"]
        records.append({"native_sa_solver": solver, **report})

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
    # A multi-evaluation solver remains exactly native, including its extra calls.
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

    def make_kj_preview(source_root=None):
        """Load a genuine KJNodes callback source with preview I/O suppressed."""
        source_root = source_root or args.kj_root
        # Keep distinct real source versions in distinct registered modules,
        # keyed by resolved path so roots sharing a basename stay distinct.
        import hashlib
        resolved = str(source_root.resolve())
        package_name = ("kj_preview_" + source_root.name.lstrip("_").replace("-", "_") + "_"
                        + hashlib.sha1(resolved.encode()).hexdigest()[:12])
        module_name = package_name + ".preview_override_node"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing._PreviewOverrideWrapper(
                512, None, 80, False, preview_frames=1, tiny_vae="none",
            )
        import ast
        import logging
        import time
        import types
        path = source_root.resolve() / "nodes/preview_override_node.py"
        tree = ast.parse(path.read_bytes(), filename=str(path))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "_PreviewOverrideWrapper")

        class NoPreviewEncoder:
            def submit(self, fn):
                return True
            def shutdown(self, drain_timeout=5.0):
                pass

        sys.modules.setdefault(package_name, types.ModuleType(package_name))
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__dict__.update({
            "torch": torch, "comfy": __import__("comfy"), "time": time,
            "logging": logging, "PromptServer": None,
            "load_tiny_vae_decoder": None,
            "_get_core_previewer": lambda *a: None,
            "_is_ltx_latent_format": lambda *a: False,
            "_is_av_model": lambda *a: False,
            "_normalize_packed_x0": lambda x, *a: x,
            "_frames_per_token": lambda *a: (1, 1),
            "_AsyncPreviewEncoder": NoPreviewEncoder,
            "latent_preview": types.SimpleNamespace(
                Latent2RGBPreviewer=lambda *a, **kw: None,
            ),
        })
        sys.modules[module_name] = module
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"),
             module.__dict__)
        klass = module._PreviewOverrideWrapper
        return klass(512, None, 80, False, preview_frames=1, tiny_vae="none")

    def with_kj(model_patcher, source_root=None):
        from comfy.patcher_extension import WrappersMP
        copied = model_patcher.clone()
        copied.add_wrapper_with_key(
            WrappersMP.OUTER_SAMPLE, "kj_preview_override", make_kj_preview(source_root),
        )
        return copied

    if args.kj_root:
        # Real native ComfyUI, original source-audited KJ OUTER_SAMPLE __call__,
        # and Spectrum SAMPLER_SAMPLE/PREDICT_NOISE wrappers in either model
        # patching order. No preview backends or network emissions are invoked.
        chosen_options = dict(preset="turbo", warmup=2, tail=1,
                              bootstrap=False, extrapolation_guard_enabled=False)
        noise_kj = torch.randn(1, 16, 1, 5, 7)
        context_kj = torch.randn(1, 3, 32)
        pos_kj = [[context_kj, {}]]
        neg_kj = [[context_kj*.2, {}]]
        native, _ = sample(model, noise_kj, pos_kj, neg_kj, 1., "er_sde", steps=14)
        native_preview, _ = sample(
            with_kj(model), noise_kj, pos_kj, neg_kj, 1., "er_sde", steps=14,
        )
        torch.testing.assert_close(native_preview, native, atol=0, rtol=0)
        baseline = node.apply(model, **chosen_options)[0]
        baseline_output, baseline_reports = sample(
            baseline, noise_kj, pos_kj, neg_kj, 1., "er_sde", steps=14,
        )
        assert baseline_reports[0]["fallback"] is None, baseline_reports
        assert baseline_reports[0]["forecast"] > 0
        for order, selected in (
            ("preview_before_spectrum", node.apply(with_kj(model), **chosen_options)[0]),
            ("preview_after_spectrum", with_kj(baseline)),
        ):
            output, reports = sample(
                selected, noise_kj, pos_kj, neg_kj, 1., "er_sde", steps=14,
            )
            report = reports[0]
            assert report["fallback"] is None and report["forecast"] > 0, report
            assert report["outer_steps"] == 14
            assert report["actual"] + report["forecast"] == 14
            torch.testing.assert_close(output, baseline_output, atol=0, rtol=0)
            records.append({"kj_preview": order, **report})

        # Validate released KJNodes revisions with nonidentical full-file
        # bytes, including their original callback implementations.
        for root in (args.kj_legacy_root, args.kj_older_root):
            if root is None:
                continue
            selected_old = node.apply(with_kj(model, root), **chosen_options)[0]
            old_output, old_reports = sample(
                selected_old, noise_kj, pos_kj, neg_kj, 1., "er_sde", steps=14,
            )
            assert len(old_reports) == 1
            report = old_reports[0]
            assert report["fallback"] is None and report["forecast"] > 0, report
            assert report["actual"] + report["forecast"] == 14
            torch.testing.assert_close(old_output, baseline_output, atol=0, rtol=0)
            records.append({"kj_historical_source": root.name, **report})

        # Callback implementation is not version-pinned. A changed callback
        # method can be admitted in a NEW invocation; a mutation during an
        # existing invocation must invalidate the prior identity snapshot.
        from comfy.patcher_extension import WrappersMP
        from comfyui_spectrum_krea2.qualified_patches import qualify, matches
        preview = selected.get_wrappers(WrappersMP.OUTER_SAMPLE, "kj_preview_override")[-1]
        # ModelPatcher stores wrappers separately; the runtime materializes
        # transformer_options.wrappers for its sampler. Build that exact shape
        # from the registered instance rather than inspecting dormant options.
        admitted_options = {
            "transformer_options": {
                "wrappers": {"outer_sample": {"kj_preview_override": [preview]}},
            },
        }
        policy, reason = qualify(admitted_options)
        assert reason is None and policy.kj_preview_wrapper is preview
        preview_cls = type(preview)
        original_preview_call = preview_cls.__call__
        try:
            preview_cls.__call__ = lambda self, executor, *a, **kw: executor(*a, **kw)
            assert not matches(policy, admitted_options["transformer_options"])
            requalified, reason = qualify(admitted_options)
            assert reason is None and requalified.kj_preview_wrapper is preview
            assert matches(requalified, admitted_options["transformer_options"])
        finally:
            preview_cls.__call__ = original_preview_call

        # A code-object swap preserves callable identity, but must invalidate
        # the active invocation. A new invocation revalidates the actual
        # solver and model state with whatever callback KJ currently supplies.
        original_code = original_preview_call.__code__
        def substitute_call(self, executor, noise, latent_image, sampler,
                            sigmas, denoise_mask, callback, disable_pbar,
                            seed, latent_shapes):
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            callback, disable_pbar, seed, latent_shapes=latent_shapes)
        try:
            original_preview_call.__code__ = substitute_call.__code__
            assert not matches(policy, admitted_options["transformer_options"])
            requalified, reason = qualify(admitted_options)
            assert reason is None and requalified.kj_preview_wrapper is preview
            assert matches(requalified, admitted_options["transformer_options"])
        finally:
            original_preview_call.__code__ = original_code

        recovered, reason = qualify(admitted_options)
        assert reason is None and recovered.kj_preview_wrapper is preview

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

        # Partial and ramped Diff-Aid sigma windows through the real node and
        # its SharedTimestepWrapper normalized sigma. Every actual evaluation
        # must apply exactly the gain Spectrum reproduced for it; history is
        # restarted at each gain change while forecasting stays enabled.
        import math
        from comfyui_spectrum_krea2.qualified_patches import DiffAidWindow
        original_regime, original_alpha = DiffAidWindow.regime, diff._compute_alpha
        for mode, window, solver, pece_policy in (
            ('text_residual', (0., .9, 0.), 'er_sde', 'off'),
            ('text_residual', (.1, .9, 0.), 'er_sde', 'off'),
            ('values_only', (.2, .7, .1), 'euler', 'off'),
            ('keys_and_values', (.1, .9, 0.), 'sa_solver_pece', 'balanced'),
        ):
            events = []
            def spy_regime(self, options, x):
                value = original_regime(self, options, x)
                events.append(('regime', value))
                return value
            def spy_alpha(reference, token_count, config, options):
                out = original_alpha(reference, token_count, config, options)
                events.append(('alpha', float(out.reshape(-1)[0]), config.strength))
                return out
            windowed, _ = diff.Krea2DiffAidSparsePatchNode().patch(
                model12, enabled=True, block_preset='custom_block_indices',
                block_indices='1', strength=.15, modulation_target=mode,
                sigma_start=window[0], sigma_end=window[1], sigma_ramp=window[2],
                token_weight_mode='none', token_tail=.35, cond_only=True,
            )
            selected = node.apply(
                windowed, preset='turbo', warmup=2, tail=1, bootstrap=False,
                extrapolation_guard_enabled=False, sa_pece_forecast_policy=pece_policy,
            )[0]
            DiffAidWindow.regime, diff._compute_alpha = spy_regime, spy_alpha
            try:
                _, reports = sample(selected, noise12, positive12, negative12, 1., solver, steps=14)
            finally:
                DiffAidWindow.regime, diff._compute_alpha = original_regime, original_alpha
            report = reports[0]
            assert report['fallback'] is None, report
            assert report['forecast'] > 0, report
            assert report['actual_reasons'].get('Diff-Aid sigma-window transition', 0) >= 1, report
            latest, gains = None, set()
            for event in events:
                if event[0] == 'regime':
                    latest = event[1]
                    continue
                assert latest is not None and len(latest) == 1, events
                expected = float(torch.tensor(event[2] * latest[0], dtype=torch.float32))
                assert math.isclose(event[1], expected, rel_tol=0, abs_tol=1e-7), (event, latest)
                gains.add(latest[0])
            assert 0.0 in gains and 1.0 in gains, (window, gains)
            assert not base12.diffusion_model.last._forward_pre_hooks
            records.append({'diffaid_sigma_window': window, 'mode': mode, 'solver': solver, **report})

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

        if args.kj_root:
            # KJ + Sol + actual Krea Diff-Aid text residual + Spectrum + ER-SDE.
            # CPU executes dense fallback for the actual Sol kernel, but the
            # real Sol sampling/diffusion wrappers and callback order execute.
            with_sol = sol.install(
                modded, sol.Settings(dense_evaluations=0, dense_layers=0, min_tokens=0),
            )
            selected = node.apply(
                with_kj(with_sol), preset="turbo", warmup=2, tail=1,
                bootstrap=False, extrapolation_guard_enabled=False,
            )[0]
            output, reports = sample(
                selected, noise12, positive12, negative12, 1., "er_sde", steps=14,
            )
            report = reports[0]
            assert report["fallback"] is None, report
            assert report["forecast"] > 0, report
            assert report["actual"] + report["forecast"] == 14, report
            assert CURRENT.get() is None and CAPTURE.get() is None
            records.append({"kj_sol_diffaid_er_sde": True, **report})

    print(json.dumps({'full_native_cases':len(records),'records':records},sort_keys=True))


if __name__=='__main__':
    main()
