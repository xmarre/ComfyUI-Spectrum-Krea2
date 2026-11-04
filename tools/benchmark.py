#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired real-ComfyUI generation, timing, NFE and peak-VRAM capture.

Run with the Python environment used by ComfyUI. Models are loaded once. Each
configuration receives an untimed warmup. Timed pair order alternates by seed.
"""
import argparse
import dataclasses
import json
import logging
import pathlib
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-root', type=pathlib.Path, required=True)
    parser.add_argument('--model', required=True, help='Diffusion model filename relative to ComfyUI diffusion_models')
    parser.add_argument('--encoder', required=True, help='Native Krea 2 Qwen3-VL-4B 12-tap encoder filename')
    parser.add_argument('--vae', required=True)
    parser.add_argument('--preset', choices=['raw','turbo'], default='raw')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--cfg', type=float)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seeds', default='1,42,12345')
    parser.add_argument('--sampler', choices=['euler','dpmpp_2m'], default='euler')
    parser.add_argument('--scheduler', default='simple')
    parser.add_argument('--prompt', default='A candid photograph of a watchmaker holding a tiny brass gear with both hands, detailed fingertips and engraved metal, daylight through a workshop window, a handwritten card reading SPECTRUM on the desk.')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('validation_results'))
    args=parser.parse_args()
    sys.path.insert(0,str(args.comfy_root.resolve()))
    sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
    # Comfy's imports may inspect argv. Pass only actual Comfy arguments.
    sys.argv=[sys.argv[0],'--use-pytorch-cross-attention','--disable-all-custom-nodes']
    import comfy.options
    comfy.options.enable_args_parsing()
    import torch
    import nodes
    import comfy.model_management as mm
    from comfy.patcher_extension import WrappersMP
    from comfyui_spectrum_krea2.nodes import SpectrumApplyKrea2
    from comfyui_spectrum_krea2.runtime import REPORTS
    if not torch.cuda.is_available():
        raise SystemExit('This benchmark requires a CUDA GPU and real Krea 2 weights')
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True,exist_ok=True)
    steps=args.steps or (52 if args.preset=='raw' else 8)
    cfg=args.cfg if args.cfg is not None else (4.5 if args.preset=='raw' else 1.)
    seeds=[int(s) for s in args.seeds.split(',')]
    device=mm.get_torch_device()
    with torch.inference_mode():
        original=nodes.UNETLoader().load_unet(args.model,'default')[0]
        encoder=nodes.CLIPLoader().load_clip(args.encoder,'krea2')[0]
        vae=nodes.VAELoader().load_vae(args.vae)[0]
        positive=nodes.CLIPTextEncode().encode(encoder,args.prompt)[0]
        negative=nodes.CLIPTextEncode().encode(encoder,'')[0]
        latent={'samples':torch.zeros(args.batch,16,args.height//8,args.width//8)}
        accelerated=SpectrumApplyKrea2().apply(original,preset=args.preset)[0]
        # Count completed native diffusion calls in the baseline using a model
        # wrapper. No module hook or replacement of native model math is used.
        baseline=original.clone()
        native_counts=[]
        def count_native(executor,*a,**kw):
            out=executor(*a,**kw)
            native_counts.append(1)
            return out
        baseline.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL,'krea2_benchmark_observer',count_native)
        models={'native':baseline,'spectrum':accelerated}
        def generate(kind,seed):
            return nodes.common_ksampler(models[kind],seed,steps,cfg,args.sampler,args.scheduler,
                                         positive,negative,latent,denoise=1.)[0]
        # Equal warmup policy, excluded from all reported timing and NFE data.
        generate('native',seeds[0])
        generate('spectrum',seeds[0])
        records=[]
        for index,seed in enumerate(seeds):
            order=('native','spectrum') if index%2==0 else ('spectrum','native')
            for kind in order:
                native_counts.clear()
                reports=[]
                token=REPORTS.set(reports)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                started=time.perf_counter()
                try:
                    sample=generate(kind,seed)
                    torch.cuda.synchronize(device)
                    seconds=time.perf_counter()-started
                    peak=torch.cuda.max_memory_allocated(device)
                finally:
                    REPORTS.reset(token)
                if kind=='spectrum':
                    if not reports:
                        raise RuntimeError('Spectrum did not produce a sampler report')
                    actual=sum(r['actual'] for r in reports)
                    forecast=sum(r['forecast'] for r in reports)
                else:
                    actual=len(native_counts); forecast=0
                # Decoding and file writes are excluded from sampler timing.
                images=vae.decode(sample['samples'])
                from PIL import Image
                for batch,image in enumerate(images):
                    Image.fromarray((image.clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(args.output/f'{args.preset}_{seed}_{batch}_{kind}.png')
                record={'preset':args.preset,'seed':seed,'kind':kind,'sampler_seconds':seconds,
                        'actual':actual,'forecast':forecast,'peak_allocated_bytes':peak,
                        'sampler':args.sampler,'scheduler':args.scheduler,'steps':steps,'cfg':cfg,
                        'shape':[args.batch,args.height,args.width], 'reports':reports}
                records.append(record)
                (args.output/'benchmark.json').write_text(json.dumps({'configuration':vars(args)|{'comfy_root':str(args.comfy_root),'output':str(args.output)},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device),'records':records},indent=2))
                print(json.dumps(record))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
