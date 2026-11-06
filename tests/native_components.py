# SPDX-License-Identifier: GPL-3.0-or-later
"""Load unchanged upstream components with CPU-only host infrastructure.

Krea 2 blocks, normalization/modulation, patchification, final projection, rotary
math, WrapperExecutor, KSAMPLER and sampler functions run their actual source.
Model management, attention dispatch and the top-level CFG owner are CPU test
substitutes. This verifies integration contracts, not real checkpoint quality.
"""
import ast
import importlib
import math
import pathlib
import os
import sys
import types
import torch
from torch import nn
from einops import rearrange


def module(name, path=None, **attrs):
    m = types.ModuleType(name)
    if path is not None:
        m.__path__ = [str(path)]
    m.__dict__.update(attrs)
    sys.modules[name] = m
    if '.' in name:
        parent, attr = name.rsplit('.', 1)
        setattr(sys.modules[parent], attr, m)
    return m


def extract(name, path, names, namespace):
    m = sys.modules.get(name) or module(name)
    m.__dict__.update(namespace)
    source = pathlib.Path(path).read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(tree, str(path), 'exec'), m.__dict__)
    return m


def load(root):
    root = pathlib.Path(root).resolve()
    comfy = module('comfy', root / 'comfy')
    module('comfy.ldm', root / 'comfy/ldm')
    module('comfy.ldm.flux', root / 'comfy/ldm/flux')
    module('comfy.ldm.modules', root / 'comfy/ldm/modules')
    module('comfy.model_management', cast_to=lambda x, **kw: x.to(**kw),
           supports_fp64=lambda _: True, in_training=True)
    module('comfy.quant_ops')
    module('comfy.rmsnorm', rms_norm=torch.nn.functional.rms_norm)
    module('comfy.utils')
    extract('comfy.utils', root/'comfy/utils.py', {'repeat_to_batch_size'}, {'torch':torch, 'math':math})

    class Container:
        def __init__(self, value):
            self.value = value
        def take(self):
            return self.value

    def attention(q, k, v, heads, mask=None, **kwargs):
        q, k, v = [t.take() if isinstance(t, Container) else t for t in (q,k,v)]
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(q.shape[0],q.shape[2],-1)

    module('comfy.ldm.modules.attention', AttentionTensorContainer=Container,
           ComfyAttention=nn.Identity, optimized_attention_masked=attention,
           optimized_attention=attention)
    rotary = importlib.import_module('comfy.ldm.flux.math')
    extract('comfy.ldm.flux.layers', root/'comfy/ldm/flux/layers.py', {'EmbedND','timestep_embedding'},
            {'torch':torch, 'Tensor':torch.Tensor, 'nn':nn, 'math':math, 'rope':rotary.rope})
    importlib.import_module('comfy.patcher_extension')
    native = importlib.import_module('comfy.ldm.krea2.model')
    module('comfy.k_diffusion', root/'comfy/k_diffusion')
    extract('comfy.k_diffusion.sampling', root/'comfy/k_diffusion/sampling.py',
            {'sample_euler','sample_dpmpp_2m','sample_res_multistep',
             'res_multistep','sample_gradient_estimation',
             'to_d','get_ancestral_step','default_noise_sampler'},
            {'torch':torch, 'trange': lambda n, **kw: range(n),
             'utils': types.SimpleNamespace(append_dims=lambda t,n:t.reshape(t.shape+(1,)*(n-t.ndim)))})
    # Load the native KSAMPLER with an algebraically neutral host model wrapper.
    class Inpaint:
        def __init__(self, owner, sigmas):
            self.owner = owner
            self.inner_model = owner
        def __call__(self, x, sigma, **kw):
            return self.owner(x, sigma, kw.get('model_options', {}), kw.get('seed'))

    sampler_root = pathlib.Path(os.environ.get('COMFYUI_SAMPLER_ROOT', root)).resolve()
    samplers = extract('comfy.samplers', sampler_root/'comfy/samplers.py', {'KSAMPLER'},
                       {'Sampler': type('Sampler', (), {'max_denoise':lambda *args:True}),
                        'KSamplerX0Inpaint': Inpaint, 'torch':torch, 'detail':lambda *a,**k:None})
    return native, comfy, samplers


def tiny_model(native, dtype=torch.float32):
    model = native.SingleStreamDiT(features=32, tdim=8, txtdim=16, heads=2, kvheads=1,
                                   multiplier=1, layers=2, patch=2, channels=2,
                                   txtlayers=2, txtheads=2, txtkvheads=1,
                                   operations=nn, dtype=dtype)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.02)
    return model.eval()
