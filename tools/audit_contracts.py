#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare inspected native function contracts; --write records a reviewed source.

Updating digests does not establish compatibility. Review the source and run the
native integration tests before recording a new audited revision.
"""
import argparse
import ast
import pathlib
import subprocess
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from comfyui_spectrum_krea2.compat import source_digest
from comfyui_spectrum_krea2.contracts import CONTRACTS

REQUESTS={
    'comfy/ldm/krea2/model.py':{
        'SingleStreamDiT':['_forward','forward','process_img','_unpack_context'],
        **{name:['forward'] for name in ['LastLayer','RMSNorm','QKNorm','SwiGLU','Attention','SimpleModulation','DoubleSharedModulation','TextFusionBlock','TextFusionTransformer','SingleStreamBlock']}},
    'comfy/k_diffusion/sampling.py':{None:['sample_euler','sample_dpmpp_2m',
                                           'sample_res_multistep','res_multistep',
                                           'sample_gradient_estimation']},
    'comfy/samplers.py':{'KSAMPLER':['sample']},
    'comfy/ldm/flux/layers.py':{None:['timestep_embedding']},
}


def collect(root):
    contracts={}
    for path,groups in REQUESTS.items():
        source=(root/path).read_text()
        tree=ast.parse(source)
        for cls,names in groups.items():
            body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
            for name in names:
                node=next(n for n in body if isinstance(n,ast.FunctionDef) and n.name==name)
                lines=source.splitlines(keepends=True)
                first=min([node.lineno]+[n.lineno for n in node.decorator_list])-1
                code=''.join(lines[first:node.end_lineno])
                prefix={'SingleStreamDiT':'model','LastLayer':'last','KSAMPLER':'sampler'}.get(cls,cls or 'native')
                value=source_digest(code)
                contracts[prefix+'.'+name]=value
                if cls in {'SingleStreamDiT','LastLayer'} and name=='forward':
                    contracts[cls+'.forward']=value
    return contracts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-root',type=pathlib.Path,required=True)
    p.add_argument('--write',action='store_true')
    args=p.parse_args()
    found=collect(args.comfy_root)
    if args.write:
        revision=subprocess.check_output(['git','-C',str(args.comfy_root),'rev-parse','HEAD'],text=True).strip()
        path=pathlib.Path(__file__).resolve().parents[1]/'comfyui_spectrum_krea2/contracts.py'
        import pprint
        path.write_text('# SPDX-License-Identifier: GPL-3.0-or-later\n# Canonical AST SHA-256 contracts: ComfyUI '+revision+'\nCONTRACTS = '+pprint.pformat(found,sort_dicts=True)+'\n')
    else:
        changed=sorted(k for k in set(found)|set(CONTRACTS) if found.get(k)!=CONTRACTS.get(k))
        if changed:
            details = '; '.join(f"{key}: expected {CONTRACTS.get(key)!r}, audited {found.get(key)!r}" for key in changed)
            raise SystemExit('Native source contracts differ: ' + details)
        print(f'{len(found)} native source contracts match')


if __name__=='__main__':
    main()
