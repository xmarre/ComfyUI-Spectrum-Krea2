#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate matched ComfyUI UI workflows and API prompts from one definition."""
import copy
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parents[1]/'examples'


def graph(preset,enabled,reference=False):
    cfg=4.5 if preset=='raw' else 1.0
    steps=52 if preset=='raw' else 8
    out={
      '1':{'class_type':'UNETLoader','inputs':{'unet_name':f'krea2_{preset}.safetensors','weight_dtype':'default'}},
      '2':{'class_type':'CLIPLoader','inputs':{'clip_name':'qwen3_vl_4b.safetensors','type':'krea2','device':'default'}},
      '3':{'class_type':'VAELoader','inputs':{'vae_name':'wan_2.1_vae.safetensors'}},
      '4':{'class_type':'CLIPTextEncode','inputs':{'clip':['2',0],'text':'A candid photograph of a watchmaker holding a tiny brass gear with both hands, detailed fingertips and engraved metal, daylight through a workshop window, a handwritten card reading SPECTRUM on the desk.'}},
      '5':{'class_type':'CLIPTextEncode','inputs':{'clip':['2',0],'text':''}},
      '6':{'class_type':'EmptySD3LatentImage','inputs':{'width':1024,'height':1024,'batch_size':1}},
      '7':{'class_type':'SpectrumApplyKrea2','inputs':{'model':['1',0],'enabled':enabled,'preset':preset,'degree':1,'ridge':.01,'warmup':4,'tail':2,'window':1,'growth':0.,'max_history':8,'storage':'vram','budget_mb':4096,'debug':True}},
      '8':{'class_type':'KSampler','inputs':{'model':['7',0],'positive':['4',0],'negative':['5',0],'latent_image':['6',0],'seed':42,'steps':steps,'cfg':cfg,'sampler_name':'euler','scheduler':'simple','denoise':1.}},
      '9':{'class_type':'VAEDecode','inputs':{'samples':['8',0],'vae':['3',0]}},
      '10':{'class_type':'SaveImage','inputs':{'images':['9',0],'filename_prefix':f'SpectrumKrea2/{preset}_'+('spectrum' if enabled else 'native')}},
    }
    if reference:
        out['11']={'class_type':'LoadImage','inputs':{'image':'reference.png'}}
        out['12']={'class_type':'VAEEncode','inputs':{'pixels':['11',0],'vae':['3',0]}}
        for branch,text in ((13,'4'),(15,'5')):
            out[str(branch)]={'class_type':'ReferenceLatent','inputs':{'conditioning':[text,0],'latent':['12',0]}}
            out[str(branch+1)]={'class_type':'FluxKontextMultiReferenceLatentMethod','inputs':{'conditioning':[str(branch),0],'reference_latents_method':'index_timestep_zero'}}
        out['8']['inputs']['positive']=['14',0]
        out['8']['inputs']['negative']=['16',0]
        out['4']['inputs']['text']='Keep the identity and arrangement from the reference image. Change the lighting to warm late-afternoon window light, preserve small details and readable lettering.'
    return out


# Native ComfyUI socket types and widget ordering at the audited revision.
SOCKETS={
 'UNETLoader':([], [('MODEL','MODEL')]), 'CLIPLoader':([], [('CLIP','CLIP')]),
 'VAELoader':([], [('VAE','VAE')]), 'CLIPTextEncode':([('clip','CLIP')],[('CONDITIONING','CONDITIONING')]),
 'EmptySD3LatentImage':([], [('LATENT','LATENT')]),
 'SpectrumApplyKrea2':([('model','MODEL')],[('MODEL','MODEL')]),
 'KSampler':([('model','MODEL'),('positive','CONDITIONING'),('negative','CONDITIONING'),('latent_image','LATENT')],[('LATENT','LATENT')]),
 'VAEDecode':([('samples','LATENT'),('vae','VAE')],[('IMAGE','IMAGE')]),
 'SaveImage':([('images','IMAGE')],[]), 'LoadImage':([], [('IMAGE','IMAGE'),('MASK','MASK')]),
 'VAEEncode':([('pixels','IMAGE'),('vae','VAE')],[('LATENT','LATENT')]),
 'ReferenceLatent':([('conditioning','CONDITIONING'),('latent','LATENT')],[('CONDITIONING','CONDITIONING')]),
 'FluxKontextMultiReferenceLatentMethod':([('conditioning','CONDITIONING')],[('CONDITIONING','CONDITIONING')]),
}
POSITIONS={1:(0,0),2:(0,330),3:(0,650),4:(370,330),5:(370,630),6:(760,650),7:(370,0),8:(1120,0),9:(1490,0),10:(1830,0),11:(0,950),12:(370,950),13:(760,950),14:(1120,950),15:(760,1180),16:(1120,1180)}


def ui(api):
    nodes=[];links=[];lookup={}
    for key,node in api.items():
        idx=int(key);kind=node['class_type'];ins,outs=SOCKETS[kind]
        input_names={name for name,_ in ins}
        widgets=[v for name,v in node['inputs'].items() if name not in input_names]
        if kind=='KSampler': widgets.insert(1,'fixed')
        if kind=='LoadImage': widgets.append('image')
        record={'id':idx,'type':kind,'pos':list(POSITIONS[idx]),'size':[330,470 if kind=='SpectrumApplyKrea2' else 270],
                'flags':{},'order':idx,'mode':0,'inputs':[{'name':name,'type':typ,'link':None} for name,typ in ins],
                'outputs':[{'name':name,'type':typ,'links':[]} for name,typ in outs],
                'properties':{'Node name for S&R':kind},'widgets_values':widgets}
        lookup[key]=record;nodes.append(record)
    for key,node in api.items():
        for slot,(name,typ) in enumerate(SOCKETS[node['class_type']][0]):
            value=node['inputs'].get(name)
            if isinstance(value,list):
                origin,output=value
                link=len(links)+1
                links.append([link,int(origin),output,int(key),slot,typ])
                lookup[origin]['outputs'][output]['links'].append(link)
                lookup[key]['inputs'][slot]['link']=link
    return {'last_node_id':max(int(k) for k in api),'last_link_id':len(links),'nodes':nodes,'links':links,'groups':[],'config':{},'extra':{'ds':{'scale':.65,'offset':[50,50]}},'version':.4}


def main():
    ROOT.mkdir(exist_ok=True)
    for preset in ('raw','turbo'):
        for enabled in (False,True):
            for reference in ((False,True) if preset=='raw' else (False,)):
                name=preset+('_reference' if reference else '')+'_'+('spectrum' if enabled else 'native')
                api=graph(preset,enabled,reference)
                (ROOT/(name+'.api.json')).write_text(json.dumps(api,indent=2)+'\n')
                (ROOT/(name+'.json')).write_text(json.dumps(ui(api),indent=2)+'\n')


if __name__=='__main__':
    main()
