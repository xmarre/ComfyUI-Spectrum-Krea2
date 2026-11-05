#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Paired saved-image PSNR, SSIM, optional LPIPS, and side-by-side panels."""
import argparse
import json
import pathlib
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=pathlib.Path)
    p.add_argument('--lpips',action='store_true',help='Requires lpips and torchvision; first use may download AlexNet weights')
    args=p.parse_args()
    metric=None
    if args.lpips:
        import torch
        import lpips
        metric=lpips.LPIPS(net='alex').eval()
    rows=[]
    for path in sorted(args.directory.glob('*_native.png')):
        candidate=path.with_name(path.name.replace('_native.png','_spectrum.png'))
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        a=Image.open(path).convert('RGB'); b=Image.open(candidate).convert('RGB')
        if a.size!=b.size:
            raise ValueError('Paired images must have identical dimensions')
        x=np.asarray(a,dtype=np.float32)/255.; y=np.asarray(b,dtype=np.float32)/255.
        mse=float(np.mean((x-y)**2))
        row={'native':path.name,'spectrum':candidate.name,'psnr_db':None if mse==0 else float(peak_signal_noise_ratio(x,y,data_range=1)),
             'identical':mse==0,'ssim':float(structural_similarity(x,y,channel_axis=2,data_range=1)), 'lpips':None}
        if metric is not None:
            with torch.inference_mode():
                tx=torch.from_numpy(x.copy()).permute(2,0,1).unsqueeze(0)*2-1
                ty=torch.from_numpy(y.copy()).permute(2,0,1).unsqueeze(0)*2-1
                row['lpips']=float(metric(tx,ty))
        panel=Image.new('RGB',(a.width*2,a.height+32),'black')
        panel.paste(a,(0,32)); panel.paste(b,(a.width,32))
        draw=ImageDraw.Draw(panel); draw.text((8,8),'Native',fill='white');draw.text((a.width+8,8),'Spectrum',fill='white')
        panel.save(path.with_name(path.name.replace('_native.png','_comparison.png')))
        rows.append(row)
    if not rows:
        raise SystemExit('No paired native PNG files found')
    (args.directory/'fidelity.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    print(json.dumps(rows,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
