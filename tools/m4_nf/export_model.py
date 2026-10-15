#!/usr/bin/env python3
"""Export a conditional_circular_hg_v1 checkpoint. No teacher arrays are read.

The fixed architecture is deliberately checked, not silently approximated.
The original FP32 parameters, physical g table, cross sections and source
identities are retained. CoopVec FP16 conversion happens in the GPU packer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()

def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()

def load_checkpoint(path: Path):
    # Narrow allow-list for NumPy scalar metadata; never unrestricted pickle.
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype,
                                          np.dtypes.Float64DType, np.dtypes.Float32DType]):
        return torch.load(path, map_location='cpu', weights_only=True)

def export(snapshot: Path, catalog_file: Path, out: Path, selection: str):
    config = json.loads((snapshot / 'config.json').read_text(encoding='utf-8-sig'))
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf-8-sig'))
    receipts = json.loads((snapshot / 'snapshot.json').read_text(encoding='utf-8-sig'))
    path = snapshot / selection
    # Explicit filename/selection, never silently choose 'last' for 'best'.
    for name in (selection,'config.json','manifest.json'):
        if sha256(snapshot/name)!=receipts['files'][name]:
            raise ValueError('Snapshot file hash mismatch: '+name)
    ck = load_checkpoint(path)
    cfg = config['model']
    required = dict(hidden_features=[64,64,64], num_coupling_layers=4, num_bins=64,
                    mirror_symmetry=True, min_bin_width=1e-5, min_bin_height=1e-5,
                    min_derivative=1e-4, wavelength_bounds_nm=[380,830],
                    incident_cosine_bounds=[-1,1], encoding='raw')
    known=set(required)|{'one_blob_bins','grid_resolutions','grid_features'}
    if set(cfg)-known:raise ValueError('Unknown model config fields')
    for key,value in required.items():
        if cfg.get(key) != value: raise ValueError(f'Unsupported {key}: {cfg.get(key)!r}')
    if ck['identity_sha256'] != manifest['identity_sha256']:
        raise ValueError('Checkpoint/training manifest identity mismatch')
    if digest(manifest['identity']) != manifest['identity_sha256']:
        raise ValueError('Training identity hash mismatch')
    state = ck['model']
    if state.get('_extra_state') != {'family':'conditional_circular_hg_v1','config':cfg}:
        raise ValueError('Checkpoint extra-state mismatch')
    arrays=[]; expected={'_extra_state'}; tensor_desc=[]
    for c in range(4):
        dims=[3,64,64,64,193 if c%2==0 else 97]
        for l in range(4):
            for suffix,shape in [('weight',(dims[l+1],dims[l])),('bias',(dims[l+1],))]:
                key=f'couplings.{c}.hyper.{2*l}.{suffix}';expected.add(key)
                t=state[key]
                if t.dtype != torch.float32 or tuple(t.shape)!=shape or not bool(torch.isfinite(t).all()):
                    raise ValueError(f'Invalid tensor: {key}')
                a=t.detach().numpy().astype('<f4',copy=False).ravel()
                tensor_desc.append(dict(key=key,shape=list(shape),offset_floats=sum(len(x) for x in arrays)))
                arrays.append(a)
    if set(state)!=expected: raise ValueError('Unexpected/missing checkpoint tensors')
    weights=np.concatenate(arrays)
    if len(weights)!=72004: raise ValueError('Parameter count mismatch')
    catalog=json.loads(catalog_file.read_text(encoding='utf-8-sig'))
    cfp=digest({k:v for k,v in catalog.items() if k not in ('root','fingerprint')})
    if catalog['fingerprint']!=cfp or cfp!=manifest['identity']['catalog_fingerprint']:
        raise ValueError('Catalog fingerprint mismatch')
    rows=catalog['records'];cond=np.asarray([r['condition'] for r in rows],dtype=np.float64)
    la=np.unique(cond[:,0]);mu=np.unique(cond[:,1])
    if len(la)*len(mu)!=len(rows): raise ValueError('Physical table is not a complete grid')
    fields=np.full((3,len(la),len(mu)),np.nan)
    li={float(v):i for i,v in enumerate(la)};mi={float(v):i for i,v in enumerate(mu)}
    for row in rows:
        a,b=map(float,row['condition']);i,j=li[a],mi[b]
        x=row['quality']['cross_sections']
        if x['model']!='geometric_nonabsorbing' or x['absorption_m2']!=0 or x['scattering_m2']!=x['extinction_m2']:
            raise ValueError('Unsupported cross-section model')
        if math.isfinite(fields[0,i,j]): raise ValueError('Duplicate condition')
        fields[:,i,j]=row['g'],x['scattering_m2'],x['extinction_m2']
    if not np.isfinite(fields).all():raise ValueError('Missing physical node')
    # Analytic HG masses use these constants instead of FP32 CDF subtraction.
    theta=np.arange(65,dtype=np.float64)*math.pi/64
    dm=2*np.cos(theta/2)**2;dp=2*np.sin(theta/2)**2
    dm[0],dm[-1],dp[0],dp[-1]=2,0,0,2
    delta=2*np.sin((theta[:-1]+theta[1:])/2)*math.sin(math.pi/128)
    angles=np.concatenate((dm,dp,delta)).astype('<f4')
    out.mkdir(parents=True,exist_ok=True)
    asset=out/'model.bin'
    if asset.exists():raise FileExistsError(f'{asset}; choose an empty output directory')
    with asset.open('wb') as f:
        f.write(struct.pack('<8s5I4f',b'NFHG4A1\0',1,72004,len(la),len(mu),int(ck['step']),380,830,1e-5,1e-4))
        for a in (weights,la,mu,fields.ravel(),angles):f.write(np.asarray(a,dtype='<f4').tobytes())
    result=dict(schema='nf_renderer.m4_nf.asset.v1',selected_file=selection,selected_step=int(ck['step']),
                architecture=required,checkpoint_sha256=sha256(path),asset_sha256=sha256(asset),
                catalog_fingerprint=cfp,source_sha256=manifest['identity']['source_sha256'],
                tensors=tensor_desc,condition_axes=dict(wavelengths=len(la),cosines=len(mu),
                cosine_range=[float(mu[0]),float(mu[-1])]),
                field_interpolation='bilinear in recorded wavelength_nm and signed incident cosine',
                condition_extrapolation='reject; no implicit endcap policy',
                physical_preparation='FP32 analytic HG bin mass identity; algebraically equivalent to FP64 CDF differences',
                nn_precision='FP32 stored; CoopVec input/weights/bias/output FP16; SiLU and flow FP32',
                no_teacher_arrays_read=True,model_claim='learned single-particle forward density, not reciprocity-enforced')
    (out/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['selected_file','selected_step','asset_sha256','condition_axes']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',required=True,type=Path)
    ap.add_argument('--catalog',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--selection',choices=['best.pt','checkpoint.pt'],default='best.pt')
    args=ap.parse_args();export(args.snapshot,args.catalog,args.out,args.selection)
