#!/usr/bin/env python3
"""Compare migrated NF inference, never teacher CDF arrays.

--reference training is the authoritative parity path and imports the supplied
training project after checking core source hashes. --reference equations is
an explicitly labeled independent PyTorch equation replay for portable tests.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from export_model import load_checkpoint,sha256

def load_bin(path):
    import struct
    with path.open('rb') as f:
        header=struct.unpack('<8s5I4f',f.read(44));data=np.frombuffer(f.read(),dtype='<f4').copy()
    _,_,nw,nl,nc,*_=header
    at=nw;la=data[at:at+nl];at+=nl;mu=data[at:at+nc];at+=nc
    return la,mu,data[at:at+nl*nc].reshape(nl,nc)

def glookup(cond,la,mu,g):
    parts=[]
    for j,axis in enumerate((la,mu)):
        if np.any(cond[:,j]<axis[0]) or np.any(cond[:,j]>axis[-1]):
            raise ValueError('Physical g lookup outside calibration domain')
        k=np.searchsorted(axis,cond[:,j],side='right')-1;k=np.clip(k,0,len(axis)-2)
        t=(cond[:,j]-axis[k])/(axis[k+1]-axis[k]);parts.append((k,t))
    (i,s),(j,t)=parts
    return (1-s)*((1-t)*g[i,j]+t*g[i,j+1])+s*((1-t)*g[i+1,j]+t*g[i+1,j+1])

def metrics(x):
    x=np.asarray(x,dtype=np.float64);good=np.isfinite(x)
    return dict(count=int(x.size),nonfinite=int((~good).sum()),
                **({k:float(v) for k,v in zip(('mean','median','p95','max'),
                 (np.mean(x[good]),np.median(x[good]),np.quantile(x[good],.95),np.max(x[good])))} if good.any() else {}))

def main(a):
    torch.set_num_threads(1);torch.manual_seed(731)
    asset_manifest=json.loads((a.asset.parent/'manifest.json').read_text())
    if sha256(a.asset)!=asset_manifest['asset_sha256']:raise ValueError('Asset hash mismatch')
    ck=load_checkpoint(a.checkpoint);state=ck['model']
    if sha256(a.checkpoint)!=asset_manifest['checkpoint_sha256']:raise ValueError('Wrong checkpoint')
    if a.reference=='training':
        if a.training_repo is None:raise ValueError('--training-repo is required')
        src=a.training_repo/'src'/'phaseflow'
        for name in ['conditional_model.py','hg.py','splines.py','sphere_splines.py']:
            if sha256(src/name)!=asset_manifest['source_sha256'][name]:raise ValueError('Training source hash mismatch: '+name)
        sys.path.insert(0,str(a.training_repo/'src'))
        from phaseflow.conditional_model import ConditionalSphereFlow,ConditionalFlowConfig
        model=ConditionalSphereFlow(ConditionalFlowConfig.from_dict(state['_extra_state']['config']),device='cpu')
        model.load_state_dict(state,strict=True);model.eval()
        prepare=lambda c,g:model.prepare_conditions(c,g)
    else:
        from reference_equations import EquationReplay,prepare
        model=EquationReplay(state)
    la,mu,g=load_bin(a.asset)
    rng=np.random.default_rng(731);n=a.samples
    cond=np.column_stack((rng.uniform(380,830,n),rng.uniform(-.9999,.9999,n))).astype(np.float32)
    # Reference uses the same represented, bilinear physical g table. The
    # comparison isolates inference arithmetic, not table-data demotion.
    gs=glookup(cond.astype(np.float64),la.astype(np.float64),mu.astype(np.float64),g.astype(np.float64))
    ctx=prepare(cond.astype(np.float64),gs)
    u=((rng.integers(0,1<<23,(n,2)).astype(np.float32)+.5)*np.float32(2**-23))
    z=rng.uniform(-1,1,n);phi=rng.uniform(-np.pi,np.pi,n);rr=np.sqrt(1-z*z)
    directions=np.column_stack((rr*np.cos(phi),rr*np.sin(phi),z)).astype(np.float32)
    with torch.no_grad():
        logp=model.log_prob(torch.from_numpy(directions),ctx).numpy()
        sampled,logq=model.sample_from_uniform(torch.from_numpy(u),ctx)
        samplereeval=model.log_prob(sampled,ctx).numpy();sampled=sampled.numpy();logq=logq.numpy()
    a.out.mkdir(parents=True,exist_ok=True)
    # Two query groups: evaluate the SAME directions; sample the SAME uniforms.
    queries=np.vstack((np.column_stack((np.zeros(n),cond,directions)),
                       np.column_stack((np.ones(n),cond,u,np.zeros(n)))))
    inp=a.out/'queries.txt';np.savetxt(inp,queries,fmt='%.9g')
    report=dict(reference=a.reference,samples_per_group=n,seed=731,checkpoint_step=int(ck['step']),
                teacher_data_comparison=False,
                reference_sample_eval_log_difference=metrics(np.abs(logq-samplereeval)))
    for backend in ('fp32','half_reference'):
        output=a.out/(backend+'.txt');start=time.perf_counter()
        subprocess.run([str(a.probe),str(a.asset),str(inp),str(output),backend],check=True)
        seconds=time.perf_counter()-start;v=np.loadtxt(output)
        d=np.linalg.norm(v[n:,2:5]-sampled,axis=1)
        angle=2*np.arcsin(np.minimum(1,d/2))
        report[backend]=dict(cpu_seconds=seconds,invalid=int((v[:,0]!=0).sum()),
            log_pdf_absolute_error=metrics(np.abs(v[:n,1]-logp)),
            common_uniform_direction_angle_rad=metrics(angle),
            sample_log_pdf_error=metrics(np.abs(v[n:,1]-logq)),
            sample_eval_log_difference=metrics(np.abs(v[n:,1]-v[n:,5])))
    report['limitations']=[
        'half_reference is a CPU rounding sensitivity model, not an OptiX/Tensor Core execution.',
        'There is no teacher-CDF or rendering-image comparison.',
        'Physical preparation is FP32 analytic masses versus reference FP64 CDF differences.',
        'Equation replay is not a claim of executing the unmodified Zuko training project.'
    ]
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['fp32']['invalid'] or report['fp32']['log_pdf_absolute_error']['max']>0.01:
        raise SystemExit('FP32 parity gate failed; see report.json')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reference',choices=['training','equations'],default='training')
    p.add_argument('--training-repo',type=Path);p.add_argument('--samples',type=int,default=4096)
    a=p.parse_args()
    if not 16<=a.samples<=1000000:p.error('--samples must be in [16,1000000]')
    main(a)
