#!/usr/bin/env python3
"""Run actual OptiX FP32 and FP16 backends on identical inference queries.

No teachers, no CDF data, no renderer scene. The probe performs a warm-up launch
and reports one timed launch; this is inference timing, not rendering FPS.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np

def stats(x):
    x=np.asarray(x);finite=np.isfinite(x)
    r={'count':int(x.size),'nonfinite':int((~finite).sum())}
    if finite.any():r.update(mean=float(x[finite].mean()),median=float(np.median(x[finite])),
                            p95=float(np.quantile(x[finite],.95)),max=float(x[finite].max()))
    return r

def compare(a,b):
    return {'log_pdf_absolute_difference':stats(np.abs(a[:,1]-b[:,1])),
            'sample_eval_log_difference':stats(np.abs(a[:,1]-a[:,5])),
            'direction_chord_difference':stats(np.linalg.norm(a[:,2:5]-b[:,2:5],axis=1))}

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--exe',type=Path,required=True);p.add_argument('--ir',type=Path,required=True)
p.add_argument('--asset',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
p.add_argument('--cpu-reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',type=int,default=0);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
rows={};report={'teacher_comparison':False,'kind':'actual OptiX inference','device':a.device}
for mode in ['fp32','coopvec']:
    dest=a.out/(mode+'.txt')
    run=subprocess.run([str(a.exe),str(a.asset),str(a.ir),str(a.queries),str(dest),mode,str(a.device)],capture_output=True,text=True)
    (a.out/(mode+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    if run.returncode:raise SystemExit(f'{mode} failed ({run.returncode}); see {a.out / (mode+".log")}')
    rows[mode]=np.loadtxt(dest);report[mode]={'invalid':int((rows[mode][:,0]!=0).sum()),'timing_log':run.stdout.strip()}
base=np.loadtxt(a.cpu_reference)
if any(x.shape!=base.shape for x in rows.values()):raise SystemExit('Mismatched query/reference count')
queries=np.loadtxt(a.queries);evaluation=queries[:,0]==0;samples=queries[:,0]==1
for mode in ['fp32','coopvec']:
    report[mode]['vs_cpu_eval']=compare(rows[mode][evaluation],base[evaluation])
    report[mode]['vs_cpu_sample']=compare(rows[mode][samples],base[samples])
report['coopvec_vs_gpu_fp32_eval']=compare(rows['coopvec'][evaluation],rows['fp32'][evaluation])
report['coopvec_vs_gpu_fp32_sample']=compare(rows['coopvec'][samples],rows['fp32'][samples])
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
if any(v['invalid'] for k,v in report.items() if k in ['fp32','coopvec']):raise SystemExit('Invalid GPU query results')
