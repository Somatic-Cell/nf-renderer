#!/usr/bin/env python3
"""Reject FP64 in emitted PTX, including hidden device-library code."""
import argparse,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('ptx',type=Path);p.add_argument('--report',type=Path)
a=p.parse_args();hits=[]
for n,line in enumerate(a.ptx.read_text().splitlines(),1):
    code=line.split('//',1)[0]
    if re.search(r'\.f64\b|\b0d[0-9a-fA-F]{16}\b',code):hits.append({'line':n,'code':code.strip()})
r={'ptx':str(a.ptx),'fp64_hits':len(hits),'hits':hits};print(json.dumps(r,indent=2))
if a.report:a.report.write_text(json.dumps(r,indent=2)+'\n')
raise SystemExit(1 if hits else 0)
