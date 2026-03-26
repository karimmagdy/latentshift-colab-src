import json, os, glob
from datetime import datetime

C = datetime(2025, 3, 18, 3, 0)
for f in sorted(glob.glob('results/er_*.json')):
    bn = os.path.basename(f)
    if '_ci' in bn:
        continue
    mt = datetime.fromtimestamp(os.path.getmtime(f))
    d = json.load(open(f))
    a = d['metrics']['average_accuracy'] * 100
    g = d['metrics']['average_forgetting'] * 100
    s = 'FRESH' if mt > C else 'OLD'
    print(f"{bn:45s} acc={a:.1f}  fgt={g:.1f}  {s}  {mt:%H:%M}")
