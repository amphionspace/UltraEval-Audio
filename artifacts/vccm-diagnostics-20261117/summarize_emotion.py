import json,collections
from pathlib import Path
O=Path('/home/yanglin/vccm-audit-20261117');rs=[json.loads(l) for l in (O/'emotion-cross.jsonl').read_text().splitlines()];g=collections.defaultdict(list)
for r in rs:g[(r['bench'],r['model'])].append(r)
report={}
for key,rows in sorted(g.items()):
 out={'n':len(rows)}
 for pred in ['base_pred','plus_pred']:
  vals=[r for r in rows if pred in r]
  if not vals:continue
  out[pred]={'acc':100*sum(r[pred]==r['target'] for r in vals)/len(vals),'per_class':{},'pred_counts':dict(collections.Counter(r[pred] for r in vals))}
  for target in sorted(set(r['target'] for r in vals)):
   tt=[r for r in vals if r['target']==target];out[pred]['per_class'][target]={'n':len(tt),'acc':100*sum(r[pred]==target for r in tt)/len(tt)}
  out[pred]['macro_six']=sum(v['acc'] for t,v in out[pred]['per_class'].items() if t!='neutral')/len([t for t in out[pred]['per_class'] if t!='neutral'])
 report['|'.join(key)]=out
(O/'emotion-summary.json').write_text(json.dumps(report,indent=2))
for k,v in report.items():print(k,v['n'],{kk:round(vv['acc'],2) for kk,vv in v.items() if kk.endswith('_pred')})
