from pathlib import Path
import json,hashlib,collections,wave
ROOT=Path('/workspace/workspace/yanglin/UltraEval-Audio')
res=ROOT/'res/dots-flexivoice-raw-interleave50-ep2-step932-default-instruction/style_bench'
bench=ROOT/'raw_data/style_bench_v1';out=Path(__file__).resolve().parent;web=out/'web';web.mkdir(exist_ok=True);(web/'audio').mkdir(exist_ok=True)
records={};files={};missing=[]
for mode in ['speaker_only','speaker_icl']:
 records[mode]={}
 for p in sorted((res/mode).glob('inference-*.jsonl')):
  for line in p.read_text().splitlines():
   r=json.loads(line);records[mode][r['key']]=r

def audio(path):
 p=Path(path)
 if not p.is_file():missing.append(str(p));return None
 key=hashlib.sha256(str(p).encode()).hexdigest()[:24]+p.suffix
 dest=web/'audio'/key
 if not dest.exists():dest.symlink_to(p)
 files[key]=str(p)
 return 'audio/'+key

items=[]
for subset in ['paraling_zh','paraling_en','emotion_zh','emotion_en','free_zh','free_en','free_ja','free_ko']:
 for line in (bench/(subset+'.jsonl')).read_text().splitlines():
  r=json.loads(line);kind=subset.split('_')[0];entry={k:r[k] for k in ['id','text','instruction','speaker','language']};entry.update(subset=subset,kind=kind,target=r.get('target','free'),tracks=[])
  conds=['easy','hard'] if kind=='emotion' else ['style','default'] if kind=='paraling' else ['instr','default']
  for cond in conds:
   rid=r['pair'] if kind=='paraling' and cond=='default' else r['id'];key=f'{subset}|{rid}|{cond}'
   for mode in records:
    gen=records[mode].get(key)
    if gen:
     assert gen['text']==r['text']
     entry['tracks'].append(dict(mode=mode,condition=cond,url=audio(gen['audio']),duration=gen['duration'],instruction=gen['instruction'],seed=gen['seed']))
    else:missing.append(key+' '+mode)
  entry['refs']=[dict(label=cond,emotion=r.get('ref_'+cond+'_emotion'),text=r['ref_'+cond+'_text'],url=audio(r['ref_'+cond])) for cond in ['easy','hard']] if kind=='emotion' else [dict(label='reference',text=r['ref_text'],url=audio(r['ref']))]
  if r.get('gt'):entry['gt']=audio(r['gt'])
  items.append(entry)
assert not missing,missing[:10]
# Round-robin styles/speakers so the first page covers more than one speaker or style.
groups=collections.defaultdict(list)
for r in items:groups[(r['subset'],r['target'])].append(r)
for g in groups.values():g.sort(key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())
ordered=[]
while any(groups.values()):
 for g in groups.values():
  if g:ordered.append(g.pop())
(web/'samples.json').write_text(json.dumps(ordered,ensure_ascii=False))
(web/'file-map.json').write_text(json.dumps(files,ensure_ascii=False,indent=2))
summary={'examples':len(items),'generated_audio':sum(len(v) for v in records.values()),'served_audio_files':len(files),'subsets':dict(collections.Counter(x['subset'] for x in items)),'missing':missing}
(web/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
