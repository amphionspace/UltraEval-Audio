import json,sys,os
from pathlib import Path
import numpy as np
import torch
from funasr import AutoModel
import funasr.utils.install_model_requirements as requirements
requirements.install_requirements=lambda path:True
U=Path('/workspace/workspace/yanglin/UltraEval-Audio');os.chdir(U)
sys.path.insert(0,str(U/'scripts'));from vccm_score import E2V_BASE,EMOTION_MAP,CS
sys.path.insert(0,str(CS/'emotion_acc'));from ckpt_utils import load_ckpt
from emotion_net import Emotion2Vec_finetume
OUT=Path('/home/yanglin/vccm-audit-20261117')
def read(d,pat):
 out={}
 for p in d.glob(pat):
  for l in p.read_text().splitlines():
   try:r=json.loads(l);out[r['key']]=r
   except ValueError:pass
 return out
names={'v5':'dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction','cv3':'cosyvoice3-latest-vccm'}
manifest={r['id']:r for r in map(json.loads,(U/'raw_data/style_bench_v1/emotion_en.jsonl').read_text().splitlines())}
vc=[];st=[]
for name,d in names.items():
 run=U/'res'/d/'vccm/speaker_only';inf=read(run,'inference-*.jsonl');scores=read(run,'scores-[0-9]*.jsonl')
 for k,r in scores.items():
  if r['set']=='test' and 'emotion_pred' in r:vc.append(dict(key=name+'|'+k,model=name,bench='vccm',target=r['emotion'],audio=inf[k]['audio'],base_pred=r['emotion_pred'],source=r['source']))
 sd=U/'res'/(d if name=='v5' else 'cosyvoice3-latest-style_bench-emotion')/'style_bench/speaker_only'
 scores=read(sd,'scores-[0-9]*.jsonl')
 for k,r in read(sd,'inference-*.jsonl').items():
  if r['subset']=='emotion_en' and r['condition']=='easy':
   m=manifest[r['item']];sc=scores[k]['emotion'];cand=['angry','happy','sad','fearful','surprised','disgusted','neutral']
   st.append(dict(key=name+'|'+k,model=name,bench='style_easy',target=m['target'].replace('fearful','fear'),audio=r['audio'],plus_pred=max(cand,key=lambda c:sc.get(c,0)).replace('fearful','fear'),plus_scores=sc,speaker=m['speaker']))
for r in map(json.loads,(U/'raw_data/vccm_v1/vccm_test.jsonl').read_text().splitlines()):
 if r['emotion'] in EMOTION_MAP:vc.append(dict(key='gt|test|'+r['id'],model='gt',bench='vccm',target=r['emotion'],audio=r['gt'],source=r['source']))
p=OUT/'emotion-cross.jsonl';done=read(OUT,'emotion-cross.jsonl')
print('planned',len(vc),len(st),flush=True)
plus=AutoModel(model=str(U/'init_model/emotion2vec/models/iic--emotion2vec_plus_large/snapshots/master'),device='cuda:0',disable_update=True,disable_pbar=True,log_level='ERROR')
with p.open('a',buffering=1) as f,torch.inference_mode():
 for i,r in enumerate(vc):
  if r['key'] in done:continue
  e=plus.generate(r['audio'],granularity='utterance',extract_embedding=False)[0];sc={k.split('/')[-1].replace('fearful','fear'):float(v) for k,v in zip(e['labels'],e['scores'])};cand=['angry','happy','sad','fear','surprised','disgusted','neutral'];r.update(plus_scores=sc,plus_pred=max(cand,key=lambda c:sc.get(c,0)));f.write(json.dumps(r)+'\n')
  if i%25==0:print('plus',i,len(vc),flush=True)
del plus;torch.cuda.empty_cache()
base=AutoModel(model=str(E2V_BASE),device='cuda:0',disable_update=True,disable_pbar=True,log_level='ERROR')
head=Emotion2Vec_finetume(768,8);load_ckpt(head,str(CS/'emotion_acc/emotion2vec_recognizer_seq_10w'),'model');head.eval().cuda();labels={v:k for k,v in EMOTION_MAP.items()}
with p.open('a',buffering=1) as f,torch.inference_mode():
 for i,r in enumerate(st):
  if r['key'] in done:continue
  e=base.generate(r['audio'],granularity='frame',extract_embedding=True)[0]['feats'];logits=head(torch.tensor(np.asarray(e),dtype=torch.float32)[None].cuda())['logits'][0];r.update(base_pred=labels[int(logits.argmax())],base_scores={labels[j]:float(x) for j,x in enumerate(logits.softmax(-1))});f.write(json.dumps(r)+'\n')
  if i%25==0:print('base',i,len(st),flush=True)
print('COMPLETE',flush=True)
