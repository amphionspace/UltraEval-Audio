import os,json,sys,re
from pathlib import Path
import torch,librosa
from transformers import WhisperProcessor,WhisperForConditionalGeneration,pipeline
U=Path('/workspace/workspace/yanglin/UltraEval-Audio');os.chdir(U);sys.path.insert(0,str(U/'scripts'));from vccm_score import edits
OUT=Path('/home/yanglin/vccm-audit-20261117');rows=json.load(open('/tmp/vccm_wer_rows.json'))
processor=WhisperProcessor.from_pretrained(str(U/'init_model/openai/whisper-large-v3'));norm=processor.tokenizer._normalize
# Diagnostic only: retain spoken text inside parentheses instead of deleting it.
def keep(s):return norm(re.sub(r'[()\[\]<>]',' ',s))
report={}
for subset in ['test','all']:
 rs=[r for r in rows if subset=='all' or r['set']==subset];out={}
 for label in ['v5','gt','cv']:
  rr=[r for r in rs if label=='v5' or r[label]];es=[edits(keep(r['text']),keep((r if label=='v5' else r[label])['hypothesis'])) for r in rr];orig=[edits(r['ref_norm'],(r if label=='v5' else r[label])['hyp_norm']) for r in rr]
  out[label]=dict(n=len(rr),official_wer=100*sum(e[0] for e in orig)/sum(e[1] for e in orig),keep_parentheses_wer=100*sum(e[0] for e in es)/sum(e[1] for e in es),errors_orig=sum(e[0] for e in orig),errors_keep=sum(e[0] for e in es))
 out['parenthesized_rows']=sum('(' in r['text'] or '[' in r['text'] for r in rs)
 r1=[r for r in rs if r['err']<100];out['exclude_repetition_outliers']=dict(n=len(r1),official=100*sum(r['err'] for r in r1)/sum(r['words'] for r in r1),keep=100*sum(edits(keep(r['text']),keep(r['hypothesis']))[0] for r in r1)/sum(len(keep(r['text']).split()) for r in r1))
 report[subset]=out
(OUT/'wer-normalization.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
model=WhisperForConditionalGeneration.from_pretrained(str(U/'init_model/openai/whisper-large-v3'),torch_dtype=torch.float16).cuda().eval()
asr=pipeline('automatic-speech-recognition',model=model,tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device=0,torch_dtype=torch.float16)
keys=['test|856c5b731fd75c84','undomain_timbre|0662f56362bff27b','undomain_style|6de6b0dd51fe7054','test|1877d915b229fcb2','test|afe430890ef28a31','test|055953f04db393e8']
with (OUT/'wer-retranscribe.jsonl').open('a',buffering=1) as f,torch.inference_mode():
 for k in keys:
  r=next(r for r in rows if r['key']==k);wav,_=librosa.load(r['audio'],sr=16000)
  text=asr(wav,generate_kwargs=dict(language='english',task='transcribe'))['text']
  inp=processor(wav,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to('cuda');ids=model.generate(inp.input_features.half(),attention_mask=inp.attention_mask,language='english',task='transcribe',num_beams=5)[0];beam=processor.decode(ids,skip_special_tokens=True)
  out=dict(key=k,duration=r['duration'],text=r['text'],original_hyp=r['hypothesis'],pipeline_hyp=text,beam5_hyp=beam,original_errors=r['err'],pipeline_errors=edits(r['ref_norm'],norm(text))[0],beam5_errors=edits(r['ref_norm'],norm(beam))[0]);f.write(json.dumps(out)+'\n');print(k,'errors',out['original_errors'],out['pipeline_errors'],out['beam5_errors'],flush=True)
