from pathlib import Path
from datetime import datetime, timezone
import json
U=Path('/workspace/workspace/yanglin/UltraEval-Audio'); R=U/'res'; O=Path('/home/yanglin/vccm-audit-20261117')
v5='dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction'
lr1='dots-flexivoice-v5-hifitts2-lr1e-5-interleave50-ep3-step723-default-instruction'
lr2='dots-flexivoice-v5-hifitts2-lr2e-5-interleave50-ep3-step723-default-instruction'
cv='cosyvoice3-latest-vccm'; cvs='cosyvoice3-latest-style_bench-emotion'
D=[]
def add(x=''):D.append(x)
def table(head,rows):
 add('| '+' | '.join(head)+' |');add('|'+'|'.join('---' for _ in head)+'|')
 for row in rows:add('| '+' | '.join(map(str,row))+' |')
 add()
def j(path):return json.loads(path.read_text())
def link(run,sub):return f'[{run}/{sub}](../res/{run}/{sub})'
add('# 语音评测结果汇总：v5 学习率、Vccm、StyleBench 与诊断')
add();add('更新时间：'+datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')+'。这是当前文件与作业状态的快照；运行中的评测尚无最终结论。');add()
add('## 当前结论');add()
add('- v5 学习率实验：三个模型均为同一 v5 数据配方、3 epochs、step723。原版 lr=5e-5，新增 lr=2e-5、1e-5；新增配置继承原版，只覆盖学习率和运行名称。')
add('- 两个较低学习率在 NVBench 四个分组的 CER 均低于原版；PCER 则均高于原版。文字准确性与副语言标签准确性呈现取舍。')
add('- lr=2e-5 的 Seed-TTS 已完成：四个语言/参考模式的错误率都低于原版，speaker_only 英文/中文相对下降约 16.3% / 19.5%。SIM 变化很小。lr=1e-5 尚待完整 Seed-TTS 结果。')
add('- 现有证据还不能确定 1e-5 与 2e-5 的总体优胜者，也不能判断是否保住情感控制：两者 StyleBench 尚未完成汇总，CV3 测试集也仍在运行。')
add('- Vccm 原版 v5、CV3 和真人基线均完成 2,606 条评分，无失败。这里的 v5 是原版 lr=5e-5，新增两个学习率尚未跑 Vccm。')
add('- 情感排名反转已做双向分类器交叉复测：任一分类器下都是 StyleBench 英文 easy 上 v5 更高、Vccm 上 CV3 更高；分类器差异不是唯一原因。')
add('- Vccm 的 v5 常规集 WER 存在一条严重 ASR 重复解码异常；修改解码设置可将该条错误数从 430 降为 1。原始 benchmark 分数与诊断结果分开保留。');add()
add('## 1. v5 不同学习率');add()
status=j(O/'v5-lr-job-status.json')
for r in status:add(f"- {r['label']}：`{r['state']}`，作业 `{r['job']}`。")
add();table(['模型','训练/导出','NVBench','Seed-TTS','CV3 测试集','StyleBench'],[
['原版 5e-5','step723 完成','完成','完成','完成','已有汇总'],
['2e-5','step723 完成','完成','完成','运行中，尚无全量汇总','评分中，尚无全量汇总'],
['1e-5','step723 完成','完成','运行中，尚无全量汇总','尚无全量汇总','运行中，尚无全量汇总']])
add('### NVBench');add();add('CER、PCER、OCER 单位为 %，越低越好；SIM 越高越好。CER 比较文本字符；PCER 比较 `[Tag]` 副语言事件标签序列；OCER 同时计入二者。single/multi 是该测试集的分组命名。');add()
for metric,label in [('cer','CER'),('pcer','PCER'),('ocer','OCER'),('sim','SIM')]:
 rows=[]
 for subset in ['zh_single','zh_multi','en_single','en_multi']:
  vals=[j(R/n/'nv_bench/summary.json')[subset][metric] for n in [v5,lr2,lr1]]
  rows.append([subset]+[f'{v:.4f}' if metric=='sim' else f'{v:.2f}' for v in vals])
 add(f'**{label}**');add();table(['分组','5e-5 原版','2e-5','1e-5'],rows)
add('1e-5 在中文两个分组及英文 multi 的 CER 更低；2e-5 在英文 single 的 CER 更低，并且四个分组 PCER 都低于 1e-5。两个新模型 PCER 均未超过原版。');add()
add('### Seed-TTS');add()
# These completed source tables retain their original precision and coverage information.
for name,label in [(v5,'原版 5e-5'),(lr2,'2e-5')]:
 add('**'+label+'**');add();add((R/name/'seed_tts/full/results_table.md').read_text().strip());add()
add('1e-5 尚未完成，不纳入最终比较。上面的 SIM 为百分制。降低学习率对 Seed-TTS 的改进仅说明这套测试的表现，不能外推为情感能力提升；也没有多随机种子显著性检验。');add()
add('### 学习率实验来源');add()
for n in [v5,lr2,lr1]:add('- '+link(n,'nv_bench/results_table.md'))
add('- '+link(lr2,'seed_tts/full/results_table.md'))
add('- 作业查询快照：[v5-lr-job-status.json](../artifacts/vccm-diagnostics-20261117/v5-lr-job-status.json)。');add()
add('## 2. Vccm：CV3、原版 v5 与真人');add()
add('使用 ControlSpeech 的 VccmDataset 与官方情感分类头、音高/能量函数。CV3 指 Fun-CosyVoice3-0.5B-2512。共覆盖原始 2,686 条中的 2,606 条：常规 1,432/1,500，域外风格 91/100，域外音色 1,083/1,086；缺失来源见 [prepare.json](../raw_data/vccm_v1/prepare.json)。全部可用样本评分完成、失败数为 0。');add()
gt=j(U/'raw_data/vccm_v1/reference/summary-reference.json');a=j(R/v5/'vccm/speaker_only/summary.json');b=j(R/cv/'vccm/speaker_only/summary.json')
for subset,title in [('test','常规'),('undomain_style','域外风格'),('undomain_timbre','域外音色')]:
 add('### '+title);add();rows=[]
 for name,src in [('真人',gt),('CV3',b),('v5 原版',a)]:
  r=src[subset];rows.append([name,r['n'],r['emotion_n']]+[f'{r[k]:.2f}' for k in ['emotion_acc','pitch_acc','energy_acc','pitch_acc_official','wer']]+[f"{r['sim']:.4f}" if r['sim'] is not None else '未计算'])
 table(['音频','n','情感 n','情感 ACC % ↑','独立音高 ACC % ↑','能量 ACC % ↑','官方音高 ACC % ↑','WER % ↓','SIM ↑'],rows)
add('情感准确率只计算有标签样本。官方音高指标要求能量也判断正确，独立音高准确率则单独统计。真人的能量准确率也低，说明绝对阈值与当前音频/标签的一致性需要检查；v5 超过真人的某个指标不能解释为听感超过真人。WER 用 Whisper-large-v3 与其英文归一化，SIM 为 WavLM base-plus-sv x-vector 余弦。语速准确率依赖 MFA，本次未计算。');add()
add('原版 v5 音色相似度三组均更高；CV3 的情感、独立音高和 WER 三组均更好。能量与官方联合音高结果随分组变化。');add()
for n in [v5,cv]:add('- '+link(n,'vccm/speaker_only/summary.json'))
add('- [真人基线](../raw_data/vccm_v1/reference/summary-reference.json)');add()
add('## 3. StyleBench 原版 v5 与 CV3');add()
add('easy 使用中性参考录音，hard 使用与目标情感冲突的参考。speaker_only 不提供参考文本的 ICL 前缀，speaker_icl 同时使用参考文本与音频。下表为情感指令准确率 ACC-I；每个条件每模型 480 条。');add()
sv=j(R/v5/'style_bench/summary.json');sc=j(R/cvs/'style_bench/summary.json');rows=[]
for mode in ['speaker_only','speaker_icl']:
 for lang in ['en','zh']:
  for cond in ['easy','hard']:
   k=f'{lang}|{mode}|{cond}';rows.append([lang,mode,cond,f"{sv['emotion'][k]['acc_i']:.2f}",f"{sc['emotion'][k]['acc_i']:.2f}"])
table(['语言','模式','参考','v5 ACC-I %','CV3 ACC-I %'],rows)
add('StyleBench 情感识别使用 emotion2vec_plus_large；Vccm 使用 emotion2vec_base v2.0.4 的帧特征与 ControlSpeech 微调八分类头。StyleBench 真人分类器基线英文 76.0%（362 条）、中文 51.0%（480 条），与生成样本不构成逐条配对，不能作为严格上限。');add()
add('原版 v5 英文/中文慢速成功率为 59.5% / 80.0%，耳语为 4.8% / 5.0%。CV3 此轮只跑情感部分，尚无副语言风格及自由描述对比。');add()
for n in [v5,cvs]:add('- '+link(n,'style_bench/results.md'))
add()
add('## 4. 情感排名反转的交叉复测');add()
add('对同一份已有生成音频换分类器，没有重新生成。StyleBench 选英文 speaker_only/easy：每模型 480 条；Vccm 选常规集有情感标签的 133 条。Plus-large 取英文六类情感与 neutral 的最大分，ControlSpeech 取八类最大分。');add()
cs=j(O/'emotion-summary.json');rows=[]
for bench in ['style_easy','vccm']:
 for model in ['v5','cv3']:
  r=cs[bench+'|'+model];rows.append([bench,model,r['n'],f"{r['plus_pred']['acc']:.2f}",f"{r['base_pred']['acc']:.2f}"])
table(['数据集','模型','n','Plus-large ACC %','ControlSpeech ACC %'],rows)
add('两个分类器下，排名均随数据集反转。Vccm 六种非中性情感等权平均后，Plus-large 为 CV3 33.88% / v5 26.98%，ControlSpeech 为 CV3 51.17% / v5 38.12%；类别比例也不是唯一解释。');add()
add('已检查两套 CV3 speaker_only 均调用 inference_instruct2，外层提示模板相同；v5 的模板和主要推理默认参数也相同。仍未拆开的因素包括：单一情感命令与复合属性描述、文本、语料来源、参考音频/说话人、训练数据覆盖。当前证据不能指定其中某项为主因。');add()
add('下一步若要定位原因，应在同一文本、参考音频和随机种子下，只替换单一情感指令/复合指令，并补少量盲听；训练数据重合也需要独立核查。');add()
add('- [逐条情感交叉评分](../artifacts/vccm-diagnostics-20261117/emotion-cross.jsonl)');add('- [情感交叉统计](../artifacts/vccm-diagnostics-20261117/emotion-summary.json)');add()
add('## 5. Vccm WER 异常排查');add()
add('原版 v5 常规集为 1,585 次词编辑 / 24,070 个参考词 = 6.58496%。');add()
add('1. 一条 7.68 秒音频 `test|856c5b731fd75c84` 在原始 Whisper 解码下产生大量 be/ever 重复，计入 430 次编辑；StyleBench 的 ASR pipeline 也复现了同一错误。对同一音频改为 num_beams=5，只剩 1 次编辑。这是该条主要受 ASR 解码异常影响的有力证据。')
add('2. 仅排除该条进行敏感性分析，WER 从 6.58% 降到 4.80%；这是诊断结果，不替代官方分数。其他几个所选严重错误在束搜索下仍存在。')
add('3. Whisper English normalizer 会删掉括号中的文本；常规集 7 条文本含括号/方括号。保留括号正文后，v5 WER 为 6.39%，CV3 为 1.75%，真人为 2.26%。影响较小，不能解释大部分差距。')
add('4. 常规集 v5 音频最长 24.16 秒；全量也无超过 30 秒的 v5 音频，未发现 30 秒 ASR 窗口截断是原因。')
add('5. 常规集按文本长度分组，v5 WER：不超过 10 词为 8.86%，11–30 词为 7.97%，超过 30 词为 3.50%。本次不能把高 WER 主要归因于长句。');add()
add('上面是六条选例的解码复核，未对两模型全量统一改用束搜索。其余错误需独立听辨或第二个 ASR 验证，才能区分真实漏读/发音问题与识别器误差。正式分数仍使用原始设置。');add()
add('- [六条 ASR 复核](../artifacts/vccm-diagnostics-20261117/wer-retranscribe.jsonl)');add('- [归一化敏感性统计](../artifacts/vccm-diagnostics-20261117/wer-normalization.json)：其中 all/CV3 为当时 2,380 条快照，只有 test 三方是完整配对。');add()
add('## 6. v6 实验补充');add()
add('两组 v6-earsexp（lr=1e-5、2e-5）已训练到 step780，Seed-TTS/CV3/NVBench/StyleBench 均有汇总。此前对比指出：相较原版 v5，v6 普通朗读错误率多项降低，英文耳语/慢速/音高控制更强，情感指令准确率降低。v5 与 v6 的数据和学习率同时变化，无法据此单独判断 v6 数据的因果影响。');add()
for lr in ['1e-5','2e-5']:
 n=f'dots-flexivoice-v6-earsexp-lr{lr}-interleave50-ep3-step780-default-instruction';add('- '+link(n,'style_bench/results.md'));add('- '+link(n,'seed_tts/full/results_table.md'))
(O/'benchmark-results-20261117.md').write_text('\n'.join(D)+'\n')
print('Document generated',len(D),'blocks')
