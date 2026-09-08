# Qwen3-TTS 与 Breeze TTS 2 Voice Clone 评测

评测开始：2026-09-07；结果核对：2026-09-08。
当前交付：五个官方克隆配置、30 个配置×split 单元，21,160 条合成及评分全部完成，并通过逐条审计。
两个 ICL-only 消融已按用户最新要求停止，不纳入以下比较。

## 范围与协议

仅评测 Qwen3-TTS-12Hz-0.6B-Base、Qwen3-TTS-12Hz-1.7B-Base 与 Breeze TTS 2 的语音克隆。
不评测 CustomVoice、VoiceDesign、风格指令控制或其他 TTS 模型。
使用与 [仓库 Qwen3-TTS 复现文档](qwen3_tts.md) 相同的六个中英文 split：
Seed-TTS-Eval en/zh，以及 CV3 zero_shot_en/zh、zero_shot_hard_en/zh。
本次比较每个 Qwen 尺寸的官方 ICL（参考上下文＋speaker embedding，简称“联合”）与 x-vector-only，
以及 Breeze 的官方参考音频＋文本克隆模式。共五个配置，每个配置 4,232 条。
所有配置使用同一份参考音频和目标文本；联合模式另提供参考转写，x-vector-only 不提供参考转写；
不对目标文本增加情绪标签。原计划七组的 ICL-only 两组已暂停，不能声称完成了三路消融比较。

硬件：2 × NVIDIA A800-SXM4-80GB，驱动 570.124.06。
按用户要求另有低负载 GPU 后台计算程序，每卡约占 461 MiB，日志为 `log/gpu_keepalive.log`。
本次主要评估质量；并发和后台负载下的耗时不能与官方单请求 TTFA/RTF 直接比较。

数据下载版本和来源保存在各目录的 `download_provenance.json`。
规范化输入清单保存于 `raw_data/voice_clone_manifests/`，包含原始样本标识和参考音频源字节的 SHA-256。
参考音频经 SoundFile 解码后保存为 PCM16 WAV，保持原始采样率，不额外降噪或裁剪；
后续必要重采样由各模型的原生实现完成。`manifest_metadata.json` 另保存各 JSONL 清单文件的 SHA-256，
汇总时会核验清单未变，并检查各模型使用的目标文本、参考转写及音频标识一致。
逐条合成音频、推理日志和指标记录保存于 `res/voice_clone_20260907/`。

`scripts/audit_voice_clone.py` 另做逐条完整性审计：检查成功记录没有重复、覆盖同一输入清单，
评分中的预测/参考路径及目标文本与合成记录一致，并实际读取每个 WAV 文件头，核对单声道、
24 kHz、PCM16、非空帧数及记录时长；同时检查八分片设置、逐条种子和 Qwen 条件路径验证记录。
五组范围的完整审计回执为 `res/voice_clone_20260907/audit_official.json`：检查了 21,160 个 WAV 文件头、
21,160 条评分身份、40 份分片设置及四个 Qwen 配置的条件验证记录；无重复成功样本，未恢复错误为 0。
回执明确列出五个配置；`all_seven_configurations=false` 表示未做完原七组计划，不是这五组审计失败。

## 评分口径

评分直接调用本仓库注册的 evaluator。额外下载的 Whisper、Paraformer、WavLM、ERes2Net、DNSMOS 是指标计算工具，不是新增待测 TTS 模型。

| 测试集 | 内容一致性 | 说话人相似度 | 语音质量 |
|---|---|---|---|
| Seed-TTS-Eval en | HF Whisper-large-v3；英文 WER | WavLM-large + ECAPA；SIM | — |
| Seed-TTS-Eval zh | SeACo-Paraformer；中文 CER | WavLM-large + ECAPA；SIM | — |
| CV3 en / hard-en | openai-whisper large-v3；英文 WER | ERes2Net；SIM | DNSMOS |
| CV3 zh / hard-zh | SeACo-Paraformer；中文 CER | ERes2Net；SIM | DNSMOS |

WER/CER 按仓库逐句计算后取算术平均，单位为百分比；它与按整个语料累计编辑距离的 corpus WER/CER 不同。
SIM 在原始记录中保留余弦相似度，汇总表乘以 100。Seed 和 CV3 使用不同说话人模型，二者的 SIM 不可直接横比。
DNSMOS 沿用仓库实现：不足 9.01 秒补零，超长音频使用首个 9.01 秒片段，因此不代表整段音频的质量。
本次分别报告 `P808_MOS` 和 `OVRL`，二者都是 DNSMOS 输出，但不是同一子指标。
其他仓库复现文档多使用 P808_MOS；旧 Qwen 文档只写 DNSMOS、没有注明子项，
因此不将其质量分数与本次某个子项直接作差或据此断言质量变化。原始记录保留全部 DNSMOS 子项。
Seed 英文的 HF Whisper 沿用仓库 30 秒输入截断；CV3 英文的 openai-whisper 按完整音频分段转写。
因此超长输出的 Seed WER 不能完整反映 30 秒之后的重复，需结合输出时长与异常样本分析。
失败条数与成功评分数单独列出；有缺失时不将成功子集冒充完整结果。
WER/CER 同时受 ASR、文本归一化和标签质量影响，不等同于人工听写；SIM 与 DNSMOS 也是自动代理指标。
DNSMOS 是预测分数，不是本次开展了人工 MOS 听测。

## 参数

Qwen 使用 Base 的参考音频 + 参考文本 ICL 模式（`x_vector_only_mode=False`）以及
仅说话人向量模式（`x_vector_only_mode=True`，`ref_text=None`）。
原计划的第三个配置是 ICL-only 消融（现已暂停，以下仅保留实现说明）：在该模型实例中使 `generate_speaker_prompt` 返回 `None`，
沿用底层已有的无说话人向量前缀分支，同时保留参考文本及声学 token 的 ICL 路径。
这会移除 speaker embedding 的输入位置，而不是把向量置零；不是官方公开的第三种模式。
它是未重新训练的推理期消融，不能将成绩解释为专门训练的 ICL-only 模型性能。

源码核对（qwen-tts 0.1.1）：默认 ICL 会同时设置 `icl_mode=True` 和提供 `ref_spk_embedding`；
底层在 `x_vector_only_mode or icl_mode` 时注入说话人向量，再根据 `icl_mode` 追加参考文本和声学 token。
所以官方的 ICL 实际是两路结合。x-vector-only 仍接收参考音频提取说话人向量，
虽然当前 SDK 也执行了 tokenizer 编码，但随后将 `ref_code` 设为 `None`，这些音频 token 不用于生成。
这里“only”描述进入生成模型的条件，而不是要求调用方从未提供音频。
若预先构造并保存 `voice_clone_prompt`，生成时可以复用其中的说话人向量，不必再次传入原始参考音频。
本次为各样本直接传入数据集参考音频，x-vector-only 的有效身份条件仍只有 speaker embedding。

| 实验配置 | Speaker embedding | 参考声学 token＋文本 | 定位 |
|---|---|---|---|
| Qwen Base / 官方 ICL | 是 | 是 | 官方组合模式 |
| Qwen x-vector-only | 是 | 否 | 官方仅说话人向量模式 |
| Qwen ICL-only | 否 | 是 | 已暂停的推理期消融，不纳入结果 |
| Breeze 官方 clone | 未发现独立 speaker-encoder 注入支路 | 是 | 保留其官方克隆模板，不直接套用 Qwen 标签 |

沿用仓库采样设置：`max_new_tokens=2048`、`do_sample=True`、`top_k=50`、`top_p=1.0`、
`temperature=0.9`、`repetition_penalty=1.05`；subtalker 使用对应的 50/1.0/0.9 设置。
主生成模型使用 BF16；音频预处理、tokenizer 和评分器沿用各自实现的精度。
具体 attention 后端、批量大小、分片数和种子以结果目录内 `settings-*.json` 为准。
正式运行采用两张 GPU、8 个独立分片进程（每卡 4 个）、单请求 batch size 1；
每条样本种子为 `42 + split 内样本 index`，同一条样本在不同配置中使用相同种子。
Qwen 使用 SDPA。每个 Qwen 批次对是否注入 speaker embedding 及 ICL 调用次数进行运行时断言，
验证记录保存于 `conditioning-verified-*.json`，不只依赖配置名称推断实际路径。

Breeze 使用官方 `ref_edit_tata` 克隆模板、参考音频及其准确转写，`cfg_scale=1.0`；
保留官方 CLI 的通用默认指令 `Speak clearly and naturally.`，不追加声音设计或方向控制指令。
上限 `max_new_tokens=1500`、`max_seq_len=2048`、`repetition_penalty=1.1`。
Breeze 全量使用通过小样本验证的官方 fast path（CUDA graph 预热），具体配置记录于 settings。
正式集的长输入触发了默认预热配置未声明的 `(batch=1, length=288)` 形状；
因此将官方预热配置的 `freeze_after_warmup` 设为 `False`，允许按需捕获新形状。
仍使用同一 CUDA graph 路径、分桶填充和采样参数，不裁剪输入，也没有改成另一种克隆条件。
初始 619 条成功音频保留并断点续算；原始设置另存于 Breeze 结果目录的 `initial_settings/`。
该兼容性处理不适合拿来声称所有请求都是预热后的固定形状延迟测试。
单次随机采样结果不等同于多次运行的均值与标准差。

## 原 repo、论文与本次结果：两张对比表

基线逐项核对自未修改的 [原仓库复现文档](qwen3_tts.md)（标注 2026/02），
以及 [Qwen3-TTS 技术报告 v1 表 5](https://arxiv.org/html/2601.15621v1#S4.T5)。
原 repo 的生成采样配置与本次所列参数一致，但原文没有提供本轮可对齐复算的逐条历史结果与完整运行元数据，
所以“与历史数值接近”不等于严格逐比特复现。

注意三处不能混用的基线：

- 论文表 5 的 12Hz 0.6B / 1.7B Base 中文数值分别为 0.92 / **0.77**，英文为 1.32 / 1.24。
  原 repo 将 1.7B 中文官方值标作 0.78；此处以核对过的论文 v1 为准，历史文件不改写。
- 表 5 未单独列出 x-vector-only 成绩。原 repo 在 xvec 行沿用的括号官方值，不能视为该模式独立的论文结果，
  因而下面 xvec 的论文列留空。论文表 6 的 SIM 来自另一套多语言测试集，不能填入 Seed 的 SIM。
- 没有可填入本次同语种 CV3 四个 split 的对应论文成绩；原 repo 也未列出 0.6B xvec、
  两个 xvec 的 CV3 成绩或 Breeze 成绩。这些格子的“—”是缺少对应基线，不是零分。

### 剔除规则与可比性

“异常”在第二张表中采用明确的操作性定义：**任一已完成配置的输出时长 >160 秒**。
不按 WER/CER 高低挑选样本；弯/直撇号等归一化问题也不据此删样本。
五个配置的异常样本编号取并集，然后在五组中同步移除，保证本次模型间使用相同保留子集。
这是针对接近生成上限的长输出所做的敏感性分析，不声称已识别全部异常。
Qwen 与 Breeze 的生成上限不同，不能仅凭 Breeze 没有 >160 秒输出就断言它没有其他异常。

共 41 个不同样本编号：Seed en 5、Seed zh 0、CV3 en 26、CV3 zh 4、hard-en 3、hard-zh 3。
每组保留 4,191/4,232 条，共 20,955 条模型输出；不是只删各模型自己表现差的样本。
完整清单及触发模型、音频路径、时长、错误率见
[excluded_samples.json](../res/voice_clone_20260907/excluded_samples.json)。
所有过滤仅发生在汇总视图，未删除音频或原始评分。

**原 repo 和论文的逐条输出不可在本轮对齐过滤，因此两张表中的历史/论文列始终是全量基线。**
第二张表只能对本次五个配置做同子集横向比较；与历史/论文的全量数字并列仅供参照，
不能据此计算“同口径超越论文”的结论，也不能为历史结果补造过滤后成绩。

表内错误率为英文 WER / 中文 CER（%），SIM 为余弦相似度×100（%）；错误率越低越好，SIM 越高越好。
† 原 repo 的 DNSMOS 未标子项，不能与本次 P808 或 OVRL 直接作差。
原 repo 的 ± 原样保留；本次是单轮结果，未估计多轮标准差。

### 表 1：不剔除异常，全部样本

| 配置 | 测试集 | 本次保留/全量 N | 论文错误率 | 原 repo 错误率 | 本次错误率 | 原 repo SIM | 本次 SIM | 原 repo DNSMOS† | 本次 P808 / OVRL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 0.6B 联合 | Seed en | 1088/1088 | 1.320 | 1.690 | 4.521 | 70.550 | 70.787 | — | — |
| Qwen 0.6B 联合 | Seed zh | 2020/2020 | 0.920 | 1.010 | 1.116 | 76.480 | 76.666 | — | — |
| Qwen 0.6B 联合 | CV3 en | 500/500 | — | 33.910±13.06 | 51.923 | — | 66.729 | — | 3.695 / 3.124 |
| Qwen 0.6B 联合 | CV3 zh | 500/500 | — | 3.400±0.09 | 3.430 | — | 72.383 | — | 3.806 / 3.285 |
| Qwen 0.6B 联合 | CV3 hard-en | 64/64 | — | 10.700±2.90 | 19.841 | 67.040 | 66.701 | 3.880 | 3.849 / 3.242 |
| Qwen 0.6B 联合 | CV3 hard-zh | 60/60 | — | 10.700±1.06 | 10.886 | 69.720 | 68.453 | 3.820 | 3.740 / 3.264 |
| Qwen 1.7B 联合 | Seed en | 1088/1088 | 1.240 | 1.580 | 1.735 | 71.240 | 71.278 | — | — |
| Qwen 1.7B 联合 | Seed zh | 2020/2020 | 0.770 | 0.870 | 0.958 | 76.890 | 76.958 | — | — |
| Qwen 1.7B 联合 | CV3 en | 500/500 | — | 3.770±0.19 | 4.005 | — | 67.147 | — | 3.750 / 3.165 |
| Qwen 1.7B 联合 | CV3 zh | 500/500 | — | 3.120±0.07 | 3.021 | — | 73.196 | — | 3.832 / 3.312 |
| Qwen 1.7B 联合 | CV3 hard-en | 64/64 | — | 7.900±1.77 | 7.136 | 66.060 | 66.967 | 3.910 | 3.878 / 3.265 |
| Qwen 1.7B 联合 | CV3 hard-zh | 60/60 | — | 11.330±1.43 | 12.912 | 70.130 | 69.452 | 3.830 | 3.817 / 3.337 |
| Qwen 0.6B xvec | Seed en | 1088/1088 | — | — | 1.678 | — | 58.380 | — | — |
| Qwen 0.6B xvec | Seed zh | 2020/2020 | — | — | 0.843 | — | 71.853 | — | — |
| Qwen 0.6B xvec | CV3 en | 500/500 | — | — | 12.785 | — | 58.864 | — | 3.755 / 3.184 |
| Qwen 0.6B xvec | CV3 zh | 500/500 | — | — | 3.169 | — | 67.431 | — | 3.810 / 3.299 |
| Qwen 0.6B xvec | CV3 hard-en | 64/64 | — | — | 5.910 | — | 58.159 | — | 3.874 / 3.269 |
| Qwen 0.6B xvec | CV3 hard-zh | 60/60 | — | — | 9.680 | — | 63.973 | — | 3.789 / 3.323 |
| Qwen 1.7B xvec | Seed en | 1088/1088 | — | 1.560 | 1.522 | 59.610 | 60.642 | — | — |
| Qwen 1.7B xvec | Seed zh | 2020/2020 | — | 0.780 | 0.846 | 72.920 | 73.132 | — | — |
| Qwen 1.7B xvec | CV3 en | 500/500 | — | — | 3.983 | — | 60.772 | — | 3.782 / 3.197 |
| Qwen 1.7B xvec | CV3 zh | 500/500 | — | — | 2.981 | — | 69.771 | — | 3.811 / 3.307 |
| Qwen 1.7B xvec | CV3 hard-en | 64/64 | — | — | 6.403 | — | 58.929 | — | 3.900 / 3.316 |
| Qwen 1.7B xvec | CV3 hard-zh | 60/60 | — | — | 11.327 | — | 67.085 | — | 3.808 / 3.360 |
| Breeze TTS 2 | Seed en | 1088/1088 | — | — | 2.125 | — | 68.884 | — | — |
| Breeze TTS 2 | Seed zh | 2020/2020 | — | — | 1.547 | — | 74.533 | — | — |
| Breeze TTS 2 | CV3 en | 500/500 | — | — | 5.293 | — | 65.147 | — | 3.619 / 3.051 |
| Breeze TTS 2 | CV3 zh | 500/500 | — | — | 4.078 | — | 71.562 | — | 3.700 / 3.163 |
| Breeze TTS 2 | CV3 hard-en | 64/64 | — | — | 7.391 | — | 65.636 | — | 3.728 / 3.172 |
| Breeze TTS 2 | CV3 hard-zh | 60/60 | — | — | 9.491 | — | 69.228 | — | 3.609 / 3.247 |

### 表 2：统一剔除异常长输出对应样本

下面只有“本次”列重新计算；“论文”和“原 repo”列仍为全量参考值。

| 配置 | 测试集 | 本次保留/全量 N | 论文错误率 | 原 repo 错误率 | 本次错误率 | 原 repo SIM | 本次 SIM | 原 repo DNSMOS† | 本次 P808 / OVRL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen 0.6B 联合 | Seed en | 1083/1088 | 1.320 | 1.690 | 1.747 | 70.550 | 70.831 | — | — |
| Qwen 0.6B 联合 | Seed zh | 2020/2020 | 0.920 | 1.010 | 1.116 | 76.480 | 76.666 | — | — |
| Qwen 0.6B 联合 | CV3 en | 474/500 | — | 33.910±13.06 | 4.496 | — | 68.230 | — | 3.716 / 3.140 |
| Qwen 0.6B 联合 | CV3 zh | 496/500 | — | 3.400±0.09 | 3.366 | — | 72.522 | — | 3.806 / 3.285 |
| Qwen 0.6B 联合 | CV3 hard-en | 61/64 | — | 10.700±2.90 | 6.263 | 67.040 | 67.494 | 3.880 | 3.850 / 3.243 |
| Qwen 0.6B 联合 | CV3 hard-zh | 57/60 | — | 10.700±1.06 | 7.243 | 69.720 | 70.138 | 3.820 | 3.757 / 3.283 |
| Qwen 1.7B 联合 | Seed en | 1083/1088 | 1.240 | 1.580 | 1.743 | 71.240 | 71.285 | — | — |
| Qwen 1.7B 联合 | Seed zh | 2020/2020 | 0.770 | 0.870 | 0.958 | 76.890 | 76.958 | — | — |
| Qwen 1.7B 联合 | CV3 en | 474/500 | — | 3.770±0.19 | 3.298 | — | 67.675 | — | 3.760 / 3.173 |
| Qwen 1.7B 联合 | CV3 zh | 496/500 | — | 3.120±0.07 | 2.954 | — | 73.237 | — | 3.832 / 3.312 |
| Qwen 1.7B 联合 | CV3 hard-en | 61/64 | — | 7.900±1.77 | 7.139 | 66.060 | 67.198 | 3.910 | 3.884 / 3.269 |
| Qwen 1.7B 联合 | CV3 hard-zh | 57/60 | — | 11.330±1.43 | 6.056 | 70.130 | 70.220 | 3.830 | 3.838 / 3.345 |
| Qwen 0.6B xvec | Seed en | 1083/1088 | — | — | 1.686 | — | 58.567 | — | — |
| Qwen 0.6B xvec | Seed zh | 2020/2020 | — | — | 0.843 | — | 71.853 | — | — |
| Qwen 0.6B xvec | CV3 en | 474/500 | — | — | 3.632 | — | 59.701 | — | 3.770 / 3.198 |
| Qwen 0.6B xvec | CV3 zh | 496/500 | — | — | 3.111 | — | 67.541 | — | 3.811 / 3.301 |
| Qwen 0.6B xvec | CV3 hard-en | 61/64 | — | — | 6.201 | — | 58.243 | — | 3.872 / 3.271 |
| Qwen 0.6B xvec | CV3 hard-zh | 57/60 | — | — | 6.738 | — | 64.148 | — | 3.814 / 3.324 |
| Qwen 1.7B xvec | Seed en | 1083/1088 | — | 1.560 | 1.529 | 59.610 | 60.643 | — | — |
| Qwen 1.7B xvec | Seed zh | 2020/2020 | — | 0.780 | 0.846 | 72.920 | 73.132 | — | — |
| Qwen 1.7B xvec | CV3 en | 474/500 | — | — | 3.937 | — | 61.368 | — | 3.787 / 3.207 |
| Qwen 1.7B xvec | CV3 zh | 496/500 | — | — | 2.915 | — | 69.794 | — | 3.812 / 3.308 |
| Qwen 1.7B xvec | CV3 hard-en | 61/64 | — | — | 6.520 | — | 58.804 | — | 3.904 / 3.318 |
| Qwen 1.7B xvec | CV3 hard-zh | 57/60 | — | — | 6.361 | — | 67.107 | — | 3.834 / 3.362 |
| Breeze TTS 2 | Seed en | 1083/1088 | — | — | 2.094 | — | 68.869 | — | — |
| Breeze TTS 2 | Seed zh | 2020/2020 | — | — | 1.547 | — | 74.533 | — | — |
| Breeze TTS 2 | CV3 en | 474/500 | — | — | 5.292 | — | 65.713 | — | 3.622 / 3.061 |
| Breeze TTS 2 | CV3 zh | 496/500 | — | — | 4.012 | — | 71.552 | — | 3.701 / 3.164 |
| Breeze TTS 2 | CV3 hard-en | 61/64 | — | — | 7.679 | — | 65.646 | — | 3.725 / 3.178 |
| Breeze TTS 2 | CV3 hard-zh | 57/60 | — | — | 7.728 | — | 69.252 | — | 3.627 / 3.255 |

### 核对后的主要结论

- **0.6B 联合模式的高错误率主要受长尾影响。** Seed en 全量 4.521%，原 repo 1.69%，论文 1.32%；
  统一过滤后为 1.747%。CV3 en 则从 51.923% 降至 4.496%，原 repo 全量为 33.91±13.06%。
  这支持长异常输出显著拉高均值，但不能把过滤后的 4.496% 当成全量复现成绩。
- **1.7B 联合模式与历史结果的差距较小，但没有达到所引用论文的 Seed 数值。**
  全量 Seed en 1.735%，对比原 repo 1.58%、论文 1.24%；Seed zh 0.958%，对比 0.87%、0.77%。
  本次 Seed SIM 为 71.278 / 76.958，历史为 71.24 / 76.89；WavLM 来源差异仍需保留，见下载记录。
- **x-vector-only 的内容分数与身份相似度有取舍。** 全量下两个尺寸的 xvec 在六个 split 的均值错误率
  均低于各自联合配置，但 SIM 也全部更低。1.7B xvec 的 Seed en/zh 错误率 1.522 / 0.846，
  原 repo 为 1.56 / 0.78，不能概括成所有项都提升；过滤后个别配置之间的内容指标排序也会变化。
- **Breeze 的优势不能脱离异常处理口径。** hard-zh 全量 CER 为 9.491%，低于 1.7B 联合的 12.912%；
  同步过滤后则为 7.728% 对 6.056%，排序反转。两张表应一起阅读。
  Breeze 的 Seed SIM 在两种口径下都低于两个 Qwen 联合配置，但高于两个 xvec 配置。
- 第二张表衡量“排除指定长尾样本后的表现”；第一张表保留长尾风险，仍是这次完整测试集的主结果。
  没有人工听测或多随机种子验证，不宣称统计显著差异。

Breeze 截至本次检索未找到独立 arXiv 技术报告；技术信息来自 [官方发布博客](https://breezeblue.ai/breeze-tts-2)、
[官方推理仓库](https://github.com/breezeblue-ai/breeze-tts) 和 [HF 模型卡](https://huggingface.co/BreezeBlue/Breeze-TTS-2)。
官方支持参考音频 + 转写的英中文 voice clone，其设计与方向控制榜单不能替代本次 Seed/CV3 克隆测试。
官方低延迟数据采用 H100 和预热 fast path，与本次 A800 并行质量评测条件不同。

## Breeze TTS 2 的结构与克隆方式

以下来自本次下载的官方源码及检查点配置，是源码核对结果，不冒充未找到的技术报告。
源码版本：`43e2ea1595297c4059477e2e4a300653761c759b`；
HF 检查点版本：`799624c0b4a1daa8db6d28bbd9850043c0270734`。

| 组件 | 本次检查点中的设置 | 作用 |
|---|---|---|
| 文本编码器 | `t5gemma2_text`，26 层，hidden size 1152 | 编码文本条件，再投影到语音骨干维度 |
| 自回归骨干 | `backbone_model_type=qwen3`，28 层，hidden size 2048 | 逐帧预测首个声学码本 token |
| Depth decoder | 12 层，hidden size 1024；16 个码本 | 根据骨干状态和首码本补齐当前帧其余码本 |
| 运行时音频 tokenizer | `audio_tokenizer/` 内的 Qwen3-TTS tokenizer | 编码参考音频，并将生成的声学 token 还原成 24 kHz 音频 |

模型主体两个 safetensors 分片合计含 3,483,206,497 个存储张量元素，主要是 BF16；
这个数字含各子模块，不等同于骨干规模，也不含单独的 `audio_tokenizer/` 文件。
配置中的 `backbone_flavor=llama-1B` 是命名字段，实际类型由 `backbone_model_type=qwen3` 和骨干配置决定，
不能据此宣称整个模型仅有 1B 参数。权重还含 `codec_model`，但官方 `load_runtime` 明确加载独立的
`Qwen3TTSTokenizer` 供克隆和波形解码使用，不能仅从旧 codec 配置字段推断运行时路径。
进一步对本地文件校验：Breeze 的 `audio_tokenizer/model.safetensors` 与两种 Qwen Base 的
`speech_tokenizer/model.safetensors` 三份文件具有相同 SHA-256：
`836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`。
这证明捆绑的 tokenizer 权重相同，不代表包含流式切块等细节在内的整个解码流程相同。

官方克隆输入按照“参考文本 → 参考音频 token → 目标文本”组织；CLI 的 `ref_edit_tata` 模板在目标文本前
另保留通用指令 `Speak clearly and naturally.`。`S0` 是序列中的说话人标记，不是由参考音频提取的独立 x-vector。
在本次核对的官方克隆路径中，没有发现 Qwen Base 那样单独提取和注入 speaker embedding 的支路，
因此其身份条件主要通过参考语音上下文传入，不能机械拆成 Qwen 的三种配置。
本次按官方默认 `cfg_scale=1.0` 测试，不把 CFG 的参考/指令分支或 fast path 当成新的克隆模式。
该版本仅公开注册 `tts_instruction`（无参考音频，本次不测）和 `ref_edit_tata`（本次克隆路径）。
内部 `_ref_clone_tata_segments` 用于 CFG 负向/参考分支，并非另一个公开模板。

检查入口：`third_party/breeze-tts/breeze_infer/runtime.py`、`breeze_infer/templates.py`、
`models/breeze.py`、`models/fast_streaming.py` 和本地模型 `config.json`。

## 已完成配置的异常与边界核查

<!-- VOICE_CLONE_RESULTS_START -->
### 已完成：Qwen 0.6B 联合模式

以下保留异常与评分边界的核查细节。五组覆盖率完整，不代表每条生成内容正确；所有异常长输出均参与表 1。


已核对的异常与评分边界：

- Seed en 只有 1/1088 条输出超过 30 秒，该条也是 163.76 秒的极端长输出：
  [样本 #1006](../res/voice_clone_20260907/qwen3-tts-0.6b-base/seed_tts_eval_en/001006.wav)。
  ASR 转写反复出现 `A pile of`，单句 WER 为 3018.182%，贡献了该 split 错误率总和的 61.36%。
  **主结果仍为 4.521%，不删除该样本。** 仅作影响分解：拿掉这一条后其余 1087 条的均值为 1.749%；
  这个数不是替代成绩。该 split 的单句 WER 中位数为 0，935/1088 条单句 WER 为 0。
- CV3 en 有 20/500 条输出超过 160 秒（4.0%），均值 WER 51.923% 而中位数为 0；
  最差 10 条贡献错误率总和的 78.10%。这支持“少量极端长尾明显影响均值”的判断，
  但单轮实验不能验证历史文档的跨随机种子标准差。
- [Seed en #227](../res/voice_clone_20260907/qwen3-tts-0.6b-base/seed_tts_eval_en/000227.wav)
  的标签是 `You aren’t positive, you’re negative.`，ASR 转写使用直撇号，自动 WER 为 40%。
  该例显示当前归一化口径对弯/直撇号敏感，不能仅凭这个分数断言发音错误。

### 已完成：Qwen 1.7B 联合模式

4,232 条合成及评分全部完成，未恢复推理/评分错误均为 0。


其 Seed en/zh 最长输出分别为 9.28/13.36 秒，无超过 30 秒的输出。
CV3 en 仍有 1/500 条 163.76 秒输出，hard-zh 也有 1/60 条；不是完全消除了长尾。
对照论文表 5，本次 Seed en 1.735% 高于 1.24%，中文 0.958% 高于 0.77%；
这是本次协议下的实测对照，不声称逐项复现了论文全部生成和评分细节。

### 已完成：Breeze TTS 2 官方克隆

4,232 条合成及评分全部完成；预热形状故障对应的 `CV3 hard-zh #49` 已恢复成功，未恢复错误为 0。


本次 Breeze 最长输出为 46.8 秒；Seed 和 CV3 普通集均没有超过 30 秒的输出。
与上述 Qwen 联合配置比较，Breeze 的 Seed 错误率高于 Qwen 1.7B 联合模式，
Seed SIM 也较低；CV3 hard-zh 的均值 CER 则较低（9.491% 对 12.912%）。
但该 split 的单句 CER 中位数为 3.055%，高于 Qwen 1.7B 的 1.828%，
说明均值差异不能解释成所有困难样本都更好，需结合长尾看待。
这些结果只评价本次克隆路径，不评价 Breeze 的声音设计或指令控制能力。

### 已完成：Qwen 0.6B x-vector-only

4,232 条合成及评分全部完成，没有推理或评分错误记录。


与同尺寸联合模式相比，本次 x-vector-only 在六个 split 的均值 WER/CER 均更低，
但六个 split 的 SIM 也均更低；Seed en/zh SIM 分别下降约 12.41/4.81 个百分点。
因此内容一致性分数的提升不能等同于克隆身份保真度提升。
CV3 en 超过 160 秒的输出由联合模式的 20/500 减少为 6/500，但并未消失。

尤其需要注意 Seed 英文的评分窗口：该模式有四条 163.76 秒输出（#66、#770、#771、#790），
它们在现有前 30 秒 ASR 评分下的 WER **全部为 0**。
例如 [#66](../res/voice_clone_20260907/qwen3-tts-0.6b-base-xvec_only/seed_tts_eval_en/000066.wav)。
这不是证明整段输出完全正确，也不能据此宣称该模式没有停止生成的问题；
本次不删除这些样本，也不将前 30 秒的内容指标当作完整尾段检查。

### 已完成：Qwen 1.7B x-vector-only

4,232 条合成及评分全部完成，没有推理或评分错误记录；逐条完整性审计通过。


与同尺寸联合模式相比，六个 split 的均值 WER/CER 更低，但 SIM 也全部更低；
Seed en/zh SIM 分别下降约 10.64/3.83 个百分点。
CV3 en 的 WER 仅从 4.005% 变为 3.983%，不应将这种单轮微小差异解读为统计显著的提升。
本次该配置没有超过 160 秒的输出；Seed en/zh 最长为 10.64/11.36 秒，
全部输出最长为 hard-zh 的 54.4 秒。相比 0.6B x-vector-only，本次未出现其 163.76 秒长输出，
但这仍只是一次固定种子运行的观察，不保证以后不会发生。

两个尺寸的 ICL-only 已按用户要求停止、不参与两张表。0.6B 停止时保留 690 条成功合成、447 条成功评分；
1.7B 未启动正式全量运行。独立 smoke 目录中的预检也不计入结果。
逐条记录、完整精度和错误率最差样本见 `scores-*.jsonl`、`official_summary.json` 与 `official_outliers.json`。
另有 `official_long_outputs.json` 列出所有超过 30 秒的输出，包括 WER 为 0 的长样本。
超过 30 秒只是便于核查的时长阈值，不自动等于失败；困难集正常长文本也可能超过该值。
<!-- VOICE_CLONE_RESULTS_END -->

## 下载与复现记录

模型优先由 Hugging Face 下载，失败可回退 hf-mirror。完整下载后写入模型 repo revision 与来源。
WavLM 原始 Google Drive 链接下载失败。本次使用 HF `hidoba/wavlm_large_finetune` 的同名检查点，
其包含 `model` 与 `best_valid_eer`，模型 state dict 含 711 个张量。
逐键审计确认推理网络的 710 个条目全部匹配、没有缺失键；唯一额外的
`loss_calculator.projection.weight` 是该推理网络不使用的训练损失头，沿用仓库 `strict=False` 加载时被忽略。
审计记录见 `log/wavlm_checkpoint_compatibility_checked.log`。
由于未取得 Google Drive 原文件，尚不能验证两份文件逐字节相同；不能据此声称复现了历史 SIM 的完全相同权重。
骨干使用 S3PRL 官方 HF `s3prl/converted_ckpts/wavlm_large.pt`，通过新增的可选
`WAVLM_LARGE_BACKBONE` 环境变量离线加载，保持原评分网络与余弦相似度计算。

CV3 的官方 Whisper 检查点 SHA-256 已验证为
`e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb`，与 OpenAI 下载地址中的摘要一致。
本次七个 TTS 权重文件与九个评分权重文件的完整校验清单分别为
`res/voice_clone_20260907/tts_artifacts.sha256` 和 `res/voice_clone_20260907/metric_artifacts.sha256`。

运行脚本：`scripts/download_voice_clone.py`、`scripts/prepare_voice_clone.py`、`scripts/voice_clone_infer.py`、`scripts/voice_clone_score.py`。
所有正式结果均使用完整 split；单独的 smoke 目录仅用于验证环境，不纳入正式结果。

仓库起始 commit：`637cc336f8495c164eb837ae8e8c01737bc91fef`。
推理环境：Python 3.10、torch/torchaudio 2.9.1（CUDA 12.8）、qwen-tts 0.1.1、transformers 4.57.3。
评分环境：torch/torchaudio 2.5.1、transformers 4.49.0、funasr 1.2.6、s3prl 0.4.18、openai-whisper 20250625。
完整依赖保存在 `log/tts_environment.txt` 和 `log/metrics_environment.txt`。
评分按语言分配 GPU，Seed 每种语言 2 个样本分片，CV3 每种语言 4 个；不改变 evaluator 或指标公式。
并发 CV3 评分中观察到 DNSMOS 单进程默认创建约 192 个线程，引发 CPU 争用。
后续评分设置 `DNSMOS_NUM_THREADS=2`，限制 ONNX 线程池并关闭空闲自旋；未设置该变量时仓库默认行为不变。
使用八条覆盖中英文普通/困难集、短音频和 163.76 秒长输出的已评分样本回归，
全部七个 DNSMOS 子项与默认线程结果的最大绝对差为 `6.93e-6`（日志：`log/dnsmos_threads_regression.log`）。
这是微小的浮点执行差异，不是逐比特一致；音频预处理、权重和评分公式均未改变。
启用后的新增 CV3 逐条评分记录保存 `dnsmos_num_threads=2`；本次此前无该字段的记录使用原默认线程配置。
已有约 164 秒异常输出通过 WavLM 全长评分验证，单进程峰值分配显存约 18.23 GiB（不含 ASR），
因此 Seed 保留并发显存余量，不裁剪长音频来规避资源问题。

原始七组启动命令（历史记录；已停止，当前不要直接重启此命令，否则会恢复 ICL-only）：

```bash
envs/runner/bin/python -u scripts/launch_voice_clone.py --workers 8 --batch-size 1
```

支持对成功且音频仍存在的样本断点续算。原始七组调度器状态保存在 `pipeline_status.json`，
其 `KeyboardInterrupt` 来自本次按用户指令停止，不代表五个已完成配置失效。
本次交付以五组严格汇总 `official_summary.json` 和审计 `audit_official.json` 为准：
30 个单元均合成数=评分数=预期数，必需指标全部有限，未恢复推理/评分错误为 0。
默认汇总仍检查原七组；核对当前五组时必须显式选择配置，不能用 `--allow-incomplete` 冒充严格通过。

重新生成当前两张表（只读取已有结果，不启动合成或评分）：

```bash
envs/runner/bin/python scripts/compare_voice_clone.py --duration-threshold 160
```

完整精度及过滤规则：`res/voice_clone_20260907/comparison_summary.json`；
独立表格：`comparison_full.md`、`comparison_filtered.md`。

运行中一次独立启动的 Qwen 1.7B Seed 中文辅助评分未继承本地 `ffmpeg` 路径，
产生两条评分错误记录。已在评分脚本中显式加入 `.tools/bin` 到 PATH 后重试，
不重新生成音频、不改变评分算法；原错误记录和重试日志均保留。
日志为 `log/voice_clone_qwen17_catchup_seed_zh.log` 与 `log/voice_clone_qwen17_catchup_seed_zh_retry.log`。
最终覆盖率以成功评分去重后的样本数和未恢复错误数核验，不以日志中是否曾出现错误判定模型质量。

Breeze 首轮在 619 条成功输出后因上述未声明预热形状停止，留下一条推理错误记录。
失败时的完整调度状态保存在 `res/voice_clone_20260907/pipeline_status_failed_breeze_prefill.json`；
恢复时跳过已成功且音频仍存在的样本，保留错误记录并重试失败样本。
这里的运行时失败与生成内容的 WER/CER 错误分开统计。

本次修复了 registry 误读 `.ipynb_checkpoints` 中重复 YAML 的问题，不修改正式注册项。
`tests/test_registry_checkpoints.py` 两项本地回归测试通过：忽略 checkpoint 副本，仍拒绝真正重复项。
