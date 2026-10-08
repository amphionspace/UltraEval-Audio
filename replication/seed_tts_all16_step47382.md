# All16 step 47382：Seed-TTS 评测入口

训练方式、16 个数据集、模块初始化与冻结、历史 Qwen 对比、诊断结论和 greedy 运行命令已统一到
[LM-TTS-Training 训练与评测报告](../../LM-TTS-Training/docs/training/all16-training-and-seedtts-evaluation.md)。两仓库位于同一父目录时可直接跳转。

只查看线上仓库时，使用 [LM-TTS-Training 中的同一文档](https://github.com/amphionspace/LM-TTS-Training/blob/main/docs/training/all16-training-and-seedtts-evaluation.md)。

本仓库相对路径：

| 内容 | 位置 |
| --- | --- |
| 已完成的采样 ICL | `res/all16-step47382-seedtts-20261008-8gpu/full/` |
| 全量 greedy ICL | `res/all16-step47382-seedtts-greedy-20261008-8gpu/` |
| 运行入口 | `scripts/run_seed_tts.py` |
| 历史 Qwen 复现 | [voice_clone_20260907.md](voice_clone_20260907.md) |

2026-10-08 文档整理时，greedy 尚在生成；speaker-only 暂停。最终完成以运行目录
`pipeline_status.json` 的 `complete` 和 `full/audit.json` 的 `passed: true` 为准。
结果在 `full/summary.json` 与 `full/results_table.md`，生成完成后会自动执行 ASR / SIM。

一次性诊断音频和脚本已清理，选样、分数与审计摘要保存在统一文档旁的
[evidence/seed-tts-step47382.json](../../LM-TTS-Training/docs/training/evidence/seed-tts-step47382.json)。
