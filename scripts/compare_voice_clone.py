"""Compare five complete clone runs; keep full and paired-filtered views separate."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen3-tts-0.6b-base": "Qwen 0.6B 联合",
    "qwen3-tts-1.7b-base": "Qwen 1.7B 联合",
    "qwen3-tts-0.6b-base-xvec_only": "Qwen 0.6B xvec",
    "qwen3-tts-1.7b-base-xvec_only": "Qwen 1.7B xvec",
    "breeze-tts-2": "Breeze TTS 2",
}
DATASETS = {"seed_tts_eval_en": "Seed en", "seed_tts_eval_zh": "Seed zh",
            "cv3_zero_shot_en": "CV3 en", "cv3_zero_shot_zh": "CV3 zh",
            "cv3_zero_shot_hard_en": "CV3 hard-en", "cv3_zero_shot_hard_zh": "CV3 hard-zh"}
# Transcribed from the unchanged replication/qwen3_tts.md, dated 2026/02.
# Values: error-rate mean, printed spread (if any), SIM on 0..100 scale, unspecified DNSMOS.
HISTORY = {
    "qwen3-tts-0.6b-base": {
        "seed_tts_eval_en": (1.69, None, 70.55, None), "seed_tts_eval_zh": (1.01, None, 76.48, None),
        "cv3_zero_shot_en": (33.91, 13.06, None, None), "cv3_zero_shot_zh": (3.40, .09, None, None),
        "cv3_zero_shot_hard_en": (10.70, 2.90, 67.04, 3.88), "cv3_zero_shot_hard_zh": (10.70, 1.06, 69.72, 3.82)},
    "qwen3-tts-1.7b-base": {
        "seed_tts_eval_en": (1.58, None, 71.24, None), "seed_tts_eval_zh": (.87, None, 76.89, None),
        "cv3_zero_shot_en": (3.77, .19, None, None), "cv3_zero_shot_zh": (3.12, .07, None, None),
        "cv3_zero_shot_hard_en": (7.90, 1.77, 66.06, 3.91), "cv3_zero_shot_hard_zh": (11.33, 1.43, 70.13, 3.83)},
    "qwen3-tts-1.7b-base-xvec_only": {
        "seed_tts_eval_en": (1.56, None, 59.61, None), "seed_tts_eval_zh": (.78, None, 72.92, None)},
}
PAPER = {"qwen3-tts-0.6b-base": {"seed_tts_eval_en": 1.32, "seed_tts_eval_zh": .92},
         "qwen3-tts-1.7b-base": {"seed_tts_eval_en": 1.24, "seed_tts_eval_zh": .77}}


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--duration-threshold", type=float, default=160.)
    parser.add_argument("--error-threshold", type=float)
    args = parser.parse_args()
    if args.duration_threshold <= 0 or (args.error_threshold is not None and args.error_threshold <= 0):
        parser.error("thresholds must be positive")
    run = ROOT / args.run_dir
    metadata = json.loads((ROOT / "raw_data/voice_clone_manifests/manifest_metadata.json").read_text())
    summary = json.loads((run / "official_summary.json").read_text())
    if len(summary) != 30 or not all(r["complete"] for r in summary):
        raise ValueError("Five complete configurations are required; partial ablations are excluded")
    inputs, scores, excluded = {}, {}, {}
    for model in MODELS:
        inputs[model], scores[model] = {}, {}
        for pattern, target in (("inference-*.jsonl", inputs[model]), ("scores-*.jsonl", scores[model])):
            for file in sorted((run / model).glob(pattern)):
                for line in file.read_text().splitlines():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        key = (row["dataset"], row["index"])
                        if key in target:
                            raise ValueError(f"Duplicate success: {model} {key}")
                        target[key] = row
        if set(inputs[model]) != set(scores[model]):
            raise ValueError(f"Coverage mismatch: {model}")
        for key, row in inputs[model].items():
            score = scores[model][key]["score"]
            error = score["wer%" if key[0].endswith("en") else "cer%"]
            reasons = []
            if row["duration"] > args.duration_threshold:
                reasons.append("duration_above_threshold")
            if args.error_threshold is not None and error > args.error_threshold:
                reasons.append("error_rate_above_threshold")
            if reasons:
                excluded.setdefault(key, []).append({"model": model, "reasons": reasons,
                    "duration_seconds": row["duration"], "error_rate_percent": error, "audio": row["audio"]})
    result = {"rule": vars(args), "filtering": "union of flagged sample IDs, removed from every configuration",
              "historical_baselines_are_unfiltered": True,
              "historical_document_sha256": hashlib.sha256((ROOT / "replication/qwen3_tts.md").read_bytes()).hexdigest(),
              "paper": "https://arxiv.org/html/2601.15621v1#S4.T5", "exclusions": [], "full": [], "filtered": []}
    for (dataset, index), triggers in sorted(excluded.items()):
        result["exclusions"].append({"dataset": dataset, "index": index, "triggers": triggers})
    for view in ("full", "filtered"):
        lines = ["| 配置 | 测试集 | 本次保留/全量 N | 论文错误率 | 原 repo 错误率 | 本次错误率 | 原 repo SIM | 本次 SIM | 原 repo DNSMOS† | 本次 P808 / OVRL |",
                 "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for model, label in MODELS.items():
            for dataset, dslabel in DATASETS.items():
                keys = [k for k in sorted(inputs[model]) if k[0] == dataset and (view == "full" or k not in excluded)]
                count = metadata[dataset]["count"]
                if sum(k[0] == dataset for k in inputs[model]) != count or not keys:
                    raise ValueError(f"Invalid coverage: {model} {dataset}")
                metric_names = ["wer%" if dataset.endswith("en") else "cer%", "simo" if dataset.startswith("seed") else "speaker_sim"]
                if dataset.startswith("cv3"):
                    metric_names += ["P808_MOS", "OVRL"]
                means = {name: statistics.mean(scores[model][k]["score"][name] for k in keys) for name in metric_names}
                error, sim = means[metric_names[0]], 100 * means[metric_names[1]]
                old = HISTORY.get(model, {}).get(dataset, (None, None, None, None))
                paper = PAPER.get(model, {}).get(dataset)
                old_error = fmt(old[0]) + (f"±{old[1]:.2f}" if old[1] is not None else "")
                dns = "—" if dataset.startswith("seed") else f"{fmt(means['P808_MOS'])} / {fmt(means['OVRL'])}"
                lines.append(f"| {label} | {dslabel} | {len(keys)}/{count} | {fmt(paper)} | {old_error} | {fmt(error)} | {fmt(old[2])} | {fmt(sim)} | {fmt(old[3])} | {dns} |")
                result[view].append({"model": model, "dataset": dataset, "retained": len(keys), "total": count,
                    "mean": means, "historical_error": old[0], "historical_error_spread": old[1],
                    "historical_sim_percent": old[2], "historical_dnsmos_unspecified": old[3], "paper_error": paper})
        (run / f"comparison_{view}.md").write_text("\n".join(lines) + "\n")
    (run / "comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (run / "excluded_samples.json").write_text(json.dumps(result["exclusions"], ensure_ascii=False, indent=2) + "\n")
    print("Exclusions shared by all five configurations:")
    for dataset in DATASETS:
        print(dataset, sum(k[0] == dataset for k in excluded), "/", metadata[dataset]["count"])
    print("Total distinct sample IDs excluded:", len(excluded))
    print("Full and paired-filtered comparisons written; all original audio and scores unchanged.")


if __name__ == "__main__":
    main()
