"""Aggregate unique successful rows; expose incomplete coverage and outliers."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base", "breeze-tts-2",
          "qwen3-tts-0.6b-base-xvec_only", "qwen3-tts-1.7b-base-xvec_only",
          "qwen3-tts-0.6b-base-icl_only", "qwen3-tts-1.7b-base-icl_only")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-prefix", default="")
    args = parser.parse_args()
    if "/" in args.output_prefix or "\\" in args.output_prefix:
        parser.error("output-prefix must be a filename prefix, not a path")
    run_dir = ROOT / args.run_dir
    metadata = json.loads((ROOT / "raw_data/voice_clone_manifests/manifest_metadata.json").read_text())
    canonical = {}
    for file in (ROOT / "raw_data/voice_clone_manifests").glob("*.jsonl"):
        raw = file.read_bytes()
        if file.stem not in metadata or hashlib.sha256(raw).hexdigest() != metadata[file.stem]["sha256"]:
            raise ValueError(f"Input manifest changed: {file}")
        for line in raw.decode("utf-8").splitlines():
            sample = json.loads(line)
            canonical[(sample["dataset"], sample["index"])] = sample
    identity_fields = ("language", "text", "prompt_text", "prompt_sha256", "prompt_audio", "source_id")
    table, outliers, long_outputs = [], [], []
    for model in args.models:
        model_dir = run_dir / model
        inf, scores, failures, score_failures = {}, {}, set(), set()
        for file in sorted(model_dir.glob("inference-*.jsonl")):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                key = (row["dataset"], row["index"])
                if row.get("status") == "ok" and Path(row["audio"]).is_file():
                    expected = canonical.get(key)
                    if (expected is None or row.get("model") != model
                            or any(row.get(k) != expected.get(k) for k in identity_fields)):
                        raise ValueError(f"Input identity mismatch: {model} {key}")
                    inf[key] = row
                else:
                    failures.add(key)
        for file in sorted(model_dir.glob("scores-*.jsonl")):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if row.get("status") == "ok":
                    scores[(row["dataset"], row["index"])] = row["score"]
                else:
                    score_failures.add((row["dataset"], row["index"]))
        for dataset, spec in metadata.items():
            generated = {k: v for k, v in inf.items() if k[0] == dataset}
            scored = {k: v for k, v in scores.items() if k in generated}
            metrics = {}
            for score in scored.values():
                for name, value in score.items():
                    if (isinstance(value, (float, int)) and not isinstance(value, bool)
                            and math.isfinite(value)):
                        metrics.setdefault(name, []).append(value)
            required = ["wer%" if dataset.endswith("en") else "cer%"]
            required += ["simo"] if dataset.startswith("seed") else ["speaker_sim", "OVRL", "P808_MOS"]
            row = {"model": model, "dataset": dataset, "expected": spec["count"],
                   "generated": len(generated), "scored": len(scored),
                   "unrecovered_inference_errors": sum(k[0] == dataset and k not in inf for k in failures),
                   "unrecovered_scoring_errors": sum(k[0] == dataset and k not in scores for k in score_failures),
                   "complete": (len(generated) == len(scored) == spec["count"]
                                and all(len(metrics.get(k, [])) == spec["count"] for k in required)),
                   "metric_counts": {k: len(v) for k, v in metrics.items()},
                   "mean": {k: statistics.mean(v) for k, v in metrics.items()}}
            durations = sorted(v["duration"] for v in generated.values())
            row["duration"] = ({"mean_seconds": statistics.mean(durations),
                                "median_seconds": statistics.median(durations),
                                "p95_seconds": durations[min(len(durations)-1, int(.95*len(durations)))],
                                "max_seconds": max(durations),
                                "over_30_seconds": sum(d > 30 for d in durations),
                                "over_160_seconds": sum(d > 160 for d in durations)} if durations else {})
            errors = sorted(metrics.get(required[0], []))
            row["error_distribution"] = ({
                "median_percent": statistics.median(errors),
                "p90_percent": errors[min(len(errors)-1, int(.9*len(errors)))],
                "zero_error_utterances": sum(e == 0 for e in errors),
                "over_50_percent": sum(e > 50 for e in errors),
                "over_100_percent": sum(e > 100 for e in errors),
                "top_10_utterances_error_sum_share": sum(errors[-10:])/sum(errors) if sum(errors) else 0,
            } if errors else {})
            table.append(row)
            errkey = "wer%" if dataset.endswith("en") else "cer%"
            for key, sample in generated.items():
                if sample["duration"] > 30:
                    score = scored.get(key, {})
                    long_outputs.append({"model": model, "dataset": dataset, "index": key[1],
                                         "duration": sample["duration"], "audio": sample["audio"],
                                         "text": sample["text"], "error_rate_percent": score.get(errkey),
                                         "transcription": score.get("transcription", "")})
            worst = sorted(scored.items(), key=lambda kv: kv[1].get(errkey, -1), reverse=True)[:10]
            for key, score in worst:
                outliers.append({"model": model, "dataset": dataset, "index": key[1],
                                 "error_rate_percent": score.get(errkey), "text": generated[key]["text"],
                                 "transcription": score.get("transcription", ""),
                                 "audio": generated[key]["audio"], "duration": generated[key]["duration"]})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{args.output_prefix}summary.json").write_text(json.dumps(table, ensure_ascii=False, indent=2))
    (run_dir / f"{args.output_prefix}outliers.json").write_text(json.dumps(outliers, ensure_ascii=False, indent=2))
    (run_dir / f"{args.output_prefix}long_outputs.json").write_text(json.dumps(long_outputs, ensure_ascii=False, indent=2))
    lines = ["| 模型 | 数据集 | 合成/总数 | 评分/总数 | CER或WER/% ↓ | SIM/% ↑ | DNSMOS P808 ↑ | DNSMOS OVRL ↑ | 状态 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in table:
        m = row["mean"]
        error = m.get("wer%", m.get("cer%"))
        sim = m.get("simo", m.get("speaker_sim"))
        dns = m.get("OVRL")
        def fmt(x):
            return "—" if x is None else f"{x:.3f}"
        lines.append(f"| {row['model']} | {row['dataset']} | {row['generated']}/{row['expected']} | {row['scored']}/{row['expected']} | {fmt(error)} | {fmt(None if sim is None else sim*100)} | {fmt(m.get('P808_MOS'))} | {fmt(dns)} | {'完整' if row['complete'] else '未完成'} |")
    (run_dir / f"{args.output_prefix}results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not args.allow_incomplete and not all(row["complete"] for row in table):
        sys.exit("Full evaluation is incomplete; consult summary.json for missing coverage.")


if __name__ == "__main__":
    main()
