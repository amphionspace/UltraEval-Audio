"""Read-only full-run audit, with a JSON receipt; never repairs or drops samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base", "breeze-tts-2",
          "qwen3-tts-0.6b-base-xvec_only", "qwen3-tts-1.7b-base-xvec_only",
          "qwen3-tts-0.6b-base-icl_only", "qwen3-tts-1.7b-base-icl_only")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(files):
    for file in sorted(files):
        for line in file.read_text().splitlines():
            yield file, json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-name")
    args = parser.parse_args()
    if args.output_name and Path(args.output_name).name != args.output_name:
        parser.error("output-name must be a filename, not a path")
    run_dir = ROOT / args.run_dir
    manifest_dir = ROOT / "raw_data/voice_clone_manifests"
    metadata = json.loads((manifest_dir / "manifest_metadata.json").read_text())
    canonical = {}
    for dataset, spec in metadata.items():
        raw = (manifest_dir / f"{dataset}.jsonl").read_bytes()
        require(hashlib.sha256(raw).hexdigest() == spec["sha256"], f"Manifest changed: {dataset}")
        rows = [json.loads(line) for line in raw.decode().splitlines()]
        require(len(rows) == spec["count"], f"Manifest count: {dataset}")
        for row in rows:
            key = (dataset, row["index"])
            require(key not in canonical, f"Duplicate manifest key: {key}")
            canonical[key] = row
    receipt = {"scope": args.models, "all_seven_configurations": set(args.models) == set(MODELS),
               "samples_per_configuration": len(canonical), "models": {}}
    for model in args.models:
        model_dir = run_dir / model
        generated, scored = {}, {}
        historical_errors = Counter()
        for file, row in records(model_dir.glob("inference-*.jsonl")):
            if row["status"] != "ok":
                historical_errors["inference"] += 1
                continue
            key = (row["dataset"], row["index"])
            require(key in canonical and key not in generated, f"Unknown/duplicate inference: {model} {key}")
            require(row["model"] == model, f"Model identity: {file} {key}")
            for field in ("language", "text", "prompt_text", "prompt_audio", "prompt_sha256", "source_id"):
                require(row[field] == canonical[key][field], f"Input identity: {model} {key} {field}")
            path = model_dir / row["dataset"] / f"{row['index']:06d}.wav"
            require(Path(row["audio"]).resolve() == path.resolve(), f"Audio path: {model} {key}")
            require(row["index"] % 8 == int(file.stem.split("-")[-1]), f"Inference shard: {file} {key}")
            require(row["seed"] == 42 + row["index"] and row["batch_size"] == 1,
                    f"Seed/batch: {model} {key}")
            info = sf.info(path)
            require(info.samplerate == row["sample_rate"] == 24000 and info.channels == 1
                    and info.frames > 0 and info.subtype == "PCM_16", f"WAV format: {path}")
            require(abs(info.duration - row["duration"]) < 1e-8, f"WAV duration: {path}")
            generated[key] = row
        require(set(generated) == set(canonical), f"Incomplete generation: {model} {len(generated)}")
        thread_modes = Counter()
        for file, row in records(model_dir.glob("scores-*.jsonl")):
            if row["status"] != "ok":
                historical_errors["scoring"] += 1
                continue
            key = (row["dataset"], row["index"])
            require(key in generated and key not in scored, f"Unknown/duplicate score: {model} {key}")
            require(row["model"] == model, f"Score model identity: {file} {key}")
            score = row["score"]
            pred = score["pred"]
            if isinstance(pred, dict):
                pred = pred["audio"]
            require(Path(pred).resolve() == Path(generated[key]["audio"]).resolve(), f"Score audio: {model} {key}")
            require(score["ref"] == generated[key]["prompt_audio"], f"Score reference: {model} {key}")
            require(score["label_text"] == generated[key]["text"], f"Score label: {model} {key}")
            if row["dataset"].startswith("cv3"):
                thread_modes[str(row.get("dnsmos_num_threads", "original_default"))] += 1
            scored[key] = row
        require(set(scored) == set(canonical), f"Incomplete scores: {model} {len(scored)}")
        settings = sorted(model_dir.glob("settings-*.json"))
        require(len(settings) == 8, f"Settings count: {model}")
        for shard, file in enumerate(settings):
            cfg = json.loads(file.read_text())
            require(cfg["model"] == model and cfg["shard"] == shard and cfg["num_shards"] == 8
                    and cfg["batch_size"] == 1 and cfg["limit"] == 0, f"Settings identity: {file}")
            if model == "breeze-tts-2":
                require(cfg["fast"] and cfg["sampling"]["warmup_freeze_after_warmup"] is False,
                        f"Breeze graph settings: {file}")
            else:
                require(cfg["icl_only_ablation"] == model.endswith("icl_only"), f"Ablation flag: {file}")
                require(cfg["sampling"]["x_vector_only_mode"] == model.endswith("xvec_only"), f"xvec flag: {file}")
                check = json.loads((model_dir / f"conditioning-verified-{shard:02d}.json").read_text())
                require(check["model"] == model and check["batch_size"] == 1
                        and check["speaker_embedding"] == (not model.endswith("icl_only"))
                        and check["icl_calls"] == (0 if model.endswith("xvec_only") else 1),
                        f"Runtime conditioning: {model} shard {shard}")
        receipt["models"][model] = {"generation_rows": len(generated), "score_rows": len(scored),
                                    "wav_headers_checked": len(generated), "settings_checked": len(settings),
                                    "historical_error_records": dict(historical_errors),
                                    "dnsmos_thread_modes": dict(thread_modes)}
        print(f"PASS {model}: {len(generated)} WAV headers, input/score identities, unique coverage, settings", flush=True)
    receipt["passed"] = True
    target = run_dir / (args.output_name or ("audit.json" if receipt["all_seven_configurations"] else "audit_partial.json"))
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(f"Audit receipt: {target}")


if __name__ == "__main__":
    main()
