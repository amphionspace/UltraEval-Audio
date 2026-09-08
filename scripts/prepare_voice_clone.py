"""Materialize the exact six voice-clone splits used by replication/qwen3_tts.md."""
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = {}
    specs = [(f"seed_tts_eval_{lang}", "TwinkStart/Seed-TTS-Eval", f"{lang}/*.parquet", lang)
             for lang in ("en", "zh")]
    specs += [(f"cv3_{split}", "yuekai/CV3-Eval", f"data/{split}-*.parquet", split.rsplit("_", 1)[1])
              for split in ("zero_shot_en", "zero_shot_zh", "zero_shot_hard_en", "zero_shot_hard_zh")]
    manifest_dir = ROOT / "raw_data" / "voice_clone_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, repo, pattern, lang in specs:
        source = ROOT / "raw_data" / repo
        files = sorted(source.glob(pattern))
        if not files or not (source / "download_provenance.json").exists():
            raise RuntimeError(f"Incomplete download: {source}")
        audio_dir = manifest_dir / name
        audio_dir.mkdir(exist_ok=True)
        records = []
        for file in files:
            for row in pq.read_table(file).to_pylist():
                index = len(records)
                audio = row["audio" if name.startswith("seed") else "prompt_audio"]
                wav, sr = sf.read(io.BytesIO(audio["bytes"]))
                path = audio_dir / f"{index:06d}.wav"
                sf.write(path, wav, sr, subtype="PCM_16")
                records.append({"index": index, "dataset": name, "language": lang,
                                "text": row["text" if name.startswith("seed") else "target_text"],
                                "prompt_text": row["prompt_text"], "prompt_audio": str(path),
                                "source_id": str(row.get("filename", row.get("id", index))),
                                "prompt_sha256": hashlib.sha256(audio["bytes"]).hexdigest()})
        output = manifest_dir / f"{name}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
        counts[name] = {"count": len(records), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
        print(name, len(records), flush=True)
    (manifest_dir / "manifest_metadata.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
