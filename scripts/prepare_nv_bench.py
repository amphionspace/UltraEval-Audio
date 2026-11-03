"""Unpack NV-Bench (CharlesNi/NV-Bench parquet) into prompt / reference WAVs and per-subset manifests.

Usage: python scripts/prepare_nv_bench.py [--source raw_data/nv_bench] [--output raw_data/nv_bench/prepared]
"""
import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("zh_single", "zh_multi", "en_single", "en_multi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "raw_data/nv_bench")
    parser.add_argument("--output", type=Path, default=ROOT / "raw_data/nv_bench/prepared")
    args = parser.parse_args()
    for subset in SUBSETS:
        rows = pq.read_table(args.source / f"data/{subset}/test.parquet").to_pylist()
        directory = (args.output / subset).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index, row in enumerate(rows):
            paths = {}
            for key, name in (("prompt_wav", "prompt"), ("wav", "gt")):
                audio, rate = sf.read(io.BytesIO(row[key]["bytes"]))
                paths[name] = directory / f"{index:04d}_{name}.wav"
                sf.write(paths[name], audio, rate)
            manifest.append({"subset": subset, "index": index, "language": subset[:2],
                             "text": row["text"], "prompt_text": row["prompt_text"],
                             "category": row["category"], "prompt_audio": str(paths["prompt"]),
                             "gt_audio": str(paths["gt"])})
        (args.output / f"{subset}.jsonl").write_text(
            "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
        print(subset, len(manifest), flush=True)


if __name__ == "__main__":
    main()
