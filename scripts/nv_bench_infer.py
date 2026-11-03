"""Generate NV-Bench audio for one shard with a dots.tts export (speaker + ICL from the NV-Bench prompt).

Resumable: a per-shard journal records finished samples; one process per shard (flock).
Usage: envs/dots/bin/python scripts/nv_bench_infer.py --model-path DIR --output DIR --shard K --num-shards N
"""
import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from local_dots import DotsVoiceClone

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("zh_single", "zh_multi", "en_single", "en_multi")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "raw_data/nv_bench/prepared")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--gpu-memory-gib", type=float, default=0, help="PyTorch allocator cap; 0 = none")
    parser.add_argument("--gpu-reserve-gib", type=float, default=2, help="Free memory left outside the cap")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f".infer-{args.shard:02d}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = [json.loads(line) for subset in SUBSETS
            for line in (args.manifest_dir / f"{subset}.jsonl").read_text().splitlines()]
    rows = [r for k, r in enumerate(rows) if k % args.num_shards == args.shard]
    journal = args.output / f"inference-{args.shard:02d}.jsonl"
    done = set()
    if journal.exists():
        done = {(r["subset"], r["index"]) for r in map(json.loads, journal.read_text().splitlines())}
    remaining = [r for r in rows if (r["subset"], r["index"]) not in done]
    print(f"shard={args.shard} remaining={len(remaining)}", flush=True)
    if not remaining:
        return
    if args.gpu_memory_gib:
        free, total = torch.cuda.mem_get_info(0)
        budget = min(args.gpu_memory_gib * 1024**3, free - args.gpu_reserve_gib * 1024**3)
        if budget < 4 * 1024**3:
            raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free; need 4 GiB plus the reserve")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    adapter = DotsVoiceClone(args.model_path, conditioning="speaker_icl")
    with journal.open("a", buffering=1) as log, torch.inference_mode():
        for i, row in enumerate(remaining):
            [wav], rate = adapter.generate([row], 42 + row["index"])
            if len(wav) == 0 or not np.isfinite(wav).all():
                raise ValueError(f"Empty or non-finite audio: {row['subset']} {row['index']}")
            path = args.output / row["subset"] / f"{row['index']:04d}.wav"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, wav, rate, subtype="PCM_16")
            log.write(json.dumps({**row, "audio": str(path), "sample_rate": rate,
                                  "duration": len(wav) / rate, "seed": 42 + row["index"],
                                  "sampling": adapter.params}, ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(remaining)}", flush=True)


if __name__ == "__main__":
    main()
