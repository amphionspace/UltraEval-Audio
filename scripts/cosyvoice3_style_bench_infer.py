"""StyleBench emotion subsets with CosyVoice3 (UltraEval's CosyVoiceOffical integration code), one shard.

Runs in envs/cosyvoice_official from the UltraEval root. Writes StyleBench journals/WAVs so
style_bench_score.py scores them like a dots.tts run:
  speaker_only  inference_instruct2(text, "You are a helpful assistant. <instruction><|endofprompt|>", ref)
                (reference gives speaker embedding + flow prompt; the LM sees no reference tokens)
  speaker_icl   inference_zero_shot(text, same prefix + reference transcript, ref)

    CUDA_VISIBLE_DEVICES=4 envs/cosyvoice_official/bin/python scripts/cosyvoice3_style_bench_infer.py \
        --model-path init_model/Fun-CosyVoice3-0.5B-2512 --output res/cosyvoice3/style_bench/speaker_only \
        --mode speaker_only --manifest-dir raw_data/style_bench_v1 --shard 0 --num-shards 4
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/CosyVoice"))
sys.path.insert(1, str(ROOT / "third_party/CosyVoice/third_party/Matcha-TTS"))

PREFIX = "You are a helpful assistant. {instruction}<|endofprompt|>"
SUBSETS = ("emotion_en", "emotion_zh")


def units(manifest_dir):
    out = []
    for subset in SUBSETS:
        for line in (manifest_dir / f"{subset}.jsonl").read_text().splitlines():
            r = json.loads(line)
            for cond in ("easy", "hard"):
                key = f"{subset}|{r['id']}|{cond}"
                out.append(dict(subset=subset, item=r["id"], condition=cond, text=r["text"],
                                instruction=r["instruction"], prompt_audio=r[f"ref_{cond}"],
                                prompt_text=r[f"ref_{cond}_text"], key=key,
                                seed=int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("speaker_only", "speaker_icl"), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.utils.common import set_all_random_seed

    args.output.mkdir(parents=True, exist_ok=True)
    journal = args.output / f"inference-{args.shard:02d}.jsonl"
    done = {json.loads(l)["key"] for l in journal.read_text().splitlines()} if journal.exists() else set()
    todo = [u for k, u in enumerate(units(args.manifest_dir)) if k % args.num_shards == args.shard
            and u["key"] not in done]
    print(f"shard={args.shard} remaining={len(todo)}", flush=True)
    model = AutoModel(model_dir=str(args.model_path))
    with journal.open("a") as log:
        for i, u in enumerate(todo, 1):
            set_all_random_seed(u["seed"])
            prefix = PREFIX.format(instruction=u["instruction"])
            if args.mode == "speaker_only":
                results = model.inference_instruct2(u["text"], prefix, u["prompt_audio"], stream=False)
            else:
                results = model.inference_zero_shot(u["text"], prefix + u["prompt_text"], u["prompt_audio"],
                                                    stream=False)
            speech = torch.concat([r["tts_speech"] for r in results], dim=1)
            path = args.output / u["subset"] / f"{u['item']}-{u['condition']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(path), speech, model.sample_rate)
            log.write(json.dumps({**u, "audio": str(path), "sample_rate": model.sample_rate,
                                  "duration": speech.shape[1] / model.sample_rate, "mode": args.mode,
                                  "template": prefix, "model": "Fun-CosyVoice3-0.5B-2512"},
                                 ensure_ascii=False) + "\n")
            log.flush()
            print(f"shard={args.shard} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
