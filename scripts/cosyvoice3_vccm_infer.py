"""VccmDataset (ControlSpeech) test sets with CosyVoice3, one shard (runs in envs/cosyvoice_official).

inference_instruct2(text, "You are a helpful assistant. <style_prompt><|endofprompt|>", speaker prompt):
the reference gives the voice, the style prompt the style, as in cosyvoice3_style_bench_infer.py.
Writes the same journal / WAV layout as scripts/vccm_infer.py.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/CosyVoice"))
sys.path.insert(1, str(ROOT / "third_party/CosyVoice/third_party/Matcha-TTS"))
sys.path.insert(2, str(ROOT / "scripts"))
from vccm_infer import units  # noqa: E402

PREFIX = "You are a helpful assistant. {instruction}<|endofprompt|>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "raw_data/vccm_v1")
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
    todo = [u for k, u in enumerate(units(args.bench_dir)) if k % args.num_shards == args.shard and u["key"] not in done]
    print(f"shard={args.shard} remaining={len(todo)}", flush=True)
    if not todo:
        return
    model = AutoModel(model_dir=str(args.model_path))
    with journal.open("a") as log:
        for i, u in enumerate(todo, 1):
            set_all_random_seed(u["seed"])
            prefix = PREFIX.format(instruction=u["instruction"])
            try:
                results = list(model.inference_instruct2(u["text"], prefix, u["ref"], stream=False))
                speech = torch.concat([r["tts_speech"] for r in results], dim=1)
            except RuntimeError as exc:
                if "out of memory" in str(exc):
                    raise
                # e.g. a reference too short for the speech-token encoder: record and move on.
                log.write(json.dumps({**u, "audio": None, "error": str(exc)[:300], "template": prefix,
                                      "model": "Fun-CosyVoice3-0.5B-2512"}, ensure_ascii=False) + "\n")
                log.flush()
                print(f"shard={args.shard} {i}/{len(todo)} FAILED {u['key']}: {str(exc)[:120]}", flush=True)
                continue
            path = args.output / u["set"] / f"{u['id']}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(path), speech, model.sample_rate)
            log.write(json.dumps({**u, "audio": str(path), "sample_rate": model.sample_rate,
                                  "duration": speech.shape[1] / model.sample_rate, "template": prefix,
                                  "model": "Fun-CosyVoice3-0.5B-2512"}, ensure_ascii=False) + "\n")
            log.flush()
            print(f"shard={args.shard} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
