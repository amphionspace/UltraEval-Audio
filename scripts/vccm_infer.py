"""Generate the VccmDataset (ControlSpeech) test sets with a dots.tts export, one shard.

Each row: the speaker prompt (another utterance of the same speaker, raw_data/vccm_v1) conditions
the voice; the row's style_prompt is the instruction, in the same layout as StyleBench
(scripts/style_bench_infer.py). Outputs <output>/<set>/<id>.wav and a resumable journal.

    envs/dots/bin/python scripts/vccm_infer.py --model-path init_model/<export> \
        --output res/<run>/vccm/speaker_only --shard 0 --num-shards 8
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
INTERLEAVE = "[流式语音合成]Instruction: {instruction}\nText:\n{{interleave}}"
SETS = ("test", "undomain_style", "undomain_timbre")


def units(bench):
    out = []
    for name in SETS:
        for line in (bench / f"vccm_{name}.jsonl").read_text().splitlines():
            r = json.loads(line)
            key = f"{name}|{r['id']}"
            out.append({**r, "key": key, "seed": int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "raw_data/vccm_v1")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--speaker-scale", type=float, default=1.0)
    parser.add_argument("--gpu-memory-gib", type=float, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f".infer-{args.shard:02d}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = args.output / f"inference-{args.shard:02d}.jsonl"
    done = {json.loads(l)["key"] for l in journal.read_text().splitlines()} if journal.exists() else set()
    todo = [u for k, u in enumerate(units(args.bench_dir)) if k % args.num_shards == args.shard and u["key"] not in done]
    print(f"shard={args.shard} remaining={len(todo)}", flush=True)
    if not todo:
        return
    if args.gpu_memory_gib:
        _, total = torch.cuda.mem_get_info(0)
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_gib * 1024**3 / total, 0)
    from dots_tts import runtime as dots_runtime

    runtime = dots_runtime.DotsTtsRuntime.from_pretrained(str(args.model_path), precision="bfloat16")
    templates = {}
    with journal.open("a", buffering=1) as log, torch.inference_mode():
        for i, u in enumerate(todo):
            template = INTERLEAVE.format(instruction=u["instruction"])
            name = templates.setdefault(template, f"vccm_{len(templates)}")
            dots_runtime.RUNTIME_TEMPLATE_BY_NAME[name] = template
            torch.manual_seed(u["seed"])
            torch.cuda.manual_seed_all(u["seed"])
            result = runtime.generate(text=u["text"], prompt_audio_path=u["ref"], prompt_text=None,
                                      template_name=name, num_steps=args.num_steps,
                                      guidance_scale=args.guidance_scale, speaker_scale=args.speaker_scale)
            wav = result["audio"].float().cpu().squeeze().numpy()
            rate = result["sample_rate"]
            if len(wav) == 0 or not np.isfinite(wav).all():
                raise ValueError(f"Empty or non-finite audio: {u['key']}")
            path = args.output / u["set"] / f"{u['id']}.wav"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, wav, rate, subtype="FLOAT")
            log.write(json.dumps({**u, "audio": str(path), "sample_rate": rate, "duration": len(wav) / rate,
                                  "template": template, "num_steps": args.num_steps,
                                  "guidance_scale": args.guidance_scale, "speaker_scale": args.speaker_scale},
                                 ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
