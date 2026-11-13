"""Generate StyleBench audio for one shard with a dots.tts export.

Units (subset, item, condition):
  emotion_*   easy / hard        neutral body; neutral or conflicting reference of the speaker
  paraling_*  style / default    the style instruction, and the default instruction once per pair
  free_*      instr / default    the row's own instruction, and the default instruction
--mode speaker_only conditions on the reference x-vector; speaker_icl also prefills the reference
audio and transcript. --plain is the baseline for exports without instruction training: the
runtime "tts" template and the body text only, for every condition.

Resumable per shard (journal + flock), like nv_bench_infer.py.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("STYLE_BENCH_DIR", ROOT / "raw_data/style_bench"))
SUBSETS = ("emotion_en", "emotion_zh", "paraling_en", "paraling_zh",
           "free_zh", "free_en", "free_ja", "free_ko")
DEFAULT_INSTRUCTION = "Speak the following text."
# LM-TTS-Training instruction layout (lm_tts/data/flexivoice.py): full prefix, then the interleave body.
INTERLEAVE = "[流式语音合成]Instruction: {instruction}\nText:\n{{interleave}}"


def units(manifest_dir):
    out = []
    for subset in SUBSETS:
        rows = [json.loads(line) for line in (manifest_dir / f"{subset}.jsonl").read_text().splitlines()]
        pairs = set()
        for r in rows:
            if subset.startswith("emotion"):
                for cond in ("easy", "hard"):
                    out.append(dict(subset=subset, item=r["id"], condition=cond, text=r["text"],
                                    instruction=r["instruction"], prompt_audio=r[f"ref_{cond}"],
                                    prompt_text=r[f"ref_{cond}_text"]))
            elif subset.startswith("paraling"):
                out.append(dict(subset=subset, item=r["id"], condition="style", text=r["text"],
                                instruction=r["instruction"], prompt_audio=r["ref"], prompt_text=r["ref_text"]))
                if r["pair"] not in pairs:
                    pairs.add(r["pair"])
                    out.append(dict(subset=subset, item=r["pair"], condition="default", text=r["text"],
                                    instruction=DEFAULT_INSTRUCTION, prompt_audio=r["ref"],
                                    prompt_text=r["ref_text"]))
            else:
                for cond, instruction in (("instr", r["instruction"]), ("default", DEFAULT_INSTRUCTION)):
                    out.append(dict(subset=subset, item=r["id"], condition=cond, text=r["text"],
                                    instruction=instruction, prompt_audio=r["ref"], prompt_text=r["ref_text"]))
    for u in out:
        u["key"] = f"{u['subset']}|{u['item']}|{u['condition']}"
        u["seed"] = int(hashlib.sha256(u["key"].encode()).hexdigest()[:8], 16) % 2**31
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("speaker_only", "speaker_icl"), required=True)
    parser.add_argument("--plain", action="store_true", help="No instruction, runtime tts template")
    parser.add_argument("--manifest-dir", type=Path, default=BENCH)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    # Training feeds the unscaled CAM++ x-vector; the runtime default 1.5 amplifies it.
    parser.add_argument("--speaker-scale", type=float, default=1.0)
    parser.add_argument("--gpu-memory-gib", type=float, default=0)
    parser.add_argument("--gpu-reserve-gib", type=float, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / f".infer-{args.shard:02d}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    todo = [u for k, u in enumerate(units(args.manifest_dir)) if k % args.num_shards == args.shard]
    journal = args.output / f"inference-{args.shard:02d}.jsonl"
    done = {json.loads(line)["key"] for line in journal.read_text().splitlines()} if journal.exists() else set()
    todo = [u for u in todo if u["key"] not in done]
    print(f"shard={args.shard} remaining={len(todo)}", flush=True)
    if not todo:
        return
    if args.gpu_memory_gib:
        free, total = torch.cuda.mem_get_info(0)
        budget = min(args.gpu_memory_gib * 1024**3, free - args.gpu_reserve_gib * 1024**3)
        if budget < 4 * 1024**3:
            raise RuntimeError(f"Only {free / 1024**3:.1f} GiB free; need 4 GiB plus the reserve")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
    from dots_tts import runtime as dots_runtime

    runtime = dots_runtime.DotsTtsRuntime.from_pretrained(str(args.model_path), precision="bfloat16")
    icl = args.mode == "speaker_icl"
    templates = {}
    with journal.open("a", buffering=1) as log, torch.inference_mode():
        for i, u in enumerate(todo):
            if args.plain:
                name, template = "tts", None
            else:
                template = INTERLEAVE.format(instruction=u["instruction"])
                name = templates.setdefault(template, f"style_bench_{len(templates)}")
                dots_runtime.RUNTIME_TEMPLATE_BY_NAME[name] = template
            torch.manual_seed(u["seed"])
            torch.cuda.manual_seed_all(u["seed"])
            result = runtime.generate(
                text=u["text"], prompt_audio_path=u["prompt_audio"],
                prompt_text=u["prompt_text"] if icl else None, template_name=name,
                num_steps=args.num_steps, guidance_scale=args.guidance_scale,
                speaker_scale=args.speaker_scale)
            wav = result["audio"].float().cpu().squeeze().numpy()
            rate = result["sample_rate"]
            if len(wav) == 0 or not np.isfinite(wav).all():
                raise ValueError(f"Empty or non-finite audio: {u['key']}")
            path = args.output / u["subset"] / f"{u['item']}-{u['condition']}.wav"
            path.parent.mkdir(exist_ok=True)
            sf.write(path, wav, rate, subtype="FLOAT")  # no clipping of loud outputs
            log.write(json.dumps({**u, "audio": str(path), "sample_rate": rate, "duration": len(wav) / rate,
                                  "mode": args.mode, "plain": args.plain, "template": template,
                                  "num_steps": args.num_steps, "guidance_scale": args.guidance_scale,
                                  "speaker_scale": args.speaker_scale, "peak": float(np.abs(wav).max())},
                                 ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
