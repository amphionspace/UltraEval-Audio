"""Resumable inference using official APIs and UltraEval replication parameters."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

# Eight Breeze workers must not each launch the default large compiler pool.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_IDS = {"qwen3-tts-0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
             "qwen3-tts-1.7b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
             "qwen3-tts-0.6b-base-xvec_only": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
             "qwen3-tts-1.7b-base-xvec_only": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
             "qwen3-tts-0.6b-base-icl_only": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
             "qwen3-tts-1.7b-base-icl_only": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
             "breeze-tts-2": "BreezeBlue/Breeze-TTS-2"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Per-split smoke-test limit; zero is full set")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                        help="Breeze only: use its official accelerated runtime")
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model_dir = ROOT / "init_model" / MODEL_IDS[args.model]
    output_dir = ROOT / args.run_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted((ROOT / "raw_data/voice_clone_manifests").glob("*.jsonl")):
        subset = [json.loads(line) for line in file.read_text().splitlines()]
        if args.limit:
            subset = subset[:args.limit]
        rows.extend(r for r in subset if r["index"] % args.num_shards == args.shard)
    journal = output_dir / f"inference-{args.shard:02d}.jsonl"
    done = {}
    if journal.exists():
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "ok" and Path(row["audio"]).exists():
                done[(row["dataset"], row["index"])] = row
    rows = [r for r in rows if (r["dataset"], r["index"]) not in done]
    print(f"model={args.model} shard={args.shard} remaining={len(rows)} CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}", flush=True)
    if not rows:
        return
    if args.model.startswith("qwen"):
        args.fast = False  # Qwen uses the native SDPA path, not Breeze's fast runtime.
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map="cuda:0",
                                             dtype=torch.bfloat16, attn_implementation="sdpa")
        params = dict(max_new_tokens=2048, do_sample=True, top_k=50, top_p=1.0,
                      temperature=0.9, repetition_penalty=1.05,
                      subtalker_dosample=True, subtalker_top_k=50,
                      subtalker_top_p=1.0, subtalker_temperature=0.9)
        params["x_vector_only_mode"] = args.model.endswith("-xvec_only")
        icl_only = args.model.endswith("-icl_only")
        paths_used = {"speaker_embedding": False, "icl_calls": 0}
        original_speaker_prompt = model.model.generate_speaker_prompt
        original_icl_prompt = model.model.generate_icl_prompt

        def speaker_prompt(voice_clone_prompt):
            # None uses the existing no-speaker prefix branch. No zero vector or
            # replacement token is inserted; reference-code ICL stays enabled.
            result = None if icl_only else original_speaker_prompt(voice_clone_prompt)
            paths_used["speaker_embedding"] = result is not None
            return result

        def icl_prompt(*positional, **keywords):
            paths_used["icl_calls"] += 1
            return original_icl_prompt(*positional, **keywords)

        model.model.generate_speaker_prompt = speaker_prompt
        model.model.generate_icl_prompt = icl_prompt

        def generate(batch, seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            paths_used["speaker_embedding"] = None
            paths_used["icl_calls"] = 0
            result = model.generate_voice_clone(
                text=[r["text"] for r in batch],
                language=["English" if r["language"] == "en" else "Chinese" for r in batch],
                ref_audio=[r["prompt_audio"] for r in batch],
                ref_text=None if params["x_vector_only_mode"] else [r["prompt_text"] for r in batch],
                **params)
            assert paths_used["speaker_embedding"] == (not icl_only), paths_used
            assert paths_used["icl_calls"] == (0 if params["x_vector_only_mode"] else len(batch)), paths_used
            validation = {**paths_used, "batch_size": len(batch), "model": args.model}
            (output_dir / f"conditioning-verified-{args.shard:02d}.json").write_text(json.dumps(validation, indent=2))
            return result
    else:
        sys.path.insert(0, str(ROOT / "third_party/breeze-tts"))
        from breeze_infer.runtime import load_runtime, set_all_seeds, update_generation_config_for_breeze
        from breeze_infer.templates import get_template, prepare_inputs
        from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
        from models.warmup_profile import load_warmup_profile
        tokenizer, model, codec = load_runtime(model_dir, device="cuda:0", attn_implementation="eager")
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(max_new_tokens=1500, max_seq_len=2048,
                                     fast_all=args.fast, repetition_penalty=1.1)
        runtime = FastBreezeStreamingRuntime(model, codec, config, tokenizer=tokenizer)
        if args.fast:
            from dataclasses import replace
            profile = load_warmup_profile(ROOT / "third_party/breeze-tts/configs/fast.json")
            # The shipped service profile freezes a small set of prompt shapes.
            # Evaluation inputs need additional buckets; use the official lazy
            # capture policy without changing padding, sampling, or conditioning.
            runtime.warmup_from_profile(replace(profile, codec_chunk_frames=runtime.codec_chunk_frames,
                                               freeze_after_warmup=False))
        args.batch_size = 1
        params = {"max_new_tokens": 1500, "max_seq_len": 2048, "cfg_scale": 1.0,
                  "repetition_penalty": 1.1, "fast": args.fast,
                  "warmup_freeze_after_warmup": False if args.fast else None}

        def generate(batch, seed):
            r = batch[0]
            set_all_seeds(seed)
            request = {"id": f"{r['dataset']}-{r['index']}", "text": r["text"],
                       "instruction": "Speak clearly and naturally.", "speaker": "S0",
                       "ref_audio_path": r["prompt_audio"], "ref_text": r["prompt_text"].strip()}
            inputs = prepare_inputs(tokenizer, codec, model, [request], get_template("ref_edit_tata"),
                                    guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None)
            chunks = [chunk.audio for chunk in runtime.iter_audio_chunks(inputs, request_id=request["id"], seed=seed)]
            return [np.concatenate(chunks)], runtime.sample_rate
    (output_dir / f"settings-{args.shard:02d}.json").write_text(json.dumps({**vars(args), "sampling": params,
        "icl_only_ablation": args.model.endswith("-icl_only"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    with journal.open("a", buffering=1) as log, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            seed = 42 + batch[0]["index"]
            try:
                torch.cuda.synchronize()
                started = time.monotonic()
                wavs, sr = generate(batch, seed)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                if len(wavs) != len(batch):
                    raise ValueError("Generation output count differs from input count")
                for row, wav in zip(batch, wavs):
                    if len(wav) == 0 or not np.isfinite(wav).all():
                        raise ValueError("Empty or non-finite audio")
                    path = output_dir / row["dataset"] / f"{row['index']:06d}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, wav, sr, subtype="PCM_16")
                    result = {**row, "model": args.model, "audio": str(path), "status": "ok",
                              "seed": seed, "sample_rate": sr, "duration": len(wav) / sr,
                              "batch_elapsed": elapsed, "batch_size": len(batch)}
                    log.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"{args.model} shard={args.shard} {start+len(batch)}/{len(rows)} batch_seconds={elapsed:.2f}", flush=True)
            except Exception:
                error = traceback.format_exc()
                print(error, flush=True)
                for row in batch:
                    log.write(json.dumps({**row, "status": "error", "error": error}, ensure_ascii=False) + "\n")
                # Infrastructure failures must stop instead of silently failing the complete split.
                raise


if __name__ == "__main__":
    main()
