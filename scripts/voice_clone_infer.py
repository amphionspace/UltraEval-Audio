"""Resumable inference using official APIs and UltraEval replication parameters."""
import argparse
import fcntl
import hashlib
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


def generate_local_batch(model, batch, params, language_auto):
    """Batch the language model while keeping reference encoding and audio decoding per sample."""
    prompts = [model.create_voice_clone_prompt(
        ref_audio=row["prompt_audio"],
        ref_text=None if params["x_vector_only_mode"] else row["prompt_text"],
        x_vector_only_mode=params["x_vector_only_mode"])[0] for row in batch]
    tokenizer = model.model.speech_tokenizer
    original_decode = tokenizer.decode

    def decode_individually(encoded):
        wavs, sample_rate = [], None
        for item in encoded:
            values, rate = original_decode([item])
            if len(values) != 1 or (sample_rate is not None and sample_rate != rate):
                raise ValueError("Inconsistent per-sample codec decoding")
            wavs.append(values[0])
            sample_rate = rate
        return wavs, sample_rate

    tokenizer.decode = decode_individually
    try:
        return model.generate_voice_clone(
            text=[r["text"] for r in batch],
            language=["Auto" if language_auto else ("English" if r["language"] == "en" else "Chinese") for r in batch],
            voice_clone_prompt=prompts, **params)
    finally:
        tokenizer.decode = original_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Registered model name or local model output label")
    parser.add_argument("--model-path", help="Exported local Qwen3-TTS checkpoint")
    parser.add_argument("--conditioning", choices=["speaker_only", "speaker_icl", "icl_only"])
    parser.add_argument("--non-streaming", action="store_true", help="Use the non-streaming training protocol")
    parser.add_argument("--language-auto", action="store_true", help="Omit explicit language IDs, matching Auto training")
    parser.add_argument("--greedy", action="store_true", help="Disable sampling in both Qwen Talker and Code Predictor")
    parser.add_argument("--manifest-dir", default="raw_data/voice_clone_manifests")
    parser.add_argument("--gpu-memory-gib", type=float, default=0,
                        help="Maximum PyTorch allocator budget; clamp to free memory minus the reserve")
    parser.add_argument("--gpu-reserve-gib", type=float, default=2,
                        help="Free GPU memory to leave outside the allocator budget")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Per-split smoke-test limit; zero is full set")
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                        help="Breeze only: use its official accelerated runtime")
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--max-audio-seconds", type=float, default=0, help="Exclude outputs longer than this duration; zero keeps all outputs")
    args = parser.parse_args()
    if args.gpu_memory_gib < 0 or args.gpu_reserve_gib < 0:
        parser.error("GPU memory budget and reserve must be nonnegative")
    if args.max_audio_seconds < 0 or (args.max_audio_seconds and not args.model_path):
        parser.error("duration cutoff requires a local Qwen checkpoint and a nonnegative duration")
    if Path(args.model).name != args.model or args.model in {".", ".."}:
        parser.error("model must be a directory name, not a path")
    if args.model_path and not args.conditioning:
        parser.error("a local checkpoint requires an explicit conditioning mode")
    if not args.model_path and args.model not in MODEL_IDS:
        parser.error("unknown model; supply --model-path for a local checkpoint")
    if args.greedy and not (args.model_path or args.model.startswith("qwen")):
        parser.error("greedy is supported only for Qwen inference")
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards or args.batch_size < 1 or args.limit < 0:
        parser.error("invalid shard, batch size or limit")
    torch.set_num_threads(2)
    model_dir = (ROOT / args.model_path if args.model_path else ROOT / "init_model" / MODEL_IDS[args.model]).resolve()
    output_dir = ROOT / args.run_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_lock = (output_dir / f".inference-{args.shard:02d}.lock").open("a")
    fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifests = sorted((ROOT / args.manifest_dir).glob("*.jsonl"))
    if not manifests:
        raise ValueError("No input manifests found")
    identity = {"arguments": vars(args).copy(), "model_path": str(model_dir),
                "manifests": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests}}
    if args.model_path:
        identity["export"] = json.loads((model_dir / "export.json").read_text())
        exported_speaker = identity["export"].get("use_speaker_embedding", True)
        model_speaker = json.loads((model_dir / "config.json").read_text())["talker_config"].get("lm_tts_use_speaker_embedding", True)
        if exported_speaker != model_speaker:
            raise ValueError("Export and model config disagree on speaker conditioning")
        if not exported_speaker and args.conditioning != "icl_only":
            raise ValueError("No-speaker checkpoints require --conditioning icl_only")
    if args.model_path and args.batch_size > 1:
        identity["codec_batch_size"] = 1
    receipt = output_dir / f"invocation-{args.shard:02d}.json"
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        previous["arguments"].setdefault("greedy", False)
        # A duration cutoff changes only the excluded tail, not retained short outputs.
        previous_cutoff = previous["arguments"].get("max_audio_seconds", 0)
        if previous_cutoff and (not args.max_audio_seconds or args.max_audio_seconds > previous_cutoff):
            raise ValueError("Cannot relax a cutoff after capped outputs were generated")
        previous["arguments"]["max_audio_seconds"] = args.max_audio_seconds
        # Allocation limits are operational settings, not generation parameters.
        previous["arguments"]["gpu_memory_gib"] = args.gpu_memory_gib
        previous["arguments"]["gpu_reserve_gib"] = args.gpu_reserve_gib
        if previous != identity:
            raise ValueError("Resume settings or input identity changed; use a new run directory")
    if receipt.exists() and previous_cutoff != args.max_audio_seconds:
        history = output_dir / f"invocation-before-duration-filter-{args.shard:02d}.json"
        if not history.exists():
            history.write_bytes(receipt.read_bytes())
    receipt.write_text(json.dumps(identity, indent=2))
    rows = []
    for file in manifests:
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
    remaining = sum((r["dataset"], r["index"]) not in done for r in rows)
    print(f"model={args.model} shard={args.shard} remaining={remaining} CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}", flush=True)
    if not remaining:
        return
    effective_budget_gib = None
    if args.gpu_memory_gib:
        free, total = torch.cuda.mem_get_info(0)
        budget = min(args.gpu_memory_gib * 1024**3, free - args.gpu_reserve_gib * 1024**3)
        if budget < 4 * 1024**3:
            raise RuntimeError(f"Only {free / 1024**3:.2f} GiB free; need at least 4 GiB for inference plus {args.gpu_reserve_gib} GiB reserve")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
        effective_budget_gib = budget / 1024**3
        print(f"Requested GPU budget={args.gpu_memory_gib} GiB; effective={effective_budget_gib:.2f} GiB", flush=True)
    paths_used = {}
    if args.model_path or args.model.startswith("qwen"):
        args.fast = False  # Qwen uses the native SDPA path, not Breeze's fast runtime.
        from qwen_tts import Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map="cuda:0",
                                             dtype=torch.bfloat16, attn_implementation="sdpa")
        params = dict(max_new_tokens=2048, do_sample=not args.greedy, top_k=50, top_p=1.0,
                      temperature=0.9, repetition_penalty=1.05,
                      subtalker_dosample=not args.greedy, subtalker_top_k=50,
                      subtalker_top_p=1.0, subtalker_temperature=0.9)
        if args.max_audio_seconds:
            codec = json.loads((model_dir / "speech_tokenizer/config.json").read_text())
            frame_rate = codec["output_sample_rate"] / codec["decode_upsample_rate"]
            # Qwen returns one fewer codec frame than generation steps. Allow a
            # full frame beyond the threshold so a capped output is excluded,
            # rather than silently scoring a truncated waveform as valid.
            params["max_new_tokens"] = min(2048, int(args.max_audio_seconds * frame_rate) + 3)
        params["x_vector_only_mode"] = (args.conditioning == "speaker_only" if args.conditioning else args.model.endswith("-xvec_only"))
        params["non_streaming_mode"] = args.non_streaming
        icl_only = args.conditioning == "icl_only" if args.model_path else args.model.endswith("-icl_only")
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

        if icl_only:
            # The wrapper stores this optional value, but our no-speaker branch
            # never consumes it. Avoid running the unused speaker encoder.
            model.model.extract_speaker_embedding = lambda audio, sr: None
        model.model.generate_speaker_prompt = speaker_prompt
        model.model.generate_icl_prompt = icl_prompt

        def generate(batch, seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            paths_used["speaker_embedding"] = None
            paths_used["icl_calls"] = 0
            if args.model_path and args.batch_size > 1:
                result = generate_local_batch(model, batch, params, args.language_auto)
            else:
                result = model.generate_voice_clone(
                    text=[r["text"] for r in batch],
                    language=["Auto" if args.language_auto else ("English" if r["language"] == "en" else "Chinese") for r in batch],
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
    settings_path = output_dir / f"settings-{args.shard:02d}.json"
    if settings_path.exists() and args.max_audio_seconds:
        history = output_dir / "settings-before-duration-filter"
        history.mkdir(exist_ok=True)
        if not (history / settings_path.name).exists():
            (history / settings_path.name).write_bytes(settings_path.read_bytes())
    settings_path.write_text(json.dumps({**vars(args), "sampling": params,
        "codec_batch_size": 1 if args.model_path else args.batch_size,
        "icl_only_ablation": args.conditioning == "icl_only" if args.model_path else args.model.endswith("-icl_only"),
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    with journal.open("a", buffering=1) as log, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            # Preserve original batch membership and seed when resuming. If a
            # journal write was interrupted, regenerate the same batch but only
            # write missing rows; never regroup the remaining samples.
            if all((r["dataset"], r["index"]) in done for r in batch):
                continue
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
                    if (row["dataset"], row["index"]) in done:
                        continue
                    if len(wav) == 0 or not np.isfinite(wav).all():
                        raise ValueError("Empty or non-finite audio")
                    path = output_dir / row["dataset"] / f"{row['index']:06d}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, wav, sr, subtype="PCM_16")
                    result = {**row, "model": args.model, "audio": str(path), "status": "ok",
                              "seed": seed, "sample_rate": sr, "duration": len(wav) / sr,
                              "batch_elapsed": elapsed, "batch_size": len(batch),
                              "conditioning_verified": dict(paths_used),
                              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                              "gpu_budget_gib": effective_budget_gib,
                              "generation_max_new_tokens": params["max_new_tokens"],
                              "duration_excluded": bool(args.max_audio_seconds and len(wav) / sr > args.max_audio_seconds)}
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
