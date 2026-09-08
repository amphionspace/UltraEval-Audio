"""Score generated audio using the repository's registered evaluators unchanged."""
import argparse
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import sys
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["PATH"] = str(ROOT / ".tools/bin") + os.pathsep + os.environ.get("PATH", "")
os.environ["AUDIO_EVALS_SKIP_ENV_SETUP"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["DNSMOS_NUM_THREADS"] = "2"
os.environ["WAVLM_LARGE_BACKBONE"] = str(ROOT / "init_model/s3prl/converted_ckpts/wavlm_large.pt")


def configure():
    from audio_evals.registry import registry
    names = ["seed-tts-whisper", "cv3-whisper", "wavlm_large", "dnsmos",
             "speech_eres2net_sv_en_voxceleb_16k",
             "speech_paraformer-speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"]
    for name in names:
        registry._model[name]["args"]["env_path"] = str(ROOT / "envs/metrics")
    registry._model["seed-tts-whisper"]["args"]["path"] = str(ROOT / "init_model/openai/whisper-large-v3")
    registry._model["cv3-whisper"]["args"]["path"] = str(ROOT / "init_model/whisper/large-v3.pt")
    registry._model["wavlm_large"]["args"]["path"] = str(ROOT / "init_model/hidoba/wavlm_large_finetune/wavlm_large_finetune.pth")
    name = names[-1]
    registry._model[name]["args"]["path"] = str(ROOT / "init_model/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    registry._model["dnsmos"]["args"].update({
        "model_path": str(ROOT / "init_model/DNS-Challenge/DNSMOS/DNSMOS/sig_bak_ovr.onnx"),
        "p_model_path": str(ROOT / "init_model/DNS-Challenge/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"),
        "p808_model_path": str(ROOT / "init_model/DNS-Challenge/DNSMOS/DNSMOS/model_v8.onnx"),
    })
    return registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "zh"], required=True)
    parser.add_argument("--group", choices=["seed", "cv3"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.workers is None:
        # Measured WavLM peak for a 164-second output: 18.23 GiB, excluding ASR.
        args.workers = 2 if args.group == "seed" else 4
    if args.workers < 1 or (args.shard is not None and not 0 <= args.shard < args.workers):
        parser.error("workers must be positive and shard must be in [0, workers)")
    # Only the orchestrator owns this lock; its child shards run concurrently.
    # A later catch-up phase waits for any earlier phase writing the same files.
    phase_lock = None
    if args.shard is None:
        def stop_phase(*_):
            raise SystemExit(128 + signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop_phase)
        phase_lock = (ROOT / args.run_dir / f".score-{args.group}-{args.language}.lock").open("a")
        print(f"Waiting for scoring phase lock: {args.group}-{args.language}", flush=True)
        fcntl.flock(phase_lock.fileno(), fcntl.LOCK_EX)
        print("Scoring phase lock acquired", flush=True)
    if args.shard is None and args.workers > 1:
        processes = []
        try:
            for shard in range(args.workers):
                command = [sys.executable, "-u", str(Path(__file__).resolve()),
                           "--group", args.group, "--language", args.language,
                           "--run-dir", args.run_dir, "--limit", str(args.limit),
                           "--workers", str(args.workers), "--shard", str(shard)]
                processes.append(subprocess.Popen(command))
            while True:
                codes = [p.poll() for p in processes]
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError(f"Scoring shard failed: {codes}")
                if all(c == 0 for c in codes):
                    return
                time.sleep(10)
        finally:
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            for p in processes:
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
        return
    registry = configure()
    name = (f"seed-tts-eval-vc-{args.language}" if args.group == "seed" else f"cv3-zero-shot-{args.language}")
    evaluator = registry.get_evaluator(name)
    for model_dir in sorted((ROOT / args.run_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        samples = {}
        for file in sorted(model_dir.glob("inference-*.jsonl")):
            for line in file.read_text().splitlines():
                row = json.loads(line)
                if (row.get("status") == "ok" and row["dataset"].startswith(args.group)
                        and row["language"] == args.language
                        and (args.shard is None or row["index"] % args.workers == args.shard)):
                    samples[(row["dataset"], row["index"])] = row
        suffix = "" if args.shard is None else f"-{args.shard:02d}"
        target = model_dir / f"scores-{args.group}-{args.language}{suffix}.jsonl"
        done = set()
        previous_files = {target, model_dir / f"scores-{args.group}-{args.language}.jsonl"}
        for previous in previous_files:
            if not previous.exists():
                continue
            for line in previous.read_text().splitlines():
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add((r["dataset"], r["index"]))
        remaining = [r for key, r in sorted(samples.items()) if key not in done]
        if args.limit:
            remaining = remaining[:args.limit]
        with target.open("a", buffering=1) as output:
            for i, row in enumerate(remaining):
                started = time.monotonic()
                try:
                    score = evaluator(row["audio"], row["prompt_audio"], text=row["text"], WavPath=row["prompt_audio"])
                    required = ["wer%" if args.language == "en" else "cer%"]
                    required += ["simo"] if args.group == "seed" else ["speaker_sim", "OVRL", "P808_MOS"]
                    for metric in required:
                        if metric not in score or not math.isfinite(float(score[metric])):
                            raise ValueError(f"Missing or non-finite metric {metric}: {score}")
                    result = {"model": model_dir.name, "dataset": row["dataset"], "index": row["index"],
                              "status": "ok", "score": score, "seconds": time.monotonic() - started}
                    if args.group == "cv3":
                        result["dnsmos_num_threads"] = int(os.environ["DNSMOS_NUM_THREADS"])
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")
                    print(f"{model_dir.name} {row['dataset']} {i+1}/{len(remaining)} {score}", flush=True)
                except Exception:
                    output.write(json.dumps({"dataset": row["dataset"], "index": row["index"],
                                             "status": "error", "error": traceback.format_exc()}, ensure_ascii=False) + "\n")
                    raise


if __name__ == "__main__":
    main()
