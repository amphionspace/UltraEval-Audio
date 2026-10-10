"""Evaluate one exported checkpoint with speaker-only, speaker+ICL, or codec/text-only ICL conditioning."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("speaker_only", "speaker_icl", "icl_only")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def offline_environment():
    env = dict(os.environ)
    for name in list(env):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            del env[name]
    env.update({"HF_ENDPOINT": "https://hf-mirror.com", "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_XET": "1",
                "HF_HOME": str(ROOT / ".cache/huggingface"),
                "TORCH_HOME": str(ROOT / ".cache/torch"),
                "NUMBA_CACHE_DIR": str(ROOT / ".cache/numba"),
                "SEED_TTS_PARAFORMER_PATH": str(ROOT / "init_model/JunHowie/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                "AUDIO_EVALS_CUDA_MEMORY_GIB": "8", "AUDIO_EVALS_SIM_CPU_FALLBACK": "1",
                "AUDIO_EVALS_PARAFORMER_MEMORY_GIB": "4",
                "PATH": str(ROOT / ".tools/bin") + os.pathsep + env.get("PATH", "")})
    return env


def parallel(commands, env, log_dir):
    """Stop sibling evaluation workers if any worker fails; logs remain resumable."""
    processes, logs = [], []
    try:
        for label, command, gpu in commands:
            log = (log_dir / f"{label}.log").open("a", buffering=1)
            logs.append(log)
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
            processes.append(subprocess.Popen(command, cwd=ROOT, env=worker_env,
                                              stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f"Evaluation worker failed; inspect {log_dir}")
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"Evaluation worker failed; inspect {log_dir}")
    finally:
        # Isolated scoring models are grandchildren; terminate only our process groups.
        for p in processes:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for log in logs:
            log.close()


def prepare_smoke_manifests(destination):
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for source in sorted((ROOT / "raw_data/seed_tts_manifests").glob("*.jsonl")):
        target = destination / source.name
        target.write_text("\n".join(source.read_text().splitlines()[:2]) + "\n")
        metadata[source.stem] = {"count": 2, "sha256": digest(target)}
    (destination / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated allocated GPU IDs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-memory-gib", type=float, default=8, help="Per-inference-worker PyTorch allocation cap")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true", help="Generate audio now; rerun without this flag to score and audit")
    parser.add_argument("--wait-for-lock", action="store_true", help="Wait for an earlier phase in the same run directory")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=["speaker_only", "speaker_icl"])
    parser.add_argument("--greedy", action="store_true", help="Disable both Talker and Code Predictor sampling")
    parser.add_argument("--max-audio-seconds", type=float, default=0, help="Exclude outputs longer than this duration; zero keeps all outputs")
    args = parser.parse_args()
    if args.max_audio_seconds < 0:
        parser.error("max-audio-seconds must be nonnegative")
    gpus = args.gpus.split(",")
    if not all(g.isdigit() for g in gpus) or len(set(gpus)) != len(gpus) or args.workers_per_gpu < 1 or args.batch_size < 1:
        parser.error("gpus must be distinct numeric IDs; workers-per-gpu must be positive")
    run = (ROOT / args.run_dir).resolve()
    model = (ROOT / args.model_path).resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / ".pipeline.lock").open("a")
    # AFS may return EAGAIN even for blocking flock; retry explicitly when requested.
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not args.wait_for_lock:
                raise
            time.sleep(5)
    report = json.loads((model / "export.json").read_text())
    if report.get("use_speaker_embedding") is False and args.modes != ["icl_only"]:
        raise ValueError("No-speaker checkpoints require --modes icl_only")
    if digest(model / "model.safetensors") != report["weights_sha256"]:
        raise ValueError("Exported weights do not match export.json")
    state = {"checkpoint": report, "model_path": str(model), "gpus": gpus,
             "workers_per_gpu": args.workers_per_gpu, "gpu_memory_gib": args.gpu_memory_gib,
             "greedy": args.greedy, "state": "starting", "batch_size": args.batch_size}
    identity_path = run / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        previous.setdefault("greedy", False)
        previous.setdefault("batch_size", 1)
        # Physical devices may be reassigned; sample partitioning must stay fixed.
        if len(previous["gpus"]) != len(gpus) or {**previous, "gpus": gpus, "gpu_memory_gib": args.gpu_memory_gib} != state:
            raise ValueError("Run identity changed; resume with identical settings or use a new run directory")
    policy_path = run / "duration_policy.json"
    if policy_path.exists():
        previous_cutoff = json.loads(policy_path.read_text())["max_audio_seconds"]
        if previous_cutoff and (not args.max_audio_seconds or args.max_audio_seconds > previous_cutoff):
            raise ValueError("Cannot relax the cutoff of a run containing capped outputs; use a new run directory")
    policy_path.write_text(json.dumps({"max_audio_seconds": args.max_audio_seconds,
        "exclusion": "independent per run; output duration strictly exceeds limit"}, indent=2))
    identity_path.write_text(json.dumps(state, indent=2))
    state["modes"] = args.modes
    state["max_audio_seconds"] = args.max_audio_seconds
    env = offline_environment()
    logs = run / "logs"
    logs.mkdir(exist_ok=True)
    env["AUDIO_EVALS_RUNTIME_LOG_DIR"] = str(logs)
    tts_python = str(ROOT / "envs/tts/bin/python")
    metric_python = str(ROOT / "envs/metrics/bin/python")
    prepare_smoke_manifests(run / "smoke_inputs")
    def status(phase):
        state.update(state=phase, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        (run / "pipeline_status.json").write_text(json.dumps(state, indent=2))
        print(phase, flush=True)
    def stop(*_):
        raise InterruptedError("Evaluation interrupted")
    signal.signal(signal.SIGTERM, stop)
    try:
        for phase in (["smoke"] if args.smoke_only else ["smoke", "full"]):
            output = run / phase
            output.mkdir(exist_ok=True)
            manifests = run / "smoke_inputs" if phase == "smoke" else ROOT / "raw_data/seed_tts_manifests"
            shards = 1 if phase == "smoke" else len(gpus) * args.workers_per_gpu
            for mode in args.modes:
                status(f"{phase}:inference:{mode}")
                commands = []
                for shard in range(shards):
                    cmd = [tts_python, "-u", "scripts/voice_clone_infer.py", "--model", mode,
                           "--model-path", str(model), "--conditioning", mode,
                           "--non-streaming", "--language-auto", "--manifest-dir", str(manifests),
                           "--run-dir", str(output), "--num-shards", str(shards), "--shard", str(shard),
                           "--gpu-memory-gib", str(args.gpu_memory_gib),
                           "--batch-size", str(args.batch_size),
                           "--max-audio-seconds", str(args.max_audio_seconds)]
                    if args.greedy:
                        cmd.append("--greedy")
                    commands.append((f"{phase}-{mode}-{shard:02d}", cmd, gpus[shard % len(gpus)]))
                parallel(commands, env, logs)
            if args.inference_only:
                continue
            status(f"{phase}:scoring")
            commands = []
            for i, lang in enumerate(("en", "zh")):
                language_gpus = (gpus[:max(1, len(gpus)//2)] if i == 0 else gpus[max(1, len(gpus)//2):]) or gpus
                if phase == "smoke":
                    language_gpus = language_gpus[:1]
                for shard, gpu in enumerate(language_gpus):
                    commands.append((f"{phase}-score-{lang}-{shard:02d}",
                        [metric_python, "-u", "scripts/voice_clone_score.py", "--group", "seed",
                         "--language", lang, "--run-dir", str(output), "--workers", str(len(language_gpus)),
                         "--shard", str(shard), "--max-audio-seconds", str(args.max_audio_seconds),
                         "--models", *args.modes], gpu))
            # Each WavLM worker may need >18 GiB for a long generated waveform.
            if len(gpus) == 1:
                for command in commands:
                    parallel([command], env, logs)
            else:
                parallel(commands, env, logs)
            status(f"{phase}:audit")
            common = ["--run-dir", str(output), "--manifest-dir", str(manifests),
                      "--max-audio-seconds", str(args.max_audio_seconds), "--models", *args.modes]
            subprocess.run([metric_python, "scripts/summarize_voice_clone.py", *common], cwd=ROOT, env=env, check=True)
            audit = [metric_python, "scripts/audit_voice_clone.py", *common,
                     "--num-shards", str(shards), "--batch-size", str(args.batch_size), "--output-name", "audit.json"]
            if args.greedy:
                audit.append("--expect-greedy")
            subprocess.run(audit, cwd=ROOT, env=env, check=True)
        status("inference_complete" if args.inference_only else ("smoke_passed" if args.smoke_only else "complete"))
    except BaseException as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        status("failed")
        raise


if __name__ == "__main__":
    main()
