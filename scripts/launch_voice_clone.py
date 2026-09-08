"""Run and monitor all authorized voice-clone evaluations across two GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen3-tts-0.6b-base", "qwen3-tts-1.7b-base", "breeze-tts-2",
          "qwen3-tts-0.6b-base-xvec_only", "qwen3-tts-1.7b-base-xvec_only",
          "qwen3-tts-0.6b-base-icl_only", "qwen3-tts-1.7b-base-icl_only"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--infer-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "config": vars(args), "stages": []}
    state_path = run_dir / "pipeline_status.json"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT), "PATH": str(ROOT / ".tools/bin") + ":" + env["PATH"],
                "HF_HOME": str(ROOT / ".cache/huggingface"), "TORCH_HOME": str(ROOT / ".cache/torch"),
                "MODELSCOPE_CACHE": str(ROOT / ".cache/modelscope"), "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})

    def save():
        state_path.write_text(json.dumps(state, indent=2))

    def stage(name, commands):
        record = {"name": name, "status": "running", "started": time.time(), "processes": []}
        state["stages"].append(record)
        running, handles = [], []
        try:
            for i, (command, gpu) in enumerate(commands):
                path = ROOT / "log" / f"voice_clone_{name}_{i}.log"
                handle = path.open("a")
                handles.append(handle)
                process = subprocess.Popen(command, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                           stdout=handle, stderr=subprocess.STDOUT)
                running.append(process)
                record["processes"].append({"pid": process.pid, "gpu": gpu, "log": str(path), "command": command})
            save()
            while True:
                codes = [p.poll() for p in running]
                record["returncodes"] = codes
                save()
                failed = [c for c in codes if c not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Stage {name} failed: returncodes={codes}")
                if all(c == 0 for c in codes):
                    break
                time.sleep(15)
            record["status"] = "complete"
            record["seconds"] = time.time() - record["started"]
            print(f"Stage {name} complete in {record['seconds']:.0f}s", flush=True)
        finally:
            for p in running:
                if p.poll() is None:
                    p.terminate()
            for p in running:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
            for handle in handles:
                handle.close()
            save()

    try:
        for model in MODELS:
            commands = []
            for shard in range(args.workers):
                commands.append(([str(ROOT / "envs/tts/bin/python"), "-u", "scripts/voice_clone_infer.py",
                                  "--model", model, "--shard", str(shard), "--num-shards", str(args.workers),
                                  "--batch-size", str(args.batch_size), "--run-dir", args.run_dir], shard % 2))
            stage(model, commands)
        if not args.infer_only:
            for group in ("seed", "cv3"):
                stage("score-" + group, [([str(ROOT / "envs/runner/bin/python"), "-u", "scripts/voice_clone_score.py",
                                          "--group", group, "--language", lang, "--run-dir", args.run_dir], gpu)
                                        for gpu, lang in enumerate(("en", "zh"))])
        subprocess.run([str(ROOT / "envs/runner/bin/python"), "scripts/summarize_voice_clone.py", "--run-dir", args.run_dir], check=True)
        state["status"] = "inference_complete" if args.infer_only else "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
