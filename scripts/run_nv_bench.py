"""Run NV-Bench for one dots.tts export: parallel generation, NVASR + SIM scoring, then summary.

Writes <run-dir>/nv_bench/ (journals, WAVs, scores, summary.json, results_table.md) and logs/.
Resumable; on ACP: python -m scripts.acp.submit --name ... --log-dir <run-dir>/logs --submit --
  scripts/run_nv_bench.py --model-path init_model/<export> --run-dir res/<run> --gpus 0,1,2,3,4,5,6,7
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_seed_tts import offline_environment, parallel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--score-workers-per-gpu", type=int, default=2)
    parser.add_argument("--inference-shards", type=int, default=24,
                        help="Fixed sample partitions, so resuming with other GPUs keeps the journals")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    run = (ROOT / args.run_dir).resolve()
    output, logs = run / "nv_bench", run / "logs"
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    env = offline_environment()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts'}"
    model = (ROOT / args.model_path).resolve()
    dots, metrics = str(ROOT / "envs/dots/bin/python"), str(ROOT / "envs/metrics/bin/python")
    commands = [(f"nv_bench-infer-{k:02d}",
                 [dots, "-u", "scripts/nv_bench_infer.py", "--model-path", str(model), "--output", str(output),
                  "--shard", str(k), "--num-shards", str(args.inference_shards)], gpus[k % len(gpus)])
                for k in range(args.inference_shards)]
    width = len(gpus) * args.workers_per_gpu
    for start in range(0, len(commands), width):
        # Inference workers need ~8 GiB plus a 2 GiB reserve each.
        parallel([(label, cmd + ["--gpu-memory-gib", "8", "--gpu-reserve-gib", "2"], gpu)
                  for label, cmd, gpu in commands[start:start + width]], env, logs, (20, 6))
    shards = len(gpus) * args.score_workers_per_gpu
    parallel([(f"nv_bench-score-{k:02d}",
               [metrics, "-u", "scripts/nv_bench_score.py", "--run-dir", str(output),
                "--shard", str(k), "--num-shards", str(shards)], gpus[k % len(gpus)])
              for k in range(shards)], env, logs, (12, 4))
    subprocess.run([metrics, "scripts/nv_bench_score.py", "--run-dir", str(output), "--summarize"],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
