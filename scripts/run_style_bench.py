"""Run StyleBench for one dots.tts export: generation (speaker_only and speaker_icl), scoring, summary.

Writes <run-dir>/style_bench/{speaker_only,speaker_icl}/ (journals, WAVs, scores), summary.json and
results.md, logs in <run-dir>/logs/. The real recordings are scored once into
raw_data/style_bench/reference/ and shared by every run. Resumable.

  python scripts/run_style_bench.py --model-path init_model/<export> --run-dir res/<run> --gpus 0,1,2,3,4
  python scripts/run_style_bench.py --model-path <release> --run-dir res/<run> --plain   # no instruction
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_seed_tts import offline_environment, parallel  # noqa: E402

MODES = ("speaker_only", "speaker_icl")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--score-workers-per-gpu", type=int, default=2)
    parser.add_argument("--plain", action="store_true", help="Baseline without instruction (runtime tts template)")
    parser.add_argument("--speaker-scale", type=float, default=1.0)
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "raw_data/style_bench_v1")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    run = (ROOT / args.run_dir).resolve()
    output, logs = run / "style_bench", run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = offline_environment()
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'scripts'}"
    env["STYLE_BENCH_DIR"] = str(args.bench_dir.resolve())
    model = (ROOT / args.model_path).resolve()
    dots, metrics = str(ROOT / "envs/dots/bin/python"), str(ROOT / "envs/metrics/bin/python")
    shards = len(gpus) * args.workers_per_gpu
    commands = [(f"style_bench-infer-{mode}-{k:02d}",
                 [dots, "-u", "scripts/style_bench_infer.py", "--model-path", str(model),
                  "--output", str(output / mode), "--mode", mode, "--shard", str(k), "--num-shards", str(shards),
                  "--gpu-memory-gib", "8", "--gpu-reserve-gib", "2", "--speaker-scale", str(args.speaker_scale),
                  *(["--plain"] if args.plain else [])],
                 gpus[(k + i * shards) % len(gpus)])
                for i, mode in enumerate(MODES) for k in range(shards)]
    width = len(gpus) * args.workers_per_gpu
    for start in range(0, len(commands), width):
        parallel(commands[start:start + width], env, logs, (20, 6))
    score_shards = len(gpus) * args.score_workers_per_gpu
    jobs = [(f"style_bench-score-{mode}-{k:02d}",
             [metrics, "-u", "scripts/style_bench_score.py", "--run-dir", str(output / mode),
              "--shard", str(k), "--num-shards", str(score_shards)], gpus[k % len(gpus)])
            for mode in MODES for k in range(score_shards)]
    reference = args.bench_dir.resolve() / "reference"
    reference.mkdir(exist_ok=True)
    jobs += [(f"style_bench-score-reference-{k:02d}",
              [metrics, "-u", "scripts/style_bench_score.py", "--run-dir", str(reference), "--reference",
               "--shard", str(k), "--num-shards", str(score_shards)], gpus[k % len(gpus)])
             for k in range(score_shards)]
    for start in range(0, len(jobs), score_shards):
        parallel(jobs[start:start + score_shards], env, logs, (12, 4))
    subprocess.run([metrics, "scripts/style_bench_summary.py", "--run-dir", str(output)],
                   cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
