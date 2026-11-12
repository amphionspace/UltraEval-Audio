"""Run several benchmarks for one dots.tts export back to back on one node.

Each benchmark keeps its own resumable run directory under --run-dir: nv_bench/, seed_tts/, cv3/.
A failed benchmark does not stop the next one; the job fails at the end if any did.
On ACP: python -m scripts.acp.submit --name ... --log-dir <run-dir>/logs --submit --
  scripts/run_dots_suite.py --model-path init_model/<export> --run-dir res/<run>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ("nv_bench", "seed_tts", "cv3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=list(BENCHMARKS))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--template", help="dots.tts runtime template for every benchmark (default: tts)")
    parser.add_argument("--instruction",
                        help="Style instruction for instruction-trained exports, e.g. the default "
                             "'Speak the following text.'; uses the interleave layout")
    args = parser.parse_args()
    # Inference workers inherit these (scripts/local_dots.py); journals record the values used.
    if args.template:
        os.environ["DOTS_TEMPLATE"] = args.template
    if args.instruction:
        os.environ["DOTS_INSTRUCTION"] = args.instruction
    common = ["--model-path", args.model_path, "--gpus", args.gpus, "--workers-per-gpu", str(args.workers_per_gpu)]
    commands = {
        "nv_bench": ["scripts/run_nv_bench.py", *common, "--run-dir", args.run_dir, "--inference-shards", "32"],
        **{name: ["scripts/run_seed_tts.py", *common, "--run-dir", f"{args.run_dir}/{name}",
                  "--benchmark", {"seed_tts": "seed", "cv3": "cv3"}[name], "--score-workers-per-gpu", "2",
                  "--modes", "speaker_only", "speaker_icl", "--max-audio-seconds", "30"]
           for name in ("seed_tts", "cv3")},
    }
    failed = []
    for name in args.benchmarks:
        print(f"start {name}", flush=True)
        code = subprocess.run([sys.executable, "-u", *commands[name]], cwd=ROOT).returncode
        print(f"done {name} exit={code}", flush=True)
        if code:
            failed.append(name)
    if failed:
        raise SystemExit(f"Failed benchmarks: {failed}")


if __name__ == "__main__":
    main()
