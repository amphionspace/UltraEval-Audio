"""Evaluate one checkpoint's watcher variants back to back on a dedicated node, at full width.

Uses the watcher's run_job, so run directories, exports and completion checks match the dev-host
watcher; only the GPU set and per-GPU concurrency are raised for an idle 8-GPU node.
Stop the dev-host watcher for the same queue first: flock is not reliable across hosts.
"""
import argparse
import json
from pathlib import Path

from scripts.watch_seed_tts_checkpoints import completed, job_paths, run_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--workers-per-gpu', type=int, default=6)
    parser.add_argument('--score-workers-per-gpu', type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(gpus=list(map(int, args.gpus.split(','))), workers_per_gpu=args.workers_per_gpu,
                  score_workers_per_gpu=args.score_workers_per_gpu)
    failed = []
    for variant in args.variants:
        print(f'start {args.step}:{variant}', flush=True)
        try:
            run_job(config, args.step, variant)
        except Exception as exc:  # Keep going so one variant cannot block the rest.
            print(f'failed {args.step}:{variant}: {exc}', flush=True)
        _, run = job_paths(config, args.step, variant)
        if not completed(run):
            failed.append(variant)
        print(f'done {args.step}:{variant} complete={completed(run)}', flush=True)
    if failed:
        raise SystemExit(f'Incomplete variants: {failed}')


if __name__ == '__main__':
    main()
