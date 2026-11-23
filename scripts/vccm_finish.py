"""Wait for Vccm inference journals, then score and summarize each complete run.

Run from the repository root with the metrics environment and CUDA_VISIBLE_DEVICES set.
Logs and an explicit status are written under res/vccm_logs. Existing scores resume.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNS = [ROOT / 'res/cosyvoice3-latest-vccm/vccm/speaker_only',
        ROOT / 'res/dots-flexivoice-v5-hifitts2-interleave50-ep3-step723-default-instruction/vccm/speaker_only']
BENCH = ROOT / 'raw_data/vccm_v1'
LOGS = ROOT / 'res/vccm_logs'


def keys(path, pattern):
    found = set()
    for p in path.glob(pattern):
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            try:
                found.add(json.loads(line)['key'])
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
    return found


def main():
    import fcntl
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = (LOGS / 'finish.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.chdir(ROOT)
    status = LOGS / 'finish-status.json'
    def update(state, **kw):
        status.write_text(json.dumps(dict(state=state, time=time.time(), **kw), indent=2)+'\n')
        print(state, kw, flush=True)
    def score(run, reference=False):
        name = 'reference' if reference else run.parts[-3]
        update('scoring', run=str(run))
        cmd = [sys.executable, '-u', 'scripts/vccm_score.py', '--run-dir', str(run)]
        if reference:
            cmd += ['--reference']
        with (LOGS / f'{name}-score-full.log').open('a') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run(cmd+['--summary'], stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        expected = {f"{name}|{json.loads(line)['id']}" for name in ('test', 'undomain_style', 'undomain_timbre')
                    for line in (BENCH / f'vccm_{name}.jsonl').read_text().splitlines()}
        score(BENCH / 'reference', True)
        pending = list(RUNS)
        deadline = time.monotonic() + 24*3600
        while pending:
            counts = {}
            for run in pending[:]:
                done = keys(run, 'inference-*.jsonl')
                counts[str(run)] = len(done & expected)
                if expected <= done:
                    score(run)
                    if not expected <= keys(run, 'scores-[0-9]*.jsonl'):
                        raise RuntimeError(f'Incomplete scores: {run}')
                    pending.remove(run)
            if pending:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Inference incomplete after 24 hours: {counts}')
                update('waiting_for_inference', counts=counts, expected=len(expected))
                time.sleep(60)
        update('complete')
    except Exception as exc:
        update('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
