"""Discover completed training checkpoints and evaluate them one at a time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from scripts.run_seed_tts import digest

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def discover_checkpoints(directory, after_step):
    ready, errors = [], {}
    for checkpoint in sorted(directory.glob('step-*')):
        suffix = checkpoint.name.removeprefix('step-')
        if not suffix.isdigit() or int(suffix) <= after_step:
            continue
        if not (checkpoint / 'COMPLETE').is_file() or not (checkpoint / 'metadata.json').is_file():
            continue
        try:
            step = int(suffix)
            metadata = json.loads((checkpoint / 'metadata.json').read_text())
            if metadata['progress']['step'] != step or not (checkpoint / 'distributed').is_dir():
                raise ValueError('Checkpoint step mismatch or missing distributed state')
            ready.append(step)
        except (ValueError, KeyError, OSError) as exc:
            errors[str(checkpoint)] = str(exc)
    return sorted(ready), errors


def completed(run):
    status, audit = run / 'pipeline_status.json', run / 'full/audit.json'
    try:
        return (status.is_file() and audit.is_file()
                and json.loads(status.read_text()).get('state') == 'complete'
                and json.loads(audit.read_text()).get('passed') is True)
    except (OSError, ValueError):
        return False  # A running pipeline may be replacing these status files.


def busy(run):
    path = run / '.pipeline.lock'
    if not path.exists():
        return False
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def job_paths(config, step, variant="default"):
    model = Path(config['model_root']) / f"{config['model_prefix']}-step-{step:08d}"
    suffix = "default-sampling" if variant == "default" else "talkert03-k10-rp2-cpgreedy"
    run = Path(config['run_root']) / f"{config['run_prefix']}-step{step}-seedtts-{suffix}-{config.get('run_suffix', '4gpu')}"
    return model, run


def migrate_jobs(jobs):
    migrated = {}
    for key, job in jobs.items():
        step = int(key) if key.isdigit() else int(key.split(":")[0])
        variant = "default" if key.isdigit() else key.split(":")[1]
        migrated[f"{step}:{variant}"] = {**job, "step": step, "variant": variant}
    return migrated


def register_jobs(jobs, steps, variants, now):
    for step in steps:
        for variant in variants:
            jobs.setdefault(f"{step}:{variant}", {"step": step, "variant": variant,
                "state": "queued", "discovered_at": now, "attempts": 0})


def next_job(jobs, now):
    eligible = [key for key, job in jobs.items()
                if job['state'] != 'complete' and job.get('retry_at', 0) <= now
                and not (job['variant'] == 'mixed'
                         and f"{job['step']}:default" in jobs
                         and jobs[f"{job['step']}:default"]['state'] != 'complete')]
    return min(eligible, key=lambda key: (jobs[key]['step'], jobs[key]['variant'] != 'default')) if eligible else None


def gpu_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    return {int(i): int(m) for i, m in (row.split(',') for row in output.splitlines())}


def required_free_gib(config):
    """Use the same per-wave budgets as the runner for device admission."""
    return max(config['minimum_free_gib'],
               config.get('workers_per_gpu', 4) * 6 + 6,
               config.get('score_workers_per_gpu', 1) * 20 + 6)


def select_gpus(config):
    free = gpu_memory()
    eligible = [gpu for gpu in config.get('gpu_candidates', config['gpus'])
                if free[gpu] >= required_free_gib(config) * 1024]
    count = len(config['gpus'])
    if len(eligible) < config.get('minimum_gpus', count):
        return None
    # Select by current free memory; logical inference partitions stay unchanged.
    return sorted(sorted(eligible, key=lambda gpu: (-free[gpu], gpu))[:count])


def resource_shortage(config):
    free = gpu_memory()
    return {str(gpu): free[gpu] for gpu in config['gpus']
            if free[gpu] < required_free_gib(config) * 1024}


def run_command(command, log, env=None, cwd=ROOT):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        if process.wait() != 0:
            raise RuntimeError(f'Command failed: {command}')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def run_job(config, step, variant="default"):
    model, run = job_paths(config, step, variant)
    checkpoint = Path(config['checkpoint_dir']) / f'step-{step:08d}'
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.job.lock').open('a') as lock, (run / 'job.log').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if completed(run):
            return
        if not (model / 'export.json').is_file():
            staging = model.with_name(model.name + '.exporting')
            if staging.exists():
                staging.rename(staging.with_name(staging.name + '.failed-' + str(time.time_ns())))
            command = [config['training_python'], '-m', 'scripts.export_checkpoint',
                       '--checkpoint', str(checkpoint), '--assembled-model', config['assembled_model'],
                       '--output', str(staging), '--precision', 'fp32', '--copy-tokenizer']
            write_json(run / 'export_command.json', command)
            run_command(command, log, dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4'),
                        cwd=Path(config['training_repo']))
            staging.rename(model)
        report = json.loads((model / 'export.json').read_text())
        if report.get('step') != step or Path(report['checkpoint']).resolve() != checkpoint.resolve():
            raise ValueError('Export does not match requested checkpoint')
        if digest(model / 'model.safetensors') != report['weights_sha256']:
            raise ValueError('Export weights hash mismatch')
        shortage = resource_shortage(config)
        if shortage:
            raise RuntimeError(f'Insufficient free GPU memory (MiB): {shortage}')
        gpus = ','.join(map(str, config['gpus']))
        # Each decoding variant gets its own journals while sharing the verified export.
        command = [str(ROOT / 'envs/tts/bin/python'), '-u', '-m', 'scripts.run_seed_tts',
                   '--model-path', str(model), '--run-dir', str(run), '--modes', 'speaker_icl',
                   '--gpus', gpus, '--workers-per-gpu', str(config.get('workers_per_gpu', 4)), '--inference-shards', '32',
                   '--score-gpus', gpus, '--score-workers-per-gpu', str(config.get('score_workers_per_gpu', 1)), '--batch-size', '1',
                   '--gpu-memory-gib', '6', '--gpu-reserve-gib', '6', '--max-audio-seconds', '30']
        if variant == 'mixed':
            command.extend(['--talker-temperature', '0.3', '--talker-top-p', '1',
                            '--talker-top-k', '10', '--talker-repetition-penalty', '2',
                            '--code-predictor-greedy'])
        write_json(run / 'eval_command.json', command)
        run_command(command, log)
        if not completed(run):
            raise RuntimeError('Evaluation is missing a complete result and passing audit')


def stop(*_):
    raise SystemExit('Checkpoint evaluation watcher stopped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--step', type=int, help='Execute one queued job instead of monitoring')
    parser.add_argument('--variant', choices=['default', 'mixed'], default='default')
    parser.add_argument('--gpus', help='Physical GPU IDs selected for one job')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.gpus:
        config['gpus'] = list(map(int, args.gpus.split(',')))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGHUP, stop)
    if args.step is not None:
        run_job(config, args.step, args.variant)
        return
    watch = Path(config['watch_dir'])
    watch.mkdir(parents=True, exist_ok=True)
    lock = (watch / '.watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = watch / 'queue.json'
    jobs = migrate_jobs(json.loads(state_path.read_text())['jobs']) if state_path.exists() else {}
    # No child belongs to this new watcher; pipeline locks protect external jobs.
    for job in jobs.values():
        if job['state'] == 'running':
            job['state'] = 'queued'
    child, child_log, active = None, None, None
    try:
        while True:
            try:
                now = time.time()
                if child is not None and child.poll() is not None:
                    _, run = job_paths(config, jobs[active]['step'], jobs[active]['variant'])
                    job = jobs[active]
                    if child.returncode == 0 and completed(run):
                        job.update(state='complete', finished_at=now)
                    else:
                        job.update(state='retry_pending', retry_at=now + config['interval_seconds'],
                                   last_exit_code=child.returncode)
                    child_log.close()
                    child, child_log, active = None, None, None
                ready, errors = discover_checkpoints(Path(config['checkpoint_dir']), config['after_step'])
                register_jobs(jobs, ready, config.get('variants', ['default', 'mixed']), now)
                for key, job in jobs.items():
                    _, run = job_paths(config, job['step'], job['variant'])
                    if key != active and completed(run):
                        job['state'] = 'complete'
                status = 'evaluating' if child else 'waiting_checkpoint'
                key = next_job(jobs, now) if child is None else None
                if key is not None:
                    job = jobs[key]
                    step, variant = job['step'], job['variant']
                    _, run = job_paths(config, step, variant)
                    if any(busy(Path(p)) for p in config['wait_for_runs']) or busy(run):
                        status = 'waiting_other_evaluation'
                    elif (selected := select_gpus(config)) is None:
                        status = 'waiting_gpu_memory'
                        errors['gpu_free_mib'] = gpu_memory()
                    else:
                        run.mkdir(parents=True, exist_ok=True)
                        child_log = (run / 'launcher.log').open('a', buffering=1)
                        child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.watch_seed_tts_checkpoints',
                            '--config', str(args.config.resolve()), '--step', str(step), '--variant', variant, '--gpus', ','.join(map(str, selected))], cwd=ROOT,
                            stdout=child_log, stderr=subprocess.STDOUT)
                        active, status = key, 'evaluating'
                        jobs[key].update(state='running', attempts=jobs[key]['attempts'] + 1,
                                               started_at=now, run_dir=str(run), pid=child.pid, selected_gpus=selected)
                record = {'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'state': status, 'active_job': active, 'jobs': jobs, 'errors': errors,
                          'interval_seconds': config['interval_seconds']}
                write_json(state_path, record)
                with (watch / 'checks.jsonl').open('a') as log:
                    log.write(json.dumps(record) + '\n')
            except Exception:
                record = {'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                          'state': 'check_error', 'active_job': active, 'jobs': jobs,
                          'error': traceback.format_exc(), 'interval_seconds': config['interval_seconds']}
                write_json(state_path, record)
                print(record['error'], flush=True)
            time.sleep(config['interval_seconds'])
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child_log is not None:
            child_log.close()
        lock.close()


if __name__ == '__main__':
    main()
