"""Checkpoint discovery, durable job ordering, exclusion locks and audited completion."""
import fcntl
import json
import sys

import pytest
from pathlib import Path

from scripts import watch_seed_tts_checkpoints as watch


def checkpoint(root, step, complete=True, recorded_step=None):
    path = root / f'step-{step:08d}'
    (path / 'distributed').mkdir(parents=True)
    (path / 'metadata.json').write_text(json.dumps({'progress': {'step': recorded_step or step}}))
    if complete:
        (path / 'COMPLETE').touch()
    return path


def test_only_new_complete_matching_checkpoints_are_queued(tmp_path):
    checkpoint(tmp_path, 22500)
    checkpoint(tmp_path, 25000)
    pending = checkpoint(tmp_path, 27500, complete=False)
    wrong = checkpoint(tmp_path, 30000, recorded_step=1)
    ready, errors = watch.discover_checkpoints(tmp_path, 22500)
    assert ready == [25000]
    assert list(errors) == [str(wrong)]
    (pending / 'COMPLETE').touch()
    assert watch.discover_checkpoints(tmp_path, 22500)[0] == [25000, 27500]


def test_completion_requires_passing_audit_and_complete_pipeline(tmp_path):
    (tmp_path / 'full').mkdir()
    (tmp_path / 'pipeline_status.json').write_text('{"state":"complete"}')
    assert not watch.completed(tmp_path)
    (tmp_path / 'full/audit.json').write_text('{"passed":false}')
    assert not watch.completed(tmp_path)
    (tmp_path / 'full/audit.json').write_text('{"passed":true}')
    assert watch.completed(tmp_path)
    (tmp_path / 'pipeline_status.json').write_text('{')
    assert not watch.completed(tmp_path)


def test_retry_delay_does_not_lose_later_checkpoints():
    jobs = {'25000': {'state': 'retry_pending', 'retry_at': 200},
            '27500': {'state': 'queued'}, '22500': {'state': 'complete'}}
    jobs = watch.migrate_jobs(jobs)
    assert watch.next_job(jobs, 100) == '27500:default'
    assert watch.next_job(jobs, 201) == '25000:default'
    jobs['25000:default']['state'] = jobs['27500:default']['state'] = 'complete'
    assert watch.next_job(jobs, 201) is None


def test_existing_pipeline_lock_defers_new_evaluation(tmp_path):
    with (tmp_path / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert watch.busy(tmp_path)
    assert not watch.busy(tmp_path)


@pytest.mark.parametrize("variant", ["default", "mixed"])
def test_job_reuses_export_and_launches_sampling_variant(tmp_path, monkeypatch, variant):
    config = {'checkpoint_dir': str(tmp_path / 'checkpoints'), 'model_root': str(tmp_path / 'models'),
              'model_prefix': 'emilia', 'run_root': str(tmp_path / 'runs'), 'run_prefix': 'emilia',
              'gpus': [0, 1, 6, 7]}
    model, run = watch.job_paths(config, 25000, variant)
    model.mkdir(parents=True)
    (model / 'model.safetensors').write_bytes(b'weights')
    (model / 'export.json').write_text(json.dumps({'step': 25000,
        'checkpoint': str(Path(config['checkpoint_dir']) / 'step-00025000'),
        'weights_sha256': watch.digest(model / 'model.safetensors')}))
    monkeypatch.setattr(watch, 'resource_shortage', lambda _: {})
    calls = []
    def execute(command, *_args, **_kwargs):
        calls.append(command)
        (run / 'pipeline_status.json').write_text('{"state":"complete"}')
        (run / 'full').mkdir()
        (run / 'full/audit.json').write_text('{"passed":true}')
    monkeypatch.setattr(watch, 'run_command', execute)
    watch.run_job(config, 25000, variant)
    assert len(calls) == 1
    command = calls[0]
    assert 'scripts.run_seed_tts' in command
    assert '--greedy' not in command
    assert ('--code-predictor-greedy' in command) == (variant == 'mixed')
    assert ('--talker-temperature' in command) == (variant == 'mixed')
    if variant == 'mixed':
        assert command[command.index('--talker-temperature') + 1] == '0.3'
        assert command[command.index('--talker-repetition-penalty') + 1] == '2'
    assert command[command.index('--modes') + 1] == 'speaker_icl'
    assert command[command.index('--inference-shards') + 1] == '32'
    watch.run_job(config, 25000, variant)
    assert len(calls) == 1


def test_migration_preserves_finished_default_and_queues_mixed_once():
    jobs = watch.migrate_jobs({'25000': {'state': 'complete', 'attempts': 1}})
    watch.register_jobs(jobs, [25000, 27500], ['default', 'mixed'], 100)
    watch.register_jobs(jobs, [25000, 27500], ['default', 'mixed'], 101)
    assert len(jobs) == 4
    assert jobs['25000:default']['state'] == 'complete'
    assert watch.next_job(jobs, 102) == '25000:mixed'
    jobs['25000:mixed']['state'] = 'complete'
    assert watch.next_job(jobs, 102) == '27500:default'


def test_main_launches_one_variant_and_persists_both_jobs(tmp_path, monkeypatch):
    root = tmp_path / 'checkpoints'
    checkpoint(root, 25000)
    config = {'checkpoint_dir': str(root), 'after_step': 22500, 'interval_seconds': 60,
              'model_root': str(tmp_path / 'models'), 'model_prefix': 'emilia',
              'run_root': str(tmp_path / 'runs'), 'run_prefix': 'emilia',
              'watch_dir': str(tmp_path / 'watch'), 'wait_for_runs': [], 'variants': ['default', 'mixed']}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['watch', '--config', str(path)])
    monkeypatch.setattr(watch, 'resource_shortage', lambda _: {})
    monkeypatch.setattr(watch.signal, 'signal', lambda *_: None)
    calls = []
    class Process:
        pid = 123
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, **_):
            return self.returncode
    def launch(command, **_):
        calls.append(command)
        return Process()
    monkeypatch.setattr(watch.subprocess, 'Popen', launch)
    def finish(_):
        raise SystemExit()
    monkeypatch.setattr(watch.time, 'sleep', finish)
    with pytest.raises(SystemExit):
        watch.main()
    saved = json.loads((tmp_path / 'watch/queue.json').read_text())
    assert saved['active_job'] == '25000:default'
    assert saved['jobs']['25000:default']['attempts'] == 1
    assert saved['jobs']['25000:mixed']['state'] == 'queued'
    assert len(calls) == 1 and calls[0][-2:] == ['--variant', 'default']
