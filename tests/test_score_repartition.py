"""Resume scoring after changing partition count without repeating saved scores."""
import json
import sys
from types import SimpleNamespace

from scripts import voice_clone_score


def test_resume_reads_successes_from_other_shards(tmp_path, monkeypatch):
    model = tmp_path / 'run' / 'speaker_icl'
    model.mkdir(parents=True)
    rows = [{'dataset': 'seed_tts_eval_en', 'index': i, 'language': 'en',
             'status': 'ok', 'duration': 2, 'audio': str(i), 'prompt_audio': 'ref', 'text': 'hello'}
            for i in (0, 16, 32)]
    (model / 'inference-00.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    # Under the old partition, index 16 was saved by a different worker.
    (model / 'scores-seed-en-01.jsonl').write_text(json.dumps({
        'dataset': 'seed_tts_eval_en', 'index': 16, 'status': 'ok'}) + '\n')
    (model / 'scores-seed-en-00.jsonl').write_text(json.dumps({
        'dataset': 'seed_tts_eval_en', 'index': 0, 'status': 'error'}) + '\n')
    called = []
    def evaluate(audio, *_args, **_kwargs):
        called.append(audio)
        return {'wer%': 0, 'simo': 0.5}
    monkeypatch.setattr(voice_clone_score, 'ROOT', tmp_path)
    monkeypatch.setattr(voice_clone_score, 'configure', lambda: SimpleNamespace(get_evaluator=lambda _: evaluate))
    monkeypatch.setattr(sys, 'argv', ['score', '--group', 'seed', '--language', 'en',
        '--run-dir', 'run', '--workers', '16', '--shard', '0', '--models', 'speaker_icl'])
    voice_clone_score.main()
    assert called == ['0', '32']
    records = [json.loads(line) for p in model.glob('scores-*.jsonl') for line in p.read_text().splitlines()]
    assert sorted(r['index'] for r in records if r['status'] == 'ok') == [0, 16, 32]


def test_score_journal_preserves_previous_records_if_replace_fails(tmp_path, monkeypatch):
    import pytest
    target = tmp_path / 'scores.jsonl'
    previous = json.dumps({'index': 1, 'status': 'ok'}) + '\n'
    target.write_text(previous)
    records = [previous]
    def fail_replace(*_):
        raise OSError('simulated interrupted publication')
    monkeypatch.setattr(voice_clone_score.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='interrupted publication'):
        voice_clone_score.save_score(target, records, {'index': 2, 'status': 'ok'})
    assert target.read_text() == previous
    assert records == [previous]
    assert list(tmp_path.iterdir()) == [target]


def test_score_journal_publishes_complete_records(tmp_path):
    target = tmp_path / 'scores.jsonl'
    records = []
    for index in range(3):
        voice_clone_score.save_score(target, records, {'index': index, 'status': 'ok'})
    assert target.read_text().endswith('\n')
    assert [json.loads(line)['index'] for line in target.read_text().splitlines()] == [0, 1, 2]
