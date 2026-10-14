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
