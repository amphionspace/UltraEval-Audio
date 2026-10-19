"""An exited ASR worker must fail scoring so the queue can retry."""
from io import StringIO
from types import SimpleNamespace

import pytest

from audio_evals.models import whisper


@pytest.mark.parametrize('output,expected', [('', None), ('request->hello\n', 'hello'), ('\nrequest->hello\n', 'hello')])
def test_seed_tts_whisper_detects_eof_and_preserves_valid_responses(monkeypatch, output, expected):
    import uuid
    monkeypatch.setattr(uuid, 'uuid4', lambda: 'request')
    model = object.__new__(whisper.SeedTTSWhisperModel)
    model.process = SimpleNamespace(stdin=StringIO(), stdout=StringIO(output), stderr=StringIO())
    def ready(read, write, *_):
        return ([model.process.stdout], [], []) if read else ([], write, [])
    monkeypatch.setattr(whisper.select, 'select', ready)
    if expected is None:
        with pytest.raises(RuntimeError, match='closed stdout'):
            model._inference({'audio': 'unused.wav'})
    else:
        assert model._inference({'audio': 'unused.wav'}) == expected
