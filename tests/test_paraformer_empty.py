"""Empty ASR hypotheses are scored as deletions; model failures still propagate."""

from unittest.mock import Mock

import pytest

from audio_evals.evaluator.seed_tts_eval_asr_wer import process_one
from audio_evals.lib.paraformer_ms.main import transcribe


def test_no_hypothesis_is_empty_transcript_and_full_deletion_error():
    model = Mock()
    model.generate.return_value = []
    text = transcribe(model, "short.wav")
    assert text == ""
    assert process_one(text, "测试语音。", "zh") == 1.0
    assert process_one(text, "Test speech.", "en") == 1.0
    model.generate.assert_called_once_with(input="short.wav", batch_size_s=300)


def test_normal_transcript_is_unchanged():
    model = Mock()
    model.generate.return_value = [{"text": "测试语音"}]
    assert transcribe(model, "normal.wav") == "测试语音"


def test_model_failure_is_not_replaced_by_empty_transcript():
    model = Mock()
    model.generate.side_effect = RuntimeError("CUDA allocation failure")
    with pytest.raises(RuntimeError, match="CUDA allocation failure"):
        transcribe(model, "input.wav")
