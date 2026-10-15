"""Talker sampling and Code Predictor greedy must remain independently selectable."""
from types import SimpleNamespace

from scripts.voice_clone_infer import qwen_generation_params


def test_mixed_decode_settings_and_unchanged_defaults():
    args = SimpleNamespace(greedy=False, code_predictor_greedy=False, talker_temperature=None,
        talker_top_p=None, talker_top_k=None, talker_repetition_penalty=None)
    original = qwen_generation_params(args)
    assert original['do_sample'] and original['subtalker_dosample']
    assert (original['temperature'], original['top_k'], original['repetition_penalty']) == (0.9, 50, 1.05)
    args.code_predictor_greedy = True
    args.talker_temperature, args.talker_top_p = 0.3, 1.0
    args.talker_top_k, args.talker_repetition_penalty = 10, 2.0
    result = qwen_generation_params(args)
    assert result['do_sample'] and not result['subtalker_dosample']
    assert (result['temperature'], result['top_p'], result['top_k'], result['repetition_penalty']) == (0.3, 1.0, 10, 2.0)
    args.greedy = True
    result = qwen_generation_params(args)
    assert not result['do_sample'] and not result['subtalker_dosample']
