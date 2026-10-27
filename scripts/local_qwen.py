"""Decode an LM-TTS Qwen export with LM-TTS's own KV-cached generation (the training input protocol
of the sibling LM-TTS-Training repo), in either layout, so streaming and non-streaming share one
decoder. Official qwen_tts only knows its overlay streaming, not interleaved chunks."""

import json
import subprocess
import sys
from pathlib import Path

import torch


class LmTtsQwenVoiceClone:
    def __init__(self, model_dir, *, conditioning, max_audio_seconds, sampling, language_auto, streaming):
        if conditioning != "speaker_icl":
            raise ValueError("LM-TTS decoding supports speaker_icl conditioning")
        project = Path(__file__).resolve().parents[2] / "LM-TTS-Training"
        sys.path.insert(0, str(project))
        from lm_tts.inference.codec import CodecSynthesizer

        # Decoding runs whatever LM-TTS-Training code is checked out; record which.
        git = ["git", "-C", str(project)]
        self.decoder = {
            "name": "lm_tts",
            "lm_tts_revision": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
            "lm_tts_dirty": bool(subprocess.check_output([*git, "status", "--porcelain", "lm_tts"], text=True)),
        }

        self.synth = CodecSynthesizer(model_dir, "cuda:0", torch.bfloat16, "sdpa")
        talker = self.synth.raw["talker_config"]
        self.languages = None if language_auto else talker["codec_language_id"]
        self.streaming = streaming
        # Official generate blocks EOS for the first two frames.
        self.params = {**sampling, "min_new_tokens": 2, "x_vector_only_mode": False,
                       "non_streaming_mode": not streaming}
        if max_audio_seconds:
            codec = json.loads((Path(model_dir) / "speech_tokenizer/config.json").read_text())
            rate = codec["output_sample_rate"] / codec["decode_upsample_rate"]
            self.params["max_new_tokens"] = min(2048, int(max_audio_seconds * rate) + 3)
        # Official generate returns one frame fewer than max_new_tokens at the cap.
        self.params["max_new_tokens"] -= 1
        self.paths_used = {}

    def generate(self, batch, seed):
        if len(batch) != 1:
            raise ValueError("Complete-prefix inference supports batch size 1")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        row = batch[0]
        settings = {k: v for k, v in self.params.items()
                    if k not in {"x_vector_only_mode", "non_streaming_mode"}}
        language = -1 if self.languages is None else self.languages[
            "english" if row["language"] == "en" else "chinese"]
        # Official generate penalizes repeats of generated codes only, not the ICL prompt.
        wav, rate = self.synth.synthesize(
            row["text"], row["prompt_audio"], row["prompt_text"], streaming=self.streaming,
            language_codec_id=language, penalize_reference=False, use_cache=True, **settings,
        )
        self.paths_used.clear()
        self.paths_used.update(self.synth.last_conditioning)
        assert self.paths_used == {"speaker_embedding": True, "icl_calls": 1}
        return [wav], rate
