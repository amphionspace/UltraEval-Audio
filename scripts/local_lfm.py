"""Adapt the sibling LM-TTS-Training LFM export to the Seed-TTS journal format."""

import json
import sys
from pathlib import Path

import torch


class LfmVoiceClone:
    def __init__(self, model_dir, *, conditioning, max_audio_seconds, greedy=False):
        if conditioning != "speaker_icl":
            raise ValueError("This LFM evaluation adapter requires speaker_icl conditioning")
        project = Path(__file__).resolve().parents[2] / "LM-TTS-Training"
        sys.path.insert(0, str(project))
        from lm_tts.inference.codec import CodecSynthesizer

        self.synth = CodecSynthesizer(model_dir, "cuda:0", torch.bfloat16, "sdpa")
        if not self.synth.model.config.lm_tts_use_speaker_embedding:
            raise ValueError("speaker_icl requires an LFM export with a speaker encoder")
        self.params = dict(
            max_new_tokens=2048, do_sample=not greedy, top_k=50, top_p=1.0,
            temperature=0.9, repetition_penalty=1.05, subtalker_dosample=not greedy,
            subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9,
            x_vector_only_mode=False, non_streaming_mode=True,
        )
        if max_audio_seconds:
            codec = json.loads((Path(model_dir) / "speech_tokenizer/config.json").read_text())
            rate = codec["output_sample_rate"] / codec["decode_upsample_rate"]
            self.params["max_new_tokens"] = min(2048, int(max_audio_seconds * rate) + 3)
        self.paths_used = {}

    def generate(self, batch, seed):
        if len(batch) != 1:
            raise ValueError("LFM complete-prefix inference supports batch size 1")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        row = batch[0]
        # These two flags describe the protocol; they are not token sampler arguments.
        settings = {k: v for k, v in self.params.items()
                    if k not in {"x_vector_only_mode", "non_streaming_mode"}}
        wav, rate = self.synth.synthesize(
            row["text"], row["prompt_audio"], row["prompt_text"], **settings
        )
        self.paths_used.clear()
        self.paths_used.update(self.synth.last_conditioning)
        assert self.paths_used == {"speaker_embedding": True, "icl_calls": 1}
        return [wav], rate
