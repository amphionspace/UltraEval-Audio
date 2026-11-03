"""Adapt a dots.tts release or LM-TTS-Training dots export to the Seed-TTS journal format."""

import torch


class DotsVoiceClone:
    """Official dots.tts runtime. speaker_icl prefills the reference transcript and audio;
    speaker_only conditions on the reference's CAM++ x-vector alone."""

    def __init__(self, model_dir, *, conditioning, num_steps=10, guidance_scale=1.2):
        if conditioning not in ("speaker_icl", "speaker_only"):
            raise ValueError("dots.tts conditions on speaker_icl or speaker_only")
        from dots_tts.runtime import DotsTtsRuntime

        self.runtime = DotsTtsRuntime.from_pretrained(str(model_dir), precision="bfloat16")
        self.icl = conditioning == "speaker_icl"
        # The protocol flags mirror the Qwen settings that the audit checks.
        self.params = dict(num_steps=num_steps, guidance_scale=guidance_scale,
                           max_new_tokens=self.runtime.max_generate_length,
                           x_vector_only_mode=not self.icl, non_streaming_mode=True)
        self.paths_used = {}
        prepare = self.runtime._prepare_inputs

        def recorded(**keywords):
            inputs = prepare(**keywords)
            self.paths_used.update(speaker_embedding="prompt_audio" in inputs,
                                   icl_calls=int(bool(inputs["prompt_text"])))
            return inputs

        self.runtime._prepare_inputs = recorded

    def generate(self, batch, seed):
        if len(batch) != 1:
            raise ValueError("dots.tts inference supports batch size 1")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        row = batch[0]
        self.paths_used.clear()
        result = self.runtime.generate(
            text=row["text"], prompt_audio_path=row["prompt_audio"],
            prompt_text=row["prompt_text"] if self.icl else None,
            num_steps=self.params["num_steps"], guidance_scale=self.params["guidance_scale"])
        assert self.paths_used == {"speaker_embedding": True, "icl_calls": int(self.icl)}, self.paths_used
        return [result["audio"].float().cpu().squeeze().numpy()], result["sample_rate"]
