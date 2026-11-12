"""Adapt a dots.tts release or LM-TTS-Training dots export to the Seed-TTS journal format."""

import os

import torch

# LM-TTS-Training instruction models (lm_tts/data/flexivoice.py) read "Instruction: …\nText:\n"
# before the body; in the interleave layout the prefix follows "[流式语音合成]" and is not
# interleaved with audio.
INSTRUCTION_TEMPLATE = "Instruction: {instruction}\nText:\n"
INTERLEAVE_PREFIX = "[流式语音合成]"


class DotsVoiceClone:
    """Official dots.tts runtime. speaker_icl prefills the reference transcript and audio;
    speaker_only conditions on the reference's CAM++ x-vector alone.

    template: runtime template name; DOTS_TEMPLATE overrides the default (the runtime's "tts").
    instruction: style instruction for instruction-trained exports (DOTS_INSTRUCTION); it needs
    the interleave layout and precedes the whole text, including the ICL reference transcript."""

    def __init__(self, model_dir, *, conditioning, num_steps=10, guidance_scale=1.2,
                 template=None, instruction=None):
        if conditioning not in ("speaker_icl", "speaker_only"):
            raise ValueError("dots.tts conditions on speaker_icl or speaker_only")
        from dots_tts import runtime as dots_runtime

        template = template or os.environ.get("DOTS_TEMPLATE") or None
        instruction = instruction or os.environ.get("DOTS_INSTRUCTION") or None
        template_text = None
        if instruction:
            if template not in (None, "tts_interleave"):
                raise ValueError("An instruction is only defined for the interleave layout")
            template_text = (INTERLEAVE_PREFIX + INSTRUCTION_TEMPLATE.format(instruction=instruction)
                             + "{interleave}")
            template = "instruction_interleave"
            dots_runtime.RUNTIME_TEMPLATE_BY_NAME[template] = template_text
        self.runtime = dots_runtime.DotsTtsRuntime.from_pretrained(str(model_dir), precision="bfloat16")
        self.icl = conditioning == "speaker_icl"
        self.template = template
        # The protocol flags mirror the Qwen settings that the audit checks.
        self.params = dict(num_steps=num_steps, guidance_scale=guidance_scale,
                           max_new_tokens=self.runtime.max_generate_length,
                           x_vector_only_mode=not self.icl, non_streaming_mode=True,
                           template_name=template or "tts", instruction=instruction,
                           template=template_text)
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
            prompt_text=row["prompt_text"] if self.icl else None, template_name=self.template,
            num_steps=self.params["num_steps"], guidance_scale=self.params["guidance_scale"])
        assert self.paths_used == {"speaker_embedding": True, "icl_calls": int(self.icl)}, self.paths_used
        return [result["audio"].float().cpu().squeeze().numpy()], result["sample_rate"]
