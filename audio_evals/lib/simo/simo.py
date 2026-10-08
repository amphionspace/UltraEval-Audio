import argparse
import json
import os
from pathlib import Path
import select
import sys

import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from models_ecapa_tdnn import ECAPA_TDNN_SMALL
import librosa
from audio_evals.gpu_budget import configure_cuda_budget

MODEL_LIST = [
    "ecapa_tdnn",
    "hubert_large",
    "wav2vec2_xlsr",
    "unispeech_sat",
    "wavlm_base_plus",
    "wavlm_large",
]


def init_model(model_name, checkpoint=None):
    if model_name == "wavlm_large":
        config_path = None
        model = ECAPA_TDNN_SMALL(
            feat_dim=1024, feat_type="wavlm_large", config_path=config_path
        )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"], strict=False)
    return model


def verification(wav1, wav2, model=None, wav2_cut_wav1=False, device="cuda:0"):
    audio_paths = {"prediction": str(wav1), "reference": str(wav2)}
    wav1, sr1 = librosa.load(wav1, sr=None, mono=False)
    if len(wav1.shape) == 2:
        wav1 = wav1[0, :]  # only use one channel
    wav2, sr2 = librosa.load(wav2, sr=None, mono=False)
    if len(wav2.shape) == 2:
        wav2 = wav2[0, :]

    wav1 = torch.from_numpy(wav1).unsqueeze(0).float()
    wav2 = torch.from_numpy(wav2).unsqueeze(0).float()
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    wav1 = wav1.cuda(device)
    wav2 = wav2.cuda(device)

    def embeddings():
        with torch.no_grad():
            return model(wav1), model(wav2)

    retry_on_cpu = False
    try:
        emb1, emb2 = embeddings()
    except torch.cuda.OutOfMemoryError:
        if os.environ.get("AUDIO_EVALS_SIM_CPU_FALLBACK") != "1":
            raise
        retry_on_cpu = True
    if retry_on_cpu:
        # Preserve the complete waveform and scoring model for unusually long outputs.
        wav1, wav2 = wav1.cpu(), wav2.cpu()
        model.cpu()
        torch.cuda.empty_cache()
        try:
            emb1, emb2 = embeddings()
            log_dir = Path(os.environ["AUDIO_EVALS_RUNTIME_LOG_DIR"])
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / f"wavlm-cpu-fallback-{os.getpid()}.jsonl").open("a") as log:
                log.write(json.dumps({**audio_paths,
                                      "device": "cpu", "reason": "cuda_allocation_limit"}) + "\n")
        finally:
            model.to(device)

    sim = F.cosine_similarity(emb1, emb2)
    print(
        "The similarity score between two audios is {:.4f} (-1.0, 1.0).".format(
            sim[0].item()
        ),
        flush=True,
    )
    return sim[0].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to checkpoint file"
    )
    config = parser.parse_args()
    configure_cuda_budget()

    # now just compare two audios similarity
    model_name = "wavlm_large"
    checkpoint = config.path
    model = init_model(model_name, checkpoint)
    model = model.cuda("cuda:0").eval()

    print(f"successfully loaded wavlm_large model from {checkpoint}", flush=True)

    while True:
        try:
            prompt = input()
            anchor = prompt.find("->")
            if anchor == -1:
                print(
                    "Error: Invalid conversation format, must contains  ->, but {}".format(
                        prompt
                    ),
                    flush=True,
                )
                continue
            prefix = prompt[:anchor].strip() + "->"
            wavs = prompt[anchor + 2 :].split(",")
            sim = verification(wavs[0], wavs[1], model=model)
            retry = 3
            while retry:
                print("{}{}".format(prefix, str(sim)), flush=True)
                rlist, _, _ = select.select([sys.stdin], [], [], 1)
                if rlist:
                    finish = sys.stdin.readline().strip()
                    if finish == "{}close".format(prefix):
                        break
                print("not found close signal, will emit again", flush=True)
                retry -= 1
        except Exception as e:
            print("Error:{}".format(e), flush=True)
