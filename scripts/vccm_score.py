"""Score VccmDataset generations with ControlSpeech's official metrics (third_party/controlspeech_eval).

Per row (shard of the journals in --run-dir, or the ground truth with --reference):
  emotion   emotion2vec_base v2.0.4 frame features -> ControlSpeech's fine-tuned 8-class head
            (emotion_acc/emotion2vec_recognizer_seq_10w, labels as in controlnet_emo_acc.py);
            only rows with an emotion label
  pitch     pitch_energy_speed_acc/energy_pitch_acc.py: get_pitch_energy + judge_pitch (gender
            thresholds) / judge_energy, unchanged
  wer       whisper-large-v3 (language english), Whisper's English normalizer on both sides
  sim       microsoft/wavlm-base-plus-sv x-vector cosine to the speaker prompt (spk_sv.py)
Speed accuracy needs MFA alignments (duration_acc.py) and is not computed here.

    envs/metrics/bin/python scripts/vccm_score.py --run-dir res/<run>/vccm/speaker_only --shard 0 --num-shards 4
    envs/metrics/bin/python scripts/vccm_score.py --summary --run-dir res/<run>/vccm/speaker_only
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / "third_party/controlspeech_eval"
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/style-bench-numba-{os.getuid()}")
EMOTION_MAP = {"surprised": 0, "contempt": 1, "sad": 2, "neutral": 3, "disgusted": 4, "happy": 5, "angry": 6, "fear": 7}
E2V_BASE = ROOT / "init_model/emotion2vec_base/models/iic--emotion2vec_base/snapshots/v2.0.4"
WHISPER = ROOT / "init_model/openai/whisper-large-v3"
WAVLM_SV = ROOT / "init_model/microsoft/wavlm-base-plus-sv"


def rows_to_score(run, reference, bench):
    if reference:
        out = []
        for name in ("test", "undomain_style", "undomain_timbre"):
            for line in (bench / f"vccm_{name}.jsonl").read_text().splitlines():
                r = json.loads(line)
                out.append({**r, "key": f"{name}|{r['id']}", "audio": r["gt"]})
        return out
    rows = {}
    for journal in sorted(run.glob("inference-*.jsonl")):
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            rows[r["key"]] = r
    return [rows[k] for k in sorted(rows)]


def score(args):
    import numpy as np
    import torch
    from funasr import AutoModel
    from transformers import (Wav2Vec2FeatureExtractor, WavLMForXVector, WhisperForConditionalGeneration,
                              WhisperProcessor)
    import librosa

    sys.path.insert(0, str(CS / "emotion_acc"))
    sys.path.insert(0, str(CS / "pitch_energy_speed_acc"))
    from ckpt_utils import load_ckpt
    from emotion_net import Emotion2Vec_finetume
    # Keep the vendored metric unchanged; adapt its old positional librosa API.
    import mel_process
    from types import SimpleNamespace
    mel = librosa.filters.mel
    mel_process.librosa = SimpleNamespace(
        core=librosa.core, stft=librosa.stft,
        filters=SimpleNamespace(mel=lambda sr, n_fft, n_mels, fmin, fmax:
                                mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)))
    from energy_pitch_acc import get_pitch_energy, judge_energy, judge_pitch
    import funasr.utils.install_model_requirements as model_requirements
    model_requirements.install_requirements = lambda path: True

    run = args.run_dir
    target = run / f"scores{'-reference' if args.reference else ''}-{args.shard:02d}.jsonl"
    done = {json.loads(l)["key"] for l in target.read_text().splitlines()} if target.exists() else set()
    todo = [r for r in rows_to_score(run, args.reference, args.bench_dir)
            if int(hashlib.sha256(r["key"].encode()).hexdigest(), 16) % args.num_shards == args.shard
            and r["key"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"shard={args.shard} remaining={len(todo)}", flush=True)
    if not todo:
        return
    device = "cuda:0"
    e2v = AutoModel(model=str(E2V_BASE), device=device, disable_update=True, disable_pbar=True, log_level="ERROR")
    head = Emotion2Vec_finetume(768, 8)
    load_ckpt(head, str(CS / "emotion_acc/emotion2vec_recognizer_seq_10w"), "model")
    head.eval().to(device)
    processor = WhisperProcessor.from_pretrained(str(WHISPER))
    whisper = WhisperForConditionalGeneration.from_pretrained(str(WHISPER), torch_dtype=torch.float16).to(device)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(str(WAVLM_SV))
    sv = WavLMForXVector.from_pretrained(str(WAVLM_SV)).to(device).eval()

    def xvec(path):
        wav, _ = librosa.load(path, sr=16000)
        inputs = fe([wav], padding=True, return_tensors="pt", sampling_rate=16000).to(device)
        return torch.nn.functional.normalize(sv(**inputs).embeddings, dim=-1)[0]

    with target.open("a", buffering=1) as log, torch.inference_mode():
        for i, r in enumerate(todo):
            out = {"key": r["key"], "set": r["set"], "source": r["source"], "gender": r["gender"],
                   "dur": r["dur"], "pitch": r["pitch"], "energy": r["energy"], "emotion": r["emotion"]}
            if not r.get("audio") or not Path(r["audio"]).exists():
                log.write(json.dumps({**out, "failed": True}) + "\n")
                continue
            if r["emotion"] in EMOTION_MAP:
                feats = e2v.generate(r["audio"], granularity="frame", extract_embedding=True)[0]["feats"]
                logits = head(torch.tensor(np.asarray(feats), dtype=torch.float32)[None].to(device))["logits"]
                out["emotion_pred"] = [k for k, v in EMOTION_MAP.items() if v == int(logits.argmax(-1))][0]
            m_pitch, m_energy = get_pitch_energy(r["audio"])
            out.update(m_pitch=float(m_pitch), m_energy=float(m_energy),
                       energy_ok=bool(judge_energy(m_energy, r["energy"])),
                       pitch_ok=bool(judge_pitch(m_pitch, r["pitch"], r["gender"])))
            wav, _ = librosa.load(r["audio"], sr=16000)
            features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, torch.float16)
            ids = whisper.generate(features, language="english", task="transcribe")[0]
            out["hypothesis"] = processor.decode(ids, skip_special_tokens=True)
            out["ref_norm"] = processor.tokenizer._normalize(r["text"])
            out["hyp_norm"] = processor.tokenizer._normalize(out["hypothesis"])
            if not args.reference:
                out["sim"] = float(torch.dot(xvec(r["audio"]), xvec(r["ref"])))
            log.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(todo)}", flush=True)


def edits(ref, hyp):
    a, b = ref.split(), hyp.split()
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1], len(a)


def summary(run, reference=False):
    pattern = "scores-reference-[0-9]*.jsonl" if reference else "scores-[0-9]*.jsonl"
    rows = list({r["key"]: r for f in sorted(run.glob(pattern))
                 for l in f.read_text().splitlines() if l.strip()
                 for r in [json.loads(l)]}.values())
    report = {}
    for name in ("test", "undomain_style", "undomain_timbre"):
        rs = [r for r in rows if r["set"] == name and not r.get("failed")]
        if not rs:
            continue
        emo = [r for r in rs if "emotion_pred" in r]
        entry = {
            "n": len(rs), "failed": sum(1 for r in rows if r["set"] == name and r.get("failed")),
            "energy_acc": 100 * sum(r["energy_ok"] for r in rs) / len(rs),
            # Official energy_pitch_acc.py counts pitch only after energy is right.
            "pitch_acc_official": 100 * sum(r["energy_ok"] and r["pitch_ok"] for r in rs) / len(rs),
            "pitch_acc": 100 * sum(r["pitch_ok"] for r in rs) / len(rs),
            "emotion_acc": 100 * sum(r["emotion_pred"] == r["emotion"] for r in emo) / len(emo) if emo else None,
            "emotion_n": len(emo),
            "wer": 100 * sum(edits(r["ref_norm"], r["hyp_norm"])[0] for r in rs)
            / max(sum(edits(r["ref_norm"], r["hyp_norm"])[1] for r in rs), 1),
        }
        sims = [r["sim"] for r in rs if r.get("sim") is not None]
        entry["sim"] = sum(sims) / len(sims) if sims else None
        report[name] = entry
    (run / ("summary-reference.json" if reference else "summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "raw_data/vccm_v1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--reference", action="store_true", help="score the ground-truth recordings")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--limit", type=int, help="maximum new rows for a smoke test")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        parser.error("require 0 <= shard < num-shards")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summary(args.run_dir, args.reference)
    else:
        score(args)


if __name__ == "__main__":
    main()
