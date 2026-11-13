"""Score StyleBench outputs (metrics env).

Per generated unit: emotion2vec_plus_large scores and embedding, WavLM SIM to the reference it was
conditioned on, ASR (Paraformer for zh, Whisper-large-v3 otherwise) and acoustic measures:
speech seconds (non-silent, top_db 40), level (dBFS over speech), median F0 (pyin 60-600 Hz),
voiced ratio (voiced frames / speech frames) and rate (text units per speech second; zh/ja/ko
characters, en words). --reference scores the real recordings: gt_emotion_* and free_* ground truth.

  score:     --run-dir DIR --shard K --num-shards N [--reference]
  summarize: --run-dir DIR --summarize   (see style_bench_summary.py)
"""
import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

# librosa.pyin segfaults on the shared numba cache (.cache/numba); keep its JIT cache node-local.
os.environ["NUMBA_CACHE_DIR"] = f"/tmp/style-bench-numba-{os.getuid()}"

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("STYLE_BENCH_DIR", ROOT / "raw_data/style_bench"))
E2V = ROOT / "init_model/emotion2vec/models/iic--emotion2vec_plus_large/snapshots/master"
WHISPER = ROOT / "init_model/openai/whisper-large-v3"
PARAFORMER = ROOT / "init_model/JunHowie/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
WHISPER_LANG = {"en": "english", "ja": "japanese", "ko": "korean", "zh": "chinese"}


def units_to_score(run, reference):
    if reference:
        out = []
        for subset in ("gt_emotion_en", "gt_emotion_zh"):
            for line in (BENCH / f"{subset}.jsonl").read_text().splitlines():
                r = json.loads(line)
                out.append(dict(key=f"{subset}|{r['id']}|gt", subset=subset, item=r["id"], condition="gt",
                                audio=r["audio"], text=r["text"], language=r["language"], prompt_audio=None))
        for language in ("zh", "en", "ja", "ko"):
            for line in (BENCH / f"free_{language}.jsonl").read_text().splitlines():
                r = json.loads(line)
                out.append(dict(key=f"free_{language}|{r['id']}|gt", subset=f"free_{language}", item=r["id"],
                                condition="gt", audio=r["gt"], text=r["text"], language=language,
                                prompt_audio=r["ref"]))
        return out
    rows = {}
    for journal in sorted(run.glob("inference-*.jsonl")):
        for line in journal.read_text().splitlines():
            r = json.loads(line)
            r["language"] = r["subset"].rsplit("_", 1)[1]
            rows[r["key"]] = r
    return [rows[k] for k in sorted(rows)]


def normalize(text, language):
    text = unicodedata.normalize("NFKC", text).lower()
    if language == "zh":
        import zhconv
        text = zhconv.convert(text, "zh-cn")
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def error_rate(reference, hypothesis, language):
    """(edits, reference length): words for en, characters otherwise."""
    ref, hyp = normalize(reference, language), normalize(hypothesis, language)
    if language == "en":
        ref_units, hyp_units = ref.split(), hyp.split()
    else:
        ref_units, hyp_units = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return distance(ref_units, hyp_units), len(ref_units)


def text_units(text, language):
    t = normalize(text, language)
    return len(t.split()) if language == "en" else len(t.replace(" ", ""))


def acoustics(path, text, language):
    import librosa

    y, sr = librosa.load(path, sr=16000)
    intervals = librosa.effects.split(y, top_db=40)
    speech = np.concatenate([y[a:b] for a, b in intervals]) if len(intervals) else y
    seconds = len(speech) / sr
    level = 20 * np.log10(np.sqrt(np.mean(speech ** 2)) + 1e-9)
    f0, voiced, _ = librosa.pyin(speech, fmin=60, fmax=600, sr=sr, frame_length=1024, hop_length=160)
    pitch = float(np.nanmedian(f0[voiced])) if voiced.any() else None
    return dict(seconds=len(y) / sr, speech_seconds=seconds, level_db=float(level), f0=pitch,
                voiced_ratio=float(voiced.mean()) if len(voiced) else 0.0,
                rate=text_units(text, language) / seconds if seconds > 0 else None)


def score(args):
    import torch
    from funasr import AutoModel
    from transformers import pipeline

    from voice_clone_score import configure

    target = args.run_dir / f"scores{'-reference' if args.reference else ''}-{args.shard:02d}.jsonl"
    done = {json.loads(line)["key"] for line in target.read_text().splitlines()} if target.exists() else set()
    rows = [r for k, r in enumerate(units_to_score(args.run_dir, args.reference))
            if k % args.num_shards == args.shard and r["key"] not in done]
    print(f"shard={args.shard} remaining={len(rows)}", flush=True)
    if not rows:
        return
    e2v = AutoModel(model=str(E2V), device="cuda:0", disable_update=True, disable_pbar=True, log_level="ERROR")
    paraformer = AutoModel(model=str(PARAFORMER), device="cuda:0", disable_update=True, disable_pbar=True,
                           log_level="ERROR")
    whisper = pipeline("automatic-speech-recognition", model=str(WHISPER), torch_dtype=torch.float16,
                       device="cuda:0")
    similarity = configure().get_evaluator("simo")
    with target.open("a", buffering=1) as log:
        for i, r in enumerate(rows):
            emotion = e2v.generate(r["audio"], granularity="utterance", extract_embedding=True)[0]
            labels = [label.split("/")[-1] for label in emotion["labels"]]
            if r["language"] == "zh":
                hypothesis = paraformer.generate(input=r["audio"])[0]["text"]
            else:
                hypothesis = whisper(r["audio"], generate_kwargs=dict(
                    language=WHISPER_LANG[r["language"]], task="transcribe"))["text"]
            edits, length = error_rate(r["text"], hypothesis, r["language"])
            sim = (float(similarity(r["audio"], r["prompt_audio"])["simo"])
                   if r.get("prompt_audio") else None)
            log.write(json.dumps(dict(
                key=r["key"], subset=r["subset"], item=r["item"], condition=r["condition"],
                emotion=dict(zip(labels, map(float, emotion["scores"]))),
                embedding=[round(float(v), 5) for v in np.asarray(emotion["feats"]).ravel()],
                hypothesis=hypothesis, edits=edits, length=length, sim=sim,
                **acoustics(r["audio"], r["text"], r["language"])), ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(rows)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    score(args)


if __name__ == "__main__":
    main()
