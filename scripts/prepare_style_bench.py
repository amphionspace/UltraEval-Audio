"""Build StyleBench: style-control test items from data that no dots instruction model trained on.

Sources (all read from pinned publications of DATA-TTS-UNIFIED):
- FlexiVoice rows held out by selection-flexivoice (reason 7001): EARS 7 speakers and RAVDESS 3
  actors (labelled emotion / paralinguistic style, English) and 14 game characters (free-form
  instructions, zh / en / ja / ko).
- CSEMOTIONS (10 Mandarin speakers, 7 emotions), not part of the FlexiVoice fine-tuning data.

Subsets (one JSONL each under raw_data/style_bench/):
- emotion_en, emotion_zh: the body is a neutral sentence of the speaker, so any emotion comes
  from the instruction. Every item has two references of the same speaker, other text:
  `ref_easy` neutral, `ref_hard` another emotion (FlexiVoice TR-easy / TR-hard).
- paraling_en, paraling_zh: whisper / loud / fast / slow / high / low on a neutral body and
  neutral reference; scored against the default instruction on the same body and reference.
- free_zh, free_en, free_ja, free_ko: a held-out row's own text and instruction, its recording as
  ground truth, reference another utterance of that character in that language.
- gt_emotion_en, gt_emotion_zh: real recordings of each target emotion, for the classifier ceiling.

Run with an environment that has lance and LM-TTS-Training on PYTHONPATH:
  PYTHONPATH=<LM-TTS-Training> python scripts/prepare_style_bench.py
"""
import argparse
import hashlib
import io
import json
import random
from collections import defaultdict
from pathlib import Path

import lance
import numpy as np
import soundfile as sf

from lm_tts.data import flexivoice as fv

ROOT = Path(__file__).resolve().parents[1]
UNIFIED = Path("/workspace/data/DATA-TTS-UNIFIED")
SELECTION = "tts-selection-supervised-tts-20261112T112338bjt-01"  # selection-flexivoice-v3
DEFAULT = fv.DEFAULT_INSTRUCTION
REF_SECONDS = (3.0, 12.0)
BODY_SECONDS = (1.5, 15.0)

# One label set for instructions, references and the classifier (emotion2vec_plus_large names).
EMOTIONS_EN = ("angry", "happy", "sad", "fearful", "surprised", "disgusted")
# fearful is left out: emotion2vec_plus_large recognises 4% of real CSEMOTIONS fearful clips.
EMOTIONS_ZH = ("angry", "happy", "sad", "surprised")
FLEX_LABEL = {"angry": "angry", "happy": "happy", "sad": "sad", "fear": "fearful",
              "surprise": "surprised", "disgust": "disgusted", "neutral": "neutral"}
CSE_LABEL = {"angry": "angry", "happy": "happy", "sad": "sad", "fearful": "fearful",
             "surprise": "surprised", "neutral": "neutral", "playfulness": None}
PARALING = ("whisper", "loud", "fast", "slow", "high", "low")

TEMPLATES = {
    "en": {
        "angry": ["Speak in an angry, furious tone.", "Say it angrily, with strong frustration."],
        "happy": ["Speak in a happy, cheerful tone.", "Say it joyfully, with a bright and lively delivery."],
        "sad": ["Speak in a sad, sorrowful tone.", "Say it sadly, sounding disappointed and low."],
        "fearful": ["Speak in a fearful, frightened tone.", "Say it as if you are scared and nervous."],
        "surprised": ["Speak in a surprised, astonished tone.", "Say it with obvious surprise."],
        "disgusted": ["Speak in a disgusted tone.", "Say it with clear disgust and distaste."],
        "whisper": ["Whisper softly and quietly.", "Say it in a whisper."],
        "loud": ["Speak loudly, projecting your voice.", "Say it in a loud voice."],
        "fast": ["Speak noticeably faster than usual while articulating clearly.", "Say it at a fast pace."],
        "slow": ["Speak noticeably slower than usual.", "Say it slowly and deliberately."],
        "high": ["Speak with a noticeably higher pitch.", "Say it in a high-pitched voice."],
        "low": ["Speak with a noticeably lower pitch.", "Say it in a low, deep voice."],
    },
    "zh": {
        "angry": ["用生气、愤怒的语气说这句话。", "带着强烈的愤怒说出来。"],
        "happy": ["用开心、愉快的语气说这句话。", "带着喜悦、轻快地说出来。"],
        "sad": ["用悲伤、难过的语气说这句话。", "带着失落和伤心说出来。"],
        "fearful": ["用害怕、恐惧的语气说这句话。", "带着紧张和恐惧说出来。"],
        "surprised": ["用惊讶的语气说这句话。", "带着明显的惊奇说出来。"],
        "whisper": ["用轻柔的耳语说这句话，音量很低。", "压低声音，用耳语说出来。"],
        "loud": ["大声地说这句话。", "提高音量，用很响亮的声音说出来。"],
        "fast": ["用明显较快的语速说这句话，保持吐字清晰。", "语速快一些地说出来。"],
        "slow": ["用明显较慢的语速说这句话。", "放慢语速，一字一句地说出来。"],
        "high": ["用明显偏高的音调说这句话。", "提高音调说出来。"],
        "low": ["用明显偏低、低沉的音调说这句话。", "压低音调，用低沉的声音说出来。"],
    },
}


def pick(key, items):
    """Deterministic choice from a key, independent of iteration order."""
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return items[h % len(items)]


def write_wav(raw, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        data, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        sf.write(path, data.mean(axis=1), rate, subtype="PCM_16")
    return str(path)


def flexivoice_heldout():
    manifest = json.loads((UNIFIED / "selections" / SELECTION / "manifest.json").read_text())
    (out,) = manifest["outputs"]
    ds = lance.dataset(UNIFIED / out["table_path"], version=out["base_version"])
    ds = ds.checkout_version((out["branch"], out["lance_version"]))
    rows = ds.to_table(
        columns=["sample_id", "source_config", "language", "text", "metadata_json",
                 "duration_seconds", "audio"],
        filter="selection_reason = 7001").to_pylist()
    result = []
    for r in rows:
        meta = json.loads(r["metadata_json"])
        upstream = meta["upstream"]
        path = upstream.get("wav_path") or ""
        style = fv.style_class(r["source_config"], path, upstream.get("meta"))
        result.append(dict(
            id=r["sample_id"], source=r["source_config"], language=r["language"] or "en",
            text=r["text"].strip(), instruction=meta["instruction"].strip(),
            seconds=r["duration_seconds"], speaker=fv.speaker_key(r["source_config"], path),
            style=fv.STYLES[style] if style >= 0 else None, audio=r["audio"]["bytes"]))
    return result


def csemotions():
    manifest = json.loads((UNIFIED / "datasets/csemotions/v0.1/manifest.json").read_text())
    ds = lance.dataset(UNIFIED / "datasets/csemotions/v0.1" / manifest["table_path"],
                       version=manifest["lance_version"])
    rows = ds.to_table(columns=["sample_id", "text", "speaker_id", "metadata_json",
                                "duration_seconds", "audio"]).to_pylist()
    return [dict(id=r["sample_id"], source="csemotions", language="zh", text=r["text"].strip(),
                 instruction=None, seconds=r["duration_seconds"], speaker=r["speaker_id"],
                 style=json.loads(r["metadata_json"])["emotion"], audio=r["audio"]["bytes"])
            for r in rows]


def fits(row, span):
    return span[0] <= row["seconds"] <= span[1] and row["text"]


def emotion_items(rows, label_of, emotions, language, per_cell, audio_dir, rng):
    """Per (speaker, target emotion): neutral body, neutral and conflicting references."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        label = label_of.get(r["style"]) if r["style"] else None
        if label:
            by[r["speaker"]][label].append(r)
    items, gt = [], []
    for speaker in sorted(by):
        styles = by[speaker]
        bodies = [r for r in styles["neutral"] if fits(r, BODY_SECONDS)]
        neutral_refs = [r for r in styles["neutral"] if fits(r, REF_SECONDS)]
        for emotion in emotions:
            conflicts = [r for e in emotions if e != emotion for r in styles[e] if fits(r, REF_SECONDS)]
            if not bodies or not conflicts or len(neutral_refs) < 2:
                continue
            # Distinct (body text, easy reference, hard reference) per cell; small corpora
            # (RAVDESS: 2 sentences) give fewer than per_cell items rather than repeats.
            seen, k, attempts = set(), 0, 0
            while k < per_cell and attempts < 50 * per_cell:
                attempts += 1
                body = rng.choice(bodies)
                easy = rng.choice([r for r in neutral_refs if r["text"] != body["text"]] or neutral_refs)
                hard = rng.choice([r for r in conflicts if r["text"] != body["text"]] or conflicts)
                combo = (body["text"], easy["id"], hard["id"])
                if combo in seen:
                    continue
                seen.add(combo)
                key = f"{speaker}|{emotion}|{k}"
                k += 1
                items.append(dict(
                    id=hashlib.sha256(key.encode()).hexdigest()[:16], speaker=speaker,
                    language=language, text=body["text"], target=emotion,
                    instruction=pick(key, TEMPLATES[language][emotion]),
                    ref_easy=write_wav(easy["audio"], audio_dir / f"{easy['id']}.wav"),
                    ref_easy_text=easy["text"], ref_easy_emotion="neutral",
                    ref_hard=write_wav(hard["audio"], audio_dir / f"{hard['id']}.wav"),
                    ref_hard_text=hard["text"], ref_hard_emotion=label_of[hard["style"]]))
            for r in styles[emotion][:per_cell]:
                gt.append(dict(id=r["id"], speaker=speaker, language=language, target=emotion,
                               text=r["text"], audio=write_wav(r["audio"], audio_dir / f"{r['id']}.wav")))
    return items, gt


def paraling_items(rows, language, per_cell, audio_dir, rng, neutral_label):
    by = defaultdict(list)
    for r in rows:
        if r["style"] == neutral_label:
            by[r["speaker"]].append(r)
    items = []
    for speaker in sorted(by):
        bodies = [r for r in by[speaker] if fits(r, BODY_SECONDS)]
        refs = [r for r in by[speaker] if fits(r, REF_SECONDS)]
        if not bodies or len(refs) < 2:
            continue
        seen, k, attempts = set(), 0, 0
        while k < per_cell and attempts < 50 * per_cell:
            attempts += 1
            body = rng.choice(bodies)
            ref = rng.choice([r for r in refs if r["text"] != body["text"]] or refs)
            if (body["text"], ref["id"]) in seen:
                continue
            seen.add((body["text"], ref["id"]))
            pair = hashlib.sha256(f"{speaker}|pair|{k}".encode()).hexdigest()[:16]
            for style in PARALING:
                key = f"{speaker}|{style}|{k}"
                items.append(dict(
                    id=hashlib.sha256(key.encode()).hexdigest()[:16], pair=pair, speaker=speaker,
                    language=language, text=body["text"], target=style,
                    instruction=pick(key, TEMPLATES[language][style]),
                    ref=write_wav(ref["audio"], audio_dir / f"{ref['id']}.wav"), ref_text=ref["text"]))
            k += 1
    return items


def free_items(rows, language, count, audio_dir, rng):
    by = defaultdict(list)
    for r in rows:
        if r["source"] in ("genshin", "starrail") and r["language"] == language and r["text"]:
            by[r["speaker"]].append(r)
    speakers = sorted(by)
    items = []
    used = set()  # each target recording at most once
    for k in range(count):
        speaker = speakers[k % len(speakers)]
        targets = [r for r in by[speaker] if fits(r, BODY_SECONDS) and r["id"] not in used]
        refs = [r for r in by[speaker] if fits(r, REF_SECONDS)]
        target = rng.choice(targets)
        used.add(target["id"])
        ref = rng.choice([r for r in refs if r["text"] != target["text"]])
        items.append(dict(
            id=target["id"][:16], speaker=speaker, language=language, text=target["text"],
            instruction=target["instruction"],
            gt=write_wav(target["audio"], audio_dir / f"{target['id']}.wav"),
            ref=write_wav(ref["audio"], audio_dir / f"{ref['id']}.wav"), ref_text=ref["text"]))
    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "raw_data/style_bench_v1")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    audio = args.output / "audio"
    rng = random.Random(args.seed)
    held = flexivoice_heldout()
    labelled_en = [r for r in held if r["source"] in ("ears", "emotion")]
    cse = csemotions()
    subsets = {}
    subsets["emotion_en"], subsets["gt_emotion_en"] = emotion_items(
        labelled_en, FLEX_LABEL, EMOTIONS_EN, "en", 8, audio, rng)
    subsets["emotion_zh"], subsets["gt_emotion_zh"] = emotion_items(
        cse, CSE_LABEL, EMOTIONS_ZH, "zh", 12, audio, rng)
    subsets["paraling_en"] = paraling_items(
        [r for r in held if r["source"] == "ears"], "en", 6, audio, rng, "neutral")
    subsets["paraling_zh"] = paraling_items(cse, "zh", 4, audio, rng, "neutral")
    for language in ("zh", "en", "ja", "ko"):
        subsets[f"free_{language}"] = free_items(held, language, 200, audio, rng)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, items in subsets.items():
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        summary[name] = dict(items=len(items), speakers=len({x["speaker"] for x in items}))
        if items and "target" in items[0]:
            summary[name]["targets"] = dict(sorted(
                {t: sum(x["target"] == t for x in items) for t in {x["target"] for x in items}}.items()))
    summary["sources"] = dict(selection=SELECTION, csemotions="datasets/csemotions/v0.1", seed=args.seed)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
