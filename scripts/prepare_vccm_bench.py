"""VccmDataset (ControlSpeech) test sets as local audio + manifests for dots.tts generation.

Text, style prompt and the style labels (dur / pitch / energy / emotion / gender) come from
third_party/controlspeech_eval/VccmDataset/*.csv unchanged; ControlSpeech's own scripts score the
outputs. Audio is gathered from local copies: LibriTTS -> DATA-TTS-UNIFIED libritts_r (same ids),
ESD / RAVDESS -> DATA-TTS-UNIFIED flexivoice (emotion subset), CREMA-D -> raw_data/crema_d_hf
(AbstractTTS/CREMA-D), MEAD -> raw_data/mead_hf (third-party HF upload: 17 male speakers only).
TESS / MESS / SAVEE rows and MEAD speakers outside that upload are skipped and counted.

The speaker prompt is another utterance of the same speaker, picked deterministically by
sha256(item, candidate): for emotion corpora a neutral one (ESD Neutral, RAVDESS emotion 01,
CREMA-D NEU, MEAD neutral), for LibriTTS one of 3-12 s with different text.

    envs/../tts-features python scripts/prepare_vccm_bench.py --out raw_data/vccm_v1
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import lance
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
VCCM = ROOT / "third_party/controlspeech_eval/VccmDataset"
UNIFIED = Path("/workspace/data/DATA-TTS-UNIFIED/datasets")
SETS = {"test": "controlspeech_test.csv", "undomain_style": "controlspeech_test_new_undomain_style.csv",
        "undomain_timbre": "controlspeech_test_undomain_timbre.csv"}


def h(*parts):
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def table(name):
    manifest = json.loads((UNIFIED / name / "v0.1/manifest.json").read_text())
    path = UNIFIED / name / "v0.1" / manifest["table_path"].split("v0.1/")[-1]
    return lance.dataset(path, version=manifest["lance_version"])


class Store:
    """Writes each source audio once as <sha256(bytes)>.wav (PCM 16 at the native rate)."""

    def __init__(self, folder):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)

    def put(self, data):
        path = self.folder / f"{hashlib.sha256(data).hexdigest()}.wav"
        if not path.exists():
            wav, rate = sf.read(io.BytesIO(data))
            sf.write(path, wav, rate, subtype="PCM_16")
        return str(path.resolve())


def libritts(needed_speakers):
    """speaker -> [(key, text, seconds, row index)] and a byte fetcher."""
    ds = table("libritts_r")
    rows = ds.to_table(columns=["source_key", "speaker_id", "text", "duration_seconds"],
                       filter="speaker_id IN ({})".format(
                           ",".join(f"'libritts_r:{s}'" for s in sorted(needed_speakers)))).to_pylist()
    by = defaultdict(list)
    for r in rows:
        by[r["speaker_id"].split(":")[1]].append(r)

    def fetch_many(keys):
        out, keys = {}, sorted(set(keys))
        for i in range(0, len(keys), 500):
            chunk = ",".join(f"'{k}'" for k in keys[i:i + 500])
            for r in ds.to_table(columns=["source_key", "audio"], filter=f"source_key IN ({chunk})").to_pylist():
                out[r["source_key"]] = r["audio"]["bytes"]
        return out
    return by, fetch_many


def flexivoice(prefixes):
    ds = table("flexivoice")
    cond = " OR ".join(f"metadata_json LIKE '%{p}%'" for p in prefixes)
    out = {}
    for r in ds.to_table(columns=["metadata_json", "text", "duration_seconds", "audio"], filter=cond).to_pylist():
        path = json.loads(r["metadata_json"])["upstream"]["wav_path"]
        out[path.split("instruction_data/")[-1]] = (r["text"], r["duration_seconds"], r["audio"]["bytes"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    store = Store(args.out / "audio")
    sets = {name: list(csv.DictReader(open(VCCM / f))) for name, f in SETS.items()}
    rows = [r for v in sets.values() for r in v]

    lib_spk = {r["spk"] for r in rows if "-" not in r["spk"]}
    lib_by, lib_fetch_many = libritts(lib_spk)
    esd_spk = {r["spk"].split("-")[1] for r in rows if r["spk"].startswith("ESD")}
    rav_act = {r["item_name"].split("-")[1] for r in rows if r["spk"].startswith("RAVDESS")}
    flex = flexivoice([f"Emotion_Speech_Dataset/{s}/" for s in esd_spk] + [f"RAVDESS/{a}/" for a in rav_act])
    crema = {}
    for f in sorted(glob.glob(str(ROOT / "raw_data/crema_d_hf/*.parquet"))):
        t = pq.read_table(f, columns=["file", "audio", "transcription"]).to_pylist()
        for r in t:
            crema[r["file"].rsplit("/", 1)[-1].rsplit(".", 1)[0]] = (r["transcription"], r["audio"]["bytes"])
    mead = {}
    for f in sorted(glob.glob(str(ROOT / "raw_data/mead_hf/*.tar"))):
        with tarfile.open(f) as tar:
            for m in tar.getmembers():
                if m.name.endswith(".wav"):
                    mead[m.name.rsplit("/", 1)[-1][:-4]] = tar.extractfile(m).read()

    def pick(item, candidates):
        return min(candidates, key=lambda c: h(item, c)) if candidates else None

    def lib_choice(r):
        item, spk = r["item_name"], r["spk"]
        cands = [c for c in lib_by.get(spk, []) if c["source_key"] != item and 3 <= c["duration_seconds"] <= 12
                 and c["text"].strip() != r["txt"].strip()]
        return pick(item, [c["source_key"] for c in cands]), {c["source_key"]: c["text"] for c in cands}

    needed = set()
    for r in rows:
        if "-" not in r["spk"]:
            choice, _ = lib_choice(r)
            if choice:
                needed |= {r["item_name"], choice}
    lib_audio = lib_fetch_many(needed)

    report = {}
    for name, items in sets.items():
        out, skipped = [], Counter()
        for r in items:
            item, spk, src = r["item_name"], r["spk"], r["spk"].split("-")[0] if "-" in r["spk"] else "LibriTTS"
            gt = ref = ref_text = None
            if src == "LibriTTS":
                choice, texts = lib_choice(r)
                if item in lib_audio and choice in lib_audio:
                    gt, ref, ref_text = store.put(lib_audio[item]), store.put(lib_audio[choice]), texts[choice]
            elif src == "ESD":
                _, s, emo, utt = item.rsplit("-", 3)
                key = f"Emotion_Speech_Dataset/{s}/{emo}/{utt}.wav"
                cands = [k for k in flex if k.startswith(f"Emotion_Speech_Dataset/{s}/Neutral/") and flex[k][1] >= 2]
                choice = pick(item, cands)
                if key in flex and choice:
                    gt, ref, ref_text = store.put(flex[key][2]), store.put(flex[choice][2]), flex[choice][0]
            elif src == "RAVDESS":
                _, actor, code = item.split("-", 2)
                key = f"RAVDESS/{actor}/{code}.wav"
                cands = [k for k in flex if k.startswith(f"RAVDESS/{actor}/") and k.rsplit("/", 1)[-1].split("-")[2] == "01"]
                choice = pick(item, cands)
                if key in flex and choice:
                    gt, ref, ref_text = store.put(flex[key][2]), store.put(flex[choice][2]), flex[choice][0]
            elif src == "CREMA":
                key = item.split("AudioWAV-")[-1]
                actor, sentence = key.split("_")[0], key.split("_")[1]
                cands = [k for k in crema if k.startswith(actor + "_") and k.split("_")[2] == "NEU" and k.split("_")[1] != sentence]
                choice = pick(item, cands)
                if key in crema and choice:
                    gt, ref, ref_text = store.put(crema[key][1]), store.put(crema[choice][1]), crema[choice][0]
            elif src == "MEAD":
                _, s, _, emo, level, num = item.split("-")
                key = f"{s}_{emo}_{level}_{num}"
                cands = [k for k in mead if k.startswith(f"{s}_neutral_") and k != key]
                choice = pick(item, cands)
                if key in mead and choice:
                    gt, ref, ref_text = store.put(mead[key]), store.put(mead[choice]), None
            if gt is None:
                skipped[src] += 1
                continue
            out.append({"id": h(name, item)[:16], "set": name, "item_name": item, "source": src, "speaker": spk,
                        "gender": r["gender"], "language": "en", "text": r["txt"], "instruction": r["style_prompt"],
                        "dur": r["dur"], "pitch": r["pitch"], "energy": r["energy"], "emotion": r["emotion"],
                        "gt": gt, "ref": ref, "ref_text": ref_text})
        with open(args.out / f"vccm_{name}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")
        report[name] = {"rows": len(items), "kept": len(out), "skipped": dict(skipped),
                        "kept_by_source": dict(Counter(o["source"] for o in out))}
    (args.out / "prepare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
