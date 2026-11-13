"""Summarize StyleBench scores of one run into summary.json and results.md.

Emotion (emotion_en: 6 emotions, emotion_zh: 4; classifier emotion2vec_plus_large, argmax over the
subset's emotions plus neutral):
  ACC-I  predicted == instructed emotion          ACC-R  predicted == the hard reference's emotion
  P-I    mean probability of the instructed emotion, renormalized over those candidates
  ceiling: ACC of the same classifier on real recordings of each emotion (gt_emotion_*)
Paralinguistic style, each item against the default-instruction generation of the same body and
reference (success thresholds): loud level +3 dB; whisper voiced ratio -0.20; fast rate x1.15;
slow rate x0.87; high F0 x1.10; low F0 x0.91. Median change reported alongside.
Free-form (game characters): emotion2vec embedding cosine to the real recording, for the row's
instruction vs the default instruction (win = instruction closer), and acoustic distance to the
real recording (|log2 F0 ratio| in octaves, |log2 rate ratio|, |level difference| dB).
All subsets: SIM (WavLM) to the conditioning reference, WER (en) / CER (zh, ja, ko).

  python scripts/style_bench_summary.py --run-dir res/<run>/style_bench
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("STYLE_BENCH_DIR", ROOT / "raw_data/style_bench"))
EMOTIONS = {"en": ("angry", "happy", "sad", "fearful", "surprised", "disgusted"),
            "zh": ("angry", "happy", "sad", "surprised")}
PARALING = {"loud": ("level_db", "diff", 3.0), "whisper": ("voiced_ratio", "diff", -0.20),
            "fast": ("rate", "ratio", 1.15), "slow": ("rate", "ratio", 0.87),
            "high": ("f0", "ratio", 1.10), "low": ("f0", "ratio", 0.91)}
MODES = ("speaker_only", "speaker_icl")


def load(paths):
    out = {}
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            out[r["key"]] = r
    return out


def manifest(subset):
    return {r.get("id"): r for r in map(json.loads, (BENCH / f"{subset}.jsonl").read_text().splitlines())}


def predict(scores, candidates):
    return max(candidates, key=lambda c: scores.get(c, 0.0))


def prob(scores, target, candidates):
    total = sum(scores.get(c, 0.0) for c in candidates)
    return scores.get(target, 0.0) / total if total > 0 else 0.0


def rate(rows):
    edits, length = sum(r["edits"] for r in rows), sum(r["length"] for r in rows)
    return 100 * edits / length if length else None


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def pct(values):
    m = mean(values)
    return None if m is None else 100 * m


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def change(style, row, base):
    field, kind, _ = PARALING[style]
    x, y = row.get(field), base.get(field)
    if x is None or y is None or (kind == "ratio" and y <= 0):
        return None
    return x - y if kind == "diff" else x / y


def success(style, value):
    _, _, threshold = PARALING[style]
    if value is None:
        return False
    return value >= threshold if threshold > (0 if PARALING[style][1] == "diff" else 1) else value <= threshold


def summarize(run):
    reference = load(sorted((BENCH / "reference").glob("scores-reference-*.jsonl")))
    report = {"emotion": {}, "paraling": {}, "free": {}, "ceiling": {}}
    for language, emotions in EMOTIONS.items():
        candidates = (*emotions, "neutral")
        gt = [r for r in reference.values() if r["subset"] == f"gt_emotion_{language}"]
        items = manifest(f"gt_emotion_{language}")
        per = defaultdict(list)
        for r in gt:
            target = items[r["item"]]["target"]
            per[target].append(predict(r["emotion"], candidates) == target)
        report["ceiling"][language] = dict(
            acc=pct([x for v in per.values() for x in v]), n=len(gt),
            per_emotion={k: round(pct(v), 1) for k, v in sorted(per.items())})
    for mode in MODES:
        scores = load(sorted((run / mode).glob("scores-*.jsonl")))
        if not scores:
            continue
        for language, emotions in EMOTIONS.items():
            candidates = (*emotions, "neutral")
            items = manifest(f"emotion_{language}")
            for condition in ("easy", "hard"):
                rows = [(items[r["item"]], r) for r in scores.values()
                        if r["subset"] == f"emotion_{language}" and r["condition"] == condition]
                if not rows:
                    continue
                per = defaultdict(list)
                for item, r in rows:
                    per[item["target"]].append(predict(r["emotion"], candidates) == item["target"])
                entry = dict(
                    n=len(rows),
                    acc_i=100 * mean([predict(r["emotion"], candidates) == i["target"] for i, r in rows]),
                    p_i=100 * mean([prob(r["emotion"], i["target"], candidates) for i, r in rows]),
                    sim=mean([r["sim"] for _, r in rows]), wer=rate([r for _, r in rows]),
                    per_emotion={k: round(100 * mean(v), 1) for k, v in sorted(per.items())})
                if condition == "hard":
                    entry["acc_r"] = 100 * mean([predict(r["emotion"], candidates) == i["ref_hard_emotion"]
                                                 for i, r in rows])
                report["emotion"][f"{language}|{mode}|{condition}"] = entry
        for language in ("en", "zh"):
            items = manifest(f"paraling_{language}")
            subset = f"paraling_{language}"
            base = {r["item"]: r for r in scores.values() if r["subset"] == subset and r["condition"] == "default"}
            for style in PARALING:
                pairs = [(r, base.get(items[r["item"]]["pair"])) for r in scores.values()
                         if r["subset"] == subset and r["condition"] == "style" and items[r["item"]]["target"] == style]
                pairs = [(r, b) for r, b in pairs if b is not None]
                if not pairs:
                    continue
                values = [change(style, r, b) for r, b in pairs]
                report["paraling"][f"{language}|{mode}|{style}"] = dict(
                    n=len(pairs), success=100 * mean([success(style, v) for v in values]),
                    median_change=float(np.median([v for v in values if v is not None])),
                    measure=f"{PARALING[style][0]} {PARALING[style][1]}",
                    sim=mean([r["sim"] for r, _ in pairs]), wer=rate([r for r, _ in pairs]))
            default_rows = list(base.values())
            if default_rows:
                report["paraling"][f"{language}|{mode}|default"] = dict(
                    n=len(default_rows), sim=mean([r["sim"] for r in default_rows]), wer=rate(default_rows))
        for language in ("zh", "en", "ja", "ko"):
            subset = f"free_{language}"
            gt = {r["item"]: r for r in reference.values() if r["subset"] == subset}
            by = defaultdict(dict)
            for r in scores.values():
                if r["subset"] == subset:
                    by[r["item"]][r["condition"]] = r
            trios = [(v["instr"], v["default"], gt[k]) for k, v in by.items()
                     if "instr" in v and "default" in v and k in gt]
            if not trios:
                continue

            def distances(r, g):
                out = {}
                if r["f0"] and g["f0"]:
                    out["f0_oct"] = abs(np.log2(r["f0"] / g["f0"]))
                if r["rate"] and g["rate"]:
                    out["rate_log2"] = abs(np.log2(r["rate"] / g["rate"]))
                out["level_db"] = abs(r["level_db"] - g["level_db"])
                return out

            entry = dict(n=len(trios))
            for name, pick in (("instr", 0), ("default", 1)):
                rows = [t[pick] for t in trios]
                entry[name] = dict(
                    e_sim=mean([cosine(t[pick]["embedding"], t[2]["embedding"]) for t in trios]),
                    **{k: mean([distances(t[pick], t[2]).get(k) for t in trios])
                       for k in ("f0_oct", "rate_log2", "level_db")},
                    sim=mean([r["sim"] for r in rows]), wer=rate(rows))
            entry["e_sim_win"] = 100 * mean([cosine(a["embedding"], g["embedding"]) > cosine(b["embedding"], g["embedding"])
                                             for a, b, g in trios])
            report["free"][f"{language}|{mode}"] = entry
    return report


def fmt(x, digits=1):
    return "—" if x is None else f"{x:.{digits}f}"


def markdown(report, title):
    lines = [f"# StyleBench · {title}", ""]
    c = report["ceiling"]
    lines += ["## 情感", "", "真实录音上限（同一分类器）：" + "，".join(
        f"{k} {fmt(v['acc'])}%（n={v['n']}）" for k, v in c.items()), "",
              "| 语言 | 参考方式 | 参考 | n | ACC-I ↑ | P-I ↑ | ACC-R ↓ | SIM ↑ | WER/CER ↓ |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for key, e in report["emotion"].items():
        language, mode, condition = key.split("|")
        lines.append(f"| {language} | {mode} | {condition} | {e['n']} | {fmt(e['acc_i'])} | {fmt(e['p_i'])} | "
                     f"{fmt(e.get('acc_r'))} | {fmt(e['sim'], 3)} | {fmt(e['wer'], 2)} |")
    lines += ["", "按情感的 ACC-I：", ""]
    for key, e in report["emotion"].items():
        lines.append(f"- {key}: " + ", ".join(f"{k} {v}" for k, v in e["per_emotion"].items()))
    lines += ["", "## 副语言风格（相对默认指令）", "",
              "| 语言 | 参考方式 | 风格 | n | 成功率 ↑ | 中位变化 | 度量 | SIM ↑ | WER/CER ↓ |",
              "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for key, e in report["paraling"].items():
        language, mode, style = key.split("|")
        lines.append(f"| {language} | {mode} | {style} | {e['n']} | {fmt(e.get('success'))} | "
                     f"{fmt(e.get('median_change'), 2)} | {e.get('measure', '默认指令本身')} | {fmt(e['sim'], 3)} | "
                     f"{fmt(e['wer'], 2)} |")
    lines += ["", "## 自由描述（游戏角色，以真实录音为锚点）", "",
              "| 语言 | 参考方式 | n | E-SIM 指令 / 默认 ↑ | 胜率 ↑ | F0 距离（倍频程）↓ | 语速距离 ↓ | 响度距离 dB ↓ | SIM ↑ | CER/WER ↓ |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, e in report["free"].items():
        language, mode = key.split("|")
        a, b = e["instr"], e["default"]
        lines.append(f"| {language} | {mode} | {e['n']} | {fmt(a['e_sim'], 3)} / {fmt(b['e_sim'], 3)} | "
                     f"{fmt(e['e_sim_win'])} | {fmt(a['f0_oct'], 3)} / {fmt(b['f0_oct'], 3)} | "
                     f"{fmt(a['rate_log2'], 3)} / {fmt(b['rate_log2'], 3)} | {fmt(a['level_db'], 2)} / {fmt(b['level_db'], 2)} | "
                     f"{fmt(a['sim'], 3)} | {fmt(a['wer'], 2)} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    report = summarize(args.run_dir)
    (args.run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = markdown(report, args.title or args.run_dir.parent.name)
    (args.run_dir / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
