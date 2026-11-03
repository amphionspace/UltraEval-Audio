"""Score NV-Bench outputs: Multilingual-NVASR transcripts (CER / PCER / OCER) and WavLM SIM to the prompt.

Symbols: each [Tag] is one symbol; other text is lowercased characters without punctuation or spaces
(both languages, as in the NV-Bench paper). CER compares characters only, PCER the ordered tag
sequence only, OCER both. Rates are corpus-level per subset (errors summed over reference length).

  score:     metrics env, one shard:  --run-dir DIR --shard K --num-shards N [--reference]
  summarize: --run-dir DIR --summarize
--reference scores the human recordings (gt_audio) instead of generated audio.
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("zh_single", "zh_multi", "en_single", "en_multi")
TAG = re.compile(r"\[[A-Za-z]+(?:-[A-Za-z]+)?\]")


def symbols(text):
    text = re.sub(r"<\|[^|]*\|>", "", text)  # SenseVoice language / emotion / event markers
    out = []
    for piece in re.split(f"({TAG.pattern})", text):
        if TAG.fullmatch(piece):
            out.append(piece)
        else:
            out += [c for c in piece.lower()
                    if not (c.isspace() or unicodedata.category(c).startswith("P"))]
    return out


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        previous, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (x != y))
    return row[-1]


def errors(reference, hypothesis):
    ref, hyp = symbols(reference), symbols(hypothesis)
    split = lambda seq: ([c for c in seq if not TAG.fullmatch(c)], [c for c in seq if TAG.fullmatch(c)])
    (ref_chars, ref_tags), (hyp_chars, hyp_tags) = split(ref), split(hyp)
    return {"cer": (distance(ref_chars, hyp_chars), len(ref_chars)),
            "pcer": (distance(ref_tags, hyp_tags), len(ref_tags)),
            "ocer": (distance(ref, hyp), len(ref))}


def inputs(run, reference):
    if reference:
        return [{**json.loads(line), "audio": json.loads(line)["gt_audio"]} for subset in SUBSETS
                for line in (ROOT / f"raw_data/nv_bench/prepared/{subset}.jsonl").read_text().splitlines()]
    rows = {}
    for journal in sorted(run.glob("inference-*.jsonl")):
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            rows[(row["subset"], row["index"])] = row
    return [rows[key] for key in sorted(rows)]


def score(args):
    from funasr import AutoModel

    from voice_clone_score import configure

    target = args.run_dir / f"scores-{args.shard:02d}.jsonl"
    done = set()
    if target.exists():
        done = {(r["subset"], r["index"]) for r in map(json.loads, target.read_text().splitlines())}
    rows = [r for k, r in enumerate(inputs(args.run_dir, args.reference))
            if k % args.num_shards == args.shard and (r["subset"], r["index"]) not in done]
    print(f"shard={args.shard} remaining={len(rows)}", flush=True)
    if not rows:
        return
    asr = AutoModel(model=str(ROOT / "init_model/CharlesNi/Multilingual-NVASR"), device="cuda:0",
                    disable_update=True, disable_pbar=True, log_level="ERROR")
    similarity = configure().get_evaluator("simo")
    with target.open("a", buffering=1) as log:
        for i, row in enumerate(rows):
            hypothesis = asr.generate(input=row["audio"], language="auto", use_itn=True)[0]["text"]
            sim = float(similarity(row["audio"], row["prompt_audio"])["simo"])
            log.write(json.dumps({"subset": row["subset"], "index": row["index"], "category": row["category"],
                                  "reference": row["text"], "hypothesis": hypothesis, "sim": sim,
                                  "errors": errors(row["text"], hypothesis)}, ensure_ascii=False) + "\n")
            print(f"shard={args.shard} {i + 1}/{len(rows)}", flush=True)


def summarize(args):
    expected = {(subset, i) for subset in SUBSETS for i, _ in enumerate(
        (ROOT / f"raw_data/nv_bench/prepared/{subset}.jsonl").read_text().splitlines())}
    scored = {}
    for path in sorted(args.run_dir.glob("scores-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            scored[(row["subset"], row["index"])] = row
    if set(scored) != expected:
        raise ValueError(f"Scored {len(scored)} of {len(expected)} samples")
    totals = defaultdict(lambda: defaultdict(float))
    for row in scored.values():
        total = totals[row["subset"]]
        total["count"] += 1
        total["sim"] += row["sim"]
        for metric, (wrong, length) in row["errors"].items():
            total[metric + "_errors"] += wrong
            total[metric + "_length"] += length
    summary = {subset: {"count": int(t["count"]),
                        **{m: round(100 * t[m + "_errors"] / t[m + "_length"], 2) for m in ("cer", "pcer", "ocer")},
                        "sim": round(t["sim"] / t["count"], 4)}
               for subset, t in ((s, totals[s]) for s in SUBSETS)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| subset | n | CER | PCER | OCER | SIM |", "|---|---|---|---|---|---|"]
    lines += [f"| {s} | {v['count']} | {v['cer']} | {v['pcer']} | {v['ocer']} | {v['sim']} |" for s, v in summary.items()]
    (args.run_dir / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args) if args.summarize else score(args)


if __name__ == "__main__":
    main()
