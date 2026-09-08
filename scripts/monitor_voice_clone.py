"""Read-only progress monitor for the long-running voice-clone pipeline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="res/voice_clone_20260907")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    run = ROOT / args.run_dir
    while True:
        try:
            state = json.loads((run / "pipeline_status.json").read_text())
            generated, scored, errors = Counter(), Counter(), Counter()
            for directory in sorted(run.iterdir()):
                if not directory.is_dir():
                    continue
                for prefix, counts in (("inference", generated), ("scores", scored)):
                    unique = set()
                    for file in directory.glob(prefix + "-*.jsonl"):
                        for line in file.read_text().splitlines():
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # A writer may be flushing its current row.
                            key = (row["dataset"], row["index"])
                            if row.get("status") == "ok":
                                unique.add(key)
                            else:
                                errors[directory.name] += 1
                    counts[directory.name] = len(unique)
            stage = state.get("stages", [{}])[-1]
            print(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "status": state["status"], "stage": stage.get("name"),
                              "generated": dict(generated), "scored": dict(scored),
                              "error_records": dict(errors)}, ensure_ascii=False), flush=True)
            if state["status"] != "running":
                return
        except (FileNotFoundError, json.JSONDecodeError):
            print("Waiting for a complete pipeline status snapshot", flush=True)
        time.sleep(min(60, max(1, args.interval)))


if __name__ == "__main__":
    main()
