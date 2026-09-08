"""Download pinned model/dataset snapshots, retrying against HF mirror."""
import concurrent.futures
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ITEMS = [
    ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "model", None),
    ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model", None),
    ("BreezeBlue/Breeze-TTS-2", "model", None),
    ("TwinkStart/Seed-TTS-Eval", "dataset", None),
    ("yuekai/CV3-Eval", "dataset", ["README.md", "data/zero_shot_en-*", "data/zero_shot_zh-*", "data/zero_shot_hard_en-*", "data/zero_shot_hard_zh-*"]),
]


def download(item):
    repo, kind, patterns = item
    destination = ROOT / ("init_model" if kind == "model" else "raw_data") / repo
    for attempt in range(6):
        endpoint = "https://huggingface.co" if attempt % 2 == 0 else "https://hf-mirror.com"
        try:
            info = HfApi(endpoint=endpoint).repo_info(repo, repo_type=kind)
            print(f"Downloading {repo}@{info.sha} from {endpoint}", flush=True)
            snapshot_download(repo, repo_type=kind, revision=info.sha,
                              endpoint=endpoint, local_dir=destination,
                              allow_patterns=patterns, max_workers=4)
            metadata = {"repo": repo, "revision": info.sha, "endpoint": endpoint,
                        "path": str(destination), "repo_type": kind}
            (destination / "download_provenance.json").write_text(json.dumps(metadata, indent=2))
            print(f"DONE {repo}", flush=True)
            return metadata
        except Exception as exc:
            print(f"Attempt {attempt + 1} failed for {repo}: {exc}", flush=True)
            time.sleep(2)
    raise RuntimeError(f"Download failed: {repo}")


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, ITEMS))
    (ROOT / "log" / "voice_clone_downloads.json").write_text(json.dumps(results, indent=2))
