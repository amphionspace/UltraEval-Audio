"""Resume slow HF large-file downloads with checked byte ranges and LFS SHA-256."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from huggingface_hub import HfApi
import requests

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 16 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def download(repo, workers):
    info = HfApi().model_info(repo, files_metadata=True)
    directory = ROOT / "init_model" / repo
    directory.mkdir(parents=True, exist_ok=True)
    for sibling in info.siblings:
        relative = sibling.rfilename
        # Whisper has duplicate framework formats; only HF safetensors and tokenizer files are needed.
        if repo == "openai/whisper-large-v3" and not (relative.endswith((".json", ".txt", ".tiktoken")) or relative == "model.safetensors"):
            continue
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        size = sibling.size
        lfs = sibling.lfs
        expected_hash = lfs.sha256 if lfs else None
        if target.exists() and target.stat().st_size == size:
            if expected_hash is None or sha256(target) == expected_hash:
                continue
        url = f"https://huggingface.co/{repo}/resolve/{info.sha}/{relative}"
        if size < CHUNK:
            with requests.get(url, timeout=(20, 120)) as response:
                response.raise_for_status()
                data = response.content
            if len(data) != size:
                raise RuntimeError(f"Incorrect size for {relative}")
            target.write_bytes(data)
            continue
        partial = target.with_name(target.name + ".ranges")
        journal = partial.with_name(partial.name + ".json")
        done = set(json.loads(journal.read_text())) if journal.exists() and partial.exists() else set()
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, size)
        lock = threading.Lock()
        if not done and expected_hash:
            candidates = list((directory / ".cache/huggingface/download").rglob(f"*.{expected_hash}.*.incomplete"))
            if candidates:
                candidate = max(candidates, key=lambda p: p.stat().st_size)
                full_chunks = min(candidate.stat().st_size, size) // CHUNK
                with candidate.open("rb") as f:
                    for index in range(full_chunks):
                        os.pwrite(fd, f.read(CHUNK), index * CHUNK)
                        done.add(index)
                print(f"Recovered {full_chunks * CHUNK / 2**20:.0f} MiB from {candidate}", flush=True)
        count = (size + CHUNK - 1) // CHUNK

        def part(index):
            start, end = index * CHUNK, min(size, (index + 1) * CHUNK) - 1
            for attempt in range(8):
                endpoint = "https://huggingface.co" if attempt % 2 == 0 else "https://hf-mirror.com"
                request_url = url.replace("https://huggingface.co", endpoint) + f"?download=true&range_start={start}"
                try:
                    with requests.get(request_url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60), stream=True) as r:
                        r.raise_for_status()
                        if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise RuntimeError(f"Incorrect range: {r.status_code} {r.headers.get('Content-Range')}")
                        data = r.content
                    if len(data) != end - start + 1:
                        raise RuntimeError("Incomplete range response")
                    os.pwrite(fd, data, start)
                    with lock:
                        done.add(index)
                        temp = journal.with_suffix(".tmp")
                        temp.write_text(json.dumps(sorted(done)))
                        os.replace(temp, journal)
                        print(f"{repo}/{relative}: {len(done)}/{count} chunks", flush=True)
                    return
                except Exception as exc:
                    print(f"retry {relative} chunk={index} attempt={attempt+1}: {exc}", flush=True)
                    time.sleep(min(attempt + 1, 5))
            raise RuntimeError(f"Failed downloading {relative} range {start}-{end}")

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(part, [i for i in range(count) if i not in done]))
        finally:
            os.close(fd)
        if expected_hash and sha256(partial) != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch: {partial}")
        os.replace(partial, target)
        print(f"VERIFIED {target}", flush=True)
    (directory / "download_provenance.json").write_text(json.dumps({"repo": repo, "revision": info.sha,
        "endpoint": "Hugging Face with hf-mirror fallback", "path": str(directory),
        "verification": "LFS SHA-256 checked", "repo_type": "model"}, indent=2))
    print(f"DONE {repo}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.repo, args.workers)
