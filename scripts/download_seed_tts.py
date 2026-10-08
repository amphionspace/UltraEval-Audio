"""Download Seed-TTS inputs and scoring weights from HF Mirror without proxies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

# Set these before importing huggingface_hub, which reads endpoint settings on import.
for key in list(os.environ):
    if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
        del os.environ[key]
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
ITEMS = {
    "data": ("TwinkStart/Seed-TTS-Eval", "dataset", ["README.md", "en/*.parquet", "zh/*.parquet"]),
    "whisper": ("openai/whisper-large-v3", "model", [
        "model.safetensors", "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
        "vocab.json", "merges.txt", "normalizer.json", "README.md"]),
    "wavlm_sv": ("hidoba/wavlm_large_finetune", "model", ["wavlm_large_finetune.pth"]),
    "wavlm_backbone": ("s3prl/converted_ckpts", "model", ["wavlm_large.pt"]),
    # The funasr HF repository is a README-only placeholder; this is the full FP32 mirror.
    "paraformer": ("JunHowie/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", "model",
                   ["README.md", "model.pt", "config.yaml", "configuration.json", "am.mvn", "tokens.json", "seg_dict"]),
}


def download(name):
    repo, kind, patterns = ITEMS[name]
    destination = ROOT / ("raw_data" if kind == "dataset" else "init_model") / repo
    destination.mkdir(parents=True, exist_ok=True)
    pin = destination / "download_revision.json"
    api = HfApi(endpoint=os.environ["HF_ENDPOINT"])
    revision = json.loads(pin.read_text())["revision"] if pin.exists() else None
    info = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
    pin.write_text(json.dumps({"repo": repo, "revision": info.sha}, indent=2))
    print(f"Downloading {repo}@{info.sha}", flush=True)
    snapshot_download(repo, repo_type=kind, revision=info.sha, local_dir=destination,
                      allow_patterns=patterns, endpoint=os.environ["HF_ENDPOINT"], max_workers=4)
    inventory = {}
    for file in info.siblings:
        path = destination / file.rfilename
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != file.size or (file.lfs and digest.hexdigest() != file.lfs.sha256):
            raise ValueError(f"Downloaded file failed verification: {path}")
        inventory[file.rfilename] = {"bytes": file.size, "sha256": digest.hexdigest()}
    provenance = {"repo": repo, "repo_type": kind, "revision": info.sha,
                  "endpoint": os.environ["HF_ENDPOINT"], "proxy": False, "files": inventory}
    (destination / "download_provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"Verified {repo}: {len(inventory)} files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", nargs="+", choices=list(ITEMS), default=list(ITEMS))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, args.items))
