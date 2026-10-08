"""Audit a custom local checkpoint, including per-sample conditioning failures."""
import hashlib
import json
import sys
import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest
import soundfile as sf

from scripts import audit_voice_clone


def setup_run(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    run = tmp_path / "run"
    model = run / "speaker_only"
    model.mkdir(parents=True)
    dataset = "seed_tts_eval_en"
    sample = {"dataset": dataset, "index": 0, "language": "en", "text": "Hello world.",
              "prompt_text": "A reference.", "prompt_audio": str(tmp_path / "reference.wav"),
              "prompt_sha256": "reference", "source_id": "sample"}
    raw = json.dumps(sample) + "\n"
    (inputs / f"{dataset}.jsonl").write_text(raw)
    (inputs / "manifest_metadata.json").write_text(json.dumps({dataset: {
        "count": 1, "sha256": hashlib.sha256(raw.encode()).hexdigest()}}))
    audio = model / dataset / "000000.wav"
    audio.parent.mkdir()
    sf.write(audio, np.zeros(2400), 24000, subtype="PCM_16")
    row = {**sample, "model": "speaker_only", "audio": str(audio), "status": "ok",
           "seed": 42, "batch_size": 1, "sample_rate": 24000, "duration": 0.1,
           "conditioning_verified": {"speaker_embedding": True, "icl_calls": 0}}
    (model / "inference-00.jsonl").write_text(json.dumps(row) + "\n")
    score = {"pred": str(audio), "ref": sample["prompt_audio"], "label_text": sample["text"],
             "wer%": 0, "simo": 0.75}
    (model / "scores-seed-en.jsonl").write_text(json.dumps({"model": "speaker_only", "dataset": dataset,
        "index": 0, "status": "ok", "score": score}) + "\n")
    (model / "settings-00.json").write_text(json.dumps({"model": "speaker_only", "shard": 0,
        "num_shards": 1, "batch_size": 1, "limit": 0, "model_path": "/export", "conditioning": "speaker_only",
        "icl_only_ablation": False, "sampling": {"x_vector_only_mode": True},
        "non_streaming": True, "language_auto": True}))
    (model / "conditioning-verified-00.json").write_text(json.dumps({"model": "speaker_only",
        "batch_size": 1, "speaker_embedding": True, "icl_calls": 0}))
    return inputs, run, model, row


@pytest.mark.parametrize("corruption", [None, "conditioning", "duplicate", "label", "sampling"])
def test_custom_checkpoint_audit(tmp_path, monkeypatch, corruption):
    inputs, run, model, row = setup_run(tmp_path)
    if corruption == "conditioning":
        row["conditioning_verified"]["icl_calls"] = 1
        (model / "inference-00.jsonl").write_text(json.dumps(row) + "\n")
    elif corruption == "duplicate":
        with (model / "inference-00.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    elif corruption == "label":
        path = model / "scores-seed-en.jsonl"
        result = json.loads(path.read_text())
        result["score"]["label_text"] = "Wrong target text."
        path.write_text(json.dumps(result) + "\n")
    elif corruption == "sampling":
        path = model / "settings-00.json"
        cfg = json.loads(path.read_text())
        cfg["greedy"] = True
        cfg["sampling"].update(do_sample=False, subtalker_dosample=True)
        path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["audit", "--run-dir", str(run), "--manifest-dir", str(inputs),
        "--models", "speaker_only", "--num-shards", "1", "--output-name", "audit.json"])
    if corruption:
        with pytest.raises(ValueError):
            audit_voice_clone.main()
        assert not (run / "audit.json").exists()
    else:
        audit_voice_clone.main()
        assert json.loads((run / "audit.json").read_text())["passed"] is True


@pytest.mark.parametrize("greedy", [True, False])
def test_audit_enforces_requested_greedy_mode(tmp_path, monkeypatch, greedy):
    inputs, run, model, _ = setup_run(tmp_path)
    path = model / "settings-00.json"
    cfg = json.loads(path.read_text())
    cfg["greedy"] = greedy
    cfg["sampling"].update(do_sample=not greedy, subtalker_dosample=not greedy)
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys, "argv", ["audit", "--run-dir", str(run), "--manifest-dir", str(inputs),
        "--models", "speaker_only", "--num-shards", "1", "--expect-greedy"])
    if greedy:
        audit_voice_clone.main()
    else:
        with pytest.raises(ValueError, match="Expected greedy inference"):
            audit_voice_clone.main()


def test_similarity_allocation_failure_retries_full_waveforms(tmp_path, monkeypatch):
    import torch
    stub = types.ModuleType("models_ecapa_tdnn")
    stub.ECAPA_TDNN_SMALL = object
    monkeypatch.setitem(sys.modules, "models_ecapa_tdnn", stub)
    path = Path(__file__).resolve().parents[1] / "audio_evals/lib/simo/simo.py"
    spec = importlib.util.spec_from_file_location("test_simo_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.librosa, "load", lambda *a, **kw: (np.zeros(32000, dtype=np.float32), 16000))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setenv("AUDIO_EVALS_SIM_CPU_FALLBACK", "1")
    monkeypatch.setenv("AUDIO_EVALS_RUNTIME_LOG_DIR", str(tmp_path))

    class Model:
        calls = 0
        restored = False

        def __call__(self, waveform):
            self.calls += 1
            if self.calls == 1:
                raise torch.cuda.OutOfMemoryError("allocation cap")
            assert waveform.shape == (1, 32000)
            assert waveform.device.type == "cpu"
            return torch.tensor([[1.0, 2.0]])

        def cpu(self):
            return self

        def to(self, device):
            self.restored = device == "cuda:0"
            return self

    model = Model()
    score = module.verification("prediction.wav", "reference.wav", model=model)
    assert score == pytest.approx(1.0)
    assert model.restored
    record = json.loads(next(tmp_path.glob("wavlm-cpu-fallback-*.jsonl")).read_text())
    assert record["prediction"] == "prediction.wav"
    assert record["reference"] == "reference.wav"
