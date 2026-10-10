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


@pytest.mark.parametrize("duration", [30.0, 30.08])
def test_duration_filter_preserves_coverage_and_threshold(tmp_path, monkeypatch, duration):
    from scripts import summarize_voice_clone

    inputs, run, model, row = setup_run(tmp_path)
    dataset = row["dataset"]
    extra = {**row, "index": 1, "seed": 43, "duration": duration,
             "audio": str(model / dataset / "000001.wav")}
    sf.write(extra["audio"], np.zeros(round(duration * 24000)), 24000, subtype="PCM_16")
    with (model / "inference-00.jsonl").open("a") as stream:
        stream.write(json.dumps(extra) + "\n")
    manifest = inputs / f"{dataset}.jsonl"
    original = json.loads(manifest.read_text())
    raw = manifest.read_text() + json.dumps({**original, "index": 1}) + "\n"
    manifest.write_text(raw)
    (inputs / "manifest_metadata.json").write_text(json.dumps({dataset: {
        "count": 2, "sha256": hashlib.sha256(raw.encode()).hexdigest()}}))
    score_path = model / "scores-seed-en.jsonl"
    extra_score = {"model": "speaker_only", "dataset": dataset, "index": 1, "status": "ok",
                   "score": {"pred": extra["audio"], "ref": extra["prompt_audio"],
                             "label_text": extra["text"], "wer%": 99, "simo": 0.1}}
    if duration <= 30:
        with score_path.open("a") as stream:
            stream.write(json.dumps(extra_score) + "\n")
    args = ["audit", "--run-dir", str(run), "--manifest-dir", str(inputs),
            "--models", "speaker_only", "--max-audio-seconds", "30"]
    monkeypatch.setattr(sys, "argv", args + ["--num-shards", "1", "--output-name", "audit.json"])
    audit_voice_clone.main()
    receipt = json.loads((run / "audit.json").read_text())["models"]["speaker_only"]
    assert receipt["generation_rows"] == 2
    assert receipt["score_rows"] == (2 if duration <= 30 else 1)
    assert receipt["excluded_rows"] == (0 if duration <= 30 else 1)
    monkeypatch.setattr(sys, "argv", args)
    summarize_voice_clone.main()
    summary = json.loads((run / "summary.json").read_text())[0]
    assert summary["complete"] and summary["expected"] == 2
    assert summary["mean"]["wer%"] == (49.5 if duration <= 30 else 0)
    if duration > 30:
        # Excluded audio must never be accepted as a scored short utterance.
        with score_path.open("a") as stream:
            stream.write(json.dumps(extra_score) + "\n")
        monkeypatch.setattr(sys, "argv", args + ["--num-shards", "1"])
        with pytest.raises(ValueError, match="Unknown/duplicate score"):
            audit_voice_clone.main()


@pytest.mark.parametrize("corruption", [None, "speaker", "missing_icl", "wrong_mode"])
def test_local_icl_only_audit(tmp_path, monkeypatch, corruption):
    inputs, run, model, row = setup_run(tmp_path)
    row["conditioning_verified"] = {"speaker_embedding": False, "icl_calls": 1}
    cfg_path = model / "settings-00.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(conditioning="icl_only", icl_only_ablation=True)
    cfg["sampling"]["x_vector_only_mode"] = False
    if corruption == "speaker":
        row["conditioning_verified"]["speaker_embedding"] = True
    elif corruption == "missing_icl":
        row["conditioning_verified"]["icl_calls"] = 0
    elif corruption == "wrong_mode":
        cfg["conditioning"] = "speaker_icl"
    cfg_path.write_text(json.dumps(cfg))
    (model / "inference-00.jsonl").write_text(json.dumps(row) + "\n")
    (model / "conditioning-verified-00.json").write_text(json.dumps({
        "model": "speaker_only", "batch_size": 1,
        "speaker_embedding": False, "icl_calls": 1}))
    monkeypatch.setattr(sys, "argv", ["audit", "--run-dir", str(run),
        "--manifest-dir", str(inputs), "--models", "speaker_only", "--num-shards", "1"])
    if corruption:
        with pytest.raises(ValueError):
            audit_voice_clone.main()
    else:
        audit_voice_clone.main()


@pytest.mark.parametrize("mode", ["speaker_only", "speaker_icl"])
def test_no_speaker_export_rejects_speaker_modes(tmp_path, monkeypatch, mode):
    from scripts import run_seed_tts

    model = tmp_path / "export"
    model.mkdir()
    (model / "export.json").write_text(json.dumps({"use_speaker_embedding": False}))
    monkeypatch.setattr(sys, "argv", ["pipeline", "--model-path", str(model),
        "--run-dir", str(tmp_path / "run"), "--modes", mode])
    with pytest.raises(ValueError, match="No-speaker checkpoints require"):
        run_seed_tts.main()


@pytest.mark.parametrize("corruption", [None, "seed", "tail_size", "icl_count"])
def test_batched_icl_audit_checks_seed_and_tail(tmp_path, monkeypatch, corruption):
    inputs, run, model, template = setup_run(tmp_path)
    manifest_path = inputs / (template["dataset"] + ".jsonl")
    sample = json.loads(manifest_path.read_text())
    raw = "".join(json.dumps({**sample, "index": i}) + "\n" for i in range(5))
    manifest_path.write_text(raw)
    (inputs / "manifest_metadata.json").write_text(json.dumps({template["dataset"]: {
        "count": 5, "sha256": hashlib.sha256(raw.encode()).hexdigest()}}))
    rows, scores = [], []
    for i in range(5):
        audio = model / template["dataset"] / f"{i:06d}.wav"
        sf.write(audio, np.zeros(2400), 24000, subtype="PCM_16")
        size = 4 if i < 4 else 1
        row = {**template, "index": i, "audio": str(audio), "seed": 42 if i < 4 else 46,
               "batch_size": size, "conditioning_verified": {"speaker_embedding": False, "icl_calls": size}}
        if corruption == "seed" and i == 1:
            row["seed"] = 43
        if corruption == "tail_size" and i == 4:
            row["batch_size"] = 4
        if corruption == "icl_count" and i == 1:
            row["conditioning_verified"]["icl_calls"] = 1
        rows.append(row)
        scores.append({"model": "speaker_only", "dataset": template["dataset"], "index": i,
                       "status": "ok", "score": {"pred": str(audio), "ref": sample["prompt_audio"],
                       "label_text": sample["text"], "wer%": 0, "simo": 0.75}})
    (model / "inference-00.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (model / "scores-seed-en.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scores))
    cfg_path = model / "settings-00.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(batch_size=4, conditioning="icl_only", icl_only_ablation=True)
    cfg["sampling"]["x_vector_only_mode"] = False
    cfg_path.write_text(json.dumps(cfg))
    (model / "conditioning-verified-00.json").write_text(json.dumps({
        "model": "speaker_only", "batch_size": 1, "speaker_embedding": False, "icl_calls": 1}))
    monkeypatch.setattr(sys, "argv", ["audit", "--run-dir", str(run), "--manifest-dir", str(inputs),
        "--models", "speaker_only", "--num-shards", "1", "--batch-size", "4"])
    if corruption:
        with pytest.raises(ValueError):
            audit_voice_clone.main()
    else:
        audit_voice_clone.main()
