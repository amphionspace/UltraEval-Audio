"""Check GPU routing and resume partition identity without loading models."""
import json
import sys

import pytest

from scripts import run_seed_tts


@pytest.fixture
def runner(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"test weights")
    (model / "export.json").write_text(json.dumps({
        "use_speaker_embedding": False,
        "weights_sha256": run_seed_tts.digest(model / "model.safetensors"),
    }))
    manifests = tmp_path / "raw_data/seed_tts_manifests"
    manifests.mkdir(parents=True)
    for language in ("en", "zh"):
        (manifests / f"{language}.jsonl").write_text('{}\n{}\n')
    monkeypatch.setattr(run_seed_tts, "ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(run_seed_tts, "parallel", lambda batch, *_: commands.extend(batch))
    monkeypatch.setattr(run_seed_tts.subprocess, "run", lambda *_, **__: None)

    def invoke(gpus="0,1,2,3,4,5,6,7", score_gpus="2,3,6,7"):
        monkeypatch.setattr(sys, "argv", ["runner", "--model-path", "model",
            "--run-dir", "run", "--modes", "icl_only", "--gpus", gpus,
            "--score-gpus", score_gpus, "--gpu-memory-gib", "4", "--gpu-reserve-gib", "0.5"])
        run_seed_tts.main()
        return commands

    return invoke


def test_eight_inference_workers_use_separate_scoring_devices(runner):
    commands = runner()
    inference = [(cmd, gpu) for label, cmd, gpu in commands if label.startswith("full-icl_only-")]
    assert [gpu for _, gpu in inference] == list("01234567")
    for shard, (cmd, _) in enumerate(inference):
        assert cmd[cmd.index("--shard") + 1] == str(shard)
        assert cmd[cmd.index("--num-shards") + 1] == "8"
        assert cmd[cmd.index("--gpu-reserve-gib") + 1] == "0.5"
    scoring = [(label, cmd, gpu) for label, cmd, gpu in commands if label.startswith("full-score-")]
    assert [(label, gpu) for label, _, gpu in scoring] == [
        ("full-score-en-00", "2"), ("full-score-en-01", "3"),
        ("full-score-zh-00", "6"), ("full-score-zh-01", "7")]
    assert all(cmd[cmd.index("--workers") + 1] == "2" for _, cmd, _ in scoring)


@pytest.mark.parametrize("gpus,score_gpus", [
    ("0,1,2", "2,3,6,7"),
    ("0,1,2,3,4,5,6,7", "2,3"),
])
def test_resume_rejects_changed_inference_or_score_partition(runner, gpus, score_gpus):
    runner()
    with pytest.raises(ValueError, match="Run identity changed"):
        runner(gpus, score_gpus)


def test_resume_allows_physical_gpu_reassignment(runner):
    runner()
    runner("7,6,5,4,3,2,1,0", "7,6,3,2")
