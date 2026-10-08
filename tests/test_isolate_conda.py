"""Conda model workers must start without a venv activation script."""
import sys

from audio_evals import isolate


def test_conda_prefix_without_activate(tmp_path, monkeypatch):
    prefix = tmp_path / "conda"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin/python").symlink_to(sys.executable)
    probe = tmp_path / "probe.py"
    probe.write_text("print('ready', flush=True)\ninput()\n")
    monkeypatch.setattr(isolate, "ensure_env", lambda *a, **kw: None)

    @isolate.isolated(str(probe))
    class Model:
        def __init__(self):
            self.command_args = {}

    model = Model(env_path=str(prefix), requirements_path="unused")
    try:
        stdout, stderr = model.process.communicate("\n", timeout=15)
        assert model.process.returncode == 0, stderr
        assert stdout.strip() == "ready"
    finally:
        if model.process.poll() is None:
            model.process.kill()
            model.process.wait()
