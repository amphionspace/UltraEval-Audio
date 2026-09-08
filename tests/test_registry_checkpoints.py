"""Local-only regression tests for recursive registry discovery."""
from pathlib import Path
import tempfile
import unittest

from audio_evals.registry import Registry


class RegistryCheckpointTests(unittest.TestCase):
    def test_nested_configs_load_but_not_notebook_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model" / "nested"
            model.mkdir(parents=True)
            config = "voice-clone-test:\n  class: example.Model\n  args: {}\n"
            (model / "model.yaml").write_text(config)
            checkpoint = model / ".ipynb_checkpoints"
            checkpoint.mkdir()
            (checkpoint / "model-checkpoint.yaml").write_text(config)
            self.assertEqual(set(Registry([root])._model), {"voice-clone-test"})

    def test_genuine_duplicates_remain_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            config = "voice-clone-test:\n  class: example.Model\n  args: {}\n"
            (model / "first.yaml").write_text(config)
            (model / "second.yaml").write_text(config)
            with self.assertRaisesRegex(AssertionError, "duplicate entry"):
                Registry([root])._model


if __name__ == "__main__":
    unittest.main()
