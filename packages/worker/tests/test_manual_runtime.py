from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.manual_runtime import materialize_manual_runtime_model
from pipeline.model_catalog import get_model_spec


class ManualRuntimeTest(unittest.TestCase):
    def test_materialize_manual_runtime_model_writes_runtime_files(self) -> None:
        spec = get_model_spec("qwen3_tts_0_6b")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "repository"

            def fake_snapshot_download(model_spec, target_dir, cache_dir):
                _ = (model_spec, cache_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                (target_dir / "config.json").write_text("{}", encoding="utf-8")
                (target_dir / "model.safetensors").write_text("fake", encoding="utf-8")

            with patch("pipeline.manual_runtime._snapshot_download", side_effect=fake_snapshot_download):
                record = materialize_manual_runtime_model(output_root=output_root, spec=spec, cache_dir=None)

            self.assertTrue(record["installed"])
            self.assertEqual(record["materialization_mode"], "opt_in_manual_prepare")
            self.assertIn("qwen-tts>=0.1.0", record["runtime_pip_packages"])

            config_path = output_root / "qwen3_tts_0_6b" / "config.pbtxt"
            model_path = output_root / "qwen3_tts_0_6b" / "1" / "model.py"
            requirements_path = output_root / "qwen3_tts_0_6b" / "requirements.txt"
            upstream_dir = output_root / "qwen3_tts_0_6b" / "1" / "upstream"

            self.assertTrue(config_path.exists())
            self.assertTrue(model_path.exists())
            self.assertTrue(requirements_path.exists())
            self.assertTrue(upstream_dir.is_dir())
            self.assertIn('name: "qwen3_tts_0_6b"', config_path.read_text(encoding="utf-8"))
            self.assertIn("Qwen3TTSModel", model_path.read_text(encoding="utf-8"))
            self.assertIn("qwen-tts>=0.1.0", requirements_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
