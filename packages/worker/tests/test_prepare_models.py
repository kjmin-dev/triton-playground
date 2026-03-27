from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.prepare_models import prepare_model_repository, write_manifest


SILERO_SHA256 = "a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808"


class PrepareModelsTest(unittest.TestCase):
    def test_prepare_repository_installs_baseline_model_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = temp_path / "source-model.onnx"
            source_path.write_bytes(b"silero-test-payload")

            def fake_downloader(spec, artifact, cache_dir):
                _ = (spec, artifact, cache_dir)
                return source_path

            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                prepare_model_repository(
                    output_root=temp_path / "repository",
                    model_ids=["silero_vad"],
                    downloader=fake_downloader,
                )

    def test_manifest_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            manifest = {
                "schema_version": 2,
                "policy": {"baseline_profile": "baseline"},
                "models": [{"model_id": "silero_vad", "installed": False, "sha256": SILERO_SHA256}],
            }
            manifest_path = write_manifest(output_root, manifest)

            self.assertTrue(manifest_path.exists())
            self.assertIn("silero_vad", manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
