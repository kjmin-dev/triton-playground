from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.prepare_models as prepare_models_module
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

    def test_stt_profile_manifest_keeps_whisper_manual_and_records_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = prepare_model_repository(
                output_root=Path(temp_dir),
                model_ids=["whisper_large_v3_turbo"],
            )

        self.assertEqual(len(manifest["models"]), 1)
        whisper = manifest["models"][0]
        self.assertEqual(whisper["model_id"], "whisper_large_v3_turbo")
        self.assertFalse(whisper["installed"])
        self.assertEqual(whisper["repository_model_name"], "whisper_large_v3_turbo")
        self.assertEqual(whisper["triton_backend"], "python")
        self.assertIn("audio_pcm: FP32[segments, padded_samples]", whisper["triton_inputs"][0])
        self.assertIn("manual download", whisper["reason"])

    def test_catalog_manifest_records_localization_contracts_without_installing_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = prepare_model_repository(
                output_root=Path(temp_dir),
                model_ids=["madlad400_3b_mt", "qwen3_tts_0_6b"],
            )

        records = {record["model_id"]: record for record in manifest["models"]}
        translation = records["madlad400_3b_mt"]
        tts = records["qwen3_tts_0_6b"]

        self.assertFalse(translation["installed"])
        self.assertEqual(translation["repository_model_name"], "madlad400_3b_mt")
        self.assertIn("translated_text: STRING[1] UTF-8 translated text", translation["triton_outputs"])

        self.assertFalse(tts["installed"])
        self.assertEqual(tts["repository_model_name"], "qwen3_tts_0_6b")
        self.assertIn("audio_pcm: FP32[segments, padded_samples]", tts["triton_outputs"][0])

    def test_manual_stub_root_writes_translation_and_tts_templates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            manifest = prepare_model_repository(
                output_root=temp_root / "repository",
                model_ids=["whisper_large_v3_turbo", "madlad400_3b_mt", "qwen3_tts_0_6b"],
                manual_stub_root=temp_root / "manual_model_stubs",
            )

            self.assertEqual(manifest["manual_stub_root"], str(temp_root / "manual_model_stubs"))
            records = {record["model_id"]: record for record in manifest["models"]}

            whisper = records["whisper_large_v3_turbo"]
            self.assertIn("manual_stub_path", whisper)
            self.assertTrue(any(path.endswith("config.pbtxt.template") for path in whisper["manual_stub_files"]))

            config_template = temp_root / "manual_model_stubs" / "qwen3_tts_0_6b" / "config.pbtxt.template"
            model_template = temp_root / "manual_model_stubs" / "qwen3_tts_0_6b" / "1" / "model.py.template"
            readme = temp_root / "manual_model_stubs" / "madlad400_3b_mt" / "README.md"

            self.assertTrue(config_template.exists())
            self.assertTrue(model_template.exists())
            self.assertTrue(readme.exists())
            self.assertIn('backend: "python"', config_template.read_text(encoding="utf-8"))
            self.assertIn("Manual Triton Python backend template", model_template.read_text(encoding="utf-8"))
            self.assertIn("Bring-up steps", readme.read_text(encoding="utf-8"))

    def test_materialize_manual_models_uses_runtime_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            def fake_materializer(output_root, spec, cache_dir):
                _ = cache_dir
                target_root = output_root / spec.repository_model_name
                version_root = target_root / "1"
                upstream_root = version_root / "upstream"
                version_root.mkdir(parents=True, exist_ok=True)
                upstream_root.mkdir(parents=True, exist_ok=True)
                (target_root / "config.pbtxt").write_text('backend: "python"\n', encoding="utf-8")
                (version_root / "model.py").write_text("# runtime\n", encoding="utf-8")
                return {
                    "installed": True,
                    "materialization_mode": "opt_in_manual_prepare",
                    "repository_path": str(version_root.relative_to(output_root)),
                    "runtime_files": [
                        str((target_root / "config.pbtxt").relative_to(output_root)),
                        str((version_root / "model.py").relative_to(output_root)),
                        str(upstream_root.relative_to(output_root)),
                    ],
                    "snapshot_allow_patterns": list(spec.snapshot_allow_patterns),
                }

            with patch("pipeline.prepare_models.materialize_manual_runtime_model", side_effect=fake_materializer):
                manifest = prepare_model_repository(
                    output_root=temp_root / "repository",
                    model_ids=["whisper_large_v3_turbo", "madlad400_3b_mt", "qwen3_tts_0_6b"],
                    materialize_manual_models=True,
                )

            records = {record["model_id"]: record for record in manifest["models"]}
            self.assertEqual(records["whisper_large_v3_turbo"]["materialization_mode"], "opt_in_manual_prepare")
            self.assertTrue(records["madlad400_3b_mt"]["installed"])
            self.assertIn("config.pbtxt", records["qwen3_tts_0_6b"]["runtime_files"][0])

    def test_materialize_manual_models_captures_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            captured: list[str] = []
            original_materializer = prepare_models_module.materialize_manual_runtime_model

            def fake_materializer(output_root, spec, cache_dir):
                _ = cache_dir
                captured.append(spec.model_id)
                runtime_dir = output_root / spec.repository_model_name / "1"
                runtime_dir.mkdir(parents=True, exist_ok=True)
                return {
                    "installed": True,
                    "materialization_mode": "opt_in_manual_prepare",
                    "repository_path": str(runtime_dir.relative_to(output_root)),
                }

            prepare_models_module.materialize_manual_runtime_model = fake_materializer
            try:
                manifest = prepare_model_repository(
                    output_root=temp_root / "repository",
                    model_ids=["whisper_large_v3_turbo", "madlad400_3b_mt"],
                    materialize_manual_models=True,
                )
            finally:
                prepare_models_module.materialize_manual_runtime_model = original_materializer

        self.assertEqual(captured, ["whisper_large_v3_turbo", "madlad400_3b_mt"])
        for record in manifest["models"]:
            self.assertTrue(record["installed"])
            self.assertEqual(record["materialization_mode"], "opt_in_manual_prepare")


if __name__ == "__main__":
    unittest.main()
