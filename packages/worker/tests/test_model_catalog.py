from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.model_catalog import get_profile_model_ids, list_model_specs


class ModelCatalogTest(unittest.TestCase):
    def test_baseline_profile_only_contains_auto_download_models(self) -> None:
        baseline_ids = get_profile_model_ids("baseline")
        stt_ids = get_profile_model_ids("stt")
        localize_ids = get_profile_model_ids("localize")
        catalog_ids = get_profile_model_ids("catalog")
        specs = {spec.model_id: spec for spec in list_model_specs()}

        self.assertEqual(baseline_ids, ("silero_vad",))
        self.assertEqual(stt_ids, ("silero_vad", "whisper_large_v3_turbo"))
        self.assertEqual(
            localize_ids,
            ("silero_vad", "whisper_large_v3_turbo", "madlad400_3b_mt", "qwen3_tts_0_6b"),
        )
        self.assertIn("bs_roformer", catalog_ids)
        for model_id in baseline_ids:
            self.assertTrue(specs[model_id].approved_for_auto_download)

    def test_hold_models_are_not_auto_downloaded(self) -> None:
        specs = {spec.model_id: spec for spec in list_model_specs()}
        self.assertFalse(specs["bs_roformer"].approved_for_auto_download)
        self.assertEqual(specs["bs_roformer"].serve_status, "hold")
        self.assertTrue(specs["bs_roformer"].next_action.startswith("Pin the exact redistributable weight source"))

    def test_whisper_declares_manual_triton_contract(self) -> None:
        specs = {spec.model_id: spec for spec in list_model_specs()}
        whisper = specs["whisper_large_v3_turbo"]

        self.assertFalse(whisper.approved_for_auto_download)
        self.assertEqual(whisper.repository_model_name, "whisper_large_v3_turbo")
        self.assertEqual(whisper.triton_backend, "python")
        self.assertTrue(any(contract.startswith("audio_pcm: FP32") for contract in whisper.triton_inputs))
        self.assertEqual(whisper.triton_outputs, ("transcript: STRING[1] UTF-8 transcript for the supplied segment",))

    def test_localization_pair_declares_manual_triton_contracts(self) -> None:
        specs = {spec.model_id: spec for spec in list_model_specs()}
        translation = specs["madlad400_3b_mt"]
        tts = specs["qwen3_tts_0_6b"]

        self.assertEqual(translation.repository_model_name, "madlad400_3b_mt")
        self.assertEqual(translation.triton_backend, "python")
        self.assertIn("translated_text: STRING[1] UTF-8 translated text", translation.triton_outputs)
        self.assertTrue(translation.snapshot_allow_patterns)
        self.assertEqual(translation.runtime_bundle, "localize-runtime")
        self.assertIn("transformers>=4.55", translation.runtime_pip_packages)

        self.assertEqual(tts.repository_model_name, "qwen3_tts_0_6b")
        self.assertEqual(tts.triton_backend, "python")
        self.assertTrue(any(contract.startswith("audio_pcm: FP32[1, samples]") for contract in tts.triton_outputs))
        self.assertEqual(tts.hf_repo_id, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.assertEqual(tts.runtime_bundle, "localize-runtime")
        self.assertEqual(tts.revision, "5d83992436eae1d760afd27aff78a71d676296fc")
        self.assertIn("qwen-tts>=0.1.0", tts.runtime_pip_packages)
        self.assertIn("voice cloning", tts.notes)
        self.assertTrue(any("ref_audio" in spec for spec in tts.triton_inputs))
        self.assertTrue(any("ref_text" in spec for spec in tts.triton_inputs))


if __name__ == "__main__":
    unittest.main()
