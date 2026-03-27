from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.audio import AudioBuffer
from pipeline.localization import LocalizationStageError, localize_audio
from pipeline.tts import SynthesizedAudio


class LocalizationPipelineTest(unittest.TestCase):
    def test_localize_audio_runs_stt_translation_and_tts(self) -> None:
        audio = AudioBuffer(samples=np.ones(512 * 8, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:
                _ = windows
                return [0.9] * 8

        class FakeSttClient:
            def transcribe(self, *args, **kwargs) -> str:
                _ = (args, kwargs)
                return "hello world"

        class FakeTranslationClient:
            def translate(self, text: str, *, source_language: str | None, target_language: str) -> str:
                self.call = (text, source_language, target_language)
                return "annyeong haseyo"

        class FakeTtsClient:
            def synthesize(self, text: str, *, language: str, speaker_prompt: str | None = None) -> SynthesizedAudio:
                self.call = (text, language, speaker_prompt)
                return SynthesizedAudio(sample_rate=24000, samples=np.linspace(-0.2, 0.2, num=480, dtype=np.float32))

        translation = FakeTranslationClient()
        tts = FakeTtsClient()

        payload = localize_audio(
            audio=audio,
            threshold=0.5,
            source_language="en",
            target_language="ko",
            prompt=None,
            speaker_prompt="warm",
            stt_model="whisper_large_v3_turbo",
            translation_model="madlad400_3b_mt",
            tts_model="qwen3_tts_0_6b",
            vad_client=FakeVadClient(),
            stt_client=FakeSttClient(),
            translation_client=translation,
            tts_client=tts,
        )

        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["transcript"], "hello world")
        self.assertEqual(payload["translated_text"], "annyeong haseyo")
        self.assertEqual(payload["stages"]["translation"]["status"], "ok")
        self.assertEqual(payload["stages"]["tts"]["status"], "ok")
        self.assertEqual(translation.call, ("hello world", "en", "ko"))
        self.assertEqual(tts.call, ("annyeong haseyo", "ko", "warm"))
        self.assertTrue(payload["stages"]["tts"]["audio_base64"])

    def test_localize_audio_skips_downstream_stages_when_no_speech_is_detected(self) -> None:
        audio = AudioBuffer(samples=np.zeros(512 * 4, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:
                _ = windows
                return [0.1] * 4

        class FakeSttClient:
            def transcribe(self, *args, **kwargs) -> str:  # pragma: no cover - should never be called
                raise AssertionError("transcribe should not run when no speech segments are found")

        class FakeTranslationClient:
            def translate(self, *args, **kwargs) -> str:  # pragma: no cover - should never be called
                raise AssertionError("translate should not run when transcript is empty")

        class FakeTtsClient:
            def synthesize(self, *args, **kwargs) -> SynthesizedAudio:  # pragma: no cover - should never be called
                raise AssertionError("tts should not run when translation is skipped")

        payload = localize_audio(
            audio=audio,
            threshold=0.5,
            source_language="auto",
            target_language="ja",
            prompt=None,
            speaker_prompt=None,
            stt_model="whisper_large_v3_turbo",
            translation_model="madlad400_3b_mt",
            tts_model="qwen3_tts_0_6b",
            vad_client=FakeVadClient(),
            stt_client=FakeSttClient(),
            translation_client=FakeTranslationClient(),
            tts_client=FakeTtsClient(),
        )

        self.assertEqual(payload["transcript"], "")
        self.assertEqual(payload["stages"]["translation"]["status"], "skipped")
        self.assertEqual(payload["stages"]["tts"]["status"], "skipped")

    def test_localize_audio_reports_translation_stage_failures_with_partial_payload(self) -> None:
        audio = AudioBuffer(samples=np.ones(512 * 8, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:
                _ = windows
                return [0.9] * 8

        class FakeSttClient:
            def transcribe(self, *args, **kwargs) -> str:
                _ = (args, kwargs)
                return "hello world"

        class FakeTranslationClient:
            def translate(self, *args, **kwargs) -> str:
                raise RuntimeError("translation backend offline")

        class FakeTtsClient:
            def synthesize(self, *args, **kwargs) -> SynthesizedAudio:  # pragma: no cover - should never be called
                raise AssertionError("tts should not run when translation fails")

        with self.assertRaises(LocalizationStageError) as context:
            localize_audio(
                audio=audio,
                threshold=0.5,
                source_language="en",
                target_language="zh",
                prompt=None,
                speaker_prompt=None,
                stt_model="whisper_large_v3_turbo",
                translation_model="madlad400_3b_mt",
                tts_model="qwen3_tts_0_6b",
                vad_client=FakeVadClient(),
                stt_client=FakeSttClient(),
                translation_client=FakeTranslationClient(),
                tts_client=FakeTtsClient(),
            )

        self.assertEqual(context.exception.stage, "translation")
        self.assertEqual(context.exception.payload["stages"]["stt"]["status"], "ok")
        self.assertEqual(context.exception.payload["stages"]["translation"]["status"], "error")
        self.assertEqual(context.exception.payload["stages"]["tts"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
