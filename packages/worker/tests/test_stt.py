from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.audio import AudioBuffer
from pipeline.stt import (
    SttAnalysis,
    TranscribedSegment,
    TritonUnavailableError,
    TritonWhisperClient,
    analyze_stt,
    normalize_whisper_language,
    validate_whisper_task,
)


class SttAnalysisTest(unittest.TestCase):
    def test_analyze_stt_runs_vad_first_and_concatenates_segment_text(self) -> None:
        audio = AudioBuffer(samples=np.ones(512 * 8, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:
                _ = windows
                return [0.9] * 8

        class FakeWhisperClient:
            def __init__(self) -> None:
                self.calls: list[tuple[tuple[int, ...], tuple[int | None, ...], str, str | None]] = []

            def transcribe_many(
                self,
                segment_audios: list[AudioBuffer],
                *,
                language: str | None,
                task: str,
                prompt: str | None = None,
            ) -> list[str]:
                _ = prompt
                self.calls.append(
                    (
                        tuple(len(segment_audio.samples) for segment_audio in segment_audios),
                        tuple(segment_audio.sample_rate for segment_audio in segment_audios),
                        task,
                        language,
                    )
                )
                return ["hello world" for _ in segment_audios]

        whisper = FakeWhisperClient()
        analysis = analyze_stt(
            audio=audio,
            vad_client=FakeVadClient(),
            stt_client=whisper,
            threshold=0.5,
            task="transcribe",
            language="en",
            min_speech_ms=32,
            min_silence_ms=32,
            pad_ms=0,
        )

        self.assertEqual(analysis.task, "transcribe")
        self.assertEqual(analysis.language, "en")
        self.assertEqual(analysis.transcript, "hello world")
        self.assertEqual(len(analysis.segments), 1)
        self.assertEqual(analysis.segments[0].text, "hello world")
        self.assertEqual(whisper.calls, [((4096,), (16000,), "transcribe", "en")])

    def test_analyze_stt_returns_empty_transcript_when_vad_finds_no_speech(self) -> None:
        audio = AudioBuffer(samples=np.zeros(512 * 4, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:
                _ = windows
                return [0.1] * 4

        class FakeWhisperClient:
            def transcribe(self, *args, **kwargs) -> str:  # pragma: no cover - should never be called
                raise TritonUnavailableError("transcribe should not be called without speech segments")

        analysis = analyze_stt(
            audio=audio,
            vad_client=FakeVadClient(),
            stt_client=FakeWhisperClient(),
            min_speech_ms=32,
            min_silence_ms=32,
            pad_ms=0,
        )

        self.assertEqual(analysis.transcript, "")
        self.assertEqual(analysis.segments, [])

    def test_analyze_stt_prefers_server_side_pipeline_when_available(self) -> None:
        audio = AudioBuffer(samples=np.ones(512 * 8, dtype=np.float32), sample_rate=16000)

        class FakeVadClient:
            def score_windows(self, windows: np.ndarray) -> list[float]:  # pragma: no cover - should not be called
                _ = windows
                raise AssertionError("worker-side VAD should be skipped when Triton STT pipeline is available")

        class FakePipelineClient:
            def analyze_audio(self, segment_audio: AudioBuffer, **kwargs) -> SttAnalysis:
                self.call = (len(segment_audio.samples), kwargs["threshold"], kwargs["task"], kwargs["language"])
                return SttAnalysis(
                    threshold=kwargs["threshold"],
                    task=kwargs["task"],
                    language=kwargs["language"] or "auto",
                    duration_ms=segment_audio.duration_ms,
                    sample_rate=segment_audio.sample_rate,
                    transcript="batched transcript",
                    segments=[
                        TranscribedSegment(
                            start_ms=0,
                            end_ms=segment_audio.duration_ms,
                            duration_ms=segment_audio.duration_ms,
                            average_probability=0.9,
                            peak_probability=0.9,
                            text="batched transcript",
                        )
                    ],
                )

        pipeline_client = FakePipelineClient()
        analysis = analyze_stt(
            audio=audio,
            vad_client=FakeVadClient(),
            stt_client=pipeline_client,
            threshold=0.6,
            task="transcribe",
            language="en",
        )

        self.assertEqual(analysis.transcript, "batched transcript")
        self.assertEqual(pipeline_client.call, (4096, 0.6, "transcribe", "en"))

    def test_language_and_task_validation_is_explicit(self) -> None:
        self.assertIsNone(normalize_whisper_language("auto"))
        self.assertEqual(normalize_whisper_language("ko"), "ko")
        self.assertEqual(validate_whisper_task("transcribe"), "transcribe")

        with self.assertRaisesRegex(ValueError, "language must be one of"):
            normalize_whisper_language("fr")

        with self.assertRaisesRegex(ValueError, "task must be one of"):
            validate_whisper_task("translate")

    def test_whisper_readiness_connection_refused_includes_startup_guidance(self) -> None:
        class FakeClient:
            def is_server_live(self) -> bool:
                raise RuntimeError(
                    "failed to connect to all addresses; last error: UNKNOWN: "
                    "ipv4:127.0.0.1:8001: Failed to connect to remote host: connect: Connection refused (111)"
                )

            def is_server_ready(self) -> bool:  # pragma: no cover - never reached after failure
                return False

            def is_model_ready(self, model_name: str) -> bool:  # pragma: no cover - never reached after failure
                _ = model_name
                return False

        client = object.__new__(TritonWhisperClient)
        client._grpcclient = object()
        client._url = "127.0.0.1:8001"
        client._model_name = "whisper_large_v3_turbo"
        client._client = FakeClient()

        with self.assertRaisesRegex(TritonUnavailableError, "docker compose up --build"):
            client.readiness()


if __name__ == "__main__":
    unittest.main()
