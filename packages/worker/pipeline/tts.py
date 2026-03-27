from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np

from pipeline.runtime_status import TritonReadiness
from pipeline.triton import check_readiness
from pipeline.translation import normalize_pipeline_language
from pipeline.triton import TritonUnavailableError
from pipeline.tts_contract import (
    DEFAULT_TTS_REPOSITORY_MODEL_NAME,
    TTS_AUDIO_OUTPUT,
    TTS_LANGUAGE_INPUT,
    TTS_SAMPLE_RATE_OUTPUT,
    TTS_SPEAKER_PROMPT_INPUT,
    TTS_TEXT_INPUT,
)


@dataclass(frozen=True)
class SynthesizedAudio:
    sample_rate: int
    samples: np.ndarray

    @property
    def duration_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return int(round(len(self.samples) * 1000 / self.sample_rate))




def encode_wav_preview(audio: SynthesizedAudio) -> str:
    clipped = np.clip(audio.samples.astype(np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm.tobytes())

    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TritonTtsClient:
    def __init__(
        self,
        url: str,
        model_name: str = DEFAULT_TTS_REPOSITORY_MODEL_NAME,
        *,
        text_input_name: str = TTS_TEXT_INPUT,
        language_input_name: str = TTS_LANGUAGE_INPUT,
        speaker_prompt_input_name: str = TTS_SPEAKER_PROMPT_INPUT,
        audio_output_name: str = TTS_AUDIO_OUTPUT,
        sample_rate_output_name: str = TTS_SAMPLE_RATE_OUTPUT,
    ) -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name
        self._text_input_name = text_input_name
        self._language_input_name = language_input_name
        self._speaker_prompt_input_name = speaker_prompt_input_name
        self._audio_output_name = audio_output_name
        self._sample_rate_output_name = sample_rate_output_name

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Failed to create Triton client for {url}: {exc}") from exc

    def readiness(self) -> TritonReadiness:
        return check_readiness(self._client, self._url, self._model_name)

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker_prompt: str | None = None,
    ) -> SynthesizedAudio:
        readiness = self.readiness()
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            text_input = self._grpcclient.InferInput(self._text_input_name, [1], "BYTES")
            text_input.set_data_from_numpy(np.asarray([text], dtype=object))

            language_input = self._grpcclient.InferInput(self._language_input_name, [1], "BYTES")
            language_input.set_data_from_numpy(np.asarray([language], dtype=object))

            speaker_prompt_input = self._grpcclient.InferInput(self._speaker_prompt_input_name, [1], "BYTES")
            speaker_prompt_input.set_data_from_numpy(np.asarray([speaker_prompt or ""], dtype=object))

            result = self._client.infer(
                self._model_name,
                [text_input, language_input, speaker_prompt_input],
                outputs=[
                    self._grpcclient.InferRequestedOutput(self._audio_output_name),
                    self._grpcclient.InferRequestedOutput(self._sample_rate_output_name),
                ],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"TTS inference request failed: {exc}") from exc

        audio_tensor = result.as_numpy(self._audio_output_name)
        sample_rate_tensor = result.as_numpy(self._sample_rate_output_name)

        if audio_tensor is None or sample_rate_tensor is None:
            raise TritonUnavailableError(
                "TTS Triton inference succeeded but did not return both audio_pcm and sample_rate."
            )

        flattened = audio_tensor.reshape(-1).astype(np.float32)
        if flattened.size == 0:
            raise TritonUnavailableError("TTS Triton inference returned an empty waveform.")

        sample_rate_value = int(np.asarray(sample_rate_tensor).reshape(-1)[0])
        if sample_rate_value <= 0:
            raise TritonUnavailableError(f"TTS Triton inference returned an invalid sample rate: {sample_rate_value}")

        return SynthesizedAudio(sample_rate=sample_rate_value, samples=flattened)


def validate_tts_language(language: str) -> str:
    normalized = normalize_pipeline_language(language, allow_auto=False)
    if normalized is None:
        raise ValueError("language must be explicitly set for TTS")
    return normalized
