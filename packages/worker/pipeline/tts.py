from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass

import numpy as np

from pipeline.runtime_status import TritonReadiness
from pipeline.translation import normalize_pipeline_language
from pipeline.triton import (
    TritonUnavailableError,
    check_readiness,
    get_fast_readiness,
    init_fast_readiness_cache,
    invalidate_fast_readiness_cache,
)
from pipeline.tts_contract import (
    DEFAULT_TTS_REPOSITORY_MODEL_NAME,
    TTS_AUDIO_LENGTHS_OUTPUT,
    TTS_AUDIO_OUTPUT,
    TTS_LANGUAGE_INPUT,
    TTS_REF_AUDIO_INPUT,
    TTS_REF_AUDIO_LENGTHS_INPUT,
    TTS_REF_TEXT_INPUT,
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


@dataclass(frozen=True)
class TtsSynthesisRequest:
    text: str
    language: str
    speaker_prompt: str | None = None
    ref_audio: np.ndarray | None = None
    ref_audio_sample_rate: int = 16000
    ref_text: str | None = None


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
        self._ref_audio_input_name = TTS_REF_AUDIO_INPUT
        self._ref_audio_lengths_input_name = TTS_REF_AUDIO_LENGTHS_INPUT
        self._ref_text_input_name = TTS_REF_TEXT_INPUT
        self._audio_output_name = audio_output_name
        self._audio_lengths_output_name = TTS_AUDIO_LENGTHS_OUTPUT
        self._sample_rate_output_name = sample_rate_output_name
        init_fast_readiness_cache(self)

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Failed to create Triton client for {url}: {exc}") from exc

    def readiness(self, *, refresh: bool = False, detailed: bool = True) -> TritonReadiness:
        if detailed:
            return check_readiness(self._client, self._url, self._model_name)
        return get_fast_readiness(self, self._client, url=self._url, model_name=self._model_name, refresh=refresh)

    def synthesize(
        self,
        text: str,
        *,
        language: str,
        speaker_prompt: str | None = None,
        ref_audio: np.ndarray | None = None,
        ref_audio_sample_rate: int = 16000,
        ref_text: str | None = None,
    ) -> SynthesizedAudio:
        synthesized = self.synthesize_many(
            [
                TtsSynthesisRequest(
                    text=text,
                    language=language,
                    speaker_prompt=speaker_prompt,
                    ref_audio=ref_audio,
                    ref_audio_sample_rate=ref_audio_sample_rate,
                    ref_text=ref_text,
                )
            ]
        )
        return synthesized[0]

    def synthesize_many(self, requests: list[TtsSynthesisRequest]) -> list[SynthesizedAudio]:
        if not requests:
            return []

        readiness = self.readiness(detailed=False)
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        max_ref_samples = max(
            (int(request.ref_audio.size) if request.ref_audio is not None and request.ref_audio.size > 1 else 1)
            for request in requests
        )
        ref_audio_batch = np.zeros((len(requests), max_ref_samples), dtype=np.float32)
        ref_audio_lengths = np.ones(len(requests), dtype=np.int32)

        for index, request in enumerate(requests):
            if request.ref_audio is None or request.ref_audio.size <= 1:
                continue
            sample_count = int(request.ref_audio.size)
            ref_audio_batch[index, :sample_count] = request.ref_audio.astype(np.float32).reshape(-1)
            ref_audio_lengths[index] = sample_count

        try:
            text_input = self._grpcclient.InferInput(self._text_input_name, [len(requests)], "BYTES")
            text_input.set_data_from_numpy(np.asarray([request.text for request in requests], dtype=object))

            language_input = self._grpcclient.InferInput(self._language_input_name, [len(requests)], "BYTES")
            language_input.set_data_from_numpy(np.asarray([request.language for request in requests], dtype=object))

            speaker_prompt_input = self._grpcclient.InferInput(
                self._speaker_prompt_input_name,
                [len(requests)],
                "BYTES",
            )
            speaker_prompt_input.set_data_from_numpy(
                np.asarray([request.speaker_prompt or "" for request in requests], dtype=object)
            )

            ref_audio_input = self._grpcclient.InferInput(
                self._ref_audio_input_name,
                list(ref_audio_batch.shape),
                "FP32",
            )
            ref_audio_input.set_data_from_numpy(ref_audio_batch)

            ref_audio_lengths_input = self._grpcclient.InferInput(
                self._ref_audio_lengths_input_name,
                [len(requests)],
                "INT32",
            )
            ref_audio_lengths_input.set_data_from_numpy(ref_audio_lengths)

            ref_text_input = self._grpcclient.InferInput(self._ref_text_input_name, [len(requests)], "BYTES")
            ref_text_input.set_data_from_numpy(
                np.asarray([request.ref_text or "" for request in requests], dtype=object)
            )

            result = self._client.infer(
                self._model_name,
                [
                    text_input,
                    language_input,
                    speaker_prompt_input,
                    ref_audio_input,
                    ref_audio_lengths_input,
                    ref_text_input,
                ],
                outputs=[
                    self._grpcclient.InferRequestedOutput(self._audio_output_name),
                    self._grpcclient.InferRequestedOutput(self._audio_lengths_output_name),
                    self._grpcclient.InferRequestedOutput(self._sample_rate_output_name),
                ],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            invalidate_fast_readiness_cache(self)
            raise TritonUnavailableError(f"TTS inference request failed: {exc}") from exc

        audio_tensor = result.as_numpy(self._audio_output_name)
        audio_lengths_tensor = result.as_numpy(self._audio_lengths_output_name)
        sample_rate_tensor = result.as_numpy(self._sample_rate_output_name)

        if audio_tensor is None or audio_lengths_tensor is None or sample_rate_tensor is None:
            raise TritonUnavailableError(
                "TTS Triton inference succeeded but did not return audio_pcm, audio_lengths, and sample_rate."
            )

        audio_batch = np.asarray(audio_tensor, dtype=np.float32)
        if audio_batch.ndim == 1:
            audio_batch = audio_batch.reshape(1, -1)
        audio_lengths = np.asarray(audio_lengths_tensor, dtype=np.int32).reshape(-1)
        sample_rates = np.asarray(sample_rate_tensor, dtype=np.int32).reshape(-1)

        if len(audio_lengths) != len(requests) or len(sample_rates) != len(requests):
            raise TritonUnavailableError(
                f"TTS Triton inference returned mismatched batch metadata for {len(requests)} requests."
            )

        synthesized: list[SynthesizedAudio] = []
        for index in range(len(requests)):
            sample_count = int(audio_lengths[index])
            if sample_count <= 0:
                raise TritonUnavailableError(f"TTS Triton inference returned an empty waveform for segment {index}.")

            sample_rate_value = int(sample_rates[index])
            if sample_rate_value <= 0:
                raise TritonUnavailableError(
                    f"TTS Triton inference returned an invalid sample rate for segment {index}: {sample_rate_value}"
                )

            waveform = audio_batch[index].reshape(-1)[:sample_count].astype(np.float32)
            if waveform.size == 0:
                raise TritonUnavailableError(
                    f"TTS Triton inference returned an empty waveform after trimming for segment {index}."
                )

            synthesized.append(SynthesizedAudio(sample_rate=sample_rate_value, samples=waveform))

        return synthesized


def validate_tts_language(language: str) -> str:
    normalized = normalize_pipeline_language(language, allow_auto=False)
    if normalized is None:
        raise ValueError("language must be explicitly set for TTS")
    return normalized
