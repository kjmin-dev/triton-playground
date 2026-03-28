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
    TTS_SPEAKER_NAME_INPUT,
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
    speaker_name: str | None = None
    ref_audio: np.ndarray | None = None
    ref_audio_sample_rate: int = 16000
    ref_text: str | None = None


@dataclass(frozen=True)
class TtsActorPreset:
    actor_id: str
    label: str
    language: str
    speaker_name: str
    description: str
    preview_variants: tuple[TtsPreviewVariant, ...]
    is_default: bool = True

    @property
    def default_preview(self) -> TtsPreviewVariant:
        for preview in self.preview_variants:
            if preview.is_default:
                return preview
        if self.preview_variants:
            return self.preview_variants[0]
        raise ValueError(f"actor {self.actor_id!r} does not define any preview variants")

    def to_dict(self) -> dict[str, object]:
        default_preview = self.default_preview
        return {
            "actor_id": self.actor_id,
            "label": self.label,
            "language": self.language,
            "speaker_name": self.speaker_name,
            "description": self.description,
            "preview_text": default_preview.text,
            "preview_prompt": default_preview.prompt,
            "default_preview_id": default_preview.preview_id,
            "preview_variants": [preview.to_dict() for preview in self.preview_variants],
            "is_default": self.is_default,
        }


@dataclass(frozen=True)
class TtsPreviewVariant:
    preview_id: str
    label: str
    emotion: str
    prompt: str
    text: str
    is_default: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "preview_id": self.preview_id,
            "label": self.label,
            "emotion": self.emotion,
            "prompt": self.prompt,
            "text": self.text,
            "is_default": self.is_default,
        }


TTS_ACTOR_PRESETS = (
    TtsActorPreset(
        actor_id="ryan",
        label="Ryan",
        language="en",
        speaker_name="Ryan",
        description="English preset actor with a clear neutral delivery.",
        preview_variants=(
            TtsPreviewVariant(
                preview_id="neutral",
                label="Neutral",
                emotion="neutral",
                prompt="clear neutral narration",
                text="Hello. This is Triton Playground with a neutral voice preview.",
                is_default=True,
            ),
            TtsPreviewVariant(
                preview_id="warm",
                label="Warm",
                emotion="warm",
                prompt="warm reassuring delivery",
                text="Thanks for stopping by. This preview uses a warm and reassuring tone.",
            ),
            TtsPreviewVariant(
                preview_id="energetic",
                label="Energetic",
                emotion="energetic",
                prompt="bright energetic performance",
                text="We are live. This preview pushes a bright and energetic performance.",
            ),
        ),
    ),
    TtsActorPreset(
        actor_id="ono_anna",
        label="Ono Anna",
        language="ja",
        speaker_name="Ono_Anna",
        description="Japanese preset actor tuned for clean conversational lines.",
        preview_variants=(
            TtsPreviewVariant(
                preview_id="neutral",
                label="Neutral",
                emotion="neutral",
                prompt="clear neutral narration",
                text="こんにちは。Triton Playground の標準ボイスプレビューです。",
                is_default=True,
            ),
            TtsPreviewVariant(
                preview_id="warm",
                label="Warm",
                emotion="warm",
                prompt="warm reassuring delivery",
                text="ようこそ。やわらかく安心感のある雰囲気でお届けします。",
            ),
            TtsPreviewVariant(
                preview_id="energetic",
                label="Energetic",
                emotion="energetic",
                prompt="bright energetic performance",
                text="それでは始めましょう。明るく勢いのあるトーンでご案内します。",
            ),
        ),
    ),
    TtsActorPreset(
        actor_id="sohee",
        label="Sohee",
        language="ko",
        speaker_name="Sohee",
        description="Korean preset actor with a stable bright timbre.",
        preview_variants=(
            TtsPreviewVariant(
                preview_id="neutral",
                label="Neutral",
                emotion="neutral",
                prompt="clear neutral narration",
                text="안녕하세요. Triton Playground 기본 음성 미리듣기입니다.",
                is_default=True,
            ),
            TtsPreviewVariant(
                preview_id="warm",
                label="Warm",
                emotion="warm",
                prompt="warm reassuring delivery",
                text="반갑습니다. 차분하고 따뜻한 분위기로 안내드리겠습니다.",
            ),
            TtsPreviewVariant(
                preview_id="energetic",
                label="Energetic",
                emotion="energetic",
                prompt="bright energetic performance",
                text="지금 바로 시작합니다. 밝고 힘 있는 톤으로 들려드리겠습니다.",
            ),
        ),
    ),
    TtsActorPreset(
        actor_id="vivian",
        label="Vivian",
        language="zh",
        speaker_name="Vivian",
        description="Chinese preset actor for clear standard Mandarin output.",
        preview_variants=(
            TtsPreviewVariant(
                preview_id="neutral",
                label="Neutral",
                emotion="neutral",
                prompt="clear neutral narration",
                text="你好，这是 Triton Playground 的标准语音预览。",
                is_default=True,
            ),
            TtsPreviewVariant(
                preview_id="warm",
                label="Warm",
                emotion="warm",
                prompt="warm reassuring delivery",
                text="欢迎来到这里。我会用温和而安心的语气来演示这段声音。",
            ),
            TtsPreviewVariant(
                preview_id="energetic",
                label="Energetic",
                emotion="energetic",
                prompt="bright energetic performance",
                text="现在开始演示。这个版本会用更明亮、更有冲劲的表达方式。",
            ),
        ),
    ),
)


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
        speaker_name_input_name: str = TTS_SPEAKER_NAME_INPUT,
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
        self._speaker_name_input_name = speaker_name_input_name
        self._ref_audio_input_name = TTS_REF_AUDIO_INPUT
        self._ref_audio_lengths_input_name = TTS_REF_AUDIO_LENGTHS_INPUT
        self._ref_text_input_name = TTS_REF_TEXT_INPUT
        self._audio_output_name = audio_output_name
        self._audio_lengths_output_name = TTS_AUDIO_LENGTHS_OUTPUT
        self._sample_rate_output_name = sample_rate_output_name
        self._supports_speaker_name_input: bool | None = None
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
        speaker_name: str | None = None,
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
                    speaker_name=speaker_name,
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

        send_speaker_name = self._supports_speaker_name_input is not False
        try:
            result = self._infer_batch(
                requests,
                ref_audio_batch=ref_audio_batch,
                ref_audio_lengths=ref_audio_lengths,
                send_speaker_name=send_speaker_name,
            )
            if send_speaker_name:
                self._supports_speaker_name_input = True
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            if send_speaker_name and _is_stale_speaker_name_contract_error(exc):
                self._supports_speaker_name_input = False
                try:
                    result = self._infer_batch(
                        requests,
                        ref_audio_batch=ref_audio_batch,
                        ref_audio_lengths=ref_audio_lengths,
                        send_speaker_name=False,
                    )
                except Exception as retry_exc:
                    invalidate_fast_readiness_cache(self)
                    raise TritonUnavailableError(f"TTS inference request failed: {retry_exc}") from retry_exc
            else:
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

    def _infer_batch(
        self,
        requests: list[TtsSynthesisRequest],
        *,
        ref_audio_batch: np.ndarray,
        ref_audio_lengths: np.ndarray,
        send_speaker_name: bool,
    ):
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

        inputs = [
            text_input,
            language_input,
            speaker_prompt_input,
        ]

        if send_speaker_name:
            speaker_name_input = self._grpcclient.InferInput(
                self._speaker_name_input_name,
                [len(requests)],
                "BYTES",
            )
            speaker_name_input.set_data_from_numpy(
                np.asarray([request.speaker_name or "" for request in requests], dtype=object)
            )
            inputs.append(speaker_name_input)

        ref_audio_input = self._grpcclient.InferInput(
            self._ref_audio_input_name,
            list(ref_audio_batch.shape),
            "FP32",
        )
        ref_audio_input.set_data_from_numpy(ref_audio_batch)
        inputs.append(ref_audio_input)

        ref_audio_lengths_input = self._grpcclient.InferInput(
            self._ref_audio_lengths_input_name,
            [len(requests)],
            "INT32",
        )
        ref_audio_lengths_input.set_data_from_numpy(ref_audio_lengths)
        inputs.append(ref_audio_lengths_input)

        ref_text_input = self._grpcclient.InferInput(self._ref_text_input_name, [len(requests)], "BYTES")
        ref_text_input.set_data_from_numpy(np.asarray([request.ref_text or "" for request in requests], dtype=object))
        inputs.append(ref_text_input)

        return self._client.infer(
            self._model_name,
            inputs,
            outputs=[
                self._grpcclient.InferRequestedOutput(self._audio_output_name),
                self._grpcclient.InferRequestedOutput(self._audio_lengths_output_name),
                self._grpcclient.InferRequestedOutput(self._sample_rate_output_name),
            ],
        )


def _is_stale_speaker_name_contract_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "expected 6 inputs but got 7" in detail and "speaker_name" in detail


def validate_tts_language(language: str) -> str:
    normalized = normalize_pipeline_language(language, allow_auto=False)
    if normalized is None:
        raise ValueError("language must be explicitly set for TTS")
    return normalized


def list_tts_actor_presets(*, language: str | None = None) -> list[TtsActorPreset]:
    if language is None:
        return list(TTS_ACTOR_PRESETS)

    normalized_language = validate_tts_language(language)
    return [preset for preset in TTS_ACTOR_PRESETS if preset.language == normalized_language]


def get_default_tts_actor_preset(language: str) -> TtsActorPreset:
    normalized_language = validate_tts_language(language)
    for preset in TTS_ACTOR_PRESETS:
        if preset.language == normalized_language and preset.is_default:
            return preset
    raise ValueError(f"no preset actor is configured for language: {normalized_language}")


def resolve_tts_preview_variant(actor: TtsActorPreset, preview_id: str | None) -> TtsPreviewVariant:
    if preview_id is None or not preview_id.strip():
        return actor.default_preview

    for preview in actor.preview_variants:
        if preview.preview_id == preview_id:
            return preview

    raise ValueError(f"actor {actor.actor_id!r} does not define preview variant {preview_id!r}")


def resolve_tts_actor_preset(language: str, actor_id: str | None) -> TtsActorPreset:
    normalized_language = validate_tts_language(language)
    if actor_id is None or not actor_id.strip():
        return get_default_tts_actor_preset(normalized_language)

    for preset in TTS_ACTOR_PRESETS:
        if preset.actor_id == actor_id:
            if preset.language != normalized_language:
                raise ValueError(f"actor {actor_id!r} does not support language {normalized_language!r}")
            return preset

    raise ValueError(f"unknown TTS actor: {actor_id}")
