from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.audio import AudioBuffer
from pipeline.runtime_status import TritonReadiness
from pipeline.stt_contract import (
    DEFAULT_WHISPER_REPOSITORY_MODEL_NAME,
    SUPPORTED_WHISPER_LANGUAGES,
    SUPPORTED_WHISPER_TASKS,
    WHISPER_AUDIO_INPUT,
    WHISPER_LANGUAGE_INPUT,
    WHISPER_PROMPT_INPUT,
    WHISPER_SAMPLE_RATE_INPUT,
    WHISPER_TASK_INPUT,
    WHISPER_TRANSCRIPT_OUTPUT,
)
from pipeline.triton import TritonUnavailableError, describe_triton_error
from pipeline.vad import analyze_vad


@dataclass(frozen=True)
class TranscribedSegment:
    start_ms: int
    end_ms: int
    duration_ms: int
    average_probability: float
    peak_probability: float
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "average_probability": round(self.average_probability, 4),
            "peak_probability": round(self.peak_probability, 4),
            "text": self.text,
        }


@dataclass(frozen=True)
class SttAnalysis:
    threshold: float
    task: str
    language: str
    duration_ms: int
    sample_rate: int
    transcript: str
    segments: list[TranscribedSegment]


def normalize_whisper_language(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return None

    if normalized not in SUPPORTED_WHISPER_LANGUAGES:
        supported = ", ".join(("auto", *SUPPORTED_WHISPER_LANGUAGES))
        raise ValueError(f"language must be one of: {supported}")

    return normalized


def validate_whisper_task(task: str) -> str:
    normalized = task.strip().lower()
    if normalized not in SUPPORTED_WHISPER_TASKS:
        supported = ", ".join(SUPPORTED_WHISPER_TASKS)
        raise ValueError(f"task must be one of: {supported}")

    return normalized


def _model_index_entry_name(entry: object) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("name")
        return str(value) if value is not None else None

    value = getattr(entry, "name", None)
    return str(value) if value is not None else None


def _model_index_entry_state(entry: object) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("state")
        return str(value) if value is not None else None

    value = getattr(entry, "state", None)
    return str(value) if value is not None else None


def _slice_audio(audio: AudioBuffer, start_ms: int, end_ms: int) -> AudioBuffer:
    total_samples = len(audio.samples)
    start_sample = max(0, min(total_samples, round(start_ms * audio.sample_rate / 1000)))
    end_sample = max(start_sample, min(total_samples, round(end_ms * audio.sample_rate / 1000)))

    if end_sample == start_sample:
        end_sample = min(total_samples, start_sample + 1)

    return AudioBuffer(samples=audio.samples[start_sample:end_sample].astype(np.float32), sample_rate=audio.sample_rate)


def _decode_transcript_output(transcript_tensor: np.ndarray | None) -> str:
    if transcript_tensor is None:
        raise TritonUnavailableError(
            "Whisper Triton inference succeeded but did not return the configured transcript output tensor."
        )

    flattened = transcript_tensor.reshape(-1)
    if flattened.size == 0:
        return ""

    fragments: list[str] = []
    for item in flattened:
        scalar = item.item() if hasattr(item, "item") else item
        if isinstance(scalar, bytes):
            fragments.append(scalar.decode("utf-8"))
        elif isinstance(scalar, str):
            fragments.append(scalar)
        else:
            fragments.append(str(scalar))

    return " ".join(fragment.strip() for fragment in fragments if fragment.strip())


class TritonWhisperClient:
    def __init__(
        self,
        url: str,
        model_name: str = DEFAULT_WHISPER_REPOSITORY_MODEL_NAME,
        *,
        audio_input_name: str = WHISPER_AUDIO_INPUT,
        sample_rate_input_name: str = WHISPER_SAMPLE_RATE_INPUT,
        task_input_name: str = WHISPER_TASK_INPUT,
        language_input_name: str = WHISPER_LANGUAGE_INPUT,
        prompt_input_name: str = WHISPER_PROMPT_INPUT,
        transcript_output_name: str = WHISPER_TRANSCRIPT_OUTPUT,
    ) -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name
        self._audio_input_name = audio_input_name
        self._sample_rate_input_name = sample_rate_input_name
        self._task_input_name = task_input_name
        self._language_input_name = language_input_name
        self._prompt_input_name = prompt_input_name
        self._transcript_output_name = transcript_output_name

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self) -> TritonReadiness:
        try:
            model_present: bool | None = None
            model_state: str | None = None
            get_model_repository_index = getattr(self._client, "get_model_repository_index", None)
            if callable(get_model_repository_index):
                try:
                    model_index = get_model_repository_index()
                except Exception:
                    model_index = None

                if model_index is not None:
                    model_present = False
                    for entry in model_index:
                        if _model_index_entry_name(entry) == self._model_name:
                            model_present = True
                            model_state = _model_index_entry_state(entry)
                            break

            return TritonReadiness.from_status(
                server_url=self._url,
                server_ready=bool(self._client.is_server_ready()),
                server_live=bool(self._client.is_server_live()),
                model_ready=bool(self._client.is_model_ready(self._model_name)),
                model_present=model_present,
                model_state=model_state,
                model_name=self._model_name,
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Querying Triton readiness", exc=exc)
            ) from exc

    def transcribe(
        self,
        audio: AudioBuffer,
        *,
        language: str | None,
        task: str,
        prompt: str | None = None,
    ) -> str:
        readiness = self.readiness()
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            audio_input = self._grpcclient.InferInput(
                self._audio_input_name,
                [1, len(audio.samples)],
                "FP32",
            )
            audio_input.set_data_from_numpy(audio.samples.reshape(1, -1).astype(np.float32))

            sample_rate_input = self._grpcclient.InferInput(self._sample_rate_input_name, [1], "INT32")
            sample_rate_input.set_data_from_numpy(np.asarray([audio.sample_rate], dtype=np.int32))

            task_input = self._grpcclient.InferInput(self._task_input_name, [1], "BYTES")
            task_input.set_data_from_numpy(np.asarray([task], dtype=object))

            language_input = self._grpcclient.InferInput(self._language_input_name, [1], "BYTES")
            language_input.set_data_from_numpy(np.asarray([language or ""], dtype=object))

            prompt_input = self._grpcclient.InferInput(self._prompt_input_name, [1], "BYTES")
            prompt_input.set_data_from_numpy(np.asarray([prompt or ""], dtype=object))

            result = self._client.infer(
                self._model_name,
                [
                    audio_input,
                    sample_rate_input,
                    task_input,
                    language_input,
                    prompt_input,
                ],
                outputs=[self._grpcclient.InferRequestedOutput(self._transcript_output_name)],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running Whisper inference", exc=exc)
            ) from exc

        return _decode_transcript_output(result.as_numpy(self._transcript_output_name))


def analyze_stt(
    audio: AudioBuffer,
    *,
    vad_client,
    stt_client: TritonWhisperClient,
    threshold: float = 0.5,
    task: str = "transcribe",
    language: str | None = None,
    prompt: str | None = None,
    min_speech_ms: int = 160,
    min_silence_ms: int = 240,
    pad_ms: int = 80,
    window_samples: int = 512,
) -> SttAnalysis:
    normalized_language = normalize_whisper_language(language)
    normalized_task = validate_whisper_task(task)

    vad_analysis = analyze_vad(
        audio=audio,
        client=vad_client,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        pad_ms=pad_ms,
        window_samples=window_samples,
    )

    segments: list[TranscribedSegment] = []
    for segment in vad_analysis.segments:
        text = stt_client.transcribe(
            _slice_audio(audio, segment.start_ms, segment.end_ms),
            language=normalized_language,
            task=normalized_task,
            prompt=prompt,
        ).strip()
        segments.append(
            TranscribedSegment(
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                duration_ms=segment.duration_ms,
                average_probability=segment.average_probability,
                peak_probability=segment.peak_probability,
                text=text,
            )
        )

    transcript = " ".join(segment.text for segment in segments if segment.text).strip()
    return SttAnalysis(
        threshold=threshold,
        task=normalized_task,
        language=normalized_language or "auto",
        duration_ms=audio.duration_ms,
        sample_rate=audio.sample_rate,
        transcript=transcript,
        segments=segments,
    )
