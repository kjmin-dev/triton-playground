from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from pipeline.audio import AudioBuffer
from pipeline.runtime_status import TritonReadiness
from pipeline.stt_contract import (
    DEFAULT_WHISPER_REPOSITORY_MODEL_NAME,
    DEFAULT_WHISPER_STT_PIPELINE_REPOSITORY_MODEL_NAME,
    SUPPORTED_WHISPER_LANGUAGES,
    SUPPORTED_WHISPER_TASKS,
    WHISPER_AUDIO_INPUT,
    WHISPER_AUDIO_LENGTHS_INPUT,
    WHISPER_LANGUAGE_INPUT,
    WHISPER_MIN_SILENCE_MS_INPUT,
    WHISPER_MIN_SPEECH_MS_INPUT,
    WHISPER_PAD_MS_INPUT,
    WHISPER_PROMPT_INPUT,
    WHISPER_SAMPLE_RATE_INPUT,
    WHISPER_SEGMENTS_JSON_OUTPUT,
    WHISPER_TASK_INPUT,
    WHISPER_THRESHOLD_INPUT,
    WHISPER_TRANSCRIPT_OUTPUT,
    WHISPER_WINDOW_SAMPLES_INPUT,
)
from pipeline.triton import (
    TritonUnavailableError,
    check_readiness,
    describe_triton_error,
    get_fast_readiness,
    init_fast_readiness_cache,
    invalidate_fast_readiness_cache,
)
from pipeline.vad import analyze_vad


@dataclass(frozen=True)
class TranscribedSegment:
    start_ms: int
    end_ms: int
    duration_ms: int
    average_probability: float
    peak_probability: float
    text: str
    speaker_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "average_probability": round(self.average_probability, 4),
            "peak_probability": round(self.peak_probability, 4),
            "text": self.text,
        }
        if self.speaker_id is not None:
            result["speaker_id"] = self.speaker_id
        return result


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


def _slice_audio(audio: AudioBuffer, start_ms: int, end_ms: int) -> AudioBuffer:
    total_samples = len(audio.samples)
    start_sample = max(0, min(total_samples, round(start_ms * audio.sample_rate / 1000)))
    end_sample = max(start_sample, min(total_samples, round(end_ms * audio.sample_rate / 1000)))

    if end_sample == start_sample:
        end_sample = min(total_samples, start_sample + 1)

    return AudioBuffer(samples=audio.samples[start_sample:end_sample].astype(np.float32), sample_rate=audio.sample_rate)


def _decode_transcript_output(transcript_tensor: np.ndarray | None) -> str:
    outputs = _decode_transcript_outputs(transcript_tensor)
    return outputs[0] if outputs else ""


def _decode_transcript_outputs(transcript_tensor: np.ndarray | None) -> list[str]:
    if transcript_tensor is None:
        raise TritonUnavailableError(
            "Whisper Triton inference succeeded but did not return the configured transcript output tensor."
        )

    flattened = transcript_tensor.reshape(-1)
    if flattened.size == 0:
        return []

    fragments: list[str] = []
    for item in flattened:
        scalar = item.item() if hasattr(item, "item") else item
        if isinstance(scalar, bytes):
            fragments.append(scalar.decode("utf-8"))
        elif isinstance(scalar, str):
            fragments.append(scalar)
        else:
            fragments.append(str(scalar))

    return [fragment.strip() for fragment in fragments]


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
        self._audio_lengths_input_name = WHISPER_AUDIO_LENGTHS_INPUT
        self._sample_rate_input_name = sample_rate_input_name
        self._task_input_name = task_input_name
        self._language_input_name = language_input_name
        self._prompt_input_name = prompt_input_name
        self._transcript_output_name = transcript_output_name
        init_fast_readiness_cache(self)

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self, *, refresh: bool = False, detailed: bool = True) -> TritonReadiness:
        if detailed:
            return check_readiness(self._client, self._url, self._model_name)
        return get_fast_readiness(self, self._client, url=self._url, model_name=self._model_name, refresh=refresh)

    def transcribe(
        self,
        audio: AudioBuffer,
        *,
        language: str | None,
        task: str,
        prompt: str | None = None,
    ) -> str:
        transcripts = self.transcribe_many(
            [audio],
            language=language,
            task=task,
            prompt=prompt,
        )
        return transcripts[0] if transcripts else ""

    def transcribe_many(
        self,
        audio_segments: list[AudioBuffer],
        *,
        language: str | None,
        task: str,
        prompt: str | None = None,
    ) -> list[str]:
        if not audio_segments:
            return []

        readiness = self.readiness(detailed=False)
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        max_samples = max(len(audio.samples) for audio in audio_segments)
        audio_batch = np.zeros((len(audio_segments), max_samples), dtype=np.float32)
        audio_lengths = np.zeros(len(audio_segments), dtype=np.int32)
        sample_rates = np.zeros(len(audio_segments), dtype=np.int32)

        for index, segment_audio in enumerate(audio_segments):
            sample_count = len(segment_audio.samples)
            audio_batch[index, :sample_count] = segment_audio.samples.astype(np.float32)
            audio_lengths[index] = sample_count
            sample_rates[index] = segment_audio.sample_rate

        repeated_task = np.asarray([task] * len(audio_segments), dtype=object)
        repeated_language = np.asarray([(language or "")] * len(audio_segments), dtype=object)
        repeated_prompt = np.asarray([(prompt or "")] * len(audio_segments), dtype=object)

        try:
            audio_input = self._grpcclient.InferInput(
                self._audio_input_name,
                list(audio_batch.shape),
                "FP32",
            )
            audio_input.set_data_from_numpy(audio_batch)

            audio_lengths_input = self._grpcclient.InferInput(
                self._audio_lengths_input_name,
                [len(audio_segments)],
                "INT32",
            )
            audio_lengths_input.set_data_from_numpy(audio_lengths)

            sample_rate_input = self._grpcclient.InferInput(
                self._sample_rate_input_name,
                [len(audio_segments)],
                "INT32",
            )
            sample_rate_input.set_data_from_numpy(sample_rates)

            task_input = self._grpcclient.InferInput(self._task_input_name, [len(audio_segments)], "BYTES")
            task_input.set_data_from_numpy(repeated_task)

            language_input = self._grpcclient.InferInput(self._language_input_name, [len(audio_segments)], "BYTES")
            language_input.set_data_from_numpy(repeated_language)

            prompt_input = self._grpcclient.InferInput(self._prompt_input_name, [len(audio_segments)], "BYTES")
            prompt_input.set_data_from_numpy(repeated_prompt)

            result = self._client.infer(
                self._model_name,
                [
                    audio_input,
                    audio_lengths_input,
                    sample_rate_input,
                    task_input,
                    language_input,
                    prompt_input,
                ],
                outputs=[self._grpcclient.InferRequestedOutput(self._transcript_output_name)],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            invalidate_fast_readiness_cache(self)
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running Whisper inference", exc=exc)
            ) from exc

        transcripts = _decode_transcript_outputs(result.as_numpy(self._transcript_output_name))
        if len(transcripts) != len(audio_segments):
            raise TritonUnavailableError(
                f"Whisper Triton inference returned {len(transcripts)} transcripts for {len(audio_segments)} segments."
            )
        return transcripts


class TritonWhisperSttPipelineClient:
    def __init__(
        self,
        url: str,
        model_name: str = DEFAULT_WHISPER_STT_PIPELINE_REPOSITORY_MODEL_NAME,
        *,
        audio_input_name: str = WHISPER_AUDIO_INPUT,
        sample_rate_input_name: str = WHISPER_SAMPLE_RATE_INPUT,
        threshold_input_name: str = WHISPER_THRESHOLD_INPUT,
        min_speech_ms_input_name: str = WHISPER_MIN_SPEECH_MS_INPUT,
        min_silence_ms_input_name: str = WHISPER_MIN_SILENCE_MS_INPUT,
        pad_ms_input_name: str = WHISPER_PAD_MS_INPUT,
        window_samples_input_name: str = WHISPER_WINDOW_SAMPLES_INPUT,
        task_input_name: str = WHISPER_TASK_INPUT,
        language_input_name: str = WHISPER_LANGUAGE_INPUT,
        prompt_input_name: str = WHISPER_PROMPT_INPUT,
        transcript_output_name: str = WHISPER_TRANSCRIPT_OUTPUT,
        segments_output_name: str = WHISPER_SEGMENTS_JSON_OUTPUT,
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
        self._threshold_input_name = threshold_input_name
        self._min_speech_ms_input_name = min_speech_ms_input_name
        self._min_silence_ms_input_name = min_silence_ms_input_name
        self._pad_ms_input_name = pad_ms_input_name
        self._window_samples_input_name = window_samples_input_name
        self._task_input_name = task_input_name
        self._language_input_name = language_input_name
        self._prompt_input_name = prompt_input_name
        self._transcript_output_name = transcript_output_name
        self._segments_output_name = segments_output_name
        init_fast_readiness_cache(self)

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self, *, refresh: bool = False, detailed: bool = True) -> TritonReadiness:
        if detailed:
            return check_readiness(self._client, self._url, self._model_name)
        return get_fast_readiness(self, self._client, url=self._url, model_name=self._model_name, refresh=refresh)

    def analyze_audio(
        self,
        audio: AudioBuffer,
        *,
        threshold: float,
        task: str,
        language: str | None,
        prompt: str | None = None,
        min_speech_ms: int = 160,
        min_silence_ms: int = 240,
        pad_ms: int = 80,
        window_samples: int = 512,
    ) -> SttAnalysis:
        readiness = self.readiness(detailed=False)
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            audio_input = self._grpcclient.InferInput(self._audio_input_name, [1, len(audio.samples)], "FP32")
            audio_input.set_data_from_numpy(audio.samples.reshape(1, -1).astype(np.float32))

            sample_rate_input = self._grpcclient.InferInput(self._sample_rate_input_name, [1], "INT32")
            sample_rate_input.set_data_from_numpy(np.asarray([audio.sample_rate], dtype=np.int32))

            threshold_input = self._grpcclient.InferInput(self._threshold_input_name, [1], "FP32")
            threshold_input.set_data_from_numpy(np.asarray([threshold], dtype=np.float32))

            min_speech_ms_input = self._grpcclient.InferInput(self._min_speech_ms_input_name, [1], "INT32")
            min_speech_ms_input.set_data_from_numpy(np.asarray([min_speech_ms], dtype=np.int32))

            min_silence_ms_input = self._grpcclient.InferInput(self._min_silence_ms_input_name, [1], "INT32")
            min_silence_ms_input.set_data_from_numpy(np.asarray([min_silence_ms], dtype=np.int32))

            pad_ms_input = self._grpcclient.InferInput(self._pad_ms_input_name, [1], "INT32")
            pad_ms_input.set_data_from_numpy(np.asarray([pad_ms], dtype=np.int32))

            window_samples_input = self._grpcclient.InferInput(self._window_samples_input_name, [1], "INT32")
            window_samples_input.set_data_from_numpy(np.asarray([window_samples], dtype=np.int32))

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
                    threshold_input,
                    min_speech_ms_input,
                    min_silence_ms_input,
                    pad_ms_input,
                    window_samples_input,
                    task_input,
                    language_input,
                    prompt_input,
                ],
                outputs=[
                    self._grpcclient.InferRequestedOutput(self._transcript_output_name),
                    self._grpcclient.InferRequestedOutput(self._segments_output_name),
                ],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            invalidate_fast_readiness_cache(self)
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running Whisper STT pipeline", exc=exc)
            ) from exc

        transcript = _decode_transcript_output(result.as_numpy(self._transcript_output_name)).strip()
        segments_tensor = result.as_numpy(self._segments_output_name)
        segments_payload = _decode_transcript_output(segments_tensor)
        raw_segments = json.loads(segments_payload) if segments_payload else []

        segments = [
            TranscribedSegment(
                start_ms=int(segment["start_ms"]),
                end_ms=int(segment["end_ms"]),
                duration_ms=int(segment["duration_ms"]),
                average_probability=float(segment["average_probability"]),
                peak_probability=float(segment["peak_probability"]),
                text=str(segment.get("text", "")),
                speaker_id=segment.get("speaker_id"),
            )
            for segment in raw_segments
        ]

        return SttAnalysis(
            threshold=threshold,
            task=task,
            language=language or "auto",
            duration_ms=audio.duration_ms,
            sample_rate=audio.sample_rate,
            transcript=transcript,
            segments=segments,
        )


def analyze_stt(
    audio: AudioBuffer,
    *,
    vad_client,
    stt_client: TritonWhisperClient | TritonWhisperSttPipelineClient,
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

    if hasattr(stt_client, "analyze_audio"):
        return stt_client.analyze_audio(
            audio,
            threshold=threshold,
            task=normalized_task,
            language=normalized_language,
            prompt=prompt,
            min_speech_ms=min_speech_ms,
            min_silence_ms=min_silence_ms,
            pad_ms=pad_ms,
            window_samples=window_samples,
        )

    vad_analysis = analyze_vad(
        audio=audio,
        client=vad_client,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        pad_ms=pad_ms,
        window_samples=window_samples,
    )

    sliced_segments = [_slice_audio(audio, segment.start_ms, segment.end_ms) for segment in vad_analysis.segments]

    if hasattr(stt_client, "transcribe_many"):
        texts = [
            transcript.strip()
            for transcript in stt_client.transcribe_many(
                sliced_segments,
                language=normalized_language,
                task=normalized_task,
                prompt=prompt,
            )
        ]
    else:
        texts = [
            stt_client.transcribe(
                segment_audio,
                language=normalized_language,
                task=normalized_task,
                prompt=prompt,
            ).strip()
            for segment_audio in sliced_segments
        ]

    segments: list[TranscribedSegment] = []
    for segment, text in zip(vad_analysis.segments, texts, strict=True):
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
