from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pipeline.audio import AudioBuffer
from pipeline.diarization import assign_speakers_to_transcribed
from pipeline.stt import SttAnalysis, TranscribedSegment, _slice_audio, analyze_stt
from pipeline.translation import TritonTranslationClient, normalize_pipeline_language
from pipeline.triton import (
    TritonUnavailableError,
    check_readiness,
    describe_triton_error,
    get_fast_readiness,
    init_fast_readiness_cache,
    invalidate_fast_readiness_cache,
)
from pipeline.tts import (
    SynthesizedAudio,
    TritonTtsClient,
    TtsSynthesisRequest,
    encode_wav_preview,
    validate_tts_language,
)

logger = logging.getLogger(__name__)

LOCALIZE_PIPELINE_MODEL_NAME = "localize_pipeline"
LOCALIZE_TEXT_PIPELINE_MODEL_NAME = "localize_text_pipeline"


@dataclass(frozen=True)
class LocalizationStageError(RuntimeError):
    stage: str
    payload: dict[str, object]
    status_code: int = 503

    def __str__(self) -> str:
        message = self.payload.get("message")
        return str(message) if isinstance(message, str) else f"{self.stage} stage failed"


@dataclass(frozen=True)
class LocalizedTextAnalysis:
    transcript: str
    translated_text: str
    segments: list[TranscribedSegment]
    stt_elapsed_ms: int
    translation_elapsed_ms: int


@dataclass(frozen=True)
class LocalizedAudioAnalysis:
    transcript: str
    translated_text: str
    segments: list[TranscribedSegment]
    stt_elapsed_ms: int
    translation_elapsed_ms: int
    tts_elapsed_ms: int
    synthesized_audio: SynthesizedAudio | None
    tts_meta: dict[str, object]


class TritonLocalizeTextPipelineClient:
    def __init__(self, url: str, model_name: str = LOCALIZE_TEXT_PIPELINE_MODEL_NAME) -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name
        init_fast_readiness_cache(self)

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self, *, refresh: bool = False, detailed: bool = True):
        if detailed:
            return check_readiness(self._client, self._url, self._model_name)
        return get_fast_readiness(self, self._client, url=self._url, model_name=self._model_name, refresh=refresh)

    def localize_text(
        self,
        *,
        audio: AudioBuffer,
        threshold: float,
        source_language: str | None,
        target_language: str,
        prompt: str | None = None,
        min_speech_ms: int = 160,
        min_silence_ms: int = 240,
        pad_ms: int = 80,
        window_samples: int = 512,
    ) -> LocalizedTextAnalysis:
        readiness = self.readiness(detailed=False)
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            audio_input = self._grpcclient.InferInput("audio_pcm", [1, len(audio.samples)], "FP32")
            audio_input.set_data_from_numpy(audio.samples.reshape(1, -1).astype(np.float32))

            sample_rate_input = self._grpcclient.InferInput("sample_rate", [1], "INT32")
            sample_rate_input.set_data_from_numpy(np.asarray([audio.sample_rate], dtype=np.int32))

            threshold_input = self._grpcclient.InferInput("threshold", [1], "FP32")
            threshold_input.set_data_from_numpy(np.asarray([threshold], dtype=np.float32))

            min_speech_ms_input = self._grpcclient.InferInput("min_speech_ms", [1], "INT32")
            min_speech_ms_input.set_data_from_numpy(np.asarray([min_speech_ms], dtype=np.int32))

            min_silence_ms_input = self._grpcclient.InferInput("min_silence_ms", [1], "INT32")
            min_silence_ms_input.set_data_from_numpy(np.asarray([min_silence_ms], dtype=np.int32))

            pad_ms_input = self._grpcclient.InferInput("pad_ms", [1], "INT32")
            pad_ms_input.set_data_from_numpy(np.asarray([pad_ms], dtype=np.int32))

            window_samples_input = self._grpcclient.InferInput("window_samples", [1], "INT32")
            window_samples_input.set_data_from_numpy(np.asarray([window_samples], dtype=np.int32))

            source_language_input = self._grpcclient.InferInput("source_language", [1], "BYTES")
            source_language_input.set_data_from_numpy(np.asarray([source_language or ""], dtype=object))

            target_language_input = self._grpcclient.InferInput("target_language", [1], "BYTES")
            target_language_input.set_data_from_numpy(np.asarray([target_language], dtype=object))

            prompt_input = self._grpcclient.InferInput("prompt", [1], "BYTES")
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
                    source_language_input,
                    target_language_input,
                    prompt_input,
                ],
                outputs=[
                    self._grpcclient.InferRequestedOutput("transcript"),
                    self._grpcclient.InferRequestedOutput("segments_json"),
                    self._grpcclient.InferRequestedOutput("translated_text"),
                    self._grpcclient.InferRequestedOutput("stt_elapsed_ms"),
                    self._grpcclient.InferRequestedOutput("translation_elapsed_ms"),
                ],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            invalidate_fast_readiness_cache(self)
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running localize text pipeline", exc=exc)
            ) from exc

        transcript_tensor = result.as_numpy("transcript")
        segments_tensor = result.as_numpy("segments_json")
        translated_text_tensor = result.as_numpy("translated_text")
        stt_elapsed_ms_tensor = result.as_numpy("stt_elapsed_ms")
        translation_elapsed_ms_tensor = result.as_numpy("translation_elapsed_ms")
        if (
            transcript_tensor is None
            or segments_tensor is None
            or translated_text_tensor is None
            or stt_elapsed_ms_tensor is None
            or translation_elapsed_ms_tensor is None
        ):
            raise TritonUnavailableError("Localize text pipeline did not return the expected output tensors.")

        transcript = _decode_bytes_tensor(transcript_tensor)
        translated_text = _decode_bytes_tensor(translated_text_tensor)
        segments = _segments_from_json_payload(_decode_bytes_tensor(segments_tensor))

        return LocalizedTextAnalysis(
            transcript=transcript,
            translated_text=translated_text,
            segments=segments,
            stt_elapsed_ms=int(np.asarray(stt_elapsed_ms_tensor).reshape(-1)[0]),
            translation_elapsed_ms=int(np.asarray(translation_elapsed_ms_tensor).reshape(-1)[0]),
        )


class TritonLocalizePipelineClient:
    def __init__(self, url: str, model_name: str = LOCALIZE_PIPELINE_MODEL_NAME) -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name
        init_fast_readiness_cache(self)

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self, *, refresh: bool = False, detailed: bool = True):
        if detailed:
            return check_readiness(self._client, self._url, self._model_name)
        return get_fast_readiness(self, self._client, url=self._url, model_name=self._model_name, refresh=refresh)

    def localize(
        self,
        *,
        audio: AudioBuffer,
        threshold: float,
        source_language: str | None,
        target_language: str,
        prompt: str | None = None,
        speaker_prompt: str | None = None,
        min_speech_ms: int = 160,
        min_silence_ms: int = 240,
        pad_ms: int = 80,
        window_samples: int = 512,
    ) -> LocalizedAudioAnalysis:
        readiness = self.readiness(detailed=False)
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            audio_input = self._grpcclient.InferInput("audio_pcm", [1, len(audio.samples)], "FP32")
            audio_input.set_data_from_numpy(audio.samples.reshape(1, -1).astype(np.float32))

            sample_rate_input = self._grpcclient.InferInput("sample_rate", [1], "INT32")
            sample_rate_input.set_data_from_numpy(np.asarray([audio.sample_rate], dtype=np.int32))

            threshold_input = self._grpcclient.InferInput("threshold", [1], "FP32")
            threshold_input.set_data_from_numpy(np.asarray([threshold], dtype=np.float32))

            min_speech_ms_input = self._grpcclient.InferInput("min_speech_ms", [1], "INT32")
            min_speech_ms_input.set_data_from_numpy(np.asarray([min_speech_ms], dtype=np.int32))

            min_silence_ms_input = self._grpcclient.InferInput("min_silence_ms", [1], "INT32")
            min_silence_ms_input.set_data_from_numpy(np.asarray([min_silence_ms], dtype=np.int32))

            pad_ms_input = self._grpcclient.InferInput("pad_ms", [1], "INT32")
            pad_ms_input.set_data_from_numpy(np.asarray([pad_ms], dtype=np.int32))

            window_samples_input = self._grpcclient.InferInput("window_samples", [1], "INT32")
            window_samples_input.set_data_from_numpy(np.asarray([window_samples], dtype=np.int32))

            source_language_input = self._grpcclient.InferInput("source_language", [1], "BYTES")
            source_language_input.set_data_from_numpy(np.asarray([source_language or ""], dtype=object))

            target_language_input = self._grpcclient.InferInput("target_language", [1], "BYTES")
            target_language_input.set_data_from_numpy(np.asarray([target_language], dtype=object))

            prompt_input = self._grpcclient.InferInput("prompt", [1], "BYTES")
            prompt_input.set_data_from_numpy(np.asarray([prompt or ""], dtype=object))

            speaker_prompt_input = self._grpcclient.InferInput("speaker_prompt", [1], "BYTES")
            speaker_prompt_input.set_data_from_numpy(np.asarray([speaker_prompt or ""], dtype=object))

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
                    source_language_input,
                    target_language_input,
                    prompt_input,
                    speaker_prompt_input,
                ],
                outputs=[
                    self._grpcclient.InferRequestedOutput("transcript"),
                    self._grpcclient.InferRequestedOutput("segments_json"),
                    self._grpcclient.InferRequestedOutput("translated_text"),
                    self._grpcclient.InferRequestedOutput("stt_elapsed_ms"),
                    self._grpcclient.InferRequestedOutput("translation_elapsed_ms"),
                    self._grpcclient.InferRequestedOutput("tts_elapsed_ms"),
                    self._grpcclient.InferRequestedOutput("audio_pcm"),
                    self._grpcclient.InferRequestedOutput("audio_length"),
                    self._grpcclient.InferRequestedOutput("synthesized_sample_rate"),
                    self._grpcclient.InferRequestedOutput("tts_meta_json"),
                ],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            invalidate_fast_readiness_cache(self)
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running localize pipeline", exc=exc)
            ) from exc

        transcript_tensor = result.as_numpy("transcript")
        segments_tensor = result.as_numpy("segments_json")
        translated_text_tensor = result.as_numpy("translated_text")
        stt_elapsed_ms_tensor = result.as_numpy("stt_elapsed_ms")
        translation_elapsed_ms_tensor = result.as_numpy("translation_elapsed_ms")
        tts_elapsed_ms_tensor = result.as_numpy("tts_elapsed_ms")
        audio_tensor = result.as_numpy("audio_pcm")
        audio_length_tensor = result.as_numpy("audio_length")
        synthesized_sample_rate_tensor = result.as_numpy("synthesized_sample_rate")
        tts_meta_tensor = result.as_numpy("tts_meta_json")
        if (
            transcript_tensor is None
            or segments_tensor is None
            or translated_text_tensor is None
            or stt_elapsed_ms_tensor is None
            or translation_elapsed_ms_tensor is None
            or tts_elapsed_ms_tensor is None
            or audio_tensor is None
            or audio_length_tensor is None
            or synthesized_sample_rate_tensor is None
            or tts_meta_tensor is None
        ):
            raise TritonUnavailableError("Localize pipeline did not return the expected output tensors.")

        transcript = _decode_bytes_tensor(transcript_tensor)
        translated_text = _decode_bytes_tensor(translated_text_tensor)
        segments = _segments_from_json_payload(_decode_bytes_tensor(segments_tensor))
        tts_meta = json.loads(_decode_bytes_tensor(tts_meta_tensor) or "{}")

        audio_length = int(np.asarray(audio_length_tensor).reshape(-1)[0])
        synthesized_audio = None
        if audio_length > 0:
            sample_rate = int(np.asarray(synthesized_sample_rate_tensor).reshape(-1)[0])
            if sample_rate <= 0:
                raise TritonUnavailableError(
                    f"Localize pipeline returned an invalid synthesized sample rate: {sample_rate}"
                )
            waveform = np.asarray(audio_tensor, dtype=np.float32).reshape(-1)[:audio_length]
            synthesized_audio = SynthesizedAudio(sample_rate=sample_rate, samples=waveform.astype(np.float32))

        return LocalizedAudioAnalysis(
            transcript=transcript,
            translated_text=translated_text,
            segments=segments,
            stt_elapsed_ms=int(np.asarray(stt_elapsed_ms_tensor).reshape(-1)[0]),
            translation_elapsed_ms=int(np.asarray(translation_elapsed_ms_tensor).reshape(-1)[0]),
            tts_elapsed_ms=int(np.asarray(tts_elapsed_ms_tensor).reshape(-1)[0]),
            synthesized_audio=synthesized_audio,
            tts_meta=tts_meta if isinstance(tts_meta, dict) else {},
        )


_REF_MIN_DURATION_MS = 2000
_REF_MAX_DURATION_MS = 10000


def _decode_bytes_tensor(tensor: np.ndarray | None) -> str:
    if tensor is None:
        return ""

    flattened = tensor.reshape(-1)
    if flattened.size == 0:
        return ""

    value = flattened[0].item() if hasattr(flattened[0], "item") else flattened[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _segments_from_json_payload(payload: str) -> list[TranscribedSegment]:
    raw_segments = json.loads(payload or "[]")
    return [
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


def _select_reference_segment(
    audio: AudioBuffer,
    segments: list[TranscribedSegment],
    speaker_id: str | None = None,
) -> tuple[np.ndarray, str, int] | None:
    """Pick the best VAD segment for voice cloning reference."""
    filtered = segments
    if speaker_id is not None:
        filtered = [s for s in segments if s.speaker_id == speaker_id]
        if not filtered:
            filtered = segments

    candidates = [
        seg for seg in filtered if seg.text.strip() and _REF_MIN_DURATION_MS <= seg.duration_ms <= _REF_MAX_DURATION_MS
    ]
    if not candidates:
        candidates = [seg for seg in filtered if seg.text.strip() and seg.duration_ms >= 500]
    if not candidates:
        return None

    best = max(candidates, key=lambda s: s.average_probability)
    ref_buf = _slice_audio(audio, best.start_ms, best.end_ms)
    return (ref_buf.samples, best.text, ref_buf.sample_rate)


def _build_base_payload(
    *,
    threshold: float,
    source_language: str,
    target_language: str,
    stt_model: str,
    translation_model: str,
    tts_model: str,
) -> dict[str, object]:
    return {
        "threshold": threshold,
        "source_language": source_language,
        "target_language": target_language,
        "models": {
            "stt": stt_model,
            "translation": translation_model,
            "tts": tts_model,
        },
    }


def _group_segments_by_speaker(
    segments: list[TranscribedSegment],
) -> dict[str, list[TranscribedSegment]]:
    groups: dict[str, list[TranscribedSegment]] = defaultdict(list)
    for seg in segments:
        groups[seg.speaker_id or "speaker_0"].append(seg)
    return dict(groups)


def _time_stretch(samples: np.ndarray, target_length: int) -> np.ndarray:
    """Resample audio to exactly target_length samples via linear interpolation."""
    if len(samples) == 0 or target_length <= 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(samples) == target_length:
        return samples
    src_pos = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    tgt_pos = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(tgt_pos, src_pos, samples).astype(np.float32)


def _synthesize_time_aligned(
    *,
    audio: AudioBuffer,
    diarized_segments: list[TranscribedSegment],
    speaker_groups: dict[str, list[TranscribedSegment]],
    translated_text: str,
    tts_language: str,
    speaker_prompt: str | None,
    tts_client: TritonTtsClient,
) -> tuple[SynthesizedAudio, dict[str, object]]:
    """Synthesize TTS per segment, time-stretched to match original timing.

    Builds a waveform that mirrors the original audio duration: each segment's
    TTS output is stretched/compressed to fit the original segment's time slot,
    with silence filling the gaps between segments.
    """
    sample_rate: int | None = None

    # Build per-speaker reference cache
    speaker_refs: dict[str, tuple[np.ndarray, str, int] | None] = {}
    for sid in speaker_groups:
        speaker_refs[sid] = _select_reference_segment(audio, diarized_segments, speaker_id=sid)

    pending_segments: list[tuple[TranscribedSegment, TtsSynthesisRequest]] = []
    for seg in diarized_segments:
        if not seg.text.strip():
            continue

        sid = seg.speaker_id or "speaker_0"
        ref = speaker_refs.get(sid)
        pending_segments.append(
            (
                seg,
                TtsSynthesisRequest(
                    text=seg.text,
                    language=tts_language,
                    speaker_prompt=speaker_prompt,
                    ref_audio=ref[0] if ref else None,
                    ref_audio_sample_rate=ref[2] if ref else 16000,
                    ref_text=ref[1] if ref else None,
                ),
            )
        )

    synthesized_segments: list[tuple[TranscribedSegment, SynthesizedAudio]] = []
    if pending_segments and hasattr(tts_client, "synthesize_many"):
        try:
            batch_outputs = tts_client.synthesize_many([request for _, request in pending_segments])
            synthesized_segments = [
                (segment, synthesized)
                for (segment, _), synthesized in zip(pending_segments, batch_outputs, strict=True)
            ]
        except Exception:
            logger.warning("Batched TTS failed, falling back to per-segment synthesis", exc_info=True)

    if not synthesized_segments:
        for seg, request in pending_segments:
            try:
                synth = tts_client.synthesize(
                    request.text,
                    language=request.language,
                    speaker_prompt=request.speaker_prompt,
                    ref_audio=request.ref_audio,
                    ref_audio_sample_rate=request.ref_audio_sample_rate,
                    ref_text=request.ref_text,
                )
            except Exception:
                logger.warning(
                    "TTS failed for segment %d-%d ms, filling silence", seg.start_ms, seg.end_ms, exc_info=True
                )
                continue
            synthesized_segments.append((seg, synth))

    if sample_rate is None:
        for _, synth in synthesized_segments:
            sample_rate = synth.sample_rate
            break
    if sample_rate is None:
        sample_rate = 24000

    total_samples = int(audio.duration_ms * sample_rate / 1000)
    output = np.zeros(total_samples, dtype=np.float32)

    segments_synthesized = 0
    for seg, synth in synthesized_segments:
        # Time-stretch TTS output to fit the original segment duration
        target_samples = int(seg.duration_ms * sample_rate / 1000)
        stretched = _time_stretch(synth.samples, target_samples)

        # Place into output at the segment's original position
        start_sample = int(seg.start_ms * sample_rate / 1000)
        end_sample = min(start_sample + len(stretched), len(output))
        fit_len = end_sample - start_sample
        if fit_len > 0:
            output[start_sample:end_sample] = stretched[:fit_len]
        segments_synthesized += 1

    # Trim trailing silence
    last_seg = max(diarized_segments, key=lambda s: s.end_ms) if diarized_segments else None
    if last_seg:
        trim_sample = min(len(output), int((last_seg.end_ms + 500) * sample_rate / 1000))
        output = output[:trim_sample]

    n_speakers = len(speaker_groups)
    vc_mode = "icl" if any(r and r[1] for r in speaker_refs.values()) else "x_vector"

    return SynthesizedAudio(sample_rate=sample_rate, samples=output), {
        "voice_cloning": True,
        "voice_cloning_mode": vc_mode,
        "speaker_count": n_speakers,
        "segments_synthesized": segments_synthesized,
        "time_aligned": True,
        "speakers": sorted(speaker_groups.keys()),
    }


def localize_audio(
    *,
    audio: AudioBuffer,
    threshold: float,
    source_language: str | None,
    target_language: str,
    prompt: str | None,
    speaker_prompt: str | None,
    stt_model: str,
    translation_model: str,
    tts_model: str,
    localize_pipeline_client: TritonLocalizePipelineClient | None = None,
    localize_text_client: TritonLocalizeTextPipelineClient | None = None,
    vad_client,
    stt_client,
    translation_client: TritonTranslationClient,
    tts_client: TritonTtsClient,
) -> dict[str, object]:
    normalized_source_language = normalize_pipeline_language(source_language, allow_auto=True)
    normalized_target_language = normalize_pipeline_language(target_language, allow_auto=False)
    if normalized_target_language is None:
        raise ValueError("target_language must be set")

    base_payload = _build_base_payload(
        threshold=threshold,
        source_language=normalized_source_language or "auto",
        target_language=normalized_target_language,
        stt_model=stt_model,
        translation_model=translation_model,
        tts_model=tts_model,
    )

    t0_pipeline = time.monotonic()

    # ── STT stage ──
    stt_analysis: SttAnalysis
    t0 = time.monotonic()
    translated_text_from_pipeline: str | None = None
    translation_elapsed_ms: int | None = None
    localized_audio: LocalizedAudioAnalysis | None = None
    localized_text: LocalizedTextAnalysis | None = None
    if localize_pipeline_client is not None:
        try:
            localized_audio = localize_pipeline_client.localize(
                audio=audio,
                threshold=threshold,
                source_language=normalized_source_language,
                target_language=normalized_target_language,
                prompt=prompt,
                speaker_prompt=speaker_prompt,
            )
        except Exception:
            logger.warning("Triton localize pipeline failed, falling back to worker orchestration", exc_info=True)
            localized_audio = None

    if localized_audio is None and localize_text_client is not None:
        try:
            localized_text = localize_text_client.localize_text(
                audio=audio,
                threshold=threshold,
                source_language=normalized_source_language,
                target_language=normalized_target_language,
                prompt=prompt,
            )
        except Exception:
            logger.warning("Triton localize text pipeline failed, falling back to worker orchestration", exc_info=True)
            localized_text = None

    if localized_audio is not None:
        stt_analysis = SttAnalysis(
            threshold=threshold,
            task="transcribe",
            language=normalized_source_language or "auto",
            duration_ms=audio.duration_ms,
            sample_rate=audio.sample_rate,
            transcript=localized_audio.transcript,
            segments=localized_audio.segments,
        )
        translated_text_from_pipeline = localized_audio.translated_text
        stt_elapsed_ms = localized_audio.stt_elapsed_ms
        translation_elapsed_ms = localized_audio.translation_elapsed_ms
    elif localized_text is not None:
        stt_analysis = SttAnalysis(
            threshold=threshold,
            task="transcribe",
            language=normalized_source_language or "auto",
            duration_ms=audio.duration_ms,
            sample_rate=audio.sample_rate,
            transcript=localized_text.transcript,
            segments=localized_text.segments,
        )
        translated_text_from_pipeline = localized_text.translated_text
        stt_elapsed_ms = localized_text.stt_elapsed_ms
        translation_elapsed_ms = localized_text.translation_elapsed_ms
    else:
        try:
            stt_analysis = analyze_stt(
                audio=audio,
                vad_client=vad_client,
                stt_client=stt_client,
                threshold=threshold,
                language=normalized_source_language,
                prompt=prompt,
            )
        except ValueError:
            raise
        except Exception as exc:
            raise LocalizationStageError(
                stage="stt",
                payload={
                    "status": "error",
                    **base_payload,
                    "stage": "stt",
                    "message": f"STT stage failed: {exc}",
                    "stages": {
                        "stt": {"status": "error", "message": str(exc)},
                        "translation": {"status": "blocked"},
                        "tts": {"status": "blocked"},
                    },
                },
            ) from exc

        stt_elapsed_ms = round((time.monotonic() - t0) * 1000)

    logger.info("STT completed in %d ms (%d segments)", stt_elapsed_ms, len(stt_analysis.segments))

    # ── Speaker diarization ──
    diarized_segments = (
        stt_analysis.segments
        if any(segment.speaker_id is not None for segment in stt_analysis.segments)
        else assign_speakers_to_transcribed(audio, stt_analysis.segments)
    )
    speaker_groups = _group_segments_by_speaker(diarized_segments)
    n_speakers = len(speaker_groups)
    if n_speakers > 1:
        logger.info("Diarization: %d speakers detected", n_speakers)

    stt_warnings: list[str] = []
    if normalized_source_language is None:
        stt_warnings.append(
            "source_language is auto; Whisper may hallucinate non-target text. "
            "Set source_language explicitly (ko, en, ja, zh) for better accuracy."
        )

    stt_stage: dict[str, object] = {
        "status": "ok",
        "elapsed_ms": stt_elapsed_ms,
        "language": stt_analysis.language,
        "task": stt_analysis.task,
        "segment_count": len(diarized_segments),
        "speaker_count": n_speakers,
        "transcript": stt_analysis.transcript,
        "segments": [segment.to_dict() for segment in diarized_segments],
    }
    if stt_warnings:
        stt_stage["warnings"] = stt_warnings

    payload: dict[str, object] = {
        "status": "ok",
        **base_payload,
        "transcript": stt_analysis.transcript,
        "translated_text": "",
        "stages": {
            "stt": stt_stage,
            "translation": {"status": "pending"},
            "tts": {"status": "pending"},
        },
    }

    if not stt_analysis.transcript:
        tts_skip_reason = "No transcript text was produced."
        if localized_audio is not None:
            tts_skip_reason = str(localized_audio.tts_meta.get("reason", tts_skip_reason))
        payload["stages"] = {
            **payload["stages"],
            "translation": {"status": "skipped", "reason": "No transcript text was produced."},
            "tts": {"status": "skipped", "reason": tts_skip_reason},
        }
        return payload

    # ── Translation stage ──
    if translated_text_from_pipeline is not None and translation_elapsed_ms is not None:
        translated_text = translated_text_from_pipeline.strip()
    else:
        t0 = time.monotonic()
        try:
            translated_text = translation_client.translate(
                stt_analysis.transcript,
                source_language=normalized_source_language,
                target_language=normalized_target_language,
            ).strip()
        except Exception as exc:
            raise LocalizationStageError(
                stage="translation",
                payload={
                    "status": "error",
                    **payload,
                    "stage": "translation",
                    "message": f"Translation stage failed: {exc}",
                    "stages": {
                        **payload["stages"],
                        "translation": {"status": "error", "message": str(exc)},
                        "tts": {"status": "blocked"},
                    },
                },
            ) from exc

        translation_elapsed_ms = round((time.monotonic() - t0) * 1000)
    logger.info("Translation completed in %d ms", translation_elapsed_ms)

    payload["translated_text"] = translated_text
    payload["stages"] = {
        **payload["stages"],
        "translation": {
            "status": "ok",
            "elapsed_ms": translation_elapsed_ms,
            "source_language": normalized_source_language or "auto",
            "target_language": normalized_target_language,
            "text": translated_text,
        },
    }

    if not translated_text:
        tts_skip_reason = "Translation returned empty text."
        if localized_audio is not None:
            tts_skip_reason = str(localized_audio.tts_meta.get("reason", tts_skip_reason))
        payload["stages"] = {
            **payload["stages"],
            "tts": {"status": "skipped", "reason": tts_skip_reason},
        }
        return payload

    if localized_audio is not None:
        total_elapsed_ms = round((time.monotonic() - t0_pipeline) * 1000)
        tts_stage = {
            "status": str(localized_audio.tts_meta.get("status", "ok")),
            "elapsed_ms": localized_audio.tts_elapsed_ms,
            "language": validate_tts_language(normalized_target_language),
            **{
                key: value
                for key, value in localized_audio.tts_meta.items()
                if key not in {"status", "sample_rate", "duration_ms", "content_type", "audio_base64"}
            },
        }
        synthesized = localized_audio.synthesized_audio
        if synthesized is not None:
            tts_stage.update(
                {
                    "sample_rate": synthesized.sample_rate,
                    "duration_ms": synthesized.duration_ms,
                    "content_type": "audio/wav",
                    "audio_base64": encode_wav_preview(synthesized),
                }
            )
        elif "reason" not in tts_stage:
            tts_stage["reason"] = "Triton localize pipeline returned no synthesized audio."

        payload["stages"] = {**payload["stages"], "tts": tts_stage}
        payload["elapsed_ms"] = total_elapsed_ms
        return payload

    # ── TTS stage (per-segment, time-aligned) ──
    tts_language = validate_tts_language(normalized_target_language)

    t0 = time.monotonic()
    try:
        synthesized, tts_meta = _synthesize_time_aligned(
            audio=audio,
            diarized_segments=diarized_segments,
            speaker_groups=speaker_groups,
            translated_text=translated_text,
            tts_language=tts_language,
            speaker_prompt=speaker_prompt,
            tts_client=tts_client,
        )
    except Exception as exc:
        raise LocalizationStageError(
            stage="tts",
            payload={
                "status": "error",
                **payload,
                "stage": "tts",
                "message": f"TTS stage failed: {exc}",
                "stages": {
                    **payload["stages"],
                    "tts": {"status": "error", "message": str(exc)},
                },
            },
        ) from exc

    tts_elapsed_ms = round((time.monotonic() - t0) * 1000)
    total_elapsed_ms = round((time.monotonic() - t0_pipeline) * 1000)
    logger.info("TTS completed in %d ms (total pipeline: %d ms)", tts_elapsed_ms, total_elapsed_ms)

    payload["stages"] = {
        **payload["stages"],
        "tts": {
            "status": "ok",
            "elapsed_ms": tts_elapsed_ms,
            "language": tts_language,
            **tts_meta,
            "sample_rate": synthesized.sample_rate,
            "duration_ms": synthesized.duration_ms,
            "content_type": "audio/wav",
            "audio_base64": encode_wav_preview(synthesized),
        },
    }
    payload["elapsed_ms"] = total_elapsed_ms

    return payload
