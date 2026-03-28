from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pipeline.audio import AudioBuffer
from pipeline.diarization import assign_speakers_to_transcribed
from pipeline.stt import SttAnalysis, TranscribedSegment, _slice_audio, analyze_stt
from pipeline.translation import TritonTranslationClient, normalize_pipeline_language
from pipeline.tts import SynthesizedAudio, TritonTtsClient, encode_wav_preview, validate_tts_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalizationStageError(RuntimeError):
    stage: str
    payload: dict[str, object]
    status_code: int = 503

    def __str__(self) -> str:
        message = self.payload.get("message")
        return str(message) if isinstance(message, str) else f"{self.stage} stage failed"


_REF_MIN_DURATION_MS = 2000
_REF_MAX_DURATION_MS = 10000


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
    total_samples = int(audio.duration_ms * audio.sample_rate / 1000)
    output = np.zeros(total_samples, dtype=np.float32)
    sample_rate: int | None = None

    # Build per-speaker reference cache
    speaker_refs: dict[str, tuple[np.ndarray, str, int] | None] = {}
    for sid in speaker_groups:
        speaker_refs[sid] = _select_reference_segment(audio, diarized_segments, speaker_id=sid)

    segments_synthesized = 0
    for seg in diarized_segments:
        if not seg.text.strip():
            continue

        sid = seg.speaker_id or "speaker_0"
        ref = speaker_refs.get(sid)

        try:
            synth = tts_client.synthesize(
                seg.text,
                language=tts_language,
                speaker_prompt=speaker_prompt,
                ref_audio=ref[0] if ref else None,
                ref_audio_sample_rate=ref[2] if ref else 16000,
                ref_text=ref[1] if ref else None,
            )
        except Exception:
            logger.warning("TTS failed for segment %d-%d ms, filling silence", seg.start_ms, seg.end_ms, exc_info=True)
            continue

        if sample_rate is None:
            sample_rate = synth.sample_rate

        # Time-stretch TTS output to fit the original segment duration
        target_samples = int(seg.duration_ms * (sample_rate or 24000) / 1000)
        stretched = _time_stretch(synth.samples, target_samples)

        # Place into output at the segment's original position
        start_sample = int(seg.start_ms * (sample_rate or 24000) / 1000)
        end_sample = min(start_sample + len(stretched), len(output))
        fit_len = end_sample - start_sample
        if fit_len > 0:
            output[start_sample:end_sample] = stretched[:fit_len]
        segments_synthesized += 1

    if sample_rate is None:
        sample_rate = 24000

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
    diarized_segments = assign_speakers_to_transcribed(audio, stt_analysis.segments)
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
        payload["stages"] = {
            **payload["stages"],
            "translation": {"status": "skipped", "reason": "No transcript text was produced."},
            "tts": {"status": "skipped", "reason": "No translated text was produced."},
        }
        return payload

    # ── Translation stage ──
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
        payload["stages"] = {
            **payload["stages"],
            "tts": {"status": "skipped", "reason": "Translation returned empty text."},
        }
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
