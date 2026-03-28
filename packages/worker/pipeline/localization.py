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
    """Pick the best VAD segment for voice cloning reference.

    When speaker_id is provided, only segments from that speaker are considered.
    Returns (samples, text, sample_rate) or None.
    """
    filtered = segments
    if speaker_id is not None:
        filtered = [s for s in segments if s.speaker_id == speaker_id]
        if not filtered:
            filtered = segments  # fallback to all segments

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
    """Group segments by speaker_id. Segments with None speaker_id go under 'speaker_0'."""
    groups: dict[str, list[TranscribedSegment]] = defaultdict(list)
    for seg in segments:
        key = seg.speaker_id or "speaker_0"
        groups[key].append(seg)
    return dict(groups)


def _synthesize_per_speaker(
    *,
    audio: AudioBuffer,
    speaker_groups: dict[str, list[TranscribedSegment]],
    translated_text: str,
    tts_language: str,
    speaker_prompt: str | None,
    tts_client: TritonTtsClient,
    all_segments: list[TranscribedSegment],
) -> tuple[SynthesizedAudio, dict[str, object]]:
    """Synthesize TTS per speaker and merge into a single waveform.

    For single-speaker audio, behaves identically to the previous implementation.
    For multi-speaker, synthesizes each speaker's portion with that speaker's
    voice reference, then concatenates in original temporal order.
    """
    speaker_ids = sorted(speaker_groups.keys())
    is_multi = len(speaker_ids) > 1

    if not is_multi:
        # Single speaker: synthesize full translated text with best reference
        ref = _select_reference_segment(audio, all_segments)
        synthesized = tts_client.synthesize(
            translated_text,
            language=tts_language,
            speaker_prompt=speaker_prompt,
            ref_audio=ref[0] if ref else None,
            ref_audio_sample_rate=ref[2] if ref else 16000,
            ref_text=ref[1] if ref else None,
        )
        vc_mode = "icl" if (ref and ref[1]) else ("x_vector" if ref else "none")
        return synthesized, {
            "voice_cloning": ref is not None,
            "voice_cloning_mode": vc_mode,
            "speaker_count": 1,
        }

    # Multi-speaker: synthesize per speaker, then concatenate
    logger.info("Multi-speaker TTS: %d speakers detected", len(speaker_ids))

    # For multi-speaker, we translate the full text once (already done upstream)
    # and synthesize per speaker. Each speaker gets their own voice reference.
    # The translation is split proportionally by speaker segment count.
    speaker_audios: list[tuple[str, SynthesizedAudio]] = []

    for sid in speaker_ids:
        speaker_segs = speaker_groups[sid]
        speaker_text = " ".join(s.text for s in speaker_segs if s.text.strip())
        if not speaker_text.strip():
            continue

        ref = _select_reference_segment(audio, all_segments, speaker_id=sid)

        try:
            synth = tts_client.synthesize(
                speaker_text,
                language=tts_language,
                speaker_prompt=speaker_prompt,
                ref_audio=ref[0] if ref else None,
                ref_audio_sample_rate=ref[2] if ref else 16000,
                ref_text=ref[1] if ref else None,
            )
            speaker_audios.append((sid, synth))
        except Exception:
            logger.warning("TTS failed for %s, skipping", sid, exc_info=True)

    if not speaker_audios:
        raise RuntimeError("TTS failed for all speakers")

    # Concatenate all speaker audio sequentially
    sample_rate = speaker_audios[0][1].sample_rate
    merged = np.concatenate([sa.samples for _, sa in speaker_audios])
    synthesized = SynthesizedAudio(sample_rate=sample_rate, samples=merged)

    return synthesized, {
        "voice_cloning": True,
        "voice_cloning_mode": "icl",
        "speaker_count": len(speaker_ids),
        "speakers": [sid for sid, _ in speaker_audios],
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

    # ── TTS stage (per-speaker voice cloning) ──
    tts_language = validate_tts_language(normalized_target_language)

    t0 = time.monotonic()
    try:
        synthesized, tts_meta = _synthesize_per_speaker(
            audio=audio,
            speaker_groups=speaker_groups,
            translated_text=translated_text,
            tts_language=tts_language,
            speaker_prompt=speaker_prompt,
            tts_client=tts_client,
            all_segments=diarized_segments,
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
