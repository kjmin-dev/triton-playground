from __future__ import annotations

from dataclasses import dataclass

from pipeline.audio import AudioBuffer
from pipeline.stt import SttAnalysis, analyze_stt
from pipeline.translation import TritonTranslationClient, normalize_pipeline_language
from pipeline.tts import TritonTtsClient, encode_wav_preview, validate_tts_language


@dataclass(frozen=True)
class LocalizationStageError(RuntimeError):
    stage: str
    payload: dict[str, object]
    status_code: int = 503

    def __str__(self) -> str:
        message = self.payload.get("message")
        return str(message) if isinstance(message, str) else f"{self.stage} stage failed"


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

    stt_analysis: SttAnalysis
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

    payload: dict[str, object] = {
        "status": "ok",
        **base_payload,
        "transcript": stt_analysis.transcript,
        "translated_text": "",
        "stages": {
            "stt": {
                "status": "ok",
                "language": stt_analysis.language,
                "task": stt_analysis.task,
                "segment_count": len(stt_analysis.segments),
                "transcript": stt_analysis.transcript,
                "segments": [segment.to_dict() for segment in stt_analysis.segments],
            },
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

    payload["translated_text"] = translated_text
    payload["stages"] = {
        **payload["stages"],
        "translation": {
            "status": "ok",
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

    tts_language = validate_tts_language(normalized_target_language)

    try:
        synthesized = tts_client.synthesize(
            translated_text,
            language=tts_language,
            speaker_prompt=speaker_prompt,
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

    payload["stages"] = {
        **payload["stages"],
        "tts": {
            "status": "ok",
            "language": tts_language,
            "sample_rate": synthesized.sample_rate,
            "duration_ms": synthesized.duration_ms,
            "content_type": "audio/wav",
            "audio_base64": encode_wav_preview(synthesized),
        },
    }

    return payload
