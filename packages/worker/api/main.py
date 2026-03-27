from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.audio import UnsupportedAudioError, decode_wav, resample_audio
from pipeline.localization import LocalizationStageError, localize_audio
from pipeline.model_catalog import get_model_spec, get_profile_model_ids, list_model_specs
from pipeline.runtime_status import TritonReadiness, build_ready_payload
from pipeline.stt import TritonWhisperClient, analyze_stt
from pipeline.stt_contract import DEFAULT_WHISPER_MODEL_ID
from pipeline.translation import TritonTranslationClient
from pipeline.translation_contract import DEFAULT_TRANSLATION_MODEL_ID
from pipeline.triton import (
    TritonUnavailableError,
    TritonVadClient,
    inspect_model_repository,
)
from pipeline.tts import TritonTtsClient, encode_wav_preview, validate_tts_language
from pipeline.tts_contract import DEFAULT_TTS_MODEL_ID
from pipeline.vad import analyze_vad

app = FastAPI(title="Triton Playground Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _triton_grpc_url() -> str:
    return os.getenv("TRITON_GRPC_URL", "localhost:8001")


def _model_profile() -> str:
    return os.getenv("MODEL_PROFILE", "baseline")


def _default_model_repository_root() -> str:
    return str(Path(__file__).resolve().parents[3] / "model_repository")


def _model_repository_root() -> str:
    return os.getenv("MODEL_REPOSITORY_ROOT") or _default_model_repository_root()


def _get_stage_model_spec(model_id: str, expected_stage: str):
    try:
        model_spec = get_model_spec(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if model_spec.stage != expected_stage or model_spec.repository_model_name is None:
        raise HTTPException(status_code=400, detail=f"{model_id} is not a configured {expected_stage} model")

    return model_spec


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "profile": _model_profile(),
        "triton_grpc_url": _triton_grpc_url(),
        "model_repository_root": _model_repository_root(),
    }


@app.get("/api/models")
async def models():
    return {
        "profile": _model_profile(),
        "baseline_model_ids": list(get_profile_model_ids("baseline")),
        "models": [spec.to_dict() for spec in list_model_specs()],
    }


@app.get("/api/ready")
async def ready():
    repository_status = inspect_model_repository(
        repository_root=_model_repository_root(),
        model_name="silero_vad",
    )

    try:
        readiness = TritonVadClient(url=_triton_grpc_url()).readiness()
    except TritonUnavailableError as exc:
        readiness = TritonReadiness.from_error(
            server_url=_triton_grpc_url(),
            model_name="silero_vad",
            issue=str(exc),
        )
        payload = build_ready_payload(_model_profile(), readiness)
        payload["repository"] = repository_status.to_dict()
        return JSONResponse(status_code=503, content=payload)

    payload = build_ready_payload(_model_profile(), readiness)
    payload["repository"] = repository_status.to_dict()
    return JSONResponse(status_code=200 if readiness.ready else 503, content=payload)


@app.post("/api/tts")
async def tts(
    text: str = Query(..., min_length=1, max_length=5000),
    language: str = Query(...),
    prompt: str | None = Query(None, max_length=200),
    model: str = Query(DEFAULT_TTS_MODEL_ID),
):
    model_spec = _get_stage_model_spec(model, "tts")

    try:
        normalized_language = validate_tts_language(language)
        synthesized = TritonTtsClient(
            url=_triton_grpc_url(),
            model_name=model_spec.repository_model_name,
        ).synthesize(
            text,
            language=normalized_language,
            speaker_prompt=prompt,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TritonUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "ok",
        "model": model,
        "repository_model_name": model_spec.repository_model_name,
        "language": normalized_language,
        "text": text,
        "content_type": "audio/wav",
        "sample_rate": synthesized.sample_rate,
        "duration_ms": synthesized.duration_ms,
        "audio_base64": encode_wav_preview(synthesized),
    }


@app.post("/api/stt")
async def stt(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.99),
    language: str | None = Query(None),
    task: str = Query("transcribe"),
    prompt: str | None = Query(None, max_length=200),
    model: str = Query(DEFAULT_WHISPER_MODEL_ID),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    model_spec = _get_stage_model_spec(model, "stt")

    try:
        audio = resample_audio(decode_wav(blob), target_sample_rate=16000)
        analysis = analyze_stt(
            audio=audio,
            vad_client=TritonVadClient(url=_triton_grpc_url()),
            stt_client=TritonWhisperClient(
                url=_triton_grpc_url(),
                model_name=model_spec.repository_model_name,
            ),
            threshold=threshold,
            language=language,
            task=task,
            prompt=prompt,
        )
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TritonUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "ok",
        "filename": file.filename,
        "model": model,
        "repository_model_name": model_spec.repository_model_name,
        "sample_rate": analysis.sample_rate,
        "duration_ms": analysis.duration_ms,
        "threshold": analysis.threshold,
        "task": analysis.task,
        "language": analysis.language,
        "segment_count": len(analysis.segments),
        "transcript": analysis.transcript,
        "segments": [segment.to_dict() for segment in analysis.segments],
    }


@app.post("/api/localize")
async def localize(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.99),
    source_language: str | None = Query(None),
    target_language: str = Query(...),
    prompt: str | None = Query(None, max_length=200),
    speaker_prompt: str | None = Query(None, max_length=200),
    stt_model: str = Query(DEFAULT_WHISPER_MODEL_ID),
    translation_model: str = Query(DEFAULT_TRANSLATION_MODEL_ID),
    tts_model: str = Query(DEFAULT_TTS_MODEL_ID),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    stt_spec = _get_stage_model_spec(stt_model, "stt")
    translation_spec = _get_stage_model_spec(translation_model, "translation")
    tts_spec = _get_stage_model_spec(tts_model, "tts")

    try:
        audio = resample_audio(decode_wav(blob), target_sample_rate=16000)
        payload = localize_audio(
            audio=audio,
            threshold=threshold,
            source_language=source_language,
            target_language=target_language,
            prompt=prompt,
            speaker_prompt=speaker_prompt,
            stt_model=stt_model,
            translation_model=translation_model,
            tts_model=tts_model,
            vad_client=TritonVadClient(url=_triton_grpc_url()),
            stt_client=TritonWhisperClient(
                url=_triton_grpc_url(),
                model_name=stt_spec.repository_model_name,
            ),
            translation_client=TritonTranslationClient(
                url=_triton_grpc_url(),
                model_name=translation_spec.repository_model_name,
            ),
            tts_client=TritonTtsClient(
                url=_triton_grpc_url(),
                model_name=tts_spec.repository_model_name,
            ),
        )
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LocalizationStageError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "filename": file.filename,
                "sample_rate": audio.sample_rate,
                "duration_ms": audio.duration_ms,
                **exc.payload,
            },
        )

    return {
        "filename": file.filename,
        "sample_rate": audio.sample_rate,
        "duration_ms": audio.duration_ms,
        **payload,
    }


@app.post("/api/separate")
async def separate(file: UploadFile):
    """Audio source separation via Triton."""
    # TODO: tritonclient gRPC call
    return {"status": "not_implemented", "filename": file.filename}


@app.post("/api/vad")
async def vad(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.99),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    try:
        audio = resample_audio(decode_wav(blob), target_sample_rate=16000)
        analysis = analyze_vad(audio=audio, client=TritonVadClient(url=_triton_grpc_url()), threshold=threshold)
    except UnsupportedAudioError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TritonUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "ok",
        "filename": file.filename,
        "model": "silero_vad",
        "sample_rate": audio.sample_rate,
        "duration_ms": analysis.duration_ms,
        "threshold": analysis.threshold,
        "window_ms": round(analysis.window_ms, 2),
        "segment_count": len(analysis.segments),
        "segments": [segment.to_dict() for segment in analysis.segments],
        "window_scores": analysis.window_scores,
    }
