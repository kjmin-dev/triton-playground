from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.audio import UnsupportedAudioError, decode_wav, resample_audio
from pipeline.model_catalog import get_profile_model_ids, list_model_specs
from pipeline.runtime_status import TritonReadiness, build_ready_payload
from pipeline.triton import (
    TritonUnavailableError,
    TritonVadClient,
    inspect_model_repository,
)
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


def _model_repository_root() -> str | None:
    return os.getenv("MODEL_REPOSITORY_ROOT")


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
async def tts(text: str, model: str = "cosyvoice3"):
    """Text-to-Speech via Triton."""
    # TODO: tritonclient gRPC call
    return {"status": "not_implemented", "model": model, "text": text}


@app.post("/api/stt")
async def stt(file: UploadFile, model: str = "whisper_v3_turbo"):
    """Speech-to-Text via Triton."""
    # TODO: tritonclient gRPC call
    return {"status": "not_implemented", "model": model, "filename": file.filename}


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
