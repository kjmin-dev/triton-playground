from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
from pathlib import Path
from typing import TypeVar

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from pipeline.audio import UnsupportedAudioError, decode_wav, resample_audio
from pipeline.localization import (
    LocalizationStageError,
    TritonLocalizeTextPipelineClient,
    localize_audio,
)
from pipeline.model_catalog import get_model_spec, get_profile_model_ids, list_model_specs
from pipeline.runtime_status import TritonReadiness, build_ready_payload
from pipeline.stt import (
    TranscribedSegment,
    TritonWhisperClient,
    TritonWhisperSttPipelineClient,
    analyze_stt,
)
from pipeline.stt_contract import DEFAULT_WHISPER_MODEL_ID, DEFAULT_WHISPER_REPOSITORY_MODEL_NAME
from pipeline.subtitles import segments_to_csv, segments_to_srt, segments_to_vtt
from pipeline.translation import TritonTranslationClient
from pipeline.translation_contract import DEFAULT_TRANSLATION_MODEL_ID, DEFAULT_TRANSLATION_REPOSITORY_MODEL_NAME
from pipeline.triton import (
    TritonUnavailableError,
    TritonVadClient,
    TritonVadStreamingClient,
    inspect_model_repository,
)
from pipeline.tts import TritonTtsClient, encode_wav_preview, validate_tts_language
from pipeline.tts_contract import DEFAULT_TTS_MODEL_ID
from pipeline.vad import analyze_vad

logger = logging.getLogger(__name__)

app = FastAPI(title="Triton Playground Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

T = TypeVar("T")


class ClientDisconnectedError(Exception):
    pass


async def _run_cancellable(request: Request, fn, *args, **kwargs) -> T:
    """Run a blocking function in a thread pool. Abort if the client disconnects.

    This keeps the event loop free so lightweight endpoints like /api/ready
    remain responsive while heavy GPU work is in progress.
    """
    work = asyncio.get_event_loop().run_in_executor(None, functools.partial(fn, *args, **kwargs))

    async def _poll_disconnect():
        while not await request.is_disconnected():
            await asyncio.sleep(0.5)

    work_task = asyncio.ensure_future(work)
    disconnect_task = asyncio.ensure_future(_poll_disconnect())

    done, pending = await asyncio.wait(
        {work_task, disconnect_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for p in pending:
        p.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await p

    if disconnect_task in done:
        logger.info("Client disconnected, aborting request")
        raise ClientDisconnectedError()

    return work_task.result()


def _triton_grpc_url() -> str:
    return os.getenv("TRITON_GRPC_URL", "localhost:18001")


def _model_profile() -> str:
    return os.getenv("MODEL_PROFILE", "baseline")


def _default_model_repository_root() -> str:
    return str(Path(__file__).resolve().parents[3] / "model_repository")


def _model_repository_root() -> str:
    return os.getenv("MODEL_REPOSITORY_ROOT") or _default_model_repository_root()


@functools.lru_cache(maxsize=4)
def _cached_vad_client(url: str) -> TritonVadClient | TritonVadStreamingClient:
    """Return streaming VAD client if available, else fall back to per-window."""
    try:
        streaming = TritonVadStreamingClient(url=url)
        readiness = streaming.readiness()
        if readiness.ready:
            logger.info("Using streaming VAD client (single gRPC call)")
            return streaming
    except TritonUnavailableError:
        pass
    return TritonVadClient(url=url)


@functools.lru_cache(maxsize=4)
def _cached_stt_client(url: str, model_name: str):
    if model_name == DEFAULT_WHISPER_REPOSITORY_MODEL_NAME:
        try:
            pipeline_client = TritonWhisperSttPipelineClient(url=url)
            readiness = pipeline_client.readiness()
            if readiness.ready:
                logger.info("Using Triton STT pipeline client (VAD + Whisper in one gRPC call)")
                return pipeline_client
        except TritonUnavailableError:
            pass
    return TritonWhisperClient(url=url, model_name=model_name)


@functools.lru_cache(maxsize=4)
def _cached_translation_client(url: str, model_name: str) -> TritonTranslationClient:
    return TritonTranslationClient(url=url, model_name=model_name)


@functools.lru_cache(maxsize=4)
def _cached_tts_client(url: str, model_name: str) -> TritonTtsClient:
    return TritonTtsClient(url=url, model_name=model_name)


@functools.lru_cache(maxsize=4)
def _cached_localize_text_client(url: str):
    try:
        client = TritonLocalizeTextPipelineClient(url=url)
        readiness = client.readiness()
        if readiness.ready:
            logger.info("Using Triton localize text pipeline client (STT + translation in one gRPC call)")
            return client
    except TritonUnavailableError:
        pass
    return None


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
    """Lightweight readiness probe — runs in the event loop, never blocked by GPU work."""
    readiness = await asyncio.to_thread(
        lambda: _check_readiness(),
    )
    return readiness


def _check_readiness():
    repository_status = inspect_model_repository(
        repository_root=_model_repository_root(),
        model_name="silero_vad",
    )

    try:
        triton_readiness = _cached_vad_client(_triton_grpc_url()).readiness(refresh=True)
    except TritonUnavailableError as exc:
        triton_readiness = TritonReadiness.from_error(
            server_url=_triton_grpc_url(),
            model_name="silero_vad",
            issue=str(exc),
        )

    payload = build_ready_payload(_model_profile(), triton_readiness)
    payload["repository"] = repository_status.to_dict()
    status_code = 200 if triton_readiness.ready else 503
    return JSONResponse(status_code=status_code, content=payload)


@app.post("/api/tts")
async def tts(
    request: Request,
    text: str = Query(..., min_length=1, max_length=5000),
    language: str = Query(...),
    prompt: str | None = Query(None, max_length=200),
    model: str = Query(DEFAULT_TTS_MODEL_ID),
):
    model_spec = _get_stage_model_spec(model, "tts")

    def _work():
        normalized_language = validate_tts_language(language)
        synthesized = _cached_tts_client(
            _triton_grpc_url(),
            model_spec.repository_model_name,
        ).synthesize(
            text,
            language=normalized_language,
            speaker_prompt=prompt,
        )
        return normalized_language, synthesized

    try:
        normalized_language, synthesized = await _run_cancellable(request, _work)
    except ClientDisconnectedError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
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
    request: Request,
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
    audio = resample_audio(decode_wav(blob), target_sample_rate=16000)

    def _work():
        return analyze_stt(
            audio=audio,
            vad_client=_cached_vad_client(_triton_grpc_url()),
            stt_client=_cached_stt_client(_triton_grpc_url(), model_spec.repository_model_name),
            threshold=threshold,
            language=language,
            task=task,
            prompt=prompt,
        )

    try:
        analysis = await _run_cancellable(request, _work)
    except ClientDisconnectedError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
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
    request: Request,
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

    audio = resample_audio(decode_wav(blob), target_sample_rate=16000)

    def _work():
        localize_text_client = None
        if (
            stt_spec.repository_model_name == DEFAULT_WHISPER_REPOSITORY_MODEL_NAME
            and translation_spec.repository_model_name == DEFAULT_TRANSLATION_REPOSITORY_MODEL_NAME
        ):
            localize_text_client = _cached_localize_text_client(_triton_grpc_url())

        return localize_audio(
            audio=audio,
            threshold=threshold,
            source_language=source_language,
            target_language=target_language,
            prompt=prompt,
            speaker_prompt=speaker_prompt,
            stt_model=stt_model,
            translation_model=translation_model,
            tts_model=tts_model,
            localize_text_client=localize_text_client,
            vad_client=_cached_vad_client(_triton_grpc_url()),
            stt_client=_cached_stt_client(_triton_grpc_url(), stt_spec.repository_model_name),
            translation_client=_cached_translation_client(_triton_grpc_url(), translation_spec.repository_model_name),
            tts_client=_cached_tts_client(_triton_grpc_url(), tts_spec.repository_model_name),
        )

    try:
        payload = await _run_cancellable(request, _work)
    except ClientDisconnectedError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
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
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.1, le=0.99),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    blob = await file.read()
    if not blob:
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    audio = resample_audio(decode_wav(blob), target_sample_rate=16000)

    def _work():
        return analyze_vad(audio=audio, client=_cached_vad_client(_triton_grpc_url()), threshold=threshold)

    try:
        analysis = await _run_cancellable(request, _work)
    except ClientDisconnectedError:
        return JSONResponse(status_code=499, content={"detail": "Client disconnected"})
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


def _parse_segments_from_json(data: dict) -> list[TranscribedSegment]:
    """Extract TranscribedSegment list from a localize/stt response payload."""
    raw_segments = []
    if "stages" in data and "stt" in data["stages"]:
        raw_segments = data["stages"]["stt"].get("segments", [])
    elif "segments" in data:
        raw_segments = data["segments"]

    return [
        TranscribedSegment(
            start_ms=s["start_ms"],
            end_ms=s["end_ms"],
            duration_ms=s["duration_ms"],
            average_probability=s.get("average_probability", 0),
            peak_probability=s.get("peak_probability", 0),
            text=s.get("text", ""),
            speaker_id=s.get("speaker_id"),
        )
        for s in raw_segments
    ]


@app.post("/api/subtitles/{fmt}")
async def subtitles(
    fmt: str,
    request: Request,
):
    """Generate subtitles from a JSON payload containing segments.

    POST the localize or STT response body as JSON. Supported formats: srt, vtt, csv.
    """
    if fmt not in ("srt", "vtt", "csv"):
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}. Use srt, vtt, or csv.")

    body = await request.json()
    segments = _parse_segments_from_json(body)
    if not segments:
        raise HTTPException(status_code=400, detail="No segments found in request body.")

    generators = {"srt": segments_to_srt, "vtt": segments_to_vtt, "csv": segments_to_csv}
    content_types = {
        "srt": "text/plain; charset=utf-8",
        "vtt": "text/vtt; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
    }

    text = generators[fmt](segments)
    return PlainTextResponse(
        content=text,
        media_type=content_types[fmt],
        headers={"Content-Disposition": f'attachment; filename="subtitles.{fmt}"'},
    )
