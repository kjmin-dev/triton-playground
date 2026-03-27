from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Triton Playground Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


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
