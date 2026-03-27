# triton-playground

PoC for serving speech and audio models on NVIDIA Triton Inference Server 24.05.

## Target Models

| Stage | Model | Role | Lane |
|-------|-------|------|------|
| Separation | BS-RoFormer | Isolate vocals from background audio | hold |
| VAD | Silero VAD | Detect speech segments | auto-download + auto-serve |
| STT | Whisper large-v3-turbo | Multilingual transcription (KO/EN/JA/ZH) | manual-download + planned-serve |
| Translation | MADLAD-400 3B | Multilingual text translation (450+ langs) | manual-download + planned-serve |
| TTS | Qwen3-TTS-0.6B Base | Speech synthesis with voice cloning | manual-download + planned-serve |

Only `silero_vad` is auto-downloaded. Whisper, MADLAD, and Qwen3-TTS weights are downloaded on demand via `bun run prepare:localize` or `bun run download:weights`. BS-RoFormer remains on hold pending provenance review.

## Monorepo

| Package | Stack | `bun run` |
|---------|-------|-----------|
| `web` | TanStack Start, React 19, shadcn (Base UI), Tailwind v4 | `dev:web` |
| `worker` | FastAPI, tritonclient[grpc], Python 3.10.12 | `dev:worker` |

## Commands

```sh
bun install                 # JS deps + moon setup (Python 3.10.12, uv)
bun run prepare:localize    # materialize model_repository/ (downloads weights + Triton configs)
bun run download:weights    # pre-download HF model weights to cache (optional, speeds up prepare)
bun run dev                 # Triton (Docker) + worker + web — full localize pipeline
bun run dev:web             # web only
bun run dev:worker          # worker only
bun run prepare:models -- [args]  # custom prepare with passthrough args
docker compose up --build   # model-init + Triton + worker + web
```

### First-run setup

```sh
bun install
bun run prepare:localize
docker build -t triton-playground-dev:localize \
  --build-arg TRITON_RUNTIME_PROFILE=localize \
  -f packages/triton/Dockerfile .
bun run dev
```

The Docker image only needs rebuilding when `packages/triton/Dockerfile` or `runtime-requirements.txt` changes. After the first build, `bun dev` reuses the cached image.

## Runtime Paths

- `MODEL_REPOSITORY_ROOT` defaults to `model_repository/`
- all model artifacts are materialized under `model_repository/`
- HF weight cache lives in `.cache/huggingface/`
- local ports and bind hosts are configured through the repo root `.env`

## Architecture

```
Browser -> TanStack Start (:WEB_PORT, default 4000)
        -> Worker API (:WORKER_PORT, default 8080)
        -> Triton gRPC (:TRITON_GRPC_PORT, default 18001)
        -> model artifacts in model_repository/
```

Worker orchestrates the localize pipeline: audio upload → VAD → STT → translation → TTS (voice cloning) → result.

TTS uses Qwen3-TTS Base model with `generate_voice_clone()` to clone the input speaker's voice. The best VAD segment is automatically selected as reference audio. Falls back to predefined speakers when no reference audio is available.

## Conventions

- **Python**: `uv`, `ruff`, `pytest`, type hints (`X | Y`)
- **Frontend**: file-based routing (`src/routes/`), shadcn Base UI Nova, `@/*` alias
- **Infra**: moonrepo, Bun, Docker Compose, Triton 24.05
- **CUDA**: host driver must support CUDA 12.4 (driver >= 550); Dockerfile pins `torch>=2.6,<2.7` with cu124
