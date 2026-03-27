# triton-playground

PoC for serving speech and audio models on NVIDIA Triton Inference Server 24.05.

## Target Models

| Stage | Model | Role | Lane |
|-------|-------|------|------|
| Separation | BS-RoFormer | Isolate vocals from background audio | hold |
| VAD | Silero VAD | Detect speech segments | auto-download + auto-serve |
| STT | Whisper large-v3-turbo | Multilingual transcription (KO/EN/JA/ZH) | manual-download + planned-serve |
| Translation | MADLAD-400 3B | Multilingual text translation (450+ langs) | manual-download + planned-serve |
| TTS | CosyVoice3-0.5B | Speech synthesis with voice cloning | manual-download + planned-serve |
| TTS | Qwen3-TTS-0.6B | Low-latency speech synthesis | manual-download + planned-serve |

Only `silero_vad` is auto-downloaded. Whisper, MADLAD, and Qwen3-TTS weights are downloaded on demand via `bun run download:weights` or during `bun run dev`. BS-RoFormer remains on hold pending provenance review.

## Monorepo

| Package | Stack | `bun run` |
|---------|-------|-----------|
| `web` | TanStack Start, React 19, shadcn (Base UI), Tailwind v4 | `dev:web` |
| `worker` | FastAPI, tritonclient[grpc], Python 3.10.12 | `dev:worker` |

## Commands

```sh
bun install                 # JS deps + moon setup (Python 3.10.12, uv)
bun run download:weights    # pre-download HF model weights (optional, speeds up first dev)
bun run dev                 # Triton (Docker fallback) + worker + web — full pipeline
bun run dev:baseline        # Triton (silero_vad only) + worker + web — lightweight
bun run dev:web             # web only
bun run dev:worker          # worker only
bun run prepare:models      # materialize localize model_repository/
bun run prepare:manual-stubs # scaffold manual Triton backends into manual_model_stubs/
docker compose up --build   # model-init + Triton + worker + web
```

`bun run dev` defaults to the full localize profile (VAD + Whisper + MADLAD + TTS). When `tritonserver` is not on PATH, it falls back to Docker automatically. Set `SKIP_TRITON=1` to use an already-running Triton instance.

## Runtime Paths

- `MODEL_REPOSITORY_ROOT` defaults to `model_repository/`
- all model artifacts are materialized under `model_repository/`
- manual backend templates (reference only) go to `manual_model_stubs/`
- HF weight cache lives in `.cache/huggingface/`
- local ports and bind hosts are configured through the repo root `.env`

## Architecture

```
Browser -> TanStack Start (:WEB_PORT, default 4000)
        -> Worker API (:WORKER_PORT, default 8080)
        -> Triton gRPC (:TRITON_GRPC_PORT, default 18001)
        -> model artifacts in model_repository/
```

Worker orchestrates the pipeline: audio upload -> VAD -> STT -> translation -> TTS -> result.

## Conventions

- **Python**: `uv`, `ruff`, `pytest`, type hints (`X | Y`)
- **Frontend**: file-based routing (`src/routes/`), shadcn Base UI Nova, `@/*` alias
- **Infra**: moonrepo, Bun, Docker Compose, Triton 24.05
