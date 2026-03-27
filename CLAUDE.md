# triton-playground

PoC for serving speech & audio AI models on NVIDIA Triton Inference Server 24.05.

## Target Models

| Stage | Model | Role |
|-------|-------|------|
| Separation | BS-RoFormer | Isolate vocals from background audio |
| VAD | Silero VAD | Detect speech segments |
| STT | Whisper large-v3-turbo | Multilingual transcription (KO/EN/JA/ZH) |
| Translation | MADLAD-400 3B | Multilingual text translation (450+ langs) |
| TTS | CosyVoice3-0.5B | Speech synthesis with voice cloning |
| TTS | Qwen3-TTS-0.6B | Low-latency speech synthesis |

All models: open-source, permissive license (MIT/Apache 2.0), downloaded from HuggingFace at runtime.

## Monorepo

| Package | Stack | `bun run` |
|---------|-------|-----------|
| `web` | TanStack Start, React 19, shadcn (Base UI), Tailwind v4 | `dev:web` |
| `worker` | FastAPI, tritonclient[grpc], Python 3.10.12 | `dev:worker` |

## Commands

```sh
bun install           # JS deps + moon setup (Python 3.10.12, uv)
bun run dev           # all dev servers
bun run dev:web       # web → localhost:4000 by default, configurable via WEB_PORT
bun run dev:worker    # worker → localhost:8080 by default, configurable via WORKER_PORT
docker compose up     # full stack with GPU (Triton + worker + web)
```

## Architecture

```
Browser → TanStack Start → Worker API (FastAPI :WORKER_PORT, default 8080) → Triton gRPC (:TRITON_GRPC_PORT, default 8001) → GPU
```

Worker orchestrates the pipeline: audio upload → separation → STT → translation → TTS → result.

## Conventions

- **Python**: `uv`, `ruff`, `pytest`, type hints (`X | Y`)
- **Frontend**: file-based routing (`src/routes/`), shadcn Base UI Nova, `@/*` alias
- **Infra**: moonrepo, Bun, Docker Compose, Triton 24.05
