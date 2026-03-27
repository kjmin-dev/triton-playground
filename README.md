# triton-playground

Compliance-first speech model playground for NVIDIA Triton.

## Quick Start

```sh
bun install
bun run download:weights    # pre-download model weights (optional, cached for reuse)
bun run dev                 # Triton + worker + web — full localization pipeline
```

`bun run dev` materializes the model repository, starts Triton (via Docker if no local binary), and launches the worker and web dev servers. Pre-downloaded weights are reused from cache.

For the lightweight VAD-only mode:

```sh
bun run dev:baseline
```

## Target Models

| Stage | Model | License | Lane | Next action |
|-------|-------|---------|------|-------------|
| Separation | BS-RoFormer | code MIT, weights pending | Hold | Pin redistributable weight source. |
| VAD | Silero VAD | MIT | Auto-download + auto-serve | Keep baseline pinned and runnable. |
| STT | Whisper large-v3-turbo | MIT | Manual-download + planned-serve | Validate opt-in runtime. |
| Translation | MADLAD-400 3B | Apache 2.0 | Manual-download + planned-serve | Validate opt-in runtime. |
| TTS | CosyVoice3-0.5B | Apache 2.0 | Manual-download + planned-serve | Define voice-cloning policy. |
| TTS | Qwen3-TTS-0.6B | Apache 2.0 | Manual-download + planned-serve | Validate CustomVoice runtime. |

Detailed governance: [`docs/model-governance.md`](docs/model-governance.md)

## How It Works

1. `bun run download:weights` fetches model weights from HuggingFace into `.cache/huggingface/`
2. `bun run dev` runs `prepare_models` (materializes `model_repository/`), then starts Triton + worker + web
3. When `tritonserver` binary is not on PATH, the script falls back to Docker (`nvcr.io/nvidia/tritonserver:24.05-py3`)
4. For the localize profile, a custom Triton image with Python ML dependencies is built automatically

## Docker Compose

Full containerized stack:

```sh
docker compose up --build
```

With GPU:

```sh
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

If Docker reports `could not select device driver "nvidia"`, install the NVIDIA Container Toolkit:

```sh
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_GRPC_PORT` | `18001` | Triton gRPC port |
| `TRITON_HTTP_PORT` | `18000` | Triton HTTP port |
| `WORKER_PORT` | `8080` | Worker API port |
| `WEB_PORT` | `4000` | Web UI port |
| `TRITON_SERVER_BIN` | `tritonserver` | Path to local Triton binary |
| `TRITON_IMAGE` | `nvcr.io/nvidia/tritonserver:24.05-py3` | Base Triton Docker image |
| `TRITON_NO_GPU` | `0` | Set to `1` to skip `--gpus all` |
| `SKIP_TRITON` | `0` | Set to `1` to skip Triton startup |
| `MODEL_PROFILE` | `localize` (dev) | `baseline`, `stt`, `localize`, or `catalog` |

Default ports are `18000/18001/18002` to avoid the common `:8000` collision.

## Architecture

```
Browser
  -> Web UI (:4000)
  -> Worker API (:8080)
  -> Triton gRPC (:18001)
  -> model artifacts in model_repository/
```

## API Endpoints

### VAD — `POST /api/vad`

```sh
curl -X POST "http://localhost:8080/api/vad?threshold=0.5" -F "file=@sample.wav"
```

### STT — `POST /api/stt`

```sh
curl -X POST "http://localhost:8080/api/stt?threshold=0.5&language=ko" -F "file=@sample.wav"
```

### Localize — `POST /api/localize`

Full pipeline: Audio -> VAD -> STT -> Translation -> TTS

```sh
curl -X POST "http://localhost:8080/api/localize?threshold=0.5&target_language=ja" -F "file=@sample.wav"
```

### TTS — `POST /api/tts`

Text-only synthesis preview.

### Health — `GET /api/ready`

Returns `200` when Triton and models are ready, `503` with diagnostics otherwise.

## Scripts

| Command | Description |
|---------|-------------|
| `bun run dev` | Full pipeline: Triton (all models) + worker + web |
| `bun run dev:baseline` | Lightweight: Triton (VAD only) + worker + web |
| `bun run dev:triton` | Triton only (full pipeline) |
| `bun run dev:triton:baseline` | Triton only (VAD only) |
| `bun run dev:web` | Web only |
| `bun run dev:worker` | Worker only |
| `bun run download:weights` | Pre-download HF model weights |
| `bun run prepare:models` | Materialize full model repository |
| `bun run prepare:manual-stubs` | Generate manual Triton backend templates |
| `bun run build` | Build web app |
| `bun run check` | Build web + compile worker + run tests |
| `docker compose up --build` | Containerized full stack |

## Requirements

- Bun >= 1.0
- Python 3.10.x for worker development
- Docker (for Triton, unless `tritonserver` binary is installed locally)
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU inference)

## License

MIT
