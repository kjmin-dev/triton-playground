# triton-playground

Compliance-first speech model playground for NVIDIA Triton.

## Quick Start

```sh
bun install                    # JS deps + moon setup (Python 3.10.12, uv)
bun run prepare:localize       # materialize model_repository/ (downloads weights on first run)
docker build \                 # build Triton dev image (one-time, cached afterwards)
  -t triton-playground-dev:localize \
  --build-arg TRITON_RUNTIME_PROFILE=localize \
  -f packages/triton/Dockerfile .
bun run dev                    # Triton (Docker) + worker + web
```

After the initial setup, only `bun run dev` is needed — the Docker image and model repository are cached.

### First-run checklist

| Step | When needed | What it does |
|------|-------------|--------------|
| `bun install` | After clone or dependency changes | Installs JS/Python deps via moonrepo |
| `bun run prepare:localize` | After clone or model changes | Downloads weights + generates Triton configs |
| `docker build ...` | After `Dockerfile` or `runtime-requirements.txt` changes | Builds Triton image with torch/transformers/qwen-tts |
| `bun run dev` | Every dev session | Starts Triton + worker + web |

### Pre-downloading weights (optional)

```sh
bun run download:weights -- --profile localize
```

Caches HF weights to `.cache/huggingface/` ahead of time. `prepare:localize` reuses this cache, making it faster on first run.

## Target Models

| Stage | Model | License | Role |
|-------|-------|---------|------|
| VAD | Silero VAD | MIT | Detect speech segments |
| STT | Whisper large-v3-turbo | MIT | Multilingual transcription (KO/EN/JA/ZH) |
| Translation | MADLAD-400 3B | Apache 2.0 | Text translation (450+ langs) |
| TTS | Qwen3-TTS-0.6B Base | Apache 2.0 | Speech synthesis with voice cloning |

Detailed governance: [`docs/model-governance.md`](docs/model-governance.md)

## How It Works

```
Browser
  → Web UI (:4000)
  → Worker API (:8080)
  → Triton gRPC (:18001)
  → model_repository/
      ├── silero_vad/          (ONNX, auto-download)
      ├── whisper_large_v3_turbo/  (Python backend)
      ├── madlad400_3b_mt/     (Python backend)
      └── qwen3_tts_0_6b/     (Python backend, voice cloning)
```

Worker orchestrates the localize pipeline:

```
Audio upload → VAD → STT → Translation → TTS (voice cloning) → Result
                                              ↑
                                    reference audio from STT segments
```

TTS automatically clones the input speaker's voice by extracting the best VAD segment as reference audio.

## API Endpoints

### Localize — `POST /api/localize`

Full pipeline: Audio → VAD → STT → Translation → TTS

```sh
curl -X POST "http://localhost:8080/api/localize?threshold=0.5&source_language=ko&target_language=en" \
  -F "file=@sample.wav"
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | `0.5` | VAD speech detection threshold |
| `source_language` | `auto` | Source language (`ko`, `en`, `ja`, `zh`, or `auto`) |
| `target_language` | required | Target language for translation + TTS |
| `speaker_prompt` | none | Speaking style instruction (fallback when no reference audio) |

**Tip:** Set `source_language` explicitly to avoid Whisper hallucination in auto-detect mode.

### VAD — `POST /api/vad`

```sh
curl -X POST "http://localhost:8080/api/vad?threshold=0.5" -F "file=@sample.wav"
```

### STT — `POST /api/stt`

```sh
curl -X POST "http://localhost:8080/api/stt?threshold=0.5&language=ko" -F "file=@sample.wav"
```

### TTS — `POST /api/tts`

Text-only synthesis preview (uses default speaker per language, no voice cloning).

### Health — `GET /api/ready`

Returns `200` when Triton and models are ready, `503` with diagnostics otherwise.

## Scripts

| Command | Description |
|---------|-------------|
| `bun run dev` | Triton + worker + web (full localize pipeline) |
| `bun run dev:web` | Web only |
| `bun run dev:worker` | Worker only |
| `bun run prepare:localize` | Materialize model_repository/ with all models |
| `bun run prepare:models -- [args]` | Custom prepare with passthrough args |
| `bun run download:weights` | Pre-download HF model weights to cache |
| `bun run build` | Build web app |
| `bun run check` | Build web + compile worker + run tests |
| `docker compose up --build` | Containerized full stack |

## Configuration

Copy `.env.example` to `.env`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRITON_GRPC_PORT` | `18001` | Triton gRPC port |
| `TRITON_HTTP_PORT` | `18000` | Triton HTTP port |
| `WORKER_PORT` | `8080` | Worker API port |
| `WEB_PORT` | `4000` | Web UI port |
| `TRITON_NO_GPU` | `0` | Set to `1` to skip `--gpus all` |
| `SKIP_TRITON` | `0` | Set to `1` to use an external Triton instance |

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

## Requirements

- Bun >= 1.0
- Python 3.10.x for worker development
- Docker (for Triton, unless `tritonserver` binary is installed locally)
- NVIDIA GPU + NVIDIA Container Toolkit (for GPU inference)
- Host CUDA driver compatible with CUDA 12.4 (driver >= 550)

## License

MIT
