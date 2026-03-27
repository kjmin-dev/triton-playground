# triton-playground

Compliance-first speech model playground for NVIDIA Triton.

The repo now has a concrete baseline happy path:

1. download an approved model artifact at a pinned revision
2. materialize a Triton model repository
3. boot Triton
4. upload a WAV file
5. receive speech segments from Silero VAD over Triton gRPC

Future stages keep the same download and provenance policy for STT, translation, and TTS.

## Target Models

| Stage | Model | License | Auto-download | Serve status |
|-------|-------|---------|---------------|--------------|
| Separation | BS-RoFormer | code MIT, weights pending provenance review | No | Hold |
| VAD | Silero VAD | MIT | Yes | Triton happy path complete |
| STT | Whisper large-v3-turbo | MIT | Manual opt-in | Planned |
| Translation | MADLAD-400 3B | Apache 2.0 | Manual opt-in | Planned |
| TTS | CosyVoice3-0.5B | Apache 2.0 | Manual opt-in | Planned |
| TTS | Qwen3-TTS-0.6B | Apache 2.0 | Manual opt-in | Planned |

Only approved models can enter the runtime download lane. The default `baseline` profile downloads `silero_vad` only.

Detailed governance and startup design: [`docs/model-governance.md`](docs/model-governance.md)

## Quick Start

```sh
bun install
cp .env.example .env
bun run prepare:models
bun run dev:worker
```

The worker expects Triton at `localhost:8001`. For the full happy path with GPU:

```sh
docker compose up --build
```

Open:

- web: http://localhost:3000
- worker: http://localhost:8080
- triton metrics: http://localhost:8002/metrics

## Architecture

```
Browser
  -> Web UI (:3000)
  -> Worker API (:8080)
  -> Triton gRPC (:8001)
  -> approved model artifact in model_repository/
```

`model-init` runs before Triton and populates the repository from pinned Hugging Face revisions. Nothing is committed to git except the empty repository directory and source code.

## Baseline Happy Path

- input: PCM WAV upload
- model: `silero_vad`
- output: speech segments with probability scores
- endpoint: `POST /api/vad`

Example:

```sh
curl -X POST "http://localhost:8080/api/vad?threshold=0.5" \
  -F "file=@sample.wav"
```

## Scripts

| Command | Description |
|---------|-------------|
| `bun run dev` | Start all dev servers |
| `bun run dev:web` | Web only |
| `bun run dev:worker` | Worker only |
| `bun run prepare:models` | Download the approved baseline model into `model_repository/` |
| `bun run build` | Build the web app |
| `bun run check` | Build web + compile worker + run stdlib tests |
| `docker compose up --build` | `model-init` + Triton + worker + web |

## Requirements

- Bun >= 1.0
- Python 3.10.x for worker development
- NVIDIA GPU + Docker for Triton runtime

## License

MIT
