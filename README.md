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

| Stage | Model | License | Governance lane | Next action |
|-------|-------|---------|-----------------|-------------|
| Separation | BS-RoFormer | code MIT, weights pending provenance review | Hold | Pin the exact redistributable weight source and review provenance. |
| VAD | Silero VAD | MIT | Auto-download + auto-serve | Keep the baseline path pinned and runnable. |
| STT | Whisper large-v3-turbo | MIT | Manual-download + planned-serve | Finalize the Whisper backend contract in Stream C. |
| Translation | MADLAD-400 3B | Apache 2.0 | Manual-download + planned-serve | Choose a serving backend and resource budget. |
| TTS | CosyVoice3-0.5B | Apache 2.0 | Manual-download + planned-serve | Define the voice-cloning policy and streaming backend contract. |
| TTS | Qwen3-TTS-0.6B | Apache 2.0 | Manual-download + planned-serve | Define the low-latency serving contract. |

Only models in the `auto-download + auto-serve` lane can enter runtime download automation. The default `baseline` profile downloads `silero_vad` only.
The `stt` profile records `silero_vad` plus the Whisper metadata for planning, and `catalog` emits the full catalog without changing the baseline runtime behavior.

Detailed governance and startup design: [`docs/model-governance.md`](docs/model-governance.md)

## Quick Start

```sh
bun install
bun run prepare:models
docker compose up --build
```

`bun install` runs `scripts/setup.sh`, which installs the pinned moon Python/uv toolchains, updates your shell PATH for proto shims, and bootstraps + syncs `packages/worker/.venv` with Python `3.10.12`.

All local bind settings live in the repo root `.env` file. Copy `.env.example` to `.env` and change `WEB_HOST`, `WEB_PORT`, `WORKER_HOST`, `WORKER_PORT`, `TRITON_HTTP_PORT`, `TRITON_GRPC_PORT`, and `TRITON_METRICS_PORT` there instead of editing code.

To inspect the policy catalog without changing the baseline runtime set:

```sh
bun run prepare:models --profile catalog
```

To stage the Whisper planning manifest without enabling automatic download:

```sh
bun run prepare:models --profile stt
```

The `stt` manifest records the manual Whisper repository name and tensor contract so the worker and operator agree on the same Triton interface before any weights are mounted.

To scaffold the first localization backends without putting half-configured models into `model_repository/`:

```sh
bun run prepare:manual-stubs
```

That keeps the runnable baseline under `model_repository/` and writes manual Triton templates for Whisper, MADLAD, and Qwen3-TTS into `manual_model_stubs/`.

For local worker development against an already-running Triton instance:

```sh
TRITON_GRPC_URL=localhost:8001 MODEL_REPOSITORY_ROOT=model_repository bun run dev:worker
```

Open:

- web: http://localhost:4000
- worker: http://localhost:8080
- triton metrics: http://localhost:8002/metrics

Those URLs are the defaults from `.env.example`; they are configurable through `.env`.
For LAN access during development, keep `WEB_HOST=0.0.0.0` and `WORKER_HOST=0.0.0.0`, then open `http://<your-machine-ip>:4000`.

## Architecture

```
Browser
  -> Web UI (:WEB_PORT, default 4000)
  -> Worker API (:WORKER_PORT, default 8080)
  -> Triton gRPC (:TRITON_GRPC_PORT, default 8001)
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

## Smoke Test

Use this after `docker compose up --build` to confirm the baseline runtime path:

```sh
curl -s http://localhost:8080/api/ready
curl -s -X POST "http://localhost:8080/api/vad?threshold=0.5" \
  -F "file=@sample.wav"
```

Expected behavior:

- `/api/ready` returns `200` with `"status": "ready"` when Triton and `silero_vad` are ready
- `/api/ready` returns `503` with diagnostic `triton.summary`, `triton.issues`, and `repository.*` details when Triton is unreachable or the model failed to load
- `/api/vad` returns speech segments only after the readiness check is healthy

## Whisper STT Lane

The second lane is manual by design. The worker exposes `POST /api/stt`, but the operator must provision a Triton Python backend named `whisper_large_v3_turbo` first.

Expected Triton contract:

- inputs: `audio_pcm`, `sample_rate`, `task`, `language`, `prompt`
- output: `transcript`
- audio shape: mono 16 kHz PCM in `FP32`

The worker runs Silero VAD first, then sends each detected speech segment to Whisper and concatenates the returned text.

Example:

```sh
curl -X POST "http://localhost:8080/api/stt?threshold=0.5&language=ko" \
  -F "file=@sample.wav"
```

If the Whisper repository is missing from Triton, `/api/stt` returns `503` with the Triton readiness summary instead of silently falling back.

## Localization Preview

The first end-to-end localization pair is:

- translation: `madlad400_3b_mt`
- TTS: `qwen3_tts_0_6b`

Both stay in the manual lane. The worker exposes:

- `POST /api/localize` for `Audio -> VAD -> STT -> Translation -> TTS`
- `POST /api/tts` for text-only synthesized preview

`/api/localize` returns stage-level status plus a browser-playable WAV preview when all manual backends are mounted in Triton.

`bun run prepare:manual-stubs` generates:

- `manual_model_stubs/whisper_large_v3_turbo/`
- `manual_model_stubs/madlad400_3b_mt/`
- `manual_model_stubs/qwen3_tts_0_6b/`

Each scaffold contains:

- `README.md` with upstream revision and bring-up steps
- `config.pbtxt.template`
- `1/model.py.template`

These are templates only. They are intentionally outside the live `model_repository/` tree so Triton does not try to load incomplete manual backends.

Example:

```sh
curl -X POST "http://localhost:8080/api/localize?threshold=0.5&target_language=ja" \
  -F "file=@sample.wav"
```

Expected manual Triton contracts:

- `madlad400_3b_mt`: `text`, `source_language`, `target_language` -> `translated_text`
- `qwen3_tts_0_6b`: `text`, `language`, `speaker_prompt` -> `audio_pcm`, `sample_rate`

If a downstream manual backend is unavailable, the response keeps the completed upstream stage results and marks the failing stage explicitly.

## Scripts

| Command | Description |
|---------|-------------|
| `bun run dev` | Start all dev servers |
| `bun run dev:web` | Web only |
| `bun run dev:worker` | Worker only |
| `bun run prepare:models` | Download the approved baseline model into `model_repository/` |
| `bun run prepare:manual-stubs` | Generate manual Triton backend templates for the first localization flow |
| `bun run build` | Build the web app |
| `bun run check` | Build web + compile worker + run stdlib tests |
| `docker compose up --build` | `model-init` + Triton + worker + web |

## Requirements

- Bun >= 1.0
- Python 3.10.x for worker development
- NVIDIA GPU + Docker for Triton runtime

## License

MIT
