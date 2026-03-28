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

## Runtime Change Guardrails

If a task changes model serving, Triton backends, `prepare_models`, worker capability logic, or TTS actor flows, read [`docs/runtime-change-guardrails.md`](docs/runtime-change-guardrails.md) before editing.

- Treat `model_repository/` and `MANIFEST.json` as generated artifacts, not the source of truth for product behavior.
- Until a central runtime registry exists, keep `model_catalog.py`, `*_contract.py`, backend templates, `prepare_models.py`, worker capability gating, and web/API gating consistent in the same change.
- Do not assume one Qwen3-TTS checkpoint supports every TTS mode.
  - `Qwen3-TTS-12Hz-0.6B-Base` is for reference voice cloning.
  - `Qwen3-TTS-12Hz-0.6B-CustomVoice` is for preset actor preview and text-only actor TTS.
- Do not mark a runtime fix complete until the affected model is materialized, live-ready in Triton, the dependent worker endpoint has been exercised, and `bun run check` passes.

## Conventions

- **Python**: `uv`, `ruff`, `pytest`, type hints (`X | Y`)
- **Frontend**: file-based routing (`src/routes/`), shadcn Base UI Nova, `@/*` alias, biome for lint/format
- **Infra**: moonrepo, Bun, Docker Compose, Triton 24.05
- **CUDA**: host driver must support CUDA 12.4 (driver >= 550); Dockerfile pins `torch>=2.6,<2.7` with cu124

## CI & Quality Gates

Before submitting any code change, you MUST pass the full check suite. This is a hard requirement, not a suggestion.

### Check commands

```sh
# Full CI (lint + format-check + typecheck + test + build)
bun run check

# Individual checks
cd packages/worker && .venv/bin/ruff check .                  # Python lint
cd packages/worker && .venv/bin/ruff format --check .         # Python format
cd packages/worker && PYTHONPATH=. .venv/bin/python -m pytest tests/ -q  # Python tests
npx biome check packages/web/src/                             # Web lint + format
cd packages/web && bunx tsc --noEmit                          # TypeScript typecheck
cd packages/web && bun --bun vite build                       # Web build
```

### Rules

- **Always run `bun run check` before considering any task complete.** If it fails, fix the issue before moving on.
- **Never bypass git hooks** (`--no-verify`). The pre-commit and pre-push hooks exist as safety nets.
- **Python linting**: ruff with `E, F, W, I, UP, B, SIM, RUF` rule sets. Auto-fix import sorting with `ruff check --fix`.
- **Python formatting**: ruff format (double quotes, 120 char line width).
- **Web linting**: biome with the project's rule set (single quotes, trailing commas, 120 char width). Auto-fix with `npx biome check --fix`.
- **TypeScript**: strict mode, `noEmit`. All type errors must be resolved.
- **Tests**: pytest for Python, all tests must pass. Do not mark a task as done with failing tests.
- **If you add a new Python file**, ensure imports are sorted (`ruff check --fix`). If you add a new `.tsx` file, ensure biome passes.
- **Do not add `# type: ignore`, `// @ts-ignore`, `noqa`, or `biome-ignore` without a clear justification in a comment.**

### Git hooks (lefthook)

| Hook | Checks | Scope |
|------|--------|-------|
| `pre-commit` | ruff check, ruff format --check, biome check | Staged files only (fast) |
| `pre-push` | pytest, tsc --noEmit, vite build | Full project (thorough) |

### Commit conventions

- Use [Conventional Commits](https://www.conventionalcommits.org/) format: `type(scope): description`
- Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`
- Scopes: `worker`, `web`, `triton`, `infra` or omit for cross-cutting changes
- Keep commits atomic: one logical change per commit
- Write commit messages that explain *why*, not *what* (the diff shows *what*)
