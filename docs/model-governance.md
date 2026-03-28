# Model Governance

## Goal

Keep model download and server startup compliant by default.

The repo separates models into three lanes:

1. `auto-download + auto-serve`
2. `manual-download + planned-serve`
3. `hold`

Lane 1 participates in startup automation. Lane 2 is materialized during `bun run dev` (localize profile) or via `bun run download:weights` + `bun run prepare:models`.

## Approval Criteria

A model can enter the catalog only if the source is explicit and the artifact path is bounded.

1. Pin the upstream repository and immutable revision in source control.
2. Record the upstream license and the exact artifact SHA256 when the artifact is meant to be materialized.
3. Keep `trust_remote_code` out of the baseline path.
4. Do not auto-download gated, token-required, or provenance-unclear weights.
5. Put heavyweight models in `manual-download + planned-serve` until their backend, VRAM, and startup contract are explicit.
6. Put models in `hold` when the weight source itself is not yet defensible.

## State Machine

- `auto-download + auto-serve`
  - eligible for `prepare_models`
  - must have pinned upstream revision, repository model name, and at least one checked artifact
  - must keep `approved_for_auto_download=true`
- `manual-download + planned-serve`
  - cataloged and reviewable
  - not installed by the baseline materializer
  - materialized by the localize profile (`bun run dev` or `bun run prepare:models`)
  - weights can be pre-downloaded via `bun run download:weights`
  - must keep a pinned upstream repository and revision
  - should declare the expected Triton repository model name and tensor contract before worker endpoints depend on it
- `hold`
  - catalog only
  - no download path
  - requires an explicit provenance decision before promotion

## Profiles

- `baseline` — `silero_vad` only
- `stt` — `silero_vad` + Whisper metadata
- `localize` — `silero_vad` + Whisper + MADLAD + Qwen3-TTS (default for `bun run dev`)
- `catalog` — full catalog inventory without changing runtime behavior

## Current Catalog

| ID | Upstream | License | Lane | Next action |
|----|----------|---------|------|-------------|
| `silero_vad` | `onnx-community/silero-vad` | MIT | `auto-download + auto-serve` | Keep baseline pinned. |
| `whisper_large_v3_turbo` | `openai/whisper-large-v3-turbo` | MIT | `manual-download + planned-serve` | Validate opt-in runtime. |
| `madlad400_3b_mt` | `google/madlad400-3b-mt` | Apache 2.0 | `manual-download + planned-serve` | Validate opt-in runtime. |
| `cosyvoice3_0_5b` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Apache 2.0 | `manual-download + planned-serve` | Define voice-cloning policy. |
| `qwen3_tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Apache 2.0 | `manual-download + planned-serve` | Validate Base voice-clone runtime. |
| `qwen3_tts_0_6b_custom_voice` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | Apache 2.0 | `manual-download + planned-serve` | Materialize when preset actor TTS is needed. |
| `bs_roformer` | pending weight provenance review | pending | `hold` | Pin redistributable weight source. |

## Startup Sequence

### Local development (`bun run dev`)

1. `prepare_models --profile localize --materialize-manual-models` materializes all models into `model_repository/`
2. Triton starts (Docker fallback if no local binary; custom image built for localize profile)
3. Worker + Web start

### Docker Compose (`docker compose up --build`)

1. `model-init` runs `prepare_models`
2. `triton` boots with `model_repository/` mounted
3. `worker` connects to Triton gRPC
4. `web` serves the UI

## Weight Downloads

```sh
bun run download:weights                              # all downloadable models
bun run download:weights -- --model-id whisper_large_v3_turbo  # specific model
bun run download:weights -- --profile localize        # all localize models
bun run download:weights -- --list                    # show downloadable models
```

Weights are cached in `.cache/huggingface/` and reused by `prepare_models` during `bun run dev`.

## Why the Baseline Is Silero VAD

- permissive MIT license
- small artifact size
- no gated access
- easy ONNX deployment in Triton
- enough to prove the full runtime chain from download to inference

## Planned Serving Strategy

- **Whisper large-v3-turbo** — Triton Python backend `whisper_large_v3_turbo`, contract: `audio_pcm`, `sample_rate`, `task`, `language`, `prompt` -> `transcript`
- **MADLAD-400 3B** — Triton Python backend `madlad400_3b_mt`, contract: `text`, `source_language`, `target_language` -> `translated_text`
- **Qwen3-TTS** — Triton Python backend `qwen3_tts_0_6b`, contract: `text`, `language`, `speaker_prompt` -> `audio_pcm`, `sample_rate`
- **CosyVoice3** — deferred until voice-cloning policy is explicit
- **BS-RoFormer** — on hold until redistributable weight source is pinned
