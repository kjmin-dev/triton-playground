# Model Governance

## Goal

Keep model download and server startup compliant by default.

The repo separates models into three lanes:

1. `auto-download + auto-serve`
2. `manual-download + planned-serve`
3. `hold`

Only lane 1 participates in startup automation.

## Approval Criteria

A model can enter the catalog only if the source is explicit and the artifact path is bounded.

1. Pin the upstream repository and immutable revision in source control.
2. Record the upstream license and the exact artifact SHA256 when the artifact is meant to be materialized.
3. Keep `trust_remote_code` out of the baseline path.
4. Do not auto-download gated, token-required, or provenance-unclear weights.
5. Put heavyweight models in `manual-download + planned-serve` until their backend, VRAM, and startup contract are explicit.
6. Put models in `hold` when the weight source itself is not yet defensible.

## State Machine

The state machine is intentionally small:

- `auto-download + auto-serve`
  - eligible for `prepare_models`
  - must have pinned upstream revision, repository model name, and at least one checked artifact
  - must keep `approved_for_auto_download=true`
- `manual-download + planned-serve`
  - cataloged and reviewable
  - may be referenced by later streams, but is not installed by the automatic baseline materializer
  - must keep a pinned upstream repository and revision
  - should declare the expected Triton repository model name and tensor contract before worker endpoints depend on it
- `hold`
  - catalog only
  - no automatic download path
  - requires an explicit provenance decision before promotion

## Profiles

The worker materializer supports these profiles:

- `baseline`
  - `silero_vad` only
  - keeps `docker compose up` predictable
- `stt`
  - `silero_vad` plus Whisper metadata
  - useful when Stream C needs a planning manifest without changing the runtime baseline
  - emits the manual Whisper repository name and tensor contract in `MANIFEST.json`
- `catalog`
  - the full catalog
  - emits the policy inventory and audit trail without widening automatic download behavior
- `localize`
  - `silero_vad`, Whisper metadata, MADLAD metadata, and Qwen3-TTS metadata
  - useful when the operator wants one manifest plus manual backend scaffolds for the first localization flow

## Current Catalog

| ID | Upstream | License | Lane | Next action |
|----|----------|---------|------|-------------|
| `silero_vad` | `onnx-community/silero-vad` | MIT | `auto-download + auto-serve` | Keep the baseline path pinned and runnable. |
| `whisper_large_v3_turbo` | `openai/whisper-large-v3-turbo` | MIT | `manual-download + planned-serve` | Provision the manual Triton Whisper repository and validate the `/api/stt` happy path. |
| `madlad400_3b_mt` | `google/madlad400-3b-mt` | Apache 2.0 | `manual-download + planned-serve` | Provision the manual MADLAD Triton repository and validate the translation stage in `/api/localize`. |
| `cosyvoice3_0_5b` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Apache 2.0 | `manual-download + planned-serve` | Define the voice-cloning policy and streaming contract. |
| `qwen3_tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Apache 2.0 | `manual-download + planned-serve` | Provision the manual Qwen3-TTS Triton repository and validate the preview stage in `/api/localize`. |
| `bs_roformer` | pending weight provenance review | pending | `hold` | Pin the exact redistributable weight source and review provenance. |

## Startup Sequence

`docker compose up --build` runs this sequence:

1. `model-init`
2. `triton`
3. `worker`
4. `web`

`model-init` executes:

```sh
python -m pipeline.prepare_models --profile baseline --output-root /models
```

This writes:

- `/models/silero_vad/1/model.onnx`
- `/models/MANIFEST.json`

The manifest now includes the selected profile, the policy lane state machine, and the next action for each catalog entry.
When `--manual-stub-root` is provided, it also records where the generated manual Triton scaffolds were written.

## Why the Baseline Is Silero VAD

- permissive MIT license
- small artifact size
- no gated access
- easy ONNX deployment in Triton
- enough to prove the full runtime chain from download to inference

## Planned Serving Strategy

- `Whisper large-v3-turbo`
  - keep auto-download off
  - serve via a manual Triton Python backend named `whisper_large_v3_turbo`
  - worker contract is `audio_pcm`, `sample_rate`, `task`, `language`, `prompt` -> `transcript`
  - worker runs Silero VAD first and transcribes one detected speech segment at a time
- `MADLAD-400 3B`
  - selected as the first translation stage for `/api/localize`
  - manual Triton Python backend named `madlad400_3b_mt`
  - worker contract is `text`, `source_language`, `target_language` -> `translated_text`
- `Qwen3-TTS`
  - selected as the first preview TTS stage for `/api/localize` and `/api/tts`
  - manual Triton Python backend named `qwen3_tts_0_6b`
  - worker contract is `text`, `language`, `speaker_prompt` -> `audio_pcm`, `sample_rate`
- `CosyVoice3`
  - remains deferred until the voice-cloning policy and asset workflow are explicit
- `BS-RoFormer`
  - remain on hold until the exact redistributable weight source is pinned and reviewed
