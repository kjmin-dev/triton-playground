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
- `catalog`
  - the full catalog
  - emits the policy inventory and audit trail without widening automatic download behavior

## Current Catalog

| ID | Upstream | License | Lane | Next action |
|----|----------|---------|------|-------------|
| `silero_vad` | `onnx-community/silero-vad` | MIT | `auto-download + auto-serve` | Keep the baseline path pinned and runnable. |
| `whisper_large_v3_turbo` | `openai/whisper-large-v3-turbo` | MIT | `manual-download + planned-serve` | Finalize the Whisper serving backend and opt-in path in Stream C. |
| `madlad400_3b_mt` | `google/madlad400-3b-mt` | Apache 2.0 | `manual-download + planned-serve` | Choose a serving backend and resource budget. |
| `cosyvoice3_0_5b` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Apache 2.0 | `manual-download + planned-serve` | Define the voice-cloning policy and streaming contract. |
| `qwen3_tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Apache 2.0 | `manual-download + planned-serve` | Define the low-latency serving contract. |
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

## Why the Baseline Is Silero VAD

- permissive MIT license
- small artifact size
- no gated access
- easy ONNX deployment in Triton
- enough to prove the full runtime chain from download to inference

## Planned Serving Strategy

- `Whisper large-v3-turbo`
  - keep auto-download off
  - Stream C can build against the manual-download catalog entry without reopening governance
- `MADLAD-400 3B`
  - likely Triton Python backend first
- `CosyVoice3` and `Qwen3-TTS`
  - require separate streaming contracts and voice asset policy before enabling auto-serve
- `BS-RoFormer`
  - remain on hold until the exact redistributable weight source is pinned and reviewed
