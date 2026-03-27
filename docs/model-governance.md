# Model Governance

## Goal

Keep model download and server startup compliant by default.

The repo therefore separates models into three lanes:

1. `auto-download + auto-serve`
2. `manual-download + planned-serve`
3. `hold`

Only lane 1 participates in startup automation.

## Policy

1. Only download from an explicit allowlist in source control.
2. Pin every approved artifact to an immutable upstream revision.
3. Record the upstream repository, revision, license, and SHA256 in the generated manifest.
4. Do not auto-download gated, token-required, or provenance-unclear weights.
5. Do not rely on random community conversions in the auto-serve lane unless redistribution and provenance are explicitly captured.
6. Keep `trust_remote_code` out of the baseline path.
7. Make heavyweight models opt-in so `docker compose up` stays predictable.

## Current Catalog

| ID | Upstream | License | Lane | Notes |
|----|----------|---------|------|-------|
| `silero_vad` | `onnx-community/silero-vad` | MIT | auto-download + auto-serve | Triton ONNX happy path |
| `whisper_large_v3_turbo` | `openai/whisper-large-v3-turbo` | MIT | manual-download + planned-serve | likely Triton Python backend or TensorRT-LLM conversion |
| `madlad400_3b_mt` | `google/madlad400-3b-mt` | Apache 2.0 | manual-download + planned-serve | translation stage |
| `cosyvoice3_0_5b` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | Apache 2.0 | manual-download + planned-serve | TTS stage |
| `qwen3_tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Apache 2.0 | manual-download + planned-serve | low-latency TTS stage |
| `bs_roformer` | pending weight provenance review | pending | hold | do not auto-download yet |

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

## Why the Baseline Is Silero VAD

- permissive MIT license
- small artifact size
- no gated access
- easy ONNX deployment in Triton
- enough to prove the full runtime chain from download to inference

## Planned Serving Strategy

- `Whisper large-v3-turbo`
  - keep auto-download off
  - decide between TensorRT-LLM conversion and Triton Python backend after benchmarking VRAM and latency
- `MADLAD-400 3B`
  - likely Triton Python backend first
- `CosyVoice3` and `Qwen3-TTS`
  - require separate streaming contracts and voice asset policy before enabling auto-serve
- `BS-RoFormer`
  - keep on hold until the exact redistributable weight source is pinned and reviewed
