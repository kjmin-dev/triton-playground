# Runtime Change Guardrails

## Purpose

Avoid repeated loops where a model variant, Triton contract, worker capability check, and UI gating drift apart.

This repo currently has multiple runtime touch points and does not yet have a single central runtime registry. Until that refactor exists, agents must treat cross-file consistency as mandatory.

## What Counts as Runtime Work

These rules apply when a task changes any of the following:

- `packages/worker/pipeline/model_catalog.py`
- `packages/worker/pipeline/*_contract.py`
- `packages/worker/pipeline/backend_templates/*`
- `packages/worker/pipeline/prepare_models.py`
- `packages/worker/pipeline/manual_runtime.py`
- `packages/worker/api/main.py`
- web UI or API gating that depends on model capabilities

## Source of Truth Rules

- `model_repository/` and `MANIFEST.json` are generated artifacts. Use them for diagnostics and verification, not as the primary source of product behavior.
- Live capability must come from explicit application logic plus Triton readiness, not from guessing based on one artifact.
- If a capability is user-visible, it must be declared intentionally in worker responses. Do not make the web infer it indirectly.
- If the current architecture forces the same fact to be represented in several files, update all of them in the same change and add tests that lock the relationship.

## Required Touch Points for a New or Changed Model Variant

When adding or changing a runtime variant, verify all of these:

1. `model_catalog.py`
   - pinned upstream repo and revision
   - correct lane/profile inclusion
   - notes reflect actual capability
2. `*_contract.py`
   - input and output tensor names match the runtime
   - optional inputs are explicit
3. `backend_templates/*`
   - a runnable template exists for every manual runtime model
   - capability-specific code paths fail clearly when unsupported
4. `prepare_models.py` and `manual_runtime.py`
   - the model can be materialized
   - profile selection and manual materialization both work
5. worker capability logic
   - availability is based on explicit logic plus live Triton readiness
   - product behavior does not depend only on generated artifacts
6. web or API gating
   - UI state uses explicit worker capability responses
   - no hidden assumptions about checkpoint behavior
7. tests
   - model catalog coverage
   - manual runtime materialization coverage
   - capability or endpoint contract coverage

## TTS-Specific Rules

- `Qwen3-TTS-12Hz-0.6B-Base` is for reference voice cloning. It requires reference audio.
- `Qwen3-TTS-12Hz-0.6B-CustomVoice` is for preset actor preview and text-only actor TTS.
- Do not silently conflate Base and CustomVoice behavior.
- If a worker endpoint can route to different TTS runtimes, make the selection rules explicit and user-visible.
- If Triton input contracts change, update the worker client, backend template, and tests together.

## Verification Checklist

Do not mark runtime work complete until all relevant items below are done:

1. materialize the affected model or profile
   - example: `moon run worker:prepare-models -- --profile localize --model-id <id> --materialize-manual-models`
2. verify the generated runtime artifacts exist
   - `model_repository/<model>/config.pbtxt`
   - `model_repository/<model>/1/model.py` for Python backend models
3. verify live Triton readiness for the affected model
4. hit the worker endpoint that depends on the model
5. if a web flow changed, exercise the API response that gates the UI
6. run `bun run check`

## Forbidden Shortcuts

- Do not infer live runtime availability from only `MANIFEST.json`.
- Do not infer live runtime availability from only filesystem presence.
- Do not change web gating without changing the worker capability response that drives it.
- Do not add a new Triton input without verifying the materialized runtime and worker client agree.
- Do not close a bug as fixed based only on unit tests when the failure involved live runtime wiring.

## Escalation Rule

If the same class of mistake appears twice, stop adding local fallbacks and strengthen the invariant instead. That usually means:

- adding a missing capability response
- adding a missing materialization test
- tightening the runtime selection logic
- documenting the constraint in `AGENTS.md` or `CLAUDE.md`
