# Parallel Execution Plan

## Purpose

This document is the coordination source of truth for the next stage of work.

It is written for multiple AI agents working in parallel on the same repo.
The main goal is to expand the current compliance-first `silero_vad` baseline into a broader speech pipeline without stepping on each other's file ownership.

## Current Baseline

- `silero_vad` is the only approved `auto-download + auto-serve` model.
- `model-init -> triton -> worker -> web` is the intended startup order.
- Worker exposes `GET /api/models`, `GET /api/ready`, and `POST /api/vad`.
- Web exposes a baseline upload-and-inspect UI for the VAD path.
- Governance rules live in `docs/model-governance.md`.

## Coordination Rules

1. Do not revert changes made by other agents.
2. Respect file ownership boundaries in this document.
3. If an agent must touch a file owned by another stream, it should do so only after that stream lands or after explicit coordination.
4. Every stream must leave behind verification steps and update docs if behavior changes.
5. Every new model addition must update both the governance doc and the model catalog.
6. Keep the `baseline` profile runnable while adding new capabilities.

## Execution Order

### Start Immediately

- Stream A: Baseline Runtime Hardening
- Stream B: Audio Ingestion and UX Robustness
- Stream E: Governance and Expansion Policy

### Start After Stream E Locks Policy

- Stream C: Whisper STT Lane

### Start After Stream C Locks STT Contract

- Stream D: Translation and TTS Orchestration

## File Ownership Matrix

| Stream | Primary Owner Files | Secondary Files | Notes |
|-------|----------------------|-----------------|-------|
| A | `docker-compose.yml`, `packages/worker/pipeline/triton.py`, `README.md` | `packages/worker/api/main.py` | Owns runtime boot and readiness semantics |
| B | `packages/worker/pipeline/audio.py`, `packages/web/src/routes/index.tsx` | `packages/worker/api/main.py` only if A already landed | Owns ingestion polish and UX |
| C | `packages/worker/pipeline/model_catalog.py`, `packages/worker/pipeline/prepare_models.py`, new STT pipeline modules | `packages/worker/api/main.py`, `packages/web/src/routes/index.tsx` | Owns Whisper serving lane |
| D | new orchestration modules under `packages/worker/pipeline/`, `packages/web/src/routes/index.tsx` | `packages/worker/api/main.py`, `docs/model-governance.md` | Owns end-to-end localization pipeline |
| E | `docs/model-governance.md`, `packages/worker/pipeline/model_catalog.py`, `packages/worker/pipeline/prepare_models.py` | `README.md` | Owns model approval and profile policy |

## Choke-Point Files

These files are likely to create merge conflicts and should be touched carefully:

- `packages/worker/api/main.py`
- `packages/web/src/routes/index.tsx`
- `packages/worker/pipeline/model_catalog.py`
- `packages/worker/pipeline/prepare_models.py`
- `docs/model-governance.md`

Preferred rule:

- one stream owns the file
- other streams add new modules around it
- shared-file edits happen after rebasing on the owning stream

## Stream A: Baseline Runtime Hardening

### Goal

Make the current `silero_vad` path reliable enough to be the foundation for all later streams.

### Scope

- Real Triton startup validation instead of static assumptions
- Better readiness semantics and error reporting
- Clear model load failure handling
- Smoke test path for `/api/ready` and `/api/vad`
- Better compose and local-dev parity

### Primary Files

- `docker-compose.yml`
- `packages/worker/pipeline/triton.py`
- `packages/worker/api/main.py`
- `README.md`

### Deliverables

- Runtime checks that tell the operator exactly what is wrong when Triton or the model is unavailable
- One documented smoke-test procedure
- Clean startup behavior when the model repository is missing or invalid

### Exit Criteria

- `docker compose up --build` reaches Triton-ready state in a healthy environment
- One sample WAV request returns VAD segments through the documented path

## Stream B: Audio Ingestion and UX Robustness

### Goal

Reduce sharp edges in the current baseline so the demo is not limited to pristine PCM WAV inputs.

### Scope

- Better upload validation
- More helpful error messages
- Long-audio limits and chunking behavior
- Optional normalization path if needed
- Better visual inspection of returned segments in the web UI

### Primary Files

- `packages/worker/pipeline/audio.py`
- `packages/web/src/routes/index.tsx`

### Secondary Files

- `packages/worker/api/main.py`

### Deliverables

- User-facing errors that explain unsupported audio clearly
- UI improvements for inspecting segments and thresholds
- Guardrails for audio duration and ingestion cost

### Exit Criteria

- Common audio failures are understandable
- Returned VAD segments are easier to inspect visually

## Stream C: Whisper STT Lane

### Goal

Add the second real model lane after VAD: `Audio -> VAD -> STT`.

### Scope

- Finalize Whisper serving strategy on Triton
- Extend governance and model preparation for a manual or opt-in Whisper lane
- Add worker STT pipeline modules
- Implement `/api/stt`
- Add an STT results panel to the web app

### Primary Files

- `packages/worker/pipeline/model_catalog.py`
- `packages/worker/pipeline/prepare_models.py`
- new STT modules under `packages/worker/pipeline/`

### Secondary Files

- `packages/worker/api/main.py`
- `packages/web/src/routes/index.tsx`

### Deliverables

- A documented Whisper serving choice
- Pinned provenance for the chosen Whisper artifact path
- Working STT endpoint contract

### Exit Criteria

- One documented `Audio -> VAD -> STT` happy path works
- Governance doc explains why Whisper is still manual or moved to auto-serve

## Stream D: Translation and TTS Orchestration

### Goal

Turn point endpoints into a real localization pipeline.

### Scope

- Choose the first translation and TTS pair
- Define job and artifact boundaries between stages
- Build worker orchestration from STT output to translation to synthesized preview
- Add output asset handling and UI states for stage-by-stage progress

### Primary Files

- new orchestration modules under `packages/worker/pipeline/`
- `packages/web/src/routes/index.tsx`

### Secondary Files

- `packages/worker/api/main.py`
- `docs/model-governance.md`

### Deliverables

- One end-to-end pipeline contract
- One previewable synthesized result in the UI
- Stage-level failure visibility

### Exit Criteria

- Uploaded audio can progress from transcription to translated text to synthesized preview in a documented opt-in flow

## Stream E: Governance and Expansion Policy

### Goal

Keep future model additions compliant, explicit, and operationally bounded.

### Scope

- Resolve or defer `bs_roformer` with explicit reasoning
- Define profiles beyond `baseline`
- Tighten approval criteria for new models and artifacts
- Evolve manifest metadata if needed
- Add source and weight audit trail expectations

### Primary Files

- `docs/model-governance.md`
- `packages/worker/pipeline/model_catalog.py`
- `packages/worker/pipeline/prepare_models.py`

### Deliverables

- A clear status for every target model: `auto-download`, `manual opt-in`, or `hold`
- A profile strategy that keeps startup predictable
- Clear approval criteria for future additions

### Exit Criteria

- Every target model has an explicit status, reason, and next action

## Merge Sequence

1. Merge Stream E or at least its policy decisions for Whisper before Stream C lands.
2. Merge Stream A before any stream depends on stronger readiness semantics.
3. Merge Stream B at any time, but prefer after A if API error shapes change.
4. Merge Stream C before Stream D starts integrating STT artifacts.
5. Merge Stream D last.

## Definition of Done for Every Stream

- Code compiles or builds in the current repo workflow
- New behavior is documented
- Verification steps are written down
- Ownership boundaries are not violated without explanation
- No stream weakens the `baseline` profile

## Agent Handoff Protocol

When one agent finishes a stream or a subtask, its handoff note should contain:

1. What changed
2. Exact files touched
3. Verification performed
4. Open risks or unresolved questions
5. What the next stream can assume safely

Recommended handoff format:

```md
## Handoff

- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Coordinator Checklist

- Confirm which streams are active
- Confirm file ownership boundaries before spawning workers
- Rebase or merge A before depending on readiness semantics
- Rebase or merge E before approving new models
- Rebase or merge C before starting D
- Keep this document updated when the plan changes
