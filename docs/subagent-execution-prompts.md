# Subagent Execution Prompts

## Purpose

These prompts are designed for parallel execution against the current repo state.

They assume:

- the current baseline is the compliance-first `silero_vad` path
- coordination rules are defined in `docs/parallel-execution-plan.md`
- agents are working concurrently and must not revert each other's edits

Use these prompts as the initial `message` for spawned worker or explorer agents.

## Coordinator Rules

Before spawning any worker:

1. Read `docs/parallel-execution-plan.md`.
2. Confirm the stream's dependency gate is open.
3. Tell the worker it is not alone in the codebase.
4. Tell the worker exactly which files it owns.
5. Require a handoff note with files changed, verification, and risks.

Recommended defaults:

- policy/design exploration: `explorer`, `gpt-5.4-mini`, `medium`
- bounded implementation: `worker`, `gpt-5.4-mini`, `medium`
- cross-cutting or high-risk implementation: `worker`, `gpt-5.4`, `high`

## Shared Guardrails

Include these expectations in every worker prompt:

- You are not alone in the codebase. Other agents may be editing nearby files.
- Do not revert or overwrite changes you did not make.
- Stay within your owned files unless the prompt explicitly allows a shared-file edit.
- Keep the `baseline` profile runnable.
- If you must touch a choke-point file, make the smallest coherent change.
- Leave a concise handoff note with:
  - scope completed
  - files touched
  - verification performed
  - risks and open questions
  - safe next step

## Prompt 0: Coordinator Preflight

Use this before assigning a stream if you want a fast planning pass.

**Recommended agent**

- type: `explorer`
- model: `gpt-5.4-mini`
- reasoning: `medium`

```text
Inspect the current repo state for the next parallel workstream.

Context:
- Coordination plan: docs/parallel-execution-plan.md
- Governance rules: docs/model-governance.md
- Current baseline: silero_vad compliance-first Triton happy path

Your task:
1. Confirm whether the target stream can start yet based on dependencies.
2. Identify the exact owned files and choke-point files involved.
3. List the minimum local context another worker must read before editing.
4. Call out merge risks with other streams.
5. Recommend whether this should be handled by a worker or explorer.

Return only:
- can_start: yes/no
- required_context
- owned_files
- shared_files
- main_risks
- recommended_agent_type
- recommended_model
```

## Prompt A: Baseline Runtime Hardening

**Recommended agent**

- type: `worker`
- model: `gpt-5.4`
- reasoning: `high`

**Ownership**

- Primary: `docker-compose.yml`, `packages/worker/pipeline/triton.py`, `README.md`
- Shared if necessary: `packages/worker/api/main.py`

```text
You are implementing Stream A: Baseline Runtime Hardening.

You are not alone in the codebase. Other agents may be editing nearby files. Do not revert or overwrite changes you did not make.

Read first:
- docs/parallel-execution-plan.md
- docs/model-governance.md
- docker-compose.yml
- packages/worker/pipeline/triton.py
- packages/worker/api/main.py
- README.md

Goal:
Make the existing silero_vad Triton path reliable enough to serve as the base for all later streams.

Your ownership:
- docker-compose.yml
- packages/worker/pipeline/triton.py
- README.md

Shared-file allowance:
- You may edit packages/worker/api/main.py only if strictly required for readiness or error-shape improvements.
- Keep shared-file edits minimal and local.

Tasks:
1. Harden Triton readiness and error reporting.
2. Make startup failures diagnosable when Triton is unreachable or the model is not loaded.
3. Improve compose/runtime semantics where necessary for clearer boot behavior.
4. Add or document one smoke-test path for /api/ready and /api/vad.
5. Keep the baseline workflow understandable from README.

Constraints:
- Do not change model governance policy unless strictly required; if needed, note it instead.
- Do not expand into STT, translation, or TTS.
- Keep the baseline profile runnable.

Deliverables:
- improved runtime behavior
- updated docs for smoke testing and operator diagnosis
- concise handoff note

Verification:
- run the smallest useful local checks for your changes
- include exact commands you ran
- if a full docker or GPU validation is not possible, say so explicitly

Return format:
## Handoff
- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Prompt B: Audio Ingestion and UX Robustness

**Recommended agent**

- type: `worker`
- model: `gpt-5.4-mini`
- reasoning: `medium`

**Ownership**

- Primary: `packages/worker/pipeline/audio.py`, `packages/web/src/routes/index.tsx`
- Shared if necessary: `packages/worker/api/main.py`

```text
You are implementing Stream B: Audio Ingestion and UX Robustness.

You are not alone in the codebase. Other agents may be editing nearby files. Do not revert or overwrite changes you did not make.

Read first:
- docs/parallel-execution-plan.md
- packages/worker/pipeline/audio.py
- packages/worker/api/main.py
- packages/web/src/routes/index.tsx

Goal:
Reduce sharp edges in the current baseline so the demo is not limited to pristine PCM WAV uploads and the returned VAD results are easier to inspect.

Your ownership:
- packages/worker/pipeline/audio.py
- packages/web/src/routes/index.tsx

Shared-file allowance:
- You may edit packages/worker/api/main.py only for ingestion validation or response-shape improvements that are tightly coupled to your work.
- Keep shared-file edits minimal and avoid changing readiness semantics owned by Stream A.

Tasks:
1. Improve audio validation and failure clarity.
2. Add guardrails for problematic or oversized audio.
3. Improve the web flow for threshold tuning and segment inspection.
4. If useful, add a normalization path or preparatory abstraction, but keep scope bounded to the VAD baseline.

Constraints:
- Do not add Whisper, translation, or TTS behavior.
- Do not take ownership of docker/runtime boot changes.
- Keep API changes backward-compatible where practical.

Deliverables:
- safer ingestion path
- better user-facing UI for the existing VAD workflow
- concise handoff note

Verification:
- run local checks appropriate for worker and web changes
- mention any browser or audio-format validation you could not complete

Return format:
## Handoff
- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Prompt C: Whisper STT Lane

**Recommended agent**

- type: `worker`
- model: `gpt-5.4`
- reasoning: `high`

**Ownership**

- Primary: `packages/worker/pipeline/model_catalog.py`, `packages/worker/pipeline/prepare_models.py`, new STT modules under `packages/worker/pipeline/`
- Shared if necessary: `packages/worker/api/main.py`, `packages/web/src/routes/index.tsx`

```text
You are implementing Stream C: Whisper STT Lane.

Dependency gate:
- Start only after the current governance policy for Whisper is settled enough to avoid thrashing.

You are not alone in the codebase. Other agents may be editing nearby files. Do not revert or overwrite changes you did not make.

Read first:
- docs/parallel-execution-plan.md
- docs/model-governance.md
- packages/worker/pipeline/model_catalog.py
- packages/worker/pipeline/prepare_models.py
- packages/worker/api/main.py
- packages/web/src/routes/index.tsx

Goal:
Add the second real model lane after VAD: Audio -> VAD -> STT.

Your ownership:
- packages/worker/pipeline/model_catalog.py
- packages/worker/pipeline/prepare_models.py
- new STT modules under packages/worker/pipeline/

Shared-file allowance:
- You may edit packages/worker/api/main.py to add the STT contract.
- You may edit packages/web/src/routes/index.tsx only for the minimum UI needed to exercise STT.
- Keep edits to shared files narrow and avoid folding in unrelated UX work.

Tasks:
1. Finalize a concrete Whisper serving approach for this repo.
2. Extend the model catalog and model preparation flow for Whisper in a compliant way.
3. Implement the worker-side STT path and endpoint contract.
4. Add the minimum UI to exercise and inspect STT output.
5. Update docs where the serving path and startup flow change.

Constraints:
- Stay focused on Whisper.
- Do not start translation or TTS orchestration.
- Preserve the current baseline path.

Deliverables:
- Whisper governance and preparation path
- working STT endpoint
- minimal UI path for testing STT
- concise handoff note

Verification:
- build and test what you can locally
- document any runtime assumptions, especially GPU/backend dependencies

Return format:
## Handoff
- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Prompt D: Translation and TTS Orchestration

**Recommended agent**

- type: `worker`
- model: `gpt-5.4`
- reasoning: `high`

**Ownership**

- Primary: new orchestration modules under `packages/worker/pipeline/`, `packages/web/src/routes/index.tsx`
- Shared if necessary: `packages/worker/api/main.py`, `docs/model-governance.md`

```text
You are implementing Stream D: Translation and TTS Orchestration.

Dependency gate:
- Start only after the STT contract is stable enough for downstream orchestration.

You are not alone in the codebase. Other agents may be editing nearby files. Do not revert or overwrite changes you did not make.

Read first:
- docs/parallel-execution-plan.md
- docs/model-governance.md
- packages/worker/api/main.py
- packages/web/src/routes/index.tsx
- current pipeline modules under packages/worker/pipeline/

Goal:
Turn the repo from isolated endpoints into a real localization pipeline.

Your ownership:
- new orchestration modules under packages/worker/pipeline/
- packages/web/src/routes/index.tsx for pipeline UX

Shared-file allowance:
- You may edit packages/worker/api/main.py to expose the orchestration path.
- You may edit docs/model-governance.md only where orchestration choices affect approved usage or model status.

Tasks:
1. Choose the first translation and TTS pair compatible with current policy.
2. Define job boundaries and artifact flow between stages.
3. Implement stage-by-stage orchestration from STT output to translated text to synthesized preview.
4. Add UI states for progress, stage results, and failures.

Constraints:
- Do not reopen resolved baseline runtime issues unless blocked.
- Do not weaken governance rules to make orchestration easier.
- Keep the baseline path intact.

Deliverables:
- an orchestrated pipeline contract
- one previewable output flow in the web app
- concise handoff note

Verification:
- include the exact commands and flows you used
- if full end-to-end runtime validation is blocked by environment or model availability, say exactly where it stops

Return format:
## Handoff
- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Prompt E: Governance and Expansion Policy

**Recommended agent**

- type: `worker`
- model: `gpt-5.4-mini`
- reasoning: `medium`

**Ownership**

- Primary: `docs/model-governance.md`, `packages/worker/pipeline/model_catalog.py`, `packages/worker/pipeline/prepare_models.py`
- Shared if necessary: `README.md`

```text
You are implementing Stream E: Governance and Expansion Policy.

You are not alone in the codebase. Other agents may be editing nearby files. Do not revert or overwrite changes you did not make.

Read first:
- docs/parallel-execution-plan.md
- docs/model-governance.md
- packages/worker/pipeline/model_catalog.py
- packages/worker/pipeline/prepare_models.py
- README.md

Goal:
Keep future model additions compliant, explicit, and operationally bounded.

Your ownership:
- docs/model-governance.md
- packages/worker/pipeline/model_catalog.py
- packages/worker/pipeline/prepare_models.py

Shared-file allowance:
- You may edit README.md only if policy or profile changes must be reflected there.

Tasks:
1. Tighten approval criteria for new models and artifacts.
2. Define or refine profiles beyond baseline if needed.
3. Clarify the status and next action for each target model.
4. Improve manifest or audit-trail requirements where useful.
5. Resolve or explicitly defer BS-RoFormer with clear reasoning.

Constraints:
- Do not implement full runtime support for heavyweight models in this stream.
- Keep policy language concrete and operational, not aspirational.

Deliverables:
- improved governance doc
- updated catalog and preparation metadata where needed
- concise handoff note

Verification:
- mention what changed in the governance state machine
- mention any assumptions made from upstream model availability or licensing

Return format:
## Handoff
- Scope completed:
- Files touched:
- Verification:
- Risks:
- Safe next step:
```

## Prompt F: Cross-Stream Reviewer

Use this after one or more streams land and you want a fast integration review.

**Recommended agent**

- type: `explorer`
- model: `gpt-5.4-mini`
- reasoning: `medium`

```text
Review the current repo after one or more parallel streams have landed.

Context:
- Coordination plan: docs/parallel-execution-plan.md
- Governance rules: docs/model-governance.md

Your task:
1. Identify conflicts between stream assumptions.
2. Identify choke-point files that now need consolidation.
3. List any baseline regressions or policy inconsistencies.
4. Recommend the next merge order.

Return only:
- integration_findings
- baseline_risks
- policy_conflicts
- choke_point_followups
- recommended_merge_order
```
