from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelArtifact:
    hf_path: str
    target_name: str
    sha256: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    stage: str
    hf_repo_id: str | None
    revision: str | None
    license_name: str
    approved_for_auto_download: bool
    serve_status: str
    repository_model_name: str | None
    notes: str
    artifacts: tuple[ModelArtifact, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["artifacts"] = [asdict(artifact) for artifact in self.artifacts]
        return payload


MODEL_CATALOG: dict[str, ModelSpec] = {
    "silero_vad": ModelSpec(
        model_id="silero_vad",
        stage="vad",
        hf_repo_id="onnx-community/silero-vad",
        revision="730bca06348210595fb8cc15f28538707e58abbb",
        license_name="MIT",
        approved_for_auto_download=True,
        serve_status="triton-ready",
        repository_model_name="silero_vad",
        notes="Pinned ONNX artifact for the baseline Triton happy path.",
        artifacts=(
            ModelArtifact(
                hf_path="onnx/model.onnx",
                target_name="model.onnx",
                sha256="a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808",
            ),
        ),
    ),
    "whisper_large_v3_turbo": ModelSpec(
        model_id="whisper_large_v3_turbo",
        stage="stt",
        hf_repo_id="openai/whisper-large-v3-turbo",
        revision="f1baaf0c070fd03fc67d773bebeff75023422b6d",
        license_name="MIT",
        approved_for_auto_download=False,
        serve_status="planned",
        repository_model_name=None,
        notes="Approved for manual download only until the Triton backend strategy is finalized.",
    ),
    "madlad400_3b_mt": ModelSpec(
        model_id="madlad400_3b_mt",
        stage="translation",
        hf_repo_id="google/madlad400-3b-mt",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status="planned",
        repository_model_name=None,
        notes="Heavy model, kept out of startup automation until resource limits are locked down.",
    ),
    "cosyvoice3_0_5b": ModelSpec(
        model_id="cosyvoice3_0_5b",
        stage="tts",
        hf_repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status="planned",
        repository_model_name=None,
        notes="Manual opt-in until the voice cloning policy and backend contract are in place.",
    ),
    "qwen3_tts_0_6b": ModelSpec(
        model_id="qwen3_tts_0_6b",
        stage="tts",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status="planned",
        repository_model_name=None,
        notes="Manual opt-in only. Streaming backend design is still pending.",
    ),
    "bs_roformer": ModelSpec(
        model_id="bs_roformer",
        stage="separation",
        hf_repo_id=None,
        revision=None,
        license_name="Pending provenance review",
        approved_for_auto_download=False,
        serve_status="hold",
        repository_model_name=None,
        notes="Do not auto-download until the exact redistributable weight source is pinned and reviewed.",
    ),
}

PROFILE_MODEL_IDS: dict[str, tuple[str, ...]] = {
    "baseline": ("silero_vad",),
    "catalog": (),
}


def list_model_specs() -> list[ModelSpec]:
    return [MODEL_CATALOG[key] for key in MODEL_CATALOG]


def get_model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_CATALOG[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model id: {model_id}") from exc


def get_profile_model_ids(profile: str) -> tuple[str, ...]:
    try:
        return PROFILE_MODEL_IDS[profile]
    except KeyError as exc:
        raise KeyError(f"Unknown profile: {profile}") from exc
