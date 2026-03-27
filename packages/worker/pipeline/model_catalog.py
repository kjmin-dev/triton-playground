from __future__ import annotations

from dataclasses import asdict, dataclass

from pipeline.stt_contract import (
    DEFAULT_WHISPER_REPOSITORY_MODEL_NAME,
    WHISPER_TRITON_BACKEND,
    WHISPER_TRITON_INPUT_SPECS,
    WHISPER_TRITON_NOTES,
    WHISPER_TRITON_OUTPUT_SPECS,
)


AUTO_DOWNLOAD_LANE = "auto-download + auto-serve"
MANUAL_PLANNED_LANE = "manual-download + planned-serve"
HOLD_LANE = "hold"
VALID_SERVE_STATUSES = (AUTO_DOWNLOAD_LANE, MANUAL_PLANNED_LANE, HOLD_LANE)


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
    next_action: str
    notes: str
    artifacts: tuple[ModelArtifact, ...] = ()
    triton_backend: str | None = None
    triton_inputs: tuple[str, ...] = ()
    triton_outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.serve_status not in VALID_SERVE_STATUSES:
            raise ValueError(f"Unknown serve status for {self.model_id}: {self.serve_status}")

        if (self.triton_backend is not None or self.triton_inputs or self.triton_outputs) and self.repository_model_name is None:
            raise ValueError(f"{self.model_id} declares a Triton contract but no repository model name")

        if self.serve_status == AUTO_DOWNLOAD_LANE:
            if not self.approved_for_auto_download:
                raise ValueError(f"{self.model_id} must be approved for auto download in the auto lane")
            if self.hf_repo_id is None or self.revision is None or self.repository_model_name is None:
                raise ValueError(f"{self.model_id} is missing required auto-download metadata")
            if not self.artifacts:
                raise ValueError(f"{self.model_id} must declare at least one artifact in the auto lane")

        if self.serve_status == MANUAL_PLANNED_LANE:
            if self.approved_for_auto_download:
                raise ValueError(f"{self.model_id} cannot be auto-approved in the manual lane")
            if self.hf_repo_id is None or self.revision is None:
                raise ValueError(f"{self.model_id} must pin upstream provenance in the manual lane")

        if self.serve_status == HOLD_LANE and self.approved_for_auto_download:
            raise ValueError(f"{self.model_id} cannot be auto-approved while on hold")

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
        serve_status=AUTO_DOWNLOAD_LANE,
        repository_model_name="silero_vad",
        next_action="None; keep revision and checksum pinned while maintaining the baseline path.",
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
        serve_status=MANUAL_PLANNED_LANE,
        repository_model_name=DEFAULT_WHISPER_REPOSITORY_MODEL_NAME,
        next_action="Provision the manual Triton Whisper repository and validate the /api/stt happy path against it.",
        notes=WHISPER_TRITON_NOTES,
        triton_backend=WHISPER_TRITON_BACKEND,
        triton_inputs=WHISPER_TRITON_INPUT_SPECS,
        triton_outputs=WHISPER_TRITON_OUTPUT_SPECS,
    ),
    "madlad400_3b_mt": ModelSpec(
        model_id="madlad400_3b_mt",
        stage="translation",
        hf_repo_id="google/madlad400-3b-mt",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status=MANUAL_PLANNED_LANE,
        repository_model_name=None,
        next_action="Choose a serving backend and resource budget before enabling downloads.",
        notes="Heavy model, kept out of startup automation until resource limits are locked down.",
    ),
    "cosyvoice3_0_5b": ModelSpec(
        model_id="cosyvoice3_0_5b",
        stage="tts",
        hf_repo_id="FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status=MANUAL_PLANNED_LANE,
        repository_model_name=None,
        next_action="Define the voice-cloning policy and streaming backend contract before enabling downloads.",
        notes="Manual opt-in until the voice cloning policy and backend contract are in place.",
    ),
    "qwen3_tts_0_6b": ModelSpec(
        model_id="qwen3_tts_0_6b",
        stage="tts",
        hf_repo_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        revision="main",
        license_name="Apache-2.0",
        approved_for_auto_download=False,
        serve_status=MANUAL_PLANNED_LANE,
        repository_model_name=None,
        next_action="Define the low-latency serving contract before enabling downloads.",
        notes="Manual opt-in only. Streaming backend design is still pending.",
    ),
    "bs_roformer": ModelSpec(
        model_id="bs_roformer",
        stage="separation",
        hf_repo_id=None,
        revision=None,
        license_name="Pending provenance review",
        approved_for_auto_download=False,
        serve_status=HOLD_LANE,
        repository_model_name=None,
        next_action="Pin the exact redistributable weight source and review provenance before any download.",
        notes="Do not auto-download until the exact redistributable weight source is pinned and reviewed.",
    ),
}

_CATALOG_MODEL_IDS = tuple(MODEL_CATALOG.keys())

PROFILE_MODEL_IDS: dict[str, tuple[str, ...]] = {
    "baseline": ("silero_vad",),
    "stt": ("silero_vad", "whisper_large_v3_turbo"),
    "catalog": _CATALOG_MODEL_IDS,
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
