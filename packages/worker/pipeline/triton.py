from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pipeline.runtime_status import TritonReadiness

if TYPE_CHECKING:
    import numpy as np


class TritonUnavailableError(RuntimeError):
    pass


def describe_triton_error(*, url: str, action: str, exc: Exception) -> str:
    detail = str(exc).strip()
    normalized = detail.lower()

    if "connection refused" in normalized or "failed to connect to all addresses" in normalized:
        hostname = urlsplit(f"grpc://{url}").hostname or url
        location_hint = (
            "No Triton server is listening on the local gRPC port."
            if hostname in {"localhost", "127.0.0.1", "::1"}
            else f"Triton is not accepting connections at {url}."
        )
        return (
            f"{action} at {url} failed because the connection was refused. "
            f"{location_hint} Start Triton first with `docker compose up --build` "
            "or point TRITON_GRPC_URL at a running Triton gRPC endpoint. "
            f"Original error: {detail}"
        )

    if "name resolution" in normalized or "dns" in normalized or "no address associated with hostname" in normalized:
        return (
            f"{action} at {url} failed because the Triton hostname could not be resolved. "
            "Verify TRITON_GRPC_URL points to a reachable host:port. "
            f"Original error: {detail}"
        )

    return f"{action} at {url} failed: {detail}"


@dataclass(frozen=True)
class ModelRepositoryStatus:
    model_name: str
    root_path: str | None
    manifest_path: str | None
    configured: bool
    manifest_present: bool
    model_directory_present: bool
    model_version_present: bool
    model_artifact_present: bool
    profile: str | None
    selected_model_ids: tuple[str, ...]
    issues: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def status(self) -> str:
        if not self.configured:
            return "not_configured"

        return "ready" if not self.issues else "unavailable"

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def summary(self) -> str:
        if self.status == "not_configured":
            return "Repository diagnostics are disabled because MODEL_REPOSITORY_ROOT is not set."

        if self.ready:
            return (
                f"Model repository at {self.root_path} contains a manifest and artifact for {self.model_name}."
            )

        return " ".join(self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ready": self.ready,
            "summary": self.summary,
            "model_name": self.model_name,
            "root_path": self.root_path,
            "manifest_path": self.manifest_path,
            "configured": self.configured,
            "manifest_present": self.manifest_present,
            "model_directory_present": self.model_directory_present,
            "model_version_present": self.model_version_present,
            "model_artifact_present": self.model_artifact_present,
            "profile": self.profile,
            "selected_model_ids": list(self.selected_model_ids),
            "issues": list(self.issues),
            "warnings": list(self.warnings),
        }


def _model_index_entry_name(entry: object) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("name")
        return str(value) if value is not None else None

    value = getattr(entry, "name", None)
    return str(value) if value is not None else None


def _model_index_entry_state(entry: object) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("state")
        return str(value) if value is not None else None

    value = getattr(entry, "state", None)
    return str(value) if value is not None else None


def check_readiness(client: object, url: str, model_name: str) -> "TritonReadiness":
    """Shared readiness check used by all Triton client classes."""
    try:
        model_present: bool | None = None
        model_state: str | None = None
        get_model_repository_index = getattr(client, "get_model_repository_index", None)
        if callable(get_model_repository_index):
            try:
                model_index = get_model_repository_index()
            except Exception:
                model_index = None

            if model_index is not None:
                model_present = False
                for entry in model_index.models:
                    if _model_index_entry_name(entry) == model_name:
                        model_present = True
                        model_state = _model_index_entry_state(entry)
                        break

        return TritonReadiness.from_status(
            server_url=url,
            server_ready=bool(client.is_server_ready()),
            server_live=bool(client.is_server_live()),
            model_ready=bool(client.is_model_ready(model_name)),
            model_present=model_present,
            model_state=model_state,
            model_name=model_name,
        )
    except Exception as exc:
        raise TritonUnavailableError(
            describe_triton_error(url=url, action="Querying Triton readiness", exc=exc)
        ) from exc


def inspect_model_repository(
    repository_root: str | None,
    model_name: str = "silero_vad",
) -> ModelRepositoryStatus:
    if not repository_root:
        return ModelRepositoryStatus(
            model_name=model_name,
            root_path=None,
            manifest_path=None,
            configured=False,
            manifest_present=False,
            model_directory_present=False,
            model_version_present=False,
            model_artifact_present=False,
            profile=None,
            selected_model_ids=(),
            issues=(),
            warnings=("MODEL_REPOSITORY_ROOT is not set.",),
        )

    root = Path(repository_root)
    manifest_path = root / "MANIFEST.json"
    model_directory = root / model_name
    model_version = model_directory / "1"
    model_artifact = model_version / "model.onnx"
    issues: list[str] = []
    warnings: list[str] = []
    profile: str | None = None
    selected_model_ids: tuple[str, ...] = ()
    manifest_present = manifest_path.is_file()

    if not root.exists():
        issues.append(f"Model repository root {root} does not exist. Run prepare_models before starting Triton.")
    elif not root.is_dir():
        issues.append(f"Model repository root {root} is not a directory.")

    if root.exists() and not manifest_present:
        issues.append(f"Manifest file {manifest_path} is missing. Run prepare_models to populate the repository.")

    if root.exists() and not model_directory.is_dir():
        issues.append(f"Model directory {model_directory} is missing.")

    if model_directory.is_dir() and not model_version.is_dir():
        issues.append(f"Model version directory {model_version} is missing.")

    if model_version.is_dir() and not model_artifact.is_file():
        issues.append(f"Expected model artifact {model_artifact} is missing.")

    if manifest_present:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"Manifest file {manifest_path} is not valid JSON: {exc}")
        else:
            profile_value = manifest.get("profile")
            profile = str(profile_value) if isinstance(profile_value, str) else None

            selected_value = manifest.get("selected_model_ids", [])
            if isinstance(selected_value, list):
                selected_model_ids = tuple(str(item) for item in selected_value)
            else:
                warnings.append("Manifest selected_model_ids is not a list.")

            if selected_model_ids and model_name not in selected_model_ids:
                issues.append(
                    f"Manifest selected_model_ids does not include {model_name}. "
                    "Run prepare_models with a profile that includes the baseline model."
                )

            model_records = manifest.get("models", [])
            if isinstance(model_records, list):
                record = next(
                    (
                        item
                        for item in model_records
                        if isinstance(item, dict) and item.get("model_id") == model_name
                    ),
                    None,
                )
                if record is None:
                    warnings.append(f"Manifest models does not include a record for {model_name}.")
                elif record.get("installed") is False:
                    reason = record.get("reason")
                    if isinstance(reason, str) and reason:
                        issues.append(f"Manifest marks {model_name} as not installed: {reason}")
                    else:
                        issues.append(f"Manifest marks {model_name} as not installed.")
            else:
                warnings.append("Manifest models field is not a list.")

    return ModelRepositoryStatus(
        model_name=model_name,
        root_path=str(root),
        manifest_path=str(manifest_path),
        configured=True,
        manifest_present=manifest_present,
        model_directory_present=model_directory.is_dir(),
        model_version_present=model_version.is_dir(),
        model_artifact_present=model_artifact.is_file(),
        profile=profile,
        selected_model_ids=selected_model_ids,
        issues=tuple(issues),
        warnings=tuple(warnings),
    )


class TritonVadClient:
    def __init__(self, url: str, model_name: str = "silero_vad") -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=url, action="Creating the Triton client", exc=exc)
            ) from exc

    def readiness(self) -> TritonReadiness:
        return check_readiness(self._client, self._url, self._model_name)

    def score_windows(self, windows: "np.ndarray") -> list[float]:
        readiness = self.readiness()
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        import numpy as np

        state = np.zeros((2, 1, 128), dtype=np.float32)
        sr_input = self._grpcclient.InferInput("sr", [1], "INT64")
        sr_input.set_data_from_numpy(np.array([16000], dtype=np.int64))
        probabilities: list[float] = []

        try:
            for window in windows:
                audio_input = self._grpcclient.InferInput("input", [1, window.shape[0]], "FP32")
                audio_input.set_data_from_numpy(window.reshape(1, -1).astype(np.float32))

                state_input = self._grpcclient.InferInput("state", list(state.shape), "FP32")
                state_input.set_data_from_numpy(state)

                outputs = [
                    self._grpcclient.InferRequestedOutput("output"),
                    self._grpcclient.InferRequestedOutput("stateN"),
                ]

                response = self._client.infer(
                    model_name=self._model_name,
                    inputs=[audio_input, state_input, sr_input],
                    outputs=outputs,
                )

                probabilities.append(float(response.as_numpy("output").reshape(-1)[0]))
                state = response.as_numpy("stateN").astype(np.float32)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(
                describe_triton_error(url=self._url, action="Running Triton VAD inference", exc=exc)
            ) from exc

        return probabilities
