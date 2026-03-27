from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TritonReadiness:
    server_url: str
    server_ready: bool
    server_live: bool
    model_ready: bool
    model_present: bool | None
    model_state: str | None
    model_name: str
    issues: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.server_ready and self.server_live and self.model_ready

    @property
    def status(self) -> str:
        return "ready" if self.ready else "unavailable"

    @property
    def summary(self) -> str:
        if self.ready:
            return f"Triton server at {self.server_url} and model {self.model_name} are ready."

        return " ".join(self.issues)

    @classmethod
    def from_status(
        cls,
        *,
        server_url: str,
        server_ready: bool,
        server_live: bool,
        model_ready: bool,
        model_present: bool | None = None,
        model_state: str | None = None,
        model_name: str,
    ) -> "TritonReadiness":
        issues: list[str] = []

        if not server_live:
            issues.append(f"Triton server at {server_url} is not live.")

        if server_live and not server_ready:
            issues.append(f"Triton server at {server_url} is live but not ready to accept inference requests.")

        if server_live and server_ready and model_present is False:
            issues.append(
                f"Model {model_name} was not found in the Triton model repository index. "
                "Check model_repository contents and MANIFEST.json."
            )
        elif server_live and server_ready and not model_ready:
            if model_state:
                issues.append(
                    f"Model {model_name} is present in Triton but not ready (state={model_state}). "
                    "Check model_repository contents and Triton startup logs."
                )
            else:
                issues.append(
                    f"Model {model_name} is not ready in Triton. Check model_repository contents and Triton startup logs."
                )

        if not issues:
            issues.append(f"Triton server at {server_url} reported ready status for model {model_name}.")

        return cls(
            server_url=server_url,
            server_ready=server_ready,
            server_live=server_live,
            model_ready=model_ready,
            model_present=model_present,
            model_state=model_state,
            model_name=model_name,
            issues=tuple(issues),
        )

    @classmethod
    def from_error(cls, *, server_url: str, model_name: str, issue: str) -> "TritonReadiness":
        return cls(
            server_url=server_url,
            server_ready=False,
            server_live=False,
            model_ready=False,
            model_present=None,
            model_state=None,
            model_name=model_name,
            issues=(issue,),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "ready": self.ready,
            "summary": self.summary,
            "issues": list(self.issues),
            "server_url": self.server_url,
            "server_ready": self.server_ready,
            "server_live": self.server_live,
            "model_ready": self.model_ready,
            "model_present": self.model_present,
            "model_state": self.model_state,
            "model_name": self.model_name,
        }


def build_ready_payload(profile: str, readiness: TritonReadiness) -> dict[str, object]:
    return {
        "status": readiness.status,
        "profile": profile,
        "triton": readiness.to_dict(),
    }
