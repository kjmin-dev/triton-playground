from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class TritonUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class TritonReadiness:
    server_ready: bool
    server_live: bool
    model_ready: bool
    model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "server_ready": self.server_ready,
            "server_live": self.server_live,
            "model_ready": self.model_ready,
            "model_name": self.model_name,
        }


class TritonVadClient:
    def __init__(self, url: str, model_name: str = "silero_vad") -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._model_name = model_name

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Failed to create Triton client for {url}: {exc}") from exc

    def readiness(self) -> TritonReadiness:
        try:
            return TritonReadiness(
                server_ready=bool(self._client.is_server_ready()),
                server_live=bool(self._client.is_server_live()),
                model_ready=bool(self._client.is_model_ready(self._model_name)),
                model_name=self._model_name,
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Failed to query Triton readiness: {exc}") from exc

    def score_windows(self, windows: np.ndarray) -> list[float]:
        state = np.zeros((2, 1, 128), dtype=np.float32)
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
                    inputs=[audio_input, state_input],
                    outputs=outputs,
                )

                probabilities.append(float(response.as_numpy("output").reshape(-1)[0]))
                state = response.as_numpy("stateN").astype(np.float32)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Triton VAD inference failed: {exc}") from exc

        return probabilities
