"""Triton Python backend that wraps Silero VAD ONNX model.

Accepts all audio windows in a single request and runs the sequential
RNN loop server-side, eliminating per-window gRPC round-trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

WINDOW_SAMPLES = 512
SAMPLE_RATE = 16000


class TritonPythonModel:
    def initialize(self, args):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise pb_utils.TritonModelException("onnxruntime is required for the streaming VAD backend.") from exc

        model_dir = Path(args["model_repository"])
        onnx_path = model_dir.parent / "silero_vad" / "1" / "model.onnx"
        if not onnx_path.is_file():
            raise pb_utils.TritonModelException(f"Silero VAD ONNX model not found at {onnx_path}")

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            windows_tensor = pb_utils.get_input_tensor_by_name(request, "audio_windows")
            if windows_tensor is None:
                raise pb_utils.TritonModelException("missing required input: audio_windows")

            windows = windows_tensor.as_numpy().astype(np.float32)
            if windows.ndim == 1:
                windows = windows.reshape(-1, WINDOW_SAMPLES)

            state = np.zeros((2, 1, 128), dtype=np.float32)
            sr = np.array([SAMPLE_RATE], dtype=np.int64)
            probabilities = np.empty(len(windows), dtype=np.float32)

            for i, window in enumerate(windows):
                ort_inputs = {
                    "input": window.reshape(1, -1),
                    "state": state,
                    "sr": sr,
                }
                output, new_state = self._session.run(["output", "stateN"], ort_inputs)
                probabilities[i] = float(output.reshape(-1)[0])
                state = new_state.astype(np.float32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("probabilities", probabilities),
                    ]
                )
            )

        return responses
