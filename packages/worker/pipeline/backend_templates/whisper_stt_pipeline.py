from __future__ import annotations

import json
import math

import numpy as np
import triton_python_backend_utils as pb_utils

WINDOW_SAMPLES = 512
DEFAULT_MODEL_SAMPLE_RATE = 16000


def _decode_string(value: object) -> str:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    return str(scalar)


def _tensor_as_string(request, name: str) -> str:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    flattened = tensor.as_numpy().reshape(-1)
    if flattened.size == 0:
        return ""
    return _decode_string(flattened[0])


def _tensor_as_int(request, name: str) -> int:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    return int(tensor.as_numpy().reshape(-1)[0])


def _tensor_as_float(request, name: str) -> float:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    return float(tensor.as_numpy().reshape(-1)[0])


def _tensor_as_audio(request, name: str) -> np.ndarray:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    samples = tensor.as_numpy().reshape(-1).astype(np.float32)
    if samples.size == 0:
        raise pb_utils.TritonModelException("audio_pcm tensor is empty")
    return samples


def _request_output(response, name: str):
    tensor = pb_utils.get_output_tensor_by_name(response, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"downstream model did not return {name}")
    return tensor.as_numpy()


def _execute_subrequest(model_name: str, inputs: list[pb_utils.Tensor], requested_output_names: list[str]):
    inference_request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=requested_output_names,
        inputs=inputs,
    )
    response = inference_request.exec()
    if response.has_error():
        raise pb_utils.TritonModelException(response.error().message())
    return response


def _speech_segments_from_probabilities(
    *,
    probabilities: list[float],
    total_samples: int,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    min_silence_ms: int,
    pad_ms: int,
    window_samples: int,
) -> list[dict[str, object]]:
    window_ms = window_samples * 1000.0 / sample_rate

    raw_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            if run_start is None:
                run_start = index
            continue

        if run_start is not None:
            raw_runs.append((run_start, index))
            run_start = None

    if run_start is not None:
        raw_runs.append((run_start, len(probabilities)))

    min_speech_windows = max(1, math.ceil(min_speech_ms / window_ms))
    min_silence_windows = max(1, math.ceil(min_silence_ms / window_ms))
    pad_samples = round(pad_ms * sample_rate / 1000)

    merged_runs: list[tuple[int, int]] = []
    for start, end in raw_runs:
        if merged_runs and start - merged_runs[-1][1] <= min_silence_windows:
            previous_start, _ = merged_runs[-1]
            merged_runs[-1] = (previous_start, end)
        else:
            merged_runs.append((start, end))

    segments: list[dict[str, object]] = []
    for start, end in merged_runs:
        if end - start < min_speech_windows:
            continue

        segment_start_sample = max(0, start * window_samples - pad_samples)
        segment_end_sample = min(total_samples, end * window_samples + pad_samples)
        run_probabilities = probabilities[start:end]
        segments.append(
            {
                "start_ms": int(round(segment_start_sample * 1000 / sample_rate)),
                "end_ms": int(round(segment_end_sample * 1000 / sample_rate)),
                "duration_ms": int(round((segment_end_sample - segment_start_sample) * 1000 / sample_rate)),
                "average_probability": float(sum(run_probabilities) / len(run_probabilities)),
                "peak_probability": float(max(run_probabilities)),
                "sample_start": int(segment_start_sample),
                "sample_end": int(segment_end_sample),
            }
        )

    return segments


class TritonPythonModel:
    def initialize(self, args):
        _ = args
        self._vad_model_name = "silero_vad_streaming"
        self._whisper_model_name = "whisper_large_v3_turbo"

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_pcm = _tensor_as_audio(request, "audio_pcm")
            sample_rate = _tensor_as_int(request, "sample_rate")
            threshold = _tensor_as_float(request, "threshold")
            min_speech_ms = _tensor_as_int(request, "min_speech_ms")
            min_silence_ms = _tensor_as_int(request, "min_silence_ms")
            pad_ms = _tensor_as_int(request, "pad_ms")
            window_samples = _tensor_as_int(request, "window_samples")
            task = _tensor_as_string(request, "task") or "transcribe"
            language = _tensor_as_string(request, "language")
            prompt = _tensor_as_string(request, "prompt")

            padded_sample_count = math.ceil(len(audio_pcm) / window_samples) * window_samples
            padded = np.pad(audio_pcm, (0, padded_sample_count - len(audio_pcm)))
            windows = padded.reshape(-1, window_samples).astype(np.float32)

            vad_response = _execute_subrequest(
                self._vad_model_name,
                [
                    pb_utils.Tensor("audio_windows", windows),
                    pb_utils.Tensor("sr", np.asarray([DEFAULT_MODEL_SAMPLE_RATE], dtype=np.int64)),
                ],
                ["probabilities"],
            )
            probabilities = _request_output(vad_response, "probabilities").astype(float).tolist()

            segments = _speech_segments_from_probabilities(
                probabilities=probabilities,
                total_samples=len(audio_pcm),
                sample_rate=sample_rate,
                threshold=threshold,
                min_speech_ms=min_speech_ms,
                min_silence_ms=min_silence_ms,
                pad_ms=pad_ms,
                window_samples=window_samples,
            )

            transcript = ""
            if segments:
                max_samples = max(segment["sample_end"] - segment["sample_start"] for segment in segments)
                audio_batch = np.zeros((len(segments), max_samples), dtype=np.float32)
                audio_lengths = np.zeros(len(segments), dtype=np.int32)

                for index, segment in enumerate(segments):
                    start_sample = int(segment["sample_start"])
                    end_sample = int(segment["sample_end"])
                    clipped = audio_pcm[start_sample:end_sample].astype(np.float32)
                    audio_batch[index, : len(clipped)] = clipped
                    audio_lengths[index] = len(clipped)

                batch_size = len(segments)
                whisper_response = _execute_subrequest(
                    self._whisper_model_name,
                    [
                        pb_utils.Tensor("audio_pcm", audio_batch),
                        pb_utils.Tensor("audio_lengths", audio_lengths),
                        pb_utils.Tensor("sample_rate", np.asarray([sample_rate] * batch_size, dtype=np.int32)),
                        pb_utils.Tensor("task", np.asarray([task] * batch_size, dtype=object)),
                        pb_utils.Tensor("language", np.asarray([language] * batch_size, dtype=object)),
                        pb_utils.Tensor("prompt", np.asarray([prompt] * batch_size, dtype=object)),
                    ],
                    ["transcript"],
                )
                transcripts = [_decode_string(item).strip() for item in _request_output(whisper_response, "transcript")]
                for segment, segment_text in zip(segments, transcripts, strict=True):
                    segment["text"] = segment_text
                    segment.pop("sample_start", None)
                    segment.pop("sample_end", None)
                transcript = " ".join(text for text in transcripts if text).strip()
            else:
                segments = []

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("transcript", np.asarray([transcript], dtype=object)),
                        pb_utils.Tensor("segments_json", np.asarray([json.dumps(segments)], dtype=object)),
                    ]
                )
            )

        return responses
