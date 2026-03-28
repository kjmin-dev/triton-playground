from __future__ import annotations

import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils


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


def _tensor_values_as_strings(tensor: np.ndarray) -> list[str]:
    return [_decode_string(value).strip() for value in tensor.reshape(-1)]


def _tensor_values_as_strings(tensor: np.ndarray) -> list[str]:
    return [_decode_string(value).strip() for value in tensor.reshape(-1)]


class TritonPythonModel:
    def initialize(self, args):
        _ = args
        self._stt_pipeline_model_name = "whisper_stt_pipeline"
        self._translation_model_name = "madlad400_3b_mt"

    def execute(self, requests):
        records: list[dict[str, object]] = []
        for request in requests:
            audio_pcm = _tensor_as_audio(request, "audio_pcm")
            sample_rate = _tensor_as_int(request, "sample_rate")
            threshold = _tensor_as_float(request, "threshold")
            min_speech_ms = _tensor_as_int(request, "min_speech_ms")
            min_silence_ms = _tensor_as_int(request, "min_silence_ms")
            pad_ms = _tensor_as_int(request, "pad_ms")
            window_samples = _tensor_as_int(request, "window_samples")
            source_language = _tensor_as_string(request, "source_language")
            target_language = _tensor_as_string(request, "target_language")
            prompt = _tensor_as_string(request, "prompt")

            t0 = time.perf_counter()
            stt_response = _execute_subrequest(
                self._stt_pipeline_model_name,
                [
                    pb_utils.Tensor("audio_pcm", audio_pcm.reshape(1, -1).astype(np.float32)),
                    pb_utils.Tensor("sample_rate", np.asarray([sample_rate], dtype=np.int32)),
                    pb_utils.Tensor("threshold", np.asarray([threshold], dtype=np.float32)),
                    pb_utils.Tensor("min_speech_ms", np.asarray([min_speech_ms], dtype=np.int32)),
                    pb_utils.Tensor("min_silence_ms", np.asarray([min_silence_ms], dtype=np.int32)),
                    pb_utils.Tensor("pad_ms", np.asarray([pad_ms], dtype=np.int32)),
                    pb_utils.Tensor("window_samples", np.asarray([window_samples], dtype=np.int32)),
                    pb_utils.Tensor("task", np.asarray(["transcribe"], dtype=object)),
                    pb_utils.Tensor("language", np.asarray([source_language], dtype=object)),
                    pb_utils.Tensor("prompt", np.asarray([prompt], dtype=object)),
                ],
                ["transcript", "segments_json"],
            )
            stt_elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
            transcript = _decode_string(_request_output(stt_response, "transcript").reshape(-1)[0]).strip()
            segments_json = _decode_string(_request_output(stt_response, "segments_json").reshape(-1)[0])
            segments = json.loads(segments_json or "[]")

            records.append(
                {
                    "transcript": transcript,
                    "segments_json": segments_json,
                    "segments": segments,
                    "stt_elapsed_ms": stt_elapsed_ms,
                    "source_language": source_language,
                    "target_language": target_language,
                    "translated_text": "",
                    "translated_segment_texts": ["" for _ in segments],
                    "translation_elapsed_ms": 0,
                }
            )

        segment_translation_items: list[tuple[int, int, str, str, str]] = []
        for record_index, record in enumerate(records):
            for segment_index, segment in enumerate(record["segments"]):
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                segment_translation_items.append(
                    (
                        record_index,
                        segment_index,
                        text,
                        str(record["source_language"]),
                        str(record["target_language"]),
                    )
                )

        if segment_translation_items:
            t1 = time.perf_counter()
            translation_response = _execute_subrequest(
                self._translation_model_name,
                [
                    pb_utils.Tensor(
                        "text",
                        np.asarray([item[2] for item in segment_translation_items], dtype=object),
                    ),
                    pb_utils.Tensor(
                        "source_language",
                        np.asarray([item[3] for item in segment_translation_items], dtype=object),
                    ),
                    pb_utils.Tensor(
                        "target_language",
                        np.asarray([item[4] for item in segment_translation_items], dtype=object),
                    ),
                ],
                ["translated_text"],
            )
            translation_elapsed_ms = int(round((time.perf_counter() - t1) * 1000))
            translated_texts = _tensor_values_as_strings(_request_output(translation_response, "translated_text"))
            if len(translated_texts) != len(segment_translation_items):
                raise pb_utils.TritonModelException(
                    "translation model returned "
                    f"{len(translated_texts)} outputs for {len(segment_translation_items)} inputs"
                )
            for item, translated_text in zip(segment_translation_items, translated_texts, strict=True):
                record_index, segment_index, _, _, _ = item
                records[record_index]["translated_segment_texts"][segment_index] = translated_text

            for record in records:
                record["translated_text"] = " ".join(
                    text for text in record["translated_segment_texts"] if text
                ).strip()
                record["translation_elapsed_ms"] = translation_elapsed_ms

        responses = []
        for record in records:
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("transcript", np.asarray([record["transcript"]], dtype=object)),
                        pb_utils.Tensor("segments_json", np.asarray([record["segments_json"]], dtype=object)),
                        pb_utils.Tensor("translated_text", np.asarray([record["translated_text"]], dtype=object)),
                        pb_utils.Tensor(
                            "translated_segments_json",
                            np.asarray([json.dumps(record["translated_segment_texts"])], dtype=object),
                        ),
                        pb_utils.Tensor("stt_elapsed_ms", np.asarray([record["stt_elapsed_ms"]], dtype=np.int32)),
                        pb_utils.Tensor(
                            "translation_elapsed_ms",
                            np.asarray([record["translation_elapsed_ms"]], dtype=np.int32),
                        ),
                    ]
                )
            )

        return responses
