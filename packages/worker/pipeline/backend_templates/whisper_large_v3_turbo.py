from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


def _decode_string(value: object) -> str:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    return str(scalar)


def _tensor_as_bytes_list(request, name: str, expected_count: int) -> list[str]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    values = [_decode_string(item) for item in tensor.as_numpy().reshape(-1)]
    if len(values) == 1 and expected_count > 1:
        return values * expected_count
    if len(values) != expected_count:
        raise pb_utils.TritonModelException(
            f"{name} tensor expected {expected_count} values but received {len(values)}"
        )
    return values


def _tensor_as_int_list(request, name: str, expected_count: int) -> list[int]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    values = [int(item) for item in tensor.as_numpy().reshape(-1)]
    if len(values) == 1 and expected_count > 1:
        return values * expected_count
    if len(values) != expected_count:
        raise pb_utils.TritonModelException(
            f"{name} tensor expected {expected_count} values but received {len(values)}"
        )
    return values


def _tensor_as_audio_batch(request, name: str, lengths_name: str) -> list[np.ndarray]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    lengths_tensor = pb_utils.get_input_tensor_by_name(request, lengths_name)
    if lengths_tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {lengths_name}")

    batch = tensor.as_numpy().astype(np.float32)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)
    if batch.size == 0:
        raise pb_utils.TritonModelException("audio_pcm tensor is empty")

    lengths = lengths_tensor.as_numpy().reshape(-1).astype(np.int32)
    if batch.shape[0] != len(lengths):
        raise pb_utils.TritonModelException(
            f"audio batch count {batch.shape[0]} does not match audio_lengths count {len(lengths)}"
        )

    segments: list[np.ndarray] = []
    for row, sample_count in zip(batch, lengths, strict=True):
        if sample_count <= 0:
            raise pb_utils.TritonModelException("audio_lengths must contain only positive values")
        segments.append(row[: int(sample_count)].astype(np.float32))
    return segments


class TritonPythonModel:
    def initialize(self, args):
        _ = args

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise pb_utils.TritonModelException(
                "Whisper backend dependencies are missing. Install transformers, accelerate, and torch in the Triton runtime."
            ) from exc

        model_dir = Path(__file__).resolve().parent / "upstream"
        if not model_dir.is_dir():
            raise pb_utils.TritonModelException(f"missing Whisper upstream assets at {model_dir}")

        self._processor = AutoProcessor.from_pretrained(model_dir)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=self._processor.tokenizer,
            feature_extractor=self._processor.feature_extractor,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=dtype,
        )
        self._pipeline_batch_size = 8

    def execute(self, requests):
        request_transcripts: list[list[str]] = []
        grouped_entries: dict[tuple[str, str, str], list[tuple[int, int, np.ndarray, int]]] = defaultdict(list)

        for request in requests:
            audio_batch = _tensor_as_audio_batch(request, "audio_pcm", "audio_lengths")
            segment_count = len(audio_batch)
            sample_rates = _tensor_as_int_list(request, "sample_rate", segment_count)
            tasks = _tensor_as_bytes_list(request, "task", segment_count)
            languages = _tensor_as_bytes_list(request, "language", segment_count)
            prompts = _tensor_as_bytes_list(request, "prompt", segment_count)

            request_index = len(request_transcripts)
            request_transcripts.append([""] * segment_count)
            for segment_index, (task, language, prompt, sample_rate, audio_segment) in enumerate(
                zip(tasks, languages, prompts, sample_rates, audio_batch, strict=True)
            ):
                grouped_entries[(task, language, prompt)].append(
                    (request_index, segment_index, audio_segment, sample_rate)
                )

        for (task, language, prompt), entries in grouped_entries.items():
            generate_kwargs: dict[str, object] = {"task": task or "transcribe"}
            normalized_language = language or None
            normalized_prompt = prompt or None
            if normalized_language is not None:
                generate_kwargs["language"] = normalized_language
            if normalized_prompt is not None and hasattr(self._processor, "get_prompt_ids"):
                prompt_ids = self._processor.get_prompt_ids(
                    normalized_prompt,
                    language=normalized_language,
                    task=task or "transcribe",
                )
                if prompt_ids is not None:
                    generate_kwargs["prompt_ids"] = prompt_ids

            for start in range(0, len(entries), self._pipeline_batch_size):
                chunk = entries[start : start + self._pipeline_batch_size]
                batch_inputs = [
                    {"array": audio_segment, "sampling_rate": sample_rate} for _, _, audio_segment, sample_rate in chunk
                ]
                result = self._pipeline(
                    batch_inputs,
                    generate_kwargs=generate_kwargs,
                    batch_size=min(self._pipeline_batch_size, len(batch_inputs)),
                )
                if isinstance(result, dict):
                    result = [result]
                if len(result) != len(chunk):
                    raise pb_utils.TritonModelException(
                        f"Whisper pipeline returned {len(result)} results for {len(chunk)} inputs"
                    )
                for (request_index, segment_index, _, _), payload in zip(chunk, result, strict=True):
                    if isinstance(payload, dict):
                        transcript = str(payload.get("text", "")).strip()
                    else:
                        transcript = str(payload).strip()
                    request_transcripts[request_index][segment_index] = transcript

        responses = [
            pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("transcript", np.asarray(transcripts, dtype=object)),
                ]
            )
            for transcripts in request_transcripts
        ]
        return responses
