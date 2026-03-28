from __future__ import annotations

from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

TARGET_PREFIX = {
    "en": "<2en>",
    "ja": "<2ja>",
    "ko": "<2ko>",
    "zh": "<2zh>",
}


def _decode_string(value: object) -> str:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    return str(scalar)


def _tensor_as_bytes_list(request, name: str, expected_count: int | None = None) -> list[str]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    values = [_decode_string(item) for item in tensor.as_numpy().reshape(-1)]
    if expected_count is None:
        return values
    if len(values) == 1 and expected_count > 1:
        return values * expected_count
    if len(values) != expected_count:
        raise pb_utils.TritonModelException(
            f"{name} tensor expected {expected_count} values but received {len(values)}"
        )
    return values


class TritonPythonModel:
    def initialize(self, args):
        _ = args

        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise pb_utils.TritonModelException(
                "MADLAD backend dependencies are missing. Install transformers, sentencepiece, and torch in the Triton runtime."
            ) from exc

        model_dir = Path(__file__).resolve().parent / "upstream"
        if not model_dir.is_dir():
            raise pb_utils.TritonModelException(f"missing MADLAD upstream assets at {model_dir}")

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self._model = self._model.to(self._device)
        self._generate_batch_size = 8

    def execute(self, requests):
        request_output_counts: list[int] = []
        flattened_inputs: list[str] = []

        for request in requests:
            texts = [text.strip() for text in _tensor_as_bytes_list(request, "text")]
            if not texts:
                raise pb_utils.TritonModelException("text tensor is empty")

            target_languages = [
                value.strip().lower() for value in _tensor_as_bytes_list(request, "target_language", len(texts))
            ]
            _ = _tensor_as_bytes_list(request, "source_language", len(texts))

            request_output_counts.append(len(texts))
            for text, target_language in zip(texts, target_languages, strict=True):
                prefix = TARGET_PREFIX.get(target_language)
                if prefix is None:
                    raise pb_utils.TritonModelException(f"unsupported translation target language: {target_language}")
                flattened_inputs.append(f"{prefix} {text}".strip())

        decoded_outputs: list[str] = []
        for start in range(0, len(flattened_inputs), self._generate_batch_size):
            chunk = flattened_inputs[start : start + self._generate_batch_size]
            encoded = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            with self._torch.inference_mode():
                generated = self._model.generate(**encoded, max_new_tokens=512)

            decoded_outputs.extend(
                text.strip()
                for text in self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            )

        responses = []
        output_index = 0
        for output_count in request_output_counts:
            translated_texts = decoded_outputs[output_index : output_index + output_count]
            output_index += output_count
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("translated_text", np.asarray(translated_texts, dtype=object)),
                    ]
                )
            )

        return responses
