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


def _tensor_as_bytes(request, name: str) -> str:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    flattened = tensor.as_numpy().reshape(-1)
    if flattened.size == 0:
        return ""

    value = flattened[0].item() if hasattr(flattened[0], "item") else flattened[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


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

    def execute(self, requests):
        responses = []
        for request in requests:
            text = _tensor_as_bytes(request, "text").strip()
            target_language = _tensor_as_bytes(request, "target_language").strip().lower()
            _ = _tensor_as_bytes(request, "source_language")

            prefix = TARGET_PREFIX.get(target_language)
            if prefix is None:
                raise pb_utils.TritonModelException(
                    f"unsupported translation target language: {target_language}"
                )

            encoded = self._tokenizer(
                f"{prefix} {text}",
                return_tensors="pt",
                truncation=True,
            )
            encoded = {key: value.to(self._device) for key, value in encoded.items()}

            with self._torch.inference_mode():
                generated = self._model.generate(**encoded, max_new_tokens=512)

            translated_text = self._tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )[0].strip()

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("translated_text", np.asarray([translated_text], dtype=object)),
                    ]
                )
            )

        return responses
