from __future__ import annotations

from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils


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


def _tensor_as_int(request, name: str) -> int:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    return int(tensor.as_numpy().reshape(-1)[0])


def _tensor_as_audio(request, name: str) -> np.ndarray:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    samples = tensor.as_numpy().reshape(-1).astype(np.float32)
    if samples.size == 0:
        raise pb_utils.TritonModelException("audio_pcm tensor is empty")
    return samples


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
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
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
            torch_dtype=torch_dtype,
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_pcm = _tensor_as_audio(request, "audio_pcm")
            sample_rate = _tensor_as_int(request, "sample_rate")
            task = _tensor_as_bytes(request, "task") or "transcribe"
            language = _tensor_as_bytes(request, "language") or None
            prompt = _tensor_as_bytes(request, "prompt") or None

            generate_kwargs: dict[str, object] = {"task": task}
            if language is not None:
                generate_kwargs["language"] = language
            if prompt is not None and hasattr(self._processor, "get_prompt_ids"):
                prompt_ids = self._processor.get_prompt_ids(prompt, language=language, task=task)
                if prompt_ids is not None:
                    generate_kwargs["prompt_ids"] = prompt_ids

            result = self._pipeline(
                {"array": audio_pcm, "sampling_rate": sample_rate},
                generate_kwargs=generate_kwargs,
            )

            if isinstance(result, dict):
                transcript = str(result.get("text", "")).strip()
            else:
                transcript = str(result).strip()

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("transcript", np.asarray([transcript], dtype=object)),
                    ]
                )
            )

        return responses
