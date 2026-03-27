from __future__ import annotations

from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
}

DEFAULT_SPEAKERS = {
    "en": "Ryan",
    "ja": "Ono_Anna",
    "ko": "Sohee",
    "zh": "Vivian",
}

REF_AUDIO_SAMPLE_RATE = 16000


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


def _tensor_as_float_audio(request, name: str) -> np.ndarray | None:
    """Read an optional FP32 audio tensor. Returns None if absent or single-element."""
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        return None
    arr = tensor.as_numpy().reshape(-1).astype(np.float32)
    if arr.size <= 1:
        return None
    return arr


class TritonPythonModel:
    def initialize(self, args):
        _ = args

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise pb_utils.TritonModelException(
                "Qwen3-TTS backend dependencies are missing. Install qwen-tts and torch in the Triton runtime."
            ) from exc

        model_dir = Path(__file__).resolve().parent / "upstream"
        if not model_dir.is_dir():
            raise pb_utils.TritonModelException(f"missing Qwen3-TTS upstream assets at {model_dir}")

        load_kwargs: dict[str, object] = {
            "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        self._model = Qwen3TTSModel.from_pretrained(model_dir, **load_kwargs)

    def execute(self, requests):
        responses = []
        for request in requests:
            text = _tensor_as_bytes(request, "text").strip()
            language = _tensor_as_bytes(request, "language").strip().lower()
            speaker_prompt = _tensor_as_bytes(request, "speaker_prompt").strip()
            ref_audio = _tensor_as_float_audio(request, "ref_audio")
            ref_text = _tensor_as_bytes(request, "ref_text").strip()

            language_name = LANGUAGE_NAMES.get(language)
            if language_name is None:
                raise pb_utils.TritonModelException(f"unsupported TTS language: {language}")

            if ref_audio is not None:
                use_icl = bool(ref_text)
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    language=language_name,
                    ref_audio=(ref_audio, REF_AUDIO_SAMPLE_RATE),
                    ref_text=ref_text if use_icl else None,
                    x_vector_only_mode=not use_icl,
                )
            else:
                speaker = DEFAULT_SPEAKERS.get(language)
                if speaker is None:
                    raise pb_utils.TritonModelException(f"no default speaker for language: {language}")
                wavs, sample_rate = self._model.generate_custom_voice(
                    text=text,
                    language=language_name,
                    speaker=speaker,
                    instruct=speaker_prompt or None,
                )

            if not wavs:
                raise pb_utils.TritonModelException("Qwen3-TTS returned an empty waveform list")

            waveform = np.asarray(wavs[0], dtype=np.float32).reshape(1, -1)
            sample_rate_tensor = np.asarray([int(sample_rate)], dtype=np.int32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("audio_pcm", waveform),
                        pb_utils.Tensor("sample_rate", sample_rate_tensor),
                    ]
                )
            )

        return responses
