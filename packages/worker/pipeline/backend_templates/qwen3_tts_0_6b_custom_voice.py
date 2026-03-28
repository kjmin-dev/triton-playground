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
    if expected_count <= 0:
        return values
    if len(values) == 1 and expected_count > 1:
        return values * expected_count
    if len(values) != expected_count:
        raise pb_utils.TritonModelException(
            f"{name} tensor expected {expected_count} values but received {len(values)}"
        )
    return values


def _tensor_as_ref_audio_batch(request, name: str, lengths_name: str) -> list[np.ndarray | None]:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")

    lengths_tensor = pb_utils.get_input_tensor_by_name(request, lengths_name)
    if lengths_tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {lengths_name}")

    batch = tensor.as_numpy().astype(np.float32)
    if batch.ndim == 1:
        batch = batch.reshape(1, -1)

    lengths = lengths_tensor.as_numpy().reshape(-1).astype(np.int32)
    if batch.shape[0] != len(lengths):
        raise pb_utils.TritonModelException(
            f"ref_audio batch count {batch.shape[0]} does not match ref_audio_lengths count {len(lengths)}"
        )

    references: list[np.ndarray | None] = []
    for row, sample_count in zip(batch, lengths, strict=True):
        if sample_count <= 1:
            references.append(None)
            continue
        references.append(row[: int(sample_count)].astype(np.float32))
    return references


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
        request_batches: list[tuple[int, int]] = []
        flattened_items: list[dict[str, object]] = []

        for request in requests:
            texts = _tensor_as_bytes_list(request, "text", expected_count=-1)
            batch_count = len(texts)
            if batch_count == 0:
                raise pb_utils.TritonModelException("text tensor is empty")
            languages = _tensor_as_bytes_list(request, "language", batch_count)
            speaker_prompts = _tensor_as_bytes_list(request, "speaker_prompt", batch_count)
            speaker_names = _tensor_as_bytes_list(request, "speaker_name", batch_count)
            ref_audios = _tensor_as_ref_audio_batch(request, "ref_audio", "ref_audio_lengths")
            if len(ref_audios) != batch_count:
                raise pb_utils.TritonModelException(
                    f"ref_audio batch count {len(ref_audios)} does not match text batch count {batch_count}"
                )
            ref_texts = _tensor_as_bytes_list(request, "ref_text", batch_count)

            start_index = len(flattened_items)
            request_batches.append((start_index, batch_count))
            for text, language, speaker_prompt, speaker_name, ref_audio, ref_text in zip(
                texts,
                languages,
                speaker_prompts,
                speaker_names,
                ref_audios,
                ref_texts,
                strict=True,
            ):
                flattened_items.append(
                    {
                        "text": text,
                        "language": language,
                        "speaker_prompt": speaker_prompt,
                        "speaker_name": speaker_name,
                        "ref_audio": ref_audio,
                        "ref_text": ref_text,
                    }
                )

        waveforms: list[np.ndarray] = []
        sample_rates: list[int] = []
        audio_lengths: list[int] = []

        for item in flattened_items:
            text = str(item["text"]).strip()
            language = str(item["language"]).strip().lower()
            speaker_prompt = str(item["speaker_prompt"])
            speaker_name = str(item["speaker_name"]).strip()
            ref_audio = item["ref_audio"]
            ref_text = str(item["ref_text"])

            language_name = LANGUAGE_NAMES.get(language)
            if language_name is None:
                raise pb_utils.TritonModelException(f"unsupported TTS language: {language}")

            if isinstance(ref_audio, np.ndarray):
                use_icl = bool(ref_text.strip())
                wavs, sample_rate = self._model.generate_voice_clone(
                    text=text,
                    language=language_name,
                    ref_audio=(ref_audio, REF_AUDIO_SAMPLE_RATE),
                    ref_text=ref_text.strip() if use_icl else None,
                    x_vector_only_mode=not use_icl,
                )
            else:
                speaker = speaker_name or DEFAULT_SPEAKERS.get(language)
                if speaker is None:
                    raise pb_utils.TritonModelException(f"no default speaker for language: {language}")
                try:
                    wavs, sample_rate = self._model.generate_custom_voice(
                        text=text,
                        language=language_name,
                        speaker=speaker,
                        instruct=speaker_prompt.strip() or None,
                    )
                except ValueError as exc:
                    if "does not support generate_custom_voice" in str(exc):
                        raise pb_utils.TritonModelException(
                            "The installed Qwen3-TTS checkpoint does not support preset actor synthesis. "
                            "Use reference voice cloning with ref_audio or materialize the optional CustomVoice checkpoint."
                        ) from exc
                    raise

            if not wavs:
                raise pb_utils.TritonModelException("Qwen3-TTS returned an empty waveform list")

            waveform = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            if waveform.size == 0:
                raise pb_utils.TritonModelException("Qwen3-TTS returned an empty waveform")

            waveforms.append(waveform)
            audio_lengths.append(int(waveform.size))
            sample_rates.append(int(sample_rate))

        responses = []
        for start_index, batch_count in request_batches:
            end_index = start_index + batch_count
            request_waveforms = waveforms[start_index:end_index]
            request_audio_lengths = audio_lengths[start_index:end_index]
            request_sample_rates = sample_rates[start_index:end_index]
            max_samples = max(request_audio_lengths)
            audio_batch = np.zeros((batch_count, max_samples), dtype=np.float32)
            for index, waveform in enumerate(request_waveforms):
                audio_batch[index, : waveform.size] = waveform

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("audio_pcm", audio_batch),
                        pb_utils.Tensor("audio_lengths", np.asarray(request_audio_lengths, dtype=np.int32)),
                        pb_utils.Tensor("sample_rate", np.asarray(request_sample_rates, dtype=np.int32)),
                    ]
                )
            )

        return responses
