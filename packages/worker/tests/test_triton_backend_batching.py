from __future__ import annotations

import importlib.util
import json
import sys
import types
import unittest
import uuid
from pathlib import Path

import numpy as np


class FakeTensor:
    def __init__(self, name: str, values: object) -> None:
        self.name = name
        self._values = np.asarray(values)

    def as_numpy(self) -> np.ndarray:
        return self._values


class FakeInferenceResponse:
    def __init__(self, output_tensors: list[FakeTensor]) -> None:
        self.output_tensors = {tensor.name: tensor for tensor in output_tensors}

    def has_error(self) -> bool:
        return False

    def error(self) -> object:
        raise AssertionError("error() should not be called on a successful fake response")


class FakeInferenceRequest:
    def __init__(self, *args, **kwargs) -> None:
        _ = (args, kwargs)
        raise AssertionError("Unexpected direct InferenceRequest use in test")


class FakePbUtils(types.ModuleType):
    class TritonModelException(Exception):
        pass

    Tensor = FakeTensor
    InferenceResponse = FakeInferenceResponse
    InferenceRequest = FakeInferenceRequest

    @staticmethod
    def get_input_tensor_by_name(request, name: str):
        return request.inputs.get(name)

    @staticmethod
    def get_output_tensor_by_name(response, name: str):
        return response.output_tensors.get(name)


class FakeRequest:
    def __init__(self, inputs: list[FakeTensor]) -> None:
        self.inputs = {tensor.name: tensor for tensor in inputs}


def _load_backend_module(module_path: Path):
    module_name = f"test_backend_{module_path.stem}_{uuid.uuid4().hex}"
    fake_pb_utils = FakePbUtils("triton_python_backend_utils")
    previous_pb_utils = sys.modules.get("triton_python_backend_utils")
    sys.modules["triton_python_backend_utils"] = fake_pb_utils
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise AssertionError(f"Failed to load backend module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, fake_pb_utils
    finally:
        if previous_pb_utils is None:
            sys.modules.pop("triton_python_backend_utils", None)
        else:
            sys.modules["triton_python_backend_utils"] = previous_pb_utils


class LocalizePipelineBackendBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend_path = Path(__file__).resolve().parents[1] / "pipeline" / "backend_templates" / "localize_pipeline.py"
        cls.module, cls.pb_utils = _load_backend_module(backend_path)

    def _make_localize_request(
        self,
        *,
        samples: np.ndarray,
        source_language: str,
        target_language: str,
        speaker_prompt: str,
    ) -> FakeRequest:
        return FakeRequest(
            [
                FakeTensor("audio_pcm", samples),
                FakeTensor("sample_rate", np.asarray([16000], dtype=np.int32)),
                FakeTensor("threshold", np.asarray([0.5], dtype=np.float32)),
                FakeTensor("min_speech_ms", np.asarray([250], dtype=np.int32)),
                FakeTensor("min_silence_ms", np.asarray([200], dtype=np.int32)),
                FakeTensor("pad_ms", np.asarray([100], dtype=np.int32)),
                FakeTensor("window_samples", np.asarray([512], dtype=np.int32)),
                FakeTensor("source_language", np.asarray([source_language], dtype=object)),
                FakeTensor("target_language", np.asarray([target_language], dtype=object)),
                FakeTensor("prompt", np.asarray([""], dtype=object)),
                FakeTensor("speaker_prompt", np.asarray([speaker_prompt], dtype=object)),
            ]
        )

    def test_execute_batches_tts_subrequests_across_records(self) -> None:
        model = self.module.TritonPythonModel()
        model.initialize({})

        stt_texts = ["hello there", "general kenobi"]
        translated_texts = ["annyeong", "ban-gap-seum-ni-da"]
        vad_calls = 0
        whisper_calls = 0
        translation_calls = 0
        tts_calls = 0

        def fake_execute_subrequest(model_name: str, inputs: list[FakeTensor], requested_output_names: list[str]):
            _ = requested_output_names
            input_map = {tensor.name: tensor.as_numpy() for tensor in inputs}
            if model_name == "silero_vad_streaming":
                nonlocal vad_calls
                vad_calls += 1
                self.assertEqual(input_map["audio_windows"].shape[0], 188)
                return FakeInferenceResponse(
                    [FakeTensor("probabilities", np.full(input_map["audio_windows"].shape[0], 0.9, dtype=np.float32))]
                )

            if model_name == "whisper_large_v3_turbo":
                nonlocal whisper_calls
                whisper_calls += 1
                self.assertEqual(input_map["audio_pcm"].shape[0], 2)
                self.assertEqual(input_map["audio_lengths"].reshape(-1).tolist(), [48000, 48000])
                return FakeInferenceResponse([FakeTensor("transcript", np.asarray(stt_texts, dtype=object))])

            if model_name == "madlad400_3b_mt":
                nonlocal translation_calls
                translation_calls += 1
                self.assertEqual([self.module._decode_string(value) for value in input_map["text"]], stt_texts)
                return FakeInferenceResponse(
                    [FakeTensor("translated_text", np.asarray(translated_texts, dtype=object))]
                )

            if model_name == "qwen3_tts_0_6b":
                nonlocal tts_calls
                tts_calls += 1
                self.assertEqual([self.module._decode_string(value) for value in input_map["text"]], translated_texts)
                self.assertEqual([self.module._decode_string(value) for value in input_map["ref_text"]], ["", ""])
                return FakeInferenceResponse(
                    [
                        FakeTensor(
                            "audio_pcm",
                            np.asarray(
                                [
                                    [0.1, 0.2, 0.3, 0.4],
                                    [0.5, 0.6, 0.7, 0.8],
                                ],
                                dtype=np.float32,
                            ),
                        ),
                        FakeTensor("audio_lengths", np.asarray([4, 4], dtype=np.int32)),
                        FakeTensor("sample_rate", np.asarray([24000, 24000], dtype=np.int32)),
                    ]
                )

            raise AssertionError(f"Unexpected model call: {model_name}")

        original_execute_subrequest = self.module._execute_subrequest
        self.module._execute_subrequest = fake_execute_subrequest
        try:
            responses = model.execute(
                [
                    self._make_localize_request(
                        samples=np.ones(16000 * 3, dtype=np.float32),
                        source_language="en",
                        target_language="ko",
                        speaker_prompt="warm",
                    ),
                    self._make_localize_request(
                        samples=np.ones(16000 * 3, dtype=np.float32) * 0.5,
                        source_language="en",
                        target_language="ko",
                        speaker_prompt="warm",
                    ),
                ]
            )
        finally:
            self.module._execute_subrequest = original_execute_subrequest

        self.assertEqual(vad_calls, 1)
        self.assertEqual(whisper_calls, 1)
        self.assertEqual(translation_calls, 1)
        self.assertEqual(tts_calls, 1)
        self.assertEqual(len(responses), 2)

        for response, expected_translation in zip(responses, translated_texts, strict=True):
            translated_text = self.module._decode_string(
                response.output_tensors["translated_text"].as_numpy().reshape(-1)[0]
            )
            tts_meta = json.loads(
                self.module._decode_string(response.output_tensors["tts_meta_json"].as_numpy().reshape(-1)[0])
            )
            audio_length = int(response.output_tensors["audio_length"].as_numpy().reshape(-1)[0])
            self.assertEqual(translated_text, expected_translation)
            self.assertGreater(audio_length, 0)
            self.assertEqual(tts_meta["status"], "ok")
            self.assertEqual(tts_meta["voice_cloning_mode"], "x_vector")
            self.assertEqual(tts_meta["segments_synthesized"], 1)

    def test_build_tts_requests_keeps_ref_text_for_same_language_clone(self) -> None:
        model = self.module.TritonPythonModel()
        model.initialize({})

        audio_pcm = np.ones(16000 * 3, dtype=np.float32)
        diarized_segments = [
            {
                "start_ms": 0,
                "end_ms": 3000,
                "duration_ms": 3000,
                "average_probability": 0.9,
                "peak_probability": 0.95,
                "text": "안녕하세요",
                "speaker_id": "speaker_0",
            }
        ]
        speaker_groups = {"speaker_0": diarized_segments}

        pending_requests, context = model._build_tts_requests(
            audio_pcm,
            16000,
            diarized_segments,
            ["안녕하세요"],
            speaker_groups,
            "ko",
            "ko",
            "clear",
        )

        self.assertTrue(context["allow_ref_text"])
        self.assertEqual(len(pending_requests), 1)
        self.assertEqual(pending_requests[0]["ref_text"], "안녕하세요")


class QwenBackendBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend_path = Path(__file__).resolve().parents[1] / "pipeline" / "backend_templates" / "qwen3_tts_0_6b.py"
        cls.module, cls.pb_utils = _load_backend_module(backend_path)

    def _make_qwen_request(
        self,
        *,
        texts: list[str],
        languages: list[str],
        speaker_prompts: list[str],
        ref_audio_batch: np.ndarray,
        ref_audio_lengths: np.ndarray,
        ref_texts: list[str],
    ) -> FakeRequest:
        return FakeRequest(
            [
                FakeTensor("text", np.asarray(texts, dtype=object)),
                FakeTensor("language", np.asarray(languages, dtype=object)),
                FakeTensor("speaker_prompt", np.asarray(speaker_prompts, dtype=object)),
                FakeTensor("ref_audio", ref_audio_batch.astype(np.float32)),
                FakeTensor("ref_audio_lengths", ref_audio_lengths.astype(np.int32)),
                FakeTensor("ref_text", np.asarray(ref_texts, dtype=object)),
            ]
        )

    def test_execute_splits_flattened_outputs_back_per_request(self) -> None:
        model = self.module.TritonPythonModel()

        class FakeQwenModel:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def generate_custom_voice(self, *, text: str, language: str, speaker: str, instruct: str | None):
                _ = (speaker, instruct)
                self.calls.append(("custom", text))
                return [np.linspace(0.1, 0.3, num=len(text), dtype=np.float32)], 24000

            def generate_voice_clone(
                self,
                *,
                text: str,
                language: str,
                ref_audio: tuple[np.ndarray, int],
                ref_text: str | None,
                x_vector_only_mode: bool,
            ):
                _ = (language, ref_audio, ref_text, x_vector_only_mode)
                self.calls.append(("clone", text))
                return [np.linspace(0.2, 0.4, num=len(text) + 1, dtype=np.float32)], 24000

        fake_model = FakeQwenModel()
        model._model = fake_model

        responses = model.execute(
            [
                self._make_qwen_request(
                    texts=["hi", "there"],
                    languages=["en", "ko"],
                    speaker_prompts=["bright", "calm"],
                    ref_audio_batch=np.zeros((2, 1), dtype=np.float32),
                    ref_audio_lengths=np.asarray([1, 1], dtype=np.int32),
                    ref_texts=["", ""],
                ),
                self._make_qwen_request(
                    texts=["dub"],
                    languages=["ko"],
                    speaker_prompts=[""],
                    ref_audio_batch=np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32),
                    ref_audio_lengths=np.asarray([4], dtype=np.int32),
                    ref_texts=["원문"],
                ),
            ]
        )

        self.assertEqual(fake_model.calls, [("custom", "hi"), ("custom", "there"), ("clone", "dub")])
        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].output_tensors["audio_pcm"].as_numpy().shape, (2, 5))
        self.assertEqual(responses[1].output_tensors["audio_pcm"].as_numpy().shape, (1, 4))
        self.assertEqual(
            responses[0].output_tensors["audio_lengths"].as_numpy().reshape(-1).tolist(),
            [2, 5],
        )
        self.assertEqual(
            responses[1].output_tensors["audio_lengths"].as_numpy().reshape(-1).tolist(),
            [4],
        )


class WhisperSttPipelineBatchingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        backend_path = (
            Path(__file__).resolve().parents[1] / "pipeline" / "backend_templates" / "whisper_stt_pipeline.py"
        )
        cls.module, cls.pb_utils = _load_backend_module(backend_path)

    def _make_stt_request(self, *, source_language: str, samples: np.ndarray) -> FakeRequest:
        return FakeRequest(
            [
                FakeTensor("audio_pcm", samples),
                FakeTensor("sample_rate", np.asarray([16000], dtype=np.int32)),
                FakeTensor("threshold", np.asarray([0.5], dtype=np.float32)),
                FakeTensor("min_speech_ms", np.asarray([30], dtype=np.int32)),
                FakeTensor("min_silence_ms", np.asarray([30], dtype=np.int32)),
                FakeTensor("pad_ms", np.asarray([0], dtype=np.int32)),
                FakeTensor("window_samples", np.asarray([512], dtype=np.int32)),
                FakeTensor("task", np.asarray(["transcribe"], dtype=object)),
                FakeTensor("language", np.asarray([source_language], dtype=object)),
                FakeTensor("prompt", np.asarray([""], dtype=object)),
            ]
        )

    def test_execute_batches_vad_and_whisper_across_requests(self) -> None:
        model = self.module.TritonPythonModel()
        model.initialize({})

        vad_calls = 0
        whisper_calls = 0
        expected_texts = ["hello", "world"]

        def fake_execute_subrequest(model_name: str, inputs: list[FakeTensor], requested_output_names: list[str]):
            _ = requested_output_names
            input_map = {tensor.name: tensor.as_numpy() for tensor in inputs}
            if model_name == "silero_vad_streaming":
                nonlocal vad_calls
                vad_calls += 1
                self.assertEqual(input_map["audio_windows"].shape, (6, 512))
                return FakeInferenceResponse(
                    [FakeTensor("probabilities", np.asarray([0.9, 0.9, 0.9, 0.8, 0.8, 0.8], dtype=np.float32))]
                )

            if model_name == "whisper_large_v3_turbo":
                nonlocal whisper_calls
                whisper_calls += 1
                self.assertEqual(input_map["audio_pcm"].shape[0], 2)
                self.assertEqual(input_map["audio_lengths"].reshape(-1).tolist(), [1536, 1536])
                return FakeInferenceResponse([FakeTensor("transcript", np.asarray(expected_texts, dtype=object))])

            raise AssertionError(f"Unexpected model call: {model_name}")

        original_execute_subrequest = self.module._execute_subrequest
        self.module._execute_subrequest = fake_execute_subrequest
        try:
            responses = model.execute(
                [
                    self._make_stt_request(source_language="en", samples=np.ones(512 * 3, dtype=np.float32)),
                    self._make_stt_request(source_language="en", samples=np.ones(512 * 3, dtype=np.float32) * 0.5),
                ]
            )
        finally:
            self.module._execute_subrequest = original_execute_subrequest

        self.assertEqual(vad_calls, 1)
        self.assertEqual(whisper_calls, 1)
        self.assertEqual(len(responses), 2)

        for response, expected_text in zip(responses, expected_texts, strict=True):
            transcript = self.module._decode_string(response.output_tensors["transcript"].as_numpy().reshape(-1)[0])
            segments = json.loads(
                self.module._decode_string(response.output_tensors["segments_json"].as_numpy().reshape(-1)[0])
            )
            self.assertEqual(transcript, expected_text)
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0]["text"], expected_text)


if __name__ == "__main__":
    unittest.main()
