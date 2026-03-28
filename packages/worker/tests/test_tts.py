from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.runtime_status import TritonReadiness
from pipeline.tts import (
    TritonTtsClient,
    TtsSynthesisRequest,
    get_default_tts_actor_preset,
    list_tts_actor_presets,
    resolve_tts_actor_preset,
    resolve_tts_preview_variant,
)


class TtsActorPresetTest(unittest.TestCase):
    def test_list_tts_actor_presets_returns_all_languages(self) -> None:
        presets = list_tts_actor_presets()
        self.assertEqual({preset.language for preset in presets}, {"en", "ja", "ko", "zh"})
        self.assertEqual({preset.actor_id for preset in presets}, {"ryan", "ono_anna", "sohee", "vivian"})

    def test_list_tts_actor_presets_filters_by_language(self) -> None:
        presets = list_tts_actor_presets(language="ko")
        self.assertEqual(len(presets), 1)
        self.assertEqual(presets[0].actor_id, "sohee")

    def test_get_default_tts_actor_preset_uses_language_default(self) -> None:
        preset = get_default_tts_actor_preset("en")
        self.assertEqual(preset.actor_id, "ryan")
        self.assertEqual(preset.speaker_name, "Ryan")
        self.assertEqual(preset.default_preview.preview_id, "neutral")

    def test_resolve_tts_actor_preset_validates_language_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support language"):
            resolve_tts_actor_preset("ko", "ryan")

    def test_resolve_tts_actor_preset_rejects_unknown_actor(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown TTS actor"):
            resolve_tts_actor_preset("en", "unknown")

    def test_resolve_tts_preview_variant_uses_actor_default(self) -> None:
        actor = get_default_tts_actor_preset("ko")
        preview = resolve_tts_preview_variant(actor, None)
        self.assertEqual(preview.preview_id, "neutral")
        self.assertTrue(preview.is_default)

    def test_resolve_tts_preview_variant_validates_id(self) -> None:
        actor = get_default_tts_actor_preset("en")
        with self.assertRaisesRegex(ValueError, "does not define preview variant"):
            resolve_tts_preview_variant(actor, "missing")


class _FakeInferInput:
    def __init__(self, name: str, shape: list[int], datatype: str) -> None:
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, data) -> None:
        self.data = data


class _FakeInferRequestedOutput:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeInferResult:
    def __init__(self) -> None:
        self._outputs = {
            "audio_pcm": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32),
            "audio_lengths": np.asarray([3], dtype=np.int32),
            "sample_rate": np.asarray([24000], dtype=np.int32),
        }

    def as_numpy(self, name: str):
        return self._outputs.get(name)


class _FakeGrpcModule:
    InferInput = _FakeInferInput
    InferRequestedOutput = _FakeInferRequestedOutput


class _FakeTritonClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_first = True

    def infer(self, model_name: str, inputs: list[_FakeInferInput], outputs) -> _FakeInferResult:
        _ = (model_name, outputs)
        input_names = [item.name for item in inputs]
        self.calls.append(input_names)
        if self.fail_first:
            self.fail_first = False
            raise RuntimeError(
                "expected 6 inputs but got 7 inputs for model 'qwen3_tts_0_6b'. "
                "Got input(s) ['ref_text','ref_audio_lengths','ref_audio','speaker_prompt','speaker_name','language','text']"
            )
        return _FakeInferResult()


class TritonTtsClientCompatTest(unittest.TestCase):
    def test_retries_without_speaker_name_when_model_contract_is_stale(self) -> None:
        client = TritonTtsClient.__new__(TritonTtsClient)
        client._grpcclient = _FakeGrpcModule
        client._client = _FakeTritonClient()
        client._model_name = "qwen3_tts_0_6b"
        client._url = "localhost:18001"
        client._text_input_name = "text"
        client._language_input_name = "language"
        client._speaker_prompt_input_name = "speaker_prompt"
        client._speaker_name_input_name = "speaker_name"
        client._ref_audio_input_name = "ref_audio"
        client._ref_audio_lengths_input_name = "ref_audio_lengths"
        client._ref_text_input_name = "ref_text"
        client._audio_output_name = "audio_pcm"
        client._audio_lengths_output_name = "audio_lengths"
        client._sample_rate_output_name = "sample_rate"
        client._supports_speaker_name_input = None
        client.readiness = lambda detailed=False: TritonReadiness.from_status(
            server_url="localhost:18001",
            server_ready=True,
            server_live=True,
            model_ready=True,
            model_name="qwen3_tts_0_6b",
        )

        result = client.synthesize_many(
            [
                TtsSynthesisRequest(
                    text="hello",
                    language="en",
                    speaker_prompt="warm reassuring delivery",
                    speaker_name="Ryan",
                )
            ]
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].sample_rate, 24000)
        self.assertEqual(
            client._client.calls[0],
            ["text", "language", "speaker_prompt", "speaker_name", "ref_audio", "ref_audio_lengths", "ref_text"],
        )
        self.assertEqual(
            client._client.calls[1],
            ["text", "language", "speaker_prompt", "ref_audio", "ref_audio_lengths", "ref_text"],
        )
        self.assertFalse(client._supports_speaker_name_input)

        client.synthesize_many([TtsSynthesisRequest(text="again", language="en", speaker_name="Ryan")])
        self.assertEqual(
            client._client.calls[2],
            ["text", "language", "speaker_prompt", "ref_audio", "ref_audio_lengths", "ref_text"],
        )


if __name__ == "__main__":
    unittest.main()
