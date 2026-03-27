from __future__ import annotations

import numpy as np

from pipeline.runtime_status import TritonReadiness
from pipeline.translation_contract import (
    DEFAULT_TRANSLATION_REPOSITORY_MODEL_NAME,
    SUPPORTED_TRANSLATION_LANGUAGES,
    TRANSLATION_SOURCE_LANGUAGE_INPUT,
    TRANSLATION_TARGET_LANGUAGE_INPUT,
    TRANSLATION_TEXT_INPUT,
    TRANSLATION_TEXT_OUTPUT,
)
from pipeline.triton import TritonUnavailableError, check_readiness


def normalize_pipeline_language(language: str | None, *, allow_auto: bool) -> str | None:
    if language is None:
        return None

    normalized = language.strip().lower()
    if not normalized:
        return None

    if normalized == "auto":
        if allow_auto:
            return None
        supported = ", ".join(SUPPORTED_TRANSLATION_LANGUAGES)
        raise ValueError(f"language must be one of: {supported}")

    if normalized not in SUPPORTED_TRANSLATION_LANGUAGES:
        supported = (
            ", ".join(("auto", *SUPPORTED_TRANSLATION_LANGUAGES))
            if allow_auto
            else ", ".join(SUPPORTED_TRANSLATION_LANGUAGES)
        )
        raise ValueError(f"language must be one of: {supported}")

    return normalized


def _decode_string_output(tensor: np.ndarray | None, *, output_name: str) -> str:
    if tensor is None:
        raise TritonUnavailableError(
            f"Triton inference succeeded but did not return the configured output tensor {output_name}."
        )

    flattened = tensor.reshape(-1)
    if flattened.size == 0:
        return ""

    fragments: list[str] = []
    for item in flattened:
        scalar = item.item() if hasattr(item, "item") else item
        if isinstance(scalar, bytes):
            fragments.append(scalar.decode("utf-8"))
        elif isinstance(scalar, str):
            fragments.append(scalar)
        else:
            fragments.append(str(scalar))

    return " ".join(fragment.strip() for fragment in fragments if fragment.strip())


class TritonTranslationClient:
    def __init__(
        self,
        url: str,
        model_name: str = DEFAULT_TRANSLATION_REPOSITORY_MODEL_NAME,
        *,
        text_input_name: str = TRANSLATION_TEXT_INPUT,
        source_language_input_name: str = TRANSLATION_SOURCE_LANGUAGE_INPUT,
        target_language_input_name: str = TRANSLATION_TARGET_LANGUAGE_INPUT,
        text_output_name: str = TRANSLATION_TEXT_OUTPUT,
    ) -> None:
        try:
            import tritonclient.grpc as grpcclient
        except ImportError as exc:
            raise TritonUnavailableError("tritonclient[grpc] is not installed.") from exc

        self._grpcclient = grpcclient
        self._url = url
        self._model_name = model_name
        self._text_input_name = text_input_name
        self._source_language_input_name = source_language_input_name
        self._target_language_input_name = target_language_input_name
        self._text_output_name = text_output_name

        try:
            self._client = grpcclient.InferenceServerClient(url=url, verbose=False)
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Failed to create Triton client for {url}: {exc}") from exc

    def readiness(self) -> TritonReadiness:
        return check_readiness(self._client, self._url, self._model_name)

    def translate(
        self,
        text: str,
        *,
        source_language: str | None,
        target_language: str,
    ) -> str:
        readiness = self.readiness()
        if not readiness.ready:
            raise TritonUnavailableError(readiness.summary)

        try:
            text_input = self._grpcclient.InferInput(self._text_input_name, [1], "BYTES")
            text_input.set_data_from_numpy(np.asarray([text], dtype=object))

            source_language_input = self._grpcclient.InferInput(self._source_language_input_name, [1], "BYTES")
            source_language_input.set_data_from_numpy(np.asarray([source_language or ""], dtype=object))

            target_language_input = self._grpcclient.InferInput(self._target_language_input_name, [1], "BYTES")
            target_language_input.set_data_from_numpy(np.asarray([target_language], dtype=object))

            result = self._client.infer(
                self._model_name,
                [text_input, source_language_input, target_language_input],
                outputs=[self._grpcclient.InferRequestedOutput(self._text_output_name)],
            )
        except Exception as exc:  # pragma: no cover - transport failures depend on the runtime
            raise TritonUnavailableError(f"Translation inference request failed: {exc}") from exc

        return _decode_string_output(result.as_numpy(self._text_output_name), output_name=self._text_output_name)
