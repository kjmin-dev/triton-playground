from __future__ import annotations

DEFAULT_TRANSLATION_MODEL_ID = "madlad400_3b_mt"
DEFAULT_TRANSLATION_REPOSITORY_MODEL_NAME = "madlad400_3b_mt"

TRANSLATION_TRITON_BACKEND = "python"
TRANSLATION_TEXT_INPUT = "text"
TRANSLATION_SOURCE_LANGUAGE_INPUT = "source_language"
TRANSLATION_TARGET_LANGUAGE_INPUT = "target_language"
TRANSLATION_TEXT_OUTPUT = "translated_text"

SUPPORTED_TRANSLATION_LANGUAGES = ("ko", "en", "ja", "zh")

TRANSLATION_TRITON_INPUT_SPECS = (
    f"{TRANSLATION_TEXT_INPUT}: STRING[texts] UTF-8 text entries to translate",
    f"{TRANSLATION_SOURCE_LANGUAGE_INPUT}: STRING[texts] optional source language hints; empty string means auto-detect",
    f"{TRANSLATION_TARGET_LANGUAGE_INPUT}: STRING[texts] required target language codes",
)

TRANSLATION_TRITON_OUTPUT_SPECS = (f"{TRANSLATION_TEXT_OUTPUT}: STRING[texts] UTF-8 translated text entries",)

TRANSLATION_TRITON_NOTES = (
    "Manual Triton Python backend for the first localization pair. The worker passes the aggregated Whisper transcript "
    "to MADLAD and expects a translated text response. The backend also accepts multi-item text tensors so Triton "
    "server-side pipelines can batch translations across concurrent requests."
)
