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
    f"{TRANSLATION_TEXT_INPUT}: STRING[1] UTF-8 text to translate",
    f"{TRANSLATION_SOURCE_LANGUAGE_INPUT}: STRING[1] optional source language hint; empty string means auto-detect",
    f"{TRANSLATION_TARGET_LANGUAGE_INPUT}: STRING[1] required target language code",
)

TRANSLATION_TRITON_OUTPUT_SPECS = (f"{TRANSLATION_TEXT_OUTPUT}: STRING[1] UTF-8 translated text",)

TRANSLATION_TRITON_NOTES = (
    "Manual Triton Python backend for the first localization pair. The worker passes the aggregated Whisper transcript "
    "to MADLAD and expects a single translated text response."
)
