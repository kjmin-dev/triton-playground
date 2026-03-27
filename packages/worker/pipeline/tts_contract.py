from __future__ import annotations

DEFAULT_TTS_MODEL_ID = "qwen3_tts_0_6b"
DEFAULT_TTS_REPOSITORY_MODEL_NAME = "qwen3_tts_0_6b"

TTS_TRITON_BACKEND = "python"
TTS_TEXT_INPUT = "text"
TTS_LANGUAGE_INPUT = "language"
TTS_SPEAKER_PROMPT_INPUT = "speaker_prompt"
TTS_REF_AUDIO_INPUT = "ref_audio"
TTS_REF_TEXT_INPUT = "ref_text"
TTS_AUDIO_OUTPUT = "audio_pcm"
TTS_SAMPLE_RATE_OUTPUT = "sample_rate"

SUPPORTED_TTS_LANGUAGES = ("ko", "en", "ja", "zh")

TTS_TRITON_INPUT_SPECS = (
    f"{TTS_TEXT_INPUT}: STRING[1] UTF-8 text to synthesize",
    f"{TTS_LANGUAGE_INPUT}: STRING[1] required target language code",
    f"{TTS_SPEAKER_PROMPT_INPUT}: STRING[1] optional prompt for speaking style; empty string means default voice",
    f"{TTS_REF_AUDIO_INPUT}: FP32[1, samples] optional reference audio for voice cloning; single-element tensor means no cloning",
    f"{TTS_REF_TEXT_INPUT}: STRING[1] optional transcript of reference audio for ICL voice cloning; empty string means x-vector only",
)

TTS_TRITON_OUTPUT_SPECS = (
    f"{TTS_AUDIO_OUTPUT}: FP32[1, samples] mono waveform normalized to [-1, 1]",
    f"{TTS_SAMPLE_RATE_OUTPUT}: INT32[1] sample rate for the synthesized waveform",
)

TTS_TRITON_NOTES = (
    "Manual Triton Python backend for the first previewable TTS path. The worker requests a mono waveform from "
    "Qwen3-TTS and wraps it as a WAV preview for the web UI."
)
