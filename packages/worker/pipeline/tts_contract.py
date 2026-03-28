from __future__ import annotations

DEFAULT_TTS_MODEL_ID = "qwen3_tts_0_6b"
DEFAULT_TTS_REPOSITORY_MODEL_NAME = "qwen3_tts_0_6b"

TTS_TRITON_BACKEND = "python"
TTS_TEXT_INPUT = "text"
TTS_LANGUAGE_INPUT = "language"
TTS_SPEAKER_PROMPT_INPUT = "speaker_prompt"
TTS_REF_AUDIO_INPUT = "ref_audio"
TTS_REF_AUDIO_LENGTHS_INPUT = "ref_audio_lengths"
TTS_REF_TEXT_INPUT = "ref_text"
TTS_AUDIO_OUTPUT = "audio_pcm"
TTS_AUDIO_LENGTHS_OUTPUT = "audio_lengths"
TTS_SAMPLE_RATE_OUTPUT = "sample_rate"

SUPPORTED_TTS_LANGUAGES = ("ko", "en", "ja", "zh")

TTS_TRITON_INPUT_SPECS = (
    f"{TTS_TEXT_INPUT}: STRING[segments] UTF-8 text to synthesize",
    f"{TTS_LANGUAGE_INPUT}: STRING[segments] required target language code",
    f"{TTS_SPEAKER_PROMPT_INPUT}: STRING[segments] optional prompt for speaking style; empty string means default voice",
    f"{TTS_REF_AUDIO_INPUT}: FP32[segments, padded_samples] optional reference audio for voice cloning; zeros mean no cloning",
    f"{TTS_REF_AUDIO_LENGTHS_INPUT}: INT32[segments] original sample count for each reference waveform",
    f"{TTS_REF_TEXT_INPUT}: STRING[segments] optional transcript of reference audio for ICL voice cloning; empty string means x-vector only",
)

TTS_TRITON_OUTPUT_SPECS = (
    f"{TTS_AUDIO_OUTPUT}: FP32[segments, padded_samples] zero-padded mono waveform normalized to [-1, 1]",
    f"{TTS_AUDIO_LENGTHS_OUTPUT}: INT32[segments] original sample count for each synthesized waveform",
    f"{TTS_SAMPLE_RATE_OUTPUT}: INT32[segments] sample rate for each synthesized waveform",
)

TTS_TRITON_NOTES = (
    "Manual Triton Python backend for the first previewable TTS path. The worker batches per-segment synthesis "
    "requests into a single Triton call and wraps the returned waveforms as a WAV preview for the web UI."
)
