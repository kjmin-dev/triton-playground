from __future__ import annotations

DEFAULT_WHISPER_MODEL_ID = "whisper_large_v3_turbo"
DEFAULT_WHISPER_REPOSITORY_MODEL_NAME = "whisper_large_v3_turbo"

WHISPER_TRITON_BACKEND = "python"
WHISPER_AUDIO_INPUT = "audio_pcm"
WHISPER_AUDIO_LENGTHS_INPUT = "audio_lengths"
WHISPER_SAMPLE_RATE_INPUT = "sample_rate"
WHISPER_TASK_INPUT = "task"
WHISPER_LANGUAGE_INPUT = "language"
WHISPER_PROMPT_INPUT = "prompt"
WHISPER_TRANSCRIPT_OUTPUT = "transcript"

SUPPORTED_WHISPER_LANGUAGES = ("ko", "en", "ja", "zh")
SUPPORTED_WHISPER_TASKS = ("transcribe",)

WHISPER_TRITON_INPUT_SPECS = (
    f"{WHISPER_AUDIO_INPUT}: FP32[segments, padded_samples] zero-padded mono PCM at 16 kHz after VAD segmentation",
    f"{WHISPER_AUDIO_LENGTHS_INPUT}: INT32[segments] original sample count for each supplied segment",
    f"{WHISPER_SAMPLE_RATE_INPUT}: INT32[segments] sample rate for each supplied segment",
    f"{WHISPER_TASK_INPUT}: STRING[segments] Whisper task selector; currently only 'transcribe' is supported",
    f"{WHISPER_LANGUAGE_INPUT}: STRING[segments] optional language hint; empty string means auto-detect",
    f"{WHISPER_PROMPT_INPUT}: STRING[segments] optional prompt; empty string means no prompt",
)

WHISPER_TRITON_OUTPUT_SPECS = (
    f"{WHISPER_TRANSCRIPT_OUTPUT}: STRING[segments] UTF-8 transcript for each supplied segment",
)

WHISPER_TRITON_NOTES = (
    "Manual Triton Python backend. The worker runs Silero VAD first, then sends all detected speech "
    "segments to the Whisper model in a single batched Triton request and concatenates the returned text."
)
