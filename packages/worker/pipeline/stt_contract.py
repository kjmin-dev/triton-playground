from __future__ import annotations

DEFAULT_WHISPER_MODEL_ID = "whisper_large_v3_turbo"
DEFAULT_WHISPER_REPOSITORY_MODEL_NAME = "whisper_large_v3_turbo"

WHISPER_TRITON_BACKEND = "python"
WHISPER_AUDIO_INPUT = "audio_pcm"
WHISPER_SAMPLE_RATE_INPUT = "sample_rate"
WHISPER_TASK_INPUT = "task"
WHISPER_LANGUAGE_INPUT = "language"
WHISPER_PROMPT_INPUT = "prompt"
WHISPER_TRANSCRIPT_OUTPUT = "transcript"

SUPPORTED_WHISPER_LANGUAGES = ("ko", "en", "ja", "zh")
SUPPORTED_WHISPER_TASKS = ("transcribe",)

WHISPER_TRITON_INPUT_SPECS = (
    f"{WHISPER_AUDIO_INPUT}: FP32[1, samples] mono PCM at 16 kHz after VAD segmentation",
    f"{WHISPER_SAMPLE_RATE_INPUT}: INT32[1] sample rate for the uploaded segment",
    f"{WHISPER_TASK_INPUT}: STRING[1] Whisper task selector; currently only 'transcribe' is supported",
    f"{WHISPER_LANGUAGE_INPUT}: STRING[1] optional language hint; empty string means auto-detect",
    f"{WHISPER_PROMPT_INPUT}: STRING[1] optional prompt; empty string means no prompt",
)

WHISPER_TRITON_OUTPUT_SPECS = (
    f"{WHISPER_TRANSCRIPT_OUTPUT}: STRING[1] UTF-8 transcript for the supplied segment",
)

WHISPER_TRITON_NOTES = (
    "Manual Triton Python backend. The worker runs Silero VAD first, then sends each detected speech "
    "segment to the Whisper model as a 16 kHz mono PCM tensor and concatenates the returned segment text."
)
