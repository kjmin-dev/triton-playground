from __future__ import annotations

from dataclasses import dataclass
import io
import wave

import numpy as np


class UnsupportedAudioError(ValueError):
    pass


MAX_UPLOAD_BYTES = 150 * 1024 * 1024
MAX_DURATION_SECONDS = 15 * 60
MAX_CHANNELS = 8
SUPPORTED_SAMPLE_WIDTHS = {1, 2, 4}


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_ms(self) -> int:
        return int(round(len(self.samples) * 1000 / self.sample_rate))


def _normalize_pcm(samples: np.ndarray, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return ((samples.astype(np.float32) - 128.0) / 128.0).astype(np.float32)

    max_value = float(2 ** (sample_width * 8 - 1))
    return (samples.astype(np.float32) / max_value).astype(np.float32)


def decode_wav(blob: bytes) -> AudioBuffer:
    if len(blob) > MAX_UPLOAD_BYTES:
        raise UnsupportedAudioError(
            f"Uploaded audio is too large: {len(blob)} bytes exceeds the {MAX_UPLOAD_BYTES} byte demo limit."
        )

    try:
        with wave.open(io.BytesIO(blob), "rb") as handle:
            sample_width = handle.getsampwidth()
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression_type = handle.getcomptype()
            compression_name = handle.getcompname()
            raw_frames = handle.readframes(frame_count)
    except wave.Error as exc:
        raise UnsupportedAudioError(f"Could not decode WAV container: {exc}") from exc

    if compression_type != "NONE":
        raise UnsupportedAudioError(
            f"Only uncompressed PCM WAV files are supported, not {compression_name!s}."
        )

    if sample_width not in SUPPORTED_SAMPLE_WIDTHS:
        raise UnsupportedAudioError(
            "Only 8-bit, 16-bit, and 32-bit PCM WAV files are supported."
        )

    if channels < 1:
        raise UnsupportedAudioError("Audio file must contain at least one channel.")

    if channels > MAX_CHANNELS:
        raise UnsupportedAudioError(
            f"Audio has {channels} channels, which exceeds the supported maximum of {MAX_CHANNELS}."
        )

    if sample_rate <= 0:
        raise UnsupportedAudioError(f"Audio file reports an invalid sample rate: {sample_rate}.")

    if frame_count <= 0 or not raw_frames:
        raise UnsupportedAudioError("Audio file is empty.")

    duration_seconds = frame_count / sample_rate
    if duration_seconds > MAX_DURATION_SECONDS:
        raise UnsupportedAudioError(
            f"Audio is too long: {duration_seconds:.1f}s exceeds the {MAX_DURATION_SECONDS}s demo limit."
        )

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw_frames, dtype=dtype)

    if samples.size == 0:
        raise UnsupportedAudioError("Audio file is empty.")

    if channels > 1:
        if samples.size % channels != 0:
            raise UnsupportedAudioError(
                "Audio file has a malformed channel layout and cannot be reshaped."
            )
        samples = samples.reshape(-1, channels).mean(axis=1)

    normalized = _normalize_pcm(samples, sample_width)
    return AudioBuffer(samples=normalized, sample_rate=sample_rate)


def resample_audio(audio: AudioBuffer, target_sample_rate: int = 16000) -> AudioBuffer:
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive")

    if audio.sample_rate == target_sample_rate:
        return audio

    if len(audio.samples) == 1:
        target_length = max(1, round(len(audio.samples) * target_sample_rate / audio.sample_rate))
        return AudioBuffer(
            samples=np.repeat(audio.samples.astype(np.float32), target_length).astype(np.float32),
            sample_rate=target_sample_rate,
        )

    target_length = max(1, round(len(audio.samples) * target_sample_rate / audio.sample_rate))
    source_positions = np.linspace(0.0, 1.0, num=len(audio.samples), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio.samples).astype(np.float32)
    return AudioBuffer(samples=resampled, sample_rate=target_sample_rate)
