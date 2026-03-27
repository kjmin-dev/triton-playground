from __future__ import annotations

from dataclasses import dataclass
import io
import wave

import numpy as np


class UnsupportedAudioError(ValueError):
    pass


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
    with wave.open(io.BytesIO(blob), "rb") as handle:
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw_frames = handle.readframes(frame_count)

    if sample_width not in {1, 2, 4}:
        raise UnsupportedAudioError("Only 8-bit, 16-bit, and 32-bit PCM WAV files are supported.")

    if channels < 1:
        raise UnsupportedAudioError("Audio file must contain at least one channel.")

    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sample_width]
    samples = np.frombuffer(raw_frames, dtype=dtype)

    if samples.size == 0:
        raise UnsupportedAudioError("Audio file is empty.")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    normalized = _normalize_pcm(samples, sample_width)
    return AudioBuffer(samples=normalized, sample_rate=sample_rate)


def resample_audio(audio: AudioBuffer, target_sample_rate: int = 16000) -> AudioBuffer:
    if audio.sample_rate == target_sample_rate:
        return audio

    target_length = max(1, round(len(audio.samples) * target_sample_rate / audio.sample_rate))
    source_positions = np.linspace(0.0, 1.0, num=len(audio.samples), endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio.samples).astype(np.float32)
    return AudioBuffer(samples=resampled, sample_rate=target_sample_rate)
