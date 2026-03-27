from __future__ import annotations

import io
import struct
import sys
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.audio import AudioBuffer, UnsupportedAudioError, decode_wav, resample_audio


def build_wav_bytes(
    *,
    sample_width: int = 2,
    channels: int = 1,
    sample_rate: int = 16_000,
    frame_count: int = 160,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)

        if sample_width == 1:
            frames = bytes([128 + ((index % 32) - 16) for index in range(frame_count * channels)])
        elif sample_width == 2:
            frames = struct.pack(
                "<" + "h" * frame_count * channels,
                *[((index % 64) - 32) * 1024 for index in range(frame_count * channels)],
            )
        elif sample_width == 4:
            frames = struct.pack(
                "<" + "i" * frame_count * channels,
                *[((index % 64) - 32) * 1_000_000 for index in range(frame_count * channels)],
            )
        else:
            frames = b"\x00" * frame_count * channels * sample_width

        handle.writeframes(frames)

    return buffer.getvalue()


class AudioDecodeTest(unittest.TestCase):
    def test_decode_wav_accepts_pcm_audio(self) -> None:
        audio = decode_wav(build_wav_bytes())

        self.assertEqual(audio.sample_rate, 16_000)
        self.assertGreater(audio.duration_ms, 0)
        self.assertIsInstance(audio, AudioBuffer)

    def test_decode_wav_rejects_non_wav_container(self) -> None:
        with self.assertRaisesRegex(UnsupportedAudioError, "Could not decode WAV container"):
            decode_wav(b"not a wav file")

    def test_decode_wav_rejects_unsupported_sample_width(self) -> None:
        with self.assertRaisesRegex(UnsupportedAudioError, "Only 8-bit, 16-bit, and 32-bit PCM WAV files"):
            decode_wav(build_wav_bytes(sample_width=3))

    def test_decode_wav_rejects_long_audio(self) -> None:
        long_wav = build_wav_bytes(sample_rate=1, frame_count=901)

        with self.assertRaisesRegex(UnsupportedAudioError, "Audio is too long"):
            decode_wav(long_wav)

    def test_decode_wav_rejects_oversized_upload(self) -> None:
        with patch("pipeline.audio.MAX_UPLOAD_BYTES", 1):
            with self.assertRaisesRegex(UnsupportedAudioError, "Uploaded audio is too large"):
                decode_wav(build_wav_bytes())

    def test_resample_audio_supports_single_sample_buffer(self) -> None:
        audio = AudioBuffer(samples=np.array([0.25], dtype=np.float32), sample_rate=8_000)

        resampled = resample_audio(audio, target_sample_rate=16_000)

        self.assertEqual(resampled.sample_rate, 16_000)
        self.assertGreaterEqual(len(resampled.samples), 1)


if __name__ == "__main__":
    unittest.main()
