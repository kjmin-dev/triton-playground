from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pipeline.audio import AudioBuffer


@dataclass(frozen=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    duration_ms: int
    average_probability: float
    peak_probability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
            "average_probability": round(self.average_probability, 4),
            "peak_probability": round(self.peak_probability, 4),
        }


@dataclass(frozen=True)
class VadAnalysis:
    threshold: float
    window_ms: float
    duration_ms: int
    window_scores: list[float]
    segments: list[SpeechSegment]


def analyze_vad(
    audio: AudioBuffer,
    client,
    threshold: float = 0.5,
    min_speech_ms: int = 160,
    min_silence_ms: int = 240,
    pad_ms: int = 80,
    window_samples: int = 512,
) -> VadAnalysis:
    total_samples = len(audio.samples)
    padded_sample_count = math.ceil(total_samples / window_samples) * window_samples
    padded = np.pad(audio.samples, (0, padded_sample_count - total_samples))
    windows = padded.reshape(-1, window_samples).astype(np.float32)

    probabilities = client.score_windows(windows)
    window_ms = window_samples * 1000.0 / audio.sample_rate

    raw_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for index, probability in enumerate(probabilities):
        if probability >= threshold:
            if run_start is None:
                run_start = index
            continue

        if run_start is not None:
            raw_runs.append((run_start, index))
            run_start = None

    if run_start is not None:
        raw_runs.append((run_start, len(probabilities)))

    min_speech_windows = max(1, math.ceil(min_speech_ms / window_ms))
    min_silence_windows = max(1, math.ceil(min_silence_ms / window_ms))
    pad_samples = round(pad_ms * audio.sample_rate / 1000)

    merged_runs: list[tuple[int, int]] = []
    for start, end in raw_runs:
        if merged_runs and start - merged_runs[-1][1] <= min_silence_windows:
            previous_start, _ = merged_runs[-1]
            merged_runs[-1] = (previous_start, end)
        else:
            merged_runs.append((start, end))

    segments: list[SpeechSegment] = []
    for start, end in merged_runs:
        if end - start < min_speech_windows:
            continue

        segment_start_sample = max(0, start * window_samples - pad_samples)
        segment_end_sample = min(total_samples, end * window_samples + pad_samples)
        run_probabilities = probabilities[start:end]

        segments.append(
            SpeechSegment(
                start_ms=int(round(segment_start_sample * 1000 / audio.sample_rate)),
                end_ms=int(round(segment_end_sample * 1000 / audio.sample_rate)),
                duration_ms=int(round((segment_end_sample - segment_start_sample) * 1000 / audio.sample_rate)),
                average_probability=float(sum(run_probabilities) / len(run_probabilities)),
                peak_probability=max(run_probabilities),
            )
        )

    return VadAnalysis(
        threshold=threshold,
        window_ms=window_ms,
        duration_ms=audio.duration_ms,
        window_scores=[round(score, 6) for score in probabilities],
        segments=segments,
    )
