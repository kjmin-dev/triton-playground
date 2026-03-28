"""Lightweight speaker diarization using mel-spectrogram embeddings.

Assigns a speaker_id to each VAD segment by computing per-segment spectral
embeddings and clustering them with agglomerative cosine distance. No external
diarization library required — runs on numpy only.
"""

from __future__ import annotations

import logging

import numpy as np

from pipeline.audio import AudioBuffer
from pipeline.stt import TranscribedSegment, _slice_audio
from pipeline.vad import SpeechSegment

logger = logging.getLogger(__name__)

_N_MELS = 40
_HOP_LENGTH = 160  # 10ms at 16kHz
_WIN_LENGTH = 400  # 25ms at 16kHz
_SAMPLE_RATE = 16000


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Build a mel filterbank matrix (n_mels, n_fft//2+1)."""
    low_freq = 0.0
    high_freq = sample_rate / 2.0
    mel_low = 2595.0 * np.log10(1.0 + low_freq / 700.0)
    mel_high = 2595.0 * np.log10(1.0 + high_freq / 700.0)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, center):
            fb[i, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fb[i, j] = (right - j) / max(right - center, 1)
    return fb


def _compute_embedding(samples: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> np.ndarray:
    """Compute a fixed-size spectral embedding for an audio segment.

    Returns a normalized vector of shape (n_mels * 2,) combining mean and std
    of the mel spectrogram across time.
    """
    n_fft = _WIN_LENGTH
    fb = _mel_filterbank(sample_rate, n_fft, _N_MELS)

    # Short-time Fourier transform
    num_frames = max(1, 1 + (len(samples) - _WIN_LENGTH) // _HOP_LENGTH)
    window = np.hanning(_WIN_LENGTH).astype(np.float32)
    mel_spec = np.zeros((num_frames, _N_MELS), dtype=np.float32)

    for i in range(num_frames):
        start = i * _HOP_LENGTH
        frame = samples[start : start + _WIN_LENGTH]
        if len(frame) < _WIN_LENGTH:
            frame = np.pad(frame, (0, _WIN_LENGTH - len(frame)))
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        mel_spec[i] = fb @ spectrum

    mel_spec = np.log1p(mel_spec)
    embedding = np.concatenate([mel_spec.mean(axis=0), mel_spec.std(axis=0)])

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance matrix."""
    similarity = embeddings @ embeddings.T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _agglomerative_cluster(distances: np.ndarray, threshold: float = 0.4) -> list[int]:
    """Simple agglomerative clustering by minimum linkage."""
    n = len(distances)
    labels = list(range(n))

    while True:
        min_dist = float("inf")
        merge_a, merge_b = -1, -1

        unique_labels = sorted(set(labels))
        if len(unique_labels) <= 1:
            break

        for i, la in enumerate(unique_labels):
            for lb in unique_labels[i + 1 :]:
                members_a = [k for k in range(n) if labels[k] == la]
                members_b = [k for k in range(n) if labels[k] == lb]
                dist = min(distances[a, b] for a in members_a for b in members_b)
                if dist < min_dist:
                    min_dist = dist
                    merge_a, merge_b = la, lb

        if min_dist > threshold:
            break

        for k in range(n):
            if labels[k] == merge_b:
                labels[k] = merge_a

    # Renumber to 0-based
    unique = sorted(set(labels))
    mapping = {old: idx for idx, old in enumerate(unique)}
    return [mapping[label] for label in labels]


def assign_speakers(
    audio: AudioBuffer,
    segments: list[SpeechSegment],
    *,
    min_segment_ms: int = 500,
    distance_threshold: float = 0.4,
) -> list[str | None]:
    """Assign speaker IDs to VAD segments via spectral embedding clustering.

    Returns a list of speaker_id strings parallel to the input segments.
    Segments shorter than min_segment_ms get None (too short for reliable embedding).
    """
    if not segments:
        return []
    if len(segments) == 1:
        return ["speaker_0"]

    embeddings: list[np.ndarray] = []
    embeddable_indices: list[int] = []

    for i, seg in enumerate(segments):
        if seg.duration_ms < min_segment_ms:
            continue
        seg_audio = _slice_audio(audio, seg.start_ms, seg.end_ms)
        emb = _compute_embedding(seg_audio.samples, seg_audio.sample_rate)
        embeddings.append(emb)
        embeddable_indices.append(i)

    if not embeddings:
        return [None] * len(segments)

    emb_matrix = np.stack(embeddings)
    distances = _cosine_distance_matrix(emb_matrix)
    cluster_labels = _agglomerative_cluster(distances, threshold=distance_threshold)

    result: list[str | None] = [None] * len(segments)
    for idx, cluster_id in zip(embeddable_indices, cluster_labels, strict=True):
        result[idx] = f"speaker_{cluster_id}"

    n_speakers = len(set(sid for sid in result if sid is not None))
    logger.info("Diarization: %d speakers detected from %d segments", n_speakers, len(segments))
    return result


def assign_speakers_to_transcribed(
    audio: AudioBuffer,
    segments: list[TranscribedSegment],
    **kwargs,
) -> list[TranscribedSegment]:
    """Return new TranscribedSegments with speaker_id populated."""
    vad_segments = [
        SpeechSegment(
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            duration_ms=s.duration_ms,
            average_probability=s.average_probability,
            peak_probability=s.peak_probability,
        )
        for s in segments
    ]
    speaker_ids = assign_speakers(audio, vad_segments, **kwargs)
    return [
        TranscribedSegment(
            start_ms=s.start_ms,
            end_ms=s.end_ms,
            duration_ms=s.duration_ms,
            average_probability=s.average_probability,
            peak_probability=s.peak_probability,
            text=s.text,
            speaker_id=sid,
        )
        for s, sid in zip(segments, speaker_ids, strict=True)
    ]
