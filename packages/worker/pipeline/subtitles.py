"""Generate subtitles (SRT, VTT, CSV) from transcribed segments."""

from __future__ import annotations

import csv
import io

from pipeline.stt import TranscribedSegment


def _fmt_srt_time(ms: int) -> str:
    """Format milliseconds as SRT timestamp: HH:MM:SS,mmm"""
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    millis = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def _fmt_vtt_time(ms: int) -> str:
    """Format milliseconds as VTT timestamp: HH:MM:SS.mmm"""
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    millis = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{millis:03d}"


def segments_to_srt(segments: list[TranscribedSegment]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        speaker_prefix = f"[{seg.speaker_id}] " if seg.speaker_id else ""
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(seg.start_ms)} --> {_fmt_srt_time(seg.end_ms)}")
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments: list[TranscribedSegment]) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        speaker_prefix = f"<v {seg.speaker_id}>" if seg.speaker_id else ""
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(seg.start_ms)} --> {_fmt_vtt_time(seg.end_ms)}")
        lines.append(f"{speaker_prefix}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def segments_to_csv(segments: list[TranscribedSegment]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["index", "start_ms", "end_ms", "duration_ms", "speaker_id", "text"])
    for i, seg in enumerate(segments, 1):
        writer.writerow([i, seg.start_ms, seg.end_ms, seg.duration_ms, seg.speaker_id or "", seg.text])
    return buf.getvalue()
