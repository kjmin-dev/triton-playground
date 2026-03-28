from __future__ import annotations

import json
import time

import numpy as np
import triton_python_backend_utils as pb_utils

_REF_MIN_DURATION_MS = 2000
_REF_MAX_DURATION_MS = 10000

_N_MELS = 40
_HOP_LENGTH = 160
_WIN_LENGTH = 400


def _decode_string(value: object) -> str:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    if isinstance(scalar, str):
        return scalar
    return str(scalar)


def _tensor_as_string(request, name: str) -> str:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    flattened = tensor.as_numpy().reshape(-1)
    if flattened.size == 0:
        return ""
    return _decode_string(flattened[0])


def _tensor_as_int(request, name: str) -> int:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    return int(tensor.as_numpy().reshape(-1)[0])


def _tensor_as_float(request, name: str) -> float:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    return float(tensor.as_numpy().reshape(-1)[0])


def _tensor_as_audio(request, name: str) -> np.ndarray:
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"missing required input tensor: {name}")
    samples = tensor.as_numpy().reshape(-1).astype(np.float32)
    if samples.size == 0:
        raise pb_utils.TritonModelException("audio_pcm tensor is empty")
    return samples


def _request_output(response, name: str):
    tensor = pb_utils.get_output_tensor_by_name(response, name)
    if tensor is None:
        raise pb_utils.TritonModelException(f"downstream model did not return {name}")
    return tensor.as_numpy()


def _execute_subrequest(model_name: str, inputs: list[pb_utils.Tensor], requested_output_names: list[str]):
    inference_request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=requested_output_names,
        inputs=inputs,
    )
    response = inference_request.exec()
    if response.has_error():
        raise pb_utils.TritonModelException(response.error().message())
    return response


def _slice_audio(samples: np.ndarray, sample_rate: int, start_ms: int, end_ms: int) -> np.ndarray:
    total_samples = len(samples)
    start_sample = max(0, min(total_samples, round(start_ms * sample_rate / 1000)))
    end_sample = max(start_sample, min(total_samples, round(end_ms * sample_rate / 1000)))
    if end_sample == start_sample:
        end_sample = min(total_samples, start_sample + 1)
    return samples[start_sample:end_sample].astype(np.float32)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
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


def _compute_embedding(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fft = _WIN_LENGTH
    fb = _mel_filterbank(sample_rate, n_fft, _N_MELS)
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
    similarity = embeddings @ embeddings.T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _agglomerative_cluster(distances: np.ndarray, threshold: float = 0.4) -> list[int]:
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

    unique = sorted(set(labels))
    mapping = {old: idx for idx, old in enumerate(unique)}
    return [mapping[label] for label in labels]


def _assign_speakers(
    audio_pcm: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, object]],
    *,
    min_segment_ms: int = 500,
    distance_threshold: float = 0.4,
) -> list[str | None]:
    if not segments:
        return []
    if len(segments) == 1:
        return ["speaker_0"]

    embeddings: list[np.ndarray] = []
    embeddable_indices: list[int] = []
    for index, segment in enumerate(segments):
        if int(segment["duration_ms"]) < min_segment_ms:
            continue
        seg_audio = _slice_audio(audio_pcm, sample_rate, int(segment["start_ms"]), int(segment["end_ms"]))
        embeddings.append(_compute_embedding(seg_audio, sample_rate))
        embeddable_indices.append(index)

    if not embeddings:
        return [None] * len(segments)

    distances = _cosine_distance_matrix(np.stack(embeddings))
    cluster_labels = _agglomerative_cluster(distances, threshold=distance_threshold)

    speaker_ids: list[str | None] = [None] * len(segments)
    for index, cluster_id in zip(embeddable_indices, cluster_labels, strict=True):
        speaker_ids[index] = f"speaker_{cluster_id}"
    return speaker_ids


def _group_segments_by_speaker(segments: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for segment in segments:
        speaker_id = str(segment.get("speaker_id") or "speaker_0")
        groups.setdefault(speaker_id, []).append(segment)
    return groups


def _select_reference_segment(
    audio_pcm: np.ndarray,
    sample_rate: int,
    segments: list[dict[str, object]],
    speaker_id: str | None = None,
) -> tuple[np.ndarray, str] | None:
    filtered = segments
    if speaker_id is not None:
        filtered = [segment for segment in segments if segment.get("speaker_id") == speaker_id]
        if not filtered:
            filtered = segments

    candidates = [
        segment
        for segment in filtered
        if str(segment.get("text", "")).strip()
        and _REF_MIN_DURATION_MS <= int(segment["duration_ms"]) <= _REF_MAX_DURATION_MS
    ]
    if not candidates:
        candidates = [
            segment
            for segment in filtered
            if str(segment.get("text", "")).strip() and int(segment["duration_ms"]) >= 500
        ]
    if not candidates:
        return None

    best = max(candidates, key=lambda segment: float(segment["average_probability"]))
    ref_audio = _slice_audio(audio_pcm, sample_rate, int(best["start_ms"]), int(best["end_ms"]))
    return ref_audio, str(best.get("text", ""))


def _time_stretch(samples: np.ndarray, target_length: int) -> np.ndarray:
    if len(samples) == 0 or target_length <= 0:
        return np.zeros(target_length, dtype=np.float32)
    if len(samples) == target_length:
        return samples.astype(np.float32)
    src_pos = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    tgt_pos = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    return np.interp(tgt_pos, src_pos, samples).astype(np.float32)


def _serialize_segments(segments: list[dict[str, object]]) -> str:
    serialized = []
    for segment in segments:
        item = dict(segment)
        if item.get("speaker_id") is None:
            item.pop("speaker_id", None)
        serialized.append(item)
    return json.dumps(serialized)


class TritonPythonModel:
    def initialize(self, args):
        _ = args
        self._localize_text_pipeline_model_name = "localize_text_pipeline"
        self._tts_model_name = "qwen3_tts_0_6b"

    def _run_tts_batch(
        self, requests: list[dict[str, object]], language: str, speaker_prompt: str
    ) -> list[tuple[np.ndarray, int]]:
        if not requests:
            return []

        max_ref_samples = max(
            (
                int(request["ref_audio"].size)
                if isinstance(request["ref_audio"], np.ndarray) and request["ref_audio"].size > 1
                else 1
            )
            for request in requests
        )
        ref_audio_batch = np.zeros((len(requests), max_ref_samples), dtype=np.float32)
        ref_audio_lengths = np.ones(len(requests), dtype=np.int32)

        for index, request in enumerate(requests):
            ref_audio = request["ref_audio"]
            if not isinstance(ref_audio, np.ndarray) or ref_audio.size <= 1:
                continue
            sample_count = int(ref_audio.size)
            ref_audio_batch[index, :sample_count] = ref_audio.astype(np.float32).reshape(-1)
            ref_audio_lengths[index] = sample_count

        response = _execute_subrequest(
            self._tts_model_name,
            [
                pb_utils.Tensor("text", np.asarray([str(request["text"]) for request in requests], dtype=object)),
                pb_utils.Tensor("language", np.asarray([language] * len(requests), dtype=object)),
                pb_utils.Tensor("speaker_prompt", np.asarray([speaker_prompt] * len(requests), dtype=object)),
                pb_utils.Tensor("ref_audio", ref_audio_batch),
                pb_utils.Tensor("ref_audio_lengths", ref_audio_lengths),
                pb_utils.Tensor(
                    "ref_text", np.asarray([str(request["ref_text"]) for request in requests], dtype=object)
                ),
            ],
            ["audio_pcm", "audio_lengths", "sample_rate"],
        )

        audio_batch = np.asarray(_request_output(response, "audio_pcm"), dtype=np.float32)
        if audio_batch.ndim == 1:
            audio_batch = audio_batch.reshape(1, -1)
        audio_lengths = np.asarray(_request_output(response, "audio_lengths"), dtype=np.int32).reshape(-1)
        sample_rates = np.asarray(_request_output(response, "sample_rate"), dtype=np.int32).reshape(-1)

        outputs: list[tuple[np.ndarray, int]] = []
        for index in range(len(requests)):
            audio_length = int(audio_lengths[index])
            waveform = audio_batch[index].reshape(-1)[:audio_length].astype(np.float32)
            outputs.append((waveform, int(sample_rates[index])))
        return outputs

    def _synthesize_time_aligned(
        self,
        audio_pcm: np.ndarray,
        sample_rate: int,
        diarized_segments: list[dict[str, object]],
        speaker_groups: dict[str, list[dict[str, object]]],
        tts_language: str,
        speaker_prompt: str,
    ) -> tuple[np.ndarray, int, dict[str, object]]:
        speaker_refs: dict[str, tuple[np.ndarray, str] | None] = {}
        for speaker_id in speaker_groups:
            speaker_refs[speaker_id] = _select_reference_segment(
                audio_pcm,
                sample_rate,
                diarized_segments,
                speaker_id=speaker_id,
            )

        pending_requests: list[dict[str, object]] = []
        for segment in diarized_segments:
            if not str(segment.get("text", "")).strip():
                continue
            speaker_id = str(segment.get("speaker_id") or "speaker_0")
            ref = speaker_refs.get(speaker_id)
            pending_requests.append(
                {
                    "segment": segment,
                    "text": str(segment.get("text", "")),
                    "ref_audio": ref[0] if ref is not None else None,
                    "ref_text": ref[1] if ref is not None else "",
                }
            )

        synthesized_segments: list[tuple[dict[str, object], np.ndarray, int]] = []
        if pending_requests:
            try:
                batch_outputs = self._run_tts_batch(pending_requests, tts_language, speaker_prompt)
                synthesized_segments = [
                    (request["segment"], waveform, synth_sample_rate)
                    for request, (waveform, synth_sample_rate) in zip(
                        pending_requests,
                        batch_outputs,
                        strict=True,
                    )
                ]
            except pb_utils.TritonModelException:
                synthesized_segments = []

        if not synthesized_segments:
            for request in pending_requests:
                try:
                    waveform, synth_sample_rate = self._run_tts_batch([request], tts_language, speaker_prompt)[0]
                except pb_utils.TritonModelException:
                    continue
                synthesized_segments.append((request["segment"], waveform, synth_sample_rate))

        output_sample_rate = 24000
        for _, _, synth_sample_rate in synthesized_segments:
            output_sample_rate = synth_sample_rate
            break

        total_samples = int(round(len(audio_pcm) * output_sample_rate / sample_rate))
        output = np.zeros(total_samples, dtype=np.float32)

        segments_synthesized = 0
        for segment, waveform, _ in synthesized_segments:
            target_samples = int(int(segment["duration_ms"]) * output_sample_rate / 1000)
            stretched = _time_stretch(waveform, target_samples)
            start_sample = int(int(segment["start_ms"]) * output_sample_rate / 1000)
            end_sample = min(start_sample + len(stretched), len(output))
            fit_len = end_sample - start_sample
            if fit_len > 0:
                output[start_sample:end_sample] = stretched[:fit_len]
            segments_synthesized += 1

        if diarized_segments:
            last_segment = max(diarized_segments, key=lambda segment: int(segment["end_ms"]))
            trim_sample = min(len(output), int((int(last_segment["end_ms"]) + 500) * output_sample_rate / 1000))
            output = output[:trim_sample]

        vc_mode = "icl" if any(ref and ref[1] for ref in speaker_refs.values()) else "x_vector"
        return (
            output,
            output_sample_rate,
            {
                "status": "ok",
                "voice_cloning": True,
                "voice_cloning_mode": vc_mode,
                "speaker_count": len(speaker_groups),
                "segments_synthesized": segments_synthesized,
                "time_aligned": True,
                "speakers": sorted(speaker_groups.keys()),
            },
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_pcm = _tensor_as_audio(request, "audio_pcm")
            sample_rate = _tensor_as_int(request, "sample_rate")
            threshold = _tensor_as_float(request, "threshold")
            min_speech_ms = _tensor_as_int(request, "min_speech_ms")
            min_silence_ms = _tensor_as_int(request, "min_silence_ms")
            pad_ms = _tensor_as_int(request, "pad_ms")
            window_samples = _tensor_as_int(request, "window_samples")
            source_language = _tensor_as_string(request, "source_language")
            target_language = _tensor_as_string(request, "target_language")
            prompt = _tensor_as_string(request, "prompt")
            speaker_prompt = _tensor_as_string(request, "speaker_prompt")

            localize_text_response = _execute_subrequest(
                self._localize_text_pipeline_model_name,
                [
                    pb_utils.Tensor("audio_pcm", audio_pcm.reshape(1, -1).astype(np.float32)),
                    pb_utils.Tensor("sample_rate", np.asarray([sample_rate], dtype=np.int32)),
                    pb_utils.Tensor("threshold", np.asarray([threshold], dtype=np.float32)),
                    pb_utils.Tensor("min_speech_ms", np.asarray([min_speech_ms], dtype=np.int32)),
                    pb_utils.Tensor("min_silence_ms", np.asarray([min_silence_ms], dtype=np.int32)),
                    pb_utils.Tensor("pad_ms", np.asarray([pad_ms], dtype=np.int32)),
                    pb_utils.Tensor("window_samples", np.asarray([window_samples], dtype=np.int32)),
                    pb_utils.Tensor("source_language", np.asarray([source_language], dtype=object)),
                    pb_utils.Tensor("target_language", np.asarray([target_language], dtype=object)),
                    pb_utils.Tensor("prompt", np.asarray([prompt], dtype=object)),
                ],
                [
                    "transcript",
                    "segments_json",
                    "translated_text",
                    "stt_elapsed_ms",
                    "translation_elapsed_ms",
                ],
            )

            transcript = _decode_string(_request_output(localize_text_response, "transcript").reshape(-1)[0]).strip()
            translated_text = _decode_string(
                _request_output(localize_text_response, "translated_text").reshape(-1)[0]
            ).strip()
            stt_elapsed_ms = int(np.asarray(_request_output(localize_text_response, "stt_elapsed_ms")).reshape(-1)[0])
            translation_elapsed_ms = int(
                np.asarray(_request_output(localize_text_response, "translation_elapsed_ms")).reshape(-1)[0]
            )
            segments_json = _decode_string(_request_output(localize_text_response, "segments_json").reshape(-1)[0])
            segments = json.loads(segments_json or "[]")

            speaker_ids = _assign_speakers(audio_pcm, sample_rate, segments)
            diarized_segments: list[dict[str, object]] = []
            for segment, speaker_id in zip(segments, speaker_ids, strict=True):
                item = dict(segment)
                if speaker_id is not None:
                    item["speaker_id"] = speaker_id
                diarized_segments.append(item)

            tts_elapsed_ms = 0
            audio_output = np.zeros(1, dtype=np.float32)
            audio_length = 0
            synthesized_sample_rate = 24000

            if not transcript:
                tts_meta = {"status": "skipped", "reason": "No transcript text was produced."}
            elif not translated_text:
                tts_meta = {"status": "skipped", "reason": "Translation returned empty text."}
            else:
                t0 = time.perf_counter()
                speaker_groups = _group_segments_by_speaker(diarized_segments)
                audio_output, synthesized_sample_rate, tts_meta = self._synthesize_time_aligned(
                    audio_pcm,
                    sample_rate,
                    diarized_segments,
                    speaker_groups,
                    target_language,
                    speaker_prompt,
                )
                tts_elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
                audio_length = int(audio_output.size)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("transcript", np.asarray([transcript], dtype=object)),
                        pb_utils.Tensor(
                            "segments_json", np.asarray([_serialize_segments(diarized_segments)], dtype=object)
                        ),
                        pb_utils.Tensor("translated_text", np.asarray([translated_text], dtype=object)),
                        pb_utils.Tensor("stt_elapsed_ms", np.asarray([stt_elapsed_ms], dtype=np.int32)),
                        pb_utils.Tensor("translation_elapsed_ms", np.asarray([translation_elapsed_ms], dtype=np.int32)),
                        pb_utils.Tensor("tts_elapsed_ms", np.asarray([tts_elapsed_ms], dtype=np.int32)),
                        pb_utils.Tensor("audio_pcm", audio_output.astype(np.float32)),
                        pb_utils.Tensor("audio_length", np.asarray([audio_length], dtype=np.int32)),
                        pb_utils.Tensor(
                            "synthesized_sample_rate",
                            np.asarray([synthesized_sample_rate], dtype=np.int32),
                        ),
                        pb_utils.Tensor("tts_meta_json", np.asarray([json.dumps(tts_meta)], dtype=object)),
                    ]
                )
            )

        return responses
