"""Split a DAW-processed concatenated WAV back into individual training clips."""

import json
import os
import sys

import numpy as np
import soundfile as sf


def _detect_silence_gaps(
    audio: np.ndarray,
    sr: int,
    min_silence_ms: float = 500.0,
    threshold_db: float = -35.0,
    frame_length_ms: float = 20.0,
) -> list[tuple[int, int]]:
    frame_len = int(sr * frame_length_ms / 1000)
    if frame_len == 0:
        return []

    num_frames = len(audio) // frame_len
    is_silent = np.zeros(num_frames, dtype=bool)

    for i in range(num_frames):
        start = i * frame_len
        frame = audio[start : start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)
        is_silent[i] = energy_db < threshold_db

    min_silent_frames = int(min_silence_ms / frame_length_ms)
    gaps = []
    gap_start = None

    for i in range(num_frames):
        if is_silent[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                if gap_len >= min_silent_frames:
                    gaps.append((gap_start * frame_len, i * frame_len))
                gap_start = None

    if gap_start is not None:
        gap_len = num_frames - gap_start
        if gap_len >= min_silent_frames:
            gaps.append((gap_start * frame_len, num_frames * frame_len))

    return gaps


def _split_by_silence(
    audio: np.ndarray,
    gaps: list[tuple[int, int]],
) -> list[np.ndarray]:
    if not gaps:
        return [audio]

    midpoints = [(start + end) // 2 for start, end in gaps]
    segments = []
    prev = 0

    for mid in midpoints:
        segments.append(audio[prev:mid])
        prev = mid

    segments.append(audio[prev:])
    return segments


def _split_by_markers(
    audio: np.ndarray,
    markers: list[dict],
    orig_sr: int,
    actual_sr: int,
) -> list[np.ndarray]:
    scale = len(audio) / (markers[-1]["end_sample"] * actual_sr / orig_sr) if markers else 1.0
    segments = []

    for m in markers:
        start = int(m["start_sample"] * actual_sr / orig_sr * scale)
        end = int(m["end_sample"] * actual_sr / orig_sr * scale)
        start = max(0, min(start, len(audio)))
        end = max(start, min(end, len(audio)))
        segments.append(audio[start:end])

    return segments


def _validate_against_markers(
    segments: list[np.ndarray],
    markers: list[dict],
    sr: int,
    tolerance_ratio: float = 0.3,
) -> list[str]:
    warnings = []

    if len(segments) != len(markers):
        warnings.append(
            f"Segment count mismatch: detected {len(segments)}, expected {len(markers)}"
        )
        return warnings

    for i, (seg, m) in enumerate(zip(segments, markers)):
        expected_dur = m["end_time"] - m["start_time"]
        actual_dur = len(seg) / sr
        if expected_dur > 0:
            ratio = abs(actual_dur - expected_dur) / expected_dur
            if ratio > tolerance_ratio:
                warnings.append(
                    f"Clip {m['file_id']}: duration {actual_dur:.2f}s vs expected {expected_dur:.2f}s "
                    f"(diff {ratio:.0%})"
                )

    return warnings


def split_wav(
    input_wav: str,
    markers_path: str,
    output_dir: str,
    config: dict,
) -> None:
    target_sr = config["sample_rate"]

    audio, file_sr = sf.read(input_wav, dtype="float32")
    if file_sr != target_sr:
        from voice_trainer.generate import _resample
        print(f"Resampling from {file_sr}Hz to {target_sr}Hz...")
        audio = _resample(audio, file_sr, target_sr)

    with open(markers_path, encoding="utf-8") as f:
        markers_data = json.load(f)

    markers = markers_data["clips"]
    orig_sr = markers_data["sample_rate"]

    print(f"Loaded {len(markers)} markers, detecting silence gaps...")

    gaps = _detect_silence_gaps(
        audio,
        target_sr,
        min_silence_ms=config.get("split_min_silence_ms", 500),
        threshold_db=config.get("split_silence_threshold_db", -35.0),
    )
    print(f"Found {len(gaps)} silence gaps")

    segments = _split_by_silence(audio, gaps)

    warnings = _validate_against_markers(segments, markers, target_sr)
    for w in warnings:
        print(f"  WARNING: {w}")

    use_markers = False
    if len(segments) != len(markers):
        print(f"Falling back to marker-based splitting...")
        segments = _split_by_markers(audio, markers, orig_sr, target_sr)
        use_markers = True

    os.makedirs(output_dir, exist_ok=True)

    clip_source = markers[:len(segments)]
    total_duration = 0.0

    for seg, m in zip(segments, clip_source):
        wav_path = os.path.join(output_dir, f"{m['file_id']}.wav")
        sf.write(wav_path, seg, target_sr)
        total_duration += len(seg) / target_sr

    metadata_path = os.path.join(os.path.dirname(output_dir), config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        for m in clip_source:
            f.write(f"{m['file_id']}|{m['text']}\n")

    method = "markers" if use_markers else "silence detection"
    print(f"\nSplit {len(clip_source)} clips via {method}")
    print(f"Total audio: {total_duration / 60:.1f} minutes")
    print(f"Output: {output_dir}")
    print(f"\nNext step: python -m voice_trainer train")


def run(args, config: dict) -> None:
    config = dict(config)

    input_wav = args.input_wav
    if not os.path.isfile(input_wav):
        print(f"Error: input WAV not found: {input_wav}", file=sys.stderr)
        sys.exit(1)

    output_base = args.output_dir or config["output_dir"]
    markers_path = args.markers or os.path.join(
        config["output_dir"], config.get("concat_markers_filename", "concat_markers.json")
    )

    if not os.path.isfile(markers_path):
        print(f"Error: markers file not found: {markers_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.join(output_base, config["processed_subdir"])
    split_wav(input_wav, markers_path, output_dir, config)
