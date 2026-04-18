"""Step 2: Batch-render sentences to WAV corpus in LJSpeech format."""

import json
import os
import sys
import time

import numpy as np
import soundfile as sf


def load_sentences(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample as scipy_resample

    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy_resample(audio, num_samples)


def _spectral_flatness(frame: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(frame))
    spectrum = spectrum[1:]
    if len(spectrum) == 0 or np.max(spectrum) < 1e-10:
        return 0.0
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
    arith_mean = np.mean(spectrum)
    if arith_mean < 1e-10:
        return 0.0
    return geo_mean / arith_mean


def _trim_trailing_artifact(
    audio: np.ndarray,
    sr: int,
    text: str = "",
    threshold_db: float = -35.0,
    frame_length_ms: float = 20.0,
    min_trailing_ms: float = 150.0,
    flatness_threshold: float = 0.4,
    chars_per_second: float = 12.0,
) -> np.ndarray:
    frame_len = int(sr * frame_length_ms / 1000)
    if frame_len == 0 or len(audio) < frame_len:
        return audio

    if text:
        max_duration = max(1.0, len(text) / chars_per_second + 1.5)
        max_samples = int(sr * max_duration)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    num_frames = len(audio) // frame_len
    last_speech_frame = 0

    for i in range(num_frames - 1, -1, -1):
        start = i * frame_len
        frame = audio[start : start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        energy_db = 20 * np.log10(rms + 1e-10)
        if energy_db <= threshold_db:
            continue
        flatness = _spectral_flatness(frame)
        if flatness < flatness_threshold:
            last_speech_frame = i
            break

    cut_sample = (last_speech_frame + 1) * frame_len
    if cut_sample < int(sr * 0.1):
        return audio

    tail = int(sr * min_trailing_ms / 1000)
    cut_sample = min(cut_sample + tail, len(audio))
    return audio[:cut_sample]


def _build_concat(
    wavs_dir: str,
    output_dir: str,
    metadata_path: str,
    config: dict,
) -> None:
    sr = config["sample_rate"]
    gap_seconds = config.get("concat_gap_seconds", 1.5)
    silence = np.zeros(int(sr * gap_seconds), dtype=np.float32)

    seen = {}
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                seen[parts[0]] = parts[1]
    entries = sorted(seen.items())

    segments = []
    markers = []
    current_sample = 0

    for file_id, text in entries:
        wav_path = os.path.join(wavs_dir, f"{file_id}.wav")
        if not os.path.isfile(wav_path):
            continue

        audio, file_sr = sf.read(wav_path, dtype="float32")
        if file_sr != sr:
            audio = _resample(audio, file_sr, sr)

        if segments:
            segments.append(silence)
            current_sample += len(silence)

        start_sample = current_sample
        end_sample = current_sample + len(audio)

        markers.append({
            "index": len(markers),
            "file_id": file_id,
            "text": text,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "start_time": start_sample / sr,
            "end_time": end_sample / sr,
        })

        segments.append(audio)
        current_sample = end_sample

    if not segments:
        print("No WAVs found to concatenate.")
        return

    concat_audio = np.concatenate(segments)

    concat_path = os.path.join(output_dir, config.get("concat_filename", "concat_full.wav"))
    sf.write(concat_path, concat_audio, sr)

    markers_data = {
        "sample_rate": sr,
        "gap_seconds": gap_seconds,
        "total_duration": len(concat_audio) / sr,
        "num_clips": len(markers),
        "clips": markers,
    }
    markers_path = os.path.join(
        output_dir, config.get("concat_markers_filename", "concat_markers.json")
    )
    with open(markers_path, "w", encoding="utf-8") as f:
        json.dump(markers_data, f, indent=2, ensure_ascii=False)

    print(f"Concatenated WAV: {concat_path} ({len(concat_audio) / sr:.1f}s)")
    print(f"Markers: {markers_path} ({len(markers)} clips)")


def generate_corpus(tts, sentences: list[str], config: dict) -> None:
    output_dir = config["output_dir"]
    wavs_dir = os.path.join(output_dir, config["wavs_subdir"])
    metadata_path = os.path.join(output_dir, config["metadata_file"])
    target_sr = config["sample_rate"]

    os.makedirs(wavs_dir, exist_ok=True)

    total = len(sentences)
    total_duration = 0.0
    skipped = 0
    failed = 0
    start_time = time.time()

    speaker_wav = config.get("speaker_wav")
    xtts_speaker = config.get("xtts_speaker")

    if not speaker_wav and not xtts_speaker:
        from voice_trainer.xtts_setup import _get_speakers

        speakers = _get_speakers(tts)
        if speakers:
            xtts_speaker = sorted(speakers.keys())[0]
            print(f"No speaker specified, using default: {xtts_speaker}")

    existing_entries = {}
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|", 1)
                if len(parts) == 2:
                    existing_entries[parts[0]] = parts[1]

    meta_file = open(metadata_path, "a", encoding="utf-8")

    try:
        for i, sentence in enumerate(sentences):
            file_id = f"{i:06d}"
            wav_path = os.path.join(wavs_dir, f"{file_id}.wav")

            if os.path.isfile(wav_path) and file_id in existing_entries:
                skipped += 1
                info = sf.info(wav_path)
                total_duration += info.duration
                continue

            print(f"[{i + 1}/{total}] {sentence[:60]}{'...' if len(sentence) > 60 else ''}")

            try:
                kwargs = {
                    "text": sentence,
                    "language": config["language"],
                }
                if speaker_wav:
                    kwargs["speaker_wav"] = speaker_wav
                elif xtts_speaker:
                    kwargs["speaker"] = xtts_speaker

                synth_kwargs = {}
                if config.get("xtts_temperature") is not None:
                    synth_kwargs["temperature"] = config["xtts_temperature"]
                if config.get("xtts_gpt_cond_len") is not None:
                    synth_kwargs["gpt_cond_len"] = config["xtts_gpt_cond_len"]
                if synth_kwargs:
                    kwargs.update(synth_kwargs)

                wav = tts.tts(**kwargs)
                audio = np.array(wav, dtype=np.float32)

                tts_sr = tts.synthesizer.output_sample_rate if hasattr(tts, "synthesizer") else 24000
                audio = _resample(audio, tts_sr, target_sr)

                if config.get("trim_trailing", True):
                    audio = _trim_trailing_artifact(
                        audio,
                        target_sr,
                        text=sentence,
                        threshold_db=config.get("trim_energy_threshold_db", -35.0),
                        frame_length_ms=config.get("trim_frame_length_ms", 20),
                        min_trailing_ms=config.get("trim_min_trailing_silence_ms", 150),
                    )

                sf.write(wav_path, audio, target_sr)

                duration = len(audio) / target_sr
                total_duration += duration

                meta_file.write(f"{file_id}|{sentence}\n")
                meta_file.flush()

            except Exception as e:
                failed += 1
                print(f"  FAILED: {e}")
                continue

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                generated = i + 1 - skipped - failed
                if generated > 0:
                    rate = elapsed / generated
                    remaining = (total - i - 1) * rate
                    print(
                        f"  ETA: {remaining / 60:.0f}min | "
                        f"Total audio: {total_duration / 60:.1f}min"
                    )
    finally:
        meta_file.close()

    print(f"\nDone: {total - skipped - failed} generated, {skipped} skipped, {failed} failed")
    print(f"Total audio duration: {total_duration / 60:.1f} minutes")
    print(f"Output: {output_dir}")

    if config.get("concat_output", True):
        print("\nBuilding concatenated WAV...")
        _build_concat(wavs_dir, output_dir, metadata_path, config)


def run(args, config: dict) -> None:
    from voice_trainer.xtts_setup import load_xtts

    config = dict(config)

    if args.sentences:
        config["sentences_file"] = args.sentences
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.speaker_wav:
        config["speaker_wav"] = args.speaker_wav

    sentences_file = config["sentences_file"]
    if not os.path.isfile(sentences_file):
        print(f"Error: sentences file not found: {sentences_file}", file=sys.stderr)
        sys.exit(1)

    sentences = load_sentences(sentences_file)
    if args.max_sentences:
        sentences = sentences[: args.max_sentences]

    print(f"Loaded {len(sentences)} sentences from {sentences_file}")

    tts = load_xtts(config)
    generate_corpus(tts, sentences, config)

    print(f"\nNext step: python -m voice_trainer postprocess")
