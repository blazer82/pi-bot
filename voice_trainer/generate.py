"""Step 2: Batch-render sentences to WAV corpus in LJSpeech format."""

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

                wav = tts.tts(**kwargs)
                audio = np.array(wav, dtype=np.float32)

                tts_sr = tts.synthesizer.output_sample_rate if hasattr(tts, "synthesizer") else 24000
                audio = _resample(audio, tts_sr, target_sr)

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
