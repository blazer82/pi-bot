"""Step 1: XTTS voice setup and testing."""

import os
import sys

import numpy as np


def _get_speakers(tts) -> dict | None:
    try:
        return tts.synthesizer.tts_model.speaker_manager.speakers
    except AttributeError:
        return None


def load_xtts(config: dict):
    from TTS.api import TTS

    model_name = config["xtts_model"]
    print(f"Loading XTTS model: {model_name}")
    print("(First run will download ~2GB — this may take a while)")

    tts = TTS(model_name)

    if hasattr(tts, "synthesizer") and hasattr(tts.synthesizer, "tts_model"):
        device = next(tts.synthesizer.tts_model.parameters()).device
        print(f"Running on: {device}")
    else:
        print("Running on: CPU (GPU recommended for faster synthesis)")

    return tts


def list_speakers(tts) -> None:
    speakers = _get_speakers(tts)
    if speakers:
        print(f"\nAvailable speakers ({len(speakers)}):")
        for name in sorted(speakers.keys()):
            print(f"  {name}")
    else:
        print("\nNo built-in speakers found. Use --speaker-wav for voice cloning.")


def test_synthesis(
    tts, text: str, config: dict, speaker_wav: str | None = None
) -> str:
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_xtts.wav")

    kwargs = {
        "text": text,
        "language": config["language"],
        "file_path": output_path,
    }

    if speaker_wav:
        if not os.path.isfile(speaker_wav):
            print(f"Error: speaker WAV not found: {speaker_wav}", file=sys.stderr)
            sys.exit(1)
        kwargs["speaker_wav"] = speaker_wav
        print(f"Using voice clone reference: {speaker_wav}")
    elif config.get("xtts_speaker"):
        kwargs["speaker"] = config["xtts_speaker"]
        print(f"Using built-in speaker: {config['xtts_speaker']}")
    else:
        speakers = _get_speakers(tts)
        if speakers:
            default_speaker = sorted(speakers.keys())[0]
            kwargs["speaker"] = default_speaker
            print(f"No speaker specified, using default: {default_speaker}")
            print("Use --speaker-wav to clone a voice, or --list-speakers to see options.")
        else:
            print("Error: no --speaker-wav provided and no built-in speakers found.", file=sys.stderr)
            sys.exit(1)

    if config.get("xtts_temperature") is not None:
        kwargs["temperature"] = config["xtts_temperature"]
    if config.get("xtts_gpt_cond_len") is not None:
        kwargs["gpt_cond_len"] = config["xtts_gpt_cond_len"]

    print(f"Synthesizing: \"{text}\"")
    tts.tts_to_file(**kwargs)

    import soundfile as sf

    info = sf.info(output_path)
    print(f"\nSaved: {output_path}")
    print(f"Duration: {info.duration:.1f}s | Sample rate: {info.samplerate}Hz")

    return output_path


def run(args, config: dict) -> None:
    tts = load_xtts(config)

    if args.list_speakers:
        list_speakers(tts)
        return

    speaker_wav = args.speaker_wav or config.get("speaker_wav")
    test_synthesis(tts, args.test_text, config, speaker_wav=speaker_wav)
    print("\nXTTS setup complete. You can now run: python -m voice_trainer generate")
