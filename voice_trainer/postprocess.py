"""Step 3: Apply audio effects to generated WAVs."""

import os
import sys

import numpy as np
import soundfile as sf
from pedalboard import Pedalboard, PitchShift, Bitcrush, LowpassFilter
from pedalboard.io import AudioFile


def build_effects_chain(config: dict) -> Pedalboard:
    effects = config["effects"]
    return Pedalboard([
        PitchShift(semitones=effects["pitch_shift_semitones"]),
        Bitcrush(bit_depth=effects["bitcrush_bit_depth"]),
        LowpassFilter(cutoff_frequency_hz=effects["lowpass_cutoff_hz"]),
    ])


def process_all(input_dir: str, output_dir: str, board: Pedalboard) -> None:
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".wav"))
    if not wav_files:
        print(f"No WAV files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    total = len(wav_files)
    print(f"Processing {total} files...")

    for i, filename in enumerate(wav_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with AudioFile(input_path) as inp:
            audio = inp.read(inp.frames)
            sr = inp.samplerate

        processed = board(audio, sr)
        sf.write(output_path, processed.T, sr)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i + 1}/{total}]")

    parent_dir = os.path.dirname(input_dir)
    metadata_src = os.path.join(parent_dir, "metadata.csv")
    if os.path.isfile(metadata_src):
        metadata_dst = os.path.join(os.path.dirname(output_dir), "metadata.csv")
        if os.path.abspath(metadata_src) != os.path.abspath(metadata_dst):
            import shutil
            shutil.copy2(metadata_src, metadata_dst)

    print(f"\nProcessed WAVs saved to: {output_dir}")


def run(args, config: dict) -> None:
    config = dict(config)
    effects = dict(config["effects"])

    if args.pitch is not None:
        effects["pitch_shift_semitones"] = args.pitch
    if args.bitcrush is not None:
        effects["bitcrush_bit_depth"] = args.bitcrush
    if args.lowpass is not None:
        effects["lowpass_cutoff_hz"] = args.lowpass

    config["effects"] = effects

    output_base = config["output_dir"]
    input_dir = args.input_dir or os.path.join(output_base, config["wavs_subdir"])
    output_dir = args.output_dir or os.path.join(output_base, config["processed_subdir"])

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        print("Run 'python -m voice_trainer generate' first.")
        sys.exit(1)

    print(f"Effects: pitch={effects['pitch_shift_semitones']:+.1f}st, "
          f"bitcrush={effects['bitcrush_bit_depth']}bit, "
          f"lowpass={effects['lowpass_cutoff_hz']}Hz")

    board = build_effects_chain(config)
    process_all(input_dir, output_dir, board)

    print(f"\nNext step: python -m voice_trainer train --dataset-dir {output_base}")
