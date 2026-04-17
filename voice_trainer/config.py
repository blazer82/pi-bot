"""Configuration for the voice training pipeline."""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PACKAGE_DIR)

TRAINER_CONFIG = {
    # XTTS
    "xtts_model": "tts_models/multilingual/multi-dataset/xtts_v2",
    "language": "de",
    "speaker_wav": None,
    "xtts_speaker": None,

    # Corpus
    "sentences_file": os.path.join(_PACKAGE_DIR, "data", "sentences_de.txt"),
    "sample_rate": 22050,

    # Output
    "output_dir": os.path.join(_REPO_DIR, "voice_trainer_output"),
    "wavs_subdir": "wavs",
    "processed_subdir": "wavs_processed",
    "metadata_file": "metadata.csv",

    # Post-processing (pedalboard)
    "effects": {
        "pitch_shift_semitones": -2.0,
        "bitcrush_bit_depth": 14,
        "lowpass_cutoff_hz": 7000,
    },

    # Piper training
    "piper_batch_size": 32,
    "piper_max_epochs": 10000,
    "piper_validation_split": 0.01,
    "piper_num_test_examples": 5,

    # pi-bot integration
    "piper_model_output_dir": os.path.join(_REPO_DIR, "models", "piper"),
}
