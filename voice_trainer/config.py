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
    "xtts_temperature": 0.3,
    "xtts_gpt_cond_len": 12,

    # Corpus
    "sentences_file": os.path.join(_PACKAGE_DIR, "data", "sentences_de.txt"),
    "sample_rate": 22050,

    # Output
    "output_dir": os.path.join(_REPO_DIR, "voice_trainer_output"),
    "wavs_subdir": "wavs",
    "processed_subdir": "wavs_processed",
    "metadata_file": "metadata.csv",

    # Trim (energy-based endpoint detection)
    "trim_trailing": True,
    "trim_energy_threshold_db": -35.0,
    "trim_frame_length_ms": 20,
    "trim_min_trailing_silence_ms": 150,

    # Concatenation
    "concat_output": True,
    "concat_gap_seconds": 1.5,
    "concat_filename": "concat_full.wav",
    "concat_markers_filename": "concat_markers.json",

    # Split
    "split_min_silence_ms": 500,
    "split_silence_threshold_db": -35.0,

    # Post-processing (pedalboard)
    "effects": {
        "pitch_shift_semitones": -2.0,
        "bitcrush_bit_depth": 14,
        "lowpass_cutoff_hz": 7000,
    },

    # Piper training (fine-tuning via piper1-gpl)
    "piper_batch_size": 32,
    "piper_max_epochs": 1000,
    "piper_voice_name": "pi-bot",
    "piper_espeak_voice": "de",
    "piper_pretrained_checkpoint": "/workspace/checkpoints/pretrained.ckpt",
    "piper_config_path": "/workspace/checkpoints/config.json",
    "piper_cache_dir": "/workspace/cache",

    # pi-bot integration
    "piper_model_output_dir": os.path.join(_REPO_DIR, "models", "piper"),
}
