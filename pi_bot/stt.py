"""Speech-to-Text via whisper.cpp."""

import numpy as np

from pi_bot.config import CONFIG


def _normalize(audio):
    """Peak-normalize int16 audio to float32 in [-1, 1]."""
    f32 = audio.astype(np.float32)
    peak = np.max(np.abs(f32))
    if peak < 1:
        return np.zeros_like(f32)
    return f32 / peak


def warmup(whisper_model):
    """Run a short silent transcription to warm up the model."""
    silence = np.zeros(CONFIG["sample_rate"], dtype=np.int16)  # 1s of silence
    transcribe(whisper_model, silence)


def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
    audio_f32 = _normalize(audio_np)
    segments = whisper_model.transcribe(
        audio_f32,
        language=CONFIG["language"],
        n_threads=4,
        single_segment=True,
        no_context=True,
        audio_ctx=768,
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text
