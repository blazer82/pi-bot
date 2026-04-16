"""Speech-to-Text via whisper.cpp."""

import os
import tempfile
import wave

import numpy as np

from pi_bot.config import CONFIG


def _normalize(audio):
    """Peak-normalize int16 audio to use the full dynamic range."""
    peak = np.max(np.abs(audio.astype(np.float32)))
    if peak < 1:
        return audio
    gain = 32767.0 / peak
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def warmup(whisper_model):
    """Run a short silent transcription to warm up the model."""
    silence = np.zeros(CONFIG["sample_rate"], dtype=np.int16)  # 1s of silence
    transcribe(whisper_model, silence)


def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
    audio_np = _normalize(audio_np)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(CONFIG["sample_rate"])
            wf.writeframes(audio_np.tobytes())

    try:
        segments = whisper_model.transcribe(
            tmp_path, language=CONFIG["language"])
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return text
