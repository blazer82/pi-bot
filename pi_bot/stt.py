"""Speech-to-Text via whisper.cpp."""

import os
import tempfile
import wave

from pi_bot.config import CONFIG


def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
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
