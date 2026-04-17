"""Text-to-Speech via Piper TTS."""

import io
import shutil
import subprocess
import wave

import sounddevice as sd
import numpy as np

from pi_bot.config import CONFIG


def _check_piper():
    """Verify piper binary is available on PATH."""
    if shutil.which("piper") is None:
        raise RuntimeError(
            "piper binary not found on PATH. Run setup.sh to install it."
        )


def speak(text):
    """Speak text via Piper TTS. Blocks until playback finishes."""
    cmd = [
        "piper",
        "--model", CONFIG["piper_model"],
        "--data-dir", CONFIG["piper_data_dir"],
        "--length-scale", str(CONFIG["piper_length_scale"]),
        "--output_file", "-",
    ]
    if CONFIG.get("piper_speaker") is not None:
        cmd += ["--speaker", str(CONFIG["piper_speaker"])]

    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        capture_output=True,
        check=True,
    )

    with wave.open(io.BytesIO(proc.stdout)) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sd.play(audio, samplerate=wf.getframerate())
        sd.wait()
