"""Text-to-Speech via espeak-ng."""

import subprocess

from pi_bot.config import CONFIG


def speak(text):
    """Speak text via espeak-ng. Blocks until done."""
    cmd = [
        "espeak-ng",
        "-v", CONFIG["language"],
        "-s", str(CONFIG["espeak_speed"]),
        "-p", str(CONFIG["espeak_pitch"]),
        text,
    ]
    subprocess.run(cmd, check=True)
