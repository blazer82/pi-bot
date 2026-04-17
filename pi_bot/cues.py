"""Audio cue playback — short WAV sounds for pipeline state changes.

Uses dedicated sd.OutputStream instances instead of the global sd.play() to
avoid conflicts with TTS playback (which also uses sd.play/sd.wait).
"""

import os
import threading
import wave

import numpy as np
import sounddevice as sd

from pi_bot.config import CONFIG

_cache = {}  # name -> (audio_array, sample_rate)
_loop_stream = None
_loop_lock = threading.Lock()


def _load(name):
    if name in _cache:
        return _cache[name]
    path = os.path.join(CONFIG["sounds_dir"], f"{name}.wav")
    if not os.path.isfile(path):
        return None
    with wave.open(path) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        rate = wf.getframerate()
    _cache[name] = (audio, rate)
    return (audio, rate)


def play(name, block=True):
    """Play a one-shot cue via a dedicated OutputStream.

    Does nothing if the file is missing.
    """
    loaded = _load(name)
    if loaded is None:
        return
    audio, rate = loaded
    stream = sd.OutputStream(
        samplerate=rate,
        channels=1,
        dtype="int16",
        device=CONFIG["speaker_device"],
    )
    stream.start()
    stream.write(audio.reshape(-1, 1))
    stream.stop()
    stream.close()
    if not block:
        pass  # write() already blocks until data is queued; kept for API compat


def start_loop(name):
    """Start looping a cue via a callback-based OutputStream."""
    global _loop_stream
    stop_loop()
    loaded = _load(name)
    if loaded is None:
        return
    audio, rate = loaded
    n = len(audio)
    pos = [0]

    def _callback(outdata, frames, time_info, status):
        written = 0
        while written < frames:
            chunk = min(frames - written, n - pos[0])
            outdata[written:written + chunk, 0] = audio[pos[0]:pos[0] + chunk]
            pos[0] = (pos[0] + chunk) % n
            written += chunk

    with _loop_lock:
        _loop_stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            device=CONFIG["speaker_device"],
            callback=_callback,
        )
        _loop_stream.start()


def stop_loop():
    """Stop the looping cue if one is running."""
    global _loop_stream
    with _loop_lock:
        if _loop_stream is None:
            return
        _loop_stream.stop()
        _loop_stream.close()
        _loop_stream = None
