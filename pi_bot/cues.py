"""Audio cue playback — short WAV sounds for pipeline state changes.

Uses dedicated sd.OutputStream instances instead of the global sd.play() to
avoid conflicts with TTS playback (which also uses sd.play/sd.wait).
"""

import os
import random
import threading
import time
import wave

import numpy as np
import sounddevice as sd

from pi_bot.config import CONFIG

_cache = {}  # name -> (audio_array, sample_rate)
_loop_stream = None
_loop_name = None
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


def _load_jingles():
    jingles = []
    for i in range(1, 11):
        loaded = _load(f"jingle{i}")
        if loaded is not None and len(loaded[0]) > 0:
            jingles.append(loaded)
    return jingles


def _shuffle_no_repeat(playlist, last_audio):
    random.shuffle(playlist)
    if len(playlist) > 1 and playlist[0][0] is last_audio:
        swap_idx = random.randint(1, len(playlist) - 1)
        playlist[0], playlist[swap_idx] = playlist[swap_idx], playlist[0]


def start_loop(name):
    """Start looping a cue via a callback-based OutputStream.

    When *name* is ``"thinking"``, plays jingle1–jingle10 in shuffled order,
    reshuffling when all have played.  Other names loop a single file.

    Idempotent: if *name* is already looping, returns immediately.
    """
    global _loop_stream, _loop_name
    with _loop_lock:
        if _loop_name == name and _loop_stream is not None:
            return
    stop_loop()

    if name == "thinking":
        jingles = _load_jingles()
        if not jingles:
            return
        rate = jingles[0][1]
        playlist = list(jingles)
        random.shuffle(playlist)
        state = [0, 0]  # [jingle_index, pos_within_jingle]

        def _callback(outdata, frames, time_info, status):
            written = 0
            while written < frames:
                if state[0] >= len(playlist):
                    last_audio = playlist[-1][0]
                    _shuffle_no_repeat(playlist, last_audio)
                    state[0] = 0
                audio = playlist[state[0]][0]
                n = len(audio)
                chunk = min(frames - written, n - state[1])
                outdata[written:written + chunk, 0] = audio[state[1]:state[1] + chunk]
                state[1] += chunk
                written += chunk
                if state[1] >= n:
                    state[0] += 1
                    state[1] = 0
    else:
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
        _loop_name = name
        _loop_stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            device=CONFIG["speaker_device"],
            callback=_callback,
        )
        _loop_stream.start()


def stop_loop():
    """Stop the looping cue if one is running.

    Includes a short delay after closing to let the audio device fully release
    before TTS playback opens it again (avoids garbled output on ALSA).
    """
    global _loop_stream, _loop_name
    with _loop_lock:
        if _loop_stream is None:
            return
        _loop_stream.stop()
        _loop_stream.close()
        _loop_stream = None
        _loop_name = None
    time.sleep(0.05)
