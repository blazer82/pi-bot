"""Audio capture: wake word detection and recording."""

import os

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from pi_bot.config import CONFIG

# openWakeWord uses the filename stem (without .onnx) as the prediction key
_WAKE_KEY = os.path.splitext(os.path.basename(CONFIG["wake_model"]))[0]

_mic_stream = None


def _read_mic(n_samples):
    """Read n_samples from the persistent mic stream, with channel selection."""
    audio, _ = _mic_stream.read(n_samples)
    if CONFIG["mic_channels"] > 1:
        audio = audio[:, CONFIG["mic_channel_select"]]
    return audio


def open_mic():
    """Open the persistent mic stream.

    Discards 2 s of samples because the mic DSP needs time to
    produce valid audio on a freshly opened stream.
    """
    global _mic_stream
    _mic_stream = sd.InputStream(
        samplerate=CONFIG["sample_rate"],
        channels=CONFIG["mic_channels"],
        dtype="int16",
        device=CONFIG["mic_device"],
    )
    _mic_stream.start()
    # The XVF3800 DSP needs time to produce valid samples on a new stream
    _read_mic(CONFIG["sample_rate"] * 2)


def close_mic():
    global _mic_stream
    if _mic_stream:
        _mic_stream.stop()
        _mic_stream.close()
        _mic_stream = None


def listen_for_wake_word(wake_model):
    """Block until the wake word is detected."""
    wake_model.reset()
    chunk_size = 1280  # 80ms at 16kHz
    while True:
        audio = _read_mic(chunk_size)
        prediction = wake_model.predict(audio.flatten())
        if prediction.get(_WAKE_KEY, 0) > CONFIG["wake_threshold"]:
            return


def record_until_silence():
    """Record audio until silence is detected. Returns int16 numpy array."""
    sr = CONFIG["sample_rate"]
    chunk_size = int(sr * 0.1)  # 100ms chunks
    silence_chunks = int(CONFIG["silence_duration"] / 0.1)
    max_chunks = int(CONFIG["max_record_seconds"] / 0.1)
    skip_chunks = 3  # ignore first 300ms for silence detection

    chunks = []
    silent_count = 0

    for i in range(max_chunks):
        audio = _read_mic(chunk_size)
        chunks.append(audio.copy())

        if i >= skip_chunks:
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < CONFIG["silence_threshold"]:
                silent_count += 1
            else:
                silent_count = 0
            if silent_count >= silence_chunks:
                break

    return np.concatenate(chunks)


def wait_for_followup():
    """Listen for follow-up speech within the timeout window.

    Phase 1: wait up to ``followup_timeout`` seconds for speech onset.
    Phase 2: once speech is detected, record until silence (same logic as
    ``record_until_silence``).

    Returns an int16 numpy array if speech was captured, or *None* if the
    timeout expired without any speech.
    """
    sr = CONFIG["sample_rate"]
    chunk_size = int(sr * 0.1)  # 100 ms chunks
    timeout_chunks = int(CONFIG["followup_timeout"] / 0.1)
    silence_chunks_needed = int(CONFIG["silence_duration"] / 0.1)
    max_record_chunks = int(CONFIG["max_record_seconds"] / 0.1)

    chunks = []
    speech_detected = False
    silent_count = 0

    # Phase 1 — wait for speech onset
    for _ in range(timeout_chunks):
        audio = _read_mic(chunk_size)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms >= CONFIG["silence_threshold"]:
            speech_detected = True
            chunks.append(audio.copy())
            break

    if not speech_detected:
        return None

    # Phase 2 — record until silence
    for _ in range(max_record_chunks):
        audio = _read_mic(chunk_size)
        chunks.append(audio.copy())
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < CONFIG["silence_threshold"]:
            silent_count += 1
        else:
            silent_count = 0
        if silent_count >= silence_chunks_needed:
            break

    return np.concatenate(chunks)
