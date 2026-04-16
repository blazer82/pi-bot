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


def calibrate_noise_floor(duration=0.5):
    """Record ambient noise and adapt the silence threshold.

    Records *duration* seconds of audio, measures the RMS level, and sets
    ``CONFIG["silence_threshold"]`` to ``rms * 1.8`` so the threshold
    tracks the actual environment instead of relying on a hard-coded value.
    """
    sr = CONFIG["sample_rate"]
    samples = int(sr * duration)
    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=samples,
        device=CONFIG["mic_device"],
    ) as stream:
        audio, _ = stream.read(samples)
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    CONFIG["silence_threshold"] = max(rms * 1.8, 100)  # floor of 100 to avoid over-sensitivity
    print(f"Noise floor calibrated: RMS={rms:.0f}, threshold={CONFIG['silence_threshold']:.0f}")


def listen_for_wake_word(wake_model):
    """Block until the wake word is detected."""
    wake_model.reset()
    chunk_size = 1280  # 80ms at 16kHz
    with sd.InputStream(
        samplerate=CONFIG["sample_rate"],
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        while True:
            audio, _ = stream.read(chunk_size)
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

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        for i in range(max_chunks):
            audio, _ = stream.read(chunk_size)
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

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        # Phase 1 — wait for speech onset
        for _ in range(timeout_chunks):
            audio, _ = stream.read(chunk_size)
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms >= CONFIG["silence_threshold"]:
                speech_detected = True
                chunks.append(audio.copy())
                break

        if not speech_detected:
            return None

        # Phase 2 — record until silence
        for _ in range(max_record_chunks):
            audio, _ = stream.read(chunk_size)
            chunks.append(audio.copy())
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < CONFIG["silence_threshold"]:
                silent_count += 1
            else:
                silent_count = 0
            if silent_count >= silence_chunks_needed:
                break

    return np.concatenate(chunks)
