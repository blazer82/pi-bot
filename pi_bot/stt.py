"""Speech-to-Text via whisper.cpp."""

import numpy as np
import noisereduce as nr
import webrtcvad
from scipy.signal import butter, sosfilt

from pi_bot.config import CONFIG

# Pre-compute high-pass filter coefficients (80 Hz cutoff, 4th order)
_HP_SOS = butter(4, 80, btype="high", fs=CONFIG["sample_rate"], output="sos")

# Target RMS level for AGC (empirically good for Whisper)
_TARGET_RMS = 0.1


def _vad_filter(audio_int16, sr=CONFIG["sample_rate"], aggressiveness=2,
                frame_ms=30, pad_frames=3):
    """Strip non-speech frames using WebRTC VAD.

    Returns only speech segments (with *pad_frames* of context on each side).
    Falls back to the original audio if no speech is detected.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_ms / 1000)

    # Split into fixed-size frames
    num_frames = len(audio_int16) // frame_size
    if num_frames == 0:
        return audio_int16

    is_speech = []
    for i in range(num_frames):
        frame = audio_int16[i * frame_size:(i + 1) * frame_size]
        is_speech.append(vad.is_speech(frame.tobytes(), sr))

    # Pad speech regions so we don't clip word edges
    padded = list(is_speech)
    for i, v in enumerate(is_speech):
        if v:
            start = max(0, i - pad_frames)
            end = min(len(padded), i + pad_frames + 1)
            for j in range(start, end):
                padded[j] = True

    speech_frames = [
        audio_int16[i * frame_size:(i + 1) * frame_size]
        for i in range(num_frames) if padded[i]
    ]
    if not speech_frames:
        return audio_int16  # no speech detected — return as-is
    return np.concatenate(speech_frames)


def _preprocess(audio):
    """Convert int16 audio to cleaned float32 in [-1, 1].

    Pipeline: VAD trim -> float32 -> high-pass filter -> noise reduction -> AGC.
    """
    audio_int16 = audio.flatten()

    # 0. Strip non-speech frames before expensive processing
    audio_int16 = _vad_filter(audio_int16)

    f32 = audio_int16.astype(np.float32) / 32768.0
    peak = np.max(np.abs(f32))
    if peak < 1e-6:
        return np.zeros_like(f32)

    # 1. High-pass filter: remove low-frequency rumble / fan noise
    f32 = sosfilt(_HP_SOS, f32)

    # 2. Spectral noise reduction
    f32 = nr.reduce_noise(y=f32, sr=CONFIG["sample_rate"], stationary=True,
                          prop_decrease=0.8)

    # 3. AGC: normalize RMS to a consistent target level
    rms = np.sqrt(np.mean(f32 ** 2))
    if rms > 1e-6:
        gain = min(_TARGET_RMS / rms, 1.0 / np.max(np.abs(f32)))
        f32 = f32 * gain

    return f32


def warmup(whisper_model):
    """Run a short silent transcription to warm up the model."""
    silence = np.zeros(CONFIG["sample_rate"], dtype=np.int16)  # 1s of silence
    transcribe(whisper_model, silence)


def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
    audio_f32 = _preprocess(audio_np)
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
