"""Speech-to-Text via whisper.cpp."""

import os
import wave

import numpy as np
import webrtcvad

from pi_bot.config import CONFIG

# Target RMS level for AGC (empirically good for Whisper)
_TARGET_RMS = 0.2


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

    Pipeline: VAD trim -> float32 -> AGC.
    """
    audio_int16 = audio.flatten()

    # 0. Strip non-speech frames before expensive processing
    audio_int16 = _vad_filter(audio_int16)

    f32 = audio_int16.astype(np.float32) / 32768.0
    peak = np.max(np.abs(f32))
    if peak < 1e-6:
        return np.zeros_like(f32)

    # AGC: normalize RMS to a consistent target level
    rms = np.sqrt(np.mean(f32 ** 2))
    if rms > 1e-6:
        gain = _TARGET_RMS / rms
        f32 = np.clip(f32 * gain, -1.0, 1.0)

    return f32


def warmup(whisper_model):
    """Run a short silent transcription to warm up the model."""
    silence = np.zeros(CONFIG["sample_rate"], dtype=np.int16)  # 1s of silence
    transcribe(whisper_model, silence)


def _save_debug_recording(audio_int16):
    path = os.path.join(CONFIG["debug_recording_dir"], "last_recording.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(CONFIG["sample_rate"])
        wf.writeframes(audio_int16.tobytes())


def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
    if CONFIG["debug_recording_dir"]:
        _save_debug_recording(audio_np.flatten())
    audio_f32 = _preprocess(audio_np)
    # Limit audio context to actual speech length (+10% margin) so whisper
    # doesn't attend over 30s of padded silence.  100 mel frames ≈ 1 second.
    mel_frames = int(len(audio_f32) / CONFIG["sample_rate"] * 100 * 1.1)
    mel_frames = max(mel_frames, 64)
    segments = whisper_model.transcribe(
        audio_f32,
        language="de",
        n_threads=4,
        no_context=True,
        audio_ctx=mel_frames,
    )
    text = " ".join(seg.text for seg in segments).strip()
    return text
