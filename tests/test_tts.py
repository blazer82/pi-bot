"""Tests for text-to-speech (Piper TTS)."""

import io
import os
import struct
import wave
from unittest import mock

import pytest

from pi_bot.config import CONFIG
from pi_bot.tts import speak, _check_piper


def _make_wav(n_frames=100, sample_rate=22050):
    """Create minimal WAV bytes for testing."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return buf.getvalue()


class TestSpeak:
    @mock.patch("pi_bot.tts.sd")
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_calls_piper_with_correct_model(self, mock_run, mock_sd):
        mock_run.return_value = mock.MagicMock(stdout=_make_wav())
        speak("Hallo Welt")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "piper"
        model_idx = cmd.index("--model") + 1
        expected = os.path.join(CONFIG["piper_data_dir"], CONFIG["piper_model"] + ".onnx")
        assert cmd[model_idx] == expected

    @mock.patch("pi_bot.tts.sd")
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_sends_text_as_stdin(self, mock_run, mock_sd):
        mock_run.return_value = mock.MagicMock(stdout=_make_wav())
        speak("Hallo Welt")

        assert mock_run.call_args[1]["input"] == b"Hallo Welt"

    @mock.patch("pi_bot.tts.sd")
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_uses_config_length_scale(self, mock_run, mock_sd):
        mock_run.return_value = mock.MagicMock(stdout=_make_wav())
        original = CONFIG["piper_length_scale"]
        try:
            CONFIG["piper_length_scale"] = 1.3
            speak("test")
            cmd = mock_run.call_args[0][0]
            scale_idx = cmd.index("--length-scale") + 1
            assert cmd[scale_idx] == "1.3"
        finally:
            CONFIG["piper_length_scale"] = original

    @mock.patch("pi_bot.tts.sd")
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_plays_audio_via_sounddevice(self, mock_run, mock_sd):
        mock_run.return_value = mock.MagicMock(stdout=_make_wav(sample_rate=22050))
        speak("test")

        mock_sd.OutputStream.assert_called_once()
        _, kwargs = mock_sd.OutputStream.call_args
        assert kwargs["samplerate"] == 22050
        assert kwargs["device"] == CONFIG["speaker_device"]
        stream = mock_sd.OutputStream.return_value
        stream.start.assert_called_once()
        stream.write.assert_called_once()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    @mock.patch("pi_bot.tts.sd")
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_raises_on_piper_failure(self, mock_run, mock_sd):
        mock_run.side_effect = Exception("piper failed")
        with pytest.raises(Exception):
            speak("test")


class TestCheckPiper:
    @mock.patch("pi_bot.tts.shutil.which", return_value="/usr/local/bin/piper")
    def test_passes_when_piper_found(self, mock_which):
        _check_piper()

    @mock.patch("pi_bot.tts.shutil.which", return_value=None)
    def test_raises_when_piper_missing(self, mock_which):
        with pytest.raises(RuntimeError, match="piper binary not found"):
            _check_piper()
