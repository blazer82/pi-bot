"""Tests for text-to-speech."""

from unittest import mock

from pi_bot.config import CONFIG
from pi_bot.tts import speak


class TestSpeak:
    @mock.patch("pi_bot.tts.subprocess.run")
    def test_calls_espeak_with_correct_args(self, mock_run):
        speak("Hello world")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "espeak-ng"
        assert "Hello world" in cmd
        assert "-v" in cmd
        assert "-s" in cmd
        assert "-p" in cmd

    @mock.patch("pi_bot.tts.subprocess.run")
    def test_uses_config_voice(self, mock_run):
        original = CONFIG["espeak_voice"]
        try:
            CONFIG["espeak_voice"] = "mb-de4"
            speak("test")
            cmd = mock_run.call_args[0][0]
            voice_idx = cmd.index("-v") + 1
            assert cmd[voice_idx] == "mb-de4"
        finally:
            CONFIG["espeak_voice"] = original

    @mock.patch("pi_bot.tts.subprocess.run")
    def test_uses_config_speed_and_pitch(self, mock_run):
        original_speed = CONFIG["espeak_speed"]
        original_pitch = CONFIG["espeak_pitch"]
        try:
            CONFIG["espeak_speed"] = 200
            CONFIG["espeak_pitch"] = 80
            speak("test")
            cmd = mock_run.call_args[0][0]
            speed_idx = cmd.index("-s") + 1
            pitch_idx = cmd.index("-p") + 1
            assert cmd[speed_idx] == "200"
            assert cmd[pitch_idx] == "80"
        finally:
            CONFIG["espeak_speed"] = original_speed
            CONFIG["espeak_pitch"] = original_pitch
