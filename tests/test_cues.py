"""Tests for pi_bot.cues — audio cue playback."""

import os
import struct
import tempfile
import wave
from unittest import mock

import numpy as np

from pi_bot import cues
from pi_bot.config import CONFIG


def _make_wav(path, n_samples=100, sample_rate=22050):
    """Write a minimal mono 16-bit WAV file."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))


class TestPlay:
    def setup_method(self):
        cues._cache.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._orig = CONFIG["sounds_dir"]
        CONFIG["sounds_dir"] = self._tmpdir

    def teardown_method(self):
        CONFIG["sounds_dir"] = self._orig
        cues._cache.clear()

    def test_play_loads_and_creates_stream(self):
        _make_wav(os.path.join(self._tmpdir, "ack.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.play("ack")

        sd.OutputStream.assert_called_once()
        stream = sd.OutputStream.return_value
        stream.start.assert_called_once()
        stream.write.assert_called_once()
        arr = stream.write.call_args[0][0]
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2 and arr.shape[1] == 1
        stream.stop.assert_called_once()
        stream.close.assert_called_once()

    def test_play_missing_file(self):
        sd = cues.sd
        sd.reset_mock()

        cues.play("nonexistent")

        sd.OutputStream.assert_not_called()

    def test_caching(self):
        _make_wav(os.path.join(self._tmpdir, "ack.wav"))

        result1 = cues._load("ack")
        result2 = cues._load("ack")

        assert result1[0] is result2[0]
        assert result1[1] == result2[1]


class TestLoop:
    def setup_method(self):
        cues._cache.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._orig = CONFIG["sounds_dir"]
        CONFIG["sounds_dir"] = self._tmpdir

    def teardown_method(self):
        CONFIG["sounds_dir"] = self._orig
        cues.stop_loop()
        cues._cache.clear()

    def test_start_and_stop(self):
        _make_wav(os.path.join(self._tmpdir, "thinking.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("thinking")
        sd.OutputStream.assert_called_once()
        stream = sd.OutputStream.return_value
        stream.start.assert_called_once()
        assert cues._loop_stream is not None

        cues.stop_loop()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert cues._loop_stream is None

    def test_stop_when_not_running(self):
        cues.stop_loop()

    def test_start_loop_missing_file(self):
        cues.start_loop("nonexistent")
        assert cues._loop_stream is None
