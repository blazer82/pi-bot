"""Tests for pi_bot.cues — audio cue playback."""

import os
import struct
import tempfile
import wave
from unittest import mock

import numpy as np

from pi_bot import cues
from pi_bot.config import CONFIG


def _make_wav(path, n_samples=100, sample_rate=22050, value=0):
    """Write a minimal mono 16-bit WAV file."""
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([value] * n_samples)))


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
        _make_wav(os.path.join(self._tmpdir, "beep.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("beep")
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

    def test_start_loop_idempotent_same_name(self):
        _make_wav(os.path.join(self._tmpdir, "beep.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("beep")
        assert sd.OutputStream.call_count == 1

        cues.start_loop("beep")
        assert sd.OutputStream.call_count == 1
        assert cues._loop_stream is not None

    def test_start_loop_different_name_restarts(self):
        _make_wav(os.path.join(self._tmpdir, "beep.wav"))
        _make_wav(os.path.join(self._tmpdir, "boop.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("beep")
        assert sd.OutputStream.call_count == 1

        cues.start_loop("boop")
        assert sd.OutputStream.call_count == 2

    def test_stop_loop_clears_name(self):
        _make_wav(os.path.join(self._tmpdir, "beep.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("beep")
        cues.stop_loop()
        cues.start_loop("beep")
        assert sd.OutputStream.call_count == 2

    def test_start_loop_missing_file(self):
        cues.start_loop("nonexistent")
        assert cues._loop_stream is None


class TestJingleRotation:
    def setup_method(self):
        cues._cache.clear()
        self._tmpdir = tempfile.mkdtemp()
        self._orig = CONFIG["sounds_dir"]
        CONFIG["sounds_dir"] = self._tmpdir

    def teardown_method(self):
        CONFIG["sounds_dir"] = self._orig
        cues.stop_loop()
        cues._cache.clear()

    def _create_jingles(self, n_samples=100):
        for i in range(1, 11):
            _make_wav(
                os.path.join(self._tmpdir, f"jingle{i}.wav"),
                n_samples=n_samples,
                value=i,
            )

    def test_start_loop_thinking_loads_jingles(self):
        self._create_jingles()
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("thinking")

        sd.OutputStream.assert_called_once()
        sd.OutputStream.return_value.start.assert_called_once()
        assert cues._loop_stream is not None
        for i in range(1, 11):
            assert f"jingle{i}" in cues._cache

    def test_jingle_callback_plays_sequentially(self):
        n_samples = 50
        self._create_jingles(n_samples=n_samples)
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("thinking")

        callback = sd.OutputStream.call_args[1]["callback"]
        seen_values = []
        for _ in range(10):
            outdata = np.zeros((n_samples, 1), dtype=np.int16)
            callback(outdata, n_samples, None, None)
            seen_values.append(outdata[0, 0])

        assert len(set(seen_values)) == 10, "All 10 jingles should play before repeating"

    def test_shuffle_no_repeat(self):
        a = np.array([1], dtype=np.int16)
        b = np.array([2], dtype=np.int16)
        c = np.array([3], dtype=np.int16)
        playlist = [(a, 22050), (b, 22050), (c, 22050)]

        for _ in range(50):
            cues._shuffle_no_repeat(playlist, a)
            assert playlist[0][0] is not a

    def test_start_loop_non_thinking_uses_single_loop(self):
        _make_wav(os.path.join(self._tmpdir, "buffering.wav"))
        sd = cues.sd
        sd.reset_mock()

        cues.start_loop("buffering")

        sd.OutputStream.assert_called_once()
        sd.OutputStream.return_value.start.assert_called_once()
        assert cues._loop_stream is not None

    def test_start_loop_thinking_missing_jingles(self):
        cues.start_loop("thinking")
        assert cues._loop_stream is None
