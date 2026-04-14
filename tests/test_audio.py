"""Tests for audio capture functions."""

from unittest import mock

import numpy as np

from pi_bot.config import CONFIG
from pi_bot.audio import listen_for_wake_word, record_until_silence, wait_for_followup


def _make_stream_mock(audio_chunks):
    """Create a mock InputStream that yields given audio chunks."""
    stream = mock.MagicMock()
    chunk_iter = iter(audio_chunks)
    stream.read.side_effect = lambda size: (next(chunk_iter), None)
    stream.__enter__ = mock.MagicMock(return_value=stream)
    stream.__exit__ = mock.MagicMock(return_value=False)
    return stream


class TestRecordUntilSilence:
    @mock.patch("pi_bot.audio.sd.InputStream")
    def test_stops_on_silence(self, mock_is_cls):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        skip = 3

        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        silent = np.zeros((chunk_size, 1), dtype=np.int16)
        silence_needed = int(CONFIG["silence_duration"] / 0.1)

        audio_chunks = [loud] * (skip + 2) + [silent] * (silence_needed + 1)
        mock_is_cls.return_value = _make_stream_mock(audio_chunks)

        result = record_until_silence()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    @mock.patch("pi_bot.audio.sd.InputStream")
    def test_respects_max_record_seconds(self, mock_is_cls):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        max_chunks = int(CONFIG["max_record_seconds"] / 0.1)

        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        audio_chunks = [loud] * (max_chunks + 5)
        mock_is_cls.return_value = _make_stream_mock(audio_chunks)

        result = record_until_silence()
        expected_samples = max_chunks * chunk_size
        assert len(result) == expected_samples


class TestWaitForFollowup:
    @mock.patch("pi_bot.audio.sd.InputStream")
    def test_returns_none_on_timeout(self, mock_is_cls):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        timeout_chunks = int(CONFIG["followup_timeout"] / 0.1)

        silent = np.zeros((chunk_size, 1), dtype=np.int16)
        audio_chunks = [silent] * (timeout_chunks + 5)
        mock_is_cls.return_value = _make_stream_mock(audio_chunks)

        result = wait_for_followup()
        assert result is None

    @mock.patch("pi_bot.audio.sd.InputStream")
    def test_records_when_speech_detected(self, mock_is_cls):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        silence_needed = int(CONFIG["silence_duration"] / 0.1)

        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        silent = np.zeros((chunk_size, 1), dtype=np.int16)

        audio_chunks = [loud] + [loud] * 3 + [silent] * (silence_needed + 1)
        mock_is_cls.return_value = _make_stream_mock(audio_chunks)

        result = wait_for_followup()
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestListenForWakeWord:
    @mock.patch("pi_bot.audio.sd.InputStream")
    def test_returns_on_wake_word(self, mock_is_cls):
        chunk_size = 1280
        audio = np.zeros((chunk_size, 1), dtype=np.int16)

        stream = mock.MagicMock()
        stream.read.return_value = (audio, None)
        stream.__enter__ = mock.MagicMock(return_value=stream)
        stream.__exit__ = mock.MagicMock(return_value=False)
        mock_is_cls.return_value = stream

        from pi_bot.audio import _WAKE_KEY

        wake_model = mock.MagicMock()
        wake_model.predict.side_effect = [
            {_WAKE_KEY: 0.1},
            {_WAKE_KEY: 0.9},
        ]

        listen_for_wake_word(wake_model)
        assert wake_model.predict.call_count == 2
