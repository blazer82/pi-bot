"""Tests for audio capture functions."""

from unittest import mock

import numpy as np

from pi_bot.config import CONFIG
from pi_bot.audio import listen_for_wake_word, record_until_silence, wait_for_followup


def _mock_read_mic(audio_chunks):
    """Return a side_effect function that yields given audio chunks."""
    chunk_iter = iter(audio_chunks)
    return lambda n: next(chunk_iter)


class TestRecordUntilSilence:
    @mock.patch("pi_bot.audio._read_mic")
    def test_stops_on_silence(self, mock_read):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        skip = 3

        loud = np.full(chunk_size, 5000, dtype=np.int16)
        silent = np.zeros(chunk_size, dtype=np.int16)
        silence_needed = int(CONFIG["silence_duration"] / 0.1)

        audio_chunks = [loud] * (skip + 2) + [silent] * (silence_needed + 1)
        mock_read.side_effect = _mock_read_mic(audio_chunks)

        result = record_until_silence()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    @mock.patch("pi_bot.audio._read_mic")
    def test_respects_max_record_seconds(self, mock_read):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        max_chunks = int(CONFIG["max_record_seconds"] / 0.1)

        loud = np.full(chunk_size, 5000, dtype=np.int16)
        audio_chunks = [loud] * (max_chunks + 5)
        mock_read.side_effect = _mock_read_mic(audio_chunks)

        result = record_until_silence()
        expected_samples = max_chunks * chunk_size
        assert len(result) == expected_samples


class TestWaitForFollowup:
    @mock.patch("pi_bot.audio._read_mic")
    def test_returns_none_on_timeout(self, mock_read):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        timeout_chunks = int(CONFIG["followup_timeout"] / 0.1)

        silent = np.zeros(chunk_size, dtype=np.int16)
        audio_chunks = [silent] * (timeout_chunks + 5)
        mock_read.side_effect = _mock_read_mic(audio_chunks)

        result = wait_for_followup()
        assert result is None

    @mock.patch("pi_bot.audio._read_mic")
    def test_records_when_speech_detected(self, mock_read):
        sr = CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        silence_needed = int(CONFIG["silence_duration"] / 0.1)

        loud = np.full(chunk_size, 5000, dtype=np.int16)
        silent = np.zeros(chunk_size, dtype=np.int16)

        audio_chunks = [loud] + [loud] * 3 + [silent] * (silence_needed + 1)
        mock_read.side_effect = _mock_read_mic(audio_chunks)

        result = wait_for_followup()
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


class TestListenForWakeWord:
    @mock.patch("pi_bot.audio._read_mic")
    def test_returns_on_wake_word(self, mock_read):
        chunk_size = 1280
        audio = np.zeros(chunk_size, dtype=np.int16)
        mock_read.return_value = audio

        from pi_bot.audio import _WAKE_KEY

        wake_model = mock.MagicMock()
        wake_model.predict.side_effect = [
            {_WAKE_KEY: 0.1},
            {_WAKE_KEY: 0.9},
        ]

        listen_for_wake_word(wake_model)
        assert wake_model.predict.call_count == 2
