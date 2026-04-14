"""Tests for speech-to-text."""

import os
from unittest import mock

import numpy as np
import pytest

from pi_bot.stt import transcribe


class TestTranscribe:
    def test_transcribes_audio_and_cleans_up(self):
        audio = np.zeros(16000, dtype=np.int16)

        mock_model = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.text = "Hello there"
        mock_model.transcribe.return_value = [mock_segment]

        result = transcribe(mock_model, audio)

        assert result == "Hello there"
        mock_model.transcribe.assert_called_once()
        tmp_path = mock_model.transcribe.call_args[0][0]
        assert not os.path.exists(tmp_path)

    def test_multiple_segments_joined(self):
        audio = np.zeros(16000, dtype=np.int16)
        mock_model = mock.MagicMock()
        seg1 = mock.MagicMock()
        seg1.text = "Hello"
        seg2 = mock.MagicMock()
        seg2.text = "world"
        mock_model.transcribe.return_value = [seg1, seg2]

        result = transcribe(mock_model, audio)
        assert result == "Hello world"

    def test_empty_transcription(self):
        audio = np.zeros(16000, dtype=np.int16)
        mock_model = mock.MagicMock()
        mock_model.transcribe.return_value = []

        result = transcribe(mock_model, audio)
        assert result == ""

    def test_temp_file_cleaned_on_error(self):
        audio = np.zeros(16000, dtype=np.int16)
        mock_model = mock.MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("whisper crash")

        with pytest.raises(RuntimeError):
            transcribe(mock_model, audio)
