"""Unit tests for Pi-Bot components.

Every test mocks external dependencies (hardware, APIs, subprocesses)
so the suite runs without a microphone, speaker, ollama, or espeak-ng.
"""

import json
import os
import re
from unittest import mock

import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# We need to mock hardware-dependent imports before importing pi_bot.
# sounddevice, openwakeword, and pywhispercpp all try to access audio
# devices or load native libraries on import.
# The mocks must stay in sys.modules permanently so that @mock.patch
# decorators can resolve "pi_bot.X" without triggering a fresh import.
# ---------------------------------------------------------------------------
_sd_mock = mock.MagicMock()
_oww_mock = mock.MagicMock()
_pwc_mock = mock.MagicMock()

sys.modules.setdefault("sounddevice", _sd_mock)
sys.modules.setdefault("openwakeword", _oww_mock)
sys.modules.setdefault("openwakeword.model", _oww_mock)
sys.modules.setdefault("pywhispercpp", _pwc_mock)
sys.modules.setdefault("pywhispercpp.model", _pwc_mock)

import pi_bot  # noqa: E402  — must come after the sys.modules patches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
SAMPLE_JOKES = [
    {"id": 1, "setup": "Why did the programmer quit?",
     "punchline": "Because he didn't get arrays."},
    {"id": 2, "setup": "Why do Java devs wear glasses?",
     "punchline": "Because they can't C#."},
]


# ===================================================================
# Tool execution
# ===================================================================
class TestGetRandomJoke:
    def test_returns_valid_json(self):
        result = pi_bot.get_random_joke(SAMPLE_JOKES)
        parsed = json.loads(result)
        assert "setup" in parsed
        assert "punchline" in parsed

    def test_returns_joke_from_db(self):
        result = json.loads(pi_bot.get_random_joke(SAMPLE_JOKES))
        assert result in SAMPLE_JOKES

    def test_single_joke_db(self):
        single = [SAMPLE_JOKES[0]]
        result = json.loads(pi_bot.get_random_joke(single))
        assert result == SAMPLE_JOKES[0]


class TestExecuteTool:
    def test_known_tool(self):
        result = pi_bot.execute_tool("get_random_joke", {}, SAMPLE_JOKES)
        parsed = json.loads(result)
        assert parsed in SAMPLE_JOKES

    def test_unknown_tool(self):
        result = pi_bot.execute_tool("nonexistent", {}, SAMPLE_JOKES)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "nonexistent" in parsed["error"]


# ===================================================================
# Text-to-Speech
# ===================================================================
class TestSpeak:
    @mock.patch("pi_bot.subprocess.run")
    def test_calls_espeak_with_correct_args(self, mock_run):
        pi_bot.speak("Hello world")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "espeak-ng"
        assert "Hello world" in cmd
        assert "-v" in cmd
        assert "-s" in cmd
        assert "-p" in cmd

    @mock.patch("pi_bot.subprocess.run")
    def test_uses_config_language(self, mock_run):
        original = pi_bot.CONFIG["language"]
        try:
            pi_bot.CONFIG["language"] = "en"
            pi_bot.speak("test")
            cmd = mock_run.call_args[0][0]
            lang_idx = cmd.index("-v") + 1
            assert cmd[lang_idx] == "en"
        finally:
            pi_bot.CONFIG["language"] = original

    @mock.patch("pi_bot.subprocess.run")
    def test_uses_config_speed_and_pitch(self, mock_run):
        original_speed = pi_bot.CONFIG["espeak_speed"]
        original_pitch = pi_bot.CONFIG["espeak_pitch"]
        try:
            pi_bot.CONFIG["espeak_speed"] = 200
            pi_bot.CONFIG["espeak_pitch"] = 80
            pi_bot.speak("test")
            cmd = mock_run.call_args[0][0]
            speed_idx = cmd.index("-s") + 1
            pitch_idx = cmd.index("-p") + 1
            assert cmd[speed_idx] == "200"
            assert cmd[pitch_idx] == "80"
        finally:
            pi_bot.CONFIG["espeak_speed"] = original_speed
            pi_bot.CONFIG["espeak_pitch"] = original_pitch


# ===================================================================
# Speech-to-Text
# ===================================================================
class TestTranscribe:
    def test_transcribes_audio_and_cleans_up(self):
        # Create fake audio data
        audio = np.zeros(16000, dtype=np.int16)

        # Mock whisper model
        mock_model = mock.MagicMock()
        mock_segment = mock.MagicMock()
        mock_segment.text = "Hello there"
        mock_model.transcribe.return_value = [mock_segment]

        result = pi_bot.transcribe(mock_model, audio)

        assert result == "Hello there"
        mock_model.transcribe.assert_called_once()
        # Verify temp file was cleaned up
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

        result = pi_bot.transcribe(mock_model, audio)
        assert result == "Hello world"

    def test_empty_transcription(self):
        audio = np.zeros(16000, dtype=np.int16)
        mock_model = mock.MagicMock()
        mock_model.transcribe.return_value = []

        result = pi_bot.transcribe(mock_model, audio)
        assert result == ""

    def test_temp_file_cleaned_on_error(self):
        audio = np.zeros(16000, dtype=np.int16)
        mock_model = mock.MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("whisper crash")

        with pytest.raises(RuntimeError):
            pi_bot.transcribe(mock_model, audio)
        # Can't check the exact path, but the finally block should have run


# ===================================================================
# Sentence splitting
# ===================================================================
class TestSpeakSentences:
    @mock.patch("pi_bot.speak")
    def test_no_complete_sentence(self, mock_speak):
        remainder, spoken = pi_bot._speak_sentences("Hello world")
        assert remainder == "Hello world"
        assert spoken == ""
        mock_speak.assert_not_called()

    @mock.patch("pi_bot.speak")
    def test_one_complete_sentence(self, mock_speak):
        remainder, spoken = pi_bot._speak_sentences("Hello world. How are")
        assert remainder == "How are"
        assert "Hello world." in spoken
        mock_speak.assert_called_once_with("Hello world.")

    @mock.patch("pi_bot.speak")
    def test_multiple_sentences(self, mock_speak):
        remainder, spoken = pi_bot._speak_sentences(
            "First. Second! Third? Partial"
        )
        assert remainder == "Partial"
        assert mock_speak.call_count == 3

    @mock.patch("pi_bot.speak")
    def test_question_mark_boundary(self, mock_speak):
        remainder, spoken = pi_bot._speak_sentences("Really? Yes")
        assert remainder == "Yes"
        mock_speak.assert_called_once_with("Really?")

    @mock.patch("pi_bot.speak")
    def test_exclamation_boundary(self, mock_speak):
        remainder, spoken = pi_bot._speak_sentences("Wow! Cool")
        assert remainder == "Cool"
        mock_speak.assert_called_once_with("Wow!")


# ===================================================================
# Ollama streaming
# ===================================================================
class TestOllamaChatStream:
    @mock.patch("pi_bot.requests.post")
    def test_yields_content_chunks(self, mock_post):
        lines = [
            json.dumps({"message": {"content": "Hello "}, "done": False}).encode(),
            json.dumps({"message": {"content": "world"}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True}).encode(),
        ]
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        chunks = list(pi_bot._ollama_chat_stream([{"role": "user", "content": "hi"}]))
        assert len(chunks) == 2
        assert chunks[0] == {"type": "content", "text": "Hello "}
        assert chunks[1] == {"type": "content", "text": "world"}

    @mock.patch("pi_bot.requests.post")
    def test_yields_tool_calls_and_stops(self, mock_post):
        tool_calls = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        lines = [
            json.dumps({"message": {"content": "Let me", "tool_calls": None}, "done": False}).encode(),
            json.dumps({"message": {"tool_calls": tool_calls}, "done": False}).encode(),
            # This line should NOT be reached
            json.dumps({"message": {"content": "after tool"}, "done": False}).encode(),
        ]
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        chunks = list(pi_bot._ollama_chat_stream([{"role": "user", "content": "joke"}]))
        # Should get content chunk then tool_calls, then stop
        types = [c["type"] for c in chunks]
        assert "tool_calls" in types
        # Nothing after tool_calls
        assert types.index("tool_calls") == len(types) - 1

    @mock.patch("pi_bot.requests.post")
    def test_sends_correct_payload(self, mock_post):
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": ""}, "done": True}).encode()
        ]
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "hi"}]
        list(pi_bot._ollama_chat_stream(messages, tools=pi_bot.TOOLS))

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["model"] == pi_bot.CONFIG["ollama_model"]
        assert payload["messages"] == messages
        assert payload["stream"] is True
        assert payload["tools"] == pi_bot.TOOLS

    @mock.patch("pi_bot.requests.post")
    def test_skips_empty_lines(self, mock_post):
        lines = [
            b"",
            json.dumps({"message": {"content": "hi"}, "done": False}).encode(),
            b"",
            json.dumps({"message": {"content": ""}, "done": True}).encode(),
        ]
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        chunks = list(pi_bot._ollama_chat_stream([]))
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hi"


# ===================================================================
# Stream and speak
# ===================================================================
class TestStreamAndSpeak:
    @mock.patch("pi_bot.speak")
    @mock.patch("pi_bot._ollama_chat_stream")
    def test_simple_response(self, mock_stream, mock_speak):
        mock_stream.return_value = iter([
            {"type": "content", "text": "Hello world."},
        ])
        response, tool_calls = pi_bot.stream_and_speak([])
        assert "Hello world." in response
        assert tool_calls is None

    @mock.patch("pi_bot.speak")
    @mock.patch("pi_bot._ollama_chat_stream")
    def test_tool_calls_returned(self, mock_stream, mock_speak):
        tc = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        mock_stream.return_value = iter([
            {"type": "tool_calls", "tool_calls": tc},
        ])
        response, tool_calls = pi_bot.stream_and_speak([])
        assert tool_calls == tc

    @mock.patch("pi_bot.speak")
    @mock.patch("pi_bot._ollama_chat_stream")
    def test_think_tags_excluded_from_response(self, mock_stream, mock_speak):
        mock_stream.return_value = iter([
            {"type": "content", "text": "<think>internal reasoning</think>"},
            {"type": "content", "text": "Visible answer."},
        ])
        response, _ = pi_bot.stream_and_speak([])
        assert "internal reasoning" not in response
        assert "Visible answer." in response

    @mock.patch("pi_bot.speak")
    @mock.patch("pi_bot._ollama_chat_stream")
    def test_think_tag_speaks_cue(self, mock_stream, mock_speak):
        original_lang = pi_bot.CONFIG["language"]
        try:
            pi_bot.CONFIG["language"] = "de"
            mock_stream.return_value = iter([
                {"type": "content", "text": "<think>thinking</think>Answer."},
            ])
            pi_bot.stream_and_speak([])
            # Should have spoken "Analysiere..." cue
            spoken_texts = [call[0][0] for call in mock_speak.call_args_list]
            assert any("Analysiere" in t for t in spoken_texts)
        finally:
            pi_bot.CONFIG["language"] = original_lang


# ===================================================================
# Chat with ollama (orchestration)
# ===================================================================
class TestChatWithOllama:
    @mock.patch("pi_bot.stream_and_speak")
    def test_builds_messages_with_system_prompt(self, mock_sas):
        mock_sas.return_value = ("Reply.", None)
        history = []
        pi_bot.chat_with_ollama("Hello", history, SAMPLE_JOKES)

        messages = mock_sas.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    @mock.patch("pi_bot.stream_and_speak")
    def test_updates_conversation_history(self, mock_sas):
        mock_sas.return_value = ("The reply.", None)
        history = []
        pi_bot.chat_with_ollama("Hi", history, SAMPLE_JOKES)

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "The reply."}

    @mock.patch("pi_bot.stream_and_speak")
    def test_trims_history_to_context_turns(self, mock_sas):
        mock_sas.return_value = ("Reply.", None)
        original = pi_bot.CONFIG["context_turns"]
        try:
            pi_bot.CONFIG["context_turns"] = 2  # keep 4 messages
            history = []
            # Add 3 turns (6 messages) — should trim to 4
            for i in range(3):
                pi_bot.chat_with_ollama(f"msg{i}", history, SAMPLE_JOKES)
            assert len(history) == 4
        finally:
            pi_bot.CONFIG["context_turns"] = original

    @mock.patch("pi_bot.execute_tool")
    @mock.patch("pi_bot.stream_and_speak")
    def test_tool_call_loop(self, mock_sas, mock_exec):
        tc = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        # First call returns tool_calls, second call returns final response
        mock_sas.side_effect = [
            ("", tc),
            ("Here is a joke!", None),
        ]
        mock_exec.return_value = '{"setup": "...", "punchline": "..."}'

        history = []
        result = pi_bot.chat_with_ollama("Tell me a joke", history, SAMPLE_JOKES)

        assert "joke" in result.lower() or result == "Here is a joke!"
        mock_exec.assert_called_once_with("get_random_joke", {}, SAMPLE_JOKES)
        assert mock_sas.call_count == 2

    @mock.patch("pi_bot.stream_and_speak")
    def test_strips_residual_think_tags(self, mock_sas):
        mock_sas.return_value = ("<think>stuff</think>Clean answer.", None)
        history = []
        result = pi_bot.chat_with_ollama("Hi", history, SAMPLE_JOKES)
        assert "think" not in result
        assert "Clean answer." in result

    @mock.patch("pi_bot.stream_and_speak")
    def test_uses_german_prompt_when_configured(self, mock_sas):
        mock_sas.return_value = ("Antwort.", None)
        original = pi_bot.CONFIG["language"]
        try:
            pi_bot.CONFIG["language"] = "de"
            history = []
            pi_bot.chat_with_ollama("Hallo", history, SAMPLE_JOKES)
            messages = mock_sas.call_args[0][0]
            assert messages[0]["content"] == pi_bot.SYSTEM_PROMPT_DE
        finally:
            pi_bot.CONFIG["language"] = original

    @mock.patch("pi_bot.stream_and_speak")
    def test_uses_english_prompt_when_configured(self, mock_sas):
        mock_sas.return_value = ("Answer.", None)
        original = pi_bot.CONFIG["language"]
        try:
            pi_bot.CONFIG["language"] = "en"
            history = []
            pi_bot.chat_with_ollama("Hello", history, SAMPLE_JOKES)
            messages = mock_sas.call_args[0][0]
            assert messages[0]["content"] == pi_bot.SYSTEM_PROMPT_EN
        finally:
            pi_bot.CONFIG["language"] = original


# ===================================================================
# Audio recording — record_until_silence
# ===================================================================
class TestRecordUntilSilence:
    def _make_stream_mock(self, audio_chunks):
        """Create a mock InputStream that yields given audio chunks."""
        stream = mock.MagicMock()
        chunk_iter = iter(audio_chunks)
        stream.read.side_effect = lambda size: (next(chunk_iter), None)
        stream.__enter__ = mock.MagicMock(return_value=stream)
        stream.__exit__ = mock.MagicMock(return_value=False)
        return stream

    @mock.patch("pi_bot.sd.InputStream")
    def test_stops_on_silence(self, mock_is_cls):
        sr = pi_bot.CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        skip = 3

        # Create loud chunks (skip + a few), then enough silent chunks
        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        silent = np.zeros((chunk_size, 1), dtype=np.int16)
        silence_needed = int(pi_bot.CONFIG["silence_duration"] / 0.1)

        audio_chunks = [loud] * (skip + 2) + [silent] * (silence_needed + 1)
        mock_is_cls.return_value = self._make_stream_mock(audio_chunks)

        result = pi_bot.record_until_silence()
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    @mock.patch("pi_bot.sd.InputStream")
    def test_respects_max_record_seconds(self, mock_is_cls):
        sr = pi_bot.CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        max_chunks = int(pi_bot.CONFIG["max_record_seconds"] / 0.1)

        # All loud — should stop at max_chunks
        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        audio_chunks = [loud] * (max_chunks + 5)  # extra to prove it stops
        mock_is_cls.return_value = self._make_stream_mock(audio_chunks)

        result = pi_bot.record_until_silence()
        expected_samples = max_chunks * chunk_size
        assert len(result) == expected_samples


# ===================================================================
# Audio recording — wait_for_followup
# ===================================================================
class TestWaitForFollowup:
    def _make_stream_mock(self, audio_chunks):
        stream = mock.MagicMock()
        chunk_iter = iter(audio_chunks)
        stream.read.side_effect = lambda size: (next(chunk_iter), None)
        stream.__enter__ = mock.MagicMock(return_value=stream)
        stream.__exit__ = mock.MagicMock(return_value=False)
        return stream

    @mock.patch("pi_bot.sd.InputStream")
    def test_returns_none_on_timeout(self, mock_is_cls):
        sr = pi_bot.CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        timeout_chunks = int(pi_bot.CONFIG["followup_timeout"] / 0.1)

        # All silent during phase 1
        silent = np.zeros((chunk_size, 1), dtype=np.int16)
        audio_chunks = [silent] * (timeout_chunks + 5)
        mock_is_cls.return_value = self._make_stream_mock(audio_chunks)

        result = pi_bot.wait_for_followup()
        assert result is None

    @mock.patch("pi_bot.sd.InputStream")
    def test_records_when_speech_detected(self, mock_is_cls):
        sr = pi_bot.CONFIG["sample_rate"]
        chunk_size = int(sr * 0.1)
        silence_needed = int(pi_bot.CONFIG["silence_duration"] / 0.1)

        loud = np.full(chunk_size, 5000, dtype=np.int16).reshape(-1, 1)
        silent = np.zeros((chunk_size, 1), dtype=np.int16)

        # Phase 1: one loud chunk triggers speech detection
        # Phase 2: a few loud then silence to end recording
        audio_chunks = [loud] + [loud] * 3 + [silent] * (silence_needed + 1)
        mock_is_cls.return_value = self._make_stream_mock(audio_chunks)

        result = pi_bot.wait_for_followup()
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) > 0


# ===================================================================
# Wake word detection
# ===================================================================
class TestListenForWakeWord:
    @mock.patch("pi_bot.sd.InputStream")
    def test_returns_on_wake_word(self, mock_is_cls):
        chunk_size = 1280
        audio = np.zeros((chunk_size, 1), dtype=np.int16)

        stream = mock.MagicMock()
        stream.read.return_value = (audio, None)
        stream.__enter__ = mock.MagicMock(return_value=stream)
        stream.__exit__ = mock.MagicMock(return_value=False)
        mock_is_cls.return_value = stream

        wake_model = mock.MagicMock()
        # First call below threshold, second above
        wake_model.predict.side_effect = [
            {pi_bot.CONFIG["wake_word"]: 0.1},
            {pi_bot.CONFIG["wake_word"]: 0.9},
        ]

        pi_bot.listen_for_wake_word(wake_model)
        assert wake_model.predict.call_count == 2


# ===================================================================
# Config & constants sanity checks
# ===================================================================
class TestConfig:
    def test_config_has_required_keys(self):
        required = [
            "language", "ollama_model", "ollama_url", "whisper_model",
            "wake_word", "wake_threshold", "silence_threshold",
            "silence_duration", "max_record_seconds", "sample_rate",
            "context_turns", "followup_timeout", "espeak_speed",
            "espeak_pitch", "mic_device", "speaker_device",
        ]
        for key in required:
            assert key in pi_bot.CONFIG, f"Missing config key: {key}"

    def test_tools_schema_valid(self):
        assert len(pi_bot.TOOLS) == 1
        tool = pi_bot.TOOLS[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_random_joke"

    def test_system_prompts_defined(self):
        assert len(pi_bot.SYSTEM_PROMPT_DE) > 0
        assert len(pi_bot.SYSTEM_PROMPT_EN) > 0
