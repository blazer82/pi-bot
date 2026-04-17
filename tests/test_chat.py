"""Tests for chat orchestration, streaming, and sentence splitting."""

import json
from unittest import mock

from tests.conftest import SAMPLE_JOKES
from pi_bot.config import CONFIG, TOOLS, SYSTEM_PROMPT_DE
from pi_bot.chat import (
    _ollama_chat_stream,
    _split_sentences,
    stream_and_speak,
    chat_with_ollama,
)


class TestSplitSentences:
    def test_no_complete_sentence(self):
        remainder, parts = _split_sentences("Hello world")
        assert remainder == "Hello world"
        assert parts == []

    def test_one_complete_sentence(self):
        remainder, parts = _split_sentences("Hello world. How are")
        assert remainder == "How are"
        assert parts == ["Hello world."]

    def test_multiple_sentences(self):
        remainder, parts = _split_sentences(
            "First. Second! Third? Partial"
        )
        assert remainder == "Partial"
        assert len(parts) == 3

    def test_question_mark_boundary(self):
        remainder, parts = _split_sentences("Really? Yes")
        assert remainder == "Yes"
        assert parts == ["Really?"]

    def test_exclamation_boundary(self):
        remainder, parts = _split_sentences("Wow! Cool")
        assert remainder == "Cool"
        assert parts == ["Wow!"]


class TestOllamaChatStream:
    @mock.patch("pi_bot.chat.requests.post")
    def test_yields_content_chunks(self, mock_post):
        lines = [
            json.dumps({"message": {"content": "Hello "}, "done": False}).encode(),
            json.dumps({"message": {"content": "world"}, "done": False}).encode(),
            json.dumps({"message": {"content": ""}, "done": True}).encode(),
        ]
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        chunks = list(_ollama_chat_stream([{"role": "user", "content": "hi"}]))
        assert len(chunks) == 2
        assert chunks[0] == {"type": "content", "text": "Hello "}
        assert chunks[1] == {"type": "content", "text": "world"}

    @mock.patch("pi_bot.chat.requests.post")
    def test_yields_tool_calls_and_stops(self, mock_post):
        tool_calls = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        lines = [
            json.dumps({"message": {"content": "Let me", "tool_calls": None}, "done": False}).encode(),
            json.dumps({"message": {"tool_calls": tool_calls}, "done": False}).encode(),
            json.dumps({"message": {"content": "after tool"}, "done": False}).encode(),
        ]
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = lines
        mock_post.return_value = mock_response

        chunks = list(_ollama_chat_stream([{"role": "user", "content": "joke"}]))
        types = [c["type"] for c in chunks]
        assert "tool_calls" in types
        assert types.index("tool_calls") == len(types) - 1

    @mock.patch("pi_bot.chat.requests.post")
    def test_sends_correct_payload(self, mock_post):
        mock_response = mock.MagicMock()
        mock_response.iter_lines.return_value = [
            json.dumps({"message": {"content": ""}, "done": True}).encode()
        ]
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "hi"}]
        list(_ollama_chat_stream(messages, tools=TOOLS))

        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        assert payload["model"] == CONFIG["ollama_model"]
        assert payload["messages"] == messages
        assert payload["stream"] is True
        assert payload["tools"] == TOOLS

    @mock.patch("pi_bot.chat.requests.post")
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

        chunks = list(_ollama_chat_stream([]))
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hi"


class TestStreamAndSpeak:
    @mock.patch("pi_bot.chat.speak")
    @mock.patch("pi_bot.chat._ollama_chat_stream")
    def test_simple_response(self, mock_stream, mock_speak):
        mock_stream.return_value = iter([
            {"type": "content", "text": "Hello world."},
        ])
        response, tool_calls = stream_and_speak([])
        assert "Hello world." in response
        assert tool_calls is None

    @mock.patch("pi_bot.chat.speak")
    @mock.patch("pi_bot.chat._ollama_chat_stream")
    def test_tool_calls_returned(self, mock_stream, mock_speak):
        tc = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        mock_stream.return_value = iter([
            {"type": "tool_calls", "tool_calls": tc},
        ])
        response, tool_calls = stream_and_speak([])
        assert tool_calls == tc

    @mock.patch("pi_bot.chat.speak")
    @mock.patch("pi_bot.chat._ollama_chat_stream")
    def test_think_tags_excluded_from_response(self, mock_stream, mock_speak):
        mock_stream.return_value = iter([
            {"type": "content", "text": "<think>internal reasoning</think>"},
            {"type": "content", "text": "Visible answer."},
        ])
        response, _ = stream_and_speak([])
        assert "internal reasoning" not in response
        assert "Visible answer." in response

    @mock.patch("pi_bot.chat.speak")
    @mock.patch("pi_bot.chat.stop_loop")
    @mock.patch("pi_bot.chat.start_loop")
    @mock.patch("pi_bot.chat._ollama_chat_stream")
    def test_think_tag_starts_thinking_cue(self, mock_stream, mock_start,
                                           mock_stop, mock_speak):
        mock_stream.return_value = iter([
            {"type": "content", "text": "<think>thinking</think>Answer."},
        ])
        stream_and_speak([])
        mock_start.assert_called_once_with("thinking")


class TestChatWithOllama:
    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_builds_messages_with_system_prompt(self, mock_sas):
        mock_sas.return_value = ("Reply.", None)
        history = []
        response, end = chat_with_ollama("Hello", history, SAMPLE_JOKES)
        assert end is False

        messages = mock_sas.call_args[0][0]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_updates_conversation_history(self, mock_sas):
        mock_sas.return_value = ("The reply.", None)
        history = []
        response, end = chat_with_ollama("Hi", history, SAMPLE_JOKES)

        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hi"}
        assert history[1] == {"role": "assistant", "content": "The reply."}

    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_trims_history_to_context_turns(self, mock_sas):
        mock_sas.return_value = ("Reply.", None)
        # chat_with_ollama now returns (response, end_conversation) tuple
        original = CONFIG["context_turns"]
        try:
            CONFIG["context_turns"] = 2  # keep 4 messages
            history = []
            for i in range(3):
                chat_with_ollama(f"msg{i}", history, SAMPLE_JOKES)
            assert len(history) == 4
        finally:
            CONFIG["context_turns"] = original

    @mock.patch("pi_bot.chat.execute_tool")
    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_tool_call_loop(self, mock_sas, mock_exec):
        tc = [{"function": {"name": "get_random_joke", "arguments": {}}}]
        mock_sas.side_effect = [
            ("", tc),
            ("Here is a joke!", None),
        ]
        mock_exec.return_value = '{"setup": "...", "punchline": "..."}'

        history = []
        result, end = chat_with_ollama("Tell me a joke", history, SAMPLE_JOKES)

        assert "joke" in result.lower() or result == "Here is a joke!"
        assert end is False
        mock_exec.assert_called_once_with("get_random_joke", {}, SAMPLE_JOKES)
        assert mock_sas.call_count == 2

    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_passes_through_clean_response(self, mock_sas):
        # stream_and_speak already strips think tags before returning
        mock_sas.return_value = ("Clean answer.", None)
        history = []
        result, end = chat_with_ollama("Hi", history, SAMPLE_JOKES)
        assert result == "Clean answer."

    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_uses_german_prompt(self, mock_sas):
        mock_sas.return_value = ("Antwort.", None)
        history = []
        response, end = chat_with_ollama("Hallo", history, SAMPLE_JOKES)
        messages = mock_sas.call_args[0][0]
        assert messages[0]["content"] == SYSTEM_PROMPT_DE

    @mock.patch("pi_bot.chat.execute_tool")
    @mock.patch("pi_bot.chat.stream_and_speak")
    def test_end_conversation_tool_returns_true(self, mock_sas, mock_exec):
        tc = [{"function": {"name": "end_conversation", "arguments": {}}}]
        mock_sas.side_effect = [
            ("", tc),
            ("Goodbye!", None),
        ]
        mock_exec.return_value = '{"status": "conversation_ended"}'

        history = []
        result, end = chat_with_ollama("That's all, thanks", history, SAMPLE_JOKES)
        assert end is True
        assert result == "Goodbye!"

