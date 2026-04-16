"""Ollama chat with streaming, tool calling, and sentence-by-sentence speech."""

import json
import re
import threading

import requests

from pi_bot.config import CONFIG, TOOLS, SYSTEM_PROMPT_DE, SYSTEM_PROMPT_EN
from pi_bot.tts import speak
from pi_bot.tools import execute_tool


def _ollama_chat_stream(messages, tools=None):
    """POST to ollama /api/chat with streaming enabled.

    Yields dicts of the form:
      {"type": "content", "text": "..."}
      {"type": "tool_calls", "tool_calls": [...]}
    """
    payload = {
        "model": CONFIG["ollama_model"],
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": CONFIG["ollama_num_ctx"]},
    }
    if tools:
        payload["tools"] = tools
    payload["think"] = CONFIG["thinking"]
    r = requests.post(
        f"{CONFIG['ollama_url']}/api/chat",
        json=payload,
        stream=True,
        timeout=None,  # no cap — speaking blocks while the stream buffers
    )
    r.raise_for_status()

    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        msg = chunk.get("message", {})
        if msg.get("tool_calls"):
            yield {"type": "tool_calls", "tool_calls": msg["tool_calls"]}
            return
        content = msg.get("content", "")
        if content:
            yield {"type": "content", "text": content}
        if chunk.get("done"):
            return


def warmup_ollama():
    """Send a minimal request to Ollama to pre-fill the KV cache.

    Ollama caches the prompt prefix (system prompt + tool definitions) from the
    last request.  By sending a cheap dummy request at startup, the first real
    user query can skip the expensive prompt evaluation for that prefix.
    """
    system_prompt = SYSTEM_PROMPT_DE if CONFIG["language"] == "de" else SYSTEM_PROMPT_EN

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "hi"},
    ]
    payload = {
        "model": CONFIG["ollama_model"],
        "messages": messages,
        "tools": TOOLS,
        "think": CONFIG["thinking"],
        "stream": False,
        "options": {"num_ctx": CONFIG["ollama_num_ctx"], "num_predict": 1},
    }
    try:
        r = requests.post(
            f"{CONFIG['ollama_url']}/api/chat",
            json=payload,
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()
        prompt_tokens = data.get("prompt_eval_count", "?")
        print(f"Ollama warmup done ({prompt_tokens} prompt tokens cached)")
    except Exception as e:
        print(f"Ollama warmup failed (non-fatal): {e}")


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:\u2014])\s+")

# Speak partial buffer if it exceeds this many chars without a boundary
_BUFFER_CHAR_LIMIT = 120


def _speak_sentences(buffer):
    """Speak every complete sentence or clause in *buffer*.

    Splits on sentence endings (.!?) first.  If no sentence boundary is found
    but the buffer exceeds ``_BUFFER_CHAR_LIMIT`` characters, falls back to
    clause boundaries (,;:—) to keep latency low.

    Returns ``(remainder, spoken_text)`` where *remainder* is the trailing
    fragment that has not been spoken yet.
    """
    # Try sentence boundaries first
    parts = _SENTENCE_BOUNDARY.split(buffer)
    if len(parts) > 1:
        spoken = ""
        for sentence in parts[:-1]:
            s = sentence.strip()
            if s:
                speak(s)
                spoken += s + " "
        return parts[-1], spoken.strip()

    # Fall back to clause boundaries when the buffer is getting long
    if len(buffer) > _BUFFER_CHAR_LIMIT:
        parts = _CLAUSE_BOUNDARY.split(buffer)
        if len(parts) > 1:
            spoken = ""
            for clause in parts[:-1]:
                s = clause.strip()
                if s:
                    speak(s)
                    spoken += s + " "
            return parts[-1], spoken.strip()

    return buffer, ""


def stream_and_speak(messages, tools=None):
    """Stream an Ollama response, handle ``<think>`` tags, and speak
    sentence-by-sentence as tokens arrive.

    Returns ``(raw_response_text, tool_calls_or_None)``.
    The raw response preserves Ollama's exact text (minus think tags) so that
    the conversation history matches the KV cache prefix on the next turn.
    """
    # Speak the cue in a background thread so the Ollama request starts
    # immediately — saves ~0.5-1s of dead time.
    cue = "Moment..." if CONFIG["language"] == "de" else "Analysing..."
    cue_thread = threading.Thread(target=speak, args=(cue,), daemon=True)
    cue_thread.start()

    buffer = ""
    raw_chunks = []       # exact text from Ollama for KV cache compatibility
    in_think = False
    think_content = ""
    tool_calls = None

    for chunk in _ollama_chat_stream(messages, tools=tools):
        if chunk["type"] == "tool_calls":
            tool_calls = chunk["tool_calls"]
            break

        raw_chunks.append(chunk["text"])
        buffer += chunk["text"]

        # -- Thinking tag handling --
        if not in_think and "<think>" in buffer:
            pre, _, post = buffer.partition("<think>")
            if pre.strip():
                cue_thread.join()
                remainder, _ = _speak_sentences(pre)
                if remainder.strip():
                    speak(remainder.strip())
            in_think = True
            buffer = post

        if in_think:
            if "</think>" in buffer:
                think_part, _, remainder = buffer.partition("</think>")
                think_content += think_part
                print(f"[Think] {think_content.strip()}")
                in_think = False
                buffer = remainder
            else:
                think_content += buffer
                buffer = ""
                continue

        # -- Sentence-by-sentence speaking --
        buffer, spoken = _speak_sentences(buffer)
        if spoken:
            # Wait for the cue to finish before speaking real content
            cue_thread.join()

    # Speak any remaining text in the buffer
    cue_thread.join()
    leftover = buffer.strip()
    if leftover and not in_think:
        speak(leftover)

    # Build raw response preserving Ollama's exact whitespace, only strip
    # think tags so the text matches the KV cache on the next turn.
    raw_response = "".join(raw_chunks)
    raw_response = re.sub(r"<think>.*?</think>", "",
                          raw_response, flags=re.DOTALL).strip()

    return raw_response, tool_calls


def chat_with_ollama(user_text, conversation_history, jokes_db):
    """Send user message to ollama, handle tool calls, stream and speak."""
    system_prompt = SYSTEM_PROMPT_DE if CONFIG["language"] == "de" else SYSTEM_PROMPT_EN

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    raw_response, tool_calls = stream_and_speak(messages, tools=TOOLS)

    # Tool-call loop (max 3 rounds)
    end_conversation = False
    tool_messages = []  # track intermediate tool messages for history
    for _ in range(3):
        if not tool_calls:
            break

        tc_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        messages.append(tc_msg)
        tool_messages.append(tc_msg)
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            if fn_name == "end_conversation":
                end_conversation = True
            result = execute_tool(fn_name, fn_args, jokes_db)
            tool_result_msg = {"role": "tool", "content": result}
            messages.append(tool_result_msg)
            tool_messages.append(tool_result_msg)

        # Stream the post-tool response too
        raw_response, tool_calls = stream_and_speak(messages, tools=TOOLS)

    # Update conversation history — preserve exact messages so the Ollama
    # KV cache prefix matches on the next turn.
    conversation_history.append({"role": "user", "content": user_text})
    for msg in tool_messages:
        conversation_history.append(msg)
    conversation_history.append(
        {"role": "assistant", "content": raw_response})
    max_msgs = CONFIG["context_turns"] * 2
    while len(conversation_history) > max_msgs:
        conversation_history.pop(0)

    return raw_response, end_conversation
