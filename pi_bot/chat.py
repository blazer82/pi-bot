"""Ollama chat with streaming, tool calling, and sentence-by-sentence speech."""

import json
import re
from datetime import datetime

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
    }
    if tools:
        payload["tools"] = tools
    if CONFIG["thinking"]:
        payload["think"] = True
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


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _speak_sentences(buffer):
    """Speak every complete sentence in *buffer*.

    Returns ``(remainder, spoken_text)`` where *remainder* is the trailing
    fragment that has not been spoken yet.
    """
    parts = _SENTENCE_BOUNDARY.split(buffer)
    if len(parts) <= 1:
        return buffer, ""

    spoken = ""
    for sentence in parts[:-1]:
        s = sentence.strip()
        if s:
            speak(s)
            spoken += s + " "
    return parts[-1], spoken.strip()


def stream_and_speak(messages, tools=None):
    """Stream an Ollama response, handle ``<think>`` tags, and speak
    sentence-by-sentence as tokens arrive.

    Returns ``(full_response_text, tool_calls_or_None)``.
    """
    cue = "Moment..." if CONFIG["language"] == "de" else "Analysing..."
    speak(cue)

    buffer = ""
    full_response = ""
    in_think = False
    think_content = ""
    tool_calls = None

    for chunk in _ollama_chat_stream(messages, tools=tools):
        if chunk["type"] == "tool_calls":
            tool_calls = chunk["tool_calls"]
            break

        buffer += chunk["text"]

        # -- Thinking tag handling --
        if not in_think and "<think>" in buffer:
            pre, _, post = buffer.partition("<think>")
            if pre.strip():
                remainder, spoken = _speak_sentences(pre)
                full_response += spoken
                if remainder.strip():
                    speak(remainder.strip())
                    full_response += (" " + remainder.strip()
                                      ) if full_response else remainder.strip()
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
            full_response += (" " + spoken) if full_response else spoken

    # Speak any remaining text in the buffer
    leftover = buffer.strip()
    if leftover and not in_think:
        speak(leftover)
        full_response += (" " + leftover) if full_response else leftover

    return full_response.strip(), tool_calls


def chat_with_ollama(user_text, conversation_history, jokes_db):
    """Send user message to ollama, handle tool calls, stream and speak."""
    system_prompt = SYSTEM_PROMPT_DE if CONFIG["language"] == "de" else SYSTEM_PROMPT_EN
    now = datetime.now()
    if CONFIG["language"] == "de":
        days = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
        system_prompt += f"\n\nAktuelles Datum und Uhrzeit: {now:%Y-%m-%d %H:%M} {days[now.weekday()]}"
    else:
        system_prompt += f"\n\nCurrent date and time: {now:%Y-%m-%d %H:%M %A}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    full_response, tool_calls = stream_and_speak(messages, tools=TOOLS)

    # Tool-call loop (max 3 rounds) — falls back to non-streaming
    for _ in range(3):
        if not tool_calls:
            break

        messages.append({"role": "assistant", "content": "",
                        "tool_calls": tool_calls})
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            result = execute_tool(fn_name, fn_args, jokes_db)
            messages.append({"role": "tool", "content": result})

        # Stream the post-tool response too
        full_response, tool_calls = stream_and_speak(messages, tools=TOOLS)

    # Strip any residual thinking tags (safety net for tool-call path)
    full_response = re.sub(r"<think>.*?</think>", "",
                           full_response, flags=re.DOTALL).strip()

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append(
        {"role": "assistant", "content": full_response})
    max_msgs = CONFIG["context_turns"] * 2
    while len(conversation_history) > max_msgs:
        conversation_history.pop(0)

    return full_response
