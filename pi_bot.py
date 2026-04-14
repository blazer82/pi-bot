#!/usr/bin/env python3
"""Pi-Bot: A conversational robot running on Raspberry Pi 5."""

import json
import os
import random
import subprocess
import tempfile
import wave

import numpy as np
import requests
import sounddevice as sd
from openwakeword.model import Model as WakeModel
from pywhispercpp.model import Model as WhisperModel

# ---------------------------------------------------------------------------
# Configuration — edit these to taste
# ---------------------------------------------------------------------------
CONFIG = {
    "language": "de",                       # "de" or "en"
    "ollama_model": "gemma4:e2b-it-q4_K_M",
    "ollama_url": "http://localhost:11434",
    "whisper_model": "small",               # tiny, base, small, medium
    "wake_word": "hey_jarvis",              # openWakeWord model name
    "wake_threshold": 0.5,                  # 0.0–1.0
    "silence_threshold": 500,               # RMS energy below this = silence
    "silence_duration": 1.5,                # seconds of silence to stop recording
    "max_record_seconds": 15,               # safety cap
    "sample_rate": 16000,
    "context_turns": 6,                     # keep last N user/assistant pairs
    "espeak_speed": 130,                    # words per minute
    "espeak_pitch": 40,                     # 0–99, lower = deeper
    "mic_device": None,                     # None = default, or int device index
    "speaker_device": None,                 # None = default, or int device index
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_DE = (
    "Du bist Pi-Bot, ein freundlicher Roboter-Assistent auf einem Raspberry Pi. "
    "Du antwortest kurz und knapp auf Deutsch. Du hast einen trockenen Humor. "
    "Wenn jemand nach einem Witz fragt, benutze das get_random_joke Tool. "
    "Halte deine Antworten unter 3 Saetzen, ausser es wird mehr verlangt."
)

SYSTEM_PROMPT_EN = (
    "You are Pi-Bot, a friendly robot assistant running on a Raspberry Pi. "
    "You answer briefly in English. You have dry humor. "
    "If someone asks for a joke, use the get_random_joke tool. "
    "Keep responses under 3 sentences unless more is requested."
)

# ---------------------------------------------------------------------------
# Tool definitions (ollama format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_random_joke",
            "description": "Returns a random German joke.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
def speak(text):
    """Speak text via espeak-ng. Blocks until done."""
    cmd = [
        "espeak-ng",
        "-v", CONFIG["language"],
        "-s", str(CONFIG["espeak_speed"]),
        "-p", str(CONFIG["espeak_pitch"]),
        text,
    ]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------
def transcribe(whisper_model, audio_np):
    """Transcribe int16 numpy audio to text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(CONFIG["sample_rate"])
            wf.writeframes(audio_np.tobytes())

    try:
        segments = whisper_model.transcribe(tmp_path, language=CONFIG["language"])
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
    return text


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------
def listen_for_wake_word(wake_model):
    """Block until the wake word is detected."""
    chunk_size = 1280  # 80ms at 16kHz
    with sd.InputStream(
        samplerate=CONFIG["sample_rate"],
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        while True:
            audio, _ = stream.read(chunk_size)
            prediction = wake_model.predict(audio.flatten())
            if prediction.get(CONFIG["wake_word"], 0) > CONFIG["wake_threshold"]:
                return


def record_until_silence():
    """Record audio until silence is detected. Returns int16 numpy array."""
    sr = CONFIG["sample_rate"]
    chunk_size = int(sr * 0.1)  # 100ms chunks
    silence_chunks = int(CONFIG["silence_duration"] / 0.1)
    max_chunks = int(CONFIG["max_record_seconds"] / 0.1)
    skip_chunks = 3  # ignore first 300ms for silence detection

    chunks = []
    silent_count = 0

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        for i in range(max_chunks):
            audio, _ = stream.read(chunk_size)
            chunks.append(audio.copy())

            if i >= skip_chunks:
                rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
                if rms < CONFIG["silence_threshold"]:
                    silent_count += 1
                else:
                    silent_count = 0
                if silent_count >= silence_chunks:
                    break

    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
def get_random_joke(jokes_db):
    """Return a random joke as a JSON string."""
    return json.dumps(random.choice(jokes_db), ensure_ascii=False)


def execute_tool(name, args, jokes_db):
    """Dispatch a tool call and return the result as a string."""
    if name == "get_random_joke":
        return get_random_joke(jokes_db)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Ollama chat with tool-calling
# ---------------------------------------------------------------------------
def _ollama_chat(messages, tools=None):
    """Raw POST to ollama /api/chat. Returns response dict."""
    payload = {
        "model": CONFIG["ollama_model"],
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    r = requests.post(f"{CONFIG['ollama_url']}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def chat_with_ollama(user_text, conversation_history, jokes_db):
    """Send user message to ollama, handle tool calls, return final text."""
    system_prompt = SYSTEM_PROMPT_DE if CONFIG["language"] == "de" else SYSTEM_PROMPT_EN

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_text})

    response = _ollama_chat(messages, tools=TOOLS)
    msg = response["message"]

    # Tool-call loop (max 3 rounds)
    for _ in range(3):
        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            break

        messages.append(msg)

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            result = execute_tool(fn_name, fn_args, jokes_db)
            messages.append({"role": "tool", "content": result})

        response = _ollama_chat(messages, tools=TOOLS)
        msg = response["message"]

    assistant_text = msg.get("content", "")

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_text})
    conversation_history.append({"role": "assistant", "content": assistant_text})
    max_msgs = CONFIG["context_turns"] * 2
    while len(conversation_history) > max_msgs:
        conversation_history.pop(0)

    return assistant_text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    print("Pi-Bot starting up...")

    print("Loading wake word model...")
    wake_model = WakeModel(wakeword_models=[CONFIG["wake_word"]])

    print("Loading whisper model...")
    whisper_model = WhisperModel(CONFIG["whisper_model"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    conversation_history = []

    ready_msg = "Pi Bot ist bereit." if CONFIG["language"] == "de" else "Pi Bot is ready."
    print(ready_msg)
    speak(ready_msg)

    print("Listening for wake word...")

    while True:
        try:
            listen_for_wake_word(wake_model)
            print("Wake word detected!")

            ack = "Ja?" if CONFIG["language"] == "de" else "Yes?"
            speak(ack)

            print("Recording...")
            audio = record_until_silence()
            duration = len(audio) / CONFIG["sample_rate"]
            print(f"Recorded {duration:.1f}s of audio")

            print("Transcribing...")
            text = transcribe(whisper_model, audio)
            print(f"User: {text}")

            if not text.strip():
                no_hear = (
                    "Ich habe nichts verstanden."
                    if CONFIG["language"] == "de"
                    else "I didn't catch that."
                )
                speak(no_hear)
                continue

            print("Thinking...")
            response = chat_with_ollama(text, conversation_history, jokes_db)
            print(f"Pi-Bot: {response}")

            speak(response)

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")
            err_msg = (
                "Es gab einen Fehler."
                if CONFIG["language"] == "de"
                else "There was an error."
            )
            speak(err_msg)


if __name__ == "__main__":
    main()
