#!/usr/bin/env python3
"""Pi-Bot: A conversational robot running on Raspberry Pi 5."""

import json
import os
import shutil
from datetime import datetime
import random
import re
import subprocess
import tempfile
import wave

import numpy as np
import psutil
import requests
try:
    import sounddevice as sd
    from openwakeword.model import Model as WakeModel
    from pywhispercpp.model import Model as WhisperModel
except ImportError:
    sd = None
    WakeModel = None
    WhisperModel = None

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
    "context_turns": 25,                    # keep last N user/assistant pairs
    "followup_timeout": 8,                  # seconds to wait for follow-up speech
    "thinking": False,                     # enable <think> reasoning in ollama
    "espeak_speed": 130,                    # words per minute
    "espeak_pitch": 40,                     # 0–99, lower = deeper
    "mic_device": None,                     # None = default, or int device index
    "speaker_device": None,                 # None = default, or int device index
    "location_name": "Frankfurt am Main",
    "location_lat": 50.1109,
    "location_lon": 8.6821,
}

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_DE = (
    "Du bist Pi-Bot, ein freundlicher Roboter-Assistent auf einem Raspberry Pi. "
    "Du antwortest kurz und knapp auf Deutsch. Du hast einen trockenen Humor. "
    "Du befindest dich in Frankfurt am Main. "
    "Wenn jemand nach einem Witz fragt, benutze das get_random_joke Tool. "
    "Wenn jemand nach dem Wetter fragt, benutze das get_weather_forecast Tool. "
    "Wenn jemand fragt wie es dir geht, benutze das get_system_status Tool und antworte kreativ basierend auf deinem Systemzustand. "
    "Halte deine Antworten unter 3 Saetzen, ausser es wird mehr verlangt."
)

SYSTEM_PROMPT_EN = (
    "You are Pi-Bot, a friendly robot assistant running on a Raspberry Pi. "
    "You answer briefly in English. You have dry humor. "
    "You are located in Frankfurt am Main. "
    "If someone asks for a joke, use the get_random_joke tool. "
    "If someone asks about the weather, use the get_weather_forecast tool. "
    "If someone asks how you are doing, use the get_system_status tool and answer creatively based on your system state. "
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": "Returns a weather forecast for the bot's location (Frankfurt am Main). Returns temperature, precipitation, and conditions for the next 3 days.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_status",
            "description": "Returns system status of the Raspberry Pi: CPU temperature, CPU usage, memory usage, disk usage, and uptime.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
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
        segments = whisper_model.transcribe(
            tmp_path, language=CONFIG["language"])
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


def wait_for_followup():
    """Listen for follow-up speech within the timeout window.

    Phase 1: wait up to ``followup_timeout`` seconds for speech onset.
    Phase 2: once speech is detected, record until silence (same logic as
    ``record_until_silence``).

    Returns an int16 numpy array if speech was captured, or *None* if the
    timeout expired without any speech.
    """
    sr = CONFIG["sample_rate"]
    chunk_size = int(sr * 0.1)  # 100 ms chunks
    timeout_chunks = int(CONFIG["followup_timeout"] / 0.1)
    silence_chunks_needed = int(CONFIG["silence_duration"] / 0.1)
    max_record_chunks = int(CONFIG["max_record_seconds"] / 0.1)

    chunks = []
    speech_detected = False
    silent_count = 0

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="int16",
        blocksize=chunk_size,
        device=CONFIG["mic_device"],
    ) as stream:
        # Phase 1 — wait for speech onset
        for _ in range(timeout_chunks):
            audio, _ = stream.read(chunk_size)
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms >= CONFIG["silence_threshold"]:
                speech_detected = True
                chunks.append(audio.copy())
                break

        if not speech_detected:
            return None

        # Phase 2 — record until silence
        for _ in range(max_record_chunks):
            audio, _ = stream.read(chunk_size)
            chunks.append(audio.copy())
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
            if rms < CONFIG["silence_threshold"]:
                silent_count += 1
            else:
                silent_count = 0
            if silent_count >= silence_chunks_needed:
                break

    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------
WMO_WEATHER_CODES = {
    0: ("Klar", "Clear sky"),
    1: ("Überwiegend klar", "Mainly clear"),
    2: ("Teilweise bewölkt", "Partly cloudy"),
    3: ("Bewölkt", "Overcast"),
    45: ("Nebel", "Fog"),
    48: ("Raunebel", "Depositing rime fog"),
    51: ("Leichter Nieselregen", "Light drizzle"),
    53: ("Mäßiger Nieselregen", "Moderate drizzle"),
    55: ("Starker Nieselregen", "Dense drizzle"),
    61: ("Leichter Regen", "Slight rain"),
    63: ("Mäßiger Regen", "Moderate rain"),
    65: ("Starker Regen", "Heavy rain"),
    66: ("Leichter Gefrierregen", "Light freezing rain"),
    67: ("Starker Gefrierregen", "Heavy freezing rain"),
    71: ("Leichter Schneefall", "Slight snowfall"),
    73: ("Mäßiger Schneefall", "Moderate snowfall"),
    75: ("Starker Schneefall", "Heavy snowfall"),
    77: ("Schneegriesel", "Snow grains"),
    80: ("Leichte Regenschauer", "Slight rain showers"),
    81: ("Mäßige Regenschauer", "Moderate rain showers"),
    82: ("Heftige Regenschauer", "Violent rain showers"),
    85: ("Leichte Schneeschauer", "Slight snow showers"),
    86: ("Starke Schneeschauer", "Heavy snow showers"),
    95: ("Gewitter", "Thunderstorm"),
    96: ("Gewitter mit leichtem Hagel", "Thunderstorm with slight hail"),
    99: ("Gewitter mit starkem Hagel", "Thunderstorm with heavy hail"),
}


def get_weather_forecast():
    """Fetch a 3-day weather forecast from Open-Meteo for the configured location."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": CONFIG["location_lat"],
                "longitude": CONFIG["location_lon"],
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return json.dumps({"error": f"Weather API error: {e}"}, ensure_ascii=False)

    daily = data.get("daily", {})
    lang_idx = 0 if CONFIG["language"] == "de" else 1
    days = []
    for i in range(len(daily.get("time", []))):
        code = daily["weathercode"][i]
        desc = WMO_WEATHER_CODES.get(code, ("Unbekannt", "Unknown"))[lang_idx]
        days.append({
            "date": daily["time"][i],
            "temp_max": daily["temperature_2m_max"][i],
            "temp_min": daily["temperature_2m_min"][i],
            "precipitation_mm": daily["precipitation_sum"][i],
            "conditions": desc,
        })

    return json.dumps({
        "location": CONFIG["location_name"],
        "forecast": days,
    }, ensure_ascii=False)


def get_system_status():
    """Return system status (CPU temp, CPU usage, memory, disk, uptime)."""
    try:
        cpu_temp = None
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                for entries in temps.values():
                    if entries:
                        cpu_temp = entries[0].current
                        break

        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        uptime_s = int(psutil.boot_time())
        uptime_delta = datetime.now() - datetime.fromtimestamp(uptime_s)
        hours, remainder = divmod(int(uptime_delta.total_seconds()), 3600)
        minutes = remainder // 60

        # Network connectivity
        network_connected = False
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family.name in ("AF_INET", "AF_INET6") and not addr.address.startswith("127.") and addr.address != "::1":
                    network_connected = True
                    break
            if network_connected:
                break

        internet_connected = False
        if network_connected:
            try:
                requests.head("http://clients3.google.com/generate_204", timeout=3)
                internet_connected = True
            except Exception:
                pass

        status = {
            "cpu_temp_c": cpu_temp,
            "cpu_percent": psutil.cpu_percent(interval=0.5),
            "memory_used_percent": mem.percent,
            "memory_available_mb": round(mem.available / 1024 / 1024),
            "disk_used_percent": round(disk.used / disk.total * 100, 1),
            "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
            "uptime": f"{hours}h {minutes}m",
            "network_connected": network_connected,
            "internet_connected": internet_connected,
        }
        return json.dumps(status, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"System status error: {e}"}, ensure_ascii=False)


def get_random_joke(jokes_db):
    """Return a random joke as a JSON string."""
    return json.dumps(random.choice(jokes_db), ensure_ascii=False)


def execute_tool(name, args, jokes_db):
    """Dispatch a tool call and return the result as a string."""
    if name == "get_random_joke":
        return get_random_joke(jokes_db)
    if name == "get_weather_forecast":
        return get_weather_forecast()
    if name == "get_system_status":
        return get_system_status()
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Ollama chat with tool-calling
# ---------------------------------------------------------------------------


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

    no_hear = (
        "Ich habe nichts verstanden."
        if CONFIG["language"] == "de"
        else "I didn't catch that."
    )
    err_msg = (
        "Es gab einen Fehler."
        if CONFIG["language"] == "de"
        else "There was an error."
    )

    ready_msg = "Pi Bot ist bereit." if CONFIG["language"] == "de" else "Pi Bot is ready."
    print(ready_msg)
    speak(ready_msg)

    while True:
        try:
            print("Listening for wake word...")
            listen_for_wake_word(wake_model)
            print("Wake word detected!")

            # Fresh conversation context per wake-word activation
            conversation_history = []

            ack = "Ja?" if CONFIG["language"] == "de" else "Yes?"
            speak(ack)

            # --- First utterance ---
            print("Recording...")
            audio = record_until_silence()
            duration = len(audio) / CONFIG["sample_rate"]
            print(f"Recorded {duration:.1f}s of audio")

            print("Transcribing...")
            text = transcribe(whisper_model, audio)
            print(f"User: {text}")

            if not text.strip():
                speak(no_hear)
                continue

            print("Thinking...")
            response = chat_with_ollama(text, conversation_history, jokes_db)
            print(f"Pi-Bot: {response}")

            # --- Follow-up loop ---
            while True:
                print("Listening for follow-up...")
                audio = wait_for_followup()
                if audio is None:
                    print("No follow-up detected, returning to wake word.")
                    break

                duration = len(audio) / CONFIG["sample_rate"]
                print(f"Recorded {duration:.1f}s of audio")

                print("Transcribing...")
                text = transcribe(whisper_model, audio)
                print(f"User: {text}")

                if not text.strip():
                    continue  # empty transcription — keep listening for follow-up

                print("Thinking...")
                response = chat_with_ollama(
                    text, conversation_history, jokes_db)
                print(f"Pi-Bot: {response}")

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"Error: {e}")
            speak(err_msg)


def chat_mode():
    """Text-input chat mode for development without microphone hardware."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    print("Pi-Bot chat mode")
    print('Type messages to chat. "reset" clears history, "exit" quits.')
    print()

    ready_msg = "Pi Bot ist bereit." if CONFIG["language"] == "de" else "Pi Bot is ready."
    print(ready_msg)
    speak(ready_msg)

    conversation_history = []

    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break

            if not text.strip():
                continue

            cmd = text.strip().lower()
            if cmd in ("exit", "quit"):
                break
            if cmd == "reset":
                conversation_history = []
                print("Conversation history cleared.")
                continue

            response = chat_with_ollama(text, conversation_history, jokes_db)
            print(f"Pi-Bot: {response}\n")
    except KeyboardInterrupt:
        pass

    print("\nBye.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pi-Bot")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Text-input chat mode (no mic/wake word needed)",
    )
    args = parser.parse_args()

    if args.chat:
        chat_mode()
    else:
        main()
