"""Configuration, system prompts, and tool definitions."""

import os

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Configuration — edit these to taste
# ---------------------------------------------------------------------------
CONFIG = {
    "language": "de",                       # "de" or "en"
    "ollama_model": "gemma4:e2b-it-q4_K_M",
    "ollama_url": "http://localhost:11434",
    "whisper_model": "small",               # tiny, base, small, medium
    "wake_model": os.path.join(_REPO_DIR, "models", "hey_pee_bot.onnx"),
    "wake_threshold": 0.5,                  # 0.0–1.0
    "silence_threshold": 500,               # RMS energy below this = silence
    "silence_duration": 1.5,                # seconds of silence to stop recording
    "max_record_seconds": 15,               # safety cap
    "sample_rate": 16000,
    "context_turns": 25,                    # keep last N user/assistant pairs
    "followup_timeout": 8,                  # seconds to wait for follow-up speech
    "thinking": False,                     # enable <think> reasoning in ollama
    "espeak_voice": "mb-de4",               # MBROLA voice (e.g. mb-de4) or espeak lang
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
    "Halte deine Antworten unter 3 Saetzen, ausser es wird mehr verlangt. "
    "Wenn du sicher bist, dass das Gespraech beendet ist, benutze das end_conversation Tool."
)

SYSTEM_PROMPT_EN = (
    "You are Pi-Bot, a friendly robot assistant running on a Raspberry Pi. "
    "You answer briefly in English. You have dry humor. "
    "You are located in Frankfurt am Main. "
    "If someone asks for a joke, use the get_random_joke tool. "
    "If someone asks about the weather, use the get_weather_forecast tool. "
    "If someone asks how you are doing, use the get_system_status tool and answer creatively based on your system state. "
    "Keep responses under 3 sentences unless more is requested. "
    "When you are confident the conversation has ended, use the end_conversation tool."
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
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": "End the current conversation. Call this when the user indicates they are done or no longer need assistance.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
