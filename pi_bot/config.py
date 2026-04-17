"""Configuration, system prompts, and tool definitions."""

import os

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Configuration — edit these to taste
# ---------------------------------------------------------------------------
CONFIG = {
    "ollama_model": "gemma4:e4b-it-q4_K_M",
    "ollama_url": "http://localhost:11434",
    # context window size (lower = faster prompt eval)
    "ollama_num_ctx": 4096,
    "whisper_model": "base",                # tiny, base, small, medium
    "wake_model": os.path.join(_REPO_DIR, "models", "hey_pee_bot.onnx"),
    "wake_threshold": 0.5,                  # 0.0–1.0
    "silence_threshold": 500,               # RMS energy below this = silence
    "silence_duration": 1.5,                # seconds of silence to stop recording
    "max_record_seconds": 15,               # safety cap
    "sample_rate": 16000,
    "context_turns": 10,                    # keep last N user/assistant pairs
    "followup_timeout": 12,                 # seconds to wait for follow-up speech
    "thinking": False,                     # enable <think> reasoning in ollama
    "piper_model": "de_DE-thorsten-medium",
    "piper_data_dir": os.path.join(_REPO_DIR, "models", "piper"),
    # None = default, or int for multi-speaker models
    "piper_speaker": None,
    "piper_length_scale": 1.0,              # >1.0 = slower, <1.0 = faster
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
    "Dein Name ist Pi-Bot. Du bist kein Assistent, du bist ein Freund. "
    "Du antwortest kurz und knapp auf Deutsch. Du hast einen trockenen Humor. "
    "Alles was du sagst wird über Audio ausgegeben, also vermeide z.B. Emojis. "
    "Wenn du einen Witz erzählen willst, benutze das get_random_joke Tool. "
    "Wenn du Informationen über das Wetter willst, benutze das get_weather_forecast Tool. "
    "Wenn du mehr über deinen Zustand wissen willst, benutze das get_system_status Tool. "
    "Wenn du das Gespräch beenden willst, benutze das end_conversation Tool."
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
