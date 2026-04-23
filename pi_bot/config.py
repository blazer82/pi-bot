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
    "whisper_model": "small-q8_0",          # tiny, base, small, medium
    "wake_model": os.path.join(_REPO_DIR, "models", "hey_pee_bot.onnx"),
    "wake_threshold": 0.5,                  # 0.0–1.0
    "silence_threshold": 150,               # RMS energy below this = silence
    "silence_duration": 1.5,                # seconds of silence to stop recording
    "max_record_seconds": 15,               # safety cap
    "sample_rate": 16000,
    "context_turns": 10,                    # keep last N user/assistant pairs
    "followup_timeout": 12,                 # seconds to wait for follow-up speech
    "thinking": False,                     # enable <think> reasoning in ollama
    "piper_model": "pibot1",
    "piper_data_dir": os.path.join(_REPO_DIR, "models", "piper"),
    # None = default, or int for multi-speaker models
    "piper_speaker": None,
    "piper_length_scale": 1.3,              # >1.0 = slower, <1.0 = faster
    "mic_device": None,                     # None = default, or int device index
    "mic_channels": 2,                      # channels to open (match device)
    "mic_channel_select": 0,                # which channel to extract
    "speaker_device": None,                 # None = default, or int device index
    # set to a directory path to save last_recording.wav
    "debug_recording_dir": None,
    "sounds_dir": os.path.join(_REPO_DIR, "sounds"),
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
    "WICHTIG: Wenn du ein Tool aufrufst, bekommst du ein Tool-Ergebnis zurück. "
    "Du MUSST den Inhalt dieses Ergebnisses dem Benutzer mitteilen. "
    "Erfinde NIEMALS eigene Antworten wenn ein Tool-Ergebnis vorliegt. "
    "get_random_joke: Gibt einen Witz mit setup und punchline zurück. "
    "Sag zuerst das setup, dann die punchline. "
    "get_weather_forecast: Fasse die zurückgegebene Vorhersage zusammen. "
    "get_system_status: Berichte die wichtigsten Werte aus dem Ergebnis. "
    "end_conversation: Beendet das Gespräch."
)

# ---------------------------------------------------------------------------
# Tool definitions (ollama format)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_random_joke",
            "description": "Gibt einen zufälligen deutschen Witz zurück.",
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
            "description": "Gibt die Wettervorhersage für Frankfurt am Main zurück: Temperatur, Niederschlag und Bedingungen für die nächsten 3 Tage.",
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
            "description": "Gibt den Systemstatus des Raspberry Pi zurück: CPU-Temperatur, CPU-Auslastung, Speicher, Festplatte und Laufzeit.",
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
            "description": "Beendet das aktuelle Gespräch. Aufrufen wenn der Benutzer sich verabschiedet oder kein weiteres Anliegen hat.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
