"""Pi-Bot: A conversational robot running on Raspberry Pi 5."""

from pi_bot.config import CONFIG, SYSTEM_PROMPT_DE, TOOLS
from pi_bot.tts import speak
from pi_bot.stt import transcribe
from pi_bot.audio import listen_for_wake_word, record_until_silence, wait_for_followup
from pi_bot.tools import (
    WMO_WEATHER_CODES,
    get_weather_forecast,
    get_system_status,
    get_random_joke,
    execute_tool,
)
from pi_bot.chat import (
    _ollama_chat_stream,
    _SENTENCE_BOUNDARY,
    _speak_sentences,
    stream_and_speak,
    chat_with_ollama,
)
