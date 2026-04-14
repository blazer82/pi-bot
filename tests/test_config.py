"""Tests for configuration and constants."""

from pi_bot.config import CONFIG, TOOLS, SYSTEM_PROMPT_DE, SYSTEM_PROMPT_EN


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
            assert key in CONFIG, f"Missing config key: {key}"

    def test_tools_schema_valid(self):
        assert len(TOOLS) == 3
        names = {t["function"]["name"] for t in TOOLS}
        assert names == {"get_random_joke", "get_weather_forecast", "get_system_status"}
        for tool in TOOLS:
            assert tool["type"] == "function"

    def test_system_prompts_defined(self):
        assert len(SYSTEM_PROMPT_DE) > 0
        assert len(SYSTEM_PROMPT_EN) > 0
