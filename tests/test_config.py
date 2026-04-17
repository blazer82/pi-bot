"""Tests for configuration and constants."""

from pi_bot.config import CONFIG, TOOLS, SYSTEM_PROMPT_DE


class TestConfig:
    def test_config_has_required_keys(self):
        required = [
            "ollama_model", "ollama_url", "whisper_model",
            "wake_model", "wake_threshold", "silence_threshold",
            "silence_duration", "max_record_seconds", "sample_rate",
            "context_turns", "followup_timeout", "piper_model", "piper_data_dir",
            "piper_length_scale", "mic_device", "speaker_device",
        ]
        for key in required:
            assert key in CONFIG, f"Missing config key: {key}"

    def test_tools_schema_valid(self):
        assert len(TOOLS) == 4
        names = {t["function"]["name"] for t in TOOLS}
        assert names == {"get_random_joke", "get_weather_forecast", "get_system_status", "end_conversation"}
        for tool in TOOLS:
            assert tool["type"] == "function"

    def test_system_prompts_defined(self):
        assert len(SYSTEM_PROMPT_DE) > 0
