"""Tests for tool implementations and dispatcher."""

import json
from unittest import mock

from tests.conftest import SAMPLE_JOKES
from pi_bot.config import CONFIG
from pi_bot.tools import (
    get_random_joke,
    get_weather_forecast,
    get_system_status,
    execute_tool,
)


class TestGetRandomJoke:
    def test_returns_valid_json(self):
        result = get_random_joke(SAMPLE_JOKES)
        parsed = json.loads(result)
        assert "setup" in parsed
        assert "punchline" in parsed

    def test_returns_joke_from_db(self):
        result = json.loads(get_random_joke(SAMPLE_JOKES))
        assert result in SAMPLE_JOKES

    def test_single_joke_db(self):
        single = [SAMPLE_JOKES[0]]
        result = json.loads(get_random_joke(single))
        assert result == SAMPLE_JOKES[0]


class TestGetWeatherForecast:
    MOCK_FORECAST_RESPONSE = {
        "daily": {
            "time": ["2026-04-14", "2026-04-15", "2026-04-16"],
            "temperature_2m_max": [18.5, 20.1, 16.3],
            "temperature_2m_min": [8.2, 10.0, 7.5],
            "precipitation_sum": [0.0, 2.3, 5.1],
            "weathercode": [0, 61, 63],
        }
    }

    @mock.patch("pi_bot.tools.requests.get")
    def test_successful_forecast(self, mock_get):
        mock_response = mock.MagicMock()
        mock_response.json.return_value = self.MOCK_FORECAST_RESPONSE
        mock_get.return_value = mock_response

        result = json.loads(get_weather_forecast())
        assert result["location"] == CONFIG["location_name"]
        assert len(result["forecast"]) == 3
        day = result["forecast"][0]
        assert "date" in day
        assert "temp_max" in day
        assert "temp_min" in day
        assert "precipitation_mm" in day
        assert "conditions" in day

    @mock.patch("pi_bot.tools.requests.get")
    def test_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection refused")
        result = json.loads(get_weather_forecast())
        assert "error" in result


class _MockAddr:
    """Minimal stand-in for psutil network address with a .family.name attribute."""
    def __init__(self, family_name, address):
        self.family = mock.MagicMock(name=family_name)
        self.family.name = family_name
        self.address = address


class TestGetSystemStatus:
    MOCK_NET_ADDRS = {"eth0": [_MockAddr("AF_INET", "192.168.1.42")]}
    MOCK_NET_LOOPBACK_ONLY = {"lo": [_MockAddr("AF_INET", "127.0.0.1")]}

    @mock.patch("pi_bot.tools.requests.head")
    @mock.patch("pi_bot.tools.psutil.net_if_addrs")
    @mock.patch("pi_bot.tools.psutil.cpu_percent", return_value=25.0)
    @mock.patch("pi_bot.tools.psutil.boot_time", return_value=1744500000.0)
    @mock.patch("pi_bot.tools.psutil.virtual_memory")
    @mock.patch("pi_bot.tools.shutil.disk_usage")
    def test_successful_status(self, mock_disk, mock_mem, mock_boot, mock_cpu, mock_net, mock_head):
        mock_temps = mock.MagicMock(return_value={"cpu_thermal": [mock.MagicMock(current=42.0)]})
        mock_mem.return_value = mock.MagicMock(percent=60.5, available=2 * 1024 * 1024 * 1024)
        mock_disk.return_value = mock.MagicMock(total=32 * 1024**3, used=16 * 1024**3, free=16 * 1024**3)
        mock_net.return_value = self.MOCK_NET_ADDRS

        with mock.patch.object(
            __import__("pi_bot.tools", fromlist=["psutil"]).psutil,
            "sensors_temperatures", mock_temps, create=True
        ):
            result = json.loads(get_system_status())
        assert result["cpu_temp_c"] == 42.0
        assert result["cpu_percent"] == 25.0
        assert result["memory_used_percent"] == 60.5
        assert "memory_available_mb" in result
        assert "disk_used_percent" in result
        assert "disk_free_gb" in result
        assert "uptime" in result
        assert result["network_connected"] is True
        assert result["internet_connected"] is True

    @mock.patch("pi_bot.tools.requests.head", side_effect=Exception("no internet"))
    @mock.patch("pi_bot.tools.psutil.net_if_addrs")
    @mock.patch("pi_bot.tools.psutil.cpu_percent", return_value=10.0)
    @mock.patch("pi_bot.tools.psutil.boot_time", return_value=1744500000.0)
    @mock.patch("pi_bot.tools.psutil.virtual_memory")
    @mock.patch("pi_bot.tools.shutil.disk_usage")
    def test_network_but_no_internet(self, mock_disk, mock_mem, mock_boot, mock_cpu, mock_net, mock_head):
        mock_mem.return_value = mock.MagicMock(percent=50.0, available=1024 * 1024 * 1024)
        mock_disk.return_value = mock.MagicMock(total=32 * 1024**3, used=16 * 1024**3, free=16 * 1024**3)
        mock_net.return_value = self.MOCK_NET_ADDRS

        result = json.loads(get_system_status())
        assert result["network_connected"] is True
        assert result["internet_connected"] is False

    @mock.patch("pi_bot.tools.psutil.net_if_addrs")
    @mock.patch("pi_bot.tools.psutil.cpu_percent", return_value=10.0)
    @mock.patch("pi_bot.tools.psutil.boot_time", return_value=1744500000.0)
    @mock.patch("pi_bot.tools.psutil.virtual_memory")
    @mock.patch("pi_bot.tools.shutil.disk_usage")
    def test_no_network(self, mock_disk, mock_mem, mock_boot, mock_cpu, mock_net):
        mock_mem.return_value = mock.MagicMock(percent=50.0, available=1024 * 1024 * 1024)
        mock_disk.return_value = mock.MagicMock(total=32 * 1024**3, used=16 * 1024**3, free=16 * 1024**3)
        mock_net.return_value = self.MOCK_NET_LOOPBACK_ONLY

        result = json.loads(get_system_status())
        assert result["network_connected"] is False
        assert result["internet_connected"] is False

    @mock.patch("pi_bot.tools.requests.head")
    @mock.patch("pi_bot.tools.psutil.net_if_addrs")
    @mock.patch("pi_bot.tools.psutil.cpu_percent", return_value=10.0)
    @mock.patch("pi_bot.tools.psutil.boot_time", return_value=1744500000.0)
    @mock.patch("pi_bot.tools.psutil.virtual_memory")
    @mock.patch("pi_bot.tools.shutil.disk_usage")
    def test_no_temp_sensors(self, mock_disk, mock_mem, mock_boot, mock_cpu, mock_net, mock_head):
        mock_mem.return_value = mock.MagicMock(percent=50.0, available=1024 * 1024 * 1024)
        mock_disk.return_value = mock.MagicMock(total=32 * 1024**3, used=16 * 1024**3, free=16 * 1024**3)
        mock_net.return_value = self.MOCK_NET_ADDRS

        result = json.loads(get_system_status())
        assert result["cpu_temp_c"] is None

    @mock.patch("pi_bot.tools.psutil.cpu_percent", side_effect=Exception("fail"))
    def test_error_handling(self, mock_cpu):
        result = json.loads(get_system_status())
        assert "error" in result


class TestExecuteTool:
    def test_known_tool(self):
        result = execute_tool("get_random_joke", {}, SAMPLE_JOKES)
        parsed = json.loads(result)
        assert parsed in SAMPLE_JOKES

    @mock.patch("pi_bot.tools.get_weather_forecast")
    def test_weather_tool_dispatch(self, mock_weather):
        mock_weather.return_value = '{"location": "Frankfurt am Main", "forecast": []}'
        result = execute_tool("get_weather_forecast", {}, SAMPLE_JOKES)
        mock_weather.assert_called_once()
        parsed = json.loads(result)
        assert parsed["location"] == "Frankfurt am Main"

    @mock.patch("pi_bot.tools.get_system_status")
    def test_system_status_dispatch(self, mock_status):
        mock_status.return_value = '{"cpu_percent": 25.0}'
        result = execute_tool("get_system_status", {}, SAMPLE_JOKES)
        mock_status.assert_called_once()
        parsed = json.loads(result)
        assert parsed["cpu_percent"] == 25.0

    def test_end_conversation_dispatch(self):
        result = execute_tool("end_conversation", {}, SAMPLE_JOKES)
        parsed = json.loads(result)
        assert parsed["status"] == "conversation_ended"

    def test_unknown_tool(self):
        result = execute_tool("nonexistent", {}, SAMPLE_JOKES)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "nonexistent" in parsed["error"]
