"""Tool implementations and dispatcher."""

import json
import random
import shutil
from datetime import datetime

import psutil
import requests

from pi_bot.config import CONFIG

# ---------------------------------------------------------------------------
# Weather codes (WMO)
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
