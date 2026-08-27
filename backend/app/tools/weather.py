"""
Weather provider implementations.

Primary: OpenMeteoProvider — Free, no API key, geocoding built-in.
Fallback: WttrInProvider — Free, no API key, simpler data.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.config import get_config
from app.logger import get_logger
from app.tools.base import BaseWeatherProvider

logger = get_logger("weather")

# ── Weather condition code to human-readable description ──
# Based on WMO Weather Interpretation Codes (used by Open-Meteo)
WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class OpenMeteoProvider(BaseWeatherProvider):
    """
    Weather provider using the Open-Meteo API.

    - Free, open-source, no API key required.
    - Geocoding API converts city names to lat/long.
    - Forecast API returns hourly/daily weather data.
    """

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self) -> None:
        config = get_config()
        self.units = config.weather.units.temperature  # "celsius" | "fahrenheit"
        self._temp_unit = "celsius" if self.units == "celsius" else "fahrenheit_2"

    async def _geocode(self, city: str) -> dict[str, Any]:
        """
        Convert a city name to latitude/longitude using Open-Meteo Geocoding API.

        Args:
            city: City name (e.g., 'Marburg', 'Frankfurt').

        Returns:
            Dict with 'latitude', 'longitude', 'name', 'country'.

        Raises:
            ValueError: If the city cannot be found.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                self.GEOCODING_URL,
                params={"name": city, "count": 1, "language": "en"},
            )
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            raise ValueError(f"Could not find location: '{city}'")

        loc = results[0]
        logger.debug(f"Geocoded '{city}' → {loc['name']}, {loc.get('country', '')} "
                      f"({loc['latitude']}, {loc['longitude']})")
        return {
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "name": loc["name"],
            "country": loc.get("country", ""),
        }

    async def get_current(self, city: str) -> dict[str, Any]:
        """Get current weather for a city."""
        geo = await self._geocode(city)

        temp_unit = "fahrenheit" if self.units == "fahrenheit" else "celsius"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                self.FORECAST_URL,
                params={
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
                    "temperature_unit": temp_unit,
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        weather_code = current.get("weather_code", 0)
        unit_symbol = "°F" if self.units == "fahrenheit" else "°C"

        result = {
            "city": geo["name"],
            "country": geo["country"],
            "temperature": current.get("temperature_2m"),
            "unit": unit_symbol,
            "condition": WMO_CODES.get(weather_code, "unknown"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_unit": "km/h",
            "precipitation": current.get("precipitation", 0),
        }

        logger.info(f"Current weather for {geo['name']}: {result['temperature']}{unit_symbol}, {result['condition']}")
        return result

    async def get_forecast(self, city: str, date: str) -> dict[str, Any]:
        """Get weather forecast for a city on a specific date."""
        geo = await self._geocode(city)

        temp_unit = "fahrenheit" if self.units == "fahrenheit" else "celsius"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                self.FORECAST_URL,
                params={
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max,wind_speed_10m_max",
                    "temperature_unit": temp_unit,
                    "wind_speed_unit": "kmh",
                    "timezone": "auto",
                    "start_date": date,
                    "end_date": date,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        daily = data.get("daily", {})

        # Extract first (and only) day's data
        if not daily.get("time"):
            raise ValueError(f"No forecast data available for {date}")

        weather_code = daily["weather_code"][0] if daily.get("weather_code") else 0
        unit_symbol = "°F" if self.units == "fahrenheit" else "°C"

        result = {
            "city": geo["name"],
            "country": geo["country"],
            "date": date,
            "temperature_max": daily.get("temperature_2m_max", [None])[0],
            "temperature_min": daily.get("temperature_2m_min", [None])[0],
            "unit": unit_symbol,
            "condition": WMO_CODES.get(weather_code, "unknown"),
            "precipitation_probability": daily.get("precipitation_probability_max", [0])[0],
            "wind_speed_max": daily.get("wind_speed_10m_max", [None])[0],
            "wind_unit": "km/h",
        }

        logger.info(
            f"Forecast for {geo['name']} on {date}: "
            f"{result['temperature_max']}{unit_symbol}/{result['temperature_min']}{unit_symbol}, "
            f"{result['condition']}, {result['precipitation_probability']}% rain"
        )
        return result


class WttrInProvider(BaseWeatherProvider):
    """
    Fallback weather provider using wttr.in.

    - Free, no API key required.
    - Returns simpler weather data as JSON.
    """

    BASE_URL = "https://wttr.in"

    async def get_current(self, city: str) -> dict[str, Any]:
        """Get current weather from wttr.in."""
        config = get_config()
        unit_param = "m" if config.weather.units.temperature == "celsius" else "u"

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/{city}",
                params={"format": "j1"},
                headers={"User-Agent": "VoiceAssistant/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current_condition", [{}])[0]
        unit_symbol = "°C" if config.weather.units.temperature == "celsius" else "°F"
        temp_key = "temp_C" if config.weather.units.temperature == "celsius" else "temp_F"

        return {
            "city": city,
            "country": "",
            "temperature": float(current.get(temp_key, 0)),
            "unit": unit_symbol,
            "condition": current.get("weatherDesc", [{}])[0].get("value", "unknown"),
            "humidity": int(current.get("humidity", 0)),
            "wind_speed": float(current.get("windspeedKmph", 0)),
            "wind_unit": "km/h",
            "precipitation": float(current.get("precipMM", 0)),
        }

    async def get_forecast(self, city: str, date: str) -> dict[str, Any]:
        """Get forecast from wttr.in (limited to 3 days)."""
        config = get_config()

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{self.BASE_URL}/{city}",
                params={"format": "j1"},
                headers={"User-Agent": "VoiceAssistant/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()

        unit_symbol = "°C" if config.weather.units.temperature == "celsius" else "°F"
        temp_max_key = "maxtempC" if config.weather.units.temperature == "celsius" else "maxtempF"
        temp_min_key = "mintempC" if config.weather.units.temperature == "celsius" else "mintempF"

        # Find matching date in forecast
        for day in data.get("weather", []):
            if day.get("date") == date:
                hourly = day.get("hourly", [{}])
                # Use midday (index 4, which is ~12:00) for condition
                midday = hourly[4] if len(hourly) > 4 else hourly[0] if hourly else {}

                return {
                    "city": city,
                    "country": "",
                    "date": date,
                    "temperature_max": float(day.get(temp_max_key, 0)),
                    "temperature_min": float(day.get(temp_min_key, 0)),
                    "unit": unit_symbol,
                    "condition": midday.get("weatherDesc", [{}])[0].get("value", "unknown"),
                    "precipitation_probability": int(midday.get("chanceofrain", 0)),
                    "wind_speed_max": float(day.get("maxtempC", 0)),
                    "wind_unit": "km/h",
                }

        raise ValueError(f"No forecast available for {date} (wttr.in supports up to 3 days)")


def get_weather_provider() -> BaseWeatherProvider:
    """
    Factory function to get the configured weather provider.

    Returns:
        An instance of the configured weather provider.
    """
    config = get_config()
    provider_name = config.weather.provider

    if provider_name == "wttr-in":
        logger.info("Using WttrIn weather provider")
        return WttrInProvider()

    # Default to Open-Meteo
    logger.info("Using Open-Meteo weather provider")
    return OpenMeteoProvider()
