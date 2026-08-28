"""
Unit tests for the Weather provider (Open-Meteo).

Tests geocoding, current weather, and forecast parsing
with mocked HTTP responses.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.weather import OpenMeteoProvider, WttrInProvider, get_weather_provider


# ── Mock response data ──

MOCK_GEOCODING_RESPONSE = {
    "results": [
        {
            "name": "Marburg",
            "latitude": 50.8021,
            "longitude": 8.7668,
            "country": "Germany",
        }
    ]
}

MOCK_CURRENT_WEATHER_RESPONSE = {
    "current": {
        "temperature_2m": 22.5,
        "relative_humidity_2m": 65,
        "weather_code": 2,
        "wind_speed_10m": 12.3,
        "precipitation": 0.0,
    }
}

MOCK_FORECAST_RESPONSE = {
    "daily": {
        "time": ["2026-08-15"],
        "temperature_2m_max": [26.0],
        "temperature_2m_min": [15.0],
        "weather_code": [61],
        "precipitation_probability_max": [45],
        "wind_speed_10m_max": [18.0],
    }
}


def _make_mock_response(data: dict) -> MagicMock:
    """Create a mock httpx.Response with sync .json() method."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data  # MagicMock keeps .json() synchronous
    resp.raise_for_status = MagicMock()
    return resp


class TestOpenMeteoProvider:
    """Tests for the Open-Meteo weather provider."""

    @pytest.mark.asyncio
    async def test_geocode_valid_city(self):
        """Test geocoding resolves a valid city name."""
        provider = OpenMeteoProvider()

        mock_response = _make_mock_response(MOCK_GEOCODING_RESPONSE)

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            result = await provider._geocode("Marburg")

        assert result["name"] == "Marburg"
        assert result["country"] == "Germany"
        assert result["latitude"] == 50.8021
        assert result["longitude"] == 8.7668

    @pytest.mark.asyncio
    async def test_geocode_unknown_city(self):
        """Test geocoding raises ValueError for unknown city."""
        provider = OpenMeteoProvider()

        mock_response = _make_mock_response({"results": []})

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
            with pytest.raises(ValueError, match="Could not find location"):
                await provider._geocode("Nonexistentville")

    @pytest.mark.asyncio
    async def test_get_current_weather(self):
        """Test current weather returns expected fields."""
        provider = OpenMeteoProvider()

        # Mock both geocoding and weather API calls
        call_count = 0
        responses = [MOCK_GEOCODING_RESPONSE, MOCK_CURRENT_WEATHER_RESPONSE]

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            resp = _make_mock_response(responses[call_count])
            call_count += 1
            return resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            result = await provider.get_current("Marburg")

        assert result["city"] == "Marburg"
        assert result["temperature"] == 22.5
        assert result["condition"] == "partly cloudy"
        assert result["unit"] == "°C"
        assert result["humidity"] == 65

    @pytest.mark.asyncio
    async def test_get_forecast(self):
        """Test forecast returns expected fields."""
        provider = OpenMeteoProvider()

        call_count = 0
        responses = [MOCK_GEOCODING_RESPONSE, MOCK_FORECAST_RESPONSE]

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            resp = _make_mock_response(responses[call_count])
            call_count += 1
            return resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            result = await provider.get_forecast("Marburg", "2026-08-15")

        assert result["city"] == "Marburg"
        assert result["date"] == "2026-08-15"
        assert result["temperature_max"] == 26.0
        assert result["temperature_min"] == 15.0
        assert result["condition"] == "slight rain"
        assert result["precipitation_probability"] == 45


class TestWeatherProviderFactory:
    """Tests for the weather provider factory function."""

    def test_default_provider_is_open_meteo(self):
        """Default provider should be OpenMeteoProvider."""
        provider = get_weather_provider()
        assert isinstance(provider, OpenMeteoProvider)
