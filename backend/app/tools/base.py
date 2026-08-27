"""
Abstract base classes for pluggable Weather and Calendar providers.

Implementing these interfaces allows seamless swapping between
local (SQLite) and cloud (Google Calendar, OpenWeatherMap) adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseWeatherProvider(ABC):
    """
    Abstract interface for weather data providers.

    Implementations: OpenMeteoProvider, WttrInProvider
    """

    @abstractmethod
    async def get_current(self, city: str) -> dict[str, Any]:
        """
        Get current weather conditions for a city.

        Args:
            city: City name (e.g., 'Marburg', 'Frankfurt').

        Returns:
            Dict with keys: temperature, condition, humidity,
            wind_speed, precipitation_probability.
        """
        ...

    @abstractmethod
    async def get_forecast(self, city: str, date: str) -> dict[str, Any]:
        """
        Get weather forecast for a city on a specific date.

        Args:
            city: City name.
            date: ISO date string (e.g., '2026-08-15').

        Returns:
            Dict with keys: date, temperature_max, temperature_min,
            condition, precipitation_probability, wind_speed.
        """
        ...


class BaseCalendarProvider(ABC):
    """
    Abstract interface for calendar storage providers.

    Implementations: SQLiteCalendarAdapter, GoogleCalendarAdapter (Phase 2)
    """

    @abstractmethod
    def create_event(
        self,
        title: str,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str = "",
        description: str = "",
        is_all_day: bool = False,
    ) -> dict[str, Any]:
        """Create a new calendar event."""
        ...

    @abstractmethod
    def get_upcoming(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get upcoming events sorted by start time."""
        ...

    @abstractmethod
    def get_events_by_date(self, date_str: str) -> list[dict[str, Any]]:
        """Get all events on a specific date."""
        ...

    @abstractmethod
    def update_event(self, event_id: int, **fields: Any) -> dict[str, Any] | None:
        """Update fields on an existing event."""
        ...

    @abstractmethod
    def delete_event(self, event_id: int) -> bool:
        """Delete an event by ID."""
        ...

    @abstractmethod
    def check_conflicts(
        self, start_time: str, end_time: str, exclude_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Check for scheduling conflicts in a time range."""
        ...

    @abstractmethod
    def find_by_title(self, title: str) -> list[dict[str, Any]]:
        """Find events matching a title (case-insensitive)."""
        ...
