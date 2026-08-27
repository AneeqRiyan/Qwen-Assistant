"""
Calendar provider implementation using local SQLite storage.

Wraps Repository calendar methods behind the BaseCalendarProvider interface
so the system can seamlessly swap to Google Calendar in Phase 2.
"""

from __future__ import annotations

from typing import Any

from app.database.repository import Repository
from app.logger import get_logger
from app.tools.base import BaseCalendarProvider

logger = get_logger("calendar")


class SQLiteCalendarAdapter(BaseCalendarProvider):
    """
    Local SQLite-backed calendar provider.

    Delegates all operations to the Repository's calendar methods.
    Implements BaseCalendarProvider for future Google Calendar swap.
    """

    def __init__(self, repository: Repository) -> None:
        self.repo = repository

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
        event = self.repo.create_event(
            title=title,
            start_time=start_time,
            end_time=end_time,
            location=location,
            description=description,
            is_all_day=is_all_day,
        )
        logger.info(f"Calendar: Created event '{title}' (id={event.get('id')})")
        return event

    def get_upcoming(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get upcoming events sorted by start time."""
        events = self.repo.get_upcoming_events(limit=limit)
        logger.debug(f"Calendar: Retrieved {len(events)} upcoming events")
        return events

    def get_events_by_date(self, date_str: str) -> list[dict[str, Any]]:
        """Get all events on a specific date."""
        events = self.repo.get_events_by_date(date_str)
        logger.debug(f"Calendar: {len(events)} events on {date_str}")
        return events

    def update_event(self, event_id: int, **fields: Any) -> dict[str, Any] | None:
        """Update fields on an existing event."""
        event = self.repo.update_event(event_id, **fields)
        if event:
            logger.info(f"Calendar: Updated event {event_id}")
        else:
            logger.warning(f"Calendar: Event {event_id} not found for update")
        return event

    def delete_event(self, event_id: int) -> bool:
        """Delete an event by ID."""
        success = self.repo.delete_event(event_id)
        return success

    def check_conflicts(
        self, start_time: str, end_time: str, exclude_id: int | None = None
    ) -> list[dict[str, Any]]:
        """Check for scheduling conflicts in a time range."""
        conflicts = self.repo.check_conflicts(start_time, end_time, exclude_id)
        if conflicts:
            logger.info(f"Calendar: Found {len(conflicts)} conflicts for {start_time} - {end_time}")
        return conflicts

    def find_by_title(self, title: str) -> list[dict[str, Any]]:
        """Find events matching a title (case-insensitive)."""
        return self.repo.find_events_by_title(title)
