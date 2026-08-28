"""
Unit tests for the Calendar tool (SQLite adapter).

Tests all CRUD operations and conflict detection logic.
"""

from __future__ import annotations

import pytest

from app.tools.calendar import SQLiteCalendarAdapter


class TestCalendarCRUD:
    """Tests for calendar Create, Read, Update, Delete operations."""

    def test_create_event(self, repository):
        """Test creating a basic event."""
        cal = SQLiteCalendarAdapter(repository)
        event = cal.create_event(
            title="Team Meeting",
            start_time="2026-01-12T14:00:00",
            end_time="2026-01-12T15:00:00",
            location="Room 204",
        )

        assert event["title"] == "Team Meeting"
        assert event["start_time"] == "2026-01-12T14:00:00"
        assert event["location"] == "Room 204"
        assert event["id"] is not None

    def test_create_event_default_end_time(self, repository):
        """Test that end_time defaults to start_time + 1 hour."""
        cal = SQLiteCalendarAdapter(repository)
        event = cal.create_event(
            title="Quick Call",
            start_time="2026-01-12T10:00:00",
        )

        assert event["end_time"] == "2026-01-12T11:00:00"

    def test_get_upcoming_events(self, repository, sample_events):
        """Test retrieving upcoming events in chronological order."""
        cal = SQLiteCalendarAdapter(repository)
        events = cal.get_upcoming(limit=10)

        # Should have the sample events
        assert len(events) >= 2
        titles = [e["title"] for e in events]
        assert "Team Meeting" in titles

    def test_get_events_by_date(self, repository, sample_events):
        """Test retrieving events on a specific date."""
        cal = SQLiteCalendarAdapter(repository)
        events = cal.get_events_by_date("2026-08-15")

        assert len(events) == 2
        titles = [e["title"] for e in events]
        assert "Team Meeting" in titles
        assert "Dentist" in titles

    def test_update_event_location(self, repository, sample_events):
        """Test updating an event's location."""
        cal = SQLiteCalendarAdapter(repository)
        event_id = sample_events[0]["id"]

        updated = cal.update_event(event_id, location="Conference Hall B")

        assert updated is not None
        assert updated["location"] == "Conference Hall B"
        assert updated["title"] == "Team Meeting"  # Other fields unchanged

    def test_update_event_title(self, repository, sample_events):
        """Test updating an event's title."""
        cal = SQLiteCalendarAdapter(repository)
        event_id = sample_events[0]["id"]

        updated = cal.update_event(event_id, title="Sprint Planning")

        assert updated is not None
        assert updated["title"] == "Sprint Planning"

    def test_delete_event(self, repository, sample_events):
        """Test deleting an event."""
        cal = SQLiteCalendarAdapter(repository)
        event_id = sample_events[0]["id"]

        success = cal.delete_event(event_id)
        assert success is True

        # Verify it's gone
        events = cal.get_events_by_date("2026-08-15")
        titles = [e["title"] for e in events]
        assert "Team Meeting" not in titles

    def test_delete_nonexistent_event(self, repository):
        """Test deleting a non-existent event returns False."""
        cal = SQLiteCalendarAdapter(repository)
        success = cal.delete_event(9999)
        assert success is False

    def test_find_by_title(self, repository, sample_events):
        """Test finding events by partial title match."""
        cal = SQLiteCalendarAdapter(repository)

        results = cal.find_by_title("meeting")
        assert len(results) == 1
        assert results[0]["title"] == "Team Meeting"

    def test_find_by_title_case_insensitive(self, repository, sample_events):
        """Test that title search is case-insensitive."""
        cal = SQLiteCalendarAdapter(repository)

        results = cal.find_by_title("LUNCH")
        assert len(results) == 1
        assert results[0]["title"] == "Lunch with Alex"


class TestCalendarConflicts:
    """Tests for calendar conflict detection."""

    def test_detect_conflict(self, repository, sample_events):
        """Test detecting a time conflict with an existing event."""
        cal = SQLiteCalendarAdapter(repository)

        # Try to book during Team Meeting (10:00-11:00)
        conflicts = cal.check_conflicts(
            start_time="2026-08-15T10:30:00",
            end_time="2026-08-15T11:30:00",
        )

        assert len(conflicts) == 1
        assert conflicts[0]["title"] == "Team Meeting"

    def test_no_conflict(self, repository, sample_events):
        """Test that non-overlapping times show no conflict."""
        cal = SQLiteCalendarAdapter(repository)

        conflicts = cal.check_conflicts(
            start_time="2026-08-15T11:00:00",
            end_time="2026-08-15T12:00:00",
        )

        assert len(conflicts) == 0

    def test_conflict_excludes_self(self, repository, sample_events):
        """Test that updating an event excludes itself from conflict check."""
        cal = SQLiteCalendarAdapter(repository)
        event_id = sample_events[0]["id"]  # Team Meeting

        conflicts = cal.check_conflicts(
            start_time="2026-08-15T10:00:00",
            end_time="2026-08-15T11:00:00",
            exclude_id=event_id,
        )

        assert len(conflicts) == 0

    def test_multiple_conflicts(self, repository, sample_events):
        """Test detecting multiple conflicts at once."""
        cal = SQLiteCalendarAdapter(repository)

        # A time range spanning both Team Meeting and Dentist
        conflicts = cal.check_conflicts(
            start_time="2026-08-15T09:00:00",
            end_time="2026-08-15T16:00:00",
        )

        assert len(conflicts) == 2
