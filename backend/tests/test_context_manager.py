"""
Unit tests for the Context Manager.

Tests 10-turn sliding window, relative date parsing,
anaphora resolution, and pending action state.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
import pytz

from app.core.context_manager import ContextManager, PendingAction


class TestContextWindow:
    """Tests for the 10-turn sliding window."""

    def test_empty_context(self, repository, sample_session):
        """Test that empty session returns no context messages."""
        ctx = ContextManager(repository, sample_session)
        messages = ctx.get_context_messages()
        assert messages == []

    def test_save_and_retrieve_turns(self, repository, sample_session):
        """Test saving and retrieving conversation turns."""
        ctx = ContextManager(repository, sample_session)

        ctx.save_turn("user", "Hello!")
        ctx.save_turn("assistant", "Hi there! How can I help?")
        ctx.save_turn("user", "What's the weather?")

        messages = ctx.get_context_messages()
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello!"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "What's the weather?"

    def test_window_limit(self, repository, sample_session):
        """Test that only the last N turns are returned."""
        ctx = ContextManager(repository, sample_session)
        ctx.window_size = 3  # Override for test

        # Save 6 turns
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            ctx.save_turn(role, f"Message {i}")

        messages = ctx.get_context_messages()
        assert len(messages) == 3
        # Should be the last 3 messages (chronological order)
        assert messages[0]["content"] == "Message 3"
        assert messages[2]["content"] == "Message 5"


class TestRelativeDateParsing:
    """Tests for relative date resolution."""

    @pytest.fixture
    def ctx(self, repository, sample_session):
        """Create a ContextManager with a fixed 'now' for deterministic testing."""
        ctx = ContextManager(repository, sample_session)
        return ctx

    def test_today(self, ctx):
        """Test 'today' resolves to current date."""
        now = ctx.get_current_datetime()
        result = ctx.resolve_relative_date("today")
        assert result == now.strftime("%Y-%m-%d")

    def test_tomorrow(self, ctx):
        """Test 'tomorrow' resolves to next day."""
        now = ctx.get_current_datetime()
        from datetime import timedelta
        expected = (now + timedelta(days=1)).strftime("%Y-%m-%d")
        result = ctx.resolve_relative_date("tomorrow")
        assert result == expected

    def test_day_of_week(self, ctx):
        """Test day-of-week references resolve to future dates."""
        result = ctx.resolve_relative_date("Friday")
        assert result is not None
        # Verify it's actually a Friday
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 4  # Friday = 4

    def test_next_monday(self, ctx):
        """Test 'next Monday' skips to next week."""
        result = ctx.resolve_relative_date("next Monday")
        assert result is not None
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed.weekday() == 0  # Monday = 0

    def test_absolute_date(self, ctx):
        """Test parsing an absolute date like '12th of January'."""
        result = ctx.resolve_relative_date("12th of January")
        assert result is not None
        assert "-01-12" in result

    def test_invalid_date(self, ctx):
        """Test that gibberish returns None."""
        result = ctx.resolve_relative_date("blorp florp")
        assert result is None


class TestAnaphoraResolution:
    """Tests for entity reference tracking."""

    def test_track_location(self, repository, sample_session):
        """Test that last mentioned location is tracked."""
        ctx = ContextManager(repository, sample_session)

        ctx.update_references(location="Frankfurt")
        assert ctx.last_mentioned_location == "Frankfurt"

        ctx.update_references(location="Marburg")
        assert ctx.last_mentioned_location == "Marburg"

    def test_track_date(self, repository, sample_session):
        """Test that last mentioned date is tracked."""
        ctx = ContextManager(repository, sample_session)

        ctx.update_references(date="2026-08-15")
        assert ctx.last_mentioned_date == "2026-08-15"

    def test_track_event_id(self, repository, sample_session):
        """Test that last created/acted event IDs are tracked."""
        ctx = ContextManager(repository, sample_session)

        ctx.update_references(event_id=42)
        assert ctx.last_created_event_id == 42

        ctx.update_references(action_event_id=42)
        assert ctx.last_action_event_id == 42

    def test_reference_summary(self, repository, sample_session):
        """Test the reference summary string generation."""
        ctx = ContextManager(repository, sample_session)

        ctx.update_references(location="Frankfurt", date="2026-08-15")
        summary = ctx.get_reference_summary()

        assert "Frankfurt" in summary
        assert "2026-08-15" in summary


class TestPendingAction:
    """Tests for multi-turn pending action state."""

    def test_set_pending_action(self, repository, sample_session):
        """Test setting a pending action."""
        ctx = ContextManager(repository, sample_session)

        ctx.set_pending_action(
            "create_event_needs_time",
            data={"title": "Meeting"},
            prompt="What time should the meeting be?",
        )

        assert ctx.pending_action is not None
        assert ctx.pending_action.action_type == "create_event_needs_time"
        assert ctx.pending_action.data["title"] == "Meeting"

    def test_clear_pending_action(self, repository, sample_session):
        """Test clearing a pending action."""
        ctx = ContextManager(repository, sample_session)

        ctx.set_pending_action("conflict_resolution")
        ctx.clear_pending_action()

        assert ctx.pending_action is None

    def test_pending_action_to_dict(self):
        """Test PendingAction serialization."""
        action = PendingAction(
            "create_event_needs_time",
            data={"title": "Lunch"},
            prompt="What time?",
        )

        d = action.to_dict()
        assert d["action_type"] == "create_event_needs_time"
        assert d["data"]["title"] == "Lunch"
        assert d["prompt"] == "What time?"
