"""
Conversation context manager.

Handles:
- 10-turn sliding window for active LLM context
- Anaphora resolution (location, date, event references)
- Pending action state for multi-turn operations
- Relative date parsing
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytz
from dateutil import parser as dateutil_parser

from app.config import get_config
from app.database.repository import Repository
from app.logger import get_logger

logger = get_logger("context")


class PendingAction:
    """
    Tracks an incomplete multi-turn operation.

    Examples:
    - Calendar create missing time: awaiting user's time input
    - Calendar conflict detected: awaiting user's resolution choice
    - Calendar update with multiple matches: awaiting user's selection
    """

    def __init__(
        self,
        action_type: str,
        data: dict[str, Any] | None = None,
        prompt: str = "",
    ) -> None:
        self.action_type = action_type  # e.g., "create_event_needs_time", "conflict_resolution"
        self.data = data or {}          # Partial data collected so far
        self.prompt = prompt            # The clarification question asked

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "data": self.data,
            "prompt": self.prompt,
        }


class ContextManager:
    """
    Manages conversation context, references, and state.

    Maintains a sliding window of recent turns for LLM context,
    tracks entity references for anaphora resolution, and manages
    pending multi-turn operations.
    """

    def __init__(self, repository: Repository, session_id: int) -> None:
        self.repo = repository
        self.session_id = session_id

        config = get_config()
        self.window_size = config.llm.context_window_turns
        self.timezone_str = config.locale.timezone
        self.timezone = pytz.timezone(self.timezone_str)

        # Anaphora tracking
        self.last_mentioned_location: str | None = None
        self.last_mentioned_date: str | None = None
        self.last_created_event_id: int | None = None
        self.last_action_event_id: int | None = None

        # Pending multi-turn action
        self.pending_action: PendingAction | None = None

    def get_current_datetime(self) -> datetime:
        """Get the current datetime in the configured timezone."""
        return datetime.now(self.timezone)

    def get_context_messages(self) -> list[dict[str, str]]:
        """
        Load the last N turns from the database as LLM message history.

        Returns:
            List of message dicts with 'role' and 'content' keys,
            ordered chronologically (oldest first).
        """
        turns = self.repo.get_recent_turns(self.session_id, n=self.window_size)

        messages = []
        for turn in turns:
            messages.append({
                "role": turn["role"],
                "content": turn["content"],
            })

        logger.debug(f"Loaded {len(messages)} context messages for session {self.session_id}")
        return messages

    def save_turn(self, role: str, content: str, tool_calls_json: str | None = None) -> int:
        """Save a conversation turn and return the turn number."""
        return self.repo.save_turn(
            session_id=self.session_id,
            role=role,
            content=content,
            tool_calls_json=tool_calls_json,
        )

    def update_references(
        self,
        location: str | None = None,
        date: str | None = None,
        event_id: int | None = None,
        action_event_id: int | None = None,
    ) -> None:
        """
        Update tracked entity references for anaphora resolution.

        Args:
            location: A city or place mentioned in the latest turn.
            date: A date mentioned in the latest turn (YYYY-MM-DD).
            event_id: ID of a newly created event.
            action_event_id: ID of the event most recently acted upon.
        """
        if location:
            self.last_mentioned_location = location
            logger.debug(f"Context: last_location = '{location}'")
        if date:
            self.last_mentioned_date = date
            logger.debug(f"Context: last_date = '{date}'")
        if event_id is not None:
            self.last_created_event_id = event_id
            logger.debug(f"Context: last_created_event = {event_id}")
        if action_event_id is not None:
            self.last_action_event_id = action_event_id
            logger.debug(f"Context: last_action_event = {action_event_id}")

    def set_pending_action(
        self,
        action_type: str,
        data: dict[str, Any] | None = None,
        prompt: str = "",
    ) -> None:
        """Set a pending action for multi-turn resolution."""
        self.pending_action = PendingAction(action_type, data, prompt)
        logger.info(f"Pending action set: {action_type}")

    def clear_pending_action(self) -> None:
        """Clear the current pending action."""
        if self.pending_action:
            logger.info(f"Pending action cleared: {self.pending_action.action_type}")
        self.pending_action = None

    def resolve_relative_date(self, date_text: str) -> str | None:
        """
        Resolve a relative date expression to an ISO date string.

        Args:
            date_text: Natural language date like 'today', 'tomorrow',
                       'this Friday', 'next Monday', 'the 12th of January'.

        Returns:
            ISO date string (YYYY-MM-DD) or None if unresolvable.
        """
        now = self.get_current_datetime()
        text = date_text.lower().strip()

        # Direct mappings
        if text == "today":
            return now.strftime("%Y-%m-%d")
        elif text == "tomorrow":
            return (now + timedelta(days=1)).strftime("%Y-%m-%d")
        elif text == "yesterday":
            return (now - timedelta(days=1)).strftime("%Y-%m-%d")

        # Day-of-week references ("this Friday", "next Monday", "Saturday")
        day_names = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
        }

        for prefix in ("this ", "next ", ""):
            for day_name, day_num in day_names.items():
                if text == f"{prefix}{day_name}" or text == day_name:
                    current_dow = now.weekday()
                    days_ahead = day_num - current_dow
                    if "next" in text:
                        days_ahead += 7
                    if days_ahead <= 0:
                        days_ahead += 7
                    target = now + timedelta(days=days_ahead)
                    return target.strftime("%Y-%m-%d")

        # Try parsing with dateutil as fallback
        try:
            parsed = dateutil_parser.parse(date_text, fuzzy=True, default=now.replace(tzinfo=None))
            # If the parsed date is in the past (same year), assume next year
            if parsed.date() < now.date() and parsed.year == now.year:
                parsed = parsed.replace(year=parsed.year + 1)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, OverflowError):
            logger.warning(f"Could not parse date: '{date_text}'")
            return None

    def get_reference_summary(self) -> str:
        """
        Build a context reference summary to inject into LLM messages.

        Returns:
            A brief string summarizing active references.
        """
        parts = []
        if self.last_mentioned_location:
            parts.append(f"Last mentioned location: {self.last_mentioned_location}")
        if self.last_mentioned_date:
            parts.append(f"Last mentioned date: {self.last_mentioned_date}")
        if self.last_created_event_id:
            parts.append(f"Last created event ID: {self.last_created_event_id}")
        if self.last_action_event_id:
            parts.append(f"Last acted-on event ID: {self.last_action_event_id}")
        if self.pending_action:
            parts.append(f"Pending action: {self.pending_action.action_type}")

        return " | ".join(parts) if parts else ""
