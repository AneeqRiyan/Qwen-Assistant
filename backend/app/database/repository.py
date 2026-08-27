"""
Data access layer (Repository pattern) for the Voice Personal Assistant.

Provides methods for session management, conversation history CRUD,
and calendar appointment CRUD with conflict detection.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.logger import get_logger

logger = get_logger("repository")


class Repository:
    """
    Central data access object for all database operations.

    Wraps a sqlite3.Connection and provides typed methods for
    sessions, conversation history, and calendar appointments.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ──────────────────────────────────────────────────────────────
    # Session Management
    # ──────────────────────────────────────────────────────────────

    def get_or_create_latest_session(self) -> dict[str, Any]:
        """
        Resume the most recently active session, or create a new one.

        Returns:
            Dict with session 'id', 'created_at', and 'last_active_at'.
        """
        cursor = self.conn.execute(
            "SELECT * FROM sessions ORDER BY last_active_at DESC LIMIT 1"
        )
        row = cursor.fetchone()

        if row:
            session = dict(row)
            # Update last_active_at timestamp
            self.conn.execute(
                "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?",
                (session["id"],),
            )
            self.conn.commit()
            logger.info(f"Resumed session {session['id']}")
            return session

        return self.create_new_session()

    def create_new_session(self) -> dict[str, Any]:
        """
        Create a brand-new conversation session.

        Returns:
            Dict with the new session's 'id', 'created_at', 'last_active_at'.
        """
        cursor = self.conn.execute(
            "INSERT INTO sessions DEFAULT VALUES"
        )
        self.conn.commit()
        session_id = cursor.lastrowid

        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

        logger.info(f"Created new session {session_id}")
        return dict(row)

    def get_all_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get all sessions ordered by last active timestamp with message count and preview.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session dicts.
        """
        cursor = self.conn.execute(
            """SELECT 
                s.id,
                s.created_at,
                s.last_active_at,
                COUNT(t.id) as turn_count,
                COALESCE(
                    (SELECT content FROM conversation_turns WHERE session_id = s.id AND role = 'user' ORDER BY turn_number ASC LIMIT 1),
                    'New Conversation'
                ) as preview
            FROM sessions s
            LEFT JOIN conversation_turns t ON s.id = t.session_id
            GROUP BY s.id
            ORDER BY s.last_active_at DESC
            LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ──────────────────────────────────────────────────────────────
    # Conversation History
    # ──────────────────────────────────────────────────────────────

    def save_turn(
        self,
        session_id: int,
        role: str,
        content: str,
        tool_calls_json: str | None = None,
    ) -> int:
        """
        Save a conversation turn to the database.

        Args:
            session_id: The session this turn belongs to.
            role: One of 'user', 'assistant', 'system', 'tool'.
            content: The text content of the turn.
            tool_calls_json: Optional JSON string of tool calls made.

        Returns:
            The turn_number assigned to this turn within the session.
        """
        # Get the next turn number for this session
        cursor = self.conn.execute(
            "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        turn_number = cursor.fetchone()[0]

        self.conn.execute(
            """INSERT INTO conversation_turns (session_id, turn_number, role, content, tool_calls_json)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, turn_number, role, content, tool_calls_json),
        )

        # Update session last_active_at
        self.conn.execute(
            "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?",
            (session_id,),
        )
        self.conn.commit()

        logger.debug(f"Saved turn {turn_number} (role={role}) to session {session_id}")
        return turn_number

    def get_recent_turns(self, session_id: int, n: int = 10) -> list[dict[str, Any]]:
        """
        Get the most recent N turns for a session (for LLM context window).

        Args:
            session_id: The session to retrieve turns from.
            n: Number of recent turns to retrieve. Defaults to 10.

        Returns:
            List of turn dicts ordered oldest-first (chronological).
        """
        cursor = self.conn.execute(
            """SELECT * FROM conversation_turns
               WHERE session_id = ?
               ORDER BY turn_number DESC
               LIMIT ?""",
            (session_id, n),
        )
        rows = [dict(row) for row in cursor.fetchall()]
        rows.reverse()  # Return in chronological order
        return rows

    def get_history_paginated(
        self,
        session_id: int,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Get paginated conversation history for Phase 2 UI.

        Args:
            session_id: The session to paginate.
            limit: Maximum number of turns to return.
            offset: Number of turns to skip (for pagination).

        Returns:
            List of turn dicts ordered newest-first.
        """
        cursor = self.conn.execute(
            """SELECT * FROM conversation_turns
               WHERE session_id = ?
               ORDER BY turn_number DESC
               LIMIT ? OFFSET ?""",
            (session_id, limit, offset),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_turn_count(self, session_id: int) -> int:
        """Get total number of turns in a session."""
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        return cursor.fetchone()[0]

    # ──────────────────────────────────────────────────────────────
    # Calendar Appointments
    # ──────────────────────────────────────────────────────────────

    def create_event(
        self,
        title: str,
        start_time: str | None = None,
        end_time: str | None = None,
        location: str = "",
        description: str = "",
        is_all_day: bool = False,
    ) -> dict[str, Any]:
        """
        Create a new calendar appointment.

        Args:
            title: Event title (required).
            start_time: ISO format datetime string (e.g., '2026-01-12T14:00:00').
            end_time: ISO format datetime string. Defaults to start_time + 1 hour.
            location: Event location.
            description: Event description.
            is_all_day: Whether this is an all-day event.

        Returns:
            Dict of the created event.
        """
        # Default end_time to start_time + 1 hour if not provided
        if start_time and not end_time:
            start_dt = datetime.fromisoformat(start_time)
            end_dt = start_dt + timedelta(hours=1)
            end_time = end_dt.isoformat()

        cursor = self.conn.execute(
            """INSERT INTO appointments (title, start_time, end_time, location, description, is_all_day)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (title, start_time, end_time, location, description, int(is_all_day)),
        )
        self.conn.commit()
        event_id = cursor.lastrowid

        logger.info(f"Created event '{title}' (id={event_id})")
        return self._get_event_by_id(event_id)

    def get_upcoming_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get upcoming events sorted by start_time.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dicts sorted by start_time ascending.
        """
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            """SELECT * FROM appointments
               WHERE start_time >= ? OR is_all_day = 1
               ORDER BY start_time ASC
               LIMIT ?""",
            (now, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_events_by_date(self, date_str: str) -> list[dict[str, Any]]:
        """
        Get all events on a specific date.

        Args:
            date_str: Date string in 'YYYY-MM-DD' format.

        Returns:
            List of event dicts on that date.
        """
        cursor = self.conn.execute(
            """SELECT * FROM appointments
               WHERE DATE(start_time) = ?
               ORDER BY start_time ASC""",
            (date_str,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_event(self, event_id: int, **fields: Any) -> dict[str, Any] | None:
        """
        Update fields on an existing appointment.

        Args:
            event_id: The ID of the event to update.
            **fields: Keyword arguments of field names and new values.
                      Valid fields: title, start_time, end_time, location, description.

        Returns:
            Updated event dict, or None if event not found.
        """
        allowed_fields = {"title", "start_time", "end_time", "location", "description", "is_all_day"}
        updates = {k: v for k, v in fields.items() if k in allowed_fields}

        if not updates:
            return self._get_event_by_id(event_id)

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values())
        values.append(event_id)

        self.conn.execute(
            f"UPDATE appointments SET {set_clause}, updated_at = datetime('now') WHERE id = ?",
            values,
        )
        self.conn.commit()

        logger.info(f"Updated event {event_id}: {list(updates.keys())}")
        return self._get_event_by_id(event_id)

    def delete_event(self, event_id: int) -> bool:
        """
        Delete an appointment by ID.

        Args:
            event_id: The ID of the event to delete.

        Returns:
            True if the event was deleted, False if not found.
        """
        cursor = self.conn.execute(
            "DELETE FROM appointments WHERE id = ?", (event_id,)
        )
        self.conn.commit()
        deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted event {event_id}")
        else:
            logger.warning(f"Event {event_id} not found for deletion")

        return deleted

    def check_conflicts(
        self,
        start_time: str,
        end_time: str,
        exclude_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Check for appointment time conflicts.

        Args:
            start_time: ISO format start datetime.
            end_time: ISO format end datetime.
            exclude_id: Optional event ID to exclude (for updates).

        Returns:
            List of conflicting event dicts.
        """
        query = """
            SELECT * FROM appointments
            WHERE start_time < ? AND end_time > ?
              AND is_all_day = 0
        """
        params: list[Any] = [end_time, start_time]

        if exclude_id is not None:
            query += " AND id != ?"
            params.append(exclude_id)

        cursor = self.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def find_events_by_title(self, title: str) -> list[dict[str, Any]]:
        """
        Find events by title (case-insensitive partial match).

        Args:
            title: Title text to search for.

        Returns:
            List of matching event dicts.
        """
        cursor = self.conn.execute(
            "SELECT * FROM appointments WHERE LOWER(title) LIKE ? ORDER BY start_time ASC",
            (f"%{title.lower()}%",),
        )
        return [dict(row) for row in cursor.fetchall()]

    def _get_event_by_id(self, event_id: int) -> dict[str, Any]:
        """Get a single event by ID."""
        row = self.conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (event_id,)
        ).fetchone()
        return dict(row) if row else {}
