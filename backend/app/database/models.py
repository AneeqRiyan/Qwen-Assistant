"""
SQLite database schema and connection management.

Defines tables for sessions, conversation turns, and appointments.
Uses Python's built-in sqlite3 module — zero external ORM dependency.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.logger import get_logger

logger = get_logger("database")

# Database file lives in the backend/ directory
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
_DB_PATH = _BACKEND_DIR / "data" / "assistant.db"


# ── Schema Definitions ──

SCHEMA_SQL = """
-- Sessions table: tracks conversation sessions
CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    last_active_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Conversation turns: full history of user/assistant exchanges
CREATE TABLE IF NOT EXISTS conversation_turns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL,
    turn_number     INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content         TEXT    NOT NULL,
    tool_calls_json TEXT,
    timestamp       TEXT    NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Appointments: local calendar storage
CREATE TABLE IF NOT EXISTS appointments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    start_time      TEXT,
    end_time        TEXT,
    location        TEXT    DEFAULT '',
    description     TEXT    DEFAULT '',
    is_all_day      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_turns_session_id ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON conversation_turns(timestamp);
CREATE INDEX IF NOT EXISTS idx_appointments_start ON appointments(start_time);
"""


def get_db_path() -> Path:
    """Get the database file path, creating parent directories if needed."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _DB_PATH


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """
    Create a new SQLite connection with recommended settings.

    Args:
        db_path: Optional override for database file path.
                 Use ':memory:' for in-memory databases (testing).

    Returns:
        Configured sqlite3.Connection with row_factory set to Row.
    """
    path = str(db_path) if db_path else str(get_db_path())

    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrent read performance
    conn.execute("PRAGMA foreign_keys=ON")    # Enforce FK constraints

    return conn


def initialize_database(db_path: Path | str | None = None) -> sqlite3.Connection:
    """
    Initialize the database: create tables if they don't exist.

    Args:
        db_path: Optional override for database path.

    Returns:
        An open connection to the initialized database.
    """
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()

    logger.info(f"Database initialized at {db_path or get_db_path()}")
    return conn
