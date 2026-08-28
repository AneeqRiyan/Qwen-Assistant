"""
Shared pytest fixtures for the Voice Personal Assistant test suite.

Provides in-memory SQLite database, test config, and mock helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure app modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force test config before any app imports
os.environ["LOG_LEVEL"] = "DEBUG"

from app.config import load_config, reset_config
from app.database.models import initialize_database
from app.database.repository import Repository


@pytest.fixture(autouse=True)
def reset_app_config():
    """Reset config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def db_connection():
    """
    Create an in-memory SQLite database for testing.

    Yields a fully initialized connection that is closed after the test.
    """
    conn = initialize_database(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def repository(db_connection):
    """Create a Repository instance backed by the in-memory database."""
    return Repository(db_connection)


@pytest.fixture
def sample_session(repository):
    """Create a sample session and return its ID."""
    session = repository.create_new_session()
    return session["id"]


@pytest.fixture
def sample_events(repository):
    """Create sample calendar events for testing."""
    events = [
        repository.create_event(
            title="Team Meeting",
            start_time="2026-08-15T10:00:00",
            end_time="2026-08-15T11:00:00",
            location="Room 204",
        ),
        repository.create_event(
            title="Dentist",
            start_time="2026-08-15T14:00:00",
            end_time="2026-08-15T15:00:00",
            location="Main Street Clinic",
        ),
        repository.create_event(
            title="Lunch with Alex",
            start_time="2026-08-16T12:00:00",
            end_time="2026-08-16T13:00:00",
            location="Café Milano",
        ),
    ]
    return events
