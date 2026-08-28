"""
Integration tests for the FastAPI API endpoints.

Uses FastAPI TestClient to test /v1/chat, /v1/history,
/v1/health, and /v1/session/new endpoints.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator for API testing."""
    mock = MagicMock()
    mock.session_id = 1
    mock.conn = MagicMock()
    mock.repo = MagicMock()

    # Mock process_text
    async def mock_process_text(text):
        return {
            "text": f"Mock response to: {text}",
            "audio": None,
            "session_id": 1,
            "request_id": "test123",
            "metrics": {"total_ms": 100},
        }

    mock.process_text = mock_process_text

    # Mock health_check
    async def mock_health():
        return {
            "status": "healthy",
            "ollama": "connected",
            "database": "connected",
            "session_id": 1,
            "model": "qwen2.5:3b",
        }

    mock.health_check = mock_health

    # Mock start_new_session
    mock.start_new_session.return_value = 2

    # Mock repo methods
    mock.repo.get_history_paginated.return_value = [
        {"role": "user", "content": "Hello", "turn_number": 1, "timestamp": "2026-08-12T10:00:00"},
        {"role": "assistant", "content": "Hi!", "turn_number": 2, "timestamp": "2026-08-12T10:00:01"},
    ]
    mock.repo.get_turn_count.return_value = 2
    mock.repo.get_all_sessions.return_value = [
        {
            "id": 1,
            "created_at": "2026-08-14T10:00:00",
            "last_active_at": "2026-08-14T10:15:00",
            "turn_count": 2,
            "preview": "Hello, how are you?",
        }
    ]

    # Mock weather provider
    mock_weather = MagicMock()
    async def mock_get_current(city):
        return {
            "city": city,
            "country": "Germany",
            "temperature": 22.5,
            "unit": "°C",
            "condition": "partly cloudy",
            "humidity": 60,
            "wind_speed": 12.0,
            "wind_unit": "km/h",
            "precipitation": 0.0,
        }
    mock_weather.get_current = mock_get_current
    mock.weather = mock_weather

    # Mock calendar provider
    mock_calendar = MagicMock()
    mock_calendar.get_upcoming.return_value = [
        {
            "id": 1,
            "title": "Team Sync",
            "start_time": "2026-08-14T10:00:00",
            "end_time": "2026-08-14T11:00:00",
            "location": "Room 204",
        }
    ]
    mock.calendar = mock_calendar

    return mock


@pytest.fixture
def client(mock_orchestrator):
    """Create a FastAPI test client with mocked orchestrator."""
    import main
    main.orchestrator = mock_orchestrator
    return TestClient(main.app)


class TestChatEndpoint:
    """Tests for POST /v1/chat."""

    def test_chat_text_success(self, client):
        """Test successful text chat."""
        response = client.post(
            "/v1/chat",
            json={"message": "Hello, how are you?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "Mock response to: Hello, how are you?" in data["text"]
        assert data["session_id"] == 1
        assert data["request_id"] == "test123"

    def test_chat_empty_message(self, client):
        """Test chat with empty message."""
        response = client.post(
            "/v1/chat",
            json={"message": ""},
        )
        # FastAPI should still process it (the LLM will handle empty input)
        assert response.status_code == 200


class TestSessionsListEndpoint:
    """Tests for GET /v1/sessions."""

    def test_get_sessions(self, client):
        """Test retrieving all sessions list."""
        response = client.get("/v1/sessions?limit=50")

        assert response.status_code == 200
        data = response.json()
        assert data["active_session_id"] == 1
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["preview"] == "Hello, how are you?"


class TestHistoryEndpoint:
    """Tests for GET /v1/history."""

    def test_get_history(self, client):
        """Test retrieving conversation history."""
        response = client.get("/v1/history?limit=20&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == 1
        assert len(data["turns"]) == 2
        assert data["total_count"] == 2
        assert data["limit"] == 20
        assert data["offset"] == 0

    def test_get_history_pagination(self, client):
        """Test history pagination parameters."""
        response = client.get("/v1/history?limit=1&offset=1")

        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 1
        assert data["offset"] == 1


class TestSessionEndpoint:
    """Tests for POST /v1/session/new."""

    def test_new_session(self, client):
        """Test creating a new session."""
        response = client.post("/v1/session/new")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == 2
        assert "New session" in data["message"]


class TestWeatherWidgetEndpoint:
    """Tests for GET /v1/weather/current."""

    def test_get_current_weather(self, client):
        """Test weather widget endpoint."""
        response = client.get("/v1/weather/current?city=Marburg")

        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Marburg"
        assert data["temperature"] == 22.5
        assert data["unit"] == "°C"


class TestCalendarWidgetEndpoint:
    """Tests for GET /v1/calendar/events."""

    def test_get_calendar_events(self, client):
        """Test calendar widget endpoint."""
        response = client.get("/v1/calendar/events?limit=10")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) == 1
        assert data["events"][0]["title"] == "Team Sync"


class TestHealthEndpoint:
    """Tests for GET /v1/health."""

    def test_health_check(self, client):
        """Test health check returns system status."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["ollama"] == "connected"
        assert data["database"] == "connected"
        assert data["model"] == "qwen2.5:3b"
