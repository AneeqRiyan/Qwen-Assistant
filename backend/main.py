"""
FastAPI web server for the Voice Personal Assistant.

Exposes REST and WebSocket endpoints for chat, history,
session management, and health checks.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import get_config
from app.core.orchestrator import Orchestrator
from app.logger import get_logger, setup_logging

logger = get_logger("api")

# ── Module-level orchestrator (initialized on startup) ──
orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global orchestrator

    # Startup
    setup_logging()
    logger.info("Starting Voice Personal Assistant API server...")

    orchestrator = Orchestrator()
    health = await orchestrator.health_check()
    logger.info(f"System health: {health}")

    # Pre-warm LLM model so first request has zero cold-start delay
    if health.get("ollama") == "connected":
        try:
            logger.info("Pre-warming LLM model into CPU memory...")
            await orchestrator.llm.chat([{"role": "user", "content": "hello"}])
            logger.info("LLM model pre-warmed and ready")
        except Exception as e:
            logger.warning(f"LLM pre-warming skipped: {e}")

    yield

    # Shutdown
    logger.info("Shutting down API server...")
    if orchestrator and orchestrator.conn:
        orchestrator.conn.close()


# ── FastAPI App ──
app = FastAPI(
    title="Voice Personal Assistant API",
    description="Privacy-first local voice assistant with weather and calendar tools.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ──

class ChatRequest(BaseModel):
    """Text chat request."""
    message: str = Field(..., description="The user's text message")
    session_id: int | None = Field(None, description="Optional session ID (uses current if omitted)")


class ChatResponse(BaseModel):
    """Chat response."""
    text: str = Field(..., description="Assistant's text response")
    audio_base64: str | None = Field(None, description="Base64-encoded WAV audio (if TTS succeeded)")
    session_id: int = Field(..., description="Session ID")
    request_id: str = Field(..., description="Unique request identifier")
    transcription: str | None = Field(None, description="STT transcription (if audio input)")


class HistoryResponse(BaseModel):
    """Paginated conversation history response."""
    session_id: int
    turns: list[dict[str, Any]]
    total_count: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    """System health check response."""
    status: str
    ollama: str
    database: str
    session_id: int
    model: str


# ── API Endpoints ──

@app.post("/v1/chat", response_model=ChatResponse)
async def chat_text(request: ChatRequest):
    """
    Send a text message to the assistant.

    Returns the assistant's text response and optional audio.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    result = await orchestrator.process_text(request.message, session_id=request.session_id)

    audio_b64 = None
    if result.get("audio"):
        audio_b64 = base64.b64encode(result["audio"]).decode("utf-8")

    return ChatResponse(
        text=result["text"],
        audio_base64=audio_b64,
        session_id=result["session_id"],
        request_id=result["request_id"],
        transcription=None,
    )


@app.post("/v1/chat/audio", response_model=ChatResponse)
async def chat_audio(audio: UploadFile = File(...), session_id: int | None = Form(None)):
    """
    Send a WAV audio file to the assistant.

    The audio is transcribed via STT, processed, and the response
    includes both text and synthesized audio.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    audio_bytes = await audio.read()

    # 10 MB audio payload size check
    if len(audio_bytes) > 10 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": "Audio file too large (max 10MB)"},
        )

    result = await orchestrator.process_audio(audio_bytes, session_id=session_id)

    audio_b64 = None
    if result.get("audio"):
        audio_b64 = base64.b64encode(result["audio"]).decode("utf-8")

    return ChatResponse(
        text=result["text"],
        audio_base64=audio_b64,
        session_id=result["session_id"],
        request_id=result["request_id"],
        transcription=result.get("transcription"),
    )


@app.get("/v1/sessions")
async def get_sessions(limit: int = 50):
    """
    Get all conversation sessions for the sidebar.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    sessions = orchestrator.repo.get_all_sessions(limit=limit)
    return {
        "active_session_id": orchestrator.session_id,
        "sessions": sessions,
    }


@app.get("/v1/history", response_model=HistoryResponse)
async def get_history(
    session_id: int | None = None,
    limit: int = 20,
    offset: int = 0,
):
    """
    Get paginated conversation history.

    Supports infinite scrolling for Phase 2 UI.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    sid = session_id or orchestrator.session_id
    turns = orchestrator.repo.get_history_paginated(sid, limit=limit, offset=offset)
    total = orchestrator.repo.get_turn_count(sid)

    return HistoryResponse(
        session_id=sid,
        turns=turns,
        total_count=total,
        limit=limit,
        offset=offset,
    )


@app.post("/v1/session/new")
async def new_session():
    """Start a new conversation session."""
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    new_id = orchestrator.start_new_session()
    return {"session_id": new_id, "message": "New session started"}


@app.get("/v1/weather/current")
async def get_current_weather(city: str = "Marburg"):
    """
    Get current weather for the display-only weather widget.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    try:
        weather_data = await orchestrator.weather.get_current(city)
        return weather_data
    except Exception as e:
        logger.error(f"Failed to fetch weather widget data: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/calendar/events")
async def get_calendar_events(limit: int = 10):
    """
    Get upcoming calendar events for the display-only calendar widget.
    """
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    events = orchestrator.calendar.get_upcoming(limit=limit)
    return {"events": events}


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """Check system health: database, Ollama, and model status."""
    if orchestrator is None:
        return JSONResponse(status_code=503, content={"error": "Server not ready"})

    health = await orchestrator.health_check()
    return HealthResponse(**health)


# ── WebSocket Endpoint (Phase 2 streaming) ──

@app.websocket("/v1/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat streaming.

    Accepts text messages and returns assistant responses.
    Designed for Phase 2 UI low-latency interaction.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_text()

            if orchestrator is None:
                await websocket.send_json({"error": "Server not ready"})
                continue

            result = await orchestrator.process_text(data)

            response = {
                "text": result["text"],
                "session_id": result["session_id"],
                "request_id": result["request_id"],
            }

            # Include audio as base64 if available
            if result.get("audio"):
                response["audio_base64"] = base64.b64encode(result["audio"]).decode("utf-8")

            await websocket.send_json(response)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


# ── Entry point ──

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
